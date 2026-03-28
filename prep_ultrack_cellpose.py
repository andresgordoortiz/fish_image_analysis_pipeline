"""
Prepare ultrack foreground + contours using boundary-gated approach.

Replaces the original prep_ultrack_lazyload.py:
  - foreground = detect_foreground(raw) WITH cellpose boundaries zeroed out
                 → physical gaps between cells that watershed cannot cross
  - contours   = robust_invert(raw)  (unchanged, full raw diversity)

How it works:
  1. detect_foreground(raw) gives the same foreground as original pipeline
  2. find_boundaries(cellpose_labels) gives 1-pixel-wide walls between cells
  3. foreground[boundaries] = 0  → cuts the foreground at cellpose walls
  4. contours = robust_invert(raw) → raw image intensity (rich hierarchy)

  Result: ultrack's watershed runs on the *original raw image* (good
  hypothesis diversity), but physically cannot merge cells across the
  boundary gaps. Each connected foreground region = one cellpose cell.

Usage:
  # Full processing (all timepoints → zarr)
  python prep_ultrack_hybrid.py \
      --raw combined_hyperstack.tif \
      --labels cellpose_labels.tif \
      --output ./ultrack_input

  # Preview selected timepoints as full Z-stack TIFFs (open in Fiji)
  python prep_ultrack_hybrid.py \
      --raw combined_hyperstack.tif \
      --labels cellpose_labels.tif \
      --preview --preview-t 0 50 100 200

  # Also export labels_to_contours method for comparison
  python prep_ultrack_hybrid.py \
      --raw combined_hyperstack.tif \
      --labels cellpose_labels.tif \
      --preview --preview-t 100 --compare --preview-output ./my_preview
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _get_available_ram_bytes():
    """Return available RAM in bytes."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass
    # fallback: parse /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return None


def _open_hyperstack(path):
    """
    Open an ImageJ hyperstack TIFF and return (array, tif_handle) with shape (T, Z, Y, X).

    The merge script writes each Z-slice as an individual 2D IFD with ImageJ
    metadata (frames=T, slices=Z).  tifffile.asarray(out='memmap') returns a
    flat (T*Z, Y, X) array; we reshape it to (T, Z, Y, X) using the ImageJ
    metadata.  Falls back to (1, Z, Y, X) for plain 3D stacks.
    """
    import tifffile

    tif = tifffile.TiffFile(path)
    data = tif.asarray(out="memmap")

    if data.ndim == 4:
        return data, tif

    if data.ndim == 3:
        n_images, Y, X = data.shape
        # Try to get T and Z from ImageJ metadata
        T, Z = 1, n_images
        if tif.is_imagej:
            ij = tif.imagej_metadata or {}
            Z = int(ij.get("slices", n_images))
            T = int(ij.get("frames", 1))
        if T * Z != n_images:
            # Metadata inconsistent; assume single timepoint
            T, Z = 1, n_images
        data = data.reshape(T, Z, Y, X)
        print(
            f"  Reshaped from (T*Z={n_images}, {Y}, {X}) to ({T}, {Z}, {Y}, {X}) using ImageJ metadata"
        )
        return data, tif

    if data.ndim == 2:
        # Single 2D image
        data = data[np.newaxis, np.newaxis, ...]
        return data, tif

    raise ValueError(f"Unexpected ndim={data.ndim} for {path}")


def _estimate_frame_memory(Z, Y, X, raw_dtype, lbl_dtype):
    """
    Estimate peak RAM per worker for one timepoint.
    Arrays: raw_frame, lbl_frame, fg, bounds, cc_labels, ct + overhead.
    """
    n_voxels = Z * Y * X
    raw_bytes = n_voxels * np.dtype(raw_dtype).itemsize
    lbl_bytes = n_voxels * np.dtype(lbl_dtype).itemsize
    fg_bytes = n_voxels * 4  # float32
    bounds_bytes = n_voxels * 1  # bool
    cc_bytes = n_voxels * 8  # int64 from ndi_label
    ct_bytes = n_voxels * 4  # float32
    overhead_factor = 1.5  # detect_foreground/robust_invert temporaries
    return int(
        (raw_bytes + lbl_bytes + fg_bytes + bounds_bytes + cc_bytes + ct_bytes)
        * overhead_factor
    )


def process_all(args):
    """Process timepoints → zarr, parallelised across frames."""
    import zarr
    import tifffile
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    try:
        import cupy as cp

        HAS_CUPY = True
        # Print GPU info
        dev = cp.cuda.Device()
        gpu_mem_total = dev.mem_info[1]
        gpu_mem_free = dev.mem_info[0]
        print(
            f"  GPU: {dev.id} — {gpu_mem_free / 1e9:.1f} / {gpu_mem_total / 1e9:.1f} GB free"
        )
    except ImportError:
        HAS_CUPY = False

    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation, label as ndi_label
    from ultrack.imgproc import robust_invert, detect_foreground

    print("Opening raw image (lazy loading)...")
    raw, raw_tif = _open_hyperstack(args.raw)
    T, Z, Y, X = raw.shape
    print(f"  Raw shape: {raw.shape}, dtype: {raw.dtype}")

    print("Opening cellpose labels (lazy loading)...")
    labels, lbl_tif = _open_hyperstack(args.labels)
    assert labels.shape == raw.shape, (
        f"Shape mismatch: raw={raw.shape} labels={labels.shape}"
    )
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")

    os.makedirs(args.output, exist_ok=True)

    fg_zarr = zarr.open(
        os.path.join(args.output, "foreground.zarr"),
        mode="w",
        shape=(T, Z, Y, X),
        dtype=np.float32,
        chunks=(1, Z, Y, X),
    )
    ct_zarr = zarr.open(
        os.path.join(args.output, "contours.zarr"),
        mode="w",
        shape=(T, Z, Y, X),
        dtype=np.float32,
        chunks=(1, Z, Y, X),
    )

    t_start = args.start_t if args.start_t is not None else 0
    t_end = args.end_t if args.end_t is not None else T
    t_start = max(0, min(t_start, T))
    t_end = max(0, min(t_end, T))
    n_frames = t_end - t_start

    # --- determine number of workers ---
    frame_mem = _estimate_frame_memory(Z, Y, X, raw.dtype, labels.dtype)
    avail_ram = _get_available_ram_bytes()
    # respect SLURM allocation if available
    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1

    if avail_ram is not None:
        # reserve 20% of RAM for OS / zarr / memmap overhead
        usable_ram = int(avail_ram * 0.8)
        max_by_ram = max(1, usable_ram // frame_mem)
    else:
        max_by_ram = cpu_count
        print("  WARNING: could not detect available RAM, using CPU count as limit")

    n_workers = min(max_by_ram, cpu_count, n_frames)
    # With a single GPU, the GPU calls are serialized behind gpu_lock.
    # Extra threads only help overlap CPU work (boundaries, cc-labeling,
    # zarr writes) with GPU work. Cap to cpu_count (SLURM-aware).
    if HAS_CUPY and args.workers is None:
        n_workers = min(n_workers, cpu_count)
    if args.workers is not None:
        n_workers = min(args.workers, n_frames)

    # if cupy is present, GPU ops are not thread-safe → serialize GPU calls
    gpu_lock = threading.Lock() if HAS_CUPY else None

    print(f"  Per-frame memory estimate: {frame_mem / 1e9:.2f} GB")
    if avail_ram is not None:
        print(
            f"  Available RAM: {avail_ram / 1e9:.1f} GB (using 80% = {avail_ram * 0.8 / 1e9:.1f} GB)"
        )
    print(f"  Workers: {n_workers} (cpu={cpu_count}, ram-limited={max_by_ram})")
    print(f"Processing timepoints {t_start}..{t_end - 1} ({n_frames} frames)")

    def _process_frame(t):
        raw_frame = np.array(raw[t])
        lbl_frame = np.array(labels[t])

        # --- foreground: detect from raw, then cut at cellpose boundaries ---
        if gpu_lock is not None:
            with gpu_lock:
                fg = np.array(
                    detect_foreground(raw_frame, sigma=args.fg_sigma), dtype=np.float32
                )
                cp.get_default_memory_pool().free_all_blocks()
        else:
            fg = np.array(
                detect_foreground(raw_frame, sigma=args.fg_sigma), dtype=np.float32
            )

        # find cellpose boundary pixels (1px wide between labels)
        bounds = find_boundaries(lbl_frame, mode="outer")
        if args.boundary_width > 1:
            bounds = binary_dilation(bounds, iterations=args.boundary_width - 1)
        fg[bounds] = 0.0

        # remove foreground islands smaller than min_area
        fg_binary = fg > 0
        cc_labels_arr, n_cc = ndi_label(fg_binary)
        n_removed = 0
        if n_cc > 0:
            cc_sizes = np.bincount(cc_labels_arr.ravel())
            small_mask = cc_sizes < args.min_area
            small_mask[0] = False
            fg[small_mask[cc_labels_arr]] = 0.0
            n_removed = int(np.sum(small_mask[1:]))

        # --- contours: robust_invert of raw (unchanged, full diversity) ---
        if gpu_lock is not None:
            with gpu_lock:
                ct = np.array(
                    robust_invert(raw_frame, sigma=args.raw_sigma), dtype=np.float32
                )
                cp.get_default_memory_pool().free_all_blocks()
        else:
            ct = np.array(
                robust_invert(raw_frame, sigma=args.raw_sigma), dtype=np.float32
            )

        fg_zarr[t] = fg
        ct_zarr[t] = ct

        return t, n_removed

    if n_workers <= 1:
        # sequential fallback
        for t in range(t_start, t_end):
            t_done, n_removed = _process_frame(t)
            msg = f"  t={t_done}/{t_end - 1}"
            if n_removed > 0:
                msg += f"  (removed {n_removed} islands < {args.min_area}v)"
            print(msg)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_frame, t): t for t in range(t_start, t_end)}
            for future in as_completed(futures):
                t_done, n_removed = future.result()
                completed += 1
                msg = f"  [{completed}/{n_frames}] t={t_done}"
                if n_removed > 0:
                    msg += f"  (removed {n_removed} islands < {args.min_area}v)"
                print(msg)

    raw_tif.close()
    lbl_tif.close()
    print(f"Done. Output in {args.output}/")


def preview(args):
    """
    Save full Z-stack TIFFs for selected timepoints so you can inspect
    them in Fiji/ImageJ. Optionally --compare to also export the
    labels_to_contours method for side-by-side comparison.

    Output files per timepoint (e.g. t=50):
      preview_t050_raw.tif              — raw Z-stack
      preview_t050_labels.tif           — cellpose labels
      preview_t050_boundaries.tif       — boundary mask (uint8)
      preview_t050_fg_original.tif      — detect_foreground(raw)
      preview_t050_fg_gated.tif         — gated foreground (boundaries zeroed)
      preview_t050_contours.tif         — robust_invert(raw) contours
    With --compare, also:
      preview_t050_fg_l2c.tif           — labels_to_contours foreground
      preview_t050_ct_l2c.tif           — labels_to_contours contours
    """
    import tifffile
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation, label as ndi_label
    from ultrack.imgproc import robust_invert, detect_foreground

    out_dir = args.preview_output
    os.makedirs(out_dir, exist_ok=True)

    print("Opening raw image (lazy loading)...")
    raw_all, raw_tif = _open_hyperstack(args.raw)
    T = raw_all.shape[0]
    print(f"  Raw shape: {raw_all.shape}, dtype: {raw_all.dtype}")

    print("Opening cellpose labels (lazy loading)...")
    lbl_all, lbl_tif = _open_hyperstack(args.labels)
    print(f"  Labels shape: {lbl_all.shape}, dtype: {lbl_all.dtype}")

    timepoints = args.preview_t
    for tp in timepoints:
        if tp < 0 or tp >= T:
            print(f"WARNING: t={tp} out of range [0, {T - 1}], skipping.")
            continue

        prefix = os.path.join(out_dir, f"preview_t{tp:03d}")
        print(f"\nProcessing t={tp} ...")

        raw_frame = np.array(raw_all[tp])
        lbl_frame = np.array(lbl_all[tp])

        # foreground & contours from raw
        fg_raw = np.array(
            detect_foreground(raw_frame, sigma=args.fg_sigma), dtype=np.float32
        )
        contours = np.array(
            robust_invert(raw_frame, sigma=args.raw_sigma), dtype=np.float32
        )

        # cellpose boundaries
        bounds = find_boundaries(lbl_frame, mode="outer")
        if args.boundary_width > 1:
            bounds = binary_dilation(bounds, iterations=args.boundary_width - 1)

        # gated foreground
        fg_gated = fg_raw.copy()
        fg_gated[bounds] = 0.0

        # remove tiny foreground islands
        fg_binary = fg_gated > 0
        cc_labels, n_cc = ndi_label(fg_binary)
        if n_cc > 0:
            cc_sizes = np.bincount(cc_labels.ravel())
            small_mask = cc_sizes < args.min_area
            small_mask[0] = False
            fg_gated[small_mask[cc_labels]] = 0.0
            n_removed = np.sum(small_mask[1:])
            if n_removed > 0:
                print(
                    f"    Removed {n_removed} foreground islands < {args.min_area} voxels"
                )

        # --- save full Z-stacks as TIFFs ---
        tifffile.imwrite(f"{prefix}_raw.tif", raw_frame, imagej=True)
        print(f"  {prefix}_raw.tif")

        tifffile.imwrite(f"{prefix}_labels.tif", lbl_frame, imagej=True)
        print(f"  {prefix}_labels.tif")

        tifffile.imwrite(
            f"{prefix}_boundaries.tif", bounds.astype(np.uint8) * 255, imagej=True
        )
        print(f"  {prefix}_boundaries.tif")

        tifffile.imwrite(f"{prefix}_fg_original.tif", fg_raw, imagej=True)
        print(f"  {prefix}_fg_original.tif")

        tifffile.imwrite(f"{prefix}_fg_gated.tif", fg_gated, imagej=True)
        print(f"  {prefix}_fg_gated.tif")

        tifffile.imwrite(f"{prefix}_contours.tif", contours, imagej=True)
        print(f"  {prefix}_contours.tif")

        if args.compare:
            # Replicate labels_to_contours logic manually to avoid cupy
            # incompatibility (ultrack converts to cupy internally, but
            # skimage.find_boundaries does not support cupy arrays).
            from scipy.ndimage import gaussian_filter

            lbl_np = np.asarray(lbl_frame)
            fg_l2c = (lbl_np > 0).astype(np.float32)
            ct_l2c = find_boundaries(lbl_np, mode="outer").astype(np.float32)
            ct_l2c = gaussian_filter(ct_l2c, sigma=2.0)
            tifffile.imwrite(f"{prefix}_fg_l2c.tif", fg_l2c, imagej=True)
            print(f"  {prefix}_fg_l2c.tif")
            tifffile.imwrite(f"{prefix}_ct_l2c.tif", ct_l2c, imagej=True)
            print(f"  {prefix}_ct_l2c.tif")

    raw_tif.close()
    lbl_tif.close()
    print(f"\nDone. Preview TIFFs saved in {out_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Prepare ultrack foreground+contours with boundary-gated approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--raw", required=True, help="Path to raw 4D hyperstack TIFF (T,Z,Y,X)"
    )
    ap.add_argument(
        "--labels",
        required=True,
        help="Path to cellpose/SAM label stack TIFF (T,Z,Y,X)",
    )
    ap.add_argument(
        "--output",
        default="./ultrack_input",
        help="Output directory for zarr files (default: ./ultrack_input)",
    )

    # Processing params
    ap.add_argument(
        "--fg-sigma",
        type=float,
        default=5.0,
        help="Gaussian sigma for detect_foreground (default: 5.0, same as original)",
    )
    ap.add_argument(
        "--raw-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for robust_invert contours (default: 1.0, same as original)",
    )
    ap.add_argument(
        "--boundary-width",
        type=int,
        default=1,
        help="Width of the boundary gap in pixels (default: 1). "
        "Increase to 2-3 if watershed still leaks across cells.",
    )
    ap.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Remove foreground islands smaller than this (voxels). "
        "Must be ≥8 for ultrack's watershed hierarchy (default: 10).",
    )
    ap.add_argument(
        "--start-t",
        type=int,
        default=None,
        help="First timepoint to process (inclusive, default: 0)",
    )
    ap.add_argument(
        "--end-t",
        type=int,
        default=None,
        help="Last timepoint to process (exclusive, default: all)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: auto from available RAM & CPUs)",
    )

    # Preview mode
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview selected timepoints as full Z-stack TIFFs",
    )
    ap.add_argument(
        "--preview-t",
        type=int,
        nargs="+",
        default=[0],
        help="Timepoint(s) to preview, e.g. --preview-t 0 50 100 (default: 0)",
    )
    ap.add_argument(
        "--preview-output",
        default="./preview",
        help="Directory to save preview TIFFs (default: ./preview)",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Also export labels_to_contours TIFFs for comparison",
    )

    args = ap.parse_args()

    if args.preview:
        preview(args)
    else:
        process_all(args)


if __name__ == "__main__":
    main()
