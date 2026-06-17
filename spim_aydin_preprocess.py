#!/usr/bin/env python3
"""
SPIM Image Preprocessing Pipeline — Aydin self-supervised denoising
IMP Vienna - Andrés Gordo & Guilherme Ventura

Aydin (https://github.com/royerlab/aydin) is plugged in here as an OPTIONAL
preprocessing step that runs **before** the deconvolution / Self-Net /
segmentation stages. Aydin is self-supervised — no clean ground truth is
needed — and supports a wide range of classical (NLM, BMnD, bilateral, …) and
self-supervised ML (Noise2Self-FGR, Noise2Self-CNN) denoisers.

The script intentionally does NOT do the full deconvolution pipeline: it only
denoises, preserving the input's dtype, shape and ImageJ metadata so the
downstream Nextflow stages (PREPROCESS_DECONVOLVE / PREPROCESS_SELFNET /
CELLPOSE_SEGMENT) consume the result transparently.

Why aydin over Noise2Self (czbiohub-sf) for this pipeline
--------------------------------------------------------
* Aydin is from the same lab as the Noise2Self paper but is the productionised
  implementation: n-dimensional arrays, 3D/volumetric first-class, CPU and
  GPU paths, a curated menu of denoisers, and an auto-tune that removes the
  hand-tuning that the czbiohub-sf Noise2Self repo forces you to do by hand.
* Aydin ships classical algorithms (NLM, BMnD, bilateral, Gaussian-median,
  Butterworth, …) that don't hallucinate. For medaka (smaller, denser nuclei
  than zebrafish) the *classical* NLM/BMnD paths are safer than the CNN
  variant — the aydin README itself warns that Noise2Self-CNN "is typically
  slower to train, and more prone to hallucination and residual noise than
  FGR" (see https://github.com/royerlab/aydin).
* The czbiohub-sf Noise2Self repo is a research artifact (98% notebooks,
  no 3D support, Masker + your-pipeline-only). It would take weeks of
  engineering to bolt it onto this Nextflow pipeline; aydin drops in.

Recommended algorithms (for medaka lightsheet)
----------------------------------------------
* `nlm`           — non-local means, 3D, edge-preserving, no hallucination.
                     **Default**. Good for cloudy embryos, preserves dim Z nuclei.
* `bmnd`          — n-D generalisation of BM3D. Sharper than NLM but ~2-3×
                     slower. Use it after NLM if you want crisper nuclei.
* `bilateral`     — edge-preserving blur. Cheaper than NLM, less sharp.
* `gaussianmedian`— mixed Gaussian / median. Good baseline.
* `noise2self-fgr`— ML (CatBoost/lightGBM regressor) if classical under-denoises.
                     CPU only, no hallucination problem (per aydin README).
* `noise2self-cnn`— ML PyTorch CNN. **Avoid** for medaka — explicitly flagged
                     as hallucination-prone in the aydin README. We expose it
                     for completeness, not as a recommendation.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import tifffile

# Shared helpers (staged alongside this script by the Nextflow process, same
# convention as spim_selfnet_preprocess.py).
from spim_pipeline_fixed import (
    print_resource_usage,
    read_nd2_voxel_size,
    read_tiff_voxel_size,
)


# Methods that the aydin ImageDenoiser API knows about. We list them here so
# we can fail fast with a clear message if the user misconfigures --method,
# rather than letting aydin raise a less helpful exception.
AYDIN_METHODS = {
    # Classical — recommended for medaka (no hallucination, edge-preserving)
    "nlm": "non-local means, 3D, edge-preserving (default for cloudy embryos)",
    "bmnd": "n-D generalisation of BM3D; sharper than NLM but slower",
    "bilateral": "edge-preserving bilateral filter; cheap, less sharp than NLM",
    "gaussianmedian": "mixed Gaussian / median baseline",
    "butterworth": "low-pass Butterworth; very cheap, smears fine detail",
    "lowpass": "alias for butterworth (aydin synonym)",
    "gaussian": "Gaussian low-pass; cheap, aggressive smoothing",
    "tv": "total-variation regularisation; tends to flatten texture",
    "wavelet": "wavelet-domain thresholding; good for Gaussian noise",
    "spectral": "Fourier-domain low-pass",
    "pca": "PCA low-rank; useful for low-rank signal + noise",
    "lipschitz": "Lipschitz-continuity regularised; conservative",
    "harmonic": "harmonic (mean-like) regularisation; rarely useful",
    # ML — use with care
    "noise2self-fgr": "Noise2Self with feature-generation/regression (CatBoost/lightGBM); CPU",
    "noise2self-cnn": "Noise2Self with PyTorch CNN; GPU; HALLUCINATION RISK for small nuclei",
}


def aydin_denoise_3d(
    img,
    method="nlm",
    tile_shape=None,
    max_memory_bytes=None,
    verbose=True,
):
    """Run aydin on a 3D ZYX stack and return the denoised array (float32).

    Parameters
    ----------
    img : np.ndarray
        3D (Z, Y, X) array. Any dtype; converted to float32 internally.
    method : str
        Aydin algorithm name (see AYDIN_METHODS).
    tile_shape : tuple or None
        Optional (z, y, x) tile shape for tile-based denoising. Aydin
        handles overlap and stitching automatically. If None, a tile shape
        is computed from ``max_memory_bytes`` when the full volume would
        exceed the budget, otherwise the full volume is denoised in one go.
    max_memory_bytes : int or None
        Optional RAM budget in bytes. Used to auto-pick a tile shape when
        ``tile_shape`` is None. ``None`` means "use as much as you need".
    verbose : bool
        If True, aydin prints its own progress; if False, aydin is silenced.

    Returns
    -------
    np.ndarray
        Denoised volume, same shape as ``img``, dtype float32.
    """
    try:
        from aydin import ImageDenoiser
    except ImportError as e:
        print(f"ERROR: aydin is not installed: {e}")
        print("Install with: pip install aydin")
        sys.exit(1)

    if method not in AYDIN_METHODS:
        print(
            f"ERROR: unknown aydin method '{method}'. "
            f"Choose one of: {', '.join(sorted(AYDIN_METHODS))}"
        )
        sys.exit(1)

    if img.ndim != 3:
        raise ValueError(
            f"aydin preprocess expects a 3D ZYX stack, got shape {img.shape}"
        )

    if tile_shape is None:
        tile_shape = _auto_tile_shape(img.shape, max_memory_bytes)

    if tile_shape is not None:
        print(f"  Aydin tile shape: {tile_shape}  (full shape: {img.shape})")
    else:
        print(f"  Aydin denoising full volume {img.shape} in one pass")

    # Aydin's auto-tune is sensitive to dtype: float32 is the safe universal
    # path. uint16 round-trips fine, but float32 avoids any silent truncation
    # while aydin explores denoiser parameters.
    img_f = img.astype(np.float32, copy=False)

    # Build the denoiser. ``max_memory_usage`` is aydin's own budget knob —
    # we still hand it our derived tile shape (or None) because aydin's
    # internal tiling is a different layer (per-channel patch vs ZYX slab).
    kwargs = {"method": method}
    if max_memory_bytes is not None and max_memory_bytes > 0:
        kwargs["max_memory_usage"] = int(max_memory_bytes)
    if tile_shape is not None:
        kwargs["tile_shape"] = tuple(int(t) for t in tile_shape)

    # ``ImageDenoiser`` lets us silence aydin's chatter. We always want
    # SOME progress for the HPC logs, but we cap it to a couple of lines.
    if not verbose:
        kwargs["verbosity"] = "ERROR"

    print(f"  Aydin args: {kwargs}")
    denoiser = ImageDenoiser(**kwargs)

    t0 = time.time()
    print("  Training aydin (self-supervised, auto-tune)…")
    # batch_axes=None means "treat the whole volume as one sample, all axes
    # are spatial". That matches the lightsheet use case (one ZYX stack per
    # timepoint, not a stack of independent 2D planes).
    denoiser.train(img_f, batch_axes=None)
    print(f"  Aydin training took {time.time() - t0:.2f} seconds")

    t0 = time.time()
    print("  Applying aydin denoiser…")
    denoised = denoiser.denoise(img_f)
    print(f"  Aydin denoising took {time.time() - t0:.2f} seconds")

    # Defensive: aydin is meant to return the same shape; assert it before
    # the downstream stage silently gets a misaligned volume.
    if denoised.shape != img.shape:
        raise RuntimeError(
            f"Aydin returned shape {denoised.shape}, expected {img.shape}. "
            "This is a bug in aydin or in the tile shape; please report."
        )

    return denoised.astype(np.float32, copy=False)


def _auto_tile_shape(volume_shape, max_memory_bytes):
    """Pick a ZYX tile shape that fits the memory budget.

    Aydin's internal RAM usage is ~8× the voxel count of the working slab
    (input + denoiser scratch + denoised output + integration buffers).
    We use that as a conservative rule of thumb. If the full volume already
    fits the budget, returns None (let aydin process it whole, which is
    usually better for continuity at the tile boundaries).
    """
    if max_memory_bytes is None or max_memory_bytes <= 0:
        return None

    BYTES_PER_VOXEL = 4  # float32
    AYDIN_OVERHEAD = 8   # input + scratch + output + integration
    bytes_per_voxel = BYTES_PER_VOXEL * AYDIN_OVERHEAD

    full_size_bytes = int(np.prod(volume_shape)) * bytes_per_voxel
    if full_size_bytes <= max_memory_bytes:
        return None

    # Scale all axes proportionally to fit the budget (cube root of the
    # volume fraction we can afford). 8 is a safe lower bound — too small
    # and aydin's patch search loses context.
    n_voxels_budget = max_memory_bytes // bytes_per_voxel
    fraction = n_voxels_budget / float(np.prod(volume_shape))
    scale = fraction ** (1.0 / 3.0)
    tile_shape = tuple(max(8, int(s * scale)) for s in volume_shape)
    # Round to multiples of 8 (good alignment for most aydin denoisers).
    tile_shape = tuple(((t + 7) // 8) * 8 for t in tile_shape)
    # Cap at full size (no oversize tiles).
    tile_shape = tuple(min(t, s) for t, s in zip(tile_shape, volume_shape))
    return tile_shape


def main():
    parser = argparse.ArgumentParser(
        description="Aydin self-supervised denoising for SPIM timepoints"
    )

    # Paths
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input TIFF"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory"
    )

    # Denoiser
    parser.add_argument(
        "--method",
        type=str,
        default="nlm",
        help=(
            "Aydin algorithm. See spim_aydin_preprocess.AYDIN_METHODS for the "
            "full list. Default: nlm (non-local means, recommended for medaka)."
        ),
    )
    parser.add_argument(
        "--tile_shape",
        type=str,
        default="",
        help=(
            "Optional (z,y,x) tile shape for tile-based denoising, e.g. "
            "'128,256,256'. If empty, a tile shape is auto-picked from "
            "--max_memory_mb (or the full volume is denoised at once when it "
            "fits)."
        ),
    )
    parser.add_argument(
        "--max_memory_mb",
        type=int,
        default=0,
        help=(
            "Optional RAM budget in MB. Aydin's full-volume denoising needs "
            "roughly 8× the input volume; if the input exceeds that, the "
            "script auto-picks a tile shape. 0 = no cap (let aydin decide)."
        ),
    )

    # Voxel sizes (used only for metadata, not for the denoiser itself)
    parser.add_argument(
        "--xy_pixel",
        type=float,
        default=0.0,
        help="Force XY pixel size (um). 0 = read from input metadata.",
    )
    parser.add_argument(
        "--z_pixel",
        type=float,
        default=0.0,
        help="Force Z pixel size (um). 0 = read from input metadata.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file does not exist: {args.input_file}")
        sys.exit(1)

    # Parse tile shape early so a typo fails fast
    if args.tile_shape.strip():
        try:
            parts = [int(x.strip()) for x in args.tile_shape.split(",")]
            assert len(parts) == 3
            tile_shape = tuple(parts)
        except Exception as e:
            print(
                f"ERROR: --tile_shape must be three comma-separated integers "
                f"like '128,256,256' (got '{args.tile_shape}': {e})"
            )
            sys.exit(1)
    else:
        tile_shape = None

    max_memory_bytes = (
        int(args.max_memory_mb) * 1024 * 1024 if args.max_memory_mb > 0 else None
    )

    print("\n" + "=" * 60)
    print("SPIM PREPROCESSING (Aydin self-supervised denoising)")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMethod:   {args.method}")
    if args.method in AYDIN_METHODS:
        print(f"          {AYDIN_METHODS[args.method]}")
    print(f"Tile:     {tile_shape if tile_shape else '(auto from memory budget)'}")
    print(f"Mem cap:  {args.max_memory_mb} MB")
    print("=" * 60 + "\n")

    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except FileExistsError:
            pass

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    image_path = args.input_file
    image_name = os.path.basename(image_path)
    print(f"[Processing] {image_name}")
    print_resource_usage()

    start_time_total = time.time()

    t0 = time.time()
    ext = os.path.splitext(image_name)[1].lower()
    try:
        if ext in [".tif", ".tiff"]:
            img = tifffile.imread(image_path).astype(np.uint16)
            voxel_size = read_tiff_voxel_size(image_path)
        elif ext == ".nd2":
            import pims  # only required for nd2 inputs
            img = pims.open(image_path)
            voxel_size = read_nd2_voxel_size(img)
            img = np.array(img, dtype=np.uint16, copy=False)
        else:
            print(f"ERROR: Unsupported format: {ext}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"[Timer] Image loading took {time.time() - t0:.2f} seconds")
    print(f"  - shape: {img.shape}, dtype: {img.dtype}")
    print(f"  - estimated size (GB): {img.nbytes / (1024**3):.3f}")

    if img.ndim != 3:
        print(f"ERROR: Aydin preprocess expects a 3D ZYX stack, got shape {img.shape}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Voxel sizes (override if user provided them)
    # ------------------------------------------------------------------
    physical_pixel_sizeX, physical_pixel_sizeY, physical_pixel_sizeZ = voxel_size
    if args.xy_pixel > 0:
        physical_pixel_sizeX = args.xy_pixel
        physical_pixel_sizeY = physical_pixel_sizeX
    if args.z_pixel > 0:
        physical_pixel_sizeZ = args.z_pixel
    print(f"  - voxel sizes (um): {physical_pixel_sizeX:.4f} x "
          f"{physical_pixel_sizeY:.4f} x {physical_pixel_sizeZ:.4f}")

    # Pull the original ImageJ metadata so we can put it back on the
    # denoised output. The downstream Nextflow stages (PREPROCESS_DECONVOLVE /
    # PREPROCESS_SELFNET / CELLPOSE_SEGMENT) all re-read this metadata and
    # patch in their own updates, so anything we drop here is recoverable.
    print("  Reading original ImageJ metadata…")
    with tifffile.TiffFile(image_path) as tif:
        original_imagej_meta = dict(tif.imagej_metadata) if tif.imagej_metadata else {}
    print(f"  Original ImageJ metadata keys: {sorted(original_imagej_meta.keys())}")

    # ------------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        denoised_f = aydin_denoise_3d(
            img,
            method=args.method,
            tile_shape=tile_shape,
            max_memory_bytes=max_memory_bytes,
            verbose=True,
        )
    except Exception as e:
        print(f"ERROR: aydin denoising failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print(f"[Timer] Aydin total took {time.time() - t0:.2f} seconds")
    print_resource_usage()

    # Aydin preserves the input intensity range by default, but a small
    # overshoot is possible. Clip to the original dtype range for safety.
    info = np.iinfo(np.uint16)
    denoised_f = np.clip(denoised_f, info.min, info.max)
    denoised_u16 = denoised_f.astype(np.uint16)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    t0 = time.time()
    base_name = os.path.splitext(image_name)[0]
    # Drop any trailing "_cropped" / "_processed" suffix so successive
    # preprocessing stages don't pile up naming: t0001_Channel 1.tif
    # becomes t0001_Channel 1_denoised.tif; a re-run after cropping yields
    # t0001_Channel 1_cropped_denoised.tif, which is what the next stage
    # expects when it falls back to the rename patterns in
    # PREPROCESS_DECONVOLVE.
    image_out_name = f"{base_name}_denoised.tif"
    img_out_path = os.path.join(args.outdir, image_out_name)
    print(f"[Check-in] Saving denoised stack to: {img_out_path}")

    # Preserve the original ImageJ metadata (axes, spacing, unit) and the
    # pixel resolution. downstream stages re-save with their own metadata
    # tweaks — this is a no-op for them.
    save_kwargs = {"imagej": True}
    if original_imagej_meta:
        # `spacing`, `unit`, `axes` are the keys the rest of the pipeline
        # reads; we re-emit them verbatim.
        save_kwargs["metadata"] = {
            k: v for k, v in original_imagej_meta.items()
            if k in ("spacing", "unit", "axes", "slices", "frames")
        }
    # X/Y resolution comes from the TIFF tags, not from ImageJ metadata.
    with tifffile.TiffFile(image_path) as tif:
        first_tags = tif.pages[0].tags
        if "XResolution" in first_tags:
            num, denom = first_tags["XResolution"].value
            x_res = denom / num if num != 0 else 1.0
        else:
            x_res = 1.0 / physical_pixel_sizeX
        if "YResolution" in first_tags:
            num, denom = first_tags["YResolution"].value
            y_res = denom / num if num != 0 else 1.0
        else:
            y_res = 1.0 / physical_pixel_sizeY
    save_kwargs["resolution"] = (1.0 / x_res, 1.0 / y_res)

    tifffile.imwrite(img_out_path, denoised_u16, **save_kwargs)

    print(f"[Timer] Saving image took {time.time() - t0:.2f} seconds")

    if not os.path.isfile(img_out_path):
        print(f"ERROR: Output file was not created: {img_out_path}")
        sys.exit(1)
    output_size = os.path.getsize(img_out_path)
    if output_size == 0:
        print(f"ERROR: Output file is empty: {img_out_path}")
        sys.exit(1)
    print(f"[Success] Output size: {output_size / (1024**2):.2f} MB")
    print(f"[Done] Elapsed Time: {time.time() - start_time_total:.4f} seconds")
    print_resource_usage()


if __name__ == "__main__":
    main()
