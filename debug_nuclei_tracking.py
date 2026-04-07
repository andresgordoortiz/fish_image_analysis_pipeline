#!/usr/bin/env python3
"""
Debug Preprocessing — Per-Stage Nuclei Analysis
=================================================
IMP Vienna — Andrés Gordo

Runs Cellpose segmentation on a SINGLE intermediate TIFF from the
preprocessing pipeline and computes detailed per-slice nuclei statistics.
Designed to be called in parallel by Nextflow (one invocation per
intermediate stage).

The companion script ``debug_nuclei_report.py`` collects all per-stage
JSON outputs and generates comparison plots.

Usage (standalone — normally called by Nextflow)
-------------------------------------------------
    python debug_nuclei_tracking.py \
        --input_file intermediates/07_after_wbns.tif \
        --stage_name 07_after_wbns \
        --cellpose_model /path/to/model \
        --diameter 30 \
        --outdir ./debug_results/ \
        --do_3d \
        --use_gpu
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import tifffile
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Nuclei analysis helpers
# ============================================================================


def per_slice_nuclei_stats(mask_3d):
    """Compute detailed nuclei statistics per Z-slice.

    Returns a dict with arrays of length Z, each holding one value per slice.
    """
    Z = mask_3d.shape[0]
    stats = {
        "z_index": list(range(Z)),
        "n_nuclei": [],          # unique labels touching this slice
        "n_nuclei_centroid": [],  # nuclei whose centroid falls in this slice
        "total_area_px": [],     # total labelled pixels in slice
        "mean_area_px": [],      # mean nucleus area in slice
        "median_area_px": [],
        "std_area_px": [],
        "min_area_px": [],
        "max_area_px": [],
        "fraction_labelled": [],  # fraction of non-zero pixels
    }

    # Pre-compute centroid Z for each label via find_objects
    slices_list = ndi.find_objects(mask_3d)
    max_label = mask_3d.max()
    centroid_z = {}
    for lab in range(1, max_label + 1):
        s = slices_list[lab - 1]
        if s is None:
            continue
        z_start, z_stop = s[0].start, s[0].stop
        # Weighted centroid along Z for this label
        sub_mask = mask_3d[z_start:z_stop] == lab
        z_coords = np.arange(z_start, z_stop)
        z_counts = np.array([np.count_nonzero(sub_mask[i]) for i in range(sub_mask.shape[0])])
        total = z_counts.sum()
        if total > 0:
            centroid_z[lab] = float(np.sum(z_coords * z_counts) / total)

    for zi in range(Z):
        sl = mask_3d[zi]
        labels_in_slice = np.unique(sl)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]
        n_labels = len(labels_in_slice)

        # Count nuclei whose centroid is in this slice
        n_centroid = sum(
            1 for lab in labels_in_slice
            if lab in centroid_z and abs(centroid_z[lab] - zi) < 0.5
        )

        if n_labels == 0:
            stats["n_nuclei"].append(0)
            stats["n_nuclei_centroid"].append(0)
            stats["total_area_px"].append(0)
            stats["mean_area_px"].append(0.0)
            stats["median_area_px"].append(0.0)
            stats["std_area_px"].append(0.0)
            stats["min_area_px"].append(0)
            stats["max_area_px"].append(0)
            stats["fraction_labelled"].append(0.0)
            continue

        areas = np.array([np.count_nonzero(sl == lab) for lab in labels_in_slice])
        stats["n_nuclei"].append(int(n_labels))
        stats["n_nuclei_centroid"].append(int(n_centroid))
        stats["total_area_px"].append(int(areas.sum()))
        stats["mean_area_px"].append(float(np.mean(areas)))
        stats["median_area_px"].append(float(np.median(areas)))
        stats["std_area_px"].append(float(np.std(areas)))
        stats["min_area_px"].append(int(areas.min()))
        stats["max_area_px"].append(int(areas.max()))
        stats["fraction_labelled"].append(float(areas.sum() / sl.size))

    return stats


def global_nuclei_stats(mask_3d):
    """Compute volume-level (3D) nuclei statistics."""
    labels_all = np.unique(mask_3d)
    labels_all = labels_all[labels_all > 0]
    n_nuclei_3d = len(labels_all)

    if n_nuclei_3d == 0:
        return {
            "n_nuclei_3d": 0,
            "mean_volume_px": 0.0,
            "median_volume_px": 0.0,
            "std_volume_px": 0.0,
            "min_volume_px": 0,
            "max_volume_px": 0,
            "volume_cv": 0.0,
            "mean_z_extent": 0.0,
            "mean_xy_extent": 0.0,
            "small_fraction": 0.0,
            "large_fraction": 0.0,
            "z_coverage": 0.0,
            "cells_per_z_mean": 0.0,
            "cells_per_z_std": 0.0,
        }

    # Volumes via bincount (single pass)
    flat = mask_3d.ravel()
    counts = np.bincount(flat)
    volumes = counts[labels_all].astype(np.float64)

    median_vol = float(np.median(volumes))
    small_thresh = 0.25 * median_vol if median_vol > 0 else 1
    large_thresh = 4.0 * median_vol if median_vol > 0 else 1e9

    # Z and XY extents via find_objects
    slices_list = ndi.find_objects(mask_3d)
    z_extents = []
    xy_extents = []
    for lab in labels_all:
        s = slices_list[lab - 1]
        if s is None:
            continue
        z_extents.append(s[0].stop - s[0].start)
        y_ext = s[1].stop - s[1].start
        x_ext = s[2].stop - s[2].start
        xy_extents.append((y_ext + x_ext) / 2.0)

    # Z coverage
    z_with_cells = np.count_nonzero(np.any(mask_3d > 0, axis=(1, 2)))
    z_coverage = z_with_cells / mask_3d.shape[0]

    # Cells per Z
    cells_per_z = np.array([
        len(np.unique(mask_3d[z][mask_3d[z] > 0])) if np.any(mask_3d[z] > 0) else 0
        for z in range(mask_3d.shape[0])
    ])
    active = cells_per_z[cells_per_z > 0]

    return {
        "n_nuclei_3d": int(n_nuclei_3d),
        "mean_volume_px": float(np.mean(volumes)),
        "median_volume_px": float(np.median(volumes)),
        "std_volume_px": float(np.std(volumes)),
        "min_volume_px": int(np.min(volumes)),
        "max_volume_px": int(np.max(volumes)),
        "volume_cv": float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.0,
        "mean_z_extent": float(np.mean(z_extents)) if z_extents else 0.0,
        "mean_xy_extent": float(np.mean(xy_extents)) if xy_extents else 0.0,
        "small_fraction": float(np.sum(volumes < small_thresh) / n_nuclei_3d),
        "large_fraction": float(np.sum(volumes > large_thresh) / n_nuclei_3d),
        "z_coverage": float(z_coverage),
        "cells_per_z_mean": float(np.mean(active)) if active.size > 0 else 0.0,
        "cells_per_z_std": float(np.std(active)) if active.size > 0 else 0.0,
    }


def intensity_in_nuclei(image, mask_3d):
    """Compute per-nucleus intensity statistics.

    Returns summary statistics, not per-nucleus detail (to keep JSON small).
    """
    labels_all = np.unique(mask_3d)
    labels_all = labels_all[labels_all > 0]

    if len(labels_all) == 0:
        return {"mean_intensity": 0.0, "std_intensity": 0.0,
                "min_intensity": 0.0, "max_intensity": 0.0,
                "snr_in_nuclei": 0.0}

    img_f = image.astype(np.float64)

    # Global: mean intensity inside nuclei vs outside (background)
    in_mask = mask_3d > 0
    nucleus_mean = float(np.mean(img_f[in_mask]))
    bg_pixels = img_f[~in_mask]
    bg_std = float(np.std(bg_pixels)) if bg_pixels.size > 100 else 1e-8

    # Per-nucleus mean intensities (via bincount weighted sum)
    flat_img = img_f.ravel()
    flat_mask = mask_3d.ravel()
    sums = np.bincount(flat_mask, weights=flat_img)
    counts = np.bincount(flat_mask)
    # Skip label 0 (background)
    valid = counts[1:] > 0
    per_nuc_means = (sums[1:][valid] / counts[1:][valid])

    return {
        "mean_intensity": float(np.mean(per_nuc_means)) if per_nuc_means.size > 0 else 0.0,
        "std_intensity": float(np.std(per_nuc_means)) if per_nuc_means.size > 0 else 0.0,
        "min_intensity": float(np.min(per_nuc_means)) if per_nuc_means.size > 0 else 0.0,
        "max_intensity": float(np.max(per_nuc_means)) if per_nuc_means.size > 0 else 0.0,
        "snr_in_nuclei": float(nucleus_mean / bg_std) if bg_std > 1e-8 else 0.0,
    }


# ============================================================================
# Cellpose runner
# ============================================================================


def run_cellpose(image, model_path, diameter, do_3d, use_gpu, flow_threshold,
                 cellprob_threshold, min_size, anisotropy=None):
    """Run Cellpose on a 3D image and return the label mask."""
    try:
        from cellpose import models
    except ImportError:
        print("ERROR: cellpose not installed. Install with: pip install cellpose")
        sys.exit(1)

    print(f"  Loading Cellpose model: {model_path}")
    model = models.CellposeModel(pretrained_model=model_path, gpu=use_gpu)

    print(f"  Running Cellpose (do_3d={do_3d}, diameter={diameter}, "
          f"flow={flow_threshold}, prob={cellprob_threshold})")

    kwargs = {
        "diameter": diameter,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "min_size": min_size,
        "do_3D": do_3d,
    }
    if anisotropy is not None:
        kwargs["anisotropy"] = anisotropy

    masks, flows, styles = model.eval(image, **kwargs)

    masks = np.asarray(masks, dtype=np.int32)
    print(f"  Cellpose found {masks.max()} nuclei")
    return masks


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Debug: analyse nuclei at one preprocessing stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_file", required=True, help="Intermediate TIFF to segment")
    parser.add_argument("--stage_name", required=True, help="Pipeline stage name (e.g. 07_after_wbns)")
    parser.add_argument("--outdir", required=True, help="Output directory for metrics JSON + mask TIFF")
    parser.add_argument("--cellpose_model", required=True, help="Path to Cellpose model")
    parser.add_argument("--diameter", type=float, default=30, help="Expected nucleus diameter (px)")
    parser.add_argument("--flow_threshold", type=float, default=0.8)
    parser.add_argument("--cellprob_threshold", type=float, default=0.0)
    parser.add_argument("--min_size", type=int, default=15)
    parser.add_argument("--do_3d", action="store_true", help="Run 3D segmentation")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for Cellpose")
    parser.add_argument("--anisotropy", type=float, default=None, help="Anisotropy for 3D segmentation")
    parser.add_argument("--save_mask", action="store_true", help="Save segmentation mask as TIFF")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stage = args.stage_name

    print(f"\n{'='*60}")
    print(f"DEBUG NUCLEI ANALYSIS — {stage}")
    print(f"{'='*60}")
    print(f"Input: {args.input_file}")

    # Load image
    t0 = time.time()
    img = tifffile.imread(args.input_file)
    print(f"Loaded: shape={img.shape}, dtype={img.dtype} ({time.time()-t0:.1f}s)")

    if img.ndim != 3:
        print(f"ERROR: Expected 3D image, got {img.ndim}D")
        sys.exit(1)

    # Normalise to float32 for Cellpose (expects [0, 1] or reasonable range)
    img_f32 = img.astype(np.float32)
    p999 = np.percentile(img_f32[img_f32 > 0], 99.9) if np.any(img_f32 > 0) else 1.0
    if p999 > 0:
        img_f32 = img_f32 / p999
    img_f32 = np.clip(img_f32, 0, 1)

    # Segment with Cellpose
    t0 = time.time()
    mask = run_cellpose(
        img_f32,
        model_path=args.cellpose_model,
        diameter=args.diameter,
        do_3d=args.do_3d,
        use_gpu=args.use_gpu,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        anisotropy=args.anisotropy,
    )
    seg_time = time.time() - t0
    print(f"Segmentation took {seg_time:.1f}s")

    # Compute per-slice stats
    print("Computing per-slice nuclei statistics...")
    t0 = time.time()
    slice_stats = per_slice_nuclei_stats(mask)
    print(f"  Per-slice stats: {time.time()-t0:.1f}s")

    # Compute global 3D stats
    print("Computing global 3D nuclei statistics...")
    t0 = time.time()
    glob_stats = global_nuclei_stats(mask)
    print(f"  Global stats: {time.time()-t0:.1f}s")

    # Compute intensity-in-nuclei stats
    print("Computing intensity statistics within nuclei...")
    t0 = time.time()
    intens_stats = intensity_in_nuclei(img, mask)
    print(f"  Intensity stats: {time.time()-t0:.1f}s")

    # Image-level statistics (for context)
    flat = img.ravel().astype(np.float64)
    image_stats = {
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(np.mean(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "frac_zero": float(np.sum(flat == 0) / flat.size),
    }

    # Assemble output
    result = {
        "stage_name": stage,
        "input_file": args.input_file,
        "segmentation_time_s": seg_time,
        "cellpose_params": {
            "model": args.cellpose_model,
            "diameter": args.diameter,
            "flow_threshold": args.flow_threshold,
            "cellprob_threshold": args.cellprob_threshold,
            "min_size": args.min_size,
            "do_3d": args.do_3d,
            "anisotropy": args.anisotropy,
        },
        "image_stats": image_stats,
        "global_nuclei_stats": glob_stats,
        "intensity_in_nuclei": intens_stats,
        "per_slice_stats": slice_stats,
    }

    # Save JSON
    json_path = os.path.join(args.outdir, f"{stage}_nuclei_stats.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved metrics: {json_path}")

    # Optionally save mask
    if args.save_mask:
        mask_path = os.path.join(args.outdir, f"{stage}_mask.tif")
        tifffile.imwrite(mask_path, mask.astype(np.uint16))
        print(f"Saved mask: {mask_path}")

    # Print summary
    print(f"\n--- Summary: {stage} ---")
    print(f"  3D nuclei:      {glob_stats['n_nuclei_3d']}")
    print(f"  Volume (median): {glob_stats['median_volume_px']:.0f} px")
    print(f"  Volume CV:       {glob_stats['volume_cv']:.3f}")
    print(f"  Z coverage:      {glob_stats['z_coverage']:.2%}")
    print(f"  Small fraction:  {glob_stats['small_fraction']:.2%}")
    print(f"  Large fraction:  {glob_stats['large_fraction']:.2%}")
    print(f"  SNR in nuclei:   {intens_stats['snr_in_nuclei']:.1f}")
    print(f"  Max per-slice:   {max(slice_stats['n_nuclei'])}")


if __name__ == "__main__":
    main()
