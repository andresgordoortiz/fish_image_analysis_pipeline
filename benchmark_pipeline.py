#!/usr/bin/env python3
"""
SPIM Pipeline Benchmark Script
IMP Vienna - Andrés Gordo & Guilherme Ventura

Computes objective quality metrics from pipeline outputs to compare parameter
experiments. Run on a results directory (or multiple) and get a CSV table
you can use to pick the best settings.

Usage:
    # Single experiment
    python benchmark_pipeline.py --results_dir ./results/

    # Compare multiple experiments
    python benchmark_pipeline.py \
        --results_dir ./results_expA/ ./results_expB/ ./results_expC/ \
        --labels "Current" "ExpA_edge_fix" "ExpB_z_depth"

    # Only benchmark a subset of timepoints (faster)
    python benchmark_pipeline.py --results_dir ./results/ --timepoints 0 10 20

    # Also compare raw vs preprocessed (needs input dir)
    python benchmark_pipeline.py --results_dir ./results/ --raw_dir /path/to/input/
"""

import argparse
import os
import sys
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# PREPROCESSING METRICS
# =============================================================================

def compute_snr(img):
    """
    Signal-to-noise ratio.
    Uses the 95th percentile as 'signal' and the standard deviation of the
    lowest-10% region as 'noise'. Higher = better deconvolution.
    """
    flat = img.ravel().astype(np.float64)
    p10 = np.percentile(flat, 10)
    noise_region = flat[flat <= p10]
    if len(noise_region) < 100 or np.std(noise_region) < 1e-10:
        return float("inf")
    signal = np.percentile(flat, 95)
    noise = np.std(noise_region)
    return float(signal / noise)


def compute_edge_artifact_ratio(img, border_px=10):
    """
    Ratio of mean border intensity to mean interior intensity.
    Values >> 1 indicate bright edge artifacts from deconvolution.
    Ideal value: ~1.0 or less.
    """
    z, y, x = img.shape
    # Create border mask
    mask = np.zeros(img.shape, dtype=bool)
    mask[:border_px, :, :] = True
    mask[-border_px:, :, :] = True
    mask[:, :border_px, :] = True
    mask[:, -border_px:, :] = True
    mask[:, :, :border_px] = True
    mask[:, :, -border_px:] = True

    interior = img[~mask].astype(np.float64)
    border = img[mask].astype(np.float64)

    if np.mean(interior) < 1e-10:
        return float("inf")
    return float(np.mean(border) / np.mean(interior))


def compute_z_uniformity(img):
    """
    Coefficient of variation (std/mean) of per-slice mean intensities.
    Lower = more uniform Z illumination after correction.
    """
    slice_means = np.array([img[z].mean() for z in range(img.shape[0])], dtype=np.float64)
    if slice_means.mean() < 1e-10:
        return float("inf")
    return float(slice_means.std() / slice_means.mean())


def compute_sharpness(img, sample_slices=5):
    """
    Gradient-based focus metric averaged over sample Z slices.
    Higher = sharper image (better deconvolution).
    Uses the variance of the Laplacian (Pech-Pacheco et al., 2000).
    """
    z_indices = np.linspace(0, img.shape[0] - 1, sample_slices, dtype=int)
    scores = []
    for z in z_indices:
        sl = img[z].astype(np.float64)
        laplacian = ndi.laplace(sl)
        scores.append(np.var(laplacian))
    return float(np.mean(scores))


def compute_dynamic_range(img):
    """
    Effective dynamic range: (p99.9 - p0.1) / dtype_max.
    Higher = better use of available bit depth after normalization.
    """
    p_low = np.percentile(img, 0.1)
    p_high = np.percentile(img, 99.9)
    if np.issubdtype(img.dtype, np.integer):
        dtype_max = np.iinfo(img.dtype).max
    else:
        dtype_max = p_high if p_high > 0 else 1.0
    return float((p_high - p_low) / dtype_max)


def compute_z_frequency_content(img, sample_columns=20):
    """
    Mean Z-axis frequency content (high-freq power ratio).
    Measured by computing the FFT along Z for a grid of (y,x) positions
    and reporting the fraction of power above the Nyquist/4 frequency.
    Higher = more Z detail preserved (better deconvolution/reslicing).
    """
    z, y, x = img.shape
    ys = np.linspace(y // 4, 3 * y // 4, sample_columns, dtype=int)
    xs = np.linspace(x // 4, 3 * x // 4, sample_columns, dtype=int)

    hf_ratios = []
    for yi in ys:
        for xi in xs:
            line = img[:, yi, xi].astype(np.float64)
            if line.std() < 1e-10:
                continue
            fft_mag = np.abs(np.fft.rfft(line))
            n = len(fft_mag)
            cutoff = n // 4  # top 75% of frequencies = "high frequency"
            total_power = np.sum(fft_mag ** 2)
            if total_power < 1e-20:
                continue
            hf_power = np.sum(fft_mag[cutoff:] ** 2)
            hf_ratios.append(hf_power / total_power)

    return float(np.mean(hf_ratios)) if hf_ratios else 0.0


# =============================================================================
# SEGMENTATION METRICS
# =============================================================================

def compute_segmentation_metrics(mask):
    """
    Compute all segmentation quality metrics from a label image.
    Returns a dict with all metrics.
    """
    labels = np.unique(mask)
    labels = labels[labels > 0]  # exclude background
    n_cells = len(labels)

    if n_cells == 0:
        return {
            "seg_n_cells": 0,
            "seg_mean_volume_px": 0,
            "seg_std_volume_px": 0,
            "seg_cv_volume": 0,
            "seg_median_volume_px": 0,
            "seg_small_fraction": 0,
            "seg_large_fraction": 0,
            "seg_mean_z_extent": 0,
            "seg_mean_xy_extent": 0,
            "seg_mean_elongation_z": 0,
            "seg_z_coverage": 0,
            "seg_z_density_cv": 0,
            "seg_mean_sphericity": 0,
            "seg_touching_fraction": 0,
        }

    # --- Volume statistics ---
    volumes = np.array([np.sum(mask == l) for l in labels], dtype=np.float64)
    mean_vol = np.mean(volumes)
    std_vol = np.std(volumes)
    cv_vol = std_vol / mean_vol if mean_vol > 0 else 0
    median_vol = np.median(volumes)

    # Small objects (potential over-segmentation): < 25% of median volume
    small_thresh = 0.25 * median_vol
    small_fraction = np.sum(volumes < small_thresh) / n_cells

    # Large objects (potential under-segmentation): > 4x median volume
    large_thresh = 4.0 * median_vol
    large_fraction = np.sum(volumes > large_thresh) / n_cells

    # --- Z vs XY extent (the key metric for your Z-depth problem) ---
    z_extents = []
    xy_extents = []
    sphericities = []

    # Sample up to 500 cells to keep runtime manageable
    sample_labels = labels if n_cells <= 500 else np.random.choice(labels, 500, replace=False)

    for l in sample_labels:
        coords = np.argwhere(mask == l)  # (N, 3) array of [z, y, x]
        if len(coords) < 3:
            continue

        z_range = coords[:, 0].max() - coords[:, 0].min() + 1
        y_range = coords[:, 1].max() - coords[:, 1].min() + 1
        x_range = coords[:, 2].max() - coords[:, 2].min() + 1

        z_extents.append(z_range)
        xy_extent = (y_range + x_range) / 2.0
        xy_extents.append(xy_extent)

        # Sphericity proxy: ratio of volume to bounding box volume
        # 1.0 = fills bounding box, π/6 ≈ 0.524 for perfect sphere
        bb_vol = z_range * y_range * x_range
        vol = np.sum(mask == l)
        sphericities.append(vol / bb_vol if bb_vol > 0 else 0)

    mean_z_extent = float(np.mean(z_extents)) if z_extents else 0
    mean_xy_extent = float(np.mean(xy_extents)) if xy_extents else 0

    # Z-elongation: ratio of Z-extent to XY-extent
    # For isotropic nuclei this should be ~1.0
    # < 1 means nuclei are flattened in Z (under-segmented in depth)
    # > 1 means nuclei are elongated in Z
    elongation_z = float(np.mean(
        [z / xy for z, xy in zip(z_extents, xy_extents) if xy > 0]
    )) if z_extents else 0

    # --- Z-coverage: fraction of Z slices containing at least 1 cell ---
    z_slices_with_cells = len(np.unique(np.argwhere(mask > 0)[:, 0]))
    z_coverage = z_slices_with_cells / mask.shape[0]

    # --- Z-density uniformity: how evenly are cells distributed across Z ---
    cells_per_z = np.zeros(mask.shape[0])
    for z in range(mask.shape[0]):
        cells_per_z[z] = len(np.unique(mask[z])) - (1 if 0 in mask[z] else 0)
    # Only consider slices that have cells
    active_slices = cells_per_z[cells_per_z > 0]
    z_density_cv = float(active_slices.std() / active_slices.mean()) if len(active_slices) > 1 and active_slices.mean() > 0 else 0

    # --- Mean sphericity ---
    mean_sphericity = float(np.mean(sphericities)) if sphericities else 0

    # --- Touching/merged fraction: cells whose bounding boxes overlap ---
    # (Fast approximation: count cells that share a face with another label)
    touching = 0
    # Dilate each axis by 1 and check overlap — sample-based
    dilated = ndi.maximum_filter(mask, size=3)
    for l in sample_labels:
        region = dilated[mask == l]
        neighbors = np.unique(region)
        neighbors = neighbors[(neighbors != l) & (neighbors != 0)]
        if len(neighbors) > 0:
            touching += 1
    touching_fraction = touching / len(sample_labels) if len(sample_labels) > 0 else 0

    return {
        "seg_n_cells": int(n_cells),
        "seg_mean_volume_px": float(mean_vol),
        "seg_std_volume_px": float(std_vol),
        "seg_cv_volume": float(cv_vol),
        "seg_median_volume_px": float(median_vol),
        "seg_small_fraction": float(small_fraction),
        "seg_large_fraction": float(large_fraction),
        "seg_mean_z_extent": mean_z_extent,
        "seg_mean_xy_extent": mean_xy_extent,
        "seg_mean_elongation_z": elongation_z,
        "seg_z_coverage": float(z_coverage),
        "seg_z_density_cv": z_density_cv,
        "seg_mean_sphericity": mean_sphericity,
        "seg_touching_fraction": float(touching_fraction),
    }


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_timepoint_files(results_dir, timepoints=None):
    """
    Discover preprocessed and segmented files in a results directory.
    Returns dict: { timepoint: { 'preprocessed': path, 'segmented': path } }
    """
    files = {}

    # Preprocessed
    preproc_dir = os.path.join(results_dir, "01_preprocessed")
    if os.path.isdir(preproc_dir):
        for f in sorted(glob.glob(os.path.join(preproc_dir, "t*_processed.tif"))):
            name = os.path.basename(f)
            # Extract timepoint number from t0000_processed.tif
            tp = int(name.split("_")[0][1:])
            if timepoints is not None and tp not in timepoints:
                continue
            files.setdefault(tp, {})["preprocessed"] = f

    # Segmented
    seg_dir = os.path.join(results_dir, "02_segmented")
    if os.path.isdir(seg_dir):
        for f in sorted(glob.glob(os.path.join(seg_dir, "t*_segmented.tif"))):
            name = os.path.basename(f)
            tp = int(name.split("_")[0][1:])
            if timepoints is not None and tp not in timepoints:
                continue
            files.setdefault(tp, {})["segmented"] = f

    # Cropped (optional)
    crop_dir = os.path.join(results_dir, "00_cropped")
    if os.path.isdir(crop_dir):
        for f in sorted(glob.glob(os.path.join(crop_dir, "t*_cropped.tif"))):
            name = os.path.basename(f)
            tp = int(name.split("_")[0][1:])
            if timepoints is not None and tp not in timepoints:
                continue
            files.setdefault(tp, {})["cropped"] = f

    return files


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def benchmark_single_experiment(results_dir, label, timepoints=None, raw_dir=None):
    """
    Run all benchmarks on one experiment's results.
    Returns list of per-timepoint metric dicts.
    """
    files = find_timepoint_files(results_dir, timepoints)
    if not files:
        print(f"  WARNING: No files found in {results_dir}")
        return []

    print(f"\n{'='*70}")
    print(f"  Benchmarking: {label}")
    print(f"  Directory:    {results_dir}")
    print(f"  Timepoints:   {sorted(files.keys())}")
    print(f"{'='*70}")

    all_metrics = []

    for tp in sorted(files.keys()):
        tp_files = files[tp]
        metrics = {"experiment": label, "timepoint": tp}

        # --- Preprocessing metrics ---
        if "preprocessed" in tp_files:
            print(f"  t{tp:04d}: analyzing preprocessed image...", end="", flush=True)
            img = tifffile.imread(tp_files["preprocessed"])

            metrics["preproc_shape"] = f"{img.shape[0]}x{img.shape[1]}x{img.shape[2]}"
            metrics["preproc_snr"] = compute_snr(img)
            metrics["preproc_edge_ratio"] = compute_edge_artifact_ratio(img, border_px=10)
            metrics["preproc_z_uniformity_cv"] = compute_z_uniformity(img)
            metrics["preproc_sharpness"] = compute_sharpness(img)
            metrics["preproc_dynamic_range"] = compute_dynamic_range(img)
            metrics["preproc_z_freq_content"] = compute_z_frequency_content(img)
            metrics["preproc_mean_intensity"] = float(img.mean())
            metrics["preproc_p99_intensity"] = float(np.percentile(img, 99))

            # Compare with raw/cropped if available
            raw_path = None
            if "cropped" in tp_files:
                raw_path = tp_files["cropped"]
            elif raw_dir:
                # Try to find matching raw file
                candidates = glob.glob(os.path.join(raw_dir, f"t{tp:04d}_*.tif"))
                if candidates:
                    raw_path = candidates[0]

            if raw_path and os.path.isfile(raw_path):
                raw_img = tifffile.imread(raw_path)
                metrics["raw_snr"] = compute_snr(raw_img)
                metrics["snr_improvement"] = metrics["preproc_snr"] / metrics["raw_snr"] if metrics["raw_snr"] > 0 else 0
                metrics["raw_sharpness"] = compute_sharpness(raw_img)
                metrics["sharpness_improvement"] = metrics["preproc_sharpness"] / metrics["raw_sharpness"] if metrics["raw_sharpness"] > 0 else 0
                del raw_img

            del img
            print(" done")

        # --- Segmentation metrics ---
        if "segmented" in tp_files:
            print(f"  t{tp:04d}: analyzing segmentation mask...", end="", flush=True)
            mask = tifffile.imread(tp_files["segmented"])
            seg_metrics = compute_segmentation_metrics(mask)
            metrics.update(seg_metrics)
            del mask
            print(" done")

        all_metrics.append(metrics)

    return all_metrics


def print_summary_table(all_metrics, labels):
    """Print a comparison summary across experiments."""
    if not all_metrics:
        print("No metrics to display.")
        return

    # Group by experiment
    by_experiment = {}
    for m in all_metrics:
        exp = m["experiment"]
        by_experiment.setdefault(exp, []).append(m)

    # Compute per-experiment averages
    avg_keys = [
        ("preproc_snr", "SNR", "higher=better", ".1f"),
        ("preproc_edge_ratio", "Edge ratio", "~1.0=good, >>1=artifacts", ".3f"),
        ("preproc_z_uniformity_cv", "Z-uniformity CV", "lower=better", ".3f"),
        ("preproc_sharpness", "Sharpness", "higher=better", ".1f"),
        ("preproc_dynamic_range", "Dynamic range", "higher=better", ".3f"),
        ("preproc_z_freq_content", "Z freq content", "higher=more Z detail", ".4f"),
        ("snr_improvement", "SNR improvement", ">1=improved", ".2f"),
        ("seg_n_cells", "N cells", "consistency", ".0f"),
        ("seg_mean_volume_px", "Mean vol (px)", "consistency", ".0f"),
        ("seg_cv_volume", "Volume CV", "lower=more uniform", ".3f"),
        ("seg_small_fraction", "Small obj %", "lower=less over-seg", ".3f"),
        ("seg_large_fraction", "Large obj %", "lower=less under-seg", ".3f"),
        ("seg_mean_elongation_z", "Z/XY elongation", "~1.0=isotropic nuclei", ".3f"),
        ("seg_z_coverage", "Z coverage", "higher=better depth", ".3f"),
        ("seg_z_density_cv", "Z density CV", "lower=more uniform", ".3f"),
        ("seg_mean_sphericity", "Sphericity", "~0.52=spherical", ".3f"),
        ("seg_touching_fraction", "Touching %", "lower=better separated", ".3f"),
    ]

    print(f"\n{'='*90}")
    print(f"  BENCHMARK COMPARISON SUMMARY")
    print(f"{'='*90}")

    # Header
    col_w = 16
    header = f"{'Metric':<28} {'Ideal':<26}"
    for label in labels:
        header += f" {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for key, name, ideal, fmt in avg_keys:
        row = f"{name:<28} {ideal:<26}"
        any_data = False
        for label in labels:
            if label in by_experiment:
                vals = [m[key] for m in by_experiment[label] if key in m]
                if vals:
                    avg = np.mean(vals)
                    row += f" {avg:{col_w}{fmt}}"
                    any_data = True
                else:
                    row += f" {'N/A':>{col_w}}"
            else:
                row += f" {'N/A':>{col_w}}"
        if any_data:
            print(row)

    print(f"{'='*90}")

    # Per-timepoint detail for segmentation Z-elongation (the key metric)
    print(f"\n--- Z/XY Elongation per timepoint (target ≈ 1.0) ---")
    header = f"{'Timepoint':<12}"
    for label in labels:
        header += f" {label:>{col_w}}"
    print(header)

    all_tps = sorted(set(m["timepoint"] for m in all_metrics))
    for tp in all_tps:
        row = f"t{tp:04d}       "
        for label in labels:
            tp_metrics = [m for m in all_metrics if m["experiment"] == label and m["timepoint"] == tp]
            if tp_metrics and "seg_mean_elongation_z" in tp_metrics[0]:
                row += f" {tp_metrics[0]['seg_mean_elongation_z']:>{col_w}.3f}"
            else:
                row += f" {'N/A':>{col_w}}"
        print(row)


def save_csv(all_metrics, output_path):
    """Save all metrics to CSV."""
    if not all_metrics:
        return

    # Collect all keys
    all_keys = []
    for m in all_metrics:
        for k in m.keys():
            if k not in all_keys:
                all_keys.append(k)

    with open(output_path, "w") as f:
        f.write(",".join(all_keys) + "\n")
        for m in all_metrics:
            row = []
            for k in all_keys:
                v = m.get(k, "")
                row.append(str(v))
            f.write(",".join(row) + "\n")

    print(f"\nFull metrics saved to: {output_path}")


def save_json(all_metrics, output_path):
    """Save all metrics to JSON for programmatic access."""
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Full metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SPIM pipeline outputs across experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python benchmark_pipeline.py --results_dir ./results/

  # Compare experiments
  python benchmark_pipeline.py \\
      --results_dir ./results_current/ ./results_expA/ ./results_expB/ \\
      --labels "Current" "Exp_A" "Exp_B"

  # Fast: only 3 timepoints
  python benchmark_pipeline.py --results_dir ./results/ --timepoints 0 10 20

Key metrics to watch:
  - Edge ratio:       ~1.0 = no edge artifacts, >>1 = bright edges
  - Z/XY elongation:  ~1.0 = isotropic nuclei, <1 = flattened in Z
  - Z coverage:       1.0 = cells found in every Z slice
  - Small obj %%:      Lower = less over-segmentation
  - SNR:              Higher = cleaner signal
        """
    )

    parser.add_argument(
        "--results_dir", type=str, nargs="+", required=True,
        help="One or more results directories to benchmark"
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Labels for each experiment (must match number of results_dirs)"
    )
    parser.add_argument(
        "--timepoints", type=int, nargs="+", default=None,
        help="Only benchmark these timepoints (default: all)"
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help="Raw input directory for SNR/sharpness improvement comparison"
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Save full metrics to CSV (default: benchmark_results.csv in first results_dir)"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Also save metrics as JSON"
    )

    args = parser.parse_args()

    # Validate labels
    if args.labels:
        if len(args.labels) != len(args.results_dir):
            print(f"ERROR: {len(args.labels)} labels provided but {len(args.results_dir)} results directories")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.normpath(d)) for d in args.results_dir]

    # Run benchmarks
    all_metrics = []
    for results_dir, label in zip(args.results_dir, labels):
        if not os.path.isdir(results_dir):
            print(f"WARNING: Directory not found: {results_dir}")
            continue
        metrics = benchmark_single_experiment(
            results_dir, label,
            timepoints=set(args.timepoints) if args.timepoints else None,
            raw_dir=args.raw_dir
        )
        all_metrics.extend(metrics)

    if not all_metrics:
        print("ERROR: No metrics computed. Check that results directories contain pipeline outputs.")
        sys.exit(1)

    # Print summary
    print_summary_table(all_metrics, labels)

    # Save outputs
    csv_path = args.output_csv or os.path.join(args.results_dir[0], "benchmark_results.csv")
    save_csv(all_metrics, csv_path)

    if args.output_json:
        save_json(all_metrics, args.output_json)

    # Print interpretation guide
    print(f"""
{'='*70}
  HOW TO INTERPRET
{'='*70}

  PREPROCESSING (comparing deconvolution quality):
    SNR                  Higher is better. Compare across experiments.
    Edge ratio           Should be ~1.0. Values > 1.5 mean edge artifacts.
    Z-uniformity CV      Lower is better. Shows Z-correction quality.
    Sharpness            Higher is better. Shows deconvolution quality.
    Z freq content       Higher means more Z-axis detail preserved.

  SEGMENTATION (comparing nuclei detection quality):
    N cells              Should be consistent across experiments.
    Z/XY elongation      THE KEY METRIC for your Z-depth problem.
                         ~1.0 = nuclei are round in all dimensions.
                         < 0.7 = nuclei are cut short in Z (under-segmented).
                         > 1.5 = nuclei are elongated in Z (over-segmented).
    Z coverage           Should be close to 1.0 (cells in all Z slices).
    Small obj %          High values = over-segmentation (fragments).
    Large obj %          High values = under-segmentation (merged nuclei).
    Sphericity           ~0.52 for perfect spheres. Consistent = good.
    Touching %           Lower = better-separated nuclei.
{'='*70}
""")


if __name__ == "__main__":
    main()
