#!/usr/bin/env python3
"""
CLAHE Parameter Sweep for Deep Nuclei Sensitivity
===================================================
IMP Vienna — Andrés Gordo

Re-applies CLAHE + final normalization on a saved intermediate (e.g.
07_after_wbns.tif) with many parameter combinations, then produces a
visual comparison panel and metrics CSV so you can pick the settings
that keep deep nuclei visible without amplifying noise.

Requires:  intermediates directory from a pipeline run with
           --save_intermediates (or save_intermediates: true in config.json).

The pipeline (spim_pipeline_fixed.py) writes intermediates using the new
dense stage numbering:

  01_after_load.tif
  02_after_downscale_xy.tif
  03_after_shading.tif
  04_after_z_correction.tif
  05_after_isotropic_reslice.tif
  06_after_deconv3d.tif
  07_after_deconv_xz.tif
  08_after_wbns.tif
  09_after_gaussian.tif
  10_after_clahe.tif
  11_after_percentile_norm.tif
  12_after_final_cast.tif

Usage
-----
    # Minimal — sweep on one timepoint's intermediates
    python clahe_sweep.py \
        --intermediates_dir ./subset_results/01_preprocessed/intermediates/ \
        --config_json ./config.json \
        --outdir ./clahe_sweep_results/

    # Provide raw + processed for SNR comparison
    python clahe_sweep.py \
        --intermediates_dir ./subset_results/01_preprocessed/intermediates/ \
        --config_json ./config.json \
        --outdir ./clahe_sweep_results/ \
        --raw_file ./subset/t0001.tif

What it produces
----------------
clahe_sweep_results/
├── sweep_config.json              # parameters tested
├── sweep_metrics.csv              # per-experiment: SNR, dynamic range, etc.
├── panel_sweep_midZ.png           # mid-Z slice for every experiment
├── panel_sweep_deepZ.png          # deep (75%-Z) slice for every experiment
├── panel_sweep_z_profiles.png     # per-slice mean intensity for each experiment
├── panel_sweep_snr.png            # per-slice SNR for each experiment
├── panel_sweep_histograms.png     # intensity histograms
├── sweep_report.txt               # text recommendations
└── experiment_XX/                 # full TIFF for each experiment (optional)
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# CLAHE (replicated from spim_pipeline_fixed.py to avoid import issues)
# ============================================================================


def clahe_3d_stack(
    stack,
    clip_limit=0.01,
    kernel_size=None,
    axis=0,
    preserve_dtype=True,
    p_low=0.5,
    p_high=99.5,
    eps=1e-8,
    bg_threshold_pct=5.0,
    min_signal_pct=2.0,
    dim_skip_mask=None,
):
    from skimage import exposure

    if stack.ndim != 3:
        raise ValueError(f"Expected 3D array, got {stack.ndim}D")
    in_dtype = stack.dtype
    s = np.moveaxis(stack, axis, 0).astype(np.float32, copy=False)
    out = np.empty_like(s, dtype=np.float32)

    if dim_skip_mask is not None and len(dim_skip_mask) != s.shape[0]:
        dim_skip_mask = None

    # Global signal reference
    slice_medians = []
    for i in range(s.shape[0]):
        nz = s[i][s[i] > 0]
        if nz.size > 100:
            slice_medians.append(float(np.median(nz)))
    global_median = np.median(slice_medians) if slice_medians else 1.0

    # Reference percentiles from non-skipped slices
    _ref_idx = list(range(s.shape[0]))
    if dim_skip_mask is not None and np.any(dim_skip_mask):
        _ref_idx = [i for i in range(s.shape[0]) if not dim_skip_mask[i]]
    if _ref_idx:
        _ref_stack = s[_ref_idx]
        _ref_nz = _ref_stack[_ref_stack > 0]
        if _ref_nz.size > 100:
            _global_p_low = float(np.percentile(_ref_nz, p_low))
            _global_p_high = float(np.percentile(_ref_nz, p_high))
        else:
            _global_p_low, _global_p_high = 0.0, 1.0
    else:
        _global_p_low, _global_p_high = 0.0, 1.0

    skipped = 0
    for i in range(s.shape[0]):
        sl = s[i]

        # Dim-skip mask: pass-through with scale normalisation
        if dim_skip_mask is not None and dim_skip_mask[i]:
            sl_norm = np.clip((sl - _global_p_low) / (_global_p_high - _global_p_low + eps), 0, 1)
            out[i] = sl_norm
            skipped += 1
            continue

        # Signal gate: skip CLAHE on near-empty slices
        nz = sl[sl > 0]
        if nz.size < 100:
            out[i] = np.zeros_like(sl)
            skipped += 1
            continue
        slice_med = float(np.median(nz))
        if min_signal_pct > 0 and slice_med < (min_signal_pct / 100.0) * global_median:
            sl_norm = np.clip((sl - _global_p_low) / (_global_p_high - _global_p_low + eps), 0, 1)
            out[i] = sl_norm
            skipped += 1
            continue

        # Normalise to [0, 1] using per-slice tissue percentiles
        nz = sl[sl > 0]
        lo = float(np.percentile(nz, p_low))
        hi = float(np.percentile(nz, p_high))
        if hi - lo < eps:
            out[i] = np.zeros_like(sl)
            skipped += 1
            continue
        sl_norm = np.clip((sl - lo) / (hi - lo), 0, 1)

        # Background suppression: voxels well below noise floor → 0
        if bg_threshold_pct > 0:
            bg_cut = bg_threshold_pct / 100.0
            sl_norm[sl_norm < bg_cut] = 0

        # CLAHE
        ks = kernel_size if kernel_size is not None else None
        sl_eq = exposure.equalize_adapthist(sl_norm, kernel_size=ks, clip_limit=clip_limit)
        out[i] = sl_eq.astype(np.float32)

    if skipped > 0:
        print(f"      CLAHE: skipped {skipped}/{s.shape[0]} slices")

    out = np.moveaxis(out, 0, axis)
    if not preserve_dtype:
        return out
    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        return np.clip(out * info.max, 0, info.max).astype(in_dtype, copy=False)
    return out.astype(in_dtype, copy=False)


# ============================================================================
# Metrics
# ============================================================================


def robust_snr_per_slice(stack, bg_percentile=10, signal_percentile=95):
    snr = np.zeros(stack.shape[0], dtype=np.float64)
    noise_std = np.zeros_like(snr)
    signal = np.zeros_like(snr)
    for zi in range(stack.shape[0]):
        sl = stack[zi].ravel().astype(np.float64)
        p_bg = np.percentile(sl, bg_percentile)
        bg_region = sl[sl <= p_bg]
        noise_std[zi] = float(np.std(bg_region)) if len(bg_region) > 50 else 0
        signal[zi] = float(np.percentile(sl, signal_percentile))
        snr[zi] = signal[zi] / noise_std[zi] if noise_std[zi] > 1e-10 else 0
    return snr, noise_std, signal


def compute_metrics(img, label):
    """Compute key quality metrics for one experiment."""
    flat = img.ravel().astype(np.float64)
    snr, noise, sig = robust_snr_per_slice(img)

    # Dynamic range (tissue only)
    tissue = flat[flat > 0]
    if tissue.size > 100:
        dr = float(np.percentile(tissue, 99.9) - np.percentile(tissue, 0.1))
    else:
        dr = 0.0

    # Deep-slice sensitivity: mean intensity of bottom-25% brightest region
    # in the deepest quarter of the stack.  Higher = better deep nuclei visibility.
    z_deep_start = int(0.75 * img.shape[0])
    deep_region = img[z_deep_start:]
    deep_tissue = deep_region[deep_region > 0].astype(np.float64)
    if deep_tissue.size > 100:
        deep_p75 = float(np.percentile(deep_tissue, 75))
        deep_p95 = float(np.percentile(deep_tissue, 95))
        deep_mean = float(np.mean(deep_tissue))
    else:
        deep_p75, deep_p95, deep_mean = 0.0, 0.0, 0.0

    # Shallow-slice reference
    z_shallow_end = int(0.25 * img.shape[0])
    shallow_region = img[:z_shallow_end]
    shallow_tissue = shallow_region[shallow_region > 0].astype(np.float64)
    if shallow_tissue.size > 100:
        shallow_p75 = float(np.percentile(shallow_tissue, 75))
    else:
        shallow_p75 = 1.0

    # Depth uniformity ratio: deep/shallow. Closer to 1.0 = more uniform
    depth_ratio = deep_p75 / shallow_p75 if shallow_p75 > 1e-6 else 0.0

    # Z-uniformity: CV of per-slice means (lower = more uniform)
    slice_means = np.array([float(np.mean(img[z])) for z in range(img.shape[0])])
    nz_means = slice_means[slice_means > 0]
    z_cv = float(nz_means.std() / nz_means.mean()) if nz_means.size > 1 and nz_means.mean() > 0 else 0.0

    # Noise in background (lower = cleaner)
    bg_region = flat[flat <= np.percentile(flat, 10)]
    bg_noise = float(np.std(bg_region)) if bg_region.size > 100 else 0.0

    return {
        "experiment": label,
        "snr_median": float(np.median(snr)),
        "snr_deep_median": float(np.median(snr[z_deep_start:])),
        "noise_bg_std": bg_noise,
        "dynamic_range": dr,
        "deep_p75": deep_p75,
        "deep_p95": deep_p95,
        "deep_mean": deep_mean,
        "depth_ratio": depth_ratio,
        "z_uniformity_cv": z_cv,
        "frac_zero": float(np.sum(flat == 0) / flat.size),
        "p99": float(np.percentile(flat, 99)),
    }


# ============================================================================
# Experiment runner
# ============================================================================


def find_best_intermediate(inter_dir):
    """Find the best intermediate to use as CLAHE input.

    Priority (deepest stage available wins): post-WBNS > post-deconv > post-reslice > post-z-correction > post-shading > post-downscale > raw.

    Filenames match the canonical stage names produced by
    ``spim_preprocessing_stages.run_pipeline``.
    """
    candidates = [
        "08_after_wbns.tif",
        "06_after_deconv3d.tif",
        "07_after_deconv_xz.tif",
        "05_after_isotropic_reslice.tif",
        "04_after_z_correction.tif",
        "03_after_shading.tif",
        "02_after_downscale_xy.tif",
        "01_after_load.tif",
    ]
    for c in candidates:
        p = os.path.join(inter_dir, c)
        if os.path.isfile(p):
            return p, c.replace(".tif", "")
    return None, None


def build_tissue_mask(img, border_px=25):
    """Rebuild tissue mask from the intermediate image."""
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_fill_holes

    smooth = ndi.gaussian_filter(img.astype(np.float32), sigma=2.0)
    mask = np.zeros(img.shape, dtype=bool)
    for zi in range(smooth.shape[0]):
        sl = smooth[zi]
        nz = sl[sl > 0]
        if nz.size < 100:
            continue
        try:
            otsu = threshold_otsu(nz)
        except ValueError:
            continue
        slice_mask = sl > (otsu * 0.03)
        slice_mask = binary_fill_holes(slice_mask)
        mask[zi] = slice_mask

    struct = ndi.generate_binary_structure(3, 2)
    mask = ndi.binary_closing(mask, structure=struct, iterations=5)
    mask = ndi.binary_dilation(mask, structure=struct, iterations=8)
    mask = binary_fill_holes(mask)

    if border_px > 0:
        mask[:, :border_px, :] = False
        mask[:, -border_px:, :] = False
        mask[:, :, :border_px] = False
        mask[:, :, -border_px:] = False

    return mask


def run_experiment(img_input, tissue_mask, params, label):
    """Run one CLAHE configuration and return (result_image, metrics)."""
    img = img_input.copy().astype(np.float32)

    # Apply tissue mask
    img[~tissue_mask] = 0

    clahe_on = params.get("clahe", True)
    clip_limit = params.get("clip_limit", 0.01)
    dual_axis = params.get("dual_axis", False)
    min_signal_pct = params.get("min_signal_pct", 15.0)
    post_smooth = params.get("post_smooth", 0.0)
    kernel_size = params.get("kernel_size", (64, 64))

    if clahe_on:
        # XZ pass first (depth equalization)
        if dual_axis:
            img = clahe_3d_stack(
                img,
                clip_limit=clip_limit,
                kernel_size=kernel_size,
                axis=1,  # XZ slices (iterate over Y)
                min_signal_pct=min_signal_pct,
            )

        # XY pass (per-plane equalization)
        img = clahe_3d_stack(
            img,
            clip_limit=clip_limit,
            kernel_size=kernel_size,
            axis=0,  # XY slices (iterate over Z)
            min_signal_pct=min_signal_pct,
        )

    if post_smooth > 0:
        img = ndi.gaussian_filter(img, sigma=post_smooth)

    # Final tissue mask cleanup
    img[~tissue_mask] = 0

    # Normalize to uint16
    tissue_vals = img[img > 0]
    if tissue_vals.size > 100:
        lo = float(np.percentile(tissue_vals, 1.0))
        hi = float(np.percentile(tissue_vals, 99.9))
    else:
        lo, hi = 0.0, 1.0
    img = np.clip(img, lo, hi) - lo
    img[img < 0] = 0
    imax = float(img.max())
    if imax > 1e-8:
        img = (img / imax * 65535).astype(np.uint16)
    else:
        img = img.astype(np.uint16)

    metrics = compute_metrics(img, label)
    return img, metrics


# ============================================================================
# Plotting
# ============================================================================


def try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("WARNING: matplotlib not available. Install with: pip install matplotlib")
        return None


def plot_slice_grid(experiments, z_idx, title, outpath):
    """Grid of one slice across all experiments."""
    plt = try_matplotlib()
    if plt is None:
        return

    n = len(experiments)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, (label, img, _) in enumerate(experiments):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if z_idx < img.shape[0]:
            sl = img[z_idx].astype(np.float64)
            vmin, vmax = np.percentile(sl, [0.5, 99.5])
            ax.imshow(sl, cmap="gray", vmin=vmin, vmax=max(vmin + 1, vmax))
        ax.set_title(label, fontsize=8, wrap=True)
        ax.axis("off")

    # Hide unused
    for i in range(len(experiments), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_z_profiles_sweep(experiments, outpath):
    """Per-slice mean intensity profiles overlaid for all experiments."""
    plt = try_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Mean intensity per slice
    ax = axes[0]
    for label, img, _ in experiments:
        means = [float(np.mean(img[z])) for z in range(img.shape[0])]
        ax.plot(means, label=label, lw=1.2)
    ax.set_xlabel("Z slice")
    ax.set_ylabel("Mean intensity")
    ax.set_title("Per-Slice Mean Intensity")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)

    # p95 per slice (signal level)
    ax = axes[1]
    for label, img, _ in experiments:
        p95s = [float(np.percentile(img[z], 95)) for z in range(img.shape[0])]
        ax.plot(p95s, label=label, lw=1.2)
    ax.set_xlabel("Z slice")
    ax.set_ylabel("p95 intensity")
    ax.set_title("Per-Slice Signal (p95)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)

    fig.suptitle("Z-Profile Comparison Across Experiments", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_snr_sweep(experiments, outpath):
    """SNR per slice overlaid for all experiments."""
    plt = try_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    for label, img, _ in experiments:
        snr, _, _ = robust_snr_per_slice(img)
        ax.plot(snr, label=label, lw=1.0)

    ax.set_xlabel("Z slice")
    ax.set_ylabel("SNR (p95 / noise_sigma)")
    ax.set_title("Per-Slice SNR Across Experiments")
    ax.axhline(3.0, color="red", ls="--", alpha=0.4, label="SNR=3 (noise-dominated)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_depth_ratio_bar(experiments, outpath):
    """Bar chart of depth uniformity ratio (deep/shallow) per experiment."""
    plt = try_matplotlib()
    if plt is None:
        return

    labels = [e[0] for e in experiments]
    ratios = [e[2]["depth_ratio"] for e in experiments]
    snrs = [e[2]["snr_median"] for e in experiments]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71" if r > 0.5 else "#e74c3c" if r < 0.3 else "#f39c12" for r in ratios]
    ax1.barh(range(len(labels)), ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel("Depth Ratio (deep_p75 / shallow_p75)")
    ax1.set_title("Depth Uniformity\n(higher = deeper nuclei better preserved)")
    ax1.axvline(1.0, color="green", ls="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="x")

    ax2.barh(range(len(labels)), snrs, color="steelblue", edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("Median SNR")
    ax2.set_title("Signal-to-Noise Ratio\n(higher = cleaner)")
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Depth vs Noise Trade-off", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CLAHE Parameter Sweep for Deep Nuclei Sensitivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--intermediates_dir",
        required=True,
        help="Path to pipeline intermediates directory (from save_intermediates=true)",
    )
    parser.add_argument("--config_json", required=True, help="Pipeline config.json")
    parser.add_argument("--outdir", required=True, help="Output directory for sweep results")
    parser.add_argument(
        "--raw_file", default=None, help="Optional raw input TIFF for SNR comparison"
    )
    parser.add_argument(
        "--save_tiffs", action="store_true", help="Save full TIFF for each experiment"
    )
    parser.add_argument(
        "--border_px", type=int, default=25, help="Tissue mask border exclusion (px)"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load config for reference
    with open(args.config_json) as f:
        config = json.load(f)

    # Find best intermediate
    inter_path, inter_stage = find_best_intermediate(args.intermediates_dir)
    if inter_path is None:
        print("ERROR: No suitable intermediate found in", args.intermediates_dir)
        print("Available files:")
        for f_name in sorted(os.listdir(args.intermediates_dir)):
            print(f"  {f_name}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("CLAHE PARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Intermediate: {inter_path} ({inter_stage})")

    t0 = time.time()
    img_input = tifffile.imread(inter_path).astype(np.float32)
    print(f"Loaded: shape={img_input.shape}, range=[{img_input.min():.0f}, {img_input.max():.0f}]")
    print(f"  ({time.time()-t0:.1f}s)")

    # Build tissue mask
    print("Building tissue mask...")
    t0 = time.time()
    tissue_mask = build_tissue_mask(img_input, border_px=args.border_px)
    pct = 100.0 * np.count_nonzero(tissue_mask) / tissue_mask.size
    print(f"  Mask: {pct:.1f}% tissue ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Define parameter sweep
    # -----------------------------------------------------------------------
    # The key trade-off: clip_limit and dual_axis control sensitivity vs noise.
    # We test a focused grid targeting deep-nuclei recovery.
    sweep_params = [
        # Baseline: no CLAHE (your current config)
        {"clahe": False, "label": "00_no_clahe"},
        # Conservative CLAHE (XY only)
        {"clahe": True, "clip_limit": 0.005, "dual_axis": False, "min_signal_pct": 10.0,
         "post_smooth": 0.8, "label": "01_clip005_xy"},
        # Moderate CLAHE (XY only)
        {"clahe": True, "clip_limit": 0.008, "dual_axis": False, "min_signal_pct": 10.0,
         "post_smooth": 0.8, "label": "02_clip008_xy"},
        # Standard CLAHE (XY only)
        {"clahe": True, "clip_limit": 0.01, "dual_axis": False, "min_signal_pct": 10.0,
         "post_smooth": 0.8, "label": "03_clip010_xy"},
        # Higher CLAHE (XY only)
        {"clahe": True, "clip_limit": 0.015, "dual_axis": False, "min_signal_pct": 10.0,
         "post_smooth": 1.0, "label": "04_clip015_xy"},
        # Conservative CLAHE with DUAL AXIS (XZ + XY) — best for depth equalization
        {"clahe": True, "clip_limit": 0.005, "dual_axis": True, "min_signal_pct": 8.0,
         "post_smooth": 0.8, "label": "05_clip005_dual"},
        # Moderate dual axis  — RECOMMENDED START
        {"clahe": True, "clip_limit": 0.008, "dual_axis": True, "min_signal_pct": 8.0,
         "post_smooth": 0.8, "label": "06_clip008_dual"},
        # Standard dual axis
        {"clahe": True, "clip_limit": 0.01, "dual_axis": True, "min_signal_pct": 8.0,
         "post_smooth": 1.0, "label": "07_clip010_dual"},
        # Higher dual axis
        {"clahe": True, "clip_limit": 0.015, "dual_axis": True, "min_signal_pct": 5.0,
         "post_smooth": 1.0, "label": "08_clip015_dual"},
        # Aggressive for maximum deep sensitivity (risk of more noise)
        {"clahe": True, "clip_limit": 0.02, "dual_axis": True, "min_signal_pct": 5.0,
         "post_smooth": 1.2, "label": "09_clip020_dual"},
        # Lower signal gate (recover dimmer deep slices)
        {"clahe": True, "clip_limit": 0.008, "dual_axis": True, "min_signal_pct": 3.0,
         "post_smooth": 0.8, "label": "10_clip008_dual_lowgate"},
    ]

    # -----------------------------------------------------------------------
    # Run sweep
    # -----------------------------------------------------------------------
    experiments = []  # (label, result_img, metrics)
    all_metrics = []

    for i, params in enumerate(sweep_params):
        label = params.pop("label")
        print(f"\n--- Experiment {i:02d}: {label} ---")
        print(f"  Params: {params}")
        t0 = time.time()
        result_img, metrics = run_experiment(img_input, tissue_mask, params, label)
        elapsed = time.time() - t0
        print(f"  Done ({elapsed:.1f}s) | SNR={metrics['snr_median']:.1f}, "
              f"depth_ratio={metrics['depth_ratio']:.3f}, "
              f"deep_p75={metrics['deep_p75']:.0f}, "
              f"noise_bg={metrics['noise_bg_std']:.1f}")
        experiments.append((label, result_img, metrics))
        all_metrics.append(metrics)
        params["label"] = label  # restore for saving

        if args.save_tiffs:
            exp_dir = os.path.join(args.outdir, label)
            os.makedirs(exp_dir, exist_ok=True)
            tifffile.imwrite(os.path.join(exp_dir, f"{label}.tif"), result_img)

    # -----------------------------------------------------------------------
    # Save sweep config
    # -----------------------------------------------------------------------
    sweep_config_path = os.path.join(args.outdir, "sweep_config.json")
    with open(sweep_config_path, "w") as f:
        json.dump({"input_intermediate": inter_path, "experiments": sweep_params}, f, indent=2)
    print(f"\nSaved: {sweep_config_path}")

    # -----------------------------------------------------------------------
    # Save metrics CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(args.outdir, "sweep_metrics.csv")
    fields = list(all_metrics[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)
    print(f"Saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Generating comparison plots...")
    print(f"{'='*60}")

    mid_z = img_input.shape[0] // 2
    deep_z = int(0.75 * img_input.shape[0])
    shallow_z = int(0.25 * img_input.shape[0])

    plot_slice_grid(experiments, mid_z,
                    f"Mid-Z Slice (z={mid_z}) — All Experiments",
                    os.path.join(args.outdir, "panel_sweep_midZ.png"))
    plot_slice_grid(experiments, deep_z,
                    f"Deep-Z Slice (z={deep_z}) — Deep Nuclei Sensitivity",
                    os.path.join(args.outdir, "panel_sweep_deepZ.png"))
    plot_slice_grid(experiments, shallow_z,
                    f"Shallow-Z Slice (z={shallow_z}) — Noise Check",
                    os.path.join(args.outdir, "panel_sweep_shallowZ.png"))
    plot_z_profiles_sweep(experiments,
                          os.path.join(args.outdir, "panel_sweep_z_profiles.png"))
    plot_snr_sweep(experiments,
                   os.path.join(args.outdir, "panel_sweep_snr.png"))
    plot_depth_ratio_bar(experiments,
                         os.path.join(args.outdir, "panel_sweep_depth_vs_noise.png"))

    # -----------------------------------------------------------------------
    # Recommendation report
    # -----------------------------------------------------------------------
    report_path = os.path.join(args.outdir, "sweep_report.txt")

    # Score each experiment: balance depth_ratio (want high) with noise (want low)
    # Composite score = depth_ratio * snr_median / (1 + noise_bg_std)
    for m in all_metrics:
        m["composite_score"] = (
            m["depth_ratio"] * m["snr_median"] / (1 + m["noise_bg_std"])
        )

    ranked = sorted(all_metrics, key=lambda m: m["composite_score"], reverse=True)

    with open(report_path, "w") as f:
        f.write("CLAHE Parameter Sweep Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input intermediate: {inter_path}\n")
        f.write(f"Shape: {img_input.shape}\n\n")

        f.write("RANKING (best composite score first)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<5} {'Experiment':<30} {'Score':<8} {'Depth':<8} {'SNR':<8} {'Noise':<8}\n")
        f.write("-" * 60 + "\n")
        for i, m in enumerate(ranked):
            f.write(
                f"{i+1:<5} {m['experiment']:<30} {m['composite_score']:<8.2f} "
                f"{m['depth_ratio']:<8.3f} {m['snr_median']:<8.1f} {m['noise_bg_std']:<8.1f}\n"
            )

        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 60 + "\n\n")

        best = ranked[0]
        f.write(f"Top pick: {best['experiment']}\n")
        f.write(f"  Composite score: {best['composite_score']:.2f}\n")
        f.write(f"  Depth ratio:     {best['depth_ratio']:.3f} (1.0 = perfect uniformity)\n")
        f.write(f"  SNR:             {best['snr_median']:.1f}\n")
        f.write(f"  Background noise: {best['noise_bg_std']:.1f}\n\n")

        # Find best for depth specifically
        best_depth = max(all_metrics, key=lambda m: m["depth_ratio"])
        f.write(f"Best for deep nuclei: {best_depth['experiment']}\n")
        f.write(f"  Depth ratio: {best_depth['depth_ratio']:.3f}\n\n")

        # Find cleanest
        best_clean = min(all_metrics, key=lambda m: m["noise_bg_std"])
        f.write(f"Cleanest background: {best_clean['experiment']}\n")
        f.write(f"  Noise: {best_clean['noise_bg_std']:.1f}\n\n")

        f.write("HOW TO APPLY THE WINNER\n")
        f.write("-" * 40 + "\n")
        f.write("Update config.json with:\n\n")

        # Find the winning params
        winner_params = None
        for p in sweep_params:
            if p["label"] == best["experiment"]:
                winner_params = p
                break

        if winner_params:
            f.write('  "correction_flags": {\n')
            f.write(f'    "no_clahe": {str(not winner_params.get("clahe", True)).lower()},\n')
            f.write(f'    "no_clahe_xy": {str(not winner_params.get("dual_axis", False)).lower()},\n')
            f.write('    "no_z_correction": false,\n')
            f.write('    "no_shading": false\n')
            f.write('  },\n')
            f.write('  "postprocessing": {\n')
            f.write(f'    "clahe_clip_limit": {winner_params.get("clip_limit", 0.01)},\n')
            f.write(f'    "clahe_post_smooth": {winner_params.get("post_smooth", 0.8)},\n')
            f.write(f'    "clahe_dual_axis": {str(winner_params.get("dual_axis", False)).lower()}\n')
            f.write('  }\n')

    print(f"\nSaved: {report_path}")

    # Print summary to terminal
    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"\nTop 3 experiments:")
    for i, m in enumerate(ranked[:3]):
        print(f"  {i+1}. {m['experiment']} — score={m['composite_score']:.2f}, "
              f"depth={m['depth_ratio']:.3f}, SNR={m['snr_median']:.1f}")
    print(f"\nFull results: {args.outdir}")
    print(f"Report:       {report_path}")
    print(f"Metrics:      {csv_path}")


if __name__ == "__main__":
    main()
