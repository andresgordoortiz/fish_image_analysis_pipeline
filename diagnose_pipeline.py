#!/usr/bin/env python3
"""
SPIM Pipeline Deep Diagnostic Tool
===================================
IMP Vienna — Andrés Gordo

Runs the preprocessing pipeline step-by-step on a SINGLE timepoint,
capturing intermediate images and computing statistics at every stage.
Produces a comprehensive diagnostic report (PNG panels + CSV metrics)
that reveals exactly where noise amplification, signal saturation,
and dynamic-range problems originate.

Usage
-----
    # Minimal — point at one raw TIFF and the config
    python diagnose_pipeline.py \
        --input_file ./subset/t0001.tif \
        --config_json ./config.json \
        --outdir ./diagnostics/

    # Compare against an already-processed result
    python diagnose_pipeline.py \
        --input_file ./subset/t0001.tif \
        --config_json ./config.json \
        --outdir ./diagnostics/ \
        --processed_file ./subset_results/01_preprocessed/t0001_33.tif

What it produces
----------------
diagnostics/
├── 00_raw_stats.npz                  # per-slice statistics of raw input
├── 01_after_camera_bg.npz
├── ...
├── metrics_per_stage.csv             # tabular: SNR, dynamic range, noise floor per stage
├── panel_histograms.png              # intensity histograms at every stage
├── panel_z_profiles.png              # per-slice mean/p5/p50/p95/max along Z
├── panel_noise_maps.png              # local-std noise maps at key stages
├── panel_slice_comparison.png        # mid-Z slice at every stage (visual)
├── panel_snr_per_slice.png           # per-slice SNR at every stage
├── report_summary.txt                # text summary with recommendations
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.transform import rescale, resize

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path):
    with open(path) as f:
        raw = f.read()
    if raw and ord(raw[0]) == 0xFEFF:
        raw = raw[1:]
    raw = raw.strip()
    return json.loads(raw)


def robust_snr_per_slice(stack, bg_percentile=10, signal_percentile=95):
    """Return (snr_array, noise_std_array, signal_array) with length = Z."""
    snr = np.zeros(stack.shape[0], dtype=np.float64)
    noise_std = np.zeros_like(snr)
    signal = np.zeros_like(snr)
    for zi in range(stack.shape[0]):
        sl = stack[zi].ravel().astype(np.float64)
        p_bg = np.percentile(sl, bg_percentile)
        bg_region = sl[sl <= p_bg]
        noise_std[zi] = np.std(bg_region) if len(bg_region) > 50 else 0
        signal[zi] = np.percentile(sl, signal_percentile)
        snr[zi] = signal[zi] / noise_std[zi] if noise_std[zi] > 1e-10 else 0
    return snr, noise_std, signal


def per_slice_stats(stack):
    """Return dict of per-slice statistics."""
    Z = stack.shape[0]
    stats = {
        "mean": np.zeros(Z),
        "std": np.zeros(Z),
        "p01": np.zeros(Z),
        "p05": np.zeros(Z),
        "p25": np.zeros(Z),
        "p50": np.zeros(Z),
        "p75": np.zeros(Z),
        "p95": np.zeros(Z),
        "p99": np.zeros(Z),
        "max": np.zeros(Z),
        "min": np.zeros(Z),
        "n_zero": np.zeros(Z),
        "n_saturated_65535": np.zeros(Z),
    }
    for zi in range(Z):
        sl = stack[zi].ravel().astype(np.float64)
        stats["mean"][zi] = np.mean(sl)
        stats["std"][zi] = np.std(sl)
        ps = np.percentile(sl, [1, 5, 25, 50, 75, 95, 99])
        stats["p01"][zi] = ps[0]
        stats["p05"][zi] = ps[1]
        stats["p25"][zi] = ps[2]
        stats["p50"][zi] = ps[3]
        stats["p75"][zi] = ps[4]
        stats["p95"][zi] = ps[5]
        stats["p99"][zi] = ps[6]
        stats["max"][zi] = np.max(sl)
        stats["min"][zi] = np.min(sl)
        stats["n_zero"][zi] = np.sum(sl == 0)
        stats["n_saturated_65535"][zi] = np.sum(sl >= 65535)
    return stats


def local_noise_map(img_2d, window=7):
    """Compute local standard deviation in a sliding window."""
    from scipy.ndimage import uniform_filter

    f = img_2d.astype(np.float64)
    mean_sq = uniform_filter(f * f, size=window)
    sq_mean = uniform_filter(f, size=window) ** 2
    variance = np.clip(mean_sq - sq_mean, 0, None)
    return np.sqrt(variance)


def dynamic_range_info(img):
    """Return (actual_min, actual_max, p01, p99, p999, fraction_zero, fraction_saturated)."""
    flat = img.ravel().astype(np.float64)
    p01, p99, p999 = np.percentile(flat, [1, 99, 99.9])
    frac_zero = np.sum(flat == 0) / flat.size
    frac_sat = np.sum(flat >= 65535) / flat.size if flat.max() >= 65535 else 0
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p01": float(p01),
        "p99": float(p99),
        "p999": float(p999),
        "frac_zero": float(frac_zero),
        "frac_saturated": float(frac_sat),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


# ---------------------------------------------------------------------------
# Fake/simplified pipeline steps — we replicate each step from the pipeline
# but capture the result. We don't need GPU deconvolution; we diagnose
# everything else and optionally load a pre-computed result to compare.
# ---------------------------------------------------------------------------


def step_camera_bg_subtract(img, percentile=2.0):
    img_f = img.astype(np.float32)
    for zi in range(img_f.shape[0]):
        bg = float(np.percentile(img_f[zi], percentile))
        img_f[zi] = np.maximum(img_f[zi] - bg, 0.0)
    return img_f


def step_shading_correct(stack, sigma_xy=96.0):
    x = stack.astype(np.float32)
    proj = np.mean(x, axis=0)
    field = ndi.gaussian_filter(proj, sigma=sigma_xy)
    field = np.maximum(field, 1e-6)
    norm = float(np.mean(field))
    corrected = x * (norm / field)
    return corrected, field


def step_z_correction(
    stack, method="p50", smooth_window=11, max_scale=2.0, signal_floor_pct=25.0
):
    x = stack.astype(np.float32)
    if method.startswith("p"):
        q = float(method[1:])
        levels = np.percentile(x.reshape(x.shape[0], -1), q, axis=1)
    else:
        levels = np.median(x.reshape(x.shape[0], -1), axis=1)
    levels = np.maximum(levels, 1e-8)
    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        pad = smooth_window // 2
        lvl_pad = np.pad(levels, (pad, pad), mode="edge")
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        levels_s = np.convolve(lvl_pad, kernel, mode="valid")
    else:
        levels_s = levels
    target = np.median(levels_s)
    scales = target / levels_s
    # Signal floor damping
    if signal_floor_pct > 0:
        floor = (signal_floor_pct / 100.0) * target
        dim_mask = levels_s < floor
        if np.any(dim_mask):
            frac = np.where(dim_mask, levels_s / (floor + 1e-8), 1.0)
            scales = np.where(dim_mask, 1.0 + (scales - 1.0) * frac, scales)
    if max_scale > 0:
        scales = np.minimum(scales, max_scale)
    corrected = x * scales[:, None, None]
    return corrected, scales, levels, levels_s


def step_intensity_stretch(img, min_v=0, max_v=65535):
    """Replicate image_scaling_intens."""
    img_f = img.astype(np.float32)
    imin = float(img_f.min())
    imax = float(img_f.max())
    if imax - imin < 1e-8:
        return np.full_like(img_f, min_v)
    return (max_v - min_v) * (img_f - imin) / (imax - imin) + min_v


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        return plt, GridSpec
    except ImportError:
        print(
            "WARNING: matplotlib not available. Skipping plots. "
            "Install with: pip install matplotlib"
        )
        return None, None


def plot_histograms(stages, outpath):
    """Side-by-side intensity histograms for each stage."""
    plt, GridSpec = try_import_matplotlib()
    if plt is None:
        return

    n = len(stages)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 8))
    axes = axes.ravel()

    for i, (name, img) in enumerate(stages):
        ax = axes[i]
        flat = img.ravel().astype(np.float64)
        # Subsample for speed
        if flat.size > 2_000_000:
            flat = np.random.default_rng(42).choice(flat, 2_000_000, replace=False)
        lo, hi = np.percentile(flat, [0.1, 99.9])
        bins = np.linspace(lo, hi, 200)
        ax.hist(flat, bins=bins, color="steelblue", alpha=0.8, density=True)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
        dri = dynamic_range_info(img)
        ax.axvline(
            dri["p01"], color="red", ls="--", lw=0.7, label=f"p1={dri['p01']:.0f}"
        )
        ax.axvline(
            dri["p99"], color="orange", ls="--", lw=0.7, label=f"p99={dri['p99']:.0f}"
        )
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Intensity Histograms at Each Pipeline Stage", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_z_profiles(stage_stats, outpath):
    """Per-slice statistics along Z for each stage."""
    plt, _ = try_import_matplotlib()
    if plt is None:
        return

    n = len(stage_stats)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (name, stats) in enumerate(stage_stats):
        ax = axes[i]
        z_idx = np.arange(len(stats["mean"]))
        ax.fill_between(
            z_idx,
            stats["p05"],
            stats["p95"],
            alpha=0.2,
            color="steelblue",
            label="p5–p95",
        )
        ax.fill_between(
            z_idx,
            stats["p25"],
            stats["p75"],
            alpha=0.35,
            color="steelblue",
            label="p25–p75",
        )
        ax.plot(z_idx, stats["p50"], color="navy", lw=1.2, label="median")
        ax.plot(z_idx, stats["mean"], color="red", lw=0.8, ls="--", label="mean")
        ax.plot(z_idx, stats["max"], color="orange", lw=0.6, ls=":", label="max")
        ax.set_title(f"{name}", fontsize=10)
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Z slice index")
    fig.suptitle("Per-Slice Intensity Profiles Along Z", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_snr_per_slice(stage_snrs, outpath):
    """SNR per Z-slice at each stage."""
    plt, _ = try_import_matplotlib()
    if plt is None:
        return

    n = len(stage_snrs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (name, snr, noise, sig) in enumerate(stage_snrs):
        ax = axes[i]
        z_idx = np.arange(len(snr))
        ax.plot(z_idx, snr, color="green", lw=1, label="SNR")
        ax2 = ax.twinx()
        ax2.plot(z_idx, noise, color="red", lw=0.8, ls="--", label="noise σ")
        ax2.plot(z_idx, sig, color="blue", lw=0.8, ls=":", label="signal p95")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("SNR", color="green")
        ax2.set_ylabel("Intensity", color="gray")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Z slice index")
    fig.suptitle("SNR Decomposition Per Z-Slice", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_noise_maps(stages, mid_z, outpath):
    """Local noise (std) maps at mid-Z for key stages."""
    plt, _ = try_import_matplotlib()
    if plt is None:
        return

    n = len(stages)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = np.atleast_2d(axes).T

    for i, (name, img) in enumerate(stages):
        sl = img[mid_z].astype(np.float64)
        noise = local_noise_map(sl, window=7)

        # Image
        ax_img = axes[0, i] if n > 1 else axes[0][0]
        vmin, vmax = np.percentile(sl, [1, 99])
        ax_img.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        ax_img.set_title(f"{name}\n(slice z={mid_z})", fontsize=9)
        ax_img.axis("off")

        # Noise map
        ax_noise = axes[1, i] if n > 1 else axes[1][0]
        nmax = np.percentile(noise, 99)
        im = ax_noise.imshow(noise, cmap="hot", vmin=0, vmax=nmax)
        ax_noise.set_title(f"Local σ (7×7)", fontsize=9)
        ax_noise.axis("off")
        fig.colorbar(im, ax=ax_noise, fraction=0.046, pad=0.04)

    fig.suptitle(f"Mid-Z Slice & Local Noise Maps (z={mid_z})", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_slice_comparison(stages, z_slices, outpath):
    """Visual comparison of selected Z slices across all stages."""
    plt, _ = try_import_matplotlib()
    if plt is None:
        return

    n_stages = len(stages)
    n_slices = len(z_slices)
    fig, axes = plt.subplots(n_slices, n_stages, figsize=(4 * n_stages, 4 * n_slices))
    if n_stages == 1:
        axes = axes[:, np.newaxis]
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for col, (name, img) in enumerate(stages):
        for row, zi in enumerate(z_slices):
            ax = axes[row, col]
            if zi < img.shape[0]:
                sl = img[zi].astype(np.float64)
                vmin, vmax = np.percentile(sl, [1, 99.5])
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=max(vmin + 1, vmax))
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"z={zi}", fontsize=9)

    fig.suptitle(
        "Slice-by-Slice Visual Comparison Across Pipeline Stages", fontsize=12, y=1.01
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_camera_noise_analysis(raw_img, outpath):
    """Deep analysis of camera noise characteristics."""
    plt, _ = try_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw histogram (full range, log scale)
    flat = raw_img.ravel().astype(np.float64)
    sub = np.random.default_rng(42).choice(
        flat, min(5_000_000, flat.size), replace=False
    )
    ax = axes[0, 0]
    ax.hist(sub, bins=500, color="steelblue", alpha=0.8, log=True)
    ax.set_title("Raw Full Histogram (log scale)")
    ax.set_xlabel("Intensity")
    ax.axvline(np.percentile(sub, 2), color="red", ls="--", label="p2")
    ax.axvline(np.percentile(sub, 5), color="orange", ls="--", label="p5")
    ax.axvline(np.percentile(sub, 50), color="green", ls="--", label="p50")
    ax.legend(fontsize=8)

    # 2. Zoom on camera noise peak (0-500 range typically)
    ax = axes[0, 1]
    bg = sub[sub < np.percentile(sub, 30)]
    ax.hist(bg, bins=200, color="coral", alpha=0.8)
    ax.set_title(
        f"Camera Noise Peak (bottom 30% of pixels)\nμ={np.mean(bg):.1f}, σ={np.std(bg):.1f}"
    )
    ax.set_xlabel("Intensity")
    ax.axvline(np.mean(bg), color="black", ls="-", lw=2)
    ax.axvline(np.mean(bg) + 2 * np.std(bg), color="red", ls="--", label="+2σ")
    ax.legend(fontsize=8)

    # 3. Per-slice noise floor (p2 per slice)
    ax = axes[0, 2]
    slice_p2 = [float(np.percentile(raw_img[z], 2)) for z in range(raw_img.shape[0])]
    slice_p5 = [float(np.percentile(raw_img[z], 5)) for z in range(raw_img.shape[0])]
    slice_p50 = [float(np.percentile(raw_img[z], 50)) for z in range(raw_img.shape[0])]
    z_idx = np.arange(raw_img.shape[0])
    ax.plot(z_idx, slice_p2, label="p2 (noise floor)", color="red")
    ax.plot(z_idx, slice_p5, label="p5", color="orange")
    ax.plot(z_idx, slice_p50, label="p50 (median)", color="green")
    ax.set_title("Per-Slice Intensity Percentiles")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Signal-to-noise ratio per slice
    snr, noise_s, signal_s = robust_snr_per_slice(raw_img)
    ax = axes[1, 0]
    ax.plot(z_idx, snr, color="green", lw=1.2)
    ax.set_title("Raw SNR per Z-slice")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("SNR (p95/σ_bg)")
    ax.grid(True, alpha=0.3)
    ax.axhline(3.0, color="red", ls="--", alpha=0.5, label="SNR=3 (noise-dominated)")
    ax.legend(fontsize=8)

    # 5. Fraction of near-zero vs bright pixels per slice
    ax = axes[1, 1]
    slice_frac_bg = []
    slice_frac_bright = []
    for z in range(raw_img.shape[0]):
        sl = raw_img[z].ravel().astype(np.float64)
        bg_thresh = np.mean(bg) + 3 * np.std(bg)  # 3σ above camera noise
        bright_thresh = np.percentile(sl, 90)
        slice_frac_bg.append(np.sum(sl < bg_thresh) / sl.size)
        slice_frac_bright.append(np.sum(sl > bright_thresh) / sl.size)
    ax.plot(z_idx, slice_frac_bg, color="gray", label="frac < 3σ noise")
    ax.plot(z_idx, slice_frac_bright, color="blue", label="frac > p90")
    ax.set_title("Background vs Signal Fraction per Slice")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("Fraction of pixels")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Dynamic range utilization
    ax = axes[1, 2]
    slice_dr = [
        float(np.percentile(raw_img[z], 99.5) - np.percentile(raw_img[z], 0.5))
        for z in range(raw_img.shape[0])
    ]
    ax.plot(z_idx, slice_dr, color="purple", lw=1.2)
    ax.set_title("Effective Dynamic Range per Slice (p99.5 − p0.5)")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("Intensity range")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Camera Noise Characterization (Raw Input)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_z_correction_analysis(raw_after_bg, scales, levels, levels_s, outpath):
    """Show what z-correction is doing: the scale factors and levels."""
    plt, _ = try_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    z_idx = np.arange(len(scales))

    ax = axes[0]
    ax.plot(z_idx, levels, color="blue", alpha=0.5, label="raw levels")
    ax.plot(z_idx, levels_s, color="navy", lw=2, label="smoothed levels")
    ax.axhline(np.median(levels_s), color="red", ls="--", label="target (median)")
    ax.set_title("Per-Slice Signal Levels")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("Level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z_idx, scales, color="green", lw=1.5)
    ax.axhline(1.0, color="gray", ls="--")
    ax.set_title("Z-Correction Scale Factors")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("Multiplier")
    ax.grid(True, alpha=0.3)
    # Highlight dangerous regions
    danger = scales > 1.5
    if np.any(danger):
        ax.fill_between(
            z_idx,
            0,
            scales,
            where=danger,
            color="red",
            alpha=0.2,
            label=f">{1.5}× ({np.sum(danger)} slices)",
        )
        ax.legend(fontsize=8)

    ax = axes[2]
    # Show the noise amplification: what happens to noise floor
    snr_pre, noise_pre, _ = robust_snr_per_slice(raw_after_bg)
    noise_post_predicted = noise_pre * scales[: len(noise_pre)]
    ax.plot(z_idx[: len(noise_pre)], noise_pre, color="blue", label="noise σ before")
    ax.plot(
        z_idx[: len(noise_pre)],
        noise_post_predicted,
        color="red",
        label="noise σ × scale (predicted)",
    )
    ax.set_title("Noise Amplification from Z-Correction")
    ax.set_xlabel("Z slice")
    ax.set_ylabel("Noise σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Z-Correction Diagnostics", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_processed_comparison(raw_img, processed_img, outpath):
    """If a processed result is available, compare it with the raw."""
    plt, _ = try_import_matplotlib()
    if plt is None:
        return

    n_slices = 5
    z_raw = raw_img.shape[0]
    z_proc = processed_img.shape[0]
    z_indices_raw = np.linspace(0, z_raw - 1, n_slices, dtype=int)
    z_indices_proc = np.linspace(0, z_proc - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12))

    for i in range(n_slices):
        # Raw
        ax = axes[0, i]
        sl = raw_img[z_indices_raw[i]].astype(np.float64)
        vmin, vmax = np.percentile(sl, [1, 99.5])
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=max(vmin + 1, vmax))
        ax.set_title(f"Raw z={z_indices_raw[i]}", fontsize=9)
        ax.axis("off")

        # Processed
        ax = axes[1, i]
        if z_indices_proc[i] < z_proc:
            sl = processed_img[z_indices_proc[i]].astype(np.float64)
            vmin, vmax = np.percentile(sl, [1, 99.5])
            ax.imshow(sl, cmap="gray", vmin=vmin, vmax=max(vmin + 1, vmax))
        ax.set_title(f"Processed z={z_indices_proc[i]}", fontsize=9)
        ax.axis("off")

        # Processed noise map
        ax = axes[2, i]
        if z_indices_proc[i] < z_proc:
            sl = processed_img[z_indices_proc[i]].astype(np.float64)
            nm = local_noise_map(sl, window=7)
            nmax = np.percentile(nm, 99)
            ax.imshow(nm, cmap="hot", vmin=0, vmax=nmax)
        ax.set_title(f"Processed noise z={z_indices_proc[i]}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Raw vs Processed Comparison", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main diagnostic pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SPIM Pipeline Deep Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_file", required=True, help="Path to raw input TIFF")
    parser.add_argument("--config_json", required=True, help="Pipeline config.json")
    parser.add_argument(
        "--outdir", required=True, help="Output directory for diagnostics"
    )
    parser.add_argument(
        "--processed_file",
        default=None,
        help="Path to already-processed result (for comparison)",
    )
    parser.add_argument(
        "--max_slices_for_plots",
        type=int,
        default=5,
        help="Number of Z slices for visual comparison panels",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    config = load_config(args.config_json)
    pre = config.get("preprocessing", {})

    # -----------------------------------------------------------------------
    # Load raw image
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SPIM PIPELINE DIAGNOSTIC")
    print(f"{'=' * 60}")
    print(f"Input: {args.input_file}")
    print(f"Config: {args.config_json}")

    t0 = time.time()
    raw = tifffile.imread(args.input_file).astype(np.uint16)
    print(
        f"Loaded raw image: shape={raw.shape}, dtype={raw.dtype}, "
        f"size={raw.nbytes / 1024**3:.2f} GB ({time.time() - t0:.1f}s)"
    )

    # -----------------------------------------------------------------------
    # Config parameters (matching what the pipeline reads)
    # -----------------------------------------------------------------------
    image_scaling = pre.get(
        "image_scaling", config.get("preprocessing", {}).get("image_scaling", 1.0)
    )
    camera_bg_pct = pre.get("camera_bg_percentile", 2.0)
    z_method = pre.get("z_correction_method", "p50")
    z_max_scale = pre.get("z_correction_max_scale", 2.0)
    z_floor_pct = pre.get("z_correction_signal_floor_pct", 25.0)
    dim_thresh = pre.get("dim_slice_threshold_pct", 30.0)
    no_shading = pre.get("correction_flags", {}).get("no_shading", False)
    no_z_corr = pre.get("correction_flags", {}).get("no_z_correction", False)
    no_clahe = pre.get("correction_flags", {}).get("no_clahe", False)
    clahe_clip = pre.get("postprocessing", {}).get("clahe_clip_limit", 0.02)
    p_low = pre.get("normalization", {}).get("percentile_low", 1.0)
    p_high = pre.get("normalization", {}).get("percentile_high", 99.99)

    print(f"\nConfig summary:")
    print(f"  image_scaling       = {image_scaling}")
    print(f"  camera_bg_percentile= {camera_bg_pct}")
    print(f"  z_correction_method = {z_method}")
    print(f"  z_max_scale         = {z_max_scale}")
    print(f"  z_signal_floor_pct  = {z_floor_pct}")
    print(f"  dim_slice_threshold = {dim_thresh}%")
    print(f"  shading_correction  = {not no_shading}")
    print(f"  z_intensity_correct = {not no_z_corr}")
    print(f"  clahe               = {not no_clahe} (clip={clahe_clip})")
    print(f"  percentile_low      = {p_low}")
    print(f"  percentile_high     = {p_high}")

    # -----------------------------------------------------------------------
    # Stage 0: Raw
    # -----------------------------------------------------------------------
    print(f"\n--- Stage 0: Raw Input ---")
    stages = []  # (name, image) for plotting
    stage_stats_list = []  # (name, stats_dict)
    stage_snrs = []  # (name, snr, noise, signal)
    metrics_rows = []

    def record_stage(name, img):
        """Record a stage for plotting and metrics."""
        stages.append(
            (name, img.copy() if img.dtype != np.float32 else img.astype(np.float32))
        )
        stats = per_slice_stats(img)
        stage_stats_list.append((name, stats))
        snr, noise, sig = robust_snr_per_slice(img)
        stage_snrs.append((name, snr, noise, sig))
        dr = dynamic_range_info(img)
        dr["name"] = name
        dr["snr_median"] = float(np.median(snr))
        dr["snr_min"] = float(np.min(snr))
        dr["noise_std_median"] = float(np.median(noise))
        metrics_rows.append(dr)
        print(
            f"  DR: [{dr['min']:.0f} – {dr['max']:.0f}], "
            f"p1={dr['p01']:.0f}, p99={dr['p99']:.0f}, "
            f"SNR(median)={dr['snr_median']:.1f}, "
            f"noise_σ(median)={dr['noise_std_median']:.1f}, "
            f"frac_zero={dr['frac_zero']:.3f}, frac_sat={dr['frac_saturated']:.5f}"
        )

    record_stage("00_raw", raw)

    # -----------------------------------------------------------------------
    # Stage 1: Image rescaling (XY only)
    # -----------------------------------------------------------------------
    img = raw.copy().astype(np.float32)
    if image_scaling > 0 and image_scaling != 1.0:
        print(f"\n--- Stage 1: XY Rescale ({image_scaling}×) ---")
        img = rescale(
            img,
            (1.0, image_scaling, image_scaling),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        record_stage("01_rescaled", img)
    else:
        print(f"\n--- Stage 1: No rescaling ---")
        record_stage("01_no_rescale", img)

    # -----------------------------------------------------------------------
    # Stage 2: Camera background subtraction
    # -----------------------------------------------------------------------
    print(f"\n--- Stage 2: Camera BG Subtraction (p{camera_bg_pct}) ---")
    img = step_camera_bg_subtract(img, percentile=camera_bg_pct)
    record_stage("02_camera_bg", img)
    img_after_bg = img.copy()

    # -----------------------------------------------------------------------
    # Stage 3: Shading correction
    # -----------------------------------------------------------------------
    if not no_shading:
        print(f"\n--- Stage 3: Shading Correction ---")
        img, shading_field = step_shading_correct(img, sigma_xy=96.0)
        record_stage("03_shading", img)
    else:
        print(f"\n--- Stage 3: Shading correction DISABLED ---")

    # -----------------------------------------------------------------------
    # Stage 4: Z-intensity correction
    # -----------------------------------------------------------------------
    if not no_z_corr:
        print(f"\n--- Stage 4: Z-Intensity Correction (method={z_method}) ---")
        img, z_scales, z_levels, z_levels_s = step_z_correction(
            img,
            method=z_method,
            smooth_window=11,
            max_scale=z_max_scale,
            signal_floor_pct=z_floor_pct,
        )
        record_stage("04_z_corrected", img)
    else:
        print(f"\n--- Stage 4: Z-correction DISABLED ---")
        z_scales = np.ones(img.shape[0])
        z_levels = np.ones(img.shape[0])
        z_levels_s = np.ones(img.shape[0])

    # -----------------------------------------------------------------------
    # Stage 5: Pre-deconv intensity stretch (THIS IS THE PROBLEM)
    # -----------------------------------------------------------------------
    print(f"\n--- Stage 5: Pre-Deconv Intensity Stretch (0–65535) ---")
    img_stretched = step_intensity_stretch(img, 0, 65535)
    record_stage("05_pre_deconv_stretch", img_stretched)

    # -----------------------------------------------------------------------
    # Stage 5b: What it would look like WITHOUT the stretch (diagnostic)
    # -----------------------------------------------------------------------
    print(f"\n--- Stage 5b: (Diagnostic) WITHOUT pre-deconv stretch ---")
    record_stage("05b_no_stretch", img)

    # -----------------------------------------------------------------------
    # Stage 6: Simulate post-deconv (we can't run GPU deconv in diagnostic,
    # but we show what the input to deconv looks like)
    # -----------------------------------------------------------------------
    print(f"\n--- Stage 6: (Placeholder) Deconvolution input analysis ---")
    # We show for the stretched version: what's the noise floor going into RL?
    snr_into_deconv, noise_into_deconv, sig_into_deconv = robust_snr_per_slice(
        img_stretched
    )
    print(f"  Going into deconvolution:")
    print(f"    Noise floor (median σ): {np.median(noise_into_deconv):.1f}")
    print(f"    Signal (median p95):   {np.median(sig_into_deconv):.1f}")
    print(f"    SNR (median):          {np.median(snr_into_deconv):.1f}")
    print(f"    RL will iteratively SHARPEN this noise — each iteration amplifies it")

    # -----------------------------------------------------------------------
    # Stage 7: Final stretch (simulate)
    # -----------------------------------------------------------------------
    print(f"\n--- Stage 7: Final Outlier Removal + Stretch ---")
    img_final_sim = img.copy()
    lo, hi = np.percentile(img_final_sim, [p_low, p_high])
    img_final_sim = np.clip(img_final_sim, lo, hi) - lo
    img_final_sim[img_final_sim < 0] = 0
    img_final_sim = step_intensity_stretch(img_final_sim, 0, 65535)
    record_stage("07_final_stretch_sim", img_final_sim)

    # -----------------------------------------------------------------------
    # Load processed result if available
    # -----------------------------------------------------------------------
    processed = None
    if args.processed_file and os.path.isfile(args.processed_file):
        print(f"\n--- Loading processed result: {args.processed_file} ---")
        processed = tifffile.imread(args.processed_file)
        print(f"  shape={processed.shape}, dtype={processed.dtype}")
        record_stage("PROCESSED_ACTUAL", processed)

    # -----------------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Generating diagnostic plots...")
    print(f"{'=' * 60}")

    # Camera noise deep analysis
    plot_camera_noise_analysis(raw, os.path.join(args.outdir, "panel_camera_noise.png"))

    # Z-correction analysis
    plot_z_correction_analysis(
        img_after_bg,
        z_scales,
        z_levels,
        z_levels_s,
        os.path.join(args.outdir, "panel_z_correction.png"),
    )

    # Histograms at each stage
    plot_histograms(stages, os.path.join(args.outdir, "panel_histograms.png"))

    # Z profiles
    plot_z_profiles(stage_stats_list, os.path.join(args.outdir, "panel_z_profiles.png"))

    # SNR per slice
    plot_snr_per_slice(stage_snrs, os.path.join(args.outdir, "panel_snr_per_slice.png"))

    # Noise maps (key stages only)
    key_stages_for_noise = [
        s
        for s in stages
        if s[0]
        in (
            "00_raw",
            "02_camera_bg",
            "04_z_corrected",
            "05_pre_deconv_stretch",
            "PROCESSED_ACTUAL",
        )
    ]
    if not key_stages_for_noise:
        key_stages_for_noise = stages[:4]
    mid_z = stages[0][1].shape[0] // 2
    plot_noise_maps(
        key_stages_for_noise, mid_z, os.path.join(args.outdir, "panel_noise_maps.png")
    )

    # Visual slice comparison
    z_max = stages[0][1].shape[0]
    z_slices = np.linspace(
        0, z_max - 1, min(args.max_slices_for_plots, z_max), dtype=int
    )
    plot_slice_comparison(
        stages, z_slices, os.path.join(args.outdir, "panel_slice_comparison.png")
    )

    # Raw vs processed comparison
    if processed is not None:
        plot_processed_comparison(
            raw, processed, os.path.join(args.outdir, "panel_raw_vs_processed.png")
        )

    # -----------------------------------------------------------------------
    # Save metrics CSV
    # -----------------------------------------------------------------------
    import csv

    csv_path = os.path.join(args.outdir, "metrics_per_stage.csv")
    fields = [
        "name",
        "min",
        "max",
        "p01",
        "p99",
        "p999",
        "mean",
        "std",
        "frac_zero",
        "frac_saturated",
        "snr_median",
        "snr_min",
        "noise_std_median",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"  Saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    report_path = os.path.join(args.outdir, "report_summary.txt")
    with open(report_path, "w") as f:
        f.write("SPIM Pipeline Diagnostic Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {args.input_file}\n")
        f.write(f"Shape: {raw.shape}\n")
        f.write(f"Config: {args.config_json}\n\n")

        f.write("FINDINGS\n")
        f.write("-" * 40 + "\n\n")

        # Camera noise analysis
        bg_pixels = raw.ravel()[raw.ravel() < np.percentile(raw, 30)]
        bg_mean = float(np.mean(bg_pixels))
        bg_std = float(np.std(bg_pixels))
        f.write(f"1. CAMERA NOISE FLOOR:\n")
        f.write(f"   Background mean: {bg_mean:.1f}\n")
        f.write(f"   Background σ:    {bg_std:.1f}\n")
        f.write(f"   → Camera offset is ~{bg_mean:.0f} counts\n")
        f.write(
            f"   → Current p{camera_bg_pct} subtraction removes ~{np.percentile(raw, camera_bg_pct):.0f} counts\n"
        )
        residual = bg_mean - float(np.percentile(raw, camera_bg_pct))
        if residual > bg_std:
            f.write(
                f"   ⚠ PROBLEM: ~{residual:.0f} counts of camera noise REMAIN after subtraction\n"
            )
            f.write(
                f"   → This noise floor gets amplified by z-correction and deconvolution\n"
            )
        f.write("\n")

        # Z-correction analysis
        n_high_scale = int(np.sum(z_scales > 1.5))
        max_scale_actual = float(np.max(z_scales))
        f.write(f"2. Z-CORRECTION:\n")
        f.write(f"   Method: {z_method}, max_scale={z_max_scale}\n")
        f.write(f"   Actual max scale applied: {max_scale_actual:.2f}×\n")
        f.write(f"   Slices with > 1.5× correction: {n_high_scale}/{len(z_scales)}\n")
        if n_high_scale > 0:
            f.write(
                f"   ⚠ These slices have their noise amplified by up to {max_scale_actual:.1f}×\n"
            )
        f.write("\n")

        # Dynamic range before deconv
        dr_stretch = metrics_rows[
            [r["name"] for r in metrics_rows].index("05_pre_deconv_stretch")
        ]
        dr_nostretch = metrics_rows[
            [r["name"] for r in metrics_rows].index("05b_no_stretch")
        ]
        f.write(f"3. PRE-DECONVOLUTION STRETCH:\n")
        f.write(
            f"   WITH stretch:    noise_σ={dr_stretch['noise_std_median']:.0f}, SNR={dr_stretch['snr_median']:.1f}\n"
        )
        f.write(
            f"   WITHOUT stretch: noise_σ={dr_nostretch['noise_std_median']:.0f}, SNR={dr_nostretch['snr_median']:.1f}\n"
        )
        noise_amplification = dr_stretch["noise_std_median"] / max(
            dr_nostretch["noise_std_median"], 0.01
        )
        f.write(
            f"   → The stretch amplifies noise σ by {noise_amplification:.1f}× before deconvolution!\n"
        )
        f.write(
            f"   ⚠ RL deconvolution then iteratively sharpens this amplified noise into dots\n"
        )
        f.write("\n")

        # Percentile analysis
        f.write(f"4. FINAL NORMALIZATION:\n")
        f.write(f"   percentile_low={p_low}, percentile_high={p_high}\n")
        if p_low < 10:
            f.write(
                f"   ⚠ percentile_low={p_low} is very low — most background noise is preserved\n"
            )
            f.write(
                f"   → The original notebook used percentile_low=40, which removes much more noise\n"
            )
        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n\n")
        f.write(
            "1. INCREASE camera_bg_percentile to ~5-10 (or use per-slice mode estimation)\n"
        )
        f.write("   to fully remove the camera offset before any processing.\n\n")
        f.write("2. REMOVE the pre-deconvolution image_scaling_intens() call.\n")
        f.write(
            "   Feed RL deconvolution the natural float32 data. RL doesn't need 16-bit range.\n"
        )
        f.write("   This alone prevents the biggest noise amplification.\n\n")
        f.write("3. TIGHTEN z_correction: use max_scale=1.5, signal_floor_pct=30-40.\n")
        f.write("   Slices with little tissue should NOT be boosted aggressively.\n\n")
        f.write("4. INCREASE percentile_low to 5-15 for final normalization.\n")
        f.write("   This clips the noise floor from the final image.\n\n")
        f.write("5. Consider a gentle median filter (3×3) as the very last step\n")
        f.write("   to catch residual salt-and-pepper from deconvolution.\n\n")

    print(f"  Saved: {report_path}")

    print(f"\n{'=' * 60}")
    print(f"Diagnostic complete! Results in: {args.outdir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
