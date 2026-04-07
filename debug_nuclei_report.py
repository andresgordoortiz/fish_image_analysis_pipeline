#!/usr/bin/env python3
"""
Debug Preprocessing — Multi-Stage Nuclei Comparison Report
============================================================
IMP Vienna — Andrés Gordo

Collects JSON outputs from ``debug_nuclei_tracking.py`` (one per
preprocessing stage) and generates a comprehensive comparison report
showing how nuclei counts, sizes, and quality evolve through the pipeline.

Usage (standalone — normally called by Nextflow)
-------------------------------------------------
    python debug_nuclei_report.py \
        --json_dir ./debug_results/ \
        --outdir ./debug_report/

What it produces
----------------
debug_report/
├── nuclei_comparison.csv           # tabular: stage × metric
├── panel_nuclei_per_slice.png      # per-slice nuclei count overlay
├── panel_nuclei_count_bar.png      # total 3D nuclei per stage (bar)
├── panel_area_per_slice.png        # mean nucleus area per slice
├── panel_volume_distribution.png   # volume box/violin per stage
├── panel_size_evolution.png        # median volume evolution
├── panel_z_coverage.png            # Z-coverage & cells-per-Z
├── panel_loss_tracking.png         # nuclei gained/lost between stages
├── panel_intensity_snr.png         # SNR inside nuclei per stage
├── debug_report.txt                # text summary with alerts
"""

import argparse
import csv
import json
import os
import sys
import warnings
from glob import glob
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Data loading
# ============================================================================


def load_stage_results(json_dir):
    """Load all *_nuclei_stats.json files and return sorted by stage order."""
    pattern = os.path.join(json_dir, "*_nuclei_stats.json")
    files = sorted(glob(pattern))
    if not files:
        print(f"ERROR: No *_nuclei_stats.json files found in {json_dir}")
        sys.exit(1)

    results = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        results.append(data)

    # Sort by stage name (lexicographic → 00_raw < 01_after_camera_bg < ...)
    results.sort(key=lambda r: r["stage_name"])
    print(f"Loaded {len(results)} stage results:")
    for r in results:
        n = r["global_nuclei_stats"]["n_nuclei_3d"]
        print(f"  {r['stage_name']}: {n} nuclei")
    return results


# ============================================================================
# Plotting helpers
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


def stage_colors(n):
    """Generate distinct colours for n stages."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab10" if n <= 10 else "tab20", n)
    return [cmap(i) for i in range(n)]


# ============================================================================
# Plots
# ============================================================================


def plot_nuclei_per_slice(results, outpath):
    """Overlay per-slice nuclei counts for all stages."""
    plt = try_matplotlib()
    if plt is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    colors = stage_colors(len(results))

    for i, r in enumerate(results):
        ss = r["per_slice_stats"]
        z_idx = ss["z_index"]
        ax1.plot(z_idx, ss["n_nuclei"], label=r["stage_name"],
                 color=colors[i], lw=1.2, alpha=0.85)
        ax2.plot(z_idx, ss["n_nuclei_centroid"], label=r["stage_name"],
                 color=colors[i], lw=1.2, alpha=0.85)

    ax1.set_ylabel("Nuclei touching slice")
    ax1.set_title("Per-Slice Nuclei Count (all labels intersecting slice)")
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Nuclei with centroid in slice")
    ax2.set_xlabel("Z slice index")
    ax2.set_title("Per-Slice Nuclei Count (centroid-based)")
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Nuclei Count Across Pipeline Stages", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_nuclei_count_bar(results, outpath):
    """Bar chart of total 3D nuclei per stage."""
    plt = try_matplotlib()
    if plt is None:
        return

    names = [r["stage_name"] for r in results]
    counts = [r["global_nuclei_stats"]["n_nuclei_3d"] for r in results]
    colors = stage_colors(len(results))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 6))
    bars = ax.bar(range(len(names)), counts, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate with count
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total 3D Nuclei")
    ax.set_title("Total Nuclei Detected at Each Pipeline Stage")
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight losses
    if len(counts) > 1:
        max_count = max(counts)
        for i in range(1, len(counts)):
            delta = counts[i] - counts[i - 1]
            pct = 100 * delta / counts[i - 1] if counts[i - 1] > 0 else 0
            color = "green" if delta >= 0 else "red"
            ax.annotate(f"{delta:+d}\n({pct:+.1f}%)",
                        xy=(i, counts[i]),
                        xytext=(i, counts[i] + max_count * 0.08),
                        fontsize=7, color=color, ha="center",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_area_per_slice(results, outpath):
    """Mean and median nucleus area per slice across stages."""
    plt = try_matplotlib()
    if plt is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    colors = stage_colors(len(results))

    for i, r in enumerate(results):
        ss = r["per_slice_stats"]
        z_idx = ss["z_index"]
        ax1.plot(z_idx, ss["mean_area_px"], label=r["stage_name"],
                 color=colors[i], lw=1.2)
        ax2.plot(z_idx, ss["median_area_px"], label=r["stage_name"],
                 color=colors[i], lw=1.2)

    ax1.set_ylabel("Mean nucleus area (px)")
    ax1.set_title("Mean Nucleus Area per Slice — Are Nuclei Growing/Shrinking?")
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Median nucleus area (px)")
    ax2.set_xlabel("Z slice index")
    ax2.set_title("Median Nucleus Area per Slice")
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Nucleus Size Evolution Across Stages", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_volume_distribution(results, outpath):
    """Box plot (or strip) of 3D volumes per stage — from summary stats."""
    plt = try_matplotlib()
    if plt is None:
        return

    names = [r["stage_name"] for r in results]
    medians = [r["global_nuclei_stats"]["median_volume_px"] for r in results]
    means = [r["global_nuclei_stats"]["mean_volume_px"] for r in results]
    stds = [r["global_nuclei_stats"]["std_volume_px"] for r in results]
    mins = [r["global_nuclei_stats"]["min_volume_px"] for r in results]
    maxs = [r["global_nuclei_stats"]["max_volume_px"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = range(len(names))
    colors = stage_colors(len(results))

    # Median + range
    ax1.bar(x, medians, color=colors, edgecolor="black", linewidth=0.5, label="median")
    ax1.errorbar(x, means, yerr=stds, fmt="none", ecolor="red", capsize=4, lw=1.5, label="mean ± std")
    for xi, med in zip(x, medians):
        ax1.text(xi, med + max(medians) * 0.02, f"{med:.0f}", ha="center", fontsize=8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Volume (px)")
    ax1.set_title("Median & Mean±Std Volume")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # CV, small/large fractions
    cvs = [r["global_nuclei_stats"]["volume_cv"] for r in results]
    smalls = [r["global_nuclei_stats"]["small_fraction"] for r in results]
    larges = [r["global_nuclei_stats"]["large_fraction"] for r in results]

    w = 0.25
    ax2.bar([xi - w for xi in x], cvs, width=w, color="steelblue", label="Volume CV", edgecolor="black", lw=0.5)
    ax2.bar([xi for xi in x], smalls, width=w, color="orange", label="Small frac (<0.25× median)", edgecolor="black", lw=0.5)
    ax2.bar([xi + w for xi in x], larges, width=w, color="red", label="Large frac (>4× median)", edgecolor="black", lw=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Fraction / CV")
    ax2.set_title("Size Heterogeneity — Splitting & Merging Indicators")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Nucleus Volume Analysis Across Stages", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_z_coverage(results, outpath):
    """Z-coverage and cells-per-Z profiles."""
    plt = try_matplotlib()
    if plt is None:
        return

    names = [r["stage_name"] for r in results]
    coverages = [r["global_nuclei_stats"]["z_coverage"] for r in results]
    cpz_mean = [r["global_nuclei_stats"]["cells_per_z_mean"] for r in results]
    cpz_std = [r["global_nuclei_stats"]["cells_per_z_std"] for r in results]
    colors = stage_colors(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(range(len(names)), coverages, color=colors, edgecolor="black", lw=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Z coverage (fraction)")
    ax1.set_title("Z-axis Coverage (fraction of slices with nuclei)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(names)), cpz_mean, color=colors, edgecolor="black", lw=0.5)
    ax2.errorbar(range(len(names)), cpz_mean, yerr=cpz_std, fmt="none", ecolor="red", capsize=3)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Nuclei per Z-slice")
    ax2.set_title("Mean Nuclei per Active Z-Slice (±std)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Depth Penetration — Are Deep Nuclei Being Lost?", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_loss_tracking(results, outpath):
    """Cumulative nuclei gain/loss between consecutive stages."""
    plt = try_matplotlib()
    if plt is None:
        return

    if len(results) < 2:
        return

    names = [r["stage_name"] for r in results]
    counts = [r["global_nuclei_stats"]["n_nuclei_3d"] for r in results]

    deltas = [0] + [counts[i] - counts[i - 1] for i in range(1, len(counts))]
    pcts = [0.0] + [100 * d / counts[i - 1] if counts[i - 1] > 0 else 0
                     for i, d in enumerate(deltas[1:], 1)]
    cumulative = np.cumsum(deltas)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Per-step delta
    bar_colors = ["green" if d >= 0 else "red" for d in deltas]
    ax1.bar(range(len(names)), deltas, color=bar_colors, edgecolor="black", lw=0.5)
    for i, (d, p) in enumerate(zip(deltas, pcts)):
        if i == 0:
            continue
        ax1.text(i, d + (max(deltas) - min(deltas)) * 0.03 * (1 if d >= 0 else -1),
                 f"{d:+d}\n({p:+.1f}%)", ha="center", fontsize=8,
                 color="darkgreen" if d >= 0 else "darkred")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Δ nuclei")
    ax1.set_title("Per-Stage Change in Nuclei Count")
    ax1.axhline(0, color="black", ls="-", lw=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # Cumulative
    ax2.plot(range(len(names)), cumulative, "o-", color="navy", lw=2)
    ax2.fill_between(range(len(names)), 0, cumulative,
                     where=np.array(cumulative) >= 0, color="green", alpha=0.2)
    ax2.fill_between(range(len(names)), 0, cumulative,
                     where=np.array(cumulative) < 0, color="red", alpha=0.2)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Cumulative Δ nuclei")
    ax2.set_title("Cumulative Nuclei Change (relative to first stage)")
    ax2.axhline(0, color="black", ls="-", lw=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Nuclei Loss/Gain Tracking Through Pipeline", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_intensity_snr(results, outpath):
    """SNR inside nuclei and mean intensity per stage."""
    plt = try_matplotlib()
    if plt is None:
        return

    names = [r["stage_name"] for r in results]
    snrs = [r["intensity_in_nuclei"]["snr_in_nuclei"] for r in results]
    mean_int = [r["intensity_in_nuclei"]["mean_intensity"] for r in results]
    colors = stage_colors(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(range(len(names)), snrs, color=colors, edgecolor="black", lw=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("SNR (nucleus mean / bg σ)")
    ax1.set_title("Signal-to-Noise Ratio Inside Nuclei")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(range(len(names)), mean_int, color=colors, edgecolor="black", lw=0.5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean nucleus intensity")
    ax2.set_title("Mean Intensity Inside Nuclei")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Signal Quality in Detected Nuclei", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_deep_vs_shallow(results, outpath):
    """Compare nuclei counts in deep (bottom 25%) vs shallow (top 25%) Z."""
    plt = try_matplotlib()
    if plt is None:
        return

    names = [r["stage_name"] for r in results]
    deep_counts = []
    shallow_counts = []
    mid_counts = []

    for r in results:
        ss = r["per_slice_stats"]
        Z = len(ss["n_nuclei"])
        q1, q3 = Z // 4, 3 * Z // 4
        shallow = sum(ss["n_nuclei_centroid"][:q1])
        mid = sum(ss["n_nuclei_centroid"][q1:q3])
        deep = sum(ss["n_nuclei_centroid"][q3:])
        shallow_counts.append(shallow)
        mid_counts.append(mid)
        deep_counts.append(deep)

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.3), 6))
    ax.bar(x - w, shallow_counts, width=w, label="Shallow (top 25% Z)", color="#2ecc71", edgecolor="black", lw=0.5)
    ax.bar(x, mid_counts, width=w, label="Mid (25-75% Z)", color="#3498db", edgecolor="black", lw=0.5)
    ax.bar(x + w, deep_counts, width=w, label="Deep (bottom 25% Z)", color="#e74c3c", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Nuclei count (centroid-based)")
    ax.set_title("Nuclei Distribution: Shallow vs Mid vs Deep — Where Are Nuclei Lost?")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================================
# Report & CSV
# ============================================================================


def write_csv(results, outpath):
    """Write comparison CSV with one row per stage."""
    fields = [
        "stage_name", "n_nuclei_3d", "median_volume_px", "mean_volume_px",
        "std_volume_px", "volume_cv", "mean_z_extent", "mean_xy_extent",
        "small_fraction", "large_fraction", "z_coverage",
        "cells_per_z_mean", "cells_per_z_std",
        "snr_in_nuclei", "mean_intensity", "std_intensity",
    ]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            g = r["global_nuclei_stats"]
            i = r["intensity_in_nuclei"]
            row = {
                "stage_name": r["stage_name"],
                **{k: g.get(k, "") for k in fields if k in g},
                "snr_in_nuclei": i["snr_in_nuclei"],
                "mean_intensity": i["mean_intensity"],
                "std_intensity": i["std_intensity"],
            }
            writer.writerow(row)
    print(f"  Saved: {outpath}")


def write_report(results, outpath):
    """Write text report with alerts for nuclei loss."""
    with open(outpath, "w") as f:
        f.write("Debug Preprocessing — Nuclei Tracking Report\n")
        f.write("=" * 60 + "\n\n")

        # Summary table
        f.write(f"{'Stage':<35} {'Nuclei':>7} {'Δ':>6} {'Δ%':>7} {'Vol_med':>8} {'Vol_CV':>7} {'Z_cov':>6}\n")
        f.write("-" * 80 + "\n")

        prev_count = None
        for r in results:
            g = r["global_nuclei_stats"]
            n = g["n_nuclei_3d"]
            if prev_count is not None:
                delta = n - prev_count
                delta_pct = 100 * delta / prev_count if prev_count > 0 else 0
            else:
                delta, delta_pct = 0, 0.0
            f.write(f"{r['stage_name']:<35} {n:>7d} {delta:>+6d} {delta_pct:>+6.1f}% "
                    f"{g['median_volume_px']:>8.0f} {g['volume_cv']:>7.3f} {g['z_coverage']:>5.1%}\n")
            prev_count = n

        f.write("\n\nALERTS\n")
        f.write("-" * 40 + "\n")

        alerts = []
        for i in range(1, len(results)):
            r_prev = results[i - 1]
            r_curr = results[i]
            n_prev = r_prev["global_nuclei_stats"]["n_nuclei_3d"]
            n_curr = r_curr["global_nuclei_stats"]["n_nuclei_3d"]

            if n_prev == 0:
                continue

            pct_change = 100 * (n_curr - n_prev) / n_prev

            # Alert: significant nuclei loss (>5%)
            if pct_change < -5:
                alerts.append(
                    f"NUCLEI LOSS: {r_prev['stage_name']} → {r_curr['stage_name']}: "
                    f"{n_prev} → {n_curr} ({pct_change:+.1f}%)"
                )

            # Alert: nuclei gain (possible splitting or noise detections)
            if pct_change > 15:
                alerts.append(
                    f"NUCLEI GAIN: {r_prev['stage_name']} → {r_curr['stage_name']}: "
                    f"{n_prev} → {n_curr} ({pct_change:+.1f}%) — possible splitting or noise"
                )

            # Alert: volume change (merging or splitting)
            vol_prev = r_prev["global_nuclei_stats"]["median_volume_px"]
            vol_curr = r_curr["global_nuclei_stats"]["median_volume_px"]
            if vol_prev > 0:
                vol_pct = 100 * (vol_curr - vol_prev) / vol_prev
                if abs(vol_pct) > 20:
                    direction = "GROWING" if vol_pct > 0 else "SHRINKING"
                    alerts.append(
                        f"VOLUME {direction}: {r_prev['stage_name']} → {r_curr['stage_name']}: "
                        f"median {vol_prev:.0f} → {vol_curr:.0f} px ({vol_pct:+.1f}%)"
                    )

            # Alert: small fraction spike (splitting)
            sf_prev = r_prev["global_nuclei_stats"]["small_fraction"]
            sf_curr = r_curr["global_nuclei_stats"]["small_fraction"]
            if sf_curr - sf_prev > 0.1:
                alerts.append(
                    f"SPLITTING?: {r_curr['stage_name']}: small nucleus fraction "
                    f"jumped {sf_prev:.1%} → {sf_curr:.1%}"
                )

            # Alert: large fraction spike (merging)
            lf_prev = r_prev["global_nuclei_stats"]["large_fraction"]
            lf_curr = r_curr["global_nuclei_stats"]["large_fraction"]
            if lf_curr - lf_prev > 0.05:
                alerts.append(
                    f"MERGING?: {r_curr['stage_name']}: large nucleus fraction "
                    f"jumped {lf_prev:.1%} → {lf_curr:.1%}"
                )

            # Alert: Z coverage drop (losing deep nuclei)
            zc_prev = r_prev["global_nuclei_stats"]["z_coverage"]
            zc_curr = r_curr["global_nuclei_stats"]["z_coverage"]
            if zc_prev > 0 and (zc_curr - zc_prev) / zc_prev < -0.05:
                alerts.append(
                    f"DEPTH LOSS: {r_curr['stage_name']}: Z-coverage dropped "
                    f"{zc_prev:.1%} → {zc_curr:.1%}"
                )

        if alerts:
            for a in alerts:
                f.write(f"  ⚠ {a}\n")
        else:
            f.write("  No significant issues detected.\n")

        # Deep vs shallow analysis
        f.write("\n\nDEPTH ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for r in results:
            ss = r["per_slice_stats"]
            Z = len(ss["n_nuclei"])
            q1, q3 = Z // 4, 3 * Z // 4
            shallow = sum(ss["n_nuclei_centroid"][:q1])
            deep = sum(ss["n_nuclei_centroid"][q3:])
            total = sum(ss["n_nuclei_centroid"])
            ratio = deep / shallow if shallow > 0 else 0
            f.write(f"  {r['stage_name']:<30}: shallow={shallow}, deep={deep}, "
                    f"ratio={ratio:.2f}, total={total}\n")

    print(f"  Saved: {outpath}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate nuclei comparison report from per-stage JSON metrics",
    )
    parser.add_argument("--json_dir", required=True,
                        help="Directory containing *_nuclei_stats.json files")
    parser.add_argument("--outdir", required=True, help="Output directory for report")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print("DEBUG PREPROCESSING — NUCLEI REPORT")
    print(f"{'='*60}")

    results = load_stage_results(args.json_dir)

    print(f"\nGenerating plots...")
    plot_nuclei_per_slice(results, os.path.join(args.outdir, "panel_nuclei_per_slice.png"))
    plot_nuclei_count_bar(results, os.path.join(args.outdir, "panel_nuclei_count_bar.png"))
    plot_area_per_slice(results, os.path.join(args.outdir, "panel_area_per_slice.png"))
    plot_volume_distribution(results, os.path.join(args.outdir, "panel_volume_distribution.png"))
    plot_z_coverage(results, os.path.join(args.outdir, "panel_z_coverage.png"))
    plot_loss_tracking(results, os.path.join(args.outdir, "panel_loss_tracking.png"))
    plot_intensity_snr(results, os.path.join(args.outdir, "panel_intensity_snr.png"))
    plot_deep_vs_shallow(results, os.path.join(args.outdir, "panel_deep_vs_shallow.png"))

    print(f"\nWriting CSV and report...")
    write_csv(results, os.path.join(args.outdir, "nuclei_comparison.csv"))
    write_report(results, os.path.join(args.outdir, "debug_report.txt"))

    print(f"\n{'='*60}")
    print("REPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {args.outdir}")


if __name__ == "__main__":
    main()
