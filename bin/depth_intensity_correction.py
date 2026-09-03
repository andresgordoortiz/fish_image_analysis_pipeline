#!/usr/bin/env python3
"""Depth (Z) intensity correction.

Compensates depth-dependent intensity variation (absorption, bleaching,
illumination fall-off along the light sheet) by rescaling every Z slice
so that a robust per-slice intensity statistic is constant along Z.

Math (ported from AIAF-32 ``depth_intensity_correction.py``):

    for z in Z:
        level[z] = statistic(slice z)         # mean / median / pXX
    level_smooth = moving_average(level, smooth_window)
    target = median(level_smooth)
    scale[z] = target / level_smooth[z]
    scale   = clip(scale, gain_min, gain_max)
    out     = stack * scale[:, None, None]

Parameters
----------
--input          Path to input TIFF (ZYX).
--output         Path to output TIFF (ZYX).
--mode           Per-slice statistic: 'mean', 'median', or 'pXX' (e.g. 'p99').
--smooth_window  Moving-average window along Z (odd integer). <=1 disables.
--gain_min       Minimum multiplicative gain (clamps bright-Z darkening).
--gain_max       Maximum multiplicative gain (clamps dark-Z brightening).

Defaults are tuned for typical SPIM embryo stacks: p99 + window 9 +
gain clip (0.25, 4.0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _tiff_io import read_tiff, write_tiff


def depth_intensity_correction(
    array: np.ndarray,
    mode: str = "p99",
    smooth_window: int = 9,
    eps: float = 1e-8,
    preserve_dtype: bool = True,
    gain_clip: tuple[float, float] | None = (0.25, 4.0),
) -> np.ndarray:
    """Apply Z-depth intensity correction to a 3D volume."""
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array (Z, Y, X), got shape {array.shape}")

    if mode not in ("median", "mean") and not mode.startswith("p"):
        raise ValueError("mode must be 'median', 'mean' or 'pXX' like 'p95'.")

    q: float | None = None
    if mode.startswith("p"):
        try:
            q = float(mode[1:])
        except ValueError as exc:
            raise ValueError("Invalid percentile mode. Use 'pXX' like 'p95'.") from exc
        if not (0.0 < q < 100.0):
            raise ValueError("Percentile in mode='pXX' must be in (0, 100).")

    src_dtype = array.dtype
    x = array.astype(np.float32, copy=False)
    z_dim = x.shape[0]
    flat = x.reshape(z_dim, -1)
    levels = np.empty(z_dim, dtype=np.float32)

    for z_idx in range(z_dim):
        vals = flat[z_idx]
        if mode == "median":
            levels[z_idx] = np.median(vals)
        elif mode == "mean":
            levels[z_idx] = np.mean(vals)
        else:
            levels[z_idx] = np.percentile(vals, q)

    levels = np.maximum(levels, eps)

    levels_s = levels
    if smooth_window is not None and smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        pad = smooth_window // 2
        lvl_pad = np.pad(levels, (pad, pad), mode="edge")
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        levels_s = np.convolve(lvl_pad, kernel, mode="valid")

    target = float(np.median(levels_s))
    scales = target / np.maximum(levels_s, eps)

    if gain_clip is not None:
        scales = np.clip(scales, gain_clip[0], gain_clip[1])

    result = x * scales[:, None, None]

    if not preserve_dtype:
        return result.astype(np.float32, copy=False)

    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        result = np.clip(result, info.min, info.max).astype(src_dtype)
    else:
        result = result.astype(src_dtype, copy=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Input TIFF (ZYX).")
    parser.add_argument("--output", type=Path, required=True, help="Output TIFF (ZYX).")
    parser.add_argument(
        "--mode",
        type=str,
        default="p99",
        help="Per-slice statistic: 'mean', 'median', or 'pXX' (e.g. 'p99').",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=9,
        help="Moving-average window along Z (odd integer). <=1 disables.",
    )
    parser.add_argument(
        "--gain_min",
        type=float,
        default=0.25,
        help="Minimum multiplicative gain applied to any slice.",
    )
    parser.add_argument(
        "--gain_max",
        type=float,
        default=4.0,
        help="Maximum multiplicative gain applied to any slice.",
    )
    args = parser.parse_args()

    vol = read_tiff(args.input)
    corrected = depth_intensity_correction(
        vol.data,
        mode=args.mode,
        smooth_window=args.smooth_window,
        gain_clip=(args.gain_min, args.gain_max),
    )
    write_tiff(args.output, corrected, vol.voxel)
    print(
        f"depth_intensity_correction: {vol.data.shape} -> {corrected.shape} "
        f"(mode={args.mode}, smooth={args.smooth_window}, voxel={vol.voxel.as_tuple()} µm)"
    )


if __name__ == "__main__":
    main()
