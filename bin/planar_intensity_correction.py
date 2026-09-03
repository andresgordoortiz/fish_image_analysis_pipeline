#!/usr/bin/env python3
"""Planar (XY) shading / illumination correction.

Corrects uneven illumination across the XY plane of a 3D volume by
estimating a single smooth "flat-field" from the mean-Z projection of the
stack and dividing every slice by it (after clamping the gain to avoid
amplifying noise in dark corners).

Math (ported from AIAF-32 ``planar_intensity_correction.py``):

    proj   = mean(stack, axis=Z)
    field  = gaussian_filter(proj, sigma=sigma_xy)
    gain   = mean(field) / max(field, eps)
    gain   = min(gain, max_ratio)
    out    = stack * gain

Parameters
----------
--input        Path to input TIFF (ZYX).
--output       Path to output TIFF (ZYX).
--sigma_xy     Gaussian sigma (in pixels) for the flat-field estimator.
               Larger = smoother field. Default: 64.

Notes
-----
The flat field is computed from the FULL volume mean, not per-slice, so
the correction is consistent across Z. If you need per-slice shading
correction (e.g. for very depth-dependent illumination) chain this with
``depth_intensity_correction.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from _tiff_io import read_tiff, write_tiff


def planar_intensity_correction(
    array: np.ndarray,
    sigma_xy: float = 64.0,
    eps: float = 1e-8,
    max_ratio: float = 2.0,
    preserve_dtype: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply XY shading correction to a 3D volume.

    Returns the corrected volume and the estimated flat-field (both as
    float32 if ``preserve_dtype=False``).
    """
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array (Z, Y, X), got shape {array.shape}")

    in_dtype = array.dtype
    x = array.astype(np.float32, copy=False)

    proj = np.mean(x, axis=0)
    field = ndi.gaussian_filter(proj, sigma=sigma_xy)
    field_safe = np.maximum(field, eps)
    ratio = float(np.mean(field_safe)) / field_safe
    ratio = np.minimum(ratio, max_ratio)  # never amplify more than 2x

    corrected = x * ratio[np.newaxis, :, :]

    if not preserve_dtype:
        return corrected, field

    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(in_dtype)
    else:
        corrected = corrected.astype(in_dtype, copy=False)
    return corrected, field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Input TIFF (ZYX).")
    parser.add_argument("--output", type=Path, required=True, help="Output TIFF (ZYX).")
    parser.add_argument(
        "--sigma_xy",
        type=float,
        default=64.0,
        help="Gaussian sigma (pixels) for the flat-field estimator. Default: 64.",
    )
    parser.add_argument(
        "--max_ratio",
        type=float,
        default=2.0,
        help="Maximum multiplicative gain (clamps dark-region amplification). Default: 2.0.",
    )
    args = parser.parse_args()

    vol = read_tiff(args.input)
    corrected, _field = planar_intensity_correction(
        vol.data, sigma_xy=args.sigma_xy, max_ratio=args.max_ratio
    )
    write_tiff(args.output, corrected, vol.voxel)
    print(
        f"planar_intensity_correction: {vol.data.shape} -> {corrected.shape} "
        f"(sigma_xy={args.sigma_xy}, voxel={vol.voxel.as_tuple()} µm)"
    )


if __name__ == "__main__":
    main()
