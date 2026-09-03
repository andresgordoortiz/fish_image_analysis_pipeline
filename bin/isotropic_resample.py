#!/usr/bin/env python3
"""Isotropic Z-resampling.

Resamples the Z axis so the output voxel size matches the (smaller) XY
voxel size. After this step every output volume has cubic voxels, which
is the precondition for the rest of the pipeline (Cellpose 3D, ultrack
segmentation, viewer overlay) to treat Z and XY uniformly.

Math (ported from AIAF-32 ``isotropic.py``):

    spacing     = (z_um, y_um, x_um)
    target      = (target_um, target_um, target_um)
    new_shape   = round(old_shape * spacing / target)
    out         = resize(stack, new_shape, order=order, anti_aliasing=True,
                          preserve_range=True)
    out         = clip(out, stack.min(), stack.max())

Parameters
----------
--input         Path to input TIFF (ZYX).
--output        Path to output TIFF (ZYX).
--target_um     Target isotropic voxel size in micrometres (default 0.374).
--order         Interpolation order: 1 (linear) or 3 (cubic). Default 3.

Note: the new voxel sizes are written to the output's ImageJ metadata as
``(target_um, target_um, target_um)`` so downstream readers see the new
geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage.transform import resize

from _tiff_io import VoxelSizes, read_tiff, write_tiff


def make_isotropic(
    array: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    target_um: float,
    order: int = 3,
) -> np.ndarray:
    """Resample a 3D volume to isotropic voxels of size ``target_um``."""
    spacing = np.asarray(spacing_zyx, dtype=float)
    target = np.asarray((target_um, target_um, target_um), dtype=float)

    spatial_shape = np.array(array.shape[-3:])
    zoom = spacing / target
    new_spatial_shape = np.round(spatial_shape * zoom).astype(int)

    if np.all(new_spatial_shape == spatial_shape):
        # Already isotropic at this resolution; avoid a no-op resize that
        # could shift dtype or introduce subtle float drift.
        return array.astype(array.dtype, copy=False)

    isotropic = resize(
        array,
        new_spatial_shape,
        order=order,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )

    # Clip to original range to remove spline overshoot.
    original_min, original_max = float(array.min()), float(array.max())
    isotropic = np.clip(isotropic, original_min, original_max)

    return isotropic.astype(array.dtype, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Input TIFF (ZYX).")
    parser.add_argument("--output", type=Path, required=True, help="Output TIFF (ZYX).")
    parser.add_argument(
        "--target_um",
        type=float,
        default=0.374,
        help="Target isotropic voxel size in micrometres (Z, Y, X).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Spline interpolation order (0..5). 1 = linear, 3 = cubic. Default 3.",
    )
    args = parser.parse_args()

    vol = read_tiff(args.input)
    iso = make_isotropic(vol.data, vol.voxel.as_tuple(), args.target_um, order=args.order)
    new_voxel = VoxelSizes(args.target_um, args.target_um, args.target_um)
    write_tiff(args.output, iso, new_voxel)
    print(
        f"isotropic_resample: {vol.data.shape} -> {iso.shape} "
        f"(target={args.target_um} µm, order={args.order})"
    )


if __name__ == "__main__":
    main()
