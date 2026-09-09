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
    parser.add_argument(
        "--out_voxel_x_um", type=float, default=None,
        help="Output X voxel size in µm. The .nf pipeline should pass the "
             "raw-input X µm/px here so the output TIFF preserves the "
             "user's config XY metadata. ISOTROPIC only resamples Z, so "
             "XY stays at the raw value. If unset, falls back to "
             "read_tiff()'s X (which may be wrong for input TIFFs with "
             "missing or bogus ImageJ metadata).",
    )
    parser.add_argument(
        "--out_voxel_y_um", type=float, default=None,
        help="Output Y voxel size in µm. See --out_voxel_x_um.",
    )
    parser.add_argument(
        "--input_voxel_z_um", type=float, default=None,
        help="Override input Z voxel size in µm (used to compute the Z "
             "resample factor). Falls back to read_tiff().vol.voxel.z. "
             "The .nf pipeline passes the canonical Z from "
             "shared_metadata.json here so the input Z is the user's "
             "configured raw-input value, not the bogus 1.0 µm default "
             "the raw TIFF's ImageJ block carries.",
    )
    args = parser.parse_args()

    vol = read_tiff(args.input)
    # Build the (z, y, x) input-voxel tuple for make_isotropic. Prefer
    # the user-provided --input_voxel_z_um (the canonical config value),
    # fall back to read_tiff()'s recovery (which may be bogus for raw
    # input TIFFs that lack proper ImageJ metadata).
    input_z_um = (
        args.input_voxel_z_um
        if args.input_voxel_z_um is not None
        else vol.voxel.z
    )
    corrected_input_voxel = (input_z_um, vol.voxel.y, vol.voxel.x)
    iso = make_isotropic(vol.data, corrected_input_voxel, args.target_um, order=args.order)
    # Output voxel sizes: Z = target_um (the new isotropic Z); XY = raw
    # input XY unless overridden. Default to the user-passed XY if any,
    # else read_tiff's XY (which may be wrong, but at least the .nf
    # pipeline always passes --out_voxel_x_um / --out_voxel_y_um so this
    # fallback is only used for standalone CLI runs).
    out_x_um = args.out_voxel_x_um if args.out_voxel_x_um is not None else vol.voxel.x
    out_y_um = args.out_voxel_y_um if args.out_voxel_y_um is not None else vol.voxel.y
    new_voxel = VoxelSizes(args.target_um, out_y_um, out_x_um)
    write_tiff(args.output, iso, new_voxel)
    print(
        f"isotropic_resample: {vol.data.shape} -> {iso.shape} "
        f"(target={args.target_um} µm, input_z={input_z_um} µm, "
        f"output voxel={new_voxel.as_tuple()} µm, order={args.order})"
    )


if __name__ == "__main__":
    main()
