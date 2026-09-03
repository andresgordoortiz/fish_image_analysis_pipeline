"""Shared TIFF I/O helpers for the modular preprocessing scripts.

The SPIM pipeline stores volumes as plain TIFFs with ImageJ metadata
(``imagej=True``, ``resolution``, ``metadata={'spacing': ..., 'unit': 'um',
'axes': 'ZYX'}``). Every preprocessing step needs to:

1. Read a 3D volume (ZYX) and its voxel sizes.
2. Run a numpy operation on the volume.
3. Write the result back to TIFF, preserving / updating the voxel sizes.

These helpers hide the boilerplate so each correction script stays focused
on its single physics step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile


@dataclass(frozen=True)
class VoxelSizes:
    """Physical voxel sizes in micrometres, Z Y X order."""

    z: float
    y: float
    x: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)

    def scale(self, factor_zyx: Sequence[float]) -> "VoxelSizes":
        """Return a new VoxelSizes where each axis is divided by the given
        scaling factor (i.e. the physical size of a single pixel becomes
        ``original / factor`` because the array dimension grew by ``factor``)."""
        fz, fy, fx = factor_zyx
        return VoxelSizes(z=self.z / fz, y=self.y / fy, x=self.x / fx)


@dataclass
class Volume:
    """A 3D volume plus its voxel sizes."""

    data: np.ndarray  # shape (Z, Y, X)
    voxel: VoxelSizes


def read_tiff(path: Path | str) -> Volume:
    """Read a 3D TIFF (ZYX) and return the volume + voxel sizes.

    Voxel sizes are recovered, in order of preference:
      1. ImageJ ``metadata['spacing']`` (Z) + ``resolution`` (X, Y).
      2. ``physical_pixel_sizes`` written by some writers.
      3. Fall back to (1.0, 1.0, 1.0) µm.
    """
    path = Path(path)
    # tifffile.imread handles both single-page TIFFs (one 2D slice) AND
    # multi-page ImageJ hyperstacks (one 2D slice per page). For a 3D volume
    # stored as Z consecutive pages it returns a (Z, Y, X) array. For a
    # single 2D TIFF it returns (Y, X), which we expand to (1, Y, X).
    img = tifffile.imread(str(path))

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim > 3:
        img = np.squeeze(img)
    if img.ndim != 3:
        raise ValueError(
            f"{path}: expected a 3D (ZYX) volume, got shape {img.shape}"
        )

    # Voxel size recovery — try every page so we tolerate multi-page stacks
    # whose metadata only appears on page 0.
    z_um, y_um, x_um = 1.0, 1.0, 1.0
    try:
        with tifffile.TiffFile(str(path)) as tf:
            for page in tf.pages:
                # ImageJ stores Z spacing in the human-readable
                # ``ImageDescription`` tag as ``spacing=<float>``.
                # Newer tifffile exposes this via page.imagej_description
                # (older versions used page.imagej_metadata, removed in 2024+).
                desc = getattr(page, "imagej_description", None)
                if desc and "spacing=" in desc:
                    for token in desc.splitlines():
                        token = token.strip()
                        if token.startswith("spacing="):
                            try:
                                z_um = float(token.split("=", 1)[1])
                            except (ValueError, IndexError):
                                pass

                # X/Y resolution: resolution is (x, y, z, unit) — sometimes
                # 3-tuple. tifffile >= 2022 exposes this as page.resolution.
                # ``1.0 / resolution`` gives the physical pixel size in
                # whatever unit ImageJ stored (typically µm when written
                # via tifffile.imwrite with metadata['unit']='um').
                res = getattr(page, "resolution", None)
                try:
                    if res is not None and res[0] not in (None, 0):
                        x_um = 1.0 / float(res[0])
                        y_um = 1.0 / float(res[1])
                except Exception:
                    pass

                if z_um != 1.0 and x_um != 1.0:
                    break  # got everything we need

            # OME-XML path (when the file is an OME-TIFF)
            ome = getattr(tf, "ome_metadata", None)
            if ome is not None and hasattr(ome, "pixels"):
                px = ome.pixels
                try:
                    z_um = float(px.physical_size_z or z_um)
                    y_um = float(px.physical_size_y or y_um)
                    x_um = float(px.physical_size_x or x_um)
                except Exception:
                    pass
    except Exception:
        # If metadata extraction fails for any reason we still return the
        # volume with 1.0 µm defaults — the caller can override from CLI.
        pass

    return Volume(data=img, voxel=VoxelSizes(z_um, y_um, x_um))


def write_tiff(
    path: Path | str,
    volume: np.ndarray,
    voxel: VoxelSizes,
    *,
    compression: str = "zlib",
) -> None:
    """Write a 3D volume to TIFF with ImageJ metadata (ZYX axes).

    ``voxel.z`` is stored in ``metadata['spacing']`` (ImageJ's Z-spacing
    convention in micrometres), and X/Y resolutions in units-per-micron
    so that downstream readers (ImageJ, napari, ultrack) recover the
    correct physical pixel sizes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_um, y_um, x_um = voxel.as_tuple()
    tifffile.imwrite(
        str(path),
        volume.astype(volume.dtype, copy=False),
        imagej=True,
        resolution=(1.0 / x_um, 1.0 / y_um),
        metadata={
            "spacing": z_um,
            "unit": "um",
            "axes": "ZYX",
        },
        compression=compression,
    )
