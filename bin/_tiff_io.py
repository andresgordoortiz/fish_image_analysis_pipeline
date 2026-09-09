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

      1. ``page.physical_pixel_sizes`` (tifffile >= 2024.9.20) — added
         AFTER this pipeline's container-shipped version (2024.6.18), so
         this path is gated behind ``hasattr``. Verified missing on
         2026-09-09 via ``AttributeError: 'TiffPage' object has no
         attribute 'physical_pixel_sizes'``.

      2. ``page.imagej_metadata['spacing']`` (tifffile >= 2024.x where
         x >= some patch level) — ALSO not present on 2024.6.18.
         Verified missing on 2026-09-09 via ``AttributeError: 'TiffPage'
         object has no attribute 'imagej_metadata'``. We try it
         defensively (gate behind ``hasattr``) so the same code works
         when the container gets bumped to a newer tifffile.

      3. ``TiffFile.imagej_metadata`` (parsed at file-open time, exists
         in tifffile 2024.6.18 as a TiffFile-level attribute even when
         the page-level dict is missing). This is the primary Z-recovery
         path on 2024.6.18 — verified available as of 2024.x.

      4. Raw ``page.tags['ImageDescription']`` byte scan for ``spacing=``
         (TIFF core tag, present in every tifffile version since 2018).
         The byte-string scan is the most reliable path for files written
         by older tifffile (< 2022) that include ``spacing=`` in the raw
         ImageDescription but not in any parsed ImageJ metadata block.

      5. Manual ``XResolution``/``YResolution`` + ``ResolutionUnit``
         conversion to µm/pixel for XY. TIFF core tags (always present).
         ImageJ writes RATIONAL ``XResolution`` as pixels-per-cm by
         default (``ResolutionUnit=3``), so a written value of (96, 1)
         means 96 pixels per cm → 1/96 cm = 0.0104 cm = 104 µm, NOT
         0.0104 µm. Without this conversion, our prior bug fix tried to
         divide 1 by ``XResolution`` directly, producing 0.0104 µm and
         silently corrupting every downstream XY calculation.

      6. Loud WARNING + fall back to ``(1.0, 1.0, 1.0)`` µm.

    The defaults are NEVER used silently — every fallback emits a
    ``WARNING:`` line on stdout so the Nextflow task log catches it.
    Caller-supplied voxel sizes (when the CLI argument is given) always
    override whatever this function returns — see ``make_isotropic`` etc.
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

    z_um, y_um, x_um = 1.0, 1.0, 1.0
    try:
        with tifffile.TiffFile(str(path)) as tf:
            # Step 3 (run once, outside the page loop): TiffFile-level
            # imagej_metadata. tifffile 2024.6.18 has this on TiffFile but
            # not on TiffPage, so we look it up here. Returns a dict for
            # ImageJ hyperstacks, None for plain TIFFs without ImageJ
            # metadata.
            tf_ij_meta = getattr(tf, "imagej_metadata", None)
            tf_ij_spacing = None
            if isinstance(tf_ij_meta, dict):
                tf_ij_spacing = tf_ij_meta.get("spacing")

            for page in tf.pages:
                # ------------------------------------------------------------
                # Step 1: physical_pixel_sizes (tifffile >= 2024.9.20 only).
                # Returns PhysicalPixelSizes(z, y, x) in the unit declared by
                # the file's ``imagej_metadata['unit']`` (defaults to µm for
                # ImageJ hyperstacks written with ``unit='um'``, which is
                # what this pipeline always uses).
                # ------------------------------------------------------------
                if hasattr(page, "physical_pixel_sizes"):
                    pps = page.physical_pixel_sizes
                    if pps is not None:
                        try:
                            if pps.z is not None and pps.z > 0:
                                z_um = float(pps.z)
                            if pps.y is not None and pps.y > 0:
                                y_um = float(pps.y)
                            if pps.x is not None and pps.x > 0:
                                x_um = float(pps.x)
                        except Exception:
                            pass

                # ------------------------------------------------------------
                # Step 2: page.imagej_metadata['spacing'] (defensive — only
                # present in tifffile >= some 2024.x patch). Tifffile 2024.6.18
                # does NOT have this attribute; hasattr gate avoids the
                # AttributeError confirmed on 2026-09-09.
                # ------------------------------------------------------------
                if z_um == 1.0 and hasattr(page, "imagej_metadata"):
                    ij_meta = page.imagej_metadata
                    if isinstance(ij_meta, dict):
                        spacing_fallback = ij_meta.get("spacing")
                        try:
                            if spacing_fallback is not None:
                                z_um = float(spacing_fallback)
                        except (TypeError, ValueError):
                            pass

                # ------------------------------------------------------------
                # Step 3: TiffFile-level imagej_metadata (primary Z path on
                # 2024.6.18). The dict is parsed from the ImageDescription
                # text by tifffile when the file is opened.
                # ------------------------------------------------------------
                if z_um == 1.0 and tf_ij_spacing is not None:
                    try:
                        z_um = float(tf_ij_spacing)
                    except (TypeError, ValueError):
                        pass

                # ------------------------------------------------------------
                # Step 4: raw ImageDescription byte scan. Works on every
                # tifffile version since 2018. Last-resort fallback for
                # files written by very old tifffile that have ``spacing=``
                # in ImageDescription but no proper ImageJ metadata block.
                # ------------------------------------------------------------
                if z_um == 1.0:
                    desc_tag = page.tags.get("ImageDescription")
                    if desc_tag is not None:
                        desc_value = desc_tag.value
                        if isinstance(desc_value, bytes):
                            try:
                                desc_value = desc_value.decode(
                                    "latin-1", errors="replace"
                                )
                            except Exception:
                                desc_value = ""
                        if isinstance(desc_value, str):
                            for token in desc_value.splitlines():
                                token = token.strip()
                                if token.startswith("spacing="):
                                    try:
                                        z_um = float(token.split("=", 1)[1])
                                    except (ValueError, IndexError):
                                        pass

                # ------------------------------------------------------------
                # Step 5: manual XY resolution conversion.
                # TIFF stores ``XResolution``/``YResolution`` as a RATIONAL
                # (numerator, denominator) in the unit given by
                # ``ResolutionUnit``: 1=none, 2=inch, 3=centimeter. ImageJ
                # writes these as ``pixels/cm`` by default
                # (``ResolutionUnit=3``), so a value of (96, 1) means
                # "96 pixels per cm" → 1/96 cm = 0.0104 cm = 104 µm. We
                # convert to µm/pixel ourselves instead of trusting the
                # 1/res arithmetic, which silently produces wrong units
                # when ``ResolutionUnit`` is anything but "none".
                # ------------------------------------------------------------
                if x_um == 1.0 or y_um == 1.0:
                    xres_tag = page.tags.get("XResolution")
                    yres_tag = page.tags.get("YResolution")
                    unit_tag = page.tags.get("ResolutionUnit")
                    if xres_tag is not None and yres_tag is not None:
                        try:
                            xnum, xden = xres_tag.value
                            ynum, yden = yres_tag.value
                            x_rational = float(xnum) / float(xden)
                            y_rational = float(ynum) / float(yden)
                            # TIFF ResolutionUnit enum: 1=none, 2=inch, 3=cm.
                            # Default for ImageJ hyperstacks written via
                            # tifffile.imwrite(imagej=True) is 3 (cm).
                            unit_code = 1
                            if unit_tag is not None:
                                try:
                                    unit_code = int(unit_tag.value)
                                except (TypeError, ValueError):
                                    unit_code = 1
                            if unit_code == 2:
                                x_um_candidate = 25400.0 / x_rational if x_rational else None
                                y_um_candidate = 25400.0 / y_rational if y_rational else None
                            else:
                                # unit_code 1 (none) or 3 (cm) → cm-based
                                x_um_candidate = 10000.0 / x_rational if x_rational else None
                                y_um_candidate = 10000.0 / y_rational if y_rational else None
                            if x_um_candidate is not None and x_um_candidate > 0:
                                x_um = x_um_candidate
                            if y_um_candidate is not None and y_um_candidate > 0:
                                y_um = y_um_candidate
                        except Exception:
                            pass

                if z_um != 1.0 and x_um != 1.0:
                    break  # got everything we need

            # Final safety net: warn loudly if any axis is still 1.0 µm.
            # 1.0 µm is the default fallback — anything else (0.347, 2.0,
            # 0.374) is a real measurement. We never want to silently use
            # the fallback for downstream Z-spacing math.
            if x_um == 1.0 or y_um == 1.0 or z_um == 1.0:
                print(
                    f"WARNING: {path}: could not recover voxel sizes from "
                    f"TIFF metadata (got x={x_um}, y={y_um}, z={z_um} µm). "
                    f"Falling back to (1.0, 1.0, 1.0) µm. Downstream "
                    f"Z-spacing and XY pixel-size calculations WILL BE WRONG. "
                    f"Re-check that this TIFF was written via "
                    f"tifffile.imwrite(..., imagej=True, bigtiff=True, "
                    f"metadata={{'spacing': ..., 'unit': 'um', "
                    f"'axes': 'ZYX'}}, resolution=(<1/x_um>, <1/y_um>)) or "
                    f"set --target_um / --x_um / --y_um / --z_um on the CLI "
                    f"for the script that called read_tiff().",
                    flush=True,
                )
    except Exception as e:
        # If metadata extraction fails entirely we still return the volume
        # with 1.0 µm defaults — but with a louder warning.
        print(
            f"WARNING: {path}: tifffile metadata parsing raised {type(e).__name__}: {e}; "
            f"falling back to (1.0, 1.0, 1.0) µm. Downstream voxel-size "
            f"calculations will be WRONG.",
            flush=True,
        )

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

    ``bigtiff=True`` is mandatory for the ISOTROPIC step: at this
    pipeline's typical Z step (2.0 µm) resampled to 0.374 µm isotropic
    the output Z grows ~5.3×, and a (458, 1576, 1576) uint16 input
    produces a ~12 GB volume. Even with zlib compression the on-disk
    data offset can exceed 4 GB, which trips the classic-TIFF 32-bit
    offset cap and crashes tifffile with
    ``struct.error: 'I' format requires 0 <= number <= 4294967295``
    (this was hit on t0198, 2026-09-09). PLANAR / DEPTH are shape-
    preserving so they don't trip this; they still benefit from BigTIFF
    consistency. Fiji/ImageJ read BigTIFF files fine, and tifffile
    preserves the ImageJ metadata block when ``imagej=True`` and
    ``bigtiff=True`` are combined.

    NOTE on the kwarg name: the canonical tifffile kwarg for forcing
    BigTIFF has always been ``bigtiff=True``. Don't be tempted to use
    ``force_bigtiff=True`` — that name does not exist in any released
    tifffile version (it was confused with Pillow's
    ``forceBigTIFF`` by an earlier fix attempt and broke every writer
    with ``TypeError: imwrite() got an unexpected keyword argument
    'force_bigtiff'`` — PLANAR (t0008) crashed for this reason on
    2026-09-09).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_um, y_um, x_um = voxel.as_tuple()
    tifffile.imwrite(
        str(path),
        volume.astype(volume.dtype, copy=False),
        imagej=True,
        bigtiff=True,
        resolution=(1.0 / x_um, 1.0 / y_um),
        metadata={
            "spacing": z_um,
            "unit": "um",
            "axes": "ZYX",
        },
        compression=compression,
    )
