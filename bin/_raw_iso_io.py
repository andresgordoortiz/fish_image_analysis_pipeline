"""Raw-export per-timepoint TIFF writer for the SPIM EXPORT_RAW_ISOTROPIC step.

This module exists because the original SPIM_INPUT_FILE heredoc that
performed XY + optional isotropic-Z rescaling on the RAW input contained
a lot of literal double-quote characters (e.g. ``imagej=True``,
``mode="constant"``, ``.decode('latin-1', errors='replace')``). Those
literals were tripping Groovy's GString parser at the embedded ``?"``
sequence in Nextflow's ``script:`` triple-quoted heredoc blocks,
aborting the script-block compile (round 5 of
``script_block_gstring_escaping.md``). Extracting to a plain Python
module -- the same pattern as ``bin/_ims_reader.py`` -- sidesteps the
issue.

Driver
------
``python3 _raw_iso_io.py export <input_tif> <out_name> <x_res> <y_res>
                              <z_pixel_in> <scale> <do_iso_true_or_false>
                              [<timepoint>]``

Called from the EXPORT_RAW_ISOTROPIC bash block in spim_pipeline.nf.
The bash wrapper provides all values that this script would otherwise
have received via GString interpolation.

Pipeline-stage stamping
-----------------------
The script writes an ImageJ TIFF with the per-timepoint bookkeeping
tags (``TimePoint``, ``WasROICropped``, ``RawExported``, etc.) appended
to the ImageDescription text, matching the convention DOWNSCALE_XY and
CELLPOSE_SEGMENT use so Fiji's Concatenate / Merge Channels dialogs
treat every per-timepoint file uniformly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from skimage.transform import rescale, resize as sk_resize


def export_raw_iso(
    input_tif: Path | str,
    out_name: Path | str,
    x_res_in: float,
    y_res_in: float,
    z_pixel_in: float,
    scale: float,
    do_iso: bool,
    timepoint: int | None = None,
) -> str:
    """Apply XY cubic rescale (+ optional isotropic Z resample) to the
    raw input and write a single per-timepoint 4D-hyperstack-compatible
    TIFF.

    Returns ``out_name`` (string) on success.
    """
    input_tif = Path(input_tif)
    out_name = str(out_name)

    print(f"[DIAG] raw_export scale={scale}  do_iso={do_iso}  "
          f"x_res={x_res_in}  y_res={y_res_in}  z_spacing={z_pixel_in}")

    if not (0.0 < scale <= 1.0):
        raise SystemExit(
            f"scale_factor must satisfy 0 < scale <= 1, got {scale}"
        )

    img = tifffile.imread(str(input_tif))
    if img.ndim != 3:
        raise SystemExit(f"Expected 3D ZYX input, got shape {img.shape}")
    print(f"Input  shape: {img.shape}, dtype: {img.dtype}, scale={scale}")

    # IMPORTANT: do NOT apply CLAHE/normalisation here -- this is RAW
    # export. Only the geometric ops so the viewer can register tracks
    # against the original signal.
    out = rescale(img, (1.0, scale, scale), order=3,
                  preserve_range=True, anti_aliasing=True)
    x_res = x_res_in / scale
    y_res = y_res_in / scale
    print(f"After XY rescale: {out.shape}  "
          f"X/Y pixel size -> {x_res:.4f} x {y_res:.4f} µm")

    if do_iso:
        zoom_z = z_pixel_in / x_res
        if abs(zoom_z - 1.0) < 1e-3:
            print("Z already isotropic, skipping.")
        else:
            expected_z = int(round(out.shape[0] * zoom_z))
            out = sk_resize(
                out,
                (expected_z, out.shape[1], out.shape[2]),
                order=1,
                mode="constant",
                anti_aliasing=False,
                preserve_range=True,
            )
            print(f"Isotropic Z reslice: expected={expected_z} "
                  f"(zoom_z={zoom_z:.4f}, got shape={out.shape})")
            assert out.shape[0] == expected_z, (
                f"Z resample failed: expected {expected_z} planes, "
                f"got {out.shape[0]}"
            )

    if out.dtype != np.uint16:
        out = np.clip(out.astype(np.int32), 0, 65535).astype(np.uint16)

    # Channel discovery: 't####_Channel N.tif' -> N
    channel = "1"
    fname = input_tif.name
    idx = fname.find("_Channel ")
    if idx >= 0:
        rest = fname[idx + len("_Channel "):]
        digits = ""
        for c in rest:
            if c.isdigit():
                digits += c
            else:
                break
        if digits:
            channel = digits

    # Build the per-timepoint output name; caller may have already
    # passed one, but re-derive if not.
    if "{channel}" in out_name:
        out_name = out_name.format(channel=channel)

    tifffile.imwrite(
        out_name,
        out,
        imagej=True,
        resolution=(1.0 / x_res, 1.0 / y_res),
        # compression='zlib' so the Compression TIFF tag (259) is 8 for
        # every per-timepoint TIFF. Without this tifffile defaults to
        # Compression=1, which Fiji's Merge Channels / Concatenate
        # dialogs reject with a 'different bit depth' error.
        compression="zlib",
        metadata={
            "spacing": z_pixel_in if not do_iso else x_res,
            "unit": "um",
            "axes": "ZYX",
            "channels": 1,
            "x_resolution_um": x_res,
            "y_resolution_um": y_res,
        },
    )

    # Stash the per-timepoint bookkeeping tags on top of the ImageJ
    # block that tifffile.imwrite just produced, matching the same
    # DOWNSCALE_XY / CELLPOSE_SEGMENT r+ overwrite pattern.
    # tifffile.imwrite's metadata={...} kwarg lowercases ALL keys when
    # building the ImageDescription text -- we don't rely on it for
    # these bookkeeping tags, we patch them via r+ overwrite.
    nl = os.linesep
    extra_tags_lines = []
    if timepoint is not None:
        extra_tags_lines.append(f"TimePoint={timepoint}")
    extra_tags_lines.extend(
        [
            "WasROICropped=False",   # raw_export pipeline stage never sees an ROI mask
            "RawExported=True",
            f"ScalingFactor={scale}",
            f"IsotropicResliced={str(bool(do_iso))}",
            "PipelineStage=raw_export",
        ]
    )
    extra_tags = nl + nl.join(extra_tags_lines) + nl

    with tifffile.TiffFile(out_name, mode="r+") as tf:
        page = tf.pages[0]
        tag = page.tags.get("ImageDescription")
        if tag is not None:
            current_desc = (
                tag.value.decode("latin-1", errors="replace")
                if isinstance(tag.value, bytes)
                else (tag.value or "")
            )
            new_desc = current_desc.rstrip(nl) + extra_tags + nl
            tag.overwrite(new_desc.encode("latin-1", errors="replace"))

    print(f"Wrote {out_name}")
    return out_name


def _resolve_metadata_voxel_um(metadata_json: Path | str) -> tuple[float, float, float]:
    """Resolve canonical (x_res, y_res, z_pixel) voxel sizes from
    ``shared_metadata.json``. Prefers the sidecar-backed keys when present
    (matched by EXTRACT_METADATA). Falls back to 1.0 µm on each axis
    only when metadata is unusable (which the caller should catch and
    error on, not us)."""
    with open(metadata_json, "r") as f:
        metadata = json.load(f)
    x = float(metadata.get("x_resolution_um", 1.0))
    y = float(metadata.get("y_resolution_um", x))
    if "imagej" in metadata and "spacing" in metadata["imagej"]:
        z = float(metadata["imagej"]["spacing"])
    else:
        z = float(metadata.get("z_resolution_um", 1.0))
    return x, y, z


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="EXPORT_RAW_ISOTROPIC per-timepoint TIFF writer.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser(
        "export",
        help="Write a single per-timepoint raw_iso TIFF with XY (+ optional Z) rescale.",
    )
    p_export.add_argument("input_tif")
    p_export.add_argument("out_name")
    p_export.add_argument("metadata_json",
                          help="shared_metadata.json from EXTRACT_METADATA")
    p_export.add_argument("--scale", type=float, required=True,
                          help="XY scale factor (0 < s <= 1)")
    p_export.add_argument("--do-iso", choices=("True", "False"), default="False")
    p_export.add_argument("--timepoint", type=int, default=None)

    args = p.parse_args(argv)
    if args.cmd == "export":
        x_res, y_res, z_px = _resolve_metadata_voxel_um(args.metadata_json)
        do_iso = args.do_iso == "True"
        try:
            export_raw_iso(
                input_tif=args.input_tif,
                out_name=args.out_name,
                x_res_in=x_res,
                y_res_in=y_res,
                z_pixel_in=z_px,
                scale=args.scale,
                do_iso=do_iso,
                timepoint=args.timepoint,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
