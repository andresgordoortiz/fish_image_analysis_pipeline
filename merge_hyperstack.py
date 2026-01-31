#!/usr/bin/env python3
"""
Merge segmented timepoints into 4D hyperstack (TIFF or BDV/HDF5 format).
Refactored to use pybdv for correct XML/HDF5 generation.
"""

import json
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import tifffile

# Try importing pybdv, install if missing (for runtime safety)
try:
    import pybdv
    from pybdv.metadata import write_xml_metadata
except ImportError:
    print("pybdv not found. Attempting install...")
    os.system("micromamba install -y -n microscopy_env -c conda-forge pybdv")
    import pybdv
    from pybdv.metadata import write_xml_metadata


def detect_flip_needed(first_path):
    """Auto-detect if Y-axis flip is needed based on TIFF orientation tag."""
    try:
        with tifffile.TiffFile(str(first_path)) as tf:
            page = tf.pages[0]
            tags = page.tags
            if "Orientation" in tags:
                val = tags["Orientation"].value
                # Orientation 3/4 (bottom-right) or 7/8 (left-bot) usually need flip
                return val in (3, 4, 7, 8)
    except Exception as e:
        print(f"  Orientation auto-detect warning: {e}")
    return False


def write_tiff_hyperstack(
    img4d, final_x_res, final_y_res, final_z_spacing, need_flip, hyperstack_meta
):
    """Write 4D array as standard TIFF hyperstack."""
    print("\n" + "=" * 60)
    print("Writing TIFF hyperstack")
    print("=" * 60)

    # Save Metadata
    with open("4D_hyperstack_metadata.json", "w") as fh:
        json.dump(hyperstack_meta, fh, indent=2)

    print(f"Writing 4D_hyperstack.tif ({img4d.nbytes / 1024**3:.2f} GB)...")

    tifffile.imwrite(
        "4D_hyperstack.tif",
        img4d.astype(np.uint16),
        imagej=True,
        resolution=(1.0 / final_x_res, 1.0 / final_y_res),
        metadata={
            "spacing": final_z_spacing,
            "unit": "um",
            "axes": "TZYX",
            "frames": img4d.shape[0],
            "slices": img4d.shape[1],
            "LabelImage": True,  # Helps Fiji recognize it as segmentation
        },
    )
    print("✓ TIFF Write Complete")


def write_bdv_hdf5(
    seg_files, final_x_res, final_y_res, final_z_spacing, need_flip, hyperstack_meta
):
    """
    Write timepoints using pybdv.
    This generates the multi-resolution pyramid and proper XML automatically.
    """
    print("\n" + "=" * 60)
    print("Writing BDV HDF5 + XML (via pybdv)")
    print("=" * 60)

    h5_path = "4D_hyperstack.h5"
    xml_path = "4D_hyperstack.xml"

    # Clean up existing files to prevent corruption
    if os.path.exists(h5_path):
        os.remove(h5_path)
    if os.path.exists(xml_path):
        os.remove(xml_path)

    # Define standard BDV setup
    setup_id = 0

    # ---------------------------------------------------------
    # 1. HDF5 Generation (with Mipmaps)
    # ---------------------------------------------------------
    print(
        f"Target Voxel Size (Z, Y, X): {final_z_spacing:.3f}, {final_y_res:.3f}, {final_x_res:.3f} um"
    )

    T = len(seg_files)

    for t, file_path in enumerate(seg_files):
        print(f"Processing timepoint {t + 1}/{T}: {file_path.name}")

        # Read Data
        data = tifffile.imread(str(file_path)).astype(np.uint16)

        # Flip Y if requested
        if need_flip:
            data = np.flip(data, axis=1)

        # Write to BDV
        # downscale_mode="nearest" is CRITICAL for segmentation to avoid interpolation artifacts
        pybdv.make_bdv(
            data,
            h5_path,
            downscale_factors=[[1, 1, 1], [2, 2, 2], [4, 4, 4]],
            downscale_mode="nearest",
            timepoint=t,
            setup_id=setup_id,
            overwrite=(t == 0),  # Overwrite file on first TP, append on others
        )

    # ---------------------------------------------------------
    # 2. XML Generation
    # ---------------------------------------------------------
    print("\nGenerating XML...")

    # pybdv expects resolution in (Z, Y, X) order matching the numpy array
    resolution = [final_z_spacing, final_y_res, final_x_res]

    write_xml_metadata(
        xml_path,
        h5_path,
        unit="um",
        resolution=resolution,
        setup_id=setup_id,
        timepoints=list(range(T)),
    )

    # ---------------------------------------------------------
    # 3. Finalize Metadata
    # ---------------------------------------------------------
    hyperstack_meta["bdv_info"] = {
        "file": h5_path,
        "xml": xml_path,
        "setup_id": setup_id,
        "pybdv_used": True,
        "downscale_mode": "nearest",
    }

    with open("4D_hyperstack_metadata.json", "w") as fh:
        json.dump(hyperstack_meta, fh, indent=2)

    print(f"✓ BDV H5/XML generation complete.")
    print(f"  H5 Size: {Path(h5_path).stat().st_size / 1024**2:.2f} MB")


def main():
    print("=" * 60)
    print("MERGE PIPELINE: TIFF -> BDV/HYPERSTACK")
    print("=" * 60)

    # 1. Parse Inputs
    if len(sys.argv) != 3:
        sys.exit("Usage: merge.py <metadata.json> <config.json>")

    meta_path = sys.argv[1]
    config_path = sys.argv[2]

    with open(meta_path) as f:
        meta = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    # 2. Configure Settings
    out_cfg = config.get("output", {})
    out_format = out_cfg.get("format", "tiff").lower()  # 'tiff' or 'bdv'

    # Find files
    seg_files = sorted(Path(".").glob("t*_segmented.tif"))
    if not seg_files:
        raise RuntimeError("No segmented files found in work directory!")

    # 3. Determine Dimensions & Resolution
    # Read first file for shape
    with tifffile.TiffFile(str(seg_files[0])) as tf:
        Z = len(tf.pages)
        Y, X = tf.pages[0].shape

    # Calculate resolutions
    scaling = config.get("preprocessing", {}).get("image_scaling", 1.0)

    # Metadata usually stores resolution in microns
    vox_x = meta.get("x_resolution_um", 1.0) / scaling
    vox_y = meta.get("y_resolution_um", 1.0) / scaling
    vox_z = meta.get("imagej", {}).get("spacing", 1.0)  # Z is typically not scaled

    # Handle Y-Flip
    correct_y = out_cfg.get("correct_y", False)
    if isinstance(correct_y, str) and correct_y == "auto":
        need_flip = detect_flip_needed(seg_files[0])
    else:
        need_flip = bool(correct_y)

    print(f"Setup:")
    print(f"  Format: {out_format}")
    print(f"  Dimensions (Z,Y,X): {Z}, {Y}, {X}")
    print(f"  Voxel Size: {vox_z:.3f}, {vox_y:.3f}, {vox_x:.3f}")
    print(f"  Y-Flip: {need_flip}")

    # 4. Create Metadata Dict
    hyperstack_meta = {
        "shape": {"T": len(seg_files), "Z": Z, "Y": Y, "X": X},
        "voxel_size": {"x_um": vox_x, "y_um": vox_y, "z_um": vox_z},
        "dtype": "uint16",
    }

    # 5. Execute Writer
    if out_format in ["bdv", "hdf5", "xml"]:
        write_bdv_hdf5(seg_files, vox_x, vox_y, vox_z, need_flip, hyperstack_meta)
    else:
        # For TIFF, we must load all to RAM to stack
        print("Loading all timepoints into RAM for TIFF stacking...")
        arrays = []
        for f in seg_files:
            arr = tifffile.imread(str(f))
            if need_flip:
                arr = np.flip(arr, axis=1)
            arrays.append(arr)

        img4d = np.stack(arrays, axis=0)  # T, Z, Y, X
        write_tiff_hyperstack(img4d, vox_x, vox_y, vox_z, need_flip, hyperstack_meta)


if __name__ == "__main__":
    main()
