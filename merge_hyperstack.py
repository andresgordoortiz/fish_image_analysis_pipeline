#!/usr/bin/env python3
"""
Merge timepoints into 4D hyperstack.
Supports both processed and segmented images.
Supports TIFF or BigDataViewer HDF5+XML formats.

Usage:
    merge_hyperstack.py <metadata.json> <config.json> [processed|segmented]

IMP Vienna - Andrés Gordo & Guilherme Ventura
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import tifffile

try:
    import pybdv
except ImportError:
    print("Installing pybdv...")
    os.system("micromamba install -y -n microscopy_env -c conda-forge pybdv")
    import pybdv


def detect_flip_needed(first_path):
    """Check TIFF orientation tag to determine if Y-flip is needed."""
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


def write_tiff_hyperstack_streaming(
    seg_files, final_x_res, final_y_res, final_z_spacing, need_flip, hyperstack_meta
):
    """Write timepoints as ImageJ-compatible TIFF hyperstack using streaming (low memory).

    ImageJ hyperstack format:
    - Data written as 2D slices in TZCYX order (T slowest, then Z, then C, then YX)
    - For single channel: write all Z slices for T=0, then all Z for T=1, etc.
    - ImageJ description string in first IFD defines the hyperstack dimensions

    Note: We use manual description instead of imagej=True because tifffile's
    imagej mode overwrites our dimension metadata when streaming 2D slices.
    """

    # Save Metadata
    meta_filename = hyperstack_meta.get("meta_filename", "4D_hyperstack_metadata.json")
    tif_filename = hyperstack_meta.get("tif_filename", "4D_hyperstack.tif")
    with open(meta_filename, "w") as fh:
        json.dump(hyperstack_meta, fh, indent=2)

    T = len(seg_files)
    # Get shape from first file
    with tifffile.TiffFile(str(seg_files[0])) as tf:
        Z = len(tf.pages)
        Y, X = tf.pages[0].shape

    # Estimate file size
    estimated_gb = (T * Z * Y * X * 2) / (1024**3)  # uint16 = 2 bytes
    print(f"Writing {tif_filename} (~{estimated_gb:.2f} GB estimated)...")
    print(f"  Dimensions: T={T}, Z={Z}, Y={Y}, X={X}")
    print(f"  Total 2D images: {T * Z}")
    print(f"  Using streaming mode (writing 2D slices)")

    # Resolution: pixels per micron (TIFF standard is pixels-per-unit)
    resolution = (1.0 / final_y_res, 1.0 / final_x_res)

    # Build ImageJ description string manually
    # This is the ONLY reliable way to set hyperstack dimensions when streaming
    # tifffile's imagej=True mode overwrites slices/frames with channels count
    imagej_description = f"""ImageJ=1.54f
images={T * Z}
channels=1
slices={Z}
frames={T}
hyperstack=true
mode=grayscale
loop=false
spacing={final_z_spacing}
unit=um
min=0.0
max=65535.0
"""

    # Write as sequence of 2D images WITHOUT imagej=True
    # We manually provide the ImageJ description on the first frame
    with tifffile.TiffWriter(tif_filename, bigtiff=True) as tif:
        first_write = True

        for t, file_path in enumerate(seg_files):
            print(f"  Writing timepoint {t + 1}/{T}: {file_path.name}", end="\r")

            # Read single timepoint as 3D volume (Z, Y, X)
            data = tifffile.imread(str(file_path)).astype(np.uint16)

            # Ensure data is 3D (Z, Y, X)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]

            # Flip Y if needed (axis 1 is Y in ZYX)
            if need_flip:
                data = np.flip(data, axis=1)

            # Write each Z-slice as a 2D image
            for z in range(data.shape[0]):
                if first_write:
                    # First frame: include ImageJ description and resolution
                    tif.write(
                        data[z],
                        resolution=resolution,
                        resolutionunit=tifffile.RESUNIT.MICROMETER,
                        description=imagej_description,
                        contiguous=True,
                    )
                    first_write = False
                else:
                    # Subsequent frames: just data
                    tif.write(data[z], contiguous=True)

            # Free memory after each timepoint
            del data

    print(f"\n✓ TIFF Hyperstack Written Successfully")
    print(f"  Total: {T * Z} images = {T} timepoints × {Z} Z-slices")
    print(f"  Open in ImageJ/Fiji: File > Open > {tif_filename}")
    print(f"  Expected sliders: Z (slices: {Z}) and T (frames: {T})")


def write_bdv_hdf5(
    seg_files, final_x_res, final_y_res, final_z_spacing, need_flip, hyperstack_meta
):
    """Write timepoints as BigDataViewer HDF5+XML using pybdv."""
    print("\n" + "=" * 60)
    print("Writing BDV HDF5 + XML (via pybdv.make_bdv)")
    print("=" * 60)

    h5_path = hyperstack_meta.get("h5_filename", "4D_hyperstack.h5")
    xml_path = hyperstack_meta.get("xml_filename", "4D_hyperstack.xml")

    # Clean up existing files to prevent corruption
    if os.path.exists(h5_path):
        os.remove(h5_path)
    if os.path.exists(xml_path):
        os.remove(xml_path)

    # Define standard BDV setup
    setup_id = 0

    # ---------------------------------------------------------
    # HDF5 + XML Generation (all handled by make_bdv)
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

        # CRITICAL: make_bdv handles EVERYTHING - HDF5 creation, downsampling, AND XML generation
        # The XML file is automatically created/updated with each call
        # downscale_mode="nearest" is CRITICAL for segmentation to avoid interpolation artifacts
        pybdv.make_bdv(
            data,
            h5_path,
            downscale_factors=[[1, 1, 1], [2, 2, 2], [4, 4, 4]],
            downscale_mode="nearest",
            resolution=[final_z_spacing, final_y_res, final_x_res],  # (Z, Y, X) order
            unit="um",
            setup_id=setup_id,
            timepoint=t,
            overwrite=(t == 0),  # Overwrite on first timepoint, append on others
        )

    # That's it! make_bdv has already created both the HDF5 and XML files.
    # NO need to call write_xml_metadata() - it's already done!

    print(f"\n✓ BDV generation complete")
    print(f"  make_bdv() automatically created both files:")
    print(f"  - HDF5: {h5_path} ({Path(h5_path).stat().st_size / 1024**2:.2f} MB)")
    print(f"  - XML:  {xml_path}")

    # ---------------------------------------------------------
    # Finalize Metadata
    # ---------------------------------------------------------
    hyperstack_meta["bdv_info"] = {
        "file": h5_path,
        "xml": xml_path,
        "setup_id": setup_id,
        "n_timepoints": T,
        "pybdv_used": True,
        "downscale_mode": "nearest",
        "note": "XML automatically generated by pybdv.make_bdv()",
    }

    meta_filename = hyperstack_meta.get("meta_filename", "4D_hyperstack_metadata.json")
    with open(meta_filename, "w") as fh:
        json.dump(hyperstack_meta, fh, indent=2)


def parse_output_formats(out_cfg):
    """
    Parse the output format configuration.

    Supports:
    - "tiff" or "tif" -> only TIFF
    - "bdv" or "hdf5" or "xml" -> only BDV
    - "both" or "all" -> both formats
    - ["tiff", "bdv"] -> list of formats

    Returns a set of formats to generate: {'tiff', 'bdv'}
    """
    format_spec = out_cfg.get("format", "tiff")

    # Handle list of formats
    if isinstance(format_spec, list):
        formats = set()
        for fmt in format_spec:
            fmt_lower = fmt.lower().strip()
            if fmt_lower in ["tiff", "tif"]:
                formats.add("tiff")
            elif fmt_lower in ["bdv", "hdf5", "xml"]:
                formats.add("bdv")
        return formats if formats else {"tiff"}

    # Handle string format
    format_spec = format_spec.lower().strip()

    if format_spec in ["both", "all"]:
        return {"tiff", "bdv"}
    elif format_spec in ["bdv", "hdf5", "xml"]:
        return {"bdv"}
    else:
        return {"tiff"}


def main():
    print("=" * 60)
    print("MERGE PIPELINE: TIFF -> BDV/HYPERSTACK")
    print("=" * 60)

    # 1. Parse Inputs
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        sys.exit("Usage: merge.py <metadata.json> <config.json> [processed|segmented]")

    meta_path = sys.argv[1]
    config_path = sys.argv[2]
    data_type = sys.argv[3] if len(sys.argv) == 4 else "segmented"

    with open(meta_path) as f:
        meta = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    # 2. Configure Settings
    out_cfg = config.get("output", {})
    output_formats = parse_output_formats(out_cfg)

    # Find files based on data type
    if data_type == "processed":
        file_pattern = "t*_processed.tif"
        label = "processed"
    elif data_type == "segmented":
        file_pattern = "t*_segmented.tif"
        label = "segmented"
    else:
        sys.exit(f"Unknown data_type '{data_type}'. Use 'processed' or 'segmented'.")

    seg_files = sorted(Path(".").glob(file_pattern))
    if not seg_files:
        # Fallback: try any .tif/.tiff file (e.g. raw input when preprocessing is skipped)
        fallback = sorted(
            f for f in Path(".").glob("*.tif")
            if not f.name.startswith("4D_hyperstack")
            and f.name != "shared_metadata.json"
        )
        fallback += sorted(
            f for f in Path(".").glob("*.tiff")
            if not f.name.startswith("4D_hyperstack")
        )
        if fallback:
            print(f"No files matching '{file_pattern}', falling back to {len(fallback)} TIF file(s)")
            seg_files = sorted(set(fallback), key=lambda f: f.name)
        else:
            raise RuntimeError(f"No {label} files matching '{file_pattern}' found in work directory!")

    # Output file naming based on data type
    base_name = f"4D_hyperstack_{label}"
    print(f"Data type: {label}")
    print(f"Found {len(seg_files)} {label} files")

    # 3. Determine Dimensions & Resolution
    # Read first file for shape
    with tifffile.TiffFile(str(seg_files[0])) as tf:
        Z = len(tf.pages)
        Y, X = tf.pages[0].shape

    # Calculate resolutions
    # When preprocessing is skipped, image_scaling was never applied — use 1.0
    skip_preproc = config.get("preprocessing", {}).get("skip_preprocessing", False)
    scaling = 1.0 if skip_preproc else config.get("preprocessing", {}).get("image_scaling", 1.0)

    # Metadata stores the ORIGINAL (pre-processing) resolution in microns
    vox_x = meta.get("x_resolution_um", 1.0) / scaling
    vox_y = meta.get("y_resolution_um", 1.0) / scaling
    original_z_spacing = meta.get("imagej", {}).get("spacing", 1.0)

    # The preprocessing script reslices the image to isotropic voxels.
    # The actual Z spacing after reslicing can be computed from the original
    # vs actual Z slice count. Read actual Z from the segmented files.
    original_z_slices = meta.get("shape", {}).get("dimensions", [Z, Y, X])[0]
    if Z != original_z_slices:
        # Image was resliced to isotropic
        vox_z = original_z_slices * original_z_spacing / Z
        print(f"  Isotropic reslicing detected: Z slices {original_z_slices} -> {Z}")
        print(f"  Z spacing: {original_z_spacing:.4f} -> {vox_z:.4f} µm")
    else:
        vox_z = original_z_spacing

    # Handle Y-Flip
    correct_y = out_cfg.get("correct_y", False)
    if isinstance(correct_y, str) and correct_y == "auto":
        need_flip = detect_flip_needed(seg_files[0])
    else:
        need_flip = bool(correct_y)

    # Format display string
    format_display = " + ".join(sorted(output_formats)).upper()

    print(f"Setup:")
    print(f"  Format(s): {format_display}")
    print(f"  Dimensions (Z,Y,X): {Z}, {Y}, {X}")
    print(f"  Voxel Size: {vox_z:.3f}, {vox_y:.3f}, {vox_x:.3f}")
    print(f"  Y-Flip: {need_flip}")

    # 4. Create Metadata Dict
    hyperstack_meta = {
        "data_type": label,
        "n_timepoints": len(seg_files),
        "shape": {"T": len(seg_files), "Z": Z, "Y": Y, "X": X},
        "voxel_size": {"x_um": vox_x, "y_um": vox_y, "z_um": vox_z},
        "dtype": "uint16",
        "output_formats": list(output_formats),
        "tif_filename": f"{base_name}.tif",
        "h5_filename": f"{base_name}.h5",
        "xml_filename": f"{base_name}.xml",
        "meta_filename": f"{base_name}_metadata.json",
    }

    # 5. Execute Writer(s)
    # Use streaming approach to avoid loading all data into RAM
    if "tiff" in output_formats:
        print("\n" + "=" * 60)
        print("Writing TIFF hyperstack (streaming mode)...")
        print("=" * 60)
        write_tiff_hyperstack_streaming(
            seg_files, vox_x, vox_y, vox_z, need_flip, hyperstack_meta
        )

    if "bdv" in output_formats:
        write_bdv_hdf5(seg_files, vox_x, vox_y, vox_z, need_flip, hyperstack_meta)

    # 6. Final Summary
    print("\n" + "=" * 60)
    print(f"MERGE COMPLETE ({label.upper()})")
    print("=" * 60)
    print(f"Output format(s): {format_display}")
    if "tiff" in output_formats:
        print(f"  ✓ {base_name}.tif")
    if "bdv" in output_formats:
        print(f"  ✓ {base_name}.h5 + {base_name}.xml")
    print(f"  ✓ {base_name}_metadata.json")


if __name__ == "__main__":
    main()
