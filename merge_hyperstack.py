#!/usr/bin/env python3
"""
Merge segmented timepoints into 4D hyperstack (TIFF or BDV/HDF5 format)

FIXED: Proper BDV HDF5 structure with correct dataset hierarchy and XML references
"""

import json
import os
import sys
from pathlib import Path
import numpy as np


def detect_flip_needed(first_path):
    """Auto-detect if Y-axis flip is needed based on TIFF orientation tag."""
    import tifffile

    try:
        with tifffile.TiffFile(str(first_path)) as tf:
            page = tf.pages[0]
            tags = page.tags
            if "Orientation" in tags:
                val = tags["Orientation"].value
                print(f"  TIFF Orientation tag: {val}")
                return val in (3, 4, 7, 8)
    except Exception as e:
        print(f"  Orientation auto-detect error: {e}")
    return False


def write_tiff_hyperstack(
    img4d, final_x_res, final_y_res, final_z_spacing, need_flip, hyperstack_meta
):
    """Write 4D array as TIFF hyperstack with ImageJ metadata."""
    import tifffile

    print("\n" + "=" * 80)
    print("Writing TIFF hyperstack")
    print("=" * 80)

    print(f"\nFinal 4D shape: {img4d.shape} (T, Z, Y, X)")
    print(f"Data type: {img4d.dtype}")
    print(f"Memory size: {img4d.nbytes / (1024**3):.2f} GB")

    with open("4D_hyperstack_metadata.json", "w") as fh:
        json.dump(hyperstack_meta, fh, indent=2)
    print("\n✓ Metadata JSON saved")

    print("\nWriting 4D_hyperstack.tif...")
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
            "LabelImage": True,
            "YFlipApplied": need_flip,
        },
    )

    output_size = Path("4D_hyperstack.tif").stat().st_size / (1024**3)
    print(f"✓ 4D_hyperstack.tif written ({output_size:.2f} GB)")


def write_bdv_hdf5(
    seg_files,
    T,
    Z,
    Y,
    X,
    final_x_res,
    final_y_res,
    final_z_spacing,
    need_flip,
    hyperstack_meta,
):
    """
    Write timepoints as BDV HDF5 + XML with CORRECT structure.
    
    BDV expects:
    /s{setup}/t{timepoint}/s{resolution_level}/cells - actual image data
    /s{setup}/resolutions - dataset (not attribute) with downsample factors
    /s{setup}/subdivisions - dataset (not attribute) with block sizes
    """
    import tifffile

    try:
        import h5py
    except ImportError:
        print("\nh5py not found, installing via micromamba...")
        rc = os.system("micromamba install -y -n microscopy_env h5py >/dev/null 2>&1")
        if rc != 0:
            raise RuntimeError("Failed to install h5py")
        import h5py

    print("\n" + "=" * 80)
    print("Writing BDV HDF5 + XML")
    print("=" * 80)

    h5_fname = "4D_hyperstack.h5"
    xml_fname = "4D_hyperstack.xml"

    # BDV standard: setup 0
    setup_id = 0
    setup_name = f"s{setup_id:02d}"

    # Resolution level 0 (full resolution)
    resolution_level = 0
    
    # Dataset paths for each timepoint
    # Format: /s00/t{timepoint:05d}/s0/cells
    dataset_paths = []
    for t in range(T):
        path = f"/{setup_name}/t{t:05d}/s{resolution_level}/cells"
        dataset_paths.append(path)

    # Chunk size optimization
    chunk_z = min(8, Z)
    chunk_y = min(64, Y)
    chunk_x = min(64, X)
    chunks = (chunk_z, chunk_y, chunk_x)

    print(f"\nHDF5 configuration:")
    print(f"  File: {h5_fname}")
    print(f"  Setup: {setup_name}")
    print(f"  Resolution level: {resolution_level}")
    print(f"  Shape per timepoint: ({Z}, {Y}, {X}) [Z, Y, X]")
    print(f"  Chunks: {chunks}")
    print(f"  Timepoints: {T}")
    print(f"  First dataset path: {dataset_paths[0]}")
    print(f"  Last dataset path: {dataset_paths[-1]}")

    # Test compression
    gzip_ok = True
    try:
        tmp = "tmp_test.h5"
        with h5py.File(tmp, "w") as fh:
            fh.create_dataset(
                "d", data=np.zeros((2,), dtype=np.uint16), compression="gzip"
            )
        os.remove(tmp)
        print("  Compression: gzip (level 4)")
    except Exception as e:
        gzip_ok = False
        print(f"  Compression: disabled ({e})")

    comp = "gzip" if gzip_ok else None
    comp_opts = 4 if gzip_ok else None

    if os.path.exists(h5_fname):
        os.remove(h5_fname)

    print(f"\nCreating HDF5 file with {T} timepoint datasets...")

    with h5py.File(h5_fname, "w") as h5f:
        # Create setup group
        setup_group = h5f.create_group(setup_name)
        
        # Add resolutions dataset at setup level
        # Shape: (num_levels, 3) - each row is [z_downsample, y_downsample, x_downsample]
        resolutions_data = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        setup_group.create_dataset("resolutions", data=resolutions_data, dtype=np.float64)
        print(f"  ✓ Created /{setup_name}/resolutions dataset: {resolutions_data.shape}")
        
        # Add subdivisions dataset at setup level
        # Shape: (num_levels, 3) - each row is [z_blocksize, y_blocksize, x_blocksize]
        subdivisions_data = np.array([[chunk_z, chunk_y, chunk_x]], dtype=np.int32)
        setup_group.create_dataset("subdivisions", data=subdivisions_data, dtype=np.int32)
        print(f"  ✓ Created /{setup_name}/subdivisions dataset: {subdivisions_data.shape}")
        
        # Write each timepoint
        print("\nWriting timepoint data:")
        for t_idx, seg_path in enumerate(seg_files):
            if (t_idx + 1) % 10 == 0 or t_idx == 0 or t_idx == len(seg_files) - 1:
                print(f"  Timepoint {t_idx + 1}/{T}: {seg_path.name} -> {dataset_paths[t_idx]}")

            # Load segmentation mask
            arr = tifffile.imread(str(seg_path)).astype(np.uint16)

            # Validate shape
            if arr.shape != (Z, Y, X):
                raise RuntimeError(
                    f"Dimension mismatch in {seg_path.name}:\n"
                    f"  Expected: ({Z}, {Y}, {X})\n"
                    f"  Got: {arr.shape}"
                )

            # Apply Y-flip if needed
            if need_flip:
                arr = np.flip(arr, axis=1)

            # Create the full path including intermediate groups
            # Path format: /s00/t00000/s0/cells
            dset_path = dataset_paths[t_idx]
            
            # Create dataset with data
            if comp:
                h5f.create_dataset(
                    dset_path,
                    data=arr,
                    dtype=np.uint16,
                    chunks=chunks,
                    compression=comp,
                    compression_opts=comp_opts,
                )
            else:
                h5f.create_dataset(
                    dset_path,
                    data=arr,
                    dtype=np.uint16,
                    chunks=chunks,
                )

        h5f.flush()

    # Verify file size
    h5_size_mb = Path(h5_fname).stat().st_size / (1024**2)
    expected_size_mb = (T * Z * Y * X * 2) / (1024**2)
    print(f"\n✓ HDF5 file written")
    print(f"  File size: {h5_size_mb:.1f} MB")
    print(f"  Expected (uncompressed): {expected_size_mb:.1f} MB")
    if comp:
        compression_ratio = expected_size_mb / h5_size_mb if h5_size_mb > 0 else 0
        print(f"  Compression ratio: {compression_ratio:.2f}x")

    # Generate BDV XML
    print(f"\nGenerating BDV XML: {xml_fname}")

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<SpimData version="0.2">',
        '  <BasePath type="relative">.</BasePath>',
        '  <SequenceDescription>',
        '    <ImageLoader format="bdv.hdf5">',
        f'      <hdf5 type="relative">{h5_fname}</hdf5>',
        '      <Partitions>',
    ]

    # Add partition for each timepoint
    for t in range(T):
        # Remove leading slash for XML
        dset_path_relative = dataset_paths[t].lstrip('/')
        xml_lines.append(
            f'        <Partition path="{dset_path_relative}" timepoint="{t}" setup="{setup_id}" />'
        )

    xml_lines.extend([
        '      </Partitions>',
        '    </ImageLoader>',
        '    <ViewSetups>',
        '      <ViewSetup>',
        f'        <id>{setup_id}</id>',
        f'        <name>channel {setup_id}</name>',
        f'        <size>{X} {Y} {Z}</size>',
        '        <voxelSize>',
        '          <unit>um</unit>',
        f'          <size>{final_x_res} {final_y_res} {final_z_spacing}</size>',
        '        </voxelSize>',
        '        <attributes>',
        f'          <channel>{setup_id}</channel>',
        '        </attributes>',
        '      </ViewSetup>',
        '      <Attributes name="channel">',
        '        <Channel>',
        f'          <id>{setup_id}</id>',
        f'          <name>{setup_id}</name>',
        '        </Channel>',
        '      </Attributes>',
        '    </ViewSetups>',
        '    <Timepoints type="range">',
        '      <first>0</first>',
        f'      <last>{T - 1}</last>',
        '    </Timepoints>',
        '  </SequenceDescription>',
        '  <ViewRegistrations>',
    ])

    # Add registration for each timepoint
    for t in range(T):
        xml_lines.extend([
            f'    <ViewRegistration timepoint="{t}" setup="{setup_id}">',
            '      <ViewTransform type="affine">',
            '        <name>calibration</name>',
            f'        <affine>{final_x_res} 0.0 0.0 0.0 0.0 {final_y_res} 0.0 0.0 0.0 0.0 {final_z_spacing} 0.0</affine>',
            '      </ViewTransform>',
            '    </ViewRegistration>',
        ])

    xml_lines.extend([
        '  </ViewRegistrations>',
        '</SpimData>',
    ])

    xml_content = '\n'.join(xml_lines)

    with open(xml_fname, 'w') as xf:
        xf.write(xml_content)

    print(f"✓ BDV XML written: {xml_fname}")
    print(f"  Setup ID: {setup_id}")
    print(f"  Timepoints: {T}")
    print(f"  Voxel size: {final_x_res:.4f} × {final_y_res:.4f} × {final_z_spacing:.4f} µm")

    # Update metadata
    hyperstack_meta["bdv_info"] = {
        "hdf5_file": h5_fname,
        "xml_file": xml_fname,
        "setup_id": setup_id,
        "setup_name": setup_name,
        "dataset_paths": dataset_paths,
        "resolutions": resolutions_data.tolist(),
        "subdivisions": subdivisions_data.tolist(),
        "structure": "Standard BDV format: /s{setup}/t{timepoint}/s{level}/cells",
    }

    with open("4D_hyperstack_metadata.json", "w") as fh:
        json.dump(hyperstack_meta, fh, indent=2)

    print("✓ Metadata JSON saved")


def main():
    """Main execution function."""
    import tifffile

    print("=" * 80)
    print("MERGE_TO_HYPERSTACK - BDV HDF5 FIXED")
    print("=" * 80)

    if len(sys.argv) != 3:
        print(
            "Usage: merge_hyperstack.py <metadata.json> <config.json>", file=sys.stderr
        )
        sys.exit(1)

    metadata_path = sys.argv[1]
    config_path = sys.argv[2]

    print("\nLoading metadata...")
    with open(metadata_path, "r") as fm:
        meta = json.load(fm)

    print(f"Metadata source: {meta.get('voxel_size_source', 'unknown')}")
    print(f"ROI cropped: {meta.get('was_roi_cropped', False)}")

    with open(config_path, "r") as fc:
        config = json.load(fc)

    out_cfg = config.get("output", {})
    out_format = out_cfg.get("format", "tiff").lower()
    correct_y_cfg = out_cfg.get("correct_y", False)

    seg_files = sorted(Path(".").glob("t*_segmented.tif"))
    if not seg_files:
        raise RuntimeError("No segmented files found (expected t*_segmented.tif)")

    print(f"\nFound {len(seg_files)} segmented files")
    print(f"  First: {seg_files[0].name}")
    print(f"  Last:  {seg_files[-1].name}")

    preprocessing_cfg = config.get("preprocessing", {})
    scaling = preprocessing_cfg.get("image_scaling", 1.0)
    print(f"Preprocessing scaling factor: {scaling}")

    print("\nReading dimensions from first segmented file...")
    with tifffile.TiffFile(str(seg_files[0])) as tf:
        Z = len(tf.pages)
        Y, X = tf.pages[0].shape
        dtype = tf.pages[0].dtype

    print(f"Dimensions: Z={Z}, Y={Y}, X={X}")
    print(f"Data type: {dtype}")

    print("\nValidating dimension consistency across all timepoints...")
    for idx, p in enumerate(seg_files):
        arr = tifffile.imread(str(p))
        if arr.shape != (Z, Y, X):
            raise RuntimeError(
                f"Dimension mismatch in {p.name} (timepoint {idx}):\n"
                f"  Expected: ({Z}, {Y}, {X})\n"
                f"  Got: {arr.shape}"
            )
        if arr.size == 0:
            raise RuntimeError(f"Empty array read from {p.name}")

    print(f"✓ All {len(seg_files)} timepoints have consistent dimensions")

    T = len(seg_files)

    print("\nCalculating final voxel sizes...")
    original_x = meta.get("x_resolution_um", 1.0)
    original_y = meta.get("y_resolution_um", 1.0)
    original_z = meta.get("imagej", {}).get("spacing", 1.0)

    final_x_res = original_x / scaling
    final_y_res = original_y / scaling
    final_z_spacing = original_z

    print(f"Original: {original_x:.4f} × {original_y:.4f} × {original_z:.4f} µm")
    print(f"Scaling: {scaling}")
    print(f"Final: {final_x_res:.4f} × {final_y_res:.4f} × {final_z_spacing:.4f} µm")

    need_flip = False
    if isinstance(correct_y_cfg, str) and correct_y_cfg.lower() == "auto":
        print("\nAuto-detecting Y-axis flip...")
        need_flip = detect_flip_needed(seg_files[0])
    elif bool(correct_y_cfg):
        need_flip = True
        print("\nY-axis flip: ENABLED (manual)")
    else:
        print("\nY-axis flip: DISABLED")

    print(f"Y-flip will be applied: {need_flip}")

    hyperstack_meta = {
        "shape": {"axes": "TZYX", "T": int(T), "Z": int(Z), "Y": int(Y), "X": int(X)},
        "voxel_size": {
            "x_um": float(final_x_res),
            "y_um": float(final_y_res),
            "z_um": float(final_z_spacing),
            "unit": "um",
            "source": meta.get("voxel_size_source", "unknown"),
        },
        "dtype": "uint16",
        "n_timepoints": int(T),
        "y_correction_applied": bool(need_flip),
        "processing": {
            "output_format": out_format,
            "preprocessing_scaling": float(scaling),
            "roi_cropped": bool(meta.get("was_roi_cropped", False)),
        },
    }

    if out_format in ("tiff", "imagej", "hyperstack"):
        print(f"\nLoading {T} timepoints into memory...")
        all_arrs = []
        for idx, p in enumerate(seg_files):
            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(seg_files) - 1:
                print(f"  Loading timepoint {idx + 1}/{T}: {p.name}")

            arr = tifffile.imread(str(p))

            if need_flip:
                arr = np.flip(arr, axis=1)

            all_arrs.append(arr.astype(np.uint16))

        print(f"\nStacking into 4D array...")
        img4d = np.stack(all_arrs, axis=0)

        write_tiff_hyperstack(
            img4d, final_x_res, final_y_res, final_z_spacing, need_flip, hyperstack_meta
        )

    elif out_format in ("bdv", "bigdataviewer", "hdf5"):
        write_bdv_hdf5(
            seg_files,
            T,
            Z,
            Y,
            X,
            final_x_res,
            final_y_res,
            final_z_spacing,
            need_flip,
            hyperstack_meta,
        )

    else:
        raise RuntimeError(f"Unsupported output.format: {out_format}")

    print("\n" + "=" * 80)
    print("✓ MERGE_TO_HYPERSTACK: SUCCESS")
    print("=" * 80)
    print(f"Format: {out_format}")
    print(f"Timepoints: {T}")
    print(f"Dimensions: {Z} × {Y} × {X} (Z × Y × X)")
    print(f"Voxel size: {final_x_res:.4f} × {final_y_res:.4f} × {final_z_spacing:.4f} µm")
    print(f"Y-flip applied: {need_flip}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)