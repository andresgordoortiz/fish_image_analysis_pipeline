#!/usr/bin/env nextflow

/*
 * SPIM 4D Image Processing Pipeline
 * IMP Vienna - Andrés Gordo & Guilherme Ventura
 *
 * Processes lightsheet microscopy data: deconvolution, segmentation, and merging
 * into a BigDataViewer-compatible 4D stack.
 */

nextflow.enable.dsl=2

import java.nio.file.FileSystems
import java.nio.file.Files
import java.nio.file.Paths

// Parameters - only config_json is required, everything else comes from it
params.config_json = null
params.preprocessing_script = './spim_pipeline_fixed.py'
params.merge_script = './merge_hyperstack.py'
params.benchmark_script = './benchmark_pipeline.py'
params.prep_ultrack_script = './prep_ultrack_cellpose.py'
params.help = false

if (params.help) {
    log.info """
    SPIM 4D Image Processing Pipeline
    ---------------------------------
    Usage: nextflow run spim_pipeline.nf --config_json config.json

    All settings are in config.json.
    Submit with: sbatch submit_pipeline.sh
    """.stripIndent()
    exit 0
}

if (!params.config_json) {
    log.error "Missing --config_json parameter"
    exit 1
}

if (!file(params.preprocessing_script).exists()) {
    log.error "Preprocessing script not found: ${params.preprocessing_script}"
    exit 1
}

// Load configuration from JSON
// Pre-process the file to handle invalid backslash escapes (e.g. "Position\ 5")
// which are common when users copy shell-escaped paths into JSON
def loadConfig(json_path) {
    def jsonSlurper = new groovy.json.JsonSlurper()
    // Use Nextflow's file() for proper path resolution relative to the launch/project directory.
    // Do NOT use Java's new File() — its CWD may differ from the Nextflow launch directory.
    def config_file = file(json_path.toString())
    if (!config_file.exists()) {
        log.error "Config file not found: ${json_path} (resolved to: ${config_file})"
        exit 1
    }
    log.info "Loading config from: ${config_file}"
    def raw = config_file.text
    // Strip UTF-8 BOM if present (common when files are edited on Windows)
    if (raw.length() > 0 && raw.charAt(0) == (char) 0xFEFF) {
        raw = raw.substring(1)
    }
    raw = raw.trim()
    // Remove backslashes that aren't valid JSON escapes (\\, \", \/, \b, \f, \n, \r, \t, \uXXXX)
    // This turns "Position\ 5" into "Position 5"
    raw = raw.replaceAll('\\\\(?![\\\\"/bfnrtu])', '')
    try {
        return jsonSlurper.parseText(raw)
    } catch (Exception e) {
        log.error "Failed to parse config JSON: ${config_file}"
        log.error "First 200 characters of file content: ${raw.take(200)}"
        log.error "Parse error: ${e.message}"
        exit 1
    }
}

config = loadConfig(params.config_json)

// Extract main parameters from config
// Sanitize directory paths: strip backslash escaping and trailing slashes
// Users sometimes write "Position\ 5" in JSON instead of "Position 5"
def sanitizePath(String p) {
    return p.replaceAll('\\\\', '').replaceAll('\\/', '/').replaceAll('/+$', '')
}

params.input_dir = sanitizePath(config.input.directory)
params.output_dir = sanitizePath(config.output.directory)
params.channel = config.input.channel
params.container = config.system?.container_image ?: 'library://andresgordoortiz/spim_imp/python_packages_spim:sha256.6ef173bb45b113a36deae4315200cd8f311de2d7108b4b73e8f17a12cffe7559'
params.fiji_container = config.system?.fiji_container_image ?: 'docker://fiji/fiji:20220415'
params.ultrack_container = config.tracking?.ultrack_container ?: null

// Validate input directory (use file() for Nextflow-native path resolution)
def input_dir_file = file(params.input_dir)
if (!input_dir_file.exists()) {
    log.error "Input directory not found: ${params.input_dir} (resolved to: ${input_dir_file})"
    log.error "If the path contains spaces, use plain spaces in config.json (not backslash-escaped)"
    exit 1
}

// Set defaults for optional config sections
if (!config.containsKey('voxel_size')) {
    config.voxel_size = [auto_detect: true]
}
if (!config.containsKey('roi_cropping')) {
    config.roi_cropping = [enabled: false]
}

// Validate ROI file if cropping is enabled
if (config.roi_cropping.enabled) {
    def roi_path = config.roi_cropping.containsKey('roi_path') ? sanitizePath(config.roi_cropping.roi_path) : null
    if (roi_path) { config.roi_cropping.roi_path = roi_path }
    if (!roi_path || !file(roi_path).exists()) {
        log.error "ROI file not found: ${roi_path} (resolved to: ${roi_path ? file(roi_path) : 'null'})"
        exit 1
    }
}

// Set defaults for skip_merge and downscale_labels
def skip_merge = config.output?.skip_merge ?: false
def skip_segmentation = config.segmentation?.enabled == false
def downscale_labels = config.segmentation?.downscale_labels != null ? config.segmentation.downscale_labels : 1.0

// Set defaults for tracking (ultrack)
if (!config.containsKey('tracking')) {
    config.tracking = [enabled: false]
}
if (!config.tracking.containsKey('prep')) {
    config.tracking.prep = [:]
}
if (!config.tracking.prep.containsKey('fg_sigma'))        { config.tracking.prep.fg_sigma = 5.0 }
if (!config.tracking.prep.containsKey('raw_sigma'))       { config.tracking.prep.raw_sigma = 1.0 }
if (!config.tracking.prep.containsKey('boundary_width'))  { config.tracking.prep.boundary_width = 1 }
if (!config.tracking.prep.containsKey('min_area'))        { config.tracking.prep.min_area = 10 }
def skip_tracking = config.tracking?.enabled != true

// Validate tracking config if enabled
if (!skip_tracking) {
    if (!params.ultrack_container) {
        log.error "Tracking enabled but no ultrack_container specified in config.tracking"
        exit 1
    }
    if (!file(params.ultrack_container).exists()) {
        log.error "Ultrack container not found: ${params.ultrack_container}"
        exit 1
    }
    def toml_path = config.tracking.ultrack_config_toml ?: './ultrack_config.toml'
    if (!file(toml_path).exists()) {
        log.error "Ultrack config TOML not found: ${toml_path}"
        exit 1
    }
    // Tracking requires segmentation + merge
    if (skip_segmentation) {
        log.error "Tracking requires segmentation (segmentation.enabled=true)"
        exit 1
    }
    if (skip_merge) {
        log.error "Tracking requires merge (output.skip_merge=false) to produce hyperstacks"
        exit 1
    }
    // ultrack prep uses tifffile, so TIFF format is required (not BDV-only)
    def out_format = config.output?.format ?: 'tiff'
    def has_tiff = (out_format instanceof List) ? out_format.any { it.toLowerCase() in ['tiff', 'tif'] }
                   : out_format.toLowerCase() in ['tiff', 'tif', 'both', 'all']
    if (!has_tiff) {
        log.error "Tracking requires TIFF output format (output.format must include 'tiff'). Current: ${out_format}"
        exit 1
    }
}

// Set defaults for new optional parameters
if (!config.preprocessing.deconvolution.containsKey('padding_mode')) {
    config.preprocessing.deconvolution.padding_mode = 'reflect'
}
if (!config.preprocessing.deconvolution.containsKey('edge_mask_px')) {
    config.preprocessing.deconvolution.edge_mask_px = 0
}
if (!config.preprocessing.deconvolution.containsKey('edge_taper_width')) {
    config.preprocessing.deconvolution.edge_taper_width = 0
}
if (!config.preprocessing.postprocessing.containsKey('clahe_clip_limit')) {
    config.preprocessing.postprocessing.clahe_clip_limit = 0.01
}
if (!config.preprocessing.postprocessing.containsKey('clahe_post_smooth')) {
    config.preprocessing.postprocessing.clahe_post_smooth = 0.0
}
if (!config.preprocessing.postprocessing.containsKey('mask_border_px')) {
    config.preprocessing.postprocessing.mask_border_px = 10
}
if (!config.preprocessing.containsKey('z_correction_method')) {
    config.preprocessing.z_correction_method = 'p75'
}
if (!config.preprocessing.postprocessing.containsKey('clahe_dual_axis')) {
    config.preprocessing.postprocessing.clahe_dual_axis = true
}
if (!config.preprocessing.containsKey('destripe')) {
    config.preprocessing.destripe = [enabled: true, sigma_long: 64, sigma_short: 2]
}
if (!config.segmentation.containsKey('anisotropy')) {
    config.segmentation.anisotropy = null
}
if (!config.segmentation.containsKey('stitch_threshold')) {
    config.segmentation.stitch_threshold = null
}
if (!config.segmentation.containsKey('min_size')) {
    config.segmentation.min_size = 15
}

// Validate downscale_labels range
if (downscale_labels < 0.1 || downscale_labels > 1.0) {
    log.error "downscale_labels must be between 0.1 and 1.0, got: ${downscale_labels}"
    exit 1
}

// Pipeline startup info
def voxel_info = config.voxel_size.auto_detect ? "Auto-detect" : "Manual: ${config.voxel_size.x_um} x ${config.voxel_size.y_um} x ${config.voxel_size.z_um} µm"
def roi_info = config.roi_cropping.enabled ? "Enabled" : "Disabled"
def merge_info = skip_merge ? "SKIPPED" : "Enabled"
def downscale_info = downscale_labels < 1.0 ? "${downscale_labels} (Fiji nearest-neighbor)" : "Disabled"
def edge_mask_info = config.preprocessing.deconvolution.edge_mask_px > 0 ? "${config.preprocessing.deconvolution.edge_mask_px}px" : "Disabled"
def edge_taper_info = config.preprocessing.deconvolution.edge_taper_width > 0 ? "${config.preprocessing.deconvolution.edge_taper_width}px" : "Disabled"
def seg_mode_info = config.segmentation.do_3d ? "3D" : (config.segmentation.stitch_threshold != null ? "2D+Stitch(${config.segmentation.stitch_threshold})" : "2D")
def tracking_info = skip_tracking ? "SKIPPED" : "Enabled (ultrack)"

log.info """
================================================
SPIM Pipeline - IMP Vienna
================================================
Input        : ${params.input_dir}
Output       : ${params.output_dir}
Channel      : ${params.channel}
ROI Cropping : ${roi_info}
Voxel Size   : ${voxel_info}
Merge        : ${merge_info}
Downscale    : ${downscale_info}
Pad mode     : ${config.preprocessing.deconvolution.padding_mode}
Edge taper   : ${edge_taper_info}
Edge mask    : ${edge_mask_info}
Seg mode     : ${seg_mode_info}
Tracking     : ${tracking_info}
================================================
""".stripIndent()

// Process: ROI-based Cropping (optional)

process CROP_WITH_ROI {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/00_cropped",
        mode: 'copy',
        pattern: "*_cropped.tif"

    publishDir "${params.output_dir}/logs/cropping",
        mode: 'copy',
        pattern: "*.log"

    container params.container  // Use main container, not Fiji!

    input:
    tuple val(timepoint), path(image_file)
    path roi_file

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_cropped.tif"), emit: cropped
    path "t${String.format('%04d', timepoint)}_crop.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    def roi_filename = roi_file.name
    """
    #!/bin/bash
    set -euo pipefail

    # Redirect all output to log
    exec > >(tee t${t_formatted}_crop.log) 2>&1

    echo "============================================"
    echo "Python-based ROI Cropping"
    echo "Timepoint: ${timepoint}"
    echo "Input: ${filename}"
    echo "ROI: ${roi_filename}"
    echo "============================================"
    echo ""

    # Activate environment
    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "Python version:"
    python3 --version
    echo ""

    # Install read-roi if not present (runtime installation)
    echo "Checking/installing read-roi..."
    python3 -c "import read_roi" 2>/dev/null || {
        echo "Installing read-roi..."
        pip install read-roi --break-system-packages
    }
    echo "✓ read-roi available"
    echo ""

    # Run Python cropping script
    python3 << 'PYTHON_CROP_SCRIPT'
import sys
import tifffile
import numpy as np
from read_roi import read_roi_file
from pathlib import Path

print("="*60)
print("Starting ROI-based cropping...")
print("="*60)

try:
    # File paths
    input_file = '${filename}'
    roi_file = '${roi_filename}'
    output_file = 't${t_formatted}_cropped.tif'

    print(f"Input: {input_file}")
    print(f"ROI: {roi_file}")
    print(f"Output: {output_file}")
    print("")

    # Check files exist
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    if not Path(roi_file).exists():
        print(f"ERROR: ROI file not found: {roi_file}")
        sys.exit(1)

    # Load ROI
    print("Loading ROI file...")
    roi_data = read_roi_file(roi_file)

    if not roi_data:
        print("ERROR: No ROI data found in file")
        sys.exit(1)

    # Get first ROI (ImageJ ROI files can contain multiple ROIs)
    roi_name = list(roi_data.keys())[0]
    roi = roi_data[roi_name]

    print(f"ROI name: {roi_name}")
    print(f"ROI type: {roi.get('type', 'unknown')}")

    # Extract bounds
    # ImageJ ROI format uses 'left', 'top', 'width', 'height'
    x = int(roi['left'])
    y = int(roi['top'])
    width = int(roi['width'])
    height = int(roi['height'])

    print(f"ROI bounds:")
    print(f"  X: {x}")
    print(f"  Y: {y}")
    print(f"  Width: {width}")
    print(f"  Height: {height}")
    print("")

    # Load image
    print("Loading image...")
    img = tifffile.imread(input_file)
    print(f"Original image shape: {img.shape}")
    print(f"Original image dtype: {img.dtype}")

    # Determine format
    if img.ndim == 3:
        axes = 'ZYX'
        nz, ny, nx = img.shape
    elif img.ndim == 2:
        axes = 'YX'
        ny, nx = img.shape
        nz = 1
    else:
        print(f"ERROR: Unexpected image dimensions: {img.ndim}")
        sys.exit(1)

    print(f"Image axes: {axes}")
    print("")

    # Validate ROI bounds
    if x < 0 or y < 0:
        print(f"ERROR: ROI has negative coordinates: x={x}, y={y}")
        sys.exit(1)

    if x + width > nx:
        print(f"ERROR: ROI extends beyond image width: {x + width} > {nx}")
        sys.exit(1)

    if y + height > ny:
        print(f"ERROR: ROI extends beyond image height: {y + height} > {ny}")
        sys.exit(1)

    print("✓ ROI bounds valid")
    print("")

    # Crop image
    print("Cropping image...")
    if img.ndim == 3:
        # 3D stack: crop in XY, keep all Z
        cropped = img[:, y:y+height, x:x+width]
    else:
        # 2D image
        cropped = img[y:y+height, x:x+width]

    print(f"Cropped shape: {cropped.shape}")
    print(f"Size reduction: {img.size / cropped.size:.2f}x")
    print("")

    # Get metadata from original image
    print("Preserving metadata...")
    with tifffile.TiffFile(input_file) as tif:
        # Get resolution tags if present
        metadata = {}
        if tif.pages:
            page = tif.pages[0]
            tags = page.tags

            if 'XResolution' in tags:
                metadata['resolution_x'] = tags['XResolution'].value
            if 'YResolution' in tags:
                metadata['resolution_y'] = tags['YResolution'].value

            # ImageJ metadata
            if tif.imagej_metadata:
                imagej_meta = tif.imagej_metadata.copy()
                # Update dimensions but keep spacing
                if 'slices' in imagej_meta:
                    imagej_meta['slices'] = cropped.shape[0] if cropped.ndim == 3 else 1
                metadata['imagej'] = imagej_meta

    # Save cropped image with metadata
    print(f"Saving to: {output_file}")

    save_kwargs = {
        'data': cropped,
        'photometric': 'minisblack'
    }

    # Add resolution if we have it
    if 'resolution_x' in metadata and 'resolution_y' in metadata:
        x_res = metadata['resolution_x']
        y_res = metadata['resolution_y']
        save_kwargs['resolution'] = (x_res[0]/x_res[1], y_res[0]/y_res[1])
        save_kwargs['metadata'] = {'unit': 'um'}
        print(f"✓ Preserving XY resolution: {save_kwargs['resolution']}")

    # Add ImageJ metadata if we have it
    if 'imagej' in metadata:
        save_kwargs['imagej'] = True
        save_kwargs['metadata'] = metadata['imagej']
        print(f"✓ Preserving ImageJ metadata")

    tifffile.imwrite(output_file, **save_kwargs)

    # Verify output
    if not Path(output_file).exists():
        print(f"ERROR: Output file was not created")
        sys.exit(1)

    output_size = Path(output_file).stat().st_size / (1024**2)  # MB
    print(f"✓ Output file created: {output_size:.2f} MB")
    print("")

    print("="*60)
    print("✓ ROI cropping completed successfully!")
    print("="*60)
    print(f"Original: {img.shape}")
    print(f"Cropped:  {cropped.shape}")
    print(f"ROI: [{x}:{x+width}, {y}:{y+height}]")

except Exception as e:
    print("")
    print("="*60)
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print("="*60)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_CROP_SCRIPT

    # Check exit status
    if [ \$? -ne 0 ]; then
        echo ""
        echo "ERROR: Python cropping script failed"
        exit 1
    fi

    # Final verification
    if [ ! -f "t${t_formatted}_cropped.tif" ]; then
        echo ""
        echo "ERROR: Output file not created despite script success"
        echo "Directory contents:"
        ls -lha
        exit 1
    fi

    echo ""
    echo "✓ Cropping completed for timepoint ${timepoint}"
    echo "✓ Output: t${t_formatted}_cropped.tif"
    FILE_SIZE=\$(du -h t${t_formatted}_cropped.tif | cut -f1)
    echo "✓ File size: \$FILE_SIZE"
    """
}

// ============================================================================
// PROCESS: Extract and Configure Metadata
// ============================================================================

process EXTRACT_METADATA {
    tag "Extracting metadata"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/metadata",
        mode: 'copy',
        pattern: "*.json"

    input:
    tuple val(timepoint), path(image_file)
    val was_cropped

    output:
    path "shared_metadata.json", emit: metadata

    container params.container

    script:
    def filename = image_file.name
    def auto_detect = config.voxel_size.auto_detect
    def manual_x = config.voxel_size.get('x_um', 0.325)
    def manual_y = config.voxel_size.get('y_um', 0.325)
    def manual_z = config.voxel_size.get('z_um', 1.0)
    def auto_detect_py = auto_detect ? 'True' : 'False'
    def was_cropped_py = was_cropped ? 'True' : 'False'
    """
    #!/usr/bin/env bash
    set -euo pipefail

    echo "============================================"
    echo "Extracting/Configuring metadata from: ${filename}"
    echo "Image was ROI-cropped: ${was_cropped}"
    echo "Voxel size mode: ${auto_detect ? 'AUTO-DETECT' : 'MANUAL'}"
    echo "============================================"

    # Initialize micromamba
    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"

    echo "Activating microscopy_env..."
    micromamba activate microscopy_env

    echo "Python version:"
    python3 --version

    echo "Checking tifffile installation:"
    python3 -c "import tifffile; print(f'tifffile version: {tifffile.__version__}')"

    echo ""
    echo "Running metadata extraction/configuration..."

    # Create Python script for metadata handling
    cat > extract_metadata.py << 'PYTHON_SCRIPT'
import sys
import json
import traceback

print("Starting metadata extraction/configuration...", file=sys.stderr)

try:
    import tifffile
    import numpy as np
    from pathlib import Path

    print("Imports successful", file=sys.stderr)

    filename = '${filename}'
    auto_detect = ${auto_detect_py}
    was_cropped = ${was_cropped_py}
    manual_x = ${manual_x}
    manual_y = ${manual_y}
    manual_z = ${manual_z}

    print(f"Processing file: {filename}", file=sys.stderr)
    print(f"Auto-detect mode: {auto_detect}", file=sys.stderr)
    print(f"Was ROI-cropped: {was_cropped}", file=sys.stderr)

    if not Path(filename).exists():
        print(f"ERROR: File not found: {filename}", file=sys.stderr)
        sys.exit(1)

    print(f"File exists, size: {Path(filename).stat().st_size} bytes", file=sys.stderr)

    # Open TIFF without loading data
    print("Opening TIFF file...", file=sys.stderr)
    with tifffile.TiffFile(filename) as tif:
        metadata = {}

        # Get number of pages (Z slices) WITHOUT loading data
        n_pages = len(tif.pages)
        print(f"Number of pages: {n_pages}", file=sys.stderr)

        # Get image dimensions from first page
        first_page = tif.pages[0]
        height = first_page.shape[0]
        width = first_page.shape[1]
        print(f"Image dimensions: {height} x {width}", file=sys.stderr)

        # Determine voxel sizes based on auto_detect setting
        if auto_detect:
            print("Using AUTO-DETECT mode for voxel sizes", file=sys.stderr)

            # ImageJ metadata for Z spacing
            if tif.imagej_metadata:
                print("Found ImageJ metadata", file=sys.stderr)
                z_spacing = tif.imagej_metadata.get('spacing', 1.0)
                unit = tif.imagej_metadata.get('unit', 'micron')
                axes = tif.imagej_metadata.get('axes', 'ZYX')
                slices = tif.imagej_metadata.get('slices', n_pages)
            else:
                print("No ImageJ metadata, using defaults", file=sys.stderr)
                z_spacing = 1.0
                unit = 'micron'
                axes = 'ZYX'
                slices = n_pages

            # TIFF tags for XY resolution
            tags = first_page.tags
            if 'XResolution' in tags:
                x_num, x_denom = tags['XResolution'].value
                x_resolution_um = x_denom / x_num if x_num != 0 else 1.0
            else:
                x_resolution_um = 1.0
                print("Warning: No XResolution tag found, using 1.0 µm", file=sys.stderr)

            if 'YResolution' in tags:
                y_num, y_denom = tags['YResolution'].value
                y_resolution_um = y_denom / y_num if y_num != 0 else 1.0
            else:
                y_resolution_um = 1.0
                print("Warning: No YResolution tag found, using 1.0 µm", file=sys.stderr)

            metadata['voxel_size_source'] = 'auto_detected'

        else:
            print("Using MANUAL mode for voxel sizes", file=sys.stderr)
            print(f"  X: {manual_x} µm", file=sys.stderr)
            print(f"  Y: {manual_y} µm", file=sys.stderr)
            print(f"  Z: {manual_z} µm", file=sys.stderr)

            x_resolution_um = manual_x
            y_resolution_um = manual_y
            z_spacing = manual_z
            unit = 'um'
            axes = 'ZYX'
            slices = n_pages

            metadata['voxel_size_source'] = 'manual_override'

        # IMPORTANT: ROI cropping does NOT change voxel spacing
        # It only changes the number of pixels in X and Y dimensions
        # The physical size of each pixel remains the same
        metadata['was_roi_cropped'] = was_cropped
        if was_cropped:
            print("NOTE: Image was ROI-cropped. Voxel spacing unchanged, but XY dimensions reduced.", file=sys.stderr)

        # Store voxel sizes
        metadata['x_resolution_um'] = x_resolution_um
        metadata['y_resolution_um'] = y_resolution_um

        # ImageJ-compatible metadata
        metadata['imagej'] = {
            'spacing': z_spacing,
            'unit': unit,
            'axes': axes,
            'slices': slices,
        }

        # Image dimensions - from page info, not loading data
        metadata['shape'] = {
            'axes': 'ZYX',
            'dimensions': [n_pages, height, width]
        }

        # Data type from first page
        metadata['dtype'] = str(first_page.dtype)

        # Software info
        tags = first_page.tags
        if 'Software' in tags:
            metadata['software'] = tags['Software'].value

    # Save metadata
    print("Saving metadata to JSON...", file=sys.stderr)
    with open('shared_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\\nMetadata Configuration:")
    print(f"  Source: {metadata['voxel_size_source']}")
    print(f"  ROI cropped: {was_cropped}")
    print(f"  Image shape: {metadata['shape']['dimensions']} (ZYX)")
    print(f"  Voxel size: {x_resolution_um:.4f} x {y_resolution_um:.4f} x {z_spacing:.4f} µm")
    print("\\nFull metadata:")
    print(json.dumps(metadata, indent=2))

    print("\\nSUCCESS: Metadata extraction/configuration completed", file=sys.stderr)

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

    # Run the Python script
    python3 extract_metadata.py

    # Verify output was created
    if [ ! -f "shared_metadata.json" ]; then
        echo "ERROR: shared_metadata.json was not created"
        exit 1
    fi

    echo ""
    echo "Metadata extraction/configuration completed successfully"
    """
}

// ============================================================================
// PROCESS: Preprocess and Deconvolve Single Timepoint
// ============================================================================

process PREPROCESS_DECONVOLVE {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/01_preprocessed",
        mode: 'copy',
        pattern: "*_processed.tif"

    publishDir "${params.output_dir}/logs/preprocessing",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json
    path preproc_script
    val preprocess_config

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_processed.tif"), emit: processed
    path "t${String.format('%04d', timepoint)}_preprocess.log", emit: log

    script:
    def cfg = preprocess_config
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    def config_json_str = groovy.json.JsonOutput.toJson(cfg).replace("'", "\\'")
    def script_name = preproc_script.name
    """
    #!/bin/bash
    set -euo pipefail

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "============================================"
    echo "Preprocessing timepoint: ${timepoint}"
    echo "File: ${filename}"
    echo "Preprocessing script: ${script_name}"
    echo "============================================"

    # Verify script is present
    if [ ! -f "${script_name}" ]; then
        echo "ERROR: Preprocessing script not found: ${script_name}"
        echo "Contents of work directory:"
        ls -lh
        exit 1
    fi

    # Run preprocessing with all parameters from config
    python3 << 'PYTHON_EOF'
import json
import sys
import os
import subprocess

# Load metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load config - parse from JSON string to handle booleans correctly
config = json.loads('${config_json_str}')

# Get voxel sizes from metadata (these are already configured - auto-detected or manual)
# NOTE: ROI cropping does not change voxel sizes, only image dimensions
xy_pixel = metadata['x_resolution_um']  # Use configured X resolution
z_pixel = metadata['imagej']['spacing']  # Use configured Z spacing

print(f"Using voxel sizes from metadata:")
print(f"  XY pixel size: {xy_pixel:.4f} µm")
print(f"  Z pixel size: {z_pixel:.4f} µm")
print(f"  Source: {metadata.get('voxel_size_source', 'unknown')}")
if metadata.get('was_roi_cropped', False):
    print(f"  Note: Image was ROI-cropped (voxel size unchanged)")
print("")

# Build command for preprocessing script
cmd = [
    'python3', '${script_name}',
    '--input_file', '${filename}',
    '--outdir', '.',
    '--psf_path', config['psf_path'],
    '--image_scaling', str(config['image_scaling']),
    '--xy_pixel', str(xy_pixel),  # Pass configured XY voxel size
    '--z_pixel', str(z_pixel),    # Pass configured Z voxel size
    '--niter', str(config['deconvolution']['niter']),
    '--niterz', str(config['deconvolution']['niterz']),
    '--percentile_low', str(config['normalization']['percentile_low']),
    '--percentile_high', str(config['normalization']['percentile_high']),
    '--sigma', str(config['postprocessing']['sigma']),
    '--min_v', str(config['normalization']['min_v']),
    '--max_v', str(config['normalization']['max_v']),
    '--resolution_px0', str(config['background_subtraction']['resolution_px0']),
    '--resolution_pz0', str(config['background_subtraction']['resolution_pz0']),
    '--noise_lvl', str(config['background_subtraction']['noise_lvl']),
    '--padding', str(config['deconvolution']['padding']),
    '--padding_mode', str(config['deconvolution'].get('padding_mode', 'reflect')),
    '--edge_mask_px', str(config['deconvolution'].get('edge_mask_px', 0)),
    '--edge_taper_width', str(config['deconvolution'].get('edge_taper_width', 0)),
    '--clahe_clip_limit', str(config['postprocessing'].get('clahe_clip_limit', 0.01)),
    '--clahe_post_smooth', str(config['postprocessing'].get('clahe_post_smooth', 0.0)),
    '--mask_border_px', str(config['postprocessing'].get('mask_border_px', 10)),
    '--z_correction_method', str(config.get('z_correction_method', 'p75')),
    '--destripe_sigma_long', str(config.get('destripe', {}).get('sigma_long', 64)),
    '--destripe_sigma_short', str(config.get('destripe', {}).get('sigma_short', 2))
]

# Add optional flags from correction_flags
if config['correction_flags'].get('no_clahe', False):
    cmd.append('--no_clahe')
if config['correction_flags'].get('no_z_correction', False):
    cmd.append('--no_z_correction')
if config['correction_flags'].get('no_shading', False):
    cmd.append('--no_shading')
if not config['postprocessing'].get('clahe_dual_axis', True):
    cmd.append('--no_clahe_xy')
if not config.get('destripe', {}).get('enabled', True):
    cmd.append('--no_destripe')

print("Preprocessing command:", ' '.join(cmd))
print("\\n" + "="*60)

# Execute
result = subprocess.run(cmd, capture_output=True, text=True)

# Save log
with open('t${t_formatted}_preprocess.log', 'w') as f:
    f.write("STDOUT:\\n")
    f.write(result.stdout)
    f.write("\\n\\nSTDERR:\\n")
    f.write(result.stderr)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)

if result.returncode != 0:
    print(f"ERROR: Preprocessing failed with exit code {result.returncode}")
    sys.exit(result.returncode)
PYTHON_EOF

    # Find and rename output to standard format with ROBUST pattern matching
    echo ""
    echo "Finding processed output file..."
    echo "Expected scaling: ${cfg.image_scaling}"
    echo "All TIF files in directory:"
    ls -lh *.tif 2>/dev/null || echo "No .tif files found"
    echo ""

    # Try multiple patterns to find the output
    ORIGINAL_OUTPUT=""

    # Pattern 1: Standard scaling string
    if [ -z "\$ORIGINAL_OUTPUT" ]; then
        SCALING_STR=\$(echo "${cfg.image_scaling}" | sed 's/\\.//g')
        echo "Trying pattern 1: *_\${SCALING_STR}*.tif"
        ORIGINAL_OUTPUT=\$(find . -maxdepth 1 -name "*_\${SCALING_STR}*.tif" -not -name "${filename}" 2>/dev/null | head -1)
        [ -n "\$ORIGINAL_OUTPUT" ] && echo "  ✓ Found: \$ORIGINAL_OUTPUT"
    fi

    # Pattern 2: Percentage string
    if [ -z "\$ORIGINAL_OUTPUT" ]; then
        SCALING_PCT=\$(python3 -c "print(int(${cfg.image_scaling} * 100))")
        echo "Trying pattern 2: *_\${SCALING_PCT}*.tif"
        ORIGINAL_OUTPUT=\$(find . -maxdepth 1 -name "*_\${SCALING_PCT}*.tif" -not -name "${filename}" 2>/dev/null | head -1)
        [ -n "\$ORIGINAL_OUTPUT" ] && echo "  ✓ Found: \$ORIGINAL_OUTPUT"
    fi

    # Pattern 3: Any new .tif file
    if [ -z "\$ORIGINAL_OUTPUT" ]; then
        echo "Trying pattern 3: any new .tif file (not input)"
        ORIGINAL_OUTPUT=\$(find . -maxdepth 1 -name "*.tif" -not -name "${filename}" -newer "${script_name}" 2>/dev/null | head -1)
        [ -n "\$ORIGINAL_OUTPUT" ] && echo "  ✓ Found: \$ORIGINAL_OUTPUT"
    fi

    # Pattern 4: Any .tif file that's not the input
    if [ -z "\$ORIGINAL_OUTPUT" ]; then
        echo "Trying pattern 4: any .tif file (not input)"
        ORIGINAL_OUTPUT=\$(find . -maxdepth 1 -name "*.tif" -not -name "${filename}" 2>/dev/null | head -1)
        [ -n "\$ORIGINAL_OUTPUT" ] && echo "  ✓ Found: \$ORIGINAL_OUTPUT"
    fi

    if [ -n "\$ORIGINAL_OUTPUT" ]; then
        echo ""
        echo "SUCCESS: Found processed file: \$ORIGINAL_OUTPUT"
        echo "Renaming to: t${t_formatted}_processed.tif"
        mv "\$ORIGINAL_OUTPUT" "t${t_formatted}_processed.tif"
        echo "✓ File renamed successfully"
    else
        echo ""
        echo "ERROR: No processed output found after trying all patterns"
        echo "Directory contents:"
        ls -lha
        exit 1
    fi

    # Restore and update metadata to processed image
    python3 << 'RESTORE_META'
import tifffile
import json

# Load original metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load processed image
img = tifffile.imread('t${t_formatted}_processed.tif')

# Recalculate voxel sizes after scaling AND isotropic reslicing
# The preprocessing script (spim_pipeline_fixed.py) does TWO things:
#   1. Rescales XY by image_scaling (0.5) -> XY voxel size doubles
#   2. Reslices Z to make voxels isotropic -> Z slices are interpolated
#
# After XY scaling:
x_res = metadata['x_resolution_um'] / ${cfg.image_scaling}
y_res = metadata['y_resolution_um'] / ${cfg.image_scaling}
original_z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

# After isotropic reslicing: the preprocessing script interpolates Z so that
# z_spacing matches the scaled XY pixel size. We can compute the actual new
# Z spacing from the original vs processed Z dimensions.
original_z_slices = metadata['shape']['dimensions'][0]  # original Z count
new_z_slices = img.shape[0]  # actual Z count after reslicing

if new_z_slices != original_z_slices:
    # Image was resliced to isotropic - recalculate Z spacing
    z_spacing = original_z_slices * original_z_spacing / new_z_slices
    print(f"Isotropic reslicing detected:")
    print(f"  Z slices: {original_z_slices} -> {new_z_slices}")
    print(f"  Z spacing: {original_z_spacing:.4f} -> {z_spacing:.4f} µm")
else:
    # No reslicing occurred (already isotropic)
    z_spacing = original_z_spacing

print(f"Voxel sizes after preprocessing:")
print(f"  Original: {metadata['x_resolution_um']:.4f} x {metadata['y_resolution_um']:.4f} x {original_z_spacing:.4f} µm")
print(f"  Final:    {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} µm (isotropic)")
if metadata.get('was_roi_cropped', False):
    print(f"  (Image was ROI-cropped before preprocessing)")

# Re-save with preserved metadata
tifffile.imwrite(
    't${t_formatted}_processed.tif',
    img,
    imagej=True,
    resolution=(1.0/x_res, 1.0/y_res),
    metadata={
        'spacing': z_spacing,
        'unit': 'um',
        'axes': 'ZYX',
        'TimePoint': ${timepoint},
        'WasROICropped': metadata.get('was_roi_cropped', False)
    }
)

print(f"Metadata restored for timepoint ${timepoint}")
RESTORE_META

    echo "Preprocessing completed for timepoint ${timepoint}"
    """
}

// ============================================================================
// PROCESS: Cellpose Segmentation
// ============================================================================

process CELLPOSE_SEGMENT {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/02_segmented",
        mode: 'copy',
        pattern: "*_segmented.tif"

    publishDir "${params.output_dir}/logs/segmentation",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(processed_file)
    path metadata_json
    val segment_config
    val image_scaling

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_segmented.tif"), emit: segmented
    path "t${String.format('%04d', timepoint)}_segment.log", emit: log

    script:
    def cfg = segment_config
    def t_formatted = String.format('%04d', timepoint)
    def filename = processed_file.name
    def config_json_str = groovy.json.JsonOutput.toJson(cfg).replace("'", "\\'")
    """
    #!/bin/bash
    set -euo pipefail

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "============================================"
    echo "Segmentation timepoint: ${timepoint}"
    echo "File: ${filename}"
    echo "============================================"

    # Build Cellpose command using Python to handle config properly
    python3 << 'PYTHON_EOF'
import json
import sys
import subprocess

# Load config - parse from JSON string to handle booleans correctly
config = json.loads('${config_json_str}')

# Build Cellpose command
cmd = [
    'cellpose',
    '--image_path', '${filename}',
    '--savedir', '.',
    '--pretrained_model', config['model'],
    '--diameter', str(config['diameter']),
    '--flow_threshold', str(config['flow_threshold']),
    '--cellprob_threshold', str(config['cellprob_threshold']),
    '--verbose'
]

# Add boolean flags
if config.get('use_gpu', False):
    cmd.append('--use_gpu')
if config.get('do_3d', False):
    cmd.append('--do_3D')
if config.get('save_tif', True):
    cmd.append('--save_tif')
if config.get('save_flows', False):
    cmd.append('--save_flows')
if not config.get('save_npy', True):
    cmd.append('--no_npy')

# Add anisotropy if specified
if config.get('anisotropy') is not None:
    cmd.extend(['--anisotropy', str(config['anisotropy'])])

# Add stitch_threshold for 2D+stitch mode (only when do_3d is False)
if config.get('stitch_threshold') is not None and not config.get('do_3d', False):
    cmd.extend(['--stitch_threshold', str(config['stitch_threshold'])])

# Add min_size if specified
if config.get('min_size') is not None:
    cmd.extend(['--min_size', str(config['min_size'])])

print("Running Cellpose:", ' '.join(cmd))
print("")

# Run Cellpose
result = subprocess.run(cmd, capture_output=True, text=True)

# Save log
with open('t${t_formatted}_segment.log', 'w') as f:
    f.write("STDOUT:\\n")
    f.write(result.stdout)
    f.write("\\n\\nSTDERR:\\n")
    f.write(result.stderr)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)

if result.returncode != 0:
    print(f"ERROR: Cellpose failed with exit code {result.returncode}")
    sys.exit(result.returncode)
PYTHON_EOF

    # Find and rename Cellpose output
    CELLPOSE_OUTPUT=\$(find . -maxdepth 1 -name "*_cp_masks.tif" 2>/dev/null | head -1)

    if [ -n "\$CELLPOSE_OUTPUT" ]; then
        echo "Found Cellpose output: \$CELLPOSE_OUTPUT"
        mv "\$CELLPOSE_OUTPUT" "t${t_formatted}_segmented.tif"
        echo "Renamed: \$CELLPOSE_OUTPUT -> t${t_formatted}_segmented.tif"

        # Preserve metadata in segmentation mask
        python3 << 'PRESERVE_MASK_META'
import tifffile
import json
import numpy as np

# Load metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load mask
mask = tifffile.imread("t${t_formatted}_segmented.tif")

# Calculate voxel sizes (accounting for preprocessing scaling AND isotropic reslicing)
x_res = metadata['x_resolution_um'] / ${image_scaling}
y_res = metadata['y_resolution_um'] / ${image_scaling}
original_z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

# After isotropic reslicing, Z spacing changes to match scaled XY pixel size.
# Compute from original vs actual Z dimensions.
original_z_slices = metadata['shape']['dimensions'][0]
new_z_slices = mask.shape[0]

if new_z_slices != original_z_slices:
    z_spacing = original_z_slices * original_z_spacing / new_z_slices
else:
    z_spacing = original_z_spacing

print(f"Segmentation mask voxel size: {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} µm")

# Re-save with metadata
tifffile.imwrite(
    "t${t_formatted}_segmented.tif",
    mask.astype(np.uint16),
    imagej=True,
    resolution=(1.0/x_res, 1.0/y_res),
    metadata={
        'spacing': z_spacing,
        'unit': 'um',
        'axes': 'ZYX',
        'TimePoint': ${timepoint},
        'LabelImage': True,
        'WasROICropped': metadata.get('was_roi_cropped', False)
    }
)
print(f"Metadata preserved in segmentation mask for timepoint ${timepoint}")
PRESERVE_MASK_META

    else
        echo "ERROR: Cellpose output not found"
        echo "Expected file matching pattern: *_cp_masks.tif"
        echo "Directory contents:"
        ls -lh
        exit 1
    fi

    echo "Segmentation completed for timepoint ${timepoint}"
    """
}

// ============================================================================
// PROCESS: Merge All Timepoints into 4D Hyperstack
// ============================================================================

process MERGE_TO_HYPERSTACK {
    tag "Merging ${data_type} -> 4D hyperstack"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/${data_type == 'processed' ? '01_preprocessed' : '02_segmented'}",
        mode: 'copy'

    input:
    tuple val(data_type), path(input_files)
    path metadata_json
    path merge_script

    output:
    val data_type, emit: dtype
    path "4D_hyperstack_${data_type}.tif", emit: hyperstack, optional: true
    path "4D_hyperstack_${data_type}_metadata.json", emit: metadata
    path "4D_hyperstack_${data_type}.h5", emit: h5, optional: true
    path "4D_hyperstack_${data_type}.xml", emit: xml, optional: true

    container params.container

    script:
    // Create properly escaped JSON string for Python heredoc
    def config_json_str = groovy.json.JsonOutput.toJson(config)
    def merge_script_name = merge_script.name
    """
    #!/usr/bin/env bash
    set -euo pipefail

    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "=== MERGE_TO_HYPERSTACK (${data_type}) ==="
    echo "Python version:"
    python3 --version
    echo ""

    # Verify merge script is present
    if [ ! -f "${merge_script_name}" ]; then
        echo "ERROR: Merge script not found: ${merge_script_name}"
        echo "Contents of work directory:"
        ls -lh
        exit 1
    fi

    echo "Merge script: ${merge_script_name}"
    echo "Data type: ${data_type}"
    echo ""

    # Ensure required packages
    echo "Checking required packages..."
    python3 -c "import tifffile, numpy" 2>/dev/null || {
        echo "Installing required packages..."
        micromamba install -y -n microscopy_env tifffile numpy
    }
    echo "✓ Required packages available"
    echo ""

    # Create temporary config file using Python to avoid escaping issues
    echo "Creating temporary config file..."
    python3 << 'PYTHON_CONFIG'
import json

# Parse the JSON string from Groovy
config_str = '''${config_json_str}'''
config_data = json.loads(config_str)

# Write it properly to file
with open('config_temp.json', 'w') as f:
    json.dump(config_data, f, indent=2)

print("✓ Config file created")
PYTHON_CONFIG

    # Verify config file was created
    if [ ! -f "config_temp.json" ]; then
        echo "ERROR: Failed to create config_temp.json"
        exit 1
    fi

    echo "Config file contents (first 10 lines):"
    head -10 config_temp.json
    echo ""

    # Run merge script with data_type argument
    echo "Running merge script for ${data_type} data..."
    python3 "${merge_script_name}" "${metadata_json}" config_temp.json "${data_type}"

    # Check exit status
    if [ \$? -ne 0 ]; then
        echo ""
        echo "ERROR: Merge script failed for ${data_type}"
        exit 1
    fi

    # Verify output files
    echo ""
    echo "Verifying output file(s)..."

    if [ ! -f "4D_hyperstack_${data_type}_metadata.json" ]; then
        echo "ERROR: Metadata file not created"
        exit 1
    fi
    echo "✓ Metadata file created"

    # Check for TIFF or HDF5 output
    if [ -f "4D_hyperstack_${data_type}.tif" ]; then
        FILE_SIZE=\$(du -h "4D_hyperstack_${data_type}.tif" | cut -f1)
        echo "✓ TIFF hyperstack created (size: \$FILE_SIZE)"
    elif [ -f "4D_hyperstack_${data_type}.h5" ]; then
        FILE_SIZE=\$(du -h "4D_hyperstack_${data_type}.h5" | cut -f1)
        echo "✓ HDF5 file created (size: \$FILE_SIZE)"
        if [ -f "4D_hyperstack_${data_type}.xml" ]; then
            echo "✓ BDV XML file created"
        fi
    else
        echo "ERROR: No output file created (expected 4D_hyperstack_${data_type}.tif or .h5)"
        exit 1
    fi

    echo ""
    echo "✓ MERGE_TO_HYPERSTACK (${data_type}) completed successfully"
    """
}

// ============================================================================
// PROCESS: Downscale Segmented Labels (Fiji headless, nearest-neighbor)
// ============================================================================

process DOWNSCALE_SEGMENTATION {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/02_segmented_downscaled",
        mode: 'copy',
        pattern: "*_downscaled.tif"

    publishDir "${params.output_dir}/logs/downscaling",
        mode: 'copy',
        pattern: "*.log"

    container params.fiji_container

    input:
    tuple val(timepoint), path(segmented_file)
    val scale_factor

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_downscaled.tif"), emit: downscaled
    path "t${String.format('%04d', timepoint)}_downscale.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def input_name = segmented_file.name
    def output_name = "t${t_formatted}_downscaled.tif"
    """
    #!/bin/bash
    set -euo pipefail

    exec > >(tee t${t_formatted}_downscale.log) 2>&1

    echo "============================================"
    echo "Downscaling segmented labels (Fiji headless)"
    echo "Timepoint: ${timepoint}"
    echo "Input: ${input_name}"
    echo "Scale factor: ${scale_factor}"
    echo "Interpolation: NONE (nearest neighbor)"
    echo "============================================"
    echo ""

    # Create Fiji macro for nearest-neighbor downscaling
    # interpolation=None ensures label IDs are preserved (no blending)
    cat > downscale.ijm << ENDMACRO
open("\$PWD/${input_name}");
factor = ${scale_factor};
w = round(getWidth() * factor);
h = round(getHeight() * factor);
d = nSlices;
print("Input dimensions: " + getWidth() + " x " + getHeight() + " x " + d);
print("Output dimensions: " + w + " x " + h + " x " + d);
print("Scale factor: " + factor);
print("Interpolation: None (nearest neighbor)");
run("Scale...", "x=" + factor + " y=" + factor + " z=1.0 width=" + w + " height=" + h + " depth=" + d + " interpolation=None process create");
saveAs("Tiff", "\$PWD/${output_name}");
print("Downscaled labels saved");
run("Quit");
ENDMACRO

    echo "Fiji macro contents:"
    cat downscale.ijm
    echo ""

    # Run Fiji headless - no GUI, nearest-neighbor preserves label IDs
    echo "Running Fiji headless..."
    ImageJ-linux64 --headless --console -macro \$PWD/downscale.ijm

    # Verify output
    if [ ! -f "${output_name}" ]; then
        echo "ERROR: Downscaled output not created: ${output_name}"
        echo "Directory contents:"
        ls -lha
        exit 1
    fi

    FILE_SIZE=\$(du -h "${output_name}" | cut -f1)
    echo ""
    echo "Downscaled labels saved: ${output_name} (\$FILE_SIZE)"
    echo "Timepoint ${timepoint} downscaling complete"
    """
}

// ============================================================================
// PROCESS: Benchmark Pipeline Outputs
// ============================================================================

process BENCHMARK {
    tag "benchmark"

    maxRetries 1
    errorStrategy 'ignore'  // Benchmark failure should not fail the pipeline

    publishDir "${params.output_dir}/benchmark",
        mode: 'copy',
        pattern: "benchmark_*"

    publishDir "${params.output_dir}/logs/benchmark",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    val results_dir
    val ready_signal  // dummy signal to ensure upstream processes are done
    path benchmark_script

    output:
    path "benchmark_results.csv", emit: csv, optional: true
    path "benchmark_results.json", emit: json, optional: true
    path "benchmark.log", emit: log, optional: true

    script:
    """
    #!/bin/bash
    set -uo pipefail

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "============================================"
    echo "Running Pipeline Benchmark"
    echo "============================================"
    echo "Results directory: ${results_dir}"
    echo "Python: \$(which python3)"
    echo ""

    # Check files exist in the publishDir
    echo "=== Preprocessed files ==="
    ls -lh ${results_dir}/01_preprocessed/*_processed.tif 2>/dev/null | head -10 || echo "  (none found)"
    PREPROC_COUNT=\$(ls ${results_dir}/01_preprocessed/*_processed.tif 2>/dev/null | wc -l | tr -d ' ')
    echo "Total preprocessed: \$PREPROC_COUNT"
    echo ""

    echo "=== Segmented files ==="
    ls -lh ${results_dir}/02_segmented/*_segmented.tif 2>/dev/null | head -10 || echo "  (none found)"
    SEG_COUNT=\$(ls ${results_dir}/02_segmented/*_segmented.tif 2>/dev/null | wc -l | tr -d ' ')
    echo "Total segmented: \$SEG_COUNT"
    echo ""

    if [ "\$PREPROC_COUNT" -eq 0 ] && [ "\$SEG_COUNT" -eq 0 ]; then
        echo "ERROR: No pipeline output files found in ${results_dir}"
        echo "Check that preprocessing and segmentation completed successfully."
        exit 1
    fi

    # Sample up to 3 timepoints for fast benchmarking
    # Find timepoint numbers from filenames and pick first, middle, last
    TIMEPOINTS=\$(ls ${results_dir}/01_preprocessed/*_processed.tif 2>/dev/null \\
        | sed 's/.*\\/t\\([0-9]*\\)_.*/\\1/' \\
        | sort -n)

    if [ -n "\$TIMEPOINTS" ]; then
        N_TP=\$(echo "\$TIMEPOINTS" | wc -l | tr -d ' ')
        if [ "\$N_TP" -le 3 ]; then
            SAMPLE_TPS=\$(echo "\$TIMEPOINTS" | tr '\\n' ' ')
        else
            FIRST=\$(echo "\$TIMEPOINTS" | head -1)
            MIDDLE=\$(echo "\$TIMEPOINTS" | sed -n "\$((N_TP/2))p")
            LAST=\$(echo "\$TIMEPOINTS" | tail -1)
            SAMPLE_TPS="\$FIRST \$MIDDLE \$LAST"
        fi
        echo "Sampling timepoints: \$SAMPLE_TPS (out of \$N_TP total)"
        TP_ARGS="--timepoints \$SAMPLE_TPS"
    else
        TP_ARGS=""
    fi
    echo ""

    # Run benchmark (output goes to stdout so .command.out gets it in real-time)
    python3 ${benchmark_script.name} \\
        --results_dir ${results_dir}/ \\
        --labels "pipeline_run" \\
        --output_csv benchmark_results.csv \\
        --output_json benchmark_results.json \\
        \$TP_ARGS \\
        2>&1 | tee benchmark.log

    echo ""
    echo "Benchmark completed"
    """
}

// ============================================================================
// PROCESS: Prepare ultrack input (foreground + contours zarrs)
// ============================================================================

process PREP_ULTRACK {
    tag "Preparing ultrack input"

    maxRetries 1
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    input:
    path raw_hyperstack
    path labels_hyperstack
    path prep_script
    val prep_config

    output:
    path "ultrack_input/foreground.zarr", emit: foreground
    path "ultrack_input/contours.zarr", emit: contours

    container params.ultrack_container

    script:
    def prep_cfg = prep_config
    def script_name = prep_script.name
    """
    #!/usr/bin/env bash
    set -euo pipefail

    echo "============================================"
    echo "PREP_ULTRACK: Preparing ultrack input"
    echo "============================================"
    echo "Raw hyperstack: ${raw_hyperstack}"
    echo "Labels hyperstack: ${labels_hyperstack}"
    echo ""

    python3 ${script_name} \\
        --raw "${raw_hyperstack}" \\
        --labels "${labels_hyperstack}" \\
        --output ./ultrack_input \\
        --fg-sigma ${prep_cfg.fg_sigma} \\
        --raw-sigma ${prep_cfg.raw_sigma} \\
        --boundary-width ${prep_cfg.boundary_width} \\
        --min-area ${prep_cfg.min_area}

    # Verify output
    if [ ! -d "ultrack_input/foreground.zarr" ] || [ ! -d "ultrack_input/contours.zarr" ]; then
        echo "ERROR: ultrack_input zarr files not created"
        ls -lR ultrack_input/ 2>/dev/null || echo "ultrack_input/ not found"
        exit 1
    fi

    echo ""
    echo "✓ ultrack input prepared"
    echo "  foreground.zarr: \$(du -sh ultrack_input/foreground.zarr | cut -f1)"
    echo "  contours.zarr:   \$(du -sh ultrack_input/contours.zarr | cut -f1)"
    """
}

// ============================================================================
// PROCESS: ultrack Step 1 — Segment (watershed hierarchy from foreground+contours)
// ============================================================================

process ULTRACK_SEGMENT {
    tag "ultrack segment"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    input:
    path foreground_zarr
    path contours_zarr
    path ultrack_config_toml

    output:
    path "local_ultrack_config.toml", emit: config_toml
    path "data.db", emit: database

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    # Override database path in config to force writing into workdir
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()

# ultrack DataConfig expects: database = "sqlite" (enum) + address = "path"
# Remove any existing database/address lines
cfg = re.sub(r'^\\s*database\\s*=.*\$', '', cfg, flags=re.M)
cfg = re.sub(r'^\\s*address\\s*=.*\$', '', cfg, flags=re.M)

# Insert correct fields under [data_config]
if '[data_config]' in cfg:
cfg = cfg.replace('[data_config]', '[data_config]\\ndatabase = "sqlite"\\naddress = "./data.db"')
else:
    cfg += '\n[data_config]\ndatabase = "sqlite"\naddress = "./data.db"\n'

pathlib.Path('local_ultrack_config.toml').write_text(cfg)
print('Database overridden: sqlite @ ./data.db')
PYEOF

    echo "============================================"
    echo "ULTRACK Step 1/4: Segment"
    echo "============================================"
    echo "Foreground: ${foreground_zarr}"
    echo "Contours:   ${contours_zarr}"
    echo ""

    ultrack segment \\
        ${foreground_zarr} \\
        ${contours_zarr} \\
        --foreground-layer foreground \\
        --contours-layer contours \\
        --config local_ultrack_config.toml \\
        --overwrite

    echo ""
    echo "✓ Segment complete"
    ls -lh data.db
    """
}

// ============================================================================
// PROCESS: ultrack Step 2 — Link (temporal associations between segments)
// ============================================================================

process ULTRACK_LINK {
    tag "ultrack link"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    input:
    path ultrack_config_toml
    path database

    output:
    path "local_ultrack_config.toml", emit: config_toml
    path "data.db", emit: database

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    # Re-patch database path to point to the local workdir copy
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()
cfg = re.sub(r'^\\s*database\\s*=.*\$', '', cfg, flags=re.M)
cfg = re.sub(r'^\\s*address\\s*=.*\$', '', cfg, flags=re.M)
if '[data_config]' in cfg:
cfg = cfg.replace('[data_config]', '[data_config]\\ndatabase = "sqlite"\\naddress = "./data.db"')
else:
    cfg += '\n[data_config]\ndatabase = "sqlite"\naddress = "./data.db"\n'
pathlib.Path('local_ultrack_config.toml').write_text(cfg)
PYEOF

    echo "============================================"
    echo "ULTRACK Step 2/4: Link"
    echo "============================================"
    echo ""

    ultrack link --config local_ultrack_config.toml

    echo ""
    echo "✓ Link complete"
    ls -lh data.db
    """
}

// ============================================================================
// PROCESS: ultrack Step 3 — Solve (ILP optimisation — memory-intensive)
// ============================================================================

process ULTRACK_SOLVE {
    tag "ultrack solve"

    maxRetries 3
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    input:
    path ultrack_config_toml
    path database

    output:
    path "local_ultrack_config.toml", emit: config_toml
    path "data.db", emit: database

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    # Re-patch database path to point to the local workdir copy
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()
cfg = re.sub(r'^\\s*database\\s*=.*\$', '', cfg, flags=re.M)
cfg = re.sub(r'^\\s*address\\s*=.*\$', '', cfg, flags=re.M)
if '[data_config]' in cfg:
cfg = cfg.replace('[data_config]', '[data_config]\\ndatabase = "sqlite"\\naddress = "./data.db"')
else:
    cfg += '\n[data_config]\ndatabase = "sqlite"\naddress = "./data.db"\n'
pathlib.Path('local_ultrack_config.toml').write_text(cfg)
PYEOF

    echo "============================================"
    echo "ULTRACK Step 3/4: Solve"
    echo "============================================"
    echo "Attempt: ${task.attempt}"
    echo "Memory:  ${task.memory}"
    echo ""

    ultrack solve --config local_ultrack_config.toml

    echo ""
    echo "✓ Solve complete"
    ls -lh data.db
    """
}

// ============================================================================
// PROCESS: ultrack Step 4 — Export (write tracking results)
// ============================================================================

process ULTRACK_EXPORT {
    tag "ultrack export"

    maxRetries 1
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/03_tracking",
        mode: 'copy'

    input:
    path ultrack_config_toml
    path database

    output:
    path "results/**", emit: results
    path "ultrack_export.log", emit: log

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    exec > >(tee ultrack_export.log) 2>&1

    # Re-patch database path to point to the local workdir copy
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()
cfg = re.sub(r'^\\s*database\\s*=.*\$', '', cfg, flags=re.M)
cfg = re.sub(r'^\\s*address\\s*=.*\$', '', cfg, flags=re.M)
if '[data_config]' in cfg:
cfg = cfg.replace('[data_config]', '[data_config]\\ndatabase = "sqlite"\\naddress = "./data.db"')
else:
    cfg += '\n[data_config]\ndatabase = "sqlite"\naddress = "./data.db"\n'
pathlib.Path('local_ultrack_config.toml').write_text(cfg)
PYEOF

    echo "============================================"
    echo "ULTRACK Step 4/4: Export"
    echo "============================================"
    echo ""

    ultrack export zarr-napari \\
        --config local_ultrack_config.toml \\
        --output-directory results/ \\
        --overwrite

    # Verify output
    if [ ! -d "results" ]; then
        echo "ERROR: results/ directory not created"
        exit 1
    fi

    echo ""
    echo "✓ ultrack export complete"
    echo "  Results: \$(du -sh results/ | cut -f1)"
    """
}

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

workflow {
    // Parse input files with pattern: t####_Channel #.tif
    // Use java.nio.file.FileSystems to handle paths with spaces correctly
    // (Nextflow's fromPath glob can break on spaces in directory names)
    def input_dir_path = Paths.get(params.input_dir)
    def glob_pattern = "t*_Channel ${params.channel}.tif"
    def matched_files = []

    def globMatcher = FileSystems.getDefault().getPathMatcher("glob:${glob_pattern}")
    Files.list(input_dir_path).each { p ->
        if (globMatcher.matches(p.getFileName())) {
            matched_files.add(p.toFile())
        }
    }

    if (matched_files.isEmpty()) {
        error "No files found matching pattern: ${params.input_dir}/${glob_pattern}"
    }

    log.info "Found ${matched_files.size()} files matching pattern in: ${params.input_dir}"

    input_channel = Channel
        .fromList(matched_files.collect { f -> file(f.toPath()) })
        .ifEmpty { error "No files found matching pattern: ${params.input_dir}/${glob_pattern}" }
        .map { file ->
            def matcher = (file.name =~ /t(\d+)_Channel/)
            if (matcher.find()) {
                def timepoint = matcher.group(1).toInteger()
                return tuple(timepoint, file)
            } else {
                error "Could not parse timepoint from filename: ${file.name}"
            }
        }
        .tap { parsed_files }

    // Log parsed files
    parsed_files.subscribe { timepoint, file ->
        log.info "Found timepoint ${timepoint}: ${file.name}"
    }

    // Create channel for preprocessing script
    preproc_script_ch = Channel.fromPath(params.preprocessing_script, checkIfExists: true)

    // Create channel for merge script (only needed if merge is enabled)
    if (!skip_merge) {
        merge_script_ch = Channel.fromPath(params.merge_script, checkIfExists: true)
    }

    // OPTIONAL: ROI Cropping Step
    if (config.roi_cropping.enabled) {
        log.info "ROI cropping enabled - using ${config.roi_cropping.roi_path}"

        // Create channel for ROI file
        roi_file_ch = Channel.fromPath(config.roi_cropping.roi_path, checkIfExists: true)

        // Apply ROI cropping to all timepoints
        CROP_WITH_ROI(
            input_channel,
            roi_file_ch.collect()
        )

        // Use cropped images as input for rest of pipeline
        processing_input = CROP_WITH_ROI.out.cropped
        was_cropped = true
    } else {
        log.info "ROI cropping disabled - using original images"
        processing_input = input_channel
        was_cropped = false
    }

    // 1. Extract/Configure metadata from FIRST timepoint
    first_timepoint = processing_input.first()
    EXTRACT_METADATA(first_timepoint, was_cropped)

    // Share the same metadata with all timepoints
    shared_metadata = EXTRACT_METADATA.out.metadata

    // 2. Preprocess and deconvolve each timepoint
    PREPROCESS_DECONVOLVE(
        processing_input,
        shared_metadata,
        preproc_script_ch.collect(),
        config.preprocessing
    )

    // 3. Segment each timepoint with Cellpose
    if (!skip_segmentation) {
        CELLPOSE_SEGMENT(
            PREPROCESS_DECONVOLVE.out.processed,
            shared_metadata,
            config.segmentation,
            config.preprocessing.image_scaling
        )

        // 4. OPTIONAL: Downscale segmented labels using Fiji headless (nearest-neighbor)
        if (downscale_labels < 1.0) {
            log.info "Label downscaling enabled: factor=${downscale_labels} (Fiji nearest-neighbor, no interpolation)"
            DOWNSCALE_SEGMENTATION(
                CELLPOSE_SEGMENT.out.segmented,
                downscale_labels
            )
        }

        // 5. OPTIONAL: Merge timepoints into 4D hyperstacks
        if (!skip_merge) {
            log.info "Hyperstack merging enabled (processed + segmented)"

            // Collect processed files into a merge job tuple: ['processed', [files...]]
            processed_merge_ch = PREPROCESS_DECONVOLVE.out.processed
                .map { timepoint, processed_file -> processed_file }
                .collect()
                .map { files -> tuple('processed', files) }

            // Collect segmented files into a merge job tuple: ['segmented', [files...]]
            segmented_merge_ch = CELLPOSE_SEGMENT.out.segmented
                .map { timepoint, segmented_file -> segmented_file }
                .collect()
                .map { files -> tuple('segmented', files) }

            // Mix both merge jobs into a single channel (each emitted item = one process invocation)
            merge_jobs_ch = processed_merge_ch.mix(segmented_merge_ch)

            MERGE_TO_HYPERSTACK(
                merge_jobs_ch,
                shared_metadata,
                merge_script_ch.collect()
            )
        } else {
            log.info "Hyperstack merging SKIPPED (skip_merge=true)"
        }

        // 6. OPTIONAL: ultrack tracking (requires merge to produce hyperstacks)
        if (!skip_tracking) {
            log.info "Ultrack tracking enabled"

            // Filter merge outputs: get processed hyperstack (raw) and segmented hyperstack (labels)
            // The TIFF hyperstack is required for ultrack prep (uses tifffile)
            // Filenames contain the data type: 4D_hyperstack_processed.tif, 4D_hyperstack_segmented.tif
            processed_hs = MERGE_TO_HYPERSTACK.out.hyperstack
                .filter { it.name.contains('processed') }

            segmented_hs = MERGE_TO_HYPERSTACK.out.hyperstack
                .filter { it.name.contains('segmented') }

            prep_ultrack_script_ch = Channel.fromPath(params.prep_ultrack_script, checkIfExists: true)

            // Step 1: Prepare foreground + contours zarrs (GPU)
            PREP_ULTRACK(
                processed_hs,
                segmented_hs,
                prep_ultrack_script_ch.collect(),
                config.tracking.prep
            )

            // Step 2: ultrack segment → link → solve → export (4 separate processes)
            ultrack_config_ch = Channel.fromPath(
                config.tracking.ultrack_config_toml ?: './ultrack_config.toml',
                checkIfExists: true
            )

            ULTRACK_SEGMENT(
                PREP_ULTRACK.out.foreground,
                PREP_ULTRACK.out.contours,
                ultrack_config_ch.collect()
            )

            ULTRACK_LINK(
                ULTRACK_SEGMENT.out.config_toml,
                ULTRACK_SEGMENT.out.database
            )

            ULTRACK_SOLVE(
                ULTRACK_LINK.out.config_toml,
                ULTRACK_LINK.out.database
            )

            ULTRACK_EXPORT(
                ULTRACK_SOLVE.out.config_toml,
                ULTRACK_SOLVE.out.database
            )
        } else {
            log.info "Ultrack tracking SKIPPED (tracking.enabled=false)"
        }

        // 7. OPTIONAL: Benchmark pipeline outputs
        def run_benchmark = config.benchmark?.enabled ?: false
        if (run_benchmark) {
            log.info "Benchmarking enabled - will compute quality metrics"

            benchmark_script_ch = Channel.fromPath(params.benchmark_script, checkIfExists: true)

            // Resolve output_dir to absolute path (it may be relative like './results/')
            def abs_output_dir = file(params.output_dir).toAbsolutePath().toString()
            log.info "Benchmark will read from: ${abs_output_dir}"

            // Wait for segmentation to finish, then run benchmark on the publishDir directly.
            // This avoids staging hundreds of GB of TIF files into the benchmark work dir.
            ready_signal = CELLPOSE_SEGMENT.out.segmented
                .map { timepoint, segmented_file -> true }
                .collect()

            BENCHMARK(
                abs_output_dir,
                ready_signal,
                benchmark_script_ch.collect()
            )
        } else {
            log.info "Benchmarking disabled"
        }
    } else {
        log.info "Segmentation SKIPPED (segmentation.enabled=false)"

        // Even without segmentation, merge processed images if requested
        if (!skip_merge) {
            log.info "Hyperstack merging enabled (processed only, no segmentation)"

            processed_merge_ch = PREPROCESS_DECONVOLVE.out.processed
                .map { timepoint, processed_file -> processed_file }
                .collect()
                .map { files -> tuple('processed', files) }

            MERGE_TO_HYPERSTACK(
                processed_merge_ch,
                shared_metadata,
                merge_script_ch.collect()
            )
        }
    }
}

// ============================================================================
// WORKFLOW COMPLETION
// ============================================================================

workflow.onComplete {
    def voxel_mode = config.voxel_size?.auto_detect ? "Auto-detected" : "Manual override"
    def roi_status = config.roi_cropping?.enabled ? "ENABLED" : "DISABLED"
    def merge_status = (config.output?.skip_merge ?: false) ? "SKIPPED" : "ENABLED"
    def seg_status = (config.segmentation?.enabled == false) ? "SKIPPED" : "ENABLED"
    def ds_factor = config.segmentation?.downscale_labels != null ? config.segmentation.downscale_labels : 1.0
    def downscale_status = ds_factor < 1.0 ? "ENABLED (${ds_factor})" : "DISABLED"

    def benchmark_status = (config.benchmark?.enabled ?: false) ? "ENABLED" : "DISABLED"
    def tracking_status = (config.tracking?.enabled ?: false) ? "ENABLED (ultrack)" : "DISABLED"

    log.info """
    ============================================================================
    Pipeline completed!
    ============================================================================
    Status       : ${workflow.success ? 'SUCCESS ✓' : 'FAILED ✗'}
    Duration     : ${workflow.duration}
    Channel      : ${params.channel}
    ROI cropping : ${roi_status}
    Segmentation : ${seg_status}
    Voxel mode   : ${voxel_mode}
    Merge        : ${merge_status}
    Downscale    : ${downscale_status}
    Tracking     : ${tracking_status}
    Benchmark    : ${benchmark_status}
    Output dir   : ${params.output_dir}

    Results:
      ${config.roi_cropping?.enabled ? "- Cropped images     : ${params.output_dir}/00_cropped/" : ""}
      - Preprocessed images : ${params.output_dir}/01_preprocessed/
      - Segmented masks     : ${params.output_dir}/02_segmented/
      ${ds_factor < 1.0 ? "- Downscaled labels  : ${params.output_dir}/02_segmented_downscaled/" : ""}
      ${!(config.output?.skip_merge ?: false) ? "- Hyperstacks        : in 01_preprocessed/ and 02_segmented/" : ""}
      ${(config.tracking?.enabled ?: false) ? "- Tracking results   : ${params.output_dir}/03_tracking/" : ""}
      ${(config.benchmark?.enabled ?: false) ? "- Benchmark          : ${params.output_dir}/benchmark/" : ""}
      - Logs                : ${params.output_dir}/logs/

    Completed at: ${workflow.complete}
    ============================================================================
    """.stripIndent()
}

workflow.onError {
    log.error """
    ============================================================================
    Pipeline execution failed!
    ============================================================================
    Error message: ${workflow.errorMessage}
    Error report : ${workflow.errorReport}
    ============================================================================
    """.stripIndent()
}
