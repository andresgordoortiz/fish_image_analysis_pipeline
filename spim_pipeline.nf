#!/usr/bin/env nextflow

/*
 * ============================================================================
 * SPIM 4D Image Processing Pipeline - WITH OPTIONAL ROI CROPPING
 * ============================================================================
 *
 * This pipeline processes 4D SPIM images through:
 * 0. OPTIONAL: ROI-based cropping using Fiji (if roi_path provided in config)
 * 1. File parsing with timepoint extraction (t0051_Channel 1.tif format)
 * 2. Preprocessing and deconvolution per timepoint
 * 3. Cellpose segmentation per timepoint
 * 4. Merging all timepoints into a single 4D hyperstack with preserved metadata
 *
 * All parameters are loaded from a JSON configuration file for reproducibility.
 *
 * NEW: Optional ROI cropping with Fiji
 * NEW: Flexible voxel size configuration - auto-detect or manual override
 * FIX: Robust file finding that handles spaces in filenames and scaling mismatches
 */

nextflow.enable.dsl=2

// ============================================================================
// PARAMETER DEFINITIONS
// ============================================================================

params.input_dir = null
params.output_dir = null
params.config_json = null
params.channel = 1
params.preprocessing_script = './spim_pipeline_fixed.py'
params.container = 'docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-ad6c6a1'
params.fiji_container = 'docker://fiji/fiji:20220415'
params.help = false

// Show help message
if (params.help) {
    log.info """
    ============================================================================
    SPIM 4D Image Processing Pipeline with Optional ROI Cropping
    ============================================================================

    Usage:
        nextflow run spim_pipeline_with_roi.nf \\
            --input_dir <path> \\
            --output_dir <path> \\
            --config_json <path> \\
            [options]

    Required Arguments:
        --input_dir         Directory containing input timepoint TIFF images
                           (format: t0001_Channel 1.tif, t0002_Channel 1.tif, etc.)
        --output_dir        Directory for all pipeline outputs
        --config_json       JSON file with processing parameters

    Optional Arguments:
        --channel               Channel number to process (default: 1)
        --preprocessing_script  Path to preprocessing Python script
                               (default: ./spim_pipeline_fixed.py)
        --container             Singularity/Docker container image
                               (default: docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-ad6c6a1)
        --fiji_container        Fiji container for ROI cropping
                               (default: docker://fiji/fiji:20220415)
        --help                 Show this help message

    ROI Cropping (in config.json):
        roi_cropping.enabled: true/false - Enable ROI cropping step
        roi_cropping.roi_path: "/path/to/file.roi" - Path to ImageJ ROI file

        NOTE: ROI should be created from one timepoint's raw image. It will be
        applied to ALL timepoints. Cropping preserves voxel spacing but reduces
        image dimensions.

    Voxel Size Configuration (in config.json):
        voxel_size.auto_detect: true  - Auto-detect from image metadata (default)
                                false - Use manual values specified in config
        voxel_size.x_um, y_um, z_um   - Manual voxel sizes (used if auto_detect=false)

    Output Structure:
        output_dir/
        ├── 00_cropped/          - ROI-cropped images (if enabled)
        ├── 01_preprocessed/     - Preprocessed & deconvolved images per timepoint
        ├── 02_segmented/        - Cellpose segmentation masks per timepoint
        ├── 03_hyperstack/       - Final 4D merged hyperstack with metadata
        ├── metadata/            - Preserved metadata files
        ├── logs/                - Processing logs
        └── reports/             - QC reports

    Example with ROI cropping:
        nextflow run spim_pipeline_with_roi.nf \\
            --input_dir ./data \\
            --output_dir ./output \\
            --config_json config_with_roi.json \\
            --channel 2

    ============================================================================
    """.stripIndent()
    exit 0
}

// Validate required parameters
if (!params.input_dir || !params.output_dir || !params.config_json) {
    log.error "ERROR: Missing required parameters!"
    log.error "Required: --input_dir, --output_dir, --config_json"
    log.error "Run with --help for usage information"
    exit 1
}

// Validate preprocessing script exists
if (!file(params.preprocessing_script).exists()) {
    log.error "ERROR: Preprocessing script not found: ${params.preprocessing_script}"
    exit 1
}

// ============================================================================
// LOAD CONFIGURATION
// ============================================================================

def loadConfig(json_path) {
    def jsonSlurper = new groovy.json.JsonSlurper()
    return jsonSlurper.parse(new File(json_path))
}

config = loadConfig(params.config_json)

// Validate voxel_size configuration
if (!config.containsKey('voxel_size')) {
    log.warn "WARNING: No 'voxel_size' section in config - defaulting to auto-detect"
    config.voxel_size = [auto_detect: true]
}

// Validate ROI cropping configuration
if (!config.containsKey('roi_cropping')) {
    config.roi_cropping = [enabled: false]
}

// Validate ROI file exists if cropping is enabled
if (config.roi_cropping.enabled) {
    if (!config.roi_cropping.containsKey('roi_path')) {
        log.error "ERROR: ROI cropping enabled but no 'roi_path' specified in config"
        exit 1
    }
    if (!file(config.roi_cropping.roi_path).exists()) {
        log.error "ERROR: ROI file not found: ${config.roi_cropping.roi_path}"
        exit 1
    }
}

// ============================================================================
// WORKFLOW HEADER
// ============================================================================

def voxel_info = config.voxel_size.auto_detect ?
    "Auto-detect from image metadata" :
    "Manual: ${config.voxel_size.x_um} x ${config.voxel_size.y_um} x ${config.voxel_size.z_um} µm"

def roi_info = config.roi_cropping.enabled ?
    "ENABLED - using ${config.roi_cropping.roi_path}" :
    "DISABLED"

log.info """
============================================================================
SPIM 4D Image Processing Pipeline - WITH ROI CROPPING
============================================================================
Input directory    : ${params.input_dir}
Output directory   : ${params.output_dir}
Configuration      : ${params.config_json}
Channel to process : ${params.channel}
ROI Cropping       : ${roi_info}
Voxel size mode    : ${voxel_info}
Preprocessing script: ${params.preprocessing_script}
Container          : ${params.container}
Fiji Container     : ${params.fiji_container}

Pipeline steps:
  ${config.roi_cropping.enabled ? '0. ROI-based cropping with Fiji (all timepoints)' : ''}
  1. Parse and sort timepoint files
  2. Extract/Configure metadata with voxel sizes
  3. Preprocessing & Deconvolution per timepoint
  4. Cellpose Segmentation per timepoint
  5. Merge into 4D hyperstack with preserved metadata

Started at: ${new Date()}
============================================================================
""".stripIndent()

// ============================================================================
// PROCESS: ROI-based Cropping with Fiji (OPTIONAL)
// ============================================================================

process CROP_WITH_ROI {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 1
    errorStrategy 'terminate'

    publishDir "${params.output_dir}/00_cropped",
        mode: 'copy',
        pattern: "*_cropped.tif"

    publishDir "${params.output_dir}/logs/cropping",
        mode: 'copy',
        pattern: "*.log"

    container params.fiji_container

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

    echo "============================================" | tee t${t_formatted}_crop.log
    echo "ROI Cropping timepoint: ${timepoint}" | tee -a t${t_formatted}_crop.log
    echo "Input file: ${filename}" | tee -a t${t_formatted}_crop.log
    echo "ROI file: ${roi_filename}" | tee -a t${t_formatted}_crop.log
    echo "============================================" | tee -a t${t_formatted}_crop.log
    echo "" | tee -a t${t_formatted}_crop.log

    # Create ImageJ macro for cropping
    cat > crop_macro.ijm << 'MACRO_EOF'
// Open the image
open("${filename}");
originalTitle = getTitle();

// Load and apply ROI
roiManager("Open", "${roi_filename}");
roiManager("Select", 0);

// Get ROI bounds for logging
Roi.getBounds(rx, ry, rwidth, rheight);
print("ROI bounds: x=" + rx + ", y=" + ry + ", width=" + rwidth + ", height=" + rheight);

// Crop all slices (3D stack)
run("Crop");

// Save cropped image
saveAs("Tiff", "t${t_formatted}_cropped.tif");

// Log final dimensions
getDimensions(width, height, channels, slices, frames);
print("Cropped dimensions: " + width + "x" + height + "x" + slices + " (WxHxZ)");

// Close
close();
print("Cropping completed successfully");
MACRO_EOF

    # Run Fiji in headless mode
    echo "Running Fiji in headless mode..." | tee -a t${t_formatted}_crop.log

    # Fiji headless execution
    ImageJ-linux64 --ij2 --headless --console --run crop_macro.ijm >> t${t_formatted}_crop.log 2>&1

    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo "ERROR: Fiji cropping failed with exit code \$EXIT_CODE" | tee -a t${t_formatted}_crop.log
        exit \$EXIT_CODE
    fi

    # Verify output exists
    if [ ! -f "t${t_formatted}_cropped.tif" ]; then
        echo "ERROR: Cropped output file not created" | tee -a t${t_formatted}_crop.log
        echo "Directory contents:" | tee -a t${t_formatted}_crop.log
        ls -lh | tee -a t${t_formatted}_crop.log
        exit 1
    fi

    echo "" | tee -a t${t_formatted}_crop.log
    echo "ROI cropping completed successfully for timepoint ${timepoint}" | tee -a t${t_formatted}_crop.log
    echo "Output: t${t_formatted}_cropped.tif" | tee -a t${t_formatted}_crop.log
    """
}

// ============================================================================
// PROCESS: Extract and Configure Metadata
// ============================================================================

process EXTRACT_METADATA {
    tag "Extracting metadata"

    maxRetries 1
    errorStrategy 'terminate'

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

    maxRetries 1
    errorStrategy 'terminate'

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
    '--padding', str(config['deconvolution']['padding'])
]

# Add optional flags from correction_flags
if config['correction_flags'].get('no_clahe', False):
    cmd.append('--no_clahe')
if config['correction_flags'].get('no_z_correction', False):
    cmd.append('--no_z_correction')
if config['correction_flags'].get('no_shading', False):
    cmd.append('--no_shading')

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

# Recalculate voxel sizes after scaling
# NOTE: ROI cropping doesn't affect voxel size, only preprocessing scaling does
x_res = metadata['x_resolution_um'] / ${cfg.image_scaling}
y_res = metadata['y_resolution_um'] / ${cfg.image_scaling}
z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

print(f"Voxel sizes after preprocessing scaling:")
print(f"  Original: {metadata['x_resolution_um']:.4f} x {metadata['y_resolution_um']:.4f} x {z_spacing:.4f} µm")
print(f"  Scaled (×{${cfg.image_scaling}}): {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} µm")
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

    maxRetries 1
    errorStrategy 'terminate'

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

# Load config
config = json.loads('${config_json_str}')

# Load mask
mask = tifffile.imread("t${t_formatted}_segmented.tif")

# Calculate voxel sizes (accounting for preprocessing scaling)
x_res = metadata['x_resolution_um'] / config['image_scaling']
y_res = metadata['y_resolution_um'] / config['image_scaling']
z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

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
    tag "Creating 4D hyperstack"

    maxRetries 1
    errorStrategy 'terminate'

    publishDir "${params.output_dir}/03_hyperstack",
        mode: 'copy'

    input:
    path all_segmented_files
    path metadata_json

    output:
    path "4D_hyperstack.tif", emit: hyperstack
    path "4D_hyperstack_metadata.json", emit: metadata

    container params.container

    script:
    def config_json_str = groovy.json.JsonOutput.toJson(config).replace("'", "\\'")
    """
    #!/bin/bash
    set -e

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 << 'EOF'
import tifffile
import numpy as np
import json
from pathlib import Path
import re

print("="*60)
print("Merging all timepoints into 4D hyperstack")
print("="*60)

# Get all segmented files and sort by timepoint number
seg_files = sorted(Path('.').glob('t*_segmented.tif'))

if not seg_files:
    raise ValueError("No segmented files found!")

print(f"Found {len(seg_files)} timepoint files")

# Extract timepoint numbers and sort
timepoint_data = []
for f in seg_files:
    match = re.search(r't(\\d+)_segmented\\.tif', f.name)
    if match:
        t = int(match.group(1))
        timepoint_data.append((t, f))
    else:
        print(f"Warning: Could not extract timepoint from {f.name}")

timepoint_data.sort(key=lambda x: x[0])

# Load reference metadata
with open('${metadata_json}', 'r') as f:
    ref_metadata = json.load(f)

# Load config
config = json.loads('${config_json_str}')

# Load all timepoints
timepoint_arrays = []
for t, f in timepoint_data:
    img = tifffile.imread(str(f))
    print(f"  Loaded t{t:04d}: shape={img.shape}, dtype={img.dtype}")

    # Ensure 3D (ZYX)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image for t{t:04d}, got {img.ndim}D")

    timepoint_arrays.append(img)

# Verify all timepoints have same shape
shapes = [arr.shape for arr in timepoint_arrays]
if len(set(shapes)) > 1:
    raise ValueError(f"Inconsistent shapes across timepoints: {set(shapes)}")

# Stack into 4D array (TZYX)
img_4d = np.stack(timepoint_arrays, axis=0)
print(f"\\nMerged 4D shape: {img_4d.shape} (TZYX)")

# Calculate metadata (accounting for preprocessing scaling)
# NOTE: ROI cropping does not change voxel size
scaling = config['segmentation']['image_scaling']
x_res = ref_metadata['x_resolution_um'] / scaling
y_res = ref_metadata['y_resolution_um'] / scaling
z_spacing = ref_metadata['imagej']['spacing'] if 'imagej' in ref_metadata else 1.0

print(f"\\nVoxel size configuration:")
print(f"  Source: {ref_metadata.get('voxel_size_source', 'unknown')}")
print(f"  Was ROI cropped: {ref_metadata.get('was_roi_cropped', False)}")
print(f"  Original: {ref_metadata['x_resolution_um']:.4f} x {ref_metadata['y_resolution_um']:.4f} x {z_spacing:.4f} µm")
print(f"  Final (after ×{scaling} scaling): {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} µm")

# Create comprehensive metadata
hyperstack_metadata = {
    'shape': {
        'axes': 'TZYX',
        'T': img_4d.shape[0],
        'Z': img_4d.shape[1],
        'Y': img_4d.shape[2],
        'X': img_4d.shape[3]
    },
    'voxel_size': {
        'x_um': x_res,
        'y_um': y_res,
        'z_um': z_spacing,
        'unit': 'um',
        'source': ref_metadata.get('voxel_size_source', 'unknown')
    },
    'dtype': str(img_4d.dtype),
    'n_timepoints': img_4d.shape[0],
    'is_label_image': True,
    'was_roi_cropped': ref_metadata.get('was_roi_cropped', False),
    'processing': {
        'roi_cropping_enabled': ref_metadata.get('was_roi_cropped', False),
        'preprocessing_scaling': config['preprocessing']['image_scaling'],
        'segmentation_scaling': scaling,
        'cellpose_diameter': config['segmentation']['diameter'],
        'cellpose_model': config['segmentation']['model']
    },
    'original_metadata': ref_metadata
}

# Save metadata JSON
with open('4D_hyperstack_metadata.json', 'w') as f:
    json.dump(hyperstack_metadata, f, indent=2)

# Save 4D TIFF with full ImageJ metadata
print("\\nSaving 4D hyperstack...")
tifffile.imwrite(
    '4D_hyperstack.tif',
    img_4d.astype(np.uint16),
    imagej=True,
    resolution=(1.0/x_res, 1.0/y_res),
    metadata={
        'spacing': z_spacing,
        'unit': 'um',
        'axes': 'TZYX',
        'frames': img_4d.shape[0],
        'slices': img_4d.shape[1],
        'LabelImage': True,
        'WasROICropped': ref_metadata.get('was_roi_cropped', False)
    }
)

print("\\n" + "="*60)
print("4D Hyperstack created successfully!")
print("="*60)
print(f"Shape: {img_4d.shape} (TZYX)")
print(f"Voxel size: {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} µm")
print(f"Voxel size source: {ref_metadata.get('voxel_size_source', 'unknown')}")
print(f"ROI cropped: {ref_metadata.get('was_roi_cropped', False)}")
print(f"Timepoints: {img_4d.shape[0]}")
print(f"Z slices per timepoint: {img_4d.shape[1]}")
print("="*60)
EOF
    """
}

// ============================================================================
// PROCESS: Generate QC Report
// ============================================================================

process GENERATE_QC_REPORT {
    maxRetries 1
    errorStrategy 'terminate'

    publishDir "${params.output_dir}/reports",
        mode: 'copy'

    input:
    path all_logs
    path hyperstack_metadata
    path config_json

    output:
    path "pipeline_report.html"
    path "pipeline_summary.json"

    container params.container

    script:
    """
    #!/bin/bash
    set -e

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 << 'EOF'
import json
from pathlib import Path
from datetime import datetime

# Load hyperstack metadata
with open('${hyperstack_metadata}', 'r') as f:
    hyperstack_meta = json.load(f)

# Load config
with open('${config_json}', 'r') as f:
    config = json.load(f)

# Collect all log files
crop_logs = list(Path('.').glob('*_crop.log'))
preprocess_logs = list(Path('.').glob('*_preprocess.log'))
segment_logs = list(Path('.').glob('*_segment.log'))

# Generate summary
summary = {
    'pipeline_version': '1.0.0-with-roi-cropping',
    'execution_date': datetime.now().isoformat(),
    'input_channel': ${params.channel},
    'roi_cropping': {
        'enabled': hyperstack_meta.get('was_roi_cropped', False),
        'n_timepoints_cropped': len(crop_logs) if crop_logs else 0
    },
    'voxel_size_configuration': {
        'mode': hyperstack_meta['voxel_size'].get('source', 'unknown'),
        'final_voxel_size_um': hyperstack_meta['voxel_size']
    },
    'configuration': config,
    'results': {
        'n_timepoints_processed': len(preprocess_logs),
        'n_timepoints_segmented': len(segment_logs),
        'final_hyperstack_shape': hyperstack_meta['shape'],
        'voxel_size_um': hyperstack_meta['voxel_size']
    }
}

# Save summary JSON
with open('pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Determine voxel size badge
voxel_source = hyperstack_meta['voxel_size'].get('source', 'unknown')
if voxel_source == 'auto_detected':
    voxel_badge = '<span style="background-color: #27ae60; color: white; padding: 5px 10px; border-radius: 3px;">AUTO-DETECTED</span>'
elif voxel_source == 'manual_override':
    voxel_badge = '<span style="background-color: #e67e22; color: white; padding: 5px 10px; border-radius: 3px;">MANUAL OVERRIDE</span>'
else:
    voxel_badge = '<span style="background-color: #95a5a6; color: white; padding: 5px 10px; border-radius: 3px;">UNKNOWN</span>'

# ROI cropping badge
roi_cropped = hyperstack_meta.get('was_roi_cropped', False)
roi_badge = '<span style="background-color: #3498db; color: white; padding: 5px 10px; border-radius: 3px;">YES</span>' if roi_cropped else '<span style="background-color: #95a5a6; color: white; padding: 5px 10px; border-radius: 3px;">NO</span>'

# Generate HTML report
html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>SPIM Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .info {{ color: #3498db; font-weight: bold; }}
        .warning {{ color: #f39c12; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; border-left: 4px solid #3498db; }}
        .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .highlight-box {{ background-color: #fff3cd; border-left: 4px solid #f39c12; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 SPIM 4D Image Processing Pipeline Report</h1>

        <div class="metric">
            <div class="metric-value">{hyperstack_meta['n_timepoints']} Timepoints</div>
            <div class="metric-label">Successfully processed and merged into 4D hyperstack</div>
        </div>

        <div class="highlight-box">
            <h3 style="margin-top: 0;">✂️ ROI Cropping</h3>
            <p><strong>Enabled:</strong> {roi_badge}</p>
            {f'<p><strong>Timepoints cropped:</strong> {len(crop_logs)}</p>' if roi_cropped else ''}
            <p><em>Note: ROI cropping reduces image dimensions but preserves voxel spacing</em></p>
        </div>

        <div class="highlight-box">
            <h3 style="margin-top: 0;">📏 Voxel Size Configuration</h3>
            <p><strong>Mode:</strong> {voxel_badge}</p>
            <p><strong>Final voxel size:</strong> {hyperstack_meta['voxel_size']['x_um']:.4f} × {hyperstack_meta['voxel_size']['y_um']:.4f} × {hyperstack_meta['voxel_size']['z_um']:.4f} µm</p>
        </div>

        <h2>📊 Execution Summary</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Pipeline Version</td><td>{summary['pipeline_version']}</td></tr>
            <tr><td>Execution Date</td><td>{summary['execution_date']}</td></tr>
            <tr><td>Input Channel</td><td class="info">{summary['input_channel']}</td></tr>
            <tr><td>ROI Cropping Enabled</td><td>{roi_badge}</td></tr>
            <tr><td>Timepoints Preprocessed</td><td class="success">{summary['results']['n_timepoints_processed']}</td></tr>
            <tr><td>Timepoints Segmented</td><td class="success">{summary['results']['n_timepoints_segmented']}</td></tr>
        </table>

        <h2>🎯 Final Hyperstack Details</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Axes Order</td><td class="info">TZYX</td></tr>
            <tr><td>T (Timepoints)</td><td>{hyperstack_meta['shape']['T']}</td></tr>
            <tr><td>Z (Slices)</td><td>{hyperstack_meta['shape']['Z']}</td></tr>
            <tr><td>Y (Height)</td><td>{hyperstack_meta['shape']['Y']}</td></tr>
            <tr><td>X (Width)</td><td>{hyperstack_meta['shape']['X']}</td></tr>
            <tr><td>ROI Cropped</td><td>{roi_badge}</td></tr>
            <tr><td>Voxel Size Source</td><td>{voxel_badge}</td></tr>
            <tr><td>X Resolution</td><td>{hyperstack_meta['voxel_size']['x_um']:.4f} µm</td></tr>
            <tr><td>Y Resolution</td><td>{hyperstack_meta['voxel_size']['y_um']:.4f} µm</td></tr>
            <tr><td>Z Spacing</td><td>{hyperstack_meta['voxel_size']['z_um']:.4f} µm</td></tr>
            <tr><td>Data Type</td><td>{hyperstack_meta['dtype']}</td></tr>
            <tr><td>Label Image</td><td class="success">Yes (segmentation masks)</td></tr>
        </table>

        <h2>⚙️ Processing Configuration</h2>

        {'<h3>ROI Cropping</h3>' if config.get('roi_cropping', {}).get('enabled', False) else ''}
        {f'<pre>{json.dumps(config.get("roi_cropping", {{}}), indent=2)}</pre>' if config.get('roi_cropping', {}).get('enabled', False) else ''}

        <h3>Voxel Size Settings</h3>
        <pre>{json.dumps(config.get('voxel_size', {{'auto_detect': True}}), indent=2)}</pre>

        <h3>Preprocessing</h3>
        <pre>{json.dumps(config['preprocessing'], indent=2)}</pre>

        <h3>Segmentation (Cellpose)</h3>
        <pre>{json.dumps(config['segmentation'], indent=2)}</pre>

        <h2>📋 Pipeline Steps</h2>
        <ol>
            <li><strong>File Parsing</strong> - Extracted timepoints from filename pattern (t####_Channel #.tif)</li>
            {f'<li><strong>ROI Cropping</strong> - Applied ROI from {config.get("roi_cropping", {}).get("roi_path", "N/A")} to all timepoints</li>' if roi_cropped else ''}
            <li><strong>Metadata Extraction/Configuration</strong> - {'Auto-detected' if voxel_source == 'auto_detected' else 'Manual override of'} voxel sizes from image metadata</li>
            <li><strong>Preprocessing & Deconvolution</strong> - Applied corrections and deconvolution per timepoint</li>
            <li><strong>Cellpose Segmentation</strong> - 3D cell segmentation per timepoint</li>
            <li><strong>Hyperstack Merging</strong> - Combined all timepoints into single 4D TIFF with preserved metadata</li>
        </ol>

        <h2>📂 Output Files</h2>
        <table>
            <tr><th>Directory</th><th>Contents</th></tr>
            {f'<tr><td>00_cropped/</td><td>ROI-cropped images per timepoint</td></tr>' if roi_cropped else ''}
            <tr><td>01_preprocessed/</td><td>Preprocessed and deconvolved images per timepoint</td></tr>
            <tr><td>02_segmented/</td><td>Cellpose segmentation masks per timepoint</td></tr>
            <tr><td>03_hyperstack/</td><td><strong>4D_hyperstack.tif</strong> - Final merged 4D image</td></tr>
            <tr><td>metadata/</td><td>JSON metadata files with voxel size information</td></tr>
            <tr><td>logs/</td><td>Processing logs for debugging</td></tr>
            <tr><td>reports/</td><td>This QC report</td></tr>
        </table>

        <div class="metric">
            <div class="metric-label">✅ Pipeline completed successfully</div>
        </div>

        <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
            <em>Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em>
        </p>
    </div>
</body>
</html>
'''

with open('pipeline_report.html', 'w') as f:
    f.write(html)

print("QC report generated successfully")
EOF
    """
}

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

workflow {
    // Parse input files with pattern: t####_Channel #.tif
    input_pattern = "${params.input_dir}/t*_Channel ${params.channel}.tif"

    input_channel = Channel
        .fromPath(input_pattern)
        .ifEmpty { error "No files found matching pattern: ${input_pattern}" }
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
    CELLPOSE_SEGMENT(
        PREPROCESS_DECONVOLVE.out.processed,
        shared_metadata,
        config.segmentation
    )

    // 4. Collect all segmented timepoints and merge into 4D hyperstack
    all_segmented = CELLPOSE_SEGMENT.out.segmented
        .map { timepoint, segmented_file -> segmented_file }
        .collect()

    MERGE_TO_HYPERSTACK(
        all_segmented,
        shared_metadata
    )

    // 5. Generate QC report
    all_logs = Channel.empty()

    if (config.roi_cropping.enabled) {
        all_logs = all_logs.mix(CROP_WITH_ROI.out.log)
    }

    all_logs = all_logs
        .mix(PREPROCESS_DECONVOLVE.out.log)
        .mix(CELLPOSE_SEGMENT.out.log)
        .collect()

    GENERATE_QC_REPORT(
        all_logs,
        MERGE_TO_HYPERSTACK.out.metadata,
        params.config_json
    )
}

// ============================================================================
// WORKFLOW COMPLETION
// ============================================================================

workflow.onComplete {
    def voxel_mode = config.voxel_size?.auto_detect ? "Auto-detected" : "Manual override"
    def roi_status = config.roi_cropping?.enabled ? "ENABLED" : "DISABLED"

    log.info """
    ============================================================================
    Pipeline completed!
    ============================================================================
    Status       : ${workflow.success ? 'SUCCESS ✓' : 'FAILED ✗'}
    Duration     : ${workflow.duration}
    Channel      : ${params.channel}
    ROI cropping : ${roi_status}
    Voxel mode   : ${voxel_mode}
    Output dir   : ${params.output_dir}

    Results:
      ${config.roi_cropping?.enabled ? "- Cropped images     : ${params.output_dir}/00_cropped/" : ""}
      - Preprocessed images : ${params.output_dir}/01_preprocessed/
      - Segmented masks     : ${params.output_dir}/02_segmented/
      - 4D Hyperstack       : ${params.output_dir}/03_hyperstack/4D_hyperstack.tif
      - QC report           : ${params.output_dir}/reports/pipeline_report.html
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
