#!/usr/bin/env nextflow

/*
 * ============================================================================
 * SPIM 4D Image Processing Pipeline - FIXED VERSION
 * ============================================================================
 *
 * This pipeline processes 4D SPIM images through:
 * 1. File parsing with timepoint extraction (t0051_Channel 1.tif format)
 * 2. Preprocessing and deconvolution per timepoint
 * 3. Cellpose segmentation per timepoint
 * 4. Merging all timepoints into a single 4D hyperstack with preserved metadata
 *
 * All parameters are loaded from a JSON configuration file for reproducibility.
 *
 * FIX: Improved metadata extraction with better error handling and micromamba activation
 */

nextflow.enable.dsl=2

// ============================================================================
// PARAMETER DEFINITIONS
// ============================================================================

params.input_dir = null
params.output_dir = null
params.config_json = null
params.channel = 1
params.container = 'docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-ed0bf47'
params.help = false

// Show help message
if (params.help) {
    log.info """
    ============================================================================
    SPIM 4D Image Processing Pipeline
    ============================================================================

    Usage:
        nextflow run spim_pipeline_fixed.nf \\
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
        --channel          Channel number to process (default: 1)
        --container        Singularity/Docker container image
                          (default: docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-ed0bf47)
        --help            Show this help message

    Output Structure:
        output_dir/
        ├── 01_preprocessed/     - Preprocessed & deconvolved images per timepoint
        ├── 02_segmented/        - Cellpose segmentation masks per timepoint
        ├── 03_hyperstack/       - Final 4D merged hyperstack with metadata
        ├── metadata/            - Preserved metadata files
        ├── logs/                - Processing logs
        └── reports/             - QC reports

    Example:
        nextflow run spim_pipeline_fixed.nf \\
            --input_dir /data/raw_images \\
            --output_dir /data/processed \\
            --config_json config.json \\
            --channel 1

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

// ============================================================================
// LOAD CONFIGURATION
// ============================================================================

def loadConfig(json_path) {
    def jsonSlurper = new groovy.json.JsonSlurper()
    return jsonSlurper.parse(new File(json_path))
}

config = loadConfig(params.config_json)

// ============================================================================
// WORKFLOW HEADER
// ============================================================================

log.info """
============================================================================
SPIM 4D Image Processing Pipeline - FIXED
============================================================================
Input directory    : ${params.input_dir}
Output directory   : ${params.output_dir}
Configuration      : ${params.config_json}
Channel to process : ${params.channel}
Container          : ${params.container}

Pipeline steps:
  1. Parse and sort timepoint files
  2. Preprocessing & Deconvolution per timepoint
  3. Cellpose Segmentation per timepoint
  4. Merge into 4D hyperstack with preserved metadata

Started at: ${new Date()}
============================================================================
""".stripIndent()

// ============================================================================
// PROCESS: Extract and Preserve Metadata (only from first timepoint)
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

    output:
    path "shared_metadata.json", emit: metadata

    container params.container

    script:
    def filename = image_file.name
    """
    #!/usr/bin/env bash
    set -euo pipefail

    echo "============================================"
    echo "Extracting metadata from: ${filename}"
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
    echo "Running metadata extraction..."

    # Create Python script as separate file for better debugging
    cat > extract_metadata.py << 'PYTHON_SCRIPT'
import sys
import json
import traceback

print("Starting metadata extraction...", file=sys.stderr)

try:
    import tifffile
    import numpy as np
    from pathlib import Path

    print("Imports successful", file=sys.stderr)

    filename = '${filename}'
    print(f"Processing file: {filename}", file=sys.stderr)

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

        # ImageJ metadata
        if tif.imagej_metadata:
            print("Found ImageJ metadata", file=sys.stderr)
            metadata['imagej'] = {
                'spacing': tif.imagej_metadata.get('spacing', 1.0),
                'unit': tif.imagej_metadata.get('unit', 'micron'),
                'axes': tif.imagej_metadata.get('axes', 'ZYX'),
                'slices': tif.imagej_metadata.get('slices', n_pages),
            }
        else:
            print("No ImageJ metadata, using defaults", file=sys.stderr)
            metadata['imagej'] = {
                'spacing': 1.0,
                'unit': 'micron',
                'axes': 'ZYX',
                'slices': n_pages,
            }

        # TIFF tags from first page
        tags = first_page.tags

        # Extract resolution
        if 'XResolution' in tags:
            x_num, x_denom = tags['XResolution'].value
            metadata['x_resolution_um'] = x_denom / x_num if x_num != 0 else 1.0
        else:
            metadata['x_resolution_um'] = 1.0

        if 'YResolution' in tags:
            y_num, y_denom = tags['YResolution'].value
            metadata['y_resolution_um'] = y_denom / y_num if y_num != 0 else 1.0
        else:
            metadata['y_resolution_um'] = 1.0

        # Image dimensions - from page info, not loading data
        metadata['shape'] = {
            'axes': 'ZYX',
            'dimensions': [n_pages, height, width]
        }

        # Data type from first page
        metadata['dtype'] = str(first_page.dtype)

        # Software info
        if 'Software' in tags:
            metadata['software'] = tags['Software'].value

    # Save metadata
    print("Saving metadata to JSON...", file=sys.stderr)
    with open('shared_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: shared_metadata.json")
    print(f"Image shape: {metadata['shape']['dimensions']} (ZYX)")
    print(json.dumps(metadata, indent=2))

    print("SUCCESS: Metadata extraction completed", file=sys.stderr)

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
    echo "Metadata extraction completed successfully"
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
    val preprocess_config

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_processed.tif"), emit: processed
    path "t${String.format('%04d', timepoint)}_preprocess.log", emit: log

    script:
    def cfg = preprocess_config
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    """
    #!/bin/bash
    set -euo pipefail

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "============================================"
    echo "Preprocessing timepoint: ${timepoint}"
    echo "File: ${filename}"
    echo "============================================"

    # Run preprocessing with all parameters from config
    python3 << 'PYTHON_EOF'
import json
import sys
import os
import subprocess

# Load metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load config
config = ${groovy.json.JsonOutput.toJson(cfg)}

# Build command for preprocessing script
cmd = [
    'python', 'spim_pipeline_fixed.py',
    '--input_file', '${filename}',
    '--outdir', '.',
    '--psf_path', config['psf_path'],
    '--image_scaling', str(config['image_scaling']),
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

    # Find and rename output to standard format
    SCALING_STR=\$(echo "${cfg.image_scaling}" | sed 's/\\.//g')
    ORIGINAL_OUTPUT=\$(ls *_\${SCALING_STR}*.tif 2>/dev/null | head -1)

    if [ -n "\$ORIGINAL_OUTPUT" ]; then
        mv "\$ORIGINAL_OUTPUT" "t${t_formatted}_processed.tif"
        echo "Renamed output to: t${t_formatted}_processed.tif"
    else
        echo "ERROR: No processed output found"
        ls -lh
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
x_res = metadata['x_resolution_um'] / ${cfg.image_scaling}
y_res = metadata['y_resolution_um'] / ${cfg.image_scaling}
z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

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
        'TimePoint': ${timepoint}
    }
)

print(f"Metadata restored for timepoint ${timepoint}:")
print(f"  X resolution: {x_res:.4f} um")
print(f"  Y resolution: {y_res:.4f} um")
print(f"  Z spacing: {z_spacing:.4f} um")
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

    # Build Cellpose command
    CELLPOSE_CMD="cellpose --image_path ${filename}"
    CELLPOSE_CMD+=" --savedir ."
    CELLPOSE_CMD+=" --pretrained_model ${cfg.model}"
    CELLPOSE_CMD+=" --diameter ${cfg.diameter}"
    CELLPOSE_CMD+=" --flow_threshold ${cfg.flow_threshold}"
    CELLPOSE_CMD+=" --cellprob_threshold ${cfg.cellprob_threshold}"

    ${cfg.use_gpu ? 'CELLPOSE_CMD+=" --use_gpu"' : ''}
    ${cfg.do_3d ? 'CELLPOSE_CMD+=" --do_3D"' : ''}
    ${cfg.save_tif ? 'CELLPOSE_CMD+=" --save_tif"' : ''}
    ${cfg.save_flows ? 'CELLPOSE_CMD+=" --save_flows"' : ''}
    ${cfg.save_npy ? '' : 'CELLPOSE_CMD+=" --no_npy"'}

    CELLPOSE_CMD+=" --verbose"

    echo "Running Cellpose: \$CELLPOSE_CMD"
    echo ""

    # Run Cellpose and capture output
    \$CELLPOSE_CMD 2>&1 | tee t${t_formatted}_segment.log

    # Find and rename Cellpose output
    CELLPOSE_OUTPUT=\$(ls *_cp_masks.tif 2>/dev/null | head -1)

    if [ -f "\$CELLPOSE_OUTPUT" ]; then
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

# Calculate voxel sizes (accounting for preprocessing scaling)
x_res = metadata['x_resolution_um'] / ${cfg.image_scaling}
y_res = metadata['y_resolution_um'] / ${cfg.image_scaling}
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
        'LabelImage': True
    }
)
print(f"Metadata preserved in segmentation mask for timepoint ${timepoint}")
PRESERVE_MASK_META

    else
        echo "ERROR: Cellpose output not found"
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
scaling = ${config.segmentation.image_scaling}
x_res = ref_metadata['x_resolution_um'] / scaling
y_res = ref_metadata['y_resolution_um'] / scaling
z_spacing = ref_metadata['imagej']['spacing'] if 'imagej' in ref_metadata else 1.0

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
        'unit': 'um'
    },
    'dtype': str(img_4d.dtype),
    'n_timepoints': img_4d.shape[0],
    'is_label_image': True,
    'processing': {
        'preprocessing_scaling': ${config.preprocessing.image_scaling},
        'segmentation_scaling': scaling,
        'cellpose_diameter': ${config.segmentation.diameter},
        'cellpose_model': '${config.segmentation.model}'
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
        'LabelImage': True
    }
)

print("\\n" + "="*60)
print("4D Hyperstack created successfully!")
print("="*60)
print(f"Shape: {img_4d.shape} (TZYX)")
print(f"Voxel size: {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} um")
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
preprocess_logs = list(Path('.').glob('*_preprocess.log'))
segment_logs = list(Path('.').glob('*_segment.log'))

# Generate summary
summary = {
    'pipeline_version': '1.0.0-fixed',
    'execution_date': datetime.now().isoformat(),
    'input_channel': ${params.channel},
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 SPIM 4D Image Processing Pipeline Report</h1>

        <div class="metric">
            <div class="metric-value">{hyperstack_meta['n_timepoints']} Timepoints</div>
            <div class="metric-label">Successfully processed and merged into 4D hyperstack</div>
        </div>

        <h2>📊 Execution Summary</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Pipeline Version</td><td>{summary['pipeline_version']}</td></tr>
            <tr><td>Execution Date</td><td>{summary['execution_date']}</td></tr>
            <tr><td>Input Channel</td><td class="info">{summary['input_channel']}</td></tr>
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
            <tr><td>X Resolution</td><td>{hyperstack_meta['voxel_size']['x_um']:.4f} µm</td></tr>
            <tr><td>Y Resolution</td><td>{hyperstack_meta['voxel_size']['y_um']:.4f} µm</td></tr>
            <tr><td>Z Spacing</td><td>{hyperstack_meta['voxel_size']['z_um']:.4f} µm</td></tr>
            <tr><td>Data Type</td><td>{hyperstack_meta['dtype']}</td></tr>
            <tr><td>Label Image</td><td class="success">Yes (segmentation masks)</td></tr>
        </table>

        <h2>⚙️ Processing Configuration</h2>
        <h3>Preprocessing</h3>
        <pre>{json.dumps(config['preprocessing'], indent=2)}</pre>

        <h3>Segmentation (Cellpose)</h3>
        <pre>{json.dumps(config['segmentation'], indent=2)}</pre>

        <h2>📋 Pipeline Steps</h2>
        <ol>
            <li><strong>File Parsing</strong> - Extracted timepoints from filename pattern (t####_Channel #.tif)</li>
            <li><strong>Metadata Extraction</strong> - Preserved original image metadata including voxel sizes</li>
            <li><strong>Preprocessing & Deconvolution</strong> - Applied corrections and deconvolution per timepoint</li>
            <li><strong>Cellpose Segmentation</strong> - 3D cell segmentation per timepoint</li>
            <li><strong>Hyperstack Merging</strong> - Combined all timepoints into single 4D TIFF with preserved metadata</li>
        </ol>

        <h2>📂 Output Files</h2>
        <table>
            <tr><th>Directory</th><th>Contents</th></tr>
            <tr><td>01_preprocessed/</td><td>Preprocessed and deconvolved images per timepoint</td></tr>
            <tr><td>02_segmented/</td><td>Cellpose segmentation masks per timepoint</td></tr>
            <tr><td>03_hyperstack/</td><td><strong>4D_hyperstack.tif</strong> - Final merged 4D image</td></tr>
            <tr><td>metadata/</td><td>JSON metadata files per timepoint</td></tr>
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
    // Extract timepoint number and filter by channel

    input_pattern = "${params.input_dir}/t*_Channel ${params.channel}.tif"

    input_channel = Channel
        .fromPath(input_pattern)
        .ifEmpty { error "No files found matching pattern: ${input_pattern}" }
        .map { file ->
            // Extract timepoint number from filename
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

    // 1. Extract metadata from FIRST timepoint only (all have same metadata)
    first_timepoint = input_channel.first()
    EXTRACT_METADATA(first_timepoint)

    // Share the same metadata with all timepoints
    shared_metadata = EXTRACT_METADATA.out.metadata

    // 2. Preprocess and deconvolve each timepoint
    PREPROCESS_DECONVOLVE(
        input_channel,
        shared_metadata,
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
    all_logs = PREPROCESS_DECONVOLVE.out.log
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
    log.info """
    ============================================================================
    Pipeline completed!
    ============================================================================
    Status       : ${workflow.success ? 'SUCCESS ✓' : 'FAILED ✗'}
    Duration     : ${workflow.duration}
    Channel      : ${params.channel}
    Output dir   : ${params.output_dir}

    Results:
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
