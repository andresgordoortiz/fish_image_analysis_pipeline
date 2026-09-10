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
// Modular preprocessing scripts (lean, single-purpose, ported from AIAF-32).
// Each one is a small CLI: read TIFF -> apply one correction -> write TIFF.
// They chain in the workflow as: planar -> depth -> isotropic.
params.planar_correction_script = './bin/planar_intensity_correction.py'
params.depth_correction_script  = './bin/depth_intensity_correction.py'
params.isotropic_resample_script = './bin/isotropic_resample.py'
params.merge_script = './merge_hyperstack.py'
params.benchmark_script = './benchmark_pipeline.py'
params.prep_ultrack_script = './prep_ultrack_cellpose.py'
params.debug_nuclei_script = './debug_nuclei_tracking.py'
params.debug_report_script = './debug_nuclei_report.py'
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

// Existence checks for the modular preprocessing scripts. Fail fast at
// launch (before submitting any SLURM jobs) if any of them are missing.
[params.planar_correction_script, params.depth_correction_script, params.isotropic_resample_script].each { script_path ->
    if (!file(script_path).exists()) {
        log.error "Modular preprocessing script not found: ${script_path}"
        exit 1
    }
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

// Resolve input path. Accept either:
//   - a directory of per-timepoint TIFFs / single hyperstack
//   - a single .czi / .tif / .tiff hyperstack file (we then use its parent
//     directory and treat the filename as input.file)
// IMPORTANT: params.X can only be assigned once in Nextflow, so we figure out
// the final values in plain locals first, then bind them to params.* below.
def _raw_input_dir  = sanitizePath(config.input.directory)
def _raw_input_file = config.input?.file ? sanitizePath(config.input.file) : null

def _input_path_file = file(_raw_input_dir)
if (!_input_path_file.exists()) {
    log.error "Input path not found: ${_raw_input_dir} (resolved to: ${_input_path_file})"
    log.error "If the path contains spaces, use plain spaces in config.json (not backslash-escaped)"
    exit 1
}
if (!_input_path_file.isDirectory()) {
    def fname = _input_path_file.getName().toLowerCase()
    if (!(fname.endsWith('.czi') || fname.endsWith('.tif') || fname.endsWith('.tiff'))) {
        log.error "input.directory points at a file but it is not .czi/.tif/.tiff: ${_raw_input_dir}"
        exit 1
    }
    log.info "input.directory points at a single file — using its parent directory as the input dir"
    _raw_input_file = _input_path_file.getName()
    _raw_input_dir  = _input_path_file.getParent().toString()
}

params.input_dir   = _raw_input_dir
// Resolve output_dir to an ABSOLUTE path. The user-facing config.json
// value is typically a relative path like "./results_mk_2508_stephane/"
// (relative to the Nextflow launch directory). Nextflow's `publishDir`
// evaluates its argument in the task's work dir context — if the path
// is relative it gets resolved against the work dir instead of the
// launch dir, and since the work dir lives inside the read-only
// container overlay, `publishDir` silently falls back to writing the
// bare filenames at the launch directory root (we saw this on
// 2026-09-09: ~190 t####_*.tif files scattered across the repo dir
// while ./results_mk_2508_stephane/ held only logs and metadata).
// `file(...).toAbsolutePath()` pins the path to the launch dir so
// every publishDir writes into the intended results tree regardless
// of where Nextflow evaluates it.
params.output_dir  = file(sanitizePath(config.output.directory)).toAbsolutePath().toString()
params.channel     = config.input?.channel ?: 0  // 0 = auto-detect (required when file has only 1 channel)
params.input_file  = _raw_input_file
// Container images are ALWAYS taken from nextflow.config (shared long-term
// folder on the cluster). Any container_image / fiji_container_image /
// ultrack_container values in config.json are intentionally ignored so every
// run uses the same pre-pulled images.
if (config.system?.container_image && config.system.container_image != params.container) {
    log.warn "Ignoring config.system.container_image (${config.system.container_image}) — using hardcoded ${params.container}"
}
if (config.system?.fiji_container_image && config.system.fiji_container_image != params.fiji_container) {
    log.warn "Ignoring config.system.fiji_container_image (${config.system.fiji_container_image}) — using hardcoded ${params.fiji_container}"
}
if (config.tracking?.ultrack_container && config.tracking.ultrack_container != params.ultrack_container) {
    log.warn "Ignoring config.tracking.ultrack_container (${config.tracking.ultrack_container}) — using hardcoded ${params.ultrack_container}"
}

def input_dir_file = file(params.input_dir)
if (!input_dir_file.isDirectory()) {
    log.error "Resolved input directory is not a directory: ${params.input_dir}"
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

// Set defaults for skip_merge, skip_preprocessing and downscale_labels
def skip_merge = config.output?.skip_merge ?: false
def skip_segmentation = config.segmentation?.enabled == false
// Two equivalent ways to disable preprocessing:
//   - preprocessing.enabled = false        (preferred, matches segmentation/tracking)
//   - preprocessing.skip_preprocessing = true   (legacy)
def skip_preprocessing = (config.preprocessing?.enabled == false) || (config.preprocessing?.skip_preprocessing ?: false)
def preprocessed_dir = config.preprocessing?.preprocessed_dir ?: null
def downscale_labels = config.segmentation?.downscale_labels != null ? config.segmentation.downscale_labels : 1.0

// Downscaling toggle — independent of preprocessing.enabled. When enabled
// and preprocessing is skipped, runs DOWNSCALE_XY (XY-only cubic rescale,
// optionally plus isotropic Z reslice in the same pass).
def downscaling_enabled = config.downscaling?.enabled == true
def downscaling_factor = config.downscaling?.factor != null ? (config.downscaling.factor as Double) : 1.0d
if (downscaling_enabled && config.downscaling?.factor == null) {
    log.error "downscaling.enabled=true requires downscaling.factor"
    exit 1
}

// Raw export — produce a downscaled + isotropic version of the RAW (unprocessed)
// input for overlaying with tracks in ultrack_viewer.py. Runs INDEPENDENTLY of
// preprocessing/segmentation/tracking; the preprocessed chain still operates on
// the original-resolution raw input. Purely additive.
def raw_export_enabled  = config.raw_export?.enabled == true
def raw_export_factor   = config.raw_export?.factor != null ? (config.raw_export.factor as Double) : 0.33d
def raw_export_iso      = config.raw_export?.isotropic_reslice != null ? (config.raw_export.isotropic_reslice as Boolean) : true
if (raw_export_enabled && (raw_export_factor <= 0.0d || raw_export_factor > 1.0d)) {
    log.error "raw_export.factor must be greater than 0 and at most 1.0, got: ${raw_export_factor}"
    exit 1
}
if (downscaling_factor <= 0.0d || downscaling_factor > 1.0d) {
    log.error "downscaling.factor must be greater than 0 and at most 1.0, got: ${downscaling_factor}"
    exit 1
}
def effective_scaling = downscaling_enabled ? downscaling_factor : 1.0d
def run_standalone_downscaling = skip_preprocessing && effective_scaling < 1.0d

// Modular preprocessing parameters. Defaults match the AIAF-32 calibrated
// values for typical SPIM embryo stacks (p99 + window 9 + isotropic at the
// smallest XY pixel size). Each parameter is optional in config.json — when
// absent we fall back to these defaults.
def planar_sigma_xy   = (config.preprocessing?.planar?.sigma_xy     != null) ? (config.preprocessing.planar.sigma_xy     as Double) : 64.0d
def depth_mode        = (config.preprocessing?.depth?.mode          ?: 'p99').toString()
def depth_smooth      = (config.preprocessing?.depth?.smooth_window != null) ? (config.preprocessing.depth.smooth_window as Integer) : 9
def depth_gain_min    = (config.preprocessing?.depth?.gain_min      != null) ? (config.preprocessing.depth.gain_min      as Double) : 0.25d
def depth_gain_max    = (config.preprocessing?.depth?.gain_max      != null) ? (config.preprocessing.depth.gain_max      as Double) : 4.0d
def iso_target_um     = (config.preprocessing?.isotropic?.target_um != null) ? (config.preprocessing.isotropic.target_um as Double) : 0.374d
def iso_order         = (config.preprocessing?.isotropic?.order     != null) ? (config.preprocessing.isotropic.order     as Integer) : 3

// Canonical voxel sizes from EXTRACT_METADATA. Declared at script scope so
// they are visible from inside every process's ``script:`` block (process
// bodies are interpolated lazily when the process is invoked, but they
// capture Groovy script-level variables the same way closures do). The
// actual values are assigned inside ``workflow { }`` right after
// EXTRACT_METADATA writes shared_metadata.json to disk; until then we
// keep conservative defaults so a stray reference during parse-time
// validation doesn't trip up Nextflow's pre-flight checks. See the
// in-workflow comment for the rationale and the source-of-truth
// provenance (short version: don't trust the input TIFF's ImageJ block
// for physical voxel sizes; the user's config.json is canonical).
double metadata_voxel_x = 0.347d
double metadata_voxel_y = 0.347d
double metadata_voxel_z = 2.0d
if (!skip_preprocessing) {
    log.info "Preprocessing method: MODULAR (planar sigma=${planar_sigma_xy} px, depth=${depth_mode}/w${depth_smooth}, isotropic=${iso_target_um} µm/order=${iso_order})"
}

// Optional: limit input to the first N timepoints (after sorting). Useful when
// late timepoints in an acquisition are unusable (sample drift, photodamage,
// etc.) and you want to discard them without rebuilding the input dataset.
// Accepts an integer >= 1; null/missing/<=0 means "use all timepoints".
def max_timepoints = config.input?.max_timepoints != null \
    ? (config.input.max_timepoints as Integer) \
    : null
if (max_timepoints != null && max_timepoints <= 0) {
    max_timepoints = null
}

// When preprocessing is skipped, raw acquisitions are typically anisotropic
// (Z spacing >> XY spacing). Cellpose-SAM with --do_3D handles anisotropy
// internally for inference but does NOT necessarily emit masks at the input
// shape, so the segmented hyperstack and the raw hyperstack may end up with
// different Z dimensions. This flag enables a lightweight Z-only linear
// resampling that makes voxels isotropic on the way in, so raw, segmented
// and tracked outputs all share the same XYZ shape and can be overlaid.
// Default: true when skip_preprocessing is on, false otherwise.
def isotropic_reslice = config.preprocessing?.isotropic_reslice != null \
    ? (config.preprocessing.isotropic_reslice as Boolean) \
    : skip_preprocessing

// Validate skip_preprocessing config
if (skip_preprocessing) {
    if (skip_segmentation) {
        log.error "skip_preprocessing=true requires segmentation.enabled=true (nothing to do otherwise)"
        exit 1
    }
    if (preprocessed_dir) {
        def preproc_dir_file = file(preprocessed_dir)
        if (!preproc_dir_file.exists()) {
            log.error "skip_preprocessing=true but preprocessed_dir not found: ${preprocessed_dir} (resolved to: ${preproc_dir_file})"
            exit 1
        }
    }
}

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

// Debug-preprocessing (per-stage nuclei tracking) was tied to the
// intermediate TIFs emitted by the old monolithic PREPROCESS_DECONVOLVE /
// PREPROCESS_SELFNET. The new modular pipeline has no intermediates to
// inspect (each step is its own process), so this feature is no longer
// available. The standalone debug_nuclei_*.py scripts are kept in the repo
// for ad-hoc QA against manually-provided stacks.
def run_debug_preprocessing = false


// Validate tracking config if enabled
if (!skip_tracking) {
    if (!params.ultrack_container) {
        log.error "Tracking enabled but no ultrack_container resolved (check params.ultrack_container in nextflow.config)"
        exit 1
    }
    if (!file(params.ultrack_container).exists()) {
        log.error "Ultrack container not found: ${params.ultrack_container}"
        exit 1
    }
    // Always prefer the ultrack_config.toml shipped in the repo root.
    // Allow an explicit override via config.tracking.ultrack_config_toml, but
    // the default is the file checked in next to spim_pipeline.nf.
    def toml_path = config.tracking.ultrack_config_toml ?: "${workflow.projectDir}/ultrack_config.toml"
    if (!file(toml_path).exists()) {
        log.error "Ultrack config TOML not found: ${toml_path}"
        exit 1
    }
    config.tracking.ultrack_config_toml = toml_path
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
def xy_downscale_info = downscaling_enabled ? "Enabled (factor=${effective_scaling})" : "Disabled"
def label_downscale_info = downscale_labels < 1.0 ? "${downscale_labels} (Fiji nearest-neighbor)" : "Disabled"
def seg_mode_info = config.segmentation.do_3d ? "3D" : (config.segmentation.stitch_threshold != null ? "2D+Stitch(${config.segmentation.stitch_threshold})" : "2D")
def tracking_info = skip_tracking ? "SKIPPED" : "Enabled (ultrack)"
def debug_info = "Disabled (modular pipeline — no intermediates to inspect)"
def preproc_info = skip_preprocessing ? (preprocessed_dir ? "SKIPPED (using ${preprocessed_dir})" : "SKIPPED (using raw input)") : "Enabled"
def raw_export_info = raw_export_enabled ? "Enabled (factor=${raw_export_factor}, iso=${raw_export_iso})" : "Disabled"

log.info """
================================================
SPIM Pipeline - IMP Vienna (vanilla)
================================================
Input        : ${params.input_dir}
Output       : ${params.output_dir}
Channel      : ${params.channel}
ROI Cropping : ${roi_info}
Voxel Size   : ${voxel_info}
Merge        : ${merge_info}
XY downscale : ${xy_downscale_info}
Label downscale : ${label_downscale_info}
Seg mode     : ${seg_mode_info}
Preprocess   : ${preproc_info}
Tracking     : ${tracking_info}
Debug preproc: ${debug_info}
Raw export   : ${raw_export_info}
================================================
""".stripIndent()

// ============================================================================
// PROCESS: Split a single CZI / hyperstack TIFF into per-timepoint TIFFs
// ============================================================================
// Used when the input is NOT a directory of per-timepoint files but instead
// a single .czi (Zeiss) or 4D/5D ImageJ/OME hyperstack .tif. The process
// extracts the requested channel and writes one ZYX TIFF per timepoint named
// "t####_Channel <c>.tif" so the rest of the pipeline can consume them
// unchanged.

process SPLIT_INPUT_FILE {
    tag { input_file.name }

    // Splitting a multi-GB hyperstack is expensive and deterministic; failures
    // are almost always config/data problems that won't fix themselves on
    // retry. Fail fast.
    maxRetries 0
    errorStrategy 'terminate'

    // Splitting parallelises across timepoints with a thread pool. Allow the
    // user to override via process.cpus in nextflow.config; default 4.
    cpus { params.split_cpus ?: 4 }

    publishDir "${params.output_dir}/00_split_input",
        mode: 'copy',
        pattern: "t*_Channel*.tif"
    publishDir "${params.output_dir}/logs/split_input",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    path input_file
    val channel_idx

    output:
    path "t*_Channel*.tif", emit: timepoints
    path "split_input.log", emit: log

    script:
    def filename = input_file.name
    """
    #!/bin/bash
    set -euo pipefail
    exec > >(tee split_input.log) 2>&1

    echo "============================================"
    echo "Splitting hyperstack input: ${filename}"
    echo "Channel index requested: ${channel_idx}"
    echo "============================================"

    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    # Expose Nextflow's CPU allocation to the Python splitter so it can size
    # its thread pool appropriately.
    export NXF_TASK_CPUS=${task.cpus}

    python3 --version

    # czifile is a small pure-python lib that handles most CZI files.
    # Install at runtime (same pattern as CROP_WITH_ROI for read-roi).
    if [[ "${filename}" == *.czi || "${filename}" == *.CZI ]]; then
        python3 -c "import czifile" 2>/dev/null || {
            echo "Installing czifile..."
            pip install czifile --break-system-packages
        }
    fi

    # zarr is required for lazy, low-memory TIFF hyperstack splitting.
    python3 -c "import zarr" 2>/dev/null || {
        echo "Installing zarr (for lazy TIFF reads)..."
        pip install 'zarr<3' --break-system-packages || pip install zarr --break-system-packages
    }

    python3 << 'PYTHON_SPLIT_SCRIPT'
import sys
import os
import numpy as np
import tifffile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

input_file = '${filename}'
_channel_cfg = ${channel_idx}  # 1-based; 0 means auto-detect

# Use Nextflow-allocated CPU count for parallel per-timepoint writes.
# Fall back to a conservative default if the env var is missing.
try:
    n_workers = max(1, int(os.environ.get('NXF_TASK_CPUS', os.cpu_count() or 1)))
except Exception:
    n_workers = 1
print(f"Input file: {input_file}")
print(f"Channel config (0=auto): {_channel_cfg}")
print(f"Worker threads:          {n_workers}")


def resolve_channel(n_channels_in_file, cfg):
    '''Return (channel_1based, ch_idx_0based), auto-selecting when cfg==0.'''
    if cfg == 0:
        if n_channels_in_file == 1:
            print("  Auto-detected single channel -> using channel 1")
            return 1, 0
        raise ValueError(
            f"File has {n_channels_in_file} channels but 'input.channel' is not set "
            f"in config.json. Please add 'channel': <1..{n_channels_in_file}> "
            f"under the 'input' section."
        )
    if cfg < 1:
        raise ValueError(f"channel must be >= 1 (got {cfg})")
    if cfg > n_channels_in_file:
        raise ValueError(
            f"Channel {cfg} out of range (file has {n_channels_in_file} channels: 1..{n_channels_in_file})"
        )
    return cfg, cfg - 1


ext = Path(input_file).suffix.lower()


def _to_uint16(arr):
    '''Cast/clip a 3D stack to uint16 without doubling memory unnecessarily.'''
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype.kind == 'f' or (arr.dtype.itemsize > 2 and arr.dtype.kind in ('u', 'i')):
        return np.clip(arr, 0, 65535).astype(np.uint16)
    return arr.astype(np.uint16, copy=False)


def save_timepoint(stack3d, t_idx, channel):
    '''Save a 3D ZYX stack as t####_Channel <c>.tif (ImageJ hyperstack).'''
    out_name = f"t{t_idx:04d}_Channel {channel}.tif"
    stack3d = _to_uint16(stack3d)
    tifffile.imwrite(
        out_name,
        stack3d,
        imagej=True,
        metadata={'axes': 'ZYX'},
    )
    print(f"  -> {out_name}  shape={stack3d.shape} dtype={stack3d.dtype}", flush=True)


def split_czi(path):
    '''Stream-split a CZI hyperstack one timepoint at a time.

    czifile.CziFile.asarray() materialises the entire 5D array in RAM, which
    is infeasible for multi-hundred-GB acquisitions. Instead we walk the
    subblock directory and, for each timepoint of the requested channel,
    allocate only a (Z, Y, X) buffer and fill it from the relevant subblocks.
    Timepoints are processed in parallel using a thread pool, but the actual
    subblock reads are serialized with a lock: czifile shares a single file
    handle and seek()s into it inside data_segment(), so concurrent reads
    race and raise "SegmentNotFoundError: not a ZISRAW segment". The lock
    only covers the I/O; decompression of the returned numpy array and the
    final TIFF write still run in parallel.
    '''
    import czifile
    import threading

    print("Reading CZI metadata (no full-array load)...")
    czi = czifile.CziFile(path)
    czi_read_lock = threading.Lock()
    try:
        axes = czi.axes  # e.g. 'BCTZYX0' or 'STCZYX'
        shape = czi.shape
        dtype = np.dtype(czi.dtype)
        print(f"  CZI axes:  {axes}")
        print(f"  CZI shape: {shape}")
        print(f"  CZI dtype: {dtype}")

        # Map axis name -> index into czi.shape / subblock start vectors
        axis_index = {a: i for i, a in enumerate(axes)}

        def axis_size(a, default=1):
            return shape[axis_index[a]] if a in axis_index else default

        nT = axis_size('T')
        nC = axis_size('C')
        nZ = axis_size('Z')
        nY = axis_size('Y')
        nX = axis_size('X')

        channel, ch_idx = resolve_channel(nC, _channel_cfg)
        print(f"  Using channel {channel} of {nC}")

        # Group subblock directory entries by timepoint for the requested channel.
        # Each DirectoryEntry has `.start` (one int per axis in czi.axes order)
        # and `.shape` plus a `.data_segment().data()` accessor.
        t_axis = axis_index.get('T')
        c_axis = axis_index.get('C')
        z_axis = axis_index.get('Z')
        y_axis = axis_index['Y']
        x_axis = axis_index['X']

        from collections import defaultdict
        per_t_entries = defaultdict(list)
        for entry in czi.filtered_subblock_directory:
            if c_axis is not None and entry.start[c_axis] != ch_idx:
                continue
            t = entry.start[t_axis] if t_axis is not None else 0
            per_t_entries[t].append(entry)

        if not per_t_entries:
            raise RuntimeError("No subblocks found for requested channel")

        print(f"Splitting {nT} timepoints (channel {channel} of {nC}) "
              f"with {n_workers} worker thread(s)...")

        def process_timepoint(t):
            entries = per_t_entries.get(t, [])
            if not entries:
                raise RuntimeError(f"No subblocks for timepoint {t}")
            # Allocate ONE timepoint at a time: (Z, Y, X)
            buf = np.zeros((nZ, nY, nX), dtype=dtype)
            for entry in entries:
                z = entry.start[z_axis] if z_axis is not None else 0
                y = entry.start[y_axis]
                x = entry.start[x_axis]
                # Subblock data is shaped to match entry.shape across czi.axes;
                # squeeze it down to (sz, sy, sx).
                # czifile.CziFile is NOT thread-safe: data_segment() seeks on
                # the shared file handle, so concurrent calls produce
                # "not a ZISRAW segment" errors. Serialize the actual read.
                with czi_read_lock:
                    tile = entry.data_segment().data()
                # tile.shape mirrors czi.axes; collapse all non-ZYX axes (size 1)
                sl = []
                for a, sz in zip(axes, tile.shape):
                    if a in ('Z', 'Y', 'X'):
                        sl.append(slice(None))
                    else:
                        sl.append(0)
                tile = tile[tuple(sl)]
                if tile.ndim == 2:
                    tile = tile[None, ...]
                sz, sy, sx = tile.shape
                buf[z:z + sz, y:y + sy, x:x + sx] = tile
            save_timepoint(buf, t, channel)
            return t

        timepoints = sorted(per_t_entries.keys())
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_timepoint, t): t for t in timepoints}
            for fut in as_completed(futs):
                fut.result()  # propagate exceptions
    finally:
        czi.close()


def split_tiff(path):
    '''Stream-split a TIFF hyperstack one timepoint at a time using tifffile's
    zarr store, which performs lazy per-page reads instead of loading the
    whole series into RAM.'''
    print("Reading TIFF metadata (lazy zarr store)...")
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes  # e.g. 'TZYX', 'TCZYX', 'ZYX'
        shape = series.shape
        print(f"  TIFF series axes:  {axes}")
        print(f"  TIFF series shape: {shape}")

        nC_here = shape[axes.index('C')] if 'C' in axes else 1
        channel, ch_idx = resolve_channel(nC_here, _channel_cfg)
        print(f"  Using channel {channel} of {nC_here}")

        # Per-timepoint files are 3D (ZYX or YX) - no splitting needed.
        if 'T' not in axes:
            print("  No T axis -> file is already a single timepoint, copying as t0000.")
            data = series.asarray()
            if data.ndim == 2:
                data = data[None, ...]  # YX -> 1YX
            if 'C' in axes:
                data = np.take(data, ch_idx, axis=axes.index('C'))
            save_timepoint(data, 0, channel)
            return

        # Open the series as a (read-only) zarr store. Slicing this store only
        # reads the needed TIFF pages from disk, so peak RAM stays at one
        # timepoint per worker thread instead of the whole hyperstack.
        try:
            import zarr
            store = series.aszarr()
            zarr_arr = zarr.open(store, mode='r')
            use_zarr = True
        except Exception as e:
            print(f"  WARNING: zarr-backed lazy access unavailable ({e}); "
                  f"falling back to per-page reads.")
            use_zarr = False
            zarr_arr = None

        # Build axis index map
        axis_index = {a: i for i, a in enumerate(axes)}
        nT = shape[axis_index['T']]

        def slice_for_t(t):
            '''Build an indexing tuple selecting (t, ch_idx) from the lazy array.'''
            idx = []
            for a in axes:
                if a == 'T':
                    idx.append(t)
                elif a == 'C':
                    idx.append(ch_idx)
                else:
                    idx.append(slice(None))
            return tuple(idx)

        print(f"Splitting {nT} timepoints (channel {channel} of {nC_here}) "
              f"with {n_workers} worker thread(s)...")

        if use_zarr:
            # tifffile's zarr store is thread-safe for independent reads.
            def process_timepoint(t):
                stack = np.asarray(zarr_arr[slice_for_t(t)])
                # Ensure ZYX layout
                if stack.ndim == 2:
                    stack = stack[None, ...]
                save_timepoint(stack, t, channel)
                return t

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(process_timepoint, t): t for t in range(nT)}
                for fut in as_completed(futs):
                    fut.result()
        else:
            # Fallback: read each timepoint sequentially via series.asarray
            # with explicit indexing - still avoids loading the full stack.
            for t in range(nT):
                stack = series.asarray(key=t) if 'C' not in axes else None
                if stack is None:
                    raise RuntimeError(
                        "Lazy TIFF access requires the 'zarr' package; "
                        "please install zarr in the container."
                    )
                if stack.ndim == 2:
                    stack = stack[None, ...]
                save_timepoint(stack, t, channel)


try:
    if ext in ('.czi',):
        split_czi(input_file)
    elif ext in ('.tif', '.tiff'):
        split_tiff(input_file)
    else:
        print(f"ERROR: Unsupported file extension: {ext}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR splitting input: {e}", file=sys.stderr)
    sys.exit(1)

print("Split complete.")
PYTHON_SPLIT_SCRIPT

    n_out=\$(ls -1 t*_Channel*.tif 2>/dev/null | wc -l || true)
    echo "Produced \$n_out per-timepoint TIFF(s)."
    if [ "\$n_out" -eq 0 ]; then
        echo "ERROR: split produced no output files"
        exit 1
    fi
    """
}

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
// PROCESS: Isotropic Z reslicing (lightweight; used when full preprocessing
// is skipped but you still need Z to match XY for overlaying raw + tracks)
// ============================================================================

process RESLICE_ISOTROPIC {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/00b_isotropic",
        mode: 'copy',
        pattern: "t*_iso_Channel*.tif"
    publishDir "${params.output_dir}/logs/isotropic",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_iso_Channel*.tif"), emit: resliced
    path "t${String.format('%04d', timepoint)}_isotropic.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    """
    #!/bin/bash
    set -euo pipefail
    exec > >(tee t${t_formatted}_isotropic.log) 2>&1

    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 << 'PYTHON_EOF'
import json
import sys
import numpy as np
import tifffile
from scipy.ndimage import zoom

with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

xy_pixel = float(metadata['x_resolution_um'])
z_pixel = float(metadata['imagej']['spacing']) if 'imagej' in metadata else 1.0
print(f"XY pixel size: {xy_pixel:.4f} um")
print(f"Z  pixel size: {z_pixel:.4f} um")

img = tifffile.imread('${filename}')
if img.ndim != 3:
    raise SystemExit(f"Expected 3D ZYX input, got shape {img.shape}")
print(f"Input shape: {img.shape}, dtype: {img.dtype}")

# Defensive: refuse to reslice an already-resliced file. This protects
# against accidental reuse of published outputs as new inputs (e.g. when
# pointing input.directory at a folder that contains prior pipeline outputs).
already_iso = False
try:
    with tifffile.TiffFile('${filename}') as _t:
        _ij = _t.imagej_metadata or {}
        if _ij.get('IsotropicResliced'):
            already_iso = True
except Exception:
    pass
if already_iso:
    raise SystemExit(
        "Refusing to reslice: input already has 'IsotropicResliced=True' in its "
        "ImageJ metadata. This usually means you fed published reslice outputs "
        "back in as raw input. Wipe the published 00b_isotropic/ directory and "
        "re-run from clean inputs."
    )

zoom_z = z_pixel / xy_pixel
if abs(zoom_z - 1.0) < 1e-3:
    print("Already isotropic, copying through.")
    out = img
else:
    print(f"Z zoom factor: {zoom_z:.4f}  ({img.shape[0]} -> ~{int(round(img.shape[0]*zoom_z))} slices)")
    out = zoom(img, (zoom_z, 1.0, 1.0), order=1, prefilter=False)

if out.dtype != np.uint16:
    # Promote to a wide enough dtype before clipping; np.clip with bounds
    # outside the source dtype range raises OverflowError on uint8 etc.
    out = np.clip(out.astype(np.int32), 0, 65535).astype(np.uint16)
print(f"Output shape: {out.shape}, dtype: {out.dtype}")

# Preserve original Channel <c> filename so downstream stages keep working
# NOTE: no backslashes in this heredoc body — Nextflow 25.04.7's bash
# heredoc pass-through was eating backslashes and breaking regex/string
# escapes (see repo memory 2026-09-09). We split on the literal
# _Channel prefix instead of using a regex.
channel = '1'
_idx = '${filename}'.find('_Channel ')
if _idx >= 0:
    _rest = '${filename}'[_idx + len('_Channel '):]
    _digits = ''
    for _ch in _rest:
        if _ch.isdigit():
            _digits += _ch
        else:
            break
    if _digits:
        channel = _digits
out_name = f"t${t_formatted}_iso_Channel {channel}.tif"

tifffile.imwrite(
    out_name,
    out,
    imagej=True,
    resolution=(1.0/xy_pixel, 1.0/xy_pixel),
    metadata={
        'spacing': xy_pixel,
        'unit': 'um',
        'axes': 'ZYX',
        'TimePoint': ${timepoint},
        'WasROICropped': metadata.get('was_roi_cropped', False),
        'IsotropicResliced': True,
    },
)
print(f"Wrote {out_name}")
PYTHON_EOF
    """
}

// ============================================================================
// PROCESS: XY Downscaling (standalone; runs when preprocessing.enabled=false
// but downscaling.enabled=true). Mirrors RESLICE_ISOTROPIC structurally.
// Optionally performs isotropic Z resampling in the same pass.
// ============================================================================

process DOWNSCALE_XY {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/00c_downscaled",
        mode: 'copy',
        pattern: "t*_dscale_Channel*.tif"
    publishDir "${params.output_dir}/logs/xy_downscaling",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json
    val scale_factor
    val reslice_isotropic
    path bin_dir            // entire bin/ so _tiff_io.py is importable

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_dscale_Channel*.tif"), emit: downscaled
    path "t${String.format('%04d', timepoint)}_downscale_xy.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    def scale_factor_py = (scale_factor as Double).toString()
    // Wrap in single quotes so Nextflow's GString interpolation emits
    // 'True' / 'False' (quoted) into the Python heredoc. Without the
    // wrapping, ${reslice_py} renders as the bare token `True`, which
    // makes `do_iso = (True == 'True')` always False in Python (different
    // types). Symptom: isotropic reslice is silently skipped even when
    // isotropic_reslice=true is set in config.
    def reslice_py = reslice_isotropic ? "'True'" : "'False'"
    """
    #!/bin/bash
    set -euo pipefail
    exec > >(tee t${t_formatted}_downscale_xy.log) 2>&1

    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 << 'PYTHON_EOF'
import json
import re as _re
import sys as _sys
import numpy as np
import tifffile
from skimage.transform import rescale

# Make the staged bin/ importable so we can use _tiff_io.read_tiff / write_tiff
# for correct metadata round-tripping (the in-script tifffile.imread here
# is only used for the actual data array; the read_tiff call below gives
# us the real voxel sizes from the input TIFF's metadata).
_sys.path.insert(0, 'bin')
from _tiff_io import VoxelSizes, read_tiff, write_tiff

with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Canonical voxel sizes from the user's config.json (passed in as
# GString interpolations of the script-scope doubles metadata_voxel_x,
# metadata_voxel_y, metadata_voxel_z defined near the top of
# spim_pipeline.nf). These are the raw-input sizes and are used as
# the authoritative XY (raw input µm/px); the actual input TIFF's
# ImageJ block is untrustworthy for this dataset (lightsheet
# acquisition software commonly writes spacing=1.0 and 96-dpi
# defaults that are not real measurements).
canonical_x_um = float(${metadata_voxel_x})
canonical_y_um = float(${metadata_voxel_y})
canonical_z_um = float(${metadata_voxel_z})

# Read the input's data array (NOT its voxel sizes — see below).
in_vol = read_tiff('${filename}')
img = in_vol.data

# Trust read_tiff()'s voxel sizes ONLY if they're in a sane physical
# range. The input's Z is the source of truth for "what Z spacing does
# this stack actually carry" (which is the post-ISOTROPIC Z if the
# upstream chain ran, e.g. 0.374 µm). The input's XY is unused here
# because DOWNSCALE_XY rescales XY by ``scale`` regardless.
def _is_sane_voxel(v):
    return v is not None and 0.05 < v < 100.0

input_z_um = in_vol.voxel.z if _is_sane_voxel(in_vol.voxel.z) else canonical_z_um

# XY: always start from the canonical (raw-input) value and divide by
# the downscale factor. This is correct regardless of what the input
# TIFF claims, because the downscale factor is the same one applied to
# the data array above.
xy_pixel_in = canonical_x_um
y_pixel_in  = canonical_y_um

print(f"Input  shape: {img.shape}, dtype: {img.dtype}, "
      f"voxel={xy_pixel_in:.4f}x{y_pixel_in:.4f}x{input_z_um:.4f} µm "
      f"(XY canonical, Z from input), scale=${scale_factor_py}")

scale = float(${scale_factor_py})
do_iso = (${reslice_py} == 'True')

if not (0.0 < scale <= 1.0):
    raise SystemExit(f"scale_factor must satisfy 0 < scale <= 1, got {scale}")

if img.ndim != 3:
    raise SystemExit(f"Expected 3D ZYX input, got shape {img.shape}")

# XY downscale using the same cubic anti-aliased algorithm the modular
# preprocessing pipeline uses (bin/isotropic_resample.py).
out = rescale(img, (1.0, scale, scale), order=3,
              preserve_range=True, anti_aliasing=True)
x_res = xy_pixel_in / scale
y_res = y_pixel_in / scale
print(f"After XY rescale: {out.shape}  X/Y pixel size -> {x_res:.4f} x {y_res:.4f} µm")

# Optional isotropic Z reslice in the same task, matching RESLICE_ISOTROPIC.
# Since the chain reorder (commit on top of Option B, 2026-09-09) put
# DOWNSCALE_XY at the END of the full preprocessing chain
# (PLANAR -> DEPTH -> DOWNSCALE_XY), this branch is what makes the final
# stack isotropic at the new (halved) XY voxel size — i.e. the standard
# case when downscaling.enabled=true AND preprocessing.enabled=true. With
# do_iso=true the output is isotropic at x_res (the new XY µm/px).
# It also fires for the standalone-downscale path when the user has
# preprocessing.enabled=false but downscaling.enabled=true AND
# preprocessing.isotropic_reslice=true.
if do_iso:
    zoom_z = input_z_um / x_res
    if abs(zoom_z - 1.0) < 1e-3:
        print("Z already isotropic, skipping.")
    else:
        from scipy.ndimage import zoom as ndi_zoom
        new_z = int(round(out.shape[0] * zoom_z))
        print(f"Isotropic Z reslice: {out.shape[0]} -> {new_z} (zoom_z={zoom_z:.4f})")
        out = ndi_zoom(out, (zoom_z, 1.0, 1.0), order=1, prefilter=False)
    out_z_um = x_res  # isotropic → Z spacing matches the new XY
else:
    # No Z reslice this pass: keep the input's Z spacing unchanged. This
    # branch only fires for the
    # skip_preprocessing-with-user-supplied-preprocessed-files path,
    # where the external preprocessed files may already carry any Z
    # geometry and we don't want to overwrite it.
    out_z_um = input_z_um

if out.dtype != np.uint16:
    out = np.clip(out.astype(np.int32), 0, 65535).astype(np.uint16)

# No backslashes — see note in RESLICE_ISOTROPIC above
channel = '1'
_idx = '${filename}'.find('_Channel ')
if _idx >= 0:
    _rest = '${filename}'[_idx + len('_Channel '):]
    _digits = ''
    for _ch in _rest:
        if _ch.isdigit():
            _digits += _ch
        else:
            break
    if _digits:
        channel = _digits
out_name = f"t${t_formatted}_dscale_Channel {channel}.tif"

write_tiff(out_name, out, VoxelSizes(out_z_um, y_res, x_res))

# Stash the extra bookkeeping tags on top of the standard ImageJ block
# that write_tiff() already produced. We re-open the file in r+ mode and
# patch the ImageDescription text (the only ImageJ-extension tag that
# survives a BigTIFF write without breaking downstream readers). This is
# the same convention PLANAR/DEPTH/ISOTROPIC use — see bin/_tiff_io.py
# callers that wrap write_tiff() with a follow-up metadata patch.
with tifffile.TiffFile(out_name) as tf:
    page = tf.pages[0]
    desc = (page.tags.get('ImageDescription').value
            if 'ImageDescription' in page.tags else '')
    if isinstance(desc, bytes):
        desc = desc.decode('latin-1', errors='replace')
# NOTE: prior heredoc versions used literal backslash-n in string
# literals and comments; the bash heredoc pass-through on Nextflow
# 25.04.7 ate the backslash and Python saw an unescaped newline (EOL
# while scanning string literal). We now use os.linesep instead of
# any backslash escape sequences for newlines (Python strings
# concat with the function return), and this comment avoids
# backslash characters entirely. See repo memory 2026-09-09.
import os as _os
_NL = _os.linesep
_tp_int = ${timepoint}
_tp_str = str(_tp_int)
_was_roi = str(metadata.get('was_roi_cropped', False)).lower()
_do_iso_str = str(bool(do_iso)).lower()
extra_tags = (
    _NL + 'TimePoint=' + _tp_str +
    _NL + 'WasROICropped=' + _was_roi +
    _NL + 'XYDownscaled=True' +
    _NL + 'ScalingFactor=' + str(scale) +
    _NL + 'IsotropicResliced=' + _do_iso_str
)
with tifffile.TiffFile(out_name, mode='r+') as tf:
    page = tf.pages[0]
    tag = page.tags.get('ImageDescription')
    if tag is not None:
        new_desc = (desc or '').rstrip(_NL) + extra_tags + _NL
        tag.overwrite(new_desc.encode('latin-1', errors='replace'))

print('Wrote ' + out_name + '  voxel=' +
      ('%.4f' % x_res) + 'x' + ('%.4f' % y_res) + 'x' + ('%.4f' % out_z_um) + ' um '
      ' (XY factor=' + str(scale) + ', do_iso=' + _do_iso_str + ')')
PYTHON_EOF
    """
}

// ============================================================================
// PROCESS: Export the RAW (unprocessed) input, sliced isotropic and downscaled
//
// Independent of the preprocessing chain: takes the post-ROI-crop raw input,
// applies the same XY cubic rescale + optional isotropic Z resample as
// DOWNSCALE_XY, and writes a per-timepoint TIFF with ImageJ metadata that
// matches what ultrack_viewer.py expects via its --processed flag.
//
// Use this when you want to overlay tracks on the ORIGINAL (raw) signal
// rather than the preprocessed one. The preprocessed chain still runs as
// usual — this step is purely additive.
// ============================================================================

process EXPORT_RAW_ISOTROPIC {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    // Per-timepoint raw_iso TIFs are intermediate products for the
    // MERGE_HYPERSTACKS task. When merge is enabled, publish ONLY the
    // merged 4D_hyperstack_raw_iso.tif (the MERGE task publishes it).
    // When merge is skipped, fall back to publishing the per-timepoint
    // TIFs so the user still gets the raw_iso output for the viewer.
    publishDir "${params.output_dir}/01b_raw_isotropic",
        mode: 'copy',
        pattern: "t*_raw_iso_Channel*.tif",
        enabled: skip_merge
    publishDir "${params.output_dir}/logs/raw_export",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json
    val scale_factor
    val reslice_isotropic

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_raw_iso_Channel*.tif"), emit: raw_iso
    path "t${String.format('%04d', timepoint)}_raw_iso.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    def scale_factor_py = (scale_factor as Double).toString()
    // Wrap in single quotes so Nextflow's GString interpolation emits
    // 'True' / 'False' (quoted) into the Python heredoc. Without the
    // wrapping, ${reslice_py} renders as the bare token `True`, which
    // makes `do_iso = (True == 'True')` always False in Python (different
    // types). Symptom: the raw_iso export keeps the input Z dimension
    // instead of reslicing to match the preprocessed chain's isotropic Z.
    def reslice_py = reslice_isotropic ? "'True'" : "'False'"
    """
    #!/bin/bash
    set -euo pipefail
    exec > >(tee t${t_formatted}_raw_iso.log) 2>&1

    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 << 'PYTHON_EOF'
import json
import re as _re
import numpy as np
import tifffile
from skimage.transform import rescale

with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

xy_pixel_in = float(metadata['x_resolution_um'])
z_pixel_in  = float(metadata['imagej']['spacing']) if 'imagej' in metadata else 1.0
scale = float(${scale_factor_py})
do_iso = (${reslice_py} == 'True')

# ── Loud diagnostics so workflow-binding mismatches surface in the log ──
# Note: reslice_py is the rendered Python literal ('True' or 'False'), so
# we wrap it in repr() instead of single quotes for the diagnostic print.
print(f"[DIAG] raw_export scale={scale}  do_iso={do_iso}  (reslice_py={${reslice_py}}!r)")
print(f"[DIAG] metadata x_res={xy_pixel_in}  z_spacing={z_pixel_in}  "
      f"isotropic→x_res={xy_pixel_in/scale:.4f}  zoom_z={z_pixel_in/(xy_pixel_in/scale):.4f}")

if not (0.0 < scale <= 1.0):
    raise SystemExit(f"scale_factor must satisfy 0 < scale <= 1, got {scale}")

img = tifffile.imread('${filename}')
if img.ndim != 3:
    raise SystemExit(f"Expected 3D ZYX input, got shape {img.shape}")
print(f"Input  shape: {img.shape}, dtype: {img.dtype}, scale={scale}")

# IMPORTANT: do NOT apply CLAHE/normalisation here — this is RAW export.
# Only the geometric ops (XY cubic rescale + isotropic Z resample) so the
# viewer can register tracks against the original signal.
out = rescale(img, (1.0, scale, scale), order=3,
              preserve_range=True, anti_aliasing=True)
x_res = xy_pixel_in / scale
y_res = float(metadata['y_resolution_um']) / scale
print(f"After XY rescale: {out.shape}  X/Y pixel size -> {x_res:.4f} x {y_res:.4f} µm")

if do_iso:
    zoom_z = z_pixel_in / x_res
    if abs(zoom_z - 1.0) < 1e-3:
        print("Z already isotropic, skipping.")
    else:
        # Use skimage.transform.resize (same as the preprocessed chain)
        # so the Z expansion matches the preprocessed output exactly.
        # Pin the output shape so the documented Z expansion is guaranteed
        # regardless of any rounding ambiguity in the float zoom factor.
        expected_z = int(round(out.shape[0] * zoom_z))
        from skimage.transform import resize as sk_resize
        out = sk_resize(
            out,
            (expected_z, out.shape[1], out.shape[2]),
            order=1,
            mode="constant",
            anti_aliasing=False,
            preserve_range=True,
        )
        print(f"Isotropic Z reslice: expected={expected_z} (zoom_z={zoom_z:.4f}, got shape={out.shape})")
        # Hard sanity check — fail loudly if the volume is wrong so
        # the bug surfaces in the log instead of silently producing a
        # mis-aligned volume.
        assert out.shape[0] == expected_z, (
            f"Z resample failed: expected {expected_z} planes, got {out.shape[0]}"
        )

if out.dtype != np.uint16:
    out = np.clip(out.astype(np.int32), 0, 65535).astype(np.uint16)

# No backslashes — see note in RESLICE_ISOTROPIC above
channel = '1'
_idx = '${filename}'.find('_Channel ')
if _idx >= 0:
    _rest = '${filename}'[_idx + len('_Channel '):]
    _digits = ''
    for _ch in _rest:
        if _ch.isdigit():
            _digits += _ch
        else:
            break
    if _digits:
        channel = _digits
out_name = f"t${t_formatted}_raw_iso_Channel {channel}.tif"

tifffile.imwrite(
    out_name,
    out,
    imagej=True,
    resolution=(1.0/x_res, 1.0/y_res),
    # IMPORTANT: compression='zlib' so the Compression TIFF tag (259)
    # is 8 for every per-timepoint TIFF. Without this, tifffile defaults
    # to no compression (Compression=1), and Fiji's Merge Channels /
    # Concatenate dialogs refuse to combine files with different
    # compression codes (misleading 'different bit depth' error).
    # Verified on 2026-09-09: dscale + segmented had Compression=8
    # (zlib via bin/_tiff_io.write_tiff default), but raw_iso had
    # Compression=1 because the heredoc didn't pass compression=.
    compression='zlib',
    metadata={
        'spacing': z_pixel_in if not do_iso else x_res,
        'unit': 'um',
        'axes': 'ZYX',
        # IMPORTANT: declare channels=1 so Fiji's Merge Channels and
        # Concatenate dialogs treat this as a single-channel Z-stack.
        # Without this, Fiji refuses to combine with other stacks.
        'channels': 1,
        # Bypass the XResolution/YResolution ResolutionUnit ambiguity
        # across tifffile versions (2024.6.18 vs 2026.x write the same
        # `resolution=(...)` kwarg differently). bin/_tiff_io.read_tiff()
        # reads these directly from tf.imagej_metadata on the reader side.
        'x_resolution_um': x_res,
        'y_resolution_um': y_res,
    },
)
# Stash the per-timepoint bookkeeping tags on top of the ImageJ block
# that tifffile.imwrite just produced, using the SAME r+ overwrite
# pattern DOWNSCALE_XY and CELLPOSE_SEGMENT use.
#
# IMPORTANT: tifffile.imwrite's `metadata={...}` kwarg lowercases ALL
# keys when building the ImageDescription text (verified: passing
# {'TimePoint': 287} produces 'timepoint=287' in the file). That made
# EXPORT_RAW_ISOTROPIC outputs inconsistent with DOWNSCALE_XY /
# CELLPOSE_SEGMENT outputs (which use TitleCase via r+ overwrite),
# breaking Fiji's Concatenate / Merge Channels dialogs with a
# misleading 'bit depth mismatch' error.
#
# Use chr(10) for newlines (no backslash-n literals in heredocs;
# see repo memory 2026-09-09).
import os as _os_raw
_NL_RAW = _os_raw.linesep
_tp_int_raw = ${timepoint}
_tp_str_raw = str(_tp_int_raw)
_was_roi_raw = str(metadata.get('was_roi_cropped', False)).lower()
_do_iso_str_raw = str(bool(do_iso)).lower()
_scale_str_raw = str(scale)
extra_tags_raw = (
    _NL_RAW + 'TimePoint=' + _tp_str_raw +
    _NL_RAW + 'WasROICropped=' + _was_roi_raw +
    _NL_RAW + 'RawExported=True' +
    _NL_RAW + 'ScalingFactor=' + _scale_str_raw +
    _NL_RAW + 'IsotropicResliced=' + _do_iso_str_raw +
    _NL_RAW + 'PipelineStage=raw_export'
)
with tifffile.TiffFile(out_name, mode='r+') as tf:
    page = tf.pages[0]
    tag = page.tags.get('ImageDescription')
    if tag is not None:
        current_desc = (tag.value.decode('latin-1', errors='replace')
                        if isinstance(tag.value, bytes) else (tag.value or ''))
        new_desc = current_desc.rstrip(_NL_RAW) + extra_tags_raw + _NL_RAW
        tag.overwrite(new_desc.encode('latin-1', errors='replace'))

print(f"Wrote {out_name}")
PYTHON_EOF
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
    # NOTE: use chr(10) instead of backslash-n for newlines in f-strings —
    # the bash heredoc pass-through on Nextflow 25.04.7 was eating backslashes
    # (see repo memory 2026-09-09).
    print("Saving metadata to JSON...", file=sys.stderr)
    with open('shared_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(chr(10) + "Metadata Configuration:")
    print(f"  Source: {metadata['voxel_size_source']}")
    print(f"  ROI cropped: {was_cropped}")
    print(f"  Image shape: {metadata['shape']['dimensions']} (ZYX)")
    print(f"  Voxel size: {x_resolution_um:.4f} x {y_resolution_um:.4f} x {z_spacing:.4f} µm")
    print(chr(10) + "Full metadata:")
    print(json.dumps(metadata, indent=2))

    print(chr(10) + "SUCCESS: Metadata extraction/configuration completed", file=sys.stderr)

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
// PROCESS: Planar (XY) shading correction — modular, ported from AIAF-32
// ============================================================================
//
// Each step in the new modular preprocessing pipeline is its own Nextflow
// process: this lets us scale resources per step (e.g. depth correction is
// O(Z) memory-light while planar correction is O(Z*Y*X) memory-heavy), and
// lets us re-run a single step with `-resume` when only one of them changes.
//
// Chain: PLANAR_CORRECTION -> DEPTH_CORRECTION -> ISOTROPIC -> segmentation.

process PLANAR_CORRECTION {
    tag "t${String.format('%04d', timepoint)}"

    publishDir "${params.output_dir}/01_preprocessed",
        mode: 'copy',
        pattern: "*_planar.tif",
        enabled: skip_merge

    publishDir "${params.output_dir}/logs/preprocessing",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json
    path bin_dir            // entire bin/ so _tiff_io.py is importable

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_planar.tif"), emit: corrected
    path "t${String.format('%04d', timepoint)}_planar.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    """
    #!/bin/bash
    set -euo pipefail

    echo "============================================"
    echo "Planar (XY) shading correction — timepoint ${timepoint}"
    echo "Input : ${filename}"
    echo "Sigma : ${planar_sigma_xy} px"
    echo "============================================"

    # The python_packages_spim container ships with micromamba and a
    # microscopy_env that contains python3 + numpy + scipy + skimage +
    # tifffile. Bare `python3` is NOT on PATH until we activate the env,
    # and MAMBA_ROOT_PREFIX is required because the base image doesn't
    # export it.
    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 bin/planar_intensity_correction.py \\
        --input     "${filename}" \\
        --output    "t${t_formatted}_planar.tif" \\
        --sigma_xy  ${planar_sigma_xy} \\
        --voxel_x_um ${metadata_voxel_x} \\
        --voxel_y_um ${metadata_voxel_y} \\
        --voxel_z_um ${metadata_voxel_z} \\
        2>&1 | tee "t${t_formatted}_planar.log"

    echo "Planar correction complete: t${t_formatted}_planar.tif"
    """
}

// ============================================================================
// PROCESS: Depth (Z) intensity correction — modular, ported from AIAF-32
// ============================================================================

process DEPTH_CORRECTION {
    tag "t${String.format('%04d', timepoint)}"

    publishDir "${params.output_dir}/01_preprocessed",
        mode: 'copy',
        pattern: "*_depth.tif",
        enabled: skip_merge

    publishDir "${params.output_dir}/logs/preprocessing",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json
    path bin_dir            // entire bin/ so _tiff_io.py is importable

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_depth.tif"), emit: corrected
    path "t${String.format('%04d', timepoint)}_depth.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    """
    #!/bin/bash
    set -euo pipefail

    echo "============================================"
    echo "Depth (Z) intensity correction — timepoint ${timepoint}"
    echo "Input  : ${filename}"
    echo "Mode   : ${depth_mode}"
    echo "Window : ${depth_smooth}"
    echo "Gain   : [${depth_gain_min}, ${depth_gain_max}]"
    echo "============================================"

    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 bin/depth_intensity_correction.py \\
        --input         "${filename}" \\
        --output        "t${t_formatted}_depth.tif" \\
        --mode          "${depth_mode}" \\
        --smooth_window ${depth_smooth} \\
        --gain_min      ${depth_gain_min} \\
        --gain_max      ${depth_gain_max} \\
        --voxel_x_um    ${metadata_voxel_x} \\
        --voxel_y_um    ${metadata_voxel_y} \\
        --voxel_z_um    ${metadata_voxel_z} \\
        2>&1 | tee "t${t_formatted}_depth.log"

    echo "Depth correction complete: t${t_formatted}_depth.tif"
    """
}

// ============================================================================
// PROCESS: Isotropic Z resampling — modular, ported from AIAF-32
// ============================================================================

process ISOTROPIC {
    tag "t${String.format('%04d', timepoint)}"

    publishDir "${params.output_dir}/01_preprocessed",
        mode: 'copy',
        pattern: "*_processed.tif",
        enabled: skip_merge

    publishDir "${params.output_dir}/logs/preprocessing",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json
    path bin_dir            // entire bin/ so _tiff_io.py is importable

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_processed.tif"), emit: processed
    path "t${String.format('%04d', timepoint)}_iso.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    """
    #!/bin/bash
    set -euo pipefail

    echo "============================================"
    echo "Isotropic Z resampling — timepoint ${timepoint}"
    echo "Input  : ${filename}"
    echo "Target : ${iso_target_um} µm (order=${iso_order})"
    echo "============================================"

    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    python3 bin/isotropic_resample.py \\
        --input             "${filename}" \\
        --output            "t${t_formatted}_processed.tif" \\
        --target_um         ${iso_target_um} \\
        --order             ${iso_order} \\
        --out_voxel_x_um    ${metadata_voxel_x} \\
        --out_voxel_y_um    ${metadata_voxel_y} \\
        --input_voxel_z_um  ${metadata_voxel_z} \\
        2>&1 | tee "t${t_formatted}_iso.log"

    echo "Isotropic resample complete: t${t_formatted}_processed.tif"
    """
}

// ============================================================================
// PROCESS: Cellpose Segmentation
// ============================================================================

process CELLPOSE_SEGMENT {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    // Only publish per-timepoint segmented TIFFs when the 4D hyperstack
    // merge is disabled. Otherwise the same data ends up in publishDir twice
    // (once per timepoint, once inside 4D_hyperstack_segmented.tif).
    publishDir "${params.output_dir}/02_segmented",
        mode: 'copy',
        pattern: "*_segmented.tif",
        enabled: skip_merge

    publishDir "${params.output_dir}/logs/segmentation",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(processed_file)
    path metadata_json
    val segment_config
    val image_scaling
    path bin_dir            // entire bin/ so _tiff_io.py is importable

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
# (null = isotropic, which is what the preprocessed stack is AFTER the
# bin/isotropic_resample.py ISOTROPIC process AND after we now write proper
# ImageJ voxel-size metadata into the TIFF. Passing --anisotropy 1.0
# explicitly tells Cellpose "voxels are isotropic, do NOT resample Z",
# which avoids a 3x Z upsample that previously OOMed the segmentation
# task on real-world stacks.)
if config.get('anisotropy') is not None:
    cmd.extend(['--anisotropy', str(config['anisotropy'])])
elif config.get('do_3d', False):
    # Preprocessed TIFFs are nominally isotropic; force Cellpose to
    # honour that rather than guessing from shape ratio.
    cmd.extend(['--anisotropy', '1.0'])

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
# NOTE: use chr(10) for newlines instead of the usual backslash-n escape
# sequence; the bash heredoc pass-through on Nextflow 25.04.7 was eating
# backslashes (see repo memory 2026-09-09).
with open('t${t_formatted}_segment.log', 'w') as f:
    f.write("STDOUT:" + chr(10))
    f.write(result.stdout)
    f.write(chr(10) + chr(10) + "STDERR:" + chr(10))
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
import sys as _sys

# Save log
# NOTE: use chr(10) for newlines instead of backslash-n escape sequences;
# the bash heredoc pass-through on Nextflow 25.04.7 was eating backslashes.
with open('t${t_formatted}_segment.log', 'a') as _flog:
    _flog.write(chr(10) + chr(10) + "=== POST-CELLPOSE METADATA WRITE ===" + chr(10))

# Make bin/ importable so we can use _tiff_io.read_tiff / write_tiff
# for correct metadata round-tripping — same pattern as the DOWNSCALE_XY
# task. Without this, the in-script tifffile.imread for the input file
# cannot recover voxel sizes from the (post-ISOTROPIC, post-DOWNSCALE_XY)
# TIFF metadata, and we'd silently fall back to 1.0 µm defaults — which
# is the root cause of the 2026-09-09 "Z=5.0 µm on segmented.tif" bug.
_sys.path.insert(0, 'bin')
from _tiff_io import VoxelSizes, read_tiff, write_tiff

with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Canonical voxel sizes from the user's config.json (passed in as
# GString interpolations of the script-scope doubles metadata_voxel_x/y/z
# defined near the top of spim_pipeline.nf). The CELL's input TIFF's
# ImageJ block is untrustworthy (lightsheet acquisition software
# commonly writes spacing=1.0 + 96-dpi defaults that are not real
# measurements), so we use the canonical config values instead.
# NOTE: do NOT write the literal "metadata_voxel_x/y/z" shorthand
# inside this triple-quoted Groovy GString comment (with or without the
# dollar-brace) — Groovy parses it as a division expression and fails
# with "No such variable: y". Just spell out the variable names.
#
# The mask inherits XY from the raw config (0.347 µm/px); DOWNSCALE_XY
# (if enabled) would have already halved XY to 0.694 µm/px in the input
# file. To pick up the actual input XY rather than the raw config XY,
# we read the input's XY via read_tiff and only fall back to the
# canonical XY when read_tiff returned the bogus 1.0 / 0.0104 defaults.
in_vol = read_tiff('${filename}')
canonical_x_um = float(${metadata_voxel_x})
canonical_y_um = float(${metadata_voxel_y})
canonical_z_um = float(${metadata_voxel_z})

# Trust read_tiff's XY ONLY if it's in a sane physical range (> 0.05
# µm/px and < 100 µm/px — wider than any real microscope voxel but
# narrower than the bogus 104 µm/px default). If it's outside that
# band, fall back to the canonical config values.
def _is_sane_voxel(v):
    return v is not None and 0.05 < v < 100.0

input_x_um = in_vol.voxel.x if _is_sane_voxel(in_vol.voxel.x) else canonical_x_um
input_y_um = in_vol.voxel.y if _is_sane_voxel(in_vol.voxel.y) else canonical_y_um
# Z: prefer the canonical config value (the user's true Z spacing) over
# whatever the input TIFF claims — the input's Z could be the post-
# ISOTROPIC 0.374 µm or the bogus 1.0 µm default or the raw 2.0 µm;
# the canonical 2.0 µm (or whatever's in config.json) is what we want
# for the input voxel sizes the user intended. After ISOTROPIC the
# effective Z spacing IS 0.374 µm regardless of what the user wrote,
# so the CELLPOSE_SEGMENT task actually wants the canonical_z_um here.
# But that's only true if the input is the post-ISOTROPIC stack. For
# now we trust the canonical value and override with read_tiff if sane.
input_z_um = in_vol.voxel.z if _is_sane_voxel(in_vol.voxel.z) else canonical_z_um

print(f"Cellpose input voxel size: {input_x_um:.4f} x {input_y_um:.4f} x {input_z_um:.4f} µm  "
      f"(shape={in_vol.data.shape})")

# Load mask and detect its axis order.
# Cellpose with do_3d=True returns the mask in the same ZYX order as the
# input — but some 2D / stitch paths emit (Y, X, Z) instead. We assume
# the axis with the SMALLEST dimension is Z (the typical lightsheet
# anisotropy) and the LARGEST two are Y and X. This is robust to both
# axis orderings without needing to read a TIFF axis tag from Cellpose's
# output (which Cellpose doesn't always write).
mask_full = tifffile.imread("t${t_formatted}_segmented.tif")
if mask_full.ndim == 2:
    # 2D mask — happens when do_3d=False and input was treated as a single
    # 2D slice. Expand to (1, Y, X) so the downstream tifffile.imwrite
    # axis declaration matches.
    mask_full = mask_full[np.newaxis, ...]
# Detect axis order by dimension: the smallest axis is Z (typical lightsheet
# anisotropy), the two larger ones are Y and X. When the two larger axes
# happen to have the same dimension (common: square XY slices), we cannot
# distinguish them by shape alone — but for a per-pixel label mask, the
# axes are interchangeable (X and Y have the same µm/px and no semantic
# distinction). We assume the existing order is correct unless the smallest
# axis is NOT at position 0, which is the only case where a transpose is
# actually needed. This avoids the bug where sorted_shape + list.index
# returned (0,1,1) for a (231, 1152, 1152) mask because list.index returns
# the first match (y_axis and x_axis both resolved to 1).
if mask_full.shape[0] < mask_full.shape[1] and mask_full.shape[0] < mask_full.shape[2]:
    z_axis, y_axis, x_axis = 0, 1, 2
elif mask_full.shape[1] < mask_full.shape[0] and mask_full.shape[1] < mask_full.shape[2]:
    z_axis, y_axis, x_axis = 1, 2, 0
    print(f"Cellpose output is in axis order ({z_axis},{y_axis},{x_axis}); "
          f"reordering to (0,1,2) ZYX")
    mask_full = np.transpose(mask_full, (z_axis, y_axis, x_axis))
elif mask_full.shape[2] < mask_full.shape[0] and mask_full.shape[2] < mask_full.shape[1]:
    z_axis, y_axis, x_axis = 2, 0, 1
    print(f"Cellpose output is in axis order ({z_axis},{y_axis},{x_axis}); "
          f"reordering to (0,1,2) ZYX")
    mask_full = np.transpose(mask_full, (z_axis, y_axis, x_axis))
# else: all axes equal, or no axis is strictly the smallest — assume (0,1,2)
# and let the downstream tifffile.imwrite axis declaration do its job.
mask = mask_full
print(f"Mask shape (ZYX): {mask.shape}")

# Voxel sizes for the mask follow the input (the mask is a per-pixel
# labelling, it cannot change geometry). Cellpose with --do_3D + a
# non-1.0 anisotropy value can in principle emit a mask at a different
# Z resolution than the input, but the standard config in this pipeline
# passes --anisotropy=1.0 (the preprocessed stack is isotropic by then),
# which guarantees the mask Z matches the input Z. We mirror the input
# voxel sizes exactly — same XY µm/px, same Z µm.
x_res = input_x_um
y_res = input_y_um
z_spacing = input_z_um

print(f"Segmentation mask voxel size: {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} µm")

# Re-save with metadata via _tiff_io.write_tiff so the ImageJ block is
# written consistently with every other writer in the pipeline (bigtiff,
# unit='um', axes='ZYX'). We then patch the extra bookkeeping tags
# (TimePoint, LabelImage, WasROICropped) onto the ImageDescription text
# in a follow-up r+ pass — same convention DOWNSCALE_XY uses.
write_tiff(
    "t${t_formatted}_segmented.tif",
    mask.astype(np.uint16),
    VoxelSizes(z_spacing, y_res, x_res),
)
with tifffile.TiffFile("t${t_formatted}_segmented.tif") as tf:
    page = tf.pages[0]
    desc = (page.tags.get('ImageDescription').value
            if 'ImageDescription' in page.tags else '')
    if isinstance(desc, bytes):
        desc = desc.decode('latin-1', errors='replace')
# NOTE: prior heredoc versions used literal backslash-n in string
# literals and comments; the bash heredoc pass-through on Nextflow
# 25.04.7 ate the backslash and Python saw an unescaped newline
# inside the string literal (EOL while scanning string literal).
# We now use os.linesep instead of any backslash escape sequences
# for newlines, and this comment avoids backslash characters
# entirely. See repo memory 2026-09-09.
import os as _os
_NL = _os.linesep
_tp_int = ${timepoint}
_tp_str = str(_tp_int)
_was_roi = str(metadata.get('was_roi_cropped', False)).lower()
extra_tags = (
    _NL + 'TimePoint=' + _tp_str +
    _NL + 'LabelImage=True' +
    _NL + 'WasROICropped=' + _was_roi
)
with tifffile.TiffFile("t${t_formatted}_segmented.tif", mode='r+') as tf:
    page = tf.pages[0]
    tag = page.tags.get('ImageDescription')
    if tag is not None:
        new_desc = (desc or '').rstrip(_NL) + extra_tags + _NL
        tag.overwrite(new_desc.encode('latin-1', errors='replace'))
print('Metadata preserved in segmentation mask for timepoint ' + _tp_str)
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
// PROCESS: Merge All Timepoints into 4D Hyperstack(s)
//
// Single process that merges ALL data types in one task. Nextflow DSL2
// forbids invoking the same process twice in the same workflow context, so
// instead of having separate process calls for processed / segmented /
// raw_iso, we collect all three file lists up front and merge them in a
// single Python task. Each data type is optional — empty lists are skipped.
//
// Each output is emitted under its own channel name so the rest of the
// workflow (tracking, viewer exports) can subscribe to the right one.
// ============================================================================

process MERGE_HYPERSTACKS {
    tag "Merging hyperstacks"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    // Fan-out: publishDir runs once per emitted output path. Same as
    // before; each (data_type, output) tuple routes to the right dir.
    publishDir "${params.output_dir}/01_preprocessed",      mode: 'copy', pattern: "4D_hyperstack_processed*"
    publishDir "${params.output_dir}/01b_raw_isotropic",    mode: 'copy', pattern: "4D_hyperstack_raw_iso*"
    publishDir "${params.output_dir}/02_segmented",         mode: 'copy', pattern: "4D_hyperstack_segmented*"

    input:
    path metadata_json
    path merge_script
    path processed_files
    path segmented_files
    path raw_iso_files

    output:
    path "4D_hyperstack_processed.tif",     emit: processed_tif, optional: true
    path "4D_hyperstack_processed_metadata.json", emit: processed_meta, optional: true
    path "4D_hyperstack_processed.h5",      emit: processed_h5,  optional: true
    path "4D_hyperstack_processed.xml",     emit: processed_xml, optional: true
    path "4D_hyperstack_segmented.tif",     emit: segmented_tif, optional: true
    path "4D_hyperstack_segmented_metadata.json", emit: segmented_meta, optional: true
    path "4D_hyperstack_segmented.h5",      emit: segmented_h5,  optional: true
    path "4D_hyperstack_segmented.xml",     emit: segmented_xml, optional: true
    path "4D_hyperstack_raw_iso.tif",       emit: raw_iso_tif,   optional: true
    path "4D_hyperstack_raw_iso_metadata.json", emit: raw_iso_meta, optional: true
    path "4D_hyperstack_raw_iso.h5",        emit: raw_iso_h5,    optional: true
    path "4D_hyperstack_raw_iso.xml",       emit: raw_iso_xml,   optional: true

    container params.container

    script:
    // Create properly escaped JSON string for Python heredoc
    def config_json_str = groovy.json.JsonOutput.toJson(config)
    def merge_script_name = merge_script.name
    // MERGE_STAGE_LAYOUT=v7 : merge script now RECURSIVELY searches the workdir
    // and symlinks any matching per-timepoint files into the workdir root.
    // This handles BOTH the original loose-root layout AND the named-subdir
    // layout that Nextflow 23.04+ uses for list-of-file `path` inputs with
    // many files (e.g. 476 timepoints) — see the 2026-09-10 bug where the
    // merge silently "skipped" all three data types and tracking never ran.
    // Bumping this version invalidates the Nextflow cache so -resume re-runs
    // merge with the corrected logic.
    def merge_stage_layout_version = "MERGE_STAGE_LAYOUT=v7-recursive-symlink"
    """
    #!/usr/bin/env bash
    set -euo pipefail

    export MAMBA_ROOT_PREFIX=/opt/conda
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "=== MERGE_HYPERSTACKS ==="
    echo "${merge_stage_layout_version}"
    python3 --version
    echo ""

    if [ ! -f "${merge_script_name}" ]; then
        echo "ERROR: Merge script not found: ${merge_script_name}"
        ls -lh
        exit 1
    fi

    # Ensure required packages
    python3 -c "import tifffile, numpy" 2>/dev/null || {
        echo "Installing required packages..."
        micromamba install -y -n microscopy_env tifffile numpy
    }

    # ----------------------------------------------------------------------
    # Input staging & merge strategy (v6: matches the pre-Aug-18 design that
    # worked reliably).
    # ----------------------------------------------------------------------
    # Nextflow stages every per-timepoint TIFF loose at the work-dir root.
    # merge_hyperstack.py uses Path('.').glob() to find files of the
    # current data type, so we just run it from CWD with the data type as
    # an argument. The Python glob is per-data-type, so processed /
    # segmented / raw_iso files coexist in the work-dir without conflict.
    #
    # We deliberately do NOT copy or symlink files into a staging subdir:
    # the older MERGE_TO_HYPERSTACK did exactly this and worked, so we
    # restore that. The previous staging layer added unnecessary fragility.
    # ----------------------------------------------------------------------

    echo "Merge workdir: \$(pwd)"
    echo "Files in workdir root:"
    ls -1 *.tif 2>/dev/null | head -5 || echo "  (no .tif files)"
    echo "  ... (total: \$(ls -1 *.tif 2>/dev/null | wc -l | tr -d ' ') .tif files)"
    echo ""

    # Temp config (merge_hyperstack.py reads it from CWD)
    python3 - << 'PYTHON_CONFIG'
import json
config_str = '''${config_json_str}'''
config_data = json.loads(config_str)
with open('config_temp.json', 'w') as f:
    json.dump(config_data, f, indent=2)
print("✓ Config file created")
PYTHON_CONFIG

    run_merge() {
        local dt="\$1"

        # Determine the per-timepoint glob for this data type.
        local pattern
        if [ "\$dt" = "raw_iso" ]; then
            pattern="t*_raw_iso_*.tif"
        else
            pattern="t*_\${dt}.tif"
        fi

        # ------------------------------------------------------------------
        # Collect per-timepoint files. Nextflow's staging strategy for a
        # `path` input that receives a LIST of files depends on the version:
        #   - Older Nextflow: stages files LOOSE at the workdir root.
        #   - Nextflow 23.04+ with many files (>= ~100): stages files into a
        #     NAMED SUBDIR whose name is derived from the input qualifier
        #     (e.g. processed/, segmented/, raw_iso/). merge_hyperstack.py
        #     only globs the workdir root, so the merge silently skips.
        #
        # Workaround: count files at the root first; if 0, recursively
        # search the entire workdir for matching per-timepoint files and
        # symlink them into the workdir root. merge_hyperstack.py then sees
        # them via its existing `Path('.').glob(...)` and the merge runs.
        #
        # `set +e` is toggled around the glob/find calls because both `ls`
        # (on a non-matching glob) and `find` (on no match) return non-zero,
        # and `set -euo pipefail` would otherwise kill the script before we
        # get a chance to fall back.
        # ------------------------------------------------------------------
        set +e
        local n_root
        n_root=\$(ls \$pattern 2>/dev/null | wc -l | tr -d ' ')
        set -e
        n_root="\${n_root:-0}"

        if [ "\$n_root" -eq 0 ]; then
            # Fallback: search recursively. `find` returns 1 on no match,
            # which would trip `set -e`; pipe the result through cat to
            # normalise the exit code.
            set +e
            local found
            found=\$(find . -name "\$pattern" -type f 2>/dev/null | sort)
            set -e

            if [ -z "\$found" ]; then
                echo "⏭  Skipping \${dt} — no files matching '\${pattern}' in work-dir"
                return 0
            fi

            local n_found
            n_found=\$(echo "\$found" | wc -l | tr -d ' ')
            echo "  [merge:\${dt}] no files at workdir root; symlinking \${n_found} file(s) from subdirs"

            while IFS= read -r src; do
                [ -z "\$src" ] && continue
                local bn
                bn=\$(basename "\$src")
                if [ ! -e "\$bn" ]; then
                    ln -sf "\$src" "\$bn"
                fi
            done <<< "\$found"

            n_root=\$n_found
        fi

        echo ""
        echo "--- Merging \${dt} (\${n_root} files) ---"

        export NXF_TASK_CPUS=\${NXF_TASK_CPUS:-1}
        python3 "${merge_script_name}" "${metadata_json}" config_temp.json "\$dt" \
            || { echo "ERROR: merge failed for \${dt}"; return 1; }

        echo "✓ \${dt} merged"
    }

    # Process each data type in turn. merge_hyperstack.py's per-data-type
    # glob ensures it only picks up files of the requested type, so the
    # three runs are independent.
    run_merge processed
    run_merge segmented
    run_merge raw_iso

    echo ""
    echo "=== MERGE_HYPERSTACKS completed ==="
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
    // Optional staged files (used as a fallback when per-timepoint TIFFs are
    // NOT published to results_dir, e.g. when output.skip_merge=false).
    // When the lists are empty, benchmark reads from results_dir directly.
    path(processed_staged, stageAs: 'staged/01_preprocessed/*')
    path(segmented_staged, stageAs: 'staged/02_segmented/*')
    path benchmark_script

    output:
    path "benchmark_results.csv", emit: csv, optional: true
    path "benchmark_results.json", emit: json, optional: true
    path "benchmark.log", emit: log, optional: true

    script:
    def n_staged_proc = (processed_staged instanceof List) ? processed_staged.size() : (processed_staged ? 1 : 0)
    def n_staged_seg  = (segmented_staged instanceof List) ? segmented_staged.size() : (segmented_staged ? 1 : 0)
    def use_staged = (n_staged_proc + n_staged_seg) > 0
    """
    #!/bin/bash
    set -uo pipefail

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "============================================"
    echo "Running Pipeline Benchmark"
    echo "============================================"
    echo "Python: \$(which python3)"
    echo ""

    # Decide source: staged work-dir files (when per-timepoint files were
    # NOT published) vs the publishDir (when they were).
    USE_STAGED=${use_staged ? 1 : 0}

    if [ "\$USE_STAGED" -eq 1 ]; then
        BENCH_DIR="\$PWD/staged"
        echo "Using STAGED inputs from work dir: \$BENCH_DIR"
        echo "  (per-timepoint TIFFs are not in publishDir because merge is enabled)"
        mkdir -p staged/01_preprocessed staged/02_segmented
    else
        BENCH_DIR="${results_dir}"
        echo "Using PUBLISHED inputs from: \$BENCH_DIR"
    fi
    echo ""

    echo "=== Preprocessed files ==="
    ls -lh "\$BENCH_DIR"/01_preprocessed/*_processed.tif 2>/dev/null | head -10 || echo "  (none found)"
    PREPROC_COUNT=\$(ls "\$BENCH_DIR"/01_preprocessed/*_processed.tif 2>/dev/null | wc -l | tr -d ' ')
    echo "Total preprocessed: \$PREPROC_COUNT"
    echo ""

    echo "=== Segmented files ==="
    ls -lh "\$BENCH_DIR"/02_segmented/*_segmented.tif 2>/dev/null | head -10 || echo "  (none found)"
    SEG_COUNT=\$(ls "\$BENCH_DIR"/02_segmented/*_segmented.tif 2>/dev/null | wc -l | tr -d ' ')
    echo "Total segmented: \$SEG_COUNT"
    echo ""

    if [ "\$PREPROC_COUNT" -eq 0 ] && [ "\$SEG_COUNT" -eq 0 ]; then
        echo "ERROR: No pipeline output files found in \$BENCH_DIR"
        echo "Check that preprocessing and segmentation completed successfully."
        exit 1
    fi

    # Sample up to 3 timepoints for fast benchmarking
    # Find timepoint numbers from filenames and pick first, middle, last
    TIMEPOINTS=\$(ls "\$BENCH_DIR"/01_preprocessed/*_processed.tif 2>/dev/null \\
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
        --results_dir "\$BENCH_DIR/" \\
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

    # Expose Nextflow's CPU/memory allocation to the Python autosizer so it
    # caps worker count to the SLURM cgroup, not the full node (otherwise
    # psutil + os.cpu_count() see the host and overcommit → OOM / exit 137).
    export NXF_TASK_CPUS=${task.cpus}
    export SLURM_CPUS_PER_TASK=${task.cpus}
    export NXF_TASK_MEMORY_BYTES=${task.memory.toBytes()}

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
    path "metadata.toml", emit: metadata

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    # Override database path in config to force writing into workdir
    # ultrack sqlite uses working_dir + database_file_name (NOT address)
    # NOTE: no backslashes in this heredoc body — Nextflow 25.04.7's bash
    # heredoc pass-through was eating backslashes (see repo memory
    # 2026-09-09). We replace via line iteration instead of re.sub.
    python3 << 'PYEOF'
import pathlib
cfg_path = pathlib.Path('${ultrack_config_toml}')
cfg = cfg_path.read_text()
cfg_lines = cfg.split(chr(10))

# Remove any existing data_config fields we need to override
remove_keys = ('database', 'address', 'working_dir', 'database_file_name')
new_lines = []
for _line in cfg_lines:
    _stripped = _line.lstrip()
    _drop = False
    for _key in remove_keys:
        if _stripped.startswith(_key + ' ') or _stripped.startswith(_key + '=') or _stripped == _key:
            _drop = True
            break
    if not _drop:
        new_lines.append(_line)
cfg = chr(10).join(new_lines)

# Insert correct fields under [data_config]
NL = chr(10)
new_fields = f'database = "sqlite"{NL}working_dir = "."{NL}database_file_name = "data.db"'
if '[data_config]' in cfg:
    cfg = cfg.replace('[data_config]', '[data_config]' + NL + new_fields)
else:
    cfg += NL + '[data_config]' + NL + new_fields + NL

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
    ls -lh metadata.toml
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
    path metadata_toml

    output:
    path "local_ultrack_config.toml", emit: config_toml
    path "data.db", emit: database
    path "metadata.toml", emit: metadata

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    # Re-patch database path to point to the local workdir copy
    # NOTE: no backslashes — see ULTRACK_CONFIG_EDIT block above
    python3 << 'PYEOF'
import pathlib
cfg_path = pathlib.Path('${ultrack_config_toml}')
cfg = cfg_path.read_text()
cfg_lines = cfg.split(chr(10))
remove_keys = ('database', 'address', 'working_dir', 'database_file_name')
new_lines = []
for _line in cfg_lines:
    _stripped = _line.lstrip()
    _drop = False
    for _key in remove_keys:
        if _stripped.startswith(_key + ' ') or _stripped.startswith(_key + '=') or _stripped == _key:
            _drop = True
            break
    if not _drop:
        new_lines.append(_line)
cfg = chr(10).join(new_lines)
NL = chr(10)
new_fields = f'database = "sqlite"{NL}working_dir = "."{NL}database_file_name = "data.db"'
if '[data_config]' in cfg:
    cfg = cfg.replace('[data_config]', '[data_config]' + NL + new_fields)
else:
    cfg += NL + '[data_config]' + NL + new_fields + NL
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
    path metadata_toml

    output:
    path "local_ultrack_config.toml", emit: config_toml
    path "data.db", emit: database
    path "metadata.toml", emit: metadata

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    # Re-patch database path to point to the local workdir copy
    # NOTE: no backslashes — see ULTRACK_CONFIG_EDIT block above
    python3 << 'PYEOF'
import pathlib
cfg_path = pathlib.Path('${ultrack_config_toml}')
cfg = cfg_path.read_text()
cfg_lines = cfg.split(chr(10))
remove_keys = ('database', 'address', 'working_dir', 'database_file_name')
new_lines = []
for _line in cfg_lines:
    _stripped = _line.lstrip()
    _drop = False
    for _key in remove_keys:
        if _stripped.startswith(_key + ' ') or _stripped.startswith(_key + '=') or _stripped == _key:
            _drop = True
            break
    if not _drop:
        new_lines.append(_line)
cfg = chr(10).join(new_lines)
NL = chr(10)
new_fields = f'database = "sqlite"{NL}working_dir = "."{NL}database_file_name = "data.db"'
if '[data_config]' in cfg:
    cfg = cfg.replace('[data_config]', '[data_config]' + NL + new_fields)
else:
    cfg += NL + '[data_config]' + NL + new_fields + NL
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
    path metadata_toml

    output:
    // Publish the entire results/ directory as-is. Using "results/**" drops
    // dotfiles (e.g. zarr's .zarray, .zgroup, .zattrs) on some Nextflow
    // versions / filesystems, which silently breaks downstream zarr readers.
    path "results", emit: results
    path "ultrack_export.log", emit: log

    container params.ultrack_container

    script:
    """
    #!/usr/bin/env bash
    set -euo pipefail

    exec > >(tee ultrack_export.log) 2>&1

    # Re-patch database path to point to the local workdir copy
    # NOTE: no backslashes — see ULTRACK_CONFIG_EDIT block above
    python3 << 'PYEOF'
import pathlib
cfg_path = pathlib.Path('${ultrack_config_toml}')
cfg = cfg_path.read_text()
cfg_lines = cfg.split(chr(10))
remove_keys = ('database', 'address', 'working_dir', 'database_file_name')
new_lines = []
for _line in cfg_lines:
    _stripped = _line.lstrip()
    _drop = False
    for _key in remove_keys:
        if _stripped.startswith(_key + ' ') or _stripped.startswith(_key + '=') or _stripped == _key:
            _drop = True
            break
    if not _drop:
        new_lines.append(_line)
cfg = chr(10).join(new_lines)
NL = chr(10)
new_fields = f'database = "sqlite"{NL}working_dir = "."{NL}database_file_name = "data.db"'
if '[data_config]' in cfg:
    cfg = cfg.replace('[data_config]', '[data_config]' + NL + new_fields)
else:
    cfg += NL + '[data_config]' + NL + new_fields + NL
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
    // When channel is 0 (auto-detect) use a wildcard so existing per-timepoint
    // TIFFs are found regardless of which channel number they carry.
    def glob_pattern = params.channel > 0
        ? "t*_Channel ${params.channel}.tif"
        : "t*_Channel *.tif"
    def matched_files = []

    def globMatcher = FileSystems.getDefault().getPathMatcher("glob:${glob_pattern}")
    Files.list(input_dir_path).each { p ->
        if (globMatcher.matches(p.getFileName())) {
            matched_files.add(p.toFile())
        }
    }

    // ---- Detect hyperstack / CZI inputs.  We use the SPLIT_INPUT_FILE
    //      process (CZI or 4D/5D ImageJ/OME hyperstack TIFF) only when the
    //      directory does NOT already contain per-timepoint TIFFs matching
    //      the expected pattern.  An explicit `input.file` in config.json
    //      always takes precedence. ----
    def hyperstack_input = null  // java.io.File for the .czi / hyperstack .tif

    if (params.input_file) {
        // Resolve relative to input_dir if not absolute
        def f = new File(params.input_file)
        if (!f.isAbsolute()) {
            f = new File(params.input_dir, params.input_file)
        }
        if (!f.exists()) {
            error "Configured input.file does not exist: ${f}"
        }
        hyperstack_input = f
        matched_files = []  // explicit override -> ignore per-timepoint files
        log.info "Using explicit input.file: ${hyperstack_input}"
    } else if (matched_files.isEmpty()) {
        // Auto-detect: prefer .czi, then a single hyperstack .tif/.tiff
        def czis = []
        def tifs = []
        def cziMatcher  = FileSystems.getDefault().getPathMatcher("glob:*.{czi,CZI}")
        def tifMatcher  = FileSystems.getDefault().getPathMatcher("glob:*.{tif,tiff,TIF,TIFF}")
        Files.list(input_dir_path).each { p ->
            def fname = p.getFileName()
            if (cziMatcher.matches(fname)) czis.add(p.toFile())
            else if (tifMatcher.matches(fname)) tifs.add(p.toFile())
        }
        if (czis.size() == 1) {
            hyperstack_input = czis[0]
            log.info "Auto-detected single CZI input: ${hyperstack_input.name}"
        } else if (czis.size() > 1) {
            error "Multiple .czi files found in ${params.input_dir}. Set input.file in config.json to choose one."
        } else if (tifs.size() == 1) {
            hyperstack_input = tifs[0]
            log.info "Auto-detected single TIFF input (will check for hyperstack axes): ${hyperstack_input.name}"
        }
    }

    if (hyperstack_input != null) {
        // Run the SPLIT step. It writes per-timepoint TIFFs that we then
        // feed into the rest of the pipeline as if they were the original
        // input directory.
        log.info "Splitting hyperstack/CZI input into per-timepoint TIFFs..."
        SPLIT_INPUT_FILE(
            Channel.fromPath(hyperstack_input.toString(), checkIfExists: true),
            params.channel
        )

        // SPLIT_INPUT_FILE.out.timepoints emits a list of files (one per t)
        input_channel = SPLIT_INPUT_FILE.out.timepoints
            .flatten()
            .map { f ->
                // Filenames are guaranteed to be t####_Channel <c>.tif
                def m = (f.getName() =~ /t(\d+)_Channel/)
                def tp = m.find() ? m.group(1).toInteger() : 0
                tuple(tp, f)
            }
            .toSortedList { a, b -> a[0] <=> b[0] }
            .flatMap { it }

        if (max_timepoints != null) {
            log.info "Limiting input to first ${max_timepoints} timepoint(s) (input.max_timepoints)"
            input_channel = input_channel.take(max_timepoints)
        }

        input_channel.subscribe { timepoint, f ->
            log.info "Split timepoint ${timepoint}: ${f.name}"
        }
    } else {
        // ---- Fallback: if no files match the expected naming convention,
        //      collect ALL .tif/.tiff files in the directory and assign
        //      timepoints based on whatever numeric info can be extracted
        //      from the filenames (or alphabetical order as last resort). ----
        def used_fallback = false
        if (matched_files.isEmpty()) {
            log.warn "No files matching '${glob_pattern}' — falling back to all TIF files in ${params.input_dir}"
            def tifMatcher  = FileSystems.getDefault().getPathMatcher("glob:*.tif")
            def tiffMatcher = FileSystems.getDefault().getPathMatcher("glob:*.tiff")
            Files.list(input_dir_path).each { p ->
                def fname = p.getFileName()
                if (tifMatcher.matches(fname) || tiffMatcher.matches(fname)) {
                    matched_files.add(p.toFile())
                }
            }
            used_fallback = true
        }

        if (matched_files.isEmpty()) {
            error "No .tif/.tiff/.czi files found in: ${params.input_dir}"
        }

        log.info "Found ${matched_files.size()} files ${used_fallback ? '(fallback – all TIFs)' : 'matching pattern'} in: ${params.input_dir}"

        // --- Helper: extract a numeric timepoint from a filename ---
        // Tries several common patterns in order:
        //   1. t####_Channel  (original convention)
        //   2. t#### or T####  (e.g. t0000.tif, T012_decon.tif)
        //   3. tp#### or TP#### (e.g. tp001.tif)
        //   4. _####. or -####. (trailing number before extension, e.g. embryo_001.tif)
        //   5. Any digit run in the filename
        // Returns null if nothing numeric is found.
        def extractTimepoint = { String name ->
            def m
            // Pattern 1: original t####_Channel
            m = (name =~ /(?i)t(\d+)_Channel/)
            if (m.find()) return m.group(1).toInteger()
            // Pattern 2: t#### or T####
            m = (name =~ /(?i)(?:^|[^a-z])t(\d+)/)
            if (m.find()) return m.group(1).toInteger()
            // Pattern 3: tp#### or TP####
            m = (name =~ /(?i)tp(\d+)/)
            if (m.find()) return m.group(1).toInteger()
            // Pattern 4: trailing number before extension  e.g. img_001.tif
            m = (name =~ /[_\-\s](\d+)\.[tT][iI][fF]{1,2}$/)
            if (m.find()) return m.group(1).toInteger()
            // Pattern 5: first digit run anywhere in the name
            m = (name =~ /(\d+)/)
            if (m.find()) return m.group(1).toInteger()
            return null
        }

        // Build (timepoint, file) tuples.  When numeric extraction fails for
        // ALL files we fall back to alphabetical ordering.
        def file_tuples = matched_files.collect { f ->
            def tp = extractTimepoint(f.name)
            return [tp, f]
        }

        def all_null = file_tuples.every { it[0] == null }

        if (all_null) {
            // No numeric info at all — sort alphabetically and assign 0, 1, 2, …
            log.warn "Could not extract numeric timepoints from filenames — using alphabetical order"
            file_tuples = file_tuples.sort { it[1].name }
            file_tuples = file_tuples.withIndex().collect { entry, idx -> [idx, entry[1]] }
        } else {
            // For any file where extraction failed, assign a unique large number so
            // it sorts to the end rather than causing a crash.
            def max_tp = file_tuples.findAll { it[0] != null }.collect { it[0] }.max() ?: 0
            def fallback_tp = max_tp + 1
            file_tuples = file_tuples.collect { tp, f ->
                if (tp == null) {
                    log.warn "Could not extract timepoint from '${f.name}' — assigning t=${fallback_tp}"
                    def assigned = fallback_tp
                    fallback_tp++
                    return [assigned, f]
                }
                return [tp, f]
            }
            // Sort by timepoint
            file_tuples = file_tuples.sort { it[0] }
        }

        if (max_timepoints != null && file_tuples.size() > max_timepoints) {
            log.info "Limiting input to first ${max_timepoints} of ${file_tuples.size()} timepoint(s) (input.max_timepoints)"
            file_tuples = file_tuples.take(max_timepoints)
        }

        input_channel = Channel
            .fromList(file_tuples.collect { tp, f -> tuple(tp, file(f.toPath())) })
            .ifEmpty { error "No TIF files could be loaded from: ${params.input_dir}" }
            .tap { parsed_files }

        // Log parsed files
        parsed_files.subscribe { timepoint, file ->
            log.info "Found timepoint ${timepoint}: ${file.name}"
        }
    } // end else (no hyperstack input)

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

    // ---------------------------------------------------------------------
    // Canonical voxel sizes from shared_metadata.json. We compute them
    // here once (the user's config.json overrides OR the auto-detected
    // values, whichever won in EXTRACT_METADATA) and pass them to every
    // preprocessing / downscale / segmentation task as GString
    // interpolations into the bash heredocs (${metadata_voxel_x} etc.).
    //
    // Why this matters: lightsheet acquisition software commonly writes
    // spacing=1.0 / XResolution=96dpi into the raw TIFF's ImageJ block,
    // and the raw TIFFs in this pipeline carry those bogus values. If a
    // task re-reads voxel sizes from the input TIFF it inherits the
    // bogus defaults and writes them straight into its output, silently
    // corrupting every downstream stack-alignment / Cellpose-anisotropy
    // / viewer-overlay calculation. Reading from shared_metadata.json
    // (the user's config) is the single source of truth.
    //
    // The script-scope doubles metadata_voxel_x/y/z (declared near the
    // top of this file alongside planar_sigma_xy etc.) are overwritten
    // here so the existing ${metadata_voxel_x} interpolations in the
    // process bodies can capture them via lexical scope.
    // ---------------------------------------------------------------------
    try {
        def _sm = new groovy.json.JsonSlurper().parseText(file(shared_metadata).text)
        metadata_voxel_x = (_sm.x_resolution_um as Double)
        metadata_voxel_y = (_sm.y_resolution_um as Double)
        metadata_voxel_z = (_sm.imagej?.spacing as Double) ?: 1.0d
    } catch (Exception _e) {
        log.warn "Could not parse shared_metadata.json for voxel sizes: ${_e.message}; " +
                 "keeping conservative defaults (0.347, 0.347, 2.0)"
    }
    log.info "Canonical voxel sizes (from shared_metadata.json): " +
             "x=${metadata_voxel_x} µm, y=${metadata_voxel_y} µm, z=${metadata_voxel_z} µm"

    // 1b. OPTIONAL: Raw export — produce a downscaled + isotropic version of
    //     the RAW (unprocessed) input for overlaying tracks in ultrack_viewer.
    //     Runs on processing_input (post-ROI-crop) so it sees the same
    //     bounding box as segmentation. INDEPENDENT of preprocessing,
    //     segmentation and tracking — purely additive.
    raw_iso_input = null
    if (raw_export_enabled) {
        log.info "Raw export ENABLED — producing downscaled+isotropic RAW volumes for track overlay (factor=${raw_export_factor}, iso=${raw_export_iso})"
        EXPORT_RAW_ISOTROPIC(
            processing_input,
            shared_metadata,
            raw_export_factor,
            raw_export_iso
        )
        raw_iso_input = EXPORT_RAW_ISOTROPIC.out.raw_iso
    } else {
        log.info "Raw export disabled (set raw_export.enabled=true in config.json to enable)"
    }

    if (skip_preprocessing) {
        // ---- Skip preprocessing ----
        if (preprocessed_dir) {
            // Load already-processed TIFs from preprocessed_dir
            log.info "Preprocessing SKIPPED — reading pre-processed images from: ${preprocessed_dir}"

            def preproc_path = Paths.get(preprocessed_dir)
            def preproc_files = []
            Files.list(preproc_path).each { p ->
                def fname = p.getFileName().toString()
                if (fname.endsWith('.tif') || fname.endsWith('.tiff')) {
                    preproc_files.add(p.toFile())
                }
            }

            if (preproc_files.isEmpty()) {
                error "No .tif/.tiff files found in preprocessed_dir: ${preprocessed_dir}"
            }

            log.info "Found ${preproc_files.size()} preprocessed files in: ${preprocessed_dir}"

            // Sort by extracted timepoint and apply max_timepoints limit
            // before building the channel.
            def preproc_tuples = preproc_files.collect { f ->
                def m = (f.name =~ /(?i)t(\d+)/)
                def tp = m.find() ? m.group(1).toInteger() : null
                if (tp == null) {
                    log.warn "Could not extract timepoint from preprocessed file: ${f.name}"
                }
                return [tp, f]
            }.findAll { it[0] != null }.sort { it[0] }

            if (max_timepoints != null && preproc_tuples.size() > max_timepoints) {
                log.info "Limiting preprocessed input to first ${max_timepoints} of ${preproc_tuples.size()} timepoint(s) (input.max_timepoints)"
                preproc_tuples = preproc_tuples.take(max_timepoints)
            }

            segmentation_input = Channel
                .fromList(preproc_tuples.collect { tp, f -> tuple(tp, file(f.toPath())) })
                .ifEmpty { error "No preprocessed files with recognizable timepoint numbers found in: ${preprocessed_dir}" }

            // Apply standalone downscaling on top of user-supplied preprocessed
            // files when downscaling is enabled. Z reslice is skipped here to
            // preserve whatever Z geometry the external files already carry.
            if (run_standalone_downscaling) {
                log.info "Applying DOWNSCALE_XY (factor=${effective_scaling}) on user-supplied preprocessed files; Z reslice skipped"
                DOWNSCALE_XY(segmentation_input, shared_metadata, effective_scaling, false, bin_dir_ch)
                segmentation_input = DOWNSCALE_XY.out.downscaled
            }
        } else if (run_standalone_downscaling) {
            log.info "Preprocessing SKIPPED — applying standalone XY downscaling before segmentation: factor=${effective_scaling}"
            if (isotropic_reslice) {
                log.info "  (also isotropic Z reslice, in the same DOWNSCALE_XY task)"
            }
            DOWNSCALE_XY(processing_input, shared_metadata, effective_scaling, isotropic_reslice, bin_dir_ch)
            segmentation_input = DOWNSCALE_XY.out.downscaled
        } else if (isotropic_reslice) {
            log.info "Preprocessing SKIPPED — applying lightweight isotropic Z reslicing only (preprocessing.isotropic_reslice=true)"
            RESLICE_ISOTROPIC(processing_input, shared_metadata)
            segmentation_input = RESLICE_ISOTROPIC.out.resliced
        } else {
            log.info "Preprocessing SKIPPED — using raw input images for segmentation (no isotropic reslicing)"
            segmentation_input = processing_input
        }

    } else {
        // ---- Normal preprocessing ----
        //
        // Chain design (commit e0c9d0a, 2026-09-09, "Option B" reorder):
        //
        //   PLANAR -> DEPTH -> [DOWNSCALE_XY(do_iso=true) | ISOTROPIC]
        //
        // The original chain ran PLANAR -> DEPTH -> ISOTROPIC -> DOWNSCALE_XY,
        // which produced an anisotropic final stack (e.g. 0.694 x 0.694 x
        // 0.374 um) and forced every viewer-overlay / tracking math to deal
        // with three different voxel sizes. Reordered so the chain ENDS at
        // isotropic voxels:
        //
        //   - When downscaling.enabled=true: DOWNSCALE_XY runs LAST with
        //     do_iso=true, doing both the XY rescale AND the Z-reslice to
        //     the new (smaller) XY voxel size in a single task. ISOTROPIC
        //     is NOT needed — DOWNSCALE_XY already produces a fully
        //     isotropic stack.
        //   - When downscaling.enabled=false: ISOTROPIC runs LAST, doing
        //     Z-reslice to the raw XY voxel size. DOWNSCALE_XY is not
        //     called.
        //
        // Both branches produce an isotropic final stack. PLANAR and DEPTH
        // are lean modular steps (ported from AIAF-32). Each is its own
        // Nextflow process so:
        //   1. resources can be tuned per step (planar is XY-heavy,
        //      depth is Z-light)
        //   2. any single step can be re-run with `nextflow run -resume`
        //      after a parameter change
        //   3. intermediate TIFFs (planar / depth) are published and useful
        //      for QA
        log.info "Preprocessing chain: planar -> depth -> " +
                 (downscaling_enabled && effective_scaling < 1.0d
                     ? "downscale_xy (XY + Z isotropic)"
                     : "isotropic (Z only)")
        // Stage the entire bin/ directory as a single input so the per-step
        // scripts (and their _tiff_io.py dependency) are all available in
        // the task workdir when the script's first line is `from _tiff_io
        // import ...`. Using path() (not file()) preserves the directory
        // name 'bin/' which the script invocation ``python3 bin/<script>.py``
        // relies on for the relative import.
        bin_dir_ch = Channel.fromPath("${projectDir}/bin", type: 'dir', checkIfExists: true).collect()

        // Step 1: planar (XY) shading correction
        PLANAR_CORRECTION(
            processing_input,
            shared_metadata,
            bin_dir_ch
        )

        // Step 2: depth (Z) intensity correction, consumes planar output
        DEPTH_CORRECTION(
            PLANAR_CORRECTION.out.corrected,
            shared_metadata,
            bin_dir_ch
        )

        // Step 3: final geometric step. DOWNSCALE_XY (when downscaling is
        // enabled) does both the XY halving AND the isotropic Z reslice in
        // one task. Otherwise ISOTROPIC does only the Z reslice to the
        // raw XY voxel size. The chain ENDS at isotropic voxels in both
        // cases.
        if (downscaling_enabled && effective_scaling < 1.0d) {
            log.info "Downscale XY (factor=${effective_scaling}) + isotropic Z reslice to the new XY voxel size"
            DOWNSCALE_XY(
                DEPTH_CORRECTION.out.corrected,
                shared_metadata,
                effective_scaling,
                true,  // do_iso=true: reslice Z to match the new (halved) XY
                bin_dir_ch
            )
            segmentation_input = DOWNSCALE_XY.out.downscaled
        } else {
            log.info "Isotropic Z reslice to the raw XY voxel size"
            ISOTROPIC(
                DEPTH_CORRECTION.out.corrected,
                shared_metadata,
                bin_dir_ch
            )
            segmentation_input = ISOTROPIC.out.processed
        }

    } // end skip_preprocessing else

    // 3. Segment each timepoint with Cellpose
    if (!skip_segmentation) {
        // effective_scaling comes from the top-level downscaling.{enabled,factor}
        // block (defined near skip_preprocessing). It is reused by CELLPOSE_SEGMENT
        // below to write correct voxel sizes into the segmentation mask metadata.

        CELLPOSE_SEGMENT(
            segmentation_input,
            shared_metadata,
            config.segmentation,
            effective_scaling,
            bin_dir_ch
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
        //
        // Single MERGE_HYPERSTACKS task that handles all data types in
        // parallel streams. Each input list is optional — empty lists
        // are skipped inside the task. This avoids the DSL2
        // "process already used" error that would occur if we called
        // the old per-data-type MERGE_TO_HYPERSTACK multiple times.
        if (!skip_merge) {
            // Build the processed file list (the chain's "processed" output).
            // Process input is `path` which requires a non-empty value →
            // wrap each empty channel with `ifEmpty` that stages a
            // placeholder file the bash script detects and skips.
            processed_ch = segmentation_input
                .map { timepoint, f -> f }
                .collect()
                .ifEmpty { [file("${workflow.projectDir}/merge_hyperstack.py")] }

            // Build the segmented file list (only meaningful if segmentation ran)
            segmented_ch = (skip_segmentation
                ? Channel.value(file("${workflow.projectDir}/merge_hyperstack.py"))
                : CELLPOSE_SEGMENT.out.segmented
                    .map { timepoint, f -> f }
                    .collect()
                    .ifEmpty { [file("${workflow.projectDir}/merge_hyperstack.py")] })

            // Build the raw_iso file list (only meaningful if raw_export ran)
            raw_iso_ch = (raw_export_enabled && raw_iso_input != null
                ? raw_iso_input.map { timepoint, f -> f }
                    .collect()
                    .ifEmpty { [file("${workflow.projectDir}/merge_hyperstack.py")] }
                : Channel.value(file("${workflow.projectDir}/merge_hyperstack.py")))

            log.info "Hyperstack merging enabled"

            MERGE_HYPERSTACKS(
                shared_metadata,
                Channel.fromPath(params.merge_script, checkIfExists: true).first(),
                processed_ch,
                segmented_ch,
                raw_iso_ch
            )
        } else {
            log.info "Hyperstack merging SKIPPED (skip_merge=true)"
        }

        // 6. OPTIONAL: ultrack tracking (requires merge to produce hyperstacks)
        if (!skip_tracking) {
            log.info "Ultrack tracking enabled"

            // The TIFF hyperstack is required for ultrack prep (uses tifffile).
            // Filenames contain the data type: 4D_hyperstack_processed.tif,
            // 4D_hyperstack_segmented.tif.
            processed_hs = MERGE_HYPERSTACKS.out.processed_tif
            segmented_hs = MERGE_HYPERSTACKS.out.segmented_tif

            prep_ultrack_script_ch = Channel.fromPath(params.prep_ultrack_script, checkIfExists: true)

            // Step 1: Prepare foreground + contours zarrs (GPU)
            PREP_ULTRACK(
                processed_hs,
                segmented_hs,
                prep_ultrack_script_ch.collect(),
                config.tracking.prep
            )

            // Step 2: ultrack segment → link → solve → export (4 separate processes)
            // ultrack_config_toml is resolved at startup (defaults to repo-root ultrack_config.toml)
            ultrack_config_ch = Channel.fromPath(
                config.tracking.ultrack_config_toml,
                checkIfExists: true
            )

            ULTRACK_SEGMENT(
                PREP_ULTRACK.out.foreground,
                PREP_ULTRACK.out.contours,
                ultrack_config_ch.collect()
            )

            ULTRACK_LINK(
                ULTRACK_SEGMENT.out.config_toml,
                ULTRACK_SEGMENT.out.database,
                ULTRACK_SEGMENT.out.metadata
            )

            ULTRACK_SOLVE(
                ULTRACK_LINK.out.config_toml,
                ULTRACK_LINK.out.database,
                ULTRACK_LINK.out.metadata
            )

            ULTRACK_EXPORT(
                ULTRACK_SOLVE.out.config_toml,
                ULTRACK_SOLVE.out.database,
                ULTRACK_SOLVE.out.metadata
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

            // When per-timepoint TIFFs are NOT published (merge enabled), stage
            // them from the work dir so benchmark can still read them. When
            // they ARE published (skip_merge=true), pass empty lists and let
            // benchmark read directly from publishDir to avoid staging GBs.
            if (skip_merge) {
                log.info "Benchmark will read per-timepoint TIFFs from publishDir: ${abs_output_dir}"
                proc_for_bench = Channel.value([])
                seg_for_bench  = Channel.value([])
            } else {
                log.info "Benchmark will stage per-timepoint TIFFs from work dir (merge enabled, files not in publishDir)"
                proc_for_bench = segmentation_input
                    .map { timepoint, f -> f }
                    .collect()
                seg_for_bench = CELLPOSE_SEGMENT.out.segmented
                    .map { timepoint, f -> f }
                    .collect()
            }

            BENCHMARK(
                abs_output_dir,
                proc_for_bench,
                seg_for_bench,
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

            processed_ch = segmentation_input
                .map { timepoint, f -> f }
                .collect()
                .ifEmpty { [file("${workflow.projectDir}/merge_hyperstack.py")] }
            raw_iso_ch = (raw_export_enabled && raw_iso_input != null
                ? raw_iso_input.map { timepoint, f -> f }
                    .collect()
                    .ifEmpty { [file("${workflow.projectDir}/merge_hyperstack.py")] }
                : Channel.value(file("${workflow.projectDir}/merge_hyperstack.py")))

            MERGE_HYPERSTACKS(
                shared_metadata,
                Channel.fromPath(params.merge_script, checkIfExists: true).first(),
                processed_ch,
                Channel.value(file("${workflow.projectDir}/merge_hyperstack.py")),
                raw_iso_ch
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
    def label_downscale_status = ds_factor < 1.0 ? "ENABLED (${ds_factor})" : "DISABLED"
    def xy_downscale_status = downscaling_enabled ? "ENABLED (factor=${effective_scaling})" : "DISABLED"

    def benchmark_status = (config.benchmark?.enabled ?: false) ? "ENABLED" : "DISABLED"
    def tracking_status = (config.tracking?.enabled ?: false) ? "ENABLED (ultrack)" : "DISABLED"
    def debug_status = (config.preprocessing?.debug_preprocessing?.enabled ?: false) ? "ENABLED" : "DISABLED"
    def raw_export_status = raw_export_enabled ? "ENABLED (factor=${raw_export_factor}, iso=${raw_export_iso})" : "DISABLED"

    log.info """
    ============================================================================
    Pipeline completed!
    ============================================================================
    Status          : ${workflow.success ? 'SUCCESS ✓' : 'FAILED ✗'}
    Duration        : ${workflow.duration}
    Channel         : ${params.channel}
    ROI cropping    : ${roi_status}
    Segmentation    : ${seg_status}
    Voxel mode      : ${voxel_mode}
    Merge           : ${merge_status}
    XY downscale    : ${xy_downscale_status}
    Label downscale : ${label_downscale_status}
    Tracking        : ${tracking_status}
    Benchmark       : ${benchmark_status}
    Debug preproc   : ${debug_status}
    Raw export      : ${raw_export_status}
    Output dir      : ${params.output_dir}

    Results:
      ${config.roi_cropping?.enabled ? "- Cropped images     : ${params.output_dir}/00_cropped/" : ""}
      ${run_standalone_downscaling ? "- Downscaled input   : ${params.output_dir}/00c_downscaled/" : ""}
      - Preprocessed images : ${params.output_dir}/01_preprocessed/
      ${raw_export_enabled ? "- Raw isotropic (for --processed in viewer): ${params.output_dir}/01b_raw_isotropic/" : ""}
      - Segmented masks     : ${params.output_dir}/02_segmented/
      ${ds_factor < 1.0 ? "- Downscaled labels  : ${params.output_dir}/02_segmented_downscaled/" : ""}
      ${!(config.output?.skip_merge ?: false) ? "- Hyperstacks        : in 01_preprocessed/, ${raw_export_enabled ? '01b_raw_isotropic/, ' : ''}and 02_segmented/" : ""}
      ${(config.tracking?.enabled ?: false) ? "- Tracking results   : ${params.output_dir}/03_tracking/" : ""}
      ${(config.benchmark?.enabled ?: false) ? "- Benchmark          : ${params.output_dir}/benchmark/" : ""}
      ${(config.preprocessing?.debug_preprocessing?.enabled ?: false) ? "- Debug nuclei report: ${params.output_dir}/debug_preprocessing/" : ""}
      - Logs                : ${params.output_dir}/logs/

    Completed at: ${workflow.complete}
    ============================================================================
    """.stripIndent()

    // Sanity check: if tracking was enabled but the merge step did NOT
    // produce a hyperstack on disk, point the user at the cause and a fix.
    // (Happened in 2026-08-25: MERGE_HYPERSTACKS saw empty inputs because
    // the stage-n logic only looked for files inside named subdirs, while
    // Nextflow actually stages list-of-file inputs loose in the work dir.)
    if ((config.tracking?.enabled ?: false) && !(config.output?.skip_merge ?: false)) {
        def proc_hs = file("${params.output_dir}/01_preprocessed/4D_hyperstack_processed.tif")
        def seg_hs  = file("${params.output_dir}/02_segmented/4D_hyperstack_segmented.tif")
        if (!proc_hs.exists() || !seg_hs.exists()) {
            log.warn """
================================================================================
TRACKING WAS REQUESTED BUT NO HYPERSTACKS WERE PRODUCED.
  - 01_preprocessed/4D_hyperstack_processed.tif exists? ${proc_hs.exists()}
  - 02_segmented/4D_hyperstack_segmented.tif exists?   ${seg_hs.exists()}

This is the symptom of MERGE_HYPERSTACKS running with empty inputs (the
pre-fix stage_n() looked for files inside named subdirs that Nextflow does
not create for list-of-file inputs). Tracking needs both hyperstacks on
disk to proceed.

The current fix invalidates the merge cache via a script-body version
stamp, so a plain -resume should re-run merge with the corrected logic.
If it does not, force-rebuild merge with:

    rm -rf work/*/8d/218dc037a542d356a650a27af6de13   # the cached merge task
    nextflow run spim_pipeline.nf --config_json config.json -resume

(use the work-dir hash printed in the previous run log for the actual
MERGE_HYPERSTACKS task; the placeholder above is the hash from 2026-08-25.)
================================================================================
""".stripIndent()
        }
    }
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
