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
// Self-Net deblurring front-end (alternative to deconvolution). The Self-Net
// script reuses helpers from the deconvolution script (spim_pipeline_fixed.py)
// and WBNS.py, so both are staged alongside it when method='selfnet'.
params.selfnet_script = './spim_selfnet_preprocess.py'
params.wbns_script = './WBNS.py'
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
params.output_dir  = sanitizePath(config.output.directory)
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

// Preprocessing method: 'deconvolution' (GPU Richardson-Lucy, default) or
// 'selfnet' (Self-Net deep-learning deblurring / isotropic reconstruction).
def preprocess_method = (config.preprocessing?.method ?: 'deconvolution').toString().toLowerCase()
if (!(preprocess_method in ['deconvolution', 'selfnet'])) {
    log.error "Unknown preprocessing.method '${preprocess_method}'. Use 'deconvolution' or 'selfnet'."
    exit 1
}
// Validate Self-Net settings up front so the run fails fast (not 50 jobs in).
if (!skip_preprocessing && preprocess_method == 'selfnet') {
    def sn = config.preprocessing?.selfnet
    if (!sn || !sn.model_path) {
        log.error "preprocessing.method='selfnet' requires preprocessing.selfnet.model_path in config.json"
        exit 1
    }
    def sn_model = sanitizePath(sn.model_path.toString())
    if (!file(sn_model).exists()) {
        log.error "Self-Net model not found: ${sn.model_path} (resolved to: ${file(sn_model)})"
        exit 1
    }
    log.info "Preprocessing method: SELF-NET (model: ${sn_model})"
} else if (!skip_preprocessing) {
    log.info "Preprocessing method: DECONVOLUTION"
}

// Optional: limit input to the first N timepoints (after any timepoints filter
// below). Useful when late timepoints in an acquisition are unusable (sample
// drift, photodamage, etc.) and you want to discard them without rebuilding
// the input dataset.
// Accepts an integer >= 1; null/missing/<=0 means "use all timepoints".
def max_timepoints = config.input?.max_timepoints != null \
    ? (config.input.max_timepoints as Integer) \
    : null
if (max_timepoints != null && max_timepoints <= 0) {
    max_timepoints = null
}

// Optional: process ONLY these specific timepoints. Each entry can be a
// numeric timepoint index (1-based: 1, 5, 10) or a filename stem
// ('t0001', 't0005', 't0001_Channel 2'). null/missing means "use all".
// `max_timepoints` is applied AFTER this filter (so e.g.
// timepoints=[1..100] + max_timepoints=3 → the first 3 of those 100).
// Invalid entries (no matching file) fail the run fast.
def timepoints_selection = config.input?.timepoints != null \
    ? (config.input.timepoints as List) \
    : null
if (timepoints_selection != null && timepoints_selection.size() == 0) {
    timepoints_selection = null
}

// Match a single (tp, file) tuple against the timepoints_selection list.
// A selection entry matches when it equals:
//   - the numeric timepoint (e.g. 5 == 5)
//   - the numeric timepoint as a string (e.g. "5" == 5)
//   - the filename stem without extension (e.g. "t0005" matches "t0005.tif")
//   - the full filename (e.g. "t0005.tif")
def _matchesTimepoint = { tp, f, sel ->
    def stem = f.name.replaceAll(/\.(tif|tiff|czi)$/, '')
    if (sel == null) return true
    if (tp != null) {
        if (sel instanceof Number && sel == tp) return true
        if (sel instanceof String && (sel == tp.toString() || sel == stem || sel == f.name)) return true
    } else {
        if (sel == stem || sel == f.name) return true
    }
    return false
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

// Set defaults for debug_preprocessing (lives inside preprocessing section)
if (!config.preprocessing.containsKey('debug_preprocessing')) {
    config.preprocessing.debug_preprocessing = [enabled: false]
}
def dbg = config.preprocessing.debug_preprocessing
if (!dbg.containsKey('save_masks')) { dbg.save_masks = false }
// Always inherit cellpose params from segmentation section so the debug
// analysis uses the exact same model & thresholds as the real segmentation.
dbg.cellpose_model              = dbg.cellpose_model              ?: config.segmentation?.model              ?: 'cyto3'
dbg.cellpose_diameter           = dbg.cellpose_diameter           ?: config.segmentation?.diameter           ?: 30
dbg.cellpose_flow_threshold     = dbg.cellpose_flow_threshold     ?: config.segmentation?.flow_threshold     ?: 0.8
dbg.cellpose_cellprob_threshold = dbg.cellpose_cellprob_threshold ?: config.segmentation?.cellprob_threshold ?: 0.0
dbg.cellpose_do_3d              = dbg.cellpose_do_3d              ?: config.segmentation?.do_3d              ?: true
dbg.cellpose_min_size           = dbg.cellpose_min_size           ?: config.segmentation?.min_size           ?: 15
def run_debug_preprocessing = dbg.enabled ?: false

// Debug preprocessing requires save_intermediates — force it on
if (run_debug_preprocessing && !config.preprocessing.save_intermediates) {
    log.warn "debug_preprocessing.enabled=true requires save_intermediates=true — forcing save_intermediates on"
    config.preprocessing.save_intermediates = true
}

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
def downscale_info = downscale_labels < 1.0 ? "${downscale_labels} (Fiji nearest-neighbor)" : "Disabled"
def seg_mode_info = config.segmentation.do_3d ? "3D" : (config.segmentation.stitch_threshold != null ? "2D+Stitch(${config.segmentation.stitch_threshold})" : "2D")
def tracking_info = skip_tracking ? "SKIPPED" : "Enabled (ultrack)"
def debug_info = run_debug_preprocessing ? "ENABLED (nuclei tracking per stage)" : "Disabled"
def preproc_info = skip_preprocessing ? (preprocessed_dir ? "SKIPPED (using ${preprocessed_dir})" : "SKIPPED (using raw input)") : "Enabled"

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
Downscale    : ${downscale_info}
Seg mode     : ${seg_mode_info}
Preprocess   : ${preproc_info}
Tracking     : ${tracking_info}
Debug preproc: ${debug_info}
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
import re
m = re.search(r'_Channel\\s*(\\d+)', '${filename}')
channel = m.group(1) if m else '1'
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

    // Only publish per-timepoint preprocessed TIFFs when the 4D hyperstack
    // merge is disabled. Otherwise the same data ends up in publishDir twice
    // (once per timepoint, once inside 4D_hyperstack_processed.tif).
    publishDir "${params.output_dir}/01_preprocessed",
        mode: 'copy',
        pattern: "*_processed.tif",
        enabled: skip_merge

    publishDir "${params.output_dir}/logs/preprocessing",
        mode: 'copy',
        pattern: "*.log"

    publishDir "${params.output_dir}/intermediates/t${String.format('%04d', timepoint)}",
        mode: 'copy',
        pattern: "intermediates/*.tif"

    container params.container

    input:
    tuple val(timepoint), path(image_file)
    path metadata_json
    path preproc_script
    val preprocess_config

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_processed.tif"), emit: processed
    path "t${String.format('%04d', timepoint)}_preprocess.log", emit: log
    tuple val(timepoint), path("intermediates/*.tif"), optional: true, emit: intermediates

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
xy_pixel = metadata['x_resolution_um']
z_pixel = metadata['imagej']['spacing']

print(f"Using voxel sizes from metadata:")
print(f"  XY pixel size: {xy_pixel:.4f} µm")
print(f"  Z pixel size: {z_pixel:.4f} µm")
print(f"  Source: {metadata.get('voxel_size_source', 'unknown')}")
if metadata.get('was_roi_cropped', False):
    print(f"  Note: Image was ROI-cropped (voxel size unchanged)")
print("")

# Write a per-timepoint preprocessing config that injects the metadata
# voxel sizes into the canonical preprocessing config. The Python script
# reads this single file (no flag-flattening in Nextflow).
preproc_config = dict(config)
preproc_config.setdefault('voxel_size', {})
preproc_config['voxel_size']['x_um'] = float(xy_pixel)
preproc_config['voxel_size']['y_um'] = float(metadata.get('y_resolution_um', xy_pixel))
preproc_config['voxel_size']['z_um'] = float(z_pixel)
preproc_config['voxel_size']['auto_detect'] = False

# Backward-compat alias (merge_hyperstack.py reads this directly).
preproc_config.setdefault('image_scaling',
                          preproc_config.get('downscale_xy', {}).get('factor', 1.0))

with open('preprocessing_config.json', 'w') as f:
    json.dump(preproc_config, f, indent=2)
print(f"Wrote preprocessing_config.json with voxel sizes from metadata.")

# Build command — config is the only knob now (legacy CLI flags are accepted
# by the script but optional).
cmd = [
    'python3', '${script_name}',
    '--input_file', '${filename}',
    '--outdir', '.',
    '--metadata_json', '${metadata_json}',
    '--config_json', 'preprocessing_config.json',
]

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
// PROCESS: Preprocess Single Timepoint with Self-Net deblurring
// (alternative to PREPROCESS_DECONVOLVE; selected via preprocessing.method)
// ============================================================================

process PREPROCESS_SELFNET {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

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
    path selfnet_script
    path shared_lib       // spim_pipeline_fixed.py (provides shared helpers)
    path wbns_lib         // WBNS.py (imported by the shared lib)
    val preprocess_config

    output:
    tuple val(timepoint), path("t${String.format('%04d', timepoint)}_processed.tif"), emit: processed
    path "t${String.format('%04d', timepoint)}_preprocess.log", emit: log
    tuple val(timepoint), path("intermediates/*.tif"), optional: true, emit: intermediates

    script:
    def cfg = preprocess_config
    def t_formatted = String.format('%04d', timepoint)
    def filename = image_file.name
    def config_json_str = groovy.json.JsonOutput.toJson(cfg).replace("'", "\\'")
    def script_name = selfnet_script.name
    """
    #!/bin/bash
    set -euo pipefail

    # Activate micromamba environment
    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "============================================"
    echo "Preprocessing timepoint: ${timepoint} (Self-Net)"
    echo "File: ${filename}"
    echo "Self-Net script: ${script_name}"
    echo "============================================"

    if [ ! -f "${script_name}" ]; then
        echo "ERROR: Self-Net script not found: ${script_name}"
        echo "Contents of work directory:"
        ls -lh
        exit 1
    fi

    # Run Self-Net preprocessing with all parameters from config
    python3 << 'PYTHON_EOF'
import json
import sys
import subprocess

# Load metadata
with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

# Load config - parse from JSON string to handle booleans correctly
config = json.loads('${config_json_str}')
selfnet = config.get('selfnet', {})

xy_pixel = metadata['x_resolution_um']
z_pixel = metadata['imagej']['spacing']

print(f"Using voxel sizes from metadata:")
print(f"  XY pixel size: {xy_pixel:.4f} um")
print(f"  Z pixel size: {z_pixel:.4f} um")
print(f"  Source: {metadata.get('voxel_size_source', 'unknown')}")
print("")

# Write a per-timepoint preprocessing config that injects the metadata
# voxel sizes into the canonical preprocessing config.
preproc_config = dict(config)
preproc_config.setdefault('voxel_size', {})
preproc_config['voxel_size']['x_um'] = float(xy_pixel)
preproc_config['voxel_size']['y_um'] = float(metadata.get('y_resolution_um', xy_pixel))
preproc_config['voxel_size']['z_um'] = float(z_pixel)
preproc_config['voxel_size']['auto_detect'] = False

preproc_config.setdefault('image_scaling',
                          preproc_config.get('downscale_xy', {}).get('factor', 1.0))

with open('preprocessing_config.json', 'w') as f:
    json.dump(preproc_config, f, indent=2)
print(f"Wrote preprocessing_config.json with voxel sizes from metadata.")

cmd = [
    'python3', '${script_name}',
    '--input_file', '${filename}',
    '--outdir', '.',
    '--metadata_json', '${metadata_json}',
    '--config_json', 'preprocessing_config.json',
]

print("Self-Net command:", ' '.join(cmd))
print("\\n" + "="*60)

result = subprocess.run(cmd, capture_output=True, text=True)

with open('t${t_formatted}_preprocess.log', 'w') as f:
    f.write("STDOUT:\\n")
    f.write(result.stdout)
    f.write("\\n\\nSTDERR:\\n")
    f.write(result.stderr)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)

if result.returncode != 0:
    print(f"ERROR: Self-Net preprocessing failed with exit code {result.returncode}")
    sys.exit(result.returncode)
PYTHON_EOF

    # Find and rename output to standard format with ROBUST pattern matching
    echo ""
    echo "Finding processed output file..."
    echo "Expected scaling: ${cfg.image_scaling}"
    echo "All TIF files in directory:"
    ls -lh *.tif 2>/dev/null || echo "No .tif files found"
    echo ""

    ORIGINAL_OUTPUT=""

    if [ -z "\$ORIGINAL_OUTPUT" ]; then
        SCALING_STR=\$(echo "${cfg.image_scaling}" | sed 's/\\.//g')
        echo "Trying pattern 1: *_\${SCALING_STR}*.tif"
        ORIGINAL_OUTPUT=\$(find . -maxdepth 1 -name "*_\${SCALING_STR}*.tif" -not -name "${filename}" 2>/dev/null | head -1)
        [ -n "\$ORIGINAL_OUTPUT" ] && echo "  ✓ Found: \$ORIGINAL_OUTPUT"
    fi

    if [ -z "\$ORIGINAL_OUTPUT" ]; then
        SCALING_PCT=\$(python3 -c "print(int(${cfg.image_scaling} * 100))")
        echo "Trying pattern 2: *_\${SCALING_PCT}*.tif"
        ORIGINAL_OUTPUT=\$(find . -maxdepth 1 -name "*_\${SCALING_PCT}*.tif" -not -name "${filename}" 2>/dev/null | head -1)
        [ -n "\$ORIGINAL_OUTPUT" ] && echo "  ✓ Found: \$ORIGINAL_OUTPUT"
    fi

    if [ -z "\$ORIGINAL_OUTPUT" ]; then
        echo "Trying pattern 3: any new .tif file (not input)"
        ORIGINAL_OUTPUT=\$(find . -maxdepth 1 -name "*.tif" -not -name "${filename}" -newer "${script_name}" 2>/dev/null | head -1)
        [ -n "\$ORIGINAL_OUTPUT" ] && echo "  ✓ Found: \$ORIGINAL_OUTPUT"
    fi

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

    # Restore and update metadata to processed image.
    # Self-Net performs the SAME two geometric operations as the deconv path:
    #   1. Rescales XY by image_scaling
    #   2. Reconstructs/upsamples Z to make voxels isotropic
    # so the Z spacing can be recomputed from the original vs processed Z count.
    python3 << 'RESTORE_META'
import tifffile
import json

with open('${metadata_json}', 'r') as f:
    metadata = json.load(f)

img = tifffile.imread('t${t_formatted}_processed.tif')

x_res = metadata['x_resolution_um'] / ${cfg.image_scaling}
y_res = metadata['y_resolution_um'] / ${cfg.image_scaling}
original_z_spacing = metadata['imagej']['spacing'] if 'imagej' in metadata else 1.0

original_z_slices = metadata['shape']['dimensions'][0]
new_z_slices = img.shape[0]

if new_z_slices != original_z_slices:
    z_spacing = original_z_slices * original_z_spacing / new_z_slices
    print(f"Isotropic reconstruction detected:")
    print(f"  Z slices: {original_z_slices} -> {new_z_slices}")
    print(f"  Z spacing: {original_z_spacing:.4f} -> {z_spacing:.4f} um")
else:
    z_spacing = original_z_spacing

print(f"Voxel sizes after Self-Net preprocessing:")
print(f"  Original: {metadata['x_resolution_um']:.4f} x {metadata['y_resolution_um']:.4f} x {original_z_spacing:.4f} um")
print(f"  Final:    {x_res:.4f} x {y_res:.4f} x {z_spacing:.4f} um (isotropic)")

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

    echo "Self-Net preprocessing completed for timepoint ${timepoint}"
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
// PROCESS: Simulation — One Experiment × One Timepoint
// (parallelised body of `python3 spim_pipeline_fixed.py --simulate`)
// ============================================================================

process SIM_ONE_EXPERIMENT_TP {
    tag "${exp_name}/t${String.format('%04d', timepoint)}"

    maxRetries 1
    // One bad experiment must NOT kill the entire sweep — the aggregator
    // will report missing metadata in summary.md.
    errorStrategy 'ignore'

    publishDir "${params.output_dir}/simulation/${exp_name}",
        mode: 'copy',
        pattern: "*_${scaling_pct}.tif"

    publishDir "${params.output_dir}/simulation/${exp_name}",
        mode: 'copy',
        pattern: "*_metadata.json"

    publishDir "${params.output_dir}/logs/simulation",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    val exp_name
    tuple val(timepoint), path(image_file)
    path preproc_script
    path sweep_file
    val exp_overrides_json
    val scaling_pct

    output:
    path "*_${scaling_pct}.tif", emit: tif, optional: true
    path "*_${scaling_pct}_mask.tif", emit: mask, optional: true
    path "*_${scaling_pct}_metadata.json", emit: meta, optional: true
    path "sim_${exp_name}_t${String.format('%04d', timepoint)}.log", emit: log

    script:
    def overrides_str = exp_overrides_json.replace("'", "\\'")
    """
    #!/bin/bash
    set -euo pipefail

    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    echo "============================================"
    echo "Simulation: experiment=${exp_name}  timepoint=${timepoint}"
    echo "  Input  : ${image_file}"
    echo "  Scaling: ${scaling_pct}%"
    echo "============================================"

    exec > >(tee sim_${exp_name}_t${String.format('%04d', timepoint)}.log) 2>&1

    # Stage inputs into the workdir under stable names so the Python script
    # can reference them via --sweep_file (relative path).
    cp '${sweep_file}' sweep.json
    cp '${preproc_script}' spim_pipeline_fixed.py
    cp '${preproc_script}' .  # ensures the module is importable from cwd

    # The sweep file's `input` array gets rewritten to the staged file path
    # for this timepoint only, so the per-task run touches exactly one image.
    python3 << 'PYTHON_EOF'
import json, os, sys
sys.path.insert(0, '.')

with open('sweep.json') as f:
    sweep = json.load(f)

# Override input to the single staged file (path relative to cwd).
staged_input = '${image_file}'
sweep['input'] = [os.path.basename(staged_input)]

# Inject per-experiment overrides (already a JSON string from Groovy).
overrides = json.loads('${overrides_str}')
# Embed overrides into the experiment entry with the same name.
exp_name = '${exp_name}'
for exp in sweep['experiments']:
    if exp.get('name') == exp_name:
        exp['overrides'] = overrides
        break
else:
    sweep['experiments'].append({'name': exp_name, 'overrides': overrides})

# Save updated sweep next to the staged inputs.
with open('sweep_one.json', 'w') as f:
    json.dump(sweep, f, indent=2)

# Run only this experiment on this one timepoint.
from spim_preprocessing_stages import (
    _load_image, _resolve_cellpose_config, _deep_merge,
    run_one_experiment_tp,
)

with open(sweep['base_config']) as f:
    base_config_path = sweep['base_config']
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join('.', base_config_path)
    base_config = json.load(f)

pp = base_config.get('preprocessing', {}) or {}
cp_cfg = _resolve_cellpose_config(sweep.get('cellpose', {}) or {}, base_config)

raw_stack, voxel = _load_image(staged_input)
print(f"Loaded {staged_input}: shape={raw_stack.shape} voxel={voxel}", flush=True)

meta = run_one_experiment_tp(
    exp_name=exp_name,
    overrides=overrides,
    raw_path=staged_input,
    raw_stack=raw_stack,
    voxel_size=voxel,
    cellpose_cfg=cp_cfg,
    pp_template=pp,
    output_dir='.',  # results land next to the staged inputs in the workdir
    save_intermediates=False,
    log=print,
)
print(f"Done: nuclei={meta['nuclei_count']} runtime={meta['runtime_s']:.1f}s", flush=True)
PYTHON_EOF

    # Rename the per-task outputs so Nextflow's emit patterns can match them
    # by timepoint + scaling suffix.
    base_name="\$(basename '${image_file}' | sed 's/\\.[^.]*\$//')"
    for f in *_\${scaling_pct}.tif *_\${scaling_pct}_mask.tif *_\${scaling_pct}_metadata.json; do
        [ -e "\$f" ] || continue
    done

    echo ""
    echo "✓ Sim task done: \${exp_name} on t\${timepoint}"
    """
}

// ============================================================================
// PROCESS: Simulation — Aggregate per-task metadata into summary.csv/md
// ============================================================================

process SIMULATION_AGGREGATE {
    tag "aggregate_simulation"

    maxRetries 1
    errorStrategy 'ignore'

    publishDir "${params.output_dir}/simulation",
        mode: 'copy',
        pattern: "summary_*"

    container params.container

    input:
    path preproc_script
    path sweep_file
    path meta_jsons
    val inputs_count

    output:
    path "summary.csv", emit: csv, optional: true
    path "summary.md", emit: md, optional: true
    path "aggregate.log", emit: log

    script:
    """
    #!/bin/bash
    set -euo pipefail

    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    exec > >(tee aggregate.log) 2>&1

    echo "============================================"
    echo "Aggregating simulation metadata"
    echo "============================================"

    # Stage the script for import.
    cp '${preproc_script}' spim_pipeline_fixed.py

    python3 << 'PYTHON_EOF'
import glob, os, sys
sys.path.insert(0, '.')

from spim_preprocessing_stages import aggregate_simulation_metadata

# Collect every metadata JSON staged into the workdir.
paths = sorted(glob.glob('*_metadata.json'))
print(f"Found {len(paths)} metadata JSON(s) to aggregate", flush=True)

# inputs_count is passed in as a Nextflow val so we don't need to re-read
# the sweep file (avoids encoding issues on HPC login nodes where LANG=C).
n_inputs = ${inputs_count}
print(f"Sweep declared {n_inputs} input timepoint(s)", flush=True)

aggregate_simulation_metadata(paths, output_dir='.', inputs_count=n_inputs)
PYTHON_EOF

    echo ""
    echo "✓ Aggregation done"
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
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()

# Remove any existing data_config fields we need to override
for key in ('database', 'address', 'working_dir', 'database_file_name'):
    cfg = re.sub(rf'^\\s*{key}\\s*=.*\$', '', cfg, flags=re.M)

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
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()
for key in ('database', 'address', 'working_dir', 'database_file_name'):
    cfg = re.sub(rf'^\\s*{key}\\s*=.*\$', '', cfg, flags=re.M)
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
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()
for key in ('database', 'address', 'working_dir', 'database_file_name'):
    cfg = re.sub(rf'^\\s*{key}\\s*=.*\$', '', cfg, flags=re.M)
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
    python3 << 'PYEOF'
import re, pathlib
cfg = pathlib.Path('${ultrack_config_toml}').read_text()
for key in ('database', 'address', 'working_dir', 'database_file_name'):
    cfg = re.sub(rf'^\\s*{key}\\s*=.*\$', '', cfg, flags=re.M)
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
// PROCESS: Debug Preprocessing — Nuclei Tracking per Stage (GPU)
// ============================================================================

process DEBUG_PREPROCESS_NUCLEI {
    tag "t${String.format('%04d', timepoint)}_${stage_name}"

    maxRetries 2
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'terminate' }

    publishDir "${params.output_dir}/debug_preprocessing/t${String.format('%04d', timepoint)}",
        mode: 'copy',
        pattern: "*.json"

    publishDir "${params.output_dir}/debug_preprocessing/t${String.format('%04d', timepoint)}/masks",
        mode: 'copy',
        pattern: "*_mask.tif"

    publishDir "${params.output_dir}/logs/debug_preprocessing",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), val(stage_name), path(stage_tif)
    path debug_script
    val debug_config

    output:
    tuple val(timepoint), path("${stage_name}_nuclei_stats.json"), emit: metrics
    path "*_mask.tif", optional: true, emit: masks
    path "debug_nuclei_${stage_name}.log", emit: log

    script:
    def cfg = debug_config
    def t_formatted = String.format('%04d', timepoint)
    def model = cfg.cellpose_model ?: 'cyto3'
    def save_mask_flag = cfg.save_masks ? '--save_mask' : ''
    """
    #!/bin/bash
    set -euo pipefail

    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    exec > >(tee debug_nuclei_${stage_name}.log) 2>&1

    echo "============================================"
    echo "Debug Preprocessing — Nuclei Tracking"
    echo "Timepoint: ${timepoint}"
    echo "Stage: ${stage_name}"
    echo "File: ${stage_tif}"
    echo "============================================"

    python3 ${debug_script} \\
        --input_file "${stage_tif}" \\
        --stage_name "${stage_name}" \\
        --outdir . \\
        --cellpose_model "${model}" \\
        --diameter ${cfg.cellpose_diameter} \\
        --flow_threshold ${cfg.cellpose_flow_threshold} \\
        --cellprob_threshold ${cfg.cellpose_cellprob_threshold} \\
        --min_size ${cfg.cellpose_min_size} \\
        ${cfg.cellpose_do_3d ? '--do_3d' : ''} \\
        ${save_mask_flag}

    echo ""
    echo "✓ Nuclei tracking complete for ${stage_name}"
    """
}

// ============================================================================
// PROCESS: Debug Preprocessing — Nuclei Comparison Report (CPU)
// ============================================================================

process DEBUG_PREPROCESS_REPORT {
    tag "t${String.format('%04d', timepoint)}"

    maxRetries 1
    errorStrategy 'terminate'

    publishDir "${params.output_dir}/debug_preprocessing/t${String.format('%04d', timepoint)}/report",
        mode: 'copy',
        pattern: "report/*"

    publishDir "${params.output_dir}/logs/debug_preprocessing",
        mode: 'copy',
        pattern: "*.log"

    container params.container

    input:
    tuple val(timepoint), path(json_files)
    path report_script

    output:
    path "report/*", emit: report
    path "debug_report_t${String.format('%04d', timepoint)}.log", emit: log

    script:
    def t_formatted = String.format('%04d', timepoint)
    """
    #!/bin/bash
    set -euo pipefail

    eval "\$(micromamba shell hook --shell bash)"
    micromamba activate microscopy_env

    exec > >(tee debug_report_t${t_formatted}.log) 2>&1

    echo "============================================"
    echo "Debug Preprocessing — Nuclei Report"
    echo "Timepoint: ${timepoint}"
    echo "JSON files: \$(ls -1 *.json | wc -l)"
    echo "============================================"

    python3 ${report_script} \\
        --json_dir . \\
        --outdir report/

    echo ""
    echo "✓ Report generated for timepoint ${timepoint}"
    """
}

// ============================================================================
// MAIN WORKFLOW
// ============================================================================

workflow {
    // ---- Simulation branch ----
    // Enabled ONLY via config.json's `simulation` block. CLI flags would
    // require nested JSON quoting through bash + sbatch + Nextflow, which is
    // too fragile to be reliable. The simulation block can live in either of
    // two locations (preprocessing.simulation is preferred, top-level is the
    // legacy location). Edit config.json to set:
    //
    //   "preprocessing": { ..., "simulation": { "enabled": true, "sweep_file": "..." } }
    //
    // and submit with just: sbatch submit_pipeline.sh config.json
    def sim_cfg = config.preprocessing?.simulation?.enabled
        ? config.preprocessing.simulation
        : (config.simulation?.enabled ? config.simulation : null)
    if (sim_cfg?.enabled && sim_cfg?.sweep_file) {
        def sweep_path_obj = file(sim_cfg.sweep_file)
        if (!sweep_path_obj.exists()) {
            error "simulation.sweep_file not found: ${sim_cfg.sweep_file}"
        }
        // Nextflow's file() returns a java.nio.Path — call toString() to get
        // the absolute path string.
        def sweep_path = sweep_path_obj.toAbsolutePath().toString()
        log.info "================================================"
        log.info "Simulation mode (one SLURM job per experiment × timepoint)"
        log.info "  sweep_file: ${sweep_path}"
        log.info "  input_dir : ${config.input?.directory}"
        log.info "  timepoints: ${config.input?.timepoints}"
        log.info "================================================"

        // Parse the sweep JSON. Read as bytes + decode UTF-8 explicitly to
        // avoid "Unable to decode string" errors when the default platform
        // encoding is not UTF-8 (e.g. on HPC login nodes where LANG=C).
        def sweep_bytes = sweep_path_obj.getBytes()
        def sweep_data = new groovy.json.JsonSlurper().parseText(
            new String(sweep_bytes, java.nio.charset.StandardCharsets.UTF_8)
        )
        def experiments = sweep_data.experiments.collect { exp ->
            [exp.name as String, exp.overrides as Map]
        }
        if (experiments.isEmpty()) {
            error "sweep must define at least one experiment"
        }

        // Read input files from config.input.directory (NOT from the sweep
        // file). This way users configure inputs in one place — config.json —
        // and the sweep only defines the experiments to run.
        def input_dir = file(config.input?.directory)
        if (!input_dir.exists()) {
            error "config.input.directory not found: ${config.input?.directory}"
        }
        // Match the same glob pattern the main pipeline uses: t*_Channel *.tif
        // (or t*_Channel <N>.tif when channel is explicit).
        def glob_pattern = config.channel > 0
            ? "t*_Channel ${config.channel}.tif"
            : "t*_Channel *.tif"
        def matcher = FileSystems.getDefault().getPathMatcher("glob:${glob_pattern}")
        def all_tp_tuples = []
        // input_dir IS already a java.nio.Path (Nextflow's file() returns Path).
        Files.list(input_dir).each { p ->
            if (matcher.matches(p.getFileName())) {
                def m = (p.getFileName().toString() =~ /(?i)t(\d+)/)
                def tp = m.find() ? m.group(1).toInteger() : 0
                all_tp_tuples << [tp, p.toFile()]
            }
        }
        all_tp_tuples = all_tp_tuples.sort { it[0] }
        if (all_tp_tuples.isEmpty()) {
            error "no per-timepoint TIFFs found in ${config.input?.directory} (pattern: ${glob_pattern})"
        }

        // Filter to config.input.timepoints (same filter the main pipeline uses).
        // Each entry can be a Number (numeric tp) or String (filename stem).
        // NOTE: locally renamed to `sim_timepoints` to avoid shadowing the
        // top-level `timepoints_selection` defined earlier in this file.
        def sim_timepoints = config.input?.timepoints
        def tp_tuples = all_tp_tuples
        if (sim_timepoints != null && (sim_timepoints as List).size() > 0) {
            def sel = (sim_timepoints as List).collect { it }
            tp_tuples = all_tp_tuples.findAll { tp, f ->
                sel.any { _matchesTimepoint(tp, f, it) }
            }
            if (tp_tuples.isEmpty()) {
                log.error "None of the requested timepoints (${sel}) matched any of the sweep's input files."
                log.error "Available timepoints: ${all_tp_tuples.collect { it[0] }}"
                error "config.input.timepoints did not match any sweep input"
            }
            log.info "Filtered to ${tp_tuples.size()} of ${all_tp_tuples.size()} timepoint(s) (config.input.timepoints)"
        }

        tp_tuples = tp_tuples.sort { it[0] }

        // Cartesian product: experiments × timepoints.
        def exp_ch = Channel.fromList(experiments)
        def tp_ch  = Channel.fromList(tp_tuples)

        // Expand into the (exp_name, overrides, tp, file) tuple the process expects.
        // Note: take a single positional arg (it) and index into it. Nextflow
        // invokes closures via MetaClass.invokeMethod which doesn't auto-spread
        // a List arg into multiple positional params — so destructuring must
        // happen INSIDE the closure body.
        def sim_ch = exp_ch.combine(tp_ch).map { tup ->
            def (exp, tp_file) = tup
            def (exp_name, overrides) = exp
            def (tp, f) = tp_file
            [exp_name, overrides, tp, f]
        }

        // Scaling factor for the output filename suffix. None of the current
        // 16 experiments override downscale_xy.factor in the sweep, so we
        // just use the base config's factor. The per-task Python script
        // computes the actual merged factor (base + overrides) for the
        // processing itself; this scaling_pct only needs to match what
        // run_pipeline writes to disk.
        def base_pp = config.preprocessing ?: [:]
        def base_factor = (base_pp.downscale_xy?.factor as BigDecimal)?.toDouble() ?: 1.0
        // Emit base_factor for every (exp, tp) pair — same scaling across
        // the whole sweep unless an experiment explicitly overrides
        // downscale_xy.factor (none do today).
        def scaling_ch = sim_ch.map { tup ->
            (int) round(base_factor * 100)
        }

        def exp_name_ch  = sim_ch.map { it[0] }
        // tp_file_ch must emit Nextflow tuples (not generic lists) so the
        // `tuple val(timepoint), path(image_file)` input of the process matches.
        def tp_file_ch   = sim_ch.map { tup -> tuple(tup[2], tup[3]) }
        def overrides_ch = sim_ch.map { groovy.json.JsonOutput.toJson(it[1] as Map) }

        // Stage the same preprocessing script + sweep file to every task.
        preproc_script_ch = Channel.fromPath(params.preprocessing_script, checkIfExists: true)
        sweep_file_ch     = Channel.fromPath(sweep_path, checkIfExists: true).collect()

        SIM_ONE_EXPERIMENT_TP(
            exp_name_ch,
            tp_file_ch,
            preproc_script_ch.collect(),
            sweep_file_ch,
            overrides_ch,
            scaling_ch,
        )

        // Aggregate after all per-task tasks complete. The aggregator reads
        // every metadata JSON and writes summary.csv / summary.md.
        // `tp_tuples.size()` is the count of timepoints that survived the
        // config.input.timepoints filter (Nextflow can serialise this
        // because tp_tuples is built at script-compile time).
        SIMULATION_AGGREGATE(
            preproc_script_ch.collect(),
            sweep_file_ch,
            SIM_ONE_EXPERIMENT_TP.out.meta.flatten().collect(),
            tp_tuples.size(),
        )

        return
    }

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

        if (timepoints_selection != null && timepoints_selection.size() > 0) {
            def sel_list = timepoints_selection
            log.info "Filtering split input to specific timepoints: ${sel_list} (input.timepoints)"
            def collected = input_channel.toList()
            def matched = collected.findAll { tp, f -> sel_list.any { _matchesTimepoint(tp, f, it) } }
            if (matched.isEmpty()) {
                log.error "None of the requested timepoints (${sel_list}) were found after splitting ${hyperstack_input.name}"
                log.error "Available timepoints: ${collected.collect { it[0] }}"
                exit 1
            }
            log.info "  matched ${matched.size()} of ${collected.size()} timepoint(s)"
            input_channel = Channel.fromList(matched)
        }

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

        if (timepoints_selection != null && timepoints_selection.size() > 0) {
            def sel_list = timepoints_selection
            log.info "Filtering input to specific timepoints: ${sel_list} (input.timepoints)"
            def before = file_tuples.size()
            file_tuples = file_tuples.findAll { tp, f ->
                sel_list.any { _matchesTimepoint(tp, f, it) }
            }
            if (file_tuples.isEmpty()) {
                log.error "None of the requested timepoints (${sel_list}) were found in ${params.input_dir}"
                log.error "Available timepoints: ${file_tuples.collect { it[0] }}"
                exit 1
            }
            // Re-sort the selected timepoints in numeric order so the
            // pipeline processes them consistently.
            file_tuples = file_tuples.sort { it[0] ?: 0 }
            log.info "  matched ${file_tuples.size()} of ${before} timepoint(s)"
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

            if (timepoints_selection != null && timepoints_selection.size() > 0) {
                def sel_list = timepoints_selection
                log.info "Filtering preprocessed input to specific timepoints: ${sel_list} (input.timepoints)"
                def before = preproc_tuples.size()
                preproc_tuples = preproc_tuples.findAll { tp, f -> sel_list.any { _matchesTimepoint(tp, f, it) } }
                if (preproc_tuples.isEmpty()) {
                    log.error "None of the requested timepoints (${sel_list}) were found in ${preprocessed_dir}"
                    exit 1
                }
                log.info "  matched ${preproc_tuples.size()} of ${before} timepoint(s)"
            }

            if (max_timepoints != null && preproc_tuples.size() > max_timepoints) {
                log.info "Limiting preprocessed input to first ${max_timepoints} of ${preproc_tuples.size()} timepoint(s) (input.max_timepoints)"
                preproc_tuples = preproc_tuples.take(max_timepoints)
            }

            segmentation_input = Channel
                .fromList(preproc_tuples.collect { tp, f -> tuple(tp, file(f.toPath())) })
                .ifEmpty { error "No preprocessed files with recognizable timepoint numbers found in: ${preprocessed_dir}" }
        } else {
            // No preprocessed_dir — use raw input images directly
            if (isotropic_reslice) {
                log.info "Preprocessing SKIPPED — applying lightweight isotropic Z reslicing only (preprocessing.isotropic_reslice=true)"
                RESLICE_ISOTROPIC(processing_input, shared_metadata)
                segmentation_input = RESLICE_ISOTROPIC.out.resliced
            } else {
                log.info "Preprocessing SKIPPED — using raw input images for segmentation (no isotropic reslicing)"
                segmentation_input = processing_input
            }
        }

    } else {
        // ---- Normal preprocessing ----

        if (preprocess_method == 'selfnet') {
            // Self-Net deblurring front-end. Stage the Self-Net script plus the
            // shared helper libs (spim_pipeline_fixed.py + WBNS.py) so all
            // imports resolve inside the task work dir.
            log.info "Preprocessing each timepoint with Self-Net deblurring"
            selfnet_script_ch = Channel.fromPath(params.selfnet_script, checkIfExists: true)
            shared_lib_ch = Channel.fromPath(params.preprocessing_script, checkIfExists: true)
            wbns_lib_ch = Channel.fromPath(params.wbns_script, checkIfExists: true)

            PREPROCESS_SELFNET(
                processing_input,
                shared_metadata,
                selfnet_script_ch.collect(),
                shared_lib_ch.collect(),
                wbns_lib_ch.collect(),
                config.preprocessing
            )

            preprocessed_processed = PREPROCESS_SELFNET.out.processed
            preprocessed_intermediates = PREPROCESS_SELFNET.out.intermediates
        } else {
            // Create channel for preprocessing script
            preproc_script_ch = Channel.fromPath(params.preprocessing_script, checkIfExists: true)

            // 2. Preprocess and deconvolve each timepoint
            PREPROCESS_DECONVOLVE(
                processing_input,
                shared_metadata,
                preproc_script_ch.collect(),
                config.preprocessing
            )

            preprocessed_processed = PREPROCESS_DECONVOLVE.out.processed
            preprocessed_intermediates = PREPROCESS_DECONVOLVE.out.intermediates
        }

        // 2b. OPTIONAL: Debug preprocessing — run Cellpose on each intermediate stage
        if (run_debug_preprocessing) {
        log.info "Debug preprocessing ENABLED — will run nuclei tracking on intermediates"
        log.info "  Cellpose model: ${config.preprocessing.debug_preprocessing.cellpose_model}"
        log.info "  save_intermediates: ${config.preprocessing.save_intermediates}"

        debug_nuclei_script_ch = Channel.fromPath(params.debug_nuclei_script, checkIfExists: true)
        debug_report_script_ch = Channel.fromPath(params.debug_report_script, checkIfExists: true)

        // The preprocessing process emits:
        //   intermediates: tuple(timepoint, path("intermediates/*.tif"))
        // flatMap fans out each intermediate TIF into a separate item so
        // Nextflow can schedule them in parallel.

        debug_paired_ch = preprocessed_intermediates
            .flatMap { timepoint, tifs ->
                if (tifs instanceof List) {
                    return tifs.collect { tif -> tuple(timepoint, tif.baseName, tif) }
                } else if (tifs != null) {
                    return [tuple(timepoint, tifs.baseName, tifs)]
                } else {
                    return []
                }
            }

        DEBUG_PREPROCESS_NUCLEI(
            debug_paired_ch,
            debug_nuclei_script_ch.collect(),
            config.preprocessing.debug_preprocessing
        )

        // Collect all JSONs per timepoint for the report
        debug_report_input = DEBUG_PREPROCESS_NUCLEI.out.metrics
            .groupTuple(by: 0)  // Group by timepoint → [timepoint, [json1, json2, ...]]

        DEBUG_PREPROCESS_REPORT(
            debug_report_input,
            debug_report_script_ch.collect()
        )
    } else {
        log.info "Debug preprocessing DISABLED"
    }

        // Use the preprocessing output as segmentation input
        segmentation_input = preprocessed_processed

    } // end skip_preprocessing else

    // 3. Segment each timepoint with Cellpose
    if (!skip_segmentation) {
        // When preprocessing is skipped, no XY scaling was applied — use 1.0
        def effective_scaling = skip_preprocessing ? 1.0 : config.preprocessing.image_scaling

        CELLPOSE_SEGMENT(
            segmentation_input,
            shared_metadata,
            config.segmentation,
            effective_scaling
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
            if (skip_preprocessing && skip_tracking) {
                // When preprocessing is skipped and no tracking, only merge segmented output
                log.info "Hyperstack merging enabled (segmented only, preprocessing was skipped)"

                segmented_merge_ch = CELLPOSE_SEGMENT.out.segmented
                    .map { timepoint, segmented_file -> segmented_file }
                    .collect()
                    .map { files -> tuple('segmented', files) }

                MERGE_TO_HYPERSTACK(
                    segmented_merge_ch,
                    shared_metadata,
                    merge_script_ch.collect()
                )
            } else {
                if (skip_preprocessing) {
                    log.info "Hyperstack merging enabled (input as processed + segmented, for tracking)"
                } else {
                    log.info "Hyperstack merging enabled (processed + segmented)"
                }

                // Collect processed and segmented files into labeled merge jobs
                processed_merge_ch = segmentation_input
                    .map { timepoint, f -> f }
                    .collect()
                    .map { files -> tuple('processed', files) }

                segmented_merge_ch = CELLPOSE_SEGMENT.out.segmented
                    .map { timepoint, segmented_file -> segmented_file }
                    .collect()
                    .map { files -> tuple('segmented', files) }

                // .collect() on each channel already waits for all items;
                // .mix() just combines the two ready merge jobs
                merge_jobs_ch = processed_merge_ch.mix(segmented_merge_ch)

                MERGE_TO_HYPERSTACK(
                    merge_jobs_ch,
                    shared_metadata,
                    merge_script_ch.collect()
                )
            }
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

            processed_merge_ch = segmentation_input
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
    def debug_status = (config.preprocessing?.debug_preprocessing?.enabled ?: false) ? "ENABLED" : "DISABLED"

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
    Debug preproc: ${debug_status}
    Output dir   : ${params.output_dir}

    Results:
      ${config.roi_cropping?.enabled ? "- Cropped images     : ${params.output_dir}/00_cropped/" : ""}
      - Preprocessed images : ${params.output_dir}/01_preprocessed/
      - Segmented masks     : ${params.output_dir}/02_segmented/
      ${ds_factor < 1.0 ? "- Downscaled labels  : ${params.output_dir}/02_segmented_downscaled/" : ""}
      ${!(config.output?.skip_merge ?: false) ? "- Hyperstacks        : in 01_preprocessed/ and 02_segmented/" : ""}
      ${(config.tracking?.enabled ?: false) ? "- Tracking results   : ${params.output_dir}/03_tracking/" : ""}
      ${(config.benchmark?.enabled ?: false) ? "- Benchmark          : ${params.output_dir}/benchmark/" : ""}
      ${(config.preprocessing?.debug_preprocessing?.enabled ?: false) ? "- Debug nuclei report: ${params.output_dir}/debug_preprocessing/" : ""}
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
