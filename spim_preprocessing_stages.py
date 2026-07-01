#!/usr/bin/env python3
"""
SPIM Preprocessing — Stage Pipeline
====================================

Modular stage-based runner for the SPIM preprocessing pipeline. Each
processing step (shading correction, Z-intensity correction, isotropic
reslicing, deconvolution, WBNS background subtraction, Gaussian smoothing,
CLAHE, percentile normalisation, final cast) is implemented as a small
function with the signature ``stage(stack, ctx) -> (stack, ctx)``.

The canonical stage order is defined by ``PIPELINE_STAGES``. Each stage has
an ``enabled`` boolean in the config that controls whether it runs. Stages
that are skipped also skip their intermediate-save.

This module is a thin layer over the existing helpers in
``spim_pipeline_fixed.py`` (which it imports at the top). The helpers
themselves are not reimplemented here.

Public entry points
-------------------
- ``run_pipeline(stack, voxel_size, psf, config, intermediates_dir=None, log=print)``
  Run the full pipeline once on a single ZYX stack.
- ``run_simulation(sweep_path, log=print)``
  Run a Cellpose-benchmarked parameter sweep over multiple preprocessing
  combinations on one or more timepoints.
- ``STAGE_NAMES`` — list of canonical stage name strings (used to build
  intermediate filenames).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Optional

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.transform import rescale

# Shared helpers (pure functions, no side effects on import). These live in
# spim_pipeline_fixed.py to keep their import paths stable for Self-Net
# (which imports from there too).
from spim_pipeline_fixed import (
    image_scaling_intens,
    z_intensity_correction,
    shading_correct_xy_estimated,
    clahe_3d_stack,
    reslice,
    image_postprocessing,
    getNormalizationThresholds,
    remove_outliers_image,
    print_resource_usage,
)


# ---------------------------------------------------------------------------
# Canonical stage names — used as intermediate-save filenames
# ---------------------------------------------------------------------------

STAGE_NAMES = [
    "01_after_load",
    "02_after_downscale_xy",
    "03_after_shading",
    "04_after_z_correction",
    "05_after_isotropic_reslice",
    "06_after_deconv3d",
    "07_after_deconv_xz",
    "08_after_wbns",
    "09_after_gaussian",
    "10_after_clahe",
    "11_after_percentile_norm",
    "12_after_final_cast",
]


# ---------------------------------------------------------------------------
# PSF loading (runs once before the stage loop)
# ---------------------------------------------------------------------------

def load_psf(ctx: dict) -> dict:
    """Load the PSF from disk, rescale it to match ``ctx['voxel_size']`` /
    ``downscale_xy.factor``, normalise to unit sum.

    Populates ``ctx['psf']`` (np.ndarray, float32, unit-sum). Returns the
    updated ctx. Skipped silently when ``deconvolution_3d.enabled=false``
    AND ``deconvolution_xz.enabled=false`` (no PSF needed).
    """
    cfg_pp = ctx["config"]
    dec3 = cfg_pp.get("deconvolution_3d", {}) or {}
    decx = cfg_pp.get("deconvolution_xz", {}) or {}
    if not (dec3.get("enabled", False) or decx.get("enabled", False)):
        return ctx

    psf_path = dec3.get("psf_path")
    if not psf_path or not os.path.isfile(psf_path):
        raise FileNotFoundError(
            f"deconvolution_3d.enabled=true but psf_path not found: {psf_path}"
        )

    factor = float(cfg_pp.get("downscale_xy", {}).get("factor", 1.0))
    psf = tifffile.imread(psf_path)
    if 0.0 < factor != 1.0:
        psf = rescale(psf, (factor, factor, factor), order=3,
                      preserve_range=True, anti_aliasing=True)
    psf = psf.astype(np.float32)
    psf = psf / psf.sum()
    ctx["psf"] = psf
    return ctx


# ---------------------------------------------------------------------------
# Stage functions — each takes (stack, ctx) and returns (stack, ctx)
# ---------------------------------------------------------------------------

def stage_load(stack, ctx):
    """01 — Load (passed in by the caller; this stage just records the
    initial voxel size and saves the intermediate if requested)."""
    return stack, ctx


def stage_downscale_xy(stack, ctx):
    """02 — XY downscale via skimage.rescale.

    Infrastructure step (no ``enabled`` toggle — the user always configures
    it via ``downscale_xy.factor``; ``factor=1.0`` is a no-op).
    """
    factor = float(ctx["config"].get("downscale_xy", {}).get("factor", 1.0))
    if factor > 0.0 and factor != 1.0:
        original_shape = stack.shape
        stack = rescale(
            stack.astype(np.float32),
            (1.0, factor, factor),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(stack.dtype, copy=False)
        # Update effective voxel size: XY doubles when factor halves.
        vx = list(ctx["voxel_size"])
        vx[0] = vx[0] / factor
        vx[1] = vx[1] / factor
        ctx["voxel_size"] = tuple(vx)
        ctx.setdefault("stage_notes", {})["downscale_xy"] = (
            f"{original_shape} -> {stack.shape}"
        )
    return stack, ctx


def stage_shading(stack, ctx):
    """03 — XY shading correction (gaussian-blurred mean projection)."""
    if not ctx["config"].get("shading_correction", {}).get("enabled", True):
        return stack, ctx
    cfg = ctx["config"]["shading_correction"]
    sigma_xy = float(cfg.get("sigma_xy", 96.0))
    per_slice = bool(cfg.get("per_slice", False))
    max_amp = cfg.get("max_amplification", 2.0)

    proj = np.mean(stack.astype(np.float32), axis=0)
    field = ndi.gaussian_filter(proj, sigma=sigma_xy)
    field = np.maximum(field, 1e-6)
    norm = float(np.mean(field))
    ratio = norm / field
    if max_amp is not None:
        ratio = np.minimum(ratio, float(max_amp))
    corrected = stack.astype(np.float32) * ratio[None, :, :]
    if np.issubdtype(stack.dtype, np.integer):
        info = np.iinfo(stack.dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(stack.dtype)
    return corrected.astype(stack.dtype, copy=False), ctx


def stage_z_correction(stack, ctx):
    """04 — Multiplicative Z-intensity correction (per-slice)."""
    if not ctx["config"].get("z_intensity_correction", {}).get("enabled", True):
        return stack, ctx
    cfg = ctx["config"]["z_intensity_correction"]
    stack, _scales = z_intensity_correction(
        stack,
        z_axis=0,
        method=cfg.get("method", "p95"),
        smooth_window=int(cfg.get("smooth_window", 11)),
    )
    return stack, ctx


def stage_isotropic_reslice(stack, ctx):
    """05 — Reslice Z so voxels are isotropic (matches XY pixel size)."""
    if not ctx["config"].get("isotropic_reslice", {}).get("enabled", True):
        return stack, ctx
    vx, vy, vz = ctx["voxel_size"]
    scale = vx / vz
    if abs(1.0 - scale) > 1e-4:
        original_shape = stack.shape
        stack = reslice(stack, "xy", vx, vz)
        # After reslice, Z spacing matches XY.
        new_vz = (original_shape[0] * vz) / stack.shape[0]
        ctx["voxel_size"] = (vx, vy, new_vz)
    return stack, ctx


def stage_deconv3d(stack, ctx):
    """06 — 3D Richardson-Lucy deconvolution (GPU)."""
    cfg = ctx["config"].get("deconvolution_3d", {}) or {}
    if not cfg.get("enabled", False):
        return stack, ctx
    niter = int(cfg.get("niter", 0))
    if niter <= 0:
        return stack, ctx
    import RedLionfishDeconv as rl  # imported lazily so non-GPU jobs don't pay the cost
    padding = int(cfg.get("padding", 32))
    fc = ctx["config"].get("final_cast", {})
    min_v = float(fc.get("min_v", 0))
    max_v = float(fc.get("max_v", 65535))
    stack = image_scaling_intens(stack, min_v, max_v, True)
    stack = np.pad(stack, padding, mode="reflect")
    out = rl.doRLDeconvolutionFromNpArrays(
        stack, ctx["psf"], niter=niter, resAsUint8=False
    )
    return out[padding:-padding, padding:-padding, padding:-padding], ctx


def stage_deconv_xz(stack, ctx):
    """07 — 2D (XZ) Richardson-Lucy deconvolution on transposed stack."""
    cfg = ctx["config"].get("deconvolution_xz", {}) or {}
    if not cfg.get("enabled", False):
        return stack, ctx
    niter = int(cfg.get("niter", 0))
    if niter <= 0:
        return stack, ctx
    import RedLionfishDeconv as rl
    padding = int(cfg.get("padding", 32))
    fc = ctx["config"].get("final_cast", {})
    min_v = float(fc.get("min_v", 0))
    max_v = float(fc.get("max_v", 65535))
    stack = image_scaling_intens(stack, min_v, max_v, True)
    img_xz = np.transpose(stack, [1, 0, 2])
    psf_xz = np.transpose(ctx["psf"], [1, 0, 2])
    img_xz = np.pad(img_xz, padding, mode="reflect")
    out = rl.doRLDeconvolutionFromNpArrays(
        img_xz, psf_xz, niter=niter, resAsUint8=False
    )
    img_xz = out[padding:-padding, padding:-padding, padding:-padding]
    return np.transpose(img_xz, [1, 0, 2]), ctx


def stage_wbns(stack, ctx):
    """08 — Wavelet background subtraction (XY then XZ)."""
    if not ctx["config"].get("background_subtraction", {}).get("enabled", True):
        return stack, ctx
    cfg = ctx["config"]["background_subtraction"]
    from WBNS import WBNS_image
    vx, vy, vz = ctx["voxel_size"]
    res_xy = int(float(cfg.get("resolution_xy", 10)) / vz)
    res_xz = int(float(cfg.get("resolution_xz", 10)) / vz)
    noise_lvl = int(cfg.get("noise_lvl", 2))
    if res_xy > 0:
        stack = WBNS_image(stack, res_xy, noise_lvl)
    if res_xz > 0:
        img_xz = np.transpose(stack, [1, 0, 2])
        img_xz = WBNS_image(img_xz, res_xz, 0)
        stack = np.transpose(img_xz, [1, 0, 2])
    return stack, ctx


def stage_gaussian(stack, ctx):
    """09 — Gaussian smoothing (final Gaussian of image_postprocessing is
    lifted into its own stage so it's individually toggleable)."""
    if not ctx["config"].get("gaussian_smooth", {}).get("enabled", True):
        return stack, ctx
    sigma = float(ctx["config"]["gaussian_smooth"].get("sigma", 1.2))
    if sigma > 0:
        stack = ndi.gaussian_filter(stack, sigma)
    return stack, ctx


def stage_clahe(stack, ctx):
    """10 — CLAHE on XZ-transposed stack (preserves behaviour of original
    pipeline)."""
    if not ctx["config"].get("clahe", {}).get("enabled", True):
        return stack, ctx
    cfg = ctx["config"]["clahe"]
    img_xz = np.transpose(stack, [1, 0, 2])
    img_xz = clahe_3d_stack(
        img_xz,
        clip_limit=float(cfg.get("clip_limit", 0.01)),
        kernel_size=(int(cfg.get("kernel_size", 64)),) * 2,
        axis=0,
        p_low=float(cfg.get("p_low", 0.5)),
        p_high=float(cfg.get("p_high", 99.5)),
    )
    return np.transpose(img_xz, [1, 0, 2]), ctx


def stage_percentile_norm(stack, ctx):
    """11 — Percentile-based outlier removal + rescale to [0, 1]."""
    if not ctx["config"].get("percentile_normalization", {}).get("enabled", True):
        return stack, ctx
    cfg = ctx["config"]["percentile_normalization"]
    p_low = float(cfg.get("percentile_low", 10.0))
    p_high = float(cfg.get("percentile_high", 99.9))
    if p_low > 0 or p_high < 100:
        lo, hi = getNormalizationThresholds(stack, (p_low, p_high))
        stack = remove_outliers_image(stack, lo, hi)
    return stack, ctx


def stage_final_cast(stack, ctx):
    """12 — Rescale to [min_v, max_v] and cast to the target dtype."""
    cfg = ctx["config"].get("final_cast", {}) or {}
    min_v = float(cfg.get("min_v", 0))
    max_v = float(cfg.get("max_v", 65535))
    dtype_name = cfg.get("dtype", "uint16")
    if dtype_name != "uint16":
        # Other dtypes are passed through (uint8 / float32). The common
        # path is uint16 for downstream Cellpose + BDV.
        return stack.astype(np.dtype(dtype_name), copy=False), ctx
    stack = image_scaling_intens(stack, min_v, max_v, True)
    return stack.astype(np.uint16), ctx


# ---------------------------------------------------------------------------
# Stage registry (execution order)
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    ("01_after_load",            stage_load,             None),                 # stage 01 is special — initial save happens before any work
    ("02_after_downscale_xy",    stage_downscale_xy,     "downscale_xy"),
    ("03_after_shading",         stage_shading,          "shading_correction"),
    ("04_after_z_correction",    stage_z_correction,     "z_intensity_correction"),
    ("05_after_isotropic_reslice", stage_isotropic_reslice, "isotropic_reslice"),
    ("06_after_deconv3d",        stage_deconv3d,         "deconvolution_3d"),
    ("07_after_deconv_xz",       stage_deconv_xz,        "deconvolution_xz"),
    ("08_after_wbns",            stage_wbns,             "background_subtraction"),
    ("09_after_gaussian",        stage_gaussian,         "gaussian_smooth"),
    ("10_after_clahe",           stage_clahe,            "clahe"),
    ("11_after_percentile_norm", stage_percentile_norm,  "percentile_normalization"),
    ("12_after_final_cast",      stage_final_cast,       "final_cast"),
]


# ---------------------------------------------------------------------------
# Intermediate I/O
# ---------------------------------------------------------------------------

def save_intermediate(stack: np.ndarray, intermediates_dir: str, stage_name: str) -> None:
    """Write a per-stage intermediate as ``<intermediates_dir>/<stage_name>.tif``.

    Casts to uint16 with clipping so downstream readers always get a
    well-defined dtype (matches the rest of the pipeline output).
    """
    os.makedirs(intermediates_dir, exist_ok=True)
    out_path = os.path.join(intermediates_dir, f"{stage_name}.tif")
    if stack.dtype != np.uint16:
        if np.issubdtype(stack.dtype, np.floating):
            stack = np.clip(stack, 0, 65535).astype(np.uint16)
        else:
            # Integer widening is safe; narrowing is not.
            if stack.dtype.itemsize <= 2:
                stack = stack.astype(np.uint16)
            else:
                stack = np.clip(stack, 0, 65535).astype(np.uint16)
    tifffile.imwrite(out_path, stack)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    stack: np.ndarray,
    voxel_size,
    config: dict,
    psf: Optional[np.ndarray] = None,
    intermediates_dir: Optional[str] = None,
    save_intermediates: bool = False,
    log: Callable[[str], None] = print,
    *,
    base_stack: Optional[np.ndarray] = None,
    base_voxel_size=None,
) -> tuple[np.ndarray, dict]:
    """Run the full preprocessing pipeline on a single ZYX stack.

    Parameters
    ----------
    stack : np.ndarray
        Input ZYX image (uint16 or float).
    voxel_size : (vx, vy, vz) in micrometres
        Voxel size of the input stack.
    config : dict
        The ``preprocessing`` sub-block of the config (with ``enabled``,
        ``method``, all per-stage sub-blocks, ``save_intermediates``,
        ``intermediates_subdir``).
    psf : np.ndarray, optional
        Pre-loaded PSF (float32, unit-sum). If ``None``, ``load_psf`` is
        called on the config. Pass a pre-loaded PSF when running the same
        config many times in simulation mode.
    intermediates_dir : str, optional
        Directory to write ``<stage_name>.tif`` snapshots. Defaults to
        ``<outdir>/intermediates`` when ``save_intermediates=true`` but
        ``intermediates_dir`` is None and ``outdir`` is in config.
    save_intermediates : bool
        Whether to write per-stage TIFFs.
    log : callable
        Function for progress output (default ``print``).
    base_stack, base_voxel_size
        Used by simulation mode to reset the stack before each experiment
        without re-loading the file from disk.

    Returns
    -------
    (out_stack, ctx)
        ``out_stack`` is the final preprocessed image (typically uint16).
        ``ctx`` carries per-stage timings and notes for diagnostics.
    """
    if base_stack is None:
        base_stack = stack
        base_voxel_size = voxel_size

    ctx = {
        "config": config,
        "voxel_size": tuple(voxel_size),
        "psf": psf,
        "intermediates_dir": intermediates_dir,
        "timings": {},
        "stage_notes": {},
    }

    # PSF: load if not passed in.
    if ctx["psf"] is None and (config.get("deconvolution_3d", {}) or {}).get("enabled", False):
        load_psf(ctx)

    save_ints = bool(save_intermediates or config.get("save_intermediates", False))
    if save_ints and not ctx["intermediates_dir"]:
        # Caller didn't pass one — write into the work dir if we can find it.
        # (Used by both Nextflow and standalone paths.)
        raise ValueError(
            "save_intermediates=true but no intermediates_dir provided. "
            "Pass it explicitly to run_pipeline()."
        )

    # Stage 01 — record the raw input. We do this BEFORE any processing so
    # downstream debugging can compare against the original.
    if save_ints:
        save_intermediate(base_stack, ctx["intermediates_dir"], "01_after_load")

    current = base_stack
    ctx["voxel_size"] = tuple(base_voxel_size)

    for stage_name, stage_fn, _cfg_key in PIPELINE_STAGES[1:]:
        t0 = time.time()
        current, ctx = stage_fn(current, ctx)
        ctx["timings"][stage_name] = time.time() - t0
        log(f"  [{stage_name}] {time.time() - t0:.2f}s  shape={current.shape} dtype={current.dtype}")
        if save_ints:
            save_intermediate(current, ctx["intermediates_dir"], stage_name)

    return current, ctx


# ---------------------------------------------------------------------------
# Simulation mode — preprocessing + Cellpose + nuclei-count benchmark
# ---------------------------------------------------------------------------

def _resolve_cellpose_config(sweep_cfg: dict, full_config: dict) -> dict:
    """Resolve Cellpose params for simulation mode.

    Priority (highest wins):
      1. ``sweep_cfg`` (per-sweep overrides in the sweep file)
      2. ``preprocessing.simulation`` block (simulation-specific overrides)
      3. ``segmentation`` block (the main Cellpose config — used by the
         Nextflow segmentation process). This is the canonical source: the
         simulation mode reuses the same model and params as the real
         segmentation so the nuclei counts are directly comparable.
    """
    sim = (full_config.get("preprocessing") or {}).get("simulation", {}) or {}
    seg = full_config.get("segmentation", {}) or {}

    def _pick(*candidates):
        for v in candidates:
            if v is not None:
                return v
        return None

    return {
        "model":              _pick(sweep_cfg.get("model"),        sim.get("cellpose_model"),        seg.get("model", "cyto3")),
        "diameter":           float(_pick(sweep_cfg.get("diameter"), sim.get("cellpose_diameter"), seg.get("diameter", 30))),
        "flow_threshold":     float(_pick(sweep_cfg.get("flow_threshold"),     sim.get("cellpose_flow_threshold"),     seg.get("flow_threshold", 0.8))),
        "cellprob_threshold": float(_pick(sweep_cfg.get("cellprob_threshold"), sim.get("cellpose_cellprob_threshold"), seg.get("cellprob_threshold", 0.0))),
        "do_3d":              bool(_pick(sweep_cfg.get("do_3d"),  sim.get("cellpose_do_3d"),  seg.get("do_3d", True))),
        "min_size":           int(_pick(sweep_cfg.get("min_size"), sim.get("cellpose_min_size"), seg.get("min_size", 15))),
        "use_gpu":            bool(_pick(sweep_cfg.get("use_gpu"), sim.get("cellpose_use_gpu"), seg.get("use_gpu", True))),
        "anisotropy":         _pick(sweep_cfg.get("anisotropy"), sim.get("cellpose_anisotropy"), seg.get("anisotropy")),
        "stitch_threshold":   _pick(sweep_cfg.get("stitch_threshold"), sim.get("cellpose_stitch_threshold"), seg.get("stitch_threshold")),
    }


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base``, returning a new dict."""
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_image(path: str):
    """Load a single TIFF/ND2 image and return (stack, voxel_size_or_None)."""
    import tifffile
    from spim_pipeline_fixed import read_tiff_voxel_size
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        img = tifffile.imread(path).astype(np.uint16)
        try:
            voxel = read_tiff_voxel_size(path)
        except Exception:
            voxel = (0.347, 0.347, 2.0)
        return img, voxel
    raise ValueError(f"Unsupported input format for simulation: {ext}")


def _run_cellpose(image: np.ndarray, cfg: dict, log: Callable[[str], None]) -> np.ndarray:
    """Run Cellpose on a 3D stack. Returns the label mask (uint16).

    Uses the same parameters as the main segmentation process (model,
    diameter, thresholds, anisotropy, stitch_threshold) so nuclei counts
    are directly comparable to what the full pipeline produces.
    """
    from cellpose import models
    model = models.CellposeModel(gpu=cfg["use_gpu"])
    log(f"  Cellpose: model={cfg['model']} diameter={cfg['diameter']} "
        f"do_3d={cfg['do_3d']} "
        f"anisotropy={cfg.get('anisotropy')} "
        f"stitch_threshold={cfg.get('stitch_threshold')}")
    eval_kwargs = dict(
        diameter=cfg["diameter"],
        flow_threshold=cfg["flow_threshold"],
        cellprob_threshold=cfg["cellprob_threshold"],
        do_3D=cfg["do_3d"],
        min_size=cfg["min_size"],
    )
    # Anisotropy (Z/Y ratio) is set only when explicitly provided — Cellpose
    # auto-derives it from voxel sizes otherwise.
    if cfg.get("anisotropy") is not None:
        eval_kwargs["anisotropy"] = float(cfg["anisotropy"])
    # Stitch threshold enables 2D+stitch mode (only meaningful when do_3d=False).
    if cfg.get("stitch_threshold") is not None and not cfg["do_3d"]:
        eval_kwargs["stitch_threshold"] = float(cfg["stitch_threshold"])
    mask, _flows, _styles = model.eval(image, **eval_kwargs)
    return mask.astype(np.uint16)


def _nuclei_count(mask: np.ndarray) -> int:
    """Count distinct labels (excluding 0 = background)."""
    return int(mask.max())


def _image_stats(image: np.ndarray) -> dict:
    """Quick intensity summary."""
    if image.dtype != np.float32 and image.dtype != np.float64:
        img = image.astype(np.float32)
    else:
        img = image
    return {
        "mean": float(np.mean(img)),
        "median": float(np.median(img)),
        "p99": float(np.percentile(img, 99)),
        "p01": float(np.percentile(img, 1)),
    }


def run_one_experiment_tp(
    exp_name: str,
    overrides: dict,
    raw_path: str,
    raw_stack: np.ndarray,
    voxel_size: tuple,
    cellpose_cfg: dict,
    pp_template: dict,
    output_dir: str,
    save_intermediates: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Run ONE experiment on ONE input timepoint.

    This is the per-task body used by both the local ``--simulate`` CLI
    orchestrator and the Nextflow ``SIM_ONE_EXPERIMENT_TP`` process. It
    deep-merges ``overrides`` into ``pp_template``, runs the modular
    preprocessing pipeline, then runs Cellpose on the result. Writes the
    preprocessed TIFF, the Cellpose mask, and a metadata JSON to
    ``<output_dir>/<exp_name>/``. Returns the metadata dict.

    Parameters
    ----------
    exp_name : str
        Experiment identifier (used as a subdirectory name).
    overrides : dict
        Per-experiment config overrides, deep-merged onto ``pp_template``.
    raw_path : str
        Path to the source TIFF (used to derive the output basename).
    raw_stack : np.ndarray
        Already-loaded ZYX image. The caller is responsible for loading
        it (in CLI mode) or passing the staged file path (in Nextflow mode).
    voxel_size : (vx, vy, vz) in micrometres
        Voxel size of ``raw_stack``.
    cellpose_cfg : dict
        Cellpose params (from ``_resolve_cellpose_config``).
    pp_template : dict
        The base ``preprocessing`` sub-block from the config.json.
    output_dir : str
        Top-level output directory (the sweep's ``output_dir``).
    save_intermediates : bool
        Whether to write per-stage TIFFs under
        ``<output_dir>/<exp_name>/intermediates_<tp_label>/``.
    log : callable
        Progress logger.
    """
    cfg_pp = _deep_merge(pp_template, overrides)
    cfg_pp["image_scaling"] = cfg_pp.get("downscale_xy", {}).get("factor", 1.0)

    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(raw_path))[0]
    scaling_pct = int(round(float(cfg_pp.get("downscale_xy", {}).get("factor", 1.0)) * 100))
    out_name = f"{base_name}_{scaling_pct}.tif"
    ints_dir = (
        os.path.join(exp_dir, f"intermediates_{base_name}") if save_intermediates else None
    )

    t0 = time.time()
    processed, ctx = run_pipeline(
        raw_stack,
        voxel_size,
        cfg_pp,
        psf=None,
        intermediates_dir=ints_dir,
        save_intermediates=save_intermediates,
        log=lambda msg: None,  # quieter per-stage log inside per-task
        base_stack=raw_stack,
        base_voxel_size=voxel_size,
    )
    runtime = time.time() - t0

    tifffile.imwrite(os.path.join(exp_dir, out_name), processed.astype(np.uint16))

    mask = _run_cellpose(processed, cellpose_cfg, log)
    mask_name = f"{base_name}_{scaling_pct}_mask.tif"
    tifffile.imwrite(os.path.join(exp_dir, mask_name), mask)
    nuclei = _nuclei_count(mask)
    stats = _image_stats(processed)

    meta = {
        "experiment": exp_name,
        "input_file": raw_path,
        "voxel_size_um": list(voxel_size),
        "config": cfg_pp,
        "cellpose_config": cellpose_cfg,
        "runtime_s": runtime,
        "stage_timings": ctx.get("timings", {}),
        "stage_notes": ctx.get("stage_notes", {}),
        "intensity_stats": stats,
        "nuclei_count": nuclei,
    }
    with open(os.path.join(exp_dir, f"{base_name}_{scaling_pct}_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log(
        f"  [{exp_name}] {os.path.basename(raw_path)} "
        f"runtime={runtime:.1f}s nuclei={nuclei} "
        f"mean_intensity={stats['mean']:.1f}"
    )
    return meta


def aggregate_simulation_metadata(
    metadata_paths,
    output_dir: str,
    inputs_count: int,
    log: Callable[[str], None] = print,
) -> dict:
    """Aggregate per-task metadata JSONs into summary.csv + summary.md.

    Reads each metadata JSON, groups by experiment, computes per-experiment
    mean/std over timepoints, and writes the two summary files into
    ``output_dir``. Returns the same dict shape as ``run_simulation``.

    Parameters
    ----------
    metadata_paths : iterable of str
        Paths to ``*_metadata.json`` files (typically one per
        ``(experiment, timepoint)`` pair). Order doesn't matter.
    output_dir : str
        Where to write ``summary.csv`` / ``summary.md``.
    inputs_count : int
        Number of input timepoints (used in the summary header).
    log : callable
        Progress logger.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group per-experiment rows.
    per_exp = {}
    for p in metadata_paths:
        with open(p) as f:
            meta = json.load(f)
        per_exp.setdefault(meta["experiment"], []).append(meta)

    summary_rows = []
    for exp_name, rows in per_exp.items():
        summary_rows.append({
            "experiment": exp_name,
            "n_tps": len(rows),
            "runtime_s_mean": float(np.mean([r["runtime_s"] for r in rows])),
            "mean_intensity": float(np.mean([r["intensity_stats"]["mean"] for r in rows])),
            "p99_intensity": float(np.mean([r["intensity_stats"]["p99"] for r in rows])),
            "nuclei_count_mean": float(np.mean([r["nuclei_count"] for r in rows])),
            "nuclei_count_std": float(np.std([r["nuclei_count"] for r in rows])) if len(rows) > 1 else 0.0,
        })

    summary_csv = os.path.join(output_dir, "summary.csv")
    summary_md = os.path.join(output_dir, "summary.md")

    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")

        median_nuclei = float(np.median([r["nuclei_count_mean"] for r in summary_rows]))
        with open(summary_md, "w") as f:
            f.write("# Simulation summary\n\n")
            f.write(f"Inputs: {inputs_count} timepoint(s)\n\n")
            f.write(
                f"{'experiment':<24} {'runtime_s':>10} {'mean_int':>10} "
                f"{'p99':>8} {'nuclei_mean':>14} {'nuclei_std':>12}\n"
            )
            f.write("-" * 90 + "\n")
            for row in summary_rows:
                rel = row["nuclei_count_mean"] - median_nuclei
                arrow = "↑" if rel > 0 else ("↓" if rel < 0 else "·")
                f.write(
                    f"{row['experiment']:<24} "
                    f"{row['runtime_s_mean']:>10.1f} "
                    f"{row['mean_intensity']:>10.1f} "
                    f"{row['p99_intensity']:>8.1f} "
                    f"{row['nuclei_count_mean']:>12.1f} {arrow}  "
                    f"{row['nuclei_count_std']:>12.1f}\n"
                )

        log(f"\n[Aggregation] Done. Summary written to:\n  {summary_csv}\n  {summary_md}")

    return {
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "per_experiment": {row["experiment"]: row for row in summary_rows},
    }


def run_simulation(sweep_path: str, log: Callable[[str], None] = print) -> dict:
    """Run a Cellpose-benchmarked preprocessing parameter sweep (local CLI).

    Loads all input images, then loops ``run_one_experiment_tp`` over the
    cartesian product of experiments × timepoints, then aggregates via
    ``aggregate_simulation_metadata``. This is the entry point for
    ``python3 spim_pipeline_fixed.py --simulate --sweep_file X.json``.

    For the parallelised cluster workflow, Nextflow calls
    ``run_one_experiment_tp`` and ``aggregate_simulation_metadata``
    directly — see ``SIM_ONE_EXPERIMENT_TP`` and ``SIMULATION_AGGREGATE``
    in ``spim_pipeline.nf``.
    """
    with open(sweep_path) as f:
        sweep = json.load(f)

    base_config_path = sweep["base_config"]
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(os.path.dirname(sweep_path), base_config_path)
    with open(base_config_path) as f:
        base_config = json.load(f)
    pp = base_config.get("preprocessing", {}) or {}

    inputs = sweep["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    output_dir = sweep["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cp_cfg = _resolve_cellpose_config(sweep.get("cellpose", {}) or {}, base_config)
    save_ints = bool(sweep.get("save_intermediates", False))

    # Load each input once and keep the raw stack + voxel sizes in memory.
    log(f"[Simulation] Loading {len(inputs)} input image(s)...")
    raw_stacks = []
    for path in inputs:
        stack, voxel = _load_image(path)
        raw_stacks.append({"path": path, "stack": stack, "voxel_size": voxel})
        log(f"  Loaded {os.path.basename(path)}: shape={stack.shape} voxel={voxel}")

    all_metadata_paths = []
    for exp in sweep["experiments"]:
        exp_name = exp["name"]
        overrides = exp.get("overrides", {}) or {}
        log(f"\n[Simulation] Experiment: {exp_name}")
        log(f"  Output: {os.path.join(output_dir, exp_name)}")

        for raw in raw_stacks:
            meta = run_one_experiment_tp(
                exp_name=exp_name,
                overrides=overrides,
                raw_path=raw["path"],
                raw_stack=raw["stack"],
                voxel_size=raw["voxel_size"],
                cellpose_cfg=cp_cfg,
                pp_template=pp,
                output_dir=output_dir,
                save_intermediates=save_ints,
                log=log,
            )
            tp_label = os.path.splitext(os.path.basename(raw["path"]))[0]
            scaling_pct = int(round(
                float(meta["config"].get("downscale_xy", {}).get("factor", 1.0)) * 100
            ))
            all_metadata_paths.append(
                os.path.join(output_dir, exp_name, f"{tp_label}_{scaling_pct}_metadata.json")
            )

    return aggregate_simulation_metadata(
        all_metadata_paths, output_dir, inputs_count=len(inputs), log=log
    )