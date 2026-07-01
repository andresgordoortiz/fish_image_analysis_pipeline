#!/usr/bin/env python3
"""
SPIM Image Preprocessing Pipeline — driver script
==================================================

Deconvolution and preprocessing for lightsheet microscopy data.

This is the script invoked by Nextflow's PREPROCESS_DECONVOLVE process.
It loads the input image, reads voxel sizes from metadata, runs the
modular preprocessing pipeline defined in ``spim_preprocessing_stages.py``,
and writes the result as a uint16 TIFF.

CLI
---
    python3 spim_pipeline_fixed.py \
        --input_file t0001.tif \
        --outdir . \
        --config_json preprocessing_config.json

``--config_json`` is the canonical entry point. ``--metadata_json`` is
optional and only used by the Nextflow wrapper (it overrides voxel sizes
in the config). For interactive parameter tuning, use ``--simulate`` with
a ``--sweep_file``.

For backwards compatibility the legacy ``--no_clahe`` / ``--no_z_correction``
/ ``--no_shading`` flags are still accepted and translated into config
overrides. They are deprecated and will be removed.

Helper functions in this file (image_scaling_intens, z_intensity_correction,
shading_correct_xy_estimated, clahe_3d_stack, reslice, image_postprocessing,
getNormalizationThresholds, remove_outliers_image, print_resource_usage,
read_tiff_voxel_size, read_nd2_voxel_size) are imported by
``spim_preprocessing_stages.py`` and ``spim_selfnet_preprocess.py`` and
should not be moved or renamed without updating both.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import tifffile
import scipy.ndimage as ndi
from skimage.transform import rescale, resize
import cv2
import pims
from WBNS import WBNS_image
from tqdm import tqdm
from typing import Optional, Tuple
import subprocess

# Stage pipeline (modular runner + simulation).
from spim_preprocessing_stages import (
    PIPELINE_STAGES,
    STAGE_NAMES,
    load_psf,
    run_pipeline,
    run_simulation,
    save_intermediate,
)


# --- INICIO DE FUNCIONES ORIGINALES (HELPERS REUSED BY STAGES + SELF-NET) ---

def image_scaling_intens(img, min_val, max_val, print_res=False):
    """Normalize image intensity to given range."""
    img_shape = img.shape
    img_type = img.dtype

    # Replace NaN/Inf with 0 to prevent downstream failures
    nan_count = np.count_nonzero(~np.isfinite(img))
    if nan_count > 0:
        print(f"    [Warning] Found {nan_count} NaN/Inf pixels — replacing with 0")
        img = np.where(np.isfinite(img), img, 0)

    img_min = np.amin(img)
    img_max = np.amax(img)

    if img_shape[0] < 300:
        img = np.reshape(img, newshape=-1)
        img = cv2.normalize(
            img,
            None,
            alpha=min_val,
            beta=max_val,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        img = np.reshape(img, newshape=img_shape)
    else:
        scale = img_max - img_min
        new_scale = max_val - min_val
        if scale == 0:
            # Constant image: assign min_val to avoid division by zero
            img = np.full(img_shape, min_val, dtype=np.float32)
        else:
            img = (new_scale * (img.astype(np.float32) - img_min) / scale) + min_val

    img = img.astype(img_type.name)

    if print_res == True:
        newimg_min = np.amin(img)
        newimg_max = np.amax(img)
        print(
            "     -Intensity Norm  from (%d , %d) to  (%d, %d) "
            % (img_min, img_max, newimg_min, newimg_max)
        )

    return img


def read_tiff_voxel_size(file_path):
    """Extract voxel size from TIFF metadata."""

    def _xy_voxel_size(tags, key):
        assert key in ["XResolution", "YResolution"]
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        return 1.0

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get("spacing", 1.0)
        else:
            z = 1.0

        tags = tiff.pages[0].tags
        y = _xy_voxel_size(tags, "YResolution")
        x = _xy_voxel_size(tags, "XResolution")

        return [x, y, z]


def read_nd2_voxel_size(image):
    """Extract voxel size from ND2 metadata."""
    md = image.metadata
    x = md["pixel_microns"]
    y = md["pixel_microns"]
    z = 3.0
    return [x, y, z]


def z_intensity_correction(
    stack,
    z_axis=0,
    method="p95",
    smooth_window=9,
    eps=1e-8,
    preserve_dtype=True,
):
    """Correct intensity variation along Z axis.

    Multiplicative Z-intensity correction: rescales each z-slice so a chosen
    robust statistic (e.g., p95) is constant along z. Good for depth
    attenuation / bleaching.

    Returns corrected_stack, scale_factors (len = Z)
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {stack.shape}")
    x = np.moveaxis(stack, z_axis, 0).astype(np.float32, copy=False)
    if method == "median":
        levels = np.median(x.reshape(x.shape[0], -1), axis=1)
    elif method.startswith("p"):
        q = float(method[1:])
        levels = np.percentile(x.reshape(x.shape[0], -1), q, axis=1)
    else:
        raise ValueError("method must be 'median' or 'pXX' like 'p95'")
    levels = np.maximum(levels, eps)
    if smooth_window is not None and smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        pad = smooth_window // 2
        lvl_pad = np.pad(levels, (pad, pad), mode="edge")
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        levels_s = np.convolve(lvl_pad, kernel, mode="valid")
    else:
        levels_s = levels
    target = np.median(levels_s)
    scales = target / levels_s

    y = x * scales[:, None, None]
    y = np.moveaxis(y, 0, z_axis)
    if not preserve_dtype:
        return y.astype(np.float32, copy=False), scales
    if np.issubdtype(stack.dtype, np.integer):
        info = np.iinfo(stack.dtype)
        y = np.clip(y, info.min, info.max).astype(stack.dtype)
    else:
        y = y.astype(stack.dtype, copy=False)
    return y, scales


def shading_correct_xy_estimated(
    stack, sigma_xy=64.0, z_axis=0, per_slice=False, eps=1e-6, preserve_dtype=True
):
    """Correct XY shading using estimated illumination profile."""
    if stack.ndim != 3:
        raise ValueError(
            f"Expected a 3D stack, got shape {stack.shape} (ndim={stack.ndim})."
        )
    in_dtype = stack.dtype
    x = np.moveaxis(stack.astype(np.float32, copy=False), z_axis, 0)
    if per_slice:
        corrected = np.empty_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            field_i = ndi.gaussian_filter(x[i], sigma=sigma_xy)
            field_i = np.maximum(field_i, eps)
            norm = float(np.mean(field_i))
            corrected[i] = x[i] * (norm / field_i)
        field = None
    else:
        proj = np.mean(x, axis=0)
        field = ndi.gaussian_filter(proj, sigma=sigma_xy)
        field = np.maximum(field, eps)
        norm = float(np.mean(field))
        # Clamp correction ratio to avoid amplifying noise in dark corners.
        # Without this, regions where the flat field is very low get divided
        # by tiny values → granular noise amplification.
        ratio = norm / field
        max_ratio = 2.0  # never amplify more than 2×
        ratio = np.minimum(ratio, max_ratio)
        corrected = x * ratio
    corrected = np.moveaxis(corrected, 0, z_axis)
    if not preserve_dtype:
        return corrected.astype(np.float32, copy=False), field
    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(in_dtype)
    else:
        corrected = corrected.astype(in_dtype, copy=False)
    return corrected, field


def clahe_3d_stack(
    stack,
    clip_limit=0.01,
    kernel_size=None,
    axis=0,
    preserve_dtype=True,
    p_low=0.5,
    p_high=99.5,
    eps=1e-8,
):
    """
    Slice-wise CLAHE for a 3D microscopy stack with robust normalization to [0,1].

    p_low/p_high: percentiles for intensity clipping per slice before scaling.
    """
    print("Applying clahe_3d_stack")
    from skimage import exposure

    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack.shape}")
    in_dtype = stack.dtype
    s = np.moveaxis(stack, axis, 0).astype(np.float32, copy=False)
    out = np.empty_like(s, dtype=np.float32)

    for i in range(s.shape[0]):
        img = s[i]

        lo = np.percentile(img, p_low)
        hi = np.percentile(img, p_high)
        if hi <= lo + eps:
            out[i] = 0.0
            continue

        img01 = np.clip(img, lo, hi)
        img01 = (img01 - lo) / (hi - lo)

        out[i] = exposure.equalize_adapthist(
            img01,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        ).astype(np.float32, copy=False)

    out = np.moveaxis(out, 0, axis)
    if not preserve_dtype:
        return out
    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        out = np.clip(out * info.max, 0, info.max).astype(in_dtype)
        return out
    return out.astype(in_dtype, copy=False)


def reslice(img, position, x_res, z_res):
    """Reslice image to isotropic voxels."""
    scale = z_res / x_res
    z, y, x = img.shape
    new_z = round(z * scale)
    img_max = np.amax(img).astype(np.float32)
    img_normalized = img.astype(np.float32) / img_max
    if position == "xz":
        reslice_img = np.transpose(img_normalized, [1, 0, 2])
        scale_img = np.zeros((y, new_z, x), dtype=np.float32)
        for i in range(y):
            scale_img[i] = resize(
                reslice_img[i], (new_z, x), order=3, anti_aliasing=True
            )
    elif position == "yz":
        reslice_img = np.transpose(img_normalized, [2, 0, 1])
        scale_img = np.zeros((x, new_z, y), dtype=np.float32)
        for i in range(x):
            scale_img[i] = resize(
                reslice_img[i], (new_z, y), order=3, anti_aliasing=True
            )
    elif position == "xy":
        reslice_img = np.transpose(img_normalized, [1, 0, 2])
        scale_img = np.zeros((y, new_z, x), dtype=np.float32)
        for i in range(y):
            scale_img[i] = resize(
                reslice_img[i], (new_z, x), order=3, anti_aliasing=True
            )
        scale_img = np.transpose(scale_img, [1, 0, 2])
    scale_img[scale_img < 0] = 0
    scale_img[scale_img > 1] = 1
    rescaled_img = (scale_img * img_max).astype(np.uint16)
    return rescaled_img


def image_postprocessing(img, resolution_px, resolution_pz, noise_lvl, sigma):
    """Apply background subtraction and Gaussian smoothing.

    Kept for backward compatibility with the Self-Net script, which calls
    it directly. The modular pipeline (``spim_preprocessing_stages.py``)
    uses ``stage_wbns`` and ``stage_gaussian`` instead — those respect
    ``enabled`` toggles in the config, while this legacy helper is gated by
    sentinel values (resolution_px > 0, sigma > 0).
    """
    steps = []
    if resolution_px > 0:
        steps.append("Remove Background/Noise")
    if resolution_pz > 0:
        steps.append("Remove Background/Noise z")
    if sigma > 0:
        steps.append("Gaussian Smoothing")
    pbar = tqdm(total=len(steps), desc="Postprocessing Image", unit="step")
    if resolution_px > 0:
        img = WBNS_image(img, resolution_px, noise_lvl)
        pbar.update(1)
    if resolution_pz > 0:
        img_xz = np.transpose(img, [1, 0, 2])
        img_xz = WBNS_image(img_xz, resolution_pz, 0)
        img = np.transpose(img_xz, [1, 0, 2])
        pbar.update(1)
    if sigma > 0:
        img = ndi.gaussian_filter(img, sigma)
        pbar.update(1)
    pbar.close()
    return img


def getNormalizationThresholds(img, percentiles):
    """Calculate intensity thresholds for normalization."""
    if np.ndim(img) > 1:
        img = img.flatten()
    low_thres = np.percentile(img, percentiles[0])
    high_thres = np.percentile(img, percentiles[1])
    return low_thres, high_thres


def remove_outliers_image(img, low_thres, high_thres, print_res=False):
    """Clip intensity outliers."""
    if print_res == True:
        img_min = np.amin(img)
        img_max = np.amax(img)
    img[img > high_thres] = high_thres
    img = img - low_thres
    img[img < 0] = 0
    if print_res == True:
        newimg_min = np.amin(img)
        newimg_max = np.amax(img)
        print(
            "Cropping Intensity from (%d , %d) to  (%d, %d) "
            % (img_min, img_max, newimg_min, newimg_max)
        )
    return img


def print_resource_usage():
    """Print current CPU, RAM, and GPU usage."""
    import psutil
    vm = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1)
    print(
        f"    [Resource] CPU: {cpu_pct:.1f}% | RAM: {vm.used / (1024**3):.2f} / {vm.total / (1024**3):.2f} GB ({vm.percent:.1f}%)"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split("\n")):
                util, mem_used, mem_total = line.split(",")
                print(
                    f"    [GPU {i}] Utilization: {util.strip()}% | Memory: {mem_used.strip()} / {mem_total.strip()} MB"
                )
    except Exception:
        pass


# --- FIN DE FUNCIONES ORIGINALES ---


# ---------------------------------------------------------------------------
# CLI / config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load a JSON config file. Strips UTF-8 BOM (Windows-edited files)."""
    with open(path) as f:
        raw = f.read()
    if raw.startswith("﻿"):
        raw = raw[1:]
    return json.loads(raw)


def _apply_legacy_flag_overrides(config_pp: dict, args: argparse.Namespace) -> dict:
    """Translate deprecated ``--no_*`` flags into per-stage ``enabled: false``
    overrides so old CLI invocations keep working while we transition."""
    if getattr(args, "no_clahe", False):
        config_pp.setdefault("clahe", {})["enabled"] = False
    if getattr(args, "no_z_correction", False):
        config_pp.setdefault("z_intensity_correction", {})["enabled"] = False
    if getattr(args, "no_shading", False):
        config_pp.setdefault("shading_correction", {})["enabled"] = False
    return config_pp


def _resolve_voxel_size(args, config_pp: dict, metadata: Optional[dict] = None):
    """Pick the voxel size to use, in priority order:

    1. ``--xy_pixel`` / ``--z_pixel`` CLI overrides (standalone use)
    2. ``metadata.json`` (Nextflow path)
    3. ``preprocessing.voxel_size`` from config (manual mode)
    4. ``voxel_size`` from config (top-level — the Nextflow pipeline
       writes voxel sizes there from auto_detect)
    5. Auto-detect from the TIFF / ND2 file

    Returns (x_um, y_um, z_um).
    """
    # CLI overrides
    if getattr(args, "xy_pixel", 0.0) and float(args.xy_pixel) > 0:
        x = float(args.xy_pixel)
        y = float(args.y_pixel or args.xy_pixel)
        z = float(args.z_pixel or 2.0)
        return (x, y, z)

    # metadata.json
    if metadata:
        x = metadata.get("x_resolution_um")
        y = metadata.get("y_resolution_um")
        z = (metadata.get("imagej") or {}).get("spacing")
        if x and y and z:
            return (float(x), float(y), float(z))

    # config.voxel_size (the location the Nextflow pipeline writes to)
    voxel_cfg = config_pp.get("voxel_size") or {}
    if voxel_cfg.get("x_um") and voxel_cfg.get("z_um"):
        return (
            float(voxel_cfg["x_um"]),
            float(voxel_cfg.get("y_um", voxel_cfg["x_um"])),
            float(voxel_cfg["z_um"]),
        )

    # fall back to file metadata
    ext = os.path.splitext(args.input_file)[1].lower()
    if ext in (".tif", ".tiff"):
        voxel = read_tiff_voxel_size(args.input_file)
    elif ext == ".nd2":
        img = pims.open(args.input_file)
        voxel = read_nd2_voxel_size(img)
    else:
        voxel = (0.347, 0.347, 2.0)
    return tuple(float(v) for v in voxel)


def _load_input_image(path: str) -> np.ndarray:
    """Load a TIFF or ND2 file as a uint16 numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path).astype(np.uint16)
    if ext == ".nd2":
        img = pims.open(path)
        return np.array(img, dtype=np.uint16, copy=False)
    raise ValueError(f"Unsupported input format: {ext}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPIM Image Preprocessing — modular pipeline driver.",
    )
    p.add_argument("--input_file", type=str,
                   help="Path to input image (not required for --simulate).")
    p.add_argument("--outdir", type=str,
                   help="Output directory (not required for --simulate).")
    p.add_argument("--metadata_json", type=str,
                   help="Optional metadata JSON (Nextflow wrapper uses this "
                        "to inject per-timepoint voxel sizes).")
    p.add_argument("--config_json", type=str,
                   help="Path to a JSON file with the full preprocessing "
                        "config (canonical entry point).")
    # Legacy flags (deprecated; translated into config overrides).
    p.add_argument("--xy_pixel", type=float, default=0.0,
                   help="[deprecated] Force XY pixel size (µm). Use "
                        "config.voxel_size instead.")
    p.add_argument("--z_pixel", type=float, default=0.0,
                   help="[deprecated] Force Z pixel size (µm).")
    p.add_argument("--no_clahe", action="store_true",
                   help="[deprecated] Disable CLAHE. Use config.clahe.enabled.")
    p.add_argument("--no_z_correction", action="store_true",
                   help="[deprecated] Disable Z-intensity correction.")
    p.add_argument("--no_shading", action="store_true",
                   help="[deprecated] Disable shading correction.")
    # Simulation mode
    p.add_argument("--simulate", action="store_true",
                   help="Run parameter sweep / Cellpose benchmark mode.")
    p.add_argument("--sweep_file", type=str,
                   help="Path to sweep JSON (required for --simulate).")
    return p


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = _build_argparser()
    args = parser.parse_args()

    # Simulation mode — separate code path.
    if args.simulate:
        if not args.sweep_file:
            parser.error("--simulate requires --sweep_file")
        print(f"[Simulation mode] Sweep file: {args.sweep_file}")
        run_simulation(args.sweep_file, log=print)
        return

    # Standard mode — require input/output/config.
    if not args.input_file:
        parser.error("--input_file is required (unless --simulate)")
    if not args.outdir:
        parser.error("--outdir is required (unless --simulate)")
    if not args.config_json:
        parser.error("--config_json is required (canonical)")

    os.makedirs(args.outdir, exist_ok=True)

    # Load config + apply legacy flag overrides for backward compatibility.
    config = _load_config(args.config_json)
    config_pp = (config.get("preprocessing") or {}).copy()
    _apply_legacy_flag_overrides(config_pp, args)

    # Backward-compat alias for downstream consumers (merge_hyperstack.py
    # reads ``config.preprocessing.image_scaling`` directly).
    config_pp["image_scaling"] = config_pp.get("downscale_xy", {}).get("factor", 1.0)

    # Resolve voxel size.
    metadata = None
    if args.metadata_json and os.path.isfile(args.metadata_json):
        with open(args.metadata_json) as f:
            metadata = json.load(f)
    voxel_size = _resolve_voxel_size(args, config_pp, metadata)
    print(f"Voxel size (µm): {voxel_size}")

    # Load image.
    print(f"\n[Processing] {os.path.basename(args.input_file)}")
    print_resource_usage()
    t0 = time.time()
    img = _load_input_image(args.input_file)
    print(f"  Loaded shape={img.shape} dtype={img.dtype} "
          f"size={img.nbytes/(1024**3):.3f} GB  ({time.time()-t0:.2f}s)")
    print_resource_usage()

    # Intermediates directory.
    save_ints = bool(config_pp.get("save_intermediates", False))
    intermediates_dir = None
    if save_ints:
        subdir = config_pp.get("intermediates_subdir", "intermediates")
        intermediates_dir = os.path.join(args.outdir, subdir)

    # Run the modular pipeline.
    print("\n=== Running modular preprocessing pipeline ===")
    print(f"  Method         : {config_pp.get('method', 'deconvolution')}")
    print(f"  Save ints      : {save_ints}")
    enabled_stages = []
    for stage_name, _fn, cfg_key in PIPELINE_STAGES[1:]:
        if cfg_key is None:
            enabled_stages.append(stage_name)
            continue
        sub_cfg = config_pp.get(cfg_key, {}) or {}
        is_enabled = sub_cfg.get("enabled", True) if cfg_key != "downscale_xy" else True
        if cfg_key == "downscale_xy":
            enabled_stages.append(f"{stage_name} (factor={config_pp.get('downscale_xy',{}).get('factor',1.0)})")
        elif is_enabled:
            enabled_stages.append(stage_name)
        else:
            enabled_stages.append(f"{stage_name} [DISABLED]")
    print("  Stages:")
    for s in enabled_stages:
        print(f"    - {s}")

    start_time = time.time()
    processed, ctx = run_pipeline(
        img,
        voxel_size,
        config_pp,
        psf=None,
        intermediates_dir=intermediates_dir,
        save_intermediates=save_ints,
    )
    elapsed = time.time() - start_time
    print(f"\n[Done] Preprocessing finished in {elapsed:.1f}s")
    print(f"  Final shape: {processed.shape}  dtype: {processed.dtype}")
    if "timings" in ctx:
        print("  Per-stage timings:")
        for stage_name, t in ctx["timings"].items():
            print(f"    {stage_name}: {t:.2f}s")

    # Save the final processed image with the legacy naming convention that
    # Nextflow expects: {base}_{int(100*factor)}.tif
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    factor = float(config_pp.get("downscale_xy", {}).get("factor", 1.0))
    scaling_pct = int(round(factor * 100))
    out_name = f"{base_name}_{scaling_pct}.tif"
    out_path = os.path.join(args.outdir, out_name)
    tifffile.imwrite(out_path, processed.astype(np.uint16))
    print(f"  Saved: {out_path}")
    print_resource_usage()


if __name__ == "__main__":
    main()