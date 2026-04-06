#!/usr/bin/env python3
"""
SPIM Image Preprocessing Pipeline
IMP Vienna - Andrés Gordo & Guilherme Ventura

Deconvolution and preprocessing for lightsheet microscopy data.
"""

import argparse
import psutil
import time
import os
import sys
import numpy as np
import tifffile
import scipy.ndimage as ndi
from skimage.transform import rescale, resize
import cv2
import pims
from WBNS import WBNS_image
import RedLionfishDeconv as rl
from tqdm import tqdm
from typing import Optional, Tuple
import subprocess


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
    max_scale=5.0,
    signal_floor_pct=10.0,
):
    """Correct intensity variation along Z axis.

    Args:
        max_scale: Maximum allowed correction factor per slice.  Prevents
            extreme amplification of noise in near-empty slices (e.g. the
            first/last slices of an embryo where only camera noise exists).
            Set to 0 to disable clamping.
        signal_floor_pct: Slices whose measured level is below this
            percentage of the target level are considered noise-dominated.
            Their correction is smoothly tapered toward 1.0 (no correction)
            so noise is not amplified.  0 disables tapering.
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

    # --- Signal-gated dampening ---
    # Slices whose level is far below the target are noise-dominated.
    # Taper their correction factor smoothly toward 1.0 (no correction)
    # so camera noise is not amplified into false detections.
    if signal_floor_pct > 0:
        floor = (signal_floor_pct / 100.0) * target
        dim_mask = levels_s < floor
        if np.any(dim_mask):
            # Linear taper: 0 at level=0 -> 1 at level=floor
            frac = np.where(dim_mask, levels_s / (floor + eps), 1.0)
            scales = np.where(dim_mask, 1.0 + (scales - 1.0) * frac, scales)
            n_dim = int(np.sum(dim_mask))
            print(
                f"    Z-correction: {n_dim}/{len(scales)} slices below signal floor "
                f"({signal_floor_pct}% of target) — correction dampened"
            )

    # Hard-clamp maximum correction factor
    if max_scale > 0:
        n_clamped = int(np.sum(scales > max_scale))
        if n_clamped > 0:
            print(
                f"    Z-correction: {n_clamped}/{len(scales)} slices clamped to max_scale={max_scale}"
            )
        scales = np.minimum(scales, max_scale)

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
        corrected = x * (norm / field)
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
    bg_threshold_pct=5.0,
    min_signal_pct=2.0,  # NEW: slices below this signal fraction → skip CLAHE
    dim_skip_mask=None,  # boolean array (length=Z): True → skip CLAHE on that slice
):
    from skimage import exposure

    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack.shape}")
    in_dtype = stack.dtype
    s = np.moveaxis(stack, axis, 0).astype(np.float32, copy=False)
    out = np.empty_like(s, dtype=np.float32)

    # Validate dim_skip_mask
    if dim_skip_mask is not None and len(dim_skip_mask) != s.shape[0]:
        print(
            f"    [Warning] dim_skip_mask length {len(dim_skip_mask)} != Z {s.shape[0]}, ignoring"
        )
        dim_skip_mask = None

    # Compute a global signal reference from the full stack
    # (median of per-slice medians, only on non-zero voxels)
    slice_medians = []
    for i in range(s.shape[0]):
        nz = s[i][s[i] > 0]
        if nz.size > 0:
            slice_medians.append(np.median(nz))
    global_median = np.median(slice_medians) if slice_medians else 1.0

    # Global percentile reference for pass-through normalisation.
    # Slices that skip CLAHE (dim-skip or signal-gate) are normalised to
    # [0, 1] on the tissue scale to avoid a scale mismatch that would
    # crush CLAHE'd tissue slices during final intensity normalisation.
    _ref_idx = list(range(s.shape[0]))
    if dim_skip_mask is not None and np.any(dim_skip_mask):
        _ref_idx = list(np.where(~dim_skip_mask)[0])
    if _ref_idx:
        _sample = _ref_idx[:: max(1, len(_ref_idx) // 20)]
        _ref_los = [float(np.percentile(s[j], p_low)) for j in _sample]
        _ref_his = [float(np.percentile(s[j], p_high)) for j in _sample]
        _global_lo = float(np.median(_ref_los))
        _global_hi = float(np.median(_ref_his))
    else:
        _global_lo, _global_hi = 0.0, 1.0

    skipped = 0
    for i in range(s.shape[0]):
        img = s[i]

        # Dim-slice skip: clip per-slice outliers, then normalise to [0,1]
        # on the tissue scale (no histogram equalisation).  The percentile
        # clip removes deconv hot-pixels that would otherwise map to full
        # brightness during the linear normalisation.
        if dim_skip_mask is not None and dim_skip_mask[i]:
            _sl_hi = np.percentile(img, p_high)
            img_c = np.minimum(img, _sl_hi)
            if _global_hi > _global_lo + eps:
                out[i] = np.clip(
                    (img_c - _global_lo) / (_global_hi - _global_lo), 0.0, 1.0
                )
            else:
                out[i] = 0.0
            skipped += 1
            continue

        # Signal gate: if this slice's median is < min_signal_pct% of the
        # global median, it's a noise-dominated slice — skip CLAHE entirely
        nz = img[img > 0]
        slice_signal = np.median(nz) if nz.size > 100 else 0.0
        if slice_signal < (global_median * min_signal_pct / 100.0):
            _sl_hi = np.percentile(img, p_high)
            img_c = np.minimum(img, _sl_hi)
            if _global_hi > _global_lo + eps:
                out[i] = np.clip(
                    (img_c - _global_lo) / (_global_hi - _global_lo), 0.0, 1.0
                )
            else:
                out[i] = 0.0
            skipped += 1
            continue

        lo = np.percentile(img, p_low)
        hi = np.percentile(img, p_high)
        if hi <= lo + eps:
            out[i] = 0.0
            continue

        if bg_threshold_pct > 0:
            bg_val = np.percentile(img, bg_threshold_pct)
            bg_mask = img <= bg_val
        else:
            bg_mask = None

        img01 = np.clip(img, lo, hi)
        img01 = (img01 - lo) / (hi - lo)
        result = exposure.equalize_adapthist(
            img01, kernel_size=kernel_size, clip_limit=clip_limit
        ).astype(np.float32, copy=False)
        np.nan_to_num(result, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        if bg_mask is not None:
            result[bg_mask] = img01[bg_mask]
        out[i] = result

    if skipped > 0:
        print(
            f"    CLAHE: skipped {skipped}/{s.shape[0]} low-signal slices (threshold: {min_signal_pct}% of global median)"
        )

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


def edge_taper_3d(img, width, skip_axes=()):
    """Apply a cosine (Tukey) taper to edges of a 3D volume.

    This smoothly fades the image intensity to zero over `width` pixels
    at every boundary. Applied BEFORE deconvolution to prevent
    Richardson-Lucy from amplifying boundary artifacts.

    Args:
        img: 3D numpy array (Z, Y, X)
        width: number of pixels over which to taper (0 = disabled)
        skip_axes: tuple of axis indices (0=Z, 1=Y, 2=X) to NOT taper.
            E.g. skip_axes=(0,) leaves Z untouched.
    Returns:
        Tapered image (same dtype as input if integer, else float32)
    """
    if width <= 0:
        return img
    in_dtype = img.dtype
    out = img.astype(np.float32, copy=True)
    z, y, x = out.shape

    # Build 1D half-cosine taper: 0 at edge -> 1 at width
    def _taper_1d(length, w):
        w = min(w, length // 2)
        t = np.ones(length, dtype=np.float32)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(w, dtype=np.float32) / w))
        t[:w] = ramp
        t[-w:] = ramp[::-1]
        return t

    if 0 not in skip_axes:
        tz = _taper_1d(z, width)
        out *= tz[:, None, None]
    if 1 not in skip_axes:
        ty = _taper_1d(y, width)
        out *= ty[None, :, None]
    if 2 not in skip_axes:
        tx = _taper_1d(x, width)
        out *= tx[None, None, :]

    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        out = np.clip(out, info.min, info.max).astype(in_dtype)
    return out


def remove_deconv_hot_pixels(img, size=5, threshold=5.0):
    """Remove isolated bright pixels created by RL deconvolution.

    For each 2D slice, pixels exceeding threshold × their local median
    (in a size×size neighbourhood) are replaced with the local median.
    This targets 1-2 px deconvolution spikes while preserving larger
    structures (nuclei are typically >=15 px diameter after downscaling).
    """
    from scipy.ndimage import median_filter

    out = img.astype(np.float32, copy=True)
    n_fixed = 0
    for zi in range(out.shape[0]):
        sl = out[zi]
        local_med = median_filter(sl, size=size)
        hot = sl > threshold * np.maximum(local_med, 1.0)
        n_hot = int(np.sum(hot))
        if n_hot > 0:
            sl[hot] = local_med[hot]
            n_fixed += n_hot
    print(
        f"    Hot-pixel filter: clipped {n_fixed} pixels "
        f"(window={size}, threshold={threshold}×)"
    )
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(out, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max).astype(
            img.dtype
        )
    return out


def image_postprocessing(img, resolution_px, resolution_pz, noise_lvl, sigma):
    """Apply background subtraction and Gaussian smoothing."""
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


def destripe_slice(img_2d, sigma_long=64, sigma_short=2):
    """Remove horizontal stripe artifacts from a single 2D image.

    Stripes in SPIM data run along the light-sheet propagation axis.
    For each row, the stripe component is estimated by heavy Gaussian
    smoothing along the stripe direction (sigma_long) and light smoothing
    perpendicular (sigma_short), then subtracted.

    Args:
        img_2d: 2D float32 array
        sigma_long: smoothing along stripe direction (columns). Larger = catches
            wider stripes but may remove real structure. 64 is a good start.
        sigma_short: smoothing perpendicular (rows). Keeps it a stripe estimate
            rather than a broad background. 1-3 is typical.
    Returns:
        Destriped 2D array (float32)
    """
    # Estimate stripe pattern: heavy blur along X (columns), tight along Y (rows)
    stripe_estimate = ndi.gaussian_filter(img_2d, sigma=(sigma_short, sigma_long))
    # The stripe is the row-wise mean of this smoothed field
    row_mean = np.mean(stripe_estimate, axis=1, keepdims=True)
    global_mean = np.mean(row_mean)
    # Stripe component = per-row deviation from the global mean
    stripe_component = row_mean - global_mean
    return img_2d - stripe_component


def destripe_3d(stack, axis=0, sigma_long=64, sigma_short=2):
    """Apply destriping to each slice of a 3D stack.

    Args:
        stack: 3D numpy array (Z, Y, X)
        axis: axis to iterate over for slicing (0=Z → destripe each XY plane)
        sigma_long: smoothing along stripe direction
        sigma_short: smoothing perpendicular to stripes
    Returns:
        Destriped stack (float32)
    """
    s = np.moveaxis(stack.astype(np.float32, copy=False), axis, 0)
    out = np.empty_like(s, dtype=np.float32)
    for i in range(s.shape[0]):
        out[i] = destripe_slice(s[i], sigma_long=sigma_long, sigma_short=sigma_short)
    out = np.moveaxis(out, 0, axis)
    # Clamp negatives from subtraction
    np.maximum(out, 0, out=out)
    return out


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


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="SPIM Image Preprocessing")

    # Paths
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input image"
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--psf_path", type=str, required=True, help="Path to PSF model")
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Save intermediate TIFFs after each major pipeline stage for debugging. "
        "Files are saved to <outdir>/intermediates/",
    )

    # Image Parameters
    parser.add_argument(
        "--image_scaling", type=float, default=1.0, help="Image scaling factor"
    )
    parser.add_argument(
        "--xy_pixel",
        type=float,
        default=0.0,
        help="Force XY pixel size (um). 0 to read from metadata",
    )
    parser.add_argument(
        "--z_pixel",
        type=float,
        default=0.0,
        help="Force Z pixel size (um). 0 to read from metadata",
    )

    # Processing Flags
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE")
    parser.add_argument(
        "--no_z_correction", action="store_true", help="Disable Z intensity correction"
    )
    parser.add_argument(
        "--no_shading", action="store_true", help="Disable Shading correction"
    )
    parser.add_argument(
        "--no_clahe_xy",
        action="store_true",
        help="Disable the second CLAHE pass on XY slices (keep only XZ pass)",
    )
    parser.add_argument(
        "--z_correction_method",
        type=str,
        default="p75",
        help="Robust statistic for z-intensity correction: 'median', 'p75', 'p95', etc. Lower targets boost dim slices more. Default: p75",
    )
    parser.add_argument(
        "--z_correction_max_scale",
        type=float,
        default=2.0,
        help="Maximum allowed Z-correction factor per slice. Prevents noise "
        "amplification in near-empty slices (e.g. embryo entry slices). 0=unlimited. Default: 2.0",
    )
    parser.add_argument(
        "--z_correction_signal_floor_pct",
        type=float,
        default=25.0,
        help="Slices whose signal level is below this %% of the target are "
        "considered noise-dominated; their correction is tapered toward 1.0 "
        "(no correction). Prevents noise amplification. 0=disabled. Default: 25.0",
    )
    parser.add_argument(
        "--camera_bg_percentile",
        type=float,
        default=2.0,
        help="Percentile used for per-slice camera background subtraction. "
        "Removes the sCMOS camera dark-current offset (~100 counts) before "
        "any processing so that background noise is not amplified by "
        "z-correction and deconvolution. 0=disabled. Default: 2.0",
    )
    parser.add_argument(
        "--dim_slice_threshold_pct",
        type=float,
        default=30.0,
        help="Slices whose std is below this %% of the stack-wide median std "
        "are attenuated (not zeroed) proportionally to their signal quality. "
        "Preserves real signal in dim entry slices while suppressing noise. "
        "0=disabled. Default: 30.0",
    )

    # Deconvolution Params
    parser.add_argument(
        "--padding", type=int, default=32, help="Padding for deconvolution"
    )
    parser.add_argument(
        "--niter", type=int, default=3, help="Iterations for 3D Deconvolution"
    )
    parser.add_argument(
        "--niterz", type=int, default=3, help="Iterations for 2D XZ Deconvolution"
    )

    # Normalization Params
    parser.add_argument(
        "--min_v", type=float, default=0, help="Min value for normalization"
    )
    parser.add_argument(
        "--max_v", type=float, default=65535, help="Max value for normalization"
    )
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=40,
        help="Low percentile for outlier removal",
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=99.99,
        help="High percentile for outlier removal",
    )

    # Background / Post-processing
    parser.add_argument(
        "--resolution_px0", type=float, default=10, help="BG Subtraction resolution"
    )
    parser.add_argument(
        "--resolution_pz0", type=float, default=10, help="BG Subtraction resolution Z"
    )
    parser.add_argument(
        "--noise_lvl", type=int, default=2, help="Noise level (MUST BE INTEGER)"
    )  # FIXED: Changed from float to int
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Gaussian smoothing sigma"
    )
    parser.add_argument(
        "--padding_mode",
        type=str,
        default="reflect",
        choices=["reflect", "edge", "constant", "symmetric", "wrap"],
        help="Padding mode for deconvolution (default: reflect)",
    )
    parser.add_argument(
        "--edge_mask_px",
        type=int,
        default=0,
        help="Zero out this many border pixels after deconvolution to remove edge artifacts (0=disabled)",
    )
    parser.add_argument(
        "--edge_taper_width",
        type=int,
        default=0,
        help="Cosine-taper this many border pixels BEFORE deconvolution to prevent edge ringing (0=disabled). Recommended: 16-32.",
    )
    parser.add_argument(
        "--clahe_clip_limit",
        type=float,
        default=0.01,
        help="CLAHE clip limit (lower = less contrast enhancement, less noise). Default: 0.01",
    )
    parser.add_argument(
        "--clahe_post_smooth",
        type=float,
        default=0.0,
        help="Gaussian sigma applied AFTER CLAHE to suppress high-frequency noise it introduces. 0=disabled. Recommended: 0.5-1.0",
    )
    parser.add_argument(
        "--mask_border_px",
        type=int,
        default=10,
        help="Trim tissue mask this many pixels from XY image borders to remove edge artifacts. 0=disabled. Default: 10",
    )
    parser.add_argument(
        "--no_destripe", action="store_true", help="Disable light-sheet stripe removal"
    )
    parser.add_argument(
        "--destripe_sigma_long",
        type=float,
        default=64,
        help="Destriping: smoothing along stripe direction (larger = catches wider stripes). Default: 64",
    )
    parser.add_argument(
        "--destripe_sigma_short",
        type=float,
        default=2,
        help="Destriping: smoothing perpendicular to stripes (1-3 typical). Default: 2",
    )
    parser.add_argument(
        "--clahe_min_signal_pct",
        type=float,
        default=15.0,
        help="Slices whose median signal is below this %% of the stack global median "
        "are skipped in CLAHE (pass-through). Prevents CLAHE from amplifying "
        "noise in entry/exit slices. Default: 15.0",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file does not exist: {args.input_file}")
        sys.exit(1)

    if args.niter > 0 or args.niterz > 0:
        if not os.path.isfile(args.psf_path):
            print(f"ERROR: PSF file does not exist: {args.psf_path}")
            sys.exit(1)

    # Log parameters for reproducibility
    print("\n" + "=" * 60)
    print("SPIM PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 60 + "\n")

    # Mapping boolean flags
    apply_clahe = not args.no_clahe
    apply_z_intensity_correction = not args.no_z_correction
    apply_shading_correct = not args.no_shading
    percentiles_source = (args.percentile_low, args.percentile_high)

    if not os.path.exists(args.outdir):
        try:
            os.makedirs(args.outdir)
        except FileExistsError:
            pass

    if args.xy_pixel > 0:
        tempScale = args.z_pixel / args.xy_pixel
    else:
        tempScale = 0

    # PSF Loading
    if args.niter > 0 or args.niterz > 0:
        t0 = time.time()
        print(f"Loading PSF from {args.psf_path}")
        psf = tifffile.imread(args.psf_path)
        psf_shape = psf.shape
        if args.image_scaling > 0 and args.image_scaling != 1.0:
            psf = rescale(
                psf,
                (args.image_scaling, args.image_scaling, args.image_scaling),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            )
            print(f"     -PSF dimension from : {psf_shape} to {psf.shape}")
        psf_f = psf.astype(np.float32)
        psf = psf_f / psf_f.sum()
        print(f"[Timer] PSF preparation took {time.time() - t0:.2f} seconds")

    # Processing Single Image
    image_path = args.input_file
    image_name = os.path.basename(image_path)

    print(f"\n[Processing] {image_name}")
    print_resource_usage()

    start_time_total = time.time()

    # --- Intermediate saving helper ---
    _inter_dir = None
    if args.save_intermediates:
        _inter_dir = os.path.join(args.outdir, "intermediates")
        os.makedirs(_inter_dir, exist_ok=True)
        print(f"[Info] Saving intermediates to {_inter_dir}")

    def _save_inter(tag, data):
        """Save a stage snapshot + print key statistics."""
        _flat = data.ravel().astype(np.float64)
        _nz = _flat[_flat > 0]
        _frac_zero = np.sum(_flat == 0) / _flat.size
        _p01 = np.percentile(_flat, 1) if _flat.size else 0
        _p50 = np.percentile(_flat, 50) if _flat.size else 0
        _p99 = np.percentile(_flat, 99) if _flat.size else 0
        _p999 = np.percentile(_flat, 99.9) if _flat.size else 0
        _nz_p10 = np.percentile(_nz, 10) if _nz.size > 100 else 0
        _nz_p99 = np.percentile(_nz, 99.9) if _nz.size > 100 else 0
        print(
            f"    [{tag}] shape={data.shape}, dtype={data.dtype}, "
            f"range=[{_flat.min():.1f}, {_flat.max():.1f}], "
            f"p1={_p01:.1f}, p50={_p50:.1f}, p99={_p99:.1f}, p99.9={_p999:.1f}, "
            f"zeros={_frac_zero:.1%}, nz_p10={_nz_p10:.1f}, nz_p99.9={_nz_p99:.1f}"
        )
        if _inter_dir is not None:
            _out = data
            if np.issubdtype(data.dtype, np.floating) and data.max() <= 1.0:
                _out = np.clip(data * 65535, 0, 65535).astype(np.uint16)
            elif np.issubdtype(data.dtype, np.floating):
                _out = np.clip(data, 0, 65535).astype(np.uint16)
            tifffile.imwrite(os.path.join(_inter_dir, f"{tag}.tif"), _out)

    # Load Image
    t0 = time.time()
    ext = os.path.splitext(image_name)[1].lower()

    try:
        if ext in [".tif", ".tiff"]:
            print("  Loading TIFF image...")
            img = tifffile.imread(image_path).astype(np.uint16)
            voxel_size = read_tiff_voxel_size(image_path)
        elif ext == ".nd2":
            print("  Loading ND2 image...")
            img = pims.open(image_path)
            voxel_size = read_nd2_voxel_size(img)
            img = np.array(img, dtype=np.uint16, copy=False)
        else:
            print(f"ERROR: Unsupported format: {ext}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        sys.exit(1)

    t1 = time.time()
    print(f"[Timer] Image loading took {t1 - t0:.2f} seconds")
    print(f"  - shape: {img.shape}, dtype: {img.dtype}")
    print(f"  - estimated size (GB): {img.nbytes / (1024**3):.3f}")
    print_resource_usage()

    physical_pixel_sizeX, physical_pixel_sizeY, physical_pixel_sizeZ = voxel_size

    if tempScale > 0:
        physical_pixel_sizeX = args.xy_pixel
        physical_pixel_sizeZ = args.z_pixel

    print(f"  - voxel sizes (um): {voxel_size}")

    # Image scaling (XY only, matching notebook)
    if args.image_scaling > 0 and args.image_scaling != 1.0:
        t0 = time.time()
        img_shape = img.shape
        print(f"  - image dimension : {img.shape}, scaling {args.image_scaling}")
        img = rescale(
            img,
            (1.0, args.image_scaling, args.image_scaling),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        )
        physical_pixel_sizeX /= args.image_scaling
        print(f"  - image dimension from : {img_shape} to {img.shape}")
        t1 = time.time()
        print(f"[Timer] Image rescaling took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Store original shape BEFORE reslicing (THIS IS THE CRITICAL FIX)
    img_shape = img.shape

    scale = physical_pixel_sizeX / physical_pixel_sizeZ

    # --- Camera background subtraction ---
    # sCMOS cameras have a per-pixel offset (~100-110 counts in raw data).
    # This DC floor dominates dim slices — when z-correction multiplies a
    # noise slice by 3-5×, the 100-count floor becomes 300-500 and then
    # deconvolution + CLAHE amplify it further into strong false signal.
    #
    # FIX: Use robust mode estimation (histogram peak) per slice instead of
    # a simple percentile. The mode captures the camera offset peak directly
    # and subtracts it fully. We also subtract an additional margin (2σ of
    # the noise) to push background pixels firmly to zero.
    _n_dim = 0
    _dim_mask_z = np.zeros(img.shape[0], dtype=bool)
    if args.camera_bg_percentile > 0:
        t0 = time.time()
        _bg_pct = args.camera_bg_percentile
        print(
            f"[Check-in] Camera background subtraction (mode-based, fallback p{_bg_pct:.0f})..."
        )
        img_f = img.astype(np.float32)
        _bg_values = []
        for _zi in range(img_f.shape[0]):
            _sl = img_f[_zi]
            # Estimate mode (peak of background distribution) via histogram
            _lo, _hi = float(np.percentile(_sl, 0.5)), float(np.percentile(_sl, 30))
            if _hi > _lo + 1:
                _bins = np.linspace(_lo, _hi, 100)
                _hist, _edges = np.histogram(_sl.ravel(), bins=_bins)
                _peak_idx = np.argmax(_hist)
                _mode = 0.5 * (_edges[_peak_idx] + _edges[_peak_idx + 1])
                # Estimate noise σ from the half-width of the mode peak
                _bg_pixels = _sl[_sl < _mode + (_hi - _lo) * 0.15]
                _bg_sigma = float(np.std(_bg_pixels)) if _bg_pixels.size > 100 else 0
                # Subtract mode + 1σ to push background firmly to zero
                _bg = _mode + _bg_sigma
            else:
                _bg = float(np.percentile(_sl, _bg_pct))
            _bg_values.append(_bg)
            img_f[_zi] = np.maximum(_sl - _bg, 0.0)
        print(
            f"    Background per slice: min={min(_bg_values):.1f}, "
            f"max={max(_bg_values):.1f}, median={np.median(_bg_values):.1f}"
        )
        # Convert back to original dtype
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            img = np.clip(img_f, 0, info.max).astype(img.dtype)
        else:
            img = img_f
        del img_f
        t1 = time.time()
        print(f"[Timer] Camera background subtraction took {t1 - t0:.2f} seconds")
        print_resource_usage()
        _save_inter("01_after_camera_bg", img)
    else:
        print("[Check-in] Camera background subtraction disabled")

    # --- Dim-slice soft attenuation ---
    # Instead of blanking (zeroing) noise-dominated slices (which loses real
    # signal like the embryo tip), scale them DOWN proportionally to their
    # signal quality. Slices with std close to 0 get scaled to ~0; slices
    # near the threshold keep most of their signal. This preserves the embryo
    # tip's bright nuclei in dim slices while suppressing background noise.
    _dim_threshold = args.dim_slice_threshold_pct
    if _dim_threshold > 0:
        t0 = time.time()
        print(f"[Check-in] Dim-slice soft attenuation (threshold={_dim_threshold}%)...")
        _raw_f32 = img.astype(np.float32)
        _slice_std = np.array(
            [float(np.std(_raw_f32[_zi])) for _zi in range(_raw_f32.shape[0])]
        )
        _global_std = float(np.median(_slice_std))
        _dim_floor_std = (_dim_threshold / 100.0) * _global_std

        # Per-slice attenuation factor: 0 at std=0, linearly ramps to 1 at
        # std=floor. Slices above the floor are untouched.
        _atten = np.clip(_slice_std / (_dim_floor_std + 1e-8), 0.0, 1.0)
        _n_attenuated = int(np.sum(_atten < 1.0))
        _dim_mask_z = _atten < 1.0  # track which slices were attenuated

        print(
            f"    Signal metric — std: global={_global_std:.1f}, floor={_dim_floor_std:.1f}"
        )
        if _n_attenuated > 0:
            print(
                f"    Soft attenuation: {_n_attenuated}/{len(_atten)} slices below floor"
            )
            for _zi in range(img.shape[0]):
                if _atten[_zi] < 1.0:
                    _factor = _atten[_zi]
                    print(
                        f"      slice {_zi}: std={_slice_std[_zi]:.1f}, attenuation={_factor:.3f}"
                    )
                    img[_zi] = (img[_zi].astype(np.float32) * _factor).astype(img.dtype)
        else:
            print(f"    No dim slices found (0/{len(_atten)} below floor)")
        _n_dim = _n_attenuated
        del _raw_f32
        t1 = time.time()
        print(f"[Timer] Dim-slice soft attenuation took {t1 - t0:.2f} seconds")
    else:
        _atten = np.ones(img.shape[0], dtype=np.float32)
        print("[Check-in] Dim-slice attenuation disabled (threshold=0%)")

    # Pre-processing
    if apply_shading_correct:
        t0 = time.time()
        print("[Check-in] Running shading_correct_xy_estimated...")
        img, field = shading_correct_xy_estimated(
            img, sigma_xy=96, z_axis=0, per_slice=False
        )
        t1 = time.time()
        print(f"[Timer] Shading correction took {t1 - t0:.2f} seconds")
        print_resource_usage()
        _save_inter("02_after_shading", img)

    if apply_z_intensity_correction:
        t0 = time.time()
        print(
            f"[Check-in] Running z_intensity_correction (method={args.z_correction_method}, "
            f"max_scale={args.z_correction_max_scale}, signal_floor={args.z_correction_signal_floor_pct}%)..."
        )
        img, scales = z_intensity_correction(
            img,
            z_axis=0,
            method=args.z_correction_method,
            smooth_window=11,
            max_scale=args.z_correction_max_scale,
            signal_floor_pct=args.z_correction_signal_floor_pct,
        )
        t1 = time.time()
        print(f"[Timer] Z-intensity correction took {t1 - t0:.2f} seconds")
        print_resource_usage()
        _save_inter("03_after_z_correction", img)

    # Isotropic Reslicing
    if abs(1.0 - scale) > 1e-4:
        t0 = time.time()
        print("[Check-in] Reslicing to isotropic...")
        img = reslice(img, "xy", physical_pixel_sizeX, physical_pixel_sizeZ)
        t1 = time.time()
        print(f"[Timer] Reslicing took {t1 - t0:.2f} seconds")

    img = img.astype(np.float32)
    new_img_shape = img.shape

    # Recalculate voxel size after reslicing (THIS NOW WORKS CORRECTLY)
    new_physical_pixel_sizeZ = img_shape[0] * physical_pixel_sizeZ / new_img_shape[0]
    print(
        f"  - image dimension from : {img_shape} to {new_img_shape} after isotropic interpolation"
    )
    print(f"  - z-space from : {physical_pixel_sizeZ} to {new_physical_pixel_sizeZ}")
    physical_pixel_sizeZ = new_physical_pixel_sizeZ
    print_resource_usage()

    # Pre-deconvolution destriping: remove light-sheet stripe artifacts BEFORE
    # deconvolution so RL doesn't amplify them. This is the primary destripe pass.
    if not args.no_destripe:
        t0 = time.time()
        print(
            f"[Check-in] Pre-deconv destriping (sigma_long={args.destripe_sigma_long}, sigma_short={args.destripe_sigma_short})..."
        )
        img = destripe_3d(
            img,
            axis=0,
            sigma_long=args.destripe_sigma_long,
            sigma_short=args.destripe_sigma_short,
        )
        t1 = time.time()
        print(f"[Timer] Pre-deconv destriping took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Build tissue mask from the RESLICED image — before deconv/WBNS/CLAHE
    # corrupt the tissue-vs-background contrast.  Applied at the very end
    # to zero-out background voxels where processing steps create noise.
    t0 = time.time()
    print("[Check-in] Computing tissue mask from resliced image...")
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_fill_holes

    # Map dim-slice mask to resliced Z indices for use by CLAHE.
    # Instead of re-attenuating (which over-suppresses), we pass a boolean
    # mask to CLAHE so it SKIPS those slices entirely.  This prevents CLAHE
    # from amplifying residual noise in dim slices while letting all other
    # processing (deconv, WBNS) run normally to preserve the embryo tip.
    _dim_skip_resliced = np.zeros(img.shape[0], dtype=bool)
    if _n_dim > 0 and img.shape[0] != len(_dim_mask_z):
        from scipy.ndimage import zoom

        _z_ratio = img.shape[0] / len(_dim_mask_z)
        _dim_skip_resliced = (
            zoom(_dim_mask_z.astype(np.float32), _z_ratio, order=0) > 0.5
        )
        _n_skip = int(np.sum(_dim_skip_resliced))
        print(
            f"    Dim-slice CLAHE skip mask: {_n_skip}/{img.shape[0]} slices will skip CLAHE"
        )
    elif _n_dim > 0:
        _dim_skip_resliced = _dim_mask_z.copy()
        print(
            f"    Dim-slice CLAHE skip mask: {_n_dim}/{img.shape[0]} slices will skip CLAHE"
        )

    # Smooth for mask computation (after re-blanking)
    _mask_img = ndi.gaussian_filter(img, sigma=2.0)

    # Per-slice adaptive Otsu: each Z-slice gets its own threshold so dim
    # slices at the top/bottom of the embryo aren't lost.
    tissue_mask = np.zeros(img.shape, dtype=bool)
    for _zi in range(_mask_img.shape[0]):
        _sl = _mask_img[_zi]
        _nonzero = _sl[_sl > 0]
        if _nonzero.size < 100:
            continue
        try:
            _otsu = threshold_otsu(_nonzero)
        except ValueError:
            continue
        # Very conservative: 3% of Otsu keeps dim nuclei at tissue edges
        _slice_mask = _sl > (_otsu * 0.03)
        # Fill holes within each slice (e.g. dark nuclei interior)
        _slice_mask = binary_fill_holes(_slice_mask)
        tissue_mask[_zi] = _slice_mask
    # 3D morphological closing to bridge small gaps between slices, then
    # generous dilation to ensure no tissue is clipped.
    struct = ndi.generate_binary_structure(3, 2)  # 18-connectivity
    tissue_mask = ndi.binary_closing(tissue_mask, structure=struct, iterations=5)
    tissue_mask = ndi.binary_dilation(tissue_mask, structure=struct, iterations=8)
    tissue_mask = binary_fill_holes(tissue_mask)

    # Trim mask from image XY borders: embryo is centred, edges are background.
    # Without this, dilation can push the mask to the image edges where
    # CLAHE/WBNS create artefacts.
    _bpx = args.mask_border_px
    if _bpx > 0:
        tissue_mask[:, :_bpx, :] = False
        tissue_mask[:, -_bpx:, :] = False
        tissue_mask[:, :, :_bpx] = False
        tissue_mask[:, :, -_bpx:] = False
        print(f"    Border trim: {_bpx}px excluded from XY edges")

    tissue_pct = 100.0 * np.count_nonzero(tissue_mask) / tissue_mask.size
    print(f"    Tissue mask: {tissue_pct:.1f}% of volume classified as tissue")
    del _mask_img
    t1 = time.time()
    print(f"[Timer] Tissue mask computation took {t1 - t0:.2f} seconds")

    # Recalculate resolution for BG subtraction
    resolution_px = int(args.resolution_px0 / new_physical_pixel_sizeZ)
    resolution_pz = int(args.resolution_pz0 / new_physical_pixel_sizeZ)
    print(f"  BG subtraction : {resolution_px},  {resolution_pz}")

    # Deconvolution (GPU)
    # Determine effective padding mode: when edge taper is active, the borders
    # are faded to ~0, so 'edge'/'reflect' padding would replicate near-zero
    # values and create a dark frame that RL rings against.  Use 'constant'
    # (zero-pad) instead so the padded region matches the tapered border.
    effective_padding_mode = args.padding_mode
    if args.edge_taper_width > 0 and args.padding_mode != "constant":
        print(
            f"  [Info] Edge taper active → overriding padding_mode '{args.padding_mode}' with 'constant'"
        )
        effective_padding_mode = "constant"

    # --- FIX: Do NOT stretch intensity to [0, 65535] before deconvolution ---
    # The old code called image_scaling_intens(img, 0, 65535) here, which
    # maps the camera noise floor (e.g. ~100 counts) to ~1300+ counts.
    # RL deconvolution then iteratively sharpens this amplified noise into
    # bright dots ("raindrops"). Instead, feed RL the natural float32 data.
    # RL works on relative intensity ratios; absolute scale doesn't matter.

    if args.niter > 0:
        t0 = time.time()
        print(
            "[Check-in] Running 3D deconvolution (natural float32, NO pre-stretch)..."
        )
        img = img.astype(np.float32, copy=False)
        # Pre-deconvolution edge taper: fade borders to zero so RL can't amplify them
        if args.edge_taper_width > 0:
            print(
                f"    Applying edge taper (width={args.edge_taper_width}px, skip Z) before 3D deconv..."
            )
            img = edge_taper_3d(img, args.edge_taper_width, skip_axes=(0,))
        img = np.pad(img, args.padding, mode=effective_padding_mode)
        imgSizeGB = img.nbytes / (1024**3)
        print(f"    -size(GB) : {imgSizeGB:.3f}")
        print_resource_usage()
        res_gpu = rl.doRLDeconvolutionFromNpArrays(
            img, psf, niter=args.niter, resAsUint8=False
        )
        img = res_gpu[
            args.padding : -args.padding,
            args.padding : -args.padding,
            args.padding : -args.padding,
        ]
        nan_count = np.count_nonzero(~np.isfinite(img))
        if nan_count > 0:
            print(
                f"    [Warning] 3D deconvolution produced {nan_count} NaN/Inf voxels — replacing with 0"
            )
            np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        t1 = time.time()
        print(f"[Timer] 3D deconvolution took {t1 - t0:.2f} seconds")
        print_resource_usage()
        _save_inter("04_after_deconv3d", img)

    if args.niterz > 0:
        t0 = time.time()
        print(
            "[Check-in] Running 2D (XZ) deconvolution (natural float32, NO pre-stretch)..."
        )
        img = img.astype(np.float32, copy=False)
        # Transpose to XZ view FIRST, then taper in the space RL will operate on
        img_xz = np.transpose(img, [1, 0, 2])
        psf_xz = np.transpose(psf, [1, 0, 2])
        if args.edge_taper_width > 0:
            # img_xz is (Y, Z, X) — skip Y (axis 0) and Z (axis 1), taper only X edges
            print(
                f"    Applying edge taper (width={args.edge_taper_width}px, X-only) before XZ deconv..."
            )
            img_xz = edge_taper_3d(img_xz, args.edge_taper_width, skip_axes=(0, 1))
        img_xz = np.pad(img_xz, args.padding, mode=effective_padding_mode)
        imgSizeGB = img_xz.nbytes / (1024**3)
        print(f"    img_xz -size(GB) : {imgSizeGB:.3f}")
        print_resource_usage()
        res_gpu = rl.doRLDeconvolutionFromNpArrays(
            img_xz, psf_xz, niter=args.niterz, resAsUint8=False
        )
        img_xz = res_gpu[
            args.padding : -args.padding,
            args.padding : -args.padding,
            args.padding : -args.padding,
        ]
        nan_count = np.count_nonzero(~np.isfinite(img_xz))
        if nan_count > 0:
            print(
                f"    [Warning] XZ deconvolution produced {nan_count} NaN/Inf voxels — replacing with 0"
            )
            np.nan_to_num(img_xz, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.transpose(img_xz, [1, 0, 2])
        t1 = time.time()
        print(f"[Timer] 2D (XZ) deconvolution took {t1 - t0:.2f} seconds")
        print_resource_usage()
        _save_inter("05_after_deconvXZ", img)

    # Edge masking: zero out border pixels to remove deconvolution edge artifacts
    # Only mask Y and X edges — embryo extends to Z boundaries
    if args.edge_mask_px > 0 and (args.niter > 0 or args.niterz > 0):
        b = args.edge_mask_px
        print(f"[Check-in] Applying edge mask: zeroing {b}px border (Y,X only)...")
        img[:, :b, :] = 0
        img[:, -b:, :] = 0
        img[:, :, :b] = 0
        img[:, :, -b:] = 0

    # Post-deconv border taper: deconvolution creates ringing at image borders
    # that extends inward. A smooth cosine fade kills this ringing gradually
    # so WBNS/CLAHE downstream don't see a hard edge to amplify.
    if args.niter > 0 or args.niterz > 0:
        _post_taper = max(args.mask_border_px, 40)
        print(f"[Check-in] Post-deconv border taper ({_post_taper}px on Y,X)...")
        img = edge_taper_3d(img, _post_taper, skip_axes=(0,))

    # Post-deconv hot-pixel removal: RL deconvolution iteratively sharpens
    # noise pixels into bright isolated dots ("raindrops").  A targeted
    # filter clips pixels that exceed 5× their local 5×5 median.
    if args.niter > 0 or args.niterz > 0:
        t0 = time.time()
        print("[Check-in] Post-deconv hot-pixel removal...")
        img = remove_deconv_hot_pixels(img, size=5, threshold=5.0)
        t1 = time.time()
        print(f"[Timer] Hot-pixel removal took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # --- Post-deconv noise floor subtraction ---
    # Even without the pre-deconv stretch, RL deconvolution raises the
    # baseline in low-signal regions. Estimate and subtract the residual
    # noise floor per slice so WBNS/CLAHE downstream see clean background.
    if args.niter > 0 or args.niterz > 0:
        t0 = time.time()
        print("[Check-in] Post-deconv noise floor subtraction...")
        img_f = img.astype(np.float32, copy=False)
        _floors = []
        for _zi in range(img_f.shape[0]):
            _sl = img_f[_zi]
            # Noise floor = mode of the bottom 25% of non-zero pixels
            _nonzero = _sl[_sl > 0]
            if _nonzero.size < 100:
                continue
            _p25 = float(np.percentile(_nonzero, 25))
            _bg_region = _nonzero[_nonzero <= _p25]
            if _bg_region.size < 50:
                continue
            _floor = float(np.median(_bg_region))
            _floor_sigma = float(np.std(_bg_region))
            # Subtract floor + 0.5σ — gentler than camera BG but catches
            # the deconv-raised baseline
            _sub = max(0.0, _floor + 0.5 * _floor_sigma)
            _floors.append(_sub)
            img_f[_zi] = np.maximum(_sl - _sub, 0.0)
        img = img_f
        if _floors:
            print(
                f"    Post-deconv floor: median={np.median(_floors):.1f}, "
                f"max={max(_floors):.1f}"
            )
        t1 = time.time()
        print(f"[Timer] Post-deconv noise floor subtraction took {t1 - t0:.2f} seconds")
        print_resource_usage()
        _save_inter("06_after_hotpix_floor", img)

    # Post-processing (WBNS + Gaussian smoothing)
    # DO NOT apply tissue mask before WBNS — a hard zero boundary causes
    # wavelet ringing inside the tissue that CLAHE then amplifies.
    # Let WBNS process the natural image with its gradual falloff.
    t0 = time.time()
    print("[Check-in] Running post-processing...")
    img = image_postprocessing(
        img, resolution_px, resolution_pz, args.noise_lvl, args.sigma
    )
    t1 = time.time()
    print(f"[Timer] Post-processing took {t1 - t0:.2f} seconds")
    print_resource_usage()
    _save_inter("07_after_wbns", img)

    # Destriping: second pass to catch any residual stripes after WBNS,
    # before CLAHE amplifies them.
    if not args.no_destripe:
        t0 = time.time()
        print(f"[Check-in] Post-WBNS destriping (residual pass)...")
        img = destripe_3d(
            img,
            axis=0,
            sigma_long=args.destripe_sigma_long,
            sigma_short=args.destripe_sigma_short,
        )
        t1 = time.time()
        print(f"[Timer] Post-WBNS destriping took {t1 - t0:.2f} seconds")
        print_resource_usage()

    if apply_clahe:
        t0 = time.time()
        # Pre-CLAHE tissue masking: zero out background so CLAHE can't
        # amplify sensor noise in tiles outside the embryo.  Without this,
        # ~70% of each slice is background where per-tile equalization
        # stretches residual noise into bright artifacts that bleed into
        # the tissue boundary through tile interpolation.
        _mz, _my, _mx = tissue_mask.shape
        if img.shape != tissue_mask.shape:
            print(
                f"    [Info] Pre-CLAHE crop: img {img.shape} → mask {tissue_mask.shape}"
            )
            img = img[:_mz, :_my, :_mx]
        print("[Check-in] Pre-CLAHE tissue masking (zeroing background)...")
        img[~tissue_mask] = 0
        _save_inter("08_pre_clahe_masked", img)

        # CLAHE on XY slices (each Z-plane equalized independently).
        # Depth normalization is already handled by z_intensity_correction.
        print(
            f"[Check-in] Applying CLAHE on XY slices (clip_limit={args.clahe_clip_limit})..."
        )
        img = clahe_3d_stack(
            img,
            clip_limit=args.clahe_clip_limit,
            kernel_size=(64, 64),
            axis=0,
            min_signal_pct=args.clahe_min_signal_pct,
            dim_skip_mask=_dim_skip_resliced,
        )
        t1 = time.time()
        print(f"[Timer] CLAHE (incl. pre-masking) took {t1 - t0:.2f} seconds")
        _save_inter("09_after_clahe", img)

        if args.clahe_post_smooth > 0:
            print(
                f"[Check-in] Post-CLAHE Gaussian smoothing (sigma={args.clahe_post_smooth})..."
            )
            img = ndi.gaussian_filter(img, sigma=args.clahe_post_smooth)
        print_resource_usage()

    # Final tissue mask application — safety pass to catch any residual
    # background signal created by CLAHE tile interpolation at boundaries.
    mz, my, mx = tissue_mask.shape
    if img.shape != tissue_mask.shape:
        print(
            f"    [Info] Shape mismatch: img {img.shape} vs mask {tissue_mask.shape} — cropping to match"
        )
        img = img[:mz, :my, :mx]
    print("[Check-in] Final tissue mask cleanup...")
    img[~tissue_mask] = 0
    del tissue_mask
    _save_inter("10_final_masked", img)

    # Normalization
    # FIX: compute percentiles on TISSUE-ONLY pixels (non-zero), not the
    # full image. After tissue masking, ~60% of the image is zero. Computing
    # percentiles on the full image gives p_low=0, making the low-end clip
    # completely ineffective. By using tissue-only pixels:
    #   - p_low clips the dim noise WITHIN the tissue (deconv residuals)
    #   - p_high clips the outlier bright voxels (prevents saturation)
    #   - The final stretch then maps [tissue_p_low, tissue_p_high] → [0, 65535]
    if percentiles_source[0] > 0 or percentiles_source[1] < 100:
        t0 = time.time()
        print("[Check-in] Removing outliers and normalizing intensities...")
        # Extract tissue-only pixels for percentile computation
        _tissue_pixels = img[img > 0]
        if _tissue_pixels.size > 0:
            low_thres = float(np.percentile(_tissue_pixels, percentiles_source[0]))
            high_thres = float(np.percentile(_tissue_pixels, percentiles_source[1]))
            print(
                f"    Tissue-only percentiles: p{percentiles_source[0]}={low_thres:.1f}, "
                f"p{percentiles_source[1]}={high_thres:.1f} "
                f"(from {_tissue_pixels.size} non-zero voxels, "
                f"{100 * _tissue_pixels.size / img.size:.1f}% of volume)"
            )
        else:
            low_thres, high_thres = getNormalizationThresholds(img, percentiles_source)
        del _tissue_pixels
        img = remove_outliers_image(img, low_thres, high_thres)
        _save_inter("11_after_percentile_clip", img)
        t1 = time.time()
        print(f"[Timer] Outlier removal and normalization took {t1 - t0:.2f} seconds")
        print_resource_usage()

    # Final Save
    t0 = time.time()
    print("[Check-in] Final intensity scaling and saving...")
    img = image_scaling_intens(img, args.min_v, args.max_v, True)
    img = img.astype(np.uint16)
    t1 = time.time()
    print(f"[Timer] Final scaling and conversion took {t1 - t0:.2f} seconds")

    # Save image with consistent naming (matching notebook)
    t0 = time.time()
    base_name = os.path.splitext(image_name)[0]
    image_out_name = f"{base_name}_{int(100 * args.image_scaling)}.tif"
    img_out_path = os.path.join(args.outdir, image_out_name)
    tifffile.imwrite(img_out_path, img)
    t1 = time.time()
    print(f"  Saved processed image to: {img_out_path}")
    print(f"[Timer] Saving image took {t1 - t0:.2f} seconds")

    # Validate output
    if not os.path.isfile(img_out_path):
        print(f"ERROR: Output file was not created: {img_out_path}")
        sys.exit(1)

    output_size = os.path.getsize(img_out_path)
    if output_size == 0:
        print(f"ERROR: Output file is empty: {img_out_path}")
        sys.exit(1)

    print(f"[Success] Output size: {output_size / (1024**2):.2f} MB")
    elapsed_time = time.time() - start_time_total
    print(f"[Done] Elapsed Time: {elapsed_time:.4f} seconds")
    print_resource_usage()


if __name__ == "__main__":
    main()
