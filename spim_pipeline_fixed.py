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
    stack, z_axis=0, method="p95", smooth_window=9, eps=1e-8, preserve_dtype=True,
    max_scale=5.0, signal_floor_pct=10.0,
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
            print(f"    Z-correction: {n_dim}/{len(scales)} slices below signal floor "
                  f"({signal_floor_pct}% of target) — correction dampened")

    # Hard-clamp maximum correction factor
    if max_scale > 0:
        n_clamped = int(np.sum(scales > max_scale))
        if n_clamped > 0:
            print(f"    Z-correction: {n_clamped}/{len(scales)} slices clamped to max_scale={max_scale}")
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
    min_signal_pct=2.0,   # NEW: slices below this signal fraction → skip CLAHE
):
    from skimage import exposure

    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack.shape}")
    in_dtype = stack.dtype
    s = np.moveaxis(stack, axis, 0).astype(np.float32, copy=False)
    out = np.empty_like(s, dtype=np.float32)

    # Compute a global signal reference from the full stack
    # (median of per-slice medians, only on non-zero voxels)
    slice_medians = []
    for i in range(s.shape[0]):
        nz = s[i][s[i] > 0]
        if nz.size > 0:
            slice_medians.append(np.median(nz))
    global_median = np.median(slice_medians) if slice_medians else 1.0

    skipped = 0
    for i in range(s.shape[0]):
        img = s[i]

        # Signal gate: if this slice's median is < min_signal_pct% of the
        # global median, it's a noise-dominated slice — skip CLAHE entirely
        nz = img[img > 0]
        slice_signal = np.median(nz) if nz.size > 100 else 0.0
        if slice_signal < (global_median * min_signal_pct / 100.0):
            out[i] = img  # pass-through unchanged
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
        print(f"    CLAHE: skipped {skipped}/{s.shape[0]} low-signal slices (threshold: {min_signal_pct}% of global median)")

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
        "--no_clahe_xy", action="store_true",
        help="Disable the second CLAHE pass on XY slices (keep only XZ pass)"
    )
    parser.add_argument(
        "--z_correction_method", type=str, default="p75",
        help="Robust statistic for z-intensity correction: 'median', 'p75', 'p95', etc. Lower targets boost dim slices more. Default: p75"
    )
    parser.add_argument(
        "--z_correction_max_scale", type=float, default=5.0,
        help="Maximum allowed Z-correction factor per slice. Prevents noise "
        "amplification in near-empty slices (e.g. embryo entry slices). 0=unlimited. Default: 5.0"
    )
    parser.add_argument(
        "--z_correction_signal_floor_pct", type=float, default=10.0,
        help="Slices whose signal level is below this %% of the target are "
        "considered noise-dominated; their correction is tapered toward 1.0 "
        "(no correction). Prevents noise amplification. 0=disabled. Default: 10.0"
    )
    parser.add_argument(
        "--dim_slice_threshold_pct", type=float, default=30.0,
        help="Slices whose p95 AND std are below this %% of the stack-wide median "
        "are considered noise-dominated and zeroed BEFORE any corrections. "
        "Critical for SPIM data where the first/last slices have no tissue. "
        "Higher = more aggressive blanking. 0=disabled. Default: 30.0"
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
        "--padding_mode", type=str, default="reflect",
        choices=["reflect", "edge", "constant", "symmetric", "wrap"],
        help="Padding mode for deconvolution (default: reflect)"
    )
    parser.add_argument(
        "--edge_mask_px", type=int, default=0,
        help="Zero out this many border pixels after deconvolution to remove edge artifacts (0=disabled)"
    )
    parser.add_argument(
        "--edge_taper_width", type=int, default=0,
        help="Cosine-taper this many border pixels BEFORE deconvolution to prevent edge ringing (0=disabled). Recommended: 16-32."
    )
    parser.add_argument(
        "--clahe_clip_limit", type=float, default=0.01,
        help="CLAHE clip limit (lower = less contrast enhancement, less noise). Default: 0.01"
    )
    parser.add_argument(
        "--clahe_post_smooth", type=float, default=0.0,
        help="Gaussian sigma applied AFTER CLAHE to suppress high-frequency noise it introduces. 0=disabled. Recommended: 0.5-1.0"
    )
    parser.add_argument(
        "--mask_border_px", type=int, default=10,
        help="Trim tissue mask this many pixels from XY image borders to remove edge artifacts. 0=disabled. Default: 10"
    )
    parser.add_argument(
        "--no_destripe", action="store_true",
        help="Disable light-sheet stripe removal"
    )
    parser.add_argument(
        "--destripe_sigma_long", type=float, default=64,
        help="Destriping: smoothing along stripe direction (larger = catches wider stripes). Default: 64"
    )
    parser.add_argument(
        "--destripe_sigma_short", type=float, default=2,
        help="Destriping: smoothing perpendicular to stripes (1-3 typical). Default: 2"
    )
    parser.add_argument(
        "--clahe_min_signal_pct", type=float, default=2.0,
        help="Slices whose median signal is below this %% of the stack global median "
        "are skipped in CLAHE (pass-through). Prevents CLAHE from amplifying "
        "noise in entry/exit slices. Default: 2.0"
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

    # --- Early dim-slice blanking ---
    # Detect noise-dominated slices on the RAW data, BEFORE any corrections.
    # This is critical: z_intensity_correction boosts noise slices toward the
    # signal floor, which would make them undetectable later.  By blanking
    # here we prevent all downstream steps from amplifying camera noise.
    #
    # Detection strategy: in sparse fluorescence data (labelled nuclei in SPIM),
    # the MEDIAN of all slices is dominated by camera background (~100 counts)
    # and is nearly identical between noise-only and tissue-containing slices.
    # The real discriminator is the SPREAD of intensity: tissue slices have
    # bright nuclei that push std and high percentiles far above the background,
    # while pure-noise slices have std ≈ 20-30.  We use p95 as the signal
    # metric because it captures sparse bright objects without being skewed
    # by a single hot pixel.
    _dim_threshold = args.dim_slice_threshold_pct
    _n_dim = 0
    _dim_mask_z = np.zeros(img.shape[0], dtype=bool)
    if _dim_threshold > 0:
        t0 = time.time()
        print(f"[Check-in] Early dim-slice detection (threshold={_dim_threshold}%)...")
        _raw_f32 = img.astype(np.float32)
        _slice_p95 = []
        _slice_std = []
        for _zi in range(_raw_f32.shape[0]):
            _sl = _raw_f32[_zi]
            _slice_p95.append(float(np.percentile(_sl, 95)))
            _slice_std.append(float(np.std(_sl)))
        _slice_p95 = np.array(_slice_p95)
        _slice_std = np.array(_slice_std)

        _global_p95 = float(np.median(_slice_p95))
        _global_std = float(np.median(_slice_std))
        _dim_floor_p95 = (_dim_threshold / 100.0) * _global_p95
        _dim_floor_std = (_dim_threshold / 100.0) * _global_std

        # A slice is noise-dominated if BOTH p95 and std are below their floors.
        _dim_mask_z = (_slice_p95 < _dim_floor_p95) & (_slice_std < _dim_floor_std)
        _n_dim = int(np.sum(_dim_mask_z))

        print(f"    Signal metrics — p95: global={_global_p95:.1f}, floor={_dim_floor_p95:.1f} | "
              f"std: global={_global_std:.1f}, floor={_dim_floor_std:.1f}")
        if _n_dim > 0:
            print(f"    Dim-slice blanking: {_n_dim}/{len(_dim_mask_z)} slices detected as noise-dominated — zeroing")
            for _zi in range(img.shape[0]):
                if _dim_mask_z[_zi]:
                    img[_zi] = 0
        else:
            print(f"    No dim slices detected (0/{len(_dim_mask_z)} below both thresholds)")
        del _raw_f32
        t1 = time.time()
        print(f"[Timer] Early dim-slice detection took {t1 - t0:.2f} seconds")
    else:
        print("[Check-in] Dim-slice blanking disabled (threshold=0%)")

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

    if apply_z_intensity_correction:
        t0 = time.time()
        print(f"[Check-in] Running z_intensity_correction (method={args.z_correction_method}, "
              f"max_scale={args.z_correction_max_scale}, signal_floor={args.z_correction_signal_floor_pct}%)...")
        img, scales = z_intensity_correction(
            img, z_axis=0, method=args.z_correction_method, smooth_window=11,
            max_scale=args.z_correction_max_scale,
            signal_floor_pct=args.z_correction_signal_floor_pct,
        )
        t1 = time.time()
        print(f"[Timer] Z-intensity correction took {t1 - t0:.2f} seconds")
        print_resource_usage()

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
        print(f"[Check-in] Pre-deconv destriping (sigma_long={args.destripe_sigma_long}, sigma_short={args.destripe_sigma_short})...")
        img = destripe_3d(
            img, axis=0,
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

    # Re-apply dim-slice blanking after reslicing: cubic interpolation during
    # reslice bleeds signal from bright slices into adjacent blanked slices.
    # Map original dim-slice mask → new (resliced) Z indices via interpolation.
    if _n_dim > 0 and img.shape[0] != len(_dim_mask_z):
        from scipy.ndimage import zoom
        _z_ratio = img.shape[0] / len(_dim_mask_z)
        # Nearest-neighbor upscale of the boolean mask to new Z count
        _dim_mask_z_resliced = zoom(_dim_mask_z.astype(np.float32), _z_ratio, order=0) > 0.5
        _n_reblanked = int(np.sum(_dim_mask_z_resliced))
        print(f"    Post-reslice re-blanking: {_n_reblanked}/{img.shape[0]} slices zeroed")
        for _zi in range(img.shape[0]):
            if _zi < len(_dim_mask_z_resliced) and _dim_mask_z_resliced[_zi]:
                img[_zi] = 0
    elif _n_dim > 0:
        # Same Z count (no reslice happened)
        for _zi in range(img.shape[0]):
            if _dim_mask_z[_zi]:
                img[_zi] = 0

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
    if args.edge_taper_width > 0 and args.padding_mode != 'constant':
        print(f"  [Info] Edge taper active → overriding padding_mode '{args.padding_mode}' with 'constant'")
        effective_padding_mode = 'constant'

    if args.niter > 0:
        t0 = time.time()
        print("[Check-in] Running 3D deconvolution...")
        img = image_scaling_intens(img, args.min_v, args.max_v, True)
        # Pre-deconvolution edge taper: fade borders to zero so RL can't amplify them
        if args.edge_taper_width > 0:
            print(f"    Applying edge taper (width={args.edge_taper_width}px, skip Z) before 3D deconv...")
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
            print(f"    [Warning] 3D deconvolution produced {nan_count} NaN/Inf voxels — replacing with 0")
            np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        t1 = time.time()
        print(f"[Timer] 3D deconvolution took {t1 - t0:.2f} seconds")
        print_resource_usage()

    if args.niterz > 0:
        t0 = time.time()
        print("[Check-in] Running 2D (XZ) deconvolution...")
        # Skip re-normalization if 3D deconv just ran — preserves dynamic range
        if args.niter == 0:
            img = image_scaling_intens(img, args.min_v, args.max_v, True)
        # Transpose to XZ view FIRST, then taper in the space RL will operate on
        img_xz = np.transpose(img, [1, 0, 2])
        psf_xz = np.transpose(psf, [1, 0, 2])
        if args.edge_taper_width > 0:
            # img_xz is (Y, Z, X) — skip Y (axis 0) and Z (axis 1), taper only X edges
            print(f"    Applying edge taper (width={args.edge_taper_width}px, X-only) before XZ deconv...")
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
            print(f"    [Warning] XZ deconvolution produced {nan_count} NaN/Inf voxels — replacing with 0")
            np.nan_to_num(img_xz, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.transpose(img_xz, [1, 0, 2])
        t1 = time.time()
        print(f"[Timer] 2D (XZ) deconvolution took {t1 - t0:.2f} seconds")
        print_resource_usage()

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

    # Destriping: second pass to catch any residual stripes after WBNS,
    # before CLAHE amplifies them.
    if not args.no_destripe:
        t0 = time.time()
        print(f"[Check-in] Post-WBNS destriping (residual pass)...")
        img = destripe_3d(
            img, axis=0,
            sigma_long=args.destripe_sigma_long,
            sigma_short=args.destripe_sigma_short,
        )
        t1 = time.time()
        print(f"[Timer] Post-WBNS destriping took {t1 - t0:.2f} seconds")
        print_resource_usage()

    if apply_clahe:
        t0 = time.time()
        # CLAHE on XY slices (each Z-plane equalized independently).
        # Depth normalization is already handled by z_intensity_correction.
        # XZ CLAHE was creating horizontal stripes because each Y-row got
        # a different equalization — switching to XY avoids that.
        print(f"[Check-in] Applying CLAHE on XY slices (clip_limit={args.clahe_clip_limit})...")
        img = clahe_3d_stack(img, clip_limit=args.clahe_clip_limit, kernel_size=(64, 64), axis=0,min_signal_pct=args.clahe_min_signal_pct)
        t1 = time.time()
        print(f"[Timer] CLAHE took {t1 - t0:.2f} seconds")

        if args.clahe_post_smooth > 0:
            print(f"[Check-in] Post-CLAHE Gaussian smoothing (sigma={args.clahe_post_smooth})...")
            img = ndi.gaussian_filter(img, sigma=args.clahe_post_smooth)
        print_resource_usage()

    # Apply tissue mask ONCE — after all processing is done.
    # This is the only safe place: WBNS and CLAHE need the natural image
    # gradients to avoid creating ringing artifacts at mask boundaries.
    # WBNS (wavelet) may pad dimensions to even numbers, so crop first.
    mz, my, mx = tissue_mask.shape
    if img.shape != tissue_mask.shape:
        print(f"    [Info] Shape mismatch: img {img.shape} vs mask {tissue_mask.shape} — cropping to match")
        img = img[:mz, :my, :mx]
    print("[Check-in] Applying tissue mask to suppress background noise...")
    img[~tissue_mask] = 0
    del tissue_mask

    # Normalization
    if percentiles_source[0] > 0 or percentiles_source[1] < 100:
        t0 = time.time()
        print("[Check-in] Removing outliers and normalizing intensities...")
        low_thres, high_thres = getNormalizationThresholds(img, percentiles_source)
        img = remove_outliers_image(img, low_thres, high_thres)
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
