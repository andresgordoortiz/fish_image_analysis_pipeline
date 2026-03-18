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
    stack, z_axis=0, method="p95", smooth_window=9, eps=1e-8, preserve_dtype=True
):
    """Correct intensity variation along Z axis."""
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
):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to 3D stack."""
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
        result = exposure.equalize_adapthist(
            img01, kernel_size=kernel_size, clip_limit=clip_limit
        ).astype(np.float32, copy=False)
        # Guard against NaN from skimage internal dtype conversions
        np.nan_to_num(result, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        out[i] = result
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


def edge_taper_3d(img, width):
    """Apply a cosine (Tukey) taper to all edges of a 3D volume.

    This smoothly fades the image intensity to zero over `width` pixels
    at every boundary. Applied BEFORE deconvolution to prevent
    Richardson-Lucy from amplifying boundary artifacts.

    Args:
        img: 3D numpy array (Z, Y, X)
        width: number of pixels over which to taper (0 = disabled)
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

    tz = _taper_1d(z, width)
    ty = _taper_1d(y, width)
    tx = _taper_1d(x, width)

    # Apply as separable product: taper(z) * taper(y) * taper(x)
    out *= tz[:, None, None]
    out *= ty[None, :, None]
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
    if args.niter > 0:
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
        print("[Check-in] Running z_intensity_correction...")
        img, scales = z_intensity_correction(
            img, z_axis=0, method="p95", smooth_window=11
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

    # Recalculate resolution for BG subtraction
    resolution_px = int(args.resolution_px0 / new_physical_pixel_sizeZ)
    resolution_pz = int(args.resolution_pz0 / new_physical_pixel_sizeZ)
    print(f"  BG subtraction : {resolution_px},  {resolution_pz}")

    # Deconvolution (GPU)
    if args.niter > 0:
        t0 = time.time()
        print("[Check-in] Running 3D deconvolution...")
        img = image_scaling_intens(img, args.min_v, args.max_v, True)
        # Pre-deconvolution edge taper: fade borders to zero so RL can't amplify them
        if args.edge_taper_width > 0:
            print(f"    Applying edge taper (width={args.edge_taper_width}px) before 3D deconv...")
            img = edge_taper_3d(img, args.edge_taper_width)
        img = np.pad(img, args.padding, mode=args.padding_mode)
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
        img = image_scaling_intens(img, args.min_v, args.max_v, True)
        # Pre-deconvolution edge taper for XZ pass
        if args.edge_taper_width > 0:
            print(f"    Applying edge taper (width={args.edge_taper_width}px) before XZ deconv...")
            img = edge_taper_3d(img, args.edge_taper_width)
        img_xz = np.transpose(img, [1, 0, 2])
        psf_xz = np.transpose(psf, [1, 0, 2])
        img_xz = np.pad(img_xz, args.padding, mode=args.padding_mode)
        imgSizeGB = img_xz.nbytes / (1024**3)
        print(f"    img_xz -size(GB) : {imgSizeGB:.3f}")
        print_resource_usage()
        res_gpu = rl.doRLDeconvolutionFromNpArrays(
            img_xz, psf_xz, niter=args.niter, resAsUint8=False
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
    if args.edge_mask_px > 0 and (args.niter > 0 or args.niterz > 0):
        b = args.edge_mask_px
        print(f"[Check-in] Applying edge mask: zeroing {b}px border...")
        img[:b, :, :] = 0
        img[-b:, :, :] = 0
        img[:, :b, :] = 0
        img[:, -b:, :] = 0
        img[:, :, :b] = 0
        img[:, :, -b:] = 0

    # Post-processing
    t0 = time.time()
    print("[Check-in] Running post-processing...")
    img = image_postprocessing(
        img, resolution_px, resolution_pz, args.noise_lvl, args.sigma
    )
    t1 = time.time()
    print(f"[Timer] Post-processing took {t1 - t0:.2f} seconds")
    print_resource_usage()

    if apply_clahe:
        t0 = time.time()
        print("[Check-in] Applying CLAHE...")
        img_xz = np.transpose(img, [1, 0, 2])
        img_xz = clahe_3d_stack(img_xz, clip_limit=0.01, kernel_size=(64, 64), axis=0)
        img = np.transpose(img_xz, [1, 0, 2])
        t1 = time.time()
        print(f"[Timer] CLAHE took {t1 - t0:.2f} seconds")
        print_resource_usage()

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
