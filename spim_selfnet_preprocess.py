#!/usr/bin/env python3
"""
SPIM Image Preprocessing Pipeline — Self-Net variant
IMP Vienna - Andrés Gordo & Guilherme Ventura

Alternative preprocessing front-end that replaces the GPU Richardson–Lucy
deconvolution + isotropic reslicing of ``spim_pipeline_fixed.py`` with a
Self-Net (CycleGAN-style ResNet) 3D deblurring / isotropic reconstruction
model.

The model takes anisotropic light-sheet stacks, reslices them along the XZ and
YZ planes (upsampling Z to the isotropic XY pixel size), deblurs each resliced
plane with a 2D ResNet generator and fuses the two views. This mirrors the
inference logic in the training repo (``Self_net_output_volume.py``) but is made
self-contained and wired into the SPIM Nextflow pipeline.

All the surrounding correction / post-processing steps (shading, Z-intensity,
WBNS background subtraction, Gaussian smoothing, CLAHE, percentile
normalisation) are reused from ``spim_preprocessing_stages.py`` so the output
is a drop-in replacement for the deconvolution path: same intensity range, same
isotropic geometry and the same ``{base}_{int(100*image_scaling)}.tif`` naming
expected by the Nextflow ``PREPROCESS_*`` rename/metadata steps.

The architecture below is copied (trimmed to the ``deblur_net`` generator) from
``Self_net_architecture.py`` so this script has no dependency on the training
codebase; it only needs the trained ``.pkl`` state-dict.

CLI
---
    python3 spim_selfnet_preprocess.py \
        --input_file t0001.tif \
        --outdir . \
        --config_json preprocessing_config.json

``--config_json`` is the canonical entry point and is shared with
``spim_pipeline_fixed.py``. The Self-Net-specific knobs live under
``preprocessing.selfnet`` in the config. Set ``preprocessing.method =
"selfnet"`` to select this front-end.
"""

import argparse
import functools
import json
import os
import sys
import time

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
from scipy import stats
from skimage.transform import rescale

# Shared helpers and stage functions. Same import path as
# spim_pipeline_fixed.py so the Nextflow process can stage both scripts
# together with WBNS.py.
from spim_pipeline_fixed import (
    image_scaling_intens,
    print_resource_usage,
    read_nd2_voxel_size,
    read_tiff_voxel_size,
    remove_outliers_image,
)
from spim_preprocessing_stages import (
    PIPELINE_STAGES,
    STAGE_NAMES,
    save_intermediate,
)


# =============================================================================
# Self-Net generator architecture (deblur_net) — trimmed from
# Self_net_architecture.py. Deblur_Net does not use the normalization layer,
# but the constructor signature is preserved for state-dict compatibility.
# =============================================================================
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class Res_Block(nn.Module):
    """A ResNet block with reflection padding and skip connection."""

    def __init__(self, dim, padding_type, use_dropout, use_bias):
        super(Res_Block, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Deblur_Net(nn.Module):
    """ResNet-based generator used by Self-Net for per-plane deblurring."""

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(Deblur_Net, self).__init__()
        use_bias = True

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ]

        n_1 = 2
        for _ in range(n_1):
            model += [
                nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
            ]

        for _ in range(n_blocks):
            model += [
                Res_Block(
                    ngf,
                    padding_type=padding_type,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for _ in range(n_1):
            model += [
                nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def build_deblur_net(device, ngf=64, n_blocks=6, norm="instance"):
    """Instantiate the deblur generator on ``device``."""
    norm_layer = get_norm_layer(norm_type=norm)
    net = Deblur_Net(
        input_nc=1,
        output_nc=1,
        ngf=ngf,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=n_blocks,
    )
    return net.to(device)


def load_deblur_net(model_path, device, ngf=64, n_blocks=6, norm="instance"):
    """Build the generator and load a trained state-dict, robust to wrappers."""
    if not os.path.isfile(model_path):
        print(f"ERROR: Self-Net model not found: {model_path}")
        sys.exit(1)

    net = build_deblur_net(device, ngf=ngf, n_blocks=n_blocks, norm=norm)
    state = torch.load(model_path, map_location=device)

    # Unwrap common checkpoint containers
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith("model.") for k in state.keys()
    ):
        state = state["state_dict"]

    try:
        net.load_state_dict(state)
    except RuntimeError:
        # Strip a possible DataParallel "module." prefix and retry
        stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
        net.load_state_dict(stripped)

    net.eval()
    return net


# =============================================================================
# Self-Net inference: reslice -> per-plane deblur -> fuse
# =============================================================================
def reslice_for_net(img, position, x_res, z_res):
    """Reslice a ZYX stack into XZ/YZ planes, upsampling Z to isotropic.

    Returns a float32 array whose leading axis is the batch of 2D planes that
    are fed to the deblur network.
    """
    scale = z_res / x_res
    z, y, x = img.shape
    new_z = max(1, int(round(z * scale)))

    if position == "xz":
        # (z, y, x) -> (y, z, x); resize each (z, x) plane to (new_z, x)
        reslice_img = np.transpose(img, [1, 0, 2])
        out = np.zeros((y, new_z, x), dtype=np.float32)
        for i in range(y):
            out[i] = cv2.resize(
                reslice_img[i].astype(np.float32),
                (x, new_z),
                interpolation=cv2.INTER_CUBIC,
            )
    elif position == "yz":
        # (z, y, x) -> (x, z, y); resize each (z, y) plane to (new_z, y)
        reslice_img = np.transpose(img, [2, 0, 1])
        out = np.zeros((x, new_z, y), dtype=np.float32)
        for i in range(x):
            out[i] = cv2.resize(
                reslice_img[i].astype(np.float32),
                (y, new_z),
                interpolation=cv2.INTER_CUBIC,
            )
    else:
        raise ValueError(f"Unknown reslice position: {position}")

    return out


def run_net(net, device, raw_planes, min_v, max_v, batch_size):
    """Apply the deblur network to a batch-of-planes array (axis 0 = batch)."""
    cpu = torch.device("cpu")
    n = raw_planes.shape[0]
    out = np.zeros_like(raw_planes, dtype=np.uint16)

    scale = float(max_v - min_v)
    if scale == 0:
        scale = 1.0

    inp = (raw_planes.astype(np.float32) - min_v) / scale
    inp = np.clip(inp, 0.0, 1.0)
    inp = np.expand_dims(inp, axis=1)  # (n, 1, H, W)
    tensor = torch.from_numpy(inp)

    n_full = n // batch_size
    res = n - n_full * batch_size

    def _infer(start, end):
        with torch.no_grad():
            o = net(tensor[start:end].to(device))
        o = o.squeeze_(1).to(cpu).numpy()
        o = o * scale + min_v
        return np.clip(o, 0, max_v).astype(np.uint16)

    for ii in range(n_full):
        s = ii * batch_size
        e = (ii + 1) * batch_size
        out[s:e] = _infer(s, e)
        print(f"    deblur {e}/{n}")

    if res != 0:
        s = n_full * batch_size
        out[s:] = _infer(s, n)
        print(f"    deblur {n}/{n}")

    return out


def _simple_fg_mask(img, thres_scale):
    """Mode-based foreground mask, matching get_image_simple_mask(blurWnd=0).

    The training-time normalization estimates the foreground from the image
    mode (background peak) scaled by ``thres_scale``; no smoothing is applied
    because the original call passes ``blurWnd=0``.
    """
    flat = img.ravel()
    flat = flat[flat != 0]
    if flat.size == 0:
        return np.ones(img.shape, dtype=np.int16)
    mode_result = stats.mode(flat, axis=None)
    mode_val = float(np.atleast_1d(mode_result.mode)[0])
    threshold_value = thres_scale * mode_val
    return (img > threshold_value).astype(np.int16)


def selfnet_input_normalization(img, percentiles, thres_scale, min_v, max_v):
    """Replicate Supporting_functions.image_normalization (training/inference).

    Percentile outlier removal — low threshold from the whole image, high
    threshold from the foreground only — followed by a linear rescale into
    ``[min_v, max_v]``. This matches the intensity distribution the
    ``deblur_net`` model was trained on, so the network receives inputs that
    look like its training data even though the surrounding SPIM corrections
    (shading, Z-intensity) have already run.
    """
    if percentiles[0] > 0 or percentiles[1] < 100:
        mask = _simple_fg_mask(img, thres_scale)
        low_thres, _ = getNormalizationThresholds(img, percentiles)
        _, high_thres = getNormalizationThresholds(img * mask, percentiles)
        img = remove_outliers_image(img, low_thres, high_thres)
    img = image_scaling_intens(img, min_v, max_v, True)
    return img.astype(np.uint16)


def selfnet_deblur(
    img, x_res, z_res, net_xz, net_yz, device, min_v, max_v, batch_size
):
    """Full Self-Net reconstruction: XZ + YZ deblur, then average-fuse.

    Returns an isotropic uint16 ZYX volume with Z upsampled by z_res/x_res.
    """
    xz_planes = reslice_for_net(img, "xz", x_res, z_res)
    yz_planes = reslice_for_net(img, "yz", x_res, z_res)
    print(f"    XZ planes: {xz_planes.shape} | YZ planes: {yz_planes.shape}")

    out_xz = run_net(net_xz, device, xz_planes, min_v, max_v, batch_size)
    out_yz = run_net(net_yz, device, yz_planes, min_v, max_v, batch_size)

    # Restore to ZYX orientation and fuse by averaging the two views.
    re_xz = np.transpose(out_xz, [1, 0, 2])  # (new_z, y, x)
    re_yz = np.transpose(out_yz, [1, 2, 0])  # (new_z, y, x)

    fusion = re_xz.astype(np.float32) / 2.0 + re_yz.astype(np.float32) / 2.0
    return fusion.astype(np.uint16)


def select_device():
    """Pick a usable compute device, falling back to CPU on incompatible GPUs.

    ``torch.cuda.is_available()`` returns True even for GPUs whose compute
    capability is too old for the installed PyTorch (e.g. a Tesla P100, sm_60,
    with a build that only ships sm_70+ kernels). On such hardware the first
    CUDA kernel launch raises ``no kernel image is available for execution on
    the device``. Because SLURM may schedule a task on any GPU in the pool,
    we verify both the reported capability *and* that a real kernel actually
    runs before committing to CUDA; otherwise we transparently use the CPU so
    the timepoint still completes (slower, but correct).
    """
    if not torch.cuda.is_available():
        print("  No CUDA device available — using CPU.")
        return torch.device("cpu")

    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
    except Exception as e:  # pragma: no cover - defensive
        print(f"  Could not query CUDA device ({e}) — using CPU.")
        return torch.device("cpu")

    # Compute capabilities supported by the installed PyTorch build.
    try:
        supported = torch.cuda.get_arch_list()
    except Exception:
        supported = []
    supported_majors = []
    for arch in supported:
        # Entries look like 'sm_70', 'sm_86', 'sm_90', 'sm_100', 'sm_120'.
        digits = "".join(ch for ch in arch if ch.isdigit())
        if len(digits) >= 2:
            supported_majors.append(int(digits[:-1]))
    min_supported_major = min(supported_majors) if supported_majors else 7

    if major < min_supported_major:
        print(
            f"  GPU '{name}' has compute capability {major}.{minor}, but the "
            f"installed PyTorch supports sm_{min_supported_major}0+ "
            f"({', '.join(supported) if supported else 'unknown'}). "
            "Falling back to CPU for Self-Net inference."
        )
        return torch.device("cpu")

    # Capability looks fine; confirm a kernel actually launches on this device.
    try:
        _t = torch.zeros((1, 1, 4, 4), device="cuda:0")
        _ = torch.nn.functional.pad(_t, (1, 1, 1, 1), mode="reflect")
        torch.cuda.synchronize()
    except Exception as e:
        print(
            f"  GPU '{name}' (cc {major}.{minor}) failed a CUDA smoke-test "
            f"({type(e).__name__}: {e}). Falling back to CPU."
        )
        return torch.device("cpu")

    print(f"  Using GPU '{name}' (compute capability {major}.{minor}).")
    return torch.device("cuda:0")


def main():
    parser = argparse.ArgumentParser(
        description="SPIM Image Preprocessing (Self-Net deblurring)"
    )

    # Paths
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input image"
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained Self-Net deblur model (.pkl state-dict)",
    )
    parser.add_argument(
        "--model_path_xz",
        type=str,
        default="",
        help="Optional separate model for the XZ view (defaults to --model_path)",
    )
    parser.add_argument(
        "--model_path_yz",
        type=str,
        default="",
        help="Optional separate model for the YZ view (defaults to --model_path)",
    )

    # Model / inference parameters
    parser.add_argument("--ngf", type=int, default=64, help="Generator base filters")
    parser.add_argument(
        "--n_blocks", type=int, default=6, help="Number of ResNet blocks"
    )
    parser.add_argument(
        "--norm", type=str, default="instance", help="Normalization layer type"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Plane batch size for inference"
    )
    parser.add_argument(
        "--net_min_v",
        type=float,
        default=0.0,
        help="Min intensity used to normalise the network input",
    )
    parser.add_argument(
        "--net_max_v",
        type=float,
        default=65535.0,
        help="Max intensity used to normalise the network input",
    )
    parser.add_argument(
        "--no_net_normalization",
        action="store_true",
        help="Disable the training-matched percentile normalization of the "
        "Self-Net input (feed the shading/Z-corrected stack directly)",
    )
    parser.add_argument(
        "--net_percentile_low",
        type=float,
        default=30.0,
        help="Low percentile for the Self-Net input normalization (training default)",
    )
    parser.add_argument(
        "--net_percentile_high",
        type=float,
        default=99.999,
        help="High percentile for the Self-Net input normalization (training default)",
    )
    parser.add_argument(
        "--net_thres_scale",
        type=float,
        default=1.5,
        help="Foreground mode multiplier for the Self-Net input normalization "
        "(training default)",
    )

    # Image Parameters
    parser.add_argument(
        "--image_scaling", type=float, default=1.0, help="XY image scaling factor"
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

    # Normalization Params
    parser.add_argument(
        "--min_v", type=float, default=0, help="Min value for final normalization"
    )
    parser.add_argument(
        "--max_v", type=float, default=65535, help="Max value for final normalization"
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
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Gaussian smoothing sigma"
    )
    # Canonical config-driven entry point (shared with spim_pipeline_fixed.py).
    # When provided, values from the JSON config populate the args namespace
    # (only if not already set on the CLI, so explicit CLI overrides still win).
    parser.add_argument(
        "--config_json", type=str, default=None,
        help="Path to JSON preprocessing config. When provided, the config "
             "populates parameters that were not set on the CLI."
    )
    parser.add_argument(
        "--metadata_json", type=str, default=None,
        help="Optional metadata JSON (Nextflow wrapper uses this to inject "
             "per-timepoint voxel sizes)."
    )

    args = parser.parse_args()

    # If --config_json was passed, populate unset args from it. CLI values
    # win over config (only None / defaults get overridden).
    if args.config_json:
        with open(args.config_json) as f:
            _raw = f.read()
        if _raw.startswith("﻿"):
            _raw = _raw[1:]
        _config = json.loads(_raw)
        _pp = (_config.get("preprocessing") or {})
        _sn = (_pp.get("selfnet") or {})

        def _set_if_default(name, value):
            """Set ``args.name = value`` only if the current value matches the
            argparse default (or is None). We rely on the fact that argparse
            only fills in defaults; explicitly-set CLI values are kept."""
            current = getattr(args, name, None)
            if current is None:
                setattr(args, name, value)

        if _pp.get("downscale_xy", {}).get("factor") is not None:
            _set_if_default("image_scaling", float(_pp["downscale_xy"]["factor"]))
        if _sn.get("ngf") is not None:
            _set_if_default("ngf", int(_sn["ngf"]))
        if _sn.get("n_blocks") is not None:
            _set_if_default("n_blocks", int(_sn["n_blocks"]))
        if _sn.get("norm") is not None:
            _set_if_default("norm", str(_sn["norm"]))
        if _sn.get("batch_size") is not None:
            _set_if_default("batch_size", int(_sn["batch_size"]))
        if _sn.get("net_min_v") is not None:
            _set_if_default("net_min_v", float(_sn["net_min_v"]))
        if _sn.get("net_max_v") is not None:
            _set_if_default("net_max_v", float(_sn["net_max_v"]))
        if _sn.get("net_percentile_low") is not None:
            _set_if_default("net_percentile_low", float(_sn["net_percentile_low"]))
        if _sn.get("net_percentile_high") is not None:
            _set_if_default("net_percentile_high", float(_sn["net_percentile_high"]))
        if _sn.get("net_thres_scale") is not None:
            _set_if_default("net_thres_scale", float(_sn["net_thres_scale"]))
        if _sn.get("model_path"):
            _set_if_default("model_path", _sn["model_path"])
        if _sn.get("model_path_xz"):
            _set_if_default("model_path_xz", _sn["model_path_xz"])
        if _sn.get("model_path_yz"):
            _set_if_default("model_path_yz", _sn["model_path_yz"])

        # No_net_normalization is a flag (False by default); only set if True.
        if _sn.get("no_net_normalization"):
            args.no_net_normalization = True

        # Apply legacy toggle flags (no_clahe / no_shading / no_z_correction).
        if _pp.get("clahe", {}).get("enabled") is False:
            args.no_clahe = True
        if _pp.get("shading_correction", {}).get("enabled") is False:
            args.no_shading = True
        if _pp.get("z_intensity_correction", {}).get("enabled") is False:
            args.no_z_correction = True

        # Voxel sizes: prefer metadata.json (Nextflow path) over config.
        if args.metadata_json and os.path.isfile(args.metadata_json):
            with open(args.metadata_json) as f:
                _meta = json.load(f)
            if _meta.get("x_resolution_um"):
                _set_if_default("xy_pixel", float(_meta["x_resolution_um"]))
            if _meta.get("y_resolution_um"):
                # legacy script had no y_pixel flag; z_pixel is fine
                pass
            if (_meta.get("imagej") or {}).get("spacing"):
                _set_if_default("z_pixel", float(_meta["imagej"]["spacing"]))
        else:
            _v = _pp.get("voxel_size") or {}
            if _v.get("x_um") is not None:
                _set_if_default("xy_pixel", float(_v["x_um"]))
            if _v.get("z_um") is not None:
                _set_if_default("z_pixel", float(_v["z_um"]))

    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file does not exist: {args.input_file}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SPIM PREPROCESSING PIPELINE (Self-Net)")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nParameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 60 + "\n")

    apply_clahe = not args.no_clahe
    apply_z_intensity_correction = not args.no_z_correction
    apply_shading_correct = not args.no_shading
    apply_net_normalization = not args.no_net_normalization
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

    # ------------------------------------------------------------------
    # Device + model loading
    # ------------------------------------------------------------------
    device = select_device()
    print(f"Self-Net device: {device}")

    t0 = time.time()
    model_xz_path = args.model_path_xz if args.model_path_xz else args.model_path
    model_yz_path = args.model_path_yz if args.model_path_yz else args.model_path

    net_xz = load_deblur_net(
        model_xz_path, device, ngf=args.ngf, n_blocks=args.n_blocks, norm=args.norm
    )
    if model_yz_path == model_xz_path:
        net_yz = net_xz
    else:
        net_yz = load_deblur_net(
            model_yz_path, device, ngf=args.ngf, n_blocks=args.n_blocks, norm=args.norm
        )
    print(f"[Timer] Model loading took {time.time() - t0:.2f} seconds")

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    image_path = args.input_file
    image_name = os.path.basename(image_path)
    print(f"\n[Processing] {image_name}")
    print_resource_usage()

    start_time_total = time.time()

    t0 = time.time()
    ext = os.path.splitext(image_name)[1].lower()
    try:
        if ext in [".tif", ".tiff"]:
            print("  Loading TIFF image...")
            img = tifffile.imread(image_path).astype(np.uint16)
            voxel_size = read_tiff_voxel_size(image_path)
        elif ext == ".nd2":
            print("  Loading ND2 image...")
            import pims

            img = pims.open(image_path)
            voxel_size = read_nd2_voxel_size(img)
            img = np.array(img, dtype=np.uint16, copy=False)
        else:
            print(f"ERROR: Unsupported format: {ext}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        sys.exit(1)

    print(f"[Timer] Image loading took {time.time() - t0:.2f} seconds")
    print(f"  - shape: {img.shape}, dtype: {img.dtype}")

    if img.ndim != 3:
        print(f"ERROR: Expected a 3D ZYX stack, got shape {img.shape}")
        sys.exit(1)

    physical_pixel_sizeX, physical_pixel_sizeY, physical_pixel_sizeZ = voxel_size
    if tempScale > 0:
        physical_pixel_sizeX = args.xy_pixel
        physical_pixel_sizeZ = args.z_pixel
    print(f"  - voxel sizes (um): {voxel_size}")
    print_resource_usage()

    # ------------------------------------------------------------------
    # XY scaling (same convention as the deconvolution path)
    # ------------------------------------------------------------------
    if args.image_scaling > 0 and args.image_scaling != 1.0:
        t0 = time.time()
        in_shape = img.shape
        img = rescale(
            img,
            (1.0, args.image_scaling, args.image_scaling),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint16)
        physical_pixel_sizeX /= args.image_scaling
        print(f"  - XY rescale {in_shape} -> {img.shape} (scaling {args.image_scaling})")
        print(f"[Timer] Image rescaling took {time.time() - t0:.2f} seconds")
        print_resource_usage()

    img_shape = img.shape  # pre-reconstruction shape (Z, Y, X)

    # ------------------------------------------------------------------
    # Pre-processing corrections (shading + Z-intensity), shared with deconv
    # ------------------------------------------------------------------
    if apply_shading_correct:
        t0 = time.time()
        print("[Check-in] Running shading_correct_xy_estimated...")
        img, _ = shading_correct_xy_estimated(img, sigma_xy=96, z_axis=0, per_slice=False)
        print(f"[Timer] Shading correction took {time.time() - t0:.2f} seconds")
        print_resource_usage()

    if apply_z_intensity_correction:
        t0 = time.time()
        print("[Check-in] Running z_intensity_correction...")
        img, _ = z_intensity_correction(img, z_axis=0, method="p95", smooth_window=11)
        print(f"[Timer] Z-intensity correction took {time.time() - t0:.2f} seconds")
        print_resource_usage()

    img = img.astype(np.uint16)

    # ------------------------------------------------------------------
    # Training-matched input normalization for the Self-Net.
    # The deblur_net was trained/validated on volumes passed through
    # image_normalization (percentile outlier removal + rescale to
    # net_min_v..net_max_v) right before inference. Apply the same here so the
    # network sees its expected intensity distribution. The surrounding SPIM
    # corrections above (shading, Z-intensity) and the post-processing below
    # (WBNS, CLAHE, final normalization) are unaffected.
    # ------------------------------------------------------------------
    if apply_net_normalization:
        t0 = time.time()
        print("[Check-in] Normalising Self-Net input (training-matched)...")
        img = selfnet_input_normalization(
            img,
            (args.net_percentile_low, args.net_percentile_high),
            args.net_thres_scale,
            args.net_min_v,
            args.net_max_v,
        )
        print(f"[Timer] Self-Net input normalization took {time.time() - t0:.2f} seconds")
        print_resource_usage()

    # ------------------------------------------------------------------
    # Self-Net deblurring + isotropic reconstruction (replaces deconv+reslice)
    # ------------------------------------------------------------------
    t0 = time.time()
    print("[Check-in] Running Self-Net deblurring / isotropic reconstruction...")
    scale = physical_pixel_sizeX / physical_pixel_sizeZ
    if abs(1.0 - scale) < 1e-4:
        print("  - Image already isotropic; Self-Net applied at scale 1.0")
    img = selfnet_deblur(
        img,
        physical_pixel_sizeX,
        physical_pixel_sizeZ,
        net_xz,
        net_yz,
        device,
        args.net_min_v,
        args.net_max_v,
        args.batch_size,
    )
    print(f"[Timer] Self-Net reconstruction took {time.time() - t0:.2f} seconds")
    print_resource_usage()

    img = img.astype(np.float32)
    new_img_shape = img.shape

    # Recompute the (now isotropic) Z spacing from the shape change.
    new_physical_pixel_sizeZ = img_shape[0] * physical_pixel_sizeZ / new_img_shape[0]
    print(
        f"  - image dimension from : {img_shape} to {new_img_shape} after Self-Net reconstruction"
    )
    print(f"  - z-space from : {physical_pixel_sizeZ} to {new_physical_pixel_sizeZ}")
    physical_pixel_sizeZ = new_physical_pixel_sizeZ

    resolution_px = int(args.resolution_px0 / new_physical_pixel_sizeZ)
    resolution_pz = int(args.resolution_pz0 / new_physical_pixel_sizeZ)
    print(f"  BG subtraction : {resolution_px},  {resolution_pz}")

    # ------------------------------------------------------------------
    # Post-processing (WBNS + Gaussian smoothing), shared with deconv
    # ------------------------------------------------------------------
    t0 = time.time()
    print("[Check-in] Running post-processing...")
    img = image_postprocessing(
        img, resolution_px, resolution_pz, args.noise_lvl, args.sigma
    )
    print(f"[Timer] Post-processing took {time.time() - t0:.2f} seconds")
    print_resource_usage()

    if apply_clahe:
        t0 = time.time()
        print("[Check-in] Applying CLAHE...")
        img_xz = np.transpose(img, [1, 0, 2])
        img_xz = clahe_3d_stack(img_xz, clip_limit=0.01, kernel_size=(64, 64), axis=0)
        img = np.transpose(img_xz, [1, 0, 2])
        print(f"[Timer] CLAHE took {time.time() - t0:.2f} seconds")
        print_resource_usage()

    if percentiles_source[0] > 0 or percentiles_source[1] < 100:
        t0 = time.time()
        print("[Check-in] Removing outliers and normalizing intensities...")
        low_thres, high_thres = getNormalizationThresholds(img, percentiles_source)
        img = remove_outliers_image(img, low_thres, high_thres)
        print(f"[Timer] Outlier removal took {time.time() - t0:.2f} seconds")
        print_resource_usage()

    # ------------------------------------------------------------------
    # Final intensity scaling + save (same naming as the deconv path)
    # ------------------------------------------------------------------
    t0 = time.time()
    print("[Check-in] Final intensity scaling and saving...")
    img = image_scaling_intens(img, args.min_v, args.max_v, True)
    img = img.astype(np.uint16)
    print(f"[Timer] Final scaling took {time.time() - t0:.2f} seconds")

    base_name = os.path.splitext(image_name)[0]
    image_out_name = f"{base_name}_{int(100 * args.image_scaling)}.tif"
    img_out_path = os.path.join(args.outdir, image_out_name)
    tifffile.imwrite(img_out_path, img)
    print(f"  Saved processed image to: {img_out_path}")

    if not os.path.isfile(img_out_path):
        print(f"ERROR: Output file was not created: {img_out_path}")
        sys.exit(1)
    output_size = os.path.getsize(img_out_path)
    if output_size == 0:
        print(f"ERROR: Output file is empty: {img_out_path}")
        sys.exit(1)

    print(f"[Success] Output size: {output_size / (1024**2):.2f} MB")
    print(f"[Done] Elapsed Time: {time.time() - start_time_total:.4f} seconds")
    print_resource_usage()


if __name__ == "__main__":
    main()
