#!/usr/bin/env python3
"""Bit-equivalence test: SPIM ``bin/`` scripts vs AIAF-32 reference scripts.

This test runs the AIAF-32 reference math (imported as a Python module — no
bioio, no OME-TIFF conversion) and compares its output array against the
output produced by our ``bin/`` script on the same input. The math is
expected to be bit-identical because we ported it verbatim.

The test uses ONLY ``tifffile`` for I/O on our side. The AIAF-32 reference
math functions operate on plain numpy arrays and don't care what container
the input came from, so no bioio import is needed.

Run with::

    /usr/local/bin/python3 tests/test_aiaf32_equivalence.py

or via pytest::

    pytest tests/test_aiaf32_equivalence.py -v
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parent.parent
AIAF32_BIN = Path.home() / "Projects" / "IMP" / "AIAF-32_embryo-cell-tracking" / "ipa" / "bin"
PYTHON = sys.executable


def _load_aiaf32_module(name: str) -> object:
    """Import an AIAF-32 bin/ script as a Python module without needing bioio."""
    spec = importlib.util.spec_from_file_location(name, AIAF32_BIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_volume() -> tuple[np.ndarray, tuple[float, float, float]]:
    """Deterministic synthetic anisotropic volume with shading + depth attenuation.

    Returns the uint16 ZYX array and the (z_um, y_um, x_um) voxel sizes.
    """
    rng = np.random.default_rng(42)
    Z, Y, X = 32, 128, 128
    img = rng.normal(200.0, 30.0, size=(Z, Y, X)).astype(np.float32)
    yy, xx = np.meshgrid(np.linspace(-1, 1, Y), np.linspace(-1, 1, X), indexing="ij")
    shading = (1.0 - 0.6 * (xx ** 2 + yy ** 2)).clip(0.4, 1.1).astype(np.float32)
    img = img * shading[None, :, :]
    depth_atten = np.linspace(1.0, 0.3, Z, dtype=np.float32).reshape(Z, 1, 1)
    img = img * depth_atten
    img_u16 = np.clip(img, 0, 65535).astype(np.uint16)
    return img_u16, (0.748, 0.374, 0.374)


def _write_input_tiff(path: Path, arr: np.ndarray, voxel_zyx: tuple[float, float, float]) -> None:
    """Write a plain TIFF with ImageJ metadata (what our bin/ scripts read)."""
    z_um, y_um, x_um = voxel_zyx
    tifffile.imwrite(
        str(path),
        arr,
        imagej=True,
        resolution=(1.0 / x_um, 1.0 / y_um),
        metadata={"spacing": z_um, "unit": "um", "axes": "ZYX"},
        compression="zlib",
    )


def _run_subprocess(args: list[str]) -> None:
    """Run a subprocess, raising on non-zero exit."""
    r = subprocess.run([PYTHON] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{args} failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}")


def test_planar_matches_aiaf32() -> None:
    arr, voxel = _make_synthetic_volume()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "input.tif"
        out = tmp / "planar.tif"
        _write_input_tiff(src, arr, voxel)

        ref_mod = _load_aiaf32_module("planar_intensity_correction")
        ref_out, _ = ref_mod.planar_intensity_correction(arr, sigma_xy=64.0)
        ref_u16 = ref_out.astype(np.uint16) if ref_out.dtype != np.uint16 else ref_out

        _run_subprocess([
            "bin/planar_intensity_correction.py",
            "--input", str(src), "--output", str(out), "--sigma_xy", "64",
        ])
        ours = tifffile.imread(str(out))

    np.testing.assert_array_equal(ref_u16, ours,
        err_msg=f"PLANAR: max diff {(ref_u16.astype(np.int32) - ours.astype(np.int32)).max()}")


def test_depth_matches_aiaf32() -> None:
    arr, voxel = _make_synthetic_volume()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "input.tif"
        out = tmp / "depth.tif"
        _write_input_tiff(src, arr, voxel)

        ref_mod = _load_aiaf32_module("depth_intensity_correction")
        ref_out = ref_mod.depth_intensity_correction(
            arr, mode="p99", smooth_window=9, preserve_dtype=True, gain_clip=(0.25, 4.0),
        )
        ref_u16 = ref_out.astype(np.uint16) if ref_out.dtype != np.uint16 else ref_out

        _run_subprocess([
            "bin/depth_intensity_correction.py",
            "--input", str(src), "--output", str(out),
            "--mode", "p99", "--smooth_window", "9",
            "--gain_min", "0.25", "--gain_max", "4.0",
        ])
        ours = tifffile.imread(str(out))

    np.testing.assert_array_equal(ref_u16, ours,
        err_msg=f"DEPTH: max diff {(ref_u16.astype(np.int32) - ours.astype(np.int32)).max()}")


def test_isotropic_matches_aiaf32() -> None:
    arr, voxel = _make_synthetic_volume()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "input.tif"
        out = tmp / "iso.tif"
        _write_input_tiff(src, arr, voxel)

        ref_mod = _load_aiaf32_module("isotropic")
        ref_out = ref_mod.make_isotropic(arr, voxel, (0.374, 0.374, 0.374), order=3)
        ref_u16 = ref_out.astype(np.uint16) if ref_out.dtype != np.uint16 else ref_out

        _run_subprocess([
            "bin/isotropic_resample.py",
            "--input", str(src), "--output", str(out),
            "--target_um", "0.374", "--order", "3",
        ])
        ours = tifffile.imread(str(out))

    np.testing.assert_array_equal(ref_u16, ours,
        err_msg=f"ISO: max diff {(ref_u16.astype(np.int32) - ours.astype(np.int32)).max()}")


def test_full_chain_end_to_end() -> None:
    """Run the whole planar -> depth -> isotropic chain on a synthetic stack."""
    arr, voxel = _make_synthetic_volume()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "input.tif"
        planar_out = tmp / "planar.tif"
        depth_out = tmp / "depth.tif"
        iso_out = tmp / "iso.tif"
        _write_input_tiff(src, arr, voxel)

        _run_subprocess(["bin/planar_intensity_correction.py",
            "--input", str(src), "--output", str(planar_out), "--sigma_xy", "64"])
        _run_subprocess(["bin/depth_intensity_correction.py",
            "--input", str(planar_out), "--output", str(depth_out),
            "--mode", "p99", "--smooth_window", "9",
            "--gain_min", "0.25", "--gain_max", "4.0"])
        _run_subprocess(["bin/isotropic_resample.py",
            "--input", str(depth_out), "--output", str(iso_out),
            "--target_um", "0.374", "--order", "3"])

        result = tifffile.imread(str(iso_out))
        # Sanity: shape changed (Z resampled), dtype preserved, voxel metadata present.
        assert result.shape[0] > arr.shape[0], "Isotropic resampling should increase Z slices"
        assert result.dtype == arr.dtype
        with tifffile.TiffFile(str(iso_out)) as tf:
            desc = tf.pages[0].imagej_description or ""
            assert "spacing=0.374" in desc, f"Z spacing not round-tripped, desc={desc!r}"
            assert "unit=um" in desc


if __name__ == "__main__":
    test_planar_matches_aiaf32()
    print("✓ planar_intensity_correction: bit-identical to AIAF-32")
    test_depth_matches_aiaf32()
    print("✓ depth_intensity_correction:  bit-identical to AIAF-32")
    test_isotropic_matches_aiaf32()
    print("✓ isotropic_resample:          bit-identical to AIAF-32")
    test_full_chain_end_to_end()
    print("✓ Full planar -> depth -> isotropic chain runs end-to-end")
    print()
    print("All tests passed.")
