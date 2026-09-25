"""Imaris / HDF5 input helpers for the SPIM SPLIT_INPUT_FILE step.

This module exists because the Imaris / HDF5 readers need to use literal
double-quote characters inside Python regex patterns (for parsing the
'PhysicalSizeX value="..."' XML bit in Imaris' DatasetInfoXml blob).
Keeping the regex inline in the Nextflow ``script:`` triple-quoted
heredoc trips Groovy's GString parser at the embedded ``?"`` sequence,
so the heavy lifting lives in this plain ``.py`` file and
``SPLIT_INPUT_FILE`` just calls us. See repo memory
``script_block_gstring_escaping.md`` round 5 (2026-09-25) for the
symptoms and fix history.

The functions are intentionally import-light so the same code paths
work both as part of a tiny driver (`python3 _ims_reader.py split-ims
<path> <channel_cfg>`) and as `from _ims_reader import ...` inside the
SPLIT_INPUT_FILE Python heredoc.

Conventions
-----------
- ``split_ims(path, channel_cfg, voxel_emit_path=None)``         -> single .ims / .h5 file
- ``split_h5_per_timepoint(directory, channel_cfg, voxel_emit_path=None)``
                                                              -> BF per-TP folder

Both write one ``t####_Channel <c>.tif`` per timepoint into the
current working directory, matching the rest of the pipeline's
``t####_Channel <c>.tif`` convention. ``voxel_emit_path`` (when
supplied) is written as a small JSON sidecar consumed by
``EXTRACT_METADATA`` when ``voxel_size.auto_detect=true``.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Constants kept module-scope so the per-TP driver can share them with
# the SPLIT_INPUT_FILE bundled heredoc.
# ---------------------------------------------------------------------------

_BF_PER_TP_PATTERN = re.compile(r'(?i)--C(\d{2,})--T(\d{4,5})\.h5?$')


# ---------------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------------

def _discover_ims_layout(h5: "h5py.File") -> dict[str, Any]:
    """Locate the Imaris ``ResolutionLevel 0`` array dataset inside an
    HDF5 / Imaris file. Returns a dict with the keys ``mode``, ``root``,
    ``nT``, ``nC``, ``nZ``, ``nY``, ``nX``, ``axes`` (and ``tp_keys`` /
    ``ch_keys`` for the ``imaris`` layout).

    Raises a helpful :class:`RuntimeError` explaining the supported
    layout if nothing recognisable is found.
    """
    candidates = []
    if 'DataSet' in h5:
        ds = h5['DataSet']
        if 'ResolutionLevel 0' in ds:
            candidates.append(ds['ResolutionLevel 0'])
        for k in ds.keys():
            if (isinstance(ds[k], h5py.Group)
                    and k not in candidates
                    and (k.startswith('ResolutionLevel')
                         or k.startswith('Resolution Level'))):
                candidates.append(ds[k])

    if not candidates:
        for k in ('DataSet/ImageData', 'Data', 'data', 'image'):
            if k in h5 and isinstance(h5[k], h5py.Dataset):
                candidates.append(k)
    if not candidates:
        raise RuntimeError(
            "Could not find an Imaris / HDF5 image dataset inside this file. "
            "Expected something like '/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'. "
            "If this file was written by an exporter other than Imaris/BioFormats, "
            "convert it once (e.g. with `python -m bfconvert in.ims out_t%d.tif`) "
            "and re-point `input.directory` at the resulting TIFF folder."
        )

    chosen = None
    for c in candidates:
        if isinstance(c, h5py.Dataset):
            chosen = ('flat', c)
            break
        sub_keys = list(c.keys())
        tp_keys = [k for k in sub_keys
                   if k.lower().startswith(('timepoint', 'time point', 't'))]
        if tp_keys:
            chosen = ('imaris', c)
            break
    if chosen is None:
        raise RuntimeError(
            "HDF5 file has DataSet/ResolutionLevel* children but none look like "
            "TimePoint <t>/Channel <c>/Data. Refusing to guess further."
        )

    mode, root = chosen

    if mode == 'flat':
        ds = root
        if ds.shape[0] > 32:
            raise RuntimeError(
                f"Flat dataset too large to materialise in one shot "
                f"(shape={ds.shape}). Use the per-timepoint .h5 folder mode "
                f"instead (point `input.directory` at the folder that "
                f"contains the *--C##--T#####.h5 files)."
            )
        arr = ds[...]
        ndim = arr.ndim
        if ndim not in (4, 5):
            raise RuntimeError(f"Expected 4D/5D HDF5 array, got shape {arr.shape}")
        if ndim == 5:
            nT, nC, nZ, nY, nX = arr.shape
        else:
            nT = 1
            nC, nZ, nY, nX = arr.shape
        return dict(mode='flat', root=ds, nT=nT, nC=nC, nZ=nZ, nY=nY, nX=nX,
                    axes=['T', 'C', 'Z', 'Y', 'X'])

    # imaris layout: walk /DataSet/ResolutionLevel 0/TimePoint/<t>/Channel/<c>/Data
    tp_keys = sorted(root.keys(),
                     key=lambda k: int(''.join(ch for ch in k if ch.isdigit()) or 0))
    nT = len(tp_keys)
    first_tp = root[tp_keys[0]]
    ch_keys = sorted(
        [k for k in first_tp.keys() if k.lower().startswith(('channel', 'c'))],
        key=lambda k: int(''.join(ch for ch in k if ch.isdigit()) or 0),
    )
    nC = len(ch_keys)
    first_ch = first_tp[ch_keys[0]]
    if 'Data' not in first_ch:
        raise RuntimeError(
            f"Imaris layout: expected '/DataSet/ResolutionLevel 0/TimePoint <t>/Channel <c>/Data' "
            f"but found children {list(first_ch.keys())} under {first_ch.name}"
        )
    sample = first_ch['Data']
    arr_sample = sample[...]
    if arr_sample.ndim not in (3, 5):
        raise RuntimeError(
            f"Imaris per-channel shape {arr_sample.shape} is not 5D (1,1,Z,Y,X). "
            f"Imaris 9.x always writes the 5D shape; older 3D exports need to "
            f"be re-exported first."
        )
    if arr_sample.ndim == 5:
        _, _, nZ, nY, nX = arr_sample.shape
    else:
        nZ, nY, nX = arr_sample.shape
    return dict(mode='imaris', root=root, nT=nT, nC=nC, nZ=nZ, nY=nY, nX=nX,
                axes=['T', 'C', 'Z', 'Y', 'X'], tp_keys=tp_keys, ch_keys=ch_keys)


# ---------------------------------------------------------------------------
# Voxel-size extraction
# ---------------------------------------------------------------------------

def _read_ims_voxel_um(h5: "h5py.File", layout: dict) -> dict | None:
    """Pull physical voxel sizes from an Imaris / HDF5 file.

    Resolution order:

      1. ``/DataSet/Info/DatasetInfoXml`` byte blob (always present in
         Imaris) — we regex for ``PhysicalSizeX value="..."``.
         Unit attribute honoured (``nm``/``um``/``mm``/``pm``).
      2. Top-level HDF5 attributes ``physical_pixel_sizes`` /
         ``voxelsize_um_xy``.
      3. Top-level HDF5 attribute ``voxel_size_um`` as a JSON object
         ``{'x': ..., 'y': ..., 'z': ...}``.

    Returns ``None`` if nothing usable is found so the caller can fall
    back to ``config.voxel_size.{x,y,z}_um``.
    """
    def _to_float(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    # Path 1: Imaris DatasetInfoXml (XML attribute or element text).
    if 'DataSet' in h5 and isinstance(h5['DataSet'], h5py.Group) and 'Info' in h5['DataSet']:
        info_grp = h5['DataSet']['Info']
        for info_key in ('DatasetInfoXml', 'ImageInfoXml'):
            if info_key not in info_grp:
                continue
            xml_node = info_grp[info_key]
            xml = None
            try:
                if xml_node.dtype.kind == 'O':
                    xml = ''.join(bytes(x).decode('utf-8', 'replace') for x in xml_node[...])
                else:
                    xml = xml_node[...].tobytes().decode('utf-8', 'replace')
            except Exception:
                continue
            if not xml:
                continue
            vals = {}
            # PhysicalSizeX value=".."   OR   <PhysicalSizeX>..</PhysicalSizeX>
            for axis in ('X', 'Y', 'Z'):
                m = (
                    re.search(
                        rf'\bPhysicalSize{axis}\b[^>]*?(?:value=|>)" ?'
                        r'([0-9eE+\-.]+) ?"',
                        xml,
                    )
                    or re.search(rf'<PhysicalSize{axis}>([^<]+)</PhysicalSize{axis}>', xml)
                )
                if m:
                    vals[axis] = _to_float(m.group(1))
            unit_m = (
                re.search(r'<PhysicalSizeUnit>(\w+)</PhysicalSizeUnit>', xml)
                or re.search(r'PhysicalSizeUnit="(\w+)"', xml)
            )
            unit = (unit_m.group(1) if unit_m else 'um').lower()
            if {'X', 'Y', 'Z'}.issubset(vals.keys()):
                if unit.startswith('nm'):
                    f = 1e-3
                elif unit.startswith('mm'):
                    f = 1e3
                elif unit.startswith('pm'):
                    f = 1e-6
                else:  # 'um' / 'µm' / 'micron'
                    f = 1.0
                return dict(
                    x_um=vals['X'] * f,
                    y_um=vals['Y'] * f,
                    z_um=vals['Z'] * f,
                    source=f'imaris-DatasetInfoXml ({unit})',
                )

    # Path 2/3: top-level or DataSet-level attributes.
    candidates = [h5.attrs]
    if 'DataSet' in h5 and isinstance(h5['DataSet'], h5py.Group):
        candidates.append(h5['DataSet'].attrs)
    if layout.get('mode') == 'imaris':
        rl0 = h5['DataSet/ResolutionLevel 0'] if 'DataSet/ResolutionLevel 0' in h5 else None
        if rl0 is not None:
            candidates.append(rl0.attrs)
            tp0_keys = layout.get('tp_keys') or []
            if tp0_keys:
                candidates.append(rl0[tp0_keys[0]].attrs)
    for attrs in candidates:
        for k_xy in ('physical_pixel_sizes', 'voxelsize_um_xy'):
            if k_xy in attrs:
                v = attrs[k_xy]
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    return dict(
                        x_um=float(v[0]),
                        y_um=float(v[1]),
                        z_um=float(attrs.get('voxelsize_um_z', v[0])),
                        source=f'HDF5 attribute {k_xy}',
                    )
        if 'voxel_size_um' in attrs:
            v = attrs['voxel_size_um']
            if isinstance(v, bytes):
                v = v.decode('utf-8', 'replace')
            try:
                d = json.loads(v) if isinstance(v, str) else json.loads(bytes(v))
                if all(k in d for k in ('x', 'y', 'z')):
                    return dict(
                        x_um=float(d['x']),
                        y_um=float(d['y']),
                        z_um=float(d['z']),
                        source='HDF5 attribute voxel_size_um',
                    )
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# TIFF writers (ZYX -> ImageJ hyperstack TIFF + optional voxel sizes)
# ---------------------------------------------------------------------------

def _to_uint16(arr: np.ndarray) -> np.ndarray:
    """Cast/clip a 3D stack to uint16 without doubling memory unnecessarily."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype.kind == 'f' or (arr.dtype.itemsize > 2 and arr.dtype.kind in ('u', 'i')):
        return np.clip(arr, 0, 65535).astype(np.uint16)
    return arr.astype(np.uint16, copy=False)


def save_timepoint(stack3d: np.ndarray, t_idx: int, channel: int,
                   voxel_xy_um: float | None = None,
                   voxel_z_um: float | None = None) -> str:
    """Save a 3D ZYX stack as ``t####_Channel <c>.tif`` (ImageJ hyperstack).

    When ``voxel_xy_um`` and ``voxel_z_um`` are supplied (micrometres)
    they are encoded into the ImageJ metadata so EXTRACT_METADATA can
    pick them up reliably instead of falling back to bogus 1.0 defaults
    lightsheet software commonly writes.
    """
    out_name = f"t{t_idx:04d}_Channel {channel}.tif"
    stack3d = _to_uint16(stack3d)
    kwargs = dict(imagej=True, metadata={'axes': 'ZYX'})
    if voxel_xy_um and voxel_xy_um > 0:
        res = (1.0 / voxel_xy_um, 1.0)
        kwargs['resolution'] = (res, res)
        if voxel_z_um and voxel_z_um > 0:
            kwargs['metadata']['spacing'] = float(voxel_z_um)
            kwargs['metadata']['unit'] = 'um'
    tifffile.imwrite(out_name, stack3d, **kwargs)
    print(f"  -> {out_name}  shape={stack3d.shape} dtype={stack3d.dtype}",
          flush=True)
    return out_name


# ---------------------------------------------------------------------------
# Channel helper (mirrors what the SPLIT_INPUT_FILE heredoc also uses for
# CZI / TIFF; duplicated here so the per-TP driver stays self-contained.)
# ---------------------------------------------------------------------------

def resolve_channel(n_channels_in_file: int, cfg: int) -> tuple[int, int]:
    """Return ``(channel_1based, ch_idx_0based)`` (auto-selects when cfg==0)."""
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


# ---------------------------------------------------------------------------
# Driver 1: single .ims / .h5 file
# ---------------------------------------------------------------------------

def split_ims(path: str, channel_cfg: int,
              n_workers: int = 1,
              voxel_emit_path: Path | None = None) -> None:
    """Stream-split an Imaris ``.ims`` (or generic 5D HDF5) one timepoint
    at a time. Mirrors the czifile "allocate only (Z,Y,X) per t" pattern
    so peak RAM stays at one timepoint per worker, not the whole
    hyperstack.

    Accepted Imaris layouts:

      - ``/DataSet/ResolutionLevel 0/TimePoint <t>/Channel <c>/Data`` (5D shape)
      - ``/DataSet/ImageData`` (flat 5D, ``TCZYX`` or ``CZYX``)
      - ``/data`` or ``/image`` (generic 5D)
    """
    print(f"Reading HDF5/Imaris metadata (no full-array load): {path}")
    f = h5py.File(path, 'r')
    ims_read_lock = threading.Lock()
    try:
        layout = _discover_ims_layout(f)
        print(
            f"  layout mode: {layout['mode']}   nT={layout['nT']}   nC={layout['nC']}   "
            f"shape (Z,Y,X) = ({layout['nZ']}, {layout['nY']}, {layout['nX']})"
        )

        channel, ch_idx = resolve_channel(layout['nC'], channel_cfg)
        print(f"  Using channel {channel} of {layout['nC']}")

        voxel = _read_ims_voxel_um(f, layout)
        if voxel:
            print(
                f"  Imaris voxel sizes (um): x={voxel['x_um']}  y={voxel['y_um']}  "
                f"z={voxel['z_um']}   [{voxel['source']}]"
            )
        else:
            print(
                "  WARNING: no Imaris voxel size found; relying on "
                "config.voxel_size.{x,y,z}_um."
            )

        def fetch_3d(t: int) -> np.ndarray:
            """Return ``(Z, Y, X)`` for the ``(t, ch_idx)`` slice."""
            if layout['mode'] == 'flat':
                with ims_read_lock:
                    arr5 = layout['root'][t, ch_idx, ...]
                if arr5.ndim == 2:
                    arr5 = arr5[None, ...]
                return arr5
            tp_node = layout['root'][layout['tp_keys'][t]]
            ch_node = tp_node[layout['ch_keys'][ch_idx]]
            with ims_read_lock:
                slab = ch_node['Data'][...]
            if slab.ndim == 5:
                _, _, nZ, nY, nX = slab.shape
            else:
                nZ, nY, nX = slab.shape
            return np.asarray(slab).reshape((nZ, nY, nX))

        def process_timepoint(t: int) -> int:
            stack = fetch_3d(t)
            save_timepoint(
                stack,
                t,
                channel,
                voxel_xy_um=(voxel['x_um'] if voxel else None),
                voxel_z_um=(voxel['z_um'] if voxel else None),
            )
            return t

        print(
            f"Splitting {layout['nT']} timepoints (channel {channel} of {layout['nC']}) "
            f"with {n_workers} worker thread(s)..."
        )

        workers = min(n_workers, 4) if (voxel is None) else 1
        if workers == 1:
            for t in range(layout['nT']):
                process_timepoint(t)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process_timepoint, t): t for t in range(layout['nT'])}
                for fut in concurrent.futures.as_completed(futs):
                    fut.result()

        if voxel and voxel_emit_path:
            Path(voxel_emit_path).write_text(json.dumps(voxel))
            print(f"  Wrote voxel size sidecar: {voxel_emit_path}")
    finally:
        f.close()


# ---------------------------------------------------------------------------
# Driver 2: per-timepoint Bio-Formats shrapnel folder
# ---------------------------------------------------------------------------

def split_h5_per_timepoint(directory: str | Path, channel_cfg: int,
                           voxel_emit_path: Path | None = None) -> None:
    """Read a directory of per-timepoint / per-channel ``.h5`` files
    (Bio-Formats "split into timepoints" exporter convention:

        ``<prefix>--C00--T00000.h5``
        ``<prefix>--C01--T00000.h5``
        ...

    ) into one ``t####_Channel <c>.tif`` per timepoint.

    TPs without files for the requested channel are skipped with a
    warning (an embryo can leave the FOV mid-acquisition).
    """
    directory = Path(directory)
    print(f"Reading per-timepoint HDF5 files in: {directory}")

    by_t: dict[int, dict[int, Path]] = {}
    for p in sorted(directory.iterdir()):
        m = _BF_PER_TP_PATTERN.search(p.name)
        if not m:
            continue
        c = int(m.group(1))
        t = int(m.group(2))
        by_t.setdefault(t, {})[c] = p

    if not by_t:
        raise RuntimeError(
            f"No files matching '<prefix>--C##--T#####.h5' found in {directory}. "
            f"If your data is one big .ims file, point `input.directory` at its "
            f"parent directory and the pipeline will auto-detect it."
        )

    n_channels = max(max(cs.keys()) + 1 for cs in by_t.values())
    channel, ch_idx = resolve_channel(n_channels, channel_cfg)
    print(f"  Using channel {channel} of {n_channels}   "
          f"({len(by_t)} timepoint(s) with data)")

    voxel_xy_um = voxel_z_um = None
    voxel_source = None
    sample_h5 = next(iter(next(iter(by_t.values())).values()))
    try:
        with h5py.File(sample_h5, 'r') as h5:
            layout = _discover_ims_layout(h5)
            voxel = _read_ims_voxel_um(h5, layout)
            if voxel:
                voxel_xy_um = voxel['x_um']
                voxel_z_um = voxel['z_um']
                voxel_source = voxel['source']
    except Exception as exc:
        # Layout discovery may legitimately fail on a per-TP file: each
        # of those is just the (T, C) slice, not a full 5D array.
        print(f"  (could not read voxel metadata from sample: {exc})")

    missing = [t for t, cs in by_t.items() if ch_idx not in cs]
    if missing:
        print(
            f"  WARNING: timepoints {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''} have no C{channel} .h5 file "
            f"— will be skipped."
        )

    print(f"  Writing {len(by_t) - len(missing)} per-timepoint TIFF(s)...")
    for t in sorted(by_t):
        if ch_idx not in by_t[t]:
            continue
        with h5py.File(by_t[t][ch_idx], 'r') as h5:
            dsets = []
            for k in ('Data', 'data', 'image'):
                if k in h5 and isinstance(h5[k], h5py.Dataset):
                    dsets.append(h5[k])
                    break
            if not dsets:
                def _visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        dsets.append(obj)
                h5.visititems(_visit)
            if not dsets:
                print(
                    f"WARNING: no dataset found in {by_t[t][ch_idx].name} "
                    f"— skipping t={t}",
                    flush=True,
                )
                continue
            arr = dsets[0][...]
            if arr.ndim == 5:
                arr = arr[0, 0, ...]
            elif arr.ndim == 2:
                arr = arr[None, ...]
        save_timepoint(arr, t, channel,
                       voxel_xy_um=voxel_xy_um, voxel_z_um=voxel_z_um)

    if voxel_source and voxel_emit_path:
        Path(voxel_emit_path).write_text(
            json.dumps({
                'x_um': voxel_xy_um,
                'y_um': voxel_xy_um,
                'z_um': voxel_z_um,
                'source': f'{voxel_source} (sampled from per-TP file)',
            })
        )
    print("  Per-timepoint .h5 split complete.")


# ---------------------------------------------------------------------------
# CLI driver (used when called as `python3 -m _ims_reader ...`)
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys
    p = argparse.ArgumentParser(
        description="Imaris / HDF5 input splitter (split a .ims/.h5/.hdf5 file "
                    "or a folder of per-TP .h5 files into t####_Channel*.tif).",
    )
    p.add_argument('mode', choices=('split-ims', 'split-per-tp-h5'))
    p.add_argument('path', help='Path to .ims/.h5/.hdf5 (mode split-ims) '
                                'or to the directory of --C##--T#####.h5 files '
                                '(mode split-per-tp-h5).')
    p.add_argument('--channel', type=int, default=1,
                   help='1-based channel index (0 = auto-detect for single-channel).')
    p.add_argument('--workers', type=int,
                   default=lambda: max(1, int(os.environ.get('NXF_TASK_CPUS', 1) or 1)),
                   help='Max worker threads for I/O parallelism (default: NXF_TASK_CPUS).')
    p.add_argument('--voxel-sidecar', default='voxel_size.json',
                   help='Path to write the voxel-size JSON sidecar.')

    args = p.parse_args(argv)
    try:
        if args.mode == 'split-ims':
            split_ims(args.path, args.channel, args.workers,
                      Path(args.voxel_sidecar))
        else:
            split_h5_per_timepoint(args.path, args.channel,
                                   Path(args.voxel_sidecar))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_main())
