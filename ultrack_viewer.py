# =============================================================================
# napari-based 4D Embryo Viewer — Orientation, Sphere Fit & Ingression Analysis
# =============================================================================
#
# PURPOSE:
#   Interactive 4D (3D + time) visualisation and annotation of light-sheet
#   microscopy nuclear-tracking data (TrackMate or ultrack exports) for medaka
#   and zebrafish gastrulation analysis.
#
# REPLACES:  R/Shiny + plotly orientation app (too slow for frame-by-frame
#            scrubbing of millions of spots).
#
# FEATURES:
#   1. GPU-accelerated 4D point cloud (millions of nuclei, smooth scrubbing)
#   2. Progressive track formation as time advances
#   3. Interactive landmark picking (Animal Pole, Dorsal)
#   4. Robust sphere fitting for thin SPIM caps
#   5. Spherical coordinate system (depth, latitude, longitude)
#   6. ROI rectangle selection (draw in 2D, selects through full 3D volume)
#   7. Ingression analysis: radial velocity, ingression flagging & colouring
#   8. Export enriched CSVs → feeds R analysis pipeline
#
# USAGE:
#   python embryo_viewer.py medaka_25082025_combined_spots.csv  \
#                           --tracks medaka_25082025_combined_tracks.csv
#
# INSTALLATION:
#   pip install "napari[all]" scipy pandas
#
# OUTPUT (via Export button):
#   oriented_spots.csv  — all spots with spherical coords + IN_ROI flag
#   sphere_params.csv   — fitted sphere centre, radius
#
# PIPELINE:
#   → Step 1: THIS SCRIPT
#     Step 2: trackmate_filter_and_validate.R  (R)
#     Step 3: trackmate_analysis.R             (R)
#
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path

# ── Wayland / DPI fixes for Fedora GNOME ──────────────────────────────────
# Must be set before any Qt import
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ["QT_QPA_PLATFORM"] = "wayland"
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_FONT_DPI", "96")
# Suppress harmless Qt C++ warnings (QSocketNotifier, DPI messages)
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
#  napari import (deferred so --help works without Qt)
# ---------------------------------------------------------------------------
_napari = None
_magicgui = None


def _import_napari():
    global _napari, _magicgui
    if _napari is None:
        import napari as _nap
        import magicgui as _mg

        _napari = _nap
        _magicgui = _mg
    return _napari, _magicgui


# ############################################################################
#                            DATA LOADING
# ############################################################################


def load_trackmate_csv(path: str | Path) -> pd.DataFrame:
    """Load a TrackMate spots/tracks CSV, auto-skipping metadata rows."""
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)

    # TrackMate CSVs have 3 metadata rows after the header.
    # Detect by checking if the first value of POSITION_X (or similar
    # numeric column) is non-numeric.
    num_cols = [
        c
        for c in df.columns
        if "POSITION" in c or c in ("FRAME", "QUALITY", "RADIUS", "ID", "TRACK_ID")
    ]
    if len(num_cols) > 0:
        test_col = num_cols[0]
        try:
            float(df[test_col].iloc[0])
        except (ValueError, TypeError):
            # Drop first 3 rows (descriptions + units)
            df = df.iloc[3:].reset_index(drop=True)

    # Convert numeric columns
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_ultrack_csv(path: str | Path) -> pd.DataFrame:
    """Load an ultrack tracks CSV and map columns to internal format.

    Ultrack exports columns: track_id, t, z, y, x, id, parent_track_id, parent_id.
    Maps to: TRACK_ID, FRAME, POSITION_Z, POSITION_Y, POSITION_X.
    """
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)

    col_map = {
        "track_id": "TRACK_ID",
        "t": "FRAME",
        "z": "POSITION_Z",
        "y": "POSITION_Y",
        "x": "POSITION_X",
    }
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        raise ValueError(f"Not a valid ultrack CSV — missing columns: {missing}")

    df = df.rename(columns=col_map)
    for c in ["POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME", "TRACK_ID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["FRAME"] = df["FRAME"].astype(int)

    # Keep parent info if present
    if "parent_track_id" in df.columns:
        df["PARENT_TRACK_ID"] = pd.to_numeric(df["parent_track_id"], errors="coerce")

    return df


def _detect_csv_format(path: str | Path) -> str:
    """Detect whether a CSV is ultrack or TrackMate format.

    Returns 'ultrack' or 'trackmate'.
    """
    path = Path(path)
    df_head = pd.read_csv(path, nrows=5, low_memory=False)
    ultrack_cols = {"track_id", "t", "z", "y", "x"}
    if ultrack_cols.issubset(set(df_head.columns)):
        return "ultrack"
    return "trackmate"


def load_spots(path: str | Path) -> pd.DataFrame:
    """Load spots CSV and return cleaned DataFrame.

    Auto-detects ultrack vs TrackMate format.
    """
    fmt = _detect_csv_format(path)
    if fmt == "ultrack":
        df = load_ultrack_csv(path)
    else:
        df = load_trackmate_csv(path)

    required = ["POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)
    df["FRAME"] = df["FRAME"].astype(int)
    return df


# ############################################################################
#                         SPHERE FITTING (cap-aware)
# ############################################################################


def extract_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_n: int = 40,
    quantile: float = 0.95,
) -> dict:
    """Extract outer and inner surface points from a SPIM thin slab.

    Returns dict with 'upper' and 'lower' surface point arrays (Nx3)
    and the identity of the thin axis.
    """
    extents = np.array([np.ptp(x), np.ptp(y), np.ptp(z)])
    thin_axis = int(np.argmin(extents))

    # Re-order so thin axis is last
    axes = [x, y, z]
    order = [i for i in range(3) if i != thin_axis] + [thin_axis]
    u, v, w = axes[order[0]], axes[order[1]], axes[order[2]]

    # Grid on (u, v)
    u_edges = np.linspace(u.min() - 1e-6, u.max() + 1e-6, grid_n + 1)
    v_edges = np.linspace(v.min() - 1e-6, v.max() + 1e-6, grid_n + 1)
    u_bin = np.digitize(u, u_edges) - 1
    v_bin = np.digitize(v, v_edges) - 1
    u_bin = np.clip(u_bin, 0, grid_n - 1)
    v_bin = np.clip(v_bin, 0, grid_n - 1)
    cell_id = u_bin * grid_n + v_bin

    upper_pts = []
    lower_pts = []
    for cid in np.unique(cell_id):
        mask = cell_id == cid
        if mask.sum() < 3:
            continue
        um, vm, wm = u[mask], v[mask], w[mask]
        # Upper surface: high-w
        q_hi = np.quantile(wm, quantile)
        sel_hi = wm >= q_hi
        upper_pts.append([um[sel_hi].mean(), vm[sel_hi].mean(), wm[sel_hi].mean()])
        # Lower surface: low-w
        q_lo = np.quantile(wm, 1 - quantile)
        sel_lo = wm <= q_lo
        lower_pts.append([um[sel_lo].mean(), vm[sel_lo].mean(), wm[sel_lo].mean()])

    upper = np.array(upper_pts)
    lower = np.array(lower_pts)

    # Map back to (x, y, z) order
    inv = [0, 0, 0]
    for new_i, orig_i in enumerate(order):
        inv[orig_i] = new_i

    def reorder(arr):
        return arr[:, inv]

    return {
        "upper": reorder(upper),
        "lower": reorder(lower),
        "thin_axis": thin_axis,
        "order": order,
    }


def _cap_initial_guess(pts: np.ndarray, thin_axis: int, side: str = "upper") -> dict:
    """Spherical-cap geometry initial guess for NLS.

    R = (a² + h²) / (2h)  where  a = base radius, h = cap height.
    """
    cx, cy, cz = pts.mean(axis=0)
    long_axes = [i for i in range(3) if i != thin_axis]
    center_lat = np.array([cx, cy, cz])

    # Lateral radius
    a = np.sqrt(
        (pts[:, long_axes[0]] - center_lat[long_axes[0]]) ** 2
        + (pts[:, long_axes[1]] - center_lat[long_axes[1]]) ** 2
    ).max()

    # Cap height along thin axis
    h = np.ptp(pts[:, thin_axis])
    h = max(h, 1.0)

    R = (a**2 + h**2) / (2 * h)

    if side == "upper":
        center_lat[thin_axis] = pts[:, thin_axis].max() - R
    else:
        center_lat[thin_axis] = pts[:, thin_axis].min() + R

    return {"center": center_lat, "R": R, "a": a, "h": h}


def _fit_sphere_nls(pts: np.ndarray, cx0, cy0, cz0, R0) -> dict:
    """Fit sphere by minimising sum((||p - c|| - R)^2)."""

    def obj(par):
        d = np.sqrt(
            (pts[:, 0] - par[0]) ** 2
            + (pts[:, 1] - par[1]) ** 2
            + (pts[:, 2] - par[2]) ** 2
        )
        return np.sum((d - par[3]) ** 2)

    p0 = np.array([cx0, cy0, cz0, R0], dtype=float)

    r1 = minimize(
        obj,
        p0,
        method="Nelder-Mead",
        options={"maxiter": 80000, "xatol": 1e-6, "fatol": 1e-10},
    )
    r2 = minimize(
        obj,
        p0,
        method="L-BFGS-B",
        bounds=[(None, None)] * 3 + [(1.0, None)],
        options={"maxiter": 50000},
    )

    best = r2 if r2.fun < r1.fun else r1
    cx, cy, cz, R = best.x
    R = abs(R)
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 + (pts[:, 2] - cz) ** 2)
    return {
        "center": np.array([cx, cy, cz]),
        "radius": R,
        "rmse": float(np.sqrt(np.mean((d - R) ** 2))),
        "residuals": d - R,
    }


def fit_sphere(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_points: int = 100_000,
    grid_n: int = 40,
    surface_quantile: float = 0.9125,
) -> dict:
    """Robust sphere fitting for thin SPIM caps.

    Parameters
    ----------
    x, y, z : position arrays (all nuclei)
    max_points : subsample for speed
    grid_n : grid resolution for surface extraction
    surface_quantile : top-% of thin axis to keep per grid cell

    Returns
    -------
    dict with center (3,), radius, rmse, coverage, surface_used, cap_height
    """
    if len(x) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    surfs = extract_surface(x, y, z, grid_n=grid_n, quantile=surface_quantile)

    # Fit upper surface
    g_hi = _cap_initial_guess(surfs["upper"], surfs["thin_axis"], "upper")
    fit_hi = _fit_sphere_nls(surfs["upper"], *g_hi["center"], g_hi["R"])

    # Fit lower surface
    g_lo = _cap_initial_guess(surfs["lower"], surfs["thin_axis"], "lower")
    fit_lo = _fit_sphere_nls(surfs["lower"], *g_lo["center"], g_lo["R"])

    # Pick the better fit
    if fit_hi["rmse"] <= fit_lo["rmse"]:
        fit, side, guess = fit_hi, "upper", g_hi
    else:
        fit, side, guess = fit_lo, "lower", g_lo

    # Coverage estimate (fraction of sphere-surface bins hit)
    d = np.sqrt(
        (x - fit["center"][0]) ** 2
        + (y - fit["center"][1]) ** 2
        + (z - fit["center"][2]) ** 2
    )
    R = fit["radius"]
    on_surf = (d > R * 0.5) & (d < R * 1.5)
    coverage = 0.0
    if on_surf.sum() > 10:
        dx = x[on_surf] - fit["center"][0]
        dy = y[on_surf] - fit["center"][1]
        dz = z[on_surf] - fit["center"][2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = np.maximum(r, 1e-10)
        theta = np.arccos(np.clip(-dy / r, -1, 1))  # AP at -Y
        phi = np.arctan2(dz, dx)
        tb = np.digitize(theta, np.linspace(0, np.pi, 13))
        pb = np.digitize(phi, np.linspace(-np.pi, np.pi, 25))
        bins = set(zip(tb.tolist(), pb.tolist()))
        coverage = len(bins) / (12 * 24)

    fit["coverage"] = coverage
    fit["surface_used"] = side
    fit["cap_height"] = guess["h"]
    fit["cap_base_radius"] = guess["a"]
    return fit


# ############################################################################
#                     ORIENTATION & SPHERICAL COORDS
# ############################################################################


def rotation_matrix(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector u to unit vector v."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    if np.allclose(u, v):
        return np.eye(3)
    if np.allclose(u, -v):
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(u, v)
    ax = ax / np.linalg.norm(ax)
    ang = np.arccos(np.clip(u @ v, -1, 1))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def orient_embryo(
    pos: np.ndarray, ap: np.ndarray, dorsal: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Rotate embryo so AP → -Y (top in napari 3D view), Dorsal → +X.

    Parameters
    ----------
    pos : (N, 3) array of positions
    ap : (3,) animal pole coordinate
    dorsal : (3,) dorsal landmark

    Returns
    -------
    (N, 3) oriented positions, dict of transform params
    """
    mid = pos.mean(axis=0)
    pos_c = pos - mid

    # Step 1: AP → -Y (napari +Y points downward in 3D, so -Y = top)
    R1 = rotation_matrix(ap - mid, np.array([0.0, -1.0, 0.0]))
    pos_r1 = (R1 @ pos_c.T).T

    # Step 2: Dorsal → +X (preserve +Y)
    d_r1 = R1 @ (dorsal - mid)
    d_xz = np.array([d_r1[0], 0.0, d_r1[2]])
    if np.linalg.norm(d_xz) < 1e-8:
        R2 = np.eye(3)
    else:
        R2 = rotation_matrix(d_xz, np.array([1.0, 0.0, 0.0]))

    pos_f = (R2 @ pos_r1.T).T
    return pos_f, {"center": mid, "R1": R1, "R2": R2}


def compute_spherical_coords(pos: np.ndarray, center: np.ndarray, R: float) -> dict:
    """Spherical coords relative to fitted sphere.

    Returns dict with keys:
      RADIAL_DIST, SPHERICAL_DEPTH, SPHERICAL_DEPTH_NORM,
      THETA_DEG (0=AP, 90=equator, 180=VP),
      PHI_DEG (0=dorsal +X direction)
    """
    d = pos - center
    r = np.sqrt(np.sum(d**2, axis=1))
    r_safe = np.maximum(r, 1e-10)
    return {
        "RADIAL_DIST": r,
        "SPHERICAL_DEPTH": R - r,
        "SPHERICAL_DEPTH_NORM": (R - r) / R,
        "THETA_DEG": np.degrees(np.arccos(np.clip(-d[:, 1] / r_safe, -1, 1))),
        "PHI_DEG": np.degrees(np.arctan2(d[:, 2], d[:, 0])),
    }


# ############################################################################
#                         SPHERE MESH (for napari)
# ############################################################################


def make_sphere_cap_mesh(
    center, radius, data_points, padding_deg=10, n_lat=30, n_lon=60
):
    """Create mesh for the sphere cap overlapping with data.

    Only generates the angular portion of the sphere near the actual
    nuclei, so a thin SPIM cap doesn't produce a giant full sphere.

    Parameters
    ----------
    center : (3,) sphere center (x, y, z)
    radius : sphere radius
    data_points : (N, 3) positions of nuclei (x, y, z)
    padding_deg : extra angular margin around the data
    """
    d = data_points - center
    r_safe = np.maximum(np.sqrt(np.sum(d**2, axis=1)), 1e-10)

    # Spherical: theta from -Y axis (AP at top in napari), phi in XZ
    theta = np.arccos(np.clip(-d[:, 1] / r_safe, -1, 1))
    phi = np.arctan2(d[:, 2], d[:, 0])

    pad = np.radians(padding_deg)
    theta_min = max(0, np.percentile(theta, 2) - pad)
    theta_max = min(np.pi, np.percentile(theta, 98) + pad)
    phi_min = np.percentile(phi, 2) - pad
    phi_max = np.percentile(phi, 98) + pad

    tg, pg = np.meshgrid(
        np.linspace(theta_min, theta_max, n_lat),
        np.linspace(phi_min, phi_max, n_lon),
        indexing="ij",
    )

    x = center[0] + radius * np.sin(tg) * np.cos(pg)
    y = center[1] - radius * np.cos(tg)  # AP at -Y (top in napari)
    z = center[2] + radius * np.sin(tg) * np.sin(pg)

    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            v0 = i * n_lon + j
            v1 = v0 + 1
            v2 = v0 + n_lon
            v3 = v2 + 1
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])

    return verts, np.array(faces)


# ############################################################################
#                     INGRESSION ANALYSIS
# ############################################################################


def compute_radial_velocity(
    df: pd.DataFrame, center: np.ndarray, smooth_window: int = 5
) -> pd.DataFrame:
    """Compute per-spot radial velocity with smoothing and enhanced track stats.

    Negative = moving inward (ingressing).
    Positive = moving outward.

    Parameters
    ----------
    df : spots DataFrame with POSITION_X/Y/Z, TRACK_ID, FRAME
    center : (3,) sphere center
    smooth_window : rolling-window size for smoothed V_r (odd, ≥3)

    Returns
    -------
    DataFrame with new columns:
      RADIAL_DIST_TO_CENTER   – distance from sphere centre
      RADIAL_VELOCITY         – instantaneous V_r (µm/frame)
      RADIAL_VELOCITY_SMOOTH  – smoothed V_r (rolling mean per track)
      TRACK_MEDIAN_RADIAL_VEL – per-track median of smoothed V_r (robust)
      TRACK_MEAN_RADIAL_VEL   – per-track mean of smoothed V_r
      TRACK_INWARD_FRACTION   – fraction of track points with V_r < 0
      TRACK_MAX_INWARD_VEL    – most negative V_r per track (peak inward)
    """
    x = df["POSITION_X"].values - center[0]
    y = df["POSITION_Y"].values - center[1]
    z = df["POSITION_Z"].values - center[2]
    rdist = np.sqrt(x**2 + y**2 + z**2)
    df["RADIAL_DIST_TO_CENTER"] = rdist

    nan_cols = {
        "RADIAL_VELOCITY": np.nan,
        "RADIAL_VELOCITY_SMOOTH": np.nan,
        "TRACK_MEDIAN_RADIAL_VEL": np.nan,
        "TRACK_MEAN_RADIAL_VEL": np.nan,
        "TRACK_INWARD_FRACTION": np.nan,
        "TRACK_MAX_INWARD_VEL": np.nan,
    }
    if "TRACK_ID" not in df.columns:
        for col in nan_cols:
            df[col] = np.full(len(df), np.nan)
        return df

    # --- Instantaneous radial velocity ---
    sorted_idx = df.sort_values(["TRACK_ID", "FRAME"]).index
    tid = df.loc[sorted_idx, "TRACK_ID"].values
    r = rdist[sorted_idx]
    f = df.loc[sorted_idx, "FRAME"].values.astype(float)

    dr = np.diff(r, prepend=np.nan)
    dt = np.diff(f, prepend=-9999)
    same_track = np.diff(tid.astype(float), prepend=-9999) == 0
    # Use np.maximum to avoid divide-by-zero; results where dt<=0 are
    # masked to NaN by the np.where condition anyway.
    dt_safe = np.maximum(dt, 1e-12)
    vel = np.where(same_track & (dt > 0), dr / dt_safe, np.nan)

    rv = np.full(len(df), np.nan)
    rv[sorted_idx] = vel
    df["RADIAL_VELOCITY"] = rv

    # --- Smoothed V_r: rolling mean within each track ---
    sw = max(3, smooth_window | 1)  # ensure odd ≥3
    tmp = pd.DataFrame(
        {
            "TRACK_ID": df["TRACK_ID"],
            "FRAME": df["FRAME"],
            "RV": rv,
            "_idx": df.index,
        }
    ).sort_values(["TRACK_ID", "FRAME"])

    tmp["RV_SMOOTH"] = tmp.groupby("TRACK_ID")["RV"].transform(
        lambda s: s.rolling(sw, center=True, min_periods=1).mean()
    )
    rv_smooth = np.full(len(df), np.nan)
    rv_smooth[tmp["_idx"].values] = tmp["RV_SMOOTH"].values
    df["RADIAL_VELOCITY_SMOOTH"] = rv_smooth

    # --- Per-track summary statistics (based on smoothed V_r) ---
    valid_mask = ~np.isnan(rv_smooth)
    if valid_mask.any():
        sub = df.loc[valid_mask, ["TRACK_ID", "RADIAL_VELOCITY_SMOOTH"]]
        grp = sub.groupby("TRACK_ID")["RADIAL_VELOCITY_SMOOTH"]
        med_map = grp.median()
        mean_map = grp.mean()
        inward_frac_map = grp.agg(lambda s: (s < 0).mean())
        max_inward_map = grp.min()  # most negative = strongest inward

        df["TRACK_MEDIAN_RADIAL_VEL"] = df["TRACK_ID"].map(med_map)
        df["TRACK_MEAN_RADIAL_VEL"] = df["TRACK_ID"].map(mean_map)
        df["TRACK_INWARD_FRACTION"] = df["TRACK_ID"].map(inward_frac_map)
        df["TRACK_MAX_INWARD_VEL"] = df["TRACK_ID"].map(max_inward_map)
    else:
        for col in [
            "TRACK_MEDIAN_RADIAL_VEL",
            "TRACK_MEAN_RADIAL_VEL",
            "TRACK_INWARD_FRACTION",
            "TRACK_MAX_INWARD_VEL",
        ]:
            df[col] = np.nan

    return df


def flag_ingressing(
    df: pd.DataFrame,
    threshold: float = -0.5,
    min_track_frames: int = 5,
    min_inward_fraction: float = 0.5,
) -> pd.DataFrame:
    """Flag tracks as ingressing using embryology-informed multi-criteria detection.

    During gastrulation in medaka/zebrafish, ingressing cells move
    radially inward through the blastoderm, transitioning from the
    epiblast to the hypoblast (or mesendoderm). True ingression is
    characterised by:
      - Sustained inward radial motion (not transient fluctuation)
      - Net radial displacement toward the sphere centre
      - Consistently negative radial velocity over most of the track

    Criteria — a track is flagged INGRESSING if ALL of:
      1. Track has ≥ *min_track_frames* time points
      2. Track **median** smoothed V_r ≤ *threshold*
      3. ≥ *min_inward_fraction* of track points have V_r < 0
      4. Net radial displacement is inward (first→last distance decreases)
      5. Track has ≥ 3 consecutive frames of inward movement (sustained)

    Also computes per-track columns for downstream analysis:
      INGRESSION_ONSET_FRAME  – first frame of sustained inward movement
      TRACK_NET_RADIAL_DISP   – net change in radial distance (neg = inward)
      TRACK_SUSTAINED_INWARD  – max consecutive inward frames

    Parameters
    ----------
    df : spots DataFrame with TRACK_MEDIAN_RADIAL_VEL, TRACK_INWARD_FRACTION
    threshold : µm/frame – tracks with median V_r ≤ threshold → ingressing
    min_track_frames : minimum track length
    min_inward_fraction : minimum fraction of inward-moving points

    Returns
    -------
    DataFrame with INGRESSING (bool) column + enrichment columns
    """
    has_med = "TRACK_MEDIAN_RADIAL_VEL" in df.columns
    has_mean = "TRACK_MEAN_RADIAL_VEL" in df.columns
    if "TRACK_ID" not in df.columns or (not has_med and not has_mean):
        df["INGRESSING"] = False
        return df

    # Use median if available (more robust), else fall back to mean
    vel_col = "TRACK_MEDIAN_RADIAL_VEL" if has_med else "TRACK_MEAN_RADIAL_VEL"
    vel = df[vel_col].values

    track_len = df.groupby("TRACK_ID")["FRAME"].transform("count")
    long_enough = track_len >= min_track_frames

    # Inward fraction criterion
    if "TRACK_INWARD_FRACTION" in df.columns:
        inward_ok = df["TRACK_INWARD_FRACTION"].values >= min_inward_fraction
    else:
        inward_ok = np.ones(len(df), dtype=bool)

    # ── Net radial displacement criterion ──
    # True ingressing cells end up closer to the sphere centre than
    # they started (epiblast→hypoblast transition).
    net_disp = np.full(len(df), np.nan)
    sustained_inward = np.zeros(len(df), dtype=int)
    onset_frame = np.full(len(df), np.nan)

    if "RADIAL_DIST_TO_CENTER" in df.columns:
        sorted_df = df.sort_values(["TRACK_ID", "FRAME"])

        # Vectorised net radial displacement: last - first per track
        first_r = sorted_df.groupby("TRACK_ID")["RADIAL_DIST_TO_CENTER"].transform(
            "first"
        )
        last_r = sorted_df.groupby("TRACK_ID")["RADIAL_DIST_TO_CENTER"].transform(
            "last"
        )
        net_per_spot = (last_r - first_r).values
        net_disp[sorted_df.index] = net_per_spot

        # Sustained inward: max consecutive frames with V_r < 0 (per track)
        # This requires run-length logic so we iterate per track
        if "RADIAL_VELOCITY_SMOOTH" in sorted_df.columns:
            for tid, grp in sorted_df.groupby("TRACK_ID"):
                idx = grp.index
                vr = grp["RADIAL_VELOCITY_SMOOTH"].values
                max_run, cur_run, first_inward = 0, 0, np.nan
                for j, v in enumerate(vr):
                    if not np.isnan(v) and v < 0:
                        cur_run += 1
                        if cur_run == 1 and np.isnan(first_inward):
                            first_inward = grp["FRAME"].values[j]
                        if cur_run > max_run:
                            max_run = cur_run
                    else:
                        if cur_run < 3:  # reset onset if run was too short
                            first_inward = np.nan
                        cur_run = 0
                sustained_inward[idx] = max_run
                onset_frame[idx] = first_inward

    df["TRACK_NET_RADIAL_DISP"] = net_disp
    df["TRACK_SUSTAINED_INWARD"] = sustained_inward
    df["INGRESSION_ONSET_FRAME"] = onset_frame

    # Net displacement is inward — require meaningful movement, not noise
    net_inward = np.where(
        np.isnan(net_disp),
        True,  # if no data, don't block
        net_disp < -0.5,  # at least 0.5 µm net inward
    )

    # Sustained movement: scale with track length — at least 15% of frames
    # or 3, whichever is larger.  Short noise bursts get filtered out while
    # long tracks need proportionally more consecutive inward evidence.
    min_sustained = np.maximum(3, (track_len * 0.15).astype(int))
    sustained_ok = sustained_inward >= min_sustained

    df["INGRESSING"] = (
        long_enough
        & (~np.isnan(vel))
        & (vel <= threshold)
        & inward_ok
        & net_inward
        & sustained_ok
    )

    # ── Continuous ingression score (0–1) for visualisation ──
    # Combines all criteria into a smooth score so the visualisation can
    # show confidence rather than binary on/off.
    abs_thresh = max(abs(threshold), 0.01)
    s_vel = np.clip((threshold - vel) / abs_thresh, 0, 2) / 2
    s_vel = np.nan_to_num(s_vel, nan=0.0)

    if "TRACK_INWARD_FRACTION" in df.columns:
        s_inward = np.clip(df["TRACK_INWARD_FRACTION"].values, 0, 1)
    else:
        s_inward = np.full(len(df), 0.5)

    s_sustained = np.clip(sustained_inward / np.maximum(track_len.values, 1), 0, 1)

    disp_for_score = np.where(np.isnan(net_disp), 0.0, -net_disp)
    s_disp = np.clip(
        disp_for_score / np.maximum(track_len.values.astype(float), 1), 0, 1
    )

    raw_score = 0.30 * s_vel + 0.20 * s_inward + 0.25 * s_sustained + 0.25 * s_disp
    # Zero out score for non-ingressing tracks to keep it interpretable
    raw_score[~df["INGRESSING"].values] = 0.0
    df["INGRESSION_SCORE"] = raw_score

    return df


def auto_ingression_threshold(df: pd.DataFrame) -> tuple[float, str]:
    """Auto-detect ingression V_r threshold using Otsu's method.

    Finds the threshold on the distribution of per-track median V_r
    that maximises between-class variance (ingressing vs non-ingressing).

    Returns
    -------
    (threshold, description_string)
    """
    col = "TRACK_MEDIAN_RADIAL_VEL"
    if col not in df.columns or "TRACK_ID" not in df.columns:
        return -0.5, "default (no data)"

    # Unique per-track values
    track_stats = df.dropna(subset=[col]).groupby("TRACK_ID")[col].first()
    vals = track_stats.values
    vals = vals[~np.isnan(vals)]

    if len(vals) < 20:
        return float(np.median(vals)) if len(vals) > 0 else -0.5, "median (few tracks)"

    # Otsu 1D: maximise between-class variance
    n_bins = 200
    hist, edges = np.histogram(vals, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return -0.5, "default (empty)"

    best_thresh, best_var = centres[0], 0.0
    w0, sum0 = 0, 0.0
    sum_total = float((hist * centres).sum())

    for i in range(n_bins):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * centres[i]
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = float(centres[i])

    # Sanity: for ingression we expect negative thresholds
    best_thresh = min(best_thresh, -0.005)
    p_below = float((vals <= best_thresh).mean() * 100)

    desc = f"Otsu={best_thresh:.3f} µm/f ({p_below:.0f}% of tracks below)"
    return best_thresh, desc


# ############################################################################
#                        TRACKS → napari format
# ############################################################################


def spots_to_napari_tracks(
    df: pd.DataFrame,
    max_tracks: int = 5000,
    min_length: int = 3,
    frame_min: int | None = None,
    frame_max: int | None = None,
) -> np.ndarray | None:
    """Convert spots DataFrame to napari Tracks format.

    napari Tracks format: [track_id, t, z, y, x] — 5 columns for 3D+T.

    Parameters
    ----------
    frame_min, frame_max : optional frame range filter for tracks.
        Only track points within [frame_min, frame_max] are included.
        Tracks that have no points in range are dropped entirely.
    """
    if "TRACK_ID" not in df.columns:
        return None

    df_t = df.dropna(subset=["TRACK_ID"])
    if df_t.empty:
        return None

    # Frame range filter
    if frame_min is not None:
        df_t = df_t[df_t["FRAME"] >= frame_min]
    if frame_max is not None:
        df_t = df_t[df_t["FRAME"] <= frame_max]
    if df_t.empty:
        return None

    # Filter by min length (within the frame window)
    track_len = df_t.groupby("TRACK_ID")["FRAME"].transform("count")
    df_t = df_t[track_len >= min_length]

    # Subsample tracks if needed
    tids = df_t["TRACK_ID"].unique()
    if len(tids) > max_tracks:
        rng = np.random.default_rng(42)
        tids = rng.choice(tids, max_tracks, replace=False)
        df_t = df_t[df_t["TRACK_ID"].isin(tids)]

    df_t = df_t.sort_values(["TRACK_ID", "FRAME"])

    # napari tracks: [ID, t, z, y, x]
    tracks = np.column_stack(
        [
            df_t["TRACK_ID"].values,
            df_t["FRAME"].values,
            df_t["POSITION_Z"].values,
            df_t["POSITION_Y"].values,
            df_t["POSITION_X"].values,
        ]
    )
    return tracks


# ############################################################################
#                       CROSS-SECTION VIEWER
# ############################################################################


class CrossSectionViewer:
    """A separate window showing a 2D cross-section through the 3D point cloud.

    The user selects a cutting axis (X, Y, or Z) and a position along that
    axis.  All spots within a configurable slab thickness around that position
    are projected onto the remaining two axes as a 2D scatter, colored by
    the selected colour mode.

    Mirrors the main viewer's controls: colour modes, display %, tracks
    within the slab, sphere intersection, time slider.
    """

    # Mapping from cutting axis to the 2 display axes
    _AXES_MAP = {
        "X": ("POSITION_Z", "POSITION_Y"),  # (vertical, horizontal)
        "Y": ("POSITION_Z", "POSITION_X"),
        "Z": ("POSITION_Y", "POSITION_X"),
    }
    _AXES_LABELS = {
        "X": ("Z", "Y"),
        "Y": ("Z", "X"),
        "Z": ("Y", "X"),
    }
    # Maps POSITION_* column names to xyz index for coordinate lookups
    _COL_TO_IDX = {"POSITION_X": 0, "POSITION_Y": 1, "POSITION_Z": 2}

    @property
    def df(self):
        """Always read the parent's current spots DataFrame (never stale)."""
        return self.parent.spots

    def __init__(self, parent: "EmbryoViewer"):
        napari, _ = _import_napari()
        from magicgui.widgets import (
            Container,
            FloatSlider,
            Label,
            ComboBox,
            PushButton,
            CheckBox,
            SpinBox,
        )

        self.parent = parent
        self._axis = "Y"  # cutting axis
        self._thickness = 20.0  # slab total thickness
        self._color_mode = parent._current_color_mode
        self._display_pct = 100.0
        self._show_all = False
        self._track_width = 2.0
        self._show_tracks = True
        self._points_visible = True  # track user's nuclei visibility preference
        self._tracks_visible_state = True  # track user's tracks visibility
        # ── User-tweakable Tracks display props (persist across rebuilds) ──
        # napari resets these whenever we add a fresh Tracks layer, so we
        # cache them here and re-apply on every rebuild + recording frame.
        self._track_tail_length = 40
        self._track_head_length = 0
        self._track_display_id = False
        self._track_display_graph = True
        self._track_opacity = 1.0

        # ── Pre-cache numpy arrays from DF (avoids repeated .values calls) ──
        df = self.df
        self._pos_x = df["POSITION_X"].values
        self._pos_y = df["POSITION_Y"].values
        self._pos_z = df["POSITION_Z"].values
        self._frames = df["FRAME"].values

        # Set max tracks to actual track count (fast on numpy)
        if "TRACK_ID" in df.columns:
            self._track_ids = df["TRACK_ID"].values
            n_total_tracks = len(np.unique(self._track_ids[~np.isnan(self._track_ids)])) if self._track_ids.dtype.kind == 'f' else len(np.unique(self._track_ids))
        else:
            self._track_ids = None
            n_total_tracks = 500
        self._max_tracks = n_total_tracks

        # Compute axis ranges (single pass on cached arrays)
        self._ranges = {
            "X": (float(self._pos_x.min()), float(self._pos_x.max())),
            "Y": (float(self._pos_y.min()), float(self._pos_y.max())),
            "Z": (float(self._pos_z.min()), float(self._pos_z.max())),
        }
        fmin = int(self._frames.min())
        fmax = int(self._frames.max())

        # Separate napari viewer in 2D
        self.viewer = napari.Viewer(title="Cross-Section View", ndisplay=2)
        self.points_layer = None
        self.tracks_layer = None
        self._sphere_layer = None
        self._landmarks_layer = None
        self._center_layer = None
        self._processed_image_layer = None
        self._seg_image_layer = None
        self._recording = False
        self._saved_view_state = None  # camera + layer settings snapshot

        # Check which image data is available from parent viewer
        self._has_processed = getattr(parent, '_processed_frames', None) is not None
        self._has_seg = getattr(parent, '_segment_frames', None) is not None

        r = self._ranges[self._axis]
        mid = (r[0] + r[1]) / 2.0

        # ── Slab controls ──
        self.axis_combo = ComboBox(value="Y", choices=["X", "Y", "Z"], label="Cut axis")
        self.axis_combo.changed.connect(self._on_axis_changed)

        self.pos_slider = FloatSlider(
            value=mid, min=r[0], max=r[1], step=1.0, label="Position"
        )
        self.pos_slider.changed.connect(self._schedule_rebuild)

        self.thickness_slider = FloatSlider(
            value=20.0, min=1.0, max=200.0, step=1.0, label="Slab thickness"
        )
        self.thickness_slider.changed.connect(self._on_thickness_changed)

        # Quick-jump buttons for common positions
        btn_jump_center = PushButton(text="Jump to sphere center")
        btn_jump_center.changed.connect(self._jump_to_center)
        btn_jump_ap = PushButton(text="Jump to animal pole")
        btn_jump_ap.changed.connect(self._jump_to_ap)

        # ── Time ──
        self.frame_slider = FloatSlider(
            value=fmin, min=fmin, max=fmax, step=1, label="Frame"
        )
        self.frame_slider.changed.connect(self._on_frame_changed)

        self.all_frames_check = CheckBox(value=False, text="Show all frames")
        self.all_frames_check.changed.connect(self._on_all_frames_changed)

        self.sync_main_check = CheckBox(value=True, text="Sync time with main viewer")
        self.sync_main_check.changed.connect(self._on_sync_main_changed)

        # ── Display ──
        self.display_pct_slider = FloatSlider(
            value=100, min=1, max=100, step=1, label="Display %"
        )
        self.display_pct_slider.changed.connect(self._on_display_pct_changed)

        self.show_tracks_check = CheckBox(value=True, text="Show tracks")
        self.show_tracks_check.changed.connect(self._on_show_tracks_changed)

        self.max_tracks_slider = FloatSlider(
            value=n_total_tracks,
            min=100,
            max=max(500, n_total_tracks),
            step=500,
            label="Max tracks",
        )
        self.max_tracks_slider.changed.connect(self._on_max_tracks_changed)

        self.track_width_slider = FloatSlider(
            value=2.0, min=0.5, max=10.0, step=0.5, label="Track width"
        )
        self.track_width_slider.changed.connect(self._on_track_width_changed)

        # ── Mirror main-viewer selection (random sample / click) ────
        self.mirror_main_selection_cb = CheckBox(
            value=True, text="Show only main-viewer selected tracks"
        )
        self.mirror_main_selection_cb.changed.connect(self._on_mirror_changed)
        self.lbl_parent_selection = Label(value="Parent selection: none")

        # ── Random subsample (no-repeat; shared with main viewer) ─────
        self.random_sample_n = SpinBox(
            value=50, min=1, max=10000, step=1, label="Random N"
        )
        self.btn_random_sample = PushButton(
            text="🎲 Sample N random (no repeat)"
        )
        self.btn_random_sample.changed.connect(self._random_sample_tracks)
        self.btn_random_sample_reset = PushButton(text="↻ Reset random pool")
        self.btn_random_sample_reset.changed.connect(
            self._reset_random_sample_pool
        )

        self.point_size_slider = FloatSlider(
            value=3.0, min=1.0, max=20.0, step=0.5, label="Point size"
        )
        self.point_size_slider.changed.connect(self._schedule_rebuild)

        # ── Colour mode buttons ──
        btn_color_frame = PushButton(text="Colour by Frame")
        btn_color_frame.changed.connect(lambda: self._set_color("frame"))
        btn_color_depth = PushButton(text="Colour by Depth")
        btn_color_depth.changed.connect(lambda: self._set_color("depth"))
        btn_color_depth_travel = PushButton(text="Colour by Depth Travel")
        btn_color_depth_travel.changed.connect(lambda: self._set_color("depth_travel"))
        btn_color_theta = PushButton(text="Colour by Latitude")
        btn_color_theta.changed.connect(lambda: self._set_color("theta"))
        btn_color_phi = PushButton(text="Colour by Longitude")
        btn_color_phi.changed.connect(lambda: self._set_color("phi"))
        btn_color_speed = PushButton(text="Colour by Speed")
        btn_color_speed.changed.connect(lambda: self._set_color("speed"))
        btn_color_radvel = PushButton(text="Colour by Radial Velocity")
        btn_color_radvel.changed.connect(lambda: self._set_color("radial_vel"))
        btn_color_track = PushButton(text="Colour by Track ID")
        btn_color_track.changed.connect(lambda: self._set_color("track_id"))

        # ── Annotation toggles ──
        self.show_landmarks_check = CheckBox(
            value=True, text="Show landmarks (AP/Dorsal)"
        )
        self.show_landmarks_check.changed.connect(self._schedule_rebuild)
        self.show_sphere_check = CheckBox(value=True, text="Show sphere intersection")
        self.show_sphere_check.changed.connect(self._schedule_rebuild)
        self.show_center_check = CheckBox(value=True, text="Show sphere center")
        self.show_center_check.changed.connect(self._schedule_rebuild)

        # ── Image layer toggles ──
        self.show_processed_check = CheckBox(
            value=self._has_processed, text="Show processed image slice"
        )
        self.show_processed_check.changed.connect(self._schedule_rebuild)
        self.show_seg_check = CheckBox(
            value=self._has_seg, text="Show segmentation slice"
        )
        self.show_seg_check.changed.connect(self._schedule_rebuild)
        self.slice_mip_check = CheckBox(
            value=True, text="Max-intensity projection (slab)"
        )
        self.slice_mip_check.changed.connect(self._schedule_rebuild)
        self.processed_opacity_slider = FloatSlider(
            value=60, min=0, max=100, step=5, label="Processed opacity %"
        )
        self.processed_opacity_slider.changed.connect(self._schedule_rebuild)
        self.seg_opacity_slider = FloatSlider(
            value=40, min=0, max=100, step=5, label="Seg opacity %"
        )
        self.seg_opacity_slider.changed.connect(self._schedule_rebuild)

        # ── Record video ──
        self.frame_start_slider = FloatSlider(
            value=fmin, min=fmin, max=fmax, step=1, label="Start frame"
        )
        self.frame_end_slider = FloatSlider(
            value=fmax, min=fmin, max=fmax, step=1, label="End frame"
        )
        self.frame_step_slider = FloatSlider(
            value=1, min=1, max=10, step=1, label="Frame step"
        )
        self.render_scale_slider = FloatSlider(
            value=2.0, min=1.0, max=4.0, step=0.5,
            label="Render scale (×)",
        )
        self.duration_slider = FloatSlider(
            value=0, min=0, max=120, step=1,
            label="Duration (s, 0=auto)",
        )
        btn_record = PushButton(text="\U0001f3ac Record Time-lapse Video")
        btn_record.changed.connect(self._record_video)
        btn_screenshot = PushButton(text="\U0001f4f7 Screenshot")
        btn_screenshot.changed.connect(self._save_screenshot)

        self.lbl_info = Label(value="")
        self.lbl_annotations = Label(value="")
        self.lbl_record = Label(value="")

        container = Container(
            widgets=[
                Label(value="── Slab ──"),
                self.axis_combo,
                self.pos_slider,
                self.thickness_slider,
                btn_jump_center,
                btn_jump_ap,
                Label(value="── Time ──"),
                self.frame_slider,
                self.all_frames_check,
                self.sync_main_check,
                Label(value="── Display ──"),
                self.display_pct_slider,
                self.point_size_slider,
                self.show_tracks_check,
                self.max_tracks_slider,
                self.track_width_slider,
                self.mirror_main_selection_cb,
                self.lbl_parent_selection,
                self.random_sample_n,
                self.btn_random_sample,
                self.btn_random_sample_reset,
                Label(value="── Annotations ──"),
                self.show_landmarks_check,
                self.show_sphere_check,
                self.show_center_check,
                Label(value="── Image Layers ──"),
                self.show_processed_check,
                self.processed_opacity_slider,
                self.show_seg_check,
                self.seg_opacity_slider,
                self.slice_mip_check,
                Label(value="── Colour ──"),
                btn_color_frame,
                btn_color_depth,
                btn_color_depth_travel,
                btn_color_theta,
                btn_color_phi,
                btn_color_speed,
                btn_color_radvel,
                btn_color_track,
                Label(value="── Record ──"),
                self.frame_start_slider,
                self.frame_end_slider,
                self.frame_step_slider,
                self.render_scale_slider,
                self.duration_slider,
                btn_record,
                btn_screenshot,
                self.lbl_record,
                Label(value="── Info ──"),
                self.lbl_info,
                self.lbl_annotations,
            ]
        )

        from qtpy.QtWidgets import QScrollArea
        from qtpy.QtCore import Qt

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(280)
        scroll.setWidget(container.native)
        self.viewer.window.add_dock_widget(scroll, name="Controls", area="right")

        # Track whether first rebuild has happened (for reset_view)
        self._first_rebuild = True
        self._last_axis = self._axis
        self._in_rebuild = False
        self._oriented_grid_cache = {}  # ds → grid dict for oriented extraction

        # Sync cross-section frame with main viewer's timepoint
        self._sync_with_main = True

        # ── Connect MAIN viewer dims → cross-section rebuild ──
        def _on_main_time_changed(event):
            if not self._sync_with_main or self._in_rebuild:
                return
            t = int(self.parent.viewer.dims.current_step[0])
            t_clamped = max(self.frame_slider.min, min(self.frame_slider.max, t))
            # Setting frame_slider.value fires .changed → _schedule_rebuild
            self.frame_slider.value = t_clamped

        try:
            self.parent.viewer.dims.events.current_step.connect(
                _on_main_time_changed
            )
        except Exception as exc:
            print(f"  ⚠ Could not connect main viewer time sync: {exc}")
        self._main_time_cb = _on_main_time_changed  # prevent GC

        # ── Connect CROSS-SECTION viewer dims → rebuild on time scroll ──
        # The Tracks layer adds a time dimension to the XS viewer.
        # When the user scrolls napari's dims slider, tracks auto-filter
        # but our 2D points/images don't update.
        def _on_xs_dims_changed(event):
            if self._in_rebuild:
                return
            step = self.viewer.dims.current_step
            if len(step) < 3:
                return  # no time dim
            frame = int(step[0])
            if frame == int(self.frame_slider.value):
                return  # no change
            # Setting frame_slider.value fires .changed → _schedule_rebuild
            self.frame_slider.value = max(
                self.frame_slider.min,
                min(self.frame_slider.max, frame),
            )

        self.viewer.dims.events.current_step.connect(_on_xs_dims_changed)
        self._xs_dims_cb = _on_xs_dims_changed  # prevent GC

        # Defer initial rebuild so the window appears immediately
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, self._rebuild)

        # Register with the parent so this cross-section viewer is notified
        # of selection changes (random sample / click / clear) and seeds
        # its own state from the parent's current selection.
        self._register_with_parent()
        self._parent_selected_tids: set[int] = set()
        self._parent_selection_active: bool = False
        self._refresh_parent_selection_label()

    # ── Quick-jump helpers ───────────────────────────────────────────

    def _jump_to_center(self):
        """Move the slice position to the sphere center along the current axis."""
        sph = self.parent.sphere_fit_result
        if sph is None:
            self.viewer.status = "Fit sphere in main viewer first!"
            return
        axis_idx = {"X": 0, "Y": 1, "Z": 2}[self._axis]
        val = sph["center"][axis_idx]
        self.pos_slider.value = np.clip(val, self.pos_slider.min, self.pos_slider.max)
        self.viewer.status = f"Jumped to sphere center along {self._axis}"

    def _jump_to_ap(self):
        """Move the slice position to the animal pole along the current axis."""
        ap = self.parent.animal_pole
        if ap is None:
            self.viewer.status = "Set animal pole in main viewer first!"
            return
        axis_idx = {"X": 0, "Y": 1, "Z": 2}[self._axis]
        val = ap[axis_idx]
        self.pos_slider.value = np.clip(val, self.pos_slider.min, self.pos_slider.max)
        self.viewer.status = f"Jumped to animal pole along {self._axis}"

    # ── Coordinate helpers ───────────────────────────────────────────

    def _project_point_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Project an (x, y, z) world point to 2D [v, h] for current axis."""
        v_col, h_col = self._AXES_MAP[self._axis]
        v = xyz[self._COL_TO_IDX[v_col]]
        h = xyz[self._COL_TO_IDX[h_col]]
        return np.array([v, h])

    def _axis_value_of_point(self, xyz: np.ndarray) -> float:
        """Return the coordinate of xyz along the current cutting axis."""
        axis_idx = {"X": 0, "Y": 1, "Z": 2}[self._axis]
        return float(xyz[axis_idx])

    # ── Widget callbacks ─────────────────────────────────────────────

    def _on_axis_changed(self):
        self._axis = self.axis_combo.value
        r = self._ranges[self._axis]
        mid = (r[0] + r[1]) / 2.0
        self.pos_slider.min = r[0]
        self.pos_slider.max = r[1]
        self.pos_slider.value = mid
        # Image dimensions change with axis — must recreate layers + grid
        self._oriented_grid_cache = {}
        self._remove_image_layers()
        self._rebuild()

    def _on_thickness_changed(self):
        self._thickness = self.thickness_slider.value
        # Thickness change affects image extraction — recreate grid
        self._oriented_grid_cache = {}
        self._remove_image_layers()
        self._schedule_rebuild()

    def _on_all_frames_changed(self):
        self._show_all = self.all_frames_check.value
        self._rebuild()

    def _on_sync_main_changed(self):
        self._sync_with_main = self.sync_main_check.value

    def _on_display_pct_changed(self):
        self._display_pct = self.display_pct_slider.value
        self._rebuild()

    def _on_show_tracks_changed(self):
        self._show_tracks = self.show_tracks_check.value
        self._rebuild()

    def _on_max_tracks_changed(self):
        self._max_tracks = int(self.max_tracks_slider.value)
        self._schedule_rebuild()

    def _on_track_width_changed(self):
        self._track_width = float(self.track_width_slider.value)
        self._schedule_rebuild()

    def _on_mirror_changed(self, value: bool):
        """User toggled 'Show only main-viewer selected tracks'."""
        self._schedule_rebuild()

    def _random_sample_tracks(self):
        """Delegate random sampling to the parent so both windows share
        the same 'already drawn' pool and selection set.
        """
        try:
            self.parent._random_sample_tracks()
        except Exception:
            traceback.print_exc()
            self.viewer.status = (
                "Random sample failed — ensure tracks are loaded"
            )
            return
        self._refresh_parent_selection_label()

    def _reset_random_sample_pool(self):
        """Reset the shared no-repeat pool via the parent."""
        try:
            self.parent._reset_random_sample_pool()
        except Exception:
            traceback.print_exc()
            return
        self._refresh_parent_selection_label()

    def _refresh_parent_selection_label(self):
        """Update the small label summarising the parent's selection."""
        lbl = getattr(self, "lbl_parent_selection", None)
        if lbl is None:
            return
        parent = self.parent
        n_sel = len(getattr(parent, "_selected_track_ids", set()))
        n_pool = len(getattr(parent, "_random_sampled_ids", set()))
        if n_sel == 0:
            lbl.value = "Parent selection: none"
        else:
            lbl.value = (
                f"Parent selection: {n_sel} track(s)"
                f" — {n_pool} ever sampled"
            )

    def on_parent_selection_changed(self):
        """Mirror parent's selection state and rebuild slab tracks.

        Called by ``EmbryoViewer`` whenever selection state changes
        (manual click, random sample, clear).
        """
        parent = self.parent
        active = bool(getattr(parent, "_selection_active", False))
        sel_set = (
            set(int(t) for t in parent._selected_track_ids)
            if active
            else set()
        )
        self._parent_selection_active = active and bool(sel_set)
        self._parent_selected_tids = sel_set
        self._refresh_parent_selection_label()
        if getattr(self, "_show_tracks", True):
            self._schedule_rebuild()

    def _register_with_parent(self):
        """Register with the parent for selection-change notifications."""
        parent = self.parent
        reg = getattr(parent, "_register_cross_section", None)
        if reg is not None:
            try:
                reg(self)
            except Exception:
                pass

    def _set_color(self, mode: str):
        needed = {
            "depth": "SPHERICAL_DEPTH",
            "depth_travel": "SPHERICAL_DEPTH",
            "theta": "THETA_DEG",
            "phi": "PHI_DEG",
            "radial_vel": "RADIAL_VELOCITY_SMOOTH",
        }
        if mode in needed and needed[mode] not in self.df.columns:
            self.viewer.status = f"'{mode}' not available — fit sphere first."
            return
        self._color_mode = mode
        self._rebuild()

    def _schedule_rebuild(self, *_args):
        """Debounce rapid slider changes with 50ms QTimer (full rebuild)."""
        from qtpy.QtCore import QTimer

        # Invalidate spatial mask cache (position/thickness/axis may have changed)
        self._spatial_mask_cache = None
        # Invalidate raw contrast limits (recalculate for new spatial region)
        self._processed_contrast_limits = None
        if not hasattr(self, "_debounce_timer"):
            self._debounce_timer = QTimer()
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._rebuild)
        self._debounce_timer.start(50)

    def _on_frame_changed(self, *_args):
        """Immediate lightweight frame update (no debounce).

        Like the main viewer's _on_segments_time, this fires on every
        single frame change for smooth real-time scrolling.
        """
        # If a full rebuild is pending, skip — it handles everything
        if hasattr(self, "_debounce_timer") and self._debounce_timer.isActive():
            return
        # During recording, _record_video handles per-frame updates itself
        if self._recording:
            return
        self._frame_update()

    def _save_camera(self):
        """Return current camera state dict."""
        cam = self.viewer.camera
        return {"center": tuple(cam.center), "zoom": cam.zoom}

    def _restore_camera(self, state):
        """Restore camera from saved state dict."""
        if state is None:
            return
        self.viewer.camera.center = state["center"]
        self.viewer.camera.zoom = state["zoom"]

    def _frame_update(self):
        """Fast path when only the frame changed.

        Pushes image textures directly to vispy (no layer add/remove).
        Updates points via vispy.  Tracks auto-filter via napari dims.
        Sphere, landmarks, and center are frame-independent.
        """
        if self._in_rebuild:
            return
        self._in_rebuild = True
        try:
            frame = int(self.frame_slider.value)

            # Save camera so layer add/remove cannot reset zoom/pan
            cam_state = self._save_camera()

            # ── Image slices (vispy texture swap) ──
            self._update_image_slices()

            # ── Points (fast filter + vispy update) ──
            # When show_all is on, points don't change per frame —
            # skip the expensive re-filter + re-color.
            if not self._show_all:
                # Read current visibility before any layer manipulation
                if self.points_layer is not None:
                    self._points_visible = self.points_layer.visible

                spatial_mask = self._get_spatial_mask()
                idx = np.where(spatial_mask & (self._frames == frame))[0]

                if self._display_pct < 100 and len(idx) > 0:
                    n_show = max(100, int(len(idx) * self._display_pct / 100))
                    if n_show < len(idx):
                        rng = np.random.default_rng(0)
                        idx = rng.choice(idx, n_show, replace=False)
                        idx.sort()

                if len(idx) > 0:
                    data = self._project(idx)
                    colors = self._get_colors(idx)
                    if self.points_layer is not None:
                        try:
                            self.points_layer.data = data
                            self.points_layer.face_color = colors
                        except Exception:
                            try:
                                self.viewer.layers.remove(self.points_layer)
                            except Exception:
                                pass
                            self.points_layer = self.viewer.add_points(
                                data, name="Nuclei",
                                size=self.point_size_slider.value,
                                face_color=colors, border_width=0, ndim=2,
                                visible=self._points_visible,
                            )
                    else:
                        self.points_layer = self.viewer.add_points(
                            data, name="Nuclei",
                            size=self.point_size_slider.value,
                            face_color=colors, border_width=0, ndim=2,
                            visible=self._points_visible,
                        )
                elif self.points_layer is not None:
                    try:
                        self.points_layer.data = np.empty((0, 2))
                    except Exception:
                        pass

            # ── Dims step for tracks auto-filtering ──
            try:
                step = list(self.viewer.dims.current_step)
                if len(step) >= 3:
                    step[0] = frame
                    self.viewer.dims.current_step = tuple(step)
            except Exception:
                pass

            # Restore camera (layer add/data changes may have shifted it)
            self._restore_camera(cam_state)
        finally:
            self._in_rebuild = False

    # ── Core rebuild ─────────────────────────────────────────────────

    def _get_slab_mask(self) -> np.ndarray:
        """Return boolean mask for spots within the current slab + frame."""
        spatial = self._get_spatial_mask()
        if not self._show_all:
            frame = int(self.frame_slider.value)
            frame_mask = self._frames == frame
            return spatial & frame_mask
        return spatial

    def _get_spatial_mask(self) -> np.ndarray:
        """Return cached spatial slab mask (axis+position+thickness only).

        Recomputes only when axis, position, or thickness changes.
        """
        key = (self._axis, round(self.pos_slider.value, 1), round(self._thickness, 1))
        cached = getattr(self, "_spatial_mask_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        axis_arr = {"X": self._pos_x, "Y": self._pos_y, "Z": self._pos_z}[self._axis]
        pos = self.pos_slider.value
        half = self._thickness / 2.0
        mask = (axis_arr >= pos - half) & (axis_arr <= pos + half)
        self._spatial_mask_cache = (key, mask)
        return mask

    def _project(self, idx: np.ndarray) -> np.ndarray:
        """Project indexed spots to 2D [v, h] coordinates."""
        v_col, h_col = self._AXES_MAP[self._axis]
        _col_arr = {"POSITION_X": self._pos_x, "POSITION_Y": self._pos_y, "POSITION_Z": self._pos_z}
        h = _col_arr[h_col][idx]
        v = _col_arr[v_col][idx]
        return np.column_stack([v, h])

    def _remove_image_layers(self):
        """Remove image layers (call when slab geometry changes)."""
        for attr in ("_processed_image_layer", "_seg_image_layer"):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _rebuild(self):
        """Full rebuild of points, tracks, annotations, image slices and sphere layers."""
        self._in_rebuild = True
        try:
            self._rebuild_inner()
        finally:
            self._in_rebuild = False

    def _rebuild_inner(self):
        # Save camera before removing/adding layers
        self._rebuild_cam_state = self._save_camera()

        # Capture visibility of managed layers before removing them
        if self.points_layer is not None:
            self._points_visible = self.points_layer.visible
        if self.tracks_layer is not None:
            self._tracks_visible_state = self.tracks_layer.visible

        # Remove managed layers
        for attr in (
            "points_layer",
            "tracks_layer",
            "_sphere_layer",
            "_landmarks_layer",
            "_center_layer",
        ):
            layer = getattr(self, attr, None)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
                setattr(self, attr, None)

        mask = self._get_slab_mask()
        idx = np.where(mask)[0]

        # Display % subsampling
        if self._display_pct < 100 and len(idx) > 0:
            n_show = max(100, int(len(idx) * self._display_pct / 100))
            if n_show < len(idx):
                rng = np.random.default_rng(0)
                idx = rng.choice(idx, n_show, replace=False)
                idx.sort()

        # ── Image slices (remove/re-add each frame) ──
        self._update_image_slices()

        # ── Points ──
        if len(idx) > 0:
            data = self._project(idx)
            colors = self._get_colors(idx)
            self.points_layer = self.viewer.add_points(
                data,
                name="Nuclei",
                size=self.point_size_slider.value,
                face_color=colors,
                border_width=0,
                ndim=2,
                visible=self._points_visible,
            )

        # ── Tracks within slab ──
        if self._show_tracks and len(idx) > 0:
            self._add_slab_tracks(mask)
            # Restore tracks visibility from saved state
            if self.tracks_layer is not None:
                self.tracks_layer.visible = self._tracks_visible_state

        # ── Sphere cross-section circle ──
        if self.show_sphere_check.value:
            self._add_sphere_intersection()

        # ── Sphere center marker ──
        if self.show_center_check.value:
            self._add_center_marker()

        # ── Landmark markers (AP / Dorsal) ──
        if self.show_landmarks_check.value:
            self._add_landmark_markers()

        # ── Axis labels ──
        v_label, h_label = self._AXES_LABELS[self._axis]
        try:
            ndim = self.viewer.dims.ndim
            if ndim >= 3:
                self.viewer.dims.axis_labels = ["Frame", v_label, h_label]
            else:
                self.viewer.dims.axis_labels = [v_label, h_label]
        except Exception:
            pass

        # ── Info label ──
        pos = self.pos_slider.value
        half = self._thickness / 2.0
        info = f"Cut {self._axis}={pos:.0f} ± {half:.0f}  |  {len(idx):,} spots"
        if not self._show_all:
            info += f"  |  frame {int(self.frame_slider.value)}"
        else:
            info += "  |  all frames"
        n_tracks_shown = 0
        if self.tracks_layer is not None:
            try:
                n_tracks_shown = len(np.unique(self.tracks_layer.data[:, 0]))
            except Exception:
                pass
        if n_tracks_shown > 0:
            info += f"  |  {n_tracks_shown:,} tracks"
        self.lbl_info.value = info

        # ── Annotations info ──
        ann_parts = []
        sph = self.parent.sphere_fit_result
        if sph is not None:
            c = sph["center"]
            ann_parts.append(
                f"Sphere: R={sph['radius']:.0f}, center=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})"
            )
        if self.parent.animal_pole is not None:
            ap = self.parent.animal_pole
            ann_parts.append(f"AP: ({ap[0]:.0f},{ap[1]:.0f},{ap[2]:.0f})")
        if self.parent.dorsal_mark is not None:
            dm = self.parent.dorsal_mark
            ann_parts.append(f"Dorsal: ({dm[0]:.0f},{dm[1]:.0f},{dm[2]:.0f})")
        self.lbl_annotations.value = (
            "\\n".join(ann_parts) if ann_parts else "No annotations set"
        )

        if self._recording:
            self._restore_view_state()
        elif self._first_rebuild or self._last_axis != self._axis:
            # Only reset camera on first build or when axis changes;
            # time/position changes should preserve zoom/pan.
            self.viewer.reset_view()
            self._first_rebuild = False
            self._last_axis = self._axis
        else:
            # Restore camera that was saved at start of _rebuild_inner
            self._restore_camera(self._rebuild_cam_state)

        # Force the vispy canvas to repaint so image layers are visible
        try:
            self.viewer.window._qt_viewer.canvas.native.update()
        except Exception:
            pass

        # Restore the dims time slider to the current frame so the
        # Tracks layer shows the correct time (the tracks layer was
        # removed and re-added, which resets dims).
        frame = int(self.frame_slider.value)
        try:
            step = list(self.viewer.dims.current_step)
            if len(step) >= 3:
                step[0] = frame
                self.viewer.dims.current_step = tuple(step)
        except Exception:
            pass

    def _snapshot_view_state(self):
        """Save camera and all layer display settings before recording."""
        cam = self.viewer.camera
        self._saved_view_state = {
            "camera": {
                "center": tuple(cam.center),
                "zoom": cam.zoom,
                "angles": tuple(cam.angles),
            },
            "layers": {},
        }
        for layer in self.viewer.layers:
            props = {
                "visible": layer.visible,
                "opacity": layer.opacity,
            }
            # Tracks-specific settings — capture every common knob from
            # napari's layer-controls panel so the user's clarity tweaks
            # survive the rebuild that happens during recording.
            for tkey in (
                "tail_length", "tail_width", "head_length",
                "display_id", "display_graph",
                "color_by", "colormap", "colormaps_dict",
                "blending",
            ):
                if hasattr(layer, tkey):
                    try:
                        props[tkey] = getattr(layer, tkey)
                    except Exception:
                        pass
            # Points-specific settings
            if hasattr(layer, "size") and not callable(layer.size):
                try:
                    props["size"] = (
                        float(layer.size)
                        if np.ndim(layer.size) == 0
                        else layer.size.copy()
                    )
                except Exception:
                    pass
            if hasattr(layer, "symbol"):
                props["symbol"] = layer.symbol
            # Shapes-specific settings
            if hasattr(layer, "edge_width"):
                props["edge_width"] = layer.edge_width
            self._saved_view_state["layers"][layer.name] = props

    def _restore_view_state(self):
        """Restore camera and layer display settings from snapshot."""
        if self._saved_view_state is None:
            return
        # Restore camera
        cam_state = self._saved_view_state["camera"]
        self.viewer.camera.center = cam_state["center"]
        self.viewer.camera.zoom = cam_state["zoom"]
        # Restore layer settings
        saved_layers = self._saved_view_state["layers"]
        for layer in self.viewer.layers:
            if layer.name in saved_layers:
                props = saved_layers[layer.name]
                for key, val in props.items():
                    try:
                        setattr(layer, key, val)
                    except Exception:
                        pass

    def _add_slab_tracks(self, slab_mask: np.ndarray):
        """Add 2D tracks for track segments within the slab.

        Uses cached numpy arrays — avoids pandas isin/groupby/sort
        on the full DataFrame for speed.
        """
        if self._track_ids is None:
            return

        track_ids = self._track_ids
        frames = self._frames

        # Get track IDs that have spots in slab (numpy, no pandas)
        slab_tids_arr = track_ids[slab_mask]
        if slab_tids_arr.dtype.kind == 'f':
            slab_tids_arr = slab_tids_arr[~np.isnan(slab_tids_arr)]
        slab_tids = np.unique(slab_tids_arr)
        if len(slab_tids) == 0:
            return

        # Restrict to tracks currently selected in the main viewer (random
        # sample or click selection).  Without this filter the cross-
        # section viewer would always show every track intersecting the
        # slab — the very reason the user opened this viewer to see
        # "just the selected tracks".
        if (
            getattr(self, "_mirror_main_selection_cb", None) is not None
            and self._mirror_main_selection_cb.value
            and self._parent_selection_active
        ):
            sel_set = self._parent_selected_tids
            slab_tids = np.array(
                [t for t in slab_tids.tolist() if t in sel_set],
                dtype=slab_tids.dtype,
            )
            if slab_tids.size == 0:
                return

        # Subsample if needed
        if len(slab_tids) > self._max_tracks:
            rng = np.random.default_rng(42)
            slab_tids = rng.choice(slab_tids, self._max_tracks, replace=False)

        # Fast membership test via set
        tid_set = set(slab_tids.tolist())
        in_tracks = np.fromiter(
            (t in tid_set for t in track_ids), dtype=bool, count=len(track_ids)
        )
        if not self._show_all:
            in_slab = self._get_spatial_mask()
            track_mask = in_tracks & in_slab
        else:
            track_mask = in_tracks & slab_mask

        sel_idx = np.where(track_mask)[0]
        if len(sel_idx) == 0:
            return

        # Extract columns using cached arrays
        sel_tids = track_ids[sel_idx]
        sel_frames = frames[sel_idx]
        _col_arr = {
            "POSITION_X": self._pos_x,
            "POSITION_Y": self._pos_y,
            "POSITION_Z": self._pos_z,
        }
        v_col, h_col = self._AXES_MAP[self._axis]
        sel_v = _col_arr[v_col][sel_idx]
        sel_h = _col_arr[h_col][sel_idx]

        # Sort by (track_id, frame) using numpy lexsort
        order = np.lexsort((sel_frames, sel_tids))
        sel_tids = sel_tids[order]
        sel_frames = sel_frames[order]
        sel_v = sel_v[order]
        sel_h = sel_h[order]
        sel_idx = sel_idx[order]

        # Keep only tracks with >=2 points (numpy, no groupby)
        unique_tids, counts = np.unique(sel_tids, return_counts=True)
        keep_tids = set(unique_tids[counts >= 2].tolist())
        if not keep_tids:
            return
        keep_mask = np.fromiter(
            (t in keep_tids for t in sel_tids), dtype=bool, count=len(sel_tids)
        )
        sel_tids = sel_tids[keep_mask]
        sel_frames = sel_frames[keep_mask]
        sel_v = sel_v[keep_mask]
        sel_h = sel_h[keep_mask]
        sel_idx = sel_idx[keep_mask]

        tracks_arr = np.column_stack([sel_tids, sel_frames, sel_v, sel_h])

        # Color tracks consistently with points
        props = {}
        color_by = "track_id"
        cmap = "hsv"

        col_map = {
            "depth": "SPHERICAL_DEPTH",
            "theta": "THETA_DEG",
            "phi": "PHI_DEG",
            "radial_vel": "RADIAL_VELOCITY_SMOOTH",
            "frame": "FRAME",
        }
        mode = self._color_mode
        df = self.df
        if mode == "depth_travel" and "SPHERICAL_DEPTH" in df.columns:
            depth_vals = df["SPHERICAL_DEPTH"].values[sel_idx].astype(float)
            valid_d = ~np.isnan(depth_vals)
            if valid_d.any():
                _, inv = np.unique(sel_tids, return_inverse=True)
                n_ut = inv.max() + 1
                dmax = np.full(n_ut, -np.inf)
                dmin = np.full(n_ut, np.inf)
                np.maximum.at(dmax, inv[valid_d], depth_vals[valid_d])
                np.minimum.at(dmin, inv[valid_d], depth_vals[valid_d])
                depth_range = np.maximum(dmax - dmin, 0.0)[inv]
                depth_range[~valid_d] = 0.0
                vmax = max(np.percentile(depth_range[valid_d], 98), 1e-6)
                depth_range = np.clip(depth_range / vmax, 0, 1)
                props["color_value"] = depth_range
                color_by = "color_value"
                cmap = "plasma"
        elif mode in col_map and col_map[mode] in df.columns:
            vals = df[col_map[mode]].values[sel_idx].astype(float)
            nan_m = np.isnan(vals)
            if not nan_m.all():
                vals[nan_m] = np.nanmedian(vals)
                props["color_value"] = vals
                color_by = "color_value"
                cmap = "turbo" if mode == "radial_vel" else "viridis"

        kw = dict(
            name="Tracks",
            tail_width=self._track_width,
            tail_length=self._track_tail_length,
            head_length=self._track_head_length,
            color_by=color_by,
            colormap=cmap,
            visible=self._tracks_visible_state,
        )
        if props:
            kw["properties"] = props

        try:
            self.tracks_layer = self.viewer.add_tracks(tracks_arr, **kw)
        except Exception:
            return  # tracks can fail if too few points; not critical

        # Apply remaining cached display knobs that aren't constructor args
        try:
            self.tracks_layer.display_id = self._track_display_id
            self.tracks_layer.display_graph = self._track_display_graph
            self.tracks_layer.opacity = self._track_opacity
        except Exception:
            pass

        # Hook napari layer events so the user's panel adjustments are
        # remembered across rebuilds (axis/position changes) AND across
        # recording (which removes & re-creates the Tracks layer).
        self._bind_tracks_events(self.tracks_layer)

    def _bind_tracks_events(self, layer):
        """Mirror user-driven Tracks property changes into cached attrs."""
        try:
            evs = layer.events
        except Exception:
            return

        def _mk(attr):
            def _cb(_e=None, _attr=attr, _layer=layer):
                try:
                    setattr(self, f"_track_{_attr}", getattr(_layer, _attr))
                except Exception:
                    pass
            return _cb

        for attr in ("tail_length", "head_length", "display_id",
                     "display_graph", "opacity"):
            ev = getattr(evs, attr, None)
            if ev is None:
                continue
            try:
                ev.connect(_mk(attr))
            except Exception:
                pass

    def _add_sphere_intersection(self):
        """Draw the sphere's intersection circle with the cutting plane."""
        sph = self.parent.sphere_fit_result
        if sph is None:
            return

        center = sph["center"]
        R = sph["radius"]
        pos = self.pos_slider.value

        axis_idx = {"X": 0, "Y": 1, "Z": 2}[self._axis]
        d = abs(pos - center[axis_idx])
        if d >= R:
            return  # plane doesn't intersect sphere

        r = np.sqrt(R**2 - d**2)
        cv, ch = self._project_point_xyz(center)

        theta = np.linspace(0, 2 * np.pi, 200)
        circle_v = cv + r * np.sin(theta)
        circle_h = ch + r * np.cos(theta)
        circle_pts = np.column_stack([circle_v, circle_h])

        self._sphere_layer = self.viewer.add_shapes(
            [circle_pts],
            shape_type="polygon",
            edge_color="cyan",
            edge_width=2,
            face_color=np.array([0.0, 1.0, 1.0, 0.05]),
            name="Sphere section",
            ndim=2,
        )

    def _add_center_marker(self):
        """Draw a crosshair at the sphere center projected into the slice."""
        sph = self.parent.sphere_fit_result
        if sph is None:
            return
        center_2d = self._project_point_xyz(sph["center"])
        arm = sph["radius"] * 0.05  # 5% of radius
        lines = [
            np.array(
                [[center_2d[0] - arm, center_2d[1]], [center_2d[0] + arm, center_2d[1]]]
            ),
            np.array(
                [[center_2d[0], center_2d[1] - arm], [center_2d[0], center_2d[1] + arm]]
            ),
        ]
        self._center_layer = self.viewer.add_shapes(
            lines,
            shape_type="line",
            edge_color="white",
            edge_width=2,
            name="Sphere center",
            ndim=2,
        )

    def _add_landmark_markers(self):
        """Show animal pole and dorsal landmarks projected into the slice."""
        pts = []
        colors = []
        texts = []

        for name, xyz, color in [
            ("AP", self.parent.animal_pole, "red"),
            ("Dorsal", self.parent.dorsal_mark, "cyan"),
        ]:
            if xyz is None:
                continue
            projected = self._project_point_xyz(xyz)
            pts.append(projected)
            texts.append(name)
            colors.append(color)

        if not pts:
            return

        pts_arr = np.array(pts)
        self._landmarks_layer = self.viewer.add_points(
            pts_arr,
            name="Landmarks",
            size=25,
            face_color=colors,
            symbol="diamond",
            border_width=1,
            border_color="white",
            ndim=2,
            text={
                "string": texts,
                "color": "white",
                "anchor": "upper_left",
                "size": 14,
                "translation": [5, 5],
            },
        )

    # ── Image slice extraction ───────────────────────────────────────

    def _extract_slab_slice(self, volume: np.ndarray, pos: float,
                            half: float, ds: float = 1.0,
                            projection: str = "max") -> np.ndarray | None:
        """Extract a 2D slice from a 3D volume at the slab position.

        Parameters
        ----------
        volume : (Z, Y, X) 3D array for the current timepoint
        pos : cutting position in world/voxel coords
        half : half-thickness in world/voxel coords
        ds : downsample factor (volume indices = world_coords / ds)
        projection : 'max' for MIP, 'mean' for average projection

        Returns
        -------
        2D array oriented as (v, h) matching the current AXES_MAP, or None.
        """
        axis = self._axis
        # Map cutting axis to volume axis index: X→2, Y→1, Z→0
        vol_axis = {"X": 2, "Y": 1, "Z": 0}[axis]
        n = volume.shape[vol_axis]

        # Convert world coords → voxel indices in (possibly downsampled) volume
        lo = max(0, int((pos - half) / ds))
        hi = min(n - 1, int((pos + half) / ds))
        if lo > hi or lo >= n:
            return None

        use_mip = getattr(self, 'slice_mip_check', None)
        use_mip = use_mip.value if use_mip is not None else True

        if use_mip and hi > lo:
            slc = [slice(None)] * 3
            slc[vol_axis] = slice(lo, hi + 1)
            slab = volume[tuple(slc)]
            if projection == "mean":
                img_2d = slab.mean(axis=vol_axis).astype(volume.dtype)
            else:
                img_2d = slab.max(axis=vol_axis)
        else:
            # Single central slice
            mid_idx = (lo + hi) // 2
            slc = [slice(None)] * 3
            slc[vol_axis] = mid_idx
            img_2d = volume[tuple(slc)]

        return np.ascontiguousarray(img_2d)

    def _get_oriented_grid(self, ds: float = 1.0):
        """Return cached (or build) the oriented sampling grid.

        The grid depends on axis, position, thickness, orientation params,
        and ds — NOT on the frame.  We cache it so that frame-to-frame
        updates only need to call map_coordinates with the pre-built coords.

        Returns
        -------
        dict with keys: coords_list, n_v, n_h, scale, translate
        or None if no orientation.
        """
        params = self.parent.transform_params
        if params is None:
            return None

        axis = self._axis
        pos = self.pos_slider.value
        half = self._thickness / 2.0

        # Include id(params) + MIP state so cache invalidates correctly
        mip_on = getattr(self, "slice_mip_check", None)
        mip_on = mip_on.value if mip_on is not None else True
        cache_key = (axis, round(pos, 1), round(half, 1), round(ds, 2), id(params), mip_on)
        ds_key = round(ds, 2)
        cached = self._oriented_grid_cache.get(ds_key)
        if cached is not None and cached.get("key") == cache_key:
            return cached

        R1, R2, center = params["R1"], params["R2"], params["center"]
        R_inv = R1.T @ R2.T

        v_col, h_col = self._AXES_MAP[axis]
        df = self.parent.spots
        v_min = float(df[v_col].min())
        v_max = float(df[v_col].max())
        h_min = float(df[h_col].min())
        h_max = float(df[h_col].max())

        # Cap grid resolution for speed (256 max)
        n_v = min(max(int((v_max - v_min) / max(ds, 1)) + 1, 10), 256)
        n_h = min(max(int((h_max - h_min) / max(ds, 1)) + 1, 10), 256)

        v_grid = np.linspace(v_min, v_max, n_v)
        h_grid = np.linspace(h_min, h_max, n_h)
        vv, hh = np.meshgrid(v_grid, h_grid, indexing="ij")

        axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis]
        v_idx = self._COL_TO_IDX[v_col]
        h_idx = self._COL_TO_IDX[h_col]

        use_mip = mip_on
        if use_mip:
            n_slab = max(1, min(int(2 * half / max(ds, 1)), 5))
            pos_vals = np.linspace(pos - half, pos + half, max(n_slab, 3))
        else:
            pos_vals = [pos]

        flat_n = n_v * n_h
        pts_template = np.zeros((flat_n, 3))
        pts_template[:, v_idx] = vv.ravel()
        pts_template[:, h_idx] = hh.ravel()

        # Pre-compute rounded integer indices + validity mask per slab slice
        coords_list = []
        for p in pos_vals:
            pts_template[:, axis_idx] = p
            pts_orig = (R_inv @ pts_template.T).T + center
            zi = np.round(pts_orig[:, 2] / ds).astype(np.intp)
            yi = np.round(pts_orig[:, 1] / ds).astype(np.intp)
            xi = np.round(pts_orig[:, 0] / ds).astype(np.intp)
            coords_list.append((zi, yi, xi))

        scale_out = (
            (v_max - v_min) / max(n_v - 1, 1),
            (h_max - h_min) / max(n_h - 1, 1),
        )
        translate_out = (v_min, h_min)

        entry = {
            "key": cache_key,
            "coords_list": coords_list,
            "n_v": n_v,
            "n_h": n_h,
            "scale": scale_out,
            "translate": translate_out,
        }
        self._oriented_grid_cache[ds_key] = entry
        return entry

    def _extract_oriented_slab_slice(
        self, volume: np.ndarray, pos: float, half: float, ds: float = 1.0,
        projection: str = "max",
    ) -> tuple[np.ndarray | None, tuple | None, tuple | None]:
        """Extract a 2D slab slice via cached oriented grid.

        Uses pre-computed integer index arrays from the grid cache
        for direct numpy fancy-indexing — zero per-frame allocation.

        Parameters
        ----------
        projection : 'max' for MIP (good for segmentation),
                     'mean' for average (good for processed microscopy).
        """
        grid = self._get_oriented_grid(ds)
        if grid is None:
            img = self._extract_slab_slice(volume, pos, half, ds,
                                           projection=projection)
            return img, None, None

        coords_list = grid["coords_list"]
        n_v, n_h = grid["n_v"], grid["n_h"]
        nz, ny, nx = volume.shape
        flat_n = n_v * n_h

        # Pre-compute validity masks (cached in grid, volume-shape-dependent)
        vol_key = (nz, ny, nx)
        if grid.get("_valid_vol_key") != vol_key:
            grid["_valid_masks"] = []
            for zi, yi, xi in coords_list:
                valid = (
                    (zi >= 0) & (zi < nz)
                    & (yi >= 0) & (yi < ny)
                    & (xi >= 0) & (xi < nx)
                )
                grid["_valid_masks"].append(valid)
            grid["_valid_vol_key"] = vol_key
        valid_masks = grid["_valid_masks"]

        if projection == "mean" and len(coords_list) > 1:
            # Mean projection: accumulate in-place, reuse buffers
            buf_key = f"_mean_buf_{flat_n}"
            cnt_key = f"_mean_cnt_{flat_n}"
            acc = getattr(self, buf_key, None)
            if acc is None or acc.shape[0] != flat_n:
                acc = np.empty(flat_n, dtype=np.float32)
                setattr(self, buf_key, acc)
            count = getattr(self, cnt_key, None)
            if count is None or count.shape[0] != flat_n:
                count = np.empty(flat_n, dtype=np.float32)
                setattr(self, cnt_key, count)
            acc[:] = 0.0
            count[:] = 0.0
            for (zi, yi, xi), valid in zip(coords_list, valid_masks):
                vals = volume[zi[valid], yi[valid], xi[valid]]
                acc[valid] += vals.astype(np.float32)
                count[valid] += 1.0
            count[count == 0] = 1.0
            np.divide(acc, count, out=acc)
            buf = acc.astype(volume.dtype)
        else:
            # Max projection (MIP) — reuse buffer
            mip_key = f"_mip_buf_{flat_n}"
            buf = getattr(self, mip_key, None)
            if buf is None or buf.shape[0] != flat_n or buf.dtype != volume.dtype:
                buf = np.empty(flat_n, dtype=volume.dtype)
                setattr(self, mip_key, buf)
            buf[:] = 0
            samp_key = f"_samp_buf_{flat_n}"
            sampled = getattr(self, samp_key, None)
            if sampled is None or sampled.shape[0] != flat_n or sampled.dtype != volume.dtype:
                sampled = np.empty(flat_n, dtype=volume.dtype)
                setattr(self, samp_key, sampled)
            for (zi, yi, xi), valid in zip(coords_list, valid_masks):
                sampled[:] = 0
                sampled[valid] = volume[zi[valid], yi[valid], xi[valid]]
                np.maximum(buf, sampled, out=buf)

        return (
            np.ascontiguousarray(buf.reshape(n_v, n_h)),
            grid["scale"],
            grid["translate"],
        )

    def _frame_to_volume_index(self, frame: int, n_tp: int) -> int:
        """Convert a FRAME value to a 0-based index into the volume list.

        FRAME values come from the DataFrame and may not start at 0.
        Volume lists (_processed_frames, _segment_frames) are 0-indexed.
        """
        fmin = int(self.frame_slider.min)
        t = frame - fmin
        return max(0, min(t, n_tp - 1))

    def _update_image_slices(self):
        """Create or update processed/segmentation image layers.

        On first call, creates the napari Image layers.
        On subsequent calls, extracts the new 2D slice and pushes it
        directly to the vispy texture node for instant GPU update,
        keeping the layer (and its panel settings) intact.
        """
        pos = self.pos_slider.value
        half = self._thickness / 2.0
        frame = int(self.frame_slider.value)
        ds = getattr(self.parent, '_display_ds', 1) or 1

        # ── Raw image slice ──
        processed_frames = getattr(self.parent, '_processed_frames', None)
        if self.show_processed_check.value and processed_frames is not None:
            n_tp = len(processed_frames)
            t = self._frame_to_volume_index(frame, n_tp)
            if 0 <= t < n_tp:
                img, scale, translate = self._extract_oriented_slab_slice(
                    processed_frames[t], pos, half, ds=1.0,
                    projection="mean",
                )
                if img is not None:
                    # Compute contrast once; reuse on subsequent frames
                    processed_cl = getattr(self, '_processed_contrast_limits', None)
                    if processed_cl is None:
                        nz = img[img > 0]
                        if len(nz) > 0:
                            c_lo = int(np.percentile(nz, 1))
                            c_hi = int(np.percentile(nz, 99.5))
                        else:
                            c_lo, c_hi = 0, 255
                        c_hi = max(c_hi, c_lo + 1)
                        self._processed_contrast_limits = [c_lo, c_hi]
                        processed_cl = self._processed_contrast_limits
                    self._push_image("_processed_image_layer", img, scale, translate,
                                     name="Raw slice", colormap="gray",
                                     contrast_limits=processed_cl,
                                     opacity=self.processed_opacity_slider.value / 100.0)

        # ── Segmentation slice ──
        seg_frames = getattr(self.parent, '_segment_frames', None)
        if self.show_seg_check.value and seg_frames is not None:
            n_tp = len(seg_frames)
            t = self._frame_to_volume_index(frame, n_tp)
            if 0 <= t < n_tp:
                img, scale, translate = self._extract_oriented_slab_slice(
                    seg_frames[t], pos, half, ds=float(ds),
                )
                if img is not None:
                    self._push_image("_seg_image_layer", img, scale, translate,
                                     name="Segmentation slice", colormap="turbo",
                                     opacity=self.seg_opacity_slider.value / 100.0,
                                     default_scale=(float(ds), float(ds)))

    def _push_image(self, attr: str, img: np.ndarray,
                    scale: tuple | None, translate: tuple | None,
                    *, name: str, colormap: str, opacity: float,
                    contrast_limits: list | None = None,
                    default_scale: tuple | None = None):
        """Push image data to an existing layer via vispy, or create a new layer.

        If the layer already exists and has the same shape, directly sets
        the vispy node's texture data for an instant GPU-only update.
        This preserves the layer's panel state (visibility, opacity tweaks, etc.).
        """
        layer = getattr(self, attr, None)

        if layer is not None:
            try:
                _ = self.viewer.layers[layer.name]  # still in viewer?
            except (KeyError, ValueError):
                layer = None
                setattr(self, attr, None)

        if layer is not None and layer.data.shape == img.shape:
            # Fast path: swap vispy texture directly
            # Preserve all layer settings (visibility, opacity, colormap, etc.)
            try:
                layer._data = img
                qt_viewer = self.viewer.window._qt_viewer
                vis = qt_viewer.canvas.layer_to_visual[layer]
                node = vis.node
                if hasattr(node, 'set_data'):
                    node.set_data(img)
                elif hasattr(node, '_data') and hasattr(node, 'update'):
                    node._data = img
                    node.update()
                qt_viewer.canvas.native.update()
                return
            except Exception:
                pass  # fall through to slow path

        # Slow path: remove old layer and create new one
        # Preserve user-modified settings from the existing layer
        cam_state = self._save_camera()
        prev_visible = True
        prev_opacity = opacity
        if layer is not None:
            try:
                prev_visible = layer.visible
                prev_opacity = layer.opacity
            except Exception:
                pass
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass
            setattr(self, attr, None)

        kw = dict(name=name, colormap=colormap, opacity=prev_opacity,
                  blending="translucent", visible=prev_visible)
        if contrast_limits is not None:
            kw["contrast_limits"] = contrast_limits
        if scale is not None:
            kw["scale"] = scale
            kw["translate"] = translate
        elif default_scale is not None:
            kw["scale"] = default_scale
        setattr(self, attr, self.viewer.add_image(img, **kw))

        # Restore camera after layer creation
        self._restore_camera(cam_state)

    def _get_colors(self, idx: np.ndarray) -> np.ndarray:
        """Color the cross-section points."""
        if len(idx) == 0:
            return np.array([[0.5, 0.5, 0.5, 1.0]])

        df = self.df
        mode = self._color_mode

        # Track ID coloring: random per-track colors
        if mode == "track_id" and "TRACK_ID" in df.columns:
            tids = df["TRACK_ID"].values[idx]
            unique_tids = np.unique(tids[~np.isnan(tids)])
            if len(unique_tids) > 0:
                try:
                    import matplotlib.pyplot as plt

                    cmap = plt.cm.tab20
                except ImportError:
                    cmap = None
                tid_to_color = {}
                for i, tid in enumerate(unique_tids):
                    if cmap is not None:
                        tid_to_color[tid] = cmap(i % 20)
                    else:
                        rng = np.random.default_rng(int(tid))
                        tid_to_color[tid] = (*rng.random(3), 1.0)
                colors = np.array(
                    [tid_to_color.get(t, (0.5, 0.5, 0.5, 0.3)) for t in tids]
                )
                return colors

        # Diverging mode for radial velocity
        if mode == "radial_vel" and "RADIAL_VELOCITY_SMOOTH" in df.columns:
            v = df["RADIAL_VELOCITY_SMOOTH"].values[idx].copy()
            nan_mask = np.isnan(v)
            v[nan_mask] = 0.0
            valid = v[~nan_mask]
            vmax = (
                max(np.abs(np.nanpercentile(valid, [2, 98])).max(), 1e-6)
                if len(valid) > 0
                else 1.0
            )
            v_clip = np.clip(v / vmax, -1, 1)
            colors = np.zeros((len(v), 4))
            colors[:, 3] = 1.0
            neg = v_clip < 0
            pos = v_clip >= 0
            t_neg = 1 + v_clip[neg]
            colors[neg, 0] = t_neg
            colors[neg, 1] = t_neg
            colors[neg, 2] = 1.0
            t_pos = v_clip[pos]
            colors[pos, 0] = 1.0
            colors[pos, 1] = 1.0 - t_pos
            colors[pos, 2] = 1.0 - t_pos
            colors[nan_mask] = [0.5, 0.5, 0.5, 0.3]
            return colors

        # Speed mode
        if mode == "speed":
            speed = self._compute_speed(idx)
            v = speed.astype(float)
        elif mode == "depth" and "SPHERICAL_DEPTH" in df.columns:
            v = df["SPHERICAL_DEPTH"].values[idx]
        elif mode == "depth_travel" and "SPHERICAL_DEPTH" in df.columns and "TRACK_ID" in df.columns:
            depth_range = df.groupby("TRACK_ID")["SPHERICAL_DEPTH"].transform(
                lambda s: s.max() - s.min()
            )
            v = depth_range.values[idx].astype(float)
            nan_mask = np.isnan(v)
            v[nan_mask] = 0.0
            valid = v[~nan_mask]
            if len(valid) > 0:
                vmax = max(np.percentile(valid, 98), 1e-6)
                v_norm = np.clip(v / vmax, 0, 1)
            else:
                v_norm = np.zeros_like(v)
            colors = _viridis_colors(v_norm)
            colors[nan_mask] = [0.4, 0.4, 0.4, 0.15]
            return colors
        elif mode == "theta" and "THETA_DEG" in df.columns:
            v = df["THETA_DEG"].values[idx]
        elif mode == "phi" and "PHI_DEG" in df.columns:
            v = df["PHI_DEG"].values[idx]
        else:
            v = df["FRAME"].values[idx].astype(float)

        v = v.astype(float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        v_norm = (v - vmin) / max(vmax - vmin, 1e-10)
        return _viridis_colors(v_norm)

    def _save_screenshot(self):
        """Save a single screenshot of the current cross-section view."""
        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True)
        axis = self._axis
        pos = int(self.pos_slider.value)
        frame = int(self.frame_slider.value)
        fname = f"cross_section_{axis}_{pos}_f{frame}.png"
        path = out_dir / fname
        scale = float(self.render_scale_slider.value)
        img = _grab_canvas_screenshot(self.viewer, scale=scale)
        if img is None:
            self.lbl_record.value = "Screenshot failed"
            return
        try:
            import imageio

            imageio.imwrite(str(path), img)
            self.lbl_record.value = f"Saved: {path}"
            self.viewer.status = f"Screenshot saved to {path}"
        except Exception as e:
            self.lbl_record.value = f"Error: {e}"

    def _record_video(self):
        """Record a time-lapse video — uses the LIVE 2D rendering path.

        For each frame we drive the existing ``_rebuild()`` machinery
        (the same code path your manual slider drag triggers) and then
        screenshot the canvas.  No layer rebuild into 3D, no view-state
        snapshot/restore, so the recorded frames look *exactly* like the
        live cross-section: same opacity, same tail-fade, same colours.
        """
        from qtpy.QtCore import QTimer

        # Guard against slider widgets that may have been destroyed if
        # the dock was closed/reopened or a previous recording crashed.
        try:
            f_start = int(self.frame_start_slider.value)
            f_end = int(self.frame_end_slider.value)
            f_step = max(1, int(self.frame_step_slider.value))
        except RuntimeError:
            self.viewer.status = (
                "Cross-section controls were destroyed — reopen the "
                "Cross-Section View and try again."
            )
            return

        all_frames = sorted(self.df["FRAME"].unique())
        frames = [int(f) for f in all_frames if f_start <= f <= f_end][::f_step]
        if not frames:
            self.viewer.status = "No frames in selected range!"
            return

        # ── Resolve fps from either the duration field or auto ──
        try:
            duration_s = float(self.duration_slider.value)
        except Exception:
            duration_s = 0.0
        if duration_s > 0:
            fps = max(1, min(60, int(round(len(frames) / duration_s))))
        else:
            fps = max(1, min(30, len(frames) // 10 or 10))

        # ── Persist the user's track-display tweaks BEFORE we touch
        #     anything, so even if napari reloads the Tracks layer mid
        #     loop those values get re-applied via _add_slab_tracks. ──
        if self.tracks_layer is not None:
            for attr in ("opacity", "tail_length", "tail_width", "head_length",
                         "display_id", "display_graph"):
                try:
                    val = getattr(self.tracks_layer, attr)
                    cache_attr = "_track_" + attr if attr != "opacity" \
                        else "_track_opacity"
                    setattr(self, cache_attr, val)
                except Exception:
                    pass

        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True)
        axis = self._axis
        slab_pos = int(self.pos_slider.value)
        vid_path = out_dir / f"cross_section_{axis}_{slab_pos}.mp4"

        self._vid_frames = frames
        self._vid_images = []
        self._vid_idx = 0
        self._vid_total = len(frames)
        self._vid_path = vid_path
        self._vid_fps = fps
        self._vid_scale = float(self.render_scale_slider.value)
        self._vid_orig_frame = int(self.frame_slider.value)
        self._recording = True
        self.lbl_record.value = f"Recording: 0/{self._vid_total} @ {fps} fps..."
        self.viewer.status = (
            f"Recording cross-section video ({len(frames)} frames @ {fps} fps)"
        )

        def _capture_next():
            if self._vid_idx >= self._vid_total:
                self._finish_video()
                return

            frame = self._vid_frames[self._vid_idx]

            # Drive the SAME code path as a manual slider drag so the
            # recorded frame is byte-identical to the live view.
            try:
                # set_value bypasses any signal coalescing; use the magicgui
                # property directly but guarded against widget deletion.
                self.frame_slider.value = frame
            except RuntimeError:
                # Slider widget was deleted (dock closed) — bail cleanly.
                self.viewer.status = "Recording aborted: dock closed."
                self._recording = False
                self._finish_video()
                return
            except ValueError:
                pass  # value out of range; skip this frame
            # _rebuild() is the same call _on_frame_changed makes — runs
            # the full 2D pipeline (image slices, points, tracks, marks).
            try:
                self._rebuild()
            except Exception:
                pass

            def _screenshot():
                img = _grab_canvas_screenshot(self.viewer, scale=self._vid_scale)
                if img is None:
                    self._vid_idx += 1
                    QTimer.singleShot(30, _capture_next)
                    return
                self._vid_images.append(img)
                self._vid_idx += 1
                if self._vid_idx % 10 == 0 or self._vid_idx == self._vid_total:
                    self.lbl_record.value = (
                        f"Recording: {self._vid_idx}/{self._vid_total} "
                        f"@ {self._vid_fps} fps..."
                    )
                QTimer.singleShot(20, _capture_next)

            # Long-enough delay for _rebuild to actually finish painting
            # before we read the framebuffer.
            QTimer.singleShot(250, _screenshot)

        QTimer.singleShot(300, _capture_next)

    def _finish_video(self):
        """Write captured frames to mp4 (or fallback to PNGs), then restore view."""
        fps = getattr(self, "_vid_fps", 10)
        ok, msg = _write_video(self._vid_path, self._vid_images, fps)
        if ok:
            self.lbl_record.value = f"Video: {self._vid_path}"
            self.viewer.status = f"Video saved to {self._vid_path} ({msg})"
        else:
            self.lbl_record.value = f"Video failed: {msg}"
            self.viewer.status = msg

        # Restore frame slider to its pre-recording position.
        orig = getattr(self, "_vid_orig_frame", None)
        if orig is not None:
            try:
                self.frame_slider.value = orig
            except Exception:
                pass
        self._recording = False
        self._saved_view_state = None
        try:
            self._rebuild()
        except Exception:
            pass

    def _compute_speed(self, idx: np.ndarray) -> np.ndarray:
        """Per-spot displacement speed for selected indices."""
        df = self.df
        speed = np.full(len(df), np.nan)
        if "TRACK_ID" not in df.columns:
            return speed[idx]
        sorted_i = df.sort_values(["TRACK_ID", "FRAME"]).index
        tid = df.loc[sorted_i, "TRACK_ID"].values
        x = df.loc[sorted_i, "POSITION_X"].values
        y = df.loc[sorted_i, "POSITION_Y"].values
        z = df.loc[sorted_i, "POSITION_Z"].values
        f = df.loc[sorted_i, "FRAME"].values.astype(float)
        dx = np.diff(x, prepend=np.nan)
        dy = np.diff(y, prepend=np.nan)
        dz = np.diff(z, prepend=np.nan)
        df_v = np.diff(f, prepend=-9999)
        same = np.diff(tid.astype(float), prepend=-9999) == 0
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dt = np.maximum(df_v, 1.0)
        sp = np.where(same & (df_v > 0), dist / dt, np.nan)
        speed[sorted_i] = sp
        return speed[idx]


# ############################################################################
#                            NAPARI VIEWER
# ############################################################################


class EmbryoViewer:
    """napari-based 4D embryo viewer with orientation & sphere fitting."""

    def __init__(
        self,
        spots_df: pd.DataFrame | None = None,
        tracks_df: pd.DataFrame | None = None,
        segments_zarr_path: str | Path | None = None,
        processed_path: str | Path | None = None,
        voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        downsample: int = 1,
        load_downsample: int = 1,
        preload: bool = False,
    ):
        napari, magicgui = _import_napari()

        self.spots = spots_df
        self.tracks_raw = tracks_df
        self._track_id_index: dict[int, np.ndarray] | None = None  # tid→row indices

        # State
        self.animal_pole: np.ndarray | None = None
        self.dorsal_mark: np.ndarray | None = None
        self.margin_landmark: np.ndarray | None = None       # margin Y cut
        self.ingression_landmark: np.ndarray | None = None   # ingression centre
        self.oriented = False
        self.sphere_fit_result: dict | None = None
        self.transform_params: dict | None = None
        self.spots_layer = None
        self.tracks_layer = None
        self._tracks_layer_secondary = None  # for dual-layer ingression view
        self._display_pct: float = 100.0
        self._max_tracks: int = 500000
        self._track_frame_min: int | None = None
        self._track_frame_max: int | None = None
        self._roi_bounds: dict | None = None  # {x_min, x_max, y_min, ...}
        self._roi_only: bool = False  # show only ROI spots
        self._roi_shapes_layer = None
        self._roi_updating: bool = False  # guard for event recursion
        self._drawing_roi: bool = False  # guard: suppress rebuilds while drawing
        self._current_color_mode: str = "frame"  # tracks colour mode
        self._tracks_cache_key: tuple | None = None  # for caching track builds
        self._display_idx: np.ndarray | None = (
            None  # indices into self.spots for displayed subset
        )
        self._track_width: float = 2.0  # track line width

        # Track selection state
        self._selected_track_ids: set[int] = set()
        self._selection_active: bool = False
        self._follow_selection: bool = False
        self._selection_lut: np.ndarray | None = None  # label→uint8 LUT for selection
        self._selection_spots_overlay = None  # napari Points (selection overlay)
        self._selection_tracks_overlay = None  # napari Tracks (selection overlay)
        # Cross-section viewers that should be notified when the selection
        # state (random sample, manual click, clear) changes.  Registered
        # by ``CrossSectionViewer.__init__`` via ``_register_cross_section``.
        self._cross_section_viewers: list["CrossSectionViewer"] = []
        # Random-subsample state (IDs already drawn, never resampled)
        self._random_sampled_ids: set[int] = set()
        # Metrics params (match gastrulation_dynamics_comparison.R)
        #   Medaka: FI=30 s, lag=4 fr (=2 min), smooth_k=5 fr (=2.5 min)
        #   Zebrafish: FI=120 s, lag=1 fr (=2 min), smooth_k=3 fr (=6 min)
        self._metrics_lag: int = 4
        self._metrics_smooth_k: int = 5
        self._metrics_fi_sec: float = 30.0

        # Segments state
        self._segments_data = None  # dask array (T, Z, Y, X)
        self._segments_layer = None  # napari Labels layer
        self._segments_original_colormap = None  # baseline cyclic LUT
        self._voxel_size = voxel_size  # (Z, Y, X) in µm
        self._downsample_factor = downsample
        self._load_downsample_factor = load_downsample
        self._preload = preload

        # Raw image state
        self._processed_path = processed_path
        self._processed_data = None        # numpy array (T, Z, Y, X)
        self._processed_layer = None       # napari Image layer
        self._processed_frames = None      # list of 3D contiguous arrays
        self._vispy_processed_volume = None
        self._segments_affine = np.eye(4)  # cumulative affine for segments/raw

        # Create viewer
        self.viewer = napari.Viewer(title="Embryo 4D Viewer", ndisplay=3)

        # Load segments if provided
        if segments_zarr_path is not None:
            self._load_segments(segments_zarr_path)

        # Load raw image if provided
        if processed_path is not None:
            self._load_processed(processed_path)

        # Load data layers if provided
        if self.spots is not None:
            self._build_track_id_index()
            self._add_spots_layer()
            self._add_tracks_layer()

        # ── Landmarks layer (3D so visible at every frame) ──
        self.landmarks_layer = self.viewer.add_points(
            np.empty((0, 3)),
            name="Landmarks",
            size=20,
            face_color="red",
            symbol="diamond",
            ndim=3,
            border_width=0,
        )

        # ── Build control widgets ──
        self._build_widgets()

        # Connect click handler for landmark selection
        self.viewer.mouse_drag_callbacks.append(self._on_click)

        # Follow-selection camera update on time change (works for all modes,
        # including lazy/non-preload where _on_segments_time is not installed).
        def _on_time_follow(event):
            if self._follow_selection and self._selection_active:
                self._center_on_selection()

        self._follow_time_cb = _on_time_follow
        self.viewer.dims.events.current_step.connect(_on_time_follow)

    # ── Segments (Labels) layer ────────────────────────────────────────

    def _load_segments(self, zarr_path: str | Path) -> None:
        """Load segments.zarr and add a Labels layer.

        Two modes controlled by self._preload:

        **Lazy (default)** — keeps data on disk via dask.  Each timepoint
        is read on demand when the slider moves.  Works on any machine
        (needs only ~200 MB per displayed frame).  Scrubbing is slower
        because of disk I/O.

        **Preload (--preload)** — loads everything into RAM, remaps
        int32→uint16, and builds a GPU-optimized display volume for
        instant, buttery-smooth time scrubbing.  Keeps full-res data
        in RAM for analysis and a spatially-downsampled display copy
        (controlled by --downsample, default 2) for fast rendering.
        """
        import dask.array as da
        import zarr
        import time as _time

        zarr_path = Path(zarr_path).resolve()
        print(f"Loading segments: {zarr_path}")
        # Try the three known ultrack layouts in order:
        #   (a) bare 4D zarr array at the root
        #   (b) zarr group with per-timepoint subarrays "0", "1", ...
        #   (c) directory of per-timepoint zarr arrays (no parent group
        #       metadata) — common with newer ultrack / zarr v3 writes
        store = None
        try:
            store = zarr.open_array(str(zarr_path), mode="r")
        except Exception:
            try:
                store = zarr.open_group(str(zarr_path), mode="r")
            except Exception:
                store = None

        if isinstance(store, zarr.Array):
            data = da.from_zarr(str(zarr_path))
            print(f"  4D zarr array: shape={data.shape}, dtype={data.dtype}")
        elif isinstance(store, zarr.Group):
            keys = sorted(int(k) for k in store.keys())
            first = store[str(keys[0])]
            lazy_frames = [
                da.from_zarr(str(zarr_path), component=str(k)) for k in keys
            ]
            data = da.stack(lazy_frames, axis=0)
            print(f"  Zarr group: {len(keys)} timepoints, "
                  f"spatial={first.shape}, dtype={first.dtype}")
        else:
            # Fallback: scan the directory for numeric subdirectories that
            # are themselves zarr arrays.
            if not zarr_path.is_dir():
                raise ValueError(
                    f"Cannot open {zarr_path}: not a zarr array, group, "
                    f"or directory of per-timepoint zarrs."
                )
            sub_keys = []
            for child in zarr_path.iterdir():
                if not child.is_dir():
                    continue
                try:
                    sub_keys.append(int(child.name))
                except ValueError:
                    continue
            if not sub_keys:
                raise ValueError(
                    f"No zarr array, group, or numeric subdirectories "
                    f"found at {zarr_path}"
                )
            sub_keys.sort()
            lazy_frames = []
            first_shape = None
            first_dtype = None
            for k in sub_keys:
                arr_path = zarr_path / str(k)
                arr = zarr.open_array(str(arr_path), mode="r")
                if first_shape is None:
                    first_shape = arr.shape
                    first_dtype = arr.dtype
                lazy_frames.append(da.from_zarr(str(arr_path)))
            data = da.stack(lazy_frames, axis=0)
            print(f"  Per-timepoint zarrs: {len(sub_keys)} timepoints, "
                  f"spatial={first_shape}, dtype={first_dtype}")

        ds = self._downsample_factor

        if self._preload:
            # ── PRELOAD MODE: fast load, consistent colors, instant scrub ─
            import multiprocessing as mp
            from concurrent.futures import ThreadPoolExecutor
            n_cores = mp.cpu_count() or 1
            lds = self._load_downsample_factor
            if lds > 1:
                # Downsample spatially before pulling into RAM so the
                # full-resolution array is never materialised.
                data = data[:, ::lds, ::lds, ::lds]
            nbytes_raw = data.nbytes
            lds_note = f", {lds}× load-downsample" if lds > 1 else ""
            print(f"  Preloading {nbytes_raw / 1e9:.1f} GB into RAM "
                  f"({n_cores} cores{lds_note}) ...")
            t0 = _time.time()
            data_np = data.compute(scheduler="threads",
                                   num_workers=n_cores)
            print(f"  Loaded in {_time.time() - t0:.0f}s")
            del data

            # Store (possibly load-downsampled) data for analysis
            self._segments_data = data_np

            # ── Build display frames ─────────────────────────────────
            # Each label gets a CONSISTENT colour across all timepoints
            # via Knuth multiplicative hash → uint8 intensity.
            # Same nucleus = same label ID = same colour always.
            #
            # Then: 3D Image layer + vispy direct texture swap bypasses
            # napari's entire 4D pipeline → zero Python overhead per
            # frame change.
            print(f"  Building {data_np.shape[0]} display frames "
                  f"({ds}× downsample) ...", end=" ", flush=True)
            t0 = _time.time()

            if ds > 1:
                self._display_labels = data_np[:, ::ds, ::ds, ::ds]
            else:
                self._display_labels = data_np
            # Total downsample of stored segment voxels vs. world coords:
            # load_downsample was applied to data_np before display ds.
            self._display_ds = ds * lds

            # Build initial frames with hash-based coloring
            self._segment_frames = self._build_segment_display_frames(
                self._display_labels, "hash", n_cores
            )

            frame_mb = self._segment_frames[0].nbytes / 1e6
            total_gb = len(self._segment_frames) * frame_mb / 1e3
            print(f"{len(self._segment_frames)} × {frame_mb:.1f} MB = "
                  f"{total_gb:.1f} GB ({_time.time() - t0:.1f}s)")

            # Transparent-background turbo colormap
            from napari.utils.colormaps import Colormap
            import matplotlib.pyplot as plt
            _lut = plt.get_cmap("turbo")(np.linspace(0, 1, 256))
            _lut[0] = [0, 0, 0, 0]  # background → fully transparent
            cmap = Colormap(colors=_lut.astype(np.float32),
                            name="turbo_nuclei")

            # Add as 3D layer — avoids napari's 4D slicing pipeline
            total_ds = ds * lds
            scale_3d = (total_ds, total_ds, total_ds)
            print(f"  Adding 3D Image layer ...")
            self._segments_layer = self.viewer.add_image(
                self._segment_frames[0],
                name="Nuclei 3D",
                scale=scale_3d,
                opacity=0.7,
                rendering="mip",
                colormap=cmap,
                contrast_limits=[0, 255],
                blending="translucent",
            )

            # Kill thumbnail (scipy.ndimage.zoom on every refresh)
            self._segments_layer._update_thumbnail = lambda: None

            # ── CRITICAL PERFORMANCE FIX ─────────────────────────────
            # napari's viewer._update_layers is connected to
            # dims.events.current_step AND dims.events.point.
            # On every frame tick it calls _slice_dims on EVERY layer:
            #   → Points: np.min/max on 2M×4 array (extent) + slice filter
            #   → Tracks: full _set_view_slice + thumbnail + extent
            #   → Image:  slice + thumbnail + extent
            # This is the ENTIRE source of lag.  GPU sits idle waiting.
            #
            # Fix: disconnect napari's _update_layers from the time
            # slider events.  Our own minimal callback handles segments
            # via vispy texture swap, and slices other layers with
            # thumbnail/extent monkey-patched out.
            viewer_model = self.viewer
            try:
                viewer_model.dims.events.current_step.disconnect(
                    viewer_model._update_layers
                )
                viewer_model.dims.events.point.disconnect(
                    viewer_model._update_layers
                )
                # Monkey-patch out the expensive per-frame operations on
                # all current non-segment layers so that _slice_dims only
                # does the cheap data-filtering part.
                for _lyr in list(self.viewer.layers):
                    if _lyr is not self._segments_layer:
                        _lyr._update_thumbnail = lambda: None
                        _lyr._clear_extent = lambda: None
                print("  ✓ Disconnected napari per-frame layer processing "
                      "(eliminates 2M-point filtering per frame)")
            except (ValueError, TypeError) as exc:
                print(f"  ⚠ Could not disconnect _update_layers: {exc}")

            # Find vispy Volume node for direct GPU texture swap
            self._vispy_volume = None
            self._vispy_canvas = None
            try:
                qt_viewer = self.viewer.window._qt_viewer
                canvas = qt_viewer.canvas
                if hasattr(canvas, 'layer_to_visual'):
                    vl = canvas.layer_to_visual.get(self._segments_layer)
                    if vl is not None and hasattr(vl, 'node'):
                        node = vl.node
                        if hasattr(node, 'set_data'):
                            self._vispy_volume = node
                            self._vispy_canvas = canvas
                            print("  ✓ Vispy direct texture path active")
            except Exception as exc:
                print(f"  ⚠ Vispy path unavailable ({exc})")

            # Connect time slider → instant frame swap (ONLY segments)
            # NOTE: we must NOT capture self._segment_frames as a default arg
            # because _recolor_segments() replaces the list; we need the
            # callback to always read the current list from self.

            # Store canvas ref for Tracks vispy visual lookup each frame.
            # We look up dynamically because _rebuild_tracks() replaces
            # Tracks layers (and thus their vispy visuals).
            self._qt_canvas = None
            try:
                self._qt_canvas = self.viewer.window._qt_viewer.canvas
            except Exception:
                pass

            def _on_segments_time(event,
                                  vol=self._vispy_volume,
                                  canv=self._vispy_canvas):
                t = int(self.viewer.dims.current_step[0])

                # ── 1. Swap segments vispy texture (instant) ──
                # When a track selection is active, apply the selection LUT
                # on-the-fly to the current frame (avoids prebuilding all).
                sel_lut = self._selection_lut
                if self._selection_active and sel_lut is not None:
                    dl = getattr(self, '_display_labels', None)
                    if dl is not None and 0 <= t < dl.shape[0]:
                        frame = sel_lut[dl[t]]
                        if vol is not None:
                            vol.set_data(frame)
                            canv.native.update()
                        elif self._segments_layer is not None:
                            self._segments_layer.data = frame
                        seg_lyr = self._segments_layer
                        if seg_lyr is not None:
                            try:
                                seg_lyr._data = frame
                            except Exception:
                                pass
                else:
                    frames = self._segment_frames
                    if frames is not None and 0 <= t < len(frames):
                        if vol is not None:
                            vol.set_data(frames[t])
                            canv.native.update()
                        elif self._segments_layer is not None:
                            self._segments_layer.data = frames[t]
                        seg_lyr = self._segments_layer
                        if seg_lyr is not None:
                            try:
                                seg_lyr._data = frames[t]
                            except Exception:
                                pass

                # ── 1b. Swap raw vispy texture (instant) ──
                processed_frames = self._processed_frames
                if processed_frames is not None and 0 <= t < len(processed_frames):
                    processed_vol = self._vispy_processed_volume
                    if processed_vol is not None:
                        processed_vol.set_data(processed_frames[t])
                    # Sync raw layer._data reference (same as segments)
                    processed_lyr = self._processed_layer
                    if processed_lyr is not None:
                        try:
                            processed_lyr._data = processed_frames[t]
                        except Exception:
                            pass

                # ── 2. Update Tracks shader time (GPU uniform, ~0 ms) ──
                # Tracks layers use a GPU shader uniform for time filtering
                # so we just poke the uniform — zero Python data processing.
                canvas = self._qt_canvas
                if canvas is not None:
                    for layer in list(self.viewer.layers):
                        if not hasattr(layer, 'current_time'):
                            continue  # not a Tracks layer
                        try:
                            vl = canvas.layer_to_visual.get(layer)
                            if vl is not None:
                                vl.node.tracks_filter.current_time = t
                                vl.node.graph_filter.current_time = t
                                vl.node.update()
                        except Exception:
                            pass

            self._segments_time_cb = _on_segments_time
            self.viewer.dims.events.current_step.connect(_on_segments_time)

            print(f"  Ready: {data_np.shape[0]} tp  |  "
                  f"full-res {data_np.shape[1:]} "
                  f"({data_np.nbytes / 1e9:.1f} GB)  |  "
                  f"display {self._segment_frames[0].shape} "
                  f"({total_gb:.1f} GB)  |  "
                  f"scale {scale_3d}")
        else:
            # ── LAZY MODE: dask, low RAM, slower scrubbing ──
            if ds > 1:
                data = data[:, ::ds, ::ds, ::ds]
                print(f"  Lazy downsample {ds}× → effective shape {data.shape}")

            self._segments_data = data
            # Match tracks coordinate system (raw voxel units)
            scale_4d = (1, ds, ds, ds)

            self._segments_layer = self.viewer.add_labels(
                data,
                name="Nuclei 3D",
                scale=scale_4d,
                opacity=0.5,
            )

            est_frame_mb = (data[0].nbytes / 1e6) if hasattr(data[0], 'nbytes') else 0
            print(f"  Labels layer (lazy): {data.shape[0]} tp, "
                  f"~{est_frame_mb:.0f} MB/frame, scale={scale_4d}")

        self.viewer.dims.axis_labels = ["t", "z", "y", "x"]

    def _build_segment_display_frames(
        self, display_labels: np.ndarray, mode: str,
        n_cores: int | None = None,
    ) -> list[np.ndarray]:
        """Build uint8 display frames from 4D label volume.

        Parameters
        ----------
        display_labels : (T, Z, Y, X) int array — possibly downsampled
        mode : one of 'hash', 'frame', 'depth', 'speed', 'theta', 'phi',
               'radial_vel', 'ingressing', 'roi', 'depth_travel'
        n_cores : workers for parallel build
        """
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor

        if n_cores is None:
            n_cores = mp.cpu_count() or 1
        n_tp = display_labels.shape[0]

        # Build a label→uint8 LUT based on the requested mode.
        # For modes that use tracks/spots data, we look up each label's
        # value from the spots DataFrame (which links track_id to labels).
        lut = self._build_label_color_lut(display_labels, mode)

        def _apply_lut(t):
            return np.ascontiguousarray(lut[display_labels[t]])

        with ThreadPoolExecutor(max_workers=n_cores) as pool:
            frames = list(pool.map(_apply_lut, range(n_tp)))
        return frames

    def _build_label_color_lut(
        self, display_labels: np.ndarray, mode: str,
    ) -> np.ndarray:
        """Build label_id → uint8 intensity lookup table for a given mode.

        Returns an array of shape (max_label+1,) dtype uint8.
        Background (0) always maps to 0 (transparent).
        """
        max_label = int(display_labels.max())
        lut = np.zeros(max_label + 1, dtype=np.uint8)

        if mode == "hash" or self.spots is None:
            # Knuth multiplicative hash for distinct per-nucleus colors
            ids = np.arange(1, max_label + 1, dtype=np.uint64)
            lut[1:] = ((ids * np.uint64(2654435761)) % np.uint64(251)
                       + np.uint64(5)).astype(np.uint8)
            return lut

        df = self.spots
        # Map TRACK_ID → label ID.  In ultrack, track_id IS the label
        # (the label in segments.zarr is the track_id for that spot).
        # Build a per-label value from spots data.
        if "TRACK_ID" not in df.columns:
            return self._build_label_color_lut(display_labels, "hash")

        # Get one representative value per track for the given mode
        grp = df.groupby("TRACK_ID")

        if mode == "frame":
            # Mean frame → early tracks dark, late tracks bright
            vals = grp["FRAME"].mean()
        elif mode == "depth" and "SPHERICAL_DEPTH" in df.columns:
            vals = grp["SPHERICAL_DEPTH"].mean()
        elif mode == "depth_travel" and "SPHERICAL_DEPTH" in df.columns:
            vals = grp["SPHERICAL_DEPTH"].apply(lambda s: s.max() - s.min())
        elif mode == "theta" and "THETA_DEG" in df.columns:
            vals = grp["THETA_DEG"].mean()
        elif mode == "phi" and "PHI_DEG" in df.columns:
            vals = grp["PHI_DEG"].mean()
        elif mode == "speed":
            speed = self._compute_spot_speed()
            df_tmp = df.copy()
            df_tmp["_speed"] = speed
            vals = df_tmp.groupby("TRACK_ID")["_speed"].mean()
        elif mode == "radial_vel" and "RADIAL_VELOCITY_SMOOTH" in df.columns:
            vals = grp["RADIAL_VELOCITY_SMOOTH"].mean()
        elif mode == "ingressing" and "INGRESSING" in df.columns:
            # 1 if any spot in track is ingressing, 0 otherwise
            vals = grp["INGRESSING"].max().astype(float)
        elif mode == "roi" and "IN_ROI" in df.columns:
            vals = grp["IN_ROI"].max().astype(float)
        else:
            return self._build_label_color_lut(display_labels, "hash")

        # Normalise to [5..255] (0 reserved for background/transparent)
        v = vals.values.astype(float)
        track_ids = vals.index.values

        # Special handling for binary modes
        if mode in ("ingressing", "roi"):
            # Binary: 0 → dim (20), 1 → bright (240)
            for tid, flag in zip(track_ids, v):
                if 0 < tid <= max_label:
                    lut[int(tid)] = 240 if flag > 0.5 else 20
            return lut

        # Special handling for diverging mode (radial_vel)
        if mode == "radial_vel":
            nan_mask = np.isnan(v)
            v[nan_mask] = 0.0
            valid = v[~nan_mask]
            if len(valid) > 0:
                vmax = max(np.abs(np.nanpercentile(valid, [2, 98])).max(), 1e-6)
                v_clip = np.clip(v / vmax, -1, 1)
                # Map [-1, 1] → [5, 255]
                mapped = ((v_clip + 1) / 2 * 250 + 5).astype(np.uint8)
            else:
                mapped = np.full(len(v), 128, dtype=np.uint8)
            for tid, val in zip(track_ids, mapped):
                if 0 < tid <= max_label:
                    lut[int(tid)] = val
            return lut

        # Continuous modes: normalise to [5..255]
        nan_mask = np.isnan(v)
        v[nan_mask] = 0.0
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        rng = max(vmax - vmin, 1e-10)
        v_norm = (v - vmin) / rng
        mapped = (v_norm * 250 + 5).astype(np.uint8)
        mapped[nan_mask] = 20  # dim for NaN

        for tid, val in zip(track_ids, mapped):
            if 0 < tid <= max_label:
                lut[int(tid)] = val

        return lut

    def _recolor_segments(self, mode: str) -> None:
        """Recolor the 3D nuclei volume to match a given color mode.

        Rebuilds all display frames with the new LUT and updates the
        current vispy texture immediately.
        """
        if not hasattr(self, '_display_labels') or self._display_labels is None:
            return
        if not hasattr(self, '_segment_frames') or self._segment_frames is None:
            return

        import time as _time
        import multiprocessing as mp

        n_cores = mp.cpu_count() or 1
        print(f"  Recoloring nuclei 3D by '{mode}' ...", end=" ", flush=True)
        t0 = _time.time()

        self._segment_frames = self._build_segment_display_frames(
            self._display_labels, mode, n_cores
        )

        # Update current frame display
        t = int(self.viewer.dims.current_step[0])
        if 0 <= t < len(self._segment_frames):
            vol = getattr(self, '_vispy_volume', None)
            if vol is not None:
                vol.set_data(self._segment_frames[t])
                canv = getattr(self, '_vispy_canvas', None)
                if canv is not None:
                    canv.native.update()
            elif self._segments_layer is not None:
                self._segments_layer.data = self._segment_frames[t]

        # Choose appropriate colormap for the mode
        self._update_segments_colormap(mode)

        print(f"done ({_time.time() - t0:.1f}s)")

    def _update_segments_colormap(self, mode: str) -> None:
        """Switch the segments Image layer colormap to match the mode."""
        if self._segments_layer is None:
            return

        from napari.utils.colormaps import Colormap
        import matplotlib.pyplot as plt

        if mode == "radial_vel":
            # Diverging: blue-white-red
            _lut = plt.get_cmap("coolwarm")(np.linspace(0, 1, 256))
        elif mode == "ingressing":
            # Binary: dim grey vs bright magenta
            _lut = plt.get_cmap("magma")(np.linspace(0, 1, 256))
        elif mode == "roi":
            # Binary: dim grey vs bright yellow
            _lut = plt.get_cmap("YlOrBr")(np.linspace(0, 1, 256))
        elif mode in ("depth", "depth_travel"):
            _lut = plt.get_cmap("viridis")(np.linspace(0, 1, 256))
        elif mode == "speed":
            _lut = plt.get_cmap("hot")(np.linspace(0, 1, 256))
        elif mode in ("theta", "phi"):
            _lut = plt.get_cmap("hsv")(np.linspace(0, 1, 256))
        else:
            # hash / frame → turbo
            _lut = plt.get_cmap("turbo")(np.linspace(0, 1, 256))

        _lut[0] = [0, 0, 0, 0]  # background → transparent
        cmap = Colormap(colors=_lut.astype(np.float32),
                        name=f"nuclei_{mode}")
        self._segments_layer.colormap = cmap

    @staticmethod
    def _remap_4d_labels(data: np.ndarray) -> np.ndarray:
        """Remap a 4D int32/int64 label array to contiguous uint16 per frame.

        Parallelised across CPU cores using shared-memory output array.
        Avoids np.unique (which sorts the entire volume) — instead uses
        a scatter approach with np.flatnonzero on a boolean mask, which
        is 5-10× faster for sparse label volumes.
        """
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor

        n_tp = data.shape[0]
        out = np.zeros(data.shape, dtype=np.uint16)

        def _remap_one(t: int) -> None:
            vol = data[t]
            if vol.max() == 0:
                return
            # Fast unique via boolean scatter — avoids sorting the
            # entire volume.  For ~5000 labels in 58M voxels this is
            # much faster than np.unique which sorts all 58M values.
            max_id = int(vol.max())
            seen = np.zeros(max_id + 1, dtype=bool)
            seen[vol.ravel()] = True
            seen[0] = False  # background
            unique_ids = np.flatnonzero(seen)
            n_labels = len(unique_ids)
            if n_labels == 0:
                return
            if n_labels > 65535:
                unique_ids = unique_ids[:65535]
                n_labels = 65535
            lut = np.zeros(max_id + 1, dtype=np.uint16)
            lut[unique_ids] = np.arange(1, n_labels + 1, dtype=np.uint16)
            out[t] = lut[vol]

        n_workers = min(n_tp, mp.cpu_count() or 1)
        # Use threads — the work is numpy C-level so releases the GIL
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_remap_one, range(n_tp)))

        return out

    def _load_segments_from_ui(self) -> None:
        """Open a folder dialog to load a segments.zarr directory."""
        from qtpy.QtWidgets import QFileDialog

        zarr_dir = QFileDialog.getExistingDirectory(
            self.viewer.window._qt_window,
            "Select segments.zarr directory",
        )
        if not zarr_dir:
            return
        try:
            # Remove existing segments layer + callback
            if self._segments_layer is not None:
                # Disconnect time callback if exists
                cb = getattr(self, '_segments_time_cb', None)
                if cb is not None:
                    try:
                        self.viewer.dims.events.current_step.disconnect(cb)
                    except (ValueError, TypeError):
                        pass
                    self._segments_time_cb = None
                try:
                    self.viewer.layers.remove(self._segments_layer)
                except (ValueError, KeyError):
                    pass
                self._segments_layer = None
                self._segments_original_colormap = None
            self._load_segments(zarr_dir)
            self.lbl_segments_status.value = (
                f"Loaded: {self._segments_data.shape}"
            )
        except Exception as e:
            self.lbl_segments_status.value = f"Error: {e}"

    def _on_segments_opacity_changed(self, value: float) -> None:
        if self._segments_layer is not None:
            self._segments_layer.opacity = value / 100.0

    def _on_segments_visible_changed(self, value: bool) -> None:
        if self._segments_layer is not None:
            self._segments_layer.visible = value

    def _deferred_texture_repush(self):
        """Re-push the current timepoint's vispy texture after napari finishes.

        NOTE: As of the in-place layer.data sync in _on_segments_time,
        this is only needed as a safety-net for edge cases (e.g. layer
        loaded while hidden).
        """
        from qtpy.QtCore import QTimer

        def _push():
            t = int(self.viewer.dims.current_step[0])
            # ── segments ──
            frames = self._segment_frames
            vol = self._vispy_volume
            canv = self._vispy_canvas
            if frames is not None and vol is not None and 0 <= t < len(frames):
                vol.set_data(frames[t])
                if canv is not None:
                    canv.native.update()
            # ── raw ──
            processed_frames = self._processed_frames
            processed_vol = self._vispy_processed_volume
            if processed_frames is not None and processed_vol is not None and 0 <= t < len(processed_frames):
                processed_vol.set_data(processed_frames[t])
                if canv is not None:
                    canv.native.update()

        QTimer.singleShot(50, _push)

    # ── Raw image loading ─────────────────────────────────────────────

    @staticmethod
    def _collect_timepoint_files(folder: Path) -> list[Path]:
        """Return a time-sorted list of per-timepoint files in ``folder``.

        Accepts both ``.tif``/``.tiff`` files and per-timepoint zarr
        directories (sub-folders that look like zarr arrays). Sorting
        uses the trailing numeric component in the stem, so any of
        ``tp_0.tif``, ``t0001.tif``, ``frame12.tif`` etc. are ordered
        correctly. Numeric stems take precedence over pure-lexicographic
        ordering.
        """
        import re

        candidates: list[Path] = []
        for child in folder.iterdir():
            if child.is_file() and child.suffix.lower() in ('.tif', '.tiff'):
                candidates.append(child)
            elif child.is_dir() and child.name.endswith('.zarr'):
                candidates.append(child)
        if not candidates:
            return []

        num_re = re.compile(r'(\d+)(?!.*\d)')  # last integer in stem

        def sort_key(p: Path):
            m = num_re.search(p.stem)
            if m is not None:
                return (0, int(m.group(1)), p.name)
            return (1, 0, p.name)

        candidates.sort(key=sort_key)
        return candidates

    def _stack_timepoint_files(self, files: list[Path]) -> np.ndarray:
        """Load a list of per-timepoint files and stack them on the time axis.

        Each file is expected to be a 3D volume (Z, Y, X) — either a
        TIFF page-stack (handled via ImageJ metadata when present) or a
        zarr array. Returns a 4D numpy array (T, Z, Y, X).
        """
        import time as _time
        import tifffile
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor

        def _load_one(p: Path) -> np.ndarray:
            if p.is_dir() and p.name.endswith('.zarr'):
                import zarr
                arr = zarr.open_array(str(p), mode="r")
                return np.asarray(arr)
            # TIFF
            with tifffile.TiffFile(str(p)) as tf:
                arr = tf.asarray()
                ij = tf.imagej_metadata or {}
            n_frames = int(ij.get('frames', 0) or 0)
            n_slices = int(ij.get('slices', 0) or 0)
            if arr.ndim == 3 and n_frames > 1 and n_slices > 1:
                expected = n_frames * n_slices
                if arr.shape[0] == expected:
                    arr = arr.reshape(
                        n_frames, n_slices, arr.shape[1], arr.shape[2]
                    )
                    arr = arr[0]  # each file = single timepoint, take Z
            return arr

        n_cores = mp.cpu_count() or 1
        n_tp = len(files)
        first_shape = None
        first_dtype = None
        print(f"  Loading {n_tp} timepoint files ({n_cores} threads) ...",
              end=" ", flush=True)
        t0 = _time.time()
        frames: list[np.ndarray] = []
        # Parallel I/O + decode (decompression is GIL-released in tifffile)
        with ThreadPoolExecutor(max_workers=min(n_cores, n_tp)) as ex:
            for i, arr in enumerate(ex.map(_load_one, files)):
                # If a TIFF came back as a multi-timepoint stack, split it.
                if arr.ndim == 4:
                    frames.extend(arr)
                else:
                    frames.append(arr)
                if first_shape is None:
                    first_shape = arr.shape[-3:]
                    first_dtype = arr.dtype
                if i % max(1, n_tp // 10) == 0:
                    print(f"{i + 1}...", end=" ", flush=True)
        print(f"done ({_time.time() - t0:.0f}s)")

        # Validate uniform shape/dtype across timepoints
        for i, arr in enumerate(frames):
            if arr.shape != first_shape:
                raise ValueError(
                    f"Timepoint {i} ({files[i].name}) has shape "
                    f"{arr.shape}, expected {first_shape}"
                )

        data_np = np.stack(frames, axis=0).astype(first_dtype, copy=False)
        print(f"  Stacked {data_np.shape[0]} tp × {first_shape} "
              f"({data_np.nbytes / 1e9:.1f} GB, dtype={data_np.dtype})")
        return data_np

    def _load_processed(self, path: str | Path) -> None:
        """Load a processed image (TIFF, zarr, or folder of timepoints).

        Accepted inputs:
          • A 4D TIFF hyperstack (T×Z×Y×X, with ImageJ metadata).
          • A zarr array, zarr group with numeric subarrays, or a `.zarr`
            directory.
          • A directory containing one file/sub-zarr per timepoint
            (e.g. ``tp_000.tif``, ``tp_001.tif``, ...). Files are
            sorted by the trailing numeric component in the stem and
            stacked along the time axis.

        Preloads into RAM and uses the same vispy direct-texture-swap
        approach as segments for instant smooth scrubbing.
        """
        import time as _time

        path = Path(path).resolve()
        print(f"Loading processed image: {path}")

        # ── Folder of per-timepoint files (TIFF or zarr) ─────────────
        if path.is_dir() and path.suffix != '.zarr':
            tp_files = self._collect_timepoint_files(path)
            if not tp_files:
                raise ValueError(
                    f"No timepoint files (.tif/.tiff or zarr sub-dirs) "
                    f"found in {path}"
                )
            print(f"  Folder of timepoints: {len(tp_files)} files detected")
            data_np = self._stack_timepoint_files(tp_files)
        elif path.suffix in ('.tif', '.tiff'):
            import tifffile
            t0 = _time.time()
            # Read pages + ImageJ metadata. Hyperstacks written by
            # merge_hyperstack.py use bigtiff=True without imagej=True
            # (the description is set manually), so tifffile returns a
            # plain 3D (T*Z, Y, X) page stack rather than a true 4D
            # (T, Z, Y, X) array. We reshape using the ImageJ metadata.
            with tifffile.TiffFile(str(path)) as tf:
                data_np = tf.asarray()
                ij = tf.imagej_metadata or {}
            n_frames = int(ij.get('frames', 0) or 0)
            n_slices = int(ij.get('slices', 0) or 0)
            n_channels = int(ij.get('channels', 1) or 1)
            print(f"  TIFF shape={data_np.shape}, dtype={data_np.dtype}, "
                  f"loaded in {_time.time() - t0:.0f}s")
            if data_np.ndim == 3 and n_frames > 1 and n_slices > 1:
                if n_channels != 1:
                    raise ValueError(
                        f"Multi-channel hyperstacks not supported "
                        f"(channels={n_channels}); split channels first."
                    )
                expected = n_frames * n_slices
                if data_np.shape[0] != expected:
                    raise ValueError(
                        f"ImageJ metadata says T={n_frames}, Z={n_slices} "
                        f"(={expected} pages) but TIFF has {data_np.shape[0]} "
                        f"pages — refusing to reshape."
                    )
                data_np = data_np.reshape(
                    n_frames, n_slices, data_np.shape[1], data_np.shape[2]
                )
                print(f"  Reshaped to 4D (T,Z,Y,X)={data_np.shape} "
                      f"using ImageJ metadata")
        elif path.is_dir() or path.suffix == '.zarr':
            import dask.array as da
            import zarr
            import multiprocessing as mp
            store = zarr.open(str(path), mode="r")
            if isinstance(store, zarr.Array):
                data = da.from_zarr(str(path))
            elif isinstance(store, zarr.Group):
                keys = sorted(int(k) for k in store.keys())
                lazy = [da.from_zarr(str(path), component=str(k)) for k in keys]
                data = da.stack(lazy, axis=0)
            else:
                raise ValueError(f"Unexpected zarr layout: {path}")
            n_cores = mp.cpu_count() or 1
            print(f"  Zarr shape={data.shape}, dtype={data.dtype}  "
                  f"({data.nbytes / 1e9:.1f} GB)")
            print(f"  Preloading into RAM ({n_cores} cores) ...", end=" ",
                  flush=True)
            t0 = _time.time()
            data_np = data.compute(scheduler="threads", num_workers=n_cores)
            print(f"done ({_time.time() - t0:.0f}s)")
            del data
        else:
            raise ValueError(
                f"Unsupported processed format: {path.suffix}  "
                "(expected .tif, .tiff, .zarr, or a folder of "
                "per-timepoint files)"
            )

        # Ensure 4D (T, Z, Y, X)
        if data_np.ndim == 3:
            data_np = data_np[np.newaxis]  # single timepoint
        if data_np.ndim != 4:
            raise ValueError(
                f"Processed image must be 4D (T,Z,Y,X), got {data_np.ndim}D"
            )

        self._processed_data = data_np
        n_tp = data_np.shape[0]

        # Build per-frame contiguous arrays for vispy texture swap.
        # Normalise to uint8 for GPU efficiency (one global contrast).
        print(f"  Building {n_tp} display frames ...", end=" ", flush=True)
        t0 = _time.time()

        # Percentile-based contrast (robust to hot pixels / outliers)
        # Subsample for speed: take every 4th frame, every 2nd voxel
        sample = data_np[::4, ::2, ::2, ::2].ravel()
        p_low, p_high = np.percentile(sample[sample > 0], [0.5, 99.5]) \
            if np.any(sample > 0) else (float(data_np.min()), float(data_np.max()))
        p_low = float(p_low)
        p_high = float(p_high)
        scale_factor = 255.0 / max(p_high - p_low, 1e-10)
        print(f"contrast range [{p_low:.0f}, {p_high:.0f}] ", end="")

        self._processed_frames = []
        for i in range(n_tp):
            frame = data_np[i]
            # Convert to uint8 with percentile-based contrast
            f8 = np.clip((frame.astype(np.float32) - p_low) * scale_factor,
                         0, 255).astype(np.uint8)
            self._processed_frames.append(np.ascontiguousarray(f8))

        frame_mb = self._processed_frames[0].nbytes / 1e6
        total_gb = n_tp * frame_mb / 1e3
        print(f"{n_tp} × {frame_mb:.1f} MB = {total_gb:.1f} GB "
              f"({_time.time() - t0:.1f}s)")

        # Processed image is loaded at full resolution → world scale (1,1,1).
        scale_3d = (1, 1, 1)

        # Add as 3D Image layer (grayscale, additive for overlay with segments)
        self._processed_layer = self.viewer.add_image(
            self._processed_frames[0],
            name="Processed",
            scale=scale_3d,
            opacity=0.7,
            rendering="mip",
            colormap="gray",
            contrast_limits=[0, 255],
            blending="additive",
            gamma=0.7,
        )
        self._processed_layer._update_thumbnail = lambda: None
        self._processed_layer._clear_extent = lambda: None

        # Find vispy Volume node for direct GPU texture swap
        self._vispy_processed_volume = None
        try:
            qt_viewer = self.viewer.window._qt_viewer
            canvas = qt_viewer.canvas
            if hasattr(canvas, 'layer_to_visual'):
                vl = canvas.layer_to_visual.get(self._processed_layer)
                if vl is not None and hasattr(vl, 'node'):
                    if hasattr(vl.node, 'set_data'):
                        self._vispy_processed_volume = vl.node
                        print("  ✓ Processed vispy direct texture path active")
        except Exception as exc:
            print(f"  ⚠ Processed vispy path unavailable ({exc})")

        print(f"  Ready: {n_tp} tp  |  shape {data_np.shape[1:]}  |  "
              f"{data_np.nbytes / 1e9:.1f} GB processed → "
              f"{total_gb:.1f} GB display (uint8)")

    def _load_processed_from_ui(self):
        """Open dialog to load a processed image (file, .zarr, or folder)."""
        from qtpy.QtWidgets import QFileDialog

        # Give the user a single dialog with both "Open File" and
        # "Open Folder" options. QFileDialog doesn't expose a built-in
        # mode switch, so we ask: if they cancel file mode, fall back
        # to directory mode.
        path, _ = QFileDialog.getOpenFileName(
            None, "Open Processed Hyperstack (or Cancel and choose a folder)",
            "", "Image files (*.tif *.tiff *.zarr);;All (*)",
        )
        if not path:
            folder = QFileDialog.getExistingDirectory(
                None,
                "Select folder of per-timepoint files "
                "(TIFFs or per-timepoint .zarr sub-dirs)",
            )
            if not folder:
                return
            path = folder
        try:
            if self._processed_layer is not None:
                try:
                    self.viewer.layers.remove(self._processed_layer)
                except (ValueError, KeyError):
                    pass
                self._processed_layer = None
            self._load_processed(path)
            self.lbl_processed_status.value = f"Loaded: {self._processed_data.shape}"
        except Exception as e:
            self.lbl_processed_status.value = f"Error: {e}"

    def _on_processed_opacity_changed(self, value: float) -> None:
        if self._processed_layer is not None:
            self._processed_layer.opacity = value / 100.0

    def _on_processed_visible_changed(self, value: bool) -> None:
        if self._processed_layer is not None:
            self._processed_layer.visible = value

    def _add_spots_layer(self):
        """Add the main nuclei points layer."""
        data, idx = self._build_display_points()
        self._display_idx = idx
        colors = self._get_display_colors()
        self.spots_layer = self.viewer.add_points(
            data,
            name="Nuclei",
            size=2,
            face_color=colors,
            border_width=0,
            ndim=4,
            out_of_slice_display=False,
        )

    def _build_display_points(self):
        """Build display arrays, respecting display %, ROI, ingression, and selection filters."""
        df = self.spots
        n = len(df)
        mask = np.ones(n, dtype=bool)

        # Track selection filter
        if self._selection_active and self._selected_track_ids:
            if "TRACK_ID" in df.columns:
                mask &= df["TRACK_ID"].isin(self._selected_track_ids).values

        # ROI filter
        if self._roi_only and self._roi_bounds is not None:
            b = self._roi_bounds
            mask &= (
                (df["POSITION_X"].values >= b["x_min"])
                & (df["POSITION_X"].values <= b["x_max"])
                & (df["POSITION_Y"].values >= b["y_min"])
                & (df["POSITION_Y"].values <= b["y_max"])
            )

        # Ingression filter (when "show ingressing tracks only" is checked)
        _ing_w = getattr(self, "ingression_tracks_only", None)
        if _ing_w is not None and _ing_w.value and "INGRESSING" in df.columns:
            ing_tids = df.loc[df["INGRESSING"].astype(bool), "TRACK_ID"].unique()
            mask &= df["TRACK_ID"].isin(ing_tids).values

        pool = np.where(mask)[0]

        pct = self._display_pct
        if pct < 100:
            n_show = max(1000, int(len(pool) * pct / 100))
            if n_show < len(pool):
                rng = np.random.default_rng(0)
                idx = rng.choice(pool, n_show, replace=False)
                idx.sort()
            else:
                idx = pool
        else:
            idx = pool

        data = np.column_stack(
            [
                df["FRAME"].values[idx],
                df["POSITION_Z"].values[idx],
                df["POSITION_Y"].values[idx],
                df["POSITION_X"].values[idx],
            ]
        )
        return data, idx

    def _get_display_colors(self) -> np.ndarray:
        """Compute face colors for the currently displayed subset based on current color mode."""
        idx = self._display_idx
        if idx is None or self.spots is None:
            return np.array([[0.5, 0.5, 0.5, 1.0]])
        df = self.spots
        mode = self._current_color_mode

        # ── Ingression mode: color by inward radial velocity ──
        # Each spot is colored by how strongly it is moving inward at
        # that moment.  Non-ingressing spots are ghostly grey; ingressing
        # spots transition from warm orange (gentle inward) to bright
        # magenta (strong inward dive), making the ingression movement
        # itself visually pop.
        if mode == "ingressing" and "INGRESSING" in df.columns:
            flags = df["INGRESSING"].values[idx].astype(bool)
            n_pts = len(idx)
            colors = np.tile([0.45, 0.45, 0.45, 0.06], (n_pts, 1))
            if flags.any() and "RADIAL_VELOCITY_SMOOTH" in df.columns:
                vr = df["RADIAL_VELOCITY_SMOOTH"].values[idx][flags].copy()
                vr = np.nan_to_num(vr, nan=0.0)
                # Normalise: 0 = no inward motion, 1 = strongest inward
                # Only negative V_r matters; clip positives to 0
                inward = np.clip(-vr, 0, None)  # flip sign: positive = inward
                vmax = (
                    np.percentile(inward[inward > 0], 95) if (inward > 0).any() else 1.0
                )
                t = np.clip(inward / max(vmax, 1e-6), 0, 1)
                # Gradient: warm orange (t=0) → hot magenta (t=0.5) → bright white-pink (t=1)
                colors[flags, 0] = 0.85 + 0.15 * t  # R: stays high
                colors[flags, 1] = 0.35 * (1 - t**0.8)  # G: fades
                colors[flags, 2] = 0.15 + 0.70 * t**0.6  # B: rises → magenta
                colors[flags, 3] = 0.65 + 0.35 * t  # A: brighter when diving
            elif flags.any():
                colors[flags] = [0.9, 0.2, 0.5, 0.85]  # flat magenta fallback
            return colors

        if mode == "roi" and "IN_ROI" in df.columns:
            flags = df["IN_ROI"].values[idx].astype(bool)
            hi = np.array([1, 0.9, 0, 1])
            lo = np.array([0.5, 0.5, 0.5, 0.12])
            return np.where(flags[:, None], hi, lo)

        # ── Diverging mode ──
        if mode == "radial_vel" and "RADIAL_VELOCITY_SMOOTH" in df.columns:
            v = df["RADIAL_VELOCITY_SMOOTH"].values[idx].copy()
            nan_mask = np.isnan(v)
            v[nan_mask] = 0.0
            valid = v[~nan_mask]
            vmax = (
                max(np.abs(np.nanpercentile(valid, [2, 98])).max(), 1e-6)
                if len(valid) > 0
                else 1.0
            )
            v_clip = np.clip(v / vmax, -1, 1)
            colors = np.zeros((len(v), 4))
            colors[:, 3] = 1.0
            neg = v_clip < 0
            pos = v_clip >= 0
            t_neg = 1 + v_clip[neg]
            colors[neg, 0] = t_neg
            colors[neg, 1] = t_neg
            colors[neg, 2] = 1.0
            t_pos = v_clip[pos]
            colors[pos, 0] = 1.0
            colors[pos, 1] = 1.0 - t_pos
            colors[pos, 2] = 1.0 - t_pos
            colors[nan_mask] = [0.5, 0.5, 0.5, 0.3]
            return colors

        # ── Continuous modes ──
        if mode == "depth" and "SPHERICAL_DEPTH" in df.columns:
            v = df["SPHERICAL_DEPTH"].values[idx]
        elif mode == "depth_travel" and "SPHERICAL_DEPTH" in df.columns and "TRACK_ID" in df.columns:
            # Per-track total depth range: max - min SPHERICAL_DEPTH
            depth_range = df.groupby("TRACK_ID")["SPHERICAL_DEPTH"].transform(
                lambda s: s.max() - s.min()
            )
            v = depth_range.values[idx].astype(float)
            # Spots without a track get NaN; give them grey
            nan_mask = np.isnan(v)
            v[nan_mask] = 0.0
            # Percentile clipping for better contrast
            valid = v[~nan_mask]
            if len(valid) > 0:
                vmax = max(np.percentile(valid, 98), 1e-6)
                v_norm = np.clip(v / vmax, 0, 1)
            else:
                v_norm = np.zeros_like(v)
            colors = _viridis_colors(v_norm)
            colors[nan_mask] = [0.4, 0.4, 0.4, 0.15]
            return colors
        elif mode == "theta" and "THETA_DEG" in df.columns:
            v = df["THETA_DEG"].values[idx]
        elif mode == "phi" and "PHI_DEG" in df.columns:
            v = df["PHI_DEG"].values[idx]
        elif mode == "speed":
            v = self._compute_spot_speed()[idx]
        else:
            # Default: colour by frame
            v = df["FRAME"].values[idx].astype(float)

        v = v.astype(float)
        v_norm = (v - np.nanmin(v)) / max(np.nanmax(v) - np.nanmin(v), 1e-10)
        return _viridis_colors(v_norm)

    def _add_tracks_layer(self):
        """Add tracks as a napari Tracks layer."""
        max_t = int(self._max_tracks)
        tracks = spots_to_napari_tracks(
            self.spots,
            max_tracks=max_t,
            frame_min=self._track_frame_min,
            frame_max=self._track_frame_max,
        )
        if tracks is not None and len(tracks) > 0:
            self.tracks_layer = self.viewer.add_tracks(
                tracks,
                name="Tracks",
                tail_width=1,
                tail_length=40,
                color_by="track_id",
            )
        else:
            self.tracks_layer = None

    def _build_tracks_data(self):
        """Build tracks array + filtered DataFrame applying all active filters.

        Respects: selection, ROI-only, ingression-only, time window, max tracks.
        Returns (tracks_array, df_filtered) or (None, None).
        """
        if self.spots is None or "TRACK_ID" not in self.spots.columns:
            return None, None

        df = self.spots.dropna(subset=["TRACK_ID"]).copy()
        if df.empty:
            return None, None

        # Track selection filter: restrict to selected tracks
        if self._selection_active and self._selected_track_ids:
            df = df[df["TRACK_ID"].isin(self._selected_track_ids)]
            if df.empty:
                return None, None

        # ROI filter: keep tracks that have at least one spot in ROI
        if self._roi_only and "IN_ROI" in df.columns:
            roi_tids = df.loc[df["IN_ROI"].astype(bool), "TRACK_ID"].unique()
            df = df[df["TRACK_ID"].isin(roi_tids)]

        # Ingression filter
        _ing_w = getattr(self, "ingression_tracks_only", None)
        if _ing_w is not None and _ing_w.value and "INGRESSING" in df.columns:
            ing_tids = df.loc[df["INGRESSING"].astype(bool), "TRACK_ID"].unique()
            df = df[df["TRACK_ID"].isin(ing_tids)]

        # Time window
        if self._track_frame_min is not None:
            df = df[df["FRAME"] >= self._track_frame_min]
        if self._track_frame_max is not None:
            df = df[df["FRAME"] <= self._track_frame_max]
        if df.empty:
            return None, None

        # Min track length
        track_len = df.groupby("TRACK_ID")["FRAME"].transform("count")
        df = df[track_len >= 3]
        if df.empty:
            return None, None

        # Subsample tracks
        tids = df["TRACK_ID"].unique()
        max_t = int(self._max_tracks)
        if len(tids) > max_t:
            rng = np.random.default_rng(42)
            tids = rng.choice(tids, max_t, replace=False)
            df = df[df["TRACK_ID"].isin(tids)]

        df = df.sort_values(["TRACK_ID", "FRAME"])
        tracks = np.column_stack(
            [
                df["TRACK_ID"].values,
                df["FRAME"].values,
                df["POSITION_Z"].values,
                df["POSITION_Y"].values,
                df["POSITION_X"].values,
            ]
        )
        return tracks, df

    def _rebuild_tracks(self):
        """Rebuild the tracks layer(s) with all active filters + current colour.

        In 'ingressing' colour mode, creates TWO layers:
          - "Other Tracks" (grey, thin, translucent)
          - "Ingressing Tracks" (magenta, thick, opaque)
        Otherwise creates a single "Tracks" layer.
        """
        if self._drawing_roi:
            return  # suppress rebuilds during ROI drawing
        self._remove_layer("tracks_layer")
        self._remove_layer("_tracks_layer_secondary")

        result = self._build_tracks_data()
        if result is None or result[0] is None:
            return
        tracks, df_t = result
        if len(tracks) == 0:
            return

        mode = self._current_color_mode

        # ── Dual-layer mode for ingression classification view ──
        #
        # Background layer: all non-ingressing tracks rendered in faint
        # uniform grey so they provide spatial context without distracting.
        #
        # Foreground layer: ingressing tracks colored by ONSET FRAME
        # using a warm 'inferno' colormap.  Early ingressors appear in
        # deep red/orange while later ones glow yellow/white, creating a
        # beautiful temporal accumulation effect as you scrub through time.
        # Ghost-point extension keeps every track visible through the
        # movie's last frame.
        #
        if mode == "ingressing" and "INGRESSING" in df_t.columns:
            ing_mask = df_t["INGRESSING"].values.astype(bool)

            # ── Background: non-ingressing tracks ──
            if (~ing_mask).any():
                df_other = df_t[~ing_mask]
                if len(df_other) >= 3:
                    tc = df_other.groupby("TRACK_ID")["FRAME"].transform("count")
                    df_other = df_other[tc >= 2].sort_values(["TRACK_ID", "FRAME"])
                    if len(df_other) > 0:
                        t_other = np.column_stack(
                            [
                                df_other["TRACK_ID"].values,
                                df_other["FRAME"].values,
                                df_other["POSITION_Z"].values,
                                df_other["POSITION_Y"].values,
                                df_other["POSITION_X"].values,
                            ]
                        )
                        self._tracks_layer_secondary = self.viewer.add_tracks(
                            t_other,
                            name="Other Tracks",
                            tail_width=max(0.5, self._track_width * 0.2),
                            tail_length=15,
                            color_by="track_id",
                            colormap="gray",
                            opacity=0.10,
                        )

            # ── Foreground: ingressing tracks (onset-coloured) ──
            if ing_mask.any():
                df_ing = df_t[ing_mask]
                tc = df_ing.groupby("TRACK_ID")["FRAME"].transform("count")
                df_ing = df_ing[tc >= 2].sort_values(["TRACK_ID", "FRAME"])
                if len(df_ing) > 0:
                    # Ghost-point extension: add synthetic point at the
                    # dataset's maximum frame so ingressing tracks persist
                    # visually through the end of the recording.
                    max_frame = int(self.spots["FRAME"].max())
                    last_pts = df_ing.groupby("TRACK_ID").tail(1).copy()
                    needs_ext = last_pts[last_pts["FRAME"] < max_frame]
                    if len(needs_ext) > 0:
                        ext = needs_ext.copy()
                        ext["FRAME"] = max_frame
                        df_ing = pd.concat([df_ing, ext], ignore_index=True)
                        df_ing = df_ing.sort_values(["TRACK_ID", "FRAME"])

                    t_ing = np.column_stack(
                        [
                            df_ing["TRACK_ID"].values,
                            df_ing["FRAME"].values,
                            df_ing["POSITION_Z"].values,
                            df_ing["POSITION_Y"].values,
                            df_ing["POSITION_X"].values,
                        ]
                    )
                    n_frames = (
                        int(self.spots["FRAME"].nunique())
                        if self.spots is not None
                        else 9999
                    )

                    # Color by radial velocity → emphasises the
                    # ingression movement itself.  Each point along the
                    # track is colored by how fast the cell dives inward
                    # at that moment: gentle inward = warm orange,
                    # strong dive = bright magenta/pink.
                    kw_ing = dict(
                        name="Ingressing Tracks",
                        tail_width=self._track_width,
                        tail_length=n_frames,
                    )
                    if "RADIAL_VELOCITY_SMOOTH" in df_ing.columns:
                        vr = df_ing["RADIAL_VELOCITY_SMOOTH"].values.copy()
                        vr = np.nan_to_num(vr, nan=0.0)
                        # Inward speed: flip sign so positive = diving in
                        inward = np.clip(-vr, 0, None)
                        vmax = (
                            np.percentile(inward[inward > 0], 95)
                            if (inward > 0).any()
                            else 1.0
                        )
                        # Normalize 0–1 for colormap
                        normed = np.clip(inward / max(vmax, 1e-6), 0, 1)
                        kw_ing["properties"] = {"inward_speed": normed}
                        kw_ing["color_by"] = "inward_speed"
                        kw_ing["colormap"] = "magma"
                    else:
                        kw_ing["color_by"] = "track_id"
                        kw_ing["colormap"] = "magenta"

                    self.tracks_layer = self.viewer.add_tracks(t_ing, **kw_ing)
            return

        # ── Single-layer mode (all other colour modes) ──
        col_map = {
            "depth": "SPHERICAL_DEPTH",
            "theta": "THETA_DEG",
            "phi": "PHI_DEG",
            "radial_vel": "RADIAL_VELOCITY_SMOOTH",
            "frame": "FRAME",
        }
        props = {}
        color_by = "track_id"
        cmap = "hsv"

        if mode == "speed":
            sp = self._compute_track_speed(df_t)
            nan_m = np.isnan(sp)
            if not nan_m.all():
                sp[nan_m] = 0.0
                props["color_value"] = sp
                color_by = "color_value"
                cmap = "viridis"
        elif mode == "depth_travel" and "SPHERICAL_DEPTH" in df_t.columns:
            # Per-track total depth range: colour each track by how far
            # it travels radially through the blastoderm
            depth_range = df_t.groupby("TRACK_ID")["SPHERICAL_DEPTH"].transform(
                lambda s: s.max() - s.min()
            ).values.astype(float)
            nan_m = np.isnan(depth_range)
            if not nan_m.all():
                depth_range[nan_m] = 0.0
                # Percentile clipping for better contrast across all tracks
                valid = depth_range[~nan_m]
                if len(valid) > 0:
                    vmax = max(np.percentile(valid, 98), 1e-6)
                    depth_range = np.clip(depth_range / vmax, 0, 1)
                props["color_value"] = depth_range
                color_by = "color_value"
                cmap = "plasma"
        elif mode in col_map and col_map[mode] in df_t.columns:
            vals = df_t[col_map[mode]].values.astype(float)
            nan_m = np.isnan(vals)
            if not nan_m.all():
                vals[nan_m] = np.nanmedian(vals)
                props["color_value"] = vals
                color_by = "color_value"
                cmap = "turbo" if mode == "radial_vel" else "viridis"

        kw = dict(
            name="Tracks",
            tail_width=self._track_width,
            tail_length=40,
            color_by=color_by,
            colormap=cmap,
        )
        if props:
            kw["properties"] = props
        self.tracks_layer = self.viewer.add_tracks(tracks, **kw)

    @staticmethod
    def _compute_track_speed(df_t: pd.DataFrame) -> np.ndarray:
        """Per-point speed (\u00b5m/frame) for a sorted tracks DataFrame."""
        tid = df_t["TRACK_ID"].values
        x, y, z = (
            df_t["POSITION_X"].values,
            df_t["POSITION_Y"].values,
            df_t["POSITION_Z"].values,
        )
        f = df_t["FRAME"].values.astype(float)
        dx = np.diff(x, prepend=np.nan)
        dy = np.diff(y, prepend=np.nan)
        dz = np.diff(z, prepend=np.nan)
        df_v = np.diff(f, prepend=-9999)
        same = np.diff(tid.astype(float), prepend=-9999) == 0
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dt = np.maximum(df_v, 1.0)
        return np.where(same & (df_v > 0), dist / dt, np.nan)

    def _remove_layer(self, attr_name: str):
        """Safely remove a napari layer stored as self.<attr_name>."""
        layer = getattr(self, attr_name, None)
        if layer is not None:
            try:
                self.viewer.layers.remove(layer)
            except (ValueError, KeyError):
                pass
            setattr(self, attr_name, None)

    def _reset_downstream(self):
        """Clear orientation / sphere / ROI state after new data load."""
        self.animal_pole = None
        self.dorsal_mark = None
        self.oriented = False
        self.sphere_fit_result = None
        self.transform_params = None
        self._roi_bounds = None
        self._roi_only = False
        self.roi_show_only.value = False
        if self._roi_shapes_layer is not None:
            try:
                self.viewer.layers.remove(self._roi_shapes_layer)
            except (ValueError, KeyError):
                pass
            self._roi_shapes_layer = None
        self.lbl_ap.value = "AP: not set"
        self.lbl_dorsal.value = "Dorsal: not set"
        self.lbl_sphere.value = "Sphere: not fitted"
        self.lbl_roi.value = "ROI: not set"
        self.lbl_ingression.value = "Ingression: not computed"
        self.ingression_tracks_only.value = False

        # Clear track selection
        self._selected_track_ids.clear()
        self._random_sampled_ids.clear()
        self._selection_active = False
        self._selection_lut = None
        self._follow_selection = False
        self._remove_layer("_selection_spots_overlay")
        self._remove_layer("_selection_tracks_overlay")
        if hasattr(self, "follow_selection_cb"):
            self.follow_selection_cb.value = False
        self._update_selection_label()

        # Update frame range sliders to new data
        if self.spots is not None:
            fmin = int(self.spots["FRAME"].min())
            fmax = int(self.spots["FRAME"].max())
            self.track_frame_start.min = fmin
            self.track_frame_start.max = fmax
            self.track_frame_start.value = fmin
            self.track_frame_end.min = fmin
            self.track_frame_end.max = fmax
            self.track_frame_end.value = fmax
            self._track_frame_min = None
            self._track_frame_max = None

            # Update max tracks slider to actual track count
            if "TRACK_ID" in self.spots.columns:
                n_tracks = max(500, int(self.spots["TRACK_ID"].nunique()))
                self.max_tracks_slider.max = n_tracks
                self.max_tracks_slider.value = n_tracks
                self._max_tracks = n_tracks

    def _load_spots_from_ui(self):
        """Open a file dialog to load a spots CSV (background thread)."""
        from qtpy.QtWidgets import QFileDialog
        from qtpy.QtCore import QTimer

        spots_path, _ = QFileDialog.getOpenFileName(
            self.viewer.window._qt_window,
            "Select Spots CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not spots_path:
            return

        self.lbl_spots_status.value = "\u23f3 Loading (please wait)..."
        self.viewer.status = "Loading spots in background thread..."

        self._pending_spots_df = None
        self._pending_spots_err = None

        def _bg():
            try:
                self._pending_spots_df = load_spots(spots_path)
            except Exception as e:
                self._pending_spots_err = str(e)

        t = threading.Thread(target=_bg, daemon=True)
        t.start()

        def _poll():
            if t.is_alive():
                QTimer.singleShot(250, _poll)
                return
            if self._pending_spots_err:
                self.lbl_spots_status.value = f"Error: {self._pending_spots_err}"
                self.viewer.status = f"Load failed: {self._pending_spots_err}"
                return
            self._finish_spots_load(self._pending_spots_df)

        QTimer.singleShot(250, _poll)

    def _finish_spots_load(self, df):
        """Finalise spots load on the main thread."""
        self.spots = df
        self._remove_layer("tracks_layer")
        self._remove_layer("_tracks_layer_secondary")
        self._remove_layer("spots_layer")

        self.viewer.status = "Building point cloud..."
        self._build_track_id_index()
        self._add_spots_layer()
        self._add_tracks_layer()

        n_s = f"{len(self.spots):,}"
        n_f = self.spots["FRAME"].nunique()
        has_tid = (
            "TRACK_ID" in self.spots.columns and self.spots["TRACK_ID"].notna().any()
        )
        msg = f"{n_s} spots, {n_f} frames"
        if self.tracks_layer is not None:
            msg += " + trajectories"
        elif not has_tid:
            msg += " (no TRACK_ID)"
        self.lbl_spots_status.value = msg
        self.viewer.status = msg + " | Use time slider at bottom to scrub frames"
        self._reset_downstream()
        self.viewer.reset_view()

    def _load_tracks_from_ui(self):
        """Open a file dialog to load a tracks summary CSV."""
        from qtpy.QtWidgets import QFileDialog, QApplication

        tracks_path, _ = QFileDialog.getOpenFileName(
            self.viewer.window._qt_window,
            "Select Tracks CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not tracks_path:
            return

        self.viewer.status = "Loading tracks summary..."
        QApplication.processEvents()
        try:
            fmt = _detect_csv_format(tracks_path)
            if fmt == "ultrack":
                self.tracks_raw = load_ultrack_csv(tracks_path)
            else:
                self.tracks_raw = load_trackmate_csv(tracks_path)
        except Exception as e:
            self.viewer.status = f"Error loading tracks: {e}"
            self.lbl_tracks_status.value = f"Error: {e}"
            return

        n_t = f"{len(self.tracks_raw):,}"
        cols = [c for c in self.tracks_raw.columns if "TRACK" in c.upper()]
        msg = f"{n_t} track records"
        self.lbl_tracks_status.value = msg
        self.viewer.status = msg

    def _build_widgets(self):
        """Create magicgui dock widgets for controls."""
        napari, magicgui = _import_napari()
        from magicgui import magicgui as mg_deco
        from magicgui.widgets import (
            PushButton,
            Label,
            FloatSlider,
            CheckBox,
            Container,
            SpinBox,
        )

        # Determine frame range for sliders
        if self.spots is not None:
            fmin_data = int(self.spots["FRAME"].min())
            fmax_data = int(self.spots["FRAME"].max())
        else:
            fmin_data, fmax_data = 0, 1000

        # ── File loading (separate buttons) ──
        btn_load_spots = PushButton(text="Load Spots CSV")
        btn_load_spots.changed.connect(self._load_spots_from_ui)
        self.lbl_spots_status = Label(
            value=f"{len(self.spots):,} spots loaded"
            if self.spots is not None
            else "No spots loaded"
        )

        btn_load_tracks = PushButton(text="Load Tracks CSV")
        btn_load_tracks.changed.connect(self._load_tracks_from_ui)
        self.lbl_tracks_status = Label(
            value=f"{len(self.tracks_raw):,} track records"
            if self.tracks_raw is not None
            else "No tracks loaded"
        )

        btn_load_segments = PushButton(text="Load Segments (.zarr)")
        btn_load_segments.changed.connect(self._load_segments_from_ui)
        self.lbl_segments_status = Label(
            value=f"Loaded: {self._segments_data.shape}"
            if self._segments_data is not None
            else "No segments loaded"
        )
        self.segments_visible_cb = CheckBox(value=True, text="Show 3D nuclei")
        self.segments_visible_cb.changed.connect(self._on_segments_visible_changed)
        self.segments_opacity_slider = FloatSlider(
            value=50, min=0, max=100, step=5, label="Nuclei 3D opacity %"
        )
        self.segments_opacity_slider.changed.connect(self._on_segments_opacity_changed)

        # ── Raw image controls ──
        btn_load_processed = PushButton(text="Load Processed Image")
        btn_load_processed.changed.connect(self._load_processed_from_ui)
        self.lbl_processed_status = Label(
            value=f"Loaded: {self._processed_data.shape}"
            if self._processed_data is not None
            else "No processed image loaded"
        )
        self.processed_visible_cb = CheckBox(value=True, text="Show processed image")
        self.processed_visible_cb.changed.connect(self._on_processed_visible_changed)
        self.processed_opacity_slider = FloatSlider(
            value=30, min=0, max=100, step=5, label="Processed opacity %"
        )
        self.processed_opacity_slider.changed.connect(self._on_processed_opacity_changed)

        # ── Display controls ──
        self.display_pct_slider = FloatSlider(
            value=100, min=1, max=100, step=1, label="Display %"
        )
        self.display_pct_slider.changed.connect(self._on_display_pct_changed)

        n_tracks_total = 500000
        if self.spots is not None and "TRACK_ID" in self.spots.columns:
            n_tracks_total = max(500, int(self.spots["TRACK_ID"].nunique()))
        self.max_tracks_slider = FloatSlider(
            value=n_tracks_total,
            min=100,
            max=n_tracks_total,
            step=500,
            label="Max tracks",
        )
        self._max_tracks = n_tracks_total
        self.max_tracks_slider.changed.connect(self._on_max_tracks_changed)

        self.track_width_slider = FloatSlider(
            value=2.0, min=0.5, max=10.0, step=0.5, label="Track width"
        )
        self.track_width_slider.changed.connect(self._on_track_width_changed)

        # ── Track time range ──
        self.track_frame_start = FloatSlider(
            value=fmin_data, min=fmin_data, max=fmax_data, step=1, label="Track start frame"
        )
        self.track_frame_end = FloatSlider(
            value=fmax_data, min=fmin_data, max=fmax_data, step=1, label="Track end frame"
        )
        self.track_frame_start.changed.connect(self._on_track_frame_changed)
        self.track_frame_end.changed.connect(self._on_track_frame_changed)

        # ── Track Selection ──
        btn_select_track = PushButton(text="\U0001f50d Select Track (click)")
        btn_select_track.changed.connect(self._toggle_select_track_mode)
        btn_clear_selection = PushButton(text="✕ Clear Selection")
        btn_clear_selection.changed.connect(self._clear_selection)
        self.lbl_selection = Label(value="No tracks selected")
        self.follow_selection_cb = CheckBox(
            value=False, text="Follow selected nuclei"
        )
        self.follow_selection_cb.changed.connect(self._on_follow_selection_changed)
        btn_goto_selection = PushButton(text="⊕ Go To Selection")
        btn_goto_selection.changed.connect(self._go_to_selection)

        # Random subsample (no-repeat)
        self.random_sample_n = SpinBox(
            value=50, min=1, max=10000, step=1, label="Random N"
        )
        btn_random_sample = PushButton(text="🎲 Sample N random (no repeat)")
        btn_random_sample.changed.connect(self._random_sample_tracks)
        btn_random_sample_reset = PushButton(text="↻ Reset random pool")
        btn_random_sample_reset.changed.connect(self._reset_random_sample_pool)

        # Filter segments volume by selection (colormap swap — fast)
        self.filter_segments_cb = CheckBox(
            value=True, text="Colour only selected nuclei in segments"
        )
        self.filter_segments_cb.changed.connect(
            self._on_filter_segments_changed)

        # Jump to a selected track's starting frame
        from magicgui.widgets import ComboBox
        self.jump_track_combo = ComboBox(
            choices=[("— none —", -1)], label="Jump to track"
        )
        self.jump_track_combo.changed.connect(self._on_jump_track_changed)

        # ── Landmark picking ──
        self._pick_mode = "none"

        btn_ap = PushButton(text="Pick Animal Pole")
        btn_ap.changed.connect(lambda: self._set_pick_mode("ap"))

        btn_dorsal = PushButton(text="Pick Dorsal")
        btn_dorsal.changed.connect(lambda: self._set_pick_mode("dorsal"))

        btn_margin = PushButton(text="Pick Margin")
        btn_margin.changed.connect(lambda: self._set_pick_mode("margin"))

        btn_ingression = PushButton(text="Pick Ingression Center")
        btn_ingression.changed.connect(lambda: self._set_pick_mode("ingression"))

        self.lbl_ap = Label(value="AP: not set")
        self.lbl_dorsal = Label(value="Dorsal: not set")
        self.lbl_margin = Label(value="Margin: not set")
        self.lbl_ingression = Label(value="Ingression: not set")

        # ── Orient ──
        btn_orient = PushButton(text="Orient Embryo")
        btn_orient.changed.connect(self._orient)

        # ── Sphere fit ──
        btn_sphere = PushButton(text="Fit Sphere")
        btn_sphere.changed.connect(self._fit_sphere)
        btn_pick_sphere_center = PushButton(text="📍 Pick Sphere Center")
        btn_pick_sphere_center.changed.connect(
            lambda: self._set_pick_mode("sphere_center")
        )
        self.lbl_sphere = Label(value="Sphere: not fitted")

        # ── ROI rectangle selection ──
        btn_draw_roi = PushButton(text="▣ Draw ROI Rectangle")
        btn_draw_roi.changed.connect(self._activate_roi_drawing)
        self.lbl_roi = Label(value="ROI: not set")

        self.roi_show_only = CheckBox(value=False, text="Show ROI spots only")
        self.roi_show_only.changed.connect(self._on_roi_toggle)

        btn_clear_roi = PushButton(text="Clear ROI")
        btn_clear_roi.changed.connect(self._clear_roi)

        # ── Ingression analysis ──
        btn_compute_ingression = PushButton(text="\u2b07 Compute Radial Velocity")
        btn_compute_ingression.changed.connect(self._compute_radial_velocity)
        self.ingression_threshold = FloatSlider(
            value=-0.1,
            min=-2.0,
            max=0.5,
            step=0.01,
            label="V_r threshold",
        )
        self.ingression_inward_frac = FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            label="Min inward frac",
        )
        self.ingression_min_frames = FloatSlider(
            value=5,
            min=2,
            max=30,
            step=1,
            label="Min track frames",
        )
        btn_auto_threshold = PushButton(text="\U0001f3af Auto-Detect Threshold")
        btn_auto_threshold.changed.connect(self._auto_detect_threshold)
        btn_flag_ingression = PushButton(text="Flag Ingressing Cells")
        btn_flag_ingression.changed.connect(self._flag_ingression)
        self.lbl_ingression = Label(value="Ingression: not computed")

        self.ingression_tracks_only = CheckBox(
            value=False, text="Show ingressing tracks only"
        )
        self.ingression_tracks_only.changed.connect(self._on_ingression_tracks_toggle)

        self.ingression_roi_only = CheckBox(value=False, text="Flag only within ROI")

        # ── Colour mode ──
        btn_color_frame = PushButton(text="Colour by Frame")
        btn_color_frame.changed.connect(lambda: self._recolor("frame"))
        btn_color_depth = PushButton(text="Colour by Depth")
        btn_color_depth.changed.connect(lambda: self._recolor("depth"))
        btn_color_depth_travel = PushButton(text="Colour by Depth Travel")
        btn_color_depth_travel.changed.connect(lambda: self._recolor("depth_travel"))
        btn_color_theta = PushButton(text="Colour by Latitude")
        btn_color_theta.changed.connect(lambda: self._recolor("theta"))
        btn_color_phi = PushButton(text="Colour by Longitude")
        btn_color_phi.changed.connect(lambda: self._recolor("phi"))
        btn_color_speed = PushButton(text="Colour by Speed")
        btn_color_speed.changed.connect(lambda: self._recolor("speed"))
        btn_color_radvel = PushButton(text="Colour by Radial Velocity")
        btn_color_radvel.changed.connect(lambda: self._recolor("radial_vel"))
        btn_color_ingress = PushButton(text="Colour Ingressing")
        btn_color_ingress.changed.connect(lambda: self._recolor("ingressing"))
        btn_color_roi = PushButton(text="Colour ROI")
        btn_color_roi.changed.connect(lambda: self._recolor("roi"))

        # ── Export & Video ──
        btn_export = PushButton(text="Export Enriched CSV")
        btn_export.changed.connect(self._export)
        self.lbl_export = Label(value="")
        self.render_scale_slider = FloatSlider(
            value=2.0, min=1.0, max=4.0, step=0.5,
            label="Render scale (×)",
        )
        self.duration_slider = FloatSlider(
            value=0, min=0, max=120, step=1,
            label="Duration (s, 0=auto)",
        )
        btn_record_video = PushButton(text="🎬 Record Time-lapse Video")
        btn_record_video.changed.connect(self._record_video)

        # ── Cross-section ──
        btn_cross_section = PushButton(text="✂ Cross-Section View")
        btn_cross_section.changed.connect(self._open_cross_section)

        container = Container(
            widgets=[
                Label(value="── Data ──"),
                btn_load_spots,
                self.lbl_spots_status,
                btn_load_tracks,
                self.lbl_tracks_status,
                btn_load_segments,
                self.lbl_segments_status,
                self.segments_visible_cb,
                self.segments_opacity_slider,
                btn_load_processed,
                self.lbl_processed_status,
                self.processed_visible_cb,
                self.processed_opacity_slider,
                Label(value="━━ TRACK FRAME RANGE ━━"),
                self.track_frame_start,
                self.track_frame_end,
                Label(value="── Display ──"),
                self.display_pct_slider,
                self.max_tracks_slider,
                self.track_width_slider,
                Label(value="━━ TRACK SELECTION ━━"),
                btn_select_track,
                btn_clear_selection,
                self.lbl_selection,
                self.follow_selection_cb,
                btn_goto_selection,
                self.random_sample_n,
                btn_random_sample,
                btn_random_sample_reset,
                self.filter_segments_cb,
                self.jump_track_combo,
                Label(value="── Landmarks ──"),
                btn_ap,
                self.lbl_ap,
                btn_dorsal,
                self.lbl_dorsal,
                btn_margin,
                self.lbl_margin,
                btn_ingression,
                self.lbl_ingression,
                Label(value="── Orientation ──"),
                btn_orient,
                Label(value="── Sphere Fit ──"),
                btn_sphere,
                btn_pick_sphere_center,
                self.lbl_sphere,
                Label(value="── ROI Selection ──"),
                btn_draw_roi,
                self.lbl_roi,
                self.roi_show_only,
                btn_clear_roi,
                Label(value="━━ ⬇ INGRESSION ━━"),
                btn_compute_ingression,
                self.ingression_threshold,
                self.ingression_inward_frac,
                self.ingression_min_frames,
                btn_auto_threshold,
                self.ingression_roi_only,
                btn_flag_ingression,
                self.lbl_ingression,
                self.ingression_tracks_only,
                Label(value="── Colouring ──"),
                btn_color_frame,
                btn_color_depth,
                btn_color_depth_travel,
                btn_color_theta,
                btn_color_phi,
                btn_color_speed,
                btn_color_radvel,
                btn_color_ingress,
                btn_color_roi,
                Label(value="── Export & Video ──"),
                btn_export,
                self.render_scale_slider,
                self.duration_slider,
                btn_record_video,
                btn_cross_section,
                self.lbl_export,
            ]
        )

        # Wrap in a QScrollArea so the panel is scrollable on small screens
        from qtpy.QtWidgets import QScrollArea, QWidget, QVBoxLayout
        from qtpy.QtCore import Qt

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        inner = container.native  # the underlying QWidget from magicgui
        scroll.setWidget(inner)

        self.viewer.window.add_dock_widget(scroll, name="Controls", area="right")

        # Metrics plot dock (selected tracks)
        self._build_metrics_dock()

    def _set_pick_mode(self, mode: str):
        if self._pick_mode == mode:
            self._pick_mode = "none"
            self.viewer.status = "Pick mode OFF"
        else:
            self._pick_mode = mode
            labels = {
                "ap": "ANIMAL POLE",
                "dorsal": "DORSAL",
                "sphere_center": "SPHERE CENTER",
                "margin": "MARGIN (click a cell at the margin boundary)",
                "ingression": "INGRESSION CENTER (click where ingression is strongest)",
            }
            label = labels.get(mode, mode.upper())
            self.viewer.status = (
                f"\U0001f3af CLICK near the {label} \u2014 "
                f"snaps to nearest spot in current frame. "
                f"Click button again to cancel."
            )

    def _on_click(self, viewer, event):
        """Handle mouse click for landmark picking and track selection."""
        if self._pick_mode == "none":
            return
        if self.spots_layer is None or self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        # Get click world coords — napari gives (t, z, y, x) for 4D
        world = np.array(event.position)
        if len(world) < 4:
            return
        click_x, click_y, click_z = world[3], world[2], world[1]

        # In 3D view, event.position is a point on the camera ray, not the
        # spot the user intended.  Use point-to-ray distance (the minimum
        # distance from each spot to the camera ray) so that tilting the
        # view doesn't break selection accuracy.
        view_dir = getattr(event, "view_direction", None)
        use_ray = view_dir is not None and self.viewer.dims.ndisplay == 3

        # Find nearest spot in the current frame
        current_frame = int(round(self.viewer.dims.current_step[0]))
        frame_mask = self.spots["FRAME"].values == current_frame
        if not frame_mask.any():
            frames = self.spots["FRAME"].unique()
            current_frame = int(frames[np.argmin(np.abs(frames - current_frame))])
            frame_mask = self.spots["FRAME"].values == current_frame

        sx = self.spots.loc[frame_mask, "POSITION_X"].values
        sy = self.spots.loc[frame_mask, "POSITION_Y"].values
        sz = self.spots.loc[frame_mask, "POSITION_Z"].values

        if use_ray:
            # view_direction is (t, z, y, x) — extract spatial components
            vd = np.array(view_dir)
            d = np.array([vd[3], vd[2], vd[1]])  # x, y, z
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-10:
                d = d / d_norm
            # Point-to-ray distance: ||cross(spot - origin, d)||
            origin = np.array([click_x, click_y, click_z])
            diff = np.column_stack([sx - origin[0],
                                    sy - origin[1],
                                    sz - origin[2]])
            cross = np.column_stack([
                diff[:, 1] * d[2] - diff[:, 2] * d[1],
                diff[:, 2] * d[0] - diff[:, 0] * d[2],
                diff[:, 0] * d[1] - diff[:, 1] * d[0],
            ])
            dists = np.sqrt((cross ** 2).sum(axis=1))
        else:
            dists = np.sqrt((sx - click_x) ** 2 + (sy - click_y) ** 2 + (sz - click_z) ** 2)
        nearest = int(dists.argmin())
        pos_xyz = np.array([sx[nearest], sy[nearest], sz[nearest]])
        snap_d = dists[nearest]

        # ── Track selection mode ──
        if self._pick_mode == "select_track":
            # Look up the track_id directly — the nearest spot is already found
            df_frame = self.spots[frame_mask]
            tid_vals = df_frame["TRACK_ID"].values if "TRACK_ID" in df_frame.columns else None
            if tid_vals is not None:
                tid = int(tid_vals[nearest])
                self._handle_track_selection_click(tid)
            else:
                tid = None
            # Exit pick mode after each selection so user can navigate
            self._pick_mode = "none"
            if tid is not None:
                self.viewer.status = (
                    f"Track {tid} toggled — click \U0001f50d again to select more"
                )
            return

        if self._pick_mode == "ap":
            self.animal_pole = pos_xyz
            self.lbl_ap.value = (
                f"AP: ({pos_xyz[0]:.1f}, {pos_xyz[1]:.1f}, {pos_xyz[2]:.1f})"
            )
            self._pick_mode = "none"
            self.viewer.status = (
                f"Animal Pole set (snapped {snap_d:.1f}\u00b5m to nearest spot)"
            )
        elif self._pick_mode == "dorsal":
            self.dorsal_mark = pos_xyz
            self.lbl_dorsal.value = (
                f"Dorsal: ({pos_xyz[0]:.1f}, {pos_xyz[1]:.1f}, {pos_xyz[2]:.1f})"
            )
            self._pick_mode = "none"
            self.viewer.status = (
                f"Dorsal set (snapped {snap_d:.1f}\u00b5m to nearest spot)"
            )
        elif self._pick_mode == "sphere_center":
            self._pick_mode = "none"
            self._apply_manual_sphere_center(pos_xyz)
        elif self._pick_mode == "margin":
            self.margin_landmark = pos_xyz
            # Compute θ if sphere is fitted
            theta_str = ""
            if self.sphere_fit_result is not None:
                c = self.sphere_fit_result["center"]
                d = pos_xyz - c
                r = np.linalg.norm(d)
                theta = np.degrees(np.arccos(np.clip(-d[1] / max(r, 1e-10), -1, 1)))
                theta_str = f" | θ={theta:.1f}°"
            self.lbl_margin.value = (
                f"Margin: Y={pos_xyz[1]:.1f}{theta_str}"
            )
            self._pick_mode = "none"
            self.viewer.status = (
                f"Margin set at Y={pos_xyz[1]:.1f}{theta_str} "
                f"(snapped {snap_d:.1f}µm to nearest spot)"
            )
        elif self._pick_mode == "ingression":
            self.ingression_landmark = pos_xyz
            theta_str, phi_str = "", ""
            if self.sphere_fit_result is not None:
                c = self.sphere_fit_result["center"]
                d = pos_xyz - c
                r = np.linalg.norm(d)
                theta = np.degrees(np.arccos(np.clip(-d[1] / max(r, 1e-10), -1, 1)))
                phi = np.degrees(np.arctan2(d[2], d[0]))
                theta_str = f" | θ={theta:.1f}°"
                phi_str = f", φ={phi:.1f}°"
            self.lbl_ingression.value = (
                f"Ingression: ({pos_xyz[0]:.1f}, {pos_xyz[1]:.1f}, {pos_xyz[2]:.1f})"
                f"{theta_str}{phi_str}"
            )
            self._pick_mode = "none"
            self.viewer.status = (
                f"Ingression center set{theta_str}{phi_str} "
                f"(snapped {snap_d:.1f}µm to nearest spot)"
            )
        self._update_landmarks()

    def _update_landmarks(self):
        """Update landmarks (3D layer \u2014 visible at every frame)."""
        pts = []
        colors = []
        if self.animal_pole is not None:
            # 3D: (z, y, x) \u2014 visible at all time steps
            pts.append([self.animal_pole[2], self.animal_pole[1], self.animal_pole[0]])
            colors.append("red")
        if self.dorsal_mark is not None:
            pts.append([self.dorsal_mark[2], self.dorsal_mark[1], self.dorsal_mark[0]])
            colors.append("cyan")
        if self.margin_landmark is not None:
            pts.append([self.margin_landmark[2], self.margin_landmark[1], self.margin_landmark[0]])
            colors.append("yellow")
        if self.ingression_landmark is not None:
            pts.append([self.ingression_landmark[2], self.ingression_landmark[1], self.ingression_landmark[0]])
            colors.append("magenta")

        if pts:
            arr = np.array(pts)
            # Remove and re-add to avoid multiple per-property refreshes
            try:
                self.viewer.layers.remove(self.landmarks_layer)
            except Exception:
                pass
            self.landmarks_layer = self.viewer.add_points(
                arr, name="Landmarks", size=20,
                face_color=colors, symbol="diamond",
                ndim=3, border_width=0,
            )
        else:
            self.landmarks_layer.data = np.empty((0, 3))

    # ── Track Selection ───────────────────────────────────────────────

    def _build_track_id_index(self):
        """Pre-build a dict mapping track_id → array of row indices.

        With 2M spots and ~140K tracks this takes ~200 ms and makes
        every subsequent selection operation O(selected_tracks) instead
        of O(all_spots).
        """
        if self.spots is None or "TRACK_ID" not in self.spots.columns:
            self._track_id_index = None
            return
        tids = self.spots["TRACK_ID"].values
        order = np.argsort(tids, kind="stable")
        sorted_tids = tids[order]
        # Find boundaries between different track IDs
        breaks = np.flatnonzero(np.diff(sorted_tids) != 0) + 1
        groups = np.split(order, breaks)
        unique_tids = sorted_tids[np.r_[0, breaks]]
        self._track_id_index = {
            int(uid): grp for uid, grp in zip(unique_tids, groups)
            if not np.isnan(uid)
        }

    def _toggle_select_track_mode(self):
        """Toggle click-to-select-track pick mode."""
        if self._pick_mode == "select_track":
            self._pick_mode = "none"
            self.viewer.status = "Track selection pick mode OFF"
        else:
            self._pick_mode = "select_track"
            self.viewer.status = (
                "\U0001f3af CLICK on a nucleus to select/deselect its track. "
                "Click the button again to stop selecting."
            )

    def _clear_selection(self):
        """Clear all selected tracks and restore full display."""
        self._selected_track_ids.clear()
        self._random_sampled_ids.clear()
        self._selection_active = False
        self._selection_lut = None
        self._pick_mode = "none"
        self._update_selection_label()
        self._apply_selection_display()
        self.viewer.status = "Selection cleared — showing all nuclei"
        for xs in self._cross_section_viewers:
            xs.on_parent_selection_changed()

    def _reset_random_sample_pool(self):
        """Clear the 'already sampled' pool so IDs may be drawn again."""
        self._random_sampled_ids.clear()
        self.viewer.status = "Random-sample pool reset (all tracks eligible)"

    def _random_sample_tracks(self):
        """Add N random track IDs to the selection, never repeating."""
        if self.spots is None or self._track_id_index is None:
            self.viewer.status = "Load tracks first"
            return
        n_req = int(self.random_sample_n.value)
        all_ids = np.fromiter(self._track_id_index.keys(),
                              dtype=np.int64,
                              count=len(self._track_id_index))
        # Eligible = not previously drawn
        if self._random_sampled_ids:
            already = np.fromiter(self._random_sampled_ids,
                                  dtype=np.int64,
                                  count=len(self._random_sampled_ids))
            eligible = np.setdiff1d(all_ids, already, assume_unique=True)
        else:
            eligible = all_ids
        if eligible.size == 0:
            self.viewer.status = (
                "All tracks have been sampled — press 'Reset random pool' "
                "to start over")
            # Always notify cross-section viewers so they re-render their
            # overlay once the pool is reset from the other window.
            for xs in self._cross_section_viewers:
                xs.on_parent_selection_changed()
            return
        n_draw = min(n_req, eligible.size)
        rng = np.random.default_rng()
        picked = rng.choice(eligible, size=n_draw, replace=False)
        if picked.size:
            self._random_sampled_ids.update(int(t) for t in picked)
            self._selected_track_ids.update(int(t) for t in picked)
            self._selection_active = True
        # Update label + status BEFORE applying the display so the user
        # sees the sampling message even if the rebuild raises.
        self._update_selection_label()
        self.viewer.status = (
            f"Sampled {n_draw} random track(s) — "
            f"{len(self._selected_track_ids)} currently selected, "
            f"{eligible.size - n_draw} still in pool")
        self._apply_selection_display()
        # Mirror the new selection onto every open cross-section viewer.
        for xs in self._cross_section_viewers:
            xs.on_parent_selection_changed()

    def _update_selection_label(self):
        """Update the selection info label."""
        lbl = getattr(self, "lbl_selection", None)
        if lbl is None:
            return
        n = len(self._selected_track_ids)
        if n == 0:
            lbl.value = "No tracks selected"
        elif n <= 5:
            ids = sorted(self._selected_track_ids)
            lbl.value = f"Selected: {', '.join(str(i) for i in ids)}"
        else:
            lbl.value = f"Selected: {n} tracks"
        self._refresh_jump_choices()

    def _handle_track_selection_click(self, tid: int):
        """Toggle track *tid* in/out of the selection and update display."""
        try:
            self._handle_track_selection_click_impl(tid)
        except Exception:
            traceback.print_exc()
            self.viewer.status = "ERROR in track selection — see terminal"

    def _handle_track_selection_click_impl(self, tid: int):
        if self.spots is None:
            return

        # Toggle: add or remove
        if tid in self._selected_track_ids:
            self._selected_track_ids.discard(tid)
            action = "deselected"
        else:
            self._selected_track_ids.add(tid)
            action = "selected"

        self._selection_active = len(self._selected_track_ids) > 0
        self._update_selection_label()
        self._apply_selection_display()

        self.viewer.status = (
            f"Track {tid} {action} — "
            f"{len(self._selected_track_ids)} track(s) selected"
        )
        for xs in self._cross_section_viewers:
            xs.on_parent_selection_changed()

    def _apply_selection_display(self):
        """Rebuild all display layers to show only selected tracks (or all)."""
        try:
            self._apply_selection_display_impl()
        except Exception:
            traceback.print_exc()
            self.viewer.status = "ERROR in selection display — see terminal"

    def _apply_selection_display_impl(self):
        """Apply track selection using persistent overlay layers.

        Main "Nuclei" and "Tracks" layers stay intact.  When a selection
        is active we hide them and show pre-created overlay layers whose
        `.data` we update in-place (no layer creation → instant even
        for hundreds of tracks).
        """
        df = self.spots
        sel_ids = self._selected_track_ids
        idx_map = getattr(self, "_track_id_index", None)

        # ── No selection: show main layers, hide overlays ────────────
        if not self._selection_active or not sel_ids or df is None:
            if self._selection_spots_overlay is not None:
                self._selection_spots_overlay.visible = False
            if self._selection_tracks_overlay is not None:
                self._selection_tracks_overlay.visible = False
            if self.spots_layer is not None:
                self.spots_layer.visible = True
            if self.tracks_layer is not None:
                self.tracks_layer.visible = True
            if getattr(self, "filter_segments_cb", None) is not None \
                    and self.filter_segments_cb.value:
                self._rebuild_selection_segments()
            self._update_metrics_plot()
            return

        # ── Gather row indices for selected tracks (O(|selected|)) ───
        if idx_map is not None:
            parts = [idx_map[t] for t in sel_ids if t in idx_map]
            sel_idx = (np.concatenate(parts)
                       if parts else np.empty(0, dtype=np.intp))
        else:
            mask = df["TRACK_ID"].isin(sel_ids).values
            sel_idx = np.flatnonzero(mask)

        if len(sel_idx) == 0:
            self.viewer.status = "No spots match current selection"
            if getattr(self, "filter_segments_cb", None) is not None \
                    and self.filter_segments_cb.value:
                self._rebuild_selection_segments()
            self._update_metrics_plot()
            return

        tids = df["TRACK_ID"].values[sel_idx].astype(np.int64)
        frames = df["FRAME"].values[sel_idx]

        # Sort by (track_id, frame) — required by napari Tracks layer
        order = np.lexsort((frames, tids))
        sel_idx = sel_idx[order]
        tids = tids[order]
        frames = frames[order]
        zz = df["POSITION_Z"].values[sel_idx]
        yy = df["POSITION_Y"].values[sel_idx]
        xx = df["POSITION_X"].values[sel_idx]

        spots_data = np.column_stack([frames, zz, yy, xx])

        # Fast filter to keep only tracks with ≥2 spots (np.unique + bincount)
        uniq, inverse = np.unique(tids, return_inverse=True)
        counts = np.bincount(inverse)
        multi = counts >= 2
        track_mask = multi[inverse]
        tracks_data = np.column_stack(
            [tids[track_mask], frames[track_mask], zz[track_mask],
             yy[track_mask], xx[track_mask]]
        )

        # Hide main layers
        if self.spots_layer is not None:
            self.spots_layer.visible = False
        if self.tracks_layer is not None:
            self.tracks_layer.visible = False

        # ── Update persistent overlay layers in-place ────────────────
        self._ensure_selection_overlays()
        sp_over = self._selection_spots_overlay
        tr_over = self._selection_tracks_overlay

        if sp_over is not None:
            sp_over.data = spots_data
            sp_over.visible = True

        if tr_over is not None:
            if len(tracks_data) > 0:
                tr_over.data = tracks_data
                tr_over.visible = True
            else:
                tr_over.visible = False

        # Segments + plot + camera
        if getattr(self, "filter_segments_cb", None) is not None \
                and self.filter_segments_cb.value:
            self._rebuild_selection_segments()
        self._update_metrics_plot()
        if self._follow_selection:
            self._center_on_selection()

    def _on_filter_segments_changed(self, value: bool):
        """Toggle segment-volume filtering on selection."""
        if value:
            self._rebuild_selection_segments()
        else:
            # Restore unfiltered segments
            seg_layer = self._segments_layer
            seg_data = self._segments_data
            if seg_layer is not None and seg_data is not None \
                    and seg_layer.data is not seg_data:
                seg_layer.data = seg_data
                try:
                    seg_layer.refresh()
                except Exception:
                    pass
            self._selection_lut = None

    def _ensure_selection_overlays(self):
        """Create the persistent Selection overlay layers if missing."""
        if self._selection_spots_overlay is None:
            dummy = np.zeros((1, 4), dtype=float)
            self._selection_spots_overlay = self.viewer.add_points(
                dummy,
                name="Selection: Nuclei",
                size=3,
                face_color="#FFD700",
                border_color="#FF4500",
                border_width=0.2,
                ndim=4,
                out_of_slice_display=False,
                visible=False,
            )
        if self._selection_tracks_overlay is None:
            # napari Tracks needs ≥2 rows with same track_id
            dummy = np.array([[0, 0, 0.0, 0.0, 0.0],
                              [0, 1, 0.0, 0.0, 0.0]], dtype=float)
            self._selection_tracks_overlay = self.viewer.add_tracks(
                dummy,
                name="Selection: Tracks",
                tail_width=max(self._track_width, 3.0),
                tail_length=40,
                color_by="track_id",
                colormap="turbo",
                visible=False,
            )

    def _rebuild_selection_segments(self):
        """Filter segments by label colormap (fast, no data rewrite)."""
        # Colormap approach works for both preload and lazy mode.
        self._rebuild_selection_segments_lazy()

    def _rebuild_selection_segments_lazy(self):
        """Filter segments by swapping the Labels *colormap* only.

        Much faster than rewriting `.data`: the volume stays intact and
        napari only re-renders with a new label-to-RGBA map.  Non-selected
        labels become transparent; selected labels keep their original
        per-track colour from the cyclic colormap.
        """
        seg_layer = self._segments_layer
        if seg_layer is None:
            return

        # Remember the original (unfiltered) cyclic colormap once
        if getattr(self, "_segments_original_colormap", None) is None:
            self._segments_original_colormap = seg_layer.colormap

        # ── Restore ───────────────────────────────────────────────────
        if not self._selection_active or not self._selected_track_ids:
            try:
                seg_layer.colormap = self._segments_original_colormap
            except Exception:
                pass
            return

        # ── Filter via DirectLabelColormap ────────────────────────────
        try:
            from napari.utils.colormaps import DirectLabelColormap
        except ImportError:
            return

        orig_cmap = self._segments_original_colormap
        # Pull the per-label colour from the original cyclic map so
        # selected nuclei keep their familiar hue.
        sel = np.fromiter(self._selected_track_ids, dtype=np.int64,
                          count=len(self._selected_track_ids))
        try:
            colors = orig_cmap.map(sel)  # (N, 4) rgba 0–1
        except Exception:
            # Fallback: uniform gold
            colors = np.tile(np.array([1.0, 0.84, 0.0, 1.0]),
                             (len(sel), 1))

        color_dict: dict = {int(t): tuple(c) for t, c in zip(sel, colors)}
        # Default (None ↦ transparent) — hides all non-selected labels
        color_dict[None] = (0.0, 0.0, 0.0, 0.0)

        try:
            seg_layer.colormap = DirectLabelColormap(color_dict=color_dict)
        except Exception:
            traceback.print_exc()

    def _swap_segment_frame(self, t: int):
        """Swap the displayed segment frame, respecting selection state."""
        sel_lut = self._selection_lut
        if self._selection_active and sel_lut is not None:
            # On-the-fly LUT application for the single current frame
            dl = getattr(self, '_display_labels', None)
            if dl is None or t < 0 or t >= dl.shape[0]:
                return
            frame = sel_lut[dl[t]]
        else:
            frames = getattr(self, '_segment_frames', None)
            if frames is None or t < 0 or t >= len(frames):
                return
            frame = frames[t]

        vol = getattr(self, '_vispy_volume', None)
        canv = getattr(self, '_vispy_canvas', None)
        if vol is not None:
            vol.set_data(frame)
            if canv is not None:
                canv.native.update()
        elif self._segments_layer is not None:
            self._segments_layer.data = frame

        # Sync layer._data reference
        seg_lyr = self._segments_layer
        if seg_lyr is not None:
            try:
                seg_lyr._data = frame
            except Exception:
                pass

    def _center_on_selection(self):
        """Center the 3D camera on the mean position of selected nuclei."""
        if not self._selected_track_ids or self.spots is None:
            return

        current_frame = int(self.viewer.dims.current_step[0])
        mask = (
            self.spots["TRACK_ID"].isin(self._selected_track_ids)
            & (self.spots["FRAME"] == current_frame)
        )
        if not mask.any():
            # Fall back to nearest frame with data
            sel_spots = self.spots[
                self.spots["TRACK_ID"].isin(self._selected_track_ids)
            ]
            if sel_spots.empty:
                return
            nearest_frame = int(sel_spots["FRAME"].iloc[
                (sel_spots["FRAME"] - current_frame).abs().argmin()
            ])
            mask = (
                self.spots["TRACK_ID"].isin(self._selected_track_ids)
                & (self.spots["FRAME"] == nearest_frame)
            )

        sub = self.spots[mask]
        cx = sub["POSITION_X"].mean()
        cy = sub["POSITION_Y"].mean()
        cz = sub["POSITION_Z"].mean()

        # Set napari 3D camera center (z, y, x order)
        cam = self.viewer.camera
        cam.center = (current_frame, cz, cy, cx)

    def _go_to_selection(self):
        """One-shot: center camera on selected nuclei."""
        if not self._selected_track_ids:
            self.viewer.status = "No tracks selected"
            return
        self._center_on_selection()
        self.viewer.status = f"Camera centered on {len(self._selected_track_ids)} selected track(s)"

    def _refresh_jump_choices(self):
        """Rebuild the Jump-to-track dropdown from the current selection."""
        combo = getattr(self, "jump_track_combo", None)
        if combo is None:
            return
        ids = sorted(self._selected_track_ids)
        if not ids:
            choices = [("— none —", -1)]
        else:
            choices = [("— pick a track —", -1)] + [
                (f"Track {t}", int(t)) for t in ids
            ]
        try:
            combo.changed.disconnect(self._on_jump_track_changed)
        except Exception:
            pass
        combo.choices = choices
        combo.value = -1
        combo.changed.connect(self._on_jump_track_changed)

    def _on_jump_track_changed(self, value):
        """Jump to the first frame of the chosen track and centre on it."""
        try:
            tid = int(value)
        except (TypeError, ValueError):
            return
        if tid < 0 or self.spots is None:
            return
        idx_map = getattr(self, "_track_id_index", None)
        if idx_map is None or tid not in idx_map:
            self.viewer.status = f"Track {tid} not in index"
            return
        rows = idx_map[tid]
        frames = self.spots["FRAME"].values[rows].astype(np.int64)
        xs = self.spots["POSITION_X"].values[rows].astype(float)
        ys = self.spots["POSITION_Y"].values[rows].astype(float)
        zs = self.spots["POSITION_Z"].values[rows].astype(float)
        k = int(np.argmin(frames))
        t0, x0, y0, z0 = int(frames[k]), float(xs[k]), float(ys[k]), float(zs[k])

        # Jump in time
        try:
            step = list(self.viewer.dims.current_step)
            step[0] = t0
            self.viewer.dims.current_step = tuple(step)
        except Exception:
            pass

        # Centre camera on the first spot (z, y, x)
        try:
            self.viewer.camera.center = (t0, z0, y0, x0)
        except Exception:
            # Some napari versions expect 3-vector
            try:
                self.viewer.camera.center = (z0, y0, x0)
            except Exception:
                pass

        self.viewer.status = (
            f"Jumped to track {tid} — first frame t={t0} "
            f"at ({x0:.1f}, {y0:.1f}, {z0:.1f})"
        )

    def _on_follow_selection_changed(self, value: bool):
        """Toggle follow-selection camera mode."""
        self._follow_selection = value
        if value and self._selection_active:
            self._center_on_selection()

    def _register_cross_section(self, viewer: "CrossSectionViewer") -> None:
        """Register a ``CrossSectionViewer`` to receive selection updates.

        Called by ``CrossSectionViewer.__init__``.  Seeds the new viewer's
        state immediately so its first rebuild is consistent with the
        parent.
        """
        if viewer not in self._cross_section_viewers:
            self._cross_section_viewers.append(viewer)
        try:
            viewer.on_parent_selection_changed()
        except Exception:
            pass

    # ── Selection metrics plot ───────────────────────────────────────

    def _build_metrics_dock(self):
        """Create a matplotlib dock widget showing metrics for selected tracks.

        Rationale (mirrors ``gastrulation_dynamics_comparison.R``):
          • Instantaneous speed is computed over a *velocity lag*
            (``MEDAKA_VEL_LAG = 4`` frames = 2 min, ``ZEBRAFISH_VEL_LAG = 1``
            frame = 2 min) rather than consecutive frames, to smooth
            nuclear-centroid segmentation jitter.
          • A rolling-mean of window ``smooth_k`` is applied afterwards
            (5 fr = 2.5 min for medaka, 3 fr = 6 min for zebrafish).
          • Speeds are reported in µm/min using the voxel calibration
            (isotropic mean of ``voxel_size``) and the frame interval.
        """
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from qtpy.QtWidgets import (
                QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
                QDoubleSpinBox, QPushButton,
            )
        except ImportError:
            self._metrics_fig = None
            self._metrics_canvas = None
            return

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Row 1: frame interval / lag / smooth-K
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("FI(s)"))
        self._sp_fi = QDoubleSpinBox()
        self._sp_fi.setRange(0.01, 3600.0)
        self._sp_fi.setDecimals(2)
        self._sp_fi.setValue(self._metrics_fi_sec)
        self._sp_fi.valueChanged.connect(self._on_metrics_params_changed)
        row1.addWidget(self._sp_fi)
        row1.addWidget(QLabel("lag"))
        self._sp_lag = QSpinBox()
        self._sp_lag.setRange(1, 50)
        self._sp_lag.setValue(self._metrics_lag)
        self._sp_lag.valueChanged.connect(self._on_metrics_params_changed)
        row1.addWidget(self._sp_lag)
        row1.addWidget(QLabel("smooth K"))
        self._sp_smooth = QSpinBox()
        self._sp_smooth.setRange(1, 50)
        self._sp_smooth.setValue(self._metrics_smooth_k)
        self._sp_smooth.valueChanged.connect(self._on_metrics_params_changed)
        row1.addWidget(self._sp_smooth)
        layout.addLayout(row1)

        # Row 2: species presets
        row2 = QHBoxLayout()
        btn_medaka = QPushButton("Medaka preset")
        btn_medaka.clicked.connect(
            lambda: self._apply_metrics_preset(fi=30.0, lag=4, k=5))
        row2.addWidget(btn_medaka)
        btn_zebra = QPushButton("Zebrafish preset")
        btn_zebra.clicked.connect(
            lambda: self._apply_metrics_preset(fi=120.0, lag=1, k=3))
        row2.addWidget(btn_zebra)
        layout.addLayout(row2)

        # Matplotlib figure
        fig = Figure(figsize=(6, 6), tight_layout=True, facecolor="#262930")
        fig.patch.set_alpha(0)
        self._metrics_fig = fig
        self._metrics_axes = fig.subplots(2, 2)
        for ax in self._metrics_axes.flat:
            ax.set_facecolor("#1a1b20")
            for spine in ax.spines.values():
                spine.set_color("#888")
            ax.tick_params(colors="#ccc", labelsize=7)
            ax.title.set_color("#fff")

        self._metrics_canvas = FigureCanvasQTAgg(fig)
        self._metrics_canvas.setMinimumWidth(380)
        self._metrics_canvas.setMinimumHeight(420)
        layout.addWidget(self._metrics_canvas)

        self.viewer.window.add_dock_widget(
            panel, name="Selection metrics", area="left"
        )
        self._update_metrics_plot()

    def _apply_metrics_preset(self, fi: float, lag: int, k: int):
        self._sp_fi.setValue(fi)
        self._sp_lag.setValue(lag)
        self._sp_smooth.setValue(k)
        # valueChanged handlers will trigger the recompute.

    def _on_metrics_params_changed(self, *_):
        self._metrics_fi_sec = float(self._sp_fi.value())
        self._metrics_lag = int(self._sp_lag.value())
        self._metrics_smooth_k = int(self._sp_smooth.value())
        self._update_metrics_plot()

    @staticmethod
    def _rolling_mean(arr: np.ndarray, k: int) -> np.ndarray:
        """Centered rolling mean with window ``k`` (NaN-tolerant)."""
        if k <= 1 or arr.size < 2:
            return arr.astype(float, copy=False)
        k = int(k)
        # Use a convolution on the masked array, then normalise by the
        # number of finite contributions so NaNs don't poison the window.
        a = np.where(np.isnan(arr), 0.0, arr)
        w = np.where(np.isnan(arr), 0.0, 1.0)
        kernel = np.ones(k, dtype=float)
        num = np.convolve(a, kernel, mode="same")
        den = np.convolve(w, kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        return out

    def _compute_selection_metrics(self) -> dict | None:
        """Compute metrics for selected tracks (R-script rationale).

        Returns ``None`` if no valid selection, otherwise a dict:
          - ``inst_speed_steps``   µm/min per lag-step (smoothed)
          - ``mean_speed_per_track`` µm/min, mean of the smoothed series
          - ``track_length``       minutes per track
          - ``straightness``       net/path
        """
        if (self.spots is None
                or not self._selected_track_ids
                or "TRACK_ID" not in self.spots.columns):
            return None

        idx_map = getattr(self, "_track_id_index", None)
        if idx_map is None:
            return None
        parts = [idx_map[t] for t in self._selected_track_ids if t in idx_map]
        if not parts:
            return None
        sel_idx = np.concatenate(parts)
        if len(sel_idx) < 2:
            return None

        df = self.spots
        tids = df["TRACK_ID"].values[sel_idx].astype(np.int64)
        frames = df["FRAME"].values[sel_idx].astype(np.int64)

        # Voxel calibration → µm.  Isotropic mean of (Z,Y,X) voxel size.
        vs = self._voxel_size
        scale = float(np.mean(vs)) if vs is not None else 1.0
        x = df["POSITION_X"].values[sel_idx].astype(float) * scale
        y = df["POSITION_Y"].values[sel_idx].astype(float) * scale
        z = df["POSITION_Z"].values[sel_idx].astype(float) * scale

        # Sort by (track_id, frame)
        order = np.lexsort((frames, tids))
        tids, frames = tids[order], frames[order]
        x, y, z = x[order], y[order], z[order]

        lag = max(1, int(self._metrics_lag))
        smooth_k = max(1, int(self._metrics_smooth_k))
        fi_min = float(self._metrics_fi_sec) / 60.0  # minutes per frame

        # Per-track aggregation
        unique_tids, first_idx = np.unique(tids, return_index=True)
        bounds = np.r_[first_idx, len(tids)]

        inst_all: list[np.ndarray] = []
        mean_speed = np.full(len(unique_tids), np.nan, dtype=float)
        track_len = np.zeros(len(unique_tids), dtype=np.int64)
        straight = np.full(len(unique_tids), np.nan, dtype=float)

        for k in range(len(unique_tids)):
            lo, hi = bounds[k], bounds[k + 1]
            n = hi - lo
            track_len[k] = n
            if n < 2:
                continue
            tx, ty, tz = x[lo:hi], y[lo:hi], z[lo:hi]
            tf = frames[lo:hi]

            # Straightness uses full consecutive path
            seg = np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2
                          + np.diff(tz) ** 2)
            path_len = seg.sum()
            net = np.sqrt((tx[-1] - tx[0]) ** 2
                          + (ty[-1] - ty[0]) ** 2
                          + (tz[-1] - tz[0]) ** 2)
            straight[k] = (net / path_len) if path_len > 0 else np.nan

            # Velocity at lag, same-track only, mask gaps
            if n <= lag:
                continue
            dx = tx[lag:] - tx[:-lag]
            dy = ty[lag:] - ty[:-lag]
            dz = tz[lag:] - tz[:-lag]
            df_fr = (tf[lag:] - tf[:-lag]).astype(float)
            disp = np.sqrt(dx * dx + dy * dy + dz * dz)
            # µm/min: only valid if frame gap == lag (no missing frames)
            valid = df_fr == lag
            spd = np.where(valid, disp / (lag * fi_min), np.nan)
            spd_s = self._rolling_mean(spd, smooth_k)
            finite = spd_s[np.isfinite(spd_s)]
            if finite.size:
                inst_all.append(finite)
                mean_speed[k] = float(np.mean(finite))

        inst_speed = (np.concatenate(inst_all)
                      if inst_all else np.empty(0, dtype=float))
        return {
            "inst_speed_steps": inst_speed,
            "mean_speed_per_track": mean_speed,
            "track_length_min": track_len.astype(float) * fi_min,
            "straightness": straight,
            "n_tracks": int(len(unique_tids)),
            "lag": lag,
            "smooth_k": smooth_k,
            "fi_sec": float(self._metrics_fi_sec),
            "scaled_um": vs is not None,
        }

    def _update_metrics_plot(self):
        """Redraw the selection-metrics figure."""
        fig = getattr(self, "_metrics_fig", None)
        canvas = getattr(self, "_metrics_canvas", None)
        if fig is None or canvas is None:
            return
        axes = self._metrics_axes
        for ax in axes.flat:
            ax.clear()
            ax.set_facecolor("#1a1b20")
            for spine in ax.spines.values():
                spine.set_color("#888")
            ax.tick_params(colors="#ccc", labelsize=7)
            ax.title.set_color("#fff")

        m = self._compute_selection_metrics()
        if m is None:
            axes[0, 0].text(
                0.5, 0.5, "Select tracks to\nshow metrics",
                ha="center", va="center",
                transform=axes[0, 0].transAxes,
                color="#888", fontsize=11,
            )
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            canvas.draw_idle()
            return

        color_hist = "#FFD700"
        color_edge = "#FF4500"
        unit = "µm/min" if m["scaled_um"] else "px/min"

        # (0,0) instantaneous speed (smoothed, per-step)
        ax = axes[0, 0]
        vals = m["inst_speed_steps"]
        if len(vals) > 1:
            ax.hist(vals, bins=min(40, max(5, int(np.sqrt(len(vals))))),
                    color=color_hist, edgecolor=color_edge)
            ax.set_title(
                f"Inst. speed\n(lag={m['lag']} fr, k={m['smooth_k']}, "
                f"n={len(vals)} steps)", fontsize=8)
            ax.set_xlabel(unit, fontsize=7)
        else:
            ax.text(0.5, 0.5, "tracks too short\nfor lag",
                    ha="center", va="center",
                    transform=ax.transAxes, color="#888")

        # (0,1) mean speed per track
        ax = axes[0, 1]
        vals = m["mean_speed_per_track"]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            bins = min(20, max(3, int(np.sqrt(len(vals)))))
            ax.hist(vals, bins=bins, color=color_hist, edgecolor=color_edge)
            ax.set_title(
                f"Mean speed / track\n(n={m['n_tracks']}, "
                f"med={np.median(vals):.2f} {unit})", fontsize=8)
            ax.set_xlabel(unit, fontsize=7)

        # (1,0) track length in minutes
        ax = axes[1, 0]
        vals = m["track_length_min"]
        if len(vals) > 0:
            bins = min(20, max(3, int(np.sqrt(len(vals)))))
            ax.hist(vals, bins=bins, color=color_hist, edgecolor=color_edge)
            ax.set_title(
                f"Track length\n(med={np.median(vals):.1f} min)", fontsize=8)
            ax.set_xlabel("minutes", fontsize=7)

        # (1,1) straightness
        ax = axes[1, 1]
        vals = m["straightness"]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            bins = min(20, max(3, int(np.sqrt(len(vals)))))
            ax.hist(vals, bins=bins, range=(0, 1),
                    color=color_hist, edgecolor=color_edge)
            ax.set_title(
                f"Straightness\n(med={np.median(vals):.2f})", fontsize=8)
            ax.set_xlabel("net / path", fontsize=7)
            ax.set_xlim(0, 1)

        canvas.draw_idle()

    # ── Orientation ───────────────────────────────────────────────────

    def _orient(self):
        """Apply AP → +Y, Dorsal → +X rotation."""
        if self.animal_pole is None or self.dorsal_mark is None:
            self.viewer.status = "Set both Animal Pole and Dorsal landmarks first!"
            return
        if self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        self.viewer.status = "Orienting embryo..."
        pos = self.spots[["POSITION_X", "POSITION_Y", "POSITION_Z"]].values
        pos_new, params = orient_embryo(pos, self.animal_pole, self.dorsal_mark)

        # Pivot used by orient_embryo (centroid of the spots).
        mid = params["center"]
        R1, R2 = params["R1"], params["R2"]
        R = R2 @ R1

        # If a sphere has been fitted, transform its centre with the same
        # rotation+pivot used by orient_embryo so the cap mesh and
        # spherical coordinates stay aligned with the rotated nuclei.
        if self.sphere_fit_result is not None:
            sphere_c_old = np.asarray(
                self.sphere_fit_result["center"], dtype=float
            )
            sphere_c_new = R @ (sphere_c_old - mid)
            self.sphere_fit_result = dict(self.sphere_fit_result)
            self.sphere_fit_result["center"] = sphere_c_new

        self.spots["POSITION_X"] = pos_new[:, 0]
        self.spots["POSITION_Y"] = pos_new[:, 1]
        self.spots["POSITION_Z"] = pos_new[:, 2]
        self.transform_params = params
        self.oriented = True

        # Transform landmark positions into the new coordinate system
        if self.animal_pole is not None:
            self.animal_pole = R @ (self.animal_pole - mid)
        if self.dorsal_mark is not None:
            self.dorsal_mark = R @ (self.dorsal_mark - mid)

        # Update layers
        self._refresh_points()
        self._refresh_tracks()
        self._update_landmarks()

        # Reorient nuclei 3D segments layer via GPU affine (no data copy)
        self._orient_segments(params)

        # Recompute spherical coords and rebuild the cap mesh so the
        # sphere overlay matches the rotated nuclei.
        if self.sphere_fit_result is not None:
            sph = compute_spherical_coords(
                self.spots[["POSITION_X", "POSITION_Y", "POSITION_Z"]].values,
                self.sphere_fit_result["center"],
                self.sphere_fit_result["radius"],
            )
            for key, vals in sph.items():
                self.spots[key] = vals
            self._add_sphere_overlay()

        # Reset camera to centre on the reoriented data.
        # NOTE: when the Nuclei-3D layer is loaded, its performance hack
        # monkey-patches `_clear_extent = lambda: None` on every other
        # layer (see _load_segments).  That freezes the cached extent, so
        # `reset_view()` would otherwise read the *pre-rotation* extent
        # and centre the camera on the old location.  Bypass that by
        # calling the original unbound method on each layer before
        # resetting the view.
        try:
            from napari.layers import Layer as _NapariLayer
            for _lyr in list(self.viewer.layers):
                try:
                    _NapariLayer._clear_extent(_lyr)
                except Exception:
                    pass
        except Exception:
            pass
        self.viewer.reset_view()
        self.viewer.status = (
            "Oriented: AP → top (-Y), Dorsal → right (+X). Camera reset."
        )

    def _orient_segments(self, params: dict) -> None:
        """Apply the orientation rotation to the 3D nuclei segments layer.

        Uses napari's affine transform (GPU-side) — no data is copied or
        resampled.  The rotation is the same R2 @ R1 used for spots/tracks,
        converted from XYZ to napari's ZYX axis ordering.

        On repeated calls, composes the new rotation on top of the
        previously accumulated affine so the volume stays aligned with
        the (repeatedly rotated) spot coordinates.
        """
        if self._segments_layer is None:
            return

        mid = params["center"]        # (X, Y, Z) center
        R1 = params["R1"]             # 3×3, XYZ space
        R2 = params["R2"]             # 3×3, XYZ space
        R_xyz = R2 @ R1               # combined rotation in XYZ

        # Permutation matrix: XYZ ↔ ZYX (self-inverse)
        P = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0]], dtype=float)
        R_zyx = P @ R_xyz @ P         # rotation in napari's ZYX order

        # mid is in (X,Y,Z) world coords — convert to ZYX
        mid_world = np.array([mid[2], mid[1], mid[0]])

        # Build the incremental 4×4 affine for this orientation step:
        #   new_world = R_zyx @ (old_world - mid_world)
        #            = R_zyx @ old_world  -  R_zyx @ mid_world
        inc = np.eye(4)
        inc[:3, :3] = R_zyx
        inc[:3, 3] = -R_zyx @ mid_world

        # Compose on top of the previously accumulated affine
        self._segments_affine = inc @ self._segments_affine
        affine_4x4 = self._segments_affine

        self._segments_layer.affine = affine_4x4

        # Apply same affine to raw image layer if present
        if self._processed_layer is not None:
            self._processed_layer.affine = affine_4x4

        print(f"  ✓ Nuclei 3D segments reoriented via GPU affine")

    # ── Sphere fitting ────────────────────────────────────────────────

    def _apply_manual_sphere_center(self, center_xyz):
        """Set sphere center manually; estimate radius from data."""
        x = self.spots["POSITION_X"].values
        y = self.spots["POSITION_Y"].values
        z = self.spots["POSITION_Z"].values

        pts = np.column_stack([x, y, z])
        dists = np.sqrt(np.sum((pts - center_xyz) ** 2, axis=1))
        radius = float(np.median(dists))
        rmse = float(np.sqrt(np.mean((dists - radius) ** 2)))

        result = {
            "center": center_xyz,
            "radius": radius,
            "rmse": rmse,
            "coverage": 1.0,
            "surface_used": len(pts),
            "cap_height": radius,
            "cap_base_radius": radius,
        }
        self._finish_sphere_fit(result)
        self.viewer.status = (
            f"Manual sphere center set — R={radius:.0f}µm, RMSE={rmse:.1f}µm"
        )

    def _fit_sphere(self):
        """Fit sphere to the nuclei positions (background thread)."""
        from qtpy.QtCore import QTimer

        if self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        x = self.spots["POSITION_X"].values.copy()
        y = self.spots["POSITION_Y"].values.copy()
        z = self.spots["POSITION_Z"].values.copy()

        self.lbl_sphere.value = "⏳ Fitting sphere..."
        self.viewer.status = "Fitting sphere in background..."

        self._pending_sphere = None
        self._pending_sphere_err = None

        def _bg():
            try:
                self._pending_sphere = fit_sphere(x, y, z)
            except Exception as e:
                self._pending_sphere_err = str(e)

        t = threading.Thread(target=_bg, daemon=True)
        t.start()

        def _poll():
            if t.is_alive():
                QTimer.singleShot(300, _poll)
                return
            if self._pending_sphere_err:
                self.lbl_sphere.value = f"Error: {self._pending_sphere_err}"
                self.viewer.status = "Sphere fit failed"
                return
            self._finish_sphere_fit(self._pending_sphere)

        QTimer.singleShot(300, _poll)

    def _finish_sphere_fit(self, result):
        """Finalise sphere fit on the main thread."""
        self.sphere_fit_result = result

        x = self.spots["POSITION_X"].values
        y = self.spots["POSITION_Y"].values
        z = self.spots["POSITION_Z"].values

        sph = compute_spherical_coords(
            np.column_stack([x, y, z]), result["center"], result["radius"]
        )
        for key, vals in sph.items():
            self.spots[key] = vals

        c = result["center"]
        self.lbl_sphere.value = (
            f"R={result['radius']:.0f}µm  RMSE={result['rmse']:.1f}µm  "
            f"Cov={result['coverage'] * 100:.0f}%\n"
            f"Center: ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})"
        )

        self._add_sphere_overlay()
        self.viewer.status = (
            f"Sphere fit: R={result['radius']:.0f}µm, RMSE={result['rmse']:.1f}µm"
        )

    def _add_sphere_overlay(self):
        """Add sphere cap overlay (only the portion near data)."""
        if self.sphere_fit_result is None:
            return

        c = self.sphere_fit_result["center"]
        R = self.sphere_fit_result["radius"]

        for name in ["Sphere mesh"]:
            try:
                self.viewer.layers.remove(name)
            except (ValueError, KeyError):
                pass

        pts = self.spots[["POSITION_X", "POSITION_Y", "POSITION_Z"]].values
        verts, faces = make_sphere_cap_mesh(c, R, pts)
        # napari Surface: vertices in (z, y, x) order for 3D
        verts_napari = verts[:, [2, 1, 0]]
        values = np.ones(len(verts))
        self.viewer.add_surface(
            (verts_napari, faces, values),
            name="Sphere mesh",
            opacity=0.25,
            colormap="cyan",
        )

    # ── Recolouring ───────────────────────────────────────────────────

    def _recolor(self, mode: str):
        """Recolour spots AND tracks by the chosen mode."""
        if self.spots_layer is None or self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        # Validate mode is available
        df = self.spots
        needed = {
            "depth": "SPHERICAL_DEPTH",
            "depth_travel": "SPHERICAL_DEPTH",
            "theta": "THETA_DEG",
            "phi": "PHI_DEG",
            "radial_vel": "RADIAL_VELOCITY_SMOOTH",
            "ingressing": "INGRESSING",
            "roi": "IN_ROI",
        }
        if mode in needed and needed[mode] not in df.columns:
            self.viewer.status = (
                f"'{mode}' not available — fit sphere / apply ROI first."
            )
            return

        self._current_color_mode = mode
        colors = self._get_display_colors()
        self.spots_layer.face_color = colors
        self._rebuild_tracks()
        self._recolor_segments(mode)
        self.viewer.status = f"Coloured by {mode}"

    def _compute_spot_speed(self) -> np.ndarray:
        """Compute per-spot displacement speed (µm/frame) from track data."""
        df = self.spots
        speed = np.full(len(df), np.nan)
        if "TRACK_ID" not in df.columns:
            return speed
        # Sort by track and frame for diff
        sorted_idx = df.sort_values(["TRACK_ID", "FRAME"]).index
        tid = df.loc[sorted_idx, "TRACK_ID"].values
        x = df.loc[sorted_idx, "POSITION_X"].values
        y = df.loc[sorted_idx, "POSITION_Y"].values
        z = df.loc[sorted_idx, "POSITION_Z"].values
        f = df.loc[sorted_idx, "FRAME"].values

        # Compute displacement to previous point in same track
        dx = np.diff(x, prepend=np.nan)
        dy = np.diff(y, prepend=np.nan)
        dz = np.diff(z, prepend=np.nan)
        df_val = np.diff(f, prepend=-9999)
        same_track = np.diff(tid.astype(float), prepend=-9999) == 0
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dt = np.maximum(df_val.astype(float), 1.0)
        sp = np.where(same_track & (df_val > 0), dist / dt, np.nan)
        speed[sorted_idx] = sp
        return speed

    # ── Ingression analysis ───────────────────────────────────────────

    def _compute_radial_velocity(self):
        """Compute per-spot radial velocity relative to the fitted sphere."""
        if self.sphere_fit_result is None:
            self.viewer.status = "Fit sphere first!"
            return
        if self.spots is None:
            return

        center = self.sphere_fit_result["center"]
        self.viewer.status = "Computing smoothed radial velocity..."
        self.spots = compute_radial_velocity(self.spots, center, smooth_window=5)

        rv = self.spots["RADIAL_VELOCITY_SMOOTH"].values
        valid = ~np.isnan(rv)
        if valid.any():
            med = np.nanmedian(rv)
            q10 = np.nanpercentile(rv, 10)
            q90 = np.nanpercentile(rv, 90)
            self.lbl_ingression.value = (
                f"V_r(smooth): med={med:.3f} µm/f  [{q10:.2f}, {q90:.2f}]"
            )
        else:
            self.lbl_ingression.value = "V_r: no track data"

        self.viewer.status = (
            "Radial velocity computed (smoothed, window=5). "
            "Use 'Auto-Detect Threshold' or 'Flag Ingressing Cells' next."
        )

    def _auto_detect_threshold(self):
        """Auto-detect the ingression V_r threshold using Otsu's method."""
        if self.spots is None or "TRACK_MEDIAN_RADIAL_VEL" not in self.spots.columns:
            self.viewer.status = "Compute radial velocity first!"
            return

        thresh, desc = auto_ingression_threshold(self.spots)
        # Clamp to slider range
        thresh = max(
            self.ingression_threshold.min, min(self.ingression_threshold.max, thresh)
        )
        self.ingression_threshold.value = thresh
        self.lbl_ingression.value = f"Auto: {desc}"
        self.viewer.status = f"Auto-threshold: {desc}"

    def _flag_ingression(self):
        """Flag ingressing cells based on multi-criteria detection."""
        if "RADIAL_VELOCITY" not in self.spots.columns:
            self.viewer.status = "Compute radial velocity first!"
            return

        thresh = self.ingression_threshold.value
        min_f = int(self.ingression_min_frames.value)
        inward_frac = self.ingression_inward_frac.value
        self.spots = flag_ingressing(
            self.spots,
            threshold=thresh,
            min_track_frames=min_f,
            min_inward_fraction=inward_frac,
        )

        # If "Flag only within ROI" is checked, apply two spatial filters:
        #
        #  (1) ROI PRESENCE — the track must have ≥1 spot inside the ROI
        #      box.  This removes false positives near the animal pole
        #      (entirely above the box) while preserving tracks that start
        #      outside the ROI and later migrate down into the marginal
        #      zone during ingression.
        #
        #  (2) VEGETAL BREACH — any track where ANY spot crosses past the
        #      vegetal boundary (y_max, after orientation +Y = vegetal) is
        #      epiboly / spreading, not ingression, so it is excluded.
        #
        if self.ingression_roi_only.value and self._roi_bounds is not None:
            b = self._roi_bounds
            df = self.spots

            # (1) Track must touch the ROI at least once
            in_box = (
                (df["POSITION_X"] >= b["x_min"])
                & (df["POSITION_X"] <= b["x_max"])
                & (df["POSITION_Y"] >= b["y_min"])
                & (df["POSITION_Y"] <= b["y_max"])
            )
            track_ever_in_box = in_box.groupby(df["TRACK_ID"]).transform("any")
            self.spots.loc[~track_ever_in_box, "INGRESSING"] = False

            # (2) Exclude tracks breaching the vegetal boundary
            vegetal_breach = (
                df.groupby("TRACK_ID")["POSITION_Y"].transform("max") > b["y_max"]
            )
            self.spots.loc[vegetal_breach, "INGRESSING"] = False

        n_ing = int(self.spots["INGRESSING"].sum())
        n_total = len(self.spots)
        n_tracks = 0
        if "TRACK_ID" in self.spots.columns:
            n_tracks = int(
                self.spots.loc[self.spots["INGRESSING"], "TRACK_ID"].nunique()
            )
        pct = n_ing / max(n_total, 1) * 100
        roi_tag = " [ROI-only]" if self.ingression_roi_only.value else ""
        self.lbl_ingression.value = (
            f"Ingressing: {n_ing:,} spots ({n_tracks} tracks, {pct:.1f}%){roi_tag}\n"
            f"V_r≤{thresh:.3f}, inward≥{inward_frac:.0%}, ≥{min_f}f, net-in+sustained"
        )
        self.viewer.status = (
            f"Flagged {n_ing:,} ingressing spots ({n_tracks} tracks){roi_tag}. "
            f"Use 'Colour Ingressing' to visualise."
        )

    def _on_ingression_tracks_toggle(self):
        """Show only tracks/spots of ingressing cells, or all."""
        if self.spots is None:
            return
        if self.ingression_tracks_only.value and "INGRESSING" not in self.spots.columns:
            self.viewer.status = "Flag ingressing cells first!"
            self.ingression_tracks_only.value = False
            return
        self._refresh_points()  # also filter spots display
        self._rebuild_tracks()

    # ── Refresh layers ────────────────────────────────────────────────

    def _refresh_points(self):
        """Update the points layer data after orientation or slider change."""
        if self.spots_layer is None:
            return
        data, idx = self._build_display_points()
        self._display_idx = idx
        colors = self._get_display_colors()
        self.spots_layer.data = data
        self.spots_layer.face_color = colors

    def _refresh_tracks(self):
        """Update the tracks layer after orientation or slider change."""
        self._rebuild_tracks()

    def _on_display_pct_changed(self):
        """Handle display percentage slider change."""
        self._display_pct = self.display_pct_slider.value
        if self.spots is not None:
            self._refresh_points()

    def _on_max_tracks_changed(self):
        """Handle max tracks slider change (debounced)."""
        self._max_tracks = int(self.max_tracks_slider.value)
        self._schedule_track_rebuild()

    def _on_track_width_changed(self):
        """Handle track width slider change (debounced)."""
        self._track_width = float(self.track_width_slider.value)
        self._schedule_track_rebuild()

    # ── Track time range ──────────────────────────────────────────────

    def _on_track_frame_changed(self):
        """Handle track frame-range slider change (debounced)."""
        f_start = int(self.track_frame_start.value)
        f_end = int(self.track_frame_end.value)
        if f_end < f_start:
            f_end = f_start
        self._track_frame_min = f_start
        self._track_frame_max = f_end
        self.viewer.status = (
            f"Tracks filtered to frames {f_start}–{f_end} "
            f"(only tracks with data in this range are shown)"
        )
        self._schedule_track_rebuild()

    def _schedule_track_rebuild(self):
        """Debounce track rebuilds — waits 200ms after last slider change."""
        from qtpy.QtCore import QTimer

        if not hasattr(self, "_debounce_timer"):
            self._debounce_timer = QTimer()
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._deferred_track_rebuild)
        self._debounce_timer.start(200)  # ms

    def _deferred_track_rebuild(self):
        """Execute the actual track rebuild after debounce delay."""
        if self.spots is not None:
            self._rebuild_tracks()

    # ── 3D ROI box ────────────────────────────────────────────────────

    def _activate_roi_drawing(self):
        """Switch to 2D, add Shapes layer in rectangle-draw mode."""
        if self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        self._drawing_roi = True  # suppress track rebuilds

        # Remove previous shapes layer
        if self._roi_shapes_layer is not None:
            try:
                self.viewer.layers.remove(self._roi_shapes_layer)
            except (ValueError, KeyError):
                pass
            self._roi_shapes_layer = None

        # Remember 3D state, switch to 2D (Shapes drawing only works in 2D)
        self._was_3d = self.viewer.dims.ndisplay == 3

        # Hide segments & raw layers during 2D ROI drawing — their
        # rotation affine triggers non-orthogonal slicing warnings and
        # misaligns the 2D coordinate space.
        self._seg_was_visible = False
        if self._segments_layer is not None:
            self._seg_was_visible = self._segments_layer.visible
            self._segments_layer.visible = False
        self._processed_was_visible = False
        if self._processed_layer is not None:
            self._processed_was_visible = self._processed_layer.visible
            self._processed_layer.visible = False

        if self._was_3d:
            self.viewer.dims.ndisplay = 2

        self._roi_shapes_layer = self.viewer.add_shapes(
            name="ROI draw",
            ndim=4,
            edge_color="yellow",
            edge_width=2,
            face_color=np.array([1.0, 1.0, 0.0, 0.15]),
        )
        self._roi_shapes_layer.events.data.connect(self._on_roi_drawn)
        self._roi_shapes_layer.mode = "add_rectangle"
        self.viewer.status = (
            "\u25a3 Draw a rectangle to select a region "
            "(selects all nuclei across all Z depths; view returns to 3D after)"
        )

    def _on_roi_drawn(self, event=None):
        """Auto-apply ROI when a rectangle is drawn, then restore 3D."""
        if self._roi_updating:
            return
        if self._roi_shapes_layer is None:
            return
        shapes = self._roi_shapes_layer.data
        if len(shapes) == 0 or self.spots is None:
            return

        self._roi_updating = True
        try:
            # Keep only the last drawn rectangle
            if len(shapes) > 1:
                self._roi_shapes_layer.data = [shapes[-1]]
                return  # re-triggers via event; guard prevents recursion

            # In 2D mode with 4D data, rectangle verts are (4, 4): [t, z, y, x]
            rect_verts = np.asarray(shapes[0])
            if rect_verts.shape[1] == 4:
                x_vals = rect_verts[:, 3]  # POSITION_X
                y_vals = rect_verts[:, 2]  # POSITION_Y
            else:
                x_vals = rect_verts[:, 2]  # fallback 3D
                y_vals = rect_verts[:, 1]

            x_min, x_max = float(x_vals.min()), float(x_vals.max())
            y_min, y_max = float(y_vals.min()), float(y_vals.max())

            z_all = self.spots["POSITION_Z"].values
            z_min, z_max = float(z_all.min()), float(z_all.max())

            self._roi_bounds = {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
            }

            in_roi = (
                (self.spots["POSITION_X"].values >= x_min)
                & (self.spots["POSITION_X"].values <= x_max)
                & (self.spots["POSITION_Y"].values >= y_min)
                & (self.spots["POSITION_Y"].values <= y_max)
            )
            self.spots["IN_ROI"] = in_roi

            n_roi = int(in_roi.sum())
            n_total = len(self.spots)
            self.lbl_roi.value = (
                f"ROI: {n_roi:,}/{n_total:,} ({n_roi / n_total * 100:.1f}%) selected"
            )

            # Remove the drawing layer and switch back to 3D
            try:
                self.viewer.layers.remove(self._roi_shapes_layer)
            except (ValueError, KeyError):
                pass
            self._roi_shapes_layer = None

            # Restore 3D view, segments visibility, and draw wireframe
            if getattr(self, "_was_3d", True):
                self.viewer.dims.ndisplay = 3
            if self._segments_layer is not None and getattr(
                self, '_seg_was_visible', False
            ):
                self._segments_layer.visible = True
            if self._processed_layer is not None and getattr(
                self, '_processed_was_visible', False
            ):
                self._processed_layer.visible = True
            self._draw_roi_box()

            self.viewer.status = (
                f"ROI applied: {n_roi:,} spots inside rectangle (all Z depths)"
            )
        finally:
            self._roi_updating = False
            self._drawing_roi = False  # release guard — allow track rebuilds
            # Now do a single rebuild with the new ROI
            if self._roi_only:
                self._refresh_points()
                self._rebuild_tracks()

    def _draw_roi_box(self):
        """Draw the ROI bounding box as a wireframe Shapes layer."""
        if self._roi_bounds is None:
            return

        b = self._roi_bounds
        # Remove old
        for name in ["ROI box"]:
            try:
                self.viewer.layers.remove(name)
            except (ValueError, KeyError):
                pass

        # 12 edges of the box, in (z, y, x) for napari 3D
        corners = np.array(
            [
                [b["z_min"], b["y_min"], b["x_min"]],
                [b["z_min"], b["y_min"], b["x_max"]],
                [b["z_min"], b["y_max"], b["x_min"]],
                [b["z_min"], b["y_max"], b["x_max"]],
                [b["z_max"], b["y_min"], b["x_min"]],
                [b["z_max"], b["y_min"], b["x_max"]],
                [b["z_max"], b["y_max"], b["x_min"]],
                [b["z_max"], b["y_max"], b["x_max"]],
            ]
        )
        edges = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],  # z_min face
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],  # z_max face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # connecting edges
        ]
        lines = [np.array([corners[e[0]], corners[e[1]]]) for e in edges]

        self.viewer.add_shapes(
            lines,
            shape_type="line",
            edge_color="yellow",
            edge_width=3,
            name="ROI box",
            ndim=3,
        )

    def _on_roi_toggle(self):
        """Toggle: show all spots/tracks or only ROI-selected."""
        self._roi_only = self.roi_show_only.value
        if self.spots is not None:
            self._refresh_points()
            self._rebuild_tracks()

    def _clear_roi(self):
        """Clear the ROI selection, shapes layer, and wireframe box."""
        self._roi_bounds = None
        self._roi_only = False
        self._drawing_roi = False
        self.roi_show_only.value = False
        if self.spots is not None and "IN_ROI" in self.spots.columns:
            self.spots.drop(columns=["IN_ROI"], inplace=True)
        self.lbl_roi.value = "ROI: not set"
        for name in ["ROI box", "ROI draw"]:
            try:
                self.viewer.layers.remove(name)
            except (ValueError, KeyError):
                pass
        self._roi_shapes_layer = None
        if self.spots is not None:
            self._refresh_points()
            self._rebuild_tracks()
        self.viewer.status = "ROI cleared"

    # ── Cross-section view ────────────────────────────────────────────

    def _open_cross_section(self):
        """Open a cross-section 2D scatter viewer in a separate window."""
        if self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        self.viewer.status = "Opening cross-section viewer..."
        CrossSectionViewer(self)

    # ── Export & Video ────────────────────────────────────────────────

    def _export(self):
        """Export comprehensive analysis output for downstream R pipeline.

        Runs the heavy CSV writes in a background thread to avoid freezing
        the UI on large datasets (millions of rows).
        """
        import json

        if self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        from qtpy.QtCore import QTimer

        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True)

        self.lbl_export.value = "Exporting (please wait)..."
        self.viewer.status = "Exporting analysis files in background..."

        # Snapshot everything we need — avoids touching GUI state from thread
        spots_df = self.spots.copy()
        tracks_raw = self.tracks_raw.copy() if self.tracks_raw is not None else None
        sphere = (
            dict(self.sphere_fit_result) if self.sphere_fit_result is not None else None
        )
        roi = dict(self._roi_bounds) if self._roi_bounds is not None else None
        roi_flag = getattr(self, "ingression_roi_only", None)
        roi_restricted = roi_flag is not None and roi_flag.value
        thresh_val = self.ingression_threshold.value
        min_f_val = int(self.ingression_min_frames.value)
        inward_frac_val = self.ingression_inward_frac.value
        oriented = self.oriented
        track_frame_min = self._track_frame_min
        track_frame_max = self._track_frame_max
        display_pct = self._display_pct
        max_tracks = self._max_tracks
        color_mode = self._current_color_mode

        # Snapshot gastrulation landmarks (margin + ingression)
        margin_lm = self.margin_landmark.copy() if self.margin_landmark is not None else None
        ingression_lm = self.ingression_landmark.copy() if self.ingression_landmark is not None else None
        sphere_for_lm = sphere  # already snapshot

        self._pending_export_err = None
        self._pending_export_files = None

        def _bg():
            try:
                files_written = []

                # 1. Enriched spots CSV (all computed columns)
                spots_df.to_csv(out_dir / "oriented_spots.csv", index=False)
                files_written.append("oriented_spots.csv")

                # 2. Enriched tracks CSV — spots filtered to tracked nuclei
                #    only, sorted by TRACK_ID + FRAME for easy downstream use
                if "TRACK_ID" in spots_df.columns:
                    tracked = spots_df.dropna(subset=["TRACK_ID"]).sort_values(
                        ["TRACK_ID", "FRAME"]
                    )
                    if len(tracked) > 0:
                        tracked.to_csv(out_dir / "oriented_tracks.csv", index=False)
                        files_written.append("oriented_tracks.csv")

                # 3. Sphere parameters
                if sphere is not None:
                    c = sphere["center"]
                    params = pd.DataFrame(
                        {
                            "parameter": [
                                "center_x",
                                "center_y",
                                "center_z",
                                "radius",
                                "rmse",
                                "coverage",
                                "surface_used",
                                "cap_height",
                                "cap_base_radius",
                            ],
                            "value": [
                                c[0],
                                c[1],
                                c[2],
                                sphere["radius"],
                                sphere["rmse"],
                                sphere["coverage"],
                                sphere["surface_used"],
                                sphere["cap_height"],
                                sphere["cap_base_radius"],
                            ],
                        }
                    )
                    params.to_csv(out_dir / "sphere_params.csv", index=False)
                    files_written.append("sphere_params.csv")

                # 4. ROI bounds
                if roi is not None:
                    roi_df = pd.DataFrame(
                        {"parameter": list(roi.keys()), "value": list(roi.values())}
                    )
                    roi_df.to_csv(out_dir / "roi_bounds.csv", index=False)
                    files_written.append("roi_bounds.csv")

                # 5. Ingression parameters
                if "INGRESSING" in spots_df.columns:
                    n_ing = int(spots_df["INGRESSING"].sum())
                    n_tracks_ing = 0
                    if "TRACK_ID" in spots_df.columns:
                        n_tracks_ing = int(
                            spots_df.loc[
                                spots_df["INGRESSING"].astype(bool), "TRACK_ID"
                            ].nunique()
                        )
                    ing_df = pd.DataFrame(
                        {
                            "parameter": [
                                "threshold_um_per_frame",
                                "min_track_frames",
                                "min_inward_fraction",
                                "roi_restricted",
                                "detection_method",
                                "criteria",
                                "n_ingressing_spots",
                                "n_ingressing_tracks",
                            ],
                            "value": [
                                thresh_val,
                                min_f_val,
                                inward_frac_val,
                                roi_restricted,
                                "embryology-informed multi-criteria",
                                "median_Vr+inward_frac+net_disp+sustained",
                                n_ing,
                                n_tracks_ing,
                            ],
                        }
                    )
                    ing_df.to_csv(out_dir / "ingression_params.csv", index=False)
                    files_written.append("ingression_params.csv")

                # 5b. Gastrulation landmarks (margin + ingression center)
                lm_rows = []
                if margin_lm is not None:
                    lm_rows.append({"landmark": "margin", "x": margin_lm[0],
                                    "y": margin_lm[1], "z": margin_lm[2]})
                    if sphere_for_lm is not None:
                        c = sphere_for_lm["center"]
                        d = margin_lm - c
                        r = np.linalg.norm(d)
                        theta = float(np.degrees(np.arccos(np.clip(-d[1] / max(r, 1e-10), -1, 1))))
                        phi = float(np.degrees(np.arctan2(d[2], d[0])))
                        lm_rows[-1].update({"theta": theta, "phi": phi})
                if ingression_lm is not None:
                    lm_rows.append({"landmark": "ingression_center", "x": ingression_lm[0],
                                    "y": ingression_lm[1], "z": ingression_lm[2]})
                    if sphere_for_lm is not None:
                        c = sphere_for_lm["center"]
                        d = ingression_lm - c
                        r = np.linalg.norm(d)
                        theta = float(np.degrees(np.arccos(np.clip(-d[1] / max(r, 1e-10), -1, 1))))
                        phi = float(np.degrees(np.arctan2(d[2], d[0])))
                        lm_rows[-1].update({"theta": theta, "phi": phi})
                if lm_rows:
                    lm_df = pd.DataFrame(lm_rows)
                    lm_df.to_csv(out_dir / "gastrulation_landmarks.csv", index=False)
                    files_written.append("gastrulation_landmarks.csv")

                # 6. Per-track summary CSV
                if "TRACK_ID" in spots_df.columns:
                    track_cols = [
                        "TRACK_MEDIAN_RADIAL_VEL",
                        "TRACK_MEAN_RADIAL_VEL",
                        "TRACK_INWARD_FRACTION",
                        "TRACK_MAX_INWARD_VEL",
                        "TRACK_NET_RADIAL_DISP",
                        "TRACK_SUSTAINED_INWARD",
                        "INGRESSION_ONSET_FRAME",
                        "INGRESSING",
                        "INGRESSION_SCORE",
                    ]
                    avail_cols = [c for c in track_cols if c in spots_df.columns]
                    if avail_cols:
                        grp = spots_df.dropna(subset=["TRACK_ID"]).groupby("TRACK_ID")
                        track_summary = grp.agg(
                            n_spots=("FRAME", "count"),
                            frame_start=("FRAME", "min"),
                            frame_end=("FRAME", "max"),
                            mean_x=("POSITION_X", "mean"),
                            mean_y=("POSITION_Y", "mean"),
                            mean_z=("POSITION_Z", "mean"),
                        )
                        first_per_track = grp[avail_cols].first()
                        track_summary = track_summary.join(first_per_track)
                        for sc in ["SPHERICAL_DEPTH", "THETA_DEG", "PHI_DEG"]:
                            if sc in spots_df.columns:
                                track_summary[f"mean_{sc}"] = grp[sc].mean()
                        if "IN_ROI" in spots_df.columns:
                            track_summary["IN_ROI"] = grp["IN_ROI"].any()
                        track_summary.to_csv(out_dir / "track_summary.csv")
                        files_written.append("track_summary.csv")

                # 7. Analysis metadata (JSON)
                meta = {
                    "oriented": oriented,
                    "sphere_fitted": sphere is not None,
                    "roi_set": roi is not None,
                    "roi_restricted_ingression": roi_restricted,
                    "time_window": {
                        "frame_min": track_frame_min,
                        "frame_max": track_frame_max,
                    },
                    "display": {
                        "display_pct": display_pct,
                        "max_tracks": max_tracks,
                        "current_color_mode": color_mode,
                    },
                    "n_spots": len(spots_df),
                    "n_frames": int(spots_df["FRAME"].nunique()),
                    "columns_exported": list(spots_df.columns),
                    "ingression_criteria": {
                        "method": "multi-criteria embryology-informed",
                        "criteria": [
                            "median V_r <= threshold",
                            "inward fraction >= min_inward_fraction",
                            "track length >= min_track_frames",
                            "net radial displacement >= 0.5 µm inward",
                            "sustained consecutive inward >= max(3, 15% of track)",
                        ],
                    },
                }
                if sphere is not None:
                    meta["sphere_radius"] = float(sphere["radius"])
                if roi is not None:
                    meta["roi_bounds"] = {k: float(v) for k, v in roi.items()}
                with open(out_dir / "analysis_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2, default=str)
                files_written.append("analysis_metadata.json")

                self._pending_export_files = files_written
            except Exception as e:
                self._pending_export_err = str(e)

        t = threading.Thread(target=_bg, daemon=True)
        t.start()

        def _poll():
            if t.is_alive():
                QTimer.singleShot(500, _poll)
                return
            if self._pending_export_err:
                self.lbl_export.value = f"Export error: {self._pending_export_err}"
                self.viewer.status = f"Export failed: {self._pending_export_err}"
                return
            files = self._pending_export_files or []
            self.lbl_export.value = f"Exported {len(files)} files to {out_dir}/"
            self.viewer.status = f"Exported: {', '.join(files)}"

        QTimer.singleShot(500, _poll)

    def _record_video(self):
        """Record a time-lapse video iterating through the time window."""
        from qtpy.QtCore import QTimer

        if self.spots is None:
            self.viewer.status = "Load spots first!"
            return

        out_dir = Path("analysis_output")
        out_dir.mkdir(exist_ok=True)
        f_start = int(self.track_frame_start.value)
        f_end = int(self.track_frame_end.value)
        all_frames = sorted(self.spots["FRAME"].unique())
        frames = [int(f) for f in all_frames if f_start <= f <= f_end]
        if not frames:
            self.viewer.status = "No frames in selected range!"
            return

        try:
            duration_s = float(self.duration_slider.value)
        except Exception:
            duration_s = 0.0
        if duration_s > 0:
            fps = max(1, min(60, int(round(len(frames) / duration_s))))
        else:
            fps = max(1, min(30, len(frames) // 10 or 10))

        self._vid_frames_list = frames
        self._vid_images = []
        self._vid_idx = 0
        self._vid_total = len(frames)
        self._vid_path = out_dir / "embryo_timelapse.mp4"
        self._vid_scale = float(self.render_scale_slider.value)
        self._vid_fps = fps
        self.lbl_export.value = f"Recording: 0/{self._vid_total} @ {fps} fps..."
        self.viewer.status = (
            f"Recording video ({len(frames)} frames @ {fps} fps)"
        )

        def _capture_next():
            if self._vid_idx >= self._vid_total:
                ok, msg = _write_video(self._vid_path, self._vid_images,
                                        self._vid_fps)
                if ok:
                    self.lbl_export.value = f"Video: {self._vid_path}"
                    self.viewer.status = f"Video saved to {self._vid_path} ({msg})"
                else:
                    self.lbl_export.value = f"Video failed: {msg}"
                    self.viewer.status = msg
                return

            frame = self._vid_frames_list[self._vid_idx]
            dims = list(self.viewer.dims.current_step)
            dims[0] = frame
            self.viewer.dims.current_step = tuple(dims)

            def _screenshot():
                img = _grab_canvas_screenshot(self.viewer, scale=self._vid_scale)
                if img is None:
                    # Skip silently but advance so the chain can finish.
                    self._vid_idx += 1
                    QTimer.singleShot(50, _capture_next)
                    return
                self._vid_images.append(img)
                self._vid_idx += 1
                if self._vid_idx % 10 == 0 or self._vid_idx == self._vid_total:
                    self.lbl_export.value = (
                        f"Recording: {self._vid_idx}/{self._vid_total} "
                        f"@ {self._vid_fps} fps..."
                    )
                QTimer.singleShot(50, _capture_next)

            # Slightly longer delay so the dim change has time to render
            # the new time-step (points/tracks/segments) before we grab.
            QTimer.singleShot(350, _screenshot)

        QTimer.singleShot(400, _capture_next)


# ############################################################################
#                          VIDEO WRITER HELPER
# ############################################################################


def _grab_canvas_screenshot(viewer, scale: float = 1.0) -> "np.ndarray | None":
    """Return an RGB uint8 screenshot of the napari canvas with even dims.

    Forces a Qt event-loop flush + canvas redraw so the captured frame
    actually reflects the current dims/layer state.  ``scale`` upsamples
    the rendered canvas (e.g. ``scale=2`` doubles each axis → 4× pixels)
    via napari's built-in ``scale`` argument.  Returns ``None`` if the
    canvas is not yet available or the resulting image is empty.
    """
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:
        QApplication = None

    # Make sure pending paint/dim events have run before grabbing pixels.
    if QApplication is not None and QApplication.instance() is not None:
        try:
            QApplication.processEvents()
            QApplication.processEvents()
        except Exception:
            pass

    # Ask napari's vispy canvas to repaint synchronously, if possible.
    try:
        canvas = viewer.window._qt_viewer.canvas  # napari ≥0.5
    except Exception:
        canvas = None
    if canvas is not None:
        for meth in ("update", "_draw_event", "render"):
            fn = getattr(canvas, meth, None)
            if callable(fn):
                try:
                    fn() if meth != "_draw_event" else fn(None)
                except Exception:
                    pass
                break
    if QApplication is not None and QApplication.instance() is not None:
        try:
            # Drain a few more event-loop iterations so vispy actually
            # paints the new frame (track tail-fade uniforms etc.) before
            # we read pixels — otherwise screenshots can capture stale
            # GPU state and tracks/points appear dim or wrong.
            for _ in range(4):
                QApplication.processEvents()
        except Exception:
            pass

    try:
        img = viewer.screenshot(canvas_only=True, flash=False, scale=scale)
    except TypeError:
        # Older napari versions without the ``flash`` / ``scale`` kwargs.
        try:
            img = viewer.screenshot(canvas_only=True, scale=scale)
        except TypeError:
            try:
                img = viewer.screenshot(canvas_only=True)
            except Exception:
                return None
        except Exception:
            return None
    except Exception:
        return None

    if img is None or not hasattr(img, "shape") or img.size == 0:
        return None
    if img.ndim != 3:
        return None

    # RGBA → RGB by alpha-compositing onto the canvas background.
    # Naively dropping the alpha channel (img[:,:,:3]) makes
    # semi-transparent layers (tracks with tail-fade, points with
    # opacity<1) look much dimmer in the saved frame than they do in
    # the live canvas, because napari returns un-premultiplied RGBA
    # against a transparent canvas background.
    if img.shape[2] == 4:
        bg = (0.0, 0.0, 0.0)  # default
        try:
            bgc = viewer.window._qt_viewer.canvas.bgcolor.rgb
            bg = (float(bgc[0]), float(bgc[1]), float(bgc[2]))
        except Exception:
            try:
                # napari ≥0.5 also exposes background through the camera
                bg_arr = np.asarray(viewer.window._qt_viewer.canvas.bgcolor.rgba)
                bg = tuple(float(c) for c in bg_arr[:3])
            except Exception:
                pass
        rgb = img[:, :, :3].astype(np.float32)
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        bg_arr = np.array(bg, dtype=np.float32) * 255.0
        composed = rgb * a + bg_arr * (1.0 - a)
        img = np.clip(composed, 0, 255).astype(np.uint8)
    elif img.shape[2] != 3:
        return None

    # libx264 + yuv420p requires even dimensions.
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return None
    if h % 2 or w % 2:
        img = img[: h - h % 2, : w - w % 2]
    if img.shape[0] < 2 or img.shape[1] < 2:
        return None

    return np.ascontiguousarray(img)


def _resolve_ffmpeg_exe() -> "str | None":
    """Locate an ffmpeg binary.

    Order: ``imageio_ffmpeg`` (bundled, preferred) → ``$IMAGEIO_FFMPEG_EXE``
    → system ``ffmpeg`` on ``PATH``.  Returns ``None`` if nothing is found.
    """
    try:
        import imageio_ffmpeg as iioff
        exe = iioff.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass

    env_exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env_exe and Path(env_exe).exists():
        return env_exe

    import shutil
    return shutil.which("ffmpeg")


def _write_video(path: "Path", images: list, fps: int) -> "tuple[bool, str]":
    """Encode ``images`` to ``path`` (mp4) by piping raw RGB to ffmpeg.

    Uses ``imageio_ffmpeg`` if installed (bundled binary), otherwise a
    system ``ffmpeg`` on ``PATH``.  Avoids ``imageio.get_writer``'s
    plugin auto-detection, which can misroute ``.mp4`` to the TIFF
    writer in some environments.

    Returns ``(success, message)``.  Frames whose shape differs from the
    first valid frame are padded/cropped to match (libx264 needs a fixed
    frame size).
    """
    valid = [im for im in images if im is not None and getattr(im, "size", 0) > 0]
    if not valid:
        return False, "no valid frames captured"

    # Lock to the first frame's shape; pad/crop any others to match.
    h0, w0 = valid[0].shape[:2]
    fixed = []
    for im in valid:
        h, w = im.shape[:2]
        if (h, w) != (h0, w0):
            new = np.zeros((h0, w0, 3), dtype=im.dtype)
            ch, cw = min(h, h0), min(w, w0)
            new[:ch, :cw] = im[:ch, :cw]
            im = new
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        fixed.append(np.ascontiguousarray(im))

    ffmpeg_exe = _resolve_ffmpeg_exe()
    if ffmpeg_exe is None:
        return _write_video_pngs(
            path, fixed,
            "no ffmpeg found — install with `mamba install -c conda-forge "
            "imageio-ffmpeg ffmpeg` or `pip install imageio-ffmpeg`"
        )

    # ── Primary path: pipe raw RGB to ffmpeg → libx264 + yuv420p ──
    ok, reason = _ffmpeg_encode(
        ffmpeg_exe, path, fixed, fps,
        codec="libx264", pix_fmt_out="yuv420p",
        extra=["-preset", "medium", "-crf", "20"],
    )
    if ok:
        return True, reason

    # Fallback codec: mpeg4 (works without libx264 in minimal ffmpeg builds)
    ok2, reason2 = _ffmpeg_encode(
        ffmpeg_exe, path, fixed, fps,
        codec="mpeg4", pix_fmt_out="yuv420p",
        extra=["-q:v", "5"],
    )
    if ok2:
        return True, f"{reason2} (mpeg4 fallback after: {reason})"

    return _write_video_pngs(path, fixed, f"{reason}; mpeg4 also failed: {reason2}")


def _ffmpeg_encode(ffmpeg_exe: str, path: "Path", fixed: list, fps: int,
                   codec: str, pix_fmt_out: str,
                   extra: list) -> "tuple[bool, str]":
    """Spawn ffmpeg, pipe raw RGB frames, capture stderr on failure."""
    import subprocess

    h0, w0 = fixed[0].shape[:2]
    cmd = [
        ffmpeg_exe,
        "-y",                            # overwrite output
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w0}x{h0}",
        "-r", str(fps),
        "-i", "-",                       # read from stdin
        "-an",                           # no audio
        "-vcodec", codec,
        "-pix_fmt", pix_fmt_out,
        *extra,
        str(path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        return False, f"{codec} spawn failed: {e}"

    try:
        for im in fixed:
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        return False, f"{codec} pipe error: {e}"

    err = proc.stderr.read().decode("utf-8", errors="replace").strip()
    rc = proc.wait()
    if rc != 0:
        return False, f"{codec} ffmpeg rc={rc}: {err[:200]}"

    if not path.exists() or path.stat().st_size <= 4096:
        return False, f"{codec} produced empty file ({err[:200]})"

    return True, f"wrote {len(fixed)} frames @ {fps} fps with {codec}"


def _write_video_pngs(path: "Path", fixed: list, reason: str) -> "tuple[bool, str]":
    """Last-resort fallback: dump frames as PNGs next to the target."""
    png_dir = path.parent / (path.stem + "_frames")
    png_dir.mkdir(exist_ok=True)
    try:
        import imageio.v2 as iio
    except Exception:
        import imageio as iio
    for i, im in enumerate(fixed):
        iio.imwrite(str(png_dir / f"frame_{i:04d}.png"), im)
    return False, f"mp4 failed ({reason}); saved {len(fixed)} PNGs to {png_dir}/"


# ############################################################################
#                               UTILITIES
# ############################################################################


def _viridis_colors(values: np.ndarray) -> np.ndarray:
    """Map normalised [0,1] values to RGBA viridis colours."""
    try:
        import matplotlib.pyplot as plt

        cmap = plt.cm.viridis
        return cmap(values)
    except ImportError:
        pass

    # Fallback: simple blue→green→yellow gradient
    n = len(values)
    colors = np.zeros((n, 4))
    colors[:, 0] = np.clip(values * 2 - 1, 0, 1)  # R
    colors[:, 1] = np.clip(1 - 2 * np.abs(values - 0.5), 0, 1)  # G
    colors[:, 2] = np.clip(1 - values * 2, 0, 1)  # B
    colors[:, 3] = 1.0  # A
    return colors


# ############################################################################
#                                  MAIN
# ############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="napari-based 4D embryo viewer (TrackMate & ultrack formats)"
    )
    parser.add_argument(
        "spots",
        nargs="?",
        default=None,
        help="Path to spots/tracks CSV — auto-detects TrackMate or ultrack format",
    )
    parser.add_argument(
        "--tracks", "-t", default=None, help="Path to tracks CSV (optional)"
    )
    parser.add_argument(
        "--segments", "-s", default=None,
        help="Path to segments.zarr (ultrack label volume, 4D)",
    )
    parser.add_argument(
        "--voxel-size", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("Z", "Y", "X"),
        help="Voxel size in µm (Z Y X) for segments layer. Default: 1.0 1.0 1.0",
    )
    parser.add_argument(
        "--max-spots",
        type=int,
        default=None,
        help="Max spots to load (subsample for speed)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        metavar="N",
        help="Spatial downsample factor for segments display layer "
             "(e.g. 2 = half-res → 8× fewer voxels, 4 = quarter-res → "
             "64× fewer). Full-res data is kept in RAM for analysis. "
             "Default: 1 (full resolution). "
             "Use 2 or 4 on machines with limited GPU memory.",
    )
    parser.add_argument(
        "--load-downsample",
        type=int,
        default=1,
        metavar="N",
        dest="load_downsample",
        help="Spatial downsample factor applied when loading segments into RAM "
             "with --preload (e.g. 2 = half-res → 8× less RAM, 4 = quarter-res → "
             "64× less RAM). Unlike --downsample (display only), this reduces the "
             "array actually stored in memory. Analysis resolution is reduced "
             "accordingly. Default: 1 (full resolution).",
    )
    parser.add_argument(
        "--processed", "-p", default=None,
        help="Path to processed image (TIFF or zarr, 4D: T×Z×Y×X), or a "
             "folder containing one TIFF/zarr per timepoint (e.g. "
             "tp_000.tif, tp_001.tif, ...). When a folder is given, files "
             "are sorted by the trailing numeric component of the stem and "
             "stacked along the time axis. Displayed as a grayscale volume "
             "alongside segments and tracks.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load entire segments.zarr into RAM with uint16 remapping "
             "and GPU-optimised display volume for instant smooth "
             "scrubbing. Recommended for machines with enough RAM.",
    )
    args = parser.parse_args()

    spots_df = None
    tracks_df = None

    # Determine spots source: explicit spots arg, or fall back to --tracks
    spots_path = args.spots
    if spots_path is None and args.tracks is not None:
        spots_path = args.tracks  # tracks CSV has all position columns

    if spots_path:
        fmt = _detect_csv_format(spots_path)
        print(f"Loading spots from {spots_path} (detected: {fmt} format)...")
        spots_df = load_spots(spots_path)
        print(f"  {len(spots_df):,} spots, {spots_df['FRAME'].nunique()} frames")

        if args.max_spots and len(spots_df) > args.max_spots:
            print(f"  Subsampling to {args.max_spots:,} spots...")
            spots_df = spots_df.sample(args.max_spots, random_state=42)

    if args.tracks:
        print(f"Loading tracks from {args.tracks}...")
        fmt_t = _detect_csv_format(args.tracks)
        if fmt_t == "ultrack":
            tracks_df = load_ultrack_csv(args.tracks)
        else:
            tracks_df = load_trackmate_csv(args.tracks)
        print(f"  {len(tracks_df):,} track records")

    if spots_df is None and args.segments is None:
        print("Launching viewer — use buttons to open files.")

    print("Launching napari viewer...")
    viewer = EmbryoViewer(
        spots_df, tracks_df,
        segments_zarr_path=args.segments,
        processed_path=args.processed,
        voxel_size=tuple(args.voxel_size),
        downsample=args.downsample,
        load_downsample=args.load_downsample,
        preload=args.preload,
    )

    napari, _ = _import_napari()
    napari.run()


if __name__ == "__main__":
    main()
