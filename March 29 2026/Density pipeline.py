===============================================================================
MW–M31 MERGER DENSITY ANALYSIS PIPELINE
===============================================================================
Author  : Abhinav Vatsa
===============================================================================

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 0 — CONFIGURATION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import os
import tarfile
import shutil
import tempfile
import warnings
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from ReadFile import Read
from CenterOfMass2 import CenterOfMass

START_SNAP = 0
END_SNAP   = 800
PTYPE = 1
R_BINS = np.logspace(-1, np.log10(400.0), 40)
MIN_PART_SHELL = 20
MAP_EXTENT_KPC = 400.0
MAP_BINS       = 256
MAP_SMOOTH_SIGMA = 2.0
R_PROJ_BINS = np.logspace(np.log10(0.5), np.log10(400.0), 35)
R_FIT_MIN_KPC = 1.0
R_FIT_MAX_KPC = 200.0
STEP_MAPS  = 8
STEP_FIT   = 4
ANIM_FPS     = 20
ANIM_DPI     = 100
ANIM_BITRATE = 2000
ANIM_STEP    = 4
G_KPC_KMS2_MSUN = 4.30091e-6
MASS_UNIT_MSUN  = 1.0e10
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": "#0d0d18",
    "axes.facecolor":   "#0d0d18",
    "axes.edgecolor":   "#2a2a4a",
    "axes.labelcolor":  "#c8c8e8",
    "axes.grid":        True,
    "grid.color":       "#1e1e36",
    "grid.linewidth":   0.6,
    "xtick.color":      "#9090b0",
    "ytick.color":      "#9090b0",
    "text.color":       "#c8c8e8",
    "legend.facecolor": "#0d0d18",
    "legend.edgecolor": "#2a2a4a",
    "legend.fontsize":  8,
    "font.family":      "monospace",
})
SNAPSHOTS    = np.arange(START_SNAP, END_SNAP + 1)
ns           = len(SNAPSHOTS)
nb_sph       = len(R_BINS) - 1
nb_proj      = len(R_PROJ_BINS) - 1
r_mid_sph    = 0.5 * (R_BINS[:-1]   + R_BINS[1:])
r_mid_proj   = 0.5 * (R_PROJ_BINS[:-1] + R_PROJ_BINS[1:])
shell_vols   = (4.0 / 3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)
ring_areas   = np.pi * (R_PROJ_BINS[1:]**2 - R_PROJ_BINS[:-1]**2)
PROFILE_FRACS   = [0.0, 0.2, 0.4, 0.65, 1.0]
PROFILE_INDICES = [int(f * (ns - 1)) for f in PROFILE_FRACS]
PROFILE_LABELS  = [f"Snap {SNAPSHOTS[k]}" for k in PROFILE_INDICES]
PROFILE_COLORS  = ["#00d4aa", "#7b9fff", "#ffaa44", "#ff6b9a", "#aa88ff"]
print(f"[Config] {ns} snapshots  ·  {nb_sph} spherical bins  ·  "
      f"{nb_proj} projected bins  ·  {MAP_BINS}² map grid")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — DATA LOADING                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def extract_snapshots(work_dir: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="density_snaps_")
    print(f"[extract] temp dir: {tmpdir}")
    tar_files = [fn for fn in os.listdir(work_dir) if fn.endswith(".tar")]
    if not tar_files:
        warnings.warn("No *.tar files found; assuming snapshot files already on disk.")
        return tmpdir
    for fn in tar_files:
        print(f"  Opening {fn} …")
        with tarfile.open(os.path.join(work_dir, fn), "r") as tar:
            targets = [m for m in tar.getmembers()
                       if m.isfile() and ("MW_" in m.name or "M31_" in m.name)]
            for m in targets:
                m.name = os.path.basename(m.name)
                tar.extract(m, path=tmpdir)
            print(f"  Extracted {len(targets)} files.")
    return tmpdir

def load_snapshot_particles(mw_path: str, m31_path: str):
    MW  = CenterOfMass(mw_path,  PTYPE)
    M31 = CenterOfMass(m31_path, PTYPE)
    x  = np.concatenate((MW.x,  M31.x))
    y  = np.concatenate((MW.y,  M31.y))
    z  = np.concatenate((MW.z,  M31.z))
    m_raw = np.concatenate((MW.m, M31.m))
    m_msun = m_raw * MASS_UNIT_MSUN
    origin = np.concatenate((
        np.zeros(len(MW.x),  dtype=np.int8),
        np.ones (len(M31.x), dtype=np.int8),
    ))
    xcom, ycom, zcom = MW.COMdefine(x, y, z, m_raw)
    pos = np.vstack((x - xcom, y - ycom, z - zcom)).T
    sim_time = None
    if hasattr(MW, "time"):
        try: sim_time = float(MW.time.value)
        except Exception:
            try: sim_time = float(MW.time)
            except Exception: pass
    return {"pos": pos, "m_msun": m_msun, "origin": origin, "time": sim_time}

tmpdir = extract_snapshots(".")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — SPHERICAL DENSITY PROFILE ENGINE                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def compute_density_profiles(snap_data: dict) -> dict:
    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    origin = snap_data["origin"]
    r_3d = np.linalg.norm(pos, axis=1)
    R_proj = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    bin_idx_3d   = np.digitize(r_3d,  R_BINS)   - 1
    bin_idx_proj = np.digitize(R_proj, R_PROJ_BINS) - 1
    rho       = np.full(nb_sph,  np.nan)
    rho_mw    = np.full(nb_sph,  np.nan)
    rho_m31   = np.full(nb_sph,  np.nan)
    Sigma     = np.full(nb_proj, np.nan)
    Sigma_mw  = np.full(nb_proj, np.nan)
    Sigma_m31 = np.full(nb_proj, np.nan)
    for b in range(nb_sph):
        mask     = bin_idx_3d == b
        n_in_bin = mask.sum()
        if n_in_bin < MIN_PART_SHELL: continue
        M_bin     = m[mask].sum()
        rho[b]    = M_bin / shell_vols[b]
        mask_mw   = mask & (origin == 0)
        mask_m31  = mask & (origin == 1)
        rho_mw [b] = m[mask_mw ].sum() / shell_vols[b] if mask_mw.sum()  >= 2 else np.nan
        rho_m31[b] = m[mask_m31].sum() / shell_vols[b] if mask_m31.sum() >= 2 else np.nan
    for b in range(nb_proj):
        mask     = bin_idx_proj == b
        n_in_bin = mask.sum()
        if n_in_bin < MIN_PART_SHELL: continue
        M_ring       = m[mask].sum()
        Sigma[b]     = M_ring / ring_areas[b]
        mask_mw      = mask & (origin == 0)
        mask_m31     = mask & (origin == 1)
        Sigma_mw [b] = m[mask_mw ].sum() / ring_areas[b] if mask_mw.sum()  >= 2 else np.nan
        Sigma_m31[b] = m[mask_m31].sum() / ring_areas[b] if mask_m31.sum() >= 2 else np.nan
    with np.errstate(invalid="ignore"):
        f_mix = np.where(
            np.isfinite(rho) & (rho > 0),
            np.minimum(
                np.where(np.isfinite(rho_mw),  rho_mw,  0.0),
                np.where(np.isfinite(rho_m31), rho_m31, 0.0),
            ) / rho,
            np.nan,
        )
    ln_r   = np.log(r_mid_sph)
    ln_rho = np.where(np.isfinite(rho) & (rho > 0), np.log(rho), np.nan)
    finite   = np.isfinite(ln_rho)
    if finite.sum() > 3:
        ln_rho_interp = np.interp(ln_r, ln_r[finite], ln_rho[finite])
        Gamma = np.gradient(ln_rho_interp, ln_r)
        Gamma[~finite] = np.nan
    else:
        Gamma = np.full(nb_sph, np.nan)
    rho0 = next((rho[b] for b in range(nb_sph) if np.isfinite(rho[b])), np.nan)
    M_total = m.sum()
    M_enc_3d = np.array([m[r_3d <= R_BINS[b+1]].sum() for b in range(nb_sph)])
    idx_half = np.searchsorted(M_enc_3d, M_total / 2.0)
    r_half_3d = r_mid_sph[idx_half] if 0 < idx_half < nb_sph else np.nan
    M_enc_proj = np.array([m[R_proj <= R_PROJ_BINS[b+1]].sum() for b in range(nb_proj)])
    idx_half_p = np.searchsorted(M_enc_proj, M_total / 2.0)
    r_half_proj = r_mid_proj[idx_half_p] if 0 < idx_half_p < nb_proj else np.nan
    inner_mask_slope = (r_mid_sph < 5.0)   & np.isfinite(Gamma)
    outer_mask_slope = (r_mid_sph > 50.0)  & np.isfinite(Gamma)
    Gamma_inner = np.nanmean(Gamma[inner_mask_slope]) if inner_mask_slope.any() else np.nan
    Gamma_outer = np.nanmean(Gamma[outer_mask_slope]) if outer_mask_slope.any() else np.nan
    return {
        "rho":         rho,
        "rho_mw":      rho_mw,
        "rho_m31":     rho_m31,
        "f_mix":       f_mix,
        "Gamma":       Gamma,
        "Sigma":       Sigma,
        "Sigma_mw":    Sigma_mw,
        "Sigma_m31":   Sigma_m31,
        "rho0":        rho0,
        "r_half_3d":   r_half_3d,
        "r_half_proj": r_half_proj,
        "Gamma_inner": Gamma_inner,
        "Gamma_outer": Gamma_outer,
        "time":        snap_data["time"],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — 2D PROJECTED SURFACE DENSITY MAPS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def compute_2d_map(snap_data: dict) -> dict:
    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    origin = snap_data["origin"]
    x = pos[:, 0]
    y = pos[:, 1]
    lim    = MAP_EXTENT_KPC
    extent = [-lim, lim, -lim, lim]
    bins_xy = [MAP_BINS, MAP_BINS]
    rng_xy  = [[-lim, lim], [-lim, lim]]
    H_total, xe, ye = np.histogram2d(x, y, bins=bins_xy, range=rng_xy,
                                     weights=m)
    mw_mask  = origin == 0
    m31_mask = origin == 1
    H_mw,  _, _ = np.histogram2d(x[mw_mask],  y[mw_mask],  bins=bins_xy,
                                  range=rng_xy, weights=m[mw_mask])
    H_m31, _, _ = np.histogram2d(x[m31_mask], y[m31_mask], bins=bins_xy,
                                  range=rng_xy, weights=m[m31_mask])
    pixel_size = 2.0 * lim / MAP_BINS
    pixel_area = pixel_size**2
    Sigma_2d      = np.where(H_total > 0, H_total  / pixel_area, np.nan)
    Sigma_2d_mw   = np.where(H_mw   > 0, H_mw   / pixel_area, np.nan)
    Sigma_2d_m31  = np.where(H_m31  > 0, H_m31  / pixel_area, np.nan)
    return {
        "Sigma_2d":     Sigma_2d,
        "Sigma_2d_mw":  Sigma_2d_mw,
        "Sigma_2d_m31": Sigma_2d_m31,
        "extent":       extent,
        "pixel_area":   pixel_area,
    }
