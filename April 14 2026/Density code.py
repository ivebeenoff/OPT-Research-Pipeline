"""

# MW–M31 MERGER DENSITY ANALYSIS PIPELINE

Author  : Abhinav Vatsa

## Overview

This is the density companion to kinematic_profiles_pipeline.py.  Where that
script asks “how fast are the particles moving?”, this one asks “where are
they, and how much mass is there?”  Together the two pipelines give a complete
picture of the MW–M31 merger in both configuration space and velocity space.

The density pipeline computes, for every snapshot:

Spherical profiles (function of r)
──────────────────────────────────
• ρ(r)       — 3D mass density profile (mass-weighted, radial shells)
• Σ(R)       — projected surface mass density (cylinder columns)
• Γ(r)       — local logarithmic slope  d ln ρ / d ln r
• ρ_MW(r)    — density of MW-origin particles only
• ρ_M31(r)   — density of M31-origin particles only
• f_mix(r)   — local mass-mixing fraction at each radius

2D projected maps (x–y plane)
──────────────────────────────
• Σ(x, y)    — 2D surface density map per snapshot
• Σ_MW(x,y)  — MW-only projected density
• Σ_M31(x,y) — M31-only projected density

NFW / Hernquist / Einasto fitting
───────────────────────────────────
• NFW ρ(r) direct density fit (vs. the enclosed-mass fit in kinematics)
• Einasto profile fit (one additional free parameter: shape index α)
• Hernquist profile fit (analytic, useful for comparison)
• Residual map  Δ = (ρ_meas − ρ_model) / ρ_model

Derived structural quantities
──────────────────────────────
• r_half     — projected half-mass radius (from Σ(R))
• r_eff_3D   — 3D half-mass radius (from ρ(r))
• ρ_0        — central density (innermost valid bin)
• Γ_inner    — inner slope (r < 5 kpc)
• Γ_outer    — outer slope (r > 50 kpc)

Time-series (one scalar per snapshot) for all of the above.

## Figures produced

§21  density_profiles_grid.png         — ρ(r) and Σ(R) at 5 epochs
§22  density_slope_heatmap.png         — Γ(r, t) logarithmic slope heatmap
§23  density_central_evolution.png     — ρ_0 and r_half vs. time
§24  density_2d_maps_grid.png          — 2D Σ maps at 5 epochs (3×5 panel)
§25  density_mw_m31_comparison.png     — ρ_MW vs. ρ_M31 at 5 epochs
§26  density_mixing_fraction.png       — f_mix(r, t) heatmap
§27  density_nfw_fit.png               — NFW / Einasto / Hernquist fits at 5 epochs
§28  density_fit_residuals.png         — model residual heatmap in (r, t)
§29  density_halfmass_evolution.png    — r_half, r_eff_3D, ρ_0 vs. time
§30  density_animation.mp4             — ρ(r) profile animation over all snaps
§31  density_2d_animation.mp4          — 2D surface density map animation
§32  density_summary_panel.png         — master 8-panel summary figure

## Units

Positions   : kpc
Masses      : M_sun  (converted from snapshot units × 10^10 M_sun)
Density ρ   : M_sun kpc^{-3}
Surface Σ   : M_sun kpc^{-2}

## Dependencies

numpy, matplotlib, scipy
ReadFile, CenterOfMass2  (project-local)

## Design notes

• All per-snapshot arrays are pre-allocated as NaN-filled so that missing
or failed snapshots appear as gaps in plots rather than zeros.
• Density profiles use the SAME R_BINS as the kinematics pipeline so the
two scripts’ outputs can be directly compared or combined.
• The 2D maps use a separate, coarser grid (MAP_BINS × MAP_BINS) for speed.
• Every figure saves to OUT_DIR and inherits the global rcParams dark theme.
• The cleanup block at §33 removes the tmp directory AFTER all sections
(unlike the kinematics pipeline, which cleaned up mid-run and caused §17
to silently skip on a cold re-run).

"""

# ── Standard library ──────────────────────────────────────────────────────────

import os
import tarfile
import shutil
import tempfile
import warnings
import time

# ── Third-party ───────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# ── Project-local ─────────────────────────────────────────────────────────────

from ReadFile import Read
from CenterOfMass2 import CenterOfMass

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 20 — CONFIGURATION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# All user-facing parameters in one place.  Variable names are UPPER_SNAKE_CASE
# to distinguish them from local function variables throughout the file.

# ── Snapshot range ─────────────────────────────────────────────────────────────
START_SNAP = 0
END_SNAP   = 800

# ── Particle type ──────────────────────────────────────────────────────────────
# 1 = dark matter halo (use for large-scale density)
# 2 = disk             (use for baryonic / stellar density comparison)
PTYPE = 1

# ── Radial bins for spherical profiles ────────────────────────────────────────
# Log-spaced: fine at the centre where ρ changes steeply, coarse at large r.
# 40 edges → 39 bins.  Must match or be a refinement of the kinematics R_BINS
# if cross-pipeline comparison is desired; kept separate here for flexibility.
R_BINS = np.logspace(-1, np.log10(400.0), 40)   # kpc, 39 bins

# Minimum particles per spherical shell before computing density.
MIN_PART_SHELL = 20

# ── 2D map parameters ─────────────────────────────────────────────────────────
# Grid side length [kpc] and number of pixels per side.
# A 256×256 grid at ±400 kpc gives a pixel scale of 3.1 kpc — adequate for
# halo-scale morphology.  Increase MAP_BINS to 512 for publication maps (slower).
MAP_EXTENT_KPC = 400.0
MAP_BINS       = 256

# Gaussian smoothing kernel applied to 2D maps before display [pixels].
# σ = 2 pixels ≈ 6 kpc — suppresses shot noise without blurring real structure.
MAP_SMOOTH_SIGMA = 2.0

# ── Projected (2D cylinder) surface density bins ──────────────────────────────
# Projected radius R = sqrt(x² + y²), log-spaced from 0.5 to 400 kpc.
R_PROJ_BINS = np.logspace(np.log10(0.5), np.log10(400.0), 35)  # 34 bins

# ── Profile fitting ────────────────────────────────────────────────────────────
# Fitting radial range [kpc].  Below R_FIT_MIN the profile is noisy; above
# R_FIT_MAX it is dominated by the other galaxy or the tidal halo.
R_FIT_MIN_KPC = 1.0
R_FIT_MAX_KPC = 200.0

# ── Temporal subsampling for expensive operations ──────────────────────────────
# Every STEP_MAPS-th snapshot is used for 2D map generation and profile fitting.
# STEP_MAPS=8 → ~100 snapshots; each 256×256 map takes ~0.5 s → ~50 s total.
STEP_MAPS  = 8

# Every STEP_FIT-th snapshot is fitted with NFW/Einasto/Hernquist models.
# curve_fit is cheap (~0.01 s) but we have three models × 800 snaps.
STEP_FIT   = 4

# ── Animation ─────────────────────────────────────────────────────────────────
ANIM_FPS     = 20
ANIM_DPI     = 100
ANIM_BITRATE = 2000
ANIM_STEP    = 4    # render every Nth snapshot

# ── Physical constants ─────────────────────────────────────────────────────────
G_KPC_KMS2_MSUN = 4.30091e-6    # kpc (km/s)² M_sun⁻¹
MASS_UNIT_MSUN  = 1.0e10        # snapshot mass unit → M_sun

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Matplotlib global dark theme ──────────────────────────────────────────────
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

# ── Derived geometry ──────────────────────────────────────────────────────────
SNAPSHOTS    = np.arange(START_SNAP, END_SNAP + 1)
ns           = len(SNAPSHOTS)
nb_sph       = len(R_BINS) - 1           # spherical bins
nb_proj      = len(R_PROJ_BINS) - 1      # projected bins
r_mid_sph    = 0.5 * (R_BINS[:-1]   + R_BINS[1:])
r_mid_proj   = 0.5 * (R_PROJ_BINS[:-1] + R_PROJ_BINS[1:])

# Shell volumes for converting particle count / mass → density.
# V_shell = (4π/3)(r_outer³ − r_inner³)
shell_vols   = (4.0 / 3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)  # kpc³

# Annular areas for surface density (projected ring area).
# A_ring = π (R_outer² − R_inner²)
ring_areas   = np.pi * (R_PROJ_BINS[1:]**2 - R_PROJ_BINS[:-1]**2)       # kpc²

# Five representative snapshot indices for profile-overlay figures.
PROFILE_FRACS   = [0.0, 0.2, 0.4, 0.65, 1.0]
PROFILE_INDICES = [int(f * (ns - 1)) for f in PROFILE_FRACS]
PROFILE_LABELS  = [f"Snap {SNAPSHOTS[k]}" for k in PROFILE_INDICES]
PROFILE_COLORS  = ["#00d4aa", "#7b9fff", "#ffaa44", "#ff6b9a", "#aa88ff"]

print(f"[Config] {ns} snapshots  ·  {nb_sph} spherical bins  ·  "
      f"{nb_proj} projected bins  ·  {MAP_BINS}² map grid")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 21 — DATA LOADING                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_snapshots(work_dir: str) -> str:
    """
    Extract MW_NNN.txt and M31_NNN.txt from all *.tar files in work_dir into a
    fresh temp directory.  Returns the temp directory path.
    
    Notes
    -----
    Uses member.name = os.path.basename(member.name) to strip subdirectory
    prefixes from archive members, so all files land flat in the temp dir
    regardless of how the archive was created.
    """
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
    """
    Load one snapshot and return particle arrays centred on the joint COM.
    
    Returns
    -------
    dict with keys:
        pos      : (N, 3) float64  — positions  [kpc]  in COM frame
        m_msun   : (N,)   float64  — masses     [M_sun]
        origin   : (N,)   int8     — 0 = MW, 1 = M31
        time     : float           — simulation time [Gyr] or None
    """
    MW  = CenterOfMass(mw_path,  PTYPE)
    M31 = CenterOfMass(m31_path, PTYPE)

    # ── Concatenate all particles from both galaxies ───────────────────────────
    x  = np.concatenate((MW.x,  M31.x))
    y  = np.concatenate((MW.y,  M31.y))
    z  = np.concatenate((MW.z,  M31.z))
    m_raw = np.concatenate((MW.m, M31.m))   # in snapshot units (10^10 M_sun)
    m_msun = m_raw * MASS_UNIT_MSUN         # convert to M_sun

    # ── Origin tag: 0 = MW particle, 1 = M31 particle ─────────────────────────
    # This lets us split the density by galaxy of origin anywhere downstream
    # without re-reading the files.
    origin = np.concatenate((
        np.zeros(len(MW.x),  dtype=np.int8),
        np.ones (len(M31.x), dtype=np.int8),
    ))

    # ── Joint centre of mass (position) ───────────────────────────────────────
    # COMdefine uses the iterative shrinking-sphere algorithm; it only needs
    # relative mass ratios so we pass m_raw (units cancel).
    xcom, ycom, zcom = MW.COMdefine(x, y, z, m_raw)

    # ── Shift to COM frame ────────────────────────────────────────────────────
    pos = np.vstack((x - xcom, y - ycom, z - zcom)).T   # (N, 3)  [kpc]

    # ── Simulation time ────────────────────────────────────────────────────────
    sim_time = None
    if hasattr(MW, "time"):
        try:
            sim_time = float(MW.time.value)
        except Exception:
            try:
                sim_time = float(MW.time)
            except Exception:
                pass

    return {"pos": pos, "m_msun": m_msun, "origin": origin, "time": sim_time}


# ── Extract snapshot files ─────────────────────────────────────────────────────
tmpdir = extract_snapshots(".")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 22 — SPHERICAL DENSITY PROFILE ENGINE                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# We compute the mass density ρ(r) by:
# 1. Calculating the 3D distance r = |pos| for every particle.
# 2. Binning particle masses into spherical shells using np.digitize.
# 3. Dividing each shell’s total mass by its volume (pre-computed in shell_vols).
# 
# Why mass-weighted rather than particle-count density?
# ─────────────────────────────────────────────────────
# N-body simulations use equal-mass particles in each component, but MW and M31
# particles may have different masses.  Using particle counts would give an
# unphysical density when comparing the two galaxies.  Mass-weighting is always
# physically correct.
# 
# The logarithmic slope Γ = d ln ρ / d ln r
# ────────────────────────────────────────────
# is computed via numerical differentiation of ln ρ with respect to ln r.
# Key reference values:
# Γ = −1   →  NFW inner cusp  (ρ ∝ r^{−1})
# Γ = −2   →  isothermal sphere / NFW intermediate
# Γ = −3   →  NFW outer fall-off  (ρ ∝ r^{−3})
# Γ > −1   →  flatter than NFW core (core-like)
# Γ ≫ −3   →  steeper than NFW outer (e.g., post-merger truncation)

def compute_density_profiles(snap_data: dict) -> dict:
    """
    Compute spherical and projected density profiles for one snapshot.
    
    Parameters
    ----------
    snap_data : dict  — output of load_snapshot_particles()

    Returns
    -------
    dict with keys:
        rho          : (nb_sph,)   — 3D density   [M_sun kpc^{-3}]
        rho_mw       : (nb_sph,)   — MW-only density
        rho_m31      : (nb_sph,)   — M31-only density
        f_mix        : (nb_sph,)   — mass-mixing fraction ρ_minor / ρ_total
        Gamma        : (nb_sph,)   — logarithmic slope d ln ρ / d ln r
        Sigma        : (nb_proj,)  — projected surface density [M_sun kpc^{-2}]
        Sigma_mw     : (nb_proj,)  — MW-only surface density
        Sigma_m31    : (nb_proj,)  — M31-only surface density
        rho0         : float       — central density (innermost finite bin)
        r_half_3d    : float       — 3D half-mass radius [kpc]
        r_half_proj  : float       — projected half-mass radius [kpc]
        Gamma_inner  : float       — mean Γ for r < 5 kpc
        Gamma_outer  : float       — mean Γ for r > 50 kpc
        time         : float | None
    """
    pos    = snap_data["pos"]       # (N, 3)  [kpc]
    m      = snap_data["m_msun"]    # (N,)    [M_sun]
    origin = snap_data["origin"]    # (N,)    0=MW, 1=M31

    # ── 3D radii ──────────────────────────────────────────────────────────────
    r_3d = np.linalg.norm(pos, axis=1)          # (N,)  [kpc]

    # ── Projected radius R = sqrt(x² + y²) in the x–y plane ──────────────────
    R_proj = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)   # (N,)

    # ── Spherical shell binning ────────────────────────────────────────────────
    # np.digitize returns 1-based indices; subtract 1 for 0-based.
    bin_idx_3d   = np.digitize(r_3d,  R_BINS)   - 1   # (N,)
    bin_idx_proj = np.digitize(R_proj, R_PROJ_BINS) - 1  # (N,)

    # ── Initialise output arrays ───────────────────────────────────────────────
    rho       = np.full(nb_sph,  np.nan)
    rho_mw    = np.full(nb_sph,  np.nan)
    rho_m31   = np.full(nb_sph,  np.nan)
    Sigma     = np.full(nb_proj, np.nan)
    Sigma_mw  = np.full(nb_proj, np.nan)
    Sigma_m31 = np.full(nb_proj, np.nan)

    # ── Fill spherical density bins ────────────────────────────────────────────
    for b in range(nb_sph):
        mask     = bin_idx_3d == b
        n_in_bin = mask.sum()
        if n_in_bin < MIN_PART_SHELL:
            continue   # leave as NaN

        M_bin     = m[mask].sum()                            # total mass in shell
        rho[b]    = M_bin / shell_vols[b]                    # [M_sun kpc^{-3}]

        mask_mw   = mask & (origin == 0)
        mask_m31  = mask & (origin == 1)
        rho_mw [b] = m[mask_mw ].sum() / shell_vols[b] if mask_mw.sum()  >= 2 else np.nan
        rho_m31[b] = m[mask_m31].sum() / shell_vols[b] if mask_m31.sum() >= 2 else np.nan

    # ── Projected surface density ──────────────────────────────────────────────
    for b in range(nb_proj):
        mask     = bin_idx_proj == b
        n_in_bin = mask.sum()
        if n_in_bin < MIN_PART_SHELL:
            continue

        M_ring       = m[mask].sum()
        Sigma[b]     = M_ring / ring_areas[b]                # [M_sun kpc^{-2}]
        mask_mw      = mask & (origin == 0)
        mask_m31     = mask & (origin == 1)
        Sigma_mw [b] = m[mask_mw ].sum() / ring_areas[b] if mask_mw.sum()  >= 2 else np.nan
        Sigma_m31[b] = m[mask_m31].sum() / ring_areas[b] if mask_m31.sum() >= 2 else np.nan

    # ── Mixing fraction f_mix ─────────────────────────────────────────────────
    # Defined as the mass fraction contributed by the *minority* population at
    # each radius.  Ranges from 0 (purely one galaxy) to 0.5 (equal mix).
    # We also store the sign: positive = M31 dominates, negative = MW dominates.
    with np.errstate(invalid="ignore"):
        f_mix = np.where(
            np.isfinite(rho) & (rho > 0),
            np.minimum(
                np.where(np.isfinite(rho_mw),  rho_mw,  0.0),
                np.where(np.isfinite(rho_m31), rho_m31, 0.0),
            ) / rho,
            np.nan,
        )

    # ── Logarithmic density slope Γ = d ln ρ / d ln r ─────────────────────────
    # Use numpy.gradient on the log-transformed arrays.  The coordinate
    # array ln(r_mid) is non-uniform; np.gradient handles this via
    # second-order central differences weighted by local step size.
    ln_r   = np.log(r_mid_sph)
    ln_rho = np.where(np.isfinite(rho) & (rho > 0), np.log(rho), np.nan)

    # Replace NaN with linear interpolation before differentiating to avoid
    # propagating NaN across multiple bins via the finite-difference stencil.
    finite   = np.isfinite(ln_rho)
    if finite.sum() > 3:
        ln_rho_interp = np.interp(ln_r, ln_r[finite], ln_rho[finite])
        Gamma = np.gradient(ln_rho_interp, ln_r)
        Gamma[~finite] = np.nan   # restore NaN where data was missing
    else:
        Gamma = np.full(nb_sph, np.nan)

    # ── Derived scalars ────────────────────────────────────────────────────────
    # Central density: value in the innermost finite spherical bin.
    rho0 = next((rho[b] for b in range(nb_sph) if np.isfinite(rho[b])), np.nan)

    # 3D half-mass radius: smallest r such that M(<r) ≥ M_total / 2.
    M_total = m.sum()
    M_enc_3d = np.array([m[r_3d <= R_BINS[b+1]].sum() for b in range(nb_sph)])
    idx_half = np.searchsorted(M_enc_3d, M_total / 2.0)
    r_half_3d = r_mid_sph[idx_half] if 0 < idx_half < nb_sph else np.nan

    # Projected half-mass radius: from cumulative Σ(R).
    M_enc_proj = np.array([m[R_proj <= R_PROJ_BINS[b+1]].sum() for b in range(nb_proj)])
    idx_half_p = np.searchsorted(M_enc_proj, M_total / 2.0)
    r_half_proj = r_mid_proj[idx_half_p] if 0 < idx_half_p < nb_proj else np.nan

    # Inner and outer logarithmic slopes.
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


def compute_2d_map(snap_data: dict) -> dict:
    """
    Compute a 2D projected surface density map Σ(x, y) by binning particle
    positions into a MAP_BINS × MAP_BINS grid on the x–y plane.
    
    Returns
    -------
    dict with keys:
        Sigma_2d      : (MAP_BINS, MAP_BINS)  — total Σ in M_sun kpc^{-2}
        Sigma_2d_mw   : (MAP_BINS, MAP_BINS)  — MW-only Σ
        Sigma_2d_m31  : (MAP_BINS, MAP_BINS)  — M31-only Σ
        extent        : [x_min, x_max, y_min, y_max]  for imshow
        pixel_area    : float  — pixel area in kpc²

    Notes on the pixel scale
    ─────────────────────────
    With MAP_EXTENT_KPC = 400 and MAP_BINS = 256:
        pixel_size = 800 / 256 ≈ 3.1 kpc
        pixel_area = 3.1² ≈ 9.6 kpc²
    All cells outside the particle distribution have zero mass → log10 would
    give −inf; we replace those with NaN before plotting so they render as
    the background colour.
    """
    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    origin = snap_data["origin"]

    x = pos[:, 0]
    y = pos[:, 1]

    lim    = MAP_EXTENT_KPC
    extent = [-lim, lim, -lim, lim]

    # 2D histogram: each bin sums the masses of all particles in that pixel.
    bins_xy = [MAP_BINS, MAP_BINS]
    rng_xy  = [[-lim, lim], [-lim, lim]]

    # Total Σ map.
    H_total, xe, ye = np.histogram2d(x, y, bins=bins_xy, range=rng_xy,
                                     weights=m)

    # MW-only and M31-only maps.
    mw_mask  = origin == 0
    m31_mask = origin == 1
    H_mw,  _, _ = np.histogram2d(x[mw_mask],  y[mw_mask],  bins=bins_xy,
                                  range=rng_xy, weights=m[mw_mask])
    H_m31, _, _ = np.histogram2d(x[m31_mask], y[m31_mask], bins=bins_xy,
                                  range=rng_xy, weights=m[m31_mask])

    # Convert from total mass per pixel to surface density [M_sun kpc^{-2}].
    pixel_size = 2.0 * lim / MAP_BINS    # kpc per pixel
    pixel_area = pixel_size**2            # kpc²

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 23 — MAIN SNAPSHOT LOOP                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# This loop pre-computes the full suite of density profiles for all 801
# snapshots and caches the results in NaN-initialised numpy arrays.  Expensive
# per-snapshot operations (2D maps, profile fits) are computed on a sparser
# temporal grid controlled by STEP_MAPS and STEP_FIT to keep wall-time
# manageable.

print("\n" + "="*70)
print("  SECTION 23 · Main Snapshot Loop")
print("="*70)

# ── Pre-allocate time-series arrays ───────────────────────────────────────────
rho_ts        = np.full((ns, nb_sph),  np.nan)    # ρ(r, t)
rho_mw_ts     = np.full((ns, nb_sph),  np.nan)    # ρ_MW(r, t)
rho_m31_ts    = np.full((ns, nb_sph),  np.nan)    # ρ_M31(r, t)
f_mix_ts      = np.full((ns, nb_sph),  np.nan)    # f_mix(r, t)
Gamma_ts      = np.full((ns, nb_sph),  np.nan)    # Γ(r, t)
Sigma_ts      = np.full((ns, nb_proj), np.nan)    # Σ(R, t)
Sigma_mw_ts   = np.full((ns, nb_proj), np.nan)
Sigma_m31_ts  = np.full((ns, nb_proj), np.nan)

# Scalar time-series (one number per snapshot).
rho0_arr        = np.full(ns, np.nan)   # central density
r_half_3d_arr   = np.full(ns, np.nan)   # 3D half-mass radius
r_half_proj_arr = np.full(ns, np.nan)   # projected half-mass radius
Gamma_inner_arr = np.full(ns, np.nan)   # mean inner slope
Gamma_outer_arr = np.full(ns, np.nan)   # mean outer slope
time_arr        = np.full(ns, np.nan)   # simulation time or snap index

# 2D maps — stored only for the subset of snapshots at STEP_MAPS intervals.
map_snap_nums = SNAPSHOTS[::STEP_MAPS]
n_maps        = len(map_snap_nums)
map_snap_idx  = {snap: i for i, snap in enumerate(map_snap_nums)}
maps_3d   = np.full((n_maps, MAP_BINS, MAP_BINS), np.nan)
maps_mw   = np.full((n_maps, MAP_BINS, MAP_BINS), np.nan)
maps_m31  = np.full((n_maps, MAP_BINS, MAP_BINS), np.nan)

# ── Loop ──────────────────────────────────────────────────────────────────────
t_loop_start = time.perf_counter()

for i, snap_num in enumerate(SNAPSHOTS):
    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue    # silently skip missing snapshots

    t_snap = time.perf_counter()

    try:
        snap_data = load_snapshot_particles(mw_file, m31_file)
        prof      = compute_density_profiles(snap_data)
    except Exception as exc:
        print(f"  [ERROR] snap {snap_num}: {exc}")
        continue

    # ── Store spherical profiles ───────────────────────────────────────────────
    rho_ts      [i, :] = prof["rho"]
    rho_mw_ts   [i, :] = prof["rho_mw"]
    rho_m31_ts  [i, :] = prof["rho_m31"]
    f_mix_ts    [i, :] = prof["f_mix"]
    Gamma_ts    [i, :] = prof["Gamma"]
    Sigma_ts    [i, :] = prof["Sigma"]
    Sigma_mw_ts [i, :] = prof["Sigma_mw"]
    Sigma_m31_ts[i, :] = prof["Sigma_m31"]

    # ── Store scalars ──────────────────────────────────────────────────────────
    rho0_arr       [i] = prof["rho0"]
    r_half_3d_arr  [i] = prof["r_half_3d"]
    r_half_proj_arr[i] = prof["r_half_proj"]
    Gamma_inner_arr[i] = prof["Gamma_inner"]
    Gamma_outer_arr[i] = prof["Gamma_outer"]
    time_arr       [i] = prof["time"] if prof["time"] is not None else float(snap_num)

    # ── 2D maps (every STEP_MAPS snapshots) ───────────────────────────────────
    if snap_num in map_snap_idx:
        mi = map_snap_idx[snap_num]
        try:
            maps = compute_2d_map(snap_data)
            maps_3d [mi] = maps["Sigma_2d"]
            maps_mw [mi] = maps["Sigma_2d_mw"]
            maps_m31[mi] = maps["Sigma_2d_m31"]
        except Exception as exc:
            print(f"  [WARN] 2D map failed for snap {snap_num}: {exc}")

    dt = time.perf_counter() - t_snap
    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t_loop_start
        print(f"  snap {snap_num:04d}  ({dt:.2f}s)  "
              f"ρ_0={prof['rho0']:.2e} M⊙/kpc³  "
              f"r_half={prof['r_half_3d']:.1f} kpc  "
              f"[{elapsed:.0f}s elapsed]")


print(f"\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total")

# ── Time axis ─────────────────────────────────────────────────────────────────
t_valid     = time_arr[np.isfinite(time_arr)]
time_is_gyr = t_valid.size > 0 and t_valid.min() > 0.05
time_label  = "Time [Gyr]" if time_is_gyr else "Snapshot index"

# Map snap-subset time axis.
time_maps = np.array([time_arr[np.where(SNAPSHOTS == s)[0][0]]
                      if len(np.where(SNAPSHOTS == s)[0]) > 0 else np.nan
                      for s in map_snap_nums])

# Map extent for imshow (same for all maps).
MAP_EXTENT = [-MAP_EXTENT_KPC, MAP_EXTENT_KPC,
              -MAP_EXTENT_KPC, MAP_EXTENT_KPC]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 24 — FIGURE 1: ρ(r) AND Σ(R) PROFILE GRID AT FIVE EPOCHS            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Purpose
# —––
# Show the radial density structure at five representative merger stages.  The
# two panels (3D spherical ρ and 2D projected Σ) are placed side-by-side so
# readers can immediately see how projection affects the apparent structure.
# 
# Expected features:
# • ρ(r) should steepen at large r after the merger as particles are
# stripped to a diffuse outer halo.
# • The central ρ_0 will spike near first pericentre (compression) then
# fall slightly post-merger as violent relaxation puffs the core.
# • Σ(R) will show a steeper outer fall-off than ρ(r) because the
# line-of-sight integral over the 3D profile is not a power law even
# if ρ itself is.

print("\n[Fig 1]  ρ(r) and Σ(R) profile grid …")

fig24, (ax_rho, ax_sigma) = plt.subplots(
    1, 2, figsize=(13, 6), facecolor="#0d0d18",
    gridspec_kw={"wspace": 0.32},
)

for ax in (ax_rho, ax_sigma):
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log")
    ax.set_yscale("log")

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    # ── ρ(r) ──────────────────────────────────────────────────────────────────
    rho_row = rho_ts[k_idx, :]
    valid   = np.isfinite(rho_row) & (rho_row > 0)
    if valid.any():
        ax_rho.plot(r_mid_sph[valid], rho_row[valid],
                    color=color, lw=2.0, label=label)

    # ── Σ(R) ──────────────────────────────────────────────────────────────────
    sig_row = Sigma_ts[k_idx, :]
    valid_s = np.isfinite(sig_row) & (sig_row > 0)
    if valid_s.any():
        ax_sigma.plot(r_mid_proj[valid_s], sig_row[valid_s],
                      color=color, lw=2.0, label=label)

# ── Reference slope lines ──────────────────────────────────────────────────────
# Overlay fiducial power-law slopes for quick visual slope comparison.
r_ref  = np.array([1.0, 100.0])
rho_ref_anchor = np.nanmedian(rho_ts[PROFILE_INDICES[2], :][np.isfinite(rho_ts[PROFILE_INDICES[2], :])])

for slope, ls, lbl in [(-1, ":", r"$\rho \propto r^{-1}$ (NFW inner)"),
                       (-2, "--", r"$\rho \propto r^{-2}$ (isothermal)"),
                       (-3, "-.", r"$\rho \propto r^{-3}$ (NFW outer)")]:
    if rho_ref_anchor > 0:
        rho_ref_line = rho_ref_anchor * (r_ref / 10.0)**slope
        ax_rho.plot(r_ref, rho_ref_line, color="#555577", ls=ls, lw=0.9, label=lbl)

ax_rho.set_xlabel("r [kpc]", fontsize=10)
ax_rho.set_ylabel(r"$\rho(r)$  [M$_\odot$ kpc$^{-3}$]", fontsize=10)
ax_rho.set_title(r"3D Spherical Density  $\rho(r)$", fontsize=11)
ax_rho.legend(fontsize=7, ncol=2)
ax_rho.set_xlim(R_BINS[0], R_BINS[-1])

ax_sigma.set_xlabel("R [kpc]", fontsize=10)
ax_sigma.set_ylabel(r"$\Sigma(R)$  [M$_\odot$ kpc$^{-2}$]", fontsize=10)
ax_sigma.set_title(r"Projected Surface Density  $\Sigma(R)$", fontsize=11)
ax_sigma.legend(fontsize=7)
ax_sigma.set_xlim(R_PROJ_BINS[0], R_PROJ_BINS[-1])

fig24.suptitle("MW–M31 Merger  ·  Density Profiles at Key Epochs", fontsize=12)
fig24.savefig(os.path.join(OUT_DIR, "density_profiles_grid.png"),
              dpi=300, bbox_inches="tight", facecolor=fig24.get_facecolor())
plt.close(fig24)
print("  Saved: density_profiles_grid.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 25 — FIGURE 2: LOGARITHMIC SLOPE Γ(r, t) HEATMAP                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Purpose
# —––
# A heatmap of Γ(r, t) = d ln ρ / d ln r is arguably the single most
# diagnostic density figure in this pipeline.  It shows:
# 
# • Where the density profile is NFW-like (Γ ≈ −1 at small r, −3 at large r)
# • How the inner cusp/core evolves — merger-driven core formation would show
# Γ → 0 (flattening) at small r after pericentre
# • Where tidal stripping is active — outer Γ steepens (becomes more negative)
# as particles are stripped beyond r_tidal and the profile is truncated
# 
# The diverging colormap (bwr: blue = steep/negative, red = shallow/positive)
# makes it easy to see the transition between inner and outer profile regimes.

print("[Fig 2]  Γ(r, t) heatmap …")

# Clip to a physically meaningful range for display.
Gamma_plot = np.clip(Gamma_ts, -5.0, 1.0)

t_min = np.nanmin(time_arr)
t_max = np.nanmax(time_arr)

fig25, (ax25a, ax25b) = plt.subplots(
    1, 2, figsize=(14, 6), facecolor="#0d0d18",
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
)
for ax in (ax25a, ax25b):
    ax.set_facecolor("#0d0d18")

# Left: heatmap in (t, r)
im25 = ax25a.imshow(
    Gamma_plot.T,
    aspect="auto", origin="lower",
    extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
    cmap="bwr",
    vmin=-4.0, vmax=0.5,
)
ax25a.set_yscale("log")
ax25a.set_xlabel(time_label, fontsize=10)
ax25a.set_ylabel("r [kpc]", fontsize=10)
ax25a.set_title(r"Logarithmic Density Slope  $\Gamma(r, t) = d\ln\rho / d\ln r$",
                fontsize=11)
cb25 = fig25.colorbar(im25, ax=ax25a, pad=0.01)
cb25.set_label(r"$\Gamma$", fontsize=9)

# Annotate reference slope values on the colorbar.
for val, lbl in [(-1, "NFW inner"), (-2, "isoth."), (-3, "NFW outer")]:
    cb25.ax.axhline(val, color="white", lw=0.7, ls="--", alpha=0.6)
    cb25.ax.text(2.5, val, lbl, color="white", fontsize=6, va="center")

# Right: time-averaged Γ(r) profile
Gamma_mean = np.nanmean(Gamma_ts, axis=0)
valid_G    = np.isfinite(Gamma_mean)
ax25b.plot(Gamma_mean[valid_G], r_mid_sph[valid_G],
           color="#e8673a", lw=2.0)
ax25b.set_xscale("linear")
ax25b.set_yscale("log")
ax25b.set_xlabel(r"$\langle\Gamma\rangle_t$", fontsize=10)
ax25b.set_ylim(R_BINS[0], R_BINS[-1])
ax25b.set_title("Time-avg.", fontsize=10)
ax25b.tick_params(labelleft=False)
for ref_val in [-1, -2, -3]:
    ax25b.axvline(ref_val, color="#555577", lw=0.7, ls="--", alpha=0.6)

fig25.suptitle("Density Slope Evolution", fontsize=12)
fig25.savefig(os.path.join(OUT_DIR, "density_slope_heatmap.png"),
              dpi=300, bbox_inches="tight", facecolor=fig25.get_facecolor())
plt.close(fig25)
print("  Saved: density_slope_heatmap.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 26 — FIGURE 3: CENTRAL DENSITY AND HALF-MASS RADIUS EVOLUTION       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Purpose
# —––
# ρ_0(t) and r_half(t) are the two most compact structural descriptors of the
# halo.  Plotting them together reveals the competition between:
# 
# Compression   →  ρ_0 ↑  and  r_half ↓  (mass concentrates inward)
# Expansion     →  ρ_0 ↓  and  r_half ↑  (violent relaxation after merger)
# 
# The outer slope Γ_outer tracks whether the halo is being tidally truncated
# (Γ → −4 or steeper) or puffed outward (Γ → −2 or shallower).
# 
# Expected behaviour for a first-infall merger:
# • ρ_0 spikes at pericentre (compression by the tidal field of M31)
# • r_half shrinks briefly then recovers as the merged remnant virialises
# • Γ_outer steepens permanently as tidal debris is stripped to large radii

print("[Fig 3]  Central density and half-mass radius …")

fig26, axes26 = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d0d18",
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
axes26 = axes26.flatten()

for ax in axes26:
    ax.set_facecolor("#0d0d18")

# (a) Central density ρ_0 vs. time.
axes26[0].semilogy(time_arr, rho0_arr, color="#e8673a", lw=1.8,
                   label=r"$\rho_0$  (innermost bin)")
axes26[0].set_ylabel(r"$\rho_0$ [M$_\odot$ kpc$^{-3}$]", fontsize=9)
axes26[0].set_title("Central Density", fontsize=10)
axes26[0].legend(fontsize=8)

# (b) 3D and projected half-mass radii vs. time.
axes26[1].plot(time_arr, r_half_3d_arr,   color="#4a8fff", lw=1.8,
               label=r"$r_{1/2, 3D}$")
axes26[1].plot(time_arr, r_half_proj_arr, color="#00d4aa", lw=1.8, ls="--",
               label=r"$R_{1/2, \rm proj}$")
axes26[1].set_ylabel("Half-mass radius [kpc]", fontsize=9)
axes26[1].set_title("Half-Mass Radii", fontsize=10)
axes26[1].legend(fontsize=8)

# (c) Inner slope Γ_inner vs. time.
axes26[2].plot(time_arr, Gamma_inner_arr, color="#ff9944", lw=1.8,
               label=r"$\langle\Gamma\rangle_{r<5, {\rm kpc}}$")
axes26[2].axhline(-1, color="#555577", lw=0.8, ls="--")
axes26[2].text(t_min * 1.02, -0.9, "NFW inner (−1)", color="#777799", fontsize=7)
axes26[2].set_ylabel(r"Mean inner $\Gamma$", fontsize=9)
axes26[2].set_xlabel(time_label, fontsize=9)
axes26[2].set_title("Inner Slope Evolution", fontsize=10)
axes26[2].legend(fontsize=8)

# (d) Outer slope Γ_outer vs. time.
axes26[3].plot(time_arr, Gamma_outer_arr, color="#aa55ff", lw=1.8,
               label=r"$\langle\Gamma\rangle_{r>50, {\rm kpc}}$")
axes26[3].axhline(-3, color="#555577", lw=0.8, ls="--")
axes26[3].text(t_min * 1.02, -2.9, "NFW outer (−3)", color="#777799", fontsize=7)
axes26[3].set_ylabel(r"Mean outer $\Gamma$", fontsize=9)
axes26[3].set_xlabel(time_label, fontsize=9)
axes26[3].set_title("Outer Slope Evolution", fontsize=10)
axes26[3].legend(fontsize=8)

fig26.suptitle("Structural Parameter Evolution", fontsize=12)
fig26.savefig(os.path.join(OUT_DIR, "density_central_evolution.png"),
              dpi=300, bbox_inches="tight", facecolor=fig26.get_facecolor())
plt.close(fig26)
print("  Saved: density_central_evolution.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 27 — FIGURE 4: 2D SURFACE DENSITY MAPS AT FIVE EPOCHS              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Purpose
# —––
# 2D surface density maps are the closest analogue to what an observer would
# see in a deep optical or infrared image of the merger.  The three rows show:
# Row 1: total Σ(x, y)    — dominated by whichever galaxy is denser there
# Row 2: MW-only Σ_MW     — tracks the MW stellar/DM distribution
# Row 3: M31-only Σ_M31   — tracks M31’s distribution
# 
# The log10 stretch is essential: the dynamic range from the dense nucleus
# (~ 10^9 M_sun kpc^{-2}) to the outer tidal streams (~ 10^4 M_sun kpc^{-2})
# spans 5 orders of magnitude.  A linear stretch would show only the nucleus.
# 
# Gaussian smoothing (MAP_SMOOTH_SIGMA pixels) suppresses Poisson noise in
# low-density outer regions without blurring the dense inner structure.

print("[Fig 4]  2D surface density maps …")

# Select five map snapshots closest to PROFILE_FRACS times.
map_fracs  = [0.0, 0.2, 0.4, 0.65, 1.0]
map_times  = np.nanmin(time_maps) + np.array(map_fracs) * (
    np.nanmax(time_maps) - np.nanmin(time_maps))
map_sel_ii = [np.nanargmin(np.abs(time_maps - mt)) for mt in map_times]

fig27 = plt.figure(figsize=(18, 11), facecolor="#0d0d18")
gs27  = gridspec.GridSpec(3, 5, figure=fig27, hspace=0.08, wspace=0.06,
                          left=0.04, right=0.95, top=0.92, bottom=0.05)

row_labels = ["Total Σ", "MW only", "M31 only"]
cmaps      = ["inferno", "Blues_r", "Reds_r"]

# Colour limits shared across all panels in a given row for consistent scaling.
vmin_row   = [4.0, 4.0, 4.0]    # log10 M_sun kpc^{-2}
vmax_row   = [9.5, 9.5, 9.5]

for col, mi in enumerate(map_sel_ii):
    snap_t = time_maps[mi]
    t_str  = f"{snap_t:.2f} Gyr" if time_is_gyr else f"Snap {map_snap_nums[mi]}"

    for row, (data_store, cmap, vmin, vmax) in enumerate(
        zip([maps_3d, maps_mw, maps_m31], cmaps, vmin_row, vmax_row)
    ):
        ax = fig27.add_subplot(gs27[row, col])
        ax.set_facecolor("#0d0d18")

        raw = data_store[mi]
        # Smooth then take log10.
        smoothed = gaussian_filter(
            np.where(np.isfinite(raw), raw, 0.0), sigma=MAP_SMOOTH_SIGMA
        )
        log_map = np.where(smoothed > 0, np.log10(smoothed), np.nan)

        ax.imshow(
            log_map.T,            # transpose: rows = y, cols = x
            origin="lower",
            extent=MAP_EXTENT,
            aspect="equal",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
        )

        # Column header (time) and row label.
        if row == 0:
            ax.set_title(t_str, fontsize=9, color="#c8c8e8", pad=4)
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=9, color="#c8c8e8")

        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

# Single shared colorbar on the right.
sm = plt.cm.ScalarMappable(cmap="inferno",
                           norm=plt.Normalize(vmin=4.0, vmax=9.5))
sm.set_array([])
cbar_ax = fig27.add_axes([0.96, 0.05, 0.012, 0.87])
fig27.colorbar(sm, cax=cbar_ax,
               label=r"$\log_{10}(\Sigma\ /\ {\rm M_\odot\ kpc^{-2}})$")

fig27.suptitle(r"2D Projected Surface Density  $\Sigma(x, y)$  ·  ±400 kpc",
               fontsize=13, color="#c8c8e8")
fig27.savefig(os.path.join(OUT_DIR, "density_2d_maps_grid.png"),
              dpi=200, bbox_inches="tight", facecolor=fig27.get_facecolor())
plt.close(fig27)
print("  Saved: density_2d_maps_grid.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 28 — FIGURE 5: MW vs. M31 DENSITY COMPARISON                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Purpose
# —––
# Overplotting ρ_MW(r) and ρ_M31(r) at five epochs reveals:
# • Which galaxy dominates the density at each radius and epoch.
# • How quickly the two density profiles merge into a single profile — the
# epoch at which they become indistinguishable is the “kinematic mixing
# radius” as a function of time.
# • Whether one galaxy’s core survives as a distinct density peak while the
# other’s is disrupted (unequal-mass merger signature).
# 
# The ratio panel (bottom) ρ_MW / ρ_M31 directly shows the mass imbalance:
# ratio > 1  →  MW dominates at this radius
# ratio = 1  →  equal contribution (boundary of mixing)
# ratio < 1  →  M31 dominates

print("[Fig 5]  MW vs. M31 density comparison …")

fig28, axes28 = plt.subplots(
    2, len(PROFILE_INDICES), figsize=(15, 8), facecolor="#0d0d18",
    sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1,
                              "wspace": 0.12},
)

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES,
                                                PROFILE_LABELS,
                                                PROFILE_COLORS)):
    ax_top = axes28[0, col]
    ax_bot = axes28[1, col]
    ax_top.set_facecolor("#0d0d18")
    ax_bot.set_facecolor("#0d0d18")
    ax_top.set_xscale("log"); ax_top.set_yscale("log")
    ax_bot.set_xscale("log")

    rho_mw_row  = rho_mw_ts [k_idx, :]
    rho_m31_row = rho_m31_ts[k_idx, :]

    valid_mw  = np.isfinite(rho_mw_row)  & (rho_mw_row  > 0)
    valid_m31 = np.isfinite(rho_m31_row) & (rho_m31_row > 0)

    if valid_mw.any():
        ax_top.plot(r_mid_sph[valid_mw],  rho_mw_row[valid_mw],
                    color="#4a8fff", lw=1.8, label="MW")
    if valid_m31.any():
        ax_top.plot(r_mid_sph[valid_m31], rho_m31_row[valid_m31],
                    color="#ff5fa0", lw=1.8, label="M31")

    ax_top.set_title(label, fontsize=9)
    if col == 0:
        ax_top.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=9)
        ax_bot.set_ylabel(r"$\rho_{\rm MW}/\rho_{\rm M31}$", fontsize=9)
    ax_top.legend(fontsize=7)

    # Ratio panel.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            valid_mw & valid_m31,
            rho_mw_row / rho_m31_row,
            np.nan,
        )
    valid_r = np.isfinite(ratio)
    if valid_r.any():
        ax_bot.plot(r_mid_sph[valid_r], ratio[valid_r], color=color, lw=1.5)
    ax_bot.axhline(1.0, color="#555577", lw=0.8, ls="--")
    ax_bot.set_ylim(0.05, 20)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("r [kpc]", fontsize=9)

fig28.suptitle(r"MW vs. M31 Density Profiles  $\rho_{\rm MW}(r)$ / $\rho_{\rm M31}(r)$",
               fontsize=12)
fig28.savefig(os.path.join(OUT_DIR, "density_mw_m31_comparison.png"),
              dpi=300, bbox_inches="tight", facecolor=fig28.get_facecolor())
plt.close(fig28)
print("  Saved: density_mw_m31_comparison.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 29 — FIGURE 6: MASS-MIXING FRACTION  f_mix(r, t)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Purpose
# —––
# f_mix(r, t) = min(ρ_MW, ρ_M31) / (ρ_MW + ρ_M31) ∈ [0, 0.5]
# 
# tracks how thoroughly the two galaxies’ mass distributions have overlapped
# at each radius and time:
# f_mix = 0    →  only one galaxy contributes (no mixing)
# f_mix = 0.5  →  equal contributions (well mixed)
# 
# The radial gradient of f_mix at fixed time shows the “mixing front” — the
# radius inside which the two halos have interpenetrated.  Plotting f_mix as
# a heatmap in (r, t) reveals how this front moves outward as the merger
# progresses.

print("[Fig 6]  Mass-mixing fraction heatmap …")

fig29, (ax29a, ax29b) = plt.subplots(
    1, 2, figsize=(14, 6), facecolor="#0d0d18",
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
)
for ax in (ax29a, ax29b):
    ax.set_facecolor("#0d0d18")

im29 = ax29a.imshow(
    f_mix_ts.T,
    aspect="auto", origin="lower",
    extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
    cmap="viridis", vmin=0.0, vmax=0.5,
)
ax29a.set_yscale("log")
ax29a.set_xlabel(time_label, fontsize=10)
ax29a.set_ylabel("r [kpc]", fontsize=10)
ax29a.set_title(
    r"Mass-mixing fraction  $f_{\rm mix}(r,t) = \min(\rho_{\rm MW}, \rho_{\rm M31})\ /\ \rho_{\rm tot}$",
    fontsize=10,
)
cb29 = fig29.colorbar(im29, ax=ax29a, pad=0.01)
cb29.set_label(r"$f_{\rm mix}$  (0 = unmixed, 0.5 = equal)", fontsize=8)

# Right panel: time-average mixing profile.
f_mean = np.nanmean(f_mix_ts, axis=0)
valid_f = np.isfinite(f_mean)
ax29b.plot(f_mean[valid_f], r_mid_sph[valid_f], color="#00d4aa", lw=2.0)
ax29b.set_yscale("log")
ax29b.set_xlim(0, 0.52)
ax29b.axvline(0.5, color="#ffffff", lw=0.7, ls="--", alpha=0.4)
ax29b.set_xlabel(r"$\langle f_{\rm mix} \rangle_t$", fontsize=10)
ax29b.set_ylim(R_BINS[0], R_BINS[-1])
ax29b.tick_params(labelleft=False)
ax29b.set_title("Time-avg.", fontsize=10)

fig29.suptitle("Halo Mass-Mixing Fraction", fontsize=12)
fig29.savefig(os.path.join(OUT_DIR, "density_mixing_fraction.png"),
              dpi=300, bbox_inches="tight", facecolor=fig29.get_facecolor())
plt.close(fig29)
print("  Saved: density_mixing_fraction.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 30 — DENSITY PROFILE FITTING: NFW, EINASTO, HERNQUIST               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Physical motivation
# —————––
# The three model families cover the main theoretical predictions for
# collisionless dark matter halo density profiles:
# 
# 1. NFW (Navarro, Frenk & White 1997)
# ρ_NFW(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
# Properties: power-law cusp ρ ∝ r^{-1} at r → 0; ρ ∝ r^{-3} at r → ∞.
# Free parameters: ρ_s [M_sun kpc^{-3}],  r_s [kpc]
# 
# 2. Einasto (Einasto 1965)
# ln(ρ_E / ρ_{-2}) = −(2/α) [(r/r_{-2})^α − 1]
# where r_{-2} is the radius where Γ = −2 and ρ_{−2} is the density there.
# Properties: no central cusp; shape index α controls profile “sharpness”.
# Preferred by modern simulations (e.g., Springel et al. 2008).
# Free parameters: ρ_{-2}, r_{-2}, α
# 
# 3. Hernquist (Hernquist 1990)
# ρ_H(r) = M / (2π) × a / (r (r + a)³)
# Properties: similar to NFW at small r (∝ r^{-1}), falls as r^{-4} at large r.
# Analytical potential is known — useful for comparison with analytic models.
# Free parameters: M [M_sun] (total mass),  a [kpc] (scale length)
# 
# Fitting is done on the *density* profiles (not enclosed mass as in the
# kinematics pipeline) to directly constrain the local slope behaviour.
# We fit in log-space (minimise residuals of ln ρ) for numerical stability
# because ρ spans many decades.

print("\n[Fitting]  NFW / Einasto / Hernquist profiles …")

# ── Model definitions ──────────────────────────────────────────────────────────

def nfw_density(r, rho_s, r_s):
    """NFW density profile ρ(r) in M_sun kpc^{-3}."""
    x = r / r_s
    return rho_s / (x * (1.0 + x)**2)

def einasto_density(r, rho_m2, r_m2, alpha):
    """
    Einasto density profile ρ(r).
    
    Parameters
    ----------
    r      : array  — radii [kpc]
    rho_m2 : float  — density at r = r_{-2}  [M_sun kpc^{-3}]
    r_m2   : float  — scale radius where Γ = −2  [kpc]
    alpha  : float  — shape index (typical range 0.1–0.3 for DM halos)
    """
    return rho_m2 * np.exp(
        -(2.0 / alpha) * ((r / r_m2)**alpha - 1.0)
    )

def hernquist_density(r, M_total, a):
    """
    Hernquist density profile ρ(r).
    
    Parameters
    ----------
    M_total : float  — total mass  [M_sun]
    a       : float  — scale length  [kpc]
    """
    return (M_total / (2.0 * np.pi)) * a / (r * (r + a)**3)


def fit_density_models(rho_row, r_mid_loc, r_min=R_FIT_MIN_KPC, r_max=R_FIT_MAX_KPC):
    """
    Fit NFW, Einasto, and Hernquist models to one snapshot’s ρ(r) profile.
    
    Fitting is performed in log-space: we minimise sum of (ln ρ_meas - ln ρ_model)²,
    which gives equal weight per decade rather than over-weighting the dense centre.
    
    Returns
    -------
    dict with keys "nfw", "einasto", "hernquist", each containing:
        popt    : array    — best-fit parameters  (or None if fit failed)
        chi2    : float    — reduced chi-squared in log-space
        success : bool
    """
    # Select finite, positive bins within the fitting radius range.
    mask    = (r_mid_loc >= r_min) & (r_mid_loc <= r_max) & \
              np.isfinite(rho_row) & (rho_row > 0)
    r_fit   = r_mid_loc[mask]
    rho_fit = rho_row[mask]

    if len(r_fit) < 5:
        empty = {"popt": None, "chi2": np.nan, "success": False}
        return {"nfw": empty, "einasto": empty, "hernquist": empty}

    ln_rho = np.log(rho_fit)   # fit in log-space

    results = {}

    # ── NFW ────────────────────────────────────────────────────────────────────
    rho_s0  = np.exp(ln_rho.max()) * 5.0
    r_s0    = 30.0
    try:
        popt_nfw, _ = curve_fit(
            lambda r, rs, rss: np.log(nfw_density(r, rs, rss)),
            r_fit, ln_rho,
            p0=[rho_s0, r_s0],
            bounds=([1e2, 0.1], [1e16, 300.0]),
            maxfev=5000,
        )
        pred_nfw  = np.log(nfw_density(r_fit, *popt_nfw))
        chi2_nfw  = np.sum((ln_rho - pred_nfw)**2) / max(1, len(r_fit) - 2)
        results["nfw"] = {"popt": popt_nfw, "chi2": chi2_nfw, "success": True}
    except Exception:
        results["nfw"] = {"popt": None, "chi2": np.nan, "success": False}

    # ── Einasto ────────────────────────────────────────────────────────────────
    rho_m2_0 = np.exp(np.median(ln_rho))
    r_m2_0   = 30.0
    alpha0   = 0.18
    try:
        popt_ein, _ = curve_fit(
            lambda r, rm2, rrm2, a: np.log(einasto_density(r, rm2, rrm2, a)),
            r_fit, ln_rho,
            p0=[rho_m2_0, r_m2_0, alpha0],
            bounds=([1e2, 0.1, 0.05], [1e16, 300.0, 1.0]),
            maxfev=10000,
        )
        pred_ein  = np.log(einasto_density(r_fit, *popt_ein))
        chi2_ein  = np.sum((ln_rho - pred_ein)**2) / max(1, len(r_fit) - 3)
        results["einasto"] = {"popt": popt_ein, "chi2": chi2_ein, "success": True}
    except Exception:
        results["einasto"] = {"popt": None, "chi2": np.nan, "success": False}

    # ── Hernquist ──────────────────────────────────────────────────────────────
    M_tot0 = rho_fit[0] * 4 * np.pi * r_fit[0]**3 * 3   # rough total mass
    a0     = 30.0
    try:
        popt_her, _ = curve_fit(
            lambda r, Mt, aa: np.log(hernquist_density(r, Mt, aa)),
            r_fit, ln_rho,
            p0=[M_tot0, a0],
            bounds=([1e6, 0.1], [1e16, 300.0]),
            maxfev=5000,
        )
        pred_her  = np.log(hernquist_density(r_fit, *popt_her))
        chi2_her  = np.sum((ln_rho - pred_her)**2) / max(1, len(r_fit) - 2)
        results["hernquist"] = {"popt": popt_her, "chi2": chi2_her, "success": True}
    except Exception:
        results["hernquist"] = {"popt": None, "chi2": np.nan, "success": False}

    return results


# ── Run fits on every STEP_FIT-th snapshot ────────────────────────────────────
fit_snap_nums = SNAPSHOTS[::STEP_FIT]
n_fit_snaps   = len(fit_snap_nums)
fit_snap_idx  = {snap: ii for ii, snap in enumerate(fit_snap_nums)}

chi2_nfw_arr    = np.full(n_fit_snaps, np.nan)
chi2_ein_arr    = np.full(n_fit_snaps, np.nan)
chi2_her_arr    = np.full(n_fit_snaps, np.nan)
alpha_ein_arr   = np.full(n_fit_snaps, np.nan)  # Einasto shape parameter
rs_nfw_arr      = np.full(n_fit_snaps, np.nan)  # NFW scale radius

t_fit_start = time.perf_counter()
for snap_num in fit_snap_nums:
    ii  = fit_snap_idx[snap_num]
    i   = np.where(SNAPSHOTS == snap_num)[0]
    if len(i) == 0:
        continue
    i = i[0]
    res = fit_density_models(rho_ts[i, :], r_mid_sph)
    chi2_nfw_arr [ii] = res["nfw"    ]["chi2"]
    chi2_ein_arr [ii] = res["einasto"]["chi2"]
    chi2_her_arr [ii] = res["hernquist"]["chi2"]
    if res["einasto"]["success"]:
        alpha_ein_arr[ii] = res["einasto"]["popt"][2]
    if res["nfw"]["success"]:
        rs_nfw_arr[ii] = res["nfw"]["popt"][1]

print(f"  Fits done in {time.perf_counter()-t_fit_start:.0f}s")


# ── Figure: model fits at five key epochs ─────────────────────────────────────
print("[Fig 7]  Density profile fits …")

fig30, axes30 = plt.subplots(1, 5, figsize=(18, 6), facecolor="#0d0d18",
                             sharey=True, gridspec_kw={"wspace": 0.06})

r_plot = np.logspace(np.log10(R_FIT_MIN_KPC), np.log10(R_FIT_MAX_KPC), 200)

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES,
                                                PROFILE_LABELS,
                                                PROFILE_COLORS)):
    ax = axes30[col]
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log"); ax.set_yscale("log")

    # Measured profile.
    rho_row = rho_ts[k_idx, :]
    valid   = np.isfinite(rho_row) & (rho_row > 0)
    ax.scatter(r_mid_sph[valid], rho_row[valid],
               color="#aaaacc", s=12, zorder=3, label="Data", alpha=0.8)

    # Model overlays.
    res = fit_density_models(rho_row, r_mid_sph)

    model_specs = [
        ("nfw",      nfw_density,      "#ff9944", "NFW"),
        ("einasto",  einasto_density,  "#00d4aa", "Einasto"),
        ("hernquist",hernquist_density,"#aa55ff", "Hernquist"),
    ]
    for model_key, model_fn, mc, ml in model_specs:
        if res[model_key]["success"]:
            ax.plot(r_plot, model_fn(r_plot, *res[model_key]["popt"]),
                    color=mc, lw=1.8, label=f"{ml} χ²={res[model_key]['chi2']:.2f}")

    ax.set_xlabel("r [kpc]", fontsize=9)
    ax.set_title(label, fontsize=9)
    ax.set_xlim(R_FIT_MIN_KPC, R_FIT_MAX_KPC)
    if col == 0:
        ax.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=9)
    ax.legend(fontsize=6)

fig30.suptitle("Density Profile Fitting  ·  NFW vs. Einasto vs. Hernquist", fontsize=12)
fig30.savefig(os.path.join(OUT_DIR, "density_nfw_fit.png"),
              dpi=300, bbox_inches="tight", facecolor=fig30.get_facecolor())
plt.close(fig30)
print("  Saved: density_nfw_fit.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 31 — FIGURE 8: FIT RESIDUAL HEATMAP IN (r, t)                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Purpose
# —––
# A single chi-squared number per snapshot tells us “how good is the fit?”
# but not “where is the fit failing?”.  The residual heatmap
# 
# Δ(r, t) = (ρ_meas − ρ_NFW) / ρ_NFW
# 
# answers the spatial question: does the NFW fit fail in the inner cusp,
# the outer halo, or everywhere at once?
# 
# Positive residuals (ρ_meas > ρ_NFW) → excess mass at that radius
# Negative residuals (ρ_meas < ρ_NFW) → mass deficit (e.g., tidal stripping)
# 
# Computing for every STEP_FIT snapshot and interpolating between them gives
# a smoothly varying heatmap.

print("[Fig 8]  Fit residual heatmap …")

# Pre-compute NFW residuals at every STEP_FIT snapshot.
# Shape: (n_fit_snaps, nb_sph)
resid_ts = np.full((n_fit_snaps, nb_sph), np.nan)

for snap_num in fit_snap_nums:
    ii = fit_snap_idx[snap_num]
    i  = np.where(SNAPSHOTS == snap_num)[0]
    if len(i) == 0:
        continue
    i = i[0]
    rho_row = rho_ts[i, :]
    res      = fit_density_models(rho_row, r_mid_sph)
    if res["nfw"]["success"]:
        rho_nfw_pred = nfw_density(r_mid_sph, *res["nfw"]["popt"])
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = np.where(
                np.isfinite(rho_row) & (rho_row > 0) & (rho_nfw_pred > 0),
                (rho_row - rho_nfw_pred) / rho_nfw_pred,
                np.nan,
            )
        resid_ts[ii, :] = delta

# Time axis for fit snapshots.
time_fit_snaps = np.array([
    time_arr[np.where(SNAPSHOTS == s)[0][0]]
    if len(np.where(SNAPSHOTS == s)[0]) > 0 else np.nan
    for s in fit_snap_nums
])

fig31, (ax31a, ax31b) = plt.subplots(
    1, 2, figsize=(14, 6), facecolor="#0d0d18",
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
)
for ax in (ax31a, ax31b):
    ax.set_facecolor("#0d0d18")

t_fit_min = np.nanmin(time_fit_snaps)
t_fit_max = np.nanmax(time_fit_snaps)

im31 = ax31a.imshow(
    np.clip(resid_ts, -1.5, 1.5).T,
    aspect="auto", origin="lower",
    extent=[t_fit_min, t_fit_max, R_BINS[0], R_BINS[-1]],
    cmap="seismic", vmin=-1.5, vmax=1.5,
)
ax31a.set_yscale("log")
ax31a.set_xlabel(time_label, fontsize=10)
ax31a.set_ylabel("r [kpc]", fontsize=10)
ax31a.set_title(
    r"NFW Density Residual  $\Delta = (\rho_{\rm meas} - \rho_{\rm NFW})\ /\ \rho_{\rm NFW}$",
    fontsize=10,
)
cb31 = fig31.colorbar(im31, ax=ax31a, pad=0.01)
cb31.set_label(r"$\Delta$  (blue = deficit, red = excess)", fontsize=8)

# Right: time-average residual profile.
resid_mean = np.nanmean(resid_ts, axis=0)
valid_res  = np.isfinite(resid_mean)
ax31b.plot(resid_mean[valid_res], r_mid_sph[valid_res],
           color="#e8673a", lw=2.0)
ax31b.axvline(0, color="#555577", lw=0.8, ls="--")
ax31b.set_yscale("log"); ax31b.set_ylim(R_BINS[0], R_BINS[-1])
ax31b.set_xlabel(r"$\langle\Delta\rangle_t$", fontsize=10)
ax31b.tick_params(labelleft=False)
ax31b.set_title("Time-avg.", fontsize=10)

fig31.suptitle("NFW Fit Residuals", fontsize=12)
fig31.savefig(os.path.join(OUT_DIR, "density_fit_residuals.png"),
              dpi=300, bbox_inches="tight", facecolor=fig31.get_facecolor())
plt.close(fig31)
print("  Saved: density_fit_residuals.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 32 — FIGURE 9: HALF-MASS RADIUS AND MODEL PARAMETER EVOLUTION       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Fig 9]  Half-mass radius and model parameter evolution …")

fig32, axes32 = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d0d18",
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
axes32 = axes32.flatten()
for ax in axes32:
    ax.set_facecolor("#0d0d18")

# (a) r_half evolution.
axes32[0].plot(time_arr, r_half_3d_arr,   color="#4a8fff", lw=1.8,
               label=r"$r_{1/2, 3D}$")
axes32[0].plot(time_arr, r_half_proj_arr, color="#00d4aa", lw=1.8, ls="--",
               label=r"$R_{1/2, \rm proj}$")
axes32[0].set_ylabel("Half-mass radius [kpc]", fontsize=9)
axes32[0].set_title("Half-Mass Radius", fontsize=10)
axes32[0].legend(fontsize=8)

# (b) NFW scale radius r_s over time.
time_fit_axis = time_fit_snaps
axes32[1].plot(time_fit_axis, rs_nfw_arr, color="#ff9944", lw=1.8,
               label=r"NFW $r_s$")
axes32[1].set_ylabel(r"$r_s$ [kpc]", fontsize=9)
axes32[1].set_title("NFW Scale Radius", fontsize=10)
axes32[1].legend(fontsize=8)

# (c) Einasto shape index α over time.
axes32[2].plot(time_fit_axis, alpha_ein_arr, color="#aa55ff", lw=1.8,
               label=r"Einasto $\alpha$")
axes32[2].axhline(0.18, color="#555577", lw=0.8, ls="--",
                  label="typical DM halo α ≈ 0.18")
axes32[2].set_ylabel(r"Einasto $\alpha$", fontsize=9)
axes32[2].set_xlabel(time_label, fontsize=9)
axes32[2].set_title("Einasto Shape Index", fontsize=10)
axes32[2].legend(fontsize=8)

# (d) Model chi-squared comparison.
axes32[3].semilogy(time_fit_axis, chi2_nfw_arr, color="#ff9944", lw=1.5,
                   label="NFW")
axes32[3].semilogy(time_fit_axis, chi2_ein_arr, color="#00d4aa", lw=1.5,
                   label="Einasto")
axes32[3].semilogy(time_fit_axis, chi2_her_arr, color="#aa55ff", lw=1.5,
                   label="Hernquist")
axes32[3].axhline(1.0, color="#ffffff", lw=0.7, ls="--", alpha=0.4)
axes32[3].set_ylabel(r"Reduced $\chi^2$ (log-space)", fontsize=9)
axes32[3].set_xlabel(time_label, fontsize=9)
axes32[3].set_title("Fit Quality Comparison", fontsize=10)
axes32[3].legend(fontsize=8)

for ax in axes32:
    ax.set_xlabel(time_label, fontsize=9)

fig32.suptitle("Density Structural Parameters Over Time", fontsize=12)
fig32.savefig(os.path.join(OUT_DIR, "density_halfmass_evolution.png"),
              dpi=300, bbox_inches="tight", facecolor=fig32.get_facecolor())
plt.close(fig32)
print("  Saved: density_halfmass_evolution.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 33 — FIGURE 10: ρ(r) PROFILE ANIMATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# This animation has three panels updating simultaneously:
# Left  : ρ(r) profile (current snapshot, coloured by time, faded history)
# Centre: ρ_MW vs. ρ_M31 profiles (current snapshot)
# Right : Γ(r) logarithmic slope (current snapshot)
# 
# The “faded history” technique on the left panel — drawing previous profiles
# at lower alpha — gives the viewer a sense of how the profile has moved
# without cluttering the plot with 800 overlapping lines.  Only the last
# N_GHOST profiles are shown.

print("\n[Anim 1]  ρ(r) profile animation …")

N_GHOST     = 15     # how many past profiles to show as ghost lines
ANIM_IDXS   = np.arange(0, ns, ANIM_STEP)
N_FRAMES    = len(ANIM_IDXS)
cmap_time   = plt.cm.plasma

fig33, axes33 = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="#0d0d18",
                             gridspec_kw={"wspace": 0.32})
for ax in axes33:
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(R_BINS[0], R_BINS[-1])

ax_rho33, ax_gal33, ax_slope33 = axes33

# Shared y-limits based on full dataset.
rho_finite = rho_ts[np.isfinite(rho_ts) & (rho_ts > 0)]
if rho_finite.size > 0:
    rho_ymin = rho_finite.min() * 0.3
    rho_ymax = rho_finite.max() * 3.0
else:
    rho_ymin, rho_ymax = 1e2, 1e12

ax_rho33.set_ylim(rho_ymin, rho_ymax)
ax_gal33.set_ylim(rho_ymin, rho_ymax)
ax_slope33.set_yscale("linear")
ax_slope33.set_ylim(-5.0, 1.0)

ax_rho33.set_xlabel("r [kpc]"); ax_rho33.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]")
ax_rho33.set_title(r"$\rho(r)$  history")
ax_gal33.set_xlabel("r [kpc]"); ax_gal33.set_title(r"$\rho_{\rm MW}$ vs. $\rho_{\rm M31}$")
ax_slope33.set_xlabel("r [kpc]"); ax_slope33.set_ylabel(r"$\Gamma$")
ax_slope33.set_title(r"Slope  $\Gamma(r)$")
ax_slope33.axhline(-1, color="#555577", lw=0.6, ls=":")
ax_slope33.axhline(-3, color="#555577", lw=0.6, ls=":")

title33 = fig33.suptitle("", fontsize=11, color="#c8c8e8")

ghost_lines  = [ax_rho33.plot([], [], lw=0.8, alpha=0.0)[0]
                for _ in range(N_GHOST)]
main_line33, = ax_rho33.plot([], [], lw=2.2, color="white", zorder=5)
mw_line33,   = ax_gal33.plot([], [], lw=2.0, color="#4a8fff", label="MW")
m31_line33,  = ax_gal33.plot([], [], lw=2.0, color="#ff5fa0", label="M31")
slope_line33,= ax_slope33.plot([], [], lw=2.0, color="#e8673a")
ax_gal33.legend(fontsize=8)

def _update_density_anim(frame_idx):
    snap_i = ANIM_IDXS[frame_idx]
    color  = cmap_time(frame_idx / N_FRAMES)

    def _xy(arr, r):
        valid = np.isfinite(arr) & (arr > 0)
        return r[valid], arr[valid]

    # Current profile.
    rx, ry = _xy(rho_ts[snap_i, :], r_mid_sph)
    main_line33.set_data(rx, ry)
    main_line33.set_color(color)

    # Ghost lines: last N_GHOST frames.
    for g, ghost in enumerate(ghost_lines):
        past_idx = frame_idx - (N_GHOST - g)
        if past_idx < 0:
            ghost.set_data([], [])
            continue
        past_snap = ANIM_IDXS[past_idx]
        px, py = _xy(rho_ts[past_snap, :], r_mid_sph)
        past_color = cmap_time(past_idx / N_FRAMES)
        ghost.set_data(px, py)
        ghost.set_color(past_color)
        ghost.set_alpha(0.08 + 0.07 * g)   # older = more faded

    # MW / M31 split.
    mwx, mwy   = _xy(rho_mw_ts  [snap_i, :], r_mid_sph)
    m31x, m31y = _xy(rho_m31_ts [snap_i, :], r_mid_sph)
    mw_line33.set_data(mwx, mwy)
    m31_line33.set_data(m31x, m31y)

    # Slope.
    Gx, Gy = r_mid_sph[np.isfinite(Gamma_ts[snap_i, :])], \
              Gamma_ts[snap_i, :][np.isfinite(Gamma_ts[snap_i, :])]
    slope_line33.set_data(Gx, Gy)

    t_val = time_arr[snap_i]
    t_str = f"{t_val:.2f} Gyr" if time_is_gyr else f"Snap {SNAPSHOTS[snap_i]}"
    title33.set_text(f"MW–M31 Density Profiles  ·  {t_str}")

    return [main_line33, mw_line33, m31_line33, slope_line33] + ghost_lines

ani33 = animation.FuncAnimation(
    fig33, _update_density_anim, frames=N_FRAMES,
    interval=1000 // ANIM_FPS, blit=True,
)
writer33 = animation.FFMpegWriter(fps=ANIM_FPS, bitrate=ANIM_BITRATE,
                                  metadata=dict(title="MW-M31 Density Profile Animation"))
ani33.save(os.path.join(OUT_DIR, "density_animation.mp4"),
           writer=writer33, dpi=ANIM_DPI)
plt.close(fig33)
print("  Saved: density_animation.mp4")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 34 — FIGURE 11: 2D SURFACE DENSITY MAP ANIMATION                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Three-panel layout:
# Left  : total log Σ(x, y)  — shows the full merger morphology
# Centre: MW-only Σ_MW       — tracks MW stellar/DM debris
# Right : M31-only Σ_M31     — tracks M31 debris
# 
# Colour limits are fixed across all frames so the viewer can directly compare
# brightness between early (bright cores, faint outer halo) and late
# (spread-out, lower peak density) frames.

print("[Anim 2]  2D surface density animation …")

# Use only the pre-computed map subset (n_maps frames at STEP_MAPS intervals).
anim_map_idxs = np.arange(n_maps)
N_MAP_FRAMES  = len(anim_map_idxs)

# Global colour scale: log10 Σ.
Sigma_finite = maps_3d[np.isfinite(maps_3d) & (maps_3d > 0)]
if Sigma_finite.size > 0:
    vmin_2d = np.log10(np.percentile(Sigma_finite, 5))
    vmax_2d = np.log10(np.percentile(Sigma_finite, 99.9))
else:
    vmin_2d, vmax_2d = 4.0, 9.5

fig34, axes34 = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="#0d0d18",
                             gridspec_kw={"wspace": 0.05})
for ax in axes34:
    ax.set_facecolor("#0d0d18")

# Initialise imshow objects with the first frame.
def _log10_smooth(arr):
    smoothed = gaussian_filter(np.where(np.isfinite(arr), arr, 0.0),
                               sigma=MAP_SMOOTH_SIGMA)
    return np.where(smoothed > 0, np.log10(smoothed), np.nan)

first_total = _log10_smooth(maps_3d[0])
first_mw    = _log10_smooth(maps_mw [0])
first_m31   = _log10_smooth(maps_m31[0])

im34_tot = axes34[0].imshow(first_total.T, origin="lower", aspect="equal",
                            extent=MAP_EXTENT, cmap="inferno",
                            vmin=vmin_2d, vmax=vmax_2d)
im34_mw  = axes34[1].imshow(first_mw.T,    origin="lower", aspect="equal",
                            extent=MAP_EXTENT, cmap="Blues_r",
                            vmin=vmin_2d, vmax=vmax_2d)
im34_m31 = axes34[2].imshow(first_m31.T,   origin="lower", aspect="equal",
                            extent=MAP_EXTENT, cmap="Reds_r",
                            vmin=vmin_2d, vmax=vmax_2d)

for ax, lbl in zip(axes34, ["Total Σ", "MW only", "M31 only"]):
    ax.set_xlabel("x [kpc]", fontsize=9)
    ax.set_title(lbl, fontsize=10, color="#c8c8e8")
axes34[0].set_ylabel("y [kpc]", fontsize=9)

title34 = fig34.suptitle("", fontsize=11, color="#c8c8e8")

def _update_map_anim(frame_idx):
    mi = anim_map_idxs[frame_idx]
    im34_tot.set_data(_log10_smooth(maps_3d [mi]).T)
    im34_mw .set_data(_log10_smooth(maps_mw [mi]).T)
    im34_m31.set_data(_log10_smooth(maps_m31[mi]).T)
    t_val = time_maps[mi]
    t_str = f"{t_val:.2f} Gyr" if time_is_gyr else f"Snap {map_snap_nums[mi]}"
    title34.set_text(f"2D Surface Density  ·  {t_str}")
    return [im34_tot, im34_mw, im34_m31]

ani34 = animation.FuncAnimation(
    fig34, _update_map_anim, frames=N_MAP_FRAMES,
    interval=1000 // ANIM_FPS, blit=True,
)
ani34.save(os.path.join(OUT_DIR, "density_2d_animation.mp4"),
           writer=writer33, dpi=ANIM_DPI)
plt.close(fig34)
print("  Saved: density_2d_animation.mp4")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 35 — FIGURE 12: MASTER SUMMARY PANEL                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# A single 4×2 figure summarising the most important density results:
# (1,1) ρ(r) profiles at 5 epochs
# (1,2) Σ(R) profiles at 5 epochs
# (2,1) Γ(r, t) slope heatmap
# (2,2) f_mix(r, t) mixing heatmap
# (3,1) ρ_0(t) and r_half(t)
# (3,2) NFW/Einasto/Hernquist chi² comparison
# (4,1) 2D Σ map at early epoch
# (4,2) 2D Σ map at late epoch
# 
# This panel is designed to be the single-figure summary for a paper or poster.

print("[Fig 12]  Master summary panel …")

fig35 = plt.figure(figsize=(16, 20), facecolor="#0d0d18")
gs35  = gridspec.GridSpec(4, 2, figure=fig35,
                          hspace=0.42, wspace=0.32,
                          left=0.08, right=0.97,
                          top=0.95, bottom=0.05)

BG    = "#0d0d18"
MUTED = "#7070a0"

def _sax(fig, gs, r, c, log_x=True, log_y=True):
    """Add a styled subplot with optional log axes."""
    ax = fig.add_subplot(gs[r, c])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")
    ax.tick_params(colors="#9090b0", labelsize=8)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    return ax

# ── Panel (0,0): ρ(r) ─────────────────────────────────────────────────────────
ax00 = _sax(fig35, gs35, 0, 0)
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y = rho_ts[k_idx, :]
    v = np.isfinite(y) & (y > 0)
    if v.any():
        ax00.plot(r_mid_sph[v], y[v], color=color, lw=1.5, label=label)
ax00.set_xlabel("r [kpc]", fontsize=8, color=MUTED)
ax00.set_ylabel(r"$\rho$ [M$_\odot$/kpc³]", fontsize=8, color=MUTED)
ax00.set_title(r"$\rho(r)$  at 5 epochs", fontsize=9)
ax00.legend(fontsize=6)

# ── Panel (0,1): Σ(R) ─────────────────────────────────────────────────────────
ax01 = _sax(fig35, gs35, 0, 1)
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y = Sigma_ts[k_idx, :]
    v = np.isfinite(y) & (y > 0)
    if v.any():
        ax01.plot(r_mid_proj[v], y[v], color=color, lw=1.5, label=label)
ax01.set_xlabel("R [kpc]", fontsize=8, color=MUTED)
ax01.set_ylabel(r"$\Sigma$ [M$_\odot$/kpc²]", fontsize=8, color=MUTED)
ax01.set_title(r"$\Sigma(R)$  at 5 epochs", fontsize=9)
ax01.legend(fontsize=6)

# ── Panel (1,0): Γ heatmap ────────────────────────────────────────────────────
ax10 = _sax(fig35, gs35, 1, 0, log_x=False, log_y=False)
im10 = ax10.imshow(np.clip(Gamma_ts, -5, 1).T, aspect="auto", origin="lower",
                    extent=[t_min, t_max, 0, nb_sph],
                    cmap="bwr", vmin=-4, vmax=0.5)
ax10.set_xlabel(time_label, fontsize=8, color=MUTED)
ax10.set_ylabel("Bin index (log r)", fontsize=8, color=MUTED)
ax10.set_title(r"$\Gamma(r,t)$ slope heatmap", fontsize=9)
fig35.colorbar(im10, ax=ax10, label=r"$\Gamma$", shrink=0.8)

# ── Panel (1,1): f_mix heatmap ────────────────────────────────────────────────
ax11 = _sax(fig35, gs35, 1, 1, log_x=False, log_y=False)
im11 = ax11.imshow(f_mix_ts.T, aspect="auto", origin="lower",
                    extent=[t_min, t_max, 0, nb_sph],
                    cmap="viridis", vmin=0, vmax=0.5)
ax11.set_xlabel(time_label, fontsize=8, color=MUTED)
ax11.set_ylabel("Bin index (log r)", fontsize=8, color=MUTED)
ax11.set_title(r"$f_{\rm mix}(r,t)$ mixing fraction", fontsize=9)
fig35.colorbar(im11, ax=ax11, label=r"$f_{\rm mix}$", shrink=0.8)

# ── Panel (2,0): ρ_0 and r_half ───────────────────────────────────────────────
ax20 = _sax(fig35, gs35, 2, 0, log_x=False, log_y=False)
ax20_r = ax20.twinx()
ax20.plot(time_arr, np.log10(np.where(rho0_arr > 0, rho0_arr, np.nan)),
          color="#e8673a", lw=1.5, label=r"$\log_{10}\rho_0$")
ax20_r.plot(time_arr, r_half_3d_arr, color="#4a8fff", lw=1.5, ls="--",
            label=r"$r_{1/2, 3D}$")
ax20.set_xlabel(time_label, fontsize=8, color=MUTED)
ax20.set_ylabel(r"$\log_{10}\rho_0$", fontsize=8, color="#e8673a")
ax20_r.set_ylabel(r"$r_{1/2}$ [kpc]", fontsize=8, color="#4a8fff")
ax20.set_title(r"Central density & half-mass radius", fontsize=9)

# ── Panel (2,1): chi² comparison ──────────────────────────────────────────────
ax21 = _sax(fig35, gs35, 2, 1, log_x=False, log_y=True)
ax21.plot(time_fit_axis, chi2_nfw_arr, color="#ff9944", lw=1.2, label="NFW")
ax21.plot(time_fit_axis, chi2_ein_arr, color="#00d4aa", lw=1.2, label="Einasto")
ax21.plot(time_fit_axis, chi2_her_arr, color="#aa55ff", lw=1.2, label="Hernquist")
ax21.axhline(1, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
ax21.set_xlabel(time_label, fontsize=8, color=MUTED)
ax21.set_ylabel(r"Reduced $\chi^2$", fontsize=8, color=MUTED)
ax21.set_title("Profile fit quality", fontsize=9)
ax21.legend(fontsize=6)

# ── Panel (3,0): early 2D map ─────────────────────────────────────────────────
ax30 = fig35.add_subplot(gs35[3, 0])
ax30.set_facecolor(BG)
early_mi = 0
ax30.imshow(_log10_smooth(maps_3d[early_mi]).T, origin="lower", aspect="equal",
            extent=MAP_EXTENT, cmap="inferno", vmin=vmin_2d, vmax=vmax_2d)
ax30.set_title(f"Early  Σ(x,y)  t={time_maps[early_mi]:.2f} Gyr"
               if time_is_gyr else f"Early  Σ(x,y)",
               fontsize=9)
ax30.set_xlabel("x [kpc]", fontsize=8, color=MUTED)
ax30.set_ylabel("y [kpc]", fontsize=8, color=MUTED)

# ── Panel (3,1): late 2D map ──────────────────────────────────────────────────
ax31_ = fig35.add_subplot(gs35[3, 1])
ax31_.set_facecolor(BG)
late_mi = n_maps - 1
ax31_.imshow(_log10_smooth(maps_3d[late_mi]).T, origin="lower", aspect="equal",
            extent=MAP_EXTENT, cmap="inferno", vmin=vmin_2d, vmax=vmax_2d)
ax31_.set_title(f"Late  Σ(x,y)  t={time_maps[late_mi]:.2f} Gyr"
               if time_is_gyr else f"Late  Σ(x,y)",
               fontsize=9)
ax31_.set_xlabel("x [kpc]", fontsize=8, color=MUTED)

fig35.suptitle("MW–M31 Density Pipeline  ·  Master Summary",
               fontsize=14, color="#c8c8e8", fontweight="bold")
fig35.savefig(os.path.join(OUT_DIR, "density_summary_panel.png"),
              dpi=200, bbox_inches="tight", facecolor=fig35.get_facecolor())
plt.close(fig35)
print("  Saved: density_summary_panel.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 36 — CLEANUP AND FINAL MANIFEST                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Note: cleanup is deferred to the END of the script (unlike in the kinematics
# pipeline where it happened mid-run and prevented §17 from re-opening files).
# This ensures all sections that need raw snapshot data have already run.

shutil.rmtree(tmpdir, ignore_errors=True)
print(f"\n[cleanup] Removed: {tmpdir}")

print("\n" + "="*70)
print("  OUTPUT MANIFEST")
print("="*70)
print(f"  {'File':<48} {'MB':>6}  Type")
print(f"  {'-'*48} {'-'*6}  ––")
total_mb = 0.0
for fn in sorted(os.listdir(OUT_DIR)):
    fp   = os.path.join(OUT_DIR, fn)
    mb   = os.path.getsize(fp) / 1e6
    total_mb += mb
    kind = "animation" if fn.endswith(".mp4") else "figure"
    print(f"  {fn:<48} {mb:6.2f}  {kind}")
print(f"  {'-'*48} {'-'*6}")
print(f"  {'TOTAL':<48} {total_mb:6.2f}")
print("="*70)
print(f"\n[DONE] density pipeline complete.")
