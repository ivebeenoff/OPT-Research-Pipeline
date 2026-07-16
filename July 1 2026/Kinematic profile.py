"""
================================================================================
                MW–M31 MERGER KINEMATIC PROFILES PIPELINE
================================================================================

Author  : Abhinav Vatsa

## Overview

This computational pipeline processes N-body simulation snapshots of the 
Milky Way (MW) and Andromeda (M31) merger to calculate a suite of mass-weighted 
kinematic profiles as a function of both radius and simulation time. It traces 
and outputs publication-ready figures detailing the gravitational restructuring 
and phase-space mixing of the joint system.

Computed profiles per snapshot:
• σ_r(r)    — mass-weighted radial velocity dispersion
• σ_t(r)    — mass-weighted tangential velocity dispersion
• v_rot(r)  — mean azimuthal velocity about the z-axis
• j(r)      — mass-weighted specific angular momentum magnitude
• M_enc(r)  — true cumulative enclosed mass profile
• v_esc(r)  — Newtonian escape speed limit
• β(r)      — Binney velocity anisotropy parameter

Global scalars are tracked over time to map a diagnostic overview of the merger's 
gravitational history.

================================================================================
"""

# ── Standard Library Imports ──────────────────────────────────────────────────
import os
import tarfile
import shutil
import tempfile
import warnings
import time

# ── Scientific & Plotting Stack ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC-safe non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import LogLocator, LogFormatter

# ── Simulation Data Readers ──────────────────────────────────────────────────
from ReadFile import Read
from CenterOfMass2 import CenterOfMass

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — CONFIGURATION & PARAMETERS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Snapshot Range Boundaries ──────────────────────────────────────────────────
START_SNAP = 0
END_SNAP   = 800

# ── Core Alignment Target ─────────────────────────────────────────────────────
# 1 = Dark Matter Halo (optimal for large-scale potential tracking)
# 2 = Disk Component (optimal for baryonic tracer dynamics)
PTYPE = 1

# ── Radial Bin Geometry ───────────────────────────────────────────────────────
# Logarithmically spaced to resolve high-gradient inner cores and diffuse envelopes.
R_BINS = np.logspace(-1, np.log10(400.0), 20)
MIN_PARTICLES_PER_BIN = 10

# ── Diagnostic Aperture Boundaries ────────────────────────────────────────────
INNER_RADIUS_KPC = 30.0
R_MAX_KPC = 400.0
COM_FALLBACK_RADIUS_KPC = 15.0

# ── Physical Constants ────────────────────────────────────────────────────────
G_KPC_KMS2_MSUN = 4.30091e-6
MASS_UNIT_MSUN = 1.0e10

# ── Workspace Directories ─────────────────────────────────────────────────────
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global Aesthetic Dark Theme ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0d18",
    "axes.facecolor":    "#0d0d18",
    "axes.edgecolor":    "#2a2a4a",
    "axes.labelcolor":   "#c8c8e8",
    "axes.grid":        True,
    "grid.color":        "#1e1e36",
    "grid.linewidth":   0.6,
    "xtick.color":       "#9090b0",
    "ytick.color":       "#9090b0",
    "text.color":        "#c8c8e8",
    "legend.facecolor":  "#0d0d18",
    "legend.edgecolor":  "#2a2a4a",
    "legend.fontsize":   8,
    "font.family":       "monospace",
})

SNAPSHOTS = np.arange(START_SNAP, END_SNAP + 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — FLAT ARCHIVE INGESTION ENGINE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_snapshots_from_tarballs(work_dir: str) -> str:
    """
    Scans the current working directory, selectively unpacking MW_NNN and
    M31_NNN text tables into a self-contained temporary workspace.
    """
    tmpdir = tempfile.mkdtemp(prefix="mwm31_snaps_")
    print(f"[extract] Temporary Ingestion Workspace: {tmpdir}")

    tar_files = [fn for fn in os.listdir(work_dir) if fn.endswith(".tar")]
    if not tar_files:
        warnings.warn("No .tar files located. Assuming local text flat-files are available.")
        return tmpdir

    for fn in tar_files:
        full_path = os.path.join(work_dir, fn)
        print(f"[extract] Opening archive {fn}...")
        with tarfile.open(full_path, "r") as tar:
            members_to_extract = [
                m for m in tar.getmembers()
                if m.isfile() and ("MW_" in m.name or "M31_" in m.name)
            ]
            if not members_to_extract:
                continue
            for member in members_to_extract:
                member.name = os.path.basename(member.name)
                tar.extract(member, path=tmpdir)
            print(f"  Extracted {len(members_to_extract)} snapshots.")

    return tmpdir


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — PHASE-SPACE CENTER OF MASS TRACKERS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_com_position(mw: CenterOfMass,
                     x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     m_raw: np.ndarray) -> np.ndarray:
    """
    Returns the iterative shrinking-sphere center of mass position (Power et al. 2003).
    """
    xcom, ycom, zcom = mw.COMdefine(x, y, z, m_raw)
    return np.array([xcom, ycom, zcom])


def get_com_velocity(mw: CenterOfMass,
                     pos_com: np.ndarray,
                     pos_all: np.ndarray,
                     vel_all: np.ndarray,
                     mass_all: np.ndarray) -> np.ndarray:
    """
    Extracts joint system center-of-mass velocity. Falls back to an
    aperture mass-weighted velocity vector if units/API exceptions arise.
    """
    try:
        v_com_qty = mw.COM_V(
            pos_com[0] * mw.x.unit,
            pos_com[1] * mw.y.unit,
            pos_com[2] * mw.z.unit,
        )[0]
        return np.array(v_com_qty)
    except Exception as exc:
        warnings.warn(f"COM_V failed ({exc}); falling back to aperture mass-weighted averaging.")

        dr = np.linalg.norm(pos_all - pos_com, axis=1)
        inner_mask = dr < COM_FALLBACK_RADIUS_KPC

        if inner_mask.sum() < 5:
            warnings.warn("Aperture density deficit (< 5 particles) — returning zero vCOM velocity.")
            return np.zeros(3)

        w = mass_all[inner_mask]
        return np.array([np.sum(w * vel_all[inner_mask, dim]) / np.sum(w) for dim in range(3)])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — KINEMATIC RESOLUTION ENGINE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_profiles_for_snapshot(mw_path: str, m31_path: str) -> dict:
    """
    Resolves positions, velocities, and angular momentum vectors into co-moving,
    re-centered coordinates, projecting kinematic metrics over the radial grid.
    """
    MW  = CenterOfMass(mw_path,  PTYPE)
    M31 = CenterOfMass(m31_path, PTYPE)

    # ── Phase-Space Concatenation ─────────────────────────────────────────────
    x  = np.concatenate((MW.x,  M31.x))
    y  = np.concatenate((MW.y,  M31.y))
    z  = np.concatenate((MW.z,  M31.z))
    vx = np.concatenate((MW.vx, M31.vx))
    vy = np.concatenate((MW.vy, M31.vy))
    vz = np.concatenate((MW.vz, M31.vz))

    m_raw = np.concatenate((MW.m, M31.m))
    m_msun = m_raw * MASS_UNIT_MSUN

    pos = np.vstack((x, y, z)).T
    vel = np.vstack((vx, vy, vz)).T

    # ── COM Frame Realignment ─────────────────────────────────────────────────
    pos_com = get_com_position(MW, x, y, z, m_raw)
    vel_com = get_com_velocity(MW, pos_com, pos, vel, m_msun)

    r_vec   = pos - pos_com
    vel_rel = vel - vel_com
    r_mag   = np.linalg.norm(r_vec, axis=1)

    # ── Vectorized Kinematic Decompositions ──
    # Compute normalized radial unit vectors avoiding central division anomalies
    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = np.where(r_mag[:, None] > 0, r_vec / r_mag[:, None], 0.0)

    # Project radial velocity scalar: v_r = v_rel · r_hat
    v_radial = np.einsum("ij,ij->i", vel_rel, r_hat)

    # Calculate tangential speed: v_t = sqrt(|v_rel|^2 - v_r^2)
    v_tang2 = np.sum(vel_rel**2, axis=1) - v_radial**2
    v_tang  = np.sqrt(np.maximum(v_tang2, 0.0))

    # Specific angular momentum magnitude: |r x v|
    j_vec = np.cross(r_vec, vel_rel)
    j_mag = np.linalg.norm(j_vec, axis=1)

    # ── Optimized Enclosed Mass Engine (Vectorized Cumulative Sum) ──
    # Pre-sort radii to compute enclosed mass M(<r) in O(N log N) time
    sort_idx = np.argsort(r_mag)
    r_sorted = r_mag[sort_idx]
    m_cumsum = np.cumsum(m_msun[sort_idx])

    # Allocate radial profiles
    nb = len(R_BINS) - 1
    sigma_r        = np.full(nb, np.nan)
    sigma_t        = np.full(nb, np.nan)
    v_rot          = np.full(nb, np.nan)
    j_spec         = np.full(nb, np.nan)
    M_enclosed_bin = np.full(nb, np.nan)
    v_esc          = np.full(nb, np.nan)
    beta_profile   = np.full(nb, np.nan)

    # Radial bin assignment
    bin_indices = np.digitize(r_mag, R_BINS) - 1

    for b in range(nb):
        mask = bin_indices == b
        n_in_bin = mask.sum()

        if n_in_bin < MIN_PARTICLES_PER_BIN:
            continue

        w   = m_msun[mask]
        W_sum = w.sum()

        vr  = v_radial[mask]
        vt  = v_tang[mask]

        # Mass-weighted dispersion profiles: σ² = Σ(m_i (v_i − ⟨v⟩_m)²) / Σ m_i
        vr_mean   = np.sum(w * vr) / W_sum
        vt_mean   = np.sum(w * vt) / W_sum
        sigma_r[b] = np.sqrt(np.sum(w * (vr - vr_mean)**2) / W_sum)
        sigma_t[b] = np.sqrt(np.sum(w * (vt - vt_mean)**2) / W_sum)

        # Cylindrical Azimuthal Velocity v_φ = (-x*v_y + y*v_x) / R
        rx_b = r_vec[mask, 0]
        ry_b = r_vec[mask, 1]
        R_cyl = np.sqrt(rx_b**2 + ry_b**2)

        nonzero_R = R_cyl > 0.0
        if nonzero_R.any():
            vphi_particles = (
                -rx_b[nonzero_R] * vel_rel[mask, 1][nonzero_R]
                + ry_b[nonzero_R] * vel_rel[mask, 0][nonzero_R]
            ) / R_cyl[nonzero_R]
            v_rot[b] = np.mean(vphi_particles)

        # Specific mass-weighted angular momentum profile
        j_spec[b] = np.sum(w * j_mag[mask]) / W_sum

        # Extract boundary enclosed mass from pre-sorted cumulative distribution
        r_outer = R_BINS[b + 1]
        search_pos = np.searchsorted(r_sorted, r_outer, side="right")
        M_encl = m_cumsum[search_pos - 1] if search_pos > 0 else 0.0
        M_enclosed_bin[b] = M_encl

        # Circular/Escape boundaries
        if r_outer > 0.0 and M_encl > 0.0:
            v_esc[b] = np.sqrt(2.0 * G_KPC_KMS2_MSUN * M_encl / r_outer)

        # Anisotropy: β = 1 - (σ_t^2 / 2*σ_r^2)
        if sigma_r[b] > 0.0:
            beta_profile[b] = 1.0 - (sigma_t[b]**2) / (2.0 * sigma_r[b]**2)

    # ── Global Scalar Metrics ─────────────────────────────────────────────────
    M_tot = m_msun.sum()
    mean_vrad = np.sum(m_msun * v_radial) / M_tot
    mean_vt   = np.sum(m_msun * v_tang)   / M_tot

    sigma_r_global = np.sqrt(np.sum(m_msun * (v_radial - mean_vrad)**2) / M_tot)
    sigma_t_global = np.sqrt(np.sum(m_msun * (v_tang   - mean_vt  )**2) / M_tot)
    total_j = np.sum(m_msun * j_mag) / M_tot

    # ── Simulation Time Metadata Parsing ──────────────────────────────────────
    sim_time = None
    if hasattr(MW, "time"):
        try:
            sim_time = float(MW.time.value)
        except Exception:
            try:
                sim_time = float(MW.time)
            except Exception:
                pass

    return {
        "r_mid":           0.5 * (R_BINS[:-1] + R_BINS[1:]),
        "sigma_r":         sigma_r,
        "sigma_t":         sigma_t,
        "v_rot":           v_rot,
        "j_spec":          j_spec,
        "M_enclosed_bin":  M_enclosed_bin,
        "v_esc":           v_esc,
        "beta":            beta_profile,
        "total_j":         total_j,
        "sigma_r_global":  sigma_r_global,
        "sigma_t_global":  sigma_t_global,
        "mean_vrad":       mean_vrad,
        "mean_vt":         mean_vt,
        "time":            sim_time,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — TEMPORAL RUN LOOP                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

tmpdir = extract_snapshots_from_tarballs(".")

nb = len(R_BINS) - 1
ns = len(SNAPSHOTS)

# Pre-allocate profile structures
sigma_r_ts = np.full((ns, nb), np.nan)
sigma_t_ts = np.full((ns, nb), np.nan)
vrot_ts    = np.full((ns, nb), np.nan)
j_ts       = np.full((ns, nb), np.nan)
beta_ts    = np.full((ns, nb), np.nan)
vesc_ts    = np.full((ns, nb), np.nan)
menc_ts    = np.full((ns, nb), np.nan)

# Pre-allocate scalar arrays
time_arr         = np.full(ns, np.nan)
sigma_r_glob_arr = np.full(ns, np.nan)
sigma_t_glob_arr = np.full(ns, np.nan)
j_glob_arr       = np.full(ns, np.nan)
mean_vrad_arr    = np.full(ns, np.nan)

print("\n" + "="*80)
print("  SECTION 5 · Temporal Profile Aggregation")
print("="*80)

t_loop_start = time.perf_counter()

for i, snap_num in enumerate(SNAPSHOTS):
    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    t_snap_start = time.perf_counter()

    try:
        out = compute_profiles_for_snapshot(mw_file, m31_file)
    except Exception as exc:
        print(f"  [ERROR] Processing failure at snap {snap_num:04d}: {exc}")
        continue

    # ── Synchronize Profiles ──────────────────────────────────────────────────
    sigma_r_ts[i, :] = out["sigma_r"]
    sigma_t_ts[i, :] = out["sigma_t"]
    vrot_ts   [i, :] = out["v_rot"]
    j_ts      [i, :] = out["j_spec"]
    beta_ts   [i, :] = out["beta"]
    vesc_ts   [i, :] = out["v_esc"]
    menc_ts   [i, :] = out["M_enclosed_bin"]

    # ── Synchronize Scalars ───────────────────────────────────────────────────
    time_arr        [i] = out["time"] if out["time"] is not None else float(snap_num)
    sigma_r_glob_arr[i] = out["sigma_r_global"]
    sigma_t_glob_arr[i] = out["sigma_t_global"]
    j_glob_arr      [i] = out["total_j"]
    mean_vrad_arr   [i] = out["mean_vrad"]

    dt = time.perf_counter() - t_snap_start
    if (i + 1) % 100 == 0 or snap_num in [START_SNAP, END_SNAP]:
        print(f"  snap {snap_num:04d} ({dt:.2f}s) | σ_r_global = {out['sigma_r_global']:.1f} km/s | j_global = {out['total_j']:.1f} kpc·km/s")

t_total = time.perf_counter() - t_loop_start
print(f"\n[Temporal loop finished] Full run processed in {t_total / 60:.1f} minutes.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — DERIVED METRICS FOR PLOTTING                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

r_mid = 0.5 * (R_BINS[:-1] + R_BINS[1:])

# ── Aperture Density Evaluations (r <= INNER_RADIUS_KPC) ──────────────────────
inner_mask = r_mid <= INNER_RADIUS_KPC

sigma_r_inner = np.nanmean(sigma_r_ts[:, inner_mask], axis=1)
sigma_t_inner = np.nanmean(sigma_t_ts[:, inner_mask], axis=1)
vrot_inner    = np.nanmean(vrot_ts[:, inner_mask], axis=1)
j_inner       = np.nanmean(j_ts[:, inner_mask], axis=1)
beta_inner    = np.nanmean(beta_ts[:, inner_mask], axis=1)

t_axis       = time_arr
t_valid      = t_axis[np.isfinite(t_axis)]
time_is_gyr  = t_valid.size > 0 and t_valid.min() > 0.05
time_label   = "Time [Gyr]" if time_is_gyr else "Snapshot index"

# Sub-select epochs for snapshot grid plots
profile_snap_fractions = [0.0, 0.2, 0.4, 0.65, 1.0]
n_snaps                = len(SNAPSHOTS)
profile_snap_indices   = [int(f * (n_snaps - 1)) for f in profile_snap_fractions]
profile_labels = [f"Snap {SNAPSHOTS[k]}" for k in profile_snap_indices]
profile_colors = ["#00d4aa", "#7b9fff", "#ffaa44", "#ff6b9a", "#aa88ff"]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — DIAGNOSTIC 1: INNER CORE KINEMATIC EVOLUTION                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n[Fig 1] Plotting inner core velocity dispersions & bulk dynamics...")

fig1, axes1 = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"hspace": 0.08})
fig1.patch.set_facecolor("#0d0d18")

# Panel (a): Radial and Tangential Dispersions
ax = axes1[0]
ax.plot(t_axis, sigma_r_inner, color="#4da6ff", lw=1.8, label=r"$\sigma_r$ (inner)")
ax.plot(t_axis, sigma_t_inner, color="#ff7755", lw=1.8, label=r"$\sigma_t$ (inner)")
ax.fill_between(t_axis, sigma_r_inner, sigma_t_inner,
                where=np.isfinite(sigma_r_inner) & np.isfinite(sigma_t_inner),
                alpha=0.12, color="#888888")
ax.set_ylabel(r"$\sigma$ [km s$^{-1}$]", fontsize=10)
ax.legend(loc="upper right")
ax.set_title(fr"Inner-halo Dynamics (Aperture: $r \leq {INNER_RADIUS_KPC:.0f}$ kpc)", fontsize=11, pad=8)

# Panel (b): core Azimuthal Velocity
ax = axes1[1]
ax.plot(t_axis, vrot_inner, color="#00d4aa", lw=1.8, label=r"$v_{\phi}$ (inner)")
ax.axhline(0, color="#555577", lw=0.8, ls="--")
ax.set_ylabel(r"$v_\phi$ [km s$^{-1}$]", fontsize=10)
ax.legend(loc="upper right")

# Panel (c): Bulk Expansion/Infall Tracking
ax = axes1[2]
ax.plot(t_axis, mean_vrad_arr, color="#ffcc44", lw=1.8, label=r"$\langle v_r \rangle$ (global)")
ax.axhline(0, color="#555577", lw=0.8, ls="--")
ax.set_ylabel(r"$\langle v_r \rangle$ [km s$^{-1}$]", fontsize=10)
ax.set_xlabel(time_label, fontsize=10)
ax.legend(loc="upper right")

fig1.savefig(os.path.join(OUT_DIR, "kinematics_inner_evolution.png"),
            dpi=300, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
print("  Saved: kinematics_inner_evolution.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — DIAGNOSTIC 2: LOG-DISPERSION & ANISOTROPY HEATMAPS              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Fig 2] Rendering log radial dispersion and Binney anisotropy heatmaps...")

fig2, ax2 = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"hspace": 0.1})
fig2.patch.set_facecolor("#0d0d18")

t_min, t_max = np.nanmin(t_axis), np.nanmax(t_axis)
r_min, r_max = R_BINS[0], R_BINS[-1]
extent = [t_min, t_max, r_min, r_max]

# Panel (a): Log Radial Dispersion log10 σ_r(r, t)
log_sigma_r = np.where(sigma_r_ts > 0, np.log10(sigma_r_ts), np.nan)

im1 = ax2[0].imshow(
    log_sigma_r.T, aspect="auto", origin="lower", extent=extent,
    cmap="plasma", vmin=0.5, vmax=3.0
)
cb1 = fig2.colorbar(im1, ax=ax2[0], pad=0.01)
cb1.set_label(r"$\log_{10}(\sigma_r\ /\ [{\rm km\ s^{-1}}])$", fontsize=9)
ax2[0].set_ylabel("Radius [kpc]", fontsize=10)
ax2[0].set_title(r"Radial Velocity Dispersion Profile $\sigma_r(r,t)$", fontsize=11)
ax2[0].set_yscale("log")

# Panel (b): Velocity Anisotropy β(r, t)
beta_plot = np.clip(beta_ts, -2.0, 1.0)

im2 = ax2[1].imshow(
    beta_plot.T, aspect="auto", origin="lower", extent=extent,
    cmap="bwr", vmin=-1.0, vmax=1.0
)
cb2 = fig2.colorbar(im2, ax=ax2[1], pad=0.01)
cb2.set_label(r"$\beta$", fontsize=9)
ax2[1].set_ylabel("Radius [kpc]", fontsize=10)
ax2[1].set_xlabel(time_label, fontsize=10)
ax2[1].set_title(r"Velocity Anisotropy Profile $\beta(r,t)$", fontsize=11)
ax2[1].set_yscale("log")

ax2[1].text(t_max * 0.98, 1.5, "β = 0 (isotropic)",
            color="white", fontsize=7, ha="right", va="bottom", alpha=0.7)

fig2.savefig(os.path.join(OUT_DIR, "kinematics_heatmaps.png"),
            dpi=300, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)
print("  Saved: kinematics_heatmaps.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — DIAGNOSTIC 3: RADIAL PROFILE GRID (SIX PANEL OVERVIEW)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Fig 3] Constructing radial profile grid...")

fig3 = plt.figure(figsize=(14, 10), facecolor="#0d0d18")
gs3  = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.42, wspace=0.38,
                          left=0.07, right=0.97, top=0.92, bottom=0.08)

PANEL_FG = "#c8c8e8"
MUTED    = "#7070a0"

profile_quantities = [
    (sigma_r_ts, r"$\sigma_r$ [km s$^{-1}$]",      False, 0),
    (sigma_t_ts, r"$\sigma_t$ [km s$^{-1}$]",      False, 1),
    (vrot_ts,    r"$v_\phi$ [km s$^{-1}$]",        False, 2),
    (j_ts,       r"$j$ [kpc km s$^{-1}$]",         True,  3),
    (vesc_ts,    r"$v_\mathrm{esc}$ [km s$^{-1}$]", False, 4),
    (beta_ts,    r"$\beta$ (Anisotropy)",          False, 5),
]

for arr, ylabel, log_y, pidx in profile_quantities:
    row, col = divmod(pidx, 3)
    ax = fig3.add_subplot(gs3[row, col])
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
        y = arr[k_idx, :]
        valid = np.isfinite(y)
        if log_y:
            valid &= y > 0
        if valid.any():
            ax.plot(r_mid[valid], y[valid], color=color, lw=1.5, label=label)

    if "beta" in ylabel.lower() or "anisotropy" in ylabel.lower():
        ax.axhline(0, color=MUTED, lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(-1.5, 1.1)

    ax.set_xlabel("r [kpc]", fontsize=9, color=PANEL_FG)
    ax.set_ylabel(ylabel,    fontsize=9, color=PANEL_FG)
    ax.set_xlim(R_BINS[0], R_BINS[-1])
    ax.legend(fontsize=7)

fig3.suptitle("Kinematic Profiles at Selected Merger Stages", fontsize=13, color=PANEL_FG)
fig3.savefig(os.path.join(OUT_DIR, "kinematics_profiles_grid.png"),
            dpi=300, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close(fig3)
print("  Saved: kinematics_profiles_grid.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — DIAGNOSTIC 4: SPECIFIC ANGULAR MOMENTUM TRANSFERS              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Fig 4] Plotting global and core specific angular momentum evolution...")

fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor="#0d0d18")
ax4.set_facecolor("#0d0d18")

ax4.plot(t_axis, j_glob_arr, color="#00d4aa", lw=2.0, label=r"Global $\langle j \rangle$")
ax4.plot(t_axis, j_inner, color="#ff9944", lw=2.0, ls="--",
         label=fr"Inner Core ($r \leq {INNER_RADIUS_KPC:.0f}$ kpc)")

ax4.set_xlabel(time_label, fontsize=10)
ax4.set_ylabel(r"Specific Angular Momentum [kpc km s$^{-1}$]", fontsize=10)
ax4.set_title("Mass-weighted Specific Angular Momentum Transfer", fontsize=11)
ax4.legend()

fig4.savefig(os.path.join(OUT_DIR, "kinematics_angular_momentum.png"),
            dpi=300, bbox_inches="tight", facecolor=fig4.get_facecolor())
plt.close(fig4)
print("  Saved: kinematics_angular_momentum.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11 — DIAGNOSTIC 5: POTENTIAL WELL DEPTH & ESCAPE SPEEDS             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Fig 5] Plotting gravitational potential escape curves...")

fig5, ax5 = plt.subplots(figsize=(9, 6), facecolor="#0d0d18")
ax5.set_facecolor("#0d0d18")
ax5.set_xscale("log")

for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
    y = vesc_ts[k_idx, :]
    valid = np.isfinite(y) & (y > 0)
    if valid.any():
        ax5.plot(r_mid[valid], y[valid], color=color, lw=2.0, label=label)

ax5.axhline(550, color="#ffffff", lw=0.8, ls=":", alpha=0.5,
            label=r"MW $v_{\rm esc}$ solar boundary circle $\approx$ 550 km/s")

ax5.set_xlabel("r [kpc]", fontsize=10)
ax5.set_ylabel(r"$v_{\rm esc}(r)$ [km s$^{-1}$]", fontsize=10)
ax5.set_title(r"Newtonian Escape Speed: $v_{\rm esc}(r) = \sqrt{\frac{2GM(<r)}{r}}$", fontsize=11)
ax5.set_xlim(R_BINS[0], R_BINS[-1])
ax5.legend()

fig5.savefig(os.path.join(OUT_DIR, "kinematics_escape_velocity.png"),
            dpi=300, bbox_inches="tight", facecolor=fig5.get_facecolor())
plt.close(fig5)
print("  Saved: kinematics_escape_velocity.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 12 — DIAGNOSTIC 6: BINNEY ANISOTROPY CHRONOLOGY                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Fig 6] Plotting velocity anisotropy transitions...")

fig6, ax6 = plt.subplots(figsize=(9, 6), facecolor="#0d0d18")
ax6.set_facecolor("#0d0d18")
ax6.set_xscale("log")

for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
    y = beta_ts[k_idx, :]
    valid = np.isfinite(y)
    if valid.any():
        ax6.plot(r_mid[valid], y[valid], color=color, lw=2.0, label=label)
        ax6.fill_between(r_mid[valid], 0, y[valid], alpha=0.08, color=color)

ax6.axhline(0, color="#555577", lw=1.0, ls="--")
ax6.text(R_BINS[0] * 1.1, 0.04, "isotropic (β=0)", color="#9090b0", fontsize=8)

ax6.axhline(1, color="#8855aa", lw=0.7, ls=":", alpha=0.6)
ax6.text(R_BINS[0] * 1.1, 1.04, "radial limit (β=1)", color="#8855aa", fontsize=7)

ax6.set_xlim(R_BINS[0], R_BINS[-1])
ax6.set_ylim(-1.6, 1.2)
ax6.set_xlabel("r [kpc]", fontsize=10)
ax6.set_ylabel(r"$\beta(r)$", fontsize=10)
ax6.set_title(r"Velocity Anisotropy Profiles: $\beta = 1 - \frac{\sigma_t^2}{2\sigma_r^2}$", fontsize=11)
ax6.legend()

fig6.savefig(os.path.join(OUT_DIR, "kinematics_beta_selected.png"),
            dpi=300, bbox_inches="tight", facecolor=fig6.get_facecolor())
plt.close(fig6)
print("  Saved: kinematics_beta_selected.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 13 — DISK TRASHING & RESOURCE RECOVERY                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Decompressing files flat on HPC storage takes up several GBs.
# We clear this workspace now, utilizing local file mapping for downstream tasks.
shutil.rmtree(tmpdir, ignore_errors=True)
print(f"\n[cleanup] Removed primary temporal workspace directory: {tmpdir}")

print("\n" + "="*80)
print("  INTERMEDIATE STORAGE DISK FOOTPRINT")
print("="*80)
for fn in sorted(os.listdir(OUT_DIR)):
    fp   = os.path.join(OUT_DIR, fn)
    size = os.path.getsize(fp) / 1e6
    print(f"  {fn:<45} {size:6.2f} MB")
print("="*80)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 14 — DYNAMIC CIRCULAR VELOCITY TRACERS                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 14 · Observational Tracer: Circular Velocity v_c(r, t)")
print("="*80)

# Calculate Circular Velocity curves from cumulative binned masses
with np.errstate(divide="ignore", invalid="ignore"):
    vc_ts = np.where(
        (menc_ts > 0) & (r_mid > 0),
        np.sqrt(G_KPC_KMS2_MSUN * menc_ts / r_mid),
        np.nan,
    )

print(f"  v_c Array Dimensions: {vc_ts.shape} (Snapshots x Radial bins)")

# Plot circular curves vs escape speed limits
fig14, ax14 = plt.subplots(figsize=(9, 6), facecolor="#0d0d18")
ax14.set_facecolor("#0d0d18")
ax14.set_xscale("log")

for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
    vc_snap  = vc_ts [k_idx, :]
    esc_snap = vesc_ts[k_idx, :]

    valid_vc  = np.isfinite(vc_snap)  & (vc_snap  > 0)
    valid_esc = np.isfinite(esc_snap) & (esc_snap > 0)

    # Solid line represents circular rotation velocity
    if valid_vc.any():
        ax14.plot(r_mid[valid_vc], vc_snap[valid_vc],
                  color=color, lw=2.0, ls="-", label=f"{label} $v_c$")

    # Dashed line tracks point-mass escape velocity
    if valid_esc.any():
        ax14.plot(r_mid[valid_esc], esc_snap[valid_esc],
                  color=color, lw=1.2, ls="--", alpha=0.55)

ax14.text(0.98, 0.96, r"Dashed lines track $v_{\rm esc} = \sqrt{2}v_c$",
          transform=ax14.transAxes, ha="right", va="top", fontsize=7, color="#8888aa")

ax14.axhline(238.0, color="#ffcc44", lw=0.9, ls=":", alpha=0.7,
            label=r"MW $v_c(R_\odot) \approx 238$ km/s")

ax14.set_xlim(R_BINS[0], R_BINS[-1])
ax14.set_ylim(0, 500)
ax14.set_xlabel("r [kpc]", fontsize=10)
ax14.set_ylabel(r"$v_c(r)$ [km s$^{-1}$]", fontsize=10)
ax14.set_title(r"Circular Velocity Profiles: $v_c(r) = \sqrt{\frac{G M(<r)}{r}}$", fontsize=11)
ax14.legend(ncol=2, fontsize=7)

fig14.savefig(os.path.join(OUT_DIR, "kinematics_circular_velocity.png"),
            dpi=300, bbox_inches="tight", facecolor=fig14.get_facecolor())
plt.close(fig14)
print("  Saved: kinematics_circular_velocity.png")

# Calculate temporal evolution of rotational peak speeds
vc_peak_arr = np.full(ns, np.nan)
r_peak_arr  = np.full(ns, np.nan)

for i in range(ns):
    row = vc_ts[i, :]
    finite = np.isfinite(row)
    if finite.sum() > 2:
        idx_peak        = np.nanargmax(row)
        vc_peak_arr[i]  = row[idx_peak]
        r_peak_arr[i]   = r_mid[idx_peak]

fig14b, (axA, axB) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                  facecolor="#0d0d18", gridspec_kw={"hspace": 0.08})
for ax in (axA, axB):
    ax.set_facecolor("#0d0d18")

axA.plot(t_axis, vc_peak_arr, color="#f5c842", lw=1.8, label=r"$v_{c,\rm peak}$")
axA.set_ylabel(r"Peak $v_c$ [km s$^{-1}$]", fontsize=10)
axA.legend()

axB.plot(t_axis, r_peak_arr, color="#4a8fff", lw=1.8, label=r"$r(v_{c,\rm peak})$")
axB.set_yscale("log")
axB.set_ylabel(r"Peak Radius [kpc]", fontsize=10)
axB.set_xlabel(time_label, fontsize=10)
axB.legend()

fig14b.suptitle("Peak Circular Velocity Evolution", fontsize=11)
fig14b.savefig(os.path.join(OUT_DIR, "kinematics_vc_peak_evolution.png"),
            dpi=300, bbox_inches="tight", facecolor=fig14b.get_facecolor())
plt.close(fig14b)
print("  Saved: kinematics_vc_peak_evolution.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 15 — LOCAL JEANS EQUATION INTEGRAL TESTS                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 15 · Hydrostatic Equilibrium: Local Jeans Analysis")
print("="*80)

def compute_jeans_residual(sigma_r_prof, beta_prof, menc_prof, r_bins):
    """
    Computes local Jeans equilibrium residuals Δ_Jeans(r):
      Δ = (dP/dr + anisotropy_term) / (-g)
    Utilizes non-uniform gradient differences across log-spaced bin coordinates.
    """
    nb = len(r_bins) - 1
    r_mid_loc = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Pressure proxy: P = σ_r^2 (tracer number density cancels in ratios)
    P = sigma_r_prof**2
    dP_dr = np.gradient(P, r_mid_loc)

    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(
            (menc_prof > 0) & (r_mid_loc > 0),
            G_KPC_KMS2_MSUN * menc_prof / r_mid_loc**2,
            np.nan,
        )

    with np.errstate(invalid="ignore"):
        aniso_term = np.where(r_mid_loc > 0, 2.0 * beta_prof * P / r_mid_loc, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(np.isfinite(g) & (g != 0), (dP_dr + aniso_term) / (-g), np.nan)

    return delta


jeans_residuals = {}
for k_idx, label in zip(profile_snap_indices, profile_labels):
    jeans_residuals[label] = compute_jeans_residual(
        sigma_r_ts[k_idx, :],
        beta_ts   [k_idx, :],
        menc_ts   [k_idx, :],
        R_BINS,
    )

fig15, ax15 = plt.subplots(figsize=(9, 6), facecolor="#0d0d18")
ax15.set_facecolor("#0d0d18")
ax15.set_xscale("log")

for (label, delta), color in zip(jeans_residuals.items(), profile_colors):
    valid = np.isfinite(delta)
    if valid.any():
        ax15.plot(r_mid[valid], delta[valid], color=color, lw=2.0, label=label)

ax15.axhline(1.0, color="#ffffff", lw=1.0, ls="--", alpha=0.5, label="Equilibrium (Δ = 1)")
ax15.axhspan(0.8, 1.2, alpha=0.06, color="#ffffff", label="±20% Equilibrium Envelope")

ax15.set_xlim(R_BINS[0], R_BINS[-1])
ax15.set_ylim(-1.0, 3.0)
ax15.set_xlabel("r [kpc]", fontsize=10)
ax15.set_ylabel(r"Jeans Residual $\Delta_{\rm Jeans}(r)$", fontsize=10)
ax15.set_title(r"Local Hydrostatic Check: $\Delta = \frac{d\sigma_r^2/dr\ +\ 2\beta\sigma_r^2/r}{-GM/r^2}$", fontsize=10)
ax15.legend(fontsize=8)

fig15.savefig(os.path.join(OUT_DIR, "kinematics_jeans_residual.png"),
            dpi=300, bbox_inches="tight", facecolor=fig15.get_facecolor())
plt.close(fig15)
print("  Saved: kinematics_jeans_residual.png")

# Calculate inner halo Jeans score (mean absolute deviation from 1.0)
inner100_mask = r_mid <= 100.0
jeans_score   = np.full(ns, np.nan)

for i in range(ns):
    delta_i = compute_jeans_residual(sigma_r_ts[i, :], beta_ts[i, :], menc_ts[i, :], R_BINS)
    valid_inner = inner100_mask & np.isfinite(delta_i)
    if valid_inner.sum() > 2:
        jeans_score[i] = np.nanmean(np.abs(delta_i[valid_inner] - 1.0))

fig15b, ax15b = plt.subplots(figsize=(10, 4), facecolor="#0d0d18")
ax15b.set_facecolor("#0d0d18")
ax15b.plot(t_axis, jeans_score, color="#e8673a", lw=1.8, label=r"$\langle |\Delta-1| \rangle_{r < 100\ {\rm kpc}}$")
ax15b.axhline(0.2, color="#ffffff", lw=0.8, ls="--", alpha=0.4, label="20% Departure Limit")
ax15b.set_xlabel(time_label, fontsize=10)
ax15b.set_ylabel("Disequilibrium Score", fontsize=10)
ax15b.set_title("Inner-Halo Jeans Equilibrium Deviations", fontsize=11)
ax15b.legend()

fig15b.savefig(os.path.join(OUT_DIR, "kinematics_jeans_score.png"),
            dpi=300, bbox_inches="tight", facecolor=fig15b.get_facecolor())
plt.close(fig15b)
print("  Saved: kinematics_jeans_score.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 16 — TIDAL STREAM IDENTIFICATION HEATMAPS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 16 · Debris Tracing: Tidal Stream Identification")
print("="*80)

# Stream criteria thresholds
STREAM_BETA_THRESH = 0.5
STREAM_J_PERCENTILE = 75.0

stream_mask_ts = np.zeros((ns, nb), dtype=bool)

for i in range(ns):
    beta_row  = beta_ts[i, :]
    j_row     = j_ts  [i, :]

    finite_j  = j_row[np.isfinite(j_row)]
    if finite_j.size < 3:
        continue
    j_thresh  = np.nanpercentile(finite_j, STREAM_J_PERCENTILE)

    stream_mask_ts[i, :] = (
        np.isfinite(beta_row) & (beta_row > STREAM_BETA_THRESH) &
        np.isfinite(j_row)   & (j_row    > j_thresh)
    )

stream_fraction = stream_mask_ts.astype(float)

fig16, (ax16a, ax16b) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d18",
                                     gridspec_kw={"width_ratios": [2, 1], "wspace": 0.08})
for ax in (ax16a, ax16b):
    ax.set_facecolor("#0d0d18")

im16 = ax16a.imshow(
    stream_fraction.T, aspect="auto", origin="lower",
    extent=[t_axis[np.isfinite(t_axis)].min() if np.isfinite(t_axis).any() else 0,
            t_axis[np.isfinite(t_axis)].max() if np.isfinite(t_axis).any() else ns,
            R_BINS[0], R_BINS[-1]],
    cmap="hot", vmin=0, vmax=1,
)
ax16a.set_yscale("log")
ax16a.set_xlabel(time_label, fontsize=10)
ax16a.set_ylabel("r [kpc]", fontsize=10)
ax16a.set_title(fr"Kinematic Stream Candidates ($\beta > {STREAM_BETA_THRESH}$, $j > {STREAM_J_PERCENTILE:.0f}$th percentile)", fontsize=10)
fig16.colorbar(im16, ax=ax16a, label="Active Bin Fraction", pad=0.01)

# Average spatial stream distribution
mean_stream = np.nanmean(stream_fraction, axis=0)
ax16b.plot(mean_stream, r_mid, color="#ff9944", lw=2.0)
ax16b.set_xscale("linear")
ax16b.set_yscale("log")
ax16b.set_xlabel("Time-averaged Stream Fraction", fontsize=10)
ax16b.set_title("Radial Distribution", fontsize=10)
ax16b.set_ylim(R_BINS[0], R_BINS[-1])
ax16b.tick_params(labelleft=False)

fig16.suptitle("Tidal Stream Identification Proxy", fontsize=12)
fig16.savefig(os.path.join(OUT_DIR, "kinematics_tidal_streams.png"),
            dpi=300, bbox_inches="tight", facecolor=fig16.get_facecolor())
plt.close(fig16)
print("  Saved: kinematics_tidal_streams.png")

# Count flagged stream-active bins per snapshot
stream_active_bins = stream_fraction.sum(axis=1)

fig16b, ax16c = plt.subplots(figsize=(10, 4), facecolor="#0d0d18")
ax16c.set_facecolor("#0d0d18")
ax16c.plot(t_axis, stream_active_bins, color="#ff5566", lw=1.5, label="Flagged Bins")
ax16c.fill_between(t_axis, 0, stream_active_bins, alpha=0.15, color="#ff5566")
ax16c.set_xlabel(time_label, fontsize=10)
ax16c.set_ylabel("Flagged Bins", fontsize=10)
ax16c.set_title("Tidal Activity Chronology", fontsize=11)
ax16c.legend()

fig16b.savefig(os.path.join(OUT_DIR, "kinematics_tidal_activity.png"),
            dpi=300, bbox_inches="tight", facecolor=fig16b.get_facecolor())
plt.close(fig16b)
print("  Saved: kinematics_tidal_activity.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 17 — SEPARATE COALECSING SYSTEM ANALYSES                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Process a sparser grid subset for separate progenitor profiles to track
# physical mixing metrics over the timeline.
#

print("\n" + "="*80)
print("  SECTION 17 · Progenitor Tracking: Separate Galaxy Dynamics")
print("="*80)

STEP_SEPARATE = 40
separate_snap_nums = SNAPSHOTS[::STEP_SEPARATE]
print(f"  Unpacking {len(separate_snap_nums)} snapshots (Sampling stride: {STEP_SEPARATE})")

n_sep = len(separate_snap_nums)
sigma_r_mw_sep  = np.full((n_sep, nb), np.nan)
sigma_r_m31_sep = np.full((n_sep, nb), np.nan)
sigma_t_mw_sep  = np.full((n_sep, nb), np.nan)
sigma_t_m31_sep = np.full((n_sep, nb), np.nan)
time_sep        = np.full(n_sep, np.nan)
mixing_score    = np.full(n_sep, np.nan)

def _sigma_r_for_galaxy(x, y, z, vx, vy, vz, m_raw, pos_com, vel_com, r_bins_loc):
    """
    Constructs mass-weighted σ_r/σ_t profiles for a single progenitor.
    """
    nb_loc = len(r_bins_loc) - 1
    m    = m_raw * MASS_UNIT_MSUN
    pos  = np.vstack((x, y, z)).T
    vel  = np.vstack((vx, vy, vz)).T

    r_vec   = pos - pos_com
    vel_rel = vel - vel_com
    r_mag   = np.linalg.norm(r_vec, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = np.where(r_mag[:, None] > 0, r_vec / r_mag[:, None], 0.0)

    v_radial = np.einsum("ij,ij->i", vel_rel, r_hat)
    v_tang2  = np.maximum(np.sum(vel_rel**2, axis=1) - v_radial**2, 0.0)
    v_tang   = np.sqrt(v_tang2)

    sigma_r_out = np.full(nb_loc, np.nan)
    sigma_t_out = np.full(nb_loc, np.nan)
    bin_idx     = np.digitize(r_mag, r_bins_loc) - 1

    for b in range(nb_loc):
        mask = bin_idx == b
        if mask.sum() < MIN_PARTICLES_PER_BIN:
            continue
        w = m[mask]
        W = w.sum()
        vr = v_radial[mask]
        vt = v_tang[mask]
        sigma_r_out[b] = np.sqrt(np.sum(w * (vr - np.sum(w*vr)/W)**2) / W)
        sigma_t_out[b] = np.sqrt(np.sum(w * (vt - np.sum(w*vt)/W)**2) / W)

    return sigma_r_out, sigma_t_out


# Re-extract separate snaps on-demand into temp workspace
tmpdir_sep = extract_snapshots_from_tarballs(".")

for ii, snap_num in enumerate(separate_snap_nums):
    mw_file  = os.path.join(tmpdir_sep, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir_sep, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    try:
        MW_obj  = CenterOfMass(mw_file,  PTYPE)
        M31_obj = CenterOfMass(m31_file, PTYPE)
    except Exception as exc:
        print(f"  [ERROR] Processing failure at snapshot {snap_num}: {exc}")
        continue

    x_all  = np.concatenate((MW_obj.x,  M31_obj.x))
    y_all  = np.concatenate((MW_obj.y,  M31_obj.y))
    z_all  = np.concatenate((MW_obj.z,  M31_obj.z))
    vx_all = np.concatenate((MW_obj.vx, M31_obj.vx))
    vy_all = np.concatenate((MW_obj.vy, M31_obj.vy))
    vz_all = np.concatenate((MW_obj.vz, M31_obj.vz))
    m_all  = np.concatenate((MW_obj.m,  M31_obj.m))

    pos_com = get_com_position(MW_obj, x_all, y_all, z_all, m_all)
    vel_com = get_com_velocity(
        MW_obj, pos_com,
        np.vstack((x_all, y_all, z_all)).T,
        np.vstack((vx_all, vy_all, vz_all)).T,
        m_all * MASS_UNIT_MSUN,
    )

    sr_mw,  st_mw  = _sigma_r_for_galaxy(MW_obj.x, MW_obj.y, MW_obj.z, MW_obj.vx, MW_obj.vy, MW_obj.vz, MW_obj.m, pos_com, vel_com, R_BINS)
    sr_m31, st_m31 = _sigma_r_for_galaxy(M31_obj.x, M31_obj.y, M31_obj.z, M31_obj.vx, M31_obj.vy, M31_obj.vz, M31_obj.m, pos_com, vel_com, R_BINS)

    sigma_r_mw_sep [ii, :] = sr_mw
    sigma_r_m31_sep[ii, :] = sr_m31
    sigma_t_mw_sep [ii, :] = st_mw
    sigma_t_m31_sep[ii, :] = st_m31

    try:
        time_sep[ii] = float(MW_obj.time.value)
    except Exception:
        time_sep[ii] = float(snap_num)

    # Overlap mixing metric: ∫ min(sr_mw, sr_m31) dr / ∫ max(sr_mw, sr_m31) dr
    valid = np.isfinite(sr_mw) & np.isfinite(sr_m31)
    if valid.sum() > 2:
        overlap = np.sum(np.minimum(sr_mw[valid], sr_m31[valid]))
        total   = np.sum(np.maximum(sr_mw[valid], sr_m31[valid]))
        mixing_score[ii] = overlap / total if total > 0 else np.nan

    print(f"  snap {snap_num:04d} | Kinematic mixing overlap score: {mixing_score[ii]:.3f}")

# Re-clean separate workspace
shutil.rmtree(tmpdir_sep, ignore_errors=True)

# Plot MW vs M31 profiles side-by-side
fig17, axes17 = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d0d18", sharey=True, gridspec_kw={"wspace": 0.08})
for ax in axes17:
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log")

early_ii = 0
late_ii  = max(0, n_sep - 1)

for ax, ii, title in zip(axes17, [early_ii, late_ii],
                         [f"Early Stage (snap {separate_snap_nums[early_ii]})", f"Late Stage (snap {separate_snap_nums[late_ii]})"]):
    sr_mw_  = sigma_r_mw_sep [ii, :]
    sr_m31_ = sigma_r_m31_sep[ii, :]

    for y, color, label in [(sr_mw_, "#4a8fff", "MW"), (sr_m31_, "#ff5fa0", "M31")]:
        valid = np.isfinite(y)
        if valid.any():
            ax.plot(r_mid[valid], y[valid], color=color, lw=2.0, label=label)
            ax.fill_between(r_mid[valid], 0, y[valid], alpha=0.1, color=color)

    ax.set_xlabel("r [kpc]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend()

axes17[0].set_ylabel(r"$\sigma_r$ [km s$^{-1}$]", fontsize=10)
fig17.suptitle(r"Radial Velocity Dispersion Profiles: Progenitor $\sigma_r(r)$ Comparison", fontsize=12)
fig17.savefig(os.path.join(OUT_DIR, "kinematics_mw_vs_m31_sigma.png"),
            dpi=300, bbox_inches="tight", facecolor=fig17.get_facecolor())
plt.close(fig17)
print("  Saved: kinematics_mw_vs_m31_sigma.png")

# Mixing score progression figure
fig17b, ax17b = plt.subplots(figsize=(10, 4), facecolor="#0d0d18")
ax17b.set_facecolor("#0d0d18")
ax17b.plot(time_sep, mixing_score, color="#00d4aa", lw=2.0, marker="o", markersize=4, label="Mixing Score")
ax17b.set_ylim(0, 1.05)
ax17b.axhline(1.0, color="#ffffff", lw=0.7, ls="--", alpha=0.4, label="Symmetric Profiles (score = 1)")
ax17b.set_xlabel(time_label, fontsize=10)
ax17b.set_ylabel("Overlap Integrations (0 -> 1)", fontsize=10)
ax17b.set_title("Kinematic Coalescence Score Timeline", fontsize=11)
ax17b.legend()

fig17b.savefig(os.path.join(OUT_DIR, "kinematics_mixing_score.png"),
            dpi=300, bbox_inches="tight", facecolor=fig17b.get_facecolor())
plt.close(fig17b)
print("  Saved: kinematics_mixing_score.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 18 — MP4 RENDERING ANIMATIONS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 18 · Dynamic Motion: Kinematic MP4 Generators")
print("="*80)

import matplotlib.animation as animation

# Subsample rates to optimize render speed
ANIM_STEP    = 4
ANIM_FPS     = 20
ANIM_DPI     = 120
ANIM_BITRATE = 2000

anim_indices = np.arange(0, ns, ANIM_STEP)
n_frames     = len(anim_indices)
print(f"  Rendering {n_frames} movie frames at {ANIM_FPS} fps...")

# ── Movie Panel 1: 4-Panel Profile Render ──
fig18, axes18 = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d0d18",
                             gridspec_kw={"hspace": 0.35, "wspace": 0.32})
axes18 = axes18.flatten()

panel_data = [
    (sigma_r_ts, r"$\sigma_r$ [km s$^{-1}$]",  "#4a8fff",  (0, 350)),
    (sigma_t_ts, r"$\sigma_t$ [km s$^{-1}$]",  "#ff9944",  (0, 350)),
    (beta_ts,    r"$\beta(r)$",                  "#e8673a",  (-1.6, 1.1)),
    (vc_ts,      r"$v_c$ [km s$^{-1}$]",         "#2de8c0",  (0, 400)),
]

lines18 = []
for ax, (arr, ylabel, color, ylims) in zip(axes18, panel_data):
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log")
    ax.set_xlim(R_BINS[0], R_BINS[-1])
    ax.set_ylim(*ylims)
    ax.set_xlabel("r [kpc]", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    y0    = arr[anim_indices[0], :]
    valid = np.isfinite(y0)
    line, = ax.plot(r_mid[valid] if valid.any() else [], y0[valid] if valid.any() else [], color=color, lw=2.0)
    lines18.append((line, arr))

    if "beta" in ylabel.lower():
        ax.axhline(0, color="#555577", lw=0.8, ls="--", alpha=0.7)
    if "v_c" in ylabel:
        ax.axhline(238, color="#ffcc44", lw=0.8, ls=":", alpha=0.6)

title18 = fig18.suptitle("", fontsize=11, color="#c8c8e8")

def _update_profiles(frame_idx):
    snap_idx = anim_indices[frame_idx]
    t_val    = t_axis[snap_idx]
    snap_num = SNAPSHOTS[snap_idx]
    t_str    = f"{t_val:.2f} Gyr" if time_is_gyr else f"snap {snap_num}"
    title18.set_text(f"MW–M31 Kinematic Profile Sequences  ·  {t_str}")

    for (line, arr) in lines18:
        y     = arr[snap_idx, :]
        valid = np.isfinite(y)
        line.set_data(r_mid[valid] if valid.any() else [], y[valid] if valid.any() else [])
    return [ln for (ln, _) in lines18]

ani18 = animation.FuncAnimation(fig18, _update_profiles, frames=n_frames, interval=1000 // ANIM_FPS, blit=True)
writer18 = animation.FFMpegWriter(fps=ANIM_FPS, bitrate=ANIM_BITRATE, metadata=dict(title="MW-M31 Kinematic Profile Sequences"))
out_ani18 = os.path.join(OUT_DIR, "kinematics_profiles_animation.mp4")
ani18.save(out_ani18, writer=writer18, dpi=ANIM_DPI)
plt.close(fig18)
print("  Saved: kinematics_profiles_animation.mp4")

# ── Movie Panel 2: Progressive Heatmap Fill ──
fig18b, ax18b = plt.subplots(figsize=(11, 5), facecolor="#0d0d18")
ax18b.set_facecolor("#0d0d18")

beta_reveal = np.full_like(beta_ts, np.nan)
beta_clipped = np.clip(beta_ts, -1.0, 1.0)

im18b = ax18b.imshow(
    beta_reveal.T, aspect="auto", origin="lower",
    extent=[t_axis[np.isfinite(t_axis)].min() if np.isfinite(t_axis).any() else 0,
            t_axis[np.isfinite(t_axis)].max() if np.isfinite(t_axis).any() else ns,
            R_BINS[0], R_BINS[-1]],
    cmap="bwr", vmin=-1.0, vmax=1.0,
)
ax18b.set_yscale("log")
ax18b.set_xlabel(time_label, fontsize=10)
ax18b.set_ylabel("r [kpc]", fontsize=10)
cb18b = fig18b.colorbar(im18b, ax=ax18b, label=r"$\beta$", pad=0.01)
title18b = ax18b.set_title("", fontsize=11)

def _update_beta_heatmap(frame_idx):
    snap_idx = anim_indices[frame_idx]
    beta_reveal[:snap_idx + 1, :] = beta_clipped[:snap_idx + 1, :]
    im18b.set_data(beta_reveal.T)
    t_val   = t_axis[snap_idx]
    t_str   = f"{t_val:.2f} Gyr" if time_is_gyr else f"snap {SNAPSHOTS[snap_idx]}"
    title18b.set_text(fr"Progressive $\beta(r,t)$ Anisotropy Mapping  ·  {t_str}")
    return [im18b]

ani18b = animation.FuncAnimation(fig18b, _update_beta_heatmap, frames=n_frames, interval=1000 // ANIM_FPS, blit=True)
out_ani18b = os.path.join(OUT_DIR, "kinematics_beta_heatmap_animation.mp4")
ani18b.save(out_ani18b, writer=writer18, dpi=ANIM_DPI)
plt.close(fig18b)
print("  Saved: kinematics_beta_heatmap_animation.mp4")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 19 — PARAMETRIC NFW PROFILE OVERLAYS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 19 · Parametric Fitting: Navarro-Frenk-White Parameters")
print("="*80)

from scipy.optimize import curve_fit

def nfw_enclosed_mass(r, rho_s, r_s):
    """
    NFW cumulative mass function.
    """
    x = r / r_s
    return 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))


def fit_nfw_to_snapshot(menc_row, r_mid_loc, r_bins_loc):
    """
    Fits NFW enclosed mass function to physical cumulative systems profiles.
    Returns convergence status flags and virially derived metrics.
    """
    r_fit = r_bins_loc[1:]
    m_fit = menc_row

    valid = np.isfinite(m_fit) & (m_fit > 0) & (r_fit > 0)
    if valid.sum() < 4:
        return {"success": False, "rho_s": np.nan, "r_s": np.nan,
                "c200": np.nan, "m200": np.nan, "chi2": np.nan}

    r_v = r_fit[valid]
    m_v = m_fit[valid]

    r_s0   = 30.0
    M_tot0 = m_v.max()
    rho_s0 = M_tot0 / (4.0 * np.pi * r_s0**3 * (np.log(2.0) - 0.5))

    try:
        popt, pcov = curve_fit(
            nfw_enclosed_mass, r_v, m_v,
            p0=[rho_s0, r_s0],
            bounds=([1e2, 0.1], [1e16, 500.0]),
            maxfev=5000,
        )
        rho_s_fit, r_s_fit = popt

        # Solve for r_200 (radius enclosing density = 200 * rho_crit)
        # rho_crit(z=0) ≈ 94.7 M_sun kpc^-3
        rho_crit_kpc3 = 9.47e1
        r_test = np.logspace(-1, 3, 1000)
        m_test = nfw_enclosed_mass(r_test, rho_s_fit, r_s_fit)
        mean_density = m_test / (4.0/3.0 * np.pi * r_test**3)
        idx_200 = np.searchsorted(-mean_density, -(200 * rho_crit_kpc3))

        r_200  = r_test[idx_200] if 0 < idx_200 < len(r_test) else np.nan
        m_200  = nfw_enclosed_mass(r_200, rho_s_fit, r_s_fit) if np.isfinite(r_200) else np.nan
        c_200  = r_200 / r_s_fit if np.isfinite(r_200) and r_s_fit > 0 else np.nan

        m_pred = nfw_enclosed_mass(r_v, *popt)
        chi2   = np.sum((m_v - m_pred)**2 / m_pred) / max(1, valid.sum() - 2)

        return {"success": True, "rho_s": rho_s_fit, "r_s": r_s_fit,
                "c200": c_200, "m200": m_200, "chi2": chi2}

    except (RuntimeError, ValueError):
        return {"success": False, "rho_s": np.nan, "r_s": np.nan,
                "c200": np.nan, "m200": np.nan, "chi2": np.nan}


c200_arr  = np.full(ns, np.nan)
r_s_arr   = np.full(ns, np.nan)
m200_arr  = np.full(ns, np.nan)
chi2_arr  = np.full(ns, np.nan)

print("  Fitting NFW boundary templates...")
t0_nfw = time.perf_counter()

for i in range(ns):
    result = fit_nfw_to_snapshot(menc_ts[i, :], r_mid, R_BINS)
    if result["success"]:
        c200_arr [i] = result["c200"]
        r_s_arr  [i] = result["r_s"]
        m200_arr [i] = result["m200"]
        chi2_arr [i] = result["chi2"]

n_fit = np.sum(np.isfinite(c200_arr))
print(f"  Fits completed: {n_fit}/{ns} converged in {time.perf_counter() - t0_nfw:.1f}s")

# Plot fitting trajectories over time
fig19, axes19 = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d0d18", gridspec_kw={"hspace": 0.35, "wspace": 0.32})
axes19 = axes19.flatten()

# (a) c_200 concentration
ax = axes19[0]
ax.set_facecolor("#0d0d18")
ax.plot(t_axis, c200_arr, color="#9b6dff", lw=1.5, label=r"$c_{200}$")
ax.set_xlabel(time_label, fontsize=9); ax.set_ylabel(r"$c_{200}$", fontsize=9)
ax.set_title("Concentration Index", fontsize=10); ax.legend(fontsize=8)

# (b) scale radius r_s
ax = axes19[1]
ax.set_facecolor("#0d0d18")
ax.plot(t_axis, r_s_arr, color="#4a8fff", lw=1.5, label=r"$r_s$")
ax.set_xlabel(time_label, fontsize=9); ax.set_ylabel(r"$r_s$ [kpc]", fontsize=9)
ax.set_title("NFW Scale Radius", fontsize=10); ax.legend(fontsize=8)

# (c) mass parameter m_200
ax = axes19[2]
ax.set_facecolor("#0d0d18")
ax.semilogy(t_axis, m200_arr, color="#2de8c0", lw=1.5, label=r"$M_{200}$")
ax.set_xlabel(time_label, fontsize=9); ax.set_ylabel(r"$M_{200}\ [M_\odot]$", fontsize=9)
ax.set_title("Virial Mass Profile", fontsize=10); ax.legend(fontsize=8)

# (d) fitting chi2 statistics
ax = axes19[3]
ax.set_facecolor("#0d0d18")
ax.semilogy(t_axis, chi2_arr, color="#ff5566", lw=1.2, alpha=0.8, label=r"Reduced $\chi^2$")
ax.axhline(1.0, color="#ffffff", lw=0.7, ls="--", alpha=0.4, label="Reference Limit")
ax.set_xlabel(time_label, fontsize=9); ax.set_ylabel(r"Reduced $\chi^2$", fontsize=9)
ax.set_title("Fit Optimization Residuals", fontsize=10); ax.legend(fontsize=8)

fig19.suptitle("Spherically Enclosed Mass Parameter Fits", fontsize=12)
fig19.savefig(os.path.join(OUT_DIR, "kinematics_nfw_parameters.png"),
            dpi=300, bbox_inches="tight", facecolor=fig19.get_facecolor())
plt.close(fig19)
print("  Saved: kinematics_nfw_parameters.png")

# Overlay NFW curve fits against raw profiles at key epochs
fig19b, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d0d18", sharey=True, gridspec_kw={"wspace": 0.08})
for ax in (axL, axR):
    ax.set_facecolor("#0d0d18")

early_i = profile_snap_indices[0]
late_i  = profile_snap_indices[-1]

for ax, snap_i, title in [(axL, early_i, f"Early Epoch (snap {SNAPSHOTS[early_i]})"),
                          (axR, late_i,  f"Late Epoch (snap {SNAPSHOTS[late_i]})")]:
    ax.set_xscale("log"); ax.set_yscale("log")

    r_outer  = R_BINS[1:]
    m_meas   = menc_ts[snap_i, :]
    valid    = np.isfinite(m_meas) & (m_meas > 0)

    ax.scatter(r_outer[valid], m_meas[valid], color="#aaaacc", s=18, zorder=3, label="Measured $M(<r)$")

    if np.isfinite(c200_arr[snap_i]):
        res   = fit_nfw_to_snapshot(menc_ts[snap_i, :], r_mid, R_BINS)
        r_plt = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), 200)
        m_plt = nfw_enclosed_mass(r_plt, res["rho_s"], res["r_s"])
        ax.plot(r_plt, m_plt, color="#ff9944", lw=2.0, label=fr"NFW Fit ($c_{{200}}$={res['c200']:.1f})")

    ax.set_xlabel("r [kpc]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

axL.set_ylabel(r"$M(<r)\ [M_\odot]$", fontsize=10)
fig19b.suptitle("Cumulative Spherically Enclosed Mass Curve Fits", fontsize=12)
fig19b.savefig(os.path.join(OUT_DIR, "kinematics_nfw_fit_overlay.png"),
            dpi=300, bbox_inches="tight", facecolor=fig19b.get_facecolor())
plt.close(fig19b)
print("  Saved: kinematics_nfw_fit_overlay.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 20 — MANIFEST GENERATION                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

print("\n" + "="*80)
print("  COMPLETE SYSTEM PIPELINE MANIFEST")
print("="*80)
print(f"  {'Filename Identifier':<48} {'Memory Size (MB)':>16}")
print(f"  {'-'*48} {'-'*16}")

total_mb = 0.0
for fn in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, fn)
    mb = os.path.getsize(fp) / 1e6
    total_mb += mb
    suffix = " [Stream mp4]" if fn.endswith(".mp4") else " [Static image]"
    print(f"  {fn:<48} {mb:>11.2f}{suffix}")

print(f"  {'-'*48} {'-'*16}")
print(f"  {'TOTAL WRITE VOLUME':<48} {total_mb:>11.2f}")
print("="*80)
print(f"\n[DONE] Kinematics diagnostic system executed to completion. {len(os.listdir(OUT_DIR))} outputs generated.")
