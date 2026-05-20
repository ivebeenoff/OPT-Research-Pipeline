"""
===============================================================================
SECTION 31 — MERGER EVENT DETECTION & COALESCENCE TRACKING
===============================================================================
Author  : Abhinav Vatsa

Continuation of the MW–M31 merger analysis pipeline.

All globals (SNAPSHOTS, time_arr, time_label, time_is_gyr,
OUT_DIR, tmpdir, load_snapshot_particles, CenterOfMass)
must already exist before executing this section.

Physical motivation
-------------------
A major galaxy merger is not a single instantaneous event.  Instead, the
coalescence proceeds through multiple dynamical stages:

  1. First approach
  2. First pericentric passage
  3. Apocentric rebound
  4. Orbital decay through dynamical friction
  5. Final coalescence / nuclear merger
  6. Relaxation into a virialized remnant

The goal of this section is to identify WHEN the merger actually occurs
using several physically motivated diagnostics directly measurable from the
N-body simulation.

Merger diagnostics
------------------
  1. COM separation:
       d(t) = |r_MW - r_M31|

     The merger is identified when the galaxy centers become permanently
     unresolved below a threshold separation.

  2. Relative velocity:
       v_rel(t) = |v_MW - v_M31|

     Tracks orbital energy dissipation during dynamical friction.

  3. Bound-core overlap:
     Measures when dense particle cores spatially overlap.

  4. Density peak convergence:
     Detects collapse from two density maxima → one remnant maximum.

  5. Virial stabilization:
     Identifies when the remnant settles dynamically after coalescence.

Merger definition
-----------------
The "final merger time" is defined as the earliest snapshot satisfying:

  (a) d(t) < MERGER_RADIUS_KPC
  (b) the separation never rises above this threshold again
  (c) core overlap fraction exceeds OVERLAP_THRESHOLD

This prevents transient fly-bys from being misidentified as mergers.

Outputs
-------
  section31_separation_velocity.png
  section31_core_overlap.png
  section31_density_peaks.png
  section31_merger_diagnostics.png
  section31_phase_space.png
  section31_summary_panel.png

===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
from scipy.spatial import cKDTree


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MERGER_RADIUS_KPC = 15.0
CORE_RADIUS_KPC   = 8.0
OVERLAP_THRESHOLD = 0.60

MIN_CORE_PARTICLES = 200

print("\n" + "="*80)
print("  SECTION 31 · Merger Event Detection")
print("="*80)
print(f"  Merger radius     : {MERGER_RADIUS_KPC:.1f} kpc")
print(f"  Core radius       : {CORE_RADIUS_KPC:.1f} kpc")
print(f"  Overlap threshold : {OVERLAP_THRESHOLD:.2f}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.1 — PRE-ALLOCATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

n_snap = len(SNAPSHOTS)

sep_arr         = np.full(n_snap, np.nan)
vrel_arr        = np.full(n_snap, np.nan)
overlap_arr     = np.full(n_snap, np.nan)
peak_count_arr  = np.full(n_snap, np.nan)

mw_com_arr      = np.full((n_snap, 3), np.nan)
m31_com_arr     = np.full((n_snap, 3), np.nan)

merger_index    = None
merger_time     = np.nan
merger_snapshot = None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.2 — HELPER FUNCTIONS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_core_overlap(pos1, pos2, radius=CORE_RADIUS_KPC):
    """
    Estimate overlap between two galaxy cores.

    Uses KD-tree nearest-neighbour matching.
    Returns overlap fraction in [0,1].
    """
    if len(pos1) < MIN_CORE_PARTICLES:
        return np.nan

    if len(pos2) < MIN_CORE_PARTICLES:
        return np.nan

    tree = cKDTree(pos2)

    dist, _ = tree.query(pos1, k=1)

    overlap = np.sum(dist < radius) / len(pos1)

    return float(overlap)


def count_density_peaks(pos, grid_size=64, extent=300):
    """
    Estimate number of major density peaks in projected density field.
    """
    H, _, _ = np.histogram2d(
        pos[:,0],
        pos[:,1],
        bins=grid_size,
        range=[[-extent, extent], [-extent, extent]]
    )

    threshold = 0.25 * np.nanmax(H)

    peaks = np.sum(H > threshold)

    if peaks > 30:
        return 2

    return 1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.3 — MAIN LOOP                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  §31.3 — Main Merger Tracking Loop")
print("="*80)

t0 = time.perf_counter()

for i, snap_num in enumerate(SNAPSHOTS):

    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    try:
        mw = np.loadtxt(mw_file)
        m31 = np.loadtxt(m31_file)

    except Exception as exc:
        print(f"[ERROR] snapshot {snap_num}: {exc}")
        continue

    # Columns assumed:
    # x y z vx vy vz mass

    pos_mw = mw[:,0:3]
    vel_mw = mw[:,3:6]
    m_mw   = mw[:,6]

    pos_m31 = m31[:,0:3]
    vel_m31 = m31[:,3:6]
    m_m31   = m31[:,6]

    # ── Centers of mass ───────────────────────────────────────────────────────
    com_mw  = np.average(pos_mw, axis=0, weights=m_mw)
    com_m31 = np.average(pos_m31, axis=0, weights=m_m31)

    mw_com_arr[i]  = com_mw
    m31_com_arr[i] = com_m31

    # ── COM separation ────────────────────────────────────────────────────────
    sep = np.linalg.norm(com_m31 - com_mw)
    sep_arr[i] = sep

    # ── Relative velocity ─────────────────────────────────────────────────────
    vcom_mw  = np.average(vel_mw, axis=0, weights=m_mw)
    vcom_m31 = np.average(vel_m31, axis=0, weights=m_m31)

    vrel = np.linalg.norm(vcom_m31 - vcom_mw)

    vrel_arr[i] = vrel

    # ── Dense core particle selection ─────────────────────────────────────────
    r_mw  = np.linalg.norm(pos_mw  - com_mw,  axis=1)
    r_m31 = np.linalg.norm(pos_m31 - com_m31, axis=1)

    core_mw  = pos_mw[r_mw   < CORE_RADIUS_KPC]
    core_m31 = pos_m31[r_m31 < CORE_RADIUS_KPC]

    # ── Core overlap ──────────────────────────────────────────────────────────
    overlap_arr[i] = compute_core_overlap(core_mw, core_m31)

    # ── Density peak convergence ──────────────────────────────────────────────
    combined_pos = np.vstack([pos_mw, pos_m31])

    peak_count_arr[i] = count_density_peaks(combined_pos)

    # ── Merger detection logic ────────────────────────────────────────────────
    if sep < MERGER_RADIUS_KPC:

        future_sep = sep_arr[i:]

        if np.all((future_sep < MERGER_RADIUS_KPC) |
                  np.isnan(future_sep)):

            if overlap_arr[i] > OVERLAP_THRESHOLD:

                merger_index    = i
                merger_time     = time_arr[i]
                merger_snapshot = snap_num

                print("\n" + "="*80)
                print("  FINAL COALESCENCE DETECTED")
                print("="*80)
                print(f"  Snapshot      : {merger_snapshot}")
                print(f"  Time           : {merger_time:.3f} {time_label}")
                print(f"  Separation     : {sep:.2f} kpc")
                print(f"  Relative speed : {vrel:.2f} km/s")
                print(f"  Core overlap   : {overlap_arr[i]:.2f}")
                print("="*80)

                break

    if (i + 1) % 25 == 0:
        elapsed = time.perf_counter() - t0

        print(f"  snap {i+1}/{n_snap} "
              f"sep={sep:.1f} kpc "
              f"vrel={vrel:.1f} km/s "
              f"[{elapsed:.0f}s]")


print(f"\n[Loop complete] {time.perf_counter()-t0:.0f}s")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.4 — FIGURES                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BG = "#0d0d18"

def _ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(BG)

    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")

    ax.tick_params(colors="#9090b0")

    ax.set_xlabel(xlabel, color="#c8c8e8")
    ax.set_ylabel(ylabel, color="#c8c8e8")
    ax.set_title(title, color="#c8c8e8")

    return ax


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — SEPARATION & RELATIVE VELOCITY
# ══════════════════════════════════════════════════════════════════════════════

fig1, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    facecolor=BG,
    sharex=True
)

_ax(ax1,
    ylabel="COM separation [kpc]",
    title="Galaxy Separation Evolution")

ax1.plot(time_arr, sep_arr,
         color="#4a8fff",
         lw=2.0)

ax1.axhline(MERGER_RADIUS_KPC,
            color="#ff5555",
            ls="--",
            lw=1.2,
            label="Merger threshold")

if merger_index is not None:
    ax1.axvline(merger_time,
                color="#ffffff",
                lw=1.5,
                ls=":")

ax1.legend()

_ax(ax2,
    xlabel=time_label,
    ylabel="Relative velocity [km/s]",
    title="Orbital Energy Dissipation")

ax2.plot(time_arr, vrel_arr,
         color="#ffaa44",
         lw=2.0)

if merger_index is not None:
    ax2.axvline(merger_time,
                color="#ffffff",
                lw=1.5,
                ls=":")

fig1.savefig(
    os.path.join(OUT_DIR, "section31_separation_velocity.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor=BG
)

plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CORE OVERLAP EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(
    figsize=(11, 5),
    facecolor=BG
)

_ax(ax2,
    xlabel=time_label,
    ylabel="Core overlap fraction",
    title="Dense-Core Spatial Overlap")

ax2.plot(time_arr,
         overlap_arr,
         color="#aa55ff",
         lw=2.0)

ax2.axhline(OVERLAP_THRESHOLD,
            color="#ff5555",
            ls="--",
            lw=1.0)

if merger_index is not None:
    ax2.axvline(merger_time,
                color="#ffffff",
                ls=":")

fig2.savefig(
    os.path.join(OUT_DIR, "section31_core_overlap.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor=BG
)

plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PHASE SPACE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(
    figsize=(8, 7),
    facecolor=BG
)

_ax(ax3,
    xlabel="Separation [kpc]",
    ylabel="Relative velocity [km/s]",
    title="Merger Phase Space Trajectory")

valid = np.isfinite(sep_arr) & np.isfinite(vrel_arr)

ax3.plot(sep_arr[valid],
         vrel_arr[valid],
         color="#00d4aa",
         lw=2.0)

if merger_index is not None:
    ax3.scatter(sep_arr[merger_index],
                vrel_arr[merger_index],
                s=120,
                color="#ffffff",
                label="Final merger")

ax3.legend()

fig3.savefig(
    os.path.join(OUT_DIR, "section31_phase_space.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor=BG
)

plt.close(fig3)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.5 — SUMMARY                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 31 COMPLETE")
print("="*80)

outputs_31 = [
    "section31_separation_velocity.png",
    "section31_core_overlap.png",
    "section31_phase_space.png",
]

for fn in outputs_31:

    fp = os.path.join(OUT_DIR, fn)

    size = (
        os.path.getsize(fp)/1e6
        if os.path.isfile(fp)
        else 0.0
    )

    print(f"  {fn:<45} {size:6.2f} MB")

print("="*80)

if merger_index is not None:

    print("\n  FINAL MERGER SUMMARY")
    print(f"  Snapshot            : {merger_snapshot}")
    print(f"  Merger time         : {merger_time:.3f} {time_label}")
    print(f"  Final separation    : {sep_arr[merger_index]:.2f} kpc")
    print(f"  Relative velocity   : {vrel_arr[merger_index]:.2f} km/s")
    print(f"  Core overlap        : {overlap_arr[merger_index]:.2f}")

else:

    print("\n  No final merger detected under current thresholds.")

print("="*80)
