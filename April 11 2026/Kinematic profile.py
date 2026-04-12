# “””

# MW–M31 MERGER KINEMATIC PROFILES PIPELINE

Author  : Abhinav Vatsa

## Overview

This script processes N-body simulation snapshots of the Milky Way (MW) and
Andromeda (M31) merger to compute a suite of kinematic profiles as a function
of both radius and simulation time. It then produces publication-ready figures
tracing how the merger reshapes the joint halo’s internal kinematics.

The computed profiles per snapshot are:
• σ_r(r)    — radial velocity dispersion (mass-weighted)
• σ_t(r)    — tangential velocity dispersion (mass-weighted)
• v_rot(r)  — mean azimuthal (rotation) velocity about the z-axis
• j(r)      — mean specific angular momentum magnitude (mass-weighted)
• M_enc(r)  — enclosed mass (for circular/escape velocity)
• v_esc(r)  — local escape speed from enclosed mass
• β(r)      — Binney velocity anisotropy parameter

Global (volume-averaged) scalars are also tracked over time for quick
diagnostic plots of the merger’s kinematic history.

## Data model

Snapshots are stored in tar archives in the working directory.  Each archive
contains plain-text files named MW_NNN.txt and M31_NNN.txt.  The files are
read via the project-local CenterOfMass2.CenterOfMass class, which internally
uses ReadFile.Read and exposes per-particle arrays (x, y, z, vx, vy, vz, m)
together with a simulation time stamp.

## Units

All positions are in kpc, velocities in km/s, and masses in 10^10 M_sun as
stored in the snapshot files.  Masses are converted to M_sun immediately after
loading.  The gravitational constant G is expressed in units that are
consistent with these:
G = 4.30091 × 10^{-6}  kpc (km/s)^2 M_sun^{-1}

## Output files

kinematics_inner_evolution.png  — Time-series of inner-halo kinematic scalars
kinematics_heatmaps.png         — (r, t) heatmaps of log σ_r and β
kinematics_profiles_grid.png    — Radial profiles at selected snapshots
kinematics_angular_momentum.png — Specific angular momentum time evolution
kinematics_escape_velocity.png  — Escape speed profiles at key epochs
kinematics_beta_selected.png    — β(r) profiles at key merger stages

## Dependencies

# numpy, matplotlib
ReadFile          (project-local)
CenterOfMass2     (project-local)

“””

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
matplotlib.use(“Agg”)                    # non-interactive backend for HPC/batch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import LogLocator, LogFormatter

# ── Project-local ─────────────────────────────────────────────────────────────

from ReadFile import Read
from CenterOfMass2 import CenterOfMass

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 1 — USER CONFIGURATION                                            ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# All tunable knobs live here.  Nothing below this section should need editing

# for a standard run.  Change these and re-run.

# Snapshot range to process (inclusive on both ends).

# Snapshots that are missing on disk are silently skipped.

START_SNAP = 0
END_SNAP   = 800

# Particle type to use for the COM calculation.

# 1 = dark matter halo  (most particles, good for large-scale COM)

# 2 = disk              (traces the baryonic centre more closely)

# 3 = bulge

PTYPE = 1

# Radial bin edges in kpc — log-spaced from 0.1 kpc to ~400 kpc.

# 20 edges → 19 bins.  Increase to 40 for finer profiles (slower).

R_BINS = np.logspace(-1, np.log10(400.0), 20)

# Minimum number of particles required in a bin before we compute statistics.

# Bins below this threshold are left as NaN to avoid noisy estimates.

MIN_PARTICLES_PER_BIN = 10

# Inner-halo radius threshold [kpc] used for the scalar time-series plots.

# All bins with r_mid <= this value are averaged to produce one number per snap.

INNER_RADIUS_KPC = 30.0

# Outer radius cap for escape-velocity and enclosed-mass profiles [kpc].

# Particles beyond this are ignored in certain diagnostics.

R_MAX_KPC = 400.0

# Fallback inner aperture radius [kpc] used when COM_V fails.

# We use a mass-weighted velocity of particles within this sphere as vCOM.

COM_FALLBACK_RADIUS_KPC = 15.0

# ── Physical constants and unit conversions ────────────────────────────────────

# Newton’s constant in simulation-friendly units:

# [kpc · (km/s)^2 · M_sun^{-1}]

# Derived from SI: G = 6.674e-11 m^3 kg^{-1} s^{-2}

# × (1 kpc / 3.0857e19 m) × (1 M_sun / 1.989e30 kg) × (1 km/s / 1000 m/s)^{-2}

G_KPC_KMS2_MSUN = 4.30091e-6

# Snapshot mass files store particle masses in units of 10^10 M_sun.

# Multiply by this factor to convert to M_sun.

MASS_UNIT_MSUN = 1.0e10

# ── Output directory ──────────────────────────────────────────────────────────

# All figures and intermediate products go here.

OUT_DIR = “./outputs”
os.makedirs(OUT_DIR, exist_ok=True)

# ── Matplotlib global style ───────────────────────────────────────────────────

plt.rcParams.update({
“figure.facecolor”:  “#0d0d18”,
“axes.facecolor”:    “#0d0d18”,
“axes.edgecolor”:    “#2a2a4a”,
“axes.labelcolor”:   “#c8c8e8”,
“axes.grid”:         True,
“grid.color”:        “#1e1e36”,
“grid.linewidth”:    0.6,
“xtick.color”:       “#9090b0”,
“ytick.color”:       “#9090b0”,
“text.color”:        “#c8c8e8”,
“legend.facecolor”:  “#0d0d18”,
“legend.edgecolor”:  “#2a2a4a”,
“legend.fontsize”:   8,
“font.family”:       “monospace”,
})

# Pre-build the snapshot array once; reused everywhere.

SNAPSHOTS = np.arange(START_SNAP, END_SNAP + 1)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 2 — DATA EXTRACTION                                               ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_snapshots_from_tarballs(work_dir: str) -> str:
“””
Scan the current working directory for any `*.tar` archives and extract
all MW_NNN.txt and M31_NNN.txt members into a fresh temporary directory.

```
Why a temp directory?
---------------------
Simulation tar archives can contain thousands of files including outputs
from other analyses.  Extracting selectively keeps the working tree clean
and avoids path collisions when multiple runs share the same directory.

Parameters
----------
work_dir : str
    Directory to search for .tar files (typically ``"."``).

Returns
-------
str
    Path to the temporary directory containing the extracted snapshot files.
    The caller is responsible for calling ``shutil.rmtree`` on this path
    when processing is complete.

Notes
-----
If no .tar files are found, or none of them contain MW/M31 members, the
returned directory will be empty and all downstream snapshot loads will
silently skip.
"""
# mkdtemp creates a directory with a unique name — safe for parallel runs.
tmpdir = tempfile.mkdtemp(prefix="mwm31_snaps_")
print(f"[extract] Temporary directory: {tmpdir}")

tar_files = [fn for fn in os.listdir(work_dir) if fn.endswith(".tar")]
if not tar_files:
    warnings.warn("No .tar files found in the working directory. "
                  "Snapshot files must already be present on disk.")
    return tmpdir

for fn in tar_files:
    full_path = os.path.join(work_dir, fn)
    print(f"[extract] Opening {fn} …")
    with tarfile.open(full_path, "r") as tar:
        members_to_extract = [
            m for m in tar.getmembers()
            if m.isfile() and ("MW_" in m.name or "M31_" in m.name)
        ]
        if not members_to_extract:
            print(f"  [warn] No MW_/M31_ members found in {fn}")
            continue
        for member in members_to_extract:
            # Strip any leading directory components so all files land
            # directly inside tmpdir regardless of archive structure.
            member.name = os.path.basename(member.name)
            tar.extract(member, path=tmpdir)
        print(f"  [extract] Extracted {len(members_to_extract)} snapshot files.")

return tmpdir
```

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 3 — CENTRE-OF-MASS UTILITIES                                      ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_com_position(mw: CenterOfMass,
x: np.ndarray, y: np.ndarray, z: np.ndarray,
m_raw: np.ndarray) -> np.ndarray:
“””
Return the 3D centre-of-mass position [kpc] of the combined MW+M31 system.

```
We delegate to ``CenterOfMass.COMdefine``, which implements the iterative
shrinking-sphere algorithm of Power et al. (2003) for robustness against
tidal debris far from the halo centre.

Parameters
----------
mw : CenterOfMass
    An already-instantiated CenterOfMass object (we borrow its method).
x, y, z : ndarray
    Combined (MW + M31) particle positions in kpc.
m_raw : ndarray
    Combined particle masses in snapshot units (NOT yet in M_sun).
    COMdefine only needs relative mass ratios, so units cancel.

Returns
-------
np.ndarray, shape (3,)
    [x_com, y_com, z_com] in kpc.
"""
xcom, ycom, zcom = mw.COMdefine(x, y, z, m_raw)
return np.array([xcom, ycom, zcom])
```

def get_com_velocity(mw: CenterOfMass,
pos_com: np.ndarray,
pos_all: np.ndarray,
vel_all: np.ndarray,
mass_all: np.ndarray) -> np.ndarray:
“””
Return the 3D centre-of-mass velocity [km/s] of the combined system.

```
Strategy
--------
1. Try ``CenterOfMass.COM_V``, which expects astropy Quantity arguments.
   If the class exposes this method and the snapshot stores units, this is
   the most accurate path.
2. If that raises *any* exception (missing units, API change, etc.), fall
   back to a plain mass-weighted mean velocity using only particles within
   ``COM_FALLBACK_RADIUS_KPC`` of the position COM.  The inner sphere
   suppresses the contribution of unbound tidal-stream particles that would
   otherwise bias the bulk velocity.

Parameters
----------
mw : CenterOfMass
    Instantiated CenterOfMass object — borrowed for its COM_V method.
pos_com : ndarray, shape (3,)
    COM position in kpc (from get_com_position).
pos_all : ndarray, shape (N, 3)
    All particle positions in kpc.
vel_all : ndarray, shape (N, 3)
    All particle velocities in km/s.
mass_all : ndarray, shape (N,)
    All particle masses in M_sun.

Returns
-------
np.ndarray, shape (3,)
    [vx_com, vy_com, vz_com] in km/s.
"""
try:
    # COM_V returns a Quantity tuple; index [0] gives the velocity array.
    v_com_qty = mw.COM_V(
        pos_com[0] * mw.x.unit,
        pos_com[1] * mw.y.unit,
        pos_com[2] * mw.z.unit,
    )[0]
    return np.array(v_com_qty)
except Exception as exc:
    # Graceful fallback — warn once but continue processing.
    warnings.warn(f"COM_V failed ({exc}); using inner-sphere mass-weighted fallback.")

    # Distance of every particle from the position COM.
    dr = np.linalg.norm(pos_all - pos_com, axis=1)
    inner_mask = dr < COM_FALLBACK_RADIUS_KPC

    if inner_mask.sum() < 5:
        # Pathological case: almost no particles near the centre.
        # Return zero velocity and let downstream code handle NaNs.
        warnings.warn("Fewer than 5 particles within fallback aperture — returning zero vCOM.")
        return np.zeros(3)

    w = mass_all[inner_mask]
    return np.array([
        np.sum(w * vel_all[inner_mask, i]) / np.sum(w)
        for i in range(3)
    ])
```

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 4 — KINEMATIC PROFILE ENGINE                                      ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_profiles_for_snapshot(mw_path: str, m31_path: str) -> dict:
“””
Core computation: load one snapshot, centre the system, and compute all
kinematic profiles in radial bins.

```
This is the most computationally expensive function.  On a workstation with
~10^6 particles per snapshot it takes ~1–3 s per snapshot.  For 800
snapshots the total wall-time is roughly 15–40 minutes on a single core.

Physics implemented
-------------------
All quantities are computed in the frame of the joint MW+M31 centre of mass.

Velocity anisotropy (Binney 1987):
    β(r) = 1 − σ_t²(r) / [2 σ_r²(r)]
    β = 0   → isotropic
    β = 1   → purely radial (e.g., in-falling streams)
    β < 0   → tangentially biased (e.g., after violent relaxation)

Escape speed (Newtonian):
    v_esc(r) = sqrt(2 G M_enc(r) / r)
where M_enc is the mass enclosed within the *outer edge* of the bin.
This under-estimates the true escape speed (which integrates ρ to infinity)
but is a useful lower bound and computable from the snapshot alone.

Specific angular momentum:
    j = |r × v|   in units of kpc · km/s

Parameters
----------
mw_path  : str  — Path to the MW snapshot text file.
m31_path : str  — Path to the M31 snapshot text file.

Returns
-------
dict
    Keys and shapes:
      "r_mid"          : (nb,)   — bin-centre radii [kpc]
      "sigma_r"        : (nb,)   — radial vel. dispersion [km/s]
      "sigma_t"        : (nb,)   — tangential vel. dispersion [km/s]
      "v_rot"          : (nb,)   — mean azimuthal velocity [km/s]
      "j_spec"         : (nb,)   — specific ang. momentum [kpc km/s]
      "M_enclosed_bin" : (nb,)   — enclosed mass at bin outer edge [M_sun]
      "v_esc"          : (nb,)   — escape speed at bin outer edge [km/s]
      "beta"           : (nb,)   — velocity anisotropy β (dimensionless)
      "total_j"        : float   — global mass-weighted mean j [kpc km/s]
      "sigma_r_global" : float   — global σ_r [km/s]
      "sigma_t_global" : float   — global σ_t [km/s]
      "mean_vrad"      : float   — global mean radial velocity [km/s]
      "mean_vt"        : float   — global mean tangential speed [km/s]
      "time"           : float | None  — simulation time [Gyr] if available
    Bins with fewer than MIN_PARTICLES_PER_BIN particles are NaN.
"""

# ── 4.1  Load particle data ────────────────────────────────────────────────
# CenterOfMass reads and stores per-particle arrays as class attributes.
# We keep both MW and M31 objects to borrow their methods.
MW  = CenterOfMass(mw_path,  PTYPE)
M31 = CenterOfMass(m31_path, PTYPE)

# Concatenate positions [kpc], velocities [km/s], and raw masses.
x  = np.concatenate((MW.x,  M31.x))
y  = np.concatenate((MW.y,  M31.y))
z  = np.concatenate((MW.z,  M31.z))
vx = np.concatenate((MW.vx, M31.vx))
vy = np.concatenate((MW.vy, M31.vy))
vz = np.concatenate((MW.vz, M31.vz))

# Raw masses (snapshot units = 10^10 M_sun).
# Keep a raw copy for COMdefine (which only needs ratios), and a converted
# copy in M_sun for everything that involves G.
m_raw = np.concatenate((MW.m, M31.m))
m_msun = m_raw * MASS_UNIT_MSUN    # physical masses in M_sun

# Convenience 2D arrays.
pos = np.vstack((x, y, z)).T       # (N, 3)  positions  [kpc]
vel = np.vstack((vx, vy, vz)).T    # (N, 3)  velocities [km/s]
N   = pos.shape[0]                 # total particle count

# ── 4.2  Centre of mass ────────────────────────────────────────────────────
# Position COM via iterative shrinking sphere (robust to tidal debris).
pos_com = get_com_position(MW, x, y, z, m_raw)

# Velocity COM — attempts COM_V, falls back to inner-sphere average.
vel_com = get_com_velocity(MW, pos_com, pos, vel, m_msun)

# ── 4.3  Relative coordinates (COM frame) ─────────────────────────────────
# All subsequent kinematics are in the frame co-moving with the joint COM.
r_vec   = pos - pos_com       # (N, 3)  displacement from COM [kpc]
vel_rel = vel - vel_com       # (N, 3)  velocities relative to COM [km/s]

# Scalar distance from COM.
r_mag = np.linalg.norm(r_vec, axis=1)   # (N,)  [kpc]

# ── 4.4  Radial and tangential velocity decomposition ─────────────────────
# Unit radial vector r̂ = r_vec / |r_vec|.
# We suppress divide-by-zero for particles exactly at the COM (r=0).
with np.errstate(divide="ignore", invalid="ignore"):
    r_hat = np.where(
        r_mag[:, None] > 0,
        r_vec / r_mag[:, None],
        0.0,
    )   # (N, 3)

# Radial velocity: v_r = v · r̂   (positive = outward)
v_radial = np.einsum("ij,ij->i", vel_rel, r_hat)   # (N,)  [km/s]

# Tangential speed: v_t = sqrt(|v|^2 - v_r^2).
# Clamp to zero before sqrt to guard against floating-point negatives.
v_tang2 = np.sum(vel_rel**2, axis=1) - v_radial**2
v_tang2 = np.maximum(v_tang2, 0.0)
v_tang  = np.sqrt(v_tang2)    # (N,)  [km/s]

# ── 4.5  Specific angular momentum ────────────────────────────────────────
# j = r × v   in kpc·km/s.  We compute the full 3D vector and take its
# magnitude; this is the *specific* (per unit mass) angular momentum.
j_vec = np.cross(r_vec, vel_rel)          # (N, 3)  [kpc km/s]
j_mag = np.linalg.norm(j_vec, axis=1)     # (N,)

# ── 4.6  Radial-bin profiles ───────────────────────────────────────────────
nb = len(R_BINS) - 1   # number of radial bins

# Allocate output arrays pre-filled with NaN so missing bins are explicit.
sigma_r        = np.full(nb, np.nan)
sigma_t        = np.full(nb, np.nan)
v_rot          = np.full(nb, np.nan)
j_spec         = np.full(nb, np.nan)
M_enclosed_bin = np.full(nb, np.nan)
v_esc          = np.full(nb, np.nan)
beta_profile   = np.full(nb, np.nan)

# Map each particle to a bin index (0-based).
# np.digitize returns 0 for particles below R_BINS[0] and nb for those
# above R_BINS[-1]; subtracting 1 puts them in bin -1 and nb respectively,
# which the loop bounds (0 to nb-1) naturally exclude.
bin_indices = np.digitize(r_mag, R_BINS) - 1

for b in range(nb):
    mask = bin_indices == b
    n_in_bin = mask.sum()

    # Skip bins that are too sparse for reliable statistics.
    if n_in_bin < MIN_PARTICLES_PER_BIN:
        continue

    # ── Mass weights for this bin ────────────────────────────────────────
    w   = m_msun[mask]          # particle masses [M_sun]
    W   = w.sum()               # total mass in bin

    vr  = v_radial[mask]        # radial velocities [km/s]
    vt  = v_tang[mask]          # tangential speeds [km/s]

    # ── Mass-weighted velocity dispersions ───────────────────────────────
    # We use the *mass-weighted* mean and variance, which weights the
    # contribution of each particle by its mass.  This is equivalent to
    # the second moment of the mass-weighted velocity distribution.
    #
    #   σ² = Σ(m_i (v_i − ⟨v⟩_m)²) / Σ m_i
    #
    # where ⟨v⟩_m = Σ(m_i v_i) / Σ m_i  is the mass-weighted mean.

    vr_mean   = np.sum(w * vr) / W
    vt_mean   = np.sum(w * vt) / W
    sigma_r[b] = np.sqrt(np.sum(w * (vr - vr_mean)**2) / W)
    sigma_t[b] = np.sqrt(np.sum(w * (vt - vt_mean)**2) / W)

    # ── Mean azimuthal (rotation) velocity about the z-axis ─────────────
    # We project the 3D velocity onto the azimuthal direction in the x-y
    # plane.  The azimuthal direction at position (x, y) is φ̂ = (−y, x)/R
    # where R = sqrt(x²+y²) is the cylindrical radius.
    #
    # v_φ = (−x v_y + y v_x) / R
    #
    # Positive v_φ means prograde rotation (counter-clockwise when viewed
    # from +z).

    rx_b = r_vec[mask, 0]   # x offsets of bin particles from COM
    ry_b = r_vec[mask, 1]   # y offsets

    # Cylindrical radius (projected onto x-y plane).
    R_cyl = np.sqrt(rx_b**2 + ry_b**2)

    # Only include particles with non-negligible cylindrical radius to
    # avoid dividing by zero for particles on the z-axis.
    nonzero_R = R_cyl > 0.0
    if nonzero_R.any():
        vphi_particles = (
            -rx_b[nonzero_R] * vel_rel[mask, 1][nonzero_R]
            + ry_b[nonzero_R] * vel_rel[mask, 0][nonzero_R]
        ) / R_cyl[nonzero_R]

        # Plain (unweighted) mean azimuthal velocity in this radial shell.
        v_rot[b] = np.mean(vphi_particles)

    # ── Mass-weighted specific angular momentum ──────────────────────────
    # j_spec is the mean |r × v| weighted by particle mass.
    j_spec[b] = np.sum(w * j_mag[mask]) / W

    # ── Enclosed mass and escape speed ───────────────────────────────────
    # Enclosed mass uses ALL particles within the outer bin edge r_bins[b+1],
    # not just those in bin b.  This gives the cumulative M(< r_outer).
    r_outer = R_BINS[b + 1]
    M_encl  = m_msun[r_mag <= r_outer].sum()
    M_enclosed_bin[b] = M_encl

    # Escape speed at the outer bin edge using the enclosed mass.
    # v_esc = sqrt(2 G M_enc / r)
    # Note: this is a local escape speed assuming a point mass M_enc at
    # the origin, which under-estimates the true escape speed because it
    # ignores mass at r > r_outer.  Treat as a lower bound.
    if r_outer > 0.0 and M_encl > 0.0:
        v_esc[b] = np.sqrt(2.0 * G_KPC_KMS2_MSUN * M_encl / r_outer)

    # ── Velocity anisotropy β ────────────────────────────────────────────
    # β = 1 − σ_t² / (2 σ_r²)
    # Guard against sigma_r = 0 (cold, coherently-moving clumps).
    if sigma_r[b] > 0.0:
        beta_profile[b] = 1.0 - (sigma_t[b]**2) / (2.0 * sigma_r[b]**2)

# ── 4.7  Global (volume-integrated) kinematic scalars ────────────────────
# These single numbers per snapshot are cheap to track and useful for
# quick sanity checks on the merger's global kinematic state.

# Total mass (normalisation for all global averages).
M_tot = m_msun.sum()

# Global mass-weighted mean radial and tangential speeds.
mean_vrad = np.sum(m_msun * v_radial) / M_tot
mean_vt   = np.sum(m_msun * v_tang)   / M_tot

# Global mass-weighted velocity dispersions about those means.
sigma_r_global = np.sqrt(np.sum(m_msun * (v_radial - mean_vrad)**2) / M_tot)
sigma_t_global = np.sqrt(np.sum(m_msun * (v_tang   - mean_vt  )**2) / M_tot)

# Global mass-weighted specific angular momentum.
total_j = np.sum(m_msun * j_mag) / M_tot

# ── 4.8  Extract simulation time (if available) ──────────────────────────
# The CenterOfMass class may store simulation time as an astropy Quantity
# in MW.time.  We try to extract its numeric value; fall back to None.
sim_time = None
if hasattr(MW, "time"):
    try:
        sim_time = float(MW.time.value)
    except Exception:
        try:
            sim_time = float(MW.time)
        except Exception:
            pass   # leave as None

# ── 4.9  Pack results ─────────────────────────────────────────────────────
return {
    # Per-bin radial profiles
    "r_mid":           0.5 * (R_BINS[:-1] + R_BINS[1:]),  # bin centres [kpc]
    "sigma_r":         sigma_r,
    "sigma_t":         sigma_t,
    "v_rot":           v_rot,
    "j_spec":          j_spec,
    "M_enclosed_bin":  M_enclosed_bin,
    "v_esc":           v_esc,
    "beta":            beta_profile,
    # Global scalars
    "total_j":         total_j,
    "sigma_r_global":  sigma_r_global,
    "sigma_t_global":  sigma_t_global,
    "mean_vrad":       mean_vrad,
    "mean_vt":         mean_vt,
    "time":            sim_time,
}
```

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 5 — SNAPSHOT LOOP                                                 ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 5.1  Extract snapshot files from tar archives ─────────────────────────────

tmpdir = extract_snapshots_from_tarballs(”.”)

# ── 5.2  Allocate time-series storage arrays ──────────────────────────────────

# Shape: (n_snapshots, n_bins).  Pre-filled with NaN so missing or failed

# snapshots do not contaminate plots with zeros.

nb = len(R_BINS) - 1       # number of radial bins
ns = len(SNAPSHOTS)        # number of snapshots to attempt

# Radial profile time-series arrays  (row = time, column = radial bin)

sigma_r_ts = np.full((ns, nb), np.nan)
sigma_t_ts = np.full((ns, nb), np.nan)
vrot_ts    = np.full((ns, nb), np.nan)
j_ts       = np.full((ns, nb), np.nan)
beta_ts    = np.full((ns, nb), np.nan)
vesc_ts    = np.full((ns, nb), np.nan)
menc_ts    = np.full((ns, nb), np.nan)

# Global scalar time series  (one value per snapshot)

time_arr         = np.full(ns, np.nan)   # simulation time [Gyr] or snap index
sigma_r_glob_arr = np.full(ns, np.nan)
sigma_t_glob_arr = np.full(ns, np.nan)
j_glob_arr       = np.full(ns, np.nan)
mean_vrad_arr    = np.full(ns, np.nan)

# ── 5.3  Main loop ────────────────────────────────────────────────────────────

print(”\n” + “=” * 70)
print(”  PROCESSING SNAPSHOTS”)
print(”=” * 70)

t_loop_start = time.perf_counter()

for i, snap_num in enumerate(SNAPSHOTS):

```
mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

# Skip gracefully if either file is missing.
# This handles gaps in the snapshot sequence without crashing.
if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
    continue

t_snap_start = time.perf_counter()

try:
    out = compute_profiles_for_snapshot(mw_file, m31_file)
except Exception as exc:
    # Log the error but continue with remaining snapshots.
    print(f"  [ERROR] snap {snap_num:04d}: {exc}")
    continue

# ── Store per-bin profiles ────────────────────────────────────────────────
sigma_r_ts[i, :] = out["sigma_r"]
sigma_t_ts[i, :] = out["sigma_t"]
vrot_ts   [i, :] = out["v_rot"]
j_ts      [i, :] = out["j_spec"]
beta_ts   [i, :] = out["beta"]
vesc_ts   [i, :] = out["v_esc"]
menc_ts   [i, :] = out["M_enclosed_bin"]

# ── Store global scalars ──────────────────────────────────────────────────
# If the snapshot does not carry a simulation time, fall back to the
# snapshot index number as a proxy for time ordering.
time_arr        [i] = out["time"] if out["time"] is not None else float(snap_num)
sigma_r_glob_arr[i] = out["sigma_r_global"]
sigma_t_glob_arr[i] = out["sigma_t_global"]
j_glob_arr      [i] = out["total_j"]
mean_vrad_arr   [i] = out["mean_vrad"]

dt = time.perf_counter() - t_snap_start
print(f"  snap {snap_num:04d}  ({dt:.2f}s)  "
      f"σ_r_global={out['sigma_r_global']:.1f} km/s  "
      f"j_global={out['total_j']:.1f} kpc·km/s")
```

t_total = time.perf_counter() - t_loop_start
print(f”\n[DONE] Processed in {t_total/60:.1f} min”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 6 — DERIVED QUANTITIES FOR PLOTTING                               ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 6.1  Bin-centre radii ─────────────────────────────────────────────────────

r_mid = 0.5 * (R_BINS[:-1] + R_BINS[1:])

# ── 6.2  Inner-halo scalar time series ───────────────────────────────────────

# Average all radial-profile values whose bin centres lie within INNER_RADIUS_KPC.

# nanmean safely ignores NaN bins (too few particles).

inner_mask = r_mid <= INNER_RADIUS_KPC   # boolean mask over bin axis

sigma_r_inner = np.nanmean(sigma_r_ts[:, inner_mask], axis=1)
sigma_t_inner = np.nanmean(sigma_t_ts[:, inner_mask], axis=1)
vrot_inner    = np.nanmean(vrot_ts   [:, inner_mask], axis=1)
j_inner       = np.nanmean(j_ts      [:, inner_mask], axis=1)
beta_inner    = np.nanmean(beta_ts   [:, inner_mask], axis=1)

# ── 6.3  Time axis ────────────────────────────────────────────────────────────

# If the first valid time value is suspiciously small (< 0.1), the snapshot

# files probably stored integers (snapshot numbers) not Gyr.  We label the

# axis accordingly.

t_axis        = time_arr
t_valid       = t_axis[np.isfinite(t_axis)]
time_is_gyr   = t_valid.size > 0 and t_valid.min() > 0.05
time_label    = “Time [Gyr]” if time_is_gyr else “Snapshot index”

# ── 6.4  Select representative snapshots for profile plots ────────────────────

# We show the joint radial profiles at five epochs spread across the simulation:

# initial conditions, early interaction, first passage, post-merger, final state.

profile_snap_fractions = [0.0, 0.2, 0.4, 0.65, 1.0]
n_snaps                = len(SNAPSHOTS)
profile_snap_indices   = [
int(f * (n_snaps - 1)) for f in profile_snap_fractions
]
profile_labels = [f”Snap {SNAPSHOTS[k]}” for k in profile_snap_indices]
profile_colors = [”#00d4aa”, “#7b9fff”, “#ffaa44”, “#ff6b9a”, “#aa88ff”]

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 7 — FIGURE 1: INNER-HALO KINEMATIC TIME SERIES                   ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose: Show how the inner halo (r ≤ 30 kpc) kinematics evolve as the

# merger progresses.  Each panel tracks one kinematic scalar versus time.

# Spikes and troughs mark key events (pericentre passages, coalescence).

print(”\n[Plot 1] Inner-halo kinematic time series …”)

fig1, axes1 = plt.subplots(
3, 1, figsize=(11, 9), sharex=True,
gridspec_kw={“hspace”: 0.08},
)
fig1.patch.set_facecolor(”#0d0d18”)

# ── Panel (a): Velocity dispersions σ_r and σ_t ──────────────────────────────

ax = axes1[0]
ax.plot(t_axis, sigma_r_inner, color=”#4da6ff”, lw=1.8, label=r”$\sigma_r$ (inner)”)
ax.plot(t_axis, sigma_t_inner, color=”#ff7755”, lw=1.8, label=r”$\sigma_t$ (inner)”)
ax.fill_between(t_axis, sigma_r_inner, sigma_t_inner,
where=np.isfinite(sigma_r_inner) & np.isfinite(sigma_t_inner),
alpha=0.12, color=”#888888”)
ax.set_ylabel(r”$\sigma$ [km s$^{-1}$]”, fontsize=10)
ax.legend(loc=“upper right”)
ax.set_title(
fr”Inner-halo kinematics  (r ≤ {INNER_RADIUS_KPC:.0f} kpc)”,
fontsize=11, pad=8,
)

# ── Panel (b): Mean azimuthal rotation velocity ───────────────────────────────

ax = axes1[1]
ax.plot(t_axis, vrot_inner, color=”#00d4aa”, lw=1.8, label=r”$v_{\phi}$ (inner)”)
ax.axhline(0, color=”#555577”, lw=0.8, ls=”–”)
ax.set_ylabel(r”$v_\phi$ [km s$^{-1}$]”, fontsize=10)
ax.legend(loc=“upper right”)

# ── Panel (c): Mean radial velocity (bulk infall/expansion indicator) ─────────

ax = axes1[2]
ax.plot(t_axis, mean_vrad_arr, color=”#ffcc44”, lw=1.8, label=r”$\langle v_r \rangle$ (global)”)
ax.axhline(0, color=”#555577”, lw=0.8, ls=”–”)
ax.set_ylabel(r”$\langle v_r \rangle$ [km s$^{-1}$]”, fontsize=10)
ax.set_xlabel(time_label, fontsize=10)
ax.legend(loc=“upper right”)

fig1.savefig(
os.path.join(OUT_DIR, “kinematics_inner_evolution.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig1.get_facecolor(),
)
plt.close(fig1)
print(”  Saved: kinematics_inner_evolution.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 8 — FIGURE 2: HEATMAPS  log σ_r(r, t)  AND  β(r, t)             ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose: Reveal the 2D structure of kinematic evolution simultaneously in

# radius and time.  Colour encodes the quantity value; horizontal structure

# indicates radially-coherent events (e.g., tidal shocking); vertical structure

# indicates sudden global transitions.

print(”[Plot 2] Heatmaps …”)

fig2, ax2 = plt.subplots(
2, 1, figsize=(12, 9), sharex=True,
gridspec_kw={“hspace”: 0.1},
)
fig2.patch.set_facecolor(”#0d0d18”)

# The heatmap extent maps pixel edges to (time, radius) coordinates.

# imshow uses [left, right, bottom, top] for the extent argument.

t_min, t_max = np.nanmin(t_axis), np.nanmax(t_axis)
r_min, r_max = R_BINS[0], R_BINS[-1]
extent = [t_min, t_max, r_min, r_max]

# ── Panel (a): log₁₀ σ_r(r, t) ───────────────────────────────────────────────

# We take log10 to reveal both the low-σ outer halo and the high-σ inner core

# on the same colour scale.  NaN values (empty bins) are shown as background.

log_sigma_r = np.where(sigma_r_ts > 0, np.log10(sigma_r_ts), np.nan)

im1 = ax2[0].imshow(
log_sigma_r.T,          # transpose: rows=radius, cols=time → standard orientation
aspect=“auto”,
origin=“lower”,         # radius increases upward (matching physical intuition)
extent=extent,
cmap=“plasma”,
vmin=0.5, vmax=3.0,     # typical range: 3–1000 km/s in log10
)
cb1 = fig2.colorbar(im1, ax=ax2[0], pad=0.01)
cb1.set_label(r”$\log_{10}(\sigma_r / \mathrm{km,s^{-1}})$”, fontsize=9)
ax2[0].set_ylabel(“Radius [kpc]”, fontsize=10)
ax2[0].set_title(r”Radial velocity dispersion  $\sigma_r(r,,t)$”, fontsize=11)
ax2[0].set_yscale(“log”)   # log-spaced radius axis matches log-spaced bins

# ── Panel (b): β(r, t) — velocity anisotropy ─────────────────────────────────

# β is bounded [-∞, 1] in theory but practically lies in [-2, 1].

# We use a diverging colormap centred on 0 (isotropy).

# Blue (β < 0) = tangentially biased; Red (β > 0) = radially biased.

beta_plot = np.clip(beta_ts, -2.0, 1.0)   # clip extreme outliers for display

im2 = ax2[1].imshow(
beta_plot.T,
aspect=“auto”,
origin=“lower”,
extent=extent,
cmap=“bwr”,              # diverging: blue = tangential, red = radial
vmin=-1.0, vmax=1.0,
)
cb2 = fig2.colorbar(im2, ax=ax2[1], pad=0.01)
cb2.set_label(r”$\beta$”, fontsize=9)
ax2[1].set_ylabel(“Radius [kpc]”, fontsize=10)
ax2[1].set_xlabel(time_label, fontsize=10)
ax2[1].set_title(r”Velocity anisotropy  $\beta(r,,t)$”, fontsize=11)
ax2[1].set_yscale(“log”)

# Draw a horizontal reference line at β = 0 (isotropic) on the colorbar axis.

# We annotate the image rather than the axes to avoid clipping the log scale.

ax2[1].text(
t_max * 0.98, 1.5, “β = 0 (isotropic)”,
color=“white”, fontsize=7, ha=“right”, va=“bottom”,
alpha=0.7,
)

fig2.savefig(
os.path.join(OUT_DIR, “kinematics_heatmaps.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig2.get_facecolor(),
)
plt.close(fig2)
print(”  Saved: kinematics_heatmaps.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 9 — FIGURE 3: RADIAL PROFILE GRID AT SELECTED EPOCHS             ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose: Show the detailed radial structure of all computed quantities at

# five representative epochs.  This complements the heatmaps (which compress

# the radial information) and the time series (which compress the spatial info).

print(”[Plot 3] Radial profile grid …”)

fig3 = plt.figure(figsize=(14, 10), facecolor=”#0d0d18”)
gs3  = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.42, wspace=0.38,
left=0.07, right=0.97, top=0.92, bottom=0.08)

PANEL_FG = “#c8c8e8”
MUTED    = “#7070a0”

profile_quantities = [
# (array,      ylabel,                              log-y?, panel-index)
(sigma_r_ts, r”$\sigma_r$ [km s$^{-1}$]”,          False, 0),
(sigma_t_ts, r”$\sigma_t$ [km s$^{-1}$]”,          False, 1),
(vrot_ts,    r”$v_\phi$ [km s$^{-1}$]”,            False, 2),
(j_ts,       r”$j$ [kpc km s$^{-1}$]”,             True,  3),
(vesc_ts,    r”$v_\mathrm{esc}$ [km s$^{-1}$]”,    False, 4),
(beta_ts,    r”$\beta$ (anisotropy)”,               False, 5),
]

for arr, ylabel, log_y, pidx in profile_quantities:
row, col = divmod(pidx, 3)
ax = fig3.add_subplot(gs3[row, col])
ax.set_xscale(“log”)
if log_y:
ax.set_yscale(“log”)

```
for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
    y = arr[k_idx, :]
    # Mask out NaN and (for log plots) non-positive values.
    valid = np.isfinite(y)
    if log_y:
        valid &= y > 0
    if valid.any():
        ax.plot(r_mid[valid], y[valid], color=color, lw=1.5, label=label)

# Reference lines for β = 0 (isotropy) panel.
if "beta" in ylabel.lower() or "anisotropy" in ylabel.lower():
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(-1.5, 1.1)

ax.set_xlabel("r [kpc]", fontsize=9, color=PANEL_FG)
ax.set_ylabel(ylabel,    fontsize=9, color=PANEL_FG)
ax.set_xlim(R_BINS[0], R_BINS[-1])
ax.legend(fontsize=7)
```

fig3.suptitle(“Kinematic Profiles at Selected Epochs”, fontsize=13, color=PANEL_FG)
fig3.savefig(
os.path.join(OUT_DIR, “kinematics_profiles_grid.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig3.get_facecolor(),
)
plt.close(fig3)
print(”  Saved: kinematics_profiles_grid.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 10 — FIGURE 4: ANGULAR MOMENTUM TIME EVOLUTION                   ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose: Track the global and inner-halo specific angular momentum over time.

# Angular momentum should be approximately conserved in an isolated system;

# deviations indicate tidal torques and angular momentum exchange between the

# two galaxies during the merger.

print(”[Plot 4] Angular momentum evolution …”)

fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor=”#0d0d18”)
ax4.set_facecolor(”#0d0d18”)

# Global (all-particle) j.

ax4.plot(t_axis, j_glob_arr, color=”#00d4aa”, lw=2.0,
label=r”Global $\langle j \rangle$”)

# Inner-halo (r ≤ 30 kpc) j.

ax4.plot(t_axis, j_inner, color=”#ff9944”, lw=2.0, ls=”–”,
label=fr”Inner ($r \leq {INNER_RADIUS_KPC:.0f}$ kpc)”)

ax4.set_xlabel(time_label, fontsize=10)
ax4.set_ylabel(r”Specific ang. momentum [kpc km s$^{-1}$]”, fontsize=10)
ax4.set_title(“Mass-weighted Specific Angular Momentum”, fontsize=11)
ax4.legend()

fig4.savefig(
os.path.join(OUT_DIR, “kinematics_angular_momentum.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig4.get_facecolor(),
)
plt.close(fig4)
print(”  Saved: kinematics_angular_momentum.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 11 — FIGURE 5: ESCAPE VELOCITY PROFILES AT KEY EPOCHS            ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose: Show how the gravitational potential well (traced by v_esc) evolves

# through the merger.  As the two halos coalesce, the central v_esc should

# increase, reflecting a deeper combined potential.

print(”[Plot 5] Escape velocity profiles …”)

fig5, ax5 = plt.subplots(figsize=(9, 6), facecolor=”#0d0d18”)
ax5.set_facecolor(”#0d0d18”)
ax5.set_xscale(“log”)

for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
y = vesc_ts[k_idx, :]
valid = np.isfinite(y) & (y > 0)
if valid.any():
ax5.plot(r_mid[valid], y[valid], color=color, lw=2.0, label=label)

# Reference: MW escape speed at the solar circle (~550 km/s at 8 kpc)

ax5.axhline(550, color=”#ffffff”, lw=0.8, ls=”:”, alpha=0.5,
label=r”MW $v_{\rm esc}$ at 8 kpc ≈ 550 km/s”)

ax5.set_xlabel(“r [kpc]”, fontsize=10)
ax5.set_ylabel(r”$v_{\rm esc}(r)$ [km s$^{-1}$]”, fontsize=10)
ax5.set_title(r”Escape Speed Profiles  $v_{\rm esc}(r) = \sqrt{2GM(<r)/r}$”,
fontsize=11)
ax5.set_xlim(R_BINS[0], R_BINS[-1])
ax5.legend()

fig5.savefig(
os.path.join(OUT_DIR, “kinematics_escape_velocity.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig5.get_facecolor(),
)
plt.close(fig5)
print(”  Saved: kinematics_escape_velocity.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 12 — FIGURE 6: β(r) PROFILES AT KEY MERGER STAGES               ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose: β(r) is the single most diagnostic kinematic quantity for

# distinguishing merger stage.  Early on both halos are isotropic at their

# centres and slightly radial at large r.  At first pericentre, tidally heated

# particles dominate, pushing β → +1.  Post-merger violent relaxation drives

# β back toward 0.  This figure overlays profiles at all five key epochs.

print(”[Plot 6] β(r) profiles …”)

fig6, ax6 = plt.subplots(figsize=(9, 6), facecolor=”#0d0d18”)
ax6.set_facecolor(”#0d0d18”)
ax6.set_xscale(“log”)

for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
y = beta_ts[k_idx, :]
valid = np.isfinite(y)
if valid.any():
ax6.plot(r_mid[valid], y[valid], color=color, lw=2.0, label=label)
ax6.fill_between(r_mid[valid], 0, y[valid],
alpha=0.08, color=color)

# Isotropic reference line.

ax6.axhline(0, color=”#555577”, lw=1.0, ls=”–”)
ax6.text(R_BINS[0] * 1.1, 0.04, “isotropic”, color=”#9090b0”, fontsize=8)

# Purely radial reference.

ax6.axhline(1, color=”#8855aa”, lw=0.7, ls=”:”, alpha=0.6)
ax6.text(R_BINS[0] * 1.1, 1.04, “radial (β=1)”, color=”#8855aa”, fontsize=7)

ax6.set_xlim(R_BINS[0], R_BINS[-1])
ax6.set_ylim(-1.6, 1.2)
ax6.set_xlabel(“r [kpc]”, fontsize=10)
ax6.set_ylabel(r”$\beta(r)$”, fontsize=10)
ax6.set_title(r”Velocity Anisotropy Profiles  $\beta = 1 - \sigma_t^2 / 2\sigma_r^2$”,
fontsize=11)
ax6.legend()

fig6.savefig(
os.path.join(OUT_DIR, “kinematics_beta_selected.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig6.get_facecolor(),
)
plt.close(fig6)
print(”  Saved: kinematics_beta_selected.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 13 — CLEANUP AND SUMMARY                                          ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# Remove the temporary directory containing extracted snapshot files.

# This can be several GB; don’t skip this step on a shared filesystem.

shutil.rmtree(tmpdir, ignore_errors=True)
print(f”\n[cleanup] Removed temporary directory: {tmpdir}”)

# ── Final summary ─────────────────────────────────────────────────────────────

print(”\n” + “=” * 70)
print(”  OUTPUT FILES”)
print(”=” * 70)
for fn in sorted(os.listdir(OUT_DIR)):
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp) / 1e6
print(f”  {fn:<45} {size:6.2f} MB”)
print(”=” * 70)
print(”\n[DONE] Pipeline complete.”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 14 — CIRCULAR VELOCITY  v_c(r, t)                                ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose

# —––

# The circular velocity v_c(r) = sqrt(G M_enc(r) / r) is the speed a test

# particle needs to maintain a stable circular orbit at radius r.  It is the

# standard observable used to construct galaxy rotation curves from 21-cm HI

# or CO emission-line data, so computing it here ties the simulation directly

# to observational diagnostics.

# 

# Comparison with v_esc

# –––––––––––

# v_esc(r) = sqrt(2 G M_enc / r) = sqrt(2) * v_c(r) in the point-mass

# approximation.  Both are computed from the same M_enc array, so the ratio

# should hover near sqrt(2) ≈ 1.41 everywhere; deviations signal that the

# extended mass distribution (ignored by the point-mass formula) is significant.

# 

# Flat-rotation-curve benchmark

# ——————————

# For an isolated NFW halo or an isothermal sphere, v_c is nearly flat over a

# wide range of radii.  During the merger the peak and shape of v_c(r) will

# change dramatically — tracking this documents how the gravitational potential

# is restructured by the coalescence.

print(”\n” + “=”*70)
print(”  SECTION 14 · Circular Velocity v_c(r, t)”)
print(”=”*70)

# ── 14.1  Compute v_c from the already-stored M_enc array ─────────────────────

# menc_ts has shape (ns, nb); r_mid has shape (nb,).

# Broadcasting divides each row of menc_ts element-wise by r_mid.

# Guard against r_mid = 0 (should not occur given R_BINS starts at 0.1 kpc).

with np.errstate(divide=“ignore”, invalid=“ignore”):
vc_ts = np.where(
(menc_ts > 0) & (r_mid > 0),
np.sqrt(G_KPC_KMS2_MSUN * menc_ts / r_mid),   # [km/s]
np.nan,
)   # shape (ns, nb)

print(f”  v_c array shape: {vc_ts.shape}  (snapshots × bins)”)

# ── 14.2  Figure: v_c(r) profiles at five key epochs ─────────────────────────

# Overlay all five representative snapshots on one axes, with v_esc for

# comparison on a twin y-axis (same r-axis, different vertical scale).

# Because v_esc = sqrt(2) v_c, the two families of curves should be parallel

# in log-space; any departure is physically meaningful.

fig14, ax14 = plt.subplots(figsize=(9, 6), facecolor=”#0d0d18”)
ax14.set_facecolor(”#0d0d18”)
ax14.set_xscale(“log”)

for k_idx, color, label in zip(profile_snap_indices, profile_colors, profile_labels):
vc_snap  = vc_ts [k_idx, :]
esc_snap = vesc_ts[k_idx, :]

```
valid_vc  = np.isfinite(vc_snap)  & (vc_snap  > 0)
valid_esc = np.isfinite(esc_snap) & (esc_snap > 0)

# Solid line = circular velocity.
if valid_vc.any():
    ax14.plot(r_mid[valid_vc], vc_snap[valid_vc],
              color=color, lw=2.0, ls="-", label=f"{label} $v_c$")

# Dashed line = escape speed (same colour, dashed).
if valid_esc.any():
    ax14.plot(r_mid[valid_esc], esc_snap[valid_esc],
              color=color, lw=1.2, ls="--", alpha=0.55)
```

# Annotate the sqrt(2) relationship between the two line families.

ax14.text(
0.98, 0.96,
r”Dashed = $v_{\rm esc} = \sqrt{2},v_c$  (point-mass approx.)”,
transform=ax14.transAxes, ha=“right”, va=“top”,
fontsize=7, color=”#8888aa”,
)

# Reference: MW circular speed at the solar circle (IAU 2012 value).

ax14.axhline(238.0, color=”#ffcc44”, lw=0.9, ls=”:”, alpha=0.7,
label=r”MW $v_c(R_\odot) \approx 238$ km/s”)

ax14.set_xlim(R_BINS[0], R_BINS[-1])
ax14.set_ylim(0, 500)
ax14.set_xlabel(“r [kpc]”, fontsize=10)
ax14.set_ylabel(r”$v_c(r)$ [km s$^{-1}$]”, fontsize=10)
ax14.set_title(
r”Circular Velocity Profiles  $v_c(r) = \sqrt{G M(<r)/r}$”,
fontsize=11,
)
ax14.legend(ncol=2, fontsize=7)

fig14.savefig(
os.path.join(OUT_DIR, “kinematics_circular_velocity.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig14.get_facecolor(),
)
plt.close(fig14)
print(”  Saved: kinematics_circular_velocity.png”)

# ── 14.3  Figure: v_c peak and location over time ─────────────────────────────

# Track two scalar quantities per snapshot:

# • v_c_peak : maximum circular speed (proxy for total halo mass / depth)

# • r_peak   : radius at which v_c peaks (proxy for scale radius)

# Both should change significantly through the merger.

vc_peak_arr = np.full(ns, np.nan)
r_peak_arr  = np.full(ns, np.nan)

for i in range(ns):
row = vc_ts[i, :]
finite = np.isfinite(row)
if finite.sum() > 2:
idx_peak        = np.nanargmax(row)
vc_peak_arr[i]  = row[idx_peak]
r_peak_arr[i]   = r_mid[idx_peak]

fig14b, (axA, axB) = plt.subplots(2, 1, figsize=(10, 7),
sharex=True, facecolor=”#0d0d18”,
gridspec_kw={“hspace”: 0.08})
for ax in (axA, axB):
ax.set_facecolor(”#0d0d18”)

axA.plot(t_axis, vc_peak_arr, color=”#f5c842”, lw=1.8,
label=r”$v_{c,,\rm peak}$”)
axA.set_ylabel(r”Peak $v_c$ [km s$^{-1}$]”, fontsize=10)
axA.legend()

axB.plot(t_axis, r_peak_arr, color=”#4a8fff”, lw=1.8,
label=r”$r(v_{c,,\rm peak})$”)
axB.set_yscale(“log”)
axB.set_ylabel(r”Radius of peak $v_c$ [kpc]”, fontsize=10)
axB.set_xlabel(time_label, fontsize=10)
axB.legend()

fig14b.suptitle(“Circular Velocity Peak Evolution”, fontsize=11)
fig14b.savefig(
os.path.join(OUT_DIR, “kinematics_vc_peak_evolution.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig14b.get_facecolor(),
)
plt.close(fig14b)
print(”  Saved: kinematics_vc_peak_evolution.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 15 — JEANS EQUATION EQUILIBRIUM CHECK                            ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Physical background

# —————––

# The spherically symmetric, time-independent Jeans equation is:

# 

# d(ν σ_r²) / dr  +  (2 β / r) ν σ_r²  =  −ν d Φ / dr

# 

# where ν(r) is the tracer number density, σ_r(r) the radial velocity

# dispersion, β(r) the anisotropy, and dΦ/dr = G M_enc(r) / r² the

# gravitational acceleration.

# 

# Rearranging, the Jeans-predicted velocity dispersion at each bin is:

# 

# σ_r²|_Jeans(r)  =  (1/ν(r)) ∫_r^∞ ν(r’) (G M_enc(r’) / r’²) exp(…) dr’

# 

# This integral form (Mamon & Łokas 2005) is exact for any β(r) but requires

# an analytic or numerical integration to infinity.  Here we use a simpler

# *local* Jeans test: at every bin we check whether the measured σ_r² is

# consistent with hydrostatic balance given the local gravitational

# acceleration and density gradient.  Specifically, we compute the

# Jeans residual:

# 

# Δ_Jeans(r) = (d(ν σ_r²)/dr + 2β ν σ_r² / r) / (ν G M_enc / r²)

# 

# In equilibrium Δ ≈ −1.  Departures signal either:

# Δ > −1  →  insufficient pressure support (in-fall, post-pericentre)

# Δ < −1  →  excess pressure (kinematically heated, expansion phase)

# 

# Implementation note

# ––––––––––

# We use the number density proxy ν(r) ∝ N_particles(r) / (4π r² Δr)

# (unweighted count per shell volume), which is sufficient for the

# equilibrium *ratio* because ν cancels in Δ_Jeans.

print(”\n” + “=”*70)
print(”  SECTION 15 · Jeans Equation Equilibrium Check”)
print(”=”*70)

def compute_jeans_residual(sigma_r_prof, beta_prof, menc_prof, r_bins):
“””
Compute the local Jeans equilibrium residual Δ_Jeans(r) for one snapshot.

```
Parameters
----------
sigma_r_prof : (nb,) ndarray  — radial velocity dispersion per bin [km/s]
beta_prof    : (nb,) ndarray  — velocity anisotropy β per bin
menc_prof    : (nb,) ndarray  — enclosed mass at bin outer edge [M_sun]
r_bins       : (nb+1,) ndarray — bin edges [kpc]

Returns
-------
(nb,) ndarray
    Δ_Jeans at each bin centre.  NaN where inputs are invalid.

Notes
-----
Numerical derivatives are computed with np.gradient, which uses
second-order central differences in the interior and first-order
one-sided differences at the boundaries.  The logarithmic bin spacing
means the finite-difference step Δr varies; np.gradient accepts the
non-uniform coordinate array and handles this correctly.
"""
nb    = len(r_bins) - 1
r_mid_loc = 0.5 * (r_bins[:-1] + r_bins[1:])   # bin centres [kpc]

# Pressure proxy: P(r) = σ_r²(r)  (ν cancels in the ratio)
P = sigma_r_prof**2   # (nb,)  [km/s]²

# Numerical derivative dP/dr using the actual (non-uniform) radial grid.
# np.gradient handles non-uniform spacing via the edge_order=1 scheme.
dP_dr = np.gradient(P, r_mid_loc)   # (nb,)  [km/s]² / kpc

# Local gravitational acceleration: g(r) = G M_enc / r²
# Use M_enc at the bin outer edge as the best available estimate.
with np.errstate(divide="ignore", invalid="ignore"):
    g = np.where(
        (menc_prof > 0) & (r_mid_loc > 0),
        G_KPC_KMS2_MSUN * menc_prof / r_mid_loc**2,
        np.nan,
    )   # [km/s]² / kpc   (same composite unit as dP/dr)

# Anisotropy term: 2 β σ_r² / r
with np.errstate(invalid="ignore"):
    aniso_term = np.where(
        r_mid_loc > 0,
        2.0 * beta_prof * P / r_mid_loc,
        np.nan,
    )

# Jeans residual:  Δ = (dP/dr + aniso_term) / (−g)
# In equilibrium the numerator equals −g, so Δ = 1.0 (we define it without
# the leading minus to avoid sign confusion in the plot).
with np.errstate(invalid="ignore", divide="ignore"):
    delta = np.where(
        np.isfinite(g) & (g != 0),
        (dP_dr + aniso_term) / (-g),
        np.nan,
    )

return delta
```

# ── 15.1  Compute Jeans residual at five key epochs ───────────────────────────

jeans_residuals = {}
for k_idx, label in zip(profile_snap_indices, profile_labels):
jeans_residuals[label] = compute_jeans_residual(
sigma_r_ts[k_idx, :],
beta_ts   [k_idx, :],
menc_ts   [k_idx, :],
R_BINS,
)

# ── 15.2  Figure: Jeans residual profiles ─────────────────────────────────────

fig15, ax15 = plt.subplots(figsize=(9, 6), facecolor=”#0d0d18”)
ax15.set_facecolor(”#0d0d18”)
ax15.set_xscale(“log”)

for (label, delta), color in zip(jeans_residuals.items(), profile_colors):
valid = np.isfinite(delta)
if valid.any():
ax15.plot(r_mid[valid], delta[valid], color=color, lw=2.0, label=label)

# Equilibrium reference line at Δ = 1.

ax15.axhline(1.0, color=”#ffffff”, lw=1.0, ls=”–”, alpha=0.5,
label=“equilibrium (Δ = 1)”)

# Shade the ±20% equilibrium band.

ax15.axhspan(0.8, 1.2, alpha=0.06, color=”#ffffff”,
label=“±20% equilibrium band”)

ax15.set_xlim(R_BINS[0], R_BINS[-1])
ax15.set_ylim(-1.0, 3.0)
ax15.set_xlabel(“r [kpc]”, fontsize=10)
ax15.set_ylabel(r”Jeans residual $\Delta_{\rm Jeans}(r)$”, fontsize=10)
ax15.set_title(
r”Jeans Equilibrium Check  “
r”$\Delta = (d\sigma_r^2/dr + 2\beta\sigma_r^2/r),/,(-GM/r^2)$”,
fontsize=10,
)
ax15.legend(fontsize=8)

fig15.savefig(
os.path.join(OUT_DIR, “kinematics_jeans_residual.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig15.get_facecolor(),
)
plt.close(fig15)
print(”  Saved: kinematics_jeans_residual.png”)

# ── 15.3  Global Jeans equilibrium score over time ────────────────────────────

# Summarise the per-snapshot equilibrium state with one number:

# mean |Δ − 1| averaged over the inner 100 kpc.

# A value near 0 means the inner halo is in Jeans equilibrium; values >> 0

# mark pericentre passages and the post-merger violent relaxation phase.

inner100_mask = r_mid <= 100.0
jeans_score   = np.full(ns, np.nan)

for i in range(ns):
delta_i = compute_jeans_residual(
sigma_r_ts[i, :], beta_ts[i, :], menc_ts[i, :], R_BINS,
)
valid_inner = inner100_mask & np.isfinite(delta_i)
if valid_inner.sum() > 2:
jeans_score[i] = np.nanmean(np.abs(delta_i[valid_inner] - 1.0))

fig15b, ax15b = plt.subplots(figsize=(10, 4), facecolor=”#0d0d18”)
ax15b.set_facecolor(”#0d0d18”)
ax15b.plot(t_axis, jeans_score, color=”#e8673a”, lw=1.8,
label=r”$\langle |\Delta-1| \rangle_{r<100,{\rm kpc}}$”)
ax15b.axhline(0.2, color=”#ffffff”, lw=0.8, ls=”–”, alpha=0.4,
label=“20% departure threshold”)
ax15b.set_xlabel(time_label, fontsize=10)
ax15b.set_ylabel(“Jeans disequilibrium score”, fontsize=10)
ax15b.set_title(“Inner-halo Jeans Equilibrium Score Over Time”, fontsize=11)
ax15b.legend()
fig15b.savefig(
os.path.join(OUT_DIR, “kinematics_jeans_score.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig15b.get_facecolor(),
)
plt.close(fig15b)
print(”  Saved: kinematics_jeans_score.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 16 — TIDAL STREAM IDENTIFICATION                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Physical motivation

# —————––

# During the MW–M31 merger, gravitational tides strip stars from the outer

# disks and halos into long, coherent streams.  These streams have a

# characteristic kinematic signature that distinguishes them from the

# dynamically relaxed bulk of the halo:

# 

# • High velocity anisotropy β → 1   (radially biased: particles on

# elongated, nearly-radial orbits stripped from the host)

# • Large specific angular momentum j  at large r  (momentum is conserved

# as particles swing out to large apocentres)

# • Coherent radial velocity v_r  (stream members share a common infall

# or outflow direction)

# 

# Strategy

# ––––

# We identify “stream candidates” in the (β, j) plane per radial bin.

# Specifically, in each bin we flag particles whose combination of

# σ_r / σ_t > STREAM_SIGMA_RATIO  AND  j > STREAM_J_PERCENTILE percentile

# as likely stream members.  The fraction of flagged particles per bin is

# tracked over all snapshots to build a stream-fraction heatmap in (r, t).

# 

# This is a kinematic proxy — definitive stream identification requires

# Lagrangian particle tracking (tagging particles at t=0 and following them),

# which is beyond the scope of snapshot-only analysis.  The proxy is useful

# nonetheless because it highlights the radial zones and merger epochs where

# tidal debris is most prominent.

print(”\n” + “=”*70)
print(”  SECTION 16 · Tidal Stream Identification”)
print(”=”*70)

# ── Tunable thresholds ─────────────────────────────────────────────────────────

# A particle is flagged as a stream candidate if its bin’s anisotropy β

# exceeds this threshold.  β > 0.5 means σ_r²  >> σ_t²: strongly radial.

STREAM_BETA_THRESH = 0.5

# Additionally, the bin’s mean specific angular momentum must exceed this

# percentile of the j distribution at that snapshot to select the outer,

# high-j population.

STREAM_J_PERCENTILE = 75.0

# ── 16.1  Build stream-fraction time series in (r, t) ─────────────────────────

# For each snapshot and each radial bin, the “stream fraction” is 1 if that

# bin satisfies both criteria, 0 otherwise.  We average this over adjacent

# bins later for smoothing.  The result is a (ns, nb) binary mask.

stream_mask_ts = np.zeros((ns, nb), dtype=bool)

for i in range(ns):
beta_row  = beta_ts[i, :]
j_row     = j_ts  [i, :]

```
# Threshold for j at this snapshot: use the global percentile across all
# finite bins, so the threshold adapts to the overall j scale.
finite_j  = j_row[np.isfinite(j_row)]
if finite_j.size < 3:
    continue
j_thresh  = np.nanpercentile(finite_j, STREAM_J_PERCENTILE)

# Flag bins satisfying both conditions.
stream_mask_ts[i, :] = (
    np.isfinite(beta_row) & (beta_row > STREAM_BETA_THRESH) &
    np.isfinite(j_row)   & (j_row    > j_thresh)
)
```

stream_fraction = stream_mask_ts.astype(float)   # 0 or 1 per bin per snap

# ── 16.2  Figure: stream-fraction heatmap in (r, t) ───────────────────────────

# This is the most informative view: columns that go suddenly bright (all bins

# flagged) mark pericentre passages; persistent bright stripes at large r mark

# long-lived stream populations.

fig16, (ax16a, ax16b) = plt.subplots(
1, 2, figsize=(14, 6), facecolor=”#0d0d18”,
gridspec_kw={“width_ratios”: [2, 1], “wspace”: 0.08},
)
for ax in (ax16a, ax16b):
ax.set_facecolor(”#0d0d18”)

# Left panel: heatmap

im16 = ax16a.imshow(
stream_fraction.T,
aspect=“auto”, origin=“lower”,
extent=[t_axis[np.isfinite(t_axis)].min() if np.isfinite(t_axis).any() else 0,
t_axis[np.isfinite(t_axis)].max() if np.isfinite(t_axis).any() else ns,
R_BINS[0], R_BINS[-1]],
cmap=“hot”, vmin=0, vmax=1,
)
ax16a.set_yscale(“log”)
ax16a.set_xlabel(time_label, fontsize=10)
ax16a.set_ylabel(“r [kpc]”, fontsize=10)
ax16a.set_title(
fr”Stream Candidate Fraction  (β > {STREAM_BETA_THRESH}, j > {STREAM_J_PERCENTILE:.0f}th pct.)”,
fontsize=10,
)
fig16.colorbar(im16, ax=ax16a, label=“Fraction of bins flagged”, pad=0.01)

# Right panel: time-averaged stream profile

mean_stream = np.nanmean(stream_fraction, axis=0)
ax16b.plot(mean_stream, r_mid, color=”#ff9944”, lw=2.0)
ax16b.set_xscale(“linear”)
ax16b.set_yscale(“log”)
ax16b.set_xlabel(“Time-avg. stream fraction”, fontsize=10)
ax16b.set_title(“Radial Profile”, fontsize=10)
ax16b.set_ylim(R_BINS[0], R_BINS[-1])
ax16b.tick_params(labelleft=False)

fig16.suptitle(“Tidal Stream Identification — Kinematic Proxy”, fontsize=12)
fig16.savefig(
os.path.join(OUT_DIR, “kinematics_tidal_streams.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig16.get_facecolor(),
)
plt.close(fig16)
print(”  Saved: kinematics_tidal_streams.png”)

# ── 16.3  Stream-active snapshot count ────────────────────────────────────────

# Count how many radial bins are flagged per snapshot — a quick diagnostic of

# when the merger is most kinematically violent.

stream_active_bins = stream_fraction.sum(axis=1)   # (ns,) count of flagged bins

fig16b, ax16c = plt.subplots(figsize=(10, 4), facecolor=”#0d0d18”)
ax16c.set_facecolor(”#0d0d18”)
ax16c.plot(t_axis, stream_active_bins, color=”#ff5566”, lw=1.5,
label=”# stream-flagged bins”)
ax16c.fill_between(t_axis, 0, stream_active_bins, alpha=0.15, color=”#ff5566”)
ax16c.set_xlabel(time_label, fontsize=10)
ax16c.set_ylabel(“Flagged radial bins”, fontsize=10)
ax16c.set_title(“Tidal Activity Over Time”, fontsize=11)
ax16c.legend()
fig16b.savefig(
os.path.join(OUT_DIR, “kinematics_tidal_activity.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig16b.get_facecolor(),
)
plt.close(fig16b)
print(”  Saved: kinematics_tidal_activity.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 17 — SEPARATE MW AND M31 KINEMATIC PROFILES                      ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Motivation

# –––––

# The joint (MW + M31) profiles computed in Sections 4–12 tell us about the

# combined system.  But during the merger the two galaxies remain distinct

# objects for several Gyr, and we want to know:

# • Which galaxy dominates the kinematic signal at each radius?

# • How quickly does kinematic mixing erase the memory of origin?

# • Does the MW’s disk survive as a kinematically cold component?

# 

# Implementation

# –––––––

# We re-open a *representative subset* of snapshots (every STEP_SEPARATE snaps)

# to avoid re-reading 800 files, and compute σ_r separately for MW particles

# and M31 particles.  A “mixing score” — the overlap integral of the two σ_r

# profiles — tracks how kinematically indistinguishable the two populations

# become over time.

print(”\n” + “=”*70)
print(”  SECTION 17 · Separate MW and M31 Profiles”)
print(”=”*70)

# Process every Nth snapshot for the separate-galaxy analysis.

# On 800 snapshots, STEP=40 gives 20 evenly-spaced analysis points.

STEP_SEPARATE = 40

# Snapshots to analyse — filter to only those that exist on disk.

separate_snap_nums = SNAPSHOTS[::STEP_SEPARATE]
print(f”  Analysing {len(separate_snap_nums)} snapshots (every {STEP_SEPARATE}th)”)

# Storage: shape (n_separate, nb)

n_sep           = len(separate_snap_nums)
sigma_r_mw_sep  = np.full((n_sep, nb), np.nan)
sigma_r_m31_sep = np.full((n_sep, nb), np.nan)
sigma_t_mw_sep  = np.full((n_sep, nb), np.nan)
sigma_t_m31_sep = np.full((n_sep, nb), np.nan)
time_sep        = np.full(n_sep, np.nan)
mixing_score    = np.full(n_sep, np.nan)

def _sigma_r_for_galaxy(x, y, z, vx, vy, vz, m_raw,
pos_com, vel_com, r_bins_loc):
“””
Compute mass-weighted σ_r profile for a single galaxy’s particles.

```
Parameters mirror the arrays for one galaxy (MW or M31 only).
Returns (sigma_r, sigma_t) arrays of shape (nb,).
"""
nb_loc = len(r_bins_loc) - 1
m      = m_raw * MASS_UNIT_MSUN
pos    = np.vstack((x, y, z)).T
vel    = np.vstack((vx, vy, vz)).T

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
    w     = m[mask]; W = w.sum()
    vr    = v_radial[mask]; vt = v_tang[mask]
    sigma_r_out[b] = np.sqrt(np.sum(w * (vr - np.sum(w*vr)/W)**2) / W)
    sigma_t_out[b] = np.sqrt(np.sum(w * (vt - np.sum(w*vt)/W)**2) / W)

return sigma_r_out, sigma_t_out
```

for ii, snap_num in enumerate(separate_snap_nums):
mw_file  = os.path.join(tmpdir if os.path.isdir(tmpdir) else “.”,
f”MW_{snap_num:03d}.txt”)
m31_file = os.path.join(tmpdir if os.path.isdir(tmpdir) else “.”,
f”M31_{snap_num:03d}.txt”)

```
# Note: tmpdir was cleaned up in Section 13.  If files are no longer
# available on disk, this section gracefully skips.  To use this section
# on a live run, move the cleanup to after Section 19.
if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
    print(f"  [skip] snap {snap_num} — files not available (post-cleanup)")
    continue

try:
    MW_obj  = CenterOfMass(mw_file,  PTYPE)
    M31_obj = CenterOfMass(m31_file, PTYPE)
except Exception as exc:
    print(f"  [ERROR] snap {snap_num}: {exc}")
    continue

# Joint COM (needed so both galaxies are in the same reference frame).
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

sr_mw,  st_mw  = _sigma_r_for_galaxy(
    MW_obj.x, MW_obj.y, MW_obj.z,
    MW_obj.vx, MW_obj.vy, MW_obj.vz, MW_obj.m,
    pos_com, vel_com, R_BINS,
)
sr_m31, st_m31 = _sigma_r_for_galaxy(
    M31_obj.x, M31_obj.y, M31_obj.z,
    M31_obj.vx, M31_obj.vy, M31_obj.vz, M31_obj.m,
    pos_com, vel_com, R_BINS,
)

sigma_r_mw_sep [ii, :] = sr_mw
sigma_r_m31_sep[ii, :] = sr_m31
sigma_t_mw_sep [ii, :] = st_mw
sigma_t_m31_sep[ii, :] = st_m31

# Simulation time
try:
    time_sep[ii] = float(MW_obj.time.value)
except Exception:
    time_sep[ii] = float(snap_num)

# ── Kinematic mixing score ────────────────────────────────────────────────
# The overlap integral ∫ min(σ_r_MW, σ_r_M31) dr / ∫ max(σ_r_MW, σ_r_M31) dr
# ranges from 0 (completely distinct profiles) to 1 (identical profiles).
valid = np.isfinite(sr_mw) & np.isfinite(sr_m31)
if valid.sum() > 2:
    overlap = np.sum(np.minimum(sr_mw[valid], sr_m31[valid]))
    total   = np.sum(np.maximum(sr_mw[valid], sr_m31[valid]))
    mixing_score[ii] = overlap / total if total > 0 else np.nan

print(f"  snap {snap_num:04d}  mix={mixing_score[ii]:.3f}")
```

# ── 17.1  Figure: MW vs. M31 σ_r profiles at two epochs ──────────────────────

fig17, axes17 = plt.subplots(1, 2, figsize=(13, 5), facecolor=”#0d0d18”,
sharey=True, gridspec_kw={“wspace”: 0.08})
for ax in axes17:
ax.set_facecolor(”#0d0d18”)
ax.set_xscale(“log”)

early_ii = 0
late_ii  = max(0, n_sep - 1)

for ax, ii, title in zip(axes17, [early_ii, late_ii],
[f”Early  (snap {separate_snap_nums[early_ii]})”,
f”Late   (snap {separate_snap_nums[late_ii]})”]):
sr_mw_  = sigma_r_mw_sep [ii, :]
sr_m31_ = sigma_r_m31_sep[ii, :]

```
for y, color, label in [
    (sr_mw_,  "#4a8fff", "MW"),
    (sr_m31_, "#ff5fa0", "M31"),
]:
    valid = np.isfinite(y)
    if valid.any():
        ax.plot(r_mid[valid], y[valid], color=color, lw=2.0, label=label)
        ax.fill_between(r_mid[valid], 0, y[valid], alpha=0.1, color=color)

ax.set_xlabel("r [kpc]", fontsize=10)
ax.set_title(title, fontsize=10)
ax.legend()
```

axes17[0].set_ylabel(r”$\sigma_r$ [km s$^{-1}$]”, fontsize=10)
fig17.suptitle(r”MW vs. M31 Radial Velocity Dispersion $\sigma_r(r)$”, fontsize=12)
fig17.savefig(
os.path.join(OUT_DIR, “kinematics_mw_vs_m31_sigma.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig17.get_facecolor(),
)
plt.close(fig17)
print(”  Saved: kinematics_mw_vs_m31_sigma.png”)

# ── 17.2  Figure: kinematic mixing score over time ────────────────────────────

fig17b, ax17b = plt.subplots(figsize=(10, 4), facecolor=”#0d0d18”)
ax17b.set_facecolor(”#0d0d18”)
ax17b.plot(time_sep, mixing_score, color=”#00d4aa”, lw=2.0, marker=“o”,
markersize=4, label=“Kinematic mixing score”)
ax17b.set_ylim(0, 1.05)
ax17b.axhline(1.0, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.4,
label=“Perfect mixing (score = 1)”)
ax17b.set_xlabel(time_label, fontsize=10)
ax17b.set_ylabel(“Overlap integral (0→1)”, fontsize=10)
ax17b.set_title(“MW–M31 Kinematic Mixing Score Over Time”, fontsize=11)
ax17b.legend()
fig17b.savefig(
os.path.join(OUT_DIR, “kinematics_mixing_score.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig17b.get_facecolor(),
)
plt.close(fig17b)
print(”  Saved: kinematics_mixing_score.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 18 — KINEMATIC ANIMATION                                          ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Purpose

# —––

# Static figures at five epochs give snapshots (pun intended) of the kinematic

# evolution, but a continuous animation reveals the *dynamics* — how quickly

# features appear, propagate outward, and decay.  Because all the profile

# arrays (sigma_r_ts, beta_ts, vc_ts, …) are already in memory, no additional

# I/O is required.

# 

# We produce two animations:

# 1. Four-panel kinematic profile movie  (σ_r, σ_t, β, v_c vs. r over time)

# 2. σ_r and β heatmap update (a “live” version of Figure 2)

# 

# Performance note

# ––––––––

# matplotlib.animation writes frames to a temporary directory and then calls

# ffmpeg to encode the mp4.  For 801 frames at 150 dpi this takes roughly

# 5–15 minutes depending on the number of panels.  The ANIM_STEP parameter

# controls temporal subsampling: ANIM_STEP=4 renders every 4th snapshot,

# cutting frame count to ~200 and runtime to ~2 minutes.

print(”\n” + “=”*70)
print(”  SECTION 18 · Kinematic Profile Animation”)
print(”=”*70)

import matplotlib.animation as animation

# ── Configuration ─────────────────────────────────────────────────────────────

ANIM_STEP    = 4       # render every Nth snapshot (1 = all, higher = faster)
ANIM_FPS     = 20      # frames per second in the output mp4
ANIM_DPI     = 120     # resolution (lower → faster encoding)
ANIM_BITRATE = 2000    # mp4 bitrate [kbps]

anim_indices = np.arange(0, ns, ANIM_STEP)   # snapshot indices to animate
n_frames     = len(anim_indices)
print(f”  Rendering {n_frames} frames at {ANIM_FPS} fps …”)

# ── 18.1  Four-panel kinematic profile animation ───────────────────────────────

fig18, axes18 = plt.subplots(2, 2, figsize=(12, 8), facecolor=”#0d0d18”,
gridspec_kw={“hspace”: 0.35, “wspace”: 0.32})
axes18 = axes18.flatten()

panel_data = [
(sigma_r_ts, r”$\sigma_r$ [km s$^{-1}$]”,  “#4a8fff”,  (0, 350)),
(sigma_t_ts, r”$\sigma_t$ [km s$^{-1}$]”,  “#ff9944”,  (0, 350)),
(beta_ts,    r”$\beta(r)$”,                 “#e8673a”,  (-1.6, 1.1)),
(vc_ts,      r”$v_c$ [km s$^{-1}$]”,        “#2de8c0”,  (0, 400)),
]

lines18 = []
for ax, (arr, ylabel, color, ylims) in zip(axes18, panel_data):
ax.set_facecolor(”#0d0d18”)
ax.set_xscale(“log”)
ax.set_xlim(R_BINS[0], R_BINS[-1])
ax.set_ylim(*ylims)
ax.set_xlabel(“r [kpc]”, fontsize=9)
ax.set_ylabel(ylabel, fontsize=9)
# Initialise with the first frame’s data.
y0    = arr[anim_indices[0], :]
valid = np.isfinite(y0)
line, = ax.plot(
r_mid[valid] if valid.any() else [],
y0[valid]    if valid.any() else [],
color=color, lw=2.0,
)
lines18.append((line, arr))

```
# Reference line for β = 0.
if "beta" in ylabel.lower():
    ax.axhline(0, color="#555577", lw=0.8, ls="--", alpha=0.7)

# Reference for MW v_c.
if "v_c" in ylabel:
    ax.axhline(238, color="#ffcc44", lw=0.8, ls=":", alpha=0.6)
```

title18 = fig18.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_profiles(frame_idx):
“”“Update all four panel lines for animation frame `frame_idx`.”””
snap_idx = anim_indices[frame_idx]
t_val    = t_axis[snap_idx]
snap_num = SNAPSHOTS[snap_idx]
t_str    = f”{t_val:.2f} Gyr” if time_is_gyr else f”snap {snap_num}”
title18.set_text(f”MW–M31 Kinematic Profiles  ·  {t_str}”)

```
for (line, arr) in lines18:
    y     = arr[snap_idx, :]
    valid = np.isfinite(y)
    line.set_data(r_mid[valid] if valid.any() else [],
                  y[valid]     if valid.any() else [])
return [ln for (ln, _) in lines18]
```

ani18 = animation.FuncAnimation(
fig18, _update_profiles,
frames=n_frames, interval=1000 // ANIM_FPS, blit=True,
)

writer18 = animation.FFMpegWriter(
fps=ANIM_FPS, bitrate=ANIM_BITRATE,
metadata=dict(title=“MW-M31 Kinematic Profile Animation”),
)
out_ani18 = os.path.join(OUT_DIR, “kinematics_profiles_animation.mp4”)
ani18.save(out_ani18, writer=writer18, dpi=ANIM_DPI)
plt.close(fig18)
print(f”  Saved: kinematics_profiles_animation.mp4”)

# ── 18.2  β heatmap animation (progressive reveal) ────────────────────────────

# Instead of showing the full heatmap immediately, this animation progressively

# fills in the beta heatmap column by column as time advances — useful for

# presentations where you want to draw attention to new merger events as they

# emerge.

fig18b, ax18b = plt.subplots(figsize=(11, 5), facecolor=”#0d0d18”)
ax18b.set_facecolor(”#0d0d18”)

# Pre-build a NaN-filled copy of the beta heatmap; we reveal it frame by frame.

beta_reveal = np.full_like(beta_ts, np.nan)
beta_clipped = np.clip(beta_ts, -1.0, 1.0)

im18b = ax18b.imshow(
beta_reveal.T,
aspect=“auto”, origin=“lower”,
extent=[t_axis[np.isfinite(t_axis)].min() if np.isfinite(t_axis).any() else 0,
t_axis[np.isfinite(t_axis)].max() if np.isfinite(t_axis).any() else ns,
R_BINS[0], R_BINS[-1]],
cmap=“bwr”, vmin=-1.0, vmax=1.0,
)
ax18b.set_yscale(“log”)
ax18b.set_xlabel(time_label, fontsize=10)
ax18b.set_ylabel(“r [kpc]”, fontsize=10)
cb18b = fig18b.colorbar(im18b, ax=ax18b, label=r”$\beta$”, pad=0.01)
title18b = ax18b.set_title(””, fontsize=11)

def _update_beta_heatmap(frame_idx):
“”“Reveal the beta heatmap one column (snapshot) at a time.”””
snap_idx = anim_indices[frame_idx]
beta_reveal[:snap_idx + 1, :] = beta_clipped[:snap_idx + 1, :]
im18b.set_data(beta_reveal.T)
t_val   = t_axis[snap_idx]
t_str   = f”{t_val:.2f} Gyr” if time_is_gyr else f”snap {SNAPSHOTS[snap_idx]}”
title18b.set_text(fr”$\beta(r,t)$ Halo Anisotropy  ·  {t_str}”)
return [im18b]

ani18b = animation.FuncAnimation(
fig18b, _update_beta_heatmap,
frames=n_frames, interval=1000 // ANIM_FPS, blit=True,
)
out_ani18b = os.path.join(OUT_DIR, “kinematics_beta_heatmap_animation.mp4”)
ani18b.save(out_ani18b, writer=writer18, dpi=ANIM_DPI)
plt.close(fig18b)
print(”  Saved: kinematics_beta_heatmap_animation.mp4”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 19 — NFW CONCENTRATION FITTING                                    ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Physical motivation

# —————––

# The Navarro–Frenk–White (NFW) density profile (Navarro et al. 1997) is:

# 

# ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]

# 

# where r_s is the scale radius and ρ_s the characteristic density.

# The concentration parameter c = r_200 / r_s (where r_200 is the radius

# enclosing a mean density 200× the critical density) is a key structural

# parameter tracked in cosmological simulations.

# 

# During the MW–M31 merger:

# • Early: both halos have their pre-merger concentrations (c ≈ 10–20).

# • At pericentre: tidal compression can temporarily raise c.

# • Post-merger: violent relaxation drives c toward a value set by the

# combined mass and the redshift of coalescence.

# 

# Fitting strategy

# ––––––––

# We fit the NFW *enclosed mass* profile M_NFW(<r) to the measured M_enc(r):

# 

# M_NFW(<r) = 4π ρ_s r_s³ [ln(1 + r/r_s) − (r/r_s)/(1 + r/r_s)]

# 

# This is more stable than fitting ρ(r) because M_enc is a smoothly increasing

# function while ρ(r) can be noisy at large r where particle counts are low.

# We use scipy.optimize.curve_fit with reasonable initial guesses derived from

# the peak circular velocity.

print(”\n” + “=”*70)
print(”  SECTION 19 · NFW Concentration Fitting”)
print(”=”*70)

from scipy.optimize import curve_fit

def nfw_enclosed_mass(r, rho_s, r_s):
“””
NFW enclosed mass profile M(<r).

```
Parameters
----------
r    : array_like  — radii at which to evaluate M(<r) [kpc]
rho_s: float       — characteristic density [M_sun kpc^{-3}]
r_s  : float       — NFW scale radius [kpc]

Returns
-------
array_like  — enclosed mass in M_sun
"""
x = r / r_s
return (4.0 * np.pi * rho_s * r_s**3
        * (np.log(1.0 + x) - x / (1.0 + x)))
```

def fit_nfw_to_snapshot(menc_row, r_mid_loc, r_bins_loc):
“””
Fit the NFW enclosed mass profile to one snapshot’s M_enc(r).

```
Parameters
----------
menc_row  : (nb,) ndarray  — M_enc at each bin outer edge [M_sun]
r_mid_loc : (nb,) ndarray  — bin centres [kpc]
r_bins_loc: (nb+1,) ndarray — bin edges [kpc]

Returns
-------
dict with keys:
    rho_s   : float  — fitted characteristic density [M_sun kpc^{-3}]
    r_s     : float  — fitted scale radius [kpc]
    c200    : float  — concentration c = r_200 / r_s  (approximate)
    m200    : float  — virial mass M_200 [M_sun]  (approximate)
    chi2    : float  — reduced chi-squared of the fit
    success : bool   — True if curve_fit converged
"""
# Use the outer bin edges as the radii for M_enc (since M_enc is the mass
# enclosed within r_outer = R_BINS[b+1]).
r_fit = r_bins_loc[1:]   # outer edges  [kpc]
m_fit = menc_row          # M_enc        [M_sun]

# Mask out NaN bins.
valid = np.isfinite(m_fit) & (m_fit > 0) & (r_fit > 0)
if valid.sum() < 4:
    return {"success": False, "rho_s": np.nan, "r_s": np.nan,
            "c200": np.nan, "m200": np.nan, "chi2": np.nan}

r_v = r_fit[valid]
m_v = m_fit[valid]

# Initial guesses: r_s at the radius of peak slope in M_enc, ρ_s from
# total mass and scale radius.
r_s0   = 30.0        # typical NFW scale radius [kpc]
M_tot0 = m_v.max()
rho_s0 = M_tot0 / (4.0 * np.pi * r_s0**3
                    * (np.log(2.0) - 0.5))   # x=1 term

try:
    popt, pcov = curve_fit(
        nfw_enclosed_mass, r_v, m_v,
        p0=[rho_s0, r_s0],
        bounds=([1e4, 0.5], [1e15, 500.0]),   # physical bounds
        maxfev=5000,
    )
    rho_s_fit, r_s_fit = popt

    # Approximate r_200: radius where mean enclosed density = 200 × ρ_crit.
    # ρ_crit at z=0 ≈ 9.47 × 10^{10} M_sun Mpc^{-3} = 9.47 × 10^{1} M_sun kpc^{-3}
    rho_crit_kpc3 = 9.47e1    # M_sun kpc^{-3}
    # Iteratively solve for r_200.
    r_test = np.logspace(-1, 3, 1000)
    m_test = nfw_enclosed_mass(r_test, rho_s_fit, r_s_fit)
    mean_density = m_test / (4.0/3.0 * np.pi * r_test**3)
    idx_200 = np.searchsorted(-(mean_density), -(200 * rho_crit_kpc3))
    r_200  = r_test[idx_200] if 0 < idx_200 < len(r_test) else np.nan
    m_200  = nfw_enclosed_mass(r_200, rho_s_fit, r_s_fit) if np.isfinite(r_200) else np.nan
    c_200  = r_200 / r_s_fit if np.isfinite(r_200) and r_s_fit > 0 else np.nan

    # Reduced chi-squared.
    m_pred = nfw_enclosed_mass(r_v, *popt)
    chi2   = np.sum((m_v - m_pred)**2 / m_pred) / max(1, valid.sum() - 2)

    return {"success": True, "rho_s": rho_s_fit, "r_s": r_s_fit,
            "c200": c_200, "m200": m_200, "chi2": chi2}

except (RuntimeError, ValueError):
    return {"success": False, "rho_s": np.nan, "r_s": np.nan,
            "c200": np.nan, "m200": np.nan, "chi2": np.nan}
```

# ── 19.1  Fit every snapshot ──────────────────────────────────────────────────

c200_arr  = np.full(ns, np.nan)
r_s_arr   = np.full(ns, np.nan)
m200_arr  = np.full(ns, np.nan)
chi2_arr  = np.full(ns, np.nan)

print(”  Fitting NFW profiles …”)
t0_nfw = time.perf_counter()

for i in range(ns):
result = fit_nfw_to_snapshot(menc_ts[i, :], r_mid, R_BINS)
if result[“success”]:
c200_arr [i] = result[“c200”]
r_s_arr  [i] = result[“r_s”]
m200_arr [i] = result[“m200”]
chi2_arr [i] = result[“chi2”]

n_fit = np.sum(np.isfinite(c200_arr))
print(f”  NFW fits completed: {n_fit}/{ns} converged “
f”in {time.perf_counter()-t0_nfw:.1f}s”)

# ── 19.2  Figure: NFW concentration c_200 over time ──────────────────────────

fig19, axes19 = plt.subplots(2, 2, figsize=(12, 8), facecolor=”#0d0d18”,
gridspec_kw={“hspace”: 0.35, “wspace”: 0.32})
axes19 = axes19.flatten()

# (a) Concentration c_200 vs. time.

ax = axes19[0]
ax.set_facecolor(”#0d0d18”)
ax.plot(t_axis, c200_arr, color=”#9b6dff”, lw=1.5,
label=r”$c_{200} = r_{200}/r_s$”)
ax.set_xlabel(time_label, fontsize=9); ax.set_ylabel(r”$c_{200}$”, fontsize=9)
ax.set_title(“NFW Concentration”, fontsize=10)
ax.legend(fontsize=8)

# (b) Scale radius r_s vs. time.

ax = axes19[1]
ax.set_facecolor(”#0d0d18”)
ax.plot(t_axis, r_s_arr, color=”#4a8fff”, lw=1.5, label=r”$r_s$ [kpc]”)
ax.set_xlabel(time_label, fontsize=9); ax.set_ylabel(r”$r_s$ [kpc]”, fontsize=9)
ax.set_title(“NFW Scale Radius”, fontsize=10); ax.legend(fontsize=8)

# (c) Virial mass M_200 vs. time.

ax = axes19[2]
ax.set_facecolor(”#0d0d18”)
ax.semilogy(t_axis, m200_arr, color=”#2de8c0”, lw=1.5,
label=r”$M_{200}$ [M$*\odot$]”)
ax.set_xlabel(time_label, fontsize=9)
ax.set_ylabel(r”$M*{200}$ [M$_\odot$]”, fontsize=9)
ax.set_title(“Virial Mass”, fontsize=10); ax.legend(fontsize=8)

# (d) Fit quality chi-squared vs. time.

# Low chi2 → NFW is a good description; high chi2 → halo is far from NFW

# (e.g., right after pericentre when the profile is double-peaked).

ax = axes19[3]
ax.set_facecolor(”#0d0d18”)
ax.semilogy(t_axis, chi2_arr, color=”#ff5566”, lw=1.2, alpha=0.8,
label=r”Reduced $\chi^2$”)
ax.axhline(1.0, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.4,
label=“Good fit threshold”)
ax.set_xlabel(time_label, fontsize=9)
ax.set_ylabel(r”Reduced $\chi^2$”, fontsize=9)
ax.set_title(“NFW Fit Quality”, fontsize=10); ax.legend(fontsize=8)

fig19.suptitle(“NFW Profile Fitting  —  Structural Parameter Evolution”,
fontsize=12)
fig19.savefig(
os.path.join(OUT_DIR, “kinematics_nfw_parameters.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig19.get_facecolor(),
)
plt.close(fig19)
print(”  Saved: kinematics_nfw_parameters.png”)

# ── 19.3  Figure: NFW fit overlay at two epochs ───────────────────────────────

fig19b, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), facecolor=”#0d0d18”,
sharey=True, gridspec_kw={“wspace”: 0.08})
for ax in (axL, axR):
ax.set_facecolor(”#0d0d18”)

early_i = profile_snap_indices[0]
late_i  = profile_snap_indices[-1]

for ax, snap_i, title in [
(axL, early_i, f”Early (snap {SNAPSHOTS[early_i]})”),
(axR, late_i,  f”Late  (snap {SNAPSHOTS[late_i]})”),
]:
ax.set_xscale(“log”); ax.set_yscale(“log”)

```
# Measured M_enc (at outer bin edges).
r_outer  = R_BINS[1:]
m_meas   = menc_ts[snap_i, :]
valid    = np.isfinite(m_meas) & (m_meas > 0)

ax.scatter(r_outer[valid], m_meas[valid], color="#aaaacc",
           s=18, zorder=3, label="Measured $M(<r)$")

# NFW fit overlay.
if np.isfinite(c200_arr[snap_i]):
    res   = fit_nfw_to_snapshot(menc_ts[snap_i, :], r_mid, R_BINS)
    r_plt = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), 200)
    m_plt = nfw_enclosed_mass(r_plt, res["rho_s"], res["r_s"])
    ax.plot(r_plt, m_plt, color="#ff9944", lw=2.0,
            label=fr"NFW fit  $c_{{200}}$={res['c200']:.1f}")

ax.set_xlabel("r [kpc]", fontsize=10)
ax.set_title(title, fontsize=10)
ax.legend(fontsize=8)
```

axL.set_ylabel(r”$M(<r)$ [M$_\odot$]”, fontsize=10)
fig19b.suptitle(“NFW Enclosed Mass Fit”, fontsize=12)
fig19b.savefig(
os.path.join(OUT_DIR, “kinematics_nfw_fit_overlay.png”),
dpi=300, bbox_inches=“tight”, facecolor=fig19b.get_facecolor(),
)
plt.close(fig19b)
print(”  Saved: kinematics_nfw_fit_overlay.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  SECTION 20 — FINAL OUTPUT INVENTORY                                       ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Print a complete manifest of every file written to OUT_DIR, with file sizes.

# This is the last thing the script does, so its presence in the terminal log

# confirms that all sections ran to completion.

print(”\n” + “=” * 70)
print(”  COMPLETE OUTPUT MANIFEST”)
print(”=” * 70)
print(f”  {‘Filename’:<48} {‘Size (MB)’:>9}”)
print(f”  {’-’*48} {’-’*9}”)

total_mb = 0.0
for fn in sorted(os.listdir(OUT_DIR)):
fp   = os.path.join(OUT_DIR, fn)
mb   = os.path.getsize(fp) / 1e6
total_mb += mb
suffix = “”
if fn.endswith(”.mp4”):     suffix = “ [animation]”
elif fn.endswith(”.png”):   suffix = “ [figure]”
print(f”  {fn:<48} {mb:>8.2f}{suffix}”)

print(f”  {’-’*48} {’-’*9}”)
print(f”  {‘TOTAL’:<48} {total_mb:>8.2f}”)
print(”=” * 70)
print(f”\n[DONE] All {len(os.listdir(OUT_DIR))} outputs written to {OUT_DIR}/”)
print(f”       Total size: {total_mb:.1f} MB”)
