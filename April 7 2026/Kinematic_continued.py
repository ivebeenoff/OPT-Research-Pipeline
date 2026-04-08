"""
===============================================================================
MW–M31 MERGER KINEMATIC PROFILES PIPELINE
===============================================================================
Author  : Abhinav Vatsa

Overview
--------
This script processes N-body simulation snapshots of the Milky Way (MW) and
Andromeda (M31) merger to compute a suite of kinematic profiles as a function
of both radius and simulation time. It then produces publication-ready figures
tracing how the merger reshapes the joint halo's internal kinematics.

The computed profiles per snapshot are:
  • σ_r(r)    — radial velocity dispersion (mass-weighted)
  • σ_t(r)    — tangential velocity dispersion (mass-weighted)
  • v_rot(r)  — mean azimuthal (rotation) velocity about the z-axis
  • j(r)      — mean specific angular momentum magnitude (mass-weighted)
  • M_enc(r)  — enclosed mass (for circular/escape velocity)
  • v_esc(r)  — local escape speed from enclosed mass
  • β(r)      — Binney velocity anisotropy parameter

Global (volume-averaged) scalars are also tracked over time for quick
diagnostic plots of the merger's kinematic history.

Data model
----------
Snapshots are stored in tar archives in the working directory.  Each archive
contains plain-text files named MW_NNN.txt and M31_NNN.txt.  The files are
read via the project-local CenterOfMass2.CenterOfMass class, which internally
uses ReadFile.Read and exposes per-particle arrays (x, y, z, vx, vy, vz, m)
together with a simulation time stamp.

Units
-----
All positions are in kpc, velocities in km/s, and masses in 10^10 M_sun as
stored in the snapshot files.  Masses are converted to M_sun immediately after
loading.  The gravitational constant G is expressed in units that are
consistent with these:
    G = 4.30091 × 10^{-6}  kpc (km/s)^2 M_sun^{-1}

Output files
------------
  kinematics_inner_evolution.png  — Time-series of inner-halo kinematic scalars
  kinematics_heatmaps.png         — (r, t) heatmaps of log σ_r and β
  kinematics_profiles_grid.png    — Radial profiles at selected snapshots
  kinematics_angular_momentum.png — Specific angular momentum time evolution
  kinematics_escape_velocity.png  — Escape speed profiles at key epochs
  kinematics_beta_selected.png    — β(r) profiles at key merger stages

Dependencies
------------
  numpy, matplotlib
  ReadFile          (project-local)
  CenterOfMass2     (project-local)
===============================================================================
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
matplotlib.use("Agg")                    # non-interactive backend for HPC/batch
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
#   1 = dark matter halo  (most particles, good for large-scale COM)
#   2 = disk              (traces the baryonic centre more closely)
#   3 = bulge
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

# Newton's constant in simulation-friendly units:
#   [kpc · (km/s)^2 · M_sun^{-1}]
# Derived from SI: G = 6.674e-11 m^3 kg^{-1} s^{-2}
#   × (1 kpc / 3.0857e19 m) × (1 M_sun / 1.989e30 kg) × (1 km/s / 1000 m/s)^{-2}
G_KPC_KMS2_MSUN = 4.30091e-6

# Snapshot mass files store particle masses in units of 10^10 M_sun.
# Multiply by this factor to convert to M_sun.
MASS_UNIT_MSUN = 1.0e10

# ── Output directory ──────────────────────────────────────────────────────────
# All figures and intermediate products go here.
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Matplotlib global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0d18",
    "axes.facecolor":    "#0d0d18",
    "axes.edgecolor":    "#2a2a4a",
    "axes.labelcolor":   "#c8c8e8",
    "axes.grid":         True,
    "grid.color":        "#1e1e36",
    "grid.linewidth":    0.6,
    "xtick.color":       "#9090b0",
    "ytick.color":       "#9090b0",
    "text.color":        "#c8c8e8",
    "legend.facecolor":  "#0d0d18",
    "legend.edgecolor":  "#2a2a4a",
    "legend.fontsize":   8,
    "font.family":       "monospace",
})

# Pre-build the snapshot array once; reused everywhere.
SNAPSHOTS = np.arange(START_SNAP, END_SNAP + 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — DATA EXTRACTION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_snapshots_from_tarballs(work_dir: str) -> str:
    """
    Scan the current working directory for any ``*.tar`` archives and extract
    all MW_NNN.txt and M31_NNN.txt members into a fresh temporary directory.

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — CENTRE-OF-MASS UTILITIES                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_com_position(mw: CenterOfMass,
                     x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     m_raw: np.ndarray) -> np.ndarray:
    """
    Return the 3D centre-of-mass position [kpc] of the combined MW+M31 system.

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


def get_com_velocity(mw: CenterOfMass,
                     pos_com: np.ndarray,
                     pos_all: np.ndarray,
                     vel_all: np.ndarray,
                     mass_all: np.ndarray) -> np.ndarray:
    """
    Return the 3D centre-of-mass velocity [km/s] of the combined system.

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — KINEMATIC PROFILE ENGINE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_profiles_for_snapshot(mw_path: str, m31_path: str) -> dict:
    """
    Core computation: load one snapshot, centre the system, and compute all
    kinematic profiles in radial bins.

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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — SNAPSHOT LOOP                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── 5.1  Extract snapshot files from tar archives ─────────────────────────────
tmpdir = extract_snapshots_from_tarballs(".")

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
print("\n" + "=" * 70)
print("  PROCESSING SNAPSHOTS")
print("=" * 70)

t_loop_start = time.perf_counter()

for i, snap_num in enumerate(SNAPSHOTS):

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

t_total = time.perf_counter() - t_loop_start
print(f"\n[DONE] Processed in {t_total/60:.1f} min")


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
time_label    = "Time [Gyr]" if time_is_gyr else "Snapshot index"

# ── 6.4  Select representative snapshots for profile plots ────────────────────
# We show the joint radial profiles at five epochs spread across the simulation:
# initial conditions, early interaction, first passage, post-merger, final state.
profile_snap_fractions = [0.0, 0.2, 0.4, 0.65, 1.0]
n_snaps                = len(SNAPSHOTS)
profile_snap_indices   = [
    int(f * (n_snaps - 1)) for f in profile_snap_fractions
]
profile_labels = [f"Snap {SNAPSHOTS[k]}" for k in profile_snap_indices]
profile_colors = ["#00d4aa", "#7b9fff", "#ffaa44", "#ff6b9a", "#aa88ff"]
