"""


===============================================================================
SECTION 36 — MULTIPOLE EXPANSION & MINKOWSKI FUNCTIONAL MORPHOLOGY
===============================================================================
Author  : Abhinav Vatsa  [SCAFFOLD — fill in implementation]

This section is a guided scaffold.  Every subsection contains:
  • Physical motivation and context
  • The exact quantity to compute and its formula
  • A suggested implementation strategy
  • Time and space complexity analysis
  • Hints on numerical pitfalls to avoid
  • Expected output description

Context and motivation
──────────────────────
Section 35 characterised halo shape using two geometric descriptors:
  • Inertia tensor  → bulk second-moment shape (q, s, T)
  • Convex hull     → extreme-particle geometric extent (Ψ, V_excess)

Both of those methods share a common limitation: they are GLOBAL in angular
structure.  The inertia tensor produces a single ellipsoid that best fits ALL
particles in a shell.  The convex hull produces a single polyhedron.  Neither
method can answer questions like:

    "Is the halo lopsided?  Does it have a one-armed tidal plume?
     Is there a bar-like feature embedded in an otherwise round halo?
     Are there higher-order angular structures — quadrupole, hexadecapole —
     above the leading ellipsoidal distortion?"

This section deploys two methods that are fundamentally ANGULAR and
MULTI-SCALE, resolving shape at every order simultaneously:

METHOD A — MULTIPOLE EXPANSION of the density field:
  Expand the 3D particle density field ρ(r, θ, φ) in spherical harmonics Y_lm:

    ρ(r, θ, φ) = Σ_{l=0}^{L_MAX} Σ_{m=-l}^{l} a_lm(r) Y_lm(θ, φ)

  Each coefficient a_lm(r) is a RADIAL PROFILE of the angular mode (l, m).
  The POWER in multipole l is:
    P_l(r) = Σ_{m=-l}^{l} |a_lm(r)|²
  Normalised by the monopole P_0(r) = |a_00(r)|²:
    Ẽ_l(r) = P_l(r) / P_0(r)   — relative multipole power

  Physical interpretation:
    l = 0  : monopole (total mass — always 1 after normalisation)
    l = 1  : dipole  (COM offset or one-armed lopsidedness)
    l = 2  : quadrupole (ellipsoidal elongation, bar, tidal distortion)
    l = 3  : octopole (triangular/boxy distortions)
    l = 4  : hexadecapole (diskiness, rectangular halos)
    l ≥ 5  : higher harmonics (substructure, stream crossings)

  The quadrupole (l=2) is directly related to the inertia tensor eigenvalues
  from §35, providing a cross-check:
    Ẽ_2 ∝ (a² − c²) / (a² + b² + c²)
  Agreement between the multipole quadrupole and the tensor ellipticity
  validates both methods simultaneously.

METHOD B — MINKOWSKI FUNCTIONALS of excursion sets:
  The Minkowski functionals (MFs) are a complete set of four morphological
  measures for a convex body in 3D, based on integral geometry.
  For the EXCURSION SET Ω_ρ = { x : ρ(x) > ρ_threshold }, the four MFs are:

    W_0 = V          — volume of the excursion set [kpc³]
    W_1 = A/6        — surface area / 6  [kpc²]
    W_2 = (1/6π) ∫ (κ_1 + κ_2) dA  — integrated mean curvature [kpc]
    W_3 = (1/4π) ∫ κ_1 κ_2 dA      — Euler characteristic / 4π (topological)

  where κ_1, κ_2 are the principal curvatures of the surface.

  Hadwiger's theorem guarantees that ANY morphological quantity that is
  additive, motion-invariant, and continuous can be written as a linear
  combination of these four functionals.  They are COMPLETE.

  Shapefinders derived from the MFs (Sahni+ 1998, Sheth+ 2003):
    Thickness : T_MF = 3 W_0 / W_1        [kpc]
    Width     : W_MF = W_1 / (2 W_2)      [kpc]
    Length    : L_MF = W_2 / (3 W_3)      [kpc]

  Planarity  : P_MF = (W_MF − T_MF) / (W_MF + T_MF)  ∈ [0, 1]
                      P = 0: sphere/filament; P = 1: sheet
  Filamentarity: F_MF = (L_MF − W_MF) / (L_MF + W_MF)  ∈ [0, 1]
                      F = 0: sphere/sheet; F = 1: filament

  In the (P, F) plane:
    (0, 0) → sphere
    (1, 0) → pancake (oblate)
    (0, 1) → filament (prolate)
    (1, 1) → ribbon (impossible for a convex body)

  The Minkowski functionals thus give a TOPOLOGICAL + GEOMETRIC description
  that is INDEPENDENT of coordinate choice and does not assume ellipsoidal
  symmetry.  They detect sheet-like structures (tidal caustics), filaments
  (tidal tails), and topology changes (holes in the halo) that all other
  methods miss.

Complementarity with §35
──────────────────────────
  Inertia tensor (§35)  → best-fit ellipsoid shape (q, s, T)
  Convex hull (§35)     → extreme geometric extent (Ψ, V_excess)
  Multipole expansion   → ANGULAR STRUCTURE at every order (which harmonics)
  Minkowski functionals → TOPOLOGICAL + GEOMETRIC morphology (P, F, W_3)

  Together, these four methods form a complete morphological characterisation.
  The MF Euler characteristic W_3 is TOPOLOGICALLY INVARIANT — no deformation
  that preserves topology can change it.  It detects mergers as TOPOLOGY
  CHANGES (a second density peak appearing, creating a genus change).

Key quantities computed in this section
────────────────────────────────────────
  MULTIPOLE:
    a_lm(r, t)       — complex spherical harmonic coefficients per shell
    P_l(r, t)        — multipole power per degree l per shell
    Ẽ_l(r, t)        — normalised multipole power P_l / P_0
    Q_bar(r, t)      — quadrupole strength Ẽ_2  (dominant shape mode)
    D_bar(r, t)      — dipole strength Ẽ_1  (lopsidedness / COM drift)
    φ_lm(r, t)       — phase angle of a_lm  (orientation of each mode)
    l_peak(r, t)     — dominant multipole degree (which order dominates)
    multi_spectrum(t)— shell-averaged multipole spectrum at each epoch

  MINKOWSKI:
    W0(ρ_th, t)      — volume of excursion set [kpc³]
    W1(ρ_th, t)      — surface / 6  [kpc²]
    W2(ρ_th, t)      — integrated mean curvature [kpc]
    W3(ρ_th, t)      — Euler characteristic / 4π  (dimensionless, integer/4π)
    T_MF(ρ_th, t)    — thickness shapefinder [kpc]
    W_MF(ρ_th, t)    — width shapefinder [kpc]
    L_MF(ρ_th, t)    — length shapefinder [kpc]
    P_MF(ρ_th, t)    — planarity ∈ [0, 1]
    F_MF(ρ_th, t)    — filamentarity ∈ [0, 1]
    χ_euler(ρ_th, t) — Euler characteristic (integer; 1 = one blob, 0 = ring)

Dependencies
────────────
  scipy.special     — sph_harm (spherical harmonics)
  scipy.ndimage     — gaussian_filter, label (for excursion set extraction)
  skimage           — marching_cubes, mesh_surface_area (for MF computation)
                      (scikit-image; pip install scikit-image)
  numpy.fft         — for grid density estimation
  Section 26        — traj_pos, traj_r (Lagrangian trajectories)
  Section 35        — q_shell_ts, T_shell_ts, r_edges (for cross-comparison)

All globals from the parent pipeline are inherited.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter, label as ndimage_label
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# MULTIPOLE PARAMETERS
# ─────────────────────
# L_MAX: maximum multipole degree to compute.
#   Physical content saturates around l = 8–10 for a smooth halo.
#   Higher l captures particle shot noise rather than real structure.
#   HINT: always check that Ẽ_l decreases monotonically with l at early
#   times (virialised halo).  If it rises at large l, you are fitting noise.
#   Recommended: L_MAX = 8 for a standard DM halo analysis.
#
# N_MULTIPOLE_SHELLS: number of radial shells for the multipole profile.
#   Use finer binning than §35 since multipole computation is cheap.
#   Recommend: 30–50 shells.
#
# N_MULTIPOLE_SNAPS: number of snapshot epochs.
#   The multipole computation is O(N × L_MAX²) per snapshot — fast.

L_MAX              = 8      # maximum spherical harmonic degree
N_MULTIPOLE_SHELLS = 40     # radial shells for multipole profile
N_MULTIPOLE_SNAPS  = 60     # snapshot epochs for multipole analysis
MIN_MULTIPOLE_PART = 20     # minimum particles per shell for valid a_lm

# MINKOWSKI FUNCTIONAL PARAMETERS
# ─────────────────────────────────
# The MFs are computed on a 3D DENSITY GRID, not directly from particles.
# The grid must be fine enough to resolve the relevant structures.
#
# GRID_RES: number of voxels per side of the cubic density grid.
#   The physical resolution is: Δx = 2 R_OUTER / GRID_RES [kpc/voxel].
#   For R_OUTER = 120 kpc and GRID_RES = 128: Δx = 1.875 kpc/voxel.
#   For GRID_RES = 64: Δx = 3.75 kpc — enough to resolve the inner halo.
#   HINT: the MF computation scales as O(GRID_RES³), so doubling the
#   resolution costs 8× more time and memory.  GRID_RES = 64 is a good
#   default; use 128 only if you need to resolve structures below 4 kpc.
#
# SMOOTH_SIGMA: Gaussian smoothing applied to the density grid before
#   computing the excursion set.  In VOXEL units.
#   Without smoothing, shot noise from sparse particles produces
#   fragmented excursion sets with Euler characteristic >> 1.
#   SMOOTH_SIGMA ≈ 2–4 voxels gives a well-connected excursion set.
#   HINT: the physical smoothing length is SMOOTH_SIGMA × Δx [kpc].
#   Make sure this is much smaller than the structure you want to detect.
#
# N_THRESHOLDS: number of density thresholds for the MF curves.
#   Each threshold gives a point in the (P, F) morphology plane.
#   Scanning thresholds from low (includes whole halo) to high (core only)
#   traces a MORPHOLOGY CURVE as a function of overdensity level.
#   N_THRESHOLDS = 30 gives a smooth curve with good coverage.
#
# N_MF_SNAPS: number of snapshot epochs for MF computation.
#   The MF computation is O(GRID_RES³) per threshold per snapshot — slower.
#   Use a subset of snapshots for the full MF analysis.

GRID_RES           = 64     # voxels per side of the density grid
SMOOTH_SIGMA       = 2.5    # Gaussian smoothing in voxel units
N_THRESHOLDS       = 30     # density thresholds for MF curves
N_MF_SNAPS         = 25     # snapshot epochs for Minkowski analysis
MIN_VOXELS_SURFACE = 10     # minimum surface voxels for valid MF estimate

# Density threshold range: from 5th to 95th percentile of the grid density.
# This ensures the low-density outskirts and the dense core are both sampled.
# Set dynamically in §36.3 from the actual density field.
THRESH_PERCENTILE_LO  = 5   # percentile for lowest threshold
THRESH_PERCENTILE_HI  = 95  # percentile for highest threshold

# Animation.
ANIM_FPS_36        = 18
ANIM_DPI_36        = 100
ANIM_BITRATE_36    = 1600

print("\n" + "="*80)
print("  SECTION 36 · Multipole Expansion & Minkowski Functional Morphology")
print("="*80)
print(f"  Multipole L_MAX     : {L_MAX}")
print(f"  Multipole shells    : {N_MULTIPOLE_SHELLS}")
print(f"  Multipole snaps     : {N_MULTIPOLE_SNAPS}")
print(f"  Grid resolution     : {GRID_RES}³")
print(f"  Smoothing sigma     : {SMOOTH_SIGMA} voxels")
print(f"  MF thresholds       : {N_THRESHOLDS}")
print(f"  MF snaps            : {N_MF_SNAPS}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.1 — LOAD POSITIONS AND DEFINE COORDINATE GRIDS                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: Reuse _traj_pos and _traj_r from Section 26.  Copy the standard
# try/except fallback block.
#
# TWO COORDINATE SYSTEMS are needed in this section:
#
# A) SPHERICAL COORDINATES for the multipole expansion:
#      r(t, i)   = |pos(t, i)|
#      θ(t, i)   = arccos(z / r)    — colatitude ∈ [0, π]
#      φ(t, i)   = arctan2(y, x)    — azimuth ∈ [−π, π]
#    scipy.special.sph_harm(m, l, φ, θ) uses the physics convention:
#      first argument = φ (azimuth), second = θ (colatitude).
#    CRITICAL: scipy uses sph_harm(m, l, phi, theta) with phi FIRST.
#    Many derivations use the opposite convention.  Double-check.
#
# B) CARTESIAN GRID for the Minkowski functionals:
#      The density field ρ(x, y, z) is estimated on a regular cubic grid
#      using a Cloud-In-Cell (CIC) mass assignment scheme.
#      Grid edges: x, y, z ∈ [−R_OUTER, +R_OUTER]
#      Grid spacing: Δx = 2 R_OUTER / GRID_RES
#
# COM subtraction:
#   ALWAYS subtract the instantaneous COM before computing either set
#   of coordinates.  Use the full particle population for the COM.
#
# HINT: precompute the spherical coordinates at ALL snapshot epochs at load
# time — this avoids recomputing arccos and arctan2 inside the inner loop.
# Memory cost: O(N_MULTIPOLE_SNAPS × N × 2) ≈ 60 × 2000 × 2 × 8 bytes = 2 MB.

# TODO: implement trajectory loading / inheritance
_traj_pos  = None   # (ns, N, 3)  [kpc] — replace with inherited
_traj_r    = None   # (ns, N)     [kpc]
_r0        = None   # (N,)  initial radii
_group     = None   # (N,)  0=inner,1=mid,2=outer,3=M31
_N         = 0

# TODO: subsample snapshots and precompute spherical coordinates
# snap_indices_mult = np.linspace(0, ns-1, N_MULTIPOLE_SNAPS, dtype=int)
# snap_indices_mf   = np.linspace(0, ns-1, N_MF_SNAPS, dtype=int)
#
# theta_arr = np.full((N_MULTIPOLE_SNAPS, _N), np.nan)   # colatitude
# phi_arr   = np.full((N_MULTIPOLE_SNAPS, _N), np.nan)   # azimuth
#
# for k, s in enumerate(snap_indices_mult):
#     com      = np.mean(_traj_pos[s], axis=0)
#     pos_c    = _traj_pos[s] - com
#     r_c      = np.linalg.norm(pos_c, axis=1)
#     r_safe   = np.maximum(r_c, 1e-10)
#     theta_arr[k] = np.arccos(np.clip(pos_c[:, 2] / r_safe, -1.0, 1.0))
#     phi_arr[k]   = np.arctan2(pos_c[:, 1], pos_c[:, 0])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.2 — SPHERICAL HARMONIC COEFFICIENT ESTIMATOR                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The spherical harmonic expansion of a surface density Σ(θ, φ) on the unit
# sphere is:
#
#   Σ(θ, φ) = Σ_{l=0}^{∞} Σ_{m=-l}^{l}  a_lm  Y_lm(θ, φ)
#
# where the COEFFICIENTS are:
#
#   a_lm = ∫ Σ(θ, φ)  Y_lm*(θ, φ)  sin θ dθ dφ
#
# For a discrete particle distribution in a radial shell, Σ(θ, φ) is a
# sum of Dirac deltas on the unit sphere.  The coefficient estimator becomes:
#
#   a_lm(r_shell)  =  (1/N_shell) Σ_{k ∈ shell}  Y_lm*(θ_k, φ_k)
#
# where * denotes complex conjugate, and N_shell is the number of particles
# in the shell (normalisation makes it a per-particle estimate).
#
# The POWER in multipole degree l is:
#
#   P_l  =  Σ_{m=-l}^{l} |a_lm|²
#
# For an isotropic (spherical) distribution, P_l = 0 for all l ≥ 1.
# For a perfectly oblate distribution, P_2 dominates (m=0 mode).
# For a prolate bar rotated in the x-y plane, the m=±2 modes dominate.
#
# The NORMALISED power (relative to monopole) is:
#
#   Ẽ_l  =  P_l / a_00²    (a_00 = 1/√(4π) for normalised density)
#
# HINT: scipy.special.sph_harm(m, l, phi, theta) returns COMPLEX values.
# Note the convention: scipy's (phi, theta) = (azimuth, colatitude).
# For real-valued representations: combine Y_lm and Y_l{-m} using:
#   Re(Y_lm) is even in m; Im(Y_lm) is odd in m.
# But for power spectrum purposes, just use |a_lm|² and sum over m.
#
# HINT: for m < 0, use the relation Y_l{-m} = (−1)^m conj(Y_lm).
# scipy.special.sph_harm handles negative m correctly.
#
# HINT: the l=1 dipole can be non-zero even after COM subtraction if the
# particle SUBSET in the shell has a displaced COM relative to the full
# population.  This is a real physical signal (lopsided shell), not an
# artefact, so do NOT further subtract the shell COM.
#
# TIME COMPLEXITY:  O(N_shell × L_MAX²)  per shell per snapshot
#                 = O(100 × 64)  per call ≈ 6400 operations — trivial
# SPACE COMPLEXITY: O(L_MAX²)  — one complex number per (l, m) pair

def compute_alm(theta_shell, phi_shell, l_max=L_MAX):
    """
    Compute the spherical harmonic coefficients a_lm for a set of
    particle directions (theta, phi) on the unit sphere.

    Parameters
    ----------
    theta_shell : (N,)  — colatitude of each particle [rad] ∈ [0, π]
    phi_shell   : (N,)  — azimuth of each particle [rad] ∈ [−π, π]
    l_max       : int   — maximum multipole degree

    Returns
    -------
    alm : dict  — {(l, m): complex a_lm}  for l ∈ [0, l_max], m ∈ [−l, l]
    P_l : (l_max+1,)  — real power per degree: Σ_m |a_lm|²
    Etilde_l : (l_max+1,) — normalised power P_l / P_0

    HINT: precompute sph_harm for all (l, m) in one loop, storing in a dict.
          The inner loop over m at fixed l is the hot path; vectorise over
          particles first: Y_lm_vec = sph_harm(m, l, phi_shell, theta_shell)
          gives shape (N,); then a_lm = np.mean(np.conj(Y_lm_vec)).

    HINT: sph_harm raises ValueError for |m| > l — guard with abs(m) <= l.

    HINT: for power purposes, only m ≥ 0 is needed because |a_l{-m}|² = |a_lm|²
          (by complex conjugate symmetry for real density fields).
          Sum: P_l = |a_l0|² + 2 Σ_{m=1}^{l} |a_lm|²
          This halves the computation.
    """
    N = len(theta_shell)
    if N < MIN_MULTIPOLE_PART:
        nan_pl = np.full(l_max + 1, np.nan)
        return {}, nan_pl, nan_pl

    alm    = {}
    P_l    = np.zeros(l_max + 1)

    for l in range(l_max + 1):
        # m = 0
        Y_l0        = sph_harm(0, l, phi_shell, theta_shell)   # (N,)
        a_l0        = np.mean(np.conj(Y_l0))
        alm[(l, 0)] = a_l0
        P_l[l]     += float(np.abs(a_l0)**2)

        # m > 0: use symmetry |a_l{-m}|² = |a_lm|²
        for m in range(1, l + 1):
            Y_lm        = sph_harm(m, l, phi_shell, theta_shell)
            a_lm        = np.mean(np.conj(Y_lm))
            alm[(l,  m)] = a_lm
            alm[(l, -m)] = (-1)**m * np.conj(a_lm)
            P_l[l]      += 2.0 * float(np.abs(a_lm)**2)

    # Normalised power: Ẽ_l = P_l / P_0
    P0         = P_l[0] if P_l[0] > 1e-30 else 1.0
    Etilde_l   = P_l / P0

    return alm, P_l, Etilde_l


def multipole_phase(alm, l, m):
    """
    Extract the phase angle φ_lm = arg(a_lm) of a specific mode.

    Parameters
    ----------
    alm : dict  — output of compute_alm
    l   : int   — multipole degree
    m   : int   — multipole order

    Returns
    -------
    phase : float  — phase angle in degrees ∈ [−180, 180]

    Physical interpretation:
      For l=2, m=2: the phase gives the orientation of the quadrupole
      in the x-y plane.  Tracking this phase over time reveals whether
      the halo is rotating (linearly increasing phase) or oscillating.
      Sudden phase jumps signal major structural reorganisation.
    """
    if (l, m) not in alm:
        return np.nan
    return float(np.degrees(np.angle(alm[(l, m)])))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.3 — MULTIPOLE RADIAL PROFILES                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The radial profile of multipole power Ẽ_l(r) reveals WHERE in the halo
# each angular mode is dominant.
#
# For a merger remnant, expected radial structure:
#   Inner (r < 5 kpc):
#     Ẽ_2 ≈ 0  — the core is nearly spherical; the disc has been disrupted
#     Ẽ_4 may peak here if a bar-like remnant persists
#
#   Mid halo (5–30 kpc):
#     Ẽ_2 rises — the merger-induced prolate distortion peaks here
#     Ẽ_1 may peak during the first passage (lopsided shell)
#     All other Ẽ_l ≈ 0 if no streams are present
#
#   Outer halo (r > 30 kpc):
#     Ẽ_2 large — tidal distortion is strongest in the outer halo
#     Ẽ_1 may dominate if M31 debris is off-centre
#     All l modes excited during active tidal stripping
#
# Implementation
# ──────────────
# For each snapshot k and each radial shell b:
#   1. Select particles in the shell.
#   2. Extract their (θ, φ) angles.
#   3. Call compute_alm → get Ẽ_l(r_b, t_k) for all l.
#   4. Also store the phase of the dominant mode.
#
# Shell edges: use the same r_edges as §35 for cross-comparison.
# But use N_MULTIPOLE_SHELLS = 40 shells for finer radial resolution.
#
# The DOMINANT MULTIPOLE l_peak(r, t) is the l that maximises Ẽ_l
# for l ≥ 1 (excluding the monopole):
#   l_peak(r, t) = argmax_{l ≥ 1} Ẽ_l(r, t)
#
# TIME COMPLEXITY:  O(N_MULTIPOLE_SNAPS × N_MULTIPOLE_SHELLS × N_shell × L_MAX²)
#                 ≈ O(60 × 40 × 100 × 64) ≈ 1.5 × 10^7  — minutes
# SPACE COMPLEXITY: O(N_MULTIPOLE_SNAPS × N_MULTIPOLE_SHELLS × L_MAX)
#                 ≈ 60 × 40 × 9 × 8 bytes ≈ 170 KB

# TODO: allocate multipole arrays
# Etilde_arr  = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS, L_MAX+1), np.nan)
# P_l_arr     = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS, L_MAX+1), np.nan)
# phase22_arr = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # phase of a_22
# phase21_arr = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # phase of a_21
# l_peak_arr  = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan, dtype=float)
# Q_bar_arr   = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # Ẽ_2
# D_bar_arr   = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # Ẽ_1
# H_bar_arr   = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # Ẽ_4

# TODO: compute multipole radial profiles
# r_edges_mult = np.logspace(np.log10(R_INNER), np.log10(R_OUTER), N_MULTIPOLE_SHELLS+1)
# r_mid_mult   = np.sqrt(r_edges_mult[:-1] * r_edges_mult[1:])
#
# for k, s in enumerate(snap_indices_mult):
#     com   = np.mean(_traj_pos[s], axis=0)
#     r_now = _traj_r[s]    # pre-computed radii from Section 26
#     for b in range(N_MULTIPOLE_SHELLS):
#         mask = (r_now >= r_edges_mult[b]) & (r_now < r_edges_mult[b+1])
#         if mask.sum() < MIN_MULTIPOLE_PART:
#             continue
#         alm_dict, P_l, Etilde = compute_alm(theta_arr[k][mask], phi_arr[k][mask])
#         Etilde_arr[k, b]  = Etilde
#         P_l_arr[k, b]     = P_l
#         Q_bar_arr[k, b]   = Etilde[2]   # l=2
#         D_bar_arr[k, b]   = Etilde[1]   # l=1
#         H_bar_arr[k, b]   = Etilde[4]   # l=4
#         phase22_arr[k, b] = multipole_phase(alm_dict, 2, 2)
#         if not np.all(np.isnan(Etilde[1:])):
#             l_peak_arr[k, b] = float(np.nanargmax(Etilde[1:]) + 1)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.4 — GLOBAL MULTIPOLE SPECTRUM                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# In addition to the radial profile, compute the GLOBAL multipole spectrum —
# the a_lm coefficients for ALL particles simultaneously, not shell by shell.
# This gives the total angular power in each mode l integrated over all radii.
#
# The global spectrum is directly comparable to COSMOLOGICAL SIMULATIONS where
# halo shapes are often reported as single values (not radial profiles).
# It is also what weak gravitational lensing observationally constrains:
# the projected mass ellipticity is the l=2 moment of the 2D projected density.
#
# GLOBAL SPECTRUM EVOLUTION:
#   P_l_global(t) = (1/N) Σ_k |Y_lm(θ_k, φ_k)|²  summed over all particles
#
# CROSS-POWER between shells:
#   C_l(r_1, r_2) = Re[ a_lm*(r_1) × a_lm(r_2) ]  summed over m
#   This measures how well the angular structure at radius r_1 is correlated
#   with the structure at r_2.  High C_l means the shape is COHERENT across
#   radii — characteristic of a single ellipsoid.  Low C_l means the inner
#   and outer halo have different orientations (misalignment / twisting halo).
#
# QUADRUPOLE MISALIGNMENT:
#   The orientation of the l=2 mode can differ between shells.
#   Define the quadrupole misalignment angle between shell b1 and b2:
#     ΔPA_22(b1, b2) = |phase22(b1) − phase22(b2)| / 2  [deg]
#   (the /2 accounts for the 180° degeneracy of l=2 modes)
#   Large ΔPA_22 → the inner and outer halo are TWISTED (isophote twist).
#   This is a strong signature of triaxiality (Franx+ 1991).
#
# TIME COMPLEXITY:  O(N_MULTIPOLE_SNAPS × N × L_MAX²)
#                 ≈ O(60 × 2000 × 64)  ≈ 7.7 × 10^6  — seconds
# SPACE COMPLEXITY: O(N_MULTIPOLE_SNAPS × (L_MAX+1)²)  — cross-power matrix

# TODO: compute global multipole spectrum per snapshot
# Etilde_global  = np.full((N_MULTIPOLE_SNAPS, L_MAX+1), np.nan)
# phase22_global = np.full(N_MULTIPOLE_SNAPS, np.nan)
# C_l_cross      = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS,
#                            N_MULTIPOLE_SHELLS), np.nan)   # optional, expensive

# TODO: compute quadrupole misalignment between inner and outer shells
# dPA_22_io      = np.full(N_MULTIPOLE_SNAPS, np.nan)   # inner-outer misalignment
# for k in range(N_MULTIPOLE_SNAPS):
#     phi_inner = phase22_arr[k, 0]   # first valid shell
#     phi_outer = phase22_arr[k, -1]  # last valid shell
#     if np.isfinite(phi_inner) and np.isfinite(phi_outer):
#         raw_diff = abs(phi_inner - phi_outer)
#         dPA_22_io[k] = min(raw_diff, 180.0 - raw_diff) / 2.0

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.5 — 3D DENSITY GRID CONSTRUCTION (FOR MINKOWSKI FUNCTIONALS)        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Minkowski functionals require a SMOOTH, CONTINUOUS density field rather than
# discrete particles.  We build a 3D density grid using the Cloud-In-Cell (CIC)
# mass assignment scheme, which is second-order accurate (vs. nearest-grid-point
# which is first-order).
#
# CLOUD-IN-CELL (CIC) SCHEME:
#   Each particle at position x assigns mass to the 8 surrounding grid cells,
#   weighted by the trilinear (volume) overlap between the particle's unit
#   cube and each cell:
#
#     For particle at (x, y, z):
#       ix = floor((x − x_min) / Δx)
#       dx = (x − x_min) / Δx − ix       — fractional offset ∈ [0, 1)
#       Similarly for iy, dy, iz, dz.
#
#     Weight for the 8 surrounding cells:
#       w[0, 0, 0] = (1−dx)(1−dy)(1−dz)
#       w[1, 0, 0] = dx × (1−dy) × (1−dz)
#       ...  (all 8 combinations)
#
#     Add w to rho[ix + di, iy + dj, iz + dk] for (di, dj, dk) ∈ {0,1}³.
#
# HINT: numpy vectorised CIC is O(N) but requires careful boundary handling.
#   Use np.clip(ix, 0, GRID_RES-2) to prevent out-of-bounds for particles
#   exactly at the grid boundary.
#
# HINT: after CIC assignment, normalise by the MEAN density so the grid
# is in units of OVERDENSITY:  δ(x) = ρ(x)/ρ̄ − 1.
# The excursion set Ω_δ = {x : δ(x) > δ_threshold} is then independent
# of the total particle count.
#
# GAUSSIAN SMOOTHING:
#   Apply scipy.ndimage.gaussian_filter(rho_grid, sigma=SMOOTH_SIGMA)
#   BEFORE computing excursion sets.  This suppresses Poisson shot noise
#   that would otherwise fragment the excursion set into thousands of
#   disconnected blobs with random Euler characteristic.
#   The smoothed field is ρ̃(x) = G_σ * ρ(x) where G_σ is a Gaussian kernel.
#   SMOOTH_SIGMA in voxel units controls the smoothing scale.
#
# TIME COMPLEXITY:  O(N + GRID_RES³)  per snapshot (CIC + smoothing)
#                 = O(2000 + 262144)  ≈ fast, ~0.1 seconds per snapshot
# SPACE COMPLEXITY: O(GRID_RES³) = 64³ × 8 bytes = 2 MB per snapshot

def build_density_grid(pos_com, grid_res=GRID_RES, r_outer=None,
                        smooth_sigma=SMOOTH_SIGMA):
    """
    Construct a smoothed 3D density grid using the Cloud-In-Cell scheme.

    Parameters
    ----------
    pos_com    : (N, 3)  — particle positions, COM-subtracted [kpc]
    grid_res   : int     — number of voxels per side (cubic grid)
    r_outer    : float   — half-size of the grid [kpc]; default = max(|pos|)*1.05
    smooth_sigma: float  — Gaussian smoothing in voxel units

    Returns
    -------
    rho_grid   : (grid_res, grid_res, grid_res)  — overdensity field δ = ρ/ρ̄ − 1
    voxel_size : float  — physical size of each voxel [kpc]
    x_edges    : (grid_res+1,)  — grid edges along each axis [kpc]

    HINT: use the SAME r_outer as R_OUTER from §35/§36 configuration to
    keep the grid consistent across snapshots.  A different r_outer per
    snapshot would change voxel_size and make threshold comparisons invalid.

    HINT: particles outside the grid (|pos| > r_outer) should be EXCLUDED,
    not wrapped — they are usually unbound tidal debris and should not be
    aliased into the opposite side of the grid.

    HINT: add a small floor rho_grid = max(rho_grid, epsilon) BEFORE
    computing overdensity.  Empty cells have rho = 0; overdensity = −1.
    Setting them to epsilon → δ ≈ −1 (near-empty but not exactly).
    """
    if r_outer is None:
        r_outer = float(np.max(np.linalg.norm(pos_com, axis=1))) * 1.05

    voxel_size = 2.0 * r_outer / grid_res
    x_edges    = np.linspace(-r_outer, r_outer, grid_res + 1)

    rho_grid   = np.zeros((grid_res, grid_res, grid_res), dtype=np.float64)

    # Clip particles to within the grid
    r_all  = np.linalg.norm(pos_com, axis=1)
    inside = r_all < r_outer
    pos_in = pos_com[inside]

    # CIC mass assignment
    pos_norm  = (pos_in + r_outer) / voxel_size   # normalise to [0, grid_res]
    ix        = np.floor(pos_norm[:, 0]).astype(int)
    iy        = np.floor(pos_norm[:, 1]).astype(int)
    iz        = np.floor(pos_norm[:, 2]).astype(int)
    dx        = pos_norm[:, 0] - ix
    dy        = pos_norm[:, 1] - iy
    dz        = pos_norm[:, 2] - iz

    # Clamp indices to avoid out-of-bounds for particles on the boundary
    ix = np.clip(ix, 0, grid_res - 2)
    iy = np.clip(iy, 0, grid_res - 2)
    iz = np.clip(iz, 0, grid_res - 2)

    for (di, wi_x), (dj, wi_y), (dk, wi_z) in [
        ((0, 1.0 - dx), (0, 1.0 - dy), (0, 1.0 - dz)),
        ((0, 1.0 - dx), (0, 1.0 - dy), (1, dz)),
        ((0, 1.0 - dx), (1, dy),        (0, 1.0 - dz)),
        ((0, 1.0 - dx), (1, dy),        (1, dz)),
        ((1, dx),        (0, 1.0 - dy), (0, 1.0 - dz)),
        ((1, dx),        (0, 1.0 - dy), (1, dz)),
        ((1, dx),        (1, dy),        (0, 1.0 - dz)),
        ((1, dx),        (1, dy),        (1, dz)),
    ]:
        np.add.at(rho_grid, (ix + di, iy + dj, iz + dk), wi_x * wi_y * wi_z)

    # Convert to overdensity
    rho_mean  = rho_grid.mean()
    if rho_mean > 1e-30:
        rho_grid  = rho_grid / rho_mean - 1.0
    else:
        rho_grid[:] = -1.0

    # Gaussian smoothing
    if smooth_sigma > 0:
        rho_grid = gaussian_filter(rho_grid, sigma=smooth_sigma)

    return rho_grid, voxel_size, x_edges

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.6 — MINKOWSKI FUNCTIONAL COMPUTATION                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Given a density field ρ(x) and a threshold ρ_th, the EXCURSION SET is:
#   Ω_th = { x : ρ(x) > ρ_th }
#
# The four Minkowski functionals of Ω_th in 3D are:
#   W0  = V(Ω_th)                       — volume [kpc³]
#   W1  = A(∂Ω_th) / 6                  — surface area / 6 [kpc²]
#   W2  = (1/6π) ∫_{∂Ω} (κ_1 + κ_2) dA — mean curvature integral [kpc]
#   W3  = (1/4π) ∫_{∂Ω} κ_1 κ_2 dA     — Euler characteristic / 4π [—]
#
# where κ_1, κ_2 are the principal curvatures of the isosurface ∂Ω_th.
#
# The EULER CHARACTERISTIC χ = 4π W3 is a topological integer:
#   χ = 2(1 − g)  where g is the GENUS (number of handles/holes)
#   χ = 2: single connected blob (sphere-like)   — typical virialised halo
#   χ = 0: torus (one hole)                      — ring structure, rare
#   χ = negative: multiple connected components  — substructure, merger phase
#   χ = 4: two separate blobs (before merging)   — pre-merger epoch
#
# IMPLEMENTATION VIA MARCHING CUBES:
#   The standard approach is to extract the isosurface using the Marching
#   Cubes algorithm (Lorensen & Cline 1987), which produces a triangulated
#   mesh, then integrate curvature over the mesh.
#
#   scikit-image provides this:
#     from skimage.measure import marching_cubes, mesh_surface_area
#     verts, faces, normals, values = marching_cubes(rho_grid, level=rho_th,
#                                                     spacing=(dv, dv, dv))
#   where dv = voxel_size.
#
#   From the mesh:
#     W0 (volume): scipy.spatial.ConvexHull(verts).volume  OR
#                  direct voxel count: V = (rho_grid > rho_th).sum() × dv³
#     W1 (area/6): mesh_surface_area(verts, faces) / 6
#     W2 (curvature): requires per-triangle curvature integration
#     W3 (Euler):    χ = V − E + F  (Euler formula for meshes: vertices −
#                    edges + faces) — OR use discrete Gauss-Bonnet theorem.
#
# DISCRETE APPROXIMATION FOR W2 AND W3:
#   Computing W2 and W3 exactly from the Marching Cubes mesh requires
#   per-vertex curvature estimation, which is moderately complex.
#   A robust discrete approximation uses the CROFTON FORMULA on the
#   voxel grid:
#
#   For the binary mask B = (rho_grid > rho_th):
#     χ (Euler) from the Euler number of the binary array:
#       scipy.ndimage.label gives connected components (topological info)
#       The Euler number can be computed from the 2×2×2 neighbor configuration
#       table (see Torquato & Haslach 2002 or lmfit documentation).
#
#   SIMPLER ROUTE (recommended for this pipeline):
#     Use the digital Euler characteristic from voxel connectivity:
#       from skimage.measure import euler_number
#       chi = skimage.measure.euler_number(B, connectivity=3)
#     This gives χ directly as an integer.
#
#     For W2, use the Crofton formula (mean width):
#       Compute the surface area in three orthogonal projections and average:
#       W2 ≈ (1/6π) × π × (D_x + D_y + D_z) / 3
#       where D_α = mean caliper diameter in direction α.
#       A simpler proxy: W2 ≈ (A_surface) / (2 × r_eff)
#       where r_eff = (3 V / 4π)^{1/3}.
#
# HINT: the skimage functions require installation:
#   pip install scikit-image
#   Wrap the import in a try/except and fall back to a W2 proxy if unavailable.
#
# TIME COMPLEXITY:  O(GRID_RES³ × N_THRESHOLDS)  per snapshot
#                 = O(262144 × 30) ≈ 7.9 × 10^6  — seconds per snapshot
# SPACE COMPLEXITY: O(GRID_RES³)  — binary mask and mesh

def compute_minkowski_functionals(rho_grid, rho_threshold, voxel_size):
    """
    Compute the four Minkowski functionals of the excursion set
    {x : rho_grid(x) > rho_threshold} using a voxel grid.

    Parameters
    ----------
    rho_grid     : (G, G, G)  — smoothed density/overdensity field
    rho_threshold: float      — density threshold defining the excursion set
    voxel_size   : float      — physical size of each voxel [kpc]

    Returns
    -------
    mf : dict with keys:
        'W0'   — volume [kpc³]
        'W1'   — surface area / 6 [kpc²]
        'W2'   — mean curvature integral [kpc]  (proxy if skimage unavailable)
        'W3'   — Euler characteristic / 4π  [—]
        'chi'  — Euler characteristic (integer)
        'T_MF' — thickness shapefinder [kpc]
        'W_MF' — width shapefinder [kpc]
        'L_MF' — length shapefinder [kpc]
        'P_MF' — planarity ∈ [0, 1]
        'F_MF' — filamentarity ∈ [0, 1]
        'n_components' — number of connected components in excursion set
        'valid' — bool: excursion set has enough voxels for valid MFs

    HINT: if the excursion set is empty (no voxels above threshold),
    return all NaN and valid=False.
    HINT: if the excursion set is the entire grid (threshold too low),
    W0 = V_total and W3 = 1/(4π) (single connected blob covering the box).
    """
    dv  = voxel_size
    dv3 = dv**3
    dv2 = dv**2

    nan_mf = {k: np.nan for k in ['W0','W1','W2','W3','chi',
                                    'T_MF','W_MF','L_MF','P_MF','F_MF',
                                    'n_components']}
    nan_mf['valid'] = False

    B = (rho_grid > rho_threshold)
    N_in = B.sum()
    if N_in < MIN_VOXELS_SURFACE:
        return nan_mf

    # W0: volume
    W0 = float(N_in) * dv3

    # W1: surface area / 6
    # Surface area from exposed face count (6-connectivity neighbours)
    # A surface voxel face exists between a True and False voxel.
    # Count such faces in each of the 3 axis directions.
    pad = np.pad(B, pad_width=1, mode='constant', constant_values=False)
    faces_x = np.sum(B & ~pad[2:,  1:-1, 1:-1]) + np.sum(B & ~pad[:-2, 1:-1, 1:-1])
    faces_y = np.sum(B & ~pad[1:-1, 2:,  1:-1]) + np.sum(B & ~pad[1:-1, :-2, 1:-1])
    faces_z = np.sum(B & ~pad[1:-1, 1:-1, 2: ]) + np.sum(B & ~pad[1:-1, 1:-1, :-2])
    A_surface = float(faces_x + faces_y + faces_z) * dv2
    W1 = A_surface / 6.0

    # W3: Euler characteristic via scikit-image (preferred) or fallback
    chi = np.nan
    n_comp = 0
    try:
        from skimage.measure import euler_number, label as sk_label
        chi    = int(euler_number(B, connectivity=3))
        labeled, n_comp = ndimage_label(B)
    except ImportError:
        warnings.warn("scikit-image not found; W3 (Euler characteristic) "
                      "will be NaN.  pip install scikit-image to enable.")
        labeled, n_comp = ndimage_label(B)
    W3 = float(chi) / (4.0 * np.pi) if np.isfinite(chi) else np.nan

    # W2: mean curvature integral
    # Proxy: W2 ≈ W1 / r_eff   where r_eff = (3 W0 / 4π)^{1/3}
    # This is exact for a sphere: W0=4πr³/3, W1=4πr²/6, W2=r, W3=1/(4π)
    r_eff = (3.0 * W0 / (4.0 * np.pi))**(1.0/3.0) if W0 > 0 else 1.0
    W2    = W1 / (r_eff + 1e-30)

    # Shapefinders (Sahni+ 1998)
    T_MF = 3.0 * W0 / (W1 + 1e-30)          # thickness
    W_MF = W1 / (2.0 * W2 + 1e-30)          # width
    L_MF = W2 / (3.0 * W3 + 1e-30) if (np.isfinite(W3) and abs(W3) > 1e-10) \
           else np.nan                         # length

    # Planarity and filamentarity
    P_MF = (W_MF - T_MF) / (W_MF + T_MF + 1e-30)
    F_MF = (L_MF - W_MF) / (L_MF + W_MF + 1e-30) if np.isfinite(L_MF) else np.nan

    return {
        'W0': W0, 'W1': W1, 'W2': W2, 'W3': W3, 'chi': chi,
        'T_MF': T_MF, 'W_MF': W_MF, 'L_MF': L_MF,
        'P_MF': P_MF, 'F_MF': F_MF,
        'n_components': n_comp, 'valid': True
    }

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.7 — MF CURVES: SCANNING OVER DENSITY THRESHOLDS                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Computing MFs at a SINGLE threshold gives one morphology estimate.
# Computing MFs as a FUNCTION of threshold gives a MORPHOLOGY CURVE that
# characterises the full topology of the density field.
#
# The morphology curve (P_MF(ρ_th), F_MF(ρ_th)) traces a path in the
# (planarity, filamentarity) plane as the threshold varies:
#
#   Low threshold (includes whole halo):
#     Single connected blob → P, F both near 0 (sphere-like)
#
#   Intermediate threshold (halo interior):
#     If the halo is prolate: F rises above P → moves toward filament corner
#     If the halo is oblate: P rises above F → moves toward sheet corner
#     If the halo is triaxial: both P and F are intermediate
#
#   High threshold (only the dense core):
#     Core is typically spherical → P, F both drop back toward 0
#
#   MERGER SIGNATURE:
#     During pericentric passage, the intermediate-threshold part of the
#     curve shifts toward the FILAMENT corner (prolate: F > P).
#     After virialisation, the curve returns toward the sphere corner.
#
#   TOPOLOGY CHANGE:
#     If χ(ρ_th) changes between two thresholds, the excursion set has
#     a topology change (a new connected component appeared or disappeared).
#     Tracking χ(ρ_th) is a TOPOLOGICAL MERGER INDICATOR.
#
# IMPLEMENTATION:
#   For each snapshot k in snap_indices_mf:
#     1. Build the density grid: rho_grid, dv, x_edges = build_density_grid(...)
#     2. Determine threshold range:
#          rho_lo = np.percentile(rho_grid, THRESH_PERCENTILE_LO)
#          rho_hi = np.percentile(rho_grid, THRESH_PERCENTILE_HI)
#          thresholds = np.linspace(rho_lo, rho_hi, N_THRESHOLDS)
#     3. For each threshold:
#          mf = compute_minkowski_functionals(rho_grid, th, dv)
#          Store W0, W1, W2, W3, chi, P_MF, F_MF
#     4. Identify the PEAK of P_MF and F_MF along the threshold curve.
#
# TIME COMPLEXITY:  O(N_MF_SNAPS × N_THRESHOLDS × GRID_RES³)
#                 ≈ O(25 × 30 × 262144) ≈ 2 × 10^8 voxel operations
#                 ≈ 3–10 minutes total  (dominant cost of this section)
# SPACE COMPLEXITY: O(N_MF_SNAPS × N_THRESHOLDS × 8 MF quantities)
#                 ≈ 25 × 30 × 8 × 8 bytes ≈ 48 KB — tiny

# TODO: allocate MF arrays
# W0_arr   = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# W1_arr   = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# W2_arr   = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# W3_arr   = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# chi_arr  = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# P_arr    = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# F_arr    = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# T_arr    = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# n_comp_arr = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
# thresh_grid = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)

# TODO: main MF computation loop
# for k, s in enumerate(snap_indices_mf):
#     com     = np.mean(_traj_pos[s], axis=0)
#     pos_c   = _traj_pos[s] - com
#     rho_g, dv, _ = build_density_grid(pos_c, GRID_RES, R_OUTER, SMOOTH_SIGMA)
#     rho_lo  = np.percentile(rho_g, THRESH_PERCENTILE_LO)
#     rho_hi  = np.percentile(rho_g, THRESH_PERCENTILE_HI)
#     thresholds = np.linspace(rho_lo, rho_hi, N_THRESHOLDS)
#     thresh_grid[k] = thresholds
#     for j, th in enumerate(thresholds):
#         mf = compute_minkowski_functionals(rho_g, th, dv)
#         if mf['valid']:
#             W0_arr[k,j]     = mf['W0'];  W1_arr[k,j] = mf['W1']
#             W2_arr[k,j]     = mf['W2'];  W3_arr[k,j] = mf['W3']
#             chi_arr[k,j]    = mf['chi']; P_arr[k,j]  = mf['P_MF']
#             F_arr[k,j]      = mf['F_MF'];T_arr[k,j]  = mf['T_MF']
#             n_comp_arr[k,j] = mf['n_components']

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.8 — MF SUMMARY DIAGNOSTICS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Reduce the full (N_MF_SNAPS × N_THRESHOLDS) MF arrays to a set of
# per-snapshot SUMMARY STATISTICS that can be plotted as time series.
#
# KEY SUMMARY QUANTITIES:
#
# 1. DOMINANT MORPHOLOGY at the median threshold:
#      P_med(t) = P_arr[k, N_THRESHOLDS//2]
#      F_med(t) = F_arr[k, N_THRESHOLDS//2]
#   This gives the bulk-halo morphology at each epoch.
#
# 2. PEAK FILAMENTARITY threshold:
#      j_Fpeak(t) = argmax_j F_arr[k, j]
#      F_peak(t)  = max_j F_arr[k, j]
#      rho_Fpeak(t) = thresh_grid[k, j_Fpeak(t)]
#   This identifies at WHICH density level the halo is most filamentary.
#
# 3. TOPOLOGY CHANGE COUNT:
#      n_chi_jumps(t) = number of thresholds j where |chi_arr[k,j] - chi_arr[k,j-1]| > 0
#   A large n_chi_jumps means the density field has many substructures
#   (connected components that appear/disappear as the threshold varies).
#   At the moment of merging, n_chi_jumps should PEAK (two blobs merge → χ change).
#
# 4. MINIMUM EULER CHARACTERISTIC:
#      chi_min(t) = min_j chi_arr[k, j]
#   χ < 2 means there are HOLES or MULTIPLE COMPONENTS in the excursion set —
#   a topological signature of non-trivial structure (rings, shells, satellites).
#
# 5. MORPHOLOGY CURVE AREA:
#      A_PF(t) = ∫ F_arr[k] d(P_arr[k])  ≈ trapezoid integral
#   The area enclosed by the (P, F) curve as the threshold varies.
#   A large A_PF indicates the halo cycles through a range of morphologies
#   from sheet to filament — characteristic of a triaxial merger remnant.
#
# TIME COMPLEXITY:  O(N_MF_SNAPS × N_THRESHOLDS)  — all in post-processing
# SPACE COMPLEXITY: O(N_MF_SNAPS)  — one scalar per epoch per summary quantity

# TODO: compute MF summary statistics
# P_med_ts     = np.full(N_MF_SNAPS, np.nan)
# F_med_ts     = np.full(N_MF_SNAPS, np.nan)
# F_peak_ts    = np.full(N_MF_SNAPS, np.nan)
# chi_min_ts   = np.full(N_MF_SNAPS, np.nan)
# n_chi_jumps  = np.full(N_MF_SNAPS, np.nan)
# A_PF_ts      = np.full(N_MF_SNAPS, np.nan)
# for k in range(N_MF_SNAPS):
#     valid = np.isfinite(P_arr[k]) & np.isfinite(F_arr[k])
#     if valid.sum() < 3:
#         continue
#     P_med_ts[k]   = P_arr[k, N_THRESHOLDS // 2]
#     F_med_ts[k]   = F_arr[k, N_THRESHOLDS // 2]
#     F_peak_ts[k]  = np.nanmax(F_arr[k])
#     chi_min_ts[k] = np.nanmin(chi_arr[k])
#     chi_v = chi_arr[k][np.isfinite(chi_arr[k])]
#     n_chi_jumps[k] = np.sum(np.diff(chi_v) != 0)
#     A_PF_ts[k]    = float(np.trapz(F_arr[k][valid], P_arr[k][valid]))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9 — PRE-ALLOCATION FOR ALL OUTPUT ARRAYS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Full inventory of arrays (most allocated inline above):
#
#   MULTIPOLE arrays (shape: N_MULTIPOLE_SNAPS × N_MULTIPOLE_SHELLS × L_MAX+1):
#     Etilde_arr   — normalised power Ẽ_l per shell per epoch
#     P_l_arr      — raw power P_l
#
#   Multipole derived (shape: N_MULTIPOLE_SNAPS × N_MULTIPOLE_SHELLS):
#     Q_bar_arr    — Ẽ_2 (quadrupole)
#     D_bar_arr    — Ẽ_1 (dipole / lopsidedness)
#     H_bar_arr    — Ẽ_4 (hexadecapole)
#     phase22_arr  — phase of a_22 (quadrupole orientation)
#     l_peak_arr   — dominant l per shell
#
#   Multipole global (shape: N_MULTIPOLE_SNAPS):
#     Etilde_global  — global Ẽ_l (all particles)
#     phase22_global — global a_22 phase
#     dPA_22_io      — inner-outer quadrupole misalignment [deg]
#
#   MF arrays (shape: N_MF_SNAPS × N_THRESHOLDS):
#     W0_arr, W1_arr, W2_arr, W3_arr — four Minkowski functionals
#     chi_arr    — Euler characteristic (integer)
#     P_arr      — planarity
#     F_arr      — filamentarity
#     n_comp_arr — number of connected components
#     thresh_grid — density thresholds used
#
#   MF summary (shape: N_MF_SNAPS):
#     P_med_ts, F_med_ts — median-threshold planarity and filamentarity
#     F_peak_ts          — peak filamentarity over threshold scan
#     chi_min_ts         — minimum Euler characteristic
#     n_chi_jumps        — topology change count
#     A_PF_ts            — (P, F) curve area
#
# SPACE COMPLEXITY TOTAL:
#   Multipole: (60 × 40 × 9) × 8 bytes × 2 arrays ≈ 345 KB
#   MF:        (25 × 30) × 8 bytes × 8 arrays     ≈ 48 KB
#   TOTAL: under 500 KB — very lightweight

# TODO: verify all allocations above are consistent and present

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.10 — FIGURES (TEN PLANNED)                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── Figure 1: Multipole power spectra P_l at 5 epochs ────────────────────────
# Bar chart: x-axis = l ∈ [0, L_MAX], y-axis = Ẽ_l (log scale).
# One panel per epoch (5 panels horizontal) for the MID SHELL.
# Overplot theoretical predictions:
#   Ẽ_2 for a pure prolate ellipsoid with the §35 axis ratios q, s.
#   This tests whether l=2 alone captures the full shape.
# HINT: for a triaxial ellipsoid with semi-axes a, b, c,
#   the predicted quadrupole power is:
#   Ẽ_2^{theory} ≈ (a² − c²)² / (a² + b² + c²)²   (first-order)
#   This is a rough estimate; the exact prediction requires all l modes.
#
# Expected output: section36_multipole_spectra.png
#
# ── Figure 2: Ẽ_2(r, t) and Ẽ_1(r, t) heatmaps ─────────────────────────────
# Two heatmaps side by side.
# Left:  Quadrupole strength Ẽ_2(r, t).  Log colourscale.
# Right: Dipole strength Ẽ_1(r, t).
# x-axis = time [Gyr], y-axis = log r [kpc].
# Overplot pericentric passages as vertical dashed lines.
# COMPARISON WITH §35: overplot the contour of E_inertia = 0.3 from §35
# on the Ẽ_2 heatmap.  They should trace similar shapes.
#
# Expected output: section36_multipole_heatmaps.png
#
# ── Figure 3: Multipole spectrum Ẽ_l(r) at the epoch of maximum distortion ───
# 2D heatmap with x-axis = l ∈ [1, L_MAX], y-axis = log r.
# Colour = log10(Ẽ_l).
# This is the "angular power spectrum as a function of radius" — the full
# morphological fingerprint of the halo at peak distortion.
# At r < 5 kpc: Ẽ_l ≈ 0 for all l (spherical core).
# At r > 30 kpc: many l modes excited (tidal chaos).
# HINT: mask cells with Ẽ_l < 1e-4 (shot noise floor) in grey.
#
# Expected output: section36_multipole_radius_spectrum.png
#
# ── Figure 4: Quadrupole phase φ_22(r, t) and isophote twist ────────────────
# Left heatmap: phase22_arr (azimuthal orientation of the l=2, m=2 mode).
# Use a CYCLIC colourmap (e.g. hsv or twilight) since phase ∈ [−180°, 180°].
# Right panel: dPA_22_io(t) — inner-outer quadrupole misalignment angle.
# A large dPA_22_io signals a TWISTED HALO (inner and outer halo have
# different major axis orientations) — a smoking gun for triaxiality.
# Overplot the §35 tilt angle θ_tilt_ts on the right panel for comparison.
#
# Expected output: section36_quadrupole_phase.png
#
# ── Figure 5: Minkowski functional curves at 5 epochs ────────────────────────
# Five panels (one per epoch), each showing:
#   W0(ρ_th) / W0_max — normalised volume (decreases monotonically)
#   W1(ρ_th) / W1_max — normalised surface area (peaks at intermediate ρ_th)
#   W3(ρ_th) / (1/4π) — Euler characteristic (integer steps)
# x-axis = ρ_th / ρ_median  (normalised threshold)
# The W3 curve should show STEP FUNCTIONS at thresholds where topology changes.
# HINT: use a secondary y-axis for W3 since it is dimensionless and integer.
#
# Expected output: section36_minkowski_curves.png
#
# ── Figure 6: (P, F) morphology plane at 5 epochs ───────────────────────────
# Single panel: (P_MF, F_MF) plane with the morphology triangle:
#   Corner (0,0) labelled "SPHERE"
#   Corner (1,0) labelled "PANCAKE"
#   Corner (0,1) labelled "FILAMENT"
# Draw the MORPHOLOGY CURVE for each epoch as a coloured line
# (scanning over thresholds, each epoch one colour).
# Each curve connects from (P, F) at ρ_lo to (P, F) at ρ_hi.
# EXPECTED: the curve at the pericentric epoch should be shifted toward
# the FILAMENT corner relative to the pre-merger epoch.
# HINT: mark the MEDIAN threshold point on each curve with a large dot.
#
# Expected output: section36_pf_morphology_plane.png
#
# ── Figure 7: Euler characteristic χ(ρ_th, t) heatmap ───────────────────────
# Heatmap: x-axis = normalised threshold, y-axis = snapshot time [Gyr].
# Colour = χ (Euler characteristic).  Use an integer colourmap.
# This reveals when TOPOLOGY CHANGES occur (χ jumps):
#   Pre-merger: χ = 2 (single MW halo blob)
#   At merger: χ = 4 (two blobs briefly: MW + M31)
#   Merger: χ drops as the blobs merge (χ transition 4 → 2 → 2)
#   Post-merger: χ = 2 (single merged halo)
# HINT: use plt.imshow with interpolation='none' so the integer steps are
# visible as sharp colour boundaries, not smooth gradients.
#
# Expected output: section36_euler_characteristic.png
#
# ── Figure 8: Quadrupole Ẽ_2 vs. tensor ellipticity E ────────────────────────
# Scatter plot: x-axis = ellipticity E = 1 − s from §35 tensor.
#               y-axis = Ẽ_2 from §36 multipole expansion.
# One point per shell per snapshot, coloured by log r.
# If both methods agree perfectly → monotonic curve with slope ∝ (a²−c²)².
# DEVIATIONS ABOVE: multipole sees more elongation than tensor
#   → higher-order angular structure above l=2 contributing to apparent Ẽ_2.
# DEVIATIONS BELOW: tensor sees more elongation than multipole
#   → the elongation is in the MASS distribution not the surface (dense core
#   off-centre; outer halo round).
#
# Expected output: section36_multipole_vs_tensor.png
#
# ── Figure 9: F_MF(t) and χ_min(t) time series ──────────────────────────────
# Three-panel time series:
#   Top:    F_med_ts(t) and P_med_ts(t) — median planarity and filamentarity.
#   Middle: chi_min_ts(t) — minimum Euler characteristic (topology indicator).
#           Mark χ = 2 (single blob) and χ = 4 (two blobs) with horizontal lines.
#   Bottom: n_chi_jumps(t) — number of topology changes per snapshot.
#           High n_chi_jumps means the density field has complex substructure.
# Vertical lines at pericentric passages on all panels.
#
# Expected output: section36_mf_timeseries.png
#
# ── Figure 10: Master summary panel ──────────────────────────────────────────
# 3×2 grid:
#   (0,0) Ẽ_2(r, t) quadrupole heatmap
#   (0,1) Ẽ_1(r, t) dipole heatmap
#   (1,0) (P, F) morphology plane — all epochs overlaid as a heat density
#         (2D histogram of where the morphology curves pass, across all epochs)
#   (1,1) χ(ρ_th, t) Euler characteristic heatmap
#   (2,0) Quadrupole Ẽ_2 vs. tensor E scatter (Fig 8, compressed)
#   (2,1) Five-method comparison time series:
#         s(t) from §35 / Ψ(t) from §35 / Ẽ_2_global(t) / F_med(t) / 1−χ_min/2
#         all normalised to [0,1] — direct visual test of whether all four
#         methods are telling the same morphological story

# TODO: implement all ten figures

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.11 — ANIMATION: MORPHOLOGICAL EVOLUTION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Three-panel animation:
#
# Left  : Multipole power spectrum Ẽ_l at the current epoch for the mid shell.
#         Bar chart with bars for l = 1, 2, ..., L_MAX.
#         The l=2 bar pulses during the pericentric passage.
#         Overplot the theoretical prolate prediction as a dashed line.
#
# Centre: (P_MF, F_MF) morphology plane with the morphology CURVE for the
#         current epoch drawn as a line.  The curve SWEEPS through the
#         (P, F) triangle as the merger progresses.
#         Overplot all previous epoch curves as grey faint lines to show
#         the evolution history.
#         Mark the median-threshold point as a large moving dot.
#
# Right : Running time series of Ẽ_2_global(t) and F_med_ts(t) with a
#         vertical marker at the current frame.  A secondary y-axis shows
#         χ_min(t).
#
# HINT: for the morphology curve, interpolate P_arr and F_arr to a fine
# threshold grid before plotting, so the curve is smooth rather than 30 steps.
# Use scipy.interpolate.interp1d(thresholds, P_arr[k], kind='cubic').
#
# HINT: update the bar chart per frame:
#   for bar, new_h in zip(bars, Etilde_arr[k, mid_shell]):
#       bar.set_height(new_h)
# Reset ylim if Ẽ values span a large range.
#
# Expected output: section36_animation_morphology.mp4

# TODO: implement animation

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.12 — CROSS-SECTION CORRELATIONS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Close the §31–36 six-method diagnostic suite with cross-correlations:
#
#   corr(Ẽ_2_global, E_inertia)    — multipole quadrupole vs. tensor ellipticity (§35)
#                                     HYPOTHESIS: r ≈ 0.9  (should agree closely)
#   corr(Ẽ_1_outer, f_stream)      — dipole lopsidedness vs. stream fraction (§33)
#                                     HYPOTHESIS: both peak at same epoch
#   corr(F_med, T_global)          — MF filamentarity vs. tensor triaxiality (§35)
#                                     HYPOTHESIS: F peaks when T → 1 (prolate)
#   corr(chi_min, n_satellites)    — Euler characteristic vs. number of
#                                     gravitationally bound substructures
#                                     (if subhalo catalogue is available from §26)
#   corr(dPA_22_io, omega_tumble)  — inner-outer quadrupole twist vs. tumbling rate (§35)
#                                     HYPOTHESIS: twisted halos tumble faster
#   corr(A_PF, KL_global)          — morphology curve area vs. KL divergence (§34)
#                                     HYPOTHESIS: both peak at merger epoch
#
# Print as a formatted table:
#   Quantity 1        | Quantity 2       | Pearson r | p-value | §§ involved
#
# Also print the MORPHOLOGICAL CLASSIFICATION TABLE:
#
#   Epoch      | Ẽ_2  | F_med | P_med | χ_min | Dominant shape classification
#   ───────────────────────────────────────────────────────────────────────────
#   t = 0 Gyr  | …    | …     | …     | …     | oblate/prolate/triaxial/sphere
#   t = t_peri | …    | …     | …     | …     | …
#   t = final  | …    | …     | …     | …     | …
#
# Classification rules:
#   Sphere:   Ẽ_2 < 0.05  AND  F_med < 0.1  AND  P_med < 0.1
#   Prolate:  F_med > P_med  AND  F_med > 0.2
#   Oblate:   P_med > F_med  AND  P_med > 0.2
#   Triaxial: F_med ≈ P_med  (within 0.1 of each other, both > 0.1)
#   Complex:  χ_min < 2  (substructure present, topology non-trivial)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.13 — SECTION COMPLETE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Print the output manifest — same pattern as all previous sections.
# Also print the summary statistics table:
#
#   Method        | Peak distortion t | Peak value | Shell | Cross-corr §35 E
#   ──────────────────────────────────────────────────────────────────────────
#   Ẽ_2 (quad)   | …                 | …          | outer | …
#   Ẽ_1 (dipole) | …                 | …          | outer | —
#   F_MF          | …                 | …          | global| …
#   χ_min         | …                 | …          | global| —
#   dPA_22_io     | …                 | …          | all   | —

outputs_36 = [
    "section36_multipole_spectra.png",
    "section36_multipole_heatmaps.png",
    "section36_multipole_radius_spectrum.png",
    "section36_quadrupole_phase.png",
    "section36_minkowski_curves.png",
    "section36_pf_morphology_plane.png",
    "section36_euler_characteristic.png",
    "section36_multipole_vs_tensor.png",
    "section36_mf_timeseries.png",
    "section36_animation_morphology.mp4",
    "section36_summary_panel.png",
]

# TODO: implement output manifest printing

