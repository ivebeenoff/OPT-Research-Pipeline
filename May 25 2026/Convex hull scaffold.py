"""


===============================================================================
SECTION 35 — CONVEX HULL & INERTIA TENSOR METRICS FOR DARK MATTER HALO SHAPE
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
Sections 31–34 were entirely KINEMATIC: they described the velocity, entropy,
and phase-space structure of individual orbits and velocity distributions.
None of them directly addressed the MORPHOLOGY of the merger remnant —
its three-dimensional shape, orientation, and prolateness.

This section asks:

    WHAT IS THE THREE-DIMENSIONAL SHAPE OF THE DARK MATTER HALO
    AFTER THE MERGER, AND HOW DID IT EVOLVE DURING THE COLLISION?

Dark matter halo shape is a critical observable because:
  1. The shape encodes the MERGER HISTORY.  A prolate (cigar-shaped) halo
     indicates a recent radial merger; an oblate (pancake-shaped) halo
     indicates a disc-plane merger; a triaxial halo is the generic outcome
     of multiple mergers from different directions.

  2. Shape drives ORBIT FAMILIES.  A prolate halo supports box orbits that
     fill the whole volume; an oblate halo supports tube orbits that avoid
     the centre.  This connects directly to §32 (frequency ratios) and
     §33 (β anisotropy).

  3. Shape is OBSERVABLE.  Weak gravitational lensing, stellar kinematics of
     dwarf satellites, and the Milky Way tidal stream (Sagittarius) all
     constrain halo shape.  Your simulation predictions can be compared
     directly to those observations.

  4. Halo shape change is a DYNAMICAL CLOCK.  Prolateness peaks just after
     pericentric passage and relaxes toward sphericity over several dynamical
     times.  Measuring this decay gives the virialisation timescale.

Two complementary methods
──────────────────────────
METHOD A — INERTIA TENSOR:
  The reduced inertia tensor I_ij captures the mass distribution's shape
  through its three eigenvalues (a² ≥ b² ≥ c²):
    • Axis ratios: q = b/a (intermediate/major), s = c/a (minor/major)
    • Triaxiality:  T = (a² − b²)/(a² − c²)   ∈ [0, 1]
                    T = 0: oblate; T = 1: prolate; T = 0.5: triaxial
    • Sphericity:   S = c/a   ∈ (0, 1]   (1 = sphere)
    • Ellipticity:  E = 1 − S ∈ [0, 1)   (0 = sphere)
  The eigenvectors define the PRINCIPAL AXES — the orientation of the halo.
  Tracking the principal axes over time gives the TUMBLING RATE of the halo.

  The inertia tensor is sensitive to the BULK MASS DISTRIBUTION.
  It smoothly interpolates shape and is well-defined even for noisy
  particle distributions.

METHOD B — CONVEX HULL:
  The convex hull is the smallest convex volume enclosing all particles
  in a given shell or the whole system.  It measures the GEOMETRIC EXTENT
  in any direction — without assuming ellipsoidal symmetry.

  Key convex hull metrics:
    • Volume V_hull              — total enclosed volume [kpc³]
    • Surface area A_hull        — [kpc²]
    • Isoperimetric ratio:  I_P = 36π V² / A³  ∈ (0, 1]   (1 = sphere)
    • Asphericity: Ψ = 1 − I_P  — departure from sphericity
    • Convex hull axis ratios via PCA of hull vertices
    • Hull volume vs. sphere volume:  V_hull / V_sphere(r_max)
      where V_sphere encloses the same particles — deviation = non-sphericity

  The convex hull is sensitive to OUTLIERS and TIDAL FEATURES.
  A single tidal tail adds enormous convex hull volume while barely changing
  the inertia tensor.  This makes the hull an ideal TIDAL DEBRIS detector.

Complementarity
───────────────
  Inertia tensor  → bulk shape, orientation, tumbling rate, triaxiality
  Convex hull     → tidal extent, asphericity from outliers, hull breathing

  Together:
    • Large Ψ (hull) + small E (inertia) = tidal tail without bulk distortion
    • Large E + small Ψ = bulk elongation without prominent tidal tails
    • Large E + large Ψ = full merger distortion

Key quantities computed in this section
────────────────────────────────────────
  INERTIA TENSOR:
    I_ij(t, R)      — reduced inertia tensor in shell [0, R] at time t
    a(t,R), b(t,R), c(t,R) — principal semi-axes [kpc]
    q(t, R)         — intermediate-to-major axis ratio b/a  ∈ (0, 1]
    s(t, R)         — minor-to-major axis ratio c/a  ∈ (0, 1]
    T(t, R)         — triaxiality parameter  ∈ [0, 1]
    S(t, R)         — sphericity c/a  ∈ (0, 1]
    E(t, R)         — ellipticity 1 − c/a  ∈ [0, 1)
    e_1(t,R),e_2,e_3 — principal axis unit vectors (orientation)
    θ_tilt(t, R)    — misalignment angle between major axis and z-axis [deg]
    Ω_tumble(t, R)  — tumbling rate of major axis [deg/Gyr]
    ΔPA(t, R)       — position angle change [deg]

  ITERATIVE ELLIPSOIDAL INERTIA TENSOR:
    a_iter, b_iter, c_iter — converged semi-axes from iterative algorithm
    q_iter, s_iter, T_iter — converged axis ratios
    n_iter(t, R)           — iterations to convergence (quality indicator)

  CONVEX HULL:
    V_hull(t, R)    — convex hull volume [kpc³]
    A_hull(t, R)    — convex hull surface area [kpc²]
    I_P(t, R)       — isoperimetric ratio 36π V²/A³  ∈ (0, 1]
    Ψ(t, R)         — asphericity 1 − I_P
    r_hull_max(t)   — maximum vertex radius (extent of outermost debris) [kpc]
    V_excess(t, R)  — V_hull / V_ellipsoid(a,b,c) (hull vs. best-fit ellipsoid)
    N_hull_verts(t) — number of vertices on the convex hull

  CROSS-METHOD:
    ΔShape(t, R)    — (Ψ_hull − E_inertia) per shell — tidal vs. bulk
    align_M31(t)    — angle between major axis and M31 direction [deg]
    PA_major(t, R)  — position angle of major axis on sky [deg]

Dependencies
────────────
  scipy.spatial     — ConvexHull (Qhull wrapping via scipy)
  numpy.linalg      — eigh for eigendecomposition of the inertia tensor
  scipy.linalg      — norm, rotation matrices
  Section 26        — traj_pos, traj_r (positions and radii)
  Section 33        — beta_profile (for shape-anisotropy correlation)
  Section 34        — H_aniso_ts (for shape-entropy correlation)

All globals from the parent pipeline are inherited.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import gaussian_filter
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: The most important design choices here are the RADIAL SHELLS and
# the ITERATIVE CONVERGENCE criterion.
#
# RADIAL SHELLS:
#   Computing I_ij and convex hull in a SINGLE global shell mixes the
#   dynamically distinct inner, mid, and outer halo into one number.
#   The standard approach (Dubinski & Carlberg 1991, Bailin & Steinmetz 2005)
#   is to use ELLIPSOIDAL shells of increasing semi-major axis a_shell.
#   We use SPHERICAL shells (simpler) and then iterate the ellipsoidal
#   correction in §35.3.
#
#   Shell edges are logarithmically spaced: this gives equal coverage in
#   log(r) and ensures both the dense inner halo and the sparse outer halo
#   have meaningful shells.
#
# ITERATIVE ELLIPSOIDAL ALGORITHM:
#   The plain inertia tensor computed in spherical shells is BIASED because
#   an oblate halo has more particles on the equatorial plane, so the spherical
#   shell includes DIFFERENT physical volumes of the halo at different angles.
#   The correct approach:
#     1. Compute I_ij in spherical shell → get trial axes a, b, c.
#     2. Redefine shell as ELLIPSOIDAL with those axes: include only particles
#        with m² = x²/a² + y²/b² + z²/c² < 1.
#     3. Recompute I_ij in this ellipsoidal shell → new axes.
#     4. Repeat until axes converge to ITER_TOL fractional change.
#   This is the Katz (1991) / Bailin & Steinmetz (2005) iterative method.
#
# CONVEX HULL SHELLS:
#   The convex hull is not well-defined for all particles simultaneously —
#   it is dominated by the outermost particles.
#   Compute the hull SEPARATELY in each radial shell to get hull(r, t).
#   Minimum particles for a valid 3D convex hull: 4.
#   Below ~20 particles, the hull is very noisy — use MIN_HULL_PARTS.

N_SHAPE_SNAPS    = 60          # number of snapshots for shape analysis
N_SHELL_BINS     = 18          # number of radial shells (log-spaced)
N_GLOBAL_BINS    = 6           # coarser global shells for summary plots
MIN_ITER_PARTS   = 50          # minimum particles per shell for inertia iter
MIN_HULL_PARTS   = 20          # minimum particles per shell for convex hull
ITER_TOL         = 1e-4        # fractional change in axis ratios for convergence
MAX_ITER         = 200         # maximum iterations for ellipsoidal algorithm
ITER_DAMP        = 0.7         # damping factor for axis update (prevents oscillation)
                                # new_a = ITER_DAMP × trial_a + (1 − ITER_DAMP) × old_a
                                # smaller values = more stable but slower convergence

# Radial shell boundaries [kpc].
# Will be set dynamically from the particle distribution:
#   R_INNER = percentile(r0, 5)    — avoid nearly-empty central bin
#   R_OUTER = percentile(r0, 95)   — avoid the very outermost outliers
# These are just sentinel values; replace in §35.1.
R_INNER         = 1.0           # [kpc] innermost shell edge
R_OUTER         = 120.0         # [kpc] outermost shell edge

# Orientation tracking.
# The major axis is identified at the FIRST snapshot and tracked forward
# by choosing the eigenvector with the smallest angle to the previous frame.
# This prevents 180° flips (eigenvectors are defined up to sign).
TRACK_ORIENT     = True         # whether to track principal axis orientation

# Convergence diagnostics.
SAVE_ITER_HISTORY = True        # save convergence curve for one shell per snap

# Animation.
ANIM_FPS_35      = 18
ANIM_DPI_35      = 100
ANIM_BITRATE_35  = 1600

print("\n" + "="*80)
print("  SECTION 35 · Convex Hull & Inertia Tensor — Halo Shape Diagnostics")
print("="*80)
print(f"  Shape snapshots  : {N_SHAPE_SNAPS}")
print(f"  Radial shells    : {N_SHELL_BINS}")
print(f"  Iter tolerance   : {ITER_TOL:.1e}")
print(f"  Max iterations   : {MAX_ITER}")
print(f"  Iter damping     : {ITER_DAMP}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.1 — LOAD POSITIONS AND DEFINE SHELLS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: This section needs ONLY positions — no velocities required.
# Reuse _traj_pos and _traj_r from Section 26 with the standard try/except.
#
# Centre of mass correction:
#   The inertia tensor and convex hull MUST be computed in the centre-of-mass
#   (COM) frame.  At early times the COM coincides with the MW centre.
#   After the merger the COM drifts.  Always subtract the instantaneous COM:
#     com(t) = np.average(pos[t], weights=mass, axis=0)
#     pos_com[t] = pos[t] - com(t)
#   If particle masses are equal (most common in DM simulations):
#     com(t) = np.mean(pos[t], axis=0)
#
#   IMPORTANT SUBTLETY: the COM should be computed from ALL particles, not
#   just the N_SHAPE_PARTICLES subset you are analysing.  Use the full
#   population for the COM and then subsample for the shape computation.
#
# Radial shell edges:
#   r_edges = np.logspace(log10(R_INNER), log10(R_OUTER), N_SHELL_BINS + 1)
#   r_mid   = 10 ** (0.5 * (log10(r_edges[:-1]) + log10(r_edges[1:])))
#
#   Set R_INNER and R_OUTER dynamically:
#     R_INNER = max(0.5, np.percentile(r0_all, 2))
#     R_OUTER = np.percentile(r0_all, 97)
#   where r0_all = _traj_r[0] is the initial radii of ALL particles.
#
# Enclosed-radius shells for cumulative analysis:
#   Some quantities (axis ratio profiles) are reported as functions of
#   ENCLOSED radius rather than shell width.  Pre-compute:
#     r_enc   = np.logspace(log10(R_INNER), log10(R_OUTER), N_GLOBAL_BINS + 1)
#   and for each r_enc[j] use ALL particles inside that radius.
#
# TIME COMPLEXITY:  O(ns × N)  — just array indexing
# SPACE COMPLEXITY: O(N_SHAPE_SNAPS × N × 3)  ≈ 60 × 2000 × 3 × 8 bytes ≈ 3 MB

# TODO: implement trajectory loading
_traj_pos   = None   # (ns, N, 3) [kpc]  — replace with inherited
_traj_r     = None   # (ns, N)    [kpc]
_r0         = None   # (N,)       initial radii
_group      = None   # (N,)       0=inner,1=mid,2=outer,3=M31
_N          = 0
_mass       = None   # (N,) or scalar — particle masses [M_sun]

# TODO: set shell edges dynamically and subsample snapshots
# snap_indices = np.linspace(0, ns - 1, N_SHAPE_SNAPS, dtype=int)
# R_INNER      = max(0.5, float(np.percentile(_traj_r[0], 2)))
# R_OUTER      = float(np.percentile(_traj_r[0], 97))
# r_edges      = np.logspace(np.log10(R_INNER), np.log10(R_OUTER), N_SHELL_BINS + 1)
# r_mid        = np.sqrt(r_edges[:-1] * r_edges[1:])   # geometric mean
# r_enc        = np.logspace(np.log10(R_INNER), np.log10(R_OUTER), N_GLOBAL_BINS + 1)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.2 — REDUCED INERTIA TENSOR                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The REDUCED inertia tensor of a set of particles is:
#
#   I_ij  =  (1/N) Σ_k  x_k,i  x_k,j
#
# where x_k = (x, y, z) is the position of particle k relative to the COM.
# The factor 1/N normalises by particle number so the tensor is independent
# of how many particles are in the shell (making shells comparable).
#
# NOTE: This is the REDUCED tensor (no mass weighting, no r² subtraction).
# It is NOT the standard moment-of-inertia tensor from classical mechanics
# (which has Σ m_k |x_k|² δ_ij − x_k,i x_k,j on the diagonal).
# The reduced tensor's eigenvalues are a², b², c² directly (the squared
# semi-axes), making axis ratio computation trivial.
#
# ALTERNATIVE: the ITERATIVE REDUCED TENSOR (§35.3) uses 1/r_k² weighting:
#   I_ij  =  (1/N) Σ_k  x_k,i x_k,j / r_k²
# This down-weights distant particles and is better for finding the
# LOCAL shape near a given ellipsoidal surface.
# The choice matters most for NESTED SHELLS where you want the shape
# AT radius r, not interior to it.
#
# Eigendecomposition:
#   I_ij is real symmetric → use numpy.linalg.eigh (not eig!).
#   eigh guarantees real eigenvalues and orthonormal eigenvectors.
#   Eigenvalues λ_1 ≤ λ_2 ≤ λ_3 (ascending order from eigh).
#   Semi-axes: a = √λ_3, b = √λ_2, c = √λ_1  (so a ≥ b ≥ c).
#
# HINT: numpy.linalg.eigh returns eigenvalues in ASCENDING order.
# Always sort: idx = np.argsort(evals)[::-1] to get DESCENDING (a, b, c).
# Failure to do this produces randomly ordered axes and q, s > 1 — a
# clear sign of the sorting bug.
#
# HINT: if fewer than MIN_ITER_PARTS particles are in the shell,
# the tensor estimate is too noisy — return NaN for all outputs.
#
# TIME COMPLEXITY:  O(N_shell)  per call — dominated by the sum, not the 3×3 eigh
# SPACE COMPLEXITY: O(1)  — just the 3×3 tensor

def reduced_inertia_tensor(pos_shell):
    """
    Compute the reduced inertia tensor and its eigendecomposition.

    Parameters
    ----------
    pos_shell : (N, 3)  — particle positions in the shell, COM-subtracted [kpc]

    Returns
    -------
    evals  : (3,)   — eigenvalues λ_1 ≥ λ_2 ≥ λ_3, i.e. a² ≥ b² ≥ c²
    evecs  : (3, 3) — eigenvectors as COLUMNS: evecs[:, 0] = major axis
    a, b, c: float  — semi-axes √λ in DESCENDING order [kpc]
    q      : float  — intermediate-to-major axis ratio b/a ∈ (0,1]
    s      : float  — minor-to-major axis ratio c/a ∈ (0,1]
    T      : float  — triaxiality (a²−b²)/(a²−c²) ∈ [0,1]

    HINT: guard against all particles at the origin (r → 0 numerical noise)
    by checking that the trace of I is above a floor before eigensolution.

    HINT: for a shell with very few particles (< 10), the tensor is rank-
    deficient or nearly so, producing near-zero eigenvalues.  Return NaN
    rather than reporting a = 0 (which would give q/s = NaN anyway).
    """
    N = len(pos_shell)
    if N < MIN_ITER_PARTS:
        nan3 = np.full(3, np.nan)
        return nan3, np.full((3, 3), np.nan), np.nan, np.nan, np.nan, np.nan, np.nan

    I = np.einsum('ki,kj->ij', pos_shell, pos_shell) / N    # (3, 3)

    if np.trace(I) < 1e-20:
        nan3 = np.full(3, np.nan)
        return nan3, np.full((3, 3), np.nan), np.nan, np.nan, np.nan, np.nan, np.nan

    evals_asc, evecs_asc = np.linalg.eigh(I)                # ascending order
    idx   = np.argsort(evals_asc)[::-1]                     # descending: a,b,c
    evals = evals_asc[idx]
    evecs = evecs_asc[:, idx]                                # columns are eigenvectors

    a  = float(np.sqrt(max(evals[0], 0.0)))
    b  = float(np.sqrt(max(evals[1], 0.0)))
    c  = float(np.sqrt(max(evals[2], 0.0)))
    q  = b / (a + 1e-30)
    s  = c / (a + 1e-30)

    # Triaxiality T = (a²−b²)/(a²−c²); 0=oblate, 1=prolate, 0.5=triaxial
    denom = evals[0] - evals[2]
    T     = float((evals[0] - evals[1]) / (denom + 1e-30)) if denom > 1e-20 else np.nan

    return evals, evecs, a, b, c, q, s, T


def axis_ratios_from_eigenvalues(evals):
    """
    Extract q, s, T, S, E from a sorted eigenvalue array (a²,b²,c²).

    Parameters
    ----------
    evals : (3,) sorted descending  — eigenvalues of the reduced inertia tensor

    Returns
    -------
    q : float  — b/a
    s : float  — c/a
    T : float  — triaxiality (a²−b²)/(a²−c²)
    S : float  — sphericity c/a (= s)
    E : float  — ellipticity 1 − c/a (= 1 − s)

    This is a convenience wrapper so the main loop is not cluttered.
    """
    a2, b2, c2 = evals[0], evals[1], evals[2]
    a  = np.sqrt(max(a2, 0.0))
    q  = np.sqrt(max(b2, 0.0)) / (a + 1e-30)
    s  = np.sqrt(max(c2, 0.0)) / (a + 1e-30)
    T  = (a2 - b2) / (a2 - c2 + 1e-30) if (a2 - c2) > 1e-20 else np.nan
    return float(q), float(s), float(T), float(s), float(1.0 - s)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.3 — ITERATIVE ELLIPSOIDAL INERTIA TENSOR                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The simple reduced tensor (§35.2) computed in a SPHERICAL shell is biased:
# it includes particles in the corners of the sphere that lie OUTSIDE the
# true ellipsoid of the same volume.
#
# The Katz–Dubinski–Bailin–Steinmetz iterative algorithm fixes this by
# redefining the shell as ELLIPSOIDAL at each iteration:
#
#   Iteration:
#     Given current semi-axes (a_n, b_n, c_n) and eigenvectors (e_1, e_2, e_3):
#     1. Rotate all particle positions into the principal frame:
#          x' = (pos · e_1, pos · e_2, pos · e_3)
#     2. Define ellipsoidal coordinate:
#          m_k = sqrt( x'_k1²/a_n² + x'_k2²/b_n² + x'_k3²/c_n² )
#     3. Select particles with m_k < 1 (inside the current trial ellipsoid)
#        AND m_k > m_inner (outside the inner boundary, if using a shell).
#     4. Recompute the reduced tensor WITH 1/r_k² weighting (reduces outer bias):
#          I_ij = (1/N) Σ_k (x_k,i x_k,j) / r_k²
#        where r_k = |pos_k| in the ORIGINAL (unrotated) frame.
#     5. Diagonalise → new (a_{n+1}, b_{n+1}, c_{n+1}), new eigenvectors.
#     6. Apply damping:
#          a_{n+1} ← ITER_DAMP × a_{n+1} + (1 − ITER_DAMP) × a_n
#     7. Check convergence: max(|Δq/q|, |Δs/s|) < ITER_TOL.
#     8. If N_inside < MIN_ITER_PARTS at any step, abort → return NaN.
#
# CONVERGENCE NOTES:
#   • Most halos converge in 5–20 iterations for a virialised halo.
#   • Halos mid-merger may take 50–100 iterations or FAIL TO CONVERGE
#     due to rapidly changing shape.  Use n_iter = MAX_ITER as a flag.
#   • The 1/r² weighting is optional but strongly recommended for
#     SHELL estimates (as opposed to enclosed estimates).
#     For enclosed estimates (all particles inside radius R), the unweighted
#     tensor is fine and converges faster.
#   • Damping prevents oscillation between two configurations; ITER_DAMP = 0.7
#     is a safe default.  Set to 1.0 to disable damping (faster but may
#     oscillate for triaxial halos).
#
# IMPORTANT: the VOLUME of the enclosing ellipsoid is NOT conserved between
# iterations unless you rescale the axes by (abc)^{1/3} at each step.
# Many implementations NORMALISE to keep V = (4π/3)abc = V_sphere(r_target).
# Without normalisation, the algorithm can shrink to a small prolate needle
# or inflate to include all particles — both are wrong.
# Always normalise: a_n → a_n / (a_n b_n c_n)^{1/3} × r_target
#
# TIME COMPLEXITY:  O(MAX_ITER × N_shell)  per call — typically O(20 × N)
#                   With N_shell ≈ 100 and MAX_ITER = 200:  2 × 10^4 ops
# SPACE COMPLEXITY: O(N_shell)  — working arrays in place

def iterative_inertia_tensor(pos_all, r_target, r_inner=0.0,
                              weighted=True, normalise_volume=True):
    """
    Compute the shape of a DM halo shell using the iterative ellipsoidal
    inertia tensor method (Katz 1991; Bailin & Steinmetz 2005).

    Parameters
    ----------
    pos_all         : (N, 3)  — ALL particle positions, COM-subtracted [kpc]
    r_target        : float   — target outer ellipsoidal semi-major axis [kpc]
    r_inner         : float   — inner ellipsoidal semi-major axis (for shells)
                               set to 0 for enclosed (cumulative) estimates
    weighted        : bool    — use 1/r² weighting (recommended for shells)
    normalise_volume: bool    — keep (abc)^{1/3} = r_target at each step

    Returns
    -------
    result : dict with keys:
        'a', 'b', 'c'         — converged semi-axes [kpc]
        'q', 's', 'T'         — axis ratios and triaxiality
        'evecs'               — (3,3) principal axes as columns (e1, e2, e3)
        'n_iter'              — iterations to convergence
        'converged'           — bool: did the algorithm converge?
        'n_particles'         — number of particles in the final ellipsoidal shell
        'iter_history'        — list of (q, s) per iteration (if SAVE_ITER_HISTORY)

    HINT: initialise axes as a = b = c = r_target (spherical start).
    HINT: initialise eigenvectors as the identity matrix.
    HINT: if at any iteration N_inside < MIN_ITER_PARTS, abort and return
          a dict with all shape quantities set to NaN and converged=False.
    HINT: track (q, s) at each iteration to diagnose oscillation — if
          |q_{n} − q_{n−2}| < ITER_TOL but |q_{n} − q_{n−1}| > ITER_TOL
          the algorithm is oscillating; reduce damping or declare convergence
          at the mean of the last two values.
    """
    N      = len(pos_all)
    r_all  = np.linalg.norm(pos_all, axis=1)

    # Initial guess: sphere of radius r_target
    a_n, b_n, c_n = r_target, r_target, r_target
    evecs_n = np.eye(3)

    q_prev, s_prev = 1.0, 1.0
    iter_history   = []
    converged      = False

    nan_result = {
        'a': np.nan, 'b': np.nan, 'c': np.nan,
        'q': np.nan, 's': np.nan, 'T': np.nan,
        'evecs': np.full((3, 3), np.nan),
        'n_iter': MAX_ITER, 'converged': False,
        'n_particles': 0, 'iter_history': []
    }

    for n in range(MAX_ITER):
        # Step 1: rotate positions into current principal frame
        x_prime = pos_all @ evecs_n               # (N, 3) — dot with column vectors

        # Step 2: compute ellipsoidal coordinate m
        m2 = (x_prime[:, 0] / (a_n + 1e-30))**2 + \
             (x_prime[:, 1] / (b_n + 1e-30))**2 + \
             (x_prime[:, 2] / (c_n + 1e-30))**2
        m  = np.sqrt(m2)

        # Step 3: select particles inside the ellipsoidal shell
        m_inner_scaled = r_inner / (r_target + 1e-30)
        mask = (m < 1.0) & (m > m_inner_scaled)
        N_in = mask.sum()

        if N_in < MIN_ITER_PARTS:
            return nan_result

        pos_in = pos_all[mask]

        # Step 4: compute (optionally weighted) reduced tensor
        if weighted:
            r_in  = r_all[mask]
            w     = 1.0 / (r_in**2 + 1e-20)         # 1/r² weights
            W     = w.sum()
            I_mat = np.einsum('k,ki,kj->ij', w, pos_in, pos_in) / (W + 1e-30)
        else:
            I_mat = np.einsum('ki,kj->ij', pos_in, pos_in) / N_in

        # Step 5: eigendecompose the new tensor
        evals_asc, evecs_asc = np.linalg.eigh(I_mat)
        idx    = np.argsort(evals_asc)[::-1]
        evals  = evals_asc[idx]
        evecs_trial = evecs_asc[:, idx]

        a_trial = float(np.sqrt(max(evals[0], 0.0)))
        b_trial = float(np.sqrt(max(evals[1], 0.0)))
        c_trial = float(np.sqrt(max(evals[2], 0.0)))

        # Step 6: volume normalisation
        if normalise_volume:
            abc = a_trial * b_trial * c_trial
            if abc > 1e-30:
                scale   = r_target / (abc**(1.0/3.0))
                a_trial *= scale
                b_trial *= scale
                c_trial *= scale

        # Step 7: damping
        a_new = ITER_DAMP * a_trial + (1.0 - ITER_DAMP) * a_n
        b_new = ITER_DAMP * b_trial + (1.0 - ITER_DAMP) * b_n
        c_new = ITER_DAMP * c_trial + (1.0 - ITER_DAMP) * c_n

        # Eigenvector sign-flip correction (ensure continuity)
        for col in range(3):
            if np.dot(evecs_trial[:, col], evecs_n[:, col]) < 0:
                evecs_trial[:, col] *= -1.0

        evecs_n = evecs_trial
        q_new   = b_new / (a_new + 1e-30)
        s_new   = c_new / (a_new + 1e-30)

        if SAVE_ITER_HISTORY:
            iter_history.append((q_new, s_new))

        # Step 8: convergence check
        delta_q = abs(q_new - q_prev) / (q_prev + 1e-30)
        delta_s = abs(s_new - s_prev) / (s_prev + 1e-30)
        if max(delta_q, delta_s) < ITER_TOL and n > 2:
            converged = True
            a_n, b_n, c_n = a_new, b_new, c_new
            break

        a_n, b_n, c_n = a_new, b_new, c_new
        q_prev, s_prev = q_new, s_new

    denom = evals[0] - evals[2]
    T     = float((evals[0] - evals[1]) / (denom + 1e-30)) if denom > 1e-20 else np.nan

    return {
        'a': a_n, 'b': b_n, 'c': c_n,
        'q': b_n / (a_n + 1e-30),
        's': c_n / (a_n + 1e-30),
        'T': T,
        'evecs': evecs_n,
        'n_iter': n + 1,
        'converged': converged,
        'n_particles': int(N_in),
        'iter_history': iter_history
    }

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.4 — ORIENTATION TRACKING AND TUMBLING RATE                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The ORIENTATION of the dark matter halo is as physically informative as
# its axis ratios.  Key questions:
#
#   1. Is the major axis aligned with the Milky Way–M31 direction?
#      If yes, this is TIDAL LOCKING — the halo has been tidally deformed
#      into alignment with the orbital plane.  This has been observed in
#      cosmological simulations (Bailin & Steinmetz 2004).
#
#   2. Is the major axis aligned with the original DISC PLANE?
#      If the halo major axis is the z-axis, the disc is embedded in a
#      prolate halo — relevant for the stability of the disc.
#
#   3. How fast is the halo TUMBLING (rotating its principal axes)?
#      The tumbling rate Ω_tumble is:
#        Ω_tumble = dθ/dt   [deg/Gyr]
#      where θ is the angle between the major axis at consecutive epochs.
#      A fast-tumbling halo is still dynamically active (recently merged).
#      A static halo (Ω_tumble → 0) is virialised.
#
# Implementation — angle tracking
# ─────────────────────────────────
# At each snapshot, the major axis is the first eigenvector e_1(t, R).
# Eigenvectors are defined only up to sign — e_1 and −e_1 are equivalent.
# To track orientation continuously:
#   1. At t=0, set reference direction e_ref = e_1(0, R).
#   2. At each subsequent time t:
#        if dot(e_1(t), e_ref) < 0:  e_1(t) ← −e_1(t)   (flip to same hemisphere)
#        θ(t) = arccos( |dot(e_1(t), e_ref)| )
#        e_ref ← e_1(t)    (update reference for NEXT step)
#   NOTE: use arccos(|dot|) not arccos(dot) — the absolute value means
#   θ ∈ [0°, 90°] instead of [0°, 180°], accounting for the sign degeneracy.
#
# The tumbling rate between consecutive epochs k and k+1:
#   Ω_k = θ_{k,k+1} / Δt_{k,k+1}    [deg/Gyr]
#
# Misalignment with the MW–M31 orbital axis:
#   r_M31(t) = mean position of M31 particles at time t
#   n_M31    = r_M31 / |r_M31|    — unit vector toward M31
#   φ_M31(t) = arccos( |dot(e_1(t), n_M31(t))| )   [deg]
#   φ_M31 ≈ 0 → halo aligned with M31 (tidally locked)
#   φ_M31 ≈ 90 → halo perpendicular to M31
#
# TIME COMPLEXITY:  O(N_SHAPE_SNAPS × N_SHELL_BINS)  — just dot products
# SPACE COMPLEXITY: O(N_SHAPE_SNAPS × N_SHELL_BINS × 3)  — eigenvectors

def track_orientation(evec_series, dt_arr):
    """
    Track the orientation of a principal axis over time, correcting
    for eigenvector sign flips, and compute the tumbling rate.

    Parameters
    ----------
    evec_series : (T, 3)  — major-axis eigenvector at each of T epochs
    dt_arr      : (T-1,) — time intervals between consecutive epochs [Gyr]

    Returns
    -------
    theta_ts    : (T,)   — cumulative angle from t=0 [deg]
    omega_ts    : (T-1,) — instantaneous tumbling rate [deg/Gyr]

    HINT: if an eigenvector at some epoch is NaN (convergence failed),
    carry forward the previous valid eigenvector and mark that epoch's
    tumbling rate as NaN.
    HINT: large single-step Δθ > 45° is usually a sign of a convergence
    failure or sign flip, not true physical tumbling.  Flag these epochs.
    """
    T        = len(evec_series)
    theta_ts = np.full(T, np.nan)
    omega_ts = np.full(T - 1, np.nan)
    theta_ts[0] = 0.0
    e_ref       = evec_series[0].copy()

    for k in range(1, T):
        e_k = evec_series[k]
        if not np.all(np.isfinite(e_k)):
            continue
        # Sign flip correction
        if np.dot(e_k, e_ref) < 0:
            e_k = -e_k
        cos_theta       = np.clip(np.dot(e_k, e_ref), -1.0, 1.0)
        dtheta          = float(np.degrees(np.arccos(abs(cos_theta))))
        theta_ts[k]     = theta_ts[k - 1] + dtheta if np.isfinite(theta_ts[k - 1]) else dtheta
        omega_ts[k - 1] = dtheta / (dt_arr[k - 1] + 1e-30)
        e_ref           = e_k

    return theta_ts, omega_ts

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.5 — CONVEX HULL METRICS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The convex hull of a point set is the smallest convex polytope that contains
# all the points.  scipy.spatial.ConvexHull wraps the Qhull library.
#
# Key metrics extracted from the hull:
#
# 1. VOLUME V_hull and SURFACE AREA A_hull
#    Available directly from hull.volume and hull.area after construction.
#
# 2. ISOPERIMETRIC RATIO I_P:
#    For a 3D shape, the isoperimetric inequality states:
#      36π V² ≤ A³   with equality only for a sphere.
#    The isoperimetric ratio:
#      I_P = 36π V² / A³   ∈ (0, 1]
#    is a pure shape measure: I_P = 1 for a sphere, I_P → 0 for a needle.
#    ASPHERICITY: Ψ = 1 − I_P   ∈ [0, 1)
#
# 3. HULL VERTICES and their distribution:
#    The vertices of the convex hull are the "extreme" particles — those
#    that define the outermost boundary.
#    hull.vertices gives the indices into the input point array.
#    The vertex distribution on the sphere (mapping each to its direction
#    unit vector r̂) reveals ANISOTROPIC EXTENT:
#      - A spherical halo has vertices roughly uniformly distributed on S².
#      - A prolate halo has vertices clustered near the poles.
#      - A tidal tail adds a cluster of vertices in one direction.
#
# 4. HULL PCA (independent of inertia tensor):
#    Apply PCA to the hull VERTEX positions to get hull axis ratios.
#    These differ from the inertia tensor ratios because:
#      - Hull PCA is sensitive to OUTLIERS (one tidal stream can dominate)
#      - Inertia tensor is weighted by ALL particles, not just extreme ones
#    hull_q_pca = b_pca / a_pca   — hull intermediate/major axis ratio
#    hull_s_pca = c_pca / a_pca   — hull minor/major axis ratio
#
# 5. MAXIMUM RADIAL EXTENT:
#    r_hull_max = max(|hull_vertices|)   — the outermost particle radius
#    This tracks tidal debris more sensitively than any radial percentile.
#
# 6. HULL VOLUME RATIO V_excess:
#    Compare the hull volume to the best-fit ellipsoid volume:
#      V_ellipsoid = (4π/3) a b c
#    V_excess = V_hull / V_ellipsoid
#    V_excess > 1 means the hull is LARGER than the best-fit ellipsoid —
#    there are particles in the corners that the ellipsoid misses.
#    This happens when tidal debris is non-ellipsoidal.
#    V_excess ≈ 1 means the particle distribution is well-described by an
#    ellipsoid (as the inertia tensor assumed).
#
# NUMERICAL PITFALLS:
#   • The convex hull FAILS (QhullError) if:
#       - All points are coplanar (rank-deficient in 3D) — very rare
#       - Fewer than 4 non-coplanar points — possible for sparse outer shells
#       - Points are numerically identical — add tiny jitter: pos += 1e-8 * randn
#   • ALWAYS wrap ConvexHull in try/except QhullError.
#   • For SHELL estimates, restrict to particles in the shell before calling
#     ConvexHull — otherwise the hull is just the global extent.
#   • Large halos (N > 10^5) can be slow for ConvexHull.  Subsample to
#     HULL_MAX_PART = 2000 randomly chosen particles from the shell.
#
# TIME COMPLEXITY:  O(N_hull log N_hull)  per call for Qhull
#                   O(N_SHAPE_SNAPS × N_SHELL_BINS × N_hull log N_hull)  total
#                 ≈ O(60 × 18 × 200 × 8) ≈ 1.7 × 10^6  — fast, ~seconds
# SPACE COMPLEXITY: O(N_hull)  per call — hull object

HULL_MAX_PART = 2000   # max particles passed to ConvexHull per shell (subsampled)

def convex_hull_metrics(pos_shell, a_tensor=None, b_tensor=None, c_tensor=None):
    """
    Compute convex hull shape metrics for a set of particle positions.

    Parameters
    ----------
    pos_shell : (N, 3)  — particle positions in shell [kpc], COM-subtracted
    a_tensor  : float   — optional: major semi-axis from inertia tensor [kpc]
                          (used to compute V_excess; pass None to skip)
    b_tensor  : float   — optional: intermediate semi-axis [kpc]
    c_tensor  : float   — optional: minor semi-axis [kpc]

    Returns
    -------
    metrics : dict with keys:
        'V_hull'         — hull volume [kpc³]
        'A_hull'         — hull surface area [kpc²]
        'I_P'            — isoperimetric ratio 36π V²/A³ ∈ (0,1]
        'Psi'            — asphericity 1 − I_P
        'r_hull_max'     — maximum vertex radius [kpc]
        'N_verts'        — number of hull vertices
        'hull_q_pca'     — hull PCA intermediate/major ratio
        'hull_s_pca'     — hull PCA minor/major ratio
        'V_excess'       — V_hull / V_ellipsoid (None if a/b/c not provided)
        'valid'          — bool: hull was successfully computed

    HINT: for PCA of hull vertices, use np.linalg.svd on the centred vertex
    matrix.  The singular values are proportional to the axis extents.
    PCA axis ratios: hull_q = singular_values[1]/singular_values[0], etc.

    HINT: add jitter pos_shell += rng.normal(0, 1e-8, pos_shell.shape) before
    calling ConvexHull to prevent QhullError from coincident points.
    """
    nan_metrics = {
        'V_hull': np.nan, 'A_hull': np.nan, 'I_P': np.nan, 'Psi': np.nan,
        'r_hull_max': np.nan, 'N_verts': 0,
        'hull_q_pca': np.nan, 'hull_s_pca': np.nan,
        'V_excess': np.nan, 'valid': False
    }

    pos = pos_shell.copy()
    if len(pos) < 4:
        return nan_metrics

    # Subsample if too many particles
    if len(pos) > HULL_MAX_PART:
        idx = np.random.choice(len(pos), HULL_MAX_PART, replace=False)
        pos = pos[idx]

    # Tiny jitter to prevent coplanar degenerate inputs
    rng = np.random.default_rng(seed=42)
    pos += rng.normal(0.0, 1e-8, pos.shape)

    try:
        hull = ConvexHull(pos)
    except QhullError:
        return nan_metrics

    V   = float(hull.volume)
    A   = float(hull.area)
    I_P = float(36.0 * np.pi * V**2 / (A**3 + 1e-30))
    I_P = min(I_P, 1.0)   # clamp: numerical errors can push slightly above 1

    verts      = pos[hull.vertices]
    r_max      = float(np.max(np.linalg.norm(verts, axis=1)))
    n_verts    = len(hull.vertices)

    # PCA of hull vertices
    verts_c    = verts - verts.mean(axis=0)
    _, sv, _   = np.linalg.svd(verts_c, full_matrices=False)
    hull_q_pca = float(sv[1] / (sv[0] + 1e-30))
    hull_s_pca = float(sv[2] / (sv[0] + 1e-30))

    # Volume excess relative to inertia tensor ellipsoid
    V_excess = np.nan
    if (a_tensor is not None) and np.isfinite(a_tensor):
        V_ell    = (4.0 * np.pi / 3.0) * a_tensor * b_tensor * c_tensor
        V_excess = float(V / (V_ell + 1e-30))

    return {
        'V_hull': V, 'A_hull': A,
        'I_P': I_P, 'Psi': 1.0 - I_P,
        'r_hull_max': r_max, 'N_verts': n_verts,
        'hull_q_pca': hull_q_pca, 'hull_s_pca': hull_s_pca,
        'V_excess': V_excess, 'valid': True
    }

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.6 — SHAPE PROFILE FUNCTIONS (ENCLOSED AND SHELL)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# There are TWO distinct shape profiles:
#
# A) ENCLOSED profile — shape of ALL particles inside radius R:
#      q_enc(R, t), s_enc(R, t), T_enc(R, t)
#    This smoothly traces how the shape changes from the centre outward.
#    The innermost bin dominates at small R; the outermost dominates at large R.
#    The enclosed profile is dominated by the DENSEST inner regions —
#    the core is nearly spherical in most simulations (NFW-like profiles).
#
# B) SHELL profile — shape of particles IN THE SHELL [R_inner, R_outer]:
#    q_shell(R, t), s_shell(R, t), T_shell(R, t)
#    This reveals where the shape TRANSITION occurs.
#    It is noisier (fewer particles per shell) but physically cleaner.
#    Use the iterative tensor (§35.3) for the shell profile.
#
# The standard approach in the literature (e.g. Vera-Ciro+ 2011, Allgood+ 2006)
# is to report the ENCLOSED profile using the iterative algorithm, with
# r_target equal to the semi-major axis of the enclosing ellipsoid.
# This avoids the shell-width ambiguity but conflates inner and outer shape.
#
# RECOMMENDATION: compute BOTH and report the shell profile as the primary
# diagnostic, with the enclosed profile as a cross-check.
#
# Radial trend expected for a merger remnant:
#   Inner (r < 5 kpc):    q → 1, s → 1  (spherical core, deep potential)
#   Mid (5–30 kpc):       q ≈ 0.7–0.9   (mildly triaxial)
#   Outer (r > 30 kpc):   q ≈ 0.4–0.7   (more prolate, tidal distortion)
#   Merger-induced:       s drops dramatically in outer shell immediately
#                         after pericentric passage.
#
# Implementation note:
#   For the ENCLOSED profile, iterate outward: start at R = r_enc[0],
#   use the converged solution as the INITIAL GUESS for R = r_enc[1].
#   This warm-starting reduces iterations by ~factor of 5.
#
# TIME COMPLEXITY:  O(N_SHAPE_SNAPS × N_SHELL_BINS × MAX_ITER × N_per_shell)
#                 ≈ O(60 × 18 × 20 × 100) = 2.2 × 10^6  — fast
# SPACE COMPLEXITY: O(N_SHAPE_SNAPS × N_SHELL_BINS)  per array

# TODO: allocate shape profile arrays
# q_shell_ts    = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# s_shell_ts    = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# T_shell_ts    = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# n_iter_ts     = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# q_enc_ts      = np.full((N_SHAPE_SNAPS, N_GLOBAL_BINS), np.nan)
# s_enc_ts      = np.full((N_SHAPE_SNAPS, N_GLOBAL_BINS), np.nan)
# T_enc_ts      = np.full((N_SHAPE_SNAPS, N_GLOBAL_BINS), np.nan)
# evecs_shell   = np.full((N_SHAPE_SNAPS, N_SHELL_BINS, 3, 3), np.nan)

# TODO: allocate hull metric arrays
# V_hull_ts     = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# A_hull_ts     = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# Psi_ts        = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# r_hull_max_ts = np.full(N_SHAPE_SNAPS, np.nan)  # global max extent
# V_excess_ts   = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# hull_q_pca_ts = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# hull_s_pca_ts = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)

# TODO: allocate orientation arrays
# major_axis_ts   = np.full((N_SHAPE_SNAPS, N_SHELL_BINS, 3), np.nan)
# theta_tilt_ts   = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)
# omega_tumble_ts = np.full((N_SHAPE_SNAPS - 1, N_SHELL_BINS), np.nan)
# phi_M31_ts      = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)

# TODO: allocate summary time series (global, all-particle)
# q_global_ts     = np.full(N_SHAPE_SNAPS, np.nan)
# s_global_ts     = np.full(N_SHAPE_SNAPS, np.nan)
# T_global_ts     = np.full(N_SHAPE_SNAPS, np.nan)
# Psi_global_ts   = np.full(N_SHAPE_SNAPS, np.nan)
# dShape_ts       = np.full((N_SHAPE_SNAPS, N_SHELL_BINS), np.nan)  # Psi − E

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.7 — MAIN COMPUTATION LOOP                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Structure of the loop
# ─────────────────────
# Outer loop over snapshots, inner loop over shells.
# Both the inertia tensor and the convex hull need the SAME shell particles,
# so compute both in the same inner loop to avoid re-reading positions.
#
#   for k, s in enumerate(snap_indices):
#       # Centre of mass correction
#       com    = np.mean(_traj_pos[s], axis=0)
#       pos_k  = _traj_pos[s] - com
#       r_k    = np.linalg.norm(pos_k, axis=1)
#
#       # Global (all-particle) shape
#       result_global = iterative_inertia_tensor(pos_k, r_target=R_OUTER)
#       q_global_ts[k] = result_global['q']
#       ...
#
#       # Shell loop
#       for b in range(N_SHELL_BINS):
#           r_lo  = r_edges[b]
#           r_hi  = r_edges[b + 1]
#           mask  = (r_k >= r_lo) & (r_k < r_hi)
#           if mask.sum() < MIN_ITER_PARTS:
#               continue
#           pos_shell = pos_k[mask]
#
#           # Inertia tensor
#           result = iterative_inertia_tensor(pos_k, r_target=r_hi, r_inner=r_lo)
#           q_shell_ts[k, b]   = result['q']
#           s_shell_ts[k, b]   = result['s']
#           T_shell_ts[k, b]   = result['T']
#           n_iter_ts[k, b]    = result['n_iter']
#           evecs_shell[k,b]   = result['evecs']
#
#           # Convex hull
#           hm = convex_hull_metrics(pos_shell,
#                                    result['a'], result['b'], result['c'])
#           V_hull_ts[k, b]    = hm['V_hull']
#           Psi_ts[k, b]       = hm['Psi']
#           V_excess_ts[k, b]  = hm['V_excess']
#           hull_q_pca_ts[k,b] = hm['hull_q_pca']
#
#       # M31 direction
#       if np.any(_group == 3):
#           m31_pos         = np.mean(pos_k[_group == 3], axis=0)
#           n_M31           = m31_pos / (np.linalg.norm(m31_pos) + 1e-30)
#           for b in range(N_SHELL_BINS):
#               e1 = evecs_shell[k, b, :, 0]
#               if np.all(np.isfinite(e1)):
#                   cos_phi = np.clip(abs(np.dot(e1, n_M31)), 0, 1)
#                   phi_M31_ts[k, b] = float(np.degrees(np.arccos(cos_phi)))
#
#   # Post-loop: orientation tracking per shell
#   dt_arr = np.diff(time_arr[snap_indices]) / 1e3   # [Gyr]
#   for b in range(N_SHELL_BINS):
#       theta_tilt_ts[:, b], omega_tumble_ts[:, b] = \
#           track_orientation(evecs_shell[:, b, :, 0], dt_arr)
#
# Numerical pitfalls
# ──────────────────
# 1. COM drift: at late times, if the merger is not fully virialised, the COM
#    of the sub-sampled particle set may drift.  Use the FULL particle set
#    for COM computation, not just the subset in the current shell.
#
# 2. Convergence failure tracking: count what fraction of shells failed to
#    converge (n_iter == MAX_ITER).  If > 20% fail, the snapshot is mid-merger
#    and shape results should be marked as uncertain.
#
# 3. Particle number per shell: log N_in(k, b) for all shells.  Shells with
#    N < 100 have noisy inertia tensors; shells with N < 20 should be masked.
#
# 4. Eigenvector continuity: at late times, if q → 1 (nearly spherical), the
#    eigenvectors are poorly defined (any axis is "major").  Flag epochs where
#    |a − b| / a < 0.05 and suppress the orientation analysis for those.
#
# TIME COMPLEXITY (full loop):
#   Inertia: O(N_SHAPE_SNAPS × N_SHELL_BINS × MAX_ITER × N_per_shell)
#          ≈ O(60 × 18 × 20 × 100) ≈ 2.2 × 10^6  — seconds
#   Hull:    O(N_SHAPE_SNAPS × N_SHELL_BINS × N_hull × log N_hull)
#          ≈ O(60 × 18 × 200 × 8) ≈ 1.7 × 10^6  — seconds
#   Total: ~ 2–5 minutes on a single CPU core

# TODO: implement main computation loop

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.8 — SHAPE DECOMPOSITION: PROLATE / OBLATE / TRIAXIAL FRACTION        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The triaxiality parameter T classifies each shell at each time:
#   Prolate-dominated:  T > 2/3  (cigar-like)
#   Triaxial:           1/3 < T < 2/3
#   Oblate-dominated:   T < 1/3  (pancake-like)
#
# Track the FRACTION of shells in each category per snapshot:
#   f_prolate_ts(t)  = fraction of valid shells with T > 2/3
#   f_oblate_ts(t)   = fraction of valid shells with T < 1/3
#   f_triaxial_ts(t) = fraction of valid shells with 1/3 < T < 2/3
#
# Physical expectations for a merger remnant:
#   t < t_peri:   oblate (disc-like; MW disc dominates shape at small r)
#   t ≈ t_peri:   prolate spike (radial infall elongates the mass distribution)
#   t > t_peri:   triaxial → slowly approaches oblate as the disc rebuilds
#
# Also track the RADIAL gradient of T:
#   dT/dr(t) = (T_outer − T_inner) / (r_outer − r_inner)
#   dT/dr > 0 → outer halo more prolate (common: outer halo = prolate,
#                                         inner halo = oblate disc)
#   dT/dr < 0 → inner halo more prolate (unusual; indicates deep tidal distortion)
#
# TIME COMPLEXITY:  O(N_SHAPE_SNAPS × N_SHELL_BINS)  — trivial arithmetic
# SPACE COMPLEXITY: O(N_SHAPE_SNAPS)  — three fractions per snapshot

# TODO: compute morphological fractions
# f_prolate_ts  = np.full(N_SHAPE_SNAPS, np.nan)
# f_oblate_ts   = np.full(N_SHAPE_SNAPS, np.nan)
# f_triaxial_ts = np.full(N_SHAPE_SNAPS, np.nan)
# for k in range(N_SHAPE_SNAPS):
#     T_valid = T_shell_ts[k][np.isfinite(T_shell_ts[k])]
#     if len(T_valid) == 0:
#         continue
#     f_prolate_ts[k]  = np.mean(T_valid > 2.0/3.0)
#     f_oblate_ts[k]   = np.mean(T_valid < 1.0/3.0)
#     f_triaxial_ts[k] = np.mean((T_valid >= 1.0/3.0) & (T_valid <= 2.0/3.0))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.9 — FIGURES (TEN PLANNED)                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── Figure 1: Axis ratio profiles q(r) and s(r) at 5 epochs ─────────────────
# Two-panel figure (q top, s bottom) showing the radial profiles at each of
# the five standard epochs (early/pre-peri/peri/post-peri/final).
# x-axis: r_mid [kpc] log scale.  y-axis: axis ratio ∈ [0, 1].
# Overplot error bars as ±1σ from the scatter within each shell.
# HINT: add horizontal lines at q = 1, s = 1 (sphere reference) and
# at q = s (oblate-prolate boundary — exactly satisfied for a prolate spheroid).
# HINT: the five epochs should be colour-coded and labelled with the snapshot
# time in Gyr in the legend.
#
# Expected output: section35_axis_ratio_profiles.png
#
# ── Figure 2: q(r, t) and s(r, t) heatmaps ───────────────────────────────────
# Two heatmaps side by side.  x-axis: time [Gyr], y-axis: log r [kpc].
# Colourmap: sequential (e.g. viridis) with range [0, 1].
# Mark pericentric passages with vertical dashed lines.
# HINT: overplot the q = 0.7 and s = 0.5 contours — these are physically
# meaningful thresholds (q < 0.7 = "significantly non-spherical";
# s < 0.5 = "strongly flattened").
#
# Expected output: section35_shape_heatmaps.png
#
# ── Figure 3: Triaxiality T(r, t) heatmap ────────────────────────────────────
# Heatmap of T_shell_ts.  Diverging colourmap centred on T = 0.5
# (blue = oblate, red = prolate, white = triaxial).
# This is the most diagnostic plot: a red stripe at the time of pericentric
# passage would confirm the standard picture of merger-induced prolateness.
# Overplot the f_prolate_ts time series in a side panel.
#
# Expected output: section35_triaxiality_heatmap.png
#
# ── Figure 4: Convex hull asphericity Ψ(r, t) heatmap ───────────────────────
# Heatmap of Psi_ts (1 − isoperimetric ratio).
# This is the COMPLEMENT to the triaxiality heatmap: compare them side by side.
# High T + high Ψ = prolate bulk + tidal tails (full merger distortion).
# High T + low Ψ = prolate bulk without prominent tidal tails.
# Low T + high Ψ = tidal tails on a spherical/oblate bulk (satellite stripping).
#
# Expected output: section35_hull_asphericity_heatmap.png
#
# ── Figure 5: V_excess(r, t) heatmap ────────────────────────────────────────
# Heatmap of V_hull / V_ellipsoid.  Log colourscale.
# V_excess > 1 means tidal debris outside the best-fit ellipsoid.
# The outer shells should show V_excess > 1 during the merger.
# After virialisation, V_excess → 1 everywhere.
# HINT: use a symmetric log scale (matplotlib's SymLogNorm) centred at 1:
#   values < 1 → bluish (hull smaller than ellipsoid — only possible for
#   very few particles and noisy hull estimates; flag as suspect)
#   values > 1 → reddish (debris outside the ellipsoid)
#
# Expected output: section35_hull_volume_excess.png
#
# ── Figure 6: Global time series — q(t), s(t), T(t), Ψ(t) ──────────────────
# Four-panel figure showing the GLOBAL (all-particle, enclosed in R_OUTER)
# shape evolution.  This is the single most important diagnostic plot.
# Panel 1: q_global_ts(t) and s_global_ts(t) on the same axis.
# Panel 2: T_global_ts(t) — triaxiality.  Mark T = 1/3 and T = 2/3 thresholds.
# Panel 3: Ψ_global_ts(t) — asphericity.
# Panel 4: f_prolate_ts, f_oblate_ts, f_triaxial_ts stacked area plot
#           ("triaxial composition" of the halo over time).
# Vertical lines at pericentric passages on all panels.
#
# Expected output: section35_global_shape_timeseries.png
#
# ── Figure 7: Principal axis orientation and tumbling ────────────────────────
# Top panel: θ_tilt_ts(t, b) for each shell — angle between the major axis
#            and a fixed reference (e.g. initial major axis or z-axis).
#            Use different line styles for inner/mid/outer shells.
# Bottom panel: ω_tumble_ts(t, b) — tumbling rate [deg/Gyr].
#            Mark the expected tumbling rate for solid-body rotation as a
#            horizontal reference line.
# Also include φ_M31_ts for the mid and outer shells — if the major axis
# tracks the M31 direction, this angle stays near zero.
#
# Expected output: section35_orientation_tumbling.png
#
# ── Figure 8: Inertia tensor vs. convex hull PCA comparison ─────────────────
# Scatter plot of (q_shell, s_shell) from the inertia tensor (x-axis)
# vs. (hull_q_pca, hull_s_pca) from the convex hull PCA (y-axis).
# One point per shell per snapshot, coloured by time.
# Perfect agreement → points lie on the diagonal.
# Points ABOVE the diagonal: hull PCA sees a MORE elongated shape (tidal tails).
# Points BELOW the diagonal: hull PCA sees a LESS elongated shape (outliers
# pulling the hull PCA toward sphericity).
# Overplot q=0.5, s=0.5 grid lines for reference.
#
# Expected output: section35_tensor_vs_hull_pca.png
#
# ── Figure 9: ΔShape(r, t) = Ψ − E heatmap (tidal vs. bulk separation) ─────
# ΔShape = Ψ_hull − E_inertia = (1 − I_P) − (1 − s) = s − I_P
# Positive = hull more aspherical than the tensor (tidal tails dominate).
# Negative = tensor more elliptical than hull (bulk elongated, no tails).
# This is the KEY cross-method diagnostic that neither method alone provides.
# HINT: use a strong diverging colourmap (RdBu_r) so zero is clearly white.
#
# Expected output: section35_delta_shape_heatmap.png
#
# ── Figure 10: Master summary panel ──────────────────────────────────────────
# 3×2 grid:
#   (0,0) q(r, t) heatmap with pericentric passage lines
#   (0,1) T(r, t) triaxiality heatmap with morphological boundaries
#   (1,0) Ψ(r, t) hull asphericity heatmap
#   (1,1) ΔShape(r, t) = Ψ − E cross-method heatmap
#   (2,0) Global q(t), s(t), T(t) time series
#   (2,1) Tumbling rate ω(t) for inner/mid/outer shells + M31 alignment φ(t)
#
# Expected output: section35_summary_panel.png

# TODO: implement all ten figures

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.10 — ANIMATION: 3D SHAPE EVOLUTION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Three-panel animation:
#
# Left  : 3D scatter of all tracked particles at the current epoch, projected
#         onto the (x, y) plane, coloured by shell group.
#         Overplot the best-fit ellipsoid as a wireframe (use parametric
#         surface: x = a cosθ sinφ, y = b sinθ sinφ, z = c cosφ, rotated
#         by the eigenvector matrix).
#         Draw the three principal axes as coloured arrows from the origin.
#
# Centre: q(r) and s(r) radial profiles at the current epoch as bar charts
#         (one bar per shell).  Overlay the same profiles from t=0 as a
#         dashed line for comparison.  Animate how the profile changes.
#
# Right : Running time series of q_global(t), s_global(t), T_global(t)
#         with a vertical marker at the current frame time.
#
# HINT: for the ellipsoid wireframe in matplotlib:
#   u = np.linspace(0, 2*np.pi, 40)
#   v = np.linspace(0, np.pi, 20)
#   x_ell = a * np.outer(np.cos(u), np.sin(v))   # (40, 20)
#   y_ell = b * np.outer(np.sin(u), np.sin(v))
#   z_ell = c * np.outer(np.ones(40), np.cos(v))
#   Then rotate: xyz_rot = evecs @ np.array([x_ell.ravel(), y_ell.ravel(), z_ell.ravel()])
#   ax.plot_surface(xyz_rot[0].reshape(40,20), ..., alpha=0.15)
#   Use plot_wireframe for a cleaner look.
#
# HINT: the 3D axes need ax = fig.add_subplot(..., projection='3d').
#   To animate a 3D scatter: scat._offsets3d = (xs, ys, zs)
#   To animate a surface: remove the old surface and redraw
#   (ax.collections.clear() before ax.plot_wireframe(...)).
#
# Expected output: section35_animation_shape.mp4

# TODO: implement animation

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.11 — CROSS-SECTION CORRELATIONS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Compute and print the following correlations to close the §31–35 suite:
#
#   corr(s_global, f_regular)      — sphericity vs. regular orbit fraction (§32)
#                                    HYPOTHESIS: spherical halo → more regular orbits
#   corr(T_global, H_aniso)        — triaxiality vs. entropy anisotropy (§34)
#                                    HYPOTHESIS: prolate (T→1) → H_r > H_t → ΔH > 0
#   corr(Ψ_global, f_stream)       — hull asphericity vs. stream fraction (§33)
#                                    HYPOTHESIS: streams inflate the hull → same timing
#   corr(ω_tumble_mid, dS_mix/dt)  — tumbling rate vs. mixing rate (§33)
#                                    HYPOTHESIS: active tumbling accompanies fast mixing
#   corr(V_excess_outer, KL_global)— tidal volume excess vs. KL divergence (§34)
#                                    HYPOTHESIS: both peak at same epoch (pericentric)
#
# Print as a formatted table with:
#   Quantity 1 | Quantity 2 | Pearson r | p-value | Section(s)
#
# Also print the SHAPE EVOLUTION SUMMARY:
#   Epoch        | q_global | s_global | T_global | Ψ_global | Dominant morphology
#   ─────────────────────────────────────────────────────────────────────────────
#   t = 0 Gyr    | …        | …        | …        | …        | …
#   t = t_peri   | …        | …        | …        | …        | …
#   t = final    | …        | …        | …        | …        | …

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §35.12 — SECTION COMPLETE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Print the output manifest — same pattern as all previous sections.
# Also print a summary statistics table per group:
#
#   Shell    | q_t0 | s_t0 | T_t0 | q_fin | s_fin | T_fin | Ψ_fin | ω_mean
#   ─────────────────────────────────────────────────────────────────────────
#   Inner    | …    | …    | …    | …     | …     | …     | …     | …
#   Mid      | …    | …    | …    | …     | …     | …     | …     | …
#   Outer    | …    | …    | …    | …     | …     | …     | …     | …
#   Global   | …    | …    | …    | …     | …     | …     | …     | …

outputs_35 = [
    "section35_axis_ratio_profiles.png",
    "section35_shape_heatmaps.png",
    "section35_triaxiality_heatmap.png",
    "section35_hull_asphericity_heatmap.png",
    "section35_hull_volume_excess.png",
    "section35_global_shape_timeseries.png",
    "section35_orientation_tumbling.png",
    "section35_tensor_vs_hull_pca.png",
    "section35_delta_shape_heatmap.png",
    "section35_animation_shape.mp4",
    "section35_summary_panel.png",
]

# TODO: implement output manifest printing
