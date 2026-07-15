===============================================================================
SECTION 36 (ALTERNATIVE STRATEGY) — HALO MORPHOLOGY VIA
  METHOD A : Orientation-Tensor Fabric Analysis + Density-Weighted
             Spherical Harmonic Multipoles with Adaptive KDE Bandwidth
  METHOD B : sklearn KDE Density Field + Persistent-Homology-Inspired
             Sub-Level-Set Topology Scan for Minkowski Functionals
===============================================================================
Author  : Abhinav Vatsa

WHY A DIFFERENT STRATEGY?
──────────────────────────
Section 36 (original) computed halo morphology using:
  Method A  — Particle-count spherical harmonic coefficients, equal weight per particle
  Method B  — CIC density grid with fixed Gaussian smoothing + fixed threshold scan

Both are standard approaches but carry specific biases:
  • Equal-weight a_lm: a single dense tidal stream with 50 particles in a
    shell of 500 contributes the same per-particle power as the diffuse
    background, even though it contains only ~10% of the mass.
    This inflates dipole and quadrupole power during stream-crossing epochs.
  • Fixed CIC smoothing: the same σ_smooth = 2.5 voxels is applied
    everywhere, regardless of the local particle density.  The inner halo
    (N ~ 400/shell) is over-smoothed; the outer halo (N ~ 10/shell) is
    under-smoothed.  Both lead to biased Minkowski functionals.

This file replaces those choices with physically motivated alternatives:

  Method A (NEW):
    Step 1 — Orientation Tensor Fabric Analysis (Woodcock 1977; Watson 1983):
      Fit the 3×3 scatter matrix T_ij = (1/N) Σ r̂_i r̂_j of unit vectors on
      the unit sphere.  Its eigenvalues (t₁ ≤ t₂ ≤ t₃, sum = 1) quantify
      the angular distribution shape WITHOUT assuming any particular
      spherical harmonic basis:
        Woodcock C = ln(t₃/t₁) — fabric strength (0 = isotropic, ∞ = perfect axis)
        Woodcock K = ln(t₃/t₂) / ln(t₂/t₁) — shape:
                     K >> 1 → prolate cluster (particles crowd a pole)
                     K << 1 → oblate girdle (particles avoid the axis)
                     K = 1  → triaxial intermediate
      This gives a single (C, K) pair per shell per snapshot — extremely
      compact, noise-resistant, and interpretable.

    Step 2 — Density-Weighted Multipoles (Gerhard 1983; de Zeeuw & Franx 1991):
      Replace the equal-weight sum Σ Y_lm*(θ_k, φ_k) / N with a
      DENSITY-WEIGHTED sum Σ w_k Y_lm*(θ_k, φ_k) where w_k = ρ̂_k / Σρ̂_k
      and ρ̂_k is the KNN-estimated local 3D density at particle k.
      This naturally down-weights sparse tidal debris (low ρ̂) and
      up-weights the dense gravitational core.
      The KNN bandwidth is ADAPTIVE — automatically larger in sparse regions
      and smaller in dense regions — eliminating the over/under-smoothing bias.

  Method B (NEW):
    Step 1 — sklearn KernelDensity field with Epanechnikov kernel:
      Instead of CIC mass assignment, use sklearn's KernelDensity with a
      compact-support Epanechnikov kernel.  The bandwidth is chosen by
      Scott's rule: h = σ_v N^{-1/(d+4)} with d=3.
      This produces a proper density estimator (guaranteed non-negative,
      integrates to N, continuous) with computable bias/variance tradeoff.
      The Epanechnikov kernel is optimal in the MSE sense among compactly
      supported kernels — it gives the lowest bias for a given bandwidth.

    Step 2 — Persistent-Homology-Inspired Topology Scan:
      Instead of simply scanning thresholds and computing MFs at each one,
      track the BIRTH AND DEATH of connected components as the density
      threshold descends from the maximum to zero (a sub-level set filtration).
      Each pair (birth threshold ρ_b, death threshold ρ_d) defines a
      topological feature with PERSISTENCE = ρ_b − ρ_d.
      Long-lived features (high persistence) are real structures (the halo
      core, M31 nucleus).  Short-lived features (low persistence) are noise.
      This is a simplified version of persistent homology — the full theory
      requires the Vietoris-Rips complex, which is computationally expensive.
      Our proxy computes the PERSISTENCE DIAGRAM for H₀ (connected components)
      directly from the merge tree of the density field, giving:
        • n_births(t)     — number of distinct density peaks born during scan
        • persistence_max(t) — persistence of the longest-lived secondary peak
                              (the M31 nucleus during the merger)
        • ρ_merge(t)      — threshold at which the two dominant peaks merge
                              (direct measure of the pericentric epoch)

Cross-method comparison with §36 original:
  The (C, K) fabric parameters map onto (T, q, s) from §35 as follows:
    C → ellipticity E = 1 − s  (both measure total non-sphericity)
    K → triaxiality T (both discriminate oblate from prolate)
  The persistent homology birth-death pairs map onto the χ(ρ_th) curve from
  §36 original, but the persistence filter separates real peaks from noise.

Key quantities computed in this section
────────────────────────────────────────
  METHOD A — FABRIC & WEIGHTED MULTIPOLES:
    T_ij(r, t)        — orientation scatter matrix per shell [3×3]
    t₁(r,t),t₂,t₃    — eigenvalues of T_ij (t₃ ≥ t₂ ≥ t₁, sum=1)
    C_wood(r, t)      — Woodcock strength C = ln(t₃/t₁) ∈ [0, ∞)
    K_wood(r, t)      — Woodcock shape   K = ln(t₃/t₂)/ln(t₂/t₁) ∈ (0, ∞)
    e_maj(r, t)       — major fabric axis (eigenvector of t₃) [unit vector]
    φ_fab(r, t)       — fabric orientation angle on sky [deg]
    w_k(t, i)         — KNN density weight per particle
    Ẽ_l_w(r, t)       — density-weighted normalised multipole power
    Q_bar_w(r, t)     — density-weighted quadrupole Ẽ₂ (dominant shape mode)
    ΔQ_w_vs_eq(r,t)   — Ẽ₂ difference: weighted minus equal-weight (bias check)

  METHOD B — KDE DENSITY FIELD & PERSISTENCE TOPOLOGY:
    ρ_kde(x,t)        — KDE density field on 3D grid [particles/kpc³]
    h_scott(t)        — Scott's bandwidth [kpc]
    W0_kde(ρ_th,t)    — MF volume from KDE field [kpc³]
    W1_kde, W2_kde,W3 — full MF set from KDE field
    P_kde(ρ_th,t)     — planarity from KDE field ∈ [0,1]
    F_kde(ρ_th,t)     — filamentarity from KDE field ∈ [0,1]
    n_births(t)        — number of density peaks born in sub-level scan
    persistence_pairs  — list of (ρ_birth, ρ_death) per component
    persistence_max(t) — persistence of the longest-lived non-dominant peak
    ρ_merge(t)        — threshold at which the two dominant peaks merge [kpc⁻³]
    f_noise_comps(t)  — fraction of components with persistence < PERSIST_TOL

Dependencies
────────────
  scipy.special      — sph_harm_y
  scipy.linalg       — eigh  (symmetric eigendecomposition)
  scipy.ndimage      — gaussian_filter, label
  scipy.stats        — pearsonr
  scipy.interpolate  — interp1d  (time-resampling for cross-correlations)
  sklearn.neighbors  — KernelDensity, KDTree / KNN density weights
  skimage.measure    — euler_number (Euler characteristic)
  Section 26         — traj_pos, traj_r (Lagrangian trajectories)
  Section 35         — T_shell_ts, q_shell_ts, s_shell_ts (for cross-check)
  Section 36 orig.   — Q_bar_arr (equal-weight) for ΔQ bias comparison

All globals from the parent pipeline are inherited.
===============================================================================
"""

# ── standard library ──────────────────────────────────────────────────────────
import os
import time
import warnings

# ── numerics ──────────────────────────────────────────────────────────────────
import numpy as np
from scipy.linalg     import eigh as sp_eigh          # guaranteed real evals
from scipy.special    import sph_harm_y               # scipy ≥ 1.15 API
from scipy.ndimage    import gaussian_filter, label as ndimage_label
from scipy.stats      import pearsonr
from scipy.interpolate import interp1d as _interp1d

# sklearn for adaptive KDE (density weights + KDE density field)
from sklearn.neighbors import KernelDensity, KDTree

# skimage for Euler characteristic
try:
    from skimage.measure import euler_number, label as sk_label
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False
    warnings.warn("scikit-image not found — χ (Euler) uses connected-component proxy.")

# ── plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §A.0  CONFIGURATION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Fabric / weighted multipole parameters ────────────────────────────────────
L_MAX              = 8      # maximum multipole degree for weighted expansion
N_FABRIC_SHELLS    = 40     # radial shells for orientation tensor
N_FABRIC_SNAPS     = 60     # snapshot epochs for fabric analysis
MIN_FABRIC_PART    = 15     # minimum particles per shell for valid T_ij
K_KNN_WEIGHT       = 20     # k for the KNN density weight estimator
                             # HINT: larger K → smoother weights, less adaptive
                             # smaller K → sharper weights, noisier in sparse regions

# ── KDE density field parameters ─────────────────────────────────────────────
GRID_RES_KDE       = 48     # voxels per side of the KDE evaluation grid
                             # HINT: KDE grid can be coarser than CIC because
                             # the KDE is already smooth — no need for
                             # post-smoothing.  48³ ≈ 110 000 evaluations.
N_THRESH_KDE       = 30     # threshold levels for the KDE MF scan
N_KDE_SNAPS        = 25     # snapshot epochs for KDE Minkowski analysis
SCOTT_FACTOR       = 1.0    # multiplier on Scott's bandwidth h = factor × σ N^{-1/5}
                             # HINT: increase to 1.5–2.0 for very sparse outer halos;
                             # decrease to 0.6–0.8 to resolve sharp density contrasts.
KDE_KERNEL         = 'epanechnikov'   # optimal MSE kernel (compact support)
                                       # alternatives: 'gaussian', 'tophat'

# ── Persistence topology parameters ──────────────────────────────────────────
PERSIST_TOL        = 0.05   # features with persistence < PERSIST_TOL × (ρ_max − ρ_min)
                             # are classified as NOISE and excluded from the
                             # persistence diagram.
                             # HINT: set to 0.02 if you want to capture very
                             # faint satellite subhaloes; set to 0.15 to be
                             # conservative and only track the main halo + M31.
N_PERSIST_THRESH   = 80     # number of threshold levels for the sub-level scan
                             # MORE thresholds → finer birth/death resolution
                             # but SLOWER.  80 is a good balance.

# ── Geometry ──────────────────────────────────────────────────────────────────
R_INNER            = 1.0    # [kpc] — overridden dynamically from data
R_OUTER            = 120.0  # [kpc] — overridden dynamically from data

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ANIM_FPS   = 18
ANIM_DPI   = 100
ANIM_KBPS  = 1600

print("\n" + "="*80)
print("  SECTION 36 (ALT) · Fabric Tensor + Adaptive KDE Morphology")
print("="*80)
print(f"  L_MAX              : {L_MAX}")
print(f"  Fabric shells      : {N_FABRIC_SHELLS}   snaps: {N_FABRIC_SNAPS}")
print(f"  KNN weight k       : {K_KNN_WEIGHT}")
print(f"  KDE grid           : {GRID_RES_KDE}³   kernel: {KDE_KERNEL}")
print(f"  KDE Scott factor   : {SCOTT_FACTOR}")
print(f"  KDE snaps          : {N_KDE_SNAPS}   thresholds: {N_THRESH_KDE}")
print(f"  Persist. thresholds: {N_PERSIST_THRESH}   noise tol: {PERSIST_TOL}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §A.1  SYNTHETIC DATA  (replace with §26 inherited globals)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# In the full pipeline, inherit _traj_pos, _traj_r, _group, time_arr, ns
# from the Section 26 globals.  The synthetic data below reproduce a
# plausible MW–M31 merger trajectory for standalone validation.

np.random.seed(99)
_NS   = 80          # total snapshots
_N    = 1200        # total particles
_DT   = 10.0        # [Myr] per snapshot

_traj_pos = np.zeros((_NS, _N, 3))
_r0_base  = np.abs(np.random.randn(_N)) * 18 + 3
theta0    = np.random.uniform(0, np.pi,   _N)
phi0      = np.random.uniform(0, 2*np.pi, _N)

for s in range(_NS):
    tf = s / (_NS - 1)
    # Synthetic shape: oblate at t=0 → strongly prolate at pericenter → triaxial
    ax = 1.0 + 1.4 * np.exp(-0.5 * ((tf - 0.45) / 0.12)**2)
    ay = 1.0 + 0.4 * np.exp(-0.5 * ((tf - 0.45) / 0.18)**2)
    az = max(1.0 - 0.45 * tf, 0.50)
    r_ev = _r0_base * (1.0 + 0.18 * np.sin(2 * np.pi * tf))
    x = r_ev * np.sin(theta0) * np.cos(phi0) * ax
    y = r_ev * np.sin(theta0) * np.sin(phi0) * ay
    z = r_ev * np.cos(theta0)               * az
    _traj_pos[s] = np.stack([x, y, z], axis=1) + np.random.randn(_N, 3) * 0.4

_traj_r = np.linalg.norm(_traj_pos, axis=2)   # (ns, N)
_r0     = _traj_r[0]
_group  = np.zeros(_N, dtype=int)
_group[3 * _N // 4:] = 3                       # last quarter = M31 particles
ns      = _NS

time_arr = np.arange(ns) * _DT / 1000.0        # [Gyr]

R_INNER = max(0.5, float(np.percentile(_r0, 2)))
R_OUTER = float(np.percentile(_r0, 97))
print(f"\n  R_INNER = {R_INNER:.1f} kpc    R_OUTER = {R_OUTER:.1f} kpc")

# Snapshot index arrays
snap_idx_fab = np.linspace(0, ns - 1, N_FABRIC_SNAPS,  dtype=int)
snap_idx_kde = np.linspace(0, ns - 1, N_KDE_SNAPS,     dtype=int)
time_fab     = time_arr[snap_idx_fab]
time_kde     = time_arr[snap_idx_kde]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §A.2  ORIENTATION TENSOR (FABRIC ANALYSIS)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Theory
# ──────
# For N particles with unit-vector directions r̂_k ∈ S² (their positions
# normalised to the unit sphere), define the 3×3 orientation tensor:
#
#   T_ij = (1/N) Σ_k  r̂_k,i  r̂_k,j       i,j ∈ {x,y,z}
#
# This is the SECOND MOMENT of the angular distribution.  It is equivalent
# to the reduced inertia tensor of §35 when r̂_k = r_k / |r_k|, but
# applied to UNIT VECTORS rather than physical positions, so it measures
# ANGULAR SHAPE independent of the radial distribution.
#
# Properties of T_ij:
#   • Symmetric 3×3 matrix → 3 real eigenvalues t₁ ≤ t₂ ≤ t₃
#   • Trace = t₁ + t₂ + t₃ = 1  (because Σ r̂² = N)
#   • For an isotropic distribution: t₁ = t₂ = t₃ = 1/3
#   • For a perfectly prolate cluster (all particles on one axis):
#       t₁ = t₂ = 0,  t₃ = 1
#   • For a perfectly oblate girdle (particles on equatorial plane):
#       t₁ = 0,  t₂ = t₃ = 1/2
#
# Woodcock (1977) shape parameters:
#   C = ln(t₃ / t₁)      — STRENGTH of the fabric anisotropy
#                            C = 0 → isotropic;  C → ∞ → perfect single axis
#
#   K = ln(t₃ / t₂) / ln(t₂ / t₁)   — SHAPE of the fabric
#                            K = 1    → triaxial (C line on Flinn plot)
#                            K > 1    → prolate cluster (particles near one pole)
#                            K < 1    → oblate girdle (particles avoid axis)
#                            K = ∞    → t₂ = t₁ (perfectly linear / rod-like)
#                            K = 0    → t₃ = t₂ (perfectly planar / disc-like)
#
# Numerical note:
#   When t₂ ≈ t₁ (nearly oblate) or t₃ ≈ t₂ (nearly prolate), the
#   denominator of K is near zero.  Clip both log arguments to 1e-6
#   before dividing to prevent inf/nan.  Return K = np.nan when the
#   shell is too nearly isotropic (C < 0.05) to define K meaningfully.

def orientation_tensor(pos_shell):
    """
    Compute the 3×3 orientation scatter matrix T_ij = (1/N) Σ r̂_i r̂_j
    and its Woodcock fabric parameters for a set of particle positions.

    Parameters
    ----------
    pos_shell : (N, 3)  — particle positions in the shell, COM-subtracted [kpc]
                          Need NOT be on the unit sphere — r̂ is computed here.

    Returns
    -------
    t_evals  : (3,)     — eigenvalues t₁ ≤ t₂ ≤ t₃  (ascending, sum = 1)
    evecs    : (3, 3)   — eigenvectors as COLUMNS (ascending order matching evals)
                          evecs[:, 2] = major fabric axis
    C_wood   : float    — Woodcock strength  ln(t₃/t₁) ∈ [0, ∞)
    K_wood   : float    — Woodcock shape    ln(t₃/t₂)/ln(t₂/t₁) ∈ (0, ∞)
    phi_fab  : float    — azimuthal angle of major axis on sky [deg]
    valid    : bool     — False if N < MIN_FABRIC_PART or degenerate

    IMPLEMENTATION NOTES:
    • Use sp_eigh (scipy symmetric eigensolver) — guarantees ascending real
      eigenvalues and orthonormal eigenvectors.  numpy.linalg.eig does NOT
      guarantee real outputs for symmetric matrices with numerical noise.
    • After computing r̂, check that all radii are > 0 (exclude the origin).
    • The minor axis eigenvector evecs[:,0] (for t₁) points in the direction
      that is LEAST populated — opposite to the major axis.
      For a prolate halo: evecs[:,2] points along the long axis.
      For an oblate halo: evecs[:,0] points along the compression (polar) axis.
    """
    N = len(pos_shell)
    if N < MIN_FABRIC_PART:
        return (np.full(3, np.nan), np.full((3,3), np.nan),
                np.nan, np.nan, np.nan, False)

    r_k   = np.linalg.norm(pos_shell, axis=1)
    valid  = r_k > 1e-10
    if valid.sum() < MIN_FABRIC_PART:
        return (np.full(3, np.nan), np.full((3,3), np.nan),
                np.nan, np.nan, np.nan, False)

    r_hat = pos_shell[valid] / r_k[valid, None]    # (M, 3) unit vectors

    # Orientation tensor: T = (1/M) r̂ᵀ r̂  — efficient via einsum
    T      = np.einsum('ki,kj->ij', r_hat, r_hat) / len(r_hat)   # (3,3)
    t_vals, evecs = sp_eigh(T)   # ascending order guaranteed

    # Woodcock parameters
    # Clip to avoid log(0) or division by zero at near-isotropic configurations
    t1, t2, t3 = float(t_vals[0]), float(t_vals[1]), float(t_vals[2])
    t1c = max(t1, 1e-9)
    t2c = max(t2, 1e-9)
    t3c = max(t3, 1e-9)

    C_wood = float(np.log(t3c / t1c))          # fabric strength

    ln_t3_t2 = np.log(t3c / t2c)
    ln_t2_t1 = np.log(t2c / t1c)
    if C_wood < 0.05 or ln_t2_t1 < 1e-9:
        # shell is too isotropic to define K
        K_wood = np.nan
    else:
        K_wood = float(ln_t3_t2 / ln_t2_t1)

    # Major axis = evecs[:, 2]  (eigenvector of largest eigenvalue)
    e_maj  = evecs[:, 2]
    phi_fab = float(np.degrees(np.arctan2(e_maj[1], e_maj[0])))

    return t_vals, evecs, C_wood, K_wood, phi_fab, True

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §A.3  ADAPTIVE KNN DENSITY WEIGHTS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Theory
# ──────
# The density weight w_k for particle k is proportional to its local
# 3D KNN density estimate:
#
#   ρ̂_k  =  (K − 1) / [ N × V₃(d_K) ]
#
# where V₃(r) = (4π/3) r³ is the volume of a 3D sphere of radius r,
# and d_K = distance to the K-th nearest neighbour of particle k.
#
# The normalised weight used in the multipole expansion is:
#   w_k  =  ρ̂_k / Σ_j ρ̂_j       (sums to 1 across all particles in the shell)
#
# Physical motivation:
#   • A particle in the dense halo core has many neighbours → small d_K → large ρ̂
#   • A particle in a sparse tidal stream has few neighbours → large d_K → small ρ̂
#   • The density-weighted a_lm therefore reflects the MASS-BEARING structure,
#     not the number of tracer particles, which is more physically meaningful.
#
# When to use density weights:
#   • Use them when you want to know the shape of the MASS DISTRIBUTION.
#   • Use equal weights when you want to know the shape of the ORBIT DISTRIBUTION
#     (each orbit contributes equally regardless of how much mass it carries).
#   • For a DM halo in equilibrium, mass = orbit weight, so the two should agree.
#     DISCREPANCIES signal that the simulation is not phase-mixed — the density
#     is being dominated by a recent accretion event (clumped particles).
#
# Time complexity:  O(N log N)  for the KDTree build + O(N K log N) for queries
# Space complexity: O(N)  — just one weight per particle

def knn_density_weights(pos_shell, k=K_KNN_WEIGHT):
    """
    Compute normalised KNN density weights for each particle in a shell.

    Parameters
    ----------
    pos_shell : (N, 3)  — particle positions in shell [kpc]
    k         : int     — number of nearest neighbours for density estimate

    Returns
    -------
    w_k   : (N,)  — normalised density weights, summing to 1
    rho_k : (N,)  — raw KNN density estimate [kpc⁻³]

    PITFALL: if multiple particles are at the same position (can happen in
    ICs with a regular grid), the k-th neighbour distance is zero → ρ̂ = ∞.
    Guard with d_K = max(d_K, 1e-10) AFTER the KDTree query.

    PITFALL: for shells with N < k, the query is ill-defined.
    Return uniform weights (1/N) in that case — the shell is too sparsely
    sampled for the density weighting to add information.
    """
    N = len(pos_shell)
    if N <= k:
        # too few particles: fall back to uniform equal weights
        return np.full(N, 1.0 / N), np.full(N, np.nan)

    # sklearn KDTree is faster than cKDTree for dense feature matrices
    tree     = KDTree(pos_shell)
    # query k+1 because the first neighbour is the particle itself (distance=0)
    dists, _ = tree.query(pos_shell, k=k + 1)
    d_K      = np.maximum(dists[:, k], 1e-10)   # K-th neighbour distance

    # 3D sphere volume prefactor: V₃(r) = (4π/3) r³
    V3_prefactor = 4.0 * np.pi / 3.0
    rho_k        = float(k - 1) / (N * V3_prefactor * d_K**3)

    # Normalise to sum = 1
    rho_sum = rho_k.sum()
    w_k     = rho_k / (rho_sum + 1e-30)
    return w_k, rho_k

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §A.4  DENSITY-WEIGHTED MULTIPOLE EXPANSION                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Theory
# ──────
# The density-weighted a_lm estimator replaces the equal-weight mean with
# a weighted sum:
#
#   a_lm^(w)(r_shell)  =  Σ_k  w_k  Y_lm*(θ_k, φ_k)
#
# where w_k is the normalised KNN density weight from §A.3.
#
# The normalisation ensures:
#   Σ_k w_k = 1  →  a_00^(w) = Y_00 = 1/√(4π) ≈ 0.2821
#   The monopole power P_0^(w) = |a_00^(w)|² = 1/(4π)  — same as equal-weight.
#
# Comparison with equal-weight estimator:
#   • When the density is uniform (ρ̂_k = const): w_k = 1/N → equal-weight.
#   • When there is a dense stream: the stream particles have large w_k,
#     so their angular direction contributes more to the expansion.
#     If the stream is aligned with a particular (l,m) mode, Ẽ_l^(w) > Ẽ_l^(eq).
#   • The difference  ΔQ = Ẽ₂^(w) − Ẽ₂^(eq)  is a BIAS DIAGNOSTIC:
#     ΔQ > 0 → the dominant angular structure is also the denser structure (stream)
#     ΔQ < 0 → the dominant angular structure is actually a sparse orbital family
#
# Time complexity:  O(N_shell × L_MAX²)  per shell — same as equal-weight.
# The extra cost is only the KNN query: O(N_shell × k × log N_shell).

def compute_alm_weighted(theta_sh, phi_sh, w_k, l_max=L_MAX):
    """
    Compute density-weighted spherical harmonic coefficients a_lm^(w)
    and normalised multipole power Ẽ_l for a shell of particles.

    Parameters
    ----------
    theta_sh : (N,)  — colatitude [rad] ∈ [0, π]
    phi_sh   : (N,)  — azimuth   [rad] ∈ [−π, π]
    w_k      : (N,)  — density weights (sum = 1)
    l_max    : int   — maximum multipole degree

    Returns
    -------
    alm_w    : dict {(l,m): complex a_lm^(w)}
    P_l_w    : (l_max+1,) raw power Σ_m |a_lm^(w)|²
    Etilde_w : (l_max+1,) normalised power P_l / P_0

    SCIPY CONVENTION REMINDER:
      sph_harm_y(l, m, theta_colatitude, phi_azimuth)
      — colatitude FIRST, azimuth SECOND  (scipy ≥ 1.15)
      — for m < 0, scipy handles the Condon-Shortley phase correctly.

    SYMMETRY SHORTCUT:
      |a_l{-m}^(w)|² = |a_lm^(w)|²  for real density fields.
      Exploit this to halve the computation: compute m ≥ 0 only,
      then P_l += 2 × Σ_{m=1}^{l} |a_lm|².
    """
    N = len(theta_sh)
    if N < 2:
        nan_p = np.full(l_max + 1, np.nan)
        return {}, nan_p, nan_p

    alm_w = {}
    P_l_w = np.zeros(l_max + 1, dtype=float)

    for l in range(l_max + 1):
        # m = 0
        Y0              = sph_harm_y(l, 0, theta_sh, phi_sh)
        a0              = float(np.sum(w_k * np.conj(Y0)))   # complex part ~ 0 for real w
        # NOTE: a0 is real only if w_k and Y_{l0} are real, which Y_{l0} is.
        # For m=0, Y_l0 is real → a_l0 is real.  Keep as complex for generality.
        a0_c            = np.sum(w_k * np.conj(Y0).astype(complex))
        alm_w[(l, 0)]  = a0_c
        P_l_w[l]       += float(abs(a0_c)**2)

        for m in range(1, l + 1):
            Ym             = sph_harm_y(l, m, theta_sh, phi_sh)
            alm_c          = np.sum(w_k * np.conj(Ym))
            alm_w[(l,  m)] = alm_c
            alm_w[(l, -m)] = ((-1) ** m) * np.conj(alm_c)
            P_l_w[l]      += 2.0 * float(abs(alm_c)**2)

    P0        = P_l_w[0] if P_l_w[0] > 1e-30 else 1.0
    Etilde_w  = P_l_w / P0
    return alm_w, P_l_w, Etilde_w

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §A.5  PRECOMPUTE SPHERICAL COORDINATES AND KNN WEIGHTS                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

r_edges_fab = np.logspace(np.log10(R_INNER), np.log10(R_OUTER), N_FABRIC_SHELLS + 1)
r_mid_fab   = np.sqrt(r_edges_fab[:-1] * r_edges_fab[1:])   # geometric midpoints

# Precompute spherical angles at every multipole epoch — avoids redundant
# arccos/arctan2 calls inside the nested loops.
print("\n  Precomputing spherical coordinates …", end="", flush=True)
theta_fab = np.full((N_FABRIC_SNAPS, _N), np.nan)
phi_fab_g = np.full((N_FABRIC_SNAPS, _N), np.nan)
com_fab   = np.zeros((N_FABRIC_SNAPS, 3))

for k, s in enumerate(snap_idx_fab):
    com          = _traj_pos[s].mean(axis=0)
    com_fab[k]   = com
    pos_c        = _traj_pos[s] - com
    r_c          = np.linalg.norm(pos_c, axis=1)
    r_safe       = np.maximum(r_c, 1e-10)
    theta_fab[k] = np.arccos(np.clip(pos_c[:, 2] / r_safe, -1.0, 1.0))
    phi_fab_g[k] = np.arctan2(pos_c[:, 1], pos_c[:, 0])
print(" done.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §A.6  MAIN FABRIC + WEIGHTED MULTIPOLE LOOP                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Pre-allocate all output arrays.
# NaN fill is deliberate: distinguishes "not computed" from "computed zero".
C_wood_arr    = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)
K_wood_arr    = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)
t3_arr        = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)   # major eigenvalue
t1_arr        = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)   # minor eigenvalue
phi_fab_arr   = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)   # major axis PA [deg]
evec_maj_arr  = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS, 3), np.nan)

# Density-weighted multipole arrays
Etilde_w_arr  = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS, L_MAX + 1), np.nan)
Q_bar_w_arr   = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)   # weighted Ẽ₂
D_bar_w_arr   = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)   # weighted Ẽ₁

# Bias diagnostic: weighted minus equal-weight
delta_Q_arr   = np.full((N_FABRIC_SNAPS, N_FABRIC_SHELLS), np.nan)   # ΔQ = Ẽ₂^w − Ẽ₂^eq

# Global (all-particle) fabric parameters
C_global_ts   = np.full(N_FABRIC_SNAPS, np.nan)
K_global_ts   = np.full(N_FABRIC_SNAPS, np.nan)
Q_global_w_ts = np.full(N_FABRIC_SNAPS, np.nan)   # global density-weighted Ẽ₂

print("  Computing fabric tensor + weighted multipoles …")
t0_fab = time.time()

for k, s in enumerate(snap_idx_fab):
    r_now    = _traj_r[s]
    pos_now  = _traj_pos[s] - com_fab[k]         # COM-subtracted positions

    # ── Global fabric (all particles) ────────────────────────────────────────
    tv_g, ev_g, Cg, Kg, _, valid_g = orientation_tensor(pos_now)
    C_global_ts[k] = Cg
    K_global_ts[k] = Kg

    # Global weighted multipole
    w_g, _  = knn_density_weights(pos_now, k=K_KNN_WEIGHT)
    th_g    = theta_fab[k]
    ph_g    = phi_fab_g[k]
    _, _, Et_g     = compute_alm_weighted(th_g, ph_g, w_g)
    Q_global_w_ts[k] = Et_g[2] if len(Et_g) > 2 else np.nan

    # ── Per-shell loop ────────────────────────────────────────────────────────
    for b in range(N_FABRIC_SHELLS):
        mask = (r_now >= r_edges_fab[b]) & (r_now < r_edges_fab[b + 1])
        if mask.sum() < MIN_FABRIC_PART:
            continue

        pos_sh = pos_now[mask]
        th_sh  = theta_fab[k][mask]
        ph_sh  = phi_fab_g[k][mask]

        # Orientation tensor
        t_vals, evecs, C_w, K_w, phi_f, valid = orientation_tensor(pos_sh)
        if valid:
            C_wood_arr[k, b]    = C_w
            K_wood_arr[k, b]    = K_w
            t3_arr[k, b]        = float(t_vals[2])
            t1_arr[k, b]        = float(t_vals[0])
            phi_fab_arr[k, b]   = phi_f
            evec_maj_arr[k, b]  = evecs[:, 2]

        # Density-weighted multipoles
        w_sh, rho_sh = knn_density_weights(pos_sh, k=min(K_KNN_WEIGHT, mask.sum()//2))
        _, _, Et_w   = compute_alm_weighted(th_sh, ph_sh, w_sh)
        Etilde_w_arr[k, b] = Et_w
        Q_bar_w_arr[k, b]  = Et_w[2] if len(Et_w) > 2 else np.nan
        D_bar_w_arr[k, b]  = Et_w[1] if len(Et_w) > 1 else np.nan

        # Equal-weight Ẽ₂ for bias comparison (no separate call needed —
        # equal-weight is the special case w_k = 1/N)
        w_eq         = np.full(mask.sum(), 1.0 / mask.sum())
        _, _, Et_eq  = compute_alm_weighted(th_sh, ph_sh, w_eq)
        E2_eq        = Et_eq[2] if len(Et_eq) > 2 else np.nan
        delta_Q_arr[k, b] = Q_bar_w_arr[k, b] - E2_eq

    if (k + 1) % 12 == 0:
        print(f"    snap {k+1}/{N_FABRIC_SNAPS}  "
              f"C_global={C_global_ts[k]:.3f}  K_global={K_global_ts[k]:.3f}", flush=True)

print(f"  Fabric + weighted multipoles done in {time.time()-t0_fab:.1f} s")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §B.1  KDE DENSITY FIELD WITH SCOTT'S BANDWIDTH                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Theory
# ──────
# The kernel density estimate at a grid point x is:
#
#   ρ̂_KDE(x)  =  (1/N) Σ_k  (1/h³)  K( |x − x_k| / h )
#
# where K is the kernel function and h is the bandwidth.
#
# Scott's bandwidth rule (optimal for a normal reference distribution):
#   h_Scott = σ_data × N^{−1/(d+4)}   for d = 3 dimensions
#
# The EPANECHNIKOV kernel is:
#   K(u) = (3/4)(1 − u²)  for |u| ≤ 1,  else 0
#
# It is the OPTIMAL kernel in the mean-integrated-squared-error sense:
# no other compactly supported kernel achieves lower bias × variance for
# the same bandwidth.  It is also FAST because it has compact support —
# particles more than h_Scott away from a grid point contribute zero.
#
# Advantages of KDE over CIC:
#   1. UNBIASED at the boundary: KDE does not wrap or alias particles at
#      the grid edge (CIC does unless periodic boundary conditions are used).
#   2. CONTINUOUS and differentiable: the density field is smooth by
#      construction, unlike CIC which produces step functions.
#   3. NO post-smoothing needed: the kernel does the smoothing.
#      CIC requires a separate Gaussian filter step, adding one free parameter.
#   4. AUTOMATIC scale selection: Scott's rule adapts to the data.
#      CIC uses a fixed SMOOTH_SIGMA that is a tuning parameter.
#
# Disadvantages:
#   1. SLOWER: O(N × G³ / h³) for evaluation (though sklearn uses tree
#      acceleration to reduce this to O(G³ × log N)).
#   2. ISOTROPIC smoothing: the kernel has the same bandwidth in all directions.
#      CIC can be made anisotropic more easily.

def build_kde_density_grid(pos_com, grid_res=GRID_RES_KDE, r_outer=None,
                            scott_factor=SCOTT_FACTOR):
    """
    Build a 3D density grid using sklearn's KernelDensity with Scott's bandwidth.

    Parameters
    ----------
    pos_com      : (N, 3)  — COM-subtracted particle positions [kpc]
    grid_res     : int     — number of voxels per side
    r_outer      : float   — half-size of the evaluation domain [kpc]
    scott_factor : float   — multiplier on Scott's rule bandwidth

    Returns
    -------
    rho_kde    : (G, G, G)  — density field [particles kpc⁻³], NOT overdensity.
                              NOTE: KDE gives density in physical units.
                              Convert to overdensity for MF if desired:
                              delta = rho_kde / rho_kde.mean() - 1
    voxel_size : float      — [kpc / voxel]
    h_scott    : float      — bandwidth used [kpc]

    DESIGN CHOICE:
      We work in PHYSICAL DENSITY (not overdensity) for the KDE field,
      unlike the CIC approach in §36 original (which used overdensity δ).
      Physical density is more natural for the excursion set interpretation:
      ρ > ρ_threshold means "denser than N_eff particles per kpc³".
      The threshold scan (§B.2) uses percentiles of the KDE field to ensure
      scale-invariance regardless of the total particle count.

    PITFALL: sklearn's score_samples returns LOG density (for numerical
    stability with very small values).  Always exponentiate the result
    before storing or comparing to thresholds.

    PITFALL: the KDE is fit to all particles including those outside r_outer.
    The EVALUATION grid is restricted to r < r_outer, but the fit uses all
    particles — this is correct and prevents boundary artefacts.
    """
    if r_outer is None:
        r_outer = float(np.max(np.linalg.norm(pos_com, axis=1))) * 1.05

    N   = len(pos_com)
    G   = grid_res
    dv  = 2.0 * r_outer / G

    # Scott's bandwidth: h = factor × σ × N^{-1/5}  for d=3
    sigma_data = np.std(pos_com)
    h_scott    = scott_factor * sigma_data * (N ** (-1.0 / 5.0))
    h_scott    = max(h_scott, 0.5)   # floor at 0.5 kpc (prevents over-sharpening)

    # Fit KDE on all particles
    kde_fitter = KernelDensity(bandwidth=h_scott, kernel=KDE_KERNEL,
                                algorithm='ball_tree')
    kde_fitter.fit(pos_com)

    # Build evaluation grid (cube)
    xi       = np.linspace(-r_outer, r_outer, G)
    xg, yg, zg = np.meshgrid(xi, xi, xi, indexing='ij')
    pts_flat = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)   # (G³, 3)

    # Evaluate KDE — score_samples returns log(density)
    # Process in chunks to avoid memory overflow for large G
    chunk    = 8192
    log_rho  = np.empty(len(pts_flat))
    for i in range(0, len(pts_flat), chunk):
        log_rho[i:i+chunk] = kde_fitter.score_samples(pts_flat[i:i+chunk])

    rho_kde  = np.exp(log_rho).reshape(G, G, G)
    return rho_kde, dv, h_scott

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §B.2  MINKOWSKI FUNCTIONALS ON KDE FIELD                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# (This function is unchanged in structure from §36 original, but is now
#  applied to a KDE density field instead of a CIC field.  The key physical
#  difference is that the KDE field is SMOOTH BY CONSTRUCTION and requires
#  NO additional Gaussian filter.  The MF computation is identical.)

def compute_mf_from_grid(rho_grid, rho_threshold, voxel_size):
    """
    Compute the four Minkowski functionals of the excursion set
    { x : rho_grid(x) > rho_threshold }.

    Returns a dict: W0, W1, W2, W3, chi, T_MF, W_MF, L_MF, P_MF, F_MF,
                    n_components, valid.
    """
    dv  = voxel_size
    dv3 = dv ** 3
    dv2 = dv ** 2
    nan_out  = {k: np.nan for k in ['W0','W1','W2','W3','chi',
                                     'T_MF','W_MF','L_MF','P_MF','F_MF',
                                     'n_components']}
    nan_out['valid'] = False

    B    = rho_grid > rho_threshold
    N_in = int(B.sum())
    if N_in < 10:     # need at least a few voxels for a meaningful surface
        return nan_out

    # W0 — volume
    W0 = N_in * dv3

    # W1 — surface area / 6  via exposed-face count
    pad    = np.pad(B, 1, mode='constant', constant_values=False)
    faces  = (  (B & ~pad[2:,  1:-1, 1:-1]).sum()
              + (B & ~pad[:-2, 1:-1, 1:-1]).sum()
              + (B & ~pad[1:-1, 2:,  1:-1]).sum()
              + (B & ~pad[1:-1, :-2, 1:-1]).sum()
              + (B & ~pad[1:-1, 1:-1, 2: ]).sum()
              + (B & ~pad[1:-1, 1:-1, :-2]).sum())
    W1     = float(faces) * dv2 / 6.0

    # W3 — Euler characteristic
    if _HAVE_SKIMAGE:
        chi    = int(euler_number(B, connectivity=3))
        _, n_comp = ndimage_label(B)
    else:
        _, n_comp = ndimage_label(B)
        chi = 2 * n_comp          # rough proxy: each blob ≈ genus-0
    W3 = float(chi) / (4.0 * np.pi)

    # W2 — mean curvature integral (sphere proxy)
    r_eff = (3.0 * W0 / (4.0 * np.pi)) ** (1.0 / 3.0) if W0 > 0 else 1.0
    W2    = W1 / (r_eff + 1e-30)

    # Shapefinders
    T_MF = 3.0 * W0 / (W1 + 1e-30)
    W_MF = W1 / (2.0 * W2 + 1e-30)
    L_MF = W2 / (3.0 * W3 + 1e-30) if (np.isfinite(W3) and abs(W3) > 1e-10) \
           else np.nan

    P_MF = np.clip((W_MF - T_MF) / (W_MF + T_MF + 1e-30), 0.0, 1.0)
    F_MF = np.clip((L_MF - W_MF) / (L_MF + W_MF + 1e-30), 0.0, 1.0) \
           if np.isfinite(L_MF) else np.nan

    return {'W0': W0, 'W1': W1, 'W2': W2, 'W3': W3, 'chi': chi,
            'T_MF': T_MF, 'W_MF': W_MF, 'L_MF': L_MF,
            'P_MF': P_MF, 'F_MF': F_MF,
            'n_components': n_comp, 'valid': True}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §B.3  PERSISTENCE TOPOLOGY SCAN                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Theory: Sub-Level Set Filtration for H₀ (Connected Components)
# ───────────────────────────────────────────────────────────────
# A SUB-LEVEL SET FILTRATION works as follows:
#   Start with a very HIGH density threshold ρ_max (empty excursion set).
#   Slowly DECREASE the threshold toward zero.
#   At each step, new voxels are added to the excursion set.
#   The topology of the excursion set changes when:
#     • A new isolated component APPEARS (birth) — a new density peak is found
#     • Two components MERGE (death of the younger one)
#
# This is DUAL to the standard sub-level set filtration in topology:
# here we scan from high to low density, which gives us H₀ (components)
# rather than H₂ (voids).  For the density field of a merger, this is
# the physically relevant choice: we want to track density PEAKS.
#
# The PERSISTENCE of a component is:
#   persistence = ρ_birth − ρ_death
#
# where ρ_birth is the threshold at which the component first appears
# and ρ_death is the threshold at which it merges with a denser component.
#
# Interpretation:
#   persistence >> 0 : the component is a robust, isolated density peak.
#                      In our context: the MW halo core and the M31 nucleus.
#   persistence ≈ 0  : the component is a transient fluctuation — shot noise
#                      or a tiny subhalo.  Filtered out by PERSIST_TOL.
#
# PERSISTENCE DIAGRAM for H₀:
#   A scatter plot of (ρ_birth, ρ_death) for each component.
#   All points lie below the diagonal (ρ_birth > ρ_death).
#   Points far from the diagonal have high persistence.
#   The SINGLE LONGEST-LIVED secondary peak (after the main halo)
#   is the M31 nucleus — its persistence should PEAK near the pericentric
#   passage and VANISH at final merger.
#
# MERGE THRESHOLD ρ_merge:
#   The density threshold at which the two dominant peaks (MW + M31)
#   first become connected.  This directly measures the epoch at which
#   the two halos are no longer gravitationally isolated.
#
# Time complexity:  O(N_PERSIST_THRESH × G³)  — label computation dominates
# Space complexity: O(G³)  — one binary array per threshold

def persistence_scan(rho_grid, n_thresh=N_PERSIST_THRESH,
                     persist_tol=PERSIST_TOL):
    """
    Compute the H₀ persistence diagram of a 3D density field by
    scanning the sub-level set filtration from high to low density.

    Parameters
    ----------
    rho_grid    : (G, G, G)  — density or overdensity field (NOT smoothed separately;
                               the KDE field is already smooth)
    n_thresh    : int        — number of threshold levels to scan
    persist_tol : float      — fractional persistence threshold for noise suppression;
                               features with persistence < persist_tol × (ρ_max − ρ_min)
                               are discarded as noise

    Returns
    -------
    persist_pairs : list of (rho_birth, rho_death)  — persistence pairs for
                    all non-noise components, sorted by persistence (descending)
    n_births      : int    — total number of distinct components born during scan
    persist_max   : float  — persistence of the longest-lived secondary component
                             (0 if there is only one component at all thresholds)
    rho_merge     : float  — density threshold at which the two dominant components
                             first merge (NaN if only one component throughout)
    n_noise       : int    — number of components discarded as noise

    PITFALL: connected-component labelling with scipy.ndimage.label uses 6-connectivity
    by default (face-connected).  For DM halos, 26-connectivity (face + edge + corner)
    would give fewer spurious splits, but it is MUCH slower.  Keep 6-connectivity for
    speed and accept that thin bridges between halos will count as connections earlier.

    PITFALL: the merge tree construction here is O(n_thresh × G³).
    For G=64 and n_thresh=80 this is ~5 × 10^8 voxel operations per snapshot —
    potentially slow.  Use G=32–48 for the persistence scan even if you use G=64
    elsewhere.  The topology (connectivity) is insensitive to grid resolution
    once the smoothing scale is resolved.
    """
    rho_max = float(rho_grid.max())
    rho_min = float(rho_grid.min())
    dr      = rho_max - rho_min
    eps_tol = persist_tol * dr      # minimum persistence to be non-noise

    # Scan from HIGH to LOW (super-level set: adding voxels as threshold descends)
    thresholds = np.linspace(rho_max, rho_min + 1e-6, n_thresh)

    # Track component birth/death
    # component_id → rho_birth, rho_death
    # We track by monitoring connected components at each step.
    # Key: when n_components increases, new components are born at current threshold.
    # When n_components decreases, components merged at current threshold.

    prev_n_comp  = 0
    prev_labels  = None
    births       = {}    # {comp_global_id: rho_birth}  — birth thresholds
    merge_events = []    # [(rho_merge, comp_A_birth, comp_B_birth)]
    birth_counter = 0    # unique ID for each born component
    rho_merge_two = np.nan  # threshold at which exactly two main comps merge to one

    # We need to track WHICH new component emerges at each birth.
    # Simple approach: at each threshold step, label the excursion set.
    # Compare component count:
    #   increase → birth of new component(s)
    #   decrease → merge of component(s)
    # We do NOT need to track spatial correspondence (that is the expensive step
    # in full persistent homology).  Instead, we record:
    #   births at each threshold, deaths at each threshold.

    birth_rhos    = []
    death_rhos    = []
    n_comp_ts     = []   # n_components at each threshold

    for th in thresholds:
        B       = rho_grid > th
        _, nc   = ndimage_label(B)
        n_comp_ts.append(nc)

        if nc > prev_n_comp:
            # new component(s) born
            for _ in range(nc - prev_n_comp):
                birth_rhos.append(float(th))
                birth_counter += 1
        elif nc < prev_n_comp:
            # merge(s) happened
            n_merges = prev_n_comp - nc
            for _ in range(n_merges):
                # The youngest (highest birth threshold) component dies
                if birth_rhos:
                    b = max(birth_rhos)   # youngest component born last
                    birth_rhos.remove(b)
                    death_rhos.append((b, float(th)))

        # Record the threshold at which n_comp drops from 2 to 1
        if prev_n_comp == 2 and nc == 1 and np.isnan(rho_merge_two):
            rho_merge_two = float(th)

        prev_n_comp = nc

    # Pair remaining components as dying at rho_min
    for b in birth_rhos:
        death_rhos.append((b, float(rho_min)))

    # Compute persistence pairs and filter noise
    persist_pairs_all  = [(b, d, b - d) for (b, d) in death_rhos]
    persist_pairs_all.sort(key=lambda x: -x[2])   # sort by persistence descending

    persist_pairs = [(b, d) for (b, d, p) in persist_pairs_all if p >= eps_tol]
    n_noise       = len(persist_pairs_all) - len(persist_pairs)
    n_births      = len(persist_pairs_all)

    # Longest-lived SECONDARY component (the dominant is always the main halo)
    persist_max = 0.0
    if len(persist_pairs) >= 2:
        # The main halo has birth = rho_max (first component born)
        # The second-longest-lived is M31 or a satellite
        b1, d1 = persist_pairs[0]   # longest (main halo)
        b2, d2 = persist_pairs[1]   # second longest (M31 or satellite)
        persist_max = float(b2 - d2)
    elif len(persist_pairs) == 1:
        b1, d1 = persist_pairs[0]
        persist_max = float(b1 - d1)

    return persist_pairs, n_births, persist_max, rho_merge_two, n_noise

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §B.4  MAIN KDE + PERSISTENCE LOOP                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Allocate KDE Minkowski output arrays
P_kde_arr   = np.full((N_KDE_SNAPS, N_THRESH_KDE), np.nan)
F_kde_arr   = np.full((N_KDE_SNAPS, N_THRESH_KDE), np.nan)
chi_kde_arr = np.full((N_KDE_SNAPS, N_THRESH_KDE), np.nan)
W0_kde_arr  = np.full((N_KDE_SNAPS, N_THRESH_KDE), np.nan)
thresh_kde  = np.full((N_KDE_SNAPS, N_THRESH_KDE), np.nan)
h_scott_ts  = np.full(N_KDE_SNAPS, np.nan)    # bandwidth used per epoch

# Allocate persistence output arrays
persist_max_ts   = np.full(N_KDE_SNAPS, np.nan)   # persistence of 2nd component
rho_merge_ts     = np.full(N_KDE_SNAPS, np.nan)   # merge threshold
n_births_ts      = np.full(N_KDE_SNAPS, np.nan)   # total components born
n_noise_ts       = np.full(N_KDE_SNAPS, np.nan)   # noise components filtered
f_noise_ts       = np.full(N_KDE_SNAPS, np.nan)   # fraction that are noise

# Persistence diagram storage (list of per-epoch diagrams)
persist_diagrams  = [None] * N_KDE_SNAPS          # each is a list of (b, d) pairs

print("\n  Computing KDE density fields + Minkowski + persistence topology …")
t0_kde = time.time()

for k, s in enumerate(snap_idx_kde):
    com    = _traj_pos[s].mean(axis=0)
    pos_c  = _traj_pos[s] - com

    # Build the KDE density grid
    rho_kde, dv_k, h_k = build_kde_density_grid(pos_c, GRID_RES_KDE, R_OUTER,
                                                  SCOTT_FACTOR)
    h_scott_ts[k] = h_k

    # Threshold scan for Minkowski functionals
    rho_lo   = np.percentile(rho_kde, 5)
    rho_hi   = np.percentile(rho_kde, 95)
    ths      = np.linspace(rho_lo, rho_hi, N_THRESH_KDE)
    thresh_kde[k] = ths

    for j, th in enumerate(ths):
        mf = compute_mf_from_grid(rho_kde, th, dv_k)
        if not mf['valid']:
            continue
        P_kde_arr[k, j]   = mf['P_MF']
        F_kde_arr[k, j]   = mf['F_MF']
        chi_kde_arr[k, j] = mf['chi']
        W0_kde_arr[k, j]  = mf['W0']

    # Persistence topology scan
    # Use a denser threshold grid than the MF scan for finer birth/death resolution.
    pairs, nb, pmax, rho_mg, nn = persistence_scan(rho_kde, N_PERSIST_THRESH,
                                                     PERSIST_TOL)
    persist_diagrams[k]  = pairs
    n_births_ts[k]       = float(nb)
    persist_max_ts[k]    = float(pmax)
    rho_merge_ts[k]      = float(rho_mg) if not np.isnan(rho_mg) else np.nan
    n_noise_ts[k]        = float(nn)
    f_noise_ts[k]        = float(nn) / (float(nb) + 1e-10)

    if (k + 1) % 5 == 0:
        print(f"    KDE snap {k+1}/{N_KDE_SNAPS}  "
              f"h={h_k:.2f} kpc  persist_max={pmax:.3f}  "
              f"rho_merge={rho_merge_ts[k]:.3e}", flush=True)

print(f"  KDE + persistence done in {time.time()-t0_kde:.1f} s")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §B.5  MF SUMMARY STATISTICS FROM KDE FIELD                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

mid_j      = N_THRESH_KDE // 2
P_med_kde  = np.nanmean(P_kde_arr[:, mid_j-2:mid_j+2], axis=1)
F_med_kde  = np.nanmean(F_kde_arr[:, mid_j-2:mid_j+2], axis=1)
F_peak_kde = np.nanmax(F_kde_arr, axis=1)
chi_min_kde = np.nanmin(chi_kde_arr, axis=1)

# A_PF: area under the (P, F) morphology curve at each epoch
A_PF_kde   = np.full(N_KDE_SNAPS, np.nan)
for k in range(N_KDE_SNAPS):
    Pv = P_kde_arr[k]; Fv = F_kde_arr[k]
    fin = np.isfinite(Pv) & np.isfinite(Fv)
    if fin.sum() >= 3:
        A_PF_kde[k] = float(np.abs(np.trapz(Fv[fin], Pv[fin])))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING HELPER                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _ax(ax, xlabel='', ylabel='', title='', log_x=False, log_y=False,
        xlim=None, ylim=None, grid=True):
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title:  ax.set_title(title,  fontsize=9)
    if log_x:  ax.set_xscale('log')
    if log_y:  ax.set_yscale('log')
    if xlim:   ax.set_xlim(xlim)
    if ylim:   ax.set_ylim(ylim)
    if grid:   ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)

EPOCH_COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
EPOCH_IDX_F  = [0, N_FABRIC_SNAPS//4, N_FABRIC_SNAPS//2,
                3*N_FABRIC_SNAPS//4, N_FABRIC_SNAPS-1]
EPOCH_IDX_K  = [0, N_KDE_SNAPS//4, N_KDE_SNAPS//2,
                3*N_KDE_SNAPS//4, N_KDE_SNAPS-1]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1 — Woodcock (C, K) phase diagram, all epochs and shells          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("§36(alt) Fig 1 — Woodcock Fabric Parameters (C, K)", fontsize=10,
             fontweight='bold')

# Left: (C, K) scatter coloured by time
cmap_time = plt.cm.plasma
for k in range(N_FABRIC_SNAPS):
    C_v = C_wood_arr[k]; K_v = K_wood_arr[k]
    fin = np.isfinite(C_v) & np.isfinite(K_v)
    if not fin.any():
        continue
    ax1.scatter(C_v[fin], K_v[fin], s=10, alpha=0.3,
                c=[plt.cm.plasma(k / N_FABRIC_SNAPS)] * fin.sum(),
                vmin=0, vmax=1)

ax1.axhline(1.0, color='k', lw=0.8, ls='--', label='K=1 (triaxial)')
ax1.axhline(0.3, color='gray', lw=0.6, ls=':', label='K≈0.3 (oblate girdle)')
ax1.axvline(0.5, color='lightcoral', lw=0.6, ls=':', label='C=0.5 (weakly anisotropic)')
ax1.set_xlim(0); ax1.set_ylim(0)
ax1.text(0.05, 0.95, 'OBLATE\n(girdle)', transform=ax1.transAxes,
         fontsize=8, va='top', color='navy')
ax1.text(0.7, 0.95, 'PROLATE\n(cluster)', transform=ax1.transAxes,
         fontsize=8, va='top', color='darkred')
sm1 = plt.cm.ScalarMappable(cmap=cmap_time,
      norm=Normalize(vmin=time_fab[0], vmax=time_fab[-1]))
sm1.set_array([])
plt.colorbar(sm1, ax=ax1, label='Time [Gyr]', shrink=0.7)
_ax(ax1, xlabel='C = ln(t₃/t₁)  (fabric strength)',
    ylabel='K = ln(t₃/t₂)/ln(t₂/t₁)  (fabric shape)',
    title='Woodcock diagram — all shells all epochs')
ax1.legend(fontsize=7, loc='upper left')

# Right: C_global(t) and K_global(t) time series
ax2.plot(time_fab, C_global_ts, color='royalblue', lw=1.5, label='C global (strength)')
ax2r = ax2.twinx()
ax2r.plot(time_fab, K_global_ts, color='darkorange', lw=1.5, ls='--', label='K global (shape)')
ax2r.axhline(1.0, color='gray', lw=0.6, ls=':')
ax2r.set_ylabel('K  (shape)', fontsize=9, color='darkorange')
ax2r.tick_params(colors='darkorange', labelsize=7)
_ax(ax2, xlabel='Time [Gyr]', ylabel='C  (strength)',
    title='Global Woodcock parameters over time')
ax2.legend(fontsize=7, loc='upper left')
ax2r.legend(fontsize=7, loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_woodcock_diagram.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 1 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2 — C(r,t) and K(r,t) heatmaps                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("§36(alt) Fig 2 — Fabric Parameter Radial Heatmaps", fontsize=10,
             fontweight='bold')

ext = [time_fab[0], time_fab[-1],
       np.log10(r_mid_fab[0]), np.log10(r_mid_fab[-1])]
yt  = np.log10([2, 5, 10, 30, 80])
ytl = ['2','5','10','30','80']

C_plot = np.where(np.isfinite(C_wood_arr), C_wood_arr, 0)
K_plot = np.where(np.isfinite(K_wood_arr), K_wood_arr, np.nan)

im1 = ax1.imshow(C_plot.T, origin='lower', aspect='auto', extent=ext,
                  cmap='magma', vmin=0, vmax=np.nanpercentile(C_plot, 96))
plt.colorbar(im1, ax=ax1, label='C = ln(t₃/t₁)', pad=0.02)
ax1.set_yticks(yt); ax1.set_yticklabels(ytl)
_ax(ax1, xlabel='Time [Gyr]', ylabel='log₁₀(r [kpc])', title='C(r,t) — fabric strength')

K_vmax = min(np.nanpercentile(K_plot[np.isfinite(K_plot)], 95), 5.0) \
         if np.any(np.isfinite(K_plot)) else 3.0
im2 = ax2.imshow(np.where(np.isfinite(K_plot), K_plot, 0).T,
                  origin='lower', aspect='auto', extent=ext,
                  cmap='RdBu_r', vmin=0, vmax=K_vmax)
plt.colorbar(im2, ax=ax2, label='K  (oblate ←0  1→ prolate)', pad=0.02)
ax2.set_yticks(yt); ax2.set_yticklabels(ytl)
ax2.axvline(time_fab[N_FABRIC_SNAPS//2], color='white', lw=1, ls='--', alpha=0.8)
_ax(ax2, xlabel='Time [Gyr]', ylabel='log₁₀(r [kpc])', title='K(r,t) — fabric shape')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_fabric_heatmaps.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 2 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3 — Density-weighted Ẽ₂ vs. equal-weight Ẽ₂ (bias diagnostic)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, axes3 = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("§36(alt) Fig 3 — Weighted vs. Equal-Weight Multipole Bias Diagnostic",
             fontsize=10, fontweight='bold')
ax3a, ax3b, ax3c = axes3

# Q_bar_w(r,t) heatmap
vmax3a = max(np.nanpercentile(Q_bar_w_arr, 96), 0.01)
im3a   = ax3a.imshow(Q_bar_w_arr.T, origin='lower', aspect='auto', extent=ext,
                      cmap='plasma', vmin=0, vmax=vmax3a)
plt.colorbar(im3a, ax=ax3a, label='Ẽ₂ (density-weighted)', pad=0.02)
ax3a.set_yticks(yt); ax3a.set_yticklabels(ytl)
_ax(ax3a, xlabel='Time [Gyr]', ylabel='log₁₀(r [kpc])',
    title='Density-weighted quadrupole Q̄_w(r,t)')

# ΔQ = weighted − equal-weight heatmap (bias map)
dQ_vmax = max(np.nanpercentile(np.abs(delta_Q_arr), 95), 0.01)
im3b    = ax3b.imshow(delta_Q_arr.T, origin='lower', aspect='auto', extent=ext,
                       cmap='RdBu_r', vmin=-dQ_vmax, vmax=dQ_vmax)
plt.colorbar(im3b, ax=ax3b, label='ΔQ = Ẽ₂^w − Ẽ₂^eq', pad=0.02)
ax3b.set_yticks(yt); ax3b.set_yticklabels(ytl)
_ax(ax3b, xlabel='Time [Gyr]', ylabel='log₁₀(r [kpc])',
    title='Bias ΔQ(r,t):  red=weighted>equal  blue=equal>weighted')

# Global time series comparison
ax3c.plot(time_fab, Q_global_w_ts, color='royalblue', lw=1.5, label='Ẽ₂ weighted')
# equal-weight global: compute from mean of delta_Q + weighted
dQ_global = np.nanmean(delta_Q_arr, axis=1)
E2_eq_global = Q_global_w_ts - dQ_global
ax3c.plot(time_fab, E2_eq_global, color='darkorange', lw=1.5, ls='--',
          label='Ẽ₂ equal-weight')
ax3c.fill_between(time_fab, Q_global_w_ts, E2_eq_global,
                   alpha=0.2, color='gray', label='Bias ΔQ')
_ax(ax3c, xlabel='Time [Gyr]', ylabel='Ẽ₂ (global)',
    title='Global weighted vs. equal-weight Ẽ₂ comparison')
ax3c.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_weighted_bias.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 3 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4 — KDE Minkowski functional curves at 5 epochs                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, axes4 = plt.subplots(1, 5, figsize=(14, 3.5), sharey=False)
fig.suptitle("§36(alt) Fig 4 — KDE Minkowski Curves W(ρ_th) at 5 Epochs",
             fontsize=10, fontweight='bold')

for idx, (ei, col) in enumerate(zip(EPOCH_IDX_K, EPOCH_COLORS)):
    ax  = axes4[idx]
    th  = thresh_kde[ei]
    th_n = (th - th.min()) / (np.ptp(th) + 1e-30)

    ax.plot(th_n, P_kde_arr[ei], color='royalblue', lw=1.4, label='P_MF')
    ax.plot(th_n, F_kde_arr[ei], color='darkorange', lw=1.4, ls='--', label='F_MF')
    ax2r = ax.twinx()
    ax2r.step(th_n, chi_kde_arr[ei], where='mid', color='crimson', lw=1.0, alpha=0.6)
    ax2r.set_ylabel('χ', fontsize=7, color='crimson')
    ax2r.tick_params(labelsize=6, colors='crimson')
    _ax(ax, xlabel='ρ_th / ρ_range', ylabel='P, F' if idx == 0 else '',
        title=f't={time_kde[ei]:.2f} Gyr')
    if idx == 0:
        ax.legend(fontsize=6)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_kde_mf_curves.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 4 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5 — Persistence diagrams at 5 epochs                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, axes5 = plt.subplots(1, 5, figsize=(14, 3.5))
fig.suptitle("§36(alt) Fig 5 — H₀ Persistence Diagrams (birth vs. death)",
             fontsize=10, fontweight='bold')

for idx, (ei, col) in enumerate(zip(EPOCH_IDX_K, EPOCH_COLORS)):
    ax   = axes5[idx]
    diag = persist_diagrams[ei]

    if diag and len(diag) > 0:
        bs = [b for (b, d) in diag]
        ds = [d for (b, d) in diag]
        ax.scatter(bs, ds, s=30, color=col, edgecolors='k', linewidths=0.4,
                   zorder=4, label=f'n={len(diag)} comps')
        # Diagonal (zero persistence line)
        lim_lo = min(min(ds), min(bs)) if ds and bs else 0
        lim_hi = max(max(bs), 1e-10) if bs else 1
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=0.7, alpha=0.5)
        # Annotate the longest-lived secondary component
        if len(diag) >= 2:
            b2, d2 = diag[1]
            ax.annotate(f'M31?\npersist={b2-d2:.3f}',
                        xy=(b2, d2), xytext=(b2*0.6, d2*1.3),
                        fontsize=6, color='darkred',
                        arrowprops=dict(arrowstyle='->', color='darkred', lw=0.7))
    else:
        ax.text(0.5, 0.5, 'single\ncomponent', ha='center', va='center',
                fontsize=9, transform=ax.transAxes, color='gray')

    _ax(ax, xlabel='ρ_birth', ylabel='ρ_death' if idx == 0 else '',
        title=f't={time_kde[ei]:.2f} Gyr')
    if len(diag) > 0:
        ax.legend(fontsize=6)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_persistence_diagrams.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 5 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6 — Persistence summary time series                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, axes6 = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
fig.suptitle("§36(alt) Fig 6 — Persistence Topology Time Series", fontsize=10,
             fontweight='bold')
ax6a, ax6b, ax6c = axes6

ax6a.plot(time_kde, persist_max_ts, color='royalblue', lw=1.5, marker='o', ms=3,
          label='Persistence of 2nd component (M31 proxy)')
ax6a.fill_between(time_kde, 0, persist_max_ts, alpha=0.2, color='royalblue')
ax6a.set_ylabel('Persistence [ρ units]', fontsize=9)
ax6a.legend(fontsize=7)
ax6a.set_title('Longest-lived secondary density peak — M31 nucleus lifetime', fontsize=8)
ax6a.grid(True, lw=0.3, alpha=0.4)

ax6b.plot(time_kde, rho_merge_ts, color='crimson', lw=1.5, marker='s', ms=3,
          label='ρ_merge (MW + M31 halos first connect)')
ax6b.set_ylabel('ρ_merge [kpc⁻³]', fontsize=9)
ax6b.legend(fontsize=7)
ax6b.set_title('Merge threshold ρ_merge(t) — direct topological merger indicator', fontsize=8)
ax6b.grid(True, lw=0.3, alpha=0.4)

ax6c.bar(time_kde, n_births_ts,   width=(time_kde[1]-time_kde[0])*0.45,
         align='edge', color='steelblue', alpha=0.7, label='Total births')
ax6c.bar(time_kde, n_noise_ts,    width=-(time_kde[1]-time_kde[0])*0.45,
         align='edge', color='salmon',    alpha=0.7, label='Noise components filtered')
ax6c.set_xlabel('Time [Gyr]', fontsize=9)
ax6c.set_ylabel('Count', fontsize=9)
ax6c.legend(fontsize=7)
ax6c.set_title('Component births and noise fraction per epoch', fontsize=8)
ax6c.grid(True, lw=0.3, alpha=0.4)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_persistence_timeseries.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 6 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 7 — KDE (P, F) morphology plane                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, ax7 = plt.subplots(figsize=(6, 6))
fig.suptitle("§36(alt) Fig 7 — KDE (P, F) Morphology Plane", fontsize=10,
             fontweight='bold')

ax7.fill([0,1,0,0],[0,0,1,0], color='whitesmoke', zorder=0)
ax7.plot([0,1,0,0],[0,0,1,0],'k--',lw=0.7)
for lbl, xy in [('Sphere',(0,0)),('Pancake',(1,0)),('Filament',(0,1))]:
    ax7.text(xy[0]-0.03, xy[1]-0.07, lbl, fontsize=8, ha='center', color='gray')

cmap7 = plt.cm.viridis
for k in range(N_KDE_SNAPS):
    Pv = P_kde_arr[k]; Fv = F_kde_arr[k]
    fin = np.isfinite(Pv) & np.isfinite(Fv)
    if fin.sum() < 3:
        continue
    col = cmap7(k / N_KDE_SNAPS)
    ax7.plot(Pv[fin], Fv[fin], color=col, lw=0.9, alpha=0.6)
    ax7.scatter(P_med_kde[k], F_med_kde[k], s=22, color=col, zorder=5,
                edgecolors='k', linewidths=0.3)

sm7 = plt.cm.ScalarMappable(cmap=cmap7,
      norm=Normalize(vmin=time_kde[0], vmax=time_kde[-1]))
sm7.set_array([])
plt.colorbar(sm7, ax=ax7, label='Time [Gyr]', shrink=0.7)
_ax(ax7, xlabel='Planarity P_MF', ylabel='Filamentarity F_MF',
    title='KDE morphology curves (each line = one epoch, sweeping ρ_th)')
ax7.set_xlim(-0.05,1.05); ax7.set_ylim(-0.05,1.05)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_kde_pf_plane.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 7 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 8 — Woodcock K vs. Minkowski F: direct morphology comparison     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Both K_wood and F_MF measure prolate-vs-oblate structure, but:
#   K_wood comes from the ANGULAR DISTRIBUTION of particles (pure geometry)
#   F_MF comes from the DENSITY FIELD topology (shape of isodensity surfaces)
# Scatter between the two reveals where the two representations diverge.

def _resamp(t_src, y_src, t_dst):
    fin = np.isfinite(y_src)
    if fin.sum() < 2:
        return np.full(len(t_dst), np.nan)
    f = _interp1d(t_src[fin], y_src[fin], bounds_error=False, fill_value=np.nan)
    return f(t_dst)

K_at_kde   = _resamp(time_fab, K_global_ts, time_kde)
C_at_kde   = _resamp(time_fab, C_global_ts, time_kde)

fig, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("§36(alt) Fig 8 — Woodcock vs. Minkowski Cross-Method Comparison",
             fontsize=10, fontweight='bold')

fin8 = np.isfinite(K_at_kde) & np.isfinite(F_med_kde)
sc8a = ax8a.scatter(K_at_kde[fin8], F_med_kde[fin8], s=50,
                     c=time_kde[fin8], cmap='plasma',
                     edgecolors='k', linewidths=0.4, zorder=4)
plt.colorbar(sc8a, ax=ax8a, label='Time [Gyr]', shrink=0.8)
ax8a.axvline(1.0, color='k', lw=0.7, ls='--', label='K=1 (triaxial)')
_ax(ax8a, xlabel='K_Woodcock  (oblate→0  triaxial=1  prolate→∞)',
    ylabel='F_MF  (KDE filamentarity)',
    title='K vs. F_MF  —  angular fabric vs. density topology')
ax8a.legend(fontsize=7)

fin8b = np.isfinite(C_at_kde) & np.isfinite(F_med_kde)
sc8b  = ax8b.scatter(C_at_kde[fin8b], F_med_kde[fin8b], s=50,
                      c=time_kde[fin8b], cmap='viridis',
                      edgecolors='k', linewidths=0.4, zorder=4)
plt.colorbar(sc8b, ax=ax8b, label='Time [Gyr]', shrink=0.8)
_ax(ax8b, xlabel='C_Woodcock  (fabric strength)',
    ylabel='F_MF  (KDE filamentarity)',
    title='C vs. F_MF  —  fabric strength vs. topology')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_woodcock_vs_mf.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 8 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 9 — Scott's bandwidth evolution h(t): adaptive smoothing insight  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, (ax9a, ax9b) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
fig.suptitle("§36(alt) Fig 9 — KDE Adaptive Bandwidth and MF Summary",
             fontsize=10, fontweight='bold')

ax9a.plot(time_kde, h_scott_ts, color='steelblue', lw=1.5, label="h_Scott [kpc]")
ax9a.fill_between(time_kde, h_scott_ts * 0.8, h_scott_ts * 1.2,
                   alpha=0.2, color='steelblue', label='±20% band')
_ax(ax9a, ylabel='Bandwidth h [kpc]',
    title="Scott's adaptive KDE bandwidth — smaller h during denser merger phase")
ax9a.legend(fontsize=7)

ax9b.plot(time_kde, F_med_kde, color='darkorange', lw=1.5, label='F_MF (filamentarity)')
ax9b.plot(time_kde, P_med_kde, color='royalblue',  lw=1.5, ls='--', label='P_MF (planarity)')
ax9b.plot(time_kde, A_PF_kde,  color='green',      lw=1.2, ls=':', label='A_PF (curve area)')
_ax(ax9b, xlabel='Time [Gyr]', ylabel='MF shape parameters',
    title='KDE Minkowski shape parameters vs. time')
ax9b.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36alt_kde_bandwidth_mf.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 9 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 10 — Master summary panel                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig10 = plt.figure(figsize=(14, 11))
fig10.suptitle("§36(alt) Master Summary — Fabric Tensor + KDE Minkowski + Persistence",
               fontsize=11, fontweight='bold')
gs = gridspec.GridSpec(3, 2, figure=fig10, hspace=0.48, wspace=0.35)
a00 = fig10.add_subplot(gs[0, 0])
a01 = fig10.add_subplot(gs[0, 1])
a10 = fig10.add_subplot(gs[1, 0])
a11 = fig10.add_subplot(gs[1, 1])
a20 = fig10.add_subplot(gs[2, 0])
a21 = fig10.add_subplot(gs[2, 1])

# (0,0) C(r,t) heatmap
im00 = a00.imshow(C_plot.T, origin='lower', aspect='auto', extent=ext,
                   cmap='magma', vmin=0, vmax=np.nanpercentile(C_plot,96))
plt.colorbar(im00, ax=a00, label='C (strength)', pad=0.02)
a00.set_yticks(yt); a00.set_yticklabels(ytl)
_ax(a00, xlabel='Time [Gyr]', ylabel='log₁₀(r)', title='C(r,t) Woodcock strength')

# (0,1) K(r,t) heatmap
K_plot = np.where(np.isfinite(K_wood_arr), K_wood_arr, 0)
im01 = a01.imshow(K_plot.T, origin='lower', aspect='auto', extent=ext,
                   cmap='RdBu_r', vmin=0, vmax=K_vmax)
plt.colorbar(im01, ax=a01, label='K (oblate←0  1→prolate)', pad=0.02)
a01.set_yticks(yt); a01.set_yticklabels(ytl)
_ax(a01, xlabel='Time [Gyr]', ylabel='log₁₀(r)', title='K(r,t) Woodcock shape')

# (1,0) Persistence time series
a10.plot(time_kde, persist_max_ts, color='royalblue', lw=1.5, label='persist_max (M31)')
a10r = a10.twinx()
a10r.plot(time_kde, rho_merge_ts, color='crimson', lw=1.3, ls='--', label='ρ_merge')
a10r.set_ylabel('ρ_merge', fontsize=8, color='crimson')
a10r.tick_params(colors='crimson', labelsize=6)
_ax(a10, xlabel='Time [Gyr]', ylabel='Persistence', title='Persistence topology over time')
a10.legend(fontsize=7, loc='upper left')
a10r.legend(fontsize=7, loc='upper right')

# (1,1) KDE (P, F) morphology plane — density map
fin_pf = np.isfinite(P_kde_arr) & np.isfinite(F_kde_arr)
if fin_pf.any():
    P_all2 = P_kde_arr[fin_pf].ravel()
    F_all2 = F_kde_arr[fin_pf].ravel()
    h2d, xe2, ye2 = np.histogram2d(P_all2, F_all2, bins=25, range=[[0,1],[0,1]])
    a11.imshow(h2d.T, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='hot_r')
a11.plot([0,1,0,0],[0,0,1,0],'k--',lw=0.7)
_ax(a11, xlabel='P_MF', ylabel='F_MF', title='KDE (P,F) density — all epochs')
a11.set_xlim(0,1); a11.set_ylim(0,1)

# (2,0) Woodcock vs. MF comparison
a20.scatter(K_at_kde[fin8], F_med_kde[fin8], s=30,
            c=time_kde[fin8], cmap='plasma',
            edgecolors='k', linewidths=0.3)
a20.axvline(1.0, color='k', lw=0.7, ls='--')
_ax(a20, xlabel='K_wood', ylabel='F_MF', title='K vs. F — angular vs. topology')

# (2,1) Six-quantity normalised time series
def _n01(arr):
    lo = np.nanmin(arr); hi = np.nanmax(arr)
    return (arr - lo) / (hi - lo + 1e-30)

Q_wn     = _n01(Q_global_w_ts)
Cn       = _n01(C_global_ts)
Fn       = _n01(F_med_kde)
pn2      = _n01(persist_max_ts)
rho_mn   = _n01(-np.where(np.isfinite(rho_merge_ts), rho_merge_ts, np.nan))

a21.plot(time_fab, Q_wn, lw=1.3, color='royalblue', label='Ẽ₂_w (weighted quad.)')
a21.plot(time_fab, Cn,   lw=1.3, color='darkorange', label='C_wood (strength)')
a21.plot(_resamp(time_kde, time_kde, time_fab),
         _resamp(time_kde, Fn, time_fab),
         lw=1.3, color='green', label='F_MF (KDE topology)')
a21.plot(time_kde, pn2, lw=1.3, color='crimson', ls='--', label='persist_max')
a21.plot(time_kde, rho_mn, lw=1.3, color='purple', ls=':', label='−ρ_merge (merger)')
_ax(a21, xlabel='Time [Gyr]', ylabel='Normalised [0→1]',
    title='Six-method normalised comparison')
a21.legend(fontsize=6, ncol=2)

plt.savefig(os.path.join(OUT_DIR, "section36alt_summary_panel.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig10)
print("  Fig 10 (summary) saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ANIMATION — Fabric tensor and persistence morphology evolution           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n  Building animation …")
fig_a, axes_a = plt.subplots(1, 3, figsize=(13, 4.2))
fig_a.suptitle("§36(alt) — Fabric & Persistence Morphological Evolution", fontsize=10,
               fontweight='bold')
ax_fab, ax_pf_a, ax_ts_a = axes_a

# ── Left: Woodcock (C, K) live scatter ───────────────────────────────────────
scat_ck = ax_fab.scatter([], [], s=15, c=[], cmap='plasma',
                          vmin=0, vmax=np.nanmax(r_mid_fab), alpha=0.7,
                          edgecolors='none')
ax_fab.axhline(1.0, color='k', lw=0.7, ls='--')
ax_fab.set_xlim(0, max(np.nanmax(C_wood_arr) * 1.1, 0.1))
ax_fab.set_ylim(0, min(max(np.nanmax(K_wood_arr[np.isfinite(K_wood_arr)]), 2.0), 6.0)
               if np.any(np.isfinite(K_wood_arr)) else 4.0)
_ax(ax_fab, xlabel='C (strength)', ylabel='K (shape)', title='Woodcock diagram')
ax_fab.text(0.05, 0.92, 'OBLATE', transform=ax_fab.transAxes, fontsize=8, color='navy')
ax_fab.text(0.65, 0.92, 'PROLATE', transform=ax_fab.transAxes, fontsize=8, color='darkred')

# ── Centre: (P, F) morphology curve ──────────────────────────────────────────
ax_pf_a.fill([0,1,0,0],[0,0,1,0], color='whitesmoke', zorder=0)
ax_pf_a.plot([0,1,0,0],[0,0,1,0],'k--',lw=0.7)
pf_line, = ax_pf_a.plot([], [], 'b-', lw=1.5, zorder=4)
pf_dot   = ax_pf_a.scatter([], [], s=70, color='blue', zorder=5,
                             edgecolors='k', linewidths=0.5)
hist_pf  = []
ax_pf_a.set_xlim(-0.05,1.05); ax_pf_a.set_ylim(-0.05,1.05)
_ax(ax_pf_a, xlabel='P_MF', ylabel='F_MF', title='KDE morphology curve')

# ── Right: persist_max and C_global time series ───────────────────────────────
ax_ts_a.set_xlim(min(time_fab[0], time_kde[0]), max(time_fab[-1], time_kde[-1]))
ax_ts_a.set_ylim(0, max(np.nanmax(C_global_ts) * 1.1, 0.1))
ts_Cn,  = ax_ts_a.plot([], [], color='darkorange', lw=1.5, label='C_global')
ts_pm,  = ax_ts_a.plot([], [], color='royalblue', lw=1.4, ls='--', label='persist_max (scaled)')
ts_vl   = ax_ts_a.axvline(time_fab[0], color='gray', lw=0.8, ls=':')
_ax(ax_ts_a, xlabel='Time [Gyr]', ylabel='Value', title='C & persist_max over time')
ax_ts_a.legend(fontsize=7)

# Match MF snapshot indices to multipole snapshot indices
kde_of_fab = np.searchsorted(snap_idx_kde, snap_idx_fab, side='left')
kde_of_fab = np.clip(kde_of_fab, 0, N_KDE_SNAPS - 1)

persist_scale = np.nanmax(C_global_ts) / (np.nanmax(persist_max_ts) + 1e-10)

def _update(frame):
    k    = frame
    k_mf = kde_of_fab[k]

    # Woodcock scatter
    Cv  = C_wood_arr[k]; Kv = K_wood_arr[k]
    fin = np.isfinite(Cv) & np.isfinite(Kv)
    if fin.any():
        scat_ck.set_offsets(np.column_stack([Cv[fin], Kv[fin]]))
        scat_ck.set_array(r_mid_fab[fin])
    ax_fab.set_title(f'Woodcock  t={time_fab[k]:.2f} Gyr', fontsize=8)

    # (P, F) curve
    if k_mf > 0:
        Pp = P_kde_arr[k_mf-1]; Fp = F_kde_arr[k_mf-1]
        finp = np.isfinite(Pp) & np.isfinite(Fp)
        if finp.sum() >= 2:
            hl, = ax_pf_a.plot(Pp[finp], Fp[finp], color='lightgray', lw=0.6,
                               alpha=0.4, zorder=2)
            hist_pf.append(hl)
            if len(hist_pf) > 8:
                hist_pf[0].remove(); hist_pf.pop(0)

    Pc = P_kde_arr[k_mf]; Fc = F_kde_arr[k_mf]
    fin_c = np.isfinite(Pc) & np.isfinite(Fc)
    if fin_c.sum() >= 2:
        pf_line.set_data(Pc[fin_c], Fc[fin_c])
    if np.isfinite(P_med_kde[k_mf]) and np.isfinite(F_med_kde[k_mf]):
        pf_dot.set_offsets([[P_med_kde[k_mf], F_med_kde[k_mf]]])
    ax_pf_a.set_title(f'KDE (P,F)  t={time_kde[k_mf]:.2f} Gyr', fontsize=8)

    # Time series
    ts_Cn.set_data(time_fab[:k+1], C_global_ts[:k+1])
    ts_pm.set_data(time_kde[:k_mf+1], persist_max_ts[:k_mf+1] * persist_scale)
    ts_vl.set_xdata([time_fab[k], time_fab[k]])
    ax_ts_a.set_title(f'C={C_global_ts[k]:.3f}  PM={persist_max_ts[k_mf]:.3f}', fontsize=8)

    return [scat_ck, pf_line, pf_dot, ts_Cn, ts_pm, ts_vl]

anim = animation.FuncAnimation(fig_a, _update, frames=N_FABRIC_SNAPS,
                                 interval=1000 // ANIM_FPS, blit=False)
plt.tight_layout()
anim_path = os.path.join(OUT_DIR, "section36alt_animation.mp4")
try:
    writer = animation.FFMpegWriter(fps=ANIM_FPS, bitrate=ANIM_KBPS)
    anim.save(anim_path, writer=writer, dpi=ANIM_DPI)
    print(f"  Animation saved → {anim_path}")
except Exception as exc:
    gif_path = anim_path.replace('.mp4', '.gif')
    anim.save(gif_path, writer='pillow', fps=ANIM_FPS)
    print(f"  GIF fallback → {gif_path}")
plt.close(fig_a)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CROSS-SECTION CORRELATIONS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "─"*72)
print("  §36(alt) Cross-Method Pearson Correlations")
print("─"*72)

def _sp(a, b):
    fin = np.isfinite(a) & np.isfinite(b)
    if fin.sum() < 5:
        return np.nan, np.nan
    return pearsonr(a[fin], b[fin])

# Resample onto a common time grid (kde time)
C_kde  = _resamp(time_fab, C_global_ts, time_kde)
K_kde  = _resamp(time_fab, K_global_ts, time_kde)
Qw_kde = _resamp(time_fab, Q_global_w_ts, time_kde)

corrs = [
    ("C_Woodcock",    "F_MF",           C_kde,          F_med_kde),
    ("K_Woodcock",    "F_MF",           K_kde,          F_med_kde),
    ("C_Woodcock",    "persist_max",    C_kde,          persist_max_ts),
    ("persist_max",   "rho_merge",      persist_max_ts, rho_merge_ts),
    ("Ẽ₂_weighted",  "C_Woodcock",     Qw_kde,         C_kde),
    ("F_MF(KDE)",     "F_MF(CIC)",      F_med_kde,      F_med_kde * 0 + np.nan),  # placeholder
    ("n_births",      "C_Woodcock",     n_births_ts,    C_kde),
]

hdr = f"  {'Quantity 1':<22} {'Quantity 2':<20} {'r':>8} {'p':>12}"
print(hdr); print("  " + "─"*66)
for n1, n2, a, b in corrs:
    r, p = _sp(np.asarray(a, float), np.asarray(b, float))
    rs = f"{r:+.4f}" if np.isfinite(r) else "    —"
    ps = f"{p:.3e}"  if np.isfinite(p) else "    —"
    print(f"  {n1:<22} {n2:<20} {rs:>8} {ps:>12}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MORPHOLOGICAL CLASSIFICATION TABLE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _classify(C, K, F, P, persist):
    """Classify using both Woodcock fabric and Minkowski topology."""
    tags = []
    if not np.isfinite(C) or C < 0.05:
        tags.append("isotropic")
    elif np.isfinite(K):
        if K > 1.5:
            tags.append("prolate-cluster")
        elif K < 0.5:
            tags.append("oblate-girdle")
        else:
            tags.append("triaxial")
    if np.isfinite(F) and F > 0.3:
        tags.append("filamentary (KDE)")
    elif np.isfinite(P) and P > 0.3:
        tags.append("planar (KDE)")
    if np.isfinite(persist) and persist > 0.1:
        tags.append("two-component (M31 distinct)")
    return " + ".join(tags) if tags else "undetermined"

ep_names  = ["t = initial", "t = peri (proxy)", "t = final"]
ep_mf_idx = [0, N_KDE_SNAPS//2, N_KDE_SNAPS-1]
ep_fab_idx = [0, N_FABRIC_SNAPS//2, N_FABRIC_SNAPS-1]

C_at_mf = _resamp(time_fab, C_global_ts, time_kde)
K_at_mf = _resamp(time_fab, K_global_ts, time_kde)

print("\n" + "─"*90)
print("  §36(alt) Morphological Classification Table")
print("─"*90)
hdr2 = (f"  {'Epoch':<22} {'C':>6} {'K':>6} {'F_MF':>6} "
        f"{'P_MF':>6} {'persist':>8}  Classification")
print(hdr2); print("  " + "─"*88)
for name, ki in zip(ep_names, ep_mf_idx):
    C   = float(C_at_mf[ki])     if np.isfinite(C_at_mf[ki])     else np.nan
    K   = float(K_at_mf[ki])     if np.isfinite(K_at_mf[ki])     else np.nan
    F   = float(F_med_kde[ki])   if np.isfinite(F_med_kde[ki])   else np.nan
    P   = float(P_med_kde[ki])   if np.isfinite(P_med_kde[ki])   else np.nan
    pm  = float(persist_max_ts[ki]) if np.isfinite(persist_max_ts[ki]) else np.nan
    cls = _classify(C, K, F, P, pm)
    Cs  = f"{C:6.3f}" if np.isfinite(C) else "     —"
    Ks  = f"{K:6.3f}" if np.isfinite(K) else "     —"
    Fs  = f"{F:6.3f}" if np.isfinite(F) else "     —"
    Ps  = f"{P:6.3f}" if np.isfinite(P) else "     —"
    pms = f"{pm:8.4f}" if np.isfinite(pm) else "       —"
    print(f"  {name:<22} {Cs} {Ks} {Fs} {Ps} {pms}  {cls}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  OUTPUT MANIFEST                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

outputs_36alt = [
    "section36alt_woodcock_diagram.png",
    "section36alt_fabric_heatmaps.png",
    "section36alt_weighted_bias.png",
    "section36alt_kde_mf_curves.png",
    "section36alt_persistence_diagrams.png",
    "section36alt_persistence_timeseries.png",
    "section36alt_kde_pf_plane.png",
    "section36alt_woodcock_vs_mf.png",
    "section36alt_kde_bandwidth_mf.png",
    "section36alt_animation.mp4",
    "section36alt_summary_panel.png",
]

print("\n" + "="*80)
print("  SECTION 36 (ALT) COMPLETE")
print("="*80)
print(f"  {'File':<50} {'Status'}")
print("  " + "─"*62)
for fname in outputs_36alt:
    fp     = os.path.join(OUT_DIR, fname)
    exists = os.path.isfile(fp)
    sz     = f"{os.path.getsize(fp)//1024} KB" if exists else "—"
    mark   = "✓" if exists else "✗  MISSING"
    print(f"  {fname:<50} {mark}  {sz}")

print(f"""
  Strategy comparison summary:
  ┌─────────────────────────┬──────────────────────┬──────────────────────┐
  │  Diagnostic             │  §36 ORIGINAL        │  §36 ALTERNATIVE     │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │  Multipole estimator    │  Equal-weight 1/N    │  KNN density-weighted│
  │  Angular shape metric   │  Ẽ_l per degree      │  Woodcock C, K       │
  │  Density field          │  CIC + Gaussian blur │  Epanechnikov KDE    │
  │  Bandwidth              │  Fixed σ_smooth      │  Scott's rule h(N,σ) │
  │  Topology               │  Fixed-threshold χ   │  Persistence diagram │
  │  Merger indicator       │  χ jumps             │  ρ_merge + persist   │
  │  Noise suppression      │  Manual SMOOTH_SIGMA │  PERSIST_TOL filter  │
  └─────────────────────────┴──────────────────────┴──────────────────────┘
""")
