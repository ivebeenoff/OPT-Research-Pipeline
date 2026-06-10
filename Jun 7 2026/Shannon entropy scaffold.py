"

===============================================================================
SECTION 34 — SHANNON ENTROPY OF VELOCITY & ANISOTROPY DISTRIBUTIONS
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
Sections 31–33 characterised chaos and mixing using three complementary
lenses: trajectory divergence (λ), spectral complexity (C), and phase-space
density (Q).  All three are PARTICLE-LEVEL diagnostics — they describe what
a single orbit is doing.

This section takes a POPULATION-LEVEL view.  Instead of asking "is particle i
chaotic?", we ask:

    HOW ORDERED IS THE VELOCITY DISTRIBUTION OF THE WHOLE SYSTEM?

The answer is the Shannon differential entropy H of the velocity distribution:

    H[f_v]  =  −∫ f_v(v) ln f_v(v) d³v

For a COLD, ORDERED system (e.g. a rotating disc before the merger):
  f_v is narrow and peaked → H is LOW (distribution is ordered, predictable)

For a HOT, ISOTROPIC, VIRIALISED system (e.g. a pressure-supported halo):
  f_v is broad and diffuse → H is HIGH (distribution is disordered)

For a system MID-MERGER with tidal streams:
  f_v has multiple peaks (stream + halo components) → H has intermediate
  values with STRUCTURE that a single temperature or σ_v cannot capture.

Shannon entropy detects asymmetries, multi-modality, and non-Gaussianity that
standard velocity dispersion σ_v CANNOT see — two distributions can have
identical σ_v but very different H.

Connection to §33 anisotropy β(r, t)
──────────────────────────────────────
Section 33 measured the RATIO of dispersions β = 1 − σ_t²/(2σ_r²).
That is a second-moment statistic — it is blind to the SHAPE of the
distribution within each component.

Here we compute H separately for the radial and tangential velocity PDFs:
    H_r(r, t)     — entropy of f(v_r) in each shell
    H_t(r, t)     — entropy of f(v_t) in each shell
    ΔH(r, t)      = H_r − H_t   — entropy anisotropy

ΔH > 0 means the radial distribution is MORE disordered than tangential —
a signature of radial infall streams contaminating an otherwise smooth halo.
ΔH < 0 means tangential is more disordered — rare, indicates retrograde mixing.

The entropy anisotropy ΔH is a STRICTLY STRONGER diagnostic than β:
  • ΔH detects multi-modality (two counter-rotating streams → double-peaked
    f(v_t) but β ≈ 0)
  • ΔH is sensitive to heavy tails (high-velocity escapers inflate H
    without much changing σ)

Key quantities computed in this section
────────────────────────────────────────
  H_r(r, t)         — Shannon entropy of radial velocity PDF per shell
  H_t(r, t)         — Shannon entropy of tangential velocity PDF per shell
  H_3D(r, t)        — joint 3D velocity entropy per shell
  ΔH(r, t)          — entropy anisotropy: H_r − H_t
  H_global(t)       — global entropy of the full 3D velocity distribution
  H_Gauss(σ)        — reference entropy of a Gaussian: ln(σ √(2πe))
  ΔH_excess(r, t)   — H − H_Gauss(σ)  excess entropy above Gaussian
  KL(r, t)          — KL divergence from a Maxwellian reference
  JS(r, t)          — Jensen–Shannon divergence between adjacent epochs
  H_aniso_ts(t)     — global entropy anisotropy vs. time
  f_multimodal(t)   — fraction of shells with multi-modal velocity PDFs
  H_groups(g, t)    — entropy per particle group (inner/mid/outer/M31)

Dependencies
────────────
  numpy              — histogram, KDE approximation
  scipy.stats        — gaussian_kde, entropy (for KL divergence)
  scipy.signal       — find_peaks (for multi-modality detection)
  Section 26         — traj_pos, traj_vel (Lagrangian trajectories)
  Section 33         — beta_profile, r_edges, r_mid_aniso (for overlay plots)

All globals from the parent pipeline are inherited.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.stats import gaussian_kde, entropy as scipy_entropy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: The key choice here is how to estimate the velocity PDF.
# Two options:
#
# OPTION A — Histogram:
#   Fast.  O(N) per bin.  Bin width choice matters enormously.
#   Scott's rule: h = 3.49 σ N^{-1/3}
#   Freedman–Diaconis rule: h = 2 IQR N^{-1/3}  ← preferred for heavy tails
#   Entropy from a histogram with bin width h:
#     H ≈ −Σ_k p_k ln(p_k / h)
#   where p_k is the probability mass in bin k.
#   The division by h converts from probability mass to probability density,
#   which is needed for the differential entropy to be meaningful.
#   WARNING: do NOT use H = −Σ p_k ln p_k (no /h) — that is the discrete
#   entropy and its value depends on the bin width, making comparisons
#   across radial bins (which have different σ) meaningless.
#
# OPTION B — Kernel Density Estimate (KDE):
#   Smoother.  O(N²) per bin (or O(N log N) with fast summation).
#   scipy.stats.gaussian_kde handles bandwidth selection automatically
#   (Scott's or Silverman's rule).
#   Entropy from a KDE: H ≈ −(1/N) Σ_i ln f̂(v_i)
#   where f̂(v_i) is the KDE evaluated at each data point.
#   This is the Monte Carlo estimator of the differential entropy.
#   More accurate than histograms for N < 200 per bin.
#
# RECOMMENDATION: use KDE for per-shell estimates (N_per_shell typically
# 10–100) and histograms for the global estimate (N ~ 5000).

N_ENTROPY_SNAPS  = 50         # number of snapshot epochs for entropy computation
N_ENTROPY_BINS   = 25         # radial bins for shell entropy profiles
N_VEL_BINS       = 50         # velocity histogram bins (global estimate)
MIN_SHELL_PARTS  = 15         # minimum particles per shell for valid KDE
KDE_BANDWIDTH    = 'scott'    # KDE bandwidth rule: 'scott' or 'silverman'
MULTIMODAL_PROM  = 0.15       # minimum peak prominence as fraction of max height
                               # for a shell velocity PDF to be "multi-modal"
N_KL_BINS        = 60         # bins for KL divergence histogram approximation

# Reference Maxwellian σ for KL divergence: use the global velocity dispersion
# at t=0 as the fixed reference so the KL divergence is a measure of
# departure from the INITIAL equilibrium, not just from any Maxwellian.
# Set dynamically in §34.1: KL_REF_SIGMA = np.std(vel_all_t0)

# Animation.
ANIM_FPS_34      = 18
ANIM_DPI_34      = 100
ANIM_BITRATE_34  = 1600

print("\n" + "="*80)
print("  SECTION 34 · Shannon Entropy of Velocity & Anisotropy Distributions")
print("="*80)
print(f"  Snapshot epochs   : {N_ENTROPY_SNAPS}")
print(f"  Radial shells     : {N_ENTROPY_BINS}")
print(f"  Min particles/shell: {MIN_SHELL_PARTS}")
print(f"  KDE bandwidth     : {KDE_BANDWIDTH}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.1 — LOAD TRAJECTORIES AND DECOMPOSE VELOCITIES                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: Reuse traj_pos and traj_vel from Section 26.  Copy the standard
# try/except fallback block.
#
# Unlike §33, we do NOT need a 6D phase-space array.  We need only:
#   pos(t, i) : (ns, N, 3)  — positions [kpc]
#   vel(t, i) : (ns, N, 3)  — velocities [km/s]
#   r(t, i)   : (ns, N)     — radial distances [kpc]
#
# VELOCITY DECOMPOSITION into spherical components at each snapshot:
#   v_r(t, i)   = v · r̂       — signed radial velocity [km/s]
#                               positive = outward, negative = inward
#   v_t(t, i)   = |v × r̂|    — tangential speed magnitude [km/s]
#                               always ≥ 0
#   v_phi(t, i) = (v × r̂) · ẑ — azimuthal velocity (signed) [km/s]
#                               positive = prograde, negative = retrograde
#
# HINT: for v_r:
#   r_hat = pos / r[:, :, np.newaxis]   shape: (ns, N, 3)
#   v_r   = np.sum(vel * r_hat, axis=2) shape: (ns, N)
#
# HINT: for v_t (tangential speed):
#   v_perp = vel - v_r[:,:,np.newaxis] * r_hat   shape: (ns, N, 3)
#   v_t    = np.linalg.norm(v_perp, axis=2)      shape: (ns, N)
#
# HINT: for v_phi (azimuthal, signed):
#   z_hat = [0, 0, 1]
#   v_phi = np.sum(np.cross(r_hat, vel, axis=2) * z_hat, axis=2)
#   This gives the component of angular momentum along z, per unit mass.
#
# WHY v_phi matters:
#   A distribution that is symmetric in v_phi (equal prograde/retrograde)
#   has MAXIMUM entropy for a given σ_phi — it is "maximally disordered"
#   azimuthally.  A one-sided (prograde-dominated) distribution has LOWER
#   entropy — it retains memory of the initial rotation.
#   Tracking H(v_phi) separately reveals when this rotational memory is lost.
#
# Also compute the reference Maxwellian sigma from t=0:
#   KL_REF_SIGMA = np.std(vel[0, :, :])   — global, over all particles and axes
#
# TIME COMPLEXITY:  O(ns × N × 3)  — vectorised, fast
# SPACE COMPLEXITY: O(ns × N × 3) for decomposed velocities ≈ same as input

# TODO: implement trajectory loading / inheritance
_traj_pos  = None   # replace with inherited or recomputed (ns, N, 3)
_traj_vel  = None   # (ns, N, 3)
_traj_r    = None   # (ns, N)
_r0        = None   # (N,) initial radii
_group     = None   # (N,) 0=inner, 1=mid, 2=outer, 3=M31
_N         = 0

# TODO: compute velocity decomposition
# r_hat   = _traj_pos / (_traj_r[:, :, np.newaxis] + 1e-10)  # (ns, N, 3)
# v_r     = np.sum(_traj_vel * r_hat, axis=2)                 # (ns, N)
# v_perp  = _traj_vel - v_r[:, :, np.newaxis] * r_hat         # (ns, N, 3)
# v_t     = np.linalg.norm(v_perp, axis=2)                    # (ns, N)
# z_hat   = np.array([0.0, 0.0, 1.0])
# v_phi   = np.sum(np.cross(r_hat, _traj_vel, axis=2) * z_hat, axis=2)  # (ns,N)
#
# KL_REF_SIGMA = float(np.nanstd(_traj_vel[0]))
# snap_indices = np.linspace(0, ns - 1, N_ENTROPY_SNAPS, dtype=int)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.2 — CORE ENTROPY ESTIMATORS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The DIFFERENTIAL Shannon entropy of a continuous 1D distribution f(v) is:
#
#   H[f]  =  −∫ f(v) ln f(v) dv          [nats]
#
# Key properties:
#   1. H is INVARIANT to the location of f (shifting f left/right preserves H).
#      This is why H captures shape, not mean — ideal for comparing shells
#      at different mean radial velocities.
#
#   2. H INCREASES monotonically with spread.
#      For a Gaussian:  H_Gauss = (1/2) ln(2πe σ²) = ln(σ) + (1/2)ln(2πe)
#      For a uniform distribution on [a, b]:  H_uniform = ln(b − a)
#      For a Laplacian with scale b:  H_Laplace = ln(2eb)
#
#   3. For a FIXED variance, H is MAXIMISED by the Gaussian.
#      This means:  H_excess = H − H_Gauss(σ) ≤ 0
#      with equality only if f is exactly Gaussian.
#      H_excess < 0 means the distribution is MORE ordered than Gaussian
#        (e.g. a stream: narrow peak with almost no tails).
#      H_excess > 0 is NOT possible for a Gaussian reference — but CAN occur
#        if you use the wrong σ.  Always use the SAME-SAMPLE σ.
#
#   4. H is NOT defined for distributions with zero density regions if those
#      regions are given zero weight bins.  The KDE approach avoids this
#      because a Gaussian kernel has infinite support.
#
# NOTE on units:
#   Natural log (ln) gives entropy in NATS.
#   Log base 2 gives BITS.
#   We use nats throughout for consistency with §33 (mixing entropy).
#   H_Gauss(σ) = ln(σ) + 0.5 × ln(2πe) ≈ ln(σ) + 1.4189

_LN_2PIE = float(0.5 * np.log(2 * np.pi * np.e))   # = 1.4189...

def h_gaussian(sigma):
    """
    Reference entropy of a 1D Gaussian with standard deviation sigma.

    H_Gauss(σ) = ln(σ √(2πe)) = ln(σ) + (1/2) ln(2πe)

    Parameters
    ----------
    sigma : float or array  — standard deviation [km/s]

    Returns
    -------
    H : float or array  — entropy in nats

    HINT: for a 3D isotropic Gaussian with independent components,
    H_3D = H_r + H_theta + H_phi = 3 × H_Gauss(σ) if all σ are equal.
    For anisotropic: H_3D = ln(σ_r) + ln(σ_t) + ln(σ_phi) + 3 × (1/2)ln(2πe).
    """
    return np.log(np.maximum(sigma, 1e-30)) + _LN_2PIE


def kde_entropy_1d(samples, bandwidth=KDE_BANDWIDTH):
    """
    Estimate the differential Shannon entropy of a 1D sample using KDE.

    Method: Monte Carlo estimator
        H ≈ −(1/N) Σ_i ln f̂(v_i)
    where f̂ is the KDE evaluated at each data point.

    Parameters
    ----------
    samples   : (N,)  — velocity samples [km/s]
    bandwidth : str or float  — KDE bandwidth rule or fixed value

    Returns
    -------
    H     : float  — estimated entropy in nats
    sigma : float  — standard deviation of samples (for H_excess computation)

    HINT: scipy.stats.gaussian_kde with bw_method=bandwidth.
    Evaluate f̂ at the sample points themselves: kde(samples).
    These self-evaluated log-densities are the Monte Carlo estimate.

    HINT: add a floor 1e-30 before log to guard against KDE values that
    round to zero at the sample location (can happen with very narrow
    bandwidth or isolated outliers).

    HINT: for N < MIN_SHELL_PARTS, return np.nan for both H and sigma.
    This prevents noisy estimates from thin shells contaminating the
    radial profiles.
    """
    samples = samples[np.isfinite(samples)]
    if len(samples) < MIN_SHELL_PARTS:
        return np.nan, np.nan
    kde   = gaussian_kde(samples, bw_method=bandwidth)
    log_f = np.log(np.maximum(kde(samples), 1e-30))
    H     = float(-np.mean(log_f))
    sigma = float(np.std(samples))
    return H, sigma


def hist_entropy_1d(samples, n_bins=N_VEL_BINS):
    """
    Estimate the differential Shannon entropy using a histogram.

    Correct formula for differential entropy from a histogram:
        H ≈ −Σ_k p_k ln(p_k / h)  =  −Σ_k p_k ln p_k + ln h
    where p_k is the probability mass in bin k and h is the bin width.

    Parameters
    ----------
    samples : (N,)  — velocity samples [km/s]
    n_bins  : int   — number of histogram bins

    Returns
    -------
    H     : float  — estimated entropy in nats
    sigma : float  — standard deviation of samples

    HINT: use np.histogram with density=False to get counts, then normalise
    to get p_k.  Bin width h = (v_max - v_min) / n_bins.

    HINT: the Freedman–Diaconis bin width is more robust than Scott's rule
    for multi-modal or heavy-tailed distributions:
        h_FD = 2 × IQR(samples) × N^{-1/3}
    Use this as the default if n_bins is set to 'auto'.

    HINT: exclude zero-count bins from the sum — 0 × ln(0) = 0 by convention,
    but ln(0) = -∞ causes NaN.  Mask with p_k > 0.
    """
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2:
        return np.nan, np.nan
    counts, bin_edges = np.histogram(samples, bins=n_bins)
    h    = bin_edges[1] - bin_edges[0]         # bin width
    p    = counts / counts.sum()               # probability mass
    mask = p > 0
    H    = float(-np.sum(p[mask] * np.log(p[mask])) + np.log(h))
    return H, float(np.std(samples))


def h_excess(H, sigma):
    """
    Compute the excess entropy above a Gaussian reference.

    H_excess = H − H_Gauss(sigma)

    A Gaussian has the maximum entropy for a given variance, so:
        H_excess ≤ 0  always (when sigma is computed from the SAME sample)

    Positive H_excess signals that the sample sigma is underestimated
    relative to the actual spread — check for outliers.

    Parameters
    ----------
    H     : float  — estimated entropy [nats]
    sigma : float  — standard deviation of the same sample [km/s]

    Returns
    -------
    dH : float  — excess entropy [nats]
    """
    return H - h_gaussian(sigma)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.3 — KL DIVERGENCE FROM A MAXWELLIAN REFERENCE                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The KL divergence from a reference distribution P to an observed distribution
# Q is:
#
#   KL(Q ‖ P)  =  ∫ Q(v) ln[ Q(v) / P(v) ] dv
#
# We use P = Maxwellian with σ = KL_REF_SIGMA (the t=0 global dispersion).
# KL(Q ‖ P) measures how much information is needed to encode Q if you only
# have P to work with — equivalently, how "surprising" the current distribution
# is relative to the initial equilibrium.
#
# For a Maxwellian: P(v) = (1/√(2πσ²)) exp(−v²/2σ²)
#
# Properties of KL divergence:
#   KL ≥ 0  always (Gibbs inequality).
#   KL = 0  iff Q = P (current distribution = reference).
#   KL is NOT symmetric: KL(Q ‖ P) ≠ KL(P ‖ Q).
#   KL is sensitive to the TAILS of the distribution — rare high-velocity
#   particles (e.g. tidal debris) produce large KL even with small mass.
#
# The KL divergence captures what σ_v and β CANNOT:
#   • Two distributions with the same σ but different shapes can have
#     KL > 0.
#   • A tidal stream adds a narrow peak far from v = 0; this is invisible
#     to σ but produces large KL.
#
# Implementation via histogram
# ──────────────────────────────
# Use a fine histogram to approximate both Q and P on the same grid:
#   v_grid    = np.linspace(v_min, v_max, N_KL_BINS)
#   Q_hist, _ = np.histogram(samples, bins=v_grid, density=True)
#   P_vals    = gaussian_pdf(v_grid_mid, sigma=KL_REF_SIGMA)
#   KL        = scipy_entropy(Q_hist + epsilon, P_vals + epsilon) × h
#
# scipy.stats.entropy(pk, qk) computes Σ pk × log(pk/qk) — discrete KL.
# Multiplying by bin width h converts to the differential version.
#
# TIME COMPLEXITY:  O(N_ENTROPY_SNAPS × N_ENTROPY_BINS × N_KL_BINS)  — fast
# SPACE COMPLEXITY: O(N_ENTROPY_SNAPS × N_ENTROPY_BINS)  — one scalar per bin

def kl_from_maxwellian(samples, sigma_ref, n_bins=N_KL_BINS):
    """
    Estimate the KL divergence KL(Q ‖ P) where P is a Maxwellian.

    Parameters
    ----------
    samples   : (N,)  — velocity samples (1D component) [km/s]
    sigma_ref : float — reference Maxwellian standard deviation [km/s]
    n_bins    : int   — histogram bins

    Returns
    -------
    kl : float  — KL divergence in nats

    HINT: use a symmetric velocity range [−5σ_ref, +5σ_ref] for the histogram
    so that the reference Maxwellian tails are well captured.

    HINT: add a small floor epsilon = 1e-10 to both histograms before
    computing scipy_entropy to avoid log(0/0) = NaN.

    HINT: if the sample is entirely within a narrow range (e.g. a cold stream),
    the histogram will be very spiky.  The KL will be large and meaningful —
    do NOT smooth it away.
    """
    samples = samples[np.isfinite(samples)]
    if len(samples) < MIN_SHELL_PARTS:
        return np.nan

    v_min  = -5.0 * sigma_ref
    v_max  =  5.0 * sigma_ref
    edges  = np.linspace(v_min, v_max, n_bins + 1)
    h      = edges[1] - edges[0]
    v_mid  = 0.5 * (edges[:-1] + edges[1:])

    Q_hist, _ = np.histogram(samples, bins=edges, density=True)
    P_vals    = (1.0 / (sigma_ref * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * (v_mid / sigma_ref)**2)

    eps = 1e-10
    kl  = float(scipy_entropy(Q_hist + eps, P_vals + eps) * h)
    return kl

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.4 — JENSEN–SHANNON DIVERGENCE BETWEEN EPOCHS                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The Jensen–Shannon divergence between two distributions P and Q is:
#
#   JSD(P ‖ Q)  =  (1/2) KL(P ‖ M) + (1/2) KL(Q ‖ M)
#   where M = (P + Q) / 2  is the mixture distribution.
#
# Unlike KL, JSD is:
#   • Symmetric: JSD(P, Q) = JSD(Q, P)
#   • Bounded: 0 ≤ JSD ≤ ln(2) ≈ 0.693 nats
#   • Defined even when P or Q have zero-probability regions
#
# We use JSD to measure HOW MUCH the velocity distribution changes between
# consecutive snapshot epochs:
#   JS(t)  =  JSD[ f_v(t),  f_v(t − Δt) ]
#
# This gives the "rate of change" of the velocity distribution in information-
# theoretic terms:
#   JS ≈ 0  : distribution is stationary (virialised)
#   JS large: rapid redistribution of velocities (merger phase)
#
# KEY DIAGNOSTIC: compare JS(t) to dS_mix/dt from §33.
# Both measure "how fast things are changing" but in different spaces:
#   dS_mix/dt lives in 6D phase space; JS(t) lives in 3D velocity space.
# Discrepancies reveal spatial reshuffling (mixing) without velocity change
# or velocity change without spatial spreading.
#
# Implementation
# ──────────────
# For each consecutive pair of epochs (k, k+1):
#   Compute P from f_v at epoch k (histogram over the full particle set)
#   Compute Q from f_v at epoch k+1
#   JSD = (1/2) scipy_entropy(P+ε, M+ε) + (1/2) scipy_entropy(Q+ε, M+ε)
#   where M = (P + Q) / 2
#
# TIME COMPLEXITY:  O(N_ENTROPY_SNAPS × N_VEL_BINS)  — negligible
# SPACE COMPLEXITY: O(N_ENTROPY_SNAPS)  — one scalar per epoch pair

def jensen_shannon(p_hist, q_hist):
    """
    Compute the Jensen–Shannon divergence between two normalised histograms.

    Parameters
    ----------
    p_hist : (M,)  — normalised probability mass vector (sums to 1)
    q_hist : (M,)  — normalised probability mass vector (sums to 1)

    Returns
    -------
    jsd : float  — Jensen–Shannon divergence in nats ∈ [0, ln 2]

    HINT: both histograms MUST be computed on the SAME bin edges.
    Use a fixed global velocity range [V_GLOBAL_MIN, V_GLOBAL_MAX] computed
    from the t=0 distribution as: ±5 × global σ_v at t=0.
    This ensures the JSD compares distributions on the same support.

    HINT: scipy.stats.entropy(pk, qk) = Σ pk log(pk/qk).
    For JSD you need TWO calls with the mixture M as the second argument.
    """
    eps  = 1e-10
    m    = 0.5 * (p_hist + q_hist)
    jsd  = 0.5 * scipy_entropy(p_hist + eps, m + eps) + \
           0.5 * scipy_entropy(q_hist + eps, m + eps)
    return float(jsd)

# TODO: compute JS divergence time series
# V_GLOBAL_RANGE = 5.0 * KL_REF_SIGMA
# v_edges_global = np.linspace(-V_GLOBAL_RANGE, V_GLOBAL_RANGE, N_VEL_BINS + 1)
# JS_ts = np.full(N_ENTROPY_SNAPS - 1, np.nan)
# for k in range(N_ENTROPY_SNAPS - 1):
#     s0, s1 = snap_indices[k], snap_indices[k + 1]
#     vel_flat_0 = _traj_vel[s0].ravel()
#     vel_flat_1 = _traj_vel[s1].ravel()
#     p0, _ = np.histogram(vel_flat_0, bins=v_edges_global, density=False)
#     p1, _ = np.histogram(vel_flat_1, bins=v_edges_global, density=False)
#     p0 = p0 / (p0.sum() + 1e-30)
#     p1 = p1 / (p1.sum() + 1e-30)
#     JS_ts[k] = jensen_shannon(p0, p1)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.5 — RADIAL SHELL ENTROPY PROFILES                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The entropy PROFILE H(r, t) resolves where the velocity distribution is
# ordered or disordered as a function of radius.
#
# Expected structure:
#   Inner halo (r < 5 kpc):
#     The deep potential mixes orbits rapidly.  H_r should be high and nearly
#     uniform — the core is pressure-supported and close to Maxwellian.
#     H_excess ≈ 0 throughout the merger.
#
#   Mid halo (5–30 kpc):
#     Tidal streams cross through this region during pericentric passages.
#     Expect H_r to DECREASE (narrow stream adds a cold spike to f(v_r))
#     followed by recovery as the stream phase-mixes.
#
#   Outer halo / M31 debris (r > 30 kpc):
#     Before pericentric passage: narrow cold distribution (low H).
#     During passage: heated by tidal shocks → H rises sharply.
#     After passage: bi-modal distribution (bound + unbound debris) → H peaks.
#
#   Entropy anisotropy ΔH = H_r − H_t:
#     ΔH > 0 during infall: radial distribution smeared by streaming; tangential
#             is relatively cold.
#     ΔH < 0 at late times: tangential mixing catches up to radial.
#     ΔH ≈ 0: isotropic entropy — virialised system.
#
# Implementation
# ──────────────
# For each snapshot epoch k and each radial shell b:
#   1. Select particles in shell b: mask = (r_edges[b] ≤ r < r_edges[b+1])
#   2. Extract v_r_shell, v_t_shell, v_phi_shell for those particles
#   3. Call kde_entropy_1d on each component
#   4. Store H_r[k, b], H_t[k, b], H_phi[k, b]
#   5. Compute H_excess_r[k, b] = h_excess(H_r[k, b], sigma_r[k, b])
#
# TIME COMPLEXITY:  O(N_ENTROPY_SNAPS × N_ENTROPY_BINS × N_per_shell²)
#                 ≈ O(50 × 25 × 30²) ≈ 1.1 × 10^6 — fast
# SPACE COMPLEXITY: O(N_ENTROPY_SNAPS × N_ENTROPY_BINS) per array ≈ 50 KB

# TODO: allocate shell entropy arrays
# H_r_profile      = np.full((N_ENTROPY_SNAPS, N_ENTROPY_BINS), np.nan)
# H_t_profile      = np.full((N_ENTROPY_SNAPS, N_ENTROPY_BINS), np.nan)
# H_phi_profile    = np.full((N_ENTROPY_SNAPS, N_ENTROPY_BINS), np.nan)
# H_excess_r_prof  = np.full((N_ENTROPY_SNAPS, N_ENTROPY_BINS), np.nan)
# dH_profile       = np.full((N_ENTROPY_SNAPS, N_ENTROPY_BINS), np.nan)  # H_r − H_t
# sigma_r_profile  = np.full((N_ENTROPY_SNAPS, N_ENTROPY_BINS), np.nan)
# KL_r_profile     = np.full((N_ENTROPY_SNAPS, N_ENTROPY_BINS), np.nan)

# TODO: implement shell entropy computation loop
# r_edges_shells = np.logspace(np.log10(r_min + 0.1), np.log10(r_max), N_ENTROPY_BINS + 1)
# r_mid_shells   = 0.5 * (r_edges_shells[:-1] + r_edges_shells[1:])
# for k, s in enumerate(snap_indices):
#     r_now = _traj_r[s]
#     for b in range(N_ENTROPY_BINS):
#         mask = (r_now >= r_edges_shells[b]) & (r_now < r_edges_shells[b+1])
#         if mask.sum() < MIN_SHELL_PARTS:
#             continue
#         H_r_profile[k, b], sigma_r_profile[k, b] = kde_entropy_1d(v_r[s, mask])
#         H_t_profile[k, b], _                     = kde_entropy_1d(v_t[s, mask])
#         H_phi_profile[k, b], _                   = kde_entropy_1d(v_phi[s, mask])
#         H_excess_r_prof[k, b] = h_excess(H_r_profile[k, b], sigma_r_profile[k, b])
#         dH_profile[k, b]      = H_r_profile[k, b] - H_t_profile[k, b]
#         KL_r_profile[k, b]    = kl_from_maxwellian(v_r[s, mask], KL_REF_SIGMA)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.6 — MULTI-MODALITY DETECTION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# A shell with TWO distinct velocity peaks is direct evidence of a tidal stream
# (or an undigested satellite) superimposed on the smooth halo background.
# Shannon entropy alone cannot distinguish a broad unimodal distribution from
# a bimodal one with the same entropy — you need to count the peaks.
#
# However, entropy and multi-modality are COMPLEMENTARY:
#   Single broad peak      → H high,  N_peaks = 1  →  hot/mixed
#   Single narrow peak     → H low,   N_peaks = 1  →  cold stream
#   Double peak            → H intermediate, N_peaks = 2  →  stream + halo
#   Many peaks             → H depends on widths,  N_peaks > 2  →  complex
#
# Detection algorithm:
#   1. Estimate the velocity PDF with KDE on a fine evaluation grid.
#   2. Find peaks using scipy.signal.find_peaks with:
#        height   = MULTIMODAL_PROM × max(f̂)   — ignore tiny bumps
#        distance = 0.5 σ_v / h_bandwidth       — peaks must be well-separated
#   3. A shell is "multi-modal" if N_peaks ≥ 2.
#
# HINT: the bandwidth of the KDE (Scott's rule) is σ × N^{-1/5}.
# For N = 30 and σ = 100 km/s: h ≈ 50 km/s.
# Two peaks separated by less than 2h cannot be resolved — report
# N_peaks = 1 in that case.  Use the `distance` parameter of find_peaks
# to enforce this minimum separation.
#
# TIME COMPLEXITY:  O(N_ENTROPY_SNAPS × N_ENTROPY_BINS × N_grid)  — fast
# SPACE COMPLEXITY: O(N_ENTROPY_SNAPS × N_ENTROPY_BINS)  — one int per bin

def count_kde_peaks(samples, n_grid=200, bandwidth=KDE_BANDWIDTH,
                    prominence=MULTIMODAL_PROM):
    """
    Count the number of significant peaks in the KDE of a 1D velocity sample.

    Parameters
    ----------
    samples    : (N,)  — velocity samples [km/s]
    n_grid     : int   — number of evaluation points for the KDE
    bandwidth  : str   — KDE bandwidth rule
    prominence : float — minimum peak height as fraction of global maximum

    Returns
    -------
    n_peaks  : int   — number of significant peaks
    peak_locs: array — locations of peaks in sample units [km/s]

    HINT: evaluate the KDE on a fine grid rather than at the sample points
    themselves — this avoids aliasing from the discrete sample positions.
    v_grid = np.linspace(samples.min() - 3σ, samples.max() + 3σ, n_grid)
    f_grid = kde(v_grid)

    HINT: the `distance` parameter of find_peaks is in GRID UNITS (indices),
    not in km/s.  Convert: min_sep_indices = int(0.5 × σ / dv × 1)
    where dv = v_grid[1] − v_grid[0].
    """
    samples = samples[np.isfinite(samples)]
    if len(samples) < MIN_SHELL_PARTS:
        return 0, np.array([])
    kde      = gaussian_kde(samples, bw_method=bandwidth)
    sigma    = float(np.std(samples))
    v_lo     = samples.min() - 3.0 * sigma
    v_hi     = samples.max() + 3.0 * sigma
    v_grid   = np.linspace(v_lo, v_hi, n_grid)
    f_grid   = kde(v_grid)
    dv       = v_grid[1] - v_grid[0]
    min_sep  = max(1, int(0.5 * sigma / (dv + 1e-10)))
    threshold = prominence * f_grid.max()
    peaks, _ = find_peaks(f_grid, height=threshold, distance=min_sep)
    return len(peaks), v_grid[peaks]

# TODO: compute multi-modality arrays
# n_peaks_r_arr   = np.zeros((N_ENTROPY_SNAPS, N_ENTROPY_BINS), dtype=int)
# n_peaks_t_arr   = np.zeros((N_ENTROPY_SNAPS, N_ENTROPY_BINS), dtype=int)
# f_multimodal_ts = np.full(N_ENTROPY_SNAPS, np.nan)
# for k, s in enumerate(snap_indices):
#     n_multi_k = 0
#     n_valid_k = 0
#     r_now = _traj_r[s]
#     for b in range(N_ENTROPY_BINS):
#         mask = (r_now >= r_edges_shells[b]) & (r_now < r_edges_shells[b + 1])
#         if mask.sum() < MIN_SHELL_PARTS:
#             continue
#         n_peaks_r_arr[k, b], _ = count_kde_peaks(v_r[s, mask])
#         n_peaks_t_arr[k, b], _ = count_kde_peaks(v_t[s, mask])
#         n_valid_k += 1
#         if n_peaks_r_arr[k, b] >= 2 or n_peaks_t_arr[k, b] >= 2:
#             n_multi_k += 1
#     if n_valid_k > 0:
#         f_multimodal_ts[k] = n_multi_k / n_valid_k

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.7 — GLOBAL ENTROPY AND GROUP-LEVEL ENTROPY                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# In addition to the radial profiles, track GLOBAL entropy (over all particles)
# and entropy per GROUP (inner/mid/outer/M31).
#
# H_global(t) = kde_entropy_1d over ALL particles' v_r at snapshot t.
# H_groups(g, t) = kde_entropy_1d over group-g particles' v_r at time t.
#
# H_global captures the bulk evolution of the whole system.
# H_groups reveals which component is driving the global signal:
#   If H_global rises but H_inner is flat → outer halo or M31 is being heated.
#   If H_M31 drops → M31 debris is forming a cold stream.
#
# Also compute the ENTROPY ANISOTROPY globally:
#   H_aniso_ts(t)  =  H_r_global(t) − H_t_global(t)
#
# This is the global version of ΔH(r, t) from §34.5.
# Compare to the mass-weighted mean of ΔH across shells — they should agree
# but shell outliers (narrow streams) inflate the mean relative to the global.
#
# TIME COMPLEXITY:  O(N_ENTROPY_SNAPS × N × log N)  — KDE
# SPACE COMPLEXITY: O(N_ENTROPY_SNAPS × 4)  — four groups + global

# TODO: compute global and group entropy time series
# GROUP_LABELS     = ['inner', 'mid', 'outer', 'M31']
# H_r_global_ts    = np.full(N_ENTROPY_SNAPS, np.nan)
# H_t_global_ts    = np.full(N_ENTROPY_SNAPS, np.nan)
# H_aniso_ts       = np.full(N_ENTROPY_SNAPS, np.nan)
# H_groups_r       = np.full((N_ENTROPY_SNAPS, 4), np.nan)
# KL_global_ts     = np.full(N_ENTROPY_SNAPS, np.nan)
# for k, s in enumerate(snap_indices):
#     H_r_global_ts[k], sigma_r_global = kde_entropy_1d(v_r[s])
#     H_t_global_ts[k], _              = kde_entropy_1d(v_t[s])
#     H_aniso_ts[k]  = H_r_global_ts[k] - H_t_global_ts[k]
#     KL_global_ts[k]= kl_from_maxwellian(v_r[s], KL_REF_SIGMA)
#     for g in range(4):
#         gmask = (_group == g) & np.isfinite(v_r[s])
#         H_groups_r[k, g], _ = kde_entropy_1d(v_r[s, gmask])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.8 — PRE-ALLOCATION FOR ALL OUTPUT ARRAYS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Summary of all arrays needed (most were allocated inline above):
#
#   Per-shell, per-epoch (shape: N_ENTROPY_SNAPS × N_ENTROPY_BINS):
#     H_r_profile      — radial velocity entropy per shell
#     H_t_profile      — tangential velocity entropy per shell
#     H_phi_profile    — azimuthal velocity entropy per shell
#     H_excess_r_prof  — excess entropy above Gaussian (H_r − H_Gauss(σ_r))
#     dH_profile       — entropy anisotropy: H_r − H_t
#     KL_r_profile     — KL divergence of v_r from reference Maxwellian
#     n_peaks_r_arr    — number of radial velocity PDF peaks per shell
#     n_peaks_t_arr    — number of tangential velocity PDF peaks per shell
#
#   Per-epoch scalars (shape: N_ENTROPY_SNAPS):
#     H_r_global_ts    — global radial velocity entropy
#     H_t_global_ts    — global tangential velocity entropy
#     H_aniso_ts       — global entropy anisotropy H_r − H_t
#     KL_global_ts     — global KL divergence from Maxwellian
#     f_multimodal_ts  — fraction of shells with multi-modal velocity PDF
#     JS_ts            — Jensen–Shannon divergence between consecutive epochs
#
#   Per-group, per-epoch (shape: N_ENTROPY_SNAPS × 4):
#     H_groups_r       — per-group radial velocity entropy
#
# SPACE COMPLEXITY TOTAL:
#   Per-shell arrays: O(N_ENTROPY_SNAPS × N_ENTROPY_BINS) × 8 arrays
#                   = 50 × 25 × 8 × 8 bytes = 80 KB
#   Per-epoch scalars: O(N_ENTROPY_SNAPS) × 6 = 50 × 6 × 8 = 2.4 KB
#   Per-group array:   50 × 4 × 8 bytes = 1.6 KB
#   TOTAL: ~ 100 KB — extremely lightweight

# TODO: allocate any arrays not yet allocated inline above

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.9 — FIGURES (NINE PLANNED)                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── Figure 1: Velocity PDFs at 5 epochs for each group ────────────────────────
# 4-row × 5-column panel grid.
# Each row = one particle group (inner, mid, outer, M31).
# Each column = one of the five profile epochs.
# Show f(v_r) estimated by KDE as a solid line.
# Overplot the reference Maxwellian N(0, σ_r) as a dashed line.
# Shade the area between them to visualise H_excess.
# HINT: use the SAME x-axis range [−600, +600] km/s for all panels so
# that differences in width are visible by eye.
# Label each panel with H_r and H_excess in the top-left corner.
#
# Expected output: section34_velocity_pdfs.png
#
# ── Figure 2: H_r(r, t) and ΔH(r, t) heatmaps ───────────────────────────────
# Left heatmap: H_r(r, t).  Colourmap: viridis (high = disordered).
# Right heatmap: ΔH(r, t) = H_r − H_t.  Diverging colourmap centred on 0.
# Both share the same x-axis (time) and y-axis (log r).
# HINT: overplot the pericentric passage times as vertical dashed lines —
# entropy dips should align with these.
#
# Expected output: section34_entropy_heatmaps.png
#
# ── Figure 3: H_excess(r, t) heatmap ─────────────────────────────────────────
# Heatmap of H_excess = H_r − H_Gauss(σ_r).
# This shows WHERE and WHEN the velocity distribution departs from Gaussian.
# H_excess < 0 everywhere at t=0 (regular orbits produce sub-Gaussian distributions).
# H_excess → 0 in the virialised core at late times.
# H_excess has a strong negative spike wherever a cold stream passes.
# HINT: choose the colour range carefully — H_excess is always negative or zero,
# so use a sequential colourmap with white at 0 (matplotlib's "RdPu_r" or
# "Blues_r" reversed so dark = more negative = more ordered).
#
# Expected output: section34_h_excess_heatmap.png
#
# ── Figure 4: KL divergence profile KL(r, t) ─────────────────────────────────
# Heatmap of KL_r_profile.  Log colourscale (KL spans orders of magnitude).
# This directly shows where the velocity distribution departs from the
# initial Maxwellian reference — the M31 debris trail should light up
# strongly during and after the pericentric passage.
# Overlay the stream fraction from §33 on a side panel for comparison.
#
# Expected output: section34_kl_profile.png
#
# ── Figure 5: Global entropy time series H_r(t), H_t(t), H_aniso(t) ──────────
# Three-panel figure:
#   Top:    H_r_global_ts and H_t_global_ts on the same axis.
#   Middle: H_aniso_ts = H_r − H_t (entropy anisotropy).
#   Bottom: KL_global_ts (KL divergence from initial Maxwellian).
# Mark pericentric passages with vertical lines on all panels.
# HINT: the SIGN of H_aniso oscillates during the merger — radial infall
# raises H_r above H_t; subsequent phase mixing levels them.
#
# Expected output: section34_global_entropy.png
#
# ── Figure 6: Group entropy H_groups_r(t) ────────────────────────────────────
# Four lines on one panel (one per group), showing H_r vs. time.
# The M31 line should show the most dramatic evolution.
# The inner-halo line should be nearly flat (fast mixing → always near max H).
# HINT: normalise each line by its t=0 value so that all groups start at 1
# and you see RELATIVE evolution rather than absolute levels (which differ
# because the groups have different σ_v at t=0).
#
# Expected output: section34_group_entropy.png
#
# ── Figure 7: Jensen–Shannon divergence JS(t) ────────────────────────────────
# Line plot of JS_ts vs. time (consecutive-epoch velocity change rate).
# Overlay dS_mix/dt from §33 on the right-hand y-axis.
# HYPOTHESIS: JS peaks LEAD dS_mix/dt peaks — velocity redistribution
# drives spatial mixing, not the other way round.
# Also show JS computed separately for the v_r, v_t, v_phi components
# to identify WHICH velocity component is driving the change.
#
# Expected output: section34_js_divergence.png
#
# ── Figure 8: Multi-modality fraction f_multimodal(t) ────────────────────────
# Line plot of f_multimodal_ts vs. time.
# Overlay f_stream_ts from §33 on the same axis.
# HYPOTHESIS: f_multimodal and f_stream should peak at the same time —
# both are detecting the same tidal debris, one in velocity space and one
# in phase space.
# Also show a 2D map: N_peaks_r(r, t) as a colour heatmap to show
# which shells are multi-modal and when.
#
# Expected output: section34_multimodal_fraction.png
#
# ── Figure 9: Master summary panel ───────────────────────────────────────────
# 2×2 grid:
#   (0,0) H_r(r, t) entropy heatmap
#   (0,1) ΔH(r, t) entropy anisotropy heatmap
#   (1,0) Global entropy H_r(t), H_t(t), H_aniso(t) time series
#   (1,1) Four-way comparison: H_aniso(t) / JS(t) / β_mean(t) / f_stream(t)
#         all normalised to [0, 1] range — direct visual test of whether
#         entropy, anisotropy, and stream diagnostics are telling the same story
#
# Expected output: section34_summary_panel.png

# TODO: implement all nine figures
# HINT: follow the exact same structure as Sections 21–33:
#   fig, ax = plt.subplots(...)
#   _ax(ax, xlabel=..., ylabel=..., title=..., log_x=..., log_y=...)
#   ... plotting code ...
#   fig.savefig(os.path.join(OUT_DIR, "section34_xxx.png"), ...)
#   plt.close(fig)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.10 — ANIMATION: VELOCITY DISTRIBUTION EVOLUTION                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Three-panel animation:
#
# Left  : Velocity PDF f(v_r) of all particles at the current epoch,
#         estimated by KDE and plotted as a filled curve.
#         Overplot the reference Maxwellian (dashed) and the t=0 KDE (dotted).
#         The filled area highlights the difference from the reference.
#         Annotate: current H_r, H_excess, KL in a text box.
#
# Centre: Entropy anisotropy heatmap H_r(r) − H_t(r) as a horizontal bar chart
#         (one bar per shell).  Bars to the right are radially disordered;
#         bars to the left are tangentially disordered.
#         Colour: diverging map centred on 0 (red = radial, blue = tangential).
#         Overplot the β profile from §33 as a line for comparison.
#
# Right : Running time series of H_r_global(t) and H_aniso_ts(t), with a
#         vertical line at the current frame.  Growing shaded region shows
#         history.
#
# HINT: For the PDF panel, use ax.fill_between with alpha=0.3.
# Update per frame: clear and redraw the KDE, or use ax.lines[0].set_ydata().
# The KDE changes shape each frame so set_ydata() is easier than cla().
#
# HINT: For the bar chart, update bar heights using:
#   for bar, new_h in zip(bar_container, dH_profile[k]):
#       bar.set_height(new_h)   # NOTE: this only works for vertical bars
#   For horizontal bars use bar.set_width(new_h).
#
# Expected output: section34_animation_entropy.mp4

# TODO: implement animation

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §34.11 — SECTION COMPLETE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Print the output manifest — same pattern as all previous sections.
# Also print a summary statistics table:
#
#   Group  | H_r(t=0) | H_r(t=fin) | ΔH_mean | KL_mean | f_multi_peak
#   ────────────────────────────────────────────────────────────────────
#   Inner  | …        | …          | …       | …       | …
#   Mid    | …        | …          | …       | …       | …
#   Outer  | …        | …          | …       | …       | …
#   M31    | …        | …          | …       | …       | …
#
# Also print the three cross-section Pearson correlations that close the
# §31–34 diagnostic suite:
#   corr(H_aniso, β)        — entropy anisotropy vs. Binney β (§33)
#   corr(KL_global, S_mix)  — velocity disorder vs. phase-space entropy (§33)
#   corr(JS, dS_mix/dt)     — velocity change rate vs. mixing rate (§33)
# Perfect agreement → corr = 1.  Discrepancies are scientifically interesting.

outputs_34 = [
    "section34_velocity_pdfs.png",
    "section34_entropy_heatmaps.png",
    "section34_h_excess_heatmap.png",
    "section34_kl_profile.png",
    "section34_global_entropy.png",
    "section34_group_entropy.png",
    "section34_js_divergence.png",
    "section34_multimodal_fraction.png",
    "section34_animation_entropy.mp4",
    "section34_summary_panel.png",
]

# TODO: implement output manifest printing
