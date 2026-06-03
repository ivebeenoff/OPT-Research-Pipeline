"""
10.5 hours of work, put that on top

===============================================================================
SECTION 33 — PHASE-SPACE DENSITY ANALYSIS & DISTRIBUTION FUNCTION EVOLUTION
===============================================================================
Author  : Abhinav Vatsa  [SCAFFOLD — fill in implementation]

This section is a guided scaffold.  Every subsection contains:
  • Physical motivation and context
  • The exact quantity to compute and its formula
  • A suggested implementation strategy
  • Time and space complexity analysis
  • Hints on numerical pitfalls to avoid
  • Expected output description

The companion sections (31, 32) characterised chaos via trajectory divergence
and spectral complexity.  This section asks a complementary question:

  HOW IS PHASE-SPACE VOLUME REDISTRIBUTED AS THE MERGER PROCEEDS?

Physical connection to Sections 31 and 32
──────────────────────────────────────────
Liouville's theorem states that, for a collisionless self-gravitating system,
the FINE-GRAINED phase-space density f(x, v, t) is conserved along any
particle trajectory:

    Df/Dt  =  ∂f/∂t + v · ∇f + F · ∂f/∂v  =  0

However, this is the fine-grained density — defined on infinitesimally small
phase-space volumes.  The COARSE-GRAINED density f̄ (averaged over a finite
volume) can only DECREASE or stay constant:

    Df̄/Dt  ≤  0     (coarse-grained mixing)

This decrease of f̄ is the hallmark of PHASE-SPACE MIXING: fine-grained
structure (streams, shells, caustics) forms on ever-smaller scales, but
the locally-averaged density smooths out.

Connection to chaos (§31) and frequency analysis (§32):
  • Chaotic orbits (λ > 0) explore an ergodic torus in phase space —
    their phase-space density diffuses outward rapidly.
  • Regular orbits stay on invariant tori — f̄ is conserved locally.
  • Frequency drift (§32) signals the torus is being deformed —
    phase-space volume is being reshuffled.

So the three sections are measuring the SAME underlying dynamics at three
levels: trajectory (§31), spectral (§32), and phase-space (§33).

Key quantities computed in this section
────────────────────────────────────────
  f(x, v, t)        — coarse-grained phase-space density [M_sun / kpc³ (km/s)³]
  Q(i, t)           — phase-space density estimator for particle i at time t
  Q_0(i)            — initial phase-space density (t=0 reference)
  ΔQ(i, t)          — fractional change in local density: (Q − Q_0) / Q_0
  S_mix(t)          — global mixing entropy: −∫ f ln f d³x d³v
  f_mixed(t)        — fraction of particles whose Q has decreased by > MIX_THRESH
  τ_mix(i)          — mixing timescale: time for Q(i) to fall to Q_0/e
  D_PS(i)           — phase-space diffusion coefficient [kpc² (km/s)² / Gyr]
  β_anis(r, t)      — velocity anisotropy parameter: 1 − σ²_t / (2 σ²_r)
  f_stream(t)       — fraction of particles in coherent streams (Q > Q_stream)

Dependencies
────────────
  scipy.spatial     — KDTree for k-nearest-neighbour density estimation
  scipy.stats       — kernel density estimation (fallback)
  sklearn (optional)— BallTree for faster KNN in 6D (if available)
  Section 26        — traj_pos, traj_vel, traj_r (Lagrangian trajectories)
  Section 31        — lambda_total (for correlation plots)
  Section 32        — complexity_arr, sigma_drift_arr (for tri-method comparison)

All globals from the parent pipeline are inherited.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: The most important parameter here is K_NEIGHBOURS — the number of
# nearest neighbours used for the KNN phase-space density estimator.
# The local density is estimated as:
#
#   Q(i) = (K − 1) / V_K(i)
#
# where V_K(i) is the 6D phase-space volume of the hypersphere enclosing
# the K nearest neighbours of particle i.
#
# Too small K → very noisy estimate (high variance).
# Too large K → over-smoothed, misses real structure (high bias).
# Rule of thumb: K ≈ 20–64 for N ~ 10^4 particles.
#
# HINT: Phase-space coordinates must be DIMENSIONALLY CONSISTENT.
# x, y, z are in kpc; vx, vy, vz are in km/s.
# These have wildly different scales — a 1 kpc displacement is NOT the same
# as a 1 km/s velocity offset in terms of phase-space topology.
# Normalise before computing distances:
#   x_norm  = x / σ_x    — divide each spatial coordinate by its global std dev
#   v_norm  = v / σ_v    — divide each velocity coordinate by its global std dev
# This ensures the phase-space metric is isotropic.
# Alternatively, use a PHYSICAL scaling: σ_x = R_vir, σ_v = v_c.
#
# HINT: The full 6D KNN for 600 particles × 800 snapshots is expensive.
# Subsample to N_PSD_PARTICLES × N_PSD_SNAPS strategically.

K_NEIGHBOURS     = 32        # k for k-nearest-neighbour density estimator
N_PSD_PARTICLES  = 500       # number of particles to track in phase space
N_PSD_SNAPS      = 40        # number of snapshot epochs to evaluate Q (evenly spaced)
MIX_THRESH       = 0.5       # Q drops below Q_0 × (1 − MIX_THRESH) → "mixed"
STREAM_THRESH    = 3.0       # Q > STREAM_THRESH × Q_median → "stream particle"
DIFF_LAG         = 5         # snapshot lag for computing phase-space displacement
V_SCALE          = 1.0       # km/s → kpc conversion for phase-space metric
                              # use V_SCALE = σ_v / σ_x  (computed dynamically below)

# Radial binning for anisotropy profiles.
N_ANISO_BINS     = 20        # number of radial bins for β(r) profile

# Animation.
ANIM_FPS_33      = 18
ANIM_DPI_33      = 100
ANIM_BITRATE_33  = 1600

print("\n" + "="*80)
print("  SECTION 33 · Phase-Space Density Analysis & Distribution Function")
print("="*80)
print(f"  KNN k            : {K_NEIGHBOURS}")
print(f"  Particles        : {N_PSD_PARTICLES}")
print(f"  Snapshot epochs  : {N_PSD_SNAPS}")
print(f"  Mix threshold    : {MIX_THRESH:.0%} drop in Q")
print(f"  Stream threshold : {STREAM_THRESH:.1f} × median Q")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.1 — LOAD TRAJECTORIES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: Reuse traj_pos and traj_vel from Section 26.  Copy the try/except
# fallback block used in Sections 31 and 32.
#
# The six-dimensional phase-space coordinates you need per particle are:
#   w(t, i) = [x, y, z, vx, vy, vz]   shape: (ns, N, 6)
#
# IMPORTANT: you do NOT need all ns snapshots for the density computation.
# Evaluate at N_PSD_SNAPS evenly spaced epochs to save compute:
#   snap_indices = np.linspace(0, ns-1, N_PSD_SNAPS, dtype=int)
#
# You WILL need all snapshots for the mixing entropy S_mix(t) time series
# (§33.5), which is cheaper because it uses pre-computed Q values rather
# than rerunning KNN.
#
# The normalisation scale (for isotropic metric) should be computed from
# the full population at t=0 to give a consistent baseline:
#   sigma_x = np.std(traj_pos[0, :, :3])  — global spatial std at t=0
#   sigma_v = np.std(traj_vel[0, :, :3])  — global velocity std at t=0
#   V_SCALE  = sigma_v / sigma_x          — converts km/s → kpc in metric
#
# TIME COMPLEXITY:  O(N_PSD_PARTICLES × N_PSD_SNAPS × 6)  for loading
# SPACE COMPLEXITY: O(N_PSD_SNAPS × N_PSD_PARTICLES × 6)
#                 ≈ 40 × 500 × 6 × 8 bytes ≈ 1 MB

# TODO: implement trajectory loading / inheritance
_traj_pos  = None   # replace with inherited or recomputed (ns, N, 3)
_traj_vel  = None   # (ns, N, 3)
_traj_r    = None   # (ns, N)
_r0        = None   # (N,) initial radii
_group     = None   # (N,) 0=inner,1=mid,2=outer,3=M31
_N         = 0

# TODO: subsample snapshots and particles, build 6D phase-space array
# snap_indices = np.linspace(0, ns - 1, N_PSD_SNAPS, dtype=int)
# sigma_x = np.std(_traj_pos[0, :N_PSD_PARTICLES, :])
# sigma_v = np.std(_traj_vel[0, :N_PSD_PARTICLES, :])
# V_SCALE  = sigma_v / sigma_x
#
# w_6d = np.empty((N_PSD_SNAPS, N_PSD_PARTICLES, 6))
# for k, s in enumerate(snap_indices):
#     w_6d[k, :, :3] = _traj_pos[s, :N_PSD_PARTICLES, :]
#     w_6d[k, :, 3:] = _traj_vel[s, :N_PSD_PARTICLES, :] / V_SCALE

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.2 — KNN PHASE-SPACE DENSITY ESTIMATOR                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# For a set of N tracers drawn from a distribution f(w),
# the k-nearest-neighbour estimator of the local phase-space density at w_i is:
#
#   Q_i  =  (K − 1) / [ N × V_d(r_K) ]
#
# where r_K is the distance to the K-th nearest neighbour of particle i,
# and V_d(r) is the volume of a d-dimensional sphere of radius r:
#
#   V_d(r)  =  π^(d/2) / Γ(d/2 + 1) × r^d
#
# For d = 6 (3 positions + 3 velocities):
#   V_6(r)  =  π³ / 6 × r⁶
#
# The KNN estimator is unbiased and consistent as K, N → ∞ with K/N → 0.
# It naturally adapts to the local density: in dense regions r_K is small,
# in voids r_K is large.
#
# HINT: use scipy.cKDTree for the KNN search.  The tree build is O(N log N)
# and each query is O(K log N).  For N = 500 and K = 32:
#   tree = cKDTree(w_6d[k])            — build tree in 6D
#   dists, _ = tree.query(w_6d[k], k=K+1)  — self-inclusive, so take k=K+1
#   r_K = dists[:, K]                  — distance to the K-th neighbour
#   Q = (K - 1) / (N * V6 * r_K**6)   — density estimate
#
# HINT: in practice r_K can be exactly zero for duplicate positions
# (common at early times if initial conditions have coincident particles).
# Add a tiny floor: r_K = np.maximum(r_K, 1e-10) before raising to power 6.
#
# HINT: the resulting Q values span many orders of magnitude.
# Work in log10(Q) for visualisation and most diagnostics.
# The physically meaningful quantity is the RELATIVE change:
#   ΔQ(i, t) = (Q(i,t) − Q(i,0)) / Q(i,0)
#
# TIME COMPLEXITY:  O(N_PSD_SNAPS × N × log N × K)
#                 = O(40 × 500 × 9 × 32)  ≈ 6 × 10^6 — fast, ~seconds
# SPACE COMPLEXITY: O(N_PSD_SNAPS × N)  ≈ 40 × 500 × 8 bytes = 160 KB

_V6_PREFACTOR = np.pi**3 / 6.0   # volume prefactor for 6D unit sphere

def knn_phase_density(w, k=K_NEIGHBOURS):
    """
    Estimate the local 6D phase-space density for each particle using
    the k-nearest-neighbour method.

    Parameters
    ----------
    w : (N, 6)  — normalised phase-space coordinates (positions in kpc,
                  velocities already divided by V_SCALE)
    k : int     — number of nearest neighbours

    Returns
    -------
    Q : (N,)    — phase-space density estimate [kpc^{-6}  (normalised units)]

    HINT: if sklearn is available, BallTree with metric='euclidean' is
    ~3× faster than cKDTree for d=6 because it avoids the kd-tree splitting
    pathology in high dimensions.  Fall back to cKDTree otherwise.
    """
    N  = len(w)
    try:
        from sklearn.neighbors import BallTree
        tree  = BallTree(w, metric='euclidean')
        dists, _ = tree.query(w, k=k + 1)    # +1 to exclude self
        r_K   = dists[:, k]
    except ImportError:
        tree  = cKDTree(w)
        dists, _ = tree.query(w, k=k + 1)
        r_K   = dists[:, k]

    r_K = np.maximum(r_K, 1e-10)             # guard against zero distances
    Q   = (k - 1) / (N * _V6_PREFACTOR * r_K**6)
    return Q

# TODO: compute Q for every snapshot epoch and store in Q_arr
# Q_arr = np.full((N_PSD_SNAPS, N_PSD_PARTICLES), np.nan)
# for k, s in enumerate(snap_indices):
#     print(f"  Computing Q at snap {s} ({k+1}/{N_PSD_SNAPS})...", end='\r')
#     Q_arr[k] = knn_phase_density(w_6d[k])
#
# Q0 = Q_arr[0]   # reference density at t=0
# delta_Q = (Q_arr - Q0[np.newaxis, :]) / (Q0[np.newaxis, :] + 1e-30)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.3 — PHASE-SPACE DIFFUSION COEFFICIENT                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# For a particle undergoing DIFFUSION in phase space, its mean-square
# displacement in the 6D phase-space grows linearly with time:
#
#   <|Δw(t)|²>  =  6 D_PS × t
#
# where D_PS is the phase-space diffusion coefficient.
# The factor 6 comes from the 6 independent degrees of freedom.
#
# For a CHAOTIC orbit, D_PS > 0 and grows as the tidal perturbation grows.
# For a REGULAR orbit, D_PS ≈ 0: the particle stays near its invariant torus.
#
# This quantity bridges Section 31 (trajectory divergence) and the present
# section: D_PS measures how rapidly a single particle diffuses through phase
# space, rather than how rapidly TWO nearby particles diverge.
#
# Implementation
# ──────────────
# For each particle i, compute the mean-square phase-space displacement
# at a series of lag times τ = DIFF_LAG, 2×DIFF_LAG, ...:
#
#   MSD_i(τ) = |w(t_0 + τ, i) − w(t_0, i)|²
#
# Then fit a line MSD_i = 6 D_PS,i × τ to get D_PS per particle.
#
# HINT: use MULTIPLE reference times t_0 and average over them to reduce
# the effect of the initial condition:
#   MSD_i(τ) = mean over t_0 of |w(t_0 + τ, i) − w(t_0, i)|²
#
# HINT: phase-space displacement must use the SAME normalised coordinates as
# the KNN estimator to be dimensionally consistent.
# |Δw|² = |Δx|² + |Δv / V_SCALE|²
#
# HINT: for particles that leave the simulation domain (NaN positions),
# skip those lag pairs.  Clipping to finite values only will bias the MSD
# downward — use np.isfinite masks.
#
# TIME COMPLEXITY:  O(N × n_lags × N_PSD_SNAPS) — all array operations
# SPACE COMPLEXITY: O(N × n_lags)  — MSD per particle per lag

def compute_msd(w_series, lag_steps):
    """
    Compute the mean-square phase-space displacement over a series of lag steps.

    Parameters
    ----------
    w_series   : (T, 6)  — normalised phase-space trajectory for one particle
    lag_steps  : array   — lag indices (in units of snapshot spacing)

    Returns
    -------
    msd : (len(lag_steps),)  — MSD at each lag [normalised kpc²]

    HINT: average over all valid pairs (t_0, t_0 + τ) for each lag τ.
    HINT: exclude pairs where either endpoint is NaN.
    """
    T   = len(w_series)
    msd = np.full(len(lag_steps), np.nan)
    for j, lag in enumerate(lag_steps):
        diffs = []
        for t0 in range(T - lag):
            w0  = w_series[t0]
            w1  = w_series[t0 + lag]
            if not (np.all(np.isfinite(w0)) and np.all(np.isfinite(w1))):
                continue
            diffs.append(np.sum((w1 - w0)**2))
        if len(diffs) > 2:
            msd[j] = np.mean(diffs)
    return msd

# TODO: compute D_PS for every particle by fitting MSD(τ) vs. τ
# lag_steps  = np.arange(1, N_PSD_SNAPS // 3, DIFF_LAG)
# D_PS_arr   = np.full(N_PSD_PARTICLES, np.nan)
# for i in range(N_PSD_PARTICLES):
#     msd = compute_msd(w_6d[:, i, :], lag_steps)
#     finite = np.isfinite(msd)
#     if finite.sum() >= 3:
#         # linear fit: msd = 6 * D_PS * tau
#         tau = lag_steps[finite] * dt_snap    # convert to Gyr
#         D_PS_arr[i] = np.polyfit(tau, msd[finite], 1)[0] / 6.0

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.4 — COARSE-GRAINED MIXING ENTROPY                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The Boltzmann mixing entropy of the coarse-grained distribution function is:
#
#   S_mix(t) = −∫ f̄(w, t) ln f̄(w, t) d⁶w
#            ≈ −(1/N) Σ_i ln Q_i(t)   [in natural units]
#
# where the approximation holds because Q_i is proportional to f at particle i.
# (This is the standard "Gibbs entropy" approximation for a KDE.)
#
# Physical interpretation:
#   S_mix INCREASES as the system mixes — phase-space density spreads out.
#   A perfectly unmixed system (all particles at one point) has S → −∞.
#   A maximally mixed system (uniform distribution) has S → ln(Volume).
#
# In a real N-body system:
#   • S_mix grows rapidly during the first pericentric passage
#   • S_mix levels off when mixing is complete (ergodic limit)
#   • Oscillations in S_mix signal coherent phase-space structures
#     (tidal streams, shells) temporarily re-assembling
#
# Connection to Section 32: compare S_mix(t) to f_regular(t).
# HYPOTHESIS: S_mix peaks are anti-correlated with f_regular dips —
# when mixing is most active, the fraction of regular orbits is lowest.
#
# TIME COMPLEXITY:  O(N_PSD_SNAPS × N)  — trivial, just log and mean
# SPACE COMPLEXITY: O(N_PSD_SNAPS)  — one scalar per snapshot

def mixing_entropy(Q_at_snap):
    """
    Compute the Boltzmann mixing entropy at a single snapshot.

    Parameters
    ----------
    Q_at_snap : (N,)  — phase-space density estimates at one time

    Returns
    -------
    S : float  — mixing entropy [nats]

    HINT: clip Q to a small positive value before log to avoid −∞.
    HINT: normalise Q so that sum(Q) = 1 before computing entropy,
          to remove the effect of the total mass normalisation.
    HINT: alternatively, compute S as the CHANGE from t=0:
          ΔS(t) = S(t) − S(0)  — this removes constant offsets
          and makes the t=0 baseline exactly zero.
    """
    q = np.clip(Q_at_snap, 1e-30, None)
    q = q / q.sum()                          # normalise to a probability
    return float(-np.sum(q * np.log(q)))

# TODO: compute S_mix_ts over all PSD snapshot epochs
# S_mix_ts = np.array([mixing_entropy(Q_arr[k]) for k in range(N_PSD_SNAPS)])
# dS_mix   = S_mix_ts - S_mix_ts[0]   # change relative to t=0

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.5 — STREAM DETECTION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# During a merger, stripped stars form TIDAL STREAMS — coherent, filamentary
# structures in phase space.  Streams are characterised by:
#   • HIGH local phase-space density (particles remain close in 6D)
#   • LOW configuration-space density (streams are spatially extended)
#   • Nearly constant energy and angular momentum along the stream
#
# Streams survive for many Gyr in a static potential because their constituent
# particles have almost identical orbital frequencies (they were stripped from
# the same progenitor at the same time).  When the merger is violent, streams
# are disrupted and phase-space density is randomised.
#
# Stream detection criterion (simplest approach):
#   A particle i is "in a stream" at time t if:
#     Q(i, t) > STREAM_THRESH × Q_median(t)
#   AND
#     the density of its position-only 3D neighbourhood is NOT high
#     (i.e. it is not in a dense gravitational clump)
#
# A more robust approach: compare the 6D KNN radius r_K to the 3D KNN radius.
#   If r_K^(position) << r_K^(full 6D), the particle is in a dense
#   velocity-space clump but a sparse position-space region — a stream.
#
# KEY OUTPUT: f_stream(t) — the fraction of OUTER particles that are in streams.
# This should PEAK after the first pericentric passage (streams form)
# and DECLINE at late times (streams phase-mix away).
#
# TIME COMPLEXITY:  O(N_PSD_SNAPS × N log N)  — two KNN trees per snapshot
# SPACE COMPLEXITY: O(N_PSD_SNAPS × N)  — stream mask per snapshot

# TODO: implement stream detection
# stream_mask = np.full((N_PSD_SNAPS, N_PSD_PARTICLES), False)
# r3d_K_arr   = np.full((N_PSD_SNAPS, N_PSD_PARTICLES), np.nan)  # 3D KNN radius
# r6d_K_arr   = np.full((N_PSD_SNAPS, N_PSD_PARTICLES), np.nan)  # 6D KNN radius
# for k, s in enumerate(snap_indices):
#     Q_median_k   = np.nanmedian(Q_arr[k])
#     high_Q_mask  = Q_arr[k] > STREAM_THRESH * Q_median_k
#
#     # compute 3D KNN radius for position-space density
#     tree3d = cKDTree(_traj_pos[s, :N_PSD_PARTICLES, :])
#     d3, _  = tree3d.query(_traj_pos[s, :N_PSD_PARTICLES, :], k=K_NEIGHBOURS+1)
#     r3d_K_arr[k] = d3[:, K_NEIGHBOURS]
#
#     # stream = high phase-space density AND NOT a position-space clump
#     pos_clump_mask = r3d_K_arr[k] < np.percentile(r3d_K_arr[k], 20)
#     stream_mask[k] = high_Q_mask & ~pos_clump_mask
#
# f_stream_ts = stream_mask.mean(axis=1)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.6 — VELOCITY ANISOTROPY PROFILE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The Binney anisotropy parameter β characterises the orbital structure:
#
#   β(r) = 1 − σ²_t(r) / (2 σ²_r(r))
#
# where σ_r is the radial velocity dispersion and σ_t = σ_θ = σ_φ is the
# tangential velocity dispersion in the radial-tangential decomposition.
#
# β = 0   : isotropic orbits (random directions, e.g. a pressure-supported core)
# β = 1   : purely radial orbits (e.g. immediately after a merger, infalling)
# β = −∞  : purely circular orbits (e.g. a cold rotating disc)
#
# In a violent merger we expect:
#   • β → +1 during infall (radial streaming)
#   • β oscillating around 0 as the system virialises
#   • β stabilising at β ≈ 0.1–0.3 in the relaxed halo (Mamon+ 2013)
#
# Computing β(r, t)
# ──────────────────
# 1. Bin particles into N_ANISO_BINS logarithmic radial bins.
# 2. For each bin, compute σ_r and σ_t:
#      v_r  = (v · r̂) r̂  — radial component
#      v_t  = v − v_r     — tangential component
#      σ_r² = var(|v_r|)
#      σ_t² = var(|v_t|) / 2  — divide by 2 for two tangential DOF
# 3. β = 1 − σ_t² / σ_r²
#
# HINT: σ_r² and σ_t² should be computed as the variance of the VELOCITY
# components in the spherical basis (r, θ, φ), not just magnitudes.
# Compute the full tensor if you have the time — it reveals velocity ellipsoids.
#
# HINT: bins with fewer than MIN_PART_BIN = 5 particles should be masked out.
#
# TIME COMPLEXITY:  O(N_PSD_SNAPS × N)  — fast vectorised binning
# SPACE COMPLEXITY: O(N_PSD_SNAPS × N_ANISO_BINS)  — one β per bin per snap

MIN_PART_BIN = 5   # minimum particles per radial bin for valid β estimate

def velocity_anisotropy(pos, vel, r_edges):
    """
    Compute the velocity anisotropy profile β(r) at a single snapshot.

    Parameters
    ----------
    pos     : (N, 3)  — particle positions [kpc]
    vel     : (N, 3)  — particle velocities [km/s]
    r_edges : (nb+1,) — radial bin edges [kpc]

    Returns
    -------
    beta    : (nb,)   — anisotropy parameter per radial bin
    r_mid   : (nb,)   — bin midpoints [kpc]

    HINT: for the radial unit vector r̂ = pos / |pos|, watch for particles
    at r ≈ 0 which give r̂ = NaN.  Add a small floor to |pos|.
    HINT: the tangential velocity vector is v_t = v − (v · r̂) r̂.
         Its magnitude squared is |v_t|² = |v|² − (v · r̂)².
    """
    r     = np.linalg.norm(pos, axis=1)
    r     = np.maximum(r, 1e-10)
    r_hat = pos / r[:, np.newaxis]

    v_r   = np.sum(vel * r_hat, axis=1)          # radial velocity component
    v_t2  = np.sum(vel**2, axis=1) - v_r**2      # tangential speed squared
    v_t2  = np.maximum(v_t2, 0.0)                # numerical floor

    nb     = len(r_edges) - 1
    r_mid  = 0.5 * (r_edges[:-1] + r_edges[1:])
    beta   = np.full(nb, np.nan)

    for b in range(nb):
        mask = (r >= r_edges[b]) & (r < r_edges[b + 1])
        if mask.sum() < MIN_PART_BIN:
            continue
        sig_r2 = np.var(v_r[mask])
        sig_t2 = np.mean(v_t2[mask]) / 2.0      # per tangential DOF
        if sig_r2 < 1e-20:
            beta[b] = np.nan
        else:
            beta[b] = 1.0 - sig_t2 / sig_r2

    return beta, r_mid

# TODO: compute β(r, t) for all snapshot epochs
# r_edges = np.logspace(np.log10(r_min), np.log10(r_max), N_ANISO_BINS + 1)
# beta_profile = np.full((N_PSD_SNAPS, N_ANISO_BINS), np.nan)
# for k, s in enumerate(snap_indices):
#     beta_profile[k], r_mid_aniso = velocity_anisotropy(
#         _traj_pos[s, :N_PSD_PARTICLES],
#         _traj_vel[s, :N_PSD_PARTICLES],
#         r_edges)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.7 — MIXING TIMESCALE                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The mixing timescale τ_mix for particle i is defined as the time at which
# its local phase-space density Q has dropped to 1/e of its initial value:
#
#   Q(i, τ_mix) = Q(i, 0) / e
#
# or equivalently, ΔQ(i, τ_mix) = (Q_0 − Q) / Q_0 ≥ 1 − 1/e ≈ 0.632.
#
# τ_mix is the phase-space analogue of the Lyapunov time (1/λ) from §31:
#   Short τ_mix → fast mixing → chaotic orbit
#   Long τ_mix  → slow mixing → regular orbit
#   τ_mix = ∞   → Q never drops — this particle's local density actually
#                 INCREASES (it falls into a denser region — tidal compression)
#
# IMPORTANT SUBTLETY: A particle can have increasing Q without being "regular".
# If it falls into the main halo core, its neighbours become more numerous
# (the core gets denser) and Q goes up — but it may still be on a chaotic orbit.
# Use D_PS (§33.3) to distinguish: high D_PS + high Q_increase = chaotic
# but compressed; low D_PS + high Q_increase = regular infall.
#
# Implementation
# ──────────────
# For each particle, iterate over the Q_arr time series and find the first
# crossing of the Q_0/e threshold.  Interpolate between snapshots.
#
# TIME COMPLEXITY:  O(N_PSD_SNAPS × N)  — one pass per particle
# SPACE COMPLEXITY: O(N)  — one scalar per particle

# TODO: compute mixing timescales
# tau_mix_arr = np.full(N_PSD_PARTICLES, np.inf)   # ∞ means never dropped to Q_0/e
# threshold   = 1.0 / np.e
# for i in range(N_PSD_PARTICLES):
#     q_rel = Q_arr[:, i] / (Q0[i] + 1e-30)        # Q(t) / Q(0)
#     for k in range(1, N_PSD_SNAPS):
#         if q_rel[k] <= threshold:
#             # linear interpolation between k-1 and k
#             frac = (threshold - q_rel[k-1]) / (q_rel[k] - q_rel[k-1] + 1e-30)
#             t_lo = snap_times[k - 1]
#             t_hi = snap_times[k]
#             tau_mix_arr[i] = t_lo + frac * (t_hi - t_lo)
#             break

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.8 — PRE-ALLOCATION FOR ALL OUTPUT ARRAYS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: allocate all arrays here before the main computation to make memory
# usage explicit and to avoid silent zero-fills masking computation failures.
# Use np.full(..., np.nan) for floats and np.full(..., False) for booleans.
#
# Arrays needed:
#
#   Per-particle scalars (shape: N_PSD_PARTICLES):
#     Q0               — initial phase-space density
#     D_PS_arr         — phase-space diffusion coefficient [normalised kpc² / Gyr]
#     tau_mix_arr      — mixing timescale [Gyr] (∞ if never mixed)
#     complexity_psd   — (optional) spectral complexity, inherited from §32
#
#   Time-resolved, per-particle (shape: N_PSD_SNAPS × N_PSD_PARTICLES):
#     Q_arr            — phase-space density at each epoch
#     delta_Q          — fractional change: (Q − Q0) / Q0
#     stream_mask      — boolean: particle in a stream
#
#   Time-resolved, per-radial-bin (shape: N_PSD_SNAPS × N_ANISO_BINS):
#     beta_profile     — velocity anisotropy β(r, t)
#
#   Time series scalars (shape: N_PSD_SNAPS):
#     S_mix_ts         — mixing entropy at each epoch
#     f_stream_ts      — stream fraction at each epoch
#     f_mixed_ts       — fraction of particles with ΔQ < −MIX_THRESH
#
# SPACE COMPLEXITY TOTAL:
#   Per-particle scalars:  O(N) ≈ 500 × 8 bytes = 4 KB
#   Time-resolved 2D:      O(N_PSD_SNAPS × N) ≈ 40 × 500 × 8 bytes = 160 KB per array
#   Anisotropy:            O(N_PSD_SNAPS × N_ANISO_BINS) ≈ 40 × 20 × 8 bytes = 6 KB
#   Time series:           O(N_PSD_SNAPS) ≈ 40 × 8 bytes = 320 bytes
#   TOTAL: ~ a few MB — lightweight

# TODO: allocate all arrays here

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.9 — MAIN COMPUTATION LOOP                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Structure of the loop
# ─────────────────────
# Unlike §32 (outer loop over particles), here the natural structure is
# OUTER LOOP OVER SNAPSHOTS because the KNN density estimator requires the
# FULL population at each time step — you cannot analyse one particle in
# isolation.
#
# OPTION A: single outer loop over snapshots.
#   for k, s in enumerate(snap_indices):
#       Q_arr[k]      = knn_phase_density(w_6d[k])
#       S_mix_ts[k]   = mixing_entropy(Q_arr[k])
#       stream_mask[k], f_stream_ts[k] = detect_streams(w_6d[k], Q_arr[k])
#       beta_profile[k], _ = velocity_anisotropy(...)
#   Then a second pass over particles for D_PS and τ_mix.
#
# OPTION B: loop over snapshots for Q, then loop over particles for MSD/τ.
#   This is cleaner for memory: you accumulate w_6d[:, i, :] per particle
#   after the first loop.
#
# RECOMMENDATION: use Option B.  The KNN loop is the expensive part and
# benefits most from batching all particles together in each snapshot.
#
# Numerical pitfalls
# ──────────────────
# 1. Phase-space metric: ensure ALL coordinates are normalised BEFORE
#    calling knn_phase_density.  A single unscaled velocity axis will
#    dominate the distance metric and produce nonsensical densities.
#
# 2. Periodic boundaries: positions in the simulation box may wrap at
#    the boundary.  Shift all particles so the centre of mass is at the
#    origin before computing distances.  The KNN distance should not
#    straddle a periodic boundary.
#
# 3. Escapers: particles that leave the simulation domain get NaN positions.
#    Exclude them from the KNN tree at that snapshot — do NOT fill NaN
#    with zero (that would place them at the origin and corrupt the density
#    of particles near the centre).
#
# 4. Mass weighting: if particles have unequal masses, the KNN estimator
#    underestimates density in regions dominated by heavy particles.
#    Weight each particle by m_i when computing Q: Q_i → Q_i × m_i / m̄.
#
# 5. Q very close to zero: the log(Q) in entropy computations diverges.
#    Always clip Q to a floor of 1e-30 before taking logs.
#
# TIME COMPLEXITY (full loop):
#   KNN phase:       O(N_PSD_SNAPS × N² log N) worst case,
#                    O(N_PSD_SNAPS × N × K × log N) with KNN tree
#                  = O(40 × 500 × 32 × 9)  ≈ 5.7 × 10^6  — seconds
#   MSD/τ phase:     O(N × N_PSD_SNAPS²) ≈ 500 × 1600 = 8 × 10^5  — fast
#   Total: ~ 1–3 minutes on a single CPU core

# TODO: implement main computation loop

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.10 — FIGURES (NINE PLANNED)                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Figure descriptions and implementation hints:
#
# ── Figure 1: Phase-space projections at 5 epochs ─────────────────────────────
# Six-panel figure showing 2D projections of the 6D phase space:
#   (x, vx), (y, vy), (z, vz)  — position-velocity pairs
#   (x, y),  (r, vr), (L_z, E) — configuration, radial PS, integrals of motion
# Colour particles by log10(Q) — streams and clumps appear as bright ridges.
# HINT: use a perceptually uniform colourmap (viridis or plasma) with vmin/vmax
# set to the 5th and 95th percentiles of log10(Q) so outliers don't wash out
# the colour scale.
#
# Expected output: section33_phasespace_projections.png
#
# ── Figure 2: Q(r_0, t) heatmap (density evolution by initial radius) ─────────
# Heatmap: x-axis = time (snapshot epoch), y-axis = initial radius r_0,
# colour = mean log10(Q) per radial bin.
# This directly visualises where phase-space density is lost or gained —
# inner shells should be compressed (Q rises) while outer shells mix (Q drops).
#
# Expected output: section33_density_heatmap.png
#
# ── Figure 3: ΔQ distributions at 5 epochs ───────────────────────────────────
# PDF of the fractional density change ΔQ = (Q − Q_0)/Q_0 at five epochs.
# The PDF should start as a delta function at 0, then spread out.
# The left tail (ΔQ << 0) captures mixed particles; the right tail (ΔQ > 0)
# captures compressed/stream particles.
# HINT: use a KDE (scipy.stats.gaussian_kde) rather than a histogram for
# smoothness, since ΔQ can span many orders of magnitude.
#
# Expected output: section33_delta_Q_distributions.png
#
# ── Figure 4: Mixing entropy S_mix(t) vs. time ───────────────────────────────
# Line plot of S_mix(t) from t=0 to final snapshot.
# Overlay the resonant fraction f_resonant(t) from §32 on the same axis
# (right-hand y-axis, different colour).
# HYPOTHESIS: peaks in dS_mix/dt correspond to dips in f_resonant —
# mixing bursts destroy resonant structures.
# Mark the pericentric passages with vertical dashed lines.
#
# Expected output: section33_mixing_entropy.png
#
# ── Figure 5: τ_mix(r_0) — mixing timescale vs. initial radius ───────────────
# Scatter plot of τ_mix per particle vs. r_0, coloured by group.
# Particles that never mix get a symbol at the top of the axis (τ = ∞).
# EXPECTATION: inner halo particles have short τ_mix (fast mixing),
# outer halo and M31 particles have long τ_mix.
# Overlay the local orbital period T_orb(r_0) as a reference curve —
# τ_mix / T_orb measures the number of orbital periods to mix.
#
# Expected output: section33_mixing_timescale.png
#
# ── Figure 6: β(r, t) anisotropy heatmap ─────────────────────────────────────
# Heatmap of β(r, t): x-axis = time, y-axis = radius r (log scale).
# Diverging colourmap centred on 0: blue = isotropic/tangential, red = radial.
# EXPECTATION: β → +1 during pericentric passage (radial infall),
# returning to ~0.2 at late times (virialised anisotropy).
# HINT: mask bins with fewer than MIN_PART_BIN particles.
#
# Expected output: section33_anisotropy_heatmap.png
#
# ── Figure 7: Phase-space diffusion D_PS vs. initial radius ──────────────────
# Scatter of D_PS per particle vs. r_0, coloured by log10(Q_0).
# Use log scale on both axes.
# Overplot the prediction from the virial theorem:
#   D_PS ~ σ_v²(r) / T_relax(r) — the two-body relaxation diffusion.
# In an N-body simulation with N ~ 10^4, T_relax << t_Hubble, so measured
# D_PS should EXCEED the two-body prediction for chaotic particles.
#
# Expected output: section33_phase_diffusion.png
#
# ── Figure 8: Stream fraction f_stream(t) and stream particles ───────────────
# Top panel: f_stream(t) vs. time for each group (inner, mid, outer, M31).
# Bottom panel: sky map (l, b) or (x, y) of stream particles at the epoch
# of maximum f_stream — shows which streams are most coherent.
# HINT: the M31 group should have the highest f_stream at early times
# (it is being disrupted into streams); the inner group should be low
# (deep potential, fast phase mixing).
#
# Expected output: section33_stream_fraction.png
#
# ── Figure 9: Master summary panel ───────────────────────────────────────────
# 2×2 grid:
#   (0,0) Q(r_0, t) heatmap (§33 analogue of the §31 λ heatmap)
#   (0,1) β(r, t) anisotropy heatmap
#   (1,0) Mixing entropy S_mix(t) with resonant fraction overlay
#   (1,1) Three-way scatter: τ_mix vs. λ (§31) vs. C (§32)
#         — each axis is one method; colour by group; a perfect 3-way
#           agreement would lie on the diagonal of this tetrahedron.
#
# Expected output: section33_summary_panel.png

# TODO: implement all nine figures
# HINT: follow the exact same structure as Sections 21–32:
#   fig, ax = plt.subplots(...)
#   _ax(ax, xlabel=..., ylabel=..., title=..., log_x=..., log_y=...)
#   ... plotting code ...
#   fig.savefig(os.path.join(OUT_DIR, "section33_xxx.png"), ...)
#   plt.close(fig)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.11 — ANIMATION: PHASE-SPACE MIXING                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Three-panel animation:
#
# Left  : 2D phase-space projection (r, v_r) scatter of all tracked particles,
#         coloured by log10(Q(t)).  As mixing progresses, the colour distribution
#         flattens — the dense clumps and streams diffuse into the background.
#         Mark the Q = Q_median contour as a dashed line.
#
# Centre: Running mixing entropy S_mix(t) history with a vertical marker
#         at the current frame time.
#
# Right : Distribution of log10(Q / Q_0) at current epoch — should start
#         as a spike at 0 and broaden rightward (compression) and leftward
#         (mixing) as the merger progresses.
#
# HINT: for the colour normalisation, fix vmin = log10(Q_global_min) and
# vmax = log10(Q_global_max) across ALL frames so that the colour scale
# is consistent over time.  Do NOT renormalise per-frame — that would
# hide the actual density evolution.
#
# HINT: update the scatter plot using scat.set_array(log10(Q_arr[k]))
# and the histogram bars using bar_container.  Build the bar_container
# once before the animation loop with ax.bar(...) and update heights.
#
# Expected output: section33_animation_phasespace.mp4

# TODO: implement animation

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §33.12 — SECTION COMPLETE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Print the output manifest — same pattern as all previous sections.
# Also print a summary statistics table:
#
#   Group  | N | Mean log Q_0 | Mean τ_mix | Mean D_PS | Mean β (final)
#   ───────────────────────────────────────────────────────────────────
#   Inner  | … | …            | …          | …         | …
#   Mid    | … | …            | …          | …         | …
#   Outer  | … | …            | …          | …         | …
#   M31    | … | …            | …          | …         | …
#
# Also print the three cross-method Pearson correlations:
#   corr(τ_mix, 1/λ)     — phase-space mixing vs. Lyapunov time
#   corr(τ_mix, C)       — phase-space mixing vs. spectral complexity
#   corr(D_PS,  λ)       — phase-space diffusion vs. Lyapunov exponent
# Perfect agreement → corr = 1.  Discrepancies are scientifically interesting.

outputs_33 = [
    "section33_phasespace_projections.png",
    "section33_density_heatmap.png",
    "section33_delta_Q_distributions.png",
    "section33_mixing_entropy.png",
    "section33_mixing_timescale.png",
    "section33_anisotropy_heatmap.png",
    "section33_phase_diffusion.png",
    "section33_stream_fraction.png",
    "section33_animation_phasespace.mp4",
    "section33_summary_panel.png",
]

# TODO: implement output manifest printing
