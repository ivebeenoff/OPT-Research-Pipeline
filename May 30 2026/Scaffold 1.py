"""
===============================================================================
SECTION 32 — ORBITAL FREQUENCY ANALYSIS & SPECTRAL DYNAMICS
===============================================================================
Author  : Abhinav Vatsa  [SCAFFOLD — fill in implementation]

This section is a guided scaffold.  Every subsection contains:
  • Physical motivation and context
  • The exact quantity to compute and its formula
  • A suggested implementation strategy
  • Time and space complexity analysis
  • Hints on numerical pitfalls to avoid
  • Expected output description

The companion section (31) computed CHAOS via trajectory divergence (λ).
This section computes the SAME underlying phenomenon from a completely
different angle: the Fourier spectrum of individual particle orbits.

Physical connection to Section 31
──────────────────────────────────
A regular orbit in a static potential has discrete, commensurate frequencies
(Ω_r, Ω_θ, Ω_φ).  Its power spectrum has sharp, isolated peaks.
A chaotic orbit has a continuous, broad power spectrum — frequencies leak
into sidebands and sub-harmonics.

So "broad spectrum = chaotic orbit" and "sharp spectrum = regular orbit"
are equivalent to "λ > 0" and "λ ≤ 0" from Section 31, but:
  • Frequency analysis gives you the SPECIFIC frequencies, not just a scalar
  • It reveals WHICH resonances are being excited (Ω_r : Ω_θ = 2:1, etc.)
  • It works without a shadow particle — just the single trajectory
  • It naturally identifies frequency drift (secular evolution) as
    a separate signature from chaos

Key quantities computed in this section
────────────────────────────────────────
  Ω_r(t, i)     — radial oscillation frequency of particle i at time t
  Ω_θ(t, i)     — azimuthal frequency
  Ω_φ(t, i)     — vertical (z) frequency
  Ω_r / Ω_θ     — frequency ratio (rational = resonant, irrational = quasiperiodic)
  S(ω, i)       — power spectral density of orbit i
  Δω(t, i)      — frequency drift rate (secular non-stationarity)
  N_peaks(i)    — number of significant spectral peaks (regularity measure)
  f_resonant(t) — fraction of particles near a low-order resonance

Dependencies
────────────
  numpy.fft     — FFT for power spectra
  scipy.signal  — peak finding, window functions, spectrogram
  Section 26    — traj_pos, traj_r, traj_vel (Lagrangian trajectories)
  Section 31    — lambda_total (for correlation plots)

All globals from the parent pipeline are inherited.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.signal import find_peaks, windows, spectrogram
from scipy.ndimage import gaussian_filter
import os
import time
import warnings


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: The most important parameter here is the FFT window length.
# Too short → poor frequency resolution (Δω = 1/T_window).
# Too long  → averages over dynamical evolution, missing frequency drift.
# A good starting point: T_window ≈ 3–5 orbital periods.
# For a particle at r = 10 kpc with v_c = 220 km/s:
#   T_orbit ≈ 2π r / v_c ≈ 280 Myr ≈ 28 snapshots at 10 Myr/snap.
# So FFT_WINDOW = 80–150 is reasonable for inner-halo particles.

FFT_WINDOW      = 128    # [snapshots] — window length for each FFT
FFT_STEP        = 16     # [snapshots] — step between consecutive windows (sliding)
N_SPEC_PARTICLES = 600   # number of particles to analyse spectrally
PEAK_THRESHOLD  = 0.05   # fraction of max power — peaks below this are ignored
N_PEAKS_REGULAR = 5      # a particle with <= this many peaks is "regular"
RESONANCE_TOL   = 0.05   # tolerance for resonance detection: |Ω_r/Ω_θ − p/q| < TOL

# Frequency ratio grid for resonance detection.
# Low-order resonances: 1:1, 2:1, 3:1, 1:2, 3:2, 4:3, etc.
# HINT: generate these as fractions p/q for p, q in 1..N_RES_ORDER.
N_RES_ORDER = 4

# Temporal subsampling for spectrogram maps.
SPEC_SNAP_STEP = 20

# Animation.
ANIM_FPS_32     = 18
ANIM_DPI_32     = 100
ANIM_BITRATE_32 = 1600

print("\n" + "="*80)
print("  SECTION 32 · Orbital Frequency Analysis & Spectral Dynamics")
print("="*80)
print(f"  FFT window   : {FFT_WINDOW} snapshots")
print(f"  FFT step     : {FFT_STEP} snapshots")
print(f"  Particles    : {N_SPEC_PARTICLES}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.1 — LOAD TRAJECTORIES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: Reuse traj_pos, traj_r, traj_vel from Section 26 exactly as
# Section 31 did.  Copy the try/except block that falls back to recomputing
# if Section 26 was not run.  Subsample to N_SPEC_PARTICLES.
#
# The three time series you need per particle are:
#   x_r(t)   = r(t)           — radial coordinate [kpc]
#   x_z(t)   = pos[t, i, 2]  — vertical coordinate [kpc]
#   x_phi(t) = arctan2(pos[t,i,1], pos[t,i,0])  — azimuthal angle [rad]
#              NOTE: unwrap x_phi with np.unwrap to remove 2π jumps
#              before taking the FFT, otherwise the spectrum is dominated
#              by the discontinuities rather than the true frequency.
#
# TIME COMPLEXITY:  O(N_SPEC_PARTICLES × ns)  for loading
# SPACE COMPLEXITY: O(ns × N_SPEC_PARTICLES × 3)  ≈ 800 × 600 × 3 × 8 bytes ≈ 11 MB

# TODO: implement trajectory loading / inheritance
_traj_pos  = None   # replace with inherited or recomputed (ns, N, 3)
_traj_r    = None   # (ns, N)
_traj_vel  = None   # (ns, N, 3)
_r0        = None   # (N,)  initial radii
_group     = None   # (N,)  0=inner,1=mid,2=outer,3=M31
_N         = 0

# TODO: subsample to N_SPEC_PARTICLES, extract x_r, x_z, x_phi
# x_r   = _traj_r                         # (ns, N)
# x_z   = _traj_pos[:, :, 2]             # (ns, N)
# x_phi = np.unwrap(np.arctan2(           # (ns, N)  — MUST unwrap!
#             _traj_pos[:, :, 1],
#             _traj_pos[:, :, 0]), axis=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.2 — SINGLE-ORBIT POWER SPECTRAL DENSITY                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The power spectral density S(ω) of a time series x(t) is:
#
#   S(ω) = |X(ω)|²  where  X(ω) = ∫ x(t) e^{-iωt} dt
#
# For a discrete time series of length T snapshots with spacing Δt:
#   frequency resolution : Δω = 1 / (T × Δt)   [snap^{-1}]
#   Nyquist frequency    : ω_max = 1 / (2 Δt)   [snap^{-1}]
#
# HINT: always apply a window function before FFT to reduce spectral leakage.
# np.blackman(N) or scipy.signal.windows.hann(N) are good choices.
# Without windowing, a pure sinusoid at frequency ω_0 will produce power at
# ALL neighbouring frequencies, obscuring nearby spectral peaks.
#
# HINT: the physically relevant quantity is the NORMALISED power spectrum:
#   S_norm(ω) = S(ω) / sum(S(ω))
# This removes the effect of orbit amplitude, making spectra from particles
# at different radii directly comparable.
#
# Implementation strategy
# ───────────────────────
# For each particle i and each coordinate (r, z, φ):
#   1. Extract the time series x[0:ns, i].
#   2. Find valid (non-NaN) segments of length >= FFT_WINDOW.
#   3. Apply Hann window: x_win = x * np.hann(len(x))
#   4. FFT: X = np.fft.rfft(x_win)
#   5. PSD: S = np.abs(X)**2
#   6. Normalise: S /= S.sum()
#   7. Find peaks: scipy.signal.find_peaks(S, height=PEAK_THRESHOLD * S.max())
#   8. Dominant frequency: Ω = freqs[argmax(S)]
#
# TIME COMPLEXITY:  O(N × ns × log(FFT_WINDOW))  — fast, this is the cheap step
# SPACE COMPLEXITY: O(N × FFT_WINDOW / 2)  — one PSD per particle ≈ 2.4 MB

def compute_psd(x_series, dt=1.0):
    """
    Compute the normalised one-sided power spectral density of a 1D time series.

    Parameters
    ----------
    x_series : (T,)  — time series (no NaNs)
    dt       : float — sampling interval [snapshots]

    Returns
    -------
    freqs : (T//2 + 1,)  — frequencies [snap^{-1}]
    psd   : (T//2 + 1,)  — normalised power spectral density

    HINT: the dominant orbital frequency is freqs[np.argmax(psd)].
    For the RADIAL frequency Ω_r, use x_series = r(t).
    For the AZIMUTHAL frequency Ω_φ, use x_series = unwrapped φ(t).
      BUT NOTE: the FFT of φ(t) gives the mean angular velocity, not
      the oscillation frequency.  Use φ(t) − ⟨dφ/dt⟩ t first to remove
      the linear trend, then FFT the residual.
    """
    T    = len(x_series)
    win  = windows.hann(T)
    x_w  = (x_series - x_series.mean()) * win   # detrend + window
    X    = np.fft.rfft(x_w)
    psd  = np.abs(X)**2
    psd /= (psd.sum() + 1e-30)   # normalise
    freqs = np.fft.rfftfreq(T, d=dt)
    return freqs, psd


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.3 — SLIDING WINDOW SPECTROGRAM (TIME-FREQUENCY ANALYSIS)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The standard FFT gives S(ω) averaged over the ENTIRE trajectory.
# But frequencies DRIFT as the gravitational potential changes during the merger.
# A spectrogram S(ω, t) resolves this by computing the FFT in successive
# sliding windows of length FFT_WINDOW with step FFT_STEP.
#
# Frequency drift rate:
#   dΩ/dt  [snap^{-2}]
# A large |dΩ/dt| means the orbit is changing rapidly —
# either due to potential evolution or chaotic wandering.
#
# HINT: scipy.signal.spectrogram is convenient but uses a fixed window.
# For variable-length valid segments, implement the sliding window manually:
#
#   for t_start in range(0, ns - FFT_WINDOW, FFT_STEP):
#       t_end   = t_start + FFT_WINDOW
#       x_win   = x[t_start:t_end, i]
#       if np.isfinite(x_win).mean() < 0.8: continue  # skip gappy windows
#       freqs, psd = compute_psd(x_win)
#       Omega_r[t_start, i] = freqs[np.argmax(psd)]
#
# TIME COMPLEXITY:  O(N × (ns / FFT_STEP) × FFT_WINDOW × log FFT_WINDOW)
#                 = O(600 × 50 × 128 × 7) ≈ 2.7 × 10^7 operations  — ~minutes
# SPACE COMPLEXITY: O(N × (ns / FFT_STEP))  ≈ 600 × 50 × 8 bytes ≈ 240 KB
#                   For the full spectrogram: O(N × (ns/FFT_STEP) × FFT_WINDOW/2)
#                   ≈ 600 × 50 × 64 × 8 bytes ≈ 15 MB  — store selectively

# Pre-allocate frequency arrays.
n_windows    = (ns - FFT_WINDOW) // FFT_STEP + 1
time_windows = np.full(n_windows, np.nan)   # time axis for sliding windows

Omega_r_ts   = np.full((n_windows, N_SPEC_PARTICLES), np.nan)  # radial freq
Omega_phi_ts = np.full((n_windows, N_SPEC_PARTICLES), np.nan)  # azimuthal freq
Omega_z_ts   = np.full((n_windows, N_SPEC_PARTICLES), np.nan)  # vertical freq

# TODO: implement the sliding window loop
# HINT: populate time_windows[w] = time_arr[t_start + FFT_WINDOW // 2]
#       to associate each window with its midpoint time.


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.4 — FREQUENCY RATIOS AND RESONANCE DETECTION                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# An orbit is RESONANT if its frequency ratio is a rational number:
#   Ω_r / Ω_φ = p / q   for small integers p, q
#
# Low-order resonances (small p, q) are the dynamically important ones.
# The most common in galaxy potentials:
#   2:1  — "banana" orbits (Ω_r = 2 Ω_φ)
#   3:2  — "fish" orbits
#   4:3  — higher-order resonance
#   1:1  — loop orbits (Ω_r = Ω_φ)
#
# Near a resonance, a small perturbation (e.g., the tidal field of M31)
# can efficiently pump energy into the orbit — this is the mechanism of
# RESONANT HEATING.
#
# Implementation
# ──────────────
# 1. Build the set of rational fractions p/q for p, q ≤ N_RES_ORDER.
#    HINT: use fractions.Fraction to avoid duplicates (2/4 = 1/2).
#
# 2. For each particle and each time window, compute:
#      ratio = Omega_r / (Omega_phi + 1e-10)
#
# 3. Check: is |ratio − p/q| < RESONANCE_TOL for any resonance p/q?
#      resonant_mask[w, i] = any(abs(ratio - pq) < RESONANCE_TOL for pq in resonances)
#
# 4. Track the resonance fraction over time:
#      f_resonant[w] = mean(resonant_mask[w, :])
#
# TIME COMPLEXITY:  O(n_windows × N × N_resonances)
#                 = O(50 × 600 × 20) ≈ 6 × 10^5 — negligible
# SPACE COMPLEXITY: O(n_windows × N)  — same as frequency arrays

# TODO: build resonance fraction array
# from fractions import Fraction
# resonances = list of unique p/q values for p,q in range(1, N_RES_ORDER+1)
# f_resonant_ts = np.full(n_windows, np.nan)
# resonant_mask = np.full((n_windows, N_SPEC_PARTICLES), False)

# HINT: a useful diagnostic is to track WHICH resonance a particle is near,
# not just whether it is resonant.  Store the nearest resonance p:q for each
# particle at each window in a string or integer array.


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.5 — SPECTRAL COMPLEXITY MEASURE                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# A regular orbit has a SPARSE spectrum: power concentrated in a few discrete
# peaks at Ω_r, 2Ω_r, 3Ω_r, and their combinations with Ω_φ.
#
# A chaotic orbit has a DIFFUSE spectrum: power spread across many frequencies.
#
# We quantify this with the spectral complexity C:
#
#   C(i) = exp( H(S_i) )   where H(S) = −Σ S_k ln S_k  (spectral entropy)
#
# This is the exponential of the Shannon entropy of the normalised PSD.
# It equals the "effective number of frequencies" in the spectrum:
#   C = 1   →  single-frequency orbit (perfectly regular)
#   C = N/2 →  white-noise orbit (maximally chaotic)
#
# HINT: This is a MUCH more robust regularity measure than just counting peaks,
# because it does not depend on a threshold parameter.
# It is also directly comparable between particles at different radii.
#
# TIME COMPLEXITY:  O(N × FFT_WINDOW/2)  — one pass over each spectrum
# SPACE COMPLEXITY: O(N)  — just one scalar per particle

def spectral_complexity(psd):
    """
    Compute the spectral complexity C = exp(H) where H is spectral entropy.

    Parameters
    ----------
    psd : (M,)  — normalised power spectral density (sums to 1)

    Returns
    -------
    C : float  — effective number of frequencies [1, M]

    HINT: clip psd to a small positive value before log to avoid log(0).
    HINT: a useful normalised version is C / (len(psd)/2) ∈ [0, 1],
          where 0 = perfectly regular and 1 = maximally chaotic.
          This makes the regularity index directly comparable to the
          Lyapunov chaos fraction from Section 31.
    """
    p    = np.clip(psd, 1e-12, None)
    H    = -np.sum(p * np.log(p))
    return float(np.exp(H))

# TODO: compute spectral_complexity for each particle over its full trajectory
# complexity_arr = np.full(N_SPEC_PARTICLES, np.nan)
# for i in range(N_SPEC_PARTICLES):
#     ...


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.6 — FREQUENCY DRIFT (SECULAR NON-STATIONARITY)                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Even a perfectly regular orbit will have slowly drifting frequencies if the
# gravitational potential changes over time (adiabatic evolution).
# This is SECULAR DRIFT and is NOT a sign of chaos.
#
# A CHAOTIC orbit shows RAPID, IRREGULAR frequency drift — the frequency
# jumps between values with no systematic trend.
#
# We separate these two behaviours by:
#   1. Computing dΩ_r/dt  (the time derivative of the radial frequency)
#   2. Computing the VARIANCE of dΩ_r/dt over time (high variance = chaotic drift)
#   3. Comparing to the MEAN drift (high mean / high variance = adiabatic)
#
# Key diagnostic:
#   σ(dΩ_r/dt)         — standard deviation of frequency drift
#   |mean(dΩ_r/dt)|    — secular drift rate
#   σ / |mean|         — chaos-to-adiabatic ratio
#
# TIME COMPLEXITY:  O(N × n_windows)  — just a gradient computation
# SPACE COMPLEXITY: O(N × n_windows)  — same as Omega arrays

# TODO: compute frequency drift per particle
# d_Omega_r = np.gradient(Omega_r_ts, axis=0)   # (n_windows, N)
# sigma_drift = np.nanstd(d_Omega_r, axis=0)    # (N,) — per particle
# mean_drift  = np.nanmean(np.abs(d_Omega_r), axis=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.7 — PRE-ALLOCATION FOR ALL OUTPUT ARRAYS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT: allocate everything here before the main loop to avoid repeated
# memory allocation inside the loop.  Use np.full(..., np.nan) so that
# missing data is explicit rather than silent zeros.
#
# Arrays needed:
#
#   Per-particle scalars (shape: N_SPEC_PARTICLES):
#     complexity_arr   — spectral complexity C ∈ [1, FFT_WINDOW/2]
#     sigma_drift_arr  — standard deviation of frequency drift
#     mean_drift_arr   — mean absolute frequency drift rate
#     n_peaks_arr      — number of significant spectral peaks
#     dominant_freq_r  — dominant radial frequency [snap^{-1}]
#     dominant_freq_phi — dominant azimuthal frequency
#
#   Time-resolved arrays (shape: n_windows × N_SPEC_PARTICLES):
#     Omega_r_ts       — already allocated in §32.3
#     Omega_phi_ts     — already allocated
#     Omega_z_ts       — already allocated
#     ratio_ts         — Omega_r / Omega_phi per window
#     resonant_mask    — boolean, particle near a resonance
#
#   Radial-bin averages (shape: n_windows × nb_sph):
#     Omega_r_radial_ts  — mean Ω_r per shell per window
#     complexity_radial_ts — mean C per shell per window
#
#   Scalars per time window (shape: n_windows):
#     f_resonant_ts    — fraction of resonant particles
#     f_regular_ts     — fraction of "regular" particles (N_peaks <= N_PEAKS_REGULAR)
#
# SPACE COMPLEXITY TOTAL:
#   Per-particle scalars: O(N) ≈ 600 × 8 bytes = 5 KB
#   Time-resolved:        O(n_windows × N) ≈ 50 × 600 × 8 bytes = 240 KB per array
#   Radial-bin averages:  O(n_windows × nb_sph) ≈ 50 × 39 × 8 bytes = 16 KB per array
#   TOTAL: ~ a few MB — very lightweight

# TODO: allocate all arrays here


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.8 — MAIN COMPUTATION LOOP                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Structure of the loop
# ─────────────────────
# There are two natural loop structures here.  Choose whichever is cleaner:
#
# OPTION A: outer loop over windows, inner loop over particles.
#   for w, t_start in enumerate(range(0, ns-FFT_WINDOW, FFT_STEP)):
#       for i in range(N_SPEC_PARTICLES):
#           ...
#   Advantage: easy to add per-window aggregates (f_resonant, etc.)
#   Disadvantage: re-reads the same time series many times.
#
# OPTION B: outer loop over particles, inner loop over windows.
#   for i in range(N_SPEC_PARTICLES):
#       for w, t_start in enumerate(...):
#           ...
#   Advantage: better cache locality (reads contiguous memory x_r[:, i]).
#   Disadvantage: harder to compute per-window aggregates without a second pass.
#
# RECOMMENDATION: use Option B — the inner loop is tight and the trajectory
# arrays are stored column-major in practice.  Then compute per-window
# aggregates in a separate vectorised pass over the filled arrays.
#
# Numerical pitfalls
# ──────────────────
# 1. NaN handling: if more than 20% of a window is NaN (tracking lost),
#    skip the window for that particle.  Do NOT interpolate NaNs before FFT —
#    interpolated values have artificially smooth spectra.
#
# 2. Azimuthal angle wrapping: ALWAYS call np.unwrap before computing
#    the FFT of φ(t).  Without unwrapping, the 2π jumps dominate the spectrum.
#    After unwrapping, remove the linear trend (mean angular velocity)
#    before windowing.
#
# 3. Short trajectories: if a particle has fewer than FFT_WINDOW valid points
#    in total, it cannot be analysed.  Skip it and leave NaN.
#
# 4. DC component: the zeroth FFT coefficient (ω=0) represents the time-mean
#    of the signal.  For r(t) it is the mean orbital radius — physically
#    meaningful but not an orbital frequency.  Exclude it from peak finding.
#
# TIME COMPLEXITY (full loop):
#   O(N × n_windows × FFT_WINDOW × log FFT_WINDOW)
#   = O(600 × 50 × 128 × 7)
#   ≈ 2.7 × 10^7 floating-point operations
#   ≈ 2–5 minutes on a single CPU core

# TODO: implement main computation loop


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.9 — FIGURES (NINE PLANNED)                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Figure descriptions and implementation hints:
#
# ── Figure 1: Individual power spectra at 5 epochs ────────────────────────────
# Show the PSD S(ω) for one representative particle from each group
# (inner, mid, outer, M31) at the five profile epochs.
# Use a log-log scale so both the sharp peaks and the continuous background
# are visible.
#
# HINT: overplot vertical lines at Ω_r, 2Ω_r, 3Ω_r (harmonics of the radial
# frequency) to verify that the identified dominant frequency is correct.
# For a regular orbit these harmonics should be visible as smaller peaks.
#
# Expected output: section32_individual_spectra.png
#
# ── Figure 2: Ω_r(r_0) and Ω_φ(r_0) frequency profiles ──────────────────────
# Plot the dominant radial and azimuthal frequencies as functions of initial
# radius r_0.  This is the "frequency space" counterpart of the ρ(r) profile.
# The ratio Ω_r / Ω_φ decreasing with radius would indicate the outer halo
# is dominated by radial (box) orbits while the inner halo is loop-dominated.
#
# Expected output: section32_frequency_profiles.png
#
# ── Figure 3: Ω_r(r_0, t) and Ω_φ(r_0, t) heatmaps ─────────────────────────
# Two heatmaps in the same figure showing the frequency evolution.
# The x-axis is the sliding window time, the y-axis is initial radius r_0.
# HINT: use a diverging colourmap if you show DRIFT (change from initial value)
# rather than absolute frequency.
#
# Expected output: section32_frequency_heatmaps.png
#
# ── Figure 4: P(Ω_r / Ω_φ) resonance distribution at 5 epochs ───────────────
# Histogram of the frequency ratio Ω_r / Ω_φ across all particles at each epoch.
# Mark the low-order resonances (2:1, 3:2, 1:1, etc.) with vertical lines.
# A spike at a resonance means many particles are trapped there.
# The relative heights of the spikes change as the potential evolves.
#
# Expected output: section32_resonance_distribution.png
#
# ── Figure 5: Resonant fraction f_resonant(t) vs. time ───────────────────────
# Line plot of the fraction of particles near any low-order resonance.
# Overlay the chaos fraction from Section 31 (if available) for comparison.
# HYPOTHESIS: resonance trapping PRECEDES chaos onset — particles first get
# trapped in a resonance, then the resonance overlaps with a neighbouring one,
# triggering the onset of chaos (Chirikov overlap criterion).
#
# Expected output: section32_resonant_fraction.png
#
# ── Figure 6: Spectral complexity C(r_0, t) heatmap ─────────────────────────
# Heatmap of mean spectral complexity C per initial-radius bin vs. time.
# This is the frequency-domain analogue of the λ(r_0, t) heatmap in §31.
# They should show similar structures (high chaos = high complexity) but
# the frequency analysis will capture slowly growing chaos that λ misses
# because it requires a shadow particle.
#
# Expected output: section32_complexity_heatmap.png
#
# ── Figure 7: Spectral complexity vs. Lyapunov exponent scatter ──────────────
# Scatter plot of C(i) vs. λ(i) for each tracked particle.
# If both methods agree perfectly, this should be a monotonic curve.
# Deviations reveal particles where the two methods disagree —
# scientifically interesting cases (e.g. particles that are spectrally
# complex but not diverging, or vice versa).
# REQUIRES: lambda_total from Section 31.
#
# Expected output: section32_complexity_vs_lambda.png
#
# ── Figure 8: Frequency drift σ(dΩ/dt) vs. initial radius ───────────────────
# Scatter of frequency drift standard deviation vs. r_0.
# Colour by spectral complexity C.
# HINT: use a log scale on both axes — drift spans many orders of magnitude.
# Particles with high σ(dΩ/dt) AND high C are unambiguously chaotic.
# Particles with high σ(dΩ/dt) but LOW C may be in a slowly evolving resonance.
#
# Expected output: section32_frequency_drift.png
#
# ── Figure 9: Master summary panel ───────────────────────────────────────────
# 2×2 grid:
#   (0,0) Ω_r(r_0, t) heatmap
#   (0,1) Spectral complexity C(r_0, t) heatmap
#   (1,0) Resonant fraction f_resonant(t) vs. time
#   (1,1) C vs. λ scatter (if Section 31 was run)
#
# Expected output: section32_summary_panel.png

# TODO: implement all nine figures
# HINT: follow the exact same structure as Sections 21–31:
#   fig, ax = plt.subplots(...)
#   _ax(ax, xlabel=..., ylabel=..., title=..., log_x=..., log_y=...)
#   ... plotting code ...
#   fig.savefig(os.path.join(OUT_DIR, "section32_xxx.png"), ...)
#   plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.10 — ANIMATION: SPECTRAL EVOLUTION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Three-panel animation:
#
# Left  : 2D scatter of particles in (Ω_r, Ω_φ) "frequency space"
#         coloured by spectral complexity C.
#         As the merger progresses, particles migrate in frequency space —
#         those that become chaotic scatter away from the resonance lines.
#         Mark the resonance lines (Ω_r/Ω_φ = p/q) as fixed dashed lines.
#
# Centre: Running resonant fraction f_resonant(t) history.
#
# Right : Spectral complexity distribution P(C) histogram for current window.
#
# TIME COMPLEXITY: O(n_windows × N) for animation — fast since data pre-computed
# SPACE COMPLEXITY: negligible additional storage beyond what is already allocated
#
# HINT: matplotlib.animation.FuncAnimation with blit=True.
# Update the scatter plot using scat.set_offsets() and scat.set_array()
# (for the colour mapping).
# Update the histogram bars using bar.set_height() for each bar.
#
# Expected output: section32_animation_frequencies.mp4

# TODO: implement animation


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.11 — SECTION COMPLETE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Print the output manifest — same pattern as all previous sections.
# Also print a summary statistics table:
#
#   Group    | N | Mean Ω_r | Mean C | f_resonant | f_regular
#   ─────────────────────────────────────────────────────────
#   Inner    | …  | …        | …      | …          | …
#   Mid      | …  | …        | …      | …          | …
#   Outer    | …  | …        | …      | …          | …
#   M31      | …  | …        | …      | …          | …

outputs_32 = [
    "section32_individual_spectra.png",
    "section32_frequency_profiles.png",
    "section32_frequency_heatmaps.png",
    "section32_resonance_distribution.png",
    "section32_resonant_fraction.png",
    "section32_complexity_heatmap.png",
    "section32_complexity_vs_lambda.png",
    "section32_frequency_drift.png",
    "section32_animation_frequencies.mp4",
    "section32_summary_panel.png",
]

# TODO: implement output manifest printing
