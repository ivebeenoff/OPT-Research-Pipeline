# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.2 — AUTOCORRELATION-BASED FREQUENCY ESTIMATION                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Instead of estimating orbital frequencies from the Fourier spectrum,
# we can estimate them directly from the temporal self-similarity of
# the orbit.
#
# For a periodic signal x(t), the autocorrelation function:
#
#   R(τ) = ⟨x(t) x(t+τ)⟩
#
# exhibits peaks at integer multiples of the oscillation period.
#
# Therefore:
#
#   T_orbit ≈ argmax(R(τ > 0))
#   Ω = 1 / T_orbit
#
# Advantages relative to FFT:
#   • More robust for short trajectories
#   • Less sensitive to spectral leakage
#   • Naturally handles mildly non-stationary signals
#   • No frequency-bin quantisation effects
#
# HINT: always remove the mean before computing the autocorrelation.
# Otherwise the DC component dominates the correlation function.
#
# HINT: normalise by R(0) so that all particles are directly comparable.
#
# TIME COMPLEXITY:
#   O(T log T) using FFT-based correlation
#
# SPACE COMPLEXITY:
#   O(T)

def compute_autocorrelation_frequency(x_series, dt=1.0):
    """
    Estimate the dominant oscillation frequency using autocorrelation.

    Parameters
    ----------
    x_series : (T,)
        Input trajectory component.

    dt : float
        Snapshot spacing.

    Returns
    -------
    omega : float
        Dominant oscillation frequency.

    period : float
        Estimated orbital period.

    acf : (T,)
        Normalised autocorrelation function.

    lags : (T,)
        Lag axis.
    """

    x = np.asarray(x_series, dtype=float)

    if len(x) < 8:
        return np.nan, np.nan, None, None

    x = x - np.nanmean(x)

    fft_size = 2 ** int(np.ceil(np.log2(2 * len(x))))

    X = np.fft.fft(x, n=fft_size)

    acf = np.fft.ifft(X * np.conjugate(X)).real
    acf = acf[:len(x)]

    acf /= (acf[0] + 1e-30)

    lags = np.arange(len(acf)) * dt

    peaks, _ = find_peaks(
        acf[1:],
        height=0.10,
        distance=max(2, len(acf) // 20)
    )

    if len(peaks) == 0:
        return np.nan, np.nan, acf, lags

    peak_idx = peaks[0] + 1

    period = lags[peak_idx]

    if period <= 0:
        return np.nan, np.nan, acf, lags

    omega = 1.0 / period

    return omega, period, acf, lags
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.3 — SLIDING AUTOCORRELATION TRACKING                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# The FFT spectrogram tracks the evolution of spectral power with time.
#
# Here we instead track the evolution of the dominant autocorrelation period.
#
# For every sliding window:
#
#   x(t) → R(τ) → T_orbit → Ω
#
# yielding:
#
#   Ω_r(t)
#   Ω_phi(t)
#   Ω_z(t)
#
# directly as functions of time.
#
# A regular orbit produces a nearly constant Ω(t).
#
# A chaotic orbit produces rapid fluctuations and discontinuous jumps
# in Ω(t) as the orbit wanders through phase space.
#
# HINT:
# Skip windows with >20% NaNs exactly as in the FFT implementation.
#
# TIME COMPLEXITY:
#   O(N × n_windows × FFT_WINDOW log FFT_WINDOW)
#
# SPACE COMPLEXITY:
#   O(N × n_windows)

n_windows = (ns - FFT_WINDOW) // FFT_STEP + 1

time_windows = np.full(n_windows, np.nan)

Omega_r_ts = np.full(
    (n_windows, N_SPEC_PARTICLES),
    np.nan
)

Omega_phi_ts = np.full(
    (n_windows, N_SPEC_PARTICLES),
    np.nan
)

Omega_z_ts = np.full(
    (n_windows, N_SPEC_PARTICLES),
    np.nan
)

for i in range(N_SPEC_PARTICLES):

    r_track = x_r[:, i]
    z_track = x_z[:, i]
    p_track = x_phi[:, i]

    for w, t0 in enumerate(
        range(
            0,
            ns - FFT_WINDOW + 1,
            FFT_STEP
        )
    ):

        t1 = t0 + FFT_WINDOW

        r_win = r_track[t0:t1]
        z_win = z_track[t0:t1]
        p_win = p_track[t0:t1]

        if np.isfinite(r_win).mean() < 0.80:
            continue

        if np.isfinite(z_win).mean() < 0.80:
            continue

        if np.isfinite(p_win).mean() < 0.80:
            continue

        omega_r, _, _, _ = compute_autocorrelation_frequency(r_win)

        omega_z, _, _, _ = compute_autocorrelation_frequency(z_win)

        p_resid = p_win - np.linspace(
            p_win[0],
            p_win[-1],
            len(p_win)
        )

        omega_phi, _, _, _ = compute_autocorrelation_frequency(
            p_resid
        )

        Omega_r_ts[w, i] = omega_r
        Omega_z_ts[w, i] = omega_z
        Omega_phi_ts[w, i] = omega_phi

        time_windows[w] = time_arr[
            t0 + FFT_WINDOW // 2
        ]
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.5 — REGULARITY INDEX FROM AUTOCORRELATION DECAY                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Spectral entropy measures how many frequencies participate in an orbit.
#
# An alternative approach is to quantify how rapidly temporal memory
# is lost.
#
# Regular orbit:
#   R(τ) remains oscillatory for many orbital periods.
#
# Chaotic orbit:
#   R(τ) decays rapidly toward zero.
#
# We therefore define:
#
#   C_corr = ∫ |R(τ)| dτ
#
# Large C_corr:
#   long-lived periodic memory
#
# Small C_corr:
#   rapidly decorrelating trajectory
#
# This acts as a chaos diagnostic without ever constructing
# a power spectrum.
#
# TIME COMPLEXITY:
#   O(T)
#
# SPACE COMPLEXITY:
#   O(T)

def correlation_complexity(x_series):
    """
    Correlation-based orbital regularity measure.
    """

    x = np.asarray(x_series, dtype=float)

    if len(x) < 8:
        return np.nan

    x = x - np.nanmean(x)

    fft_size = 2 ** int(np.ceil(np.log2(2 * len(x))))

    X = np.fft.fft(x, n=fft_size)

    acf = np.fft.ifft(
        X * np.conjugate(X)
    ).real

    acf = acf[:len(x)]

    acf /= (acf[0] + 1e-30)

    return np.trapz(
        np.abs(acf),
        dx=1.0
    )

complexity_arr = np.full(
    N_SPEC_PARTICLES,
    np.nan
)

for i in range(N_SPEC_PARTICLES):

    valid = np.isfinite(x_r[:, i])

    if valid.sum() < FFT_WINDOW:
        continue

    complexity_arr[i] = correlation_complexity(
        x_r[valid, i]
    )
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.4 — PERIOD RATIO ANALYSIS & RESONANCE IDENTIFICATION                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Rather than comparing orbital frequencies directly, we can compare the
# corresponding orbital periods:
#
#     T = 1 / Ω
#
# Resonant orbits satisfy:
#
#     T_φ / T_r = q / p
#
# which is mathematically equivalent to Ω_r / Ω_φ = p / q.
#
# Since the orbital period is measured directly from successive peaks in the
# autocorrelation function, resonance detection becomes independent of the FFT.
#
# HINT:
# Construct the list of candidate resonances once before entering the loop.
#
# HINT:
# Compare every measured period ratio against the low-order rational ratios.
# The closest resonance is assigned provided it lies within RESONANCE_TOL.
#
# TIME COMPLEXITY:
#     O(n_windows × N × N_resonances)
#
# SPACE COMPLEXITY:
#     O(n_windows × N)

from fractions import Fraction

resonances = []

for p in range(1, N_RES_ORDER + 1):
    for q in range(1, N_RES_ORDER + 1):

        frac = Fraction(q, p)

        value = frac.numerator / frac.denominator

        if value not in resonances:
            resonances.append(value)

resonances = np.array(sorted(resonances))

ratio_ts = np.full(
    (n_windows, N_SPEC_PARTICLES),
    np.nan
)

nearest_resonance = np.full(
    (n_windows, N_SPEC_PARTICLES),
    np.nan
)

resonant_mask = np.zeros(
    (n_windows, N_SPEC_PARTICLES),
    dtype=bool
)

f_resonant_ts = np.full(
    n_windows,
    np.nan
)

for w in range(n_windows):

    for i in range(N_SPEC_PARTICLES):

        omega_r = Omega_r_ts[w, i]
        omega_phi = Omega_phi_ts[w, i]

        if not np.isfinite(omega_r):
            continue

        if not np.isfinite(omega_phi):
            continue

        period_ratio = omega_phi / (omega_r + 1e-30)

        ratio_ts[w, i] = period_ratio

        idx = np.argmin(
            np.abs(resonances - period_ratio)
        )

        nearest = resonances[idx]

        nearest_resonance[w, i] = nearest

        if abs(period_ratio - nearest) < RESONANCE_TOL:
            resonant_mask[w, i] = True

    f_resonant_ts[w] = np.nanmean(
        resonant_mask[w]
    )
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.6 — TEMPORAL STABILITY OF ORBITAL PERIODS                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Physical background
# ───────────────────
# Instead of studying the evolution of orbital frequencies directly,
# we analyse the stability of the measured orbital periods.
#
# For every particle:
#
#     T_r(t)
#
# is tracked throughout the simulation.
#
# Slowly varying periods indicate adiabatic evolution.
#
# Large stochastic fluctuations indicate chaotic evolution.
#
# Diagnostics computed:
#
#     σ(T)
#         standard deviation of orbital period
#
#     mean(|dT/dt|)
#         average drift rate
#
#     CV = σ / mean(T)
#         coefficient of variation
#
# HINT:
# The coefficient of variation provides a dimensionless measure that is
# directly comparable across different orbital families.
#
# TIME COMPLEXITY:
#     O(N × n_windows)
#
# SPACE COMPLEXITY:
#     O(N)

period_ts = 1.0 / (Omega_r_ts + 1e-30)

d_period = np.gradient(
    period_ts,
    axis=0
)

sigma_period = np.nanstd(
    period_ts,
    axis=0
)

mean_period = np.nanmean(
    period_ts,
    axis=0
)

period_drift = np.nanmean(
    np.abs(d_period),
    axis=0
)

period_cv = sigma_period / (
    mean_period + 1e-30
)
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.7 — PRE-ALLOCATION OF OUTPUT ARRAYS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HINT:
# Allocate every array before entering the expensive computation stage.
# Missing values should always be represented explicitly using NaN.
#
# Arrays required:
#
#     complexity_arr
#     sigma_period
#     period_drift
#     period_cv
#     dominant_period
#     dominant_frequency
#
#     ratio_ts
#     resonant_mask
#     nearest_resonance
#
#     f_resonant_ts
#     f_regular_ts
#
# SPACE COMPLEXITY:
#     Only a few MB for the complete section.

dominant_period = np.full(
    N_SPEC_PARTICLES,
    np.nan
)

dominant_frequency = np.full(
    N_SPEC_PARTICLES,
    np.nan
)

f_regular_ts = np.full(
    n_windows,
    np.nan
)

regular_mask = np.zeros(
    (n_windows, N_SPEC_PARTICLES),
    dtype=bool
)

for i in range(N_SPEC_PARTICLES):

    omega = np.nanmedian(
        Omega_r_ts[:, i]
    )

    if not np.isfinite(omega):
        continue

    dominant_frequency[i] = omega

    dominant_period[i] = 1.0 / (
        omega + 1e-30
    )

for w in range(n_windows):

    regular_mask[w] = (
        period_cv < 0.10
    )

    f_regular_ts[w] = np.nanmean(
        regular_mask[w]
    )
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.8 — MAIN COMPUTATION LOOP                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Structure of the loop
# ─────────────────────
# Since the autocorrelation method operates independently on each trajectory,
# we iterate particle-by-particle and recover all orbital diagnostics from the
# corresponding correlation functions.
#
# Recommended workflow:
#
#   1. Compute the global orbital period.
#   2. Compute the correlation-based regularity index.
#   3. Recover the dominant orbital frequency.
#   4. Store all quantities for later population statistics.
#
# Numerical pitfalls
# ──────────────────
# 1. Remove NaNs before computing any correlation function.
# 2. Ignore trajectories shorter than FFT_WINDOW snapshots.
# 3. Always subtract the mean before correlation.
# 4. Correlation peaks occurring at lag = 0 are not physical periods.
#
# TIME COMPLEXITY:
#     O(N × ns log ns)
#
# SPACE COMPLEXITY:
#     O(ns)

for i in range(N_SPEC_PARTICLES):

    valid = (
        np.isfinite(x_r[:, i]) &
        np.isfinite(x_phi[:, i]) &
        np.isfinite(x_z[:, i])
    )

    if valid.sum() < FFT_WINDOW:
        continue

    r_track = x_r[valid, i]
    z_track = x_z[valid, i]
    phi_track = x_phi[valid, i]

    phi_track = phi_track - np.linspace(
        phi_track[0],
        phi_track[-1],
        len(phi_track)
    )

    omega_r, period_r, _, _ = compute_autocorrelation_frequency(
        r_track
    )

    omega_z, period_z, _, _ = compute_autocorrelation_frequency(
        z_track
    )

    omega_phi, period_phi, _, _ = compute_autocorrelation_frequency(
        phi_track
    )

    dominant_frequency[i] = omega_r
    dominant_period[i] = period_r

    complexity_arr[i] = correlation_complexity(
        r_track
    )

    if np.isfinite(period_r):

        sigma_period[i] = np.nanstd(
            period_ts[:, i]
        )

        period_drift[i] = np.nanmean(
            np.abs(d_period[:, i])
        )
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.9 — FIGURES (NINE PLANNED)                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Figure descriptions
# ───────────────────
# The diagnostics now visualise correlation-derived quantities instead of
# Fourier power spectra.
#
# Figure 1:
#     Representative autocorrelation functions.
#
# Figure 2:
#     Dominant orbital period versus initial radius.
#
# Figure 3:
#     Orbital period evolution heatmap.
#
# Figure 4:
#     Period-ratio distribution.
#
# Figure 5:
#     Resonant fraction history.
#
# Figure 6:
#     Correlation complexity heatmap.
#
# Figure 7:
#     Correlation complexity versus Lyapunov exponent.
#
# Figure 8:
#     Period variability versus radius.
#
# Figure 9:
#     Master diagnostic summary.
#
# HINT:
# Maintain the same plotting utilities used throughout previous sections so
# that the appearance remains consistent.

fig, ax = plt.subplots(
    figsize=(8, 5)
)

sample = np.nanargmin(_r0)

_, _, acf, lags = compute_autocorrelation_frequency(
    x_r[:, sample][np.isfinite(x_r[:, sample])]
)

ax.plot(
    lags,
    acf,
    lw=2
)

_ax(
    ax,
    xlabel="Lag [snapshots]",
    ylabel="Autocorrelation",
    title="Representative Orbital Autocorrelation"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "section32_individual_spectra.png"
    ),
    dpi=200
)

plt.close(fig)
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.10 — ANIMATION: CORRELATION EVOLUTION                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Three-panel animation
#
# Left:
#     Particle distribution in
#         (T_r , T_phi)
#     space.
#
# Centre:
#     Resonant fraction history.
#
# Right:
#     Distribution of correlation complexity.
#
# HINT:
# Since every diagnostic has already been precomputed, each animation frame
# only updates artists rather than recomputing orbital quantities.
#
# TIME COMPLEXITY:
#     O(n_windows × N)
#
# SPACE COMPLEXITY:
#     Negligible additional memory.

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

scat = axes[0].scatter(
    [],
    [],
    s=8
)

line, = axes[1].plot(
    [],
    [],
    lw=2
)

hist_vals = np.linspace(
    0,
    np.nanmax(complexity_arr),
    25
)

bars = axes[2].bar(
    hist_vals[:-1],
    np.zeros(len(hist_vals) - 1),
    width=np.diff(hist_vals)
)

def update(frame):

    periods = 1.0 / (
        Omega_r_ts[frame] + 1e-30
    )

    periods_phi = 1.0 / (
        Omega_phi_ts[frame] + 1e-30
    )

    scat.set_offsets(
        np.column_stack(
            (
                periods,
                periods_phi
            )
        )
    )

    line.set_data(
        time_windows[:frame + 1],
        f_resonant_ts[:frame + 1]
    )

    counts, _ = np.histogram(
        complexity_arr,
        bins=hist_vals
    )

    for bar, h in zip(bars, counts):
        bar.set_height(h)

    return (
        scat,
        line,
        *bars
    )

anim = animation.FuncAnimation(
    fig,
    update,
    frames=n_windows,
    interval=60,
    blit=False
)

anim.save(
    os.path.join(
        OUT_DIR,
        "section32_animation_frequencies.mp4"
    ),
    fps=ANIM_FPS_32,
    dpi=ANIM_DPI_32,
    bitrate=ANIM_BITRATE_32
)

plt.close(fig)
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §32.11 — SECTION COMPLETE                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Print the output manifest together with the principal correlation-based
# diagnostics for each radial population.

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

print("\n")
print("=" * 80)
print("SECTION 32 COMPLETE")
print("=" * 80)

for name in outputs_32:
    print(f"  ✓ {name}")

print("\nSummary Statistics\n")

labels = [
    "Inner",
    "Mid",
    "Outer",
    "M31"
]

for g, label in enumerate(labels):

    mask = (_group == g)

    print(
        f"{label:8s} | "
        f"N={mask.sum():4d} | "
        f"T={np.nanmean(dominant_period[mask]):8.3f} | "
        f"C={np.nanmean(complexity_arr[mask]):8.3f}"
    )

print("=" * 80)
