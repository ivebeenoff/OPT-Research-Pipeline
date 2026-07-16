# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 28 — DIAGNOSTIC 5: MW vs. M31 DENSITY DISPERSION & PROFILE RATIO   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Technical Objective:
# Analyze the local density dominance of the Milky Way and Andromeda progenitors.
# By comparing rho_MW(r) and rho_M31(r) side-by-side, we map:
#   - Spatial dominance regions: Which system dictates the local gravitational 
#     potential at a given radial distance and snapshot epoch.
#   - Mixing profiles: The radius where the two density curves merge into a single 
#     asymptotic profile marks the physical extent of spatial mixing.
#   - Core survival signatures: Highlights whether one system's inner core 
#     remains intact while the other is tidally disrupted (unequal merger trace).
#
# The bottom ratio panels (rho_MW / rho_M31) display the local mass balance:
#   - Ratio > 1 : Milky Way material dominates the radial zone.
#   - Ratio = 1 : Equal mass contribution (active mixing interface).
#   - Ratio < 1 : Andromeda material dominates.
#

print("[Fig 5] Generating progenitor density comparisons & radial mass ratios...")

fig28, axes28 = plt.subplots(
    2, len(PROFILE_INDICES), figsize=(15, 8), facecolor="#0d0d18",
    sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1, "wspace": 0.12},
)

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS)):
    ax_top = axes28[0, col]
    ax_bot = axes28[1, col]
    
    ax_top.set_facecolor("#0d0d18")
    ax_bot.set_facecolor("#0d0d18")
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_bot.set_xscale("log")

    rho_mw_row  = rho_mw_ts[k_idx, :]
    rho_m31_row = rho_m31_ts[k_idx, :]

    valid_mw  = np.isfinite(rho_mw_row)  & (rho_mw_row  > 0)
    valid_m31 = np.isfinite(rho_m31_row) & (rho_m31_row > 0)

    # ── Render Progenitor Densities ──
    if valid_mw.any():
        ax_top.plot(r_mid_sph[valid_mw], rho_mw_row[valid_mw],
                    color="#4a8fff", lw=1.8, label="MW")
    if valid_m31.any():
        ax_top.plot(r_mid_sph[valid_m31], rho_m31_row[valid_m31],
                    color="#ff5fa0", lw=1.8, label="M31")

    ax_top.set_title(label, fontsize=9)
    ax_top.set_xlim(R_BINS[0], R_BINS[-1])
    
    if col == 0:
        ax_top.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=9)
        ax_bot.set_ylabel(r"$\rho_{\rm MW}/\rho_{\rm M31}$", fontsize=9)
    ax_top.legend(fontsize=7)

    # ── Mass Balance Ratio Panels ──
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(valid_mw & valid_m31, rho_mw_row / rho_m31_row, np.nan)
        
    valid_r = np.isfinite(ratio)
    if valid_r.any():
        ax_bot.plot(r_mid_sph[valid_r], ratio[valid_r], color=color, lw=1.5)
        
    ax_bot.axhline(1.0, color="#555577", lw=0.8, ls="--")
    ax_bot.set_ylim(0.05, 20)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("r [kpc]", fontsize=9)

fig28.suptitle(r"Radial Progenitor Profile Comparison: $\rho_{\rm MW}(r)$ vs. $\rho_{\rm M31}(r)$", fontsize=12)
fig28.savefig(os.path.join(OUT_DIR, "density_mw_m31_comparison.png"),
              dpi=300, bbox_inches="tight", facecolor=fig28.get_facecolor())
plt.close(fig28)
print("  Saved: density_mw_m31_comparison.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 29 — DIAGNOSTIC 6: MASS-MIXING PROFILE CHRONOLOGY f_mix(r, t)        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Technical Objective:
# Track core interpenetration scales using the mass-mixing fraction index:
#   f_mix(r, t) = min(rho_MW, rho_M31) / rho_tot  ∈ [0, 0.5]
#
# Interpreting the Index:
#   - f_mix = 0.0 : Single progenitor dominance (no local structural overlap).
#   - f_mix = 0.5 : Homogeneous mixture (equal mass representation).
#
# The spatial gradient of f_mix over log radial bins exposes the active "mixing front".
# Mapping this as an (r, t) heatmap visualizes how the merger boundary expands
# progressively outward as the systems coalesce.
#

print("[Fig 6] Rendering dynamic mass-mixing fraction heatmap...")

fig29, (ax29a, ax29b) = plt.subplots(
    1, 2, figsize=(14, 6), facecolor="#0d0d18",
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
)
for ax in (ax29a, ax29b):
    ax.set_facecolor("#0d0d18")

# Coordinate mapping to match log-spaced radial bins
y_uniform_mix = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), 200)
f_mix_interp_map = np.zeros((len(y_uniform_mix), ns))

for snap_idx in range(ns):
    non_nan_mask = np.isfinite(f_mix_ts[snap_idx, :])
    if non_nan_mask.sum() > 2:
        f_mix_interp_map[:, snap_idx] = np.interp(
            np.log10(y_uniform_mix),
            np.log10(r_mid_sph[non_nan_mask]),
            f_mix_ts[snap_idx, non_nan_mask]
        )
    else:
        f_mix_interp_map[:, snap_idx] = np.nan

im29 = ax29a.imshow(
    f_mix_interp_map,
    aspect="auto", origin="lower",
    extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])],
    cmap="viridis", vmin=0.0, vmax=0.5,
)

ax29a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax29a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])

ax29a.set_xlabel(time_label, fontsize=10)
ax29a.set_ylabel("r [kpc]", fontsize=10)
ax29a.set_title(r"Mixing Front Boundary Evolution: $f_{\rm mix}(r,t)$", fontsize=10)

cb29 = fig29.colorbar(im29, ax=ax29a, pad=0.01)
cb29.set_label(r"$f_{\rm mix}$ (0.0 = Unmixed, 0.5 = Fully Blended)", fontsize=8)

# ── Right Panel: Time-Averaged Mixing Profile ──
f_mean = np.nanmean(f_mix_ts, axis=0)
valid_f = np.isfinite(f_mean)

ax29b.plot(f_mean[valid_f], r_mid_sph[valid_f], color="#00d4aa", lw=2.0)
ax29b.set_yscale("log")
ax29b.set_xlim(0, 0.52)
ax29b.axvline(0.5, color="#ffffff", lw=0.7, ls="--", alpha=0.4)
ax29b.set_xlabel(r"$\langle f_{\rm mix} \rangle_t$", fontsize=10)
ax29b.set_ylim(R_BINS[0], R_BINS[-1])
ax29b.tick_params(labelleft=False)
ax29b.set_title("Time-Avg", fontsize=10)

fig29.suptitle("Halo Progenitor Mass Mixing", fontsize=12)
fig29.savefig(os.path.join(OUT_DIR, "density_mixing_fraction.png"),
              dpi=300, bbox_inches="tight", facecolor=fig29.get_facecolor())
plt.close(fig29)
print("  Saved: density_mixing_fraction.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 30 — MATHEMATICAL PROFILE FITTING: NFW, EINASTO, HERNQUIST          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Mathematical Formulations:
#
# 1. Navarro-Frenk-White (NFW 1997)
#    $$\rho_{\rm NFW}(r) = \frac{\rho_s}{\frac{r}{r_s}\left(1 + \frac{r}{r_s}\right)^2}$$
#    Characterized by a central cusp profile index of -1 and outer envelope index of -3.
#
# 2. Einasto (Einasto 1965)
#    $$\rho_{\rm E}(r) = \rho_{-2} \exp\left(-\frac{2}{\alpha}\left[\left(\frac{r}{r_{-2}}\right)^\alpha - 1\right]\right)$$
#    Features a core transition without central cusps; governed by a shape index alpha.
#
# 3. Hernquist (Hernquist 1990)
#    $$\rho_{\rm H}(r) = \frac{M_{\rm tot}}{2\pi} \frac{a}{r(r + a)^3}$$
#    Asymptotes to -1 inside core radii and transitions to -4 in the outer boundaries.
#
# Fitting Strategy:
# Performed in logarithmic density space to resolve multi-decade dynamic ranges,
# ensuring inner core structures and diffuse outer halos receive balanced weights.
#

print("\n[Fitting] Estimating structural parameters (NFW, Einasto, Hernquist)...")

# ── Radial Profile Analytical Equations ────────────────────────────────────────

def nfw_density(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1.0 + x)**2)

def einasto_density(r, rho_m2, r_m2, alpha):
    return rho_m2 * np.exp(-(2.0 / alpha) * ((r / r_m2)**alpha - 1.0))

def hernquist_density(r, M_total, a):
    return (M_total / (2.0 * np.pi)) * a / (r * (r + a)**3)

# ── Optimization & Parameter Estimator Engine ─────────────────────────────────

def fit_density_models(rho_row, r_mid_loc, r_min=R_FIT_MIN_KPC, r_max=R_FIT_MAX_KPC):
    """
    Fits radial mass profiles using log-space least-squares minimization.
    """
    mask = (r_mid_loc >= r_min) & (r_mid_loc <= r_max) & np.isfinite(rho_row) & (rho_row > 0)
    r_fit   = r_mid_loc[mask]
    rho_fit = rho_row[mask]

    if len(r_fit) < 5:
        empty = {"popt": None, "chi2": np.nan, "success": False}
        return {"nfw": empty, "einasto": empty, "hernquist": empty}

    ln_rho = np.log(rho_fit)
    results = {}

    # ── (a) NFW Parameter Estimation ──
    # Guess parameters estimated from core scaling
    rho_s0 = np.exp(ln_rho.max()) * 5.0
    r_s0   = 30.0
    try:
        popt_nfw, _ = curve_fit(
            lambda r, rs, rss: np.log(nfw_density(r, rs, rss)),
            r_fit, ln_rho,
            p0=[rho_s0, r_s0],
            bounds=([1e2, 0.1], [1e16, 300.0]),
            maxfev=5000,
        )
        pred_nfw = np.log(nfw_density(r_fit, *popt_nfw))
        chi2_nfw = np.sum((ln_rho - pred_nfw)**2) / max(1, len(r_fit) - 2)
        results["nfw"] = {"popt": popt_nfw, "chi2": chi2_nfw, "success": True}
    except Exception:
        results["nfw"] = {"popt": None, "chi2": np.nan, "success": False}

    # ── (b) Einasto Parameter Estimation ──
    # Locates physical scale radius r_-2 where local power-law slope = -2
    try:
        # Initial guess from the profile median and slope markers
        rho_m2_0 = np.exp(np.median(ln_rho))
        r_m2_0   = 30.0
        alpha0   = 0.18
        popt_ein, _ = curve_fit(
            lambda r, rm2, rrm2, a: np.log(einasto_density(r, rm2, rrm2, a)),
            r_fit, ln_rho,
            p0=[rho_m2_0, r_m2_0, alpha0],
            bounds=([1e2, 0.1, 0.05], [1e16, 300.0, 1.0]),
            maxfev=10000,
        )
        pred_ein = np.log(einasto_density(r_fit, *popt_ein))
        chi2_ein = np.sum((ln_rho - pred_ein)**2) / max(1, len(r_fit) - 3)
        results["einasto"] = {"popt": popt_ein, "chi2": chi2_ein, "success": True}
    except Exception:
        results["einasto"] = {"popt": None, "chi2": np.nan, "success": False}

    # ── (c) Hernquist Parameter Estimation ──
    M_tot0 = rho_fit[0] * 4.0 * np.pi * r_fit[0]**3 * 3.0
    a0     = 30.0
    try:
        popt_her, _ = curve_fit(
            lambda r, Mt, aa: np.log(hernquist_density(r, Mt, aa)),
            r_fit, ln_rho,
            p0=[M_tot0, a0],
            bounds=([1e6, 0.1], [1e16, 300.0]),
            maxfev=5000,
        )
        pred_her = np.log(hernquist_density(r_fit, *popt_her))
        chi2_her = np.sum((ln_rho - pred_her)**2) / max(1, len(r_fit) - 2)
        results["hernquist"] = {"popt": popt_her, "chi2": chi2_her, "success": True}
    except Exception:
        results["hernquist"] = {"popt": None, "chi2": np.nan, "success": False}

    return results


# ── Run fits on every STEP_FIT-th snapshot ────────────────────────────────────
fit_snap_nums = SNAPSHOTS[::STEP_FIT]
n_fit_snaps   = len(fit_snap_nums)
fit_snap_idx  = {snap: ii for ii, snap in enumerate(fit_snap_nums)}

chi2_nfw_arr  = np.full(n_fit_snaps, np.nan)
chi2_ein_arr  = np.full(n_fit_snaps, np.nan)
chi2_her_arr  = np.full(n_fit_snaps, np.nan)
alpha_ein_arr = np.full(n_fit_snaps, np.nan)
rs_nfw_arr    = np.full(n_fit_snaps, np.nan)

t_fit_start = time.perf_counter()
for snap_num in fit_snap_nums:
    ii = fit_snap_idx[snap_num]
    i  = np.where(SNAPSHOTS == snap_num)[0]
    if len(i) == 0:
        continue
    i = i[0]
    res = fit_density_models(rho_ts[i, :], r_mid_sph)
    
    chi2_nfw_arr[ii] = res["nfw"]["chi2"]
    chi2_ein_arr[ii] = res["einasto"]["chi2"]
    chi2_her_arr[ii] = res["hernquist"]["chi2"]
    
    if res["einasto"]["success"]:
        alpha_ein_arr[ii] = res["einasto"]["popt"][2]
    if res["nfw"]["success"]:
        rs_nfw_arr[ii] = res["nfw"]["popt"][1]

print(f"  Fits resolved in {time.perf_counter() - t_fit_start:.0f}s")


# ── Figure: model fits at five key epochs ─────────────────────────────────────
print("[Fig 7] Overplotting analytical profile models against measured profiles...")

fig30, axes30 = plt.subplots(1, 5, figsize=(18, 6), facecolor="#0d0d18",
                             sharey=True, gridspec_kw={"wspace": 0.06})

r_plot = np.logspace(np.log10(R_FIT_MIN_KPC), np.log10(R_FIT_MAX_KPC), 200)

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS)):
    ax = axes30[col]
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log")
    ax.set_yscale("log")

    rho_row = rho_ts[k_idx, :]
    valid   = np.isfinite(rho_row) & (rho_row > 0)
    ax.scatter(r_mid_sph[valid], rho_row[valid],
               color="#aaaacc", s=12, zorder=3, label="Data", alpha=0.8)

    # ── Model Curve Evaluation ──
    res = fit_density_models(rho_row, r_mid_sph)
    model_specs = [
        ("nfw",       nfw_density,       "#ff9944", "NFW"),
        ("einasto",   einasto_density,   "#00d4aa", "Einasto"),
        ("hernquist", hernquist_density, "#aa55ff", "Hernquist"),
    ]
    for model_key, model_fn, mc, ml in model_specs:
        if res[model_key]["success"]:
            ax.plot(r_plot, model_fn(r_plot, *res[model_key]["popt"]),
                    color=mc, lw=1.8, label=f"{ml} χ²={res[model_key]['chi2']:.2f}")

    ax.set_xlabel("r [kpc]", fontsize=9)
    ax.set_title(label, fontsize=9)
    ax.set_xlim(R_FIT_MIN_KPC, R_FIT_MAX_KPC)
    if col == 0:
        ax.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=9)
    ax.legend(fontsize=6)

fig30.suptitle("Density Profile Fitting: NFW vs. Einasto vs. Hernquist Models", fontsize=12)
fig30.savefig(os.path.join(OUT_DIR, "density_nfw_fit.png"),
              dpi=300, bbox_inches="tight", facecolor=fig30.get_facecolor())
plt.close(fig30)
print("  Saved: density_nfw_fit.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 31 — DIAGNOSTIC 8: SPATIOTEMPORAL FIT RESIDUES MAP IN (r, t)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Technical Objective:
# Evaluate spatial fit accuracy as a function of time.
#
# The localized residual index is defined as:
#   $$\Delta(r, t) = \frac{\rho_{\rm meas} - \rho_{\rm NFW}}{\rho_{\rm NFW}}$$
#
# Values are mapped to answer structural questions:
#   - Positive Residuals (Excess, Red): Higher concentration zones than NFW models
#     can accommodate (such as post-merger cores or tidal structures).
#   - Negative Residuals (Deficit, Blue): Extreme outer-envelope depletion 
#     indicating extensive tidal mass-stripping.
#

print("[Fig 8] Plotting Spatiotemporal NFW Fit Residual Heatmaps...")

resid_ts = np.full((n_fit_snaps, nb_sph), np.nan)

for snap_num in fit_snap_nums:
    ii = fit_snap_idx[snap_num]
    i  = np.where(SNAPSHOTS == snap_num)[0]
    if len(i) == 0:
        continue
    i = i[0]
    rho_row = rho_ts[i, :]
    res     = fit_density_models(rho_row, r_mid_sph)
    if res["nfw"]["success"]:
        rho_nfw_pred = nfw_density(r_mid_sph, *res["nfw"]["popt"])
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = np.where(
                np.isfinite(rho_row) & (rho_row > 0) & (rho_nfw_pred > 0),
                (rho_row - rho_nfw_pred) / rho_nfw_pred,
                np.nan,
            )
        resid_ts[ii, :] = delta

# Map fitting snap timeline
time_fit_snaps = np.array([
    time_arr[np.where(SNAPSHOTS == s)[0][0]]
    if len(np.where(SNAPSHOTS == s)[0]) > 0 else np.nan
    for s in fit_snap_nums
])

fig31, (ax31a, ax31b) = plt.subplots(
    1, 2, figsize=(14, 6), facecolor="#0d0d18",
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
)
for ax in (ax31a, ax31b):
    ax.set_facecolor("#0d0d18")

t_fit_min, t_fit_max = np.nanmin(time_fit_snaps), np.nanmax(time_fit_snaps)

# Rescale and interpolate spatial dimensions logarithmic bins
resid_interp_map = np.zeros((len(y_uniform_mix), n_fit_snaps))
for ii in range(n_fit_snaps):
    non_nan_mask = np.isfinite(resid_ts[ii, :])
    if non_nan_mask.sum() > 2:
        resid_interp_map[:, ii] = np.interp(
            np.log10(y_uniform_mix),
            np.log10(r_mid_sph[non_nan_mask]),
            resid_ts[ii, non_nan_mask]
        )
    else:
        resid_interp_map[:, ii] = np.nan

im31 = ax31a.imshow(
    np.clip(resid_interp_map, -1.5, 1.5),
    aspect="auto", origin="lower",
    extent=[t_fit_min, t_fit_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])],
    cmap="seismic", vmin=-1.5, vmax=1.5,
)

ax31a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax31a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])

ax31a.set_xlabel(time_label, fontsize=10)
ax31a.set_ylabel("r [kpc]", fontsize=10)
ax31a.set_title(r"NFW Profiling Residual Topology: $\Delta = \frac{\rho_{\rm meas} - \rho_{\rm NFW}}{\rho_{\rm NFW}}$", fontsize=10)

cb31 = fig31.colorbar(im31, ax=ax31a, pad=0.01)
cb31.set_label(r"$\Delta$ (Blue = Mass Deficit, Red = Mass Excess)", fontsize=8)

# ── Right Panel: Time-Averaged Residual Profile ──
resid_mean = np.nanmean(resid_ts, axis=0)
valid_res  = np.isfinite(resid_mean)

ax31b.plot(resid_mean[valid_res], r_mid_sph[valid_res], color="#e8673a", lw=2.0)
ax31b.axvline(0, color="#555577", lw=0.8, ls="--")
ax31b.set_yscale("log")
ax31b.set_ylim(R_BINS[0], R_BINS[-1])
ax31b.set_xlabel(r"$\langle\Delta\rangle_t$", fontsize=10)
ax31b.tick_params(labelleft=False)
ax31b.set_title("Time-Avg", fontsize=10)

fig31.suptitle("Spatiotemporal Model Residual Tracking", fontsize=12)
fig31.savefig(os.path.join(OUT_DIR, "density_fit_residuals.png"),
              dpi=300, bbox_inches="tight", facecolor=fig31.get_facecolor())
plt.close(fig31)
print("  Saved: density_fit_residuals.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 32 — DIAGNOSTIC 9: PROFILE TELEMETRY COMPARISON OVER TIME           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Fig 9] Plotting analytical parameter evolutionary trajectories...")

fig32, axes32 = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d0d18",
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
axes32 = axes32.flatten()
for ax in axes32:
    ax.set_facecolor("#0d0d18")

# (a) Half-Mass Radius Evolution
axes32[0].plot(time_arr, r_half_3d_arr,   color="#4a8fff", lw=1.8, label=r"$r_{1/2,\ 3D}$")
axes32[0].plot(time_arr, r_half_proj_arr, color="#00d4aa", lw=1.8, ls="--", label=r"$R_{1/2,\ \rm proj}$")
axes32[0].set_ylabel("Half-Mass Radius [kpc]", fontsize=9)
axes32[0].set_title("Dynamic Mass Radius", fontsize=10)
axes32[0].legend(fontsize=8)

# (b) NFW scale radius r_s over time
time_fit_axis = time_fit_snaps
axes32[1].plot(time_fit_axis, rs_nfw_arr, color="#ff9944", lw=1.8, label=r"NFW $r_s$")
axes32[1].set_ylabel(r"$r_s$ [kpc]", fontsize=9)
axes32[1].set_title("NFW Scale Boundary Scale", fontsize=10)
axes32[1].legend(fontsize=8)

# (c) Einasto shape index alpha over time
axes32[2].plot(time_fit_axis, alpha_ein_arr, color="#aa55ff", lw=1.8, label=r"Einasto $\alpha$")
axes32[2].axhline(0.18, color="#555577", lw=0.8, ls="--", label="Typical CDM Halo (α ≈ 0.18)")
axes32[2].set_ylabel(r"Einasto $\alpha$", fontsize=9)
axes32[2].set_title("Einasto Index Tracking", fontsize=10)
axes32[2].legend(fontsize=8)

# (d) Analytical Model goodness-of-fit Comparison
axes32[3].semilogy(time_fit_axis, chi2_nfw_arr,  color="#ff9944", lw=1.5, label="NFW")
axes32[3].semilogy(time_fit_axis, chi2_ein_arr,  color="#00d4aa", lw=1.5, label="Einasto")
axes32[3].semilogy(time_fit_axis, chi2_her_arr,  color="#aa55ff", lw=1.5, label="Hernquist")
axes32[3].axhline(1.0, color="#ffffff", lw=0.7, ls="--", alpha=0.4)
axes32[3].set_ylabel(r"Reduced $\chi^2$ (Log space)", fontsize=9)
axes32[3].set_title("Model Fit Optimization Accuracy", fontsize=10)
axes32[3].legend(fontsize=8)

for ax in axes32:
    ax.set_xlabel(time_label, fontsize=9)

fig32.suptitle("Model Structural Parameter Chronology", fontsize=12)
fig32.savefig(os.path.join(OUT_DIR, "density_halfmass_evolution.png"),
              dpi=300, bbox_inches="tight", facecolor=fig32.get_facecolor())
plt.close(fig32)
print("  Saved: density_halfmass_evolution.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 33 — DENSITY PROFILE DYNAMIC ANIMATION ENGINE                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Implements a synchronized multi-panel MP4 generator tracking:
#   - Left   : Total spherical density profile ρ(r) (current step colored 
#              chronologically, over plotting N_GHOST historic lines to 
#              provide visual trajectory mapping).
#   - Center : Progenitor-split radial mass profiles.
#   - Right  : Local power-law density index Gamma(r).
#

print("\n[Anim 1] Initializing multi-panel profile animation frame-loop...")

N_GHOST   = 15
ANIM_IDXS = np.arange(0, ns, ANIM_STEP)
N_FRAMES  = len(ANIM_IDXS)
cmap_time = plt.cm.plasma

fig33, axes33 = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="#0d0d18",
                             gridspec_kw={"wspace": 0.32})
for ax in axes33:
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(R_BINS[0], R_BINS[-1])

ax_rho33, ax_gal33, ax_slope33 = axes33

# Establish dynamic canvas boundaries from finite measurements
rho_finite = rho_ts[np.isfinite(rho_ts) & (rho_ts > 0)]
rho_ymin, rho_ymax = (rho_finite.min() * 0.3, rho_finite.max() * 3.0) if rho_finite.size > 0 else (1e2, 1e12)

ax_rho33.set_ylim(rho_ymin, rho_ymax)
ax_gal33.set_ylim(rho_ymin, rho_ymax)
ax_slope33.set_yscale("linear")
ax_slope33.set_ylim(-5.0, 1.0)

ax_rho33.set_xlabel("r [kpc]"); ax_rho33.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]")
ax_rho33.set_title(r"Trajectory History: $\rho(r)$")
ax_gal33.set_xlabel("r [kpc]"); ax_gal33.set_title(r"Progenitor Splits: $\rho_{\rm MW}$ vs. $\rho_{\rm M31}$")
ax_slope33.set_xlabel("r [kpc]"); ax_slope33.set_ylabel(r"$\Gamma$")
ax_slope33.set_title(r"Radial Slope Index: $\Gamma(r)$")
ax_slope33.axhline(-1, color="#555577", lw=0.6, ls=":")
ax_slope33.axhline(-3, color="#555577", lw=0.6, ls=":")

title33 = fig33.suptitle("", fontsize=11, color="#c8c8e8")

# Initialize structural artist lines
ghost_lines  = [ax_rho33.plot([], [], lw=0.8, alpha=0.0)[0] for _ in range(N_GHOST)]
main_line33, = ax_rho33.plot([], [], lw=2.2, color="white", zorder=5)
mw_line33,   = ax_gal33.plot([], [], lw=2.0, color="#4a8fff", label="MW")
m31_line33,  = ax_gal33.plot([], [], lw=2.0, color="#ff5fa0", label="M31")
slope_line33,= ax_slope33.plot([], [], lw=2.0, color="#e8673a")
ax_gal33.legend(fontsize=8)

def _update_density_anim(frame_idx):
    snap_i = ANIM_IDXS[frame_idx]
    color  = cmap_time(frame_idx / N_FRAMES)

    def _xy_clean(arr, r):
        val_mask = np.isfinite(arr) & (arr > 0)
        return r[val_mask], arr[val_mask]

    # Current Profiles Updates
    rx, ry = _xy_clean(rho_ts[snap_i, :], r_mid_sph)
    main_line33.set_data(rx, ry)
    main_line33.set_color(color)

    # Historic Ghost Profiling Trace Updates
    for g, ghost in enumerate(ghost_lines):
        past_idx = frame_idx - (N_GHOST - g)
        if past_idx < 0:
            ghost.set_data([], [])
            continue
        past_snap = ANIM_IDXS[past_idx]
        px, py = _xy_clean(rho_ts[past_snap, :], r_mid_sph)
        past_color = cmap_time(past_idx / N_FRAMES)
        ghost.set_data(px, py)
        ghost.set_color(past_color)
        ghost.set_alpha(0.08 + 0.07 * g)

    # Progenitor Splits Updates
    mwx, mwy   = _xy_clean(rho_mw_ts[snap_i, :], r_mid_sph)
    m31x, m31y = _xy_clean(rho_m31_ts[snap_i, :], r_mid_sph)
    mw_line33.set_data(mwx, mwy)
    m31_line33.set_data(m31x, m31y)

    # Gradient Slope Update
    slope_idx = np.isfinite(Gamma_ts[snap_i, :])
    slope_line33.set_data(r_mid_sph[slope_idx], Gamma_ts[snap_i, :][slope_idx])

    t_val = time_arr[snap_i]
    t_str = f"{t_val:.2f} Gyr" if time_is_gyr else f"Snap {SNAPSHOTS[snap_i]}"
    title33.set_text(f"MW–M31 Spatial Coalescence Profiles  ·  {t_str}")

    return [main_line33, mw_line33, m31_line33, slope_line33] + ghost_lines

ani33 = animation.FuncAnimation(
    fig33, _update_density_anim, frames=N_FRAMES,
    interval=1000 // ANIM_FPS, blit=True,
)
writer33 = animation.FFMpegWriter(fps=ANIM_FPS, bitrate=ANIM_BITRATE,
                                  metadata=dict(title="MW-M31 Density Profile Evolution"))
ani33.save(os.path.join(OUT_DIR, "density_animation.mp4"), writer=writer33, dpi=ANIM_DPI)
plt.close(fig33)
print("  Saved: density_animation.mp4")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 34 — 2D SURFACE DENSITY MORPHOLOGY ANIMATION                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("[Anim 2] Initializing 2D morphology projection spatial animation...")

anim_map_idxs = np.arange(n_maps)
N_MAP_FRAMES  = len(anim_map_idxs)

Sigma_finite = maps_3d[np.isfinite(maps_3d) & (maps_3d > 0)]
if Sigma_finite.size > 0:
    vmin_2d = np.log10(np.percentile(Sigma_finite, 5))
    vmax_2d = np.log10(np.percentile(Sigma_finite, 99.9))
else:
    vmin_2d, vmax_2d = 4.0, 9.5

fig34, axes34 = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="#0d0d18",
                             gridspec_kw={"wspace": 0.05})
for ax in axes34:
    ax.set_facecolor("#0d0d18")

def _log10_smooth(arr):
    """
    Constructs normalized log-smoothed projections, mitigating unpopulated edge leaking.
    """
    is_val = np.isfinite(arr)
    smoothed = gaussian_filter(np.where(is_val, arr, 0.0), sigma=MAP_SMOOTH_SIGMA)
    smoothed_norm = gaussian_filter(is_val.astype(float), sigma=MAP_SMOOTH_SIGMA)
    with np.errstate(invalid="ignore", divide="ignore"):
        cleaned = np.where(smoothed_norm > 0.1, smoothed / smoothed_norm, np.nan)
        return np.where(cleaned > 0, np.log10(cleaned), np.nan)

# Initialize plots using index zero configurations
first_total = _log10_smooth(maps_3d[0])
first_mw    = _log10_smooth(maps_mw[0])
first_m31   = _log10_smooth(maps_m31[0])

im34_tot = axes34[0].imshow(first_total.T, origin="lower", aspect="equal",
                            extent=MAP_EXTENT, cmap="inferno", vmin=vmin_2d, vmax=vmax_2d)
im34_mw  = axes34[1].imshow(first_mw.T,    origin="lower", aspect="equal",
                            extent=MAP_EXTENT, cmap="Blues_r", vmin=vmin_2d, vmax=vmax_2d)
im34_m31 = axes34[2].imshow(first_m31.T,   origin="lower", aspect="equal",
                            extent=MAP_EXTENT, cmap="Reds_r", vmin=vmin_2d, vmax=vmax_2d)

for ax, lbl in zip(axes34, ["Total Σ", "MW Only", "M31 Only"]):
    ax.set_xlabel("x [kpc]", fontsize=9)
    ax.set_title(lbl, fontsize=10, color="#c8c8e8")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
axes34[0].set_ylabel("y [kpc]", fontsize=9)

title34 = fig34.suptitle("", fontsize=11, color="#c8c8e8")

def _update_map_anim(frame_idx):
    mi = anim_map_idxs[frame_idx]
    im34_tot.set_data(_log10_smooth(maps_3d[mi]).T)
    im34_mw.set_data(_log10_smooth(maps_mw[mi]).T)
    im34_m31.set_data(_log10_smooth(maps_m31[mi]).T)
    t_val = time_maps[mi]
    t_str = f"{t_val:.2f} Gyr" if time_is_gyr else f"Snap {map_snap_nums[mi]}"
    title34.set_text(f"2D Projected System Surface Densities  ·  {t_str}")
    return [im34_tot, im34_mw, im34_m31]

ani34 = animation.FuncAnimation(
    fig34, _update_map_anim, frames=N_MAP_FRAMES,
    interval=1000 // ANIM_FPS, blit=True,
)
ani34.save(os.path.join(OUT_DIR, "density_2d_animation.mp4"), writer=writer33, dpi=ANIM_DPI)
plt.close(fig34)
print("  Saved: density_2d_animation.mp4")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 35 — MASTER PIPELINE SUMMARY GRID                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Constructing a comprehensive 4x2 final master summary graphic.
# Combines 1D radial profiles, spatiotemporal diagnostics, parameter traces,
# and projection snapshots into a single publication-ready poster panel.
#

print("[Fig 12] Drawing complete master publication overview panel...")

fig35 = plt.figure(figsize=(16, 20), facecolor="#0d0d18")
gs35  = gridspec.GridSpec(4, 2, figure=fig35, hspace=0.42, wspace=0.32,
                          left=0.08, right=0.97, top=0.95, bottom=0.05)

BG_COLOR = "#0d0d18"
MUTED_TX = "#7070a0"

def _sax_styled(fig, gs, r, c, log_x=True, log_y=True):
    ax = fig.add_subplot(gs[r, c])
    ax.set_facecolor(BG_COLOR)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")
    ax.tick_params(colors="#9090b0", labelsize=8)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    return ax

# ── Panel (0,0): Radial Spherical Densities ──
ax00 = _sax_styled(fig35, gs35, 0, 0)
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y = rho_ts[k_idx, :]
    v = np.isfinite(y) & (y > 0)
    if v.any():
        ax00.plot(r_mid_sph[v], y[v], color=color, lw=1.5, label=label)
ax00.set_xlabel("r [kpc]", fontsize=8, color=MUTED_TX)
ax00.set_ylabel(r"$\rho$ [M$_\odot$/kpc³]", fontsize=8, color=MUTED_TX)
ax00.set_title(r"$\rho(r)$ Evolution (3D)", fontsize=9)
ax00.set_xlim(R_BINS[0], R_BINS[-1])
ax00.legend(fontsize=6)

# ── Panel (0,1): Radial Surface Densities ──
ax01 = _sax_styled(fig35, gs35, 0, 1)
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y = Sigma_ts[k_idx, :]
    v = np.isfinite(y) & (y > 0)
    if v.any():
        ax01.plot(r_mid_proj[v], y[v], color=color, lw=1.5, label=label)
ax01.set_xlabel("R [kpc]", fontsize=8, color=MUTED_TX)
ax01.set_ylabel(r"$\Sigma$ [M$_\odot$/kpc²]", fontsize=8, color=MUTED_TX)
ax01.set_title(r"$\Sigma(R)$ Evolution (Projected)", fontsize=9)
ax01.set_xlim(R_PROJ_BINS[0], R_PROJ_BINS[-1])
ax01.legend(fontsize=6)

# ── Panel (1,0): Local Gradient Index Gamma ──
ax10 = _sax_styled(fig35, gs35, 1, 0, log_x=False, log_y=False)
im10 = ax10.imshow(
    gamma_interp_map, aspect="auto", origin="lower",
    extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])],
    cmap="bwr", vmin=-4.0, vmax=0.5
)
ax10.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax10.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
ax10.set_xlabel(time_label, fontsize=8, color=MUTED_TX)
ax10.set_ylabel("r [kpc]", fontsize=8, color=MUTED_TX)
ax10.set_title(r"$\Gamma(r,t)$ Slope Local Indices", fontsize=9)
fig35.colorbar(im10, ax=ax10, label=r"$\Gamma$", shrink=0.8)

# ── Panel (1,1): Mass-Mixing Boundaries ──
ax11 = _sax_styled(fig35, gs35, 1, 1, log_x=False, log_y=False)
im11 = ax11.imshow(
    f_mix_interp_map, aspect="auto", origin="lower",
    extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])],
    cmap="viridis", vmin=0.0, vmax=0.5
)
ax11.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax11.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
ax11.set_xlabel(time_label, fontsize=8, color=MUTED_TX)
ax11.set_ylabel("r [kpc]", fontsize=8, color=MUTED_TX)
ax11.set_title(r"$f_{\rm mix}(r,t)$ Mass-Mixing Fronts", fontsize=9)
fig35.colorbar(im11, ax=ax11, label=r"$f_{\rm mix}$", shrink=0.8)

# ── Panel (2,0): Central Core Densities vs. Half-Mass Limits ──
ax20 = _sax_styled(fig35, gs35, 2, 0, log_x=False, log_y=False)
ax20_r = ax20.twinx()
ax20_r.tick_params(colors="#9090b0", labelsize=8)

ax20.plot(time_arr, np.log10(np.where(rho0_arr > 0, rho0_arr, np.nan)),
          color="#e8673a", lw=1.5, label=r"$\log_{10}\rho_0$")
ax20_r.plot(time_arr, r_half_3d_arr, color="#4a8fff", lw=1.5, ls="--",
            label=r"$r_{1/2,\ 3D}$")
ax20.set_xlabel(time_label, fontsize=8, color=MUTED_TX)
ax20.set_ylabel(r"$\log_{10}\rho_0$ [M$_\odot$ kpc$^{-3}$]", fontsize=8, color="#e8673a")
ax20_r.set_ylabel(r"$r_{\rm half}$ [kpc]", fontsize=8, color="#4a8fff")
ax20.set_title("Core Compaction vs. Global Core Radius", fontsize=9)

# ── Panel (2,1): Model chi-squared evolution comparison ──
ax21 = _sax_styled(fig35, gs35, 2, 1, log_x=False, log_y=True)
ax21.plot(time_fit_axis, chi2_nfw_arr,  color="#ff9944", lw=1.2, label="NFW")
ax21.plot(time_fit_axis, chi2_ein_arr,  color="#00d4aa", lw=1.2, label="Einasto")
ax21.plot(time_fit_axis, chi2_her_arr,  color="#aa55ff", lw=1.2, label="Hernquist")
ax21.axhline(1.0, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
ax21.set_xlabel(time_label, fontsize=8, color=MUTED_TX)
ax21.set_ylabel(r"Reduced $\chi^2$", fontsize=8, color=MUTED_TX)
ax21.set_title("Analytical Model Profiling Fit Accuracy", fontsize=9)
ax21.legend(fontsize=6)

# ── Panel (3,0): Initial 2D Morphological Projections ──
ax30 = fig35.add_subplot(gs35[3, 0])
ax30.set_facecolor(BG_COLOR)
early_mi = 0
ax30.imshow(_log10_smooth(maps_3d[early_mi]).T, origin="lower", aspect="equal",
            extent=MAP_EXTENT, cmap="inferno", vmin=vmin_2d, vmax=vmax_2d)
ax30.set_title(f"Early Epoch Morphology (t={time_maps[early_mi]:.2f} Gyr)" if time_is_gyr else "Early Epoch Topology", fontsize=9)
ax30.set_xlabel("x [kpc]", fontsize=8, color=MUTED_TX)
ax30.set_ylabel("y [kpc]", fontsize=8, color=MUTED_TX)
ax30.tick_params(colors="#9090b0", labelsize=8)

# ── Panel (3,1): Final Coalesced 2D Morphological Projections ──
ax31 = fig35.add_subplot(gs35[3, 1])
ax31.set_facecolor(BG_COLOR)
late_mi = n_maps - 1
ax31.imshow(_log10_smooth(maps_3d[late_mi]).T, origin="lower", aspect="equal",
            extent=MAP_EXTENT, cmap="inferno", vmin=vmin_2d, vmax=vmax_2d)
ax31.set_title(f"Post-Coalescence Morphology (t={time_maps[late_mi]:.2f} Gyr)" if time_is_gyr else "Post-Coalescence Topology", fontsize=9)
ax31.set_xlabel("x [kpc]", fontsize=8, color=MUTED_TX)
ax31.tick_params(colors="#9090b0", labelsize=8)

fig35.suptitle("MW–M31 Spatial Coalescence Diagnostics Summary Panel",
               fontsize=14, color="#c8c8e8", fontweight="bold")
fig35.savefig(os.path.join(OUT_DIR, "density_summary_panel.png"),
              dpi=200, bbox_inches="tight", facecolor=fig35.get_facecolor())
plt.close(fig35)
print("  Saved: density_summary_panel.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 36 — FILE SYSTEM DESTRUCTION & PIPELINE MANIFEST                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Clears the active virtual filesystem workspace memory allocations.
# Deferring execution to this point ensures all spatial projections are fully written.
#

shutil.rmtree(tmpdir, ignore_errors=True)
print(f"\n[cleanup] Dereferencing dynamic local file system: {tmpdir}")

print("\n" + "="*80)
print("  SYSTEM RUN OUTPUT FILESYSTEM MANIFEST")
print("="*80)
print(f"  {'File Identifier':<48} {'Memory (MB)':>12}  Component Classification")
print(f"  {'-'*48} {'-'*12}  {'-'*24}")

total_mb = 0.0
for fn in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, fn)
    mb = os.path.getsize(fp) / 1e6
    total_mb += mb
    kind = "Stream Media (mp4)" if fn.endswith(".mp4") else "Static Image (png)"
    print(f"  {fn:<48} {mb:12.2f}  {kind}")

print(f"  {'-'*48} {'-'*12}")
print(f"  {'TOTAL WRITE VOLUME':<48} {total_mb:12.2f}")
print("="*80)
print("\n[SUCCESS] Profile & spatial density diagnostics engine complete.")
