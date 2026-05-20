"""
===============================================================================
SECTION 30 — LUMINOSITY PROFILE FITTING
===============================================================================
Author  : Abhinav Vatsa

Continuation of the MW–M31 analysis pipeline.  All globals (SNAPSHOTS, ns,
R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL,
G_KPC_KMS2_MSUN, PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr,
time_label, time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

Physical motivation
-------------------
This is an N-body dark matter simulation — there are no stars and therefore
no luminosity in the strict observational sense.  However, luminosity profile
fitting is still directly applicable here for two reasons:

  1. MASS-TO-LIGHT RATIO PROXY
     Under a constant (or radially varying) stellar mass-to-light ratio
     Υ = M_*/L, the projected surface mass density Σ(R) is directly
     proportional to the surface brightness I(R):
         I(R) = Σ(R) / Υ
     Fitting luminosity profile models to Σ(R) therefore gives the same
     structural parameters (R_e, n, I_e) as would be obtained from actual
     photometry.  This connects the simulation directly to observational
     diagnostics used for real galaxies (e.g. from HST, JWST imaging).

  2. MERGER REMNANT CLASSIFICATION
     The Sérsic index n is the primary observational classifier of galaxy
     morphology.  Tracking n(t) across the merger documents the
     transformation from:
         n ≈ 1  (exponential disk — pre-merger MW and M31)
         n ≈ 4  (de Vaucouleurs elliptical — final merger remnant)
     This is the N-body counterpart of the observed morphological
     transformation in major galaxy mergers.

Models fitted
-------------
  1. Sérsic profile           I(R) = I_e exp{−b_n [(R/R_e)^{1/n} − 1]}
  2. de Vaucouleurs (n=4)     Special case of Sérsic, 1 fewer free parameter
  3. Exponential disk (n=1)   Special case of Sérsic
  4. Core-Sérsic              Sérsic with a central power-law core flattening
                               I(R) = I' [1 + (R_b/R)^α]^{γ/α} exp{−b[(R^α + R_b^α)/R_e^α]^{1/(nα)}}
  5. Double Sérsic            Sum of two Sérsic components (bulge + halo)
  6. King profile             I(R) = I_0 [1/sqrt(1 + (R/R_c)²) − 1/sqrt(1 + (R_t/R_c)²)]²
                               (useful for the dense core post-merger)

All models are fit in log-space to Σ(R) projected surface density profiles
computed directly from the N-body snapshot, using the projected R_BINS from
the density pipeline.  The assumed mass-to-light ratio is Υ = 1 M_sun/L_sun
throughout (i.e. we report surface mass density but label axes in both units).

Outputs
-------
  section30_sersic_fits.png            Sérsic fits at 5 epochs
  section30_devauc_fits.png            de Vaucouleurs fits at 5 epochs
  section30_doublesersic_fits.png      Double-Sérsic decomposition at 5 epochs
  section30_coresersic_fits.png        Core-Sérsic fits at 5 epochs
  section30_king_fits.png              King profile fits at 5 epochs
  section30_n_evolution.png            Sérsic n(t) and R_e(t) evolution
  section30_component_evolution.png    Bulge/halo component evolution
  section30_residuals_heatmap.png      Fit residuals Δ(R,t) heatmap
  section30_model_comparison.png       All models at one epoch side by side
  section30_summary_panel.png          Master 4-panel summary

===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit, minimize
import os
import time
import warnings


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §30.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Fitting radius range in kpc — same as density pipeline.
FIT_RMIN = 0.5
FIT_RMAX = 100.0

# Assumed constant stellar mass-to-light ratio [M_sun / L_sun].
# Under this assumption Σ [M_sun kpc^{-2}] / UPSILON = I [L_sun kpc^{-2}].
UPSILON = 1.0

# Projected surface density bins — reuse R_PROJ_BINS from density pipeline
# if available, otherwise use R_BINS.
try:
    R_PROJ = R_PROJ_BINS
    r_proj_mid = 0.5 * (R_PROJ[:-1] + R_PROJ[1:])
    nb_proj    = len(r_proj_mid)
except NameError:
    R_PROJ     = R_BINS
    r_proj_mid = r_mid_sph
    nb_proj    = nb_sph

# Temporal step for luminosity profile fits.
LFIT_STEP = 4    # fit every 4th snapshot

print("\n" + "="*80)
print("  SECTION 30 · Luminosity Profile Fitting")
print("="*80)
print(f"  Fitting range  : {FIT_RMIN}–{FIT_RMAX} kpc")
print(f"  Υ (M/L)        : {UPSILON} M_sun/L_sun")
print(f"  Fit step       : every {LFIT_STEP} snapshots")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §30.1 — PROFILE MODEL DEFINITIONS                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def b_n(n):
    """
    Ciotti & Bertin (1999) approximation for the Sérsic normalisation constant.
    Valid for n > 0.5.  b_n is defined so that R_e encloses half the total flux.
    """
    n = max(float(n), 0.5)
    return 2.0*n - 1.0/3.0 + 4.0/(405.0*n) + 46.0/(25515.0*n**2)


def sersic(R, I_e, R_e, n):
    """
    Sérsic surface brightness profile.
    I(R) = I_e × exp{−b_n × [(R/R_e)^{1/n} − 1]}
    """
    n   = max(float(n), 0.5)
    R_e = max(float(R_e), 1e-3)
    return I_e * np.exp(-b_n(n) * ((R / R_e)**(1.0/n) - 1.0))


def devauc(R, I_e, R_e):
    """de Vaucouleurs profile — Sérsic with n=4 fixed."""
    return sersic(R, I_e, R_e, 4.0)


def exponential_disk(R, I_0, h):
    """Exponential disk — Sérsic with n=1 fixed."""
    return I_0 * np.exp(-R / h)


def double_sersic(R, I_e1, R_e1, n1, I_e2, R_e2, n2):
    """
    Two-component Sérsic: bulge (component 1) + halo (component 2).
    Component 1 is constrained to have a smaller scale radius (R_e1 < R_e2).
    """
    return sersic(R, I_e1, R_e1, n1) + sersic(R, I_e2, R_e2, n2)


def core_sersic(R, I_prime, R_b, R_e, n, alpha, gamma):
    """
    Core-Sérsic profile (Graham et al. 2003):
    I(R) = I' [1 + (R_b/R)^α]^{γ/α} × exp{−b[(R^α + R_b^α)/R_e^α]^{1/(n α)}}

    Parameters
    ----------
    R_b    : break (core) radius [kpc]
    n      : outer Sérsic index
    alpha  : sharpness of the break (α=10 → sharp transition)
    gamma  : inner power-law slope of the core
    """
    n     = max(float(n), 0.5)
    alpha = max(float(alpha), 0.1)
    R_b   = max(float(R_b), 1e-3)
    R_e   = max(float(R_e), R_b)
    bn    = b_n(n)
    with np.errstate(over="ignore", invalid="ignore"):
        term1 = (1.0 + (R_b / np.maximum(R, 1e-10))**alpha)**(gamma / alpha)
        term2 = np.exp(-bn * ((R**alpha + R_b**alpha) / R_e**alpha)**(1.0/(n*alpha)))
    return I_prime * term1 * term2


def king_profile(R, I_0, R_c, R_t):
    """
    King (1962) profile:
    I(R) = I_0 × [1/sqrt(1 + (R/R_c)²) − 1/sqrt(1 + (R_t/R_c)²)]²

    R_c = core radius, R_t = tidal radius.
    The profile truncates to zero at R = R_t.
    """
    R_c = max(float(R_c), 1e-3)
    R_t = max(float(R_t), R_c * 2.0)
    with np.errstate(invalid="ignore"):
        term = (1.0/np.sqrt(1.0 + (R/R_c)**2)
                - 1.0/np.sqrt(1.0 + (R_t/R_c)**2))
        return I_0 * np.where(R < R_t, np.maximum(term, 0.0)**2, 0.0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §30.2 — FITTING ENGINE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def prepare_fit_data(Sigma: np.ndarray,
                      r_mid: np.ndarray,
                      rmin:  float = FIT_RMIN,
                      rmax:  float = FIT_RMAX) -> tuple:
    """
    Select finite, positive Σ values within the fitting radius range.
    Convert to I = Σ / Υ and return log-space arrays for fitting.

    Returns (R_fit, I_fit, lnI_fit) or (None, None, None) if too few points.
    """
    mask = ((r_mid >= rmin) & (r_mid <= rmax) &
            np.isfinite(Sigma) & (Sigma > 0))
    if mask.sum() < 5:
        return None, None, None
    R_fit   = r_mid[mask]
    I_fit   = Sigma[mask] / UPSILON          # [L_sun kpc^{-2}]
    lnI_fit = np.log(I_fit)
    return R_fit, I_fit, lnI_fit


def fit_model(model_fn, R_fit, lnI_fit, p0, bounds,
              model_name="model") -> dict:
    """
    Fit a profile model in log-space using curve_fit.

    Returns dict with keys: popt, chi2, success, model_name.
    """
    try:
        popt, _ = curve_fit(
            lambda R, *p: np.log(model_fn(R, *p)),
            R_fit, lnI_fit,
            p0=p0, bounds=bounds,
            maxfev=8000,
        )
        pred = np.log(model_fn(R_fit, *popt))
        chi2 = np.sum((lnI_fit - pred)**2) / max(1, len(R_fit) - len(p0))
        return {"popt": popt, "chi2": float(chi2),
                "success": True, "model": model_name}
    except Exception as exc:
        return {"popt": None, "chi2": np.nan,
                "success": False, "model": model_name}


def fit_all_models(Sigma: np.ndarray, r_mid: np.ndarray) -> dict:
    """
    Fit all six luminosity profile models to Σ(R).

    Returns a dict keyed by model name, each containing the fit result dict.
    """
    R_fit, I_fit, lnI_fit = prepare_fit_data(Sigma, r_mid)
    if R_fit is None:
        empty = {"popt": None, "chi2": np.nan, "success": False}
        return {m: {**empty, "model": m}
                for m in ["sersic","devauc","exp_disk",
                           "double_sersic","core_sersic","king"]}

    I_med  = float(np.exp(np.median(lnI_fit)))
    R_half = float(R_fit[len(R_fit)//2])

    results = {}

    # ── Sérsic ────────────────────────────────────────────────────────────────
    results["sersic"] = fit_model(
        sersic, R_fit, lnI_fit,
        p0=[I_med, R_half, 2.0],
        bounds=([1e-6, 0.1, 0.5], [1e20, 300.0, 10.0]),
        model_name="sersic",
    )

    # ── de Vaucouleurs ────────────────────────────────────────────────────────
    results["devauc"] = fit_model(
        devauc, R_fit, lnI_fit,
        p0=[I_med, R_half],
        bounds=([1e-6, 0.1], [1e20, 300.0]),
        model_name="devauc",
    )

    # ── Exponential disk ──────────────────────────────────────────────────────
    results["exp_disk"] = fit_model(
        exponential_disk, R_fit, lnI_fit,
        p0=[I_fit[0], R_half],
        bounds=([1e-6, 0.1], [1e20, 300.0]),
        model_name="exp_disk",
    )

    # ── Double Sérsic (bulge + halo) ──────────────────────────────────────────
    # Constrain R_e1 < R_e2 (bulge smaller than halo).
    results["double_sersic"] = fit_model(
        double_sersic, R_fit, lnI_fit,
        p0=[I_med*0.7, R_half*0.3, 4.0,
            I_med*0.3, R_half*2.0, 1.0],
        bounds=([1e-6, 0.1, 0.5, 1e-6, 0.5, 0.5],
                [1e20, 50.0, 10.0, 1e20, 300.0, 6.0]),
        model_name="double_sersic",
    )

    # ── Core-Sérsic ───────────────────────────────────────────────────────────
    results["core_sersic"] = fit_model(
        core_sersic, R_fit, lnI_fit,
        p0=[I_med, 1.0, R_half, 4.0, 10.0, 0.1],
        bounds=([1e-6, 0.05, 0.5, 0.5, 1.0, 0.0],
                [1e20, 20.0, 300.0, 10.0, 20.0, 1.5]),
        model_name="core_sersic",
    )

    # ── King profile ──────────────────────────────────────────────────────────
    results["king"] = fit_model(
        king_profile, R_fit, lnI_fit,
        p0=[I_med, R_half*0.2, R_half*5.0],
        bounds=([1e-6, 0.05, 1.0], [1e20, 50.0, 500.0]),
        model_name="king",
    )

    return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §30.3 — PRE-ALLOCATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

lfit_snap_nums  = SNAPSHOTS[::LFIT_STEP]
n_lfit          = len(lfit_snap_nums)
lfit_snap_map   = {s: i for i, s in enumerate(lfit_snap_nums)}
time_lfit       = np.full(n_lfit, np.nan)

# ── Sérsic parameters ─────────────────────────────────────────────────────────
sersic_n_arr   = np.full(n_lfit, np.nan)
sersic_Re_arr  = np.full(n_lfit, np.nan)
sersic_Ie_arr  = np.full(n_lfit, np.nan)
sersic_chi2_arr = np.full(n_lfit, np.nan)

# ── de Vaucouleurs ────────────────────────────────────────────────────────────
devauc_Re_arr   = np.full(n_lfit, np.nan)
devauc_chi2_arr = np.full(n_lfit, np.nan)

# ── Double Sérsic: bulge and halo components ──────────────────────────────────
ds_Re_bulge_arr = np.full(n_lfit, np.nan)
ds_n_bulge_arr  = np.full(n_lfit, np.nan)
ds_Re_halo_arr  = np.full(n_lfit, np.nan)
ds_n_halo_arr   = np.full(n_lfit, np.nan)
ds_chi2_arr     = np.full(n_lfit, np.nan)

# ── King profile ──────────────────────────────────────────────────────────────
king_Rc_arr    = np.full(n_lfit, np.nan)
king_Rt_arr    = np.full(n_lfit, np.nan)
king_chi2_arr  = np.full(n_lfit, np.nan)

# ── Residuals: (r, t) grid — Sérsic residuals only for the heatmap ────────────
resid_ts       = np.full((n_lfit, nb_proj), np.nan)

# ── Projected surface density cache (recomputed per snapshot) ─────────────────
Sigma_cache    = {}    # snap_num → Σ(R) array, kept for figure re-use

print(f"\n[Pre-alloc]  Fit snaps    : {n_lfit}")
print(f"             Residual ts  : {resid_ts.shape}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §30.4 — MAIN LOOP                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  §30.4 — Main Fitting Loop")
print("="*80)

ring_areas = np.pi * (R_PROJ[1:]**2 - R_PROJ[:-1]**2)
t_loop_start = time.perf_counter()

for i_lf, snap_num in enumerate(lfit_snap_nums):

    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    try:
        snap_data = load_snapshot_particles(mw_file, m31_file)
    except Exception as exc:
        print(f"  [ERROR] snap {snap_num}: {exc}")
        continue

    pos   = snap_data["pos"]
    m     = snap_data["m_msun"]
    R_proj_particles = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)

    # ── Compute projected surface density Σ(R) ────────────────────────────────
    Sigma = np.full(nb_proj, np.nan)
    bid   = np.digitize(R_proj_particles, R_PROJ) - 1
    for b in range(nb_proj):
        mask = bid == b
        if mask.sum() >= MIN_PART_SHELL:
            Sigma[b] = m[mask].sum() / ring_areas[b]

    Sigma_cache[snap_num] = Sigma.copy()

    snap_global_i = np.where(SNAPSHOTS == snap_num)[0]
    time_lfit[i_lf] = (time_arr[snap_global_i[0]]
                        if len(snap_global_i) > 0 else float(snap_num))

    # ── Fit all models ─────────────────────────────────────────────────────────
    results = fit_all_models(Sigma, r_proj_mid)

    # ── Store Sérsic ──────────────────────────────────────────────────────────
    if results["sersic"]["success"]:
        p = results["sersic"]["popt"]
        sersic_Ie_arr [i_lf] = p[0]
        sersic_Re_arr [i_lf] = p[1]
        sersic_n_arr  [i_lf] = p[2]
        sersic_chi2_arr[i_lf] = results["sersic"]["chi2"]

    # ── Store de Vaucouleurs ──────────────────────────────────────────────────
    if results["devauc"]["success"]:
        devauc_Re_arr  [i_lf] = results["devauc"]["popt"][1]
        devauc_chi2_arr[i_lf] = results["devauc"]["chi2"]

    # ── Store double Sérsic ───────────────────────────────────────────────────
    if results["double_sersic"]["success"]:
        p = results["double_sersic"]["popt"]
        # Identify bulge (smaller R_e) vs halo (larger R_e).
        if p[1] <= p[4]:
            ds_Re_bulge_arr[i_lf] = p[1]; ds_n_bulge_arr[i_lf] = p[2]
            ds_Re_halo_arr [i_lf] = p[4]; ds_n_halo_arr [i_lf] = p[5]
        else:
            ds_Re_bulge_arr[i_lf] = p[4]; ds_n_bulge_arr[i_lf] = p[5]
            ds_Re_halo_arr [i_lf] = p[1]; ds_n_halo_arr [i_lf] = p[2]
        ds_chi2_arr[i_lf] = results["double_sersic"]["chi2"]

    # ── Store King ────────────────────────────────────────────────────────────
    if results["king"]["success"]:
        p = results["king"]["popt"]
        king_Rc_arr  [i_lf] = p[1]
        king_Rt_arr  [i_lf] = p[2]
        king_chi2_arr[i_lf] = results["king"]["chi2"]

    # ── Sérsic residuals Δ(R) = (Σ_meas − Σ_model) / Σ_model ────────────────
    if results["sersic"]["success"]:
        p         = results["sersic"]["popt"]
        Sigma_mod = sersic(r_proj_mid, *p) * UPSILON
        with np.errstate(invalid="ignore", divide="ignore"):
            resid_ts[i_lf, :] = np.where(
                np.isfinite(Sigma) & (Sigma_mod > 0),
                (Sigma - Sigma_mod) / Sigma_mod,
                np.nan,
            )

    if (i_lf + 1) % 40 == 0:
        elapsed = time.perf_counter() - t_loop_start
        print(f"  lfit {i_lf+1}/{n_lfit}  "
              f"n={sersic_n_arr[i_lf]:.2f}  "
              f"R_e={sersic_Re_arr[i_lf]:.1f} kpc  "
              f"[{elapsed:.0f}s]")

print(f"\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §30.5 — FIGURES                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BG    = "#0d0d18"
MUTED = "#7070a0"

def _ax(ax, xlabel="", ylabel="", title="", log_x=False, log_y=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")
    ax.tick_params(colors="#9090b0", labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9,  color="#c8c8e8")
    ax.set_ylabel(ylabel, fontsize=9,  color="#c8c8e8")
    ax.set_title(title,   fontsize=10, color="#c8c8e8", pad=5)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    return ax

# Profile epochs mapped to lfit indices.
lfit_profile_ii = [int(f * (n_lfit - 1)) for f in [0.0, 0.2, 0.4, 0.65, 1.0]]
lfit_labels     = [f"Snap {lfit_snap_nums[ii]}" for ii in lfit_profile_ii]

r_plot = np.logspace(np.log10(FIT_RMIN), np.log10(FIT_RMAX), 300)
t_lfit_min = np.nanmin(time_lfit)
t_lfit_max = np.nanmax(time_lfit)


def _profile_axes(axes, title):
    fig = axes[0].figure
    for ax in axes:
        _ax(ax, xlabel="R [kpc]", log_x=True, log_y=True)
    axes[0].set_ylabel(r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]", fontsize=9)
    fig.suptitle(title, fontsize=12, color="#c8c8e8")


def _plot_model_panel(ax, snap_num, model_fn, popt, color,
                       model_label, Sigma=None, r_mid=None):
    """Plot data and model overlay on a single axes."""
    if Sigma is None:
        Sigma = Sigma_cache.get(snap_num)
    if r_mid is None:
        r_mid = r_proj_mid
    if Sigma is not None:
        valid = np.isfinite(Sigma) & (Sigma > 0)
        ax.scatter(r_mid[valid], Sigma[valid],
                   s=12, color=color, alpha=0.7, zorder=3)
    if popt is not None:
        I_plot = model_fn(r_plot, *popt) * UPSILON
        valid_p = np.isfinite(I_plot) & (I_plot > 0)
        if valid_p.any():
            ax.plot(r_plot[valid_p], I_plot[valid_p],
                    color="white", lw=1.8, label=model_label)
    ax.set_xlim(FIT_RMIN, FIT_RMAX)
    ax.legend(fontsize=6)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — SÉRSIC FITS AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Fig 1]  Sérsic fits …")

fig1, axes1 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.06})
_profile_axes(axes1, r"Sérsic Profile  $I(R) = I_e\,\exp\{-b_n[(R/R_e)^{1/n}-1]\}$")

for col, (ii, label, color) in enumerate(zip(
        lfit_profile_ii, lfit_labels, PROFILE_COLORS)):
    ax = axes1[col]; ax.set_title(label, fontsize=9, color="#c8c8e8")
    snap_num = lfit_snap_nums[ii]
    popt = None
    if (np.isfinite(sersic_Ie_arr[ii]) and
            np.isfinite(sersic_Re_arr[ii]) and
            np.isfinite(sersic_n_arr[ii])):
        popt = [sersic_Ie_arr[ii], sersic_Re_arr[ii], sersic_n_arr[ii]]
    ml = (fr"Sérsic n={sersic_n_arr[ii]:.2f}, "
          fr"$R_e$={sersic_Re_arr[ii]:.1f} kpc" if popt else "No fit")
    _plot_model_panel(ax, snap_num, sersic, popt, color, ml)
    if col == 0:
        ax.set_ylabel(r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]", fontsize=9)

fig1.savefig(os.path.join(OUT_DIR, "section30_sersic_fits.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print("  Saved: section30_sersic_fits.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — DE VAUCOULEURS FITS AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 2]  de Vaucouleurs fits …")

fig2, axes2 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.06})
_profile_axes(axes2, r"de Vaucouleurs  $I(R) = I_e\,\exp\{-7.67[(R/R_e)^{1/4}-1]\}$")

for col, (ii, label, color) in enumerate(zip(
        lfit_profile_ii, lfit_labels, PROFILE_COLORS)):
    ax = axes2[col]; ax.set_title(label, fontsize=9, color="#c8c8e8")
    snap_num = lfit_snap_nums[ii]
    popt = None
    if (np.isfinite(sersic_Ie_arr[ii]) and np.isfinite(devauc_Re_arr[ii])):
        popt = [sersic_Ie_arr[ii], devauc_Re_arr[ii]]
    ml = (fr"deVauc $R_e$={devauc_Re_arr[ii]:.1f} kpc"
          if popt else "No fit")
    _plot_model_panel(ax, snap_num, devauc, popt, color, ml)
    if col == 0:
        ax.set_ylabel(r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]", fontsize=9)

fig2.savefig(os.path.join(OUT_DIR, "section30_devauc_fits.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("  Saved: section30_devauc_fits.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — DOUBLE-SÉRSIC (BULGE + HALO DECOMPOSITION) AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 3]  Double-Sérsic fits …")

fig3, axes3 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.06})
_profile_axes(axes3, "Double-Sérsic Bulge + Halo Decomposition")

for col, (ii, label, color) in enumerate(zip(
        lfit_profile_ii, lfit_labels, PROFILE_COLORS)):
    ax = axes3[col]; ax.set_title(label, fontsize=9, color="#c8c8e8")
    snap_num = lfit_snap_nums[ii]

    Sigma = Sigma_cache.get(snap_num)
    if Sigma is not None:
        valid = np.isfinite(Sigma) & (Sigma > 0)
        ax.scatter(r_proj_mid[valid], Sigma[valid],
                   s=12, color=color, alpha=0.6, zorder=3, label="Data")

    # Refit double Sérsic for the figure.
    if Sigma is not None:
        res_ds = fit_all_models(Sigma, r_proj_mid)["double_sersic"]
        if res_ds["success"]:
            p = res_ds["popt"]
            I1 = sersic(r_plot, p[0], p[1], p[2]) * UPSILON
            I2 = sersic(r_plot, p[3], p[4], p[5]) * UPSILON
            It = (I1 + I2)
            for Ic, ls, lbl in [(I1, "--", "Bulge"), (I2, ":", "Halo"), (It, "-", "Total")]:
                v = np.isfinite(Ic) & (Ic > 0)
                if v.any():
                    ax.plot(r_plot[v], Ic[v], color="white" if lbl=="Total" else color,
                            lw=1.8 if lbl=="Total" else 1.0, ls=ls, alpha=1.0 if lbl=="Total" else 0.6,
                            label=lbl)
    ax.set_xlim(FIT_RMIN, FIT_RMAX)
    ax.legend(fontsize=6)
    if col == 0:
        ax.set_ylabel(r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]", fontsize=9)

fig3.savefig(os.path.join(OUT_DIR, "section30_doublesersic_fits.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print("  Saved: section30_doublesersic_fits.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — CORE-SÉRSIC FITS AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 4]  Core-Sérsic fits …")

fig4, axes4 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.06})
_profile_axes(axes4, "Core-Sérsic Profile (inner power-law core + outer Sérsic)")

for col, (ii, label, color) in enumerate(zip(
        lfit_profile_ii, lfit_labels, PROFILE_COLORS)):
    ax = axes4[col]; ax.set_title(label, fontsize=9, color="#c8c8e8")
    snap_num = lfit_snap_nums[ii]
    Sigma = Sigma_cache.get(snap_num)
    if Sigma is not None:
        res_cs = fit_all_models(Sigma, r_proj_mid)["core_sersic"]
        _plot_model_panel(ax, snap_num, core_sersic,
                          res_cs["popt"] if res_cs["success"] else None,
                          color,
                          f"CoreSérsic χ²={res_cs['chi2']:.2f}" if res_cs["success"] else "No fit")
    if col == 0:
        ax.set_ylabel(r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]", fontsize=9)

fig4.savefig(os.path.join(OUT_DIR, "section30_coresersic_fits.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig4)
print("  Saved: section30_coresersic_fits.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — KING PROFILE FITS AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 5]  King profile fits …")

fig5, axes5 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.06})
_profile_axes(axes5, r"King Profile  $I(R) \propto [1/\sqrt{1+(R/R_c)^2} - 1/\sqrt{1+(R_t/R_c)^2}]^2$")

for col, (ii, label, color) in enumerate(zip(
        lfit_profile_ii, lfit_labels, PROFILE_COLORS)):
    ax = axes5[col]; ax.set_title(label, fontsize=9, color="#c8c8e8")
    snap_num = lfit_snap_nums[ii]
    Sigma = Sigma_cache.get(snap_num)
    popt  = None
    ml    = "No fit"
    if np.isfinite(king_Rc_arr[ii]) and np.isfinite(king_Rt_arr[ii]):
        popt = [sersic_Ie_arr[ii] if np.isfinite(sersic_Ie_arr[ii]) else 1e6,
                king_Rc_arr[ii], king_Rt_arr[ii]]
        ml   = fr"King $R_c$={king_Rc_arr[ii]:.1f}, $R_t$={king_Rt_arr[ii]:.1f} kpc"
    _plot_model_panel(ax, snap_num, king_profile, popt, color, ml)
    if col == 0:
        ax.set_ylabel(r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]", fontsize=9)

fig5.savefig(os.path.join(OUT_DIR, "section30_king_fits.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig5)
print("  Saved: section30_king_fits.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — SÉRSIC n(t) AND R_e(t) EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════
#
# The n → 4 transition is the central observational result of this section.
# A galaxy beginning as a disk (n≈1) and ending as an elliptical (n≈4) after
# a major merger is one of the most robust predictions of galaxy formation theory.

print("[Fig 6]  n(t) and R_e(t) evolution …")

fig6, (ax6a, ax6b, ax6c) = plt.subplots(3, 1, figsize=(11, 10), facecolor=BG,
                                          sharex=True, gridspec_kw={"hspace":0.08})

valid_n = np.isfinite(sersic_n_arr)
_ax(ax6a, ylabel="Sérsic index  n",
    title="Luminosity Profile Parameter Evolution")
ax6a.plot(time_lfit[valid_n], sersic_n_arr[valid_n], color="#aa55ff", lw=2.0)
ax6a.axhline(4.0, color="#ffcc44", lw=0.9, ls="--", alpha=0.6,
             label="n = 4  (de Vaucouleurs / elliptical)")
ax6a.axhline(1.0, color="#555577", lw=0.8, ls=":",
             label="n = 1  (exponential disk)")
ax6a.legend(fontsize=8); ax6a.set_ylim(0, 8)

valid_Re = np.isfinite(sersic_Re_arr)
_ax(ax6b, ylabel=r"Effective radius $R_e$ [kpc]")
ax6b.plot(time_lfit[valid_Re], sersic_Re_arr[valid_Re], color="#4a8fff", lw=2.0,
          label=r"Sérsic $R_e$")
ax6b.plot(time_lfit[np.isfinite(devauc_Re_arr)],
          devauc_Re_arr[np.isfinite(devauc_Re_arr)],
          color="#00d4aa", lw=1.5, ls="--", label=r"deVauc $R_e$")
ax6b.legend(fontsize=8)

_ax(ax6c, xlabel=time_label, ylabel=r"$\chi^2$ (log-space)")
for arr, color, label in [
    (sersic_chi2_arr, "#aa55ff",  "Sérsic"),
    (devauc_chi2_arr, "#00d4aa", "deVauc"),
    (ds_chi2_arr,     "#ff9944",  "Double-Sérsic"),
    (king_chi2_arr,   "#ff5566",  "King"),
]:
    valid = np.isfinite(arr)
    if valid.any():
        ax6c.semilogy(time_lfit[valid], arr[valid], color=color, lw=1.5, label=label)
ax6c.axhline(1.0, color="#ffffff", lw=0.7, ls="--", alpha=0.4)
ax6c.legend(fontsize=7, ncol=2)
ax6c.set_title("Fit Quality Comparison", fontsize=10, color="#c8c8e8", pad=5)

fig6.savefig(os.path.join(OUT_DIR, "section30_n_evolution.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig6)
print("  Saved: section30_n_evolution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — DOUBLE-SÉRSIC BULGE AND HALO COMPONENT EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 7]  Bulge/halo component evolution …")

fig7, axes7 = plt.subplots(2, 2, figsize=(12, 8), facecolor=BG,
                            gridspec_kw={"hspace":0.32, "wspace":0.28})
axes7 = axes7.flatten()

for ax, arr, ylabel, title, color in [
    (axes7[0], ds_Re_bulge_arr, r"$R_{e,\,\rm bulge}$ [kpc]", "Bulge $R_e$", "#ff9944"),
    (axes7[1], ds_n_bulge_arr,  "Bulge Sérsic  n",             "Bulge index n", "#ff9944"),
    (axes7[2], ds_Re_halo_arr,  r"$R_{e,\,\rm halo}$ [kpc]",  "Halo $R_e$",  "#4a8fff"),
    (axes7[3], ds_n_halo_arr,   "Halo Sérsic  n",              "Halo index n",  "#4a8fff"),
]:
    _ax(ax, xlabel=time_label, ylabel=ylabel, title=title)
    valid = np.isfinite(arr)
    if valid.any():
        ax.plot(time_lfit[valid], arr[valid], color=color, lw=1.8)

axes7[1].axhline(4.0, color="#555577", lw=0.7, ls="--", alpha=0.5)
axes7[3].axhline(1.0, color="#555577", lw=0.7, ls=":", alpha=0.5)

fig7.suptitle("Double-Sérsic Component Evolution  (bulge = orange, halo = blue)",
              fontsize=11, color="#c8c8e8")
fig7.savefig(os.path.join(OUT_DIR, "section30_component_evolution.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig7)
print("  Saved: section30_component_evolution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — SÉRSIC RESIDUAL HEATMAP Δ(R, t)
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 8]  Sérsic residual heatmap …")

resid_max = np.nanpercentile(np.abs(resid_ts[np.isfinite(resid_ts)]), 95)

fig8, ax8 = plt.subplots(figsize=(12, 5), facecolor=BG)
im8 = ax8.imshow(
    np.clip(resid_ts, -resid_max, resid_max).T,
    aspect="auto", origin="lower",
    extent=[t_lfit_min, t_lfit_max, R_PROJ[0], R_PROJ[-1]],
    cmap="seismic",
    norm=TwoSlopeNorm(vmin=-resid_max, vcenter=0.0, vmax=resid_max),
)
ax8.set_yscale("log")
_ax(ax8, xlabel=time_label, ylabel="R [kpc]",
    title=r"Sérsic Fit Residuals  $\Delta = (\Sigma_{\rm meas} - \Sigma_{\rm Sérsic})/\Sigma_{\rm Sérsic}$")
cb8 = fig8.colorbar(im8, ax=ax8, pad=0.01)
cb8.set_label(r"$\Delta$  (red = excess, blue = deficit)", fontsize=8)

fig8.savefig(os.path.join(OUT_DIR, "section30_residuals_heatmap.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig8)
print("  Saved: section30_residuals_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9 — ALL MODELS COMPARED AT ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 9]  Model comparison at mid epoch …")

mid_lf_ii  = n_lfit // 2
snap_mid   = lfit_snap_nums[mid_lf_ii]
Sigma_mid  = Sigma_cache.get(snap_mid)

fig9, ax9 = plt.subplots(figsize=(9, 7), facecolor=BG)
_ax(ax9, xlabel="R [kpc]",
    ylabel=r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]",
    title=f"All Profile Models  ·  Snap {snap_mid}",
    log_x=True, log_y=True)

if Sigma_mid is not None:
    valid_d = np.isfinite(Sigma_mid) & (Sigma_mid > 0)
    ax9.scatter(r_proj_mid[valid_d], Sigma_mid[valid_d],
                s=18, color="#aaaacc", alpha=0.8, zorder=3, label="Data Σ(R)")

    res_all = fit_all_models(Sigma_mid, r_proj_mid)

    model_plot_specs = [
        ("sersic",         sersic,         "#aa55ff", "-",  "Sérsic"),
        ("devauc",         devauc,         "#ffcc44", "--", "de Vaucouleurs"),
        ("exp_disk",       exponential_disk,"#00d4aa", ":",  "Exp. disk"),
        ("double_sersic",  double_sersic,  "#4a8fff", "-.", "Double Sérsic"),
        ("core_sersic",    core_sersic,    "#ff9944", "-",  "Core-Sérsic"),
        ("king",           king_profile,   "#ff5566", "--", "King"),
    ]
    for key, fn, color, ls, label in model_plot_specs:
        if res_all[key]["success"]:
            I_m = fn(r_plot, *res_all[key]["popt"]) * UPSILON
            v   = np.isfinite(I_m) & (I_m > 0)
            if v.any():
                chi2_str = f" χ²={res_all[key]['chi2']:.2f}"
                ax9.plot(r_plot[v], I_m[v], color=color, lw=1.8,
                         ls=ls, label=label + chi2_str)

ax9.set_xlim(FIT_RMIN, FIT_RMAX)
ax9.legend(fontsize=7)

fig9.savefig(os.path.join(OUT_DIR, "section30_model_comparison.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig9)
print("  Saved: section30_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10 — MASTER SUMMARY PANEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Summary]  Master summary panel …")

fig10 = plt.figure(figsize=(16, 10), facecolor=BG)
gs10  = gridspec.GridSpec(2, 2, figure=fig10,
                           hspace=0.38, wspace=0.32,
                           left=0.08, right=0.97,
                           top=0.93, bottom=0.07)

# (0,0) Sérsic n(t).
ax_s00 = fig10.add_subplot(gs10[0, 0])
_ax(ax_s00, xlabel=time_label, ylabel="Sérsic index n",
    title="Sérsic Index n(t)  [disk→elliptical transition]")
valid_n = np.isfinite(sersic_n_arr)
if valid_n.any():
    ax_s00.plot(time_lfit[valid_n], sersic_n_arr[valid_n],
                color="#aa55ff", lw=2.0)
ax_s00.axhline(4.0, color="#ffcc44", lw=0.8, ls="--", alpha=0.6, label="n=4")
ax_s00.axhline(1.0, color="#555577", lw=0.7, ls=":", label="n=1")
ax_s00.set_ylim(0, 8); ax_s00.legend(fontsize=8)

# (0,1) R_e(t).
ax_s01 = fig10.add_subplot(gs10[0, 1])
_ax(ax_s01, xlabel=time_label, ylabel=r"$R_e$ [kpc]",
    title="Effective Radius Evolution")
for arr, color, label in [
    (sersic_Re_arr, "#4a8fff", r"Sérsic $R_e$"),
    (devauc_Re_arr, "#00d4aa", r"deVauc $R_e$"),
]:
    valid = np.isfinite(arr)
    if valid.any():
        ax_s01.plot(time_lfit[valid], arr[valid],
                    color=color, lw=1.8, label=label)
ax_s01.legend(fontsize=8)

# (1,0) Residual heatmap.
ax_s10 = fig10.add_subplot(gs10[1, 0])
im_s10 = ax_s10.imshow(
    np.clip(resid_ts, -resid_max, resid_max).T,
    aspect="auto", origin="lower",
    extent=[t_lfit_min, t_lfit_max, R_PROJ[0], R_PROJ[-1]],
    cmap="seismic",
    norm=TwoSlopeNorm(vmin=-resid_max, vcenter=0.0, vmax=resid_max),
)
ax_s10.set_yscale("log")
_ax(ax_s10, xlabel=time_label, ylabel="R [kpc]",
    title=r"Sérsic Residuals $\Delta(R,t)$")
fig10.colorbar(im_s10, ax=ax_s10, shrink=0.8, label=r"$\Delta$")

# (1,1) χ² comparison.
ax_s11 = fig10.add_subplot(gs10[1, 1])
_ax(ax_s11, xlabel=time_label, ylabel=r"Reduced $\chi^2$",
    title="Fit Quality — All Models", log_y=True)
for arr, color, label in [
    (sersic_chi2_arr,  "#aa55ff", "Sérsic"),
    (devauc_chi2_arr,  "#00d4aa", "deVauc"),
    (ds_chi2_arr,      "#ff9944", "Double-Sérsic"),
    (king_chi2_arr,    "#ff5566", "King"),
]:
    valid = np.isfinite(arr)
    if valid.any():
        ax_s11.semilogy(time_lfit[valid], arr[valid],
                        color=color, lw=1.5, label=label)
ax_s11.axhline(1.0, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
ax_s11.legend(fontsize=7)

fig10.suptitle("Section 30 Summary  ·  Luminosity Profile Fitting",
               fontsize=13, color="#c8c8e8", fontweight="bold")
fig10.savefig(os.path.join(OUT_DIR, "section30_summary_panel.png"),
              dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig10)
print("  Saved: section30_summary_panel.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §30.5 — SECTION COMPLETE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 30 COMPLETE")
print("="*80)
outputs_30 = [
    "section30_sersic_fits.png",
    "section30_devauc_fits.png",
    "section30_doublesersic_fits.png",
    "section30_coresersic_fits.png",
    "section30_king_fits.png",
    "section30_n_evolution.png",
    "section30_component_evolution.png",
    "section30_residuals_heatmap.png",
    "section30_model_comparison.png",
    "section30_summary_panel.png",
]
for fn in outputs_30:
    fp   = os.path.join(OUT_DIR, fn)
    size = os.path.getsize(fp)/1e6 if os.path.isfile(fp) else 0.0
    print(f"  {fn:<48} {size:6.2f} MB")
print("="*80)

# Print final structural parameter summary.
valid_final = np.isfinite(sersic_n_arr) & np.isfinite(sersic_Re_arr)
if valid_final.any():
    print(f"\n  SÉRSIC PARAMETER SUMMARY")
    print(f"  Initial snap: n = {sersic_n_arr[valid_final][0]:.2f},  "
          f"R_e = {sersic_Re_arr[valid_final][0]:.1f} kpc")
    print(f"  Final   snap: n = {sersic_n_arr[valid_final][-1]:.2f},  "
          f"R_e = {sersic_Re_arr[valid_final][-1]:.1f} kpc")
    print(f"  Δn = {sersic_n_arr[valid_final][-1] - sersic_n_arr[valid_final][0]:.2f}  "
          f"(+ve = disk → elliptical)")
print("="*80)
