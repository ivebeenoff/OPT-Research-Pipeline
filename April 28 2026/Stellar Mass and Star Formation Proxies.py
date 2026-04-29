"""
===============================================================================
SECTION 25 — STELLAR MASS & STAR FORMATION PROXIES
===============================================================================
Author  : Abhinav Vatsa

Continuation of the MW–M31 analysis pipeline.  All globals (SNAPSHOTS, ns,
R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL,
G_KPC_KMS2_MSUN, PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr,
time_label, time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

Physical motivation
-------------------
This is a pure N-body simulation — there is no explicit star formation, gas
physics, or stellar feedback.  Nevertheless, the dark matter distribution
encodes strong proxies for where and when star formation *would* occur in a
hydrodynamic counterpart simulation:

  (A) HIGH LOCAL DENSITY  — regions exceeding a threshold ρ > ρ_SF are
      star-forming in almost all sub-grid models (Schmidt–Kennicutt law).
      We track the mass fraction above this threshold as a proxy for the
      instantaneous star-forming gas reservoir.

  (B) LOW VELOCITY DISPERSION  — cold, kinematically quiet regions are
      gravitationally unstable (Toomre Q < 1 in the disk limit).  We
      compute a Toomre-like stability parameter from the local σ_r and
      surface density Σ(R) at each snapshot.

  (C) COMPACTNESS  — the effective radius r_eff and Sérsic index n track
      whether the stellar remnant is building a compact elliptical
      (n → 4, small r_eff) or a diffuse merger product.  We fit a
      Sérsic profile to the projected surface density Σ(R) at each epoch.

  (D) STELLAR MASS BUILDUP PROXY  — the mass enclosed within the
      "star-forming radius" (r < r_SF) as a function of time acts as a
      proxy for the cumulative stellar mass that could have formed.

Note on particle types
----------------------
PTYPE = 1 (dark matter) is used throughout the pipeline.  Disk particles
(PTYPE = 2) would give a better stellar proxy because they trace the
baryonic component.  This section is written to work with PTYPE = 1 but
includes a configuration switch STELLAR_PTYPE that can be set to 2 if
disk particles are available in the snapshots.

Outputs
-------
  section25_sf_fraction.png        Star-forming mass fraction vs. time
  section25_sf_density_map.png     2D map of SF-eligible regions at 5 epochs
  section25_toomre_profile.png     Toomre Q(R) profiles at 5 epochs
  section25_toomre_heatmap.png     Q(R,t) heatmap
  section25_sersic_fit.png         Sérsic profile fits at 5 epochs
  section25_sersic_params.png      n(t) and r_eff(t) evolution
  section25_sf_radius.png          Star-forming radius r_SF(t)
  section25_cumulative_sfr.png     Proxy cumulative stellar mass vs. time
  section25_summary_panel.png      Master 4-panel summary

===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import os
import time
import warnings


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §25.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Particle type for stellar mass proxy.
# Set to 2 (disk) if disk particles are available; 1 (DM) as fallback.
STELLAR_PTYPE = PTYPE   # inherited from parent pipeline

# Star-formation density threshold  [M_sun kpc^{-3}].
# Typical values in SPH simulations: 0.1–10 cm^{-3} ≈ 1e6–1e8 M_sun kpc^{-3}.
# We use a conservative threshold suited to DM density proxies.
RHO_SF_THRESH = 1e6     # [M_sun kpc^{-3}]

# Inner radius within which to compute star-forming mass [kpc].
# Particles beyond this are considered halo, not disk/bulge.
R_SF_MAX_KPC  = 30.0

# Toomre Q parameters.
# Q = σ_R κ / (π G Σ)  where κ is the epicyclic frequency.
# We approximate κ ≈ sqrt(2) v_c / r (flat rotation curve limit).
# Q < 1 → gravitationally unstable → star-forming.
TOOMRE_KAPPA_FACTOR = np.sqrt(2.0)  # flat rotation curve approximation

# Sérsic profile fitting range  [kpc].
SERSIC_RMIN = 0.5
SERSIC_RMAX = 50.0

# Temporal subsampling for the Sérsic fitting (expensive).
SERSIC_STEP = 8

# 2D map parameters for the SF density map.
SF_MAP_BINS   = 256
SF_MAP_EXTENT = 50.0    # [kpc]  — zoom in to the inner disk region

print("\n" + "="*80)
print("  SECTION 25 · Stellar Mass & Star Formation Proxies")
print("="*80)
print(f"  ρ_SF threshold  : {RHO_SF_THRESH:.0e} M_sun kpc^-3")
print(f"  R_SF max        : {R_SF_MAX_KPC} kpc")
print(f"  Sérsic step     : every {SERSIC_STEP} snapshots")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §25.1 — UTILITY FUNCTIONS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def sersic_profile(R: np.ndarray, I_e: float, R_e: float, n: float) -> np.ndarray:
    """
    Sérsic surface brightness profile:

        I(R) = I_e × exp{ −b_n × [(R/R_e)^{1/n} − 1] }

    where b_n ≈ 2n − 1/3 + 4/(405n) is the normalisation constant
    (Ciotti & Bertin 1999 approximation, valid for n > 0.5).

    Parameters
    ----------
    R   : (nb,)  — projected radii  [kpc]
    I_e : float  — surface density at R_e  [M_sun kpc^{-2}]
    R_e : float  — effective (half-mass) radius  [kpc]
    n   : float  — Sérsic index (n=1: exponential disk, n=4: de Vaucouleurs)

    Returns
    -------
    I : (nb,)  [M_sun kpc^{-2}]
    """
    n   = max(n, 0.5)   # guard against n → 0
    b_n = 2.0 * n - 1.0/3.0 + 4.0/(405.0 * n)
    return I_e * np.exp(-b_n * ((R / R_e)**(1.0 / n) - 1.0))


def fit_sersic(R_mid: np.ndarray, Sigma: np.ndarray) -> dict:
    """
    Fit a Sérsic profile to the projected surface density Σ(R).

    Fitting is done in log-space to give equal weight per decade of Σ,
    consistent with the density fitting in §30.

    Parameters
    ----------
    R_mid  : (nb,)  — projected bin centres  [kpc]
    Sigma  : (nb,)  — surface density  [M_sun kpc^{-2}]

    Returns
    -------
    dict with keys:
        I_e, R_e, n   — best-fit Sérsic parameters
        chi2          — reduced chi-squared in log-space
        success       — bool
    """
    mask = (R_mid >= SERSIC_RMIN) & (R_mid <= SERSIC_RMAX) & \
           np.isfinite(Sigma) & (Sigma > 0)

    if mask.sum() < 5:
        return {"I_e": np.nan, "R_e": np.nan, "n": np.nan,
                "chi2": np.nan, "success": False}

    R_fit  = R_mid[mask]
    lnI_fit = np.log(Sigma[mask])

    # Initial guesses.
    I_e0 = np.exp(np.median(lnI_fit))
    R_e0 = R_fit[len(R_fit) // 2]
    n0   = 2.0

    try:
        popt, _ = curve_fit(
            lambda R, Ie, Re, n: np.log(sersic_profile(R, Ie, Re, n)),
            R_fit, lnI_fit,
            p0=[I_e0, R_e0, n0],
            bounds=([1e-3, 0.1, 0.5], [1e16, 200.0, 10.0]),
            maxfev=5000,
        )
        pred = np.log(sersic_profile(R_fit, *popt))
        chi2 = np.sum((lnI_fit - pred)**2) / max(1, len(R_fit) - 3)
        return {"I_e": popt[0], "R_e": popt[1], "n": popt[2],
                "chi2": chi2, "success": True}
    except Exception:
        return {"I_e": np.nan, "R_e": np.nan, "n": np.nan,
                "chi2": np.nan, "success": False}


def toomre_q_profile(Sigma: np.ndarray,
                      sigma_r: np.ndarray,
                      vc:      np.ndarray,
                      r_mid:   np.ndarray) -> np.ndarray:
    """
    Compute the Toomre stability parameter Q(R) for a disk population.

        Q(R) = σ_R(R) × κ(R) / (π G Σ(R))

    where κ(R) = TOOMRE_KAPPA_FACTOR × v_c(R) / R is the epicyclic frequency
    in the flat rotation curve approximation.

    Q < 1   →  gravitationally unstable (star-forming)
    Q = 1   →  marginally stable
    Q >> 1  →  stable (pressure-supported)

    Parameters
    ----------
    Sigma   : (nb,)  — projected surface density  [M_sun kpc^{-2}]
    sigma_r : (nb,)  — radial velocity dispersion  [km/s]
    vc      : (nb,)  — circular velocity  [km/s]
    r_mid   : (nb,)  — projected bin centres  [kpc]

    Returns
    -------
    Q : (nb,)  dimensionless
    """
    nb    = len(r_mid)
    Q_arr = np.full(nb, np.nan)

    for b in range(nb):
        if not (np.isfinite(Sigma[b]) and Sigma[b] > 0 and
                np.isfinite(sigma_r[b]) and
                np.isfinite(vc[b]) and vc[b] > 0 and
                r_mid[b] > 0):
            continue

        kappa    = TOOMRE_KAPPA_FACTOR * vc[b] / r_mid[b]  # [km/s/kpc]
        # G in units consistent with km/s and kpc:
        # G [kpc (km/s)^2 M_sun^{-1}] = G_KPC_KMS2_MSUN
        Q_arr[b] = sigma_r[b] * kappa / (np.pi * G_KPC_KMS2_MSUN * Sigma[b])

    return Q_arr


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §25.2 — PRE-ALLOCATION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Time-series scalars ────────────────────────────────────────────────────────
sf_frac_arr    = np.full(ns, np.nan)   # fraction of inner mass above ρ_SF
M_sf_arr       = np.full(ns, np.nan)   # total mass above ρ_SF within R_SF_MAX
r_sf_arr       = np.full(ns, np.nan)   # radius enclosing 90% of SF-eligible mass
cumM_sf_arr    = np.full(ns, np.nan)   # cumulative integral of M_sf (proxy SFH)

# ── Toomre Q profile (ns, nb_sph) ─────────────────────────────────────────────
Q_ts = np.full((ns, nb_sph), np.nan)

# ── Sérsic parameters ─────────────────────────────────────────────────────────
sersic_snap_nums = SNAPSHOTS[::SERSIC_STEP]
n_sersic         = len(sersic_snap_nums)
sersic_snap_map  = {s: i for i, s in enumerate(sersic_snap_nums)}
time_sersic      = np.full(n_sersic, np.nan)

sersic_n_arr   = np.full(n_sersic, np.nan)   # Sérsic index n
sersic_Re_arr  = np.full(n_sersic, np.nan)   # effective radius R_e  [kpc]
sersic_chi2_arr = np.full(n_sersic, np.nan)  # fit quality

# ── 2D SF density maps (one per profile epoch) ────────────────────────────────
sf_maps = np.zeros((len(PROFILE_INDICES), SF_MAP_BINS, SF_MAP_BINS))

print(f"\n[Pre-alloc]  Q_ts        : {Q_ts.shape}")
print(f"             Sérsic snaps: {n_sersic}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §25.3 — MAIN LOOP                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  §25.3 — Main Snapshot Loop")
print("="*80)

# Profile index to enumerate position in PROFILE_INDICES list
profile_snap_set = {SNAPSHOTS[k]: pi for pi, k in enumerate(PROFILE_INDICES)}

t_loop_start = time.perf_counter()

for i, snap_num in enumerate(SNAPSHOTS):

    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    try:
        snap_data = load_snapshot_particles(mw_file, m31_file)
        MW_obj    = CenterOfMass(mw_file,  PTYPE)
        M31_obj   = CenterOfMass(m31_file, PTYPE)
    except Exception as exc:
        print(f"  [ERROR] snap {snap_num}: {exc}")
        continue

    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    r_3d   = np.linalg.norm(pos, axis=1)
    R_proj = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)

    # ── Per-shell density (recompute locally for this section) ────────────────
    shell_vols_loc = (4.0/3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)
    bin_id_3d      = np.digitize(r_3d, R_BINS) - 1
    rho_local      = np.full(nb_sph, np.nan)
    for b in range(nb_sph):
        mask = bin_id_3d == b
        if mask.sum() >= MIN_PART_SHELL:
            rho_local[b] = m[mask].sum() / shell_vols_loc[b]

    # ── Projected surface density Σ(R) ───────────────────────────────────────
    ring_areas_loc = np.pi * (R_BINS[1:]**2 - R_BINS[:-1]**2)
    bin_id_proj    = np.digitize(R_proj, R_BINS) - 1
    Sigma_local    = np.full(nb_sph, np.nan)
    for b in range(nb_sph):
        mask = bin_id_proj == b
        if mask.sum() >= MIN_PART_SHELL:
            Sigma_local[b] = m[mask].sum() / ring_areas_loc[b]

    # ── Velocity dispersion σ_r(R) (projected radial) ─────────────────────────
    # Recompute locally using radial velocities in the COM frame.
    vx_all = np.concatenate((MW_obj.vx, M31_obj.vx))
    vy_all = np.concatenate((MW_obj.vy, M31_obj.vy))
    vz_all = np.concatenate((MW_obj.vz, M31_obj.vz))
    m_raw  = np.concatenate((MW_obj.m,  M31_obj.m))
    x_all  = np.concatenate((MW_obj.x,  M31_obj.x))
    y_all  = np.concatenate((MW_obj.y,  M31_obj.y))
    z_all  = np.concatenate((MW_obj.z,  M31_obj.z))
    xcom, ycom, zcom = MW_obj.COMdefine(x_all, y_all, z_all, m_raw)
    dr_c = np.sqrt((x_all-xcom)**2 + (y_all-ycom)**2 + (z_all-zcom)**2)
    inn  = dr_c < 15.0
    if inn.sum() >= 5:
        wi = m[inn]
        vxcom = np.sum(wi*vx_all[inn])/wi.sum()
        vycom = np.sum(wi*vy_all[inn])/wi.sum()
        vzcom = np.sum(wi*vz_all[inn])/wi.sum()
    else:
        vxcom = vycom = vzcom = 0.0

    vel   = np.vstack((vx_all-vxcom, vy_all-vycom, vz_all-vzcom)).T
    r_hat = np.where(r_3d[:,None] > 0, pos/r_3d[:,None], 0.0)
    v_r   = np.einsum("ij,ij->i", vel, r_hat)

    sigma_r_local = np.full(nb_sph, np.nan)
    for b in range(nb_sph):
        mask = bin_id_proj == b
        if mask.sum() < MIN_PART_SHELL:
            continue
        w       = m[mask]; W = w.sum()
        vr_b    = v_r[mask]
        mean_vr = np.sum(w * vr_b) / W
        sigma_r_local[b] = np.sqrt(np.sum(w * (vr_b - mean_vr)**2) / W)

    # ── Circular velocity (from enclosed mass) ────────────────────────────────
    vc_local = np.full(nb_sph, np.nan)
    for b in range(nb_sph):
        r_out = R_BINS[b+1]
        M_enc = m[r_3d <= r_out].sum()
        if r_out > 0 and M_enc > 0:
            vc_local[b] = np.sqrt(G_KPC_KMS2_MSUN * M_enc / r_out)

    # ── Toomre Q profile ──────────────────────────────────────────────────────
    Q_ts[i, :] = toomre_q_profile(
        Sigma_local, sigma_r_local, vc_local, r_mid_sph
    )

    # ── Star-forming mass fraction ────────────────────────────────────────────
    # Particles within R_SF_MAX_KPC with local shell density > RHO_SF_THRESH.
    inner_mask    = r_3d < R_SF_MAX_KPC
    M_inner       = m[inner_mask].sum()

    # Map each inner particle's shell density using the bin it falls in.
    rho_per_part  = np.full(len(m), np.nan)
    for b in range(nb_sph):
        mask = bin_id_3d == b
        if np.isfinite(rho_local[b]):
            rho_per_part[mask] = rho_local[b]

    sf_mask = inner_mask & (rho_per_part > RHO_SF_THRESH)
    M_sf    = m[sf_mask].sum()

    sf_frac_arr[i] = M_sf / (M_inner + 1e-30) if M_inner > 0 else np.nan
    M_sf_arr[i]    = M_sf

    # Star-forming radius: smallest r enclosing 90% of SF-eligible mass.
    if M_sf > 0:
        r_sf_sorted = np.sort(r_3d[sf_mask])
        cum_m_sf    = np.cumsum(m[sf_mask][np.argsort(r_3d[sf_mask])])
        idx_90      = np.searchsorted(cum_m_sf, 0.9 * M_sf)
        r_sf_arr[i] = r_sf_sorted[min(idx_90, len(r_sf_sorted)-1)]

    # ── 2D SF density map (at profile epochs only) ────────────────────────────
    if snap_num in profile_snap_set:
        pi_map = profile_snap_set[snap_num]
        sf_x   = pos[sf_mask, 0]
        sf_y   = pos[sf_mask, 1]
        H, _, _ = np.histogram2d(
            sf_x, sf_y,
            bins=SF_MAP_BINS,
            range=[[-SF_MAP_EXTENT, SF_MAP_EXTENT],
                   [-SF_MAP_EXTENT, SF_MAP_EXTENT]],
            weights=m[sf_mask],
        )
        sf_maps[pi_map] = H

    # ── Sérsic fitting ────────────────────────────────────────────────────────
    if snap_num in sersic_snap_map:
        si = sersic_snap_map[snap_num]
        time_sersic[si] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)
        res = fit_sersic(r_mid_sph, Sigma_local)
        sersic_n_arr  [si] = res["n"]
        sersic_Re_arr [si] = res["R_e"]
        sersic_chi2_arr[si] = res["chi2"]

    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t_loop_start
        print(f"  snap {snap_num:04d}  "
              f"f_SF={sf_frac_arr[i]:.3f}  "
              f"r_SF={r_sf_arr[i]:.1f} kpc  "
              f"[{elapsed:.0f}s]")

print(f"\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total")

# ── Cumulative SF mass proxy ───────────────────────────────────────────────────
# Simple running integral of M_sf(t) treating each snapshot as Δt = 1 unit.
cumM_sf_arr = np.nancumsum(np.where(np.isfinite(M_sf_arr), M_sf_arr, 0.0))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §25.4 — FIGURES                                                           ║
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

t_min = np.nanmin(time_arr)
t_max = np.nanmax(time_arr)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — STAR-FORMING MASS FRACTION vs. TIME
# ══════════════════════════════════════════════════════════════════════════════
#
# f_SF(t) peaks at pericentre passages when the tidal compression drives gas
# (and in our proxy, DM) to high densities above the SF threshold.  Troughs
# between pericentres mark quenching episodes as the galaxies separate and
# the density falls below ρ_SF.

print("\n[Fig 1]  Star-forming mass fraction …")

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
                                   sharex=True, gridspec_kw={"hspace": 0.08})

_ax(ax1a, ylabel=r"$f_{\rm SF}$",
    title=fr"Star-Forming Mass Fraction  ($\rho > {RHO_SF_THRESH:.0e}$ M$_\odot$ kpc$^{{-3}}$, "
          fr"$r < {R_SF_MAX_KPC:.0f}$ kpc)")
ax1a.plot(time_arr, sf_frac_arr, color="#ff9944", lw=1.8)
ax1a.fill_between(time_arr,
                  np.where(np.isfinite(sf_frac_arr), sf_frac_arr, 0),
                  alpha=0.15, color="#ff9944")
ax1a.set_ylim(0, 1.05)

_ax(ax1b, xlabel=time_label, ylabel=r"$r_{\rm SF}$ [kpc]",
    title="Star-Forming Radius  (enclosing 90% of SF-eligible mass)")
ax1b.plot(time_arr, r_sf_arr, color="#e8673a", lw=1.8)
ax1b.fill_between(time_arr,
                  np.where(np.isfinite(r_sf_arr), r_sf_arr, 0),
                  alpha=0.12, color="#e8673a")

fig1.savefig(os.path.join(OUT_DIR, "section25_sf_fraction.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print("  Saved: section25_sf_fraction.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — 2D MAP OF SF-ELIGIBLE REGIONS AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 2]  2D SF density maps …")

fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.04})

for col, (color, label) in enumerate(zip(PROFILE_COLORS, PROFILE_LABELS)):
    ax = axes2[col]
    ax.set_facecolor(BG)
    H  = sf_maps[col]
    Hs = gaussian_filter(np.where(H > 0, H, 0.0), sigma=2.0)
    H_log = np.where(Hs > 0, np.log10(Hs), np.nan)

    all_vals = H_log[np.isfinite(H_log)]
    vmin = np.percentile(all_vals, 5)  if all_vals.size > 0 else 0
    vmax = np.percentile(all_vals, 99) if all_vals.size > 0 else 10

    ax.imshow(H_log.T, origin="lower", aspect="equal",
              extent=[-SF_MAP_EXTENT, SF_MAP_EXTENT,
                      -SF_MAP_EXTENT, SF_MAP_EXTENT],
              cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title(label, fontsize=9, color="#c8c8e8")
    ax.tick_params(colors="#9090b0", labelsize=7)
    ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")
    if col == 0:
        ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")

fig2.suptitle(r"SF-Eligible Mass Distribution  $\Sigma_{\rm SF}(x,y)$  "
              fr"($\rho > {RHO_SF_THRESH:.0e}$, inner {SF_MAP_EXTENT:.0f} kpc)",
              fontsize=11)
fig2.savefig(os.path.join(OUT_DIR, "section25_sf_density_map.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("  Saved: section25_sf_density_map.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — TOOMRE Q(R) PROFILES AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════
#
# The Q = 1 line marks the boundary of gravitational instability.  Profiles
# dipping below Q = 1 at certain radii and epochs identify the annuli where
# star formation would be triggered in a hydrodynamic simulation.

print("[Fig 3]  Toomre Q profiles …")

fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax3, xlabel="R [kpc]", ylabel=r"Toomre $Q(R)$",
    title=r"Toomre Stability Parameter  $Q = \sigma_R \kappa / (\pi G \Sigma)$",
    log_x=True)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y     = Q_ts[k_idx, :]
    valid = np.isfinite(y) & (y > 0) & (y < 100)
    if valid.any():
        ax3.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax3.axhline(1.0, color="#ffffff", lw=1.0, ls="--", alpha=0.6,
            label="Q = 1  (marginal stability)")
ax3.fill_between([R_BINS[0], R_BINS[-1]], 0, 1,
                 alpha=0.06, color="#ff5566", label="Unstable (Q < 1)")
ax3.set_ylim(0, 10)
ax3.set_xlim(R_BINS[0], R_SF_MAX_KPC * 2)
ax3.legend(fontsize=7)

fig3.savefig(os.path.join(OUT_DIR, "section25_toomre_profile.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print("  Saved: section25_toomre_profile.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — TOOMRE Q(R, t) HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
#
# The heatmap shows which shells are unstable (Q < 1, warm colours) at each
# epoch.  Horizontal bands of instability at pericentre document the
# compression-driven star-formation trigger.

print("[Fig 4]  Toomre Q heatmap …")

Q_clipped = np.clip(Q_ts, 0.0, 5.0)

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                                   gridspec_kw={"width_ratios": [3,1], "wspace": 0.06})

im4 = ax4a.imshow(Q_clipped.T, aspect="auto", origin="lower",
                  extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
                  cmap="RdYlGn", vmin=0.0, vmax=3.0)
ax4a.set_yscale("log")
_ax(ax4a, xlabel=time_label, ylabel="R [kpc]",
    title=r"Toomre $Q(R,\,t)$  — red = unstable, green = stable")
ax4a.axhline(R_SF_MAX_KPC, color="#ffffff", lw=0.6, ls="--", alpha=0.3)
cb4 = fig4.colorbar(im4, ax=ax4a, pad=0.01)
cb4.set_label(r"$Q$", fontsize=9)
cb4.ax.axhline(1.0, color="white", lw=0.8, ls="--")

# Side panel: time-average Q(R).
Q_mean = np.nanmean(Q_ts, axis=0)
valid  = np.isfinite(Q_mean) & (Q_mean > 0) & (Q_mean < 100)
_ax(ax4b, xlabel=r"$\langle Q \rangle_t$", title="Time avg.")
ax4b.plot(Q_mean[valid], r_mid_sph[valid], color="#ff9944", lw=2.0)
ax4b.axvline(1.0, color="#ffffff", lw=0.8, ls="--", alpha=0.5)
ax4b.set_yscale("log")
ax4b.set_ylim(R_BINS[0], R_BINS[-1])
ax4b.set_xlim(0, 5)
ax4b.tick_params(labelleft=False)

fig4.savefig(os.path.join(OUT_DIR, "section25_toomre_heatmap.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig4)
print("  Saved: section25_toomre_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — SÉRSIC PROFILE FITS AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 5]  Sérsic profile fits …")

r_plot = np.logspace(np.log10(SERSIC_RMIN), np.log10(SERSIC_RMAX), 200)

fig5, axes5 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.06})

for col, (k_idx, color, label) in enumerate(zip(
        PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS)):
    ax = axes5[col]
    _ax(ax, xlabel="R [kpc]", title=label, log_x=True, log_y=True)

    # Recompute Σ for this snapshot (we only stored Sérsic params, not Σ_ts).
    # Use the already-available sersic_snap_map to find the nearest stored snap.
    snap_num = SNAPSHOTS[k_idx]
    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")
    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    try:
        sd = load_snapshot_particles(mw_file, m31_file)
    except Exception:
        continue

    pos_s  = sd["pos"]
    m_s    = sd["m_msun"]
    R_p    = np.sqrt(pos_s[:,0]**2 + pos_s[:,1]**2)
    ring_a = np.pi * (R_BINS[1:]**2 - R_BINS[:-1]**2)
    bid    = np.digitize(R_p, R_BINS) - 1
    Sig_s  = np.full(nb_sph, np.nan)
    for b in range(nb_sph):
        mask = bid == b
        if mask.sum() >= MIN_PART_SHELL:
            Sig_s[b] = m_s[mask].sum() / ring_a[b]

    valid = np.isfinite(Sig_s) & (Sig_s > 0)
    ax.scatter(r_mid_sph[valid], Sig_s[valid],
               color=color, s=14, alpha=0.8, zorder=3, label="Data")

    res = fit_sersic(r_mid_sph, Sig_s)
    if res["success"]:
        ax.plot(r_plot, sersic_profile(r_plot, res["I_e"], res["R_e"], res["n"]),
                color="white", lw=1.8,
                label=fr"Sérsic n={res['n']:.2f}, $R_e$={res['R_e']:.1f} kpc")

    if col == 0:
        ax.set_ylabel(r"$\Sigma$ [M$_\odot$ kpc$^{-2}$]", fontsize=9)
    ax.legend(fontsize=6)
    ax.set_xlim(SERSIC_RMIN, SERSIC_RMAX)

fig5.suptitle(r"Sérsic Profile Fits to $\Sigma(R)$", fontsize=12)
fig5.savefig(os.path.join(OUT_DIR, "section25_sersic_fit.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig5)
print("  Saved: section25_sersic_fit.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — SÉRSIC PARAMETERS n(t) AND R_e(t)
# ══════════════════════════════════════════════════════════════════════════════
#
# n increasing toward 4 signals the formation of a de Vaucouleurs (elliptical)
# profile — the classical end-state of a major merger.
# R_e shrinking documents the compaction of the remnant.

print("[Fig 6]  Sérsic parameter evolution …")

fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG,
                                   sharex=True, gridspec_kw={"hspace": 0.08})

valid_n  = np.isfinite(sersic_n_arr)
valid_Re = np.isfinite(sersic_Re_arr)

_ax(ax6a, ylabel="Sérsic index  n",
    title="Sérsic Profile Parameter Evolution")
ax6a.plot(time_sersic[valid_n], sersic_n_arr[valid_n],
          color="#aa55ff", lw=1.8, label="n")
ax6a.axhline(4.0, color="#ffffff", lw=0.7, ls="--", alpha=0.4,
             label="n = 4  (de Vaucouleurs / elliptical)")
ax6a.axhline(1.0, color="#555577", lw=0.7, ls=":",
             label="n = 1  (exponential disk)")
ax6a.legend(fontsize=7)

_ax(ax6b, xlabel=time_label, ylabel=r"$R_e$ [kpc]",
    title="Effective Radius Evolution")
ax6b.plot(time_sersic[valid_Re], sersic_Re_arr[valid_Re],
          color="#4a8fff", lw=1.8, label=r"$R_e$")
ax6b.legend(fontsize=8)

fig6.savefig(os.path.join(OUT_DIR, "section25_sersic_params.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig6)
print("  Saved: section25_sersic_params.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — STAR-FORMING RADIUS r_SF(t)
# ══════════════════════════════════════════════════════════════════════════════

print("[Fig 7]  Star-forming radius …")

fig7, ax7 = plt.subplots(figsize=(10, 4), facecolor=BG)
_ax(ax7, xlabel=time_label, ylabel=r"$r_{\rm SF}$ [kpc]",
    title="Star-Forming Radius  (90th percentile of SF-eligible mass)")
ax7.plot(time_arr, r_sf_arr, color="#e8673a", lw=1.8)
ax7.fill_between(time_arr,
                 np.where(np.isfinite(r_sf_arr), r_sf_arr, 0),
                 alpha=0.12, color="#e8673a")
ax7.axhline(R_SF_MAX_KPC, color="#555577", lw=0.7, ls="--", alpha=0.5,
            label=f"Search radius {R_SF_MAX_KPC:.0f} kpc")
ax7.legend(fontsize=8)

fig7.savefig(os.path.join(OUT_DIR, "section25_sf_radius.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig7)
print("  Saved: section25_sf_radius.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — CUMULATIVE SF MASS PROXY
# ══════════════════════════════════════════════════════════════════════════════
#
# The running integral ∫ M_sf(t) dt is a rough proxy for the cumulative
# stellar mass that could have formed if the SF efficiency were constant.
# The slope dM*/dt is steepest at pericentre — the expected starburst epoch.

print("[Fig 8]  Cumulative SF mass proxy …")

fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG,
                                   sharex=True, gridspec_kw={"hspace": 0.08})

_ax(ax8a, ylabel=r"$M_{\rm SF}$ [M$_\odot$]",
    title="Star-Forming Mass and Cumulative Proxy SFH", log_y=True)
valid_Msf = np.isfinite(M_sf_arr) & (M_sf_arr > 0)
ax8a.plot(time_arr[valid_Msf], M_sf_arr[valid_Msf],
          color="#ff9944", lw=1.6, label=r"$M_{\rm SF}(t)$")
ax8a.legend(fontsize=8)

_ax(ax8b, xlabel=time_label,
    ylabel=r"Cumulative $\int M_{\rm SF}\,dt$  [M$_\odot$ snap]",
    title="Proxy Cumulative Stellar Mass", log_y=True)
valid_cum = np.isfinite(cumM_sf_arr) & (cumM_sf_arr > 0)
ax8b.plot(time_arr[valid_cum], cumM_sf_arr[valid_cum],
          color="#00d4aa", lw=1.8)

fig8.savefig(os.path.join(OUT_DIR, "section25_cumulative_sfr.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig8)
print("  Saved: section25_cumulative_sfr.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9 — MASTER SUMMARY PANEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Summary]  Master summary panel …")

fig9 = plt.figure(figsize=(16, 10), facecolor=BG)
gs9  = gridspec.GridSpec(2, 2, figure=fig9,
                          hspace=0.38, wspace=0.32,
                          left=0.08, right=0.97,
                          top=0.93, bottom=0.07)

# (0,0) SF fraction.
ax_s00 = fig9.add_subplot(gs9[0, 0])
_ax(ax_s00, xlabel=time_label, ylabel=r"$f_{\rm SF}$",
    title="Star-Forming Fraction")
ax_s00.plot(time_arr, sf_frac_arr, color="#ff9944", lw=1.8)
ax_s00.set_ylim(0, 1.05)

# (0,1) Toomre Q heatmap.
ax_s01 = fig9.add_subplot(gs9[0, 1])
im_s01 = ax_s01.imshow(np.clip(Q_ts, 0, 3).T, aspect="auto",
                        origin="lower",
                        extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
                        cmap="RdYlGn", vmin=0, vmax=3)
ax_s01.set_yscale("log")
_ax(ax_s01, xlabel=time_label, ylabel="R [kpc]",
    title=r"Toomre $Q(R,t)$")
fig9.colorbar(im_s01, ax=ax_s01, shrink=0.8, label="Q")

# (1,0) Sérsic index.
ax_s10 = fig9.add_subplot(gs9[1, 0])
_ax(ax_s10, xlabel=time_label, ylabel="Sérsic n",
    title="Sérsic Index n(t)")
valid_n = np.isfinite(sersic_n_arr)
if valid_n.any():
    ax_s10.plot(time_sersic[valid_n], sersic_n_arr[valid_n],
                color="#aa55ff", lw=1.8)
ax_s10.axhline(4.0, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
ax_s10.axhline(1.0, color="#555577", lw=0.6, ls=":")

# (1,1) Effective radius.
ax_s11 = fig9.add_subplot(gs9[1, 1])
_ax(ax_s11, xlabel=time_label, ylabel=r"$R_e$ [kpc]",
    title=r"Effective Radius $R_e(t)$")
valid_Re = np.isfinite(sersic_Re_arr)
if valid_Re.any():
    ax_s11.plot(time_sersic[valid_Re], sersic_Re_arr[valid_Re],
                color="#4a8fff", lw=1.8)

fig9.suptitle("Section 25 Summary  ·  Stellar Mass & Star Formation Proxies",
              fontsize=13, color="#c8c8e8", fontweight="bold")
fig9.savefig(os.path.join(OUT_DIR, "section25_summary_panel.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig9)
print("  Saved: section25_summary_panel.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §25.5 — SECTION COMPLETE                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 25 COMPLETE")
print("="*80)
outputs_25 = [
    "section25_sf_fraction.png",
    "section25_sf_density_map.png",
    "section25_toomre_profile.png",
    "section25_toomre_heatmap.png",
    "section25_sersic_fit.png",
    "section25_sersic_params.png",
    "section25_sf_radius.png",
    "section25_cumulative_sfr.png",
    "section25_summary_panel.png",
]
for fn in outputs_25:
    fp   = os.path.join(OUT_DIR, fn)
    size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
    print(f"  {fn:<45} {size:6.2f} MB")
print("="*80)
