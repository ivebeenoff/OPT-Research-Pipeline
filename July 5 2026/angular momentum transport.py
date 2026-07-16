# """
# ==============================================================================
#      SECTION 21 — ANGULAR MOMENTUM TRANSPORT & PHASE-SPACE DIAGNOSTICS
# ==============================================================================
#
# Author  : Abhinav Vatsa
# Date    : April 20th 2026
#
# DESCRIPTION:
# Extension module built on density_pipeline.py. Inherits and requires all core 
# parent pipeline parameters, workspace definitions, data ingestion configurations,
# and snapshot alignment states.
#
# DYNAMICAL REGIMES TRACKED:
# While total angular momentum remains globally invariant within a closed, non-isolated 
# system, live galactic encounters induce severe time-dependent potential fluctuations. 
# These non-axisymmetric potentials continually redistribute angular momentum across:
#   (A) Progenitor stellar disks and hot dark matter envelopes.
#   (B) Spatial channels via inward/outward radial transport streams.
#   (C) Diffuse unbound structures carrying momentum out to massive tidal radii.
#
# This diagnostic engine quantifies these transfers across five coordinate frameworks:
#   §21.1: Specific Angular Momentum Vector Topography j(r, t)
#   §21.2: Radial Angular Momentum Flux Fields <j v_r>(r, t)
#   §21.3: Cumulative Integrated Enclosed Momentum Profiles L(<r, t)
#   §21.4: Spatial Component Decomposition (j_z, j_perp, Circularity parameter ε)
#   §21.5: Coarse-Grained Shannon Phase-Space Entropy S(r, j) & Mixing Scales
#
# ==============================================================================
# """

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
import warnings
import os
import time

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.0 — SUB-SYSTEM CONFIGURATION                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Phase-Space 2D Spatial Limits ─────────────────────────────────────────────
J_MIN_KPC_KMS = 0.0
J_MAX_KPC_KMS = 30000.0
N_JBINS       = 60                              # Momentum grid vertical resolution

# ── Orbital Circularity Bounds ────────────────────────────────────────────────
# Velocity ratio: ε = j_z / j_c(E) mapping bounded projections:
# ε -> +1 (Purely circular prograde), ε -> 0 (Radial trajectory), ε -> -1 (Retrograde)
N_EPS_BINS = 60

# ── Averaging Apertures ───────────────────────────────────────────────────────
J_INNER_KPC   = 30.0                            # Inner halo core bounds
J_OUTER_KPC   = 150.0                           # Dissipative envelope bounds
J_TRANSPORT_RMAX = 100.0                        # Flux transport cap limit

# ── Structural Entropy Matrix ─────────────────────────────────────────────────
ENTROPY_RBINS = 50                              # Spatial bins for Shannon mapping
ENTROPY_JBINS = 50                              # Momentum bins for Shannon mapping

# ── Temporal Animation Strides ────────────────────────────────────────────────
PHASESPACE_ANIM_STEP = 8                        # Decimation stride for 2D density maps
ANIM_FPS_21     = 20
ANIM_DPI_21     = 100
ANIM_BITRATE_21 = 2000
J_ANIM_STEP     = 4
N_GHOST         = 20                            # Trailing profile line depth

print("\n" + "="*80)
print("  SECTION 21 · Vectorized Angular Momentum Transport Engine Initialized")
print("="*80)
print(f"  Radial Channels : {nb_sph}")
print(f"  Momentum Space  : {J_MIN_KPC_KMS:.0f} - {J_MAX_KPC_KMS:.0f} kpc km/s")
print(f"  Entropy Mesh    : {ENTROPY_RBINS} x {ENTROPY_JBINS}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.1 — HIGH-PERFORMANCE VECTORIZED UTILITIES                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_angular_momentum(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes per-particle 3D specific angular momentum vectors.
    Conserves coordinates shell-by-shell to track non-axisymmetric scattering.
    """
    j_vec  = np.cross(pos, vel)
    j_mag  = np.linalg.norm(j_vec, axis=1)
    j_z    = j_vec[:, 2]
    j_perp = np.linalg.norm(j_vec[:, :2], axis=1)
    return j_vec, j_mag, j_z, j_perp, j_perp


def mass_weighted_bin(values: np.ndarray, r: np.ndarray, m: np.ndarray, r_bins: np.ndarray) -> np.ndarray:
    """
    Vectorized mass-weighted average engine utilizing fast binned statistics.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        num = binned_statistic(r, values * m, statistic="sum", bins=r_bins)[0]
        den = binned_statistic(r, m, statistic="sum", bins=r_bins)[0]
        counts = binned_statistic(r, m, statistic="count", bins=r_bins)[0]
        prof = num / den
        prof[counts < MIN_PART_SHELL] = np.nan
    return prof


def mass_weighted_std_bin(values: np.ndarray, r: np.ndarray, m: np.ndarray, r_bins: np.ndarray) -> np.ndarray:
    """
    Vectorized mass-weighted variance engine tracking phase-space dispersion.
    """
    nb = len(r_bins) - 1
    std_prof = np.full(nb, np.nan)
    bin_id = np.digitize(r, r_bins) - 1

    # Sorting particles by bin optimizes consecutive memory cache hit rates
    sort_idx = np.argsort(bin_id)
    bin_id, values, m = bin_id[sort_idx], values[sort_idx], m[sort_idx]
    
    split_splits = np.where(np.diff(bin_id) > 0)[0] + 1
    b_splits = np.split(np.arange(len(values)), split_splits)
    
    for idxs in b_splits:
        if len(idxs) < MIN_PART_SHELL:
            continue
        b_idx = bin_id[idxs[0]]
        if b_idx < 0 or b_idx >= nb:
            continue
        w = m[idxs]
        v = values[idxs]
        w_sum = w.sum()
        mean = np.sum(w * v) / w_sum
        std_prof[b_idx] = np.sqrt(np.sum(w * (v - mean)**2) / w_sum)
        
    return std_prof


def compute_orbital_circularity(j_z: np.ndarray, j_mag: np.ndarray, r_mag: np.ndarray, m_msun: np.ndarray, vc_arr: np.ndarray, r_bins: np.ndarray) -> np.ndarray:
    """
    Determines dimensionless eccentric circularity fractions: ε = j_z / j_c(r)
    """
    bin_id = np.clip(np.digitize(r_mag, r_bins) - 1, 0, len(vc_arr) - 1)
    vc_particle = np.where(np.isfinite(vc_arr[bin_id]), vc_arr[bin_id], np.nan)
    j_c = vc_particle * r_mag

    with np.errstate(invalid="ignore", divide="ignore"):
        epsilon = np.where(j_c > 0, j_z / j_c, np.nan)
    return epsilon


def compute_circular_velocity_profile(m_msun: np.ndarray, r_mag: np.ndarray, r_bins: np.ndarray) -> np.ndarray:
    """
    Generates circularized background potential profiles from integrated enclosed masses.
    """
    nb = len(r_bins) - 1
    vc = np.full(nb, np.nan)
    
    # Vectorized fast cumulative enclosure tracking via O(N log N) sorting
    sort_idx = np.argsort(r_mag)
    r_sorted = r_mag[sort_idx]
    m_cumsum = np.cumsum(m_msun[sort_idx])
    
    for b in range(nb):
        r_outer = r_bins[b + 1]
        search_pos = np.searchsorted(r_sorted, r_outer, side="right")
        M_encl = m_cumsum[search_pos - 1] if search_pos > 0 else 0.0
        if r_outer > 0 and M_encl > 0:
            vc[b] = np.sqrt(G_KPC_KMS2_MSUN * M_encl / r_outer)
    return vc


def phase_space_entropy(r_mag: np.ndarray, j_mag: np.ndarray, r_bins: int = ENTROPY_RBINS, j_bins: int = ENTROPY_JBINS, r_range: tuple = (0.1, 400.0), j_range: tuple = (J_MIN_KPC_KMS, J_MAX_KPC_KMS)) -> float:
    """
    Evaluates coarse-grained multi-dimensional Shannon phase-space entropy S(r, j).
    """
    H, _, _ = np.histogram2d(r_mag, j_mag, bins=[r_bins, j_bins], range=[r_range, j_range])
    total = H.sum()
    if total == 0:
        return np.nan
    P = H / total
    with np.errstate(divide="ignore", invalid="ignore"):
        S = -np.nansum(np.where(P > 0, P * np.log(P), 0.0))
    return float(S)


def compute_mixing_length(r_mag: np.ndarray, j_mag: np.ndarray, m_msun: np.ndarray, r_bins: np.ndarray) -> float:
    """
    Calculates spatial coherence decorrelation scales λ_mix using radial autocorrelation bounds.
    """
    j_prof = mass_weighted_bin(j_mag, r_mag, m_msun, r_bins)
    valid  = np.isfinite(j_prof)

    if valid.sum() < 4:
        return np.nan

    j_v = j_prof[valid]
    r_v = r_mid_sph[valid]

    j_cent = j_v - j_v.mean()
    norm = np.dot(j_cent, j_cent)
    if norm == 0:
        return np.nan

    nb_v = len(j_v)
    max_lag = nb_v // 2
    
    C = np.array([np.dot(j_cent[:nb_v - lag], j_cent[lag:]) / norm for lag in range(1, max_lag + 1)])
    dr_lags = np.array([r_v[lag] - r_v[0] for lag in range(1, max_lag + 1)])

    sign_changes = np.where(np.diff(np.sign(C)))[0]
    if len(sign_changes) == 0:
        return np.nan

    idx = sign_changes[0]
    dr_zero = dr_lags[idx] + (dr_lags[idx+1] - dr_lags[idx]) * (-C[idx] / (C[idx+1] - C[idx] + 1e-30))
    return float(dr_zero)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.2 — SYSTEM TENSOR ALLOCATION                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Pre-allocating time-series tensors to anchor consistent runtime memory
j_ts        = np.full((ns, nb_sph), np.nan)
j_std_ts    = np.full((ns, nb_sph), np.nan)
jz_ts       = np.full((ns, nb_sph), np.nan)
jperp_ts    = np.full((ns, nb_sph), np.nan)
jflux_ts    = np.full((ns, nb_sph), np.nan)
eps_ts      = np.full((ns, nb_sph), np.nan)
eps_std_ts  = np.full((ns, nb_sph), np.nan)
L_enc_ts    = np.full((ns, nb_sph), np.nan)

# Pre-allocating scalar arrays
j_inner_arr   = np.full(ns, np.nan)
j_outer_arr   = np.full(ns, np.nan)
j_total_arr   = np.full(ns, np.nan)
L_total_arr   = np.full(ns, np.nan)
entropy_arr   = np.full(ns, np.nan)
entropy_mw_arr  = np.full(ns, np.nan)
entropy_m31_arr = np.full(ns, np.nan)
mix_length_arr  = np.full(ns, np.nan)
transport_arr   = np.full(ns, np.nan)

# Frame-mapping allocation for animations
phasespace_snap_ids = np.arange(0, ns, PHASESPACE_ANIM_STEP)
n_ps_frames         = len(phasespace_snap_ids)
phasespace_hists    = np.zeros((n_ps_frames, ENTROPY_RBINS, ENTROPY_JBINS))

# Discrete array maps for tracking exact per-particle circularity histograms
eps_hist_store = np.zeros((ns, N_EPS_BINS))
eps_bin_edges = np.linspace(-1.05, 1.05, N_EPS_BINS + 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.3 — DYNAMIC EXECUTION ENGINE                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  §21.3 — Core Vectorized Phase-Space Run Loop")
print("="*80)

t_loop_start = time.perf_counter()
ps_frame_map = {SNAPSHOTS[idx]: fi for fi, idx in enumerate(phasespace_snap_ids)}

for i, snap_num in enumerate(SNAPSHOTS):
    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    try:
        snap_data = load_snapshot_particles(mw_file, m31_file)
    except Exception as exc:
        print(f"  [ERROR] Ingestion Failure at Snapshot {snap_num}: {exc}")
        continue

    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    origin = snap_data["origin"]

    r_mag = np.linalg.norm(pos, axis=1)

    # ── Phase-Space Coordinate Extraction & Alignment ──
    try:
        MW_obj  = CenterOfMass(mw_file,  PTYPE)
        M31_obj = CenterOfMass(m31_file, PTYPE)
    except Exception as exc:
        continue

    vx = np.concatenate((MW_obj.vx, M31_obj.vx))
    vy = np.concatenate((MW_obj.vy, M31_obj.vy))
    vz = np.concatenate((MW_obj.vz, M31_obj.vz))

    # Resolving iterative aperture velocity centers
    m_raw = np.concatenate((MW_obj.m, M31_obj.m))
    xcom, ycom, zcom = MW_obj.COMdefine(np.concatenate((MW_obj.x, M31_obj.x)),
                                       np.concatenate((MW_obj.y, M31_obj.y)),
                                       np.concatenate((MW_obj.z, M31_obj.z)), m_raw)

    dr_com = np.sqrt((np.concatenate((MW_obj.x, M31_obj.x)) - xcom)**2 +
                     (np.concatenate((MW_obj.y, M31_obj.y)) - ycom)**2 +
                     (np.concatenate((MW_obj.z, M31_obj.z)) - zcom)**2)
    inner_aperture = dr_com < 15.0
    
    if inner_aperture.sum() >= 5:
        w_inner = m[inner_aperture]
        vxcom = np.sum(w_inner * vx[inner_aperture]) / w_inner.sum()
        vycom = np.sum(w_inner * vy[inner_aperture]) / w_inner.sum()
        vzcom = np.sum(w_inner * vz[inner_aperture]) / w_inner.sum()
    else:
        vxcom = vycom = vzcom = 0.0

    vel = np.vstack((vx - vxcom, vy - vycom, vz - vzcom)).T

    # ── Map Phase-Space Components ──
    j_vec, j_mag, j_z, j_perp, _ = compute_angular_momentum(pos, vel)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = np.where(r_mag[:, None] > 0, pos / r_mag[:, None], 0.0)
    v_r = np.einsum("ij,ij->i", vel, r_hat)
    j_flux = j_mag * v_r

    # Circularization models
    vc_snap = compute_circular_velocity_profile(m, r_mag, R_BINS)
    epsilon = compute_orbital_circularity(j_z, j_mag, r_mag, m, vc_snap, R_BINS)

    # ── Radial Channel Statistics Synchronization ──
    j_ts     [i, :] = mass_weighted_bin(j_mag,  r_mag, m, R_BINS)
    j_std_ts [i, :] = mass_weighted_std_bin(j_mag, r_mag, m, R_BINS)
    jz_ts    [i, :] = mass_weighted_bin(j_z,   r_mag, m, R_BINS)
    jperp_ts [i, :] = mass_weighted_bin(j_perp, r_mag, m, R_BINS)
    jflux_ts [i, :] = mass_weighted_bin(j_flux, r_mag, m, R_BINS)

    valid_eps = np.isfinite(epsilon)
    if valid_eps.sum() > MIN_PART_SHELL:
        eps_ts    [i, :] = mass_weighted_bin(np.where(valid_eps, epsilon, 0.0), r_mag, m, R_BINS)
        eps_std_ts[i, :] = mass_weighted_std_bin(np.where(valid_eps, epsilon, 0.0), r_mag, m, R_BINS)

    # Dynamic tracking of raw particle distribution across bounded circularities
    if valid_eps.sum() > 10:
        eps_hist_store[i, :], _ = np.histogram(epsilon[valid_eps], bins=eps_bin_edges, weights=m[valid_eps])

    # Enclosed integration vectors
    bin_id = np.digitize(r_mag, R_BINS) - 1
    # Replacing the original nested b-loop with an optimized binned cumulative sum
    bin_mass_j = binned_statistic(r_mag, m * j_mag, statistic="sum", bins=R_BINS)[0]
    L_enc_ts[i, :] = np.cumsum(bin_mass_j)

    # ── Dynamic Scalar Reductions ──
    M_total = m.sum()
    j_total_arr[i] = np.sum(m * j_mag) / M_total

    inner_core = r_mag <= J_INNER_KPC
    outer_env  = r_mag >= J_OUTER_KPC
    if inner_core.sum() >= MIN_PART_SHELL:
        j_inner_arr[i] = np.sum(m[inner_core] * j_mag[inner_core]) / m[inner_core].sum()
    if outer_env.sum() >= MIN_PART_SHELL:
        j_outer_arr[i] = np.sum(m[outer_env] * j_mag[outer_env]) / m[outer_env].sum()

    L_total_arr[i] = L_enc_ts[i, -1]

    # Structural Shannon Entropy Reduction
    entropy_arr[i] = phase_space_entropy(r_mag, j_mag)
    mw_mask  = (origin == 0)
    m31_mask = (origin == 1)
    if mw_mask.sum() > 50:
        entropy_mw_arr[i] = phase_space_entropy(r_mag[mw_mask], j_mag[mw_mask])
    if m31_mask.sum() > 50:
        entropy_m31_arr[i] = phase_space_entropy(r_mag[m31_mask], j_mag[m31_mask])

    mix_length_arr[i] = compute_mixing_length(r_mag, j_mag, m, R_BINS)

    # Matrix mapping for specialized phase-space animations
    if snap_num in ps_frame_map:
        fi = ps_frame_map[snap_num]
        H, _, _ = np.histogram2d(r_mag, j_mag, bins=[ENTROPY_RBINS, ENTROPY_JBINS],
                                 range=[[0.1, 400.0], [J_MIN_KPC_KMS, J_MAX_KPC_KMS]], weights=m)
        phasespace_hists[fi] = H

    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t_loop_start
        print(f"  snap {snap_num:04d} | j_inner = {j_inner_arr[i]:.0f} | Entropy = {entropy_arr[i]:.3f} | λ_mix = {mix_length_arr[i]:.1f} kpc | [{elapsed:.0f}s elapsed]")

print(f"\n[Run complete] Matrix processing loop executed in {time.perf_counter() - t_loop_start:.0f}s total.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.4 — GRADIENT ANALYSIS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Continuous angular momentum transport indices derived across snapshot intervals
dj_dt  = np.gradient(j_ts, axis=0)
djz_dt = np.gradient(jz_ts, axis=0)
dL_dt  = np.gradient(L_enc_ts, axis=0)

# Localize scalar transport intensity
inner_r_mask = r_mid_sph < J_TRANSPORT_RMAX
transport_arr = np.nanmean(np.abs(dj_dt[:, inner_r_mask]), axis=1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.5 — COALESCING PLOT EXECUTIONS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

t_min, t_max = np.nanmin(time_arr), np.nanmax(time_arr)
BG, MUTED = "#0d0d18", "#7070a0"

def _styled_ax(ax, xlabel="", ylabel="", title="", log_x=False, log_y=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")
    ax.tick_params(colors="#9090b0", labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#c8c8e8")
    ax.set_ylabel(ylabel, fontsize=9, color="#c8c8e8")
    ax.set_title(title, fontsize=10, color="#c8c8e8", pad=5)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    return ax

# ── FIGURE 1 — j(r, t) MOMENTUM HEATMAP ──
print("\n[Fig 1] Plotting structural specific angular momentum heatmap...")
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06})

y_uniform_j = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), 200)
j_interp_map = np.zeros((len(y_uniform_j), ns))
for snap_idx in range(ns):
    nm = np.isfinite(j_ts[snap_idx, :])
    j_interp_map[:, snap_idx] = np.interp(np.log10(y_uniform_j), np.log10(r_mid_sph[nm]), j_ts[snap_idx, nm]) if nm.sum() > 2 else np.nan

im1 = ax1a.imshow(j_interp_map, aspect="auto", origin="lower", extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="plasma",
                  norm=LogNorm(vmin=np.nanpercentile(j_ts[j_ts > 0], 5), vmax=np.nanpercentile(j_ts[j_ts > 0], 99)))
_styled_ax(ax1a, xlabel=time_label, ylabel="r [kpc]", title=r"Specific Angular Momentum Profile: $j(r,t)\ [{\rm kpc\ km\ s^{-1}}]$")
ax1a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax1a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])

cb1 = fig1.colorbar(im1, ax=ax1a, pad=0.01)
cb1.set_label(r"$j\ [{\rm kpc\ km\ s^{-1}}]$", fontsize=8)

j_mean = np.nanmean(j_ts, axis=0)
valid = np.isfinite(j_mean)
_styled_ax(ax1b, xlabel=r"$\langle j \rangle_t$", title="Time Avg", log_x=False, log_y=True)
ax1b.plot(j_mean[valid], r_mid_sph[valid], color="#ff9944", lw=2.0)
ax1b.set_ylim(R_BINS[0], R_BINS[-1])
ax1b.tick_params(labelleft=False)

fig1.suptitle("Angular Momentum Profile Evolution", fontsize=12)
fig1.savefig(os.path.join(OUT_DIR, "section21_j_heatmap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig1)

# ── FIGURE 2 — RADIAL ANGULAR MOMENTUM FLUX HEATMAP ──
print("[Fig 2] Plotting radial flux maps...")
flux_max = np.nanpercentile(np.abs(jflux_ts[np.isfinite(jflux_ts)]), 97)
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06})

flux_interp_map = np.zeros((len(y_uniform_j), ns))
for snap_idx in range(ns):
    nm = np.isfinite(jflux_ts[snap_idx, :])
    flux_interp_map[:, snap_idx] = np.interp(np.log10(y_uniform_j), np.log10(r_mid_sph[nm]), jflux_ts[snap_idx, nm]) if nm.sum() > 2 else np.nan

im2 = ax2a.imshow(flux_interp_map, aspect="auto", origin="lower", extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="coolwarm",
                  norm=TwoSlopeNorm(vmin=-flux_max, vcenter=0.0, vmax=flux_max))
_styled_ax(ax2a, xlabel=time_label, ylabel="r [kpc]", title=r"Radial Flux Transfer Fields: $\langle j v_r \rangle(r,t)$")
ax2a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax2a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])

cb2 = fig2.colorbar(im2, ax=ax2a, pad=0.01)
cb2.set_label(r"$\langle j v_r \rangle\ [{\rm kpc^2\ km^2\ s^{-2}}]$", fontsize=8)

flux_mean = np.nanmean(jflux_ts, axis=0)
valid_f = np.isfinite(flux_mean)
_styled_ax(ax2b, xlabel=r"$\langle j v_r \rangle$", title="Time Avg")
ax2b.plot(flux_mean[valid_f], r_mid_sph[valid_f], color="#00d4aa", lw=2.0)
ax2b.axvline(0, color="#555577", lw=0.8, ls="--")
ax2b.set_yscale("log")
ax2b.set_ylim(R_BINS[0], R_BINS[-1])
ax2b.tick_params(labelleft=False)

fig2.suptitle("Radial Flux Cross-Sections", fontsize=12)
fig2.savefig(os.path.join(OUT_DIR, "section21_j_flux_heatmap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig2)

# ── FIGURE 3 — FLUX PROFILES AT FIVE EPOCHS ──
print("[Fig 3] Plotting discrete flux profile overlays...")
fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor=BG)
_styled_ax(ax3, xlabel="r [kpc]", ylabel=r"$\langle j v_r \rangle\ [{\rm kpc^2\ km^2\ s^{-2}}]$", title=r"Angular Momentum Flux Profiles: $\langle j v_r \rangle(r)$", log_x=True)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y = jflux_ts[k_idx, :]
    valid = np.isfinite(y)
    if valid.any():
        ax3.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax3.axhline(0, color="#555577", lw=1.0, ls="--", alpha=0.6)
ax3.text(R_BINS[0] * 1.2, 0, "Outward Transport ->", color=MUTED, fontsize=7, va="bottom")
ax3.text(R_BINS[0] * 1.2, 0, "<- Inward Flux", color=MUTED, fontsize=7, va="top")
ax3.set_xlim(R_BINS[0], R_BINS[-1])
ax3.legend(fontsize=8)
fig3.savefig(os.path.join(OUT_DIR, "section21_jflux_profiles.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig3)

# ── FIGURE 4 — CUMULATIVE ENCLOSED MASS BOUNDS L(<r) ──
print("[Fig 4] Plotting cumulative integrated mass-momentum boundaries...")
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG, gridspec_kw={"wspace": 0.32})

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    L_row = L_enc_ts[k_idx, :]
    valid = np.isfinite(L_row) & (L_row > 0)
    _styled_ax(ax4a, xlabel="r [kpc]", ylabel=r"$L(<r)\ [{\rm M_\odot\ kpc\ km\ s^{-1}}]$", title=r"Cumulative Enclosed Budget $L(<r)$", log_x=True, log_y=True)
    if valid.any():
        ax4a.plot(r_mid_sph[valid], L_row[valid], color=color, lw=2.0, label=label)

    _styled_ax(ax4b, xlabel="r [kpc]", ylabel=r"$L(<r)\ /\ L_{\rm tot}$", title=r"Fractional Angular Distribution", log_x=True)
    if valid.any():
        ax4b.plot(r_mid_sph[valid], L_row[valid] / L_row[valid][-1], color=color, lw=2.0, label=label)

ax4a.legend(fontsize=7)
ax4b.axhline(0.5, color="#555577", lw=0.8, ls="--")
ax4b.set_ylim(0, 1.05)
fig4.savefig(os.path.join(OUT_DIR, "section21_L_enclosed_profiles.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig4)

# ── FIGURE 5 — CORE vs ENVELOPE SCALAR CHRONOLOGY ──
print("[Fig 5] Plotting specialized scalar evolution curves...")
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})
_styled_ax(ax5a, ylabel=r"$j\ [{\rm kpc\ km\ s^{-1}}]$", title=r"Spatial Sub-Domain Angular Momentum Evolution")
ax5a.plot(time_arr, j_inner_arr, color="#4a8fff", lw=1.8, label=fr"Inner Core Core ($r < {J_INNER_KPC:.0f}$ kpc)")
ax5a.plot(time_arr, j_outer_arr, color="#ff9944", lw=1.8, label=fr"Outer Extended Envelope ($r > {J_OUTER_KPC:.0f}$ kpc)")
ax5a.plot(time_arr, j_total_arr, color="#aaaacc", lw=1.0, ls=":", label="Bulk Average System")
ax5a.legend(fontsize=8)

_styled_ax(ax5b, xlabel=time_label, ylabel=r"$|\partial j / \partial t|\ [{\rm kpc\ km\ s^{-1}\ /\ snap}]$", title=r"Local Core Instability Rate: $\langle |\partial j / \partial t| \rangle$")
ax5b.plot(time_arr, transport_arr, color="#e8673a", lw=1.5)
ax5b.fill_between(time_arr, np.where(np.isfinite(transport_arr), transport_arr, 0), alpha=0.15, color="#e8673a")
fig5.savefig(os.path.join(OUT_DIR, "section21_j_inner_outer.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig5)

# ── FIGURE 6 — COMPONENT MOMENTUM VECTORS & ε ACCURACY ──
print("[Fig 6] Plotting structural component vectors...")
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG, gridspec_kw={"wspace": 0.32})
ax_jz, ax_jp, ax_eps = axes6

for ax, data_ts, ylabel, title in [(ax_jz, jz_ts, r"$j_z\ [{\rm kpc\ km\ s^{-1}}]$", r"Coherent Rotation: $j_z(r)$"),
                                   (ax_jp, jperp_ts, r"$j_\perp\ [{\rm kpc\ km\ s^{-1}}]$", r"Out-of-Plane Dispersion: $j_\perp(r)$"),
                                   (ax_eps, eps_ts, r"Mean Circularity parameter $\varepsilon$", r"Orbital Alignment Index $\varepsilon(r)$")]:
    _styled_ax(ax, xlabel="r [kpc]", ylabel=ylabel, title=title, log_x=True)
    ax.set_xlim(R_BINS[0], R_BINS[-1])
    for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
        y = data_ts[k_idx, :]
        valid = np.isfinite(y)
        if valid.any(): ax.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax_jz.axhline(0, color="#555577", lw=0.8, ls="--")
ax_eps.axhline(0, color="#555577", lw=0.8, ls="--")
ax_eps.set_ylim(-1.1, 1.1)
ax_eps.legend(fontsize=7)
fig6.suptitle("Structural Vector Parameter Overlays", fontsize=12)
fig6.savefig(os.path.join(OUT_DIR, "section21_component_profiles.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig6)

# ── FIGURE 7 — ORIENTATION ANALYSIS HEATMAP j_z(r, t) ──
print("[Fig 7] Plotting coherent rotational heatmaps...")
jz_abs_max = np.nanpercentile(np.abs(jz_ts[np.isfinite(jz_ts)]), 97)
fig7, ax7 = plt.subplots(figsize=(11, 5), facecolor=BG)

jz_interp_map = np.zeros((len(y_uniform_j), ns))
for snap_idx in range(ns):
    nm = np.isfinite(jz_ts[snap_idx, :])
    jz_interp_map[:, snap_idx] = np.interp(np.log10(y_uniform_j), np.log10(r_mid_sph[nm]), jz_ts[snap_idx, nm]) if nm.sum() > 2 else np.nan

im7 = ax7.imshow(jz_interp_map, aspect="auto", origin="lower", extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="RdBu_r",
                  norm=TwoSlopeNorm(vmin=-jz_abs_max, vcenter=0.0, vmax=jz_abs_max))
_styled_ax(ax7, xlabel=time_label, ylabel="r [kpc]", title=r"Rotational Directivity Map $j_z(r,t)$: Prograde (Red) vs. Retrograde (Blue)")
ax7.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax7.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
cb7 = fig7.colorbar(im7, ax=ax7, pad=0.01)
cb7.set_label(r"$j_z\ [{\rm kpc\ km\ s^{-1}}]$", fontsize=8)
fig7.savefig(os.path.join(OUT_DIR, "section21_jz_heatmap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig7)

# ── FIGURE 8 — ORBITAL ALIGNMENT CHRONOLOGY ε(r, t) ──
print("[Fig 8] Plotting eccentricity circularity heatmaps...")
fig8, ax8 = plt.subplots(figsize=(11, 5), facecolor=BG)

eps_interp_map = np.zeros((len(y_uniform_j), ns))
for snap_idx in range(ns):
    nm = np.isfinite(eps_ts[snap_idx, :])
    eps_interp_map[:, snap_idx] = np.interp(np.log10(y_uniform_j), np.log10(r_mid_sph[nm]), eps_ts[snap_idx, nm]) if nm.sum() > 2 else np.nan

im8 = ax8.imshow(np.clip(eps_interp_map, -1.0, 1.0), aspect="auto", origin="lower", extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="bwr", vmin=-1.0, vmax=1.0)
_styled_ax(ax8, xlabel=time_label, ylabel="r [kpc]", title=r"Circular Orbit Stability Tracking: $\varepsilon(r,t) = j_z / j_c(r)$")
ax8.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax8.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
cb8 = fig8.colorbar(im8, ax=ax8, pad=0.01)
cb8.set_label(r"Alignment Index $\varepsilon$ (Blue = Retrograde, Red = Circular Prograde)", fontsize=8)
fig8.savefig(os.path.join(OUT_DIR, "section21_circularity_heatmap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig8)

# ── FIGURE 9 — RAW PARTICLE DISTRIBUTION HISTOGRAMS ──
print("[Fig 9] Plotting true un-approximated orbital circularity distributions...")
fig9, axes9 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG, sharey=True, gridspec_kw={"wspace": 0.06})
mid_eps_bins = 0.5 * (eps_bin_edges[:-1] + eps_bin_edges[1:])

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS)):
    ax = axes9[col]
    _styled_ax(ax, xlabel=r"Circularity $\varepsilon$", title=label)
    
    # Render true binned counts populated natively during the snapshot loop
    h_data = eps_hist_store[k_idx, :]
    if h_data.sum() > 0:
        norm_h = h_data / h_data.max()
        ax.fill_between(mid_eps_bins, norm_h, alpha=0.3, color=color)
        ax.plot(mid_eps_bins, norm_h, color=color, lw=1.5)
        
    ax.axvline(0, color="#555577", lw=0.7, os="--")
    if col == 0: ax.set_ylabel("Normalized Mass Fraction $P(\varepsilon)$", fontsize=9)

fig9.suptitle(r"Global Orbital Circularity Distribution Dynamics Across Epochs $P(\varepsilon)$", fontsize=11)
fig9.savefig(os.path.join(OUT_DIR, "section21_circularity_dist.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig9)

# ── FIGURE 10 — COARSE-GRAINED LOG ENTROPY TRACES ──
print("[Fig 10] Plotting multi-component Shannon entropy diagnostics...")
fig10, (ax10a, ax10b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})
_styled_ax(ax10a, ylabel=r"$S(r,j)\ [{\rm nats}]$", title=r"Total System Phase-Space Chaos Indices")
ax10a.plot(time_arr, entropy_arr, color="#e8673a", lw=1.8, label="Cumulative Particles")
ax10a.legend(fontsize=8)

_styled_ax(ax10b, xlabel=time_label, ylabel=r"$S\ [{\rm nats}]$", title="Entropy Divergence by Progenitor Core")
ax10b.plot(time_arr, entropy_mw_arr,  color="#4a8fff", lw=1.8, label="Milky Way System")
ax10b.plot(time_arr, entropy_m31_arr, color="#ff5fa0", lw=1.8, label="Andromeda System")
ax10b.legend(fontsize=8)
fig10.savefig(os.path.join(OUT_DIR, "section21_entropy.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig10)

# ── FIGURE 11 — AUTOCORRELATION SCALE LENGTH TRACKING ──
print("[Fig 11] Plotting angular coherence scales...")
fig11, ax11 = plt.subplots(figsize=(10, 4), facecolor=BG)
_styled_ax(ax11, xlabel=time_label, ylabel=r"$\lambda_{\rm mix}\ [{\rm kpc}]$", title=r"Autocorrelation Scale Profile Length: $\lambda_{\rm mix}(t)$")
ax11.plot(time_arr, mix_length_arr, color="#00d4aa", lw=1.8)
ax11.fill_between(time_arr, np.where(np.isfinite(mix_length_arr), mix_length_arr, 0), alpha=0.12, color="#00d4aa")
ax11.axhline(3.5, color="#ffcc44", lw=0.8, ls="--", label="Initial Stellar Disk Scale Scale (~3.5 kpc)")
ax11.legend(fontsize=8)
fig11.savefig(os.path.join(OUT_DIR, "section21_mixing_length.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig11)

# ── FIGURE 12 — PHASE-SPACE MULTI-DECADE MESHING ──
print("[Fig 12] Plotting 2D phase-space density matrices...")
fig12, axes12 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG, sharey=True, gridspec_kw={"wspace": 0.06})
ps_snap_nums = SNAPSHOTS[phasespace_snap_ids]

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS)):
    ax = axes12[col]
    _styled_ax(ax, xlabel="r [kpc]", title=label)
    if col == 0: ax.set_ylabel(r"$j = |r \times v|\ [{\rm kpc\ km\ s^{-1}}]$", fontsize=8)

    target_snap = SNAPSHOTS[k_idx]
    nearest_fi  = np.argmin(np.abs(ps_snap_nums - target_snap))
    H           = phasespace_hists[nearest_fi]
    H_log       = np.where(H > 0, np.log10(H), np.nan)

    ax.imshow(H_log.T, origin="lower", aspect="auto", extent=[0.1, 400.0, J_MIN_KPC_KMS, J_MAX_KPC_KMS], cmap="inferno")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 400.0)
    ax.set_ylim(J_MIN_KPC_KMS, J_MAX_KPC_KMS)

fig12.suptitle(r"2D Phase-Space Concentration Landscapes Across Time $(r,j)$", fontsize=12)
fig12.savefig(os.path.join(OUT_DIR, "section21_phasespace_snap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig12)

# ── FIGURE 13 — QUANTITATIVE EXHAUSTIVE SYSTEM BUDGETS ──
print("[Fig 13] Compiling embedded latex data tables...")
fig13, ax13 = plt.subplots(figsize=(13, 3.5), facecolor=BG)
ax13.set_facecolor(BG)
ax13.axis("off")

col_headers = ["Epoch Phase", r"$j_{\rm inner}$ Core", r"$j_{\rm outer}$ Envelope", r"$j_{\rm total}$ Bulk", r"Integrated $L_{\rm enc}$", r"Entropy $S$ [nats]", r"Decorrelation $\lambda_{\rm mix}$ [kpc]"]
row_data = []
for k_idx, label in zip(PROFILE_INDICES, PROFILE_LABELS):
    row_data.append([label, f"{j_inner_arr[k_idx]:.0f}", f"{j_outer_arr[k_idx]:.0f}", f"{j_total_arr[k_idx]:.0f}", f"{L_total_arr[k_idx]:.2e}", f"{entropy_arr[k_idx]:.3f}", f"{mix_length_arr[k_idx]:.1f}"])

tbl = ax13.table(cellText=row_data, colLabels=col_headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)

for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a3a" if r == 0 else ("#0d0d18" if r % 2 == 0 else "#141428"))
    cell.set_edgecolor("#2a2a4a")
    cell.set_text_props(color="#c8c8e8")

ax13.set_title("Conserved Mass-Momentum Supplementary System Parameters", fontsize=11, color="#c8c8e8", pad=12)
fig13.savefig(os.path.join(OUT_DIR, "section21_transport_budget.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig13)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.6 — MEDIA GENERATION ENGINES                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── ANIMATION 1 — HISTORIC GHOST TRACKING PROFILES ──
print("\n[Anim 1] Initializing trailing ghost line profile animations...")
j_anim_idxs = np.arange(0, ns, J_ANIM_STEP)
n_j_frames  = len(j_anim_idxs)
cmap_time   = plt.cm.plasma

fig_a1, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=BG, gridspec_kw={"wspace": 0.32})
for ax in (axA, axB):
    ax.set_facecolor(BG)
    ax.set_xscale("log")

j_finite = j_ts[np.isfinite(j_ts) & (j_ts > 0)]
j_ymin, j_ymax = (j_finite.min() * 0.5, j_finite.max() * 2.0) if j_finite.size > 0 else (1.0, 1e5)
axA.set_ylim(j_ymin, j_ymax); axA.set_yscale("log")
axB.set_ylim(0, np.nanpercentile(j_std_ts[np.isfinite(j_std_ts)], 99) * 1.2)
axA.set_xlim(R_BINS[0], R_BINS[-1]); axB.set_xlim(R_BINS[0], R_BINS[-1])

axA.set_xlabel("r [kpc]", color="#c8c8e8"); axA.set_ylabel(r"$j(r)\ [{\rm kpc\ km\ s^{-1}}]$", color="#c8c8e8")
axA.set_title(r"Temporal Specific Profile: $j(r)$", color="#c8c8e8")
axB.set_xlabel("r [kpc]", color="#c8c8e8"); axB.set_ylabel(r"$\sigma_j(r)\ [{\rm kpc\ km\ s^{-1}}]$", color="#c8c8e8")
axB.set_title(r"Momentum Phase Dispersion: $\sigma_j(r)$", color="#c8c8e8")

ghost_lines = [axA.plot([], [], lw=0.8)[0] for _ in range(N_GHOST)]
main_j_line, = axA.plot([], [], lw=2.2, color="white", zorder=5)
std_j_line,  = axB.plot([], [], lw=2.0, color="#e8673a")
title_a1 = fig_a1.suptitle("", fontsize=11, color="#c8c8e8")

def _update_j_anim(frame_idx):
    snap_i = j_anim_idxs[frame_idx]
    color  = cmap_time(frame_idx / n_j_frames)

    def _xy(arr):
        v = np.isfinite(arr) & (arr > 0)
        return r_mid_sph[v], arr[v]

    rx, ry = _xy(j_ts[snap_i, :])
    main_j_line.set_data(rx, ry)
    main_j_line.set_color(color)

    for g, ghost in enumerate(ghost_lines):
        past_idx = frame_idx - (N_GHOST - g)
        if past_idx < 0:
            ghost.set_data([], [])
            continue
        past_snap  = j_anim_idxs[past_idx]
        px, py     = _xy(j_ts[past_snap, :])
        ghost.set_data(px, py)
        ghost.set_color(cmap_time(past_idx / n_j_frames))
        ghost.set_alpha(0.06 + 0.06 * g)

    std_row = j_std_ts[snap_i, :]
    sv = np.isfinite(std_row)
    std_j_line.set_data(r_mid_sph[sv], std_row[sv])

    t_val = time_arr[snap_i]
    title_a1.set_text(f"Angular Momentum Profile Flow Dynamics  ·  ({t_val:.2f} Gyr)" if time_is_gyr else f"Angular Momentum Profile Flow Dynamics  ·  Snap {SNAPSHOTS[snap_i]}")
    return [main_j_line, std_j_line] + ghost_lines

ani_j = animation.FuncAnimation(fig_a1, _update_j_anim, frames=n_j_frames, interval=1000 // ANIM_FPS_21, blit=True)
writer_j = animation.FFMpegWriter(fps=ANIM_FPS_21, bitrate=ANIM_BITRATE_21, metadata=dict(title="MW-M31 Structural Momentum Profiles"))
ani_j.save(os.path.join(OUT_DIR, "section21_animation_j.mp4"), writer=writer_j, dpi=ANIM_DPI_21)
plt.close(fig_a1)

# ── ANIMATION 2 — (r, j) PHASE-SPACE CONTINUOUS MATRICES ──
print("[Anim 2] Initializing dynamic 2D phase-space matrix media loops...")
H_all    = phasespace_hists[phasespace_hists > 0]
H_logmin = np.log10(np.percentile(H_all, 5))  if H_all.size > 0 else 0
H_logmax = np.log10(np.percentile(H_all, 99)) if H_all.size > 0 else 10

fig_a2, (axPS, axMarg) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06})
axPS.set_facecolor(BG); axMarg.set_facecolor(BG)

first_H     = phasespace_hists[0]
first_H_log = np.where(first_H > 0, np.log10(first_H), np.nan)

im_ps = axPS.imshow(first_H_log.T, origin="lower", aspect="auto", extent=[0.1, 400.0, J_MIN_KPC_KMS, J_MAX_KPC_KMS], cmap="magma", vmin=H_logmin, vmax=H_logmax)
axPS.set_xscale("log")
axPS.set_xlabel("r [kpc]", color="#c8c8e8"); axPS.set_ylabel(r"$j\ [{\rm kpc\ km\ s^{-1}}]$", color="#c8c8e8")

j_marg     = first_H.sum(axis=0)
j_axis_arr = np.linspace(J_MIN_KPC_KMS, J_MAX_KPC_KMS, ENTROPY_JBINS)
marg_line, = axMarg.plot(j_marg / (j_marg.max() + 1e-30), j_axis_arr, color="#ff9944", lw=1.8)
axMarg.set_xlabel("Integrated Projected Mass fraction", color="#c8c8e8")
axMarg.set_ylim(J_MIN_KPC_KMS, J_MAX_KPC_KMS)
axMarg.tick_params(labelleft=False)
axMarg.set_title(r"Marginal $P(j)$", color="#c8c8e8", fontsize=10)

title_a2 = fig_a2.suptitle("", fontsize=11, color="#c8c8e8")

def _update_ps_anim(frame_idx):
    H     = phasespace_hists[frame_idx]
    im_ps.set_data(np.where(H > 0, np.log10(H), np.nan).T)
    j_m   = H.sum(axis=0)
    marg_line.set_xdata(j_m / (j_m.max() + 1e-30))

    snap_i  = phasespace_snap_ids[frame_idx]
    t_val   = time_arr[snap_i]
    title_a2.set_text(fr"Phase-Space Density Continuums $(r,j)$  ·  ({t_val:.2f} Gyr)" if time_is_gyr else fr"Phase-Space Density Continuums $(r,j)$  ·  Snap {SNAPSHOTS[snap_i]}")
    return [im_ps, marg_line]

ani_ps = animation.FuncAnimation(fig_a2, _update_ps_anim, frames=n_ps_frames, interval=1000 // ANIM_FPS_21, blit=True)
ani_ps.save(os.path.join(OUT_DIR, "section21_animation_phasespace.mp4"), writer=writer_j, dpi=ANIM_DPI_21)
plt.close(fig_a2)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.7 — EXHAUSTIVE SUMMARY LAYOUT GRID                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

print("\n[Summary] Consolidating sub-system variables into publication panel maps...")
fig_sum = plt.figure(figsize=(16, 14), facecolor=BG)
gs_sum  = gridspec.GridSpec(3, 2, figure=fig_sum, hspace=0.42, wspace=0.32, left=0.08, right=0.97, top=0.94, bottom=0.06)

# (0,0) j(r,t)
ax_s00 = fig_sum.add_subplot(gs_sum[0, 0]); ax_s00.set_facecolor(BG)
im_s00 = ax_s00.imshow(j_interp_map, aspect="auto", origin="lower", extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="plasma",
                       norm=LogNorm(vmin=np.nanpercentile(j_ts[j_ts > 0], 5), vmax=np.nanpercentile(j_ts[j_ts > 0], 99)))
ax_s00.set_yscale("log"); ax_s00.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0]))); ax_s00.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
ax_s00.set_xlabel(time_label, fontsize=8, color="#c8c8e8"); ax_s00.set_ylabel("r [kpc]", fontsize=8, color="#c8c8e8"); ax_s00.set_title(r"Mean Specific Vector Field $j(r,t)$", fontsize=9, color="#c8c8e8")
fig_sum.colorbar(im_s00, ax=ax_s00, shrink=0.8, label="[kpc km/s]")

# (0,1) j_z(r,t)
ax_s01 = fig_sum.add_subplot(gs_sum[0, 1]); ax_s01.set_facecolor(BG)
im_s01 = ax_s01.imshow(jz_interp_map, aspect="auto", origin="lower", extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-jz_abs_max, vcenter=0, vmax=jz_abs_max))
ax_s01.set_yscale("log"); ax_s01.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0]))); ax_s01.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
ax_s01.set_xlabel(time_label, fontsize=8, color="#c8c8e8"); ax_s01.set_title(r"Rotational Directivity $j_z(r,t)$", fontsize=9, color="#c8c8e8")
fig_sum.colorbar(im_s01, ax=ax_s01, shrink=0.8, label="[kpc km/s]")

# (1,0) Sub-aperture histories
ax_s10 = fig_sum.add_subplot(gs_sum[1, 0]); ax_s10.set_facecolor(BG)
ax_s10.plot(time_arr, j_inner_arr, color="#4a8fff", lw=1.5, label="Core")
ax_s10.plot(time_arr, j_outer_arr, color="#ff9944", lw=1.5, label="Envelope")
ax_s10.set_xlabel(time_label, fontsize=8, color="#c8c8e8"); ax_s10.set_ylabel(r"$j\ [{\rm kpc\ km\ s^{-1}}]$", fontsize=8, color="#c8c8e8")
ax_s10.set_title("Sub-Aperture Chronology", fontsize=9, color="#c8c8e8"); ax_s10.legend(fontsize=7)

# (1,1) Structural Shannon Entropy
ax_s11 = fig_sum.add_subplot(gs_sum[1, 1]); ax_s11.set_facecolor(BG)
ax_s11.plot(time_arr, entropy_arr, color="#e8673a", lw=1.5)
ax_s11.set_xlabel(time_label, fontsize=8, color="#c8c8e8"); ax_s11.set_ylabel(r"$S\ [{\rm nats}]$", fontsize=8, color="#c8c8e8")
ax_s11.set_title("Coarse Shannon Entropy S(t)", fontsize=9, color="#c8c8e8")

# (2,0) Discretized flux profiles
ax_s20 = fig_sum.add_subplot(gs_sum[2, 0]); ax_s20.set_facecolor(BG); ax_s20.set_xscale("log")
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y = jflux_ts[k_idx, :]
    if np.isfinite(y).any(): ax_s20.plot(r_mid_sph[np.isfinite(y)], y[np.isfinite(y)], color=color, lw=1.5, label=label)
ax_s20.axhline(0, color="#555577", lw=0.7, ls="--")
ax_s20.set_xlabel("r [kpc]", fontsize=8, color="#c8c8e8"); ax_s20.set_ylabel(r"$\langle j v_r \rangle$", fontsize=8, color="#c8c8e8")
ax_s20.set_title("Radial Flux Overlays", fontsize=9, color="#c8c8e8"); ax_s20.legend(fontsize=6)

# (2,1) Autocorrelation Mixing scales
ax_s21 = fig_sum.add_subplot(gs_sum[2, 1]); ax_s21.set_facecolor(BG)
ax_s21.plot(time_arr, mix_length_arr, color="#00d4aa", lw=1.5)
ax_s21.axhline(3.5, color="#ffcc44", lw=0.7, ls="--", alpha=0.7, label="Initial Disk Scale")
ax_s21.set_xlabel(time_label, fontsize=8, color="#c8c8e8"); ax_s21.set_ylabel(r"$\lambda_{\rm mix}\ [{\rm kpc}]$", fontsize=8, color="#c8c8e8")
ax_s21.set_title("Momentum Autocorrelation Length", fontsize=9, color="#c8c8e8"); ax_s21.legend(fontsize=7)

fig_sum.suptitle("Section 21 Master Summary Grid Panel: Angular Momentum Transport Diagnostics", fontsize=13, color="#c8c8e8", fontweight="bold")
fig_sum.savefig(os.path.join(OUT_DIR, "section21_summary_panel.png"), dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig_sum)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §21.8 — STORAGE REPORT MANIFEST                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

print("\n" + "="*80)
print("  SECTION 21 SYSTEM MANIFEST")
print("="*80)
outputs_21 = [
    "section21_j_heatmap.png", "section21_j_flux_heatmap.png", "section21_jflux_profiles.png",
    "section21_L_enclosed_profiles.png", "section21_j_inner_outer.png", "section21_component_profiles.png",
    "section21_jz_heatmap.png", "section21_circularity_heatmap.png", "section21_circularity_dist.png",
    "section21_entropy.png", "section21_mixing_length.png", "section21_phasespace_snap.png",
    "section21_transport_budget.png", "section21_animation_j.mp4", "section21_animation_phasespace.mp4",
    "section21_summary_panel.png"
]
for fn in outputs_21:
    fp = os.path.join(OUT_DIR, fn)
    size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
    kind = "Stream Media (mp4)" if fn.endswith(".mp4") else "Static Panel (png)"
    print(f"  {fn:<50} {size:10.2f} MB  [{kind}]")
print("="*80)
