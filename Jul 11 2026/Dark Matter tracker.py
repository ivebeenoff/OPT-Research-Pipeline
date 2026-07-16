# """
# ==============================================================================
#      SECTION 27 — DARK MATTER CLOSE ENCOUNTERS & COLLISION TRACKING
# ==============================================================================
#
# Author  : Abhinav Vatsa
#
# DESCRIPTION:
# Specialized companion extension module to the MW-M31 structural analysis pipeline.
# Inherits all parent parameters, directory configurations, constants, and global
# variables ($SNAPSHOTS, ns, R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN$).
#
# N-BODY DYNAMICAL PHENOMENOLOGY:
# Physical dark matter envelopes are collisionless fields operating under continuous Vlasov
# environments. However, discrete macro-particles ($m \sim 10^5\ M_\odot$) in N-body engines
# undergo localized phase-space close approaches. Quantifying these local events probes:
#   (A) Two-Body Gravitational Scattering: Highlighting momentum exchanges and 
#       deflections tracing the local dynamical friction field.
#   (B) Spatial Concentration Micro-Mapping: Unveiling real core density maxima 
#       and localized subhalo centering without shell grid binning constraints.
#   (C) Thermal Relative Velocity Horizons: P(v_rel) distributions isolate bound 
#       clumps from high-velocity pericentric stream interpenetrations.
#
# OPTIMIZATION METHODOLOGY:
# Pairwise tracking scale scales as $O(N^2)$, which is heavily prohibitive. This engine
# deploys an optimized k-dimensional spatial tree ($cKDTree$) to map bounded search spheres,
# extracting sample metrics across a controlled subset of $N_{\rm probe}$ tracer indices.
#
# ==============================================================================
# """

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.0 — DIAGNOSTIC CONFIGURATION                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

D_ENC_KPC        = 1.0                          # Capture aperture border (twice the softening scale)
N_PROBE          = 5000                         # Spatial sampler index size
V_COLD_KMS       = 50.0                         # Focus transition threshold
ENC_MAP_BINS     = 200                          # Spatial matrix sizing grid
ENC_MAP_EXTENT   = 200.0                        # Domain focus coordinate (±kpc)
ENC_STEP         = 5                            # Snapshot sub-sampling stride
ANIM_FPS_27      = 18
ANIM_DPI_27      = 100
ANIM_BITRATE_27  = 1600

print("\n" + "="*80)
print("  SECTION 27 · Vectorized Close Encounter Spatial Ingestor Active")
print("="*80)
print(f"  Encounter Search Horizon : {D_ENC_KPC} kpc")
print(f"  Tracer Sample Size       : {N_PROBE} indices / snapshot")
print(f"  Temporal Stride Size    : {ENC_STEP} snapshots")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.1 — COMPILING VECTORIZED DYNAMICAL MATRIX CONVERTERS                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def find_close_encounters(pos: np.ndarray, vel: np.ndarray, m: np.ndarray, d_enc: float, n_probe: int, rng: np.random.Generator) -> dict:
    """
    Deploys k-dimensional space tree lookups to calculate interaction parameters
    across a representative tracer subset.
    """
    N = len(pos)
    if N < 10 or n_probe < 1:
        return {k: np.array([]) for k in ["d_arr", "vrel_arr", "theta_arr", "dE_arr", "midpoint_arr", "r_mid_enc", "probe_r", "enc_per_probe"]}

    n_probe = min(n_probe, N)
    probe_idx = rng.choice(N, size=n_probe, replace=False)

    tree = cKDTree(pos)
    
    d_list = []
    vrel_list = []
    theta_list = []
    dE_list = []
    midpoint_list = []
    enc_per_probe = np.zeros(n_probe, dtype=int)

    # Pre-caching vector metrics blocks optimization checks
    for pi, probe_i in enumerate(probe_idx):
        neighbours = tree.query_ball_point(pos[probe_i], d_enc)
        neighbours = [j for j in neighbours if j != probe_i]
        enc_per_probe[pi] = len(neighbours)

        if not neighbours:
            continue

        # ── Vectorized Local Neighborhood Processing ──
        # Replaces internal iterative neighbor parsing loops with fast array arithmetic
        nb_pos = pos[neighbours]
        nb_vel = vel[neighbours]
        nb_mass = m[neighbours]

        d_ij = np.linalg.norm(pos[probe_i] - nb_pos, axis=1)
        valid_pairs = d_ij > 1e-10
        
        if not np.any(valid_pairs):
            continue

        d_ij = d_ij[valid_pairs]
        nb_vel = nb_vel[valid_pairs]
        nb_mass = nb_mass[valid_pairs]
        nb_pos = nb_pos[valid_pairs]

        v_ij = vel[probe_i] - nb_vel
        vrel = np.linalg.norm(v_ij, axis=1)

        # Gravitational Deflection Rutherford Scattering Analogue: tan(θ/2) = G(m1+m2)/(b * v_rel^2)
        m_pair = m[probe_i] + nb_mass
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = d_ij * vrel**2
            theta = np.where((vrel > 0) & (d_ij > 0), 2.0 * np.arctan(G_KPC_KMS2_MSUN * m_pair / np.maximum(denom, 1e-30)), 0.0)
            dE = np.where((vrel > 0) & (d_ij > 0), (G_KPC_KMS2_MSUN * nb_mass / d_ij)**2 / vrel**2, 0.0)

        mid = 0.5 * (pos[probe_i] + nb_pos)

        d_list.extend(d_ij.tolist())
        vrel_list.extend(vrel.tolist())
        theta_list.extend(theta.tolist())
        dE_list.extend(dE.tolist())
        midpoint_list.extend(mid.tolist())

    if not d_list:
        return {
            "n_enc": 0, "d_arr": np.array([]), "vrel_arr": np.array([]), "theta_arr": np.array([]),
            "dE_arr": np.array([]), "midpoint_arr": np.empty((0, 3)), "r_mid_enc": np.array([]),
            "probe_r": np.linalg.norm(pos[probe_idx], axis=1), "enc_per_probe": enc_per_probe
        }

    midpoints = np.array(midpoint_list)
    return {
        "n_enc": len(d_list), "d_arr": np.array(d_list), "vrel_arr": np.array(vrel_list),
        "theta_arr": np.array(theta_list), "dE_arr": np.array(dE_list), "midpoint_arr": midpoints,
        "r_mid_enc": np.linalg.norm(midpoints, axis=1), "probe_r": np.linalg.norm(pos[probe_idx], axis=1),
        "enc_per_probe": enc_per_probe
    }


def two_body_relaxation_time(rho: np.ndarray, sigma: np.ndarray, m_part: float, ln_lam: float = 10.0) -> np.ndarray:
    """
    Computes hydrostatic structural relaxation bounds (Binney & Tremaine 2008):
      t_relax = (0.34 * σ³) / (G² * m * ρ * ln Λ)
    """
    KPC_KMS_TO_GYR = 0.9778
    with np.errstate(divide="ignore", invalid="ignore"):
        num = 0.34 * sigma**3
        den = G_KPC_KMS2_MSUN**2 * m_part * rho * ln_lam
        t_relax = np.where((rho > 0) & (sigma > 0) & (den > 0), (num / den) * KPC_KMS_TO_GYR, np.nan)
    return t_relax


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.2 — TIME SERIES MATRIX SPACE ALLOCATION                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

enc_snap_nums   = SNAPSHOTS[::ENC_STEP]
n_enc_snaps     = len(enc_snap_nums)
enc_snap_map    = {s: idx for idx, s in enumerate(enc_snap_nums)}
time_enc        = np.full(n_enc_snaps, np.nan)

# Pre-allocating telemetry tensors
n_enc_arr       = np.full(n_enc_snaps, np.nan)
mean_vrel_arr   = np.full(n_enc_snaps, np.nan)
mean_theta_arr  = np.full(n_enc_snaps, np.nan)
mean_dE_arr     = np.full(n_enc_snaps, np.nan)
cold_frac_arr   = np.full(n_enc_snaps, np.nan)

gamma_radial_ts = np.full((n_enc_snaps, nb_sph), np.nan)
t_relax_ts      = np.full((n_enc_snaps, nb_sph), np.nan)
enc_maps        = np.zeros((n_enc_snaps, ENC_MAP_BINS, ENC_MAP_BINS))

profile_enc_ii  = [int(f * (n_enc_snaps - 1)) for f in [0.0, 0.2, 0.4, 0.65, 1.0]]
vrel_dists      = [np.array([]) for _ in range(5)]

rng27 = np.random.default_rng(seed=27)

print(f"\n[Pre-alloc] Encounter Topography Space Allocation:")
print(f"  gamma_radial_ts : {gamma_radial_ts.shape}")
print(f"  t_relax_ts      : {t_relax_ts.shape}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.3 — TEMPORAL DYNAMICAL PROCESSING                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  §27.3 — Main Encounter Ingestion Stride Loop")
print("="*80)

t_loop_start = time.perf_counter()
shell_vols_loc = (4.0 / 3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)

for i, snap_num in enumerate(enc_snap_nums):
    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue

    try:
        snap_data = load_snapshot_particles(mw_file, m31_file)
        MW_obj    = CenterOfMass(mw_file,  PTYPE)
        M31_obj   = CenterOfMass(m31_file, PTYPE)
    except Exception as exc:
        print(f"  [ERROR] Framework tracking broken at snap {snap_num}: {exc}")
        continue

    pos   = snap_data["pos"]
    m     = snap_data["m_msun"]
    r_mag = np.linalg.norm(pos, axis=1)

    # ── Velocity Frame Translation ────────────────────────────────────────────
    vx_all = np.concatenate((MW_obj.vx, M31_obj.vx))
    vy_all = np.concatenate((MW_obj.vy, M31_obj.vy))
    vz_all = np.concatenate((MW_obj.vz, M31_obj.vz))
    m_raw  = np.concatenate((MW_obj.m,  M31_obj.m))
    
    xcom, ycom, zcom = MW_obj.COMdefine(np.concatenate((MW_obj.x, M31_obj.x)),
                                       np.concatenate((MW_obj.y, M31_obj.y)),
                                       np.concatenate((MW_obj.z, M31_obj.z)), m_raw)
    
    dr_c = np.sqrt((np.concatenate((MW_obj.x, M31_obj.x)) - xcom)**2 + 
                   (np.concatenate((MW_obj.y, M31_obj.y)) - ycom)**2 + 
                   (np.concatenate((MW_obj.z, M31_obj.z)) - zcom)**2)
    inn = dr_c < 15.0
    
    if inn.sum() >= 5:
        wi = m[inn]
        vxcom = np.sum(wi * vx_all[inn]) / wi.sum()
        vycom = np.sum(wi * vy_all[inn]) / wi.sum()
        vzcom = np.sum(wi * vz_all[inn]) / wi.sum()
    else:
        vxcom = vycom = vzcom = 0.0

    vel = np.vstack((vx_all - vxcom, vy_all - vycom, vz_all - vzcom)).T

    # Map tracking time dimensions
    snap_idx_match = np.where(SNAPSHOTS == snap_num)[0]
    time_enc[i] = time_arr[snap_idx_match[0]] if len(snap_idx_match) > 0 else float(snap_num)
    m_typical = float(np.median(m))

    # ── Execute Spatial cKDTree Lookup ──
    enc = find_close_encounters(pos, vel, m, D_ENC_KPC, N_PROBE, rng27)
    n_enc_arr[i] = enc["n_enc"]

    if enc["n_enc"] > 0:
        mean_vrel_arr [i] = float(np.mean(enc["vrel_arr"]))
        mean_theta_arr[i] = float(np.mean(enc["theta_arr"]))
        mean_dE_arr   [i] = float(np.mean(enc["dE_arr"]))
        cold_frac_arr [i] = float(np.mean(enc["vrel_arr"] < V_COLD_KMS))

        # ── Vectorized Encounter Profile Extraction ──
        # Replaces old iterative counting loops with parallel array histograms
        counts_enc_shells, _ = np.histogram(enc["r_mid_enc"], bins=R_BINS)
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma_radial_ts[i, :] = np.where(shell_vols_loc > 0, (counts_enc_shells / shell_vols_loc) * (len(pos) / N_PROBE), np.nan)

        # ── Populating Cartesian Projections Maps ──
        if len(enc["midpoint_arr"]) > 0:
            H, _, _ = np.histogram2d(enc["midpoint_arr"][:, 0], enc["midpoint_arr"][:, 1], bins=ENC_MAP_BINS,
                                     range=[[-ENC_MAP_EXTENT, ENC_MAP_EXTENT], [-ENC_MAP_EXTENT, ENC_MAP_EXTENT]])
            enc_maps[i] = H

        # Sync profile epochs limits
        for pi_idx, profile_ii in enumerate(profile_enc_ii):
            if i == profile_ii:
                vrel_dists[pi_idx] = enc["vrel_arr"].copy()

    # ── High-Performance Relaxation Reductions ──
    # Computes localized rho and sigma across bins without manual loop passes
    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = np.where(r_mag[:, None] > 0, pos / r_mag[:, None], 0.0)
    v_r_all = np.einsum("ij,ij->i", vel, r_hat)

    # Vectorized computation of mass totals and velocities variances per radial shell
    mass_sum_shells = binned_statistic(r_mag, m, statistic="sum", bins=R_BINS)[0]
    counts_shells = binned_statistic(r_mag, m, statistic="count", bins=R_BINS)[0]
    
    rho_loc = np.where(counts_shells >= MIN_PART_SHELL, mass_sum_shells / shell_vols_loc, np.nan)
    
    # Calculate binned dispersion profiles
    num_vr = binned_statistic(r_mag, m * v_r_all, statistic="sum", bins=R_BINS)[0]
    den_vr = binned_statistic(r_mag, m, statistic="sum", bins=R_BINS)[0]
    v_r_mean_shells = num_vr / den_vr
    
    # Map back mean markers to execute localized variances checks
    bin_id_all = np.clip(np.digitize(r_mag, R_BINS) - 1, 0, nb_sph - 1)
    v_r_diff_sq = (v_r_all - v_r_mean_shells[bin_id_all])**2
    num_var = binned_statistic(r_mag, m * v_r_diff_sq, statistic="sum", bins=R_BINS)[0]
    
    sigma_loc = np.where(counts_shells >= MIN_PART_SHELL, np.sqrt(num_var / den_vr), np.nan)
    t_relax_ts[i, :] = two_body_relaxation_time(rho_loc, sigma_loc, m_typical)

    if (i + 1) % 20 == 0:
        elapsed = time.perf_counter() - t_loop_start
        print(f"  enc-snap {snap_num:04d} | Active pairs = {int(n_enc_arr[i]) if np.isfinite(n_enc_arr[i]) else 0} | ⟨v_rel⟩ = {mean_vrel_arr[i]:.0f} km/s | [{elapsed:.0f}s elapsed]")

print(f"\n[Loop complete] Diagnostic tensors populated in {time.perf_counter() - t_loop_start:.0f}s total.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.4 — GRAPHICAL EXTRACTION RUNS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BG, MUTED = "#0d0d18", "#7070a0"

def _ax(ax, xlabel="", ylabel="", title="", log_x=False, log_y=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a4a")
    ax.tick_params(colors="#9090b0", labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#c8c8e8")
    ax.set_ylabel(ylabel, fontsize=9, color="#c8c8e8")
    ax.set_title(title, fontsize=10, color="#c8c8e8", pad=5)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    return ax

t_enc_min, t_enc_max = np.nanmin(time_enc), np.nanmax(time_enc)
enc_profile_labels = [f"Snap {enc_snap_nums[idx]}" for idx in profile_enc_ii]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — TELEMETRY ENCOUNTER RATES VS TIME
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Plotting continuous collision frequency telemetry channels...")
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})
valid = np.isfinite(n_enc_arr)

_ax(ax1a, ylabel=r"Encounter Intensity $\Gamma(t)$", title=fr"DM Structural Convergence Flux Rates (Search Radius boundary: $d < {D_ENC_KPC}$ kpc)", log_y=True)
ax1a.plot(time_enc[valid], n_enc_arr[valid], color="#ff9944", lw=1.8)
ax1a.fill_between(time_enc[valid], 1, n_enc_arr[valid], alpha=0.12, color="#ff9944")

_ax(ax1b, xlabel=time_label, ylabel="Focused Cold Fraction", title=fr"Low-Velocity Dispersion Subhalo Pairs ($v_{{\rm rel}} < {V_COLD_KMS:.0f}$ km s$^{{-1}}$)")
valid_c = valid & np.isfinite(cold_frac_arr)
ax1b.plot(time_enc[valid_c], cold_frac_arr[valid_c], color="#4a8fff", lw=1.8)
ax1b.fill_between(time_enc[valid_c], 0, cold_frac_arr[valid_c], alpha=0.12, color="#4a8fff")
ax1b.set_ylim(0, 1.05)

fig1.savefig(os.path.join(OUT_DIR, "section27_encounter_rate.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — MAP GRID CORE TRACKING AT EPOCAL INTERSECTS
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Rendering multi-panel collision map projections...")
fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG, sharey=True, gridspec_kw={"wspace": 0.04})

for col, (ii, label) in enumerate(zip(profile_enc_ii, enc_profile_labels)):
    ax = axes2[col]
    ax.set_facecolor(BG)
    H = enc_maps[ii]
    Hs = gaussian_filter(np.where(H > 0, H, 0.0), sigma=2.0)
    H_log = np.where(Hs > 0, np.log10(Hs + 1), np.nan)

    vals = H_log[np.isfinite(H_log)]
    vmin = np.percentile(vals, 5) if vals.size > 0 else 0
    vmax = np.percentile(vals, 99) if vals.size > 0 else 5

    ax.imshow(H_log.T, origin="lower", aspect="equal", extent=[-ENC_MAP_EXTENT, ENC_MAP_EXTENT, -ENC_MAP_EXTENT, ENC_MAP_EXTENT], cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title(label, fontsize=9, color="#c8c8e8")
    ax.tick_params(colors="#9090b0", labelsize=7)
    ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")
    if col == 0: ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")

fig2.suptitle(r"Encounter Concentration Landscapes $\log_{10}(\Gamma_{\rm enc}(x,y) + 1)$", fontsize=11)
fig2.savefig(os.path.join(OUT_DIR, "section27_encounter_spatial.png"), dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — VELOCITY ORIENTATION FREQUENCY SPECTRUMS P(v_rel)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 3] Plotting distribution velocity histories...")
fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax3, xlabel=r"$v_{\rm rel}$ [km s$^{-1}$]", ylabel="Probability Density", title=r"Relative Coordinate Velocity Spectrums Across Stages $P(v_{\rm rel})$")

for ii_idx, (vrel_arr, label, color) in enumerate(zip(vrel_dists, enc_profile_labels, PROFILE_COLORS)):
    if len(vrel_arr) < 5: continue
    ax3.hist(vrel_arr, bins=50, density=True, alpha=0.55, color=color, edgecolor="none", label=label)

ax3.axvline(V_COLD_KMS, color="#ffffff", lw=0.9, ls="--", alpha=0.5, label=fr"Adiabatic Transition Barrier ({V_COLD_KMS:.0f} km s$^{{-1}}$)")
ax3.set_xlim(0, None)
ax3.legend(fontsize=7)
fig3.savefig(os.path.join(OUT_DIR, "section27_vrel_dist.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig3)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — RADIALLY ALIGNED HEATMAP DURATION PROFILE Γ(r, t)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 4] Plotting logarithmic radial collision frequency profiles...")
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06})

y_uniform_enc = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), 200)
gamma_interp_map = np.zeros((len(y_uniform_enc), n_enc_snaps))
for snap_idx in range(n_enc_snaps):
    nm = np.isfinite(gamma_radial_ts[snap_idx, :])
    gamma_interp_map[:, snap_idx] = np.interp(np.log10(y_uniform_enc), np.log10(r_mid_sph[nm]), gamma_radial_ts[snap_idx, nm]) if nm.sum() > 2 else np.nan

gamma_for_plot = np.where(gamma_interp_map > 0, np.log10(gamma_interp_map), np.nan)

im4 = ax4a.imshow(gamma_for_plot, aspect="auto", origin="lower", extent=[t_enc_min, t_enc_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="inferno")
ax4a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax4a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
_ax(ax4a, xlabel=time_label, ylabel="r [kpc]", title=r"Spherically Resolved Local Interaction Flux Profile $\log_{10}\Gamma(r,t)\ [{\rm enc\ kpc^{-3}}]$")

cb4 = fig4.colorbar(im4, ax=ax4a, pad=0.01)
cb4.set_label(r"$\log_{10}\Gamma$", fontsize=8)

gamma_mean = np.nanmean(gamma_radial_ts, axis=0)
valid_gm = np.isfinite(gamma_mean) & (gamma_mean > 0)
_ax(ax4b, xlabel=r"$\langle\Gamma\rangle_t$", title="Time Avg", log_x=True)
ax4b.plot(gamma_mean[valid_gm], r_mid_sph[valid_gm], color="#ff9944", lw=2.0)
ax4b.set_yscale("log")
ax4b.set_ylim(R_BINS[0], R_BINS[-1])
ax4b.tick_params(labelleft=False)
fig4.savefig(os.path.join(OUT_DIR, "section27_encounter_radius.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — SCATTER DEFLECTION DEVIATION ANCHORS
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 5] Plotting perturbative momentum transfer vectors...")
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})

valid_dE = valid & np.isfinite(mean_dE_arr)
_ax(ax5a, ylabel=r"$\langle|\Delta E|\rangle\ [{\rm km^2\ s^{-2}}]$", title="Specific Energy Perturbation Vector Magnitudes", log_y=True)
ax5a.plot(time_enc[valid_dE], mean_dE_arr[valid_dE], color="#e8673a", lw=1.8)

valid_th = valid & np.isfinite(mean_theta_arr)
_ax(ax5b, xlabel=time_label, ylabel=r"$\langle\theta\rangle\ [{\rm deg}]$", title="Mean Cumulative Deflection Bounds")
ax5b.plot(time_enc[valid_th], np.degrees(mean_theta_arr[valid_th]), color="#aa55ff", lw=1.8)
fig5.savefig(os.path.join(OUT_DIR, "section27_energy_transfer.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig5)

fig5b, ax5b2 = plt.subplots(figsize=(10, 4), facecolor=BG)
_ax(ax5b2, xlabel=time_label, ylabel=r"$\langle\theta\rangle\ [{\rm deg}]$", title="Mean Scattering Deflection Angles")
ax5b2.plot(time_enc[valid_th], np.degrees(mean_theta_arr[valid_th]), color="#aa55ff", lw=1.8)
fig5b.savefig(os.path.join(OUT_DIR, "section27_deflection_angle.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig5b)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — TWO-BODY TIMESTEP DISPERSIONS t_relax(r)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 6] Plotting relaxation duration profiles...")
fig6, ax6 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax6, xlabel="r [kpc]", ylabel=r"$t_{\rm relax}\ [{\rm Gyr}]$", title=r"Two-Body Diffusion Relaxation Lifetimes: $t_{\rm relax}(r) = \frac{0.34\sigma^3}{G^2 m \rho \ln\Lambda}$", log_x=True, log_y=True)

for ii_idx, (ii, label, color) in enumerate(zip(profile_enc_ii, enc_profile_labels, PROFILE_COLORS)):
    y = t_relax_ts[ii, :]
    valid_r = np.isfinite(y) & (y > 0)
    if valid_r.any(): ax6.plot(r_mid_sph[valid_r], y[valid_r], color=color, lw=2.0, label=label)

ax6.axhline(13.8, color="#ffcc44", lw=0.9, ls="--", alpha=0.6, label="Hubble Boundary Horizon (13.8 Gyr)")
ax6.axhline(1.0,  color="#ffffff", lw=0.7, ls=":",  alpha=0.4, label="Unit Scale Guard (1 Gyr)")
ax6.set_xlim(R_BINS[0], R_BINS[-1])
ax6.legend(fontsize=7)
fig6.savefig(os.path.join(OUT_DIR, "section27_two_body_relaxation.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig6)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.5 — ANIMATION AND CONTOURS ENGINE                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

print("\n[Anim] Packaging multi-dimensional projection video streams...")
fig_a, (ax_map, ax_hist, ax_pv) = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG, gridspec_kw={"width_ratios": [3, 2, 2], "wspace": 0.12})

for ax in (ax_map, ax_hist, ax_pv): ax.set_facecolor(BG)
ax_map.set_xlim(-ENC_MAP_EXTENT, ENC_MAP_EXTENT); ax_map.set_ylim(-ENC_MAP_EXTENT, ENC_MAP_EXTENT)
ax_map.set_xlabel("x [kpc]", color="#c8c8e8"); ax_map.set_ylabel("y [kpc]", color="#c8c8e8")

ax_hist.set_xlim(t_enc_min, t_enc_max)
n_max = np.nanmax(n_enc_arr) * 1.15 if np.isfinite(np.nanmax(n_enc_arr)) else 1.0
ax_hist.set_ylim(1, n_max); ax_hist.set_yscale("log")
ax_hist.set_xlabel(time_label, color="#c8c8e8"); ax_hist.set_ylabel(r"$\Gamma(t)$", color="#c8c8e8")
ax_hist.set_title("Integrated Flow Rates", color="#c8c8e8", fontsize=9)

ax_pv.set_xlim(0, 500); ax_pv.set_xlabel(r"$v_{\rm rel}\ [{\rm km/s}]$", color="#c8c8e8")
ax_pv.set_title(r"Marginal Phase distribution $P(v_{\rm rel})$", color="#c8c8e8", fontsize=9)

all_H = enc_maps[enc_maps > 0]
vmin_a = np.log10(np.percentile(all_H, 10) + 1) if all_H.size > 0 else 0
vmax_a = np.log10(np.percentile(all_H, 99) + 1) if all_H.size > 0 else 5

H0 = enc_maps[0]
Hs0 = gaussian_filter(np.where(H0 > 0, H0, 0.0), sigma=2.0)
im_map = ax_map.imshow(np.where(Hs0 > 0, np.log10(Hs0 + 1), np.nan).T, origin="lower", aspect="equal",
                       extent=[-ENC_MAP_EXTENT, ENC_MAP_EXTENT, -ENC_MAP_EXTENT, ENC_MAP_EXTENT], cmap="hot", vmin=vmin_a, vmax=vmax_a)
cnt_line, = ax_hist.plot([], [], color="#ff9944", lw=1.8)

pv_bins = np.linspace(0, 500, 41)
pv_cents = 0.5 * (pv_bins[:-1] + pv_bins[1:])
pv_bars = ax_pv.bar(pv_cents, np.zeros(40), width=np.diff(pv_bins), color="#4a8fff", alpha=0.7, edgecolor="none")
title_a = fig_a.suptitle("", fontsize=11, color="#c8c8e8")

def _update_enc_anim(frame_idx):
    H = enc_maps[frame_idx]
    Hs = gaussian_filter(np.where(H > 0, H, 0.0), sigma=2.0)
    im_map.set_data(np.where(Hs > 0, np.log10(Hs + 1), np.nan).T)

    valid_f = np.isfinite(time_enc[:frame_idx+1]) & np.isfinite(n_enc_arr[:frame_idx+1])
    cnt_line.set_data(time_enc[:frame_idx+1][valid_f], n_enc_arr[:frame_idx+1][valid_f])

    snap_num = enc_snap_nums[frame_idx]
    if frame_idx < len(vrel_dists) and len(vrel_dists[frame_idx]) > 5:
        counts, _ = np.histogram(vrel_dists[frame_idx], bins=pv_bins, density=True)
    else:
        counts = np.zeros(40)
    for bar, h in zip(pv_bars, counts): bar.set_height(h)
    ax_pv.set_ylim(0, max(counts.max() * 1.15, 1e-5))

    t_val = time_enc[frame_idx]
    t_str = f"{t_val:.2f} Gyr" if (np.isfinite(t_val) and time_is_gyr) else f"Snap {snap_num}"
    n_str = f"N_pairs={int(n_enc_arr[frame_idx])}" if np.isfinite(n_enc_arr[frame_idx]) else ""
    title_a.set_text(fr"Phase approach landscape  ·  {t_str}  ·  {n_str}")
    return [im_map, cnt_line] + list(pv_bars)

ani_enc = animation.FuncAnimation(fig_a, _update_enc_anim, frames=n_enc_snaps, interval=1000 // ANIM_FPS_27, blit=True)
writer_27 = animation.FFMpegWriter(fps=ANIM_FPS_27, bitrate=ANIM_BITRATE_27, metadata=dict(title="MW-M31 Phase Interaction Movie"))
ani_enc.save(os.path.join(OUT_DIR, "section27_animation_encounters.mp4"), writer=writer_27, dpi=ANIM_DPI_27)
plt.close(fig_a)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.6 — SYSTEM COMPACT MASTER PANEL                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

print("\n[Summary] Binding section variables into cumulative posters blocks...")
fig_s = plt.figure(figsize=(16, 10), facecolor=BG)
gs_s  = gridspec.GridSpec(2, 2, figure=fig_s, hspace=0.38, wspace=0.32, left=0.08, right=0.97, top=0.93, bottom=0.07)

# (0,0) Cumulative rate limits
ax_s00 = fig_s.add_subplot(gs_s[0, 0])
_ax(ax_s00, xlabel=time_label, ylabel=r"$\Gamma(t)$", title="Total Encounter Rate Trajectory", log_y=True)
if valid.any(): ax_s00.plot(time_enc[valid], n_enc_arr[valid], color="#ff9944", lw=1.8)

# (0,1) Spherically binned heatmap
ax_s01 = fig_s.add_subplot(gs_s[0, 1])
im_s01 = ax_s01.imshow(gamma_for_plot, aspect="auto", origin="lower", extent=[t_enc_min, t_enc_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="inferno")
ax_s01.set_yscale("log"); ax_s01.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0]))); ax_s01.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
_ax(ax_s01, xlabel=time_label, ylabel="r [kpc]", title=r"Radial Density Concentration Flux $\log_{10}\Gamma(r,t)$")
fig_s.colorbar(im_s01, ax=ax_s01, shrink=0.8)

# (1,0) Perturbative transfers energy profiles
ax_s10 = fig_s.add_subplot(gs_s[1, 0])
_ax(ax_s10, xlabel=time_label, ylabel=r"$\langle|\Delta E|\rangle\ [{\rm km^2\ s^{-2}}]$", title="Specific Perturbative Energy Deflections", log_y=True)
if valid_dE.any(): ax_s10.plot(time_enc[valid_dE], mean_dE_arr[valid_dE], color="#e8673a", lw=1.8)

# (1,1) Diffusion timescale profiles
ax_s11 = fig_s.add_subplot(gs_s[1, 1])
_ax(ax_s11, xlabel="r [kpc]", ylabel=r"$t_{\rm relax}\ [{\rm Gyr}]$", title="Two-Body Diffusion Relaxation Profiles", log_x=True, log_y=True)
for ii_idx, (ii, color) in enumerate(zip(profile_enc_ii, PROFILE_COLORS)):
    y = t_relax_ts[ii, :]
    if np.isfinite(y).any(): ax_s11.plot(r_mid_sph[np.isfinite(y)], y[np.isfinite(y)], color=color, lw=1.5)
ax_s11.axhline(13.8, color="#ffcc44", lw=0.8, ls="--", alpha=0.5)

fig_s.suptitle("Section 27 Summary  ·  Encounter Flux Dynamics & Potential Remapping", fontsize=13, color="#c8c8e8", fontweight="bold")
fig_s.savefig(os.path.join(OUT_DIR, "section27_summary_panel.png"), dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig_s)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §27.7 — COMPLETE MANIFEST INVENTORY CLOSURE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

print("\n" + "="*80)
print("  SECTION 27 COMPLETE OUTPUT REVIEWS")
print("="*80)
outputs_27 = [
    "section27_encounter_rate.png", "section27_encounter_spatial.png", "section27_vrel_dist.png",
    "section27_encounter_radius.png", "section27_energy_transfer.png", "section27_deflection_angle.png",
    "section27_two_body_relaxation.png", "section27_animation_encounters.mp4", "section27_summary_panel.png"
]
for fn in outputs_27:
    fp = os.path.join(OUT_DIR, fn)
    size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
    kind = "Stream Media (mp4)" if fn.endswith(".mp4") else "Static Panel (png)"
    print(f"  {fn:<50} {size:10.2f} MB  [{kind}]")
print("="*80)
