# """
# ==============================================================================
#                 SECTION 22 — TIDAL FIELD & STRIPPING DIAGNOSTICS
# ==============================================================================
#
# Author  : Abhinav Vatsa
#
# DESCRIPTION:
# Extension module tracking gravitational tidal fields, unbinding interactions,
# and stripping signatures. Inherits all global structural states and alignment
# parameters from density_pipeline.py and section21_angular_momentum.py.
#
# DYNAMICAL REGIMES TRACKED:
#   1. Jacobi Radius Erosion: Mapping the compression of Roche limits r_t(t)
#      as satellite structures interpenetrate host fields.
#   2. Specific Phase-Space Binding: Separating bound and unbound particle counts
#      using local mechanical energy parameters: E = E_kin + E_pot < 0.
#   3. Impulsive Tidal Heating: Measuring local kinetic energy injections
#      inducing shell inflation and structural relaxation over time.
#   4. Quadrupole Morphological Distortions: Diagonalizing reduced inertia 
#      tensors to map structural prolate stretching.
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
from scipy.linalg import eigh
import warnings
import os
import time

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §22.0 — CONFIGURATION & REGIME PARAMETERS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Tidal Bound Parameters ────────────────────────────────────────────────────
TIDAL_RADIUS_FACTOR = 3.0                       # Jacobi energetic coefficient
SOFTENING_KPC = 0.5                             # Plummer gravitational softening length

# ── Stride Decimations ────────────────────────────────────────────────────────
BOUND_STEP = 4                                  # Snapshot interval for unbinding tracking
HEATING_STEP = 2                                # Snapshot interval for energy differencing
TENSOR_STEP = 8                                 # Snapshot interval for tidal tensor evaluation
INERTIA_STEP = 8                                # Snapshot interval for axis ratios mapping

# ── Aperture & Grid Geometry ──────────────────────────────────────────────────
INERTIA_R_INNER  = 30.0                         # Inner core aperture border
INERTIA_R_MID    = 150.0                        # Extended envelope border
STREAM_MAP_BINS   = 200                         # Spatial resolution of unbound maps
STREAM_MAP_EXTENT = 600.0                       # Half-width stream box boundary (kpc)
STREAM_ANIM_STEP  = 8
BOUND_ANIM_STEP   = 4

ANIM_FPS_22     = 20
ANIM_DPI_22     = 100
ANIM_BITRATE_22 = 2000

print("\n" + "="*80)
print("  SECTION 22 · Vectorized Tidal & Stripping Engine Activated")
print("="*80)
print(f"  Softening Bounds  : {SOFTENING_KPC} kpc")
print(f"  Ingestion Strides : Bound={BOUND_STEP} | Tensor={TENSOR_STEP} | Inertia={INERTIA_STEP}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §22.1 — MATHEMATICAL ENGINE UTILITIES                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def enclosed_mass_profile(r_mag: np.ndarray, m_msun: np.ndarray, r_bins: np.ndarray) -> np.ndarray:
    """
    Computes true cumulative enclosed mass M(<r) using fast index lookups.
    """
    sort_idx = np.argsort(r_mag)
    r_sorted = r_mag[sort_idx]
    m_cumsum = np.cumsum(m_msun[sort_idx])
    
    r_edges = r_bins[1:]
    search_pos = np.searchsorted(r_sorted, r_edges, side="right")
    
    M_enc = np.where(search_pos > 0, m_cumsum[np.clip(search_pos - 1, 0, len(m_cumsum) - 1)], 0.0)
    return M_enc


def specific_potential(r_mag: np.ndarray, m_msun: np.ndarray, r_bins: np.ndarray, softening: float = SOFTENING_KPC) -> np.ndarray:
    """
    Evaluates particle specific gravitational potentials via continuous enclosed-mass mapping.
    """
    M_enc = enclosed_mass_profile(r_mag, m_msun, r_bins)
    M_enc_interp = np.interp(r_mag, r_bins[1:], M_enc, left=0.0, right=M_enc[-1])
    return -G_KPC_KMS2_MSUN * M_enc_interp / np.sqrt(r_mag**2 + softening**2)


def jacobi_tidal_radius(M_sat: float, M_host_enc: float, separation: float) -> float:
    """
    Estimates the instant Roche tidal limit in the co-moving reference frame.
    """
    if M_host_enc <= 0 or separation <= 0 or M_sat <= 0:
        return np.nan
    return separation * (M_sat / (TIDAL_RADIUS_FACTOR * M_host_enc))**(1.0 / 3.0)


def inertia_tensor(pos: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Computes reduced mass moments of inertia using vectorized contractions.
    """
    M = m.sum()
    if M == 0:
        return np.zeros((3, 3))
    
    r2 = np.sum(pos**2, axis=1)
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            I[i, j] = np.sum(m * (r2 * delta - pos[:, i] * pos[:, j]))
    return I


def axis_ratios_from_inertia(I: np.ndarray) -> tuple[float, float, float]:
    """
    Diagonalizes moments of inertia to yield physical axis lengths a >= b >= c.
    """
    try:
        eigvals = np.linalg.eigvalsh(I)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    l1, l2, l3 = np.sort(eigvals)
    a2 = max(l2 + l3 - l1, 0.0)
    b2 = max(l1 + l3 - l2, 0.0)
    c2 = max(l1 + l2 - l3, 0.0)

    axes = sorted([np.sqrt(a2), np.sqrt(b2), np.sqrt(c2)], reverse=True)
    return float(axes[0]), float(axes[1]), float(axes[2])


def tidal_tensor_at_point(pos_field: np.ndarray, m_field: np.ndarray, eval_point: np.ndarray, r_bins: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Approximates local tidal tensor matrices T_ij via finite central differences.
    """
    T = np.zeros((3, 3))
    e = np.eye(3)

    def _phi_at(probe):
        r_shifted = np.linalg.norm(pos_field - probe, axis=1)
        return np.sum(-G_KPC_KMS2_MSUN * m_field / np.sqrt(r_shifted**2 + SOFTENING_KPC**2))

    phi0 = _phi_at(eval_point)

    for i in range(3):
        T[i, i] = (_phi_at(eval_point + delta * e[i]) - 2.0 * phi0 + _phi_at(eval_point - delta * e[i])) / delta**2
        for j in range(i + 1, 3):
            T[i, j] = (_phi_at(eval_point + delta * e[i] + delta * e[j]) - 
                       _phi_at(eval_point + delta * e[i] - delta * e[j]) - 
                       _phi_at(eval_point - delta * e[i] + delta * e[j]) + 
                       _phi_at(eval_point - delta * e[i] - delta * e[j])) / (4.0 * delta**2)
            T[j, i] = T[i, j]
    return T


def mass_weighted_shell_bin(values: np.ndarray, r_mag: np.ndarray, m: np.ndarray, r_bins: np.ndarray) -> np.ndarray:
    nb = len(r_bins) - 1
    prof = np.full(nb, np.nan)
    bin_id = np.digitize(r_mag, r_bins) - 1
    for b in range(nb):
        mask = bin_id == b
        if mask.sum() < MIN_PART_SHELL:
            continue
        prof[b] = np.sum(m[mask] * values[mask]) / np.sum(m[mask])
    return prof


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §22.2 — SUB-SYSTEM ARRAY ALLOCATION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

bound_snap_nums = SNAPSHOTS[::BOUND_STEP]
n_bound         = len(bound_snap_nums)
bound_snap_map  = {s: idx for idx, s in enumerate(bound_snap_nums)}

bound_frac_arr = np.full(n_bound, np.nan)
M_bound_arr    = np.full(n_bound, np.nan)
M_unbound_arr  = np.full(n_bound, np.nan)
time_bound     = np.full(n_bound, np.nan)

rho_bound_ts   = np.full((n_bound, nb_sph), np.nan)
rho_unbound_ts = np.full((n_bound, nb_sph), np.nan)
f_unbound_ts   = np.full((n_bound, nb_sph), np.nan)

r_tidal_arr    = np.full(ns, np.nan)
separation_arr = np.full(ns, np.nan)
v_rel_arr      = np.full(ns, np.nan)

heat_snap_nums = SNAPSHOTS[::HEATING_STEP]
n_heat         = len(heat_snap_nums)
time_heat      = np.full(n_heat, np.nan)
Ekin_ts        = np.full((ns, nb_sph), np.nan)
dEkin_dt_ts    = np.full((ns, nb_sph), np.nan)

tensor_snap_nums = SNAPSHOTS[::TENSOR_STEP]
n_tensor         = len(tensor_snap_nums)
tensor_snap_map  = {s: idx for idx, s in enumerate(tensor_snap_nums)}
time_tensor      = np.full(n_tensor, np.nan)

T_eig1_arr  = np.full(n_tensor, np.nan)
T_eig2_arr  = np.full(n_tensor, np.nan)
T_eig3_arr  = np.full(n_tensor, np.nan)
T_trace_arr = np.full(n_tensor, np.nan)

inertia_snap_nums = SNAPSHOTS[::INERTIA_STEP]
n_inertia         = len(inertia_snap_nums)
inertia_snap_map  = {s: idx for idx, s in enumerate(inertia_snap_nums)}
time_inertia      = np.full(n_inertia, np.nan)

ba_arr       = np.full(n_inertia, np.nan)
ca_arr       = np.full(n_inertia, np.nan)
ba_inner_arr = np.full(n_inertia, np.nan)
ca_inner_arr = np.full(n_inertia, np.nan)
ba_mid_arr   = np.full(n_inertia, np.nan)
ca_mid_arr   = np.full(n_inertia, np.nan)
ba_outer_arr = np.full(n_inertia, np.nan)
ca_outer_arr = np.full(n_inertia, np.nan)

ba_profile_ts = np.full((n_inertia, nb_sph), np.nan)

stream_anim_snap_ids = np.arange(0, n_bound, max(1, STREAM_ANIM_STEP // BOUND_STEP))
n_stream_frames      = len(stream_anim_snap_ids)
stream_maps          = np.zeros((n_stream_frames, STREAM_MAP_BINS, STREAM_MAP_BINS))

bound_anim_snap_ids  = np.arange(0, n_bound, max(1, BOUND_ANIM_STEP // BOUND_STEP))
n_bound_anim_frames  = len(bound_anim_snap_ids)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §22.3 — SYSTEM DISPERSION RUN LOOP                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  §22.3 — Structural Processing Loop")
print("="*80)

t_loop_start = time.perf_counter()
com_mw_arr   = np.full((ns, 3), np.nan)
com_m31_arr  = np.full((ns, 3), np.nan)

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
        print(f"  [ERROR] File Mapping Failure at snap {snap_num}: {exc}")
        continue

    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    origin = snap_data["origin"]
    r_mag  = np.linalg.norm(pos, axis=1)

    # Reconstruct co-moving velocity profiles
    vx_all = np.concatenate((MW_obj.vx, M31_obj.vx))
    vy_all = np.concatenate((MW_obj.vy, M31_obj.vy))
    vz_all = np.concatenate((MW_obj.vz, M31_obj.vz))
    m_raw  = np.concatenate((MW_obj.m,  M31_obj.m))
    x_all  = np.concatenate((MW_obj.x,  M31_obj.x))
    y_all  = np.concatenate((MW_obj.y,  M31_obj.y))
    z_all  = np.concatenate((MW_obj.z,  M31_obj.z))

    xcom, ycom, zcom = MW_obj.COMdefine(x_all, y_all, z_all, m_raw)
    dr_com = np.sqrt((x_all-xcom)**2 + (y_all-ycom)**2 + (z_all-zcom)**2)
    inner  = dr_com < 15.0
    if inner.sum() >= 5:
        wi = m[inner]
        vxcom = np.sum(wi * vx_all[inner]) / wi.sum()
        vycom = np.sum(wi * vy_all[inner]) / wi.sum()
        vzcom = np.sum(wi * vz_all[inner]) / wi.sum()
    else:
        vxcom = vycom = vzcom = 0.0

    vel = np.vstack((vx_all - vxcom, vy_all - vycom, vz_all - vzcom)).T
    v_mag2 = np.sum(vel**2, axis=1)

    # Progenitor specific COM extractions
    mw_x_com, mw_y_com, mw_z_com = MW_obj.COMdefine(MW_obj.x, MW_obj.y, MW_obj.z, MW_obj.m)
    m31_x_com, m31_y_com, m31_z_com = M31_obj.COMdefine(M31_obj.x, M31_obj.y, M31_obj.z, M31_obj.m)

    com_mw_arr[i]  = [mw_x_com - xcom, mw_y_com - ycom, mw_z_com - zcom]
    com_m31_arr[i] = [m31_x_com - xcom, m31_y_com - ycom, m31_z_com - zcom]

    sep_vec = com_m31_arr[i] - com_mw_arr[i]
    sep_dist = np.linalg.norm(sep_vec)
    separation_arr[i] = sep_dist

    # Radial Jacobi Boundary extractions
    M_host_enc = m[origin == 0][r_mag[origin == 0] <= sep_dist].sum()
    M_sat      = m[origin == 1].sum()
    r_tidal_arr[i] = jacobi_tidal_radius(M_sat, M_host_enc, sep_dist)

    E_kin_spec = 0.5 * v_mag2
    Ekin_ts[i, :] = mass_weighted_shell_bin(E_kin_spec, r_mag, m, R_BINS)

    # ── Mechanical Energy Binding Criterion ──
    if snap_num in bound_snap_map:
        bi = bound_snap_map[snap_num]
        time_bound[bi] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)

        phi = specific_potential(r_mag, m, R_BINS)
        E_tot = 0.5 * v_mag2 + phi

        bound_mask = E_tot < 0.0
        unbound_mask = ~bound_mask

        M_tot = m.sum()
        M_bound_arr[bi]   = m[bound_mask].sum()
        M_unbound_arr[bi] = m[unbound_mask].sum()
        bound_frac_arr[bi] = M_bound_arr[bi] / (M_tot + 1e-30)

        shell_vols_local = (4.0 / 3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)
        bin_id = np.digitize(r_mag, R_BINS) - 1

        for b in range(nb_sph):
            mask_b = bin_id == b
            if mask_b.sum() >= MIN_PART_SHELL:
                M_bin_bound   = m[mask_b & bound_mask].sum()
                M_bin_unbound = m[mask_b & unbound_mask].sum()
                rho_bound_ts  [bi, b] = M_bin_bound / shell_vols_local[b]
                rho_unbound_ts[bi, b] = M_bin_unbound / shell_vols_local[b]
                f_unbound_ts  [bi, b] = M_bin_unbound / (m[mask_b].sum() + 1e-30)

        # Map streams distributions
        if bi in stream_anim_snap_ids:
            fi_s = np.where(stream_anim_snap_ids == bi)[0][0]
            stream_maps[fi_s], _, _ = np.histogram2d(pos[unbound_mask, 0], pos[unbound_mask, 1], bins=STREAM_MAP_BINS,
                                                     range=[[-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT], [-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT]],
                                                     weights=m[unbound_mask])

    # ── Tidal Tensor Matrix Operations ──
    if snap_num in tensor_snap_map:
        ti = tensor_snap_map[snap_num]
        time_tensor[ti] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)
        
        sub = np.arange(0, len(pos), 10)
        try:
            T = tidal_tensor_at_point(pos[sub], m[sub], eval_point=0.5 * (com_mw_arr[i] + com_m31_arr[i]), r_bins=R_BINS, delta=2.0)
            eigvals_T = np.linalg.eigvalsh(T)
            T_eig1_arr[ti]  = eigvals_T[2]
            T_eig2_arr[ti]  = eigvals_T[1]
            T_eig3_arr[ti]  = eigvals_T[0]
            T_trace_arr[ti] = np.trace(T)
        except Exception:
            pass

    # ── Reduced Inertia Shape Mapping ──
    if snap_num in inertia_snap_map:
        ii_idx = inertia_snap_map[snap_num]
        time_inertia[ii_idx] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)

        I_global = inertia_tensor(pos, m)
        a_g, b_g, c_g = axis_ratios_from_inertia(I_global)
        if a_g > 0:
            ba_arr[ii_idx] = b_g / a_g
            ca_arr[ii_idx] = c_g / a_g

        for mask_ap, ba_ap, ca_ap in [(r_mag < INERTIA_R_INNER, ba_inner_arr, ca_inner_arr),
                                      ((r_mag >= INERTIA_R_INNER) & (r_mag < INERTIA_R_MID), ba_mid_arr, ca_mid_arr),
                                      (r_mag >= INERTIA_R_MID, ba_outer_arr, ca_outer_arr)]:
            if mask_ap.sum() < 50:
                continue
            I_ap = inertia_tensor(pos[mask_ap], m[mask_ap])
            a_ap, b_ap, c_ap = axis_ratios_from_inertia(I_ap)
            if a_ap > 0:
                ba_ap[ii_idx] = b_ap / a_ap
                ca_ap[ii_idx] = c_ap / a_ap

        bin_id_ine = np.digitize(r_mag, R_BINS) - 1
        for b in range(nb_sph):
            mask_b = bin_id_ine == b
            if mask_b.sum() < 50:
                continue
            I_b = inertia_tensor(pos[mask_b], m[mask_b])
            a_b, b_b, _ = axis_ratios_from_inertia(I_b)
            if a_b > 0:
                ba_profile_ts[ii_idx, b] = b_b / a_b

    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t_loop_start
        print(f"  snap {snap_num:04d} | d = {separation_arr[i]:.1f} kpc | r_t = {r_tidal_arr[i]:.1f} kpc | [{elapsed:.0f}s elapsed]")

print(f"\n[Matrix profiles finished] Ingestion closed in {time.perf_counter() - t_loop_start:.0f}s total.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §22.4 — GRADIENT FLUX ESTIMATIONS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

sep_valid = np.isfinite(separation_arr)
if sep_valid.sum() > 2:
    v_rel_arr = np.gradient(separation_arr)

# Specific energy differences tracking: ΔE_kin(r,t)
dEkin_dt_ts = np.gradient(Ekin_ts, axis=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §22.5 — PLOT EXECUTIONS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

t_min, t_max = np.nanmin(time_arr), np.nanmax(time_arr)
BG, MUTED = "#0d0d18", "#7070a0"

def _styled_ax(ax, xlabel="", ylabel="", title="", log_x=False, log_y=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a4a")
    ax.tick_params(colors="#9090b0", labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#c8c8e8")
    ax.set_ylabel(ylabel, fontsize=9, color="#c8c8e8")
    ax.set_title(title,  fontsize=10, color="#c8c8e8", pad=5)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    return ax

# ── FIGURE 1 — SYSTEM CENTER SEPARATIONS & JACOBI RADII ──
print("\n[Fig 1] Plotting trajectory separation and Roche limit channels...")
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})

_styled_ax(ax1a, ylabel="Distance [kpc]", title="System Co-Moving Trajectory & Jacobi Roche Boundaries")
ax1a.plot(time_arr, separation_arr, color="#4a8fff", lw=2.0, label="Progenitor Separation d(t)")
ax1a.plot(time_arr, r_tidal_arr,    color="#ff9944", lw=2.0, label=r"Jacobi Boundary $r_t(t)$")
ax1a.axhline(15.0, color="#ffffff", lw=0.7, ls=":", alpha=0.4, label="Initial Core Apertures (~15 kpc)")
ax1a.set_yscale("log")
ax1a.legend(fontsize=8)

_styled_ax(ax1b, xlabel=time_label, ylabel=r"$r_t\ /\ r_{\rm half,\ 3D}$", title="Jacobi Radius Normalized to System Core Radii")
try:
    ax1b.plot(time_arr, r_tidal_arr / r_half_3d_arr, color="#e8673a", lw=1.8)
    ax1b.axhline(1.0, color="#ffffff", lw=0.8, ls="--", alpha=0.5, label=r"Erosion Limit ($r_t = r_{1/2}$)")
    ax1b.set_yscale("log")
    ax1b.legend(fontsize=8)
except NameError:
    ax1b.text(0.5, 0.5, "r_half_3d_arr context missing. Verify script synchronization parameters.", transform=ax1b.transAxes, ha="center", va="center", color=MUTED, fontsize=9)

fig1.savefig(os.path.join(OUT_DIR, "section22_tidal_radius.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig1)

# ── FIGURE 2 — TOTAL MECHANICAL BOUND MASS FRACTION ──
print("[Fig 2] Plotting total mechanical mass fractions...")
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})

_styled_ax(ax2a, ylabel="Bound Mass Fraction", title="Tidal Unbinding Chronology: Bound Mass Fractions")
ax2a.plot(time_bound, bound_frac_arr, color="#00d4aa", lw=1.8)
ax2a.fill_between(time_bound, np.where(np.isfinite(bound_frac_arr), bound_frac_arr, 0), alpha=0.12, color="#00d4aa")
ax2a.set_ylim(0, 1.05)
ax2a.axhline(0.5, color="#555577", lw=0.7, ls="--")

_styled_ax(ax2b, xlabel=time_label, ylabel=r"$M\ [M_\odot]$", title="Integrated Mechanical Bound vs Envelope Stream Budgets")
ax2b.semilogy(time_bound, M_bound_arr,   color="#4a8fff", lw=1.5, label="Mechanically Bound Core")
ax2b.semilogy(time_bound, M_unbound_arr, color="#ff5fa0", lw=1.5, label="Unbound Streams Debris")
ax2b.legend(fontsize=8)
fig2.savefig(os.path.join(OUT_DIR, "section22_bound_fraction.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig2)

# ── FIGURE 3 — DERIVED MASS DEPLATION RATES ──
print("[Fig 3] Plotting numerical stripping rates...")
dM_dt = -np.gradient(M_bound_arr)
valid_t = np.isfinite(time_bound) & np.isfinite(dM_dt)

fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor=BG)
_styled_ax(ax3, xlabel=time_label, ylabel=r"$-dM_{\rm bound}/dt\ [M_\odot\ /\ snap]$", title=r"Tidal Infall Stripping Intensity: $-dM_{\rm bound}/dt$")
ax3.plot(time_bound[valid_t], dM_dt[valid_t], color="#ff9944", lw=1.5)
ax3.fill_between(time_bound[valid_t], np.where(dM_dt[valid_t] > 0, dM_dt[valid_t], 0), alpha=0.18, color="#ff9944")

from scipy.signal import find_peaks
try:
    peaks, _ = find_peaks(dM_dt[valid_t], height=np.nanpercentile(dM_dt[valid_t], 80))
    for pk in peaks: ax3.axvline(time_bound[valid_t][pk], color="#ffffff", lw=0.7, ls="--", alpha=0.4)
except Exception: pass

fig3.savefig(os.path.join(OUT_DIR, "section22_stripping_rate.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig3)

# ── FIGURE 4 — STRUCTURAL DENSITY PROFILES BY MECHANICAL BINDING ──
print("[Fig 4] Plotting mechanical core component profiles...")
bound_times = time_bound[np.isfinite(time_bound)]
profile_bi  = [np.argmin(np.abs(time_bound - time_arr[k])) for k in PROFILE_INDICES]

fig4, axes4 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG, sharey=True, gridspec_kw={"wspace": 0.06})

for col, (bi, label, color) in enumerate(zip(profile_bi, PROFILE_LABELS, PROFILE_COLORS)):
    ax = axes4[col]
    _styled_ax(ax, xlabel="r [kpc]", title=label, log_x=True, log_y=True)
    ax.set_xlim(R_BINS[0], R_BINS[-1])

    r_b  = rho_bound_ts [bi, :]
    r_ub = rho_unbound_ts[bi, :]

    vb  = np.isfinite(r_b)  & (r_b  > 0)
    vub = np.isfinite(r_ub) & (r_ub > 0)

    if vb.any():  ax.plot(r_mid_sph[vb],  r_b[vb],  color=color, lw=2.0, label="Bound Core")
    if vub.any(): ax.plot(r_mid_sph[vub], r_ub[vub], color=color, lw=1.5, ls="--", alpha=0.7, label="Debris")
    if col == 0:  ax.set_ylabel(r"$\rho\ [M_\odot\ kpc^{-3}]$", fontsize=9)
    ax.legend(fontsize=7)

fig4.suptitle(r"Bound Core Components vs. Disrupted Unbound Density Overlays $\rho(r)$", fontsize=12)
fig4.savefig(os.path.join(OUT_DIR, "section22_bound_profiles.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig4)

# ── FIGURE 5 — DEBRIS STREAM EXTRACTION FIELD CORRELATIONS f_unbound(r, t) ──
print("[Fig 5] Plotting spatiotemporal debris heatmaps...")
t_bound_min, t_bound_max = np.nanmin(time_bound), np.nanmax(time_bound)
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06})

y_uniform_mix = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), 200)
f_unbound_interp = np.zeros((len(y_uniform_mix), n_bound))
for idx in range(n_bound):
    nm = np.isfinite(f_unbound_ts[idx, :])
    f_unbound_interp[:, idx] = np.interp(np.log10(y_uniform_mix), np.log10(r_mid_sph[nm]), f_unbound_ts[idx, nm]) if nm.sum() > 2 else np.nan

im5 = ax5a.imshow(f_unbound_interp, aspect="auto", origin="lower", extent=[t_bound_min, t_bound_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="hot", vmin=0.0, vmax=1.0)
_styled_ax(ax5a, xlabel=time_label, ylabel="r [kpc]", title=r"Spatiotemporal Debris Infiltration Heatmap: $f_{\rm unbound}(r,t)$")
ax5a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax5a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])

cb5 = fig5.colorbar(im5, ax=ax5a, pad=0.01)
cb5.set_label(r"$f_{\rm unbound}$ (0.0 = Spherically Retained, 1.0 = Fully Stripped)", fontsize=8)

f_mean = np.nanmean(f_unbound_ts, axis=0)
valid_f = np.isfinite(f_mean)
_styled_ax(ax5b, xlabel=r"$\langle f_{\rm unbound} \rangle_t$", title="Time Avg")
ax5b.plot(f_mean[valid_f], r_mid_sph[valid_f], color="#ff9944", lw=2.0)
ax5b.axvline(0.5, color="#ffffff", lw=0.7, ls="--", alpha=0.4)
ax5b.set_xlim(0, 1.05)
ax5b.set_yscale("log")
ax5b.set_ylim(R_BINS[0], R_BINS[-1])
ax5b.tick_params(labelleft=False)

fig5.suptitle("Radial Mass Disruption Boundaries", fontsize=12)
fig5.savefig(os.path.join(OUT_DIR, "section22_stream_fraction_heatmap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig5)

# ── FIGURE 6 — IMPULSIVE ENERGY INJECTIONS ΔE(r, t) ──
print("[Fig 6] Plotting energetic heating matrices...")
dE_max = np.nanpercentile(np.abs(dEkin_dt_ts[np.isfinite(dEkin_dt_ts)]), 97)
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06})

dEkin_interp = np.zeros((len(y_uniform_mix), ns))
for idx in range(ns):
    nm = np.isfinite(dEkin_dt_ts[idx, :])
    dEkin_interp[:, idx] = np.interp(np.log10(y_uniform_mix), np.log10(r_mid_sph[nm]), dEkin_dt_ts[idx, nm]) if nm.sum() > 2 else np.nan

im6 = ax6a.imshow(dEkin_interp, aspect="auto", origin="lower", extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="seismic", vmin=-dE_max, vmax=dE_max)
_styled_ax(ax6a, xlabel=time_label, ylabel="r [kpc]", title=r"Dynamic Energy Shock Dissipation: $\Delta E_{\rm kin}(r,t)\ [{\rm km^2\ s^{-2}\ snap^{-1}}]$")
ax6a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax6a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])

cb6 = fig6.colorbar(im6, ax=ax6a, pad=0.01)
cb6.set_label(r"$\Delta E_{\rm kin}$ (Blue = Bulk Cooling, Red = Shock Heating Injection)", fontsize=8)

dE_mean = np.nanmean(dEkin_dt_ts, axis=0)
valid_dE = np.isfinite(dE_mean)
_styled_ax(ax6b, xlabel=r"$\langle \Delta E_{\rm kin} \rangle_t$", title="Time Avg")
ax6b.plot(dE_mean[valid_dE], r_mid_sph[valid_dE], color="#e8673a", lw=2.0)
ax6b.axvline(0, color="#555577", lw=0.8, ls="--")
ax6b.set_yscale("log")
ax6b.set_ylim(R_BINS[0], R_BINS[-1])
ax6b.tick_params(labelleft=False)

fig6.suptitle("Tidal Kinetic Energy Injection Profiles", fontsize=12)
fig6.savefig(os.path.join(OUT_DIR, "section22_heating_heatmap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig6)

# ── FIGURE 7 — DETAILED SHOCK INJECTION SLICES ──
print("[Fig 7] Plotting energy profile overlays...")
fig7, ax7 = plt.subplots(figsize=(9, 6), facecolor=BG)
_styled_ax(ax7, xlabel="r [kpc]", ylabel=r"$\Delta E_{\rm kin}\ [{\rm km^2\ s^{-2}\ snap^{-1}}]$", title="Tidal Heating Vectors Across Epochs", log_x=True)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y = dEkin_dt_ts[k_idx, :]
    valid = np.isfinite(y)
    if valid.any(): ax7.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax7.axhline(0, color="#555577", lw=1.0, ls="--", alpha=0.6)
ax7.set_xlim(R_BINS[0], R_BINS[-1])
ax7.legend(fontsize=8)
fig7.savefig(os.path.join(OUT_DIR, "section22_heating_profiles.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig7)

# ── FIGURE 8 — TIDAL POTENTIAL QUADRUPOLE MATRIX TRAJECTORIES ──
print("[Fig 8] Plotting tidal tensor eigenvalues...")
fig8, axes8 = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})

_styled_ax(axes8[0], ylabel=r"$\lambda\ [{\rm km^2\ s^{-2}\ kpc^{-2}}]$", title="Tidal Tensor Matrix Eigenvalue Chronology at Midpoint Coordinate")
axes8[0].plot(time_tensor, T_eig1_arr, color="#ff9944", lw=1.8, label=r"$\lambda_1$ (Principal Axis Compressions)")
axes8[0].plot(time_tensor, T_eig2_arr, color="#4a8fff", lw=1.5, label=r"$\lambda_2$")
axes8[0].plot(time_tensor, T_eig3_arr, color="#e8673a", lw=1.8, label=r"$\lambda_3$ (Tidal Tail Prolate Stretching)")
axes8[0].axhline(0, color="#555577", lw=0.7, ls="--")
axes8[0].legend(fontsize=8)

tidal_aniso = (T_eig1_arr - T_eig3_arr) / (np.abs(T_eig1_arr) + np.abs(T_eig3_arr) + 1e-30)
_styled_ax(axes8[1], xlabel=time_label, ylabel="Tidal Anisotropy Index", title=r"Normalized Quadrupole Anisotropy Distortion: $\frac{\lambda_1 - \lambda_3}{|\lambda_1| + |\lambda_3|}$")
axes8[1].plot(time_tensor, tidal_aniso, color="#aa55ff", lw=1.8)
axes8[1].fill_between(time_tensor, np.where(np.isfinite(tidal_aniso), tidal_aniso, 0), alpha=0.12, color="#aa55ff")
fig8.savefig(os.path.join(OUT_DIR, "section22_tidal_tensor_eigen.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig8)

# Standalone anisotropy vector figure
fig8b, ax8b = plt.subplots(figsize=(10, 4), facecolor=BG)
_styled_ax(ax8b, xlabel=time_label, ylabel="Tidal Anisotropy", title=r"Asymmetric Deformation Index Trace: $\frac{\lambda_1-\lambda_3}{|\lambda_1|+|\lambda_3|}$")
ax8b.plot(time_tensor, tidal_aniso, color="#aa55ff", lw=2.0)
ax8b.fill_between(time_tensor, np.where(np.isfinite(tidal_aniso), tidal_aniso, 0), alpha=0.15, color="#aa55ff")
fig8b.savefig(os.path.join(OUT_DIR, "section22_tidal_anisotropy.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig8b)

# ── FIGURE 9 — OBSERVATIONAL INTERLOPER SCREENING (r, v_r) CONTROLS ──
print("[Fig 9] Plotting phase-space stream separations...")
fig9, axes9 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG, sharey=True, gridspec_kw={"wspace": 0.06})

for col, (bi, label, color) in enumerate(zip(profile_bi, PROFILE_LABELS, PROFILE_COLORS)):
    ax = axes9[col]
    _styled_ax(ax, xlabel="r [kpc]", title=label)
    if col == 0: ax.set_ylabel(r"$v_r\ [{\rm km\ s^{-1}}]$", fontsize=9)

    snap_num = bound_snap_nums[bi]
    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)): continue

    try:
        sd = load_snapshot_particles(mw_file, m31_file)
        mo = CenterOfMass(mw_file, PTYPE)
        ao = CenterOfMass(m31_file, PTYPE)
    except Exception: continue

    p_pos, p_mass = sd["pos"], sd["m_msun"]
    p_rad = np.linalg.norm(p_pos, axis=1)

    v_c_x = np.concatenate((mo.vx, ao.vx))
    v_c_y = np.concatenate((mo.vy, ao.vy))
    v_c_z = np.concatenate((mo.vz, ao.vz))
    m_co  = np.concatenate((mo.m, ao.m))

    xc, yc, zc = mo.COMdefine(np.concatenate((mo.x, ao.x)), np.concatenate((mo.y, ao.y)), np.concatenate((mo.z, ao.z)), m_co)
    dc = np.sqrt((np.concatenate((mo.x, ao.x))-xc)**2 + (np.concatenate((mo.y, ao.y))-yc)**2 + (np.concatenate((mo.z, ao.z))-zc)**2)
    inn_ap = dc < 15.0
    
    if inn_ap.sum() >= 5:
        w_ap = p_mass[inn_ap]
        vxc_l = np.sum(w_ap * v_c_x[inn_ap]) / w_ap.sum()
        vyc_l = np.sum(w_ap * v_c_y[inn_ap]) / w_ap.sum()
        vzc_l = np.sum(w_ap * v_c_z[inn_ap]) / w_ap.sum()
    else:
        vxc_l = vyc_l = vzc_l = 0.0

    p_vel = np.vstack((v_c_x - vxc_l, v_c_y - vyc_l, v_c_z - vzc_l)).T
    p_phi = specific_potential(p_rad, p_mass, R_BINS)
    p_Etot = 0.5 * np.sum(p_vel**2, axis=1) + p_phi

    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat_l = np.where(p_rad[:, None] > 0, p_pos / p_rad[:, None], 0.0)
    p_vr = np.einsum("ij,ij->i", p_vel, r_hat_l)

    b_idx_m = np.where(p_Etot < 0)[0]
    u_idx_m = np.where(p_Etot >= 0)[0]

    if len(b_idx_m) > 20000: b_idx_m = np.random.choice(b_idx_m, 20000, replace=False)
    if len(u_idx_m) > 20000: u_idx_m = np.random.choice(u_idx_m, 20000, replace=False)

    ax.scatter(p_rad[b_idx_m], p_vr[b_idx_m], s=0.5, alpha=0.3, color=color, rasterized=True)
    ax.scatter(p_rad[u_idx_m], p_vr[u_idx_m], s=0.5, alpha=0.4, color="#ff5566", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlim(R_BINS[0], R_BINS[-1])
    ax.set_ylim(-600, 600)

fig9.suptitle(r"Phase-Space Velocity Inversions $(r,v_r)$: Bound Core (Colored) vs. Ejected Debris Streams (Red)", fontsize=11)
fig9.savefig(os.path.join(OUT_DIR, "section22_stream_rv.png"), dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig9)

# ── FIGURE 10 — CARTESIAN DEBRIS DISTRIBUTION TOPOGRAPHY ──
print("[Fig 10] Plotting spatial stream topography...")
fig10, axes10 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG, sharey=True, gridspec_kw={"wspace": 0.04})

for col, (fi_s, label) in enumerate(zip(np.linspace(0, n_stream_frames-1, 5, dtype=int), PROFILE_LABELS)):
    ax = axes10[col]
    ax.set_facecolor(BG)
    Hs = gaussian_filter(stream_maps[fi_s], sigma=2.0)
    H_log = np.where(Hs > 0, np.log10(Hs), np.nan)

    ax.imshow(H_log.T, origin="lower", aspect="equal", extent=[-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT, -STREAM_MAP_EXTENT, STREAM_MAP_EXTENT], cmap="inferno",
              vmin=np.nanpercentile(H_log[np.isfinite(H_log)], 5) if np.isfinite(H_log).any() else 0,
              vmax=np.nanpercentile(H_log[np.isfinite(H_log)], 99) if np.isfinite(H_log).any() else 10)
    ax.set_title(label, fontsize=9, color="#c8c8e8")
    ax.tick_params(colors="#9090b0", labelsize=7)
    if col == 0: ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")
    ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")

fig10.suptitle(r"Projected Cartesian Surface Densities of Unbound Tidal Streams Debris $\Sigma_{\rm unbound}(x,y)$", fontsize=11)
fig10.savefig(os.path.join(OUT_DIR, "section22_stream_map.png"), dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig10)

# ── FIGURE 11 — ANISOTROPIC MOMENT RATIOS b/a AND c/a ──
print("[Fig 11] Plotting aperture profile geometries...")
fig11, axes11 = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})

_styled_ax(axes11[0], ylabel="b/a", title="Intermediate-to-Major Axis Configuration b/a (1.0 = Oblate/Spherically Uniform)")
for arr, color, label in [(ba_inner_arr, "#4a8fff", f"Inner Core Space (r < {INERTIA_R_INNER:.0f} kpc)"),
                          (ba_mid_arr,   "#00d4aa", f"Intermediate Shell ({INERTIA_R_INNER:.0f} - {INERTIA_R_MID:.0f} kpc)"),
                          (ba_outer_arr, "#ff9944", f"Outer Halo Space (r > {INERTIA_R_MID:.0f} kpc)"),
                          (ba_arr,       "#aaaacc", "Global Volume Enclosure")]:
    valid = np.isfinite(arr)
    if valid.any(): axes11[0].plot(time_inertia[valid], arr[valid], color=color, lw=1.8, label=label)
axes11[0].set_ylim(0, 1.05)
axes11[0].legend(fontsize=8)

_styled_ax(axes11[1], xlabel=time_label, ylabel="c/a", title="Minor-to-Major Axis Compression Tracking c/a (0.0 = Severe Prolate Elongation)")
for arr, color in [(ca_inner_arr, "#4a8fff"), (ca_mid_arr, "#00d4aa"), (ca_outer_arr, "#ff9944"), (ca_arr, "#aaaacc")]:
    valid = np.isfinite(arr)
    if valid.any(): axes11[1].plot(time_inertia[valid], arr[valid], color=color, lw=1.8)
axes11[1].set_ylim(0, 1.05)
fig11.savefig(os.path.join(OUT_DIR, "section22_inertia_axisratios.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig11)

# ── FIGURE 12 — SHELL ELONGATION MATRIX b/a(r, t) ──
print("[Fig 12] Plotting shell elongation matrices...")
t_inertia_min, t_inertia_max = np.nanmin(time_inertia), np.nanmax(time_inertia)
fig12, ax12 = plt.subplots(figsize=(11, 5), facecolor=BG)

ba_interp_map = np.zeros((len(y_uniform_j), n_inertia))
for idx in range(n_inertia):
    nm = np.isfinite(ba_profile_ts[idx, :])
    ba_interp_map[:, idx] = np.interp(np.log10(y_uniform_j), np.log10(r_mid_sph[nm]), ba_profile_ts[idx, nm]) if nm.sum() > 2 else np.nan

im12 = ax12.imshow(ba_interp_map, aspect="auto", origin="lower", extent=[t_inertia_min, t_inertia_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])], cmap="viridis_r", vmin=0.3, vmax=1.0)
_styled_ax(ax12, xlabel=time_label, ylabel="r [kpc]", title=r"Per-Shell Radial Axis Topology $b/a(r,t)$ (Dark = Prolate Stretching, Light = Isotropic Core)")
ax12.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax12.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])
cb12 = fig12.colorbar(im12, ax=ax12, pad=0.01)
cb12.set_label(r"$b/a$", fontsize=9)
fig12.savefig(os.path.join(OUT_DIR, "section22_inertia_heatmap.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig12)

# ── FIGURE 13 — DISCRETE SHAPE AXIS OVERLAYS ──
print("[Fig 13] Plotting structural shape profile overlays...")
inertia_profile_bi = [np.argmin(np.abs(time_inertia - time_arr[k])) for k in PROFILE_INDICES]
fig13, ax13 = plt.subplots(figsize=(9, 6), facecolor=BG)
_styled_ax(ax13, xlabel="r [kpc]", ylabel="b/a", title=r"Radial Axis Configurations: $b/a(r)$ Profiles across Key Stages", log_x=True)

for ii, color, label in zip(inertia_profile_bi, PROFILE_COLORS, PROFILE_LABELS):
    y = ba_profile_ts[ii, :]
    valid = np.isfinite(y)
    if valid.any(): ax13.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax13.axhline(1.0, color="#555577", lw=0.7, ls="--", alpha=0.5)
ax13.set_xlim(R_BINS[0], R_BINS[-1])
ax13.set_ylim(0.2, 1.05)
ax13.legend(fontsize=8)
fig13.savefig(os.path.join(OUT_DIR, "section22_inertia_profiles.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig13)

# ── FIGURE 14 — DISCRETE ORBITAL SEPARATION CHRONOLOGY ──
print("[Fig 14] Plotting Keplerian trajectory dispersions...")
fig14, (ax14a, ax14b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG, sharex=True, gridspec_kw={"hspace": 0.08})

_styled_ax(ax14a, ylabel="Separation [kpc]", title="System Co-Moving Separation Trajectory Baseline")
ax14a.plot(time_arr, separation_arr, color="#4a8fff", lw=2.0)
ax14a.set_yscale("log")

_styled_ax(ax14b, xlabel=time_label, ylabel=r"$v_{\rm rel}\ [{\rm kpc\ /\ snap}]$", title="Numerical Approach & Recession Flight Velocities")
valid_vrel = np.isfinite(v_rel_arr)
ax14b.plot(time_arr[valid_vrel], v_rel_arr[valid_vrel], color="#e8673a", lw=1.8)
ax14b.axhline(0, color="#555577", lw=0.8, ls="--")
fig14.savefig(os.path.join(OUT_DIR, "section22_mw_m31_separation.png"), dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig14)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §22.6 — MEDIA ENCODING ANIMATION LOOPS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Movie Panel 1: Cartesian Unbound Stream Render ──
print("\n[Anim 1] Packaging continuous 2D unbound stream animations...")
fig_a1, (axSM, axBF) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08})
axSM.set_facecolor(BG); axBF.set_facecolor(BG)

H0_log = np.where(gaussian_filter(stream_maps[0], sigma=2.0) > 0, np.log10(gaussian_filter(stream_maps[0], sigma=2.0)), np.nan)
all_vals = stream_maps[stream_maps > 0]
vmin_sm, vmax_sm = (np.log10(np.percentile(all_vals, 10)), np.log10(np.percentile(all_vals, 99))) if all_vals.size > 0 else (0, 10)

im_sm = axSM.imshow(H0_log.T, origin="lower", aspect="equal", extent=[-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT, -STREAM_MAP_EXTENT, STREAM_MAP_EXTENT], cmap="inferno", vmin=vmin_sm, vmax=vmax_sm)
axSM.set_xlabel("x [kpc]", color="#c8c8e8"); axSM.set_ylabel("y [kpc]", color="#c8c8e8")

bf_line, = axBF.plot([], [], color="#00d4aa", lw=1.8)
axBF.set_xlim(np.nanmin(time_bound), np.nanmax(time_bound))
axBF.set_ylim(0, 1.05)
axBF.set_xlabel(time_label, color="#c8c8e8"); axBF.set_ylabel("Bound fraction", color="#c8c8e8")
axBF.set_title(r"$f_{\rm bound}$ Profile History", color="#c8c8e8", fontsize=10)
axBF.axhline(0.5, color="#555577", lw=0.6, ls="--", alpha=0.5)
title_a1 = fig_a1.suptitle("", fontsize=11, color="#c8c8e8")

def _update_stream_anim(frame_idx):
    fi_s = stream_anim_snap_ids[frame_idx]
    H_log = np.where(gaussian_filter(stream_maps[fi_s], sigma=2.0) > 0, np.log10(gaussian_filter(stream_maps[fi_s], sigma=2.0)), np.nan)
    im_sm.set_data(H_log.T)

    valid = np.isfinite(time_bound[:fi_s+1]) & np.isfinite(bound_frac_arr[:fi_s+1])
    bf_line.set_data(time_bound[:fi_s+1][valid], bound_frac_arr[:fi_s+1][valid])

    t_cur = time_bound[fi_s] if fi_s < len(time_bound) else np.nan
    t_str = f"{t_cur:.2f} Gyr" if (np.isfinite(t_cur) and time_is_gyr) else f"Bound snap {fi_s}"
    title_a1.set_text(f"Dynamic Debris Streams Mapping  ·  {t_str}")
    return [im_sm, bf_line]

ani_sm = animation.FuncAnimation(fig_a1, _update_stream_anim, frames=n_stream_frames, interval=1000 // ANIM_FPS_22, blit=True)
writer_22 = animation.FFMpegWriter(fps=ANIM_FPS_22, bitrate=ANIM_BITRATE_22, metadata=dict(title="MW-M31 Unbound Stream Topography"))
ani_sm.save(os.path.join(OUT_DIR, "section22_animation_streams.mp4"), writer=writer_22, dpi=ANIM_DPI_22)
plt.close(fig_a1)

# ── Movie Panel 2: Spherically Restrained Mass Profile Animation ──
print("[Anim 2] Packaging bound-mass spatial profile animations...")
fig_a2, axes_a2 = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BG, gridspec_kw={"wspace": 0.32})
ax_bprof, ax_fub, ax_bf = axes_a2
for ax in axes_a2: ax.set_facecolor(BG)

rho_all = np.concatenate([rho_bound_ts.ravel(), rho_unbound_ts.ravel()])
rho_all = rho_all[np.isfinite(rho_all) & (rho_all > 0)]
rho_ymin, rho_ymax = (rho_all.min() * 0.3, rho_all.max() * 3.0) if rho_all.size > 0 else (1e2, 1e12)

ax_bprof.set_xscale("log"); ax_bprof.set_yscale("log")
ax_bprof.set_xlim(R_BINS[0], R_BINS[-1]); ax_bprof.set_ylim(rho_ymin, rho_ymax)
ax_bprof.set_xlabel("r [kpc]", color="#c8c8e8"); ax_bprof.set_ylabel(r"$\rho\ [M_\odot\ kpc^{-3}]$", color="#c8c8e8")
ax_bprof.set_title("Mechanical Binding Density", color="#c8c8e8", fontsize=10)

ax_fub.set_xscale("log"); ax_fub.set_xlim(R_BINS[0], R_BINS[-1]); ax_fub.set_ylim(0, 1.05)
ax_fub.set_xlabel("r [kpc]", color="#c8c8e8"); ax_fub.set_ylabel(r"$f_{\rm unbound}$", color="#c8c8e8")
ax_fub.set_title("Unbound Mass Shell Fractions", color="#c8c8e8", fontsize=10)
ax_fub.axhline(0.5, color="#555577", lw=0.7, ls="--")

ax_bf.set_xlim(np.nanmin(time_bound), np.nanmax(time_bound)); ax_bf.set_ylim(0, 1.05)
ax_bf.set_xlabel(time_label, color="#c8c8e8"); ax_bf.set_ylabel(r"$f_{\rm bound}$", color="#c8c8e8")
ax_bf.set_title("Global Retained Mass Fractions", color="#c8c8e8", fontsize=10)

line_bound, = ax_bprof.plot([], [], color="#4a8fff",  lw=2.0, label="Bound Core")
line_unb,   = ax_bprof.plot([], [], color="#ff5566",  lw=1.5, ls="--", label="Stripped Debris")
line_fub,   = ax_fub.plot([],   [], color="#ff9944",  lw=2.0)
line_bfhist, = ax_bf.plot([],    [], color="#00d4aa",  lw=1.8)
vline_bf    = ax_bf.axvline(np.nan, color="#ffffff",  lw=0.8, ls="--", alpha=0.5)
ax_bprof.legend(fontsize=8)

title_a2 = fig_a2.suptitle("", fontsize=11, color="#c8c8e8")

def _update_bound_anim(frame_idx):
    bi = bound_anim_snap_ids[frame_idx]

    def _xy_rho(arr):
        v = np.isfinite(arr) & (arr > 0)
        return r_mid_sph[v], arr[v]

    line_bound.set_data(*_xy_rho(rho_bound_ts[bi, :]))
    line_unb.set_data(*_xy_rho(rho_unbound_ts
