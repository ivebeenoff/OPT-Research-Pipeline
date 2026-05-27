"""
===============================================================================
SECTION 31 — LYAPUNOV EXPONENTS & ORBITAL CHAOS
===============================================================================
Author  : Abhinav Vatsa

Continuation of the MW–M31 analysis pipeline.  All globals (SNAPSHOTS, ns,
R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL,
G_KPC_KMS2_MSUN, PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr,
time_label, time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

This section REQUIRES the Lagrangian tracking arrays from Section 26:
    traj_pos  : (ns, N_TRACKED, 3)  — tracked particle positions
    traj_vel  : (ns, N_TRACKED, 3)  — tracked particle velocities
    traj_r    : (ns, N_TRACKED)     — tracked particle radii
    r0_tracked : (N_TRACKED,)       — initial radii
    group_label : (N_TRACKED,)      — 0=inner,1=mid,2=outer,3=M31
    GROUP_COLORS, GROUP_NAMES       — colour/name per group

If Section 26 was not run, this section recomputes a minimal set of
trajectories internally.

Physical motivation — Why Lyapunov exponents?
─────────────────────────────────────────────
The Lyapunov exponent λ measures the rate at which two initially nearby
trajectories diverge in phase space:

    δ(t) = δ(0) × e^{λ t}

where δ(t) = |Δx(t)|  is the separation between two trajectories that
started δ(0) apart.

  λ > 0  →  CHAOTIC orbit: exponential divergence.
             The orbit has no long-term predictability.
  λ ≤ 0  →  REGULAR orbit: polynomial or oscillatory divergence.
             Orbit is quasi-periodic and lies on a KAM torus.

In the context of the MW–M31 merger, tracking λ(t) per particle reveals:

  (A) CHAOS ONSET: At which merger epoch do initially regular orbits
      become chaotic?  This marks the transition from the pre-merger
      ordered phase to the violent relaxation phase.

  (B) RADIAL DEPENDENCE: Are inner-halo particles more or less chaotic
      than outer-halo particles?  Theory predicts outer particles are
      more chaotic (weaker restoring force, stronger tidal perturbation).

  (C) CHAOS FRACTION: What fraction of all tracked particles have λ > 0
      at any given time?  This is a direct measure of dynamical disorder.

  (D) CORRELATION WITH STRIPPING: Do particles that are eventually
      stripped (high r_f) have systematically higher λ than particles
      that remain bound?  This would link chaos to the stripping mechanism.

  (E) STOCHASTIC LAYER THICKNESS: In the (r, v_r) plane, chaotic orbits
      fill a "stochastic layer" around the separatrix of the effective
      potential.  Its width measures the degree of phase-space mixing.

Implementation — finite-time Lyapunov exponent (FTLE)
──────────────────────────────────────────────────────
Computing the true asymptotic Lyapunov exponent requires an infinitely
long trajectory.  From snapshot data we compute the FINITE-TIME Lyapunov
exponent (FTLE) over a window of W snapshots:

    λ_W(t, i) = (1 / (W Δt)) × ln(δ(t + W Δt) / δ(t))

where δ(t) is the distance between particle i and its "shadow" particle —
a nearby particle that started closest to it at t=0.

We use two complementary approaches:

  METHOD 1 — TRAJECTORY DIVERGENCE
    For each tracked particle i, find its nearest Lagrangian neighbour
    j* = argmin |x_j(t=0) − x_i(t=0)|.
    Track the separation δ_ij(t) = |x_i(t) − x_j(t)| over time.
    Fit log(δ_ij(t)) ~ λ t to estimate λ.

  METHOD 2 — STRETCHING FACTOR (FTLE MAP)
    At each snapshot, compute the deformation gradient tensor F of a
    small group of neighbouring particles.  The largest singular value
    of F gives the FTLE:
        λ_FTLE = (1/t) ln(σ_max(F))
    This is the standard computational method used in fluid dynamics
    for Lagrangian coherent structures (LCS).

Outputs
───────
  section31_lyapunov_trajectories.png   log(δ(t)) for sampled particle pairs
  section31_lambda_distribution.png     P(λ) distribution at 5 epochs
  section31_lambda_vs_r.png             λ(r_0) scatter by initial radius
  section31_chaos_fraction.png          Fraction of chaotic particles vs. time
  section31_lambda_heatmap.png          ⟨λ⟩(r_0, t) heatmap
  section31_ftle_map.png                2D FTLE field at 5 epochs
  section31_chaos_vs_stripping.png      λ vs. final radius r_f scatter
  section31_stochastic_layer.png        (r, v_r) coloured by λ at 5 epochs
  section31_animation_chaos.mp4         Chaos fraction and λ field animation
  section31_summary_panel.png           Master 4-panel summary

===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm
from scipy.spatial import cKDTree
from scipy.stats import linregress
import os
import time
import warnings


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.0 — CONFIGURATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── FTLE window width [snapshots] ─────────────────────────────────────────────
# λ is estimated over a sliding window of FTLE_WINDOW consecutive snapshots.
# Too short → dominated by short-time oscillations.
# Too long  → mixes different dynamical epochs.
# W ≈ 20–50 snapshots (200–500 Myr) is appropriate for halo orbits.
FTLE_WINDOW = 30

# ── Number of tracked particles for the FTLE computation ─────────────────────
# We use the trajectories from Section 26.  If not available, we track
# N_LYAP_PARTICLES fresh particles.
N_LYAP_PARTICLES = 800

# ── Chaos threshold: λ > CHAOS_THRESH is considered chaotic ──────────────────
# In units of inverse snapshots.  For Δt ≈ 10 Myr per snapshot,
# λ = 0.05 snap^{-1} corresponds to an e-folding time of 20 snaps = 200 Myr.
CHAOS_THRESH = 0.02   # [snap^{-1}]

# ── FTLE 2D map parameters ────────────────────────────────────────────────────
FTLE_MAP_BINS   = 100
FTLE_MAP_EXTENT = 300.0   # [kpc]
FTLE_MAP_STEP   = 40      # compute FTLE map every Nth snapshot

# ── Animation ─────────────────────────────────────────────────────────────────
ANIM_FPS_31     = 18
ANIM_DPI_31     = 100
ANIM_BITRATE_31 = 1600

print("\n" + "="*80)
print("  SECTION 31 · Lyapunov Exponents & Orbital Chaos")
print("="*80)
print(f"  FTLE window  : {FTLE_WINDOW} snapshots")
print(f"  Chaos thresh : λ > {CHAOS_THRESH} snap⁻¹")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.1 — LOAD OR INHERIT TRAJECTORIES FROM SECTION 26                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n  Loading trajectories …")

try:
    # Attempt to use the trajectories computed in Section 26.
    _traj_pos   = traj_pos    # (ns, N_TRACKED, 3)
    _traj_r     = traj_r      # (ns, N_TRACKED)
    _traj_vel   = traj_vel    # (ns, N_TRACKED, 3)
    _r0         = r0_tracked  # (N_TRACKED,)
    _group      = group_label # (N_TRACKED,)
    _N          = _traj_pos.shape[1]

    # Subsample to N_LYAP_PARTICLES if Section 26 tracked more.
    if _N > N_LYAP_PARTICLES:
        rng31  = np.random.default_rng(seed=31)
        sel    = rng31.choice(_N, size=N_LYAP_PARTICLES, replace=False)
        _traj_pos = _traj_pos[:, sel, :]
        _traj_r   = _traj_r  [:, sel  ]
        _traj_vel = _traj_vel[:, sel, :]
        _r0       = _r0      [sel]
        _group    = _group   [sel]
        _N        = N_LYAP_PARTICLES

    print(f"  Using {_N} trajectories from Section 26.")

except NameError:
    # Section 26 not run — recompute trajectories here using phase-space
    # nearest-neighbour tracking (same method as Section 26 fallback).
    print("  Section 26 trajectories not found — recomputing …")
    rng31 = np.random.default_rng(seed=31)

    # Load snapshot 0.
    snap0_num = SNAPSHOTS[0]
    mw0  = os.path.join(tmpdir, f"MW_{snap0_num:03d}.txt")
    m310 = os.path.join(tmpdir, f"M31_{snap0_num:03d}.txt")
    sd0  = load_snapshot_particles(mw0, m310)

    pos0_all = sd0["pos"]
    N_all    = len(pos0_all)
    sel      = rng31.choice(N_all, size=min(N_LYAP_PARTICLES, N_all), replace=False)
    _N       = len(sel)

    _r0    = np.linalg.norm(pos0_all[sel], axis=1)
    _group = np.zeros(_N, dtype=int)
    _group[_r0 < 10]  = 0
    _group[(_r0>=10) & (_r0<50)] = 1
    _group[_r0 >= 50] = 2

    _traj_pos = np.full((ns, _N, 3), np.nan)
    _traj_r   = np.full((ns, _N),    np.nan)
    _traj_vel = np.full((ns, _N, 3), np.nan)
    _traj_pos[0] = pos0_all[sel]
    _traj_r  [0] = _r0

    V_SCALE = 0.01
    prev6d  = np.zeros((_N, 6))
    prev6d[:, :3] = pos0_all[sel]

    MW0 = CenterOfMass(mw0,  PTYPE)
    M31_0 = CenterOfMass(m310, PTYPE)

    for i, snap_num in enumerate(SNAPSHOTS[1:], start=1):
        mw_f  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
        m31_f = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")
        if not (os.path.isfile(mw_f) and os.path.isfile(m31_f)):
            continue
        try:
            sd  = load_snapshot_particles(mw_f, m31_f)
            MW_ = CenterOfMass(mw_f,  PTYPE)
            M31_= CenterOfMass(m31_f, PTYPE)
        except Exception:
            continue

        pos_a = sd["pos"]
        m_a   = sd["m_msun"]
        vx_a  = np.concatenate((MW_.vx, M31_.vx))
        vy_a  = np.concatenate((MW_.vy, M31_.vy))
        vz_a  = np.concatenate((MW_.vz, M31_.vz))
        m_r   = np.concatenate((MW_.m, M31_.m))
        x_a   = np.concatenate((MW_.x, M31_.x))
        y_a   = np.concatenate((MW_.y, M31_.y))
        z_a   = np.concatenate((MW_.z, M31_.z))
        xc,yc,zc = MW_.COMdefine(x_a,y_a,z_a,m_r)
        dr_c  = np.sqrt((x_a-xc)**2+(y_a-yc)**2+(z_a-zc)**2)
        inn   = dr_c < 15
        if inn.sum()>=5:
            wi=m_a[inn]; vxc=np.sum(wi*vx_a[inn])/wi.sum()
            vyc=np.sum(wi*vy_a[inn])/wi.sum(); vzc=np.sum(wi*vz_a[inn])/wi.sum()
        else:
            vxc=vyc=vzc=0.0
        vel_a = np.vstack((vx_a-vxc, vy_a-vyc, vz_a-vzc)).T
        pos6d = np.hstack([pos_a, vel_a*V_SCALE])
        tree  = cKDTree(pos6d)
        dists, nn_idx = tree.query(prev6d, k=1, workers=-1)
        for t_idx in range(_N):
            if dists[t_idx] > 5.0:
                continue
            k = nn_idx[t_idx]
            _traj_pos[i, t_idx] = pos_a[k]
            _traj_vel[i, t_idx] = vel_a[k]
            _traj_r  [i, t_idx] = np.linalg.norm(pos_a[k])
            prev6d[t_idx, :3]   = pos_a[k]
            prev6d[t_idx, 3:]   = vel_a[k] * V_SCALE

    print(f"  Computed {_N} trajectories.")

    try:
        GROUP_COLORS = ["#ff5566","#ffaa44","#4a8fff","#aa55ff"]
        GROUP_NAMES  = ["Inner","Mid","Outer","M31"]
    except NameError:
        pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.2 — FIND NEAREST LAGRANGIAN NEIGHBOURS (SHADOW PARTICLES)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# For each tracked particle i, we find the particle j* that was closest to it
# at t=0 in phase space.  This pair (i, j*) forms the "trajectory + shadow"
# system for the FTLE calculation.  The initial separation δ_0 = |x_i(0) − x_j*(0)|
# is the seed distance for the divergence measurement.

print("\n  Finding nearest Lagrangian neighbours …")

pos0_tracked = _traj_pos[0]   # (N, 3)  initial positions

# Build a KD-tree over the initial positions.
tree0  = cKDTree(pos0_tracked)
# Query k=2 so the nearest neighbour is not the particle itself.
dists0, nn_idx0 = tree0.query(pos0_tracked, k=2)

# Shadow particle index for each tracked particle.
shadow_idx = nn_idx0[:, 1]    # (N,)  — index of nearest neighbour in tracked set
delta_0    = dists0[:, 1]     # (N,)  — initial separation [kpc]

print(f"  Mean initial separation δ_0 = {delta_0.mean():.2f} kpc")
print(f"  Min / Max : {delta_0.min():.3f} / {delta_0.max():.1f} kpc")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.3 — COMPUTE TRAJECTORY SEPARATIONS δ(t)                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("  Computing trajectory separations δ(t) …")

# Separation between each particle and its shadow at every snapshot.
# Shape: (ns, N)
delta_ts = np.full((ns, _N), np.nan)

for i in range(ns):
    pos_i      = _traj_pos[i]                    # (N, 3)
    pos_shadow = _traj_pos[i, shadow_idx, :]     # (N, 3)
    valid      = (np.isfinite(pos_i).all(axis=1) &
                  np.isfinite(pos_shadow).all(axis=1))
    d          = np.linalg.norm(pos_i - pos_shadow, axis=1)
    delta_ts[i, valid] = d[valid]

# Log separation: ln(δ(t) / δ(0)).
# A positive slope in this quantity vs. time is the Lyapunov exponent.
with np.errstate(divide="ignore", invalid="ignore"):
    ln_delta_ts = np.where(
        (delta_ts > 0) & (delta_0[None, :] > 0),
        np.log(delta_ts / delta_0[None, :]),
        np.nan,
    )   # shape (ns, N)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.4 — COMPUTE FINITE-TIME LYAPUNOV EXPONENTS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# For each particle i and each time step t, compute the FTLE over the window
# [t, t + FTLE_WINDOW]:
#
#   λ_W(t, i) = (ln δ(t+W) − ln δ(t)) / W
#             = (1/W) × Δ ln δ
#
# This is equivalent to fitting a line to ln δ vs. snapshot index over the
# window and taking the slope.  We use the simple finite-difference version
# (endpoint difference) for speed, and also compute a full linear regression
# version for the final snapshot to get a more accurate λ estimate.

print("  Computing FTLEs …")

# FTLE at every snapshot (endpoint finite difference over FTLE_WINDOW).
# Shape: (ns, N)
lambda_ts = np.full((ns, _N), np.nan)

for t in range(ns - FTLE_WINDOW):
    t_end = t + FTLE_WINDOW
    ln_start = ln_delta_ts[t,    :]
    ln_end   = ln_delta_ts[t_end, :]
    valid    = np.isfinite(ln_start) & np.isfinite(ln_end)
    lambda_ts[t, valid] = (ln_end[valid] - ln_start[valid]) / FTLE_WINDOW

# ── Also compute λ via linear regression over the full time series ────────────
# This gives a single "best estimate" λ per particle for the scatter plots.
lambda_total = np.full(_N, np.nan)   # best-fit λ over the full trajectory

for p in range(_N):
    ln_d = ln_delta_ts[:, p]
    valid = np.isfinite(ln_d)
    if valid.sum() < 10:
        continue
    t_idx_valid = np.arange(ns)[valid]
    slope, intercept, r_val, _, _ = linregress(t_idx_valid, ln_d[valid])
    lambda_total[p] = float(slope)

# ── Chaos classification ───────────────────────────────────────────────────────
is_chaotic = lambda_total > CHAOS_THRESH   # (N,) boolean

# ── Mean FTLE per snapshot (scalar time series) ───────────────────────────────
lambda_mean_ts  = np.nanmean(lambda_ts, axis=1)   # (ns,)
chaos_frac_ts   = np.nanmean(lambda_ts > CHAOS_THRESH, axis=1)  # (ns,)

# ── Mean FTLE per initial-radius bin × snapshot ──────────────────────────────
bin_id_r0   = np.digitize(_r0, R_BINS) - 1
lambda_r_ts = np.full((ns, nb_sph), np.nan)

for t in range(ns):
    for b in range(nb_sph):
        mask = (bin_id_r0 == b) & np.isfinite(lambda_ts[t])
        if mask.sum() >= 3:
            lambda_r_ts[t, b] = np.nanmean(lambda_ts[t, mask])

print(f"  Chaotic particles (λ > {CHAOS_THRESH}): "
      f"{is_chaotic.sum()}/{_N} = {is_chaotic.mean()*100:.1f}%")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.5 — FTLE 2D MAP VIA DEFORMATION GRADIENT                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# At selected snapshots we compute the 2D FTLE field in the x–y midplane.
# For each grid cell we find the 6 nearest tracked particles and compute the
# deformation gradient tensor F:
#
#   F_ij = ∂x_i(t) / ∂x_j(0)  (linearised flow map)
#
# The FTLE is then:  λ_FTLE = (1/t) ln(σ_max(F))
# where σ_max is the largest singular value of F.
#
# This produces a spatially resolved map of chaos that reveals the
# Lagrangian Coherent Structures (LCS) — boundaries between chaotic and
# regular regions in the phase space of the merger.

print("  Computing 2D FTLE maps …")

ftle_map_snap_nums = SNAPSHOTS[::FTLE_MAP_STEP]
n_ftle_maps        = len(ftle_map_snap_nums)
ftle_maps          = np.full((n_ftle_maps, FTLE_MAP_BINS, FTLE_MAP_BINS), np.nan)
time_ftle_maps     = np.full(n_ftle_maps, np.nan)

cell_size_ftle = 2.0 * FTLE_MAP_EXTENT / FTLE_MAP_BINS

for fi, snap_num in enumerate(ftle_map_snap_nums):
    snap_global_i = np.where(SNAPSHOTS == snap_num)[0]
    if len(snap_global_i) == 0:
        continue
    si = snap_global_i[0]
    time_ftle_maps[fi] = time_arr[si] if np.isfinite(time_arr[si]) else float(snap_num)

    # Get current and initial positions of tracked particles.
    pos_now   = _traj_pos[si]              # (N, 3)
    pos_init  = _traj_pos[0]              # (N, 3)
    valid_p   = (np.isfinite(pos_now).all(axis=1) &
                 np.isfinite(pos_init).all(axis=1))
    if valid_p.sum() < 10:
        continue

    pos_now_v  = pos_now [valid_p]
    pos_init_v = pos_init[valid_p]

    # Build KD-tree over initial x–y positions for neighbour lookup.
    tree_init = cKDTree(pos_init_v[:, :2])

    for xi in range(FTLE_MAP_BINS):
        for yi in range(FTLE_MAP_BINS):
            x_c = -FTLE_MAP_EXTENT + (xi + 0.5) * cell_size_ftle
            y_c = -FTLE_MAP_EXTENT + (yi + 0.5) * cell_size_ftle

            # Find 6 nearest tracked particles to this grid point (at t=0).
            k_nbr = 6
            dists_g, idx_g = tree_init.query([x_c, y_c], k=k_nbr)

            if dists_g.max() > cell_size_ftle * 5:
                continue   # no nearby particles — skip

            # Initial and final 2D positions of the k neighbours.
            x0 = pos_init_v[idx_g, :2]  # (k, 2)
            xt = pos_now_v [idx_g, :2]  # (k, 2)

            # Deformation gradient F ≈ least-squares fit of xt = F x0.
            # Solve: xt.T = F x0.T  →  F = xt.T @ pinv(x0.T)
            try:
                F = np.linalg.lstsq(x0, xt, rcond=None)[0].T   # (2, 2)
                sigma_max = np.linalg.svd(F, compute_uv=False).max()
                if sigma_max > 1.0 and si > 0:
                    ftle_maps[fi, xi, yi] = np.log(sigma_max) / si
            except Exception:
                pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.6 — FIGURES                                                           ║
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
profile_snap_ii = [int(f * (ns - 1)) for f in [0.0, 0.2, 0.4, 0.65, 1.0]]

try:
    G_COLORS = GROUP_COLORS
    G_NAMES  = GROUP_NAMES
except NameError:
    G_COLORS = ["#ff5566","#ffaa44","#4a8fff","#aa55ff"]
    G_NAMES  = ["Inner","Mid","Outer","M31"]


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — log δ(t) TRAJECTORIES FOR SAMPLED PAIRS
# ══════════════════════════════════════════════════════════════════════════════
#
# Plotting ln(δ/δ_0) vs. time for individual particle pairs directly reveals
# the two regimes: a positive linear slope (chaotic, λ > 0) and a flat or
# oscillating profile (regular, λ ≈ 0).  The slope of the linear regime IS
# the Lyapunov exponent.

print("\n[Fig 1]  log-separation trajectories …")

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                            gridspec_kw={"wspace": 0.28})

for ax, title, mask_fn in [
    (axes1[0], "Chaotic particles  (λ > threshold)",
     lambda: is_chaotic),
    (axes1[1], "Regular particles  (λ ≤ threshold)",
     lambda: ~is_chaotic),
]:
    _ax(ax, xlabel=time_label,
        ylabel=r"$\ln(\delta / \delta_0)$",
        title=title)

    idx_show = np.where(mask_fn() & np.isfinite(lambda_total))[0][:30]
    lam_show = lambda_total[idx_show]
    lam_norm = Normalize(vmin=lam_show.min() if lam_show.size > 0 else 0,
                         vmax=lam_show.max() if lam_show.size > 0 else 1)

    for p in idx_show:
        y     = ln_delta_ts[:, p]
        valid = np.isfinite(y)
        if valid.sum() < 5:
            continue
        color = plt.cm.plasma(lam_norm(lambda_total[p]))
        ax.plot(time_arr[valid], y[valid], lw=0.7, alpha=0.5, color=color)

    ax.axhline(0, color="#555577", lw=0.8, ls="--", alpha=0.5)
    # Reference line for λ = CHAOS_THRESH.
    t_ref = np.linspace(t_min, t_max, 100)
    t_idx_ref = np.linspace(0, ns, 100)
    ax.plot(t_ref, CHAOS_THRESH * t_idx_ref, color="#ff9944", lw=1.0,
            ls=":", label=fr"λ = {CHAOS_THRESH} threshold")
    ax.legend(fontsize=7)

fig1.suptitle(r"Trajectory Separation  $\ln(\delta(t)/\delta_0)$  "
              r"— Chaotic vs. Regular",
              fontsize=12, color="#c8c8e8")
fig1.savefig(os.path.join(OUT_DIR, "section31_lyapunov_trajectories.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print("  Saved: section31_lyapunov_trajectories.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — P(λ) DISTRIBUTION AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════
#
# The distribution of FTLE values across all tracked particles shifts as
# the merger progresses.  A broadening distribution and rightward shift of
# the peak marks the onset of widespread chaos at pericentre.
# The negative tail (λ < 0) represents particles on temporarily converging
# trajectories — not truly regular, but momentarily compressing.

print("[Fig 2]  P(λ) distributions …")

fig2, ax2 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax2, xlabel=r"FTLE  $\lambda$  [snap$^{-1}$]",
    ylabel="Probability density",
    title=r"Distribution of Finite-Time Lyapunov Exponents  $P(\lambda)$")

for k_idx, color, label in zip(profile_snap_ii, PROFILE_COLORS, PROFILE_LABELS):
    lam = lambda_ts[k_idx, :]
    valid = np.isfinite(lam)
    if valid.sum() < 5:
        continue
    ax2.hist(lam[valid], bins=50, density=True, alpha=0.5,
             color=color, edgecolor="none", label=label)

ax2.axvline(CHAOS_THRESH, color="#ffffff", lw=1.0, ls="--", alpha=0.6,
            label=f"Chaos threshold λ = {CHAOS_THRESH}")
ax2.axvline(0, color="#555577", lw=0.8, ls=":", alpha=0.5)
ax2.legend(fontsize=7)

fig2.savefig(os.path.join(OUT_DIR, "section31_lambda_distribution.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("  Saved: section31_lambda_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — λ vs. INITIAL RADIUS r_0
# ══════════════════════════════════════════════════════════════════════════════
#
# If outer particles are more chaotic than inner particles, there should be
# a positive correlation between r_0 and λ.  A clear trend would support the
# "outside-in chaos" picture where tidal perturbations destabilise outer orbits
# first and the chaotic zone propagates inward with each pericentre passage.

print("[Fig 3]  λ vs. r_0 scatter …")

fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax3, xlabel=r"Initial radius $r_0$ [kpc]",
    ylabel=r"Total FTLE  $\lambda$  [snap$^{-1}$]",
    title=r"Orbital Chaos vs. Initial Radius",
    log_x=True)

valid_lam = np.isfinite(lambda_total) & np.isfinite(_r0)
sc = ax3.scatter(
    _r0[valid_lam], lambda_total[valid_lam],
    c=lambda_total[valid_lam],
    cmap="plasma", s=6, alpha=0.6, rasterized=True,
    norm=Normalize(vmin=np.nanpercentile(lambda_total[valid_lam], 5),
                   vmax=np.nanpercentile(lambda_total[valid_lam], 95)),
)
fig3.colorbar(sc, ax=ax3, label=r"$\lambda$  [snap$^{-1}$]", pad=0.01)
ax3.axhline(CHAOS_THRESH, color="#ff9944", lw=0.9, ls="--",
            label=f"Chaos threshold {CHAOS_THRESH}")
ax3.axhline(0, color="#555577", lw=0.7, ls=":", alpha=0.5)
ax3.set_xlim(_r0[valid_lam].min(), _r0[valid_lam].max())
ax3.legend(fontsize=8)

fig3.savefig(os.path.join(OUT_DIR, "section31_lambda_vs_r.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print("  Saved: section31_lambda_vs_r.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — CHAOS FRACTION vs. TIME
# ══════════════════════════════════════════════════════════════════════════════
#
# The fraction of particles with λ > CHAOS_THRESH at each snapshot is the
# simplest global measure of dynamical disorder.  It should:
#   • Start low (ordered pre-merger halos)
#   • Rise sharply at first pericentre
#   • Plateau near 1 post-merger (virialised chaotic halo)
# This is the N-body analogue of the "chaos fraction" tracked in theoretical
# studies of violent relaxation (Merritt 1999, Valluri et al. 2005).

print("[Fig 4]  Chaos fraction vs. time …")

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
                                   sharex=True, gridspec_kw={"hspace": 0.08})

_ax(ax4a, ylabel="Chaos fraction  f(λ > threshold)",
    title=fr"Fraction of Chaotic Particles  (λ > {CHAOS_THRESH} snap$^{{-1}}$)")
valid_cf = np.isfinite(chaos_frac_ts)
ax4a.plot(time_arr[valid_cf], chaos_frac_ts[valid_cf],
          color="#e8673a", lw=2.0)
ax4a.fill_between(time_arr[valid_cf],
                  0, chaos_frac_ts[valid_cf],
                  alpha=0.14, color="#e8673a")
ax4a.set_ylim(0, 1.05)

_ax(ax4b, xlabel=time_label,
    ylabel=r"Mean FTLE  $\langle\lambda\rangle$  [snap$^{-1}$]",
    title="Mean Finite-Time Lyapunov Exponent")
valid_lm = np.isfinite(lambda_mean_ts)
ax4b.plot(time_arr[valid_lm], lambda_mean_ts[valid_lm],
          color="#aa55ff", lw=1.8)
ax4b.axhline(CHAOS_THRESH, color="#ff9944", lw=0.8, ls="--",
             alpha=0.6, label=f"Threshold {CHAOS_THRESH}")
ax4b.axhline(0, color="#555577", lw=0.7, ls=":", alpha=0.5)
ax4b.legend(fontsize=8)

fig4.savefig(os.path.join(OUT_DIR, "section31_chaos_fraction.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig4)
print("  Saved: section31_chaos_fraction.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — ⟨λ⟩(r_0, t) HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
#
# The spatiotemporal map of mean FTLE binned by initial radius shows exactly
# when and at which radii chaos spreads through the halo.  A wave of high λ
# propagating inward from the outer halo after pericentre would directly
# demonstrate the "chaotic heating front" mechanism of violent relaxation.

print("[Fig 5]  ⟨λ⟩(r_0, t) heatmap …")

lam_max = np.nanpercentile(np.abs(lambda_r_ts[np.isfinite(lambda_r_ts)]), 95)

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                                   gridspec_kw={"width_ratios":[3,1],"wspace":0.06})

im5 = ax5a.imshow(
    np.clip(lambda_r_ts, -lam_max, lam_max).T,
    aspect="auto", origin="lower",
    extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
    cmap="seismic",
    norm=TwoSlopeNorm(vmin=-lam_max, vcenter=0.0, vmax=lam_max),
)
ax5a.set_yscale("log")
_ax(ax5a, xlabel=time_label, ylabel=r"Initial radius $r_0$ [kpc]",
    title=r"Mean FTLE  $\langle\lambda\rangle(r_0,\,t)$  "
          r"[red = chaotic, blue = converging]")
cb5 = fig5.colorbar(im5, ax=ax5a, pad=0.01)
cb5.set_label(r"$\langle\lambda\rangle$  [snap$^{-1}$]", fontsize=8)

lam_time_mean = np.nanmean(lambda_r_ts, axis=0)
valid_ltm = np.isfinite(lam_time_mean)
_ax(ax5b, xlabel=r"$\langle\lambda\rangle_t$", title="Time avg.")
ax5b.plot(lam_time_mean[valid_ltm], r_mid_sph[valid_ltm],
          color="#e8673a", lw=2.0)
ax5b.axvline(CHAOS_THRESH, color="#ff9944", lw=0.8, ls="--", alpha=0.6)
ax5b.axvline(0, color="#555577", lw=0.7, ls=":")
ax5b.set_yscale("log"); ax5b.set_ylim(R_BINS[0], R_BINS[-1])
ax5b.tick_params(labelleft=False)

fig5.savefig(os.path.join(OUT_DIR, "section31_lambda_heatmap.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig5)
print("  Saved: section31_lambda_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — 2D FTLE FIELD AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════
#
# The spatial FTLE field reveals the Lagrangian Coherent Structures (LCS) —
# the ridges of high FTLE that act as transport barriers in the flow.
# In a galaxy merger context these are the boundaries between the streaming
# orbits (tidal arms) and the well-mixed chaotic halo interior.

print("[Fig 6]  2D FTLE maps …")

ftle_profile_fi = [int(f * (n_ftle_maps - 1)) for f in [0.0, 0.2, 0.4, 0.65, 1.0]]

fig6, axes6 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.04})

for col, fi in enumerate(ftle_profile_fi):
    ax = axes6[col]
    ax.set_facecolor(BG)
    F_map = ftle_maps[fi]
    F_map_smooth = gaussian_filter(
        np.where(np.isfinite(F_map), F_map, 0.0), sigma=1.5)

    all_F = F_map_smooth[F_map_smooth > 0]
    vmax_f = np.percentile(all_F, 97) if all_F.size > 0 else 0.1
    vmin_f = 0.0

    ax.imshow(F_map_smooth.T, origin="lower", aspect="equal",
              extent=[-FTLE_MAP_EXTENT, FTLE_MAP_EXTENT,
                      -FTLE_MAP_EXTENT, FTLE_MAP_EXTENT],
              cmap="inferno", vmin=vmin_f, vmax=vmax_f)

    t_val = time_ftle_maps[fi]
    t_str = f"{t_val:.2f} Gyr" if (np.isfinite(t_val) and time_is_gyr) \
            else f"Snap {ftle_map_snap_nums[fi]}"
    ax.set_title(t_str, fontsize=9, color="#c8c8e8")
    ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")
    ax.tick_params(colors="#9090b0", labelsize=7)
    if col == 0:
        ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")

fig6.suptitle(r"2D FTLE Field  (Lagrangian Coherent Structures)  "
              r"— bright = chaotic boundary",
              fontsize=11, color="#c8c8e8")
fig6.savefig(os.path.join(OUT_DIR, "section31_ftle_map.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig6)
print("  Saved: section31_ftle_map.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — CHAOS vs. STRIPPING: λ vs. FINAL RADIUS r_f
# ══════════════════════════════════════════════════════════════════════════════
#
# Do particles that end up at large radii (stripped into streams) have
# systematically higher Lyapunov exponents?  If so, chaos is causally
# linked to stripping — the orbit became chaotic before the particle
# was ejected.  If not, stripping is a separate process (tidal truncation
# of regular outer-halo orbits).

print("[Fig 7]  Chaos vs. stripping …")

try:
    r_final_31 = r_final  # from Section 26
except NameError:
    r_final_31 = _traj_r[np.where(np.isfinite(np.nanmean(_traj_r, axis=1)))[0][-1]]

fig7, ax7 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax7, xlabel=r"Final radius $r_f$ [kpc]",
    ylabel=r"Total FTLE  $\lambda$  [snap$^{-1}$]",
    title="Orbital Chaos vs. Final Radius  (chaos → stripping link)",
    log_x=True)

valid_both = np.isfinite(lambda_total) & np.isfinite(r_final_31)
sc7 = ax7.scatter(
    r_final_31[valid_both], lambda_total[valid_both],
    c=_r0[valid_both],
    cmap="viridis", s=5, alpha=0.5, rasterized=True,
    norm=LogNorm(vmin=max(_r0[valid_both].min(), 0.1),
                 vmax=_r0[valid_both].max()),
)
fig7.colorbar(sc7, ax=ax7, label=r"Initial radius $r_0$ [kpc]", pad=0.01)
ax7.axhline(CHAOS_THRESH, color="#ff9944", lw=0.8, ls="--", alpha=0.7,
            label=f"Chaos threshold")
ax7.axhline(0, color="#555577", lw=0.7, ls=":")
ax7.legend(fontsize=8)

fig7.savefig(os.path.join(OUT_DIR, "section31_chaos_vs_stripping.png"),
             dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig7)
print("  Saved: section31_chaos_vs_stripping.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — (r, v_r) PHASE SPACE COLOURED BY λ AT FIVE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════
#
# Chaotic particles fill the "stochastic layer" around the separatrix of the
# effective potential — a broad band in (r, v_r) space.  Regular particles
# trace thin curves (KAM tori).  Colouring the (r, v_r) scatter by λ makes
# the stochastic layer directly visible as a diffuse coloured halo around the
# ordered orbit curves.

print("[Fig 8]  (r, v_r) coloured by λ …")

try:
    _traj_vr_local = traj_vr[:, :_N]  # from Section 26
except NameError:
    # Recompute v_r from trajectories.
    _traj_vr_local = np.full((ns, _N), np.nan)
    for i in range(ns):
        pos_t = _traj_pos[i]
        vel_t = _traj_vel[i]
        r_t   = np.linalg.norm(pos_t, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_hat = np.where(r_t[:,None] > 0, pos_t/r_t[:,None], 0.0)
        _traj_vr_local[i] = np.einsum("ij,ij->i", vel_t, r_hat)

fig8, axes8 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
                            sharey=True, gridspec_kw={"wspace": 0.06})

lam_norm8 = Normalize(
    vmin=np.nanpercentile(lambda_total[np.isfinite(lambda_total)], 5),
    vmax=np.nanpercentile(lambda_total[np.isfinite(lambda_total)], 95),
)

for col, (k_idx, label) in enumerate(zip(profile_snap_ii, PROFILE_LABELS)):
    ax = axes8[col]
    _ax(ax, xlabel="r [kpc]", title=label, log_x=True)
    if col == 0:
        ax.set_ylabel(r"$v_r$ [km s$^{-1}$]", fontsize=9)

    r_now  = _traj_r       [k_idx]
    vr_now = _traj_vr_local[k_idx]
    valid  = np.isfinite(r_now) & np.isfinite(vr_now) & np.isfinite(lambda_total)

    sc8 = ax.scatter(
        r_now[valid], vr_now[valid],
        c=lambda_total[valid],
        cmap="seismic", s=4, alpha=0.6,
        norm=TwoSlopeNorm(
            vmin=lam_norm8.vmin, vcenter=CHAOS_THRESH,
            vmax=lam_norm8.vmax
        ),
        rasterized=True,
    )
    ax.axhline(0, color="#555577", lw=0.5, ls="--")
    ax.set_xlim(0.1, 400); ax.set_ylim(-500, 500)

fig8.suptitle(r"$(r,\,v_r)$ Phase Space Coloured by FTLE  "
              r"(red = chaotic, blue = converging, white = threshold)",
              fontsize=10, color="#c8c8e8")
fig8.savefig(os.path.join(OUT_DIR, "section31_stochastic_layer.png"),
             dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig8)
print("  Saved: section31_stochastic_layer.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.7 — ANIMATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Three-panel animation:
#   Left  : 2D FTLE field (spatial chaos map)
#   Centre: chaos fraction f(t) running history
#   Right : P(λ) histogram for current snapshot

print("\n[Anim]  Chaos animation …")

anim_idxs_31 = np.arange(0, n_ftle_maps)
N_FRAMES_31  = len(anim_idxs_31)

fig_a, (ax_fl, ax_cf, ax_ph) = plt.subplots(
    1, 3, figsize=(16, 5.5), facecolor=BG,
    gridspec_kw={"width_ratios": [3, 2, 2], "wspace": 0.1},
)
for ax in (ax_fl, ax_cf, ax_ph):
    ax.set_facecolor(BG)

# FTLE map panel.
ax_fl.set_xlim(-FTLE_MAP_EXTENT, FTLE_MAP_EXTENT)
ax_fl.set_ylim(-FTLE_MAP_EXTENT, FTLE_MAP_EXTENT)
ax_fl.set_xlabel("x [kpc]", color="#c8c8e8")
ax_fl.set_ylabel("y [kpc]", color="#c8c8e8")

all_ftle_vals = ftle_maps[np.isfinite(ftle_maps) & (ftle_maps > 0)]
vmax_a = np.percentile(all_ftle_vals, 97) if all_ftle_vals.size > 0 else 0.1

F0     = ftle_maps[0]
F0_s   = gaussian_filter(np.where(np.isfinite(F0), F0, 0.0), sigma=1.5)
im_fl  = ax_fl.imshow(F0_s.T, origin="lower", aspect="equal",
                       extent=[-FTLE_MAP_EXTENT, FTLE_MAP_EXTENT,
                               -FTLE_MAP_EXTENT, FTLE_MAP_EXTENT],
                       cmap="inferno", vmin=0, vmax=vmax_a)

# Chaos fraction panel.
ax_cf.set_xlim(t_min, t_max)
ax_cf.set_ylim(0, 1.05)
ax_cf.set_xlabel(time_label, color="#c8c8e8")
ax_cf.set_ylabel("Chaos fraction", color="#c8c8e8")
cf_line, = ax_cf.plot([], [], color="#e8673a", lw=1.8)

# P(λ) histogram panel.
lam_bins = np.linspace(-0.1, 0.2, 51)
lam_cents = 0.5*(lam_bins[:-1]+lam_bins[1:])
ax_ph.set_xlim(lam_bins[0], lam_bins[-1])
ax_ph.set_xlabel(r"$\lambda$ [snap$^{-1}$]", color="#c8c8e8")
ph_bars = ax_ph.bar(lam_cents, np.zeros(50), width=np.diff(lam_bins),
                     color="#aa55ff", alpha=0.75, edgecolor="none")
ax_ph.axvline(CHAOS_THRESH, color="#ff9944", lw=0.9, ls="--", alpha=0.7)

title_a = fig_a.suptitle("", fontsize=11, color="#c8c8e8")


def _update_chaos_anim(frame_idx):
    fi = anim_idxs_31[frame_idx]
    snap_num = ftle_map_snap_nums[fi]
    snap_i   = np.where(SNAPSHOTS == snap_num)[0]
    si       = snap_i[0] if len(snap_i) > 0 else 0

    # FTLE map.
    F = ftle_maps[fi]
    Fs = gaussian_filter(np.where(np.isfinite(F), F, 0.0), sigma=1.5)
    im_fl.set_data(Fs.T)

    # Chaos fraction history.
    valid = np.isfinite(time_arr[:si+1]) & np.isfinite(chaos_frac_ts[:si+1])
    cf_line.set_data(time_arr[:si+1][valid], chaos_frac_ts[:si+1][valid])

    # P(λ) histogram.
    lam_now = lambda_ts[si, :]
    valid_l = np.isfinite(lam_now)
    if valid_l.sum() > 5:
        counts, _ = np.histogram(lam_now[valid_l], bins=lam_bins, density=True)
    else:
        counts = np.zeros(50)
    for bar, h in zip(ph_bars, counts):
        bar.set_height(h)
    ax_ph.set_ylim(0, max(counts.max()*1.15, 0.1))

    t_val = time_ftle_maps[fi]
    t_str = f"{t_val:.2f} Gyr" if (np.isfinite(t_val) and time_is_gyr) \
            else f"Snap {snap_num}"
    cf_val = chaos_frac_ts[si] if np.isfinite(chaos_frac_ts[si]) else np.nan
    title_a.set_text(f"Orbital Chaos  ·  {t_str}  "
                     f"·  f_chaos = {cf_val:.2f}")
    return [im_fl, cf_line] + list(ph_bars)


ani31 = animation.FuncAnimation(
    fig_a, _update_chaos_anim, frames=N_FRAMES_31,
    interval=1000 // ANIM_FPS_31, blit=True,
)
writer31 = animation.FFMpegWriter(
    fps=ANIM_FPS_31, bitrate=ANIM_BITRATE_31,
    metadata=dict(title="MW-M31 Orbital Chaos Animation"),
)
ani31.save(os.path.join(OUT_DIR, "section31_animation_chaos.mp4"),
           writer=writer31, dpi=ANIM_DPI_31)
plt.close(fig_a)
print("  Saved: section31_animation_chaos.mp4")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.8 — MASTER SUMMARY PANEL                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n[Summary]  Master summary panel …")

fig_s = plt.figure(figsize=(16, 10), facecolor=BG)
gs_s  = gridspec.GridSpec(2, 2, figure=fig_s,
                           hspace=0.38, wspace=0.32,
                           left=0.08, right=0.97,
                           top=0.93, bottom=0.07)

# (0,0) Chaos fraction.
ax_s00 = fig_s.add_subplot(gs_s[0, 0])
_ax(ax_s00, xlabel=time_label, ylabel="Chaos fraction",
    title="Fraction of Chaotic Particles")
valid_cf = np.isfinite(chaos_frac_ts)
if valid_cf.any():
    ax_s00.plot(time_arr[valid_cf], chaos_frac_ts[valid_cf],
                color="#e8673a", lw=2.0)
ax_s00.set_ylim(0, 1.05)

# (0,1) ⟨λ⟩(r_0, t) heatmap.
ax_s01 = fig_s.add_subplot(gs_s[0, 1])
im_s01 = ax_s01.imshow(
    np.clip(lambda_r_ts, -lam_max, lam_max).T,
    aspect="auto", origin="lower",
    extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
    cmap="seismic",
    norm=TwoSlopeNorm(vmin=-lam_max, vcenter=0.0, vmax=lam_max),
)
ax_s01.set_yscale("log")
_ax(ax_s01, xlabel=time_label, ylabel=r"$r_0$ [kpc]",
    title=r"$\langle\lambda\rangle(r_0,\,t)$")
fig_s.colorbar(im_s01, ax=ax_s01, shrink=0.8, label=r"$\lambda$")

# (1,0) λ vs. r_0 scatter.
ax_s10 = fig_s.add_subplot(gs_s[1, 0])
_ax(ax_s10, xlabel=r"$r_0$ [kpc]", ylabel=r"$\lambda$ [snap$^{-1}$]",
    title=r"FTLE vs. Initial Radius", log_x=True)
valid_lam = np.isfinite(lambda_total) & np.isfinite(_r0)
ax_s10.scatter(_r0[valid_lam], lambda_total[valid_lam],
               c=lambda_total[valid_lam], cmap="plasma",
               s=4, alpha=0.5, rasterized=True)
ax_s10.axhline(CHAOS_THRESH, color="#ff9944", lw=0.8, ls="--", alpha=0.7)
ax_s10.axhline(0, color="#555577", lw=0.7, ls=":")

# (1,1) 2D FTLE map at mid epoch.
ax_s11 = fig_s.add_subplot(gs_s[1, 1])
ax_s11.set_facecolor(BG)
mid_fi = n_ftle_maps // 2
F_mid  = ftle_maps[mid_fi]
F_s    = gaussian_filter(np.where(np.isfinite(F_mid), F_mid, 0.0), sigma=1.5)
ax_s11.imshow(F_s.T, origin="lower", aspect="equal",
              extent=[-FTLE_MAP_EXTENT, FTLE_MAP_EXTENT,
                      -FTLE_MAP_EXTENT, FTLE_MAP_EXTENT],
              cmap="inferno", vmin=0, vmax=vmax_a)
_ax(ax_s11, xlabel="x [kpc]", ylabel="y [kpc]",
    title="2D FTLE Field (mid epoch)")

fig_s.suptitle("Section 31 Summary  ·  Lyapunov Exponents & Orbital Chaos",
               fontsize=13, color="#c8c8e8", fontweight="bold")
fig_s.savefig(os.path.join(OUT_DIR, "section31_summary_panel.png"),
              dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig_s)
print("  Saved: section31_summary_panel.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §31.9 — SECTION COMPLETE                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 31 COMPLETE")
print("="*80)
outputs_31 = [
    "section31_lyapunov_trajectories.png",
    "section31_lambda_distribution.png",
    "section31_lambda_vs_r.png",
    "section31_chaos_fraction.png",
    "section31_lambda_heatmap.png",
    "section31_ftle_map.png",
    "section31_chaos_vs_stripping.png",
    "section31_stochastic_layer.png",
    "section31_animation_chaos.mp4",
    "section31_summary_panel.png",
]
for fn in outputs_31:
    fp   = os.path.join(OUT_DIR, fn)
    size = os.path.getsize(fp)/1e6 if os.path.isfile(fp) else 0.0
    kind = "animation" if fn.endswith(".mp4") else "figure"
    print(f"  {fn:<50} {size:6.2f} MB  [{kind}]")

# Print chaos statistics.
print(f"\n  CHAOS STATISTICS")
print(f"  {'Group':<20} {'N':>5} {'f_chaotic':>12} {'⟨λ⟩':>12}")
print(f"  {'-'*20} {'-'*5} {'-'*12} {'-'*12}")
for g, gname in enumerate(G_NAMES[:4]):
    mask  = (_group == g) & np.isfinite(lambda_total)
    f_c   = (lambda_total[mask] > CHAOS_THRESH).mean() if mask.sum() > 0 else np.nan
    lam_m = np.nanmean(lambda_total[mask]) if mask.sum() > 0 else np.nan
    print(f"  {gname:<20} {mask.sum():>5} {f_c:>12.3f} {lam_m:>12.4f}")
print("="*80)
