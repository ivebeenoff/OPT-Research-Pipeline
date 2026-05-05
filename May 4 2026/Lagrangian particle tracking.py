# “””

# SECTION 26 — LAGRANGIAN PARTICLE TRACKING & TRAJECTORY ANALYSIS

Author  : Abhinav Vatsa

Continuation of the MW–M31 analysis pipeline.  All globals (SNAPSHOTS, ns,
R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL,
G_KPC_KMS2_MSUN, PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr,
time_label, time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

## Physical motivation

Every diagnostic in Sections 21–25 is Eulerian — it asks “what is the
density/velocity/anisotropy at radius r at time t?”  This section takes the
complementary Lagrangian view: it tags individual particles at snapshot 0
and follows *the same particles* across all 801 snapshots.

This answers questions that are invisible to Eulerian statistics:

(A) MIGRATION  — which particles that started deep inside the MW disk
end up in the outer halo or tidal streams?  Which particles from the
outer halo of M31 fall into the MW core?

(B) MIXING  — what fraction of particles at a given final radius r_f
originated from the MW vs. M31?  How does this fraction vary
with r_f and with the merger stage?

(C) TRAJECTORY MORPHOLOGY  — do particles follow smooth inspiral paths,
or do they undergo chaotic scattering at pericentre?  The path length
and curvature of individual trajectories diagnoses the degree of
violent relaxation.

(D) RADIAL DISPLACEMENT  — the distribution P(Δr) = P(r_f − r_0) shows
whether the merger is inside-out (inner particles pushed outward) or
outside-in (outer particles pulled inward).

(E) PHASE-SPACE DISPLACEMENT  — tracking (r, v) together reveals whether
particles move along adiabatic invariants or cross phase-space
boundaries (non-adiabatic heating).

## Implementation strategy

N-body snapshot files do not guarantee particle ordering is preserved between
snapshots.  We identify the same particle across snapshots by matching on:

1. Particle ID  (if the snapshot format stores IDs — check first)
1. Nearest-neighbour matching in phase space (r, v) at consecutive steps
   (fallback if IDs are absent)

This section implements BOTH methods and selects automatically.

We track a SAMPLE of N_TRACK particles rather than all ~10^6, for two reasons:
• Memory: storing (801, N_ALL, 3) position arrays is ~20 GB for N_ALL=10^6
• Visualisation: individual trajectories are only meaningful for O(1000) particles

The sample is drawn in three groups:
• INNER sample   — particles starting at r_0 < 10 kpc  (disk/bulge region)
• MID sample     — particles starting at 10 < r_0 < 50 kpc  (inner halo)
• OUTER sample   — particles starting at r_0 > 50 kpc  (outer halo)
• M31 sample     — particles from M31 (origin tag = 1)

## Outputs

section26_displacement_hist.png      P(Δr) = P(r_f − r_0) distribution
section26_radial_migration.png       r(t) trajectories for sampled particles
section26_origin_map.png             Final position coloured by initial radius
section26_mixing_fraction.png        f_MW(r_f) and f_M31(r_f) at final snap
section26_path_length.png            Total path length per particle vs. r_0
section26_phasespace_tracks.png      (r, v_r) tracks for sampled particles
section26_displacement_heatmap.png   ⟨Δr⟩(r_0, t) heatmap
section26_trajectory_map.png         2D x-y trajectory spaghetti plot
section26_animation_tracks.mp4       Animated particle positions over time
section26_summary_panel.png          Master 4-panel summary

===============================================================================
“””

import numpy as np
import matplotlib
matplotlib.use(“Agg”)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.0 — CONFIGURATION                                                     ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# Number of particles to track in each spatial group.

N_TRACK_INNER =  300    # r_0 < R_INNER_KPC
N_TRACK_MID   =  300    # R_INNER_KPC < r_0 < R_MID_KPC
N_TRACK_OUTER =  300    # r_0 > R_MID_KPC
N_TRACK_M31   =  300    # origin == 1  (M31 particles)

# Radial boundaries that define the three spatial groups.

R_INNER_KPC   = 10.0
R_MID_KPC     = 50.0

# Phase-space matching fallback parameters (used when particle IDs unavailable).

# We match particles between consecutive snapshots using a KD-tree in 6D

# (x, y, z, vx, vy, vz) with rescaled velocity units.

# V_SCALE converts km/s to kpc so both position and velocity contribute

# comparably to the distance metric.  ~1 kpc per 100 km/s is typical.

V_SCALE_KPC_PER_KMS = 0.01   # [kpc / (km/s)]

# Maximum allowed 6D distance for a valid match.

# Particles with no neighbour within this distance are flagged as “lost”.

MATCH_MAX_DIST = 5.0   # [kpc, after velocity rescaling]

# Animation subsampling — render every Nth snapshot.

ANIM_STEP       = 8
ANIM_FPS_26     = 20
ANIM_DPI_26     = 100
ANIM_BITRATE_26 = 1800

# 2D trajectory map extent.

TRAJ_MAP_EXTENT = 400.0   # [kpc]

# Path length computation: subsample every Nth snapshot to save time.

PATHLENGTH_STEP = 4

print(”\n” + “=”*80)
print(”  SECTION 26 · Lagrangian Particle Tracking & Trajectory Analysis”)
print(”=”*80)
print(f”  Tracking {N_TRACK_INNER} inner + {N_TRACK_MID} mid + “
f”{N_TRACK_OUTER} outer + {N_TRACK_M31} M31 particles”)
print(f”  Total tracked : {N_TRACK_INNER+N_TRACK_MID+N_TRACK_OUTER+N_TRACK_M31}”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.1 — PARTICLE ID DETECTION                                             ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def detect_particle_ids(mw_file: str, m31_file: str) -> bool:
“””
Check whether the snapshot files store particle IDs.

```
Returns True if IDs are available and can be used for tracking;
False if we must fall back to phase-space nearest-neighbour matching.

The CenterOfMass class may expose particle IDs as self.id or similar.
We probe the MW object's attributes to detect this.
"""
try:
    MW = CenterOfMass(mw_file, PTYPE)
    has_id = hasattr(MW, "id") or hasattr(MW, "IDs") or hasattr(MW, "pids")
    if has_id:
        print("  [ID detection] Particle IDs found — using ID-based tracking.")
    else:
        print("  [ID detection] No particle IDs — using phase-space KD-tree matching.")
    return has_id
except Exception:
    return False
```

def get_particle_ids(MW_obj, M31_obj) -> np.ndarray | None:
“””
Extract concatenated particle IDs from MW and M31 CenterOfMass objects.
Returns None if IDs are not available.
“””
for attr in (“id”, “IDs”, “pids”, “particle_ids”):
if hasattr(MW_obj, attr):
ids_mw  = getattr(MW_obj,  attr)
ids_m31 = getattr(M31_obj, attr)
return np.concatenate((ids_mw, ids_m31))
return None

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.2 — SNAPSHOT 0: SELECT TRACKED PARTICLES                             ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n  Loading snapshot 0 to select tracked particles …”)

snap0_num = SNAPSHOTS[0]
mw0_file  = os.path.join(tmpdir, f”MW_{snap0_num:03d}.txt”)
m31_0_file = os.path.join(tmpdir, f”M31_{snap0_num:03d}.txt”)

if not (os.path.isfile(mw0_file) and os.path.isfile(m31_0_file)):
raise FileNotFoundError(
f”Snapshot 0 files not found: {mw0_file}, {m31_0_file}\n”
“Ensure the tar archive has been extracted to tmpdir.”
)

snap0_data = load_snapshot_particles(mw0_file, m31_0_file)
MW0_obj    = CenterOfMass(mw0_file,  PTYPE)
M31_0_obj  = CenterOfMass(m31_0_file, PTYPE)

pos0    = snap0_data[“pos”]      # (N, 3)  COM-centred  [kpc]
m0      = snap0_data[“m_msun”]   # (N,)    [M_sun]
origin0 = snap0_data[“origin”]   # 0=MW, 1=M31
r0      = np.linalg.norm(pos0, axis=1)   # initial 3D radius

N_total = len(pos0)
print(f”  Snapshot 0: {N_total:,} particles”)

# ── Detect particle ID availability ───────────────────────────────────────────

USE_IDS    = detect_particle_ids(mw0_file, m31_0_file)
ids0       = get_particle_ids(MW0_obj, M31_0_obj) if USE_IDS else None

# ── Select tracked particle indices in snapshot 0 ─────────────────────────────

rng = np.random.default_rng(seed=42)   # reproducible sampling

def sample_from_mask(mask: np.ndarray, n: int) -> np.ndarray:
“”“Draw up to n random indices from where mask is True.”””
candidates = np.where(mask)[0]
if len(candidates) == 0:
return np.array([], dtype=int)
n_draw = min(n, len(candidates))
return rng.choice(candidates, size=n_draw, replace=False)

idx_inner = sample_from_mask(r0 < R_INNER_KPC,                          N_TRACK_INNER)
idx_mid   = sample_from_mask((r0 >= R_INNER_KPC) & (r0 < R_MID_KPC),   N_TRACK_MID)
idx_outer = sample_from_mask(r0 >= R_MID_KPC,                           N_TRACK_OUTER)
idx_m31   = sample_from_mask(origin0 == 1,                               N_TRACK_M31)

# Combine all tracked indices; remove duplicates; record group membership.

tracked_idx0 = np.unique(np.concatenate([idx_inner, idx_mid,
idx_outer, idx_m31]))
N_TRACKED    = len(tracked_idx0)
print(f”  Tracking {N_TRACKED} unique particles “
f”(inner={len(idx_inner)}, mid={len(idx_mid)}, “
f”outer={len(idx_outer)}, M31={len(idx_m31)})”)

# Group labels for each tracked particle: 0=inner, 1=mid, 2=outer, 3=M31.

group_label = np.full(N_TRACKED, -1, dtype=int)
for g, idx_g in enumerate([idx_inner, idx_mid, idx_outer, idx_m31]):
for k, gi in enumerate(tracked_idx0):
if gi in idx_g:
group_label[k] = g

GROUP_COLORS = [”#ff5566”, “#ffaa44”, “#4a8fff”, “#aa55ff”]
GROUP_NAMES  = [f”Inner (r₀ < {R_INNER_KPC:.0f} kpc)”,
f”Mid ({R_INNER_KPC:.0f}–{R_MID_KPC:.0f} kpc)”,
f”Outer (r₀ > {R_MID_KPC:.0f} kpc)”,
“M31 origin”]

# Store initial conditions for the tracked particles.

r0_tracked   = r0     [tracked_idx0]   # initial radius [kpc]
pos0_tracked = pos0   [tracked_idx0]   # initial position (3,)
m0_tracked   = m0     [tracked_idx0]   # mass [M_sun]
orig_tracked = origin0[tracked_idx0]   # 0=MW, 1=M31
ids0_tracked = ids0   [tracked_idx0] if ids0 is not None else None

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.3 — TRAJECTORY STORAGE                                                ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# Full position trajectories: shape (ns, N_TRACKED, 3)

# This is the main data structure.  For 800 snaps × 1200 particles × 3 floats

# × 8 bytes = ~23 MB — comfortably fits in RAM.

traj_pos = np.full((ns, N_TRACKED, 3), np.nan)   # [kpc]
traj_vel = np.full((ns, N_TRACKED, 3), np.nan)   # [km/s]
traj_r   = np.full((ns, N_TRACKED),   np.nan)    # 3D radius [kpc]
traj_vr  = np.full((ns, N_TRACKED),   np.nan)    # radial velocity [km/s]

# Store snapshot-0 positions directly.

traj_pos[0, :, :] = pos0_tracked
traj_r  [0, :]    = r0_tracked

print(f”\n[Storage]  traj_pos : {traj_pos.shape}  “
f”({traj_pos.nbytes/1e6:.1f} MB)”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.4 — MAIN TRACKING LOOP                                                ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# For each snapshot after snap 0:

# IF particle IDs are available:

# — load the snapshot, extract positions of particles matching ids0_tracked

# ELSE (phase-space nearest-neighbour fallback):

# — build a 6D KD-tree of all particles in the new snapshot

# — find the nearest neighbour in (x,y,z, vx*V_SCALE, vy*V_SCALE, vz*V_SCALE)

# to each tracked particle’s position at the PREVIOUS snapshot

# — flag particles where the match distance exceeds MATCH_MAX_DIST as “lost”

print(”\n” + “=”*80)
print(”  §26.4 — Main Tracking Loop”)
print(”=”*80)

# Track cumulative path length per particle.

path_length = np.zeros(N_TRACKED)   # [kpc]

t_loop_start = time.perf_counter()

# Previous snapshot’s 6D state (for KD-tree matching).

prev_pos6d = np.zeros((N_TRACKED, 6))
prev_pos6d[:, :3] = pos0_tracked

for i, snap_num in enumerate(SNAPSHOTS):

```
if i == 0:
    continue   # snap 0 already stored

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

pos_all = snap_data["pos"]
m_all   = snap_data["m_msun"]

# COM-frame velocities.
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
    wi   = m_all[inn]
    vxcom = np.sum(wi*vx_all[inn])/wi.sum()
    vycom = np.sum(wi*vy_all[inn])/wi.sum()
    vzcom = np.sum(wi*vz_all[inn])/wi.sum()
else:
    vxcom = vycom = vzcom = 0.0

vel_all = np.vstack((vx_all-vxcom, vy_all-vycom, vz_all-vzcom)).T

# ── ID-based tracking ─────────────────────────────────────────────────────
if USE_IDS:
    ids_now = get_particle_ids(MW_obj, M31_obj)
    if ids_now is not None:
        # Build a dict from ID → index in this snapshot.
        id_to_idx = {pid: k for k, pid in enumerate(ids_now)}
        for t_idx, pid in enumerate(ids0_tracked):
            k = id_to_idx.get(pid, None)
            if k is not None:
                traj_pos[i, t_idx, :] = pos_all[k]
                traj_vel[i, t_idx, :] = vel_all[k]
        # Fall through to compute r and vr below.

# ── Phase-space KD-tree fallback ──────────────────────────────────────────
else:
    # Build 6D feature array: (x, y, z, vx*V_SCALE, vy*V_SCALE, vz*V_SCALE)
    pos6d_all = np.hstack([
        pos_all,
        vel_all * V_SCALE_KPC_PER_KMS
    ])   # (N_total, 6)

    # Query tree: for each tracked particle, find its nearest neighbour.
    tree = cKDTree(pos6d_all)
    dists, nn_idx = tree.query(prev_pos6d, k=1, workers=-1)

    for t_idx in range(N_TRACKED):
        if dists[t_idx] > MATCH_MAX_DIST:
            # Particle is "lost" — leave NaN, don't update prev_pos6d.
            continue
        k = nn_idx[t_idx]
        traj_pos[i, t_idx, :] = pos_all[k]
        traj_vel[i, t_idx, :] = vel_all[k]
        # Update previous state for next step's KD-tree query.
        prev_pos6d[t_idx, :3] = pos_all[k]
        prev_pos6d[t_idx, 3:] = vel_all[k] * V_SCALE_KPC_PER_KMS

# ── Compute scalar radial quantities ──────────────────────────────────────
pos_t = traj_pos[i]   # (N_TRACKED, 3)
vel_t = traj_vel[i]

r_t   = np.linalg.norm(pos_t, axis=1)   # (N_TRACKED,)
traj_r[i] = r_t

# Radial velocity v_r = v · r̂
with np.errstate(divide="ignore", invalid="ignore"):
    r_hat = np.where(r_t[:,None] > 0, pos_t/r_t[:,None], 0.0)
v_r_t = np.einsum("ij,ij->i", vel_t, r_hat)
traj_vr[i] = v_r_t

# ── Cumulative path length (every PATHLENGTH_STEP snapshots) ──────────────
if i % PATHLENGTH_STEP == 0 and i > 0:
    prev_i = max(0, i - PATHLENGTH_STEP)
    step_disp = np.linalg.norm(
        traj_pos[i] - traj_pos[prev_i], axis=1
    )
    finite = np.isfinite(step_disp)
    path_length[finite] += step_disp[finite]

if (i + 1) % 100 == 0:
    elapsed = time.perf_counter() - t_loop_start
    n_lost  = np.sum(~np.isfinite(traj_r[i]))
    print(f"  snap {snap_num:04d}  lost={n_lost}/{N_TRACKED}  [{elapsed:.0f}s]")
```

print(f”\n[Tracking done]  {time.perf_counter()-t_loop_start:.0f}s total”)

# ── Final state ───────────────────────────────────────────────────────────────

last_valid_snap = ns - 1
while last_valid_snap > 0 and not np.isfinite(traj_r[last_valid_snap]).any():
last_valid_snap -= 1

r_final     = traj_r[last_valid_snap]      # final 3D radius per particle [kpc]
pos_final   = traj_pos[last_valid_snap]    # final position (N_TRACKED, 3)
delta_r     = r_final - r0_tracked         # radial displacement Δr [kpc]

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.5 — FIGURES                                                           ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

BG    = “#0d0d18”
MUTED = “#7070a0”

def _ax(ax, xlabel=””, ylabel=””, title=””, log_x=False, log_y=False):
ax.set_facecolor(BG)
for sp in ax.spines.values():
sp.set_edgecolor(”#2a2a4a”)
ax.tick_params(colors=”#9090b0”, labelsize=8)
ax.set_xlabel(xlabel, fontsize=9,  color=”#c8c8e8”)
ax.set_ylabel(ylabel, fontsize=9,  color=”#c8c8e8”)
ax.set_title(title,   fontsize=10, color=”#c8c8e8”, pad=5)
if log_x: ax.set_xscale(“log”)
if log_y: ax.set_yscale(“log”)
return ax

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 1 — RADIAL DISPLACEMENT DISTRIBUTION P(Δr)

# ══════════════════════════════════════════════════════════════════════════════

# 

# P(Δr) = P(r_f − r_0) shows the net radial migration of DM particles

# over the full merger.  A symmetric distribution centred near zero would

# indicate no net migration.  A positive tail (Δr > 0) means the merger

# has pushed mass outward.  A bimodal distribution reveals two populations:

# particles that stayed put (virialised remnant) and particles that were

# ejected to large radii (tidal streams).

print(”\n[Fig 1]  Radial displacement distribution …”)

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG,
gridspec_kw={“wspace”: 0.32})

# (a) All particles together.

ax = axes1[0]
_ax(ax, xlabel=r”$\Delta r = r_f - r_0$  [kpc]”,
ylabel=“Count”, title=r”All Tracked Particles  $P(\Delta r)$”)
valid_dr = np.isfinite(delta_r)
ax.hist(delta_r[valid_dr], bins=60, color=”#4a8fff”, alpha=0.8,
edgecolor=“none”)
ax.axvline(0, color=”#ffffff”, lw=0.8, ls=”–”, alpha=0.5)
ax.axvline(np.nanmedian(delta_r), color=”#ff9944”, lw=1.5,
label=f”Median Δr = {np.nanmedian(delta_r):.1f} kpc”)
ax.legend(fontsize=8)

# (b) By initial group.

ax = axes1[1]
_ax(ax, xlabel=r”$\Delta r$  [kpc]”,
ylabel=“Density”, title=r”$P(\Delta r)$ by Initial Radius Group”)
for g, (gname, gcol) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
mask = group_label == g
dr_g = delta_r[mask & valid_dr]
if len(dr_g) > 5:
ax.hist(dr_g, bins=40, density=True, alpha=0.5,
color=gcol, edgecolor=“none”, label=gname)
ax.axvline(0, color=”#ffffff”, lw=0.8, ls=”–”, alpha=0.5)
ax.legend(fontsize=7)

# (c) Scatter r_0 vs. r_f.

ax = axes1[2]
_ax(ax, xlabel=r”$r_0$  [kpc] (initial)”,
ylabel=r”$r_f$  [kpc] (final)”,
title=r”Initial vs. Final Radius”, log_x=True, log_y=True)
for g, gcol in enumerate(GROUP_COLORS):
mask = group_label == g
r0_g  = r0_tracked[mask & valid_dr]
rf_g  = r_final   [mask & valid_dr]
ax.scatter(r0_g, rf_g, s=3, color=gcol, alpha=0.5, rasterized=True)

# 1:1 line.

lim = (R_BINS[0], R_BINS[-1])
ax.plot(lim, lim, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.4)
ax.set_xlim(*lim); ax.set_ylim(*lim)

fig1.savefig(os.path.join(OUT_DIR, “section26_displacement_hist.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig1)
print(”  Saved: section26_displacement_hist.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 2 — r(t) TRAJECTORIES FOR SAMPLED PARTICLES

# ══════════════════════════════════════════════════════════════════════════════

# 

# Plotting r(t) for each tracked particle reveals the character of individual

# orbits: smooth inspiral (virialising) vs. large oscillations (chaotic

# scattering) vs. monotonic outward drift (tidal ejection).

print(”[Fig 2]  r(t) trajectory lines …”)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG,
gridspec_kw={“hspace”: 0.32, “wspace”: 0.28})
axes2 = axes2.flatten()

for g, (gname, gcol) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
ax = axes2[g]
_ax(ax, xlabel=time_label, ylabel=“r [kpc]”,
title=gname, log_y=True)

```
mask = group_label == g
g_indices = np.where(mask)[0]

for t_idx in g_indices[:50]:   # show up to 50 trajectories per group
    r_track = traj_r[:, t_idx]
    valid   = np.isfinite(r_track)
    if valid.sum() < 5:
        continue
    ax.plot(time_arr[valid], r_track[valid],
            color=gcol, lw=0.6, alpha=0.25)

# Overlay the median trajectory.
r_group = traj_r[:, mask]
r_med   = np.nanmedian(r_group, axis=1)
valid   = np.isfinite(r_med)
ax.plot(time_arr[valid], r_med[valid],
        color="white", lw=2.0, label="Median", zorder=5)
ax.legend(fontsize=8)
ax.set_ylim(0.1, TRAJ_MAP_EXTENT)
```

fig2.suptitle(“Radial Trajectories r(t) by Initial Position Group”,
fontsize=12, color=”#c8c8e8”)
fig2.savefig(os.path.join(OUT_DIR, “section26_radial_migration.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig2)
print(”  Saved: section26_radial_migration.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 3 — 2D FINAL POSITION MAP COLOURED BY INITIAL RADIUS

# ══════════════════════════════════════════════════════════════════════════════

# 

# Colouring particles by r_0 on a plot of their FINAL (x,y) positions shows

# where each initial shell ends up.  Particles from the innermost regions

# should cluster near the final remnant core; outer particles spread into

# tidal tails and streams.

print(”[Fig 3]  Origin map …”)

fig3, ax3 = plt.subplots(figsize=(8, 7), facecolor=BG)
_ax(ax3, xlabel=“x [kpc]”, ylabel=“y [kpc]”,
title=“Final Positions Coloured by Initial Radius $r_0$”)

valid_f = np.isfinite(r_final) & np.isfinite(r0_tracked)
sc = ax3.scatter(pos_final[valid_f, 0], pos_final[valid_f, 1],
c=np.log10(r0_tracked[valid_f]),
cmap=“plasma”, s=3, alpha=0.6, rasterized=True,
norm=Normalize(vmin=np.log10(0.1),
vmax=np.log10(TRAJ_MAP_EXTENT)))
cb = fig3.colorbar(sc, ax=ax3, pad=0.01)
cb.set_label(r”$\log_{10}(r_0)$ [kpc]”, fontsize=8)
ax3.set_xlim(-TRAJ_MAP_EXTENT, TRAJ_MAP_EXTENT)
ax3.set_ylim(-TRAJ_MAP_EXTENT, TRAJ_MAP_EXTENT)

fig3.savefig(os.path.join(OUT_DIR, “section26_origin_map.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig3)
print(”  Saved: section26_origin_map.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 4 — MW/M31 MIXING FRACTION AT FINAL SNAPSHOT

# ══════════════════════════════════════════════════════════════════════════════

# 

# Binning tracked particles by their FINAL radius and computing the fraction

# that originated from MW vs. M31 gives the mass-mixing profile at the

# end of the simulation.  This is the Lagrangian counterpart of f_mix(r)

# from the density pipeline (§29), and the two should agree if the tracking

# is accurate.

print(”[Fig 4]  MW/M31 mixing fraction …”)

fig4, ax4 = plt.subplots(figsize=(9, 5), facecolor=BG)
_ax(ax4, xlabel=r”Final radius $r_f$ [kpc]”,
ylabel=“Mass fraction”,
title=“MW vs. M31 Origin Fraction at Final Snapshot”,
log_x=True)

# Bin tracked particles by final radius.

bin_id_final = np.digitize(r_final, R_BINS) - 1
f_mw_final   = np.full(nb_sph, np.nan)
f_m31_final  = np.full(nb_sph, np.nan)

for b in range(nb_sph):
mask = (bin_id_final == b) & valid_f
if mask.sum() < 5:
continue
M_bin     = m0_tracked[mask].sum()
M_mw_bin  = m0_tracked[mask & (orig_tracked == 0)].sum()
M_m31_bin = m0_tracked[mask & (orig_tracked == 1)].sum()
f_mw_final [b] = M_mw_bin  / (M_bin + 1e-30)
f_m31_final[b] = M_m31_bin / (M_bin + 1e-30)

valid_frac = np.isfinite(f_mw_final)
ax4.plot(r_mid_sph[valid_frac], f_mw_final[valid_frac],
color=”#4a8fff”, lw=2.0, label=“MW origin”)
ax4.plot(r_mid_sph[valid_frac], f_m31_final[valid_frac],
color=”#ff5fa0”, lw=2.0, label=“M31 origin”)
ax4.axhline(0.5, color=”#555577”, lw=0.7, ls=”–”, alpha=0.5)
ax4.fill_between(r_mid_sph[valid_frac],
f_mw_final[valid_frac], 0.5,
where=f_mw_final[valid_frac] > 0.5,
alpha=0.08, color=”#4a8fff”)
ax4.set_ylim(0, 1.0)
ax4.set_xlim(R_BINS[0], R_BINS[-1])
ax4.legend(fontsize=8)

fig4.savefig(os.path.join(OUT_DIR, “section26_mixing_fraction.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig4)
print(”  Saved: section26_mixing_fraction.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 5 — PATH LENGTH vs. INITIAL RADIUS

# ══════════════════════════════════════════════════════════════════════════════

# 

# The total path length L = ∫|dr/dt|dt of each particle’s trajectory measures

# how far it has physically travelled through configuration space.  Particles

# with large L relative to their initial radius have undergone chaotic orbits

# or been dragged through tidal streams.  Particles with small L have stayed

# on nearly fixed orbits (adiabatic evolution).

print(”[Fig 5]  Path length vs. initial radius …”)

fig5, ax5 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax5, xlabel=r”Initial radius $r_0$ [kpc]”,
ylabel=r”Total path length $L$ [kpc]”,
title=“Particle Path Length vs. Initial Radius”,
log_x=True, log_y=True)

for g, gcol in enumerate(GROUP_COLORS):
mask = (group_label == g) & np.isfinite(r0_tracked) & (path_length > 0)
ax5.scatter(r0_tracked[mask], path_length[mask],
s=4, color=gcol, alpha=0.5, rasterized=True,
label=GROUP_NAMES[g])

# 1:1 line: path length = initial radius (reference for minimal migration).

r_ref = np.logspace(np.log10(0.1), np.log10(TRAJ_MAP_EXTENT), 100)
ax5.plot(r_ref, r_ref, color=”#ffffff”, lw=0.7, ls=”–”,
alpha=0.3, label=“L = r₀”)
ax5.set_xlim(0.1, TRAJ_MAP_EXTENT)
ax5.legend(fontsize=7)

fig5.savefig(os.path.join(OUT_DIR, “section26_path_length.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5)
print(”  Saved: section26_path_length.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 6 — (r, v_r) PHASE-SPACE TRACKS

# ══════════════════════════════════════════════════════════════════════════════

# 

# Plotting the trajectory of each particle in (r, v_r) phase space traces out

# the orbital loops.  A bound orbit with no energy dissipation would appear

# as a closed loop.  Energy loss (dynamical friction) causes the loop to

# spiral inward.  Tidal ejection sends the particle on an open, outward track.

print(”[Fig 6]  (r, v_r) phase-space tracks …”)

fig6, axes6 = plt.subplots(1, 4, figsize=(18, 5), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.06})

for g, (gname, gcol) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
ax = axes6[g]
_ax(ax, xlabel=“r [kpc]”, title=gname, log_x=True)
if g == 0:
ax.set_ylabel(r”$v_r$ [km s$^{-1}$]”, fontsize=9)

```
mask = group_label == g
g_indices = np.where(mask)[0]

for t_idx in g_indices[:30]:
    r_track  = traj_r[:, t_idx]
    vr_track = traj_vr[:, t_idx]
    valid    = np.isfinite(r_track) & np.isfinite(vr_track)
    if valid.sum() < 5:
        continue
    # Colour the track by time.
    t_norm_track = (np.arange(ns)[valid] / ns)
    ax.scatter(r_track[valid], vr_track[valid],
               c=t_norm_track, cmap="plasma",
               s=1.5, alpha=0.5, rasterized=True)

ax.axhline(0, color="#555577", lw=0.5, ls="--")
ax.set_xlim(0.1, TRAJ_MAP_EXTENT)
ax.set_ylim(-600, 600)
```

fig6.suptitle(r”$(r,,v_r)$ Phase-Space Tracks  (colour = time, early→late)”,
fontsize=11, color=”#c8c8e8”)
fig6.savefig(os.path.join(OUT_DIR, “section26_phasespace_tracks.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig6)
print(”  Saved: section26_phasespace_tracks.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 7 — ⟨Δr⟩(r_0, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# The mean radial displacement ⟨Δr⟩(r_0, t) = ⟨r(t) − r_0⟩ as a function

# of both initial radius r_0 and time t.  This is the Lagrangian analogue of

# the tidal heating heatmap in §22.  Horizontal bands of positive ⟨Δr⟩ at

# certain r_0 mark shells that are being pushed outward by the tidal field.

print(”[Fig 7]  ⟨Δr⟩(r_0, t) heatmap …”)

# Bin particles by their initial radius r_0 into R_BINS.

bin_id_r0 = np.digitize(r0_tracked, R_BINS) - 1

# Mean Δr per (initial-radius bin, time) — shape (ns, nb_sph).

delta_r_mean_ts = np.full((ns, nb_sph), np.nan)

for i in range(ns):
for b in range(nb_sph):
mask = (bin_id_r0 == b) & np.isfinite(traj_r[i])
if mask.sum() < 3:
continue
delta_r_ts_b = traj_r[i, mask] - r0_tracked[mask]
delta_r_mean_ts[i, b] = np.mean(delta_r_ts_b)

dr_max = np.nanpercentile(np.abs(delta_r_mean_ts[np.isfinite(delta_r_mean_ts)]), 97)

fig7, ax7 = plt.subplots(figsize=(12, 5), facecolor=BG)
from matplotlib.colors import TwoSlopeNorm
im7 = ax7.imshow(
delta_r_mean_ts.T,
aspect=“auto”, origin=“lower”,
extent=[np.nanmin(time_arr), np.nanmax(time_arr), R_BINS[0], R_BINS[-1]],
cmap=“seismic”,
norm=TwoSlopeNorm(vmin=-dr_max, vcenter=0.0, vmax=dr_max),
)
ax7.set_yscale(“log”)
_ax(ax7, xlabel=time_label, ylabel=r”Initial radius $r_0$ [kpc]”,
title=r”Mean Radial Displacement  $\langle \Delta r \rangle(r_0,,t)$  “
r”[kpc]  — red = outward, blue = inward”)
cb7 = fig7.colorbar(im7, ax=ax7, pad=0.01)
cb7.set_label(r”$\langle\Delta r\rangle$ [kpc]”, fontsize=8)

fig7.savefig(os.path.join(OUT_DIR, “section26_displacement_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig7)
print(”  Saved: section26_displacement_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 8 — 2D x-y TRAJECTORY SPAGHETTI PLOT

# ══════════════════════════════════════════════════════════════════════════════

# 

# Drawing the full x-y paths of a subsample of particles across all snapshots

# produces the iconic “spaghetti plot” that visually captures the chaotic

# complexity of the merger.  Lines are coloured by group to show which

# initial populations contribute to the visible tidal structures.

print(”[Fig 8]  2D trajectory spaghetti plot …”)

fig8, ax8 = plt.subplots(figsize=(9, 9), facecolor=BG)
_ax(ax8, xlabel=“x [kpc]”, ylabel=“y [kpc]”,
title=“Particle Trajectories  (x–y plane)”)

for g, gcol in enumerate(GROUP_COLORS):
mask = group_label == g
g_indices = np.where(mask)[0]

```
# Show up to 80 trajectories per group.
for t_idx in g_indices[:80]:
    x_track = traj_pos[:, t_idx, 0]
    y_track = traj_pos[:, t_idx, 1]
    valid   = np.isfinite(x_track) & np.isfinite(y_track)
    if valid.sum() < 5:
        continue
    ax8.plot(x_track[valid], y_track[valid],
             color=gcol, lw=0.4, alpha=0.18)

# Mark final positions.
valid_f_g = mask & valid_f
ax8.scatter(pos_final[valid_f_g, 0], pos_final[valid_f_g, 1],
            s=6, color=gcol, alpha=0.6, zorder=4, rasterized=True,
            label=GROUP_NAMES[g])
```

ax8.set_xlim(-TRAJ_MAP_EXTENT, TRAJ_MAP_EXTENT)
ax8.set_ylim(-TRAJ_MAP_EXTENT, TRAJ_MAP_EXTENT)
ax8.legend(fontsize=7, loc=“upper right”)

fig8.savefig(os.path.join(OUT_DIR, “section26_trajectory_map.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig8)
print(”  Saved: section26_trajectory_map.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.6 — ANIMATION                                                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Two-panel animation:

# Left  : 2D (x,y) positions of tracked particles, coloured by group,

# with tails showing the last N_TAIL steps.

# Right : r(t) running median per group, accumulated as the animation plays.

print(”\n[Anim]  Particle tracking animation …”)

N_TAIL      = 15    # number of past positions to show as a fading tail
ANIM_IDXS   = np.arange(0, ns, ANIM_STEP)
N_FRAMES    = len(ANIM_IDXS)

fig_a, (ax_xy, ax_rt) = plt.subplots(
1, 2, figsize=(14, 6), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 2], “wspace”: 0.1},
)
ax_xy.set_facecolor(BG)
ax_rt.set_facecolor(BG)
ax_xy.set_xlim(-TRAJ_MAP_EXTENT, TRAJ_MAP_EXTENT)
ax_xy.set_ylim(-TRAJ_MAP_EXTENT, TRAJ_MAP_EXTENT)
ax_xy.set_xlabel(“x [kpc]”, color=”#c8c8e8”)
ax_xy.set_ylabel(“y [kpc]”, color=”#c8c8e8”)

ax_rt.set_xlim(np.nanmin(time_arr), np.nanmax(time_arr))
ax_rt.set_ylim(0.1, TRAJ_MAP_EXTENT)
ax_rt.set_yscale(“log”)
ax_rt.set_xlabel(time_label, color=”#c8c8e8”)
ax_rt.set_ylabel(“Median r [kpc]”, color=”#c8c8e8”)
ax_rt.set_title(“Median r(t) per group”, color=”#c8c8e8”, fontsize=10)

# Create scatter objects and tail line collections per group.

scats = []
tail_lines = {g: [] for g in range(4)}
rt_lines   = []

for g, gcol in enumerate(GROUP_COLORS):
sc = ax_xy.scatter([], [], s=4, color=gcol, alpha=0.7,
rasterized=True, label=GROUP_NAMES[g])
scats.append(sc)
for _ in range(N_TAIL):
ln, = ax_xy.plot([], [], color=gcol, lw=0.5, alpha=0.1)
tail_lines[g].append(ln)
rl, = ax_rt.plot([], [], color=gcol, lw=1.8, label=GROUP_NAMES[g])
rt_lines.append(rl)

ax_xy.legend(fontsize=7, loc=“upper right”)
ax_rt.legend(fontsize=7)
title_a = fig_a.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_track_anim(frame_idx):
snap_i = ANIM_IDXS[frame_idx]
t_val  = time_arr[snap_i]
t_str  = f”{t_val:.2f} Gyr” if (np.isfinite(t_val) and time_is_gyr)   
else f”Snap {SNAPSHOTS[snap_i]}”
title_a.set_text(f”Lagrangian Tracking  ·  {t_str}”)

```
artists = []

for g, gcol in enumerate(GROUP_COLORS):
    mask     = group_label == g
    x_now    = traj_pos[snap_i, mask, 0]
    y_now    = traj_pos[snap_i, mask, 1]
    valid_g  = np.isfinite(x_now) & np.isfinite(y_now)

    scats[g].set_offsets(
        np.column_stack((x_now[valid_g], y_now[valid_g]))
        if valid_g.any() else np.empty((0, 2))
    )
    artists.append(scats[g])

    # Tail lines.
    for tail_idx, ln in enumerate(tail_lines[g]):
        past_snap_idx = frame_idx - (N_TAIL - tail_idx)
        if past_snap_idx < 0:
            ln.set_data([], [])
        else:
            past_i = ANIM_IDXS[past_snap_idx]
            x_p    = traj_pos[past_i, mask, 0]
            y_p    = traj_pos[past_i, mask, 1]
            valid_p = np.isfinite(x_p) & np.isfinite(y_p)
            if valid_p.any():
                ln.set_data(x_p[valid_p], y_p[valid_p])
                ln.set_alpha(0.04 + 0.04 * tail_idx)
            else:
                ln.set_data([], [])
        artists.append(ln)

    # r(t) running median.
    r_g_hist = traj_r[:snap_i+1, mask]
    r_med    = np.nanmedian(r_g_hist, axis=1)
    valid_r  = np.isfinite(r_med) & np.isfinite(time_arr[:snap_i+1])
    rt_lines[g].set_data(time_arr[:snap_i+1][valid_r],
                          r_med[valid_r])
    artists.append(rt_lines[g])

return artists
```

ani_t = animation.FuncAnimation(
fig_a, _update_track_anim, frames=N_FRAMES,
interval=1000 // ANIM_FPS_26, blit=True,
)
writer_26 = animation.FFMpegWriter(
fps=ANIM_FPS_26, bitrate=ANIM_BITRATE_26,
metadata=dict(title=“MW-M31 Lagrangian Particle Tracking”),
)
ani_t.save(os.path.join(OUT_DIR, “section26_animation_tracks.mp4”),
writer=writer_26, dpi=ANIM_DPI_26)
plt.close(fig_a)
print(”  Saved: section26_animation_tracks.mp4”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.7 — MASTER SUMMARY PANEL                                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n[Summary]  Master summary panel …”)

fig_s = plt.figure(figsize=(16, 10), facecolor=BG)
gs_s  = gridspec.GridSpec(2, 2, figure=fig_s,
hspace=0.38, wspace=0.32,
left=0.08, right=0.97,
top=0.93, bottom=0.07)

# (0,0) r_0 vs. r_f scatter.

ax_s00 = fig_s.add_subplot(gs_s[0, 0])
_ax(ax_s00, xlabel=r”$r_0$ [kpc]”, ylabel=r”$r_f$ [kpc]”,
title=“Initial vs. Final Radius”, log_x=True, log_y=True)
for g, gcol in enumerate(GROUP_COLORS):
mask = (group_label == g) & valid_f
ax_s00.scatter(r0_tracked[mask], r_final[mask],
s=3, color=gcol, alpha=0.4, rasterized=True)
ax_s00.plot([0.1, TRAJ_MAP_EXTENT], [0.1, TRAJ_MAP_EXTENT],
color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.3)

# (0,1) Δr distribution.

ax_s01 = fig_s.add_subplot(gs_s[0, 1])
_ax(ax_s01, xlabel=r”$\Delta r$ [kpc]”, ylabel=“Count”,
title=r”Radial Displacement $P(\Delta r)$”)
ax_s01.hist(delta_r[valid_dr], bins=50, color=”#4a8fff”, alpha=0.8, edgecolor=“none”)
ax_s01.axvline(0, color=”#ffffff”, lw=0.8, ls=”–”, alpha=0.5)
ax_s01.axvline(np.nanmedian(delta_r), color=”#ff9944”, lw=1.5,
label=f”Median = {np.nanmedian(delta_r):.1f} kpc”)
ax_s01.legend(fontsize=8)

# (1,0) ⟨Δr⟩(r_0,t) heatmap.

ax_s10 = fig_s.add_subplot(gs_s[1, 0])
im_s10 = ax_s10.imshow(
delta_r_mean_ts.T, aspect=“auto”, origin=“lower”,
extent=[np.nanmin(time_arr), np.nanmax(time_arr), R_BINS[0], R_BINS[-1]],
cmap=“seismic”,
norm=TwoSlopeNorm(vmin=-dr_max, vcenter=0.0, vmax=dr_max),
)
ax_s10.set_yscale(“log”)
_ax(ax_s10, xlabel=time_label, ylabel=r”$r_0$ [kpc]”,
title=r”$\langle\Delta r\rangle(r_0,t)$”)
fig_s.colorbar(im_s10, ax=ax_s10, shrink=0.8, label=”⟨Δr⟩ [kpc]”)

# (1,1) MW/M31 mixing.

ax_s11 = fig_s.add_subplot(gs_s[1, 1])
_ax(ax_s11, xlabel=r”$r_f$ [kpc]”, ylabel=“Mass fraction”,
title=“Origin Mixing at Final Snapshot”, log_x=True)
if np.isfinite(f_mw_final).any():
ax_s11.plot(r_mid_sph[valid_frac], f_mw_final [valid_frac],
color=”#4a8fff”, lw=1.8, label=“MW origin”)
ax_s11.plot(r_mid_sph[valid_frac], f_m31_final[valid_frac],
color=”#ff5fa0”, lw=1.8, label=“M31 origin”)
ax_s11.axhline(0.5, color=”#555577”, lw=0.7, ls=”–”)
ax_s11.set_ylim(0, 1)
ax_s11.legend(fontsize=8)

fig_s.suptitle(“Section 26 Summary  ·  Lagrangian Particle Tracking”,
fontsize=13, color=”#c8c8e8”, fontweight=“bold”)
fig_s.savefig(os.path.join(OUT_DIR, “section26_summary_panel.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig_s)
print(”  Saved: section26_summary_panel.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §26.8 — SECTION COMPLETE                                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  SECTION 26 COMPLETE”)
print(”=”*80)
outputs_26 = [
“section26_displacement_hist.png”,
“section26_radial_migration.png”,
“section26_origin_map.png”,
“section26_mixing_fraction.png”,
“section26_path_length.png”,
“section26_phasespace_tracks.png”,
“section26_displacement_heatmap.png”,
“section26_trajectory_map.png”,
“section26_animation_tracks.mp4”,
“section26_summary_panel.png”,
]
for fn in outputs_26:
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
kind = “animation” if fn.endswith(”.mp4”) else “figure”
print(f”  {fn:<48} {size:6.2f} MB  [{kind}]”)

# Print displacement statistics to stdout.

print(f”\n  DISPLACEMENT STATISTICS”)
print(f”  {‘Group’:<35} {‘N’:>5} {‘Median Δr’:>12} {‘Mean Δr’:>12} {‘σ(Δr)’:>12}”)
print(f”  {’-’*35} {’-’*5} {’-’*12} {’-’*12} {’-’*12}”)
for g, gname in enumerate(GROUP_NAMES):
mask  = (group_label == g) & valid_f
dr_g  = delta_r[mask]
print(f”  {gname:<35} {mask.sum():>5} “
f”{np.nanmedian(dr_g):>12.1f} “
f”{np.nanmean(dr_g):>12.1f} “
f”{np.nanstd(dr_g):>12.1f}”)
print(”=”*80)
