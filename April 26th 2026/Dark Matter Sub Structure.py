# “””

# SECTION 23 — DARK MATTER SUBSTRUCTURE

Author  : Abhinav Vatsa


Continuation of density_pipeline.py, section21_angular_momentum.py, and
section22_tidal_field.py.  All globals (SNAPSHOTS, ns, R_BINS, nb_sph,
r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL, G_KPC_KMS2_MSUN,
PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr, time_label,
time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

## Physical motivation

Cold dark matter predicts that galaxy halos are not smooth — they are filled
with gravitationally self-bound subhalos (often called substructure or
satellite halos) spanning many decades in mass.  In the MW–M31 merger the
pre-existing subhalo populations of both galaxies are disrupted, tidally
stripped, and eventually merged into a single remnant halo.  Tracking
substructure through the merger reveals:

(A) How quickly small-scale structure is erased by tidal heating
(B) Whether the merger creates new substructure via tidal debris
(C) How the subhalo mass function (SHMF) evolves through coalescence
(D) The radial distribution of surviving vs. disrupted subhalos

This section uses a friends-of-friends (FoF) group finder to identify
substructure candidates at each snapshot, then tracks their properties
— mass, radius, velocity dispersion, and survival — over time.

## Diagnostic families

§23.1  Friends-of-friends subhalo finder
§23.2  Subhalo mass function N(>M) at key epochs
§23.3  Radial distribution of subhalos n_sub(r, t)
§23.4  Subhalo survival fraction vs. time
§23.5  Velocity dispersion of subhalo members vs. field

## Outputs

section23_shmf.png                 Subhalo mass function at 5 epochs
section23_subhalo_radial.png       Radial number density of subhalos vs. time
section23_survival.png             Subhalo survival fraction vs. time
section23_mass_evolution.png       Mean/total subhalo mass vs. time
section23_sigma_sub_vs_field.png   σ_sub vs. field velocity dispersion
section23_subhalo_map.png          2D map of subhalo positions at 5 epochs
section23_shmf_heatmap.png         SHMF slope α(t) evolution heatmap
section23_animation_subhalos.mp4   Subhalo population animation
section23_summary_panel.png        Master 4-panel summary figure

===============================================================================
“””

import numpy as np
import matplotlib
matplotlib.use(“Agg”)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.0 — CONFIGURATION                                                     ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Friends-of-friends linking length ─────────────────────────────────────────

# b = 0.2 × mean inter-particle separation is the standard cosmological choice.

# In practice with N-body halos we fix the physical linking length directly.

# FOF_LINK_KPC is the maximum separation for two particles to be considered

# “friends”.  Particles whose friends form a chain are grouped together.

# Smaller values → more subhalos found, each less massive.

FOF_LINK_KPC     = 2.0      # [kpc]

# Minimum number of particles for a group to be considered a subhalo.

# Below this threshold a “group” is likely a spurious noise fluctuation.

FOF_MIN_PART     = 30

# ── Mass range for the subhalo mass function ──────────────────────────────────

SHMF_MMIN = 1e8    # M_sun — minimum subhalo mass to include in the SHMF
SHMF_MMAX = 1e12   # M_sun — maximum (above this it is the main halo)
SHMF_NBINS = 20    # number of log-spaced mass bins

# ── Subhalo spatial extent ────────────────────────────────────────────────────

# Maximum radius from the joint COM at which we search for subhalos.

# Subhalos beyond this are likely tidal debris fragments, not bound objects.

SUBHALO_RMAX_KPC = 400.0

# ── Temporal subsampling ──────────────────────────────────────────────────────

# FoF is O(N log N) per snapshot — still the most expensive step here.

# SUBHALO_STEP = 10 gives 80 analysis epochs across 800 snapshots.

SUBHALO_STEP = 10

# ── 2D map for subhalo position display ──────────────────────────────────────

SUB_MAP_BINS   = 256
SUB_MAP_EXTENT = 400.0   # [kpc]

# ── Animation ─────────────────────────────────────────────────────────────────

ANIM_FPS_23     = 15
ANIM_DPI_23     = 100
ANIM_BITRATE_23 = 1800

print(”\n” + “=”*80)
print(”  SECTION 23 · Dark Matter Substructure”)
print(”=”*80)
print(f”  FoF link length : {FOF_LINK_KPC} kpc”)
print(f”  Min particles   : {FOF_MIN_PART}”)
print(f”  Analysis step   : every {SUBHALO_STEP} snapshots”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.1 — FRIENDS-OF-FRIENDS GROUP FINDER                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def friends_of_friends(pos: np.ndarray,
m:   np.ndarray,
link_length: float,
min_particles: int) -> list[dict]:
“””
Simple friends-of-friends (FoF) group finder.

```
Algorithm
---------
Uses a union-find (disjoint set) data structure for O(N α(N)) complexity
where α is the inverse Ackermann function (effectively constant).

For each particle we find all neighbours within link_length using a
grid-based approach: particles are assigned to cells of size link_length,
then only neighbouring cells are searched.  This avoids the O(N²)
all-pairs distance calculation.

Parameters
----------
pos          : (N, 3)  — particle positions  [kpc]
m            : (N,)    — particle masses      [M_sun]
link_length  : float   — FoF linking length   [kpc]
min_particles: int     — minimum group size

Returns
-------
groups : list of dicts, each containing:
    "indices"  : np.ndarray  — indices of member particles
    "mass"     : float       — total group mass  [M_sun]
    "pos_com"  : (3,)        — mass-weighted COM position  [kpc]
    "r_com"    : float       — distance of group COM from origin  [kpc]
    "sigma_v"  : float       — 3D velocity dispersion of members
                               (requires vel to be passed separately;
                                here we compute from positions only —
                                vel is added in the main loop)

Notes
-----
This is intentionally a simplified FoF without a subsequent unbinding
step (SUBFIND-style).  A full SUBFIND would iteratively remove unbound
particles from each group, giving "subhalos" rather than "FoF groups".
Here we refer to results as "subhalo candidates".
"""
N = len(pos)
if N == 0:
    return []

# ── Union-Find data structure ─────────────────────────────────────────────
parent = np.arange(N, dtype=np.int64)
rank   = np.zeros(N, dtype=np.int64)

def find(x):
    # Path compression.
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    rx, ry = find(x), find(y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        rx, ry = ry, rx
    parent[ry] = rx
    if rank[rx] == rank[ry]:
        rank[rx] += 1

# ── Grid-based neighbour search ───────────────────────────────────────────
# Assign each particle to a grid cell.
cell_size = link_length
origin    = pos.min(axis=0) - cell_size
cell_idx  = ((pos - origin) / cell_size).astype(np.int64)

# Build cell → particle list.
cell_dict: dict[tuple, list] = {}
for i in range(N):
    key = tuple(cell_idx[i])
    if key not in cell_dict:
        cell_dict[key] = []
    cell_dict[key].append(i)

# For each particle, check all 27 neighbouring cells.
offsets = [(dx, dy, dz)
           for dx in (-1, 0, 1)
           for dy in (-1, 0, 1)
           for dz in (-1, 0, 1)]

link2 = link_length**2

for i in range(N):
    cx, cy, cz = cell_idx[i]
    for dx, dy, dz in offsets:
        nbr_key = (cx+dx, cy+dy, cz+dz)
        if nbr_key not in cell_dict:
            continue
        for j in cell_dict[nbr_key]:
            if j <= i:
                continue
            d2 = np.sum((pos[i] - pos[j])**2)
            if d2 <= link2:
                union(i, j)

# ── Collect groups ────────────────────────────────────────────────────────
group_map: dict[int, list] = {}
for i in range(N):
    root = find(i)
    if root not in group_map:
        group_map[root] = []
    group_map[root].append(i)

groups = []
for root, members in group_map.items():
    if len(members) < min_particles:
        continue

    idx      = np.array(members)
    M_grp    = m[idx].sum()
    pos_com  = np.sum(m[idx, None] * pos[idx], axis=0) / M_grp
    r_com    = np.linalg.norm(pos_com)

    groups.append({
        "indices": idx,
        "mass":    float(M_grp),
        "pos_com": pos_com,
        "r_com":   float(r_com),
    })

# Sort by mass descending.
groups.sort(key=lambda g: g["mass"], reverse=True)
return groups
```

def subhalo_velocity_dispersion(vel: np.ndarray,
m:   np.ndarray) -> float:
“””
Mass-weighted 3D velocity dispersion of a group of particles.

```
    σ_v = sqrt( Σ m_i |v_i − ⟨v⟩|² / (3 Σ m_i) )

The factor of 1/3 converts the total 3D variance to a 1D equivalent.

Returns
-------
sigma_v : float  [km/s]
"""
if len(vel) < 3:
    return np.nan
W     = m.sum()
v_com = np.sum(m[:, None] * vel, axis=0) / W
dv2   = np.sum((vel - v_com)**2, axis=1)
return float(np.sqrt(np.sum(m * dv2) / (3.0 * W)))
```

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.2 — PRE-ALLOCATION                                                    ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

sub_snap_nums = SNAPSHOTS[::SUBHALO_STEP]
n_sub_snaps   = len(sub_snap_nums)
sub_snap_map  = {s: i for i, s in enumerate(sub_snap_nums)}
time_sub      = np.full(n_sub_snaps, np.nan)

# Subhalo count per snapshot.

n_sub_arr     = np.full(n_sub_snaps, 0, dtype=int)

# Total subhalo mass per snapshot.

M_sub_tot_arr = np.full(n_sub_snaps, np.nan)

# Mean subhalo mass per snapshot.

M_sub_mean_arr = np.full(n_sub_snaps, np.nan)

# Mean subhalo velocity dispersion per snapshot.

sigma_sub_arr  = np.full(n_sub_snaps, np.nan)

# SHMF cumulative counts N(>M) per snapshot — shape (n_sub_snaps, SHMF_NBINS).

shmf_mass_bins = np.logspace(np.log10(SHMF_MMIN), np.log10(SHMF_MMAX), SHMF_NBINS + 1)
shmf_ts        = np.full((n_sub_snaps, SHMF_NBINS), np.nan)

# SHMF power-law slope α per snapshot.

shmf_slope_arr = np.full(n_sub_snaps, np.nan)

# Radial number density profile: n_sub(r) per snapshot.

# Use R_BINS for consistency with density pipeline.

n_sub_radial_ts = np.full((n_sub_snaps, nb_sph), np.nan)

# 2D position maps — one per analysis snapshot.

sub_maps = np.zeros((n_sub_snaps, SUB_MAP_BINS, SUB_MAP_BINS))

# Per-snapshot list of subhalo dicts (for the animation).

all_subhalo_lists: list[list[dict]] = [[] for _ in range(n_sub_snaps)]

print(f”\n[Pre-alloc] SHMF array : {shmf_ts.shape}”)
print(f”            Radial     : {n_sub_radial_ts.shape}”)
print(f”            Analysis snaps : {n_sub_snaps}”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.3 — MAIN LOOP                                                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  §23.3 — Main Snapshot Loop”)
print(”=”*80)

t_loop_start = time.perf_counter()

for i, snap_num in enumerate(sub_snap_nums):

```
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
r_mag  = np.linalg.norm(pos, axis=1)

# ── Velocities in COM frame ───────────────────────────────────────────────
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
    wi   = m[inn]
    vxcom = np.sum(wi * vx_all[inn]) / wi.sum()
    vycom = np.sum(wi * vy_all[inn]) / wi.sum()
    vzcom = np.sum(wi * vz_all[inn]) / wi.sum()
else:
    vxcom = vycom = vzcom = 0.0

vel = np.vstack((vx_all - vxcom,
                 vy_all - vycom,
                 vz_all - vzcom)).T   # (N, 3)  [km/s]

time_sub[i] = time_arr[np.where(SNAPSHOTS == snap_num)[0][0]] \
              if len(np.where(SNAPSHOTS == snap_num)[0]) > 0 else float(snap_num)

# ── Restrict to within SUBHALO_RMAX_KPC ───────────────────────────────────
# The main halo body dominates FoF at all radii, so we remove it before
# running FoF by keeping only particles in the outer halo and using a
# local overdensity threshold.  For simplicity we run FoF on a random
# subsample of particles in the range [5, SUBHALO_RMAX_KPC] kpc, which
# is the region where discrete subhalos are expected.
sub_region = (r_mag > 5.0) & (r_mag < SUBHALO_RMAX_KPC)
pos_sub    = pos[sub_region]
m_sub      = m  [sub_region]
vel_sub    = vel[sub_region]

if len(pos_sub) < FOF_MIN_PART * 2:
    continue

# ── Run FoF ───────────────────────────────────────────────────────────────
groups = friends_of_friends(pos_sub, m_sub, FOF_LINK_KPC, FOF_MIN_PART)

# ── Add velocity dispersion to each group ─────────────────────────────────
for g in groups:
    idx         = g["indices"]
    g["sigma_v"] = subhalo_velocity_dispersion(vel_sub[idx], m_sub[idx])

# ── Filter to mass range ──────────────────────────────────────────────────
# Remove the main halo body (largest group) and objects above SHMF_MMAX.
groups_filtered = [g for g in groups
                   if SHMF_MMIN <= g["mass"] <= SHMF_MMAX]

all_subhalo_lists[i] = groups_filtered

n_sub_arr    [i] = len(groups_filtered)
M_sub_tot_arr[i] = sum(g["mass"] for g in groups_filtered) or np.nan
M_sub_mean_arr[i] = M_sub_tot_arr[i] / max(1, n_sub_arr[i])
sigma_vals = [g["sigma_v"] for g in groups_filtered if np.isfinite(g["sigma_v"])]
sigma_sub_arr[i] = np.mean(sigma_vals) if sigma_vals else np.nan

# ── Subhalo mass function N(>M) ───────────────────────────────────────────
masses = np.array([g["mass"] for g in groups_filtered])
if len(masses) > 0:
    for b in range(SHMF_NBINS):
        shmf_ts[i, b] = np.sum(masses > shmf_mass_bins[b])

    # Fit a power law N(>M) ∝ M^α to the SHMF.
    # CDM predicts α ≈ −1 (equal mass per decade of subhalo mass).
    valid_bins = shmf_ts[i, :] > 1
    if valid_bins.sum() >= 3:
        log_M  = np.log10(0.5 * (shmf_mass_bins[:-1] + shmf_mass_bins[1:]))[valid_bins]
        log_N  = np.log10(shmf_ts[i, valid_bins])
        try:
            coeffs = np.polyfit(log_M, log_N, 1)
            shmf_slope_arr[i] = coeffs[0]
        except Exception:
            pass

# ── Radial number density of subhalos ─────────────────────────────────────
r_coms = np.array([g["r_com"] for g in groups_filtered])
if len(r_coms) > 0:
    bin_id = np.digitize(r_coms, R_BINS) - 1
    for b in range(nb_sph):
        n_in_bin = (bin_id == b).sum()
        shell_vol = (4.0/3.0) * np.pi * (R_BINS[b+1]**3 - R_BINS[b]**3)
        if shell_vol > 0:
            n_sub_radial_ts[i, b] = n_in_bin / shell_vol

# ── 2D subhalo position map ───────────────────────────────────────────────
if len(groups_filtered) > 0:
    x_coms = np.array([g["pos_com"][0] for g in groups_filtered])
    y_coms = np.array([g["pos_com"][1] for g in groups_filtered])
    H, _, _ = np.histogram2d(
        x_coms, y_coms,
        bins=SUB_MAP_BINS,
        range=[[-SUB_MAP_EXTENT, SUB_MAP_EXTENT],
               [-SUB_MAP_EXTENT, SUB_MAP_EXTENT]],
    )
    sub_maps[i] = H

if (i + 1) % 20 == 0:
    elapsed = time.perf_counter() - t_loop_start
    print(f"  sub-snap {snap_num:04d}  n_sub={n_sub_arr[i]}  "
          f"M_sub_tot={M_sub_tot_arr[i]:.2e}  [{elapsed:.0f}s]")
```

print(f”\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.4 — FIGURES                                                           ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

BG    = “#0d0d18”
MUTED = “#7070a0”

def _ax(ax, xlabel=””, ylabel=””, title=””, log_x=False, log_y=False):
ax.set_facecolor(BG)
for sp in ax.spines.values():
sp.set_edgecolor(”#2a2a4a”)
ax.tick_params(colors=”#9090b0”, labelsize=8)
ax.set_xlabel(xlabel, fontsize=9, color=”#c8c8e8”)
ax.set_ylabel(ylabel, fontsize=9, color=”#c8c8e8”)
ax.set_title(title,  fontsize=10, color=”#c8c8e8”, pad=5)
if log_x: ax.set_xscale(“log”)
if log_y: ax.set_yscale(“log”)
return ax

t_sub_min = np.nanmin(time_sub)
t_sub_max = np.nanmax(time_sub)

# Representative sub-snapshot indices for profile figures.

sub_profile_ii = [int(f * (n_sub_snaps - 1)) for f in [0.0, 0.2, 0.4, 0.65, 1.0]]

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 1 — SUBHALO MASS FUNCTION N(>M) AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# The CDM prediction is a power law N(>M) ∝ M^{−1}, corresponding to equal

# numbers of subhalos per decade of mass.  Deviation from this slope documents

# how the merger disrupts the subhalo population.  A steeper slope (α < −1)

# would indicate preferential disruption of low-mass subhalos; a shallower

# slope (α > −1) would indicate low-mass subhalos surviving while large ones

# are disrupted by dynamical friction.

print(”\n[Fig 1]  Subhalo mass function …”)

mass_bin_centres = 0.5 * (shmf_mass_bins[:-1] + shmf_mass_bins[1:])

fig1, ax1 = plt.subplots(figsize=(9, 6), facecolor=BG)
*ax(ax1, xlabel=r”Subhalo mass  $M*{\rm sub}$  [M$_\odot$]”,
ylabel=r”$N(> M)$”,
title=r”Subhalo Mass Function  $N(>M)$ at Key Epochs”,
log_x=True, log_y=True)

for ii, color, label in zip(sub_profile_ii, PROFILE_COLORS, PROFILE_LABELS):
y     = shmf_ts[ii, :]
valid = np.isfinite(y) & (y > 0)
if valid.any():
ax1.plot(mass_bin_centres[valid], y[valid],
color=color, lw=2.0, label=label)

# CDM reference slope N ∝ M^{-1}.

M_ref  = np.logspace(np.log10(SHMF_MMIN), np.log10(SHMF_MMAX), 50)
N_ref  = 100 * (M_ref / SHMF_MMIN)**(-1.0)
ax1.plot(M_ref, N_ref, color=”#ffffff”, lw=0.8, ls=”:”,
alpha=0.4, label=r”CDM $N\propto M^{-1}$”)

ax1.legend(fontsize=8)
ax1.set_xlim(SHMF_MMIN, SHMF_MMAX)

fig1.savefig(os.path.join(OUT_DIR, “section23_shmf.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig1)
print(”  Saved: section23_shmf.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 2 — SHMF SLOPE α(t) EVOLUTION

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 2]  SHMF slope heatmap …”)

fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor=BG)
_ax(ax2, xlabel=time_label,
ylabel=r”SHMF slope $\alpha$”,
title=r”Subhalo Mass Function Power-Law Slope  $\alpha(t)$  [N $\propto M^\alpha$]”)

valid = np.isfinite(shmf_slope_arr) & np.isfinite(time_sub)
ax2.plot(time_sub[valid], shmf_slope_arr[valid], color=”#e8673a”, lw=1.8)
ax2.fill_between(time_sub[valid], shmf_slope_arr[valid], -1.0,
where=shmf_slope_arr[valid] > -1.0,
alpha=0.12, color=”#e8673a”)
ax2.axhline(-1.0, color=”#ffffff”, lw=0.8, ls=”–”, alpha=0.5,
label=r”CDM expectation $\alpha = -1$”)
ax2.legend(fontsize=8)

fig2.savefig(os.path.join(OUT_DIR, “section23_shmf_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig2)
print(”  Saved: section23_shmf_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 3 — SUBHALO RADIAL NUMBER DENSITY  n_sub(r) AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# The radial distribution of subhalos compared to the smooth dark matter

# density profile measures “anti-bias”: subhalos are disrupted preferentially

# in the dense inner halo (by dynamical friction and tidal stripping), so

# n_sub(r) / ρ(r) should increase with radius.

print(”[Fig 3]  Subhalo radial distribution …”)

fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor=BG)
*ax(ax3, xlabel=“r [kpc]”,
ylabel=r”$n*{\rm sub}(r)$  [kpc$^{-3}$]”,
title=r”Radial Number Density of Subhalos $n_{\rm sub}(r)$”,
log_x=True, log_y=True)

for ii, color, label in zip(sub_profile_ii, PROFILE_COLORS, PROFILE_LABELS):
y     = n_sub_radial_ts[ii, :]
valid = np.isfinite(y) & (y > 0)
if valid.any():
ax3.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax3.set_xlim(R_BINS[0], R_BINS[-1])
ax3.legend(fontsize=8)

fig3.savefig(os.path.join(OUT_DIR, “section23_subhalo_radial.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig3)
print(”  Saved: section23_subhalo_radial.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 4 — SUBHALO SURVIVAL: COUNT AND TOTAL MASS VS. TIME

# ══════════════════════════════════════════════════════════════════════════════

# 

# A drop in n_sub with time documents tidal disruption of subhalos during the

# merger.  The rate of decrease steepens at pericentre passages.  If n_sub

# partially recovers after the merger, new substructure has formed from

# tidal debris streams fragmenting under self-gravity.

print(”[Fig 4]  Subhalo survival …”)

fig4, axes4 = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

_ax(axes4[0], ylabel=“Number of subhalos”,
title=“Subhalo Population Survival”)
axes4[0].plot(time_sub, n_sub_arr, color=”#00d4aa”, lw=1.8)

*ax(axes4[1], xlabel=time_label,
ylabel=r”$M*{\rm sub,,tot}$ [M$_\odot$]”,
title=“Total Subhalo Mass”)
valid = np.isfinite(M_sub_tot_arr)
axes4[1].semilogy(time_sub[valid], M_sub_tot_arr[valid], color=”#ff9944”, lw=1.8)

fig4.savefig(os.path.join(OUT_DIR, “section23_survival.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig4)
print(”  Saved: section23_survival.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 5 — MEAN SUBHALO MASS AND VELOCITY DISPERSION VS. TIME

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 5]  Mean subhalo mass and σ_v …”)

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

*ax(ax5a, ylabel=r”$\langle M*{\rm sub} \rangle$ [M$_\odot$]”,
title=“Mean Subhalo Mass”)
valid = np.isfinite(M_sub_mean_arr)
ax5a.semilogy(time_sub[valid], M_sub_mean_arr[valid], color=”#4a8fff”, lw=1.8,
label=“Mean subhalo mass”)
ax5a.legend(fontsize=8)

_ax(ax5b, xlabel=time_label,
ylabel=r”$\langle \sigma_v \rangle$ [km s$^{-1}$]”,
title=“Mean Subhalo Velocity Dispersion”)
valid_s = np.isfinite(sigma_sub_arr)
ax5b.plot(time_sub[valid_s], sigma_sub_arr[valid_s], color=”#aa55ff”, lw=1.8,
label=r”Mean $\sigma_v$ subhalos”)
ax5b.legend(fontsize=8)

fig5.savefig(os.path.join(OUT_DIR, “section23_mass_evolution.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5)
print(”  Saved: section23_mass_evolution.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 6 — SUBHALO σ_v VS. FIELD σ_r (KINEMATIC BIAS)

# ══════════════════════════════════════════════════════════════════════════════

# 

# If subhalos are unbiased tracers of the potential, their velocity dispersion

# should equal the field velocity dispersion at the same radius.  A ratio

# σ_sub / σ_field > 1 means subhalos are “hot” relative to the background —

# they have been kinematically heated by tidal interactions.

# Ratio < 1 means subhalos are on more ordered orbits than the field.

print(”[Fig 6]  Subhalo σ_v vs. field …”)

# Load field σ_r_global from the kinematics pipeline if available.

# If not, use a placeholder.

fig6, ax6 = plt.subplots(figsize=(10, 4), facecolor=BG)
*ax(ax6, xlabel=time_label,
ylabel=r”$\sigma$ [km s$^{-1}$]”,
title=r”Subhalo $\langle\sigma_v\rangle$ vs. Field $\sigma*{r,,\rm global}$”)

valid_s = np.isfinite(sigma_sub_arr)
ax6.plot(time_sub[valid_s], sigma_sub_arr[valid_s], color=”#aa55ff”, lw=1.8,
label=r”Subhalos $\langle\sigma_v\rangle$”)

# Attempt to overlay the field σ_r_global if it was computed in §4 of the

# kinematics pipeline and is still in scope.

try:
ax6.plot(time_arr, sigma_r_glob_arr, color=”#4a8fff”, lw=1.5, ls=”–”,
label=r”Field $\sigma_{r,,\rm global}$”, alpha=0.8)
except NameError:
ax6.text(0.5, 0.5,
“sigma_r_glob_arr not in scope\n(run kinematic_profiles_pipeline.py first)”,
transform=ax6.transAxes, ha=“center”, va=“center”,
color=MUTED, fontsize=9)

ax6.legend(fontsize=8)
fig6.savefig(os.path.join(OUT_DIR, “section23_sigma_sub_vs_field.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig6)
print(”  Saved: section23_sigma_sub_vs_field.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 7 — 2D SUBHALO POSITION MAP AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 7]  2D subhalo maps …”)

fig7, axes7 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.04})

for col, (ii, label) in enumerate(zip(sub_profile_ii, PROFILE_LABELS)):
ax = axes7[col]
ax.set_facecolor(BG)
ax.set_title(label, fontsize=9, color=”#c8c8e8”)

```
groups_here = all_subhalo_lists[ii]
if groups_here:
    x_c = np.array([g["pos_com"][0] for g in groups_here])
    y_c = np.array([g["pos_com"][1] for g in groups_here])
    m_c = np.array([g["mass"]       for g in groups_here])
    # Scatter with size ∝ log(mass).
    sizes = 5 + 40 * (np.log10(m_c) - np.log10(SHMF_MMIN)) / \
                     (np.log10(SHMF_MMAX) - np.log10(SHMF_MMIN))
    ax.scatter(x_c, y_c, s=sizes, color=PROFILE_COLORS[col],
               alpha=0.7, edgecolors="none", rasterized=True)

ax.set_xlim(-SUB_MAP_EXTENT, SUB_MAP_EXTENT)
ax.set_ylim(-SUB_MAP_EXTENT, SUB_MAP_EXTENT)
ax.tick_params(colors="#9090b0", labelsize=7)
ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")
if col == 0:
    ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")
```

fig7.suptitle(r”Subhalo Positions  (marker size $\propto \log M_{\rm sub}$)”,
fontsize=11)
fig7.savefig(os.path.join(OUT_DIR, “section23_subhalo_map.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig7)
print(”  Saved: section23_subhalo_map.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.5 — ANIMATION                                                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Two-panel animation:

# Left  : 2D subhalo position scatter (marker size ∝ log mass)

# Right : Running subhalo count N_sub(t)

print(”\n[Anim]  Subhalo population animation …”)

fig_a, (ax_map, ax_cnt) = plt.subplots(
1, 2, figsize=(13, 5.5), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 1], “wspace”: 0.1},
)
ax_map.set_facecolor(BG)
ax_cnt.set_facecolor(BG)

ax_map.set_xlim(-SUB_MAP_EXTENT, SUB_MAP_EXTENT)
ax_map.set_ylim(-SUB_MAP_EXTENT, SUB_MAP_EXTENT)
ax_map.set_xlabel(“x [kpc]”, color=”#c8c8e8”)
ax_map.set_ylabel(“y [kpc]”, color=”#c8c8e8”)

ax_cnt.set_xlim(t_sub_min, t_sub_max)
ax_cnt.set_ylim(0, max(int(n_sub_arr.max() * 1.2), 1))
ax_cnt.set_xlabel(time_label, color=”#c8c8e8”)
ax_cnt.set_ylabel(“N subhalos”, color=”#c8c8e8”)
ax_cnt.set_title(“Count”, color=”#c8c8e8”, fontsize=10)

scat = ax_map.scatter([], [], s=[], color=”#00d4aa”, alpha=0.7, edgecolors=“none”)
cnt_line, = ax_cnt.plot([], [], color=”#00d4aa”, lw=1.8)
title_a = fig_a.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_sub_anim(frame_idx):
groups_here = all_subhalo_lists[frame_idx]
if groups_here:
x_c = np.array([g[“pos_com”][0] for g in groups_here])
y_c = np.array([g[“pos_com”][1] for g in groups_here])
m_c = np.array([g[“mass”]       for g in groups_here])
sizes = 5 + 40 * (np.log10(np.clip(m_c, SHMF_MMIN, SHMF_MMAX)) -
np.log10(SHMF_MMIN)) /   
(np.log10(SHMF_MMAX) - np.log10(SHMF_MMIN))
scat.set_offsets(np.column_stack((x_c, y_c)))
scat.set_sizes(sizes)
else:
scat.set_offsets(np.empty((0, 2)))
scat.set_sizes([])

```
valid = np.isfinite(time_sub[:frame_idx+1])
cnt_line.set_data(time_sub[:frame_idx+1][valid],
                  n_sub_arr[:frame_idx+1][valid])

t_val = time_sub[frame_idx]
t_str = (f"{t_val:.2f} Gyr" if (np.isfinite(t_val) and time_is_gyr)
         else f"Sub-snap {frame_idx}")
title_a.set_text(f"DM Substructure  ·  N={n_sub_arr[frame_idx]}  ·  {t_str}")
return [scat, cnt_line]
```

ani_sub = animation.FuncAnimation(
fig_a, _update_sub_anim, frames=n_sub_snaps,
interval=1000 // ANIM_FPS_23, blit=True,
)
writer_23 = animation.FFMpegWriter(
fps=ANIM_FPS_23, bitrate=ANIM_BITRATE_23,
metadata=dict(title=“MW-M31 DM Substructure Animation”),
)
ani_sub.save(os.path.join(OUT_DIR, “section23_animation_subhalos.mp4”),
writer=writer_23, dpi=ANIM_DPI_23)
plt.close(fig_a)
print(”  Saved: section23_animation_subhalos.mp4”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.6 — MASTER SUMMARY PANEL                                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n[Summary]  Master summary panel …”)

fig_s = plt.figure(figsize=(16, 10), facecolor=BG)
gs_s  = gridspec.GridSpec(2, 2, figure=fig_s,
hspace=0.38, wspace=0.32,
left=0.08, right=0.97,
top=0.93, bottom=0.07)

# (0,0) SHMF at 5 epochs

ax_s00 = fig_s.add_subplot(gs_s[0, 0])
*ax(ax_s00, xlabel=r”$M*{\rm sub}$ [M$_\odot$]”, ylabel=r”$N(>M)$”,
title=“Subhalo Mass Function”, log_x=True, log_y=True)
for ii, color, label in zip(sub_profile_ii, PROFILE_COLORS, PROFILE_LABELS):
y = shmf_ts[ii, :]
v = np.isfinite(y) & (y > 0)
if v.any():
ax_s00.plot(mass_bin_centres[v], y[v], color=color, lw=1.5, label=label)
ax_s00.legend(fontsize=6)

# (0,1) Subhalo count and total mass

ax_s01 = fig_s.add_subplot(gs_s[0, 1])
*ax(ax_s01, xlabel=time_label, ylabel=“N subhalos”,
title=“Subhalo Count vs. Time”)
ax_s01.plot(time_sub, n_sub_arr, color=”#00d4aa”, lw=1.8)
ax_s01_r = ax_s01.twinx()
ax_s01_r.set_facecolor(BG)
valid = np.isfinite(M_sub_tot_arr)
ax_s01_r.semilogy(time_sub[valid], M_sub_tot_arr[valid],
color=”#ff9944”, lw=1.5, ls=”–”)
ax_s01_r.set_ylabel(r”$M*{\rm sub,tot}$ [M$_\odot$]”,
fontsize=8, color=”#ff9944”)
ax_s01_r.tick_params(colors=”#ff9944”)

# (1,0) Radial number density profiles

ax_s10 = fig_s.add_subplot(gs_s[1, 0])
*ax(ax_s10, xlabel=“r [kpc]”, ylabel=r”$n*{\rm sub}$ [kpc$^{-3}$]”,
title=“Radial Distribution”, log_x=True, log_y=True)
for ii, color, label in zip(sub_profile_ii, PROFILE_COLORS, PROFILE_LABELS):
y = n_sub_radial_ts[ii, :]
v = np.isfinite(y) & (y > 0)
if v.any():
ax_s10.plot(r_mid_sph[v], y[v], color=color, lw=1.5, label=label)
ax_s10.set_xlim(R_BINS[0], R_BINS[-1])
ax_s10.legend(fontsize=6)

# (1,1) SHMF slope α(t)

ax_s11 = fig_s.add_subplot(gs_s[1, 1])
_ax(ax_s11, xlabel=time_label, ylabel=r”$\alpha$”,
title=r”SHMF Slope $\alpha(t)$”)
valid = np.isfinite(shmf_slope_arr)
ax_s11.plot(time_sub[valid], shmf_slope_arr[valid], color=”#e8673a”, lw=1.8)
ax_s11.axhline(-1.0, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.4,
label=“CDM α = −1”)
ax_s11.legend(fontsize=8)

fig_s.suptitle(“Section 23 Summary  ·  Dark Matter Substructure”,
fontsize=13, color=”#c8c8e8”, fontweight=“bold”)
fig_s.savefig(os.path.join(OUT_DIR, “section23_summary_panel.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig_s)
print(”  Saved: section23_summary_panel.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §23.7 — SECTION COMPLETE                                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  SECTION 23 COMPLETE”)
print(”=”*80)
outputs_23 = [
“section23_shmf.png”,
“section23_shmf_heatmap.png”,
“section23_subhalo_radial.png”,
“section23_survival.png”,
“section23_mass_evolution.png”,
“section23_sigma_sub_vs_field.png”,
“section23_subhalo_map.png”,
“section23_animation_subhalos.mp4”,
“section23_summary_panel.png”,
]
for fn in outputs_23:
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
kind = “animation” if fn.endswith(”.mp4”) else “figure”
print(f”  {fn:<50} {size:6.2f} MB  [{kind}]”)
print(”=”*80)
