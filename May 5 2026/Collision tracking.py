# “””

# SECTION 27 — DARK MATTER CLOSE ENCOUNTERS & COLLISION TRACKING

Author  : Abhinav Vatsa

All globals (SNAPSHOTS, ns,
R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL,
G_KPC_KMS2_MSUN, PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr,
time_label, time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

## A note on “collisions” in N-body simulations

True dark matter is collisionless — DM particles interact only through
gravity, never through short-range scattering the way gas particles do.
In an N-body simulation each “particle” represents ~10^5 M_sun of DM and
has no physical size.  So there are no literal collisions.

What we track instead are CLOSE ENCOUNTERS — pairs of particles that pass
within a threshold distance d_enc of each other.  These are physically
meaningful for several reasons:

(A) GRAVITATIONAL FOCUSING — two particles on a close approach trajectory
exchange momentum via their mutual gravitational pull, deflecting
each other’s paths.  This is the N-body analogue of two-body
relaxation and is the mechanism behind dynamical friction.

(B) HIGH LOCAL DENSITY EVENTS — regions where many close encounters occur
simultaneously mark density peaks (halo centres, subhalo cores).
Tracking the spatial distribution of encounters maps the density
field without binning into shells.

(C) RELATIVE VELOCITY DISTRIBUTION — the distribution of relative speeds
at closest approach, P(v_rel), distinguishes cold encounters (v_rel
small, deep in the potential well) from hot encounters (v_rel large,
high-speed fly-bys at pericentre).

(D) ENCOUNTER RATE EVOLUTION — the number of close encounters per
snapshot Γ(t) is proportional to n² σ v_rel (the Boltzmann collision
rate analogue) and peaks at pericentre passages where both density
and relative velocity are high simultaneously.

## Implementation

Computing all pairwise distances for N ~ 10^6 particles is O(N^2) and
completely impractical.  We use two strategies:

1. GRID-BASED SPATIAL HASHING — divide the simulation volume into cubic
   cells of size d_enc.  Only particles in the same or adjacent cells
   can be within d_enc of each other.  This reduces the search to O(N)
   on average.
1. KD-TREE RADIUS SEARCH — for a random subsample of N_PROBE particles,
   use scipy.spatial.cKDTree.query_ball_point to find all particles
   within d_enc.  This gives exact encounter counts for the subsample.

We run the KD-tree approach on N_PROBE particles per snapshot, which gives
a statistically representative sample of the encounter field.

## Outputs

section27_encounter_rate.png         Γ(t) — total encounter rate vs. time
section27_encounter_spatial.png      2D map of encounter density at 5 epochs
section27_vrel_dist.png              P(v_rel) distributions at 5 epochs
section27_encounter_radius.png       Encounter rate profile Γ(r,t) heatmap
section27_energy_transfer.png        Mean |ΔE| per encounter vs. time
section27_deflection_angle.png       Mean deflection angle θ vs. time
section27_two_body_relaxation.png    t_relax(r) profile at 5 epochs
section27_animation_encounters.mp4   2D encounter density animation
section27_summary_panel.png          Master 4-panel summary

===============================================================================
“””

import numpy as np
import matplotlib
matplotlib.use(“Agg”)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.0 — CONFIGURATION                                                     ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# Close-encounter distance threshold [kpc].

# Two particles within this distance constitute a “close encounter”.

# Set to ~2× the gravitational softening length to capture physically

# meaningful interactions while avoiding softening-dominated pairs.

D_ENC_KPC = 1.0

# Number of probe particles per snapshot for the KD-tree encounter search.

# Higher = more accurate statistics, slower runtime.

# N_PROBE = 5000 gives ~1–2 s per snapshot on a modern CPU.

N_PROBE = 5000

# Maximum relative velocity to be considered a “cold” encounter [km/s].

# Encounters with v_rel < V_COLD_KMS are in the gravitationally focused regime.

V_COLD_KMS = 50.0

# Radial bins for the encounter rate profile Γ(r).

# Reuse R_BINS from the parent pipeline.

# 2D encounter density map parameters.

ENC_MAP_BINS   = 200
ENC_MAP_EXTENT = 200.0   # [kpc] — zoomed in vs. stream maps

# Temporal step for the encounter analysis.

# Every ENC_STEP snapshots are analysed; intermediate snaps are skipped.

ENC_STEP = 5

# Animation subsampling.

ANIM_FPS_27     = 18
ANIM_DPI_27     = 100
ANIM_BITRATE_27 = 1600

print(”\n” + “=”*80)
print(”  SECTION 27 · Dark Matter Close Encounters & Collision Tracking”)
print(”=”*80)
print(f”  Encounter threshold : {D_ENC_KPC} kpc”)
print(f”  Probe particles     : {N_PROBE} per snapshot”)
print(f”  Analysis step       : every {ENC_STEP} snapshots”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.1 — UTILITY FUNCTIONS                                                 ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def find_close_encounters(pos:     np.ndarray,
vel:     np.ndarray,
m:       np.ndarray,
d_enc:   float,
n_probe: int,
rng:     np.random.Generator) -> dict:
“””
Find close encounters between a random probe set and all other particles.

```
Algorithm
---------
1. Draw n_probe particle indices uniformly at random.
2. Build a cKDTree over all particles.
3. For each probe particle, find all neighbours within d_enc.
4. For each (probe, neighbour) pair, compute:
     - Separation distance d  [kpc]
     - Relative velocity v_rel = |v_probe − v_neighbour|  [km/s]
     - Impact parameter b ≈ d  (instantaneous approximation)
     - Deflection angle  θ = 2 arctan(G(m1+m2) / (b v_rel²))
     - Specific energy transfer |ΔE| ≈ (G m / b)² / v_rel²  [km²/s²]
     - 3D position of the encounter midpoint

Parameters
----------
pos     : (N, 3)  — particle positions  [kpc]
vel     : (N, 3)  — particle velocities [km/s]
m       : (N,)    — particle masses      [M_sun]
d_enc   : float   — encounter radius     [kpc]
n_probe : int     — number of probe particles
rng     : Generator — seeded RNG for reproducibility

Returns
-------
dict with keys:
    n_enc          : int     — total number of pairs found
    d_arr          : (n_enc,) — separations  [kpc]
    vrel_arr       : (n_enc,) — relative speeds [km/s]
    theta_arr      : (n_enc,) — deflection angles [rad]
    dE_arr         : (n_enc,) — |ΔE| specific energy transfer [km²/s²]
    midpoint_arr   : (n_enc, 3) — encounter midpoints [kpc]
    r_mid_enc      : (n_enc,) — distance of midpoint from joint COM
    probe_r        : (n_probe,) — radii of probe particles [kpc]
    enc_per_probe  : (n_probe,) — number of encounters per probe
"""
N = len(pos)
if N < 10 or n_probe < 1:
    return {k: np.array([]) for k in
            ["d_arr","vrel_arr","theta_arr","dE_arr",
             "midpoint_arr","r_mid_enc","probe_r","enc_per_probe"]}

n_probe = min(n_probe, N)
probe_idx = rng.choice(N, size=n_probe, replace=False)

# Build KD-tree over all particles.
tree = cKDTree(pos)

d_list        = []
vrel_list     = []
theta_list    = []
dE_list       = []
midpoint_list = []
enc_per_probe = np.zeros(n_probe, dtype=int)

for pi, probe_i in enumerate(probe_idx):
    neighbours = tree.query_ball_point(pos[probe_i], d_enc)
    neighbours = [j for j in neighbours if j != probe_i]

    enc_per_probe[pi] = len(neighbours)

    for j in neighbours:
        d_ij   = float(np.linalg.norm(pos[probe_i] - pos[j]))
        if d_ij < 1e-10:
            continue

        v_ij   = vel[probe_i] - vel[j]
        vrel   = float(np.linalg.norm(v_ij))

        # Deflection angle in the two-body gravitational scattering
        # (Rutherford scattering analogue):
        #   tan(θ/2) = G(m1+m2) / (b × v_rel²)
        # where b ≈ d_ij is the instantaneous impact parameter.
        M_pair = m[probe_i] + m[j]
        if vrel > 0 and d_ij > 0:
            theta = 2.0 * np.arctan(
                G_KPC_KMS2_MSUN * M_pair / max(d_ij * vrel**2, 1e-30)
            )
        else:
            theta = 0.0

        # Specific energy transfer (perturbative estimate):
        #   |ΔE| ≈ (G m_partner / b)² / v_rel²
        if vrel > 0 and d_ij > 0:
            dE = (G_KPC_KMS2_MSUN * m[j] / d_ij)**2 / vrel**2
        else:
            dE = 0.0

        mid = 0.5 * (pos[probe_i] + pos[j])

        d_list.append(d_ij)
        vrel_list.append(vrel)
        theta_list.append(theta)
        dE_list.append(dE)
        midpoint_list.append(mid)

if not d_list:
    probe_r = np.linalg.norm(pos[probe_idx], axis=1)
    return {
        "n_enc": 0,
        "d_arr":        np.array([]),
        "vrel_arr":     np.array([]),
        "theta_arr":    np.array([]),
        "dE_arr":       np.array([]),
        "midpoint_arr": np.empty((0, 3)),
        "r_mid_enc":    np.array([]),
        "probe_r":      probe_r,
        "enc_per_probe": enc_per_probe,
    }

midpoints   = np.array(midpoint_list)
r_mid_enc   = np.linalg.norm(midpoints, axis=1)
probe_r     = np.linalg.norm(pos[probe_idx], axis=1)

return {
    "n_enc":          len(d_list),
    "d_arr":          np.array(d_list),
    "vrel_arr":       np.array(vrel_list),
    "theta_arr":      np.array(theta_list),
    "dE_arr":         np.array(dE_list),
    "midpoint_arr":   midpoints,
    "r_mid_enc":      r_mid_enc,
    "probe_r":        probe_r,
    "enc_per_probe":  enc_per_probe,
}
```

def two_body_relaxation_time(rho:    np.ndarray,
sigma:  np.ndarray,
m_part: float,
ln_lam: float = 10.0) -> np.ndarray:
“””
Compute the two-body relaxation timescale t_relax(r) per radial shell.

```
The standard formula (Binney & Tremaine 2008, eq. 1.37):

    t_relax = (0.34 σ³) / (G² m ρ ln Λ)

where:
    σ     [km/s]          — local 1D velocity dispersion
    ρ     [M_sun kpc^{-3}]— local density
    m     [M_sun]         — particle mass
    ln Λ  — Coulomb logarithm ≈ ln(N_particles / 2) ≈ 10 for a galaxy

Unit check:
    [σ³ / (G² m ρ)] = [(km/s)³ / ((kpc (km/s)² M_sun^{-1})² × M_sun × M_sun kpc^{-3})]
                    = [(km/s)³ × M_sun² × kpc³] / [(kpc² (km/s)⁴ M_sun^{-2}) × M_sun × M_sun kpc^{-3}]
                    = [kpc / (km/s)] = [kpc s / km]
Convert to Gyr: multiply by (kpc / km) × (1 s / Gyr_in_s).

Parameters
----------
rho     : (nb,)  — density per shell  [M_sun kpc^{-3}]
sigma   : (nb,)  — 1D velocity dispersion per shell  [km/s]
m_part  : float  — representative particle mass  [M_sun]
ln_lam  : float  — Coulomb logarithm

Returns
-------
t_relax : (nb,)  [Gyr]
"""
# Unit conversion: 1 kpc / (km/s) = 0.978 Gyr
KPC_KMS_TO_GYR = 0.9778

nb      = len(rho)
t_relax = np.full(nb, np.nan)

for b in range(nb):
    if not (np.isfinite(rho[b])   and rho[b]   > 0 and
            np.isfinite(sigma[b]) and sigma[b]  > 0):
        continue
    numerator   = 0.34 * sigma[b]**3
    denominator = G_KPC_KMS2_MSUN**2 * m_part * rho[b] * ln_lam
    if denominator > 0:
        t_relax[b] = (numerator / denominator) * KPC_KMS_TO_GYR

return t_relax
```

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.2 — PRE-ALLOCATION                                                    ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

enc_snap_nums = SNAPSHOTS[::ENC_STEP]
n_enc_snaps   = len(enc_snap_nums)
enc_snap_map  = {s: i for i, s in enumerate(enc_snap_nums)}
time_enc      = np.full(n_enc_snaps, np.nan)

# Total encounter count per snapshot.

n_enc_arr        = np.full(n_enc_snaps, np.nan)

# Mean kinematic quantities per snapshot.

mean_vrel_arr    = np.full(n_enc_snaps, np.nan)
mean_theta_arr   = np.full(n_enc_snaps, np.nan)
mean_dE_arr      = np.full(n_enc_snaps, np.nan)

# Cold encounter fraction (v_rel < V_COLD_KMS).

cold_frac_arr    = np.full(n_enc_snaps, np.nan)

# Encounter rate profile: encounters per unit volume per shell.

gamma_radial_ts  = np.full((n_enc_snaps, nb_sph), np.nan)

# Two-body relaxation time profile.

t_relax_ts       = np.full((n_enc_snaps, nb_sph), np.nan)

# 2D encounter density maps.

enc_maps         = np.zeros((n_enc_snaps, ENC_MAP_BINS, ENC_MAP_BINS))

# Store v_rel distributions at profile epochs for the histogram figure.

profile_enc_ii   = [int(f * (n_enc_snaps - 1)) for f in [0.0, 0.2, 0.4, 0.65, 1.0]]
vrel_dists       = [np.array([]) for _ in range(5)]

rng27 = np.random.default_rng(seed=27)

print(f”\n[Pre-alloc]  gamma_radial_ts : {gamma_radial_ts.shape}”)
print(f”             t_relax_ts      : {t_relax_ts.shape}”)
print(f”             Encounter snaps : {n_enc_snaps}”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.3 — MAIN LOOP                                                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  §27.3 — Main Encounter Loop”)
print(”=”*80)

t_loop_start = time.perf_counter()

for i, snap_num in enumerate(enc_snap_nums):

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

# ── COM-frame velocities ──────────────────────────────────────────────────
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
    wi    = m[inn]
    vxcom = np.sum(wi*vx_all[inn])/wi.sum()
    vycom = np.sum(wi*vy_all[inn])/wi.sum()
    vzcom = np.sum(wi*vz_all[inn])/wi.sum()
else:
    vxcom = vycom = vzcom = 0.0

vel = np.vstack((vx_all-vxcom, vy_all-vycom, vz_all-vzcom)).T

time_enc[i] = time_arr[np.where(SNAPSHOTS == snap_num)[0][0]] \
              if len(np.where(SNAPSHOTS == snap_num)[0]) > 0 else float(snap_num)

# ── Typical particle mass for t_relax ─────────────────────────────────────
m_typical = float(np.median(m))

# ── Find close encounters ─────────────────────────────────────────────────
enc = find_close_encounters(pos, vel, m, D_ENC_KPC, N_PROBE, rng27)

n_enc_arr[i] = enc["n_enc"]

if enc["n_enc"] > 0:
    mean_vrel_arr [i] = float(np.mean(enc["vrel_arr"]))
    mean_theta_arr[i] = float(np.mean(enc["theta_arr"]))
    mean_dE_arr   [i] = float(np.mean(enc["dE_arr"]))
    cold_frac_arr [i] = float(np.mean(enc["vrel_arr"] < V_COLD_KMS))

    # ── Encounter rate profile Γ(r) ───────────────────────────────────────
    # Bin encounter midpoints by their distance from the joint COM.
    shell_vols_loc = (4.0/3.0)*np.pi*(R_BINS[1:]**3 - R_BINS[:-1]**3)
    bin_id_enc = np.digitize(enc["r_mid_enc"], R_BINS) - 1
    for b in range(nb_sph):
        n_in_bin = (bin_id_enc == b).sum()
        if shell_vols_loc[b] > 0:
            # Normalise by probe fraction to get an absolute rate estimate.
            gamma_radial_ts[i, b] = (n_in_bin / shell_vols_loc[b]) * \
                                     (len(pos) / N_PROBE)

    # ── 2D encounter density map ──────────────────────────────────────────
    if enc["midpoint_arr"].shape[0] > 0:
        x_enc = enc["midpoint_arr"][:, 0]
        y_enc = enc["midpoint_arr"][:, 1]
        H, _, _ = np.histogram2d(
            x_enc, y_enc,
            bins=ENC_MAP_BINS,
            range=[[-ENC_MAP_EXTENT, ENC_MAP_EXTENT],
                   [-ENC_MAP_EXTENT, ENC_MAP_EXTENT]],
        )
        enc_maps[i] = H

    # ── Store v_rel distribution at profile epochs ─────────────────────────
    for pi_idx, profile_ii in enumerate(profile_enc_ii):
        if i == profile_ii:
            vrel_dists[pi_idx] = enc["vrel_arr"].copy()

# ── Two-body relaxation time profile ─────────────────────────────────────
# Recompute ρ and σ_r locally for this (subsampled) snapshot.
shell_vols_loc = (4.0/3.0)*np.pi*(R_BINS[1:]**3 - R_BINS[:-1]**3)
bin_id_3d = np.digitize(r_mag, R_BINS) - 1
rho_loc   = np.full(nb_sph, np.nan)
sigma_loc = np.full(nb_sph, np.nan)

with np.errstate(divide="ignore", invalid="ignore"):
    r_hat = np.where(r_mag[:,None] > 0, pos/r_mag[:,None], 0.0)
v_r_all = np.einsum("ij,ij->i", vel, r_hat)

for b in range(nb_sph):
    mask = bin_id_3d == b
    if mask.sum() < MIN_PART_SHELL:
        continue
    M_bin      = m[mask].sum()
    rho_loc[b] = M_bin / shell_vols_loc[b]
    w          = m[mask]; W = w.sum()
    vr_b       = v_r_all[mask]
    vr_mean    = np.sum(w*vr_b)/W
    sigma_loc[b] = np.sqrt(np.sum(w*(vr_b-vr_mean)**2)/W)

t_relax_ts[i, :] = two_body_relaxation_time(rho_loc, sigma_loc, m_typical)

if (i + 1) % 20 == 0:
    elapsed = time.perf_counter() - t_loop_start
    print(f"  enc-snap {snap_num:04d}  "
          f"n_enc={int(n_enc_arr[i]) if np.isfinite(n_enc_arr[i]) else 0}  "
          f"⟨v_rel⟩={mean_vrel_arr[i]:.0f} km/s  "
          f"cold_frac={cold_frac_arr[i]:.2f}  [{elapsed:.0f}s]")
```

print(f”\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.4 — FIGURES                                                           ║

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

t_enc_min = np.nanmin(time_enc)
t_enc_max = np.nanmax(time_enc)
enc_profile_labels = [f”Snap {enc_snap_nums[ii]}” for ii in profile_enc_ii]

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 1 — ENCOUNTER RATE Γ(t) VS. TIME

# ══════════════════════════════════════════════════════════════════════════════

# 

# Γ(t) is the total number of close encounters detected per snapshot.

# It is proportional to n² σ v_rel — so it peaks at pericentre where both

# density and relative velocity are simultaneously high.  The decay after

# first pericentre reflects both the fall in density (expansion) and the

# gradual thermalisation that reduces n² faster than v_rel grows.

print(”\n[Fig 1]  Encounter rate vs. time …”)

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

valid = np.isfinite(n_enc_arr)

_ax(ax1a, ylabel=r”Close encounters per snap  $\Gamma(t)$”,
title=fr”DM Close Encounter Rate  (d < {D_ENC_KPC} kpc,  probe N = {N_PROBE})”,
log_y=True)
ax1a.plot(time_enc[valid], n_enc_arr[valid], color=”#ff9944”, lw=1.8)
ax1a.fill_between(time_enc[valid], 1, n_enc_arr[valid],
alpha=0.12, color=”#ff9944”)

*ax(ax1b, xlabel=time_label,
ylabel=“Cold encounter fraction”,
title=fr”Cold Encounters  ($v*{{\rm rel}} < {V_COLD_KMS:.0f}$ km s$^{{-1}}$)”)
valid_c = valid & np.isfinite(cold_frac_arr)
ax1b.plot(time_enc[valid_c], cold_frac_arr[valid_c], color=”#4a8fff”, lw=1.8)
ax1b.fill_between(time_enc[valid_c], 0, cold_frac_arr[valid_c],
alpha=0.12, color=”#4a8fff”)
ax1b.set_ylim(0, 1.05)

fig1.savefig(os.path.join(OUT_DIR, “section27_encounter_rate.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig1)
print(”  Saved: section27_encounter_rate.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 2 — 2D ENCOUNTER DENSITY MAP AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 2]  2D encounter density maps …”)

fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.04})

for col, (ii, label) in enumerate(zip(profile_enc_ii, enc_profile_labels)):
ax = axes2[col]
ax.set_facecolor(BG)
H  = enc_maps[ii]
Hs = gaussian_filter(np.where(H > 0, H, 0.0), sigma=2.0)
H_log = np.where(Hs > 0, np.log10(Hs + 1), np.nan)

```
vals = H_log[np.isfinite(H_log)]
vmin = np.percentile(vals, 5)  if vals.size > 0 else 0
vmax = np.percentile(vals, 99) if vals.size > 0 else 5

ax.imshow(H_log.T, origin="lower", aspect="equal",
          extent=[-ENC_MAP_EXTENT, ENC_MAP_EXTENT,
                  -ENC_MAP_EXTENT, ENC_MAP_EXTENT],
          cmap="hot", vmin=vmin, vmax=vmax)
ax.set_title(label, fontsize=9, color="#c8c8e8")
ax.tick_params(colors="#9090b0", labelsize=7)
ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")
if col == 0:
    ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")
```

fig2.suptitle(r”Encounter Density Map  $\log_{10}(\Gamma_{\rm enc}(x,y) + 1)$”,
fontsize=11)
fig2.savefig(os.path.join(OUT_DIR, “section27_encounter_spatial.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig2)
print(”  Saved: section27_encounter_spatial.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 3 — RELATIVE VELOCITY DISTRIBUTION P(v_rel) AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# P(v_rel) shifts to higher velocities at pericentre as the two galaxy

# populations interpenetrate at their orbital speed (~100–200 km/s).

# The cold peak at small v_rel persists throughout — these are bound pairs

# within surviving subhalos.  The warm/hot tail grows at pericentre then

# partially recedes post-merger as the system thermalises.

print(”[Fig 3]  v_rel distributions …”)

fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor=BG)
*ax(ax3, xlabel=r”$v*{\rm rel}$ [km s$^{-1}$]”,
ylabel=“Probability density”,
title=r”Relative Velocity Distribution at Key Epochs  $P(v_{\rm rel})$”)

for ii_idx, (vrel_arr, label, color) in enumerate(zip(
vrel_dists, enc_profile_labels, PROFILE_COLORS)):
if len(vrel_arr) < 5:
continue
ax3.hist(vrel_arr, bins=50, density=True, alpha=0.55,
color=color, edgecolor=“none”, label=label)

ax3.axvline(V_COLD_KMS, color=”#ffffff”, lw=0.9, ls=”–”, alpha=0.5,
label=fr”Cold threshold {V_COLD_KMS:.0f} km s$^{{-1}}$”)
ax3.set_xlim(0, None)
ax3.legend(fontsize=7)

fig3.savefig(os.path.join(OUT_DIR, “section27_vrel_dist.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig3)
print(”  Saved: section27_vrel_dist.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 4 — ENCOUNTER RATE PROFILE Γ(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# The radial encounter rate profile shows where in the halo most close

# encounters occur.  Expected: a strong central peak (highest density)

# that broadens at pericentre as the two galaxy cores overlap.

print(”[Fig 4]  Encounter rate heatmap …”)

gamma_for_plot = np.where(gamma_radial_ts > 0, np.log10(gamma_radial_ts), np.nan)

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
gridspec_kw={“width_ratios”: [3,1], “wspace”: 0.06})

im4 = ax4a.imshow(
gamma_for_plot.T, aspect=“auto”, origin=“lower”,
extent=[t_enc_min, t_enc_max, R_BINS[0], R_BINS[-1]],
cmap=“inferno”,
)
ax4a.set_yscale(“log”)
*ax(ax4a, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Encounter Rate Profile  $\log*{10},\Gamma(r,,t)$  [enc kpc$^{-3}$]”)
cb4 = fig4.colorbar(im4, ax=ax4a, pad=0.01)
cb4.set_label(r”$\log_{10},\Gamma$”, fontsize=8)

gamma_mean = np.nanmean(gamma_radial_ts, axis=0)
valid_gm   = np.isfinite(gamma_mean) & (gamma_mean > 0)
_ax(ax4b, xlabel=r”$\langle\Gamma\rangle_t$”, title=“Time avg.”, log_x=True)
ax4b.plot(gamma_mean[valid_gm], r_mid_sph[valid_gm], color=”#ff9944”, lw=2.0)
ax4b.set_yscale(“log”)
ax4b.set_ylim(R_BINS[0], R_BINS[-1])
ax4b.tick_params(labelleft=False)

fig4.savefig(os.path.join(OUT_DIR, “section27_encounter_radius.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig4)
print(”  Saved: section27_encounter_radius.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 5 — MEAN ENERGY TRANSFER ⟨|ΔE|⟩ AND DEFLECTION ANGLE ⟨θ⟩ VS. TIME

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 5]  Energy transfer and deflection angle …”)

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

valid_dE = valid & np.isfinite(mean_dE_arr)
_ax(ax5a, ylabel=r”$\langle|\Delta E|\rangle$  [km$^2$ s$^{-2}$]”,
title=“Mean Specific Energy Transfer per Encounter”, log_y=True)
ax5a.plot(time_enc[valid_dE], mean_dE_arr[valid_dE], color=”#e8673a”, lw=1.8)

valid_th = valid & np.isfinite(mean_theta_arr)
_ax(ax5b, xlabel=time_label,
ylabel=r”$\langle\theta\rangle$  [rad]”,
title=“Mean Deflection Angle per Encounter”)
ax5b.plot(time_enc[valid_th],
np.degrees(mean_theta_arr[valid_th]),
color=”#aa55ff”, lw=1.8)
ax5b.set_ylabel(r”$\langle\theta\rangle$  [deg]”, fontsize=9, color=”#c8c8e8”)

fig5.savefig(os.path.join(OUT_DIR, “section27_energy_transfer.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5)
print(”  Saved: section27_energy_transfer.png”)

# Deflection angle standalone.

fig5b, ax5b2 = plt.subplots(figsize=(10, 4), facecolor=BG)
_ax(ax5b2, xlabel=time_label,
ylabel=r”$\langle\theta\rangle$  [deg]”,
title=“Mean Deflection Angle per Encounter”)
ax5b2.plot(time_enc[valid_th], np.degrees(mean_theta_arr[valid_th]),
color=”#aa55ff”, lw=1.8)
fig5b.savefig(os.path.join(OUT_DIR, “section27_deflection_angle.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5b)
print(”  Saved: section27_deflection_angle.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 6 — TWO-BODY RELAXATION TIME t_relax(r) AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# t_relax(r) shows how long it takes for two-body scattering to significantly

# alter particle orbits at each radius.  For a real galaxy t_relax >> t_Hubble

# everywhere, confirming the collisionless approximation is valid.

# In this simulation t_relax may be artificially short in the inner halo due

# to the finite particle mass — this is the N-body “two-body relaxation” artefact

# and is worth quantifying.

print(”[Fig 6]  Two-body relaxation time profiles …”)

fig6, ax6 = plt.subplots(figsize=(9, 6), facecolor=BG)
*ax(ax6, xlabel=“r [kpc]”, ylabel=r”$t*{\rm relax}$ [Gyr]”,
title=r”Two-Body Relaxation Timescale  $t_{\rm relax}(r) = 0.34,\sigma^3/(G^2 m \rho \ln\Lambda)$”,
log_x=True, log_y=True)

for ii_idx, (ii, label, color) in enumerate(zip(
profile_enc_ii, enc_profile_labels, PROFILE_COLORS)):
y     = t_relax_ts[ii, :]
valid = np.isfinite(y) & (y > 0)
if valid.any():
ax6.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

# Reference lines: Hubble time and MW crossing time.

ax6.axhline(13.8, color=”#ffcc44”, lw=0.9, ls=”–”, alpha=0.6,
label=“Hubble time 13.8 Gyr”)
ax6.axhline(1.0,  color=”#ffffff”, lw=0.7, ls=”:”,  alpha=0.4,
label=“1 Gyr reference”)

ax6.set_xlim(R_BINS[0], R_BINS[-1])
ax6.legend(fontsize=7)

fig6.savefig(os.path.join(OUT_DIR, “section27_two_body_relaxation.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig6)
print(”  Saved: section27_two_body_relaxation.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.5 — ANIMATION                                                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Three-panel animation:

# Left  : 2D encounter density map

# Centre: Γ(t) running history

# Right : P(v_rel) histogram for the current snapshot

print(”\n[Anim]  Encounter density animation …”)

fig_a, (ax_map, ax_hist, ax_pv) = plt.subplots(
1, 3, figsize=(16, 5.5), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 2, 2], “wspace”: 0.12},
)
for ax in (ax_map, ax_hist, ax_pv):
ax.set_facecolor(BG)

ax_map.set_xlim(-ENC_MAP_EXTENT, ENC_MAP_EXTENT)
ax_map.set_ylim(-ENC_MAP_EXTENT, ENC_MAP_EXTENT)
ax_map.set_xlabel(“x [kpc]”, color=”#c8c8e8”)
ax_map.set_ylabel(“y [kpc]”, color=”#c8c8e8”)

ax_hist.set_xlim(t_enc_min, t_enc_max)
n_max = np.nanmax(n_enc_arr) * 1.15 if np.isfinite(np.nanmax(n_enc_arr)) else 1
ax_hist.set_ylim(1, n_max)
ax_hist.set_yscale(“log”)
ax_hist.set_xlabel(time_label, color=”#c8c8e8”)
ax_hist.set_ylabel(r”$\Gamma(t)$”, color=”#c8c8e8”)
ax_hist.set_title(“Encounter rate”, color=”#c8c8e8”, fontsize=9)

ax_pv.set_xlim(0, 500)
ax_pv.set_xlabel(r”$v_{\rm rel}$ [km/s]”, color=”#c8c8e8”)
ax_pv.set_title(r”$P(v_{\rm rel})$”, color=”#c8c8e8”, fontsize=9)

# Initialise.

all_H = enc_maps[enc_maps > 0]
vmin_a = np.log10(np.percentile(all_H, 10) + 1) if all_H.size > 0 else 0
vmax_a = np.log10(np.percentile(all_H, 99) + 1) if all_H.size > 0 else 5

H0     = enc_maps[0]
Hs0    = gaussian_filter(np.where(H0>0, H0, 0.0), sigma=2.0)
H0_log = np.where(Hs0>0, np.log10(Hs0+1), np.nan)

im_map = ax_map.imshow(H0_log.T, origin=“lower”, aspect=“equal”,
extent=[-ENC_MAP_EXTENT, ENC_MAP_EXTENT,
-ENC_MAP_EXTENT, ENC_MAP_EXTENT],
cmap=“hot”, vmin=vmin_a, vmax=vmax_a)
cnt_line, = ax_hist.plot([], [], color=”#ff9944”, lw=1.8)

# PV histogram bars.

pv_bins  = np.linspace(0, 500, 41)
pv_cents = 0.5*(pv_bins[:-1]+pv_bins[1:])
bar_widths = np.diff(pv_bins)
pv_bars  = ax_pv.bar(pv_cents, np.zeros(40), width=bar_widths,
color=”#4a8fff”, alpha=0.7, edgecolor=“none”)

title_a = fig_a.suptitle(””, fontsize=11, color=”#c8c8e8”)

N_ANIM_FRAMES = n_enc_snaps

def _update_enc_anim(frame_idx):
# Map.
H  = enc_maps[frame_idx]
Hs = gaussian_filter(np.where(H>0, H, 0.0), sigma=2.0)
H_log = np.where(Hs>0, np.log10(Hs+1), np.nan)
im_map.set_data(H_log.T)

```
# Rate history.
valid_f = np.isfinite(time_enc[:frame_idx+1]) & np.isfinite(n_enc_arr[:frame_idx+1])
cnt_line.set_data(time_enc[:frame_idx+1][valid_f],
                  n_enc_arr[:frame_idx+1][valid_f])

# v_rel histogram.
snap_num = enc_snap_nums[frame_idx]
if frame_idx < len(vrel_dists) and len(vrel_dists[frame_idx]) > 5:
    counts, _ = np.histogram(vrel_dists[frame_idx], bins=pv_bins, density=True)
else:
    counts = np.zeros(40)
for bar, h in zip(pv_bars, counts):
    bar.set_height(h)
ax_pv.set_ylim(0, max(counts.max() * 1.15, 1e-5))

t_val = time_enc[frame_idx]
t_str = (f"{t_val:.2f} Gyr" if (np.isfinite(t_val) and time_is_gyr)
         else f"Snap {snap_num}")
n_str = f"N_enc={int(n_enc_arr[frame_idx])}" \
        if np.isfinite(n_enc_arr[frame_idx]) else ""
title_a.set_text(f"Close Encounters  ·  {t_str}  ·  {n_str}")

return [im_map, cnt_line] + list(pv_bars)
```

ani_enc = animation.FuncAnimation(
fig_a, _update_enc_anim, frames=N_ANIM_FRAMES,
interval=1000 // ANIM_FPS_27, blit=True,
)
writer_27 = animation.FFMpegWriter(
fps=ANIM_FPS_27, bitrate=ANIM_BITRATE_27,
metadata=dict(title=“MW-M31 DM Close Encounter Animation”),
)
ani_enc.save(os.path.join(OUT_DIR, “section27_animation_encounters.mp4”),
writer=writer_27, dpi=ANIM_DPI_27)
plt.close(fig_a)
print(”  Saved: section27_animation_encounters.mp4”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.6 — MASTER SUMMARY PANEL                                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n[Summary]  Master summary panel …”)

fig_s = plt.figure(figsize=(16, 10), facecolor=BG)
gs_s  = gridspec.GridSpec(2, 2, figure=fig_s,
hspace=0.38, wspace=0.32,
left=0.08, right=0.97,
top=0.93, bottom=0.07)

# (0,0) Encounter rate.

ax_s00 = fig_s.add_subplot(gs_s[0, 0])
_ax(ax_s00, xlabel=time_label,
ylabel=r”$\Gamma(t)$”, title=“Encounter Rate”, log_y=True)
valid = np.isfinite(n_enc_arr)
if valid.any():
ax_s00.plot(time_enc[valid], n_enc_arr[valid], color=”#ff9944”, lw=1.8)

# (0,1) Γ(r,t) heatmap.

ax_s01 = fig_s.add_subplot(gs_s[0, 1])
im_s01 = ax_s01.imshow(gamma_for_plot.T, aspect=“auto”, origin=“lower”,
extent=[t_enc_min, t_enc_max, R_BINS[0], R_BINS[-1]],
cmap=“inferno”)
ax_s01.set_yscale(“log”)
*ax(ax_s01, xlabel=time_label, ylabel=“r [kpc]”,
title=r”$\log*{10},\Gamma(r,t)$”)
fig_s.colorbar(im_s01, ax=ax_s01, shrink=0.8)

# (1,0) Mean energy transfer.

ax_s10 = fig_s.add_subplot(gs_s[1, 0])
_ax(ax_s10, xlabel=time_label,
ylabel=r”$\langle|\Delta E|\rangle$ [km$^2$s$^{-2}$]”,
title=“Mean Energy Transfer”, log_y=True)
valid_dE = valid & np.isfinite(mean_dE_arr)
if valid_dE.any():
ax_s10.plot(time_enc[valid_dE], mean_dE_arr[valid_dE],
color=”#e8673a”, lw=1.8)

# (1,1) t_relax profiles.

ax_s11 = fig_s.add_subplot(gs_s[1, 1])
*ax(ax_s11, xlabel=“r [kpc]”, ylabel=r”$t*{\rm relax}$ [Gyr]”,
title=“Two-Body Relaxation Time”, log_x=True, log_y=True)
for ii_idx, (ii, color) in enumerate(zip(profile_enc_ii, PROFILE_COLORS)):
y     = t_relax_ts[ii, :]
valid_r = np.isfinite(y) & (y > 0)
if valid_r.any():
ax_s11.plot(r_mid_sph[valid_r], y[valid_r], color=color, lw=1.5)
ax_s11.axhline(13.8, color=”#ffcc44”, lw=0.8, ls=”–”, alpha=0.5)

fig_s.suptitle(“Section 27 Summary  ·  DM Close Encounters & Collision Tracking”,
fontsize=13, color=”#c8c8e8”, fontweight=“bold”)
fig_s.savefig(os.path.join(OUT_DIR, “section27_summary_panel.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig_s)
print(”  Saved: section27_summary_panel.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §27.7 — SECTION COMPLETE                                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  SECTION 27 COMPLETE”)
print(”=”*80)
outputs_27 = [
“section27_encounter_rate.png”,
“section27_encounter_spatial.png”,
“section27_vrel_dist.png”,
“section27_encounter_radius.png”,
“section27_energy_transfer.png”,
“section27_deflection_angle.png”,
“section27_two_body_relaxation.png”,
“section27_animation_encounters.mp4”,
“section27_summary_panel.png”,
]
for fn in outputs_27:
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
kind = “animation” if fn.endswith(”.mp4”) else “figure”
print(f”  {fn:<50} {size:6.2f} MB  [{kind}]”)
print(”=”*80)
