# “””

# SECTION 21 — ANGULAR MOMENTUM TRANSPORT & PHASE-SPACE DIAGNOSTICS

Author  : Abhinav Vatsa
Date    : March 2026
Runtime : ~10 hours on a standard workstation (CPU-only)

Continuation of density_pipeline.py.  All globals (SNAPSHOTS, ns, R_BINS,
nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL, PROFILE_INDICES,
PROFILE_LABELS, PROFILE_COLORS, load_snapshot_particles) are inherited from
that script and must be defined before this section is executed.

## Physical motivation

In the isolated limit angular momentum is conserved globally.  In a live
merger the time-varying gravitational potential continuously redistributes
angular momentum between radii, between the two galaxy components, and into
the external tidal field.  Tracking this redistribution reveals:

(A) When the disk angular momentum is stripped into the halo
(B) The direction of radial transport  (inward vs. outward)
(C) How quickly the remnant virialises toward a new equilibrium
(D) The epoch at which phase-space structure is erased by mixing

This section implements five diagnostic families, each producing its own
set of storage arrays, figures, and animations.  The families are:

§21.1  Specific angular momentum profile j(r, t)
§21.2  Radial angular momentum flux  ⟨j v_r⟩(r, t)
§21.3  Cumulative enclosed angular momentum L(<r, t)
§21.4  Component decomposition: j_z, j_⊥, orbital circularity ε
§21.5  Phase-space entropy and mixing diagnostics

## Outputs

section21_j_heatmap.png            j(r,t) evolution heatmap
section21_j_flux_heatmap.png       ⟨j v_r⟩(r,t) transport heatmap
section21_L_enclosed_profiles.png  L(<r) at 5 epochs
section21_j_inner_outer.png        Inner vs. outer j scalar time series
section21_j_transport_rate.png     |∂j/∂t| transport intensity vs. time
section21_component_profiles.png   j_z, j_⊥, ε profiles at 5 epochs
section21_jz_heatmap.png           j_z(r,t) heatmap (prograde vs. retrograde)
section21_circularity_heatmap.png  ε(r,t) orbital circularity heatmap
section21_circularity_dist.png     ε distribution histograms at 5 epochs
section21_entropy.png              Phase-space entropy S(r,j) vs. time
section21_entropy_decomposed.png   Entropy by MW / M31 component vs. time
section21_mixing_length.png        Characteristic mixing length λ_mix vs. time
section21_phasespace_snap.png      (r, j) phase-space scatter at 5 epochs
section21_jflux_profiles.png       ⟨j v_r⟩(r) profiles at 5 epochs
section21_transport_budget.png     Angular momentum budget table figure
section21_animation_j.mp4         j(r) profile animation with ghost history
section21_animation_phasespace.mp4 2D (r,j) phase-space density animation
section21_summary_panel.png        Master 6-panel summary figure

===============================================================================
“””

import numpy as np
import matplotlib
matplotlib.use(“Agg”)
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

# ║  §21.0 — SECTION CONFIGURATION                                             ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# These constants are local to Section 21 and do not conflict with the

# configuration block in density_pipeline.py.

# ── Radial binning (inherited) ─────────────────────────────────────────────────

# R_BINS, r_mid_sph, nb_sph are already defined by the parent pipeline.

# ── j-space binning for 2D phase-space histograms ─────────────────────────────

# Specific angular momentum range in kpc km/s.

# A typical halo particle at r = 100 kpc, v = 150 km/s → j ~ 15000 kpc km/s.

J_MIN_KPC_KMS = 0.0
J_MAX_KPC_KMS = 30000.0
N_JBINS       = 60     # bins along the j axis in 2D phase-space histograms

# ── Orbital circularity binning ────────────────────────────────────────────────

# ε = j_z / j_c(E), where j_c(E) is the circular orbit j at the same energy.

# ε ∈ [−1, 1]: prograde (ε > 0), retrograde (ε < 0), radial (ε ≈ 0).

N_EPS_BINS = 60

# ── Inner / outer radial thresholds for scalar time series ────────────────────

J_INNER_KPC   = 30.0    # kpc — “inner halo” threshold for scalar averages
J_OUTER_KPC   = 150.0   # kpc — “outer halo” threshold

# ── Transport intensity: radius cap for |∂j/∂t| diagnostic ────────────────────

J_TRANSPORT_RMAX = 100.0   # kpc

# ── Entropy histogram grid ─────────────────────────────────────────────────────

ENTROPY_RBINS = 50   # bins in r for the (r, j) entropy histogram
ENTROPY_JBINS = 50   # bins in j for the (r, j) entropy histogram

# ── Temporal subsampling for the phase-space animation ────────────────────────

PHASESPACE_ANIM_STEP = 8    # render every Nth snapshot in the 2D animation

# ── Animation settings (inherited from parent pipeline) ───────────────────────

ANIM_FPS_21     = 20
ANIM_DPI_21     = 100
ANIM_BITRATE_21 = 2000
J_ANIM_STEP     = 4         # profile animation: every Nth snapshot

# ── Ghost-line history depth ───────────────────────────────────────────────────

N_GHOST = 20    # past profiles shown as faded lines in the j(r) animation

print(”\n” + “=”*80)
print(”  SECTION 21 · Angular Momentum Transport & Phase-Space Diagnostics”)
print(”=”*80)
print(f”  Radial bins   : {nb_sph}”)
print(f”  j range       : {J_MIN_KPC_KMS:.0f} – {J_MAX_KPC_KMS:.0f}  kpc km/s”)
print(f”  Entropy grid  : {ENTROPY_RBINS} × {ENTROPY_JBINS}”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.1 — CORE UTILITY FUNCTIONS                                            ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_angular_momentum(pos: np.ndarray,
vel: np.ndarray) -> tuple[np.ndarray, np.ndarray,
np.ndarray, np.ndarray,
np.ndarray]:
“””
Compute per-particle angular momentum components from position and velocity.

```
Parameters
----------
pos : (N, 3)  float64 — positions  [kpc]
vel : (N, 3)  float64 — velocities [km/s]

Returns
-------
j_vec  : (N, 3)  — full angular momentum vector  r × v  [kpc km/s]
j_mag  : (N,)    — |j|  total specific angular momentum
j_z    : (N,)    — z-component  j_z = x v_y − y v_x  (disk rotation axis)
j_perp : (N,)    — magnitude of transverse component  sqrt(j_x² + j_y²)
j_xy   : (N,)    — projected angular momentum in the x–y plane (= |j_perp|)

Notes
-----
j_z > 0  → prograde rotation (same sense as the initial disk)
j_z < 0  → retrograde
j_perp   → orbits tilted out of the x–y plane (vertical structure)

The decomposition j_z + j_perp is conserved shell-by-shell in a spherical
potential.  In a non-spherical merger potential they exchange with each
other, so tracking both components separately reveals the growth of
out-of-plane orbital structure.
"""
j_vec  = np.cross(pos, vel)              # (N, 3)
j_mag  = np.linalg.norm(j_vec, axis=1)  # (N,)
j_z    = j_vec[:, 2]                     # (N,)  — z-component
j_perp = np.sqrt(j_vec[:, 0]**2 + j_vec[:, 1]**2)  # (N,)
j_xy   = j_perp                          # alias for clarity in some contexts
return j_vec, j_mag, j_z, j_perp, j_xy
```

def mass_weighted_bin(values: np.ndarray,
r:      np.ndarray,
m:      np.ndarray,
r_bins: np.ndarray) -> np.ndarray:
“””
Mass-weighted mean of `values` in each radial shell.

```
This is the primary aggregation utility reused across every diagnostic
in Section 21.  It is identical in spirit to radial_bin_aggregate in
the density engine but returns NaN (not zero) for sparse bins, which is
critical for log-scale plotting.

Parameters
----------
values : (N,)   — per-particle quantity to aggregate
r      : (N,)   — 3D radii  [kpc]
m      : (N,)   — particle masses  [M_sun]
r_bins : (nb+1,) — bin edges

Returns
-------
prof : (nb,)  — NaN where bin has fewer than MIN_PART_SHELL particles
"""
nb     = len(r_bins) - 1
prof   = np.full(nb, np.nan)
bin_id = np.digitize(r, r_bins) - 1

for b in range(nb):
    mask = bin_id == b
    if mask.sum() < MIN_PART_SHELL:
        continue
    w       = m[mask]
    prof[b] = np.sum(w * values[mask]) / np.sum(w)

return prof
```

def mass_weighted_std_bin(values: np.ndarray,
r:      np.ndarray,
m:      np.ndarray,
r_bins: np.ndarray) -> np.ndarray:
“””
Mass-weighted standard deviation of `values` in each radial shell.

```
Used to compute the *dispersion* of j within a shell — a measure of
how mixed the angular momentum distribution is at a given radius.
Large σ_j means the shell contains particles on a wide range of orbits
(well mixed); small σ_j means coherent rotation survives there.

Returns
-------
std_prof : (nb,)
"""
nb     = len(r_bins) - 1
std_prof = np.full(nb, np.nan)
bin_id   = np.digitize(r, r_bins) - 1

for b in range(nb):
    mask = bin_id == b
    if mask.sum() < MIN_PART_SHELL:
        continue
    w    = m[mask]
    W    = w.sum()
    mean = np.sum(w * values[mask]) / W
    std_prof[b] = np.sqrt(np.sum(w * (values[mask] - mean)**2) / W)

return std_prof
```

def compute_orbital_circularity(j_z:    np.ndarray,
j_mag:  np.ndarray,
r_mag:  np.ndarray,
m_msun: np.ndarray,
vc_arr: np.ndarray,
r_bins: np.ndarray) -> np.ndarray:
“””
Compute per-particle orbital circularity ε = j_z / j_c(r).

```
The circular orbit specific angular momentum at radius r is:
    j_c(r) = v_c(r) × r
where v_c(r) is the circular velocity at that radius, approximated from
the shell-by-shell mean tangential speed vc_arr.

ε interpretation:
    ε = +1  →  perfect prograde circular orbit
    ε =  0  →  radial orbit (zero net angular momentum)
    ε = −1  →  perfect retrograde circular orbit

Parameters
----------
j_z    : (N,)    — z-component of angular momentum  [kpc km/s]
j_mag  : (N,)    — total |j|
r_mag  : (N,)    — 3D radius  [kpc]
m_msun : (N,)    — masses  [M_sun]
vc_arr : (nb,)   — circular velocity profile v_c(r)  [km/s]
r_bins : (nb+1,) — bin edges  [kpc]

Returns
-------
epsilon : (N,)   — orbital circularity per particle
"""
# Build a per-particle circular velocity by looking up which bin each
# particle falls in and reading the shell v_c.
bin_id = np.digitize(r_mag, r_bins) - 1
bin_id = np.clip(bin_id, 0, len(vc_arr) - 1)

# j_c(r) = v_c(r) * r.  Guard against zero radius or NaN v_c.
vc_particle = np.where(np.isfinite(vc_arr[bin_id]), vc_arr[bin_id], np.nan)
j_c         = vc_particle * r_mag                     # [kpc km/s]

with np.errstate(invalid="ignore", divide="ignore"):
    epsilon = np.where(j_c > 0, j_z / j_c, np.nan)

return epsilon
```

def compute_circular_velocity_profile(m_msun: np.ndarray,
r_mag:  np.ndarray,
r_bins: np.ndarray) -> np.ndarray:
“””
Compute v_c(r) = sqrt(G M_enc(r) / r) from enclosed mass.

```
This is an internal helper used within Section 21 to compute j_c(r)
for the circularity diagnostic.  It mirrors the logic in §14 of the
kinematics pipeline but operates on the already-loaded particle arrays
rather than the stored menc_ts array.

Returns
-------
vc : (nb,)  [km/s]
"""
nb    = len(r_bins) - 1
r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
vc    = np.full(nb, np.nan)

for b in range(nb):
    r_outer = r_bins[b + 1]
    M_encl  = m_msun[r_mag <= r_outer].sum()
    if r_outer > 0 and M_encl > 0:
        vc[b] = np.sqrt(G_KPC_KMS2_MSUN * M_encl / r_outer)

return vc
```

def phase_space_entropy(r_mag:   np.ndarray,
j_mag:   np.ndarray,
r_bins:  int = ENTROPY_RBINS,
j_bins:  int = ENTROPY_JBINS,
r_range: tuple = (0.1, 400.0),
j_range: tuple = (J_MIN_KPC_KMS, J_MAX_KPC_KMS)
) -> float:
“””
Compute the Shannon entropy of the (r, j) phase-space distribution.

```
S = − Σ_i p_i log(p_i)   where p_i = H_i / Σ H

H is the 2D histogram of particles in the (r, j) plane.  High entropy
means the phase-space distribution is broad and nearly uniform (well
mixed).  Low entropy means particles cluster tightly in (r, j) space
(coherent structure survives — e.g., tidal streams, disk orbits).

The entropy is computed in linear r-space (not log) so that equal-volume
shells contribute equally to the entropy budget.  A log-r binning would
give disproportionate weight to the inner halo where particle counts are
highest regardless of mixing state.

Parameters
----------
r_mag   : (N,)  — 3D radii      [kpc]
j_mag   : (N,)  — |j|           [kpc km/s]
r_bins  : int   — number of r bins in the histogram
j_bins  : int   — number of j bins in the histogram
r_range : (rmin, rmax)
j_range : (jmin, jmax)

Returns
-------
S : float  — Shannon entropy in nats
"""
H, _, _ = np.histogram2d(
    r_mag, j_mag,
    bins=[r_bins, j_bins],
    range=[r_range, j_range],
)
# Normalise to a probability distribution.
total = H.sum()
if total == 0:
    return np.nan
P = H / total

# Shannon entropy: treat p=0 cells as contributing zero (0 log 0 = 0).
with np.errstate(divide="ignore", invalid="ignore"):
    S = -np.nansum(np.where(P > 0, P * np.log(P), 0.0))

return float(S)
```

def compute_mixing_length(r_mag:   np.ndarray,
j_mag:   np.ndarray,
m_msun:  np.ndarray,
r_bins:  np.ndarray) -> float:
“””
Compute a characteristic mixing length λ_mix from the radial correlation
of specific angular momentum.

```
λ_mix is defined as the decorrelation length of the j(r) field:
    λ_mix = ∫ C(Δr) d(Δr)  (truncated at first zero crossing)

where C(Δr) = Corr(j(r), j(r+Δr)) is the radial autocorrelation of the
mass-weighted j profile.

Physical interpretation:
    Small λ_mix  →  j changes rapidly with r (mixed / disrupted)
    Large λ_mix  →  j is correlated over many kpc (coherent structure)

For a purely coherent disk, λ_mix would be comparable to the disk scale
length (~3–5 kpc).  After complete mixing it approaches the halo scale
radius (~30 kpc).

Parameters
----------
r_mag, j_mag, m_msun : per-particle arrays
r_bins               : radial bin edges

Returns
-------
lambda_mix : float  [kpc]  — NaN if the correlation function does not
                              cross zero within the radial range
"""
j_prof = mass_weighted_bin(j_mag, r_mag, m_msun, r_bins)
valid  = np.isfinite(j_prof)

if valid.sum() < 4:
    return np.nan

j_v    = j_prof[valid]
r_v    = r_mid_sph[valid]

# Subtract mean and normalise.
j_cent = j_v - j_v.mean()
norm   = np.dot(j_cent, j_cent)
if norm == 0:
    return np.nan

# Compute autocorrelation at each lag.
nb_v   = len(j_v)
max_lag = nb_v // 2
C = np.array([
    np.dot(j_cent[:nb_v - lag], j_cent[lag:]) / norm
    for lag in range(1, max_lag + 1)
])
dr_lags = np.array([r_v[lag] - r_v[0] for lag in range(1, max_lag + 1)])

# Find first zero crossing.
sign_changes = np.where(np.diff(np.sign(C)))[0]
if len(sign_changes) == 0:
    return np.nan

# Interpolate to the exact zero crossing.
idx = sign_changes[0]
dr_zero = dr_lags[idx] + (dr_lags[idx+1] - dr_lags[idx]) * (
    -C[idx] / (C[idx+1] - C[idx] + 1e-30)
)

return float(dr_zero)
```

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.2 — PRE-ALLOCATION OF ALL TIME-SERIES ARRAYS                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Every per-snapshot, per-bin array is allocated here so the loop body below

# is purely assignment.  NaN fill ensures missing snapshots appear as gaps

# in all figures rather than spurious zeros.

# ── j(r, t): specific angular momentum profile ────────────────────────────────

j_ts        = np.full((ns, nb_sph), np.nan)   # mean |j| per shell
j_std_ts    = np.full((ns, nb_sph), np.nan)   # j dispersion per shell

# ── Component profiles ────────────────────────────────────────────────────────

jz_ts       = np.full((ns, nb_sph), np.nan)   # mean j_z per shell (signed)
jperp_ts    = np.full((ns, nb_sph), np.nan)   # mean j_⊥ per shell

# ── Orbital circularity profile ε(r, t) ──────────────────────────────────────

eps_ts      = np.full((ns, nb_sph), np.nan)   # mean circularity per shell
eps_std_ts  = np.full((ns, nb_sph), np.nan)   # circularity dispersion

# ── Angular momentum flux ⟨j v_r⟩(r, t) ─────────────────────────────────────

jflux_ts    = np.full((ns, nb_sph), np.nan)

# ── Cumulative enclosed angular momentum L(<r) ────────────────────────────────

L_enc_ts    = np.full((ns, nb_sph), np.nan)   # [M_sun kpc km/s]

# ── Scalar time series (one number per snapshot) ──────────────────────────────

j_inner_arr   = np.full(ns, np.nan)   # mean j for r < J_INNER_KPC
j_outer_arr   = np.full(ns, np.nan)   # mean j for r > J_OUTER_KPC
j_total_arr   = np.full(ns, np.nan)   # global mass-weighted j
L_total_arr   = np.full(ns, np.nan)   # total enclosed angular momentum
entropy_arr   = np.full(ns, np.nan)   # phase-space entropy (all particles)
entropy_mw_arr  = np.full(ns, np.nan) # entropy of MW-origin particles only
entropy_m31_arr = np.full(ns, np.nan) # entropy of M31-origin particles only
mix_length_arr  = np.full(ns, np.nan) # mixing length λ_mix [kpc]
transport_arr   = np.full(ns, np.nan) # |∂j/∂t| averaged over r < J_TRANSPORT_RMAX

# Storage for the 2D phase-space density histograms used in the animation.

# Shape: (n_phasespace_frames, ENTROPY_RBINS, ENTROPY_JBINS)

phasespace_snap_ids = np.arange(0, ns, PHASESPACE_ANIM_STEP)
n_ps_frames         = len(phasespace_snap_ids)
phasespace_hists    = np.zeros((n_ps_frames, ENTROPY_RBINS, ENTROPY_JBINS))

print(f”\n[Pre-alloc] j_ts:          {j_ts.shape}”)
print(f”            L_enc_ts:      {L_enc_ts.shape}”)
print(f”            Phase-space frames: {n_ps_frames}”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.3 — MAIN LOOP                                                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# For each snapshot we:

# 1. Load particle data (positions, velocities, masses, origin tags)

# 2. Compute COM-frame coordinates (reusing load_snapshot_particles)

# 3. Compute angular momentum components

# 4. Compute the circular velocity profile for circularity ε

# 5. Aggregate into all radial-bin arrays

# 6. Compute global scalar quantities

# 7. If this is a phase-space animation frame, store the 2D histogram

print(”\n” + “=”*80)
print(”  §21.3 — Main Snapshot Loop”)
print(”=”*80)

t_loop_start = time.perf_counter()

# Map from snapshot number to phase-space frame index.

ps_frame_map = {SNAPSHOTS[idx]: fi
for fi, idx in enumerate(phasespace_snap_ids)}

for i, snap_num in enumerate(SNAPSHOTS):

```
mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
    continue

t_snap = time.perf_counter()

try:
    snap_data = load_snapshot_particles(mw_file, m31_file)
except Exception as exc:
    print(f"  [ERROR] snap {snap_num}: {exc}")
    continue

pos    = snap_data["pos"]       # (N, 3)  [kpc]   COM-centred
m      = snap_data["m_msun"]    # (N,)    [M_sun]
origin = snap_data["origin"]    # (N,)    0=MW, 1=M31

# ── 3D radii and radial velocities ────────────────────────────────────────
r_mag = np.linalg.norm(pos, axis=1)

# We need velocities relative to COM.  load_snapshot_particles returns
# positions already in the COM frame.  We reconstruct velocities here
# by re-loading — this matches the pattern used in §17 of the kinematics
# pipeline where _sigma_r_for_galaxy accepted raw particle arrays.
#
# For efficiency, we read velocities from the stored MW/M31 objects.
# If the parent pipeline's load_snapshot_particles stored velocities,
# we would use those directly.  Since it does not, we reload here.
try:
    MW_obj  = CenterOfMass(mw_file,  PTYPE)
    M31_obj = CenterOfMass(m31_file, PTYPE)
except Exception as exc:
    print(f"  [WARN] snap {snap_num}: CenterOfMass reload failed ({exc})")
    continue

vx = np.concatenate((MW_obj.vx, M31_obj.vx))
vy = np.concatenate((MW_obj.vy, M31_obj.vy))
vz = np.concatenate((MW_obj.vz, M31_obj.vz))

# COM velocity for this snapshot (reuse fallback logic from §3 of kinematics).
m_raw = np.concatenate((MW_obj.m, M31_obj.m))
x_all = np.concatenate((MW_obj.x, M31_obj.x))
y_all = np.concatenate((MW_obj.y, M31_obj.y))
z_all = np.concatenate((MW_obj.z, M31_obj.z))
xcom, ycom, zcom = MW_obj.COMdefine(x_all, y_all, z_all, m_raw)

# Inner-sphere mass-weighted COM velocity.
dr_com = np.sqrt((x_all - xcom)**2 + (y_all - ycom)**2 + (z_all - zcom)**2)
inner  = dr_com < 15.0
if inner.sum() >= 5:
    w_inner  = m[inner]
    vxcom = np.sum(w_inner * vx[inner]) / w_inner.sum()
    vycom = np.sum(w_inner * vy[inner]) / w_inner.sum()
    vzcom = np.sum(w_inner * vz[inner]) / w_inner.sum()
else:
    vxcom = vycom = vzcom = 0.0

vel = np.vstack((vx - vxcom, vy - vycom, vz - vzcom)).T   # (N, 3)

# ── Angular momentum components ───────────────────────────────────────────
j_vec, j_mag, j_z, j_perp, _ = compute_angular_momentum(pos, vel)

# ── Radial velocity (for flux diagnostic) ─────────────────────────────────
with np.errstate(divide="ignore", invalid="ignore"):
    r_hat = np.where(r_mag[:, None] > 0, pos / r_mag[:, None], 0.0)
v_r = np.einsum("ij,ij->i", vel, r_hat)       # (N,)  [km/s]

# ── Angular momentum flux proxy ⟨j v_r⟩ ──────────────────────────────────
# Positive flux → angular momentum being transported outward.
# Negative flux → angular momentum flowing inward.
j_flux = j_mag * v_r   # (N,)  [kpc km/s × km/s = kpc km²/s²]

# ── Circular velocity profile (for ε) ────────────────────────────────────
vc_snap = compute_circular_velocity_profile(m, r_mag, R_BINS)

# ── Orbital circularity ε per particle ────────────────────────────────────
epsilon = compute_orbital_circularity(j_z, j_mag, r_mag, m, vc_snap, R_BINS)

# ── Radial aggregation ────────────────────────────────────────────────────
j_ts     [i, :] = mass_weighted_bin(j_mag,  r_mag, m, R_BINS)
j_std_ts [i, :] = mass_weighted_std_bin(j_mag, r_mag, m, R_BINS)
jz_ts    [i, :] = mass_weighted_bin(j_z,    r_mag, m, R_BINS)
jperp_ts [i, :] = mass_weighted_bin(j_perp, r_mag, m, R_BINS)
jflux_ts [i, :] = mass_weighted_bin(j_flux, r_mag, m, R_BINS)

valid_eps = np.isfinite(epsilon)
if valid_eps.sum() > MIN_PART_SHELL:
    eps_ts    [i, :] = mass_weighted_bin(
        np.where(valid_eps, epsilon, 0.0), r_mag, m, R_BINS)
    eps_std_ts[i, :] = mass_weighted_std_bin(
        np.where(valid_eps, epsilon, 0.0), r_mag, m, R_BINS)

# ── Cumulative enclosed angular momentum L(<r) ────────────────────────────
bin_id = np.digitize(r_mag, R_BINS) - 1
L_enc  = np.zeros(nb_sph)
for b in range(nb_sph):
    mask_enc   = bin_id <= b
    L_enc[b]   = np.sum(m[mask_enc] * j_mag[mask_enc])
L_enc_ts[i, :] = L_enc

# ── Scalar diagnostics ────────────────────────────────────────────────────
M_total = m.sum()
j_total_arr[i] = np.sum(m * j_mag) / M_total

inner_mask_j = r_mag <= J_INNER_KPC
outer_mask_j = r_mag >= J_OUTER_KPC
if inner_mask_j.sum() >= MIN_PART_SHELL:
    j_inner_arr[i] = np.sum(m[inner_mask_j] * j_mag[inner_mask_j]) / m[inner_mask_j].sum()
if outer_mask_j.sum() >= MIN_PART_SHELL:
    j_outer_arr[i] = np.sum(m[outer_mask_j] * j_mag[outer_mask_j]) / m[outer_mask_j].sum()

L_total_arr[i] = L_enc[-1]   # total enclosed j-momentum (outermost bin)

# ── Phase-space entropy ───────────────────────────────────────────────────
entropy_arr   [i] = phase_space_entropy(r_mag, j_mag)
mw_mask   = origin == 0
m31_mask  = origin == 1
if mw_mask.sum()  > 50:
    entropy_mw_arr [i] = phase_space_entropy(r_mag[mw_mask],  j_mag[mw_mask])
if m31_mask.sum() > 50:
    entropy_m31_arr[i] = phase_space_entropy(r_mag[m31_mask], j_mag[m31_mask])

# ── Mixing length ─────────────────────────────────────────────────────────
mix_length_arr[i] = compute_mixing_length(r_mag, j_mag, m, R_BINS)

# ── Phase-space 2D histogram for animation ────────────────────────────────
if snap_num in ps_frame_map:
    fi = ps_frame_map[snap_num]
    H, _, _ = np.histogram2d(
        r_mag, j_mag,
        bins=[ENTROPY_RBINS, ENTROPY_JBINS],
        range=[[0.1, 400.0], [J_MIN_KPC_KMS, J_MAX_KPC_KMS]],
        weights=m,
    )
    phasespace_hists[fi] = H

# ── Progress ──────────────────────────────────────────────────────────────
if (i + 1) % 100 == 0:
    elapsed = time.perf_counter() - t_loop_start
    print(f"  snap {snap_num:04d}  j_inner={j_inner_arr[i]:.0f} kpc km/s  "
          f"S={entropy_arr[i]:.3f}  λ_mix={mix_length_arr[i]:.1f} kpc  "
          f"[{elapsed:.0f}s]")
```

print(f”\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.4 — TEMPORAL DERIVATIVES                                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Angular momentum *transport rate* ∂j/∂t is computed via np.gradient along

# the time axis (axis=0) of the j_ts array.

# 

# Physical meaning:

# dj/dt > 0  at radius r  →  angular momentum is accumulating there

# (particles on more circular orbits arriving,

# or local injection from tidal torques)

# dj/dt < 0  at radius r  →  angular momentum is leaving

# (stripping, radial infall, or transport outward)

# 

# The gradient uses the snapshot index as the time coordinate.  To convert

# to physical units (kpc km/s / Gyr), divide by Δt per snapshot in Gyr

# (approximately 10 Myr = 0.01 Gyr per snapshot for the standard run).

dj_dt  = np.gradient(j_ts,    axis=0)   # (ns, nb_sph)
djz_dt = np.gradient(jz_ts,   axis=0)
dL_dt  = np.gradient(L_enc_ts, axis=0)

# Global transport intensity: mean |∂j/∂t| over r < J_TRANSPORT_RMAX.

inner_r_mask = r_mid_sph < J_TRANSPORT_RMAX
transport_arr = np.nanmean(np.abs(dj_dt[:, inner_r_mask]), axis=1)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.5 — FIGURES                                                           ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

t_min = np.nanmin(time_arr)
t_max = np.nanmax(time_arr)

# ── Shared style helpers ───────────────────────────────────────────────────────

BG    = “#0d0d18”
MUTED = “#7070a0”

def _styled_ax(ax, xlabel=””, ylabel=””, title=””,
log_x=False, log_y=False):
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

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 1 — j(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# The heatmap encodes the mean specific angular momentum in every radial shell

# at every snapshot.  Bright horizontal bands at fixed r mark epochs when j

# builds up locally; diagonal streaks indicate angular momentum transport

# propagating outward (or inward) at some characteristic speed.

print(”\n[Fig 1]  j(r,t) heatmap …”)

fig1, (ax1a, ax1b) = plt.subplots(
1, 2, figsize=(14, 6), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 1], “wspace”: 0.06},
)

im1 = ax1a.imshow(
j_ts.T,
aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“plasma”,
norm=LogNorm(vmin=np.nanpercentile(j_ts[j_ts > 0], 5),
vmax=np.nanpercentile(j_ts[j_ts > 0], 99)),
)
_styled_ax(ax1a,
xlabel=time_label,
ylabel=“r [kpc]”,
title=r”Specific Angular Momentum  $j(r,,t)$  [kpc km s$^{-1}$]”)
ax1a.set_yscale(“log”)
cb1 = fig1.colorbar(im1, ax=ax1a, pad=0.01)
cb1.set_label(r”$j$ [kpc km s$^{-1}$]”, fontsize=8)

# Right panel: time-averaged j(r) profile.

j_mean = np.nanmean(j_ts, axis=0)
valid  = np.isfinite(j_mean)
_styled_ax(ax1b,
xlabel=r”$\langle j \rangle_t$”,
title=“Time avg.”,
log_x=False, log_y=True)
ax1b.plot(j_mean[valid], r_mid_sph[valid], color=”#ff9944”, lw=2.0)
ax1b.set_ylim(R_BINS[0], R_BINS[-1])
ax1b.tick_params(labelleft=False)

fig1.suptitle(“Angular Momentum Profile Evolution”, fontsize=12)
fig1.savefig(os.path.join(OUT_DIR, “section21_j_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig1)
print(”  Saved: section21_j_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 2 — ANGULAR MOMENTUM FLUX ⟨j v_r⟩(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# The flux proxy ⟨j v_r⟩ has the sign of the transport direction:

# > 0  → angular momentum flowing outward (tidal stripping, expansion)

# < 0  → angular momentum flowing inward  (disk assembly, compaction)

# 

# A TwoSlopeNorm colourmap centred at zero makes the sign immediately visible.

print(”[Fig 2]  Angular momentum flux heatmap …”)

flux_max = np.nanpercentile(np.abs(jflux_ts[np.isfinite(jflux_ts)]), 97)

fig2, (ax2a, ax2b) = plt.subplots(
1, 2, figsize=(14, 6), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 1], “wspace”: 0.06},
)

im2 = ax2a.imshow(
jflux_ts.T,
aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“coolwarm”,
norm=TwoSlopeNorm(vmin=-flux_max, vcenter=0.0, vmax=flux_max),
)
_styled_ax(ax2a,
xlabel=time_label, ylabel=“r [kpc]”,
title=r”Angular Momentum Flux  $\langle j,v_r \rangle(r,,t)$”)
ax2a.set_yscale(“log”)
cb2 = fig2.colorbar(im2, ax=ax2a, pad=0.01)
cb2.set_label(r”$\langle j,v_r \rangle$  [kpc$^2$ km$^2$ s$^{-2}$]”, fontsize=8)
ax2a.axhline(J_INNER_KPC, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.3)
ax2a.text(t_min, J_INNER_KPC * 1.05, f”{J_INNER_KPC:.0f} kpc”,
color=”#9090b0”, fontsize=7)

# Right panel: time-average flux profile.

flux_mean = np.nanmean(jflux_ts, axis=0)
valid_f   = np.isfinite(flux_mean)
_styled_ax(ax2b, xlabel=r”$\langle j v_r \rangle$”, title=“Time avg.”)
ax2b.plot(flux_mean[valid_f], r_mid_sph[valid_f], color=”#00d4aa”, lw=2.0)
ax2b.axvline(0, color=”#555577”, lw=0.8, ls=”–”)
ax2b.set_yscale(“log”)
ax2b.set_ylim(R_BINS[0], R_BINS[-1])
ax2b.tick_params(labelleft=False)

fig2.suptitle(“Radial Angular Momentum Transport”, fontsize=12)
fig2.savefig(os.path.join(OUT_DIR, “section21_j_flux_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig2)
print(”  Saved: section21_j_flux_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 3 — FLUX PROFILES AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# Radial profiles of ⟨j v_r⟩(r) at the five representative epochs.

# The zero-crossing radius separates the outward-transport region from the

# inward-transport region and should migrate outward as the merger progresses.

print(”[Fig 3]  Flux profiles at 5 epochs …”)

fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor=BG)
_styled_ax(ax3, xlabel=“r [kpc]”,
ylabel=r”$\langle j,v_r \rangle$ [kpc$^2$ km$^2$ s$^{-2}$]”,
title=r”Angular Momentum Flux Profiles  $\langle j,v_r \rangle(r)$”,
log_x=True)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
y     = jflux_ts[k_idx, :]
valid = np.isfinite(y)
if valid.any():
ax3.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax3.axhline(0, color=”#555577”, lw=1.0, ls=”–”, alpha=0.6)
ax3.text(R_BINS[0] * 1.2, 0, “outward →”, color=MUTED, fontsize=7, va=“bottom”)
ax3.text(R_BINS[0] * 1.2, 0, “← inward”, color=MUTED, fontsize=7, va=“top”)
ax3.set_xlim(R_BINS[0], R_BINS[-1])
ax3.legend(fontsize=8)

fig3.savefig(os.path.join(OUT_DIR, “section21_jflux_profiles.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig3)
print(”  Saved: section21_jflux_profiles.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 4 — CUMULATIVE L(<r) PROFILES AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# L(<r) is the total angular momentum enclosed within radius r.  Its slope

# dL/dr = 4π r² ρ(r) j(r) shows where the angular momentum budget is

# concentrated.  A cusp in dL/dr at small r indicates a surviving disk

# (dense + high j); a flat dL/dr at large r indicates a spread-out tidal halo.

print(”[Fig 4]  Cumulative L(<r) profiles …”)

fig4, (ax4a, ax4b) = plt.subplots(
1, 2, figsize=(13, 5), facecolor=BG,
gridspec_kw={“wspace”: 0.32},
)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
# Absolute L(<r).
L_row  = L_enc_ts[k_idx, :]
valid  = np.isfinite(L_row) & (L_row > 0)
*styled_ax(ax4a,
xlabel=“r [kpc]”, ylabel=r”$L(<r)$ [M$*\odot$ kpc km s$^{-1}$]”,
title=r”Enclosed Angular Momentum $L(<r)$”, log_x=True, log_y=True)
if valid.any():
ax4a.plot(r_mid_sph[valid], L_row[valid], color=color, lw=2.0, label=label)

```
# Normalised L(<r) / L_total — fraction of total angular momentum inside r.
L_norm = L_row / (L_row[valid][-1] if valid.any() else 1.0)
_styled_ax(ax4b,
           xlabel="r [kpc]", ylabel=r"$L(<r)\,/\,L_{\rm tot}$",
           title=r"Fractional Angular Momentum", log_x=True)
if valid.any():
    ax4b.plot(r_mid_sph[valid], L_norm[valid], color=color, lw=2.0, label=label)
```

ax4a.legend(fontsize=7)
ax4b.axhline(0.5, color=”#555577”, lw=0.8, ls=”–”)
ax4b.text(R_BINS[-1] * 0.6, 0.52, “50%”, color=MUTED, fontsize=7)
ax4b.set_ylim(0, 1.05)

fig4.savefig(os.path.join(OUT_DIR, “section21_L_enclosed_profiles.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig4)
print(”  Saved: section21_L_enclosed_profiles.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 5 — INNER vs. OUTER j SCALAR TIME SERIES

# ══════════════════════════════════════════════════════════════════════════════

# 

# Tracking the average j inside 30 kpc and outside 150 kpc as scalar time

# series reveals angular momentum exchange between the inner and outer halo.

# When the inner j drops while the outer j rises, angular momentum is being

# transported outward — a signature of tidal torquing during pericentre.

print(”[Fig 5]  Inner vs. outer j time series …”)

fig5, (ax5a, ax5b) = plt.subplots(
2, 1, figsize=(10, 8), facecolor=BG, sharex=True,
gridspec_kw={“hspace”: 0.08},
)

_styled_ax(ax5a, ylabel=r”$j$ [kpc km s$^{-1}$]”,
title=r”Inner vs. Outer Specific Angular Momentum”)
ax5a.plot(time_arr, j_inner_arr, color=”#4a8fff”, lw=1.8,
label=fr”Inner  ($r < {J_INNER_KPC:.0f}$ kpc)”)
ax5a.plot(time_arr, j_outer_arr, color=”#ff9944”, lw=1.8,
label=fr”Outer  ($r > {J_OUTER_KPC:.0f}$ kpc)”)
ax5a.plot(time_arr, j_total_arr, color=”#aaaacc”, lw=1.0, ls=”:”,
label=“Global mean”)
ax5a.legend(fontsize=8)

_styled_ax(ax5b, xlabel=time_label,
ylabel=r”$|dj/dt|$ inner  [kpc km s$^{-1}$ / snap]”,
title=r”Transport Intensity  $\langle|\partial j/\partial t|\rangle$”)
ax5b.plot(time_arr, transport_arr, color=”#e8673a”, lw=1.5)
ax5b.fill_between(time_arr,
np.where(np.isfinite(transport_arr), transport_arr, 0),
alpha=0.15, color=”#e8673a”)

fig5.savefig(os.path.join(OUT_DIR, “section21_j_inner_outer.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5)
print(”  Saved: section21_j_inner_outer.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 6 — j_z, j_⊥, AND ε COMPONENT PROFILES AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# j_z  tracks the coherent rotation of the disk.  Before the merger it

# dominates at small r.  After disruption it falls and j_⊥ (out-of-plane

# component) may rise as orbits are scattered into inclined trajectories.

# 

# ε(r) tracks whether orbits are circular (|ε| ≈ 1) or radial (ε ≈ 0).

# A peak at ε = +1 at small r in the early snapshots reflects the intact MW

# disk.  Post-merger ε should broaden toward 0, reflecting radial orbit

# injection by the tidal field.

print(”[Fig 6]  Component profiles j_z, j_⊥, ε …”)

fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG,
gridspec_kw={“wspace”: 0.32})
ax_jz, ax_jp, ax_eps = axes6

for ax, data_ts, ylabel, title in [
(ax_jz,  jz_ts,   r”$j_z$ [kpc km s$^{-1}$]”,
r”Azimuthal $j_z(r)$  (signed)”),
(ax_jp,  jperp_ts, r”$j_\perp$ [kpc km s$^{-1}$]”,
r”Out-of-plane $j_\perp(r)$”),
(ax_eps, eps_ts,   r”Mean circularity $\varepsilon$”,
r”Orbital Circularity $\varepsilon(r)$”),
]:
_styled_ax(ax, xlabel=“r [kpc]”, ylabel=ylabel, title=title, log_x=True)
ax.set_xlim(R_BINS[0], R_BINS[-1])

```
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    y     = data_ts[k_idx, :]
    valid = np.isfinite(y)
    if valid.any():
        ax.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)
```

ax_jz.axhline(0, color=”#555577”, lw=0.8, ls=”–”)
ax_eps.axhline(0, color=”#555577”, lw=0.8, ls=”–”)
ax_eps.axhline(1, color=”#8855aa”, lw=0.7, ls=”:”)
ax_eps.set_ylim(-1.1, 1.1)
ax_eps.legend(fontsize=7)

fig6.suptitle(“Angular Momentum Component Profiles”, fontsize=12)
fig6.savefig(os.path.join(OUT_DIR, “section21_component_profiles.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig6)
print(”  Saved: section21_component_profiles.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 7 — j_z(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# j_z(r, t) distinguishes prograde-dominated shells (warm colours) from

# retrograde-dominated shells (cool colours).  A coherent disk at the centre

# should appear as a persistent warm patch at small r in the early snapshots.

# As tidal disruption proceeds this patch should fragment and cool.

print(”[Fig 7]  j_z(r,t) heatmap …”)

jz_abs_max = np.nanpercentile(np.abs(jz_ts[np.isfinite(jz_ts)]), 97)

fig7, ax7 = plt.subplots(figsize=(11, 5), facecolor=BG)
im7 = ax7.imshow(
jz_ts.T,
aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“RdBu_r”,
norm=TwoSlopeNorm(vmin=-jz_abs_max, vcenter=0.0, vmax=jz_abs_max),
)
_styled_ax(ax7, xlabel=time_label, ylabel=“r [kpc]”,
title=r”$j_z(r,,t)$  — Prograde (warm) vs. Retrograde (cool)”)
ax7.set_yscale(“log”)
cb7 = fig7.colorbar(im7, ax=ax7, pad=0.01)
cb7.set_label(r”$j_z$ [kpc km s$^{-1}$]”, fontsize=8)
cb7.ax.axhline(0, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.6)

fig7.savefig(os.path.join(OUT_DIR, “section21_jz_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig7)
print(”  Saved: section21_jz_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 8 — ORBITAL CIRCULARITY ε(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# A persistent stripe at ε ≈ +1 at small r in early snapshots is the

# signature of an intact disk.  Broadening and shift toward ε ≈ 0 marks

# disk disruption.  Re-emergence of coherent ε > 0 structure at intermediate

# r in late snapshots would suggest orbital re-circularisation in the remnant.

print(”[Fig 8]  Orbital circularity ε(r,t) heatmap …”)

fig8, ax8 = plt.subplots(figsize=(11, 5), facecolor=BG)
im8 = ax8.imshow(
np.clip(eps_ts, -1.0, 1.0).T,
aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“bwr”,
vmin=-1.0, vmax=1.0,
)
_styled_ax(ax8, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Orbital Circularity  $\varepsilon(r,,t) = j_z / j_c(r)$”)
ax8.set_yscale(“log”)
cb8 = fig8.colorbar(im8, ax=ax8, pad=0.01)
cb8.set_label(r”$\varepsilon$  (blue = retrograde, red = prograde)”, fontsize=8)

fig8.savefig(os.path.join(OUT_DIR, “section21_circularity_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig8)
print(”  Saved: section21_circularity_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 9 — CIRCULARITY DISTRIBUTION HISTOGRAMS AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# The distribution of ε across *all* particles (not binned by radius) gives

# the global orbital structure of the halo at each epoch.  A disk-dominated

# system produces a sharp peak near ε = +1.  A virialised halo produces a

# broad distribution peaked at ε ≈ 0.  The transition between these shapes

# marks the moment of disk disruption.

print(”[Fig 9]  Circularity distribution histograms …”)

fig9, axes9 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.06})
eps_bins = np.linspace(-1.05, 1.05, N_EPS_BINS + 1)

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES,
PROFILE_LABELS,
PROFILE_COLORS)):
ax = axes9[col]
_styled_ax(ax, xlabel=r”$\varepsilon$”, title=label)

```
# Retrieve stored mean circularity per bin — we need the raw per-particle
# distribution.  Because Section 21 pre-computes only mass-weighted means,
# we re-derive the histogram from the stored eps_ts standard deviation as
# a proxy for display, and note the limitation.
eps_mean = eps_ts[k_idx, :]
eps_std  = eps_std_ts[k_idx, :]
valid    = np.isfinite(eps_mean) & np.isfinite(eps_std)

# Reconstruct a synthetic Gaussian distribution per bin for display.
# This is an approximation — a full re-read would give exact histograms.
eps_plot = np.linspace(-1.1, 1.1, 300)
pdf_sum  = np.zeros(300)
for b in np.where(valid)[0]:
    mu  = eps_mean[b]
    sig = max(eps_std[b], 0.05)
    w_b = 1.0 / nb_sph
    pdf_sum += w_b * np.exp(-0.5 * ((eps_plot - mu) / sig)**2) / (sig * np.sqrt(2 * np.pi))

ax.fill_between(eps_plot, pdf_sum, alpha=0.3, color=color)
ax.plot(eps_plot, pdf_sum, color=color, lw=1.5)
ax.axvline(0, color="#555577", lw=0.7, ls="--")
ax.axvline(1, color="#8855aa", lw=0.7, ls=":")
if col == 0:
    ax.set_ylabel("Relative density", fontsize=9)
```

fig9.suptitle(
r”Orbital Circularity Distribution  $P(\varepsilon)$  at Key Epochs”
“\n(Reconstructed from per-bin moments — see notes)”,
fontsize=11,
)
fig9.savefig(os.path.join(OUT_DIR, “section21_circularity_dist.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig9)
print(”  Saved: section21_circularity_dist.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 10 — PHASE-SPACE ENTROPY vs. TIME

# ══════════════════════════════════════════════════════════════════════════════

# 

# S(t) should increase monotonically in a purely mixing system (second law

# for coarse-grained phase-space).  Dips in S mark epochs where tidal streams

# temporarily create coherent structure in (r, j) space.  The merger-induced

# peak in S marks the epoch of maximum phase-space disorder, after which

# violent relaxation and phase mixing drive S toward the virialised value.

# 

# Comparing S_MW(t) and S_M31(t) separately shows which galaxy retains

# coherent phase-space structure longer.

print(”[Fig 10]  Phase-space entropy …”)

fig10, (ax10a, ax10b) = plt.subplots(
2, 1, figsize=(10, 8), facecolor=BG, sharex=True,
gridspec_kw={“hspace”: 0.08},
)

_styled_ax(ax10a, ylabel=r”$S(r,j)$ [nats]”,
title=r”Phase-Space Entropy  $S = -\Sigma,p,\ln p$”)
ax10a.plot(time_arr, entropy_arr, color=”#e8673a”, lw=1.8, label=“All particles”)
ax10a.legend(fontsize=8)

_styled_ax(ax10b, xlabel=time_label, ylabel=r”$S$ [nats]”,
title=“Entropy by Galaxy of Origin”)
ax10b.plot(time_arr, entropy_mw_arr,  color=”#4a8fff”, lw=1.8, label=“MW only”)
ax10b.plot(time_arr, entropy_m31_arr, color=”#ff5fa0”, lw=1.8, label=“M31 only”)
ax10b.legend(fontsize=8)

fig10.savefig(os.path.join(OUT_DIR, “section21_entropy.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig10)
print(”  Saved: section21_entropy.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 11 — MIXING LENGTH λ_mix vs. TIME

# ══════════════════════════════════════════════════════════════════════════════

# 

# λ_mix is the e-folding scale of the angular momentum autocorrelation function

# in radius.  A large λ_mix means j(r) changes slowly with r — the halo

# retains long-range orbital structure.  A small λ_mix means j decorrelates

# over short distances — the phase-space structure has been scrambled.

# 

# We expect λ_mix to drop sharply at first pericentre (disruption) and then

# slowly recover as the remnant virialises onto a new equilibrium with a

# scale radius set by the merged system’s energy.

print(”[Fig 11]  Mixing length …”)

fig11, ax11 = plt.subplots(figsize=(10, 4), facecolor=BG)
*styled_ax(ax11,
xlabel=time_label, ylabel=r”$\lambda*{\rm mix}$ [kpc]”,
title=r”Angular Momentum Correlation Length  $\lambda_{\rm mix}(t)$”)
ax11.plot(time_arr, mix_length_arr, color=”#00d4aa”, lw=1.8)
ax11.fill_between(time_arr,
np.where(np.isfinite(mix_length_arr), mix_length_arr, 0),
alpha=0.12, color=”#00d4aa”)

# Reference line at MW disk scale length (~3.5 kpc).

ax11.axhline(3.5, color=”#ffcc44”, lw=0.8, ls=”–”,
label=“MW disk scale length ≈ 3.5 kpc”)
ax11.legend(fontsize=8)

fig11.savefig(os.path.join(OUT_DIR, “section21_mixing_length.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig11)
print(”  Saved: section21_mixing_length.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 12 — PHASE-SPACE SCATTER AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# A direct scatter (or binned-density) view of the (r, j) distribution at

# five epochs.  This is the most intuitive figure in the section: it shows

# exactly where in (r, j) space the mass is concentrated at each epoch.

# 

# Because individual scatter plots with 10^6 particles are unreadable, we

# show the 2D histogram (log-scaled) from the pre-computed phasespace_hists

# array.  The five epochs are selected from the stored histograms using the

# same PROFILE_INDICES used throughout the pipeline.

print(”[Fig 12]  Phase-space density scatter …”)

fig12, axes12 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.06})

# Map each profile index to the nearest stored phase-space frame.

ps_snap_nums = SNAPSHOTS[phasespace_snap_ids]

for col, (k_idx, label, color) in enumerate(zip(PROFILE_INDICES,
PROFILE_LABELS,
PROFILE_COLORS)):
ax  = axes12[col]
_styled_ax(ax, xlabel=“r [kpc]”, title=label)
if col == 0:
ax.set_ylabel(r”$j = |r \times v|$ [kpc km s$^{-1}$]”, fontsize=8)

```
target_snap = SNAPSHOTS[k_idx]
# Find the nearest stored phase-space frame.
nearest_fi  = np.argmin(np.abs(ps_snap_nums - target_snap))
H           = phasespace_hists[nearest_fi]
H_log       = np.where(H > 0, np.log10(H), np.nan)

ax.imshow(
    H_log.T,
    origin="lower", aspect="auto",
    extent=[0.1, 400.0, J_MIN_KPC_KMS, J_MAX_KPC_KMS],
    cmap="inferno",
)
ax.set_xscale("log")
ax.set_xlim(0.1, 400.0)
ax.set_ylim(J_MIN_KPC_KMS, J_MAX_KPC_KMS)
```

fig12.suptitle(r”Phase-Space Density  $(r,,j)$  at Key Epochs”,
fontsize=12)
fig12.savefig(os.path.join(OUT_DIR, “section21_phasespace_snap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig12)
print(”  Saved: section21_phasespace_snap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 13 — ANGULAR MOMENTUM BUDGET TABLE FIGURE

# ══════════════════════════════════════════════════════════════════════════════

# 

# A quantitative summary figure showing the angular momentum budget at the

# five key epochs as a formatted table embedded in a figure.  This is useful

# for paper supplements where a concise numerical comparison is needed.

# Columns: epoch, j_inner, j_outer, j_total, L_total, entropy, λ_mix.

print(”[Fig 13]  Angular momentum budget table …”)

fig13, ax13 = plt.subplots(figsize=(13, 3.5), facecolor=BG)
ax13.set_facecolor(BG)
ax13.axis(“off”)

col_headers = [“Epoch”, r”$j_{\rm inner}$”, r”$j_{\rm outer}$”,
r”$j_{\rm total}$”, r”$L_{\rm enc}$”,
r”$S$ [nats]”, r”$\lambda_{\rm mix}$ [kpc]”]
row_data = []
for k_idx, label in zip(PROFILE_INDICES, PROFILE_LABELS):
row_data.append([
label,
f”{j_inner_arr[k_idx]:.0f}”,
f”{j_outer_arr[k_idx]:.0f}”,
f”{j_total_arr[k_idx]:.0f}”,
f”{L_total_arr[k_idx]:.2e}”,
f”{entropy_arr[k_idx]:.3f}”,
f”{mix_length_arr[k_idx]:.1f}”,
])

tbl = ax13.table(
cellText=row_data,
colLabels=col_headers,
loc=“center”,
cellLoc=“center”,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)

# Style header cells.

for (r, c), cell in tbl.get_celld().items():
cell.set_facecolor(”#1a1a3a” if r == 0 else (”#0d0d18” if r % 2 == 0 else “#141428”))
cell.set_edgecolor(”#2a2a4a”)
cell.set_text_props(color=”#c8c8e8”)

ax13.set_title(“Angular Momentum Budget at Key Epochs”,
fontsize=11, color=”#c8c8e8”, pad=12)

fig13.savefig(os.path.join(OUT_DIR, “section21_transport_budget.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig13)
print(”  Saved: section21_transport_budget.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.6 — ANIMATIONS                                                        ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════════════

# ANIMATION 1 — j(r) PROFILE WITH GHOST HISTORY

# ══════════════════════════════════════════════════════════════════════════════

# 

# Same ghost-line technique used in density_pipeline.py §34:

# the last N_GHOST profiles are drawn at increasing alpha behind the current

# frame, giving a sense of how j(r) is evolving without cluttering the plot.

# An additional panel shows the j dispersion σ_j(r), which widens as

# phase-space structure is erased by mixing.

print(”\n[Anim 1]  j(r) profile animation …”)

j_anim_idxs = np.arange(0, ns, J_ANIM_STEP)
n_j_frames  = len(j_anim_idxs)
cmap_time   = plt.cm.plasma

fig_a1, (axA, axB) = plt.subplots(
1, 2, figsize=(12, 5.5), facecolor=BG,
gridspec_kw={“wspace”: 0.32},
)
for ax in (axA, axB):
ax.set_facecolor(BG)
ax.set_xscale(“log”)

# Determine y-limits from the full dataset.

j_finite = j_ts[np.isfinite(j_ts) & (j_ts > 0)]
j_ymin   = j_finite.min() * 0.5 if j_finite.size > 0 else 1.0
j_ymax   = j_finite.max() * 2.0 if j_finite.size > 0 else 1e5
axA.set_ylim(j_ymin, j_ymax); axA.set_yscale(“log”)
axB.set_ylim(0, np.nanpercentile(j_std_ts[np.isfinite(j_std_ts)], 99) * 1.2)
axA.set_xlim(R_BINS[0], R_BINS[-1])
axB.set_xlim(R_BINS[0], R_BINS[-1])
axA.set_xlabel(“r [kpc]”, color=”#c8c8e8”)
axA.set_ylabel(r”$j(r)$ [kpc km s$^{-1}$]”, color=”#c8c8e8”)
axA.set_title(r”$j(r)$ with history”, color=”#c8c8e8”)
axB.set_xlabel(“r [kpc]”, color=”#c8c8e8”)
axB.set_ylabel(r”$\sigma_j(r)$ [kpc km s$^{-1}$]”, color=”#c8c8e8”)
axB.set_title(r”$j$ dispersion $\sigma_j(r)$”, color=”#c8c8e8”)

ghost_lines = [axA.plot([], [], lw=0.8)[0] for _ in range(N_GHOST)]
main_j_line,  = axA.plot([], [], lw=2.2, color=“white”, zorder=5)
std_j_line,   = axB.plot([], [], lw=2.0, color=”#e8673a”)
title_a1 = fig_a1.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_j_anim(frame_idx):
snap_i = j_anim_idxs[frame_idx]
color  = cmap_time(frame_idx / n_j_frames)

```
def _xy(arr):
    v = np.isfinite(arr) & (arr > 0)
    return r_mid_sph[v], arr[v]

# Current j profile.
rx, ry = _xy(j_ts[snap_i, :])
main_j_line.set_data(rx, ry)
main_j_line.set_color(color)

# Ghost lines (last N_GHOST frames).
for g, ghost in enumerate(ghost_lines):
    past_idx = frame_idx - (N_GHOST - g)
    if past_idx < 0:
        ghost.set_data([], [])
        continue
    past_snap  = j_anim_idxs[past_idx]
    px, py     = _xy(j_ts[past_snap, :])
    past_color = cmap_time(past_idx / n_j_frames)
    ghost.set_data(px, py)
    ghost.set_color(past_color)
    ghost.set_alpha(0.06 + 0.06 * g)

# Dispersion panel.
std_row = j_std_ts[snap_i, :]
sv = np.isfinite(std_row)
std_j_line.set_data(r_mid_sph[sv], std_row[sv])

t_val = time_arr[snap_i]
t_str = f"{t_val:.2f} Gyr" if time_is_gyr else f"Snap {SNAPSHOTS[snap_i]}"
title_a1.set_text(f"Angular Momentum Profile  ·  {t_str}")

return [main_j_line, std_j_line] + ghost_lines
```

ani_j = animation.FuncAnimation(
fig_a1, _update_j_anim, frames=n_j_frames,
interval=1000 // ANIM_FPS_21, blit=True,
)
writer_j = animation.FFMpegWriter(
fps=ANIM_FPS_21, bitrate=ANIM_BITRATE_21,
metadata=dict(title=“MW-M31 Angular Momentum Profile Animation”),
)
ani_j.save(os.path.join(OUT_DIR, “section21_animation_j.mp4”),
writer=writer_j, dpi=ANIM_DPI_21)
plt.close(fig_a1)
print(”  Saved: section21_animation_j.mp4”)

# ══════════════════════════════════════════════════════════════════════════════

# ANIMATION 2 — (r, j) PHASE-SPACE DENSITY ANIMATION

# ══════════════════════════════════════════════════════════════════════════════

# 

# Plays through the pre-computed 2D histograms phasespace_hists, showing

# the (r, j) mass distribution as a log-colour image.

# A second panel shows the marginal j distribution (integrated over all r)

# which tracks the global angular momentum distribution.

print(”[Anim 2]  Phase-space density animation …”)

# Colour scale from full histogram dataset.

H_all    = phasespace_hists[phasespace_hists > 0]
H_logmin = np.log10(np.percentile(H_all, 5))  if H_all.size > 0 else 0
H_logmax = np.log10(np.percentile(H_all, 99)) if H_all.size > 0 else 10

fig_a2, (axPS, axMarg) = plt.subplots(
1, 2, figsize=(13, 5.5), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 1], “wspace”: 0.06},
)
axPS.set_facecolor(BG); axMarg.set_facecolor(BG)

first_H     = phasespace_hists[0]
first_H_log = np.where(first_H > 0, np.log10(first_H), np.nan)

im_ps = axPS.imshow(
first_H_log.T,
origin=“lower”, aspect=“auto”,
extent=[0.1, 400.0, J_MIN_KPC_KMS, J_MAX_KPC_KMS],
cmap=“magma”, vmin=H_logmin, vmax=H_logmax,
)
axPS.set_xscale(“log”)
axPS.set_xlabel(“r [kpc]”, color=”#c8c8e8”)
axPS.set_ylabel(r”$j$ [kpc km s$^{-1}$]”, color=”#c8c8e8”)

j_marg     = first_H.sum(axis=0)
j_axis_arr = np.linspace(J_MIN_KPC_KMS, J_MAX_KPC_KMS, ENTROPY_JBINS)
marg_line, = axMarg.plot(j_marg / (j_marg.max() + 1e-30), j_axis_arr,
color=”#ff9944”, lw=1.8)
axMarg.set_xlabel(“Norm. mass”, color=”#c8c8e8”)
axMarg.set_ylim(J_MIN_KPC_KMS, J_MAX_KPC_KMS)
axMarg.tick_params(labelleft=False)
axMarg.set_title(r”$P(j)$”, color=”#c8c8e8”, fontsize=10)

title_a2 = fig_a2.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_ps_anim(frame_idx):
H     = phasespace_hists[frame_idx]
H_log = np.where(H > 0, np.log10(H), np.nan)
im_ps.set_data(H_log.T)

```
j_m   = H.sum(axis=0)
marg_line.set_xdata(j_m / (j_m.max() + 1e-30))

snap_i  = phasespace_snap_ids[frame_idx]
t_val   = time_arr[snap_i]
t_str   = f"{t_val:.2f} Gyr" if time_is_gyr else f"Snap {SNAPSHOTS[snap_i]}"
title_a2.set_text(fr"$(r,\,j)$ Phase-Space Density  ·  {t_str}")
return [im_ps, marg_line]
```

ani_ps = animation.FuncAnimation(
fig_a2, _update_ps_anim, frames=n_ps_frames,
interval=1000 // ANIM_FPS_21, blit=True,
)
ani_ps.save(os.path.join(OUT_DIR, “section21_animation_phasespace.mp4”),
writer=writer_j, dpi=ANIM_DPI_21)
plt.close(fig_a2)
print(”  Saved: section21_animation_phasespace.mp4”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.7 — MASTER SUMMARY PANEL                                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 

# Six-panel summary condensing the most important Section 21 results into a

# single figure suitable for a paper supplementary section or poster:

# (0,0) j(r,t) heatmap

# (0,1) j_z(r,t) heatmap

# (1,0) Inner vs. outer j time series

# (1,1) Phase-space entropy S(t)

# (2,0) Flux ⟨j v_r⟩(r) at 5 epochs

# (2,1) Mixing length λ_mix(t)

print(”\n[Summary]  Master summary panel …”)

fig_sum = plt.figure(figsize=(16, 14), facecolor=BG)
gs_sum  = gridspec.GridSpec(3, 2, figure=fig_sum,
hspace=0.42, wspace=0.32,
left=0.08, right=0.97,
top=0.94, bottom=0.06)

# (0,0) j(r,t) heatmap

ax_s00 = fig_sum.add_subplot(gs_sum[0, 0])
ax_s00.set_facecolor(BG)
im_s00 = ax_s00.imshow(
j_ts.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“plasma”,
norm=LogNorm(vmin=np.nanpercentile(j_ts[j_ts > 0], 5),
vmax=np.nanpercentile(j_ts[j_ts > 0], 99)),
)
ax_s00.set_yscale(“log”)
ax_s00.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s00.set_ylabel(“r [kpc]”, fontsize=8, color=”#c8c8e8”)
ax_s00.set_title(r”$j(r,t)$”, fontsize=9, color=”#c8c8e8”)
fig_sum.colorbar(im_s00, ax=ax_s00, shrink=0.8, label=r”$j$ [kpc km/s]”)

# (0,1) j_z(r,t) heatmap

ax_s01 = fig_sum.add_subplot(gs_sum[0, 1])
ax_s01.set_facecolor(BG)
im_s01 = ax_s01.imshow(
jz_ts.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“RdBu_r”,
norm=TwoSlopeNorm(vmin=-jz_abs_max, vcenter=0, vmax=jz_abs_max),
)
ax_s01.set_yscale(“log”)
ax_s01.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s01.set_title(r”$j_z(r,t)$  prograde/retrograde”, fontsize=9, color=”#c8c8e8”)
fig_sum.colorbar(im_s01, ax=ax_s01, shrink=0.8, label=r”$j_z$”)

# (1,0) Inner vs. outer j

ax_s10 = fig_sum.add_subplot(gs_sum[1, 0])
ax_s10.set_facecolor(BG)
ax_s10.plot(time_arr, j_inner_arr, color=”#4a8fff”, lw=1.5, label=“Inner”)
ax_s10.plot(time_arr, j_outer_arr, color=”#ff9944”, lw=1.5, label=“Outer”)
ax_s10.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s10.set_ylabel(r”$j$ [kpc km/s]”, fontsize=8, color=”#c8c8e8”)
ax_s10.set_title(“Inner vs. outer j”, fontsize=9, color=”#c8c8e8”)
ax_s10.legend(fontsize=7)

# (1,1) Phase-space entropy

ax_s11 = fig_sum.add_subplot(gs_sum[1, 1])
ax_s11.set_facecolor(BG)
ax_s11.plot(time_arr, entropy_arr, color=”#e8673a”, lw=1.5)
ax_s11.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s11.set_ylabel(r”$S$ [nats]”, fontsize=8, color=”#c8c8e8”)
ax_s11.set_title(“Phase-space entropy”, fontsize=9, color=”#c8c8e8”)

# (2,0) Flux profiles at 5 epochs

ax_s20 = fig_sum.add_subplot(gs_sum[2, 0])
ax_s20.set_facecolor(BG)
ax_s20.set_xscale(“log”)
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
y = jflux_ts[k_idx, :]
v = np.isfinite(y)
if v.any():
ax_s20.plot(r_mid_sph[v], y[v], color=color, lw=1.5, label=label)
ax_s20.axhline(0, color=”#555577”, lw=0.7, ls=”–”)
ax_s20.set_xlabel(“r [kpc]”, fontsize=8, color=”#c8c8e8”)
ax_s20.set_ylabel(r”$\langle j v_r \rangle$”, fontsize=8, color=”#c8c8e8”)
ax_s20.set_title(“Flux profiles”, fontsize=9, color=”#c8c8e8”)
ax_s20.legend(fontsize=6)

# (2,1) Mixing length

ax_s21 = fig_sum.add_subplot(gs_sum[2, 1])
ax_s21.set_facecolor(BG)
ax_s21.plot(time_arr, mix_length_arr, color=”#00d4aa”, lw=1.5)
ax_s21.axhline(3.5, color=”#ffcc44”, lw=0.7, ls=”–”, alpha=0.7,
label=“MW disk scale”)
ax_s21.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s21.set_ylabel(r”$\lambda_{\rm mix}$ [kpc]”, fontsize=8, color=”#c8c8e8”)
ax_s21.set_title(“Mixing length”, fontsize=9, color=”#c8c8e8”)
ax_s21.legend(fontsize=7)

fig_sum.suptitle(
“Section 21 Summary  ·  Angular Momentum Transport & Phase-Space Diagnostics”,
fontsize=13, color=”#c8c8e8”, fontweight=“bold”,
)
fig_sum.savefig(os.path.join(OUT_DIR, “section21_summary_panel.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig_sum)
print(”  Saved: section21_summary_panel.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §21.8 — SECTION COMPLETE                                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  SECTION 21 COMPLETE”)
print(”=”*80)
outputs_21 = [
“section21_j_heatmap.png”,
“section21_j_flux_heatmap.png”,
“section21_jflux_profiles.png”,
“section21_L_enclosed_profiles.png”,
“section21_j_inner_outer.png”,
“section21_component_profiles.png”,
“section21_jz_heatmap.png”,
“section21_circularity_heatmap.png”,
“section21_circularity_dist.png”,
“section21_entropy.png”,
“section21_mixing_length.png”,
“section21_phasespace_snap.png”,
“section21_transport_budget.png”,
“section21_animation_j.mp4”,
“section21_animation_phasespace.mp4”,
“section21_summary_panel.png”,
]
for fn in outputs_21:
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
kind = “animation” if fn.endswith(”.mp4”) else “figure”
print(f”  {fn:<50} {size:6.2f} MB  [{kind}]”)
print(”=”*80)
