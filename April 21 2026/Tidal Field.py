# “””

# SECTION 22 — TIDAL FIELD & STRIPPING DIAGNOSTICS

Author  : Abhinav Vatsa
Date    : April 21st 2026

Continuation of density_pipeline.py and section21_angular_momentum.py.
All globals (SNAPSHOTS, ns, R_BINS, nb_sph, r_mid_sph, OUT_DIR,
MASS_UNIT_MSUN, MIN_PART_SHELL, G_KPC_KMS2_MSUN, PROFILE_INDICES,
PROFILE_LABELS, PROFILE_COLORS, time_arr, time_label, time_is_gyr,
tmpdir, PTYPE, load_snapshot_particles) are inherited from density_pipeline.py
and must be defined before this section is executed.

## Physical motivation

In the MW–M31 merger the gravitational tidal field of each galaxy acts on
the other, and on every particle within both systems.  Tides do three things:

1. STRIPPING   — particles near the tidal radius r_t are unbound and
   ejected into tidal streams or the inter-galactic medium.
   r_t shrinks as the galaxies approach, so progressively
   more deeply-bound material is stripped with each pericentric
   passage.
1. HEATING     — tidal shocks at pericentre pump kinetic energy into the
   halo, raising velocity dispersions and puffing the
   effective radius.  The impulsive approximation gives an
   energy injection ΔE ∝ M / (v_peri r²).
1. MORPHOLOGICAL DISTORTION — the quadrupole tidal force stretches the
   density distribution along the line connecting the two
   galaxy centres, producing the characteristic tidal tails
   and bridges seen in merger images.

This section measures all three effects directly from the N-body snapshots:

§22.1  Tidal radius r_t(t)       — Jacobi radius from M_MW / d_MW-M31³
§22.2  Bound / unbound fraction  — energy criterion E_kin + E_pot < 0
§22.3  Tidal stripping rate       — dM_bound/dt per snapshot
§22.4  Tidal heating ΔE(r, t)    — specific energy injected per shell
§22.5  Tidal tensor eigenvalues   — λ_1, λ_2, λ_3 of ∂²Φ/∂x_i∂x_j
§22.6  Stream detection           — unbound particles in (r, v_r) space
§22.7  Morphological elongation   — axis ratios a/b, a/c from inertia tensor

## Outputs

section22_tidal_radius.png            r_t(t) and r_t / r_half vs. time
section22_bound_fraction.png          Bound mass fraction vs. time
section22_stripping_rate.png          dM_bound/dt vs. time
section22_bound_profiles.png          Bound/unbound ρ(r) at 5 epochs
section22_heating_heatmap.png         ΔE(r,t) tidal heating heatmap
section22_heating_profiles.png        ΔE(r) profiles at 5 epochs
section22_tidal_tensor_eigen.png      λ_1,2,3(t) tidal tensor eigenvalues
section22_tidal_anisotropy.png        Tidal anisotropy (λ_1−λ_3)/(λ_1+λ_3)
section22_stream_rv.png               (r, v_r) unbound particle scatter at 5 epochs
section22_stream_map.png              2D xy map of unbound particles at 5 epochs
section22_stream_fraction_heatmap.png Unbound fraction f_unbound(r,t) heatmap
section22_inertia_axisratios.png      a/b, a/c axis ratios vs. time
section22_inertia_heatmap.png         Shape elongation b/a(r,t) heatmap
section22_inertia_profiles.png        Axis ratio profiles at 5 epochs
section22_mw_m31_separation.png       Separation d(t) and relative speed v_rel(t)
section22_animation_streams.mp4       2D unbound particle map animation
section22_animation_boundmass.mp4     Bound-mass profile animation
section22_summary_panel.png           Master 6-panel summary figure

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
from scipy.linalg import eigh   # for symmetric eigenvalue decomposition
import warnings
import os
import time

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.0 — SECTION CONFIGURATION                                             ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Tidal radius computation ──────────────────────────────────────────────────

# The Jacobi (Roche) tidal radius for a satellite of mass M_sat orbiting a

# host of mass M_host at separation d is:

# r_t ≈ d × (M_sat / (3 M_host))^{1/3}

# We use the MW enclosed mass within the current separation as M_host and the

# M31 total mass as M_sat.  The factor of 3 comes from the tidal vs. centrifugal

# balance in the Jacobi energy.

TIDAL_RADIUS_FACTOR = 3.0     # Jacobi formula coefficient

# ── Energy binding criterion ──────────────────────────────────────────────────

# A particle is “bound” if its specific total energy E = 0.5|v|² + Φ(r) < 0,

# where Φ(r) is approximated as the Newtonian potential from the enclosed mass.

# We use a softened potential with softening length ε_soft to avoid divergences

# at small r.

SOFTENING_KPC = 0.5     # gravitational softening length [kpc]

# ── Potential computation subsampling ─────────────────────────────────────────

# Computing Φ(r) = −G M_enc(r) / r for every particle is O(N) per snapshot

# if we use the enclosed-mass approximation (spherical symmetry assumed).

# This is fast.  A full tree-code potential evaluation is not implemented here.

BOUND_STEP = 4          # compute bound fraction every Nth snapshot

# ── Tidal heating ─────────────────────────────────────────────────────────────

# We measure tidal heating as the change in mean specific kinetic energy per

# radial shell between consecutive snapshots:

# ΔE(r, t) = ⟨E_kin(r)⟩*{t+Δt} − ⟨E_kin(r)⟩*{t}

# Positive ΔE → kinetic energy injected (heating)

# Negative ΔE → kinetic energy removed (cooling, e.g. during initial infall)

HEATING_STEP = 2        # compute ΔE every Nth snapshot pair

# ── Tidal tensor ──────────────────────────────────────────────────────────────

# The tidal tensor T_ij = ∂²Φ/∂x_i∂x_j is computed at the CENTRE OF MASS

# of each galaxy using finite differences of the force field estimated from

# the particle distribution via the enclosed-mass gradient.

# We evaluate T at the position of M31’s COM relative to MW’s COM.

TENSOR_STEP = 8         # compute tidal tensor every Nth snapshot

# ── Inertia tensor ────────────────────────────────────────────────────────────

# The inertia tensor I_ij = Σ m_k (|r_k|² δ_ij − r_ki r_kj) gives the

# axis ratios a ≥ b ≥ c of the mass distribution via its eigenvalues.

# We compute it for three radial apertures: inner (r < 30 kpc), intermediate

# (30–150 kpc), and outer (r > 150 kpc).

INERTIA_R_INNER  = 30.0    # kpc
INERTIA_R_MID    = 150.0   # kpc

# Per-bin inertia axis ratio b/a(r, t) — computed at every INERTIA_STEP snap.

INERTIA_STEP = 8

# ── Stream detection ─────────────────────────────────────────────────────────

# Unbound particles (E > 0) are collected into a 2D (r, v_r) histogram for

# stream visualisation.  We also store their 2D (x, y) positions.

STREAM_MAP_BINS   = 200    # pixels per side for the 2D stream map
STREAM_MAP_EXTENT = 600.0  # half-width [kpc] — wider than density map to catch streams
STREAM_ANIM_STEP  = 8      # render every Nth snapshot in stream animation
BOUND_ANIM_STEP   = 4      # render every Nth snapshot in bound-mass animation

# ── Animation settings ────────────────────────────────────────────────────────

ANIM_FPS_22     = 20
ANIM_DPI_22     = 100
ANIM_BITRATE_22 = 2000

print(”\n” + “=”*80)
print(”  SECTION 22 · Tidal Field & Stripping Diagnostics”)
print(”=”*80)
print(f”  Softening length : {SOFTENING_KPC} kpc”)
print(f”  Bound step       : every {BOUND_STEP} snapshots”)
print(f”  Tensor step      : every {TENSOR_STEP} snapshots”)
print(f”  Inertia step     : every {INERTIA_STEP} snapshots”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.1 — CORE UTILITY FUNCTIONS                                            ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def enclosed_mass_profile(r_mag: np.ndarray,
m_msun: np.ndarray,
r_bins: np.ndarray) -> np.ndarray:
“””
Compute the cumulative enclosed mass M(<r) at each bin outer edge.

```
Parameters
----------
r_mag  : (N,)    — 3D radii  [kpc]
m_msun : (N,)    — masses    [M_sun]
r_bins : (nb+1,) — bin edges [kpc]

Returns
-------
M_enc : (nb,)   — enclosed mass at each bin outer edge  [M_sun]
"""
nb    = len(r_bins) - 1
M_enc = np.zeros(nb)
for b in range(nb):
    M_enc[b] = m_msun[r_mag <= r_bins[b + 1]].sum()
return M_enc
```

def specific_potential(r_mag:  np.ndarray,
m_msun: np.ndarray,
r_bins: np.ndarray,
softening: float = SOFTENING_KPC) -> np.ndarray:
“””
Estimate the specific gravitational potential Φ(r) for each particle
using the spherically-symmetric enclosed-mass approximation:

```
    Φ(r) = −G M_enc(r) / sqrt(r² + ε²)

where ε = softening is the Plummer softening length.

This is the "shell theorem" approximation — exact for a spherically
symmetric distribution, and a good approximation for the near-spherical
halo at r > a few kpc.  It under-estimates the potential at very small r
where the disk contributes non-spherically.

Parameters
----------
r_mag     : (N,)    — particle radii  [kpc]
m_msun    : (N,)    — particle masses [M_sun]
r_bins    : (nb+1,) — bin edges       [kpc]
softening : float   — Plummer softening [kpc]

Returns
-------
phi : (N,)   — specific potential  [km²/s²]

Notes
-----
The unit conversion:  G = 4.301×10⁻⁶ kpc (km/s)² M_sun⁻¹
So [G × M_sun / kpc] = [km²/s²]  ✓
"""
M_enc  = enclosed_mass_profile(r_mag, m_msun, r_bins)

# Interpolate M_enc onto each particle's radius.
# r_bins[1:] are the outer edges used by enclosed_mass_profile.
r_edges = r_bins[1:]
M_enc_interp = np.interp(r_mag, r_edges, M_enc,
                          left=0.0, right=M_enc[-1])

r_soft = np.sqrt(r_mag**2 + softening**2)   # softened radius

phi = -G_KPC_KMS2_MSUN * M_enc_interp / r_soft   # [km²/s²]
return phi
```

def jacobi_tidal_radius(M_sat: float,
M_host_enc: float,
separation: float) -> float:
“””
Compute the Jacobi (Roche) tidal radius:

```
    r_t = d × (M_sat / (TIDAL_RADIUS_FACTOR × M_host_enc))^{1/3}

where d is the separation between the two galaxy centres.

Parameters
----------
M_sat       : float — satellite enclosed mass    [M_sun]
M_host_enc  : float — host enclosed mass within the satellite's orbit [M_sun]
separation  : float — current galaxy-galaxy separation  [kpc]

Returns
-------
r_t : float  [kpc]   — NaN if either mass is zero or separation is zero

Physical notes
--------------
The Jacobi radius marks the boundary of the satellite's gravitational
sphere of influence in the co-rotating frame.  Particles outside r_t are
tidally unbound and will be stripped on the next orbit.

The formula assumes a circular orbit.  For the highly eccentric MW–M31
orbit the actual stripping radius at pericentre is smaller than this
estimate by a factor of ~2, so r_t gives an upper bound on the true
stripping radius.
"""
if M_host_enc <= 0 or separation <= 0 or M_sat <= 0:
    return np.nan
return separation * (M_sat / (TIDAL_RADIUS_FACTOR * M_host_enc))**(1.0 / 3.0)
```

def inertia_tensor(pos: np.ndarray,
m:   np.ndarray) -> np.ndarray:
“””
Compute the reduced inertia tensor of a set of particles:

```
    I_ij = Σ_k  m_k (|r_k|² δ_ij − r_ki r_kj)

This is the standard moment-of-inertia tensor.  Its eigenvalues I_1 ≤ I_2 ≤ I_3
correspond to principal axes a ≥ b ≥ c where:
    a = sqrt(5 (I_2 + I_3 − I_1) / (2 M_tot))
and similarly for b, c (Bett et al. 2007 convention).

Parameters
----------
pos : (N, 3) — particle positions [kpc]
m   : (N,)   — particle masses    [M_sun]

Returns
-------
I : (3, 3) symmetric ndarray   [M_sun kpc²]
"""
r2 = np.sum(pos**2, axis=1)   # (N,)
M  = m.sum()

if M == 0:
    return np.zeros((3, 3))

I = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        delta_ij = 1.0 if i == j else 0.0
        I[i, j]  = np.sum(m * (r2 * delta_ij - pos[:, i] * pos[:, j]))
return I
```

def axis_ratios_from_inertia(I: np.ndarray) -> tuple[float, float, float]:
“””
Extract the principal-axis lengths a ≥ b ≥ c from the inertia tensor I.

```
Using the Bett et al. (2007) convention:
    eigenvalues of I:  λ_1 ≤ λ_2 ≤ λ_3   (sorted ascending)
    a = sqrt((5 / 2M) × (λ_2 + λ_3 − λ_1))
    b = sqrt((5 / 2M) × (λ_1 + λ_3 − λ_2))
    c = sqrt((5 / 2M) × (λ_1 + λ_2 − λ_3))

Returns
-------
(a, b, c)  with a ≥ b ≥ c  [kpc]

Notes
-----
The axis ratios b/a and c/a are the standard shape descriptors:
    b/a → 1   →  oblate or spherical
    c/a → 1   →  spherical
    c/a → 0   →  prolate (cigar-shaped), typical of tidal disruption
"""
try:
    eigvals = np.linalg.eigvalsh(I)   # sorted ascending
except np.linalg.LinAlgError:
    return np.nan, np.nan, np.nan

eigvals = np.sort(eigvals)   # λ_1 ≤ λ_2 ≤ λ_3
l1, l2, l3 = eigvals

# Total mass is not directly available here; we normalise so that the
# largest axis equals 1 (shape only, not size).  The caller can scale.
# Compute squared semi-axes from Bett convention (unnormalised).
a2 = l2 + l3 - l1
b2 = l1 + l3 - l2
c2 = l1 + l2 - l3

# Guard against numerical negatives.
a2 = max(a2, 0.0)
b2 = max(b2, 0.0)
c2 = max(c2, 0.0)

a = np.sqrt(a2)
b = np.sqrt(b2)
c = np.sqrt(c2)

# Sort descending.
axes = sorted([a, b, c], reverse=True)
return float(axes[0]), float(axes[1]), float(axes[2])
```

def tidal_tensor_at_point(pos_field:  np.ndarray,
m_field:    np.ndarray,
eval_point: np.ndarray,
r_bins:     np.ndarray,
delta:      float = 1.0) -> np.ndarray:
“””
Estimate the tidal tensor T_ij = ∂²Φ/∂x_i∂x_j at eval_point using
the finite-difference method.

```
T_ij ≈ (Φ(r + δ ê_i + δ ê_j) − Φ(r + δ ê_i) − Φ(r + δ ê_j) + Φ(r)) / δ²

where Φ is the softened enclosed-mass potential and δ = delta [kpc] is
the finite-difference step.  The diagonal terms use a second-order
central difference:
    T_ii ≈ (Φ(r + δ ê_i) − 2 Φ(r) + Φ(r − δ ê_i)) / δ²

Physical meaning of T eigenvalues:
    λ_1 > 0  →  tidal compression along this axis
    λ_1 < 0  →  tidal stretching
The traceless part of T is the tidal deformation tensor; its trace gives
∇²Φ = 4πGρ (Poisson equation).

Parameters
----------
pos_field  : (N, 3) — all particle positions  [kpc]
m_field    : (N,)   — all particle masses      [M_sun]
eval_point : (3,)   — point at which to evaluate T  [kpc]
r_bins     : bin edges
delta      : float  — finite-difference step  [kpc]

Returns
-------
T : (3, 3) symmetric ndarray   [km²/s²/kpc²]

Notes
-----
The potential at each probe point is estimated by shifting the entire
particle distribution by −eval_point (so that the probe is at the
origin), then calling specific_potential.  This is equivalent to
evaluating the potential at shifted probe positions.
"""
T = np.zeros((3, 3))
e = np.eye(3)   # basis vectors

def _phi_at(probe):
    # Shift particles so probe is at origin.
    r_shifted = np.linalg.norm(pos_field - probe, axis=1)
    return np.sum(
        -G_KPC_KMS2_MSUN * m_field / np.sqrt(r_shifted**2 + SOFTENING_KPC**2)
    )

phi0 = _phi_at(eval_point)

for i in range(3):
    # Diagonal: second-order central difference.
    phi_plus  = _phi_at(eval_point + delta * e[i])
    phi_minus = _phi_at(eval_point - delta * e[i])
    T[i, i]   = (phi_plus - 2 * phi0 + phi_minus) / delta**2

    # Off-diagonal: mixed second derivatives.
    for j in range(i + 1, 3):
        phi_pp = _phi_at(eval_point + delta * e[i] + delta * e[j])
        phi_pm = _phi_at(eval_point + delta * e[i] - delta * e[j])
        phi_mp = _phi_at(eval_point - delta * e[i] + delta * e[j])
        phi_mm = _phi_at(eval_point - delta * e[i] - delta * e[j])
        T[i, j] = (phi_pp - phi_pm - phi_mp + phi_mm) / (4 * delta**2)
        T[j, i] = T[i, j]   # symmetry

return T
```

def mass_weighted_shell_bin(values: np.ndarray,
r_mag:  np.ndarray,
m:      np.ndarray,
r_bins: np.ndarray) -> np.ndarray:
“””
Mass-weighted mean of values in each radial shell.
Identical to the utility in §21 — reproduced here for self-containment.
“””
nb     = len(r_bins) - 1
prof   = np.full(nb, np.nan)
bin_id = np.digitize(r_mag, r_bins) - 1
for b in range(nb):
mask = bin_id == b
if mask.sum() < MIN_PART_SHELL:
continue
w       = m[mask]
prof[b] = np.sum(w * values[mask]) / np.sum(w)
return prof

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.2 — PRE-ALLOCATION                                                    ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Bound fraction arrays ─────────────────────────────────────────────────────

bound_snap_nums  = SNAPSHOTS[::BOUND_STEP]
n_bound          = len(bound_snap_nums)
bound_snap_map   = {s: i for i, s in enumerate(bound_snap_nums)}

bound_frac_arr   = np.full(n_bound, np.nan)   # fraction of total mass that is bound
M_bound_arr      = np.full(n_bound, np.nan)   # bound mass  [M_sun]
M_unbound_arr    = np.full(n_bound, np.nan)   # unbound mass  [M_sun]
time_bound       = np.full(n_bound, np.nan)   # time axis for bound arrays

# Bound/unbound density profiles — shape (n_bound, nb_sph)

rho_bound_ts   = np.full((n_bound, nb_sph), np.nan)
rho_unbound_ts = np.full((n_bound, nb_sph), np.nan)

# Unbound fraction per shell — f_unbound(r, t) for the heatmap

f_unbound_ts   = np.full((n_bound, nb_sph), np.nan)

# ── Tidal radius ──────────────────────────────────────────────────────────────

r_tidal_arr    = np.full(ns, np.nan)   # Jacobi radius [kpc] at every snapshot
separation_arr = np.full(ns, np.nan)   # MW–M31 separation [kpc]
v_rel_arr      = np.full(ns, np.nan)   # relative approach speed [km/s]

# ── Tidal heating ─────────────────────────────────────────────────────────────

heat_snap_nums  = SNAPSHOTS[::HEATING_STEP]
n_heat          = len(heat_snap_nums)
heat_snap_map   = {s: i for i, s in enumerate(heat_snap_nums)}
time_heat       = np.full(n_heat, np.nan)

# E_kin profile per shell — stored so ΔE can be differenced

Ekin_ts         = np.full((ns, nb_sph), np.nan)   # mean specific KE per shell
dEkin_dt_ts     = np.full((ns, nb_sph), np.nan)   # ΔE_kin between consecutive snaps

# ── Tidal tensor eigenvalues ──────────────────────────────────────────────────

tensor_snap_nums = SNAPSHOTS[::TENSOR_STEP]
n_tensor         = len(tensor_snap_nums)
tensor_snap_map  = {s: i for i, s in enumerate(tensor_snap_nums)}
time_tensor      = np.full(n_tensor, np.nan)

T_eig1_arr = np.full(n_tensor, np.nan)   # largest eigenvalue (compression)
T_eig2_arr = np.full(n_tensor, np.nan)
T_eig3_arr = np.full(n_tensor, np.nan)   # smallest (stretching)
T_trace_arr = np.full(n_tensor, np.nan)  # trace ≈ 4πGρ at COM

# ── Inertia tensor axis ratios ────────────────────────────────────────────────

inertia_snap_nums = SNAPSHOTS[::INERTIA_STEP]
n_inertia         = len(inertia_snap_nums)
inertia_snap_map  = {s: i for i, s in enumerate(inertia_snap_nums)}
time_inertia      = np.full(n_inertia, np.nan)

# Global axis ratios (all particles within 400 kpc)

ba_arr  = np.full(n_inertia, np.nan)   # b/a intermediate-to-major ratio
ca_arr  = np.full(n_inertia, np.nan)   # c/a minor-to-major ratio

# Per-aperture axis ratios: inner, mid, outer

ba_inner_arr = np.full(n_inertia, np.nan)
ca_inner_arr = np.full(n_inertia, np.nan)
ba_mid_arr   = np.full(n_inertia, np.nan)
ca_mid_arr   = np.full(n_inertia, np.nan)
ba_outer_arr = np.full(n_inertia, np.nan)
ca_outer_arr = np.full(n_inertia, np.nan)

# Per-bin b/a(r, t) — shape (n_inertia, nb_sph)

ba_profile_ts = np.full((n_inertia, nb_sph), np.nan)

# ── Stream animation storage ──────────────────────────────────────────────────

stream_anim_snap_ids = np.arange(0, n_bound, max(1, STREAM_ANIM_STEP // BOUND_STEP))
n_stream_frames      = len(stream_anim_snap_ids)

# Store 2D (x, y) histograms of unbound particles.

stream_maps = np.zeros((n_stream_frames, STREAM_MAP_BINS, STREAM_MAP_BINS))

# ── Bound mass animation storage ──────────────────────────────────────────────

bound_anim_snap_ids  = np.arange(0, n_bound, max(1, BOUND_ANIM_STEP // BOUND_STEP))
n_bound_anim_frames  = len(bound_anim_snap_ids)

print(f”\n[Pre-alloc] Bound arrays      : {rho_bound_ts.shape}”)
print(f”            Ekin_ts           : {Ekin_ts.shape}”)
print(f”            Tidal tensor snaps: {n_tensor}”)
print(f”            Inertia snaps     : {n_inertia}”)
print(f”            Stream frames     : {n_stream_frames}”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.3 — MAIN SNAPSHOT LOOP                                                ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  §22.3 — Main Snapshot Loop”)
print(”=”*80)

t_loop_start = time.perf_counter()

# We also need to track the position of each galaxy’s COM over time to compute

# the separation vector.  We store these separately for finite-difference

# velocity estimation.

com_mw_arr  = np.full((ns, 3), np.nan)   # MW COM position [kpc] at each snap
com_m31_arr = np.full((ns, 3), np.nan)   # M31 COM position [kpc]

for i, snap_num in enumerate(SNAPSHOTS):

```
mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
    continue

t_snap = time.perf_counter()

# ── Load particles ────────────────────────────────────────────────────────
try:
    snap_data = load_snapshot_particles(mw_file, m31_file)
    MW_obj    = CenterOfMass(mw_file,  PTYPE)
    M31_obj   = CenterOfMass(m31_file, PTYPE)
except Exception as exc:
    print(f"  [ERROR] snap {snap_num}: {exc}")
    continue

pos    = snap_data["pos"]      # (N, 3) COM-centred  [kpc]
m      = snap_data["m_msun"]   # (N,)   [M_sun]
origin = snap_data["origin"]   # 0=MW, 1=M31
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
dr_com = np.sqrt((x_all-xcom)**2 + (y_all-ycom)**2 + (z_all-zcom)**2)
inner  = dr_com < 15.0
if inner.sum() >= 5:
    wi = m[inner]
    vxcom = np.sum(wi * vx_all[inner]) / wi.sum()
    vycom = np.sum(wi * vy_all[inner]) / wi.sum()
    vzcom = np.sum(wi * vz_all[inner]) / wi.sum()
else:
    vxcom = vycom = vzcom = 0.0

vel   = np.vstack((vx_all - vxcom, vy_all - vycom, vz_all - vzcom)).T
v_mag2 = np.sum(vel**2, axis=1)   # |v|²  [km²/s²]

# ── Individual galaxy COMs ────────────────────────────────────────────────
# MW COM position (centred on joint COM).
mw_x_com, mw_y_com, mw_z_com = MW_obj.COMdefine(
    MW_obj.x, MW_obj.y, MW_obj.z, MW_obj.m)
m31_x_com, m31_y_com, m31_z_com = M31_obj.COMdefine(
    M31_obj.x, M31_obj.y, M31_obj.z, M31_obj.m)

com_mw_arr [i] = [mw_x_com  - xcom, mw_y_com  - ycom, mw_z_com  - zcom]
com_m31_arr[i] = [m31_x_com - xcom, m31_y_com - ycom, m31_z_com - zcom]

# ── Galaxy–galaxy separation ──────────────────────────────────────────────
sep_vec  = com_m31_arr[i] - com_mw_arr[i]
sep_dist = np.linalg.norm(sep_vec)
separation_arr[i] = sep_dist

# ── Tidal radius (Jacobi) ─────────────────────────────────────────────────
# Use the MW enclosed mass within the current separation as M_host.
# Use the total M31 mass as M_sat.
M_host_enc = m[origin == 0][r_mag[origin == 0] <= sep_dist].sum()
M_sat      = m[origin == 1].sum()
r_tidal_arr[i] = jacobi_tidal_radius(M_sat, M_host_enc, sep_dist)

# ── Specific kinetic energy profile ───────────────────────────────────────
E_kin_spec = 0.5 * v_mag2   # [km²/s²]
Ekin_ts[i, :] = mass_weighted_shell_bin(E_kin_spec, r_mag, m, R_BINS)

# ── Bound / unbound classification ────────────────────────────────────────
if snap_num in bound_snap_map:
    bi = bound_snap_map[snap_num]
    time_bound[bi] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)

    # Specific potential energy (softened enclosed-mass approximation).
    phi = specific_potential(r_mag, m, R_BINS)

    # Total specific energy per particle.
    E_tot = 0.5 * v_mag2 + phi   # [km²/s²]

    bound_mask   = E_tot < 0.0
    unbound_mask = ~bound_mask

    M_tot          = m.sum()
    M_bound_arr[bi]   = m[bound_mask].sum()
    M_unbound_arr[bi] = m[unbound_mask].sum()
    bound_frac_arr[bi] = M_bound_arr[bi] / (M_tot + 1e-30)

    # Density profiles for bound and unbound populations.
    shell_vols_local = (4.0/3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)
    bin_id = np.digitize(r_mag, R_BINS) - 1

    for b in range(nb_sph):
        mask_b = bin_id == b
        M_bin_bound   = m[mask_b & bound_mask].sum()
        M_bin_unbound = m[mask_b & unbound_mask].sum()
        M_bin_total   = m[mask_b].sum()

        if mask_b.sum() >= MIN_PART_SHELL:
            rho_bound_ts  [bi, b] = M_bin_bound   / shell_vols_local[b]
            rho_unbound_ts[bi, b] = M_bin_unbound / shell_vols_local[b]
            f_unbound_ts  [bi, b] = M_bin_unbound / (M_bin_total + 1e-30)

    # ── 2D stream map ─────────────────────────────────────────────────────
    # Find the nearest stream animation frame.
    anim_frame_idx = np.where(
        np.array(list(bound_snap_map.values())) == bi
    )
    if bi in stream_anim_snap_ids:
        fi_stream = np.where(stream_anim_snap_ids == bi)[0]
        if len(fi_stream) > 0:
            fi_s  = fi_stream[0]
            xu    = pos[unbound_mask, 0]
            yu    = pos[unbound_mask, 1]
            H, _, _ = np.histogram2d(
                xu, yu,
                bins=STREAM_MAP_BINS,
                range=[[-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT],
                       [-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT]],
                weights=m[unbound_mask],
            )
            stream_maps[fi_s] = H

# ── Tidal tensor ──────────────────────────────────────────────────────────
if snap_num in tensor_snap_map:
    ti = tensor_snap_map[snap_num]
    time_tensor[ti] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)

    # Evaluate tidal tensor at the midpoint between the two COMs.
    # Subsampling particles for speed (every 10th particle is sufficient
    # for the potential gradient estimate).
    sub = np.arange(0, len(pos), 10)
    try:
        T = tidal_tensor_at_point(
            pos[sub], m[sub],
            eval_point=0.5 * (com_mw_arr[i] + com_m31_arr[i]),
            r_bins=R_BINS, delta=2.0,
        )
        eigvals_T = np.linalg.eigvalsh(T)   # sorted ascending
        T_eig1_arr[ti]  = eigvals_T[2]   # largest
        T_eig2_arr[ti]  = eigvals_T[1]
        T_eig3_arr[ti]  = eigvals_T[0]   # smallest (most stretching)
        T_trace_arr[ti] = np.trace(T)
    except Exception as exc:
        pass   # leave as NaN for this snapshot

# ── Inertia tensor ────────────────────────────────────────────────────────
if snap_num in inertia_snap_map:
    ii_idx = inertia_snap_map[snap_num]
    time_inertia[ii_idx] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)

    # Global axis ratios.
    I_global = inertia_tensor(pos, m)
    a_g, b_g, c_g = axis_ratios_from_inertia(I_global)
    if a_g > 0:
        ba_arr[ii_idx] = b_g / a_g
        ca_arr[ii_idx] = c_g / a_g

    # Per-aperture.
    for mask_ap, ba_ap, ca_ap, label in [
        (r_mag < INERTIA_R_INNER,                          ba_inner_arr, ca_inner_arr, "inner"),
        ((r_mag >= INERTIA_R_INNER) & (r_mag < INERTIA_R_MID), ba_mid_arr,   ca_mid_arr,   "mid"),
        (r_mag >= INERTIA_R_MID,                           ba_outer_arr, ca_outer_arr, "outer"),
    ]:
        if mask_ap.sum() < 50:
            continue
        I_ap = inertia_tensor(pos[mask_ap], m[mask_ap])
        a_ap, b_ap, c_ap = axis_ratios_from_inertia(I_ap)
        if a_ap > 0:
            ba_ap[ii_idx] = b_ap / a_ap
            ca_ap[ii_idx] = c_ap / a_ap

    # Per-bin axis ratios: compute inertia tensor in each shell.
    bin_id = np.digitize(r_mag, R_BINS) - 1
    for b in range(nb_sph):
        mask_b = bin_id == b
        if mask_b.sum() < 50:
            continue
        I_b = inertia_tensor(pos[mask_b], m[mask_b])
        a_b, b_b, c_b = axis_ratios_from_inertia(I_b)
        if a_b > 0:
            ba_profile_ts[ii_idx, b] = b_b / a_b

# ── Progress ──────────────────────────────────────────────────────────────
if (i + 1) % 100 == 0:
    elapsed = time.perf_counter() - t_loop_start
    sep_str = f"d={separation_arr[i]:.0f} kpc" if np.isfinite(separation_arr[i]) else "d=?"
    rt_str  = f"r_t={r_tidal_arr[i]:.0f} kpc"  if np.isfinite(r_tidal_arr[i])  else "r_t=?"
    print(f"  snap {snap_num:04d}  {sep_str}  {rt_str}  [{elapsed:.0f}s]")
```

print(f”\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total”)

# ── Relative approach velocity from finite differences of COM positions ───────

# v_rel = d(separation) / dt  (positive = separating, negative = approaching)

sep_valid = np.isfinite(separation_arr)
if sep_valid.sum() > 2:
v_rel_arr = np.gradient(separation_arr)   # [kpc / snapshot]
# Convert to km/s: Δt ≈ 10 Myr = 10.22 km/s per kpc/snap (rough)
# v_rel_arr *= 97.8    # uncomment and calibrate to actual snapshot cadence

# ── Tidal heating: finite difference of E_kin profile over time ───────────────

# ΔE_kin(r, t) = E_kin(r, t+Δt) − E_kin(r, t)

dEkin_dt_ts = np.gradient(Ekin_ts, axis=0)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.4 — FIGURES                                                           ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

t_min = np.nanmin(time_arr)
t_max = np.nanmax(time_arr)
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

# FIGURE 1 — GALAXY SEPARATION AND TIDAL RADIUS

# ══════════════════════════════════════════════════════════════════════════════

# 

# The separation d(t) defines the merger timeline: each dip corresponds to a

# pericentre passage.  Overlaying r_t(t) on the same plot shows directly when

# the tidal radius reaches inside a given physical scale (e.g., the MW disk

# at ~15 kpc), marking the onset of significant disk stripping.

# r_t / r_half is plotted below: when this ratio < 1 the majority of the

# satellite’s bound mass lies within its tidal radius — the satellite is

# fully stripped.

print(”\n[Fig 1]  Galaxy separation and tidal radius …”)

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

_styled_ax(ax1a, ylabel=“Distance [kpc]”,
title=“MW–M31 Separation and Jacobi Tidal Radius”)
ax1a.plot(time_arr, separation_arr, color=”#4a8fff”, lw=2.0, label=“MW–M31 separation d(t)”)
ax1a.plot(time_arr, r_tidal_arr,    color=”#ff9944”, lw=2.0, label=r”Jacobi $r_t(t)$”)
ax1a.axhline(15.0, color=”#ffffff”, lw=0.7, ls=”:”, alpha=0.4, label=“MW disk ≈ 15 kpc”)
ax1a.set_yscale(“log”)
ax1a.legend(fontsize=8)

# r_t / r_half ratio — requires r_half_3d_arr from density pipeline.

# If not available, skip with a warning.

*styled_ax(ax1b, xlabel=time_label, ylabel=r”$r_t / r*{1/2,,3D}$”,
title=“Tidal Radius in Units of Half-Mass Radius”)
try:
rt_over_rhalf = r_tidal_arr / r_half_3d_arr
ax1b.plot(time_arr, rt_over_rhalf, color=”#e8673a”, lw=1.8)
ax1b.axhline(1.0, color=”#ffffff”, lw=0.8, ls=”–”, alpha=0.5,
label=r”$r_t = r_{1/2}$  (full stripping)”)
ax1b.set_yscale(“log”)
ax1b.legend(fontsize=8)
except NameError:
ax1b.text(0.5, 0.5, “r_half_3d_arr not available\n(run density_pipeline.py first)”,
transform=ax1b.transAxes, ha=“center”, va=“center”,
color=MUTED, fontsize=9)

fig1.savefig(os.path.join(OUT_DIR, “section22_tidal_radius.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig1)
print(”  Saved: section22_tidal_radius.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 2 — BOUND MASS FRACTION

# ══════════════════════════════════════════════════════════════════════════════

# 

# The bound fraction f_bound(t) = M_bound / M_total is the most direct measure

# of tidal stripping progress.  A monotonically decreasing f_bound after first

# pericentre confirms that the merger is destructive (mass-losing satellite).

# Plateaus between pericentre passages indicate that stripping pauses when the

# two galaxies are at apocentre.

print(”[Fig 2]  Bound mass fraction …”)

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

_styled_ax(ax2a, ylabel=“Bound fraction”,
title=“Tidal Stripping: Bound Mass Fraction”)
ax2a.plot(time_bound, bound_frac_arr, color=”#00d4aa”, lw=1.8)
ax2a.fill_between(time_bound,
np.where(np.isfinite(bound_frac_arr), bound_frac_arr, 0),
alpha=0.12, color=”#00d4aa”)
ax2a.set_ylim(0, 1.05)
ax2a.axhline(0.5, color=”#555577”, lw=0.7, ls=”–”, alpha=0.6)
ax2a.text(time_bound[np.isfinite(time_bound)][0], 0.52,
“50% stripped”, color=MUTED, fontsize=7)

*styled_ax(ax2b, xlabel=time_label,
ylabel=r”$M$ [M$*\odot$]”,
title=“Bound and Unbound Mass Budgets”)
ax2b.semilogy(time_bound, M_bound_arr,   color=”#4a8fff”, lw=1.5, label=“Bound”)
ax2b.semilogy(time_bound, M_unbound_arr, color=”#ff5fa0”, lw=1.5, label=“Unbound (streams)”)
ax2b.legend(fontsize=8)

fig2.savefig(os.path.join(OUT_DIR, “section22_bound_fraction.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig2)
print(”  Saved: section22_bound_fraction.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 3 — STRIPPING RATE dM_bound/dt

# ══════════════════════════════════════════════════════════════════════════════

# 

# The stripping rate −dM_bound/dt peaks at pericentre passages.  The ratio of

# successive peak heights tells us whether the stripping is primarily occurring

# on the first passage (efficient stripping in one go) or distributed across

# multiple passages (gradual erosion).

print(”[Fig 3]  Stripping rate …”)

dM_dt = -np.gradient(M_bound_arr)   # positive = mass being stripped
valid_t = np.isfinite(time_bound) & np.isfinite(dM_dt)

fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor=BG)
*styled_ax(ax3, xlabel=time_label,
ylabel=r”$-dM*{\rm bound}/dt$  [M$*\odot$ snap$^{-1}$]”,
title=r”Tidal Stripping Rate  $-dM*{\rm bound}/dt$”)
ax3.plot(time_bound[valid_t], dM_dt[valid_t], color=”#ff9944”, lw=1.5)
ax3.fill_between(time_bound[valid_t],
np.where(dM_dt[valid_t] > 0, dM_dt[valid_t], 0),
alpha=0.18, color=”#ff9944”)

# Mark pericentre passages as the peaks in the stripping rate.

from scipy.signal import find_peaks
peaks, _ = find_peaks(dM_dt[valid_t], height=np.nanpercentile(dM_dt[valid_t], 80))
for pk in peaks:
ax3.axvline(time_bound[valid_t][pk], color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.4)

fig3.savefig(os.path.join(OUT_DIR, “section22_stripping_rate.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig3)
print(”  Saved: section22_stripping_rate.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 4 — BOUND AND UNBOUND DENSITY PROFILES AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# Overplotting ρ_bound and ρ_unbound at five epochs shows where in radius the

# stripping is occurring.  The crossover radius (where ρ_unbound > ρ_bound)

# moves inward with each pericentre passage, documenting the progressive

# inside-out destruction of the satellite.

print(”[Fig 4]  Bound/unbound density profiles …”)

# Map profile indices to nearest bound-array indices.

bound_times  = time_bound[np.isfinite(time_bound)]
profile_bi   = [np.argmin(np.abs(time_bound - time_arr[k])) for k in PROFILE_INDICES]

fig4, axes4 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.06})

for col, (bi, label, color) in enumerate(zip(profile_bi, PROFILE_LABELS, PROFILE_COLORS)):
ax = axes4[col]
_styled_ax(ax, xlabel=“r [kpc]”, title=label, log_x=True, log_y=True)
ax.set_xlim(R_BINS[0], R_BINS[-1])

```
r_b  = rho_bound_ts  [bi, :]
r_ub = rho_unbound_ts[bi, :]

vb  = np.isfinite(r_b)  & (r_b  > 0)
vub = np.isfinite(r_ub) & (r_ub > 0)

if vb.any():
    ax.plot(r_mid_sph[vb],  r_b[vb],  color=color,   lw=2.0, label="Bound")
if vub.any():
    ax.plot(r_mid_sph[vub], r_ub[vub], color=color, lw=1.5,
            ls="--", alpha=0.7, label="Unbound")

if col == 0:
    ax.set_ylabel(r"$\rho$ [M$_\odot$ kpc$^{-3}$]", fontsize=9)
ax.legend(fontsize=7)
```

fig4.suptitle(r”Bound vs. Unbound Density Profiles $\rho(r)$”, fontsize=12)
fig4.savefig(os.path.join(OUT_DIR, “section22_bound_profiles.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig4)
print(”  Saved: section22_bound_profiles.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 5 — UNBOUND FRACTION f_unbound(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# f_unbound(r, t) = ρ_unbound / (ρ_bound + ρ_unbound) shows which radial

# shells become stripped first.  The outermost shells should be stripped

# earliest (smallest binding energy); the progression inward documents the

# inside-out stripping sequence.

print(”[Fig 5]  f_unbound(r,t) heatmap …”)

t_bound_min = np.nanmin(time_bound)
t_bound_max = np.nanmax(time_bound)

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 1], “wspace”: 0.06})

im5 = ax5a.imshow(
f_unbound_ts.T,
aspect=“auto”, origin=“lower”,
extent=[t_bound_min, t_bound_max, R_BINS[0], R_BINS[-1]],
cmap=“hot”, vmin=0.0, vmax=1.0,
)
*styled_ax(ax5a, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Unbound Fraction  $f*{\rm unbound}(r,,t)$”)
ax5a.set_yscale(“log”)
cb5 = fig5.colorbar(im5, ax=ax5a, pad=0.01)
cb5.set_label(r”$f_{\rm unbound}$  (0 = all bound, 1 = all stripped)”, fontsize=8)

# Right panel: time-average unbound fraction profile.

f_mean = np.nanmean(f_unbound_ts, axis=0)
valid_f = np.isfinite(f_mean)
*styled_ax(ax5b, xlabel=r”$\langle f*{\rm unbound} \rangle_t$”,
title=“Time avg.”)
ax5b.plot(f_mean[valid_f], r_mid_sph[valid_f], color=”#ff9944”, lw=2.0)
ax5b.axvline(0.5, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.4)
ax5b.set_xlim(0, 1.05)
ax5b.set_yscale(“log”)
ax5b.set_ylim(R_BINS[0], R_BINS[-1])
ax5b.tick_params(labelleft=False)

fig5.suptitle(“Tidal Stripping Radial Profile”, fontsize=12)
fig5.savefig(os.path.join(OUT_DIR, “section22_stream_fraction_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5)
print(”  Saved: section22_stream_fraction_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 6 — TIDAL HEATING ΔE(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# Tidal heating is measured as the snapshot-to-snapshot change in mean

# specific kinetic energy per radial shell.  Red = heating (kinetic energy

# injected), blue = cooling (bulk infall removes kinetic energy temporarily).

# The heatmap reveals the radial structure of tidal shocks: sharp horizontal

# features at pericentre, propagating outward as a sound-speed-like

# “tidal wave” after first passage.

print(”[Fig 6]  Tidal heating heatmap …”)

dE_max = np.nanpercentile(np.abs(dEkin_dt_ts[np.isfinite(dEkin_dt_ts)]), 97)

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 1], “wspace”: 0.06})

im6 = ax6a.imshow(
dEkin_dt_ts.T,
aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“seismic”,
norm=TwoSlopeNorm(vmin=-dE_max, vcenter=0.0, vmax=dE_max),
)
*styled_ax(ax6a, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Tidal Heating  $\Delta E*{\rm kin}(r,,t)$  [km$^2$ s$^{-2}$ snap$^{-1}$]”)
ax6a.set_yscale(“log”)
cb6 = fig6.colorbar(im6, ax=ax6a, pad=0.01)
cb6.set_label(r”$\Delta E_{\rm kin}$  (red = heating, blue = cooling)”, fontsize=8)
cb6.ax.axhline(0, color=”#ffffff”, lw=0.6, ls=”–”, alpha=0.5)

# Right: time-average heating profile.

dE_mean = np.nanmean(dEkin_dt_ts, axis=0)
valid_dE = np.isfinite(dE_mean)
*styled_ax(ax6b, xlabel=r”$\langle\Delta E*{\rm kin}\rangle_t$”, title=“Time avg.”)
ax6b.plot(dE_mean[valid_dE], r_mid_sph[valid_dE], color=”#e8673a”, lw=2.0)
ax6b.axvline(0, color=”#555577”, lw=0.8, ls=”–”)
ax6b.set_yscale(“log”)
ax6b.set_ylim(R_BINS[0], R_BINS[-1])
ax6b.tick_params(labelleft=False)

fig6.suptitle(“Tidal Energy Injection”, fontsize=12)
fig6.savefig(os.path.join(OUT_DIR, “section22_heating_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig6)
print(”  Saved: section22_heating_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 7 — TIDAL HEATING PROFILES AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 7]  Tidal heating profiles …”)

fig7, ax7 = plt.subplots(figsize=(9, 6), facecolor=BG)
*styled_ax(ax7, xlabel=“r [kpc]”,
ylabel=r”$\Delta E*{\rm kin}$ [km$^2$ s$^{-2}$ snap$^{-1}$]”,
title=“Tidal Heating Profiles at Key Epochs”, log_x=True)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
y     = dEkin_dt_ts[k_idx, :]
valid = np.isfinite(y)
if valid.any():
ax7.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax7.axhline(0, color=”#555577”, lw=1.0, ls=”–”, alpha=0.6)
ax7.text(R_BINS[0] * 1.2, 0, “heating →”, color=MUTED, fontsize=7, va=“bottom”)
ax7.text(R_BINS[0] * 1.2, 0, “← cooling”, color=MUTED, fontsize=7, va=“top”)
ax7.set_xlim(R_BINS[0], R_BINS[-1])
ax7.legend(fontsize=8)

fig7.savefig(os.path.join(OUT_DIR, “section22_heating_profiles.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig7)
print(”  Saved: section22_heating_profiles.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 8 — TIDAL TENSOR EIGENVALUES

# ══════════════════════════════════════════════════════════════════════════════

# 

# The three eigenvalues λ_1 ≥ λ_2 ≥ λ_3 of the tidal tensor encode the

# direction and strength of the tidal distortion:

# λ_1 > 0  →  compression along this axis

# λ_3 < 0  →  stretching (tidal tails form along this axis)

# 

# The tidal anisotropy (λ_1 − λ_3) / (|λ_1| + |λ_3|) peaks at pericentre and

# quantifies how asymmetric the tidal distortion is — purely compressive

# (λ_1 = λ_2 = λ_3) vs. uniaxial stretching (λ_3 ≪ λ_1, λ_2).

print(”[Fig 8]  Tidal tensor eigenvalues …”)

fig8, axes8 = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

_styled_ax(axes8[0], ylabel=r”$\lambda$ [km$^2$ s$^{-2}$ kpc$^{-2}$]”,
title=“Tidal Tensor Eigenvalues at MW–M31 Midpoint”)
axes8[0].plot(time_tensor, T_eig1_arr, color=”#ff9944”, lw=1.8, label=r”$\lambda_1$ (compression)”)
axes8[0].plot(time_tensor, T_eig2_arr, color=”#4a8fff”, lw=1.5, label=r”$\lambda_2$”)
axes8[0].plot(time_tensor, T_eig3_arr, color=”#e8673a”, lw=1.8, label=r”$\lambda_3$ (stretching)”)
axes8[0].axhline(0, color=”#555577”, lw=0.7, ls=”–”)
axes8[0].legend(fontsize=8)

tidal_aniso = (T_eig1_arr - T_eig3_arr) / (
np.abs(T_eig1_arr) + np.abs(T_eig3_arr) + 1e-30
)
_styled_ax(axes8[1], xlabel=time_label,
ylabel=“Tidal anisotropy”,
title=r”Tidal Anisotropy  $(\lambda_1 - \lambda_3) / (|\lambda_1| + |\lambda_3|)$”)
axes8[1].plot(time_tensor, tidal_aniso, color=”#aa55ff”, lw=1.8)
axes8[1].fill_between(time_tensor,
np.where(np.isfinite(tidal_aniso), tidal_aniso, 0),
alpha=0.12, color=”#aa55ff”)

fig8.savefig(os.path.join(OUT_DIR, “section22_tidal_tensor_eigen.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig8)
print(”  Saved: section22_tidal_tensor_eigen.png”)

# ── Tidal anisotropy standalone figure ────────────────────────────────────────

fig8b, ax8b = plt.subplots(figsize=(10, 4), facecolor=BG)
_styled_ax(ax8b, xlabel=time_label, ylabel=“Tidal anisotropy”,
title=r”Tidal Anisotropy  $(\lambda_1-\lambda_3)/(|\lambda_1|+|\lambda_3|)$”)
ax8b.plot(time_tensor, tidal_aniso, color=”#aa55ff”, lw=2.0)
ax8b.fill_between(time_tensor,
np.where(np.isfinite(tidal_aniso), tidal_aniso, 0),
alpha=0.15, color=”#aa55ff”)
fig8b.savefig(os.path.join(OUT_DIR, “section22_tidal_anisotropy.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig8b)
print(”  Saved: section22_tidal_anisotropy.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 9 — STREAM DETECTION: (r, v_r) SCATTER AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# Unbound particles plotted in (r, v_r) phase space form the characteristic

# “S-shaped” or fan-shaped stream features.  Bound particles cluster tightly

# near v_r ≈ 0 (virialised orbits); unbound particles scatter to large |v_r|

# at large r.  The separation of bound and unbound populations in this plane

# is the basis of observational stream detection methods (e.g., interloper

# rejection in globular cluster surveys).

print(”[Fig 9]  Stream (r, v_r) scatter …”)

fig9, axes9 = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.06})

for col, (bi, label, color) in enumerate(zip(profile_bi, PROFILE_LABELS, PROFILE_COLORS)):
ax = axes9[col]
_styled_ax(ax, xlabel=“r [kpc]”, title=label)
if col == 0:
ax.set_ylabel(r”$v_r$ [km s$^{-1}$]”, fontsize=9)

```
snap_num = bound_snap_nums[bi]
mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
    continue

try:
    snap_data = load_snapshot_particles(mw_file, m31_file)
    MW_obj    = CenterOfMass(mw_file,  PTYPE)
    M31_obj   = CenterOfMass(m31_file, PTYPE)
except Exception:
    continue

pos   = snap_data["pos"]
m_p   = snap_data["m_msun"]
r_mag = np.linalg.norm(pos, axis=1)

vx_a = np.concatenate((MW_obj.vx, M31_obj.vx))
vy_a = np.concatenate((MW_obj.vy, M31_obj.vy))
vz_a = np.concatenate((MW_obj.vz, M31_obj.vz))
m_r  = np.concatenate((MW_obj.m, M31_obj.m))
x_a  = np.concatenate((MW_obj.x, M31_obj.x))
y_a  = np.concatenate((MW_obj.y, M31_obj.y))
z_a  = np.concatenate((MW_obj.z, M31_obj.z))

xcom, ycom, zcom = MW_obj.COMdefine(x_a, y_a, z_a, m_r)
dr_c = np.sqrt((x_a-xcom)**2 + (y_a-ycom)**2 + (z_a-zcom)**2)
inn  = dr_c < 15.0
if inn.sum() >= 5:
    wi   = m_p[inn]
    vxc  = np.sum(wi*vx_a[inn])/wi.sum()
    vyc  = np.sum(wi*vy_a[inn])/wi.sum()
    vzc  = np.sum(wi*vz_a[inn])/wi.sum()
else:
    vxc = vyc = vzc = 0.0

vel   = np.vstack((vx_a-vxc, vy_a-vyc, vz_a-vzc)).T
v_mag2 = np.sum(vel**2, axis=1)
phi    = specific_potential(r_mag, m_p, R_BINS)
E_tot  = 0.5 * v_mag2 + phi

with np.errstate(divide="ignore", invalid="ignore"):
    r_hat = np.where(r_mag[:,None] > 0, pos/r_mag[:,None], 0.0)
v_r = np.einsum("ij,ij->i", vel, r_hat)

bound_m   = E_tot < 0
unbound_m = ~bound_m

# Subsample for display (max 20k points each).
def _sub(mask, n=20000):
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = np.random.choice(idx, n, replace=False)
    return idx

ib = _sub(bound_m)
iu = _sub(unbound_m)

ax.scatter(r_mag[ib], v_r[ib], s=0.5, alpha=0.3, color=color,    rasterized=True)
ax.scatter(r_mag[iu], v_r[iu], s=0.5, alpha=0.4, color="#ff5566", rasterized=True)
ax.set_xscale("log")
ax.set_xlim(R_BINS[0], R_BINS[-1])
ax.set_ylim(-600, 600)
```

fig9.suptitle(r”$(r,,v_r)$ Phase Space — Bound (colour) vs. Unbound (red)”, fontsize=11)
fig9.savefig(os.path.join(OUT_DIR, “section22_stream_rv.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig9)
print(”  Saved: section22_stream_rv.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 10 — 2D STREAM MAP AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# The projected distribution of unbound particles in the x–y plane directly

# shows the tidal streams and bridges formed during the merger.  Early

# snapshots should show two distinct galaxy positions; pericentre passages

# produce the first bridge; late snapshots show the merged system with trailing

# tidal tails extending to several hundred kpc.

print(”[Fig 10]  2D stream maps …”)

fig10, axes10 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG,
sharey=True, gridspec_kw={“wspace”: 0.04})

for col, (fi_s, label) in enumerate(zip(
np.linspace(0, n_stream_frames-1, 5, dtype=int), PROFILE_LABELS
)):
ax = axes10[col]
ax.set_facecolor(BG)
H  = stream_maps[fi_s]
Hs = gaussian_filter(np.where(H > 0, H, 0.0), sigma=2.0)
H_log = np.where(Hs > 0, np.log10(Hs), np.nan)

```
ax.imshow(H_log.T, origin="lower", aspect="equal",
          extent=[-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT,
                  -STREAM_MAP_EXTENT, STREAM_MAP_EXTENT],
          cmap="inferno",
          vmin=np.nanpercentile(H_log[np.isfinite(H_log)], 5) if np.isfinite(H_log).any() else 0,
          vmax=np.nanpercentile(H_log[np.isfinite(H_log)], 99) if np.isfinite(H_log).any() else 10)
ax.set_title(label, fontsize=9, color="#c8c8e8")
ax.tick_params(colors="#9090b0", labelsize=7)
if col == 0:
    ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")
ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")
```

fig10.suptitle(r”Tidal Stream Map — Unbound Particle Distribution  $\Sigma_{\rm unbound}(x,y)$”,
fontsize=11)
fig10.savefig(os.path.join(OUT_DIR, “section22_stream_map.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig10)
print(”  Saved: section22_stream_map.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 11 — MORPHOLOGICAL AXIS RATIOS b/a, c/a

# ══════════════════════════════════════════════════════════════════════════════

# 

# Axis ratios from the inertia tensor track morphological evolution:

# b/a → 1  →  axisymmetric (oblate or spherical)

# c/a → 0  →  prolate elongation (tidal distortion)

# 

# Comparing inner, mid, and outer aperture axis ratios reveals at which

# radii the shape distortion penetrates first (outside-in, as expected for

# tidal distortion from the external field).

print(”[Fig 11]  Morphological axis ratios …”)

fig11, axes11 = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

_styled_ax(axes11[0], ylabel=“b/a”,
title=“Intermediate-to-Major Axis Ratio  b/a  (1 = oblate/spherical)”)
for arr, color, label in [
(ba_inner_arr, “#4a8fff”, f”Inner r < {INERTIA_R_INNER:.0f} kpc”),
(ba_mid_arr,   “#00d4aa”, f”Mid {INERTIA_R_INNER:.0f}–{INERTIA_R_MID:.0f} kpc”),
(ba_outer_arr, “#ff9944”, f”Outer r > {INERTIA_R_MID:.0f} kpc”),
(ba_arr,       “#aaaacc”, “Global”),
]:
valid = np.isfinite(arr)
if valid.any():
axes11[0].plot(time_inertia[valid], arr[valid],
color=color, lw=1.8, label=label)
axes11[0].set_ylim(0, 1.05)
axes11[0].legend(fontsize=8)

_styled_ax(axes11[1], xlabel=time_label, ylabel=“c/a”,
title=“Minor-to-Major Axis Ratio  c/a  (→ 0 = prolate/cigar)”)
for arr, color in [(ca_inner_arr, “#4a8fff”), (ca_mid_arr, “#00d4aa”),
(ca_outer_arr, “#ff9944”), (ca_arr, “#aaaacc”)]:
valid = np.isfinite(arr)
if valid.any():
axes11[1].plot(time_inertia[valid], arr[valid], color=color, lw=1.8)
axes11[1].set_ylim(0, 1.05)

fig11.savefig(os.path.join(OUT_DIR, “section22_inertia_axisratios.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig11)
print(”  Saved: section22_inertia_axisratios.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 12 — b/a(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# The per-bin axis ratio b/a(r, t) shows at which radius and epoch the halo

# becomes prolate.  A band of low b/a (elongated shells) propagating inward

# after pericentre would indicate that tidal distortion is working its way

# into the interior of the halo.

print(”[Fig 12]  b/a(r,t) heatmap …”)

t_inertia_min = np.nanmin(time_inertia)
t_inertia_max = np.nanmax(time_inertia)

fig12, ax12 = plt.subplots(figsize=(11, 5), facecolor=BG)
im12 = ax12.imshow(
ba_profile_ts.T,
aspect=“auto”, origin=“lower”,
extent=[t_inertia_min, t_inertia_max, R_BINS[0], R_BINS[-1]],
cmap=“viridis_r”, vmin=0.3, vmax=1.0,
)
_styled_ax(ax12, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Per-Shell Axis Ratio  $b/a(r,,t)$  (dark = prolate, light = oblate)”)
ax12.set_yscale(“log”)
cb12 = fig12.colorbar(im12, ax=ax12, pad=0.01)
cb12.set_label(r”$b/a$”, fontsize=9)

fig12.savefig(os.path.join(OUT_DIR, “section22_inertia_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig12)
print(”  Saved: section22_inertia_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 13 — AXIS RATIO PROFILES AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 13]  Axis ratio profiles …”)

inertia_profile_bi = [np.argmin(np.abs(time_inertia - time_arr[k]))
for k in PROFILE_INDICES]

fig13, ax13 = plt.subplots(figsize=(9, 6), facecolor=BG)
_styled_ax(ax13, xlabel=“r [kpc]”, ylabel=“b/a”,
title=r”Axis Ratio Profile  $b/a(r)$  at Key Epochs”, log_x=True)

for ii, color, label in zip(inertia_profile_bi, PROFILE_COLORS, PROFILE_LABELS):
y     = ba_profile_ts[ii, :]
valid = np.isfinite(y)
if valid.any():
ax13.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax13.axhline(1.0, color=”#555577”, lw=0.7, ls=”–”, alpha=0.5)
ax13.text(R_BINS[0] * 1.2, 1.01, “spherical”, color=MUTED, fontsize=7)
ax13.set_xlim(R_BINS[0], R_BINS[-1])
ax13.set_ylim(0.2, 1.05)
ax13.legend(fontsize=8)

fig13.savefig(os.path.join(OUT_DIR, “section22_inertia_profiles.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig13)
print(”  Saved: section22_inertia_profiles.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 14 — MW–M31 SEPARATION AND RELATIVE SPEED

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 14]  MW–M31 separation and relative speed …”)

fig14, (ax14a, ax14b) = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

_styled_ax(ax14a, ylabel=“Separation [kpc]”,
title=“MW–M31 Orbital Trajectory”)
ax14a.plot(time_arr, separation_arr, color=”#4a8fff”, lw=2.0)
ax14a.set_yscale(“log”)

*styled_ax(ax14b, xlabel=time_label,
ylabel=r”$v*{\rm rel}$ [kpc snap$^{-1}$]”,
title=“Relative Approach/Recession Speed”)
valid_vrel = np.isfinite(v_rel_arr)
ax14b.plot(time_arr[valid_vrel], v_rel_arr[valid_vrel], color=”#e8673a”, lw=1.8)
ax14b.axhline(0, color=”#555577”, lw=0.8, ls=”–”)
ax14b.text(time_arr[valid_vrel][0], 0.05, “approaching →”, color=MUTED, fontsize=7, va=“bottom”)

fig14.savefig(os.path.join(OUT_DIR, “section22_mw_m31_separation.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig14)
print(”  Saved: section22_mw_m31_separation.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.5 — ANIMATIONS                                                        ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════════════

# ANIMATION 1 — 2D STREAM MAP

# ══════════════════════════════════════════════════════════════════════════════

# 

# Plays through the 2D unbound particle maps stored in stream_maps.

# A side panel shows the bound fraction scalar, which falls as the streams

# grow — connecting the spatial morphology to the mass-loss rate.

print(”\n[Anim 1]  Stream map animation …”)

fig_a1, (axSM, axBF) = plt.subplots(
1, 2, figsize=(13, 5.5), facecolor=BG,
gridspec_kw={“width_ratios”: [3, 1], “wspace”: 0.08},
)
axSM.set_facecolor(BG)
axBF.set_facecolor(BG)

H0 = stream_maps[0]
Hs0 = gaussian_filter(np.where(H0 > 0, H0, 0.0), sigma=2.0)
H0_log = np.where(Hs0 > 0, np.log10(Hs0), np.nan)

all_vals = stream_maps[stream_maps > 0]
vmin_sm = np.log10(np.percentile(all_vals, 10)) if all_vals.size > 0 else 0
vmax_sm = np.log10(np.percentile(all_vals, 99)) if all_vals.size > 0 else 10

im_sm = axSM.imshow(H0_log.T, origin=“lower”, aspect=“equal”,
extent=[-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT,
-STREAM_MAP_EXTENT, STREAM_MAP_EXTENT],
cmap=“inferno”, vmin=vmin_sm, vmax=vmax_sm)
axSM.set_xlabel(“x [kpc]”, color=”#c8c8e8”)
axSM.set_ylabel(“y [kpc]”, color=”#c8c8e8”)

# Bound fraction line plot — accumulates up to current frame.

bf_line, = axBF.plot([], [], color=”#00d4aa”, lw=1.8)
axBF.set_xlim(np.nanmin(time_bound), np.nanmax(time_bound))
axBF.set_ylim(0, 1.05)
axBF.set_xlabel(time_label, color=”#c8c8e8”)
axBF.set_ylabel(“Bound fraction”, color=”#c8c8e8”)
axBF.set_title(”$f_{\rm bound}$”, color=”#c8c8e8”, fontsize=10)
axBF.axhline(0.5, color=”#555577”, lw=0.6, ls=”–”, alpha=0.5)
title_a1 = fig_a1.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_stream_anim(frame_idx):
fi_s = stream_anim_snap_ids[frame_idx]
H    = stream_maps[fi_s]
Hs   = gaussian_filter(np.where(H > 0, H, 0.0), sigma=2.0)
H_log = np.where(Hs > 0, np.log10(Hs), np.nan)
im_sm.set_data(H_log.T)

```
# Bound fraction history up to this frame.
t_cur = time_bound[fi_s] if fi_s < len(time_bound) else np.nan
valid = np.isfinite(time_bound[:fi_s+1]) & np.isfinite(bound_frac_arr[:fi_s+1])
bf_line.set_data(time_bound[:fi_s+1][valid], bound_frac_arr[:fi_s+1][valid])

t_str = (f"{t_cur:.2f} Gyr" if (np.isfinite(t_cur) and time_is_gyr)
         else f"Bound snap {fi_s}")
title_a1.set_text(f"Tidal Streams  ·  {t_str}")
return [im_sm, bf_line]
```

ani_sm = animation.FuncAnimation(
fig_a1, _update_stream_anim, frames=n_stream_frames,
interval=1000 // ANIM_FPS_22, blit=True,
)
writer_22 = animation.FFMpegWriter(
fps=ANIM_FPS_22, bitrate=ANIM_BITRATE_22,
metadata=dict(title=“MW-M31 Tidal Stream Animation”),
)
ani_sm.save(os.path.join(OUT_DIR, “section22_animation_streams.mp4”),
writer=writer_22, dpi=ANIM_DPI_22)
plt.close(fig_a1)
print(”  Saved: section22_animation_streams.mp4”)

# ══════════════════════════════════════════════════════════════════════════════

# ANIMATION 2 — BOUND MASS DENSITY PROFILE

# ══════════════════════════════════════════════════════════════════════════════

# 

# Three-panel profile animation:

# Left  : ρ_bound(r) and ρ_unbound(r)

# Centre: f_unbound(r) — fraction of each shell that is unbound

# Right : scalar bound fraction f_bound(t) history

print(”[Anim 2]  Bound mass profile animation …”)

fig_a2, axes_a2 = plt.subplots(
1, 3, figsize=(15, 5.5), facecolor=BG,
gridspec_kw={“wspace”: 0.32},
)
ax_bprof, ax_fub, ax_bf = axes_a2
for ax in axes_a2:
ax.set_facecolor(BG)

# Y-limits from full dataset.

rho_all = np.concatenate([rho_bound_ts.ravel(), rho_unbound_ts.ravel()])
rho_all = rho_all[np.isfinite(rho_all) & (rho_all > 0)]
rho_ymin = rho_all.min() * 0.3 if rho_all.size > 0 else 1e2
rho_ymax = rho_all.max() * 3.0 if rho_all.size > 0 else 1e12

ax_bprof.set_xscale(“log”); ax_bprof.set_yscale(“log”)
ax_bprof.set_xlim(R_BINS[0], R_BINS[-1])
ax_bprof.set_ylim(rho_ymin, rho_ymax)
ax_bprof.set_xlabel(“r [kpc]”, color=”#c8c8e8”)
ax_bprof.set_ylabel(r”$\rho$ [M$_\odot$ kpc$^{-3}$]”, color=”#c8c8e8”)
ax_bprof.set_title(“Bound / unbound density”, color=”#c8c8e8”, fontsize=10)

ax_fub.set_xscale(“log”)
ax_fub.set_xlim(R_BINS[0], R_BINS[-1])
ax_fub.set_ylim(0, 1.05)
ax_fub.set_xlabel(“r [kpc]”, color=”#c8c8e8”)
ax_fub.set_ylabel(r”$f_{\rm unbound}$”, color=”#c8c8e8”)
ax_fub.set_title(“Unbound fraction”, color=”#c8c8e8”, fontsize=10)
ax_fub.axhline(0.5, color=”#555577”, lw=0.7, ls=”–”)

ax_bf.set_xlim(np.nanmin(time_bound), np.nanmax(time_bound))
ax_bf.set_ylim(0, 1.05)
ax_bf.set_xlabel(time_label, color=”#c8c8e8”)
ax_bf.set_ylabel(r”$f_{\rm bound}$”, color=”#c8c8e8”)
ax_bf.set_title(“Bound fraction”, color=”#c8c8e8”, fontsize=10)

line_bound, = ax_bprof.plot([], [], color=”#4a8fff”,  lw=2.0, label=“Bound”)
line_unb,   = ax_bprof.plot([], [], color=”#ff5566”,  lw=1.5, ls=”–”, label=“Unbound”)
line_fub,   = ax_fub.plot([],   [], color=”#ff9944”,  lw=2.0)
line_bfhist,= ax_bf.plot([],    [], color=”#00d4aa”,  lw=1.8)
vline_bf    = ax_bf.axvline(np.nan, color=”#ffffff”,  lw=0.8, ls=”–”, alpha=0.5)
ax_bprof.legend(fontsize=8)

title_a2 = fig_a2.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_bound_anim(frame_idx):
bi = bound_anim_snap_ids[frame_idx]

```
def _xy_rho(arr):
    v = np.isfinite(arr) & (arr > 0)
    return r_mid_sph[v], arr[v]

line_bound.set_data(*_xy_rho(rho_bound_ts[bi, :]))
line_unb.set_data(*_xy_rho(rho_unbound_ts[bi, :]))

fub = f_unbound_ts[bi, :]
vf  = np.isfinite(fub)
line_fub.set_data(r_mid_sph[vf], fub[vf])

t_cur = time_bound[bi]
valid  = np.isfinite(time_bound[:bi+1]) & np.isfinite(bound_frac_arr[:bi+1])
line_bfhist.set_data(time_bound[:bi+1][valid], bound_frac_arr[:bi+1][valid])
vline_bf.set_xdata([t_cur, t_cur])

t_str = (f"{t_cur:.2f} Gyr" if (np.isfinite(t_cur) and time_is_gyr)
         else f"Bound snap {bi}")
title_a2.set_text(f"Bound Mass Structure  ·  {t_str}")
return [line_bound, line_unb, line_fub, line_bfhist, vline_bf]
```

ani_bm = animation.FuncAnimation(
fig_a2, _update_bound_anim, frames=n_bound_anim_frames,
interval=1000 // ANIM_FPS_22, blit=True,
)
ani_bm.save(os.path.join(OUT_DIR, “section22_animation_boundmass.mp4”),
writer=writer_22, dpi=ANIM_DPI_22)
plt.close(fig_a2)
print(”  Saved: section22_animation_boundmass.mp4”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.6 — MASTER SUMMARY PANEL                                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n[Summary]  Master summary panel …”)

fig_sum = plt.figure(figsize=(16, 14), facecolor=BG)
gs_sum  = gridspec.GridSpec(3, 2, figure=fig_sum,
hspace=0.42, wspace=0.32,
left=0.08, right=0.97,
top=0.94, bottom=0.06)

# (0,0) Separation and tidal radius

ax_s00 = fig_sum.add_subplot(gs_sum[0, 0])
ax_s00.set_facecolor(BG)
ax_s00.plot(time_arr, separation_arr, color=”#4a8fff”, lw=1.5, label=“Separation”)
ax_s00.plot(time_arr, r_tidal_arr,    color=”#ff9944”, lw=1.5, label=r”$r_t$”)
ax_s00.set_yscale(“log”)
ax_s00.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s00.set_ylabel(“kpc”, fontsize=8, color=”#c8c8e8”)
ax_s00.set_title(“Separation & Tidal Radius”, fontsize=9, color=”#c8c8e8”)
ax_s00.legend(fontsize=7)

# (0,1) Bound fraction

ax_s01 = fig_sum.add_subplot(gs_sum[0, 1])
ax_s01.set_facecolor(BG)
ax_s01.plot(time_bound, bound_frac_arr, color=”#00d4aa”, lw=1.8)
ax_s01.set_ylim(0, 1.05)
ax_s01.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s01.set_ylabel(r”$f_{\rm bound}$”, fontsize=8, color=”#c8c8e8”)
ax_s01.set_title(“Bound Mass Fraction”, fontsize=9, color=”#c8c8e8”)

# (1,0) f_unbound(r,t) heatmap

ax_s10 = fig_sum.add_subplot(gs_sum[1, 0])
ax_s10.set_facecolor(BG)
im_s10 = ax_s10.imshow(f_unbound_ts.T, aspect=“auto”, origin=“lower”,
extent=[t_bound_min, t_bound_max, R_BINS[0], R_BINS[-1]],
cmap=“hot”, vmin=0, vmax=1)
ax_s10.set_yscale(“log”)
ax_s10.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s10.set_ylabel(“r [kpc]”, fontsize=8, color=”#c8c8e8”)
ax_s10.set_title(r”$f_{\rm unbound}(r,t)$”, fontsize=9, color=”#c8c8e8”)
fig_sum.colorbar(im_s10, ax=ax_s10, shrink=0.8)

# (1,1) Tidal heating heatmap

ax_s11 = fig_sum.add_subplot(gs_sum[1, 1])
ax_s11.set_facecolor(BG)
im_s11 = ax_s11.imshow(dEkin_dt_ts.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“seismic”,
norm=TwoSlopeNorm(vmin=-dE_max, vcenter=0.0, vmax=dE_max))
ax_s11.set_yscale(“log”)
ax_s11.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s11.set_title(r”Tidal Heating $\Delta E_{\rm kin}(r,t)$”, fontsize=9, color=”#c8c8e8”)
fig_sum.colorbar(im_s11, ax=ax_s11, shrink=0.8)

# (2,0) Tidal tensor eigenvalues

ax_s20 = fig_sum.add_subplot(gs_sum[2, 0])
ax_s20.set_facecolor(BG)
ax_s20.plot(time_tensor, T_eig1_arr, color=”#ff9944”, lw=1.2, label=r”$\lambda_1$”)
ax_s20.plot(time_tensor, T_eig2_arr, color=”#4a8fff”, lw=1.2, label=r”$\lambda_2$”)
ax_s20.plot(time_tensor, T_eig3_arr, color=”#e8673a”, lw=1.2, label=r”$\lambda_3$”)
ax_s20.axhline(0, color=”#555577”, lw=0.6, ls=”–”)
ax_s20.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s20.set_ylabel(r”$\lambda$ [km$^2$s$^{-2}$kpc$^{-2}$]”, fontsize=8, color=”#c8c8e8”)
ax_s20.set_title(“Tidal Tensor Eigenvalues”, fontsize=9, color=”#c8c8e8”)
ax_s20.legend(fontsize=6)

# (2,1) Axis ratios

ax_s21 = fig_sum.add_subplot(gs_sum[2, 1])
ax_s21.set_facecolor(BG)
for arr, color, label in [(ba_arr, “#4a8fff”, “b/a”),
(ca_arr, “#ff9944”, “c/a”)]:
valid = np.isfinite(arr)
if valid.any():
ax_s21.plot(time_inertia[valid], arr[valid], color=color, lw=1.5, label=label)
ax_s21.set_ylim(0, 1.05)
ax_s21.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s21.set_ylabel(“Axis ratio”, fontsize=8, color=”#c8c8e8”)
ax_s21.set_title(“Morphological Axis Ratios”, fontsize=9, color=”#c8c8e8”)
ax_s21.legend(fontsize=7)

fig_sum.suptitle(
“Section 22 Summary  ·  Tidal Field & Stripping Diagnostics”,
fontsize=13, color=”#c8c8e8”, fontweight=“bold”,
)
fig_sum.savefig(os.path.join(OUT_DIR, “section22_summary_panel.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig_sum)
print(”  Saved: section22_summary_panel.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §22.7 — SECTION COMPLETE                                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  SECTION 22 COMPLETE”)
print(”=”*80)
outputs_22 = [
“section22_tidal_radius.png”,
“section22_bound_fraction.png”,
“section22_stripping_rate.png”,
“section22_bound_profiles.png”,
“section22_stream_fraction_heatmap.png”,
“section22_heating_heatmap.png”,
“section22_heating_profiles.png”,
“section22_tidal_tensor_eigen.png”,
“section22_tidal_anisotropy.png”,
“section22_stream_rv.png”,
“section22_stream_map.png”,
“section22_inertia_axisratios.png”,
“section22_inertia_heatmap.png”,
“section22_inertia_profiles.png”,
“section22_mw_m31_separation.png”,
“section22_animation_streams.mp4”,
“section22_animation_boundmass.mp4”,
“section22_summary_panel.png”,
]
for fn in outputs_22:
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp) / 1e6 if os.path.isfile(fp) else 0.0
kind = “animation” if fn.endswith(”.mp4”) else “figure”
print(f”  {fn:<55} {size:6.2f} MB  [{kind}]”)
print(”=”*80)
