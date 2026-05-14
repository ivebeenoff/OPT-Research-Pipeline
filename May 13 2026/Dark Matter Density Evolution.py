# “””

# SECTION 28 — DARK MATTER DENSITY EVOLUTION TRACKING

Author  : Abhinav Vatsa

Continuation of the MW–M31 analysis pipeline.  All globals (SNAPSHOTS, ns,
R_BINS, nb_sph, r_mid_sph, OUT_DIR, MASS_UNIT_MSUN, MIN_PART_SHELL,
G_KPC_KMS2_MSUN, PROFILE_INDICES, PROFILE_LABELS, PROFILE_COLORS, time_arr,
time_label, time_is_gyr, tmpdir, PTYPE, load_snapshot_particles, CenterOfMass)
are inherited and must be defined before this section is executed.

## Physical motivation

This section is dedicated entirely to tracking how dark matter density
evolves — in every form that is physically useful:

(A) SPHERICAL PROFILE ρ(r, t)
The radial density profile at every snapshot.  The central density
ρ_0(t) and scale radius r_s(t) from NFW fits capture compaction
and expansion of the halo core through the merger.

(B) 3D VOLUME DENSITY CUBE
A 3D mass-weighted density grid evaluated at selected epochs,
enabling isosurface and slice visualisation of the merger geometry.

(C) PHASE-SPACE DENSITY f(r, σ_r)
The combination ρ / σ_r³ is the coarse-grained phase-space density
(Lynden-Bell 1967).  It is a pseudo-conserved quantity that decreases
monotonically under mixing — tracking it reveals the degree of
phase-space compression vs. dilution at each radius.

(D) DENSITY CONTRAST δ(r, t) = ρ(r,t)/⟨ρ⟩(t) − 1
The overdensity relative to the mean interior density at each time.
Peaks in δ mark the density enhancements that would collapse to form
structure in a cosmological context.

(E) DENSITY POWER SPECTRUM P(k, t)
The 1D spherically-averaged power spectrum of the density field,
computed from the 3D grid via FFT.  Tracks how power is redistributed
between scales as the merger progresses.

## Outputs

section28_rho_profiles.png          ρ(r) at 5 epochs with NFW fits
section28_rho_heatmap.png           ρ(r,t) heatmap (log scale)
section28_rho_central.png           ρ_0(t) and r_s(t) scalar evolution
section28_phase_space_density.png   f(r,σ) = ρ/σ_r³ profiles at 5 epochs
section28_psd_heatmap.png           Phase-space density f(r,t) heatmap
section28_overdensity.png           δ(r,t) overdensity heatmap
section28_3d_slice.png              2D midplane slice of 3D density at 5 epochs
section28_power_spectrum.png        P(k,t) density power spectra at 5 epochs
section28_pk_heatmap.png            P(k,t) heatmap showing power evolution
section28_animation_rho.mp4         ρ(r) profile animation
section28_animation_slice.mp4       2D density slice animation
section28_summary_panel.png         Master 4-panel summary

===============================================================================
“””

import numpy as np
import matplotlib
matplotlib.use(“Agg”)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import os
import time
import warnings

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.0 — CONFIGURATION                                                     ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# 3D density grid parameters.

# GRID_BINS^3 cells covering ±GRID_EXTENT_KPC.

# 64^3 = 262,144 cells — lightweight but sufficient for power spectrum.

# Increase to 128 for better spatial resolution (8× memory cost).

GRID_BINS      = 64
GRID_EXTENT    = 300.0   # [kpc] half-width of the 3D cube

# Snapshots at which to compute the full 3D grid (expensive).

GRID_STEP      = 10      # every 10th snapshot

# Power spectrum k-bins.

N_KBINS        = 30

# NFW fitting range.

NFW_RMIN       = 1.0    # [kpc]
NFW_RMAX       = 200.0  # [kpc]

# Phase-space density: velocity dispersion bin reuse.

# σ_r per radial bin inherited from main loop computation.

# Animation subsampling.

ANIM_FPS_28    = 20
ANIM_DPI_28    = 100
ANIM_BITRATE_28 = 1800
ANIM_STEP_28   = 4

print(”\n” + “=”*80)
print(”  SECTION 28 · Dark Matter Density Evolution Tracking”)
print(”=”*80)
print(f”  3D grid      : {GRID_BINS}³  ±{GRID_EXTENT} kpc”)
print(f”  Grid step    : every {GRID_STEP} snapshots”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.1 — UTILITY FUNCTIONS                                                 ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

def nfw_density(r, rho_s, r_s):
“”“NFW profile: ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]”””
x = r / r_s
return rho_s / (x * (1.0 + x)**2)

def fit_nfw(r_mid, rho_prof):
“”“Fit NFW to ρ(r) in log-space. Returns (rho_s, r_s, chi2, success).”””
mask = ((r_mid >= NFW_RMIN) & (r_mid <= NFW_RMAX) &
np.isfinite(rho_prof) & (rho_prof > 0))
if mask.sum() < 4:
return np.nan, np.nan, np.nan, False
r_f  = r_mid[mask]
ln_p = np.log(rho_prof[mask])
try:
popt, _ = curve_fit(
lambda r, rs, rss: np.log(nfw_density(r, rs, rss)),
r_f, ln_p,
p0=[np.exp(ln_p.max()) * 5, 30.0],
bounds=([1e2, 0.1], [1e16, 300.0]),
maxfev=4000,
)
pred  = np.log(nfw_density(r_f, *popt))
chi2  = np.sum((ln_p - pred)**2) / max(1, mask.sum() - 2)
return popt[0], popt[1], chi2, True
except Exception:
return np.nan, np.nan, np.nan, False

def build_3d_density_grid(pos, m, grid_bins, grid_extent):
“””
Bin particle masses into a 3D cubic grid of shape (grid_bins,)³.

```
Returns
-------
rho_3d : (grid_bins, grid_bins, grid_bins)  — density [M_sun kpc^{-3}]
cell_size : float  — side length of one cell [kpc]
"""
cell_size  = 2.0 * grid_extent / grid_bins
cell_vol   = cell_size**3   # [kpc³]

lim = grid_extent
H, edges = np.histogramdd(
    pos,
    bins=grid_bins,
    range=[[-lim, lim], [-lim, lim], [-lim, lim]],
    weights=m,
)
rho_3d = H / cell_vol   # [M_sun kpc^{-3}]
return rho_3d, cell_size
```

def density_power_spectrum(rho_3d, cell_size, n_kbins):
“””
Compute the spherically-averaged 1D power spectrum P(k) of the
density field via 3D FFT.

```
P(k) = (cell_size³ / V_box) × ⟨|ρ̃(k)|²⟩_shell

where ρ̃(k) is the 3D Fourier transform of the density field and
the average is over all Fourier modes in a thin spherical shell at
wavenumber k.

Parameters
----------
rho_3d    : (N,N,N)  — density grid [M_sun kpc^{-3}]
cell_size : float    — cell size [kpc]
n_kbins   : int      — number of k bins

Returns
-------
k_centres : (n_kbins,)  — wavenumber bin centres [kpc^{-1}]
P_k       : (n_kbins,)  — power spectrum [M_sun² kpc^{-3}]
"""
N       = rho_3d.shape[0]
V_box   = (N * cell_size)**3

# Subtract mean density so we get the overdensity power spectrum.
rho_mean = rho_3d.mean()
delta    = (rho_3d - rho_mean) / (rho_mean + 1e-30)

# 3D FFT (numpy rfft for efficiency on real data).
delta_k = np.fft.rfftn(delta)
Pk_3d   = (np.abs(delta_k)**2) * (cell_size**3 / V_box)

# Wavenumber coordinates.
k_1d    = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi   # [kpc^{-1}]
k_r     = np.fft.rfftfreq(N, d=cell_size) * 2.0 * np.pi

kx, ky, kz = np.meshgrid(k_1d, k_1d, k_r, indexing="ij")
k_mag   = np.sqrt(kx**2 + ky**2 + kz**2)

# Bin into spherical shells.
k_max   = k_mag.max()
k_edges = np.linspace(0, k_max, n_kbins + 1)
k_cents = 0.5 * (k_edges[:-1] + k_edges[1:])
P_k     = np.full(n_kbins, np.nan)

for b in range(n_kbins):
    shell = (k_mag >= k_edges[b]) & (k_mag < k_edges[b+1])
    if shell.sum() > 0:
        P_k[b] = float(Pk_3d[shell].mean())

return k_cents, P_k
```

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.2 — PRE-ALLOCATION                                                    ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# Full-resolution density profile time series (every snapshot).

rho_ts        = np.full((ns, nb_sph), np.nan)   # ρ(r, t)
sigma_r_ts    = np.full((ns, nb_sph), np.nan)   # σ_r(r, t) — for PSD
psd_ts        = np.full((ns, nb_sph), np.nan)   # phase-space density f = ρ/σ_r³
delta_ts      = np.full((ns, nb_sph), np.nan)   # overdensity δ(r, t)
rho0_arr      = np.full(ns, np.nan)             # central density scalar
mean_rho_arr  = np.full(ns, np.nan)             # mean density within 300 kpc

# NFW fit parameters (every snapshot).

nfw_rhos_arr  = np.full(ns, np.nan)
nfw_rs_arr    = np.full(ns, np.nan)
nfw_chi2_arr  = np.full(ns, np.nan)

# 3D grid snapshots.

grid_snap_nums = SNAPSHOTS[::GRID_STEP]
n_grid         = len(grid_snap_nums)
grid_snap_map  = {s: i for i, s in enumerate(grid_snap_nums)}
time_grid      = np.full(n_grid, np.nan)

# Power spectrum — shape (n_grid, N_KBINS).

k_centres_arr  = np.full((n_grid, N_KBINS), np.nan)
Pk_ts          = np.full((n_grid, N_KBINS), np.nan)

# 2D midplane slices z=0 — shape (n_grid, GRID_BINS, GRID_BINS).

slices_2d      = np.zeros((n_grid, GRID_BINS, GRID_BINS))

# Animation frames: store midplane slice at ANIM_STEP intervals.

anim_snap_nums = SNAPSHOTS[::ANIM_STEP_28]
n_anim_frames  = len(anim_snap_nums)
anim_snap_map  = {s: i for i, s in enumerate(anim_snap_nums)}
anim_slices    = np.zeros((n_anim_frames, GRID_BINS, GRID_BINS))

print(f”\n[Pre-alloc]  rho_ts     : {rho_ts.shape}”)
print(f”             Pk_ts      : {Pk_ts.shape}”)
print(f”             Grid snaps : {n_grid}  ·  Anim frames : {n_anim_frames}”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.3 — MAIN LOOP                                                         ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  §28.3 — Main Density Loop”)
print(”=”*80)

t_loop_start = time.perf_counter()
shell_vols   = (4.0/3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)

for i, snap_num in enumerate(SNAPSHOTS):

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

pos   = snap_data["pos"]
m     = snap_data["m_msun"]
r_mag = np.linalg.norm(pos, axis=1)

# ── COM-frame velocities for σ_r ─────────────────────────────────────────
vx_all = np.concatenate((MW_obj.vx, M31_obj.vx))
vy_all = np.concatenate((MW_obj.vy, M31_obj.vy))
vz_all = np.concatenate((MW_obj.vz, M31_obj.vz))
m_raw  = np.concatenate((MW_obj.m,  M31_obj.m))
x_all  = np.concatenate((MW_obj.x,  MW_obj.x))   # note: borrow COMdefine
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

vel   = np.vstack((vx_all-vxcom, vy_all-vycom, vz_all-vzcom)).T
with np.errstate(divide="ignore", invalid="ignore"):
    r_hat = np.where(r_mag[:,None] > 0, pos/r_mag[:,None], 0.0)
v_r = np.einsum("ij,ij->i", vel, r_hat)

# ── Radial shell binning ──────────────────────────────────────────────────
bin_id = np.digitize(r_mag, R_BINS) - 1
M_tot  = m.sum()
M_inner = m[r_mag < GRID_EXTENT].sum()

for b in range(nb_sph):
    mask = bin_id == b
    if mask.sum() < MIN_PART_SHELL:
        continue

    M_bin        = m[mask].sum()
    rho_ts[i, b] = M_bin / shell_vols[b]

    w   = m[mask]; W = w.sum()
    vr_b = v_r[mask]
    vr_m = np.sum(w * vr_b) / W
    sigma_r_ts[i, b] = np.sqrt(np.sum(w*(vr_b - vr_m)**2) / W)

    # Phase-space density f = ρ / σ_r³.
    if sigma_r_ts[i, b] > 0:
        psd_ts[i, b] = rho_ts[i, b] / sigma_r_ts[i, b]**3

# ── Mean density and overdensity ──────────────────────────────────────────
rho0_arr[i]     = rho_ts[i, np.where(np.isfinite(rho_ts[i]))[0][0]] \
                  if np.isfinite(rho_ts[i]).any() else np.nan
mean_rho_arr[i] = M_inner / ((4.0/3.0)*np.pi*GRID_EXTENT**3)

if np.isfinite(mean_rho_arr[i]) and mean_rho_arr[i] > 0:
    delta_ts[i, :] = rho_ts[i, :] / mean_rho_arr[i] - 1.0

# ── NFW fit ───────────────────────────────────────────────────────────────
rho_s, r_s, chi2, ok = fit_nfw(r_mid_sph, rho_ts[i, :])
if ok:
    nfw_rhos_arr[i] = rho_s
    nfw_rs_arr  [i] = r_s
    nfw_chi2_arr[i] = chi2

# ── 3D density grid + power spectrum ─────────────────────────────────────
if snap_num in grid_snap_map:
    gi = grid_snap_map[snap_num]
    time_grid[gi] = time_arr[i] if np.isfinite(time_arr[i]) else float(snap_num)

    rho_3d, cell_size = build_3d_density_grid(pos, m, GRID_BINS, GRID_EXTENT)

    # Midplane slice (z = 0 plane = central z-layer).
    mid_z = GRID_BINS // 2
    slices_2d[gi] = rho_3d[:, :, mid_z]

    # Power spectrum.
    try:
        k_c, P_k = density_power_spectrum(rho_3d, cell_size, N_KBINS)
        k_centres_arr[gi] = k_c
        Pk_ts        [gi] = P_k
    except Exception as exc:
        warnings.warn(f"P(k) failed for snap {snap_num}: {exc}")

# ── Animation slice ───────────────────────────────────────────────────────
if snap_num in anim_snap_map:
    ai = anim_snap_map[snap_num]
    rho_3d_a, _ = build_3d_density_grid(pos, m, GRID_BINS, GRID_EXTENT)
    anim_slices[ai] = rho_3d_a[:, :, GRID_BINS // 2]

if (i + 1) % 100 == 0:
    elapsed = time.perf_counter() - t_loop_start
    print(f"  snap {snap_num:04d}  ρ_0={rho0_arr[i]:.2e}  "
          f"r_s={nfw_rs_arr[i]:.1f} kpc  [{elapsed:.0f}s]")
```

print(f”\n[Loop done]  {time.perf_counter()-t_loop_start:.0f}s total”)

t_min = np.nanmin(time_arr)
t_max = np.nanmax(time_arr)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.4 — FIGURES                                                           ║

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

# FIGURE 1 — ρ(r) PROFILES WITH NFW FITS AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

print(”\n[Fig 1]  ρ(r) profiles with NFW fits …”)

r_plot = np.logspace(np.log10(NFW_RMIN), np.log10(NFW_RMAX), 200)

fig1, ax1 = plt.subplots(figsize=(9, 6), facecolor=BG)
*ax(ax1, xlabel=“r [kpc]”,
ylabel=r”$\rho(r)$  [M$*\odot$ kpc$^{-3}$]”,
title=r”DM Density Profile $\rho(r)$ with NFW Fits”,
log_x=True, log_y=True)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
y     = rho_ts[k_idx, :]
valid = np.isfinite(y) & (y > 0)
if valid.any():
ax1.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

```
# NFW overlay.
if np.isfinite(nfw_rhos_arr[k_idx]) and np.isfinite(nfw_rs_arr[k_idx]):
    ax1.plot(r_plot,
             nfw_density(r_plot, nfw_rhos_arr[k_idx], nfw_rs_arr[k_idx]),
             color=color, lw=1.0, ls="--", alpha=0.6)
```

# Reference slope lines.

rho_ref = rho_ts[PROFILE_INDICES[2], :]
rho_anc = np.nanmedian(rho_ref[np.isfinite(rho_ref)])
if rho_anc > 0:
for slope, ls, lbl in [(-1,”:”,r”r$^{-1}$”),(-2,”–”,r”r$^{-2}$”),(-3,”-..”,r”r$^{-3}$”)]:
ax1.plot(r_plot, rho_anc*(r_plot/10.0)**slope,
color=”#555577”, lw=0.7, ls=ls, alpha=0.5, label=lbl)

ax1.set_xlim(R_BINS[0], R_BINS[-1])
ax1.legend(fontsize=7, ncol=2)

fig1.savefig(os.path.join(OUT_DIR, “section28_rho_profiles.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig1)
print(”  Saved: section28_rho_profiles.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 2 — ρ(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 2]  ρ(r,t) heatmap …”)

rho_log = np.where(rho_ts > 0, np.log10(rho_ts), np.nan)
vmin_rho = np.nanpercentile(rho_log, 5)
vmax_rho = np.nanpercentile(rho_log, 99)

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
gridspec_kw={“width_ratios”:[3,1],“wspace”:0.06})

im2 = ax2a.imshow(rho_log.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“inferno”, vmin=vmin_rho, vmax=vmax_rho)
ax2a.set_yscale(“log”)
*ax(ax2a, xlabel=time_label, ylabel=“r [kpc]”,
title=r”$\log*{10},\rho(r,,t)$  [M$*\odot$ kpc$^{-3}$]”)
cb2 = fig2.colorbar(im2, ax=ax2a, pad=0.01)
cb2.set_label(r”$\log*{10},\rho$”, fontsize=8)

rho_mean_profile = np.nanmean(rho_ts, axis=0)
valid_rm = np.isfinite(rho_mean_profile) & (rho_mean_profile > 0)
_ax(ax2b, xlabel=r”$\langle\rho\rangle_t$”, title=“Time avg.”, log_x=True)
ax2b.plot(rho_mean_profile[valid_rm], r_mid_sph[valid_rm],
color=”#ff9944”, lw=2.0)
ax2b.set_yscale(“log”)
ax2b.set_ylim(R_BINS[0], R_BINS[-1])
ax2b.tick_params(labelleft=False)

fig2.savefig(os.path.join(OUT_DIR, “section28_rho_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig2)
print(”  Saved: section28_rho_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 3 — ρ_0(t), r_s(t), AND NFW χ²(t)

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 3]  Central density and NFW parameters …”)

fig3, axes3 = plt.subplots(3, 1, figsize=(11, 10), facecolor=BG,
sharex=True, gridspec_kw={“hspace”: 0.08})

*ax(axes3[0], ylabel=r”$\rho_0$ [M$*\odot$ kpc$^{-3}$]”,
title=“NFW Parameter Evolution”, log_y=True)
axes3[0].plot(time_arr, rho0_arr, color=”#ff9944”, lw=1.8,
label=r”$\rho_0$ (innermost bin)”)
axes3[0].legend(fontsize=8)

_ax(axes3[1], ylabel=r”NFW $r_s$ [kpc]”,
title=””)
valid_rs = np.isfinite(nfw_rs_arr)
axes3[1].plot(time_arr[valid_rs], nfw_rs_arr[valid_rs],
color=”#4a8fff”, lw=1.8, label=r”NFW $r_s$”)
axes3[1].legend(fontsize=8)

_ax(axes3[2], xlabel=time_label,
ylabel=r”NFW $\chi^2$”, title=””, log_y=True)
valid_chi = np.isfinite(nfw_chi2_arr)
axes3[2].plot(time_arr[valid_chi], nfw_chi2_arr[valid_chi],
color=”#e8673a”, lw=1.5, label=r”Reduced $\chi^2$”)
axes3[2].axhline(1.0, color=”#ffffff”, lw=0.7, ls=”–”, alpha=0.4,
label=“Good fit”)
axes3[2].legend(fontsize=8)

fig3.savefig(os.path.join(OUT_DIR, “section28_rho_central.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig3)
print(”  Saved: section28_rho_central.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 4 — PHASE-SPACE DENSITY f(r) = ρ/σ_r³ AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# f(r) = ρ/σ_r³ is Lynden-Bell’s coarse-grained phase-space density.

# Under violent relaxation it is *diluted* — regions of high phase-space

# density are mixed with lower-density surroundings.  The central peak in

# f(r) should shrink after each pericentre passage, documenting the

# irreversible mixing of phase space during the merger.

print(”[Fig 4]  Phase-space density profiles …”)

fig4, ax4 = plt.subplots(figsize=(9, 6), facecolor=BG)
*ax(ax4, xlabel=“r [kpc]”,
ylabel=r”$f = \rho / \sigma_r^3$  [M$*\odot$ kpc$^{-3}$ (km/s)$^{-3}$]”,
title=r”Coarse-Grained Phase-Space Density  $f(r) = \rho/\sigma_r^3$”,
log_x=True, log_y=True)

for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
y     = psd_ts[k_idx, :]
valid = np.isfinite(y) & (y > 0)
if valid.any():
ax4.plot(r_mid_sph[valid], y[valid], color=color, lw=2.0, label=label)

ax4.set_xlim(R_BINS[0], R_BINS[-1])
ax4.legend(fontsize=8)

fig4.savefig(os.path.join(OUT_DIR, “section28_phase_space_density.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig4)
print(”  Saved: section28_phase_space_density.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 5 — PHASE-SPACE DENSITY f(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 5]  Phase-space density heatmap …”)

psd_log = np.where(psd_ts > 0, np.log10(psd_ts), np.nan)

fig5, ax5 = plt.subplots(figsize=(12, 5), facecolor=BG)
im5 = ax5.imshow(psd_log.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“plasma”)
ax5.set_yscale(“log”)
*ax(ax5, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Phase-Space Density  $\log*{10}(\rho/\sigma_r^3)$  [M$*\odot$ kpc$^{-3}$ (km/s)$^{-3}$]”)
cb5 = fig5.colorbar(im5, ax=ax5, pad=0.01)
cb5.set_label(r”$\log*{10} f$”, fontsize=8)

fig5.savefig(os.path.join(OUT_DIR, “section28_psd_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig5)
print(”  Saved: section28_psd_heatmap.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 6 — OVERDENSITY δ(r, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 6]  Overdensity δ(r,t) heatmap …”)

delta_clipped = np.clip(delta_ts, -1.0, 1e4)
delta_log     = np.where(delta_clipped > 0, np.log10(delta_clipped + 1), np.nan)

fig6, ax6 = plt.subplots(figsize=(12, 5), facecolor=BG)
im6 = ax6.imshow(delta_log.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“hot”)
ax6.set_yscale(“log”)
*ax(ax6, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Density Contrast  $\log*{10}(\delta + 1)$  where  $\delta = \rho/\bar\rho - 1$”)
cb6 = fig6.colorbar(im6, ax=ax6, pad=0.01)
cb6.set_label(r”$\log_{10}(\delta+1)$”, fontsize=8)

fig6.savefig(os.path.join(OUT_DIR, “section28_overdensity.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig6)
print(”  Saved: section28_overdensity.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 7 — 2D MIDPLANE DENSITY SLICE AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 7]  2D density slice grid …”)

fig7, axes7 = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG,
sharey=True, gridspec_kw={“wspace”:0.04})

# Pick five grid snapshots closest to PROFILE_INDICES times.

grid_profile_ii = [
np.argmin(np.abs(time_grid - time_arr[k]))
for k in PROFILE_INDICES
]

for col, (gi, label) in enumerate(zip(grid_profile_ii, PROFILE_LABELS)):
ax = axes7[col]
ax.set_facecolor(BG)
sl = slices_2d[gi]
sl_smooth = gaussian_filter(np.where(sl>0, sl, 0.0), sigma=1.5)
sl_log    = np.where(sl_smooth>0, np.log10(sl_smooth), np.nan)

```
vals = sl_log[np.isfinite(sl_log)]
vmin = np.percentile(vals, 5)  if vals.size > 0 else 0
vmax = np.percentile(vals, 99) if vals.size > 0 else 10

ax.imshow(sl_log.T, origin="lower", aspect="equal",
          extent=[-GRID_EXTENT, GRID_EXTENT, -GRID_EXTENT, GRID_EXTENT],
          cmap="inferno", vmin=vmin, vmax=vmax)
ax.set_title(label, fontsize=9, color="#c8c8e8")
ax.tick_params(colors="#9090b0", labelsize=7)
ax.set_xlabel("x [kpc]", fontsize=8, color="#c8c8e8")
if col == 0:
    ax.set_ylabel("y [kpc]", fontsize=8, color="#c8c8e8")
```

fig7.suptitle(r”DM Density Midplane Slice  $\rho(x,y,z=0)$  [log$*{10}$ M$*\odot$ kpc$^{-3}$]”,
fontsize=11)
fig7.savefig(os.path.join(OUT_DIR, “section28_3d_slice.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig7)
print(”  Saved: section28_3d_slice.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 8 — DENSITY POWER SPECTRUM P(k) AT FIVE EPOCHS

# ══════════════════════════════════════════════════════════════════════════════

# 

# P(k) shows how power is distributed across spatial scales.

# Large k (small scales) — power from substructure and halo cores.

# Small k (large scales) — power from the bulk mass distribution.

# At pericentre, large-scale power increases as the two halo centres

# approach; small-scale power is partially suppressed as substructure

# is tidally disrupted.

print(”[Fig 8]  Power spectra …”)

fig8, ax8 = plt.subplots(figsize=(9, 6), facecolor=BG)
_ax(ax8, xlabel=r”$k$  [kpc$^{-1}$]”,
ylabel=r”$P(k)$  [dimensionless]”,
title=r”DM Density Power Spectrum $P(k)$ at Key Epochs”,
log_x=True, log_y=True)

for ii, color, label in zip(grid_profile_ii, PROFILE_COLORS, PROFILE_LABELS):
k_c = k_centres_arr[ii]
P_k = Pk_ts[ii]
valid = np.isfinite(k_c) & np.isfinite(P_k) & (k_c > 0) & (P_k > 0)
if valid.any():
ax8.plot(k_c[valid], P_k[valid], color=color, lw=2.0, label=label)

ax8.legend(fontsize=8)

fig8.savefig(os.path.join(OUT_DIR, “section28_power_spectrum.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig8)
print(”  Saved: section28_power_spectrum.png”)

# ══════════════════════════════════════════════════════════════════════════════

# FIGURE 9 — P(k, t) HEATMAP

# ══════════════════════════════════════════════════════════════════════════════

print(”[Fig 9]  P(k,t) heatmap …”)

Pk_log = np.where(Pk_ts > 0, np.log10(Pk_ts), np.nan)
k_mean = np.nanmean(k_centres_arr, axis=0)

t_grid_min = np.nanmin(time_grid)
t_grid_max = np.nanmax(time_grid)

fig9, ax9 = plt.subplots(figsize=(12, 5), facecolor=BG)
im9 = ax9.imshow(Pk_log.T, aspect=“auto”, origin=“lower”,
extent=[t_grid_min, t_grid_max, k_mean[0], k_mean[-1]],
cmap=“viridis”)
ax9.set_yscale(“log”)
*ax(ax9, xlabel=time_label, ylabel=r”$k$ [kpc$^{-1}$]”,
title=r”Density Power Spectrum Evolution  $\log*{10},P(k,,t)$”)
cb9 = fig9.colorbar(im9, ax=ax9, pad=0.01)
cb9.set_label(r”$\log_{10},P(k)$”, fontsize=8)

fig9.savefig(os.path.join(OUT_DIR, “section28_pk_heatmap.png”),
dpi=300, bbox_inches=“tight”, facecolor=BG)
plt.close(fig9)
print(”  Saved: section28_pk_heatmap.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.5 — ANIMATIONS                                                        ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Animation 1: ρ(r) profile with ghost history ──────────────────────────────

print(”\n[Anim 1]  ρ(r) profile animation …”)

anim_idx_arr = np.arange(0, ns, ANIM_STEP_28)
n_frames_rho = len(anim_idx_arr)
N_GHOST      = 12
cmap_t       = plt.cm.plasma

fig_a1, ax_a1 = plt.subplots(figsize=(8, 6), facecolor=BG)
ax_a1.set_facecolor(BG)
ax_a1.set_xscale(“log”); ax_a1.set_yscale(“log”)
ax_a1.set_xlim(R_BINS[0], R_BINS[-1])

rho_finite = rho_ts[np.isfinite(rho_ts) & (rho_ts > 0)]
ax_a1.set_ylim(rho_finite.min()*0.3 if rho_finite.size>0 else 1e2,
rho_finite.max()*3.0  if rho_finite.size>0 else 1e12)
ax_a1.set_xlabel(“r [kpc]”, color=”#c8c8e8”)
ax_a1.set_ylabel(r”$\rho(r)$ [M$_\odot$ kpc$^{-3}$]”, color=”#c8c8e8”)

ghosts   = [ax_a1.plot([], [], lw=0.7)[0] for _ in range(N_GHOST)]
main_rho, = ax_a1.plot([], [], lw=2.2, color=“white”, zorder=5)
title_a1  = fig_a1.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_rho_anim(frame_idx):
snap_i = anim_idx_arr[frame_idx]
color  = cmap_t(frame_idx / n_frames_rho)

```
def _xy(arr):
    v = np.isfinite(arr) & (arr > 0)
    return r_mid_sph[v], arr[v]

main_rho.set_data(*_xy(rho_ts[snap_i, :]))
main_rho.set_color(color)

for g, ghost in enumerate(ghosts):
    past = frame_idx - (N_GHOST - g)
    if past < 0:
        ghost.set_data([], [])
        continue
    gc = cmap_t(past / n_frames_rho)
    ghost.set_data(*_xy(rho_ts[anim_idx_arr[past], :]))
    ghost.set_color(gc)
    ghost.set_alpha(0.05 + 0.06 * g)

t_val = time_arr[snap_i]
t_str = f"{t_val:.2f} Gyr" if (np.isfinite(t_val) and time_is_gyr) \
        else f"Snap {SNAPSHOTS[snap_i]}"
title_a1.set_text(fr"DM Density Profile  ·  {t_str}")
return [main_rho] + ghosts
```

ani_rho = animation.FuncAnimation(
fig_a1, _update_rho_anim, frames=n_frames_rho,
interval=1000//ANIM_FPS_28, blit=True,
)
writer_28 = animation.FFMpegWriter(
fps=ANIM_FPS_28, bitrate=ANIM_BITRATE_28,
metadata=dict(title=“MW-M31 DM Density Profile Animation”),
)
ani_rho.save(os.path.join(OUT_DIR, “section28_animation_rho.mp4”),
writer=writer_28, dpi=ANIM_DPI_28)
plt.close(fig_a1)
print(”  Saved: section28_animation_rho.mp4”)

# ── Animation 2: 2D density slice ────────────────────────────────────────────

print(”[Anim 2]  2D density slice animation …”)

all_vals = anim_slices[anim_slices > 0]
vmin_sl  = np.log10(np.percentile(all_vals, 5))  if all_vals.size>0 else 0
vmax_sl  = np.log10(np.percentile(all_vals, 99)) if all_vals.size>0 else 10

fig_a2, ax_a2 = plt.subplots(figsize=(7, 7), facecolor=BG)
ax_a2.set_facecolor(BG)

sl0     = anim_slices[0]
sl0_s   = gaussian_filter(np.where(sl0>0, sl0, 0.0), sigma=1.5)
sl0_log = np.where(sl0_s>0, np.log10(sl0_s), np.nan)

im_sl = ax_a2.imshow(sl0_log.T, origin=“lower”, aspect=“equal”,
extent=[-GRID_EXTENT, GRID_EXTENT,
-GRID_EXTENT, GRID_EXTENT],
cmap=“inferno”, vmin=vmin_sl, vmax=vmax_sl)
ax_a2.set_xlabel(“x [kpc]”, color=”#c8c8e8”)
ax_a2.set_ylabel(“y [kpc]”, color=”#c8c8e8”)
title_sl = fig_a2.suptitle(””, fontsize=11, color=”#c8c8e8”)

def _update_slice_anim(frame_idx):
snap_num = anim_snap_nums[frame_idx]
ai       = anim_snap_map[snap_num]
sl       = anim_slices[ai]
sl_s     = gaussian_filter(np.where(sl>0, sl, 0.0), sigma=1.5)
sl_log   = np.where(sl_s>0, np.log10(sl_s), np.nan)
im_sl.set_data(sl_log.T)

```
snap_global_i = np.where(SNAPSHOTS == snap_num)[0]
t_val = time_arr[snap_global_i[0]] if len(snap_global_i)>0 else np.nan
t_str = f"{t_val:.2f} Gyr" if (np.isfinite(t_val) and time_is_gyr) \
        else f"Snap {snap_num}"
title_sl.set_text(fr"DM Density Slice  $\rho(x,y,z=0)$  ·  {t_str}")
return [im_sl]
```

ani_sl = animation.FuncAnimation(
fig_a2, _update_slice_anim, frames=n_anim_frames,
interval=1000//ANIM_FPS_28, blit=True,
)
ani_sl.save(os.path.join(OUT_DIR, “section28_animation_slice.mp4”),
writer=writer_28, dpi=ANIM_DPI_28)
plt.close(fig_a2)
print(”  Saved: section28_animation_slice.mp4”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.6 — MASTER SUMMARY PANEL                                              ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n[Summary]  Master summary panel …”)

fig_s = plt.figure(figsize=(16, 10), facecolor=BG)
gs_s  = gridspec.GridSpec(2, 2, figure=fig_s,
hspace=0.38, wspace=0.32,
left=0.08, right=0.97,
top=0.93, bottom=0.07)

# (0,0) ρ(r,t) heatmap.

ax_s00 = fig_s.add_subplot(gs_s[0, 0])
im_s00 = ax_s00.imshow(rho_log.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“inferno”, vmin=vmin_rho, vmax=vmax_rho)
ax_s00.set_yscale(“log”)
*ax(ax_s00, xlabel=time_label, ylabel=“r [kpc]”,
title=r”$\log*{10},\rho(r,t)$”)
fig_s.colorbar(im_s00, ax=ax_s00, shrink=0.8)

# (0,1) ρ_0(t) and r_s(t).

ax_s01 = fig_s.add_subplot(gs_s[0, 1])
ax_s01.set_facecolor(BG)
ax_s01.semilogy(time_arr, rho0_arr, color=”#ff9944”, lw=1.8, label=r”$\rho_0$”)
ax_s01.set_xlabel(time_label, fontsize=8, color=”#c8c8e8”)
ax_s01.set_ylabel(r”$\rho_0$ [M$_\odot$ kpc$^{-3}$]”, fontsize=8, color=”#c8c8e8”)
ax_s01.set_title(“Central Density”, fontsize=9, color=”#c8c8e8”)
ax_s01r = ax_s01.twinx()
ax_s01r.set_facecolor(BG)
ax_s01r.plot(time_arr[valid_rs], nfw_rs_arr[valid_rs],
color=”#4a8fff”, lw=1.5, ls=”–”)
ax_s01r.set_ylabel(r”NFW $r_s$ [kpc]”, fontsize=8, color=”#4a8fff”)
ax_s01r.tick_params(colors=”#4a8fff”)

# (1,0) Phase-space density heatmap.

ax_s10 = fig_s.add_subplot(gs_s[1, 0])
im_s10 = ax_s10.imshow(psd_log.T, aspect=“auto”, origin=“lower”,
extent=[t_min, t_max, R_BINS[0], R_BINS[-1]],
cmap=“plasma”)
ax_s10.set_yscale(“log”)
*ax(ax_s10, xlabel=time_label, ylabel=“r [kpc]”,
title=r”Phase-Space Density  $\log*{10}(\rho/\sigma_r^3)$”)
fig_s.colorbar(im_s10, ax=ax_s10, shrink=0.8)

# (1,1) Power spectrum.

ax_s11 = fig_s.add_subplot(gs_s[1, 1])
_ax(ax_s11, xlabel=r”$k$ [kpc$^{-1}$]”, ylabel=r”$P(k)$”,
title=“Density Power Spectrum”, log_x=True, log_y=True)
for ii, color in zip(grid_profile_ii, PROFILE_COLORS):
k_c = k_centres_arr[ii]
P_k = Pk_ts[ii]
valid = np.isfinite(k_c)&np.isfinite(P_k)&(k_c>0)&(P_k>0)
if valid.any():
ax_s11.plot(k_c[valid], P_k[valid], color=color, lw=1.5)

fig_s.suptitle(“Section 28 Summary  ·  Dark Matter Density Evolution”,
fontsize=13, color=”#c8c8e8”, fontweight=“bold”)
fig_s.savefig(os.path.join(OUT_DIR, “section28_summary_panel.png”),
dpi=200, bbox_inches=“tight”, facecolor=BG)
plt.close(fig_s)
print(”  Saved: section28_summary_panel.png”)

# ╔══════════════════════════════════════════════════════════════════════════════╗

# ║  §28.7 — SECTION COMPLETE                                                  ║

# ╚══════════════════════════════════════════════════════════════════════════════╝

print(”\n” + “=”*80)
print(”  SECTION 28 COMPLETE”)
print(”=”*80)
outputs_28 = [
“section28_rho_profiles.png”,
“section28_rho_heatmap.png”,
“section28_rho_central.png”,
“section28_phase_space_density.png”,
“section28_psd_heatmap.png”,
“section28_overdensity.png”,
“section28_3d_slice.png”,
“section28_power_spectrum.png”,
“section28_pk_heatmap.png”,
“section28_animation_rho.mp4”,
“section28_animation_slice.mp4”,
“section28_summary_panel.png”,
]
for fn in outputs_28:
fp   = os.path.join(OUT_DIR, fn)
size = os.path.getsize(fp)/1e6 if os.path.isfile(fp) else 0.0
kind = “animation” if fn.endswith(”.mp4”) else “figure”
print(f”  {fn:<48} {size:6.2f} MB  [{kind}]”)
print(”=”*80)
