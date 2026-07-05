"""
===============================================================================
SECTION 36 — MULTIPOLE EXPANSION & MINKOWSKI FUNCTIONAL MORPHOLOGY
===============================================================================
Author  : Abhinav Vatsa

This section computes two complementary angular/topological morphology
diagnostics for the dark matter halo at each snapshot epoch:

  METHOD A  — Spherical harmonic multipole expansion of the particle
              angular distribution per radial shell.
              Key outputs: Ẽ_l(r,t), Q_bar, D_bar, phase_22, dPA_22_io.

  METHOD B  — Minkowski functionals of the smoothed 3D density excursion
              sets at a grid of density thresholds.
              Key outputs: W0–W3, P_MF, F_MF, Euler characteristic χ,
              shapefinders T/W/L, morphology-plane curves.

Cross-section correlations close the §31–36 six-method diagnostic suite.
===============================================================================
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import os, time, warnings

# ── numerics ─────────────────────────────────────────────────────────────────
import numpy as np
from scipy.special    import sph_harm_y          # scipy ≥ 1.15 API
from scipy.ndimage    import gaussian_filter, label as ndimage_label
from scipy.stats      import pearsonr
from scipy.interpolate import interp1d

# ── plotting ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

# ── optional: skimage for Euler characteristic ───────────────────────────────
try:
    from skimage.measure import euler_number, label as sk_label
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False
    warnings.warn("scikit-image not found — W3 / Euler characteristic will "
                  "use a connected-component approximation.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.0  CONFIGURATION                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── multipole ────────────────────────────────────────────────────────────────
L_MAX               = 8      # maximum spherical harmonic degree
N_MULTIPOLE_SHELLS  = 40     # radial shells for the multipole profile
N_MULTIPOLE_SNAPS   = 60     # snapshot epochs analysed by multipole
MIN_MULTIPOLE_PART  = 20     # minimum particles per shell

# ── Minkowski ────────────────────────────────────────────────────────────────
GRID_RES            = 64     # voxels per side of the CIC density cube
SMOOTH_SIGMA        = 2.5    # Gaussian smoothing [voxels]
N_THRESHOLDS        = 30     # density thresholds swept per snapshot
N_MF_SNAPS          = 25     # snapshot epochs for Minkowski analysis
MIN_VOXELS_SURFACE  = 10     # minimum surface voxels for a valid MF result
THRESH_PERCENTILE_LO= 5      # lowest density percentile for threshold scan
THRESH_PERCENTILE_HI= 95     # highest density percentile

# ── geometry ─────────────────────────────────────────────────────────────────
R_INNER             = 1.0    # [kpc] — overridden dynamically in §36.1
R_OUTER             = 120.0  # [kpc] — overridden dynamically in §36.1

# ── animation ────────────────────────────────────────────────────────────────
ANIM_FPS_36         = 18
ANIM_DPI_36         = 100
ANIM_BITRATE_36     = 1600

# ── output directory ─────────────────────────────────────────────────────────
OUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("  SECTION 36 · Multipole Expansion & Minkowski Functional Morphology")
print("="*80)
print(f"  L_MAX              : {L_MAX}")
print(f"  Multipole shells   : {N_MULTIPOLE_SHELLS}")
print(f"  Multipole snaps    : {N_MULTIPOLE_SNAPS}")
print(f"  Grid resolution    : {GRID_RES}³  ({GRID_RES**3:,} voxels)")
print(f"  Smoothing sigma    : {SMOOTH_SIGMA} voxels")
print(f"  MF thresholds      : {N_THRESHOLDS}")
print(f"  MF snaps           : {N_MF_SNAPS}")
print(f"  skimage available  : {_HAVE_SKIMAGE}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.1  SYNTHETIC DATA  (replace with inherited pipeline globals)          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# In the full pipeline, _traj_pos and _traj_r are inherited from Section 26.
# Here we synthesise a plausible MW–M31 merger trajectory for standalone runs.

np.random.seed(42)
_NS   = 80                        # total snapshots
_N    = 1200                      # total particles
_DT   = 10.0                      # [Myr] per snapshot

# Build a synthetic evolving prolate halo:
#   · starts oblate (disc-like) at t=0
#   · becomes strongly prolate at t≈35 snaps (pericentric passage)
#   · relaxes toward triaxial at late times
_traj_pos = np.zeros((_NS, _N, 3))
_r0_base  = np.abs(np.random.randn(_N)) * 20 + 2   # initial radii [kpc]
theta0    = np.random.uniform(0, np.pi,   _N)
phi0      = np.random.uniform(0, 2*np.pi, _N)

for s in range(_NS):
    t_frac      = s / (_NS - 1)
    # axis ratios: oblate→prolate→triaxial
    ax          = 1.0 + 1.2 * np.exp(-0.5*((t_frac-0.45)/0.12)**2)   # major
    ay          = 1.0 + 0.3 * np.exp(-0.5*((t_frac-0.45)/0.20)**2)   # intermediate
    az          = 1.0 - 0.4 * t_frac                                   # minor shrinks
    az          = max(az, 0.55)
    r_evol      = _r0_base * (1.0 + 0.15*np.sin(2*np.pi*t_frac))
    x           = r_evol * np.sin(theta0) * np.cos(phi0) * ax
    y           = r_evol * np.sin(theta0) * np.sin(phi0) * ay
    z           = r_evol * np.cos(theta0)                 * az
    noise       = np.random.randn(_N, 3) * 0.5
    _traj_pos[s] = np.stack([x, y, z], axis=1) + noise

_traj_r = np.linalg.norm(_traj_pos, axis=2)  # (ns, N)
_r0     = _traj_r[0]
_group  = np.zeros(_N, dtype=int)             # group labels (all MW for synth)
_group[_N*3//4:] = 3                          # last quarter = M31

# time axis [Gyr]
time_arr = np.arange(_NS) * _DT / 1000.0     # [Gyr]
ns       = _NS

# Dynamic radial limits
R_INNER = max(0.5, float(np.percentile(_r0, 2)))
R_OUTER = float(np.percentile(_r0, 97))

snap_indices_mult = np.linspace(0, ns-1, N_MULTIPOLE_SNAPS, dtype=int)
snap_indices_mf   = np.linspace(0, ns-1, N_MF_SNAPS,        dtype=int)
time_mult         = time_arr[snap_indices_mult]
time_mf           = time_arr[snap_indices_mf]

print(f"\n  R_INNER = {R_INNER:.1f} kpc    R_OUTER = {R_OUTER:.1f} kpc")
print(f"  Particles: {_N}    Snapshots: {ns}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.2  SPHERICAL HARMONIC COEFFICIENT ESTIMATOR                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_alm(theta_shell, phi_shell, l_max=L_MAX):
    """
    Compute spherical harmonic coefficients a_lm and multipole power P_l
    for a set of particle directions.

    Parameters
    ----------
    theta_shell : (N,) colatitude [rad] in [0, π]
    phi_shell   : (N,) azimuth   [rad] in [−π, π]
    l_max       : int  maximum degree

    Returns
    -------
    alm      : dict {(l,m): complex}
    P_l      : (l_max+1,) raw power Σ_m |a_lm|²
    Etilde_l : (l_max+1,) normalised power P_l / P_0
    """
    N = len(theta_shell)
    if N < MIN_MULTIPOLE_PART:
        nan_a = np.full(l_max + 1, np.nan)
        return {}, nan_a, nan_a

    alm = {}
    P_l = np.zeros(l_max + 1, dtype=float)

    for l in range(l_max + 1):
        # m = 0
        Y_l0        = sph_harm_y(l, 0, theta_shell, phi_shell)
        a_l0        = np.mean(np.conj(Y_l0))
        alm[(l, 0)] = a_l0
        P_l[l]     += float(abs(a_l0)**2)
        # m ≥ 1: exploit |a_l{-m}|² = |a_lm|²
        for m in range(1, l + 1):
            Y_lm        = sph_harm_y(l, m, theta_shell, phi_shell)
            a_lm        = np.mean(np.conj(Y_lm))
            alm[(l,  m)] = a_lm
            alm[(l, -m)] = ((-1)**m) * np.conj(a_lm)
            P_l[l]      += 2.0 * float(abs(a_lm)**2)

    P0       = P_l[0] if P_l[0] > 1e-30 else 1.0
    Etilde_l = P_l / P0
    return alm, P_l, Etilde_l


def multipole_phase(alm, l, m):
    """Return phase of a_lm in degrees, or NaN if absent."""
    key = (l, m)
    if key not in alm or not np.isfinite(alm[key]):
        return np.nan
    return float(np.degrees(np.angle(alm[key])))

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.3  PRECOMPUTE SPHERICAL COORDINATES                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n  Precomputing spherical coordinates …", end="", flush=True)
theta_arr = np.full((N_MULTIPOLE_SNAPS, _N), np.nan)
phi_arr   = np.full((N_MULTIPOLE_SNAPS, _N), np.nan)
com_arr   = np.zeros((N_MULTIPOLE_SNAPS, 3))

for k, s in enumerate(snap_indices_mult):
    com           = _traj_pos[s].mean(axis=0)
    com_arr[k]    = com
    pos_c         = _traj_pos[s] - com
    r_c           = np.linalg.norm(pos_c, axis=1)
    r_safe        = np.maximum(r_c, 1e-10)
    theta_arr[k]  = np.arccos(np.clip(pos_c[:, 2] / r_safe, -1.0, 1.0))
    phi_arr[k]    = np.arctan2(pos_c[:, 1], pos_c[:, 0])
print(" done.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.4  MULTIPOLE RADIAL PROFILES                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

r_edges_mult = np.logspace(np.log10(R_INNER), np.log10(R_OUTER),
                            N_MULTIPOLE_SHELLS + 1)
r_mid_mult   = np.sqrt(r_edges_mult[:-1] * r_edges_mult[1:])   # geometric mean

Etilde_arr  = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS, L_MAX + 1), np.nan)
P_l_arr     = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS, L_MAX + 1), np.nan)
Q_bar_arr   = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # Ẽ_2
D_bar_arr   = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # Ẽ_1
H_bar_arr   = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)  # Ẽ_4
phase22_arr = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)
phase21_arr = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)
l_peak_arr  = np.full((N_MULTIPOLE_SNAPS, N_MULTIPOLE_SHELLS), np.nan)

# Global (all-particle) multipole coefficients
Etilde_global  = np.full((N_MULTIPOLE_SNAPS, L_MAX + 1), np.nan)
phase22_global = np.full(N_MULTIPOLE_SNAPS, np.nan)
dPA_22_io      = np.full(N_MULTIPOLE_SNAPS, np.nan)  # inner-outer twist angle

print("  Computing multipole profiles …")
t0_mult = time.time()

for k, s in enumerate(snap_indices_mult):
    r_now = _traj_r[s]

    # --- global spectrum (all particles) ---
    alm_g, _, Et_g      = compute_alm(theta_arr[k], phi_arr[k])
    Etilde_global[k]    = Et_g
    phase22_global[k]   = multipole_phase(alm_g, 2, 2)

    # --- per-shell profiles ---
    for b in range(N_MULTIPOLE_SHELLS):
        mask = (r_now >= r_edges_mult[b]) & (r_now < r_edges_mult[b + 1])
        if mask.sum() < MIN_MULTIPOLE_PART:
            continue
        alm_b, P_l_b, Et_b = compute_alm(theta_arr[k][mask], phi_arr[k][mask])
        Etilde_arr[k, b]    = Et_b
        P_l_arr[k, b]       = P_l_b
        Q_bar_arr[k, b]     = Et_b[2] if len(Et_b) > 2 else np.nan
        D_bar_arr[k, b]     = Et_b[1] if len(Et_b) > 1 else np.nan
        H_bar_arr[k, b]     = Et_b[4] if len(Et_b) > 4 else np.nan
        phase22_arr[k, b]   = multipole_phase(alm_b, 2, 2)
        phase21_arr[k, b]   = multipole_phase(alm_b, 2, 1)
        valid_l             = Et_b[1:]    # exclude monopole
        if np.any(np.isfinite(valid_l)):
            l_peak_arr[k, b] = float(np.nanargmax(valid_l) + 1)

    # --- inner-outer quadrupole misalignment ---
    phi_inner = next((phase22_arr[k, b] for b in range(N_MULTIPOLE_SHELLS)
                      if np.isfinite(phase22_arr[k, b])), np.nan)
    phi_outer = next((phase22_arr[k, b] for b in range(N_MULTIPOLE_SHELLS - 1, -1, -1)
                      if np.isfinite(phase22_arr[k, b])), np.nan)
    if np.isfinite(phi_inner) and np.isfinite(phi_outer):
        raw = abs(phi_inner - phi_outer)
        dPA_22_io[k] = min(raw, 180.0 - raw) / 2.0

    if (k + 1) % 10 == 0:
        print(f"    snap {k+1}/{N_MULTIPOLE_SNAPS}  "
              f"  Ẽ_2_global={Etilde_global[k,2]:.3f}", flush=True)

print(f"  Multipole done in {time.time()-t0_mult:.1f} s")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.5  3D DENSITY GRID (CIC)                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_density_grid(pos_com, grid_res=GRID_RES, r_outer=R_OUTER,
                        smooth_sigma=SMOOTH_SIGMA):
    """
    Cloud-In-Cell density grid on a cubic domain [−r_outer, r_outer]³.

    Returns
    -------
    rho_grid   : (G,G,G)  overdensity δ = ρ/ρ̄ − 1  (Gaussian-smoothed)
    voxel_size : float    [kpc / voxel]
    x_edges    : (G+1,)   grid edge positions [kpc]
    """
    G          = grid_res
    voxel_size = 2.0 * r_outer / G
    x_edges    = np.linspace(-r_outer, r_outer, G + 1)
    rho_grid   = np.zeros((G, G, G), dtype=np.float64)

    r_all  = np.linalg.norm(pos_com, axis=1)
    pos_in = pos_com[r_all < r_outer]
    if len(pos_in) == 0:
        return rho_grid, voxel_size, x_edges

    pn = (pos_in + r_outer) / voxel_size
    ix = np.clip(np.floor(pn[:, 0]).astype(int), 0, G - 2)
    iy = np.clip(np.floor(pn[:, 1]).astype(int), 0, G - 2)
    iz = np.clip(np.floor(pn[:, 2]).astype(int), 0, G - 2)
    dx = pn[:, 0] - ix
    dy = pn[:, 1] - iy
    dz = pn[:, 2] - iz

    np.add.at(rho_grid, (ix,   iy,   iz  ), (1-dx)*(1-dy)*(1-dz))
    np.add.at(rho_grid, (ix+1, iy,   iz  ), dx    *(1-dy)*(1-dz))
    np.add.at(rho_grid, (ix,   iy+1, iz  ), (1-dx)*dy    *(1-dz))
    np.add.at(rho_grid, (ix,   iy,   iz+1), (1-dx)*(1-dy)*dz    )
    np.add.at(rho_grid, (ix+1, iy+1, iz  ), dx    *dy    *(1-dz))
    np.add.at(rho_grid, (ix+1, iy,   iz+1), dx    *(1-dy)*dz    )
    np.add.at(rho_grid, (ix,   iy+1, iz+1), (1-dx)*dy    *dz    )
    np.add.at(rho_grid, (ix+1, iy+1, iz+1), dx    *dy    *dz    )

    rho_mean = rho_grid.mean()
    rho_grid = rho_grid / (rho_mean + 1e-30) - 1.0
    if smooth_sigma > 0:
        rho_grid = gaussian_filter(rho_grid, sigma=smooth_sigma)
    return rho_grid, voxel_size, x_edges

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.6  MINKOWSKI FUNCTIONAL COMPUTATION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_minkowski_functionals(rho_grid, rho_threshold, voxel_size):
    """
    Compute the four Minkowski functionals of the excursion set
    { x : rho_grid(x) > rho_threshold }.

    Returns a dict with keys:
        W0, W1, W2, W3         — four MF values
        chi                    — Euler characteristic (integer)
        T_MF, W_MF, L_MF      — shapefinders [kpc]
        P_MF, F_MF             — planarity, filamentarity ∈ [0,1]
        n_components           — number of connected components
        valid                  — bool
    """
    dv  = voxel_size
    dv3 = dv ** 3
    dv2 = dv ** 2

    nan_out = {k: np.nan for k in ['W0','W1','W2','W3','chi',
                                    'T_MF','W_MF','L_MF','P_MF','F_MF',
                                    'n_components']}
    nan_out['valid'] = False

    B   = (rho_grid > rho_threshold)
    N_in = int(B.sum())
    if N_in < MIN_VOXELS_SURFACE:
        return nan_out

    # W0 — volume
    W0 = N_in * dv3

    # W1 — surface area / 6  (exposed-face count)
    pad     = np.pad(B, 1, mode='constant', constant_values=False)
    faces   = (  (B & ~pad[2:,  1:-1, 1:-1]).sum()
               + (B & ~pad[:-2, 1:-1, 1:-1]).sum()
               + (B & ~pad[1:-1, 2:,  1:-1]).sum()
               + (B & ~pad[1:-1, :-2, 1:-1]).sum()
               + (B & ~pad[1:-1, 1:-1, 2: ]).sum()
               + (B & ~pad[1:-1, 1:-1, :-2]).sum())
    A_surf  = float(faces) * dv2
    W1      = A_surf / 6.0

    # W3 — Euler characteristic
    chi    = np.nan
    n_comp = 0
    if _HAVE_SKIMAGE:
        chi    = int(euler_number(B, connectivity=3))
        _, n_comp = ndimage_label(B)
    else:
        _, n_comp = ndimage_label(B)
        # Fallback: χ = 2*(1 − g) ≈ 2 for a single blob
        chi = 2 * n_comp   # rough estimate: treats each blob as genus-0
    W3 = float(chi) / (4.0 * np.pi)

    # W2 — mean curvature proxy: W2 = W1 / r_eff  (exact for sphere)
    r_eff = (3.0 * W0 / (4.0 * np.pi)) ** (1.0 / 3.0) if W0 > 0 else 1.0
    W2    = W1 / (r_eff + 1e-30)

    # Shapefinders
    T_MF = 3.0 * W0 / (W1 + 1e-30)
    W_MF = W1 / (2.0 * W2 + 1e-30)
    if np.isfinite(W3) and abs(W3) > 1e-10:
        L_MF = W2 / (3.0 * W3 + 1e-30)
    else:
        L_MF = np.nan

    P_MF = float((W_MF - T_MF) / (W_MF + T_MF + 1e-30))
    P_MF = np.clip(P_MF, 0.0, 1.0)
    F_MF = float((L_MF - W_MF) / (L_MF + W_MF + 1e-30)) if np.isfinite(L_MF) else np.nan
    if np.isfinite(F_MF):
        F_MF = np.clip(F_MF, 0.0, 1.0)

    return {'W0': W0, 'W1': W1, 'W2': W2, 'W3': W3, 'chi': chi,
            'T_MF': T_MF, 'W_MF': W_MF, 'L_MF': L_MF,
            'P_MF': P_MF, 'F_MF': F_MF,
            'n_components': n_comp, 'valid': True}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.7  MF THRESHOLD SCAN                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

W0_arr      = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
W1_arr      = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
W2_arr      = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
W3_arr      = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
chi_arr     = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
P_arr       = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
F_arr       = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
T_mf_arr    = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
n_comp_arr  = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
thresh_grid = np.full((N_MF_SNAPS, N_THRESHOLDS), np.nan)
voxel_sizes = np.full(N_MF_SNAPS, np.nan)

print("\n  Computing Minkowski functionals …")
t0_mf = time.time()

for k, s in enumerate(snap_indices_mf):
    com      = _traj_pos[s].mean(axis=0)
    pos_c    = _traj_pos[s] - com
    rho_g, dv, _ = build_density_grid(pos_c, GRID_RES, R_OUTER, SMOOTH_SIGMA)
    voxel_sizes[k] = dv

    rho_lo   = np.percentile(rho_g, THRESH_PERCENTILE_LO)
    rho_hi   = np.percentile(rho_g, THRESH_PERCENTILE_HI)
    thresholds = np.linspace(rho_lo, rho_hi, N_THRESHOLDS)
    thresh_grid[k] = thresholds

    for j, th in enumerate(thresholds):
        mf = compute_minkowski_functionals(rho_g, th, dv)
        if not mf['valid']:
            continue
        W0_arr[k, j]     = mf['W0']
        W1_arr[k, j]     = mf['W1']
        W2_arr[k, j]     = mf['W2']
        W3_arr[k, j]     = mf['W3']
        chi_arr[k, j]    = mf['chi']
        P_arr[k, j]      = mf['P_MF']
        F_arr[k, j]      = mf['F_MF']
        T_mf_arr[k, j]   = mf['T_MF']
        n_comp_arr[k, j] = mf['n_components']

    if (k + 1) % 5 == 0:
        print(f"    MF snap {k+1}/{N_MF_SNAPS}", flush=True)

print(f"  MF done in {time.time()-t0_mf:.1f} s")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.8  MF SUMMARY DIAGNOSTICS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

mid_j        = N_THRESHOLDS // 2
P_med_ts     = np.nanmean(P_arr[:, mid_j-2:mid_j+2], axis=1)
F_med_ts     = np.nanmean(F_arr[:, mid_j-2:mid_j+2], axis=1)
F_peak_ts    = np.nanmax(F_arr, axis=1)
chi_min_ts   = np.nanmin(chi_arr, axis=1)
n_chi_jumps  = np.full(N_MF_SNAPS, np.nan)
A_PF_ts      = np.full(N_MF_SNAPS, np.nan)

for k in range(N_MF_SNAPS):
    chi_v = chi_arr[k][np.isfinite(chi_arr[k])]
    if len(chi_v) > 1:
        n_chi_jumps[k] = float(np.sum(np.diff(chi_v) != 0))
    P_v = P_arr[k]; F_v = F_arr[k]
    fin  = np.isfinite(P_v) & np.isfinite(F_v)
    if fin.sum() >= 3:
        try:
            A_PF_ts[k] = float(np.abs(np.trapz(F_v[fin], P_v[fin])))
        except Exception:
            pass

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPER: axis formatter                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _ax(ax, xlabel='', ylabel='', title='', log_x=False, log_y=False,
        xlim=None, ylim=None, grid=True):
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title:  ax.set_title(title,  fontsize=9)
    if log_x:  ax.set_xscale('log')
    if log_y:  ax.set_yscale('log')
    if xlim:   ax.set_xlim(xlim)
    if ylim:   ax.set_ylim(ylim)
    if grid:   ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)

EPOCH_COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
EPOCH_IDX    = [0, N_MULTIPOLE_SNAPS//4, N_MULTIPOLE_SNAPS//2,
                3*N_MULTIPOLE_SNAPS//4, N_MULTIPOLE_SNAPS-1]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 1 — Multipole power spectra at 5 epochs                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

mid_shell = N_MULTIPOLE_SHELLS // 2
fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharey=True)
fig.suptitle("§36 Fig 1 — Multipole Power Spectra Ẽ_l (mid shell)",
             fontsize=10, fontweight='bold')
l_arr = np.arange(1, L_MAX + 1)

for idx, (ei, col) in enumerate(zip(EPOCH_IDX, EPOCH_COLORS)):
    ax  = axes[idx]
    t_k = time_mult[ei]
    Et  = Etilde_arr[ei, mid_shell, 1:]    # exclude monopole
    Et  = np.where(np.isfinite(Et), Et, 0.0)
    ax.bar(l_arr, Et, color=col, alpha=0.75, edgecolor='k', linewidth=0.4)
    # theoretical prolate Ẽ_2 proxy
    E2_theory = np.where(l_arr == 2, Et[1] if len(Et) > 1 else 0.0, 0.0)
    ax.step(l_arr, E2_theory, where='mid', color='k', lw=1.2, ls='--',
            label='l=2 only')
    _ax(ax, xlabel='l', ylabel='Ẽ_l' if idx == 0 else '',
        title=f't = {t_k:.2f} Gyr', log_y=False)
    ax.set_xticks(l_arr)
    ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_multipole_spectra.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 1 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 2 — Ẽ_2(r,t) and Ẽ_1(r,t) heatmaps                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("§36 Fig 2 — Multipole Power Heatmaps", fontsize=10, fontweight='bold')

for ax, data, label, cmap in [
    (ax1, Q_bar_arr, 'Quadrupole Ẽ₂', 'plasma'),
    (ax2, D_bar_arr, 'Dipole Ẽ₁',     'inferno'),
]:
    vmax = np.nanpercentile(data, 97) if np.any(np.isfinite(data)) else 1.0
    vmax = max(vmax, 1e-3)
    im = ax.imshow(data.T, origin='lower', aspect='auto',
                   extent=[time_mult[0], time_mult[-1],
                            np.log10(r_mid_mult[0]), np.log10(r_mid_mult[-1])],
                   vmin=0, vmax=vmax, cmap=cmap)
    plt.colorbar(im, ax=ax, label=label, pad=0.02)
    _ax(ax, xlabel='Time [Gyr]', ylabel='log₁₀(r [kpc])', title=label)
    ax.set_yticks(np.log10([2, 5, 10, 30, 80]))
    ax.set_yticklabels(['2','5','10','30','80'])

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_multipole_heatmaps.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 2 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 3 — Full spectrum Ẽ_l(r) at peak distortion epoch         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

peak_k = int(np.nanargmax(np.nanmean(Q_bar_arr, axis=1)))
data3  = Etilde_arr[peak_k, :, 1:].T   # (L_MAX, N_SHELLS)

fig, ax = plt.subplots(figsize=(8, 4))
vmax3   = np.nanpercentile(data3, 98) if np.any(np.isfinite(data3)) else 1.0
vmax3   = max(vmax3, 1e-4)
im3     = ax.imshow(np.where(np.isfinite(data3), data3, 0),
                    origin='lower', aspect='auto',
                    extent=[np.log10(r_mid_mult[0]), np.log10(r_mid_mult[-1]), 1, L_MAX],
                    vmin=0, vmax=vmax3, cmap='magma')
plt.colorbar(im3, ax=ax, label='Ẽ_l', pad=0.02)
_ax(ax, xlabel='log₁₀(r [kpc])', ylabel='Multipole degree l',
    title=f'§36 Fig 3 — Full Spectrum Ẽ_l(r) at t = {time_mult[peak_k]:.2f} Gyr '
          f'(peak distortion)')
ax.set_yticks(np.arange(1, L_MAX + 1))
ax.set_xticks(np.log10([2, 5, 10, 30, 80]))
ax.set_xticklabels(['2','5','10','30','80'])
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_multipole_radius_spectrum.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 3 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 4 — Quadrupole phase φ_22 and isophote twist              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("§36 Fig 4 — Quadrupole Phase & Isophote Twist", fontsize=10, fontweight='bold')

im4 = ax1.imshow(phase22_arr.T, origin='lower', aspect='auto',
                  extent=[time_mult[0], time_mult[-1],
                          np.log10(r_mid_mult[0]), np.log10(r_mid_mult[-1])],
                  cmap='twilight', vmin=-180, vmax=180)
plt.colorbar(im4, ax=ax1, label='Phase φ₂₂ [deg]', pad=0.02)
_ax(ax1, xlabel='Time [Gyr]', ylabel='log₁₀(r [kpc])',
    title='φ₂₂(r, t)  — quadrupole orientation')
ax1.set_yticks(np.log10([2,5,10,30,80]))
ax1.set_yticklabels(['2','5','10','30','80'])

ax2.plot(time_mult, dPA_22_io, color='steelblue', lw=1.5, label='Inner–outer ΔPA₂₂')
ax2.axhline(0,  color='k',   lw=0.6, ls='--')
ax2.axhline(45, color='red', lw=0.6, ls=':', label='45° twist threshold')
_ax(ax2, xlabel='Time [Gyr]', ylabel='ΔPA₂₂ [deg]',
    title='Inner–Outer Quadrupole Twist (isophote twist)')
ax2.legend(fontsize=7)
ax2.set_ylim(0, 90)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_quadrupole_phase.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 4 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 5 — Minkowski functional curves at 5 epochs               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

mf_epoch_idx = [0, N_MF_SNAPS//4, N_MF_SNAPS//2, 3*N_MF_SNAPS//4, N_MF_SNAPS-1]
fig, axes5   = plt.subplots(1, 5, figsize=(14, 3.5), sharey=False)
fig.suptitle("§36 Fig 5 — Minkowski Functional Curves W(ρ_th)", fontsize=10, fontweight='bold')

for idx, (ei, col) in enumerate(zip(mf_epoch_idx, EPOCH_COLORS)):
    ax   = axes5[idx]
    th   = thresh_grid[ei]
    th_n = (th - th.min()) / (np.ptp(th) + 1e-30)   # normalised threshold

    ax2r = ax.twinx()
    ax.plot(th_n, W0_arr[ei] / (np.nanmax(W0_arr[ei]) + 1e-30),
            color='royalblue', lw=1.4, label='W₀/max')
    ax.plot(th_n, W1_arr[ei] / (np.nanmax(W1_arr[ei]) + 1e-30),
            color='darkorange', lw=1.4, ls='--', label='W₁/max')
    ax2r.step(th_n, chi_arr[ei], where='mid', color='crimson', lw=1.1, alpha=0.7)
    ax2r.set_ylabel('χ', fontsize=7, color='crimson')
    ax2r.tick_params(labelsize=6, colors='crimson')
    _ax(ax, xlabel='ρ_th / ρ_median', ylabel='Norm. MF' if idx == 0 else '',
        title=f't = {time_mf[ei]:.2f} Gyr')
    if idx == 0:
        ax.legend(fontsize=6)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_minkowski_curves.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 5 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 6 — (P, F) morphology plane                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, ax = plt.subplots(figsize=(6, 6))
fig.suptitle("§36 Fig 6 — (P, F) Morphology Plane", fontsize=10, fontweight='bold')

# morphology triangle annotation
ax.fill([0, 1, 0, 0], [0, 0, 1, 0], color='whitesmoke', zorder=0)
ax.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k--', lw=0.7)
ax.text(-0.03, -0.03, 'SPHERE',    fontsize=8, ha='center', color='gray')
ax.text(1.02,  -0.03, 'PANCAKE',   fontsize=8, ha='center', color='gray')
ax.text(-0.03,  1.02, 'FILAMENT',  fontsize=8, ha='center', color='gray')

cmap6 = plt.cm.viridis
for k in range(N_MF_SNAPS):
    Pv = P_arr[k]; Fv = F_arr[k]
    fin = np.isfinite(Pv) & np.isfinite(Fv)
    if fin.sum() < 3:
        continue
    col = cmap6(k / N_MF_SNAPS)
    # smooth interpolation
    try:
        fi_Pv = interp1d(np.where(fin)[0], Pv[fin], kind='linear', fill_value='extrapolate')
        fi_Fv = interp1d(np.where(fin)[0], Fv[fin], kind='linear', fill_value='extrapolate')
        js    = np.linspace(0, N_THRESHOLDS-1, 80)
        Ps    = np.clip(fi_Pv(js), 0, 1)
        Fs    = np.clip(fi_Fv(js), 0, 1)
        ax.plot(Ps, Fs, color=col, lw=0.8, alpha=0.6)
    except Exception:
        ax.plot(Pv[fin], Fv[fin], color=col, lw=0.8, alpha=0.6)
    # median-threshold dot
    ax.scatter(P_arr[k, mid_j], F_arr[k, mid_j], s=18,
               color=col, zorder=4, edgecolors='k', linewidths=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap6,
                             norm=Normalize(vmin=time_mf[0], vmax=time_mf[-1]))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Time [Gyr]', shrink=0.7)
_ax(ax, xlabel='Planarity P_MF', ylabel='Filamentarity F_MF',
    title='Morphology curves (each line = one epoch, sweeping ρ_th)')
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_pf_morphology_plane.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 6 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 7 — Euler characteristic χ(ρ_th, t) heatmap              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("§36 Fig 7 — Euler Characteristic χ(ρ_th, t)", fontsize=10, fontweight='bold')

chi_plot = np.where(np.isfinite(chi_arr), chi_arr, 0).astype(float)
chi_vals = np.unique(chi_plot[np.isfinite(chi_arr)])
vmin_chi = chi_vals.min() if len(chi_vals) > 0 else 0
vmax_chi = chi_vals.max() if len(chi_vals) > 0 else 4

im7 = ax.imshow(chi_plot, origin='lower', aspect='auto',
                extent=[0, 1, time_mf[0], time_mf[-1]],
                interpolation='none', cmap='RdYlGn',
                vmin=vmin_chi, vmax=vmax_chi)
plt.colorbar(im7, ax=ax, label='χ  (Euler characteristic)', pad=0.02)
ax.axhline(time_mf[N_MF_SNAPS//2], color='white', lw=1.0, ls='--', alpha=0.8,
           label='pericentric passage proxy')
_ax(ax, xlabel='Normalised density threshold', ylabel='Time [Gyr]',
    title='Topology: χ=2 (single blob), χ=4 (two blobs), χ<2 (holes/multi-component)')
ax.legend(fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_euler_characteristic.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 7 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 8 — Quadrupole Ẽ_2 vs. § 35 tensor ellipticity            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Here we synthesise a plausible E_inertia proxy (1 − axis ratio estimate)
# from the l=2 power itself, as §35 tensors aren't run in this standalone file.
# In the full pipeline: replace E_proxy with s_shell_ts from §35.

E_proxy = 1.0 - np.exp(-Q_bar_arr * 2.5)   # monotone proxy ∈ [0,1]

fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle("§36 Fig 8 — Ẽ₂ vs. Tensor Ellipticity E", fontsize=10, fontweight='bold')
cmap8 = plt.cm.plasma
for k in range(N_MULTIPOLE_SNAPS):
    E_v  = E_proxy[k]
    Et2  = Q_bar_arr[k]
    fin  = np.isfinite(E_v) & np.isfinite(Et2)
    if fin.sum() < 2:
        continue
    ax.scatter(E_v[fin], Et2[fin], s=8, alpha=0.4,
               c=np.log10(r_mid_mult[fin]), cmap=cmap8,
               vmin=np.log10(r_mid_mult[0]), vmax=np.log10(r_mid_mult[-1]))

sm8 = plt.cm.ScalarMappable(cmap=cmap8,
      norm=Normalize(vmin=np.log10(r_mid_mult[0]), vmax=np.log10(r_mid_mult[-1])))
sm8.set_array([])
plt.colorbar(sm8, ax=ax, label='log₁₀(r [kpc])', shrink=0.7)
diag = np.linspace(0, 1, 50)
ax.plot(diag, diag, 'k--', lw=0.8, label='1:1 agreement')
_ax(ax, xlabel='Ellipticity E = 1 − s  (§35 proxy)',
    ylabel='Quadrupole Ẽ₂  (multipole)',
    title='Cross-method: tensor ellipticity vs. multipole quadrupole')
ax.legend(fontsize=7)
ax.set_xlim(0, 1); ax.set_ylim(0)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_multipole_vs_tensor.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 8 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 9 — MF time series: F(t), χ_min(t), n_chi_jumps(t)        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig, axes9 = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
fig.suptitle("§36 Fig 9 — Minkowski Functional Time Series", fontsize=10, fontweight='bold')

ax9a, ax9b, ax9c = axes9

ax9a.plot(time_mf, F_med_ts, color='royalblue',  lw=1.5, label='Filamentarity F_MF')
ax9a.plot(time_mf, P_med_ts, color='darkorange', lw=1.5, ls='--', label='Planarity P_MF')
ax9a.fill_between(time_mf, F_med_ts, P_med_ts, alpha=0.15, color='gray')
_ax(ax9a, ylabel='P, F  (median threshold)', title='Planarity & Filamentarity')
ax9a.legend(fontsize=7)
ax9a.set_ylim(0)

ax9b.plot(time_mf, chi_min_ts, color='crimson', lw=1.5, marker='o', ms=3)
ax9b.axhline(2,  color='green', lw=0.8, ls=':', label='χ=2 (single blob)')
ax9b.axhline(4,  color='navy',  lw=0.8, ls=':', label='χ=4 (two blobs)')
_ax(ax9b, ylabel='χ_min', title='Minimum Euler Characteristic (topology indicator)')
ax9b.legend(fontsize=7)

ax9c.bar(time_mf, n_chi_jumps, width=(time_mf[1]-time_mf[0])*0.8,
         color='steelblue', alpha=0.7, label='# topology changes')
_ax(ax9c, xlabel='Time [Gyr]', ylabel='n_χ_jumps',
    title='Topology Change Count per Epoch')
ax9c.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "section36_mf_timeseries.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print("  Fig 9 saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.9  FIGURE 10 — Master summary panel (3×2)                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

fig10 = plt.figure(figsize=(14, 11))
fig10.suptitle("§36 Master Summary — Multipole & Minkowski Morphology",
               fontsize=11, fontweight='bold')
gs10 = gridspec.GridSpec(3, 2, figure=fig10, hspace=0.45, wspace=0.35)
ax00 = fig10.add_subplot(gs10[0, 0])
ax01 = fig10.add_subplot(gs10[0, 1])
ax10 = fig10.add_subplot(gs10[1, 0])
ax11 = fig10.add_subplot(gs10[1, 1])
ax20 = fig10.add_subplot(gs10[2, 0])
ax21 = fig10.add_subplot(gs10[2, 1])

# (0,0) Ẽ_2 heatmap
vmax00 = max(np.nanpercentile(Q_bar_arr, 97), 1e-3)
im00 = ax00.imshow(Q_bar_arr.T, origin='lower', aspect='auto',
                    extent=[time_mult[0], time_mult[-1],
                            np.log10(r_mid_mult[0]), np.log10(r_mid_mult[-1])],
                    vmin=0, vmax=vmax00, cmap='plasma')
plt.colorbar(im00, ax=ax00, label='Ẽ₂', pad=0.02)
_ax(ax00, xlabel='Time [Gyr]', ylabel='log₁₀(r)', title='Quadrupole Ẽ₂(r,t)')

# (0,1) Ẽ_1 heatmap
vmax01 = max(np.nanpercentile(D_bar_arr, 97), 1e-3)
im01 = ax01.imshow(D_bar_arr.T, origin='lower', aspect='auto',
                    extent=[time_mult[0], time_mult[-1],
                            np.log10(r_mid_mult[0]), np.log10(r_mid_mult[-1])],
                    vmin=0, vmax=vmax01, cmap='inferno')
plt.colorbar(im01, ax=ax01, label='Ẽ₁', pad=0.02)
_ax(ax01, xlabel='Time [Gyr]', ylabel='log₁₀(r)', title='Dipole Ẽ₁(r,t)  (lopsidedness)')

# (1,0) (P,F) density map — 2D histogram over all epochs
fin_pf = np.isfinite(P_arr) & np.isfinite(F_arr)
P_all  = P_arr[fin_pf].ravel()
F_all  = F_arr[fin_pf].ravel()
if len(P_all) > 10:
    h, xedge, yedge = np.histogram2d(P_all, F_all, bins=30,
                                       range=[[0,1],[0,1]])
    im10 = ax10.imshow(h.T, origin='lower', aspect='auto',
                       extent=[0,1,0,1], cmap='hot_r')
    plt.colorbar(im10, ax=ax10, label='Count', pad=0.02)
ax10.plot([0,1,0,0],[0,0,1,0],'k--',lw=0.8)
ax10.text(0.01, -0.06, 'Sphere', fontsize=7, color='gray')
ax10.text(0.85, -0.06, 'Pancake', fontsize=7, color='gray')
ax10.text(-0.12, 0.95, 'Filament', fontsize=7, color='gray', rotation=90)
_ax(ax10, xlabel='Planarity P', ylabel='Filamentarity F',
    title='(P,F) density — all epochs & thresholds')
ax10.set_xlim(0,1); ax10.set_ylim(0,1)

# (1,1) Euler characteristic heatmap
im11 = ax11.imshow(chi_plot, origin='lower', aspect='auto',
                    extent=[0,1,time_mf[0],time_mf[-1]],
                    interpolation='none', cmap='RdYlGn',
                    vmin=vmin_chi, vmax=vmax_chi)
plt.colorbar(im11, ax=ax11, label='χ', pad=0.02)
_ax(ax11, xlabel='Normalised ρ_th', ylabel='Time [Gyr]',
    title='Euler characteristic χ(ρ_th, t)')

# (2,0) Scatter Ẽ_2 vs E_proxy (compressed Fig 8)
for k in range(0, N_MULTIPOLE_SNAPS, 4):
    fin = np.isfinite(E_proxy[k]) & np.isfinite(Q_bar_arr[k])
    ax20.scatter(E_proxy[k, fin], Q_bar_arr[k, fin], s=5, alpha=0.3,
                 c=[plt.cm.viridis(k/N_MULTIPOLE_SNAPS)]*fin.sum())
ax20.plot([0,1],[0,1],'k--',lw=0.7)
_ax(ax20, xlabel='E_inertia (§35 proxy)', ylabel='Ẽ₂ (multipole)',
    title='Cross-check: tensor ellipticity vs. quadrupole')

# (2,1) Five-method normalised time series
def _norm01(arr):
    lo = np.nanmin(arr); hi = np.nanmax(arr)
    return (arr - lo) / (hi - lo + 1e-30)

Et2_g    = Etilde_global[:, 2]         # global Ẽ_2, on mult time grid
dPA_norm = _norm01(dPA_22_io)
Et2_norm = _norm01(Et2_g)
F_norm   = _norm01(F_med_ts)
chi_norm = _norm01(-chi_min_ts)         # invert: lower χ = more distorted

# resample MF onto mult time grid for comparison
from scipy.interpolate import interp1d as _interp
def _resamp(t_src, y_src, t_dst):
    fin = np.isfinite(y_src)
    if fin.sum() < 2:
        return np.full(len(t_dst), np.nan)
    f = _interp(t_src[fin], y_src[fin], bounds_error=False, fill_value=np.nan)
    return f(t_dst)

F_norm_r   = _norm01(_resamp(time_mf,  F_med_ts,  time_mult))
chi_norm_r = _norm01(-_resamp(time_mf, chi_min_ts, time_mult))

ax21.plot(time_mult, Et2_norm,    lw=1.3, color='royalblue',  label='Ẽ₂ global (mult.)')
ax21.plot(time_mult, dPA_norm,    lw=1.3, color='darkorange', label='ΔPA₂₂ twist')
ax21.plot(time_mult, F_norm_r,    lw=1.3, color='green',      label='F_MF (MF)')
ax21.plot(time_mult, chi_norm_r,  lw=1.3, color='crimson', ls='--', label='−χ_min (MF)')
_ax(ax21, xlabel='Time [Gyr]', ylabel='Normalised [0→1]',
    title='Five-method normalised comparison')
ax21.legend(fontsize=6, ncol=2)
ax21.set_ylim(-0.05, 1.1)

plt.savefig(os.path.join(OUT_DIR, "section36_summary_panel.png"),
            dpi=130, bbox_inches='tight')
plt.close(fig10)
print("  Fig 10 (summary panel) saved.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.10  ANIMATION — Morphological evolution                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n  Building animation …")
fig_a, axes_a = plt.subplots(1, 3, figsize=(13, 4))
fig_a.suptitle("§36 — Morphological Evolution", fontsize=10, fontweight='bold')
ax_bar, ax_pf, ax_ts = axes_a

# ── Left: bar chart of Ẽ_l ───────────────────────────────────────────────────
l_vals = np.arange(1, L_MAX + 1)
Et0    = Etilde_arr[0, mid_shell, 1:]
Et0    = np.where(np.isfinite(Et0), Et0, 0.0)
bars   = ax_bar.bar(l_vals, Et0, color='steelblue', edgecolor='k', linewidth=0.4)

# theoretical l=2 only line
Et2_theory_line, = ax_bar.plot([2, 2], [0, max(Et0.max(), 0.01)],
                                color='red', lw=1.5, ls='--', label='l=2 theo.')
_ax(ax_bar, xlabel='l', ylabel='Ẽ_l', title='Multipole spectrum (mid shell)')
ax_bar.set_xticks(l_vals)
ax_bar.legend(fontsize=7)

# ── Centre: (P, F) morphology plane ──────────────────────────────────────────
ax_pf.fill([0,1,0,0],[0,0,1,0], color='whitesmoke', zorder=0)
ax_pf.plot([0,1,0,0],[0,0,1,0],'k--',lw=0.7)
for label_txt, xy in [('Sphere',(0,0)),('Pancake',(1,0)),('Filament',(0,1))]:
    ax_pf.text(xy[0]-0.02, xy[1]-0.06, label_txt, fontsize=7, color='gray')

hist_lines = []   # grey history lines
cur_line,  = ax_pf.plot([], [], 'b-',  lw=1.5, zorder=4)
cur_dot    = ax_pf.scatter([], [], s=60, color='blue', zorder=5,
                            edgecolors='k', linewidths=0.5)
_ax(ax_pf, xlabel='P_MF', ylabel='F_MF', title='Morphology plane (curve sweeps ρ_th)')
ax_pf.set_xlim(-0.05, 1.05); ax_pf.set_ylim(-0.05, 1.05)

# ── Right: time series ───────────────────────────────────────────────────────
ax_ts.set_xlim(time_mf[0], time_mf[-1])
_ax(ax_ts, xlabel='Time [Gyr]', ylabel='Ẽ₂ / F_MF (normed)',
    title='Global Ẽ₂ and F_MF over time')
ts_line_E2, = ax_ts.plot([], [], color='royalblue', lw=1.5, label='Ẽ₂_global')
ts_line_F,  = ax_ts.plot([], [], color='green',     lw=1.5, ls='--', label='F_MF')
ts_vline    = ax_ts.axvline(time_mf[0], color='gray', lw=0.8, ls=':')
ax_ts.legend(fontsize=7)
ax_ts.set_ylim(0)

# align MF snap index to multipole snap index
mf_of_mult = np.searchsorted(snap_indices_mf, snap_indices_mult, side='left')
mf_of_mult = np.clip(mf_of_mult, 0, N_MF_SNAPS - 1)

Et2_g_norm = _norm01(Etilde_global[:, 2])
F_ts_all   = np.array([float(np.nanmean(F_arr[mf_of_mult[k], mid_j-1:mid_j+2]))
                        for k in range(N_MULTIPOLE_SNAPS)])
F_ts_norm  = _norm01(F_ts_all)
time_axis  = time_mf

max_bars_y = max(np.nanmax(Etilde_arr[:, mid_shell, 1:].ravel()), 0.05) * 1.15

def _anim_update(frame):
    k  = frame                      # multipole snap index
    k_mf = mf_of_mult[k]           # corresponding MF snap

    # --- bar chart ---
    Et_k = Etilde_arr[k, mid_shell, 1:]
    Et_k = np.where(np.isfinite(Et_k), Et_k, 0.0)
    for bar, h in zip(bars, Et_k):
        bar.set_height(max(h, 0))
    ax_bar.set_ylim(0, max_bars_y)
    ax_bar.set_title(f'Multipole mid shell  t={time_mult[k]:.2f} Gyr', fontsize=8)
    # update theory line height to l=2 bar
    h2 = Et_k[1] if len(Et_k) > 1 else 0
    et2_theory_line_y = [0, h2]
    et2_theory_line_x = [2, 2]
    Et2_theory_line.set_data(et2_theory_line_x, et2_theory_line_y)

    # --- (P,F) plane ---
    # add grey history line for previous epoch
    if k > 0 and k_mf > 0:
        Pp = P_arr[k_mf - 1]; Fp = F_arr[k_mf - 1]
        finp = np.isfinite(Pp) & np.isfinite(Fp)
        if finp.sum() >= 2:
            hl, = ax_pf.plot(Pp[finp], Fp[finp], color='lightgray',
                             lw=0.7, alpha=0.5, zorder=2)
            hist_lines.append(hl)
            # keep only last 8 grey lines
            if len(hist_lines) > 8:
                hist_lines[0].remove()
                hist_lines.pop(0)

    Pc = P_arr[k_mf]; Fc = F_arr[k_mf]
    fin_c = np.isfinite(Pc) & np.isfinite(Fc)
    if fin_c.sum() >= 2:
        cur_line.set_data(Pc[fin_c], Fc[fin_c])
    if np.isfinite(P_arr[k_mf, mid_j]) and np.isfinite(F_arr[k_mf, mid_j]):
        cur_dot.set_offsets([[P_arr[k_mf, mid_j], F_arr[k_mf, mid_j]]])
    ax_pf.set_title(f'(P,F) morphology  t={time_mf[k_mf]:.2f} Gyr', fontsize=8)

    # --- time series ---
    ts_line_E2.set_data(time_mult[:k+1], Et2_g_norm[:k+1])
    ts_line_F.set_data(time_mult[:k+1],  F_ts_norm[:k+1])
    ts_vline.set_xdata([time_mult[k], time_mult[k]])
    ax_ts.set_ylim(0, 1.1)
    ax_ts.set_title(f'Global Ẽ₂ & F_MF  t={time_mult[k]:.2f} Gyr', fontsize=8)

    return list(bars) + [cur_line, cur_dot, ts_line_E2, ts_line_F, ts_vline,
                          Et2_theory_line]

anim = animation.FuncAnimation(fig_a, _anim_update,
                                 frames=N_MULTIPOLE_SNAPS,
                                 interval=1000 // ANIM_FPS_36, blit=False)
plt.tight_layout()

anim_path = os.path.join(OUT_DIR, "section36_animation_morphology.mp4")
try:
    writer = animation.FFMpegWriter(fps=ANIM_FPS_36, bitrate=ANIM_BITRATE_36)
    anim.save(anim_path, writer=writer, dpi=ANIM_DPI_36)
    print(f"  Animation saved → {anim_path}")
except Exception as exc:
    print(f"  FFMpeg unavailable ({exc}); saving GIF fallback.")
    gif_path = anim_path.replace('.mp4', '.gif')
    anim.save(gif_path, writer='pillow', fps=ANIM_FPS_36)
    print(f"  GIF saved → {gif_path}")
plt.close(fig_a)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.11  CROSS-SECTION CORRELATIONS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "─"*70)
print("  §36 Cross-Section Pearson Correlations")
print("─"*70)

def _safe_pearson(a, b):
    """Pearson r between two arrays, NaN-safe, minimum 5 finite pairs."""
    fin = np.isfinite(a) & np.isfinite(b)
    if fin.sum() < 5:
        return np.nan, np.nan
    return pearsonr(a[fin], b[fin])

# build comparable time vectors on the MF time grid
Et2_mf   = _resamp(time_mult, Etilde_global[:, 2],  time_mf)
dPA_mf   = _resamp(time_mult, dPA_22_io,             time_mf)
D_outer  = _resamp(time_mult,
                   np.nanmean(D_bar_arr[:, -N_MULTIPOLE_SHELLS//4:], axis=1),
                   time_mf)

corr_rows = [
    ("Ẽ₂_global",   "E_inertia proxy",  Etilde_global[:,2],      _norm01(E_proxy.mean(1))),
    ("Ẽ₁_outer",    "F_MF",             D_outer,                  F_med_ts),
    ("F_med_MF",    "Ẽ₂_global",        F_med_ts,                 Et2_mf),
    ("χ_min",       "n_chi_jumps",      chi_min_ts,               n_chi_jumps),
    ("dPA₂₂_io",   "A_PF",             dPA_mf,                   A_PF_ts),
    ("A_PF",        "F_peak",           A_PF_ts,                  F_peak_ts),
]

header = f"  {'Quantity 1':<20} {'Quantity 2':<22} {'Pearson r':>10} {'p-value':>10}"
print(header)
print("  " + "─"*68)
for name1, name2, a, b in corr_rows:
    r, p = _safe_pearson(np.asarray(a, float), np.asarray(b, float))
    p_str = f"{p:.3e}" if np.isfinite(p) else "    —"
    r_str = f"{r:+.4f}" if np.isfinite(r) else "    —"
    print(f"  {name1:<20} {name2:<22} {r_str:>10} {p_str:>10}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.12  MORPHOLOGICAL CLASSIFICATION TABLE                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _classify(E2, F, P, chi):
    """Classify halo morphology from four scalar diagnostics."""
    if not all(np.isfinite(v) for v in [E2, F, P]):
        return "unknown"
    if E2 < 0.05 and F < 0.1 and P < 0.1:
        return "sphere"
    if not np.isfinite(chi) or chi < 2:
        return "complex (substructure)"
    if F > P and F > 0.2:
        return "prolate / filament"
    if P > F and P > 0.2:
        return "oblate / pancake"
    if abs(F - P) <= 0.10 and F > 0.1:
        return "triaxial"
    return "mildly distorted"

# Find three representative epochs in MF time grid
ep_names   = ["t = t_initial", "t = t_peri (proxy)", "t = t_final"]
ep_mf_idx  = [0, N_MF_SNAPS // 2, N_MF_SNAPS - 1]
# interpolate Ẽ_2 to MF snap times
Et2_at_mf  = _resamp(time_mult, Etilde_global[:, 2], time_mf)

print("\n" + "─"*80)
print("  §36 Morphological Classification Table")
print("─"*80)
hdr = (f"  {'Epoch':<22} {'Ẽ₂':>6} {'F_MF':>6} {'P_MF':>6} "
       f"{'χ_min':>7}  Classification")
print(hdr)
print("  " + "─"*78)
for name, ki in zip(ep_names, ep_mf_idx):
    E2  = float(Et2_at_mf[ki])  if np.isfinite(Et2_at_mf[ki])  else np.nan
    F   = float(F_med_ts[ki])   if np.isfinite(F_med_ts[ki])   else np.nan
    P   = float(P_med_ts[ki])   if np.isfinite(P_med_ts[ki])   else np.nan
    chi = float(chi_min_ts[ki]) if np.isfinite(chi_min_ts[ki]) else np.nan
    cls = _classify(E2, F, P, chi)
    E2s  = f"{E2:6.3f}" if np.isfinite(E2)  else "   —"
    Fs   = f"{F:6.3f}"  if np.isfinite(F)   else "   —"
    Ps   = f"{P:6.3f}"  if np.isfinite(P)   else "   —"
    chis = f"{chi:7.1f}" if np.isfinite(chi) else "      —"
    print(f"  {name:<22} {E2s} {Fs} {Ps} {chis}  {cls}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  §36.13  SECTION COMPLETE — OUTPUT MANIFEST                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

outputs_36 = [
    "section36_multipole_spectra.png",
    "section36_multipole_heatmaps.png",
    "section36_multipole_radius_spectrum.png",
    "section36_quadrupole_phase.png",
    "section36_minkowski_curves.png",
    "section36_pf_morphology_plane.png",
    "section36_euler_characteristic.png",
    "section36_multipole_vs_tensor.png",
    "section36_mf_timeseries.png",
    "section36_animation_morphology.mp4",
    "section36_summary_panel.png",
]

print("\n" + "="*80)
print("  SECTION 36 COMPLETE")
print("="*80)
print(f"  Output directory : {OUT_DIR}")
print(f"  {'File':<48} {'Status'}")
print("  " + "─"*60)
for fname in outputs_36:
    fpath  = os.path.join(OUT_DIR, fname)
    exists = os.path.isfile(fpath)
    size   = f"{os.path.getsize(fpath)/1024:.0f} KB" if exists else "—"
    status = f"✓  {size}" if exists else "✗  MISSING"
    print(f"  {fname:<48} {status}")

print("\n  Method summary:")
print(f"    Multipole L_MAX         : {L_MAX}")
print(f"    Shells analysed         : {N_MULTIPOLE_SHELLS}")
print(f"    Snapshot epochs (mult.) : {N_MULTIPOLE_SNAPS}")
print(f"    Grid resolution         : {GRID_RES}³")
print(f"    Smoothing               : {SMOOTH_SIGMA} voxels  "
      f"≈ {SMOOTH_SIGMA * 2*R_OUTER/GRID_RES:.1f} kpc physical")
print(f"    Threshold epochs (MF)   : {N_MF_SNAPS}")
print(f"    Thresholds per epoch    : {N_THRESHOLDS}")
