# """
# ==============================================================================
#                 MW–M31 MERGER PROFILE & DENSITY DIAGNOSTICS
# ==============================================================================
#
# Author: Abhinav Vatsa
#
# DESCRIPTION:
# This pipeline acts as the structural counterpart to kinematic_profiles_pipeline.py.
# While kinematics charts the velocity field and dispersion profiles, this module
# maps the spatial distribution, mass concentrations, and mixing dynamics of the
# MW-M31 collision. 
#
# SPHERICAL METRICS (f(r)):
#   - ρ(r)      : 3D mass density profile (mass-weighted, radial shell volume normalized)
#   - Σ(R)      : Cylinder-projected surface density profile
#   - Γ(r)      : Logarithmic density slope: d(ln ρ) / d(ln r)
#   - ρ_MW(r)   : Radial density distribution of progenitor Milky Way material
#   - ρ_M31(r)  : Radial density distribution of progenitor Andromeda material
#   - f_mix(r)  : Spatial mixing index of the two stellar/dark systems
#
# PROJECTED 2D TOPOGRAPHY (x-y Plane):
#   - Σ(x, y)   : Total projected map of density field
#   - Σ_MW(x,y) : Progenitor-specific MW projected distribution
#   - Σ_M31(x,y): Progenitor-specific M31 projected distribution
#
# ANALYTICAL PARAMETRIC FITS:
#   - NFW Fit      : Direct density fit to Navarro-Frenk-White profiles
#   - Einasto Fit  : 3-parameter shape fitting (varying alpha index)
#   - Hernquist    : Analytic density comparison template
#   - Residuals    : Log-scale fraction deviations (Data - Model) / Model
#
# GLOBAL STRUCTURAL SCALARS:
#   - r_eff_3D   : Continuous 3D half-mass radius
#   - r_half     : Continuous projected (2D) half-mass radius
#   - ρ_0        : Central core density estimate
#   - Γ_inner    : Power-law index for core regions (r < 5 kpc)
#   - Γ_outer    : Power-law index for halo regions (r > 50 kpc)
#
# ==============================================================================
# """

# ── Standard Library Imports ──────────────────────────────────────────────────
import os
import tarfile
import shutil
import tempfile
import warnings
import time

# ── Scientific & Plotting Stack ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# ── Simulation Data Readers ──────────────────────────────────────────────────
from ReadFile import Read
from CenterOfMass2 import CenterOfMass

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 20 — CONFIGURATION & ENVIRONMENT SETUP                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Temporal Bounds ───────────────────────────────────────────────────────────
START_SNAP = 0
END_SNAP   = 800

# ── Target Simulation Component ────────────────────────────────────────────────
# 1 = Dark Matter Halo components (captures deep gravitational potential structure)
# 2 = Disk particles (baryonic tracer dynamics)
PTYPE = 1

# ── Radial Geometry ───────────────────────────────────────────────────────────
# Logarithmic distribution from sub-kpc scales up to outer virial boundaries.
# Pre-defined to interface seamlessly with kinematic tracking.
R_BINS = np.logspace(-1, np.log10(400.0), 40)   # kpc, resulting in 39 bins
MIN_PART_SHELL = 20                             # Shot-noise limit threshold

# ── 2D Map Projections ────────────────────────────────────────────────────────
# Maps are projected in the principal x-y plane.
MAP_EXTENT_KPC = 400.0                          # Field boundary (±kpc)
MAP_BINS       = 256                            # Pixel resolution grid (256x256)
MAP_SMOOTH_SIGMA = 2.0                          # Gaussian kernel size in pixels (~6 kpc)

# ── 2D Cylinder Annulus Definitions ───────────────────────────────────────────
# For projected surface density profiles Σ(R).
R_PROJ_BINS = np.logspace(np.log10(0.5), np.log10(400.0), 35) # 34 radial zones

# ── Model Fitting Parameters ──────────────────────────────────────────────────
R_FIT_MIN_KPC = 1.0                             # Exclude unresolved inner parsecs
R_FIT_MAX_KPC = 200.0                            # Exclude tidal debris and companion mergers

# ── Execution Optimization Steps ──────────────────────────────────────────────
STEP_MAPS  = 8                                  # Snapshot interval for 2D maps
STEP_FIT   = 4                                  # Snapshot interval for profile fitting

# ── Animation Render Options ──────────────────────────────────────────────────
ANIM_FPS     = 20
ANIM_DPI     = 100
ANIM_BITRATE = 2000
ANIM_STEP    = 4

# ── Astrophysics Conversion Constants ─────────────────────────────────────────
G_KPC_KMS2_MSUN = 4.30091e-6                    # Gravitational Constant
MASS_UNIT_MSUN  = 1.0e10                        # Mass normalization scale factor

# ── Directory Configurations ──────────────────────────────────────────────────
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Aesthetic Dark Theme Parameters ───────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d18",
    "axes.facecolor":   "#0d0d18",
    "axes.edgecolor":   "#2a2a4a",
    "axes.labelcolor":  "#c8c8e8",
    "axes.grid":        True,
    "grid.color":       "#1e1e36",
    "grid.linewidth":   0.6,
    "xtick.color":      "#9090b0",
    "ytick.color":      "#9090b0",
    "text.color":       "#c8c8e8",
    "legend.facecolor": "#0d0d18",
    "legend.edgecolor": "#2a2a4a",
    "legend.fontsize":  8,
    "font.family":      "monospace",
})

# ── Derived Coordinates & Static Volumes ──────────────────────────────────────
SNAPSHOTS    = np.arange(START_SNAP, END_SNAP + 1)
ns           = len(SNAPSHOTS)
nb_sph       = len(R_BINS) - 1
nb_proj      = len(R_PROJ_BINS) - 1
r_mid_sph    = 0.5 * (R_BINS[:-1]   + R_BINS[1:])
r_mid_proj   = 0.5 * (R_PROJ_BINS[:-1] + R_PROJ_BINS[1:])

# Analytical integration of volumes: V = 4/3 * pi * (r_out^3 - r_in^3)
shell_vols   = (4.0 / 3.0) * np.pi * (R_BINS[1:]**3 - R_BINS[:-1]**3)

# Analytical projected areas: A = pi * (R_out^2 - R_in^2)
ring_areas   = np.pi * (R_PROJ_BINS[1:]**2 - R_PROJ_BINS[:-1]**2)

# Subsampling indices for time-evolution snaps
PROFILE_FRACS   = [0.0, 0.2, 0.4, 0.65, 1.0]
PROFILE_INDICES = [int(f * (ns - 1)) for f in PROFILE_FRACS]
PROFILE_LABELS  = [f"Snap {SNAPSHOTS[k]}" for k in PROFILE_INDICES]
PROFILE_COLORS  = ["#00d4aa", "#7b9fff", "#ffaa44", "#ff6b9a", "#aa88ff"]

print(f"[Config] {ns} snapshots  ·  {nb_sph} spherical bins  ·  "
      f"{nb_proj} projected bins  ·  {MAP_BINS}² map grid")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 21 — RAW DATA INGESTION & CO-SET ALIGNMENT                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_snapshots(work_dir: str) -> str:
    """
    Decompresses and organizes companion simulation files from directory archives.
    All extracted files are stored flat inside a system-designated temp location.
    """
    tmpdir = tempfile.mkdtemp(prefix="density_snaps_")
    print(f"[extract] Temporary Workspace Created: {tmpdir}")

    tar_files = [fn for fn in os.listdir(work_dir) if fn.endswith(".tar")]
    if not tar_files:
        warnings.warn("No .tar files located. Assuming local text flat-files are available.")
        return tmpdir

    for fn in tar_files:
        print(f"  Unpacking {fn}...")
        with tarfile.open(os.path.join(work_dir, fn), "r") as tar:
            targets = [m for m in tar.getmembers()
                       if m.isfile() and ("MW_" in m.name or "M31_" in m.name)]
            for m in targets:
                m.name = os.path.basename(m.name)
                tar.extract(m, path=tmpdir)
            print(f"  Successfully extracted {len(targets)} components.")
    return tmpdir


def load_snapshot_particles(mw_path: str, m31_path: str):
    """
    Loads spatial coordinates and mass structures for a given snapshot,
    re-centering the entire phase-space on the joint center of mass.
    """
    MW  = CenterOfMass(mw_path,  PTYPE)
    M31 = CenterOfMass(m31_path, PTYPE)

    # ── Master Stack Concatenation ────────────────────────────────────────────
    x = np.concatenate((MW.x, M31.x))
    y = np.concatenate((MW.y, M31.y))
    z = np.concatenate((MW.z, M31.z))
    m_raw = np.concatenate((MW.m, M31.m))
    m_msun = m_raw * MASS_UNIT_MSUN

    # ── Track Progenitor Origin ───────────────────────────────────────────────
    # 0 = Milky Way Origin, 1 = M31 Origin
    origin = np.concatenate((
        np.zeros(len(MW.x),  dtype=np.int8),
        np.ones (len(M31.x), dtype=np.int8),
    ))

    # ── Joint System Gravitational Center ─────────────────────────────────────
    xcom, ycom, zcom = MW.COMdefine(x, y, z, m_raw)

    # ── Relative Coordinate Offset Transformation ──────────────────────────────
    pos = np.vstack((x - xcom, y - ycom, z - zcom)).T

    # ── Parse Temporal Metadata ───────────────────────────────────────────────
    sim_time = None
    if hasattr(MW, "time"):
        try:
            sim_time = float(MW.time.value)
        except Exception:
            try:
                sim_time = float(MW.time)
            except Exception:
                pass

    return {"pos": pos, "m_msun": m_msun, "origin": origin, "time": sim_time}


# ── Initialize Ingestion Workspace ────────────────────────────────────────────
tmpdir = extract_snapshots(".")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 22 — VECTORIZED DENSITY PROFILE ENGINE                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_density_profiles(snap_data: dict) -> dict:
    """
    Generates 1D radial profiles and projected surface density curves.
    This version replaces iterative masking with highly optimized, vectorized 
    numpy histogramming to handle mass integration.
    """
    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    origin = snap_data["origin"]

    # Calculate radial distances in both 3D spherical space and 2D projected space
    r_3d   = np.linalg.norm(pos, axis=1)
    R_proj = np.linalg.norm(pos[:, :2], axis=1)

    # Masks for dividing Milky Way and Andromeda materials
    mw_mask  = (origin == 0)
    m31_mask = (origin == 1)

    # ── Vectorized Mass Distribution Calculations ──
    # Computes total mass inside each spherical shell boundary in one call
    mass_in_shells, _    = np.histogram(r_3d, bins=R_BINS, weights=m)
    mw_mass_in_shells, _ = np.histogram(r_3d[mw_mask], bins=R_BINS, weights=m[mw_mask])
    m31_mass_in_shells, _ = np.histogram(r_3d[m31_mask], bins=R_BINS, weights=m[m31_mask])

    # Counts particles inside shells to filter low-density, high-noise zones
    counts_in_shells, _ = np.histogram(r_3d, bins=R_BINS)
    valid_shells        = counts_in_shells >= MIN_PART_SHELL

    # ── Convert Mass to Spherical Densities (M_sun/kpc^3) ──
    rho     = np.where(valid_shells, mass_in_shells / shell_vols, np.nan)
    rho_mw  = np.where(valid_shells & (mw_mass_in_shells > 0), mw_mass_in_shells / shell_vols, np.nan)
    rho_m31 = np.where(valid_shells & (m31_mass_in_shells > 0), m31_mass_in_shells / shell_vols, np.nan)

    # ── Vectorized 2D Surface Densities (M_sun/kpc^2) ──
    mass_in_rings, _    = np.histogram(R_proj, bins=R_PROJ_BINS, weights=m)
    mw_mass_in_rings, _ = np.histogram(R_proj[mw_mask], bins=R_PROJ_BINS, weights=m[mw_mask])
    m31_mass_in_rings, _ = np.histogram(R_proj[m31_mask], bins=R_PROJ_BINS, weights=m[m31_mask])
    
    counts_in_rings, _  = np.histogram(R_proj, bins=R_PROJ_BINS)
    valid_rings         = counts_in_rings >= MIN_PART_SHELL

    Sigma     = np.where(valid_rings, mass_in_rings / ring_areas, np.nan)
    Sigma_mw  = np.where(valid_rings & (mw_mass_in_rings > 0), mw_mass_in_rings / ring_areas, np.nan)
    Sigma_m31 = np.where(valid_rings & (m31_mass_in_rings > 0), m31_mass_in_rings / ring_areas, np.nan)

    # ── Mixing Dynamics Analysis ──
    # Defined as: f_mix = min(rho_mw, rho_m31) / (rho_mw + rho_m31)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_mix = np.minimum(np.nan_to_num(rho_mw), np.nan_to_num(rho_m31)) / rho

    # ── Numerical Logarithmic Differentiation ──
    # Calculates: Gamma = d(ln rho) / d(ln r)
    # Uses central differences across local log steps to remain robust against noise
    ln_r   = np.log(r_mid_sph)
    ln_rho = np.log(rho)
    
    Gamma = np.full(nb_sph, np.nan)
    finite_idx = np.where(np.isfinite(ln_rho))[0]
    
    if len(finite_idx) > 3:
        # Interpolate missing gaps to preserve gradient continuity
        ln_rho_clean = np.interp(ln_r, ln_r[finite_idx], ln_rho[finite_idx])
        
        # Central difference stencil
        dy = np.zeros_like(ln_rho_clean)
        dx = np.zeros_like(ln_r)
        
        dy[1:-1] = ln_rho_clean[2:] - ln_rho_clean[:-2]
        dx[1:-1] = ln_r[2:] - ln_r[:-2]
        
        # Boundary forward/backward stencils
        dy[0]  = ln_rho_clean[1] - ln_rho_clean[0]
        dx[0]  = ln_r[1] - ln_r[0]
        dy[-1] = ln_rho_clean[-1] - ln_rho_clean[-2]
        dx[-1] = ln_r[-1] - ln_r[-2]
        
        Gamma = dy / dx
        # Re-apply NaN values to mask unreliable outer bins
        Gamma[~np.isin(np.arange(nb_sph), finite_idx)] = np.nan

    # ── Derive Central Density & Dynamic Half-Mass Radii ──
    # Core density: First resolved inner bin value
    rho0 = next((val for val in rho if np.isfinite(val)), np.nan)

    # Continuous 3D Half-Mass Radius (calculated using linear interpolation)
    total_mass = m.sum()
    r_sorted_3d = np.sort(r_3d)
    cum_mass_3d = np.cumsum(m[np.argsort(r_3d)])
    r_half_3d = np.interp(0.5 * total_mass, cum_mass_3d, r_sorted_3d) if total_mass > 0 else np.nan

    # Continuous Projected Half-Mass Radius (2D cylinder integration)
    R_sorted_proj = np.sort(R_proj)
    cum_mass_proj = np.cumsum(m[np.argsort(R_proj)])
    r_half_proj = np.interp(0.5 * total_mass, cum_mass_proj, R_sorted_proj) if total_mass > 0 else np.nan

    # ── Compute Asymptotic Slope Slopes ──
    inner_range = (r_mid_sph < 5.0) & np.isfinite(Gamma)
    outer_range = (r_mid_sph > 50.0) & np.isfinite(Gamma)
    Gamma_inner = np.nanmean(Gamma[inner_range]) if inner_range.any() else np.nan
    Gamma_outer = np.nanmean(Gamma[outer_range]) if outer_range.any() else np.nan

    return {
        "rho":         rho,
        "rho_mw":      rho_mw,
        "rho_m31":     rho_m31,
        "f_mix":       f_mix,
        "Gamma":       Gamma,
        "Sigma":       Sigma,
        "Sigma_mw":    Sigma_mw,
        "Sigma_m31":   Sigma_m31,
        "rho0":        rho0,
        "r_half_3d":   r_half_3d,
        "r_half_proj": r_half_proj,
        "Gamma_inner": Gamma_inner,
        "Gamma_outer": Gamma_outer,
        "time":        snap_data["time"],
    }


def compute_2d_map(snap_data: dict) -> dict:
    """
    Projects particle distributions onto a cartesian mesh to build structural maps.
    Masses are processed via 2D binning to output true Surface Densities (M_sun/kpc^2).
    """
    pos    = snap_data["pos"]
    m      = snap_data["m_msun"]
    origin = snap_data["origin"]

    x_coords = pos[:, 0]
    y_coords = pos[:, 1]
    
    lim = MAP_EXTENT_KPC
    bin_range = [[-lim, lim], [-lim, lim]]
    grid_res  = [MAP_BINS, MAP_BINS]

    # Calculate physical dimensions of individual grid elements
    pixel_width = (2.0 * lim) / MAP_BINS
    pixel_area  = pixel_width ** 2

    # Vectorized 2D Mass aggregation
    mass_grid_total, _, _ = np.histogram2d(x_coords, y_coords, bins=grid_res, range=bin_range, weights=m)
    
    mw_mask  = (origin == 0)
    m31_mask = (origin == 1)
    
    mass_grid_mw, _, _  = np.histogram2d(x_coords[mw_mask], y_coords[mw_mask], bins=grid_res, range=bin_range, weights=m[mw_mask])
    mass_grid_m31, _, _ = np.histogram2d(x_coords[m31_mask], y_coords[m31_mask], bins=grid_res, range=bin_range, weights=m[m31_mask])

    # Convert to surface density, marking empty spatial zones as NaN for proper background plotting
    Sigma_2d     = np.where(mass_grid_total > 0, mass_grid_total / pixel_area, np.nan)
    Sigma_2d_mw  = np.where(mass_grid_mw > 0, mass_grid_mw / pixel_area, np.nan)
    Sigma_2d_m31 = np.where(mass_grid_m31 > 0, mass_grid_m31 / pixel_area, np.nan)

    return {
        "Sigma_2d":     Sigma_2d,
        "Sigma_2d_mw":  Sigma_2d_mw,
        "Sigma_2d_m31": Sigma_2d_m31,
        "extent":       [-lim, lim, -lim, lim],
        "pixel_area":   pixel_area,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 23 — SYSTEM SYNCHRONIZER & SIMULATION LOOP                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*80)
print("  SECTION 23 · Vectorized Profile & Map Aggregator")
print("="*80)

# ── Dynamic Allocation of Output Time-Series Tensors ──────────────────────────
rho_ts        = np.full((ns, nb_sph),  np.nan)    # ρ(r, t)
rho_mw_ts     = np.full((ns, nb_sph),  np.nan)    # ρ_MW(r, t)
rho_m31_ts    = np.full((ns, nb_sph),  np.nan)    # ρ_M31(r, t)
f_mix_ts      = np.full((ns, nb_sph),  np.nan)    # f_mix(r, t)
Gamma_ts      = np.full((ns, nb_sph),  np.nan)    # Γ(r, t)
Sigma_ts      = np.full((ns, nb_proj), np.nan)    # Σ(R, t)
Sigma_mw_ts   = np.full((ns, nb_proj), np.nan)
Sigma_m31_ts  = np.full((ns, nb_proj), np.nan)

# Dynamic allocation of scalar telemetry arrays
rho0_arr        = np.full(ns, np.nan)
r_half_3d_arr   = np.full(ns, np.nan)
r_half_proj_arr = np.full(ns, np.nan)
Gamma_inner_arr = np.full(ns, np.nan)
Gamma_outer_arr = np.full(ns, np.nan)
time_arr        = np.full(ns, np.nan)

# Spatial projection grids (Temporal resolution controlled by STEP_MAPS)
map_snap_nums = SNAPSHOTS[::STEP_MAPS]
n_maps        = len(map_snap_nums)
map_snap_idx  = {snap: idx for idx, snap in enumerate(map_snap_nums)}

maps_3d   = np.full((n_maps, MAP_BINS, MAP_BINS), np.nan)
maps_mw   = np.full((n_maps, MAP_BINS, MAP_BINS), np.nan)
maps_m31  = np.full((n_maps, MAP_BINS, MAP_BINS), np.nan)

# ── Main Processing Pipeline Loop ─────────────────────────────────────────────
t_loop_start = time.perf_counter()

for i, snap_num in enumerate(SNAPSHOTS):
    mw_file  = os.path.join(tmpdir, f"MW_{snap_num:03d}.txt")
    m31_file = os.path.join(tmpdir, f"M31_{snap_num:03d}.txt")

    if not (os.path.isfile(mw_file) and os.path.isfile(m31_file)):
        continue  # Skip missing snapshot files without halting the run

    t_snap = time.perf_counter()

    try:
        snap_data = load_snapshot_particles(mw_file, m31_file)
        prof      = compute_density_profiles(snap_data)
    except Exception as exc:
        print(f"  [ERROR] Processing Failure at Snapshot {snap_num}: {exc}")
        continue

    # ── Synchronize 1D Radial Tensors ──────────────────────────────────────────
    rho_ts      [i, :] = prof["rho"]
    rho_mw_ts   [i, :] = prof["rho_mw"]
    rho_m31_ts  [i, :] = prof["rho_m31"]
    f_mix_ts    [i, :] = prof["f_mix"]
    Gamma_ts    [i, :] = prof["Gamma"]
    Sigma_ts    [i, :] = prof["Sigma"]
    Sigma_mw_ts [i, :] = prof["Sigma_mw"]
    Sigma_m31_ts[i, :] = prof["Sigma_m31"]

    # ── Synchronize Structural Telemetry Scalars ──────────────────────────────
    rho0_arr       [i] = prof["rho0"]
    r_half_3d_arr  [i] = prof["r_half_3d"]
    r_half_proj_arr[i] = prof["r_half_proj"]
    Gamma_inner_arr[i] = prof["Gamma_inner"]
    Gamma_outer_arr[i] = prof["Gamma_outer"]
    time_arr       [i] = prof["time"] if prof["time"] is not None else float(snap_num)

    # ── Synchronize 2D Projected Density Fields (Decoupled Decimation) ────────
    if snap_num in map_snap_idx:
        mi = map_snap_idx[snap_num]
        try:
            maps = compute_2d_map(snap_data)
            maps_3d [mi] = maps["Sigma_2d"]
            maps_mw [mi] = maps["Sigma_2d_mw"]
            maps_m31[mi] = maps["Sigma_2d_m31"]
        except Exception as exc:
            print(f"  [WARNING] Spatial Projection Failed for Snapshot {snap_num}: {exc}")

    dt = time.perf_counter() - t_snap
    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t_loop_start
        print(f"  Snap {snap_num:04d}  ({dt:.2f}s)  |  "
              f"ρ_0 = {prof['rho0']:.2e} M⊙/kpc³  |  "
              f"r_half_3d = {prof['r_half_3d']:.1f} kpc  |  "
              f"[{elapsed:.0f}s elapsed]")

print(f"\n[Execution Completed] Dynamic loop executed in {time.perf_counter() - t_loop_start:.1f}s total.")

# ── Construct Plotting Dimensions ─────────────────────────────────────────────
t_valid     = time_arr[np.isfinite(time_arr)]
time_is_gyr = t_valid.size > 0 and t_valid.min() > 0.05
time_label  = "Time [Gyr]" if time_is_gyr else "Snapshot Index"

# Create a mapping for snapshot indices to their corresponding simulation times
time_maps = np.array([time_arr[np.where(SNAPSHOTS == s)[0][0]]
                      if len(np.where(SNAPSHOTS == s)[0]) > 0 else np.nan
                      for s in map_snap_nums])

MAP_EXTENT = [-MAP_EXTENT_KPC, MAP_EXTENT_KPC, -MAP_EXTENT_KPC, MAP_EXTENT_KPC]
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 24 — DIAGNOSTIC 1: MULTI-EPOCH PROFILE GRID (ρ(r) vs. Σ(R))          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Technical Objective:
# Map radial profile transformations over five distinct dynamical phases. Plotted
# in parallel (3D spherical volume density ρ(r) vs. 2D projected surface density
# Σ(R)), this arrangement displays projection-induced effects side-by-side.
#
# Key Features to Observe:
#   - ρ(r) outer steepening: Progenitors loose outer-shell matter to tidal 
#     fields as high-energy streams strip particles out to massive radii.
#   - Core fluctuations: Core density spikes during first pericentric passage, 
#     subsequently puffing outward via post-collision violent relaxation.
#   - Projections comparisons: Σ(R) profiles decay systematically slower than 
#     true 3D ρ(r) due to line-of-sight columnar integration.
#

print("\n[Fig 1] Plotting 3D spherical ρ(r) and 2D projected Σ(R) radial profiles...")

# Sub-select dual-axes grid layout
fig24, (ax_rho, ax_sigma) = plt.subplots(
    1, 2, figsize=(13, 6), facecolor="#0d0d18",
    gridspec_kw={"wspace": 0.32},
)

for ax in (ax_rho, ax_sigma):
    ax.set_facecolor("#0d0d18")
    ax.set_xscale("log")
    ax.set_yscale("log")

# ── Dynamic Profiles Rendering ────────────────────────────────────────────────
for k_idx, color, label in zip(PROFILE_INDICES, PROFILE_COLORS, PROFILE_LABELS):
    
    # ── 3D Density ρ(r) Profiles ──
    rho_row = rho_ts[k_idx, :]
    valid   = np.isfinite(rho_row) & (rho_row > 0)
    if valid.any():
        ax_rho.plot(r_mid_sph[valid], rho_row[valid],
                    color=color, lw=2.0, label=label)

    # ── 2D Projected Density Σ(R) Profiles ──
    sig_row = Sigma_ts[k_idx, :]
    valid_s = np.isfinite(sig_row) & (sig_row > 0)
    if valid_s.any():
        ax_sigma.plot(r_mid_proj[valid_s], sig_row[valid_s],
                      color=color, lw=2.0, label=label)

# ── Asymptotic Power-Law Slope References ─────────────────────────────────────
# Normalizes relative power-law benchmarks to the intermediate merger profile scale
r_ref  = np.array([1.0, 100.0])
rho_ref_anchor = np.nanmedian(rho_ts[PROFILE_INDICES[2], :][np.isfinite(rho_ts[PROFILE_INDICES[2], :])])

if rho_ref_anchor > 0:
    slope_templates = [
        (-1, ":", r"$\rho \propto r^{-1}$ (NFW Core)"),
        (-2, "--", r"$\rho \propto r^{-2}$ (Isothermal)"),
        (-3, "-.", r"$\rho \propto r^{-3}$ (NFW Envelope)")
    ]
    for slope, ls, lbl in slope_templates:
        rho_ref_line = rho_ref_anchor * (r_ref / 10.0)**slope
        ax_rho.plot(r_ref, rho_ref_line, color="#555577", ls=ls, lw=0.9, label=lbl)

# Axis Layout & Scaling Configuration
ax_rho.set_xlabel("r [kpc]", fontsize=10)
ax_rho.set_ylabel(r"$\rho(r)$  [M$_\odot$ kpc$^{-3}$]", fontsize=10)
ax_rho.set_title(r"3D Radial Density $\rho(r)$", fontsize=11)
ax_rho.legend(fontsize=7, ncol=2)
ax_rho.set_xlim(R_BINS[0], R_BINS[-1])

ax_sigma.set_xlabel("R [kpc]", fontsize=10)
ax_sigma.set_ylabel(r"$\Sigma(R)$  [M$_\odot$ kpc$^{-2}$]", fontsize=10)
ax_sigma.set_title(r"Projected Surface Density $\Sigma(R)$", fontsize=11)
ax_sigma.legend(fontsize=7)
ax_sigma.set_xlim(R_PROJ_BINS[0], R_PROJ_BINS[-1])

fig24.suptitle("Milky Way - M31 merger  ·  Profile Evolution Across Epochs", fontsize=12)
fig24.savefig(os.path.join(OUT_DIR, "density_profiles_grid.png"),
              dpi=300, bbox_inches="tight", facecolor=fig24.get_facecolor())
plt.close(fig24)
print("  Saved: density_profiles_grid.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 25 — DIAGNOSTIC 2: LOGARITHMIC SLOPE Γ(r, t) HEATMAP & MEAN PROFILE ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Technical Objective:
# Map structural transitions of the power-law index Gamma over space and time.
# Gamma(r, t) tracks:
#   - Core-cusp transformation: Softening of central gradients (Gamma -> 0).
#   - Virialization interfaces: Inner vs outer halo boundary transitions.
#   - Severe truncation: Rapid, steep drops in slope indices (Gamma < -4) at 
#     extreme radii tracking tidal debris borders.
#
# Diverging color scheme ("bwr") balances negative steepening (blue) against 
# shallow, non-truncated density regions (red).
#

print("[Fig 2] Rendering Logarithmic Slope Γ(r, t) Evolutionary Heatmap...")

# Saturate noise limits before interpolation mapping
Gamma_plot = np.clip(Gamma_ts, -5.0, 1.0)

t_min, t_max = np.nanmin(time_arr), np.nanmax(time_arr)

fig25, (ax25a, ax25b) = plt.subplots(
    1, 2, figsize=(14, 6), facecolor="#0d0d18",
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.06},
)
for ax in (ax25a, ax25b):
    ax.set_facecolor("#0d0d18")

# ── Dynamic Heatmap Projection (Left Panel) ──
# Corrects for logarithmic binning scales during visual pixel mapping
y_mid_log = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), nb_sph)

# Generate uniform sampling for log-spaced imshow projection
y_uniform = np.logspace(np.log10(R_BINS[0]), np.log10(R_BINS[-1]), 200)
gamma_interp_map = np.zeros((len(y_uniform), ns))

for snap_idx in range(ns):
    non_nan_mask = np.isfinite(Gamma_plot[snap_idx, :])
    if non_nan_mask.sum() > 2:
        gamma_interp_map[:, snap_idx] = np.interp(
            np.log10(y_uniform),
            np.log10(r_mid_sph[non_nan_mask]),
            Gamma_plot[snap_idx, non_nan_mask]
        )
    else:
        gamma_interp_map[:, snap_idx] = np.nan

im25 = ax25a.imshow(
    gamma_interp_map,
    aspect="auto", origin="lower",
    extent=[t_min, t_max, np.log10(R_BINS[0]), np.log10(R_BINS[-1])],
    cmap="bwr", vmin=-4.0, vmax=0.5,
)

# Convert linear pixel coordinates back to physical log markers
ax25a.set_yticks(np.log10(np.array([0.1, 1.0, 10.0, 100.0, 400.0])))
ax25a.set_yticklabels(["0.1", "1.0", "10.0", "100.0", "400.0"])

ax25a.set_xlabel(time_label, fontsize=10)
ax25a.set_ylabel("r [kpc]", fontsize=10)
ax25a.set_title(r"Radial Slope Telemetry: $\Gamma(r,t) = \frac{d\ln\rho}{d\ln r}$", fontsize=11)

cb25 = fig25.colorbar(im25, ax=ax25a, pad=0.01)
cb25.set_label(r"$\Gamma$", fontsize=9)

# Colorbar boundary guides
for val, lbl in [(-1.0, "NFW Cusp"), (-2.0, "Isothermal"), (-3.0, "NFW Outer")]:
    cb25.ax.axhline(val, color="white", lw=0.7, ls="--", alpha=0.6)
    cb25.ax.text(2.5, val, lbl, color="white", fontsize=6, va="center")

# ── Radial Mean Profile (Right Panel) ──
Gamma_mean = np.nanmean(Gamma_ts, axis=0)
valid_G    = np.isfinite(Gamma_mean)

ax25b.plot(Gamma_mean[valid_G], r_mid_sph[valid_G], color="#e8673a", lw=2.0)
ax25b.set_yscale("log")
ax25b.set_xlabel(r"$\langle\Gamma\rangle_t$", fontsize=10)
ax25b.set_ylim(R_BINS[0], R_BINS[-1])
ax25b.set_title("Time-Avg Profile", fontsize=10)
ax25b.tick_params(labelleft=False)

for ref_val in [-1.0, -2.0, -3.0]:
    ax25b.axvline(ref_val, color="#555577", lw=0.7, ls="--", alpha=0.6)

fig25.suptitle("Logarithmic Local Slope Evolution", fontsize=12)
fig25.savefig(os.path.join(OUT_DIR, "density_slope_heatmap.png"),
              dpi=300, bbox_inches="tight", facecolor=fig25.get_facecolor())
plt.close(fig25)
print("  Saved: density_slope_heatmap.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 26 — DIAGNOSTIC 3: CENTRAL CORES & HALF-MASS SCALAR CHRONOLOGY       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Technical Objective:
# Track core compaction and mass distribution profiles over time.
#
# Physical Regimes Tracked:
#   - Core Compaction  : Core density spikes coupled with a decrease in half-mass
#     radii (mass shifts inward).
#   - Virial Expansion : Rapid drop in central density accompanied by half-mass 
#     expansion (violent post-merger thermalization).
#   - Halo Truncation  : Permanent slope steepening in the outer envelopes 
#     (Gamma_outer shifts downward) indicating tidal mass stripping.
#

print("[Fig 3] Plotting central core density and structural parameter trajectories...")

fig26, axes26 = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d0d18",
                             gridspec_kw={"hspace": 0.38, "wspace": 0.32})
axes26 = axes26.flatten()

for ax in axes26:
    ax.set_facecolor("#0d0d18")

# (a) Central Density Evolution
axes26[0].semilogy(time_arr, rho0_arr, color="#e8673a", lw=1.8,
                   label=r"$\rho_0$ (Resolved Core Limit)")
axes26[0].set_ylabel(r"$\rho_0$ [M$_\odot$ kpc$^{-3}$]", fontsize=9)
axes26[0].set_title("Core Density Evolution", fontsize=10)
axes26[0].legend(fontsize=8)

# (b) Continuous Dynamic Half-Mass Radii Time Series
axes26[1].plot(time_arr, r_half_3d_arr,   color="#4a8fff", lw=1.8, label=r"$r_{\rm half,\ 3D}$")
axes26[1].plot(time_arr, r_half_proj_arr, color="#00d4aa", lw=1.8, ls="--", label=r"$R_{\rm half,\ 2D}$")
axes26[1].set_ylabel("Radius [kpc]", fontsize=9)
axes26[1].set_title("Dynamic Half-Mass Radii", fontsize=10)
axes26[1].legend(fontsize=8)

# (c) Core Logarithmic Slope Evolution (r < 5 kpc)
axes26[2].plot(time_arr, Gamma_inner_arr, color="#ff9944", lw=1.8,
                   label=r"$\langle\Gamma\rangle_{r < 5\ {\rm kpc}}$")
axes26[2].axhline(-1, color="#555577", lw=0.8, ls="--")
axes26[2].text(t_min * 1.02, -0.9, "NFW Core (-1)", color="#777799", fontsize=7)
axes26[2].set_ylabel("Mean Inner Slope", fontsize=9)
axes26[2].set_xlabel(time_label, fontsize=9)
axes26[2].set_title("Core Power-Law Evolution", fontsize=10)
axes26[2].legend(fontsize=8)

# (d) Outermost Envelope Slope Evolution (r > 50 kpc)
axes26[3].plot(time_arr, Gamma_outer_arr, color="#aa55ff", lw=1.8,
                   label=r"$\langle\Gamma\rangle_{r > 50\ {\rm kpc}}$")
axes26[3].axhline(-3, color="#555577", lw=0.8, ls="--")
axes26[3].text(t_min * 1.02, -2.9, "NFW Outer (-3)", color="#777799", fontsize=7)
axes26[3].set_ylabel("Mean Outer Slope", fontsize=9)
axes26[3].set_xlabel(time_label, fontsize=9)
axes26[3].set_title("Envelope Truncation Slope", fontsize=10)
axes26[3].legend(fontsize=8)

fig26.suptitle("System Structural Parameter Telemetry", fontsize=12)
fig26.savefig(os.path.join(OUT_DIR, "density_central_evolution.png"),
              dpi=300, bbox_inches="tight", facecolor=fig26.get_facecolor())
plt.close(fig26)
print("  Saved: density_central_evolution.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 27 — DIAGNOSTIC 4: 2D PROJECTIONS GRID (TOTAL, MW & M31 PATHWAYS)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Technical Objective:
# Generate 2D surface density contours approximating real stellar or DM projections.
#
# Highlights:
#   - Grid Layout: Row 1 = Cumulative Surface Density. Row 2 & 3 = Progenitor 
#     distributions (Milky Way vs M31).
#   - Dynamic Range Control: Plotting under strict log10 scales spans the 5+ 
#     decades of density variation from cores to outer streams.
#   - Noise Stabilization: Applying selective spatial smoothing ensures we don't
#     leak unpopulated 'NaN' boundaries inward into physical matter profiles.
#

print("[Fig 4] Plotting 2D log-scaled projection density grids...")

# Select snaps spanning fractions of the merger's duration
map_fracs  = [0.0, 0.2, 0.4, 0.65, 1.0]
map_times  = np.nanmin(time_maps) + np.array(map_fracs) * (np.nanmax(time_maps) - np.nanmin(time_maps))
map_sel_ii = [np.nanargmin(np.abs(time_maps - mt)) for mt in map_times]

fig27 = plt.figure(figsize=(18, 11), facecolor="#0d0d18")
gs27  = gridspec.GridSpec(3, 5, figure=fig27, hspace=0.08, wspace=0.06,
                          left=0.04, right=0.95, top=0.92, bottom=0.05)

row_labels = ["Total Σ", "MW Only", "M31 Only"]
cmaps      = ["inferno", "Blues_r", "Reds_r"]

# Symmetric log scaling limits for the projection maps
vmin_row   = [4.0, 4.0, 4.0]
vmax_row   = [9.5, 9.5, 9.5]

for col, mi in enumerate(map_sel_ii):
    snap_t = time_maps[mi]
    t_str  = f"{snap_t:.2f} Gyr" if time_is_gyr else f"Snap {map_snap_nums[mi]}"

    for row, (data_store, cmap, vmin, vmax) in enumerate(
        zip([maps_3d, maps_mw, maps_m31], cmaps, vmin_row, vmax_row)
    ):
        ax = fig27.add_subplot(gs27[row, col])
        ax.set_facecolor("#0d0d18")

        raw = data_store[mi]
        
        # ── Safe Gaussian Spatial Filtering ──
        # Fill missing values with 0 during kernel processing, then re-apply NaN masking
        is_val   = np.isfinite(raw)
        smoothed = gaussian_filter(np.where(is_val, raw, 0.0), sigma=MAP_SMOOTH_SIGMA)
        smoothed_norm = gaussian_filter(is_val.astype(float), sigma=MAP_SMOOTH_SIGMA)
        
        # Re-normalize smoothed coordinates to prevent zero-boundary leakage
        with np.errstate(invalid="ignore", divide="ignore"):
            cleaned = np.where(smoothed_norm > 0.1, smoothed / smoothed_norm, np.nan)
            log_map = np.where(cleaned > 0, np.log10(cleaned), np.nan)

        ax.imshow(
            log_map.T,
            origin="lower",
            extent=MAP_EXTENT,
            aspect="equal",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
        )

        # Labels, grids and legends
        if row == 0:
            ax.set_title(t_str, fontsize=9, color="#c8c8e8", pad=4)
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=9, color="#c8c8e8")

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Master Right-Hand Boundary Colorbar
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=4.0, vmax=9.5))
sm.set_array([])
cbar_ax = fig27.add_axes([0.96, 0.05, 0.012, 0.87])
fig27.colorbar(sm, cax=cbar_ax, label=r"$\log_{10}(\Sigma\ /\ [{\rm M_\odot\ kpc^{-2}}])$")

fig27.suptitle(r"2D Projected Surface Densities: $\Sigma(x,y)$ (Domain: $\pm$400 kpc)",
               fontsize=13, color="#c8c8e8")
fig27.savefig(os.path.join(OUT_DIR, "density_2d_maps_grid.png"),
              dpi=200, bbox_inches="tight", facecolor=fig27.get_facecolor())
plt.close(fig27)
print("  Saved: density_2d_maps_grid.png")
