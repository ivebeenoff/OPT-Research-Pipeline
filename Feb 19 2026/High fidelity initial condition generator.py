"""
MW–M31 HIGH-FIDELITY INITIAL CONDITION GENERATOR
Author: Abhinav Vatsa
Environment: Python + NumPy (Spyder / Jupyter friendly)

============================================================
PHYSICAL MODEL
============================================================

This module generates dynamically stable galaxy halos suitable
for high-fidelity N-body simulations of the Local Group.

Density profile:
    Navarro–Frenk–White (NFW)

Velocity distribution:
    Isotropic Jeans equilibrium

Geometry:
    Spherical halo truncated at rmax

WHY NFW?
---------
ΛCDM cosmological simulations show that virialized dark-matter
halos follow the NFW profile over a wide mass range.

Toy Gaussian halos are NOT in equilibrium and artificially
expand before merger simulations begin.

============================================================
NUMERICAL STRATEGY
============================================================

1. Sample radii using inverse transform sampling of M(<r).
2. Sample isotropic angular distribution.
3. Compute σ(r) from Jeans equation approximation.
4. Sample Maxwellian velocities.
5. Optional virial rescaling.

All routines are fully vectorized NumPy for speed,
reproducibility, and compatibility with computational
physics workflows.

============================================================
UNITS
============================================================
Distance: kpc
Mass: Msun
Velocity: km/s
G = 4.30091e-6  (kpc / Msun) (km/s)^2
"""

import numpy as np

G = 4.30091e-6


# ============================================================
# 1. NFW PROFILE UTILITIES
# ============================================================

def nfw_cumulative_mass(r, rs):
    """
    Dimensionless enclosed mass for NFW profile.

    M(<r) ∝ ln(1+x) - x/(1+x)
    where x = r / rs

    Used for inverse transform sampling.

    Complexity: O(N)
    """
    x = r / rs
    return np.log(1 + x) - x / (1 + x)


# ============================================================
# 2. POSITION SAMPLING
# ============================================================

def sample_nfw_positions(N, rs, rmax, seed=None):
    """
    Sample particle positions from truncated NFW halo.

    METHOD
    ------
    • Build radius grid
    • Compute cumulative mass
    • Invert using interpolation
    • Sample isotropic angles

    Complexity: O(N)
    Memory: O(N)
    """

    if seed is not None:
        np.random.seed(seed)

    # Radius grid for inverse transform
    r_grid = np.linspace(1e-5, rmax, 20000)
    Menc = nfw_cumulative_mass(r_grid, rs)
    Menc /= Menc[-1]

    u = np.random.rand(N)
    r = np.interp(u, Menc, r_grid)

    # Isotropic angles
    phi = 2*np.pi*np.random.rand(N)
    cos_t = 2*np.random.rand(N) - 1
    sin_t = np.sqrt(1 - cos_t**2)

    pos = np.column_stack([
        r * sin_t * np.cos(phi),
        r * sin_t * np.sin(phi),
        r * cos_t
    ])

    return pos


# ============================================================
# 3. JEANS VELOCITY DISPERSION
# ============================================================

def nfw_sigma(r, M, rs):
    """
    Approximate isotropic velocity dispersion.

    Derived from Jeans equation:

        d(ρσ²)/dr = -ρ GM(<r)/r²

    We approximate σ² ≈ GM(<r)/(2r).

    This is sufficient for equilibrium halos
    in first-order galaxy simulations.

    Complexity: O(N)
    """

    Menc = nfw_cumulative_mass(r, rs)
    Menc *= M / Menc.max()

    sigma2 = G * Menc / (2*r + 1e-5)
    return np.sqrt(sigma2)


def sample_nfw_velocities(pos, M, rs):
    """
    Sample Maxwellian velocities consistent with σ(r).

    Assumes isotropic velocity ellipsoid.
    """

    r = np.linalg.norm(pos, axis=1)
    sigma = nfw_sigma(r, M, rs)

    vel = np.random.normal(0, sigma[:, None], size=(len(pos), 3))
    return vel


# ============================================================
# 4. VIRIAL RESCALING (OPTIONAL)
# ============================================================

def virial_rescale(pos, vel, compute_energy):
    """
    Rescale velocities so 2K/|U| = 1.

    Requires external potential-energy routine.
    Useful for stabilizing halos in real simulations.
    """

    KE = 0.5 * np.sum(np.sum(vel**2, axis=1))
    E = compute_energy(pos, vel)
    PE = E - KE

    Q = 2 * KE / abs(PE)
    scale = np.sqrt(1 / Q)

    return vel * scale


# ============================================================
# 5. MILKY WAY HALO
# ============================================================

def generate_mw_halo(N=100000, seed=None):
    """
    Milky Way halo parameters (typical literature values):

        Mhalo ≈ 1.1e12 Msun
        rs    ≈ 20 kpc
        rmax  ≈ 250 kpc
    """

    M = 1.1e12
    rs = 20
    rmax = 250

    pos = sample_nfw_positions(N, rs, rmax, seed)
    vel = sample_nfw_velocities(pos, M, rs)

    return pos, vel


# ============================================================
# 6. ANDROMEDA HALO
# ============================================================

def generate_m31_halo(N=100000, seed=None):
    """
    Andromeda halo parameters.

        Mhalo ≈ 1.5e12 Msun
        rs    ≈ 25 kpc
        rmax  ≈ 300 kpc
    """

    M = 1.5e12
    rs = 25
    rmax = 300

    pos = sample_nfw_positions(N, rs, rmax, seed)
    vel = sample_nfw_velocities(pos, M, rs)

    return pos, vel


# ============================================================
# 7. MW–M31 LOCAL GROUP INITIAL CONDITIONS
# ============================================================

def generate_local_group(Nmw=100000, Nm31=100000):
    """
    Construct MW + M31 system.

    Observational constraints:
        Distance ≈ 780 kpc
        Relative radial velocity ≈ −110 km/s
    """

    pos_mw, vel_mw = generate_mw_halo(Nmw, seed=1)
    pos_m31, vel_m31 = generate_m31_halo(Nm31, seed=2)

    pos_m31[:, 0] += 780
    vel_m31[:, 0] -= 110

    pos = np.vstack([pos_mw, pos_m31])
    vel = np.vstack([vel_mw, vel_m31])

    return pos, vel


# ============================================================
# 8. VALIDATION ROUTINES
# ============================================================

def density_profile(pos, bins=60):
    """
    Compute radial density profile for validation.

    Compare against analytic NFW curve.
    """

    r = np.linalg.norm(pos, axis=1)
    hist, edges = np.histogram(r, bins=bins)

    rmid = 0.5 * (edges[1:] + edges[:-1])
    vol = 4/3 * np.pi * (edges[1:]**3 - edges[:-1]**3)

    rho = hist / vol
    return rmid, rho


# ============================================================
# 9. TEST DRIVER
# ============================================================

def run_test():
    """
    Quick sanity test.
    """

    pos, vel = generate_mw_halo(50000)
    r, rho = density_profile(pos)

    print("Generated halo with", len(pos), "particles")
    print("Median radius:", np.median(np.linalg.norm(pos, axis=1)))
    print("Mean speed:", np.mean(np.linalg.norm(vel, axis=1)))


if __name__ == "__main__":
    run_test()
