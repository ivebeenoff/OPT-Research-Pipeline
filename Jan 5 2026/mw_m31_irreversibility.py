"""
Phase-Space Irreversibility and Stellar Disk Heating in a Milky Way–M31 Merger Simulation
================================

Author
------
Abhinav Vatsa

Purpose
-------
End-to-end computational astrophysics pipeline developed for Optional Practical
Training (OPT). This script simulates the Milky Way–Andromeda (M31) merger using an
approximate collisionless N-body framework and applies physically motivated,
statistical diagnostics to study dynamical irreversibility and disk heating.

Scientific Focus
----------------
- Galaxy–galaxy interaction dynamics
- Phase-space evolution
- Time-reversal asymmetry (irreversibility)
- Stellar disk heating vs dark matter halo response

Core Assumptions
----------------
- Collisionless dynamics (gravity only)
- No gas physics, star formation, or feedback
- Approximate direct-summation N-body forces
- Educational + research-grade fidelity (not production cosmological code)

Units
-----
- Length: kiloparsecs (kpc)
- Velocity: km/s
- Mass: solar masses (Msun)
- Time: seconds internally, converted to Gyr for plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

G = 4.302e-6
"""Gravitational constant in units of kpc (km/s)^2 Msun^-1."""

EPS = 0.5
"""Plummer-equivalent gravitational softening length in kpc."""

DT = 50e6 * 3.154e7
"""Simulation timestep: 50 Myr expressed in seconds."""

NSTEPS = 120
"""Total number of integration steps (~6 Gyr total evolution)."""

BINS = 24
"""Number of bins per dimension for phase-space coarse graining."""


# =============================================================================
# INITIAL CONDITION GENERATORS
# =============================================================================

def hernquist_sphere(N, M, a, tag):
    """
    Generate a Hernquist-profile spherical distribution.

    Parameters
    ----------
    N : int
        Number of particles.
    M : float
        Total mass of the halo (Msun).
    a : float
        Scale radius of the Hernquist profile (kpc).
    tag : int
        Integer identifier for particle type (used for later analysis).

    Returns
    -------
    ndarray
        Array of shape (N, 8) containing:
        [x, y, z, vx, vy, vz, mass, tag]

    Notes
    -----
    - Velocities are initialized to zero (cold halo approximation).
    - The system heats dynamically during the merger.
    """
    u = np.random.rand(N)
    r = a * np.sqrt(u) / (1 - np.sqrt(u))
    costheta = 2*np.random.rand(N) - 1
    phi = 2*np.pi*np.random.rand(N)

    x = r * np.sqrt(1 - costheta**2) * np.cos(phi)
    y = r * np.sqrt(1 - costheta**2) * np.sin(phi)
    z = r * costheta

    vx = np.zeros(N)
    vy = np.zeros(N)
    vz = np.zeros(N)

    m = np.full(N, M / N)
    tags = np.full(N, tag)

    return np.column_stack([x, y, z, vx, vy, vz, m, tags])


def exponential_disk(N, M, Rd, z0, tag):
    """
    Generate an exponential stellar disk with rotational support.

    Parameters
    ----------
    N : int
        Number of disk particles.
    M : float
        Total disk mass (Msun).
    Rd : float
        Radial scale length (kpc).
    z0 : float
        Vertical scale height (kpc).
    tag : int
        Integer identifier for particle type.

    Returns
    -------
    ndarray
        Array of shape (N, 8) containing:
        [x, y, z, vx, vy, vz, mass, tag]

    Notes
    -----
    - Circular velocities are approximated assuming a dominant central mass.
    - This is sufficient for qualitative disk heating studies.
    """
    R = -Rd * np.log(1 - np.random.rand(N))
    phi = 2*np.pi*np.random.rand(N)
    z = z0 * np.random.randn(N)

    x = R * np.cos(phi)
    y = R * np.sin(phi)

    vphi = np.sqrt(G * M / (R + 1e-3))
    vx = -vphi * np.sin(phi)
    vy =  vphi * np.cos(phi)
    vz = np.zeros(N)

    m = np.full(N, M / N)
    tags = np.full(N, tag)

    return np.column_stack([x, y, z, vx, vy, vz, m, tags])


# =============================================================================
# N-BODY DYNAMICS
# =============================================================================

def acceleration(pos, mass):
    """
    Compute gravitational acceleration using softened Newtonian gravity.

    Parameters
    ----------
    pos : ndarray
        Particle positions of shape (N, 3).
    mass : ndarray
        Particle masses of shape (N,).

    Returns
    -------
    ndarray
        Accelerations of shape (N, 3).

    Notes
    -----
    - Direct O(N^2) summation (not optimized).
    - Softening prevents numerical divergences at small separations.
    """
    N = len(pos)
    acc = np.zeros_like(pos)

    for i in range(N):
        r = pos[i] - pos
        dist = np.linalg.norm(r, axis=1) + EPS
        acc[i] -= G * np.sum((mass[:, None] * r) / dist[:, None]**3, axis=0)

    return acc


def leapfrog(state, dt):
    """
    Advance the system by one timestep using leapfrog integration.

    Parameters
    ----------
    state : ndarray
        Particle state array of shape (N, 8).
    dt : float
        Timestep in seconds.

    Returns
    -------
    ndarray
        Updated particle state.
    """
    pos = state[:, :3]
    vel = state[:, 3:6]
    mass = state[:, 6]

    vel_half = vel + 0.5 * dt * acceleration(pos, mass)
    pos_new = pos + dt * vel_half
    vel_new = vel_half + 0.5 * dt * acceleration(pos_new, mass)

    state[:, :3] = pos_new
    state[:, 3:6] = vel_new
    return state


# =============================================================================
# PHASE-SPACE ANALYSIS
# =============================================================================

def coarse_grain(state, bins=BINS):
    """
    Construct a coarse-grained 6D phase-space distribution.

    Parameters
    ----------
    state : ndarray
        Particle state array.
    bins : int
        Number of histogram bins per dimension.

    Returns
    -------
    ndarray
        Normalized phase-space density histogram.
    """
    pos = state[:, :3]
    vel = state[:, 3:6]

    H, _ = np.histogramdd(np.hstack([pos, vel]), bins=bins)
    return H / np.sum(H)


def kl_div(P, Q):
    """
    Compute Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    P, Q : ndarray
        Normalized probability distributions.

    Returns
    -------
    float
        KL divergence D_KL(P || Q).

    Interpretation
    --------------
    Measures time-reversal asymmetry when applied to forward vs backward snapshots.
    """
    mask = (P > 0) & (Q > 0)
    return entropy(P[mask], Q[mask])


# =============================================================================
# PHYSICAL DIAGNOSTICS
# =============================================================================

def vertical_heating(state):
    """
    Measure disk vertical heating via velocity dispersion.

    Parameters
    ----------
    state : ndarray
        Disk particle state array.

    Returns
    -------
    float
        Variance of vertical velocity σ_z².
    """
    return np.var(state[:, 5])


def angular_momentum(state):
    """
    Compute z-component of angular momentum.

    Parameters
    ----------
    state : ndarray
        Particle state array.

    Returns
    -------
    ndarray
        Angular momentum values for each particle.
    """
    x, y = state[:, 0], state[:, 1]
    vx, vy = state[:, 3], state[:, 4]
    return x * vy - y * vx


# =============================================================================
# MAIN SIMULATION DRIVER
# =============================================================================

def run_simulation():
    """
    Run the full MW–M31 merger simulation.

    Returns
    -------
    list of ndarray
        Time-ordered snapshots of the full system state.
    """
    # Milky Way
    mw_disk = exponential_disk(600, 6e10, 3.0, 0.3, tag=0)
    mw_halo = hernquist_sphere(600, 1e12, 30.0, tag=1)

    # Andromeda (M31)
    m31_disk = exponential_disk(600, 8e10, 5.0, 0.4, tag=2)
    m31_halo = hernquist_sphere(600, 1.5e12, 40.0, tag=3)

    # Initial separation and approach velocity
    m31_disk[:, 0] += 780
    m31_halo[:, 0] += 780
    m31_disk[:, 3] -= 120
    m31_halo[:, 3] -= 120

    state = np.vstack([mw_disk, mw_halo, m31_disk, m31_halo])

    snapshots = []
    for step in range(NSTEPS):
        state = leapfrog(state, DT)
        snapshots.append(state.copy())
        if step % 10 == 0:
            print(f"Step {step}/{NSTEPS}")

    return snapshots


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def analyze(snapshots):
    """
    Analyze merger irreversibility and disk heating.

    Parameters
    ----------
    snapshots : list of ndarray
        Time-ordered system snapshots.

    Returns
    -------
    tuple of lists
        (disk irreversibility, halo irreversibility, disk heating)
    """
    irreversibility_disk = []
    irreversibility_halo = []
    heating_disk = []

    for i in range(1, len(snapshots) // 2):
        fwd = snapshots[i]
        bwd = snapshots[-i]

        disk_fwd = fwd[(fwd[:, 7] == 0) | (fwd[:, 7] == 2)]
        disk_bwd = bwd[(bwd[:, 7] == 0) | (bwd[:, 7] == 2)]
        halo_fwd = fwd[(fwd[:, 7] == 1) | (fwd[:, 7] == 3)]
        halo_bwd = bwd[(bwd[:, 7] == 1) | (bwd[:, 7] == 3)]

        irreversibility_disk.append(kl_div(coarse_grain(disk_fwd),
                                           coarse_grain(disk_bwd)))
        irreversibility_halo.append(kl_div(coarse_grain(halo_fwd),
                                           coarse_grain(halo_bwd)))
        heating_disk.append(vertical_heating(disk_fwd))

    return irreversibility_disk, irreversibility_halo, heating_disk


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    snapshots = run_simulation()
    irr_d, irr_h, heat = analyze(snapshots)

    t = np.arange(len(irr_d)) * DT / 3.154e16  # Time in Gyr

    plt.figure(figsize=(10, 6))
    plt.plot(t, irr_d, label="Disk irreversibility")
    plt.plot(t, irr_h, label="Halo irreversibility")
    plt.xlabel("Time [Gyr]")
    plt.ylabel("KL Divergence")
    plt.title("MW–M31 Merger: Onset of Dynamical Irreversibility")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(t, heat)
    plt.xlabel("Time [Gyr]")
    plt.ylabel("σ_z²")
    plt.title("Disk Vertical Heating During Merger")
    plt.show()
