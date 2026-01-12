"""
Phase-Space Irreversibility and Stellar Disk Heating
in an Idealised Milky Way–Andromeda Merger

Author
------
Abhinav Vatsa

Description
-----------
Collisionless N-body simulation framework for an idealised MW–M31 merger.
Computes phase-space irreversibility via coarse-grained KL divergence and
stellar disk heating via vertical velocity dispersion.

Assumptions
-----------
- Collisionless dynamics (gravity only)
- No gas physics, star formation, or feedback
- Direct-summation N-body forces with fixed softening
- Non-equilibrium initial conditions, isolated system

Units
-----
- Length: kpc
- Velocity: km/s
- Mass: M_sun
- Time: seconds internally; Gyr for analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# =============================================================================
# NUMERICAL PARAMETERS
# =============================================================================
G = 4.302e-6        # kpc (km/s)^2 M_sun^-1
SOFTENING = 0.5     # kpc
DT = 50e6 * 3.154e7 # 50 Myr in seconds
N_STEPS = 120       # ~6 Gyr total
PHASESPACE_BINS = 24

# =============================================================================
# INITIAL CONDITION GENERATORS
# =============================================================================
def sample_hernquist_halo(n_particles, total_mass, scale_radius, component_id):
    """
    Sample particle positions from a Hernquist density profile.
    Velocities are initially zero (cold halo configuration).
    """
    u = np.random.rand(n_particles)
    r = scale_radius * np.sqrt(u) / (1 - np.sqrt(u))
    costheta = 2*np.random.rand(n_particles) - 1
    phi = 2*np.pi*np.random.rand(n_particles)

    x = r * np.sqrt(1 - costheta**2) * np.cos(phi)
    y = r * np.sqrt(1 - costheta**2) * np.sin(phi)
    z = r * costheta

    vx = np.zeros(n_particles)
    vy = np.zeros(n_particles)
    vz = np.zeros(n_particles)

    mass = np.full(n_particles, total_mass / n_particles)
    ids = np.full(n_particles, component_id)

    return np.column_stack([x, y, z, vx, vy, vz, mass, ids])

def sample_exponential_disk(n_particles, total_mass, scale_length, scale_height, component_id):
    """
    Sample an exponential stellar disk with approximate rotational support.
    """
    R = -scale_length * np.log(1 - np.random.rand(n_particles))
    phi = 2*np.pi*np.random.rand(n_particles)
    z = scale_height * np.random.randn(n_particles)

    x = R * np.cos(phi)
    y = R * np.sin(phi)

    vphi = np.sqrt(G * total_mass / (R + 1e-3))
    vx = -vphi * np.sin(phi)
    vy =  vphi * np.cos(phi)
    vz = np.zeros(n_particles)

    mass = np.full(n_particles, total_mass / n_particles)
    ids = np.full(n_particles, component_id)

    return np.column_stack([x, y, z, vx, vy, vz, mass, ids])

# =============================================================================
# N-BODY DYNAMICS
# =============================================================================
def compute_accelerations(positions, masses):
    """
    Compute gravitational accelerations with softened Newtonian gravity.
    """
    N = len(positions)
    acc = np.zeros_like(positions)

    for i in range(N):
        r = positions[i] - positions
        dist = np.linalg.norm(r, axis=1) + SOFTENING
        acc[i] -= G * np.sum((masses[:, None] * r) / dist[:, None]**3, axis=0)

    return acc

def leapfrog_step(state, dt):
    """
    Advance particle state by one leapfrog timestep.
    """
    pos = state[:, :3]
    vel = state[:, 3:6]
    mass = state[:, 6]

    vel_half = vel + 0.5 * dt * compute_accelerations(pos, mass)
    pos_new = pos + dt * vel_half
    vel_new = vel_half + 0.5 * dt * compute_accelerations(pos_new, mass)

    state[:, :3] = pos_new
    state[:, 3:6] = vel_new
    return state

# =============================================================================
# PHASE-SPACE ANALYSIS
# =============================================================================
def coarse_grained_phase_space(state, n_bins=PHASESPACE_BINS):
    """
    Construct a coarse-grained 6D phase-space density histogram.
    """
    pos = state[:, :3]
    vel = state[:, 3:6]
    H, _ = np.histogramdd(np.hstack([pos, vel]), bins=n_bins)
    return H / np.sum(H)

def kl_divergence(P, Q):
    """
    Compute Kullback–Leibler divergence D_KL(P || Q).
    """
    mask = (P > 0) & (Q > 0)
    return entropy(P[mask], Q[mask])

# =============================================================================
# PHYSICAL DIAGNOSTICS
# =============================================================================
def vertical_velocity_dispersion(state):
    """
    Return variance of vertical velocities for disk particles.
    """
    return np.var(state[:, 5])

def z_angular_momentum(state):
    """
    Return z-component of angular momentum for each particle.
    """
    x, y = state[:, 0], state[:, 1]
    vx, vy = state[:, 3], state[:, 4]
    return x * vy - y * vx

# =============================================================================
# SIMULATION DRIVER
# =============================================================================
def run_merger_simulation():
    """
    Initialize galaxies and evolve MW–M31 merger.
    Returns a list of snapshots.
    """
    # Milky Way
    mw_disk = sample_exponential_disk(600, 6e10, 3.0, 0.3, component_id=0)
    mw_halo = sample_hernquist_halo(600, 1e12, 30.0, component_id=1)

    # Andromeda
    m31_disk = sample_exponential_disk(600, 8e10, 5.0, 0.4, component_id=2)
    m31_halo = sample_hernquist_halo(600, 1.5e12, 40.0, component_id=3)

    # Set initial separation and bulk velocity
    m31_disk[:, 0] += 780
    m31_halo[:, 0] += 780
    m31_disk[:, 3] -= 120
    m31_halo[:, 3] -= 120

    state = np.vstack([mw_disk, mw_halo, m31_disk, m31_halo])

    snapshots = []
    for step in range(N_STEPS):
        state = leapfrog_step(state, DT)
        snapshots.append(state.copy())
    return snapshots

# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================
def analyse_irreversibility_and_heating(snapshots):
    """
    Compute disk/halo KL divergence and disk vertical heating.
    """
    disk_kl = []
    halo_kl = []
    disk_heating = []

    for i in range(1, len(snapshots)//2):
        fwd = snapshots[i]
        bwd = snapshots[-i]

        disk_fwd = fwd[(fwd[:, 7] == 0) | (fwd[:, 7] == 2)]
        disk_bwd = bwd[(bwd[:, 7] == 0) | (bwd[:, 7] == 2)]
        halo_fwd = fwd[(fwd[:, 7] == 1) | (fwd[:, 7] == 3)]
        halo_bwd = bwd[(bwd[:, 7] == 1) | (bwd[:, 7] == 3)]

        disk_kl.append(kl_divergence(coarse_grained_phase_space(disk_fwd),
                                     coarse_grained_phase_space(disk_bwd)))
        halo_kl.append(kl_divergence(coarse_grained_phase_space(halo_fwd),
                                     coarse_grained_phase_space(halo_bwd)))
        disk_heating.append(vertical_velocity_dispersion(disk_fwd))

    return disk_kl, halo_kl, disk_heating

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    snapshots = run_merger_simulation()
    disk_kl, halo_kl, disk_heating = analyse_irreversibility_and_heating(snapshots)

    time_gyr = np.arange(len(disk_kl)) * DT / 3.154e16

    # Plot KL divergence
    plt.figure(figsize=(10,6))
    plt.plot(time_gyr, disk_kl, label="Disk KL Divergence")
    plt.plot(time_gyr, halo_kl, label="Halo KL Divergence")
    plt.xlabel("Time [Gyr]")
    plt.ylabel("KL Divergence")
    plt.title("MW–M31 Merger: Phase-Space Irreversibility")
    plt.legend()
    plt.show()

    # Plot disk vertical heating
    plt.figure(figsize=(10,6))
    plt.plot(time_gyr, disk_heating)
    plt.xlabel("Time [Gyr]")
    plt.ylabel(r"$\sigma_z^2$")
    plt.title("Disk Vertical Heating During Merger")
    plt.show()
