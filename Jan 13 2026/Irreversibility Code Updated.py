"""
Phase-Space Irreversibility and Stellar Disk Heating
in an Idealised Milky Way–Andromeda Merger

Author
------
Abhinav Vatsa

Abstract
--------
This module implements a controlled, collisionless N-body experiment designed
to probe the emergence of macroscopic irreversibility in galaxy mergers.
Irreversibility is quantified using coarse-grained phase-space divergence
metrics, while stellar disk heating is measured via kinematic thickening
diagnostics.

The code is intentionally structured for clarity, extensibility, and physical
interpretability rather than raw computational performance.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable

# =============================================================================
# GLOBAL CONSTANTS AND UNITS
# =============================================================================

G = 4.302e-6        # kpc (km/s)^2 M_sun^-1
SEC_PER_GYR = 3.154e16

# =============================================================================
# NUMERICAL CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """
    Container for all numerical and discretization parameters.

    Notes
    -----
    Centralizing numerical parameters allows controlled exploration of
    numerical irreversibility arising from discretization, force softening,
    and coarse-graining.
    """
    softening_length: float = 0.5        # kpc
    timestep: float = 50e6 * 3.154e7      # seconds
    n_steps: int = 120
    phasespace_bins: int = 24
    snapshot_stride: int = 1
    enable_energy_tracking: bool = True
    enable_angular_momentum_tracking: bool = True

# =============================================================================
# GALAXY COMPONENT ABSTRACTIONS
# =============================================================================

@dataclass
class GalaxyComponent:
    """
    Abstract representation of a gravitationally bound stellar component.

    This class does not enforce equilibrium or self-consistency; it merely
    provides structured bookkeeping for component-level diagnostics.
    """
    name: str
    component_id: int
    total_mass: float
    particle_count: int
    scale_length: float
    scale_height: float = 0.0
    component_type: str = "collisionless"

# =============================================================================
# INITIAL CONDITION GENERATORS
# =============================================================================

class InitialConditionFactory:
    """
    Factory class for generating idealized galaxy components.

    Design choice:
    --------------
    Sampling routines are intentionally explicit and verbose to make
    assumptions transparent rather than implicit.
    """

    @staticmethod
    def hernquist_halo(component: GalaxyComponent) -> np.ndarray:
        """
        Generate a Hernquist-profile dark matter halo.

        Velocities are initialized to zero, representing an extremely cold,
        non-equilibrium configuration intended to maximize phase mixing.
        """
        n = component.particle_count
        a = component.scale_length
        M = component.total_mass

        u = np.random.rand(n)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))
        costheta = 2*np.random.rand(n) - 1
        phi = 2*np.pi*np.random.rand(n)

        x = r*np.sqrt(1-costheta**2)*np.cos(phi)
        y = r*np.sqrt(1-costheta**2)*np.sin(phi)
        z = r*costheta

        state = np.zeros((n, 8))
        state[:, :3] = np.column_stack([x, y, z])
        state[:, 6] = M / n
        state[:, 7] = component.component_id

        return state

    @staticmethod
    def exponential_disk(component: GalaxyComponent) -> np.ndarray:
        """
        Generate an exponential stellar disk with approximate centrifugal support.

        Important:
        ----------
        This disk is not in detailed equilibrium. Deviations from equilibrium
        contribute intentionally to secular heating and phase-space diffusion.
        """
        n = component.particle_count
        Rd = component.scale_length
        hz = component.scale_height
        M = component.total_mass

        R = -Rd * np.log(1 - np.random.rand(n))
        phi = 2*np.pi*np.random.rand(n)
        z = hz * np.random.randn(n)

        x = R * np.cos(phi)
        y = R * np.sin(phi)

        vphi = np.sqrt(G * M / (R + 1e-2))
        vx = -vphi * np.sin(phi)
        vy =  vphi * np.cos(phi)

        state = np.zeros((n, 8))
        state[:, :3] = np.column_stack([x, y, z])
        state[:, 3:6] = np.column_stack([vx, vy, np.zeros(n)])
        state[:, 6] = M / n
        state[:, 7] = component.component_id

        return state

# =============================================================================
# FORCE CALCULATION
# =============================================================================

class GravitySolver:
    """
    Direct-summation Newtonian gravity with Plummer softening.

    This solver is intentionally naive to preserve exact time reversibility
    at the algorithmic level.
    """

    def __init__(self, softening: float):
        self.softening = softening

    def accelerations(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        N = len(positions)
        acc = np.zeros_like(positions)

        for i in range(N):
            dr = positions[i] - positions
            r2 = np.sum(dr**2, axis=1) + self.softening**2
            inv_r3 = r2**(-1.5)
            acc[i] -= G * np.sum(masses[:, None] * dr * inv_r3[:, None], axis=0)

        return acc

# =============================================================================
# TIME INTEGRATION
# =============================================================================

class LeapfrogIntegrator:
    """
    Kick-drift-kick leapfrog integrator.

    Notes
    -----
    Symplecticity ensures formal reversibility, allowing irreversibility
    to be attributed to coarse-graining and numerical truncation.
    """

    def __init__(self, solver: GravitySolver):
        self.solver = solver

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        pos = state[:, :3]
        vel = state[:, 3:6]
        m   = state[:, 6]

        a0 = self.solver.accelerations(pos, m)
        vel_half = vel + 0.5 * dt * a0
        pos_new = pos + dt * vel_half
        a1 = self.solver.accelerations(pos_new, m)
        vel_new = vel_half + 0.5 * dt * a1

        state[:, :3] = pos_new
        state[:, 3:6] = vel_new
        return state

# =============================================================================
# PHASE-SPACE AND IRREVERSIBILITY METRICS
# =============================================================================

class PhaseSpaceMetrics:
    """
    Collection of coarse-grained phase-space diagnostics.
    """

    def __init__(self, bins: int):
        self.bins = bins

    def density(self, state: np.ndarray) -> np.ndarray:
        X = np.hstack([state[:, :3], state[:, 3:6]])
        H, _ = np.histogramdd(X, bins=self.bins)
        return H / np.sum(H)

    def kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        mask = (P > 0) & (Q > 0)
        return entropy(P[mask], Q[mask])

# =============================================================================
# DISK HEATING AND DYNAMICS
# =============================================================================

class DiskDiagnostics:
    """
    Stellar disk heating and angular momentum diagnostics.
    """

    @staticmethod
    def vertical_velocity_dispersion(state: np.ndarray) -> float:
        return np.var(state[:, 5])

    @staticmethod
    def angular_momentum_z(state: np.ndarray) -> np.ndarray:
        x, y = state[:, 0], state[:, 1]
        vx, vy = state[:, 3], state[:, 4]
        return x * vy - y * vx

# =============================================================================
# ENERGY BOOKKEEPING
# =============================================================================

class ConservedQuantities:
    """
    Tracks global conserved quantities for numerical validation.
    """

    @staticmethod
    def total_energy(state: np.ndarray, softening: float) -> float:
        pos = state[:, :3]
        vel = state[:, 3:6]
        m = state[:, 6]

        KE = 0.5 * np.sum(m * np.sum(vel**2, axis=1))
        PE = 0.0
        for i in range(len(pos)):
            r = np.linalg.norm(pos[i] - pos, axis=1) + softening
            PE -= G * np.sum(m[i] * m / r)

        return KE + 0.5 * PE

# =============================================================================
# MERGER ORCHESTRATION
# =============================================================================

class MergerSimulation:
    """
    High-level controller for MW–M31 merger experiments.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.gravity = GravitySolver(config.softening_length)
        self.integrator = LeapfrogIntegrator(self.gravity)
        self.metrics = PhaseSpaceMetrics(config.phasespace_bins)

    def initialize(self) -> np.ndarray:
        """
        Assemble all galaxy components into a single phase-space state vector.
        """
        mw_disk = GalaxyComponent("MW Disk", 0, 6e10, 600, 3.0, 0.3)
        mw_halo = GalaxyComponent("MW Halo", 1, 1e12, 600, 30.0)

        m31_disk = GalaxyComponent("M31 Disk", 2, 8e10, 600, 5.0, 0.4)
        m31_halo = GalaxyComponent("M31 Halo", 3, 1.5e12, 600, 40.0)

        state = np.vstack([
            InitialConditionFactory.exponential_disk(mw_disk),
            InitialConditionFactory.hernquist_halo(mw_halo),
            InitialConditionFactory.exponential_disk(m31_disk),
            InitialConditionFactory.hernquist_halo(m31_halo),
        ])

        state[state[:,7] >= 2, 0] += 780
        state[state[:,7] >= 2, 3] -= 120

        return state

    def run(self) -> List[np.ndarray]:
        """
        Execute forward integration and return stored snapshots.
        """
        state = self.initialize()
        snapshots = []

        for step in range(self.config.n_steps):
            state = self.integrator.step(state, self.config.timestep)
            if step % self.config.snapshot_stride == 0:
                snapshots.append(state.copy())

        return snapshots

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    config = SimulationConfig()
    sim = MergerSimulation(config)
    snapshots = sim.run()

    # Analysis intentionally deferred to post-processing notebooks
