"""
HST-Aligned Milky Way – Andromeda Merger Simulation Pipeline

Author: Abhinav Vatsa
Purpose: Study phase-space irreversibility, disk heating, 
         and angular momentum evolution in a controlled, 
         HST-constrained MW–M31 merger setup.

Key Features:
-------------
1. HST-aligned initial center-of-mass positions & velocities
2. Disk + halo galaxies initialized with realistic structure
3. In-memory trajectory and snapshot storage
4. Diagnostics: entropy, disk heating, angular momentum, energy
5. Optional snapshot export for visualization or analysis
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from dataclasses import dataclass
from typing import List

# Import COM module for MW/M31 tracking (Customizable for other COM modules, only the filename needs to be changed)
from CenterOfMass2 import CenterOfMass2

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
G = 4.302e-6        # Gravitational constant in kpc (km/s)^2 / Msun
SEC_PER_GYR = 3.154e16  # Seconds per Gyr

# =============================================================================
# NUMERICAL CONFIGURATION
# =============================================================================
@dataclass
class SimulationConfig:
    softening_length: float = 0.5               # kpc
    timestep: float = 50e6 * 3.154e7           # seconds
    n_steps: int = 120                          # number of integration steps
    phasespace_bins: int = 24                   # for entropy computation
    snapshot_stride: int = 1                    # store every N steps
    enable_energy_tracking: bool = True
    enable_angular_momentum_tracking: bool = True
    save_snapshots: bool = False                # toggle file output

# =============================================================================
# GALAXY COMPONENT ABSTRACTIONS
# =============================================================================
@dataclass
class GalaxyComponent:
    name: str
    component_id: int
    total_mass: float
    particle_count: int
    scale_length: float
    scale_height: float = 0.0
    component_type: str = "collisionless"  # disk or halo

# =============================================================================
# INITIAL CONDITIONS FACTORY
# =============================================================================
class InitialConditionFactory:

    @staticmethod
    def hernquist_halo(component: GalaxyComponent) -> np.ndarray:
        """Generate a Hernquist halo with placeholder zero velocities (to be improved)."""
        n, a, M = component.particle_count, component.scale_length, component.total_mass

        # Spherically random positions (Hernquist cumulative distribution)
        u = np.random.rand(n)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))
        costheta = 2*np.random.rand(n) - 1
        phi = 2*np.pi*np.random.rand(n)

        x = r*np.sqrt(1-costheta**2)*np.cos(phi)
        y = r*np.sqrt(1-costheta**2)*np.sin(phi)
        z = r*costheta

        # State array: x,y,z,vx,vy,vz,mass,component_id
        state = np.zeros((n, 8))
        state[:, :3] = np.column_stack([x, y, z])
        state[:, 6] = M / n
        state[:, 7] = component.component_id

        # TODO: Replace zero velocities with equilibrium-sampled velocities
        return state

    @staticmethod
    def exponential_disk(component: GalaxyComponent) -> np.ndarray:
        """Generate an exponential disk in cylindrical coordinates."""
        n, Rd, hz, M = component.particle_count, component.scale_length, component.scale_height, component.total_mass

        R = -Rd * np.log(1 - np.random.rand(n))
        phi = 2*np.pi*np.random.rand(n)
        z = hz * np.random.randn(n)

        x = R * np.cos(phi)
        y = R * np.sin(phi)

        # Circular velocity (approximate)
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
# GRAVITY SOLVER
# =============================================================================
class GravitySolver:
    def __init__(self, softening: float):
        self.softening = softening

    def accelerations(self, pos: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Compute pairwise Newtonian accelerations with softening."""
        N = len(pos)
        acc = np.zeros_like(pos)
        for i in range(N):
            dr = pos[i] - pos
            r2 = np.sum(dr**2, axis=1) + self.softening**2
            inv_r3 = r2**(-1.5)
            acc[i] -= G * np.sum(m[:, None] * dr * inv_r3[:, None], axis=0)
        return acc

# =============================================================================
# LEAPFROG INTEGRATOR
# =============================================================================
class LeapfrogIntegrator:
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
# SNAPSHOT STORAGE
# =============================================================================
@dataclass
class Snapshot:
    state: np.ndarray
    time: float
    step: int

class Trajectory:
    def __init__(self):
        self.snapshots: List[Snapshot] = []

    def add(self, state: np.ndarray, time: float, step: int):
        self.snapshots.append(Snapshot(state.copy(), time, step))

    def times(self) -> np.ndarray:
        return np.array([s.time for s in self.snapshots])

# =============================================================================
# PHASE-SPACE METRICS
# =============================================================================
class PhaseSpaceMetrics:
    def __init__(self, bins: int):
        self.bins = bins

    def density(self, state: np.ndarray) -> np.ndarray:
        X = np.hstack([state[:, :3], state[:, 3:6]])
        H, _ = np.histogramdd(X, bins=self.bins)
        return H / np.sum(H)

# =============================================================================
# ENTROPY
# =============================================================================
class EntropyEvolution:
    def __init__(self, bins: int):
        self.metrics = PhaseSpaceMetrics(bins)

    def entropy(self, state: np.ndarray) -> float:
        P = self.metrics.density(state)
        mask = P > 0
        return -np.sum(P[mask] * np.log(P[mask]))

    def series(self, traj: Trajectory) -> np.ndarray:
        return np.array([self.entropy(s.state) for s in traj.snapshots])

# =============================================================================
# DISK DIAGNOSTICS
# =============================================================================
class DiskHeatingTracker:
    @staticmethod
    def scale_height(state: np.ndarray) -> float:
        return np.sqrt(np.mean(state[:, 2]**2))

    @staticmethod
    def velocity_dispersion(state: np.ndarray) -> np.ndarray:
        return np.var(state[:, 3:6], axis=0)

    def track(self, traj: Trajectory, disk_id: int):
        hz, sig = [], []
        for s in traj.snapshots:
            disk = s.state[s.state[:, 7] == disk_id]
            hz.append(self.scale_height(disk))
            sig.append(self.velocity_dispersion(disk))
        return np.array(hz), np.array(sig)

# =============================================================================
# ANGULAR MOMENTUM
# =============================================================================
class AngularMomentumBudget:
    @staticmethod
    def total_Lz(state: np.ndarray) -> float:
        x, y = state[:, 0], state[:, 1]
        vx, vy = state[:, 3], state[:, 4]
        return np.sum(state[:, 6] * (x*vy - y*vx))

    def component_Lz(self, traj: Trajectory, component_id: int):
        return np.array([self.total_Lz(s.state[s.state[:, 7] == component_id])
                         for s in traj.snapshots])

# =============================================================================
# ENERGY
# =============================================================================
class ConservedQuantities:
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

class EnergyDiagnostics:
    def __init__(self, softening: float):
        self.softening = softening

    def series(self, traj: Trajectory):
        return np.array([ConservedQuantities.total_energy(s.state, self.softening)
                         for s in traj.snapshots])

# =============================================================================
# MERGER SIMULATION
# =============================================================================
class MergerSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.gravity = GravitySolver(config.softening_length)
        self.integrator = LeapfrogIntegrator(self.gravity)

    def initialize(self):
        """HST-aligned MW–M31 initial condition setup."""
        # --- Galaxy definitions ---
        mw_disk  = GalaxyComponent("MW Disk", 0, 6e10, 600, 3.0, 0.3)
        mw_halo  = GalaxyComponent("MW Halo", 1, 1e12, 600, 30.0)
        m31_disk = GalaxyComponent("M31 Disk", 2, 8e10, 600, 5.0, 0.4)
        m31_halo = GalaxyComponent("M31 Halo", 3, 1.5e12, 600, 40.0)

        # --- Particle generation ---
        state = np.vstack([
            InitialConditionFactory.exponential_disk(mw_disk),
            InitialConditionFactory.hernquist_halo(mw_halo),
            InitialConditionFactory.exponential_disk(m31_disk),
            InitialConditionFactory.hernquist_halo(m31_halo)
        ])

        # --- HST-aligned COM separation & velocity ---
        r_sep = np.array([780.0, 0.0, 0.0])     # kpc along x-axis
        v_rel = np.array([-109.0, 20.0, 0.0])   # km/s radial+tangential

        # Use your COM module to compute current MW/M31 COMs
        mw_COM = CenterOfMass2(state, ptype=0)
        mw_pos = mw_COM.COM_P()
        mw_vel = mw_COM.COM_V(*mw_pos)[0]

        m31_COM = CenterOfMass2(state, ptype=2)
        m31_pos = m31_COM.COM_P()
        m31_vel = m31_COM.COM_V(*m31_pos)[0]

        # Compute required shifts
        delta_pos = r_sep - (m31_pos.value - mw_pos.value)
        delta_vel = v_rel - (m31_vel.value - mw_vel.value)

        # Apply shift to all M31 particles (component_id >= 2)
        m31_mask = state[:, 7] >= 2
        state[m31_mask, :3] += delta_pos
        state[m31_mask, 3:6] += delta_vel

        return state

    def run(self) -> Trajectory:
        """Integrate the merger forward in time."""
        state = self.initialize()
        traj = Trajectory()
        time = 0.0

        for step in range(self.config.n_steps):
            state = self.integrator.step(state, self.config.timestep)
            time += self.config.timestep

            if step % self.config.snapshot_stride == 0:
                traj.add(state, time, step)
                if self.config.save_snapshots:
                    np.save(f"snapshot_{step:04d}.npy", state)

        return traj

# =============================================================================
# POST-PROCESSING PIPELINE
# =============================================================================
class PostProcessingPipeline:
    """Compute diagnostics from the merger trajectory."""
    def __init__(self, config: SimulationConfig):
        self.entropy = EntropyEvolution(config.phasespace_bins)
        self.heating = DiskHeatingTracker()
        self.angular = AngularMomentumBudget()
        self.energy = EnergyDiagnostics(config.softening_length)

    def run(self, traj: Trajectory):
        return {
            "entropy": self.entropy.series(traj),
            "energy": self.energy.series(traj),
            "mw_disk_heating": self.heating.track(traj, 0),
            "m31_disk_heating": self.heating.track(traj, 2),
            "mw_Lz": self.angular.component_Lz(traj, 0),
            "m31_Lz": self.angular.component_Lz(traj, 2),
        }

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    config = SimulationConfig()
    sim = MergerSimulation(config)
    trajectory = sim.run()

    pipeline = PostProcessingPipeline(config)
    results = pipeline.run(trajectory)

    print("Simulation complete.")
    print("Final entropy:", results["entropy"][-1])

    # Optional: plot MW–M31 separation
    mw_com = [CenterOfMass2(s.state, ptype=0).COM_P().value for s in trajectory.snapshots]
    m31_com = [CenterOfMass2(s.state, ptype=2).COM_P().value for s in trajectory.snapshots]
    separation = [np.linalg.norm(m31 - mw) for mw, m31 in zip(mw_com, m31_com)]

    plt.plot(np.arange(len(separation)) * config.timestep / SEC_PER_GYR, separation)
    plt.xlabel("Time [Gyr]")
    plt.ylabel("MW–M31 separation [kpc]")
    plt.title("HST-aligned MW–M31 merger separation")
    plt.show()
