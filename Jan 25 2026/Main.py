# =============================================================================
# GRAVITY SOLVER
# =============================================================================

class GravitySolver:
    """
    Direct-summation Newtonian gravity solver with softening.

    This is O(N^2) by design:
    • avoids tree-code approximations
    • maximizes physical transparency
    • suitable for controlled, moderate-N experiments
    """

    def __init__(self, softening: float):
        # Softening length prevents numerical divergences at small separations
        self.softening = softening

    def accelerations(self, pos: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Compute pairwise Newtonian accelerations with softening."""

        # Number of particles
        N = len(pos)

        # Initialize acceleration array
        acc = np.zeros_like(pos)

        # Brute-force force accumulation
        for i in range(N):

            # Relative displacement vectors r_i - r_j
            dr = pos[i] - pos

            # Squared distance with Plummer softening
            r2 = np.sum(dr**2, axis=1) + self.softening**2

            # Inverse r^3 factor for Newtonian force
            inv_r3 = r2**(-1.5)

            # Sum contributions from all particles j ≠ i
            acc[i] -= G * np.sum(
                m[:, None] * dr * inv_r3[:, None],
                axis=0
            )

        return acc


# =============================================================================
# LEAPFROG INTEGRATOR
# =============================================================================

class LeapfrogIntegrator:
    """
    Second-order symplectic leapfrog integrator.

    Chosen because it:
    • approximately conserves energy over long times
    • is time-reversible
    • is standard for collisionless N-body dynamics
    """

    def __init__(self, solver: GravitySolver):
        # Gravity solver injected for modularity
        self.solver = solver

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance the system by one timestep using Kick–Drift–Kick.
        """

        # Unpack state vector
        pos = state[:, :3]
        vel = state[:, 3:6]
        m   = state[:, 6]

        # First kick (half-step velocity update)
        a0 = self.solver.accelerations(pos, m)
        vel_half = vel + 0.5 * dt * a0

        # Drift (full-step position update)
        pos_new = pos + dt * vel_half

        # Second kick (complete velocity update)
        a1 = self.solver.accelerations(pos_new, m)
        vel_new = vel_half + 0.5 * dt * a1

        # Write updated values back into state array
        state[:, :3] = pos_new
        state[:, 3:6] = vel_new

        return state


# =============================================================================
# SNAPSHOT STORAGE
# =============================================================================

@dataclass
class Snapshot:
    # Full particle state at a given time
    state: np.ndarray

    # Physical simulation time (seconds)
    time: float

    # Integer integration step index
    step: int


class Trajectory:
    """
    Container class holding the full simulation history.

    Stores snapshots in memory to enable
    post-processing without recomputation.
    """

    def __init__(self):
        # List of Snapshot objects
        self.snapshots: List[Snapshot] = []

    def add(self, state: np.ndarray, time: float, step: int):
        # Store a deep copy to prevent mutation during integration
        self.snapshots.append(Snapshot(state.copy(), time, step))

    def times(self) -> np.ndarray:
        # Convenience accessor for time series analysis
        return np.array([s.time for s in self.snapshots])


# =============================================================================
# PHASE-SPACE METRICS
# =============================================================================

class PhaseSpaceMetrics:
    """
    Provides coarse-grained estimates of phase-space density.

    Phase space here is 6D: (x, y, z, vx, vy, vz).
    """

    def __init__(self, bins: int):
        # Number of bins per dimension
        self.bins = bins

    def density(self, state: np.ndarray) -> np.ndarray:
        # Concatenate position and velocity into 6D vectors
        X = np.hstack([state[:, :3], state[:, 3:6]])

        # Histogram approximation to f(x, v)
        H, _ = np.histogramdd(X, bins=self.bins)

        # Normalize to obtain probability distribution
        return H / np.sum(H)


# =============================================================================
# ENTROPY
# =============================================================================

class EntropyEvolution:
    """
    Tracks Shannon entropy of the coarse-grained phase-space distribution.

    Interpreted as a diagnostic of irreversible mixing,
    not thermodynamic entropy.
    """

    def __init__(self, bins: int):
        self.metrics = PhaseSpaceMetrics(bins)

    def entropy(self, state: np.ndarray) -> float:
        # Compute phase-space probability density
        P = self.metrics.density(state)

        # Mask zero-probability bins to avoid log(0)
        mask = P > 0

        # Shannon entropy
        return -np.sum(P[mask] * np.log(P[mask]))

    def series(self, traj: Trajectory) -> np.ndarray:
        # Entropy evaluated at each stored snapshot
        return np.array([self.entropy(s.state) for s in traj.snapshots])


# =============================================================================
# DISK DIAGNOSTICS
# =============================================================================

class DiskHeatingTracker:
    """
    Quantifies disk heating via:
    • vertical thickening (scale height)
    • velocity dispersion growth
    """

    @staticmethod
    def scale_height(state: np.ndarray) -> float:
        # RMS vertical height as a proxy for disk thickness
        return np.sqrt(np.mean(state[:, 2]**2))

    @staticmethod
    def velocity_dispersion(state: np.ndarray) -> np.ndarray:
        # Velocity variance along each Cartesian axis
        return np.var(state[:, 3:6], axis=0)

    def track(self, traj: Trajectory, disk_id: int):
        # Time series containers
        hz, sig = [], []

        for s in traj.snapshots:
            # Select particles belonging to the specified disk
            disk = s.state[s.state[:, 7] == disk_id]

            hz.append(self.scale_height(disk))
            sig.append(self.velocity_dispersion(disk))

        return np.array(hz), np.array(sig)


# =============================================================================
# ANGULAR MOMENTUM
# =============================================================================

class AngularMomentumBudget:
    """
    Tracks angular momentum transfer between components.

    Focuses on Lz (disk symmetry axis).
    """

    @staticmethod
    def total_Lz(state: np.ndarray) -> float:
        # Extract planar coordinates and velocities
        x, y = state[:, 0], state[:, 1]
        vx, vy = state[:, 3], state[:, 4]

        # Lz = Σ m (x vy − y vx)
        return np.sum(state[:, 6] * (x*vy - y*vx))

    def component_Lz(self, traj: Trajectory, component_id: int):
        # Compute Lz time series for a given component
        return np.array([
            self.total_Lz(s.state[s.state[:, 7] == component_id])
            for s in traj.snapshots
        ])

# =============================================================================
# ENERGY
# =============================================================================

class ConservedQuantities:
    """
    Energy diagnostics for monitoring numerical stability.

    Uses Plummer-softened gravity, consistent with the force law
    employed in the GravitySolver.
    """

    @staticmethod
    def total_energy(state: np.ndarray, softening: float) -> float:
        pos = state[:, :3]
        vel = state[:, 3:6]
        m   = state[:, 6]

        N = len(pos)

        # ------------------------------------------------------------------
        # Kinetic Energy
        # ------------------------------------------------------------------
        KE = 0.5 * np.sum(m * np.sum(vel**2, axis=1))

        # ------------------------------------------------------------------
        # Potential Energy (pairwise, Plummer-softened)
        # ------------------------------------------------------------------
        PE = 0.0
        for i in range(N):
            dr = pos[i] - pos
            r2 = np.sum(dr**2, axis=1) + softening**2

            # Avoid self-interaction
            inv_r = np.where(r2 > 0, 1.0 / np.sqrt(r2), 0.0)

            PE -= G * np.sum(m[i] * m * inv_r)

        # Factor of 1/2 avoids double counting
        return KE + 0.5 * PE


class EnergyDiagnostics:
    """
    Wrapper class for energy time series computation.
    """

    def __init__(self, softening: float):
        self.softening = softening

    def series(self, traj: Trajectory) -> np.ndarray:
        return np.array([
            ConservedQuantities.total_energy(s.state, self.softening)
            for s in traj.snapshots
        ])
