# =============================================================================
# MERGER SIMULATION
# =============================================================================

class MergerSimulation:
    """
    High-level driver for the MW–M31 merger simulation.

    Handles:
    • initial condition construction
    • HST-aligned COM corrections
    • time integration
    """

    def __init__(self, config: SimulationConfig):
        # Store configuration object
        self.config = config

        # Initialize gravity solver with chosen softening length
        self.gravity = GravitySolver(config.softening_length)

        # Leapfrog integrator coupled to the gravity solver
        self.integrator = LeapfrogIntegrator(self.gravity)

    def initialize(self):
        """HST-aligned MW–M31 initial condition setup."""

        # --- Galaxy definitions ---
        # Milky Way: disk + halo
        mw_disk  = GalaxyComponent("MW Disk", 0, 6e10, 600, 3.0, 0.3)
        mw_halo  = GalaxyComponent("MW Halo", 1, 1e12, 600, 30.0)

        # Andromeda: disk + halo
        m31_disk = GalaxyComponent("M31 Disk", 2, 8e10, 600, 5.0, 0.4)
        m31_halo = GalaxyComponent("M31 Halo", 3, 1.5e12, 600, 40.0)

        # --- Particle generation ---
        # Combine all components into a single state array
        state = np.vstack([
            InitialConditionFactory.exponential_disk(mw_disk),
            InitialConditionFactory.hernquist_halo(mw_halo),
            InitialConditionFactory.exponential_disk(m31_disk),
            InitialConditionFactory.hernquist_halo(m31_halo)
        ])

        # --- HST-aligned COM separation & velocity ---
        # Observationally constrained relative separation and velocity
        r_sep = np.array([780.0, 0.0, 0.0])     # kpc along x-axis
        v_rel = np.array([-109.0, 20.0, 0.0])   # km/s radial + tangential

        # Use your COM module to compute current MW/M31 COMs
        # ptype corresponds to the disk particle IDs
        mw_COM = CenterOfMass2(state, ptype=0)
        mw_pos = mw_COM.COM_P()
        mw_vel = mw_COM.COM_V(*mw_pos)[0]

        m31_COM = CenterOfMass2(state, ptype=2)
        m31_pos = m31_COM.COM_P()
        m31_vel = m31_COM.COM_V(*m31_pos)[0]

        # Compute required shifts so that:
        # (M31 − MW) matches HST constraints
        delta_pos = r_sep - (m31_pos.value - mw_pos.value)
        delta_vel = v_rel - (m31_vel.value - mw_vel.value)

        # Apply shift to all M31 particles (component_id >= 2)
        # This preserves internal structure while correcting bulk motion
        m31_mask = state[:, 7] >= 2
        state[m31_mask, :3] += delta_pos
        state[m31_mask, 3:6] += delta_vel

        return state

    def run(self) -> Trajectory:
        """Integrate the merger forward in time."""

        # Initialize system state
        state = self.initialize()

        # Trajectory object for storing snapshots
        traj = Trajectory()

        # Simulation clock
        time = 0.0

        # Main integration loop
        for step in range(self.config.n_steps):

            # Advance system by one timestep
            state = self.integrator.step(state, self.config.timestep)
            time += self.config.timestep

            # Store snapshot according to stride
            if step % self.config.snapshot_stride == 0:
                traj.add(state, time, step)

                # Optional on-disk snapshot output
                if self.config.save_snapshots:
                    np.save(f"snapshot_{step:04d}.npy", state)

        return traj


# =============================================================================
# POST-PROCESSING PIPELINE
# =============================================================================

class PostProcessingPipeline:
    """
    Compute all diagnostics from the stored merger trajectory.

    Designed to keep analysis separate from simulation logic.
    """

    def __init__(self, config: SimulationConfig):
        # Entropy evolution (phase-space mixing)
        self.entropy = EntropyEvolution(config.phasespace_bins)

        # Disk heating diagnostics
        self.heating = DiskHeatingTracker()

        # Angular momentum tracking
        self.angular = AngularMomentumBudget()

        # Energy conservation diagnostics
        self.energy = EnergyDiagnostics(config.softening_length)

    def run(self, traj: Trajectory):
        # Compute and return all diagnostics as a dictionary
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

    # Instantiate configuration with default parameters
    config = SimulationConfig()

    # Create merger simulation object
    sim = MergerSimulation(config)

    # Run the simulation and collect trajectory
    trajectory = sim.run()

    # Initialize post-processing pipeline
    pipeline = PostProcessingPipeline(config)

    # Compute diagnostics
    results = pipeline.run(trajectory)

    print("Simulation complete.")
    print("Final entropy:", results["entropy"][-1])

    # Optional: plot MW–M31 separation over time

    # Compute COM positions for each snapshot
    mw_com = [
        CenterOfMass2(s.state, ptype=0).COM_P().value
        for s in trajectory.snapshots
    ]
    m31_com = [
        CenterOfMass2(s.state, ptype=2).COM_P().value
        for s in trajectory.snapshots
    ]

    # Relative separation magnitude
    separation = [
        np.linalg.norm(m31 - mw)
        for mw, m31 in zip(mw_com, m31_com)
    ]

    # Convert time axis to Gyr for interpretability
    plt.plot(
        np.arange(len(separation)) * config.timestep / SEC_PER_GYR,
        separation
    )

    plt.xlabel("Time [Gyr]")
    plt.ylabel("MW–M31 separation [kpc]")
    plt.title("HST-aligned MW–M31 merger separation")
    plt.show()
