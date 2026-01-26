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

# Core numerical array library (vectorized math, linear algebra)
import numpy as np

# Plotting library used only in post-processing / visualization
import matplotlib.pyplot as plt

# Importing entropy module to calculate Shannon entropy of the given distribution
# (Shannon entropy quantifies the expected uncertainty inherent in the possible outcomes of a discrete random variable)
# Here, entropy is used as a coarse-grained proxy for phase-space mixing / irreversibility
from scipy.stats import entropy 

# Importing a decorator (allows one to add extra behavior to a function, without changing the function's code)
# (A decorator is a dataclass allows us to generate special objects like __init__)
# Dataclasses are used throughout to store physical parameters cleanly
from dataclasses import dataclass

# Importing the List type hint from the typing module for better readability. 
# (A type hint is a syntactic construct used to indicate the expected data types of variables, function arguments, and return values)
from typing import List

# Import COM module for MW/M31 tracking (Customizable for other COM modules, only the filename needs to be changed)
# This module is critical for enforcing HST-aligned relative positions and velocities
from CenterOfMass2 import CenterOfMass2


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# Gravitational constant in kpc (km/s)^2 / Msun
# Astrophysical unit choice avoids repeated unit conversions
G = 4.302e-6        

# Seconds per Gyr (used only for plotting and interpretability)
SEC_PER_GYR = 3.154e16  


# =============================================================================
# NUMERICAL CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    # Softening length to regularize the Newtonian force at small separations
    softening_length: float = 0.5               # kpc

    # Fixed timestep (50 Myr converted to seconds)
    # Chosen to balance stability vs computational cost
    timestep: float = 50e6 * 3.154e7           # seconds

    # Number of integration steps
    n_steps: int = 120                          # number of integration steps

    # Number of bins per dimension for phase-space entropy calculation
    phasespace_bins: int = 24                   # for entropy computation

    # Store every Nth snapshot (1 = store all)
    snapshot_stride: int = 1                    # store every N steps

    # Diagnostic toggles
    enable_energy_tracking: bool = True
    enable_angular_momentum_tracking: bool = True

    # Toggle for writing snapshots to disk
    save_snapshots: bool = False                # toggle file output


# =============================================================================
# GALAXY COMPONENT ABSTRACTIONS
# =============================================================================

@dataclass
class GalaxyComponent:
    # Human-readable name (used mainly for debugging / clarity)
    name: str

    # Integer ID used to tag particles belonging to this component
    component_id: int

    # Total mass of the component (Msun)
    total_mass: float

    # Number of particles used to represent the component
    particle_count: int

    # Characteristic scale length (disk Rd or halo a)
    scale_length: float

    # Vertical scale height (disk only; halo defaults to zero)
    scale_height: float = 0.0

    # Collisionless by construction (no hydrodynamics)
    component_type: str = "collisionless"  # disk or halo


# =============================================================================
# INITIAL CONDITIONS FACTORY
# =============================================================================

class InitialConditionFactory:
    """
    Factory class for generating initial particle distributions.
    Separates structural assumptions from the simulation logic itself.
    """

    @staticmethod
    def hernquist_halo(component: GalaxyComponent) -> np.ndarray:
        """Generate a Hernquist halo with placeholder zero velocities (to be improved)."""

        n, a, M = component.particle_count, component.scale_length, component.total_mass

        # Spherically random positions (Hernquist cumulative distribution)
        # Uses inverse-transform sampling
        u = np.random.rand(n)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))

        # Isotropic angular distribution
        costheta = 2*np.random.rand(n) - 1
        phi = 2*np.pi*np.random.rand(n)

        x = r*np.sqrt(1-costheta**2)*np.cos(phi)
        y = r*np.sqrt(1-costheta**2)*np.sin(phi)
        z = r*costheta

        # State array: x,y,z,vx,vy,vz,mass,component_id
        # Velocities are initialized to zero (cold halo approximation)
        state = np.zeros((n, 8))
        state[:, :3] = np.column_stack([x, y, z])
        state[:, 6] = M / n
        state[:, 7] = component.component_id

        # TODO: Replace zero velocities with equilibrium-sampled velocities
        # This will reduce artificial relaxation in future versions
        return state


    @staticmethod
    def exponential_disk(component: GalaxyComponent) -> np.ndarray:
        """Generate an exponential disk in cylindrical coordinates."""

        n, Rd, hz, M = component.particle_count, component.scale_length, component.scale_height, component.total_mass

        # Radial exponential profile
        R = -Rd * np.log(1 - np.random.rand(n))

        # Uniform azimuthal angle
        phi = 2*np.pi*np.random.rand(n)

        # Gaussian vertical thickness
        z = hz * np.random.randn(n)

        x = R * np.cos(phi)
        y = R * np.sin(phi)

        # Circular velocity (approximate)
        # Ignores halo contribution intentionally for controlled dynamics
        vphi = np.sqrt(G * M / (R + 1e-2))
        vx = -vphi * np.sin(phi)
        vy =  vphi * np.cos(phi)

        state = np.zeros((n, 8))
        state[:, :3] = np.column_stack([x, y, z])
        state[:, 3:6] = np.column_stack([vx, vy, np.zeros(n)])
        state[:, 6] = M / n
        state[:, 7] = component.component_id

        return state
