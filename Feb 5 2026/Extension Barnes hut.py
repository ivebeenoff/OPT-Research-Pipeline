###############################################################################
# MW–M31 N-BODY SIMULATION EXTENSIONS
# -------------------------------------------------------------
# This module adds:
#   1) Energy & angular momentum diagnostics
#   2) Vectorized direct N^2 gravity (NO nested for-loops)
#   3) Barnes–Hut vs direct comparison for small N
#   4) Adaptive timestep control
#
# Design constraints:
#   - Units: kpc, km/s, Gyr, Msun
#   - No nested Python for-loops anywhere
#   - O(N^2) work ONLY via NumPy broadcasting
#   - Barnes–Hut traversal uses a single loop per particle.
###############################################################################

import numpy as np

# Gravitational constant in simulation units
G = 4.302e-6  # kpc (km/s)^2 / Msun

# Conversion factor for leapfrog drift/kick
# (1 km/s * 1 Gyr ≈ 1.0227 kpc)
KM_S_TO_KPC_GYR = 1.0227121650537077


###############################################################################
# 1. BASIC DIAGNOSTICS (FULLY VECTORIZED)
###############################################################################

def kinetic_energy(state):
    """
    Total kinetic energy.

    state[:,3:6] : velocities [km/s]
    state[:,6]   : masses [Msun]

    Returns:
        Scalar kinetic energy
    """
    v = state[:,3:6]
    m = state[:,6]
    return 0.5 * np.sum(m * np.einsum("ij,ij->i", v, v))


def angular_momentum(state):
    """
    Total angular momentum vector.

    L = sum_i m_i (r_i x v_i)

    Returns:
        3-vector (Lx, Ly, Lz)
    """
    r = state[:,0:3]
    v = state[:,3:6]
    m = state[:,6]
    return np.sum(np.cross(r, v) * m[:,None], axis=0)


###############################################################################
# 2. DIRECT N^2 GRAVITY (VECTORIZED, NO NESTED LOOPS)
###############################################################################

def accelerations_direct(pos, m, eps):
    """
    Vectorized O(N^2) gravitational acceleration with Plummer softening.

    IMPORTANT:
    - Uses NumPy broadcasting
    - No nested Python loops
    - Only suitable for small N (validation, diagnostics)

    pos : (N,3) positions [kpc]
    m   : (N,)  masses [Msun]
    eps : softening length [kpc]
    """
    # Pairwise separation vectors: r_i - r_j
    dx = pos[:,None,:] - pos[None,:,:]      # shape (N, N, 3)

    # Squared distance + softening
    r2 = np.sum(dx*dx, axis=2) + eps*eps    # shape (N, N)

    # Remove self-interaction safely
    np.fill_diagonal(r2, np.inf)

    # 1 / |r|^3 term
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))

    # Newtonian acceleration sum over j
    acc = -G * np.sum(
        dx * inv_r3[:,:,None] * m[None,:,None],
        axis=1
    )

    return acc


def potential_energy_direct(state, eps):
    """
    Vectorized gravitational potential energy.

    Uses:
        U = -1/2 * sum_{i != j} G m_i m_j / r_ij

    The factor 1/2 prevents double-counting.
    """
    pos = state[:,0:3]
    m = state[:,6]

    dx = pos[:,None,:] - pos[None,:,:]
    r = np.sqrt(np.sum(dx*dx, axis=2) + eps*eps)

    np.fill_diagonal(r, np.inf)

    U = -0.5 * G * np.sum(m[:,None] * m[None,:] / r)
    return U


###############################################################################
# 3. BARNES–HUT VS DIRECT COMPARISON (NO NESTED LOOPS)
###############################################################################

def compare_bh_vs_direct(state, eps, theta, Ntest=200):
    """
    Quantitative Barnes–Hut accuracy check.

    Procedure:
    - Take a small subset of particles
    - Compute accelerations using:
        (1) Barnes–Hut tree
        (2) Vectorized direct solver
    - Return relative L2 error

    NOTE:
    Barnes–Hut necessarily uses one loop over particles.
    This is unavoidable in pure Python and is standard practice.
    """
    test = state[:Ntest]

    # Barnes–Hut accelerations
    tree = BarnesHutTree(test[:,0:3], test[:,6], eps, theta)
    a_bh = tree.accelerations()

    # Direct accelerations (vectorized)
    a_dir = accelerations_direct(test[:,0:3], test[:,6], eps)

    rel_err = np.linalg.norm(a_bh - a_dir) / np.linalg.norm(a_dir)
    return rel_err


###############################################################################
# 4. ADAPTIVE TIMESTEP (ACCELERATION-BASED, VECTORIZED)
###############################################################################

def adaptive_timestep(acc, eps, eta=0.2, dt_max=0.05):
    """
    Conservative adaptive timestep criterion:

        dt = eta * sqrt(eps / |a|max)

    This dramatically improves energy conservation near close encounters.

    acc    : (N,3) accelerations
    eps    : softening length
    eta    : safety factor (0.1–0.3 typical)
    dt_max : global timestep cap
    """
    amax = np.max(np.linalg.norm(acc, axis=1))

    if amax == 0.0:
        return dt_max

    return min(dt_max, eta * np.sqrt(eps / amax))


###############################################################################
# 5. HOW THIS IS USED IN THE MAIN INTEGRATOR (EXCERPT)
###############################################################################
#
# acc = tree.accelerations()
# dt  = adaptive_timestep(acc, eps)
#
# vel += 0.5 * dt * acc * KM_S_TO_KPC_GYR
# pos += dt * vel * KM_S_TO_KPC_GYR
#
# acc = tree.accelerations()
# vel += 0.5 * dt * acc * KM_S_TO_KPC_GYR
#
###############################################################################

# End of module
###############################################################################
