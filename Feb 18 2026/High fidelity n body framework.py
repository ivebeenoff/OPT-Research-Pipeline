# ================================================================
# MW–M31 HIGH-FIDELITY N-BODY FRAMEWORK
# ================================================================
#
# PURPOSE
# -------
# This file defines the core infrastructure for a research-grade
# collisionless N-body simulation suitable for galaxy interactions
# such as Milky Way–Andromeda mergers.
#
# IMPORTANT
# ---------
# This file DOES NOT RUN A SIMULATION.
# It defines data structures and numerical tools.
#
# To run a simulation you still need:
#   • Initial conditions generator
#   • Symplectic integrator (e.g., leapfrog)
#   • Time loop
#   • Visualization
#
# ================================================================
#
# UNITS
# -----
#   length = kpc
#   time   = Gyr
#   mass   = Msun
#
# This avoids huge/small floating-point values.
#
# ================================================================
#
# COMPLEXITY OVERVIEW
# -------------------
#
# Let N = number of particles.
#
# Brute-force gravity:
#     Time  = O(N^2)
#     Space = O(N)
#
# Barnes–Hut with multipoles:
#     Build tree       = O(N log N)
#     Force evaluation = O(N log N)
#     Memory           = O(N)
#
# Density estimation (k-nearest neighbors):
#     Time = O(N^2)   ← expensive prototype
#     Space = O(N)
#
# Eddington inversion grid:
#     Precompute = O(M) where M = grid size (~2000)
#     Sampling    = stochastic
#
# ================================================================


import numpy as np


# ================================================================
# CONSTANTS
# ================================================================
#
# Conversion km/s → kpc/Gyr
# Many observed velocities are km/s.
#
KM_S_TO_KPC_GYR = 1.0227121650537077

# Gravitational constant in these units.
G = 4.302e-6 * KM_S_TO_KPC_GYR**2


# ================================================================
# OCTREE NODE
# ================================================================
#
# Barnes–Hut groups particles into cubes.
#
# Each node stores:
#   • total mass
#   • center of mass
#   • quadrupole tensor
#
# Quadrupole improves accuracy for elongated mass distributions.
#
# SPACE COMPLEXITY
# ----------------
# Tree has ~O(N) nodes in practice.
#
# Each node stores ~O(1) data.
#
class OctreeNode:

    def __init__(self, center, half_size, indices):

        # Cube geometry
        self.center = center
        self.half   = half_size

        # Indices of particles in this node
        self.indices = indices

        # Children nodes (max 8)
        self.children = []

        # Multipole data
        self.mass = 0.0
        self.com  = np.zeros(3)
        self.Q    = np.zeros((3,3))

        self.is_leaf = False



# ================================================================
# BARNES–HUT TREE
# ================================================================
#
# PURPOSE
# -------
# Replace O(N^2) force calculation with O(N log N).
#
# IDEA
# ----
# Distant particle clusters can be approximated as one object.
#
# Opening criterion:
#     size / distance < theta
#
# Smaller theta → more accurate, slower.
#
# TIME COMPLEXITY
# ---------------
# Build tree        ≈ O(N log N)
# Force evaluation  ≈ O(N log N)
#
# Worst case (pathological particle distribution):
#     O(N^2)
#
class BarnesHutTree:

    def __init__(self, pos, mass, eps, theta):

        self.pos   = pos
        self.mass  = mass
        self.eps   = eps
        self.theta = theta

        # Build tree once
        self.root = self._build_root()


    # ================================================================
    # ROOT CONSTRUCTION
    # ================================================================
    #
    # We enclose all particles in a cube.
    #
    # TIME COMPLEXITY
    # ---------------
    # Mean + ptp operations = O(N)
    #
    def _build_root(self):

        center = np.mean(self.pos, axis=0)
        half   = np.max(np.ptp(self.pos, axis=0)) / 2

        return self._build_node(center, half, np.arange(len(self.pos)))


    # ================================================================
    # NODE BUILDING
    # ================================================================
    #
    # Recursively subdivide until 1 particle per leaf.
    #
    # TIME COMPLEXITY
    # ---------------
    # Average case ≈ O(N log N)
    #
    # Because each particle descends log N levels.
    #
    def _build_node(self, center, half, idx):

        node = OctreeNode(center, half, idx)

        m   = self.mass[idx]
        pos = self.pos[idx]

        # Total mass
        node.mass = np.sum(m)

        # Center of mass
        node.com = np.sum(pos * m[:,None], axis=0) / node.mass


        # ------------------------------------------------
        # QUADRUPOLE MOMENT
        # ------------------------------------------------
        #
        # Physics meaning:
        # Encodes shape of mass distribution.
        #
        # Important for tidal fields.
        #
        # TIME COMPLEXITY
        # ---------------
        # O(n_node) per node.
        #
        # Total tree cost still ≈ O(N log N).
        #
        Q = np.zeros((3,3))

        for k in range(len(idx)):
            r  = pos[k] - node.com
            r2 = np.dot(r, r)

            for i in range(3):
                for j in range(3):
                    Q[i,j] += m[k]*(3*r[i]*r[j] - r2*(i==j))

        node.Q = Q


        if len(idx) <= 1:
            node.is_leaf = True
            return node


        # ------------------------------------------------
        # PARTICLE SORTING INTO OCTANTS
        # ------------------------------------------------
        #
        # TIME COMPLEXITY
        # ---------------
        # O(n_node)
        #
        octants = ((pos > center).astype(int))
        keys = octants[:,0]*4 + octants[:,1]*2 + octants[:,2]


        for k in range(8):
            mask = keys == k
            if np.any(mask):

                offset = ((np.array([(k>>2)&1,(k>>1)&1,k&1]) - 0.5)*half)
                child_center = center + offset

                node.children.append(
                    self._build_node(child_center, half/2, idx[mask])
                )

        return node



    # ================================================================
    # FORCE CALCULATION
    # ================================================================
    #
    # TIME COMPLEXITY
    # ---------------
    # Each particle visits ≈ log N nodes → O(N log N).
    #
    # Worst case = O(N^2).
    #
    def _accel(self, i, node):

        if node.mass == 0:
            return np.zeros(3)

        if node.is_leaf and len(node.indices)==1 and node.indices[0]==i:
            return np.zeros(3)

        r = self.pos[i] - node.com
        d = np.linalg.norm(r)

        # Opening criterion
        if not node.children or (2*node.half)/d < self.theta:

            d2 = d*d + self.eps*self.eps
            d5 = d2**(5/2)
            d7 = d2**(7/2)

            # Monopole
            a_mono = -G * node.mass * r / (d2*np.sqrt(d2))

            # Quadrupole
            Qr  = node.Q @ r
            rQr = r @ Qr

            a_quad = (G/(2*d5))*Qr - (5*G/(2*d7))*rQr*r

            return a_mono + a_quad


        acc = np.zeros(3)
        for c in node.children:
            acc += self._accel(i, c)
        return acc


    def accelerations(self):
        """
        Returns N×3 acceleration array.

        TIME COMPLEXITY
        ---------------
        O(N log N)
        """
        acc = np.zeros_like(self.pos)
        for i in range(len(self.pos)):
            acc[i] = self._accel(i, self.root)
        return acc



# ================================================================
# ENERGY TRACKING
# ================================================================
#
# Why track energy?
# -----------------
# Good integrators conserve total energy.
#
# Large drift → numerical instability.
#
# TIME COMPLEXITY
# ---------------
# Tree potential evaluation ≈ O(N log N).
#
def compute_total_energy(state, tree):

    pos = state[:,0:3]
    vel = state[:,3:6]
    m   = state[:,6]

    KE = 0.5*np.sum(m*np.sum(vel**2,axis=1))

    PE = 0.0
    for i in range(len(state)):
        phi_i = tree._potential(i, tree.root)
        PE += 0.5*m[i]*phi_i

    return KE + PE



# ================================================================
# ADAPTIVE SOFTENING
# ================================================================
#
# Fixed softening fails in galaxy simulations.
#
# Dense cores need smaller ε.
# Sparse halos need larger ε.
#
# CURRENT IMPLEMENTATION
# ----------------------
# Brute-force k-nearest neighbors.
#
# TIME COMPLEXITY
# ---------------
# O(N^2)  ← bottleneck.
#
# Real codes use kd-trees.
#
def estimate_density(state, k=32):

    pos = state[:,0:3]
    m   = state[:,6]
    N   = len(state)

    rho = np.zeros(N)

    for i in range(N):
        r = pos - pos[i]
        d = np.sqrt(np.sum(r*r,axis=1))
        idx = np.argsort(d)[1:k+1]

        h = d[idx[-1]]
        volume = (4/3)*np.pi*h**3

        rho[i] = np.sum(m[idx]) / volume

    return rho



def compute_softening(state, eta=1.2):

    rho = estimate_density(state)
    m   = state[:,6]

    eps = eta*(m/rho)**(1/3)
    return eps



# ================================================================
# EDDINGTON INVERSION
# ================================================================
#
# Generates equilibrium velocity distribution for halos.
#
# Avoids artificial collapse/expansion.
#
# TIME COMPLEXITY
# ---------------
# Precomputation = O(M)
# Sampling = stochastic.
#
class EddingtonSampler:

    def __init__(self, M, a, r_max=200, n_grid=2000):

        self.M = M
        self.a = a

        self.r = np.linspace(1e-4, r_max, n_grid)
        self.psi = G*M/(self.r + a)

        rho = (M/(2*np.pi))*a/(self.r*(self.r+a)**3)
        self.rho = rho

        self.d2rho_dpsi2 = self._compute_second_derivative()


    def _compute_second_derivative(self):
        drho_dpsi = np.gradient(self.rho, self.psi)
        return np.gradient(drho_dpsi, self.psi)


    def f_E(self, E):

        mask = self.psi <= E
        psi  = self.psi[mask]

        integrand = self.d2rho_dpsi2[mask] / np.sqrt(E-psi)

        return (1/(np.sqrt(8)*np.pi**2))*np.trapz(integrand, psi)


    def sample_velocity(self, r):

        psi_r = G*self.M/(r + self.a)
        v_esc = np.sqrt(2*psi_r)

        while True:
            v = np.random.rand()*v_esc
            E = psi_r - 0.5*v*v
            if np.random.rand() < self.f_E(E)*v*v:
                return v



# ================================================================
# END OF FRAMEWORK
# ================================================================
#
# NEXT STEPS
# ----------
# 1. Leapfrog integrator
# 2. MW + M31 initial conditions
# 3. Energy error tracking
# 4. Visualization
#
# With NumPy + Jupyter (your setup), you can realistically simulate
# ~10^5 particles on a laptop.
#
# ================================================================
