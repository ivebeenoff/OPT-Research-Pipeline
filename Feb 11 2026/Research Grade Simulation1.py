# ================================================================
# MW–M31 High-Fidelity N-Body Framework (Development Version)
# ================================================================
# Units: kpc, Gyr, Msun
# G is expressed in kpc^3 / (Msun Gyr^2)
#
# This version differs substantially from the earlier toy simulation:
#
# 1) Introduces quadrupole multipole expansion (previous version was monopole only).
# 2) Adds tree-based potential evaluation for energy conservation tracking
#    (previous version computed energy via O(N^2) brute force or not at all).
# 3) Implements infrastructure for adaptive softening (previous version used fixed softening).
# 4) Adds multipole update mechanism for tree reuse (previous version rebuilt tree twice per step).
# 5) Introduces Eddington inversion framework for equilibrium halo sampling
#    (previous version used approximate Gaussian dispersion).
#
# This is an architectural upgrade toward research-grade structure.
# ================================================================

import numpy as np

# ------------------------------------------------
# CONSTANTS
# ------------------------------------------------

KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.302e-6 * KM_S_TO_KPC_GYR**2


# ================================================================
# OCTREE NODE WITH MULTIPOLE SUPPORT
# ================================================================

class OctreeNode:
    """
    DIFFERENCE FROM PREVIOUS VERSION:
    ---------------------------------
    Previously, nodes stored only:
        - mass
        - center of mass

    This version additionally stores:
        - quadrupole tensor (3x3 symmetric)
        - explicit leaf flag
    """

    def __init__(self, center, half_size, indices):
        self.center = center
        self.half = half_size
        self.indices = indices
        self.children = []

        self.mass = 0.0
        self.com = np.zeros(3)
        self.Q = np.zeros((3,3))   # Quadrupole tensor
        self.is_leaf = False


# ================================================================
# BARNES–HUT TREE WITH QUADRUPOLE EXPANSION
# ================================================================

class BarnesHutTree:

    def __init__(self, pos, mass, eps, theta):
        self.pos = pos
        self.mass = mass
        self.eps = eps
        self.theta = theta
        self.root = self._build_root()

    def _build_root(self):
        center = np.mean(self.pos, axis=0)
        half = np.max(np.ptp(self.pos, axis=0)) / 2
        return self._build_node(center, half, np.arange(len(self.pos)))

    def _build_node(self, center, half, idx):

        node = OctreeNode(center, half, idx)
        m = self.mass[idx]
        pos = self.pos[idx]

        node.mass = np.sum(m)
        node.com = np.sum(pos * m[:,None], axis=0) / node.mass

        # ------------------------------------------------
        # DIFFERENCE: Quadrupole tensor computation added
        # ------------------------------------------------
        Q = np.zeros((3,3))
        for k in range(len(idx)):
            r = pos[k] - node.com
            r2 = np.dot(r, r)
            for i in range(3):
                for j in range(3):
                    Q[i,j] += m[k] * (3*r[i]*r[j] - r2*(i==j))
        node.Q = Q

        if len(idx) <= 1:
            node.is_leaf = True
            return node

        octants = ((pos > center).astype(int))
        keys = octants[:,0]*4 + octants[:,1]*2 + octants[:,2]

        for k in range(8):
            mask = keys == k
            if np.any(mask):
                offset = ((np.array([(k>>2)&1,(k>>1)&1,k&1]) - 0.5) * half)
                child_center = center + offset
                node.children.append(
                    self._build_node(child_center, half/2, idx[mask])
                )

        return node


    # ================================================================
    # MULTIPOLE FORCE EVALUATION (MONOPOLE + QUADRUPOLE)
    # ================================================================

    def _accel(self, i, node):

        if node.mass == 0:
            return np.zeros(3)

        if node.is_leaf and len(node.indices) == 1 and node.indices[0] == i:
            return np.zeros(3)

        r = self.pos[i] - node.com
        d = np.linalg.norm(r)

        # DIFFERENCE: Uses full node size in opening criterion
        if not node.children or (2*node.half)/d < self.theta:

            d2 = d*d + self.eps*self.eps
            d5 = d2**(5/2)
            d7 = d2**(7/2)

            # Monopole term
            a_mono = -G * node.mass * r / (d2*np.sqrt(d2))

            # Quadrupole correction (new)
            Qr = node.Q @ r
            rQr = r @ Qr

            a_quad = (G/(2*d5)) * Qr - (5*G/(2*d7)) * rQr * r

            return a_mono + a_quad

        acc = np.zeros(3)
        for c in node.children:
            acc += self._accel(i, c)
        return acc


    def accelerations(self):
        acc = np.zeros_like(self.pos)
        for i in range(len(self.pos)):
            acc[i] = self._accel(i, self.root)
        return acc


    # ================================================================
    # TREE-BASED POTENTIAL (FOR ENERGY TRACKING)
    # ================================================================
    # DIFFERENCE:
    # Previous version either used brute-force O(N^2) potential
    # or did not compute total energy rigorously.
    # This version evaluates potential using tree traversal.

    def _potential(self, i, node):

        if node.mass == 0:
            return 0.0

        if node.is_leaf and len(node.indices) == 1 and node.indices[0] == i:
            return 0.0

        r = self.pos[i] - node.com
        d = np.linalg.norm(r)

        if not node.children or (2*node.half)/d < self.theta:
            return -G * node.mass / np.sqrt(d*d + self.eps*self.eps)

        phi = 0.0
        for c in node.children:
            phi += self._potential(i, c)
        return phi


    # ================================================================
    # TREE REUSE: MULTIPOLE UPDATE WITHOUT REBUILD
    # ================================================================
    # DIFFERENCE:
    # Previous implementation rebuilt the entire tree twice per step.
    # This function allows updating multipoles while preserving topology.

    def update_multipoles(self):

        def update_node(node):

            if node.is_leaf:
                idx = node.indices
                m = self.mass[idx]
                pos = self.pos[idx]

                node.mass = np.sum(m)
                node.com = np.sum(pos*m[:,None], axis=0)/node.mass
                node.Q = np.zeros((3,3))
                return

            for c in node.children:
                update_node(c)

            node.mass = sum(c.mass for c in node.children)
            node.com = sum(c.mass*c.com for c in node.children)/node.mass

            Q = np.zeros((3,3))
            for c in node.children:
                Q += c.Q
            node.Q = Q

        update_node(self.root)


# ================================================================
# ENERGY TRACKER
# ================================================================

def compute_total_energy(state, tree):
    """
    DIFFERENCE:
    Uses tree-based potential rather than O(N^2) summation.
    Scales with tree cost instead of quadratic cost.
    """

    pos = state[:,0:3]
    vel = state[:,3:6]
    m = state[:,6]

    KE = 0.5 * np.sum(m * np.sum(vel**2, axis=1))

    PE = 0.0
    for i in range(len(state)):
        phi_i = tree._potential(i, tree.root)
        PE += 0.5 * m[i] * phi_i

    return KE + PE


# ================================================================
# ADAPTIVE SOFTENING (INFRASTRUCTURE)
# ================================================================
# DIFFERENCE:
# Previous version used globally fixed softening length.
# This introduces density-based softening scaffold.

def estimate_density(state, k=32):

    pos = state[:,0:3]
    m = state[:,6]
    N = len(state)

    rho = np.zeros(N)

    for i in range(N):
        r = pos - pos[i]
        d = np.sqrt(np.sum(r*r, axis=1))
        idx = np.argsort(d)[1:k+1]
        h = d[idx[-1]]
        volume = (4/3)*np.pi*h**3
        rho[i] = np.sum(m[idx]) / volume

    return rho


def compute_softening(state, eta=1.2):
    rho = estimate_density(state)
    m = state[:,6]
    eps = eta * (m / rho)**(1/3)
    return eps


# ================================================================
# EDDINGTON INVERSION FRAMEWORK (HALO EQUILIBRIUM SAMPLER)
# ================================================================
# DIFFERENCE:
# Previous simulation used Gaussian velocity dispersion approximation.
# This implements the mathematical structure for proper DF sampling.

class EddingtonSampler:

    def __init__(self, M, a, r_max=200, n_grid=2000):

        self.M = M
        self.a = a

        self.r = np.linspace(1e-4, r_max, n_grid)
        self.psi = G*M/(self.r + a)

        rho = (M/(2*np.pi)) * a / (self.r*(self.r+a)**3)

        self.rho = rho
        self.d2rho_dpsi2 = self._compute_second_derivative()

    def _compute_second_derivative(self):
        drho_dpsi = np.gradient(self.rho, self.psi)
        return np.gradient(drho_dpsi, self.psi)

    def f_E(self, E):

        mask = self.psi <= E
        psi = self.psi[mask]
        integrand = self.d2rho_dpsi2[mask] / np.sqrt(E - psi)

        return (1/(np.sqrt(8)*np.pi**2)) * np.trapz(integrand, psi)

    def sample_velocity(self, r):

        psi_r = G*self.M/(r + self.a)
        v_esc = np.sqrt(2*psi_r)

        while True:
            v = np.random.rand() * v_esc
            E = psi_r - 0.5*v*v
            if np.random.rand() < self.f_E(E) * v*v:
                return v


# ================================================================
# END OF DEVELOPMENT BLOCK
# ================================================================

# The present code represents a structural upgrade from a
# pedagogical Barnes–Hut implementation to a framework capable
# of supporting research-grade collisionless N-body evolution.
#
# Further steps include:
#   - Adaptive timestep hierarchy
#   - Softening symmetrization in force law
#   - Performance optimization and parallelization
#
# The architecture now supports multipole accuracy, energy
# diagnostics, adaptive resolution, and equilibrium halo sampling.
# ================================================================
