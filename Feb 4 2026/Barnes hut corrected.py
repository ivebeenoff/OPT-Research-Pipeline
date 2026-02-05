# =============================================================================
# MW–M31 Merger Simulation (Corrected Barnes–Hut, Unit-Consistent)
# =============================================================================
"""
Units:
- Distance: kpc
- Velocity: km/s
- Time: Gyr
- Mass: Msun

This is an approximate N-body toy model intended for qualitative
orbital evolution, not precision cosmology.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =============================================================================
# CONSTANTS
# =============================================================================
G = 4.302e-6  # kpc (km/s)^2 / Msun
KM_S_TO_KPC_GYR = 1.0227121650537077  # exact conversion

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class SimulationConfig:
    dt: float = 0.05              # Gyr
    n_steps: int = 200
    softening: float = 0.5        # kpc
    theta: float = 0.7
    snapshot_stride: int = 1

# =============================================================================
# GALAXY COMPONENT
# =============================================================================
@dataclass
class GalaxyComponent:
    name: str
    component_id: int
    total_mass: float
    particle_count: int
    scale_length: float
    scale_height: float = 0.0

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================
class InitialConditionFactory:

    @staticmethod
    def hernquist_halo(comp: GalaxyComponent):
        n, a, M = comp.particle_count, comp.scale_length, comp.total_mass

        u = np.random.rand(n)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))
        cost = 2*np.random.rand(n) - 1
        phi = 2*np.pi*np.random.rand(n)

        x = r*np.sqrt(1-cost**2)*np.cos(phi)
        y = r*np.sqrt(1-cost**2)*np.sin(phi)
        z = r*cost

        # crude isotropic equilibrium (documented approximation)
        sigma = np.sqrt(G*M/(r + a))
        v = sigma[:,None] * np.random.randn(n,3)

        state = np.zeros((n,8))
        state[:,0:3] = np.column_stack((x,y,z))
        state[:,3:6] = v
        state[:,6] = M/n
        state[:,7] = comp.component_id
        return state

    @staticmethod
    def exponential_disk(comp: GalaxyComponent):
        n, Rd, hz, M = (
            comp.particle_count,
            comp.scale_length,
            comp.scale_height,
            comp.total_mass
        )

        R = -Rd*np.log(1-np.random.rand(n))
        phi = 2*np.pi*np.random.rand(n)
        z = hz*np.random.randn(n)

        x = R*np.cos(phi)
        y = R*np.sin(phi)

        vphi = np.sqrt(G*M/(R + Rd))
        vx = -vphi*np.sin(phi)
        vy =  vphi*np.cos(phi)

        state = np.zeros((n,8))
        state[:,0:3] = np.column_stack((x,y,z))
        state[:,3:6] = np.column_stack((vx,vy,np.zeros(n)))
        state[:,6] = M/n
        state[:,7] = comp.component_id
        return state

# =============================================================================
# CENTER OF MASS
# =============================================================================
def center_of_mass(state, mask):
    m = state[mask,6]
    x = state[mask,0:3]
    v = state[mask,3:6]
    return (
        np.sum(x*m[:,None], axis=0)/np.sum(m),
        np.sum(v*m[:,None], axis=0)/np.sum(m)
    )

# =============================================================================
# BARNES–HUT TREE
# =============================================================================
class OctreeNode:
    def __init__(self, center, half_size, indices):
        self.center = center
        self.half = half_size
        self.indices = indices
        self.children = []
        self.mass = 0.0
        self.com = np.zeros(3)

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
        node.mass = np.sum(m)
        node.com = np.sum(self.pos[idx]*m[:,None], axis=0)/node.mass

        if len(idx) <= 1:
            return node

        octants = ((self.pos[idx] > center).astype(int))
        keys = octants[:,0]*4 + octants[:,1]*2 + octants[:,2]

        for k in range(8):
            mask = keys == k
            if np.any(mask):
                offset = ((np.array([
                    (k>>2)&1,
                    (k>>1)&1,
                    k&1
                ]) - 0.5) * 2 * half / 2)
                child_center = center + offset
                node.children.append(
                    self._build_node(child_center, half/2, idx[mask])
                )
        return node

    def _accel(self, i, node):
        if node.mass == 0:
            return np.zeros(3)

        if len(node.indices) == 1 and node.indices[0] == i:
            return np.zeros(3)

        r = self.pos[i] - node.com
        d = np.linalg.norm(r)

        if not node.children or node.half/d < self.theta:
            d2 = d*d + self.eps*self.eps
            return -G * node.mass * r / (d2*np.sqrt(d2))

        acc = np.zeros(3)
        for c in node.children:
            acc += self._accel(i, c)
        return acc

    def accelerations(self):
        acc = np.zeros_like(self.pos)
        for i in range(len(self.pos)):
            acc[i] = self._accel(i, self.root)
        return acc

# =============================================================================
# LEAPFROG INTEGRATOR
# =============================================================================
class LeapfrogIntegrator:
    def __init__(self, eps, theta):
        self.eps = eps
        self.theta = theta

    def step(self, state, dt):
        pos = state[:,0:3]
        vel = state[:,3:6]
        m = state[:,6]

        tree = BarnesHutTree(pos, m, self.eps, self.theta)
        acc = tree.accelerations()

        vel += 0.5 * dt * acc * KM_S_TO_KPC_GYR
        pos += dt * vel * KM_S_TO_KPC_GYR

        tree = BarnesHutTree(pos, m, self.eps, self.theta)
        acc = tree.accelerations()
        vel += 0.5 * dt * acc * KM_S_TO_KPC_GYR

        state[:,0:3] = pos
        state[:,3:6] = vel
        return state

# =============================================================================
# SIMULATION
# =============================================================================
class MergerSimulation:
    def __init__(self, config):
        self.config = config
        self.integrator = LeapfrogIntegrator(config.softening, config.theta)

    def initialize(self):
        mw_d = GalaxyComponent("MW Disk",0,6e10,600,3.0,0.3)
        mw_h = GalaxyComponent("MW Halo",1,1e12,600,30.0)
        m31_d = GalaxyComponent("M31 Disk",2,8e10,600,5.0,0.4)
        m31_h = GalaxyComponent("M31 Halo",3,1.5e12,600,40.0)

        state = np.vstack([
            InitialConditionFactory.exponential_disk(mw_d),
            InitialConditionFactory.hernquist_halo(mw_h),
            InitialConditionFactory.exponential_disk(m31_d),
            InitialConditionFactory.hernquist_halo(m31_h)
        ])

        mw_mask = state[:,7] < 2
        m31_mask = state[:,7] >= 2

        mw_p, mw_v = center_of_mass(state, mw_mask)
        m31_p, m31_v = center_of_mass(state, m31_mask)

        r_sep = np.array([780.0,0,0])
        v_rel = np.array([-109.0,20.0,0])

        state[m31_mask,0:3] += r_sep - (m31_p - mw_p)
        state[m31_mask,3:6] += v_rel - (m31_v - mw_v)

        return state

    def run(self):
        state = self.initialize()
        history = []

        for step in range(self.config.n_steps):
            state = self.integrator.step(state, self.config.dt)
            if step % self.config.snapshot_stride == 0:
                history.append(state.copy())

        return history

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    config = SimulationConfig()
    sim = MergerSimulation(config)
    traj = sim.run()

    mw_sep = []
    for s in traj:
        mw_p,_ = center_of_mass(s, s[:,7] < 2)
        m31_p,_ = center_of_mass(s, s[:,7] >= 2)
        mw_sep.append(np.linalg.norm(m31_p - mw_p))

    t = np.arange(len(mw_sep))*config.dt

    plt.plot(t, mw_sep)
    plt.xlabel("Time [Gyr]")
    plt.ylabel("MW–M31 separation [kpc]")
    plt.title("MW–M31 Merger (Corrected Barnes–Hut)")
    plt.show()
