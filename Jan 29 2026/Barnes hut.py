# =============================================================================
# HST-Aligned MW–M31 Merger with Barnes–Hut Gravity
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from dataclasses import dataclass
from typing import List

from CenterOfMass2 import CenterOfMass2

# =============================================================================
# CONSTANTS
# =============================================================================
G = 4.302e-6        # kpc (km/s)^2 / Msun
SEC_PER_GYR = 3.154e16

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class SimulationConfig:
    softening_length: float = 0.5
    timestep: float = 50e6 * 3.154e7
    n_steps: int = 120
    phasespace_bins: int = 24
    snapshot_stride: int = 1
    save_snapshots: bool = False
    bh_theta: float = 0.6

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
    def hernquist_halo(component: GalaxyComponent) -> np.ndarray:
        n, a, M = component.particle_count, component.scale_length, component.total_mass

        u = np.random.rand(n)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))
        cost = 2*np.random.rand(n) - 1
        phi = 2*np.pi*np.random.rand(n)

        x = r*np.sqrt(1-cost**2)*np.cos(phi)
        y = r*np.sqrt(1-cost**2)*np.sin(phi)
        z = r*cost

        # isotropic velocity dispersion (approximate equilibrium)
        sigma = np.sqrt(G*M/(2*(r+a)))
        v = sigma[:,None] * np.random.randn(n,3)

        state = np.zeros((n,8))
        state[:,0:3] = np.column_stack([x,y,z])
        state[:,3:6] = v
        state[:,6] = M/n
        state[:,7] = component.component_id
        return state

    @staticmethod
    def exponential_disk(component: GalaxyComponent) -> np.ndarray:
        n, Rd, hz, M = component.particle_count, component.scale_length, component.scale_height, component.total_mass

        R = -Rd*np.log(1-np.random.rand(n))
        phi = 2*np.pi*np.random.rand(n)
        z = hz*np.random.randn(n)

        x = R*np.cos(phi)
        y = R*np.sin(phi)

        vphi = np.sqrt(G*M/(R+0.1))
        vx = -vphi*np.sin(phi)
        vy =  vphi*np.cos(phi)

        state = np.zeros((n,8))
        state[:,0:3] = np.column_stack([x,y,z])
        state[:,3:6] = np.column_stack([vx,vy,np.zeros(n)])
        state[:,6] = M/n
        state[:,7] = component.component_id
        return state

# =============================================================================
# BARNES–HUT TREE
# =============================================================================
class OctreeNode:
    def __init__(self, center, size, idx):
        self.center = center
        self.size = size
        self.idx = idx
        self.children = []
        self.mass = 0.0
        self.com = np.zeros(3)

class BarnesHutTree:
    def __init__(self, pos, m, theta, softening):
        self.pos = pos
        self.m = m
        self.theta = theta
        self.softening = softening
        self.root = self._build()

    def _build(self):
        center = np.mean(self.pos, axis=0)
        size = np.max(np.ptp(self.pos, axis=0))/2
        return self._build_node(center, size, np.arange(len(self.pos)))

    def _build_node(self, center, size, idx):
        node = OctreeNode(center, size, idx)
        node.mass = np.sum(self.m[idx])
        node.com = np.sum(self.pos[idx]*self.m[idx][:,None], axis=0)/node.mass

        if len(idx) <= 1:
            return node

        offsets = np.array([[dx,dy,dz] for dx in (-1,1)
                                         for dy in (-1,1)
                                         for dz in (-1,1)]) * size/2

        for off in offsets:
            c = center + off
            mask = np.all(np.abs(self.pos[idx]-c) <= size/2, axis=1)
            if np.any(mask):
                node.children.append(
                    self._build_node(c, size/2, idx[mask])
                )
        return node

    def _force(self, i, node):
        r = self.pos[i] - node.com
        d = np.linalg.norm(r) + self.softening

        if len(node.children) == 0 or node.size/d < self.theta:
            return -G * node.mass * r / d**3

        acc = np.zeros(3)
        for child in node.children:
            acc += self._force(i, child)
        return acc

    def accelerations(self):
        acc = np.zeros_like(self.pos)
        for i in range(len(self.pos)):
            acc[i] = self._force(i, self.root)
        return acc

# =============================================================================
# GRAVITY SOLVER
# =============================================================================
class BarnesHutSolver:
    def __init__(self, softening, theta):
        self.softening = softening
        self.theta = theta

    def accelerations(self, pos, m):
        tree = BarnesHutTree(pos, m, self.theta, self.softening)
        return tree.accelerations()

# =============================================================================
# LEAPFROG
# =============================================================================
class LeapfrogIntegrator:
    def __init__(self, solver):
        self.solver = solver

    def step(self, state, dt):
        pos = state[:,0:3]
        vel = state[:,3:6]
        m = state[:,6]

        a0 = self.solver.accelerations(pos, m)
        vel_half = vel + 0.5*dt*a0
        pos_new = pos + dt*vel_half
        a1 = self.solver.accelerations(pos_new, m)
        vel_new = vel_half + 0.5*dt*a1

        state[:,0:3] = pos_new
        state[:,3:6] = vel_new
        return state

# =============================================================================
# SNAPSHOTS
# =============================================================================
@dataclass
class Snapshot:
    state: np.ndarray
    time: float
    step: int

class Trajectory:
    def __init__(self):
        self.snapshots = []

    def add(self, state, time, step):
        self.snapshots.append(Snapshot(state.copy(), time, step))

# =============================================================================
# MERGER SIMULATION
# =============================================================================
class MergerSimulation:
    def __init__(self, config):
        self.config = config
        self.gravity = BarnesHutSolver(config.softening_length, config.bh_theta)
        self.integrator = LeapfrogIntegrator(self.gravity)

    def initialize(self):
        mw_disk  = GalaxyComponent("MW Disk",0,6e10,600,3.0,0.3)
        mw_halo  = GalaxyComponent("MW Halo",1,1e12,600,30.0)
        m31_disk = GalaxyComponent("M31 Disk",2,8e10,600,5.0,0.4)
        m31_halo = GalaxyComponent("M31 Halo",3,1.5e12,600,40.0)

        state = np.vstack([
            InitialConditionFactory.exponential_disk(mw_disk),
            InitialConditionFactory.hernquist_halo(mw_halo),
            InitialConditionFactory.exponential_disk(m31_disk),
            InitialConditionFactory.hernquist_halo(m31_halo)
        ])

        r_sep = np.array([780.0,0,0])
        v_rel = np.array([-109.0,20.0,0])

        mw = CenterOfMass2(state, ptype=0)
        m31 = CenterOfMass2(state, ptype=2)

        dpos = r_sep - (m31.COM_P().value - mw.COM_P().value)
        dvel = v_rel - (m31.COM_V(*m31.COM_P())[0].value -
                        mw.COM_V(*mw.COM_P())[0].value)

        mask = state[:,7] >= 2
        state[mask,0:3] += dpos
        state[mask,3:6] += dvel
        return state

    def run(self):
        state = self.initialize()
        traj = Trajectory()
        time = 0.0

        for step in range(self.config.n_steps):
            state = self.integrator.step(state, self.config.timestep)
            time += self.config.timestep
            if step % self.config.snapshot_stride == 0:
                traj.add(state, time, step)
        return traj

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    config = SimulationConfig()
    sim = MergerSimulation(config)
    traj = sim.run()

    mw = [CenterOfMass2(s.state,ptype=0).COM_P().value for s in traj.snapshots]
    m31 = [CenterOfMass2(s.state,ptype=2).COM_P().value for s in traj.snapshots]
    sep = [np.linalg.norm(m31[i]-mw[i]) for i in range(len(mw))]

    plt.plot(np.arange(len(sep))*config.timestep/SEC_PER_GYR, sep)
    plt.xlabel("Time [Gyr]")
    plt.ylabel("MW–M31 separation [kpc]")
    plt.title("MW–M31 Merger (Barnes–Hut)")
    plt.show()
