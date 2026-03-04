"""
MOVIE 1: Structural Evolution of MW–M31 Merger
----------------------------------------------
Visualizes halo morphology and principal axes evolution.

Shows:
- MW particles
- M31 particles
- Principal axes (from inertia tensor)
- COM-centered system
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from CenterOfMass2 import CenterOfMass

# ============================================================
# CONFIGURATION
# ============================================================

snapshots = np.arange(0, 800)
output_movie = "MW_M31_structure_evolution.mp4"
fps = 20
dpi = 200
R_LIMIT = 300  # fixed spatial limit in kpc for consistent scaling

# ============================================================
# INERTIA TENSOR + PRINCIPAL AXES
# ============================================================

def inertia_tensor(pos, masses):
    """
    Computes full 3D inertia tensor of halo.
    """
    I = np.zeros((3,3))
    for i in range(len(masses)):
        x, y, z = pos[i]
        m = masses[i]
        I += m * np.array([
            [y**2 + z**2, -x*y, -x*z],
            [-x*y, x**2 + z**2, -y*z],
            [-x*z, -y*z, x**2 + y**2]
        ])
    return I

def principal_axes(pos, masses):
    """
    Returns principal axis lengths (a ≥ b ≥ c)
    and eigenvectors.
    """
    I = inertia_tensor(pos, masses)
    eigvals, eigvecs = np.linalg.eigh(I)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    a = np.sqrt(eigvals[0])
    b = np.sqrt(eigvals[1])
    c = np.sqrt(eigvals[2])

    return a, b, c, eigvecs

# ============================================================
# FIGURE SETUP
# ============================================================

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

writer = FFMpegWriter(fps=fps)

# ============================================================
# MOVIE LOOP
# ============================================================

with writer.saving(fig, output_movie, dpi=dpi):

    for snap in snapshots:

        # -------------------------------------------
        # Load MW and M31 snapshots
        # -------------------------------------------

        MW  = CenterOfMass(f"MW_{snap:03d}.txt", 1)
        M31 = CenterOfMass(f"M31_{snap:03d}.txt", 1)

        # Combine halos
        x = np.concatenate((MW.x, M31.x))
        y = np.concatenate((MW.y, M31.y))
        z = np.concatenate((MW.z, M31.z))
        m = np.concatenate((MW.m, M31.m))

        pos = np.vstack((x,y,z)).T

        # -------------------------------------------
        # COM centering (critical for morphology)
        # -------------------------------------------

        xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
        pos -= np.array([xcom, ycom, zcom])

        # Split back into MW and M31 after centering
        N_mw = len(MW.x)
        pos_mw = pos[:N_mw]
        pos_m31 = pos[N_mw:]

        # -------------------------------------------
        # Compute principal axes
        # -------------------------------------------

        a, b, c, eigvecs = principal_axes(pos, m)

        # -------------------------------------------
        # Render frame
        # -------------------------------------------

        ax.clear()

        # Particle rendering (low alpha to show structure)
        ax.scatter(pos_mw[:,0], pos_mw[:,1], pos_mw[:,2],
                   s=0.2, alpha=0.4)

        ax.scatter(pos_m31[:,0], pos_m31[:,1], pos_m31[:,2],
                   s=0.2, alpha=0.4)

        # Draw principal axes
        scale = a * 0.5  # scale for visualization
        origin = np.array([0,0,0])

        for i in range(3):
            vec = eigvecs[:,i] * scale
            ax.plot([0, vec[0]],
                    [0, vec[1]],
                    [0, vec[2]])

        ax.set_xlim(-R_LIMIT, R_LIMIT)
        ax.set_ylim(-R_LIMIT, R_LIMIT)
        ax.set_zlim(-R_LIMIT, R_LIMIT)

        ax.set_title(f"Structural Evolution | Snapshot {snap:03d}")

        writer.grab_frame()

        print(f"Rendered structure frame {snap:03d}")

print("Structure movie complete.")
"""
MOVIE 2: Angular Momentum Evolution
-----------------------------------
Visualizes halo angular momentum vector over time.

Shows:
- Particles
- Specific angular momentum vector arrow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from CenterOfMass2 import CenterOfMass

# ============================================================
# CONFIG
# ============================================================

snapshots = np.arange(0, 800)
output_movie = "MW_M31_spin_evolution.mp4"
fps = 20
dpi = 200
R_LIMIT = 300

# ============================================================
# ANGULAR MOMENTUM FUNCTION
# ============================================================

def angular_momentum_vector(pos, vel, masses):
    """
    Computes mass-weighted specific angular momentum vector.
    """
    L = np.sum(np.cross(pos, vel) * masses[:,None], axis=0)
    return L / np.sum(masses)

# ============================================================
# FIGURE SETUP
# ============================================================

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

writer = FFMpegWriter(fps=fps)

# ============================================================
# MOVIE LOOP
# ============================================================

with writer.saving(fig, output_movie, dpi=dpi):

    for snap in snapshots:

        # -------------------------------------------
        # Load snapshots
        # -------------------------------------------

        MW  = CenterOfMass(f"MW_{snap:03d}.txt", 1)
        M31 = CenterOfMass(f"M31_{snap:03d}.txt", 1)

        x = np.concatenate((MW.x, M31.x))
        y = np.concatenate((MW.y, M31.y))
        z = np.concatenate((MW.z, M31.z))
        vx = np.concatenate((MW.vx, M31.vx))
        vy = np.concatenate((MW.vy, M31.vy))
        vz = np.concatenate((MW.vz, M31.vz))
        m = np.concatenate((MW.m, M31.m))

        pos = np.vstack((x,y,z)).T
        vel = np.vstack((vx,vy,vz)).T

        # -------------------------------------------
        # COM centering
        # -------------------------------------------

        xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
        vxcom, vycom, vzcom = MW.COMdefine(vx, vy, vz, m)

        pos -= np.array([xcom, ycom, zcom])
        vel -= np.array([vxcom, vycom, vzcom])

        # -------------------------------------------
        # Compute angular momentum vector
        # -------------------------------------------

        L_vec = angular_momentum_vector(pos, vel, m)

        # Normalize for display scaling
        L_display = L_vec / np.linalg.norm(L_vec) * 150

        # -------------------------------------------
        # Render frame
        # -------------------------------------------

        ax.clear()

        ax.scatter(pos[:,0], pos[:,1], pos[:,2],
                   s=0.2, alpha=0.3)

        # Angular momentum arrow
        ax.quiver(0,0,0,
                  L_display[0],
                  L_display[1],
                  L_display[2],
                  linewidth=2)

        ax.set_xlim(-R_LIMIT, R_LIMIT)
        ax.set_ylim(-R_LIMIT, R_LIMIT)
        ax.set_zlim(-R_LIMIT, R_LIMIT)

        ax.set_title(f"Angular Momentum Evolution | Snapshot {snap:03d}")

        writer.grab_frame()

        print(f"Rendered spin frame {snap:03d}")

print("Spin movie complete.")
