"""
MOVIE 3 & MOVIE 4
MW–M31 Merger Diagnostic Visualizations

Movie 3 → Convex Hull / Volume Inflation
Movie 4 → Orbital Separation Dynamics

Designed for use with:
- MW_###.txt
- M31_###.txt
- CenterOfMass2 class
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.spatial import ConvexHull
from CenterOfMass2 import CenterOfMass


# ============================================================
# COMMON CONFIGURATION
# ============================================================

snapshots = np.arange(0, 800)
fps = 20
dpi = 200
R_LIMIT = 300


# ============================================================
# ===================== MOVIE 3 ==============================
#        Convex Hull / Volume Inflation Diagnostic
# ============================================================

"""
Purpose:
--------
Visualize outer halo expansion and tidal debris growth
using the 3D convex hull.

Scientific interpretation:
- Hull expansion → tidal stripping
- Sudden volume jumps → merger interaction
- Persistent expansion → structural reconfiguration
"""

output_movie_hull = "MW_M31_hull_evolution.mp4"

fig3 = plt.figure(figsize=(8,8))
ax3 = fig3.add_subplot(111, projection='3d')

writer3 = FFMpegWriter(fps=fps)

with writer3.saving(fig3, output_movie_hull, dpi=dpi):

    for snap in snapshots:

        MW  = CenterOfMass(f"MW_{snap:03d}.txt", 1)
        M31 = CenterOfMass(f"M31_{snap:03d}.txt", 1)

        x = np.concatenate((MW.x, M31.x))
        y = np.concatenate((MW.y, M31.y))
        z = np.concatenate((MW.z, M31.z))
        m = np.concatenate((MW.m, M31.m))

        pos = np.vstack((x,y,z)).T

        # COM centering
        xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
        pos -= np.array([xcom, ycom, zcom])

        ax3.clear()

        # Particle rendering
        ax3.scatter(pos[:,0], pos[:,1], pos[:,2],
                    s=0.2, alpha=0.3)

        # Convex hull computation
        # NOTE: ConvexHull scales ~ O(N log N)
        # If N is very large, consider random downsampling
        hull = ConvexHull(pos)
        V_hull = hull.volume

        # Draw hull edges
        for simplex in hull.simplices:
            ax3.plot(pos[simplex,0],
                     pos[simplex,1],
                     pos[simplex,2])

        ax3.set_xlim(-R_LIMIT, R_LIMIT)
        ax3.set_ylim(-R_LIMIT, R_LIMIT)
        ax3.set_zlim(-R_LIMIT, R_LIMIT)

        ax3.set_title(f"Convex Hull Evolution | Snapshot {snap:03d}\n"
                      f"Hull Volume = {V_hull:.2e} kpc^3")

        writer3.grab_frame()
        print(f"Rendered hull frame {snap:03d}")

print("Hull movie complete.")


# ============================================================
# ===================== MOVIE 4 ==============================
#          MW–M31 Orbital Separation Dynamics
# ============================================================

"""
Purpose:
--------
Track pure orbital decay of MW and M31.

Scientific interpretation:
- Pericenter identification
- Apocenter identification
- Dynamical friction signature
- Orbital tightening over time

This movie isolates large-scale orbital mechanics.
No particles rendered.
"""

output_movie_sep = "MW_M31_orbital_evolution.mp4"

fig4 = plt.figure(figsize=(8,8))
ax4 = fig4.add_subplot(111, projection='3d')

writer4 = FFMpegWriter(fps=fps)

# Storage for orbital trails
mw_trail = []
m31_trail = []

with writer4.saving(fig4, output_movie_sep, dpi=dpi):

    for snap in snapshots:

        MW  = CenterOfMass(f"MW_{snap:03d}.txt", 1)
        M31 = CenterOfMass(f"M31_{snap:03d}.txt", 1)

        # Compute COM positions independently
        mw_com = np.array(MW.COM_P())
        m31_com = np.array(M31.COM_P())

        mw_trail.append(mw_com)
        m31_trail.append(m31_com)

        mw_trail_arr = np.array(mw_trail)
        m31_trail_arr = np.array(m31_trail)

        separation = np.linalg.norm(mw_com - m31_com)

        ax4.clear()

        # Plot trajectory trails
        ax4.plot(mw_trail_arr[:,0],
                 mw_trail_arr[:,1],
                 mw_trail_arr[:,2])

        ax4.plot(m31_trail_arr[:,0],
                 m31_trail_arr[:,1],
                 m31_trail_arr[:,2])

        # Current positions
        ax4.scatter(*mw_com)
        ax4.scatter(*m31_com)

        ax4.set_xlim(-R_LIMIT*2, R_LIMIT*2)
        ax4.set_ylim(-R_LIMIT*2, R_LIMIT*2)
        ax4.set_zlim(-R_LIMIT*2, R_LIMIT*2)

        ax4.set_title(f"Orbital Evolution | Snapshot {snap:03d}\n"
                      f"Separation = {separation:.2f} kpc")

        writer4.grab_frame()
        print(f"Rendered orbital frame {snap:03d}")

print("Orbital movie complete.")
"""
MOVIE 5 & MOVIE 6
MW–M31 Halo Shape Metrics Visualization

Movie 5 → Volume Inflation Evolution (V_hull / V_ellipsoid)
Movie 6 → Axis Ratio Evolution (b/a and c/a time series)

These movies assume:
- MW_###.txt and M31_###.txt snapshot format
- CenterOfMass2 class
- Precomputation performed inside loop
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.spatial import ConvexHull
from CenterOfMass2 import CenterOfMass

# ============================================================
# CONFIGURATION
# ============================================================

snapshots = np.arange(0, 800)
fps = 20
dpi = 200

# Storage arrays (precompute metrics once)
b_over_a = []
c_over_a = []
volume_inflation = []

# ============================================================
# PHYSICS FUNCTIONS
# ============================================================

def inertia_tensor(pos, masses):
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
    I = inertia_tensor(pos, masses)
    eigvals, eigvecs = np.linalg.eigh(I)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]

    a = np.sqrt(eigvals[0])
    b = np.sqrt(eigvals[1])
    c = np.sqrt(eigvals[2])

    return a, b, c

# ============================================================
# PRECOMPUTATION LOOP
# ============================================================

print("Precomputing shape metrics...")

for snap in snapshots:

    MW  = CenterOfMass(f"MW_{snap:03d}.txt", 1)
    M31 = CenterOfMass(f"M31_{snap:03d}.txt", 1)

    x = np.concatenate((MW.x, M31.x))
    y = np.concatenate((MW.y, M31.y))
    z = np.concatenate((MW.z, M31.z))
    m = np.concatenate((MW.m, M31.m))

    pos = np.vstack((x,y,z)).T

    # COM centering
    xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
    pos -= np.array([xcom, ycom, zcom])

    # Inertia-based ellipsoid
    a, b, c = principal_axes(pos, m)
    V_ellipsoid = (4/3) * np.pi * a * b * c

    # Convex hull
    hull = ConvexHull(pos)
    V_hull = hull.volume

    b_over_a.append(b/a)
    c_over_a.append(c/a)
    volume_inflation.append(V_hull / V_ellipsoid)

    print(f"Processed snapshot {snap:03d}")

b_over_a = np.array(b_over_a)
c_over_a = np.array(c_over_a)
volume_inflation = np.array(volume_inflation)

print("Precomputation complete.")


# ============================================================
# ===================== MOVIE 5 ==============================
#      Volume Inflation Evolution (Hull / Ellipsoid)
# ============================================================

output_movie_vol = "MW_M31_volume_inflation.mp4"

fig5, ax5 = plt.subplots(figsize=(8,6))
writer5 = FFMpegWriter(fps=fps)

with writer5.saving(fig5, output_movie_vol, dpi=dpi):

    for i, snap in enumerate(snapshots):

        ax5.clear()

        ax5.plot(snapshots[:i+1], volume_inflation[:i+1])

        ax5.set_xlabel("Snapshot")
        ax5.set_ylabel("V_hull / V_ellipsoid")
        ax5.set_title("Volume Inflation Evolution")

        ax5.set_xlim(snapshots[0], snapshots[-1])
        ax5.set_ylim(0, np.max(volume_inflation)*1.1)

        writer5.grab_frame()

print("Volume inflation movie complete.")


# ============================================================
# ===================== MOVIE 6 ==============================
#        Axis Ratio Evolution (b/a and c/a)
# ============================================================

output_movie_axis = "MW_M31_axis_ratios.mp4"

fig6, ax6 = plt.subplots(figsize=(8,6))
writer6 = FFMpegWriter(fps=fps)

with writer6.saving(fig6, output_movie_axis, dpi=dpi):

    for i, snap in enumerate(snapshots):

        ax6.clear()

        ax6.plot(snapshots[:i+1], b_over_a[:i+1], label="b/a")
        ax6.plot(snapshots[:i+1], c_over_a[:i+1], label="c/a")

        ax6.set_xlabel("Snapshot")
        ax6.set_ylabel("Axis Ratio")
        ax6.set_title("Halo Axis Ratio Evolution")

        ax6.set_xlim(snapshots[0], snapshots[-1])
        ax6.set_ylim(0, 1.1)

        ax6.legend()

        writer6.grab_frame()

print("Axis ratio movie complete.")
