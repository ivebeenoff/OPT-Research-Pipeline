# ===============================================================
# MOVIE 10 — Velocity Dispersion Evolution
# ===============================================================
#
# PURPOSE
# -------
# Visualize the evolution of the velocity dispersion of the
# combined MW–M31 dark matter halo during the merger process.
#
# Velocity dispersion is a measure of the random kinetic motion
# of particles and is defined as the standard deviation of
# particle velocities relative to the mean velocity of the system.
#
# This movie produces a 2D spatial map of velocity dispersion
# projected onto the XY plane for each snapshot.
#
# PHYSICAL INTERPRETATION
# -----------------------
# Velocity dispersion highlights:
#
# • dynamical heating during close encounters
# • merger-driven energy redistribution
# • relaxation of the merged halo
#
# Regions of high dispersion indicate dynamically hot regions
# where particle velocities are highly randomized.
#
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from CenterOfMass2 import CenterOfMass

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

snapshots = np.arange(0, 800)

# spatial region to visualize (kpc)
xlim = (-300, 300)
ylim = (-300, 300)

# grid resolution
bins = 200

fps = 20
bitrate = 2000

# ---------------------------------------------------------------
# MOVIE WRITER SETUP
# ---------------------------------------------------------------

writer = FFMpegWriter(fps=fps, bitrate=bitrate)
fig, ax = plt.subplots(figsize=(7,7))

# ---------------------------------------------------------------
# MAIN SNAPSHOT LOOP
# ---------------------------------------------------------------

with writer.saving(fig, "movie10_velocity_dispersion.mp4", dpi=200):

    for snap in snapshots:

        # -------------------------------------------------------
        # LOAD SNAPSHOT DATA
        # -------------------------------------------------------

        mw_file  = f"MW_{snap:03d}.txt"
        m31_file = f"M31_{snap:03d}.txt"

        MW  = CenterOfMass(mw_file, 1)
        M31 = CenterOfMass(m31_file, 1)

        # combine particle sets
        x = np.concatenate((MW.x, MW.x))
        y = np.concatenate((MW.y, M31.y))
        z = np.concatenate((MW.z, M31.z))

        vx = np.concatenate((MW.vx, M31.vx))
        vy = np.concatenate((MW.vy, M31.vy))
        vz = np.concatenate((MW.vz, M31.vz))

        m = np.concatenate((MW.m, M31.m))

        # -------------------------------------------------------
        # CENTER THE SYSTEM
        # -------------------------------------------------------

        xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
        vxcom, vycom, vzcom = MW.COMdefine(vx, vy, vz, m)

        x -= xcom
        y -= ycom
        z -= zcom

        vx -= vxcom
        vy -= vycom
        vz -= vzcom

        # -------------------------------------------------------
        # COMPUTE PARTICLE SPEEDS
        # -------------------------------------------------------

        v = np.sqrt(vx**2 + vy**2 + vz**2)

        # -------------------------------------------------------
        # GRID SETUP
        # -------------------------------------------------------

        xedges = np.linspace(xlim[0], xlim[1], bins)
        yedges = np.linspace(ylim[0], ylim[1], bins)

        disp_map = np.zeros((bins-1, bins-1))

        # -------------------------------------------------------
        # COMPUTE VELOCITY DISPERSION PER CELL
        # -------------------------------------------------------
        #
        # For each spatial cell we compute:
        #
        # σ = sqrt( <v²> - <v>² )
        #
        # Complexity:
        #
        # O(N_particles)
        #
        # because each particle contributes to one grid cell.
        #

        for i in range(bins-1):
            for j in range(bins-1):

                mask = (
                    (x >= xedges[i]) & (x < xedges[i+1]) &
                    (y >= yedges[j]) & (y < yedges[j+1])
                )

                if np.any(mask):

                    vcell = v[mask]

                    mean_v = np.mean(vcell)
                    mean_v2 = np.mean(vcell**2)

                    sigma = np.sqrt(mean_v2 - mean_v**2)

                    disp_map[i, j] = sigma

        # -------------------------------------------------------
        # LOG SCALE FOR VISUALIZATION
        # -------------------------------------------------------

        disp_map = np.log10(disp_map + 1e-5)

        # -------------------------------------------------------
        # PLOT FRAME
        # -------------------------------------------------------

        ax.cla()

        im = ax.imshow(
            disp_map.T,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            aspect="equal",
            cmap="plasma"
        )

        ax.set_title(f"Velocity Dispersion Evolution (Snapshot {snap})")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")

        writer.grab_frame()

plt.close()

print("Velocity dispersion movie completed.")
