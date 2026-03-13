# ===============================================================
# MOVIE — Velocity Anisotropy Profile Evolution
# ===============================================================
#
# PURPOSE
# -------
# Track the evolution of the velocity anisotropy parameter β(r)
# during the MW–M31 merger.
#
# β(r) measures whether particle motions are preferentially
# radial or tangential within radial shells.
#
# This movie shows how the anisotropy profile evolves as the
# merger proceeds.
#
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from CenterOfMass2 import CenterOfMass

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

snapshots = np.arange(0,800)

# radial bins (kpc)
r_bins = np.logspace(-1, 2.7, 40)

fps = 20
bitrate = 2000

# ---------------------------------------------------------------
# MOVIE SETUP
# ---------------------------------------------------------------

writer = FFMpegWriter(fps=fps, bitrate=bitrate)
fig, ax = plt.subplots(figsize=(7,5))

# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------

with writer.saving(fig, "movie_velocity_anisotropy.mp4", dpi=200):

    for snap in snapshots:

        # -------------------------------------------------------
        # LOAD SNAPSHOTS
        # -------------------------------------------------------

        mw_file  = f"MW_{snap:03d}.txt"
        m31_file = f"M31_{snap:03d}.txt"

        MW  = CenterOfMass(mw_file,1)
        M31 = CenterOfMass(m31_file,1)

        x = np.concatenate((MW.x, M31.x))
        y = np.concatenate((MW.y, M31.y))
        z = np.concatenate((MW.z, M31.z))

        vx = np.concatenate((MW.vx, M31.vx))
        vy = np.concatenate((MW.vy, M31.vy))
        vz = np.concatenate((MW.vz, M31.vz))

        m = np.concatenate((MW.m, M31.m))

        pos = np.vstack((x,y,z)).T
        vel = np.vstack((vx,vy,vz)).T

        # -------------------------------------------------------
        # CENTER SYSTEM
        # -------------------------------------------------------

        xcom,ycom,zcom = MW.COMdefine(x,y,z,m)
        vxcom,vycom,vzcom = MW.COMdefine(vx,vy,vz,m)

        pos -= np.array([xcom,ycom,zcom])
        vel -= np.array([vxcom,vycom,vzcom])

        # -------------------------------------------------------
        # COMPUTE RADIAL COORDINATES
        # -------------------------------------------------------

        r = np.linalg.norm(pos, axis=1)

        # radial velocity component
        vr = np.sum(pos * vel, axis=1) / r

        # tangential velocity magnitude
        vt = np.sqrt(np.sum(vel**2,axis=1) - vr**2)

        # -------------------------------------------------------
        # COMPUTE ANISOTROPY PROFILE
        # -------------------------------------------------------

        beta_profile = np.zeros(len(r_bins)-1)

        for i in range(len(r_bins)-1):

            mask = (r >= r_bins[i]) & (r < r_bins[i+1])

            if np.sum(mask) > 10:

                vr_bin = vr[mask]
                vt_bin = vt[mask]

                sigma_r = np.std(vr_bin)
                sigma_t = np.std(vt_bin)

                if sigma_r > 0:
                    beta_profile[i] = 1 - (sigma_t**2)/(2*sigma_r**2)
                else:
                    beta_profile[i] = 0

        # radial midpoints
        r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])

        # -------------------------------------------------------
        # PLOT FRAME
        # -------------------------------------------------------

        ax.cla()

        ax.plot(r_mid, beta_profile)

        ax.set_xscale("log")

        ax.set_ylim(-1,1)

        ax.set_xlabel("Radius [kpc]")
        ax.set_ylabel("Velocity Anisotropy β")

        ax.set_title(f"Velocity Anisotropy Evolution (Snapshot {snap})")

        writer.grab_frame()

plt.close()

print("Velocity anisotropy movie completed.")
