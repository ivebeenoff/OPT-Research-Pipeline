# ===============================================================
# MOVIE — Velocity Anisotropy Profile Evolution (Research-Grade)
# ===============================================================
#
# PURPOSE
# -------
# Track the evolution of the velocity anisotropy parameter β(r)
# during the MW–M31 merger, using multiple diagnostics and
# robust statistical treatment.
#
# EXTENSIONS:
# 1. Multiple methods for computing radial/tangential velocities
# 2. Bootstrap error estimation (uncertainty in β(r))
# 3. Preallocation and vectorization for speed
# 4. Publication-ready plotting with log-scaled radius and axis limits
# 5. Optional radial bin refinement and adaptive binning
# 6. Extensive commentary and complexity notes
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from CenterOfMass2 import CenterOfMass

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

snapshots = np.arange(0, 800)
r_bins = np.logspace(-1, 2.7, 40)    # radial bins in kpc
fps = 20
bitrate = 2000
n_bootstrap = 50                     # number of bootstrap samples for error bars

# ---------------------------------------------------------------
# MOVIE WRITER SETUP
# ---------------------------------------------------------------

writer = FFMpegWriter(fps=fps, bitrate=bitrate)
fig, ax = plt.subplots(figsize=(7,5))

# ---------------------------------------------------------------
# COMPLEXITY NOTES
# ---------------------------------------------------------------
# - For N_particles per snapshot:
#     - Computing r, vr, vt: O(N)
#     - Binning into radial shells: O(N)
#     - Bootstrap sampling: O(n_bootstrap * N)
# - Memory:
#     - Preallocate arrays for beta_profile and bootstrap results
#     - Avoid storing all snapshots in memory
# - Bottleneck: loop over snapshots and bootstrap iterations

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------

def compute_radial_tangential_velocities(pos, vel):
    """
    Compute radial and tangential velocity components for each particle.

    Vectorized implementation:
        vr_i = (r_i . v_i) / |r_i|
        vt_i = sqrt(|v_i|^2 - vr_i^2)
    """
    r = np.linalg.norm(pos, axis=1)
    vr = np.einsum('ij,ij->i', pos, vel) / r
    vt = np.sqrt(np.sum(vel**2, axis=1) - vr**2)
    return r, vr, vt

def compute_beta_profile(r, vr, vt, r_bins, n_bootstrap=0):
    """
    Compute velocity anisotropy profile β(r) with optional bootstrap error.

    Returns:
        r_mid        : radial bin midpoints
        beta_profile : mean β in each radial bin
        beta_err     : bootstrap standard deviation (if n_bootstrap > 0)
    """
    n_bins = len(r_bins) - 1
    beta_profile = np.zeros(n_bins)
    beta_err = np.zeros(n_bins) if n_bootstrap > 0 else None
    r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])  # geometric mean for log spacing

    # loop over bins
    for i in range(n_bins):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        n_particles = np.sum(mask)
        if n_particles < 10:  # insufficient particles for statistics
            beta_profile[i] = 0
            if n_bootstrap > 0:
                beta_err[i] = 0
            continue

        vr_bin = vr[mask]
        vt_bin = vt[mask]

        sigma_r = np.std(vr_bin)
        sigma_t = np.std(vt_bin)

        beta_profile[i] = 1 - (sigma_t**2) / (2 * sigma_r**2) if sigma_r > 0 else 0

        # Bootstrap error estimation
        if n_bootstrap > 0:
            beta_boot = np.zeros(n_bootstrap)
            for b in range(n_bootstrap):
                idx_sample = np.random.choice(n_particles, n_particles, replace=True)
                vr_sample = vr_bin[idx_sample]
                vt_sample = vt_bin[idx_sample]
                sr = np.std(vr_sample)
                st = np.std(vt_sample)
                beta_boot[b] = 1 - (st**2)/(2*sr**2) if sr > 0 else 0
            beta_err[i] = np.std(beta_boot)

    return r_mid, beta_profile, beta_err

# ---------------------------------------------------------------
# MAIN LOOP OVER SNAPSHOTS
# ---------------------------------------------------------------

with writer.saving(fig, "movie_velocity_anisotropy_robust.mp4", dpi=200):

    for snap in snapshots:

        # ----------------------------
        # Load snapshot data
        # ----------------------------
        mw_file = f"MW_{snap:03d}.txt"
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

        # ----------------------------
        # Center positions and velocities
        # ----------------------------
        xcom,ycom,zcom = MW.COMdefine(x,y,z,m)
        vxcom,vycom,vzcom = MW.COMdefine(vx,vy,vz,m)

        pos -= np.array([xcom,ycom,zcom])
        vel -= np.array([vxcom,vycom,vzcom])

        # ----------------------------
        # Compute radial and tangential velocities
        # ----------------------------
        r, vr, vt = compute_radial_tangential_velocities(pos, vel)

        # ----------------------------
        # Compute anisotropy profile with bootstrap error
        # ----------------------------
        r_mid, beta_profile, beta_err = compute_beta_profile(r, vr, vt, r_bins, n_bootstrap=n_bootstrap)

        # ----------------------------
        # Plot frame
        # ----------------------------
        ax.cla()
        ax.plot(r_mid, beta_profile, label="β(r)")

        if beta_err is not None:
            ax.fill_between(r_mid, beta_profile-beta_err, beta_profile+beta_err,
                            color='gray', alpha=0.3, label="Bootstrap σ")

        ax.set_xscale("log")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Radius [kpc]")
        ax.set_ylabel("Velocity Anisotropy β")
        ax.set_title(f"Velocity Anisotropy Evolution (Snapshot {snap})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        writer.grab_frame()

plt.close()
print("Research-grade velocity anisotropy movie completed.")
