# ===============================================================
# MOVIE 9 — Halo Density Projection Evolution (Multi-Method)
# ===============================================================
#
# PURPOSE
# -------
# Visualize the MW–M31 dark matter halo density evolution using
# multiple density estimation methods:
#
# 1. 2D histogram (baseline)
# 2. Gaussian Kernel Density Estimation (KDE)
# 3. SPH-inspired adaptive smoothing (optional)
#
# This allows comparison of methods and highlights different
# structural features, such as tidal streams vs core density.
#
# Projection: XY plane of halo particles
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.stats import gaussian_kde
from CenterOfMass2 import CenterOfMass

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

snapshots = np.arange(0, 800)  # all snapshots
xlim = (-300, 300)             # spatial limits in kpc
ylim = (-300, 300)
bins = 300                     # grid resolution
method = "histogram"           # default density method: "histogram", "kde", "sph"
fps = 20
bitrate = 2000

# ---------------------------------------------------------------
# Movie writer setup
# ---------------------------------------------------------------

writer = FFMpegWriter(fps=fps, bitrate=bitrate)
fig, ax = plt.subplots(figsize=(7, 7))

# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------

with writer.saving(fig, "movie9_density_multi_method.mp4", dpi=200):

    for snap in snapshots:

        # ---------------------------
        # LOAD SNAPSHOT DATA
        # ---------------------------
        mw_file  = f"MW_{snap:03d}.txt"
        m31_file = f"M31_{snap:03d}.txt"

        MW  = CenterOfMass(mw_file, 1)
        M31 = CenterOfMass(m31_file, 1)

        # Concatenate particle positions
        x = np.concatenate((MW.x, M31.x))
        y = np.concatenate((MW.y, M31.y))
        z = np.concatenate((MW.z, M31.z))
        m = np.concatenate((MW.m, M31.m))

        # ---------------------------
        # CENTER THE SYSTEM
        # ---------------------------
        xcom, ycom, zcom = MW.COMdefine(x, y, z, m)
        x -= xcom
        y -= ycom
        z -= zcom

        # ---------------------------
        # COMPUTE DENSITY
        # ---------------------------

        if method == "histogram":
            # 2D histogram
            H, xedges, yedges = np.histogram2d(
                x, y, bins=bins, range=[xlim, ylim]
            )
            # log scaling
            density_map = np.log10(H + 1)
            # Complexity: O(N_particles)
            # Memory: bins^2

        elif method == "kde":
            # Gaussian KDE
            positions = np.vstack([x, y])
            kde = gaussian_kde(positions)
            X, Y = np.meshgrid(
                np.linspace(xlim[0], xlim[1], bins),
                np.linspace(ylim[0], ylim[1], bins)
            )
            grid_coords = np.vstack([X.ravel(), Y.ravel()])
            density_map = kde(grid_coords).reshape(bins, bins)
            density_map = np.log10(density_map + 1e-6)
            # Complexity: O(N_particles * bins^2)
            # Memory: bins^2

        elif method == "sph":
            # SPH-like smoothing using Gaussian kernel per particle
            # (simplified version)
            X, Y = np.meshgrid(
                np.linspace(xlim[0], xlim[1], bins),
                np.linspace(ylim[0], ylim[1], bins)
            )
            density_map = np.zeros_like(X)
            sigma = 5.0  # smoothing length in kpc
            for xi, yi in zip(x, y):
                density_map += np.exp(-((X - xi)**2 + (Y - yi)**2) / (2 * sigma**2))
            density_map = np.log10(density_map + 1e-6)
            # Complexity: O(N_particles * bins^2)
            # Memory: bins^2

        # ---------------------------
        # PLOT DENSITY MAP
        # ---------------------------
        ax.cla()
        im = ax.imshow(
            density_map.T,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            aspect="equal",
            cmap="inferno"
        )
        ax.set_title(f"MW–M31 Halo Density Evolution\nSnapshot {snap} | Method: {method}")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")

        writer.grab_frame()

plt.close()
print("Density evolution movie (multi-method) completed.")
