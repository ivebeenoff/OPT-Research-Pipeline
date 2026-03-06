# ===============================================================
# MOVIE MODULES 7–8
# Angular Momentum Vector Evolution
# Spin Parameter Evolution
#
# These movies visualize dynamical quantities computed earlier
# in the MW–M31 merger analysis pipeline.
#
# Required Inputs (already produced earlier in the pipeline):
#
# j_spec_matrix : (N_snapshots, 3)
#     Specific angular momentum vector for each snapshot
#
# spin_param : (N_snapshots,)
#     Bullock spin parameter for each snapshot
#
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


# ===============================================================
# MOVIE 7 — Angular Momentum Vector Evolution
# ===============================================================
#
# Physical Meaning
# ----------------
# The specific angular momentum vector
#
#     j = L / M
#
# tracks the *orientation and magnitude* of the halo's rotation.
#
# During a major merger we expect:
#
# • orbital angular momentum transfer
# • tidal torques
# • spin flips
# • relaxation into a new equilibrium orientation
#
# This movie visualizes the trajectory of the angular momentum
# vector through time in 3-dimensional phase space.
#
# ===============================================================

def make_angular_momentum_vector_movie(j_spec_matrix,
                                       output_file="movie7_angular_momentum_vector.mp4",
                                       fps=20):

    n_snaps = j_spec_matrix.shape[0]

    # -----------------------------------------------------------
    # Determine symmetric axis limits
    # -----------------------------------------------------------
    # We compute the maximum magnitude of the vector components
    # so the trajectory remains centered.
    #
    # Complexity:
    # O(N_snapshots)
    #
    lim = np.max(np.abs(j_spec_matrix)) * 1.1

    # -----------------------------------------------------------
    # Create figure
    # -----------------------------------------------------------
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    writer = FFMpegWriter(fps=fps, bitrate=2000)

    with writer.saving(fig, output_file, dpi=200):

        # -------------------------------------------------------
        # Frame generation loop
        # -------------------------------------------------------
        #
        # Complexity per frame:
        # O(i) because we plot the trajectory up to frame i.
        #
        # Total complexity:
        #
        # Sum_{i=1..N} O(i) = O(N^2)
        #
        # However N = 800 snapshots which is small,
        # so runtime remains negligible.
        #
        for i in range(n_snaps):

            ax.cla()

            ax.set_title("Specific Angular Momentum Vector Evolution")

            ax.set_xlabel("j_x")
            ax.set_ylabel("j_y")
            ax.set_zlabel("j_z")

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)

            # ---------------------------------------------------
            # Plot trajectory history
            # ---------------------------------------------------
            ax.plot(
                j_spec_matrix[:i+1,0],
                j_spec_matrix[:i+1,1],
                j_spec_matrix[:i+1,2],
                linewidth=2
            )

            # Current vector location
            ax.scatter(
                j_spec_matrix[i,0],
                j_spec_matrix[i,1],
                j_spec_matrix[i,2],
                s=80
            )

            writer.grab_frame()

    plt.close()




# ===============================================================
# MOVIE 8 — Spin Parameter Evolution
# ===============================================================
#
# Physical Meaning
# ----------------
#
# Bullock spin parameter:
#
#     λ' = j / ( √2 * Vvir * Rvir )
#
# where
#
#     j     = specific angular momentum
#     Vvir  = virial velocity
#     Rvir  = virial radius
#
# This dimensionless parameter measures the degree of rotational
# support of the dark matter halo.
#
# Cosmological simulations typically find
#
#     λ' ≈ 0.02 – 0.05
#
# Major mergers often produce spikes or fluctuations in spin
# due to tidal torques and angular momentum redistribution.
#
# ===============================================================

def make_spin_parameter_movie(spin_param,
                              output_file="movie8_spin_parameter.mp4",
                              fps=20):

    n_snaps = len(spin_param)

    fig, ax = plt.subplots(figsize=(7,5))

    writer = FFMpegWriter(fps=fps, bitrate=2000)

    with writer.saving(fig, output_file, dpi=200):

        # -------------------------------------------------------
        # Frame generation loop
        # -------------------------------------------------------
        #
        # Each frame plots the history of spin up to time i.
        #
        # Time complexity per frame:
        #
        # O(i)
        #
        # Total runtime complexity:
        #
        # O(N_snapshots^2)
        #
        # Again, with N≈800 this is trivial.
        #
        for i in range(n_snaps):

            ax.cla()

            ax.set_title("Halo Spin Parameter Evolution")

            ax.set_xlabel("Snapshot")
            ax.set_ylabel("Spin Parameter λ'")

            ax.set_xlim(0, n_snaps)
            ax.set_ylim(0, np.max(spin_param)*1.2)

            ax.plot(
                np.arange(i+1),
                spin_param[:i+1],
                linewidth=2
            )

            writer.grab_frame()

    plt.close()



# ===============================================================
# EXAMPLE USAGE
# ===============================================================
#
# After the main simulation pipeline has produced the
# angular momentum matrix and spin parameter array:
#
# make_angular_momentum_vector_movie(j_spec_matrix)
#
# make_spin_parameter_movie(spin_param)
#
# ===============================================================
