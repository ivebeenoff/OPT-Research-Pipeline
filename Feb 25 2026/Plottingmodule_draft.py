"""
MW–M31 Halo Shape Evolution – Analysis & Plotting Module
=========================================================

Scientific Goal
---------------
This module generates publication-quality figures that quantify the
morphological evolution of the Milky Way (MW) and Andromeda (M31) dark
matter halos during their interaction and eventual merger.

The central hypothesis being tested is:

    Halo shapes evolve systematically as a function of
    MW–M31 separation due to tidal torques and anisotropic mass accretion.

Key diagnostics:
    • Axis ratios (b/a, c/a)
    • Triaxiality parameter
    • Convex hull volume inflation
    • Method comparison (inertia tensor vs convex hull)
    • Correlation with MW–M31 separation

Why This Matters Astrophysically
--------------------------------
Halo morphology encodes dynamical history.
If halo flattening correlates with separation, it provides evidence that:

    • tidal interactions reshape halos before merger
    • inertia-tensor methods may underestimate halo anisotropy
    • convex-hull geometry captures tidal streams / outskirts better

These are publishable results if statistically robust.

Designed For
------------
• Computational astrophysics workflow
• Spyder / Jupyter environment
• NumPy-optimized research pipeline
• RNASS / ApJ / MNRAS figure standards

Reproducibility Notes
---------------------
All figures are generated deterministically from an HDF5 metrics file.
No manual data handling is required, ensuring reproducibility.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# Publication plotting style
# ---------------------------------------------------------

"""
Why styling matters:
--------------------
Referees judge clarity immediately. Poor figures = desk rejection.

These parameters enforce:

• readable fonts
• consistent sizing
• vector-graphics compatibility
• grayscale-print readability
"""

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (7,5),   # Standard journal column width
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------
# Utility: Load metrics
# ---------------------------------------------------------

def load_metrics(hdf5_file):
    """
    Load precomputed halo metrics from HDF5.

    Scientific Context
    ------------------
    Metrics are assumed to be generated from a prior pipeline step
    that computed halo shapes using:

        • inertia tensor eigenvalues
        • convex hull geometry
        • halo volume estimates

    Keeping analysis separate from simulation is critical for:

        ✔ reproducibility
        ✔ pipeline modularity
        ✔ independent verification

    Returns
    -------
    data : dict
        Numerical arrays of halo metrics.
    metadata : dict
        Simulation parameters (snapshot spacing, units, etc.)
    """

    with h5py.File(hdf5_file, "r") as f:
        data = {key: f[key][:] for key in f.keys()}
        metadata = dict(f.attrs)

    return data, metadata


# ---------------------------------------------------------
# Optional: Compute MW–M31 separation
# ---------------------------------------------------------

def compute_separation(snapshot_range, data_dir="."):
    """
    Compute MW–M31 center-of-mass separation.

    Astrophysical Motivation
    ------------------------
    Halo deformation should correlate with tidal field strength,
    which scales ~ 1 / separation^3.

    Measuring shape vs separation allows us to test whether
    tidal forces drive halo triaxiality evolution.

    Notes
    -----
    If separation is already stored in the metrics file,
    use that instead to avoid recomputation noise.
    """

    from CenterOfMass2 import CenterOfMass

    separation = []

    for s in snapshot_range:
        mw_file = os.path.join(data_dir, f"MW_{s:03d}.txt")
        m31_file = os.path.join(data_dir, f"M31_{s:03d}.txt")

        MW = CenterOfMass(mw_file, 1)
        M31 = CenterOfMass(m31_file, 1)

        mw_com = MW.COMdefine(MW.x, MW.y, MW.z, MW.m)
        m31_com = M31.COMdefine(M31.x, M31.y, M31.z, M31.m)

        d = np.linalg.norm(np.array(mw_com) - np.array(m31_com))
        separation.append(d)

    return np.array(separation)


# ---------------------------------------------------------
# Figure 1: Axis Ratio Evolution
# ---------------------------------------------------------

def plot_axis_ratio_evolution(data, snapshot_range, output_prefix="fig1_axis_ratios"):
    """
    Plot axis ratios vs time.

    Scientific Meaning
    ------------------
    Axis ratios measure halo flattening.

    c/a ↓  → halo becomes more oblate/prolate
    b/a ↓  → halo becomes more triaxial

    Comparing inertia vs convex hull reveals whether
    outer tidal debris biases shape estimates.
    """

    plt.figure()
    plt.plot(snapshot_range, data["ba_inertia"], label="Inertia b/a")
    plt.plot(snapshot_range, data["ca_inertia"], label="Inertia c/a")
    plt.plot(snapshot_range, data["ba_hull"], "--", label="Hull b/a")
    plt.plot(snapshot_range, data["ca_hull"], "--", label="Hull c/a")

    plt.xlabel("Snapshot")
    plt.ylabel("Axis Ratio")
    plt.title("Halo Axis Ratio Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_prefix + ".pdf")
    plt.savefig(output_prefix + ".svg")
    plt.close()


# ---------------------------------------------------------
# Figure 2: Volume Inflation
# ---------------------------------------------------------

def plot_volume_inflation(data, snapshot_range, output_prefix="fig2_volume_inflation"):
    """
    Plot convex hull / ellipsoid volume ratio.

    Interpretation
    --------------
    Inflation > 1 indicates:

        • tidal streams
        • asymmetry
        • non-ellipsoidal structure

    This is strong evidence of interaction-induced deformation.
    """

    plt.figure()
    plt.plot(snapshot_range, data["volume_inflation"])
    plt.xlabel("Snapshot")
    plt.ylabel("Hull Volume / Ellipsoid Volume")
    plt.title("Halo Volume Inflation Factor")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_prefix + ".pdf")
    plt.savefig(output_prefix + ".svg")
    plt.close()


# ---------------------------------------------------------
# Figure 3: Triaxiality Evolution
# ---------------------------------------------------------

def plot_triaxiality(data, snapshot_range, output_prefix="fig3_triaxiality"):
    """
    Plot triaxiality parameter.

    Definition
    ----------
    T = (a^2 − b^2) / (a^2 − c^2)

    T ≈ 0 → Oblate
    T ≈ 1 → Prolate
    T ≈ 0.5 → Fully triaxial

    Tracking T over time reveals dynamical evolution
    during MW–M31 interaction.
    """

    plt.figure()
    plt.plot(snapshot_range, data["triax_inertia"], label="Inertia T")
    plt.plot(snapshot_range, data["triax_hull"], "--", label="Hull T")

    plt.xlabel("Snapshot")
    plt.ylabel("Triaxiality T")
    plt.title("Halo Triaxiality Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_prefix + ".pdf")
    plt.savefig(output_prefix + ".svg")
    plt.close()


# ---------------------------------------------------------
# Figure 4: Axis Ratio vs Separation
# ---------------------------------------------------------

def plot_axis_vs_separation(data, separation, output_prefix="fig4_axis_vs_sep"):
    """
    Correlate halo flattening with MW–M31 separation.

    If correlation exists, it supports tidal interaction hypothesis.

    Expected trend:
        separation ↓ → flattening ↑
    """

    plt.figure()
    plt.scatter(separation, data["ca_inertia"], label="Inertia c/a", alpha=0.7)
    plt.scatter(separation, data["ca_hull"], label="Hull c/a", alpha=0.7)

    plt.xlabel("MW–M31 Separation")
    plt.ylabel("c/a")
    plt.title("Halo Flattening vs Separation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_prefix + ".pdf")
    plt.savefig(output_prefix + ".svg")
    plt.close()


# ---------------------------------------------------------
# Figure 5: Method Comparison
# ---------------------------------------------------------

def plot_method_comparison(data, output_prefix="fig5_method_comparison"):
    """
    Compare inertia vs convex hull shape estimates.

    Purpose
    -------
    Tests methodological bias.

    If hull systematically gives lower c/a,
    inertia tensor may underestimate tidal distortion.
    """

    plt.figure()
    plt.scatter(data["ca_inertia"], data["ca_hull"], alpha=0.7)

    plt.xlabel("Inertia c/a")
    plt.ylabel("Hull c/a")
    plt.title("Method Comparison: Inertia vs Convex Hull")

    x = np.linspace(0,1,100)
    plt.plot(x, x, linestyle="--")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_prefix + ".pdf")
    plt.savefig(output_prefix + ".svg")
    plt.close()


# ---------------------------------------------------------
# Master function
# ---------------------------------------------------------

def generate_all_figures(hdf5_file, snapshot_range, compute_sep=False, data_dir="."):
    """
    Generate full figure suite.

    Publication Workflow
    --------------------
    Typical usage in Jupyter / Spyder:

        generate_all_figures("halo_metrics.h5", np.arange(1,801))

    Output files are vector graphics suitable for LaTeX inclusion.

    This function ensures:
        ✔ reproducibility
        ✔ consistent figure styling
        ✔ automated analysis pipeline
    """

    data, metadata = load_metrics(hdf5_file)

    plot_axis_ratio_evolution(data, snapshot_range)
    plot_volume_inflation(data, snapshot_range)
    plot_triaxiality(data, snapshot_range)
    plot_method_comparison(data)

    if compute_sep:
        separation = compute_separation(snapshot_range, data_dir)
        plot_axis_vs_separation(data, separation)

    print("All figures generated successfully.")
