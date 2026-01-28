import numpy as np
from typing import Callable

# ============================================================
# Core numerical tools
# ============================================================

def enclosed_mass(r: float, rho: Callable, n: int = 4000) -> float:
    """
    Compute enclosed mass M(<r) assuming spherical symmetry.

    Parameters
    ----------
    r : float
        Radius at which to compute enclosed mass
    rho : Callable
        Density function rho(r)
    n : int
        Number of radial integration points

    Returns
    -------
    float
        Enclosed mass within radius r
    """
    rs = np.linspace(0, r, n)[1:]     # avoid r = 0
    dr = rs[1] - rs[0]
    return 4 * np.pi * np.sum(rho(rs) * rs**2) * dr


# ============================================================
# Density profile definitions
# ============================================================

def rho_einasto(r, rho_minus2, r_minus2, alpha):
    """
    Einasto density profile.
    """
    x = r / r_minus2
    return rho_minus2 * np.exp(-(2.0 / alpha) * (x**alpha - 1))


def rho_nfw(r, rho_s, r_s):
    """
    Navarro–Frenk–White (NFW) profile.
    """
    x = r / r_s
    return rho_s / (x * (1 + x)**2)


def rho_hernquist(r, M, a):
    """
    Hernquist profile.
    """
    return (M / (2 * np.pi)) * (a / (r * (r + a)**3))


def rho_plummer(r, M, a):
    """
    Plummer sphere.
    """
    return (3 * M / (4 * np.pi * a**3)) * (1 + (r / a)**2)**(-2.5)


def rho_isothermal(r, sigma, G=1.0):
    """
    Singular isothermal sphere.
    """
    return sigma**2 / (2 * np.pi * G * r**2)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    r_test = 10.0

    rho_func = lambda r: rho_einasto(
        r,
        rho_minus2=1.0,
        r_minus2=5.0,
        alpha=0.2
    )

    M_r = enclosed_mass(r_test, rho_func)

    print(f"Enclosed mass at r = {r_test}: {M_r:.4e}")
