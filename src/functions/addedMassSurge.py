import numpy as np


def addedMassSurge(m, L, rho=1025):
    """
    Approximate the added mass in surge using Söding (1982):

        A11 = -Xudot = 2.7 * rho * nabla^(5/3) / L^2

    Parameters
    ----------
    m : float
        Mass of the vessel (kg)
    L : float
        Length of the vessel (m)
    rho : float, optional
        Density of water (kg/m^3), default is 1025

    Returns
    -------
    A11 : float
        Added mass in surge (kg)
    ratio : float
        Ratio of added mass to actual mass (A11 / m)
    """

    nabla = m / rho                      # Volume displacement
    A11 = 2.7 * rho * nabla**(5/3) / L**2
    ratio = A11 / m

    return A11, ratio