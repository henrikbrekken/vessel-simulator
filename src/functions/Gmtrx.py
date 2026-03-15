import numpy as np
from .Hmtrx import Hmtrx

def Gmtrx(nabla, A_wp, GMT, GML, LCF, r_bp):
    """
    Compute the 6x6 hydrostatic stiffness matrix for a floating vessel.

    Parameters
    ----------
    nabla : float
        Volume displacement (m^3)
    A_wp : float
        Waterplane area (m^2)
    GMT : float
        Transverse metacentric height (m)
    GML : float
        Longitudinal metacentric height (m)
    LCF : float
        x-coordinate from CO to center of flotation (m)
    r_bp : array-like (3,)
        Location of point P relative to CO [x_p, y_p, z_p]

    Returns
    -------
    G : (6,6) ndarray
        Hydrostatic stiffness matrix
    """

    rho = 1025.0
    g = 9.81

    r_bp = np.asarray(r_bp)

    # Center of flotation location
    r_bf = np.array([LCF, 0.0, 0.0])

    # Hydrostatic coefficients at CF
    G33_CF = rho * g * A_wp
    G44_CF = rho * g * nabla * GMT
    G55_CF = rho * g * nabla * GML

    G_CF = np.diag([0, 0, G33_CF, G44_CF, G55_CF, 0])

    # Transform CF → CO
    H_cf = Hmtrx(r_bf)
    G_CO = H_cf.T @ G_CF @ H_cf

    # Transform CO → arbitrary point P
    H_p = Hmtrx(r_bp)
    G = H_p.T @ G_CO @ H_p

    return G