import numpy as np

from .addedMassSurge import addedMassSurge


def forceSurgeDamping(flag, u_r, m, S, L, T1, rho, u_max, thrust_max=None):
    """
    Compute hydrodynamic damping in surge.

    Parameters
    ----------
    flag : int
        1 = plot damping terms, 0 = numerical output only
    u_r : float or ndarray
        Relative surge velocity (m/s)
    m : float
        Mass (kg)
    S : float
        Wetted surface area (m^2)
    L : float
        Vessel length (m)
    T1 : float
        Surge time constant (s)
    rho : float
        Water density (kg/m^3)
    u_max : float
        Maximum surge speed (m/s)
    thrust_max : float, optional
        Maximum thrust corresponding to u_max

    Returns
    -------
    X : float or ndarray
        Total surge damping force (N)
    Xuu : float
        Quadratic damping coefficient
    Xu : float
        Linear damping coefficient
    """

    u_cross = 2.0  # crossover speed

    # Linear damping coefficient
    Xudot = -addedMassSurge(m, L, rho)[0]
    Xu = -(m - Xudot) / T1

    # Quadratic damping coefficient
    if thrust_max is not None:
        Xuu = -thrust_max / (u_max**2)
    else:
        nu_kin = 1e-6
        k = 0.1
        eps = 1e-10

        Rn = (L / nu_kin) * np.abs(u_r)
        Cf = 0.075 / (np.log10(Rn + eps) - 2) ** 2
        Xuu = -0.5 * rho * S * (1 + k) * Cf

    # Blending function
    sigma = 1 - np.tanh(u_r / u_cross)

    # Damping force
    X = sigma * Xu * u_r + (1 - sigma) * Xuu * np.abs(u_r) * u_r

    return X, Xuu, Xu