import numpy as np


def Tzyx(phi, theta):
    """
    Euler angle transformation matrix for the ZYX convention.

    Parameters
    ----------
    phi : float
        Roll angle (rad)
    theta : float
        Pitch angle (rad)

    Returns
    -------
    T : (3,3) ndarray
        Euler angle transformation matrix
    """

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)

    if np.isclose(cth, 0.0):
        raise ValueError("Tzyx is singular for theta = ±90 degrees")

    T = np.array([
        [1, sphi * sth / cth, cphi * sth / cth],
        [0, cphi,            -sphi],
        [0, sphi / cth,       cphi / cth]
    ])

    return T