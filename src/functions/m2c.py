import numpy as np
from .Smtrx import Smtrx


def m2c(M, nu):
    """
    Compute the Coriolis-centripetal matrix C(nu).

    Parameters
    ----------
    M : ndarray (6x6 or 3x3)
        System inertia matrix (MRB or MA)
    nu : ndarray (6,) or (3,)
        Velocity vector [u, v, w, p, q, r] or [u, v, r]

    Returns
    -------
    C : ndarray
        Coriolis-centripetal matrix
    """

    M = 0.5 * (M + M.T)  # symmetrize inertia matrix
    nu = np.asarray(nu)

    if len(nu) == 6:  # 6-DOF model

        M11 = M[0:3, 0:3]
        M12 = M[0:3, 3:6]
        M21 = M12.T
        M22 = M[3:6, 3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]

        nu1_dot = M11 @ nu1 + M12 @ nu2
        nu2_dot = M21 @ nu1 + M22 @ nu2

        C = np.block([
            [np.zeros((3, 3)), -Smtrx(nu1_dot)],
            [-Smtrx(nu1_dot), -Smtrx(nu2_dot)]
        ])

    else:  # 3-DOF model (surge, sway, yaw)

        C = np.array([
            [0, 0, -M[1, 1]*nu[1] - M[1, 2]*nu[2]],
            [0, 0,  M[0, 0]*nu[0]],
            [M[1, 1]*nu[1] + M[1, 2]*nu[2], -M[0, 0]*nu[0], 0]
        ])

    return C