import numpy as np

def Dmtrx(T_126, zeta_45, MRB, MA, hydrostatics):
    """
    Compute the 6x6 linear damping matrix for marine craft.

    Parameters
    ----------
    T_126 : array-like
        [T1, T2, T6] time constants for DOFs 1, 2, and 6
    zeta_45 : array-like
        [zeta4, zeta5] relative damping ratios in DOFs 4 and 5
    MRB : (6,6) ndarray
        Rigid-body mass matrix
    MA : (6,6) ndarray
        Added mass matrix
    hydrostatics : ndarray
        Either:
        - 6x6 hydrostatic restoring matrix G (surface craft)
        - vector [W, r_bg(3), r_bb(3)] for submerged vehicles

    Returns
    -------
    D : (6,6) ndarray
        Linear damping matrix
    """

    M = MRB + MA

    T1, T2, T6 = T_126
    zeta4, zeta5 = zeta_45

    hydrostatics = np.asarray(hydrostatics)

    # Submerged vehicle case
    if hydrostatics.ndim == 1:
        W = hydrostatics[0]
        r_bg = hydrostatics[1:4]
        r_bb = hydrostatics[4:7]

        T3 = T2  # assume same time constant in sway and heave

        w4 = np.sqrt(W * (r_bg[2] - r_bb[2]) / M[3, 3])
        w5 = np.sqrt(W * (r_bg[2] - r_bb[2]) / M[4, 4])

        D = np.diag([
            M[0, 0] / T1,
            M[1, 1] / T2,
            M[2, 2] / T3,
            M[3, 3] * 2 * zeta4 * w4,
            M[4, 4] * 2 * zeta5 * w5,
            M[5, 5] / T6
        ])

    # Surface vessel case
    else:
        G33 = hydrostatics[2, 2]
        G44 = hydrostatics[3, 3]
        G55 = hydrostatics[4, 4]

        zeta3 = 0.2

        w3 = np.sqrt(G33 / M[2, 2])
        w4 = np.sqrt(G44 / M[3, 3])
        w5 = np.sqrt(G55 / M[4, 4])

        D = np.diag([
            M[0, 0] / T1,
            M[1, 1] / T2,
            M[2, 2] * 2 * zeta3 * w3,
            M[3, 3] * 2 * zeta4 * w4,
            M[4, 4] * 2 * zeta5 * w5,
            M[5, 5] / T6
        ])

    return D