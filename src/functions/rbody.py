import numpy as np
from .Smtrx import Smtrx
from .Hmtrx import Hmtrx

def rbody(m, R44, R55, R66, nu2, r_bG):
    """
    Compute rigid-body mass matrix MRB and Coriolis matrix CRB.

    Parameters
    ----------
    m : float
        Vessel mass
    R44, R55, R66 : float
        Radii of gyration about CG
    nu2 : array_like (3,)
        Angular velocity vector [p, q, r]
    r_bG : array_like (3,)
        Vector from CO to CG

    Returns
    -------
    MRB : (6,6) ndarray
        Rigid body mass matrix about CO
    CRB : (6,6) ndarray
        Rigid body Coriolis matrix about CO
    """

    nu2 = np.asarray(nu2)
    r_bG = np.asarray(r_bG)

    I3 = np.eye(3)
    O3 = np.zeros((3, 3))

    # Inertia matrix about CG
    I_G = m * np.diag([R44**2, R55**2, R66**2])

    # Rigid body mass matrix about CG
    MRB_CG = np.block([
        [m * I3, O3],
        [O3, I_G]
    ])

    # Coriolis matrix about CG
    CRB_CG = np.block([
        [m * Smtrx(nu2), O3],
        [O3, -Smtrx(I_G @ nu2)]
    ])

    # Transform from CG to CO
    H = Hmtrx(r_bG)

    MRB = H.T @ MRB_CG @ H
    CRB = H.T @ CRB_CG @ H

    return MRB, CRB