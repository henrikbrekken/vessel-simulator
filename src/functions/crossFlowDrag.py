import numpy as np

from .Hoerner import Hoerner
from .cylinderDrag import cylinderDrag


def crossFlowDrag(L, B, T, nu_r, drag_model="Hoerner"):
    """
    Compute cross-flow drag integrals using strip theory.

    Parameters
    ----------
    L : float
        Vessel length (m)
    B : float
        Beam (m)
    T : float
        Draft (m)
    nu_r : array-like (6,)
        Relative velocity vector [u_r, v_r, w_r, p, q, r]
    drag_model : str, optional
        Drag model: 'Hoerner' (default) or 'cylinder'

    Returns
    -------
    tau_crossflow : ndarray (6,)
        Cross-flow drag vector [0, Yh, Zh, 0, Mh, Nh]
    """

    rho = 1025.0
    dx = L / 20.0

    nu_r = np.asarray(nu_r)

    # Select drag model
    if drag_model == "Hoerner":
        Cd_2D = Hoerner(B, T)
    elif drag_model == "cylinder":
        Cd_2D = cylinderDrag(L, B, nu_r)
    else:
        raise ValueError(f"Unsupported drag model {drag_model}")

    Yh = 0.0
    Zh = 0.0
    Mh = 0.0
    Nh = 0.0

    v_r = nu_r[1]
    w_r = nu_r[2]
    q = nu_r[4]
    r = nu_r[5]

    # Strip integration
    x_positions = np.arange(-L/2, L/2 + dx, dx)

    for xL in x_positions:

        U_h = abs(v_r + xL * r) * (v_r + xL * r)
        U_v = abs(w_r + xL * q) * (w_r + xL * q)

        Yh -= 0.5 * rho * T * Cd_2D * U_h * dx
        Zh -= 0.5 * rho * T * Cd_2D * U_v * dx
        Mh -= 0.5 * rho * T * Cd_2D * xL * U_v * dx
        Nh -= 0.5 * rho * T * Cd_2D * xL * U_h * dx

    tau_crossflow = np.array([0.0, Yh, Zh, 0.0, Mh, Nh])

    return tau_crossflow