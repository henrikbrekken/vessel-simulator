import numpy as np
import numpy.typing as npt
from vessel import FarSpica


class Controller:
    def __init__(self, vessel:FarSpica, wn:npt.NDArray[np.float64], zeta:npt.NDArray[np.float64], h):
        self.vessel = vessel
        self.wn = wn
        self.zeta = zeta
        self.z_int = np.zeros(3)
        self.h = h

    def compute_control(self, eta_d):
        # Pole placement
        Kp = self.vessel.Mdiag_3dof * self.wn * self.wn
        Kd = self.vessel.Mdiag_3dof * 2 * self.zeta * self.wn
        Ki = 1/10 * Kp * self.wn

        # Rotation matrix
        R = np.array([[np.cos(self.vessel.eta[5]),  -np.sin(self.vessel.eta[5]),    0],
                      np.sin(self.vessel.eta[5]),   np.cos(self.vessel.eta[5]),     0],
                      [0,                           0,                              1])

        # Error
        e = self.vessel.eta - eta_d

        # Tau
        tau = - R.T @ (Kp @ e + Kd @ self.z_int)

        # Integrator state
        z_int = z_int + self.h * e

        return np.array([tau[0], tau[1], 0, 0, 0, tau[2]])