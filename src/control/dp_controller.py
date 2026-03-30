import numpy as np
import numpy.typing as npt

from vessel.vessel import Vessel


class DPController:
    def __init__(
        self,
        vessel: Vessel,
        wn: npt.NDArray[np.float64],
        zeta: npt.NDArray[np.float64],
        h,
    ):
        self.vessel: Vessel = vessel
        self.wn = wn
        self.zeta = zeta
        self.z_int = np.zeros(3)
        self.h = h

    def compute_control(self, eta_d: npt.NDArray[np.float64]):
        # Pole placement
        Kp = self.vessel.M_3DOF * self.wn * self.wn
        Kd = self.vessel.M_3DOF * 2 * self.zeta * self.wn - self.vessel.D_3DOF
        Ki = 1 / 10 * Kp * self.wn
        # Error

        e = self.vessel.eta_3DOF - eta_d

        # Tau
        # tau = -self.vessel.Rz.T @ (Kp @ e + Ki @ self.z_int) - Kd @ self.vessel.nu_3DOF
        tau = -self.vessel.Rz.T @ Kp @ e
        # Integrator state
        self.z_int = self.z_int + self.h * e

        return np.array([tau[0], tau[1], 0, 0, 0, tau[2]])
