import numpy as np
import numpy.typing as npt

from vessel.polar_transformation import polarTransformation


class PolarController:
    def __init__(
        self,
        polar_vessel: polarTransformation,
        wn: npt.NDArray[np.float64],
        zeta: npt.NDArray[np.float64],
        h,
    ):
        self.polar_vessel: polarTransformation = polar_vessel
        self.wn = wn
        self.zeta = zeta
        self.z_int = np.zeros(3)
        self.h = h

    def compute_control(self, eta_d):
        # Pole placement
        Kp = self.polar_vessel.M * self.wn * self.wn
        Kd = self.polar_vessel.M * 2 * self.zeta * self.wn
        Ki = 1 / 10 * Kp * self.wn

        # Error
        e = self.polar_vessel.eta - eta_d

        # Tau
        tau = self.polar_vessel.Tinv @ (
            Kp @ e + Kd * self.polar_vessel.eta_dot + Ki @ self.z_int
        )

        # Integrator state
        self.z_int = self.z_int + self.h * e

        return np.array([tau[0], tau[1], 0, 0, 0, tau[2]])
