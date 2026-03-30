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
        self.wn = np.diag(wn)
        self.zeta = np.diag(zeta)
        self.z_int = np.zeros(3)
        self.h = h

    def compute_control(self, eta_d: npt.NDArray[np.float64]):
        # Pole placement
        Kp = np.diag(np.diag(self.polar_vessel.M)) * self.wn * self.wn
        Kd = (
            np.diag(np.diag(self.polar_vessel.M)) * 2 * self.zeta * self.wn
            - self.polar_vessel.D
        )
        Ki = 1 / 10 * Kp * self.wn
        # Error
        self.e = self.polar_vessel.eta - eta_d

        # Tau
        self.tau = -self.polar_vessel.T.T @ (
            Kp @ self.e + Kd @ self.polar_vessel.eta_dot + Ki @ self.z_int
        )
        # self.tau = -self.polar_vessel.T.T @ Kp @ self.e

        # Integrator state
        self.z_int = self.z_int + self.h * self.e

        return np.array([self.tau[0], self.tau[1], 0, 0, 0, self.tau[2]])
