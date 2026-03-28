import numpy as np

from vessel.vessel import Vessel


class polarTransformation:
    """RUN UPDATE STATES FOR EACH LOOP IN SIMULATION"""

    def __init__(self, vessel: Vessel, x0=0, y0=0):
        self.vessel: Vessel = vessel
        self.x0 = x0
        self.y0 = y0
        self._update_states()

    def _update_x0(self, x0):
        self.x0 = x0

    def _update_y0(self, y0):
        self.y0 = y0

    def _update_generalized_coordinates(self):
        dx = self.vessel.eta_3DOF[0] - self.x0
        dy = self.vessel.eta_3DOF[1] - self.y0

        self.rho = np.sqrt((dx) ** 2 + (dy) ** 2)
        self.gamma = np.atan2(dy, dx)
        self.psi = self.vessel.eta_3DOF[2]

        self.eta = np.array([self.rho, self.gamma, self.psi])
        self.nu = self.vessel.nu_3DOF

    def _update_T(self):
        a = self.psi - self.gamma
        self.T = np.array(
            [
                [np.cos(a), -np.sin(a), 0],
                [1 / self.rho * np.sin(a), 1 / self.rho * np.cos(a), 0],
                [0, 0, 1],
            ]
        )
        self.Tinv = np.array(
            [
                [np.cos(a), self.rho * np.sin(a), 0],
                [-np.sin(a), self.rho * np.cos(a), 0],
                [0, 0, 1],
            ]
        )

    def _update_eta_dot(self):
        self.eta_dot = self.T @ np.array(
            [self.vessel.nu[0], self.vessel.nu[1], self.vessel.nu[5]]
        )

    def _update_M(self):
        self.M = self.Tinv @ self.vessel.M_3DOF @ self.T

    def _update_D(self):
        self.D = self.Tinv @ self.vessel.D_3DOF @ self.T

    def update_polar_frame_co(self, x0, y0):
        self._update_x0(x0)
        self._update_y0(y0)

    def integrate_dynamics(self, dt, tau, Vc, betaVc):
        self.vessel.integrate_dynamics(dt, tau, Vc, betaVc)
        self._update_states()

    def _update_states(self):
        self._update_generalized_coordinates()
        self._update_T()
        self._update_M()
        self._update_D()
