from abc import ABC, abstractmethod

import numpy.typing as npt


class Vessel(ABC):
    eta_3DOF: npt.NDArray
    nu_3DOF: npt.NDArray
    M_3DOF: npt.NDArray
    D_3DOF: npt.NDArray

    @abstractmethod
    def integrate_dynamics(self, dt, tau, Vc, betaV):
        pass
