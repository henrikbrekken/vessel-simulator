from abc import ABC
import numpy.typing as npt

class Vessel(ABC):
    eta: npt.NDArray
    M_3DOF: npt.NDArray
    D_3DOF: npt.NDArray