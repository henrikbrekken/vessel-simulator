import numpy as np
from vessel.vessel import Vessel
from vessel.far_spicka import FarSpica

class polarTransformation():
    """RUN UPDATE STATES FOR EACH LOOP IN SIMULATION"""
    def __init__(self, vessel:Vessel, x0=0, y0=0):
        self.vessel:FarSpica = vessel
        self.update_states()

    def _update_x0(self, x0):
        self.x0 = x0

    def _update_y0(self, y0):
        self.y0 = y0

    def _update_generalized_coordinates(self):
        dx = self.vessel.eta[0] - self.x0
        dy = self.vessel.eta[1] - self.y0

        self.rho = np.sqrt((dx)**2 + (dy)**2)
        self.gamma = np.atan2(dy,dx)
        self.psi = self.vessel.eta[5]

    def _update_T(self):
        a = self.psi - self.gamma
        self.T = np.array([[np.cos(a),               -np.sin(a),             0],
                           [1/self.rho*np.sin(a),    1/self.rho*np.cos(a),   0],
                           [0,                       0,                      1]])
        self.Tinv = np.array([[np.cos(a),    self.rho*np.sin(a),     0],
                              [-np.sin(a),   self.rho*np.cos(a),     0],
                              [0,            0,                      1]])
        
    def _update_M(self):
        self.M = self.Tinv @ self.vessel.M_3DOF @ self.T

    def _update_D(self):
        self.D = self.Tinv @ self.vessel.D_3DOF @ self.T

    def update_states(self, x0=0, y0=0):
        self._update_x0(x0)
        self._update_y0(y0)
        self._update_generalized_coordinates()
        self._update_T()
        self._update_M()
        self._update_D()