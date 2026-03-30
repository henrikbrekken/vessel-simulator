import numpy as np

from functions.ssa import ssa


class SecondOrderVelocityAndAccelerationSaturationReferenceModel:
    def __init__(self, x_init:float, h, wn_f:float, zeta_f:float, delta_f:float, v_max:float, a_max:float):
        self._x = x_init
        self._h = h
        self._wn_f = wn_f
        self._zeta_f = zeta_f
        self._delta_f = delta_f
        self.v_max = v_max
        self.a_max = a_max
        self._v = 0
        self._a = 0
        self._is_angle = False

    @classmethod
    def angular(cls, *args, **kwargs):
        c = cls(*args, **kwargs)
        c._is_angle = True
        return c

    def x_d(self, x_r):
        if self._is_angle:
            self.error = ssa(x_r - self._x)
        else:
            self.error = x_r - self._x
        x_dot = self._v
        self._a = (
            self._wn_f**2 * self.error 
            - 2 * self._zeta_f * self._wn_f * self._v
            - self._delta_f * np.abs(self._v) * self._v
        )
        if (self._a > self.a_max and self._v >= 0) or (
            self._a < -self.a_max and self._v <= 0
        ):
            self._a = np.sign(self._a) * self.a_max
        self._v = self._v + self._h * self._a
        if np.abs(self._v) > self.v_max:
            self._v = np.sign(self._v) * self.v_max

        self._x = self._x + self._h * x_dot

        return self._x

