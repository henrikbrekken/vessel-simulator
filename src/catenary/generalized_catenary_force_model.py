import numpy as np

class GeneralizedCatenaryForceModel:
    def __init__(
        self,
        towline_weight_per_meter: float,
        towline_length: float,
        modulus_of_elasticity: float,
        cross_sectional_area: float,
    ):
        self._towline_weight_per_meter = towline_weight_per_meter
        self.towline_length = towline_length
        self._modulus_of_elasticity = modulus_of_elasticity
        self._cross_sectional_area = cross_sectional_area

    def _parameter_estimation(self, x1: float, x2: float, z1: float = 0, z2: float = 0):
        dx = x2 - x1
        x_tilde = (x1 + x2) / 2
        dz = z2 - z1
        z_tilde = (z1 + z2) / 2

        if self.towline_length**2 < (dx**2 + dz**2):
            raise ValueError("Towline span cannot exceed the towline lenght.")

        r = np.sqrt(self.towline_length**2 + dz**2) / dx
        A = self._solve_A(r, 1e-6, 100)
        a = dx / (2 * A)
        b = x_tilde - a * np.atanh(dz / self.towline_length)
        c = z_tilde - self.towline_length / (2 * np.tanh(A))
        return a, b, c

    @staticmethod
    def _solve_A(r, tol, max_iterations):
        if r < 3:
            A = np.sqrt(6 * (r - 1))
        else:
            A = np.log(2 * r) + np.log((np.log(2 * r)))

        for _ in range(max_iterations):
            A_new = A - (np.sinh(A) - r * A) / (np.cosh(A) - r)

            if abs(A_new - A) < tol:
                return A_new

            A = A_new

        return A_new

    def forces(self, x1: float, x2: float, z1: float = 0, z2: float = 0):
        a, b, c = self._parameter_estimation(x1, x2, z1, z2)

        V1 = -self._towline_weight_per_meter * a * np.sinh((x1 - b) / a)
        V2 = self._towline_weight_per_meter * a * np.sinh((x2 - b) / a)
        H = self._towline_weight_per_meter * a

        return H, V1, V2

    def horizontal_force(self, x1: float, x2: float, z1: float = 0, z2: float = 0):
        a, b, c = self._parameter_estimation(x1, x2, z1, z2)
        H = self._towline_weight_per_meter * a
        return H
