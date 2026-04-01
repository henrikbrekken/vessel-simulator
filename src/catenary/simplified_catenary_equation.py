import numpy as np


class SimplifiedCatenaryForceModel:
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

    def _catenary_equation(
        self,
        towline_force: float,
        towline_span: float,
    ) -> float:
        return np.sinh(
            (self._towline_weight_per_meter * towline_span)
            / (2 * towline_force)
            - (
                (self._towline_weight_per_meter * self.towline_length)
                / (2 * self._modulus_of_elasticity * self._cross_sectional_area)
            )
        ) - (self._towline_weight_per_meter * self.towline_length) / (2 * towline_force)

    def _bisection(
        self,
        function,
        lower_bound: float = 0.1,
        upper_bound: float = 9e15,
        tolerance: float = 1e-12,
        max_iterations: int = 100,
    ) -> float:
        function_lower = function(lower_bound)
        function_upper = function(upper_bound)

        if function_lower * function_upper > 0:
            raise ValueError("Function values at bounds must have opposite signs.")

        for _ in range(max_iterations):
            midpoint = 0.5 * (lower_bound + upper_bound)
            function_mid = function(midpoint)

            if abs(function_mid) < tolerance:
                return midpoint

            if function_lower * function_mid < 0:
                upper_bound = midpoint
                function_upper = function_mid
            else:
                lower_bound = midpoint
                function_lower = function_mid

        raise RuntimeError("Bisection did not converge.")

    def compute_towline_force(
        self,
        towline_span: float,
    ) -> float:
        def equation_for_force(towline_force: float) -> float:
            return self._catenary_equation(towline_force, towline_span)

        return self._bisection(
            equation_for_force
        )

    def compute_towline_span(
        self,
        towline_force: float,
    ) -> float:

        def equation_for_span(towline_span: float) -> float:
            return self._catenary_equation(towline_force, towline_span)

        return self._bisection(
            equation_for_span
        )
