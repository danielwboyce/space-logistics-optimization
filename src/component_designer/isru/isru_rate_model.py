import sys
from math import exp


class ISRUDesign():
    @staticmethod
    def get_isru_rate_carbothermal_O2H2(isru_mass: float) -> float:
        # The minimum mass for this will be enforced when creating
        # the piecewise linear constraints and some big M constraints.
        return (
            -0.43798
            + 6.96226 * (1 - exp(-isru_mass / 812.15628))
            + 2.01727 * (1 - exp(-isru_mass / 3967.2644))
        )

    @staticmethod
    def get_isru_rate_mre_metal(isru_mass: float) -> float:
        # The minimum mass for this will be enforced when creating
        # the piecewise linear constraints and some big M constraints.
        return 0.75 * ISRUDesign.get_isru_rate_carbothermal_O2H2(isru_mass)

    @staticmethod
    def get_isru_rate_workshop(isru_mass: float) -> float:
        return 5.0
