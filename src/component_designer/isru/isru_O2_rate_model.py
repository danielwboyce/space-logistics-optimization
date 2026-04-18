import sys
from math import exp


class ISRUDesign():
    @staticmethod
    def get_isru_carbothermal_O2_rate(isru_mass: float) -> float:
        if isru_mass >= 400:
            return isru_mass * (
                -0.43798
                + 6.96226 * (1 - exp(-isru_mass / 812.15628))
                + 2.01727 * (1 - exp(-isru_mass / 3967.2644))
            )
        else:
            return 0
