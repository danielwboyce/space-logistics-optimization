from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer_class import Optimizer


class PWLApproximation:
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def solve_w_pwl_approx(
        self,
        pwl_increment: float,
    ) -> dict[str, float | np.ndarray]:
        """Solves with PieceWise Linear (PWL) approximation
        Args:
            pwl_increment: increment used for PWL approximation of spacecraft design
        Returns:
            dict containing:
                FMLEO (float): optimal objective function (FMLEO) value
                sc_vars (np.array): SC design variables, subject to PWL approx.
        """
        self.optimizer._model_builder.mode = "Piecewise Linear Approx"
        model = self.optimizer._model_builder.build_model(pwl_increment)
        model = self.optimizer.solver.solve_model(model)
        FMLEO = model.fmleo.value
        sc_vars = self.optimizer.output.get_sc_vars(model)
        return {"obj": FMLEO, "design vars": sc_vars}
