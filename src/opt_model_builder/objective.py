from __future__ import annotations
from typing import TYPE_CHECKING
from pyomo.kernel import (
    objective,
    constraint,
    minimize,
)
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.expr.numeric_expr import SumExpression

if TYPE_CHECKING:
    from .opt_model_builder_class import OptModelBuilder


class Objective:
    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_objective(self, m: ConcreteModel) -> ConcreteModel:
        """
        Define the objective function of the model.
        Add augmented Lagrandian terms if the model is an ADMM subproblem.

        Args:
            m: pyomo.kernel model
        """
        if self.builder.is_stochastic:
            m.imleo_def = constraint(
                m.imleo
                == (
                    self._get_obj_term(m, 0, self.builder.first_mis_time_steps)
                    + sum(
                        self.builder.scenario_prob[scnr]
                        * self._get_obj_term(
                            m, scnr, self.builder.second_mis_time_steps
                        )
                        for scnr in m.scnr_idx
                    )
                )
            )
        else:
            m.imleo_def = constraint(m.imleo == (self._get_obj_term(m, 0, m.time_idx)))
        if self.builder.mode != "ALCsubproblem":
            m.obj = objective(m.imleo, sense=minimize)
        else:
            # WARNING: For some reason, the augmented objective function
            # CANNOT be defined as a constraint due to its quadratic nature.
            # It will be flagged as a nonconvex quadratic constraint
            # by Gurobi 10, even though it is not.
            m.obj = objective(
                expr=(
                    m.imleo
                    + sum(
                        m.lag_mult[sc_des, sc_var] * m.rel_consis_vio[sc_des, sc_var]
                        for sc_des in m.sc_des_idx
                        for sc_var in m.sc_var_idx
                    )
                    + sum(
                        (
                            m.penalty_weight[sc_des, sc_var]
                            * m.abs_consis_vio[sc_des, sc_var]
                        )
                        ** 2
                        for sc_des in m.sc_des_idx
                        for sc_var in m.sc_var_idx
                    )
                ),
                sense=minimize,
            )
        return m

    def _get_obj_term(self, m: ConcreteModel, scnr: int, time_list: list[int]) -> SumExpression:
        """Returns sum of commodities and sc mass launched from Earth to LEO
        for a specific scenario over given time interval.

        Args:
            m: pyomo.kernel model
            scnr: scenario id
            time_list: list of time steps
        Returns:
            SumExpression: sum of commodities and sc mass launched to LEO
        """
        term = (
            sum(
                self.builder.int_com_costs[int_com]
                * sum(
                    m.int_com[
                        sc_des,
                        sc_cp,
                        self.builder.node_dict["Earth"],
                        self.builder.node_dict["LEO"],
                        int_com,
                        self.builder.flow_dict["out"],
                        t,
                        scnr,
                    ]
                    for sc_des in m.sc_des_idx
                    for sc_cp in m.sc_copy_idx
                    for t in time_list
                )
                for int_com in m.int_com_idx
            )
            + sum(
                self.builder.cnt_com_costs[cnt_com]
                * sum(
                    m.cnt_com[
                        sc_des,
                        sc_cp,
                        self.builder.node_dict["Earth"],
                        self.builder.node_dict["LEO"],
                        cnt_com,
                        self.builder.flow_dict["out"],
                        t,
                        scnr,
                    ]
                    for sc_des in m.sc_des_idx
                    for sc_cp in m.sc_copy_idx
                    for t in time_list
                )
                for cnt_com in m.cnt_com_idx
            )
            + sum(
                m.sc_fly_var[
                    sc_des,
                    sc_cp,
                    self.builder.sc_var_dict["dry mass"],
                    self.builder.node_dict["Earth"],
                    self.builder.node_dict["LEO"],
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                for t in time_list
            )
        )
        return term
