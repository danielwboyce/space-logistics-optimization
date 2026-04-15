from __future__ import annotations
from typing import TYPE_CHECKING
from pyomo.kernel import (
    objective,
    constraint,
    minimize,
    maximize,
    block,
)
from pyomo.core.expr.numeric_expr import SumExpression

if TYPE_CHECKING:
    from .opt_model_builder_class import OptModelBuilder


class Objective:
    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_objective(self, m: block) -> block:
        """
        Define the objective function of the model.
        Add augmented Lagrandian terms if the model is an ADMM subproblem.

        Args:
            m: pyomo.kernel model
        """
        if self.builder.is_stochastic:
            m.fmleo_def = constraint(
                m.fmleo
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
            m.fmleo_def = constraint(m.fmleo == (self._get_obj_term(m, 0, m.time_idx)))
        if self.builder.mode != "ALCsubproblem":
            m.obj = objective(m.fmleo, sense=maximize)
        else:
            # WARNING: For some reason, the augmented objective function
            # CANNOT be defined as a constraint due to its quadratic nature.
            # It will be flagged as a nonconvex quadratic constraint
            # by Gurobi 10, even though it is not.
            m.obj = objective(
                expr=(
                    m.fmleo
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
                sense=maximize,
            )
        return m

    def _get_obj_term(self, m: block, scnr: int, time_list: list[int]) -> SumExpression:
        """Returns sum of LOX that is present in the LEO node at the final
        time step.

        Args:
            m: pyomo.kernel model
            scnr: scenario id
            time_list: list of time steps
        Returns:
            SumExpression: sum of commodities and sc mass launched to LEO
        """
        term = (
            self.builder.cnt_com_costs[self.builder.cnt_com_dict["oxygen_storage"]]
            * sum(
                m.cnt_com[
                    sc_des,
                    sc_cp,
                    j,
                    self.builder.node_dict["LEO"],
                    self.builder.cnt_com_dict["oxygen_storage"],
                    self.builder.flow_dict["in"],
                    time_list[-1] - self.builder.delta_t[j][self.builder.node_dict["LEO"]][len(time_list) - 1],
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                for j in m.arr_node_idx
                if self.builder.is_feasible_arc(self.builder.node_dict["LEO"], j, sc_des, sc_cp)
                if time_list[-1] - self.builder.delta_t[j][self.builder.node_dict["LEO"]][len(time_list) - 1] in m.time_idx
            )
            - self.builder.cnt_com_costs[self.builder.cnt_com_dict["oxygen_storage"]]
            * sum(
                m.cnt_com[
                    sc_des,
                    sc_cp,
                    self.builder.node_dict["Earth"],
                    self.builder.node_dict["LEO"],
                    self.builder.cnt_com_dict["oxygen_storage"],
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                for t in time_list
            )
            - self.builder.cnt_com_costs[self.builder.cnt_com_dict["oxygen"]]
            * sum(
                m.cnt_com[
                    sc_des,
                    sc_cp,
                    self.builder.node_dict["Earth"],
                    self.builder.node_dict["LEO"],
                    self.builder.cnt_com_dict["oxygen"],
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
