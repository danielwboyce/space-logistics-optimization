from __future__ import annotations
from typing import TYPE_CHECKING
from itertools import product
from pyomo.kernel import (
    constraint,
    constraint_dict,
    block,
)

if TYPE_CHECKING:
    from ..opt_model_builder_class import OptModelBuilder


class NonAnticipativity:
    """
    Class to enforce first stage variables to be equal for all scenarios in stochastic problems.
    """

    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_nonanticipativity_constraints(self, m: block) -> block:
        """
        First stage variables must be equal for all scenarios,
        even if the second stage has uncertainty.

        The word "nonanticipativity" is a jardon in stochastic programming.
        """

        m.int_com_nonant = constraint_dict()
        m.cnt_com_nonant = constraint_dict()
        m.sc_fly_ind_nonant = constraint_dict()
        m.sc_fly_var_nonant = constraint_dict()
        for sc_des, sc_cp, i, j, io, t, scnr in product(
            m.sc_des_idx,
            m.sc_copy_idx,
            m.dep_node_idx,
            m.arr_node_idx,
            m.io_idx,
            self.builder.first_mis_time_steps,
            range(self.builder.n_scenarios - 1),
        ):
            if not self.builder.is_feasible_arc(i, j):
                continue
            for pl_i in m.int_com_idx:
                m.int_com_nonant[sc_des, sc_cp, i, j, pl_i, io, t, scnr] = constraint(
                    m.int_com[sc_des, sc_cp, i, j, pl_i, io, t, scnr]
                    == m.int_com[sc_des, sc_cp, i, j, pl_i, io, t, scnr + 1]
                )
            for pl_c in m.cnt_com_idx:
                m.cnt_com_nonant[sc_des, sc_cp, i, j, pl_c, io, t, scnr] = constraint(
                    m.cnt_com[sc_des, sc_cp, i, j, pl_c, io, t, scnr]
                    == m.cnt_com[sc_des, sc_cp, i, j, pl_c, io, t, scnr + 1]
                )
            m.sc_fly_ind_nonant[sc_des, sc_cp, i, j, io, t, scnr] = constraint(
                m.sc_fly_ind[sc_des, sc_cp, i, j, io, t, scnr]
                == m.sc_fly_ind[sc_des, sc_cp, i, j, io, t, scnr + 1]
            )
            for sc_var in m.sc_var_idx:
                m.sc_fly_var_nonant[sc_des, sc_cp, sc_var, i, j, io, t, scnr] = (
                    constraint(
                        m.sc_fly_var[sc_des, sc_cp, sc_var, i, j, io, t, scnr]
                        == m.sc_fly_var[sc_des, sc_cp, sc_var, i, j, io, t, scnr + 1]
                    )
                )

        if self.builder.use_isru:
            m.isru_mass_nonant = constraint_dict()
            m.isru_O2rate_nonant = constraint_dict()
            for t, scnr in product(
                self.builder.first_mis_time_steps,
                range(self.builder.n_scenarios - 1),
            ):
                m.isru_mass_nonant[t, scnr] = constraint(
                    m.isru_mass[t, scnr] == m.isru_mass[t, scnr + 1]
                )
                m.isru_O2rate_nonant[t, scnr] = constraint(
                    m.isru_O2rate[t, scnr] == m.isru_O2rate[t, scnr + 1]
                )

        return m
