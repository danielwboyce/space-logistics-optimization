from __future__ import annotations
from typing import TYPE_CHECKING
from itertools import product
from pyomo.kernel import (
    constraint,
    constraint_dict,
    block,
    variable,
)
from bidict import bidict

if TYPE_CHECKING:
    from ..opt_model_builder_class import OptModelBuilder


class MatrixMassConservationTransformation:
    """Class to set integer commodity conservation constraints."""

    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_matrix_mass_conserv_transform_constraints(self, m: block) -> block:
        # Create an array with all variables for (i, j, t, scnr)

        # Construct a matrix with transforming from out to in for each major
        # process.
        #     The real challenge is figuring out a good way to making sure the
        #     indexing remains consistent.


        # Multiply each successive matrix with each other

    def create_array_of_vars_at_node(
            self,
            m: block,
            i: int,
            j: int,
            t: int,
            scnr: int
    ) -> bidict[variable, int]:
        """
        Create bidirectional dictionary of Pyomo variables that maps to its 
        position in mass flow arrays (in and out) at the node/time step given
        by i, j, t.
        
        Args:
            m: Pyomo kernel block for overall model.
            i: The departure node index.
            j: The arrival node index.
            t: The index of the time step.
            scnr: The scenario index.
        """
        var_list_out = []
        var_list_in = []
        for sc_des, sc_cp, pl in product(
            m.sc_des_idx,
            m.sc_cp_idx,
            m.int_com_idx,
        ):
            if self.builder.is_feasible_arc(i, j, sc_des, sc_cp):
                var_list_out.append(m.int_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    pl,
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ])
                var_list_in.append(m.int_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    pl,
                    self.builder.flow_dict["in"],
                    t,
                    scnr,
                ])
        for sc_des, sc_cp, pl in product(
            m.sc_des_idx,
            m.sc_cp_idx,
            m.cnt_com_idx,
        ):
            if self.builder.is_feasible_arc(i, j, sc_des, sc_cp):
                var_list_out.append(m.cnt_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    pl,
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ])
                var_list_in.append(m.cnt_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    pl,
                    self.builder.flow_dict["in"],
                    t,
                    scnr,
                ])
        for sc_des, sc_cp, pl in product(
            m.sc_des_idx,
            m.sc_cp_idx,
        ):
            if self.builder.is_feasible_arc(i, j, sc_des, sc_cp):
                var_list_out.append(m.sc_fly_ind[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ])
                var_list_in.append(m.sc_fly_ind[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.flow_dict["in"],
                    t,
                    scnr,
                ])

    def set_integer_com_conserv_constraints(self, m: block) -> block:
        self._set_crew_conservation_constraints(m)
        self._set_sc_conservation_constraints(m)
        return m

    def _set_crew_conservation_constraints(self, m: block) -> block:
        """crew outflow must be equal to crew inflow"""
        m.int_com_mass_cnsv = constraint_dict()
        for i, j, t, scnr in product(
            m.dep_node_idx,
            m.arr_node_idx,
            m.time_idx,
            m.scnr_idx,
        ):
            if not self.builder.is_feasible_arc(i, j):
                continue
            m.int_com_mass_cnsv[i, j, self.builder.int_com_dict["crew #"], t, scnr] = (
                constraint(
                    sum(
                        m.int_com[
                            sc_des,
                            sc_cp,
                            i,
                            j,
                            self.builder.int_com_dict["crew #"],
                            self.builder.flow_dict["in"],
                            t,
                            scnr,
                        ]
                        for sc_des in m.sc_des_idx
                        for sc_cp in m.sc_copy_idx
                        if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
                    )
                    == sum(
                        m.int_com[
                            sc_des,
                            sc_cp,
                            i,
                            j,
                            self.builder.int_com_dict["crew #"],
                            self.builder.flow_dict["out"],
                            t,
                            scnr,
                        ]
                        for sc_des in m.sc_des_idx
                        for sc_cp in m.sc_copy_idx
                        if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
                    )
                )
            )
        return m

    def _set_sc_conservation_constraints(self, m: block) -> block:
        """spacecraft outflow must be equal to spacecraft inflow"""
        m.sc_cnsv = constraint_dict()
        for sc_des, sc_cp, i, j, t, scnr in product(
            m.sc_des_idx,
            m.sc_copy_idx,
            m.dep_node_idx,
            m.arr_node_idx,
            m.time_idx,
            m.scnr_idx,
        ):
            if not self.builder.is_feasible_arc(i, j, sc_des, sc_cp):
                continue
            m.sc_cnsv[sc_des, sc_cp, i, j, t, scnr] = constraint(
                m.sc_fly_ind[sc_des, sc_cp, i, j, self.builder.flow_dict["in"], t, scnr]
                == m.sc_fly_ind[
                    sc_des, sc_cp, i, j, self.builder.flow_dict["out"], t, scnr
                ]
            )
        return m
