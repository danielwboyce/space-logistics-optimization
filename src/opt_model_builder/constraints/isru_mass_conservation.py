from __future__ import annotations
from typing import TYPE_CHECKING
from itertools import product
from pyomo.kernel import (
    constraint,
    constraint_dict,
    block,
)
from pyomo.core.expr.ndarray import NumericNDArray
import numpy as np

if TYPE_CHECKING:
    from ..opt_model_builder_class import OptModelBuilder


class ISRUConservation:
    """
    Class to set ISRU conservation and consumption constraints.
    """

    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_isru_conservation_constraints(self, m: block) -> block:
        if not self.builder.use_isru:
            return m

        m.isru_mass_cnsv = constraint_dict()
        if not self.builder.isru_use_convex_relaxation:
            m.isru_bilinear_definitions = constraint_dict()
            m.isru_trilinear_definitions = constraint_dict()
        else:
            m.isru_bilinear_relaxation1 = constraint_dict()
            m.isru_bilinear_relaxation2 = constraint_dict()
            m.isru_bilinear_relaxation3 = constraint_dict()
            m.isru_bilinear_relaxation4 = constraint_dict()
            m.isru_trilinear_relaxation1 = constraint_dict()
            m.isru_trilinear_relaxation2 = constraint_dict()
            m.isru_trilinear_relaxation3 = constraint_dict()
            m.isru_trilinear_relaxation4 = constraint_dict()

        i = self.builder.node_dict["LS"]
        j = i
        for t, scnr in product(
            m.time_idx,
            m.scnr_idx,
        ):
            if not self.builder.can_operate_ISRU(i, j):
                continue

            for comdty_name in self.builder.isru_io_dict:
                m = self._calculate_isru_constraint_per_node(
                    m=m,
                    i=i,
                    j=j,
                    comdty_name=comdty_name,
                    t=t,
                    scnr=scnr,
                )

            m = self._calculate_bilinear_constraints(m=m, i=i, t=t, scnr=scnr)

        return m

    def _calculate_isru_constraint_per_node(
            self,
            m: block,
            i: int,
            j: int,
            comdty_name: str,
            t: int,
            scnr: int
    ) -> block:
        """Calculates the ISRU constraints at each time step."""

        comdty_id = self.builder.com_dict[comdty_name]
        t_id = self.builder._network_def.date_to_time_idx_dict[t]
        if comdty_name in self.builder.int_com_names:
            constraint_lhs = sum(
                m.int_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.int_com_dict[comdty_name],
                    self.builder.flow_dict["in"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
            )
            constraint_rhs = sum(
                m.int_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.int_com_dict[comdty_name],
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
            )
        elif comdty_name in self.builder.cnt_com_names:
            constraint_lhs = sum(
                m.cnt_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.cnt_com_dict[comdty_name],
                    self.builder.flow_dict["in"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
            )
            constraint_rhs = sum(
                m.cnt_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.cnt_com_dict[comdty_name],
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
            )
        else:
            assert comdty_name in self.builder.int_com_names + self.builder.cnt_com_names

        for isru_des_id in m.isru_des_idx:
            isru_design = self.builder.isru.isru_designs[isru_des_id]
            if comdty_name in isru_design.outputs:
                constraint_rhs = (
                    constraint_rhs
                    + isru_design.outputs[comdty_name] # output fraction
                    * m.isru_total_prod[isru_des_id, t, scnr]
                )
            if isru_design.inputs is not None and comdty_name in isru_design.inputs:
                constraint_rhs = (
                    constraint_rhs
                    - isru_design.inputs[comdty_name] # input fraction
                    * m.isru_total_prod[isru_des_id, t, scnr]
                )
            if comdty_name == "maintenance":
                # This code should only be executed if "maintenace" is one of the IOs
                assert("maintenance" in self.builder.isru_io_dict), """
                Error: "maintenance" commodity getting handled by
                ISRUConservation class even though it's not one of the
                ISRU I/O commodities."""
                constraint_rhs = (
                    constraint_rhs
                    - (isru_design.maintenance_cost
                        * self.builder.isru_work_time[i][t_id]
                        / 365.0)
                    * m.isru_mass[isru_des_id, t, scnr]
                )
                if isru_design.reactor_name == "workshop":
                    for output in isru_design.outputs:
                        if output not in self.builder.isru_reactor_dict:
                            continue
                        isru_design_child = self.builder.isru.isru_designs[
                            self.builder.isru_reactor_dict[output]]
                        constraint_rhs = (
                            constraint_rhs
                            - isru_design.outputs[output] # output fraction
                            * (isru_design_child.maintenance_cost
                                * self.builder.isru_work_time[i][t_id]
                                / 365.0)
                            * m.isru_total_prod[isru_des_id, t, scnr]
                            )
            if comdty_name == isru_design.reactor_mass_commodity:
                constraint_rhs = (constraint_rhs
                                    - isru_design.decay_rate
                                    * (self.builder.isru_work_time[i][t_id] / 365.0)
                                    * m.isru_mass[isru_des_id, t, scnr])
            # if isru_design.reactor_name == "workshop":
            #     # Adding in trilinear constraints
            #     for isru_design_child_name in isru_design.outputs:
            #         if isru_design_child_name not in self.builder.isru_reactor_dict:
            #             continue
            #         isru_design_child_id = self.builder.isru_reactor_dict[isru_design_child_name]
            #         isru_design_child = self.builder.isru.isru_designs[isru_design_child_id]
            #         if (isru_design_child.inputs is not None and
            #                 comdty_name in isru_design_child.inputs):
            #             constraint_rhs = (
            #                 constraint_rhs
            #                 - isru_design.outputs[isru_design_child_name]
            #                 * isru_design_child.inputs[comdty_name]
            #                 * m.isru_trilinear_prod[isru_design_child_id, t, scnr]
            #             )
            #         if comdty_name in isru_design_child.outputs:
            #             if isru_design_child.inputs is not None and comdty_name in isru_design_child.inputs:
            #                 constraint_rhs = (
            #                     constraint_rhs
            #                     + isru_design.outputs[isru_design_child_name]
            #                     * isru_design_child.inputs[comdty_name]
            #                     * m.isru_trilinear_prod[isru_design_child_id, t, scnr]
            #                 )
        m.isru_mass_cnsv[i, j, comdty_id, t, scnr] = constraint(constraint_lhs == constraint_rhs)
        return m

    def _calculate_bilinear_constraints(
            self,
            m: block,
            i: int,
            t: int,
            scnr: int
    ) -> block:
        """Calculates the bilinear constraints for ISRU."""
        t_id = self.builder._network_def.date_to_time_idx_dict[t]
        if not self.builder.isru_use_convex_relaxation:
            for isru_des in m.isru_des_idx:
                m.isru_bilinear_definitions[isru_des, t, scnr] = constraint(
                    m.isru_total_prod[isru_des, t, scnr]
                    == (self.builder.isru_work_time[i][t_id] / 365.0)
                    * m.isru_rate[isru_des, t, scnr]
                    * m.isru_mass[isru_des, t, scnr]
                )
                # if "plant_workshop" in self.builder.isru_reactor_dict:
                #     m.isru_trilinear_definitions[isru_des, t, scnr] = constraint(
                #         m.isru_trilinear_prod[isru_des, t, scnr]
                #         == m.isru_rate[isru_des, t, scnr]
                #         * m.isru_total_prod[self.builder.isru_reactor_dict["plant_workshop"], t, scnr]
                #     )
        else:
            # Convex relation (McCormick Envelope) on bilinear constraints
            for isru_des in m.isru_des_idx:
                work_time = self.builder.isru_work_time[i][t_id] / 365.0
                # assert(work_time > 0), """
                # Error:
                # If ISRU can be run, the ISRU work time must be greater
                # than zero.
                # Received value:
                #     self.builder.isru_work_time[{}][{}] / 365.0 = {}
                # """.format(i, t_id, self.builder.isru_work_time[i][t_id] / 365.0)

                z = m.isru_total_prod[isru_des, t, scnr]
                x1 = m.isru_mass[isru_des, t, scnr]
                x2 = m.isru_rate[isru_des, t, scnr]
                x1_lb = 0.0
                x1_ub = self.builder.isru.get_mass_upper_bound()
                x2_lb = 0.0
                x2_ub = self.builder.isru.isru_designs[isru_des].production_rate(x1_ub)

                m.isru_bilinear_relaxation1[isru_des, t, scnr] = constraint(
                    z >= work_time * ((x1_lb * x2) + (x2_lb * x1) - (x1_lb * x2_lb))
                )
                m.isru_bilinear_relaxation2[isru_des, t, scnr] = constraint(
                    z >= work_time * ((x1_ub * x2) + (x2_ub * x1) - (x1_ub * x2_ub))
                )
                m.isru_bilinear_relaxation3[isru_des, t, scnr] = constraint(
                    z <= work_time * ((x1_lb * x2) + (x2_ub * x1) - (x1_lb * x2_ub))
                )
                m.isru_bilinear_relaxation4[isru_des, t, scnr] = constraint(
                    z <= work_time * ((x1_ub * x2) + (x2_lb * x1) - (x1_ub * x2_lb))
                )

            # if "plant_workshop" in self.builder.isru_reactor_dict:
            #     # Convex relation (McCormick Envelope) on trilinear constraints.
            #     # Because the isru_total_prod already has a set of convex relation
            #     # constraints active, we don't need to anything special to handle
            #     # the trilinear linear case other than create a bilinear relaxation
            #     # between the third term and the original bilinear term.
            #     for isru_des in m.isru_des_idx:
            #         isru_des_ws = self.builder.isru_reactor_dict["plant_workshop"]
            #         mass_ub = self.builder.isru.get_mass_upper_bound()
            #         z = m.isru_trilinear_prod[isru_des, t, scnr]
            #         x1 = m.isru_total_prod[isru_des_ws, t, scnr]
            #         x2 = m.isru_rate[isru_des, t, scnr]
            #         x1_lb = 0.0
            #         x1_ub = mass_ub * self.builder.isru.isru_designs[isru_des_ws].production_rate(mass_ub)
            #         x2_lb = 0.0
            #         x2_ub = self.builder.isru.isru_designs[isru_des].production_rate(mass_ub)

            #         m.isru_trilinear_relaxation1[isru_des, t, scnr] = constraint(
            #             z >= (x1_lb * x2) + (x2_lb * x1) - (x1_lb * x2_lb)
            #         )
            #         m.isru_trilinear_relaxation2[isru_des, t, scnr] = constraint(
            #             z >= (x1_ub * x2) + (x2_ub * x1) - (x1_ub * x2_ub)
            #         )
            #         m.isru_trilinear_relaxation3[isru_des, t, scnr] = constraint(
            #             z <= (x1_lb * x2) + (x2_ub * x1) - (x1_lb * x2_ub)
            #         )
            #         m.isru_trilinear_relaxation4[isru_des, t, scnr] = constraint(
            #             z <= (x1_ub * x2) + (x2_lb * x1) - (x1_ub * x2_lb)
            #         )

        return m

    # def _create_intermediate_variables_array(self, m, i, j, t, scnr):
    #     """
    #     Create arrays that correspond to ISRU IO variables and reactor masses
    #     that are under interest during any/all ISRU processes (limited to a
    #     particular arc and time step).

    #     These arrays correspond to the commodity flow transformations
    #     specifically for the ISRU commodities/arcs, and supports creating
    #     the commodity transformation matrix.
    #     """
    #     out_array = [None] * self.builder.n_isru_io
    #     in_array = [None] * self.builder.n_isru_io

    #     for index in range(self.builder.n_isru_io):
    #         comdty = self.builder.isru_io_dict.inverse[index]
    #         out_array[index] = sum(
    #             m.cnt_com[
    #                 sc_des,
    #                 sc_cp,
    #                 i,
    #                 j,
    #                 self.builder.cnt_com_dict[comdty],
    #                 self.builder.flow_dict["out"],
    #                 t,
    #                 scnr,
    #             ]
    #             for sc_des in m.sc_des_idx
    #             for sc_cp in m.sc_copy_idx
    #             if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
    #         )
    #         in_array[index] = sum(
    #             m.cnt_com[
    #                 sc_des,
    #                 sc_cp,
    #                 i,
    #                 j,
    #                 self.builder.cnt_com_dict[comdty],
    #                 self.builder.flow_dict["in"],
    #                 t,
    #                 scnr,
    #             ]
    #             for sc_des in m.sc_des_idx
    #             for sc_cp in m.sc_copy_idx
    #             if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
    #         )
    #     return (out_array, in_array)

    # def _calculate_isru_work_matrix_for_reactor(
    #         self,
    #         m: block,
    #         isru_des: int,
    #         isru_work_time: float,
    #         t: int,
    #         scnr: int
    # ) -> np.ndarray:
    #     """Calcualte the flow transformation matrix for a specific subprocess/ISRU reactor.

    #     This assumes the matrix relation
    #     x- = Q * x+
    #     where
    #              [ input res.  ]-
    #         x- = [ output res. ]
    #              [ isru mass   ]
    #              [ input res.  ]+
    #         x+ = [ output res. ]
    #              [ isru mass   ]
    #                   ( [0, 0, beta_i * input_frac;  ] )
    #         Q  = expm ( [0, 0, beta_o * output_frac; ] ) * ISRU_work_time
    #                   ( [0, 0, alpha_r               ] )
    #     Note that if multiple subprocesses are occurring simultatneously, we
    #     can model this by matrix multiplying each value of Q together. This
    #     method calculates the Q matrix for the ISRU process related to the
    #     specific ISRU reactor design designated by isru_des.

    #     Args:
    #         isru_des: Index of the ISRU reactor design of interest.
    #         isru_work_time: Work time [days] for which the ISRU system is
    #             expected to work.
    #     Returns:
    #         Q: The commodity flow transformation matrix for the given ISRU
    #             reactor for the given work time. A numpy array with size
    #             (self.builder.n_isru_io, self.builder.n_isru_io)
    #     """

    #     Q = NumericNDArray((self.builder.n_isru_io, self.builder.n_isru_io))
    #     # for i in range(self.builder.n_isru_io):
    #     #     Q[i, i] = 1.0
    #     Q[0, 2] = m.isru_rate[isru_des, t, scnr]
