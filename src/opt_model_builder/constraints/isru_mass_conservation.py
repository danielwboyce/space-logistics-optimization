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

        for i, j, t, scnr, pl_name in product(
            m.dep_node_idx,
            m.arr_node_idx,
            m.time_idx,
            m.scnr_idx,
            self.builder.cnt_com_names,
        ):
            if not self.builder.can_operate_ISRU(i, j):
                continue
            if pl_name not in self.builder.isru_io_dict:
                continue

            (out_array, in_array) = self._create_intermediate_variables_array(m, i, j, t, scnr)
            trans_matrix = np.eye(self.builder.n_isru_io)
            isru_work_time = self.builder.isru_work_time[i][
                self.builder._network_def.date_to_time_idx_dict[t]
            ]
            self._calculate_isru_work_matrix_for_reactor(m, 0, isru_work_time, t, scnr)

        return m

    def _create_intermediate_variables_array(self, m, i, j, t, scnr):
        """
        Create arrays that correspond to ISRU IO variables and reactor masses
        that are under interest during any/all ISRU processes (limited to a
        particular arc and time step).
        
        These arrays correspond to the commodity flow transformations
        specifically for the ISRU commodities/arcs, and supports creating
        the commodity transformation matrix.
        """
        out_array = [None] * self.builder.n_isru_io
        in_array = [None] * self.builder.n_isru_io

        for index in range(self.builder.n_isru_io):
            comdty = self.builder.isru_io_dict.inverse[index]
            out_array[index] = sum(
                m.cnt_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.cnt_com_dict[comdty],
                    self.builder.flow_dict["out"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
            )
            in_array[index] = sum(
                m.cnt_com[
                    sc_des,
                    sc_cp,
                    i,
                    j,
                    self.builder.cnt_com_dict[comdty],
                    self.builder.flow_dict["in"],
                    t,
                    scnr,
                ]
                for sc_des in m.sc_des_idx
                for sc_cp in m.sc_copy_idx
                if self.builder.is_feasible_arc(i, j, sc_des, sc_cp)
            )
        return (out_array, in_array)
    
    def _calculate_isru_work_matrix_for_reactor(self, m: block, isru_des: int, isru_work_time: float, t: int, scnr: int) -> np.ndarray:
        """Calcualte the flow transformation matrix for a specific subprocess/ISRU reactor.
        
        This assumes the matrix relation
        x- = Q * x+
        where
                 [ input res.  ]-
            x- = [ output res. ] 
                 [ isru mass   ] 
                 [ input res.  ]+
            x+ = [ output res. ] 
                 [ isru mass   ] 
                      ( [0, 0, beta_i * input_frac;  ] )                 
            Q  = expm ( [0, 0, beta_o * output_frac; ] ) * ISRU_work_time
                      ( [0, 0, alpha_r               ] )                 
        Note that if multiple subprocesses are occurring simultatneously, we
        can model this by matrix multiplying each value of Q together. This
        method calculates the Q matrix for the ISRU process related to the
        specific ISRU reactor design designated by isru_des.
            
        Args:
            isru_des: Index of the ISRU reactor design of interest.
            isru_work_time: Work time [days] for which the ISRU system is
                expected to work.
        Returns:
            Q: The commodity flow transformation matrix for the given ISRU
                reactor for the given work time. A numpy array with size
                (self.builder.n_isru_io, self.builder.n_isru_io)
        """

        Q = NumericNDArray((self.builder.n_isru_io, self.builder.n_isru_io))
        # for i in range(self.builder.n_isru_io):
        #     Q[i, i] = 1.0
        Q[0, 2] = m.isru_rate[isru_des, t, scnr]

