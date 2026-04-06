from __future__ import annotations
from typing import TYPE_CHECKING
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core import Set, RangeSet

if TYPE_CHECKING:
    from .opt_model_builder_class import OptModelBuilder


class Indices:
    """Class to define indices for variables and constraints"""

    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_indices(self, m: ConcreteModel) -> ConcreteModel:
        """Define indices and create a dict of all indices

        sc_des_idx: spacecraft design index
        sc_copy_idx: spacecraft copy index/how many spacecraft with
            the same design are allowed in mission
        sc_var_idx: spacecraft design variables index
        dep_node_idx: departure node index
        arr_node_idx: arrival node index
        int_com_idx: integer commodity index
        cnt_com_idx: continuous commodity index
        io_idx: inflow/outflow index: 1-inflow; 0-outflow
        time_idx: time index. For computational puposes, the model is constructed
            so that outbound and inbound (return) missions happen in a single day
            (one day for outbound and another day for inbound). To calculate
            time-related quantities (eg, crew consumables per day), the actual
            time of flight is used.
        scnr_idx: scenario index for stochastic cases.
            If deterministic, only one scenario is used.
        """
        m.sc_des_idx = RangeSet(initialize=(range(self.builder.n_sc_design)), name="sc_des")
        m.sc_copy_idx = RangeSet(0, self.builder.n_sc_per_design - 1, name="sc_cp")
        m.sc_var_idx = RangeSet(0, self.builder.n_sc_vars - 1, name="sc_var")
        m.dep_node_idx = RangeSet(0, self.builder.n_nodes - 1, name="dep_node")
        m.arr_node_idx = RangeSet(0, self.builder.n_nodes - 1, name="arr_node")
        m.int_com_idx = RangeSet(0, self.builder.n_int_com - 1, name="int_com")
        m.cnt_com_idx = RangeSet(0, self.builder.n_cnt_com - 1, name="cnt_com")
        m.io_idx = RangeSet(0, 1, name="io")
        m.time_idx = RangeSet([time for time in self.builder.time_steps], name="time")
        m.scnr_idx = RangeSet(0, self.builder.n_scenarios - 1, name="scnr")

        if self.builder.use_isru:
            m.isru_des_idx = Set(initialize=range(self.builder.n_isru_design), name="isru")

        return m
