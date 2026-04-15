from input_data_class import (
    InputData,
    MissionParameters,
    SCParameters,
    DepotParameters,
    ISRUParameters,
    ALCParameters,
    CommodityDetails,
    NodeDetails,
    RuntimeSettings,
    ScenarioDistribution,
)
from opt_model_builder.model_helpers import (
    get_dependency_matrix,
)
from bidict import bidict
import sys

try:
    sys.path.append(".")
    sys.path.append("..")
    from lib.alc.alc import DimensionConverter
except (ModuleNotFoundError, ImportError):
    from lib.alc.alc import DimensionConverter


class InitMixin:
    """Mixin class to initialize attributes for different class instances."""

    def __init__(self, input_data: InputData):
        """
        Args:
            input_data: InputData dataclass containing data input from user
        """
        # dataclass arguments
        self.input_data: InputData = input_data
        self.mis: MissionParameters = input_data.mission
        self.sc: SCParameters = input_data.sc
        self.depot: DepotParameters = input_data.depot
        self.isru: ISRUParameters = input_data.isru
        self.alc: ALCParameters = input_data.alc
        self.comdty: CommodityDetails = input_data.comdty
        self.node: NodeDetails = input_data.node
        self.runtime: RuntimeSettings = input_data.runtime
        if input_data.scenario:
            self.scnr: ScenarioDistribution = input_data.scenario

        # individual data from dataclasses
        self.n_mis: int = input_data.mission.n_mis
        """Number of missions."""
        self.n_sc_design: int = input_data.mission.n_sc_design
        """Number of SC design"""
        self.n_sc_per_design: int = input_data.mission.n_sc_per_design
        """Number of SC per design"""
        self.t_mis_tot: float = input_data.mission.t_mis_tot
        """Total single mission duration, days"""
        self.t_surf_mis: float = input_data.mission.t_surf_mis
        """Lunar surface mission duration, days"""
        self.n_crew: int = input_data.mission.n_crew
        """Number of crew needed on lunar surface"""
        # FIXME: for attributes that are changed later, it may be better to
        # avoid copying the list directly.
        self.sample_mass_ls: float | list[float] = input_data.mission.sample_mass
        """Sample collected from lunar surface, kg. Can be a list of
            float if value is different for each mission"""
        self.habit_pl_mass_ls: float | list[float] = input_data.mission.habit_pl_mass
        """Habitat and payload mass to lunar surface, kg. Can be a list
            of float if value is different for each mission"""
        self.use_increased_pl: bool = input_data.mission.use_increased_pl
        """True if increased demand is used. Defaults to False."""
        self.use_depots: bool = self.depot.get_use_depots()
        """True if depots are used. Defaults to False."""
        self.depot_nodes : list[str] | None = self.depot.depot_nodes
        """List of names where the depots are located."""
        self.n_depots : int = self.depot.get_n_depots()
        """Number of depots."""
        self.depot_sc_des_idx : int = -1
        """The index for the sc designation that corresponds to depots."""
        self.use_isru: bool = input_data.isru.use_isru
        """True if ISRU is used"""
        self.n_isru_design: int = input_data.isru.n_isru_design
        """Number of ISRU design"""
        self.n_isru_vars: int = input_data.isru.n_isru_vars
        """Number of variables per ISRU design. Defaults to 1"""
        self.com_dict: bidict[str, int] = input_data.com_dict
        """Bidrectional dictionary corresponding to commodity names and indicies."""
        self.int_com_dict: bidict[str, int] = input_data.int_com_dict
        """Bidrectional dictionary corresponding to integer commodity names and indicies."""
        self.cnt_com_dict: bidict[str, int] = input_data.cnt_com_dict
        """Bidrectional dictionary corresponding to continuous commodity names and indicies."""
        self.node_dict: bidict[str, int] = input_data.node_dict
        """Bidrectional dictionary corresponding to node names and indicies."""
        self.flow_dict: bidict[str, int] = input_data.flow_dict
        """Bidrectional dictionary corresponding to flow names and indicies."""
        self.sc_var_dict: bidict[str, int] = input_data.sc_var_dict
        """Bidrectional dictionary corresponding to spacecraft variable names and indicies."""
        self.n_com: int = input_data.comdty.n_com
        """Total number of commodities."""
        self.n_int_com: int = input_data.comdty.n_int_com
        """Total number of integer commodities."""
        self.n_cnt_com: int = input_data.comdty.n_cnt_com
        """Total number of continuous commodities."""
        self.int_com_names: list[str] = input_data.comdty.int_com_names
        """List of integer commodities' names."""
        self.int_com_costs: list[float] = input_data.comdty.int_com_costs
        """List of integer commodities' cost coefficients."""
        self.cnt_com_names: list[str] = input_data.comdty.cnt_com_names
        """List of continuous commodities' names."""
        self.cnt_com_costs: list[float] = input_data.comdty.cnt_com_costs
        """List of continuous commodities' cost coefficients."""
        self.prop_com_names: list[str] = input_data.comdty.prop_com_names
        """List of propellant commodities' names."""
        self.n_sc_vars: int = input_data.sc.n_sc_vars
        """Total number of spacecraft dimensions."""
        self.n_nodes: int = input_data.node.n_nodes
        """Total number of location nodes."""
        self.dc: DimensionConverter = DimensionConverter(
            dependency_matrix=get_dependency_matrix(self),
            dim_all_var=self.n_sc_vars * self.n_sc_design,  # lf.n_isru_design,
        )
        """Instance of DimensionConverter class."""
        self.is_stochastic: bool = input_data.is_stochastic
        """Attribute controlling whether the scenarios are stochastic."""
        self.n_scenarios: int = input_data.n_scenarios
        """Number of scenarios."""
        if hasattr(self, "scnr"):
            self.sample_mass_2nd: list[float] = self.scnr.sample_mass_2nd
            """Sample mass collected from LS for each scenario of 2nd mission"""
            self.habit_pl_mass_2nd: list[float] = self.scnr.habit_pl_mass_2nd
            """Habitat and payload mass for each scenario of 2nd mission"""
            self.scenario_prob: list[float] = self.scnr.scenario_prob
            """List of scenario probabilities. Defaults to equal
                probability."""
