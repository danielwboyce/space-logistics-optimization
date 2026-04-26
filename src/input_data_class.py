"""
Data classes for user-defined input
Children data classes are wrapped in a parent data class `InputData`
"""

import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass, field
from bidict import bidict
from pyomo.environ import SolverFactory
from typing import Any
from math import isclose
from collections.abc import Callable
from numpy import logspace
from component_designer.isru.isru_rate_model import ISRUDesign


@dataclass
class MissionParameters:
    """Data class containing mission parameters

    Args:
        n_mis: Number of missions
        n_sc_design: Number of SC design
        n_sc_per_design: Number of SC per design
        t_mis_tot: Total single mission duration, days
        t_surf_mis: Lunar surface mission duration, days
        n_crew: Number of crew needed on lunar surface
        sample_mass: Sample collected from lunar surface, kg
            can be a list of float if value is different for each mission
        habit_pl_mass: Habitat and payload mass to lunar surface, kg
            can be a list of float if value is different for each mission
        consumption_cost: Consumption cost (food+water+oxygen), kg/(day*person)
            Defaults to 8.655 (food=1.105, water=6.37, oxygen=1.14)
        maintenance_cost: Maintenance cost per flight
            Defualts to 0.01 (1% mass per flight)
        time_interval: Time interval between missions, days. Defaults
            to 365.
        use_increased_pl (optional): True if increased demand is used.
            Defaults to False.
        increased_pl_factor (optional): If use_increased_pl is True,
            factor by which pl increases. Defaults to 1.5
    """

    n_mis: int
    n_sc_design: int
    n_sc_per_design: int
    t_mis_tot: float
    t_surf_mis: float
    n_crew: int
    sample_mass: float | list[float]
    habit_pl_mass: float | list[float]
    consumption_cost: float = 8.655
    maintenance_cost: float = 0.01
    time_interval: int = 365
    use_increased_pl: bool = False
    increased_pl_factor: float = 1.5

    def __post_init__(self):
        """Sanity check for input values"""

        assert all(
            value > 0
            for value in [
                self.n_mis,
                self.n_sc_design,
                self.n_sc_per_design,
                self.t_mis_tot,
                self.t_surf_mis,
                self.consumption_cost,
                self.maintenance_cost,
                self.time_interval,
            ]
        ), """
        Error:
        All of the following must be positive.
        Received values:
            Numer of mission: {}
            Number of SC design: {}
            Number of SC per design: {}
            Total single mission duration: {}
            Lunar surface mission duration: {}
            consumption cost: {}
            Maintenance cost: {}
            Mission time interval: {}
        """.format(
            self.n_mis,
            self.n_sc_design,
            self.n_sc_per_design,
            self.t_mis_tot,
            self.t_surf_mis,
            self.consumption_cost,
            self.maintenance_cost,
            self.time_interval,
        )
        assert all(
            value >= 0
            for value in [
                self.n_crew,
            ]
        ), """
        Error:
        This must be greater than or equal to zero.
        Received values:
            Number of crew: {}
        """.format(
            self.n_crew,
        )

        if isinstance(self.sample_mass, list):
            assert len(self.sample_mass) == self.n_mis, """
            Size of sample mass list must be the same as the number of missions."""
            assert all(value >= 0 for value in self.sample_mass)
            self.sample_mass_ls = self.sample_mass
        else:
            assert self.sample_mass >= 0
            self.sample_mass_ls = [self.sample_mass] * self.n_mis

        if isinstance(self.habit_pl_mass, list):
            assert len(self.habit_pl_mass) == self.n_mis, """
            Size of habitat+payload mass list must be the same as the number of missions."""
            assert all(value >= 0 for value in self.habit_pl_mass)
            self.habit_pl_mass_ls = self.habit_pl_mass
        else:
            assert self.habit_pl_mass >= 0
            self.habit_pl_mass_ls = [self.habit_pl_mass] * self.n_mis


@dataclass
class ObjectiveParameters:
    """
    Data class containing objective data.

    This allows the user to select either
    - IMLEO: minimizing the initial mass to low Earth orbit.
    - FMLEO: maximizing the net final mass delivered to low Earth orbit
      (calculated by using mass delivered from nodes other than the Earth,
      e.g. the lunar surface, and subtracting out mass launched from Earth).

    Args:
        objective_type: A string denoting the objective being used. This
            should be "imleo" or "fmleo".
    """

    objective_type: str = "imleo"

    def __post_init__(self):
        """Sanity check for input values"""

        assert ((self.objective_type == "imleo") or
                (self.objective_type == "fmleo")),"""
        Error:
        The objective_type string should be "imleo" or "fmleo".
            Received value: {}""".format(self.objective_type)


@dataclass
class SCParameters:
    """Data class containing SC data

    Args:
        isp: specific impulse
        oxi_fuel_ratio: SC propellant oxidizer to fuel ratio
        prop_density: SC propellant density, kg/m^3
        misc_mass_fraction: SC misc. mass fraction; higher = conservative
        var_names (optional): list of spacecraft attribute/variable names
        var_lb (optional): list of spacecraft attribute/variable lower bounds
        var_ub (optional): list of spacecraft attribute/variable upper bounds
        aggressive_SC_design (optional): True if aggressive sizing model is used
        g0 (optional): standard gravitational acceleration, m/s^2
    """

    isp: float
    oxi_fuel_ratio: float
    prop_density: float
    misc_mass_fraction: float
    var_names: list[str] = field(
        default_factory=lambda: ["payload", "propellant", "dry mass"]
    )
    var_lb: list[float] = field(default_factory=lambda: [500, 1000, 4000])
    var_ub: list[float] = field(default_factory=lambda: [10000, 300000, 100000])
    aggressive_SC_design: bool = False
    g0: float = 9.8

    def __post_init__(self):
        """Sanity check for input values"""

        assert all(
            value > 0
            for value in [
                self.isp,
                self.oxi_fuel_ratio,
                self.prop_density,
                self.misc_mass_fraction,
            ]
        ), """
        Error:
        All of the following must be positive.
        Received values:
            Specific impulse: {}
            Oxidizer to fuel ratio: {}
            Propellant density: {}
            Misc. mass fraction: {}
        """.format(
            self.isp,
            self.oxi_fuel_ratio,
            self.prop_density,
            self.misc_mass_fraction,
        )
        assert all(value > 0 for value in self.var_ub), """
        All spacecraft variable upper bounds must be positive."""
        assert len(self.var_names) == len(self.var_ub), """
        Number of spacecraft variable names and their lower bounds must be the same."""
        assert 0 <= self.misc_mass_fraction < 1, """
        Misc. mass fraction must be in (0, 1]. Received value: {}
        """.format(self.misc_mass_fraction)

        self.oxi_prop_ratio: float = 1 / (1 + self.oxi_fuel_ratio)
        self.fuel_prop_ratio: float = 1 - self.oxi_prop_ratio
        self.n_sc_vars: int = len(self.var_names)

@dataclass
class DepotParameters:
    """Data class containing depo data. It is assumed:
    - The depot is prestaged (i.e., before time_steps[0]).
    - The depot may never be moved from the node where it is initially placed.
    - The depot may store an arbitrarily large amount of non-propellant
        commodities.
    - The depot has zero structural mass.
    - The depot spacecraft type should not be considered as part of the
        spacecraft design subproblem.

    
    Args:
        depot_nodes: List of names of the nodes where depots are located. For
            now, at least, the depot_nodes should also all be holdover nodes.
    """

    depot_nodes: list[str] | None = None
    # __use_depots : bool = False
    # __n_depots: int = 0

    # def __post_init__(self):
    #     """Sanity check for input values"""

    #     if ((self.depot_nodes is not None) and (len(self.depot_nodes) > 0)):
    #         __use_depots = True
    #         __n_depots = len(self.depot_nodes)

    def get_n_depots(self):
        """Returns the number of depots.
        
        Returns:
            __n_depots: Number of depots."""
        if self.depot_nodes is not None:
            return len(self.depot_nodes)
        else:
            return 0

    def get_use_depots(self):
        """Returns whether depots are being used.
        
        Returns:
            __use_depots: True if any depots are used, False otherwise."""
        return self.get_n_depots() > 0


@dataclass(frozen=False)
class ISRUReactorParameters:
    """Data class containing ISRU data for a particular reactor type.
    
    Args:
        reactor_name: Name of the reactor.
        inputs: Either None or a dictionary, where the keys are
            commodity names of the reactor's inputs and values specify
            the proportions in which each commodity is required. The
            proportions represent a fraction of the of the reactor's
            overall production rate, so the proportions don't
            necessarily need to add up to 1 (often they will be greater
            than 1, assuming some inefficiency/waste in converting
            inputs to outputs).
        outputs: A dictionary where the keys are commodity names of the
            reactor's inputs and values specify the proportions in
            which each commodity is output (as a fraction of the
            reactor's overall production). The sum of the values must
            add up to 1.0.
        minimum_mass: Minimum mass required for reactor to be
            operational [kg].
        decay_rate: The reactor's productivity decay rate. Units:
            decayed mass [kg] per year per reactor mass [kg].
        maintenance_cost: The reactor's maintenance cost. Units:
            cost[kg] per year per reactor mass [kg].
        production_rate: This is a callable interface that returns the
            production rate of the reactor. It is assumed that the
            parameter will be the sum of the reactor mass for this type
            (provided by reactor_mass_commodity) available at the ISRU
            node. Units: total output [kg] per year per reactor mass
            [kg].
        is_production_rate_constant: True if the production_rate
            callable interface returns a constant rate. False otherwise.
        reactor_mass_commodity: Name of commodity type that corresponds
            to the reactor mass.
        pwl_breakpoints (optional): List of the preferred breakpoints
            (in the reactor_mass_commodity) for applying a piecewise
            linear approximation of the production_rate function.
    """

    reactor_name: str
    inputs: dict[str, float] | None
    outputs: dict[str, float]
    minimum_mass: float
    decay_rate: float
    maintenance_cost: float
    production_rate: Callable[[float], float]
    is_production_rate_constant: bool
    reactor_mass_commodity: str
    pwl_breakpoints: list[float] | None = None


    def __post_init__(self):
        """Sanity check for input values"""

        # Checking inputs add up to 1.0, if present
        if self.inputs is not None:
            assert(sum(self.inputs.values()) > 0.0), """
            The reactor input proportions must be at least 0.0.
            Received value:
                inputs: {}
                sum of values: {}
            """.format(self.inputs, sum(self.inputs.values()))

        # Checking outputs add up to 1.0
        assert(isclose(abs(sum(self.outputs.values())), 1.0, abs_tol=1.0e-6)), """
        The reactor output proportions must add up to 1.0.
        Received value:
            outputs: {}
            sum of values: {}
        """.format(self.outputs, sum(self.outputs.values()))

        # Checking decay rate is in 0 <= decay_rate <= 1
        assert(self.decay_rate >= 0.0 and self.decay_rate <= 1.0), """
        The reactor decay_rate must be in [0.0, 1.0] (assuming that
        the minimum operation time is 1 year).
        Received value: {}
        """.format(self.decay_rate)

        # Checking minimum mass is at least 0.0
        assert(self.minimum_mass >= 0.0), """
        The reactor's minimum mass must be at least 0.0 kg.
        Received value: {}
        """.format(self.minimum_mass)

        # Checking maintenance cost is in 0 <= maintenance_cost <= 1
        assert(self.maintenance_cost >= 0.0 and self.maintenance_cost <= 1.0), """
        The reactor maintenance_cost must be in [0.0, 1.0].
        Received value: {}
        """.format(self.maintenance_cost)

        # Checking production_rate is a callable object
        assert(callable(self.production_rate)), """
        The reactor's production_rate option must be callable.
        Received value: {}""".format(self.production_rate)

        # Checking that if is_production_rate_constant matches what is
        # returned by the production_rate callable
        test_point_1 = 1.5 * self.minimum_mass
        test_point_2 = 3.0 * test_point_1
        result_1 = self.production_rate(test_point_1)
        result_2 = self.production_rate(test_point_2)
        if self.is_production_rate_constant:
            assert(isclose(result_1, result_2)), """
            If is_production_rate_constant is True, then the production
            rates returned at different test points should be the same.
            Received values:
                is_production_rate_constant = {},
                test_point_1 = {},
                test_point_2 = {},
                production_rate(test_point_1) = {},
                production_rate(test_point_2) = {},
                isclose(production_rate(test_point_1), production_rate(test_point_1)) = {}
            """.format(
                self.is_production_rate_constant,
                test_point_1,
                test_point_2,
                self.production_rate(test_point_1),
                self.production_rate(test_point_2),
                isclose(result_1, result_2)
            )
        else:
            assert(not isclose(result_1, result_2)), """
            If is_production_rate_constant is False, then the production
            rates returned at different test points should be different.
            Received values:
                is_production_rate_constant = {},
                test_point_1 = {},
                test_point_2 = {},
                production_rate(test_point_1) = {},
                production_rate(test_point_2) = {},
                isclose(production_rate(test_point_1), production_rate(test_point_1)) = {}
            """.format(
                self.is_production_rate_constant,
                test_point_1,
                test_point_2,
                self.production_rate(test_point_1),
                self.production_rate(test_point_2),
                isclose(result_1, result_2)
            )

        # Checking the piecewise linear breakpoints if they exist,
        # generating some if they don't
        if not self.is_production_rate_constant:
            if self.pwl_breakpoints is not None and len(self.pwl_breakpoints) > 0:
                assert(all(pwl_bp >= 0.0 for pwl_bp in self.pwl_breakpoints)), """
                If piecewise linear breakpoints are received, they must
                all be positive.
                Received value: {}
                """.format(self.pwl_breakpoints)
            else:
                if abs(self.minimum_mass) <= 1.0e-6:
                    self.pwl_breakpoints = logspace(0.0, 10.0e3, 7)
                else:
                    self.pwl_breakpoints = [0.0]
                    self.pwl_breakpoints.extend(logspace(self.minimum_mass, 10.0e3, 6))
        else:
            assert(self.pwl_breakpoints is None or len(self.pwl_breakpoints) == 0), """
            If production rate is constant, then pwl_breakpoints
            shouldn't be provided.
            Received values:
                is_production_rate_constant = {},
                pwl_breakpoints = {}
            """.format(
                self.is_production_rate_constant,
                self.pwl_breakpoints,
            )



@dataclass(frozen=True)
class ISRUParameters:
    """Data class containing ISRU data
    Note that H2O produces LO2/LH2 + extra O2

    Args:
        use_isru: True if ISRU is used
        isru_designs: List of ISRU design data, where each entry is an
            ISRUReactorParameters object.
    """

    use_isru: bool
    use_convex_relaxation: bool = False
    isru_designs: list[ISRUReactorParameters] = field(
        default_factory=lambda: [
            ISRUReactorParameters(
                reactor_name="carbothermal_O2H2",
                inputs=None,
                outputs={"oxygen": 1.0 - 1.0/9.0, "hydrogen": 1.0/9.0},
                minimum_mass=400.0,
                decay_rate=0.1,
                maintenance_cost=0.05,
                production_rate=ISRUDesign.get_isru_rate_carbothermal_O2H2,
                is_production_rate_constant=False,
                reactor_mass_commodity="plant_carbothermal_O2H2",
                pwl_breakpoints=[0, 400, 2000, 4000, 6000, 8000, 10000],
            )
        ]
    )

    def get_isru_io_list(self) -> list[str]:
        """Returns a list of all the input and output commodities used by the
        ISRU designs provided.
        """
        isru_io: set[str] = set([])
        for isru_design in self.isru_designs:
            isru_io.add(isru_design.reactor_mass_commodity)
            if isru_design.inputs is not None:
                for input in isru_design.inputs.keys():
                    isru_io.add(input)
            for output in isru_design.outputs.keys():
                isru_io.add(output)
        return sorted(list(isru_io))

    @staticmethod
    def get_mass_upper_bound() -> float:
        return 100.0e3

    def __post_init__(self):
        """Sanity check for input values"""

        if self.use_isru:
            assert(len(self.isru_designs) > 0), """
            Error:
            If use_isru=True, then at least one ISRU reactor design
            must be provided.
            """

            list_of_reactor_names = [
                isru_design.reactor_name for isru_design in self.isru_designs
            ]
            set_of_reactor_names = set(list_of_reactor_names)
            assert(len(set_of_reactor_names) == len(list_of_reactor_names)), """
            Error:
            The reactor_name values for the entries in the isru_designs
            list must all be unique.
            Received values:
                List of reactor_names: {}
            """.format(list_of_reactor_names)

            # # ZZZ TEMP
            # assert(len(self.isru_designs) == 1),"""
            # For now, only 1 ISRU design is allowed.
            # Received value:
            #     Number of ISRU design: {}
            # """.format(len(self.isru_designs))


@dataclass(frozen=True)
class ALCParameters:
    """Data class containing ALC parameters

    Args:
        initial_weight_coefficient: ALC parameter
        weight_update_coefficient: ALC parameter
        weight_update_fraction: ALC parameter
        tol_outer: ALC outer loop tolerance
        tol_inner: ALC inner loop tolerance
        prioritized_var_name (optional): Name of variable that may be
            prioritized in ALC subproblem. Defaults to None.
        use_admm (optional): If set True, uses ADMM as part of ALC
            subproblem; if False it doesn't. Defaults to False.
        parallel_mode (optional): True if subproblems solved in
            parallel. Defaults to False.
        update_initial_weight (optional): indicator to update initial
            weight based on the first inner loop iteration. Defaults to
            False.
    """

    initial_weight: float
    initial_weight_coefficient: float
    weight_update_coefficient: float
    weight_update_fraction: float
    tol_outer: float
    tol_inner: float
    prioritized_var_name: str | None = None
    use_admm: bool = False
    parallel_mode: bool = False
    update_initial_weight: bool = False

    def __post_init__(self):
        """Sanity check for input values"""

        assert self.initial_weight > 0, """
        Initial weight must be positive. Received value: {}
        """.format(self.initial_weight)

        assert 1e-3 < self.initial_weight_coefficient < 1, """
        Initial weight coefficient must be in (1e-3, 1). Received value: {}
        """.format(self.initial_weight_coefficient)

        assert self.weight_update_coefficient > 1, """
        Weight update coefficient must be greater than 1.
        Recommended range is (2,3). Received value: {}
        """.format(self.weight_update_coefficient)

        assert 0 < self.weight_update_fraction < 1, """
        Weight update fraction must be in (0,1). Recommended value is 0.25.
        Received value: {}""".format(self.weight_update_fraction)

@dataclass(frozen=True)
class SupplyDemandDetails:
    """Data class containing details of supply and demand.

    Args:
        commodity_name: Name of the commodity being supplied or
            demanded.
        node_name: Name of the node where the supply or demand occurs.
        mission: Either a valid integer denoting the mission of
            interest or the string "all".
        io: String denoting whether the supply is at mission starts,
            mission ends, or both. Valid values are "start", "end", or
            "all".
        value: Float containing the amount of supply or demand being
            specified. If positive, it denotes a supply of the
            commodity at the node. If negative, it specifies a demand.
            The value may never be zero, but float(inf) and float(-inf)
            are valid possible values.
    """

    commodity_name: str
    node_name: str
    mission: int | str
    io: str
    value: float

    def __post_init__(self):
        """Sanity checks for input values."""
        
        assert(self.mission == "all" or (isinstance(self.mission, int) and self.mission >= 0)),"""
        The mission input must either be "all" or a positive integer.
        Received value:
            mission = {}
        """.format(self.mission)
        
        assert(self.io == "all" or self.io == "start" or self.io == "end"),"""
        The io input must either be "start", "end", or "all".
        Received value:
            io = {}
        """.format(self.io)

        assert(isinstance(self.value, float | int) and self.value != 0.0), """
        The value must be a nonzero float.
        Received value:
            value = {}
        """.format(self.value)

@dataclass
class CommodityDetails:
    """Data class containing commodity details
    Args:
        int_com_names: List of integer commodity names
        int_com_costs: List of integer commodity costs per unit
        cnt_com_names: List of continuous commodity names
        prop_com_names (optional): List of propellant commodity names
        supply_demand_list (optional): List of SupplyDemandDetails
            objects specifying which nodes supply or demand which
            commodities. Note multiple objects may never refer have the
            same combination of commodity_name, node_name, mission,
            
    """

    int_com_names: list[str]
    int_com_costs: list[float]
    cnt_com_names: list[str]
    prop_com_names: list[str] = field(default_factory=lambda: ["oxygen", "hydrogen"])
    supply_demand_list: list[SupplyDemandDetails] = field(
        default_factory = lambda:  CommodityDetails._create_default_supply_demand_list(
            4, 2, 3, 8.655, [2000.0, 3000.0], [1000.0, 1100.0])
    )

    @staticmethod
    def _create_default_supply_demand_list(
            n_crew: int,
            n_mis: int,
            t_surf_mis: int,
            consumption_cost: float,
            habitat_pl_mass: list[float] | None = None,
            sample_mass: list[float] | None = None,
    ) -> list[SupplyDemandDetails]:
        retval = [
                SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", "all", "start", float("inf")),
                SupplyDemandDetails("maintenance",             "Earth", "all", "start", float("inf")),
                SupplyDemandDetails("oxygen",                  "Earth", "all", "start", float("inf")),
                SupplyDemandDetails("hydrogen",                "Earth", "all", "start", float("inf")),
        ]
        if n_crew > 0:
            retval.extend([
                SupplyDemandDetails("consumption",             "Earth", "all", "start", float("inf")),
                SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * consumption_cost)])
            retval.extend([
                SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
                SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
                SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
                SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew)])
        if habitat_pl_mass is not None and any(mass > 0.0 for mass in habitat_pl_mass):
            retval.append(
                SupplyDemandDetails("habitat",                 "Earth", "all", "start", float("inf")))
            retval.extend([
                SupplyDemandDetails("habitat",                 "LS",    i,     "start", -habitat_pl_mass[i])
                for i in range(n_mis)
            ])
        if sample_mass is not None and any(mass > 0.0 for mass in sample_mass):
            retval.append(
                SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")))
            retval.extend([
                SupplyDemandDetails("sample",                  "Earth", i,     "end",   -sample_mass[i])
                for i in range(n_mis)
            ])
        return retval


    def __post_init__(self):
        """Sanity check for input values and define derived variables"""

        assert len(self.int_com_names) == len(self.int_com_costs), """
        Number of integer commodity names and costs must be the same."""

        assert all(cost > 0 for cost in self.int_com_costs), """
        All commodity costs must be positive."""

        assert all(prop in self.cnt_com_names for prop in self.prop_com_names), """
        All propellant commodity names must be
        in the continuous commodity names list."""

        supply_demand_tuples = [
            (detail.commodity_name, detail.node_name, detail.mission, detail.io)
            for detail in self.supply_demand_list
        ]
        assert all(
            supply_demand[0] in self.int_com_names + self.cnt_com_names
            for supply_demand in supply_demand_tuples
        ), """
        All commodity names in the supply_demand_list must be in one of
        the commodity names lists.
        Received values:
            Supply/Demand Commodity Names = {},
            Integer Commodity Names = {},
            Continuous Commodity Names = {}
        """.format(
            [supply_demand[0] for supply_demand in supply_demand_tuples],
            self.int_com_names,
            self.cnt_com_names,
        )

        assert(len(supply_demand_tuples) == len(set(supply_demand_tuples))), """
        All combinations of commodity_name, node_name, mission, io must be unique.
        Received values:
            Supply/Demand List values = {}
        """.format(supply_demand_tuples)

        self.n_int_com: int = len(self.int_com_names)
        self.n_cnt_com: int = len(self.cnt_com_names)
        self.n_com: int = self.n_int_com + self.n_cnt_com
        self.cnt_com_costs: list[float] = [1.0] * self.n_cnt_com
        self.com_names: list[str] = self.int_com_names + self.cnt_com_names
        self.com_costs: list[float] = self.int_com_costs + self.cnt_com_costs
        self.nonprop_com_names: list[str] = [
            com for com in self.com_names if com not in self.prop_com_names
        ]


@dataclass
class NodeDetails:
    """Data class containing node details

    Args:
        node_names: List of node names
        is_path_graph (optional): True if the defined graph is a path graph
            (a graph with only one path like o-o-o-o). Defaults to True.
        outbound_path (optional): Sequence of nodes from source node
            to desitnation, in terms of node names.
            Only needed if the graph is a path graph.
        holdover_nodes (optional): Set of nodes where holdover arcs are allowed
        inbound_path (optional): Sequence of nodes from destination to source,
            in terms of node names. If not specified, reverse of outboud is assumed.
        source_node (optional): Name of source node. Defaults to
            outbound_path[0].
        destination_node (optional): Name of destination node. Defaults
            to outbound_path[-1].
    """

    node_names: list[str]
    is_path_graph: bool = True
    outbound_path: list[str] = field(
        default_factory=lambda: ["Earth", "LEO", "LLO", "LS"]
    )
    inbound_path: list[str] | None = None
    holdover_nodes: list[str] = field(default_factory=lambda: ["LLO", "LS"])
    source_node: str | None = None
    destination_node: str | None = None

    def __post_init__(self):
        self.n_nodes = len(self.node_names)

        if self.is_path_graph:
            assert all(node in self.node_names for node in self.outbound_path), """
            One or more nodes in the specified outbound path
            (sequense of nodes from source to destination) cannot be found in
            the node name list. If the graph is not a path graph,
            set is_path_graph to False."""
            assert all(node in self.outbound_path for node in self.node_names), """
            Not all nodes appear in the specified outbound path
            (sequense of nodes from source to destination).
            If the graph is not a path graph, set is_path_graph to False."""
            for node in self.holdover_nodes:
                assert node in self.node_names, """
                Node {} in holdover nodes is not in the defined set of nodes.""".format(
                    node
                )
            if self.inbound_path:
                assert self.inbound_path == self.outbound_path[::-1]
            else:
                self.inbound_path = self.outbound_path[::-1]
            if not self.source_node:
                self.source_node = self.outbound_path[0]
            else:
                assert self.source_node in self.node_names, """
                The specified source node is not in the defined set of nodes."""
            if not self.destination_node:
                self.destination_node = self.outbound_path[-1]
            else:
                assert self.destination_node in self.node_names, """
                The specified destination node is not in the defined set of nodes."""
        else:
            warnings.warn(
                """The specified graph is not a path graph.
                Some features may be limited, especially in the output file"""
            )


@dataclass
class ScenarioDistribution:
    """Data class containing scnenario info and distribution for 2SSP

    Args:
        sample_mass_2nd: Sample mass collected from LS for each scenario of 2nd mission
        habit_pl_mass_2nd: Habitat and payload mass for each scenario of 2nd mission
        scenario_prob(optional): List of scenario probabilities.
            Defaults to equal probability.
    """

    sample_mass_2nd: list[float]
    habit_pl_mass_2nd: list[float]
    scenario_prob: list[float] = field(default_factory=lambda: [])

    def __post_init__(self):
        assert len(self.sample_mass_2nd) == len(self.habit_pl_mass_2nd), """
        Size of sample mass distribution and
        habitat/payload mass distribution must be the same."""
        assert all(
            value >= 0 for value in self.sample_mass_2nd + self.habit_pl_mass_2nd
        ), """
        All values in sample and habitat mass must be nonnegative.
        """
        self.n_scenarios: int = len(self.sample_mass_2nd)
        if self.scenario_prob:
            assert len(self.scenario_prob) == len(self.sample_mass_2nd), """
            Size of scenarios probability must be the same as the other distributions."""
            assert sum(self.scenario_prob) == 1, """
            Sum of scenario probabilities must be 1."""
            assert all(value > 0 and value < 1 for value in self.scenario_prob), """
                All scenario probabilities must be in (0, 1).
                """
        else:
            self.scenario_prob = [1 / self.n_scenarios] * self.n_scenarios


@dataclass
class RuntimeSettings:
    """Data class containing code settings

    Args:
        pwl_increment_list: List of PWL increments to try
        store_results_to_csv(optional): True if results stored to a .csv file.
            Defaults to False.
        mip_solver(optional): MIP solver name. Defaults to "gurobi".
        mip_subsolver(optional): MIP subsolver name for spatial branch &
            bounds. Defaults to "cplex"
        max_time(optional): maximum computation time in seconds. Defaults to
            1000.
        max_time_wo_imprv(optional): maximum computation time without
            improvement in seconds. Used for Baron and defaults to 1000.
        max_threads(optional): maximum number of CPU threads allowed to use
            for computation
        solver_verbose(optional): True if solver output is needed on terminal.
            Defaults to False.
        keep_files(optional): True if misc. solver output files (log file,
            solution file, and model file) are kept. Defaults to False.
        files_postfix(optional): If keep_files is True, a postfix that 
            may be added to log files to distinguish them.
        cplex_path(optional): Abosolute path to CPLEX executable
            (libcplex****.so). Relative path is not supported.
    """

    pwl_increment_list: list[float]
    store_results_to_csv: bool = False
    mip_solver: str = field(default_factory=lambda: "gurobi")
    mip_subsolver: str = field(default_factory=lambda: "cplex")
    max_time: float = 1000
    max_time_wo_imprv: float = 1000
    max_threads: int = mp.cpu_count()
    solver_verbose: bool = False
    keep_files: bool = False
    files_postfix: str = ""
    cplex_path: str = field(
        default_factory=lambda: (
            "/home/masafumi/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/libcplex2211.so"
        )
    )

    def __post_init__(self):
        if self.mip_solver in ["gurobi", "Gurobi", "GUROBI"]:
            self.mip_solver = "gurobi"
        elif self.mip_solver in ["cplex", "CPLEX", "Cplex"]:
            self.mip_solver = "cplex"
        elif self.mip_solver in ["Baron", "baron"]:
            self.mip_solver = "baron"
            if self.mip_subsolver in ["cplex", "CPLEX", "Cplex"]:
                self.mip_subsolver = "cplex"
            else:
                warnings.warn("""
                Specified MIP subsolver is not compatible with Baron or already the default subsolver of Baron.
                Proceeding with the default MIP subsolver.""")
        elif not SolverFactory(self.mip_solver).available():
            raise ValueError(
                """Invalid MIP solver.
                Please make sure the specified solver is supported by pyomo,
                and the solver is available in your environment."""
            )

        if self.mip_solver == "baron" and not os.path.exists(self.cplex_path):
            print(
                """CPLEX path is not found - unless the environmental variables are configured for CPLEX,
                another subsolver will be used for Baron """
            )

        assert self.max_threads <= mp.cpu_count(), """
        Maximum number of threads must be less than
        or equal to the number of CPU threads."""
        assert all(value > 0 for value in [self.max_time, self.max_threads]), """
        Maximum time and thread count must be positive."""


@dataclass
class InputData:
    """Parent data class containing all input parameters"""

    mission: MissionParameters
    objective: ObjectiveParameters
    alc: ALCParameters
    sc: SCParameters
    depot: DepotParameters
    isru: ISRUParameters
    comdty: CommodityDetails
    node: NodeDetails
    runtime: RuntimeSettings
    scenario: ScenarioDistribution | None = None

    def __post_init__(self):
        self._create_bidicts()
        self._check_depots()
        self._check_supply_demand_data()
        self._check_isru_commodity_parameters()

        if self.alc.prioritized_var_name:
            assert self.alc.prioritized_var_name in self.sc.var_names, """
            prioritized variable name is not valid"""

        self.n_scenarios: int = 1
        self.is_stochastic: bool = False
        self.scenario_prob: list[float] = [1]
        if self.scenario:
            self.has_scenario_info: bool = True
        else:
            self.has_scenario_info: bool = False
        if self.scenario and self.mission.n_mis > 2:
            raise ValueError(
                "Stochastic optimization is only supported up to two stages."
            )

    def activate_stochasticity(self):
        if not self.scenario:
            raise ValueError("Scenario information is missing.")
        self.n_scenarios: int = self.scenario.n_scenarios
        self.is_stochastic: bool = True
        self.scenario_prob: list[float] = self.scenario.scenario_prob

    def _check_depots(self):
        """Sanity checks for depots."""
        if self.depot.get_use_depots():
            assert all(
                depot in self.node.holdover_nodes
                for depot in self.depot.depot_nodes
            ), """
            Error:
            If depots are used, all of the depot nodes must be listed in the
            list of holdover nodes (NodeParameters).
            Received values:
                DepotParameters depot_nodes: {}
                NodeParameters holdover_nodes: {}
            """.format(
                self.depot.depot_nodes,
                self.node.holdover_nodes,
            )

    def _check_supply_demand_data(self):
        """Sanity checks for supply_demand_list."""
        for supply_demand_data in self.comdty.supply_demand_list:
            comdty_name = supply_demand_data.commodity_name
            node_name = supply_demand_data.node_name
            assert(node_name in self.node.node_names), """
            Unrecognized node in supply_demand_list for commodity {}
                Received value: {}
                Allowed node names: {}""".format(
                    comdty_name,
                    node_name,
                    self.node.node_names
                )

            mission = supply_demand_data.mission
            if isinstance(mission, int):
                assert(mission < self.mission.n_mis),"""
            "If mission is an integer, it should be less than the total number of missions.
                Received value: {}
                Number of missions: {}""".format(
                    comdty_name,
                    mission,
                    self.mission.n_mis
                )

    def _check_isru_commodity_parameters(self):
        """Sanity checks of commodities pointed to the ISRU parameters."""

        if not self.isru.use_isru:
            return
        
        for reactor_design in self.isru.isru_designs:
            reactor_name = reactor_design.reactor_name
            if reactor_design.inputs is not None:
                assert(
                    all(
                        input in self.comdty.com_names
                        for input in reactor_design.inputs.keys()
                    )
                ), """
                Error:
                All input commodity types in the {} reactor parameters
                must also be listed as one of the commodities in the
                CommodityDetails input.
                Received values:
                    Reactor inputs: {}
                    List of all commodities: {}
                """.format(
                    reactor_name,
                    reactor_design.inputs.keys(),
                    self.comdty.com_names,
                )

            assert(
                all(
                    output in self.comdty.com_names
                    for output in reactor_design.outputs.keys()
                )
            ), """
            Error:
            All output commodity types in the {} reactor parameters
            must also be listed as one of the commodities in the
            CommodityDetails input.
            Received values:
                Reactor outputs: {}
                List of all commodities: {}
            """.format(
                reactor_name,
                reactor_design.outputs.keys(),
                self.comdty.com_names,
            )

            assert(
                reactor_design.reactor_mass_commodity in self.comdty.com_names
            ), """
            Error:
            The reactor mass commodity type in the {} reactor
            parameters must also be listed as one of the commodities in
            the CommodityDetails input.
            Received values:
                Reactor mass commodity: {}
                List of all commodities: {}
            """.format(
                reactor_name,
                reactor_design.reactor_mass_commodity,
                self.comdty.com_names,
            )

    def _create_bidicts(self):
        """Create bidirectional dictionaries (key <-> attribute)

        This is for avoiding hardcoding of indices.
        The indicies can be extracted by names, and vice veersa.
        E.g., dict['out'] -> 0, dict[0] -> 'out'
        Bidicts created for integer and continusous commodity,
        total commodity (integer and continuous commodity with shared indices),
        node, flow in/out, SC variable/attribute
        """
        com_dict: dict[str, int] = {}
        int_com_dict: dict[str, int] = {}
        cnt_com_dict: dict[str, int] = {}
        depot_dict: dict[str, int] = {}
        isru_reactor_dict: dict[str, int] = {}
        isru_io_dict: dict[str, int] = {}

        for com_id in range(self.comdty.n_int_com):
            int_com_name = self.comdty.int_com_names[com_id]
            com_dict[int_com_name] = com_id
            int_com_dict[int_com_name] = com_id

        for com_id in range(self.comdty.n_cnt_com):
            cnt_com_name = self.comdty.cnt_com_names[com_id]
            com_dict[cnt_com_name] = com_id + self.comdty.n_int_com
            cnt_com_dict[cnt_com_name] = com_id

        node_dict: dict[str, int] = {}
        for node_id in range(self.node.n_nodes):
            node_name = self.node.node_names[node_id]
            node_dict[node_name] = node_id

        for depot_id in range(self.depot.get_n_depots()):
            depot_name = self.depot.depot_nodes[depot_id]
            depot_dict[depot_name] = depot_id

        for isru_id in range(len(self.isru.isru_designs)):
            isru_reactor_name = self.isru.isru_designs[isru_id].reactor_mass_commodity
            isru_reactor_dict[isru_reactor_name] = isru_id

        isru_io_list = self.isru.get_isru_io_list()
        for isru_io_id in range(len(isru_io_list)):
            isru_io_comdty = isru_io_list[isru_io_id]
            isru_io_dict[isru_io_comdty] = isru_io_id

        flow_dict: dict[str, int] = {"out": 0, "in": 1}

        sc_var_dict: dict[str, int] = {}
        for sc_var_id in range(self.sc.n_sc_vars):
            sc_var_name = self.sc.var_names[sc_var_id]
            sc_var_dict[sc_var_name] = sc_var_id

        # make dictionaries bidriectional and store them as attributes
        self.com_dict = bidict(com_dict)
        self.int_com_dict = bidict(int_com_dict)
        self.cnt_com_dict = bidict(cnt_com_dict)
        self.node_dict = bidict(node_dict)
        self.depot_dict = bidict(depot_dict)
        self.isru_reactor_dict = bidict(isru_reactor_dict)
        self.isru_io_dict = bidict(isru_io_dict)
        self.flow_dict = bidict(flow_dict)
        self.sc_var_dict = bidict(sc_var_dict)
