import pytest
import numpy as np
from src.space_logistics import SpaceLogistics
from src.input_data_class import (
    InputData,
    MissionParameters,
    ObjectiveParameters,
    SCParameters,
    DepotParameters,
    ISRUParameters,
    ALCParameters,
    CommodityDetails,
    NodeDetails,
    RuntimeSettings,
)

mission_parameters = MissionParameters(
    n_mis=2,  # number of missions
    n_sc_design=1,  # number of SC design
    n_sc_per_design=6,  # number of SC per design
    t_mis_tot=13,  # total single mission duration, days
    t_surf_mis=3,  # lunar surface mission duration, days
    n_crew=4,  # number of crew needed on lunar surface
    sample_mass=[1000, 1000],  # sample collected from lunar surface, kg
    habit_pl_mass=[2000, 2000],  # habitat and payload mass, kg
    # consumption cost (food+water+oxygen), kg/(day*person)
    consumption_cost=8.655,
    # maintenance cost, fraction/flight (0.01 means 1% per flight)
    maintenance_cost=0.01,
    time_interval=365,  # time interval between missions, days
    use_increased_pl=False,  # true if increased demand is used
)

objective_parameters = ObjectiveParameters(
    objective_type="imleo", # Objective, should be "imleo" or "fmleo"
)

sc_parameters = SCParameters(
    isp=420,  # specific impulse, s
    oxi_fuel_ratio=5.5,  # oxidizer to fuel ratio
    prop_density=360,  # propellant density, kg/m^3
    misc_mass_fraction=0.05,  # misc mass factor
    aggressive_SC_design=False,  # true if aggressive sizng model is used
)

depot_parameters = DepotParameters(
    depot_nodes=None,
)

isru_parameters = ISRUParameters(
    use_isru=False,  # True if ISRU is used
)

alc_parameters = ALCParameters(
    initial_weight=1,
    initial_weight_coefficient=0.01,  # ALC parameter
    weight_update_coefficient=2,  # ALC parameter
    weight_update_fraction=0.5,  # ALC parameter
    tol_outer=0.001,  # outer loop tolerance
    tol_inner=0.0001,  # inner loop tolerance
    # name of shared variable you want prioritized update
    prioritized_var_name="dry mass",
    parallel_mode=False,  # True if subproblems solved in parallel
)

comdty_details = CommodityDetails(
    int_com_names=["crew #"],  # list of integer commodity names
    int_com_costs=[100],  # list of integer commodity costs
    # list of continuous commodity names
    cnt_com_names=[
        "plant_carbothermal_O2H2",
        "maintenance",
        "consumption",
        "habitat",
        "sample",
        "oxygen",
        "hydrogen",
    ],
    # list of propellant commodity names
    prop_com_names=["oxygen", "hydrogen"],
    supply_demand_list=CommodityDetails._create_default_supply_demand_list(
        n_crew=mission_parameters.n_crew,
        n_mis=mission_parameters.n_mis,
        t_surf_mis=mission_parameters.t_surf_mis,
        consumption_cost=mission_parameters.consumption_cost,
        habitat_pl_mass=mission_parameters.habit_pl_mass,
        sample_mass=mission_parameters.sample_mass
    ),
)

node_details = NodeDetails(
    node_names=["Earth", "LEO", "LLO", "LS"],  # list of node names
)

runtime_settings = RuntimeSettings(
    pwl_increment_list=[5000],  # List of PWL increment to try
    store_results_to_csv=False,  # True if results stored to a .csv file
    solver_verbose=False,
)

input_data = InputData(
    mission=mission_parameters,
    objective=objective_parameters,
    sc=sc_parameters,
    depot=depot_parameters,
    isru=isru_parameters,
    alc=alc_parameters,
    comdty=comdty_details,
    node=node_details,
    runtime=runtime_settings,
)

sl = SpaceLogistics(input_data)
# WARNING: value before refactoring, do NOT delete
# known_fixed_sc_imleo = 694223.5193465501
known_fixed_sc_imleo = 694264.4171277


def test_fixed_sc_design_optimization():
    ref_sc_des = np.array(
        [
            [
                2837.11768506991,
                44362.37800273237,
                13071.917273428024,
            ]
        ]
    )
    fixed_sc_imleo = sl.optimizer.fixed_sc.solve_network_flow_MILP(
        fixed_sc_vars=ref_sc_des
    )
    assert fixed_sc_imleo == pytest.approx(expected=known_fixed_sc_imleo, rel=1e-4)
