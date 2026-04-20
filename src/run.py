"""
This code specifies parameters/settings needed to run space logsitics/mission planning optimization
with nonlinar spacecraft (SC) sizing constraint
via decomposition-based optimization algorithm (augmented Lagrangian coordination/ALC).
One of the most impactful parameters is the increment used to generate a mesh for
piecewise linear (PWL) approximation of the nonlinear SC sizing constraint.
Although PWL approximations is only used in initial guess generation,
the initial guess quality is critical for the following nonlinear optimization performance.
The user can pass a list of increments, and the code will run for each increment.

For details, refer to:
Multidisciplinary Design Optimization Approach to Integrated Space Logistics and Mission Planning and Spacecraft Design
by M. Isaji, Y. Takubo, and K. Ho
doi: https://doi.org/10.2514/1.A35284
"""

import numpy as np
from space_logistics import SpaceLogistics
from input_data_class import (
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
    ScenarioDistribution,
)


def main():
    # Scenario: 2 crewed missions, no isru, no depots
    if False:
        files_postfix = "scen_2_crewed_no_isru_no_depots_imleo"
        n_mis = 2
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        sample_mass = [1000, 1100]
        habit_pl_mass = [2000, 3000]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = None
        use_isru = False
        n_isru_design = 0
        infinite_supply_dict={
            "plant":          [{ "node": "Earth", "mission": "all", "io": "start" }],
            "maintenance":    [{ "node": "Earth", "mission": "all", "io": "start" }],
            "consumption":    [{ "node": "Earth", "mission": "all", "io": "start" }],
            "habitat":        [{ "node": "Earth", "mission": "all", "io": "start" }],
            "oxygen":         [{ "node": "Earth", "mission": "all", "io": "start" }],
            "hydrogen":       [{ "node": "Earth", "mission": "all", "io": "start" }],
            "sample":         [{ "node": "LS",    "mission": "all", "io": "end"   }],
        }
        holdover_nodes = ["LLO", "LS"]
        use_fixed_sc_designs = False
        # These are the returns for when use_fixed_sc_designs=False
        fixed_sc_designs = np.array(
            [
                [
                     2167.59307965386, # payload (max)
                    14871.2848783733,  # propellant (max)
                     7131.58477923172, # dry mass
                ],
                [
                      500.0,          # payload (max)
                    57554.0718996,    # propellant (max)
                    14295.2063130593, # dry mass
                ],
            ]
        )
    # Scenario: 2 uncrewed missions, no isru, no depots
    if False:
        files_postfix = "scen_2_uncrewed_no_isru_no_depots_fmleo"
        n_mis = 2
        t_mis_tot = 11
        t_surf_mis = 1
        n_crew = 0
        sample_mass = np.zeros(n_mis).tolist()
        habit_pl_mass = np.zeros(n_mis).tolist()
        time_interval = 11
        objective_type = "fmleo"
        isp = 460.0
        depot_nodes = None
        use_isru = False
        n_isru_design = 0
        infinite_supply_dict={
            "plant":          [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "maintenance":    [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "consumption":    [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "habitat":        [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "oxygen":         [{ "node": "Earth", "mission": "0",   "io": "start" }, { "node": "LS",    "mission": "all", "io": "end"   }],
            "hydrogen":       [{ "node": "Earth", "mission": "0",   "io": "start" }, { "node": "LS",    "mission": "all", "io": "end"   }],
            "sample":         [{ "node": "LS",    "mission": "all", "io": "end"   }],
            "oxygen_storage": [{ "node": "LS",    "mission": "all", "io": "end"   }],
        }
        holdover_nodes = ["LEO", "LLO", "LS"]
        use_fixed_sc_designs = False
        # These are the returns for when use_fixed_sc_designs=False
        fixed_sc_designs = np.array(
            [
                [
                    10000.0,           # payload (max)
                    23157.3159202894,  # propellant (max)
                    12104.3364467333,  # dry mass
                ],
                [
                     2596.27723636357, # payload (max)
                    48813.8307342091,  # propellant (max)
                    12653.80774804,    # dry mass
                ],
            ]
        )
    # Scenario: 2 uncrewed missions, no isru, with depots
    if False:
        files_postfix = "scen_2_uncrewed_no_isru_with_depots_fmleo"
        n_mis = 2
        t_mis_tot = 11
        t_surf_mis = 1
        n_crew = 0
        sample_mass = np.zeros(n_mis).tolist()
        habit_pl_mass = np.zeros(n_mis).tolist()
        time_interval = 11
        objective_type = "fmleo"
        isp = 460.0
        depot_nodes = ["LEO", "LS"]
        use_isru = False
        n_isru_design = 0
        infinite_supply_dict={
            "plant":          [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "maintenance":    [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "consumption":    [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "habitat":        [{ "node": "Earth", "mission": "0",   "io": "start" }],
            "oxygen":         [{ "node": "Earth", "mission": "0",   "io": "start" }, { "node": "LS",    "mission": "all", "io": "end"   }],
            "hydrogen":       [{ "node": "Earth", "mission": "0",   "io": "start" }, { "node": "LS",    "mission": "all", "io": "end"   }],
            "sample":         [{ "node": "LS",    "mission": "all", "io": "end"   }],
            "oxygen_storage": [{ "node": "LS",    "mission": "all", "io": "end"   }],
        }
        holdover_nodes = ["LEO", "LLO", "LS"]
        use_fixed_sc_designs = False
        # These are the returns for when use_fixed_sc_designs=False
        fixed_sc_designs = np.array(
            [
                [
                    10000.0,           # payload (max)
                    36332.3613593395,  # propellant (max)
                    15052.7143185748,  # dry mass
                ],
                [
                      500.0,           # payload (max)
                    53768.0653044512,  # propellant (max)
                    12162.7074715113,  # dry mass
                ],
                [
                    300000.0,          # payload (max)
                         0.0,          # propellant (max)
                         0.0,          # dry mass
                ],
            ]
        )
    # Scenario: 2 crewed missions, with isru, no depots
    if False:
        files_postfix = "scen_2_crewed_with_isru_no_depots_imleo"
        n_mis = 2
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        sample_mass = [1000, 1100]
        habit_pl_mass = [2000, 3000]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = None
        use_isru = True
        n_isru_design = 1
        infinite_supply_dict={
            "plant":          [{ "node": "Earth", "mission": "all", "io": "start" }],
            "maintenance":    [{ "node": "Earth", "mission": "all", "io": "start" }],
            "consumption":    [{ "node": "Earth", "mission": "all", "io": "start" }],
            "habitat":        [{ "node": "Earth", "mission": "all", "io": "start" }],
            "oxygen":         [{ "node": "Earth", "mission": "all", "io": "start" }],
            "hydrogen":       [{ "node": "Earth", "mission": "all", "io": "start" }],
            "sample":         [{ "node": "LS",    "mission": "all", "io": "end"   }],
        }
        holdover_nodes = ["LLO", "LS"]
        use_fixed_sc_designs = False
        # These are the returns for when use_fixed_sc_designs=False
        fixed_sc_designs = np.array(
            [
                [
                     2167.59307965386, # payload (max)
                    14871.2848783733,  # propellant (max)
                     7131.58477923172, # dry mass
                ],
                [
                      500.0,          # payload (max)
                    57554.0718996,    # propellant (max)
                    14295.2063130593, # dry mass
                ],
            ]
        )
    # Scenario: 2 crewed missions, with isru, with depots
    if True:
        files_postfix = "scen_2_crewed_with_isru_with_depots_imleo"
        n_mis = 2
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        sample_mass = [1000, 1100]
        habit_pl_mass = [2000, 3000]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = ["LEO", "LS"]
        use_isru = True
        n_isru_design = 1
        infinite_supply_dict={
            "plant":          [{ "node": "Earth", "mission": "all", "io": "start" }],
            "maintenance":    [{ "node": "Earth", "mission": "all", "io": "start" }],
            "consumption":    [{ "node": "Earth", "mission": "all", "io": "start" }],
            "habitat":        [{ "node": "Earth", "mission": "all", "io": "start" }],
            "oxygen":         [{ "node": "Earth", "mission": "all", "io": "start" }],
            "hydrogen":       [{ "node": "Earth", "mission": "all", "io": "start" }],
            "sample":         [{ "node": "LS",    "mission": "all", "io": "end"   }],
        }
        holdover_nodes = ["LEO", "LLO", "LS"]
        use_fixed_sc_designs = False
        # These are the returns for when use_fixed_sc_designs=False
        fixed_sc_designs = np.array(
            [
                [
                     2167.59307965386, # payload (max)
                    14871.2848783733,  # propellant (max)
                     7131.58477923172, # dry mass
                ],
                [
                      500.0,          # payload (max)
                    57554.0718996,    # propellant (max)
                    14295.2063130593, # dry mass
                ],
            ]
        )

    # Revision string, adjust files_postfix
    files_postfix = "." + files_postfix

    revision = "a7da8f6"
    # revision = None
    if revision is not None:
        files_postfix = "." + revision + files_postfix

    mission_parameters = MissionParameters(
        n_mis=n_mis,  # number of missions
        n_sc_design=2,  # number of SC design
        n_sc_per_design=3,  # number of SC per design
        t_mis_tot=t_mis_tot,  # total single mission duration, days
        t_surf_mis=t_surf_mis,  # lunar surface mission duration, days
        n_crew=n_crew,  # number of crew needed on lunar surface
        sample_mass=sample_mass,  # sample collected from lunar surface, kg
        habit_pl_mass=habit_pl_mass,  # habitat and payload mass, kg
        # consumption cost (food+water+oxygen), kg/(day*person)
        consumption_cost=8.655,
        # maintenance cost, fraction/flight (0.01 means 1% per flight)
        maintenance_cost=0.01,
        time_interval=time_interval,  # time interval between missions, days
        use_increased_pl=False,  # true if increased demand is used
    )

    objective_parameters = ObjectiveParameters(
        objective_type=objective_type, # Objective, should be "imleo" or "fmleo"
    )

    sc_parameters = SCParameters(
        isp=isp,  # specific impulse, s
        oxi_fuel_ratio=5.5,  # oxidizer to fuel ratio
        prop_density=360,  # propellant density, kg/m^3
        misc_mass_fraction=0.05,  # misc mass factor
        aggressive_SC_design=False,  # true if aggressive sizng model is used
        # var_lb=[500, 0, 4000],
    )

    depot_parameters = DepotParameters(
        depot_nodes=depot_nodes,
    )

    isru_parameters = ISRUParameters(
        use_isru=use_isru,  # True if ISRU is used
        n_isru_design=n_isru_design,  # number of ISRU design
        H2_H2O_ratio=1 / 9,  # H2 production per H2O
        O2_H2O_ratio=1 - 1 / 9,  # O2 production per H2O
        production_rate=5,  # production [kg] per year and per mass [kg]
        decay_rate=0.1,  # productivity decay rate per year
        maintenance_cost=0.05,  # cost[kg] per year and per ISRU mass [kg]
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
        use_admm=True,  # True if ADMM is used
    )

    comdty_details = CommodityDetails(
        int_com_names=["crew #"],  # list of integer commodity names
        int_com_costs=[100],  # list of integer commodity costs
        # list of continuous commodity names
        cnt_com_names=[
            "plant",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ],
        # list of propellant commodity names
        prop_com_names=["oxygen", "hydrogen"],
        infinite_supply_dict=infinite_supply_dict,
    )

    node_details = NodeDetails(
        node_names=["Earth", "LEO", "LLO", "LS"],  # list of node names
        is_path_graph=True,
        holdover_nodes=holdover_nodes, #["LEO", "LLO", "LS"],
        outbound_path=["Earth", "LEO", "LLO", "LS"],
    )

    runtime_settings = RuntimeSettings(
        pwl_increment_list=[2500],  # List of PWL increment to try
        store_results_to_csv=True,  # True if results stored to a .csv file
        mip_solver="gurobi",
        mip_subsolver="cplex",
        solver_verbose=True,
        max_time=3600 * 3,  # maximum time allowed for optimization in seconds
        max_time_wo_imprv=3600 * 3,
        keep_files=True,
        files_postfix=files_postfix,
    )

    scenario_dist = None
    # scenario_dist = ScenarioDistribution(
    #     # sample mass for each scenario of 2nd mission
    #     sample_mass_2nd=[800, 900, 1100, 1200],
    #     # habitat and payload mass for 2nd mission
    #     habit_pl_mass_2nd=[2000, 2500, 3500, 4000],
    # )

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
        scenario=scenario_dist,
    )
    sl_cls = SpaceLogistics(input_data)
    # sl_cls.optimizer.admm.run_alc_loop()
    if use_fixed_sc_designs:
        sl_cls.optimizer.fixed_sc.solve_network_flow_MILP(fixed_sc_designs)
    else:
        sl_cls.optimizer.pwl.solve_w_pwl_approx(pwl_increment=2500)


if __name__ == "__main__":
    main()
