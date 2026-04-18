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
    n_mis = 2
    mission_parameters = MissionParameters(
        n_mis=n_mis,  # number of missions
        n_sc_design=2,  # number of SC design
        n_sc_per_design=3,  # number of SC per design
        t_mis_tot=11,  # total single mission duration, days
        t_surf_mis=1,  # lunar surface mission duration, days
        n_crew=0,  # number of crew needed on lunar surface
        sample_mass=np.zeros(n_mis).tolist(), #[0, 0],  # sample collected from lunar surface, kg
        habit_pl_mass=np.zeros(n_mis).tolist(), #[0, 0],  # habitat and payload mass, kg
        # consumption cost (food+water+oxygen), kg/(day*person)
        consumption_cost=8.655,
        # maintenance cost, fraction/flight (0.01 means 1% per flight)
        maintenance_cost=0.01,
        time_interval=11,  # time interval between missions, days
        use_increased_pl=False,  # true if increased demand is used
    )

    objective_parameters = ObjectiveParameters(
        objective_type="fmleo", # Objective, should be "imleo" or "fmleo"
    )

    sc_parameters = SCParameters(
        isp=460,  # specific impulse, s
        oxi_fuel_ratio=5.5,  # oxidizer to fuel ratio
        prop_density=360,  # propellant density, kg/m^3
        misc_mass_fraction=0.05,  # misc mass factor
        aggressive_SC_design=False,  # true if aggressive sizng model is used
        # var_lb=[500, 0, 4000],
    )

    depot_parameters = DepotParameters(
        depot_nodes=None,
        # depot_nodes=["LEO", "LS"],
    )

    isru_parameters = ISRUParameters(
        use_isru=False,  # True if ISRU is used
        n_isru_design=0,  # number of ISRU design
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
    )

    node_details = NodeDetails(
        node_names=["Earth", "LEO", "LLO", "LS"],  # list of node names
        is_path_graph=True,
        holdover_nodes=["LEO", "LLO", "LS"],
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
    )

    scenario_dist = ScenarioDistribution(
        # sample mass for each scenario of 2nd mission
        sample_mass_2nd=[800, 900, 1100, 1200],
        # habitat and payload mass for 2nd mission
        habit_pl_mass_2nd=[2000, 2500, 3500, 4000],
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
        scenario=None, #scenario_dist,
    )
    sl_cls = SpaceLogistics(input_data)
    # sl_cls.optimizer.admm.run_alc_loop()
    # sl_cls.optimizer.pwl.solve_w_pwl_approx(pwl_increment=2500)

    # # Calculate some spacecraft params (for a spacecraft that does an
    # # out-and-back delivery of a payload)
    # sc_payload = 20.0e3
    # sc_dry_mass = 16.0e3
    # total_dv1 = (sl_cls.network_def._get_delta_v_km_s("LS", "LLO")
    #              + sl_cls.network_def._get_delta_v_km_s("LLO", "LEO"))
    # total_dv2 = total_dv1
    # z1 = np.exp(total_dv1 * 1.0e3 / (sc_parameters.isp * sc_parameters.g0))
    # z2 = z1
    # sc_prop2 = (z2 - 1) * sc_dry_mass
    # sc_prop1 = (z1 - 1) * (sc_dry_mass + sc_prop2 + sc_payload)
    # sc_prop = sc_prop1 + sc_prop2

    # # depot parameters
    # depot_payload = 100.0e3
    # depot_dry_mass = depot_payload * 0.20
    # total_dv_depot = sl_cls.network_def._get_delta_v_km_s("Earth", "LEO")
    # z_depot = np.exp(total_dv_depot * 1.0e3 / (sc_parameters.isp * sc_parameters.g0))
    # depot_prop = (z_depot - 1) * depot_dry_mass * 1.05

    fixed_sc_designs = np.array(
        [
            [
                10000,  # payload (max)
                23157.3159202894,     # propellant (max)
                12104.3364467333, # dry mass
            ],
            [
                2596.27723636357,
                48813.8307342091,
                12653.80774804,
            ],
            # [
            #     10000,
            #     36332.3613593395,
            #     15052.7143185748
            # ],
            # [
            #     500,
            #     53768.0653044512,
            #     12162.7074715113
            # ],
            # [
            #     300000,
            #     0.0,
            #     0.0
            # ],
            # [
            #     depot_payload,
            #     0.0,
            #     depot_dry_mass
            # ],
            # [
            #     sc_payload,  # payload (max)
            #     sc_prop,     # propellant (max)
            #     sc_dry_mass, # dry mass
            # ],
        ]
    )
    sl_cls.optimizer.fixed_sc.solve_network_flow_MILP(fixed_sc_designs)


if __name__ == "__main__":
    main()
