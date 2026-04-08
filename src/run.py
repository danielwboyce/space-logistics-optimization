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
    SCParameters,
    ISRUParameters,
    ALCParameters,
    CommodityDetails,
    NodeDetails,
    RuntimeSettings,
    ScenarioDistribution,
)


def main():
    mission_parameters = MissionParameters(
        n_mis=2,  # number of missions
        n_sc_design=2,  # number of SC design
        n_sc_per_design=3,  # number of SC per design
        t_mis_tot=13,  # total single mission duration, days
        t_surf_mis=3,  # lunar surface mission duration, days
        n_crew=4,  # number of crew needed on lunar surface
        sample_mass=[1000, 1100],  # sample collected from lunar surface, kg
        habit_pl_mass=[2000, 3000],  # habitat and payload mass, kg
        # consumption cost (food+water+oxygen), kg/(day*person)
        consumption_cost=8.655,
        # maintenance cost, fraction/flight (0.01 means 1% per flight)
        maintenance_cost=0.01,
        time_interval=365,  # time interval between missions, days
        use_increased_pl=False,  # true if increased demand is used
    )

    sc_parameters = SCParameters(
        isp=420,  # specific impulse, s
        oxi_fuel_ratio=5.5,  # oxidizer to fuel ratio
        prop_density=360,  # propellant density, kg/m^3
        misc_mass_fraction=0.05,  # misc mass factor
        aggressive_SC_design=False,  # true if aggressive sizng model is used
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
        ],
        # list of propellant commodity names
        prop_com_names=["oxygen", "hydrogen"],
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
        sc=sc_parameters,
        isru=isru_parameters,
        alc=alc_parameters,
        comdty=comdty_details,
        node=node_details,
        runtime=runtime_settings,
        scenario=scenario_dist,
    )
    sl_cls = SpaceLogistics(input_data)
    # sl_cls.optimizer.admm.run_alc_loop()
    # sl_cls.optimizer.pwl.solve_w_pwl_approx(pwl_increment=2500)
    fixed_sc_designs = np.array(
        [
            [
                2167.593079653862,
                14871.284878373288,
                7131.5847792317145,
            ],
            [
                500.0,
                57325.31844683161,
                14236.423242779982
            ]
        ]
    )
    sl_cls.optimizer.fixed_sc.solve_network_flow_MILP(fixed_sc_designs)


if __name__ == "__main__":
    main()
