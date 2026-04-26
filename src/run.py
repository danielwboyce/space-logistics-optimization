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
    ISRUReactorParameters,
    ISRUParameters,
    ALCParameters,
    SupplyDemandDetails,
    CommodityDetails,
    NodeDetails,
    RuntimeSettings,
    ScenarioDistribution,
)
from component_designer.isru.isru_rate_model import ISRUDesign


def main():
    # Revision string
    revision = "06_7c59572"
    # revision = None

    # Scenario: 2 crewed missions, no isru, no depots
    if False:
        n_mis = 2
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        crew_consumption_cost = 8.655
        sample_mass = [1000, 1100]
        habit_pl_mass = [2000, 3000]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = None
        use_isru = False
        use_convex_relaxation = False
        isru_designs = None # If this is set to None, we'll just use the default
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ]
        supply_demand_list = [
            SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
        ]
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
        n_mis = 2
        t_mis_tot = 11
        t_surf_mis = 1
        n_crew = 0
        crew_consumption_cost = 8.655
        sample_mass = np.zeros(n_mis).tolist()
        habit_pl_mass = np.zeros(n_mis).tolist()
        time_interval = 11
        objective_type = "fmleo"
        isp = 460.0
        depot_nodes = None
        use_isru = False
        use_convex_relaxation = False
        isru_designs = None # If this is set to None, we'll just use the default
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ]
        supply_demand_list = [
            # SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            # SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            # SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            # SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            # SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            # SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            # SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            # SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            # SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
            SupplyDemandDetails("oxygen",                  "LS",    "all", "end",   float("inf")),
            SupplyDemandDetails("hydrogen",                "LS",    "all", "end",   float("inf")),
            SupplyDemandDetails("oxygen_storage",          "LS",    "all", "end",   float("inf")),
        ]
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
        n_mis = 2
        t_mis_tot = 11
        t_surf_mis = 1
        n_crew = 0
        crew_consumption_cost = 8.655
        sample_mass = np.zeros(n_mis).tolist()
        habit_pl_mass = np.zeros(n_mis).tolist()
        time_interval = 11
        objective_type = "fmleo"
        isp = 460.0
        depot_nodes = ["LEO", "LS"]
        use_isru = False
        use_convex_relaxation = False
        isru_designs = None # If this is set to None, we'll just use the default
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ]
        supply_demand_list = [
            # SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            # SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            # SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            # SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            # SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            # SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            # SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            # SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            # SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", 0,     "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
            SupplyDemandDetails("oxygen",                  "LS",    "all", "end",   float("inf")),
            SupplyDemandDetails("hydrogen",                "LS",    "all", "end",   float("inf")),
            SupplyDemandDetails("oxygen_storage",          "LS",    "all", "end",   float("inf")),
        ]
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
        n_mis = 2
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        crew_consumption_cost = 8.655
        sample_mass = [1000, 1100]
        habit_pl_mass = [2000, 3000]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = None
        use_isru = True
        use_convex_relaxation = False
        isru_designs = None # If this is set to None, we'll just use the default
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ]
        supply_demand_list = [
            SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
        ]
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
    if False:
        n_mis = 2
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        crew_consumption_cost = 8.655
        sample_mass = [1000, 1100]
        habit_pl_mass = [2000, 3000]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = ["LEO", "LS"]
        use_isru = True
        use_convex_relaxation = False
        isru_designs = None # If this is set to None, we'll just use the default
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ]
        supply_demand_list = [
            SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
        ]
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
    # Scenario: 3 crewed missions, with isru, no depots
    if False:
        n_mis = 3
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        crew_consumption_cost = 8.655
        sample_mass = [300.0, 1100, 1200]
        habit_pl_mass = [2000, 3000, 500.0]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = None
        use_isru = True
        use_convex_relaxation = False
        isru_designs = None # If this is set to None, we'll just use the default
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ]
        supply_demand_list = [
            SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            SupplyDemandDetails("habitat",                 "LS",    2,     "start", -habit_pl_mass[2]),
            SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("sample",                  "Earth", 2,     "end",   -sample_mass[2]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
        ]
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
    # Scenario: 3 crewed missions, with isru, 2 depots
    if False:
        n_mis = 3
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        crew_consumption_cost = 8.655
        sample_mass = [300.0, 1100, 1200]
        habit_pl_mass = [2000, 3000, 500.0]
        time_interval = 365
        objective_type = "imleo"
        isp = 420.0
        depot_nodes = ["LEO", "LS"]
        use_isru = True
        use_convex_relaxation = False
        isru_designs = None # If this is set to None, we'll just use the default
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
        ]
        supply_demand_list = [
            SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            SupplyDemandDetails("habitat",                 "LS",    2,     "start", -habit_pl_mass[2]),
            SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("sample",                  "Earth", 2,     "end",   -sample_mass[2]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
        ]
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
    # Scenario: 2 uncrewed missions, with multiple isru, with depots
    if True:
        n_mis = 2
        t_mis_tot = 13
        t_surf_mis = 3
        n_crew = 4
        crew_consumption_cost = 8.655
        sample_mass = [1000, 1100] #np.zeros(n_mis).tolist()
        habit_pl_mass = [2000, 3000] #np.zeros(n_mis).tolist()
        time_interval = 365
        objective_type = "fmleo"
        isp = 460.0
        depot_nodes = ["LEO", "LS"]
        use_isru = True
        use_convex_relaxation = False
        isru_designs = [
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
                pwl_breakpoints=[0, 400, 2000, 4000, 6000, 8000, 10000, 20000, 40000],
            ),
            ISRUReactorParameters(
                reactor_name="mre_metal",
                inputs=None,
                outputs={"metal": 1.0},
                minimum_mass=600.0,
                decay_rate=0.1,
                maintenance_cost=0.05,
                production_rate=ISRUDesign.get_isru_rate_mre_metal,
                is_production_rate_constant=False,
                reactor_mass_commodity="plant_mre_metal",
                pwl_breakpoints=[0, 600, 2000, 4000, 6000, 8000, 10000, 20000, 40000],
            ),
            # ISRUReactorParameters(
            #     reactor_name="workshop",
            #     inputs={"metal": 2.0},
            #     outputs={
            #         "maintenance": 1/4,
            #         "plant_carbothermal_O2H2": 1/4,
            #         "plant_mre_metal": 1/4,
            #         "plant_workshop": 1/4
            #     },
            #     minimum_mass=600.0,
            #     decay_rate=0.1,
            #     maintenance_cost=0.05,
            #     production_rate=ISRUDesign.get_isru_rate_workshop,
            #     is_production_rate_constant=True,
            #     reactor_mass_commodity="plant_workshop",
            # ),
        ]
        cnt_com_names = [
            "plant_carbothermal_O2H2",
            "plant_mre_metal",
            # "plant_workshop",
            "maintenance",
            "consumption",
            "habitat",
            "sample",
            "oxygen",
            "hydrogen",
            "oxygen_storage",
            "metal",
        ]
        supply_demand_list = [
            SupplyDemandDetails("crew #",                  "Earth", "all", "start",  n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "start", -n_crew),
            SupplyDemandDetails("habitat",                 "LS",    0,     "start", -habit_pl_mass[0]),
            SupplyDemandDetails("habitat",                 "LS",    1,     "start", -habit_pl_mass[1]),
            SupplyDemandDetails("crew #",                  "Earth", "all", "end",   -n_crew),
            SupplyDemandDetails("crew #",                  "LS",    "all", "end",    n_crew),
            SupplyDemandDetails("consumption",             "LS",    "all", "end",   -n_crew * t_surf_mis * crew_consumption_cost),
            SupplyDemandDetails("sample",                  "Earth", 0,     "end",   -sample_mass[0]),
            SupplyDemandDetails("sample",                  "Earth", 1,     "end",   -sample_mass[1]),
            SupplyDemandDetails("plant_carbothermal_O2H2", "LS",    0,     "start", 5000.0),
            SupplyDemandDetails("plant_mre_metal",         "LS",    0,     "start", 5000.0),
            SupplyDemandDetails("plant_carbothermal_O2H2", "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("plant_mre_metal",         "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("maintenance",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("consumption",             "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("habitat",                 "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("oxygen",                  "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("hydrogen",                "Earth", "all", "start", float("inf")),
            SupplyDemandDetails("sample",                  "LS",    "all", "end",   float("inf")),
        ]
        holdover_nodes = ["LEO", "LLO", "LS"]
        use_fixed_sc_designs = False
        # # These are the returns for when use_fixed_sc_designs=False
        # fixed_sc_designs = np.array(
        #     []
        # )

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

    if isru_designs is not None:
        isru_parameters = ISRUParameters(
            use_isru=use_isru,  # True if ISRU is used
            use_convex_relaxation=use_convex_relaxation,
            isru_designs=isru_designs,
        )
    else:
        isru_parameters = ISRUParameters(
            use_isru=use_isru,  # True if ISRU is used
            use_convex_relaxation=use_convex_relaxation,
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
        cnt_com_names=cnt_com_names,
        # list of propellant commodity names
        prop_com_names=["oxygen", "hydrogen"],
        supply_demand_list=supply_demand_list,
    )

    node_details = NodeDetails(
        node_names=["Earth", "LEO", "LLO", "LS"],  # list of node names
        is_path_graph=True,
        holdover_nodes=holdover_nodes, #["LEO", "LLO", "LS"],
        outbound_path=["Earth", "LEO", "LLO", "LS"],
    )

    files_postfix = ".scen_"
    if revision is not None:
        files_postfix = "." + revision + files_postfix
    files_postfix = (files_postfix
                     + str(n_mis) + "mis_"
                     + str(n_crew) + "crew_"
                     + ("0" if not isru_parameters.use_isru else str(len(isru_parameters.isru_designs))) + "isru_"
                     + ("1" if isru_parameters.use_convex_relaxation else "0") + "convexrelax_"
                     + ("0" if depot_nodes is None else str(len(depot_nodes))) + "depots_"
                     + objective_type)
    runtime_settings = RuntimeSettings(
        pwl_increment_list=[2500],  # List of PWL increment to try
        store_results_to_csv=True,  # True if results stored to a .csv file
        mip_solver="gurobi",
        mip_subsolver="cplex",
        solver_verbose=True,
        max_time=3600 * 3,  # maximum time allowed for optimization in seconds
        max_time_wo_imprv=600, #3600 * 3,
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
