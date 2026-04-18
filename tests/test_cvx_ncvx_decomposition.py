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
        "carbothermal_O2_plant",
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
)

runtime_settings = RuntimeSettings(
    pwl_increment_list=[2500],  # List of PWL increment to try
    store_results_to_csv=False,  # True if results stored to a .csv file
    solver_verbose=False,
)

input_data_1sc = InputData(
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
sl_1sc = SpaceLogistics(input_data_1sc)

ref_imleo_1sc_pwl = 677034.5575013
ref_sc_des_1sc_pwl = np.array([[2827.61503357, 42879.43140199, 12725.75167836]])


def test_cvx_plw_1sc():
    res_1sc_pwl = sl_1sc.optimizer.pwl.solve_w_pwl_approx(2500)
    assert res_1sc_pwl["obj"] == pytest.approx(expected=ref_imleo_1sc_pwl, rel=1e-3)
    np.testing.assert_allclose(
        res_1sc_pwl["design vars"],
        ref_sc_des_1sc_pwl,
        rtol=0.05,
    )


# WARNING: old resutls before refactoring, do NOT delete
# ref_imleo_1sc = 676821.5088531917
# ref_sc_des_1sc = np.array(
#     [
#         [2839.0327915, 42880.94924564, 12735.35439205],
#     ]
# )

ref_imleo_1sc = 677041.0078049527
ref_sc_des_1sc = np.array([[2827.60943793, 42879.70093057, 12726.7297336]])


def test_cvx_ncvx_decomp_1sc():
    results_1sc = sl_1sc.optimizer.admm.run_alc_loop()
    assert results_1sc["objectives"][0] == pytest.approx(
        expected=ref_imleo_1sc, rel=1e-3
    )
    np.testing.assert_allclose(
        results_1sc["design vars"],
        ref_sc_des_1sc,
        rtol=0.05,
    )


mission_parameters_2sc = MissionParameters(
    n_mis=2,  # number of missions
    n_sc_design=2,  # number of SC design
    n_sc_per_design=3,  # number of SC per design
    t_mis_tot=13,  # total single mission duration, days
    t_surf_mis=3,  # lunar surface mission duration, days
    n_crew=4,  # number of crew needed on lunar surface
    sample_mass=[1000, 1000],  # sample collected from lunar surface, kg
    habit_pl_mass=[2000, 2000],  # habitat and payload mass, kg
)
input_data_2sc = InputData(
    mission=mission_parameters_2sc,
    objective=objective_parameters,
    sc=sc_parameters,
    depot=depot_parameters,
    isru=isru_parameters,
    alc=alc_parameters,
    comdty=comdty_details,
    node=node_details,
    runtime=runtime_settings,
)
sl_2sc = SpaceLogistics(input_data_2sc)
ref_imleo_2sc_pwl = 401327.7125853
ref_sc_des_2sc_pwl = np.array(
    [
        [2723.30697999, 15349.05682545, 7510.34899967],
        [507.53094029, 55695.85864014, 13829.06781459],
    ]
)


def test_cvx_plw_2sc():
    res_2sc_pwl = sl_2sc.optimizer.pwl.solve_w_pwl_approx(2500)
    assert res_2sc_pwl["obj"] == pytest.approx(expected=ref_imleo_2sc_pwl, rel=1e-3)
    np.testing.assert_allclose(
        res_2sc_pwl["design vars"],
        ref_sc_des_2sc_pwl,
        rtol=0.05,
    )


# WARNING: old resutls before refactoring, do NOT delete
# ref_imleo_2sc = 401110.91349270893
# ref_sc_des_2sc = np.array(
#     [
#         [
#             2778.66842373,
#             15408.16437385,
#             7553.66731067,
#         ],
#         [500, 55680.93563771, 13816.64965672],
#     ]
# )

ref_imleo_2sc = 401331.09130643186
ref_sc_des_2sc = np.array(
    [
        [2723.30883911, 15350.10929283, 7515.59695],
        [507.50303643, 55696.60162696, 13828.63586438],
    ]
)

# WITHOUT prioritized variable
# IMLEO:   401300.3654841139
# Design variables:  [[ 2722.36498171 15349.85138625  7512.81754549]
#  [  507.1186416  55692.63459609 13826.41600393]]


def test_cvx_ncvx_decomp_2sc():
    results_2sc = sl_2sc.optimizer.admm.run_alc_loop()
    assert results_2sc["objectives"][0] == pytest.approx(
        expected=ref_imleo_2sc, rel=1e-3
    )
    np.testing.assert_allclose(
        results_2sc["design vars"],
        ref_sc_des_2sc,
        rtol=0.05,
    )
