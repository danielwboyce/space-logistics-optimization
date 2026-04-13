from __future__ import annotations
import numpy as np
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer_class import Optimizer

try:
    from initializer import InitMixin
except (ModuleNotFoundError, ImportError):
    sys.path.append("..")
    from initializer import InitMixin
try:
    from lib.alc.alc import InnerLoop, OuterLoop, SubproblemResult, AllSubpDict
except (ModuleNotFoundError, ImportError):
    sys.path.append("..")
    sys.path.append("...")
    from lib.alc.alc import InnerLoop, OuterLoop, SubproblemResult, AllSubpDict


class ADMMLoop(InitMixin):
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        super().__init__(self.optimizer._input_data)
        self._set_subproblem_dict()
        self._set_prioritized_var_tuple()

    @property
    def initial_guess(self) -> np.ndarray:
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, initial_guess: np.ndarray) -> None:
        assert isinstance(initial_guess, np.ndarray), (
            """Initial guess must be a numpy array."""
        )
        if initial_guess.shape != (
            self.n_sc_design,
            self.n_sc_vars,
        ):
            raise ValueError(
                """Fixed SC variables has invalid nupmy array shape.
                Received: {}
                Expected: ({},{})""".format(
                    initial_guess.shape,
                    self.n_sc_design,
                    self.n_sc_vars,
                )
            )
        self._initial_guess = initial_guess

    # TODO: modify this to run for single PWL increment,
    # then create a separate wrapper method for multiple increments
    # Also allow user-defined initial guess
    def run_alc_loop(self) -> dict:
        """Run ALC loop"""

        for pwl_increment in self.runtime.pwl_increment_list:
            t0 = time.time()
            self._solve_alc_initial_guess_problem(pwl_increment)
            inner_loop = InnerLoop(
                dc=self.dc,
                all_subprob_dict=self._subp_dict,
                initial_guess=np.ravel(self.initial_guess),
                use_admm=self.alc.use_admm,
                tol_inner=self.alc.tol_inner,
                prioritized_var=self._prioritized_var_tuple,
            )
            outer_loop = OuterLoop(
                dc=self.dc,
                inner_loop=inner_loop,
                initial_weight=self.alc.initial_weight,
                initial_weight_coefficient=self.alc.initial_weight_coefficient,
                weight_update_coefficient=self.alc.weight_update_coefficient,
                weight_update_fraction=self.alc.weight_update_fraction,
                tol_outer=self.alc.tol_outer,
                update_initial_weight=self.alc.update_initial_weight,
            )
            results = outer_loop.run()
            results["design vars"] = results["design vars"].reshape(
                self.n_sc_design, self.n_sc_vars
            )
            print("### Optimization Results ###")
            print("FMLEO:\t", results["objectives"][0])
            print("Design variables: ", results["design vars"])
            print("Run time: ", time.time() - t0)
            # WARNING: only works for single PWL increment
            return results

    def solve_alc_subprob(
        self,
        target_shared_var: np.ndarray,
        lagrange_est: np.ndarray,
        penalty_weight: np.ndarray,
        local_var_idx: list[int],
        aux_shared_var_idx: list[int],
        args=None,
        initial_guess=None,
    ) -> SubproblemResult:
        """solves Space Logistics ALC subproblem

        Can be used for both deterministic and stochastic subproblems.
        For stochastic subproblems, the stochasticity must be activated
        in InputData class instance, ie, self._input_data.activate_stochasticity()
        before calling this function.

        Args:
        Returns (list):
            FMLEO (float): optimal objective function (FMLEO) value
            sc_vars (np.array): SC design variables
        """
        self.optimizer._model_builder.mode = "ALCsubproblem"
        self.optimizer._model_builder.global_shared_vars = target_shared_var
        self.optimizer._model_builder.lagrange_mult_est = lagrange_est
        self.optimizer._model_builder.penelty_weight = penalty_weight
        model = self.optimizer._model_builder.build_model()
        model = self.optimizer.solver.solve_model(model)
        fmleo = model.fmleo.value
        sc_vars = self.optimizer.output.get_sc_vars(model)

        subp_res: SubproblemResult = {
            "objective": fmleo,
            "design var": np.ravel(sc_vars),
        }
        return subp_res

    def _set_subproblem_dict(self) -> None:
        """Set the subproblem dictionary for ALC

        MIP and NLP subproblems must be defined as Pyomo models.

        Returns:
            dict: subproblem dictionary that contains subproblem id as key
                and subproblem details as value.
        """
        subp_dict = {
            0: {
                "optim type": "MIP",
                "function": self.solve_alc_subprob,
                "args": None,
            },
        }
        for sc_des in range(self.n_sc_design):
            sp_id = sc_des + 1
            subp_dict[sp_id] = {
                "optim type": "NLP",
                "function": self.optimizer._comp_design.sc_sizing.solve_sc_sizing_subprob,
                "args": None,
            }
        self._subp_dict: AllSubpDict = subp_dict

    def _set_prioritized_var_tuple(self) -> None:
        """Set the prioritized variable tuple for ALC

        Returns:
            tuple: prioritized variable tuple that contains
                prioritized variable index and subproblem id.
        """
        if not self.alc.prioritized_var_name:
            self._prioritized_var_tuple = ()
            return
        pri_var_idx = self.sc.var_names.index(self.alc.prioritized_var_name)
        prioritized_var_list: list[dict[str, int]] = []
        for sc_id in range(1, self.n_sc_design + 1):
            prioritized_var_list.append(
                {
                    "prioritized var idx": pri_var_idx + (sc_id - 1) * self.n_sc_vars,
                    "prioritized subp id": sc_id,
                }
            )
        self._prioritized_var_tuple = tuple(prioritized_var_list)

    def _solve_alc_initial_guess_problem(self, pwl_increment: float) -> None:
        """Solver for initial guess problem in ALC and assign initial guess

        The SC design nonconvexity is convexified using PWL approximation.
        It first solves the problem with PWL approximation,
        re-calculate the SC design based on the nonconvex constraint,
        and then solves the problem with the new SC design.
        Can be used for both cvx-ncvx and ncvx-ncvx decompositions.
        """
        t0 = time.time()
        results = self.optimizer.pwl.solve_w_pwl_approx(pwl_increment)
        initial_guess = self.optimizer._comp_design.sc_sizing.reeval_drymass(
            results["design vars"]
        )
        fmleo_initial = self.optimizer.fixed_sc.solve_network_flow_MILP(
            fixed_sc_vars=initial_guess
        )
        t_initial_guess = time.time() - t0

        print("########## Initial MILP run results ##########")
        print("Increment:\t", pwl_increment)
        print("SC Variety:\t", self.n_sc_design)
        print("# of SC/Variety:", self.n_sc_per_design)
        print("total # of SC: \t", self.n_sc_design * self.n_sc_per_design)
        print("FMLEO:\t\t", fmleo_initial)
        print("PWL FMLEO:\t", results["obj"])
        for sc_id in range(self.n_sc_design):
            print("SC Type", sc_id + 1)
            print(
                "pl cap:\t\t",
                results["design vars"][sc_id][self.sc_var_dict["payload"]],
            )
            print(
                "prop cap:\t",
                results["design vars"][sc_id][self.sc_var_dict["propellant"]],
            )
            print(
                "dry mass:\t",
                results["design vars"][sc_id][self.sc_var_dict["dry mass"]],
            )
        print("Run time:\t", t_initial_guess)
        self.initial_guess = initial_guess
