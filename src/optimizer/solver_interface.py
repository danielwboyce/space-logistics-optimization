from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer_class import Optimizer
import os
import sys
from pyomo.opt import SolverFactory, OptSolver, TerminationCondition, SolverStatus
from pyomo.opt.results.results_ import SolverResults
# from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds
# from pyomo.contrib.iis.mis import compute_infeasibility_explanation
# from pyomo.common.log import logging
from pyomo.environ import ConcreteModel
from pyomo.kernel import block
# from pyomo.repn.plugins.lp_writer import LPWriter

try:
    from initializer import InitMixin
except (ModuleNotFoundError, ImportError):
    sys.path.append("..")
    from initializer import InitMixin


class SolverInterface(InitMixin):
    """Class with methods to use pyomo solvers"""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        super().__init__(self.optimizer._input_data)

    def solve_model(
        self,
        model: block,
    ) -> block:
        """solve the model and return the solved model and write results

        Args:
            model: constucted pyomo kernel model
        Returns:
            block: solved pyomo kernel model
        """
        if isinstance(model, ConcreteModel):
            NotImplementedError(
                "Pyomo environ ConcreteModel is not supported. Use pyomo kernel block instead."
            )
        # lp_writer = LPWriter()
        # lp_writer.write(model, open('mymodel.new.lp', 'w'), symbolic_solver_labels=True)
        opt = self._set_solver_options()
        solved_model: SolverResults = opt.solve(
            model,
            tee=self.runtime.solver_verbose,
            keepfiles=self.runtime.keep_files,
            logfile="solver_logfile.log",
        )
        print("Termination Condition: ", solved_model.solver.termination_condition)
        # if solved_model.solver.termination_condition in {
        #     TerminationCondition.infeasible,
        #     TerminationCondition.infeasibleOrUnbounded
        # }:
        #     logger = logging.getLogger('pyomo.util.infeasible')
        #     logger.setLevel(logging.INFO)
        #     handler = logging.FileHandler('infeasible_constraints.log')
        #     # handler = logging.FileHandler('infeasible_bounds.log')
        #     handler.setFormatter(logging.Formatter('%(message)s'))
        #     logger.addHandler(handler)
        #     log_infeasible_constraints(model, logger=logger)
        #     # log_infeasible_bounds(model, logger=logger)

        # #     # logger = logging.getLogger('pyomo.contrib.iis')
        # #     # logger.setLevel(logging.INFO)
        # #     # handler = logging.FileHandler('iis.log')
        # #     # handler.setFormatter(logging.Formatter('%(message)s'))
        # #     # logger.addHandler(handler)
        # #     # compute_infeasibility_explanation(solved_model, solver=solved_model.solver, logger=logger)
        if solved_model.solver.termination_condition not in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.maxIterations,
            TerminationCondition.maxEvaluations,
            TerminationCondition.feasible,
        }:
            print("Optimal solution not found.")
            return model
        if self.runtime.store_results_to_csv:
            self.optimizer.output.write_results(model)
        return model

    def _set_solver_options(self) -> OptSolver:
        """Set solver options as specified by user and return pyomo OptSolver.

        Returns:
            OptSolver: pyomo genetic optimization solver class instance
        """
        # TODO: Implement settings for other solvers
        # Add options to specify optimality gaps
        if self.runtime.mip_solver == "baron":
            opt = SolverFactory("baron")
            opt.options["MaxTime"] = self.runtime.max_time
            opt.options["MaxIter"] = -1
            if self.runtime.mip_subsolver == "cplex":
                opt.options["LPSol"] = 3  # use CPLEX
            if os.path.exists(self.runtime.cplex_path):
                opt.options["CplexLibName"] = self.runtime.cplex_path
            if self.runtime.max_time_wo_imprv:
                opt.options["DeltaTerm"] = 1
                opt.options["DeltaT"] = (
                    self.runtime.max_time_wo_imprv * self.runtime.max_threads / 2
                )  # approximate coversion from wall time to CPU time
        elif self.runtime.mip_solver == "cplex":
            opt = SolverFactory(
                self.runtime.mip_solver,
                executable=self.runtime.cplex_path,
            )
            opt.options["timelimit"] = self.runtime.max_time
        else:
            opt = SolverFactory(self.runtime.mip_solver)
            opt.options["timelimit"] = self.runtime.max_time
        opt.options["threads"] = self.runtime.max_threads
        return opt
