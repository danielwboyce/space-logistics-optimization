from __future__ import annotations
from typing import TYPE_CHECKING
from itertools import product
import numpy as np
from pyomo.kernel import (
    variable,
    variable_dict,
    parameter,
    parameter_dict,
    constraint,
    constraint_dict,
    Reals,
    NonNegativeReals,
)

if TYPE_CHECKING:
    from ..opt_model_builder_class import OptModelBuilder


class ADMMSubprobComponents:
    """
    Class to define additional variables, parameters, and constraints for
    modelling ALC/ADMM subproblems.
    """

    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_admm_subprob_components(self, m):
        self._set_admm_subprob_params(m)
        self._set_admm_subprob_variables(m)
        return m

    def _set_admm_subprob_params(self, m):
        """
        Define parameters for ALC/ADMM subproblems

        - lag_mult: Lagrange multipliers estimates
        - penalty_weight: augmented Lagrangian penalty weights
        - global_shared_vars: shared variables from the master problem
        """
        lag_mult = np.array_split(
            self.builder.lagrange_mult_est, self.builder.n_sc_design
        )
        penalty_weight = np.array_split(
            self.builder.penelty_weight, self.builder.n_sc_design
        )
        global_shared_vars = np.array_split(
            self.builder.global_shared_vars, self.builder.n_sc_design
        )
        m.lag_mult = parameter_dict()
        m.penalty_weight = parameter_dict()
        m.global_shared_vars = parameter_dict()
        m.local_shared_vars_def = constraint_dict()
        for sc_des, sc_var in product(m.sc_des_idx, m.sc_var_idx):
            m.lag_mult[sc_des, sc_var] = parameter(lag_mult[sc_des][sc_var])
            m.penalty_weight[sc_des, sc_var] = parameter(
                penalty_weight[sc_des][sc_var])
            m.global_shared_vars[sc_des, sc_var] = parameter(
                global_shared_vars[sc_des][sc_var]
            )

        return m

    def _set_admm_subprob_variables(self, m):
        """
        Define additional variables for ALC/ADMM subproblems.
        Local shared variables are defined as separate variables and
        connected to corresponding via equality constraints.
        Similarly, relative and absolute consistency violation variables
        are defined via constraints. Two separate inequalities are used to
        model absolute values linearly.

        - local_shared_vars: local (this subproblem's) copy of shared variables
        - rel_consis_vio: relative consistency violation of local and
            global shared variables given by the master problem.
        - abs_consis_vio: absolute consistency violation of local and global
            shared variables
        """
        m.local_shared_vars = variable_dict()
        m.local_shared_vars_def = constraint_dict()
        m.rel_consis_vio = variable_dict()
        m.rel_consis_vio_def = constraint_dict()
        m.abs_consis_vio = variable_dict()
        m.abs_consis_vio_def_ineq1 = constraint_dict()
        m.abs_consis_vio_def_ineq2 = constraint_dict()
        for sc_des, sc_var in product(m.sc_des_idx, m.sc_var_idx):
            m.abs_consis_vio[sc_des, sc_var] = variable(
                domain=NonNegativeReals)
            m.rel_consis_vio[sc_des, sc_var] = variable(domain=Reals)
            m.local_shared_vars[sc_des, sc_var] = variable(
                domain=NonNegativeReals)
            if self.builder.sc_var_dict.inverse[sc_var] == "payload":
                m.local_shared_vars_def[sc_des, sc_var] = constraint(
                    m.local_shared_vars[sc_des, sc_var] == m.pl_cap[sc_des]
                )
            elif self.builder.sc_var_dict.inverse[sc_var] == "propellant":
                m.local_shared_vars_def[sc_des, sc_var] = constraint(
                    m.local_shared_vars[sc_des, sc_var] == m.prop_cap[sc_des]
                )
            elif self.builder.sc_var_dict.inverse[sc_var] == "dry mass":
                m.local_shared_vars_def[sc_des, sc_var] = constraint(
                    m.local_shared_vars[sc_des, sc_var] == m.dry_mass[sc_des]
                )
            else:
                raise ValueError(
                    "Unknown SC design variable type: ",
                    self.builder.sc_var_dict.inverse[sc_var],
                )
            m.rel_consis_vio_def[sc_des, sc_var] = constraint(
                m.rel_consis_vio[sc_des, sc_var]
                == m.global_shared_vars[sc_des, sc_var]
                - m.local_shared_vars[sc_des, sc_var]
            )
            m.abs_consis_vio_def_ineq1[sc_des, sc_var] = constraint(
                -m.rel_consis_vio[sc_des,
                                  sc_var] <= m.abs_consis_vio[sc_des, sc_var]
            )
            m.abs_consis_vio_def_ineq2[sc_des, sc_var] = constraint(
                m.rel_consis_vio[sc_des,
                                 sc_var] <= m.abs_consis_vio[sc_des, sc_var]
            )

        self.builder.idx_name_dict["local_shared_vars"] = ["sc_des", "sc_cp"]
        self.builder.idx_name_dict["abs_consis_vio"] = ["sc_des", "sc_cp"]
        self.builder.idx_name_dict["rel_consis_vio"] = ["sc_des", "sc_cp"]
        self.builder._test_index_variable_mapping(m)
        return m
