from __future__ import annotations
from typing import TYPE_CHECKING
from itertools import product
from pyomo.kernel import (
    constraint,
    constraint_dict,
    block,
)

if TYPE_CHECKING:
    from ..opt_model_builder_class import OptModelBuilder


class ISRUBigM:
    """Class to add big-M constraints for the ISRU variables.
    Comoposed to the OptModelBuilder class.

    ISRU plant mass (isru_mass) is defined as the mass of the ISRU plants
    present at the lunar surface ("LS") node. There is also an ISRU usage
    variable (isru_use_ind), a binary variable, which indicates whether the
    ISRU system is getting used. The ISRU production rate (isru_O2rate)
    variable gives the answer as to how much product the ISRU system is able
    to produce.

    This binary-continuous bilinear term can be expresses as a disjunction of
    two polytopes: {0} and [0, M], where M is a big-M constant (variable upper
    bound). Refer to any integer programming textbook for info on disjunction.
    """

    def __init__(self, builder: OptModelBuilder) -> None:
        self.builder = builder

    def set_isru_big_M_constraints(self, m: block) -> block:
        """Set big-M constraints for the ISRU variables."""
        if not self.builder.use_isru:
            return m

        little_M_val: float = 400
        big_M_val: float = 100000
        m.isru_bigM_const_1 = constraint_dict()
        m.isru_bigM_const_2 = constraint_dict()
        m.isru_bigM_const_3 = constraint_dict()
        for isru_des, t, scnr in product(
            m.isru_des_idx,
            m.time_idx,
            m.scnr_idx,
        ):
            m.isru_bigM_const_1[isru_des, t, scnr] = constraint(
                m.isru_mass[isru_des, t, scnr]
                >= m.isru_use_ind[isru_des, t, scnr] * little_M_val
            )

            m.isru_bigM_const_2[isru_des, t, scnr] = constraint(
                m.isru_mass[isru_des, t, scnr]
                <= little_M_val + m.isru_use_ind[isru_des, t, scnr] * big_M_val
            )

            m.isru_bigM_const_3[isru_des, t, scnr] = constraint(
                m.isru_O2rate[isru_des, t, scnr]
                <= m.isru_use_ind[isru_des, t, scnr] * big_M_val
            )

        return m
