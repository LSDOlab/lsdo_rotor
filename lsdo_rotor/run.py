import numpy as np
import openmdao.api as om

from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.bemt.bemt_group import BEMTGroup
from lsdo_rotor.polar.polar_group import PolarGroup
from lsdo_rotor.induced_velocity.induced_velocity_group import InducedVelocityGroup
from lsdo_rotor.constant.constant_group import ConstantGroup
from lsdo_rotor.efficiency.efficiency_group import EfficiencyGroup
from lsdo_rotor.quartic_solver.quartic_solver_group import BracketedImplicitComp

shape = (1,)

prob = om.Problem()

group = InputsGroup(
    shape=shape,
)
prob.model.add_subsystem('inputs_group',group, promotes=['*'])

group = BEMTGroup(
    shape=shape,
)
prob.model.add_subsystem('bemt_group', group, promotes=['*'])

group = PolarGroup(
    shape=shape,
)
prob.model.add_subsystem('polar_group', group, promotes=['*'])

group = InducedVelocityGroup(
    shape=shape,
)
prob.model.add_subsystem('induced_velocity_group', group, promotes=['*'])

group = ConstantGroup(
    shape=shape,
)
prob.model.add_subsystem('constant_group', group, promotes=['*'])

group = EfficiencyGroup(
    shape=shape,
)
prob.model.add_subsystem('efficiency_group', group, promotes=['*'])

group = BracketedImplicitComp(
    shape=shape,
)
prob.model.add_subsystem('bracketed_implicit_group', group, promotes=['*'])




prob.setup(check=True)
prob.run_model()


# prob.model.list_outputs()


print(prob['eta'],'eta')

