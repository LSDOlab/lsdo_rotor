import numpy as np
import openmdao.api as om

from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.bemt.bemt_group import BEMTGroup
from lsdo_rotor.polar.polar_group import PolarGroup
from lsdo_rotor.induced_velocity.induced_velocity_group import InducedVelocityGroup
from lsdo_rotor.constant.constant_group import ConstantGroup
from lsdo_rotor.efficiency.efficiency_group import EfficiencyGroup


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

prob.setup(check=True)
prob.run_model()

# prob.model.list_outputs()
print(prob['phi'],'phi')


# print(prob['Cx'],'Cx_A')
# print(prob['Ct'],'Ct_B')
# print(prob['Vx'],'Vx_C')
# print(prob['Vt'],'Vt_D')
# print(prob['sigma'],'sigma_E')
# print(prob['phi'],'phi_F')
print(prob['ux'],'ux')
print(prob['ut'],'ut')
# print(prob['a'],'a')
# print(prob['b'],'b')
# print(prob['e'],'c')
# print(prob['d'],'d')
# print(prob['e'],'e')
# print(prob['p'],'p')
# print(prob['q'],'q')
# print(prob['Q'],'Q')
# print(prob['S'],'S')
# print(prob['delta_0'],'delta_0')
# print(prob['delta_1'],'delta_1')
print(prob['eta'],'eta')
print(prob['C'],'Constant C')
