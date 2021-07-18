import numpy as np
from csdl import Model
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")

from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.bemt.bemt_group import BEMTGroup
from lsdo_rotor.polar.polar_group import PolarGroup
from lsdo_rotor.induced_velocity.induced_velocity_group import InducedVelocityGroup
from lsdo_rotor.constant.constant_group import ConstantGroup
from lsdo_rotor.efficiency.efficiency_group import EfficiencyGroup

shape = (1, )

model = Model()

group = InputsGroup(shape=shape, )
model.add(group, 'inputs_group', promotes=['*'])

group = BEMTGroup(shape=shape, )
model.add(group, 'bemt_group', promotes=['*'])

group = PolarGroup(shape=shape, )
model.add(group, 'polar_group', promotes=['*'])

group = InducedVelocityGroup(shape=shape, )
model.add(group, 'induced_velocity_group', promotes=['*'])

group = ConstantGroup(shape=shape, )
model.add(group, 'constant_group', promotes=['*'])

group = EfficiencyGroup(shape=shape, )
model.add(group, 'efficiency_group', promotes=['*'])

sim = Simulator(model)
sim.run()

# model.list_outputs()
print(sim['phi'], 'phi')

# print(sim['Cx'],'Cx_A')
# print(sim['Ct'],'Ct_B')
# print(sim['Vx'],'Vx_C')
# print(sim['Vt'],'Vt_D')
# print(sim['sigma'],'sigma_E')
# print(sim['phi'],'phi_F')
print(sim['ux'], 'ux')
print(sim['ut'], 'ut')
# print(sim['a'],'a')
# print(sim['b'],'b')
# print(sim['e'],'c')
# print(sim['d'],'d')
# print(sim['e'],'e')
# print(sim['p'],'p')
# print(sim['q'],'q')
# print(sim['Q'],'Q')
# print(sim['S'],'S')
# print(sim['delta_0'],'delta_0')
# print(sim['delta_1'],'delta_1')
print(sim['eta'], 'eta')
print(sim['C'], 'Constant C')
