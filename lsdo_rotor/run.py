import numpy as np
import openmdao.api as om
from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.bemt.bemt_group import BEMTGroup


shape = (1,)

prob = om.Problem()

# comp = om.IndepVarComp()
# comp.add_output('twist', val=50. * np.pi / 180.)
# comp.add_output('Vx', val=50)
# comp.add_output('Vt', val=100.)
# comp.add_output('sigma', val=0.15)
# prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

group = BEMTGroup(
    shape=shape,
)
prob.model.add_subsystem('bemt_group', group, promotes=['*'])

prob.setup(check=True)
prob.run_model()

prob.model.list_outputs()
# print(prob['phi'])