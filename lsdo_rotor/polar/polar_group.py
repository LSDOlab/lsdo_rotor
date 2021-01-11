import omtools.api as ot

from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.bemt.bemt_group import BEMTGroup
from lsdo_rotor.inputs.inputs_group import InputsGroup

class PolarGroup(ot.Group):
    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        

        phi = self.declare_input('phi')
        twist = self.declare_input('twist')

        alpha = twist - phi
        self.register_output('alpha', alpha)

        group = QuadraticAirfoilGroup(shape=shape)
        self.add_subsystem('airfoil_group', group, promotes=['*'])

        Cl = self.declare_input('Cl')
        Cd = self.declare_input('Cd')

        Cx = Cl * ot.cos(phi) - Cd * ot.sin(phi)
        Ct = Cl * ot.sin(phi) + Cd * ot.cos(phi)

        self.register_output('Cx',Cx)
        self.register_output('Ct',Ct)


