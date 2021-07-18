from csdl import Model
import csdl

from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.bemt.bemt_group import BEMTGroup
from lsdo_rotor.inputs.inputs_group import InputsGroup


class PolarGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']

        phi = self.declare_variable('phi')
        twist = self.declare_variable('twist')

        alpha = twist - phi
        self.register_output('alpha', alpha)

        group = QuadraticAirfoilGroup(shape=shape)
        self.add(group, name='airfoil_group', promotes=['*'])

        Cl = self.declare_variable('Cl')
        Cd = self.declare_variable('Cd')

        Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

        self.register_output('Cx', Cx)
        self.register_output('Ct', Ct)
