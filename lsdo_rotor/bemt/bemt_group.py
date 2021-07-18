import numpy as np

from csdl import ImplicitModel
import csdl

from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.inputs.inputs_group import InputsGroup


class BEMTGroup(ImplicitModel):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']

        # group = Model()
        # group.create_input('twist', val=50. * np.pi / 180.)
        # group.create_input('Vx', val=50)
        # group.create_input('Vt', val=100.)
        # group.create_input('sigma', val=0.15)
        # self.add(group, name='inputs_group', promotes=['*'])

        phi = self.create_implicit_output('phi', shape=shape)
        Vx = self.declare_variable('Vx')
        Vt = self.declare_variable('Vt')
        sigma = self.declare_variable('sigma')
        twist = self.declare_variable('twist')

        alpha = twist - phi
        self.register_output('alpha', alpha)

        group = QuadraticAirfoilGroup(shape=shape)
        self.add(group, name='airfoil_group', promotes=['*'])

        Cl = self.declare_variable('Cl')
        Cd = self.declare_variable('Cd')

        Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)
        term1 = Vt * (2 * Ct * csdl.sin(2 * phi) / Cx - Cx * sigma)
        term2 = Vx * (2 * csdl.sin(2 * phi) + Ct * sigma)
        residual = term1 - term2

        phi.define_residual_bracketed(
            residual,
            x1=0.,
            x2=np.pi / 2.,
        )
