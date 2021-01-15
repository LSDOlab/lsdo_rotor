import numpy as np

import omtools.api as ot

from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.inputs.inputs_group import InputsGroup


class BEMTGroup(ot.ImplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        g = self.group

        # group = ot.Group()
        # group.create_indep_var('twist', val=50. * np.pi / 180.)
        # group.create_indep_var('Vx', val=50)
        # group.create_indep_var('Vt', val=100.)
        # group.create_indep_var('sigma', val=0.15)
        # g.add_subsystem('inputs_group', group, promotes=['*'])

        phi = g.create_implicit_output('phi', shape=shape)
        Vx = g.declare_input('Vx')
        Vt = g.declare_input('Vt')
        sigma = g.declare_input('sigma')
        twist = g.declare_input('twist')
        print(type(Vx),'Vx')
        alpha = twist - phi
        print(alpha,'alpha')
        g.register_output('alpha', alpha)

        group = QuadraticAirfoilGroup(shape=shape)
        g.add_subsystem('airfoil_group', group, promotes=['*'])

        Cl = g.declare_input('Cl')
        Cd = g.declare_input('Cd')

        Cx = Cl * ot.cos(phi) - Cd * ot.sin(phi)
        print(type(Cx),'Cx_type')
        Ct = Cl * ot.sin(phi) + Cd * ot.cos(phi)
        term1 = Vt * (2 * Ct * ot.sin(2 * phi) / Cx  -  Cx * sigma)
        term2 = Vx * (2 * ot.sin(2 * phi) + Ct * sigma)
        residual = term1 - term2

        phi.define_residual_bracketed(
            residual,
            x1=0.,
            x2=np.pi / 2.,
        )