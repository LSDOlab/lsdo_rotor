import numpy as np

import omtools.api as ot
import openmdao.api as om

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.airfoil.airfoil_surrogate_model import AirfoilSurrogateModel


class BEMTImplicitComponent2(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('rotor', types=RotorParameters)

    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']

        self.input_names = input_names = [
            '_pitch',
            '_blade_solidity',
            '_axial_inflow_velocity',
            '_tangential_inflow_velocity',
            '_radius',
            '_rotor_radius',
            '_hub_radius',
            '_Cl',
            '_Cd'
        ]

        for input_name in input_names:
            self.add_input(input_name, shape=shape)

        self.add_output('_phi_BEMT')
        self.add_output('_alpha')

        for input_name in input_names:
            self.declare_partials('_phi_BEMT', input_name)

    def apply_nonlinear(self, inputs,outputs, residuals):
        shape = self.options['shape']
        rotor = self.options['rotor']


        B = rotor['num_blades']
        twist = inputs['_pitch']
        sigma = inputs['_blade_solidity']
        Vx = inputs['_axial_inflow_velocity']
        Vt = inputs['_tangential_inflow_velocity']
        radius = inputs['_radius']
        rotor_radius = inputs['_rotor_radius']
        hub_radius = inputs['_hub_radius']
        phi_BEMT = outputs['_phi_BEMT']
        alpha = outputs['_alpha']

        # outputs['_alpha'] = twist - phi_BEMT

    def solve_nonlinear(self,inputs,outputs):
        shape = self.options['shape']
        rotor = self.options['rotor']
        
        B = rotor['num_blades']
        twist = inputs['_pitch']
        sigma = inputs['_blade_solidity']
        Vx = inputs['_axial_inflow_velocity']
        Vt = inputs['_tangential_inflow_velocity']
        radius = inputs['_radius']
        rotor_radius = inputs['_rotor_radius']
        hub_radius = inputs['_hub_radius']
        phi_BEMT = outputs['_phi_BEMT']
        alpha = outputs['_alpha']

        outputs['_alpha'] = twist - phi_BEMT
        
        comp  = AirfoilSurrogateModel(
                shape = shape,
                rotor = rotor,
            )
        self.add_subsystem('airfoil_surrogate_model', comp, promotes = ['*'])

        Cl = inputs['_Cl']
        Cd = inputs['_Cd']

        f_tip = B / 2 * (rotor_radius - radius) / radius / np.sin(phi_BEMT)
        f_hub = B / 2 * (radius - hub_radius) / hub_radius / np.sin(phi_BEMT)

        F_tip = 2 / np.pi * np.arccos(np.exp(-f_tip))
        F_hub = 2 / np.pi * np.arccos(np.exp(-f_hub))

        F = F_tip * F_hub

        Cx = Cl * np.cos(phi_BEMT) - Cd * np.sin(phi_BEMT)
        Ct = Cl * np.sin(phi_BEMT) + Cd * np.cos(phi_BEMT)

        term1 = Vt * (sigma * Cx - 4 * F * np.sin(phi_BEMT)**2)
        term2 = Vx * (2 * F * np.sin(2 * phi_BEMT) + Ct * sigma)
        residual = term1 + term2

        eps = 1e-6
        phi_BEMT.define_residual_bracketed(
            residual,
            x1=eps,
            x2=np.pi / 2. - eps,
        )

        outputs['_phi_BEMT'] = phi_BEMT.reshape(shape)

    def linearize(self, inputs, outputs, partials):
        shape = self.options['shape']

        of = ['residual']
        wrt = ['_phi_BEMT'] + self.input_names

        jac = self.prob.compute_totals(of=of,wrt=wrt)

        for name in wrt:
            partials['_phi_BEMT', name] = jac['residual',name]
        
        self.derivs = np.diag(jac['residual','_phi_BEMT']).reshape(shape)


    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['_phi_BEMT'] = 1. / self.derivs * d_residuals['_phi_BEMT']
        else:
            d_residuals['_phi_BEMT'] = 1. / self.derivs * d_outputs['_phi_BEMT']



