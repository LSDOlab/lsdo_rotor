import numpy as np

import omtools.api as ot

from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.smoothing_explicit_component import SmoothingExplicitComponent
from lsdo_rotor.core.viterna_explicit_component import ViternaExplicitComponent
class BEMTImplicitComponent(ot.ImplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('mode', types = int)
        self.options.declare('num_radial', types = int)
        self.options.declare('rotor', types=RotorParameters)

    def setup(self):
        shape = self.options['shape']
        mode = self.options['mode']
        num_radial = self.options['num_radial']
        rotor = self.options['rotor']

        B = rotor['num_blades']

        if mode == 1:

        
            g = self.group

            phi_ideal_loading = g.create_implicit_output('_phi_ideal_loading', shape=shape)
            Vx = g.declare_input('reference_axial_inflow_velocity', shape=shape)
            Vt = g.declare_input('reference_tangential_inflow_velocity', shape=shape)
            reference_sigma = g.declare_input('reference_blade_solidity', shape=shape)
            Cl0_ref = g.declare_input('Cl0',shape=shape)
            Cla_ref = g.declare_input('Cla',shape=shape)
            Cdmin_ref = g.declare_input('Cdmin',shape=shape)
            K_ref = g.declare_input('K',shape=shape)
            alpha_Cdmin_ref = g.declare_input('alpha_Cdmin',shape=shape)
            alpha_ref = g.declare_input('alpha',shape=shape)
            Cl_ref = Cl0_ref + Cla_ref * alpha_ref
            Cd_ref = Cdmin_ref + K_ref * alpha_ref + alpha_Cdmin_ref * alpha_ref**2 
            
            # AoA = g.declare_input('_alpha', shape=shape)

        # 
        # alpha = AoA
        # g.register_output('alpha', alpha)

        # For LSDO_rotor:       mode 1 --> alpha = g.create_indep_var
        #                       mode 2 --> BEMT: alpha = twist-phi 
        # group = QuadraticAirfoilGroup(shape=shape)
        # g.add_subsystem('airfoil_group', group, promotes=['*'])

            Cl = g.declare_input('_Cl', shape=shape)
            Cl = ot.expand(ot.reshape(Cl,(1,)), shape)
        # print(Cl)
            Cd = g.declare_input('_Cd', shape=shape)
            Cd = ot.expand(ot.reshape(Cd,(1,)), shape)

            Cx = Cl_ref * ot.cos(phi_ideal_loading) - Cd_ref * ot.sin(phi_ideal_loading)
            Ct = Cl_ref * ot.sin(phi_ideal_loading) + Cd_ref * ot.cos(phi_ideal_loading)
            term1 = Vt * (reference_sigma * Cx - 4 * ot.sin(phi_ideal_loading)**2)
            term2 = Vx * (2 * ot.sin(2 * phi_ideal_loading) + Ct * reference_sigma)
            residual = term1 + term2
            # print(sigma.shape)
        # g.register_output('Cx',Cx)

            phi_ideal_loading.define_residual_bracketed(
                residual,
                x1=0.,
                x2=np.pi / 2.,
            )

        elif mode == 2:
            
            
            g = self.group

            phi_BEMT = g.create_implicit_output('_phi_BEMT', shape=shape)

            twist = g.declare_input('_pitch', shape=shape)
            sigma = g.declare_input('_blade_solidity', shape=shape)
            Vx = g.declare_input('_axial_inflow_velocity', shape=shape)
            Vt = g.declare_input('_tangential_inflow_velocity', shape=shape)
            radius = g.declare_input('_radius',shape= shape)
            rotor_radius = g.declare_input('_rotor_radius', shape= shape)
            hub_radius = g.declare_input('_hub_radius', shape = shape)
            
            alpha = twist - phi_BEMT
            g.register_output('_alpha', alpha)
        

            comp  = ViternaExplicitComponent(
                shape = shape,
                rotor = rotor,
            )
            g.add_subsystem('viterna_explicit_component', comp, promotes = ['*'])

            Cl = g.declare_input('_Cl' ,shape=shape)
            Cd = g.declare_input('_Cd', shape=shape)

            f_tip = B / 2 * (rotor_radius - radius) / radius / ot.sin(phi_BEMT)
            f_hub = B / 2 * (radius - hub_radius) / hub_radius / ot.sin(phi_BEMT)

            F_tip = 2 / np.pi * ot.arccos(ot.exp(-f_tip))
            F_hub = 2 / np.pi * ot.arccos(ot.exp(-f_hub))

            F = F_tip * F_hub
    
            Cx = Cl * ot.cos(phi_BEMT) - Cd * ot.sin(phi_BEMT)
            Ct = Cl * ot.sin(phi_BEMT) + Cd * ot.cos(phi_BEMT)
            
            term1 = Vt * (sigma * Cx - 4 * F * ot.sin(phi_BEMT)**2)
            term2 = Vx * (2 * F * ot.sin(2 * phi_BEMT) + Ct * sigma)
            residual = term1 + term2
            
            eps = 1e-6
            phi_BEMT.define_residual_bracketed(
                residual,
                x1=eps,
                x2=np.pi / 2. - eps,
            )

            # g.register_output('_phi_BEMT',phi_BEMT)
                
        




