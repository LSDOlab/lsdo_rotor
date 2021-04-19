import numpy as np

import omtools.api as ot

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.inputs_group import InputsGroup
from lsdo_rotor.core.preprocess_group import PreprocessGroup
from lsdo_rotor.core.efficiency_coeffs_group import EfficiencyCoeffsGroup
from lsdo_rotor.core.efficiency_implicit_component import EfficiencyImplicitComponent
from lsdo_rotor.core.ideal_blade_group import IdealBladeGroup

class PostProcessGroup(ot.Group):

    def initialize(self):
        self.options.declare('rotor', types=RotorParameters)
        self.options.declare('shape', types=tuple)
        
        
        # self.options.declare('num_radial', types=int)

    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']
        # num_radial = self.options['num_radial']

        num_blades = rotor['num_blades']
        n = self.declare_input('reference_rotational_speed')
        Vx = self.declare_input('_axial_inflow_velocity', shape=shape)
        Vx_ref = self.declare_input('reference_axial_inflow_velocity')
        Vt = self.declare_input('_tangential_inflow_velocity', shape=shape)
        ux = self.declare_input('_axial_induced_velocity', shape=shape)
        ut = self.declare_input('_tangential_induced_velocity', shape = shape)
        ax = self.declare_input('_axial_induction_factor', shape=shape)
        ay = self.declare_input('_tangential_induction_factor', shape = shape)
        dr = self.declare_input('_slice_thickness', shape = shape)
        rotor_radius = self.declare_input('_rotor_radius', shape=shape)
        radius = self.declare_input('_radius', shape = shape)
        alpha = self.declare_input('_alpha', shape=shape)
        # num_blades = self.declare_input('num_blades',shape=shape)
        Cl = self.declare_input('_Cl', shape=shape)
        Cd = self.declare_input('_Cd', shape=shape)

        dT = 4 * np.pi * 1.2 * ux * (ux-Vx) * radius * dr
        dQ = 2 * np.pi * 1.2 * ux * ut * radius**2 * dr

        

        T = ot.sum(dT)
        # print(T.shape,'T shape')
        Q = ot.sum(dQ)

        eta_total = T * Vx_ref / (Q * n * 2 * np.pi)


        phi = ot.arctan(ux/(Vt - 0.5*ut)) 
        
        c = 2 * dQ / (1.2 * dr * num_blades * (ux**2 + (Vt - 0.5 * ut)**2) * radius * (Cl * ot.sin(phi) + Cd * ot.cos(phi))) 
        

        theta = (phi + alpha) * 180 / np.pi
        phi = phi * 180 / np.pi

        

        

        self.register_output('_local_thrust', dT)
        self.register_output('_local_torque', dQ)

        self.register_output('_total_thrust', T)
        # self.add_design_var('_total_thrust')

        self.register_output('_total_torque', Q)
        self.add_objective('_total_torque')

        self.register_output('_total_efficiency', eta_total)

        self.register_output('_local_inflow_angle',phi)

        self.register_output('_local_twist_angle_deg', theta)

        self.register_output('_local_chord',c)
        self.add_design_var('_local_chord',lower=0.07, upper = 0.2)


        



        



















      