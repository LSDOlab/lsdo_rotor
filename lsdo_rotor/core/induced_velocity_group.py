import numpy as np
from csdl import Model
import csdl

class InducedVelocityGroup(Model):
    
    def initialize(self):
        self.parameters.declare('rotor')
        self.parameters.declare('mode', types = int)
        self.parameters.declare('shape',types = tuple)

    def define(self):
        rotor = self.parameters['rotor']
        mode = self.parameters['mode']
        shape = self.parameters['shape']

        B = num_blades = rotor['num_blades']

        if mode == 1:
            phi = self.declare_variable('phi_reference_ildm')
            Vx = self.declare_variable('reference_axial_inflow_velocity')
            Vt = self.declare_variable('reference_tangential_inflow_velocity')

            reference_radius = self.declare_variable('reference_radius')
            rotor_radius = self.declare_variable('rotor_radius')
            hub_radius = self.declare_variable('hub_radius')
            sigma = self.declare_variable('reference_blade_solidity')

            Cl_ref_chord = rotor['ideal_Cl_ref_chord']
            Cd_ref_chord = rotor['ideal_Cd_ref_chord']
            alpha_ref_chord = rotor['ideal_alpha_ref_chord']

            f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi)
            f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi)

            F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
            F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

            F = F_tip * F_hub

            Cx = Cl_ref_chord * csdl.cos(phi) - Cd_ref_chord * csdl.sin(phi)
            Ct = Cl_ref_chord * csdl.sin(phi) + Cd_ref_chord * csdl.cos(phi)

            ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct)
            ut = 2 * Vt * sigma * Ct / (2 * F * csdl.sin(2 * phi) + sigma * Ct)

            C = Vt * ut /(2 * ux - Vx) + 2 * Vt * ux /(Vt - ut) - 2 * Vx

            self.register_output('ideal_loading_constant', C)
            self.register_output('axial_induced_velocity_ideal_loading_BEM_step',ux)
            self.register_output('tangential_induced_velocity_ideal_loading_BEM_step',ut)

        elif mode == 2:
            print('TEST')
            phi = self.declare_variable('phi_distribution', shape=shape)
            twist = self.declare_variable('pitch_distribution', shape=shape)

            Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
            Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
            Vx_ref = self.declare_variable('reference_axial_inflow_velocity')
            
            sigma = self.declare_variable('_blade_solidity', shape=shape)
            chord = self.declare_variable('chord_distribution',shape=shape)
            radius = self.declare_variable('_radius', shape = shape)
            dr = self.declare_variable('_slice_thickness', shape = shape)
            rho = self.declare_variable('rho', shape = shape)
            
            F = self.declare_variable('F', shape=shape)

            rotational_speed = self.declare_variable('_rotational_speed', shape=shape)
            n = self.declare_variable('reference_rotational_speed')

            Cl = self.declare_variable('Cl', shape=shape)
            Cd = self.declare_variable('Cd', shape=shape)

            Cx1 = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
            Ct1 = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

            Cx2 = self.declare_variable('Cx', shape=shape)
            Ct2 = self.declare_variable('Ct', shape=shape)

            ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct2)
            ux_2 = Vx + sigma * Cx2 * Vt / (4 * F * csdl.sin(phi) * csdl.cos(phi) + sigma * Ct1)
        
            ut = 2 * Vt * sigma * Ct2 / (2 * F * csdl.sin(2 * phi) + sigma * Ct1)

            dT = 4 * np.pi * radius * rho * ux * (ux - Vx) * F * dr
            dQ = 2 * np.pi * radius**2 * rho * ux * ut * F * dr

            dT2 = num_blades * Cx1 * 0.5 * rho * (ux**2 + (Vt - 0.5 * ut)**2) * chord * dr
            dQ2 = num_blades * Ct1 * 0.5 * rho * (ux**2 + (Vt - 0.5 * ut)**2) * chord * dr * radius

            T2 = csdl.sum(dT2)
            Q2 = csdl.sum(dQ2)

            T = csdl.sum(dT)
            Q = csdl.sum(dQ)

            self.register_output('_ux',ux)
            self.register_output('_ux_2',ux_2)
            self.register_output('_ut', ut)

            self.register_output('_local_thrust', dT)
            self.register_output('total_thrust', T)
            self.register_output('_local_thrust_2', dT2)
            self.register_output('total_thrust_2', T2)


            self.register_output('_local_torque', dQ)
            self.register_output('total_torque', Q)
            self.register_output('_local_torque_2', dQ2)
            self.register_output('total_torque_2', Q2)



