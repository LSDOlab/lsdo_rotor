import numpy as np
from csdl import Model
import csdl

class ILDMBackCompGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor')
        self.parameters.declare('shape', types = tuple)

    def define(self):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']

        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vx_ref = self.declare_variable('reference_axial_inflow_velocity')
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        n = self.declare_variable('reference_rotational_speed')

        eta = self.declare_variable('eta_2', shape=shape)
        dr = self.declare_variable('_slice_thickness', shape = shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape=shape)
        radius = self.declare_variable('_radius', shape = shape)
        hub_radius = self.declare_variable('_hub_radius', shape = shape)
        c_ref = rotor['c_ref']
        # c_ref_exp = csdl.expand(csdl.reshape(c_ref, (1,)), shape)
        rho = self.declare_variable('rho_ildm', shape = shape)


        Cl_ref_chord = rotor['ideal_Cl_ref_chord']
        Cd_ref_chord = rotor['ideal_Cd_ref_chord']
        alpha_ref_chord = rotor['ideal_alpha_ref_chord']
        num_blades = rotor['num_blades']

        a = 2 * Cl_ref_chord
        b = 2 * Cd_ref_chord * Vt - 2 * Cl_ref_chord * Vx
        c = - 2 * Vt * eta * (Cd_ref_chord * Vx + Cl_ref_chord * Vt - Cl_ref_chord * Vt * eta)

        ux = (-b + (b**2 - 4 * a * c)**0.5)/ (2 * a)
        ut = 2 * Vt * (1 + (-1 * eta))
        # 2 * Vt * (-1 * (eta - 1)) try this 
        
        phi = csdl.arctan(ux/(Vt - 0.5*ut)) 

        f_tip = num_blades / 2 * (rotor_radius - radius) / radius / csdl.sin(phi)
        f_hub = num_blades / 2 * (radius - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))
        F = F_tip * F_hub

        dT = 4 * np.pi * rho * ux * (ux-Vx) * radius * F *  dr
        T = csdl.sum(dT)
       
        dQ = 2 * np.pi * rho * ux * ut * radius**2 * F * dr
        Q = csdl.sum(dQ)
        
        dE = 2 * np.pi * radius * rho * (Vt * ux * ut - 2 * Vx * ux**2 + 2 * Vx**2 * ux) * F * dr
        E = csdl.sum(dE)

        # eta_total_rotor = T * Vx_ref / (Q * n * 2 * np.pi)

        c = 2 * dQ / (rho * dr * num_blades * (ux**2 + (Vt - 0.5 * ut)**2) * radius * (Cl_ref_chord * csdl.sin(phi) + Cd_ref_chord * csdl.cos(phi)))
        theta = (phi + alpha_ref_chord)

        weights_1 = csdl.exp(-5.5 * radius)
        weights_2 = 1 + (-1 * weights_1)

        c_mod = 2.5 * c_ref *  weights_1  + c * weights_2


        self.register_output('_local_thrust', dT)
        self.register_output('total_thrust', T)
        
        self.register_output('_local_torque', dQ)
        self.register_output('total_torque', Q)

        self.register_output('_local_inflow_angle',phi)
        self.register_output('_local_twist_angle', theta)
        self.register_output('_local_chord',c)
        self.register_output('_mod_local_chord', c_mod)

        self.register_output('local_ideal_energy_loss',dE)
        self.register_output('total_energy_loss',E)

        self.register_output('_back_comp_axial_induced_velocity', ux)
        self.register_output('_back_comp_tangential_induced_velocity', ut)

        self.register_output('weights_1',weights_1)
        self.register_output('weights_2',weights_2)

        # self.register_output('c_ref_exp', c_ref_exp)

        # self.register_output('total_efficiency', eta_total_rotor)