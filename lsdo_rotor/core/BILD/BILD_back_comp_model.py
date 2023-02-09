import numpy as np
from csdl import Model
import csdl

class BILDBackCompModel(Model):

    def initialize(self):
        self.parameters.declare('num_blades', types=int)
        self.parameters.declare('shape', types=tuple)

    def define(self):
        num_blades = self.parameters['num_blades']
        shape = self.parameters['shape']
        print('BILD SHAPE', shape)

        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        angular_speed = self.declare_variable('_angular_speed', shape=shape)
        n = angular_speed / 2 / np.pi

        eta = self.declare_variable('eta_2', shape=shape)
        dr = self.declare_variable('_dr', shape=shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape=shape)
        radius = self.declare_variable('_radius', shape=shape)
        hub_radius = self.declare_variable('_hub_radius', shape=shape)
        
        chord  = self.declare_variable('reference_chord', shape = (shape[0],))
        # c_ref = rotor['c_ref']
        c_ref = csdl.expand(chord, shape, 'i->ijk')
        rho_exp = csdl.expand(self.declare_variable('density', shape=(shape[0],)),shape,'i->ijk')
        # self.print_var(rho_exp)
        rho = self.declare_variable('density', shape=(shape[0],))

        Cl = self.declare_variable('Cl_max_BILD', shape = (shape[0],))
        Cd = self.declare_variable('Cd_min_BILD', shape = (shape[0],))
        Cl_ref_chord = csdl.expand(Cl, shape, 'i->ijk')
        Cd_ref_chord = csdl.expand(Cd, shape, 'i->ijk')

        alpha_max_LD = self.declare_variable('alpha_max_LD', shape = (shape[0],))
        alpha_ref_chord = csdl.expand(alpha_max_LD, shape, 'i->ijk')
        
        a = 2 * Cl_ref_chord
        b = 2 * Cd_ref_chord * Vt - 2 * Cl_ref_chord * Vx
        c = - 2 * Vt * eta * (Cd_ref_chord * Vx + Cl_ref_chord * Vt - Cl_ref_chord * Vt * eta)
        
        self.register_output('c',c)
        self.register_output('b',b)

        ux_num = (-b + (b**2 -4 * a * c)**0.5)
        ux_den = (2 * a)
        
        ux =  (-b + (b**2 -4 * a * c)**0.5)/ (2 * a)
        ut = 2 * Vt * (1 + (-1 * eta))
        # ut =  2 * Vt * (-1 * (eta - 1)) #try this 
        
        phi = csdl.arctan(ux/(Vt - 0.5*ut)) 

        f_tip = num_blades / 2 * (rotor_radius - radius) / radius / csdl.sin(phi)
        f_hub = num_blades / 2 * (radius - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))
        F = F_tip * F_hub
        self.register_output('F_dist',F)

        dT = 4 * np.pi * rho_exp * ux * (ux-Vx) * radius * F *  dr
        T = csdl.sum(dT, axes = (1, 2))
       
        dQ = 2 * np.pi * rho_exp * ux * ut * radius**2 * F * dr
        Q = csdl.sum(dQ, axes = (1, 2))
        
        dE = 2 * np.pi * radius * rho_exp * (Vt * ux * ut - 2 * Vx * ux**2 + 2 * Vx**2 * ux) * F * dr
        E = csdl.sum(dE, axes=(1, 2))
        c = 2 * dQ / (rho_exp * dr * num_blades * (ux**2 + (Vt - 0.5 * ut)**2) * radius * (Cl_ref_chord * csdl.sin(phi) + Cd_ref_chord * csdl.cos(phi)))
        

        Cx = Cl_ref_chord * csdl.cos(phi) - Cd_ref_chord * csdl.sin(phi)
        Ct = Cl_ref_chord * csdl.sin(phi) + Cd_ref_chord * csdl.cos(phi)
        dT2 = num_blades * Cx * 0.5 * rho_exp * (ux**2 + (Vt - 0.5 * ut)**2) * c * dr
        dQ2 = num_blades * Ct * 0.5 * rho_exp * (ux**2 + (Vt - 0.5 * ut)**2) * c * dr * radius

        theta = (phi + alpha_ref_chord)

        weights_1 = csdl.exp(-5.5 * radius)
        weights_2 = 1 + (-1 * weights_1)

        c_mod = 2.5 * c_ref *  weights_1  + c * weights_2

        # Performance coefficients:
        C_T = T / rho / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**4
        C_Q = Q / rho / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**5
        C_P = 2 * np.pi * C_Q
        J = csdl.sum((Vx / n /  (2 * rotor_radius)),axes=(1,2))/shape[1]/shape[2]
        eta = C_T * J / C_P

        self.register_output('_local_thrust', dT)
        self.register_output('total_thrust', T)
        
        self.register_output('_local_torque', dQ)
        self.register_output('total_torque', Q)

        self.register_output('_local_thrust_2', dT2)
        self.register_output('total_thrust_2', csdl.sum(dT2,axes = (1,)))
        
        self.register_output('_local_torque_2', dQ2)
        self.register_output('total_torque_2', csdl.sum(dQ2,axes = (1,)))

        self.register_output('_local_inflow_angle',phi)
        self.register_output('_local_twist_angle', theta)
        self.register_output('_local_chord',c)
        self.register_output('_mod_local_chord', c_mod)

        self.register_output('local_ideal_energy_loss',dE)
        self.register_output('total_energy_loss',E)

        self.register_output('_back_comp_axial_induced_velocity', ux)
        self.register_output('_back_comp_tangential_induced_velocity', ut)

        self.register_output('C_T', C_T)
        self.register_output('C_Q', C_Q)
        self.register_output('C_P', C_P)
        self.register_output('eta', eta)
        self.register_output('J', J)

        # self.register_output('weights_1',weights_1)
        # self.register_output('weights_2',weights_2)

        # self.register_output('c_ref_exp', c_ref_exp)

        # self.register_output('total_efficiency', eta_total_rotor)