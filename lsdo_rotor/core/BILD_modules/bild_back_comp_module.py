import numpy as np

import csdl

class BILDBackCompModuleCSDL(csdl.Model):

    def initialize(self):
        self.parameters.declare('num_blades', types=int)
        self.parameters.declare('shape', types=tuple)

    def define(self):
        num_blades = self.parameters['num_blades']
        shape = self.parameters['shape']

        Vx = self.register_module_input('_axial_inflow_velocity', shape=shape)
        Vt = self.register_module_input('_tangential_inflow_velocity', shape=shape)
        angular_speed = self.register_module_input('_angular_speed', shape=shape)
        n = angular_speed / 2 / np.pi

        eta = self.register_module_input('eta_2', shape=shape)
        dr = self.register_module_input('_dr', shape=shape)
        rotor_radius = self.register_module_input('_rotor_radius', shape=shape)
        radius = self.register_module_input('_radius', shape=shape)
        hub_radius = self.register_module_input('_hub_radius', shape=shape)
        
        chord  = self.register_module_input('reference_chord', shape = (shape[0],))
        c_ref = csdl.expand(chord, shape, 'i->ijk')
        rho_exp = csdl.expand(self.register_module_input('density', shape=(shape[0],)),shape,'i->ijk')
        rho = self.register_module_input('density', shape=(shape[0],))

        Cl = self.register_module_input('Cl_max_BILD', shape = (shape[0],))
        Cd = self.register_module_input('Cd_min_BILD', shape = (shape[0],))
        Cl_ref_chord = csdl.expand(Cl, shape, 'i->ijk')
        Cd_ref_chord = csdl.expand(Cd, shape, 'i->ijk')
        self.register_module_output('Cl_2', Cl_ref_chord)
        self.register_module_output('Cd_2', Cd_ref_chord)

        alpha_max_LD = self.register_module_input('alpha_max_LD', shape = (shape[0],))
        alpha_ref_chord = csdl.expand(alpha_max_LD, shape, 'i->ijk')
        self.register_module_output('AoA', alpha_ref_chord)
        
        a = 2 * Cl_ref_chord
        b = 2 * Cd_ref_chord * Vt - 2 * Cl_ref_chord * Vx
        c = - 2 * Vt * eta * (Cd_ref_chord * Vx + Cl_ref_chord * Vt - Cl_ref_chord * Vt * eta)
        
        self.register_module_output('c',c)
        self.register_module_output('b',b)
        
        ux =  (-b + (b**2 -4 * a * c)**0.5)/ (2 * a)
        ut = 2 * Vt * (1 + (-1 * eta))
        
        phi = csdl.arctan(ux/(Vt - 0.5*ut)) 

        f_tip = num_blades / 2 * (rotor_radius - radius) / radius / csdl.sin(phi)
        f_hub = num_blades / 2 * (radius - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))
        F = F_tip * F_hub
        self.register_module_output('F_dist',F)

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
        FOM = C_T * (C_T/2)**0.5 / C_P


        self.register_module_output('_local_thrust', dT)
        self.register_module_output('_dT', dT*1)
        self.register_module_output('total_thrust', T)
        self.register_module_output('T', T*1)
        
        self.register_module_output('_local_torque', dQ)
        self.register_module_output('_dQ', dQ*1)
        self.register_module_output('total_torque', Q)
        self.register_module_output('Q', Q*1)

        self.register_module_output('_local_thrust_2', dT2)
        self.register_module_output('total_thrust_2', csdl.sum(dT2,axes = (1,)))
        
        self.register_module_output('_local_torque_2', dQ2)
        self.register_module_output('total_torque_2', csdl.sum(dQ2,axes = (1,)))

        self.register_module_output('_local_inflow_angle',phi)
        self.register_module_output('_local_twist_angle', theta)
        self.register_module_output('_local_chord',c)
        self.register_module_output('_mod_local_chord', c_mod)
        self.register_module_output('_chord', c_mod*1)
        self.register_module_output('_pitch', theta*1)

        self.register_module_output('local_ideal_energy_loss',dE)
        self.register_module_output('total_energy_loss',E)

        self.register_module_output('_back_comp_axial_induced_velocity', ux)
        self.register_module_output('_back_comp_tangential_induced_velocity', ut)

        self.register_module_output('C_T', C_T)
        self.register_module_output('C_Q', C_Q)
        self.register_module_output('C_P', C_P)
        self.register_module_output('eta', eta)
        self.register_module_output('J', J)
        self.register_module_output('FOM', FOM)

        # self.register_module_output('weights_1',weights_1)
        # self.register_module_output('weights_2',weights_2)

        # self.register_module_output('c_ref_exp', c_ref_exp)

        # self.register_module_output('total_efficiency', eta_total_rotor)