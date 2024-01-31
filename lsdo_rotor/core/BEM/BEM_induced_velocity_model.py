import numpy as np
from csdl import Model
import csdl


class BEMInducedVelocityModel(Model):
    
    def initialize(self):
        self.parameters.declare('shape',types=tuple)
        self.parameters.declare('num_blades', types=int)
        

    def define(self):
        shape = self.parameters['shape']
        
        B = num_blades = self.parameters['num_blades'] 
        
        phi = self.declare_variable('phi_distribution', shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)

        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)

        angular_speed = self.declare_variable('_angular_speed', shape=shape)
        n = angular_speed / 2 / np.pi
        # print(( (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2).shape,'BEM test SHAPE')
        
        sigma = self.declare_variable('_blade_solidity', shape=shape)
        chord = self.declare_variable('_chord',shape=shape)
        radius = self.declare_variable('_radius', shape=shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape=shape)
        dr = self.declare_variable('_dr', shape=shape)
        
        rho_exp = csdl.expand(self.declare_variable('density', shape=(shape[0],)),shape,'i->ijk')
        # self.print_var(rho_exp)
        rho = self.declare_variable('density', shape=(shape[0],))
        # self.print_var(rho)
        F = self.declare_variable('prandtl_loss_factor', shape=shape)


        # Cl = self.declare_variable('Cl_2', shape=shape)
        # Cd = self.declare_variable('Cd_2', shape=shape)

        Cl = self.declare_variable('Cl', shape=shape)
        Cd = self.declare_variable('Cd', shape=shape)

        Cx1 = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct1 = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

        Cx2 = Cl * csdl.cos(phi)
        Ct2 = Cl * csdl.sin(phi)


        ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct1)
        ux_2 = Vx + sigma * Cx1 * Vt / (4 * F * csdl.sin(phi) * csdl.cos(phi) + sigma * Ct1)
    
        ut = 2 * Vt * sigma * Ct1 / (2 * F * csdl.sin(2 * phi) + sigma * Ct1)

        dT = 4 * np.pi * radius * rho_exp * ux * (ux - Vx) * F * dr
        dC_T = dT / rho_exp / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**4
        dQ = 2 * np.pi * radius**2 * rho_exp * ux * ut * F * dr

        dT2 = num_blades * Cx1 * 0.5 * rho_exp * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord * dr
        dQ2 = num_blades * Ct1 * 0.5 * rho_exp * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord * dr * radius
        
        dDrag = num_blades * Cd * 0.5 * rho_exp * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord * dr

        dT_induced = num_blades * Cx2 * 0.5 * rho_exp * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord * dr
        dQ_induced = num_blades * Ct2 * 0.5 * rho_exp * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord * dr

        T2 = csdl.sum(dT2, axes = (1,2)) / shape[2]
        Q2 = csdl.sum(dQ2, axes = (1,2)) / shape[2]

        T = csdl.sum(dT, axes = (1,2)) / shape[2]
        # print(rho.shape,'rho shape')
        
        Q = csdl.sum(dQ, axes = (1,2)) / shape[2]

        dT_star = 2 * np.pi * rho_exp * (Vt - 0.5 * ut) * ut * radius * dr 
        T_star = csdl.sum(dT_star, axes = (1,2)) / shape[2]
        

        dE = 2 * np.pi * radius * 1.2 * (Vt * ux * ut - 2 * Vx * ux**2 + 2 * Vx**2 * ux) * F * dr
        E = csdl.sum(dE, axes=(1, 2))

        C_T = T / rho / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**4
        T_C = C_T / (np.pi**3 / 4)
        # C_T = T/ ( rho * np.pi R^2 * Omega^2 * R^2) 
        # self.print_var(C_T)
        C_Q = Q / rho / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**5
        C_P = 2 * np.pi * C_Q
        J = csdl.sum((Vx / n /  (2 * rotor_radius)),axes=(1,2))/shape[1]/shape[2]
        eta = C_T * J / C_P
        FOM = C_T * (C_T/2)**0.5 / C_P

        self.register_output('_ux_compute',ux)
        self.register_output('_ux_2_compute',ux_2)
        self.register_output('_ut_compute', ut)

        self.register_output('_local_thrust_compute', dT)
        self.register_output('_dT_compute', dT*1)
        self.register_output('_local_thrust_induced_compute', dT_induced)
        # self.register_output('total_thrust', T)
        self.register_output('T_compute', T)
        self.register_output('Q_compute', Q*1)
        # self.print_var(T)
        self.register_output('dC_T_compute',dC_T)
        
        self.register_output('_local_thrust_2_compute', dT2)
        self.register_output('total_thrust_2_compute', T2)
        self.register_output('_dD_compute', dDrag)

        self.register_output('_local_thrust_star_compute', dT_star)
        self.register_output('total_thrust_star_compute', T_star)

        # self.register_output('')

        self.register_output('_local_torque_compute', dQ)
        self.register_output('_dQ_compute', dQ*1)
        self.register_output('_local_torque_induced_compute', dQ_induced)
        self.register_output('total_torque_compute', Q)
        
        self.register_output('_local_torque_2_compute', dQ2)
        self.register_output('total_torque_2_compute', Q2)

        self.register_output('_local_energy_loss_compute', dE)
        self.register_output('total_energy_loss_compute', E)
        
        self.register_output('C_T_compute',C_T)
        # self.register_output('C_T',T_C)
        self.register_output('C_Q_compute',C_Q)
        self.register_output('C_P_compute',C_P)
        self.register_output('eta_compute', eta)
        self.register_output('J_compute',J)
        self.register_output('FOM_compute', FOM)

        # self.add_objective('total_torque')
        # self.add_objective('total_energy_loss')

        # # self.add_constraint('T', equals = 230)


