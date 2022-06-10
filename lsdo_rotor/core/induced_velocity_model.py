import numpy as np
from csdl import Model
import csdl
from lsdo_rotor.rotor_parameters import RotorParameters

class InducedVelocityModel(Model):
    
    def initialize(self):
        self.parameters.declare('rotor', types=RotorParameters)
        self.parameters.declare('mode', types = int)
        self.parameters.declare('shape',types=tuple)

    def define(self):
        rotor = self.parameters['rotor']
        mode = self.parameters['mode']
        shape = self.parameters['shape']

        B = num_blades = rotor['num_blades']

        if mode == 1:
            phi = self.declare_variable('phi_reference_ildm', shape = (shape[0],))
            Vx = self.declare_variable('ildm_axial_inflow_velocity', shape = (shape[0],))
            Vt = self.declare_variable('ildm_tangential_inflow_velocity', shape = (shape[0],))

            reference_radius = self.declare_variable('reference_radius', shape = (shape[0],))
            rotor_radius = self.declare_variable('rotor_radius', shape = (shape[0],))
            hub_radius = self.declare_variable('hub_radius', shape = (shape[0],))
            sigma = self.declare_variable('reference_blade_solidity', shape = (shape[0],))
            
            # Cl_ref_chord = rotor['ideal_Cl_ref_chord']
            # Cd_ref_chord = rotor['ideal_Cd_ref_chord']
            # alpha_ref_chord = rotor['ideal_alpha_ref_chord']

            Cl = self.declare_variable('Cl_max_ildm', shape = (shape[0],))
            Cd = self.declare_variable('Cd_min_ildm', shape = (shape[0],))

            f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi)
            f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi)

            F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
            F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

            F = F_tip * F_hub
            

            Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
            Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

            ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct)
            # ux = Vx + sigma * Cx * Vt / (4 * F * csdl.sin(phi) * csdl.cos(phi) + sigma * Ct)

            # ux_num = (4 * F * Vt * csdl.sin(phi)**2)
            # ux_den = (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct)
            
            # ux = ux_num/ux_den

            # self.register_output('ux_num', ux_num)
            # self.register_output('ux_den', ux_den)
            # self.register_output('ux_num_by_den', ux_num/ux_den)

            ut = 2 * Vt * sigma * Ct / (2 * F * csdl.sin(2 * phi) + sigma * Ct)

            C = Vt * ut /(2 * ux - Vx) + 2 * Vt * ux /(Vt - ut) - 2 * Vx

            self.register_output('ideal_loading_constant', C)
            # self.register_output('axial_induced_velocity_ideal_loading_BEM_step',ux)
            # self.register_output('tangential_induced_velocity_ideal_loading_BEM_step',ut)

        elif mode == 2:
            # print('TEST')
            phi = self.declare_variable('phi_distribution', shape=shape)
            twist = self.declare_variable('_pitch', shape=shape)

            Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
            Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
            Vx_ref = self.declare_variable('ildm_axial_inflow_velocity')

            angular_speed = self.declare_variable('_angular_speed', shape=shape)
            n = angular_speed / 2 / np.pi
            
            sigma = self.declare_variable('_blade_solidity', shape=shape)
            chord = self.declare_variable('_chord',shape=shape)
            radius = self.declare_variable('_radius', shape=shape)
            rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
            dr = self.declare_variable('_dr', shape=shape)
            rho = self.declare_variable('rho', shape=shape)
            
            
            F = self.declare_variable('prandtl_loss_factor', shape=shape)

            # rotational_speed = self.declare_variable('_rotational_speed', shape=shape)
            # n = self.declare_variable('ildm_rotational_speed')

            Cl = self.declare_variable('Cl_2', shape=shape)
            Cd = self.declare_variable('Cd_2', shape=shape)

            Cx1 = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
            Ct1 = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

            # Cx2 = self.declare_variable('Cx', shape=shape)
            # Ct2 = self.declare_variable('Ct', shape=shape)

            ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct1)
            ux_2 = Vx + sigma * Cx1 * Vt / (4 * F * csdl.sin(phi) * csdl.cos(phi) + sigma * Ct1)
        
            ut = 2 * Vt * sigma * Ct1 / (2 * F * csdl.sin(2 * phi) + sigma * Ct1)

            dT = 4 * np.pi * radius * rho * ux * (ux - Vx) * F * dr
            dC_T = dT / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**4
            dQ = 2 * np.pi * radius**2 * rho * ux * ut * F * dr

            dT2 = num_blades * Cx1 * 0.5 * rho * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord * dr
            dQ2 = num_blades * Ct1 * 0.5 * rho * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord * dr * radius

            T2 = csdl.sum(dT2, axes = (1,2)) / shape[2]
            Q2 = csdl.sum(dQ2, axes = (1,2)) / shape[2]

            T = csdl.sum(dT, axes = (1,2)) / shape[2]
            Q = csdl.sum(dQ, axes = (1,2)) / shape[2]

            

            dE = 2 * np.pi * radius * 1.2 * (Vt * ux * ut - 2 * Vx * ux**2 + 2 * Vx**2 * ux) * F * dr
            E = csdl.sum(dE)
            print(rotor['density'],'RHO')
            C_T = T / rotor['density'] / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**4
            C_Q = Q / rotor['density'] / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**5
            C_P = 2 * np.pi * C_Q
            J = csdl.sum((Vx / n /  (2 * rotor_radius)),axes=(1,2))/shape[1]/shape[2]
            eta = C_T * J / C_P

            self.register_output('_ux',ux)
            self.register_output('_ux_2',ux_2)
            self.register_output('_ut', ut)

            self.register_output('_local_thrust', dT)
            self.register_output('total_thrust', T)
            self.register_output('dC_T',dC_T)
            
            self.register_output('_local_thrust_2', dT2)
            self.register_output('total_thrust_2', T2)

            self.register_output('_local_torque', dQ)
            self.register_output('total_torque', Q)
            
            self.register_output('_local_torque_2', dQ2)
            self.register_output('total_torque_2', Q2)

            self.register_output('_local_energy_loss', dE)
            self.register_output('total_energy_loss', E)
            
            self.register_output('C_T',C_T)
            self.register_output('C_Q',C_Q)
            self.register_output('C_P',C_P)
            self.register_output('eta',eta)
            self.register_output('J',J)

            # self.add_objective('total_torque')
            self.add_objective('total_energy_loss')

            self.add_constraint('total_thrust', equals = 613.959)


