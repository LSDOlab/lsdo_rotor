import omtools.api as ot
import numpy as np
from lsdo_rotor.rotor_parameters import RotorParameters

class InducedVelocityGroup(ot.Group):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('mode', types = int)
        self.options.declare('rotor', types=RotorParameters)

    def setup(self):
        shape = self.options['shape']
        mode = self.options['mode']
        rotor = self.options['rotor']

        num_blades = rotor['num_blades']

        if mode == 1:
            phi = self.declare_input('_phi_ideal_loading', shape=shape)
            
            Vx = self.declare_input('reference_axial_inflow_velocity', shape=shape)
            Vt = self.declare_input('reference_tangential_inflow_velocity', shape=shape)
            sigma = self.declare_input('reference_blade_solidity', shape=shape)
            Cl0_ref = self.declare_input('Cl0',shape=shape)
            Cla_ref = self.declare_input('Cla',shape=shape)
            Cdmin_ref = self.declare_input('Cdmin',shape=shape)
            K_ref = self.declare_input('K',shape=shape)
            alpha_Cdmin_ref = self.declare_input('alpha_Cdmin',shape=shape)
            alpha_ref = self.declare_input('alpha',shape=shape)
            Cl_ref = Cl0_ref + Cla_ref * alpha_ref
            Cd_ref = Cdmin_ref + K_ref * alpha_ref + alpha_Cdmin_ref * alpha_ref**2 
    

            Cl = self.declare_input('_Cl', shape=shape)
            Cd = self.declare_input('_Cd', shape=shape)

            Cx = Cl_ref * ot.cos(phi) - Cd_ref * ot.sin(phi)
            Ct = Cl_ref * ot.sin(phi) + Cd_ref * ot.cos(phi)
        
            ux = 4 * Vt * ot.sin(phi)**2 / (2 * (ot.sin(2 * phi) +  sigma * Ct))        
            ut = 2 * Vt * sigma * Ct / (2 * ot.sin(2 * phi) + sigma * Ct)

            C = Vt * ut /(2 * ux - Vx) + 2 * Vt * ux /(Vt - ut) - 2 * Vx  


            # ax = Cx * sigma/ (4 * ot.sin(phi)**2 - Cx * sigma)
            # ay = Ct * sigma/ (2 * ot.sin(2 * phi) + Ct * sigma)

            # x = Vt/Vx

            # C2 = x**2 * ay / (1 + 2 * ax) + (1 + ax)/(1 - 2 * ay)

            # self.register_output('ideal_loading_constant_non_dimensional',C2)
 
            self.register_output('_axial_induced_velocity_BEMT', ux)
            self.register_output('_tangential_induced_velocity_BEMT', ut)
            self.register_output('ideal_loading_constant', C)

        elif mode == 2:
            # print(shape)
            phi_BEMT = self.declare_input('_phi_BEMT', shape=shape)
            Vx = self.declare_input('_axial_inflow_velocity', shape=shape)
            Vx_ref = self.declare_input('reference_axial_inflow_velocity')
            Vt = self.declare_input('_tangential_inflow_velocity', shape=shape)
            sigma = self.declare_input('_blade_solidity', shape=shape)
            chord = self.declare_input('_chord',shape=shape)
            radius = self.declare_input('_radius', shape = shape)
            dr = self.declare_input('_slice_thickness', shape = shape)
            twist = self.declare_input('_pitch', shape=shape)
            F = self.declare_input('BEMT_loss_factor', shape=shape)

            rotational_speed = self.declare_input('_rotational_speed', shape=shape)
            n = self.declare_input('reference_rotational_speed') 

            Cl = self.declare_input('_Cl', shape=shape)
            Cd = self.declare_input('_Cd', shape=shape)

            Cx = Cl * ot.cos(phi_BEMT) - Cd * ot.sin(phi_BEMT)
            Ct = Cl * ot.sin(phi_BEMT) + Cd * ot.cos(phi_BEMT)
            
            ux = (4 * F * Vt * ot.sin(phi_BEMT)**2) / (4 * F * ot.sin(phi_BEMT) * ot.cos(phi_BEMT) +  sigma * Ct)         
            # ux = Vx + sigma * Cx * Vt / (4 * F * ot.sin(phi_BEMT) * ot.cos(phi_BEMT) + sigma * Ct)  
           
            ut = 2 * Vt * sigma * Ct / (2 * F * ot.sin(2 * phi_BEMT) + sigma * Ct)

            # Old equations for induced velocities that break down at Vx = 0
            # ux = Vx * 4 * F * ot.sin(phi_BEMT)**2 / (4 * F *  ot.sin(phi_BEMT)**2 -sigma * Cx)
            # ut = Vx * 2 * F * sigma * Ct / (4 * F * ot.sin(phi_BEMT)**2 -sigma * Cx)

            dT = 4 * np.pi * radius * 1.2 * ux * (ux - Vx) * F * dr            
            dQ = 2 * np.pi * radius**2 * 1.2 * ux * ut * F * dr

            T = ot.sum(dT)
            Q = ot.sum(dQ)

            dT2 = num_blades * Cx * 0.5 * 1.2 * (ux**2 + (Vt - 0.5 * ut)**2) * chord * dr
            dQ2 = num_blades * Ct * 0.5 * 1.2 * (ux**2 + (Vt - 0.5 * ut)**2) * chord * dr * radius

            T2 = ot.sum(dT2)
            Q2 = ot.sum(dQ2)

            eta = Vx * dT / rotational_speed / 2 / np.pi / dQ

            eta_1 = Vx / ux
            eta_2 = (Vt - 0.5 * ut) / Vt
            eta_3 = 2 * ux * (ux - Vx) / (ut * (Vt - 0.5 * ut))

            # eta_3 = eta /eta_1/eta_2
            FOM  = eta_2 * eta_3

            eta_total = Vx_ref * T / (n * 2 * np.pi * Q)
            alpha = twist - phi_BEMT
            


            self.register_output('test_output',ut * (Vt - 0.5 * ut))
            self.register_output('BEMT_axial_induced_velocity',ux)
            self.register_output('BEMT_tangential_induced_velocity',ut)
            self.register_output('BEMT_local_thrust', dT)
            self.register_output('BEMT_local_thrust_2', dT2)
           
            
            self.register_output('BEMT_local_torque', dQ)
            self.register_output('BEMT_local_torque_2', dQ2)


            self.register_output('BEMT_total_torque',Q)
            self.register_output('BEMT_total_thrust',T)
            self.register_output('BEMT_total_torque_2',Q2)
            self.register_output('BEMT_total_thrust_2',T2)
            self.register_output('BEMT_local_efficiency',eta)
            self.register_output('BEMT_total_efficiency',eta_total)
            self.register_output('BEMT_local_AoA',alpha)
            self.register_output('BEMT_eta_1', eta_1)
            self.register_output('BEMT_eta_2', eta_2)
            self.register_output('BEMT_eta_3', eta_3)
            self.register_output('FOM', FOM)
            self.register_output('Cx',Cx)
            self.register_output('Ct',Ct)
            

            # self.add_constraint('BEMT_total_thrust', equals = 1)
            # self.add_objective('BEMT_total_torque')
