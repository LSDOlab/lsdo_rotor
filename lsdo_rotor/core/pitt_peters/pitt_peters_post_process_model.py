import numpy as np
from csdl import Model
import csdl

from lsdo_rotor.airfoil.pitt_peters_airfoil_model_2 import PittPetersAirfoilModel2
from lsdo_rotor.core.pitt_peters.pitt_peters_rotor_parameters import PittPetersRotorParameters

class PittPetersPostProcessModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=PittPetersRotorParameters)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        ne = shape[0]
        nr = shape[1]
        nt = shape[2]

        B = self.parameters['num_blades']
        mu_z = self.declare_variable('mu_z',shape=(ne,))
        mu_z_exp = csdl.expand(mu_z,shape,'i->ijk')
        # mu_z_exp = np.einsum(
        #     'i,ijk->ijk',
        #     mu_z,
        #     np.ones((ne, nr,nt)),  
        # )

        Re = self.declare_variable('_re_pitt_peters', shape=shape)
        chord = self.declare_variable('_chord',shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)
        norm_radius = self.declare_variable('_normalized_radius', shape=shape)
        radius = self.declare_variable('_radius', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        psi = self.declare_variable('_theta', shape=shape)
        angular_speed = self.declare_variable('_angular_speed', shape=shape)
        n = angular_speed / 2 / np.pi
        rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
        dr = self.declare_variable('_dr', shape=shape)
        sigma = self.declare_variable('_blade_solidity', shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)

        rho = self.declare_variable('density', shape=(ne,))
        rho_exp = self.declare_variable('_density_expanded', shape=shape)
        
        lamb = self.declare_variable('_lambda', shape = (shape[0],3))

        lamb_0 = self.create_output('lamb_0_pp', shape = (shape[0],1))
        lamb_s = self.create_output('lamb_s_pp', shape = (shape[0],1))
        lamb_c = self.create_output('lamb_c_pp', shape = (shape[0],1))
        for i in range(shape[0]):
            lamb_0[i,0] = lamb[i,0]
            lamb_c[i,0] = lamb[i,1]
            lamb_s[i,0] = lamb[i,2]

        lamb_0_exp = csdl.expand(csdl.reshape(lamb_0,new_shape=(shape[0], )), shape, 'i->ijk')      
        lamb_s_exp = csdl.expand(csdl.reshape(lamb_s,new_shape=(shape[0], )), shape, 'i->ijk')
        lamb_c_exp = csdl.expand(csdl.reshape(lamb_c,new_shape=(shape[0], )), shape, 'i->ijk')

        lamb_exp =  1 * lamb_0_exp + lamb_s_exp * norm_radius * csdl.sin(psi) + lamb_c_exp * norm_radius * csdl.cos(psi)
        # self.print_var(lamb_exp)
        ux = (lamb_exp + mu_z_exp) * angular_speed * rotor_radius
        # self.print_var(ux)
        phi = csdl.arctan(ux/Vt)
        alpha = twist - phi 
        self.register_output('AoA_pitt_peters_2', alpha)

        airfoil_model_output = csdl.custom(Re,alpha, op= PittPetersAirfoilModel2(
                rotor=rotor,
                shape=shape,
            ))
        self.register_output('Cl_pitt_peters_2',airfoil_model_output[0])
        self.register_output('Cd_pitt_peters_2',airfoil_model_output[1])

        Cl = self.declare_variable('Cl_pitt_peters_2', shape=shape)
        Cd = self.declare_variable('Cd_pitt_peters_2', shape=shape)
        
        Cx = (Cl * csdl.cos(phi) - Cd * csdl.sin(phi))
        Ct = (Cl * csdl.sin(phi) + Cd * csdl.cos(phi))

        dT = 0.5 * B * rho_exp * (ux**2 + (Vt)**2) * chord * Cx * dr
        T = csdl.sum(dT, axes = (1,2)) / shape[2]
        dQ = 0.5 * B * rho_exp * (ux**2 + (Vt)**2) * chord * Ct * radius * dr
        Q = csdl.sum(dQ, axes = (1,2)) / shape[2]


        dL_mom = radius * csdl.sin(psi) * dT
        # L_mom  = csdl.sum(dL_mom, axes = (1,2))  / shape[2]
        dM_mom = radius * csdl.cos(psi) * dT
        # M_mom  = csdl.sum(dM_mom, axes = (1,2)) / shape[2]

        dC_T = dT / rho_exp / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**4
        dC_L = dL_mom / rho_exp / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5
        dC_M = dM_mom / rho_exp / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5

        C_T = csdl.reshape(csdl.sum(dC_T, axes = (1,2))  / shape[2], new_shape = (shape[0],1))
        C_L = csdl.reshape(csdl.sum(dC_L, axes = (1,2))  / shape[2], new_shape = (shape[0],1))
        C_M = csdl.reshape(csdl.sum(dC_M, axes = (1,2))  / shape[2], new_shape = (shape[0],1))



        T = csdl.sum(dT, axes = (1,2)) / shape[2]
        Q = csdl.sum(dQ, axes = (1,2)) / shape[2]

        # print(n[:,0,0].shape,'SHAPE')
        # print(T.shape,'SHAPE')

        C_T_2 = T / rho / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**4
        C_Q = Q / rho / (csdl.sum(n,axes=(1,2))/shape[1]/shape[2])**2 / (2 * csdl.sum(rotor_radius,axes=(1,2))/shape[1]/shape[2])**5
        C_P = 2 * np.pi * C_Q
        J = csdl.sum((Vx / n /  (2 * rotor_radius)),axes=(1,2))/shape[1]/shape[2]
        eta = C_T_2 * J / C_P

        Q_total = csdl.sum(Q)
        T_total = csdl.sum(T)

        self.register_output('total_torque', Q)
        self.register_output('total_torque_all_rotors', Q_total)
        self.register_output('total_thrust_all_rotors', T_total)
        # self.register_output('total_thrust', T)
        self.register_output('T', T)
        # self.print_var(T)
        self.register_output('_dT',dT)
        self.register_output('_dQ',dQ)
        self.register_output('_ux',ux)

        self.register_output('C_T',C_T)
        self.register_output('dC_T',dC_T)
        self.register_output('C_T_2',C_T_2)
        self.register_output('C_Q',C_Q)
        self.register_output('C_P',C_P)
        self.register_output('eta',eta)
        self.register_output('C_L',C_L)
        self.register_output('C_M',C_M)
        self.register_output('J',J)



        # self.add_objective('total_torque')
        # self.add_constraint('T', equals = 1500)
