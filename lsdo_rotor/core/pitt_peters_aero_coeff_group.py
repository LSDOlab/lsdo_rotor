from csdl import Model
import csdl
import numpy as np

from lsdo_rotor.rotor_parameters import RotorParameters

class PittPetersAeroCoeffGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=RotorParameters)


    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']

        B = rotor['num_blades']


        rho = self.declare_variable('_rho_pitt_peters', shape=shape)
        chord = self.declare_variable('_chord',shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)
        radius = self.declare_variable('_radius', shape=shape)

        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        ux = self.declare_variable('ux_pitt_peters',shape =shape)

        psi = self.declare_variable('_theta', shape=shape)

        angular_speed = self.declare_variable('_angular_speed', shape=shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
        dr = self.declare_variable('_dr', shape=shape)
        sigma = self.declare_variable('_blade_solidity', shape=shape)

        Cl = self.declare_variable('Cl_pitt_peters', shape=shape)
        Cd = self.declare_variable('Cd_pitt_peters', shape=shape)

        phi = self.declare_variable('phi_pitt_peters',shape=shape)

        Cx = (Cl * csdl.cos(phi) - Cd * csdl.sin(phi))
        Ct = (Cl * csdl.sin(phi) + Cd * csdl.cos(phi))

        dT = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Cx * dr
        T = csdl.sum(dT, axes = (1,2)) / shape[2]
        dQ = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Ct * radius * dr
        Q = csdl.sum(dQ, axes = (1,2)) / shape[2]


        dL_mom = radius * csdl.sin(psi) * dT
        L_mom  = csdl.sum(dL_mom, axes = (1,2))  / shape[2]
        dM_mom = radius * csdl.cos(psi) * dT
        M_mom  = csdl.sum(dM_mom, axes = (1,2)) / shape[2]

        # Compute coefficients 
        dC_T = dT / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**4
        dC_L = dL_mom / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5
        dC_M = dM_mom / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5

        C_T = csdl.sum(dC_T, axes = (1,2))  / shape[2]
        C_L = csdl.sum(dC_L, axes = (1,2))  / shape[2]
        C_M = csdl.sum(dC_M, axes = (1,2))  / shape[2]

        print(C_T.shape,'TESTING')

        self.register_output('C_T_pitt_peters',C_T)
        self.register_output('C_L_pitt_peters',C_L)
        self.register_output('C_M_pitt_peters',C_M)


# class PittPetersAeroCoeffGroup(csdl.CustomExplicitOperation):
#     def initialize(self):
#         self.parameters.declare('shape', types=tuple)
#         self.parameters.declare('rotor', types=RotorParameters)

#     def define(self):
#         shape = self.parameters['shape']
#         rotor = self.parameters['rotor']

#         # Adding inputs 
#         self.add_input('_rho_pitt_peters', shape=shape)
#         self.add_input('_chord', shape=shape)
#         self.add_input('_pitch', shape=shape)
#         self.add_input('_normalized_radius', shape=shape)
#         self.add_input('_tangential_inflow_velocity', shape=shape)
#         self.add_input('_theta', shape=shape)
#         self.add_input('_angular_speed', shape=shape)
#         self.add_input('_rotor_radius', shape=shape)
#         self.add_input('_dr', shape=shape)
#         self.add_input('_blade_solidity', shape=shape)

#         self.add_input('ux_pitt_peters',shape =shape)

#         self.add_input('Cl_pitt_peters', shape=shape)
#         self.add_input('Cd_pitt_peters', shape=shape)

#         self.add_input('phi_pitt_peters',shape=shape)

#     def compute(self, inputs,outputs):
#         shape       = self.parameters['shape']
#         rotor       = self.parameters['rotor']
#         B           = rotor['num_blades']


#         rho         = inputs['_rho_pitt_peters']
#         chord       = inputs['_chord']
#         twist       = inputs['_pitch']
#         radius      = inputs['_normalized_radius']
#         Vt          = inputs['_tangential_inflow_velocity']
#         psi         = inputs['_theta']
#         Omega       = inputs['_angular_speed']
#         R           = inputs['_rotor_radius']
#         dr          = inputs['_dr']
#         sigma       = inputs['_blade_solidity']

#         Cl          = inputs['Cl_pitt_peters']
#         Cd          = inputs['Cd_pitt_peters']
#         phi         = inputs['phi_pitt_peters']

#         ux          = inputs['ux_pitt_peters']

#         Cx = (Cl * np.cos(phi) - Cd * np.sin(phi))
#         Ct = (Cl * np.sin(phi) + Cd * np.cos(phi))


#         dT = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Cx * dr
#         dQ = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Ct * radius * dr
#         dL_mom = radius * np.sin(psi) * dT
#         dM_mom = radius * csdl.cos(psi) * dT