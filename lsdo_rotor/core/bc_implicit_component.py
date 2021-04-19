import numpy as np

import omtools.api as ot

from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
# from lsdo_rotor.core.inputs_group import InputsGroup
# from lsdo_rotor.core.ideal_blade_group import IdealBladeGroup

class BCImplicitComponent(ot.ImplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        g = self.group

        alpha = g.create_implicit_output('alpha', shape=shape)
        
        # print(shape[1])
        

        phi = g.declare_input('_local_inflow_angle_phi', shape =  shape)
        phi1 = phi[0,2,0]
        # phi2 = self.declare_input('_inflow_angle_phi2', shape =  shape)
        dT = g.declare_input('_local_thrust', shape = shape)
        dT1 = dT[0,2,0]
        # print(dT1)
    
        dQ = g.declare_input('_local_torque', shape = shape)
        dQ1 = dQ[0,2,0]

        radius = g.declare_input('_radius', shape = shape)
        radius1 = radius[0,2,0]
       
        # Vt = g.declare_input('_tangential_inflow_velocity', shape=shape)
        # sigma = g.declare_input('_blade_solidity', shape=shape)
        # twist = g.declare_input('_pitch', shape=shape)
        # Vx = g.declare_input('_axial_inflow_velocity', shape=shape)

        # alpha = twist - phi
        # g.register_output('alpha', alpha)

        group = QuadraticAirfoilGroup(shape=shape)
        g.add_subsystem('airfoil_group', group, promotes=['*'])

        Cl = g.declare_input('_Cl', shape=shape)
        Cla = Cl[0,2,0]

        Cd = g.declare_input('_Cd', shape=shape)
        K = Cd[0,2,0]

        Cx = Cla * ot.cos(phi1) - K * ot.sin(phi1)
        Ct = Cla * ot.sin(phi1) + K * ot.cos(phi1)

        term1 = radius1 * Ct * dT1
        term2 = Cx * dQ1
        residual = term1 - term2

        alpha.define_residual_bracketed(
            residual,
            x1 = 0.,
            x2 = 0.1 * np.pi/2.,
        )




        # Cl0 = self.declare_input('_Cl0', shape=shape)
        # Cla = self.declare_input('_Cla', shape=shape)
        # Cdmin = self.declare_input('_Cdmin', shape=shape)
        # K = self.declare_input('_K', shape=shape)
        # alpha_Cdmin = self.declare_input('_alpha_Cdmin', shape=shape)

        # ut = self.declare_input('_tangential_induced_velocity', shape=shape)
        # ux = self.declare_input('_axial_induced_velocity', shape=shape)
        # Vx = self.declare_input('_axial_inflow_velocity', shape=shape)
        
        # ax = self.declare_input('_axial_induction_factor', shape=shape)
        # ay = self.declare_input('_tangential_induction_factor', shape = shape)

        # # Cx = (Cl0 + Cla * alpha) * ot.cos(phi) - (Cdmin + K * alpha + alpha_Cdmin * alpha ** 2) * ot.sin(phi)
        # # Ct = Cl0 + Cla * alpha * ot.sin(phi) + (Cdmin + K * alpha + alpha_Cdmin * alpha ** 2) * ot.cos(phi)
        
        
        # # term1 = radius * Ct * dT
        # # term2 = Cx * dQ
        # # residual = term1 - term2
        
        # # LD = (2 * ax * ot.sin(phi2)**2 * (1 - ay) * ot.sin(phi2) + ay * ot.sin(2 * phi2) * (1 + ax) * ot.cos(phi2))/(2 * ax * ot.sin(phi2)**2 * (1 - ay) * ot.cos(phi2) - ay * ot.sin(2 * phi2) * (1 + ax) * ot.sin(phi2))
        # LD = (2 * ax * ot.sin(phi2)**2 * (1 - ay) * ot.sin(phi2) - ay * ot.sin(2 * phi2) * (1 + ax) * ot.cos(phi2))

        # # LD = (2 * ax * ot.sin(phi2)**2 * (1 - ay) * ot.cos(phi2) + ay * ot.sin(2 * phi2) * (1 + ax) * ot.sin(phi2))/(2 * ax * ot.sin(phi2)**2 * (1 - ay) * ot.sin(phi2) - ay * ot.sin(2 * phi2) * (1 + ax) * ot.cos(phi2))
        # a = alpha_Cdmin * ot.cos(phi) * radius * dT + alpha_Cdmin *  ot.sin(phi) * dQ
        # b = (K * ot.cos(phi) + Cla * ot.sin(phi)) * radius * dT  - (Cla * ot.cos(phi) - K * ot.sin(phi))*dQ
        # c = (Cl0 * ot.sin(phi) + Cdmin * ot.cos(phi)) * radius * dT - (Cl0 * ot.cos(phi) - Cdmin * ot.sin(phi)) * dQ

        # a2 = 1 * alpha_Cdmin
        # b2 = 1 * K
        # c2 = 2 * ax * ot.sin(phi2)**2 * (1 - ay) * ot.sin(phi2) - ay * ot.sin(2 * phi2) * (1 + ax) * ot.cos(phi2) + Cdmin

        # A = 2 * ax * ot.sin(phi2)**2 * (1 - ay)
        # B = ay * ot.sin(2 * phi2) * (1 + ax)

        # a3 = A * ot.cos(phi2) * alpha_Cdmin + B * ot.sin(phi2) * alpha_Cdmin
        # b3 = A * ot.sin(phi2) * Cla + A * ot.cos(phi2) * K - B * ot.cos(phi2) * Cla + B * ot.sin(phi2) * K
        # c3 = A * ot.sin(phi2) * Cl0 + A * ot.cos(phi2) * Cdmin - B * ot.cos(phi2) * Cl0 + B * ot.sin(phi2) * Cdmin
        # # a = alpha_Cdmin * ot.sin(phi) * ut + 2 * (ux - Vx) * alpha_Cdmin * ot.cos(phi)
        # # b = - ut * (Cla * ot.cos(phi) - K * ot.sin(phi)) + 2 * (ux - Vx) * (Cla * ot.sin(phi) + K * ot.cos(phi))
        # # c = - ut * (Cl0 * ot.cos(phi) - Cdmin * ot.sin(phi)) + 2 * (ux - Vx) * (Cl0 * ot.sin(phi) + Cdmin * ot.cos(phi))

        
        # # alpha = (-((K * ot.cos(phi) + Cla * ot.sin(phi)) * radius * dT  - (Cla * ot.cos(phi) - K * ot.sin(phi))*dQ) + 
        # #         ((((K * ot.cos(phi) + Cla * ot.sin(phi)) * radius * dT  - (Cla * ot.cos(phi) - K * ot.sin(phi))*dQ))**2 - 
        # #         4 * (alpha_Cdmin * ot.cos(phi) * radius * dT + alpha_Cdmin *  ot.sin(phi) * dQ) * ((Cl0 * ot.sin(phi) + Cdmin * ot.cos(phi)) * radius * dT - (Cl0 * ot.cos(phi) - Cdmin * ot.sin(phi)) * dQ))**0.5 
        # #         ) / (2 * (alpha_Cdmin * ot.cos(phi) * radius * dT + alpha_Cdmin *  ot.sin(phi) * dQ))

        # det = b**2 - 4 * a * c

        # disc = -1*(b3**2 - 4 * a3 * c3)

        # alpha = (-b2 + disc**0.5)/(2 * a2)

        # alpha2 = ((2 * ax * ot.sin(phi2)**2 * (1 - ay) * ot.cos(phi2) + ay * ot.sin(2 * phi2) * (1 + ax) * ot.sin(phi2)) - Cl0)/Cla

        # alpha3 = disc**0.5

        # self.register_output('_det', det)
        # self.register_output('_a', a3)
        # self.register_output('_b', b3)
        # self.register_output('_c', c3)
        # self.register_output('_LD',LD)
        # self.register_output('_discriminant',disc)
        # self.register_output('_AoA',alpha)
        # self.register_output('_AoA2',alpha3)
        # alpha.define_residual_bracketed(
        #     residual,
        #     x1=0.,
        #     x2 =np.pi / 4.,
        # )


        # size = 10

        # for i in range(size):
        #     alpha.define_residual_bracketed(
        #         residual,
        #         x1=0.,
        #         x2 =np.pi / 2.,
        #     )
