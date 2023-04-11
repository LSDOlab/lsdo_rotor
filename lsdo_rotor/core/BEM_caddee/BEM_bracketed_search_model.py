import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl

from lsdo_rotor.airfoil.BEM_airfoil_surrogate_model_group import BEMAirfoilSurrogateModelGroup
from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters

class BEMBracketedSearchGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor',types=BEMRotorParameters)
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        B = num_blades = self.parameters['num_blades']

        model = Model()
        
        sigma = model.declare_variable('_blade_solidity', shape=shape)
        Vx = model.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = model.declare_variable('_tangential_inflow_velocity', shape=shape)
        radius = model.declare_variable('_radius',shape= shape)
        rotor_radius = model.declare_variable('_rotor_radius', shape= shape)
        hub_radius = model.declare_variable('_hub_radius', shape=shape)
        chord = model.declare_variable('_chord',shape=shape)
        twist = model.declare_variable('_pitch', shape=shape)
        
        # phi is state (inflow angle) we're solving for in the bracketed search
        phi = model.declare_variable('phi_distribution', shape=shape)

        Re = model.declare_variable('Re', shape=shape)
        # self.print_var(Re)

        alpha = twist - phi
        model.register_output('alpha_distribution', alpha)
        
        # Adding custom component to embed airfoil model in the bracketed search
        if not rotor['custom_polar']:
            airfoil_model_output = csdl.custom(Re, alpha, chord, op= BEMAirfoilSurrogateModelGroup(
                rotor=rotor,
                shape=shape,
            ))
        else:
            print('custom polar')
            airfoil_model_output = csdl.custom(Re, alpha, chord, op= BEMAirfoilSurrogateModelGroup(
                rotor=rotor,
                shape=shape,
            ))
        model.register_output('Cl',airfoil_model_output[0])
        model.register_output('Cd',airfoil_model_output[1])
        
        Cl = airfoil_model_output[0]
        Cd = airfoil_model_output[1]

        model.declare_variable('Cl', shape=shape)
        model.declare_variable('Cd', shape=shape)
        
        # Prandtl tip losses 
        f_tip = B / 2 * (rotor_radius - radius) / radius / csdl.sin(phi)
        f_hub = B / 2 * (radius - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

        F = F_tip * F_hub
        model.register_output('F',F)

        # Setting up residual function
        Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

        model.register_output('Cx',Cx)
        model.register_output('Ct',Ct)

        term1 = Vt * (sigma * Cx - 4 * F * csdl.sin(phi)**2)
        term2 = Vx * (2 * F * csdl.sin(2 * phi) + Ct * sigma)
        
        BEM_residual = term1 + term2
        
        model.register_output('BEM_residual_function', BEM_residual)
        
        # Solving residual function for state phi 
        eps = 1e-7
        solve_BEM_residual = self.create_implicit_operation(model)
        solve_BEM_residual.declare_state('phi_distribution', residual='BEM_residual_function', bracket=(eps, np.pi/2 - eps))

        sigma = self.declare_variable('_blade_solidity', shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        radius = self.declare_variable('_radius',shape= shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
        hub_radius = self.declare_variable('_hub_radius', shape=shape)
        chord = self.declare_variable('_chord',shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)
        Re = self.declare_variable('Re', shape=shape)
    

        # phi, Cl, Cd,F, Cx, Ct = solve_BEM_residual(sigma,Vx,Vt,radius,rotor_radius,hub_radius,chord,twist,Re, expose=['Cl', 'Cd','F','Cx','Ct'])
        phi = solve_BEM_residual(sigma,Vx,Vt,radius,rotor_radius,hub_radius,chord,twist,Re)
