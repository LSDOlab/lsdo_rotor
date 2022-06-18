import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl

class ILDMPhiBracketedSearchModel(Model):

    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        shape = self.parameters['shape']
        B = num_blades = self.parameters['num_blades']
        num_nodes = shape[0]

        model = Model()

        Vx = model.declare_variable('u' , shape=(num_nodes,))
        Vt = model.declare_variable('ildm_tangential_inflow_velocity', shape=(num_nodes,))
        reference_sigma = model.declare_variable('reference_blade_solidity', shape=(num_nodes,))
        reference_radius = model.declare_variable('reference_radius', shape=(num_nodes,))
        rotor_radius = model.declare_variable('rotor_radius', shape=(num_nodes,))
        hub_radius = model.declare_variable('hub_radius', shape=(num_nodes,))

        
        phi_reference = model.declare_variable('phi_reference_ildm', shape=(num_nodes,))

        # Cl = rotor['ideal_Cl_ref_chord']
        # Cd = rotor['ideal_Cd_ref_chord']

        Cl = model.declare_variable('Cl_max_ildm', shape=(num_nodes,))
        Cd = model.declare_variable('Cd_min_ildm', shape=(num_nodes,))
        self.print_var(Cl)
        self.print_var(Cd)

        f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi_reference)
        f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi_reference)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

        F = F_tip * F_hub
        model.register_output('F',F)

        Cx = Cl * csdl.cos(phi_reference) - Cd * csdl.sin(phi_reference)
        Ct = Cl * csdl.sin(phi_reference) + Cd * csdl.cos(phi_reference)
        
        term1 = Vt * (reference_sigma * Cx - 4 * F * csdl.sin(phi_reference)**2)
        term2 = Vx * (2 * F * csdl.sin(2 * phi_reference) + Ct * reference_sigma)

        residual_function = term1 + term2
        model.register_output('residual_function', residual_function)

        eps = 1e-6
        # setting up callable object
        solve_residual_function = self.create_implicit_operation(model)
        solve_residual_function.declare_state('phi_reference_ildm', residual='residual_function',  bracket=(eps, np.pi/2 - eps))
        solve_residual_function.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_residual_function.linear_solver = ScipyKrylov()


        Vx = self.declare_variable('u', shape=(num_nodes,))
        Vt = self.declare_variable('ildm_tangential_inflow_velocity', shape=(num_nodes,))
        reference_sigma = self.declare_variable('reference_blade_solidity', shape=(num_nodes,))
        reference_radius = self.declare_variable('reference_radius', shape=(num_nodes,))
        rotor_radius = self.declare_variable('rotor_radius', shape=(num_nodes,))
        hub_radius = self.declare_variable('hub_radius', shape=(num_nodes,))
        Cl = self.declare_variable('Cl_max_ildm', shape=(num_nodes,))
        Cd = self.declare_variable('Cd_min_ildm', shape=(num_nodes,))

        # For good practice change name
        phi_reference = solve_residual_function(Vx,Vt,reference_sigma, reference_radius,rotor_radius, hub_radius, Cl, Cd) #creates implicit operation
        # if no inputs connections will be left open 
        


        # state = self.._bracketed_search(implicit_metadata = None,
        #     states = 'phi_reference_ildm', 
        #     residuals='residual_function', 
        #     model = model, 
        #     brackets=(eps, np.pi/2 - eps))

        
            # 