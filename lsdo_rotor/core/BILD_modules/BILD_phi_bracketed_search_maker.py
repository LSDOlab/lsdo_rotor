import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl
from lsdo_modules.module.module_maker import ModuleMaker


class BILDPhiBracketedSearchMaker(ModuleMaker):

    def initialize_module(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_blades', types=int)

    def define_module(self):
        shape = self.parameters['shape']
        B = num_blades = self.parameters['num_blades']
        num_nodes = shape[0]

        module = ModuleMaker()

        Vx = module.register_module_input('u' , shape=(num_nodes,))
        Vt = module.register_module_input('BILD_tangential_inflow_velocity', shape=(num_nodes,))
        reference_sigma = module.register_module_input('reference_blade_solidity', shape=(num_nodes,))
        reference_radius = module.register_module_input('reference_radius', shape=(num_nodes,))
        rotor_radius = module.register_module_input('propeller_radius', shape=(num_nodes,))
        hub_radius = module.register_module_input('hub_radius', shape=(num_nodes,))

        
        phi_reference = module.register_module_input('phi_reference_BILD', shape=(num_nodes,))

        # Cl = rotor['ideal_Cl_ref_chord']
        # Cd = rotor['ideal_Cd_ref_chord']

        Cl = module.register_module_input('Cl_max_BILD', shape=(num_nodes,))
        Cd = module.register_module_input('Cd_min_BILD', shape=(num_nodes,))
        # self.print_var(Cl)
        # self.print_var(Cd)

        f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi_reference)
        f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi_reference)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

        F = F_tip * F_hub
        module.register_module_output('F', F)

        Cx = Cl * csdl.cos(phi_reference) - Cd * csdl.sin(phi_reference)
        Ct = Cl * csdl.sin(phi_reference) + Cd * csdl.cos(phi_reference)
        
        term1 = Vt * (reference_sigma * Cx - 4 * F * csdl.sin(phi_reference)**2)
        term2 = Vx * (2 * F * csdl.sin(2 * phi_reference) + Ct * reference_sigma)

        residual_function = term1 + term2
        module.register_module_output('residual_function', residual_function)

        eps = 1e-6
        # setting up callable object
        solve_residual_function = self.create_implicit_operation(module)
        solve_residual_function.declare_state('phi_reference_BILD', residual='residual_function',  bracket=(eps, np.pi/2 - eps))
        solve_residual_function.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )

        Vx = self.register_module_input('u', shape=(num_nodes,))
        Vt = self.register_module_input('BILD_tangential_inflow_velocity', shape=(num_nodes,))
        reference_sigma = self.register_module_input('reference_blade_solidity', shape=(num_nodes,))
        reference_radius = self.register_module_input('reference_radius', shape=(num_nodes,))
        rotor_radius = self.register_module_input('propeller_radius', shape=(num_nodes,))
        hub_radius = self.register_module_input('hub_radius', shape=(num_nodes,))
        Cl = self.register_module_input('Cl_max_BILD', shape=(num_nodes,))
        Cd = self.register_module_input('Cd_min_BILD', shape=(num_nodes,))

        # For good practice change name
        phi_reference = solve_residual_function(Vx,Vt,reference_sigma, reference_radius,rotor_radius, hub_radius, Cl, Cd) #creates implicit operation
        # if no inputs connections will be left open 

        self.add_module(module, 'imp_mod_brack_phi')
        