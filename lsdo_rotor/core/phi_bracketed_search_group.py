import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl

from lsdo_rotor.airfoil.airfoil_surrogate_model_group import AirfoilSurrogateModelGroup
from lsdo_rotor.core.atmosphere_model import AtmosphereModel
from lsdo_rotor.rotor_parameters import RotorParameters

class PhiBracketedSearchGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor',types=RotorParameters)
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('mode', types = int)

    def define(self):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        mode = self.parameters['mode']
        B = num_blades = rotor['num_blades']

        if mode == 1:
            model = Model()

            Vx = model.declare_variable('ildm_axial_inflow_velocity' , shape = (shape[0],))
            Vt = model.declare_variable('ildm_tangential_inflow_velocity', shape = (shape[0],))
            self.register_output('test_ildm_axial_inflow_velocity', Vx*1)
            self.register_output('test_ildm_tangential_inflow_velocity', Vt*1)
            reference_sigma = model.declare_variable('reference_blade_solidity', shape = (shape[0],))
            self.register_output('test_reference_blade_solidity', reference_sigma *1)
            reference_radius = model.declare_variable('reference_radius', shape = (shape[0],))
            self.register_output('test_reference_radius',reference_radius * 1)
            rotor_radius = model.declare_variable('rotor_radius', shape = (shape[0],))
            self.register_output('test_rotor_radius', rotor_radius *1)
            hub_radius = model.declare_variable('hub_radius', shape = (shape[0],))
            self.register_output('test_hub_radius', hub_radius * 1)
            
            phi_reference = model.declare_variable('phi_reference_ildm', shape = (shape[0],))

            # Cl = rotor['ideal_Cl_ref_chord']
            # Cd = rotor['ideal_Cd_ref_chord']

            Cl = model.declare_variable('Cl_max_ildm', shape = (shape[0],))
            Cd = model.declare_variable('Cd_min_ildm', shape = (shape[0],))
            self.register_output('test_Cl',Cl*1)
            self.register_output('test_Cd',Cd*1)

            f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi_reference)
            f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi_reference)

            F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
            F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

            F = F_tip * F_hub
            model.register_output('F',F)
            self.register_output('test_F',F*1)

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


            Vx = self.declare_variable('ildm_axial_inflow_velocity', shape = (shape[0],))
            Vt = self.declare_variable('ildm_tangential_inflow_velocity', shape = (shape[0],))
            reference_sigma = self.declare_variable('reference_blade_solidity', shape = (shape[0],))
            reference_radius = self.declare_variable('reference_radius', shape = (shape[0],))
            rotor_radius = self.declare_variable('rotor_radius', shape = (shape[0],))
            hub_radius = self.declare_variable('hub_radius', shape = (shape[0],))
            Cl = self.declare_variable('Cl_max_ildm', shape = (shape[0],))
            Cd = self.declare_variable('Cd_min_ildm', shape = (shape[0],))

            # For good practice change name
            phi_reference = solve_residual_function(Vx,Vt,reference_sigma, reference_radius,rotor_radius, hub_radius, Cl, Cd) #creates implicit operation
            # if no inputs connections will be left open 
            


            # state = self.._bracketed_search(implicit_metadata = None,
            #     states = 'phi_reference_ildm', 
            #     residuals='residual_function', 
            #     model = model, 
            #     brackets=(eps, np.pi/2 - eps))

            
            # 


        elif mode == 2:
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
  
            # Adding atmosphere group to compute Reynolds number 
            model.add(AtmosphereModel(
                shape=shape,
                rotor=rotor,
                mode = mode,
            ), name = 'atmosphere_model', promotes = ['*'])

            Re = model.declare_variable('Re', shape=shape)

            alpha = twist - phi
            model.register_output('alpha_distribution', alpha)
            
            # Adding custom component to embed airfoil model in the bracketed search
            airfoil_model_output = csdl.custom(Re,alpha,chord, op= AirfoilSurrogateModelGroup(
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

            # term1 = Vt * sigma * Cx 
            # term2 = -4 * csdl.sin(phi) * (Vt * csdl.sin(phi) - Vx * csdl.cos(phi)) * F
            
            BEM_residual = term1 + term2
            
            model.register_output('BEM_residual_function', BEM_residual)
            
            # Solving residual function for state phi 
            eps = 1e-7
            # print(eps)
            # from csdl_om import Simulator
            # sim = Simulator(model)
            # sim.run()
            # sim.prob.check_partials(compact_print=True)
            # exit()
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
