import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl

from lsdo_rotor.airfoil.airfoil_surrogate_model_group import AirfoilSurrogateModelGroup
from lsdo_rotor.core.atmosphere_group import AtmosphereGroup

class PhiBracketedSearchGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor')
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('mode', types = int)

    def define(self):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        mode = self.parameters['mode']
        B = num_blades = rotor['num_blades']

        if mode == 1:
            model = Model()

            Vx = model.declare_variable('reference_axial_inflow_velocity')
            Vt = model.declare_variable('reference_tangential_inflow_velocity')
            reference_sigma = model.declare_variable('reference_blade_solidity')
            reference_radius = model.declare_variable('reference_radius')
            rotor_radius = model.declare_variable('rotor_radius')
            hub_radius = model.declare_variable('hub_radius')
            
            phi_reference = model.declare_variable('phi_reference_ildm')

            Cl = rotor['ideal_Cl_ref_chord']
            Cd = rotor['ideal_Cd_ref_chord']

            f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi_reference)
            f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi_reference)

            F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
            F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

            F = F_tip * F_hub

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


            Vx = model.declare_variable('reference_axial_inflow_velocity')
            Vt = model.declare_variable('reference_tangential_inflow_velocity')
            reference_sigma = model.declare_variable('reference_blade_solidity')
            reference_radius = model.declare_variable('reference_radius')
            rotor_radius = model.declare_variable('rotor_radius')
            hub_radius = model.declare_variable('hub_radius')

            # For good practice change name
            phi_reference = solve_residual_function(Vx,Vt,reference_sigma, reference_radius,rotor_radius, hub_radius) #creates implicit operation
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
            hub_radius = model.declare_variable('_hub_radius', shape = shape)
            chord = model.declare_variable('chord_distribution',shape=shape)
            twist = model.declare_variable('pitch_distribution', shape=shape)
            
            # phi is state (inflow angle) we're solving for in the bracketed search
            phi = model.declare_variable('phi_distribution', shape = shape)
  
            # Adding atmosphere group to compute Reynolds number 
            model.add(AtmosphereGroup(
                shape = shape,
                rotor = rotor,
                mode = mode,
            ), name = 'atmosphere_group', promotes = ['*'])

            Re = model.declare_variable('Re', shape = shape)

            alpha = twist - phi
            model.register_output('alpha_distribution', alpha)
            
            # Adding custom component to embed airfoil model in the bracketed search
            airfoil_model_output = csdl.custom(Re,alpha,chord, op= AirfoilSurrogateModelGroup(
                rotor = rotor,
                shape = shape,
            ))
            model.register_output('Cl',airfoil_model_output[0])
            model.register_output('Cd',airfoil_model_output[1])
            
            Cl = airfoil_model_output[0]
            Cd = airfoil_model_output[1]
          
            # Embedding Prandtl tip losses 
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
            # print(eps)
            solve_BEM_residual = self.create_implicit_operation(model)
            solve_BEM_residual.declare_state('phi_distribution', residual='BEM_residual_function', bracket=(eps, np.pi/2 - eps))

            # sigma = model.declare_variable('_blade_solidity', shape=shape)
            # Vx = model.declare_variable('_axial_inflow_velocity', shape=shape)
            # Vt = model.declare_variable('_tangential_inflow_velocity', shape=shape)
            # radius = model.declare_variable('_radius',shape= shape)
            # rotor_radius = model.declare_variable('_rotor_radius', shape= shape)
            # hub_radius = model.declare_variable('_hub_radius', shape = shape)
            # chord = model.declare_variable('chord_distribution',shape=shape)
            # twist = model.declare_variable('pitch_distribution', shape=shape)
            # Re = model.declare_variable('Re', shape = shape)
        

            phi, Cl, Cd,F, Cx, Ct = solve_BEM_residual(sigma,Vx,Vt,radius,rotor_radius,hub_radius,chord,twist,Re, expose = ['Cl', 'Cd','F','Cx','Ct'])



            
# from lsdo_rotor_csdl.airfoil.get_surrogate_model import get_surrogate_model
# from lsdo_rotor_csdl.functions.get_rotor_dictionary import get_rotor_dictionary
# from lsdo_rotor_csdl.functions.get_max_LD_parameters_reference_chord import get_max_LD_parameters_reference_chord

# airfoil             = 'NACA_4412_extended_range'
# interp              = get_surrogate_model(airfoil)

# rotor_diameter      = 1.5
# RPM                 = 1500
# V_inf               = 0
# num_blades          = 3
# altitude            = 10        # (in m)

# reference_chord     = 0.15
# reference_twist     = 40
# reference_radius    = 0.456

# root_chord          = 0.3       # Chord length at the root
# root_twist          = 60        # Twist angle at the blade root (deg)

# tip_chord           = 0.05
# tip_twist           = 20    

# num_evaluations     = 1         # Discretization in time               
# num_radial          = 50        # Discretization in spanwise direction   
# num_tangential      = 1         # Discretization in tangential/azimuthal direction


# """
#     Mode settings
#         1 --> Ideal-Loading Design Method
#         2 --> BEM
# """
# mode = 2 

# plot_rotor_blade_shape  = 'y'     # Do you wish to plot the chord and twist distribution of the blade? [y/n]
# plot_rotor_performance  = 'n'     # Do you wish to plot rotor performance quantities along the blade span? [y/n]
# print_rotor_performance = 'y'

# #---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
# ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, V_inf, RPM, altitude)
# rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord)

# shape = (num_evaluations, num_radial, num_tangential)


# from csdl_om import Simulator
# sim = Simulator(PhiBracketedSearchGroup(mode=mode,
# shape=shape,
# rotor=rotor))
# sim.visualize_implementation()









# class M(Model):
#     def define(self):
#         model = self

#         B = 3
            
#         sigma = model.declare_variable('_blade_solidity', shape=shape)
#         Vx = model.declare_variable('_axial_inflow_velocity', shape=shape)
#         Vt = model.declare_variable('_tangential_inflow_velocity', shape=shape)
#         radius = model.declare_variable('_radius',shape= shape)
#         rotor_radius = model.declare_variable('_rotor_radius', shape= shape)
#         hub_radius = model.declare_variable('_hub_radius', shape = shape)
#         chord = model.declare_variable('chord_distribution',shape=shape)
#         twist = model.declare_variable('pitch_distribution', shape=shape)
#         Re = model.declare_variable('Re', shape = shape)

#         phi = model.declare_variable('phi_distribution', shape = shape)

#         # altitude = rotor['altitude']
#         # L           = 6.5
#         # R           = 287
#         # T0          = 288.16
#         # P0          = 101325
#         # g0          = 9.81
#         # mu0         = 1.735e-5
#         # S1          = 110.4


#         # # Temperature 
#         # T           = T0 - L * altitude
        
#         # # Pressure 
#         # P           = P0 * (T/T0)**(g0/L/R)
        
#         # # Density
#         # rho         = P/R/T 
        
#         # # Dynamic viscosity (using Sutherland's law)  
#         # mu          = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1)

#         # # Reynolds number
#         # W           = (Vx**2 + Vt**2)**0.5
#         # Re          = rho * W * chord / mu

#         # model.register_output('Re', Re)

#         alpha = twist - phi
#         model.register_output('alpha_distribution', alpha)
#         # exit()
#         model.add(AirfoilSurrogateModelGroup(
#             rotor = rotor,
#             shape = shape,
#         ), name = 'airfoil_surrogate_model_group', promotes = ['*'])
        
#         Cl = model.declare_variable('_Cl', shape = shape)
#         Cd = model.declare_variable('_Cd', shape = shape)

#         # print(Cl)

#         f_tip = B / 2 * (rotor_radius - radius) / radius / csdl.sin(phi)
#         f_hub = B / 2 * (radius - hub_radius) / hub_radius / csdl.sin(phi)

#         F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
#         F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

#         F = F_tip * F_hub

#         Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
#         Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)
        
#         term1 = Vt * (sigma * Cx - 4 * F * csdl.sin(phi)**2)
#         term2 = Vx * (2 * F * csdl.sin(2 * phi) + Ct * sigma)
        
#         BEM_residual = term1 + term2
        

#         model.register_output('BEM_residual_function', BEM_residual)
            

# class PhiBracketedSearchGroup(Model):

#     def initialize(self):
#         self.parameters.declare('rotor')
#         self.parameters.declare('shape', types=tuple)
#         self.parameters.declare('mode', types = int)

#     def define(self):
#         rotor = self.parameters['rotor']
#         shape = self.parameters['shape']
#         mode = self.parameters['mode']
#         B = num_blades = rotor['num_blades']

#         if mode == 1:
#             model = Model()

#             Vx = model.declare_variable('reference_axial_inflow_velocity')
#             Vt = model.declare_variable('reference_tangential_inflow_velocity')
#             reference_sigma = model.declare_variable('reference_blade_solidity')
#             reference_radius = model.declare_variable('reference_radius')
#             rotor_radius = model.declare_variable('rotor_radius')
#             hub_radius = model.declare_variable('hub_radius')
            
#             phi_reference = model.declare_variable('phi_reference')

#             Cl = rotor['ideal_Cl_ref_chord']
#             Cd = rotor['ideal_Cd_ref_chord']

#             f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi_reference)
#             f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi_reference)

#             F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
#             F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

#             F = F_tip * F_hub

#             Cx = Cl * csdl.cos(phi_reference) - Cd * csdl.sin(phi_reference)
#             Ct = Cl * csdl.sin(phi_reference) + Cd * csdl.cos(phi_reference)
           
#             term1 = Vt * (reference_sigma * Cx - 4 * F * csdl.sin(phi_reference)**2)
#             term2 = Vx * (2 * F * csdl.sin(2 * phi_reference) + Ct * reference_sigma)

#             residual_function = term1 + term2
#             model.register_output('residual_function', residual_function)

#             eps = 1e-6
#             # setting up callable object
#             solve_residual_function = self.create_implicit_operation(model)
#             solve_residual_function.declare_state('phi_reference', residual='residual_function', bracket=(eps, np.pi/2 - eps))

#             # For good practice change name
#             phi_reference = solve_residual_function() #creates implicit operation
#             # if no inputs connections will be left open 
#             # 


#         elif mode == 2:
            

#             eps = 1e-6
#             solve_BEM_residual = self.create_implicit_operation(M())
#             solve_BEM_residual.declare_state('phi_distribution', residual='BEM_residual_function', bracket=(eps, np.pi/2 - eps))

           

#             # phi, x = solve_BEM_residual(*[], expose = ['phi_distribution'])
#             phi = solve_BEM_residual()
#             # self.print_var(x)
#             print('************#################@@@@@@@@@@@@@')
            
# from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
# from lsdo_rotor.functions.get_rotor_dictionary import get_rotor_dictionary
# from lsdo_rotor.functions.get_max_LD_parameters_reference_chord import get_max_LD_parameters_reference_chord

# airfoil             = 'NACA_4412_extended_range'
# interp              = get_surrogate_model(airfoil)

# rotor_diameter      = 1.5
# RPM                 = 1500
# V_inf               = 0
# num_blades          = 3
# altitude            = 10        # (in m)

# reference_chord     = 0.15
# reference_twist     = 40
# reference_radius    = 0.456

# root_chord          = 0.3       # Chord length at the root
# root_twist          = 60        # Twist angle at the blade root (deg)

# tip_chord           = 0.05
# tip_twist           = 20    

# num_evaluations     = 1         # Discretization in time               
# num_radial          = 50        # Discretization in spanwise direction   
# num_tangential      = 1         # Discretization in tangential/azimuthal direction


# """
#     Mode settings
#         1 --> Ideal-Loading Design Method
#         2 --> BEM
# """
# mode = 2 

# plot_rotor_blade_shape  = 'y'     # Do you wish to plot the chord and twist distribution of the blade? [y/n]
# plot_rotor_performance  = 'n'     # Do you wish to plot rotor performance quantities along the blade span? [y/n]
# print_rotor_performance = 'y'

# #---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
# ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, V_inf, RPM, altitude)
# rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord)

# shape = (num_evaluations, num_radial, num_tangential)


# from csdl_om import Simulator
# # p = PhiBracketedSearchGroup(mode=mode,
# # shape=shape,
# # rotor=rotor)
# # p.define()
# sim = Simulator(M())
# sim.visualize_implementation()









