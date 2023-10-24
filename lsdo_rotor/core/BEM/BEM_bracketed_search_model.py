import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl
from lsdo_rotor.core.airfoil.BEM_airfoil_surrogate_model_group import BEMAirfoilSurrogateModelGroup
from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters


class BEMBracketedSearchGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor',types=BEMRotorParameters)
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_blades', types=int)
        self.parameters.declare('use_airfoil_ml', default=False)

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
        if rotor['use_airfoil_ml'] is True:
            from lsdo_airfoil.core.airfoil_models import CdModel, ClModel
            
            cl_model = rotor['cl_ml_model']
            cd_model = rotor['cd_ml_model']

            control_points = model.declare_variable('control_points', shape=(shape[0] * shape[1] * shape[2], 32))
            X_min = model.declare_variable('X_min', shape=(shape[0] * shape[1] * shape[2], 35))
            X_max = model.declare_variable('X_max', shape=(shape[0] * shape[1] * shape[2], 35))

            mach_ml = model.declare_variable('mach_number_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))
            Re_ml = model.declare_variable('Re_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))
            alpha_ml = csdl.reshape(alpha, new_shape=(shape[0] * shape[1] * shape[2], 1)) * 180 / np.pi
            model.register_output('alpha_ml_input', alpha_ml)

            inputs = model.create_output('neural_net_input_extrap_unscaled', shape=(shape[0] * shape[1] * shape[2], 35))
            inputs[:, 0:32] = control_points
            inputs[:, 32] = alpha_ml
            inputs[:, 33] = Re_ml
            inputs[:, 34] = mach_ml

            # model.print_var(inputs)

            scaled_inputs_poststall = (inputs - X_min) / (X_max - X_min)
            x_extrap = model.register_output('neural_net_input_extrap', scaled_inputs_poststall)

            output_Cl = csdl.custom(x_extrap, op=ClModel(
                    neural_net=cl_model,
                    num_nodes=int(shape[0] * shape[1] * shape[2]),
                )
            )

            output_Cd = csdl.custom(x_extrap, op=CdModel(
                    neural_net=cd_model,
                    num_nodes=int(shape[0] * shape[1] * shape[2]),
                )
            )
            model.register_output('Cd', output_Cd) #csdl.reshape(cd, new_shape=shape))
            model.register_output('Cl', output_Cl) #csdl.reshape(cl, new_shape=shape))

            Cl = csdl.reshape(model.declare_variable('Cl', shape=(shape[0] * shape[1] * shape[2],)), new_shape=shape)
            Cd = csdl.reshape(model.declare_variable('Cd', shape=(shape[0] * shape[1] * shape[2],)), new_shape=shape)

            # Cl = output_Cl
            # Cd = output_Cd

        elif rotor['use_custom_airfoil_ml'] is True:
            from lsdo_rotor.core.airfoil.ml_trained_models.custom_ml_surrogate_model import CdModel, ClModel

            cl_model = rotor['cl_ml_model']
            cd_model = rotor['cd_ml_model']

            X_min = model.declare_variable('X_min', shape=(shape[0] * shape[1] * shape[2], 3))
            X_max = model.declare_variable('X_max', shape=(shape[0] * shape[1] * shape[2], 3))

            mach_ml = model.declare_variable('mach_number_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))
            Re_ml = model.declare_variable('Re_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))
            alpha_ml = csdl.reshape(alpha, new_shape=(shape[0] * shape[1] * shape[2], 1)) * 180 / np.pi
            model.register_output('alpha_ml_input', alpha_ml)

            inputs = model.create_output('neural_net_input_extrap_unscaled', shape=(shape[0] * shape[1] * shape[2], 3))
            inputs[:, 0] = alpha_ml
            inputs[:, 1] = Re_ml
            inputs[:, 2] = mach_ml

            scaled_inputs_poststall = (inputs - X_min) / (X_max - X_min)
            x_extrap = model.register_output('neural_net_input_extrap', scaled_inputs_poststall)

            output_Cl = csdl.custom(x_extrap, op=ClModel(
                    neural_net=cl_model,
                    num_nodes=int(shape[0] * shape[1] * shape[2]),
                )
            )

            output_Cd = csdl.custom(x_extrap, op=CdModel(
                    neural_net=cd_model,
                    num_nodes=int(shape[0] * shape[1] * shape[2]),
                )
            )
            model.register_output('Cd', output_Cd) #csdl.reshape(cd, new_shape=shape))
            model.register_output('Cl', output_Cl) #csdl.reshape(cl, new_shape=shape))

            Cl = csdl.reshape(model.declare_variable('Cl', shape=(shape[0] * shape[1] * shape[2],)), new_shape=shape)
            Cd = csdl.reshape(model.declare_variable('Cd', shape=(shape[0] * shape[1] * shape[2],)), new_shape=shape)

        
        elif not rotor['custom_polar']:
            airfoil_model_output = csdl.custom(Re, alpha, chord, op= BEMAirfoilSurrogateModelGroup(
                rotor=rotor,
                shape=shape,
            ))
            model.register_output('Cl',airfoil_model_output[0])
            model.register_output('Cd',airfoil_model_output[1])

            Cl = airfoil_model_output[0]
            Cd = airfoil_model_output[1]
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

        # model.declare_variable('Cl', shape=shape)
        # model.declare_variable('Cd', shape=shape)
        
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

        if rotor['use_airfoil_ml'] is False and rotor['use_custom_airfoil_ml'] is False:
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
            phi = solve_BEM_residual(sigma,Vx,Vt,radius,rotor_radius,hub_radius,chord,twist,Re, expose=['Cl', 'Cd', 'alpha_distribution'])

        elif rotor['use_airfoil_ml'] is True and rotor['use_custom_airfoil_ml'] is False:
            sigma = self.declare_variable('_blade_solidity', shape=shape)
            Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
            Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
            radius = self.declare_variable('_radius',shape= shape)
            rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
            hub_radius = self.declare_variable('_hub_radius', shape=shape)
            twist = self.declare_variable('_pitch', shape=shape)
            
            control_points = self.declare_variable('control_points', shape=(shape[0] * shape[1] * shape[2], 32))
            X_min = self.declare_variable('X_min', shape=(shape[0] * shape[1] * shape[2], 35))
            X_max = self.declare_variable('X_max', shape=(shape[0] * shape[1] * shape[2], 35))

            mach_ml = self.declare_variable('mach_number_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))
            Re_ml = self.declare_variable('Re_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))

            phi = solve_BEM_residual(sigma, Vx, Vt, radius, rotor_radius, hub_radius, twist, Re_ml, mach_ml, X_max, X_min, control_points, expose=['Cl', 'Cd', 'alpha_distribution'])

        
        elif rotor['use_airfoil_ml'] is False and rotor['use_custom_airfoil_ml'] is True:
            sigma = self.declare_variable('_blade_solidity', shape=shape)
            Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
            Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
            radius = self.declare_variable('_radius',shape= shape)
            rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
            hub_radius = self.declare_variable('_hub_radius', shape=shape)
            twist = self.declare_variable('_pitch', shape=shape)
            
            X_min = self.declare_variable('X_min', shape=(shape[0] * shape[1] * shape[2], 3))
            X_max = self.declare_variable('X_max', shape=(shape[0] * shape[1] * shape[2], 3))

            mach_ml = self.declare_variable('mach_number_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))
            Re_ml = self.declare_variable('Re_ml_input', shape=(shape[0] * shape[1] * shape[2], 1))

            phi = solve_BEM_residual(sigma, Vx, Vt, radius, rotor_radius, hub_radius, twist, Re_ml, mach_ml, X_max, X_min,  expose=['Cl', 'Cd', 'alpha_distribution'])


        else:
            raise NotImplementedError