from email.policy import default
from xmlrpc.client import Boolean
import numpy as np
from csdl import Model
import csdl

# from lsdo_rotor.rotor_parameters import RotorParameters

from lsdo_rotor.core.BEM.inputs.BEM_external_inputs_model import BEMExternalInputsModel
from lsdo_rotor.core.BEM.inputs.BEM_core_inputs_model import BEMCoreInputsModel
from lsdo_rotor.core.BEM.inputs.BEM_pre_process_model import BEMPreprocessModel
from lsdo_rotor.core.BEM.BEM_bracketed_search_model import BEMBracketedSearchGroup
from lsdo_rotor.core.BEM.BEM_prandtl_loss_factor_model import BEMPrandtlLossFactorModel
from lsdo_rotor.core.BEM.BEM_induced_velocity_model import BEMInducedVelocityModel
from lsdo_rotor.core.airfoil.BEM_airfoil_surrogate_model_group_2 import BEMAirfoilSurrogateModelGroup2

from lsdo_rotor.core.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.BEM.functions.get_BEM_rotor_dictionary import get_BEM_rotor_dictionary

from lsdo_rotor.utils.atmosphere_model import AtmosphereModel

from lsdo_rotor.core.BEM.functions.get_bspline_mtx import get_bspline_mtx
from lsdo_rotor.core.BEM.BEM_b_spline_comp import BsplineComp


class BEMModel(Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('BEM_parameters')# , types=BEMMesh)
        self.parameters.declare('num_nodes')
        self.parameters.declare('operation')
        self.parameters.declare('stability_flag', types=bool, default=False)

    def define(self):
        bem_parameters = self.parameters['BEM_parameters']
        num_nodes = self.parameters['num_nodes']
        stability_flag = self.parameters['stability_flag']
        
        operation = self.parameters['operation']
        arguments = operation.arguments
        name = operation.name

        num_radial = bem_parameters.parameters['num_radial']
        num_tangential = bem_parameters.parameters['num_tangential']
        airfoil = bem_parameters.parameters['airfoil']
        norm_hub_rad = bem_parameters.parameters['normalized_hub_radius']
        custom_polar = bem_parameters.parameters['airfoil_polar']
        reference_point = bem_parameters.parameters['ref_pt']
        num_blades = bem_parameters.parameters['num_blades']

        use_airfoil_ml = bem_parameters.parameters['use_airfoil_ml']
        use_custom_airfoil_ml = bem_parameters.parameters['use_custom_airfoil_ml']
        units = bem_parameters.parameters['mesh_units']

        shape = (num_nodes, num_radial, num_tangential)       
        
        omega_a = self.declare_variable('rpm', shape=(num_nodes, 1), units='rpm') #, computed_upstream=False)

        u_a = self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s')
        v_a = self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s', val=0) 
        w_a = self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s', val=0) 
        p_a = self.declare_variable(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
        q_a = self.declare_variable(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
        r_a = self.declare_variable(name='r', shape=(num_nodes, 1), units='rad/s', val=0)
        phi = self.declare_variable(name='phi', shape=(num_nodes, 1), val=0)
        theta = self.declare_variable(name='theta', shape=(num_nodes, 1), val=0)
        psi = self.declare_variable(name='psi', shape=(num_nodes, 1), val=0)
       
        rotation_matrix = self.create_output('rotation_matrix', shape=(num_nodes, 3, 3), val=0)
        for i in range(num_nodes):

            # T_theta = np.array([
            #     [np.cos(theta), 0, np.sin(theta)],
            #     [0, 1, 0],
            #     [-np.sin(theta), 0, np.cos(theta)],
            #  ])
            rotation_matrix[i, 0, 0] = csdl.reshape(csdl.cos(theta[i, 0]), new_shape=(1, 1, 1))
            rotation_matrix[i, 0, 2] = csdl.reshape(1 * csdl.sin(theta[i, 0]), new_shape=(1, 1, 1))
            rotation_matrix[i, 1, 1] = csdl.reshape((theta[i, 0] + 1) / (theta[i, 0] + 1), new_shape=(1, 1, 1))
            rotation_matrix[i, 2, 0] = csdl.reshape(-1 * csdl.sin(theta[i, 0]), new_shape=(1, 1, 1))
            rotation_matrix[i, 2, 2] = csdl.reshape(1 * csdl.cos(theta[i, 0]), new_shape=(1, 1, 1))

        if (use_airfoil_ml is False) and (use_custom_airfoil_ml is False):
            interp = get_surrogate_model(airfoil, custom_polar)
            rotor = get_BEM_rotor_dictionary(airfoil, interp, custom_polar)
        
        elif (use_custom_airfoil_ml is True) and (use_airfoil_ml is False):
            X_max_numpy = np.array([90., 8e6, 0.65])
            X_min_numpy = np.array([-90., 1e5, 0.])
            
            X_min = self.create_input('X_min', val=np.tile(X_min_numpy.reshape(3, 1), num_nodes*num_radial*num_tangential).T)
            X_max = self.create_input('X_max', val=np.tile(X_max_numpy.reshape(3, 1), num_nodes*num_radial*num_tangential).T)

            from lsdo_rotor.core.airfoil.ml_trained_models.get_airfoil_model import get_airfoil_models

            neural_nets = get_airfoil_models(airfoil=airfoil)
            cl_model = neural_nets.Cl
            cd_model = neural_nets.Cd
            rotor = get_BEM_rotor_dictionary(airfoil_name=airfoil, ml_cl=cl_model, ml_cd=cd_model, use_airfoil_ml=use_airfoil_ml,use_custom_airfoil_ml=use_custom_airfoil_ml)
            

        elif (use_custom_airfoil_ml is False) and (use_airfoil_ml is True):
            from lsdo_airfoil.utils.load_control_points import load_control_points
            from lsdo_airfoil.utils.get_airfoil_model import get_airfoil_models
            from lsdo_airfoil.core.airfoil_model_csdl import X_max_numpy_poststall, X_min_numpy_poststall


            X_min = self.create_input('X_min', val=np.tile(X_min_numpy_poststall.reshape(35, 1), num_nodes*num_radial*num_tangential).T)
            X_max = self.create_input('X_max', val=np.tile(X_max_numpy_poststall.reshape(35, 1), num_nodes*num_radial*num_tangential).T)
            
            control_points_numpy = np.tile(load_control_points(airfoil_name=airfoil), num_nodes*num_radial*num_tangential).T
            control_points = self.create_input('control_points', val=control_points_numpy)
            airfoil_models = get_airfoil_models()
            cl_model = airfoil_models['Cl']
            cd_model = airfoil_models['Cd']
            rotor = get_BEM_rotor_dictionary(airfoil_name=airfoil, ml_cl=cl_model, ml_cd=cd_model, use_airfoil_ml=use_airfoil_ml)
        
        elif (use_custom_airfoil_ml is True) and (use_airfoil_ml is True):
            raise ValueError("Cannot specify 'use_custom_airfoil_ml=True' and 'use_airfoil_ml=True' at the same time")

        else:
            raise NotImplementedError

        # --------------- Rotor geometry --------------- #
        if units == 'ft':
            r = self.declare_variable('R', shape=(1, ))
            radius = self.register_output('propeller_radius', r * 0.3048) 
            # self.print_var(radius)

            to = self.declare_variable('to', shape=(num_nodes, 3)) 
            self.register_output('thrust_origin', to * 0.3048)
            self.print_var(to)


            if 'chord_dist' in arguments: 
                chord = self.declare_variable('chord_dist', shape=(num_radial, ))
                chord_in_m = self.register_output('chord_profile', chord * 0.3048)
                # self.print_var(chord_in_m)
                

            elif arguments['chord_cp']:
                chord_cp_shape = arguments['chord_cp'].shape
                chord_cp = self.declare_variable(name='chord_cp', shape=chord_cp_shape)
                order = bem_parameters.parameters['b_spline_order']
                num_cp = bem_parameters.parameters['num_cp']
                chord_A = get_bspline_mtx(num_cp, num_radial, order=order)
                comp_chord = csdl.custom(chord_cp,op=BsplineComp(
                    num_pt=num_radial,
                    num_cp=num_cp,
                    in_name='chord_cp',
                    jac=chord_A,
                    out_name='chord_profile',
                ))
                self.register_output('chord_profile', comp_chord * 0.3048)

            else:
                raise NotImplementedError

        else:
            r = self.declare_variable('R', shape=(1, ))
            radius = self.register_output('propeller_radius', r * 1) 
            # self.print_var(radius)

            to = self.declare_variable('to', shape=(num_nodes, 3)) 
            self.register_output('thrust_origin', to * 1)
            self.print_var(to)


            if 'chord_dist' in arguments: 
                chord = self.declare_variable('chord_dist', shape=(num_radial, ))
                self.register_output('chord_profile', chord * 1)
            
            elif arguments['chord_cp']:
                chord_cp_shape = arguments['chord_cp'].shape
                chord_cp = self.declare_variable(name='chord_cp', shape=chord_cp_shape)
                order = bem_parameters.parameters['b_spline_order']
                num_cp = bem_parameters.parameters['num_cp']
                chord_A = get_bspline_mtx(num_cp, num_radial, order=order)
                comp_chord = csdl.custom(chord_cp,op=BsplineComp(
                    num_pt=num_radial,
                    num_cp=num_cp,
                    in_name='chord_cp',
                    jac=chord_A,
                    out_name='chord_profile',
                ))
                self.register_output('chord_profile', comp_chord)


        if 'twist_profile' in arguments:
            self.declare_variable('twist_profile', shape=(num_radial, ))
        
        elif arguments['twist_cp']:
            twist_cp_shape = arguments['twist_cp'].shape
            twist_cp = self.declare_variable(name='twist_cp', shape=twist_cp_shape)
            order = bem_parameters.parameters['b_spline_order']
            num_cp = bem_parameters.parameters['num_cp']
            pitch_A = get_bspline_mtx(num_cp, num_radial, order=order)
            comp = csdl.custom(twist_cp,op=BsplineComp(
                num_pt=num_radial,
                num_cp=num_cp,
                in_name='twist_cp',
                jac=pitch_A,
                out_name='twist_profile',
            ))
            self.register_output('twist_profile', comp)
                    
        tv = self.declare_variable('thrust_vector', shape=(num_nodes, 3)) 
        
        

        # External inputs
        self.add(BEMExternalInputsModel(
            shape=shape,
            hub_radius_percent=norm_hub_rad,
        ), name='BEM_external_inputs_model')

        # Core inputs 
        self.add(BEMCoreInputsModel(
            shape=shape,
        ),name='BEM_core_inputs_model')

        # Preprocess
        self.add(BEMPreprocessModel(
            shape=shape,
            num_blades=num_blades,
        ),name='BEM_pre_process_model')

        # self.add(AtmosphereModel(
        #     shape=(num_nodes, 1),
        # ),name='atmosphere_model')
    
        # Computing Reynolds and Mach numbers
        chord = self.declare_variable('_chord',shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        W = (Vx**2 + Vt**2)**0.5
        rho = csdl.expand(self.declare_variable('density', shape=(num_nodes, )), shape,'i->ijk')
        mu = csdl.expand(self.declare_variable('dynamic_viscosity', shape=(num_nodes, )), shape, 'i->ijk')
        speed_of_sound = csdl.expand(self.declare_variable('speed_of_sound', shape=(num_nodes, )), shape, 'i->ijk')
        
        Re = rho * W * chord / mu
        mach = W / speed_of_sound
        
        self.register_output('Re', Re)
        self.register_output('mach_number', mach)

        self.register_output('Re_ml_input', csdl.reshape(Re, new_shape=(num_nodes * num_radial * num_tangential, 1)))
        self.register_output('mach_number_ml_input', csdl.reshape(mach, new_shape=(num_nodes * num_radial * num_tangential, 1)))

        # Bracketed search
        self.add(BEMBracketedSearchGroup(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

       

        self.add(BEMPrandtlLossFactorModel(
            shape=shape,
            num_blades=num_blades,
        ), name = 'prandtl_loss_factor_model')#, promotes = ['*'])

        self.add(BEMInducedVelocityModel(
            shape=shape,
            num_blades=num_blades,
        ), name = 'induced_velocity_model')#, promotes = ['*'])

        # Post-Processing
        if stability_flag:
            T = self.declare_variable('T_compute', shape=(num_nodes,))
            F = self.create_output('F_compute', shape=(num_nodes,3))
            ref_pt = self.declare_variable('reference_point',shape=(num_nodes,3), val=np.tile(reference_point,(num_nodes,1)))
            thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes, 3))
            # self.print_var(thrust_vector)
            thrust_origin = self.declare_variable('thrust_origin', shape=(num_nodes, 3))
            # loop over pt set list 
            

            for i in range(num_nodes):
                # F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
                F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] #- 9 * hub_drag[i,0]
                F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1]
                F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
                

            M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
            self.register_output('M_compute', csdl.transpose(M))

            self.register_output(name='F', var=F[0, :])
            self.register_output(name='F_perturbed', var=F[1:, :])
            self.register_output(name='M', var=M[0, :])
            self.register_output(name='M_perturbed', var=M[1:, :])

            C_T = self.declare_variable('C_T_compute', shape=(num_nodes, ))
            C_Q = self.declare_variable('C_Q_compute', shape=(num_nodes, ))
            Q = self.declare_variable('Q_compute', shape=(num_nodes, ))
            T = self.declare_variable('T_compute', shape=(num_nodes, ))
            eta = self.declare_variable('eta_compute', shape=(num_nodes, ))
            FOM = self.declare_variable('FOM_compute', shape=(num_nodes, ))
            dT = self.declare_variable('_dT_compute', shape=(num_nodes, num_radial, num_tangential))
            dQ = self.declare_variable('_dQ_compute', shape=(num_nodes, num_radial, num_tangential))
            dD = self.declare_variable('_dD_compute', shape=(num_nodes, num_radial, num_tangential))
            ux = self.declare_variable('_ux_compute', shape=(num_nodes, num_radial, num_tangential))
            phi = self.declare_variable('phi_distribution', shape=(num_nodes, num_radial, num_tangential))

            self.register_output('C_T', C_T[0])
            self.register_output('C_Q', C_Q[0])
            self.register_output('Q', Q[0])
            self.register_output('T', T[0])
            self.register_output('eta', eta[0])
            self.register_output('FOM', FOM[0])
            self.register_output('_dT', dT[0, :, :])
            self.register_output('_dQ', dQ[0, :, :])
            self.register_output('_dD', dD[0, :, :])
            self.register_output('_ux', ux[0, :, :])
            self.register_output('_phi', phi[0, :, :])
        # self.print_var(F)
        # self.print_var(M)
        # self.print_var(thrust_origin)
        # self.print_var(ref_pt)

        else:
            T = self.declare_variable('T_compute', shape=(num_nodes,))
            F = self.create_output('F_compute', shape=(num_nodes,3))
            ref_pt = self.declare_variable('reference_point',shape=(num_nodes,3), val=np.tile(reference_point,(num_nodes,1)))
            thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes, 3))
            # self.print_var(thrust_vector)
            thrust_origin = self.declare_variable('thrust_origin', shape=(num_nodes, 3))
            self.print_var(thrust_origin)
            # loop over pt set list 
            

            for i in range(num_nodes):
                # F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
                F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] #- 9 * hub_drag[i,0]
                F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1]
                F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
                

            M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
            self.register_output('M_compute', csdl.transpose(M))

            self.register_output(name='F', var=F*1)
            self.register_output(name='M', var=M*1)

            C_T = self.declare_variable('C_T_compute', shape=(num_nodes, ))
            C_Q = self.declare_variable('C_Q_compute', shape=(num_nodes, ))
            Q = self.declare_variable('Q_compute', shape=(num_nodes, ))
            T = self.declare_variable('T_compute', shape=(num_nodes, ))
            eta = self.declare_variable('eta_compute', shape=(num_nodes, ))
            FOM = self.declare_variable('FOM_compute', shape=(num_nodes, ))
            dT = self.declare_variable('_dT_compute', shape=(num_nodes, num_radial, num_tangential))
            dQ = self.declare_variable('_dQ_compute', shape=(num_nodes, num_radial, num_tangential))
            dD = self.declare_variable('_dD_compute', shape=(num_nodes, num_radial, num_tangential))
            ux = self.declare_variable('_ux_compute', shape=(num_nodes, num_radial, num_tangential))
            phi = self.declare_variable('phi_distribution', shape=(num_nodes, num_radial, num_tangential))

            self.register_output('C_T', C_T*1)
            self.register_output('C_Q', C_Q*1)
            self.register_output('Q', Q*1)
            self.register_output('T', T*1)
            self.register_output('eta', eta*1)
            self.register_output('FOM', FOM*1)
            self.register_output('_dT', dT*1)
            self.register_output('_dQ', dQ*1)
            self.register_output('_dD', dD*1)
            self.register_output('_ux', ux*1)
            self.register_output('_phi', phi*1)