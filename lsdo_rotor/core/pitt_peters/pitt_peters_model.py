import numpy as np
from csdl import Model
import csdl

from lsdo_rotor.core.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.pitt_peters.functions.get_pitt_peters_rotor_dictionary import get_pitt_peters_rotor_dictionary


from lsdo_rotor.core.pitt_peters.inputs.pitt_peters_external_inputs_model import PittPetersExternalInputsModel 
from lsdo_rotor.core.pitt_peters.inputs.pitt_peters_core_inputs_model import PittPetersCoreInputsModel
from lsdo_rotor.core.pitt_peters.inputs.pitt_peters_pre_process_model import PittPetersPreprocessModel
from lsdo_rotor.core.pitt_peters.pitt_peters_custom_implicit_operation import PittPetersCustomImplicitOperation
from lsdo_rotor.core.pitt_peters.pitt_peters_post_process_model import PittPetersPostProcessModel
from lsdo_rotor.core.BEM.functions.get_bspline_mtx import get_bspline_mtx
from lsdo_rotor.core.BEM.BEM_b_spline_comp import BsplineComp
from lsdo_rotor.utils.atmosphere_model import AtmosphereModel

class PittPetersModel(csdl.Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('pitt_peters_parameters')
        self.parameters.declare('num_nodes')
        self.parameters.declare('operation')
        self.parameters.declare('stability_flag', types=bool, default=False)
        self.parameters.declare('rotation_direction', values=['cw', 'ccw', 'ignore'], allow_none=False)


    def define(self):
        pitt_peters_parameters = self.parameters['pitt_peters_parameters']
        num_nodes = self.parameters['num_nodes']
        stability_flag = self.parameters['stability_flag']
        rotation_direction = self.parameters['rotation_direction']
        operation = self.parameters['operation']
        arguments = operation.arguments
        name = operation.name

        num_radial = pitt_peters_parameters.parameters['num_radial']
        num_tangential = pitt_peters_parameters.parameters['num_tangential']
        airfoil = pitt_peters_parameters.parameters['airfoil']
        norm_hub_rad = pitt_peters_parameters.parameters['normalized_hub_radius']
        custom_polar = pitt_peters_parameters.parameters['airfoil_polar']
        reference_point = pitt_peters_parameters.parameters['ref_pt']
        num_blades = pitt_peters_parameters.parameters['num_blades']

        use_airfoil_ml = pitt_peters_parameters.parameters['use_airfoil_ml']
        use_custom_airfoil_ml = pitt_peters_parameters.parameters['use_custom_airfoil_ml']
        units = pitt_peters_parameters.parameters['mesh_units']

        shape = (num_nodes, num_radial, num_tangential)      
        
        omega_a = self.declare_variable('rpm', shape=(num_nodes, 1), units='rpm')

        u_a = self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s')
        v_a = self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s') 
        w_a = self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s') 
        p_a = self.declare_variable(name='p', shape=(num_nodes, 1), units='rad/s')
        q_a = self.declare_variable(name='q', shape=(num_nodes, 1), units='rad/s')
        r_a = self.declare_variable(name='r', shape=(num_nodes, 1), units='rad/s')
        phi = self.declare_variable(name='phi', shape=(num_nodes, 1), val=0)
        theta = self.declare_variable(name='theta', shape=(num_nodes, 1), val=0)
        psi = self.declare_variable(name='psi', shape=(num_nodes, 1), val=0)

        rotation_matrix = self.create_output('rotation_matrix', shape=(num_nodes, 3, 3), val=0)
        for i in range(num_nodes):

            T_theta = self.create_output(f'T_theta_{i}', shape=(3, 3), val=0)
            T_theta[0, 0] = csdl.cos(theta[i, 0])
            T_theta[0, 2] = csdl.sin(theta[i, 0])
            T_theta[1, 1] = self.create_input(f'one_entry_theta_{i}', val=1., shape=(1, 1))
            T_theta[2, 0] = csdl.sin(theta[i, 0]) * -1
            T_theta[2, 2] = csdl.cos(theta[i, 0])

            T_psi = self.create_output(f'T_psi_{i}', shape=(3, 3), val=0)
            T_psi[0, 0] = csdl.cos(psi[i, 0])
            T_psi[0, 1] = csdl.sin(psi[i, 0]) * -1
            T_psi[1, 0] = csdl.sin(psi[i, 0])
            T_psi[1, 1] = csdl.cos(psi[i, 0])
            T_psi[2, 2] = self.create_input(f'one_entry_psi_{i}', val=1., shape=(1, 1))

            T_phi = self.create_output(f'T_phi_{i}', shape=(3, 3), val=0)
            T_phi[0, 0] = self.create_input(f'one_entry_phi_{i}', val=1., shape=(1, 1))
            T_phi[1, 1] = csdl.cos(phi[i, 0])
            T_phi[1, 2] = csdl.sin(phi[i, 0]) * -1
            T_phi[2, 1] = csdl.sin(phi[i, 0])
            T_phi[2, 2] = csdl.cos(phi[i, 0])

            T_mat = csdl.reshape(csdl.matmat(T_psi, csdl.matmat(T_theta, T_phi)), new_shape=(1, 3, 3))

            rotation_matrix[i, :, :] = T_mat

        if (use_airfoil_ml is False) and (use_custom_airfoil_ml is False):
            interp = get_surrogate_model(airfoil, custom_polar)
        
        elif (use_custom_airfoil_ml is True) and (use_airfoil_ml is False):
            raise NotImplementedError
            # X_max_numpy = np.array([90., 8e6, 0.65])
            # X_min_numpy = np.array([-90., 1e5, 0.])
            
            # X_min = self.create_input('X_min', val=np.tile(X_min_numpy.reshape(3, 1), num_nodes*num_radial*num_tangential).T)
            # X_max = self.create_input('X_max', val=np.tile(X_max_numpy.reshape(3, 1), num_nodes*num_radial*num_tangential).T)

            # from lsdo_rotor.core.airfoil.ml_trained_models.get_airfoil_model import get_airfoil_models

            # neural_nets = get_airfoil_models(airfoil=airfoil)
            # cl_model = neural_nets.Cl
            # cd_model = neural_nets.Cd
            

        elif (use_custom_airfoil_ml is False) and (use_airfoil_ml is True):
            raise NotImplementedError
            # from lsdo_airfoil.utils.load_control_points import load_control_points
            # from lsdo_airfoil.utils.get_airfoil_model import get_airfoil_models
            # from lsdo_airfoil.core.airfoil_model_csdl import X_max_numpy_poststall, X_min_numpy_poststall


            # X_min = self.create_input('X_min', val=np.tile(X_min_numpy_poststall.reshape(35, 1), num_nodes*num_radial*num_tangential).T)
            # X_max = self.create_input('X_max', val=np.tile(X_max_numpy_poststall.reshape(35, 1), num_nodes*num_radial*num_tangential).T)
            
            # control_points_numpy = np.tile(load_control_points(airfoil_name=airfoil), num_nodes*num_radial*num_tangential).T
            # control_points = self.create_input('control_points', val=control_points_numpy)
            # airfoil_models = get_airfoil_models()
            # cl_model = airfoil_models['Cl']
            # cd_model = airfoil_models['Cd']
        
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
            thrust_origin = self.register_output('thrust_origin', to * 0.3048)
            # self.print_var(to)


            if 'chord_dist' in arguments: 
                chord = self.declare_variable('chord_dist', shape=(num_radial, ))
                chord_in_m = self.register_output('chord_profile', chord * 0.3048)
                # self.print_var(chord_in_m)
                

            elif arguments['chord_cp']:
                chord_cp_shape = arguments['chord_cp'].shape
                chord_cp = self.declare_variable(name='chord_cp', shape=chord_cp_shape)
                order = pitt_peters_parameters.parameters['b_spline_order']
                num_cp = pitt_peters_parameters.parameters['num_cp']
                chord_A = get_bspline_mtx(num_cp, num_radial, order=order)
                comp_chord = csdl.custom(chord_cp,op=BsplineComp(
                    num_pt=num_radial,
                    num_cp=num_cp,
                    in_name='chord_cp',
                    jac=chord_A,
                    out_name='chord_profile_comp',
                ))
                self.register_output('chord_profile', comp_chord * 0.3048)

            else:
                raise NotImplementedError

        else:
            r = self.declare_variable('R', shape=(1, ))
            radius = self.register_output('propeller_radius', r * 1) 
            # self.print_var(radius)

            to = self.declare_variable('to', shape=(num_nodes, 3)) 
            thrust_origin = self.register_output('thrust_origin', to * 1)
            # self.print_var(to)


            if 'chord_dist' in arguments: 
                chord = self.declare_variable('chord_dist', shape=(num_radial, ))
                self.register_output('chord_profile', chord * 1)
            
            elif arguments['chord_cp']:
                chord_cp_shape = arguments['chord_cp'].shape
                chord_cp = self.declare_variable(name='chord_cp', shape=chord_cp_shape)
                order = pitt_peters_parameters.parameters['b_spline_order']
                num_cp = pitt_peters_parameters.parameters['num_cp']
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
            order = pitt_peters_parameters.parameters['b_spline_order']
            num_cp = pitt_peters_parameters.parameters['num_cp']
            pitch_A = get_bspline_mtx(num_cp, num_radial, order=order)
            comp = csdl.custom(twist_cp,op=BsplineComp(
                num_pt=num_radial,
                num_cp=num_cp,
                in_name='twist_cp',
                jac=pitch_A,
                out_name='twist_profile',
            ))
            self.register_output('twist_profile', comp)
                    
        tv = self.declare_variable('tv', shape=(num_nodes, 3)) 
        thrust_vector = self.create_output('thrust_vector', shape=(num_nodes, 3), val=0)
        for i in range(num_nodes):
            rot_mat = csdl.reshape(rotation_matrix[i, :, :], (3, 3))
            t_vec = csdl.reshape(tv[i, :], new_shape=(3, 1))
            rot_t_vec = csdl.matmat(rot_mat, t_vec)

            thrust_vector[i, :] = csdl.reshape(rot_t_vec, new_shape=(1, 3))

        self.add(PittPetersExternalInputsModel(
            shape=shape,
            hub_radius_percent=norm_hub_rad,
        ), name = 'pitt_peters_external_inputs_model')
      
        normal_inflow = self.declare_variable('normal_inflow', shape=(num_nodes,))
        in_plane_inflow = self.declare_variable('in_plane_inflow', shape=(num_nodes,))
        n = self.declare_variable('rotational_speed', shape=(num_nodes,))        
        rotor = get_pitt_peters_rotor_dictionary(airfoil, interp, normal_inflow, in_plane_inflow, n, radius, num_nodes, num_radial, num_tangential)

        self.add(PittPetersCoreInputsModel(
            shape=shape,
        ), name = 'pitt_peters_core_inputs_model')

        self.add(PittPetersPreprocessModel(
            shape=shape,
            num_blades=num_blades,
            rotation_direction=rotation_direction,
        ), name = 'pitt_peters_preprocess_model')

        # self.add(AtmosphereModel(
        #     shape=(num_nodes,),
        # ), name = 'atmosphere_model')

        chord = self.declare_variable('_chord',shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        W = (Vx**2 + Vt**2)**0.5
        rho_exp = csdl.expand(self.declare_variable('density', shape=(num_nodes,)), shape,'i->ijk')
        self.register_output('_density_expanded',rho_exp)
        rho = self.declare_variable('density', shape=(num_nodes,))
        mu = csdl.expand(self.declare_variable('dynamic_viscosity', shape=(num_nodes,)), shape, 'i->ijk')
        speed_of_sound = csdl.expand(self.declare_variable('speed_of_sound', shape=(num_nodes, )), shape, 'i->ijk')
        Re = rho_exp * W * chord / mu
        mach = W / speed_of_sound
        # self.print_var(Re)
        self.register_output('_re_pitt_peters', Re)
        self.register_output('mach_number', mach)


        twist_dist = self.declare_variable('_pitch', shape=shape)
        r = self.declare_variable('_radius', shape=shape)
        R = self.declare_variable('_rotor_radius', shape=shape)
        Omega = self.declare_variable('_angular_speed', shape=shape)
        R = self.declare_variable('_rotor_radius', shape= shape)
        dr = self.declare_variable('_dr', shape=shape)
        mu_inflow = self.declare_variable('mu', shape=(num_nodes,))
        mu_z_inflow = self.declare_variable('mu_z', shape=(num_nodes,))
        pitt_peters = csdl.custom(
            Re,
            chord,
            twist_dist,
            Vt,
            dr,
            R,
            r,
            Omega,
            rho_exp,
            rho,
            mu_inflow,
            mu_z_inflow,
            op= PittPetersCustomImplicitOperation(
                shape=shape,
                rotor=rotor,
                num_blades=num_blades,
        ))
        self.register_output('_lambda', pitt_peters)
        # self.add_objective('_lambda')
        # PittPetersCustomImplicitOperation().visualize_sparsity(recursive=True)
        # pitt_peters.visualize_sparsity(recursive=True)


        self.add(PittPetersPostProcessModel(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name = 'pitt_peters_post_process_model_2')


        # Post-Processing
        if stability_flag:
            T = self.declare_variable('T_compute', shape=(num_nodes,))
            F = self.create_output('F_compute', shape=(num_nodes,3))
            ref_pt = csdl.expand(self.declare_variable('reference_point',shape=(3,), val=0), shape=(num_nodes, 3), indices='j->ij')
            # thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes, 3))

            for i in range(num_nodes):
                F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] 
                F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1]
                F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
                

            Q = self.declare_variable('Q_compute', shape=(num_nodes, ))
            M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
            if rotation_direction == 'cw':
                self.register_output(name='M', var=M*-1 - csdl.expand(Q, shape=thrust_vector.shape, indices='i->ij') * thrust_vector)
            elif rotation_direction == 'ccw':
                self.register_output(name='M', var=M*-1 + csdl.expand(Q, shape=thrust_vector.shape, indices='i->ij') * thrust_vector)
            else:
                self.register_output(name='M', var=M*-1)

            self.register_output(name='F', var=F*1)
            # M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
            # self.register_output('M_compute', csdl.transpose(M))

            # self.register_output(name='F', var=F*1)
            # # self.register_output(name='F_perturbed', var=F[1:, :])
            # self.register_output(name='M', var=M*1)
            # # self.register_output(name='M_perturbed', var=M[1:, :])

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
        
        else:
            T = self.declare_variable('T_compute', shape=(num_nodes,))
            F = self.create_output('F_compute', shape=(num_nodes,3))
            ref_pt = self.declare_variable('reference_point',shape=(num_nodes,3), val=0 )
            # thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes, 3))

            for i in range(num_nodes):
                F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] 
                F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1]
                F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
                
            
            Q = self.declare_variable('Q_compute', shape=(num_nodes, ))
            M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
            if rotation_direction == 'cw':
                self.register_output(name='M', var=M*-1 - csdl.expand(Q, shape=thrust_vector.shape) * thrust_vector)
            elif rotation_direction == 'ccw':
                self.register_output(name='M', var=M*-1 + csdl.expand(Q, shape=thrust_vector.shape) * thrust_vector)
            else:
                self.register_output(name='M', var=M*-1)

            
            self.register_output(name='F', var=F*1)

            # M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
            # self.register_output('M_compute', csdl.transpose(M))

            # self.register_output(name='F', var=F*1)
            # self.register_output(name='M', var=M*1)

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
            # self.add_constraint('Q', lower=20, scaler=1e-2)
            self.register_output('Q', Q*1)
            self.register_output('T', T*1)
            # self.add_constraint('T', lower=10, scaler=1e-3)
            self.register_output('eta', eta*1)
            # # self.add_constraint('eta', upper=0.999)
            self.register_output('FOM', FOM*1)
            self.register_output('_dT', dT*1)
            self.register_output('_dQ', dQ*1)
            self.register_output('_dD', dD*1)
            self.register_output('_ux', ux*1)
            self.register_output('_phi', phi*1)
