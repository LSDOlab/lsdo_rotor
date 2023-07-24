from email.policy import default
from xmlrpc.client import Boolean
import numpy as np
from csdl import Model
import csdl

# from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_rotor.core.BEM_caddee.inputs.BEM_external_inputs_model import BEMExternalInputsModel
from lsdo_rotor.core.BEM_caddee.inputs.BEM_core_inputs_model import BEMCoreInputsModel
from lsdo_rotor.core.BEM_caddee.inputs.BEM_pre_process_model import BEMPreprocessModel
from lsdo_rotor.core.BEM_caddee.BEM_bracketed_search_model import BEMBracketedSearchGroup
from lsdo_rotor.core.BEM_caddee.BEM_prandtl_loss_factor_model import BEMPrandtlLossFactorModel
from lsdo_rotor.core.BEM_caddee.BEM_induced_velocity_model import BEMInducedVelocityModel
from lsdo_rotor.airfoil.BEM_airfoil_surrogate_model_group_2 import BEMAirfoilSurrogateModelGroup2

from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.BEM_caddee.functions.get_BEM_rotor_dictionary import get_BEM_rotor_dictionary

from lsdo_rotor.utils.atmosphere_model import AtmosphereModel

from lsdo_rotor.core.BEM_caddee.functions.get_bspline_mtx import get_bspline_mtx
from lsdo_rotor.core.BEM_caddee.BEM_b_spline_comp import BsplineComp
# from lsdo_rotor.core.BEM.BEM_caddee import BEMMesh


class BEMModel(ModuleCSDL):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('mesh')# , types=BEMMesh)
        self.parameters.declare('disk_prefix', default='rotor_disk', types=str)
        self.parameters.declare('disk_suffix', types=str, default=None, allow_none=True)
        self.parameters.declare('blade_prefix', default='rotor_blade', types=str)
        self.parameters.declare('num_nodes')
        self.parameters.declare('use_caddee', types=bool, default=True)

    def define(self):
        mesh = self.parameters['mesh']
        name = self.parameters['name']
        disk_prefix = self.parameters['disk_prefix']
        disk_suffix = self.parameters['disk_suffix']
        blade_prefix = self.parameters['blade_prefix']
        num_nodes = self.parameters['num_nodes']

        num_radial = mesh.parameters['num_radial']
        num_tangential = mesh.parameters['num_tangential']
        airfoil = mesh.parameters['airfoil']
        norm_hub_rad = mesh.parameters['normalized_hub_radius']
        custom_polar = mesh.parameters['airfoil_polar']
        reference_point = mesh.parameters['ref_pt']
        num_blades = mesh.parameters['num_blades']

        use_airfoil_ml = mesh.parameters['use_airfoil_ml']
        use_geometry = mesh.parameters['use_rotor_geometry']
        use_caddee = self.parameters['use_caddee']
        units = mesh.parameters['mesh_units']

        shape = (num_nodes, num_radial, num_tangential)       
        
        omega_a = self.register_module_input('rpm', shape=(num_nodes, 1), units='rpm', computed_upstream=False)

        if use_caddee is True:
            u_a = self.register_module_input(name='u', shape=(num_nodes, 1), units='m/s')
            v_a = self.register_module_input(name='v', shape=(num_nodes, 1), units='m/s') 
            w_a = self.register_module_input(name='w', shape=(num_nodes, 1), units='m/s') 
            p_a = self.register_module_input(name='p', shape=(num_nodes, 1), units='rad/s')
            q_a = self.register_module_input(name='q', shape=(num_nodes, 1), units='rad/s')
            r_a = self.register_module_input(name='r', shape=(num_nodes, 1), units='rad/s')
            theta = self.register_module_input(name='theta', shape=(num_nodes, 1))
        else:
            u_a = self.register_module_input(name='u', shape=(num_nodes, 1), units='m/s', computed_upstream=False)
            v_a = self.register_module_input(name='v', shape=(num_nodes, 1), units='m/s', computed_upstream=False) 
            w_a = self.register_module_input(name='w', shape=(num_nodes, 1), units='m/s', computed_upstream=False) 
            p_a = self.create_input(name='p', shape=(num_nodes, 1), val=0, units='rad/s')
            q_a = self.create_input(name='q', shape=(num_nodes, 1), val=0, units='rad/s')
            r_a = self.create_input(name='r', shape=(num_nodes, 1), val=0, units='rad/s')
            theta = self.create_input(name='theta', shape=(num_nodes, 1), val=0)

        rotation_matrix = self.create_output('rotation_matrix', shape=(3, 3), val=0)
        rotation_matrix[0, 0] = csdl.cos(theta)
        rotation_matrix[0, 2] = -1 * csdl.sin(theta)
        rotation_matrix[1, 1] = (theta + 1) / (theta + 1)
        rotation_matrix[2, 0] = -1 * csdl.sin(theta)
        rotation_matrix[2, 2] = -1 * csdl.cos(theta)

        if use_airfoil_ml is False:
            interp = get_surrogate_model(airfoil, custom_polar)
            rotor = get_BEM_rotor_dictionary(airfoil, interp, custom_polar)
        else:
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
        

        if use_geometry is True:
            # Chord 
            chord = self.register_module_input(f'{blade_prefix}_chord_length', shape=(num_radial, 3), promotes=True)
            chord_length = csdl.reshape(csdl.pnorm(chord, 2, axis=1), (num_radial, 1))
            if units == 'ft':
                chord_profile = self.register_output('chord_profile', chord_length * 0.3048)
            else:
                chord_profile = self.register_output('chord_profile', chord_length)
                
            # Twist
            # rel_v_dist_array = self.register_module_input(f'{blade_prefix}_twist', shape=(num_radial, 3), promotes=True)
            # rel_v_dist = csdl.reshape(csdl.sum(rel_v_dist_array, axes=(1, )), (num_radial, 1))
            # twist_profile = self.register_output('twist_profile', csdl.arcsin(rel_v_dist/chord_length))#  + np.deg2rad(5))
            # Twist
            num_cp = mesh.parameters['num_cp']
            order = mesh.parameters['b_spline_order']
            twist_cp = self.register_module_input(name=f'{blade_prefix}_twist_cp', shape=(num_cp,), units='rad', promotes=True) 
            self.print_var(twist_cp)
            pitch_A = get_bspline_mtx(num_cp, num_radial, order=order)
            comp = csdl.custom(twist_cp,op=BsplineComp(
                num_pt=num_radial,
                num_cp=num_cp,
                in_name=f'{blade_prefix}_twist_cp',
                jac=pitch_A,
                out_name='twist_profile',
            ))
            self.register_output('twist_profile', comp)
        
            # Thrust vector and origin
            if units == 'ft':
                if disk_suffix:
                    in_plane_y = self.register_module_input(f'{disk_prefix}_in_plane_1_{disk_suffix}', shape=(3, ), promotes=True) * 0.3048
                    in_plane_x = self.register_module_input(f'{disk_prefix}_in_plane_2_{disk_suffix}', shape=(3, ), promotes=True) * 0.3048
                    to = self.register_module_input(f'{disk_prefix}_origin', shape=(3, ), promotes=True) * 0.3048
                else:
                    in_plane_y = self.register_module_input(f'{disk_prefix}_in_plane_1', shape=(3, ), promotes=True) * 0.3048
                    in_plane_x = self.register_module_input(f'{disk_prefix}_in_plane_2', shape=(3, ), promotes=True) * 0.3048
                    to = self.register_module_input(f'{disk_prefix}_origin', shape=(3, ), promotes=True) * 0.3048
            else:
                if disk_suffix:
                    in_plane_y = self.register_module_input(f'{disk_prefix}_in_plane_1_{disk_suffix}', shape=(3, ), promotes=True) 
                    in_plane_x = self.register_module_input(f'{disk_prefix}_in_plane_2_{disk_suffix}', shape=(3, ), promotes=True) 
                    to = self.register_module_input(f'{disk_prefix}_origin', shape=(3, ), promotes=True) * 0.3048

                else: 
                    in_plane_y = self.register_module_input(f'{disk_prefix}_in_plane_1', shape=(3, ), promotes=True)
                    in_plane_x = self.register_module_input(f'{disk_prefix}_in_plane_2', shape=(3, ), promotes=True)
                    to = self.register_module_input(f'{disk_prefix}_origin', shape=(3, ), promotes=True)
                        
            R = csdl.pnorm(in_plane_y, 2) / 2
            self.register_module_output('propeller_radius', R)

            tv_raw = csdl.cross(in_plane_x, in_plane_y, axis=0)
            # tv = tv_raw / csdl.expand(csdl.pnorm(tv_raw), (3, ))
            tv = csdl.matvec(rotation_matrix, tv_raw / csdl.expand(csdl.pnorm(tv_raw), (3, )))
            # TODO: This assumes a fixed thrust vector and doesn't take into account actuations
            self.register_module_output('thrust_vector', csdl.expand(tv, (num_nodes, 3), 'j->ij'))
            self.register_module_output('thrust_origin', csdl.expand(to, (num_nodes, 3), 'j->ij'))

        else:
            pitch_b_spline = mesh.parameters['twist_b_spline_rep']
            if pitch_b_spline is True:
                num_cp = mesh.parameters['num_cp']
                order = mesh.parameters['b_spline_order']
                twist_cp = self.register_module_input(name='twist_cp', shape=(num_cp,), units='rad', computed_upstream=False)
                pitch_A = get_bspline_mtx(num_cp, num_radial, order=order)
                comp = csdl.custom(twist_cp,op=BsplineComp(
                    num_pt=num_radial,
                    num_cp=num_cp,
                    in_name='twist_cp',
                    jac=pitch_A,
                    out_name='twist_profile',
                ))
                self.register_output('twist_profile', comp)
            
            else:
                self.register_module_input('chord_profile', shape=(num_radial, ), computed_upstream=False)
            
            chord_b_spline = mesh.parameters['chord_b_spline_rep']
            if chord_b_spline is True:
                num_cp = mesh.parameters['num_cp']
                order = mesh.parameters['b_spline_order']
                chord_cp = self.register_module_input(name='chord_cp', shape=(num_cp,), units='rad', computed_upstream=False)
                chord_A = get_bspline_mtx(num_cp, num_radial, order=order)
                comp_chord = csdl.custom(chord_cp,op=BsplineComp(
                    num_pt=num_radial,
                    num_cp=num_cp,
                    in_name='chord_cp',
                    jac=chord_A,
                    out_name='chord_profile',
                ))
                self.register_output('chord_profile', comp_chord)

            else:
                self.register_module_input('twist_profile', shape=(num_radial, ), computed_upstream=False)
            
            
            
            
            self.register_module_input('propeller_radius', shape=(1, ), computed_upstream=False)
            self.register_module_input('thrust_vector', shape=(num_nodes, 3), computed_upstream=False)
            self.register_module_input('thrust_origin', shape=(num_nodes, 3), computed_upstream=False)
        
        
        # prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')
        # pitch_b_spline = mesh.parameters['twist_b_spline_rep']
        # chord_b_spline = mesh.parameters['chord_b_spline_rep']
        # order = mesh.parameters['b_spline_order']
        # num_cp = mesh.parameters['num_cp']
        # if pitch_b_spline == True:
        #     twist_cp = self.declare_variable(name='twist_cp', shape=(num_cp,), units='rad', val=np.linspace(50, 10, num_cp) *np.pi/180)
            
        #     pitch_A = get_bspline_mtx(num_cp, num_radial, order=order)
        #     comp = csdl.custom(twist_cp,op=BsplineComp(
        #         num_pt=num_radial,
        #         num_cp=num_cp,
        #         in_name='twist_cp',
        #         jac=pitch_A,
        #         out_name='twist_profile',
        #     ))
        #     self.register_output('twist_profile', comp)
        # else:
        #     pass

        # if chord_b_spline == True:
        #     chord_cp = self.declare_variable(name='chord_cp', shape=(num_cp,), units='rad', val=np.linspace(0.3, 0.1, num_cp))
        #     chord_A = get_bspline_mtx(num_cp, num_radial, order=order)
        #     comp_chord = csdl.custom(chord_cp,op=BsplineComp(
        #         num_pt=num_radial,
        #         num_cp=num_cp,
        #         in_name='chord_cp',
        #         jac=chord_A,
        #         out_name='chord_profile',
        #     ))
        #     self.register_output('chord_profile', comp_chord)
        # else:
        #     pass

        
        # self.print_var(twist_profile * 180/np.pi)

        # twist_profile = self.create_input('twist_profile', val=np.linspace(np.deg2rad(15), np.deg2rad(5), num_radial))
        

        # self.print_var(tv)

        self.add(BEMExternalInputsModel(
            shape=shape,
            hub_radius_percent=norm_hub_rad,
        ), name='BEM_external_inputs_model')

        self.add(BEMCoreInputsModel(
            shape=shape,
        ),name='BEM_core_inputs_model')

        self.add(BEMPreprocessModel(
            shape=shape,
            num_blades=num_blades,
        ),name='BEM_pre_process_model')

        self.add(AtmosphereModel(
            shape=(num_nodes, 1),
        ),name='atmosphere_model')
    
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


        self.add(BEMBracketedSearchGroup(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

        # phi = self.declare_variable('phi_distribution', shape=shape)
        # twist = self.declare_variable('_pitch', shape=shape)        
        # alpha = twist - phi
        # self.register_output('AoA', alpha)

        # if not rotor['custom_polar']:
        #     airfoil_model_output_2 = csdl.custom(Re, alpha, chord, op= BEMAirfoilSurrogateModelGroup2(
        #         rotor=rotor,
        #         shape=shape,
        #     ))
        # else:
        #     airfoil_model_output_2 = csdl.custom(alpha, op= BEMAirfoilSurrogateModelGroup2(
        #         rotor=rotor,
        #         shape=shape,
        #     ))
        # self.register_output('Cl_2', airfoil_model_output_2[0])
        # self.register_output('Cd_2', airfoil_model_output_2[1])

        self.add(BEMPrandtlLossFactorModel(
            shape=shape,
            num_blades=num_blades,
        ), name = 'prandtl_loss_factor_model')#, promotes = ['*'])

        self.add(BEMInducedVelocityModel(
            shape=shape,
            num_blades=num_blades,
        ), name = 'induced_velocity_model')#, promotes = ['*'])

        # Post-Processing
        T = self.declare_variable('T', shape=(num_nodes,))
        F = self.create_output('F', shape=(num_nodes,3))
        ref_pt = self.declare_variable('reference_point',shape=(num_nodes,3), val=np.tile(reference_point,(num_nodes,1)))
        thrust_vector = self.register_module_input('thrust_vector', shape=(num_nodes, 3))
        # self.print_var(thrust_vector)
        thrust_origin = self.register_module_input('thrust_origin', shape=(num_nodes, 3))
        # loop over pt set list 
        

        for i in range(num_nodes):
            # F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
            F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] #- 9 * hub_drag[i,0]
            F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1]
            F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
            

        M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
        self.register_module_output('M', csdl.transpose(M))

        # self.print_var(F)
        # self.print_var(M)
        # self.print_var(thrust_origin)
        # self.print_var(ref_pt)