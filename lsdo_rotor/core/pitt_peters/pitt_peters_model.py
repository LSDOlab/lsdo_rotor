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
        self.parameters.declare('mesh')
        self.parameters.declare('disk_prefix', default='rotor_disk', types=str)
        self.parameters.declare('blade_prefix', default='rotor_blade', types=str)
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('use_caddee', types=bool, default=True)


    def define(self):
        mesh = self.parameters['mesh']
        name = self.parameters['name']
        disk_prefix = self.parameters['disk_prefix']
        blade_prefix = self.parameters['blade_prefix']
        num_nodes = self.parameters['num_nodes']

        num_radial = mesh.parameters['num_radial']
        num_tangential = mesh.parameters['num_tangential']
        airfoil = mesh.parameters['airfoil']
        norm_hub_rad = mesh.parameters['normalized_hub_radius']
        custom_polar = mesh.parameters['airfoil_polar']
        reference_point = mesh.parameters['ref_pt']
        num_blades = mesh.parameters['num_blades']

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

        interp = get_surrogate_model(airfoil, custom_polar)


        if use_geometry is True:
            # Chord 
            chord = self.register_module_input(f'{blade_prefix}_chord_length', shape=(num_radial, 3), promotes=True)
            chord_length = csdl.reshape(csdl.pnorm(chord, 2, axis=1), (num_radial, 1))
            if units == 'ft':
                chord_profile = self.register_output('chord_profile', chord_length * 0.3048)
            else:
                chord_profile = self.register_output('chord_profile', chord_length)
                
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
            # rel_v_dist_array = self.register_module_input(f'{blade_prefix}_twist', shape=(num_radial, 3), promotes=True)
            # rel_v_dist = csdl.reshape(csdl.sum(rel_v_dist_array, axes=(1, )), (num_radial, 1))
            # twist_profile = self.register_output('twist_profile', csdl.arcsin(rel_v_dist/chord_length) + np.deg2rad(5))
        
            # Thrust vector and origin
            if units == 'ft':
                in_plane_y = self.register_module_input(f'{disk_prefix}_in_plane_1', shape=(3, ), promotes=True) * 0.3048
                in_plane_x = self.register_module_input(f'{disk_prefix}_in_plane_2', shape=(3, ), promotes=True) * 0.3048
                to = self.register_module_input(f'{disk_prefix}_origin', shape=(3, ), promotes=True) * 0.3048
            else:
                in_plane_y = self.register_module_input(f'{disk_prefix}_in_plane_1', shape=(3, ), promotes=True)
                in_plane_x = self.register_module_input(f'{disk_prefix}_in_plane_2', shape=(3, ), promotes=True)
                to = self.register_module_input(f'{disk_prefix}_origin', shape=(3, ), promotes=True)
                        
            R = csdl.pnorm(in_plane_y, 2) / 2
            prop_radius = self.register_module_output('propeller_radius', R)

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
            
            prop_radius = self.register_module_input('propeller_radius', shape=(1, ), computed_upstream=False)
            self.register_module_input('thrust_vector', shape=(num_nodes, 3), computed_upstream=False)
            self.register_module_input('thrust_origin', shape=(num_nodes, 3), computed_upstream=False)


        self.add(PittPetersExternalInputsModel(
            shape=shape,
        ), name = 'pitt_peters_external_inputs_model')
      
        normal_inflow = self.declare_variable('normal_inflow', shape=(num_nodes,))
        in_plane_inflow = self.declare_variable('in_plane_inflow', shape=(num_nodes,))
        n = self.declare_variable('rotational_speed', shape=(num_nodes,))        
        rotor = get_pitt_peters_rotor_dictionary(airfoil, interp, normal_inflow, in_plane_inflow, n, prop_radius, num_nodes, num_radial, num_tangential)

        self.add(PittPetersCoreInputsModel(
            shape=shape,
        ), name = 'pitt_peters_core_inputs_model')

        self.add(PittPetersPreprocessModel(
            shape=shape,
            num_blades=num_blades,
        ), name = 'pitt_peters_preprocess_model')

        self.add(AtmosphereModel(
            shape=(num_nodes,),
        ), name = 'atmosphere_model')

        chord = self.declare_variable('_chord',shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        W = (Vx**2 + Vt**2)**0.5
        rho_exp = csdl.expand(self.declare_variable('density', shape=(num_nodes,)), shape,'i->ijk')
        self.register_output('_density_expanded',rho_exp)
        rho = self.declare_variable('density', shape=(num_nodes,))
        mu = csdl.expand(self.declare_variable('dynamic_viscosity', shape=(num_nodes,)), shape, 'i->ijk')
        Re = rho_exp * W * chord / mu
        # self.print_var(Re)
        self.register_output('_re_pitt_peters',Re)


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
        self.register_output('_lambda',pitt_peters)
        # self.add_objective('_lambda')
        # PittPetersCustomImplicitOperation().visualize_sparsity(recursive=True)
        # pitt_peters.visualize_sparsity(recursive=True)


        self.add(PittPetersPostProcessModel(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name = 'pitt_peters_post_process_model_2')

        # Post-Processing
        T = self.declare_variable('T', shape=(num_nodes,))
        F = self.create_output('F', shape=(num_nodes,3))
        thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes,3))
        thrust_origin = self.declare_variable('thrust_origin', shape=(num_nodes,3))
        ref_pt = self.declare_variable('reference_point',shape=(num_nodes,3),val=np.tile(reference_point,(num_nodes,1)))
        # loop over pt set list 
        for i in range(num_nodes):
            # F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
            F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] #- 9 * hub_drag[i,0]
            F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1]
            F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
            

        M = csdl.cross(thrust_origin-ref_pt, F, axis=1)
        self.register_module_output('M', csdl.transpose(M))
