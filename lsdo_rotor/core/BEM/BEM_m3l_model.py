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


class BEMModel(csdl.Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('mesh')# , types=BEMMesh)
        self.parameters.declare('prefix')
        self.parameters.declare('num_nodes', default=1, types=int, allow_none=True)

    def define(self):
        mesh = self.parameters['mesh']
        name = self.parameters['name']
        prefix = self.parameters['prefix']
        num_nodes = self.parameters['num_nodes']

        num_radial = mesh.parameters['num_radial']
        num_tangential = mesh.parameters['num_tangential']
        airfoil = mesh.parameters['airfoil']
        norm_hub_rad = mesh.parameters['normalized_hub_radius']
        custom_polar = mesh.parameters['airfoil_polar']
        reference_point = mesh.parameters['ref_pt']

        num_blades = mesh.parameters['num_blades']
        
        shape = (num_nodes, num_radial, num_tangential)       
        
        interp = get_surrogate_model(airfoil, custom_polar)
        rotor = get_BEM_rotor_dictionary(airfoil, interp, custom_polar)
    
        # prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')
        pitch_b_spline = mesh.parameters['twist_b_spline_rep']
        chord_b_spline = mesh.parameters['chord_b_spline_rep']
        order = mesh.parameters['b_spline_order']
        num_cp = mesh.parameters['num_cp']
        if pitch_b_spline == True:
            twist_cp = self.declare_variable(name='twist_cp', shape=(num_cp,), units='rad', val=np.linspace(50, 10, num_cp) *np.pi/180)
            
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
            pass

        if chord_b_spline == True:
            chord_cp = self.declare_variable(name='chord_cp', shape=(num_cp,), units='rad', val=np.linspace(0.3, 0.1, num_cp))
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
            pass


        omega = self.register_module_input('rpm', shape=(num_nodes, 1), units='rpm', vectorized=True, computed_upstream=False)
        u = self.register_module_input(name='u', shape=(num_nodes, 1), units='m/s', vectorized=True)
        v = self.register_module_input(name='v', shape=(num_nodes, 1), units='m/s', vectorized=True) 
        w = self.register_module_input(name='w', shape=(num_nodes, 1), units='m/s', vectorized=True) 
        p = self.register_module_input(name='p', shape=(num_nodes, 1), units='rad/s', vectorized=True)
        q = self.register_module_input(name='q', shape=(num_nodes, 1), units='rad/s', vectorized=True)
        r = self.register_module_input(name='r', shape=(num_nodes, 1), units='rad/s', vectorized=True)
        
  

        in_plane_y = self.register_module_input(f'{prefix}_in_plane_y', shape=(3, ), promotes=True) * 0.3048
        in_plane_x = self.register_module_input(f'{prefix}_in_plane_x', shape=(3, ), promotes=True) * 0.3048
        to = self.register_module_input(f'{prefix}_origin', shape=(3, ), promotes=True) * 0.3048
        # R = self.register_module_input(f'{prefix}_radius', shape=(1, ), promotes=True)
        R = csdl.pnorm(in_plane_y, 2) / 2
        self.register_module_output('propeller_radius', R)
        # self.print_var(in_plane_y)


        tv_raw = csdl.cross(in_plane_x, in_plane_y, axis=0)
        tv = tv_raw / csdl.expand(csdl.pnorm(tv_raw), (3, ))
        
        # TODO: This assumes a fixed thrust vector and doesn't take into account actuations
        self.register_module_output('thrust_vector', csdl.expand(tv, (num_nodes, 3), 'j->ij'))
        self.register_module_output('thrust_origin', csdl.expand(to, (num_nodes, 3), 'j->ij'))

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
            shape=(num_nodes,1),
        ),name='atmosphere_model')
    
        chord = self.declare_variable('_chord',shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        W = (Vx**2 + Vt**2)**0.5
        rho = csdl.expand(self.declare_variable('density', shape=(num_nodes,)), shape,'i->ijk')
        mu = csdl.expand(self.declare_variable('dynamic_viscosity', shape=(num_nodes,)), shape, 'i->ijk')
        Re = rho * W * chord / mu
        self.register_output('Re', Re)


        self.add(BEMBracketedSearchGroup(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

        phi = self.declare_variable('phi_distribution', shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)        
        alpha = twist - phi
        self.register_output('AoA', alpha)

        if not rotor['custom_polar']:
            airfoil_model_output_2 = csdl.custom(Re, alpha, chord, op= BEMAirfoilSurrogateModelGroup2(
                rotor=rotor,
                shape=shape,
            ))
        else:
            airfoil_model_output_2 = csdl.custom(alpha, op= BEMAirfoilSurrogateModelGroup2(
                rotor=rotor,
                shape=shape,
            ))
        self.register_output('Cl_2', airfoil_model_output_2[0])
        self.register_output('Cd_2', airfoil_model_output_2[1])

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
        M = self.create_output('M', shape=(num_nodes,3))
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
            

        moments = csdl.cross(thrust_origin-ref_pt, F, axis=1)
        M[:,0] = moments[:,0] 
        M[:,1] = moments[:,1] 
        M[:,2] = moments[:,2] 

        # expansion_mat = np.zeros((num_active_nodes, num_nodes))
        # for i in range(num_active_nodes):
        #     column_index = int(active_nodes[i])
        #     expansion_mat[i, column_index] = 1
        # expansion_mat_csdl = self.create_input('expansion_mat', shape=(num_active_nodes, num_nodes), val=expansion_mat)

        # F = csdl.matmat(csdl.reshape(F_active, (3, num_active_nodes)), expansion_mat_csdl)
        # M = csdl.matmat(csdl.reshape(M_active, (3, num_active_nodes)), expansion_mat_csdl)

        # self.register_module_output('F', csdl.transpose(F))
        # self.register_module_output('M', csdl.transpose(M))

        # self.print_var(F)
        # self.print_var(M)
        # self.print_var(thrust_origin)
        # self.print_var(ref_pt)