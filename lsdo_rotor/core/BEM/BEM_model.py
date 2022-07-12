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
from lsdo_rotor.airfoil.BEM_airfoil_surrogate_model_group_2 import BEMAirfoilSurrogateModelGroup2

from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.BEM.functions.get_BEM_rotor_dictionary import get_BEM_rotor_dictionary

from lsdo_atmos.atmosphere_model import AtmosphereModel

from lsdo_rotor.core.BEM.functions.get_bspline_mtx import   get_bspline_mtx
from lsdo_rotor.core.BEM.BEM_b_spline_comp import BsplineComp

class BEMModel(Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=30)
        self.parameters.declare(name='airfoil', default='NACA_4412')
        # self.parameters.declare('shape', types=tuple)

        # self.parameters.declare('thrust_vector', types=np.ndarray)
        # self.parameters.declare('thrust_origin', types=np.ndarray)
        self.parameters.declare('ref_pt', types=np.ndarray)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        airfoil = self.parameters['airfoil']
        # shape = self.parameters['shape']
        
        # thrust_vector = self.parameters['thrust_vector']
        # thrust_origin = self.parameters['thrust_origin']
        ref_pt = self.parameters['ref_pt']

        # --------------------------------------------------------------------------- #
        # BEM_pt_set_list = []

        # for i in num_pt_sets:
        #     BEM_pt_set_list.append(self.declare_variable('',shape=(num_nodes,3)))

        # thrust_vector = BEM_pt_set_list[0]
        # --------------------------------------------------------------------------- #

        # num_nodes = shape[0]
        # num_radial = shape[1]
        # num_tangential = shape[2]
        # print(num_tangential, 'num_tangential')
        # print(num_radial,'num_radial')

        num_blades = self.parameters['num_blades']
        
        shape = (num_nodes, num_radial, num_tangential)       
        
        interp = get_surrogate_model(airfoil)
        rotor = get_BEM_rotor_dictionary(airfoil,interp)

        prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')
        pitch_cp = self.declare_variable(name='pitch_cp', shape=(4,), units='rad', val=np.linspace(50,10,4)*np.pi/180)
        # self.add_design_variable('pitch_cp', lower=5*np.pi/180,upper=60*np.pi/180)
        # Inputs changing across conditions (segments)
        omega = self.declare_variable('omega', shape=(num_nodes, 1), units='rpm')

        self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s', val=1)
        self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s', val=0)
        self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s', val=0)

        self.declare_variable(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
        self.declare_variable(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
        self.declare_variable(name='r', shape=(num_nodes, 1), units='rad/s', val=0)

        self.declare_variable(name='Phi', shape=(num_nodes, 1), units='rad', val=0)
        self.declare_variable(name='Theta',shape=(num_nodes, 1), units='rad', val=0)
        self.declare_variable(name='Psi', shape=(num_nodes, 1), units='rad', val=0)

        self.declare_variable(name='x', shape=(num_nodes,  1), units='m', val=0)
        self.declare_variable(name='y', shape=(num_nodes,  1), units='m', val=0)
        self.declare_variable(name='z', shape=(num_nodes,  1), units='m', val=0)

        pitch_A = get_bspline_mtx(4, num_radial, order=4)
        comp = csdl.custom(pitch_cp,op=BsplineComp(
            num_pt=num_radial,
            num_cp=4,
            in_name='pitch_cp',
            jac=pitch_A,
            out_name='twist_profile',
        ))
        # self.print_var(comp)
        # self.create_input(name='twist_profile', shape=(num_radial,), units='rad',val=comp)
        self.register_output('twist_profile', comp)
        # self.add('pitch_bspline_comp', comp, promotes = ['*'])

        self.add(BEMExternalInputsModel(
            shape=shape,
            # thrust_vector=thrust_vector,
        ), name='BEM_external_inputs_model')#, promotes = ['*'])

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
        self.register_output('Re',Re)


        self.add(BEMBracketedSearchGroup(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

        phi = self.declare_variable('phi_distribution', shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)        
        alpha = twist - phi
        self.register_output('AoA', alpha)

        airfoil_model_output_2 = csdl.custom(Re,alpha,chord, op= BEMAirfoilSurrogateModelGroup2(
            rotor=rotor,
            shape=shape,
        ))
        self.register_output('Cl_2',airfoil_model_output_2[0])
        self.register_output('Cd_2',airfoil_model_output_2[1])

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
        n = self.declare_variable('thrust_vector', shape=(num_nodes,3))
        thrust_origin = self.declare_variable('thrust_origin', shape=(num_nodes,3))
        # loop over pt set list 
        for i in range(num_nodes):
            F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
            M[i,0] = F[i,2] * (thrust_origin[i,1] - ref_pt[1])
            M[i,1] = F[i,2] * (thrust_origin[i,0] - ref_pt[0])
            M[i,2] = F[i,0] * (thrust_origin[i,1] - ref_pt[1])


# from email.policy import default
# import numpy as np
# from csdl import Model
# import csdl

# # from lsdo_rotor.rotor_parameters import RotorParameters
# from lsdo_rotor.core.BEM.inputs.BEM_external_inputs_model import BEMExternalInputsModel
# from lsdo_rotor.core.BEM.inputs.BEM_core_inputs_model import BEMCoreInputsModel
# from lsdo_rotor.core.BEM.inputs.BEM_pre_process_model import BEMPreprocessModel
# from lsdo_rotor.core.BEM.BEM_bracketed_search_model import BEMBracketedSearchGroup
# from lsdo_rotor.core.BEM.BEM_prandtl_loss_factor_model import BEMPrandtlLossFactorModel
# from lsdo_rotor.core.BEM.BEM_induced_velocity_model import BEMInducedVelocityModel
# from lsdo_rotor.airfoil.BEM_airfoil_surrogate_model_group_2 import BEMAirfoilSurrogateModelGroup2

# from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
# from lsdo_rotor.core.BEM.functions.get_BEM_rotor_dictionary import get_BEM_rotor_dictionary

# from lsdo_atmos.atmosphere_model import AtmosphereModel

# from lsdo_utils.comps.bspline_comp import   get_bspline_mtx
# from lsdo_rotor.core.BEM.BEM_b_spline_comp import BsplineComp

# class BEMModel(Model):

#     def initialize(self):
#         self.parameters.declare(name='name', default='propulsion')
#         self.parameters.declare('num_nodes', default=1)
#         self.parameters.declare('num_radial', types=int, default=30)
#         self.parameters.declare('num_tangential', types=int, default=30)
#         self.parameters.declare(name='airfoil', default='NACA_4412')
#         # self.parameters.declare('shape', types=tuple)

#         # self.parameters.declare('thrust_vector', types=np.ndarray)
#         # self.parameters.declare('thrust_origin', types=np.ndarray)
#         # self.parameters.declare('ref_pt', types=np.ndarray)
#         self.parameters.declare('num_blades', types=int, default=3)

#     def define(self):
#         name = self.parameters['name']
#         num_nodes = self.parameters['num_nodes']
#         num_radial = self.parameters['num_radial']
#         num_tangential = self.parameters['num_tangential']
#         airfoil = self.parameters['airfoil']
#         # shape = self.parameters['shape']
        
#         # thrust_vector = self.parameters['thrust_vector']
#         # thrust_origin = self.parameters['thrust_origin']
#         # ref_pt = self.parameters['ref_pt']

#         # --------------------------------------------------------------------------- #
#         # BEM_pt_set_list = []

#         # for i in num_pt_sets:
#         #     BEM_pt_set_list.append(self.declare_variable('',shape=(num_nodes,3)))

#         # thrust_vector = BEM_pt_set_list[0]
#         # --------------------------------------------------------------------------- #

#         # num_nodes = shape[0]
#         # num_radial = shape[1]
#         # num_tangential = shape[2]
#         # print(num_tangential, 'num_tangential')
#         # print(num_radial,'num_radial')

#         num_blades = self.parameters['num_blades']
        
#         shape = (num_nodes, num_radial, num_tangential)       
        
#         interp = get_surrogate_model(airfoil)
#         rotor = get_BEM_rotor_dictionary(airfoil,interp)

#         prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')
#         pitch_cp = self.declare_variable(name='pitch_cp', shape=(4,), units='rad', val=np.linspace(50,10,4)*np.pi/180)
#         # self.add_design_variable('pitch_cp', lower=5*np.pi/180,upper=60*np.pi/180)
#         # Inputs changing across conditions (segments)
#         omega = self.declare_variable('omega', shape=(num_nodes, 1), units='rpm')
#         n = self.declare_variable('thrust_vector', shape=(num_nodes,3))
#         self.print_var(n)

#         self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s', val=1)
#         self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s', val=0)
#         self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s', val=0)

#         self.declare_variable(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
#         self.declare_variable(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
#         self.declare_variable(name='r', shape=(num_nodes, 1), units='rad/s', val=0)

#         self.declare_variable(name='Phi', shape=(num_nodes, 1), units='rad', val=0)
#         self.declare_variable(name='Theta',shape=(num_nodes, 1), units='rad', val=0)
#         self.declare_variable(name='Psi', shape=(num_nodes, 1), units='rad', val=0)

#         self.declare_variable(name='x', shape=(num_nodes,  1), units='m', val=0)
#         self.declare_variable(name='y', shape=(num_nodes,  1), units='m', val=0)
#         self.declare_variable(name='z', shape=(num_nodes,  1), units='m', val=0)

#         pitch_A = get_bspline_mtx(4, num_radial, order=4)
#         comp = csdl.custom(pitch_cp,op=BsplineComp(
#             num_pt=num_radial,
#             num_cp=4,
#             in_name='pitch_cp',
#             jac=pitch_A,
#             out_name='twist_profile',
#         ))
#         # self.print_var(comp)
#         # self.create_input(name='twist_profile', shape=(num_radial,), units='rad',val=comp)
#         self.register_output('twist_profile', comp)
#         # self.add('pitch_bspline_comp', comp, promotes = ['*'])

#         self.add(BEMExternalInputsModel(
#             shape=shape,
#             # thrust_vector=thrust_vector,
#         ), name='BEM_external_inputs_model')#, promotes = ['*'])

#         self.add(BEMCoreInputsModel(
#             shape=shape,
#         ),name='BEM_core_inputs_model')

#         self.add(BEMPreprocessModel(
#             shape=shape,
#             num_blades=num_blades,
#         ),name='BEM_pre_process_model')

#         self.add(AtmosphereModel(
#             shape=(num_nodes,1),
#         ),name='atmosphere_model')
    
#         chord = self.declare_variable('_chord',shape=shape)
#         Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
#         Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
#         W = (Vx**2 + Vt**2)**0.5
#         rho = csdl.expand(self.declare_variable('density', shape=(num_nodes,)), shape,'i->ijk')
#         mu = csdl.expand(self.declare_variable('dynamic_viscosity', shape=(num_nodes,)), shape, 'i->ijk')
#         Re = rho * W * chord / mu
#         self.register_output('Re',Re)


#         self.add(BEMBracketedSearchGroup(
#             rotor=rotor,
#             shape=shape,
#             num_blades=num_blades,
#         ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

#         phi = self.declare_variable('phi_distribution', shape=shape)
#         twist = self.declare_variable('_pitch', shape=shape)        
#         alpha = twist - phi
#         self.register_output('AoA', alpha)

#         airfoil_model_output_2 = csdl.custom(Re,alpha,chord, op= BEMAirfoilSurrogateModelGroup2(
#             rotor=rotor,
#             shape=shape,
#         ))
#         self.register_output('Cl_2',airfoil_model_output_2[0])
#         self.register_output('Cd_2',airfoil_model_output_2[1])

#         self.add(BEMPrandtlLossFactorModel(
#             shape=shape,
#             num_blades=num_blades,
#         ), name = 'prandtl_loss_factor_model')#, promotes = ['*'])

#         self.add(BEMInducedVelocityModel(
#             shape=shape,
#             num_blades=num_blades,
#         ), name = 'induced_velocity_model')#, promotes = ['*'])

#         # Post-Processing
#         T = self.declare_variable('T', shape=(num_nodes,))
        
#         for vector_origin_pair_tuple in bem_mesh_list:
#             origin = vector_origin_pair_tuple[0]
#             vector = vector_origin_pair_tuple[1]
            
#             F = self.create_output(f'{vector.name}_F', shape=(num_nodes,3))
#             M = self.create_output(f'{vector.name}_M', shape=(num_nodes,3))
#             # n = self.declare_variable('thrust_vector', shape=(num_nodes,3))
#             thrust_origin = self.declare_variable('thrust_origin', shape=(num_nodes,3))
#             ref_pt = self.declare_variable(name='ref_pt', shape=(num_nodes,3), units='m')
#             # loop over pt set list 
            
#             for i in range(num_nodes):
#                 F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
#                 M[i,0] = F[i,2] * (thrust_origin[i,1] - ref_pt[i,1])
#                 M[i,1] = F[i,2] * (thrust_origin[i,0] - ref_pt[i,0])
#                 M[i,2] = F[i,0] * (thrust_origin[i,1] - ref_pt[i,1])
    



            
            
