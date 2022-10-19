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
        
        self.parameters.declare('thrust_vector',types=np.ndarray)
        self.parameters.declare('thrust_origin',types=np.ndarray)
        self.parameters.declare('ref_pt', types=np.ndarray)
        self.parameters.declare('num_blades', types=int)
        self.parameters.declare('chord_b_spline',types=Boolean, default=False)
        self.parameters.declare('pitch_b_spline',types=Boolean,default=False)
        self.parameters.declare('normalized_hub_radius',default=0.2)

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        airfoil = self.parameters['airfoil']
        norm_hub_rad = self.parameters['normalized_hub_radius']
        
        reference_point = self.parameters['ref_pt']
        t_v = self.parameters['thrust_vector']
        print(t_v)
        t_o = self.parameters['thrust_origin']
        print(t_o)
        if t_v.shape[0] == 1:
            n = 1
            self.create_input('thrust_vector', shape=(num_nodes,3), val=np.tile(t_v,(num_nodes,1)))
        elif len(t_v.shape) > 2:
            raise ValueError('Thrust vector cannot be a tensor; It must be at most a matrix of size (num_nodes,3')
        elif t_v.shape[1] != 3:
            raise ValueError('Thrust vector matrix must have shape (num_nodes,3')
        else:
            n = t_v.shape[0]
            print(n)
            print(t_v.shape)
            if n != num_nodes:
                raise ValueError('If number of thrust vectors is greater than 1, it must be equal to num_nodes')
            else:
                self.create_input('thrust_vector',shape=(num_nodes,3),val=t_v)


        if t_o.shape[0] == 1:
            m = 1
            self.create_input('thrust_origin', shape=(num_nodes,3), val=np.tile(t_o,(num_nodes,1)))
        elif len(t_o.shape) > 2:
            raise ValueError('Thrust origin cannot be a tensor; It must be at most a matrix of size (num_nodes,3')
        elif t_o.shape[1] != 3:
            raise ValueError('Thrust origin matrix must have shape (num_nodes,3')
        else:
            m = t_o.shape[0]
            if m != num_nodes:
                raise ValueError('If number of thrust origin vector is greater than 1, it must be equal to num_nodes')
            else:
                self.create_input('thrust_origin',shape=(num_nodes,3),val=t_o)


        num_blades = self.parameters['num_blades']
        
        shape = (num_nodes, num_radial, num_tangential)       
        
        interp = get_surrogate_model(airfoil)
        rotor = get_BEM_rotor_dictionary(airfoil,interp)

        # prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')
        pitch_b_spline = self.parameters['pitch_b_spline']
        chord_b_spline = self.parameters['chord_b_spline']

        if pitch_b_spline == True:
            pitch_cp = self.declare_variable(name='pitch_cp', shape=(4,), units='rad', val=np.linspace(50,10,4)*np.pi/180)
            
            pitch_A = get_bspline_mtx(4, num_radial, order=4)
            comp = csdl.custom(pitch_cp,op=BsplineComp(
                num_pt=num_radial,
                num_cp=4,
                in_name='pitch_cp',
                jac=pitch_A,
                out_name='twist_profile',
            ))
            self.register_output('twist_profile', comp)
        else:
            pass

        if chord_b_spline == True:
            chord_cp = self.declare_variable(name='chord_cp', shape=(2,), units='rad', val=np.array([0.3,0.1]))
            chord_A = get_bspline_mtx(2, num_radial, order=2)
            comp_chord = csdl.custom(chord_cp,op=BsplineComp(
                num_pt=num_radial,
                num_cp=2,
                in_name='chord_cp',
                jac=chord_A,
                out_name='chord_profile',
            ))
            self.register_output('chord_profile', comp_chord)
        else:
            pass

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
        thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes,3))
        thrust_origin = self.declare_variable('thrust_origin', shape=(num_nodes,3))
        ref_pt = self.declare_variable('reference_point',shape=(num_nodes,3),val=np.tile(reference_point,(num_nodes,1)))
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

#