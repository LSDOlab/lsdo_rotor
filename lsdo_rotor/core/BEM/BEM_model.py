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

class BEMModel(Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=30)
        self.parameters.declare(name='airfoil', default='NACA_4412')
        # self.parameters.declare('shape', types=tuple)

        self.parameters.declare('thrust_vector', types=np.ndarray)
        self.parameters.declare('thrust_origin', types=np.ndarray)
        self.parameters.declare('ref_pt', types=np.ndarray)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        airfoil = self.parameters['airfoil']
        # shape = self.parameters['shape']
        
        thrust_vector = self.parameters['thrust_vector']
        thrust_origin = self.parameters['thrust_origin']
        ref_pt = self.parameters['ref_pt']

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

        self.declare_variable(name='x', shape=(num_nodes, ), units='m', val=0)
        self.declare_variable(name='y', shape=(num_nodes, ), units='m', val=0)
        self.declare_variable(name='z', shape=(num_nodes, ), units='m', val=0)

        self.add(BEMExternalInputsModel(
            shape=shape,
            thrust_vector=thrust_vector,
        ), name='BEM_external_inputs_model')#, promotes = ['*'])

        self.add(BEMCoreInputsModel(
            shape=shape,
        ),name='BEM_core_inputs_model')

        self.add(BEMPreprocessModel(
            shape=shape,
            num_blades=num_blades,
        ),name='BEM_pre_process_model')

        self.add(AtmosphereModel(
            shape=(num_nodes,),
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
        n = self.declare_variable('normal_vector', shape=(1,3))
        for i in range(num_nodes):
            F[i,:] = csdl.expand(T[i],(1,3)) * n
            M[i,0] = F[i,2] * (thrust_origin[1] - ref_pt[1])
            M[i,1] = F[i,2] * (thrust_origin[0] - ref_pt[0])
            M[i,2] = F[i,0] * (thrust_origin[1] - ref_pt[1])



    



            
            
