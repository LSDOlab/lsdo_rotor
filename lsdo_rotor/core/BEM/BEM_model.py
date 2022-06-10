import numpy as np
from csdl import Model
import csdl

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.inputs.external_inputs_model import ExternalInputsModel
from lsdo_rotor.inputs.core_inputs_model import CoreInputsModel
from lsdo_rotor.inputs.preprocess_model import PreprocessModel
from lsdo_rotor.core.atmosphere_model import AtmosphereModel
from lsdo_rotor.airfoil.airfoil_surrogate_model_group import AirfoilSurrogateModelGroup
from lsdo_rotor.airfoil.airfoil_surrogate_model_group_2 import AirfoilSurrogateModelGroup2
from lsdo_rotor.core.phi_bracketed_search_group import PhiBracketedSearchGroup
from lsdo_rotor.core.induced_velocity_model import InducedVelocityModel
from lsdo_rotor.core.quartic_coeffs_group import QuarticCoeffsGroup
from lsdo_rotor.core.quartic_solver_group import QuarticSolverGroup
from lsdo_rotor.core.prandtl_loss_factor_group import PrandtlLossFactorGroup

class RotorModel(Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')

        self.parameters.declare('rotor', types=RotorParameters)
        self.parameters.declare('shape', types=tuple)

        self.parameters.declare('thrust_vector', types=np.ndarray)
        self.parameters.declare('thrust_origin', types=np.ndarray)
        self.parameters.declare('ref_pt', types=np.ndarray)

    def define(self):
        name = self.parameters['name']

        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        
        thrust_vector = self.parameters['thrust_vector']
        thrust_origin = self.parameters['thrust_origin']
        ref_pt = self.parameters['ref_pt']

        num_nodes = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]
        
        shape = (num_nodes, num_radial, num_tangential)       
        
        # Inputs constant across conditions (segments)
        prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')

        # Inputs changing across conditions (segments)
        omega = self.declare_variable('omega', shape=(num_nodes, 1), units='rpm')

        u = self.declare_variable(name='u',
                                  shape=(num_nodes, 1), units='m/s', val=1)
        v = self.declare_variable(name='v',
                                  shape=(num_nodes, 1), units='m/s', val=0)
        w = self.declare_variable(name='w',
                                  shape=(num_nodes, 1), units='m/s', val=0)

        p = self.declare_variable(name='p',
                                  shape=(num_nodes, 1), units='rad/s', val=0)
        q = self.declare_variable(name='q',
                                  shape=(num_nodes, 1), units='rad/s', val=0)
        r = self.declare_variable(name='r',
                                  shape=(num_nodes, 1), units='rad/s', val=0)

        Phi = self.declare_variable(name='Phi',
                                    shape=(num_nodes, 1), units='rad', val=0)
        Theta = self.declare_variable(name='Theta',
                                      shape=(num_nodes, 1), units='rad', val=0)
        Psi = self.declare_variable(name='Psi',
                                    shape=(num_nodes, 1), units='rad', val=0)

        x = self.declare_variable(name='x',
                                  shape=(num_nodes, 1), units='m', val=0)
        y = self.declare_variable(name='y',
                                  shape=(num_nodes, 1), units='m', val=0)
        z = self.declare_variable(name='z',
                                  shape=(num_nodes, 1), units='m', val=0)



        self.add(ExternalInputsModel(
            shape=shape,
        ), name = 'external_inputs_model')#, promotes = ['*'])
    
        self.add(CoreInputsModel(
            shape=shape,
        ), name = 'core_inputs_model')#, promotes=['*'])

        self.add(PreprocessModel(
            rotor=rotor,
            shape=shape,
        ), name = 'preprocess_model')#, promotes = ['*'])

        self.add(AtmosphereModel(
            rotor=rotor,
            shape=shape,
            mode  = mode,
        ), name = 'atmosphere_model')#, promotes = ['*']

        self.add(PhiBracketedSearchGroup(
            rotor=rotor,
            shape=shape,
            mode = mode,
        ), name = 'phi_bracketed_search_group')#, promotes = ['*'])
        # PhiBracketedSearchGroup(
        #     rotor=rotor,
        #     shape=shape,
        #     mode = mode,
        # ).visualize_sparsity(recursive=True)

        phi = self.declare_variable('phi_distribution', shape=shape)
        twist_dist = self.declare_variable('_pitch', shape=shape)
        chord_dist = self.declare_variable('_chord',shape=shape)
        alpha_dist = twist_dist - phi
        self.register_output('AoA', alpha_dist)

        Re_dist = self.declare_variable('Re', shape=shape)
        # self.register_output('Re_test',Re *1)

        airfoil_model_output_2 = csdl.custom(Re_dist,alpha_dist,chord_dist, op= AirfoilSurrogateModelGroup2(
            rotor=rotor,
            shape=shape,
        ))
        self.register_output('Cl_2',airfoil_model_output_2[0])
        self.register_output('Cd_2',airfoil_model_output_2[1])

        # self.add(AirfoilSurrogateModelGroup(
        #     rotor=rotor,
        #     shape=shape,
        # ), name = 'airfoil_surrogate_model_group',)# promotes = ['*'])


        self.add(PrandtlLossFactorGroup(
            rotor=rotor,
            shape=shape,
        ), name = 'prandtl_loss_factor_group')#, promotes = ['*'])

        self.add(InducedVelocityModel(
            rotor=rotor,
            mode  = mode,
            shape=shape,
        ), name = 'induced_velocity_model')#, promotes = ['*'])

    



            
            
