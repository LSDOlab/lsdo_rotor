import numpy as np
from csdl import Model
import csdl

from rotor_parameters import RotorParameters
from inputs.external_inputs_group import ExternalInputsGroup
from inputs.core_inputs_group import CoreInputsGroup
from inputs.preprocess_group import PreprocessGroup
from core.atmosphere_group import AtmosphereGroup
from airfoil.airfoil_surrogate_model_group import AirfoilSurrogateModelGroup
from core.phi_bracketed_search_group import PhiBracketedSearchGroup
from core.induced_velocity_group import InducedVelocityGroup
from core.quartic_coeffs_group import QuarticCoeffsGroup
from core.quartic_solver_group import QuarticSolverGroup
from core.ildm_back_comp_group import ILDMBackCompGroup
from core.prandtl_loss_factor_group import PrandtlLossFactorGroup

class BEMGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor')#, types=RotorParameters)
        self.parameters.declare('mode', types=int)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)

    def define(self):
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        mode = self.parameters['mode']
        rotor = self.parameters['rotor']

        shape = (num_evaluations, num_radial, num_tangential)        

        if mode == 1:
            self.add(ExternalInputsGroup(
                shape = shape,
                num_evaluations = num_evaluations,
                num_radial = num_radial,
                num_tangential = num_tangential,
            ), name = 'external_inputs_group')#, promotes = ['*'])
      
            self.add(CoreInputsGroup(
                num_evaluations=num_evaluations,
                num_radial=num_radial,
                num_tangential=num_tangential,
            ), name = 'core_inputs_group')#, promotes=['*'])

            self.add(PreprocessGroup(
                rotor = rotor,
                shape = shape,
            ), name = 'preprocess_group')#, promotes = ['*'])

            self.add(AtmosphereGroup(
                rotor = rotor,
                shape = shape,
                mode  = mode,
            ), name = 'atmosphere_group')# , promotes = ['*'])

            self.add(PhiBracketedSearchGroup(
                rotor = rotor,
                shape = shape,
                mode = mode,
            ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

            self.add(InducedVelocityGroup(
                rotor = rotor,
                mode  = mode,
                shape = shape,
            ), name = 'induced_velocity_group')#, promotes = ['*'])

            self.add(QuarticCoeffsGroup(
                rotor=rotor,
                shape=shape,
            ), name = 'quartic_coefficient_group')#, promotes = ['*'])

            self.add(QuarticSolverGroup(
                shape = shape,
            ), name = 'quartic_solver_group')#, promotes = ['*'])

            self.add(ILDMBackCompGroup(
                shape = shape,
                rotor = rotor,
            ), name = 'ildm_back_comp_group')#, promotes = ['*'])

        
        elif mode == 2:
            self.add(ExternalInputsGroup(
                shape = shape,
                num_evaluations = num_evaluations,
                num_radial = num_radial,
                num_tangential = num_tangential,
            ), name = 'external_inputs_group')#, promotes = ['*'])
      
            self.add(CoreInputsGroup(
                num_evaluations=num_evaluations,
                num_radial=num_radial,
                num_tangential=num_tangential,
            ), name = 'core_inputs_group')#, promotes=['*'])

            self.add(PreprocessGroup(
                rotor = rotor,
                shape = shape,
            ), name = 'preprocess_group')#, promotes = ['*'])

            self.add(AtmosphereGroup(
                rotor = rotor,
                shape = shape,
                mode  = mode,
            ), name = 'atmosphere_group')#, promotes = ['*']

            self.add(PhiBracketedSearchGroup(
                rotor = rotor,
                shape = shape,
                mode = mode,
            ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

            # phi = self.declare_variable('phi_distribution', shape = shape)
            # pitch = self.declare_variable('pitch_distribution', shape = shape)
            # chord = self.declare_variable('chord_distribution',shape=shape)
            # alpha = pitch - phi
            # self.register_output('AoA', alpha)

            # Re = self.declare_variable('Re', shape = shape)
            # self.register_output('Re_test',Re *1)

            # airfoil_model_output = csdl.custom(Re,alpha,chord, op= AirfoilSurrogateModelGroup2(
            #     rotor = rotor,
            #     shape = shape,
            # ))
            # self.register_output('Cl_2',airfoil_model_output[0])
            # self.register_output('Cd_2',airfoil_model_output[1])

            # self.add(AirfoilSurrogateModelGroup(
            #     rotor = rotor,
            #     shape = shape,
            # ), name = 'airfoil_surrogate_model_group',)# promotes = ['*'])


            # self.add(PrandtlLossFactorGroup(
            #     rotor = rotor,
            #     shape = shape,
            # ), name = 'prandtl_loss_factor_group')#, promotes = ['*'])

            self.add(InducedVelocityGroup(
                rotor = rotor,
                mode  = mode,
                shape = shape,
            ), name = 'induced_velocity_group')#, promotes = ['*'])

            
            
