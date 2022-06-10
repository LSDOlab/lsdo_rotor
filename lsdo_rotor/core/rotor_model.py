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

from lsdo_rotor.core.pitt_peters import PittPeters
from lsdo_rotor.core.pitt_peters_group import PittPetersGroup
from lsdo_rotor.core.pitt_peters_post_process_model import PittPetersPostProcessModel
from lsdo_rotor.core.pitt_peters_custom_implicit_operation import PittPetersCustomImplicitOperation

from lsdo_rotor.core.ildm_back_comp_group import ILDMBackCompGroup
from lsdo_rotor.core.ildm_airfoil_parameters import ILDMAirfoilParameters

class RotorModel(Model):

    def initialize(self):
        self.parameters.declare('rotor', types=RotorParameters)
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
            self.add(ExternalInputsModel(
                shape=shape,
                num_evaluations = num_evaluations,
                num_radial = num_radial,
                num_tangential = num_tangential,
            ), name = 'external_inputs_model')#, promotes = ['*'])
      
            self.add(CoreInputsModel(
                num_evaluations=num_evaluations,
                num_radial=num_radial,
                num_tangential=num_tangential,
            ), name = 'core_inputs_model')#, promotes=['*'])

            self.add(PreprocessModel(
                rotor=rotor,
                shape=shape,
            ), name = 'preprocess_model')#, promotes = ['*'])

            self.add(AtmosphereModel(
                rotor=rotor,
                shape=shape,
                mode  = mode,
            ), name = 'atmosphere_model')# , promotes = ['*'])

            Re_ildm = self.declare_variable('Re_ildm', shape = (shape[0],))

            ildm_parameters = csdl.custom(Re_ildm, op= ILDMAirfoilParameters(
                rotor=rotor,
                shape=shape,
            ))
            self.register_output('Cl_max_ildm',ildm_parameters[0])
            self.register_output('Cd_min_ildm',ildm_parameters[1])
            self.register_output('alpha_max_LD',ildm_parameters[2])
            # print('test')
            # exit()

            self.add(PhiBracketedSearchGroup(
                rotor=rotor,
                shape=shape,
                mode = mode,
            ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

            self.add(InducedVelocityModel(
                rotor=rotor,
                mode  = mode,
                shape=shape,
            ), name = 'induced_velocity_model')#, promotes = ['*'])

            self.add(QuarticCoeffsGroup(
                rotor=rotor,
                shape=shape,
            ), name = 'quartic_coefficient_group')#, promotes = ['*'])

            self.add(QuarticSolverGroup(
                shape=shape,
            ), name = 'quartic_solver_group')#, promotes = ['*'])

            self.add(ILDMBackCompGroup(
                shape=shape,
                rotor=rotor,
            ), name = 'ildm_back_comp_group')#, promotes = ['*'])

        
        elif mode == 2:
            self.add(ExternalInputsModel(
                shape=shape,
                # num_evaluations = num_evaluations,
                # num_radial = num_radial,
                # num_tangential = num_tangential,
            ), name = 'external_inputs_model')#, promotes = ['*'])
      
            self.add(CoreInputsModel(
                num_evaluations=num_evaluations,
                num_radial=num_radial,
                num_tangential=num_tangential,
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

        elif mode == 3:
            self.add(ExternalInputsModel(
                shape=shape,
                # num_evaluations = num_evaluations,
                # num_radial = num_radial,
                # num_tangential = num_tangential,
            ), name = 'external_inputs_model')#, promotes = ['*'])
      
            self.add(CoreInputsModel(
                num_evaluations=num_evaluations,
                num_radial=num_radial,
                num_tangential=num_tangential,
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

    
            Re_dist = self.declare_variable('_re_pitt_peters', shape=shape)
            rho_dist = self.declare_variable('_rho_pitt_peters', shape=shape)
            chord_dist = self.declare_variable('_chord',shape=shape)
            twist_dist = self.declare_variable('_pitch', shape=shape)
            r_norm = self.declare_variable('_normalized_radius', shape=shape)
            r = self.declare_variable('_radius', shape=shape)
            R = self.declare_variable('_rotor_radius', shape=shape)
            Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
            psi = self.declare_variable('_theta', shape=shape)
            Omega = self.declare_variable('_angular_speed', shape=shape)
            R = self.declare_variable('_rotor_radius', shape= shape)
            dr = self.declare_variable('_dr', shape=shape)
            pitt_peters = csdl.custom(
                Re_dist,
                chord_dist,
                twist_dist,
                Vt,
                dr,
                R,
                r,
                # psi,
                Omega,
                op= PittPetersCustomImplicitOperation(
                    shape=shape,
                    rotor=rotor,
            ))
            self.register_output('_lambda',pitt_peters)
            # self.add_objective('_lambda')
            # PittPetersCustomImplicitOperation().visualize_sparsity(recursive=True)
            # pitt_peters.visualize_sparsity(recursive=True)


            self.add(PittPetersPostProcessModel(
                rotor=rotor,
                shape=shape,
            ), name = 'pitt_peters_post_process_model')



            
            
