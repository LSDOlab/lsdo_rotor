import omtools.api as ot

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.core_analysis_group import CoreAnalysisGroup
from lsdo_rotor.core.ideal_blade_group import IdealBladeGroup
from lsdo_rotor.core.postprocess_group import PostProcessGroup
from lsdo_rotor.core.bc_phi_group import BCPhiGroup
from lsdo_rotor.core.inputs_group import InputsGroup
from lsdo_rotor.core.preprocess_group import PreprocessGroup
from lsdo_rotor.core.bemt_implicit_component import BEMTImplicitComponent
from lsdo_rotor.core.induced_velocity_group import InducedVelocityGroup
from lsdo_rotor.core.loss_group import LossGroup
from lsdo_rotor.core.viterna_explicit_component import ViternaExplicitComponent
from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup
from lsdo_rotor.core.smoothing_explicit_component import SmoothingExplicitComponent
# from lsdo_rotor.core.bc_implicit_component import BCImplicitComponent


class IdealizedBEMTGroup(ot.Group):

    def initialize(self):
        self.options.declare('rotor', types=RotorParameters)
        self.options.declare('mode', types=int)
        self.options.declare('num_evaluations', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('num_tangential', types=int)
        

    def setup(self):
        rotor = self.options['rotor']
        mode = self.options['mode']
        num_evaluations = self.options['num_evaluations']
        num_radial = self.options['num_radial']
        num_tangential = self.options['num_tangential']
        shape = (num_evaluations, num_radial, num_tangential)

        if mode == 1:
            group = CoreAnalysisGroup(
                rotor=rotor,
                mode=mode,
                num_evaluations=1,
                num_radial=1,
                num_tangential=1,
            )
            self.add_subsystem(
                'core_analysis_group', group, 
                promotes=[
                    ('hub_radius', 'reference_radius'),
                    ('position', 'reference_position'),
                    ('x_dir', 'reference_x_dir'),
                    ('y_dir', 'reference_y_dir'),
                    ('z_dir', 'reference_z_dir'),
                    ('inflow_velocity', 'reference_inflow_velocity'),
                    ('pitch', 'reference_pitch'),
                    ('chord', 'reference_chord'),
                    'ideal_loading_constant',
                    # 'ideal_loading_constant_non_dimensional',
                    'reference_chord',
                    'reference_radius',
                    'reference_axial_inflow_velocity',
                    'alpha',
                    'Cl0','Cla','Cdmin','K','alpha_Cdmin',
                    'reference_blade_solidity',
                    'reference_tangential_inflow_velocity',
                ],
            )

            group = IdealBladeGroup(
                mode = mode,
                rotor=rotor,
                num_evaluations=num_evaluations,
                num_radial=num_radial,
                num_tangential=num_tangential,
            )
            self.add_subsystem('ideal_blade_group', group, promotes=['*'])

            group = PostProcessGroup(
                shape=shape,
                rotor=rotor,
            )
            self.add_subsystem('postprocess_group', group, promotes=['*'])


        elif mode == 2:
            group = InputsGroup(
                num_evaluations=num_evaluations,
                num_radial=num_radial,
                num_tangential=num_tangential,
            )
            self.add_subsystem('inputs_group', group, promotes=['*'])

            group = PreprocessGroup(
                rotor=rotor,
                shape=shape,
            )
            self.add_subsystem('preprocess_group', group, promotes=['*'])
            
            comp = BEMTImplicitComponent(
                rotor = rotor,
                shape=shape,
                mode = mode,
                num_radial=num_radial,
                )
            self.add_subsystem('bemt_implicit_component', comp, promotes=['*'])
            
            _phi_BEMT = self.declare_input('_phi_BEMT', shape=shape)
            _pitch = self.declare_input('_pitch', shape=shape)

            alpha = _pitch - _phi_BEMT
            self.register_output('_alpha', alpha)

            # group = QuadraticAirfoilGroup(
            #     shape=shape,
            #     rotor=rotor,
            # )
            # self.add_subsystem('airfoil_group', group, promotes=['*'])

            comp = ViternaExplicitComponent(
                shape = shape,
                rotor = rotor,
            )
            self.add_subsystem('viterna_explicit_component', comp, promotes=['*'])

            group = LossGroup(
                rotor = rotor,
                shape = shape,
                mode = mode,
            )
            self.add_subsystem('loss_group', group, promotes=['*'])

            group = InducedVelocityGroup(
                rotor=rotor,
                shape=shape,
                mode = mode,
            )
            self.add_subsystem('induced_velocity_group', group, promotes=['*'])
        


