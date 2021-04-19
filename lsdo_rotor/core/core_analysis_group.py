import omtools.api as ot

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.inputs_group import InputsGroup
from lsdo_rotor.core.preprocess_group import PreprocessGroup
from lsdo_rotor.core.bemt_implicit_component import BEMTImplicitComponent
from lsdo_rotor.core.induced_velocity_group import InducedVelocityGroup
# from lsdo_rotor.core.airfoil_group import AirfoilGroup


class CoreAnalysisGroup(ot.Group):

    def initialize(self):
        self.options.declare('rotor', types=RotorParameters)
        self.options.declare('num_evaluations', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('num_tangential', types=int)
        self.options.declare('mode', types=int)

    def setup(self):
        rotor = self.options['rotor']
        num_evaluations = self.options['num_evaluations']
        num_radial = self.options['num_radial']
        num_tangential = self.options['num_tangential']
        mode = self.options['mode']

        shape = (num_evaluations, num_radial, num_tangential)

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
            shape=shape,
            mode = mode,
            num_radial=num_radial,
            rotor = rotor,
            )
        self.add_subsystem('bemt_implicit_component', comp, promotes=['*'])

        group = InducedVelocityGroup(
            rotor=rotor,
            shape=shape,
            mode = mode,
        )
        self.add_subsystem('induced_velocity_group', group, promotes=['*'])
        
