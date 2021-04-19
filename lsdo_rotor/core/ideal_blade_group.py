import omtools.api as ot

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.inputs_group import InputsGroup
from lsdo_rotor.core.preprocess_group import PreprocessGroup
from lsdo_rotor.core.efficiency_coeffs_group import EfficiencyCoeffsGroup
from lsdo_rotor.core.efficiency_implicit_component import EfficiencyImplicitComponent
from lsdo_rotor.core.airfoil_group import AirfoilGroup


class IdealBladeGroup(ot.Group):

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

        if mode == 1:

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

            group = AirfoilGroup(shape=shape)
            self.add_subsystem('airfoil_group',group, promotes=['*'])

            group = EfficiencyCoeffsGroup(shape=shape)
            self.add_subsystem('efficiency_coeffs_group', group, promotes=['*'])

            comp = EfficiencyImplicitComponent(shape=shape)
            self.add_subsystem('efficiency_implicit_component', comp, promotes=['*'])

        

            Vx = self.declare_input('_axial_inflow_velocity', shape=shape)
            Vt = self.declare_input('_tangential_inflow_velocity', shape=shape)
            eta = self.declare_input('_efficiency', shape=shape)
            eta2 = self.declare_input('_efficiency2', shape=shape)
            Cl = self.declare_input('_Cl',shape = shape)
            Cd = self.declare_input('_Cd',shape = shape)


            # x = Vt/Vx
            # ax = x**2 * eta2 * (1 - eta2)/ (1 + x**2 * eta2**2)
            # ay = (1 - eta2)/(1 + x**2 * eta2**2)

            a = 2 * Cl
            b = 2 * Cd * Vt - 2 * Cl * Vx
            c = - 2 * Vt * eta * (Cd * Vx + Cl * Vt - Cl * Vt * eta)

            ux = (-b + (b**2 - 4 * a * c)**0.5)/ (2 * a)
            ut = 2. * Vt * (1. - eta)

            eta_1 = Vx / ux
            # eta_2 = (Vt - 0.5 * ut) / Vt

            self.register_output('_axial_induced_velocity', ux)
            self.register_output('_tangential_induced_velocity', ut)
            self.register_output('_eta1',eta_1)
            # self.register_output('_axial_induction_factor', ax)
            # self.register_output('_tangential_induction_factor',ay)
            