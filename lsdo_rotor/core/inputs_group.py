import numpy as np

import omtools.api as ot
# from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup

class InputsGroup(ot.Group):

    def initialize(self):
        self.options.declare('num_evaluations', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('num_tangential', types=int)

    def setup(self):
        num_evaluations = self.options['num_evaluations']
        num_radial = self.options['num_radial']
        num_tangential = self.options['num_tangential']
        shape = (num_evaluations, num_radial, num_tangential)

        Cl0 = self.declare_input('Cl0')
        Cla = self.declare_input('Cla')
        Cdmin = self.declare_input('Cdmin')
        K = self.declare_input('K')
        alpha_Cdmin = self.declare_input('alpha_Cdmin')
        Cl_stall = self.declare_input('Cl_stall')
        Cd_stall = self.declare_input('Cd_stall')
        Cl_stall_minus = self.declare_input('Cl_stall_minus')
        Cd_stall_minus = self.declare_input('Cd_stall_minus')

        hub_radius = self.declare_input('hub_radius')
        rotor_radius = self.declare_input('rotor_radius')
        slice_thickness = self.declare_input('slice_thickness')
        reference_chord = self.declare_input('reference_chord')
        reference_radius = self.declare_input('reference_radius')
        alpha = self.declare_input('alpha')
        alpha_stall = self.declare_input('alpha_stall')
        alpha_stall_minus = self.declare_input('alpha_stall_minus')
        AR = self.declare_input('AR')

        position = self.declare_input('position', shape=(num_evaluations,3))
        x_dir = self.declare_input('x_dir', shape=(num_evaluations, 3))
        y_dir = self.declare_input('y_dir', shape=(num_evaluations, 3))
        z_dir = self.declare_input('z_dir', shape=(num_evaluations, 3))
        inflow_velocity = self.declare_input('inflow_velocity', shape=shape + (3,))
        pitch = self.declare_input('pitch', shape=(num_radial,))
        chord = self.declare_input('chord', shape=(num_radial,))
        

        # rotational_speed = self.create_indep_var('rotational_speed')
        rotational_speed = self.declare_input('rotational_speed')
        direction = self.create_indep_var('direction', val=1., shape=num_evaluations)

        # Cl = self.declare_input('Cl')
        # Cd = self.declare_input('Cd')

        # self.register_output('_Cl', ot.expand(Cl,shape))
        # self.register_output('_Cd', ot.expand(Cd,shape))

        self.register_output('_hub_radius', ot.expand(hub_radius, shape))
        self.register_output('_rotor_radius', ot.expand(rotor_radius, shape))
        self.register_output('_slice_thickness', ot.expand(slice_thickness, shape))
        # self.register_output('_alpha', ot.expand(alpha, shape))
        self.register_output('_reference_chord', ot.expand(reference_chord,shape))
        self.register_output('_reference_radius', ot.expand(reference_radius,shape))

        self.register_output('_position', ot.expand(position, shape + (3,), 'il->ijkl'))
        self.register_output('_x_dir', ot.expand(x_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_y_dir', ot.expand(y_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_z_dir', ot.expand(z_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_inflow_velocity', 1. * inflow_velocity)
        # print(shape,pitch.shape)
        self.register_output('_pitch', ot.expand(pitch, shape, 'j->ijk'))
        self.register_output('_chord', ot.expand(chord, shape, 'j->ijk'))

        self.register_output('_rotational_speed', ot.expand(rotational_speed, shape))
        # self.register_output('_direction', ot.expand(direction, shape, 'i->ijk'))

        _theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_evaluations, num_radial)),
            np.linspace(0., 2. * np.pi, num_tangential),
        )
        self.create_indep_var('_theta', val=_theta)

        normalized_radial_discretization = 1. / num_radial / 2. \
            + np.linspace(0., 1. - 1. / num_radial, num_radial)

        _normalized_radius = np.einsum(
            'ik,j->ijk',
            np.ones((num_evaluations, num_tangential)),
            normalized_radial_discretization,
        )
        self.create_indep_var('_normalized_radius', val=_normalized_radius)