import numpy as np
from csdl import Model
import csdl


class CoreInputsGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)

    def define(self):
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        shape = (num_evaluations, num_radial, num_tangential)

        hub_radius = self.declare_variable('hub_radius')
        rotor_radius = self.declare_variable('rotor_radius')
        slice_thickness = self.declare_variable('slice_thickness')
        reference_chord = self.declare_variable('reference_chord')
        reference_radius = self.declare_variable('reference_radius')
        alpha = self.declare_variable('alpha')
        alpha_stall = self.declare_variable('alpha_stall')
        alpha_stall_minus = self.declare_variable('alpha_stall_minus')
        AR = self.declare_variable('AR')

        position = self.declare_variable('position', shape=(num_evaluations,3))
        x_dir = self.declare_variable('x_dir', shape=(num_evaluations, 3))
        y_dir = self.declare_variable('y_dir', shape=(num_evaluations, 3))
        z_dir = self.declare_variable('z_dir', shape=(num_evaluations, 3))
        inflow_velocity = self.declare_variable('inflow_velocity', shape=shape + (3,))
        pitch = self.declare_variable('pitch', shape=(num_radial,))
        chord = self.declare_variable('chord', shape=(num_radial,))
        
        rotational_speed = self.declare_variable('rotational_speed')
        direction = self.create_input('direction', val=1., shape=num_evaluations)

        self.register_output('_hub_radius', csdl.expand(hub_radius, shape))
        self.register_output('_rotor_radius', csdl.expand(rotor_radius, shape))
        self.register_output('_slice_thickness', csdl.expand(slice_thickness, shape))
        self.register_output('_reference_chord', csdl.expand(reference_chord,shape))
        self.register_output('_reference_radius', csdl.expand(reference_radius,shape))

        self.register_output('_position', csdl.expand(position, shape + (3,), 'il->ijkl'))
        self.register_output('_x_dir', csdl.expand(x_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_y_dir', csdl.expand(y_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_z_dir', csdl.expand(z_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_inflow_velocity', 1. * inflow_velocity)
        self.register_output('pitch_distribution', csdl.expand(pitch, shape, 'j->ijk'))
        self.register_output('chord_distribution', csdl.expand(chord, shape, 'j->ijk'))

        self.register_output('_rotational_speed', csdl.expand(rotational_speed, shape))

        _theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_evaluations, num_radial)),
            np.linspace(0., 2. * np.pi, num_tangential),
        )
        self.create_input('_theta', val=_theta)

        normalized_radial_discretization = 1. / num_radial / 2. \
            + np.linspace(0., 1. - 1. / num_radial, num_radial)

        _normalized_radius = np.einsum(
            'ik,j->ijk',
            np.ones((num_evaluations, num_tangential)),
            normalized_radial_discretization,
        )
        self.create_input('_normalized_radius', val=_normalized_radius)