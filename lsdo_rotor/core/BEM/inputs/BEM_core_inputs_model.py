import numpy as np
from csdl import Model
import csdl


class BEMCoreInputsModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)


    def define(self):
        ft2m = 1/3.281
        shape = self.parameters['shape']
        num_evaluations, num_radial, num_tangential = shape[0], shape[1], shape[2]

        hub_radius = self.declare_variable('hub_radius', shape=(1,))
        rotor_radius = self.declare_variable('propeller_radius', shape=(1,))  # * ft2m / 2 October 21
        # self.print_var(rotor_radius)
        dr = self.declare_variable('dr', shape=(1,))
        
        rotational_speed = self.declare_variable('rotational_speed', shape=(num_evaluations,))
        


        # position = self.declare_variable('position', shape=(num_evaluations,3))
        x_dir = self.declare_variable('x_dir', shape=(num_evaluations, 3))
        y_dir = self.declare_variable('y_dir', shape=(num_evaluations, 3))
        z_dir = self.declare_variable('z_dir', shape=(num_evaluations, 3))
        inflow_velocity = self.declare_variable('inflow_velocity', shape=shape + (3,))
        
        pitch = self.declare_variable('twist_profile', shape=(num_radial,))
        chord = self.declare_variable('chord_profile', shape=(num_radial,))

        # self.print_var(chord)
        # self.print_var(pitch)
        
        direction = self.create_input('direction', val=1., shape=num_evaluations)
        

        self.register_output('_hub_radius', csdl.expand(hub_radius, shape,'l->ijk'))
        self.register_output('_rotor_radius', csdl.expand(rotor_radius, shape,'l->ijk'))
        self.register_output('_dr', csdl.expand(dr, shape,'l->ijk'))
        
        self.register_output('_rotational_speed', csdl.expand(rotational_speed, shape,'i->ijk'))

        # self.register_output('_position', csdl.expand(position, shape + (3,), 'il->ijkl'))
        self.register_output('_x_dir', csdl.expand(x_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_y_dir', csdl.expand(y_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_z_dir', csdl.expand(z_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_inflow_velocity', 1. * inflow_velocity)
        self.register_output('_pitch', csdl.expand(pitch, shape, 'j->ijk'))
        self.register_output('_chord', csdl.expand(chord, shape, 'j->ijk'))


        v = np.linspace(0, np.pi * 2 - np.pi * 2 / num_tangential, num_tangential)

        _theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_evaluations, num_radial)),
            v,
            # np.linspace(0., 2. * np.pi, num_tangential),
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