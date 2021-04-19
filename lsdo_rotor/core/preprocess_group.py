import numpy as np

import omtools.api as ot
from lsdo_rotor.rotor_parameters import RotorParameters


class PreprocessGroup(ot.Group):

    def initialize(self):
        self.options.declare('rotor', types=RotorParameters)
        self.options.declare('shape', types=tuple)

    def setup(self):
        rotor = self.options['rotor']
        shape = self.options['shape']

        num_blades = rotor['num_blades']

        # -----

        _rotational_speed = self.declare_input('_rotational_speed', shape=shape)
        _chord = self.declare_input('_chord', shape=shape)
        _hub_radius = self.declare_input('_hub_radius', shape=shape)
        _rotor_radius = self.declare_input('_rotor_radius', shape=shape)
        _theta = self.declare_input('_theta', shape=shape)
        _normalized_radius = self.declare_input('_normalized_radius', shape=shape)

        _inflow_velocity = self.declare_input('_inflow_velocity', shape=shape + (3,))
        _x_dir = self.declare_input('_x_dir', shape=shape + (3,))
        _y_dir = self.declare_input('_y_dir', shape=shape + (3,))
        _z_dir = self.declare_input('_z_dir', shape=shape + (3,))
        _direction = self.declare_input('_direction', shape=shape)

        _reference_chord = self.declare_input('_reference_chord', shape=shape)
        _reference_radius = self.declare_input('_reference_radius',shape=shape)

        # -----

        _angular_speed = 2 * np.pi * _rotational_speed
        self.register_output('_angular_speed', _angular_speed)

        _radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_output('_radius', _radius)

        _ref_radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_output('_ref_radius',_ref_radius)
        # -----

        self.register_output('_blade_solidity', num_blades * _chord / 2. / np.pi / _radius)
        self.register_output('_reference_blade_solidity', num_blades * _reference_chord / 2. / np.pi / _reference_radius)
        # -----

        # _inflow_x = ot.dot(_inflow_velocity, _x_dir, axes=[3, 3])
        # _inflow_y = ot.dot(_inflow_velocity, _y_dir, axes=[3, 3])
        # _inflow_z = ot.dot(_inflow_velocity, _z_dir, axes=[3, 3])
        _inflow_x = ot.einsum(_inflow_velocity, _x_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_y = ot.einsum(_inflow_velocity, _y_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_z = ot.einsum(_inflow_velocity, _z_dir, subscripts='ijkl,ijkl->ijk')
        
        self.register_output('_axial_inflow_velocity', _inflow_x)

        self.register_output(
            '_tangential_inflow_velocity', 
            _direction * _inflow_y * ot.sin(_theta) - 
            _direction * _inflow_z * ot.cos(_theta) + 
            _radius * _angular_speed
        )

        self.register_output(
            '_reference_tangnetial_inflow_velocity',
            _direction * _inflow_y * ot.sin(_theta) - 
            _direction * _inflow_z * ot.cos(_theta) + 
            _reference_radius * _angular_speed
        )