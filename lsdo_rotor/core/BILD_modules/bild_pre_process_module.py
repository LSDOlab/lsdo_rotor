from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import numpy as np
import csdl


class BILDPreprocessModuleCSDL(ModuleCSDL):

    def initialize(self):
        self.parameters.declare('shape', types=tuple)
    
    def define(self):
        shape = self.parameters['shape']

        # -----
        _rotational_speed = self.register_module_input('_rotational_speed', shape=shape)
        _hub_radius = self.register_module_input('_hub_radius', shape=shape)
        _rotor_radius = self.register_module_input('_rotor_radius', shape=shape)
        _theta = self.register_module_input('_theta', shape=shape)
        _normalized_radius = self.register_module_input('_normalized_radius', shape=shape)

        _inflow_velocity = self.register_module_input('_inflow_velocity', shape=shape + (3,))
        _x_dir = self.register_module_input('_x_dir', shape=shape + (3,))
        _y_dir = self.register_module_input('_y_dir', shape=shape + (3,))
        _z_dir = self.register_module_input('_z_dir', shape=shape + (3,))
        _direction = self.register_module_input('_direction', shape=shape)

        # -----
        _angular_speed = 2 * np.pi * _rotational_speed
        self.register_module_output('_angular_speed', _angular_speed)

        _radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_module_output('_radius', _radius, importance=1)

        _ref_radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_module_output('_ref_radius',_ref_radius)
        # -----


        _inflow_x = csdl.einsum(_inflow_velocity, _x_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_y = csdl.einsum(_inflow_velocity, _y_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_z = csdl.einsum(_inflow_velocity, _z_dir, subscripts='ijkl,ijkl->ijk')

        self.register_module_output('_axial_inflow_velocity', _inflow_x, importance=1)
        self.register_module_output('_in_plane_inflow_velocity',_inflow_y)
        self.register_module_output('inflow_z', _inflow_z)
        
        self.register_module_output(
            '_tangential_inflow_velocity', 
            _direction * _inflow_z * csdl.cos(_theta) + 
            _direction * _inflow_y * csdl.sin(_theta) + 
            _radius * _angular_speed, importance=1
        )
        