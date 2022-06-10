import numpy as np
from csdl import Model
import csdl
from lsdo_rotor.rotor_parameters import RotorParameters

class PreprocessModel(Model):

    def initialize(self):
        self.parameters.declare('rotor', types=RotorParameters)
        self.parameters.declare('shape', types=tuple)

    def define(self):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']

        num_blades = rotor['num_blades']

        # -----
        _rotational_speed = self.declare_variable('_rotational_speed', shape=shape)
        _chord = self.declare_variable('_chord', shape=shape)
        _hub_radius = self.declare_variable('_hub_radius', shape=shape)
        _rotor_radius = self.declare_variable('_rotor_radius', shape=shape)
        _theta = self.declare_variable('_theta', shape=shape)
        _normalized_radius = self.declare_variable('_normalized_radius', shape=shape)

        _inflow_velocity = self.declare_variable('_inflow_velocity', shape=shape + (3,))
        _x_dir = self.declare_variable('_x_dir', shape=shape + (3,))
        _y_dir = self.declare_variable('_y_dir', shape=shape + (3,))
        _z_dir = self.declare_variable('_z_dir', shape=shape + (3,))
        _direction = self.declare_variable('_direction', shape=shape)

        # -----
        _angular_speed = 2 * np.pi * _rotational_speed
        self.register_output('_angular_speed', _angular_speed)

        _radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_output('_radius', _radius)

        _ref_radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_output('_ref_radius',_ref_radius)
        # -----

        self.register_output('_blade_solidity', num_blades * _chord / 2. / np.pi / _radius)
        # -----

        _inflow_x = csdl.einsum(_inflow_velocity, _x_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_y = csdl.einsum(_inflow_velocity, _y_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_z = csdl.einsum(_inflow_velocity, _z_dir, subscripts='ijkl,ijkl->ijk')

        self.register_output('_axial_inflow_velocity', _inflow_x)
        self.register_output('_in_plane_inflow_velocity',_inflow_y)
        # self.register_output('inflow_y',_inflow_y)
        self.register_output('inflow_z', _inflow_z)

        # i_vec = rotor['rotor_disk_tilt_angle']
        # lamb = rotor['speed_ratio']
        # mu = rotor['inflow_ratio']
        # L = np.zeros((shape[0],3,3))
        # L_inv = np.zeros((shape[0],3,3))

        # L_list = []
        # L_inv_list = []

        # for i in range(shape[0]):
        #     L[i,0,0] = 0.5
        #     L[i,0,2] = 15 * np.pi/64 * ((1 - np.sin(i_vec[i]))/(1 + np.sin(i_vec[i])))**0.5
        #     L[i,1,1] = - 4 / (1 + np.sin(i_vec[i]))
        #     L[i,2,0] = L[i,0,2]
        #     L[i,2,2] = - 4 * np.sin(i_vec[i]) / (1 + np.sin(i_vec[i]))

        #     V_eff = (lamb[i]**2 + mu[i]**2)**0.5
        #     print(V_eff,'V_eff')

        #     L[i,:,:] = L[i,:,:] / 1
        #     L_list.append(L[i,:,:])

        #     L_inv[i,:,:] = np.linalg.inv(L[i,:,:])
        #     L_inv_list.append(L_inv[i,:,:])

        # from scipy.linalg import block_diag
        # L_block_diag = block_diag(*L_list)
        # L_inv_block_diag = block_diag(*L_inv_list)


        # self.create_input('L_block_diag_matrix', val=L_block_diag)
        # self.create_input('L_inv_block_diag_matrix', val=L_inv_block_diag)
        

        self.register_output(
            '_tangential_inflow_velocity', 
            _direction * _inflow_y * csdl.cos(_theta) - 
            _direction * _inflow_z * csdl.sin(_theta) + 
            _radius * _angular_speed
        )

