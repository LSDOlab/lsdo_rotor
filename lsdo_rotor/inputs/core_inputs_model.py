import numpy as np
from csdl import Model
import csdl


class CoreInputsModel(Model):
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

        hub_radius = self.declare_variable('hub_radius', shape=(num_evaluations,))
        rotor_radius = self.declare_variable('rotor_radius', shape=(num_evaluations,))
        rotational_speed = self.declare_variable('rotational_speed', shape=(num_evaluations,))
        dr = self.declare_variable('dr', shape=(num_evaluations,))


        position = self.declare_variable('position', shape=(num_evaluations,3))
        x_dir = self.declare_variable('x_dir', shape=(num_evaluations, 3))
        y_dir = self.declare_variable('y_dir', shape=(num_evaluations, 3))
        z_dir = self.declare_variable('z_dir', shape=(num_evaluations, 3))
        inflow_velocity = self.declare_variable('inflow_velocity', shape=shape + (3,))
        
        pitch = self.declare_variable('pitch', shape=(num_radial,))
        chord = self.declare_variable('chord', shape=(num_radial,))
        
        direction = self.create_input('direction', val=1., shape=num_evaluations)
        

        # Dynamic inflow
        M = self.declare_variable('M', shape=(3,3))
        M_inv = self.declare_variable('M_inv', shape=(3,3))
        nu_0_vec = self.declare_variable('nu_0_vec', shape=(num_evaluations,))
        nu_s_vec = self.declare_variable('nu_s_vec', shape=(num_evaluations,))
        nu_c_vec = self.declare_variable('nu_c_vec', shape=(num_evaluations,))
        

        self.register_output('M_matrix', csdl.expand(M, (num_evaluations,3,3),'ll->ill'))
        self.register_output('inv_M_matrix', csdl.expand(M_inv, (num_evaluations,3,3),'ll->ill'))
        self.register_output('nu_0_vec_exp', csdl.expand(nu_0_vec,(shape),'i->ijk'))
        self.register_output('nu_s_vec_exp', csdl.expand(nu_s_vec,(shape),'i->ijk'))
        self.register_output('nu_c_vec_exp', csdl.expand(nu_c_vec,(shape),'i->ijk'))

        self.register_output('_hub_radius', csdl.expand(hub_radius, shape,'i->ijk'))
        self.register_output('_rotor_radius', csdl.expand(rotor_radius, shape,'i->ijk'))
        self.register_output('_rotational_speed', csdl.expand(rotational_speed, shape,'i->ijk'))
        self.register_output('_dr', csdl.expand(dr, shape,'i->ijk'))
        

        self.register_output('_position', csdl.expand(position, shape + (3,), 'il->ijkl'))
        self.register_output('_x_dir', csdl.expand(x_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_y_dir', csdl.expand(y_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_z_dir', csdl.expand(z_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_inflow_velocity', 1. * inflow_velocity)
        self.register_output('_pitch', csdl.expand(pitch, shape, 'j->ijk'))
        self.register_output('_chord', csdl.expand(chord, shape, 'j->ijk'))

        

        angle1 = np.linspace(0., 1. * np.pi, num_tangential)
        angle2 = np.linspace(np.pi, 2. * np.pi, num_tangential)
        angle = np.concatenate((angle1,angle2))
        angle = angle[0::2]
        v = np.linspace(0, np.pi * 2 - np.pi * 2 / num_tangential, num_tangential)
        # v = np.linspace(0,2 * np.pi, num_tangential)
        _theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_evaluations, num_radial)),
            v,
            # np.linspace(0., 2. * np.pi, num_tangential),
        )
        # print(_theta.shape,'THETA')
        self.create_input('_theta', val=_theta)

        normalized_radial_discretization = 1. / num_radial / 2. \
            + np.linspace(0., 1. - 1. / num_radial, num_radial)

        _normalized_radius = np.einsum(
            'ik,j->ijk',
            np.ones((num_evaluations, num_tangential)),
            normalized_radial_discretization,
        )
        self.create_input('_normalized_radius', val=_normalized_radius)