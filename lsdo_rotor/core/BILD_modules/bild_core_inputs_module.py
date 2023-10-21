
import numpy as np
import csdl


class BILDCoreInputsModuleCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)


    def define(self):
        shape = self.parameters['shape']
        num_evaluations, num_radial, num_tangential = shape[0], shape[1], shape[2]

        hub_radius = self.register_module_input('hub_radius', shape=(num_evaluations,))
        rotor_radius = self.register_module_input('propeller_radius', shape=(num_evaluations,))
        dr = self.register_module_input('dr', shape=(num_evaluations,))

        # NOTE: csdl will not throw an error if the shapes of variables that are 
        # computed in an upstream model don't match
        rotational_speed = self.register_module_input('rotational_speed', shape=(num_evaluations ,))
        
        x_dir = self.register_module_input('x_dir', shape=(num_evaluations, 3))
        y_dir = self.register_module_input('y_dir', shape=(num_evaluations, 3))
        z_dir = self.register_module_input('z_dir', shape=(num_evaluations, 3))
        inflow_velocity = self.register_module_input('inflow_velocity', shape=shape + (3,))        

        self.register_module_output('_hub_radius', csdl.expand(hub_radius, shape,'i->ijk'))
        self.register_module_output('_rotor_radius', csdl.expand(rotor_radius, shape,'i->ijk'))
        self.register_module_output('_dr', csdl.expand(dr, shape,'i->ijk'))
        
        self.register_module_output('_rotational_speed', csdl.expand(rotational_speed, shape,'i->ijk'))
        self.register_module_output('_x_dir', csdl.expand(x_dir, shape + (3,), 'il->ijkl'))
        self.register_module_output('_y_dir', csdl.expand(y_dir, shape + (3,), 'il->ijkl'))
        self.register_module_output('_z_dir', csdl.expand(z_dir, shape + (3,), 'il->ijkl'))
        self.register_module_output('_inflow_velocity', 1. * inflow_velocity)


        v = np.linspace(0, np.pi * 2 - np.pi * 2 / num_tangential, num_tangential)

        _theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_evaluations, num_radial)),
            v,
            # np.linspace(0., 2. * np.pi, num_tangential),
        )
        theta = self.register_module_input('theta', val=_theta)
        theta_out = self.register_module_output('_theta', shape=shape)
        theta_out[:, :, :] = theta

        normalized_radial_discretization = 1. / num_radial / 2. \
            + np.linspace(0., 1. - 1. / num_radial, num_radial)
        

        _normalized_radius = np.einsum(
            'ik,j->ijk',
            np.ones((num_evaluations, num_tangential)),
            normalized_radial_discretization,
        )
        norm_rad = self.register_module_input('normalized_radius', val=_normalized_radius)

        norm_rad_out = self.register_module_output('_normalized_radius', shape=shape)
        norm_rad_out[:, :, :] = norm_rad