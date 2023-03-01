from weakref import ref
import numpy as np
from csdl import Model
import csdl


class BILDExternalInputsModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        shape = self.parameters['shape']
        shape = (shape[0], shape[1], shape[2])
        num_blades = self.parameters['num_blades']

        proj_vec = np.array([0,1,0]).reshape(1,3)
        projection_vec = self.create_input('projection_vector',val=proj_vec)

        num_nodes = num_evaluations = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]

        # radius = self.declare_variable(name='propeller_radius', shape=(num_nodes,), units='m')
        rotor_radius = self.declare_variable(name='propeller_radius', shape=(num_nodes,), units='m')
        # self.print_var(rotor_radius)
        # rotor_radius = csdl.expand(radius,(num_nodes,))
        ref_radius = self.declare_variable('reference_radius', shape=(num_nodes,))
        ref_chord = self.declare_variable('reference_chord', shape=(num_nodes,))

        omega = self.declare_variable('omega', shape=(num_nodes, ), units='rpm')

        u = self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s') * -1
        v = self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s', val=0) 
        w = self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s', val=0) 
        
        V = self.create_output('velocity_vector', shape=(num_nodes,3), units='m/s')
        
        inflow_velocity = self.create_output('inflow_velocity', shape=shape + (3,))
        x_dir = np.zeros((num_evaluations,3))
        y_dir = np.zeros((num_evaluations,3))
        z_dir = np.zeros((num_evaluations,3))
       
        
        R_h = 0.2 * rotor_radius
        self.register_output('hub_radius', R_h)
        dr = ((rotor_radius)-(0.2 * rotor_radius))/ (num_radial - 1)
        self.register_output('dr',dr)

        BILD_Vt = self.create_output('BILD_tangential_inflow_velocity', shape=(num_nodes,))
        ref_sigma = self.create_output('reference_blade_solidity', shape=(num_nodes,))

        n = self.create_output('rotational_speed', shape=(num_evaluations,))
        thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes,3))
        for i in range(num_nodes):
            normal_vec = thrust_vector[i,:]
            normal_vec_axial_induced = -1 * normal_vec

            in_plane_1 = projection_vec - csdl.expand(csdl.dot(projection_vec,normal_vec,axis=1),(1,3) ) * normal_vec
            in_plane_ey =  (in_plane_1 / csdl.expand(csdl.pnorm(in_plane_1,pnorm_type=2),(1,3)))
            in_plane_ex = csdl.cross(normal_vec,in_plane_ey, axis=1)

            x_dir[i,0] = 1 
            y_dir[i,1] = 1
            z_dir[i,2] = 1
            
            n[i] = omega[i] / 60
            BILD_Vt[i] = omega[i] / 60 * 2 * np.pi * ref_radius[i]
            ref_sigma[i] = num_blades * ref_chord[i] / 2 / np.pi / ref_radius[i]
            
            V[i,0] = u[i,0]
            V[i,1] = v[i,0]
            V[i,2] = w[i,0]

            in_plane_ux = csdl.dot(V[i,:], in_plane_ex,axis=1)
            in_plane_uy = csdl.dot(V[i,:], in_plane_ey,axis=1)
            normal_uz = csdl.dot(V[i,:],normal_vec_axial_induced,axis=1)
            
            inflow_velocity[i,:,:,0] = csdl.expand(normal_uz,(1,num_radial,num_tangential,1),'i->ijkl')
            inflow_velocity[i,:,:,1] = csdl.expand(-in_plane_ux,(1,num_radial,num_tangential,1),'i->ijkl')
            inflow_velocity[i,:,:,2] = csdl.expand(in_plane_uy, (1,num_radial,num_tangential,1),'i->ijkl')
        
        self.create_input('x_dir',val=x_dir)
        self.create_input('y_dir',val=y_dir)
        self.create_input('z_dir',val=z_dir)
