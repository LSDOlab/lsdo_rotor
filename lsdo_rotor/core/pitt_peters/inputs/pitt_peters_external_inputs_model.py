import numpy as np
from csdl import Model
import csdl


class PittPetersExternalInputsModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']
        shape = (shape[0], shape[1], shape[2])
        
        num_nodes = num_evaluations = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]
        

        proj_vec = np.array([0,1,0]).reshape(1,3)
        projection_vec = self.create_input('projection_vector',val=proj_vec)
        

        ft2m = 1/3.281
        rotor_radius = self.declare_variable(name='propeller_radius', shape=(1,), units='m') #* ft2m / 2
        # Inputs changing across conditions (segments)
        omega = self.declare_variable('rpm', shape=(num_nodes, 1), units='rpm') #* 1000

        u = self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s') * -1
        v = self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s') 
        w = self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s') 

        V = self.create_output('velocity_vector', shape=(num_nodes,3), units='m/s')

        p = self.declare_variable(name='p', shape=(num_nodes, 1), units='rad/s')
        self.register_output('p1',p*1)
        q = self.declare_variable(name='q', shape=(num_nodes, 1), units='rad/s')
        self.register_output('q1',q*1)
        r = self.declare_variable(name='r', shape=(num_nodes, 1), units='rad/s')
        self.register_output('r1',r*1)

        inflow_velocity = self.create_output('inflow_velocity', shape=shape + (3,))
        x_dir = np.zeros((num_evaluations,3))
        y_dir = np.zeros((num_evaluations,3))
        z_dir = np.zeros((num_evaluations,3))

        R_h = 0.2 * rotor_radius
        self.register_output('hub_radius', R_h)
        dr = ((rotor_radius)-(0.2 * rotor_radius))/ (num_radial - 1)
        self.register_output('dr', dr)

        n = self.create_output('rotational_speed', shape=(num_evaluations, 1))
        thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes, 3))
       

        # normal_vec = self.create_input('thrust_vector', val=thrust_vector)
        # normal_vec_axial_induced = self.create_input('thrust_vector_axial_induced', val=thrust_vector_axial_induced)
        # projection_vec = self.create_input('projection_vector',val=proj_vec)
        
        # in_plane_1 = projection_vec - csdl.expand(csdl.dot(projection_vec,normal_vec,axis=1),(1,3) ) * normal_vec
        # in_plane_ex = in_plane_1 / csdl.expand(csdl.pnorm(in_plane_1,pnorm_type=2),(1,3))
        # in_plane_ey = csdl.cross(in_plane_ex,normal_vec, axis=1)
              

        # inflow_velocity = self.create_output('inflow_velocity', shape=shape + (3,))
        # x_dir = np.zeros((num_evaluations,3))
        # y_dir = np.zeros((num_evaluations,3))
        # z_dir = np.zeros((num_evaluations,3))
       
        
        # R_h = self.create_output('hub_radius', shape=(1,))
        # dr = self.create_output('dr', shape=(1,))
        # R_h = 0.2 * rotor_radius
        # self.register_output('hub_radius',R_h)
        # dr = ((rotor_radius)-(0.2 * rotor_radius))/ (num_radial - 1)
        # self.register_output('dr',dr)
        # n = self.create_output('rotational_speed', shape=(num_evaluations,))

        in_plane_inflow = self.create_output('in_plane_inflow', shape=(num_nodes,))
        normal_inflow = self.create_output('normal_inflow', shape=(num_nodes,))

        mu_z = self.create_output('mu_z', shape=(num_nodes,1))
        mu = self.create_output('mu', shape=(num_nodes,1))

        for i in range(num_nodes):
            print('i        ',i)
            normal_vec = thrust_vector[i,:]
            normal_vec_axial_induced = -1 * normal_vec 

            in_plane_1 = projection_vec - csdl.expand(csdl.dot(projection_vec,normal_vec,axis=1),(1,3) ) * normal_vec
            in_plane_ex = in_plane_1 / csdl.expand(csdl.pnorm(in_plane_1,pnorm_type=2),(1,3))
            in_plane_ey = csdl.cross(in_plane_ex,normal_vec, axis=1)
            
            x_dir[i,0] = 1 
            y_dir[i,1] = 1
            z_dir[i,2] = 1
            n[i,0] = omega[i,0] / 60
            
            
            V[i,0] = u[i,0]
            V[i,1] = v[i,0]
            V[i,2] = w[i,0]

            in_plane_ux = csdl.dot(V[i,:], in_plane_ex,axis=1)
            in_plane_uy = csdl.dot(V[i,:], in_plane_ey,axis=1)
            normal_uz = csdl.dot(V[i,:],normal_vec_axial_induced,axis=1)
            
            in_plane_inflow[i] = (in_plane_ux**2 + in_plane_uy**2)**0.5
            normal_inflow[i] = normal_uz

            print(normal_uz.shape)
            print(n[i,0].shape)
            f = csdl.reshape(rotor_radius,(1,1))
            print(f.shape)
            mu_z[i,0] = csdl.reshape(normal_uz,(1,1)) / (n[i,0] * 2 * np.pi) / csdl.reshape(rotor_radius,(1,1))
            mu[i,0] = (csdl.reshape(in_plane_ux,(1,1))**2 + csdl.reshape(in_plane_uy,(1,1))**2)**0.5 / (n[i,0] * 2 * np.pi) / csdl.reshape(rotor_radius,(1,1))

            inflow_velocity[i,:,:,0] = csdl.expand(normal_uz,(1,num_radial,num_tangential,1),'i->ijkl')
            inflow_velocity[i,:,:,1] = csdl.expand(-in_plane_ux,(1,num_radial,num_tangential,1),'i->ijkl')
            inflow_velocity[i,:,:,2] = csdl.expand(in_plane_uy, (1,num_radial,num_tangential,1),'i->ijkl')
        
        # self.register_output('_mu_z_expanded', csdl.expand(mu_z,shape,'il->ijk'))
        self.create_input('x_dir',val=x_dir)
        self.create_input('y_dir',val=y_dir)
        self.create_input('z_dir',val=z_dir)
