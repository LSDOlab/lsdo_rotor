import numpy as np
from csdl import Model
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


class BEMExternalInputsModel(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('hub_radius_percent')

    def define(self):
        shape = self.parameters['shape']
        shape = (shape[0], shape[1], shape[2])
        
        r_h_percent = self.parameters['hub_radius_percent']
        
        num_nodes = num_evaluations = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]
        

        proj_vec = np.array([0,1,0]).reshape(1,3)
        projection_vec = self.create_input('projection_vector',val=proj_vec)


       
        ft2m = 1/3.281
        rotor_radius = self.register_module_input(name='propeller_radius', shape=(1,), units='m') #* ft2m / 2
        # self.print_var(rotor_radius)

        # Inputs changing across conditions (segments)
        omega = self.register_module_input('rpm', shape=(num_nodes, 1), units='rpm') #* 1000

        u = self.register_module_input(name='u', shape=(num_nodes, 1), units='m/s') * -1
        v = self.register_module_input(name='v', shape=(num_nodes, 1), units='m/s', val=0) 
        w = self.register_module_input(name='w', shape=(num_nodes, 1), units='m/s', val=0) 

        V = self.create_output('velocity_vector', shape=(num_nodes,3), units='m/s')

        p = self.register_module_input(name='p', shape=(num_nodes, 1), units='rad/s')
        self.register_output('p1',p*1)
        q = self.register_module_input(name='q', shape=(num_nodes, 1), units='rad/s')
        self.register_output('q1',q*1)
        r = self.register_module_input(name='r', shape=(num_nodes, 1), units='rad/s')
        self.register_output('r1',r*1)

        inflow_velocity = self.create_output('inflow_velocity', shape=shape + (3,))
        x_dir = np.zeros((num_evaluations,3))
        y_dir = np.zeros((num_evaluations,3))
        z_dir = np.zeros((num_evaluations,3))
       

        R_h = r_h_percent * rotor_radius
        self.register_output('hub_radius',R_h)
        dr = ((rotor_radius)-(0.2 * rotor_radius))/ (num_radial - 1)
        self.register_output('dr',dr)

        n = self.create_output('rotational_speed', shape=(num_evaluations,1))
        thrust_vector = self.register_module_input('thrust_vector', shape=(num_nodes,3))
        for i in range(num_nodes):
            normal_vec = thrust_vector[i,:]
            normal_vec_axial_induced = -1 * normal_vec 
            
            in_plane_1 = projection_vec - csdl.expand(csdl.dot(projection_vec,normal_vec,axis=1),(1,3) ) * normal_vec
            in_plane_ey =  (in_plane_1 / csdl.expand(csdl.pnorm(in_plane_1,pnorm_type=2),(1,3)))
            in_plane_ex = csdl.cross(normal_vec,in_plane_ey, axis=1)
            
            
            x_dir[i,0] = 1 
            y_dir[i,1] = 1
            z_dir[i,2] = 1
            n[i,0] = omega[i,0] / 60
            
            
            V[i,0] = u[i,0]
            V[i,1] = v[i,0]
            V[i,2] = w[i,0]

            in_plane_ux = csdl.dot(V[i,:], in_plane_ex, axis=1)
            in_plane_uy = csdl.dot(V[i,:], in_plane_ey, axis=1)
            normal_uz = csdl.dot(V[i,:],normal_vec_axial_induced, axis=1)

            inflow_velocity[i,:,:,0] = csdl.expand(normal_uz,(1,num_radial,num_tangential,1),'i->ijkl')
            inflow_velocity[i,:,:,1] = csdl.expand(-in_plane_ux,(1,num_radial,num_tangential,1),'i->ijkl')
            inflow_velocity[i,:,:,2] = csdl.expand(in_plane_uy, (1,num_radial,num_tangential,1),'i->ijkl')
        
        self.register_output('in_plane_ex', in_plane_ex)
        self.register_output('in_plane_ey', in_plane_ey)
        self.create_input('x_dir',val=x_dir)
        self.create_input('y_dir',val=y_dir)
        self.create_input('z_dir',val=z_dir)

