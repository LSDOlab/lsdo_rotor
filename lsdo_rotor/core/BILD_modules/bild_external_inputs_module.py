from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
import numpy as np
import csdl


class BILDExternalInputsModuleCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        shape = self.parameters['shape']
        num_blades = self.parameters['num_blades']

        proj_vec = np.array([0,1,0]).reshape(1,3)
        projection_vec = self.register_module_input('projection_vector',val=proj_vec)

        num_nodes = num_evaluations = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]

        # radius = self.register_module_input(name='propeller_radius', shape=(num_nodes,), units='m')
        rotor_radius = self.register_module_input(name='propeller_radius', shape=(num_nodes,))
        # self.print_var(rotor_radius)
        # rotor_radius = csdl.expand(radius,(num_nodes,))
        ref_radius = self.register_module_input('reference_radius', shape=(num_nodes,))
        ref_chord = self.register_module_input('reference_chord', shape=(num_nodes,))

        omega = self.register_module_input('omega', shape=(num_nodes, ), units='rpm')

        u = self.register_module_input(name='u', shape=(num_nodes, 1), units='m/s') * -1
        v = self.register_module_input(name='v', shape=(num_nodes, 1), units='m/s', val=0) 
        w = self.register_module_input(name='w', shape=(num_nodes, 1), units='m/s', val=0) 
        
        V = self.register_module_output('velocity_vector', shape=(num_nodes,3))
        
        inflow_velocity = self.register_module_output('inflow_velocity', shape=shape + (3,), importance=1)
        
        x_dir = self.register_module_output('x_dir', val=0, shape=(num_evaluations,3))
        y_dir = self.register_module_output('y_dir', val=0, shape=(num_evaluations,3))
        z_dir = self.register_module_output('z_dir', val=0, shape=(num_evaluations,3))
       
        
        R_h = 0.2 * rotor_radius
        self.register_module_output('hub_radius', R_h)
        dr = ((rotor_radius)-(0.2 * rotor_radius))/ (num_radial - 1)
        self.register_module_output('dr',dr)

        BILD_Vt = self.register_module_output('BILD_tangential_inflow_velocity', shape=(num_nodes,))
        ref_sigma = self.register_module_output('reference_blade_solidity', shape=(num_nodes,))

        n = self.register_module_output('rotational_speed', shape=(num_evaluations,))
        thrust_vector = self.register_module_input('thrust_vector', shape=(num_nodes,3))
        for i in range(num_nodes):
            normal_vec = thrust_vector[i,:]
            normal_vec_axial_induced = -1 * normal_vec

            in_plane_1 = projection_vec - csdl.expand(csdl.dot(projection_vec,normal_vec,axis=1),(1,3) ) * normal_vec
            in_plane_ey =  (in_plane_1 / csdl.expand(csdl.pnorm(in_plane_1,pnorm_type=2),(1,3)))
            in_plane_ex = csdl.cross(normal_vec,in_plane_ey, axis=1)

            x_dir[i,0] = self.register_module_input('x_dir_temp', shape=(1, 1), val=1) 
            y_dir[i,1] = self.register_module_input('y_dir_temp', shape=(1, 1), val=1) 
            z_dir[i,2] = self.register_module_input('z_dir_temp', shape=(1, 1), val=1) 
            
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
        

