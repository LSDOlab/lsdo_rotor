import numpy as np
from csdl import Model
import csdl

from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.pitt_peters.functions.get_pitt_peters_rotor_dictionary import get_pitt_peters_rotor_dictionary

from lsdo_rotor.core.pitt_peters.inputs.pitt_peters_external_inputs_model import PittPetersExternalInputsModel 
from lsdo_rotor.core.pitt_peters.inputs.pitt_peters_core_inputs_model import PittPetersCoreInputsModel
from lsdo_rotor.core.pitt_peters.inputs.pitt_peters_pre_process_model import PittPetersPreprocessModel
from lsdo_rotor.core.pitt_peters.pitt_peters_custom_implicit_operation import PittPetersCustomImplicitOperation
from lsdo_rotor.core.pitt_peters.pitt_peters_post_process_model import PittPetersPostProcessModel

from lsdo_atmos.atmosphere_model import AtmosphereModel

class PittPetersModel(Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('num_radial', types=int, default=20)
        self.parameters.declare('num_tangential', types=int, default=20)
        self.parameters.declare(name='airfoil')

        self.parameters.declare('thrust_vector', types=np.ndarray)
        self.parameters.declare('thrust_origin', types=np.ndarray)
        self.parameters.declare('ref_pt', types=np.ndarray)
        self.parameters.declare('num_blades', types=int)

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        airfoil = self.parameters['airfoil']
        # shape = self.parameters['shape']
        
        reference_point = self.parameters['ref_pt']
        t_v = self.parameters['thrust_vector']
        print(t_v)
        t_o = self.parameters['thrust_origin']
        print(t_o)
        if len(t_v.shape) == 1:
            n = 1
            self.create_input('thrust_vector', shape=(num_nodes,3), val=np.tile(t_v,(num_nodes,1)))
        elif len(t_v.shape) > 2:
            raise ValueError('Thrust vector cannot be a tensor; It must be at most a matrix of size (num_nodes,3')
        elif t_v.shape[1] != 3:
            raise ValueError('Thrust vector matrix must have shape (num_nodes,3')
        else:
            n = t_v.shape[0]
            if n != num_nodes:
                raise ValueError('If number of thrust vectors is greater than 1, it must be equal to num_nodes')
            else:
                self.create_input('thrust_vector',shape=(num_nodes,3),val=t_v)


        if len(t_o.shape) == 1:
            m = 1
            self.create_input('thrust_origin', shape=(num_nodes,3), val=np.tile(t_o,(num_nodes,1)))
        elif len(t_o.shape) > 2:
            raise ValueError('Thrust origin cannot be a tensor; It must be at most a matrix of size (num_nodes,3')
        elif t_o.shape[1] != 3:
            raise ValueError('Thrust origin matrix must have shape (num_nodes,3')
        else:
            m = t_o.shape[0]
            if m != num_nodes:
                raise ValueError('If number of thrust origin vector is greater than 1, it must be equal to num_nodes')
            else:
                self.create_input('thrust_origin',shape=(num_nodes,3),val=t_o)
        
        num_blades = self.parameters['num_blades']
        shape = (num_nodes, num_radial, num_tangential)       
        
        interp = get_surrogate_model(airfoil)
        
        prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')

        self.add(PittPetersExternalInputsModel(
            shape=shape,
        ), name = 'pitt_peters_external_inputs_model')
      
        normal_inflow = self.declare_variable('normal_inflow', shape=(num_nodes,))
        in_plane_inflow = self.declare_variable('in_plane_inflow', shape=(num_nodes,))
        n = self.declare_variable('rotational_speed', shape=(num_nodes,))        
        rotor = get_pitt_peters_rotor_dictionary(airfoil,interp,normal_inflow,in_plane_inflow,n,prop_radius,num_nodes,num_radial, num_tangential)

        self.add(PittPetersCoreInputsModel(
            shape=shape,
        ), name = 'pitt_peters_core_inputs_model')

        self.add(PittPetersPreprocessModel(
            shape=shape,
            num_blades=num_blades,
        ), name = 'pitt_peters_preprocess_model')

        self.add(AtmosphereModel(
            shape=(num_nodes,),
        ), name = 'atmosphere_model')

        chord = self.declare_variable('_chord',shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        W = (Vx**2 + Vt**2)**0.5
        rho_exp = csdl.expand(self.declare_variable('density', shape=(num_nodes,)), shape,'i->ijk')
        self.register_output('_density_expanded',rho_exp)
        rho = self.declare_variable('density', shape=(num_nodes,))
        mu = csdl.expand(self.declare_variable('dynamic_viscosity', shape=(num_nodes,)), shape, 'i->ijk')
        Re = rho_exp * W * chord / mu
        # self.print_var(Re)
        self.register_output('_re_pitt_peters',Re)


        twist_dist = self.declare_variable('_pitch', shape=shape)
        r = self.declare_variable('_radius', shape=shape)
        R = self.declare_variable('_rotor_radius', shape=shape)
        Omega = self.declare_variable('_angular_speed', shape=shape)
        R = self.declare_variable('_rotor_radius', shape= shape)
        dr = self.declare_variable('_dr', shape=shape)
        mu_inflow = self.declare_variable('mu', shape=(num_nodes,))
        mu_z_inflow = self.declare_variable('mu_z', shape=(num_nodes,))
        pitt_peters = csdl.custom(
            Re,
            chord,
            twist_dist,
            Vt,
            dr,
            R,
            r,
            Omega,
            rho_exp,
            rho,
            mu_inflow,
            mu_z_inflow,
            op= PittPetersCustomImplicitOperation(
                shape=shape,
                rotor=rotor,
                num_blades=num_blades,
        ))
        self.register_output('_lambda',pitt_peters)
        # self.add_objective('_lambda')
        # PittPetersCustomImplicitOperation().visualize_sparsity(recursive=True)
        # pitt_peters.visualize_sparsity(recursive=True)


        self.add(PittPetersPostProcessModel(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name = 'pitt_peters_post_process_model_2')

        # Post-Processing
        T = self.declare_variable('T', shape=(num_nodes,))
        F = self.create_output('F', shape=(num_nodes,3))
        M = self.create_output('M', shape=(num_nodes,3))
        thrust_vector = self.declare_variable('thrust_vector', shape=(num_nodes,3))
        thrust_origin = self.declare_variable('thrust_origin', shape=(num_nodes,3))
        ref_pt = self.declare_variable('reference_point',shape=(num_nodes,3),val=np.tile(reference_point,(num_nodes,1)))
        # loop over pt set list 
        for i in range(num_nodes):
            # F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
            F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] #- 9 * hub_drag[i,0]
            F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1]
            F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
            

        moments = csdl.cross(thrust_origin-ref_pt, F, axis=1)
        M[:,0] = moments[:,0]
        M[:,1] = moments[:,1]
        M[:,2] = moments[:,2]
