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
        self.parameters.declare(name='airfoil', default='NACA_4412')

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
        
        thrust_vector = self.parameters['thrust_vector']
        thrust_origin = self.parameters['thrust_origin']
        ref_pt = self.parameters['ref_pt']
        num_blades = self.parameters['num_blades']

        shape = (num_nodes, num_radial, num_tangential)       
        
        interp = get_surrogate_model(airfoil)
        
        prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')

        # Inputs changing across conditions (segments)
        omega = self.declare_variable('omega', shape=(num_nodes, ), units='rpm')

        self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s', val=1)
        self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s', val=0)
        self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s', val=0)

        self.declare_variable(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
        self.declare_variable(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
        self.declare_variable(name='r', shape=(num_nodes, 1), units='rad/s', val=0)

        self.declare_variable(name='Phi', shape=(num_nodes, 1), units='rad', val=0)
        self.declare_variable(name='Theta',shape=(num_nodes, 1), units='rad', val=0)
        self.declare_variable(name='Psi', shape=(num_nodes, 1), units='rad', val=0)

        self.declare_variable(name='x', shape=(num_nodes, ), units='m', val=0)
        self.declare_variable(name='y', shape=(num_nodes, ), units='m', val=0)
        self.declare_variable(name='z', shape=(num_nodes, ), units='m', val=0)

        self.add(PittPetersExternalInputsModel(
            shape=shape,
            thrust_vector=thrust_vector,
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
        self.print_var(T)
        F = self.create_output('F', shape=(num_nodes,3))
        M = self.create_output('M', shape=(num_nodes,3))
        n = self.declare_variable('normal_vector', shape=(1,3))
        for i in range(num_nodes):
            F[i,:] = csdl.expand(T[i],(1,3)) * n
            M[i,0] = F[i,2] * (thrust_origin[1] - ref_pt[1])
            M[i,1] = F[i,2] * (thrust_origin[0] - ref_pt[0])
            M[i,2] = F[i,0] * (thrust_origin[1] - ref_pt[1])
