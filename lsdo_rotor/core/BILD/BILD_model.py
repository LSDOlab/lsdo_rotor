

import numpy as np
from csdl import Model
import csdl

from lsdo_rotor.core.BILD.inputs.BILD_external_inputs_model import BILDExternalInputsModel
from lsdo_rotor.core.BILD.inputs.BILD_core_inputs_model import BILDCoreInputsModel
from lsdo_rotor.core.BILD.inputs.BILD_pre_process_model import BILDPreprocessModel

from lsdo_rotor.core.BILD.BILD_airfoil_parameters_model import BILDAirfoilParametersModel
from lsdo_rotor.core.BILD.BILD_phi_bracketed_search_model import BILDPhiBracketedSearchModel
from lsdo_rotor.core.BILD.BILD_induced_velocity_model import BILDInducedVelocityModel
from lsdo_rotor.core.BILD.BILD_quartic_coefficient_model import BILDQuarticCoeffsModel
from lsdo_rotor.core.BILD.BILD_quartic_solver_model import BILDQuarticSolverModel
from lsdo_rotor.core.BILD.BILD_back_comp_model import BILDBackCompModel

from lsdo_atmos.atmosphere_model import AtmosphereModel

from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.BILD.functions.get_BILD_rotor_dictionary import get_BILD_rotor_dictionary

class BILDModel(Model):

    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=30)
        self.parameters.declare(name='airfoil', default='NACA_4412')
        # self.parameters.declare('shape', types=tuple)

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
        
        thrust_vector = self.parameters['thrust_vector']
        thrust_origin = self.parameters['thrust_origin']
        ref_pt = self.parameters['ref_pt']

        num_blades = self.parameters['num_blades']
        
        shape = (num_nodes, num_radial, num_tangential)       
        
        interp = get_surrogate_model(airfoil)
        rotor = get_BILD_rotor_dictionary(airfoil,interp)

        prop_radius = self.declare_variable(name='propeller_radius', shape=(1, ), units='m')

        self.add(BILDExternalInputsModel(
            shape=shape,
            thrust_vector=thrust_vector,
            num_blades=num_blades
        ), name='BILD_external_inputs_model')#, promotes = ['*'])

        self.add(BILDCoreInputsModel(
            shape=shape,
        ),name='BILD_core_inputs_model')

        self.add(BILDPreprocessModel(
            shape=shape,
            num_blades=num_blades,
        ),name='BILD_pre_process_model')

        self.add(AtmosphereModel(
            shape=(num_nodes,),
        ),name='atmosphere_model')

    
        ref_chord = self.declare_variable('reference_chord',shape=(num_nodes,))
        Vx = self.declare_variable('u', shape=(num_nodes,))
        Vt = self.declare_variable('BILD_tangential_inflow_velocity', shape=(num_nodes,))
        W = (Vx**2 + Vt**2)**0.5
        rho = self.declare_variable('density', shape=(num_nodes,))
        mu = self.declare_variable('dynamic_viscosity', shape=(num_nodes,))
        Re = rho * W * ref_chord / mu
        self.register_output('Re_BILD',Re)

        BILD_parameters = csdl.custom(Re, op= BILDAirfoilParametersModel(
            rotor=rotor,
            shape=shape,
        ))
        self.register_output('Cl_max_BILD',BILD_parameters[0])
        self.register_output('Cd_min_BILD',BILD_parameters[1])
        self.register_output('alpha_max_LD',BILD_parameters[2])

        self.add(BILDPhiBracketedSearchModel(
                shape=shape,
                num_blades=num_blades,
            ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

        self.add(BILDInducedVelocityModel(
            num_blades=num_blades,
            shape=shape,
        ), name = 'induced_velocity_model')#, promotes = ['*'])

        self.add(BILDQuarticCoeffsModel(
            shape=shape,
        ), name = 'quartic_coefficient_group')#, promotes = ['*'])

        self.add(BILDQuarticSolverModel(
            shape=shape,
        ), name = 'quartic_solver_group')#, promotes = ['*'])

        self.add(BILDBackCompModel(
            shape=shape,
            num_blades=num_blades,
        ), name = 'BILD_back_comp_group')#, promotes = ['*'])

        # # Post-Processing
        # T = self.declare_variable('T', shape=(num_nodes,))
        # F = self.create_output('F', shape=(num_nodes,3))
        # M = self.create_output('M', shape=(num_nodes,3))
        # n = self.declare_variable('thrust_vector', shape=(1,3))
        # for i in range(num_nodes):
        #     F[i,:] = csdl.expand(T[i],(1,3)) * n
        #     M[i,0] = F[i,2] * (thrust_origin[1] - ref_pt[1])
        #     M[i,1] = F[i,2] * (thrust_origin[0] - ref_pt[0])
        #     M[i,2] = F[i,0] * (thrust_origin[1] - ref_pt[1])



    



            
            
