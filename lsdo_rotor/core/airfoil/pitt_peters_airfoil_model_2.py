import numpy as np
from csdl import Model
import csdl
# # from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.pitt_peters.pitt_peters_rotor_parameters import PittPetersRotorParameters
import openmdao.api as om



class PittPetersAirfoilModel2(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=PittPetersRotorParameters)
    
    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        
        self.add_input('_re_pitt_peters', shape=shape)
        self.add_input('alpha_distribution', shape=shape)
        # self.add_input('_chord', shape=shape)
        

        self.add_output('Cl', shape=shape)
        self.add_output('Cd', shape=shape)

        indices = np.arange(shape[0] * shape[1] * shape[2])
        # print(indices,'INDICES')
        # self.declare_derivatives('Cl', 'Re')
        # self.declare_derivatives('Cl', 'alpha_distribution')
        self.declare_derivatives('Cl', '_re_pitt_peters', rows=indices, cols=indices)
        self.declare_derivatives('Cl', 'alpha_distribution', rows=indices, cols=indices)
       
        
        # self.declare_derivatives('Cd', 'Re')
        # self.declare_derivatives('Cd', 'alpha_distribution')
        self.declare_derivatives('Cd', '_re_pitt_peters', rows=indices, cols=indices)
        self.declare_derivatives('Cd', 'alpha_distribution', rows=indices, cols=indices)
        

        self.x_1 = np.zeros((shape[0] * shape[1] * shape[2], 2))
        # self.x_2 = np.zeros((shape[0] * shape[1] * shape[2], 2))

        

    def compute(self, inputs, outputs):
        shape       = self.parameters['shape']
        rotor       = self.parameters['rotor']
        interp      = rotor['interp']

        alpha       = inputs['alpha_distribution'].flatten()
        Re          = inputs['_re_pitt_peters'].flatten()
        # print(Re)
        # AoA         = inputs['AoA'].flatten()

        self.x_1[:, 0] = alpha
        self.x_1[:, 1] = Re/2e6


 
        y_1 = interp.predict_values(self.x_1).reshape((shape[0] , shape[1] , shape[2], 2))
    
        outputs['Cl'] = y_1[:,:,:,0]
        outputs['Cd'] = y_1[:,:,:,1]



    def compute_derivatives(self, inputs, derivatives):
        rotor       = self.parameters['rotor']
        interp      = rotor['interp']
        
        alpha       = inputs['alpha_distribution'].flatten()
        Re          = inputs['_re_pitt_peters'].flatten()
        # AoA         = inputs['AoA'].flatten()
       
        self.x_1[:, 0] = alpha
        self.x_1[:, 1] = Re/2e6


        dy_dalpha = interp.predict_derivatives(self.x_1, 0)
        dy_dRe = interp.predict_derivatives(self.x_1, 1)

        derivatives['Cl', 'alpha_distribution'] = dy_dalpha[:, 0]
        derivatives['Cd', 'alpha_distribution'] = dy_dalpha[:, 1]

        derivatives['Cl', '_re_pitt_peters'] = dy_dRe[:, 0] /2e6
        derivatives['Cd', '_re_pitt_peters'] = dy_dRe[:, 1] /2e6


