import numpy as np
from csdl import Model
import csdl
# from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters



class BEMAirfoilSurrogateModelGroup(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=BEMRotorParameters)
    
    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']

        if not rotor['custom_polar']:
            self.add_input('Re', shape=shape)
            self.add_input('alpha_distribution', shape=shape)
            # self.add_input('AoA', shape=shape)
            self.add_input('_chord', shape=shape)
            

            self.add_output('Cl', shape=shape)
            # self.add_output('Cl_2', shape=shape)
            
            self.add_output('Cd', shape=shape)
            # self.add_output('Cd_2', shape=shape)

            indices = np.arange(shape[0] * shape[1] * shape[2])
            # print(indices,'INDICES')
            # self.declare_derivatives('Cl', 'Re')
            # self.declare_derivatives('Cl', 'alpha_distribution')
            self.declare_derivatives('Cl', 'Re', rows=indices, cols=indices)
            self.declare_derivatives('Cl', 'alpha_distribution', rows=indices, cols=indices)
        
            # self.declare_derivatives ('Cl_2', 'Re', rows=indices, cols=indices)
            # self.declare_derivatives ('Cl_2', 'AoA', rows=indices, cols=indices)
            
            # self.declare_derivatives('Cd', 'Re')
            # self.declare_derivatives('Cd', 'alpha_distribution')
            self.declare_derivatives('Cd', 'Re', rows=indices, cols=indices)
            self.declare_derivatives('Cd', 'alpha_distribution', rows=indices, cols=indices)
            
            # self.declare_derivatives ('Cd_2', 'Re', rows=indices, cols=indices)
            # self.declare_derivatives ('Cd_2', 'AoA', rows=indices, cols=indices)

            self.x_1 = np.zeros((shape[0] * shape[1] * shape[2], 2))
            # self.x_2 = np.zeros((shape[0] * shape[1] * shape[2], 2))

        else:
            self.add_input('Re', shape=shape)
            self.add_input('alpha_distribution', shape=shape)
            self.add_input('_chord', shape=shape)
            self.add_output('Cl', shape=shape)
            self.add_output('Cd', shape=shape)

            indices = np.arange(shape[0] * shape[1] * shape[2])

            self.declare_derivatives('Cl', 'alpha_distribution', rows=indices, cols=indices)
            self.declare_derivatives('Cd', 'alpha_distribution', rows=indices, cols=indices)
        

    def compute(self, inputs, outputs):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        interp = rotor['interp']

        if not rotor['custom_polar']:
            chord = inputs['_chord'].flatten()
            alpha = inputs['alpha_distribution'].flatten()
            Re = inputs['Re'].flatten()
            # AoA         = inputs['AoA'].flatten()

            self.x_1[:, 0] = alpha
            self.x_1[:, 1] = Re/2e6

            # self.x_2[:, 0] = AoA
            # self.x_2[:, 1] = Re/2e6
    
            y_1 = interp.predict_values(self.x_1).reshape((shape[0], shape[1], shape[2], 2))
            # y_2 = interp.predict_values(self.x_2).reshape((shape[0] , shape[1] , shape[2], 2))

            # print( y_1[:,:,:,0])
            outputs['Cl'] = y_1[:,:,:,0]
            outputs['Cd'] = y_1[:,:,:,1]

            # outputs['Cl_2'] = y_2[:,:,:,0]
            # outputs['Cd_2'] = y_2[:,:,:,1]

        else:
            alpha = inputs['alpha_distribution'].flatten()
            Cl, Cd = interp.predict_values(alpha)
            outputs['Cl'] = Cl.reshape(shape[0], shape[1], shape[2])
            outputs['Cd'] = Cd.reshape(shape[0], shape[1], shape[2])

    def compute_derivatives(self, inputs, derivatives):
        rotor       = self.parameters['rotor']
        interp      = rotor['interp']
        
        if not rotor['custom_polar']:
            alpha       = inputs['alpha_distribution'].flatten()
            Re          = inputs['Re'].flatten()
            # AoA         = inputs['AoA'].flatten()
        
            self.x_1[:, 0] = alpha
            self.x_1[:, 1] = Re/2e6

            dy_dalpha = interp.predict_derivatives(self.x_1, 0)
            dy_dRe = interp.predict_derivatives(self.x_1, 1)

            derivatives['Cl', 'alpha_distribution'] = dy_dalpha[:, 0]
            derivatives['Cd', 'alpha_distribution'] = dy_dalpha[:, 1]

            derivatives['Cl', 'Re'] = dy_dRe[:, 0] /2e6
            derivatives['Cd', 'Re'] = dy_dRe[:, 1] /2e6

        else:
            alpha = inputs['alpha_distribution'].flatten()
            dCl_daoa, dCd_daoa = interp.predict_derivatives(alpha)
            derivatives['Cl', 'alpha_distribution'] = dCl_daoa
            derivatives['Cd', 'alpha_distribution'] = dCd_daoa

