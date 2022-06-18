import numpy as np
from csdl import Model
import csdl
from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters
import openmdao.api as om

class ILDMAirfoilParametersModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=BEMRotorParameters)

    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']

        self.add_input('Re_ildm', shape = (shape[0],))
        
        self.add_output('Cl_max_ildm', shape = (shape[0],))
        self.add_output('Cd_min_ildm', shape = (shape[0],))
        self.add_output('alpha_max_LD', shape = (shape[0],))

        self.x = np.zeros((shape[0], 2))

        
    def compute(self, inputs, outputs):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        interp = rotor['interp']

        Re = inputs['Re_ildm'].flatten()
        alpha_range                     = np.linspace(-2*np.pi/180,10*np.pi/180,100)
        Re_alpha_design_space           = np.zeros((len(alpha_range),2))
        Re_alpha_design_space[:,0]      = alpha_range
        
        alpha_max_LD                    = np.zeros((shape[0],))

        for i in range(shape[0]):
            Re_alpha_design_space[:,1]      = Re[i]/2e6
            
            Re_alpha_prediction             = interp.predict_values(Re_alpha_design_space)
            LD_design_space                 = Re_alpha_prediction[:,0] / Re_alpha_prediction[:,1]
            LD_max                          = np.max(LD_design_space)
            # print(LD_max, 'max_LD')

            alpha_LD_max_index              = np.where(LD_design_space == LD_max)
            alpha_max_LD[i]                 = alpha_range[alpha_LD_max_index]
            # print(alpha_max_LD)

            max_Re_alpha_combination        = np.array([alpha_max_LD[i] , Re[i]/2e6],dtype=object)
            max_Re_alpha_combination        = max_Re_alpha_combination.reshape((1,2))
            
            self.x[i,:]                     =  max_Re_alpha_combination[0,:]

            
        Cl_Cd_prediction                = interp.predict_values(self.x)
        
        outputs['Cl_max_ildm']          = Cl_Cd_prediction[:,0]
        outputs['Cd_min_ildm']          = Cl_Cd_prediction[:,1]
        outputs['alpha_max_LD']         = alpha_max_LD

        # TO DO: Derivatives!

