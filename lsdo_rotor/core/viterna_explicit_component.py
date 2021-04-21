import numpy as np 

import openmdao.api as om 
import scipy.sparse.linalg as splinalg
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.smoothing_explicit_component import SmoothingExplicitComponent

class ViternaExplicitComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types = tuple)
        self.options.declare('rotor', types = RotorParameters)
    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']
    
        
        self.add_input('_alpha', shape= shape)

        self.add_output('_Cl', shape = shape)
        self.add_output('_Cd', shape = shape)

        

    def setup_partials(self):
        self.declare_partials('*','*')

    def compute(self, inputs, outputs):
        shape = self.options['shape']
        rotor = self.options['rotor']
        
        alpha             = inputs['_alpha']
        eps_plus          = rotor['eps_plus']
        eps_minus         = rotor['eps_minus'] 
        eps_cd            = rotor['eps_cd']
        coeff_Cl_minus    = rotor['coeff_Cl_minus'] 
        # print(coeff_Cl_minus, 'eps_minus')
        coeff_Cl_plus     = rotor['coeff_Cl_plus'] 
        coeff_Cd_minus    = rotor['coeff_Cd_minus'] 
        coeff_Cd_plus     = rotor['coeff_Cd_plus'] 
        A1                = rotor['A1']
        B1                = rotor['B1']
        A2_plus           = rotor['A2_plus']
        B2_plus           = rotor['B2_plus']
        A2_minus          = rotor['A2_minus']
        B2_minus          = rotor['B2_minus']

        alpha_stall_plus  = rotor['a_stall_plus']
        alpha_stall_minus = rotor['a_stall_minus']
        Cl0               = rotor['Cl0']
        Cla               = rotor['Cla']
        Cdmin             = rotor['Cdmin']
        alpha_Cdmin       = rotor['alpha_Cdmin']
        K                 = rotor['K']
        # print(K,'K')

        alpha = alpha.flatten()
        size = len(alpha)
        Cl = np.empty((size))
        Cd = np.empty((size))
        # print(size)
        # print(alpha)
        for i in range(size):
            if alpha[i] < (alpha_stall_minus - eps_minus):
                Cl[i] = A1 * np.sin(2 * alpha[i]) + A2_minus * np.cos(alpha[i])**2 / np.sin(alpha[i])           
            elif (alpha[i] >= (alpha_stall_minus - eps_minus)) and (alpha[i] <= (alpha_stall_minus + eps_minus)):
                Cl[i] = coeff_Cl_minus[0] * alpha[i]**3 + coeff_Cl_minus[1] * alpha[i]**2 + coeff_Cl_minus[2] * alpha[i] + coeff_Cl_minus[3] 
            elif ((alpha_stall_plus - eps_plus) <= alpha[i]) and ((alpha_stall_plus + eps_plus) >= alpha[i]):
                Cl[i] = coeff_Cl_plus[0] * alpha[i]**3 + coeff_Cl_plus[1] * alpha[i]**2 + coeff_Cl_plus[2] * alpha[i] + coeff_Cl_plus[3] 
            elif alpha[i] > (alpha_stall_plus + eps_plus): 
                Cl[i] = A1 * np.sin(2 * alpha[i]) + A2_plus * np.cos(alpha[i])**2 / np.sin(alpha[i])
            else:
                Cl[i] = Cl0 + Cla * alpha[i]
        # else:
        #     Cl = Cl_BEMT

        for i in range(size):
            if alpha[i] < (alpha_stall_minus - eps_cd):
                Cd[i] = B1 * np.sin(alpha[i])**2 + B2_minus * np.cos(alpha[i])
            elif (alpha[i] >= (alpha_stall_minus - eps_cd)) and (alpha[i] <= (alpha_stall_minus + eps_cd)):
                Cd[i] = coeff_Cd_minus[0] * alpha[i]**3 + coeff_Cd_minus[1] * alpha[i]**2 + coeff_Cd_minus[2] * alpha[i] + coeff_Cd_minus[3] 
            elif ((alpha_stall_plus - eps_cd) <= alpha[i]) and ((alpha_stall_plus + eps_cd) >= alpha[i]):
                Cd[i] = coeff_Cd_plus[0] * alpha[i]**3 + coeff_Cd_plus[1] * alpha[i]**2 + coeff_Cd_plus[2] * alpha[i] + coeff_Cd_plus[3] 
            elif alpha[i] > (alpha_stall_plus + eps_cd): 
                Cd[i] = B1 * np.sin(alpha[i])**2 + B2_plus * np.cos(alpha[i])
            else: 
                Cd[i] = Cdmin + K * (alpha[i] - alpha_Cdmin)**2
        # else:
            # Cd = Cd_BEMT

        # print(Cl,'Cl_vit')
        outputs['_Cl'] = Cl.reshape(shape)
        outputs['_Cd'] = Cd.reshape(shape)

        # print('-'*60)
        # print(alpha)
        # print(Cl)
        # print(Cd)



if __name__ == '__main__':

    prob = om.Problem()

    comp = om.IndepVarComp()

    d2r = np.pi/180
    comp.add_output('alpha_stall', val = 12 * np.pi/180)
    comp.add_output('alpha_stall_minus', val = -7 * d2r)
    comp.add_output('Cl_stall', val = 1.5)
    comp.add_output('Cd_stall', val = 0.03)
    comp.add_output('Cl_stall_minus', val = -0.5)
    comp.add_output('Cd_stall_minus', val = 0.025)
    comp.add_output('AR', val =10)
    comp.add_output('smoothing_tolerance', val = 5 *d2r)
    
    
    comp.add_output('_alpha')

    comp.add_output('Cl0', val =0.2)
    comp.add_output('Cla', val = 0.9 * 2 * np.pi)
    comp.add_output('Cdmin', val = 0.002)
    comp.add_output('K', val = 2.1)
    comp.add_output('alpha_Cdmin', val = 0.2 * d2r) 
    prob.model.add_subsystem('inputs', comp, promotes=['*'])
    
    comp = ViternaExplicitComponent(shape=(1,))
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()

    num = 500
    alpha = np.linspace(-np.pi/2., np.pi/2., num)
    CL = np.empty(num)
    CD = np.empty(num)

    for i in range(num):
        prob['_alpha'] = alpha[i]
        prob.run_model()
        CL[i] = prob['_Cl']
        print(CL,'CL')
        CD[i] = prob['_Cd']

    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.plot(alpha / d2r, CL)
    plt.subplot(2, 1, 2)
    plt.plot(alpha / d2r, CD)
    plt.show()

        # self.add_output('_Cl', shape = shape)
        # self.add_output('_Cd', shape = shape)