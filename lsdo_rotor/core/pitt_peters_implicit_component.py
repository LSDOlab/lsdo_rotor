import numpy as np
from csdl import Model
import csdl
from lsdo_rotor.rotor_parameters import RotorParameters
import openmdao.api as om

# class PittPetersImplicitComponent(csdl.CustomImplicitOperation):
class PittPetersImplicitComponent(csdl.CustomImplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=RotorParameters)
    
    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        
        # self.add_input('nu_state_vec', shape = (shape[0],3,1))
        self.add_input('L_matrix', shape=(shape[0],3,3))
        self.add_input('inv_L_matrix', shape=(shape[0],3,3))
        self.add_input('M_matrix', shape=(shape[0],3,3))
        self.add_input('inv_M_matrix', shape=(shape[0],3,3))
        self.add_input('C', shape=(shape[0],3,1))


        self.add_output('nu',shape = (shape[0],3,1))
    
    # def solve_residual_equations(self, inputs, outputs):
    def solve_residual_equations(self, inputs,outputs):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        
        L = inputs['L_matrix']
        L_inv = inputs['inv_L_matrix']
        M = inputs['M_matrix']
        M_inv = inputs['inv_M_matrix']
        C = inputs['C']

        # print(L)
        # print(M)

        term1 = 1 * np.einsum(
                'ijk,ikl->ijl',
                L,
                M,
                )
        print(term1)
        term3 = np.einsum(
                'ijk,ikl->ijl',
                M_inv,
                L_inv,
            )
        print(term3,'TERM3')
        nu = np.zeros((shape[0],3,1))
        for i in range(10):
            
            term2 = np.einsum(
                'ijk,ikl->ijl',
                M_inv,
                C,
                ) 
            
            term4 = np.einsum(
                'ijk,ikl->ijl',
                term3,
                nu,
            )
            term5 = term2 - term4
            term6 = np.einsum(
                'ijk,ikl->ijl',
                term1,
                term5,
            )
            nu_new = nu + term6
            nu = nu_new
            # print(term6)


        

        # for i in range(20):
        

        outputs['nu'] = nu
            
            