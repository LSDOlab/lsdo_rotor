import numpy as np
from csdl import Model
import csdl


class ExternalInputsModel(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']
        shape = (shape[0], shape[1], shape[2])

        num_nodes = num_evaluations = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]

        # Dynamic inflow    
        m = np.array([128/75/np.pi, -16/45/np.pi, -16/45/np.pi])
        M  = np.diag(m)
        M_inv = np.linalg.inv(M)
        M_block_list = []
        M_inv_block_list = []
        for i in range(num_evaluations):
            M_block_list.append(M)
            M_inv_block_list.append(M_inv)
        from scipy.linalg import block_diag
        M_block_diagonal = block_diag(*M_block_list)
        M_inv_block_diagonal = block_diag(*M_inv_block_list)

        self.create_input('M_block_diag_matrix', val=M_block_diagonal)
        self.create_input('M_inv_block_diag_matrix', val=M_inv_block_diagonal)

        nu_0_vec = np.random.randn(num_evaluations,)
        nu_s_vec = np.random.randn(num_evaluations,)
        nu_c_vec = np.random.randn(num_evaluations,)

        nu_vec = np.random.randn(num_evaluations,3)
        nu_vec = nu_vec.reshape((num_evaluations,3,1))

        self.create_input('M', val = M)
        self.create_input('M_inv', val = M_inv)
        self.create_input('nu_0_vec', val = nu_0_vec)
        self.create_input('nu_s_vec', val = nu_s_vec)
        self.create_input('nu_c_vec', val = nu_c_vec)

        self.create_input('nu_vec', val = nu_vec)

        # ILDM
        self.create_input('reference_radius', shape=(num_evaluations,))
        self.create_input('reference_chord', shape=(num_evaluations,))
        self.create_input('ildm_axial_inflow_velocity', shape=(num_evaluations,))
        self.create_input('reference_blade_solidity', shape =(num_evaluations,))
        self.create_input('ildm_tangential_inflow_velocity', shape=(num_evaluations,))
        self.create_input('ildm_rotational_speed',shape=(num_evaluations,))

        # General
        self.create_input('hub_radius', shape=(num_evaluations,))
        self.create_input('rotor_radius', shape=(num_evaluations,))
        self.create_input('rotational_speed', shape = (num_evaluations,))
        self.create_input('dr', shape =(num_evaluations,))
        
        self.create_input('position', shape=(num_evaluations, 3))
        self.create_input('x_dir', shape=(num_evaluations, 3))
        self.create_input('y_dir', shape=(num_evaluations, 3))
        self.create_input('z_dir', shape=(num_evaluations, 3))
        self.create_input('inflow_velocity', shape=shape + (3,))
        

        self.create_input('pitch', shape = (num_radial,))
        self.create_input('chord', shape = (num_radial,))

        self.add_design_variable('chord',lower = 0.002, upper = 0.25)
        self.add_design_variable('pitch',lower = 5*np.pi/180, upper = 85*np.pi/180)

        # self.create_input('pitch_cp', shape = (pitch_num_cp,))
        # self.create_input('chord_cp', shape = (chord_num_cp,))
