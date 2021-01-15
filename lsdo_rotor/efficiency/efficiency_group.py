import numpy as np
import omtools.api as ot
import openmdao.api as om
from openmdao.api import  ExplicitComponent, ImplicitComponent
from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.constant.constant_group import ConstantGroup

class EfficiencyGroup(ot.Group): 
    def initialize(self):
        self.options.declare('shape', types=tuple)
        # Expand in different group/comp
    def setup(self):
        shape = self.options['shape']
        
        Vx_scalar = self.declare_input('Vx')
        Vx = ot.expand(Vx_scalar, shape=shape)        
    
        C_scalar = self.declare_input('C')
        C = ot.expand(C_scalar, shape=shape)

        Vt_vec = self.declare_input('Vt_vec', shape=shape)

    
        coeff_4 = -64 * C**2 * Vt_vec**2 - 128 *  C * Vx * Vt_vec**2 - 64 * Vx**2 * Vt_vec**2 - 64 * Vt_vec**4
        coeff_3 = 128 * C**2 * Vt_vec**2 + 288 * C* Vx * Vt_vec**2 + 160 * Vx**2 * Vt_vec**2 + 160 * Vt_vec**4
        coeff_2 = 16 * C**2 * Vx**2 - 80 * C**2 * Vt_vec**2 + 32 * C* Vx**3 - 208 * C* Vx * Vt_vec**2 + 16 * Vx**4 - 116 * Vx**2 * Vt_vec**2 - 132 * Vt_vec**4 
        coeff_1  = -16 * C**2 * Vx**2 + 16 * C**2 * Vt_vec**2 - 40 * C* Vx**3 + 48 * C* Vx * Vt_vec**2 - 24 * Vx**4 + 16 * Vx**2 * Vt_vec**2 + 40 * Vt_vec**4
        coeff_0 = 12 * C* Vx**3 + 8 * Vx**4 - 4 * Vt_vec**4 + 4 * C**2 * Vx**2 + 4 * Vx**2 * Vt_vec**2

        self.register_output('coeff_4', coeff_4)
        self.register_output('coeff_3', coeff_3)
        self.register_output('coeff_2', coeff_2)
        self.register_output('coeff_1', coeff_1)
        self.register_output('coeff_0', coeff_0)

    # class EfficiencyGroup(om.ImplicitComponent):

#     def initialize(self):
#         self.options.declare('shape', types=tuple)

#     def setup(self):
#         shape = self.options['shape']

#         self.add_input('Vt_vec', shape = (20,))
#         self.add_input('Vx', shape = shape)
#         self.add_input('C', shape = shape)

#         self.add_output('coeff_4', shape= (20,))      
#         self.add_output('coeff_3', shape= (20,))
#         self.add_output('coeff_2', shape= (20,))
#         self.add_output('coeff_1', shape= (20,))
#         self.add_output('coeff_0', shape= (20,))

#         self.declare_partials(of='*', wrt='*')

#     def apply_nonlinear(self, inputs, outputs, residuals):
#         Vt_vec = inputs['Vt_vec']
#         Vx = inputs['Vx']
#         C = inputs['C']

#         coeff_4 = outputs['coeff_4']
#         coeff_3 = outputs['coeff_3']
#         coeff_2 = outputs['coeff_2']
#         coeff_1 = outputs['coeff_1']
#         coeff_0 = outputs['coeff_0']
       
        
#         outputs['coeff_4'] = -64 * C**2 * Vt_vec**2 - 128 *  C * Vx * Vt_vec**2 - 64 * Vx**2 * Vt_vec**2 - 64 * Vt_vec**4
#         outputs['coeff_3'] = 128 * C**2 * Vt_vec**2 + 288 * C* Vx * Vt_vec**2 + 160 * Vx**2 * Vt_vec**2 + 160 * Vt_vec**4
#         outputs['coeff_2'] = 16 * C**2 * Vx**2 - 80 * C**2 * Vt_vec**2 + 32 * C* Vx**3 - 208 * C* Vx * Vt_vec**2 + 16 * Vx**4 - 116 * Vx**2 * Vt_vec**2 - 132 * Vt_vec**4 
#         outputs['coeff_1'] = -16 * C**2 * Vx**2 + 16 * C**2 * Vt_vec**2 - 40 * C* Vx**3 + 48 * C* Vx * Vt_vec**2 - 24 * Vx**4 + 16 * Vx**2 * Vt_vec**2 + 40 * Vt_vec**4
#         outputs['coeff_0'] = 12 * C* Vx**3 + 8 * Vx**4 - 4 * Vt_vec**4 + 4 * C**2 * Vx**2 + 4 * Vx**2 * Vt_vec**2


#         def solve_nonlinear(self, inputs, outputs):
#             shape = self.options['shape']

#             Vt_vec = inputs['Vt_vec']
#             Vx = inputs['Vx']
#             C = inputs['C']       

#             # size= len(Vt_vec)
#             # coeff_0 = np.empty(size)
#             # for i in range(size):
#             #     coeff_0[i] = 12 * C* Vx**3 + 8 * Vx**4 - 4 * Vt_vec[i]**4 + 4 * C**2 * Vx**2 + 4 * Vx**2 * Vt_vec[i]**2


#             outputs['coeff_4'] = coeff_4
#             outputs['coeff_3'] = coeff_3
#             outputs['coeff_2'] = coeff_2
#             outputs['coeff_1'] = coeff_1
#             outputs['coeff_0'] = coeff_0