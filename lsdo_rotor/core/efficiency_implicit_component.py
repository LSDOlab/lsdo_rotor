import numpy as np

import openmdao.api as om


class EfficiencyImplicitComponent(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        
    def setup(self):
        shape = self.options['shape']
        
        self.add_input('coeff_4', shape=shape)
        self.add_input('coeff_3', shape=shape)
        self.add_input('coeff_2', shape=shape)
        self.add_input('coeff_1', shape=shape)
        self.add_input('coeff_0', shape=shape)
        self.add_output('_efficiency', shape=shape)

        # self.add_input('p4', shape=shape)
        # self.add_input('p3', shape=shape)
        # self.add_input('p2', shape=shape)
        # self.add_input('p1', shape=shape)
        # self.add_input('p0', shape=shape)
        # self.add_output('_efficiency2', shape=shape)

        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        coeff_4 = inputs['coeff_4']#.flatten()
        coeff_3 = inputs['coeff_3']#.flatten()
        coeff_2 = inputs['coeff_2']#.flatten()
        coeff_1 = inputs['coeff_1']#.flatten()
        coeff_0 = inputs['coeff_0']#.flatten()
        eta = outputs['_efficiency']
    
        outputs['_efficiency'] = coeff_4 * eta ** 4. + coeff_3 * eta ** 3. + coeff_2 * eta ** 2. + coeff_1 * eta + coeff_0

        # p4 = inputs['p4']
        # p3 = inputs['p3']
        # p2 = inputs['p2']
        # p1 = inputs['p1']
        # p0 = inputs['p0']
        # eta2 = outputs['_efficiency2']
    
        # outputs['_efficiency2'] = p4 * eta2 ** 4. + p3 * eta2 ** 3. + p2 * eta2 ** 2. + p1 * eta2 + p0

    def solve_nonlinear(self, inputs, outputs):
        shape = self.options['shape']

        coeff_4 = inputs['coeff_4']#.flatten()
        coeff_3 = inputs['coeff_3']#.flatten()
        coeff_2 = inputs['coeff_2']#.flatten()
        coeff_1 = inputs['coeff_1']#.flatten()
        coeff_0 = inputs['coeff_0']#.flatten()
    
        size = len(coeff_0)
        eta = np.empty((size))

        for i in range(size):
            all_roots = np.roots([
                coeff_4[i],
                coeff_3[i],
                coeff_2[i],
                coeff_1[i],
                coeff_0[i],
            ])
            real_roots = [root for root in all_roots if np.isreal(root)]
            valid_roots = [root for root in real_roots if 0. <= root <= 1.]
            root = np.max(np.array(valid_roots))

            eta[i] = root

        outputs['_efficiency'] = eta.reshape(shape)

    def linearize(self, inputs, outputs, partials):
        coeff_4 = inputs['coeff_4']#.flatten()
        coeff_3 = inputs['coeff_3']#.flatten()
        coeff_2 = inputs['coeff_2']#.flatten()
        coeff_1 = inputs['coeff_1']#.flatten()
        eta = outputs['_efficiency']

        partials['_efficiency', 'coeff_4'] = eta ** 4.
        partials['_efficiency', 'coeff_3'] = eta ** 3.
        partials['_efficiency', 'coeff_2'] = eta ** 2.
        partials['_efficiency', 'coeff_1'] = eta
        partials['_efficiency', 'coeff_0'] = 1.
        partials['_efficiency', '_efficiency'] = 4 * coeff_4 * eta ** 3. + 3 * coeff_3 * eta ** 2. + 2 * coeff_2 * eta + coeff_1

        self.inv_jac = 1.0 / partials['_efficiency', '_efficiency']
       
    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['_efficiency'] = self.inv_jac * d_residuals['_efficiency']
        
        elif mode == 'rev':
            d_residuals['_efficiency'] = self.inv_jac * d_outputs['_efficiency']
            
            
