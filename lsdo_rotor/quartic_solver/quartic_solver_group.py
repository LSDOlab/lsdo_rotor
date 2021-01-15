import numpy as np
from openmdao.api import Group, ImplicitComponent, NewtonSolver, DirectSolver
import omtools.api as ot
import openmdao.api as om
from omtools.api import Group, ImplicitComponent
from lsdo_rotor.efficiency.efficiency_group import EfficiencyGroup

class BracketedImplicitComp(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        
    def setup(self):
        shape = self.options['shape']
        
        self.add_input('coeff_4', shape=shape)
        self.add_input('coeff_3', shape=shape)
        self.add_input('coeff_2', shape=shape)
        self.add_input('coeff_1', shape=shape)
        self.add_input('coeff_0', shape=shape)
        self.add_output('eta', shape=shape)

        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        coeff_4 = inputs['coeff_4']
        coeff_3 = inputs['coeff_3']
        coeff_2 = inputs['coeff_2']
        coeff_1 = inputs['coeff_1']
        coeff_0 = inputs['coeff_0']
        eta = outputs['eta']
    
        outputs['eta'] = coeff_4 * eta ** 4. + coeff_3 * eta ** 3. + coeff_2 * eta ** 2. + coeff_1 * eta + coeff_0

    def solve_nonlinear(self, inputs, outputs):
        shape = self.options['shape']

        coeff_4 = inputs['coeff_4'].flatten()
        coeff_3 = inputs['coeff_3'].flatten()
        coeff_2 = inputs['coeff_2'].flatten()
        coeff_1 = inputs['coeff_1'].flatten()
        coeff_0 = inputs['coeff_0'].flatten()
    
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

        outputs['eta'] = eta.reshape(shape)

    def linearize(self, inputs, outputs, partials):
        coeff_4 = inputs['coeff_4']
        coeff_3 = inputs['coeff_3']
        coeff_2 = inputs['coeff_2']
        coeff_1 = inputs['coeff_1']
        coeff_0 = inputs['coeff_0']
        eta = outputs['eta']
    
        # outputs['eta'] = coeff_4 * eta ** 4. + coeff_3 * eta ** 3. + coeff_2 * eta ** 2. + coeff_1 * eta + coeff_0

        partials['eta', 'coeff_4'] = eta ** 4.
        partials['eta', 'coeff_3'] = eta ** 3.
        partials['eta', 'coeff_2'] = eta ** 2.
        partials['eta', 'coeff_1'] = eta
        partials['eta', 'coeff_0'] = 1.
        partials['eta', 'eta'] = 4 * coeff_4 * eta ** 3. + 3 * coeff_3 * eta ** 2. + 2 * coeff_2 * eta + coeff_1

        self.inv_jac = 1.0 / partials['eta', 'eta']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']