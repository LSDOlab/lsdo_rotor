import numpy as np
from lsdo_modules.module.module_maker import ModuleMaker


class BILDQuarticSolverMaker(ModuleMaker):

    def initialize_module(self):
        self.parameters.declare('shape', types=tuple)

    def define_module(self):
        shape = self.parameters['shape']

        module = ModuleMaker()

        coeff_4 =  module.register_module_input('coeff_4', shape=shape)
        coeff_3 =  module.register_module_input('coeff_3', shape=shape)
        coeff_2 =  module.register_module_input('coeff_2', shape=shape)
        coeff_1 =  module.register_module_input('coeff_1', shape=shape)
        coeff_0 =  module.register_module_input('coeff_0', shape=shape)

        eta = module.register_module_input('eta_2', shape=shape)

        R = coeff_4 * eta ** 4. + coeff_3 * eta ** 3. + coeff_2 * eta ** 2. + coeff_1 * eta + coeff_0
        module.register_module_output('quartic_residual',R)

        eps = 1e-6
        solve_quartic_function = self.create_implicit_operation(module)
        solve_quartic_function.declare_state('eta_2', residual='quartic_residual', bracket=(0.5,1-eps))
        
        coeff_4 =  self.register_module_input('coeff_4', shape=shape)
        coeff_3 =  self.register_module_input('coeff_3', shape=shape)
        coeff_2 =  self.register_module_input('coeff_2', shape=shape)
        coeff_1 =  self.register_module_input('coeff_1', shape=shape)
        coeff_0 =  self.register_module_input('coeff_0', shape=shape)

        eta_2 = solve_quartic_function(coeff_4,coeff_3,coeff_2,coeff_1,coeff_0)
        self.add_module(module, 'imp_mod_brack_eta_2')
        
        