import numpy as np
from csdl import Model
import csdl

class QuarticSolverGroup(Model):

    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']

        model = Model()

        coeff_4 =  model.declare_variable('coeff_4', shape=shape)
        coeff_3 =  model.declare_variable('coeff_3', shape=shape)
        coeff_2 =  model.declare_variable('coeff_2', shape=shape)
        coeff_1 =  model.declare_variable('coeff_1', shape=shape)
        coeff_0 =  model.declare_variable('coeff_0', shape=shape)

        eta = model.declare_variable('eta_2', shape=shape)

        R = coeff_4 * eta ** 4. + coeff_3 * eta ** 3. + coeff_2 * eta ** 2. + coeff_1 * eta + coeff_0
        model.register_output('quartic_residual',R)

        eps = 1e-6
        solve_quartic_function = self.create_implicit_operation(model)
        solve_quartic_function.declare_state('eta_2', residual='quartic_residual', bracket=(0.5,1-eps))
        
        coeff_4 =  self.declare_variable('coeff_4', shape=shape)
        coeff_3 =  self.declare_variable('coeff_3', shape=shape)
        coeff_2 =  self.declare_variable('coeff_2', shape=shape)
        coeff_1 =  self.declare_variable('coeff_1', shape=shape)
        coeff_0 =  self.declare_variable('coeff_0', shape=shape)

        eta_2 = solve_quartic_function(coeff_4,coeff_3,coeff_2,coeff_1,coeff_0)
        