import numpy as np

from csdl import Model
import csdl


class QuadraticAirfoilGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('Cl0', default=0.)
        self.parameters.declare('Cl1', default=2 * np.pi)
        self.parameters.declare('Cd0', default=0.005)
        self.parameters.declare('Cd1', default=0.)
        self.parameters.declare('Cd2', default=0.5)

    def define(self):
        Cl0 = self.parameters['Cl0']
        Cl1 = self.parameters['Cl1']
        Cd0 = self.parameters['Cd0']
        Cd1 = self.parameters['Cd1']
        Cd2 = self.parameters['Cd2']

        alpha = self.declare_variable('alpha')

        Cl = Cl0 + Cl1 * alpha
        Cd = Cd0 + Cd1 * alpha + Cd2 * alpha**2

        self.register_output('Cl', Cl)
        self.register_output('Cd', Cd)
