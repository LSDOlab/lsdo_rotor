import numpy as np

import omtools.api as ot


class QuadraticAirfoilGroup(ot.Group):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('Cl0', default=0.)
        self.options.declare('Cl1', default=2 * np.pi)
        self.options.declare('Cd0', default=0.005)
        self.options.declare('Cd1', default=0.)
        self.options.declare('Cd2', default=0.5)

    def setup(self):
        Cl0 = self.options['Cl0']
        Cl1 = self.options['Cl1']
        Cd0 = self.options['Cd0']
        Cd1 = self.options['Cd1']
        Cd2 = self.options['Cd2']

        alpha = self.declare_input('alpha')

        Cl = Cl0 + Cl1 * alpha
        Cd = Cd0 + Cd1 * alpha + Cd2 * alpha ** 2

        self.register_output('Cl', Cl)
        self.register_output('Cd', Cd)