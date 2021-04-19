import numpy as np

import omtools.api as ot

from lsdo_rotor.rotor_parameters import RotorParameters

class QuadraticAirfoilGroup(ot.Group):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('rotor', types=RotorParameters)
        # self.options.declare('Cl0', default=0.2)
        # self.options.declare('Cla', default=1 * 1.8 * np.pi)
        # self.options.declare('Cdmin', default=0.006)
        # self.options.declare('K', default=0.)
        # self.options.declare('alpha_Cdmin', default=0.4)

    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']

        Cl0 = rotor['Cl0']
        Cla = rotor['Cla']
        Cdmin = rotor['Cdmin']
        K = rotor['K']
        alpha_Cdmin = rotor['alpha_Cdmin']
        alpha_stall = rotor['a_stall_plus']


        alpha = self.declare_input('_alpha',shape=shape)

        Cl = Cl0 + Cla * alpha
        Cd = Cdmin + K * (alpha - alpha_Cdmin)**2

        self.register_output('Cl', Cl)
        self.register_output('Cd', Cd)