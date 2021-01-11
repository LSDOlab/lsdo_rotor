import numpy as np
import omtools.api as ot
import cmath

from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.constant.constant_group import ConstantGroup

# class EfficiencyGroup(ot.Group): 

class EfficiencyGroup(ot.ImplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        g = self.group
        
        eta = g.create_implicit_output('eta', shape=shape)
        Vx = g.declare_input('Vx')
        Vt = g.declare_input('Vt')
        C = g.declare_input('C')

        a = -64 * C**2 * Vt**2 - 128 * C * Vx * Vt**2 - 64 * Vx**2 * Vt**2 - 64 * Vt**4
        b = 128 * C**2 * Vt**2 + 288 * C * Vx * Vt**2 + 160 * Vx**2 * Vt**2 + 160 * Vt**4
        c = 16 * C**2 * Vx**2 - 80 * C**2 * Vt**2 + 32 * C * Vx**3 - 208 * C * Vx * Vt**2 + 16 * Vx**4 - 116 * Vx**2 * Vt**2 - 132 * Vt**4 
        d = -16 * C**2 * Vx**2 + 16 * C**2 * Vt**2 - 40 * C * Vx**3 + 48 * C * Vx * Vt**2 - 24 * Vx**4 + 16 * Vx**2 * Vt**2 + 40 * Vt**4
        e = 12 * C * Vx**3 + 8 * Vx**4 - 4 * Vt**4 + 4 * C**2 * Vx**2 + 4 * Vx**2 * Vt**2

        residual = a * eta**4 + b * eta**3 + c * eta**2 + d * eta + e

        eta.define_residual_bracketed(
            residual,
            x1=0.,
            x2=np.pi / 2.,
        )

        ##Here I was trying to use the closed form solution to a quartic equation but ran into some trouble on lines 53 & 54

        # self.register_output('a', a)
        # self.register_output('b', b)
        # self.register_output('c', c)
        # self.register_output('d', d)
        # self.register_output('e', e)

        # p = (8 * a * c - 3 * b**2) / (8 * a**2)
        # q  = (b**3 - 4 * a * b * c + 8 * a**2 * d)/ (8 * a**3)

        # delta_0 = c**2 - 3 * b * d + 12 * a * e
        # delta_1 = 2 * c**3 - 9 * b * c * d + 27 * b**2 * e + 27 * a * d**2 - 72 * a * c * e

        # # Q = ((delta_1 + cmath.sqrt(delta_1**2 - 4 * delta_0**3)) / 2)**(1/3) # This gave me some strange error
        # S = 0.5 * -2/3 * p + 1/(3 * a) * (Q + delta_0/Q) #

        # self.register_output('p',p)
        # self.register_output('q',q)
        # self.register_output('delta_0',delta_0)
        # self.register_output('delta_1',delta_1)
        # self.register_output('Q',Q)
        # self.register_output('S',S)