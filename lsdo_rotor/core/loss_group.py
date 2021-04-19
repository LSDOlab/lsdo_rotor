import numpy as np

import omtools.api as ot
from lsdo_rotor.rotor_parameters import RotorParameters

class LossGroup(ot.Group):
    def initialize(self):
        self.options.declare('rotor', types=RotorParameters)
        self.options.declare('shape', types=tuple)
        self.options.declare('mode', types= int)

    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']
        mode = self.options['mode']

        B = rotor['num_blades']

        radius = self.declare_input('_radius',shape= shape)
        rotor_radius = self.declare_input('_rotor_radius', shape= shape)
        hub_radius = self.declare_input('_hub_radius', shape = shape)

        if mode == 1:
            print('Mode 1')

        elif mode == 2: 
            phi = self.declare_input('_phi_BEMT', shape = shape)

            f_tip = B / 2 * (rotor_radius - radius) / radius / ot.sin(phi)
            f_hub = B / 2 * (radius - hub_radius) / hub_radius / ot.sin(phi)

            F_tip = 2 / np.pi * ot.arccos(ot.exp(-f_tip))
            F_hub = 2 / np.pi * ot.arccos(ot.exp(-f_hub))

            F = F_tip * F_hub
            self.register_output('BEMT_loss_factor', F)




        