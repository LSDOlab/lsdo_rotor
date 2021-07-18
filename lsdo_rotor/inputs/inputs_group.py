import numpy as np
from csdl import Model
import csdl


class InputsGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        twist = self.create_input('twist', val=50. * np.pi / 180.)
        Vx = self.create_input('Vx', val=50)

        init_dist = self.declare_variable(
            'init_dist', val=0.5
        )  # initial normalized distance from hub (i.e. percent radius)
        num_blades = self.declare_variable('num_blades', val=3)
        radius = self.declare_variable('radius', val=1)
        chord = self.declare_variable('chord', val=0.1)
        RPM = self.declare_variable('RPM', val=1500.)

        sigma = num_blades * chord / (2 * np.pi * init_dist * radius)
        self.register_output('sigma', sigma)

        Vt = RPM * 2 * np.pi * init_dist * radius / (60)
        self.register_output('Vt', Vt)
