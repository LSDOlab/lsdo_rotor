import numpy as np
from csdl import Model
import csdl
from lsdo_rotor.bemt.bemt_group import BEMTGroup
from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.polar.polar_group import PolarGroup


class InducedVelocityGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):

        phi = self.declare_variable('phi')
        Vt = self.declare_variable('Vt')
        Vx = self.declare_variable('Vx')
        Ct = self.declare_variable('Ct')
        Cx = self.declare_variable('Cx')
        sigma = self.declare_variable('sigma')

        ux = Vt * csdl.sin(
            2 * phi) * Ct / (Cx * (csdl.sin(2 * phi) + 0.5 * sigma * Ct))
        ut = 2 * Vt * sigma * Ct / (2 * csdl.sin(2 * phi) + sigma * Ct)

        self.register_output('ux', ux)
        self.register_output('ut', ut)
