import numpy as np
from csdl import Model
import csdl
from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.induced_velocity.induced_velocity_group import InducedVelocityGroup


class ConstantGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):

        Vt = self.declare_variable('Vt')
        Vx = self.declare_variable('Vx')
        ux = self.declare_variable('ux')
        ut = self.declare_variable('ut')

        C = Vt * ut / (2 * (2 * ux - Vx)) + Vt * ux / (Vt - ut) - Vx

        self.register_output('C', C)
