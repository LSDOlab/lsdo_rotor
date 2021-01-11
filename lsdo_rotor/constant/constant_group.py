import numpy as np
import omtools.api as ot
from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.induced_velocity.induced_velocity_group import InducedVelocityGroup

class ConstantGroup(ot.Group):
    def initialize(self):
        self.options.declare('shape', types=tuple)
    
    def setup(self):

        Vt = self.declare_input('Vt')
        Vx = self.declare_input('Vx')
        ux = self.declare_input('ux')
        ut = self.declare_input('ut')
        
        C = Vt * ut /(2 * (2 * ux - Vx)) + Vt * ux /(Vt - ut) - Vx

        self.register_output('C', C)
        