import numpy as np
import omtools.api as ot
from lsdo_rotor.bemt.bemt_group import BEMTGroup
from lsdo_rotor.inputs.inputs_group import InputsGroup
from lsdo_rotor.polar.polar_group import PolarGroup

class InducedVelocityGroup(ot.Group):
    def initialize(self):
        self.options.declare('shape', types=tuple)
    
    def setup(self):

        phi = self.declare_input('phi')
        Vt = self.declare_input('Vt')
        Vx = self.declare_input('Vx')
        Ct = self.declare_input('Ct')
        Cx = self.declare_input('Cx')
        sigma = self.declare_input('sigma')
        
        
        ux = Vt * ot.sin(2 * phi) * Ct / (Cx * (ot.sin(2 * phi) + 0.5 * sigma * Ct))
        ut = 2 * Vt * sigma * Ct / (2 * ot.sin(2 * phi) + sigma * Ct)

        self.register_output('ux', ux)
        self.register_output('ut', ut)
        