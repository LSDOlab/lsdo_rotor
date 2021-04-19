import omtools.api as ot

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.inputs_group import InputsGroup
from lsdo_rotor.core.preprocess_group import PreprocessGroup
from lsdo_rotor.core.ideal_blade_group import IdealBladeGroup


class BCPhiGroup(ot.Group):

    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        Vx = self.declare_input('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_input('_tangential_inflow_velocity', shape=shape)
        ux = self.declare_input('_axial_induced_velocity', shape=shape)
        ut = self.declare_input('_tangential_induced_velocity', shape=shape)
        ax = self.declare_input('_axial_induction_factor', shape=shape)
        ay = self.declare_input('_tangential_induction_factor', shape = shape)


        phi = ot.arctan(ux/(Vt - 0.5*ut))        

        phi2 = ot.arctan((Vx * (1 + ax))/(Vt * (1 - ay))) 

        
        self.register_output('_inflow_angle_phi', phi)
        self.register_output('_inflow_angle_phi2', phi2)
       