import omtools.api as ot
import openmdao.api as om
from lsdo_utils.miscellaneous.options_dictionary import OptionsDictionary


# class RotorParameters(om.OptionsDictionary):
class RotorParameters(OptionsDictionary):

    def initialize(self):
        self.declare('num_blades', types=int)
        self.declare('mode', types=int)
        self.declare('Cl0', types = float)
        self.declare('Cla', types = float)
        self.declare('Cdmin', types = float)
        self.declare('K', types = float)
        self.declare('alpha_Cdmin', types = float)
        self.declare('Cl_stall_plus',types = float)
        self.declare('Cl_stall_minus',types = float)
        self.declare('Cd_stall_plus',types = float)
        self.declare('Cd_stall_minus',types = float)
        self.declare('a_stall_plus',types = float)
        self.declare('a_stall_minus',types = float)
        self.declare('AR',types = int)