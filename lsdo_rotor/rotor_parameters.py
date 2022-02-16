from lsdo_utils.miscellaneous.options_dictionary import OptionsDictionary

class RotorParameters(OptionsDictionary):

    def initialize(self):
        self.declare('num_blades', types=int)
        self.declare('mode', types=int)
        self.declare('altitude')
        self.declare('airfoil_name', types = str)
        self.declare('interp')
        # self.declare('ideal_alpha')
        # self.declare('ideal_Cl')
        # self.declare('ideal_Cd')
        self.declare('ideal_alpha_ref_chord')
        self.declare('ideal_Cl_ref_chord')
        self.declare('ideal_Cd_ref_chord')
        self.declare('c_ref',types = float)