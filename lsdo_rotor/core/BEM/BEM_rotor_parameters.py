from lsdo_rotor.utils.options_dicitonary import OptionsDictionary


class BEMRotorParameters(OptionsDictionary):

    def initialize(self):
        self.declare('airfoil_name', types=str)
        self.declare('interp')
