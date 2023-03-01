from lsdo_rotor.utils.options_dicitonary import OptionsDictionary


class BEMRotorParameters(OptionsDictionary):
    def initialize(self):
        self.declare('airfoil_name', types=str, allow_none=True)
        self.declare('interp')
        self.declare('custom_polar', types=dict, allow_none=True)
