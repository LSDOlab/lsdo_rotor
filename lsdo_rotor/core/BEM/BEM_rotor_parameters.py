from lsdo_rotor.utils.options_dicitonary import OptionsDictionary


class BEMRotorParameters(OptionsDictionary):
    def initialize(self):
        self.declare('airfoil_name', types=str, allow_none=True)
        self.declare('interp')
        self.declare('custom_polar', types=dict, allow_none=True)
        self.declare('cl_ml_model', default=None, allow_none=True)
        self.declare('cd_ml_model', default=None, allow_none=True)
        self.declare('use_airfoil_ml', default=False, types=bool)
        self.declare('use_custom_airfoil_ml', default=False, types=bool)
        self.declare('use_byu_airfoil_model', default=False, types=bool)
