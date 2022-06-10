from lsdo_rotor.utils.options_dicitonary import OptionsDictionary


class RotorParameters(OptionsDictionary):

    def initialize(self):
        self.declare('num_blades', types=int)
        self.declare('mode', types=int)
        self.declare('altitude')
        self.declare('airfoil_name', types=str)
        self.declare('interp')
        self.declare('density')
        self.declare('mu')
        self.declare('mu_z')
        self.declare('dL_dlambda_function')
        self.declare('M_block_matrix')
        self.declare('M_inv_block_matrix')
        self.declare('azimuth_angle')
