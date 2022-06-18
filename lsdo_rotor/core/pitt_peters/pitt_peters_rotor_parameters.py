from lsdo_rotor.utils.options_dicitonary import OptionsDictionary


class PittPetersRotorParameters(OptionsDictionary):

    def initialize(self):
        self.declare('airfoil_name', types=str)
        self.declare('interp')
        self.declare('mu')
        self.declare('mu_z')
        self.declare('dL_dlambda_function')
        self.declare('dL_dmu_function')
        self.declare('dL_dmu_z_function')
        self.declare('M_block_matrix')
        self.declare('M_inv_block_matrix')
        self.declare('azimuth_angle')
