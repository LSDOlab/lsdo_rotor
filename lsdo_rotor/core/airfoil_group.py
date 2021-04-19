import omtools.api as ot


class AirfoilGroup(ot.Group):
    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        Cl0 = self.declare_input('_Cl0', shape=shape)
        Cla = self.declare_input('_Cla', shape=shape)
        Cdmin = self.declare_input('_Cdmin', shape=shape)
        K = self.declare_input('_K', shape=shape)
        alpha_Cdmin = self.declare_input('_alpha_Cdmin', shape=shape)
        twist = self.declare_input('_pitch',shape=shape)
        phi_BEMT = self.declare_input('_phi_BEMT',shape=shape)

        alpha_BEMT = twist - phi_BEMT

        alpha = self.declare_input('_alpha', shape=shape)
        

        Cl = Cl0 + Cla * alpha
        Cd = Cdmin + K * (alpha - alpha_Cdmin)**2

        Cl_BEMT = Cl0 + Cla * alpha_BEMT
        Cd_BEMT = Cdmin + K * (alpha_BEMT - alpha_Cdmin)**2

        self.register_output('_Cl',Cl)
        self.register_output('_Cd',Cd)

        self.register_output('_Cl_BEMT',Cl_BEMT)
        self.register_output('_Cd_BEMT',Cd_BEMT)
        

