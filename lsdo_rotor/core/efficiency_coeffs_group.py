import omtools.api as ot


class EfficiencyCoeffsGroup(ot.Group):

    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        Vx = self.declare_input('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_input('_tangential_inflow_velocity', shape=shape)
        reference_C = self.declare_input('ideal_loading_constant')
        Cl = self.declare_input('Cl', shape=shape)
        Cd = self.declare_input('Cd', shape=shape)

        reference_C_non_dimensional = self.declare_input('ideal_loading_constant_non_dimensional')

        
        C = ot.expand(ot.reshape(reference_C, (1,)), shape)
        # print(C.shape,'C shape')
        # C2 = ot.expand(ot.reshape(reference_C_non_dimensional, (1,)), shape)

        coeff_0 = (Vt**2*(2*Vt**2 + 2*Vx**2 + C*Vx)*(4*Cd**2*Vt**2 - 10*Cd*Cl*Vt*Vx - 2*C*Cd*Cl*Vt - 2*Cl**2*Vt**2 + 4*Cl**2*Vx**2 + C*Cl**2*Vx))/Cl**2
        coeff_1 = (Vt**2*(- 16*Cd**2*C*Vt**2*Vx - 24*Cd**2*Vt**4 - 24*Cd**2*Vt**2*Vx**2 + 12*Cd*Cl*C**2*Vt*Vx + 12*Cd*Cl*C*Vt**3 + 68*Cd*Cl*C*Vt*Vx**2 + 84*Cd*Cl*Vt**3*Vx + 84*Cd*Cl*Vt*Vx**3 + 4*Cl**2*C**2*Vt**2 - 4*Cl**2*C**2*Vx**2 + 24*Cl**2*C*Vt**2*Vx - 20*Cl**2*C*Vx**3 + 40*Cl**2*Vt**4 + 16*Cl**2*Vt**2*Vx**2 - 24*Cl**2*Vx**4))/Cl**2
        coeff_2 =  - (Vt**2*(- 16*Cd**2*C*Vt**2*Vx - 16*Cd**2*Vt**4 - 16*Cd**2*Vt**2*Vx**2 + 24*Cd*Cl*C**2*Vt*Vx + 8*Cd*Cl*C*Vt**3 + 112*Cd*Cl*C*Vt*Vx**2 + 128*Cd*Cl*Vt**3*Vx + 128*Cd*Cl*Vt*Vx**3 + 20*Cl**2*C**2*Vt**2 - 4*Cl**2*C**2*Vx**2 + 104*Cl**2*C*Vt**2*Vx - 16*Cl**2*C*Vx**3 + 132*Cl**2*Vt**4 + 116*Cl**2*Vt**2*Vx**2 - 16*Cl**2*Vx**4))/Cl**2
        coeff_3 = (16*Vt**3*(2*Cl*C**2*Vt + Cd*C**2*Vx + 9*Cl*C*Vt*Vx + 4*Cd*C*Vx**2 + 10*Cl*Vt**3 + 4*Cd*Vt**2*Vx + 10*Cl*Vt*Vx**2 + 4*Cd*Vx**3))/Cl
        coeff_4 =  - 16*Vt**4*(C**2 + 4*C*Vx + 4*Vt**2 + 4*Vx**2)

        # x = Vx/Vt

        # p4 = -C2 * x**4
        # p3 = 2 * C2 * x**4 - 2 * C2 * x**2 + 2 * x**4
        # p2 = 6 * C2 * x**2 + 3 * x**2 - 3 * x**4
        # p1 = -2 * C2 * x**2 + 2 * C2 - 6 * x**2
        # p0 = x**2 - 1 -C2

        self.register_output('coeff_4', coeff_4)
        self.register_output('coeff_3', coeff_3)
        self.register_output('coeff_2', coeff_2)
        self.register_output('coeff_1', coeff_1)
        self.register_output('coeff_0', coeff_0)

        # self.register_output('p4', p4)
        # self.register_output('p3', p3)
        # self.register_output('p2', p2)
        # self.register_output('p1', p1)
        # self.register_output('p0', p0)