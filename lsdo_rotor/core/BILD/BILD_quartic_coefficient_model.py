import numpy as np
from csdl import Model
import csdl

class BILDQuarticCoeffsModel(Model):

    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']
        num_nodes = shape[0]

        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        S = self.declare_variable('ideal_loading_constant', shape=(num_nodes,))

        Cl = self.declare_variable('Cl_max_BILD', shape=(num_nodes,))
        Cd = self.declare_variable('Cd_min_BILD', shape=(num_nodes,))

        # self.print_var(Cl/Cd)

        # C = csdl.expand(csdl.reshape(S, (1,)), shape)

        C = csdl.expand(S, shape, 'i->ijk')
        Cl_ref_chord = csdl.expand(Cl, shape, 'i->ijk')
        Cd_ref_chord = csdl.expand(Cd, shape, 'i->ijk')


        coeff_0 = (Vt**2*(2*Vt**2 + 2*Vx**2 + C*Vx)*(4*Cd_ref_chord**2*Vt**2 - 10*Cd_ref_chord*Cl_ref_chord*Vt*Vx - 2*C*Cd_ref_chord*Cl_ref_chord*Vt - 2*Cl_ref_chord**2*Vt**2 + 4*Cl_ref_chord**2*Vx**2 + C*Cl_ref_chord**2*Vx))/Cl_ref_chord**2
        coeff_1 = (Vt**2*(- 16*Cd_ref_chord**2*C*Vt**2*Vx - 24*Cd_ref_chord**2*Vt**4 - 24*Cd_ref_chord**2*Vt**2*Vx**2 + 12*Cd_ref_chord*Cl_ref_chord*C**2*Vt*Vx + 12*Cd_ref_chord*Cl_ref_chord*C*Vt**3 + 68*Cd_ref_chord*Cl_ref_chord*C*Vt*Vx**2 + 84*Cd_ref_chord*Cl_ref_chord*Vt**3*Vx + 84*Cd_ref_chord*Cl_ref_chord*Vt*Vx**3 + 4*Cl_ref_chord**2*C**2*Vt**2 - 4*Cl_ref_chord**2*C**2*Vx**2 + 24*Cl_ref_chord**2*C*Vt**2*Vx - 20*Cl_ref_chord**2*C*Vx**3 + 40*Cl_ref_chord**2*Vt**4 + 16*Cl_ref_chord**2*Vt**2*Vx**2 - 24*Cl_ref_chord**2*Vx**4))/Cl_ref_chord**2
        coeff_2 =  - (Vt**2*(- 16*Cd_ref_chord**2*C*Vt**2*Vx - 16*Cd_ref_chord**2*Vt**4 - 16*Cd_ref_chord**2*Vt**2*Vx**2 + 24*Cd_ref_chord*Cl_ref_chord*C**2*Vt*Vx + 8*Cd_ref_chord*Cl_ref_chord*C*Vt**3 + 112*Cd_ref_chord*Cl_ref_chord*C*Vt*Vx**2 + 128*Cd_ref_chord*Cl_ref_chord*Vt**3*Vx + 128*Cd_ref_chord*Cl_ref_chord*Vt*Vx**3 + 20*Cl_ref_chord**2*C**2*Vt**2 - 4*Cl_ref_chord**2*C**2*Vx**2 + 104*Cl_ref_chord**2*C*Vt**2*Vx - 16*Cl_ref_chord**2*C*Vx**3 + 132*Cl_ref_chord**2*Vt**4 + 116*Cl_ref_chord**2*Vt**2*Vx**2 - 16*Cl_ref_chord**2*Vx**4))/Cl_ref_chord**2
        coeff_3 = (16*Vt**3*(2*Cl_ref_chord*C**2*Vt + Cd_ref_chord*C**2*Vx + 9*Cl_ref_chord*C*Vt*Vx + 4*Cd_ref_chord*C*Vx**2 + 10*Cl_ref_chord*Vt**3 + 4*Cd_ref_chord*Vt**2*Vx + 10*Cl_ref_chord*Vt*Vx**2 + 4*Cd_ref_chord*Vx**3))/Cl_ref_chord
        coeff_4 =  - 16*Vt**4*(C**2 + 4*C*Vx + 4*Vt**2 + 4*Vx**2)

        self.register_output('coeff_4', coeff_4)
        self.register_output('coeff_3', coeff_3)
        self.register_output('coeff_2', coeff_2)
        self.register_output('coeff_1', coeff_1)
        self.register_output('coeff_0', coeff_0)