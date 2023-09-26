import numpy as np
import julia
from julia import CCBlade as ccb
import csdl 
from lsdo_rotor import BYU_AIRFOIL_FOLDER


def airfoil_polar(alpha, Re, M):
    file = "/home/marius_ruh/packages/lsdo_lab/lsdo_rotor/lsdo_rotor/airfoil/byu_model/naca4412_Re1e6_Mach0_smooth_extended_rotation.dat"

    #create CCBlade airfoil object
    af = ccb.AlphaAF(file, radians=True)

    #grab interpolated values from af
    cl, cd = ccb.afeval(af, alpha, Re, M) #only interpolates alpha

    #apply Re correction - based on flat plate with turbulent flow
    cd *= (1e6 / Re)**0.2

    # apply mach correction - Prandtl-Glauert correction
    cl = cl / np.sqrt(1 - M**2)

    return cl, cd


class BYUAirfoilModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']
        self.add_input('alpha_distribution', shape=shape)
        self.add_input('Re', shape=shape)
        self.add_input('mach_number', shape=shape)

        self.add_output('Cl', shape=shape)
        self.add_output('Cd', shape=shape)

    def compute(self, inputs, outputs):
        shape = self.parameters['shape']
        aoa = inputs['alpha_distribution'].flatten()
        Re = inputs['Re'].flatten()
        mach = inputs['mach_number'].flatten()

        Cl, Cd = airfoil_polar(aoa, Re, mach)

        outputs['Cl'] = Cl.reshape(shape)
        outputs['Cd'] = Cd.reshape(shape)

if __name__ == '__main__':
    cl, cd = airfoil_polar(np.array([0., 0., 0., 0.,]), np.array([1e6, 1e6,1e6,1e6]), np.array([0, 0.1, 0.2, 0.3]))

    print(cl)
    print(cd)