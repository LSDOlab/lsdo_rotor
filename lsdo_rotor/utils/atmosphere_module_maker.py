import numpy as np
from csdl import Model
from lsdo_modules.module.module_maker import ModuleMaker



class AtmosphereMaker(ModuleMaker):
    def initialize_module(self):
        self.parameters.declare('shape', types=tuple)

    def define_module(self):
        shape = self.parameters['shape']
        h = self.register_module_input('z', shape=shape,) * 1e-3 # value in meters; then convert to km
        L = 6.5 # K/km
        R = 287
        T0 = 288.16
        P0 = 101325
        g0 = 9.81
        mu0 = 1.735e-5
        S1 = 110.4
        gamma = 1.4

        # Temperature 
        T = - h * L + T0

        # Pressure 
        P = P0 * (T/T0)**(g0/(L * 1e-3)/R)
        
        # Density
        rho         = P/R/T
        
        # Dynamic viscosity (using Sutherland's law)  
        mu          = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1)

        # speed of sound 
        a = (gamma * R * T)**0.5

        self.register_module_output('temperature',T)
        self.register_module_output('pressure',P)
        self.register_module_output('density',rho)
        self.register_module_output('dynamic_viscosity',mu)
        self.register_module_output('speed_of_sound', a)

