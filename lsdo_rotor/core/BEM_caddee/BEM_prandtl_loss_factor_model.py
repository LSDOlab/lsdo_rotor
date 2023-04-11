import numpy as np 
from csdl import Model
import csdl 

class BEMPrandtlLossFactorModel(Model):
    
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_blades', types=int)
        

    def define(self):
        shape = self.parameters['shape']
        B = num_blades = self.parameters['num_blades']

        radius = self.declare_variable('_radius',shape= shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
        hub_radius = self.declare_variable('_hub_radius', shape=shape)
        phi = self.declare_variable('phi_distribution', shape=shape)
        

        f_tip = B / 2 * (rotor_radius - radius) / radius / csdl.sin(phi)
        f_hub = B / 2 * (radius - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

        F = F_tip * F_hub
        self.register_output('prandtl_loss_factor', F)