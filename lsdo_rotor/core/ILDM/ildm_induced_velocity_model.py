import numpy as np
from csdl import Model
import csdl


class ILDMInducedVelocityModel(Model):
    
    def initialize(self):
        self.parameters.declare('shape',types=tuple)
        self.parameters.declare('num_blades',types=int)

    def define(self):
        shape = self.parameters['shape']
        B = num_blades = self.parameters['num_blades']
        num_nodes = shape[0]

        phi = self.declare_variable('phi_reference_ildm', shape=(num_nodes,))
        Vx = self.declare_variable('u', shape=(num_nodes,))
        Vt = self.declare_variable('ildm_tangential_inflow_velocity', shape=(num_nodes,))

        reference_radius = self.declare_variable('reference_radius', shape=(num_nodes,))
        rotor_radius = self.declare_variable('rotor_radius', shape=(num_nodes,))
        hub_radius = self.declare_variable('hub_radius', shape=(num_nodes,))
        sigma = self.declare_variable('reference_blade_solidity', shape=(num_nodes,))

        Cl = self.declare_variable('Cl_max_ildm', shape=(num_nodes,))
        Cd = self.declare_variable('Cd_min_ildm', shape=(num_nodes,))

        f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi)
        f_hub = B / 2 * (reference_radius - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

        F = F_tip * F_hub
        

        Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

        ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct)
        # ux = Vx + sigma * Cx * Vt / (4 * F * csdl.sin(phi) * csdl.cos(phi) + sigma * Ct)
        ut = 2 * Vt * sigma * Ct / (2 * F * csdl.sin(2 * phi) + sigma * Ct)

        C = Vt * ut /(2 * ux - Vx) + 2 * Vt * ux /(Vt - ut) - 2 * Vx

        self.register_output('ideal_loading_constant', C)

    