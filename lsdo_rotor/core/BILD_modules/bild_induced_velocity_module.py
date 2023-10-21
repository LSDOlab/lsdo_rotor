import numpy as np

import csdl


class BILDInducedVelocityModuleCSDL(csdl.Model):
    
    def initialize(self):
        self.parameters.declare('shape',types=tuple)
        self.parameters.declare('num_blades',types=int)

    def define(self):
        shape = self.parameters['shape']
        B = num_blades = self.parameters['num_blades']
        num_nodes = shape[0]

        phi = self.register_module_input('phi_reference_BILD', shape=(num_nodes,))
        # self.print_var(phi)
        Vx = self.register_module_input('u', shape=(num_nodes,))
        Vt = self.register_module_input('BILD_tangential_inflow_velocity', shape=(num_nodes,))

        reference_radius = self.register_module_input('reference_radius', shape=(num_nodes,))
        rotor_radius = self.register_module_input('propeller_radius', shape=(num_nodes,))
        # self.print_var(rotor_radius)
        hub_radius = self.register_module_input('hub_radius', shape=(num_nodes,))
        sigma = self.register_module_input('reference_blade_solidity', shape=(num_nodes,))

        Cl = self.register_module_input('Cl_max_BILD', shape=(num_nodes,))
        Cd = self.register_module_input('Cd_min_BILD', shape=(num_nodes,))

        f_tip = B / 2 * (rotor_radius - reference_radius) / reference_radius / csdl.sin(phi)
        # self.print_var(rotor_radius - reference_radius)
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

        self.register_module_output('ideal_loading_constant', C)
        # self.print_var(C)

    