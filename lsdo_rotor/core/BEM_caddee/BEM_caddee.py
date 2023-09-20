from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
# from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from caddee.core.caddee_core.system_representation.component.component import Component
import numpy as np
import m3l
import csdl


class BEM(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', types=Component, allow_none=True)
        self.parameters.declare('mesh', allow_none=True)#, types=BEMMesh)
        self.parameters.declare('disk_prefix', types=str)
        self.parameters.declare('disk_suffix', types=str, default=None, allow_none=True)
        self.parameters.declare('blade_prefix', types=str)
        self.parameters.declare('num_nodes', types=int, default=1)
        self.parameters.declare('use_caddee', types=bool, default=True)
        super().initialize(kwargs=kwargs)

    
    def compute(self) -> csdl.Model:
        from lsdo_rotor.core.BEM_caddee.BEM_model import BEMModel
        mesh = self.parameters['mesh']
        disk_prefix = self.parameters['disk_prefix']
        disk_suffix = self.parameters['disk_suffix']
        blade_prefix = self.parameters['blade_prefix']
        num_nodes = self.parameters['num_nodes']
        use_caddee = self.parameters['use_caddee']
        csdl_model = BEMModel(
            module=self,
            mesh=mesh,
            blade_prefix=blade_prefix,
            disk_prefix=disk_prefix,
            disk_suffix=disk_suffix,
            num_nodes=num_nodes,
            use_caddee=use_caddee,
        )
        # csdl_model.register_module_input('thrust_vector', )
        return csdl_model

    def evaluate(self, ac_states, design_condition=None):
        component = self.parameters['component']
        if component is not None:
            component_name = component.parameters['name'] 
        else:
            component_name = 'rotor'
        
        if design_condition:
            dc_name = design_condition.parameters['name']
            self.name = f"{dc_name}_{component_name}_bem_model"
        else:
            self.name = f"{component_name}_bem_model"

        self.arguments = {}
        self.arguments['u'] = ac_states['u']
        self.arguments['v'] = ac_states['v']
        self.arguments['w'] = ac_states['w']
        self.arguments['p'] = ac_states['p']
        self.arguments['q'] = ac_states['q']
        self.arguments['r'] = ac_states['r']
        self.arguments['theta'] = ac_states['theta']

        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['mesh'].parameters['num_radial'] 
        num_tangential = self.parameters['mesh'].parameters['num_tangential'] 



        forces = m3l.Variable(name='F', shape=(num_nodes, 3), operation=self)
        C_T = m3l.Variable(name='C_T', shape=(num_nodes, 3), operation=self)
        Q = m3l.Variable(name='Q', shape=(num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(num_nodes, 3), operation=self)
        dT = m3l.Variable(name='_dT', shape=(num_nodes, num_radial, num_tangential), operation=self)
        dQ = m3l.Variable(name='_dQ', shape=(num_nodes, num_radial, num_tangential), operation=self)
        dD = m3l.Variable(name='_dD', shape=(num_nodes, num_radial, num_tangential), operation=self)
        ux = m3l.Variable(name='_ux', shape=(num_nodes, num_radial, num_tangential), operation=self)

        return forces, moments, dT, dQ, dD, C_T, Q, ux
        

class BEMMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict, default=None, allow_none=True)
        self.parameters.declare('num_blades', types=int, default=3)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=1)
        self.parameters.declare('airfoil', types=str, allow_none=True)
        self.parameters.declare('airfoil_polar', default=None, types=dict, allow_none=True)

        self.parameters.declare('use_airfoil_ml', types=bool, default=False)
        self.parameters.declare('use_custom_airfoil_ml', types=bool, default=False)
        self.parameters.declare('use_rotor_geometry', types=bool, default=True)
        
        self.parameters.declare('mesh_units', values=['ft', 'm'], default='m')

        # NOTE: 'thrust_vector' and 'thrust_origin' won't always be parameters
        # self.parameters.declare('thrust_vector', types=np.ndarray, allow_none=True)
        # self.parameters.declare('thrust_origin', types=np.ndarray, allow_none=True)
        self.parameters.declare('ref_pt', types=np.ndarray, default=np.array([0, 0, 0]))
        self.parameters.declare('num_blades', types=int)
        self.parameters.declare('chord_b_spline_rep',types=bool, default=False)
        self.parameters.declare('twist_b_spline_rep',types=bool, default=False)
        self.parameters.declare('num_cp', default=4, allow_none=True)
        self.parameters.declare('b_spline_order', default=3, allow_none=True)
        self.parameters.declare('normalized_hub_radius', default=0.2)
