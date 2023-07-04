from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
# from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from caddee.core.caddee_core.system_representation.component.component import Component
import numpy as np
import m3l
import csdl


class BEM(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', types=Component)
        self.parameters.declare('mesh', default=None, allow_none=True)#, types=BEMMesh)
        self.parameters.declare('disk_prefix', types=str)
        self.parameters.declare('blade_prefix', types=str)
        self.num_nodes = 1
    
    def compute(self) -> csdl.Model:
        from lsdo_rotor.core.BEM_caddee.BEM_model import BEMModel
        mesh = self.parameters['mesh']
        disk_prefix = self.parameters['disk_prefix']
        blade_prefix = self.parameters['blade_prefix']
        csdl_model = BEMModel(
            module=self,
            mesh=mesh,
            blade_prefix=blade_prefix,
            disk_prefix=disk_prefix,
            num_nodes=self.num_nodes,
        )
        # csdl_model.register_module_input('thrust_vector', )
        return csdl_model

    def evaluate(self, ac_states):
        component_name = self.parameters['component'].parameters['name']
        self.name = f"{component_name}_bem_model"
        self.arguments = {}
        self.arguments['u'] = ac_states['u']
        self.arguments['v'] = ac_states['v']
        self.arguments['w'] = ac_states['w']
        self.arguments['p'] = ac_states['p']
        self.arguments['q'] = ac_states['q']
        self.arguments['r'] = ac_states['r']

        forces = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(self.num_nodes, 3), operation=self)
    
        return forces, moments
        
    def _assemble_csdl(self):
        from lsdo_rotor.core.BEM_caddee.BEM_model import BEMModel
        mesh = self.parameters['mesh']
        component = self.parameters['component']
        prefix = component.parameters['name']
        csdl_model = BEMModel(
            # module=self,
            mesh=mesh,
            model_selection=self.model_selection,
            prefix=prefix,
        )
        # csdl_model.register_module_input('thrust_vector', )
        return csdl_model


class BEMMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict, default=None, allow_none=True)
        self.parameters.declare('num_blades', types=int, default=3)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=1)
        self.parameters.declare('airfoil', types=str, allow_none=True)
        self.parameters.declare('airfoil_polar', default=None, types=dict, allow_none=True)

        # NOTE: 'thrust_vector' and 'thrust_origin' won't always be parameters
        # self.parameters.declare('thrust_vector', types=np.ndarray, allow_none=True)
        # self.parameters.declare('thrust_origin', types=np.ndarray, allow_none=True)
        self.parameters.declare('ref_pt', types=np.ndarray, default=np.array([0, 0, 0]))
        self.parameters.declare('num_blades', types=int)
        self.parameters.declare('chord_b_spline_rep',types=bool, default=False)
        self.parameters.declare('twist_b_spline_rep',types=bool, default=False)
        self.parameters.declare('chord_dv', types=bool, default=False)
        self.parameters.declare('twist_dv', types=bool, default=False)
        self.parameters.declare('num_cp', default=4, allow_none=True)
        self.parameters.declare('b_spline_order', default=3, allow_none=True)
        self.parameters.declare('normalized_hub_radius', default=0.2)
