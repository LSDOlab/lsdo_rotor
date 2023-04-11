from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from caddee.caddee_core.system_configuration.component.component import Component
import numpy as np


class BEM(MechanicsModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', types=Component)
        self.parameters.declare('mesh', default=None, allow_none=True)#, types=BEMMesh)
        self.num_nodes = 3
        self.num_active_nodes = 1
        self.model_selection = None
    
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
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('num_blades', types=int, default=3)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=1)
        self.parameters.declare('airfoil', types=str, allow_none=True)
        self.parameters.declare('airfoil_polar', types=dict, allow_none=True)

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
