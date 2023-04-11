from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from lsdo_rotor.core.BILD_modules.bild_module import BILDModuleCSDL
import numpy as np
from python_csdl_backend import Simulator
from csdl import GraphRepresentation


airfoil_polar = {
    'Cl_0': 0.25,
    'Cl_alpha': 5.1566,
    'Cd_0': 0.01,
    'Cl_stall': [-1, 1.5], 
    'Cd_stall': [0.02, 0.06],
    'alpha_Cl_stall': [-10, 15],
}

num_nodes = 1
num_radial = 40
num_tangential = 1

class BILDModule(Module):
    def initialize(self, kwargs): pass
    def assemble_csdl(self):
        bild_module_csdl = BILDModuleCSDL(
            module=self,
            airfoil_polar=airfoil_polar,
            num_blades=3,
            shape=(num_nodes, num_radial, num_tangential)
        )
        GraphRepresentation(bild_module_csdl)
        return bild_module_csdl
    
bild_module = BILDModule()
bild_module.set_module_input('thrust_vector', val=np.array([1, 0, 0]))
bild_module.set_module_input('thrust_origin', val=np.array([0, 0, 0,]))
bild_module.set_module_input('u', val=50, dv_flag=True)
bild_module.set_module_input('omega', val=1500)
bild_module.set_module_input('z', val=1000)
bild_module.set_module_input('propeller_radius', val=1.2, dv_flag=True)
bild_module.set_module_input('reference_radius', val=0.78)
bild_module.set_module_input('reference_chord', val=0.15)

bild_module_csdl = bild_module.assemble_csdl()
bild_module_csdl.add_objective('Re_BILD')
# bild_module_csdl.add_design_variable('u')
bild_module_csdl.visualize_implementation(importance=2)

sim = Simulator(bild_module_csdl)
sim.run()
# bild_module_maker.generate_html(sim=sim)


print(sim['_local_chord'])
# print(sim['_tangential_inflow_velocity'])
# print(sim['x_dir'])


