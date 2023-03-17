from lsdo_modules.module.module_maker import ModuleMaker
from lsdo_modules.module.module import Module
from lsdo_rotor.core.BILD_modules.bild_module_maker import BILDModuleMaker
import numpy as np
from python_csdl_backend import Simulator


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
    def assemble_module(self):
        bild_module_maker = BILDModuleMaker(
            module=self,
            airfoil_polar=airfoil_polar,
            num_blades=3,
            shape=(num_nodes, num_radial, num_tangential)
        )
        return bild_module_maker
    
bild_module = BILDModule()
bild_module.set_module_input('thrust_vector', val=np.array([1, 0, 0]))
bild_module.set_module_input('thrust_origin', val=np.array([0, 0, 0,]))
bild_module.set_module_input('u', val=50)
bild_module.set_module_input('omega', val=1500)
bild_module.set_module_input('z', val=1000)
bild_module.set_module_input('propeller_radius', val=1.2)
bild_module.set_module_input('reference_radius', val=0.78)
bild_module.set_module_input('reference_chord', val=0.15)

bild_module_maker = bild_module.assemble_module()
bild_csdl = bild_module_maker.assemble_csdl()
bild_module_maker.generate_html()


sim = Simulator(bild_csdl)
sim.run()
bild_module_maker.generate_html(sim=sim)



# print(sim['_tangential_inflow_velocity'])
# print(sim['x_dir'])


