# lsdo_rotor

This is a low-fidelity rotor analysis and design tool based on blade element momentum theory, developed by the LSDO lab. Please follow these instructions on installation and proper usage. 

# Installation 

lsdo_rotor requires the following packages to be installed before it can be used:

* [csdl](https://lsdolab.github.io/csdl/docs/tutorial/install); CSDL is an algebraic, domain embedded modeling language recently developed in the LSDO lab.
* [csdl backend](https://github.com/LSDOlab/python_csdl_backend)
* [Surrogate Modeling Toolbox](https://smt.readthedocs.io/en/latest/_src_docs/getting_started.html). Note that SMT is only needed for airfoil models that are trained based on XFOIL data. A custom airfoil polar can be described in terms of angle of attack only.
* [vedo](https://pypi.org/project/vedo/) vedo is used for visualizing the rotor blades. It can be installed with the command ```pip install vedo```.

Please follow the installation instructions provided in the above links. Once these packages are installed you can proceed as follows with the installation of lsdo_rotor:

* Clone this repository via ``git clone https://github.com/LSDOlab/lsdo_rotor.git`` or download it as a .zip file. If the repository is cloned successfully, future versions of lsdo_rotor can be downloaded via `git pull`.
* Execute the next two commands to install lsdo_rotor
  * ``cd lsdo_rotor``
  * ``pip install -e .``
* If the installation is successful, check that the run.py file executes by typing
  * ``cd lsdo_rotor``
  * ``python BEM_run_script.py`` or ``python BILD_run_script.py``
  
# User guidelines

The user will only have to change parameters in the execution scripts file, which has comments to explain how to properly use the code. Currently, our rotor analysis tool currently supports two modes of operation, a rotor design method (1) and classical BEM analysis (2).

1) BEM-based ideal-loading design [`BILD`](https://arc.aiaa.org/doi/abs/10.2514/6.2021-2598) method:
  This is a rotor design tool to efficiently compute the most aerodynamic efficient blade geometry of a rotor for given operating conditions. Unless otherwise indicated, all quantities have SI units. The following code can be found in the BILD_run_script.py file:
  ```python 

import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor.core.BILD.BILD_run_model import BILDRunModel
from lsdo_rotor.utils.print_output import print_output
from lsdo_rotor.utils.visualize_blade import visualize_blade
from lsdo_rotor.utils.rotor_dash import RotorDash


num_nodes = 1
num_radial = 40
num_tangential = 1

# Thrust vector is the unit normal vector w.r.t the rotor disk
thrust_vector =  np.array([
    [1,0,0],]
)

# Thrust origin is the point at which the thrust acts (usually the center of the rotor disk)
thrust_origin =  np.array([
    [8.5, 0, 5],]
)

# Design parameters
rotor_radius = 1
reference_chord = 0.15
reference_radius = 0.6 * rotor_radius # Expressed as a fraction of the radius

# Operating conditions 
Vx = 0 # (for axial flow or hover only)
rpm = 800
altitude = 0 # in (m)

num_blades = 3

shape = (num_nodes, num_radial, num_tangential)

airfoil_polar = {
    'Cl_0': 0.25,
    'Cl_alpha': 5.1566,
    'Cd_0': 0.01,
    'Cl_stall': [-1, 1.5], 
    'Cd_stall': [0.02, 0.06],
    'alpha_Cl_stall': [-10, 15],
}

sim_BILD = Simulator(BILDRunModel(
    rotor_radius=rotor_radius,
    reference_chord=reference_chord,
    reference_radius=reference_radius,
    rpm=rpm,
    Vx=Vx,
    altitude=altitude,
    shape=shape,
    num_blades=num_blades,
    airfoil_name='NACA_4412',
    airfoil_polar=airfoil_polar,
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
))

rotor_dash = RotorDash()
sim_BILD.add_recorder(rotor_dash.get_recorder())
sim_BILD.run()
print_output(sim=sim_BILD)
visualize_blade(dash=rotor_dash)
```
 
  
2) Blade element momentum (BEM) theory
  This is a rotor analysis tool that computes the aerodynamic performance of an rotor using BEM theory. The following code can be found in the BEM_run_script.py file.
 
  ```python 
  import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel
from lsdo_rotor.utils.print_output import print_output
from lsdo_rotor.utils.visualize_blade import visualize_blade
from lsdo_rotor.utils.rotor_dash import RotorDash


num_nodes = 1
num_radial = 50
num_tangential = num_azimuthal = 1

# Thrust vector is the unit normal vector w.r.t the rotor disk
thrust_vector =  np.array([
    [1,0,0],]
)

# Thrust origin is the point at which the thrust acts (usually the center of the rotor disk)
thrust_origin =  np.array([
    [8.5, 5, 5],]
)

# Reference point is the point about which the moments due to thrust will be computed
reference_point = np.array([8.5, 0, 5])

shape = (num_nodes, num_radial, num_tangential)

rotor_radius = 1
rpm = 1200
Vx = 40 # (for axial flow or hover only)
altitude = 1000
num_blades = 3

airfoil_polar = {
    'Cl_0': 0.25,
    'Cl_alpha': 5.1566,
    'Cd_0': 0.01,
    'Cl_stall': [-1, 1.5], 
    'Cd_stall': [0.02, 0.06],
    'alpha_Cl_stall': [-10, 15],
}

chord = np.linspace(0.3, 0.2, num_radial)
twist = np.linspace(60, 15, num_radial)

sim_BEM = Simulator(BEMRunModel(
    rotor_radius=rotor_radius,
    rpm=rpm,
    Vx=Vx,
    altitude=altitude,
    shape=shape,
    num_blades=num_blades,
    airfoil_name='NACA_4412',
    airfoil_polar=airfoil_polar,
    chord_distribution=chord,
    twist_distribution=twist,
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
))


rotor_dash = RotorDash()
sim_BEM.add_recorder(rotor_dash.get_recorder())
sim_BEM.run()
print_output(sim=sim_BEM, write_to_csv=True)
visualize_blade(dash=rotor_dash)

  ```

  
  
