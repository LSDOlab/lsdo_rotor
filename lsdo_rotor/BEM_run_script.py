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
