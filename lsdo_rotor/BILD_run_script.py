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

