import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor.core.BILD.BILD_run_model import BILDRunModel

num = 50
reference_radius_fraction = 0.5
reference_chord_fraction = 0.15
rpm = 800
altitude = 1000
shape = tuple((1, 40, 1))
num_blades = 3
J = np.linspace(0, 3, 50)
R = np.linspace(0.5, 1, 50)


# for i in range(50):
#     for j in range(50):
