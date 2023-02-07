import numpy as np 
from csdl import Model
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")

from python_csdl_backend import Simulator

from lsdo_rotor.core.BILD.BILD_model import BILDModel


# Design parameters
rotor_radius = np.array([1])
reference_chord = np.array([0.1])
reference_radius = np.array([0.64])
num_blades = 3

# Operating conditions
rpm = np.array([1500])
Vx = np.array([20])
altitude = np.array([1000]) # in (m)

# Visualize optimal blade design
visualize_blade_design = 'y' # 'y'/'n'


num_nodes = len(rotor_radius)

class RunModel(Model):
    def define(self):
        self.create_input(name='propeller_radius', val=rotor_radius)
        self.create_input(name='reference_chord', val=reference_chord)
        self.create_input(name='reference_radius', val=reference_radius)

        self.create_input('omega', shape=(num_nodes, ), units='rpm', val=rpm)
        self.create_input(name='u', shape=(num_nodes, ), units='m/s', val=Vx)
        self.create_input(name='z', shape=(num_nodes, ), units='m', val=altitude)
                
        self.add(BILDModel(   
            name='propulsion',
            num_nodes=num_nodes,
            num_radial=40,
            num_tangential=1,
            airfoil='NACA_4412',
            thrust_vector=np.array([1,0,0]),
            thrust_origin=np.array([8.5, 0, 5], dtype=float),
            ref_pt=np.array([4.5, 0, 5]),
            num_blades=num_blades,
        ),name='BEM_model')


sim = Simulator(RunModel())
sim.run()


if visualize_blade_design == 'y':
    from lsdo_rotor.core.BILD.functions.plot_ideal_loading_blade_shape import plot_ideal_loading_blade_shape
    plot_ideal_loading_blade_shape(sim)
