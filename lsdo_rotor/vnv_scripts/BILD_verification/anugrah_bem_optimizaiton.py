import numpy as np 
from csdl import Model
from python_csdl_backend import Simulator

# from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel
from lsdo_rotor.core.BILD.BILD_run_model import BILDRunModel
from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel

import time
from smt.sampling_methods import FullFactorial
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams.update({'font.size': 13})
from modopt.scipy_library import SLSQP
from modopt.optimization_algorithms import SQP
# from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
import pickle


rotor_radius = 0.75
reference_chord = 0.1
reference_radius_fraction = 0.5
rpm = 2000
Vx = 40
altitude = 1000
num_radial = 30
shape = tuple((1, num_radial, 1))
num_blades = 3

# Run and time BILD method 
sim_BILD = Simulator(BILDRunModel(
    rotor_radius=rotor_radius,
    reference_chord=reference_chord,
    reference_radius=reference_radius_fraction*rotor_radius,
    rpm=rpm,
    Vx=Vx,
    altitude=altitude,
    shape=shape,
    num_blades=num_blades,
    airfoil_name='NACA_4412',
    airfoil_polar=None,
    thrust_vector=np.array([[1,0,0]]),
    thrust_origin=np.array([[8.5, 5, 5]])
))
t_BILD_start = time.time()
sim_BILD.run()
t_BILD_end = time.time()

# Thrust value from BILD 
T_BILD = sim_BILD['total_thrust'].flatten()

E_total_BILD = sim_BILD['total_energy_loss'].flatten()
chord_BILD = sim_BILD['_local_chord'].flatten()
twist_BILD = sim_BILD['_local_twist_angle'].flatten()

# Preform BEM optimization and time how long it takes
sim_BEM = Simulator(BEMRunModel(
    rpm=rpm,
    Vx=Vx,
    altitude=altitude,
    shape=shape,
    num_blades=3,
    airfoil_name='NACA_4412',
    airfoil_polar=None,
    BILD_chord=chord_BILD,
    BILD_twist=twist_BILD,
    BILD_thrust_constraint=T_BILD,
    E_total_BILD=E_total_BILD,
    rotor_radius=rotor_radius,
    chord_B_spline_rep=True,
    twist_B_spline_rep=True,
), analytics=True)

sim_BEM.run()

prob = CSDLProblem(problem_name='blade_optimization_problem', simulator=sim_BEM)
optimizer = SLSQP(
    prob, 
)
optimizer.solve()
optimizer.print_results()
print(T_BILD )
print(sim_BEM['T'].flatten())
print(sim_BILD['eta'])
print(sim_BEM['eta'])

optimizer = SQP(
    prob, 
)
optimizer.solve()
    
optimizer.print_results(summary_table=True)





