import numpy as np 
from csdl import Model
from python_csdl_backend import Simulator

from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel
from lsdo_rotor.core.BILD.BILD_run_model import BILDRunModel

import time
from smt.sampling_methods import FullFactorial
import matplotlib.pyplot as plt
from modopt.scipy_library import SLSQP
from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
import pickle


# Sampling
# Vx, rpm, radius, altitude, reference chord length
xlimits = np.array([[40, 60], [1000, 2000], [0.75, 1.5], [0, 3000], [0.3, 0.35]])
sampling = FullFactorial(xlimits=xlimits)
num = 7**5
x = sampling(num)
# Advance ratio, radius, altitude, reference radius
# xlimits2 = np.array([[0, 3], [1, 1.5], [0, 3000], [0.30, 0.35]])
# sampling2 = FullFactorial(xlimits=xlimits2)
# num2 = 10**4
# x2 = sampling2(num2)
# J_list = []
# Vx_list = []
# M_tip_list= []

# Fixed rpm 
# rpm = 800
# n = rpm/60

statistics = np.zeros((num, 10))

for i in range(num):
    # J = x2[i, 0]
    Vx = x[i, 0]
    # rotor_radius = x2[i, 1]
    rpm = x[i, 1]
    rotor_radius = x[i, 2]
    # D = 2 * rotor_radius
    # Vx = J * n * D
    # altitude = x2[i, 2]
    altitude = x[i, 3]
    # reference_chord = x2[i, 3]
    reference_chord = x[i, 4]

    # print(rotor_radius)
    # print(reference_chord)
    # print(altitude)
    # print(Vx)
    # exit()

    reference_radius = 0.55 * rotor_radius
    num_blades = 3
    num_radial = 30
    num_nodes = 1
    num_tangential = num_azimuthal = 1
   


    # Run and time BILD method 
    sim_BILD = Simulator(BILDRunModel(
        rotor_radius=rotor_radius,
        reference_chord=reference_chord,
        reference_radius=reference_radius,
        rpm=rpm,
        Vx=Vx,
        altitude=altitude,
        shape=(num_nodes, num_radial, num_tangential),
        num_blades=3,
    ))
    t_BILD_start = time.time()
    sim_BILD.run()
    t_BILD_end = time.time()

    # Thrust value from BILD 
    T_BILD = sim_BILD['total_thrust'].flatten()
    chord_BILD = sim_BILD['_local_chord'].flatten()
    twist_BILD = sim_BILD['_local_twist_angle'].flatten()
    E_total_BILD = sim_BILD['total_energy_loss'].flatten()
    eta_BILD = sim_BILD['eta']

    # Preform BEM optimization and time how long it takes
    sim_BEM = Simulator(BEMRunModel(
        rotor_radius=rotor_radius,
        reference_chord=reference_chord,
        reference_radius=reference_radius,
        rpm=rpm,
        Vx=Vx,
        altitude=altitude,
        shape=(num_nodes, num_radial, num_tangential),
        num_blades=3,
    ))
    t_optimization_start = time.time()

    prob = CSDLProblem(problem_name='blade_optimization_problem', simulator=sim_BEM)
    optimizer = SNOPT(
        prob, 
        Major_iterations = 100, 
        Major_optimality=1e-7, 
        Major_feasibility=1e-5,
        append2file=True,
    )
    # optimizer = SLSQP(prob, maxiter=100, ftol=1e-7)
    
    optimizer.solve()
    optimizer.print_results()
    t_optimization_end = time.time()
    
    exit_code = optimizer.snopt_output.info

    BILD_time = t_BILD_end - t_BILD_start
    BEM_opt_time = t_optimization_end-t_optimization_start


    T_BEM = sim_BEM['T'].flatten()
    E_total_BEM = sim_BEM['total_energy_loss'].flatten()
    eta_BEM = sim_BEM['eta']
    # FOM_BILD = sim_BILD['FOM']
    # FOM_BEM = sim_BEM['FOM']
    chord_BEM = sim_BEM['_chord'].flatten()
    twist_BEM = sim_BEM['_pitch'].flatten()

    Cl_max_BILD = sim_BILD['Cl_max_BILD'].flatten()
    Cd_min_BILD = sim_BILD['Cd_min_BILD'].flatten()
    max_LD_BILD = Cl_max_BILD / Cd_min_BILD

    Cl_BEM = sim_BEM['Cl_2'].flatten()
    Cd_BEM = sim_BEM['Cd_2'].flatten()
    max_LD_BEM = Cl_BEM / Cd_BEM
    

    Vx =  sim_BEM['_axial_inflow_velocity'].flatten()[-1]
    Vt = sim_BEM['_tangential_inflow_velocity'].flatten()[-1]
    V_tip = np.sqrt(Vx**2 + Vt**2)
    a = np.sqrt(1.4 * 287 * 293)
    M_tip = V_tip / a

    # Errors: 
    eps_energy = abs(E_total_BILD - E_total_BEM) / E_total_BEM * 100
    eps_chord = (np.linalg.norm(chord_BILD - chord_BEM) / np.linalg.norm(chord_BEM) * 100)
    eps_twist = (np.linalg.norm(twist_BILD - twist_BEM) / np.linalg.norm(twist_BEM) * 100)
    eps_max_LD = (np.linalg.norm(max_LD_BILD - max_LD_BEM) / np.linalg.norm(max_LD_BEM) * 100)
    eps_eta = abs(eta_BILD-eta_BEM) / abs(eta_BEM) * 100
    # eps_FOM = abs(FOM_BILD-FOM_BEM) / abs(FOM_BEM) * 100

    # Save data
    statistics[i, 0] = exit_code
    statistics[i, 1] = BILD_time
    statistics[i, 2] = BEM_opt_time
    statistics[i, 3] = eps_energy
    statistics[i, 4] = eps_chord
    statistics[i, 5] = eps_twist
    statistics[i, 6] = eps_max_LD
    statistics[i, 7] = M_tip
    statistics[i, 8] = eps_eta
    # statistics[i, 9] = eps_FOM



    with open(f'full_factorial_sweep_2.pickle', 'wb') as handle:
        pickle.dump(statistics[0:i, :], handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n')
    print('Sweep number:                    ', i)
    print('BILD run time:                   ', BILD_time)
    print('BEM optimization time:           ', BEM_opt_time)
    print(f'BILD is {round(BEM_opt_time/BILD_time, 3)} times faster.')
    print('Thrust constraint:               ', T_BEM - T_BILD)
    print('\n')

    print('--------------------ERRORS-----------------------')
    print('Energy % error                   ', eps_energy)
    print('Chord % error:                   ', eps_chord)
    print('Twist % error:                   ', eps_twist)
    print('Max L/D % error:                 ', eps_max_LD)
    print('eta % error:                     ', eps_eta)
    # print('FOM % error:                     ', eps_FOM)
    # print(eta_BEM)
    # print(eta_BILD)

   
    visualize_blade_design = 'y'
    if visualize_blade_design == 'y':
        from lsdo_rotor.core.BILD.functions.plot_ideal_loading_blade_shape import plot_ideal_loading_blade_shape
        plot_ideal_loading_blade_shape(sim_BILD, sim_BEM)
    exit()

    print('\n')
    print(M_tip)






# np.savetxt('statistics.txt', statistics.flatten())






# plt.show()


