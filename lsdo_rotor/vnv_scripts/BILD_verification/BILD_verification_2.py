import numpy as np 
from csdl import Model
from python_csdl_backend import Simulator

# from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel
from lsdo_rotor.core.BILD.BILD_run_model import BILDRunModel
from lsdo_rotor.core.BEM_caddee.BEM_run_model import BEMRunModel

import time
from smt.sampling_methods import FullFactorial
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams.update({'font.size': 13})
from modopt.scipy_library import SLSQP
# from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem
import pickle


# rotor_radius = 0.75
# reference_chord = 0.1
reference_radius_fraction = 0.5
# rpm = 800
# Vx = 0
# altitude = 0
num_radial = 30
shape = tuple((1, num_radial, 1))
num_blades = 3

xlimits = np.array([[0, 3], [0.5, 1.5], [0, 3000], [0.08, 0.22]])
sampling = FullFactorial(xlimits=xlimits)
num = 10**4
x = sampling(num)
# print(x[:,x[:,0]==0.])
# print(np.where(x[:,0]==0))
# print(x[0:1000,:])
# exit()
rpm = 2095


num_run = 9596 #2990 + 138 # 6463 #9636 # 6424 # 3212 
statistics = np.zeros((num-num_run, 17))

J_lst = [0.5, 1, 2, 3]
twist_chord_array = np.zeros((4, len(J_lst), num_radial))
for i in range(num-num_run):
# for i in range(len(J_lst)):
    j = i + num_run
    # j = i
    J =  0 #x[j, 0]
    rotor_radius = 0.75 #x[j, 1]
    altitude = 500# x[j, 2]
    reference_chord =  0.12 # x[j, 3]
    n = rpm/60
    D = 2*rotor_radius
    Vx = J * n * D

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
    print(T_BILD * 8 - (725 * 9.81))
    # exit()
    
    C_T_BILD = sim_BILD['C_T'].flatten()
    C_Q_BILD = sim_BILD['C_Q'].flatten()
    C_P_BILD = sim_BILD['C_P'].flatten()
    E_total_BILD = sim_BILD['total_energy_loss'].flatten()
    chord_BILD = sim_BILD['_local_chord'].flatten()
    twist_BILD = sim_BILD['_local_twist_angle'].flatten()
    Cl_max_BILD = sim_BILD['Cl_max_BILD'].flatten()
    Cd_min_BILD = sim_BILD['Cd_min_BILD'].flatten()
    max_LD_BILD = Cl_max_BILD / Cd_min_BILD
    eta_BILD = sim_BILD['eta']
    print(eta_BILD)

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
    T_BEM = sim_BEM['T'].flatten()
    T_BEM_2 = sim_BEM['total_thrust_2'].flatten()
    T_BEM_3 = sim_BEM['total_thrust_star'].flatten()
    chord_BEM = sim_BEM['_chord'].flatten()
    twist_BEM = sim_BEM['_pitch'].flatten()
    # print(T_BEM)
    # print(T_BEM_2)
    # print(T_BEM_3)
    # exit()
    t_optimization_start = time.time()


    prob = CSDLProblem(problem_name='blade_optimization_problem', simulator=sim_BEM)
    optimizer = SNOPT(
        prob, 
        Major_iterations = 100, 
        Major_optimality=1e-7, 
        Major_feasibility=1e-8,
        append2file=True,
    )
    # optimizer = SLSQP(prob, maxiter=100, ftol=1e-7)

    optimizer.solve()
    t_optimization_end = time.time()
    optimizer.print_results()

    exit_code = optimizer.snopt_output.info

    BILD_time = t_BILD_end - t_BILD_start
    BEM_opt_time = t_optimization_end-t_optimization_start

    T_BEM = sim_BEM['T'].flatten()
    C_T_BEM = sim_BEM['C_T'].flatten()
    C_Q_BEM = sim_BEM['C_Q'].flatten()
    C_P_BEM = sim_BEM['C_P'].flatten()
    E_total_BEM = sim_BEM['total_energy_loss'].flatten()
    eta_BEM = sim_BEM['eta']
    FOM_BILD = sim_BILD['FOM']
    FOM_BEM = sim_BEM['FOM']
    chord_BEM = sim_BEM['_chord'].flatten()
    twist_BEM = sim_BEM['_pitch'].flatten()
    Cl_BEM = sim_BEM['Cl_2'].flatten()
    Cd_BEM = sim_BEM['Cd_2'].flatten()
    max_LD_BEM = Cl_BEM / Cd_BEM

    # Errors: 
    eps_energy = abs(E_total_BILD - E_total_BEM) / E_total_BEM * 100
    eps_chord = (np.linalg.norm(chord_BILD - chord_BEM) / np.linalg.norm(chord_BEM) * 100)
    eps_twist = (np.linalg.norm(twist_BILD - twist_BEM) / np.linalg.norm(twist_BEM) * 100)
    eps_max_LD = (np.linalg.norm(max_LD_BILD - max_LD_BEM) / np.linalg.norm(max_LD_BEM) * 100)
    eps_eta = abs(eta_BILD-eta_BEM) / abs(eta_BEM) * 100
    eps_FOM = abs(FOM_BILD-FOM_BEM) / abs(FOM_BEM) * 100  
    
    # dQ_BILD = sim_BILD['_dQ'].flatten()
    
    # np.savetxt('r', r)
    # np.savetxt('dT_BEM', dT_BEM)
    # np.savetxt('dQ_BEM', dQ_BEM)
    # np.savetxt('dT_BILD', dT_BILD)
    # np.savetxt('dQ_BILD', dQ_BILD)

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 10))
    
    # axs[0].plot(r, dT_BEM, color='navy')
    # axs[0].plot(r, dT_BILD, color='maroon')

    # axs[1].plot(r, dQ_BEM, color='navy')
    # axs[1].plot(r, dQ_BILD, color='maroon')

    # # plt.show()

    exit()
    # # Save data
    statistics[i, 0] = C_T_BEM
    statistics[i, 1] = C_T_BILD
    statistics[i, 2] = C_Q_BEM
    statistics[i, 3] = C_Q_BILD
    statistics[i, 4] = C_P_BEM
    statistics[i, 5] = C_P_BILD
    statistics[i, 6] = eta_BEM
    statistics[i, 7] = eta_BILD
    statistics[i, 8] = E_total_BEM
    statistics[i, 9] = E_total_BILD
    statistics[i, 10] = exit_code
    statistics[i, 11] = BILD_time
    statistics[i, 12] = BEM_opt_time
    statistics[i, 13] = eps_chord
    statistics[i, 14] = eps_twist
    statistics[i, 15] = eps_energy
    statistics[i, 16] = eps_max_LD

    # twist_chord_array[0, i, :] = chord_BILD
    # twist_chord_array[1, i, :] = chord_BEM
    # twist_chord_array[2, i, :] = twist_BILD
    # twist_chord_array[3, i, :] = twist_BEM
    
    # with open(f'full_factorial_sweep_w_advance_ratio_performance_metrics_fixed_bug_4_run_2.pickle', 'wb') as handle:
    #     pickle.dump(statistics[0:i, :], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if exit_code == 1:
        print('\n')
        print('Sweep number:                    ', j)
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

    else:
        print('NOT CONVERGED!!')
    # visualize_blade_design = 'y'
    # if visualize_blade_design == 'y':
    #     from lsdo_rotor.core.BILD.functions.plot_ideal_loading_blade_shape import plot_ideal_loading_blade_shape
    #     plot_ideal_loading_blade_shape(sim_BILD, sim_BEM)

# print(twist_chord_array)
# radius = sim_BILD['_radius'].flatten()


# fig, axs = plt.subplots(4,2, figsize = (14,10))

# for i in range(len(J_lst)):
#     axs[i, 0].plot(radius, twist_chord_array[0, i, :]/2, color='maroon')
#     axs[i, 0].plot(radius, twist_chord_array[0, i, :]/-2, color='maroon')
#     axs[i, 0].plot(radius, twist_chord_array[1, i, :]/2, color='navy')
#     axs[i, 0].plot(radius, twist_chord_array[1, i, :]/-2, color='navy')
#     axs[i, 0].set_xlabel('radius (m)')
#     axs[i, 0].set_ylabel('blade shape (m)')
#     axs[i, 0].axis('equal')

    
#     axs[i, 1].plot(radius, twist_chord_array[2, i, :]*180/np.pi, color='maroon', label='BILD method')
#     axs[i, 1].plot(radius, twist_chord_array[3, i, :]*180/np.pi, color='navy', label='BEM optimization')
#     axs[i, 1].set_xlabel('radius (m)')
#     axs[i, 1].set_ylabel('twist angle (deg)')
#     ax2 = axs[i, 1].twinx()
#     ax2.set_ylabel(f'J={J_lst[i]}') 
#     if i == 0:
#         axs[i, 1].legend()

#     # axs[i, 1].axis('equal')

# fig.tight_layout()
# plt.show()



