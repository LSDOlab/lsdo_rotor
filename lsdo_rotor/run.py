import numpy as np 
import openmdao.api as om
from csdl import Model 
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")


from core.rotor_model import RotorModel

from airfoil.get_surrogate_model import get_surrogate_model
from functions.get_rotor_dictionary import get_rotor_dictionary
from functions.get_max_LD_parameters_reference_chord import get_max_LD_parameters_reference_chord

# Conference papers 
# - VFS: more the studies (Darshan)
# - AIAA aviation (summer 2023) in SD; will have ULI special sessions 
# - AIAA scitech (Jan 2023): more software side (potentially Marius)

# Point sets --> Extract twist and chord from central geometry represented in
# point sets contain coordinate in the body-fixed frame; the x-axis of the body fixed frame is not aligned with the x-axis of propeller/rotor

# first point set nr by 2 point sets 
# second point set necessary (2,): going from centroid of hub to tip of the nose of hub 
# third point set: nr,nt,3: set of velocity vectors; this takes care of
# stability derivatives 

# generalize everything 
# modifications: n by 2 point set and (n,) from caddee and convert internally to
# twist and chord, radius, hub radius 
# compute partials: re-compute Cl and Cd 

# weights model also has point sets 

"""
    Mode settings
        1 --> Ideal-Loading Design Method
        2 --> BEM
        3 --> Dynamic Inflow 1: Pitt-Peters
"""
mode = 3

# The following airfoils are currently available: 'NACA_4412', 'Clark_Y', 'NACA_0012', 'mh117'; We recommend 'NACA_4412' or 'Clark_Y'
airfoil = 'NACA_4412_extended_range' 
interp = get_surrogate_model(airfoil)
ne = 1
diameter_vec = 2 * np.ones((ne,))#np.array([2,2])                 # Rotor diameter in (m)
RPM_vec = 1000 * np.ones((ne,))#np.array([1000,1000])
Omega_vec = RPM_vec * 2 * np.pi / 60                                                    
V_inf_vec = np.linspace(0,40,ne)#np.array([40,40])              # Cruise speed in (m/s)

i_vec = 90 * np.ones((ne,))#np.array([45,90])              # Rotor disk tilt angle in (deg); 
#   --> 90 degrees means purely axial inflow
#   --> 0  degrees means purely edgewise flight    

Vx_vec = V_inf_vec * np.sin(i_vec * np.pi/180)
Vy_vec = V_inf_vec * np.cos(i_vec * np.pi/180)
Vz_vec = np.zeros((ne,))#np.array([0,0])

# x, y, z velocity components (m/s); Vx is the axial velocity component 
# Vx                  = 25   # Axial inflow velocity (i.e. V_inf) 
# Vy                  = Vx * np.cos(beta * np.pi/180)   # Side slip velocity in the propeller plane
# Vz                  = 0   # Side slip velocity in the propeller plane

# Specify number of blades and altitude
num_blades          = 5
altitude            = 1000        # (in m)

# The following 3 parameters are used for mode 1 only! The user has to
# specify three parameters for optimal blade design 
reference_radius_vec = 0.61 * np.ones((ne,))#np.array([0.61,0.61])  # Specify the reference radius; We recommend radius / 2
reference_chord_vec  = 0.1 * np.ones((ne,))# np.array([0.1,0.1])                # Specify the reference chord length at the reference radius (in m)

# The following parameters are used for mode 2 only
# Change these parameters if you want to chord and twist profile to vary
# linearly from rotor hub to tip
root_chord          = 0.10       # Chord length at the root/hub
root_twist          = 40        # Twist angle at the blade root/hub (deg)
tip_chord           = 0.10       # Chord length at the tip
tip_twist           = 20        # Twist angle at the blade tip (deg)

# Consider the following two lines if you want to use an exiting rotor geometry:
# IMPORTANT: you can only use mode if you want to use an exiting rotor geometry.
use_external_rotor_geometry = 'n'           # [y/n] If you want to use an existing rotor geometry 
geom_data = np.loadtxt('ildm_geometry_1.txt')  # if 'y', make sure you have the right .txt file with the chord distribution in the
                                            # second column and the twist distribution in the third column

# The following parameters specify the radial and tangential mesh as well as the
# number of time steps; 
num_evaluations     = ne        # Discretization in time:                 Only required if your operating conditions change in time
if use_external_rotor_geometry == 'y':
    num_radial      = len(geom_data[:,1])     
else:
    num_radial      = 30      # Discretization in spanwise direction:   Should always be at least 25
num_tangential      = 30       # Discretization in tangential direction: Only required if Vy,Vz are non-zero; recommend at least 20


# Specify some post-processing options 
plot_rotor_blade_shape  = 'y'     # Only implemented for mode 1 [y/n]
plot_rotor_performance  = 'n'     # Not yet implemented [y/n]
print_rotor_performance = 'y'     # [y/n]


#---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
# ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, Vx, RPM, altitude)
rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, i_vec,RPM_vec,V_inf_vec,diameter_vec,num_evaluations,num_radial,num_tangential) #, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord, beta)

shape = (num_evaluations, num_radial, num_tangential)



model = Model()

rotor_model = RotorModel(
    mode=mode,
    rotor=rotor,
    num_evaluations=num_evaluations,
    num_radial=num_radial,
    num_tangential=num_tangential,
)
model.add(rotor_model,'rotor_model')#, promotes = ['*'])


import csdl_lite 

# sim = csdl_lite.Simulator(
#     rotor_model,
#     analytics=True,
#     node_rvs=['inflow_velocity','rotational_speed','_rho_pitt_peters','rotor_radius'],
#     )

# exit()
sim = Simulator(model)


if use_external_rotor_geometry == 'y':
    sim['chord'] = geom_data[:,1] 
    sim['pitch'] = geom_data[:,2] * np.pi/180
else:
    sim['chord'] = np.linspace(
        root_chord,
        tip_chord,
        num_radial,
    )

    sim['pitch'] = np.linspace(
        root_twist * np.pi / 180.,
        tip_twist * np.pi / 180.,
        num_radial,
    )

# Adjust axial incoming velocity V_inf

for i in range(num_evaluations):
    sim['x_dir'][i, :] = [1., 0., 0.]
    sim['y_dir'][i, :] = [0., 1., 0.]
    sim['z_dir'][i, :] = [0., 0., 1.]
    for j in range(num_radial):    
        for k in range(num_tangential):    
            sim['inflow_velocity'][i, j, k, :] = [Vx_vec[i], Vy_vec[i], Vz_vec[i]]
    sim['rotational_speed'][i] = RPM_vec[i] /60.
    sim['rotor_radius'][i] = diameter_vec[i] / 2
    sim['hub_radius'][i]   = 0.2 * diameter_vec[i] / 2
    sim['dr'][i] = ((diameter_vec[i] / 2)-(0.2 * diameter_vec[i] / 2))/ (num_radial -1)

    # ILDM parameters     
    sim['reference_chord'][i] = reference_chord_vec[i]
    sim['reference_radius'][i] = reference_radius_vec[i]
    sim['reference_blade_solidity'][i] = num_blades * reference_chord_vec[i] / 2. / np.pi / reference_radius_vec[i]
    sim['ildm_tangential_inflow_velocity'][i] = RPM_vec[i]/60. * 2. * np.pi * reference_radius_vec[i]
    sim['ildm_axial_inflow_velocity'][i] = Vx_vec[i]  
    sim['ildm_rotational_speed'][i] = RPM_vec[i]/60.







# sim.prob.driver = om.pyOptSparseDriver()
# sim.prob.driver.options['optimizer'] = 'SNOPT'
# sim.prob.driver.opt_settings['Major feasibility tolerance'] = 2.e-5
# sim.prob.driver.opt_settings['Major optimality tolerance'] = 2.e-5

# sim.prob.driver = om.ScipyOptimizeDriver()
# sim.prob.driver.options['optimizer'] = 'SLSQP'
# sim.prob.driver.opt_settings['ACC'] = 1e-4
# sim.prob.driver.opt_settings['maxiter'] = 200


# sim.prob.model.add_design_variable('chord',lower = 0.002, upper = 0.30)
# sim.prob.model.add_design_variable('pitch',lower = 10*np.pi/180, upper = 80*np.pi/180)

# sim.prob.setup(check=True)
sim.run()

# model.visualize_sparsity(recursive=True)

# import matplotlib.pyplot as plt 
# fig, axs = plt.subplots(3,1,figsize=(8, 9))
# T = sim['total_thrust']
# Q = sim['total_torque']

# np.savetxt('PP_thrust.txt',T.flatten())
# np.savetxt('PP_torque.txt',T.flatten())


# C_T = T / (rotor['density'] * np.pi * (diameter_vec/2)**2 * (Omega_vec *(diameter_vec/2))**2)
# C_Q = Q / (rotor['density'] * np.pi * (diameter_vec/2)**3 * (Omega_vec *(diameter_vec/2))**2)

# eta = V_inf_vec * T.flatten() / (Omega_vec * Q.flatten())
# J = V_inf_vec / (RPM_vec / 60 * diameter_vec)


# axs[0].plot(J,eta)
# axs[1].plot(J,C_T)
# axs[2].plot(J,C_Q)

# print(eta)
# print(J)

# print(T)
# # print(ux)
# print(Q)
# plt.show()
# sim.prob.run_driver()
# sim.prob.check_partials(compact_print=True,step=1e-5, form='central')
# sim.prob.check_totals(step=1e-5, form='central')
# totals = sim.prob.compute_totals()
# print(totals)
exit()


import matplotlib.pyplot as plt 
plt.plot(sim['_radius'][0,:,:].flatten(), sim['_local_chord'][0,:,:].flatten()/2,color = '#1f77b4', marker = 'o')
plt.plot(sim['_radius'][0,:,:].flatten(), sim['_local_chord'][0,:,:].flatten()/-2,color = '#1f77b4', marker = 'o')

plt.plot(sim['_radius'][1,:,:].flatten(), sim['_mod_local_chord'][1,:,:].flatten()/2, color= '#ff7f0e', marker = 'o')
plt.plot(sim['_radius'][1,:,:].flatten(), sim['_mod_local_chord'][1,:,:].flatten()/-2, color= '#ff7f0e', marker = 'o')

plt.plot(sim['_radius'][2,:,:].flatten(), sim['_local_chord'][2,:,:].flatten()/2, color ='#2ca02c', marker = 'o')
plt.plot(sim['_radius'][2,:,:].flatten(), sim['_local_chord'][2,:,:].flatten()/-2, color ='#2ca02c', marker = 'o')
plt.show()
exit()
    # output_array[n,0] = sim['total_thrust']
    # output_array[n,1] = sim['total_torque']
    # output_array[n,2] = sim['pitching_moment']
    # output_array[n,3] = sim['rolling_moment']




# fig, axs = plt.subplots(2,2,figsize=(8, 9))
# axs[0,0].plot(i_vec, output_array[:,0])
# axs[0,0].set_xlabel('Rotor disk tilt angle (deg)')
# axs[0,0].set_ylabel('Thrust (N)')
# axs[0,0].scatter(i_vec[-1],1209.92626989, label = 'BEM prediction')
# axs[0,0].legend()

# axs[0,1].plot(i_vec, output_array[:,1])
# axs[0,1].set_xlabel('Rotor disk tilt angle (deg)')
# axs[0,1].set_ylabel('Torque (N-m)')
# axs[0,1].scatter(i_vec[-1],259.70831792, label = 'BEM prediction')
# axs[0,1].legend()

# axs[1,0].plot(i_vec, output_array[:,2])
# axs[1,0].set_xlabel('Rotor disk tilt angle (deg)')
# axs[1,0].set_ylabel('Rolling moment (N-m)')

# axs[1,1].plot(i_vec, output_array[:,3])
# axs[1,1].set_xlabel('Rotor disk tilt angle (deg)')
# axs[1,1].set_ylabel('Pitching moment (N-m)')


# print(sim['phi_distribution'].flatten() * 180/np.pi)
# print(sim['total_thrust'])
# print(sim['total_thrust_2'])
# print(np.sum(sim['_local_thrust'][0,:,:]))
# print(np.sum(sim['_local_thrust_2'][0,:,:]))
# print(np.sum(sim['_local_thrust'][1,:,:]))
# print(np.sum(sim['_local_thrust_2'][1,:,:]))
# print(np.sum(sim['_local_thrust'][2,:,:]))
# print(np.sum(sim['_local_thrust_2'][2,:,:]))


# print(sim['total_torque'])
# print(np.sum(sim['_local_torque'][0,:,:]))
# print(np.sum(sim['_local_torque_2'][0,:,:]))
# print(np.sum(sim['_local_torque'][1,:,:]))
# print(np.sum(sim['_local_torque_2'][1,:,:]))
# print(np.sum(sim['_local_torque'][2,:,:]))
# print(np.sum(sim['_local_torque_2'][2,:,:]))

# 

# print(sim['_radius'])



# print(sim['ildm_tangential_inflow_velocity'])
# print(sim['_tangential_inflow_velocity'])
# print(sim['eta_2'])

# print(sim['_ux'][0,:,:])
# print(sim['_ux_2'][0,:,:])


# print(sim['_ux'][0,:,:]-sim['_ux_2'][0,:,:])


# print(np.sum(sim['_local_thrust'][1,:,:]))


# print(sim['total_thrust'])
# print(sim['total_thrust_2'])

# print(sim['total_torque'])
# print(sim['total_torque_2'])

# print(sim['phi_distribution'])
# print(sim['Cl_2'])

# print((sim['_pitch'] - sim['phi_distribution']) * 180/np.pi)
# print(sim['Cl_2'])
# print(sim['_re_pitt_peters'][0,:,0])
# print(sim['_tangential_inflow_velocity'][0,:,0])
# print(np.sum(np.cos(sim['_theta'][0,0,:])))
# print(sim['Re'])
# print(np.average(sim['_ux']))
# plt.suptitle(r'Pitt-Peters dynamic inflow' + 
#              '\n' + r'$V_{\infty}$' + ' = {} (m/s), '.format(25) + 
#              'D = {} (m), '.format(rotor_diameter) + 'RPM = {} '.format(RPM))
# plt.tight_layout()
# plt.savefig('pitt_peters_wo_ut_V_inf_0.png')
# plt.show()
exit()
# sim.prob.run_driver()
# print(sim['phi_distribution']*180/np.pi)
# exit()

# test_array = np.zeros((num_radial,6))

# test_array[:,0] = sim['_chord'].flatten()
# test_array[:,1] = sim['_pitch'].flatten() * 180/np.pi
# test_array[:,2] = sim['Cl_2'].flatten()
# test_array[:,3] = sim['Cd_2'].flatten() 
# test_array[:,4] = sim['_local_energy_loss'].flatten() 
# test_array[:,5] = sim['_radius'].flatten() 
# np.savetxt('BEM_opt_geometry_J_2_R_2_Vx_60.txt',test_array)

# test_array[:,0] = sim['_local_chord'].flatten()
# test_array[:,1] = sim['_local_twist_angle'].flatten() * 180/np.pi
# test_array[:,2] = np.ones((num_radial,)) * rotor['ideal_Cl_ref_chord']
# test_array[:,3] = np.ones((num_radial,)) * rotor['ideal_Cd_ref_chord']
# test_array[:,4] = sim['local_ideal_energy_loss'].flatten()
# test_array[:,5] = sim['_radius'].flatten()
# np.savetxt('ildm_geometry_J_25_R_2_Vx_75.txt',test_array)




# sim.prob.check_totals(compact_print=True)
# sim.prob.check_partials(compact_print=True)


# print(sim['total_thrust'])
# print(sim['total_thrust_2'])
# print(sim['total_torque'])
# print(sim['total_torque_2'])
# print(sim['_chord'])
# print(sim['Cl_2'])
# print(sim['Cd_2'])
# print(sim['phi_distribution'])
# print(sim['_local_inflow_angle'])
# print(sim['_pitch']*180/np.pi)
# np.savetxt('BEM_inflow_angle.txt',sim['phi_distribution'].flatten())
# np.savetxt('ideal_inflow_angle.txt',sim['_local_inflow_angle'].flatten())
# print(sim['eta_2'])
# print(sim['AoA'])

# print(sim['Cl_2'])
# print(sim['Cl'])
# print(sim['prandtl_loss_factor'])
# print(sim['F'])

# print(sim['Cl'])
# print(sim['rho_test'])
# print(sim['_ux_2'])
# print(sim['_ux'])
# print(sim['chord'])
# print(sim['pitch']* 180/np.pi)
# print(sim['_radius'])
# print(sim['total_thrust_2'])
    # print(sim['_local_thrust'].shape)
    # print(sim['_local_thrust'])
    # # print(sim['ux_num'],'ux_num')
    # # print(sim['ux_den'],'ux_den')
    # # print(sim['ux_num_by_den'],'ux_num_by_den')
    # # 
    # # print(sim['rho_ildm'])
    
    # # print(sim['F'])
    # # print(sim['phi_reference_ildm'],'phi_ref')
    # # print(sim['axial_induced_velocity_ideal_loading_BEM_step'], 'ux')
    # # print(sim['tangential_induced_velocity_ideal_loading_BEM_step'],'ut')
    # # print(sim['ideal_loading_constant'],'ideal_loading_constant')
    # # print(sim['eta_2'])
    # # print(sim['c'],'c')
    # # print(sim['b'],'b')
    # # print(sim['eta_2'].flatten().reshape(6,5))
    # # print(sim['_back_comp_axial_induced_velocity'].flatten().reshape(6,5),'ux_dist')
    # # print(sim['_back_comp_tangential_induced_velocity'].flatten().reshape(6,5),'ut_dist')
    # # print(rotor['ideal_Cl_ref_chord'])
    # # print(sim['_local_thrust'].flatten().reshape(6,5))
    # # print(sim['_local_torque'].flatten().reshape(6,5))
    # # print(sim['_local_chord'].flatten().reshape(6,5))
    # # print(sim['_radius'])
    # # print(sim['F_dist'].flatten().reshape(6,5))

# import matplotlib.pyplot as plt 
# from pytikz.matplotlib_utils import use_latex_fonts
# import seaborn as sns
# sns.set()
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times"],
# })
# fig = plt.figure(figsize=(8, 9))




# sub1 = fig.add_subplot(5,3,1)
# sub1.plot(radius, ideal_chord1 / 2, marker = 'o',markersize = 4,color= 'firebrick')
# sub1.plot(radius,  ideal_chord1 / -2, marker = 'o',markersize = 4,color= 'firebrick')
# sub1.plot(radius, BEM_opt_chord1 / 2,marker = 'o',markersize = 4, color= 'navy')
# sub1.plot(radius, BEM_opt_chord1 / -2, marker = 'o',markersize = 4,color= 'navy')



# plt.show()
# exit()
# fig, axs = plt.subplots(5, 3,figsize=(8, 9))

# ildm_data1 = np.loadtxt('ildm_geometry_J_05_R_2_Vx_15.txt')
# ideal_chord1 = ildm_data1[:,0]
# ideal_twist1 = ildm_data1[:,1]
# ideal_loss1  = ildm_data1[:,4]

# ildm_data2 = np.loadtxt('ildm_geometry_J_1_R_2_Vx_30.txt')
# ideal_chord2 = ildm_data2[:,0]
# ideal_twist2 = ildm_data2[:,1]
# ideal_loss2  = ildm_data2[:,4]

# ildm_data3 = np.loadtxt('ildm_geometry_J_15_R_2_Vx_45.txt')
# ideal_chord3 = ildm_data3[:,0]
# ideal_twist3 = ildm_data3[:,1]
# ideal_loss3  = ildm_data3[:,4]

# ildm_data4 = np.loadtxt('ildm_geometry_J_2_R_2_Vx_60.txt')
# ideal_chord4 = ildm_data4[:,0]
# ideal_twist4 = ildm_data4[:,1]
# ideal_loss4  = ildm_data4[:,4]

# ildm_data5 = np.loadtxt('ildm_geometry_J_25_R_2_Vx_75.txt')
# ideal_chord5 = ildm_data5[:,0]
# ideal_twist5 = ildm_data5[:,1]
# ideal_loss5  = ildm_data5[:,4]

# BEM_opt_data1 = np.loadtxt('BEM_opt_geometry_J_05_R_2_Vx_15.txt')
# BEM_opt_chord1 = BEM_opt_data1[:,0]
# BEM_opt_twist1 = BEM_opt_data1[:,1]
# BEM_opt_loss1 = BEM_opt_data1[:,4]

# BEM_opt_data2 = np.loadtxt('BEM_opt_geometry_J_1_R_2_Vx_30.txt')
# BEM_opt_chord2 = BEM_opt_data2[:,0]
# BEM_opt_twist2 = BEM_opt_data2[:,1]
# BEM_opt_loss2 = BEM_opt_data2[:,4]

# BEM_opt_data3 = np.loadtxt('BEM_opt_geometry_J_15_R_2_Vx_45.txt')
# BEM_opt_chord3 = BEM_opt_data3[:,0]
# BEM_opt_twist3 = BEM_opt_data3[:,1]
# BEM_opt_loss3 = BEM_opt_data3[:,4]

# BEM_opt_data4 = np.loadtxt('BEM_opt_geometry_J_2_R_2_Vx_60.txt')
# BEM_opt_chord4 = BEM_opt_data4[:,0]
# BEM_opt_twist4 = BEM_opt_data4[:,1]
# BEM_opt_loss4 = BEM_opt_data4[:,4]

# BEM_opt_data5 = np.loadtxt('BEM_opt_geometry_J_25_R_2_Vx_75.txt')
# BEM_opt_chord5 = BEM_opt_data5[:,0]
# BEM_opt_twist5 = BEM_opt_data5[:,1]
# BEM_opt_loss5 = BEM_opt_data5[:,4]

# radius = BEM_opt_data1[:,-1]

# axs[0,0].plot(radius, ideal_chord1 / 2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[0,0].plot(radius,  ideal_chord1 / -2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[0,0].plot(radius, BEM_opt_chord1 / 2,marker = 'o',markersize = 4, color= 'navy')
# axs[0,0].plot(radius, BEM_opt_chord1 / -2, marker = 'o',markersize = 4,color= 'navy')
# axs[0,1].plot(radius, ideal_twist1, marker = 'o',markersize = 4,color= 'firebrick')
# axs[0,1].plot(radius, BEM_opt_twist1, marker = 'o',markersize = 4,color= 'navy')
# axs[0,2].plot(radius, ideal_loss1, marker = 'o',markersize = 4,color= 'firebrick')
# axs[0,2].plot(radius, BEM_opt_loss1, marker = 'o',markersize = 4,color= 'navy')

# axs[1,0].plot(radius, ideal_chord2 / 2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[1,0].plot(radius,  ideal_chord2 / -2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[1,0].plot(radius, BEM_opt_chord2 / 2,marker = 'o',markersize = 4, color= 'navy')
# axs[1,0].plot(radius, BEM_opt_chord2 / -2, marker = 'o',markersize = 4,color= 'navy')
# axs[1,1].plot(radius, ideal_twist2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[1,1].plot(radius, BEM_opt_twist2, marker = 'o',markersize = 4,color= 'navy')
# axs[1,2].plot(radius, ideal_loss2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[1,2].plot(radius, BEM_opt_loss2, marker = 'o',markersize = 4,color= 'navy')

# axs[2,0].plot(radius, ideal_chord3 / 2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[2,0].plot(radius,  ideal_chord3 / -2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[2,0].plot(radius, BEM_opt_chord3 / 2,marker = 'o',markersize = 4, color= 'navy')
# axs[2,0].plot(radius, BEM_opt_chord3 / -2, marker = 'o',markersize = 4,color= 'navy')
# axs[2,1].plot(radius, ideal_twist3, marker = 'o',markersize = 4,color= 'firebrick')
# axs[2,1].plot(radius, BEM_opt_twist3, marker = 'o',markersize = 4,color= 'navy')
# axs[2,2].plot(radius, ideal_loss3, marker = 'o',markersize = 4,color= 'firebrick')
# axs[2,2].plot(radius, BEM_opt_loss3, marker = 'o',markersize = 4,color= 'navy')

# axs[3,0].plot(radius, ideal_chord4 / 2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[3,0].plot(radius,  ideal_chord4 / -2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[3,0].plot(radius, BEM_opt_chord4 / 2,marker = 'o',markersize = 4, color= 'navy')
# axs[3,0].plot(radius, BEM_opt_chord4 / -2, marker = 'o',markersize = 4,color= 'navy')
# axs[3,1].plot(radius, ideal_twist4, marker = 'o',markersize = 4,color= 'firebrick')
# axs[3,1].plot(radius, BEM_opt_twist4, marker = 'o',markersize = 4,color= 'navy')
# axs[3,2].plot(radius, ideal_loss4, marker = 'o',markersize = 4,color= 'firebrick')
# axs[3,2].plot(radius, BEM_opt_loss4, marker = 'o',markersize = 4,color= 'navy')

# axs[4,0].plot(radius, ideal_chord5 / 2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[4,0].plot(radius,  ideal_chord5 / -2, marker = 'o',markersize = 4,color= 'firebrick')
# axs[4,0].plot(radius, BEM_opt_chord5 / 2,marker = 'o',markersize = 4, color= 'navy')
# axs[4,0].plot(radius, BEM_opt_chord5 / -2, marker = 'o',markersize = 4,color= 'navy')
# axs[4,1].plot(radius, ideal_twist5, marker = 'o',markersize = 4,color= 'firebrick')
# axs[4,1].plot(radius, BEM_opt_twist5, marker = 'o',markersize = 4,color= 'navy')
# axs[4,2].plot(radius, ideal_loss5, marker = 'o',markersize = 4,color= 'firebrick')
# axs[4,2].plot(radius, BEM_opt_loss5, marker = 'o',markersize = 4,color= 'navy')

# fig.tight_layout()
# plt.show()
# exit()
if mode == 1:
    if print_rotor_performance == 'y':
        from functions.print_ideal_loading_output import print_ideal_loading_output
        print_ideal_loading_output(sim['total_thrust'].flatten(),sim['total_torque'].flatten(), \
            Vx, RPM/60,sim['_radius'].flatten(),sim['_local_thrust'].flatten(),sim['_local_torque'].flatten(), \
            sim['_back_comp_axial_induced_velocity'].flatten(), sim['_back_comp_tangential_induced_velocity'].flatten(), \
            sim['_local_chord'].flatten(), sim['_local_twist_angle'].flatten(), num_blades, sim['dr'].flatten(), rotor_diameter/2 )      
    if plot_rotor_blade_shape == 'y':    
        from functions.plot_ideal_loading_blade_shape import plot_ideal_loading_blade_shape
        plot_ideal_loading_blade_shape(sim['_radius'].flatten(),sim['_local_chord'].flatten(),sim['_mod_local_chord'].flatten(),sim['_local_twist_angle'].flatten())
else:
    if print_rotor_performance == 'y':
        from functions.print_bem_analysis_output import print_bem_analysis_output
        print_bem_analysis_output(sim['total_thrust'].flatten(),sim['total_torque'].flatten(), \
            Vx, RPM/60,sim['_radius'].flatten(),sim['_local_thrust'].flatten(),sim['_local_torque'].flatten(), \
            sim['_ux'].flatten(), sim['_ut'].flatten(), \
            sim['_chord'].flatten(), sim['_pitch'].flatten(), num_blades, sim['dr'].flatten(), rotor_diameter/2 )
        
        




