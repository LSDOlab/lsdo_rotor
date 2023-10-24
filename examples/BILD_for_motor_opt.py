import numpy as np 
from python_csdl_backend import Simulator
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 13})

from lsdo_rotor.core.BILD.BILD_run_model import BILDRunModel


# Constants
g = 9.81
num_blades = 3
altitude = 1000 # in (m)

# Design parameters
nominal_radius = 1.3716
nominal_gross_mass = 2992.349
nominal_ref_chord = 0.2
r_R = 0.5
L_o_D = 9.2
cruise_speed = 58 

# Sweep variabls
gross_mass = np.array([2680, 2880, 3080, 3280, 3480, 3680])
gross_weight = gross_mass * g
drag = gross_weight / L_o_D

num_nodes = 250
num_radial = 40
num_tangential = 1
shape = tuple((num_nodes, num_radial, num_tangential))

# Visualize optimal blade design
visualize_blade_design = 'y' # 'y'/'n'

num_sweeps = len(gross_mass)

radius_vec = np.zeros((num_sweeps, num_radial))
chord = np.zeros((num_sweeps, num_radial))
twist = np.zeros((num_sweeps, num_radial))
torque_vec = np.zeros((num_sweeps, ))
rpm_sweep = np.zeros((num_sweeps, ))
R_vec = np.zeros((num_sweeps, ))
eta_vec = np.zeros((num_sweeps, ))

for i in range(len(gross_mass)):

    # "free" variable
    if i == 0 or i == 1:
        rpm_vec = np.linspace(500, 1400, num_nodes)
    elif i == 1:
        rpm_vec = np.linspace(500, 1200, num_nodes)
    else:
        rpm_vec = np.linspace(500, 1100, num_nodes)
    
    scale_factor = gross_mass[i] / nominal_gross_mass
    radius = scale_factor * nominal_radius
    reference_chord = scale_factor * nominal_ref_chord
    

    sim_BILD = Simulator(BILDRunModel(
                rotor_radius=radius,
                reference_chord=reference_chord,
                reference_radius= r_R * radius,
                rpm=rpm_vec,
                Vx=cruise_speed,
                altitude=altitude,
                shape=shape,
                num_blades=num_blades,
                airfoil_name='NACA_4412',
                airfoil_polar=None,
            ), display_scripts=True)
    sim_BILD.run()

    thrust_sweep = sim_BILD['total_thrust'].flatten()
    torque_sweep = sim_BILD['total_torque'].flatten()
    eta_sweep = sim_BILD['eta'].flatten()

    thrust_difference = abs(thrust_sweep - drag[i])
    index = np.where(thrust_difference == min(thrust_difference))[0]

    chord_sweep = sim_BILD['_mod_local_chord'][index, :, :].flatten()
    twist_sweep = sim_BILD['_local_twist_angle'][index, :, :].flatten() * 180/np.pi
    radius_sweep = sim_BILD['_radius'][index, :, :].flatten()

    thrust = thrust_sweep[index]
    torque = torque_sweep[index]
    rpm = rpm_vec[index]
    eta = eta_sweep[index]

    chord[i, :] = chord_sweep
    twist[i, :] = twist_sweep
    radius_vec[i, :] = radius_sweep
    torque_vec[i] = torque
    rpm_sweep[i] = rpm
    R_vec[i] = radius
    eta_vec[i] = eta



    print('Thrust:          ', thrust[0])
    print('Drag:            ', drag[i])
    print('---------------')
    print('Torque:          ', torque[0])
    print('RPM:             ', rpm[0])
    print('eta:             ', eta[0])


fig, axs = plt.subplots(3, 2, figsize = (16,12))

axs[0, 0].plot(radius_vec[0, :], chord[0, :]/2, color='navy', label='chord')
axs[0, 0].plot(radius_vec[0, :], chord[0, :]/-2, color='navy')
axs[0, 0].set_title(f'Gross mass: {gross_mass[0]}, rpm: {round(rpm_sweep[0], 2)}, Radius: {round(R_vec[0], 2)} m,' + '\n' + f'torque: {round(torque_vec[0], 2)} (N-m), eta: {round(eta_vec[0],3)}')
axs[0, 0].axis('equal')
axs[0, 0].legend(loc='upper center')
t0 = axs[0, 0].twinx()
axs[0, 0].set_ylabel('Blade shape (m)')
t0.plot(radius_vec[0, :], twist[0, :], color='maroon', label='twist')
t0.legend(loc='upper right')
t0.set_ylabel('Twist angle (deg)')

axs[0, 1].plot(radius_vec[1, :], chord[1, :]/2, color='navy')
axs[0, 1].plot(radius_vec[1, :], chord[1, :]/-2, color='navy')
axs[0, 1].set_title(f'Gross mass: {gross_mass[1]}, rpm: {round(rpm_sweep[1], 2)}, Radius: {round(R_vec[1], 2)} m,' + '\n' + f'torque: {round(torque_vec[1], 2)} (N-m), eta: {round(eta_vec[1], 3)}')
axs[0, 1].axis('equal')
t1 = axs[0, 1].twinx()
axs[0, 1].set_ylabel('Blade shape (m)')
t1.plot(radius_vec[1, :], twist[1, :], color='maroon')
t1.set_ylabel('Twist angle (deg)')

axs[1, 0].plot(radius_vec[2, :], chord[2, :]/2, color='navy')
axs[1, 0].plot(radius_vec[2, :], chord[2, :]/-2, color='navy')
axs[1, 0].set_title(f'Gross mass: {gross_mass[2]}, rpm: {round(rpm_sweep[2], 2)}, Radius: {round(R_vec[2], 2)} m,' + '\n' + f'torque: {round(torque_vec[2], 2)} (N-m), eta: {round(eta_vec[2], 3)}')
axs[1, 0].axis('equal')
t2 = axs[1, 0].twinx()
axs[1, 0].set_ylabel('Blade shape (m)')
t2.plot(radius_vec[2, :], twist[2, :], color='maroon')
t2.set_ylabel('Twist angle (deg)')

axs[1, 1].plot(radius_vec[3, :], chord[3, :]/2, color='navy')
axs[1, 1].plot(radius_vec[3, :], chord[3, :]/-2, color='navy')
axs[1, 1].set_title(f'Gross mass: {gross_mass[3]}, rpm: {round(rpm_sweep[3], 2)}, Radius: {round(R_vec[3], 2)} m,' + '\n' + f'torque: {round(torque_vec[3], 2)} (N-m), eta: {round(eta_vec[3], 3)}')
axs[1, 1].axis('equal')
t3 = axs[1, 1].twinx()
axs[1, 1].set_ylabel('Blade shape (m)')
t3.plot(radius_vec[3, :], twist[3, :], color='maroon')
t3.set_ylabel('Twist angle (deg)')

axs[2, 0].plot(radius_vec[4, :], chord[4, :]/2, color='navy')
axs[2, 0].plot(radius_vec[4, :], chord[4, :]/-2, color='navy')
axs[2, 0].set_title(f'Gross mass: {gross_mass[4]}, rpm: {round(rpm_sweep[4], 2)}, Radius: {round(R_vec[4], 2)} m,' + '\n' + f'torque: {round(torque_vec[4], 2)} (N-m), eta: {round(eta_vec[4], 3)}')
axs[2, 0].axis('equal')
t4 = axs[2, 0].twinx()
axs[2, 0].set_ylabel('Blade shape (m)')
t4.plot(radius_vec[4, :], twist[4, :], color='maroon')
t4.set_ylabel('Twist angle (deg)')

axs[2, 1].plot(radius_vec[5, :], chord[5, :]/2, color='navy')
axs[2, 1].plot(radius_vec[5, :], chord[5, :]/-2, color='navy')
axs[2, 1].set_title(f'Gross mass: {gross_mass[5]}, rpm: {round(rpm_sweep[5], 2)}, Radius: {round(R_vec[5], 2)} m,' + '\n' + f'torque: {round(torque_vec[5], 2)} (N-m), eta: {round(eta_vec[5], 2)}')
axs[2, 1].axis('equal')
t5 = axs[2, 1].twinx()
axs[2, 1].set_ylabel('Blade shape (m)')
t5.plot(radius_vec[5, :], twist[5, :], color='maroon')
t5.set_ylabel('Twist angle (deg)')


fig.tight_layout()
plt.show()

exit()
print(best_thrust[best_thrust_ind])
print(rpm[best_thrust_ind])
print(sim_BILD['total_thrust'].flatten()[best_thrust_ind])
print(drag[0])
print(sim_BILD['_local_chord'][best_thrust_ind,:,:].flatten())
print(sim_BILD['_radius'][best_thrust_ind,:,:].flatten())

print(np.where(best_thrust == min(best_thrust)))
# print(sim_BILD['total_torque'])
# print(sim_BILD['eta'])

exit()


for i in range(len(J)):
    if not Vx:
        n = rpm/60
        Vx = J[i] * n * 2 * rotor_radius
        print('Vx---------------------------------',Vx)
    for j in range(len(reference_radius)):
        sim_BILD = Simulator(BILDRunModel(
            rotor_radius=rotor_radius,
            reference_chord=reference_chord,
            reference_radius=reference_radius[j],
            rpm=rpm,
            Vx=Vx,
            altitude=altitude,
            shape=shape,
            num_blades=num_blades,
        ))
        sim_BILD.run()

        local_chord = sim_BILD['_local_chord'].flatten()
        # print(local_chord)
        local_twist_angle = sim_BILD['_local_twist_angle'].flatten() * 180 /np.pi

        twist_chord_array[0, i, j, :] = local_chord
        twist_chord_array[1, i, j, :] = local_twist_angle

radius = sim_BILD['_radius'].flatten()

fig, axs = plt.subplots(2,2, figsize = (14,10))
color = np.linspace(0, 0.8, len(reference_radius))
for i in range(len(reference_radius)):
    axs[0, 0].plot(radius, twist_chord_array[0, 0, i, :]/2, color=f'{color[i]}', label=f'r/R={reference_radius[i]}')
    axs[0, 0].plot(radius, twist_chord_array[0, 0, i, :]/-2, color=f'{color[i]}')
    # axs[0, 0].vlines(x=reference_radius[i], ymin=0.15/-2, ymax=0.15/2, color='maroon')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Radius (m)')
    axs[0, 0].set_ylabel('Blade Shape (m)')
    axs[0, 0].set_title(f'J={J[0]}')
    axs[0, 0].axis('equal')

    # axs[0, 1].vlines(x=reference_radius[i], ymin=0.15/-2, ymax=0.15/2, color='maroon')
    # axs[0, 1].scatter(reference_radius[i], 0.15/-2, color='maroon')
    # axs[0, 1].scatter(reference_radius[i], 0.15/2, color='maroon')
    axs[0, 1].plot(radius, twist_chord_array[0, 1, i, :]/2, color=f'{color[i]}', label=f'r/R={reference_radius[i]}')
    axs[0, 1].plot(radius, twist_chord_array[0, 1, i, :]/-2, color=f'{color[i]}')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Radius (m)')
    axs[0, 1].set_ylabel('Blade Shape (m)')
    axs[0, 1].set_title(f'J={J[1]}')
    axs[0, 1].axis('equal')
    
    # axs[1, 0].vlines(x=reference_radius[i], ymin=0.15/-2, ymax=0.15/2, color='maroon')
    axs[1, 0].plot(radius, twist_chord_array[0, 2, i, :]/2, color=f'{color[i]}', label=f'r/R={reference_radius[i]}')
    axs[1, 0].plot(radius, twist_chord_array[0, 2, i, :]/-2, color=f'{color[i]}')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Radius (m)')
    axs[1, 0].set_ylabel('Blade Shape (m)')
    axs[1, 0].set_title(f'J={J[2]}')
    axs[1, 0].axis('equal')

    # axs[1, 1].vlines(x=reference_radius[i], ymin=0.15/-2, ymax=0.15/2, color='maroon')
    axs[1, 1].plot(radius, twist_chord_array[0, 3, i, :]/2, color=f'{color[i]}', label=f'r/R={reference_radius[i]}')
    axs[1, 1].plot(radius, twist_chord_array[0, 3, i, :]/-2, color=f'{color[i]}')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Radius (m)')
    axs[1, 1].set_ylabel('Blade Shape (m)')
    axs[1, 1].set_title(f'J={J[3]}')
    axs[1, 1].axis('equal')

fig.tight_layout()
plt.show()
exit()
if visualize_blade_design == 'y':
    from lsdo_rotor.core.BILD.functions.plot_ideal_loading_blade_shape import plot_ideal_loading_blade_shape
    plot_ideal_loading_blade_shape(sim_BILD)
