import numpy as np 
import openmdao.api as om
import omtools.api as ot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("darkgrid")

from lsdo_rotor.core.idealized_bemt_group import IdealizedBEMTGroup
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.get_smoothing_parameters import get_smoothing_parameters
from lsdo_rotor.get_airfoil_parameters import get_airfoil_parameters
from lsdo_rotor.get_rotor_dictionary import get_rotor_dictionary
from lsdo_rotor.get_external_rotor_data import get_external_rotor_data

def get_plot_sweeps_vs_r(airfoil_files, airfoil_names, rotor_files, geometry_array, RPM, rotor_diameter, V_inf, Re):
    num_airfoils = len(airfoil_files)
    if not rotor_files: 
            num_radial = len(geometry_array[0,:])
    else:
        external_rotor_data = get_external_rotor_data(rotor_files[0],rotor_files[1])
        num_radial = len(external_rotor_data[0])

    J = np.zeros((num_airfoils,num_radial))
    C_T_mat = np.zeros((num_airfoils,num_radial))
    C_P_mat = np.zeros((num_airfoils,num_radial))
    eta_total_mat = np.zeros((num_airfoils, num_radial))

    for k in range(len(airfoil_files)):
        rotor = get_rotor_dictionary(airfoil_files[k])
        mode = 2
        num_blades = rotor['num_blades']

        num_evaluations = 1
        
        if not rotor_files: 
            num_radial = len(geometry_array[0,:])
        else:
            external_rotor_data = get_external_rotor_data(rotor_files[0],rotor_files[1])
            num_radial = len(external_rotor_data[0])
        
        # print(num_radial)
        num_tangential = 1

        shape = (num_evaluations, num_radial, num_tangential)

        prob = om.Problem()

        group = ot.Group()
       

        group.create_indep_var('reference_radius', shape=1)
        group.create_indep_var('reference_position', shape=(1,3))
        group.create_indep_var('reference_x_dir', shape=(1, 3))
        group.create_indep_var('reference_y_dir', shape=(1, 3))
        group.create_indep_var('reference_z_dir', shape=(1, 3))
        group.create_indep_var('reference_inflow_velocity', shape=(1, 1, 1, 3))
        group.create_indep_var('reference_pitch', shape=1)
        group.create_indep_var('reference_chord', shape=1)
        group.create_indep_var('reference_axial_inflow_velocity', shape=1)
        group.create_indep_var('reference_blade_solidity', shape =1)
        group.create_indep_var('reference_tangential_inflow_velocity', shape=1)
        group.create_indep_var('reference_rotational_speed',shape=1)

        group.create_indep_var('hub_radius', shape=1)
        group.create_indep_var('rotor_radius', shape=1)
        group.create_indep_var('alpha', shape = 1)

        group.create_indep_var('slice_thickness', shape =1)
        group.create_indep_var('position', shape=(num_evaluations, 3))
        group.create_indep_var('x_dir', shape=(num_evaluations, 3))
        group.create_indep_var('y_dir', shape=(num_evaluations, 3))
        group.create_indep_var('z_dir', shape=(num_evaluations, 3))
        group.create_indep_var('inflow_velocity', shape=shape + (3,))
        group.create_indep_var('rotational_speed', shape = 1)
        group.create_indep_var('pitch', shape=(num_radial,))
        group.create_indep_var('chord', shape=(num_radial,))

        prob.model.add_subsystem('external_inputs_group', group, promotes=['*'])

        #------------------------------Adding main working group --> where majority of the model is implemented---------------------------------------------------------# 
        group = IdealizedBEMTGroup(
            mode = mode,
            rotor=rotor,
            num_evaluations=num_evaluations,
            num_radial=num_radial,
            num_tangential=num_tangential,
        )
        prob.model.add_subsystem('idealized_bemt_group', group, promotes=['*'])

        prob.setup(check=True)


        prob['rotor_radius'] = rotor_diameter/2
        prob['hub_radius'] = 0.15 * prob['rotor_radius']
        prob['slice_thickness'] = (prob['rotor_radius']-prob['hub_radius'])/ (num_radial-1)

        if not rotor_files:
            prob['chord'] = geometry_array[0,:]
            prob['pitch'] = geometry_array[1,:]
        else:
            external_rotor_data = get_external_rotor_data(rotor_files[0],rotor_files[1])
            prob['chord'] = external_rotor_data[0] * prob['rotor_radius']
            prob['pitch'] = external_rotor_data[1] * np.pi / 180. 
            rotor_J = external_rotor_data[2]
            rotor_CT = external_rotor_data[3]
            rotor_CP = external_rotor_data[4]
            rotor_eta = external_rotor_data[5]

        RPM = RPM

        # Set reference variables for ideal loading mode
        prob['reference_axial_inflow_velocity'] = V_inf                      # Adjust axial incoming velocity U_inf
        prob['reference_radius'] = 0.5
        prob['rotational_speed'] = RPM/60.
        prob['reference_rotational_speed'] = RPM/60.
        prob['reference_chord'] = 0.1
        prob['reference_blade_solidity'] = num_blades * prob['reference_chord'] / 2. / np.pi / prob['reference_radius']
        prob['reference_tangential_inflow_velocity'] = prob['rotational_speed'] * 2. * np.pi * prob['reference_radius']



        prob['reference_x_dir'][0, :] = [1., 0., 0.]
        prob['reference_y_dir'][0, :] = [0., 1., 0.]
        prob['reference_z_dir'][0, :] = [0., 0., 1.]
        prob['reference_inflow_velocity'][0, 0, 0, :] = [V_inf, 0., 0.]   


        for i in range(num_evaluations):
            prob['x_dir'][i, :] = [1., 0., 0.]
            prob['y_dir'][i, :] = [0., 1., 0.]
            prob['z_dir'][i, :] = [0., 0., 1.]
            for j in range(num_radial):
                prob['inflow_velocity'][i, j, 0, :] = [V_inf, 0., 0.]         # Adjust axial incoming velocity U_inf
        prob.run_model()

        torque = prob['BEMT_local_torque'].flatten()
        thrust = prob['BEMT_local_thrust'].flatten()
        radius = prob['_radius'].flatten()

        V = prob['reference_axial_inflow_velocity'] 
        Omega = prob['reference_rotational_speed'] * 2 * np.pi
        J[k,:] = prob['reference_axial_inflow_velocity']/prob['reference_rotational_speed']/(2 * prob['rotor_radius'])
        C_T_mat[k,:] = thrust / 1.2 / prob['rotational_speed']**2 / (2 * prob['rotor_radius'])**4
        P = 2 * np.pi * prob['rotational_speed'] * torque
        C_P_mat[k,:] = P / 1.2 / prob['rotational_speed']**3 / (2 * prob['rotor_radius'])**5
        eta_total_mat[k,:] = J[k,:] * C_T_mat[k,:] / C_P_mat[k,:]

        # print(thrust)
        # print(torque)

        #---------------------------------------Running Model-----------------------------------------------------#
        

    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    fig.suptitle('Rotor Parameter Sweeps vs. Radius with Blade Element Momentum Theory' + '\n' + 'RPM = {}, Re = {}, V = {}'.format(RPM, Re, V_inf),fontsize=23)
    for i in range(len(airfoil_files)):
        axs[0].plot(radius,  eta_total_mat[i,:], label = r'$\eta_{BEM}$ ' + airfoil_names[i])
        axs[1].plot(radius, C_T_mat[i,:], label = r'$C_{T, BEM}$ '+ airfoil_names[i])
        axs[2].plot(radius, C_P_mat[i,:], label = r'$C_{P, BEM}$ '+ airfoil_names[i])

    axs[0].set_xlim(left = 0)
    axs[0].set_ylabel(r'Efficiency $\eta$',fontsize=19)
    axs[0].set_xlabel('Radius',fontsize=19)
    axs[0].legend(fontsize=19)
    axs[1].set_xlim(left=0)
    axs[1].set_ylabel(r'Thrust Coefficient $C_T$',fontsize=19)
    axs[1].set_xlabel('Radius',fontsize=19)
    axs[1].legend(fontsize=19)
    axs[2].set_xlim(left=0)
    axs[2].set_ylabel(r'Power Coefficient $C_P$',fontsize=19)
    axs[2].set_xlabel('Radius',fontsize=19)
    axs[2].legend(fontsize=19)

    plt.tight_layout( rect=[0, 0.03, 1, 0.9])
    plt.show()

