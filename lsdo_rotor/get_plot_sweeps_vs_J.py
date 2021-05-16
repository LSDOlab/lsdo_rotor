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

def get_plot_sweeps_vs_J(airfoil_files, airfoil_names, rotor_files, geometry_array, RPM, rotor_diameter, max_V_inf, num_V_points, Re):
    num_airfoils = len(airfoil_files)
    
    Cl0 = np.zeros(num_airfoils,)
    Cdmin = np.zeros(num_airfoils,)
    a_cdmin = np.zeros(num_airfoils,)
    K = np.zeros(num_airfoils,)
    Cl_stall_plus = np.zeros(num_airfoils,)
    Cl_stall_minus = np.zeros(num_airfoils,)
    Cl_stall_minus = np.zeros(num_airfoils,)
    a_stall_plus = np.zeros(num_airfoils,)
    a_stall_minus = np.zeros(num_airfoils,)
    Cla = np.zeros(num_airfoils,)
    Cd_stall_plus = np.zeros(num_airfoils,)
    Cd_stall_minus = np.zeros(num_airfoils,)

    velocity_vec = np.linspace(0,max_V_inf,num_V_points)

    J = np.zeros((num_airfoils,len(velocity_vec)))
    C_T_mat = np.zeros((num_airfoils,len(velocity_vec)))
    C_P_mat = np.zeros((num_airfoils,len(velocity_vec)))
    eta_total_mat = np.zeros((num_airfoils, len(velocity_vec)))
    total_torque = np.zeros((num_airfoils, len(velocity_vec)))
    total_thrust = np.zeros((num_airfoils, len(velocity_vec)))

    for k in range(len(airfoil_files)):
        rotor = get_rotor_dictionary(airfoil_files[k],2,1000)    
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
        # 
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
        group.create_indep_var('alpha_stall', shape=1)
        group.create_indep_var('alpha_stall_minus', shape=1)
        # 
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

        

        # Set pitch and chord distribution for BEMT mode
        


        # Set reference variables for ideal loading mode
        prob['rotational_speed'] = RPM/60.
        prob['reference_rotational_speed'] = RPM/60.
        prob['reference_chord'] = 0.1
        prob['reference_blade_solidity'] = num_blades * prob['reference_chord'] / 2. / np.pi / prob['reference_radius']
        prob['reference_tangential_inflow_velocity'] = prob['rotational_speed'] * 2. * np.pi * prob['reference_radius']

        prob['reference_x_dir'][0, :] = [1., 0., 0.]
        prob['reference_y_dir'][0, :] = [0., 1., 0.]
        prob['reference_z_dir'][0, :] = [0., 0., 1.]
        

        for n in range(len(velocity_vec)):
            prob['reference_axial_inflow_velocity'] = velocity_vec[n]
            prob['reference_inflow_velocity'][0, 0, 0, :] = [velocity_vec[n], 0., 0.]
            for i in range(num_evaluations):
                prob['x_dir'][i, :] = [1., 0., 0.]
                prob['y_dir'][i, :] = [0., 1., 0.]
                prob['z_dir'][i, :] = [0., 0., 1.]
                for j in range(num_radial):
                    prob['inflow_velocity'][i, j, 0, :] = [velocity_vec[n], 0., 0.]

            prob.run_model()
            
            torque = prob['BEMT_local_torque']
            torque = torque[0]
            total_torque[k,n] = np.sum(torque)

            thrust = prob['BEMT_local_thrust']
            thrust = thrust[0]
            total_thrust[k,n] = np.sum(thrust)

            V = prob['reference_axial_inflow_velocity'] 
            Omega = prob['reference_rotational_speed'] * 2 * np.pi
            J[k,n] = prob['reference_axial_inflow_velocity']/prob['reference_rotational_speed']/(2 * prob['rotor_radius'])
            C_T_mat[k,n] = total_thrust[k,n] / 1.2 / prob['rotational_speed']**2 / (2 * prob['rotor_radius'])**4
            P = 2 * np.pi * prob['rotational_speed'] * total_torque[k,n]
            C_P_mat[k,n] = P / 1.2 / prob['rotational_speed']**3 / (2 * prob['rotor_radius'])**5
            eta_total_mat[k,n] = J[k,n] * C_T_mat[k,n] / C_P_mat[k,n]

            if eta_total_mat[k,n] < 0:
                eta_total_mat[k,n] = -1
            elif eta_total_mat[k,n] > 1:
                eta_total_mat[k,n] = -1
            # print(eta_total_mat)
            # print(k)
    
            prob.run_model()



    fig, axs = plt.subplots(1, 3, figsize=(12, 8))

    fig.suptitle('Rotor Parameter Sweeps vs. Advance Ratio with Blade Element Momentum Theory' + '\n' + 'RPM = {}'.format(RPM),fontsize=23)
    
    for i in range(len(airfoil_files)):
        axs[0].plot(J[0,:],  eta_total_mat[i,:], label = r'$\eta_{BEM}$ ' + airfoil_names[i])
        axs[1].plot(J[0,:], C_T_mat[i,:], label = r'$C_{T, BEM}$ '+ airfoil_names[i])
        axs[2].plot(J[0,:], C_P_mat[i,:], label = r'$C_{P, BEM}$ '+ airfoil_names[i])
        
    if rotor_files:
        axs[0].plot(rotor_J, rotor_eta, marker = '*', linestyle = 'None', label = r'$\eta_{data}$ ')
        axs[1].plot(rotor_J, rotor_CT, marker = '*',linestyle = 'None', label = r'$C_{T, data}$ ')
        axs[2].plot(rotor_J, rotor_CP, marker = '*',linestyle = 'None', label = r'$C_{P, data}$ ')  
        
    axs[0].set_ylim([0,1])
    axs[0].set_xlim(left = 0)
    axs[0].set_ylabel(r'Efficiency $\eta$',fontsize=19)
    axs[0].set_xlabel('Advance Ratio J',fontsize=19)
    axs[0].legend(fontsize=19)
    axs[1].set_xlim(left=0)
    axs[1].set_ylim(bottom=0)
    axs[1].set_ylabel(r'Thrust Coefficient $C_T$',fontsize=19)
    axs[1].set_xlabel('Advance Ratio J',fontsize=19)
    axs[1].legend(fontsize=19)
    axs[2].set_xlim(left=0)
    axs[2].set_ylim(bottom=0)
    axs[2].set_ylabel(r'Power Coefficient $C_P$',fontsize=19)
    axs[2].set_xlabel('Advance Ratio J',fontsize=19)
    axs[2].legend(fontsize=19)
        # axs[2].grid()


    plt.tight_layout( rect=[0, 0.03, 1, 0.9])
    plt.show()

    