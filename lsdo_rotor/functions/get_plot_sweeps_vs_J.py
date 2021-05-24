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
from lsdo_rotor.functions.get_airfoil_parameters import get_airfoil_parameters
from lsdo_rotor.functions.get_rotor_dictionary import get_rotor_dictionary
from lsdo_rotor.functions.get_external_rotor_data import get_external_rotor_data
from lsdo_utils.comps.bspline_comp import   get_bspline_mtx, BsplineComp

def get_plot_sweeps_vs_J(airfoil, rotor_files, RPM, rotor_diameter, max_V_inf, num_V_points, num_blades, altitude, n_radial, optimized_pitch_cp, optimized_chord_cp, pitch_cp, chord_cp ):
    
    velocity_vec = np.linspace(0,max_V_inf,num_V_points)

    J = np.zeros((len(velocity_vec),))
    C_T_mat = np.zeros((len(velocity_vec),))
    C_P_mat = np.zeros((len(velocity_vec),))
    eta_total_mat = np.zeros(( len(velocity_vec),))
    total_torque = np.zeros((len(velocity_vec),))
    total_thrust = np.zeros((len(velocity_vec),))


    rotor = get_rotor_dictionary(airfoil,num_blades,altitude)    
    mode = 2
    num_blades = rotor['num_blades']

    num_evaluations = 1
    num_radial = n_radial
    num_tangential = 1

    chord_num_cp = chord_cp
    pitch_num_cp = pitch_cp

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

    group.create_indep_var('pitch_cp', shape = (pitch_num_cp,))
    group.create_indep_var('chord_cp', shape = (chord_num_cp,))

    prob.model.add_subsystem('external_inputs_group', group, promotes=['*'])

    chord_A = get_bspline_mtx(chord_num_cp, num_radial, order = 4)
    pitch_A = get_bspline_mtx(pitch_num_cp, num_radial, order = 4)
    
    comp = BsplineComp(
        num_pt=num_radial,
        num_cp=chord_num_cp,
        in_name='chord_cp',
        jac=chord_A,
        out_name='chord',
    )
    prob.model.add_subsystem('chord_bspline_comp', comp, promotes = ['*'])


    comp = BsplineComp(
        num_pt=num_radial,
        num_cp=pitch_num_cp,
        in_name='pitch_cp',
        jac=pitch_A,
        out_name='pitch',
    )
    prob.model.add_subsystem('pitch_bspline_comp', comp, promotes = ['*'])

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


    prob['chord_cp'] = np.genfromtxt(optimized_chord_cp)

    prob['pitch_cp'] = np.genfromtxt(optimized_pitch_cp)

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
        total_torque[n] = np.sum(torque)

        thrust = prob['BEMT_local_thrust']
        thrust = thrust[0]
        total_thrust[n] = np.sum(thrust)

        V = prob['reference_axial_inflow_velocity'] 
        Omega = prob['reference_rotational_speed'] * 2 * np.pi
        J[n] = prob['reference_axial_inflow_velocity']/prob['reference_rotational_speed']/(2 * prob['rotor_radius'])
        C_T_mat[n] = total_thrust[n] / 1.2 / prob['rotational_speed']**2 / (2 * prob['rotor_radius'])**4
        P = 2 * np.pi * prob['rotational_speed'] * total_torque[n]
        C_P_mat[n] = P / 1.2 / prob['rotational_speed']**3 / (2 * prob['rotor_radius'])**5
        eta_total_mat[n] = J[n] * C_T_mat[n] / C_P_mat[n]

        if eta_total_mat[n] < 0:
            eta_total_mat[n] = -1
        elif eta_total_mat[n] > 1:
            eta_total_mat[n] = -1
        # print(eta_total_mat)
        # print(k)

        prob.run_model()



    fig, axs = plt.subplots(1, 3, figsize=(12, 8))

    fig.suptitle('Rotor Parameter Sweeps vs. Advance Ratio with Blade Element Momentum Theory' + '\n' + 'RPM = {}'.format(RPM),fontsize=23)
    

    axs[0].plot(J,  eta_total_mat, label = r'$\eta_{BEM}$ ')
    axs[1].plot(J, C_T_mat, label = r'$C_{T, BEM}$ ')
    axs[2].plot(J, C_P_mat, label = r'$C_{P, BEM}$ ')
        
    # if rotor_files:
    #     axs[0].plot(rotor_J, rotor_eta, marker = '*', linestyle = 'None', label = r'$\eta_{data}$ ')
    #     axs[1].plot(rotor_J, rotor_CT, marker = '*',linestyle = 'None', label = r'$C_{T, data}$ ')
    #     axs[2].plot(rotor_J, rotor_CP, marker = '*',linestyle = 'None', label = r'$C_{P, data}$ ')  
        
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

    