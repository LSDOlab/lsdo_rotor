import numpy as np
import openmdao.api as om
import omtools.api as ot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("darkgrid")


from lsdo_rotor.core.idealized_bemt_group import IdealizedBEMTGroup
from lsdo_rotor.rotor_parameters import RotorParameters

propeller_name = ' APC Sport 10 x 6 '
airfoil_filenames = [
    'xf-NACA4412-il-50000.txt',
    # 'xf-NACA4412-il-100000.txt',
    # 'xf-NACA4412-il-200000.txt',
    'xf-clarky-il-50000.txt',
    # 'xf-clarky-il-100000.txt',
    # 'xf-clarky-il-200000.txt',
    'xf-e63-il-50000.txt',
    # 'xf-e63-il-100000.txt',
    # 'xf-e63-il-200000.txt',
]

# Cl0 = np.zeros(9,)
# Cdmin = np.zeros(9,)
# a_cdmin = np.zeros(9,)
# K = np.zeros(9,)
# Cl_stall_plus = np.zeros(9,)
# Cl_stall_minus = np.zeros(9,)
# Cl_stall_minus = np.zeros(9,)
# a_stall_plus = np.zeros(9,)
# a_stall_minus = np.zeros(9,)
# Cla = np.zeros(9,)
# Cd_stall_plus = np.zeros(9,)
# Cd_stall_minus = np.zeros(9,)


Cl0 = np.zeros(3,)
Cdmin = np.zeros(3,)
a_cdmin = np.zeros(3,)
K = np.zeros(3,)
Cl_stall_plus = np.zeros(3,)
Cl_stall_minus = np.zeros(3,)
Cl_stall_minus = np.zeros(3,)
a_stall_plus = np.zeros(3,)
a_stall_minus = np.zeros(3,)
Cla = np.zeros(3,)
Cd_stall_plus = np.zeros(3,)
Cd_stall_minus = np.zeros(3,)


velocity_vec = np.linspace(0,30,101)

# J = np.zeros((9,len(velocity_vec)))
# C_T_mat = np.zeros((9,len(velocity_vec)))
# C_P_mat = np.zeros((9,len(velocity_vec)))
# eta_total_mat = np.zeros((9, len(velocity_vec)))
# total_torque = np.zeros((9, len(velocity_vec)))
# total_thrust = np.zeros((9, len(velocity_vec)))

J = np.zeros((3,len(velocity_vec)))
C_T_mat = np.zeros((3,len(velocity_vec)))
C_P_mat = np.zeros((3,len(velocity_vec)))
eta_total_mat = np.zeros((3, len(velocity_vec)))
total_torque = np.zeros((3, len(velocity_vec)))
total_thrust = np.zeros((3, len(velocity_vec)))

# C_T_mat = np.zeros((3,100))
# C_P_mat = np.zeros((3,100))
# eta_total_mat = np.zeros((3,100))
# P = np.zeros((3,100))

for k in range(len(airfoil_filenames)):
    data = np.loadtxt(airfoil_filenames[k], delimiter=',', skiprows=1, dtype=str)
    data = data[11:]
    np.savetxt("data.txt", data, fmt="%s")
    with open('data.txt', 'r') as f:
        data = f.read().split()
        floats = []
        for elem in data:
            try:
                floats.append(float(elem))
            except ValueError:
                pass
    airfoil_data = np.array(floats)
    num_col = 7
    num_row = int(len(airfoil_data)/ num_col)
    airfoil_data = airfoil_data.reshape(num_row, num_col)
    alpha_data = airfoil_data[:,0]
    CL_data = airfoil_data[:,1]
    CD_data = airfoil_data[:,2]
    
    Cl0[k] = float(CL_data[np.where(alpha_data == 0)])
    # print(Cl0)
    Cdmin[k] = float(np.min(CD_data))
    a_cdmin_raw = alpha_data[np.where(CD_data == Cdmin[k])] * np.pi / 180
    if len(a_cdmin_raw) > 1:
        a_cdmin[k] = float(np.average(a_cdmin_raw))
    elif len(a_cdmin_raw) == 1:
        a_cdmin[k] = float(a_cdmin_raw)

    K[k] = float((CD_data[np.where(alpha_data == 5)] - Cdmin[k]) / ((5 * np.pi / 180) - a_cdmin[k]) **2)

    Cl_stall_plus[k] = float(np.max(CL_data))
    Cl_stall_minus[k] = float(np.min(CL_data))

    a_stall_plus[k] = float(alpha_data[np.where(CL_data == Cl_stall_plus[k])] * np.pi / 180)
    a_stall_minus[k] = float(alpha_data[np.where(CL_data == Cl_stall_minus[k])] * np.pi / 180)

    Cla[k] = float((CL_data[np.where(alpha_data == 5)] - CL_data[np.where(alpha_data == -2.5)]) / (7.5 * np.pi / 180))

    # Cla = float((Cl_stall_plus - Cl_stall_minus) / (( alpha_data[np.where(CL_data == Cl_stall_plus)] -alpha_data[np.where(CL_data == Cl_stall_minus)]) * np.pi/180) )

    Cd_stall_plus[k] = float(CD_data[np.where(CL_data == Cl_stall_plus[k])])
    Cd_stall_minus[k] = float(CD_data[np.where(CL_data == Cl_stall_minus[k])])
    for key in [
        Cl0,
        Cdmin,
        a_cdmin,
        K,   
        Cl_stall_plus,
        Cl_stall_minus,
        a_stall_plus,
        a_stall_minus,
        Cla,
        Cd_stall_plus,
        Cd_stall_minus,
    ]:
        print(key) 
    # exit()

# for i in range(len(airfoil_filenames)):
    rotor = RotorParameters(
        num_blades=2,
        Cl0 = Cl0[k],
        Cla = Cla[k],
        Cdmin = Cdmin[k],
        K = K[k],
        alpha_Cdmin = a_cdmin[k],
        a_stall_plus = a_stall_plus[k],
        a_stall_minus = a_stall_minus[k],
        Cl_stall_plus = Cl_stall_plus[k],
        Cl_stall_minus = Cl_stall_minus[k],
        Cd_stall_plus = Cd_stall_plus[k],
        Cd_stall_minus = Cd_stall_minus[k],
        AR = 10,

    )
    """
        Mode settings
            1 --> Ideal Loading
            2 --> BEMT
    """
    mode = 2
    print(Cl0[1])
    num_blades = rotor['num_blades']

    num_evaluations = 1
    num_radial = 18
    num_tangential = 1

    shape = (num_evaluations, num_radial, num_tangential)

    prob = om.Problem()

    group = ot.Group()
    #
    group.create_indep_var('Cl0', shape = 1)
    group.create_indep_var('Cla', shape = 1)
    group.create_indep_var('Cdmin', shape = 1)
    group.create_indep_var('K', shape = 1)
    group.create_indep_var('alpha_Cdmin', shape = 1)
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

    group.create_indep_var('AR', shape=1)
    group.create_indep_var('Cl_stall', shape = 1)
    group.create_indep_var('Cd_stall', shape = 1)
    group.create_indep_var('Cl_stall_minus', shape = 1)
    group.create_indep_var('Cd_stall_minus', shape = 1)
    group.create_indep_var('smoothing_tolerance', shape = 1)

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

    geometry = 'APC_10_6_geometry.txt'
    data = np.loadtxt(geometry, delimiter=',', skiprows=1, dtype=str)
    with open('APC_10_6_geometry.txt', 'r') as f:
        data = f.read().split()
        floats = []
        for elem in data:
            try:
                floats.append(float(elem))
            except ValueError:
                pass
    floats = np.array(floats)
    num_rows = int(len(floats)/3)
    num_cols = 3
    apc_geometry_data = np.reshape(floats, (num_rows, num_cols))
    normalized_chord = apc_geometry_data[:,1]
    twist = apc_geometry_data[:,2]

    RPM = 5000

    prob['rotor_radius'] = 0.254/2
    prob['hub_radius'] = 0.15 * prob['rotor_radius']
    prob['slice_thickness'] = (prob['rotor_radius']-prob['hub_radius'])/ (num_radial-1)

    # Set pitch and chord distribution for BEMT mode
    
    # prob['pitch'] = np.linspace(65,20,num_radial) * np.pi / 180.
    prob['pitch'] = twist * np.pi / 180.
    # prob['chord'] = 0.1 * np.ones((1,num_radial))
    prob['chord'] = normalized_chord * prob['rotor_radius']
    # print(len(prob['chord'])) # = num_radial



    prob.model.add_design_var('chord',lower=0.07, upper=0.2)
    prob.model.add_design_var('pitch',lower=0, upper= np.pi/2)



    # Set reference variables for ideal loading mode
    prob['reference_axial_inflow_velocity'] = 15.
    prob['reference_radius'] = 0.5
    prob['rotational_speed'] = RPM/60.
    prob['reference_rotational_speed'] = RPM/60.
    prob['reference_chord'] = 0.1
    prob['reference_blade_solidity'] = num_blades * prob['reference_chord'] / 2. / np.pi / prob['reference_radius']
    prob['reference_tangential_inflow_velocity'] = prob['rotational_speed'] * 2. * np.pi * prob['reference_radius']

    prob['reference_x_dir'][0, :] = [1., 0., 0.]
    prob['reference_y_dir'][0, :] = [0., 1., 0.]
    prob['reference_z_dir'][0, :] = [0., 0., 1.]
    prob['reference_inflow_velocity'][0, 0, 0, :] = [15., 0., 0.]
    for i in range(num_evaluations):
        prob['x_dir'][i, :] = [1., 0., 0.]
        prob['y_dir'][i, :] = [0., 1., 0.]
        prob['z_dir'][i, :] = [0., 0., 1.]
        for j in range(num_radial):
            prob['inflow_velocity'][i, j, 0, :] = [15., 0., 0.]

    prob['alpha'] = 6. * np.pi /180.

    prob['Cl0'] = Cl0[k]

    prob['Cdmin'] =  Cdmin[k]
    prob['alpha_Cdmin'] =  a_cdmin[k]

    prob['alpha_stall'] = a_stall_plus[k]
    prob['alpha_stall_minus'] = a_stall_minus[k]

    prob['AR'] = 10.

    prob['Cl_stall'] = Cl_stall_plus[k]
    prob['Cl_stall_minus'] = Cl_stall_minus[k]

    prob['Cla'] = Cla[k]
    prob['K'] = K[k]

    prob['Cd_stall'] = Cd_stall_plus[k]
    prob['Cd_stall_minus'] = Cd_stall_minus[k]

    prob['smoothing_tolerance'] = 5 * np.pi / 180
    
#     prob.run_model()

#     torque = prob['BEMT_local_torque'].flatten()
#     thrust = prob['BEMT_local_thrust'].flatten()

#     J = prob['reference_axial_inflow_velocity']/prob['reference_rotational_speed']/(2 * prob['rotor_radius'])
#     C_T_mat[k,:] = thrust / 1.2 / prob['rotational_speed']**2 / (2 * prob['rotor_radius'])**4
#     P[k,:] = 2 * np.pi * prob['rotational_speed'] * torque
#     C_P_mat[k,:] = P[k,:] / 1.2 / prob['rotational_speed']**3 / (2 * prob['rotor_radius'])**5
#     eta_total_mat[k,:] = prob['BEMT_local_efficiency'].flatten()

# fig, axs = plt.subplots(1, 3, figsize=(12, 8))
# fig.suptitle('BEM parameter sweeps vs. radius for three airfoils' + '\n' + r'$V_{\infty}$ = 15 m/s',fontsize=20)
# axs[0].plot(prob['_radius'].flatten(), eta_total_mat[0,:], label = r'$\eta_{BEM}$ NACA 4412 ')
# axs[0].plot(prob['_radius'].flatten(), eta_total_mat[1,:], label = r'$\eta_{BEM}$ Clark-Y ')
# axs[0].plot(prob['_radius'].flatten(), eta_total_mat[2,:], label = r'$\eta_{BEM}$ Eppler E63')
# axs[0].set_ylabel(r'Efficiency $\eta$',fontsize=16)
# axs[0].set_xlabel('Radius r (m)',fontsize=16)
# axs[0].set_ylim([0, 1])
# axs[0].legend(fontsize = 16)

# axs[1].plot(prob['_radius'].flatten(), C_T_mat[0,:], label = r'$C_{T, BEM}$ NACA 4412 ')
# axs[1].plot(prob['_radius'].flatten(), C_T_mat[1,:], label = r'$C_{T, BEM}$ Clark-Y ')
# axs[1].plot(prob['_radius'].flatten(), C_T_mat[2,:], label = r'$C_{T, BEM}$ Eppler E63')
# axs[1].set_ylabel(r'Thrust Coefficient $C_T$',fontsize=16)
# axs[1].set_xlabel('Radius r (m)',fontsize=16)
# axs[1].legend(fontsize = 16)

# axs[2].plot(prob['_radius'].flatten(), C_P_mat[0,:], label = r'$C_{P, BEM}$ NACA 4412 ')
# axs[2].plot(prob['_radius'].flatten(), C_P_mat[1,:], label = r'$C_{P, BEM}$ Clark-Y ')
# axs[2].plot(prob['_radius'].flatten(), C_P_mat[2,:], label = r'$C_{P, BEM}$ Eppler E63')
# axs[2].set_ylabel(r'Power Coefficient $C_P$',fontsize=16)
# axs[2].set_xlabel('Radius r (m)',fontsize=16)
# axs[2].legend(fontsize = 16)

# plt.tight_layout( rect=[0, 0.03, 1, 0.9])
# plt.show()
# exit()
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
        # print(J[k,:], 'J')
        C_T_mat[k,n] = total_thrust[k,n] / 1.2 / prob['rotational_speed']**2 / (2 * prob['rotor_radius'])**4
        P = 2 * np.pi * prob['rotational_speed'] * total_torque[k,n]
        C_P_mat[k,n] = P / 1.2 / prob['rotational_speed']**3 / (2 * prob['rotor_radius'])**5
        eta_total_mat[k,n] = J[k,n] * C_T_mat[k,n] / C_P_mat[k,n]

        if eta_total_mat[k,n] < 0:
            eta_total_mat[k,n] = -1
        elif eta_total_mat[k,n] > 1:
            eta_total_mat[k,n] = -1
        print(eta_total_mat)
        print(k)
        # print(velocity_vec[n])
        prob.run_model()

performance = 'APC_10_6_performance_5000.txt'
data = np.loadtxt(performance, delimiter=',', skiprows=1, dtype=str)
with open('APC_10_6_performance_5000.txt', 'r') as f:
    data = f.read().split()
    floats = []
    for elem in data:
        try:
            floats.append(float(elem))
        except ValueError:
            pass
floats = np.array(floats)
num_rows = int(len(floats)/4)
num_cols = 4
apc_performance_data = np.reshape(floats, (num_rows, num_cols))
APC_J = apc_performance_data[:,0]
APC_CT = apc_performance_data[:,1]
APC_CP = apc_performance_data[:,2]
APC_eta = apc_performance_data[:,3]

# fig, axs = plt.subplots(3, 3, figsize=(14, 10))

# fig.suptitle('Model validation of BEMT mode using'  + propeller_name +  'propeller data at {} RPM'.format(RPM),fontsize=16)
# axs[0,0].plot(J[0,:],  eta_total_mat[0,:], label = r'$\eta_{BEMT}$ NACA 4412 ')
# axs[0,0].plot(J[0,:],  eta_total_mat[3,:], label = r'$\eta_{BEMT}$ Clark-Y ')
# axs[0,0].plot(J[0,:],  eta_total_mat[6,:], label = r'$\eta_{BEMT}$ Eppler E63')
# axs[0,0].plot(APC_J, APC_eta, marker = '*', label = r'$\eta_{APC}$ ')
# axs[0,0].set_ylim([0,1])
# axs[0,0].set_xlim([0, 1.15])
# axs[0,0].set_ylabel('Re = 50 000'+ '\n' + '\n' + r'$\eta$')
# axs[0,0].set_xlabel('J')
# axs[0,0].legend()
# axs[0,0].grid()

# axs[0,1].plot(J[0,:], C_T_mat[0,:], label = r'$C_{T, BEMT}$ NACA 4412 ')
# axs[0,1].plot(J[0,:], C_T_mat[3,:], label = r'$C_{T, BEMT}$ Clark-Y ')
# axs[0,1].plot(J[0,:], C_T_mat[6,:], label = r'$C_{T, BEMT}$ Eppler E63')
# axs[0,1].plot(APC_J, APC_CT, marker = '*', label = r'$C_{T, APC}$ ')
# axs[0,1].set_xlim([0, 1.15])
# axs[0,1].set_ylim([0,0.2])
# axs[0,1].set_ylabel(r'$C_T$')
# axs[0,1].set_xlabel('J')
# axs[0,1].legend()
# axs[0,1].grid()

# axs[0,2].plot(J[0,:], C_P_mat[0,:], label = r'$C_{P, BEMT}$ NACA 4412 ')
# axs[0,2].plot(J[0,:], C_P_mat[3,:], label = r'$C_{P, BEMT}$ Clark-Y ')
# axs[0,2].plot(J[0,:], C_P_mat[6,:], label = r'$C_{P, BEMT}$ Eppler E63')
# axs[0,2].plot(APC_J, APC_CP, marker = '*', label = r'$C_{P, APC}$ ')
# axs[0,2].set_xlim([0, 1.15])
# axs[0,2].set_ylim([0,0.125])
# axs[0,2].set_ylabel(r'$C_P$')
# axs[0,2].set_xlabel('J')
# axs[0,2].legend()
# axs[0,2].grid()

# axs[1,0].plot(J[0,:],  eta_total_mat[1,:], label = r'$\eta_{BEMT}$ NACA 4412 ')
# axs[1,0].plot(J[0,:],  eta_total_mat[4,:], label = r'$\eta_{BEMT}$ Clark-Y ')
# axs[1,0].plot(J[0,:],  eta_total_mat[7,:], label = r'$\eta_{BEMT}$ Eppler E63')
# axs[1,0].plot(APC_J, APC_eta, marker = '*', label = r'$\eta_{APC}$ ')
# axs[1,0].set_ylim([0,1])
# axs[1,0].set_xlim([0, 1.15])
# axs[1,0].set_ylabel('Re = 100 000'+ '\n' + '\n' + r'$\eta$')
# axs[1,0].set_xlabel('J')
# axs[1,0].legend()
# axs[1,0].grid()

# axs[1,1].plot(J[0,:], C_T_mat[1,:], label = r'$C_{T, BEMT}$ NACA 4412 ')
# axs[1,1].plot(J[0,:], C_T_mat[4,:], label = r'$C_{T, BEMT}$ Clark-Y ')
# axs[1,1].plot(J[0,:], C_T_mat[7,:], label = r'$C_{T, BEMT}$ Eppler E63')
# axs[1,1].plot(APC_J, APC_CT, marker = '*', label = r'$C_{T, APC}$ ')
# axs[1,1].set_xlim([0, 1.15])
# axs[1,1].set_ylim([0,0.2])
# axs[1,1].set_ylabel(r'$C_T$')
# axs[1,1].set_xlabel('J')
# axs[1,1].legend()
# axs[1,1].grid()

# axs[1,2].plot(J[0,:], C_P_mat[1,:], label = r'$C_{P, BEMT}$ NACA 4412 ')
# axs[1,2].plot(J[0,:], C_P_mat[4,:], label = r'$C_{P, BEMT}$ Clark-Y ')
# axs[1,2].plot(J[0,:], C_P_mat[7,:], label = r'$C_{P, BEMT}$ Eppler E63')
# axs[1,2].plot(APC_J, APC_CP, marker = '*', label = r'$C_{P, APC}$ ')
# axs[1,2].set_xlim([0, 1.15])
# axs[1,2].set_ylim([0,0.125])
# axs[1,2].set_ylabel(r'$C_P$')
# axs[1,2].set_xlabel('J')
# axs[1,2].legend()
# axs[1,2].grid()

# axs[2,0].plot(J[0,:],  eta_total_mat[2,:], label = r'$\eta_{BEMT}$ NACA 4412 ')
# axs[2,0].plot(J[0,:],  eta_total_mat[5,:], label = r'$\eta_{BEMT}$ Clark-Y ')
# axs[2,0].plot(J[0,:],  eta_total_mat[8,:], label = r'$\eta_{BEMT}$ Eppler E63')
# axs[2,0].plot(APC_J, APC_eta, marker = '*', label = r'$\eta_{APC}$ ')
# axs[2,0].set_ylim([0,1])
# axs[2,0].set_xlim([0, 1.15])
# axs[2,0].set_ylabel('Re = 200 000'+ '\n' + '\n' + r'$\eta$')
# axs[2,0].set_xlabel('J')
# axs[2,0].legend()
# axs[2,0].grid()

# axs[2,1].plot(J[0,:], C_T_mat[2,:], label = r'$C_{T, BEMT}$ NACA 4412 ')
# axs[2,1].plot(J[0,:], C_T_mat[5,:], label = r'$C_{T, BEMT}$ Clark-Y ')
# axs[2,1].plot(J[0,:], C_T_mat[8,:], label = r'$C_{T, BEMT}$ Eppler E63')
# axs[2,1].plot(APC_J, APC_CT, marker = '*', label = r'$C_{T, APC}$ ')
# axs[2,1].set_xlim([0, 1.15])
# axs[2,1].set_ylim([0,0.2])
# axs[2,1].set_ylabel(r'$C_T$')
# axs[2,1].set_xlabel('J')
# axs[2,1].legend()
# axs[2,1].grid()

# axs[2,2].plot(J[0,:], C_P_mat[2,:], label = r'$C_{P, BEMT}$ NACA 4412 ')
# axs[2,2].plot(J[0,:], C_P_mat[5,:], label = r'$C_{P, BEMT}$ Clark-Y ')
# axs[2,2].plot(J[0,:], C_P_mat[8,:], label = r'$C_{P, BEMT}$ Eppler E63')
# axs[2,2].plot(APC_J, APC_CP, marker = '*', label = r'$C_{P, APC}$ ')
# axs[2,2].set_xlim([0, 1.15])
# axs[2,2].set_ylim([0,0.125])
# axs[2,2].set_ylabel(r'$C_P$')
# axs[2,2].set_xlabel('J')
# axs[2,2].legend()
# axs[2,2].grid()

# plt.tight_layout( rect=[0, 0.03, 1, 0.95])
# plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 8))

fig.suptitle('BEM model validation using'  + propeller_name +  'propeller data' + '\n' + 'RPM = {}, Re = 50 000'.format(RPM),fontsize=23)
axs[0].plot(J[0,:],  eta_total_mat[0,:], label = r'$\eta_{BEM}$ NACA 4412 ')
axs[0].plot(J[0,:],  eta_total_mat[1,:], label = r'$\eta_{BEM}$ Clark-Y ')
axs[0].plot(J[0,:],  eta_total_mat[2,:], label = r'$\eta_{BEM}$ Eppler E63')
axs[0].plot(APC_J, APC_eta, marker = '*', linestyle = 'None', label = r'$\eta_{APC}$ ')
axs[0].set_ylim([0,1])
axs[0].set_xlim([0, 1.15])
axs[0].set_ylabel(r'Efficiency $\eta$',fontsize=19)
axs[0].set_xlabel('Advance Ratio J',fontsize=19)
axs[0].legend(fontsize=19)
# axs[0].grid()

axs[1].plot(J[0,:], C_T_mat[0,:], label = r'$C_{T, BEM}$ NACA 4412 ')
axs[1].plot(J[0,:], C_T_mat[1,:], label = r'$C_{T, BEM}$ Clark-Y ')
axs[1].plot(J[0,:], C_T_mat[2,:], label = r'$C_{T, BEM}$ Eppler E63')
axs[1].plot(APC_J, APC_CT, marker = '*',linestyle = 'None', label = r'$C_{T, APC}$ ')
axs[1].set_xlim([0, 1.15])
axs[1].set_ylim([0,0.2])
axs[1].set_ylabel(r'Thrust Coefficient $C_T$',fontsize=19)
axs[1].set_xlabel('Advance Ratio J',fontsize=19)
axs[1].legend(fontsize=19)
# axs[1].grid()

axs[2].plot(J[0,:], C_P_mat[0,:], label = r'$C_{P, BEM}$ NACA 4412 ')
axs[2].plot(J[0,:], C_P_mat[1,:], label = r'$C_{P, BEM}$ Clark-Y ')
axs[2].plot(J[0,:], C_P_mat[2,:], label = r'$C_{P, BEM}$ Eppler E63')
axs[2].plot(APC_J, APC_CP, marker = '*',linestyle = 'None', label = r'$C_{P, APC}$ ')
axs[2].set_xlim([0, 1.15])
axs[2].set_ylim([0,0.125])
axs[2].set_ylabel(r'Power Coefficient $C_P$',fontsize=19)
axs[2].set_xlabel('Advance Ratio J',fontsize=19)
axs[2].legend(fontsize=19)
# axs[2].grid()


plt.tight_layout( rect=[0, 0.03, 1, 0.9])
plt.show()

