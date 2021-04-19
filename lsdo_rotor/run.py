import numpy as np
import openmdao.api as om
import omtools.api as ot
import matplotlib.pyplot as plt

from lsdo_rotor.core.idealized_bemt_group import IdealizedBEMTGroup
from lsdo_rotor.rotor_parameters import RotorParameters


#------------------------------Set airfoil file name-----------------------------------------------------------------------------#
airfoil_filename = 'xf-e63-il-50000.txt'                                                            

#------------------------------Extracting Data From Airfoil Polar----------------------------------------------------------------#
data = np.loadtxt(airfoil_filename, delimiter=',', skiprows=1, dtype=str)
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

Cl0 = float(CL_data[np.where(alpha_data == 0)])
# Cl0 = set manually if need be

Cdmin = float(np.min(CD_data))
# Cdmin 

a_cdmin = alpha_data[np.where(CD_data == Cdmin)] * np.pi / 180
# a_cdmin = set manually if need be
if len(a_cdmin) > 1:
    a_cdmin = float(np.average(a_cdmin))
elif len(a_cdmin) == 1:
    a_cdmin = float(a_cdmin)

K = float((CD_data[np.where(alpha_data == 5)] - Cdmin) / ((5 * np.pi / 180) - a_cdmin) **2)
# K = set manually if need be

Cl_stall_plus = float(np.max(CL_data))
# Cl_stall_plus = set manually if need be
Cl_stall_minus = float(np.min(CL_data))
# Cl_stall_minus = set manually if need be

a_stall_plus = float(alpha_data[np.where(CL_data == Cl_stall_plus)] * np.pi / 180)
# a_stall_plus = set manually if need be
a_stall_minus = float(alpha_data[np.where(CL_data == Cl_stall_minus)] * np.pi / 180)
# a_stall_minus = set manually if need be

Cla = float((CL_data[np.where(alpha_data == 5)] - CL_data[np.where(alpha_data == -2.5)]) / (7.5 * np.pi / 180))
# Cla = set manually if need be

Cd_stall_plus = float(CD_data[np.where(CL_data == Cl_stall_plus)])
#Cd_stall_plus = set manually if need be
Cd_stall_minus = float(CD_data[np.where(CL_data == Cl_stall_minus)])
# Cd_stall_minus = set manually if need be

## Printing airfoil polar variable --> optional
# for key in [
#     Cl0,
#     Cdmin,
#     a_cdmin,
#     K,   
#     Cl_stall_plus,
#     Cl_stall_minus,
#     a_stall_plus,
#     a_stall_minus,
#     Cla,
#     Cd_stall_plus,
#     Cd_stall_minus,
# ]:
#     print(key) 

#------------------------------Setting Rotor Dictionary------------------------------------------------------------------------------#
rotor = RotorParameters(
    num_blades=2,
    Cl0 = Cl0,
    Cla = Cla,
    Cdmin = Cdmin,
    K = K,
    alpha_Cdmin = a_cdmin,
    Cl_stall_plus = Cl_stall_plus,
    Cl_stall_minus = Cl_stall_minus,
    Cd_stall_plus = Cd_stall_plus,
    Cd_stall_minus = Cd_stall_minus,
    a_stall_plus = a_stall_plus,
    a_stall_minus = a_stall_minus,

)
"""
    Mode settings
        1 --> Ideal Loading
        2 --> BEMT
"""
mode = 2

num_blades = rotor['num_blades']
Cl0 = rotor['Cl0']
Cla = rotor['Cla']
Cdmin = rotor['Cdmin']
K = rotor['K']
alpha_Cdmin = rotor['alpha_Cdmin']


num_evaluations = 1         # Rotor blade discretization in time:                            Can stay 1 
num_radial = 100             # Rotor blade discretization in spanwise direction:              Has to be the same as the size of the chord and pitch vector
num_tangential = 1          # Rotor blade discretization in tangential/azimuthal direction:  Can stay 1 

#------------------------------Setting variable shape-----------------------------------------------------------------------------#
shape = (num_evaluations, num_radial, num_tangential)

#------------------------------Creating OpenMDAO problem class--------------------------------------------------------------------# 
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


#------------------------------Importing external rotor geometry --> Optional------------------------------------#
geometry = 'APC_thin_electric_19_12_geometry.txt'
data = np.loadtxt(geometry, delimiter=',', skiprows=1, dtype=str)
with open('APC_thin_electric_19_12_geometry.txt', 'r') as f:
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

#------------------------------Setting RPM--------------------------------------------------------# 
RPM = 2100

#------------------------------Setting rotor parameters-------------------------------------------# 
prob['rotor_radius'] = 1/2   
prob['hub_radius'] = 0.15 * prob['rotor_radius']
prob['slice_thickness'] = (prob['rotor_radius']-prob['hub_radius'])/ (num_radial-1)

# Set pitch and chord distribution for BEMT mode
prob['pitch'] = np.linspace(65,20,num_radial) * np.pi / 180.    # Arbitrary twist distribution
# prob['pitch'] = twist * np.pi / 180.                          # Real twist distribution from UIUC data base

prob['chord'] = 0.1 * np.ones((1,num_radial))                   # Arbitrary chord distribution
# prob['chord'] = normalized_chord * prob['rotor_radius']       # Real chord distribution from UIUC data base
# print(len(prob['chord'])) # = num_radial


# Set reference variables for ideal loading mode
prob['reference_axial_inflow_velocity'] = 20.                   # Adjust axial incoming velocity U_inf
prob['reference_radius'] = 0.5
prob['rotational_speed'] = RPM/60.
prob['reference_rotational_speed'] = RPM/60.
prob['reference_chord'] = 0.1
prob['reference_blade_solidity'] = num_blades * prob['reference_chord'] / 2. / np.pi / prob['reference_radius']
prob['reference_tangential_inflow_velocity'] = prob['rotational_speed'] * 2. * np.pi * prob['reference_radius']



prob['reference_x_dir'][0, :] = [1., 0., 0.]
prob['reference_y_dir'][0, :] = [0., 1., 0.]
prob['reference_z_dir'][0, :] = [0., 0., 1.]
prob['reference_inflow_velocity'][0, 0, 0, :] = [0., 0., 0.]   


for i in range(num_evaluations):
    prob['x_dir'][i, :] = [1., 0., 0.]
    prob['y_dir'][i, :] = [0., 1., 0.]
    prob['z_dir'][i, :] = [0., 0., 1.]
    for j in range(num_radial):
        prob['inflow_velocity'][i, j, 0, :] = [20., 0., 0.]         # Adjust axial incoming velocity U_inf



prob['alpha'] = 6. * np.pi /180.

#----------------------------Define Airfoil Characteristics-------------------------------------------------#
prob['Cl0'] = Cl0

prob['Cdmin'] =  Cdmin
prob['alpha_Cdmin'] =  a_cdmin

prob['alpha_stall'] = a_stall_plus
prob['alpha_stall_minus'] = a_stall_minus

prob['AR'] = 10.

prob['Cl_stall'] = Cl_stall_plus
prob['Cl_stall_minus'] = Cl_stall_minus

prob['Cla'] = Cla
prob['K'] = K

prob['Cd_stall'] = Cd_stall_plus
prob['Cd_stall_minus'] = Cd_stall_minus

prob['smoothing_tolerance'] = 5 * np.pi / 180.



#---------------------------------------Running Model-----------------------------------------------------#
prob.run_model()


#----------------------------Printing output for ideal loading mode----------------------------------------#
if mode == 1:
    for var_name in [
    '_axial_inflow_velocity',
    '_tangential_inflow_velocity',
    '_efficiency',
    '_total_efficiency',
    '_axial_induced_velocity',
    '_tangential_induced_velocity',
    '_local_thrust',
    '_total_thrust',
    '_local_torque',
    '_total_torque',
    '_local_inflow_angle',
    '_local_twist_angle_deg',
    '_local_chord',
    ]:
        print(var_name, prob[var_name])

#----------------------------Printing output for BEMT loading mode----------------------------------------#
elif mode == 2:
    for var_name in [
    # '_axial_inflow_velocity',
    # '_tangential_inflow_velocity',
    # 'BEMT_local_efficiency',
    # 'BEMT_total_efficiency',
    # 'BEMT_axial_induced_velocity',
    # 'BEMT_tangential_induced_velocity',
    'BEMT_local_thrust_2',
    # 'BEMT_total_thrust',
    'BEMT_local_torque_2',
    # 'BEMT_total_torque',
    # '_phi_BEMT',
    # 'BEMT_local_AoA',
    # 'pitch',
    # 'chord', 
    # '_true_Cl',
    # '_Cl_BEMT',
    # '_true_Cd',
    # '_Cd_BEMT',
    # '_radius',
    # '_rotor_radius',
    # '_hub_radius',
    # 'BEMT_loss_factor',
    # 'Cx',
    # 'Ct',
    ]:
        print(var_name, prob[var_name])

fig, axs = plt.subplots(3, 3, figsize=(14, 10))

axs[0,0].plot(prob['_radius'].flatten(), prob['BEMT_local_AoA'].flatten()*180/np.pi,label = r'$\alpha$')
axs[0,0].plot(prob['_radius'].flatten(), prob['pitch'].flatten()*180/np.pi, label = r'$\theta$')
axs[0,0].plot(prob['_radius'].flatten(), prob['_phi_BEMT'].flatten()*180/np.pi,label = r'$\phi$')
axs[0,0].legend()
axs[0,0].grid()

axs[0,1].plot(prob['_radius'].flatten(), prob['BEMT_local_thrust'].flatten(),label = 'dT mom. w. Vit.')
axs[0,1].plot(prob['_radius'].flatten(), prob['BEMT_local_torque'].flatten(),label = 'dQ mom. w. Vit.')
axs[0,1].plot(prob['_radius'].flatten(), prob['BEMT_local_thrust_2'].flatten(),linestyle = '--',label = 'dT BE w. Vit.')
axs[0,1].plot(prob['_radius'].flatten(), prob['BEMT_local_torque_2'].flatten(),linestyle = '--',label = 'dQ BE w. Vit.')
axs[0,1].legend()
axs[0,1].grid()

axs[0,2].plot(prob['_radius'].flatten(), prob['BEMT_local_thrust_1'].flatten(),label = 'dT mom. wo. Vit.')
axs[0,2].plot(prob['_radius'].flatten(), prob['BEMT_local_torque_1'].flatten(),label = 'dQ mom. wo. Vit.')
axs[0,2].plot(prob['_radius'].flatten(), prob['BEMT_local_thrust_22'].flatten(),linestyle = '--',label = 'dT BE wo. Vit.')
axs[0,2].plot(prob['_radius'].flatten(), prob['BEMT_local_torque_22'].flatten(),linestyle = '--',label = 'dQ BE wo. Vit.')
axs[0,2].legend()
axs[0,2].grid()

axs[1,0].plot(prob['_radius'].flatten(), prob['_true_Cl'].flatten(),label = 'Cl w. Viterna')
axs[1,0].plot(prob['_radius'].flatten(), prob['_Cl_BEMT'].flatten(),linestyle = '--',label = 'Cl wo. Viterna')
axs[1,0].axhline(y = Cl_stall_plus,linestyle = ':',label = 'Cl stall')
axs[1,0].legend()
axs[1,0].grid()

axs[2,0].plot(prob['_radius'].flatten(), prob['_true_Cd'].flatten(),label = 'Cd w. Viterna')
axs[2,0].plot(prob['_radius'].flatten(), prob['_Cd_BEMT'].flatten(),linestyle = '--',label = 'Cd wo. Viterna')
axs[2,0].legend()
axs[2,0].grid()

axs[1,1].plot(prob['_radius'].flatten(), prob['Cx'].flatten(),label = 'Cx w. Viterna')
axs[1,1].plot(prob['_radius'].flatten(), prob['Cx2'].flatten(),label = 'Cx wo. Viterna')
axs[1,1].plot(prob['_radius'].flatten(), prob['Ct'].flatten(),label = 'Ct w. Viterna')
axs[1,1].plot(prob['_radius'].flatten(), prob['Ct2'].flatten(),label = 'Ct wo. Viterna')
axs[1,1].legend()
axs[1,1].grid()

plt.tight_layout( rect=[0, 0.03, 1, 0.95])
plt.show()

# performance = 'APC_10_8_performance_3000.txt'
# data = np.loadtxt(performance, delimiter=',', skiprows=1, dtype=str)
# with open('APC_10_8_performance_3000.txt', 'r') as f:
#     data = f.read().split()
#     floats = []
#     for elem in data:
#         try:
#             floats.append(float(elem))
#         except ValueError:
#             pass
# floats = np.array(floats)
# num_rows = int(len(floats)/4)
# num_cols = 4
# apc_performance_data = np.reshape(floats, (num_rows, num_cols))
# APC_J = apc_performance_data[:,0]
# APC_CT = apc_performance_data[:,1]
# APC_CP = apc_performance_data[:,2]
# APC_eta = apc_performance_data[:,3]

# om.n2(prob)