import numpy as np
import openmdao.api as om
import omtools.api as ot
import matplotlib.pyplot as plt

from lsdo_rotor.core.idealized_bemt_group import IdealizedBEMTGroup
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.get_smoothing_parameters import get_smoothing_parameters
from lsdo_rotor.get_airfoil_parameters import get_airfoil_parameters
from lsdo_rotor.get_rotor_dictionary import get_rotor_dictionary
from lsdo_rotor.get_external_rotor_data import get_external_rotor_data
from lsdo_rotor.get_plot_sweeps_vs_J import get_plot_sweeps_vs_J
from lsdo_rotor.get_plot_sweeps_vs_r import get_plot_sweeps_vs_r

#------------------------------Set airfoil file name-----------------------------------------------------------------------------#
airfoil_filename = 'xf-e63-il-50000.txt'   
rotor_diameter = 0.254 
RPM = 3000
V_inf = 5                                                       

#------------------------------Extracting Data From Airfoil Polar----------------------------------------------------------------#
airfoil_parameters = get_airfoil_parameters(airfoil_filename)

#------------------------------Getting Smoothing Parameters----------------------------------------------------------------#
smoothing_parameters = get_smoothing_parameters(airfoil_parameters[0] ,airfoil_parameters[1] , airfoil_parameters[2], airfoil_parameters[3], airfoil_parameters[4], airfoil_parameters[5], 10, airfoil_parameters[6], airfoil_parameters[7], airfoil_parameters[8], airfoil_parameters[9], airfoil_parameters[10])

#------------------------------Setting Rotor Dictionary------------------------------------------------------------------------------#
rotor = get_rotor_dictionary(airfoil_filename)

"""
    Mode settings
        1 --> Ideal Loading
        2 --> BEMT
"""
mode = 2


num_blades = rotor['num_blades']

num_evaluations = 1         # Rotor blade discretization in time:                            Can stay 1 
num_radial = 18             # Rotor blade discretization in spanwise direction:              Has to be the same as the size of the chord and pitch vector
num_tangential = 1          # Rotor blade discretization in tangential/azimuthal direction:  Can stay 1 

#------------------------------Setting variable shape-----------------------------------------------------------------------------#
shape = (num_evaluations, num_radial, num_tangential)

#------------------------------Creating OpenMDAO problem class--------------------------------------------------------------------# 
prob = om.Problem()

#------------------------------Creating omtools group for external inputs---------------------------------------------------------# 
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


#------------------------------Importing external rotor geometry --> Optional------------------------------------#
rotor_geometry      = 'APC_10_6_geometry.txt'
rotor_performance   = 'APC_10_6_performance_5000.txt'

external_rotor_data = get_external_rotor_data(rotor_geometry, rotor_performance)



#------------------------------Setting rotor parameters----------------------------------------------------------# 
prob['rotor_radius'] = rotor_diameter/2   
prob['hub_radius'] = 0.15 * prob['rotor_radius']
prob['slice_thickness'] = (prob['rotor_radius']-prob['hub_radius'])/ (num_radial-1)


#-----------------------------Set chord and pitch distribution for BEMT mode-------------------------------------#
# prob['chord'] = 0.1 * np.ones((1,num_radial))                   # Arbitrary chord distribution
prob['chord'] = external_rotor_data[0] * prob['rotor_radius']     # Real chord distribution from UIUC data base
# print(len(prob['chord'])) # = num_radial


# prob['pitch'] = np.linspace(65,20,num_radial) * np.pi / 180.    # Arbitrary twist distribution
prob['pitch'] = external_rotor_data[1] * np.pi / 180.             # Real twist distribution from UIUC data base




# Set reference variables for ideal loading mode
prob['reference_axial_inflow_velocity'] = V_inf                      # Adjust axial incoming velocity V_inf
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
        prob['inflow_velocity'][i, j, 0, :] = [V_inf, 0., 0.]         # Adjust axial incoming velocity V_inf



prob['alpha'] = 6. * np.pi /180.

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
    'BEMT_total_efficiency',
    
    # 'BEMT_axial_induced_velocity',
    # 'BEMT_tangential_induced_velocity',
    
    # 'BEMT_local_thrust',
    # 'BEMT_local_torque',
    
    'BEMT_total_thrust',
    'BEMT_total_torque',
    
    # '_phi_BEMT',
    'BEMT_local_AoA',
    # 'pitch',

    # 'chord', 
    
    # '_Cl',
    # '_Cd',
    
    # '_radius',
    # '_rotor_radius',
    # '_hub_radius',
    
    # 'BEMT_loss_factor',
    ]:
        print(var_name, prob[var_name])


#------------------------------Plotting--------------------------------------------------#
airfoil_files = [
    'xf-NACA4412-il-50000.txt',
    'xf-clarky-il-50000.txt',
    'xf-e63-il-50000.txt',
]
airfoil_names = [
    'NACA 4412',
    'Clark-Y',
    'Eppler E63',
]
rotor_files = [
    'APC_10_6_geometry.txt',
    'APC_10_6_performance_5000.txt',
]
geometry_array = np.array([
    # 0.1 * np.ones((50,)), 
    # np.linspace(65,20,50) * np.pi / 180.,
])

get_plot_sweeps_vs_J(airfoil_files, airfoil_names, rotor_files, geometry_array, RPM, rotor_diameter, 20, 21, '50 000')

# get_plot_sweeps_vs_r(airfoil_files, airfoil_names, rotor_files, geometry_array, RPM, rotor_diameter, V_inf, '50 000')

# om.n2(prob)