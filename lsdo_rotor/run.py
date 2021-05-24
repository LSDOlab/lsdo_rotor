import numpy as np
import openmdao.api as om
import omtools.api as ot
import matplotlib.pyplot as plt


from lsdo_rotor.core.idealized_bemt_group import IdealizedBEMTGroup
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.get_smoothing_parameters import get_smoothing_parameters
from lsdo_rotor.functions.get_airfoil_parameters import get_airfoil_parameters
from lsdo_rotor.functions.get_rotor_dictionary import get_rotor_dictionary
# from lsdo_rotor.get_parameter_dictionary import get_parameter_dictionary
from lsdo_rotor.functions.get_external_rotor_data import get_external_rotor_data
from lsdo_rotor.functions.get_plot_sweeps_vs_J import get_plot_sweeps_vs_J
from lsdo_rotor.functions.get_plot_sweeps_vs_r import get_plot_sweeps_vs_r
from lsdo_rotor.functions.get_plot_opt_geom import get_plot_opt_geom
from lsdo_utils.comps.bspline_comp import   get_bspline_mtx, BsplineComp

"""
OPTIMIZATION INSTRUCTIONS
    0) Decide for what scenario you want to optimize your rotor (either hover or cruise)
    1) Decide what objective you want to maximize/minimize --> most commonly this will be minimizing torque
    2) Decide on constraints (if you have any) --> if you want to minimize torque you want to constrain thrust, otherwise the optimizer will drive both to zero
    3) Decide on design variables (e.g. this could pitch, chord, RPM... ) and their bounds (e.g. you want your bound your pitch agngle between "lower < theta < upper")

    4) VERY IMPORTANT
       Run the model once (without optimization) and check how close your constraint output is to the value you desire
        --> e.g. if you want to minimize torque in hover and require that the rotor provides __x__ N of thrust to support the aircraft,
            make sure that when you run the model once, the thrust is not too far away off your constraint
            Recommendation: thrust should be within +/- 200 N of desired constraint 

    5) If your constraint variable is too far off, tweak your RPM value, (radius and/or V_inf) so that you get closer to the desired constraint
    6) Run optimization by commenting out "prob.run_model()" in line 242 and uncomment "prob.run_driver()" in line 243

General recommendations:
    - Do not make RPM an design variable because this may lead to unphysical designs where total efficiency exceeds 1 --> If you do make it a design variable constrain total efficiency
    - If your optimization fails to converge:
        * Check how close your constraint is to the desired value; if it is close to the tolerance set in line 167 increase the number of max iterations in line 168 
        * Comment out lines 198-202 & 204-208 and uncomment lines 197 & 203 --> Re-run optimization; 
            --> We are saving the pitch and chord output (lines 305-306) from the unconverged optimization and make them the starting point for the rerun
        * Be sure that the optimization problem you are trying to solve makes sense physically (optimization will be unsuccessful if you are trying to solve an unphysical problem)
        * It is easier for the optimizer if pitch is the only design variable --> Try optimizing pitch first, save the output (line 305) and make that your initial pitch distribution in line 197, 
          then re-run the optimization with both pitch and chord as design vars. (more relevant in hover)  
        * Increase the optimizer tolerance (line 167) 
        * If none of the above work try:
            o adjusting lines 64-68 --> this changes this initial chord and pitch distribution
            o adjusting lines 88 and 89 to match num_radial 
    - Unconstrained optimization problems are easier for the optimizer:
        --> e.g if you are just interested in maximizing thrust without constraints on torque, you will achieve convergence faster
        --> HOWEVER: make sure that the output is physical and that you set appropriate bounds on the design variables
"""


#------------------------------Set airfoil name, rotor parameters and flight condition-----------------------------------------------------------------------------#
airfoil = 'NACA_4412' # Available airfoils: 'NACA_4412', 'Clark_Y', 'mh117' make sure to adjust airfoil.txt file if you change airfoil

rotor_diameter = 1.5# 0.254
RPM = 2500
V_inf = 67
num_blades = 4
altitude = 1000

#------------------------------Define pitch/chord and blade root/tip for INITIAL geometry-----------------------------------------------------------------------------#
root_pitch = 65 #degrees
tip_pitch  = 20 #degrees

root_chord = 0.1
tip_chord  = 0.1

#------------------------------Setting Rotor Dictionary------------------------------------------------------------------------------#
rotor = get_rotor_dictionary(airfoil,num_blades, altitude)

"""
    Mode settings
        1 --> Ideal Loading
        2 --> BEMT
"""
mode = 2


num_blades = rotor['num_blades']


num_evaluations = 1         # Rotor blade discretization in time:                            Can stay 1
num_radial = 25             # Rotor blade discretization in spanwise direction:              Has to be the same as the size of the chord and pitch vector
num_tangential = 1          # Rotor blade discretization in tangential/azimuthal direction:  Can stay 1

chord_num_cp = 5            # Number of control points for B-spline chord parameterization
pitch_num_cp = 8            # Number of control points for B-spline pitch parameterization
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

group.create_indep_var('pitch_cp', shape = (pitch_num_cp,))
group.create_indep_var('chord_cp', shape = (chord_num_cp,))

prob.model.add_subsystem('external_inputs_group', group, promotes=['*'])

#------------------------------Adding main working group --> where majority of the model is implemented---------------------------------------------------------#

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



# Defining driver and design variables 
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-4
prob.driver.options['maxiter'] = 150
#
# prob.driver = driver = om.pyOptSparseDriver()
# driver.options['optimizer'] = 'SNOPT'
# driver.opt_settings['Major feasibility tolerance'] = 2.e-6
# driver.opt_settings['Major optimality tolerance'] = 2.e-6
# driver.opt_settings['Verify level'] = 3

prob.model.add_design_var('chord_cp',lower = 0.01, upper = 0.30)
prob.model.add_design_var('pitch_cp',lower = 10*np.pi/180, upper = 80*np.pi/180)
prob.model.add_constraint('BEMT_total_thrust', equals = 2000)
prob.model.add_objective('BEMT_total_torque')

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
# prob['chord_cp'] = np.genfromtxt('optimized_chord_cp.txt')
prob['chord_cp'] = np.linspace(
    root_chord,
    tip_chord,
    chord_num_cp,
)
# prob['pitch_cp'] = np.genfromtxt('optimized_pitch_cp.txt')
prob['pitch_cp'] = np.linspace(
    root_pitch * np.pi / 180.,
    tip_pitch * np.pi / 180.,
    pitch_num_cp,
)
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
# prob.run_driver()
# om.n2(prob)

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
    # 'Re',
    # '_Re_test',

    # '_phi_BEMT',
    # 'BEMT_local_AoA',
    # '_pitch',
    # 'pitch_cp',

    # '_chord',
    # 'chord_cp',

    # '_Cl',
    # '_Cd',

    # '_radius',
    # '_rotor_radius',
    # '_hub_radius',

    # 'BEMT_loss_factor',
    ]:
        print(var_name, prob[var_name])


np.savetxt('optimized_pitch_cp.txt', prob['pitch_cp'],delimiter='  ')
np.savetxt('optimized_chord_cp.txt', prob['chord_cp'],delimiter='  ')


#------------------------------Plotting--------------------------------------------------#
airfoil_names = [
    # 'NACA 4412',
    'Clark-Y',
]
rotor_files = [
    'APC_10_6_geometry.txt',
    'APC_10_6_performance_5000.txt',
]


# get_plot_opt_geom(np.linspace(root_chord, tip_chord,num_radial),prob['chord'],np.linspace(root_pitch, tip_pitch, num_radial), prob['pitch'] * 180/np.pi,prob['_radius'])

# get_plot_sweeps_vs_J(airfoil,rotor_files, RPM, rotor_diameter, 100, 21, num_blades, altitude, num_radial, 'optimized_pitch_cp.txt', 'optimized_chord_cp.txt', pitch_num_cp, chord_num_cp)

# get_plot_sweeps_vs_r(airfoil, rotor_files, RPM, rotor_diameter, V_inf, num_blades, altitude, num_radial, 'optimized_pitch_cp.txt', 'optimized_chord_cp.txt', pitch_num_cp, chord_num_cp)

# om.n2(prob)
