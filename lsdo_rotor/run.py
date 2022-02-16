import numpy as np 
from csdl import Model 
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")


from inputs.external_inputs_group import ExternalInputsGroup
from core.BEM_group import BEMGroup

from airfoil.get_surrogate_model import get_surrogate_model
from functions.get_rotor_dictionary import get_rotor_dictionary
from functions.get_max_LD_parameters_reference_chord import get_max_LD_parameters_reference_chord


"""
    Mode settings
        1 --> Ideal-Loading Design Method
        2 --> BEM
"""
mode = 2

# The following airfoils are currently available: 'NACA_4412', 'Clark_Y', 'NACA_0012', 'mh117'; We recommend 'NACA_4412' or 'Clark_Y'
airfoil             = 'Clark_Y' 
interp              = get_surrogate_model(airfoil)

rotor_diameter      = 2        # (in m)
RPM                 = 1500

# x, y, z velocity components (m/s); Vx is the axial velocity component 
Vx                  = 60   # Axial inflow velocity (i.e. V_inf) 
Vy                  = 0   # Side slip velocity in the propeller plane
Vz                  = 0   # Side slip velocity in the propeller plane

# Specify number of blades and altitude
num_blades          = 3
altitude            = 100        # (in m)

# The following 3 parameters are used for mode 1 only! The user has to
# specify three parameters for optimal blade design 
reference_radius    = rotor_diameter / 4  # Specify the reference radius; We recommend radius / 2
reference_chord     = 0.15                # Specify the reference chord length at the reference radius (in m)

# The following parameters are used for mode 2 only
# Change these parameters if you want to chord and twist profile to vary
# linearly from rotor hub to tip
root_chord          = 0.2       # Chord length at the root/hub
root_twist          = 80        # Twist angle at the blade root/hub (deg)
tip_chord           = 0.1       # Chord length at the tip
tip_twist           = 20        # Twist angle at the blade tip (deg)

# Consider the following two lines if you want to use an exiting rotor geometry:
# IMPORTANT: you can only use mode if you want to use an exiting rotor geometry.
use_external_rotor_geometry = 'n'           # [y/n] If you want to use an existing rotor geometry 
geom_data = np.loadtxt('APC_9_6_geom.txt')  # if 'y', make sure you have the right .txt file with the chord distribution in the
                                            # second column and the twist distribution in the third column

# The following parameters specify the radial and tangential mesh as well as the
# number of time steps; 
num_evaluations     = 1         # Discretization in time:                 Only required if your operating conditions change in time
if use_external_rotor_geometry == 'y':
    num_radial      = len(geom_data[:,1])     
else:
    num_radial      = 30        # Discretization in spanwise direction:   Should always be at least 25
num_tangential      = 1         # Discretization in tangential direction: Only required if Vy,Vz are non-zero; recommend at least 20


# Specify some post-processing options 
plot_rotor_blade_shape  = 'y'     # Only implemented for mode 1 [y/n]
plot_rotor_performance  = 'n'     # Not yet implemented [y/n]
print_rotor_performance = 'y'     # [y/n]


#---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, Vx, RPM, altitude)
rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord)

shape = (num_evaluations, num_radial, num_tangential)


rotor_model = Model()

group = BEMGroup(
    mode = mode,
    rotor=rotor,
    num_evaluations=num_evaluations,
    num_radial=num_radial,
    num_tangential=num_tangential,
)
rotor_model.add(group,'BEM_group')#, promotes = ['*'])



sim = Simulator(rotor_model)

sim['rotor_radius'] = rotor_diameter / 2
sim['hub_radius'] = 0.2 * sim['rotor_radius']
sim['slice_thickness'] = (sim['rotor_radius']-sim['hub_radius'])/ (num_radial -1)



if use_external_rotor_geometry == 'y':
    sim['chord'] = geom_data[:,1] * rotor_diameter/2
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


sim['reference_axial_inflow_velocity'] = 50                      # Adjust axial incoming velocity V_inf
sim['reference_radius'] = reference_radius
sim['rotational_speed'] = RPM/60.
sim['reference_rotational_speed'] = RPM/60.
sim['reference_chord'] = reference_chord
sim['reference_blade_solidity'] = num_blades * sim['reference_chord'] / 2. / np.pi / sim['reference_radius']
sim['reference_tangential_inflow_velocity'] = sim['rotational_speed'] * 2. * np.pi * sim['reference_radius']


for i in range(num_evaluations):
        sim['x_dir'][i, :] = [1., 0., 0.]
        sim['y_dir'][i, :] = [0., 1., 0.]
        sim['z_dir'][i, :] = [0., 0., 1.]
        for j in range(num_radial):    
            for k in range(num_tangential):    
                sim['inflow_velocity'][i, j, k, :] = [Vx, Vy, Vz]         


sim.run()



if mode == 1:
    if print_rotor_performance == 'y':
        from functions.print_ideal_loading_output import print_ideal_loading_output
        print_ideal_loading_output(sim['total_thrust'].flatten(),sim['total_torque'].flatten(), \
            Vx, RPM/60,sim['_radius'].flatten(),sim['_local_thrust'].flatten(),sim['_local_torque'].flatten(), \
            sim['_back_comp_axial_induced_velocity'].flatten(), sim['_back_comp_tangential_induced_velocity'].flatten(), \
            sim['_mod_local_chord'].flatten(), sim['_local_twist_angle'].flatten(), num_blades, sim['slice_thickness'].flatten(), rotor_diameter/2 )      
    if plot_rotor_blade_shape == 'y':    
        from functions.plot_ideal_loading_blade_shape import plot_ideal_loading_blade_shape
        plot_ideal_loading_blade_shape(sim['_radius'].flatten(),sim['_local_chord'].flatten(),sim['_mod_local_chord'].flatten(),sim['_local_twist_angle'].flatten())
else:
    if print_rotor_performance == 'y':
        from functions.print_bem_analysis_output import print_bem_analysis_output
        print_bem_analysis_output(sim['total_thrust'].flatten(),sim['total_torque'].flatten(), \
            Vx, RPM/60,sim['_radius'].flatten(),sim['_local_thrust'].flatten(),sim['_local_torque'].flatten(), \
            sim['_ux'].flatten(), sim['_ut'].flatten(), \
            sim['chord_distribution'].flatten(), sim['pitch_distribution'].flatten(), num_blades, sim['slice_thickness'].flatten(), rotor_diameter/2 )
        
        




