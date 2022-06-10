import numpy as np 
import openmdao.api as om
from csdl import Model 
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")

from inputs.external_inputs_model import ExternalInputsModel
from core.rotor_model import RotorModel

from airfoil.get_surrogate_model import get_surrogate_model
from functions.get_rotor_dictionary import get_rotor_dictionary
from functions.get_max_LD_parameters_reference_chord import get_max_LD_parameters_reference_chord


def get_simulator(analysis_dict,rotor_dict,operating_dict):
    mode = analysis_dict['rotor_model']

    airfoil = analysis_dict['airfoil']
    interp = get_surrogate_model(airfoil)

    ne = analysis_dict['num_evaluations']
    nr = analysis_dict['num_radial']
    print(nr,'NR')
    nt = analysis_dict['num_azimuthal']

    diameter_vec = rotor_dict['rotor_diameter']                 # Rotor diameter in (m)
    
    RPM_vec = operating_dict['RPM']
    Omega_vec = RPM_vec * 2 * np.pi / 60                                                    
    V_inf_vec = operating_dict['V_inf']              # Cruise speed in (m/s)

    i_vec = operating_dict['rotor_disk_tilt_angle']#np.array([45,90])   
    print(i_vec,'tilt_angle')           # Rotor disk tilt angle in (deg); 
    #   --> 90 degrees means purely axial inflow
    #   --> 0  degrees means purely edgewise flight    

    Vx_vec = operating_dict['Vx']#V_inf_vec * np.sin(i_vec * np.pi/180)
    print(Vx_vec,'Vx')
    Vy_vec = operating_dict['Vy']#V_inf_vec * np.cos(i_vec * np.pi/180)
    print(Vy_vec,'Vy')
    Vz_vec = np.zeros((ne,))#np.array([0,0])



    # Specify number of blades and altitude
    num_blades          = rotor_dict['num_blades']
    altitude            = operating_dict['altitude']        # (in m)

    # The following 3 parameters are used for mode 1 only! The user has to
    # specify three parameters for optimal blade design 
    reference_radius_vec = 0.61 * np.ones((ne,))#np.array([0.61,0.61])  # Specify the reference radius; We recommend radius / 2
    reference_chord_vec  = 0.1 * np.ones((ne,))# np.array([0.1,0.1])                # Specify the reference chord length at the reference radius (in m)

    chord = rotor_dict['blade_chord_distribution']
    twist = rotor_dict['blade_twist_distribution']

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
        num_radial      = nr      # Discretization in spanwise direction:   Should always be at least 25
    num_tangential      = nt       # Discretization in tangential direction: Only required if Vy,Vz are non-zero; recommend at least 20


    # Specify some post-processing options 
    plot_rotor_blade_shape  = 'y'     # Only implemented for mode 1 [y/n]
    plot_rotor_performance  = 'n'     # Not yet implemented [y/n]
    print_rotor_performance = 'y'     # [y/n]


    #---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
    # ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, Vx, RPM, altitude)
    rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, Vx_vec,RPM_vec,Vy_vec,diameter_vec,num_evaluations,num_radial,num_tangential) #, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord, beta)

    shape = (num_evaluations, num_radial, num_tangential)
    print(shape,'SHAPE')
    print(rotor['density'])

    rotor_model = Model()

    group = RotorModel(
        mode=mode,
        rotor=rotor,
        num_evaluations=num_evaluations,
        num_radial=num_radial,
        num_tangential=num_tangential,
    )
    rotor_model.add(group,'rotor_model')#, promotes = ['*'])



    sim = Simulator(rotor_model)

    sim['chord'] = chord
    sim['pitch'] = twist * np.pi / 180

    # if use_external_rotor_geometry == 'y':
    #     sim['chord'] = geom_data[:,1] 
    #     sim['pitch'] = geom_data[:,2] * np.pi/180
    # else:
    #     sim['chord'] = np.linspace(
    #         root_chord,
    #         tip_chord,
    #         num_radial,
    #     )

    #     sim['pitch'] = np.linspace(
    #         root_twist * np.pi / 180.,
    #         tip_twist * np.pi / 180.,
    #         num_radial,
    #     )

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
        sim['hub_radius'][i]   = 0.15 * diameter_vec[i] / 2
        sim['dr'][i] = ((diameter_vec[i] / 2)-(0.2 * diameter_vec[i] / 2))/ (num_radial -1)

        # ILDM parameters     
        sim['reference_chord'][i] = reference_chord_vec[i]
        sim['reference_radius'][i] = reference_radius_vec[i]
        sim['reference_blade_solidity'][i] = num_blades * reference_chord_vec[i] / 2. / np.pi / reference_radius_vec[i]
        sim['ildm_tangential_inflow_velocity'][i] = RPM_vec[i]/60. * 2. * np.pi * reference_radius_vec[i]
        sim['ildm_axial_inflow_velocity'][i] = Vx_vec[i]  
        sim['ildm_rotational_speed'][i] = RPM_vec[i]/60.



    return sim
