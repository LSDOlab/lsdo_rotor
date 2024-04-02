# Imports
import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor import RotorAnalysis, BEM, BEMParameters, AcStates, get_atmosphere, print_output
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from lsdo_acoustics import Acoustics, Lowson, SKM, GL, TotalAircraftNoise


rotor_analysis = RotorAnalysis()

# create a subset of the aircraft states ('u' only is sufficient) and altitude 
u = rotor_analysis.create_input('u', val=0., shape=(1, ))
altitude = rotor_analysis.create_input('altitude', val=100, shape=(1, ))

# instantiate AcStates and use helper function to compute atmospheric properties
ac_states = AcStates(u=u)
atmos = get_atmosphere(altitude=altitude)

# set up radial and azimuthal discretization
num_radial = 30
num_tangential = 30

num_blades = 5

# create variables for chord control/twist (control points), thrust vector/origin, radius and rpm
chord_profile = rotor_analysis.create_input('chord_profile', val=np.linspace(0.2, 0.2, num_radial))
twist_profile = rotor_analysis.create_input('twist_profile', val=np.deg2rad(np.linspace(65, 20, num_radial)))
thrust_vector = rotor_analysis.create_input('thrust_vector', val=np.array([0, 0, -1])) # NOTE: orientation with respect to flight dynamics frame
thrust_origin = rotor_analysis.create_input('thrust_origin', val=np.array([0, 0, altitude.value]))
rotor_radius = rotor_analysis.create_input('rotor_radius', val=0.61)
rpm = rotor_analysis.create_input('rpm', val=4000)

# set up BEM parameters;
##### Option 1 (recommended) 'NACA 4412' airfoil has well-trained machine learning model (indicated by 'use_custom_ml=True') ####
# Pros: 
#   - trained on XFOIL data 
#   - reasonable accuracy
#   - models Cl/Cd as a function of AoA, Re, Mach
# Cons:
#   - Slower
bem_parameters = BEMParameters(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=True,         # if you want to use ML airfoil model
)

##### Option 2: define custom (quadratic) airfoil polar (only as a function of AoA) ##### 
# Pros: 
#   - Fast
#   - Can define for other airfoils
# Cons:
#   - inaccurate
#   - models Cl/Cd as a function of AoA only (gross simplification)
# airfoil_polar = {
#     'Cl_0': 0.25,
#     'Cl_alpha': 5.1566,
#     'Cd_0': 0.01,
#     'Cl_stall': [-1, 1.5], 
#     'Cd_stall': [0.02, 0.06],
#     'alpha_Cl_stall': [-10, 15],
# }
# bem_parameters = BEMParameters(
#     num_radial=num_radial,
#     num_tangential=num_tangential,
#     num_blades=num_blades,
#     airfoil='NACA_4412',
#     airfoil_polar=airfoil_polar,        # set custom airfoil polar
#     num_cp=num_bspline_cp,
# )

# Instantiate BEM solver
bem_model = BEM(
    name='bem_analysis',
    BEM_parameters=bem_parameters,
    num_nodes=1,
)


# evaluate BEM solver and register its outputs
bem_outputs = bem_model.evaluate(ac_states=ac_states, rpm=rpm, rotor_radius=rotor_radius, thrust_vector=thrust_vector, thrust_origin=thrust_origin,
                                 atmosphere=atmos, blade_chord=chord_profile, blade_twist=twist_profile)
rotor_analysis.register_output(bem_outputs)


hover_acoustics = Acoustics(
    aircraft_position = np.array([0.,0., altitude.value])
)
observer_mode = 'directivity'

# # ==== SINGLE OBSERVER ====
if observer_mode == 'single':
    hover_acoustics.add_observer(
        name='obs1',
        obs_position=np.array([0., 0., 0.]),
        time_vector=np.array([0.]),
    )
elif observer_mode == 'directivity':
    obs_radius = 100.
    num_observers = 37 * 2
    theta = np.linspace(0, 2*np.pi, num_observers)
    z = obs_radius * np.cos(theta)
    x = obs_radius * np.sin(theta)

    obs_position_array = np.zeros((num_observers, 3))
    obs_position_array[:,0] = x
    obs_position_array[:,2] = z

    for i in range(num_observers):
        hover_acoustics.add_observer(
            name=f'obs_{i}',
            obs_position=obs_position_array[i,:],
            time_vector=np.array([0.])
        )

Lowson_model = Lowson(
    name='Lowson_model',
    num_nodes=1,
    rotor_parameters=bem_parameters,
    acoustics_data=hover_acoustics,
)
hover_tonal_SPL, hover_tonal_SPL_A_weighted = Lowson_model.evaluate_tonal_noise(bem_outputs.dT, bem_outputs.dD, ac_states,
                                                                                   rpm=rpm, rotor_origin=thrust_origin,
                                                                                   thrust_vector=thrust_vector, 
                                                                                   rotor_radius=rotor_radius, altitude=altitude,
                                                                                   chord_length=chord_profile, phi_profile=bem_outputs.phi)
rotor_analysis.register_output(hover_tonal_SPL)
rotor_analysis.register_output(hover_tonal_SPL_A_weighted)


SKM_model = SKM(
    name='SKM_model',
    num_nodes=1, 
    rotor_parameters=bem_parameters,
    acoustics_data=hover_acoustics,
)
hover_broadband_SPL, hover_broadband_SPL_A_weighted = SKM_model.evaluate_broadband_noise(ac_states, bem_outputs.C_T, rpm=rpm,
                                                                                           disk_origin=thrust_origin,
                                                                                           thrust_vector=thrust_vector,
                                                                                           radius=rotor_radius, chord_length=chord_profile)
rotor_analysis.register_output(hover_broadband_SPL)
rotor_analysis.register_output(hover_broadband_SPL_A_weighted)

total_noise_model = TotalAircraftNoise(
    name='total_noise',
    acoustics_data=hover_acoustics,
)
noise_components = [hover_tonal_SPL, hover_broadband_SPL]
noise_components_A = [hover_tonal_SPL_A_weighted, hover_broadband_SPL_A_weighted]

hover_total_SPL, hover_total_SPL_A_weighted = total_noise_model.evaluate(noise_components, A_weighted_noise_components=noise_components_A)
rotor_analysis.register_output(hover_total_SPL)
rotor_analysis.register_output(hover_total_SPL_A_weighted)
# endregion

csdl_model = rotor_analysis.assemble_csdl()
sim = Simulator(csdl_model, analytics=True)
sim.run()

print_output(sim, rotor_analysis, comprehensive_print=True)

# print('\n')
# print('Lowson tonal SPL')
# print(sim['Lowson_model_Lowson_tonal_model.Lowson_model_Lowson_tonal_model_tonal_spl'])
# print('\n')
# print('Lowson A-weighted tonal SPL')
# print(sim['Lowson_model_Lowson_tonal_model.Lowson_model_Lowson_tonal_model_tonal_spl_A_weighted'])
# print('\n')
# print('SKM broadband_spl')
# print(sim['SKM_model_SKM_broadband_model.SKM_model_SKM_broadband_model_broadband_spl'])
# print('\n')
# print('SKM broadband_spl_A_weighted')
# print(sim['SKM_model_SKM_broadband_model.SKM_model_SKM_broadband_model_broadband_spl_A_weighted'])
# print('\n')
# print('total_spl', sim['total_noise.total_spl'])
# print('A_weighted_total_spl', sim['total_noise.A_weighted_total_spl'])


if observer_mode == 'directivity':

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    theta_plot = np.linspace(np.pi/2, -3*np.pi/2, num_observers)
    ax.plot(theta_plot, sim['Lowson_model_Lowson_tonal_model.Lowson_model_Lowson_tonal_model_tonal_spl'].reshape((num_observers,)), label='Overall tonal')
    ax.set_rlabel_position(-120)

    ax.grid(True)
    plt.legend(loc='best')
    plt.title('SPL distribution in hover and edgewise flight')
    plt.show()
