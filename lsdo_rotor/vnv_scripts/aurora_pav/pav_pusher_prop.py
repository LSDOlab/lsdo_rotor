import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor import RotorAnalysis, BEM, BEMParameters, AcStates, get_atmosphere, print_output
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


rotor_analysis = RotorAnalysis()

u = rotor_analysis.create_input('u', val=50.06848, shape=(1, ))
v = rotor_analysis.create_input('v', val=0, shape=(1, ))
w = rotor_analysis.create_input('w', val=0, shape=(1, ))
altitude = rotor_analysis.create_input('altitude', val=0, shape=(1, )) # in meter

ac_states = AcStates(u=u, v=v, w=w)
atmos = get_atmosphere(altitude=altitude)

num_nodes = 1
num_radial = 25
num_tangential = num_azimuthal = 1
num_blades = 5

ft2m = 0.3048
m2in = 39.3701

# twist_cp_guess = np.deg2rad(np.linspace(55, 11, 6))  # rad
twist_cp_guess = np.array([0.55207943, 0.35981639, 0.16753661, 0.12377559, 0.17724111, 0.07146789]) 
chord_cp_guess = np.array([0.07295861, 0.10717677, 0.09075833, 0.06437597, 0.03848824, 0.02721645])  # m

chord_cp = rotor_analysis.create_input('chord_cp', val=chord_cp_guess, dv_flag=True, lower=0.01, upper=0.4)
twist_cp = rotor_analysis.create_input('twist_cp', val=twist_cp_guess, dv_flag=True, lower=np.deg2rad(0), upper=np.deg2rad(85))
thrust_vector = rotor_analysis.create_input('thrust_vector', val=np.array([1., 0, 0]).reshape(num_nodes, 3))
thrust_origin = rotor_analysis.create_input('thrust_origin', val=np.array([19.700, 0., 2.625]).reshape(num_nodes, 3)) # in m
propeller_radius = rotor_analysis.create_input('propeller_radius', val=4/2*ft2m)
rpm = rotor_analysis.create_input('rpm', val=4000)

bem_parameters = BEMParameters(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=True,
    normalized_hub_radius=0.2,
    num_cp=6, 
)

bem_model = BEM(
    name='bem_analysis',
    BEM_parameters=bem_parameters,
    num_nodes=1,
)
bem_outputs = bem_model.evaluate(ac_states=ac_states, rpm=rpm, rotor_radius=propeller_radius, thrust_vector=thrust_vector, thrust_origin=thrust_origin,
                                 atmosphere=atmos, blade_chord_cp=chord_cp, blade_twist_cp=twist_cp)

rotor_analysis.register_output(bem_outputs)

efficiency = 0.8
LoverD = 8
m = 800  # kg
W = m*9.81  # N
D = W/LoverD
expected_thrust = 1.*D
max_torque = 160. # Nm

csdl_model = rotor_analysis.assemble_csdl()
csdl_model.add_constraint('bem_analysis.eta', equals=efficiency)
csdl_model.add_constraint('bem_analysis.Q', upper=max_torque, scaler=1e-2)
bem_thrust = csdl_model.declare_variable('computed_thrust')
csdl_model.connect('bem_analysis.T', 'computed_thrust')
thrust_needed = csdl_model.create_input('expected_thrust', val=expected_thrust)
csdl_model.register_output('thrust_res', ((bem_thrust + -1* thrust_needed)**2))
csdl_model.add_objective('thrust_res', scaler=1e-2)

sim =  Simulator(csdl_model, analytics=True)
sim.run()

prob = CSDLProblem(problem_name='pav_lift_rotor_opt', simulator=sim)
optimizer = SLSQP(
    prob, 
    maxiter=250, 
    ftol=1e-4,
)
optimizer.solve()
optimizer.print_results()

print_output(sim, rotor_analysis)
print(sim['thrust_res'])
print(sim['twist_cp'])
print(sim['chord_cp'])
print(sim['bem_analysis.u'])
print(sim['bem_analysis.v'])
print(sim['bem_analysis.w'])
