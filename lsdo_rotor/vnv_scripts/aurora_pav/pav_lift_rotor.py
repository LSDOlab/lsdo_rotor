import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor import RotorAnalysis, BEM, BEMParameters, AcStates, get_atmosphere, print_output
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


rotor_analysis = RotorAnalysis()

u = rotor_analysis.create_input('u', val=0, shape=(1, ))
v = rotor_analysis.create_input('v', val=0, shape=(1, ))
w = rotor_analysis.create_input('w', val=0, shape=(1, ))
altitude = rotor_analysis.create_input('altitude', val=0, shape=(1, )) # in meter

ac_states = AcStates(u=u, v=v, w=w)
atmos = get_atmosphere(altitude=altitude)

num_nodes = 1
num_radial = 25
num_tangential = num_azimuthal = 1
num_blades = 2

ft2m = 0.3048
m2in = 39.3701

twist_cp_guess = np.array([0.55207943, 0.35981639, 0.16753661, 0.12377559, 0.17724111, 0.07146789])  # rad
chord_cp_guess = np.array([0.07295861, 0.10717677, 0.09075833, 0.06437597, 0.03848824, 0.02721645])  # m

chord_cp = rotor_analysis.create_input('chord_cp', val=chord_cp_guess, dv_flag=True, lower=0.01, upper=0.4)
twist_cp = rotor_analysis.create_input('twist_cp', val=twist_cp_guess, dv_flag=True, lower=np.deg2rad(0), upper=np.deg2rad(85))
thrust_vector = rotor_analysis.create_input('thrust_vector', val=np.array([0, 0, -1]).reshape(num_nodes, 3))
thrust_origin = rotor_analysis.create_input('thrust_origin', val=np.array([-1.146, 1.619, -0.162]).reshape(num_nodes, 3)) # in m
propeller_radius = rotor_analysis.create_input('propeller_radius', val=6/2*ft2m)
rpm = rotor_analysis.create_input('rpm', val=2800, dv_flag=False, lower=800, upper=2000, scaler=1e-3)

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


csdl_model = rotor_analysis.assemble_csdl()
csdl_model.add_constraint('bem_analysis.T', equals=2000, scaler=1e-3)
csdl_model.add_constraint('bem_analysis.Q', upper=160, scaler=1e-2)
FOM = csdl_model.declare_variable('bem_analysis.FOM', shape=(num_nodes, ))
csdl_model.register_output('FOM_obj', FOM * -1)
csdl_model.add_objective('FOM_obj')

sim =  Simulator(csdl_model, analytics=True)
prob = CSDLProblem(problem_name='pav_lift_rotor_opt', simulator=sim)
optimizer = SLSQP(
    prob, 
    maxiter=150, 
    ftol=1e-4,
)
optimizer.solve()
optimizer.print_results()

print_output(sim, rotor_analysis)
