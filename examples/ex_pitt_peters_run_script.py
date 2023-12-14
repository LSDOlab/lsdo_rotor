"""
Example 2: Rotor chord and twist profile optimization using Pitt--Peters
inflow solver
"""
import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor import RotorAnalysis, PittPeters, PittPetersParameters, AcStates, get_atmosphere, print_output
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


rotor_analysis = RotorAnalysis()

u = rotor_analysis.create_input('u', val=10, shape=(1, ))
v = rotor_analysis.create_input('v', val=0., shape=(1, ))
w = rotor_analysis.create_input('w', val=0, shape=(1, ))
p = rotor_analysis.create_input('p', val=0.2, shape=(1, ))
q = rotor_analysis.create_input('q', val=0.0, shape=(1, ))
r = rotor_analysis.create_input('r', val=0.0, shape=(1, ))
theta = rotor_analysis.create_input('theta', val=np.deg2rad(0))
altitude = rotor_analysis.create_input('altitude', val=0, shape=(1, ))

ac_states = AcStates(u=u, v=v, w=w, p=p, q=q, r=r, theta=theta)
atmos = get_atmosphere(altitude=altitude)


num_nodes = 1
num_radial = 100
num_tangential = num_azimuthal = 50
num_blades = 3
num_bspline_cp = 6

chord_cp = rotor_analysis.create_input('chord_cp', val=np.linspace(0.2, 0.2, num_bspline_cp), dv_flag=True, lower=0.01, upper=0.4)
twist_cp = rotor_analysis.create_input('twist_cp', val=np.deg2rad(np.linspace(45, 20, num_bspline_cp)), dv_flag=True, lower=np.deg2rad(0), upper=np.deg2rad(85))
in_plane_1 = rotor_analysis.create_input('in_plane_1', val=np.array([1., 0., 0]))
in_plane_2 = rotor_analysis.create_input('in_plane_2', val=np.array([0., 1., 0]))
thrust_vector = rotor_analysis.create_input('thrust_vector', val=np.array([0, 0, -1]).reshape(num_nodes, 3))
thrust_origin = rotor_analysis.create_input('thrust_origin', val=np.array([5, 6, 3]).reshape(num_nodes, 3))
propeller_radius = rotor_analysis.create_input('propeller_radius', val=1.)
rpm = rotor_analysis.create_input('rpm', val=1500)

pitt_peters_parameters = PittPetersParameters(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=False,
    num_cp=num_bspline_cp,
)


pitt_peters_model = PittPeters(
    name='pitt_peters_analysis',
    pitt_peters_parameters=pitt_peters_parameters,
    num_nodes=1,
    rotation_direction='cw'
)

pitt_peters_outputs = pitt_peters_model.evaluate(ac_states=ac_states, rpm=rpm, rotor_radius=propeller_radius, thrust_vector=thrust_vector, thrust_origin=thrust_origin,
                                 atmosphere=atmos, blade_chord_cp=chord_cp, blade_twist_cp=twist_cp, in_plane_1=in_plane_1, in_plane_2=in_plane_2)
rotor_analysis.register_output(pitt_peters_outputs)

thrust = pitt_peters_outputs.T
desired_thrust = 981
thrust_residual = ((thrust-desired_thrust)**2)**0.5
rotor_analysis.register_output(thrust_residual)

# # Option 1) Minimize torque subject to a thrust constraint
# rotor_analysis.add_constraint(pitt_peters_outputs.T, equals=desired_thrust, scaler=1e-3)
# rotor_analysis.add_objective(pitt_peters_outputs.Q, scaler=1e-2)

# Option 2) Minimize a thrust residual subject to a constant efficiency
rotor_analysis.add_constraint(pitt_peters_outputs.FOM, equals=0.75)
rotor_analysis.add_objective(thrust_residual, scaler=1e-2)

csdl_model = rotor_analysis.assemble_csdl()

sim = Simulator(csdl_model, analytics=True)
sim.run()

print_output(sim, rotor=rotor_analysis, comprehensive_print=False, write_to_csv=True, file_name='test_pitt_peters')
# Optimization
# prob = CSDLProblem(problem_name='pitt_peters_blade_shape_optimization', simulator=sim)
# optimizer = SLSQP(prob, maxiter=100, ftol=1E-7)
# optimizer.solve()
# optimizer.print_results()

# print_output(sim, rotor=rotor_analysis, comprehensive_print=False, write_to_csv=True, file_name='test_pitt_peters')
print('Vt mean',np.mean(sim['pitt_peters_analysis._tangential_inflow_velocity']))
print( np.mean(sim['pitt_peters_analysis._ux']))
print(sim['pitt_peters_analysis.mu'])
print(sim['pitt_peters_analysis.mu_z'])
print(sim['pitt_peters_analysis._lambda'])


import matplotlib.pyplot as plt

radius = sim['pitt_peters_analysis._radius'][0, :, :]
theta = sim['pitt_peters_analysis._theta'][0, :, :]
Vt = sim['pitt_peters_analysis._tangential_inflow_velocity'][0, :, :]

print(theta.shape)
print(radius.shape)
print(Vt.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.pcolor(theta, radius, Vt)
plt.show()


