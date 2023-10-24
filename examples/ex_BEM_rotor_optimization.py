"""
Example 1: Rotor chord and twist profile optimization
"""
import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor import RotorAnalysis, BEM, BEMParameters, AcStates, get_atmosphere, print_output
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


rotor_analysis = RotorAnalysis()

u = rotor_analysis.create_input('u', val=50.06, shape=(1, ))
altitude = rotor_analysis.create_input('altitude', val=1000, shape=(1, ))

ac_states = AcStates(u=u)
atmos = get_atmosphere(altitude=altitude)


num_nodes = 1
num_radial = 25
num_tangential = num_azimuthal = 1
num_blades = 5
num_bspline_cp = 6

chord_cp = rotor_analysis.create_input('chord_cp', val=np.linspace(0.2, 0.2, num_bspline_cp), dv_flag=True, lower=0.01, upper=0.4)
twist_cp = rotor_analysis.create_input('twist_cp', val=np.deg2rad(np.linspace(65, 20, num_bspline_cp)), dv_flag=True, lower=np.deg2rad(0), upper=np.deg2rad(85))
thrust_vector = rotor_analysis.create_input('thrust_vector', val=np.array([1, 0, 0]).reshape(num_nodes, 3))
thrust_origin = rotor_analysis.create_input('thrust_origin', val=np.array([0, 0, 0]).reshape(num_nodes, 3))
propeller_radius = rotor_analysis.create_input('propeller_radius', val=0.61)
rpm = rotor_analysis.create_input('rpm', val=4000)

bem_parameters = BEMParameters(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=True,
    num_cp=num_bspline_cp,
)


bem_model = BEM(
    name='bem_analysis',
    BEM_parameters=bem_parameters,
    num_nodes=1,
)

bem_outputs = bem_model.evaluate(ac_states=ac_states, rpm=rpm, rotor_radius=propeller_radius, thrust_vector=thrust_vector, thrust_origin=thrust_origin,
                                 atmosphere=atmos, blade_chord_cp=chord_cp, blade_twist_cp=twist_cp)
rotor_analysis.register_output(bem_outputs)

thrust = bem_outputs.T
desired_thrust = 981
thrust_residual = ((thrust-desired_thrust)**2)**0.5
rotor_analysis.register_output(thrust_residual)

# Option 1) Minimize torque subject to a thrust constraint
rotor_analysis.add_constraint(bem_outputs.T, equals=desired_thrust, scaler=1e-3)
rotor_analysis.add_objective(bem_outputs.Q, scaler=1e-2)

# Option 2) Minimize a thrust residual subject to a constant efficiency
# rotor_analysis.add_constraint(bem_outputs.eta, equals=0.8)
# rotor_analysis.add_objective(thrust_residual, scaler=1e-1)

csdl_model = rotor_analysis.assemble_csdl()

sim = Simulator(csdl_model, analytics=True)
sim.run()

print_output(sim, rotor=rotor_analysis, comprehensive_print=True, write_to_csv=True, file_name='test_BEM')

# Optimization
prob = CSDLProblem(problem_name='bem_blade_shape_optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=100, ftol=1E-7)
optimizer.solve()
optimizer.print_results()

print_output(sim, rotor=rotor_analysis, comprehensive_print=True, write_to_csv=True, file_name='test_BEM')
