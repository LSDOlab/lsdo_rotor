import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor import RotorAnalysis, BEM, BEMParameters, AcStates, get_atmosphere, print_output
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


rotor_analysis = RotorAnalysis()

u = rotor_analysis.create_input('u', val=50, shape=(1, ))
v = rotor_analysis.create_input('v', val=0, shape=(1, ))
w = rotor_analysis.create_input('w', val=0, shape=(1, ))
altitude = rotor_analysis.create_input('altitude', val=1000, shape=(1, ))

ac_states = AcStates(u=u, v=v, w=w)
atmos = get_atmosphere(altitude=altitude)


num_nodes = 1
num_radial = 30
num_tangential = num_azimuthal = 1
num_blades = 3

chord_cp = rotor_analysis.create_input('chord_cp', val=np.linspace(0.3, 0.1, 4), dv_flag=True, lower=0.01, upper=0.4)
twist_cp = rotor_analysis.create_input('twist_cp', val=np.deg2rad(np.linspace(75, 10, 4)), dv_flag=True, lower=np.deg2rad(5), upper=np.deg2rad(85))
thrust_vector = rotor_analysis.create_input('thrust_vector', val=np.array([1, 0, 0]).reshape(num_nodes, 3))
thrust_origin = rotor_analysis.create_input('thrust_origin', val=np.array([0, 0, 0]).reshape(num_nodes, 3))
propeller_radius = rotor_analysis.create_input('propeller_radius', val=0.8)
rpm = rotor_analysis.create_input('rpm', val=2000, dv_flag=False, lower=800, upper=2000, scaler=1e-3)

bem_parameters = BEMParameters(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=True,
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

sim = Simulator(csdl_model, analytics=True)
sim.run()

print(sim[f'{bem_outputs.T.operation.name}.{bem_outputs.T.name}'])
print(rotor_analysis.operations)

print_output(sim, rotor=rotor_analysis, write_to_csv=True, file_name='test_BEM')
exit()

prob = CSDLProblem(problem_name='blade_shape_opt', simulator=sim)
optimizer = SLSQP(
    prob,
    maxiter=150, 
    ftol=1e-4,
)
optimizer.solve()
optimizer.print_results()

chord = sim['chord_profile'].flatten()
twist = sim['twist_profile'].flatten()
radius = sim['_radius'].flatten()

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False

plt.figure(1)
plt.plot(radius, chord)
plt.figure(2)
plt.plot(radius, twist * 180/np.pi)



print(sim['F'])
print(sim['eta'])
# print(sim['alpha_distribution'] * 180/np.pi)
# print(sim['alpha_ml_input'])

print(sim['Cl'])
print(sim['Cd'])
print(sim['Q'])


print(sim['mach_number'])
print(sim['Re'])
print(np.rad2deg(sim['alpha_distribution']))
print(sim['Q'])
print(sim['T'])


Cl = sim['Cl'].flatten()
Cd = sim['Cd'].flatten()

outputs = np.zeros((num_radial, 2))
outputs[:, 0] = Cl
outputs[:, 1] = Cd

M = sim['mach_number'].flatten()
Re = sim['Re'].flatten()
alpha =  np.rad2deg(sim['alpha_distribution'].flatten())

inputs = np.zeros((num_radial, 3))
inputs[:, 0] = alpha
inputs[:, 1] = Re
inputs[:, 2] = M

np.save('airfoil_inputs_3.npy', inputs)
np.save('airfoil_outputs_3.npy', outputs)


plt.show()


exit()



# sim.check_totals(of='eta', wrt='control_points', step=1e-5)


