import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel
from lsdo_rotor.utils.print_output import print_output
from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


num_nodes = 1
num_radial = 30
num_tangential = num_azimuthal = 1
num_blades = 3


bem_mesh = BEMMesh(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    use_rotor_geometry=False,
    airfoil='NACA_4412',
    use_airfoil_ml=True,
    twist_b_spline_rep=True,
    chord_b_spline_rep=True,
)


bem_model = BEM(
    component=None,
    mesh=bem_mesh,
    disk_prefix='disk',
    blade_prefix='blade',
    use_caddee=False,
)
bem_model.set_module_input('chord_cp', val=np.linspace(0.3, 0.1, 4), dv_flag=True, lower=0.01, upper=0.4)
bem_model.set_module_input('twist_cp', val=np.deg2rad(np.linspace(35, 10, 4)), dv_flag=True, lower=np.deg2rad(5), upper=np.deg2rad(85))
bem_model.set_module_input('thrust_vector', val=np.array([1, 0, 0]).reshape(num_nodes, 3))
bem_model.set_module_input('thrust_origin', val=np.array([0, 0, 0]).reshape(num_nodes, 3))
bem_model.set_module_input('propeller_radius', val=1.2)
bem_model.set_module_input('rpm', val=1400, dv_flag=False, lower=800, upper=2000, scaler=1e-3)
bem_model.set_module_input('u', val=0)
bem_model.set_module_input('v', val=0)
bem_model.set_module_input('w', val=0)

bem_csdl = bem_model.compute()

bem_csdl.add_objective('Q')
bem_csdl.add_constraint('T', equals=3500)

sim = Simulator(bem_csdl)
sim.run()


print(sim['F'])
print(sim['eta'])
# print(sim['alpha_distribution'] * 180/np.pi)
# print(sim['alpha_ml_input'])

print(sim['Cl'])
print(sim['Cd'])

# exit()
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

plt.show()


# sim.check_totals(of='eta', wrt='control_points', step=1e-5)

exit()
# Thrust vector is the unit normal vector w.r.t the rotor disk
thrust_vector =  np.array([
    [1,0,0],]
)

# Thrust origin is the point at which the thrust acts (usually the center of the rotor disk)
thrust_origin =  np.array([
    [8.5, 5, 5],]
)

# Reference point is the point about which the moments due to thrust will be computed
reference_point = np.array([8.5, 0, 5])

shape = (num_nodes, num_radial, num_tangential)

rotor_radius = 1
rpm = 1200
Vx = 40 # (for axial flow or hover only)
altitude = 2000 # in (m)

airfoil_polar = {
    'Cl_0': 0.25,
    'Cl_alpha': 5.1566,
    'Cd_0': 0.01,
    'Cl_stall': [-1, 1.5], 
    'Cd_stall': [0.02, 0.06],
    'alpha_Cl_stall': [-10, 15],
}

chord = np.linspace(0.3, 0.2, num_radial)
twist = np.linspace(60, 15, num_radial)

sim_BEM = Simulator(BEMRunModel(
    rotor_radius=rotor_radius,
    rpm=rpm,
    Vx=Vx,
    altitude=altitude,
    shape=shape,
    num_blades=num_blades,
    airfoil_name='NACA_4412',
    airfoil_polar=airfoil_polar,
    chord_distribution=chord,
    twist_distribution=twist,
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
))
rotor_dash = RotorDash()
sim_BEM.add_recorder(rotor_dash.get_recorder())
sim_BEM.run()
print_output(sim=sim_BEM, write_to_csv=True)
visualize_blade(dash=rotor_dash)
