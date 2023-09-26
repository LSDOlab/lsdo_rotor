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
num_blades = 2

# APC 10 x 5 thin electric geometry
R = 0.2540 / 2
geometry = np.array([
[0.15, 0.130,   32.76],
[0.20, 0.149,   37.19], 
[0.25, 0.173,   33.54],
[0.30, 0.189,   29.25],
[0.35, 0.197,   25.64],
[0.40, 0.201,   22.54],
[0.45, 0.200,   20.27],
[0.50, 0.194,   18.46],
[0.55, 0.186,  17.05],
[0.60, 0.174,  15.97],
[0.65, 0.160,  14.87],
[0.70, 0.145,   14.09],
[0.75, 0.128,   13.39],
[0.80, 0.112,   12.84],
[0.85, 0.096,   12.25],
[0.90, 0.081,   11.37],
[0.95, 0.061,   10.19],
[1.00, 0.041,   8.99]])

# APC 10 x 5 thin electric performance at 5000 RPM
performance = np.array([
    [0.113,   0.0912,   0.0381,   0.271],
    [0.145,  0.0890,   0.0386,   0.335],
    [0.174,   0.0864,   0.0389,   0.387],
    [0.200,   0.0834,   0.0389,   0.429],
    [0.233,   0.0786,   0.0387,   0.474],
    [0.260,   0.0734,   0.0378,   0.505],
    [0.291,   0.0662,   0.0360,   0.536],
    [0.316,   0.0612,   0.0347,   0.557],
    [0.346,   0.0543,   0.0323,   0.580],
    [0.375,   0.0489,   0.0305,   0.603],
    [0.401,   0.0451,   0.0291,   0.620],
    [0.432,   0.0401,   0.0272,   0.635],
    [0.466,   0.0345,   0.0250,   0.644],
    [0.493,   0.0297,   0.0229,   0.640],
    [0.519,   0.0254, 0.0210,   0.630],
    [0.548,   0.0204,  0.0188,   0.595],
    [0.581,   0.0145,   0.0162,   0.520],
])

from scipy import interpolate
chord_interp = interpolate.interp1d(geometry[:, 0], geometry[:, 1])
twist_interp = interpolate.interp1d(geometry[:, 0], geometry[:, 2])

twist = twist_interp(np.linspace(0.15, 1, 18))
chord = chord_interp(np.linspace(0.15, 1, 18)) * R

num_radial = twist.shape[0]

bem_mesh = BEMMesh(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    use_rotor_geometry=False,
    airfoil='NACA_4412',
    # use_custom_airfoil_ml=True,
    # use_airfoil_ml=False,
    use_byu_airfoil_model=True,
    twist_b_spline_rep=False,
    chord_b_spline_rep=False,
    normalized_hub_radius=0.15,
)

num_nodes = 30
RPM = 5000
J = np.linspace(0.1, 0.6, 30)
V = J * (RPM/60 * 2* R)

bem_model = BEM(
    component=None,
    mesh=bem_mesh,
    disk_prefix='disk',
    blade_prefix='blade',
    use_caddee=False,
    num_nodes = 30,
)



bem_model.set_module_input('chord_profile', val=chord)
bem_model.set_module_input('twist_profile', val=twist * np.pi/180)
bem_model.set_module_input('thrust_vector', val=np.tile(np.array([1, 0, 0]), (30, )).reshape(num_nodes, 3))
bem_model.set_module_input('thrust_origin', val=np.tile(np.array([0, 0, 0]), (30, )).reshape(num_nodes, 3))
bem_model.set_module_input('propeller_radius', val=R)
bem_model.set_module_input('rpm', val=RPM, dv_flag=False, lower=800, upper=5000, scaler=1e-3)
bem_model.set_module_input('u', val=V)
bem_model.set_module_input('v', val=0)
bem_model.set_module_input('w', val=0)

bem_csdl = bem_model.compute()

# bem_csdl.add_objective('Q')
bem_csdl.add_constraint('T', equals=800)

sim = Simulator(bem_csdl, analytics=True)
sim.run()

CT = sim['C_T']
CP = sim['C_P']
eta = sim['eta']
J = sim['J']

output = np.zeros((num_nodes, 4))
output[:, 0] = J
output[:, 1] = CT
output[:, 2] = CP
output[:, 3] = eta

np.save('APC_10_5_thin_electric_byu_airfoil.npy', output)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
plt.figure(1)
plt.plot(J, CT, label=r'$C_T$')
plt.plot(J, CP, label=r'$C_P$')
plt.scatter(performance[:, 0], performance[:, 1], label='exp')
plt.scatter(performance[:, 0], performance[:, 2])
plt.xlabel('J')
plt.legend()

plt.figure(2)
plt.plot(J, eta, label='lsdo_rotor')
plt.scatter(performance[:, 0], performance[:, 3], label='exp')
plt.xlabel('J')
plt.ylabel(r'$\eta$')
plt.legend()
plt.show()

exit()


