import numpy as np 
from csdl import Model
import csdl
from lsdo_rotor.core.pitt_peters.pitt_peters_model import PittPetersModel
from lsdo_rotor.core.BEM_caddee.functions.get_bspline_mtx import get_bspline_mtx
from lsdo_rotor.core.BEM_caddee.BEM_b_spline_comp import BsplineComp
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False


from lsdo_rotor.core.pitt_peters.pitt_peters_m3l import PittPeters, PittPetersMesh

pitt_peters_mesh = PittPetersMesh(
    num_blades=3,
    num_radial=25,
    num_tangential=25,
    airfoil='NACA_4412',
    use_airfoil_ml=False,
    use_rotor_geometry=False,
    mesh_units='m',
    chord_b_spline_rep=True,
    twist_b_spline_rep=True,
)

pitt_peters_model = PittPeters(mesh=pitt_peters_mesh, disk_prefix='', blade_prefix='', use_caddee=False)
pitt_peters_model.set_module_input('rpm', val=1200, dv_flag=True, lower=1100, upper=1300, scaler=1e-3)
pitt_peters_model.set_module_input('u', val=50)
pitt_peters_model.set_module_input('v', val=0)
pitt_peters_model.set_module_input('w', val=0)
pitt_peters_model.set_module_input('twist_cp', val=np.deg2rad(np.linspace(60, 10, 4)), dv_flag=True, lower=np.deg2rad(1), upper=np.deg2rad(60))
pitt_peters_model.set_module_input('chord_cp', val=np.linspace(0.3, 0.1, 4), dv_flag=True, lower=0.01, upper=0.3)
pitt_peters_model.set_module_input('propeller_radius', val=1.2)
pitt_peters_model.set_module_input('thrust_vector', val=np.array([0, 0, -1]))
pitt_peters_model.set_module_input('thrust_origin', val=np.array([0, 0, 0]))

csdl_model = pitt_peters_model.compute()
csdl_model.add_constraint('T', equals=3000, scaler=5e-4)
csdl_model.add_objective('total_torque', scaler=1e-3)
sim = Simulator(csdl_model)
sim.run()

Q_initial = sim['total_torque']

r = sim['_radius'][0, :, 0]
chord_initial = sim['chord_profile']
twist_initial = sim['twist_profile'] * 180/np.pi

# sim.check_totals(step=1e-7)

prob = CSDLProblem(problem_name='lpc', simulator=sim)
optimizer = SLSQP(prob, maxiter=100, ftol=1E-5)
optimizer.solve()
optimizer.print_results()


# print('Thrust: ',sim['T'])
# print('F:' ,sim['F'] )
print('Torque initial:   ', Q_initial)
print('Torque optimized: ',sim['total_torque'])
# print('M: ', sim['M'])
# print('eta', sim['eta'])



chord_opt = sim['chord_profile']
twist_opt = sim['twist_profile'] * 180/np.pi



fig, axs = plt.subplots(2, 1, figsize=(10, 15))
# sim.check_totals()
axs[0].plot(r, chord_initial / 2, color='blue')
axs[0].plot(r, chord_initial / -2, color='blue')
axs[0].plot(r, chord_opt / 2, color='red')
axs[0].plot(r, chord_opt / -2, color='red')

axs[1].plot(r, twist_initial, color='blue')
axs[1].plot(r, twist_opt, color='red')

plt.show()

exit()





















num_nodes = 1
num_radial = 30
num_tangential = num_azimuthal = 30

# Thrust vector is the unit normal vector w.r.t the rotor disk
thrust_vector =  np.array([
    [0, 0, -1],]
)

# Thrust origin is the point at which the thrust acts (usually the center of the rotor disk)
thrust_origin =  np.array([
    [8.5, 5, 5],])

# Reference point is the point about which the moments due to thrust will be computed
reference_point = np.array([8.5, 0, 5])

shape = (num_nodes,num_radial,num_tangential)

class RunModel(Model):
    def define(self):
        # Inputs not changing across conditions (segments)
        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=1.2)
        
        num_cp = 4
        order = 3
        twist_cp = self.create_input(name='twist_cp', shape=(num_cp, ), val=np.linspace(60, 10, num_cp)*np.pi/180, units='rad')
        pitch_A = get_bspline_mtx(num_cp, num_radial, order=order)
        comp = csdl.custom(twist_cp,op=BsplineComp(
            num_pt=num_radial,
            num_cp=num_cp,
            in_name='twist_cp',
            jac=pitch_A,
            out_name='twist_profile',
        ))
        self.register_output('twist_profile', comp)
        
        num_cp = 4
        order = 3
        chord_cp = self.create_input(name='chord_cp', shape=(num_cp,), val=np.linspace(0.35,0.14,num_cp))
        chord_A = get_bspline_mtx(num_cp, num_radial, order=order)
        comp_chord = csdl.custom(chord_cp,op=BsplineComp(
            num_pt=num_radial,
            num_cp=num_cp,
            in_name='chord_cp',
            jac=chord_A,
            out_name='chord_profile',
        ))
        self.register_output('chord_profile', comp_chord)


        # self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=np.linspace(0.35,0.14,num_radial))
        # self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=np.linspace(60, 10, num_radial)*np.pi/180)
        # self.add_design_variable('chord_cp',lower = np.array([0.05, 0.05, 0.05, 0.01]), upper=np.array([0.2, 0.4, 0.3, 0.3]))
        # self.add_design_variable('chord_cp',lower = 0.02, upper=0.4)
        # self.add_design_variable('twist_cp',lower = 1 * np.pi/180, upper=60*np.pi/180)

        # Inputs changing across conditions (segments), 
        #   - If the quantities are scalars, they will be expanded into shape (num_nodes,1)
        #   - If the quantities are vectors (numpy arrays), they must be specified s.t. they have shape (num_nodes,1)
        self.create_input('omega', shape=(num_nodes, 1), units='rpm/1000', val=1200)
        self.add_design_variable('omega', lower=500, upper=2000, scaler=1e-3)

        self.create_input(name='u', shape=(num_nodes, 1), units='m/s', val=40)#np.linspace(0,100,num_nodes).reshape(num_nodes,1))
        self.create_input(name='v', shape=(num_nodes, 1), units='m/s', val=0)
        self.create_input(name='w', shape=(num_nodes, 1), units='m/s', val=0)

        self.create_input(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='r', shape=(num_nodes, 1), units='rad/s', val=0)

        self.create_input(name='Phi', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Theta', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Psi', shape=(num_nodes, 1), units='rad', val=0)

        self.create_input(name='x', shape=(num_nodes,  1), units='m', val=0)
        self.create_input(name='y', shape=(num_nodes,  1), units='m', val=0)
        self.create_input(name='z', shape=(num_nodes,  1), units='m', val=1000)
                
        self.add(PittPetersModel(   
            name='propulsion',
            num_nodes=num_nodes,
            num_radial=num_radial,
            num_tangential=num_azimuthal,
            airfoil='NACA_4412',
            thrust_vector=thrust_vector,
            thrust_origin=thrust_origin,
            ref_pt=reference_point,
            num_blades=3,
sim = Simulator(pp_model)
        ),name='pitt_peters_model')

import time
import openmdao.api as om

t1 = time.time()

pp_model = RunModel()
pp_model.add_objective('total_torque', scaler=1e-3)
pp_model.add_constraint('T', equals=4000, scaler=5e-4)
sim = Simulator(pp_model)

sim.run()
t2 = time.time()
print(t2-t1)

r = sim['_radius'][0, :, 0]

chord_initial = sim['chord_profile']
twist_initial = sim['twist_profile'] * 180/np.pi

# sim.check_totals()


print('Thrust: ',sim['T'])
print('F:' ,sim['F'] )
print('Torque: ',sim['total_torque'])
print('M: ', sim['M'])
print('eta', sim['eta'])
# exit()
prob = CSDLProblem(problem_name='lpc', simulator=sim)
optimizer = SLSQP(prob, maxiter=100, ftol=1E-5)
optimizer.solve()
optimizer.print_results()


print('Thrust: ',sim['T'])
print('F:' ,sim['F'] )
print('Torque: ',sim['total_torque'])
print('M: ', sim['M'])
print('eta', sim['eta'])
chord_opt = sim['chord_profile']
twist_opt = sim['twist_profile'] * 180/np.pi



fig, axs = plt.subplots(2, 1, figsize=(10, 15))
# sim.check_totals()
axs[0].plot(r, chord_initial / 2, color='blue')
axs[0].plot(r, chord_initial / -2, color='blue')
axs[0].plot(r, chord_opt / 2, color='red')
axs[0].plot(r, chord_opt / -2, color='red')

axs[1].plot(r, twist_initial, color='blue')
axs[1].plot(r, twist_opt, color='red')

plt.show()

