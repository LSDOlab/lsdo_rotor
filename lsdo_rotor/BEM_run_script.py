import numpy as np 
import csdl
from csdl import Model
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")
# from python_csdl_backend import Simulator

from lsdo_rotor.core.BEM.BEM_model import BEMModel



num_nodes = 2
num_radial = 50
num_tangential = num_azimuthal = 1

# Thrust vector is the unit normal vector w.r.t the rotor disk
thrust_vector =  np.array([
    [0,0,-1],
    [0,0,-1]]
)

# Thrust origin is the point at which the thrust acts (usually the center of the rotor disk)
thrust_origin =  np.array([
    [8.5, 5, 5],
    [8.5, -5, 5]],)

# Reference point is the point about which the moments due to thrust will be computed
reference_point = np.array([8.5, 0, 5])

shape = (num_nodes,num_radial,num_tangential)

class RunModel(Model):
    def define(self):
        # Inputs not changing across conditions (segments)
        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=1.2)
        self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=np.linspace(0.2,0.1,num_radial))
        self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=np.linspace(50,10,num_radial)*np.pi/180)
        # pitch_cp = self.create_input(name='pitch_cp', shape=(4,), units='rad', val=np.linspace(80,10,4)*np.pi/180) #np.array([8.60773973e-01,6.18472835e-01,3.76150609e-01,1.88136239e-01]))#np.linspace(35,10,4)*np.pi/180)
        # chord_cp = self.create_input(name='chord_cp', shape=(2,), units='rad', val=np.array([0.35,0.14]))
        # self.add_design_variable('pitch_cp', lower=5*np.pi/180,upper=60*np.pi/180)
        
        # Inputs changing across conditions (segments), 
        #   - If the quantities are scalars, they will be expanded into shape (num_nodes,1)
        #   - If the quantities are vectors (numpy arrays), they must be specified s.t. they have shape (num_nodes,1)
        self.create_input('omega', shape=(num_nodes, 1), units='rpm', val=1500)

        self.create_input(name='u', shape=(num_nodes, 1), units='m/s', val=np.array([[0],[0]]))#np.linspace(0,100,num_nodes).reshape(num_nodes,1))
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


        self.add(BEMModel(   
            name='BEM_instance_1',
            num_nodes=num_nodes,
            num_radial=num_radial,
            num_tangential=num_azimuthal,
            airfoil='NACA_4412',
            thrust_vector=thrust_vector,
            thrust_origin=thrust_origin,
            ref_pt=reference_point,
            num_blades=3,
            chord_b_spline=False,
            pitch_b_spline=False,
            normalized_hub_radius=0.20,
        ),name='BEM_model_1')


import time

t1 = time.time()
sim = Simulator(RunModel())

sim.run()
t2 = time.time()

print('Model evaluation time: ',t2-t1)
print('radius: ', sim['_radius'][0,:,0])
print('Thrust: ',sim['T'])
print('F:' ,sim['F'] )
print('Torque: ',sim['total_torque'])
print('M: ', sim['M'])
print('eta', sim['eta'])



