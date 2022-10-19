import numpy as np 
import os
import csdl
from csdl import Model
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")
# from python_csdl_backend import Simulator

from lsdo_rotor.core.BEM.BEM_model import BEMModel

num_nodes = 50
num_radial = 40
num_tangential = num_azimuthal = 1


in_to_m = 0.0254

# region APC_sport_14_13
rel_dir = os.path.dirname(__file__)
APC_14_13_blade_geometry = np.loadtxt(rel_dir + '/APC_sport_14_13_blade_geometry.txt')
APC_14_13_radius = 14 * in_to_m / 2
APC_14_13_radial_nodes = APC_14_13_blade_geometry[:,0] * APC_14_13_radius
APC_14_13_interpolated_radial_nodes = np.linspace(APC_14_13_radial_nodes[0],APC_14_13_radial_nodes[-1],num_radial)
# raw and interpolated blade chord distribution
APC_14_13_blade_chord_raw = APC_14_13_blade_geometry[:,1] * APC_14_13_radius
APC_14_13_blade_chord_interpolated = np.interp(APC_14_13_interpolated_radial_nodes,APC_14_13_radial_nodes,APC_14_13_blade_chord_raw)
# raw and interpolated blade twist distribution
APC_14_13_blade_twist_raw = APC_14_13_blade_geometry[:,2]
APC_14_13_blade_twist_interpolated = np.interp(APC_14_13_interpolated_radial_nodes,APC_14_13_radial_nodes,APC_14_13_blade_twist_raw)
# wind tunnel data
APC_14_13_performance = np.loadtxt(rel_dir + '/APC_sport_14_13_performance.txt')
# endregion

import matplotlib.pyplot as plt 

# plt.plot(APC_14_13_radial_nodes,APC_14_13_blade_twist_raw,marker='*')
# plt.plot(APC_14_13_interpolated_radial_nodes,APC_14_13_blade_twist_interpolated)
# plt.show()
# exit()


# Thrust vector is the unit normal vector w.r.t the rotor disk
thrust_vector =  np.array([
    [1,0,0]]
)

# Thrust origin is the point at which the thrust acts (usually the center of the rotor disk)
thrust_origin =  np.array([
    [8.5, 0, 5]]
)

# Reference point is the point about which the moments due to thrust will be computed
reference_point = np.array([8.5, 0, 5])

shape = (num_nodes,num_radial,num_tangential)

class RunModel(Model):
    def define(self):
        # Inputs not changing across conditions (segments)
        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=APC_14_13_radius)
        self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=APC_14_13_blade_chord_interpolated)
        self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=APC_14_13_blade_twist_interpolated*np.pi/180)
        # pitch_cp = self.create_input(name='pitch_cp', shape=(4,), units='rad', val=np.linspace(80,10,4)*np.pi/180) #np.array([8.60773973e-01,6.18472835e-01,3.76150609e-01,1.88136239e-01]))#np.linspace(35,10,4)*np.pi/180)
        # chord_cp = self.create_input(name='chord_cp', shape=(2,), units='rad', val=np.array([0.35,0.14]))
        # self.add_design_variable('pitch_cp', lower=5*np.pi/180,upper=60*np.pi/180)
        
        # Inputs changing across conditions (segments), 
        #   - If the quantities are scalars, they will be expanded into shape (num_nodes,1)
        #   - If the quantities are vectors (numpy arrays), they must be specified s.t. they have shape (num_nodes,1)
        self.create_input('omega', shape=(num_nodes, 1), units='rpm/1000', val=3500)

        self.create_input(name='u', shape=(num_nodes, 1), units='m/s', val=np.array(np.linspace(0,22,num_nodes).reshape(num_nodes,1)))
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
        self.create_input(name='z', shape=(num_nodes,  1), units='m', val=0)


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
            normalized_hub_radius=0.15,
        ),name='BEM_model_1')

import time

def return_validation_model():
    return RunModel()

if __name__ == '__main__':
    t1 = time.time()
    sim = Simulator(RunModel())

    sim.run()
    t2 = time.time()

    print('Model evaluation time: ',t2-t1)
    print('Thrust: ',sim['T'])
    print('Torque: ',sim['total_torque'])
    print('eta', sim['eta'])

    import matplotlib.pyplot as plt

    chord = sim['chord_profile']
    twist = sim['twist_profile']
    radius = sim['_radius'][0,:,0].flatten()
    J_BEM = sim['J']
    eta_BEM = sim['eta']
    C_T_BEM = sim['C_T']
    C_P_BEM = sim['C_P']


    fig, axs = plt.subplots(2,2, figsize=(10,8))


    APC_14_13_J = APC_14_13_performance[:,0]
    APC_14_13_CT = APC_14_13_performance[:,1]
    APC_14_13_CP = APC_14_13_performance[:,2]
    APC_14_13_eta = APC_14_13_performance[:,3]


    axs[0,0].plot(radius,twist*180/np.pi)
    axs[0,0].set_xlabel('radius (m)')
    axs[0,0].set_ylabel('blade twist (deg)')

    axs[0,1].plot(APC_14_13_J,APC_14_13_eta,marker='o')
    axs[0,1].plot(J_BEM,eta_BEM)
    axs[0,1].set_ylim([0,1])
    axs[0,1].set_xlabel('J')
    axs[0,1].set_ylabel(r'$\eta$')

    axs[1,0].plot(APC_14_13_J,APC_14_13_CT,marker='o')
    axs[1,0].plot(J_BEM,C_T_BEM)
    axs[1,0].set_xlabel('J')
    axs[1,0].set_ylabel(r'$C_T$')

    axs[1,1].plot(APC_14_13_J,APC_14_13_CP,marker='o')
    axs[1,1].plot(J_BEM,C_P_BEM)
    axs[1,1].set_xlabel('J')
    axs[1,1].set_ylabel(r'$C_P$')

    plt.show()

