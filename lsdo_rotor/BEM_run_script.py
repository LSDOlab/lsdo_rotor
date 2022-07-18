import numpy as np 
import csdl
from csdl import Model
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")

from lsdo_rotor.core.BEM.BEM_model import BEMModel



num_nodes = 2
num_radial = 30
num_tangential = num_azimuthal = 1

# thrust_vector = np.array([[1/2**0.5,0,-1/2**0.5]])
# thrust_vector = np.array([[0,0,-1]])
thrust_vector = np.array([[1,0,0]])

# thrust_vector = np.array([[1,0,0],
#                           [0,0,-1],
#                           [1/2**0.5,0,-1/2**0.5]])

thrust_origin=np.array([[8.5, 0, 5]])

# thrust_origin=np.array([[8.5, 0, 5],
#                         [8.5, 0, 5],
#                         [8.5, 0, 5]], dtype=float)

reference_point = np.array([4.5, 0, 5])

shape = (num_nodes,num_radial,num_tangential)

class RunModel(Model):
    def define(self):
        # Inputs not changing across conditions (segments)
        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=0.94)
        self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=np.linspace(0.2,0.1,num_radial))
        # self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=np.linspace(50,10,num_radial)*np.pi/180)
        pitch_cp = self.create_input(name='pitch_cp', shape=(4,), units='rad', val=np.array([8.60773973e-01,6.18472835e-01,3.76150609e-01,1.88136239e-01]))#np.linspace(35,10,4)*np.pi/180)
        self.add_design_variable('pitch_cp', lower=5*np.pi/180,upper=60*np.pi/180)
        # Inputs changing across conditions (segments)
        self.create_input('omega', shape=(num_nodes, 1), units='rpm/1000', val=1.44764202)

        self.create_input(name='u', shape=(num_nodes, 1), units='m/s', val=50.39014388)
        self.create_input(name='v', shape=(num_nodes, 1), units='m/s', val=0)
        self.create_input(name='w', shape=(num_nodes, 1), units='m/s', val=2.75142193)

        self.create_input(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='r', shape=(num_nodes, 1), units='rad/s', val=0)

        self.create_input(name='Phi', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Theta', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Psi', shape=(num_nodes, 1), units='rad', val=0)

        self.create_input(name='x', shape=(num_nodes,  1), units='m', val=0)
        self.create_input(name='y', shape=(num_nodes,  1), units='m', val=0)
        self.create_input(name='z', shape=(num_nodes,  1), units='m', val=1000)

        self.create_input(name='thrust_vector', shape=(num_nodes,3), val=np.tile(thrust_vector,(num_nodes,1)))
        self.create_input(name='thrust_origin', shape=(num_nodes,3), val=np.tile(thrust_origin,(num_nodes,1)))
    

        self.add(BEMModel(   
            name='propulsion',
            num_nodes=num_nodes,
            num_radial=num_radial,
            num_tangential=num_azimuthal,
            airfoil='NACA_4412',
            # thrust_vector=thrust_vector,
            # thrust_origin=thrust_origin,
            ref_pt=reference_point,
            num_blades=3,
        ),name='BEM_model')

import time

t1 = time.time()
sim = Simulator(RunModel())

sim.run()
t2 = time.time()

print(t2-t1)
print('Thrust: ',sim['T'])
print('Thrust: ',sim['total_thrust_2'])
print('Torque: ',sim['total_torque'])
print('twist', sim['twist_profile']*180/np.pi)
print('eta', sim['eta'])
# print('Torque: ',sim['total_torque_2'])
# print('in_plane_ey: ', sim['in_plane_ey'])
# print('in_plane_ex: ', sim['in_plane_ex'])
# # exit()

# # print(sim['T'].shape)
# # print(sim['total_thrust_2'])
# print('normal vectors: ',sim['thrust_vector'])
# print('Forces:',sim['F'])
# print('Moments: ',sim['M'])
# pitch_1 = sim['_pitch'][0,:,0].flatten()*180/np.pi
# pitch_cp_1 = sim['pitch_cp'].flatten()
# print('\n')
# print(sim['propeller_radius'])
# print(sim['_radius'])

# print(sim['_in_plane_inflow_velocity'])
# print(sim['inflow_z'])
# sim.prob.check_partials(compact_print=True)
# print(sim.compute_total_derivatives())
exit()
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLProblem(problem_name='Sample BEM optimization', simulator=sim)
# Setup your preferred optimizer (SLSQP) with the Problem object
optimizer = SLSQP(prob, maxiter=100, ftol=1e-10)
# Solve your optimization problem
optimizer.solve()
# Print results of optimization
optimizer.print_results()

print(sim['total_torque'])
print(sim['T'])
pitch_2 = sim['_pitch'][0,:,0].flatten()*180/np.pi
pitch_cp_2 = sim['pitch_cp'].flatten()
r_hub = sim['_radius'][0,0,0].flatten()
r_tip = sim['_radius'][0,-1,0].flatten()
pitch_cp_x = np.linspace(r_hub,r_tip,4)
# exit()
# print(sim['twist_profile'])
import matplotlib.pyplot as plt 

plt.plot(sim['_radius'][0,:,0].flatten(), pitch_2,color='navy',label='optimized curve points')
plt.scatter(pitch_cp_x, pitch_cp_2*180/np.pi,color='navy',label='optimized control points')
plt.plot(sim['_radius'][0,:,0].flatten(), pitch_1, color='maroon',label='original curve points')
plt.scatter(pitch_cp_x, pitch_cp_1*180/np.pi,color='maroon',label='original control points')
plt.legend()
plt.xlabel('radius (m)')
plt.ylabel('twist angle (deg)')
plt.title('Blade twist optimization with 4 B-spline control points in cruise')
plt.savefig('sample_blade_twist_optimization_cruise_2.png')
plt.show()
exit()
azimuth = np.linspace(0,2*np.pi,num_azimuthal) #sim2['_theta'][0,0,:].flatten()
radius = sim['_radius'][0,:,0].flatten()

dT_dist_BEM = sim['_local_thrust'].reshape(num_radial,num_azimuthal)
Cl_dist_BEM = sim['Cl_2'].reshape(num_radial,num_azimuthal)
Cd_dist_BEM = sim['Cd_2'].reshape(num_radial,num_azimuthal)
ux_BEM = sim['_ux'].reshape(num_radial,num_azimuthal)
phi_dist_BEM = sim['phi_distribution'].reshape(num_radial,num_azimuthal)

r,th = np.meshgrid(radius,azimuth)
# print(th.shape)
# print(r.shape)
import matplotlib.pyplot as plt 

fig, axs = plt.subplots(3,2, figsize=(15,10), subplot_kw={'polar':'polar'})

# vmin1 = np.min(np.array([dT_dist_pp,dT_dist_BEM]))
# vmax1 = np.max(np.array([dT_dist_pp,dT_dist_BEM]))
# p1 = axs[0,0].pcolormesh(th,r,dT_dist_pp.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
# axs[0,0].plot(azimuth,r, color='k', ls='none')
# axs[0,0].set_yticklabels([])
# # axs[0,0].set_title('Pitt-Peters sectional thrust coefficient')
# axs[0,0].set_title('Pitt–Peters sectional thrust')

p2 = axs[0,1].pcolormesh(th,r,dT_dist_BEM.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
axs[0,1].plot(azimuth,r, color='k', ls='none')
axs[0,1].set_yticklabels([])
# axs[0,1].set_title('BEM sectional thrust coefficient')
axs[0,1].set_title('BEM sectional thrust')

# vmin2 = np.min(np.array([Cl_dist_pp,Cl_dist_BEM]))
# vmax2 = np.max(np.array([Cl_dist_pp,Cl_dist_BEM]))
# p3 = axs[1,0].pcolormesh(th,r,Cl_dist_pp.T)#,vmin=vmin2,vmax=vmax2, shading='auto')
# axs[1,0].plot(azimuth,r, color='k', ls='none')
# axs[1,0].set_yticklabels([])
# axs[1,0].set_title('Pitt-Peters sectional lift coefficient')

p4 = axs[1,1].pcolormesh(th,r,Cl_dist_BEM.T)#,vmin=vmin2,vmax=vmax2, shading='auto')
axs[1,1].plot(azimuth,r, color='k', ls='none')
axs[1,1].set_yticklabels([])
axs[1,1].set_title('BEM sectional lift coefficient')

# vmin3 = np.min(np.array([ux_dist_pp,ux_BEM]))
# vmax3 = np.max(np.array([ux_dist_pp,ux_BEM]))
# # p5 = axs[2,0].pcolormesh(th,r,ux_dist_pp.T)#, vmin=vmin3,vmax=vmax3, shading='auto')
# p5 = axs[2,0].pcolormesh(th,r,Cd_dist_pp.T)
# axs[2,0].plot(azimuth,r, color='k', ls='none')
# axs[2,0].set_yticklabels([])
# axs[2,0].set_title('Pitt–Peters sectional drag coefficient')

# p6 = axs[2,1].pcolormesh(th,r,ux_BEM.T)#, vmin=vmin3,vmax=vmax3, shading='auto')
p6 = axs[2,1].pcolormesh(th,r,Cd_dist_BEM.T)
axs[2,1].plot(azimuth,r, color='k', ls='none')
axs[2,1].set_yticklabels([])
axs[2,1].set_title('BEM sectional drag coefficient')



# cb1 = fig.colorbar(p1, ax=axs[0,0], label='dT [N]')
cb2 = fig.colorbar(p2, ax=axs[0,1],  label='dT [N]')
# cb3 = fig.colorbar(p3, ax=axs[1,0],  label=r'$C_{l}$')
cb4 = fig.colorbar(p4, ax=axs[1,1],  label=r'$C_{l}$')
# cb5 = fig.colorbar(p5, ax=axs[2,0], label='Axial induced inflow [m/s]')
# cb6 = fig.colorbar(p6, ax=axs[2,1], label='Axial induced inflow [m/s]')
# cb5 = fig.colorbar(p5, ax=axs[2,0], label=r'$C_{d}$')
cb6 = fig.colorbar(p6, ax=axs[2,1], label=r'$C_{d}$')
plt.tight_layout()
# plt.show()


fig2, axs2 = plt.subplots(3,2, figsize=(15,10), subplot_kw={'polar':'polar'})
# p12 = axs2[0,0].pcolormesh(th,r,ux_dist_pp.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
# axs2[0,0].plot(azimuth,r, color='k', ls='none')
# axs2[0,0].set_yticklabels([])
# axs2[0,0].set_title('Pitt–Peters axial-induced inflow')

p22 = axs2[0,1].pcolormesh(th,r,ux_BEM.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
axs2[0,1].plot(azimuth,r, color='k', ls='none')
axs2[0,1].set_yticklabels([])
axs2[0,1].set_title('BEM axial-induced inflow')

# p32 = axs2[1,0].pcolormesh(th,r,phi_dist_pp.T * 180 / np.pi)#,vmin=vmin2,vmax=vmax2, shading='auto')
# axs2[1,0].plot(azimuth,r, color='k', ls='none')
# axs2[1,0].set_yticklabels([])
# axs2[1,0].set_title('Pitt-Peters sectional inflow angle')

p42 = axs2[1,1].pcolormesh(th,r,phi_dist_BEM.T * 180 / np.pi)#,vmin=vmin2,vmax=vmax2, shading='auto')
axs2[1,1].plot(azimuth,r, color='k', ls='none')
axs2[1,1].set_yticklabels([])
axs2[1,1].set_title('BEM sectional sectional inflow angle')


# cb12 = fig2.colorbar(p12, ax=axs2[0,0], label='axial-induced velocity [m/s]')
cb22 = fig2.colorbar(p22, ax=axs2[0,1],  label='axial-induced velocity [m/s]')
# cb32 = fig2.colorbar(p32, ax=axs2[1,0],  label=r'$\phi$ (deg)')
cb42 = fig2.colorbar(p42, ax=axs2[1,1],  label=r'$\phi$ (deg)')


plt.tight_layout()
plt.show()
