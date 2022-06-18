import numpy as np 
from csdl import Model
try:
    from csdl_om import Simulator
except:
    raise ModuleNotFoundError("This run file requires a backend for CSDL")

from lsdo_rotor.core.BEM.BEM_model import BEMModel



num_nodes = 1
num_radial = 30
num_tangential = num_azimuthal = 1

# normal_vector = np.array([1/2**0.5,0,-1/2**0.5])
normal_vector = np.array([1,0,0])

thrust_origin=np.array([8.5, 0, 5], dtype=float)
reference_point = np.array([4.5, 0, 5])

shape = (num_nodes,num_radial,num_tangential)

class RunModel(Model):
    def define(self):
        # Inputs not changing across conditions (segments)
        self.create_input(name='rotor_radius', shape=(1, ), units='m', val=1)
        self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=np.linspace(0.2,0.1,num_radial))
        self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=np.linspace(50,10,num_radial)*np.pi/180)


        # Inputs changing across conditions (segments)
        self.create_input('omega', shape=(num_nodes, 1), units='rpm', val=1500)

        self.create_input(name='u', shape=(num_nodes, 1), units='m/s', val=50)
        self.create_input(name='v', shape=(num_nodes, 1), units='m/s', val=0)
        self.create_input(name='w', shape=(num_nodes, 1), units='m/s', val=0)

        self.create_input(name='p', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='q', shape=(num_nodes, 1), units='rad/s', val=0)
        self.create_input(name='r', shape=(num_nodes, 1), units='rad/s', val=0)

        self.create_input(name='Phi', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Theta', shape=(num_nodes, 1), units='rad', val=0)
        self.create_input(name='Psi', shape=(num_nodes, 1), units='rad', val=0)

        self.create_input(name='x', shape=(num_nodes, ), units='m', val=0)
        self.create_input(name='y', shape=(num_nodes, ), units='m', val=0)
        self.create_input(name='z', shape=(num_nodes, ), units='m', val=1000)
                
        self.add(BEMModel(   
            name='propulsion',
            num_nodes=num_nodes,
            num_radial=num_radial,
            num_tangential=num_azimuthal,
            airfoil='NACA_4412',
            thrust_vector=normal_vector,
            thrust_origin=thrust_origin,
            ref_pt=reference_point,
            num_blades=3,
        ),name='BEM_model')

import time

t1 = time.time()
sim = Simulator(RunModel())

sim.run()
t2 = time.time()

print(t2-t1)
exit()

print(sim['T'].shape)
print(sim['total_thrust_2'])
print(sim['normal_vector'])
print(sim['F'])
print(sim['M'])
# print(sim['_in_plane_inflow_velocity'])
# print(sim['inflow_z'])

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
