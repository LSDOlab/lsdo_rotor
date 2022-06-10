import numpy as np 
from get_simulator import get_simulator

num_evaluations = 1
num_radial = 50
num_azimuthal = 150

apc_geom_data = np.loadtxt('APC_19_12_thin_electric_geom.txt')
apc_perform_data = np.loadtxt('APC_19_12_thin_electric_perform.txt')

apc_J = apc_perform_data[:,0]
apc_C_T = apc_perform_data[:,1]
apc_C_P = apc_perform_data[:,2]
apc_eta = apc_perform_data[:,3]

apc_diameter = 0.4826
apc_radius = apc_diameter / 2

apc_norm_radius = apc_geom_data[:,0]
apc_norm_chord = apc_geom_data[:,1] 
apc_twist = apc_geom_data[:,2]

from scipy import interpolate 

f_chord = interpolate.interp1d(apc_norm_radius,apc_norm_chord)
f_twist = interpolate.interp1d(apc_norm_radius,apc_twist)

apc_chord_interp = f_chord(np.linspace(0.15,1,num_radial)) * apc_radius
apc_twist_interp = f_twist(np.linspace(0.15,1,num_radial))




analysis_dict = {
    'rotor_model': 2,
    'airfoil': 'NACA_4412_extended_range',#'Clark_Y', #E63_low_Re
    'num_evaluations': num_evaluations,
    'num_radial': num_radial,
    'num_azimuthal': num_azimuthal,  
}

analysis_dict2 = {
    'rotor_model': 3,
    'airfoil':'NACA_4412_extended_range',#NACA_4412_extended_range
    'num_evaluations': num_evaluations,
    'num_radial': num_radial,
    'num_azimuthal': num_azimuthal,  
}

rotor_dict = {
    'rotor_diameter': 2 * np.ones((num_evaluations,)),
    'num_blades': 3,
    'blade_chord_distribution': np.linspace(0.2,0.10,num_radial),
    'blade_twist_distribution': np.linspace(50,10,num_radial),
}

# rotor_dict = {
#     'rotor_diameter': apc_diameter * np.ones((num_evaluations,)),
#     'num_blades': 2,
#     'blade_chord_distribution': apc_chord_interp,
#     'blade_twist_distribution': apc_twist_interp,
# }

operating_dict = {
    'V_inf': 20 * np.ones(num_evaluations,),
    'RPM': 1200 * np.ones((num_evaluations,)),
    'rotor_disk_tilt_angle': np.array([0]),
    'Vx': np.array([0]),
    'Vy': np.array([5]),#np.linspace(0,50,num_evaluations),
    'altitude': 10,
}

# operating_dict = {
#     'V_inf': np.linspace(0,20,num_evaluations),
#     'RPM': 3000 * np.ones((num_evaluations,)),
#     'rotor_disk_tilt_angle': 90 * np.ones((num_evaluations,)),
#     'altitude': 10,
# }


sim = get_simulator(analysis_dict,rotor_dict,operating_dict)
sim2 = get_simulator(analysis_dict2,rotor_dict,operating_dict)
sim.run()
sim2.run()


# rotor_output = np.zeros((num_evaluations,7))
# rotor_output_2 = np.zeros((num_evaluations,5))
# C_T_BEM = sim['C_T'].flatten()
# C_P_BEM = sim['C_P'].flatten()
# C_Q_BEM = sim['C_Q'].flatten()
# eta_BEM = sim['eta'].flatten()

# ux_PP = np.loadtxt('ux_PP.txt').reshape(num_evaluations, num_radial, num_azimuthal)
# ux_BEM = np.loadtxt('ux_BEM.txt').reshape(num_evaluations, num_radial, num_azimuthal)
Vt = sim2['_tangential_inflow_velocity'][0,:,:].T
# print(np.max(Vt),'MAX VT')
# print(sim2['_ux'][0,15,:])


Cl_dist_pp = np.loadtxt('lift_distribution.txt').reshape(num_radial,num_azimuthal)
Cl_dist_BEM = sim['Cl_2'].reshape(num_radial,num_azimuthal)

Cd_dist_pp = np.loadtxt('drag_distribution.txt').reshape(num_radial,num_azimuthal)
Cd_dist_BEM = sim['Cd_2'].reshape(num_radial,num_azimuthal)

dT_dist_pp = np.loadtxt('section_thrust_distribution.txt').reshape(num_radial,num_azimuthal)
dT_dist_BEM = sim['_local_thrust'].reshape(num_radial,num_azimuthal)

CT_dist_pp = np.loadtxt('sectional_CT.txt').reshape(num_radial,num_azimuthal)
CT_dist_BEM = sim['dC_T'].reshape(num_radial,num_azimuthal)

phi_dist_pp = np.loadtxt('phi_distribution.txt').reshape(num_radial,num_azimuthal)
phi_dist_BEM = sim['phi_distribution'].reshape(num_radial,num_azimuthal)

ux_dist_pp = np.loadtxt('ux_distribution.txt').reshape(num_radial,num_azimuthal)
ux_BEM = sim['_ux'].reshape(num_radial,num_azimuthal)
# ux_PP = sim2['_ux']
# ux_2 = sim['_ux_2']

# np.savetxt('ux_BEM.txt', ux_BEM.flatten())
# np.savetxt('ux_PP.txt', ux_PP.flatten())
# np.savetxt('radius.txt', sim2['_radius'].flatten())

# radius = sim2['_radius'][0,:,0].flatten()
azimuth = np.linspace(0,2*np.pi,num_azimuthal) #sim2['_theta'][0,0,:].flatten()
radius = sim2['_radius'][0,:,0].flatten()
# radius = np.loadtxt('radius.txt').reshape(num_evaluations,num_radial,num_azimuthal)[0,:,0].flatten()
# azimuth = sim['_theta'][0,0,:].flatten() #np.linspace(0,2*np.pi,num_azimuthal)#



r,th = np.meshgrid(radius,azimuth)
print(th.shape)
print(r.shape)
import matplotlib.pyplot as plt 

# fig, axs = plt.subplots(3,2, figsize=(15,10), subplot_kw={'polar':'polar'})

# # plt.subplot(projection='polar')
# # p1 = axs[0].pcolormesh(th,r,ux_PP[0,:,:].T)#, shading='auto')
# # p1 = axs[0].pcolormesh(th,r,dT_dist_pp.T)
# # axs[0].plot(azimuth,r, color='k', ls='none')
# # axs[0].set_yticklabels([])
# # axs[0].set_title('Pitt-Peters Dynamic Inflow Cl distribution')

# # # p2 = axs[1].pcolormesh(th,r,ux_BEM[0,:,:].T)#, shading='auto')
# # p2 = axs[1].pcolormesh(th,r,dT_dist_BEM.T)
# # axs[1].plot(azimuth,r, color='k', ls='none')
# # axs[1].set_yticklabels([])
# # axs[1].set_title('BEM Inflow Cl distribution')
# vmin1 = np.min(np.array([dT_dist_pp,dT_dist_BEM]))
# vmax1 = np.max(np.array([dT_dist_pp,dT_dist_BEM]))
# p1 = axs[0,0].pcolormesh(th,r,dT_dist_pp.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
# axs[0,0].plot(azimuth,r, color='k', ls='none')
# axs[0,0].set_yticklabels([])
# # axs[0,0].set_title('Pitt-Peters sectional thrust coefficient')
# axs[0,0].set_title('Pitt–Peters sectional thrust')

# p2 = axs[0,1].pcolormesh(th,r,dT_dist_BEM.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
# axs[0,1].plot(azimuth,r, color='k', ls='none')
# axs[0,1].set_yticklabels([])
# # axs[0,1].set_title('BEM sectional thrust coefficient')
# axs[0,1].set_title('BEM sectional thrust')

# vmin2 = np.min(np.array([Cl_dist_pp,Cl_dist_BEM]))
# vmax2 = np.max(np.array([Cl_dist_pp,Cl_dist_BEM]))
# p3 = axs[1,0].pcolormesh(th,r,Cl_dist_pp.T)#,vmin=vmin2,vmax=vmax2, shading='auto')
# axs[1,0].plot(azimuth,r, color='k', ls='none')
# axs[1,0].set_yticklabels([])
# axs[1,0].set_title('Pitt-Peters sectional lift coefficient')

# p4 = axs[1,1].pcolormesh(th,r,Cl_dist_BEM.T)#,vmin=vmin2,vmax=vmax2, shading='auto')
# axs[1,1].plot(azimuth,r, color='k', ls='none')
# axs[1,1].set_yticklabels([])
# axs[1,1].set_title('BEM sectional lift coefficient')

# vmin3 = np.min(np.array([ux_dist_pp,ux_BEM]))
# vmax3 = np.max(np.array([ux_dist_pp,ux_BEM]))
# # p5 = axs[2,0].pcolormesh(th,r,ux_dist_pp.T)#, vmin=vmin3,vmax=vmax3, shading='auto')
# p5 = axs[2,0].pcolormesh(th,r,Cd_dist_pp.T)
# axs[2,0].plot(azimuth,r, color='k', ls='none')
# axs[2,0].set_yticklabels([])
# axs[2,0].set_title('Pitt–Peters sectional drag coefficient')

# # p6 = axs[2,1].pcolormesh(th,r,ux_BEM.T)#, vmin=vmin3,vmax=vmax3, shading='auto')
# p6 = axs[2,1].pcolormesh(th,r,Cd_dist_BEM.T)
# axs[2,1].plot(azimuth,r, color='k', ls='none')
# axs[2,1].set_yticklabels([])
# axs[2,1].set_title('BEM sectional drag coefficient')


# plt.annotate('pixel offset from axes fraction',
#             xy=(0, 0.45), xycoords='axes fraction',
#             xytext=(-200, 5), textcoords='offset pixels',
#             horizontalalignment='right',
#             verticalalignment='bottom')

# p7 = axs[3,0].pcolormesh(th,r,phi_dist_pp.T * 180/np.pi)
# axs[3,0].plot(azimuth,r, color='k', ls='none')
# axs[3,0].set_yticklabels([])
# axs[3,0].set_title('Pitt–Peters sectional inflow angle')


# p8 = axs[3,1].pcolormesh(th,r,phi_dist_BEM.T * 180/np.pi)
# axs[3,1].plot(azimuth,r, color='k', ls='none')
# axs[3,1].set_yticklabels([])
# axs[3,1].set_title('BEM sectional inflow angle')




# cb1 = fig.colorbar(p1, ax=axs[0], label='axial inflow [m/s]')
# cb2 = fig.colorbar(p2, ax=axs[1],  label='axial inflow [m/s]')

# cb1 = fig.colorbar(p1, ax=axs[0,0], label='axial inflow [m/s]')
# cb2 = fig.colorbar(p2, ax=axs[0,1],  label='axial inflow [m/s]')
# cb3 = fig.colorbar(p3, ax=axs[1,0],  label='axial inflow [m/s]')
# cb4 = fig.colorbar(p4, ax=axs[1,1],  label='axial inflow [m/s]')
# cb5 = fig.colorbar(p5, ax=axs[2,0], label='axial inflow [m/s]')
# cb6 = fig.colorbar(p6, ax=axs[2,1], label='axial inflow [m/s]')

# cb1 = fig.colorbar(p1, ax=axs[0,0], label='dT [N]')
# cb2 = fig.colorbar(p2, ax=axs[0,1],  label='dT [N]')
# cb3 = fig.colorbar(p3, ax=axs[1,0],  label=r'$C_{l}$')
# cb4 = fig.colorbar(p4, ax=axs[1,1],  label=r'$C_{l}$')
# # cb5 = fig.colorbar(p5, ax=axs[2,0], label='Axial induced inflow [m/s]')
# # cb6 = fig.colorbar(p6, ax=axs[2,1], label='Axial induced inflow [m/s]')
# cb5 = fig.colorbar(p5, ax=axs[2,0], label=r'$C_{d}$')
# cb6 = fig.colorbar(p6, ax=axs[2,1], label=r'$C_{d}$')
# plt.tight_layout()
# plt.show()


fig2, axs2 = plt.subplots(3,2, figsize=(15,10), subplot_kw={'polar':'polar'})
p12 = axs2[0,0].pcolormesh(th,r,ux_dist_pp.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
axs2[0,0].plot(azimuth,r, color='k', ls='none')
axs2[0,0].set_yticklabels([])
axs2[0,0].set_title('Pitt–Peters axial-induced inflow')

p22 = axs2[0,1].pcolormesh(th,r,ux_BEM.T)#, vmin=vmin1, vmax=vmax1, shading='auto')
axs2[0,1].plot(azimuth,r, color='k', ls='none')
axs2[0,1].set_yticklabels([])
axs2[0,1].set_title('BEM axial-induced inflow')

p32 = axs2[1,0].pcolormesh(th,r,phi_dist_pp.T * 180 / np.pi)#,vmin=vmin2,vmax=vmax2, shading='auto')
axs2[1,0].plot(azimuth,r, color='k', ls='none')
axs2[1,0].set_yticklabels([])
axs2[1,0].set_title('Pitt-Peters sectional inflow angle')

p42 = axs2[1,1].pcolormesh(th,r,phi_dist_BEM.T * 180 / np.pi)#,vmin=vmin2,vmax=vmax2, shading='auto')
axs2[1,1].plot(azimuth,r, color='k', ls='none')
axs2[1,1].set_yticklabels([])
axs2[1,1].set_title('BEM sectional sectional inflow angle')


cb12 = fig2.colorbar(p12, ax=axs2[0,0], label='axial-induced velocity [m/s]')
cb22 = fig2.colorbar(p22, ax=axs2[0,1],  label='axial-induced velocity [m/s]')
cb32 = fig2.colorbar(p32, ax=axs2[1,0],  label=r'$\phi$ (deg)')
cb42 = fig2.colorbar(p42, ax=axs2[1,1],  label=r'$\phi$ (deg)')


plt.tight_layout()
plt.show()


# eta_BEM[eta_BEM>1] = -1

# C_T_PP = sim2['C_T'].flatten()
# C_P_PP = sim2['C_P'].flatten()
# eta_PP = sim2['eta'].flatten()
# eta_PP[eta_PP>1] = -1

# J = sim['J'].flatten()

# rotor_output_2[:,0] = C_T_BEM
# rotor_output_2[:,1] = C_P_BEM
# rotor_output_2[:,2] = C_T_PP
# rotor_output_2[:,3] = C_P_PP
# rotor_output_2[:,4] = operating_dict['rotor_disk_tilt_angle']
# rotor_output_2[:,5] = eta_PP

# rotor_output[:,0] = C_T_BEM
# rotor_output[:,1] = C_P_BEM
# rotor_output[:,2] = eta_BEM
# rotor_output[:,3] = C_T_PP
# rotor_output[:,4] = C_P_PP
# rotor_output[:,5] = eta_PP
# rotor_output[:,6] = J

# np.savetxt('txt_files/non_axial_flow_Clark_Y.txt',rotor_output_2)

exit()

# rotor_data_1 = np.loadtxt('txt_files/APC_19_12_thin_electric_Eppler_E63.txt')
# rotor_data_2 = np.loadtxt('txt_files/APC_19_12_thin_electric_NACA_4412.txt')
# C_T_BEM_NACA4412 = rotor_data_2[:,0]
# C_P_BEM_NACA4412 = rotor_data_2[:,1]
# eta_BEM_NACA4412 = rotor_data_2[:,2]
# C_T_PP_NACA4412 = rotor_data_2[:,3]
# C_P_PP_NACA4412 = rotor_data_2[:,4]
# eta_PP_NACA4412 = rotor_data_2[:,5]

# C_T_BEM_Eppler_E63 = rotor_data_1[:,0]
# C_P_BEM_Eppler_E63 = rotor_data_1[:,1]
# eta_BEM_Eppler_E63 = rotor_data_1[:,2]
# C_T_PP_Eppler_E63 = rotor_data_1[:,3]
# C_P_PP_Eppler_E63 = rotor_data_1[:,4]
# eta_PP_Eppler_E63 = rotor_data_1[:,5]

# J = rotor_data_1[:,6]

# print(sim['rho'])
# print(sim['Re'])
# print(sim2['_re_pitt_peters'])
# exit()
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times"],
# })

# fig, axs = plt.subplots(1,3, figsize=(10.5, 4))
# color_string = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# font = {'fontname':'sans serif'}


# axs[0].plot(J,C_T_BEM_NACA4412, color=color_string[0])
# axs[0].plot(J,C_T_BEM_Eppler_E63, color=color_string[0], linestyle='--')
# axs[0].plot(J,C_T_PP_NACA4412,color=color_string[1])
# axs[0].plot(J,C_T_PP_Eppler_E63, color=color_string[1], linestyle='--')
# axs[0].plot(apc_J,apc_C_T,marker='*',color=color_string[2], linestyle='None')
# axs[0].set_xlabel(r'Advance Ratio J',**font)
# axs[0].set_ylabel(r'Thrust Coefficient $C_T$',**font)

# axs[1].plot(J,C_P_BEM_NACA4412, color=color_string[0])
# axs[1].plot(J,C_P_BEM_Eppler_E63, color=color_string[0], linestyle='--')
# axs[1].plot(J,C_P_PP_NACA4412,color=color_string[1])
# axs[1].plot(J,C_P_PP_Eppler_E63, color=color_string[1], linestyle='--')
# axs[1].plot(apc_J,apc_C_P,marker='*',color=color_string[2], linestyle='None')
# axs[1].set_xlabel(r'Advance Ratio J',**font)
# axs[1].set_ylabel(r'Power Coefficient $C_P$',**font)

# axs[2].plot(J,eta_BEM_NACA4412, color=color_string[0])
# axs[2].plot(J,eta_BEM_Eppler_E63, color=color_string[0], linestyle='--')
# axs[2].plot(J,eta_PP_NACA4412,color=color_string[1])
# axs[2].plot(J,eta_PP_Eppler_E63, color=color_string[1], linestyle='--')
# axs[2].plot(apc_J,apc_eta,marker='*',color=color_string[2], linestyle='None')
# axs[2].set_xlabel(r'Advance Ratio J',**font)
# axs[2].set_ylabel(r'Efficiency $\eta$',**font)
# axs[2].set_ylim([0,0.8])

# line_labels = [ r'BEM with NACA 4412',
#                 r'BEM with Eppler E63',
#                 r'Pitt-Peters with NACA 4412',
#                 r'Pitt-Peters with Eppler E63',
#                 r'APC thin electric 19x12 wind tunnel data']

# fig.legend( ncol=3, labels=line_labels,loc='upper center')


fig, axs = plt.subplots(1,2, figsize=(6.531, 3.5))

Vy = operating_dict['Vy']
Vt = (operating_dict['RPM'] * 2 * np.pi / 60) * (rotor_dict['rotor_diameter'] / 2)
 
axs[0].plot(Vy/Vt,C_T_BEM)
axs[0].plot(Vy/Vt,C_T_PP)
axs[0].set_xlabel(r'$V_y / \Omega R$')
axs[0].set_ylabel(r'Thrust Coefficient $C_T$')

axs[1].plot(Vy/Vt,C_P_BEM)
axs[1].plot(Vy/Vt,C_P_PP)
axs[1].set_xlabel(r'$V_y / \Omega R$')
axs[1].set_ylabel(r'Power Coefficient $C_P$')

line_labels = [ r'BEM',
                r'Pitt-Peters',
            ]

fig.legend( ncol=2, labels=line_labels,loc='upper center')



plt.tight_layout(rect=[0.0, 0.0, 1, 0.92])
# plt.tight_layout()

plt.show()

