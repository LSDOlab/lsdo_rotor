import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 13})
import pickle


r = np.loadtxt('r')
dT_BEM = np.loadtxt('dT_BEM')
dQ_BEM = np.loadtxt('dQ_BEM')
dT_BILD = np.loadtxt('dT_BILD' )
dQ_BILD = np.loadtxt('dQ_BILD' )

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
axs[0].plot(r, dT_BEM, color='navy', label='BEM optimized')
axs[0].plot(r, dT_BILD, color='maroon', label='BILD method')
axs[0].set_xlabel('radius (m)')
axs[0].set_ylabel('dT (N)')
axs[0].legend()

axs[1].plot(r, dQ_BEM, color='navy')
axs[1].plot(r, dQ_BILD, color='maroon')
axs[1].set_xlabel('radius (m)')
axs[1].set_ylabel('dQ (N-m)')

fig.tight_layout()

plt.show()
# _Vx = [0, 10, 20, 30, 40, 50, 60]
# _Vt = [10, 20, 30, 40, 50, 60, 70]
# _eta2 = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# L_o_D = np.linspace(1, 150, 100)
# color = np.linspace(0, 0.8, len(_Vx))
# fig, axs = plt.subplots(1, 3, figsize=(14, 5))

# for i in range(len(_Vx)):
#     ux1 = u_x(_Vx[i], _Vt[3], L_o_D, _eta2[3])
#     axs[0].plot(L_o_D, ux1/max(ux1), color=f'{color[i]}', label=r'$V_x$ = {} m/s'.format(_Vx[i]))
#     axs[0].set_xlabel(r'Lift-to-drag ratio')
#     axs[0].set_ylabel(r'Normalized $u_x$')
#     axs[0].set_title(r'$V_\theta$ = {} m/s and $\eta_2$ = {}'.format(_Vt[3], _eta2[3]))
#     axs[0].legend()

#     ux2 = u_x(_Vx[3], _Vt[i], L_o_D, _eta2[3])
#     axs[1].plot(L_o_D, ux2/max(ux2), color=f'{color[i]}', label=r'$V_\theta$ = {} m/s'.format(_Vt[i]))
#     axs[1].set_xlabel(r'Lift-to-drag ratio')
#     axs[1].set_ylabel(r'Normalized $u_x$')
#     axs[1].set_title(r'$V_x$ = {} m/s and $\eta_2$ = {}'.format(_Vx[3], _eta2[3]))
#     axs[1].legend()

#     ux3 = u_x(_Vx[3], _Vt[3], L_o_D, _eta2[i])
#     axs[2].plot(L_o_D, ux3/max(ux3), color=f'{color[i]}', label=r'$\eta_2$ = {}'.format(_eta2[i]))
#     axs[2].set_xlabel(r'Lift-to-drag ratio')
#     axs[2].set_ylabel(r'Normalized $u_x$')
#     axs[2].set_title(r'$V_x$ = {} m/s and $V_\theta$ = {} m/s'.format(_Vx[3], _Vt[3]))
#     axs[2].legend()
    
# fig.tight_layout()

# plt.show()

# with open(f'full_factorial_sweep_w_advance_ratio_performance_metrics_fixed_bug_1_run_2.pickle', 'rb') as handle:
#     data = pickle.load(handle)
# print(data.shape)
# exit()
sweep_results = np.zeros((1, 17))

for i in range(4):
    j = i + 1
    with open(f'full_factorial_sweep_w_advance_ratio_performance_metrics_fixed_bug_{j}_run_2.pickle', 'rb') as handle:
        data = pickle.load(handle)
        sweep_results = np.append(data, sweep_results, axis=0)

num_total = sweep_results.shape[0]    
exit_codes = sweep_results[:, 10]
converged_ind = np.where(exit_codes==1)[0]
non_converged_ind = np.where(exit_codes!=1)[0]
num_converged = len(converged_ind)
num_non_converged = len(non_converged_ind)

converged_results_only = sweep_results[converged_ind, :]
non_converged_results_only = sweep_results[non_converged_ind, :]

BILD_times_conv = converged_results_only[:, 11]
BEM_opt_time_conv = converged_results_only[:, 12]

# print(np.mean(BEM_opt_time_conv)/np.mean(BILD_times_conv))
# print(np.mean(BEM_opt_time_conv))
# exit()

# print(np.mean(BILD_times_conv))
# print(np.mean(BEM_opt_time_conv))
# exit()

energy_error = converged_results_only[:, 15]
eps_chord = converged_results_only[:, 13]
eps_twist = converged_results_only[:, 14]
eps_eta = converged_results_only[:, 8]
eps_eta_no_nan = np.array([eta for eta in eps_eta if str(eta) != 'nan'])

BILD_times_non_conv_BEM = [time for time in non_converged_results_only[:, 11] if time != 0]
BEM_opt_time_non_conv = [time for time in non_converged_results_only[:, 12] if time != 0]



print(np.mean(BILD_times_conv))
print(np.mean(BEM_opt_time_conv))
exit()

print('Percentage of converged optimizations:   ', num_converged/num_total * 100)
print('-----Efficiency-----')
print('Average eta error:                       ', np.mean(eps_eta_no_nan))
print('Max eta error:                           ', np.max(eps_eta_no_nan))
print('Min eta error:                           ', np.min(eps_eta_no_nan))
print('-----Chord/ twist/ energy-----')
print('Chord error:                             ', np.mean(eps_chord))
print('twist error:                             ', np.mean(eps_twist))
print('energy error:                            ', np.mean(energy_error))
# plt.plot(BILD_times, BEM_opt_time)
plt.scatter(BILD_times_conv, BEM_opt_time_conv, s=6, label='Converged BEM optimization', alpha=0.5)
plt.scatter(BILD_times_non_conv_BEM, BEM_opt_time_non_conv, s=6, label='Non-converged BEM optimization', alpha=0.5)

plt.xlabel('BILD method time (s)')
plt.ylabel('SNOPT optimization time (s)')
plt.legend()

# plt.show()
# exit()

fig0, axs0 = plt.subplots(2, 1, figsize=(7, 7))
axs0[0].hist(eps_chord, bins=50)
axs0[0].set_xlabel('Percent error of chord profile')
axs0[0].set_ylabel('Frequency')

axs0[1].hist(eps_twist, bins=50)
axs0[1].set_xlabel('Percent error of twist profile')
axs0[1].set_ylabel('Frequency')
fig0.tight_layout()
# plt.hist(energy_error, bins=50)

# plt.title(f'Percentage of converged BEM optimziations: {num_converged/num_total * 100}%')
# plt.title(r'Rotor Efficiency % Error'+ '\n' + r'$(\eta_{BILD}-\eta_{BEM opt.})/\eta_{BEM opt.} * 100$')


sweep_results = np.zeros((1, 17))

for i in range(4):
    j = i + 1
    with open(f'full_factorial_sweep_w_advance_ratio_performance_metrics_fixed_bug_{j}_run_2.pickle', 'rb') as handle:
        data = pickle.load(handle)
        sweep_results = np.append(data, sweep_results, axis=0)

# hover_results_only = sweep_results[0:1000,:]

num_total = sweep_results.shape[0] # hover_results_only.shape[0] #
exit_codes = sweep_results[:, 10] # hover_results_only[:,- 1] #
converged_ind = np.where(exit_codes==1)[0]
non_converged_ind = np.where(exit_codes!=1)[0]
num_converged = len(converged_ind)
num_non_converged = len(non_converged_ind)

converged_results_only = sweep_results[converged_ind, :] #hover_results_only[converged_ind, :] 

# HOVER
hover_ind = np.where(converged_results_only[:, 6]==0.)[0]
cruise_ind = np.where(converged_results_only[:, 6]!=0.)[0]

hover_C_T_BEM = converged_results_only[hover_ind, 0]
hover_C_T_BILD = converged_results_only[hover_ind, 1]
hover_C_P_BEM = converged_results_only[hover_ind, 4]
hover_C_P_BILD = converged_results_only[hover_ind, 5]
hover_C_Q_BEM = converged_results_only[hover_ind, 2]
hover_C_Q_BILD = converged_results_only[hover_ind, 3]
FM_BEM = hover_C_T_BEM * (hover_C_T_BEM/2)**0.5 / hover_C_P_BEM
FM_BILD = hover_C_T_BILD * (hover_C_T_BILD/2)**0.5 / hover_C_P_BILD
hover_eps_chord = converged_results_only[hover_ind, 13]
hover_eps_twist = converged_results_only[hover_ind, 14]
hover_eps_energy = converged_results_only[hover_ind, 16]

cruise_eps_chord = converged_results_only[cruise_ind, 13]
cruise_eps_twist = converged_results_only[cruise_ind, 14]
cruise_eps_energy = converged_results_only[cruise_ind, 16]

cruise_energy_BILD = converged_results_only[cruise_ind, 9]
hover_energy_BILD = converged_results_only[hover_ind, 9]

cruise_energy_BEM = converged_results_only[cruise_ind, 8]
hover_energy_BEM = converged_results_only[hover_ind, 8]

print('Cruise energy BEM', [np.min(cruise_energy_BEM), np.max(cruise_energy_BEM)])
print('Cruise energy BILD', [np.min(cruise_energy_BILD), np.max(cruise_energy_BILD)])
print('\n')
print('Hover energy BEM', [np.min(hover_energy_BEM), np.max(hover_energy_BEM)])
print('Hover energy BILD', [np.min(hover_energy_BILD), np.max(hover_energy_BILD)])

exit()
print('Hover chord error', [np.min(hover_eps_chord), np.mean(hover_eps_chord), np.max(hover_eps_chord)])
print('Hover twist error', [np.min(hover_eps_twist), np.mean(hover_eps_twist), np.max(hover_eps_twist)])
print('Hover energy error', [np.min(hover_eps_energy), np.mean(hover_eps_energy), np.max(hover_eps_energy)])

print('Cruise chord error', [np.min(cruise_eps_chord), np.mean(cruise_eps_chord), np.max(cruise_eps_chord)])
print('Cruise twist error', [np.min(cruise_eps_twist), np.mean(cruise_eps_twist), np.max(cruise_eps_twist)])
print('Cruise energy error', [np.min(cruise_eps_energy), np.mean(cruise_eps_energy), np.max(cruise_eps_energy)])
exit()
print('\n')
print('Average hover C_T error', np.mean(abs(hover_C_T_BILD-hover_C_T_BEM)/hover_C_T_BEM)*100)
print('Min hover C_T error', np.min(abs(hover_C_T_BILD-hover_C_T_BEM)/hover_C_T_BEM)*100)
print('max hover C_T error', np.max(abs(hover_C_T_BILD-hover_C_T_BEM)/hover_C_T_BEM)*100)
print(f'min max BILD [{min(hover_C_T_BILD)},{max(hover_C_T_BILD)}]')
print(f'min max BEM [{min(hover_C_T_BEM)},{max(hover_C_T_BEM)}]')
hover_C_T_min = min(min(hover_C_T_BEM),min(hover_C_T_BILD)) - 0.02
hover_C_T_max = max(max(hover_C_T_BEM),max(hover_C_T_BILD)) + 0.02

print('\n')
print('Average hover C_Q error', np.mean(abs(hover_C_Q_BILD-hover_C_Q_BEM)/hover_C_Q_BEM)*100)
print('Min hover C_Q error', np.min(abs(hover_C_Q_BILD-hover_C_Q_BEM)/hover_C_Q_BEM)*100)
print('Max hover C_Q error', np.max(abs(hover_C_Q_BILD-hover_C_Q_BEM)/hover_C_Q_BEM)*100)
print(f'min max BILD [{min(hover_C_Q_BILD)},{max(hover_C_Q_BILD)}]')
print(f'min max BEM [{min(hover_C_Q_BEM)},{max(hover_C_Q_BEM)}]')
hover_C_Q_min = min(min(hover_C_Q_BILD),min(hover_C_Q_BEM)) - 0.005
hover_C_Q_max = max(max(hover_C_Q_BILD),max(hover_C_Q_BEM)) + 0.005

print('\n')
print('Average C_P error', np.mean(abs(hover_C_P_BILD-hover_C_P_BEM)/hover_C_P_BEM)*100)
print('Min C_P error', np.min(abs(hover_C_P_BILD-hover_C_P_BEM)/hover_C_P_BEM)*100)
print('Max C_P error', np.max(abs(hover_C_P_BILD-hover_C_P_BEM)/hover_C_P_BEM)*100)
print(f'min max BILD [{min(hover_C_P_BILD)},{max(hover_C_P_BILD)}]')
print(f'min max BEM [{min(hover_C_P_BEM)},{max(hover_C_P_BEM)}]')
hover_C_P_min = min(min(hover_C_P_BILD), min(hover_C_P_BEM)) - 0.03
hover_C_P_max = max(max(hover_C_P_BILD), max(hover_C_P_BEM)) + 0.03

print('\n')
print('Average FM error', np.mean(abs(FM_BILD-FM_BEM)/FM_BEM)*100)
print('Min FM error', np.min(abs(FM_BILD-FM_BEM)/FM_BEM)*100)
print('Max FM error', np.max(abs(FM_BILD-FM_BEM)/FM_BEM)*100)
print(f'min max FM BILD [{min(FM_BILD)},{max(FM_BILD)}]')
print(f'min max FM BEM [{min(FM_BEM)},{max(FM_BEM)}]')
FM_min = min(min(FM_BEM),min(FM_BILD)) - 0.01
FM_max = max(max(FM_BEM),max(FM_BILD)) + 0.01

# CRUISE
cruise_ind = np.where(converged_results_only[:, 6]!=0.)[0]
cruise_C_T_BEM = converged_results_only[cruise_ind, 0]
cruise_C_T_BILD = converged_results_only[cruise_ind, 1]
cruise_C_P_BEM = converged_results_only[cruise_ind, 4]
cruise_C_P_BILD = converged_results_only[cruise_ind, 5]
cruise_C_Q_BEM = converged_results_only[cruise_ind, 2]
cruise_C_Q_BILD = converged_results_only[cruise_ind, 3]
cruise_eta_BEM = converged_results_only[cruise_ind, 6]
cruise_eta_BILD = converged_results_only[cruise_ind, 7]

print('\n')
print('\n')
print('Average cruise C_T error', np.mean(abs(cruise_C_T_BILD-cruise_C_T_BEM)/cruise_C_T_BEM)*100)
print('Min cruise C_T error', np.min(abs(cruise_C_T_BILD-cruise_C_T_BEM)/cruise_C_T_BEM)*100)
print('Max cruise C_T error', np.max(abs(cruise_C_T_BILD-cruise_C_T_BEM)/cruise_C_T_BEM)*100)
print(f'min max BILD [{min(cruise_C_T_BILD)},{max(cruise_C_T_BILD)}]')
print(f'min max BEM [{min(cruise_C_T_BEM)},{max(cruise_C_T_BEM)}]')
cruise_C_T_min = min(min(cruise_C_T_BEM),min(cruise_C_T_BILD)) - 0.02
cruise_C_T_max = max(max(cruise_C_T_BEM),max(cruise_C_T_BILD)) + 0.02

print('\n')
print('Average cruise C_Q error', np.mean(abs(cruise_C_Q_BILD-cruise_C_Q_BEM)/cruise_C_Q_BEM)*100)
print('Min cruise C_Q error', np.min(abs(cruise_C_Q_BILD-cruise_C_Q_BEM)/cruise_C_Q_BEM)*100)
print('Max cruise C_Q error', np.max(abs(cruise_C_Q_BILD-cruise_C_Q_BEM)/cruise_C_Q_BEM)*100)
print(f'min max BILD [{min(cruise_C_Q_BILD)},{max(cruise_C_Q_BILD)}]')
print(f'min max BEM [{min(cruise_C_Q_BEM)},{max(cruise_C_Q_BEM)}]')
cruise_C_Q_min = min(min(cruise_C_Q_BILD),min(cruise_C_Q_BEM)) - 0.005
cruise_C_Q_max = max(max(cruise_C_Q_BILD),max(cruise_C_Q_BEM)) + 0.005

print('\n')
print('Average C_P error', np.mean(abs(cruise_C_P_BILD-cruise_C_P_BEM)/cruise_C_P_BEM)*100)
print('Min C_P error', np.min(abs(cruise_C_P_BILD-cruise_C_P_BEM)/cruise_C_P_BEM)*100)
print('Max C_P error', np.max(abs(cruise_C_P_BILD-cruise_C_P_BEM)/cruise_C_P_BEM)*100)
print(f'min max BILD [{min(cruise_C_P_BILD)},{max(cruise_C_P_BILD)}]')
print(f'min max BEM [{min(cruise_C_P_BEM)},{max(cruise_C_P_BEM)}]')
cruise_C_P_min = min(min(cruise_C_P_BILD), min(cruise_C_P_BEM)) - 0.03
cruise_C_P_max = max(max(cruise_C_P_BILD), max(cruise_C_P_BEM)) + 0.03

print('\n')
print('Average eta error', np.mean(abs(cruise_eta_BILD-cruise_eta_BEM)/cruise_eta_BEM)*100)
print('Min eta error', np.min(abs(cruise_eta_BILD-cruise_eta_BEM)/cruise_eta_BEM)*100)
print('Max eta error', np.max(abs(cruise_eta_BILD-cruise_eta_BEM)/cruise_eta_BEM)*100)
print(f'min max BILD [{min(cruise_eta_BILD)},{max(cruise_eta_BILD)}]')
print(f'min max BEM [{min(cruise_eta_BEM)},{max(cruise_eta_BEM)}]')
eta_min = min(min(cruise_eta_BEM),min(cruise_eta_BILD)) - 0.04
eta_max = max(max(cruise_eta_BEM),max(cruise_eta_BILD)) + 0.04


energy_BEM = converged_results_only[:, 8]
energy_BILD = converged_results_only[:, 9]
# print(f'min max Energy BILD [{min(energy_BILD)/1000},{max(energy_BILD)/1000}]')
# print(f'min max Energy BEM [{min(energy_BEM)/1000},{max(energy_BEM)/1000}]')
# print(np.mean(abs(energy_BILD-energy_BEM)/energy_BEM)*100)
# exit()

print('\n')
print('\n')
print('num_total', num_total)
print('Percentage of converged optimizations:   ', num_converged/num_total * 100)


not_converged_int = np.where(exit_codes!=1)[0]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axs[0, 0].axline((0, 0), slope=1, color='k')
axs[0, 0].scatter(cruise_C_T_BEM, cruise_C_T_BILD, s=5)
axs[0, 0].set_ylim([cruise_C_T_min, cruise_C_T_max])
axs[0, 0].set_xlim([cruise_C_T_min, cruise_C_T_max])
axs[0, 0].set_xlabel(r'$C_{T}$ BEM optimization')
axs[0, 0].set_ylabel(r'$C_{T}$ BILD')


axs[0, 1].axline((0, 0), slope=1, color='k')
axs[0, 1].scatter(cruise_C_P_BEM, cruise_C_P_BILD, s=5)
axs[0, 1].set_ylim([cruise_C_P_min, cruise_C_P_max])
axs[0, 1].set_xlim([cruise_C_P_min, cruise_C_P_max])
axs[0, 1].set_xlabel(r'$C_{P}$ BEM optimization')
axs[0, 1].set_ylabel(r'$C_{P}$ BILD')


axs[1, 0].axline((0, 0), slope=1, color='k')
axs[1, 0].scatter(cruise_C_Q_BEM, cruise_C_Q_BILD, s=5)
axs[1, 0].set_ylim([cruise_C_Q_min, cruise_C_Q_max])
axs[1, 0].set_xlim([cruise_C_Q_min, cruise_C_Q_max])
axs[1, 0].set_xlabel(r'$C_{Q}$ BEM optimization')
axs[1, 0].set_ylabel(r'$C_{Q}$ BILD')


axs[1, 1].axline((0, 0), slope=1, color='k')
axs[1, 1].scatter(cruise_eta_BEM, cruise_eta_BILD, s=5)
axs[1, 1].set_ylim([eta_min, eta_max])
axs[1, 1].set_xlim([eta_min, eta_max])
axs[1, 1].set_xlabel(r'$\eta$ BEM optimization')
axs[1, 1].set_ylabel(r'$\eta$ BILD')

fig.tight_layout()


fig2, axs2 = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axs2[0, 0].axline((0, 0), slope=1, color='k')
axs2[0, 0].scatter(hover_C_T_BEM, hover_C_T_BILD, s=5)
axs2[0, 0].set_ylim([hover_C_T_min, hover_C_T_max])
axs2[0, 0].set_xlim([hover_C_T_min, hover_C_T_max])
axs2[0, 0].set_xlabel(r'$C_{T}$ BEM optimization')
axs2[0, 0].set_ylabel(r'$C_{T}$ BILD')


axs2[0, 1].axline((0, 0), slope=1, color='k')
axs2[0, 1].scatter(hover_C_P_BEM, hover_C_P_BILD, s=5)
axs2[0, 1].set_ylim([hover_C_P_min, hover_C_P_max])
axs2[0, 1].set_xlim([hover_C_P_min, hover_C_P_max])
axs2[0, 1].set_xlabel(r'$C_{P}$ BEM optimization')
axs2[0, 1].set_ylabel(r'$C_{P}$ BILD')


axs2[1, 0].axline((0, 0), slope=1, color='k')
axs2[1, 0].scatter(hover_C_Q_BEM, hover_C_Q_BILD, s=5)
axs2[1, 0].set_ylim([hover_C_Q_min, hover_C_Q_max])
axs2[1, 0].set_xlim([hover_C_Q_min, hover_C_Q_max])
axs2[1, 0].set_xlabel(r'$C_{Q}$ BEM optimization')
axs2[1, 0].set_ylabel(r'$C_{Q}$ BILD')


axs2[1, 1].axline((0, 0), slope=1, color='k')
axs2[1, 1].scatter(FM_BEM, FM_BILD, s=5)
axs2[1, 1].set_ylim([FM_min, FM_max])
axs2[1, 1].set_xlim([FM_min, FM_max])
axs2[1, 1].set_xlabel(r'FM BEM optimization')
axs2[1, 1].set_ylabel(r'FM BILD')

fig2.tight_layout()
plt.show()

# exit()