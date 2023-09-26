import numpy as np
import matplotlib.pyplot as plt


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

BEM_w_byu_airfoil = np.load('APC_10_5_thin_electric_byu_airfoil.npy')
# BEM_w_byu_airfoil = np.load('/home/marius_ruh/packages/lsdo_lab/lsdo_rotor/lsdo_rotor/APC_10_5_thin_electric_byu_airfoil.npy')
BEM_w_ucsd_airfoil = np.load('APC_10_5_thin_electric_ucsd_airfoil.npy')
# BEM_w_ucsd_airfoil = np.load('/home/marius_ruh/packages/lsdo_lab/lsdo_rotor/lsdo_rotor/APC_10_5_thin_electric_ucsd_airfoil.npy')

CC_blade_J = np.loadtxt('CC_blade_J.txt')
# CC_blade_J = np.loadtxt('/home/marius_ruh/packages/lsdo_lab/lsdo_rotor/lsdo_rotor/CC_blade_J.txt')
CC_blade_CT = np.loadtxt('CC_blade_CT.txt')
# CC_blade_CT = np.loadtxt('/home/marius_ruh/packages/lsdo_lab/lsdo_rotor/lsdo_rotor/CC_blade_CT.txt')
CC_blade_CP = np.loadtxt('CC_blade_CP.txt')
# CC_blade_CP = np.loadtxt('/home/marius_ruh/packages/lsdo_lab/lsdo_rotor/lsdo_rotor/CC_blade_CP.txt')
CC_blade_eta = np.loadtxt('CC_blade_eta.txt')
# CC_blade_eta = np.loadtxt('/home/marius_ruh/packages/lsdo_lab/lsdo_rotor/lsdo_rotor/CC_blade_eta.txt')

fig, axs = plt.subplots(1, 2, figsize=[20, 10])



axs[0].plot(BEM_w_ucsd_airfoil[:, 0], BEM_w_ucsd_airfoil[:, 1], color='red', label=r'$C_T$ lsdo_rotor with UCSD ML airfoil')
axs[0].plot(BEM_w_byu_airfoil[:, 0], BEM_w_byu_airfoil[:, 1], color='red', linestyle='--', label=r'$C_T$ lsdo_rotor with BYU airfoil')
axs[0].plot(CC_blade_J, CC_blade_CT, color='red', linestyle=':', label=r'$C_T$ CCBlade')

axs[0].plot(BEM_w_ucsd_airfoil[:, 0], BEM_w_ucsd_airfoil[:, 2], color='blue', label=r'$C_P$ lsdo_rotor with UCSD airfoil')
axs[0].plot(BEM_w_byu_airfoil[:, 0], BEM_w_byu_airfoil[:, 2], color='blue', linestyle='--', label=r'$C_P$ lsdo_rotor with BYU airfoil')
axs[0].plot(CC_blade_J, CC_blade_CP, color='blue', linestyle=':', label=r'$C_P$ CCBlade')

axs[0].scatter(performance[:, 0], performance[:, 1], color='k', label=r'experiment')
axs[0].scatter(performance[:, 0], performance[:, 2], color='k')
axs[0].set_xlabel('J')
axs[0].legend()

axs[1].plot(BEM_w_ucsd_airfoil[:, 0], BEM_w_ucsd_airfoil[:, 3], color='green', label=r'$\eta$ lsdo_rotor with UCSD airfoil')
axs[1].plot(BEM_w_byu_airfoil[:, 0], BEM_w_byu_airfoil[:, 3], color='green', linestyle='--', label=r'$\eta$ lsdo_rotor with BYU airfoil')
axs[1].plot(CC_blade_J,CC_blade_eta, color='green', linestyle=':', label=r'$\eta$ CC_blade')
axs[1].scatter(performance[:, 0], performance[:, 3], color='k', label='experiment')
axs[1].set_ylabel(r'$\eta$')
axs[1].set_xlabel('J')
axs[1].legend()

fig.suptitle('V&V of LSDO_rotor against CCBlade and APC 10x5 thin electric experiments' + '\n' + 'Comparing BYU airfoil model with UCSD ML model')
plt.savefig('lsdo_rotor_verification')
plt.show()