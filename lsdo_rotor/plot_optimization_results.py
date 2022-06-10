import numpy as np
import matplotlib.pyplot as plt 
from pytikz.matplotlib_utils import use_latex_fonts
import seaborn as sns
sns.set()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})

fig, big_axes = plt.subplots(figsize=(6, 8), nrows= 5, ncols =1)

# for row, big_ax in enumerate(big_axes, start=1):
#     big_ax.set_title("Subplot row %s \n" % row, fontsize=12)

#     # Turn off axis lines and ticks of the big subplot 
#     # obs alpha is 0 in RGBA string!
#     # big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
#     # removes the white frame
#     big_ax._frameon = False

ildm_data1 = np.loadtxt('ildm_geometry_J_05_R_2_Vx_15.txt')
ideal_chord1 = ildm_data1[:,0]
ideal_twist1 = ildm_data1[:,1]
ideal_loss1  = ildm_data1[:,4]

ildm_data2 = np.loadtxt('ildm_geometry_J_1_R_2_Vx_30.txt')
ideal_chord2 = ildm_data2[:,0]
ideal_twist2 = ildm_data2[:,1]
ideal_loss2  = ildm_data2[:,4]

ildm_data3 = np.loadtxt('ildm_geometry_J_15_R_2_Vx_45.txt')
ideal_chord3 = ildm_data3[:,0]
ideal_twist3 = ildm_data3[:,1]
ideal_loss3  = ildm_data3[:,4]

ildm_data4 = np.loadtxt('ildm_geometry_J_2_R_2_Vx_60.txt')
ideal_chord4 = ildm_data4[:,0]
ideal_twist4 = ildm_data4[:,1]
ideal_loss4  = ildm_data4[:,4]

ildm_data5 = np.loadtxt('ildm_geometry_J_25_R_2_Vx_75.txt')
ideal_chord5 = ildm_data5[:,0]
ideal_twist5 = ildm_data5[:,1]
ideal_loss5  = ildm_data5[:,4]

BEM_opt_data1 = np.loadtxt('BEM_opt_geometry_J_05_R_2_Vx_15.txt')
BEM_opt_chord1 = BEM_opt_data1[:,0]
BEM_opt_twist1 = BEM_opt_data1[:,1]
BEM_opt_loss1 = BEM_opt_data1[:,4]

BEM_opt_data2 = np.loadtxt('BEM_opt_geometry_J_1_R_2_Vx_30.txt')
BEM_opt_chord2 = BEM_opt_data2[:,0]
BEM_opt_twist2 = BEM_opt_data2[:,1]
BEM_opt_loss2 = BEM_opt_data2[:,4]

BEM_opt_data3 = np.loadtxt('BEM_opt_geometry_J_15_R_2_Vx_45.txt')
BEM_opt_chord3 = BEM_opt_data3[:,0]
BEM_opt_twist3 = BEM_opt_data3[:,1]
BEM_opt_loss3 = BEM_opt_data3[:,4]

BEM_opt_data4 = np.loadtxt('BEM_opt_geometry_J_2_R_2_Vx_60.txt')
BEM_opt_chord4 = BEM_opt_data4[:,0]
BEM_opt_twist4 = BEM_opt_data4[:,1]
BEM_opt_loss4 = BEM_opt_data4[:,4]

BEM_opt_data5 = np.loadtxt('BEM_opt_geometry_J_25_R_2_Vx_75.txt')
BEM_opt_chord5 = BEM_opt_data5[:,0]
BEM_opt_twist5 = BEM_opt_data5[:,1]
BEM_opt_loss5 = BEM_opt_data5[:,4]

radius = BEM_opt_data1[:,-1]

sub1 = fig.add_subplot(5,3,1)
sub1.plot(radius, ideal_chord1 / 2, marker = 'o',markersize = 2,color= 'firebrick')
sub1.plot(radius, BEM_opt_chord1 / 2,marker = 'o',markersize = 2, color= 'navy')
sub1.plot(radius,  ideal_chord1 / -2, marker = 'o',markersize = 2,color= 'firebrick')
sub1.plot(radius, BEM_opt_chord1 / -2, marker = 'o',markersize = 2,color= 'navy')
# sub1.set_xlabel('radius (m)')
sub1.set_ylabel(r'c (m)')

sub2 = fig.add_subplot(5,3,2)
sub2.plot(radius, ideal_twist1, marker = 'o',markersize = 2,color= 'firebrick')
sub2.plot(radius, BEM_opt_twist1, marker = 'o',markersize = 2,color= 'navy')
# sub2.set_xlabel('radius (m)')
sub2.set_ylabel(r'$\theta$ (deg)')

sub3 = fig.add_subplot(5,3,3)
sub3.plot(radius, ideal_loss1, marker = 'o',markersize = 2,color= 'firebrick')
sub3.plot(radius, BEM_opt_loss1, marker = 'o',markersize = 2,color= 'navy')
# sub3.set_xlabel('radius (m)')
sub3.set_ylabel(r'dE (J/s)')

big_axes[0].set_title(r'J = 0.5, $T_{ildm} = T_{BEM \,opt.} = 614\,N$', y=1.1)
big_axes[0]._frameon = False
big_axes[0].set_yticklabels([])
big_axes[0].set_xticklabels([])

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
sub4 = fig.add_subplot(5,3,4)
sub4.plot(radius, ideal_chord2 / 2, marker = 'o',markersize = 2,color= 'firebrick')
sub4.plot(radius,  ideal_chord2 / -2, marker = 'o',markersize = 2,color= 'firebrick')
sub4.plot(radius, BEM_opt_chord2 / 2,marker = 'o',markersize = 2, color= 'navy')
sub4.plot(radius, BEM_opt_chord2 / -2, marker = 'o',markersize = 2,color= 'navy')
# sub4.set_xlabel('radius (m)')
sub4.set_ylabel(r'c (m)')

sub5 = fig.add_subplot(5,3,5)
sub5.plot(radius, ideal_twist2, marker = 'o',markersize = 2,color= 'firebrick')
sub5.plot(radius, BEM_opt_twist2, marker = 'o',markersize = 2,color= 'navy')
# sub5.set_xlabel('radius (m)')
sub5.set_ylabel(r'$\theta$ (deg)')

sub6 = fig.add_subplot(5,3,6)
sub6.plot(radius, ideal_loss2, marker = 'o',markersize = 2,color= 'firebrick')
sub6.plot(radius, BEM_opt_loss2, marker = 'o',markersize = 2,color= 'navy')
# sub6.set_xlabel('radius (m)')
sub6.set_ylabel(r'dE (J/s)')

big_axes[1].set_title(r'J = 1, $T_{ildm} = T_{BEM \,opt.} = 637\,N$', y=1.1)
big_axes[1]._frameon = False
big_axes[1].set_yticklabels([])
big_axes[1].set_xticklabels([])


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
sub7 = fig.add_subplot(5,3,7)
sub7.plot(radius, ideal_chord3 / 2, marker = 'o',markersize = 2,color= 'firebrick')
sub7.plot(radius,  ideal_chord3 / -2, marker = 'o',markersize = 2,color= 'firebrick')
sub7.plot(radius, BEM_opt_chord3 / 2,marker = 'o',markersize = 2, color= 'navy')
sub7.plot(radius, BEM_opt_chord3 / -2, marker = 'o',markersize = 2,color= 'navy')
# sub7.set_xlabel('radius (m)')
sub7.set_ylabel(r'c (m)')

sub8 = fig.add_subplot(5,3,8)
sub8.plot(radius, ideal_twist3, marker = 'o',markersize = 2,color= 'firebrick')
sub8.plot(radius, BEM_opt_twist3, marker = 'o',markersize = 2,color= 'navy')
# sub8.set_xlabel('radius (m)')
sub8.set_ylabel(r'$\theta$ (deg)')

sub9 = fig.add_subplot(5,3,9)
sub9.plot(radius, ideal_loss3, marker = 'o',markersize = 2,color= 'firebrick')
sub9.plot(radius, BEM_opt_loss3, marker = 'o',markersize = 2,color= 'navy')
# sub9.set_xlabel('radius (m)')
sub9.set_ylabel(r'dE (J/s)')

big_axes[2].set_title(r'J = 1.5, $T_{ildm} = T_{BEM \,opt.} = 717\,N$', y=1.1)
big_axes[2]._frameon = False
big_axes[2].set_yticklabels([])
big_axes[2].set_xticklabels([])


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
sub10 = fig.add_subplot(5,3,10)
sub10.plot(radius, ideal_chord4 / 2, marker = 'o',markersize = 2,color= 'firebrick')
sub10.plot(radius,  ideal_chord4 / -2, marker = 'o',markersize = 2,color= 'firebrick')
sub10.plot(radius, BEM_opt_chord4 / 2,marker = 'o',markersize = 2, color= 'navy')
sub10.plot(radius, BEM_opt_chord4 / -2, marker = 'o',markersize = 2,color= 'navy')
# sub10.set_xlabel('radius (m)')
sub10.set_ylabel(r'c (m)')

sub11 = fig.add_subplot(5,3,11)
sub11.plot(radius, ideal_twist4, marker = 'o',markersize = 2,color= 'firebrick')
sub11.plot(radius, BEM_opt_twist4, marker = 'o',markersize = 2,color= 'navy')
# sub11.set_xlabel('radius (m)')
sub11.set_ylabel(r'$\theta$ (deg)')

sub12 = fig.add_subplot(5,3,12)
sub12.plot(radius, ideal_loss4, marker = 'o',markersize = 2,color= 'firebrick')
sub12.plot(radius, BEM_opt_loss4, marker = 'o',markersize = 2,color= 'navy')
# sub12.set_xlabel('radius (m)')
sub12.set_ylabel(r'dE (J/s)')

big_axes[3].set_title(r'J = 2, $T_{ildm} = T_{BEM \,opt.} = 820\,N$', y=1.1)
big_axes[3]._frameon = False
big_axes[3].set_yticklabels([])
big_axes[3].set_xticklabels([])


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
sub13 = fig.add_subplot(5,3,13)
sub13.plot(radius, ideal_chord5 / 2, marker = 'o',markersize = 2,color= 'firebrick')
sub13.plot(radius,  ideal_chord5 / -2, marker = 'o',markersize = 2,color= 'firebrick')
sub13.plot(radius, BEM_opt_chord5 / 2,marker = 'o',markersize = 2, color= 'navy')
sub13.plot(radius, BEM_opt_chord5 / -2, marker = 'o',markersize = 2,color= 'navy')
sub13.set_xlabel('radius (m)')
sub13.set_ylabel(r'c (m)')

sub14 = fig.add_subplot(5,3,14)
sub14.plot(radius, ideal_twist5, marker = 'o',markersize = 2,color= 'firebrick')
sub14.plot(radius, BEM_opt_twist5, marker = 'o',markersize = 2,color= 'navy')
sub14.set_xlabel('radius (m)')
sub14.set_ylabel(r'$\theta$ (deg)')

sub15 = fig.add_subplot(5,3,15)
sub15.plot(radius, ideal_loss5, marker = 'o',markersize = 2,color= 'firebrick')
sub15.plot(radius, BEM_opt_loss5, marker = 'o',markersize = 2,color= 'navy')
sub15.set_xlabel('radius (m)')
sub15.set_ylabel(r'dE (J/s)')

ttl4 = big_axes[4].set_title(r'J = 2.5, $T_{ildm} = T_{BEM \,opt.} = 932\,N$', y=1.1)
# ttl4.set_position([.5, 5])
big_axes[4]._frameon = False
big_axes[4].set_yticklabels([])
big_axes[4].set_xticklabels([])


# label = big_axes[4].set_xlabel('radius (m)')
# big_axes[4].xaxis.set_label_coords(0.5,-0.5)

line_labels = [ r'Ideal-loading design method (ildm)',
                    r'BEM optimization']

fig.legend( ncol = 2, labels = line_labels,loc = 'upper center')

fig.tight_layout(rect=[0.0, 0.0, 1, 0.97])
plt.show()