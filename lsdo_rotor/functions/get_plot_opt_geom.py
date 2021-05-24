from lsdo_utils.comps.bspline_comp import   get_bspline_mtx, BsplineComp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("darkgrid")



def get_plot_opt_geom(original_chord, optimized_chord, original_pitch, optimized_twist, radius):
    
    c_orig      = original_chord.flatten()
    c_opt       = optimized_chord.flatten()

    theta_orig  = original_pitch.flatten()
    theta_opt   = optimized_twist.flatten()

    rad         = radius.flatten()

    plt.subplots(1,2,figsize=(12,8))
    plt.subplot(1, 2, 1)
    plt.plot(rad, theta_opt , marker = 'o',color = 'maroon',label = 'Optimized twist')
    plt.plot(rad, theta_orig , marker = 'o',color = 'darkcyan',label = 'Original twist')
    plt.xlabel('Radius (m)',fontsize=19)
    plt.ylabel(r'Twist Angle ($^{\circ}$)',fontsize=19)
    plt.legend(fontsize=19)
    # plt.title('pitch')

    plt.subplot(1, 2, 2)
    plt.plot(rad,c_opt/2, marker = 'o',color = 'maroon',label = 'Optimized chord')
    plt.plot(rad,c_opt/-2, marker = 'o',color = 'maroon')

    plt.plot(rad,c_orig/2, marker = 'o',color = 'darkcyan',label = 'Original chord')
    plt.plot(rad,c_orig/-2, marker = 'o',color = 'darkcyan')

    # plt.ylim([-0.4,0.4])

    plt.xlabel('Radius (m)',fontsize=19)
    plt.ylabel('Blade shape',fontsize=19)
    plt.legend(fontsize=19)
    # plt.title('chord')


    # plt.suptitle('SNOPT Optimization for Arbitrary Propeller' + '\n' + r'min. Q w.r.t. to c, $\theta$, RPM, R' + '\n' + r'subject to T = 1500, $\eta \leq 1$')
    plt.tight_layout(rect = [0,0.05,1,0.99])
    plt.show()


