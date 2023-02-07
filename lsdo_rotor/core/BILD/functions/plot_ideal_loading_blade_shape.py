import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(1,2, figsize = (12,3))


def plot_ideal_loading_blade_shape(sim):

    radius = sim['_radius'].flatten()
    local_chord = sim['_local_chord'].flatten()
    mod_local_chord = sim['_mod_local_chord'].flatten()
    local_twist_angle = sim['_local_twist_angle'].flatten()

    axs[0].plot( radius,local_chord.flatten()/2, marker = 'o', color = 'maroon', label = 'raw chord length')
    axs[0].plot( radius.flatten(),local_chord.flatten()/-2, marker = 'o', color = 'maroon')
    
    axs[0].plot( radius.flatten(),mod_local_chord.flatten()/2, marker = '*', color = 'navy', label = 'chord length with root correction')
    axs[0].plot( radius.flatten(),mod_local_chord.flatten()/-2, marker = '*', color = 'navy')
    axs[0].set_xlabel('Radius (m)')
    axs[0].set_ylabel('Blade Shape (m)')
    axs[0].legend()

    axs[1].plot(radius.flatten(), local_twist_angle.flatten()* 180 /np.pi, color = 'maroon', marker = 'o')
    axs[1].set_xlabel('Radius (m)')
    axs[1].set_ylabel('Twist Angle (deg)')
    fig.suptitle('Ideal twist and chord profile')
    plt.tight_layout( rect=[0, 0.05, 1, 1])

    plt.show()