import numpy as np
import vedo
from lsdo_rotor.utils.vedo_interface import CaddeeVedoContainer
from lsdo_rotor.utils.base_dash import BaseDash
from lsdo_rotor.utils.plot_blade import get_blade_geo


class RotorDash(BaseDash):
    def setup(self):
        
        self.set_clientID('simulator')

        # Define frames
        height = 6.
        width = 8.

        self.add_frame('1',
            height_in=height,
            width_in=width*2,
            ncols=12,
            nrows=10,
            wspace=0.0,
            hspace=0.0)

        self.add_frame('2',
            height_in=height,
            width_in=height,
            ncols=1,
            nrows=1,
            wspace=0.5,
            hspace=0.5)
    
        self.chord = '_chord'
        self.twist = '_pitch'
        self.radius = '_radius'
        self.num_blades = 'blade_number'

        self.save_variable(self.chord)
        self.save_variable(self.twist)
        self.save_variable(self.radius)
        self.save_variable(self.num_blades)

    def plot(self,
             frames,
             data_dict_current,
             data_dict_history,
             limits_dict,
             video=False):

        ddcs = data_dict_current['simulator']

        # sim = self.sim
        # shape = sim['_chord'].shape
        # if shape[0] > 1:
        #     raise Exception('Plotting blades for cases when num_nodes>1 not implemented')
        chord = ddcs[self.chord].flatten()
        twist = ddcs[self.twist].flatten()
        radius = ddcs[self.radius].flatten()
        num_blades = int(ddcs[self.num_blades].flatten())

        frame = frames['2']
        ax_3d_fli = frame[0, 0]
        
        blade_plot = CaddeeVedoContainer()
        blades = get_blade_geo(radius, twist, chord, num_blades)
        blade_color = '#2ca02c'
        for i_blade in range(num_blades):
            # print(blades[i_blade])
            blade_plot.add_mesh_surface(blades[i_blade]*3.28, c=blade_color, add_to_plotter=True)
        camera_settings = {
            'pos': (0.0, 0.0, 16.0),
            'viewup': (0, 1, 0),
            'focalPoint': (0.0, 0.0, 0)
        }
        blade_plot.draw_to_axes(ax_3d_fli, camera=camera_settings, size=(2800, 2300))

        frame = frames['1']
        chord_frame = frame[:, 0:5]
        chord_frame.plot(radius, chord/-2, color='#d62728')
        chord_frame.plot(radius, chord/2, color='#d62728')
        chord_frame.axis('equal')
        chord_frame.set_xlabel('radius (m)')
        chord_frame.set_ylabel('blade shape (m)')

        twist_frame = frame[:, 7:12]
        twist_frame.plot(radius, twist * 180/np.pi, color='#d62728')
        twist_frame.set_xlabel('radius (m)')
        twist_frame.set_ylabel('blade twist (deg)')

        frame.fig.tight_layout()
        frame.write()


