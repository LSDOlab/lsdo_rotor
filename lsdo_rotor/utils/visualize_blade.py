from lsdo_rotor.utils.dashboard.rotor_dash import RotorDash
import numpy as np


def visualize_blade(dash):
    # rotor_dash = RotorDash()
    # sim.add_recorder(rotor_dash.get_recorder())
    dash.visualize(frame_ind=[0], show=True)
    # dash.visualize_all()


# def visualize_blade(sim):
#     chord = sim['_chord'].flatten()
#     twist = sim['_pitch'].flatten()*180/np.pi
#     radius = ddcs[self.radius].flatten()