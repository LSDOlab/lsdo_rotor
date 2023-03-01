from lsdo_rotor.utils.rotor_dash import RotorDash


def visualize_blade(dash):
    # rotor_dash = RotorDash()
    # sim.add_recorder(rotor_dash.get_recorder())
    dash.visualize(frame_ind=[0], show=True)
    # dash.visualize_all()