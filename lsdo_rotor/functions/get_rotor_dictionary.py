from lsdo_rotor.rotor_parameters import RotorParameters


def get_rotor_dictionary(airfoil_name, blades, altitude, mode, interp, ideal_alpha_ref_chord, ideal_Cl_ref_chord, ideal_Cd_ref_chord, c_ref):
    rotor = RotorParameters(
        airfoil_name = airfoil_name,
        num_blades=blades,
        altitude = altitude,
        mode = mode,
        interp = interp,
        ideal_alpha_ref_chord = ideal_alpha_ref_chord,
        ideal_Cl_ref_chord = ideal_Cl_ref_chord,
        ideal_Cd_ref_chord = ideal_Cd_ref_chord,
        c_ref =  c_ref,
    )

    return rotor