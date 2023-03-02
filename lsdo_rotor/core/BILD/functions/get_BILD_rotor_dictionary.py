from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters


def get_BILD_rotor_dictionary(airfoil_name, interp, custom_polar):
   
    rotor=BEMRotorParameters(
        airfoil_name=airfoil_name,
        interp=interp,
        custom_polar=custom_polar,
    )

    return rotor