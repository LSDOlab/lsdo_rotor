from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters
import numpy as np


def get_BEM_rotor_dictionary(airfoil_name, interp=None, custom_polar=None, ml_cl=None, ml_cd=None, use_airfoil_ml=False):
   
    rotor=BEMRotorParameters(
        airfoil_name=airfoil_name,
        interp=interp,
        custom_polar=custom_polar,
        cl_ml_model=ml_cl,
        cd_ml_model=ml_cd,
        use_airfoil_ml=use_airfoil_ml,
    )

    return rotor