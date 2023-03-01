from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters
import numpy as np
from scipy.linalg import block_diag

import sympy as sym 
from sympy import *

def get_BILD_rotor_dictionary(airfoil_name, interp, custom_polar):
   
    rotor=BEMRotorParameters(
        airfoil_name=airfoil_name,
        interp=interp,
        custom_polar=custom_polar,
    )

    return rotor