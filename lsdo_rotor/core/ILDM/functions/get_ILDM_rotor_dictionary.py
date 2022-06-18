from lsdo_rotor.core.BEM.BEM_rotor_parameters import BEMRotorParameters
import numpy as np
from scipy.linalg import block_diag

import sympy as sym 
from sympy import *

def get_ILDM_rotor_dictionary(airfoil_name, interp):
   
    rotor=BEMRotorParameters(
        airfoil_name=airfoil_name,
        interp=interp,
    )

    return rotor