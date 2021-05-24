from lsdo_rotor.rotor_parameters import RotorParameters

from lsdo_rotor.get_smoothing_parameters import get_smoothing_parameters
from lsdo_rotor.get_airfoil_parameters import get_airfoil_parameters


def get_parameter_dictionary(text_file, blades):
    airfoil_parameters = get_airfoil_parameters(text_file)
    smoothing_parameters = get_smoothing_parameters(airfoil_parameters[0] ,airfoil_parameters[1] , airfoil_parameters[2], airfoil_parameters[3], airfoil_parameters[4], airfoil_parameters[5], 10, airfoil_parameters[6], airfoil_parameters[7], airfoil_parameters[8], airfoil_parameters[9], airfoil_parameters[10])
    
    rotor = RotorParameters(
        a_stall_plus    = airfoil_parameters[0],
        Cl_stall_plus   = airfoil_parameters[1],
        Cd_stall_plus   = airfoil_parameters[2],
        a_stall_minus   = airfoil_parameters[3],
        Cl_stall_minus  = airfoil_parameters[4],
        Cd_stall_minus  = airfoil_parameters[5],
        Cl0             = airfoil_parameters[6],
        Cla             = airfoil_parameters[7],
        K               = airfoil_parameters[8],
        Cdmin           = airfoil_parameters[9],
        alpha_Cdmin     = airfoil_parameters[10],

        AR = 10,
        eps_plus        = smoothing_parameters[0],
        eps_minus       = smoothing_parameters[1],
        eps_cd          = smoothing_parameters[2],
        A1              = smoothing_parameters[3],
        B1              = smoothing_parameters[4],
        A2_plus         = smoothing_parameters[5],
        B2_plus         = smoothing_parameters[6],
        A2_minus        = smoothing_parameters[7],
        B2_minus        = smoothing_parameters[8],
        coeff_Cl_plus   = smoothing_parameters[9],
        coeff_Cl_minus  = smoothing_parameters[10],
        coeff_Cd_plus   = smoothing_parameters[11],
        coeff_Cd_minus  = smoothing_parameters[12],

        num_blades=blades,
    )
    return rotor
