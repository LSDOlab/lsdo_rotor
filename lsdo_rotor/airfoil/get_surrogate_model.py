import numpy as np
import os

from smt.surrogate_models import RMTB
from lsdo_rotor.airfoil.training_data import *


def get_surrogate_model(airfoil_name, custom_polar):
    if not airfoil_name and not custom_polar:
        raise Exception('Specify airfoil name and/or a custom polar')
    elif (airfoil_name and custom_polar) or (not airfoil_name and custom_polar):
        from lsdo_rotor.airfoil.custom_airfoil_polar import CustomAirfoilPolar
        airfoil_polar = CustomAirfoilPolar(airfoil_polar=custom_polar)
        return airfoil_polar
        # raise Exception('Attempting to specify airfoil surrogate model and custom polar. Can only specify one or the other')
    
    elif airfoil_name and not custom_polar:
        smt_trained_airfoils = ['Clark_Y', 'mh117', 'NACA_0012', 'NACA_4412']
        if airfoil_name not in smt_trained_airfoils:
            raise Exception(f'Attempting to specify a pretrained airfoil surrgote model for {airfoil_name}. Available models that are trained with smt are {smt_trained_airfoils}. If you want to use a different airfoil, please provide a custom airfoil polar (based on AoA only).')

        else:
        # elif airfoil_name:    

            # = = = = = = = = = = = =  OLD  = = = = = = = = = = = =
            # this_dir = os.path.split(__file__)[0]
            # script_dir_name = os.path.dirname(__file__) 
            # file_name = 'training_data/{}_xt.txt'.format(airfoil_name)
            # file_path = os.path.join(this_dir, file_name)
            # xt = np.loadtxt(file_path)

            # file_name = 'training_data/{}_yt.txt'.format(airfoil_name)
            # file_path = os.path.join(this_dir, file_name)
            # yt = np.loadtxt(file_path)

            # dir_path = 'data_directory'
            # cashe_path = os.path.join(this_dir,dir_path)


            # xlimits = np.array([
            #         [ -np.pi / 2., np.pi / 2.  ],
            #         [0, 1.5e6/1.5e6],
            #     ])

            # interp = RMTB(num_ctrl_pts=180, xlimits=xlimits,nonlinear_maxiter=20,
            #         solver_tolerance=1e-16, energy_weight=1e-8, regularization_weight=1e-14,grad_weight = 0.5,
            #         print_global = False, print_solver = False, print_prediction=False,print_problem = False, print_training = False, data_dir = cashe_path)

            # interp.set_training_values(xt,yt)
            # interp.train()

            # = = = = = = = = = = = =  NEW  = = = = = = = = = = = =
            interp = SurrogateSingleton(airfoil_name).surrogate_models[airfoil_name]
            print('INTERP', interp)
            return interp

    # else:

class SurrogateSingleton():
    _instance = None
    surrogate_models = {}

    def __init__(self, airfoil_name):
        if airfoil_name not in self.surrogate_models:
            this_dir = os.path.split(__file__)[0]
            script_dir_name = os.path.dirname(__file__) 
            file_name = 'training_data/{}_xt.txt'.format(airfoil_name)
            file_path = os.path.join(this_dir, file_name)
            xt = np.loadtxt(file_path)

            file_name = 'training_data/{}_yt.txt'.format(airfoil_name)
            file_path = os.path.join(this_dir, file_name)
            yt = np.loadtxt(file_path)

            dir_path = 'data_directory'
            cashe_path = os.path.join(this_dir,dir_path)


            xlimits = np.array([
                    [ -np.pi / 2., np.pi / 2.  ],
                    [0, 1.5e6/1.5e6],
                ])

            interp = RMTB(num_ctrl_pts=180, xlimits=xlimits,nonlinear_maxiter=20,
                    solver_tolerance=1e-16, energy_weight=1e-8, regularization_weight=1e-14,grad_weight = 0.5,
                    print_global = False, print_solver = False, print_prediction=False,print_problem = False, print_training = False, data_dir = cashe_path)

            interp.set_training_values(xt,yt)
            interp.train()

            self.surrogate_models[airfoil_name] = interp

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)

        return cls._instance



if __name__ == '__main__':
    custom_polar = {
        'Cl_0' : 0.25,
        'Cl_alpha' : 5.1566,
        'Cd_0' : 0.01,
        'Cl_stall' : [-1, 1.5], 
        'Cd_stall' : [0.02, 0.06],
        'alpha_Cl_stall' : [-10, 15],
    }
    airfoil_model = get_surrogate_model(custom_polar=custom_polar)
    alpha = np.array([np.linspace(-np.pi/2, np.pi/2, 400)])
    Cl, Cd = airfoil_model.predict(alpha)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(np.rad2deg(alpha.flatten()), Cl)
    axs[1].plot(np.rad2deg(alpha.flatten()), Cd)

    plt.show()




# import numpy as np
# import os

# from smt.surrogate_models import RMTB
# from lsdo_rotor.airfoil.training_data import *


# def get_surrogate_model(airfoil_name, custom_polar):
#     if not airfoil_name and not custom_polar:
#         raise Exception('Specify airfoil name and/or a custom polar')
#     elif (airfoil_name and custom_polar) or (not airfoil_name and custom_polar):
#         from lsdo_rotor.airfoil.custom_airfoil_polar import CustomAirfoilPolar
#         airfoil_polar = CustomAirfoilPolar(airfoil_polar=custom_polar)
#         return airfoil_polar
#         # raise Exception('Attempting to specify airfoil surrogate model and custom polar. Can only specify one or the other')
    
#     elif airfoil_name and not custom_polar:
#         smt_trained_airfoils = ['Clark_Y', 'mh117', 'NACA_0012', 'NACA_4412']
#         if airfoil_name not in smt_trained_airfoils:
#             raise Exception(f'Attempting to specify a pretrained airfoil surrgote model for {airfoil_name}. Available models that are trained with smt are {smt_trained_airfoils}. If you want to use a different airfoil, please provide a custom airfoil polar (based on AoA only).')

#         else:
#         # elif airfoil_name:    
#             this_dir = os.path.split(__file__)[0]
#             script_dir_name = os.path.dirname(__file__) 
#             file_name = 'training_data/{}_xt.txt'.format(airfoil_name)
#             file_path = os.path.join(this_dir, file_name)
#             xt = np.loadtxt(file_path)

#             file_name = 'training_data/{}_yt.txt'.format(airfoil_name)
#             file_path = os.path.join(this_dir, file_name)
#             yt = np.loadtxt(file_path)

#             dir_path = 'data_directory'
#             cashe_path = os.path.join(this_dir,dir_path)


#             xlimits = np.array([
#                     [ -np.pi / 2., np.pi / 2.  ],
#                     [0, 1.5e6/1.5e6],
#                 ])

#             interp = RMTB(num_ctrl_pts=180, xlimits=xlimits,nonlinear_maxiter=20,
#                     solver_tolerance=1e-16, energy_weight=1e-8, regularization_weight=1e-14,grad_weight = 0.5,
#                     print_global = False, print_solver = False, print_prediction=False,print_problem = False, print_training = False, data_dir = cashe_path)

#             interp.set_training_values(xt,yt)
#             interp.train()
#             return interp

#     # else:
        


# if __name__ == '__main__':
#     custom_polar = {
#         'Cl_0' : 0.25,
#         'Cl_alpha' : 5.1566,
#         'Cd_0' : 0.01,
#         'Cl_stall' : [-1, 1.5], 
#         'Cd_stall' : [0.02, 0.06],
#         'alpha_Cl_stall' : [-10, 15],
#     }
#     airfoil_model = get_surrogate_model(custom_polar=custom_polar)
#     alpha = np.array([np.linspace(-np.pi/2, np.pi/2, 400)])
#     Cl, Cd = airfoil_model.predict(alpha)

#     import matplotlib.pyplot as plt

#     fig, axs = plt.subplots(1, 2, figsize=(10,5))
#     axs[0].plot(np.rad2deg(alpha.flatten()), Cl)
#     axs[1].plot(np.rad2deg(alpha.flatten()), Cd)

#     plt.show()

