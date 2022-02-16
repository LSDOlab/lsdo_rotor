import numpy as np
import os

from smt.surrogate_models import RMTB
from lsdo_rotor.airfoil.training_data import *


def get_surrogate_model(airfoil_name):

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

    

    return interp
