from torch import nn 
import torch


# cd_reg = nn.Sequential(
#     nn.Linear(3, 104),
#     nn.ReLU(),
    
#     nn.Linear(104, 93),
#     nn.ReLU(),

#     nn.Linear(93, 40),
#     nn.ReLU(),

#     nn.Linear(40, 22),
#     nn.ReLU(),

#     nn.Linear(22, 50),
#     nn.ReLU(),

#     nn.Linear(50, 115),
#     nn.ReLU(),

#     nn.Linear(115, 115),
#     nn.ReLU(),

#     nn.Linear(115, 1), 
# )

cd_reg = nn.Sequential(
    nn.Linear(3, 104),
    nn.GELU(),
    
    nn.Linear(104, 93),
    nn.ReLU(),

    nn.Linear(93, 40),
    nn.SELU(),

    nn.Linear(40, 22),
    nn.ReLU(),

    nn.Linear(22, 50),
    nn.LeakyReLU(),

    nn.Linear(50, 115),
    nn.ReLU(),

    nn.Linear(115, 115),
    nn.GELU(),

    nn.Linear(115, 1), 
)


cd_reg.load_state_dict(torch.load(f'lsdo_rotor/airfoil/ml_trained_models/NACA_4412/NACA_4412_Cd_nn_optuna_trial_10', map_location=torch.device('cpu')))


cl_reg = nn.Sequential(
            nn.Linear(3, 82), 
            nn.ReLU(),

            nn.Linear(82, 61),
            nn.ReLU(),

            nn.Linear(61, 121), 
            nn.ReLU(),
            
            nn.Linear(121, 30), 
            nn.ReLU(),

            nn.Linear(30, 87), 
            nn.ReLU(),
            
            nn.Linear(87, 81), 
            nn.ReLU(),
            
            nn.Linear(81, 1), 
)

cl_reg.load_state_dict(torch.load(f'lsdo_rotor/airfoil/ml_trained_models/NACA_4412/NACA_4412_Cl_nn_optuna_trial_1', map_location=torch.device('cpu')))
