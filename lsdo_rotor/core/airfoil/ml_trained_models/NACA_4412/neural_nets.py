from torch import nn 
import torch
from lsdo_rotor import CUSTOM_ML_FOLDER


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


cd_reg.load_state_dict(torch.load(CUSTOM_ML_FOLDER / 'NACA_4412/Cd_neural_net', map_location=torch.device('cpu')))


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

cl_reg.load_state_dict(torch.load(CUSTOM_ML_FOLDER / 'NACA_4412/Cl_neural_net', map_location=torch.device('cpu')))
