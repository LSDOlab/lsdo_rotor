import importlib
from dataclasses import dataclass


@dataclass
class NeuralNets:
    Cd=None,
    Cl=None,


def get_airfoil_models(airfoil) -> NeuralNets:
    module_name = f'lsdo_rotor.core.airfoil.ml_trained_models.{airfoil}.neural_nets'
    
    module = importlib.import_module(module_name)
    
    neural_nets = NeuralNets
    neural_nets.Cd = module.cd_reg
    neural_nets.Cl = module.cl_reg

    return neural_nets
    

