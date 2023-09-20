from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh, BEMOutputs
from pathlib import Path


_REPO_ROOT_FOLDER = Path(__file__).parents[0]
CUSTOM_ML_FOLDER = _REPO_ROOT_FOLDER / 'airfoil' / 'ml_trained_models'