from typing import List, Union
import m3l
from dataclasses import dataclass, field


@dataclass
class RotorMeshes:
    """
    Data class for rotor meshes
    """
    thrust_origin : m3l.Variable
    thrust_vector : m3l.Variable
    radius : m3l.Variable
    in_plane_1 : m3l.Variable
    in_plane_2 : m3l.Variable
    disk_mesh : m3l.Variable = None
    chord_profile : m3l.Variable = None
    twist_profile : m3l.Variable = None
    vlm_meshes : List[m3l.Variable] = field(default_factory=list)


@dataclass
class AcStates:
    """
    Container data class for aircraft states and time (time for steady cases only)
    """
    u: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    v: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    w: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    p: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    q: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    r: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    theta: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    phi: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    gamma: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    psi: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    x: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    y: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    z: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    time: m3l.Variable = m3l.Variable(shape=(1, ), value=0.)
    stability_flag: bool = False


@dataclass
class BEMOutputs:
    """
    Data class containing BEM outputs. All quantities are in SI units 
    unless otherwise specified.

    Parameters
    ----------
    forces : m3l.Variable
        The forces vector in the body-fixed reference frame

    moments : m3l.Variable
        The moments vector in the body-fixed reference frame

    T : m3l.Variable
        The total rotor thrust

    C_T : m3l.Variable
        The total thrust coefficient 

    Q : m3l.Variable
        The total rotor torque

    C_Q : m3l.Variable
        The total torque coefficient

    eta : m3l.Variable
        The total rotor efficiency

    FOM : m3l.Variable
        The total rotor figure of merit
    
    dT : m3l.Variable
        The sectional thrust in the span-wise direction 

    dQ : m3l.Variable
        The sectional torque in the span-wise direction

    dD : m3l.Variable
        The sectional drag in the span-wise direction 

    u_x : m3l.Variable
        The sectional axial-induced velocity 

    phi : m3l.Variable
        The sectional inflow angle

        
    """
    
    forces : m3l.Variable = None
    forces_perturbed : m3l.Variable = None
    moments : m3l.Variable = None
    moments_perturbed : m3l.Variable = None
    T : m3l.Variable = None
    C_T : m3l.Variable = None
    Q : m3l.Variable = None
    C_Q : m3l.Variable = None
    eta : m3l.Variable = None
    FOM : m3l.Variable = None
    dT : m3l.Variable = None
    dQ : m3l.Variable = None
    dD : m3l.Variable = None
    u_x : m3l.Variable = None
    phi : m3l.Variable = None