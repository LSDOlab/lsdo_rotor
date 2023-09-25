from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
# from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from caddee.core.caddee_core.system_representation.component.component import Component
import numpy as np
import m3l
import csdl
from dataclasses import dataclass
import caddee.api as cd
from typing import Union


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
    moments : m3l.Variable = None
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



class BEM(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', types=Component, allow_none=True)
        self.parameters.declare('mesh', allow_none=True)#, types=BEMMesh)
        self.parameters.declare('disk_prefix', types=str)
        self.parameters.declare('disk_suffix', types=str, default=None, allow_none=True)
        self.parameters.declare('blade_prefix', types=str)
        self.parameters.declare('num_nodes', types=int)
        super().initialize(kwargs=kwargs)

    
    def compute(self) -> csdl.Model:
        from lsdo_rotor.core.BEM_caddee.BEM_model import BEMModel
        mesh = self.parameters['mesh']
        disk_prefix = self.parameters['disk_prefix']
        disk_suffix = self.parameters['disk_suffix']
        blade_prefix = self.parameters['blade_prefix']
        num_nodes = self.parameters['num_nodes']
        csdl_model = BEMModel(
            mesh=mesh,
            blade_prefix=blade_prefix,
            disk_prefix=disk_prefix,
            disk_suffix=disk_suffix,
            num_nodes=num_nodes,
            operation=self,
        )
        # csdl_model.register_module_input('thrust_vector', )
        return csdl_model

    def evaluate(self, ac_states : cd.AcStates, rpm : m3l.Variable, rotor_radius : m3l.Variable, 
                 thrust_vector : m3l.Variable, thrust_origin : m3l.Variable,
                 atmosphere : cd.AtmosphericProperties, blade_chord : Union[m3l.Variable, None] = None,
                 blade_twist : Union[m3l.Variable, None] = None, blade_chord_cp : Union[m3l.Variable, None] = None,
                 blade_twist_cp : Union[m3l.Variable, None] = None) -> BEMOutputs:
        """
        This method evaluates BEM and returns a data class with top-level analysis outputs

        Parameters
        ----------
        ac_states : AcStates
            An instance of the AcStates data class, containing the aircraft states

        rpm : m3l.Variable
            The operaing rotations per minute (rpm) of the rotor

        rotor_radius : m3l.Variable
            The radius of the rotor
        """

        if all(variable is None for variable in [blade_chord, blade_twist, blade_chord_cp, blade_twist_cp]):
            raise ValueError("Not enough information to specify blade geometry. Must specify chord/twist or chord_cp/twist_cp.")
        elif all([blade_chord, blade_chord_cp]):
            raise ValueError("Can't specifiy 'blade_chord' and 'blade_chord_cp' at the same time.")
        elif all([blade_twist, blade_twist_cp]):
            raise ValueError("Can't specifiy 'blade_twist' and 'blade_twist_cp' at the same time.")
        else:
            pass


        self.arguments = {}
        self.arguments['u'] = ac_states.u
        self.arguments['v'] = ac_states.v
        self.arguments['w'] = ac_states.w
        self.arguments['p'] = ac_states.p
        self.arguments['q'] = ac_states.q
        self.arguments['r'] = ac_states.r
        self.arguments['theta'] = ac_states.theta
        self.arguments['rpm'] = rpm

        self.arguments['density'] = atmosphere.density
        self.arguments['dynamic_viscosity'] = atmosphere.dynamic_viscosity
        self.arguments['speed_of_sound'] = atmosphere.speed_of_sound

        self.arguments['propeller_radius'] = rotor_radius
        self.arguments['thrust_vector'] = thrust_vector
        self.arguments['thrust_origin'] = thrust_origin

        if blade_chord:
            self.arguments['chord_profile'] = blade_chord
        elif blade_chord_cp:
            self.arguments['chord_cp'] = blade_chord_cp
        
        if blade_twist:
            self.arguments['twist_profile'] = blade_twist
        elif blade_twist_cp:
            self.arguments['twist_cp'] = blade_twist_cp


        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['mesh'].parameters['num_radial'] 
        num_tangential = self.parameters['mesh'].parameters['num_tangential']

            


        forces = m3l.Variable(name='F', shape=(num_nodes, 3), operation=self)
        moments = m3l.Variable(name='M', shape=(num_nodes, 3), operation=self)
        C_T = m3l.Variable(name='C_T', shape=(num_nodes, ), operation=self)
        C_Q = m3l.Variable(name='C_Q', shape=(num_nodes, ), operation=self)
        Q = m3l.Variable(name='Q', shape=(num_nodes, ), operation=self)
        T = m3l.Variable(name='T', shape=(num_nodes, ), operation=self)
        eta = m3l.Variable(name='eta', shape=(num_nodes, ), operation=self)
        FOM = m3l.Variable(name='FOM', shape=(num_nodes, ), operation=self)
        dT = m3l.Variable(name='_dT', shape=(num_nodes, num_radial, num_tangential), operation=self)
        dQ = m3l.Variable(name='_dQ', shape=(num_nodes, num_radial, num_tangential), operation=self)
        dD = m3l.Variable(name='_dD', shape=(num_nodes, num_radial, num_tangential), operation=self)
        u_x = m3l.Variable(name='_ux', shape=(num_nodes, num_radial, num_tangential), operation=self)
        phi = m3l.Variable(name='phi_distribution', shape=(num_nodes, num_radial, num_tangential), operation=self)

        bem_outputs = BEMOutputs(
            forces=forces,
            moments=moments,
            T=T,
            C_T=C_T,
            Q=Q,
            C_Q=C_Q,
            eta=eta,
            FOM=FOM,
            dT=dT,
            dQ=dQ,
            dD=dD,
            u_x=u_x,
            phi=phi,
        )


        return bem_outputs
        

class BEMMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict, default=None, allow_none=True)
        self.parameters.declare('num_blades', types=int, default=3)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=1)
        self.parameters.declare('airfoil', types=str, allow_none=True)
        self.parameters.declare('airfoil_polar', default=None, types=dict, allow_none=True)

        self.parameters.declare('use_airfoil_ml', types=bool, default=False)
        self.parameters.declare('use_custom_airfoil_ml', types=bool, default=False)
        self.parameters.declare('use_rotor_geometry', types=bool, default=True)
        
        self.parameters.declare('mesh_units', values=['ft', 'm'], default='m')

        self.parameters.declare('ref_pt', types=np.ndarray, default=np.array([0, 0, 0]))
        self.parameters.declare('num_blades', types=int)
        # self.parameters.declare('chord_b_spline_rep',types=bool, default=False)
        # self.parameters.declare('twist_b_spline_rep',types=bool, default=False)
        self.parameters.declare('num_cp', default=4, allow_none=True)
        self.parameters.declare('b_spline_order', default=3, allow_none=True)
        self.parameters.declare('normalized_hub_radius', default=0.2)
