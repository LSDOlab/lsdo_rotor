import numpy as np
import m3l
import csdl
from dataclasses import dataclass
from typing import Union
from lsdo_rotor.utils.atmosphere_model import AtmosphericProperties


@dataclass
class AcStates:
    """
    Container data class for aircraft states and time (time for steady cases only)
    """
    u: m3l.Variable = None
    v: m3l.Variable = None
    w: m3l.Variable = None
    p: m3l.Variable = None
    q: m3l.Variable = None
    r: m3l.Variable = None
    theta: m3l.Variable = None
    phi: m3l.Variable = None
    gamma: m3l.Variable = None
    psi: m3l.Variable = None
    x: m3l.Variable = None
    y: m3l.Variable = None
    z: m3l.Variable = None
    time: m3l.Variable = None
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




class StabilityAdapterModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('arguments', types=dict)
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('neglect', types=list, default=[])
    
    def define(self):
        args = self.parameters['arguments']
        num_nodes = self.parameters['num_nodes']
        ac_states = ['u', 'v', 'w', 'p', 'q', 'r', 'theta', 'phi']
        special_cases = self.parameters['neglect']
        for key, value in args.items():
            if key in ac_states:
                csdl_var = self.declare_variable(key, shape=(num_nodes * 9, ))
                self.register_output(name=f'{key}_exp', var=csdl_var * 1)
            elif key in special_cases:
                csdl_var = self.declare_variable(key, shape=value.shape)
                self.register_output(name=f'{key}_exp', var=csdl_var * 1)
            else:
                csdl_var = self.declare_variable(key, shape=value.shape)
                if len(value.shape) == 1 and value.shape[0] == 1:
                    # print(key, value.shape)
                    csdl_var_exp = csdl.expand(csdl_var, shape=(num_nodes * 9, ))
                    self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                elif len(value.shape) == 1 and value.shape[0] != 1:
                    # print(key, (9, ) + value.shape)
                    csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(9, ) + value.shape, indices='i->ji'), new_shape=(9, value.shape[0]))
                    self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                elif len(value.shape) == 2:
                    if num_nodes == value.shape[0]:
                        csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(9, ) + value.shape, indices='ij->kij'), new_shape=(9*num_nodes, value.shape[1]))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    elif num_nodes == value.shape[1]:
                        csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(9, ) + value.shape, indices='ij->kij'), new_shape=(9*num_nodes, value.shape[0]))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                elif len(value.shape) > 2:
                    raise NotImplementedError



class BEM(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('BEM_parameters', types=BEMParameters)
        
        self._stability_flag = False
        super().initialize(kwargs=kwargs)

    
    def compute(self) -> csdl.Model:
        from lsdo_rotor.core.BEM.BEM_model import BEMModel
        num_nodes = self.parameters['num_nodes']
        bem_parameters = self.parameters['BEM_parameters']
        if self._stability_flag:
            csdl_model = StabilityAdapterModelCSDL(
                arguments=self.arguments,
                num_nodes=num_nodes,
                neglect=['chord_profile', 'chord_cp', 'twist_profile','twist_cp', 'propeller_radius']
                
            )

            solver_model = BEMModel(
                BEM_parameters=bem_parameters,
                num_nodes=num_nodes * 9,
                operation=self,
                stability_flag=self._stability_flag,
            )

            operation_name = self.parameters['name']
            
            csdl_model.add(solver_model, operation_name, promotes=[])
            for key, value in self.arguments.items():
                csdl_model.connect(f'{key}_exp',f'{operation_name}.{key}')

        else:
            csdl_model = BEMModel(
                BEM_parameters=bem_parameters,
                num_nodes=num_nodes,
                operation=self,
            )


        return csdl_model

    def evaluate(self, ac_states : AcStates, rpm : m3l.Variable, rotor_radius : m3l.Variable, 
                 thrust_vector : m3l.Variable, thrust_origin : m3l.Variable,
                 atmosphere : AtmosphericProperties, blade_chord : Union[m3l.Variable, None] = None,
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
        self._stability_flag = ac_states.stability_flag
        num_nodes = self.parameters['num_nodes']

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


        num_radial = self.parameters['BEM_parameters'].parameters['num_radial'] 
        num_tangential = self.parameters['BEM_parameters'].parameters['num_tangential']

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
        phi = m3l.Variable(name='_phi', shape=(num_nodes, num_radial, num_tangential), operation=self)

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

        if self._stability_flag:
            F_perturbed = m3l.Variable(name='F_perturbed', shape=(8, 3), operation=self)
            M_perturbed = m3l.Variable(name='M_perturbed', shape=(8, 3), operation=self)

            bem_outputs.forces_perturbed = F_perturbed
            bem_outputs.moments_perturbed = M_perturbed

        return bem_outputs
        

class BEMParameters(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='BEM_parameters')
        self.parameters.declare('num_blades', types=int, default=3)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=1)
        self.parameters.declare('airfoil', types=str, allow_none=True)
        self.parameters.declare('airfoil_polar', default=None, types=dict, allow_none=True)

        self.parameters.declare('use_airfoil_ml', types=bool, default=False)
        self.parameters.declare('use_custom_airfoil_ml', types=bool, default=False)
        
        self.parameters.declare('mesh_units', values=['ft', 'm'], default='m')

        self.parameters.declare('ref_pt', types=np.ndarray, default=np.array([0, 0, 0]))
        self.parameters.declare('num_blades', types=int)
        # self.parameters.declare('chord_b_spline_rep',types=bool, default=False)
        # self.parameters.declare('twist_b_spline_rep',types=bool, default=False)
        self.parameters.declare('num_cp', default=4, allow_none=True)
        self.parameters.declare('b_spline_order', default=3, allow_none=True)
        self.parameters.declare('normalized_hub_radius', default=0.2)

    def assign_attributes(self):
        self.name = self.parameters['name']