
import numpy as np
import m3l
import csdl
from lsdo_rotor.utils.atmosphere_model import AtmosphericProperties
from lsdo_rotor.utils.helper_classes import RotorMeshes, AcStates, PittPetersOutputs
from typing import Union, List
from lsdo_rotor import BEMParameters, BEM


class StabilityAdapterModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('arguments', types=dict)
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('neglect', types=list, default=[])
    
    def define(self):
        args = self.parameters['arguments']
        num_nodes = self.parameters['num_nodes']
        ac_states = ['u', 'v', 'w', 'p', 'q', 'r', 'theta', 'phi', 'psi', 'x', 'y', 'z']
        special_cases = self.parameters['neglect']
        for key, value in args.items():
            if key in ac_states:
                csdl_var = self.declare_variable(key, shape=(num_nodes * 13, ))
                self.register_output(name=f'{key}_exp', var=csdl_var * 1)
            elif key in special_cases:
                csdl_var = self.declare_variable(key, shape=value.shape)
                self.register_output(name=f'{key}_exp', var=csdl_var * 1)
            else:
                csdl_var = self.declare_variable(key, shape=value.shape)
                if len(value.shape) == 1 and value.shape[0] == 1:
                    # print(key, value.shape)
                    csdl_var_exp = csdl.expand(csdl_var, shape=(num_nodes * 13, ))
                    self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                elif len(value.shape) == 1 and value.shape[0] != 1:
                    # print(key, (13, ) + value.shape)
                    csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(13, ) + value.shape, indices='i->ji'), new_shape=(13, value.shape[0]))
                    self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                elif len(value.shape) == 2:
                    if num_nodes == value.shape[0]:
                        csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(13, ) + value.shape, indices='ij->kij'), new_shape=(13*num_nodes, value.shape[1]))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    elif num_nodes == value.shape[1]:
                        csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(13, ) + value.shape, indices='ij->kij'), new_shape=(13*num_nodes, value.shape[0]))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                elif len(value.shape) > 2:
                    raise NotImplementedError

class PittPeters(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('num_nodes', types=int, default=1)
        self.parameters.declare('pitt_peters_parameters', types=PittPetersParameters)
        self.parameters.declare('rotation_direction', values=['cw', 'ccw', 'ignore'], default='cw', allow_none=True)
        self._stability_flag = False

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self) -> csdl.Model:
        from lsdo_rotor.core.pitt_peters.pitt_peters_model import PittPetersModel
        num_nodes = self.parameters['num_nodes']
        pitt_peters_parameters = self.parameters['pitt_peters_parameters']
        rotation_direction = self.parameters['rotation_direction']
        
        if self._stability_flag:
            csdl_model = StabilityAdapterModelCSDL(
                arguments=self.arguments,
                num_nodes=num_nodes,
                neglect=['chord_dist', 'chord_cp', 'twist_profile','twist_cp', 'R', 'reference_point']
                
            )
            solver_model = PittPetersModel(
                pitt_peters_parameters=pitt_peters_parameters,
                num_nodes=num_nodes * 13,
                operation=self,
                stability_flag=self._stability_flag,
                rotation_direction=rotation_direction,
            )

            operation_name = self.parameters['name']
            
            csdl_model.add(solver_model, operation_name, promotes=[])
            for key, value in self.arguments.items():
                csdl_model.connect(f'{key}_exp',f'{operation_name}.{key}')

        else:
            csdl_model = PittPetersModel(
                pitt_peters_parameters=pitt_peters_parameters,
                num_nodes=num_nodes,
                operation=self,
                rotation_direction=rotation_direction,
            )

        return csdl_model

    def evaluate(self, ac_states : AcStates, rpm : m3l.Variable, rotor_radius : m3l.Variable, 
                 thrust_vector : m3l.Variable, thrust_origin : m3l.Variable,
                 atmosphere : AtmosphericProperties, blade_chord : Union[m3l.Variable, None] = None,
                 blade_twist : Union[m3l.Variable, None] = None, blade_chord_cp : Union[m3l.Variable, None] = None,
                 blade_twist_cp : Union[m3l.Variable, None] = None, reference_point : m3l.Variable=None,
                 in_plane_1 : Union[m3l.Variable, None]=None, in_plane_2 : Union[m3l.Variable, None]=None, cg_vec : m3l.Variable=None) -> PittPetersOutputs:
        
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
        self.arguments['phi'] = ac_states.phi
        self.arguments['psi'] = ac_states.psi
        
        self.arguments['rpm'] = rpm

        self.arguments['density'] = atmosphere.density
        self.arguments['dynamic_viscosity'] = atmosphere.dynamic_viscosity
        self.arguments['speed_of_sound'] = atmosphere.speed_of_sound

        self.arguments['R'] = rotor_radius
        self.arguments['tv'] = thrust_vector
        self.arguments['to'] = thrust_origin
        
        if cg_vec == None:
            self.arguments['eval_point'] = m3l.Variable(shape=(3, ), value=np.array([0., 0., 0.,]))
        else:
            self.arguments['eval_point'] = cg_vec

        if reference_point == None:
            self.arguments['reference_point'] = m3l.Variable(shape=(3, ), value=np.array([0., 0., 0.,]))
        else:
            self.arguments['reference_point'] = reference_point

        if blade_chord:
            self.arguments['chord_dist'] = blade_chord
        elif blade_chord_cp:
            self.arguments['chord_cp'] = blade_chord_cp
        
        if blade_twist:
            self.arguments['twist_profile'] = blade_twist
        elif blade_twist_cp:
            self.arguments['twist_cp'] = blade_twist_cp

        if in_plane_1 is None:
            self.arguments['in_plane_1'] = m3l.Variable(shape=(3, ), name='in_plane_1', value=np.array([1., 0., 0.]))
        else:
            self.arguments['in_plane_1'] = in_plane_1

        if in_plane_2 is None:
            self.arguments['in_plane_2'] = m3l.Variable(shape=(3, ), name='in_plane_2', value=np.array([0., 1., 0.]))
        else:
            self.arguments['in_plane_2'] = in_plane_2

        num_radial = self.parameters['pitt_peters_parameters'].parameters['num_radial'] 
        num_tangential = self.parameters['pitt_peters_parameters'].parameters['num_tangential'] 

        if self._stability_flag:
            num_nodes *= 13
            forces = m3l.Variable(name=f'{self.name}.F', shape=(num_nodes, 3), operation=self)
            moments = m3l.Variable(name=f'{self.name}.M', shape=(num_nodes, 3), operation=self)
            # F_perturbed = m3l.Variable(name=f'{self.name}.F_perturbed', shape=(num_nodes, 3), operation=self)
            # M_perturbed = m3l.Variable(name=f'{self.name}.M_perturbed', shape=(num_nodes, 3), operation=self)
            Q = m3l.Variable(name=f'{self.name}.Q', shape=(num_nodes,), operation=self)
            T = m3l.Variable(name=f'{self.name}.T', shape=(num_nodes,), operation=self)
            C_T = m3l.Variable(name=f'{self.name}.C_T', shape=(num_nodes,), operation=self)
            C_Q = m3l.Variable(name=f'{self.name}.C_Q', shape=(num_nodes,), operation=self)
            dT = m3l.Variable(name=f'{self.name}._dT', shape=(num_nodes, num_radial, num_tangential), operation=self)
            dQ = m3l.Variable(name=f'{self.name}._dQ', shape=(num_nodes, num_radial, num_tangential), operation=self)
            dD = m3l.Variable(name=f'{self.name}._dD', shape=(num_nodes, num_radial, num_tangential), operation=self)
            u_x = m3l.Variable(name=f'{self.name}._ux', shape=(num_nodes, num_radial, num_tangential), operation=self)
            phi = m3l.Variable(name=f'{self.name}._phi', shape=(num_nodes, num_radial, num_tangential), operation=self)
            eta = m3l.Variable(name=f'{self.name}.eta', shape=(num_nodes, ), operation=self)
            FOM = m3l.Variable(name=f'{self.name}.FOM', shape=(num_nodes, ), operation=self)
        
            pitt_peters_outputs = PittPetersOutputs(
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
                # forces_perturbed = F_perturbed,
                # moments_perturbed = M_perturbed,
            )

            if blade_chord_cp is not None:
                chord_profile = m3l.Variable(name=f'{self.name}.chord_profile', shape=(num_radial, ), operation=self)
                pitt_peters_outputs._chord_profile = chord_profile

            if blade_twist_cp is not None:
                twist_profile = m3l.Variable(name=f'{self.name}.twist_profile', shape=(num_radial, ), operation=self)
                pitt_peters_outputs._twist_profile = twist_profile
            

        else:
            forces = m3l.Variable(name='F', shape=(num_nodes, 3), operation=self)
            moments = m3l.Variable(name='M', shape=(num_nodes, 3), operation=self)
            Q = m3l.Variable(name='Q', shape=(num_nodes,), operation=self)
            T = m3l.Variable(name='T', shape=(num_nodes,), operation=self)
            C_T = m3l.Variable(name='C_T', shape=(num_nodes,), operation=self)
            C_Q = m3l.Variable(name='C_Q', shape=(num_nodes,), operation=self)
            dT = m3l.Variable(name='_dT', shape=(num_nodes, num_radial, num_tangential), operation=self)
            dQ = m3l.Variable(name='_dQ', shape=(num_nodes, num_radial, num_tangential), operation=self)
            dD = m3l.Variable(name='_dD', shape=(num_nodes, num_radial, num_tangential), operation=self)
            u_x = m3l.Variable(name='_ux', shape=(num_nodes, num_radial, num_tangential), operation=self)
            phi = m3l.Variable(name='_phi', shape=(num_nodes, num_radial, num_tangential), operation=self)
            eta = m3l.Variable(name='eta', shape=(num_nodes, ), operation=self)
            FOM = m3l.Variable(name='FOM', shape=(num_nodes, ), operation=self)
        
            pitt_peters_outputs = PittPetersOutputs(
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

            if blade_chord_cp is not None:
                chord_profile = m3l.Variable(name=f'chord_profile', shape=(num_radial, ), operation=self)
                pitt_peters_outputs._chord_profile = chord_profile

            if blade_twist_cp is not None:
                twist_profile = m3l.Variable(name=f'twist_profile', shape=(num_radial, ), operation=self)
                pitt_peters_outputs._twist_profile = twist_profile

        return pitt_peters_outputs


class PittPetersParameters(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='pitt_peters_parameters')
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


def check_same_length(*lists):
    return all(len(lst) == len(lists[0]) for lst in lists)

def are_objects_of_same_type(*objects):
    return all(isinstance(obj, type(objects[0])) for obj in objects)

def check_required_string(required_strings, given_strings):
    return all(entry in required_strings for entry in given_strings)

def evaluate_multiple_pitt_peters_models(
        name_prefix : str,
        pitt_peters_parameters : List[PittPetersParameters],
        pitt_peters_mesh_list : List[RotorMeshes],
        rpm_list : List[m3l.Variable],
        ac_states : AcStates,
        atmoshpere : AtmosphericProperties,
        num_nodes : int = 1, 
        m3l_model : m3l.Model=None,
        rotation_direction_list : List[str]=None,
        chord_cp:bool=False,
        twist_cp:bool=False,
) -> List[PittPetersOutputs]:
    """
    Helper function to create multiple BEM instances at once
    """

    if rotation_direction_list is not None:
        if not are_objects_of_same_type(pitt_peters_mesh_list, rpm_list, rotation_direction_list):
            raise TypeError("pitt_peters_mesh_list, rpm_list, and rotation_direction_list must all be lists")
        if not check_same_length(pitt_peters_mesh_list, rpm_list, rotation_direction_list):    
            raise ValueError("pitt_peters_mesh_list, rpm_list, and rotation_direction_list must all have the same length")
        if not check_required_string(required_strings=['cw', 'ccw', 'ignore'], given_strings=rotation_direction_list):
            raise ValueError("rotation_direction can only be 'cw', 'ccw', or 'ignore'")
    
    else:
        if not are_objects_of_same_type(pitt_peters_mesh_list, rpm_list):
            print(type(rpm_list))
            print(type(pitt_peters_mesh_list))
            print(pitt_peters_mesh_list)
            print(rpm_list)
            raise TypeError("pitt_peters_mesh_list, rpm_list must all be lists")
        if not check_same_length(pitt_peters_mesh_list, rpm_list):    
            raise ValueError("pitt_peters_mesh_list, rpm_list must all have the same length")

    
    
    num_instances = len(rpm_list)

    if not isinstance(pitt_peters_parameters, list):
        pitt_peters_parameters = [pitt_peters_parameters] * num_instances

    if rotation_direction_list is None:
        rotation_direction_list = ['ignore'] * num_instances

    pitt_peters_output_list = []
    for i in range(num_instances):
        parameters = pitt_peters_parameters[i]
        if isinstance(parameters, PittPetersParameters):
            pitt_peters_instance = PittPeters(
                name=f'{name_prefix}_{i}',
                pitt_peters_parameters=pitt_peters_parameters[i],
                num_nodes=num_nodes,
                rotation_direction=rotation_direction_list[i]
            )

            pitt_peters_mesh = pitt_peters_mesh_list[i]
            rpm = rpm_list[i]
            thrust_vector = pitt_peters_mesh.thrust_vector
            thrust_origin = pitt_peters_mesh.thrust_origin
            chord_profile = pitt_peters_mesh.chord_profile
            twist_profile = pitt_peters_mesh.twist_profile
            in_plane_1 = pitt_peters_mesh.in_plane_2*1
            in_plane_2 = pitt_peters_mesh.in_plane_1*1

            if (chord_cp is True) and (twist_cp is True):
                chord_cps = pitt_peters_mesh.chord_cps
                twist_cps = pitt_peters_mesh.twist_cps
                pitt_peters_outputs = pitt_peters_instance.evaluate(ac_states=ac_states, rpm=rpm, atmosphere=atmoshpere, 
                                        thrust_origin=thrust_origin, thrust_vector=thrust_vector, rotor_radius=pitt_peters_mesh.radius,
                                        blade_chord_cp=chord_cps, blade_twist_cp=twist_cps, in_plane_1=in_plane_1, in_plane_2=in_plane_2)
            
            elif (chord_cp is Flase) and (twist_cp is False):
                pitt_peters_outputs = pitt_peters_instance.evaluate(ac_states=ac_states, rpm=rpm, atmosphere=atmoshpere, 
                                        thrust_origin=thrust_origin, thrust_vector=thrust_vector, rotor_radius=pitt_peters_mesh.radius,
                                        blade_chord=chord_profile, blade_twist=twist_profile, in_plane_1=in_plane_1, in_plane_2=in_plane_2)
            else: 
                raise NotImplementedError
            
            if m3l_model is not None:
                m3l_model.register_output(pitt_peters_outputs)
            
            pitt_peters_output_list.append(pitt_peters_outputs)
        
        elif isinstance(parameters, BEMParameters):
            bem_instance = BEM(
                name=f'{name_prefix}_{i}',
                BEM_parameters=pitt_peters_parameters[i],
                num_nodes=num_nodes,
                rotation_direction=rotation_direction_list[i],
            )

            bem_mesh = pitt_peters_mesh_list[i]
            rpm = rpm_list[i]
            thrust_vector = bem_mesh.thrust_vector
            thrust_origin = bem_mesh.thrust_origin
            

            if (chord_cp is True) and (twist_cp is True):
                chord_cps = bem_mesh.chord_cps
                twist_cps = bem_mesh.twist_cps
                bem_outputs = bem_instance.evaluate(ac_states=ac_states, rpm=rpm, atmosphere=atmoshpere, 
                                        thrust_origin=thrust_origin, thrust_vector=thrust_vector, rotor_radius=bem_mesh.radius,
                                        blade_chord_cp=chord_cps, blade_twist_cp=twist_cps)
            elif (chord_cp is Flase) and (twist_cp is False):
                chord_profile = bem_mesh.chord_profile
                twist_profile = bem_mesh.twist_profile
                bem_outputs = bem_instance.evaluate(ac_states=ac_states, rpm=rpm, atmosphere=atmoshpere, 
                                        thrust_origin=thrust_origin, thrust_vector=thrust_vector, rotor_radius=bem_mesh.radius,
                                        blade_chord=chord_profile, blade_twist=twist_profile)
            else: 
                raise NotImplementedError
            
            if m3l_model is not None:
                m3l_model.register_output(bem_outputs)
            
            pitt_peters_output_list.append(bem_outputs)

        else:
            raise NotImplementedError

    return pitt_peters_output_list
