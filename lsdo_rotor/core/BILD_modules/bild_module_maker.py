from lsdo_modules.module.module_maker import ModuleMaker
from lsdo_rotor.core.BILD_modules.bild_external_inputs_maker import BILDExternalInputsModuleMaker
from lsdo_rotor.core.BILD_modules.bild_core_inputs_maker import BILDCoreInputsMaker
from lsdo_rotor.core.BILD_modules.bild_pre_process_maker import BILDPreprocessMaker
from lsdo_rotor.utils.atmosphere_module_maker import AtmosphereMaker
import csdl
from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.BILD.functions.get_BILD_rotor_dictionary import get_BILD_rotor_dictionary
from lsdo_rotor.core.BILD.BILD_airfoil_parameters_model import BILDAirfoilParametersModel
from lsdo_rotor.core.BILD_modules.BILD_phi_bracketed_search_maker import BILDPhiBracketedSearchMaker
from lsdo_rotor.core.BILD_modules.bild_induced_velocity_maker import BILDInducedVelocityMaker
from lsdo_rotor.core.BILD_modules.bild_quartic_coeff_maker import BILDQuarticCoeffsMaker
from lsdo_rotor.core.BILD_modules.bilde_quartic_solver_maker import BILDQuarticSolverMaker
from lsdo_rotor.core.BILD_modules.bild_back_comp_maker import BILDBackCompMaker


class BILDModuleMaker(ModuleMaker):
    def initialize_module(self):
        self.parameters.declare('airfoil_polar')
        self.parameters.declare('airfoil', default='NACA_4412')
        self.parameters.declare('num_blades')
        self.parameters.declare('shape')

    def define_module(self):
        airfoil_polar = self.parameters['airfoil_polar']
        airfoil = self.parameters['airfoil']
        num_blades = self.parameters['num_blades']
        shape = self.parameters['shape']

        num_nodes = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]

        t_v = self.register_module_input('thrust_vector')
        t_o = self.register_module_input('thrust_origin')
        self.register_module_input('u')
        self.register_module_input('omega')
        self.register_module_input('z')
        self.register_module_input('reference_radius')
        self.register_module_input('reference_chord')
        self.register_module_input('propeller_radius')

        external_inputs = BILDExternalInputsModuleMaker(
            shape=shape,
            num_blades=3,
        )
        self.add_module(external_inputs, 'external_inputs')

        core_inputs = BILDCoreInputsMaker(
            shape=shape,
        )
        self.add_module(core_inputs, 'core_inputs')

        pre_process = BILDPreprocessMaker(
            shape=shape,
        )
        self.add_module(pre_process, 'pre_process')

        atmosphere = AtmosphereMaker(
            shape=(num_nodes, )
        )
        self.add_module(atmosphere, 'atmosphere')

        ref_chord = self.register_module_input('reference_chord',shape=(num_nodes,))
        Vx = self.register_module_input('u', shape=(num_nodes,))
        Vt = self.register_module_input('BILD_tangential_inflow_velocity', shape=(num_nodes,))
        W = (Vx**2 + Vt**2)**0.5
        rho = self.register_module_input('density', shape=(num_nodes,))
        mu = self.register_module_input('dynamic_viscosity', shape=(num_nodes,))
        Re = rho * W * ref_chord / mu
        self.register_module_output('Re_BILD',Re)

        interp = get_surrogate_model(airfoil, airfoil_polar)
        rotor = get_BILD_rotor_dictionary(airfoil, interp, airfoil_polar)

        BILD_parameters = csdl.custom(Re, op= BILDAirfoilParametersModel(
            rotor=rotor,
            shape=shape,
        ))
        self.register_module_output('Cl_max_BILD',BILD_parameters[0])
        self.register_module_output('Cd_min_BILD',BILD_parameters[1])
        self.register_module_output('alpha_max_LD',BILD_parameters[2])

        bracketed_search = BILDPhiBracketedSearchMaker(
                shape=shape,
                num_blades=num_blades,
        )
        self.add_module(bracketed_search, 'phi_bracketed_search')

        induced_velocity_reference = BILDInducedVelocityMaker(
            num_blades=num_blades,
            shape=shape,
        )
        self.add_module(induced_velocity_reference, name='induced_velocity_reference')

        quartic_coeff = BILDQuarticCoeffsMaker(
            shape=shape
        )
        self.add_module(quartic_coeff, 'quart_poly_coeff')

        quartic_solver = BILDQuarticSolverMaker(
            shape=shape,
        )
        self.add_module(quartic_solver, 'quart_poly_solver')

        back_comp = BILDBackCompMaker(
            shape=shape,
            num_blades=num_blades,
        )
        self.add_module(back_comp, 'back_comp')