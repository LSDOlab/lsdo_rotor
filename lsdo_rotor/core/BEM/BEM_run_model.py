from csdl import Model
from lsdo_rotor.core.BEM.BEM_model import BEMModel
import numpy as np 


class BEMRunModel(Model):
    def initialize(self):
        self.parameters.declare('rotor_radius')
        self.parameters.declare('rpm')
        self.parameters.declare('Vx')
        self.parameters.declare('altitude')
        self.parameters.declare('shape')
        self.parameters.declare('num_blades')
        self.parameters.declare('airfoil_name' , types=str, allow_none=True)
        self.parameters.declare('airfoil_polar', types=dict, allow_none=True)
        self.parameters.declare('chord_distribution', types=np.ndarray, allow_none=True)
        self.parameters.declare('twist_distribution', types=np.ndarray, allow_none=True)
        
        self.parameters.declare('thrust_vector', default=np.array([[1, 0, 0]]))
        self.parameters.declare('thrust_origin', default=np.array([[8.5, 5, 5]]))
        self.parameters.declare('reference_point', default=np.array([8.5, 0, 5]))
        self.parameters.declare('BILD_thrust_constraint', default=None, allow_none=True)
        self.parameters.declare('E_total_BILD', default=None)
        self.parameters.declare('BILD_chord', default=None, allow_none=True)
        self.parameters.declare('BILD_twist', default=None, allow_none=True)
        self.parameters.declare('chord_B_spline_rep', default=False, types=bool)
        self.parameters.declare('twist_B_spline_rep', default=False, types=bool)

    def define(self):
        rotor_radius = self.parameters['rotor_radius']
        rpm = self.parameters['rpm']
        Vx = self.parameters['Vx']
        altitude = self.parameters['altitude']
        shape = self.parameters['shape']
        num_nodes = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]
        thrust_vector = self.parameters['thrust_vector']
        thrust_origin = self.parameters['thrust_origin']
        reference_point = self.parameters['reference_point']
        num_blades = self.parameters['num_blades']
        airfoil_name = self.parameters['airfoil_name']
        airfoil_polar = self.parameters['airfoil_polar']

        chord = None #self.parameters['chord_distribution']
        twist = None #self.parameters['twist_distribution']
        
        T_BILD = self.parameters['BILD_thrust_constraint']
        E_total_BILD = self.parameters['E_total_BILD']
        BILD_chord = self.parameters['BILD_chord']
        BILD_twist= self.parameters['BILD_twist']

        chord_B_spline_rep = self.parameters['chord_B_spline_rep']
        twist_B_spline_rep = self.parameters['twist_B_spline_rep']


        num_cp = 10
        order = 4

        # Inputs not changing across conditions (segments)
        self.create_input(name='propeller_radius', shape=(1, ), units='m', val=rotor_radius)
        self.create_input(name='blade_number', val=num_blades)
        # self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=np.linspace(0.2,0.1,num_radial))
        # self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=np.linspace(50,10,num_radial)*np.pi/180)
        # self.create_input(name='twist_cp', shape=(num_cp,), units='rad', val=np.linspace(50, 10, num_cp)*180/np.pi) #np.array([8.60773973e-01,6.18472835e-01,3.76150609e-01,1.88136239e-01]))#np.linspace(35,10,4)*np.pi/180)
        # self.create_input(name='chord_cp', shape=(num_cp,), units='rad', val=np.linspace(0.3, 0.1, num_cp))

        if BILD_twist is not None and BILD_chord is not None and chord_B_spline_rep is True and twist_B_spline_rep is True:     
            self.create_input(name='twist_cp', shape=(num_cp,), units='rad', val=np.linspace(BILD_twist[0], BILD_twist[-1], num_cp)) #np.array([8.60773973e-01,6.18472835e-01,3.76150609e-01,1.88136239e-01]))#np.linspace(35,10,4)*np.pi/180)
            self.create_input(name='chord_cp', shape=(num_cp,), units='rad', val=np.linspace(BILD_chord[0], BILD_chord[-1], num_cp))
            self.add_design_variable('twist_cp', lower=min(BILD_twist),upper=max(BILD_twist))
            self.add_design_variable('chord_cp', lower=min(BILD_chord), upper=max(BILD_chord))
        # elif chord_B_spline_rep and not BILD_twist
        elif chord is not None and twist is not None:
            self.create_input(name='chord_profile', shape=(num_radial,), units='m', val=chord)
            self.create_input(name='twist_profile', shape=(num_radial,), units='rad', val=twist*np.pi/180)
            
        
        # self.add_design_variable('twist_profile', lower=5*np.pi/180,upper=60*np.pi/180)
        # self.add_design_variable('chord_profile', lower=0.01, upper=0.3)
        
        
        self.create_input('rpm', shape=(num_nodes, 1), units='rpm', val=rpm)
        self.create_input(name='u', shape=(num_nodes, 1), units='m/s', val=Vx)
        self.create_input(name='z', shape=(num_nodes,  1), units='m', val=altitude)


        self.add(BEMModel(   
            name='BEM_analysis',
            num_nodes=num_nodes,
            num_radial=num_radial,
            num_tangential=num_tangential,
            airfoil=airfoil_name,
            airfoil_polar=airfoil_polar,
            thrust_vector=thrust_vector,
            thrust_origin=thrust_origin,
            ref_pt=reference_point,
            num_blades=num_blades,
            chord_b_spline_rep=chord_B_spline_rep,
            twist_b_spline_rep=twist_B_spline_rep,
            num_cp=num_cp,
            b_spline_order=order,
            normalized_hub_radius=0.20,
        ), name='BEM_model')

        if T_BILD:
            T_BEM = self.declare_variable('T')    
            T_BILD_csdl = self.create_input('T_BILD', val=T_BILD)
            # self.print_var(T_BEM)
            # self.print_var(T_BILD_csdl)
            thrust_constraint = T_BEM-T_BILD_csdl
            self.register_output('thrust_constraint_optimization', thrust_constraint)
            # self.add_constraint('thrust_constraint_optimization', equals=0)
            self.add_objective('total_energy_loss', scaler=1/E_total_BILD)