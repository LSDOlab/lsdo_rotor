from csdl import Model
from lsdo_rotor.core.BILD.BILD_model import BILDModel
import numpy as np 


class BILDRunModel(Model):
    def initialize(self):
        self.parameters.declare('rotor_radius')
        self.parameters.declare('reference_chord')
        self.parameters.declare('reference_radius')
        self.parameters.declare('rpm')
        self.parameters.declare('Vx')
        self.parameters.declare('reference_radius')
        self.parameters.declare('altitude')
        self.parameters.declare('shape')
        self.parameters.declare('num_blades')
        self.parameters.declare('airfoil_name' , types=str, allow_none=True)
        self.parameters.declare('airfoil_polar', types=dict, allow_none=True)

        self.parameters.declare('thrust_vector', default=np.array([[1, 0, 0]]))
        self.parameters.declare('thrust_origin', default=np.array([[8.5, 5, 5]]))
        self.parameters.declare('reference_point', default=np.array([8.5, 0, 5]))
        
    
    def define(self):
        rotor_radius = self.parameters['rotor_radius']
        reference_chord = self.parameters['reference_chord']
        reference_radius = self.parameters['reference_radius']
        rotor_radius = self.parameters['rotor_radius']
        rpm = self.parameters['rpm']
        Vx = self.parameters['Vx']
        altitude = self.parameters['altitude']
        shape = self.parameters['shape']
        num_nodes = shape[0]
        num_radial = shape[1]
        num_tangential = shape[2]
        
        airfoil_name = self.parameters['airfoil_name']
        airfoil_polar = self.parameters['airfoil_polar']
        
        thrust_vector = self.parameters['thrust_vector']
        thrust_origin = self.parameters['thrust_origin']
        reference_point = self.parameters['reference_point']
        num_blades = self.parameters['num_blades']

        self.create_input(name='blade_number', val=num_blades)
        self.create_input(name='propeller_radius', shape=(num_nodes, ), val=rotor_radius)
        self.create_input(name='reference_chord', shape=(num_nodes, ), val=reference_chord)
        self.create_input(name='reference_radius', shape=(num_nodes, ), val=reference_radius)

        self.create_input('omega', shape=(num_nodes, ), units='rpm', val=rpm)
        self.create_input(name='u', shape=(num_nodes, ), units='m/s', val=Vx)
        self.create_input(name='z', shape=(num_nodes, ), units='m', val=altitude)
                
        self.add(BILDModel(   
            name='blade_design',
            num_nodes=num_nodes,
            num_radial=num_radial,
            num_tangential=num_tangential,
            airfoil=airfoil_name,
            airfoil_polar=airfoil_polar,
            thrust_vector=thrust_vector,
            thrust_origin=thrust_origin,
            ref_pt=reference_point,
            num_blades=num_blades,
        ), name='BILD_model')