import numpy as np
from csdl import Model
import m3l
from dataclasses import dataclass


def get_atmosphere(altitude : m3l.Variable):
    atmos = Atmosphere(
        name='rotor_atmosphere'
    )

    atmos_properties = atmos.evaluate(altitude=altitude)

    return atmos_properties


@dataclass
class AtmosphericProperties:
    """
    Container data class for atmospheric variables 
    """
    density: m3l.Variable = None
    temperature: m3l.Variable = None
    pressure: m3l.Variable = None
    dynamic_viscosity: m3l.Variable = None
    speed_of_sound: m3l.Variable = None


class Atmosphere(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.num_nodes = 1
        super().initialize(kwargs=kwargs)

    def compute(self):
        csdl_model = AtmosphereModel(
            atmosphere_model=self,
        )

        return csdl_model

    def evaluate(self, altitude : m3l.Variable) -> AtmosphericProperties:
        """
        Returns data class containing atmospheric properties.


        Parameters
        ----------
        altitude : m3l Variable
            The altitude at which the atmospheric properties are to be evaluated
        """

        name = self.parameters['name']

        self.arguments = {}
        self.arguments['altitude'] = altitude


        rho = m3l.Variable(name='density', shape=(self.num_nodes, ), operation=self)
        mu = m3l.Variable(name='dynamic_viscosity', shape=(self.num_nodes, ), operation=self)
        pressure = m3l.Variable(name='pressure', shape=(self.num_nodes, ), operation=self)

        a = m3l.Variable(name='speed_of_sound', shape=(self.num_nodes, ), operation=self)
        temp = m3l.Variable(name='temperature', shape=(self.num_nodes, ), operation=self)

        atmosphere = AtmosphericProperties(
            density=rho,
            dynamic_viscosity=mu,
            pressure=pressure,
            speed_of_sound=a,
            temperature=temp,
        )
        
        return atmosphere


class AtmosphereModel(Model):
    def initialize(self):
        self.parameters.declare('atmosphere_model', types=Atmosphere)
    
    def define(self):
        atmosphere = self.parameters['atmosphere_model']
        num_nodes = atmosphere.num_nodes

        h = self.declare_variable('altitude',shape=(num_nodes, ), val=0) * 1e-3 # value in meters; then convert to km
        L = 6.5 # K/km
        R = 287
        T0 = 288.16
        P0 = 101325
        g0 = 9.81
        mu0 = 1.735e-5
        S1 = 110.4
        gamma = 1.4

        # Temperature 
        T           =  - h * L + T0

        # Pressure 
        P           = P0 * (T/T0)**(g0/(L * 1e-3)/R)
        
        # Density
        rho         = P/R/T
        
        # Dynamic viscosity (using Sutherland's law)  
        mu          = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1)

        # speed of sound 
        a = (gamma * R * T)**0.5

        self.register_output('temperature',T)
        self.register_output('pressure',P)
        self.register_output('density',rho)
        self.register_output('dynamic_viscosity',mu)
        self.register_output('speed_of_sound', a)

