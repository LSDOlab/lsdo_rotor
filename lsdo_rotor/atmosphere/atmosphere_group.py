import omtools.api as ot
import openmdao.api as om

from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.preprocess_group import PreprocessGroup
from lsdo_rotor.core.inputs_group import InputsGroup


class AtmosphereGroup(ot.Group):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('rotor', types=RotorParameters)
        self.options.declare('mode', types=int)
        # self.options.declare('num_evaluations', types=int)
        # self.options.declare('num_radial', types=int)
        # self.options.declare('num_tangential', types=int)

    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']
        mode = self.options['mode']


        altitude    = rotor['altitude'] * 1e-3
        chord       = self.declare_input('_chord',shape=shape)
        Vx          = self.declare_input('_axial_inflow_velocity', shape=shape)
        Vt          = self.declare_input('_tangential_inflow_velocity', shape=shape)


        # Constants
        L           = 6.5
        R           = 287
        T0          = 288.16
        P0          = 101325
        g0          = 9.81
        mu0         = 1.735e-5
        S1          = 110.4


        # Temperature 
        T           = T0 - L * altitude
        
        # Pressure 
        P           = P0 * (T/T0)**(g0/L/R)
        
        # Density
        rho         = P/R/T     
        
        # Dynamic viscosity (using Sutherland's law)  
        mu          = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1)

        # Reynolds number
        W           = (Vx**2 + Vt**2)**0.5
        Re          = rho * W * chord / mu


        self.register_output('Re', Re)
        
# class AtmosphereGroup(om.ExplicitComponent):

#     def initialize(self):
#         self.options.declare('shape', types=tuple)
#         self.options.declare('rotor', types=RotorParameters)
#         self.options.declare('mode', types=int)
#         # self.options.declare('num_evaluations', types=int)
#         # self.options.declare('num_radial', types=int)
#         # self.options.declare('num_tangential', types=int)

#     def setup(self):
#         shape = self.options['shape']
#         rotor = self.options['rotor']
#         mode = self.options['mode']


#         self.add_input('_chord',shape=shape)
#         self.add_input('_axial_inflow_velocity', shape=shape)
#         self.add_input('_tangential_inflow_velocity', shape=shape)

#         self.add_output('Re', shape = shape)

#     def setup_partials(self):
#         self.declare_partials('*','*')

#     def compute(self, inputs, outputs):
#         shape = self.options['shape']
#         rotor = self.options['rotor']

#         altitude    = rotor['altitude'] * 1e-3
#         chord       = inputs['_chord']
#         Vx          = inputs['_axial_inflow_velocity']
#         Vt          = inputs['_tangential_inflow_velocity']
#         # Constants
#         L           = 6.5
#         R           = 287
#         T0          = 288.16
#         P0          = 101325
#         g0          = 9.81
#         mu0         = 1.735e-5
#         S1          = 110.4


#         # Temperature 
#         T           = T0 - L * altitude
        
#         # Pressure 
#         P           = P0 * (T/T0)**(g0/L/R)
        
#         # Density
#         rho         = P/R/T     
        
#         # Dynamic viscosity (using Sutherland's law)  
#         mu          = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1)

#         # Reynolds number
#         W           = (Vx**2 + Vt**2)**0.5
#         Re          = rho * W * chord / mu

#         outputs['Re'] = Re

        


