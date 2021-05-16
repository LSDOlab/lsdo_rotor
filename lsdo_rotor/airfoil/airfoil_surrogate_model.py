import numpy as np

import omtools.api as ot
import openmdao.api as om

from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.atmosphere.atmosphere_group import AtmosphereGroup

airfoil_name = 'full_training'
# airfoil_name = 'mh117'

interp = get_surrogate_model(airfoil_name)

class AirfoilSurrogateModel(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types = tuple)
        self.options.declare('rotor', types = RotorParameters)
    
    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']

        self.add_input('_alpha', shape = shape)
        self.add_input('Re', shape = shape)

        self.add_output('_Cl', shape = shape)
        self.add_output('_Cd', shape = shape)


    def setup_partials(self):
        self.declare_partials('*','*')

    def compute(self, inputs, outputs):
        shape       = self.options['shape']
        rotor       = self.options['rotor']

        alpha       = inputs['_alpha'].flatten()
        # print(alpha)
        Re          = inputs['Re'].flatten()
        # print(Re)
        # print('Reynolds Number',Re)

        size = len(alpha)
        Cl = np.empty((size))
        Cd = np.empty((size))


        for i in range(size):
            x       = np.array([alpha[i], Re[i]/1.5e6])
            # print('Reynolds number',Re[i])
            x       = x.reshape((1,2))

            # x2       = x.reshape((1,2))

            y       = interp.predict_values(x)
            # print(y)
            # print(y[0])
            # print(y[1])
            Cl[i]   = y[0][0]
            Cd[i]   = y[0][1]

        # print(Cl)
        # print(Cd)

        outputs['_Cl'] = Cl.reshape(shape)
        outputs['_Cd'] = Cd.reshape(shape)

    def compute_partials(self, inputs, partials):
        alpha       = inputs['_alpha'].flatten()
        Re          = inputs['Re'].flatten()


        size = len(alpha)

        for i in range(size):
            x       = np.array([alpha[i], Re[i]/1.5e6])
            # print('Reynolds number',Re[i])
            x       = x.reshape((1,2))



            dy_dalpha = interp.predict_derivatives(x, 0)
            dy_dRe = interp.predict_derivatives(x, 1)

            partials['_Cl', '_alpha'] = dy_dalpha[:, 0]
            partials['_Cd', '_alpha'] = dy_dalpha[:, 1]

            partials['_Cl', 'Re'] = dy_dRe[:, 0]
            partials['_Cd', 'Re'] = dy_dRe[:, 1]


        
             

