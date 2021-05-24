import numpy as np

import omtools.api as ot
import openmdao.api as om

from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.atmosphere.atmosphere_group import AtmosphereGroup

with open('airfoil.txt', 'r') as file:
    airfoil_name = file.read().replace('\n', '')


# airfoil_name = 'NACA_4412'
interp = get_surrogate_model(airfoil_name)


class AirfoilSurrogateModel(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types = tuple)
        self.options.declare('rotor', types = RotorParameters)
    
    def setup(self):
        shape = self.options['shape']
        # print(shape)
        rotor = self.options['rotor']
        airfoil = rotor['airfoil_name']
        # num_evaluations = self.options['num_evaluations']
        # num_radial = self.options['num_radial']
        # num_tangential = self.options['num_tangential']
        shape = (shape[0], shape[1], shape[2])

        self.add_input('_alpha', shape = shape)
        self.add_input('Re', shape = shape)

        self.add_output('_Cl', shape = shape)
        self.add_output('_Cd', shape = shape)

        indices = np.arange(shape[0] * shape[1] * shape[2])
        self.declare_partials('_Cl', 'Re', rows = indices, cols = indices)
        self.declare_partials('_Cl', '_alpha', rows = indices, cols = indices)
        self.declare_partials('_Cd', 'Re', rows = indices, cols = indices)
        self.declare_partials('_Cd', '_alpha', rows = indices, cols = indices)

        self.x = np.zeros((shape[0] * shape[1] * shape[2], 2))

    def compute(self, inputs, outputs):
        shape       = self.options['shape']
        rotor       = self.options['rotor']
        airfoil     = rotor['airfoil_name']
        # num_evaluations = self.options['num_evaluations']
        # num_radial = self.options['num_radial']
        # num_tangential = self.options['num_tangential']
    
        alpha       = inputs['_alpha'].flatten()
        # print(alpha)
        Re          = inputs['Re'].flatten()
        # print(Re)
        # print('Reynolds Number',Re)
        self.x[:, 0] = alpha
        self.x[:, 1] = Re/1e6

        # interp = get_surrogate_model(airfoil)
        y = interp.predict_values(self.x).reshape((shape[0] , shape[1] , shape[2], 2))

        outputs['_Cl'] = y[:,:,:,0]
        outputs['_Cd'] = y[:,:,:,1]

        # size = len(alpha)
        # Cl = np.empty((size))
        # Cd = np.empty((size))

        # x = np.empty((size,2))

        # x[:,0]      = alpha
        # x[:,1]      = Re/1.5e6
        # y           = interp.predict_values(x)
        # Cl          = y[:,0]
        # Cd          = y[:,1]

        # print(Cl)
        # print(Cd)

        # outputs['_Cl'] = Cl.reshape(shape)
        # outputs['_Cd'] = Cd.reshape(shape)

    def compute_partials(self, inputs, partials):
        alpha       = inputs['_alpha'].flatten()
        Re          = inputs['Re'].flatten()
        rotor       = self.options['rotor']
        airfoil     = rotor['airfoil_name']
        # size = len(alpha)
        # x = np.empty((size,2))
        # x[:,0]      = alpha
        # x[:,1]      = Re/1.5e6

        self.x[:, 0] = alpha
        self.x[:, 1] = Re/1e6

        # interp = get_surrogate_model(airfoil)
        dy_dalpha = interp.predict_derivatives(self.x, 0)
        dy_dRe = interp.predict_derivatives(self.x, 1)

        partials['_Cl', '_alpha'] = dy_dalpha[:, 0]
        partials['_Cd', '_alpha'] = dy_dalpha[:, 1]

        partials['_Cl', 'Re'] = dy_dRe[:, 0] / 1e6
        partials['_Cd', 'Re'] = dy_dRe[:, 1] / 1e6
             


# class AirfoilSurrogateModel(om.ExplicitComponent):

#     def initialize(self):
#         self.options.declare('shape', types = tuple)
#         self.options.declare('rotor', types = RotorParameters)
    
#     def setup(self):
#         shape = self.options['shape']
#         rotor = self.options['rotor']

#         self.add_input('_alpha', shape = shape)
#         self.add_input('Re', shape = shape)

#         self.add_output('_Cl', shape = shape)
#         self.add_output('_Cd', shape = shape)

#         self.declare_partials('_Cl', 'Re')
#         self.declare_partials('_Cl', '_alpha')
#         self.declare_partials('_Cd', 'Re')
#         self.declare_partials('_Cd', '_alpha')


#     def compute(self, inputs, outputs):
#         shape       = self.options['shape']
#         rotor       = self.options['rotor']
    
#         alpha       = inputs['_alpha'].flatten()
#         # print(alpha)
#         Re          = inputs['Re'].flatten()
#         # print(Re)
#         # print('Reynolds Number',Re)

#         size = len(alpha)
#         Cl = np.empty((size))
#         Cd = np.empty((size))

#         x = np.empty((size,2))

#         x[:,0]      = alpha
#         x[:,1]      = Re/1.5e6
#         y           = interp.predict_values(x)
#         Cl          = y[:,0]
#         Cd          = y[:,1]

#         # print(Cl)
#         # print(Cd)

#         outputs['_Cl'] = Cl.reshape(shape)
#         outputs['_Cd'] = Cd.reshape(shape)

#     def compute_partials(self, inputs, partials):
#         alpha       = inputs['_alpha'].flatten()
#         Re          = inputs['Re'].flatten()

#         size = len(alpha)
#         x = np.empty((size,2))
#         x[:,0]      = alpha
#         x[:,1]      = Re/1.5e6

#         dy_dalpha = interp.predict_derivatives(x, 0)
#         dy_dRe = interp.predict_derivatives(x, 1)

#         partials['_Cl', '_alpha'] = dy_dalpha[:, 0]
#         partials['_Cd', '_alpha'] = dy_dalpha[:, 1]

#         partials['_Cl', 'Re'] = dy_dRe[:, 0]
#         partials['_Cd', 'Re'] = dy_dRe[:, 1]
