import numpy as np
import omtools.api as ot


class InputsGroup(ot.Group):
    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        twist = self.create_indep_var('twist', val=50. * np.pi / 180.)
        Vx = self.create_indep_var('Vx', val=50)
        nr = self.create_indep_var('nr', val = 10)
        
        init_dist = self.declare_input('init_dist', val = 0.5)      # initial normalized distance from hub (i.e. percent radius)
        num_blades = self.declare_input('num_blades', val = 3)
        radius = self.declare_input('radius', val=1.) 
        chord = self.declare_input('chord', val = 0.1)
        RPM = self.declare_input('RPM', val=1500.)

        sigma = num_blades * chord / (2 * np.pi * init_dist * radius)
        self.register_output('sigma',sigma)

        Vt = RPM * 2 * np.pi * init_dist * radius / (60)
        self.register_output('Vt', Vt)

        nr = 1
        radius = 1
        u = self.declare_input('u', val= np.linspace(0,radius,nr))
        rad_vec = self.create_output('rad_vec', shape=(nr, ))
        rad_vec[0:nr] =  u

        RPM = 1500
        vt = self.declare_input('vt',val = (RPM * 2 * np.pi /60) * np.linspace(0.1,radius,nr))
        Vt_vec = self.create_output('Vt_vec', shape=(nr,))
        Vt_vec[0:nr] = vt
        
    