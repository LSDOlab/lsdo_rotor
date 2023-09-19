"""
Aurora PAV lift rotor analysis and optimization
"""

import numpy as np 
import pandas as pd
from python_csdl_backend import Simulator
from lsdo_rotor.core.BEM.BEM_run_model import BEMRunModel
from lsdo_rotor.utils.print_output import print_output
from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


ft2m = 0.3048
m2in = 39.3701


def analyze_lift_rotor(rotor_radius: float, 
                       rotor_rpm: float,
                       twist_cp,
                       chord_cp,
                       normalized_hub_radius: float
                       ):

    bem_mesh = BEMMesh(
        num_cp=len(twist_cp),
        num_radial=25,
        num_tangential=1,
        num_blades=2,
        use_rotor_geometry=False,
        airfoil='NACA_4412',
        use_custom_airfoil_ml=True,
        # use_airfoil_ml=False,
        twist_b_spline_rep=True,
        chord_b_spline_rep=True,
        mesh_units='m',
        normalized_hub_radius=normalized_hub_radius
    )

    bem_model = BEM(
        component=None,
        mesh=bem_mesh,
        disk_prefix='disk',
        blade_prefix='blade',
        use_caddee=False,
    )

    bem_model.set_module_input('chord_cp', 
                            val=chord_cp, 
                            dv_flag=False, 
                            lower=0.01, upper=0.4)
    bem_model.set_module_input('twist_cp', 
                            val=twist_cp, 
                            dv_flag=False, 
                            lower=np.deg2rad(0), upper=np.deg2rad(85))
    bem_model.set_module_input('thrust_vector', val=np.array([0., 0., -1.]))
    bem_model.set_module_input('thrust_origin', val=np.array([-1.146, 1.619, -0.162]))  # m
    bem_model.set_module_input('propeller_radius', val=rotor_radius)
    bem_model.set_module_input('rpm', val=rotor_rpm)
    bem_model.set_module_input('u', val=0)
    bem_model.set_module_input('v', val=0)
    bem_model.set_module_input('w', val=0)

    bem_csdl = bem_model.compute()
    sim = Simulator(bem_csdl, analytics=True)
    sim.run()

    chord_in = sim['chord_profile'].flatten()*m2in
    twist_deg = np.rad2deg(sim['twist_profile'].flatten())
    radius_in = sim['_radius'].flatten()*m2in
    normradius_in = radius_in/(rotor_radius*m2in)
    mach_number = sim['mach_number'].flatten()
    re = sim['Re'].flatten()
    alpha =  np.rad2deg(sim['alpha_distribution'].flatten())
    Cl = sim['Cl'].flatten()
    Cd = sim['Cd'].flatten()


    rotorDf = pd.DataFrame(
        data={
            'Normalized radius': normradius_in,
            'Chord (in)': chord_in,
            'Twist (deg)': twist_deg,
            'Mach number': mach_number,
            'Reynolds number': re,
            'AoA (deg)': alpha,
            'Cl': Cl,
            'Cd': Cd,
            # 'F': F,
            # 'eta': eta
        }
    )
    print(rotorDf)

    print('Forces about refenence point (N): ', sim['F'].flatten())
    print('Moments about refenence point (Nm): ', sim['M'].flatten())
    print('Thrust (N): ', sim['T'])
    print('Torque (Nm): ', sim['Q'])
    return


if __name__ == '__main__':
    rotor_radius = 6/2*ft2m  # 6ft rotor diameter 
    normalized_hub_radius = 0.1
    rotor_rpm = 2400.
    twist_cp = np.array([0.55207943, 0.35981639, 0.16753661, 0.12377559, 0.17724111, 0.07146789])  # rad
    chord_cp = np.array([0.07295861, 0.10717677, 0.09075833, 0.06437597, 0.03848824, 0.02721645])  # m

    analyze_lift_rotor(rotor_radius=rotor_radius,
                       rotor_rpm=rotor_rpm,
                       twist_cp=twist_cp,
                       chord_cp=chord_cp,
                       normalized_hub_radius=normalized_hub_radius)
    