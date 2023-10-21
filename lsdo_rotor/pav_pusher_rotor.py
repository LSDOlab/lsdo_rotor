"""
Aurora PAV pusher rotor analysis and optimization
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


def optimize_pusher_rotor(rotor_radius: float,
                          rotor_rpm: float,
                          velocity: float,
                          twist_cp_guess,
                          chord_cp_guess,
                          normalized_hub_radius: float,
                          expected_thrust: float,
                          maximum_torque: float,
                          efficiency: float
                          ):
    # region BEM Mesh
    bem_mesh = BEMMesh(
        num_cp=len(twist_cp),
        num_radial=25,
        num_tangential=1,
        num_blades=5,
        use_rotor_geometry=False,
        airfoil='NACA_4412',
        use_custom_airfoil_ml=True,
        # use_airfoil_ml=False,
        twist_b_spline_rep=True,
        chord_b_spline_rep=True,
        mesh_units='m',
        normalized_hub_radius=normalized_hub_radius
    )
    # endregion

    # region BEM Model
    bem_model = BEM(
        component=None,
        mesh=bem_mesh,
        disk_prefix='disk',
        blade_prefix='blade',
        use_caddee=False,
    )

    bem_model.set_module_input('chord_cp',
                               val=chord_cp_guess,
                               dv_flag=True,
                               lower=0.01, upper=0.4)
    bem_model.set_module_input('twist_cp',
                               val=twist_cp_guess,
                               dv_flag=True,
                               lower=np.deg2rad(0), upper=np.deg2rad(85))
    bem_model.set_module_input('thrust_vector', val=np.array([1., 0., 0.]))
    bem_model.set_module_input('thrust_origin', val=np.array([19.700, 0., 2.625])* ft2m)  # m
    bem_model.set_module_input('propeller_radius', val=rotor_radius)
    bem_model.set_module_input('rpm', val=rotor_rpm)
    bem_model.set_module_input('u', val=velocity)
    bem_model.set_module_input('v', val=0)
    bem_model.set_module_input('w', val=0)

    bem_csdl = bem_model.compute()
    # endregion

    # region Optimization setup
    bem_csdl.add_constraint('T', scaler=1e-3, equals=expected_thrust)
    bem_csdl.add_objective('Q', scaler=1e-2)
    # endregion

    sim = Simulator(bem_csdl, analytics=True)
    sim.run()

    prob = CSDLProblem(
        problem_name='pav_pusher_rotor_shape_opt',
        simulator=sim)
    optimizer = SLSQP(
        prob,
        maxiter=250,
        ftol=1e-4,
    )
    optimizer.solve()
    optimizer.print_results()

    chord_in = sim['chord_profile'].flatten() * m2in
    twist_deg = np.rad2deg(sim['twist_profile'].flatten())
    radius_in = sim['_radius'].flatten() * m2in
    normradius_in = radius_in / (rotor_radius * m2in)
    mach_number = sim['mach_number'].flatten()
    re = sim['Re'].flatten()
    alpha = np.rad2deg(sim['alpha_distribution'].flatten())
    Cl = sim['Cl'].flatten()
    Cd = sim['Cd'].flatten()
    dT = sim['induced_velocity_model._dT'].flatten()
    dQ = sim['induced_velocity_model._dQ'].flatten()

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
            'Sectional T': dT,
            'Sectional Q': dQ,
        }
    )
    print(rotorDf)
    rotorDf.to_excel('pav_pusher_rotor_analysis.xlsx')

    print('Forces about reference point (N): ', sim['F'].flatten())
    print('Moments about reference point (Nm): ', sim['M'].flatten())
    print('Thrust computed (N): ', sim['T'])
    print('Torque (Nm): ', sim['Q'])
    print('Cruise efficiency: ', sim['eta'])
    print('Twist cp: ', sim['twist_cp'])
    print('Chord cp: ', sim['chord_cp'])
    print('Rotor total thrust coefficient: ', sim['C_T'])
    print('Velocity: ', sim['u'], sim['v'], sim['w'])
    return


if __name__ == '__main__':
    rotor_radius = 4 / 2 * ft2m  # 4 ft rotor diameter
    normalized_hub_radius = 0.2
    velocity = 50.06848  # 112 mph = 50.06848 m/s
    rotor_rpm = 4000.
    twist_cp = np.array([0.55207943, 0.35981639, 0.16753661, 0.12377559])  # rad
    chord_cp = np.array([0.07295861, 0.10717677, 0.09075833, 0.06437597])  # m
    efficiency = 0.8
    LoverD = 8
    m = 800  # kg
    W = m*9.81  # N
    D = W/LoverD
    expected_thrust = 1.*D
    max_torque = 160. # Nm

    optimize_pusher_rotor(rotor_radius=rotor_radius,
                          rotor_rpm=rotor_rpm,
                          twist_cp_guess=twist_cp,
                          chord_cp_guess=chord_cp,
                          normalized_hub_radius=normalized_hub_radius,
                          expected_thrust=expected_thrust,
                          maximum_torque=max_torque,
                          efficiency=efficiency,
                          velocity=velocity)
