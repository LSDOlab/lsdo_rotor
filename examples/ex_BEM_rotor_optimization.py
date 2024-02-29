"""
Example 1: Rotor chord and twist profile optimization using blade element momentum theory
"""

"""
Overview
--------
In this example we optimize a rotor blade subject to a thrust constraint. The blade is discretized
using 6 B-spline control points for chord and twist each, meaning the optimizer has 12 degrees of freedom
to manipulate the blade shape. Using control points to discretize the blade is a way of reducign the
dimensionality of the problem. However, if we do not want to perform optimization we can also
directly input the twist and chord distribution, e.g., as a numpy array, as long as length of the array
is equal to 'num_radial', meaning the number of radial stations. 

Notable inputs (assume SI units)
--------
    u:              free stream x-velocity in the body fixed aircraft reference frame
    altitude:       important for computing atmospheric properties
    chord_cp:       control points for the chord distribution (can be replaced by full distribution)
    twist_cp:       control points for the twist distribution (can be replaced by full distribution)
    thrust_vector:  unit vector pointing in the direction of the thrust 
                    [1, 0, 0] is forward (+x), [0, 0, -1] is upward (-z); two recommended values
                    NOTE: Keep in mind that BEM works best of hover or axial flow, based on the underlying 
                    phyiscs. Although BEM has been used for edgewise flow cases, we recommend usage for 
                    hover, axial, or mild disk tilt cases only (i.e., small angle approximation)
    thrust_origin:  the origin at which the thrust acts (relevant for computing moments)
    radius:         self-explanatory
    rpm:            self-explanatory

Creating design variables
--------
When creating variables with the 'create_input' method, the 'dv_flag' argument indicates
that a variable is a design variable. The 'lower' and 'upper' keywords indicate the bounds on 
the design variables and the 'scaler' argument will scale the design variable. 
NOTE: Scaling of design variables is very important for any optimization to converge. It is recommended
that design variables are scaled to the order of unity. E.g., a rotor rpm variable will typically 
be on the order of a thousand and should be scaled accordingly. 

Choosing the optimization objective
--------
The optimization objective should be chosen carefully. Keep in mind that by convention, optimization
problems are formulated as a minimization problem. That means that if you want to maximize a variable
it needs to be scaled with a negative value. A potential challenge is that the solver (i.e., BEM model) 
behavior may be erratic and the solver may even fail to converge depending on the objective. E.g.,
choosing rotor aerodynamic efficiency as an objective may seem like a good choice. However, the 
optimizer does not know that an efficiency of greater than one does not make sense and it may converge
to a design that actually produces negative thrust or torque. In this case, a better objective would be
to minimize torque or power. 
In this example, we provide two options for the objective: 1) formulate a thrust residual (i.e., 
target thrust - computed thrust) and set that residual as the objective. 2)  aerodynamic torque subject 
to a constraint on the efficiency. 

Post-processing
--------
We provide a convenience function to print an abundance of output data and write it to a .csv file. 
"""

# Imports
import numpy as np 
from python_csdl_backend import Simulator
from lsdo_rotor import RotorAnalysis, BEM, BEMParameters, AcStates, get_atmosphere, print_output
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem

# Instantiate a RotorAnalysis object (inherits from m3l.Model)
rotor_analysis = RotorAnalysis()

# create a subset of the aircraft states ('u' only is sufficient) and altitude 
u = rotor_analysis.create_input('u', val=50.06, shape=(1, ))
altitude = rotor_analysis.create_input('altitude', val=0, shape=(1, ))

# instantiate AcStates and use helper function to compute atmospheric properties
ac_states = AcStates(u=u)
atmos = get_atmosphere(altitude=altitude)

# set up radial and azimuthal discretization
num_radial = 25
num_tangential = 1

# define number of blades and number of B-spline control points (for chord and twist parameterization)
num_blades = 5
num_bspline_cp = 6

# create variables for chord control/twist (control points), thrust vector/origin, radius and rpm
chord_cp = rotor_analysis.create_input('chord_cp', val=np.linspace(0.2, 0.2, num_bspline_cp), dv_flag=True, lower=0.01, upper=0.4)
twist_cp = rotor_analysis.create_input('twist_cp', val=np.deg2rad(np.linspace(65, 20, num_bspline_cp)), dv_flag=True, lower=np.deg2rad(0), upper=np.deg2rad(85))
thrust_vector = rotor_analysis.create_input('thrust_vector', val=np.array([1, 0, 0]))
thrust_origin = rotor_analysis.create_input('thrust_origin', val=np.array([0, 0, 0]))
propeller_radius = rotor_analysis.create_input('propeller_radius', val=0.61)
rpm = rotor_analysis.create_input('rpm', val=4000)

# set up BEM parameters;
##### Option 1 (recommended) 'NACA 4412' airfoil has well-trained machine learning model (indicated by 'use_custom_ml=True') ####
# Pros: 
#   - trained on XFOIL data 
#   - reasonable accuracy
#   - models Cl/Cd as a function of AoA, Re, Mach
# Cons:
#   - Slower
bem_parameters = BEMParameters(
    num_radial=num_radial,
    num_tangential=num_tangential,
    num_blades=num_blades,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=True,         # if you want to use ML airfoil model
    num_cp=num_bspline_cp,
)

##### Option 2: define custom (quadratic) airfoil polar (only as a function of AoA) ##### 
# Pros: 
#   - Fast
#   - Can define for other airfoils
# Cons:
#   - inaccurate
#   - models Cl/Cd as a function of AoA only (gross simplification)
# airfoil_polar = {
#     'Cl_0': 0.25,
#     'Cl_alpha': 5.1566,
#     'Cd_0': 0.01,
#     'Cl_stall': [-1, 1.5], 
#     'Cd_stall': [0.02, 0.06],
#     'alpha_Cl_stall': [-10, 15],
# }
# bem_parameters = BEMParameters(
#     num_radial=num_radial,
#     num_tangential=num_tangential,
#     num_blades=num_blades,
#     airfoil='NACA_4412',
#     airfoil_polar=airfoil_polar,        # set custom airfoil polar
#     num_cp=num_bspline_cp,
# )


# Instantiate BEM solver
bem_model = BEM(
    name='bem_analysis',
    BEM_parameters=bem_parameters,
    num_nodes=1,
)

# evaluate BEM solver and register its outputs
bem_outputs = bem_model.evaluate(ac_states=ac_states, rpm=rpm, rotor_radius=propeller_radius, thrust_vector=thrust_vector, thrust_origin=thrust_origin,
                                 atmosphere=atmos, blade_chord_cp=chord_cp, blade_twist_cp=twist_cp)
rotor_analysis.register_output(bem_outputs)

# Define thrust constraints and register it as an output (NOTE: registering the output is important!!)
thrust = bem_outputs.T
desired_thrust = 981
thrust_residual = ((thrust-desired_thrust)**2)**0.5
rotor_analysis.register_output(thrust_residual)

# Option 1) Minimize torque subject to a thrust constraint
rotor_analysis.add_constraint(bem_outputs.T, equals=desired_thrust, scaler=1e-3)
rotor_analysis.add_objective(bem_outputs.Q, scaler=1e-2)

# Option 2) Minimize a thrust residual subject to a constant efficiency
# rotor_analysis.add_constraint(bem_outputs.eta, equals=0.8)
# rotor_analysis.add_objective(thrust_residual, scaler=1e-2)

# Assemble CSDL model and instantiate Simulator and run model
csdl_model = rotor_analysis.assemble_csdl()
sim = Simulator(csdl_model, analytics=True)
sim.run()

# print outputs and save outputs to csv file if desired
print_output(sim, rotor=rotor_analysis, comprehensive_print=False, write_to_csv=False)

# Optimization using SLSQP through modOpt
prob = CSDLProblem(problem_name='bem_blade_shape_optimization', simulator=sim)
optimizer = SLSQP(prob, maxiter=100, ftol=1E-7)
optimizer.solve()
optimizer.print_results()

# print outputs again after optimization
print_output(sim, rotor=rotor_analysis, comprehensive_print=True, write_to_csv=True, file_name='test_BEM_opt')
