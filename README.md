# lsdo_rotor

This is the rotor analysis and design tool developed by the LSDO lab. Please following these instructions on installation and proper usage. 

# Installation 

lsdo_rotor requires the following packages to be installed before it can be used:

* [csdl](https://lsdolab.github.io/csdl/docs/tutorial/install) 
* [smt](https://smt.readthedocs.io/en/latest/_src_docs/getting_started.html)

Please follow the installation instructions provided in the above links. Once these packages are installed you can proceed as follows with the installation of lsdo_rotor:

* Clone this repository via ``git clone https://github.com/MariusLRuh/lsdo_rotor.git`` or download it as a .zip file. If the repository is cloned successfully, future versions of lsdo_rotor can be downloaded via `git pull`.
* Execute the next two commands to install lsdo_rotor
  * ``cd lsdo_rotor``
  * ``pip install -e .``
* If the installation is successful, check that the run.py file executes by typing
  * ``cd lsdo_rotor``
  * ``python run.py``
  
# User guidelines

The user will only have to change parameters in the `run.py` file, which has comments to explain how to properly use the code. Currently, our rotor analysis tool currently supports two modes of operation:

1) Ideal-loading design method [`ILDM`](https://arc.aiaa.org/doi/abs/10.2514/6.2021-2598):
  This is a rotor DESIGN tool to efficiently compute the most aerodynamic blade geometry of a rotor for given operating conditions. Unless otherwise indicated, all quantities have SI units. The following parameters can be adjusted in the run.py file:
    * `Vx` This is the axial inflow velocity perpendicular to the rotor disc. `Vx [m/s]` can be any reasonable number greater or equal to 0. (Note: If `Vx = 0` the aircraft is hovering) 
    * `Vy` `Vz` These are "sideslip" velocity components in the rotor plane. Because the `ILDM` is based on  BEM theory, the deisgn are most reliable if there is no sideslip. Therefore, we recommend `Vx = Vy = 0` if the user wants to use this design tool. 
    * `reference_radius` The user needs to specify a reference radius at which a value for the chord length is specified. We recommend `reference_radius = rotor_radius / 2 = rotor_diameter / 4`
    * `reference_chord` The user needs to specify a reference chord length AT the above mentioned `reference_radius`. We defer to the judgment of user to specify a reasonable value
    * `num_radial` This specifies the number of radial nodes. The larger the value the more accurate the results will. We recommend a value of at least 25. 
    These are only the parameters that require some more explanation. Please follow the comments in the run.py file for the other parameters
  
  The output of the `ILDM` is the following
    * All performance related parameters (e.g. thrust, torque, efficiency, etc); The user can print these by setting `print_rotor_performance = 'y'`
    * The ideal, back-computed blade shape given by twist and chord; The user can plot the ideal blade profile by setting `plot_rotor_blade_shape  = 'y'`
  
2) Blade element momentum (BEM) theory
  This is a rotor ANALYSIS tool that computes the aerodynamic performance of an EXISTING rotor using BEM theory. In the run.py file the user can change the following parameters:
    * `Vx` This is the axial inflow velocity perpendicular to the rotor disc. `Vx [m/s]` can be any reasonable number greater or equal to 0. (Note: If `Vx = 0` the aircraft is hovering) 
    * `Vy` `Vz` These are "sideslip" velocity components in the rotor plane. Because the momentum part of BEM theory assumes strictly axial inflow, results will be most accurate if `Vx = Vy = 0`. However, for small sideslip velocities (or if Vx dominates Vy and Vz), results are still reliable.
    * `root_chord` `tip_chord` If the user does not define a rotor geometry, a linearly varying chord profile can defined by specifying the chord length at the rotor hub and tip
    * `root_twist` `tip_twist [deg]` Likewise, a linearly varying twist profile can be specified
    * `use_external_rotor_geometry = 'y/n' ` The user has the option the specify the rotor geometry of an existing rotor. A good database for small rotors can be found on the [UIUC](https://m-selig.ae.illinois.edu/props/propDB.html) website. The rotor geometry should be stored in .txt file. Please also follow the instructions in the run.py file.
  

  
  
