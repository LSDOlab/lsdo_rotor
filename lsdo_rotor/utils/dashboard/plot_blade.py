import numpy as np
from lsdo_rotor.utils.dashboard.get_naca_airfoil import get_airfoil_thickness, get_airfoil_camber_4digit

def get_rotation_x(angles):
    mtx = np.zeros((len(angles), 3, 3))
    mtx[:, 0, 0] = 1.
    mtx[:, 1, 1] = np.cos(angles)
    mtx[:, 2, 1] = -np.sin(angles)
    mtx[:, 1, 2] = np.sin(angles)
    mtx[:, 2, 2] = np.cos(angles)
    return mtx


def get_rotation_z(angles):
    mtx = np.zeros((len(angles), 3, 3))
    mtx[:, 2, 2] = 1.
    mtx[:, 0, 0] = np.cos(angles)
    mtx[:, 1, 0] = -np.sin(angles)
    mtx[:, 0, 1] = np.sin(angles)
    mtx[:, 1, 1] = np.cos(angles)
    return mtx


def compute_blades(normal_vec, radii, twist, chord, origin, airfoil_root, airfoil_tip, num_blades):
    """
    Parameters
    ----------
    normal_vec : ndarray[3]
    radii : ndarray[num_spanwise]
    twist : ndarray[num_spanwise]
    chord : ndarray[num_spanwise]
    origin : ndarray[num_spanwise]
    airfoil_root : ndarray[num_chordwise, 2]
    airfoil_tip : ndarray[num_chordwise, 2]
    num_blades : int

    Returns
    -------
    ndarray[num_blades, num_spanwise, num_chordwise, 3]
    """
    num_spanwise = len(radii)
    num_chordwise = airfoil_root.shape[0]


    # First, we will compute one rectangular blade with x spanwise and z thicknesswise
    single_blade = np.zeros((num_spanwise, num_chordwise, 3))
    
    lins_spanwise = np.linspace(0., 1., num_spanwise)
    ones_chordwise = np.ones(num_chordwise)

    # Set the x values as the radii
    single_blade[:, :, 0] = np.outer(radii, ones_chordwise)

    # Set the y and z values based on the given airfoils
    for i_coord in range(2):
        single_blade[:, :, i_coord + 1] += np.outer(lins_spanwise, airfoil_tip[:, i_coord])
        single_blade[:, :, i_coord + 1] += np.outer(1 - lins_spanwise, airfoil_root[:, i_coord])

    # Translate so that each section's origin is at the origin of the y-z axes
    single_blade[:, :, 1] -= np.outer(origin, ones_chordwise)

    # Scale each section up by the chord
    for i_coord in range(2):
        single_blade[:, :, i_coord + 1] *= np.outer(chord, ones_chordwise)

    # Rotate each section by the twist
    mtx = np.einsum('ikl,j->ijkl', get_rotation_x(twist), ones_chordwise)
    twisted_single_blade = np.zeros((num_spanwise, num_chordwise, 3))
    for i in range(3):
        for j in range(3):
            twisted_single_blade[:, :, i] += mtx[:, :, i, j] * single_blade[:, :, j]

    # Rotate each blade
    blade_rotation_angles = np.arange(num_blades) / num_blades * 2 * np.pi * np.ones(num_blades)
    mtx = np.einsum(
        'bij,sc->bscij', 
        get_rotation_z(blade_rotation_angles), 
        np.ones((num_spanwise, num_chordwise)),
    )
    blades = np.zeros((num_blades, num_spanwise, num_chordwise, 3))
    for i_blade in range(num_blades):
        for i in range(3):
            for j in range(3):
                blades[i_blade, :, :, i] += mtx[i_blade, :, :, i, j] * twisted_single_blade[:, :, j]

    return blades


def get_blade_geo(radii, twist, chord, num_blades):
    num_spanwise = len(radii)
    normal_vec = np.array([1., 0., 0.])
    # radii = np.linspace(0.1, 1., num_spanwise)
    # twist = np.linspace(50., 20., num_spanwise) * np.pi / 180.
    # chord = np.linspace(0.3, 0.1, num_spanwise)

    origin = 0.25

    # num_blades = 5

    num = 200
    num_chordwise = 2 * num
    x = np.linspace(0., 1., num)
    t = get_airfoil_thickness(x, 0.15)
    c = get_airfoil_camber_4digit(x, 0.02, 0.4)
    airfoil = np.empty((2, num, 2))
    airfoil[0, :, 0] = x
    airfoil[1, :, 0] = x
    airfoil[0, :, 1] = c + t
    airfoil[1, :, 1] = c - t
    airfoil = airfoil.reshape((num_chordwise, 2))

    blades = compute_blades(normal_vec, radii, twist, chord, origin, airfoil, airfoil, num_blades)

    # Plot
    # cvd = CaddeeVedoContainer()
    return_surfaces = []
    for i_blade in range(num_blades):
        return_surfaces.append(blades[i_blade, :, :, :])
    
    return return_surfaces