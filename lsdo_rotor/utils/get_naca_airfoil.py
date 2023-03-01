import numpy as np


def get_airfoil_thickness(x, max_thickness):
    """
    # Given a NACA-abcd airfoil,
    max_thickness = cd * 0.01
    """
    thickness = 5 * max_thickness * (
        0.2969 * x ** 0.5 - 0.1260 * x - 0.3516 * x **2 + 0.2843 * x ** 3 - 0.1015 * x ** 4
    )
    return thickness


def get_airfoil_camber_4digit(x, max_camber, loc_max_camber):
    """
    # Given a NACA-abcd airfoil,
    max_camber = a * 0.01
    loc_max_camber = b * 0.1
    """
    camber = np.array(x)
    
    mask_front = x < loc_max_camber
    mask_rear = np.logical_not(mask_front)

    tmp = max_camber / loc_max_camber ** 2 * (2 * loc_max_camber * x - x ** 2)
    camber[mask_front] = tmp[mask_front]

    tmp = max_camber / (1 - loc_max_camber) ** 2 * ((1 - 2 * loc_max_camber) + 2 * loc_max_camber * x - x ** 2)
    camber[mask_rear] = tmp[mask_rear]

    return camber


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num = 200
    x = np.linspace(0., 1., num)
    t = get_airfoil_thickness(x, 0.1)
    c = get_airfoil_camber_4digit(x, 0.02, 0.4)

    plt.plot(x, t + c, 'k')
    plt.plot(x, -t + c, 'k')
    plt.axis('equal')
    plt.show()