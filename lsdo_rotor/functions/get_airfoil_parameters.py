import numpy as np

def get_airfoil_parameters(text_file):
    airfoil_filename = text_file
    data = np.loadtxt(airfoil_filename, delimiter=',', skiprows=1, dtype=str)
    data = data[11:]
    np.savetxt("data.txt", data, fmt="%s")
    with open('data.txt', 'r') as f:
        data = f.read().split()
        floats = []
        for elem in data:
            try:
                floats.append(float(elem))
            except ValueError:
                pass
    airfoil_data = np.array(floats)

    num_col = 7
    num_row = int(len(airfoil_data)/ num_col)
    airfoil_data = airfoil_data.reshape(num_row, num_col)
    alpha_data = airfoil_data[:,0]
    CL_data = airfoil_data[:,1]
    CD_data = airfoil_data[:,2]

    Cl0 = float(CL_data[np.where(alpha_data == 0)])
    # Cl0 = set manually if need be

    Cdmin = float(np.min(CD_data))
    # Cdmin 

    a_cdmin = alpha_data[np.where(CD_data == Cdmin)] * np.pi / 180
    # a_cdmin = set manually if need be
    if len(a_cdmin) > 1:
        a_cdmin = float(np.average(a_cdmin))
    elif len(a_cdmin) == 1:
        a_cdmin = float(a_cdmin)

    K = float((CD_data[np.where(alpha_data == 5)] - Cdmin) / ((5 * np.pi / 180) - a_cdmin) **2)
    # K = set manually if need be

    Cl_stall_plus = float(np.max(CL_data))
    # Cl_stall_plus = set manually if need be
    Cl_stall_minus = float(np.min(CL_data))
    # Cl_stall_minus = set manually if need be

    a_stall_plus = float(alpha_data[np.where(CL_data == Cl_stall_plus)] * np.pi / 180)
    # a_stall_plus = set manually if need be
    a_stall_minus = float(alpha_data[np.where(CL_data == Cl_stall_minus)] * np.pi / 180)
    # a_stall_minus = set manually if need be

    Cla = float((CL_data[np.where(alpha_data == 5)] - CL_data[np.where(alpha_data == -2.5)]) / (7.5 * np.pi / 180))
    # Cla = set manually if need be

    Cd_stall_plus = float(CD_data[np.where(CL_data == Cl_stall_plus)])
    #Cd_stall_plus = set manually if need be
    Cd_stall_minus = float(CD_data[np.where(CL_data == Cl_stall_minus)])
    # Cd_stall_minus = set manually if need be

    return a_stall_plus, Cl_stall_plus, Cd_stall_plus, a_stall_minus, Cl_stall_minus, Cd_stall_minus, Cl0, Cla, K, Cdmin, a_cdmin  