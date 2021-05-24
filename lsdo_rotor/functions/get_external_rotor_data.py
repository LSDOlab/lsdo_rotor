import numpy as np 



def get_external_rotor_data(rotor_geometry, rotor_performance):
    
    geometry = rotor_geometry
    data = np.loadtxt(geometry, delimiter=',', skiprows=1, dtype=str)
    with open(rotor_geometry, 'r') as f:
        data = f.read().split()
        floats = []
        for elem in data:
            try:
                floats.append(float(elem))
            except ValueError:
                pass
    floats = np.array(floats)
    num_rows = int(len(floats)/3)
    num_cols = 3
    apc_geometry_data = np.reshape(floats, (num_rows, num_cols))
    normalized_chord = apc_geometry_data[:,1]
    twist = apc_geometry_data[:,2]

    performance = rotor_performance
    data = np.loadtxt(performance, delimiter=',', skiprows=1, dtype=str)
    with open(performance, 'r') as f:
        data = f.read().split()
        floats = []
        for elem in data:
            try:
                floats.append(float(elem))
            except ValueError:
                pass
    floats = np.array(floats)
    num_rows = int(len(floats)/4)
    num_cols = 4
    apc_performance_data = np.reshape(floats, (num_rows, num_cols))
    rotor_J = apc_performance_data[:,0]
    rotor_CT = apc_performance_data[:,1]
    rotor_CP = apc_performance_data[:,2]
    rotor_eta = apc_performance_data[:,3]


    return normalized_chord, twist, rotor_J, rotor_CT, rotor_CP, rotor_eta
