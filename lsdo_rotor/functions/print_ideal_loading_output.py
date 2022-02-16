import numpy as np 
import pandas as pd 
pd.set_option('display.colheader_justify', 'center')

def print_ideal_loading_output(total_thrust, total_torque, Vx, n, radius, dT, dQ, ux, ut, c, theta, num_blades, dr, R):
    print('\n')
    
    print('---- ---- ---- ---- ---- ---- ---- ' + '\n' + 'IDEAL-LOADING DESIGN METHOD OUTPUT' + '\n' +'---- ---- ---- ---- ---- ---- ---- ')

    print('\n')
    
    T = total_thrust
    C_T = T / 1.2 / n**2 / (2 * R)**4 
    Q = total_torque
    C_Q = Q/ 1.2 / n**2/ (2 * R)**5
    P = 2 * np.pi * n * Q
    C_P = P / 1.2 / n**3 / (2 * R)**5
    eta = T * Vx / (Q * n * 2 * np.pi)
    FM = T / P * (T / 1.2 / 2 / np.pi / R**2)**0.5

    sigma = num_blades * np.sum(c) * dr / (np.pi * R**2)
    total_blade_area = num_blades * np.sum(c) * dr
    disk_loading = T / (np.pi * R**2)

    data = np.column_stack((np.round(T,3), np.round(C_T,3), np.round(Q,3), np.round(C_Q,3),
                            np.round(P,3), np.round(C_P,3), np.round(eta,3), np.round(FM,3), 
                            np.round(disk_loading,3) , np.round(sigma,3), np.round(total_blade_area,3) ))

    column_headers_1 = ['Thrust (N) ',
                        'Thrust coefficient ',
                        'Torque (N-m) ',
                        'Torque coefficient ',
                        'Power (N-m/s) ',
                        'Power coefficient ',
                        'Aerodynamic efficiency ',
                        'Figure of merit ', 
                        'Disk loading (N/m^2) ',
                        'Rotor solidity ', 
                        'Rotor blade area ']

    df1 = pd.DataFrame(data, columns = column_headers_1)
    print(df1.to_string(index = False))

    r = np.round(radius,3)
    dT = np.round(dT,3)
    dQ = np.round(dQ,3)
    ux = np.round(ux,3)
    ut = np.round(ut,3)
    c = np.round(c,3)
    theta = np.round(theta * 180/np.pi,3)
    
    data_2 = np.column_stack((r, dT, dQ, ux, ut, c, theta))


    print('\n' + '\n')

    column_headers_2 = ['Radius (m) (hub->tip) ',
                      'Local thrust dT (N) ', 
                      'Local torque dQ (N-m) ', 
                      'Axial induced velocity (m/s) ', 
                      'Angular induced velocity (m/s) ',
                      'Chord distribution (m) ',
                      'Twist distribution (deg) ']

    df2 = pd.DataFrame(data_2, columns = column_headers_2)
    print(df2.to_string(index = False))
