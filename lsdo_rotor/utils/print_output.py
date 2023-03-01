import numpy as np
import pandas as pd 
pd.set_option('colheader_justify', 'center')


def print_output(sim, write_to_csv : bool=False):
    T = sim['T'].flatten()
    Q = sim['Q'].flatten()
    M = sim['M'].flatten()
    eta = sim['eta'].flatten()
    FM = sim['FOM'].flatten()
    C_T = sim['C_T'].flatten()
    C_Q = sim['C_Q'].flatten()
    C_P = sim['C_P'].flatten()

    chord = sim['_chord'].flatten()
    dr = sim['_dr'].flatten()
    twist = sim['_pitch'].flatten()
    radius = sim['_radius'].flatten()
    B = sim['blade_number'].flatten()
    R = sim['propeller_radius'].flatten()

    sigma = B * sum(chord * dr) / np.pi / R**2
    na = '-----'
    high_level_data = {
        'Efficiency' : [na, np.round(eta, 2)],
        'Figure of merit' : [na, np.round(FM, 2)],
        'Thrust' : [np.round(T, 2), np.round(C_T, 2)], 
        'Torque' : [np.round(Q, 2), np.round(C_Q, 2)], 
        'Blade solidity' : [na, np.round(sigma, 2)],
        'Blade loading': [na, np.round(C_T/sigma, 2)],
        'Disk loading' : [np.round(T / np.pi/R**2, 2),  na],
        'Moments (Mx, My, Mz)' : [np.round(M, 2), na],
    }

    high_level_df = pd.DataFrame(data=high_level_data)
    # s = high_level_df.style.format('{:.0f}').hide([('Random', 'Tumour'), ('Random', 'Non-Tumour')], axis="columns")
    high_level_df.style.set_properties(**{'text-align': 'center'})
    high_level_df.index = ['Value (SI)', 'Coeff./ dim.-less qty.']

    # distributions 
    radius = sim['_radius'].flatten()
    twist = sim['_pitch'].flatten() * 180/np.pi
    chord = sim['_chord'].flatten()
    dT = sim['_dT'].flatten()
    dQ = sim['_dQ'].flatten()
    aoa = sim['AoA'].flatten()
    Cl = sim['Cl_2'].flatten()
    Cd = sim['Cd_2'].flatten()
    LoD = Cl/Cd

    distributions = {
        'radius' : np.round(radius, 3),
        'chord length' : np.round(chord, 3),
        'blade twist (deg)' : np.round(twist, 3), 
        'dT' : np.round(dT, 3),
        'dQ' : np.round(dQ, 3),
        'AoA (deg)' : np.round(aoa, 3),
        'Cl' : np.round(Cl, 3),
        'Cd' : np.round(Cd, 3),
        'Cl/Cd' : np.round(LoD, 3),
    }
    
    distributions_df = pd.DataFrame(data=distributions)
    print('\n')
    message1 =  '-------------------------------------' + '\n' \
                '| High-level performance parameters |' + '\n' + \
                '-------------------------------------'
    print(message1)
    print(high_level_df)
    print('\n')
    message2 =  '----------------' + '\n' \
                '| Distributions |' + '\n' + \
                '----------------'
    print(message2)
    print(distributions_df)
    # print(s)
