import numpy as np 

from lsdo_rotor.rotor_parameters import RotorParameters

def get_smoothing_parameters(alpha_stall_plus ,Cl_stall_plus, Cd_stall_plus, alpha_stall_minus ,Cl_stall_minus , Cd_stall_minus, AR, Cl0, Cla, K, Cdmin, alpha_Cdmin):
    Cd_max = 1.11 + 0.018 * AR 
    A1 = Cd_max / 2
    B1 = Cd_max
    A2_plus = (Cl_stall_plus - Cd_max * np.sin(alpha_stall_plus) * np.cos(alpha_stall_plus)) * np.sin(alpha_stall_plus) / (np.cos(alpha_stall_plus)**2)
    B2_plus = (Cd_stall_plus - Cd_max * np.sin(alpha_stall_plus)**2) / np.cos(alpha_stall_plus)
    A2_minus = (Cl_stall_minus - Cd_max * np.sin(alpha_stall_minus) * np.cos(alpha_stall_minus)) * np.sin(alpha_stall_minus) / (np.cos(alpha_stall_minus)**2)
    B2_minus = (Cd_stall_minus - Cd_max * np.sin(alpha_stall_minus)**2) / np.cos(alpha_stall_minus)

    condition_1 = True
    epsilon_guess_plus      = 5 * np.pi / 180
    epsilon_guess_minus     = 4 * np.pi / 180
    epsilon_cd              = 9 * np.pi/180
    eps_step                = 0.01 * np.pi/180
    tol                     = 1e-3
    
    while condition_1:
        mat_plus = np.array([
            [(alpha_stall_plus - epsilon_guess_plus)**3, (alpha_stall_plus - epsilon_guess_plus)**2, (alpha_stall_plus - epsilon_guess_plus), 1 ],
            [(alpha_stall_plus + epsilon_guess_plus)**3, (alpha_stall_plus + epsilon_guess_plus)**2, (alpha_stall_plus + epsilon_guess_plus), 1 ],
            [3 * (alpha_stall_plus - epsilon_guess_plus)**2, 2 * (alpha_stall_plus - epsilon_guess_plus), 1, 0 ],
            [3 * (alpha_stall_plus + epsilon_guess_plus)**2, 2 * (alpha_stall_plus + epsilon_guess_plus), 1, 0 ],
        ], dtype='float')

        RHS_Cl_plus = np.array([
            Cl0 + Cla * (alpha_stall_plus - epsilon_guess_plus),
            A1 * np.sin(2 * (alpha_stall_plus + epsilon_guess_plus)) + A2_plus * np.cos(alpha_stall_plus + epsilon_guess_plus)**2 / np.sin(alpha_stall_plus + epsilon_guess_plus),
            Cla,
            2 *  A1 * np.cos(2 * (alpha_stall_plus + epsilon_guess_plus)) - A2_plus * (np.cos(alpha_stall_plus + epsilon_guess_plus) + np.cos(alpha_stall_plus + epsilon_guess_plus) / np.sin(alpha_stall_plus + epsilon_guess_plus)**2),
        ], dtype='float')
        coeff_Cl_plus = np.linalg.solve(mat_plus,RHS_Cl_plus)

        alpha_plus = np.linspace(alpha_stall_plus - epsilon_guess_plus, alpha_stall_plus + epsilon_guess_plus, 200) 
        Cl_smoothing = np.zeros((len(alpha_plus)))

        for i in range(len(alpha_plus)):
            Cl_smoothing[i] = coeff_Cl_plus[0] * alpha_plus[i]**3 + coeff_Cl_plus[1] * alpha_plus[i]**2 + coeff_Cl_plus[2] * alpha_plus[i] + coeff_Cl_plus[3]

        Cl_max = np.max(Cl_smoothing)
        # print(Cl_max - Cl_stall_plus)
        if np.abs(Cl_max - Cl_stall_plus) > tol:
            if Cl_max > Cl_stall_plus:
                epsilon_guess_plus = epsilon_guess_plus + eps_step
            elif Cl_max < Cl_stall_plus:
                epsilon_guess_plus = epsilon_guess_plus - eps_step
        else:
            break
        epsilon_plus = epsilon_guess_plus
        condition_1 = (np.abs(Cl_max - Cl_stall_plus) > tol)

    condition_2 = True
    eps_step_2 = 0.05 * np.pi/180
    
    while condition_2:
        mat_minus = np.array([
            [(alpha_stall_minus + epsilon_guess_minus)**3, (alpha_stall_minus + epsilon_guess_minus)**2, (alpha_stall_minus + epsilon_guess_minus), 1 ],
            [(alpha_stall_minus - epsilon_guess_minus)**3, (alpha_stall_minus - epsilon_guess_minus)**2, (alpha_stall_minus - epsilon_guess_minus), 1 ],
            [3 * (alpha_stall_minus + epsilon_guess_minus)**2, 2 * (alpha_stall_minus + epsilon_guess_minus), 1, 0 ],
            [3 * (alpha_stall_minus - epsilon_guess_minus)**2, 2 * (alpha_stall_minus - epsilon_guess_minus), 1, 0 ],
        ], dtype='float')

        RHS_Cl_minus = np.array([
            Cl0 + Cla * (alpha_stall_minus + epsilon_guess_minus),
            A1 * np.sin(2 * (alpha_stall_minus - epsilon_guess_minus)) + A2_minus * np.cos(alpha_stall_minus - epsilon_guess_minus)**2 / np.sin(alpha_stall_minus - epsilon_guess_minus),
            Cla,
            2 *  A1 * np.cos(2 * (alpha_stall_minus - epsilon_guess_minus)) - A2_minus * (np.cos(alpha_stall_minus - epsilon_guess_minus) + np.cos(alpha_stall_minus - epsilon_guess_minus) / np.sin(alpha_stall_minus - epsilon_guess_minus)**2),
        ], dtype='float')  
        coeff_Cl_minus = np.linalg.solve(mat_minus,RHS_Cl_minus)
    
        alpha_minus = np.linspace(alpha_stall_minus - epsilon_guess_minus, alpha_stall_minus + epsilon_guess_minus, 200) 
        Cl_smoothing_minus = np.zeros((len(alpha_minus)))

        for i in range(len(alpha_minus)):
            Cl_smoothing_minus[i] = coeff_Cl_minus[0] * alpha_minus[i]**3 + coeff_Cl_minus[1] * alpha_minus[i]**2 + coeff_Cl_minus[2] * alpha_minus[i] + coeff_Cl_minus[3]

        da = np.abs(alpha_minus[1] - alpha_minus[0])
        na = len(alpha_minus)
        dClda = np.zeros(len(Cl_smoothing_minus))
        dClda_diff = np.zeros(len(Cl_smoothing_minus))

        for i in range(len(Cl_smoothing_minus)):
            if i == (na-1):
                dClda[i] =  (Cl_smoothing_minus[i] - Cl_smoothing_minus[i-1]) / da
                dClda_diff[i] = dClda[i] - dClda[i-1]
            else:
                dClda[i] =  (Cl_smoothing_minus[i+1] - Cl_smoothing_minus[i]) / da
                dClda_diff[i] = dClda[i+1] - dClda[i]

        if any(dClda_diff < 0):
            epsilon_guess_minus = epsilon_guess_minus + eps_step_2

        if epsilon_guess_minus >= (np.abs(alpha_stall_minus) + 2.5 * np.pi/180):
            condition_2 = False
        
        if all(dClda_diff > 0):
            condition_2 = False

        epsilon_minus = epsilon_guess_minus

    mat_cd_plus = np.array([
                [(alpha_stall_plus - epsilon_cd)**3, (alpha_stall_plus - epsilon_cd)**2, (alpha_stall_plus - epsilon_cd), 1 ],
                [(alpha_stall_plus + epsilon_cd)**3, (alpha_stall_plus + epsilon_cd)**2, (alpha_stall_plus + epsilon_cd), 1 ],
                [3 * (alpha_stall_plus - epsilon_cd)**2, 2 * (alpha_stall_plus - epsilon_cd), 1, 0 ],
                [3 * (alpha_stall_plus + epsilon_cd)**2, 2 * (alpha_stall_plus + epsilon_cd), 1, 0 ],
            ], dtype='float')

    RHS_Cd_plus = np.array([
        Cdmin + K * ((alpha_stall_plus - epsilon_cd) - alpha_Cdmin)**2,
        B1 * np.sin(alpha_stall_plus + epsilon_cd)**2 + B2_plus * np.cos(alpha_stall_plus + epsilon_cd),
        2 * K * ((alpha_stall_plus - epsilon_cd) - alpha_Cdmin),
        B1 * np.sin(2 * (alpha_stall_plus + epsilon_cd)) - B2_plus * np.sin(alpha_stall_plus + epsilon_cd),
    ], dtype='float')   
    coeff_Cd_plus = np.linalg.solve(mat_cd_plus,RHS_Cd_plus)

    mat_cd_minus = np.array([
            [(alpha_stall_minus + epsilon_cd)**3, (alpha_stall_minus + epsilon_cd)**2, (alpha_stall_minus + epsilon_cd), 1 ],
            [(alpha_stall_minus - epsilon_cd)**3, (alpha_stall_minus - epsilon_cd)**2, (alpha_stall_minus - epsilon_cd), 1 ],
            [3 * (alpha_stall_minus + epsilon_cd)**2, 2 * (alpha_stall_minus + epsilon_cd), 1, 0 ],
            [3 * (alpha_stall_minus - epsilon_cd)**2, 2 * (alpha_stall_minus - epsilon_cd), 1, 0 ],
        ], dtype='float')

    RHS_Cd_minus = np.array([
        Cdmin + K * ((alpha_stall_minus + epsilon_cd) - alpha_Cdmin)**2,
        B1 * np.sin(alpha_stall_minus - epsilon_cd)**2 + B2_minus * np.cos(alpha_stall_minus - epsilon_cd),
        2 * K * ((alpha_stall_minus + epsilon_cd) - alpha_Cdmin),
        B1 * np.sin(2 * (alpha_stall_minus - epsilon_cd)) - B2_minus * np.sin(alpha_stall_minus - epsilon_cd),
    ], dtype='float')   
    coeff_Cd_minus = np.linalg.solve(mat_cd_minus,RHS_Cd_minus)



    return epsilon_plus, epsilon_minus, epsilon_cd, A1, B1, A2_plus, B2_plus, A2_minus, B2_minus, coeff_Cl_plus, coeff_Cl_minus, coeff_Cd_plus, coeff_Cd_minus