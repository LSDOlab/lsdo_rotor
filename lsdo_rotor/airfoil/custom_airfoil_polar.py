import numpy as np


class CustomAirfoilPolar:
    def __init__(self, airfoil_polar):
        # self.airfoil_polar = airfoil_polar
        # stall angle > 0
        self.aoa_stall_p = np.deg2rad(airfoil_polar['alpha_Cl_stall'][1])
        # Cl at stall > 0 
        Cl_stall_p = airfoil_polar['Cl_stall'][1]
        # Cd at stall
        Cd_stall_p = airfoil_polar['Cd_stall'][1]

        # stall angle < 0
        self.aoa_stall_m = np.deg2rad(airfoil_polar['alpha_Cl_stall'][0])
        # Cl at stall < 0
        Cl_stall_m = airfoil_polar['Cl_stall'][0]
        # Cd at stall
        Cd_stall_m = airfoil_polar['Cd_stall'][0]

        # Cl at zero angle of attack
        self.Cl_0 = airfoil_polar['Cl_0']
        # Lift curve slope
        self.Cl_alpha = airfoil_polar['Cl_alpha']
        # Cd at zero angle of attack
        self.Cd_0 = airfoil_polar['Cd_0']
        # K for quadratic lift polar
        self.k  = 0.5 * ((Cd_stall_p - self.Cd_0) / (self.Cl_0-Cl_stall_p)**2 + (Cd_stall_m - self.Cd_0) / (self.Cl_0-Cl_stall_m)**2)
      
        # Smoothing region 
        self.eps = np.deg2rad(1.5)

        # Viterna Extrapolation 
        AR = 10.
        Cd_max = 1.11 + 0.018 * AR
        self.A1 = Cd_max / 2
        self.B1 = Cd_max
        self.A2_p = (Cl_stall_p - Cd_max * np.sin(self.aoa_stall_p) * np.cos(self.aoa_stall_p)) * np.sin(self.aoa_stall_p) / (np.cos(self.aoa_stall_p)**2)
        self.A2_m = (Cl_stall_m - Cd_max * np.sin(self.aoa_stall_m) * np.cos(self.aoa_stall_m)) * np.sin(self.aoa_stall_m) / (np.cos(self.aoa_stall_m)**2)
        self.B2_p = (Cd_stall_p - Cd_max * np.sin(self.aoa_stall_p)**2) / np.cos(self.aoa_stall_p)
        self.B2_m = (Cd_stall_m - Cd_max * np.sin(self.aoa_stall_m)**2) / np.cos(self.aoa_stall_m)

        # Polynomial Smoothing alpha > 0 
        mat_cl_p = mat_cd_p =  np.array([
            [(self.aoa_stall_p-self.eps)**3, (self.aoa_stall_p-self.eps)**2, (self.aoa_stall_p-self.eps), 1],
            [(self.aoa_stall_p+self.eps)**3, (self.aoa_stall_p+self.eps)**2, (self.aoa_stall_p+self.eps), 1],
            [3 * (self.aoa_stall_p-self.eps)**2, 2*(self.aoa_stall_p-self.eps), 1, 0],
            [3 * (self.aoa_stall_p+self.eps)**2, 2*(self.aoa_stall_p+self.eps), 1, 0],
        ])

        lhs_cl_p = np.array([
            [self.Cl_0 + self.Cl_alpha * (self.aoa_stall_p-self.eps)],
            [ self.A1 * np.sin(2 * (self.aoa_stall_p+self.eps)) + self.A2_p * np.cos(self.aoa_stall_p+self.eps)**2 / np.sin(self.aoa_stall_p+self.eps)],
            [self.Cl_alpha],
            [2 * self.A1 * np.cos(2 * (self.aoa_stall_p+self.eps)) - self.A2_p * (np.cos(self.aoa_stall_p+self.eps) * (1+1/(np.sin(self.aoa_stall_p+self.eps))**2))],
        ])
        self.coeff_cl_p = np.linalg.solve(mat_cl_p, lhs_cl_p)

        lhs_cd_p = np.array([
            [self.Cd_0 + self.k * (self.Cl_alpha * (self.aoa_stall_p-self.eps))**2],
            [self.B1 * np.sin(self.aoa_stall_p+self.eps)**2 + self.B2_p * np.cos(self.aoa_stall_p+self.eps)],
            [2 * self.k * self.Cl_alpha * (self.Cl_alpha * (self.aoa_stall_p-self.eps))],
            [self.B1 * np.sin(2 * (self.aoa_stall_p+self.eps)) - self.B2_p * np.sin(self.aoa_stall_p+self.eps)],
        ])
        self.coeff_cd_p = np.linalg.solve(mat_cd_p, lhs_cd_p)

        # Polynomial Smoothing alpha < 0 
        mat_cl_m = mat_cd_m =  np.array([
            [(self.aoa_stall_m-self.eps)**3, (self.aoa_stall_m-self.eps)**2, (self.aoa_stall_m-self.eps), 1],
            [(self.aoa_stall_m+self.eps)**3, (self.aoa_stall_m+self.eps)**2, (self.aoa_stall_m+self.eps), 1],
            [3 * (self.aoa_stall_m-self.eps)**2, 2*(self.aoa_stall_m-self.eps), 1, 0],
            [3 * (self.aoa_stall_m+self.eps)**2, 2*(self.aoa_stall_m+self.eps), 1, 0],
        ])

        lhs_cl_m = np.array([
            [self.A1 * np.sin(2 * (self.aoa_stall_m-self.eps)) + self.A2_m * np.cos(self.aoa_stall_m-self.eps)**2 / np.sin(self.aoa_stall_m-self.eps)],
            [self.Cl_0 + self.Cl_alpha * (self.aoa_stall_m+self.eps)],
            [2 * self.A1 * np.cos(2 * (self.aoa_stall_m-self.eps)) - self.A2_m * (np.cos(self.aoa_stall_m-self.eps) * (1+1/(np.sin(self.aoa_stall_m-self.eps))**2))],
            [self.Cl_alpha],
        ])
        self.coeff_cl_m = np.linalg.solve(mat_cl_m, lhs_cl_m)

        lhs_cd_m = np.array([
            [self.B1 * np.sin(self.aoa_stall_m-self.eps)**2 + self.B2_m * np.cos(self.aoa_stall_m-self.eps)],
            [self.Cd_0 + self.k * (self.Cl_alpha * (self.aoa_stall_m+self.eps))**2],
            [self.B1 * np.sin(2 * (self.aoa_stall_m-self.eps)) - self.B2_m * np.sin(self.aoa_stall_m-self.eps)],
            [2 * self.k * self.Cl_alpha * (self.Cl_alpha * (self.aoa_stall_m+self.eps))],
        ])
        self.coeff_cd_m = np.linalg.solve(mat_cd_m, lhs_cd_m)
    
    def predict_values(self, AoA_array):
        aoa = AoA_array.flatten()
        cond_list = [
            aoa <= (self.aoa_stall_m-self.eps),
            (aoa > (self.aoa_stall_m-self.eps)) & (aoa <= (self.aoa_stall_m+self.eps)),
            (aoa > (self.aoa_stall_m+self.eps)) & (aoa <= (self.aoa_stall_p-self.eps)),
            (aoa > (self.aoa_stall_p-self.eps)) & (aoa <= (self.aoa_stall_p+self.eps)),
            aoa > (self.aoa_stall_p+self.eps)
        ]
        
        Cl_fun_list = [
            lambda aoa : self.A1 * np.sin(2 * aoa) + self.A2_m * np.cos(aoa)**2 / np.sin(aoa),
            lambda aoa : self.coeff_cl_m[3] + self.coeff_cl_m[2] * aoa + self.coeff_cl_m[1] * aoa**2 + self.coeff_cl_m[0] * aoa**3,
            lambda aoa : self.Cl_0 + self.Cl_alpha * aoa,
            lambda aoa : self.coeff_cl_p[3] + self.coeff_cl_p[2] * aoa + self.coeff_cl_p[1] * aoa**2 + self.coeff_cl_p[0] * aoa**3,
            lambda aoa : self.A1 * np.sin(2 * aoa) + self.A2_p * np.cos(aoa)**2 / np.sin(aoa),
        ]

        Cd_fun_list = [
            lambda aoa : self.B1 * np.sin(aoa)**2 + self.B2_m * np.cos(aoa),
            lambda aoa : self.coeff_cd_m[3] + self.coeff_cd_m[2] * aoa + self.coeff_cd_m[1] * aoa**2 + self.coeff_cd_m[0] * aoa**3,
            lambda aoa : self.Cd_0 + self.k * (self.Cl_alpha * aoa)**2,
            lambda aoa : self.coeff_cd_p[3] + self.coeff_cd_p[2] * aoa + self.coeff_cd_p[1] * aoa**2 + self.coeff_cd_p[0] * aoa**3,
            lambda aoa : self.B1 * np.sin(aoa)**2 + self.B2_p * np.cos(aoa),
        ]    
        
        Cl = np.piecewise(
            aoa, 
            cond_list, 
            Cl_fun_list,
        )

        Cd = np.piecewise(
            aoa, 
            cond_list, 
            Cd_fun_list
        )

        return Cl, Cd

    def predict_derivatives(self, AoA_array):
        aoa = AoA_array.flatten()
        cond_list = [
            aoa <= (self.aoa_stall_m-self.eps),
            (aoa > (self.aoa_stall_m-self.eps)) & (aoa <= (self.aoa_stall_m+self.eps)),
            (aoa > (self.aoa_stall_m+self.eps)) & (aoa <= (self.aoa_stall_p-self.eps)),
            (aoa > (self.aoa_stall_p-self.eps)) & (aoa <= (self.aoa_stall_p+self.eps)),
            aoa > (self.aoa_stall_p+self.eps)
        ]

        dCl_daoa_fun_list = [
            lambda aoa : 2 * self.A1 * np.cos(2 * aoa) - self.A2_m * (np.cos(aoa) * (1+1/(np.sin(aoa))**2)),
            lambda aoa : self.coeff_cl_m[2] + 2 * self.coeff_cl_m[1] * aoa + 3 * self.coeff_cl_m[0] * aoa**2,
            lambda aoa : self.Cl_alpha,
            lambda aoa : self.coeff_cl_p[2] + 2 * self.coeff_cl_p[1] * aoa + 3 * self.coeff_cl_p[0] * aoa**2,
            lambda aoa : 2 * self.A1 * np.cos(2 * aoa) - self.A2_p * (np.cos(aoa) * (1+1/(np.sin(aoa))**2)),
        ]

        dCd_daoa_fun_list = [
            lambda aoa : self.B1 * np.sin(2 * aoa) - self.B2_m * np.sin(aoa),
            lambda aoa : self.coeff_cd_m[2] + 2 * self.coeff_cd_m[1] * aoa + 3 * self.coeff_cd_m[0] * aoa**2,
            lambda aoa : 2 * self.k * self.Cl_alpha * (self.Cl_alpha * aoa),
            lambda aoa : self.coeff_cd_p[2] + 2 * self.coeff_cd_p[1] * aoa + 3 * self.coeff_cd_p[0] * aoa**2,
            lambda aoa : self.B1 * np.sin(2 * aoa) - self.B2_p * np.sin(aoa),
        ]    

        dCl_daoa = np.piecewise(
            aoa,
            cond_list,
            dCl_daoa_fun_list,
        )

        dCd_daoa = np.piecewise(
            aoa,
            cond_list,
            dCd_daoa_fun_list,
        )

        return dCl_daoa, dCd_daoa