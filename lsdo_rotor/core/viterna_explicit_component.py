import numpy as np 

import openmdao.api as om 
import scipy.sparse.linalg as splinalg


class ViternaExplicitComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)

    def setup(self):
        shape = self.options['shape']

        self.add_input('alpha_stall')
        self.add_input('alpha_stall_minus')
        self.add_input('_phi_BEMT', shape = shape)
        self.add_input('_pitch', shape = shape)
        self.add_input('_Cl_BEMT', shape = shape)
        self.add_input('_Cd_BEMT', shape = shape)
        self.add_input('Cl_stall')
        self.add_input('Cd_stall')
        self.add_input('Cl_stall_minus')
        self.add_input('Cd_stall_minus')
        self.add_input('AR')
        self.add_input('smoothing_tolerance')

        self.add_input('Cl0')
        self.add_input('Cla')
        self.add_input('Cdmin')
        self.add_input('K')
        self.add_input('alpha_Cdmin') 

        self.add_output('_true_Cl', shape = shape)
        self.add_output('_true_Cd', shape = shape)
        # self.add_output('coeff_plus', shape = (4,))

        

    def setup_partials(self):
        self.declare_partials('*','*')

    def compute(self, inputs, outputs):
        shape = self.options['shape']

        alpha_stall = inputs['alpha_stall']
        alpha_stall_minus = inputs['alpha_stall_minus']
        phi_BEMT = inputs['_phi_BEMT'].flatten()
        twist = inputs['_pitch'].flatten()
        Cl_BEMT = inputs['_Cl_BEMT'].flatten()
        Cd_BEMT = inputs['_Cd_BEMT'].flatten()
        Cl_stall = inputs['Cl_stall']
        # print(Cl_stall)
        Cd_stall = inputs['Cd_stall']
        Cl_stall_minus = inputs['Cl_stall_minus']
        Cd_stall_minus = inputs['Cd_stall_minus']
        AR = inputs['AR']
        eps = inputs['smoothing_tolerance']

        Cl0 = inputs['Cl0']
        Cla = inputs['Cla']
        Cdmin = inputs['Cdmin']
        K = inputs['K']
        alpha_Cdmin = inputs['alpha_Cdmin']

        alpha  = twist - phi_BEMT
        # print(alpha * 180 / np.pi)
        size = len(twist)
        Cl = np.empty(size)
        Cd = np.empty(size)

        Cd_max = 1.11 + 0.018 * AR 
        A1 = Cd_max / 2
        B1 = Cd_max
        A2_plus = (Cl_stall - Cd_max * np.sin(alpha_stall) * np.cos(alpha_stall)) * np.sin(alpha_stall) / (np.cos(alpha_stall)**2)
        B2_plus = (Cd_stall - Cd_max * np.sin(alpha_stall)**2) / np.cos(alpha_stall)
        A2_minus = (Cl_stall_minus - Cd_max * np.sin(alpha_stall_minus) * np.cos(alpha_stall_minus)) * np.sin(alpha_stall_minus) / (np.cos(alpha_stall_minus)**2)
        B2_minus = (Cd_stall_minus - Cd_max * np.sin(alpha_stall_minus)**2) / np.cos(alpha_stall_minus)

        condition_1 = True
        epsilon_guess_plus = 5 * np.pi / 180
        epsilon_guess_minus = 4 * np.pi / 180
        epsilon_cd = 9 * np.pi/180
        eps_step = 0.01 * np.pi/180
        tol = 1e-3
        
        while condition_1:
            mat_plus = np.array([
                [(alpha_stall - epsilon_guess_plus)**3, (alpha_stall - epsilon_guess_plus)**2, (alpha_stall - epsilon_guess_plus), 1 ],
                [(alpha_stall + epsilon_guess_plus)**3, (alpha_stall + epsilon_guess_plus)**2, (alpha_stall + epsilon_guess_plus), 1 ],
                [3 * (alpha_stall - epsilon_guess_plus)**2, 2 * (alpha_stall - epsilon_guess_plus), 1, 0 ],
                [3 * (alpha_stall + epsilon_guess_plus)**2, 2 * (alpha_stall + epsilon_guess_plus), 1, 0 ],
            ], dtype='float')

            RHS_Cl_plus = np.array([
                Cl0 + Cla * (alpha_stall - epsilon_guess_plus),
                A1 * np.sin(2 * (alpha_stall + epsilon_guess_plus)) + A2_plus * np.cos(alpha_stall + epsilon_guess_plus)**2 / np.sin(alpha_stall + epsilon_guess_plus),
                Cla,
                2 *  A1 * np.cos(2 * (alpha_stall + epsilon_guess_plus)) - A2_plus * (np.cos(alpha_stall + epsilon_guess_plus) + np.cos(alpha_stall + epsilon_guess_plus) / np.sin(alpha_stall + epsilon_guess_plus)**2),
            ], dtype='float')
            coeff_Cl_plus = np.linalg.solve(mat_plus,RHS_Cl_plus)

            alpha_plus = np.linspace(alpha_stall - epsilon_guess_plus, alpha_stall + epsilon_guess_plus, 200) 
            Cl_smoothing = np.zeros((len(alpha_plus)))

            for i in range(len(alpha)):
                Cl_smoothing[i] = coeff_Cl_plus[0] * alpha_plus[i]**3 + coeff_Cl_plus[1] * alpha_plus[i]**2 + coeff_Cl_plus[2] * alpha_plus[i] + coeff_Cl_plus[3]
    
            Cl_max = np.max(Cl_smoothing)
            # print(Cl_max - Cl_stall_plus)
            if np.abs(Cl_max - Cl_stall) > tol:
                if Cl_max > Cl_stall:
                    epsilon_guess_plus = epsilon_guess_plus + eps_step
                elif Cl_max < Cl_stall:
                    epsilon_guess_plus = epsilon_guess_plus - eps_step
            else:
                break
            epsilon_plus = epsilon_guess_plus
            condition_1 = (np.abs(Cl_max - Cl_stall) > tol)


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

            for i in range(len(alpha)):
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
    
        # print(epsilon_plus * 180 / np.pi,'eps_p')
        # print(epsilon_minus* 180 / np.pi,'eps_m')
        mat_cd_plus = np.array([
                [(alpha_stall - epsilon_cd)**3, (alpha_stall - epsilon_cd)**2, (alpha_stall - epsilon_cd), 1 ],
                [(alpha_stall + epsilon_cd)**3, (alpha_stall + epsilon_cd)**2, (alpha_stall + epsilon_cd), 1 ],
                [3 * (alpha_stall - epsilon_cd)**2, 2 * (alpha_stall - epsilon_cd), 1, 0 ],
                [3 * (alpha_stall + epsilon_cd)**2, 2 * (alpha_stall + epsilon_cd), 1, 0 ],
            ], dtype='float')

        RHS_Cd_plus = np.array([
            Cdmin + K * ((alpha_stall - epsilon_cd) - alpha_Cdmin)**2,
            B1 * np.sin(alpha_stall + epsilon_cd)**2 + B2_plus * np.cos(alpha_stall + epsilon_cd),
            2 * K * ((alpha_stall - epsilon_cd) - alpha_Cdmin),
            B1 * np.sin(2 * (alpha_stall + epsilon_cd)) - B2_plus * np.sin(alpha_stall + epsilon_cd),
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


        for i in range(size):
            if alpha[i] < (alpha_stall_minus - epsilon_minus):
                Cl[i] = A1 * np.sin(2 * alpha[i]) + A2_minus * np.cos(alpha[i])**2 / np.sin(alpha[i])           
            elif (alpha[i] >= (alpha_stall_minus - epsilon_minus)) and (alpha[i] <= (alpha_stall_minus + epsilon_minus)):
                Cl[i] = coeff_Cl_minus[0] * alpha[i]**3 + coeff_Cl_minus[1] * alpha[i]**2 + coeff_Cl_minus[2] * alpha[i] + coeff_Cl_minus[3] 
            elif ((alpha_stall - epsilon_plus) <= alpha[i]) and ((alpha_stall + epsilon_plus) >= alpha[i]):
                Cl[i] = coeff_Cl_plus[0] * alpha[i]**3 + coeff_Cl_plus[1] * alpha[i]**2 + coeff_Cl_plus[2] * alpha[i] + coeff_Cl_plus[3] 
            elif alpha[i] > (alpha_stall + epsilon_plus): 
                Cl[i] = A1 * np.sin(2 * alpha[i]) + A2_plus * np.cos(alpha[i])**2 / np.sin(alpha[i])
            else:
                Cl[i] = Cl0 + Cla * alpha[i]
            # else:
            #     Cl[i] = Cl_BEMT[i]

        for i in range(size):
            if alpha[i] < (alpha_stall_minus - epsilon_cd):
                Cd[i] = B1 * np.sin(alpha[i])**2 + B2_minus * np.cos(alpha[i])
            elif (alpha[i] >= (alpha_stall_minus - epsilon_cd)) and (alpha[i] <= (alpha_stall_minus + epsilon_cd)):
                Cd[i] = coeff_Cd_minus[0] * alpha[i]**3 + coeff_Cd_minus[1] * alpha[i]**2 + coeff_Cd_minus[2] * alpha[i] + coeff_Cd_minus[3] 
            elif ((alpha_stall - epsilon_cd) <= alpha[i]) and ((alpha_stall + epsilon_cd) >= alpha[i]):
                Cd[i] = coeff_Cd_plus[0] * alpha[i]**3 + coeff_Cd_plus[1] * alpha[i]**2 + coeff_Cd_plus[2] * alpha[i] + coeff_Cd_plus[3] 
            elif alpha[i] > (alpha_stall + epsilon_cd): 
                Cd[i] = B1 * np.sin(alpha[i])**2 + B2_plus * np.cos(alpha[i])
            else: 
                Cd[i] = Cdmin + K * (alpha[i] - alpha_Cdmin)**2
            # else:
                # Cd[i] = Cd_BEMT[i]

        # print(Cl)
        outputs['_true_Cl'] = Cl.reshape(shape)
        outputs['_true_Cd'] = Cd.reshape(shape)



