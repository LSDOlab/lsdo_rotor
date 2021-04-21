import numpy as np

import openmdao.api as om 

from lsdo_rotor.rotor_parameters import RotorParameters

class SmoothingExplicitComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types = tuple)
        self.options.declare('rotor', types = RotorParameters)

    def setup(self):
        shape = self.options['shape']
        rotor = self.options['rotor']

        self.add_output('_eps_plus')
        self.add_output('_eps_minus')
        self.add_output('_eps_cd')
        self.add_output('_coeff_Cl_plus', shape = (4,))
        self.add_output('_coeff_Cl_minus', shape = (4,))
        self.add_output('_coeff_Cd_plus', shape = (4,))
        self.add_output('_coeff_Cd_minus', shape = (4,))
        self.add_output('_A1')
        self.add_output('_B1')
        self.add_output('_A2_plus')
        self.add_output('_B2_plus')
        self.add_output('_A2_minus')
        self.add_output('_B2_minus')

    # def setup_partials(self):
    #     self.declare_partials('*','*')

    def compute(self, inputs, outputs):
        shape = self.options['shape']
        rotor = self.options['rotor']

        Cl_stall_plus     = rotor['Cl_stall_plus']
        Cd_stall_plus     = rotor['Cd_stall_plus']
        Cl_stall_minus    = rotor['Cl_stall_minus']
        Cd_stall_minus    = rotor['Cd_stall_minus']
        AR                = rotor['AR']
        alpha_stall_plus  = rotor['a_stall_plus']
        alpha_stall_minus = rotor['a_stall_minus']
        Cl0               = rotor['Cl0']
        Cla               = rotor['Cla']
        K                 = rotor['K']
        Cdmin             = rotor['Cdmin']
        alpha_Cdmin       = rotor['alpha_Cdmin']

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


        # print(epsilon_plus * 180/np.pi,  'eps_plus')
        # print(epsilon_guess_plus * 180/np.pi)
        # print(epsilon_minus * 180/np.pi, 'eps_minus')
        # print(epsilon_guess_minus * 180/np.pi)
        # print(coeff_Cl_minus)
        
        outputs['_eps_plus']        = epsilon_plus
        outputs['_eps_minus']       = epsilon_minus
        outputs['_eps_cd']          = epsilon_cd
        outputs['_coeff_Cl_minus']  = coeff_Cl_minus
        outputs['_coeff_Cl_plus']   = coeff_Cl_plus
        outputs['_coeff_Cd_minus']  = coeff_Cd_minus
        outputs['_coeff_Cd_plus']   = coeff_Cd_plus
        outputs['_A1']              = A1
        outputs['_B1']              = B1
        outputs['_A2_plus']         = A2_plus
        outputs['_B2_plus']         = B2_plus
        outputs['_A2_minus']        = A2_minus
        outputs['_B2_minus']        = B2_minus
        

# class SmoothingExplicitComponent(om.ExplicitComponent):

#     def initialize(self):
#         self.options.declare('shape', types = tuple)
#         self.options.declare('rotor', types = RotorParameters)

#     def setup(self):
#         shape = self.options['shape']
#         rotor = self.options['rotor']



#         self.add_input('_Cl0', shape = shape)
#         self.add_input('_Cla', shape = shape)
#         self.add_input('_Cdmin', shape = shape)
#         self.add_input('_K', shape = shape)
#         self.add_input('_alpha_Cdmin', shape = shape)
#         self.add_input('_Cl_stall', shape = shape)
#         self.add_input('_Cd_stall', shape = shape)
#         self.add_input('_Cl_stall_minus', shape = shape)
#         self.add_input('_Cd_stall_minus', shape = shape)
#         self.add_input('_alpha_stall', shape = shape)
#         self.add_input('_alpha_stall_minus', shape = shape)
#         self.add_input('_AR', shape = shape)


#         self.add_output('_eps_plus', shape = shape)
#         self.add_output('_eps_minus', shape = shape)
#         self.add_output('_eps_cd, shape = shape')
#         self.add_output('_coeff_Cl_plus', shape = shape + (4,))
#         self.add_output('_coeff_Cl_minus', shape = shape + (4,))
#         self.add_output('_coeff_Cd_plus', shape = shape +( 4,))
#         self.add_output('_coeff_Cd_minus', shape = shape + (4,))
#         self.add_output('_A1', shape = shape)
#         self.add_output('_B1', shape = shape)
#         self.add_output('_A2_plus', shape = shape)
#         self.add_output('_B2_plus', shape = shape)
#         self.add_output('_A2_minus', shape = shape)
#         self.add_output('_B2_minus', shape = shape)

#     def setup_partials(self):
#         self.declare_partials('*','*')

#     def compute(self, inputs, outputs):
#         shape = self.options['shape']
#         rotor = self.options['rotor']

#         Cl_stall_plus     = inputs['_Cl_stall'].flatten()
#         print(Cl_stall_plus)
#         Cd_stall_plus     = inputs['_Cd_stall'].flatten()
#         Cl_stall_minus    = inputs['_Cl_stall_minus'].flatten()
#         Cd_stall_minus    = inputs['_Cd_stall_minus'].flatten()
#         AR                = inputs['_AR'].flatten()
#         alpha_stall_plus  = inputs['_alpha_stall'].flatten()
#         alpha_stall_minus = inputs['_alpha_stall_minus'].flatten()
#         Cl0               = inputs['_Cl0'].flatten()
#         Cla               = inputs['_Cla'].flatten()
#         K                 = inputs['_K'].flatten()
#         Cdmin             = inputs['_Cdmin'].flatten()
#         alpha_Cdmin       = inputs['_alpha_Cdmin'].flatten()

#         Cd_max = 1.11 + 0.018 * AR 
#         A1 = Cd_max / 2
#         B1 = Cd_max
#         A2_plus = (Cl_stall_plus - Cd_max * np.sin(alpha_stall_plus) * np.cos(alpha_stall_plus)) * np.sin(alpha_stall_plus) / (np.cos(alpha_stall_plus)**2)
#         B2_plus = (Cd_stall_plus - Cd_max * np.sin(alpha_stall_plus)**2) / np.cos(alpha_stall_plus)
#         A2_minus = (Cl_stall_minus - Cd_max * np.sin(alpha_stall_minus) * np.cos(alpha_stall_minus)) * np.sin(alpha_stall_minus) / (np.cos(alpha_stall_minus)**2)
#         B2_minus = (Cd_stall_minus - Cd_max * np.sin(alpha_stall_minus)**2) / np.cos(alpha_stall_minus)

#         condition_1 = True
#         epsilon_guess_plus      = 5 * np.pi / 180
#         epsilon_guess_minus     = 4 * np.pi / 180
#         epsilon_cd              = 9 * np.pi/180
#         eps_step                = 0.01 * np.pi/180
#         tol                     = 1e-3
        
#         while condition_1:
#             mat_plus = np.array([
#                 [(alpha_stall_plus - epsilon_guess_plus)**3, (alpha_stall_plus - epsilon_guess_plus)**2, (alpha_stall_plus - epsilon_guess_plus), 1 ],
#                 [(alpha_stall_plus + epsilon_guess_plus)**3, (alpha_stall_plus + epsilon_guess_plus)**2, (alpha_stall_plus + epsilon_guess_plus), 1 ],
#                 [3 * (alpha_stall_plus - epsilon_guess_plus)**2, 2 * (alpha_stall_plus - epsilon_guess_plus), 1, 0 ],
#                 [3 * (alpha_stall_plus + epsilon_guess_plus)**2, 2 * (alpha_stall_plus + epsilon_guess_plus), 1, 0 ],
#             ], dtype='float')

#             RHS_Cl_plus = np.array([
#                 Cl0 + Cla * (alpha_stall_plus - epsilon_guess_plus),
#                 A1 * np.sin(2 * (alpha_stall_plus + epsilon_guess_plus)) + A2_plus * np.cos(alpha_stall_plus + epsilon_guess_plus)**2 / np.sin(alpha_stall_plus + epsilon_guess_plus),
#                 Cla,
#                 2 *  A1 * np.cos(2 * (alpha_stall_plus + epsilon_guess_plus)) - A2_plus * (np.cos(alpha_stall_plus + epsilon_guess_plus) + np.cos(alpha_stall_plus + epsilon_guess_plus) / np.sin(alpha_stall_plus + epsilon_guess_plus)**2),
#             ], dtype='float')
#             coeff_Cl_plus = np.linalg.solve(mat_plus,RHS_Cl_plus)

#             alpha_plus = np.linspace(alpha_stall_plus - epsilon_guess_plus, alpha_stall_plus + epsilon_guess_plus, 200) 
#             Cl_smoothing = np.zeros((len(alpha_plus)))

#             for i in range(len(alpha_plus)):
#                 Cl_smoothing[i] = coeff_Cl_plus[0] * alpha_plus[i]**3 + coeff_Cl_plus[1] * alpha_plus[i]**2 + coeff_Cl_plus[2] * alpha_plus[i] + coeff_Cl_plus[3]
    
#             Cl_max = np.max(Cl_smoothing)
#             # print(Cl_max - Cl_stall_plus)
#             if np.abs(Cl_max - Cl_stall_plus) > tol:
#                 if Cl_max > Cl_stall_plus:
#                     epsilon_guess_plus = epsilon_guess_plus + eps_step
#                 elif Cl_max < Cl_stall_plus:
#                     epsilon_guess_plus = epsilon_guess_plus - eps_step
#             else:
#                 break
#             epsilon_plus = epsilon_guess_plus
#             condition_1 = (np.abs(Cl_max - Cl_stall_plus) > tol)

#         condition_2 = True
#         eps_step_2 = 0.05 * np.pi/180
        
#         while condition_2:
#             mat_minus = np.array([
#                 [(alpha_stall_minus + epsilon_guess_minus)**3, (alpha_stall_minus + epsilon_guess_minus)**2, (alpha_stall_minus + epsilon_guess_minus), 1 ],
#                 [(alpha_stall_minus - epsilon_guess_minus)**3, (alpha_stall_minus - epsilon_guess_minus)**2, (alpha_stall_minus - epsilon_guess_minus), 1 ],
#                 [3 * (alpha_stall_minus + epsilon_guess_minus)**2, 2 * (alpha_stall_minus + epsilon_guess_minus), 1, 0 ],
#                 [3 * (alpha_stall_minus - epsilon_guess_minus)**2, 2 * (alpha_stall_minus - epsilon_guess_minus), 1, 0 ],
#             ], dtype='float')

#             RHS_Cl_minus = np.array([
#                 Cl0 + Cla * (alpha_stall_minus + epsilon_guess_minus),
#                 A1 * np.sin(2 * (alpha_stall_minus - epsilon_guess_minus)) + A2_minus * np.cos(alpha_stall_minus - epsilon_guess_minus)**2 / np.sin(alpha_stall_minus - epsilon_guess_minus),
#                 Cla,
#                 2 *  A1 * np.cos(2 * (alpha_stall_minus - epsilon_guess_minus)) - A2_minus * (np.cos(alpha_stall_minus - epsilon_guess_minus) + np.cos(alpha_stall_minus - epsilon_guess_minus) / np.sin(alpha_stall_minus - epsilon_guess_minus)**2),
#             ], dtype='float')  
#             coeff_Cl_minus = np.linalg.solve(mat_minus,RHS_Cl_minus)
        
#             alpha_minus = np.linspace(alpha_stall_minus - epsilon_guess_minus, alpha_stall_minus + epsilon_guess_minus, 200) 
#             Cl_smoothing_minus = np.zeros((len(alpha_minus)))

#             for i in range(len(alpha_minus)):
#                 Cl_smoothing_minus[i] = coeff_Cl_minus[0] * alpha_minus[i]**3 + coeff_Cl_minus[1] * alpha_minus[i]**2 + coeff_Cl_minus[2] * alpha_minus[i] + coeff_Cl_minus[3]
    
#             da = np.abs(alpha_minus[1] - alpha_minus[0])
#             na = len(alpha_minus)
#             dClda = np.zeros(len(Cl_smoothing_minus))
#             dClda_diff = np.zeros(len(Cl_smoothing_minus))

#             for i in range(len(Cl_smoothing_minus)):
#                 if i == (na-1):
#                     dClda[i] =  (Cl_smoothing_minus[i] - Cl_smoothing_minus[i-1]) / da
#                     dClda_diff[i] = dClda[i] - dClda[i-1]
#                 else:
#                     dClda[i] =  (Cl_smoothing_minus[i+1] - Cl_smoothing_minus[i]) / da
#                     dClda_diff[i] = dClda[i+1] - dClda[i]

#             if any(dClda_diff < 0):
#                 epsilon_guess_minus = epsilon_guess_minus + eps_step_2

#             if epsilon_guess_minus >= (np.abs(alpha_stall_minus) + 2.5 * np.pi/180):
#                 condition_2 = False
            
#             if all(dClda_diff > 0):
#                 condition_2 = False

#             epsilon_minus = epsilon_guess_minus

#         mat_cd_plus = np.array([
#                 [(alpha_stall_plus - epsilon_cd)**3, (alpha_stall_plus - epsilon_cd)**2, (alpha_stall_plus - epsilon_cd), 1 ],
#                 [(alpha_stall_plus + epsilon_cd)**3, (alpha_stall_plus + epsilon_cd)**2, (alpha_stall_plus + epsilon_cd), 1 ],
#                 [3 * (alpha_stall_plus - epsilon_cd)**2, 2 * (alpha_stall_plus - epsilon_cd), 1, 0 ],
#                 [3 * (alpha_stall_plus + epsilon_cd)**2, 2 * (alpha_stall_plus + epsilon_cd), 1, 0 ],
#             ], dtype='float')

#         RHS_Cd_plus = np.array([
#             Cdmin + K * ((alpha_stall_plus - epsilon_cd) - alpha_Cdmin)**2,
#             B1 * np.sin(alpha_stall_plus + epsilon_cd)**2 + B2_plus * np.cos(alpha_stall_plus + epsilon_cd),
#             2 * K * ((alpha_stall_plus - epsilon_cd) - alpha_Cdmin),
#             B1 * np.sin(2 * (alpha_stall_plus + epsilon_cd)) - B2_plus * np.sin(alpha_stall_plus + epsilon_cd),
#         ], dtype='float')   
#         coeff_Cd_plus = np.linalg.solve(mat_cd_plus,RHS_Cd_plus)

#         mat_cd_minus = np.array([
#                 [(alpha_stall_minus + epsilon_cd)**3, (alpha_stall_minus + epsilon_cd)**2, (alpha_stall_minus + epsilon_cd), 1 ],
#                 [(alpha_stall_minus - epsilon_cd)**3, (alpha_stall_minus - epsilon_cd)**2, (alpha_stall_minus - epsilon_cd), 1 ],
#                 [3 * (alpha_stall_minus + epsilon_cd)**2, 2 * (alpha_stall_minus + epsilon_cd), 1, 0 ],
#                 [3 * (alpha_stall_minus - epsilon_cd)**2, 2 * (alpha_stall_minus - epsilon_cd), 1, 0 ],
#             ], dtype='float')

#         RHS_Cd_minus = np.array([
#             Cdmin + K * ((alpha_stall_minus + epsilon_cd) - alpha_Cdmin)**2,
#             B1 * np.sin(alpha_stall_minus - epsilon_cd)**2 + B2_minus * np.cos(alpha_stall_minus - epsilon_cd),
#             2 * K * ((alpha_stall_minus + epsilon_cd) - alpha_Cdmin),
#             B1 * np.sin(2 * (alpha_stall_minus - epsilon_cd)) - B2_minus * np.sin(alpha_stall_minus - epsilon_cd),
#         ], dtype='float')   
#         coeff_Cd_minus = np.linalg.solve(mat_cd_minus,RHS_Cd_minus)


#         print(epsilon_plus * 180/np.pi,  'eps_plus')
#         print(epsilon_guess_plus * 180/np.pi)
#         print(epsilon_minus * 180/np.pi, 'eps_minus')
#         print(epsilon_guess_minus * 180/np.pi)
#         print(coeff_Cl_minus)
        
#         outputs['_eps_plus']        = epsilon_plus
#         outputs['_eps_minus']       = epsilon_minus
#         outputs['_eps_cd']          = epsilon_cd
#         outputs['_coeff_Cl_minus']  = coeff_Cl_minus
#         outputs['_coeff_Cl_plus']   = coeff_Cl_plus
#         outputs['_coeff_Cd_minus']  = coeff_Cd_minus
#         outputs['_coeff_Cd_plus']   = coeff_Cd_plus
#         outputs['_A1']              = A1
#         outputs['_B1']              = B1
#         outputs['_A2_plus']         = A2_plus
#         outputs['_B2_plus']         = B2_plus
#         outputs['_A2_minus']        = A2_minus
#         outputs['_B2_minus']        = B2_minus

    


