import numpy as np 
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl 
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.airfoil.pitt_peters_airfoil_model import PittPetersAirfoilModel
from lsdo_rotor.core.pitt_peters_aero_coeff_group import PittPetersAeroCoeffGroup
from lsdo_rotor.core.pitt_peters_implicit_component import PittPetersImplicitComponent
import openmdao.api as om 

# Custom implicit operation for nu_0, nu_s, nu_c
# Explicit component for quantities of interest (T,Q,etc)



# class PittPetersGroup(Model):
#     def initialize(self):
#         self.parameters.declare('shape', types=tuple)
#         self.parameters.declare('rotor', types=RotorParameters)

#     def define(self):
#         shape = self.parameters['shape']
#         rotor = self.parameters['rotor']

#         B = rotor['num_blades']

#         # model = Model()

#         # BET inputs
#         Re = self.declare_variable('_re_pitt_peters', shape=shape)
#         rho = self.declare_variable('_rho_pitt_peters', shape=shape)
#         chord = self.declare_variable('_chord',shape=shape)
#         twist = self.declare_variable('_pitch', shape=shape)
#         norm_radius = self.declare_variable('_normalized_radius', shape=shape)
#         radius = self.declare_variable('_radius', shape=shape)
#         Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
#         psi = self.declare_variable('_theta', shape=shape)
#         angular_speed = self.declare_variable('_angular_speed', shape=shape)
#         rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
#         dr = self.declare_variable('_dr', shape=shape)
#         sigma = self.declare_variable('_blade_solidity', shape=shape)

#         # Matrices
#         L = self.declare_variable('L_matrix', shape = (shape[0],3,3))
#         L_inv = self.declare_variable('inv_L_matrix', shape = (shape[0],3,3))
#         M = self.declare_variable('M_matrix', shape = (shape[0],3,3))
#         M_inv = self.declare_variable('inv_M_matrix', shape = (shape[0],3,3))
        
#         nu_vec = self.declare_variable('nu_vec', shape = (shape[0],3))
       

#         for j in range(3):
#             nu_0_string = 'nu_0_{}'.format(j)
#             nu_s_string = 'nu_s_{}'.format(j)
#             nu_c_string = 'nu_c_{}'.format(j)
#             nu_0 = self.create_output(nu_0_string, shape = (shape[0],1))
#             nu_s = self.create_output(nu_s_string, shape = (shape[0],1))
#             nu_c = self.create_output(nu_c_string, shape = (shape[0],1))

#             print(j)
#             self.print_var(nu_vec)

#             for i in range(shape[0]):
#                 nu_0[i,0] = nu_vec[i,0]
#                 nu_s[i,0] = nu_vec[i,1]
#                 nu_c[i,0] = nu_vec[i,2]

#             nu_0_exp = csdl.expand(csdl.reshape(nu_0,new_shape=(shape[0], )), shape, 'i->ijk')        
#             nu_s_exp = csdl.expand(csdl.reshape(nu_s,new_shape=(shape[0], )), shape, 'i->ijk')
#             nu_c_exp = csdl.expand(csdl.reshape(nu_c,new_shape=(shape[0], )), shape, 'i->ijk')

#             nu =  nu_0_exp + nu_s_exp * norm_radius * csdl.sin(psi) + nu_c_exp * norm_radius * csdl.cos(psi)
#             ux = nu * angular_speed * rotor_radius

#             phi = csdl.arctan(ux/Vt)
#             phi_output_string = 'phi_pitt_peters_{}'.format(j)
#             self.register_output(phi_output_string, phi)
            
            
#             alpha = twist - phi 
#             AoA_output_string = 'AoA_pitt_peters'#_{}'.format(j)
#             self.register_output(AoA_output_string, alpha)

#             airfoil_model_output = csdl.custom(Re,alpha,chord, op= PittPetersAirfoilModel(
#                     rotor=rotor,
#                     shape=shape,
#                 ))
#             # self.register_output('Cl_pitt_peters',airfoil_model_output[0])
#             # self.register_output('Cd_pitt_peters',airfoil_model_output[1])

#             Cl = airfoil_model_output[0]
#             Cd = airfoil_model_output[1]

#             Cx = (Cl * csdl.cos(phi) - Cd * csdl.sin(phi))
#             Ct = (Cl * csdl.sin(phi) + Cd * csdl.cos(phi))

#             dT = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Cx * dr
#             T = csdl.sum(dT, axes = (1,2)) / shape[2]
#             dQ = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Ct * radius * dr
#             Q = csdl.sum(dQ, axes = (1,2)) / shape[2]

#             dL_mom = radius * csdl.sin(psi) * dT
#             L_mom  = csdl.sum(dL_mom, axes = (1,2))  / shape[2]
#             dM_mom = radius * csdl.cos(psi) * dT
#             M_mom  = csdl.sum(dM_mom, axes = (1,2)) / shape[2]

#             dC_T = dT / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**4
#             dC_L = dL_mom / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5
#             dC_M = dM_mom / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5

#             C_T = csdl.reshape(csdl.sum(dC_T, axes = (1,2))  / shape[2], new_shape = (shape[0],1))
#             C_L = csdl.reshape(csdl.sum(dC_L, axes = (1,2))  / shape[2], new_shape = (shape[0],1))
#             C_M = csdl.reshape(csdl.sum(dC_M, axes = (1,2))  / shape[2], new_shape = (shape[0],1))

#             C_string = 'aero_coeff_{}'.format(j)
#             C = self.create_output(C_string, shape=(shape[0],3))
#             for k in range(shape[0]):
#                 C[k,0] = C_T[k,0]
#                 C[k,1] = -C_L[k,0]
#                 C[k,2] = -C_M[k,0]           
    
#             C_new = csdl.reshape(C,new_shape=(shape[0],3,1))


        
#             term1 = csdl.einsum(L,M, subscripts = 'ijk,ikl->ijl')
#             term2 = csdl.einsum(M_inv,C_new, subscripts = 'ijk,ikl->ijl')
#             term3 = csdl.einsum(M_inv,L_inv,subscripts = 'ijk,ikl->ijl')
#             term4 = csdl.einsum(term3,csdl.reshape(nu_vec,new_shape =(shape[0],3,1)), subscripts=  'ijk,ikl->ijl')
#             term5 = term2 - term4
#             nu_vec_new = csdl.reshape(nu_vec, new_shape = (shape[0],3,1)) + csdl.einsum(term1,term5, subscripts=  'ijk,ikl->ijl')
#             nu_vec = csdl.reshape(nu_vec_new, new_shape = (shape[0],3))
            
#         self.register_output('nu_state_vec',nu_vec)
        # exit()
        # break
        

        # term7 = csdl.reshape(term6,new_shape = (shape[0],3))
        
        




class PittPetersGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=RotorParameters)

    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']

        B = rotor['num_blades']

        model = Model()

        # BET inputs
        Re = model.declare_variable('_re_pitt_peters', shape=shape)
        rho = model.declare_variable('_rho_pitt_peters', shape=shape)
        chord = model.declare_variable('_chord',shape=shape)
        twist = model.declare_variable('_pitch', shape=shape)
        norm_radius = model.declare_variable('_normalized_radius', shape=shape)
        Vt = model.declare_variable('_tangential_inflow_velocity', shape=shape)
        psi = model.declare_variable('_theta', shape=shape)
        angular_speed = model.declare_variable('_angular_speed', shape=shape)
        rotor_radius = model.declare_variable('_rotor_radius', shape= shape)
        dr = model.declare_variable('_dr', shape=shape)
        # sigma = model.declare_variable('_blade_solidity', shape=shape)
        radius = model.declare_variable('_radius', shape=shape)

        # Matrices
        L = model.declare_variable('L_matrix', shape = (shape[0],3,3))
        # model.print_var(L)
        L_inv = model.declare_variable('inv_L_matrix', shape = (shape[0],3,3))
        self.print_var(L_inv)
        # model.print_var(L_inv)
        M = model.declare_variable('M_matrix', shape = (shape[0],3,3))
        M_inv = model.declare_variable('inv_M_matrix', shape = (shape[0],3,3))
        

        # Declaring state variable to be computed from residual
        nu_vec = model.declare_variable('nu_state_vec', val =  np.zeros((shape[0],3,1)))
        nu_0 = self.create_output('nu_0', shape = (shape[0],1,1))
        nu_s = self.create_output('nu_s', shape = (shape[0],1,1))
        nu_c = self.create_output('nu_c', shape = (shape[0],1,1))
        for i in range(shape[0]):
            nu_0[i,0,0] = nu_vec[i,0,0]
            nu_s[i,0,0] = nu_vec[i,1,0]
            nu_c[i,0,0] = nu_vec[i,2,0]

        nu_0_exp = csdl.expand(csdl.reshape(nu_0,new_shape=(shape[0], )), shape, 'i->ijk')        
        nu_s_exp = csdl.expand(csdl.reshape(nu_s,new_shape=(shape[0], )), shape, 'i->ijk')
        nu_c_exp = csdl.expand(csdl.reshape(nu_c,new_shape=(shape[0], )), shape, 'i->ijk')

        nu =  nu_0_exp + nu_s_exp * norm_radius * csdl.sin(psi) + nu_c_exp * norm_radius * csdl.cos(psi)
        ux = nu * angular_speed * rotor_radius

        phi = csdl.arctan(ux/Vt)
        alpha = twist - phi 
        model.register_output('AoA_pitt_peters', alpha)


        airfoil_model_output = csdl.custom(Re,alpha,chord, op= PittPetersAirfoilModel(
                rotor=rotor,
                shape=shape,
            ))
        model.register_output('Cl_pitt_peters',airfoil_model_output[0])
        model.register_output('Cd_pitt_peters',airfoil_model_output[1])

        Cl = airfoil_model_output[0]
        Cd = airfoil_model_output[1]

        Cx = (Cl * csdl.cos(phi) - Cd * csdl.sin(phi))
        Ct = (Cl * csdl.sin(phi) + Cd * csdl.cos(phi))

        dT = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Cx * dr
        T = csdl.sum(dT, axes = (1,2)) / shape[2]
        dQ = 0.5 * B * rho * (ux**2 + (Vt)**2) * chord * Ct * radius * dr
        Q = csdl.sum(dQ, axes = (1,2)) / shape[2]

        dL_mom = radius * csdl.sin(psi) * dT
        L_mom  = csdl.sum(dL_mom, axes = (1,2))  / shape[2]
        dM_mom = radius * csdl.cos(psi) * dT
        M_mom  = csdl.sum(dM_mom, axes = (1,2)) / shape[2]

         # Compute coefficients 
        dC_T = dT / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**4
        dC_L = dL_mom / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5
        dC_M = dM_mom / rho / (angular_speed / 2 / np.pi)**2 / (rotor_radius * 2)**5

        C_T = csdl.reshape(csdl.sum(dC_T, axes = (1,2))  / shape[2], new_shape = (shape[0],1))
        C_L = csdl.reshape(csdl.sum(dC_L, axes = (1,2))  / shape[2], new_shape = (shape[0],1))
        C_M = csdl.reshape(csdl.sum(dC_M, axes = (1,2))  / shape[2], new_shape = (shape[0],1))
        
        C = self.create_output('aero_coeff', shape=(shape[0],3))
        for i in range(shape[0]):
            C[i,0] = C_T[i,0]
            C[i,1] = -C_L[i,0]
            C[i,2] = -C_M[i,0]

        C_new = csdl.reshape(C,new_shape=(shape[0],3,1))
        
        
        term1 = 1 * csdl.einsum(L,M, subscripts='ijk,ikl->ijl')
        term2 = csdl.einsum(M_inv,C_new, subscripts='ijk,ikl->ijl')
        term3 = csdl.einsum(M_inv,L_inv,subscripts='ijk,ikl->ijl')
        term4 = csdl.einsum(term3,nu_vec, subscripts='ijk,ikl->ijl')
        term5 = term2 - term4        
        
        term6 =  csdl.einsum(term1,term5, subscripts='ijk,ikl->ijl')        

        model.register_output('pitt_peters_residual', term6)
        model.register_output('C', C_new)
        # ode_solver = csdl.custom(L,L_inv,M,M_inv,C_new, op=PittPetersImplicitComponent(
        #     rotor=rotor,
        #     shape=shape,
        # ))
        # self.register_output('nu',ode_solver)


        
        solve_residual = self.create_implicit_operation(model)
        solve_residual.declare_state('nu_state_vec', residual = 'pitt_peters_residual',bracket = (-5 * np.ones((shape[0],3,1)), 5 * np.ones((shape[0],3,1)) ))
        # solve_residual.nonlinear_solver = NewtonSolver(
        #         solve_subsystems=False,
        #         maxiter=100,
        #         iprint=True,
        #     )
        # solve_residual.linear_solver = ScipyKrylov()

        Re = self.declare_variable('_re_pitt_peters', shape=shape)
        rho = self.declare_variable('_rho_pitt_peters', shape=shape)
        chord = self.declare_variable('_chord',shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)
        norm_radius = self.declare_variable('_normalized_radius', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        psi = self.declare_variable('_theta', shape=shape)
        angular_speed = self.declare_variable('_angular_speed', shape=shape)
        rotor_radius = self.declare_variable('_rotor_radius', shape= shape)
        dr = self.declare_variable('_dr', shape=shape)
        radius = self.declare_variable('_radius', shape=shape)
        L = self.declare_variable('L_matrix', shape = (shape[0],3,3))
        L_inv = self.declare_variable('inv_L_matrix', shape = (shape[0],3,3))
        M = self.declare_variable('M_matrix', shape = (shape[0],3,3))
        M_inv = self.declare_variable('inv_M_matrix', shape = (shape[0],3,3))
 


        nu_vec = solve_residual(Re, rho, chord,twist, norm_radius,Vt,psi, angular_speed,rotor_radius,dr,radius, L, L_inv,M, M_inv)# C_T, C_L, C_M)
        # nu_vec = solve_residual(nu_vec,L, L_inv,M, M_inv, C_T, C_L, C_M)

        
    








        


# class PittPetersGroup(csdl.CustomExplicitOperation):
#     def initialize(self):
#         self.parameters.declare('shape', types=tuple)
#         self.parameters.declare('rotor', types=RotorParameters)

#     def define(self):
#         shape = self.parameters['shape']
#         rotor = self.parameters['rotor']

#         # Adding inputs 
#         self.add_input('_re_pitt_peters', shape=shape)
#         self.add_input('_rho_pitt_peters', shape=shape)
#         self.add_input('_chord', shape=shape)
#         self.add_input('_pitch', shape=shape)
#         self.add_input('_normalized_radius', shape=shape)
#         self.add_input('_tangential_inflow_velocity', shape=shape)
#         self.add_input('_theta', shape=shape)
#         self.add_input('_angular_speed', shape=shape)
#         self.add_input('_rotor_radius', shape=shape)
#         self.add_input('_dr', shape=shape)
#         self.add_input('_blade_solidity', shape=shape)


#         self.airfoil_model_inputs = np.zeros((shape[0] * shape[1] * shape[2], 2))

#         # self.add_input('nu_0')
#         # self.add_input('nu_s')
#         # self.add_input('nu_c')


#         # Adding outputs 
#         self.add_output('total_thrust')
#         self.add_output('total_torque')
#         self.add_output('pitching_moment')
#         self.add_output('rolling_moment')
#         # self.add_output('C_L')

#     def compute(self, inputs,outputs):
#         shape       = self.parameters['shape']
#         rotor       = self.parameters['rotor']
#         interp      = rotor['interp']
#         beta        = rotor['rotor_disk_tilt_angle'] * np.pi / 180
#         B           = rotor['num_blades']


#         Re          = inputs['_re_pitt_peters']
#         rho         = inputs['_rho_pitt_peters']
#         chord       = inputs['_chord']
#         twist       = inputs['_pitch']
#         radius      = inputs['_normalized_radius']
#         Vt          = inputs['_tangential_inflow_velocity']
#         psi         = inputs['_theta']
#         Omega       = inputs['_angular_speed']
#         R           = inputs['_rotor_radius']
#         dr          = inputs['_dr']
#         sigma       = inputs['_blade_solidity']

#         # print(Re[0,:,0])
#         # print(rho)
#         # print(chord)
#         # print(twist)
#         # print(radius)
#         # print(Vt)
#         # print(psi)
#         # print(Omega)
#         # print(R)
#         # print(beta)

#         M = np.zeros((3,3))
#         L = np.zeros((3,3))
#         diag_entries_M = np.array([128/75/np.pi, -16/45/np.pi, -16/45/np.pi])
#         diag_entries_L = np.array([1/2, -4/(1 + np.sin(beta)), -4 * np.sin(beta) /(1 + np.sin(beta))])
    
#         M[np.diag_indices_from(M)] = diag_entries_M
#         M_inv = np.linalg.inv(M)
#         L[np.diag_indices_from(L)] = diag_entries_L
#         L[0,2] = 15 * np.pi / 64 * ((1 - np.sin(beta))/(1 + np.sin(beta)))**0.5
#         L[2,0] = 15 * np.pi / 64 * ((1 - np.sin(beta))/(1 + np.sin(beta)))**0.5
#         L_inv = np.linalg.inv(L)
#         # print(L,'L_MAT')



#         nu_0 = 0 # TO DO: needs to be num_evaluations; need 2 copies nu_0_vec 
#         nu_s = 0
#         nu_c = 0

#         omega = 1

#         for i in range(300):
#             # Compute nu and ux 
#             nu =  (nu_0 + nu_s * radius * np.sin(psi) + nu_c * radius * np.cos(psi))
#             ux =  nu * Omega * R 
#             # print(nu_c)
#             # print(ux[0,:,0])
            
#             # Compute inflow angle phi with the assumption of ut = 0
#             phi = np.arctan(ux / Vt)
#             alpha = twist - phi
#             # print(phi[0,:,0] * 180/np.pi)
            

#             # Evaluate Cl and Cd using airfoil surrogate model
#             self.airfoil_model_inputs[:,0] = alpha.flatten()
#             self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6
#             airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
#             Cl = airfoil_model_outputs[:,:,:,0]
#             Cd = airfoil_model_outputs[:,:,:,1]

#             Cx = (Cl * np.cos(phi) - Cd * np.sin(phi))
#             Ct = (Cl * np.sin(phi) + Cd * np.cos(phi))

#             # Compute tangential induced velocity 
#             ut = 2 * sigma * Ct * Vt / (4 * np.sin(phi) * np.cos(phi) + sigma * Ct)

#             # Compute thrust and torque using blade element theory
#             dT = 0.5 * B * rho * (ux**2 + (Vt - 0. * ut)**2) * chord * Cx * dr
#             T = np.sum(dT) / shape[0] / shape[2]
#             dQ = 0.5 * B * rho * (ux**2 + (Vt - 0. * ut)**2) * chord * Ct * radius * dr
#             Q = np.sum(dQ) / shape[0] / shape[2]
            
#             # Compute roll and pitch moments from thrust
#             dL_mom = radius * np.sin(psi) * dT
#             L_mom  = np.sum(dL_mom) / shape[0] / shape[2]
#             dM_mom = radius * np.cos(psi) * dT
#             M_mom  = np.sum(dM_mom) / shape[0] / shape[2]

#             # Compute coefficients 
#             C_T = T / rho[0][0][0] / (Omega[0][0][0] / 2 / np.pi)**2 / (R[0][0][0] * 2)**4
#             C_L = L_mom / rho[0][0][0] / (Omega[0][0][0] / 2 / np.pi)**2 / (R[0][0][0] * 2)**5
#             C_M = M_mom / rho[0][0][0] / (Omega[0][0][0] / 2 / np.pi)**2 / (R[0][0][0] * 2)**5
            
#             # C_T = T / np.pi / R[0][0][0]**2 / rho[0][0][0] / Omega[0][0][0]**2 / (R[0][0][0])**2
#             # C_L = L_mom / np.pi / R[0][0][0]**2 / rho[0][0][0] / Omega[0][0][0]**2 / (R[0][0][0])**3
#             # C_M = M_mom / np.pi / R[0][0][0]**2 / rho[0][0][0] / Omega[0][0][0]**2 / (R[0][0][0])**3
           
#             # Updating nu values via successive over-relaxation 
#             coeff_vec = np.array([C_T,-C_L,-C_M])
#             nu_vec    = np.array([nu_0,nu_s,nu_c])
            
#             term1 = delta_t = omega * np.matmul(L,M)
#             term2 = np.matmul(M_inv,coeff_vec) - np.matmul(np.matmul(M_inv,L_inv),nu_vec)
#             nu_updated = nu_vec + np.matmul(delta_t, term2)
#             # nu_updated = nu_vec + 0.2 * term2
            
#             nu_0 = nu_updated[0]
#             nu_s = nu_updated[1]
#             nu_c = nu_updated[2]

#             error = np.linalg.norm(nu_updated - nu_vec)

#             # print(np.linalg.norm(nu_updated - nu_vec), 'NORM')

#             if error < 1e-8:
#                 print('ODE solved in {}'.format(i) + ' iterations')
#                 break

#         # print(phi[0,:,0] * 180/np.pi)
#         # print(np.sum(dM_mom), 'moment sum')
#         # print(dM_mom[0,1,:])
#         # print(np.sum(np.cos(psi[0,0,:])),'sum of cos')
#         # print(psi[0,0,:])
#         print(L_mom)
#         print(M_mom)
#         # print(alpha[0,:,0]*180/np.pi)
#         print(ux[0,:,1])
#         print(alpha[0,:,0] * 180/np.pi)
#         # print(dT[0,-1,:])
#         # print(dM_mom[0,:,0])
        
#         outputs['total_thrust'] = T
#         outputs['total_torque'] = Q
#         outputs['pitching_moment'] = L_mom
#         outputs['rolling_moment'] = M_mom
        # outputs['C_L'] = C_L
        
