import numpy as np 
from csdl import Model
import csdl 
from lsdo_rotor.rotor_parameters import RotorParameters
import openmdao.api as om 


class PittPeters(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=RotorParameters)

    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']

        # Adding inputs 
        self.add_input('_re_pitt_peters', shape=shape)
        self.add_input('_rho_pitt_peters', shape=shape)
        self.add_input('_chord', shape=shape)
        self.add_input('_pitch', shape=shape)
        self.add_input('_normalized_radius', shape=shape)
        self.add_input('_radius', shape=shape)
        self.add_input('_tangential_inflow_velocity', shape=shape)
        self.add_input('_theta', shape=shape)
        self.add_input('_angular_speed', shape=shape)
        self.add_input('_rotor_radius', shape=shape)
        self.add_input('_dr', shape=shape)

        self.airfoil_model_inputs = np.zeros((shape[0] * shape[1] * shape[2], 2))

        self.add_output('dT', shape=shape)
        self.add_output('dQ', shape=shape)

        indices = np.arange(shape[0] * shape[1] * shape[2])
        # self.declare_derivatives('dT','_rho_pitt_peters', rows=indices, cols=indices)
        # self.declare_derivatives('dT','_chord', rows=indices, cols=indices)
        # self.declare_derivatives('dT','_pitch', rows=indices, cols=indices)
        # self.declare_derivatives('dT','_normalized_radius', rows=indices, cols=indices)
        # self.declare_derivatives('dT','_tangential_inflow_velocity', rows=indices, cols=indices)
        # self.declare_derivatives('dT','_theta' , rows=indices, cols=indices)
        # self.declare_derivatives('dT','_angular_speed', rows=indices, cols=indices)
        # self.declare_derivatives('dT','_rotor_radius', rows=indices, cols=indices)
        # self.declare_derivatives('dT','_dr', rows=indices, cols=indices)

        # self.declare_derivatives('dQ','_rho_pitt_peters', rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_chord', rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_pitch', rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_normalized_radius', rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_tangential_inflow_velocity', rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_theta' , rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_angular_speed', rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_rotor_radius', rows=indices, cols=indices)
        # self.declare_derivatives('dQ','_dr', rows=indices, cols=indices)

        self.declare_derivatives('dT','_rho_pitt_peters')#, method = 'cs')
        self.declare_derivatives('dT','_re_pitt_peters')
        self.declare_derivatives('dT','_chord')#, method ='cs')
        self.declare_derivatives('dT','_pitch')#, method = 'cs')
        self.declare_derivatives('dT','_normalized_radius')#, method = 'cs')
        self.declare_derivatives('dT','_radius')
        self.declare_derivatives('dT','_tangential_inflow_velocity')#, method = 'cs')
        self.declare_derivatives('dT','_theta' )#, method = 'cs')
        self.declare_derivatives('dT','_angular_speed')#, method = 'cs')
        self.declare_derivatives('dT','_rotor_radius')#, method = 'cs')
        self.declare_derivatives('dT','_dr')#, method = 'cs')

        self.declare_derivatives('dQ','_rho_pitt_peters')#, method = 'cs')
        self.declare_derivatives('dQ','_chord')#, method = 'cs')
        self.declare_derivatives('dQ','_pitch')#, method = 'cs')
        self.declare_derivatives('dQ','_normalized_radius')#, method = 'cs')
        self.declare_derivatives('dQ','_radius')
        self.declare_derivatives('dQ','_tangential_inflow_velocity')#, method = 'cs')
        self.declare_derivatives('dQ','_theta')#, method = 'cs')
        self.declare_derivatives('dQ','_angular_speed')#, method = 'cs')
        self.declare_derivatives('dQ','_rotor_radius')#, method = 'cs')
        self.declare_derivatives('dQ','_dr')#, method = 'cs')


        

    def compute(self, inputs,outputs):
        shape       = self.parameters['shape']
        rotor       = self.parameters['rotor']
        interp      = rotor['interp']
        beta        = rotor['rotor_disk_tilt_angle'] 
        print(beta,'BETA')
        B           = rotor['num_blades']


        Re          = inputs['_re_pitt_peters']
        rho         = inputs['_rho_pitt_peters']
        chord       = inputs['_chord']
        # print(chord)
        twist       = inputs['_pitch']
        radius      = inputs['_normalized_radius']
        r           = inputs['_radius']
        Vt          = inputs['_tangential_inflow_velocity']
        psi         = inputs['_theta']
        Omega       = inputs['_angular_speed']
        R           = inputs['_rotor_radius']
        dr          = inputs['_dr']

        lamb = rotor['speed_ratio']
        mu = rotor['inflow_ratio']

        # M = np.zeros((3,3))
        # diag_entries_M = np.array([128/75/np.pi, -16/45/np.pi, -16/45/np.pi])
        # M[np.diag_indices_from(M)] = diag_entries_M
        # M_inv = np.linalg.inv(M)
        # M = np.broadcast_to(M, (shape[0],3,3))
        # M_inv = np.broadcast_to(M_inv, (shape[0],3,3))
        # L = np.zeros((shape[0],3,3))
        # L_inv = np.zeros((shape[0],3,3))


        # M =  np.broadcast_to(M,(shape[0],3,3))
        # M_inv = np.broadcast_to(M_inv,(shape[0],3,3))
        
        # for i in range(shape[0]):
        #     diag_entries_L = np.array([1/2, -4/(1 + np.sin(beta[i])), -4 * np.sin(beta[i]) /(1 + np.sin(beta[i]))])
        #     entry = np.diag(diag_entries_L)
        #     L[i,:,:] = entry
        #     L[i,0,2] = 15 * np.pi / 64 * ((1 - np.sin(beta[i]))/(1 + np.sin(beta[i])))**0.5
        #     L[i,2,0] = 15 * np.pi / 64 * (1 - np.sin(beta[i]))/(1 + np.sin(beta[i]))
            
        #     V_eff = (lamb[i]**2 + mu[i]**2)**0.5
        #     L[i,:,:] = L[i,:,:] / V_eff

        #     L_inv[i,:,:] = np.linalg.inv(L[i,:,:])

        i_vec = rotor['rotor_disk_tilt_angle']
        lamb = rotor['speed_ratio']
        mu = rotor['inflow_ratio']
        L = np.zeros((shape[0],3,3))
        L_inv = np.zeros((shape[0],3,3))
        L_list = []
        L_inv_list = []
        m = np.array([128/75/np.pi, -16/45/np.pi, -16/45/np.pi])
        M  = np.diag(m)
        M_inv = np.linalg.inv(M)
        M_block_list = []
        M_inv_block_list = []

        for i in range(shape[0]):
            L[i,0,0] = 0.5
            L[i,0,2] = 15 * np.pi/64 * ((1 - np.sin(i_vec[i]))/(1 + np.sin(i_vec[i])))**0.5
            L[i,1,1] = - 4 / (1 + np.sin(i_vec[i]))
            L[i,2,0] = L[i,0,2]
            L[i,2,2] = - 4 * np.sin(i_vec[i]) / (1 + np.sin(i_vec[i]))

            V_eff = (lamb[i]**2 + mu[i]**2)**0.5
            print(V_eff)

            L[i,:,:] = L[i,:,:] / V_eff

            L_list.append(L[i,:,:])

            L_inv[i,:,:] = np.linalg.inv(L[i,:,:])
            L_inv_list.append(L_inv[i,:,:])

            M_block_list.append(M)
            M_inv_block_list.append(M_inv)

        from scipy.linalg import block_diag
        L_block = block_diag(*L_list)
        L_inv_block = block_diag(*L_inv_list)
        M_block = block_diag(*M_block_list)
        M_inv_block = block_diag(*M_inv_block_list)


        nu_0 = np.ones(shape[0],)
        nu_s = np.ones(shape[0],)
        nu_c = np.ones(shape[0],)

        omega = 0.3

        for i in range(40):
            print(i)
            self.nu_0_exp = np.einsum(
            'i,ijk->ijk',
            nu_0,
            np.ones((shape[0], shape[1],shape[2])),         
            )
            self.nu_s_exp = np.einsum(
            'i,ijk->ijk',
            nu_s,
            np.ones((shape[0], shape[1],shape[2])),         
            )
            self.nu_c_exp = np.einsum(
            'i,ijk->ijk',
            nu_c,
            np.ones((shape[0],shape[1],shape[2])),         
            )


            # Compute nu and ux 
            self.nu =  self.nu_0_exp + self.nu_s_exp * radius * np.sin(psi) + self.nu_c_exp * radius * np.cos(psi)
            self.ux =  self.nu * Omega * R 
            
            # Compute inflow angle phi with the assumption of ut = 0
            self.phi = np.arctan(self.ux / Vt)
            self.alpha = twist - self.phi

            # Evaluate Cl and Cd using airfoil surrogate model
            self.airfoil_model_inputs[:,0] = self.alpha.flatten()
            self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6
            airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
            self.Cl = airfoil_model_outputs[:,:,:,0]
            self.Cd = airfoil_model_outputs[:,:,:,1]

            # Compute thrust and torque using blade element theory
            self.dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
            # self.dTdc[i,:,:,:] = self.dT 
            T = np.sum(np.sum(self.dT,axis =1),axis = 1) / shape[2]
            print(T)
            self.dQ = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.sin(self.phi) + self.Cd * np.cos(self.phi)) * r * dr
            Q = np.sum(np.sum(self.dQ,axis = 1),axis = 1) / shape[2]
            
            # Compute roll and pitch moments from thrust
            dL_mom = radius * np.sin(psi) * self.dT
            L_mom  = np.sum(dL_mom) / shape[0] / shape[2]
            dM_mom = radius * np.cos(psi) * self.dT
            M_mom  = np.sum(dM_mom) / shape[0] / shape[2]

            # Compute coefficients 
            dC_T = self.dT / rho / (Omega / 2 / np.pi)**2 / (R * 2)**4
            dC_L = dL_mom / rho / (Omega / 2 / np.pi)**2 / (R * 2)**5
            dC_M = dM_mom / rho / (Omega / 2 / np.pi)**2 / (R * 2)**5

            C_T = np.sum(np.sum(dC_T, axis = 1), axis = 1) / shape[2]
            C_L = np.sum(np.sum(dC_L, axis = 1), axis = 1) / shape[2]
            C_M = np.sum(np.sum(dC_M, axis = 1), axis = 1) / shape[2]

            # Updating nu values via successive over-relaxation 
            coeff_vec = np.zeros((shape[0],3))
            nu_vec = np.zeros((shape[0],3))
            for i in range(shape[0]):
                coeff_vec[i,0] = C_T[i]
                coeff_vec[i,1] = -C_L[i]
                coeff_vec[i,2] = -C_M[i] 

                nu_vec[i,0] = nu_0[i]
                nu_vec[i,1] = nu_s[i]
                nu_vec[i,2] = nu_c[i] 
           
            # coeff_vec = coeff_vec.reshape((shape[0],3,1))
            coeff_vec = coeff_vec.flatten()
            # nu_vec = nu_vec.reshape((shape[0],3,1))
            nu_vec = nu_vec.flatten()
           
            
            # term1 = delta_t = omega * np.einsum(
            #     'ijk,ikl->ijl',
            #     L,
            #     M,
            #     )
            # term2 = np.einsum(
            #     'ijk,ikl->ijl',
            #     M_inv,
            #     coeff_vec,
            #     ) 
            # term3 = np.einsum(
            #     'ijk,ikl->ijl',
            #     M_inv,
            #     L_inv,
            # )
            # term4 = np.einsum(
            #     'ijk,ikl->ijl',
            #     term3,
            #     nu_vec,
            # )
            # term5 = term2 - term4
            # term6 = np.einsum(
            #     'ijk,ikl->ijl',
            #     term1,
            #     term5,
            # )
            term1 = 0.1 * np.matmul(L_block,M_block)
            term2 = np.matmul(M_inv_block,coeff_vec)
            term3 = np.matmul(M_inv_block,L_inv_block)
            term4 = np.matmul(term3,nu_vec)
            term5 = term2 - term4
            term6 = np.matmul(term1,term5)
            
            nu_updated = nu_vec.reshape((shape[0],3,1)) + term6.reshape((shape[0],3,1))
            print(nu_updated.shape)

            nu_0 = nu_updated[:,0,0]
            nu_s = nu_updated[:,1,0]
            nu_c = nu_updated[:,2,0]

            error = np.linalg.norm(nu_updated - nu_vec)

            if error < 1e-8:
                print('ODE solved')
                break
            print(error)
        
        # print(self.dT)
        
        outputs['dT'] = self.dT
        outputs['dQ'] = self.dQ

    def compute_derivatives(self, inputs, derivatives):
        shape       = self.parameters['shape']
        rotor       = self.parameters['rotor']
        interp      = rotor['interp']
        beta        = rotor['rotor_disk_tilt_angle'] 
        
        Re          = inputs['_re_pitt_peters']
        rho         = inputs['_rho_pitt_peters']
        chord       = inputs['_chord']
        twist       = inputs['_pitch']
        radius      = inputs['_normalized_radius']
        r           = inputs['_radius']
        Vt          = inputs['_tangential_inflow_velocity']
        psi         = inputs['_theta']
        Omega       = inputs['_angular_speed']
        R           = inputs['_rotor_radius']
        dr          = inputs['_dr']
        # print(dr, 'dR')

        B = rotor['num_blades']

        self.airfoil_model_inputs[:,0] = self.alpha.flatten()
        self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6


        # Thrust derivatives
        # dTdrho = 0.5 * B * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
        dTdrho = self.dT / rho
        # print(dTdrho.shape,'dT_drho SHAPE')
        derivatives['dT', '_rho_pitt_peters'] = np.diag(dTdrho.flatten())
        # dTdchord = (0.5 * B * rho * (self.ux**2 + Vt**2) * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr).flatten()
        dTdchord = self.dT.flatten() / chord.flatten()
        # dTdchord = (self.dTdc[0,:,:,:] * self.dTdc[1,:,:,:]).flatten()
        derivatives['dT', '_chord'] = np.diag(dTdchord.flatten())

        dTdpitch = ((0.5 * B * rho * chord * (self.ux**2 + Vt**2) * np.cos(self.phi)  * dr) * \
            interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,0].reshape(shape) * 1).flatten()
        derivatives['dT', '_pitch'] = np.diag(dTdpitch.flatten())
        # =  dT/dCl * dCl/dalpha * dalpha/twist

        dTdRe = (0.5 * B * chord * rho * (self.ux**2 + Vt**2) * np.cos(self.phi)  * dr).flatten() * (interp.predict_derivatives(self.airfoil_model_inputs, 1)[:,0]/2e6)
        derivatives['dT','_re_pitt_peters'] = np.diag(dTdRe.flatten())
        # dTdRe = dTdCl * dCldRe

        dTdnormr = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr * \
            Omega * R * self.nu_s_exp  * np.sin(psi) + self.nu_c_exp * np.cos(psi)).flatten()
        derivatives['dT','_normalized_radius'] = np.diag(dTdnormr.flatten())
        # = dT/dux * dux/dnu * dnu/dradius 

        dim = len(dTdnormr.flatten())
        derivatives['dT','_radius'] = np.zeros((dim,dim))

        dTdVt = (0.5 * B * rho * 2 * Vt * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr).flatten()
        derivatives['dT','_tangential_inflow_velocity'] = np.diag(dTdVt.flatten())

        dTdtheta = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr * \
             Omega * R * self.nu_s_exp * radius * np.cos(psi) - self.nu_c_exp * radius * np.sin(psi)).flatten()
        derivatives['dT','_theta'] = np.diag(dTdtheta)
        # = dT/dux * dux/dnu * dnu/dpsi

        dT_dCl =  0.5 * B * rho * (self.ux**2 + Vt**2) * chord * np.cos(self.phi) * dr
        dCl_dalpha = interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,0]
        dalpha_dphi = -1
        dphi_dux = Vt / (self.ux**2 + Vt**2)
        dux_domega = self.nu * R
        # dTdomega = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr * \
        #     self.nu * R).flatten()
        dTdomega = dT_dCl.flatten() * dCl_dalpha * dalpha_dphi * dphi_dux.flatten() * dux_domega.flatten()
        derivatives['dT','_angular_speed'] = np.diag(dTdomega.flatten())
        # # = dT/dux * dux/domega

        dTdR = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr * \
            self.nu * Omega).flatten()
        derivatives['dT','_rotor_radius'] = np.diag(dTdR) 
        # # dT/dux * dux/dR 

        dTddr = self.dT / dr #(0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi))).flatten()
        derivatives['dT','_dr'] = np.diag(dTddr.flatten())
        

        # # Torque derivatives
        dQdrho =  (0.5 * B *  (self.ux**2 + Vt**2) * chord * (self.Cl * np.sin(self.phi) + self.Cd * np.cos(self.phi)) * r * dr).flatten()
        derivatives['dQ', '_rho_pitt_peters'] = np.diag(dQdrho.flatten())
        
        dQdc = (0.5 * B * rho * (self.ux**2 + Vt**2) *  (self.Cl * np.sin(self.phi) + self.Cd * np.cos(self.phi)) * r * dr).flatten()
        derivatives['dQ', '_chord'] = np.diag(dQdc)
        
        dQdpitch = ((0.5 * B * rho * (self.ux**2 + Vt**2) * np.cos(self.phi) * R * dr) * \
            interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,0].reshape(shape) * 1).flatten()
        derivatives['dQ', '_pitch'] = np.diag(dQdpitch)
        # =  dQ/dCl * dCl/dalpha * dalpha/twist

        dQdrnorm = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * r * dr * \
            Omega * R * self.nu_s_exp  * np.sin(psi) + self.nu_c_exp * np.cos(psi)).flatten()
        derivatives['dQ','_normalized_radius'] = np.diag(dQdrnorm)
        # # = dT/dux * dux/dnu * dnu/dradius 

        dQdVt = (0.5 * B * rho * 2 * Vt * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * r * dr).flatten()
        derivatives['dQ','_tangential_inflow_velocity'] = np.diag(dQdVt)

        dQdtheta = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * r * dr * \
             Omega * R * self.nu_s_exp * radius * np.cos(psi) - self.nu_c_exp * radius * np.sin(psi)).flatten()
        derivatives['dQ','_theta'] = np.diag(dQdtheta)
        # # = dT/dux * dux/dnu * dnu/dpsi

        dQdomega = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * r * dr * \
            self.nu * R ).flatten()
        derivatives['dQ','_angular_speed'] = np.diag(dQdomega)
        # # = dT/dux * dux/domega

        dQdR = (0.5 * B * rho * 2 * self.ux * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * r * dr * \
            self.nu * Omega).flatten()
        derivatives['dQ','_rotor_radius'] = np.diag(dQdR)
        # # dT/dux * dux/dR 

        # dQddr = (0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * radius).flatten()
        dQddr = self.dQ / dr
        derivatives['dQ','_dr'] = np.diag(dQddr.flatten())
                       
        # dQ = 0.5 * B * rho * (ux**2 + Vt**2) * chord * (Cl * np.sin(phi) + Cd * np.cos(phi)) * R * dr
        # dT = 0.5 * B * rho * (ux**2 + Vt**2) * chord * (Cl * np.cos(phi) - Cd * np.sin(phi)) * dr
