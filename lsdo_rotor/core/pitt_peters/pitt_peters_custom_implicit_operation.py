import numpy as np
from csdl import Model
import csdl

from lsdo_rotor.core.pitt_peters.pitt_peters_rotor_parameters import PittPetersRotorParameters

# # from lsdo_rotor.rotor_parameters import RotorParameters
# from lsdo_rotor.functions.get_rotor_dictionary import get_rotor_dictionary
# from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
# import openmdao.api as om
from scipy.linalg import block_diag
import time
import sympy as sym 
from sympy import *

class PittPetersCustomImplicitOperation(csdl.CustomImplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=PittPetersRotorParameters)
        self.parameters.declare('num_blades', types=int)
    
    def define(self):
        shape = self.parameters['shape']

        self.add_input('_re_pitt_peters', shape=shape)
        self.add_input('_chord', shape=shape)
        self.add_input('_pitch', shape=shape)        
        self.add_input('_tangential_inflow_velocity', shape=shape)
        self.add_input('_dr', shape=shape)
        self.add_input('_rotor_radius', shape=shape)
        self.add_input('_radius', shape=shape)
        # self.add_input('_psi', shape=shape)
        self.add_input('_angular_speed', shape=shape)
        self.add_input('_density_expanded', shape=shape)
        self.add_input('density', shape=(shape[0],))
        self.add_input('mu', shape=(shape[0],))
        self.add_input('mu_z', shape=(shape[0],))

        self.add_output('_lambda', shape=(shape[0],3))

        self.declare_derivatives('_lambda', '_dr',method='exact')
        self.declare_derivatives('_lambda', '_chord',method='exact')
        self.declare_derivatives('_lambda', '_pitch',method='exact')
        self.declare_derivatives('_lambda', '_radius',method='exact')
        self.declare_derivatives('_lambda', '_re_pitt_peters',method='exact')
        self.declare_derivatives('_lambda', '_tangential_inflow_velocity',method='exact')
        self.declare_derivatives('_lambda', '_rotor_radius',method='exact')
        # self.declare_derivatives('_lambda', '_psi',method='exact')
        self.declare_derivatives('_lambda', '_angular_speed',method='exact')
        self.declare_derivatives('_lambda', '_density_expanded',method='exact')
        self.declare_derivatives('_lambda', 'density',method='exact')
        self.declare_derivatives('_lambda', 'mu',method='exact')
        self.declare_derivatives('_lambda', 'mu_z',method='exact')
        
        self.declare_derivatives('_lambda','_lambda',method='exact')
        
        self.airfoil_model_inputs = np.zeros((shape[0] * shape[1] * shape[2], 2))

    def evaluate_residuals(self,inputs,outputs,residuals):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        B = self.parameters['num_blades']
        ne = shape[0]
        nr = shape[1]
        nt = shape[2]
        
        # -----objects and variable from rotor dictionary ----- #
        interp = rotor['interp']
        # B = rotor['num_blades']
        # print(B)
        # print('------------NUM BLADES---------')
        # rho = rotor['density']
        psi = rotor['azimuth_angle']
        M_block = rotor['M_block_matrix']
        M_inv_block = rotor['M_inv_block_matrix']
        mu = inputs['mu']
        mu_z = inputs['mu_z']
        mu_z_exp = np.einsum(
            'i,ijk->ijk',
            mu_z,
            np.ones((ne, nr,nt)),  
        )

        # ------input variables ------- # 
        Omega = inputs['_angular_speed']
        dr = inputs['_dr']
        R = inputs['_rotor_radius']
        r = inputs['_radius']
        Re = inputs['_re_pitt_peters']
        chord = inputs['_chord']
        twist = inputs['_pitch']
        Vt = inputs['_tangential_inflow_velocity']
        rho = inputs['density']
        rho_exp = inputs['_density_expanded']

        normalized_radial_discretization = 1. / nr / 2. \
        + np.linspace(0., 1. - 1. / nr, nr)
        self.r_norm = np.einsum(
            'ik,j->ijk',
            np.ones((ne, nt)),
            normalized_radial_discretization,
        )


        self.lamb = outputs['_lambda']
        self.C = np.zeros((shape[0],3))
        self.C_T = np.zeros((shape[0],))
        self.C_Mx = np.zeros((shape[0],))
        self.C_My = np.zeros((shape[0],))


        self.lamb_0_exp = np.einsum(
            'i,ijk->ijk',
            self.lamb[:,0],
            np.ones((ne, nr,nt)),         
            )
        self.lamb_c_exp = np.einsum(
            'i,ijk->ijk',
            self.lamb[:,1],
            np.ones((ne, nr,nt)),         
            )
        self.lamb_s_exp = np.einsum(
            'i,ijk->ijk',
            self.lamb[:,2],
            np.ones((ne, nr,nt)),         
            )
        
        # compute solidity 
        sigma = B * chord / 2 / np.pi / r 
        Cl = 0

        # Compute nu and self.ux (ne, nr, nt)
        self.lamb_exp = self.lamb_0_exp + self.lamb_c_exp * self.r_norm * np.cos(psi) + self.lamb_s_exp * self.r_norm * np.sin(psi)
        self.ux = (self.lamb_exp + mu_z_exp) * Omega * R

        # Compute inflow angle self.phi (ne, nr, nt) (ignore u_theta) 
        # self.phi = np.arctan(self.ux / Vt)
        self.phi = np.arctan(self.ux / Vt * (1 + sigma * Cl / 4))
            # self.phi = np.arctan(self.ux / Vt)
        u_theta = 2 * sigma * Cl * np.sin(self.phi) * Vt / (4 * np.sin(self.phi) * np.cos(self.phi) + sigma * Cl * np.sin(self.phi))

        # Compute sectional AoA (ne, nr, nt)
        alpha = twist - self.phi 

        # Apply airfoil surrogate model to compute Cl and Cd (ne, nr, nt)
        self.airfoil_model_inputs[:,0] = alpha.flatten()
        self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6
        self.airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
        self.Cl = self.airfoil_model_outputs[:,:,:,0]
        self.Cd = self.airfoil_model_outputs[:,:,:,1]

        # self.dT = 0.5 * B * rho_exp * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
        self.dT = 0.5 * B * rho_exp * (self.ux**2 + (Vt - 0.5 * u_theta)**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
        self.T = np.sum(np.sum(self.dT,axis =1),axis = 1) / shape[2]
        
        # Compute roll and pitch moments from thrust
        dMx = r * np.sin(psi) * self.dT
        Mx  = np.sum(np.sum(dMx,axis=1),axis=1) / shape[2]
        dMy = r * np.cos(psi) * self.dT
        My  = np.sum(np.sum(dMy,axis=1),axis=1) / shape[2]

        # nu_vec = np.zeros((shape[0],3))
        L = np.zeros((shape[0],3,3))
        L_inv = np.zeros((shape[0],3,3))
        L_list = []
        L_inv_list = []
        for i in range(ne):
            self.C_T[i] = self.T[i] / (rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)  #(Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
            self.C_My[i] = My[i] / (rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])
            self.C_Mx[i] = Mx[i] / (rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) # rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
             #  rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5

            lamb_i = np.mean(self.lamb_exp[i,:,:])
            lamb = lamb_i + mu_z[i]
    

            Chi = np.arctan(mu[i]/lamb) # Wake skew angle

            V_eff = (mu[i]**2 + lamb * (lamb + lamb_i)) / (mu[i]**2 + lamb**2)**0.5
            L[i,0,0] = 0.5
            L[i,0,1] = -15 * np.pi/64 * ((1 - np.cos(Chi))/(1 + np.cos(Chi)))**0.5
            L[i,1,1] = 4 * np.cos(Chi) / (1 + np.cos(Chi))
            L[i,1,0] = -L[i,0,1]
            L[i,2,2] = 4 / (1 + np.cos(Chi))

            L[i,:,:] = L[i,:,:] / V_eff
            L_list.append(L[i,:,:])

            L_inv[i,:,:] = np.linalg.inv(L[i,:,:])
            L_inv_list.append(L_inv[i,:,:])

            self.C[i,0] = self.C_T[i]
            self.C[i,1] = -self.C_My[i]
            self.C[i,2] = self.C_Mx[i] 

        L_block = block_diag(*L_list)
        L_inv_block = np.linalg.inv(L_block)#block_diag(*L_inv_list)
        # self.C_new = C.reshape((shape[0],3,1))
        self.C_new = self.C.flatten()
        # nu = nu.reshape((shape[0],3,1))
        self.lamb_new = self.lamb.flatten()

        self.term1 = 0.5 * np.matmul(L_block,M_block)
        self.term2 = np.matmul(M_inv_block,self.C_new)
        self.term3 = np.matmul(M_inv_block,L_inv_block)
       
        self.term4 = np.matmul(self.term3,self.lamb.flatten())
        self.term5 = self.term2 - self.term4
        self.term6 = np.matmul(self.term1,self.term5)
        # self.lamb += self.term6.reshape((ne,3))

        residuals['_lambda'] = self.term6    

    def compute_derivatives(self, inputs, outputs, derivatives):
        shape = self.parameters['shape']
        ne = shape[0]
        nr = shape[1]
        nt = shape[2]
        rotor = self.parameters['rotor']
        B = self.parameters['num_blades']
        # -----objects and variable from rotor dictionary ----- #
       
        # rho = rotor['density'] 
        psi = rotor['azimuth_angle']   
        # mu = rotor['mu']
        # mu_z = rotor['mu_z']
        mu = inputs['mu']
        mu_z = inputs['mu_z']
        mu_z_exp = np.einsum(
            'i,ijk->ijk',
            mu_z,
            np.ones((ne, nr,nt)),  
        )
        interp = rotor['interp']
        dL_dlambda_func = rotor['dL_dlambda_function']
        dL_dmu_func = rotor['dL_dmu_function']
        dL_dmu_z_func = rotor['dL_dmu_z_function']

        
        # ------input variables ------- # 
        Re = inputs['_re_pitt_peters']
        chord = inputs['_chord']
        twist = inputs['_pitch']
        Vt = inputs['_tangential_inflow_velocity']
        dr = inputs['_dr']
        R = inputs['_rotor_radius']
        r = inputs['_radius']
        Omega = inputs['_angular_speed']
        rho = inputs['density']
        rho_exp = inputs['_density_expanded']

        normalized_radial_discretization = 1. / nr / 2. \
        + np.linspace(0., 1. - 1. / nr, nr)
        self.r_norm = np.einsum(
            'ik,j->ijk',
            np.ones((ne, nt)),
            normalized_radial_discretization,
        )

        self.lamb = outputs['_lambda']
        
        self.lamb_0_exp = np.einsum(
            'i,ijk->ijk',
            self.lamb[:,0],
            np.ones((ne, nr,nt)),         
            )
        self.lamb_c_exp = np.einsum(
            'i,ijk->ijk',
            self.lamb[:,1],
            np.ones((ne, nr,nt)),         
            )
        self.lamb_s_exp = np.einsum(
            'i,ijk->ijk',
            self.lamb[:,2],
            np.ones((ne, nr,nt)),         
            )


        # Compute nu and self.ux (ne, nr, nt)
        self.lamb_exp = self.lamb_0_exp + self.lamb_c_exp * self.r_norm * np.cos(psi) + self.lamb_s_exp * self.r_norm * np.sin(psi)
        self.ux = (self.lamb_exp + mu_z_exp) * Omega * R

        # Compute inflow angle self.phi (ne, nr, nt) (ignore u_theta) 
        # self.phi = np.arctan(self.ux / Vt)
        # self.phi = np.arctan(self.ux / Vt * (1 + sigma * Cl / 4))
        self.phi = np.arctan(self.ux / Vt)
        # u_theta = 2 * sigma * Cl * np.sin(self.phi) * Vt / (4 * np.sin(self.phi) * np.cos(self.phi) + sigma * Cl * np.sin(self.phi))

        # Compute sectional AoA (ne, nr, nt)
        alpha = twist - self.phi 


        self.airfoil_model_inputs[:,0] = alpha.flatten()
        self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6
        self.airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
        self.Cl = self.airfoil_model_outputs[:,:,:,0]
        self.Cd = self.airfoil_model_outputs[:,:,:,1]

        

        a = np.ones((1,nt))
        b = np.ones((nr,1))

        #  Initializing derivative arrays 
        dC_ddT = np.zeros((ne*3,ne*nr*nt))
        dC_dr = np.zeros((ne*3,ne*nr*nt))
        dC_dpsi = np.zeros((ne*3,ne*nr*nt))
        dC_dR = np.zeros((ne*3,ne*nr*nt))
        dC_dOmega = np.zeros((ne*3,ne*nr*nt))
        dC_drho = np.zeros((ne*3,ne))
        dlambda_exp_dlambda = np.zeros((ne*3,ne*nr*nt))

        # Initializing empty L matrix
        L = np.zeros((ne,3,3))
        
        # Initializing empty lists (to later form block diagonal matrices)
        L_list = []
        ddT_ddr_list = []
        ddT_drhoexp_list = []
        ddT_dc_list = []
        ddT_dpitch_list = []
        ddT_dre_list = []
        ddT_dlambda_exp_list = []
        ddT_dR_list = []
        ddT_dOmega_list = []
        ddT_dpsi_list = []
        ddT_dVt_list = []
        ddT_dmuz_list = []
        dR_dlamb_1_list = []
        dR_dmu_list = []
        dR_dmu_z_list = []

        # Compute derivatives of Cl,Cd w.r.t. AoA, Re
        dCl_dalpha = interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,0].reshape((ne,nr,nt))
        dCd_dalpha = interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,1].reshape((ne,nr,nt))
        dCl_dre = (interp.predict_derivatives(self.airfoil_model_inputs, 1)[:,0] / 2e6).reshape((ne,nr,nt))
        dCd_dre = (interp.predict_derivatives(self.airfoil_model_inputs, 1)[:,1] / 2e6).reshape((ne,nr,nt))
        
       
        vec = np.zeros((ne*nr*nt,ne))
        for i in range(ne):
            vec[i*nr*nt:(i+1)*nr*nt,i] = 1 

        for i in range(ne):
            # mean of r cos(psi) amd r sin(psi) 
            r_cos_psi_mean = np.mean(self.r_norm[i,:,:] * np.cos(psi[i,:,:]))
            r_sin_psi_mean = np.mean(self.r_norm[i,:,:] * np.sin(psi[i,:,:]))
    
            dL_dlamb_subs = dL_dlambda_func(
                self.lamb[i,0],
                self.lamb[i,1],
                self.lamb[i,2],
                r_cos_psi_mean,
                r_sin_psi_mean,
                mu_z[i],
                mu[i],
                )

            dL_dlamb = np.nan_to_num(np.array([dL_dlamb_subs]).astype(np.float64).reshape(3,3,3), copy=False, nan=0.0)
            dL_dlamb[0,:,:] = dL_dlamb[0,:,:].T
            dL_dlamb[1,:,:] = dL_dlamb[1,:,:].T
            dL_dlamb[2,:,:] = dL_dlamb[2,:,:].T

            dR_dlamb = np.matmul(self.C[i,:],dL_dlamb)
            dlambda_dlambda = 0.5 * dR_dlamb
            
            dL_dmu_subs = dL_dmu_func(
                self.lamb[i,0],
                self.lamb[i,1],
                self.lamb[i,2],
                r_cos_psi_mean,
                r_sin_psi_mean,
                mu_z[i],
                mu[i],
                )
            # print(dL_dmu_subs,'dL_dmu_subs')
            dL_dmu = np.nan_to_num(np.array([dL_dmu_subs]).astype(np.float64).reshape(3,3), copy=False, nan=0.0)
            # dL_dmu[0,:] = dL_dmu[0,:].T
            # dL_dmu[1,:] = dL_dmu[1,:].T
            # dL_dmu[2,:] = dL_dmu[2,:].T
            dR_dmu = np.matmul(dL_dmu,self.C[i,:])
            dlambda_dmu = 0.5 * dR_dmu

            dL_dmu_z_subs = dL_dmu_z_func(
                self.lamb[i,0],
                self.lamb[i,1],
                self.lamb[i,2],
                r_cos_psi_mean,
                r_sin_psi_mean,
                mu_z[i],
                mu[i],
                )
            # print(dL_dmu_z_subs,'dL_dmu_z_subs shape')
            dL_dmu_z = np.nan_to_num(np.array([dL_dmu_z_subs]).astype(np.float64).reshape(3,3), copy=False, nan=0.0)
            # dL_dmu_z[0,:] = dL_dmu_z[0,:].T
            # dL_dmu_z[1,:] = dL_dmu_z[1,:].T
            # dL_dmu_z[2,:] = dL_dmu_z[2,:].T

            dR_dmu_z = np.matmul(dL_dmu_z,self.C[i,:])
            dlambda_dmu_z = 0.5 * dR_dmu_z

            


            dC_dCT = np.matmul(b,a) / (nt * rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)
            dC_dCMy = (np.matmul(b,a) / (nt * rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.cos(psi[i,:,:])
            dC_dCMx = (np.matmul(b,a) / (nt * rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.sin(psi[i,:,:])
            dC_ddT[3*i,i*nr*nt:(i+1)*nr*nt] = dC_dCT.flatten()
            dC_ddT[3*i+1,i*nr*nt:(i+1)*nr*nt] = -dC_dCMy.flatten()
            dC_ddT[3*i+2,i*nr*nt:(i+1)*nr*nt] = dC_dCMx.flatten()
            
            dC_dr[3*i+1,i*nr*nt:(i+1)*nr*nt] = -((np.matmul(b,a) / (nt * rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * np.cos(psi[i,:,:]) * self.dT[i,:,:] ).flatten()
            dC_dr[3*i+2,i*nr*nt:(i+1)*nr*nt] = ((np.matmul(b,a) / (nt * rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * np.sin(psi[i,:,:]) * self.dT[i,:,:]).flatten()
            
            dC_dpsi[3*i+1,i*nr*nt:(i+1)*nr*nt] = ((np.matmul(b,a) / (nt * rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.sin(psi[i,:,:]) * self.dT[i,:,:] ).flatten()
            dC_dpsi[3*i+2,i*nr*nt:(i+1)*nr*nt] = ((np.matmul(b,a) / (nt * rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.cos(psi[i,:,:]) * self.dT[i,:,:]).flatten()
            
            dC_dR_CT = -4 * self.C_T[i] / R[i,0,0]
            dC_dR_CMy = -5 * self.C_My[i] / R[i,0,0]
            dC_dR_CMx = -5 * self.C_Mx[i] / R[i,0,0]
            dC_dR[3*i,i*nr*nt] = dC_dR_CT
            dC_dR[3*i+1,i*nr*nt] = -dC_dR_CMy
            dC_dR[3*i+2,i*nr*nt] = dC_dR_CMx

            dC_dOmega_CT = -2 * self.C_T[i] / Omega[i,0,0]
            dC_dOmega_CMy = -2 * self.C_My[i] / Omega[i,0,0]
            dC_dOmega_CMx = -2 * self.C_Mx[i] / Omega[i,0,0]
            dC_dOmega[3*i,i*nr*nt] = dC_dOmega_CT
            dC_dOmega[3*i+1,i*nr*nt] = -dC_dOmega_CMy
            dC_dOmega[3*i+2,i*nr*nt] = dC_dOmega_CMx

            dC_drho_CT = -1 * self.C_T[i] / rho[i]
            dC_drho_CMy = -1 * self.C_My[i] / rho[i]
            dC_drho_CMx = -1 * self.C_Mx[i] / rho[i]
            dC_drho[3*i,i] = dC_drho_CT
            dC_drho[3*i+1,i] = -dC_drho_CMy
            dC_drho[3*i+2,i] = dC_drho_CMx

            dlambda_exp_dlambda_0_exp = np.ones((nr,nt))
            dlambda_exp_dlambda_c_exp = self.r_norm[i,:,:] * np.cos(psi[i,:,:])
            dlambda_exp_dlambda_s_exp = self.r_norm[i,:,:] * np.sin(psi[i,:,:])
            dlambda_exp_dlambda[3*i,i*nr*nt:(i+1)*nr*nt] = dlambda_exp_dlambda_0_exp.flatten()
            dlambda_exp_dlambda[3*i+1,i*nr*nt:(i+1)*nr*nt] = dlambda_exp_dlambda_c_exp.flatten()
            dlambda_exp_dlambda[3*i+2,i*nr*nt:(i+1)*nr*nt] = dlambda_exp_dlambda_s_exp.flatten()

            
            ddT_dCl = 0.5 * B * rho_exp[i,:,:] * (self.ux[i,:,:]**2 + Vt[i,:,:]**2) * chord[i,:,:] * np.cos(self.phi[i,:,:]) * dr[i,:,:]
            ddT_dCd = -0.5 * B * rho_exp[i,:,:] * (self.ux[i,:,:]**2 + Vt[i,:,:]**2) * chord[i,:,:] * np.sin(self.phi[i,:,:]) * dr[i,:,:]
            dalpha_dpitch = 1
            dalpha_dphi = -1 
            
            
            
            dphi_dux = Vt[i,:,:] / (Vt[i,:,:]**2 + self.ux[i,:,:]**2)
            dcosphi_dux = -Vt[i,:,:] * self.ux[i,:,:] / (Vt[i,:,:]**2 + self.ux[i,:,:]**2)**1.5
            dsinphi_dux = Vt[i,:,:]**2 / (Vt[i,:,:]**2 + self.ux[i,:,:]**2)**1.5
            
            dux_dR = (self.lamb_exp[i,:,:] + mu_z_exp[i,:,:]) * Omega[i,:,:]
            dux_dlambda_exp = Omega[i,:,:] * R[i,:,:]
            dux_dOmega = (self.lamb_exp[i,:,:] + mu_z_exp[i,:,:]) * R[i,:,:]
            dux_dmu_exp = Omega[i,:,:] * R[i,:,:]

            const = 0.5 * B * rho[i] * chord[i,:,:] * dr[i,:,:]

            # --------- ddT_ddr ---------#
            ddT_ddr = np.diag((self.dT[i,:,:]/dr[i,:,:]).flatten())

            # --------- ddT_drhoexp ---------#
            ddT_drhoexp = np.diag((self.dT[i,:,:]/rho_exp[i,:,:]).flatten())

            # --------- ddT_dc ---------#
            ddT_dc = np.diag((self.dT[i,:,:]/chord[i,:,:]).flatten())        

            # --------- ddT_dpitch ---------#
            ddT_dpitch = np.diag((ddT_dCl * dCl_dalpha[i,:,:] * dalpha_dpitch + ddT_dCd * dCd_dalpha[i,:,:] * dalpha_dpitch).flatten())
            
            # --------- ddT_dre ---------#
            ddT_dre = np.diag((ddT_dCl * dCl_dre[i,:,:] + ddT_dCd * dCd_dre[i,:,:]).flatten())

            # --------- ddT_dux ---------#
            dCl_dux = dCl_dalpha[i,:,:] * dalpha_dphi * dphi_dux
            dCd_dux = dCd_dalpha[i,:,:] * dalpha_dphi * dphi_dux 
            ddT_dux_1 = const * Vt[i,:,:]**2 * (dCl_dux * np.cos(self.phi[i,:,:]) + self.Cl[i,:,:] * dcosphi_dux - dCd_dux * np.sin(self.phi[i,:,:]) - self.Cd[i,:,:] * dsinphi_dux)
            ddT_dux_2 = const * (2 * self.ux[i,:,:] * self.Cl[i,:,:] * np.cos(self.phi[i,:,:]) + self.ux[i,:,:]**2 * dCl_dux * np.cos(self.phi[i,:,:]) + self.ux[i,:,:]**2 * self.Cl[i,:,:] * dcosphi_dux
                            -2 * self.ux[i,:,:] * self.Cd[i,:,:] * np.sin(self.phi[i,:,:]) - self.ux[i,:,:]**2 * dCd_dux * np.sin(self.phi[i,:,:]) - self.ux[i,:,:]**2 * self.Cd[i,:,:] * dsinphi_dux)
            ddT_dux =  ddT_dux_2 + ddT_dux_1 

            # --------- ddT_dlambda_exp ---------#
            ddT_dlambda_exp = np.diag((ddT_dux * dux_dlambda_exp).flatten())

            # --------- ddT_dR ---------#
            ddT_dR = np.diag((ddT_dux * dux_dR).flatten())

            # --------- ddT_dOmega ---------#
            ddT_dOmega = np.diag((ddT_dux * dux_dOmega).flatten())

            # --------- ddT_dmuzexp ---------#
            ddT_dmuz_exp = np.diag((ddT_dux * dux_dmu_exp).flatten())

            # --------- ddT_dpsi ---------#
            dlambda_exp_dpsi = -self.lamb_c_exp[i,:,:] * self.r_norm[i,:,:] * np.sin(psi[i,:,:]) + self.lamb_s_exp[i,:,:] * self.r_norm[i,:,:] * np.cos(psi[i,:,:])
            ddT_dpsi = np.diag((ddT_dux * dux_dlambda_exp * dlambda_exp_dpsi).flatten())

            # --------- ddT_dVt ---------#
            ddT_dVt_1 = 2 * const * Vt[i,:,:] * (self.Cl[i,:,:] * np.cos(self.phi[i,:,:]) - self.Cd[i,:,:] * np.sin(self.phi[i,:,:]))
            dCl_dVt =  dCl_dalpha[i,:,:] * -1 * -self.ux[i,:,:]/(self.ux[i,:,:]**2 + Vt[i,:,:]**2)
            dCd_dVt =  dCd_dalpha[i,:,:] * -1 * -self.ux[i,:,:]/(self.ux[i,:,:]**2 + Vt[i,:,:]**2)
            dcosphi_dVt = self.ux[i,:,:]**2 / (self.ux[i,:,:]**2 + Vt[i,:,:]**2)**1.5 
            dsinphi_dVt = -self.ux[i,:,:] * Vt[i,:,:] / (self.ux[i,:,:]**2 + Vt[i,:,:]**2) / (self.ux[i,:,:]**2 + Vt[i,:,:]**2)**0.5
            ddT_dVt_2 = const * Vt[i,:,:]**2 * (dCl_dVt * np.cos(self.phi[i,:,:]) - dCd_dVt * np.sin(self.phi[i,:,:]) + self.Cl[i,:,:] * dcosphi_dVt - self.Cd[i,:,:] * dsinphi_dVt)
            ddT_dVt_3 = const * self.ux[i,:,:]**2 * (dCl_dVt * np.cos(self.phi[i,:,:]) - dCd_dVt * np.sin(self.phi[i,:,:]) + self.Cl[i,:,:] * dcosphi_dVt - self.Cd[i,:,:] * dsinphi_dVt)

            ddT_dVt = np.diag((ddT_dVt_1 + ddT_dVt_2 + ddT_dVt_3).flatten())

       
            # Compute L matrix
            lamb_i = np.mean(self.lamb_exp[i,:,:])

            lamb = lamb_i + mu_z[i]
            Chi = np.arctan(mu[i]/lamb)
            V_eff = (mu[i]**2 + lamb * (lamb + lamb_i)) / (mu[i]**2 + lamb**2)**0.5
            L[i,0,0] = 0.5
            L[i,0,1] = -15 * np.pi/64 * ((1 - np.cos(Chi))/(1 + np.cos(Chi)))**0.5
            L[i,1,1] = 4 * np.cos(Chi) / (1 + np.cos(Chi))
            L[i,1,0] = -L[i,0,1]
            L[i,2,2] = 4 / (1 + np.cos(Chi))

            L[i,:,:] = L[i,:,:] / V_eff

            # Appending all lists 
            L_list.append(L[i,:,:])
            ddT_ddr_list.append(ddT_ddr)
            ddT_drhoexp_list.append(ddT_drhoexp)
            ddT_dc_list.append(ddT_dc)
            ddT_dpitch_list.append(ddT_dpitch)
            ddT_dre_list.append(ddT_dre)
            ddT_dlambda_exp_list.append(ddT_dlambda_exp)
            ddT_dR_list.append(ddT_dR)
            ddT_dOmega_list.append(ddT_dOmega)
            ddT_dmuz_list.append(ddT_dmuz_exp)
            ddT_dpsi_list.append(ddT_dpsi)
            ddT_dVt_list.append(ddT_dVt)
            dR_dlamb_1_list.append(dlambda_dlambda)
            dR_dmu_list.append(dlambda_dmu)
            dR_dmu_z_list.append(dlambda_dmu_z)


        # Creating all block diagonal matrices
        L_block = block_diag(*L_list)
        ddT_ddr_block = block_diag(*ddT_ddr_list)
        ddT_drhoexp_block = block_diag(*ddT_drhoexp_list)
        ddT_dc_block = block_diag(*ddT_dc_list)
        ddT_dpitch_block = block_diag(*ddT_dpitch_list)
        ddT_dre_block = block_diag(*ddT_dre_list)
        ddT_dVt_block = block_diag(*ddT_dVt_list)
        ddT_dR_block = block_diag(*ddT_dR_list)
        ddT_dpsi_block = block_diag(*ddT_dpsi_list)
        ddT_dOmega_block = block_diag(*ddT_dOmega_list)
        ddT_dmuz_exp_block = block_diag(*ddT_dmuz_list)
        dR_dlamb_1_block = block_diag(*dR_dlamb_1_list)
        dR_dmu_block = block_diag(*dR_dmu_list)
        # print(dR_dmu_block,'dR_dmu_block')
        dR_dmu_z_block = block_diag(*dR_dmu_z_list)

        # Chain rule for residual 
        dlambda_dC = 0.5 * L_block
        dlambda_ddT = np.matmul(dlambda_dC,dC_ddT)
        
        dlambda_ddr = np.matmul(dlambda_ddT,ddT_ddr_block)
        dlambda_drhoexp = np.matmul(dlambda_ddT,ddT_drhoexp_block)
        dlambda_dc = np.matmul(dlambda_ddT,ddT_dc_block)
        dlambda_dpitch = np.matmul(dlambda_ddT,ddT_dpitch_block)
        dlambda_dre = np.matmul(dlambda_ddT,ddT_dre_block)
        dlambda_dVt = np.matmul(dlambda_ddT,ddT_dVt_block)
        dlambda_dR = np.matmul(dlambda_ddT,ddT_dR_block)
        dlambda_dpsi = np.matmul(dlambda_ddT,ddT_dpsi_block)
        dlambda_dOmega = np.matmul(dlambda_ddT,ddT_dOmega_block)
        
        dlambda_dmu_z_exp2 = np.matmul(dlambda_ddT,ddT_dmuz_exp_block)
        # print(ddT_dmuz_exp_block)
        # print(dlambda_dmu_z_exp2.shape)

        
        # vec[0,0] = 1
        # dlambda_dmu_z_exp = np.matmul(dlambda_dmu_z_exp2,np.ones((ne*nr*nt,1)))
        dlambda_dmu_z_exp = np.matmul(dlambda_dmu_z_exp2,vec)
        # print(dlambda_dmu_z_exp)
        # print(dlambda_dmu_z.shape,'dlambda_dmu_z')

        dlambda_dr = np.matmul(dlambda_dC,dC_dr)

       
        # Setting partials of residual w.r.t inputs
        derivatives['_lambda','_dr'] = dlambda_ddr
        derivatives['_lambda','_density_expanded'] = dlambda_drhoexp
        derivatives['_lambda','density'] =  np.matmul(dlambda_dC,dC_drho)
        derivatives['_lambda','_chord'] = dlambda_dc
        derivatives['_lambda','_pitch'] = dlambda_dpitch
        derivatives['_lambda','_re_pitt_peters'] = dlambda_dre
        derivatives['_lambda','_tangential_inflow_velocity'] = dlambda_dVt
        derivatives['_lambda','_radius'] = dlambda_dr
        derivatives['_lambda','_rotor_radius'] =   np.matmul(dlambda_dC,dC_dR) + dlambda_dR  
        # derivatives['_lambda','_psi'] =   np.matmul(dlambda_dC,dC_dpsi) + dlambda_dpsi  
        derivatives['_lambda','_angular_speed'] =   np.matmul(dlambda_dC,dC_dOmega) + dlambda_dOmega  
        # print(np.matmul(dlambda_dC,dC_dpsi))
        # print(dlambda_dpsi,'dlambda_dpsi')
        derivatives['_lambda','mu'] = dR_dmu_block.T
        derivatives['_lambda','mu_z'] = dR_dmu_z_block.T + dlambda_dmu_z_exp
        # print(derivatives['_lambda','mu'],'derivatives lambda wrt mu')
        # print(derivatives['_lambda','mu_z'],'derivatives lambda wrt mu_z')

        # Setting partial of residual w.r.t state
        ddT_dlambda_exp_block = block_diag(*ddT_dlambda_exp_list)
        dlambda_dlambda_exp = np.matmul(dlambda_ddT,ddT_dlambda_exp_block)
        dR_dlamb_2_block = np.matmul(dlambda_dlambda_exp,np.transpose(dlambda_exp_dlambda))
        derivatives['_lambda', '_lambda'] =  (- 0.5 * np.eye(ne*3) + dR_dlamb_2_block.T + dR_dlamb_1_block).T
        
        self.inv_jac = np.linalg.inv(derivatives['_lambda', '_lambda'])
        

    def solve_residual_equations(self, inputs,outputs):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        ne = shape[0]
        nr = shape[1]
        nt = shape[2]
        interp = rotor['interp']
        B = self.parameters['num_blades']
        M_block = rotor['M_block_matrix']
        M_inv_block = rotor['M_inv_block_matrix']
        psi = rotor['azimuth_angle']
        # mu = np.array([0])#rotor['mu']
        # mu_z = np.array([0.1])# rotor['mu_z']
        mu = inputs['mu']
        mu_z = inputs['mu_z']
        mu_z_exp = np.einsum(
            'i,ijk->ijk',
            mu_z,
            np.ones((ne, nr,nt)),  
        )

        dr = inputs['_dr']
        R = inputs['_rotor_radius']
        r = inputs['_radius']
        Re = inputs['_re_pitt_peters']
        # rho = rotor['density']
        chord = inputs['_chord']
        twist = inputs['_pitch']
        Vt = inputs['_tangential_inflow_velocity']
        Omega = inputs['_angular_speed']
        rho = inputs['density']
        rho_exp = inputs['_density_expanded']

        # compute solidity 
        sigma = B * chord / 2 / np.pi / r 
        
        normalized_radial_discretization = 1. / nr / 2. \
        + np.linspace(0., 1. - 1. / nr, nr)
        r_norm = np.einsum(
            'ik,j->ijk',
            np.ones((ne, nt)),
            normalized_radial_discretization,
        )

        lambda_0 = 0.01 * np.ones((ne,))#np.random.randn(shape[0],)#np.zeros((ne,))#0.01 * np.ones((ne,))#
        lambda_c = 0.01 * np.ones((ne,))#np.random.randn(shape[0],)#np.zeros((ne,))#
        lambda_s = 0.01 * np.ones((ne,))#np.random.randn(shape[0],)#np.zeros((ne,))#
        

        self.lamb = np.zeros((shape[0],3))
        self.C = np.zeros((shape[0],3))
        self.C_T = np.zeros((shape[0],))
        self.C_Mx = np.zeros((shape[0],))
        self.C_My = np.zeros((shape[0],))

        self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6

        # Cl = 0
        for j in range(500):
            # print(j,'iteration')
            lambda_0_exp = np.einsum(
                'i,ijk->ijk',
                lambda_0,
                np.ones((ne, nr,nt)),         
                )
            lambda_c_exp = np.einsum(
                'i,ijk->ijk',
                lambda_c,
                np.ones((ne, nr,nt)),         
                )
            lambda_s_exp = np.einsum(
                'i,ijk->ijk',
                lambda_s,
                np.ones((ne, nr,nt)),         
                )
            
            
            # Compute nu and self.ux (ne, nr, nt)
            self.lamb_exp = lambda_0_exp + lambda_c_exp * r_norm * np.cos(psi) + lambda_s_exp * r_norm * np.sin(psi)
            self.ux = (self.lamb_exp + mu_z_exp) * Omega * R
            # Compute inflow angle self.phi (ne, nr, nt)
            # ignore u_theta
            # self.phi = np.arctan(self.ux / Vt * (1 + sigma * Cl / 4))
            self.phi = np.arctan(self.ux / Vt)
            # u_theta = 2 * sigma * Cl * np.sin(self.phi) * Vt / (4 * np.sin(self.phi) * np.cos(self.phi) + sigma * Cl * np.sin(self.phi))
            # print(u_theta, 'u_theta')
            # Compute sectional AoA (ne, nr, nt)
            alpha = twist - self.phi 

            # Compute Cl, Cd  (ne, nr, nt)
            self.airfoil_model_inputs[:,0] = alpha.flatten()
            airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
            Cl = airfoil_model_outputs[:,:,:,0]
            Cd = airfoil_model_outputs[:,:,:,1]

            # self.dT = 0.5 * B * rho_exp * (self.ux**2 + (Vt - 0.5 * u_theta)**2) * chord * (Cl * np.cos(self.phi) - Cd * np.sin(self.phi)) * dr
            self.dT = 0.5 * B * rho_exp * (self.ux**2 + (Vt)**2) * chord * (Cl * np.cos(self.phi) - Cd * np.sin(self.phi)) * dr
            self.dC_T = self.dT / (rho_exp * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2)
            T = np.sum(np.sum(self.dT,axis=1),axis=1) / shape[2]
            # print(T)
            # self.dQ = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (Cl * np.sin(self.phi) + Cd * np.cos(self.phi)) * r * dr
            # Q = np.sum(np.sum(self.dQ,axis=1),axis=1) / shape[2]
            
            # Compute roll and pitch moments from thrust
            dMx = r * np.sin(psi) * self.dT
            Mx  = np.sum(np.sum(dMx,axis=1),axis=1) / shape[2]
            dMy = r * np.cos(psi) * self.dT
            My  = np.sum(np.sum(dMy,axis=1),axis=1) / shape[2]

            L = np.zeros((shape[0],3,3))
            L_inv = np.zeros((shape[0],3,3))
            L_list = []
            L_inv_list = []
            for i in range(ne):
                self.C_T[i] = T[i] / (rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)  #(Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
                self.C_Mx[i] = Mx[i] / (rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) # rho[i] / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
                self.C_My[i] = My[i] / (rho[i] * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) #  rho[i] / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
                # print(self.C_My[i],'C_My')
                # print(self.C_Mx[i],'C_Mx')
                lamb_i = np.mean(self.lamb_exp[i,:,:])
                lamb = lamb_i + mu_z[i]

                Chi = np.arctan(mu[i]/lamb)
                V_eff = (mu[i]**2 + lamb * (lamb + lamb_i)) / (mu[i]**2 + lamb**2)**0.5
                L[i,0,0] = 0.5
                L[i,0,1] = -15 * np.pi/64 * ((1 - np.cos(Chi))/(1 + np.cos(Chi)))**0.5
                L[i,1,1] = 4 * np.cos(Chi) / (1 + np.cos(Chi))
                L[i,1,0] = -L[i,0,1]
                L[i,2,2] = 4 / (1 + np.cos(Chi))

                L[i,:,:] = L[i,:,:] / V_eff
                L_list.append(L[i,:,:])

                # print(L[i,:,:])

                L_inv[i,:,:] = np.linalg.inv(L[i,:,:])
                L_inv_list.append(L_inv[i,:,:])

        
                self.C[i,0] = self.C_T[i]
                self.C[i,1] = -self.C_My[i]
                self.C[i,2] = self.C_Mx[i] 
                

                self.lamb[i,0] = lambda_0[i]
                self.lamb[i,1] = lambda_c[i]
                self.lamb[i,2] = lambda_s[i] 
                
            
            L_block = block_diag(*L_list)
            L_inv_block = block_diag(*L_inv_list)
            # self.C_new = C.reshape((shape[0],3,1))
            self.C_new = self.C.flatten()
            # nu = nu.reshape((shape[0],3,1))
            self.lamb_new = self.lamb.flatten()

            self.term1 = 0.5 * np.matmul(L_block,M_block)
            self.term2 = np.matmul(M_inv_block,self.C_new)
            self.term3 = np.matmul(M_inv_block,L_inv_block)
            self.term4 = np.matmul(self.term3,self.lamb_new)
            self.term5 = self.term2 - self.term4
            self.term6 = np.matmul(self.term1,self.term5)            
            self.lamb_new += self.term6
            if np.linalg.norm(self.lamb.flatten()-self.lamb_new) < 1e-15:
                # print('Pitt-Peters ODE solved for steady state')
                break
            


            # print(np.linalg.norm(self.lamb.flatten()-self.lamb_new))
            self.lamb = self.lamb_new.reshape((shape[0],3))
            
            lambda_0 = self.lamb[:,0]
            lambda_c = self.lamb[:,1]
            lambda_s = self.lamb[:,2]
        # print(Cl)
        # print(Cl.shape)
        # np.savetxt('txt_files/lift_distribution.txt',Cl.flatten() )
        # np.savetxt('txt_files/drag_distribution.txt',Cd.flatten() )
        # np.savetxt('txt_files/phi_distribution.txt',self.phi.flatten() )
        # print(self.phi.shape)
        # np.savetxt('txt_files/section_thrust_distribution.txt',self.dT.flatten() )
        # np.savetxt('txt_files/ux_distribution.txt',self.ux.flatten() )
        # np.savetxt('sectional_CT.txt',self.dC_T.flatten())
        if np.linalg.norm(self.lamb.flatten()-self.lamb_new) > 1e-15:
            print('Pitt-Peters not converged to tolerance!')
            print('Norm between successive iterates: {}'.format(np.linalg.norm(self.lamb.flatten()-self.lamb_new)))
        outputs['_lambda'] = self.lamb    

    def apply_inverse_jacobian( self, d_outputs, d_residuals, mode):
        shape = self.parameters['shape']
        ne = shape[0]
        if mode == 'fwd':
            d_outputs_shape = d_outputs['_lambda'].shape
            d_outputs['_lambda'] = np.matmul(self.inv_jac ,d_residuals['_lambda'].flatten()).reshape(d_outputs_shape)
        elif mode == 'rev':
            d_residuals_shape = d_residuals['_lambda'].shape
            d_residuals['_lambda'] = np.matmul(self.inv_jac.T, d_outputs['_lambda'].flatten()).reshape(d_residuals_shape)






# import numpy as np
# from csdl import Model
# import csdl
# # from lsdo_rotor.rotor_parameters import RotorParameters
# import openmdao.api as om
# from scipy.linalg import block_diag


# class PittPetersCustomImplicitOperation(csdl.CustomImplicitOperation):
#     def initialize(self):
#         self.parameters.declare('shape', types=tuple)
#         self.parameters.declare('rotor', types=RotorParameters)
    
#     def define(self):
#         shape = self.parameters['shape']
        

#         # self.add_input('L_block_diag_matrix', shape=(shape[0]*3,shape[0]*3))
#         # self.add_input('M_block_diag_matrix', shape=(shape[0]*3,shape[0]*3))

#         self.add_input('_re_pitt_peters', shape=shape)
#         self.add_input('_chord', shape=shape)
#         self.add_input('_pitch', shape=shape)
        
#         self.add_input('_tangential_inflow_velocity', shape=shape)
#         self.add_input('_dr', shape=shape)
#         self.add_input('_rotor_radius', shape=shape)
#         self.add_input('_radius', shape=shape)


#         self.add_output('_nu', shape=(shape[0],3))

#         # self.declare_derivatives('_nu', 'L_block_diag_matrix',method='exact')
#         # self.declare_derivatives('_nu', 'M_block_diag_matrix',method='exact')
#         self.declare_derivatives('_nu', '_dr',method='exact')
#         self.declare_derivatives('_nu', '_chord',method='exact')
#         self.declare_derivatives('_nu', '_pitch',method='exact')
#         self.declare_derivatives('_nu', '_radius',method='exact')
#         self.declare_derivatives('_nu', '_re_pitt_peters',method='exact')
#         self.declare_derivatives('_nu', '_tangential_inflow_velocity',method='exact')
#         self.declare_derivatives('_nu', '_rotor_radius',method='exact')
#         self.declare_derivatives('_nu','_nu',method='exact')
        
#         self.airfoil_model_inputs = np.zeros((shape[0] * shape[1] * shape[2], 2))

#     def evaluate_residuals(self,inputs,outputs,residuals):
#         rotor = self.parameters['rotor']
#         shape = self.parameters['shape']
#         ne = shape[0]
#         nr = shape[1]
#         nt = shape[2]
#         interp = rotor['interp']
#         B = rotor['num_blades']
#         i_vec = rotor['rotor_disk_tilt_angle'][0:ne]
#         rho = rotor['density']
#         angular_speed = rotor['Omega'][0:ne]
#         V_inf = rotor['V_infinity'][0:ne]
#         Omega = np.einsum(
#             'i,ijk->ijk',
#             angular_speed,
#             np.ones((ne, nr,nt)),         
#         )
        
#         dr = inputs['_dr']
#         R = inputs['_rotor_radius']
#         r = inputs['_radius']

#         # M (block matrices)
#         m = np.array([128/75/np.pi, -16/45/np.pi, -16/45/np.pi])
#         M  = np.diag(m)
#         M_inv = np.linalg.inv(M)
#         M_block_list = []
#         M_inv_block_list = []
#         for i in range(ne):
#             M_block_list.append(M)
#             M_inv_block_list.append(M_inv)
#         M_block = block_diag(*M_block_list)
#         M_inv_block = block_diag(*M_inv_block_list)

#         # L_block = inputs['L_block_diag_matrix']
#         # L_inv_block = np.linalg.inv(L_block)
#         # M_block = inputs['M_block_diag_matrix']
#         # M_inv_block = np.linalg.inv(M_block)

#         Re = inputs['_re_pitt_peters']
        
#         chord = inputs['_chord']
#         twist = inputs['_pitch']
#         Vt = inputs['_tangential_inflow_velocity']
#         v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
#         psi = np.einsum(
#             'ij,k->ijk',
#             np.ones((ne, nr)),
#             v,
#         )

#         normalized_radial_discretization = 1. / nr / 2. \
#         + np.linspace(0., 1. - 1. / nr, nr)
#         r_norm = np.einsum(
#             'ik,j->ijk',
#             np.ones((ne, nt)),
#             normalized_radial_discretization,
#         )

#         # nu_0 = np.zeros((shape[0],))
#         # nu_s = np.zeros((shape[0],))
#         # nu_c = np.zeros((shape[0],))

#         self.nu = outputs['_nu']
#         self.C = np.zeros((shape[0],3))
#         self.C_T = np.zeros((shape[0],))
#         self.C_L = np.zeros((shape[0],))
#         self.C_M = np.zeros((shape[0],))

#         self.mu_z = np.zeros((shape[0],))
#         self.mu = np.zeros((shape[0],))
#         for i in range(ne):
#             self.mu_z[i] = V_inf[i] * np.sin(i_vec[i]) / angular_speed[i] / R[i,0,0]
#             self.mu[i] = V_inf[i] * np.cos(i_vec[i]) / angular_speed[i] / R[i,0,0]

#         mu_z_exp = np.einsum(
#             'i,ijk->ijk',
#             self.mu_z,
#             np.ones((ne, nr,nt)),  
#         )

#         mu_exp = np.einsum(
#             'i,ijk->ijk',
#             self.mu,
#             np.ones((ne, nr,nt)),  
#         )

#         nu_0_exp = np.einsum(
#             'i,ijk->ijk',
#             self.nu[:,0],
#             np.ones((ne, nr,nt)),         
#             )
#         nu_s_exp = np.einsum(
#             'i,ijk->ijk',
#             self.nu[:,1],
#             np.ones((ne, nr,nt)),         
#             )
#         nu_c_exp = np.einsum(
#             'i,ijk->ijk',
#             self.nu[:,2],
#             np.ones((ne, nr,nt)),         
#             )
        
#         # Compute nu and self.ux (ne, nr, nt)
#         self.nu_exp = nu_0_exp + nu_s_exp * r_norm * np.sin(psi) + nu_c_exp * r_norm * np.cos(psi)
#         self.ux = (self.nu_exp + mu_z_exp) * Omega * R

#         # Compute inflow angle self.phi (ne, nr, nt)
#         # ignore u_theta
#         self.phi = np.arctan(self.ux / Vt)

#         # Compute sectional AoA (ne, nr, nt)
#         alpha = twist - self.phi 
#         # print(alpha,'ALPHA')
#         self.airfoil_model_inputs[:,0] = alpha.flatten()
#         self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6
#         self.airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
#         self.Cl = self.airfoil_model_outputs[:,:,:,0]
#         self.Cd = self.airfoil_model_outputs[:,:,:,1]

#         self.dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
#         self.T = np.sum(np.sum(self.dT,axis =1),axis = 1) / shape[2]
        
#         # Compute roll and pitch moments from thrust
#         dL_mom = r * np.sin(psi) * self.dT
#         L_mom  = np.sum(np.sum(dL_mom,axis=1),axis=1) / shape[2]
#         dM_mom = r * np.cos(psi) * self.dT
#         M_mom  = np.sum(np.sum(dM_mom,axis=1),axis=1) / shape[2]

#         # nu_vec = np.zeros((shape[0],3))
#         L = np.zeros((shape[0],3,3))
#         L_inv = np.zeros((shape[0],3,3))
#         L_list = []
#         L_inv_list = []
#         for i in range(ne):
#             self.C_T[i] = self.T[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)  #(Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
#             self.C_L[i] = L_mom[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) # rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
#             self.C_M[i] = M_mom[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) #  rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5

#             # Compute L (block) matrix
#             # coeff = [1, 2 * mu_z[i], mu[i]**2 + mu_z[i]**2, 0, -self.C_T[i]**2 / 4]
#             # roots = np.roots(coeff)
#             # real_roots = roots[np.isreal(roots)]
#             # positive_roots = lamb_i = np.real(real_roots[real_roots>0][0])
            
#             lamb_i = np.mean(self.nu_exp[i,:,:])
#             lamb = lamb_i + self.mu_z[i]
#             # lamb_i_test = self.C_T[i] / (2 * (mu[i]**2 + lamb**2)**0.5)

#             # print(lamb,'lamb_test_1')
#             # print(lamb_i,'lambda_solved')
#             # print(lamb_i_thrust,'lambda due to thrust')


#             Chi = np.arctan(self.mu[i]/lamb)
#             # print(Chi * 180/np.pi,'wake skew angle')
#             V_eff = (self.mu[i]**2 + lamb * (lamb + lamb_i)) / (self.mu[i]**2 + lamb**2)**0.5
#             L[i,0,0] = 0.5
#             L[i,0,1] = -15 * np.pi/64 * ((1 - np.cos(Chi))/(1 + np.cos(Chi)))**0.5
#             L[i,1,1] = 4 * np.cos(Chi) / (1 + np.cos(Chi))
#             L[i,1,0] = -L[i,0,1]
#             L[i,2,2] = 4 / (1 + np.cos(Chi))

#             L[i,:,:] = L[i,:,:] / V_eff
#             L_list.append(L[i,:,:])

#             L_inv[i,:,:] = np.linalg.inv(L[i,:,:])
#             L_inv_list.append(L_inv[i,:,:])

#             print(self.C_T,'C_T')
#             print(self.C_L,'C_L')
#             print(self.C_M,'C_M')
#             print(self.nu[:,0],'nu_0')
#             print(self.nu[:,1],'nu_c')
#             print(self.nu[:,2],'nu_s')
#             print(np.mean(r[i,:,:] * np.cos(psi[i,:,:])),'rcos_mean')
#             print(np.mean(r[i,:,:] * np.sin(psi[i,:,:])),'rsin_mean')
#             print(self.mu,'mu')
#             print(self.mu_z,'mu_z')
#             print(r,'r')
#             print(psi,'psi')
#             print(Vt,'V_theta')
#             print(Omega,'Omega')
#             print(R,'R')
#             print(self.phi,'phi')
#             print(self.Cl,'Cl')
#             print(self.Cd,'Cd')
#             print(mu_z_exp,'mu_z')
#             print(chord,'chord')
#             print(dr,'dr')
#             self.C[i,0] = self.C_T[i]
#             self.C[i,1] = -self.C_L[i]
#             self.C[i,2] = self.C_M[i] 

#             # self.nu[i,0] = nu_0[i]
#             # self.nu[i,1] = nu_s[i]
#             # self.nu[i,2] = nu_c[i] 
        
#         L_block = block_diag(*L_list)
#         L_inv_block = np.linalg.inv(L_block)#block_diag(*L_inv_list)
#         # self.C_new = C.reshape((shape[0],3,1))
#         self.C_new = self.C.flatten()
#         # nu = nu.reshape((shape[0],3,1))
#         self.nu_new = self.nu.flatten()

#         self.term1 = 0.5 * np.matmul(L_block,M_block)
#         self.term2 = np.matmul(M_inv_block,self.C_new)
#         self.term3 = np.matmul(M_inv_block,L_inv_block)
#         # print(0.1 * self.term3,'TERM 3')
#         self.term4 = np.matmul(self.term3,self.nu.flatten())
#         self.term5 = self.term2 - self.term4
#         self.term6 = np.matmul(self.term1,self.term5)
#         self.nu += self.term6.reshape((ne,3))

        
#         residuals['_nu'] = self.term6    

#     def compute_derivatives(self, inputs, outputs, derivatives):
#         shape = self.parameters['shape']
#         rotor = self.parameters['rotor']
#         B = rotor['num_blades']
#         interp = rotor['interp']
#         ne = shape[0]
#         nr = shape[1]
#         nt = shape[2]
        
#         # L_block = inputs['L_block_diag_matrix']
#         # L_inv_block = np.linalg.inv(L_block)
#         # M_block = inputs['M_block_diag_matrix']
#         # M_inv_block = np.linalg.inv(M_block)

#         Re = inputs['_re_pitt_peters']
#         rho = rotor['density']
#         V_inf = rotor['V_infinity'][0:ne]
#         i_vec = rotor['rotor_disk_tilt_angle'][0:ne]
#         chord = inputs['_chord']
#         twist = inputs['_pitch']
#         Vt = inputs['_tangential_inflow_velocity']

#         v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
#         psi = np.einsum(
#             'ij,k->ijk',
#             np.ones((ne, nr)),
#             v,
#         )
#         dr = inputs['_dr']
#         R = inputs['_rotor_radius']
#         r = inputs['_radius']

#         angular_speed = rotor['Omega'][0:ne]
#         Omega = np.einsum(
#             'i,ijk->ijk',
#             angular_speed,
#             np.ones((ne, nr,nt)),         
#         )

#         nu = outputs['_nu']

#         L = np.zeros((shape[0],3,3))
#         L_inv = np.zeros((shape[0],3,3))
#         L_list = []
#         L_inv_list = []
#         for i in range(ne):
#             # Compute L (block) matrix
#             lamb_i = np.mean(self.nu_exp[i,:,:])
#             lamb = lamb_i + self.mu_z[i]
#             Chi = np.arctan(self.mu[i]/lamb)
#             # print(Chi * 180/np.pi,'wake skew angle')
#             V_eff = (self.mu[i]**2 + lamb * (lamb + lamb_i)) / (self.mu[i]**2 + lamb**2)**0.5
#             L[i,0,0] = 0.5
#             L[i,0,1] = -15 * np.pi/64 * ((1 - np.cos(Chi))/(1 + np.cos(Chi)))**0.5
#             L[i,1,1] = 4 * np.cos(Chi) / (1 + np.cos(Chi))
#             L[i,1,0] = -L[i,0,1]
#             L[i,2,2] = 4 / (1 + np.cos(Chi))

#             L[i,:,:] = L[i,:,:] / V_eff
#             L_list.append(L[i,:,:])

#             L_inv[i,:,:] = np.linalg.inv(L[i,:,:])
#             L_inv_list.append(L_inv[i,:,:])

#         L_block = block_diag(*L_list)
#         L_inv_block = block_diag(*L_inv_list)


#         a = np.ones((1,nt))
#         b = np.ones((nr,1))
        
#         dC_ddT = np.zeros((ne*3,ne*nr*nt))
#         dC_dOmega = np.zeros((ne*3,ne*nr*nt))
#         dC_dR = np.zeros((ne*3,ne*nr*nt))
#         dC_rad = np.zeros((ne*3,ne*nr*nt))
#         vec = np.zeros((nr*nt,))
#         vec[0]= 1

#         ddT_dr_list = []
#         ddT_dc_list = []
#         ddT_dalpha_list = []
#         ddT_dre_list = []
#         ddT_dVt_list = []
#         ddT_domega_list = []

#         dCl_dalpha = interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,0].reshape((ne,nr,nt))
#         dCl_dre = (interp.predict_derivatives(self.airfoil_model_inputs, 1)[:,0] / 2e6 ).reshape((ne,nr,nt))
#         for i in range(ne):
#             D = 2 * R[i,0,0]
#             n = Omega[i,0,0] / 2 / np.pi
#             dC_dCT = np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)
#             dC_dCL = (np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.cos(psi[i,:,:])
#             dC_dCM = (np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.sin(psi[i,:,:])
#             dC_ddT[3*i , i*nr*nt:(i+1)*nr*nt] = dC_dCT.flatten()
#             dC_ddT[3*i+1 , i*nr*nt:(i+1)*nr*nt] = -dC_dCL.flatten()
#             dC_ddT[3*i+2 , i*nr*nt:(i+1)*nr*nt] = dC_dCM.flatten()

#             dC_dR1 = -8 * np.matmul(np.matmul(np.transpose(b),self.dT[i,:,:]),np.transpose(a)) / (nt * rho * n**2 * (2 * R[i,0,0])**5)
#             dC_dR2 = -10 * np.matmul(np.matmul(np.transpose(b),self.dT[i,:,:] * r[i,:,:] * np.sin(psi[i,:,:])),np.transpose(a)) / (nt * rho * n**2 * (2 * R[i,:,:])**6)
#             dC_dR3 = -10 * np.matmul(np.matmul(np.transpose(b),self.dT[i,:,:] * r[i,:,:] * np.cos(psi[i,:,:])),np.transpose(a)) / (nt * rho * n**2 * (2 * R[i,:,:])**6)
#             dC_dR[3*i , i*nr*nt:(i+1)*nr*nt] = dC_dR1.flatten() * vec
#             dC_dR[3*i+1 , i*nr*nt:(i+1)*nr*nt] = -dC_dR2.flatten() * vec
#             dC_dR[3*i+2 , i*nr*nt:(i+1)*nr*nt] = -dC_dR3.flatten() * vec
     
#             dC_dr1 = np.matmul(b,a) / (nt * rho * n**2 * D**4)
#             dC_dr2 = (np.matmul(b,a) / (nt * rho * n**2 * D**5))  * np.sin(psi[i,:,:]) * self.dT[i,:,:]
#             dC_dr3 = (np.matmul(b,a) / (nt * rho * n**2 * D**5))  * np.cos(psi[i,:,:]) * self.dT[i,:,:]
#             dC_rad[3*i , i*nr*nt:(i+1)*nr*nt] = dC_dr1.flatten() 
#             dC_rad[3*i+1 , i*nr*nt:(i+1)*nr*nt] = -dC_dr2.flatten()
#             dC_rad[3*i+2 , i*nr*nt:(i+1)*nr*nt] = -dC_dr3.flatten()

#             DR = dr[i,0,0]
#             c = chord[i,:,:]

#             ddT_dr_list.append(self.dT[i,:,:].flatten()/DR)
#             ddT_dc_list.append((self.dT[i,:,:]/c).flatten())

#             ddT_dCl = 0.5 * B * rho * (self.ux[i,:,:]**2 + Vt[i,:,:]**2) * chord[i,:,:] * np.cos(self.phi[i,:,:]) * dr[i,:,:]
#             ddT_dalpha = ddT_dCl * dCl_dalpha[i,:,:]
#             dalpha_dphi = 1
#             dphi_dVt = - self.ux[i,:,:] / (self.ux[i,:,:]**2 + Vt[i,:,:]**2)
#             ddT_dVt1 = ddT_dCl * ddT_dalpha * dalpha_dphi * dphi_dVt
#             ddT_dalpha_list.append(ddT_dalpha.flatten())
#             # print(self.ux)
            
#             ddT_dre = ddT_dCl * dCl_dre[i,:,:]
#             ddT_dre_list.append(ddT_dre.flatten())

#             ddT_dVt2 = self.dT[i,:,:] * 2 * Vt[i,:,:] / (self.ux[i,:,:]**2 + Vt[i,:,:]**2)
#             ddT_dVt_list.append((ddT_dVt1).flatten())

        
#         if ne == 1:
#             ddT_dr = np.diag(ddT_dr_list[0])
#             ddT_dc = np.diag(ddT_dc_list[0])
#             ddT_dalpha = np.diag(ddT_dalpha_list[0])
#             ddT_dre = np.diag(ddT_dre_list[0])
#             ddT_dVt = np.diag(ddT_dVt_list[0])
#         else:
#             ddT_dr = np.diag(np.append(*ddT_dr_list))
#             ddT_dc = np.diag(np.append(*ddT_dc_list))
#             ddT_dalpha = np.diag(np.append(*ddT_dalpha_list))
#             ddT_dre = np.diag(np.append(*ddT_dre_list))
#             ddT_dVt = np.diag(np.append(*ddT_dVt_list))
    
        
#         dnu_dC = 0.5 * L_block

#         dnu_ddr = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dr)
#         print(dnu_ddr.shape,'SHAPE')
#         dnu_dc = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dc)
#         dnu_dpitch = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dalpha)
#         dnu_dre = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dre)
#         dnu_dVt = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dVt)
    

#         dnu_dL = 0.5 * self.C_new 
#         dnu_dM = 0
#         dnu_dL_list = []
#         for i in range(ne*3):
#             dnu_dL_list.append(dnu_dL)

#         # derivatives['_nu','M_block_diag_matrix'] = dnu_dM
#         # derivatives['_nu','L_block_diag_matrix'] = block_diag(*dnu_dL_list)
#         derivatives['_nu','_dr'] = np.array([[-0.00079923, -0.00079923, -0.00079923, -0.00079923, -0.00079923 , 0.00087521,
#                                                 0.00087521 , 0.00087521 , 0.00087521,  0.00087521,  0.00068109 , 0.00068109,
#                                                 0.00068109 , 0.00068109,  0.00068109],
#                                                 [ 0. ,         0.00101348 , 0.00062637, -0.00062637 ,-0.00101348 , 0.,
#                                                 -0.00199771, -0.00123465 , 0.00123465 , 0.00199771,  0.  ,      -0.00225418,
#                                                 -0.00139316, 0.00139316 , 0.00225418],
#                                                 [-0.00106564, -0.0003293, 0.00086212, 0.00086212, -0.0003293, 0.00210052,
#                                                 0.00064909 ,-0.00169935, -0.00169935 , 0.00064909, 0.00237019, 0.00073243,
#                                                 -0.00191752 ,-0.00191752 , 0.00073243]]
#                                             )#dnu_ddr
#         derivatives['_nu','_chord'] = dnu_dc
#         derivatives['_nu','_pitch'] = dnu_dpitch
        
#         derivatives['_nu','_re_pitt_peters'] = dnu_dre
#         derivatives['_nu','_tangential_inflow_velocity'] = dnu_dVt
#         derivatives['_nu','_radius'] = np.matmul(dnu_dC,dC_rad)#np.matmul(np.matmul(0.1 * L_block,dC_rad),dQ_dr)
#         derivatives['_nu','_rotor_radius'] = np.matmul(0.5 * L_block,dC_dR)
        
#         derivatives['_nu', '_nu'] = np.array([[-5.08732005e-01, -3.94961210e-03, -2.41733527e-03],
#                                               [ 4.14233087e-11, -5.00000000e-01,  1.45254093e-11],
#                                               [-2.83369141e-11, -1.78244645e-11, -5.00000000e-01]])

#         self.inv_jac = np.linalg.inv(derivatives['_nu', '_nu'])




#     def solve_residual_equations(self, inputs,outputs):
#         rotor = self.parameters['rotor']
#         shape = self.parameters['shape']
#         ne = shape[0]
#         nr = shape[1]
#         nt = shape[2]
#         interp = rotor['interp']
#         B = rotor['num_blades']
#         i_vec = rotor['rotor_disk_tilt_angle'][0:ne]
#         V_inf = rotor['V_infinity'][0:ne]
#         # L_block = inputs['L_block_diag_matrix']
#         # L_inv_block = np.linalg.inv(L_block)
#         # M_block = inputs['M_block_diag_matrix']
#         # M_inv_block = np.linalg.inv(M_block)

#         dr = inputs['_dr']
#         R = inputs['_rotor_radius']
#         r = inputs['_radius']

#         m = np.array([128/75/np.pi, -16/45/np.pi, -16/45/np.pi])
#         M  = np.diag(m)
#         M_inv = np.linalg.inv(M)
#         M_block_list = []
#         M_inv_block_list = []
#         for i in range(ne):
#             M_block_list.append(M)
#             M_inv_block_list.append(M_inv)
#         M_block = block_diag(*M_block_list)
#         M_inv_block = block_diag(*M_inv_block_list)

#         Re = inputs['_re_pitt_peters']
#         rho = rotor['density']
#         chord = inputs['_chord']
#         twist = inputs['_pitch']
#         Vt = inputs['_tangential_inflow_velocity']
#         v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
#         psi = np.einsum(
#             'ij,k->ijk',
#             np.ones((ne, nr)),
#             v,
#         )
#         angular_speed = rotor['Omega'][0:ne]
#         Omega = np.einsum(
#             'i,ijk->ijk',
#             angular_speed,
#             np.ones((ne, nr,nt)),         
#         )
        
#         normalized_radial_discretization = 1. / nr / 2. \
#         + np.linspace(0., 1. - 1. / nr, nr)
#         r_norm = np.einsum(
#             'ik,j->ijk',
#             np.ones((ne, nt)),
#             normalized_radial_discretization,
#         )

#         mu_z = np.zeros((shape[0],))
#         mu = np.zeros((shape[0],))
#         for i in range(ne):
#             mu_z[i] = V_inf[i] * np.sin(i_vec[i]) / angular_speed[i] / R[i,0,0]
#             mu[i] = V_inf[i] * np.cos(i_vec[i]) / angular_speed[i] / R[i,0,0]

#         mu_z_exp = np.einsum(
#             'i,ijk->ijk',
#             mu_z,
#             np.ones((ne, nr,nt)),  
#         )

#         mu_exp = np.einsum(
#             'i,ijk->ijk',
#             mu,
#             np.ones((ne, nr,nt)),  
#         )

#         nu_0 = np.zeros((ne,))#np.random.randn(shape[0],)
#         nu_s = np.zeros((ne,))#np.random.randn(shape[0],)
#         nu_c = np.zeros((ne,))#np.random.randn(shape[0],)

#         self.nu = np.zeros((shape[0],3))
#         self.C = np.zeros((shape[0],3))
#         self.C_T = np.zeros((shape[0],))
#         self.C_L = np.zeros((shape[0],))
#         self.C_M = np.zeros((shape[0],))
#         # nu = np.zeros((shape[0],3,1))

#         self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6

#         for j in range(400):
#             print(j,'iteration')
#             nu_0_exp = np.einsum(
#                 'i,ijk->ijk',
#                 nu_0,
#                 np.ones((ne, nr,nt)),         
#                 )
#             nu_s_exp = np.einsum(
#                 'i,ijk->ijk',
#                 nu_s,
#                 np.ones((ne, nr,nt)),         
#                 )
#             nu_c_exp = np.einsum(
#                 'i,ijk->ijk',
#                 nu_c,
#                 np.ones((ne, nr,nt)),         
#                 )
            
#             # Compute nu and self.ux (ne, nr, nt)
#             self.nu_exp = nu_0_exp + nu_s_exp * r_norm * np.sin(psi) + nu_c_exp * r_norm * np.cos(psi)
#             self.ux = (self.nu_exp + mu_z_exp) * Omega * R
#             # self.ux = self.nu_exp * Omega * R
#             # Compute inflow angle self.phi (ne, nr, nt)
#             # ignore u_theta
#             self.phi = np.arctan(self.ux / Vt)

#             # Compute sectional AoA (ne, nr, nt)
#             alpha = twist - self.phi 

#             # Compute Cl, Cd  (ne, nr, nt)
#             self.airfoil_model_inputs[:,0] = alpha.flatten()
#             airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
#             Cl = airfoil_model_outputs[:,:,:,0]
#             Cd = airfoil_model_outputs[:,:,:,1]

#             self.dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (Cl * np.cos(self.phi) - Cd * np.sin(self.phi)) * dr
#             T = np.sum(np.sum(self.dT,axis=1),axis=1) / shape[2]
#             print(T)
#             # self.dQ = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (Cl * np.sin(self.phi) + Cd * np.cos(self.phi)) * r * dr
#             # Q = np.sum(np.sum(self.dQ,axis=1),axis=1) / shape[2]
            
#             # Compute roll and pitch moments from thrust
#             dL_mom = r * np.sin(psi) * self.dT
#             L_mom  = np.sum(np.sum(dL_mom,axis=1),axis=1) / shape[2]
#             dM_mom = r * np.cos(psi) * self.dT
#             M_mom  = np.sum(np.sum(dM_mom,axis=1),axis=1) / shape[2]

#             L = np.zeros((shape[0],3,3))
#             L_inv = np.zeros((shape[0],3,3))
#             L_list = []
#             L_inv_list = []
#             for i in range(ne):
#                 self.C_T[i] = T[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)  #(Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
#                 self.C_L[i] = L_mom[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) # rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
#                 self.C_M[i] = M_mom[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) #  rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5

#                 # Compute L (block) matrix
#                 # coeff = [1, 2 * mu_z[i], mu[i]**2 + mu_z[i]**2, 0, -self.C_T[i]**2 / 4]
#                 # roots = np.roots(coeff)
#                 # real_roots = roots[np.isreal(roots)]
#                 # positive_roots = lamb_i = np.real(real_roots[real_roots>0][0])
                
#                 lamb_i = np.mean(self.nu_exp[i,:,:])
#                 lamb = lamb_i + mu_z[i]
#                 # lamb_i_test = self.C_T[i] / (2 * (mu[i]**2 + lamb**2)**0.5)
   
#                 # print(lamb,'lamb_test_1')
#                 # print(lamb_i,'lambda_solved')
#                 # print(lamb_i_thrust,'lambda due to thrust')


#                 Chi = np.arctan(mu[i]/lamb)
#                 # print(Chi * 180/np.pi,'wake skew angle')
#                 V_eff = (mu[i]**2 + lamb * (lamb + lamb_i)) / (mu[i]**2 + lamb**2)**0.5
#                 L[i,0,0] = 0.5
#                 L[i,0,1] = -15 * np.pi/64 * ((1 - np.cos(Chi))/(1 + np.cos(Chi)))**0.5
#                 L[i,1,1] = 4 * np.cos(Chi) / (1 + np.cos(Chi))
#                 L[i,1,0] = -L[i,0,1]
#                 L[i,2,2] = 4 / (1 + np.cos(Chi))

#                 L[i,:,:] = L[i,:,:] / V_eff
#                 L_list.append(L[i,:,:])

#                 L_inv[i,:,:] = np.linalg.inv(L[i,:,:])
#                 L_inv_list.append(L_inv[i,:,:])

        
#                 self.C[i,0] = self.C_T[i]
#                 self.C[i,1] = -self.C_L[i]
#                 self.C[i,2] = self.C_M[i] 

#                 self.nu[i,0] = nu_0[i]
#                 self.nu[i,1] = nu_s[i]
#                 self.nu[i,2] = nu_c[i] 
                
            
#             L_block = block_diag(*L_list)
#             L_inv_block = block_diag(*L_inv_list)
#             # self.C_new = C.reshape((shape[0],3,1))
#             self.C_new = self.C.flatten()
#             # nu = nu.reshape((shape[0],3,1))
#             self.nu_new = self.nu.flatten()

#             self.term1 = 0.5 * np.matmul(L_block,M_block)
#             self.term2 = np.matmul(M_inv_block,self.C_new)
#             self.term3 = np.matmul(M_inv_block,L_inv_block)
#             self.term4 = np.matmul(self.term3,self.nu_new)
#             self.term5 = self.term2 - self.term4
#             self.term6 = np.matmul(self.term1,self.term5)            
#             self.nu_new += self.term6
#             if np.linalg.norm(self.nu.flatten()-self.nu_new) < 1e-15:
#                 print('Pitt-Peters ODE solved for steady state')
#                 break


#             self.nu = self.nu_new.reshape((shape[0],3))
            
#             nu_0 = self.nu[:,0]
            
#             nu_s = self.nu[:,1]
#             # print(nu_c,'nu_s')
#             nu_c = self.nu[:,2]

#         outputs['_nu'] = self.nu    

#     def apply_inverse_jacobian( self, d_outputs, d_residuals, mode):
#         shape = self.parameters['shape']
#         ne = shape[0]
#         if mode == 'fwd':
#             d_outputs['_nu'] = self.inv_jac * d_residuals['_nu'].reshape((ne,3))
#         elif mode == 'rev':
#             d_residuals['_nu'] = np.matmul(self.inv_jac, d_outputs['_nu'].flatten()).reshape(ne,3)
        

# import numpy as np
# from csdl import Model
# import csdl
# # from lsdo_rotor.rotor_parameters import RotorParameters
# import openmdao.api as om
# from scipy.linalg import block_diag


# class PittPetersCustomImplicitOperation(csdl.CustomImplicitOperation):
#     def initialize(self):
#         self.parameters.declare('shape', types=tuple)
#         self.parameters.declare('rotor', types=RotorParameters)
    
#     def define(self):
#         shape = self.parameters['shape']
        

#         self.add_input('L_block_diag_matrix', shape=(shape[0]*3,shape[0]*3))
#         self.add_input('M_block_diag_matrix', shape=(shape[0]*3,shape[0]*3))

#         self.add_input('_re_pitt_peters', shape=shape)
#         self.add_input('_chord', shape=shape)
#         self.add_input('_pitch', shape=shape)
        
#         self.add_input('_tangential_inflow_velocity', shape=shape)
#         self.add_input('_dr', shape=shape)
#         self.add_input('_rotor_radius', shape=shape)
#         self.add_input('_radius', shape=shape)


#         # self.add_output('_nu', shape=(shape[0],3))
#         self.add_output('_nu_0', shape=(shape[0]))
#         self.add_output('_nu_s', shape=(shape[0]))
#         self.add_output('_nu_c', shape=(shape[0]))

#         # self.declare_derivatives('_nu', 'L_block_diag_matrix',method='exact')
#         # self.declare_derivatives('_nu', 'M_block_diag_matrix',method='exact')
#         # self.declare_derivatives('_nu', '_dr',method='exact')
#         # self.declare_derivatives('_nu', '_chord',method='exact')
#         # self.declare_derivatives('_nu', '_pitch',method='exact')
#         # self.declare_derivatives('_nu', '_radius',method='exact')
#         # self.declare_derivatives('_nu', '_re_pitt_peters',method='exact')
#         # self.declare_derivatives('_nu', '_tangential_inflow_velocity',method='exact')
#         # self.declare_derivatives('_nu', '_rotor_radius',method='exact')
#         # self.declare_derivatives('_nu','_nu',method='exact')

#         self.declare_derivatives('_nu_0', 'L_block_diag_matrix',method='exact')
#         self.declare_derivatives('_nu_s', 'L_block_diag_matrix',method='exact')
#         self.declare_derivatives('_nu_c', 'L_block_diag_matrix',method='exact')
#         self.declare_derivatives('_nu_0', 'M_block_diag_matrix',method='exact')
#         self.declare_derivatives('_nu_0', '_dr',method='exact')
#         self.declare_derivatives('_nu_0', '_chord',method='exact')
#         self.declare_derivatives('_nu_0', '_pitch',method='exact')
#         self.declare_derivatives('_nu_0', '_radius',method='exact')
#         self.declare_derivatives('_nu_0', '_re_pitt_peters',method='exact')
#         self.declare_derivatives('_nu_0', '_tangential_inflow_velocity',method='exact')
#         self.declare_derivatives('_nu_0', '_rotor_radius',method='exact')
#         self.declare_derivatives('_nu_0','_nu_0',method='exact')

#         self.declare_derivatives('_nu_s', '_dr',method='exact')
#         self.declare_derivatives('_nu_s', '_chord',method='exact')
#         self.declare_derivatives('_nu_s', '_pitch',method='exact')
#         self.declare_derivatives('_nu_s', '_radius',method='exact')
#         self.declare_derivatives('_nu_s', '_re_pitt_peters',method='exact')
#         self.declare_derivatives('_nu_s', '_tangential_inflow_velocity',method='exact')
#         self.declare_derivatives('_nu_s', '_rotor_radius',method='exact')
#         self.declare_derivatives('_nu_s','_nu_0',method='exact')

#         self.declare_derivatives('_nu_c', '_dr',method='exact')
#         self.declare_derivatives('_nu_c', '_chord',method='exact')
#         self.declare_derivatives('_nu_c', '_pitch',method='exact')
#         self.declare_derivatives('_nu_c', '_radius',method='exact')
#         self.declare_derivatives('_nu_c', '_re_pitt_peters',method='exact')
#         self.declare_derivatives('_nu_c', '_tangential_inflow_velocity',method='exact')
#         self.declare_derivatives('_nu_c', '_rotor_radius',method='exact')
#         self.declare_derivatives('_nu_c','_nu_0',method='exact')
        
#         self.airfoil_model_inputs = np.zeros((shape[0] * shape[1] * shape[2], 2))

#     def evaluate_residuals(self,inputs,outputs,residuals):
#         rotor = self.parameters['rotor']
#         shape = self.parameters['shape']
#         ne = shape[0]
#         nr = shape[1]
#         nt = shape[2]
#         interp = rotor['interp']
#         B = rotor['num_blades']
#         beta = rotor['rotor_disk_tilt_angle']
#         rho = rotor['density']
#         angular_speed = rotor['Omega']
#         Omega = np.einsum(
#             'i,ijk->ijk',
#             angular_speed,
#             np.ones((ne, nr,nt)),         
#         )
        
#         L_block = inputs['L_block_diag_matrix']
#         L_inv_block = np.linalg.inv(L_block)
#         M_block = inputs['M_block_diag_matrix']
#         M_inv_block = np.linalg.inv(M_block)

#         Re = inputs['_re_pitt_peters']
        
#         chord = inputs['_chord']
#         twist = inputs['_pitch']
#         Vt = inputs['_tangential_inflow_velocity']
#         v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
#         psi = np.einsum(
#             'ij,k->ijk',
#             np.ones((ne, nr)),
#             v,
#         )

#         normalized_radial_discretization = 1. / nr / 2. \
#         + np.linspace(0., 1. - 1. / nr, nr)
#         r_norm = np.einsum(
#             'ik,j->ijk',
#             np.ones((ne, nt)),
#             normalized_radial_discretization,
#         )
#         dr = inputs['_dr']
#         R = inputs['_rotor_radius']
#         r = inputs['_radius']

#         # nu_0 = np.zeros((shape[0],))
#         # nu_s = np.zeros((shape[0],))
#         # nu_c = np.zeros((shape[0],))

#         nu_0 = outputs['_nu_0']
#         nu_s = outputs['_nu_s']
#         nu_c = outputs['_nu_c']
        
#         self.nu =  np.zeros((shape[0],3))
#         # self.nu = outputs['_nu']
#         self.C = np.zeros((shape[0],3))
#         self.C_T = np.zeros((shape[0],))
#         self.C_L = np.zeros((shape[0],))
#         self.C_M = np.zeros((shape[0],))



#         nu_0_exp = np.einsum(
#             'i,ijk->ijk',
#             nu_0,
#             np.ones((ne, nr,nt)),         
#             )
#         nu_s_exp = np.einsum(
#             'i,ijk->ijk',
#             nu_s,
#             np.ones((ne, nr,nt)),         
#             )
#         nu_c_exp = np.einsum(
#             'i,ijk->ijk',
#             nu_c,
#             np.ones((ne, nr,nt)),         
#             )
        
#         # Compute nu and self.ux (ne, nr, nt)
#         self.nu_exp = nu_0_exp + nu_s_exp * r_norm * np.sin(psi) + nu_c_exp * r_norm * np.cos(psi)
#         self.ux = self.nu_exp * Omega * R

#         # Compute inflow angle self.phi (ne, nr, nt)
#         # ignore u_theta
#         self.phi = np.arctan(self.ux / Vt)

#         # Compute sectional AoA (ne, nr, nt)
#         alpha = twist - self.phi 
#         # print(alpha,'ALPHA')
#         self.airfoil_model_inputs[:,0] = alpha.flatten()
#         self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6
#         self.airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
#         self.Cl = self.airfoil_model_outputs[:,:,:,0]
#         self.Cd = self.airfoil_model_outputs[:,:,:,1]

#         self.dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
#         T = np.sum(np.sum(self.dT,axis =1),axis = 1) / shape[2]
        
#         # Compute roll and pitch moments from thrust
#         dL_mom = r * np.sin(psi) * self.dT
#         L_mom  = np.sum(np.sum(dL_mom,axis=1),axis=1) / shape[2]
#         dM_mom = r * np.cos(psi) * self.dT
#         M_mom  = np.sum(np.sum(dM_mom,axis=1),axis=1) / shape[2]

#         # nu_vec = np.zeros((shape[0],3))
#         for i in range(ne):
#             self.C_T[i] = T[i] / rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
#             self.C_L[i] = L_mom[i] / rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
#             self.C_M[i] = M_mom[i] / rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5

#             self.C[i,0] = self.C_T[i]
#             self.C[i,1] = -self.C_L[i]
#             self.C[i,2] = -self.C_M[i] 

#             self.nu[i,0] = nu_0[i]
#             self.nu[i,1] = nu_s[i]
#             self.nu[i,2] = nu_c[i] 
        
#         # self.C_new = self.C.reshape((shape[0],3,1))
#         self.C_new = self.C.flatten()
#         # self.nu = self.nu.reshape((shape[0],3,1))
#         # self.nu = self.nu.flatten()

#         self.term1 = 0.05 * np.matmul(L_block,M_block)
#         self.term2 = np.matmul(M_inv_block,self.C_new)
#         self.term3 = np.matmul(M_inv_block,L_inv_block)
#         self.term4 = np.matmul(self.term3,self.nu.flatten())
#         self.term5 = self.term2 - self.term4
#         self.term6 = np.matmul(self.term1,self.term5)
#         self.nu += self.term6.reshape((ne,3))

        
#         residuals['_nu_0'] = self.nu[:,0]
#         residuals['_nu_s'] = self.nu[:,1]
#         residuals['_nu_c'] = self.nu[:,2]    

#     def compute_derivatives(self, inputs, outputs, derivatives):
#         shape = self.parameters['shape']
#         rotor = self.parameters['rotor']
#         B = rotor['num_blades']
#         interp = rotor['interp']
#         ne = shape[0]
#         nr = shape[1]
#         nt = shape[2]
        
#         L_block = inputs['L_block_diag_matrix']
#         L_inv_block = np.linalg.inv(L_block)
#         M_block = inputs['M_block_diag_matrix']
#         M_inv_block = np.linalg.inv(M_block)

#         Re = inputs['_re_pitt_peters']
#         rho = rotor['density']
#         chord = inputs['_chord']
#         twist = inputs['_pitch']
#         Vt = inputs['_tangential_inflow_velocity']

#         v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
#         psi = np.einsum(
#             'ij,k->ijk',
#             np.ones((ne, nr)),
#             v,
#         )
#         dr = inputs['_dr']
#         R = inputs['_rotor_radius']
#         r = inputs['_radius']

#         angular_speed = rotor['Omega']
#         Omega = np.einsum(
#             'i,ijk->ijk',
#             angular_speed,
#             np.ones((ne, nr,nt)),         
#         )

#         # nu = outputs['_nu']

        


#         a = np.ones((1,nt))
#         b = np.ones((nr,1))
        
#         dC_ddT = np.zeros((ne*3,ne*nr*nt))
#         dC_dOmega = np.zeros((ne*3,ne*nr*nt))
#         dC_dR = np.zeros((ne*3,ne*nr*nt))
#         dC_rad = np.zeros((ne*3,ne*nr*nt))
#         vec = np.zeros((nr*nt,))
#         vec[0]= 1

#         ddT_dr_list = []
#         ddT_dc_list = []
#         ddT_dalpha_list = []
#         ddT_dre_list = []
#         ddT_dVt_list = []
#         ddT_domega_list = []

#         dCl_dalpha = interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,0].reshape((ne,nr,nt))
#         dCl_dre = (interp.predict_derivatives(self.airfoil_model_inputs, 1)[:,0] / 2e6 ).reshape((ne,nr,nt))
#         for i in range(ne):
#             D = 2 * R[i,0,0]
#             n = Omega[i,0,0] / 2 / np.pi
#             dC_dCT = np.matmul(b,a) / (nt * rho * n**2 * D**4)
#             dC_dCL = (np.matmul(b,a) / (nt * rho * n**2 * D**5)) * r[i,:,:] * np.sin(psi[i,:,:])
#             dC_dCM = (np.matmul(b,a) / (nt * rho * n**2 * D**5)) * r[i,:,:] * np.cos(psi[i,:,:])
#             dC_ddT[3*i , i*nr*nt:(i+1)*nr*nt] = dC_dCT.flatten()
#             dC_ddT[3*i+1 , i*nr*nt:(i+1)*nr*nt] = -dC_dCL.flatten()
#             dC_ddT[3*i+2 , i*nr*nt:(i+1)*nr*nt] = -dC_dCM.flatten()

#             dC_dR1 = -8 * np.matmul(np.matmul(np.transpose(b),self.dT[i,:,:]),np.transpose(a)) / (nt * rho * n**2 * (2 * R[i,0,0])**5)
#             dC_dR2 = -10 * np.matmul(np.matmul(np.transpose(b),self.dT[i,:,:] * r[i,:,:] * np.sin(psi[i,:,:])),np.transpose(a)) / (nt * rho * n**2 * (2 * R[i,:,:])**6)
#             dC_dR3 = -10 * np.matmul(np.matmul(np.transpose(b),self.dT[i,:,:] * r[i,:,:] * np.cos(psi[i,:,:])),np.transpose(a)) / (nt * rho * n**2 * (2 * R[i,:,:])**6)
#             dC_dR[3*i , i*nr*nt:(i+1)*nr*nt] = dC_dR1.flatten() * vec
#             dC_dR[3*i+1 , i*nr*nt:(i+1)*nr*nt] = -dC_dR2.flatten() * vec
#             dC_dR[3*i+2 , i*nr*nt:(i+1)*nr*nt] = -dC_dR3.flatten() * vec
     
#             dC_dr1 = np.matmul(b,a) / (nt * rho * n**2 * D**4)
#             dC_dr2 = (np.matmul(b,a) / (nt * rho * n**2 * D**5))  * np.sin(psi[i,:,:]) * self.dT[i,:,:]
#             dC_dr3 = (np.matmul(b,a) / (nt * rho * n**2 * D**5))  * np.cos(psi[i,:,:]) * self.dT[i,:,:]
#             dC_rad[3*i , i*nr*nt:(i+1)*nr*nt] = dC_dr1.flatten() 
#             dC_rad[3*i+1 , i*nr*nt:(i+1)*nr*nt] = -dC_dr2.flatten()
#             dC_rad[3*i+2 , i*nr*nt:(i+1)*nr*nt] = -dC_dr3.flatten()

#             DR = dr[i,0,0]
#             c = chord[i,:,:]

#             ddT_dr_list.append(self.dT[i,:,:].flatten()/DR)
#             ddT_dc_list.append((self.dT[i,:,:]/c).flatten())

#             ddT_dCl = 0.5 * B * rho * (self.ux[i,:,:]**2 + Vt[i,:,:]**2) * chord[i,:,:] * np.cos(self.phi[i,:,:]) * dr[i,:,:]
#             ddT_dalpha = ddT_dCl * dCl_dalpha[i,:,:]
#             ddT_dalpha_list.append(ddT_dalpha.flatten())
            
#             ddT_dre = ddT_dCl * dCl_dre[i,:,:]
#             ddT_dre_list.append(ddT_dre.flatten())

#             ddT_dVt = self.dT[i,:,:] * 2 * Vt[i,:,:] / (self.ux[i,:,:]**2 + Vt[i,:,:]**2)
#             ddT_dVt_list.append(ddT_dVt.flatten())

        
#         if ne == 1:
#             ddT_dr = np.diag(ddT_dr_list[0])
#             ddT_dc = np.diag(ddT_dc_list[0])
#             ddT_dalpha = np.diag(ddT_dalpha_list[0])
#             ddT_dre = np.diag(ddT_dre_list[0])
#             ddT_dVt = np.diag(ddT_dVt_list[0])
#         else:
#             ddT_dr = np.diag(np.append(*ddT_dr_list))
#             ddT_dc = np.diag(np.append(*ddT_dc_list))
#             ddT_dalpha = np.diag(np.append(*ddT_dalpha_list))
#             ddT_dre = np.diag(np.append(*ddT_dre_list))
#             ddT_dVt = np.diag(np.append(*ddT_dVt_list))
    
        
#         dnu_dC = 0.05 * L_block

#         dnu_ddr = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dr)
#         dnu_dc = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dc)
#         dnu_dpitch = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dalpha)
#         dnu_dre = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dre)
#         dnu_dVt = np.matmul(np.matmul(dnu_dC, dC_ddT),ddT_dVt)



#         dnu0_dL = np.zeros((ne,(ne*3)**2))
#         dnus_dL = np.zeros((ne,(ne*3)**2))
#         dnuc_dL = np.zeros((ne,(ne*3)**2))
#         for i in range(ne):
#             dnu0_dL[i,i*ne*3:i*ne*3+ne*3] = 0.05 * self.C_new
#             dnus_dL[i,(i+1)*ne*3:(i+1)*ne*3+ne*3] = 0.05 * self.C_new
#             dnuc_dL[i,(i+2)*ne*3:(i+2)*ne*3+ne*3] = 0.05 * self.C_new
        
#         dnu0_dC = np.zeros((ne,ne*3))
#         dnu0_dC[0,:] = L_block[0,:]
#         dnu0_dC[1,:] = L_block[3,:]

#         dnus_dC = np.zeros((ne,ne*3))
#         dnus_dC[0,:] = L_block[1,:]
#         dnus_dC[1,:] = L_block[4,:]


#         print(0.05 * dnu0_dC,'dnu0_dC')
#         print(dnu_dC,'dnu_dC')

#         dnu_dL = 0.05 * self.C_new 
#         dnu_dM = 0
#         dnu_dL_list = []
#         for i in range(ne*3):
#             dnu_dL_list.append(dnu_dL)

#         dnu_0_dL = 0.05 * self.C_new
#         print(dnu_0_dL.shape,'hello')
        

#         derivatives['_nu_0','L_block_diag_matrix'] = dnu0_dL
#         derivatives['_nu_s','L_block_diag_matrix'] = dnus_dL
#         derivatives['_nu_c','L_block_diag_matrix'] = dnuc_dL
#         derivatives['_nu_0','_tangential_inflow_velocity'] = np.matmul(np.matmul(0.05 * dnu0_dC,dC_ddT),ddT_dVt)
#         derivatives['_nu_0','_chord'] = np.matmul(np.matmul(0.05 * dnu0_dC,dC_ddT),ddT_dc)
#         derivatives['_nu_0','_re_pitt_peters'] = np.matmul(np.matmul(0.05 * dnu0_dC,dC_ddT),ddT_dre)
#         derivatives['_nu_0','_pitch'] = np.matmul(np.matmul(0.05 * dnu0_dC,dC_ddT),ddT_dalpha)
#         derivatives['_nu_0','_dr'] = np.matmul(np.matmul(0.05 * dnu0_dC,dC_ddT),ddT_dr)
#         derivatives['_nu_0','_radius'] = np.matmul(0.05 * dnu0_dC,dC_rad)
#         derivatives['_nu_0','_rotor_radius'] = np.matmul(0.05 * dnu0_dC,dC_dR)
#         derivatives['_nu_0','_nu_0'] = np.diag(self.nu[:,0])

#         derivatives['_nu_s','_tangential_inflow_velocity'] = np.matmul(np.matmul(0.05 * dnus_dC,dC_ddT),ddT_dVt)
#         derivatives['_nu_s','_chord'] = np.matmul(np.matmul(0.05 * dnus_dC,dC_ddT),ddT_dc)
#         derivatives['_nu_s','_re_pitt_peters'] = np.matmul(np.matmul(0.05 * dnus_dC,dC_ddT),ddT_dre)
#         derivatives['_nu_s','_pitch'] = np.matmul(np.matmul(0.05 * dnus_dC,dC_ddT),ddT_dalpha)
#         derivatives['_nu_s','_dr'] = np.matmul(np.matmul(0.05 * dnus_dC,dC_ddT),ddT_dr)
#         derivatives['_nu_s','_radius'] = np.matmul(0.05 * dnus_dC,dC_rad)
#         derivatives['_nu_s','_rotor_radius'] = np.matmul(0.05 * dnus_dC,dC_dR)
#         derivatives['_nu_s','_nu_0'] = np.diag(self.nu[:,1])
#         # print(derivatives['_nu_0','_nu_0'].shape,'HELP_ME_SHAPE')
#         print(dC_ddT,'dC_ddT')
    




#     def solve_residual_equations(self, inputs,outputs):
#         rotor = self.parameters['rotor']
#         shape = self.parameters['shape']
#         ne = shape[0]
#         nr = shape[1]
#         nt = shape[2]
#         interp = rotor['interp']
#         B = rotor['num_blades']
#         beta = rotor['rotor_disk_tilt_angle']

#         L_block = inputs['L_block_diag_matrix']
#         L_inv_block = np.linalg.inv(L_block)
#         M_block = inputs['M_block_diag_matrix']
#         M_inv_block = np.linalg.inv(M_block)

#         Re = inputs['_re_pitt_peters']
#         rho = rotor['density']
#         chord = inputs['_chord']
#         twist = inputs['_pitch']
#         Vt = inputs['_tangential_inflow_velocity']
#         v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
#         psi = np.einsum(
#             'ij,k->ijk',
#             np.ones((ne, nr)),
#             v,
#         )
#         angular_speed = rotor['Omega']
#         Omega = np.einsum(
#             'i,ijk->ijk',
#             angular_speed,
#             np.ones((ne, nr,nt)),         
#         )
        
#         dr = inputs['_dr']
#         R = inputs['_rotor_radius']
#         r = inputs['_radius']
#         normalized_radial_discretization = 1. / nr / 2. \
#         + np.linspace(0., 1. - 1. / nr, nr)
#         r_norm = np.einsum(
#             'ik,j->ijk',
#             np.ones((ne, nt)),
#             normalized_radial_discretization,
#         )

#         nu_0 = np.zeros((shape[0],))
#         nu_s = np.zeros((shape[0],))
#         nu_c = np.zeros((shape[0],))

#         self.nu = np.zeros((shape[0],3))
#         self.C = np.zeros((shape[0],3))
#         self.C_T = np.zeros((shape[0],))
#         self.C_L = np.zeros((shape[0],))
#         self.C_M = np.zeros((shape[0],))
#         # nu = np.zeros((shape[0],3,1))

#         self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6

#         for j in range(80):
#             nu_0_exp = np.einsum(
#                 'i,ijk->ijk',
#                 nu_0,
#                 np.ones((ne, nr,nt)),         
#                 )
#             nu_s_exp = np.einsum(
#                 'i,ijk->ijk',
#                 nu_s,
#                 np.ones((ne, nr,nt)),         
#                 )
#             nu_c_exp = np.einsum(
#                 'i,ijk->ijk',
#                 nu_c,
#                 np.ones((ne, nr,nt)),         
#                 )
            
#             # Compute nu and self.ux (ne, nr, nt)
#             self.nu_exp = nu_0_exp + nu_s_exp * r_norm * np.sin(psi) + nu_c_exp * r_norm * np.cos(psi)
#             self.ux = self.nu_exp * Omega * R
#             # Compute inflow angle self.phi (ne, nr, nt)
#             # ignore u_theta
#             self.phi = np.arctan(self.ux / Vt)

#             # Compute sectional AoA (ne, nr, nt)
#             alpha = twist - self.phi 

#             # Compute Cl, Cd  (ne, nr, nt)
#             self.airfoil_model_inputs[:,0] = alpha.flatten()
#             airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
#             Cl = airfoil_model_outputs[:,:,:,0]
#             Cd = airfoil_model_outputs[:,:,:,1]

#             self.dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (Cl * np.cos(self.phi) - Cd * np.sin(self.phi)) * dr
#             T = np.sum(np.sum(self.dT,axis=1),axis=1) / shape[2]
#             # print(T)
#             # self.dQ = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (Cl * np.sin(self.phi) + Cd * np.cos(self.phi)) * r * dr
#             # Q = np.sum(np.sum(self.dQ,axis=1),axis=1) / shape[2]
            
#             # Compute roll and pitch moments from thrust
#             dL_mom = r * np.sin(psi) * self.dT
#             L_mom  = np.sum(np.sum(dL_mom,axis=1),axis=1) / shape[2]
#             dM_mom = r * np.cos(psi) * self.dT
#             M_mom  = np.sum(np.sum(dM_mom,axis=1),axis=1) / shape[2]

#             for i in range(ne):
#                 self.C_T[i] = T[i] / rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
#                 self.C_L[i] = L_mom[i] / rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
#                 self.C_M[i] = M_mom[i] / rho/ (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5

#                 self.C[i,0] = self.C_T[i]
#                 self.C[i,1] = -self.C_L[i]
#                 self.C[i,2] = -self.C_M[i] 

#                 self.nu[i,0] = nu_0[i]
#                 self.nu[i,1] = nu_s[i]
#                 self.nu[i,2] = nu_c[i] 
            
#             # self.C_new = C.reshape((shape[0],3,1))
#             self.C_new = self.C.flatten()
#             # nu = nu.reshape((shape[0],3,1))
#             self.nu_new = self.nu.flatten()

#             self.term1 = 0.05 * np.matmul(L_block,M_block)
#             self.term2 = np.matmul(M_inv_block,self.C_new)
#             self.term3 = np.matmul(M_inv_block,L_inv_block)
#             self.term4 = np.matmul(self.term3,self.nu_new)
#             self.term5 = self.term2 - self.term4
#             self.term6 = np.matmul(self.term1,self.term5)
#             self.nu_new += self.term6
            
#             self.nu = self.nu_new.reshape((shape[0],3))
            
#             nu_0 = self.nu[:,0]
#             nu_s = self.nu[:,1]
#             nu_c = self.nu[:,2]

#         # outputs['_nu'] = self.nu
#         outputs['_nu_0'] = nu_0
#         outputs['_nu_s'] = nu_s
#         outputs['_nu_c'] = nu_c
#         # print(outputs['_nu'],'TEST')
        