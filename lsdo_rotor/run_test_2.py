import numpy as np
from csdl import Model
import csdl
from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.functions.get_rotor_dictionary import get_rotor_dictionary
from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
import openmdao.api as om
from scipy.linalg import block_diag
import time
import sympy as sym 
from sympy import *

class PittPetersCustomImplicitOperation(csdl.CustomImplicitOperation):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('rotor', types=RotorParameters)
    
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
        
        self.declare_derivatives('_lambda','_lambda',method='exact')
        
        self.airfoil_model_inputs = np.zeros((shape[0] * shape[1] * shape[2], 2))

    def evaluate_residuals(self,inputs,outputs,residuals):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        ne = shape[0]
        nr = shape[1]
        nt = shape[2]
        
        # -----objects and variable from rotor dictionary ----- #
        interp = rotor['interp']
        B = rotor['num_blades']
        rho = rotor['density']
        psi = rotor['azimuth_angle']
        M_block = rotor['M_block_matrix']
        M_inv_block = rotor['M_inv_block_matrix']
        mu = rotor['mu']
        mu_z = rotor['mu_z']
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
        
        # Compute nu and self.ux (ne, nr, nt)
        self.lamb_exp = self.lamb_0_exp + self.lamb_c_exp * self.r_norm * np.cos(psi) + self.lamb_s_exp * self.r_norm * np.sin(psi)
        self.ux = (self.lamb_exp + mu_z_exp) * Omega * R

        # Compute inflow angle self.phi (ne, nr, nt) (ignore u_theta) 
        self.phi = np.arctan(self.ux / Vt)

        # Compute sectional AoA (ne, nr, nt)
        alpha = twist - self.phi 

        # Apply airfoil surrogate model to compute Cl and Cd (ne, nr, nt)
        self.airfoil_model_inputs[:,0] = alpha.flatten()
        self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6
        self.airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
        self.Cl = self.airfoil_model_outputs[:,:,:,0]
        self.Cd = self.airfoil_model_outputs[:,:,:,1]

        self.dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
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
            self.C_T[i] = self.T[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)  #(Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
            self.C_My[i] = My[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])
            self.C_Mx[i] = Mx[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) # rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
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
        self.lamb += self.term6.reshape((ne,3))

        residuals['_lambda'] = self.term6    

    def compute_derivatives(self, inputs, outputs, derivatives):
        shape = self.parameters['shape']
        ne = shape[0]
        nr = shape[1]
        nt = shape[2]
        rotor = self.parameters['rotor']
        
        # -----objects and variable from rotor dictionary ----- #
        B = rotor['num_blades']
        rho = rotor['density'] 
        psi = rotor['azimuth_angle']   
        mu = rotor['mu']
        mu_z = rotor['mu_z']
        mu_z_exp = np.einsum(
            'i,ijk->ijk',
            mu_z,
            np.ones((ne, nr,nt)),  
        )
        interp = rotor['interp']
        dL_dlambda_func = rotor['dL_dlambda_function']

        
        # ------input variables ------- # 
        Re = inputs['_re_pitt_peters']
        chord = inputs['_chord']
        twist = inputs['_pitch']
        Vt = inputs['_tangential_inflow_velocity']
        dr = inputs['_dr']
        R = inputs['_rotor_radius']
        r = inputs['_radius']
        Omega = inputs['_angular_speed']



        a = np.ones((1,nt))
        b = np.ones((nr,1))

        #  Initializing derivative arrays 
        dC_ddT = np.zeros((ne*3,ne*nr*nt))
        dC_dr = np.zeros((ne*3,ne*nr*nt))
        dC_dpsi = np.zeros((ne*3,ne*nr*nt))
        dC_dR = np.zeros((ne*3,ne*nr*nt))
        dC_dOmega = np.zeros((ne*3,ne*nr*nt))
        dlambda_exp_dlambda = np.zeros((ne*3,ne*nr*nt))

        # Initializing empty L matrix
        L = np.zeros((ne,3,3))
        
        # Initializing empty lists (to later form block diagonal matrices)
        L_list = []
        ddT_ddr_list = []
        ddT_dc_list = []
        ddT_dpitch_list = []
        ddT_dre_list = []
        ddT_dlambda_exp_list = []
        ddT_dR_list = []
        ddT_dOmega_list = []
        ddT_dpsi_list = []
        ddT_dVt_list = []
        dR_dlamb_1_list = []

        # Compute derivatives of Cl,Cd w.r.t. AoA, Re
        dCl_dalpha = interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,0].reshape((ne,nr,nt))
        dCd_dalpha = interp.predict_derivatives(self.airfoil_model_inputs, 0)[:,1].reshape((ne,nr,nt))
        dCl_dre = (interp.predict_derivatives(self.airfoil_model_inputs, 1)[:,0] / 2e6).reshape((ne,nr,nt))
        dCd_dre = (interp.predict_derivatives(self.airfoil_model_inputs, 1)[:,1] / 2e6).reshape((ne,nr,nt))
        
        for i in range(ne):
            # mean of r cos(psi) amd r sin(psi) 
            r_cos_psi_mean = np.mean(r[i,:,:] * np.cos(psi[i,:,:]))
            r_sin_psi_mean = np.mean(r[i,:,:] * np.sin(psi[i,:,:]))
    
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


            dC_dCT = np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)
            dC_dCMy = (np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.cos(psi[i,:,:])
            dC_dCMx = (np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.sin(psi[i,:,:])
            dC_ddT[3*i,i*nr*nt:(i+1)*nr*nt] = dC_dCT.flatten()
            dC_ddT[3*i+1,i*nr*nt:(i+1)*nr*nt] = -dC_dCMy.flatten()
            dC_ddT[3*i+2,i*nr*nt:(i+1)*nr*nt] = dC_dCMx.flatten()
            
            dC_dr[3*i+1,i*nr*nt:(i+1)*nr*nt] = -((np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * np.cos(psi[i,:,:]) * self.dT[i,:,:] ).flatten()
            dC_dr[3*i+2,i*nr*nt:(i+1)*nr*nt] = ((np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * np.sin(psi[i,:,:]) * self.dT[i,:,:]).flatten()
            
            dC_dpsi[3*i+1,i*nr*nt:(i+1)*nr*nt] = ((np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.sin(psi[i,:,:]) * self.dT[i,:,:] ).flatten()
            dC_dpsi[3*i+2,i*nr*nt:(i+1)*nr*nt] = ((np.matmul(b,a) / (nt * rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0])) * r[i,:,:] * np.cos(psi[i,:,:]) * self.dT[i,:,:]).flatten()
            
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


            dlambda_exp_dlambda_0_exp = np.ones((nr,nt))
            dlambda_exp_dlambda_c_exp = self.r_norm[i,:,:] * np.cos(psi[i,:,:])
            dlambda_exp_dlambda_s_exp = self.r_norm[i,:,:] * np.sin(psi[i,:,:])
            dlambda_exp_dlambda[3*i,i*nr*nt:(i+1)*nr*nt] = dlambda_exp_dlambda_0_exp.flatten()
            dlambda_exp_dlambda[3*i+1,i*nr*nt:(i+1)*nr*nt] = dlambda_exp_dlambda_c_exp.flatten()
            dlambda_exp_dlambda[3*i+2,i*nr*nt:(i+1)*nr*nt] = dlambda_exp_dlambda_s_exp.flatten()

            
            ddT_dCl = 0.5 * B * rho * (self.ux[i,:,:]**2 + Vt[i,:,:]**2) * chord[i,:,:] * np.cos(self.phi[i,:,:]) * dr[i,:,:]
            ddT_dCd = -0.5 * B * rho * (self.ux[i,:,:]**2 + Vt[i,:,:]**2) * chord[i,:,:] * np.sin(self.phi[i,:,:]) * dr[i,:,:]
            dalpha_dpitch = 1
            dalpha_dphi = -1 
            
            
            
            dphi_dux = Vt[i,:,:] / (Vt[i,:,:]**2 + self.ux[i,:,:]**2)
            dcosphi_dux = -Vt[i,:,:] * self.ux[i,:,:] / (Vt[i,:,:]**2 + self.ux[i,:,:]**2)**1.5
            dsinphi_dux = Vt[i,:,:]**2 / (Vt[i,:,:]**2 + self.ux[i,:,:]**2)**1.5
            
            dux_dR = (self.lamb_exp[i,:,:] + mu_z_exp[i,:,:]) * Omega[i,:,:]
            dux_dlambda_exp = Omega[i,:,:] * R[i,:,:]
            dux_dOmega = (self.lamb_exp[i,:,:] + mu_z_exp[i,:,:]) * R[i,:,:]

            const = 0.5 * B * rho * chord[i,:,:] * dr[i,:,:]

            # --------- ddT_ddr ---------#
            ddT_ddr = np.diag((self.dT[i,:,:]/dr[i,:,:]).flatten())
            
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
            ddT_dc_list.append(ddT_dc)
            ddT_dpitch_list.append(ddT_dpitch)
            ddT_dre_list.append(ddT_dre)
            ddT_dlambda_exp_list.append(ddT_dlambda_exp)
            ddT_dR_list.append(ddT_dR)
            ddT_dOmega_list.append(ddT_dOmega)
            ddT_dpsi_list.append(ddT_dpsi)
            ddT_dVt_list.append(ddT_dVt)
            dR_dlamb_1_list.append(dlambda_dlambda)


        # Creating all block diagonal matrices
        L_block = block_diag(*L_list)
        ddT_ddr_block = block_diag(*ddT_ddr_list)
        ddT_dc_block = block_diag(*ddT_dc_list)
        ddT_dpitch_block = block_diag(*ddT_dpitch_list)
        ddT_dre_block = block_diag(*ddT_dre_list)
        ddT_dVt_block = block_diag(*ddT_dVt_list)
        ddT_dR_block = block_diag(*ddT_dR_list)
        ddT_dpsi_block = block_diag(*ddT_dpsi_list)
        ddT_dOmega_block = block_diag(*ddT_dOmega_list)
        dR_dlamb_1_block = block_diag(*dR_dlamb_1_list)

        # Chain rule for residual 
        dlambda_dC = 0.5 * L_block
        dlambda_ddT = np.matmul(dlambda_dC,dC_ddT)
        
        dlambda_ddr = np.matmul(dlambda_ddT,ddT_ddr_block)
        dlambda_dc = np.matmul(dlambda_ddT,ddT_dc_block)
        dlambda_dpitch = np.matmul(dlambda_ddT,ddT_dpitch_block)
        dlambda_dre = np.matmul(dlambda_ddT,ddT_dre_block)
        dlambda_dVt = np.matmul(dlambda_ddT,ddT_dVt_block)
        dlambda_dR = np.matmul(dlambda_ddT,ddT_dR_block)
        dlambda_dpsi = np.matmul(dlambda_ddT,ddT_dpsi_block)
        dlambda_dOmega = np.matmul(dlambda_ddT,ddT_dOmega_block)

        dlambda_dr = np.matmul(dlambda_dC,dC_dr)

       
        # Setting partials of residual w.r.t inputs
        derivatives['_lambda','_dr'] = dlambda_ddr
        derivatives['_lambda','_chord'] = dlambda_dc
        derivatives['_lambda','_pitch'] = dlambda_dpitch
        derivatives['_lambda','_re_pitt_peters'] = dlambda_dre
        derivatives['_lambda','_tangential_inflow_velocity'] = dlambda_dVt
        derivatives['_lambda','_radius'] = dlambda_dr
        derivatives['_lambda','_rotor_radius'] =   np.matmul(dlambda_dC,dC_dR) + dlambda_dR  
        # derivatives['_lambda','_psi'] =   np.matmul(dlambda_dC,dC_dpsi) + dlambda_dpsi  
        derivatives['_lambda','_angular_speed'] =   np.matmul(dlambda_dC,dC_dOmega) + dlambda_dOmega  
        print(np.matmul(dlambda_dC,dC_dpsi))
        print(dlambda_dpsi,'dlambda_dpsi')

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
        B = rotor['num_blades']
        M_block = rotor['M_block_matrix']
        M_inv_block = rotor['M_inv_block_matrix']
        psi = rotor['azimuth_angle']
        mu = rotor['mu']
        mu_z = rotor['mu_z']
        mu_z_exp = np.einsum(
            'i,ijk->ijk',
            mu_z,
            np.ones((ne, nr,nt)),  
        )

        dr = inputs['_dr']
        R = inputs['_rotor_radius']
        r = inputs['_radius']
        Re = inputs['_re_pitt_peters']
        rho = rotor['density']
        chord = inputs['_chord']
        twist = inputs['_pitch']
        Vt = inputs['_tangential_inflow_velocity']
        Omega = inputs['_angular_speed']
        
        normalized_radial_discretization = 1. / nr / 2. \
        + np.linspace(0., 1. - 1. / nr, nr)
        r_norm = np.einsum(
            'ik,j->ijk',
            np.ones((ne, nt)),
            normalized_radial_discretization,
        )

        lambda_0 = np.zeros((ne,))#np.random.randn(shape[0],)
        lambda_c = np.zeros((ne,))#np.random.randn(shape[0],)
        lambda_s = np.zeros((ne,))#np.random.randn(shape[0],)
        

        self.lamb = np.zeros((shape[0],3))
        self.C = np.zeros((shape[0],3))
        self.C_T = np.zeros((shape[0],))
        self.C_Mx = np.zeros((shape[0],))
        self.C_My = np.zeros((shape[0],))

        self.airfoil_model_inputs[:,1] = Re.flatten() / 2e6

        for j in range(400):
            print(j,'iteration')
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
            self.phi = np.arctan(self.ux / Vt)

            # Compute sectional AoA (ne, nr, nt)
            alpha = twist - self.phi 

            # Compute Cl, Cd  (ne, nr, nt)
            self.airfoil_model_inputs[:,0] = alpha.flatten()
            airfoil_model_outputs = interp.predict_values(self.airfoil_model_inputs).reshape((shape[0] , shape[1] , shape[2], 2))
            Cl = airfoil_model_outputs[:,:,:,0]
            Cd = airfoil_model_outputs[:,:,:,1]

            self.dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (Cl * np.cos(self.phi) - Cd * np.sin(self.phi)) * dr
            T = np.sum(np.sum(self.dT,axis=1),axis=1) / shape[2]
            print(T)
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
                self.C_T[i] = T[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2)  #(Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**4
                self.C_Mx[i] = Mx[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) # rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
                self.C_My[i] = My[i] / (rho * np.pi * R[i,0,0]**2 * (Omega[i,0,0] * R[i,0,0])**2 * R[i,0,0]) #  rho / (Omega[i,0,0] / 2 / np.pi)**2 / (R[i,0,0] * 2)**5
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
                print('Pitt-Peters ODE solved for steady state')
                break


            self.lamb = self.lamb_new.reshape((shape[0],3))
            
            lambda_0 = self.lamb[:,0]
            lambda_c = self.lamb[:,1]
            lambda_s = self.lamb[:,2]

        outputs['_lambda'] = self.lamb    

    def apply_inverse_jacobian( self, d_outputs, d_residuals, mode):
        shape = self.parameters['shape']
        ne = shape[0]
        if mode == 'fwd':
            d_outputs['_lambda'] = self.inv_jac * d_residuals['_lambda'].reshape((ne,3))
        elif mode == 'rev':
            d_residuals['_lambda'] = np.matmul(self.inv_jac, d_outputs['_lambda'].flatten()).reshape(ne,3)
            


Reynolds = np.array([[282783.04121835, 450174.34645058, 383749.44521553, 200677.16473479, 171871.30195311],
  [442818.77313996,622917.36109499, 553337.85253731, 337040.97226833, 276432.78165348],
  [615277.36518994, 799844.16490358, 729011.29910544, 503266.79043484,435483.60926184]])
Reynolds_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    Reynolds,
)

chord = np.array([[0.1, 0.1, 0.1, 0.1, 0.1],
  [0.1, 0.1, 0.1, 0.1, 0.1],
  [0.1, 0.1, 0.1, 0.1, 0.1]])
chord_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    chord,
)

theta = np.array([[0.6981317,  0.6981317,  0.6981317,  0.6981317,  0.6981317 ],
  [0.52359878, 0.52359878, 0.52359878, 0.52359878, 0.52359878],
  [0.34906585, 0.34906585, 0.34906585, 0.34906585, 0.34906585]])
theta_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    theta,
)

Vtheta = np.array([[34.90658504,  64.04864742,  52.91737009,  16.89579999,   5.76452266],
  [ 62.83185307 , 91.97391545,  80.84263812,  44.82106802 , 33.6897907 ],
  [ 90.7571211 , 119.89918348, 108.76790615,  72.74633605 , 61.61505873 ]])
Vtheta_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    Vtheta,
)

  
slice_thickness = np.array([[0.4, 0.4, 0.4, 0.4, 0.4],
  [0.4, 0.4, 0.4, 0.4, 0.4],
  [0.4, 0.4, 0.4, 0.4, 0.4]])
slice_thickness_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    slice_thickness,
)


Radius = np.array([[1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1.]])
Radius_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    Radius,
)

radius = np.array([[1/3,1/3,1/3,1/3,1/3],[0.6,0.6,0.6,0.6,0.6],[0.87,0.87,0.87,0.87,0.87]])
radius_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    radius,
)

azimuth_angle = np.array([[0,1.25663706, 2.51327412, 3.76991118, 5.02654825],
  [0, 1.25663706, 2.51327412, 3.76991118, 5.02654825],
  [0, 1.25663706, 2.51327412, 3.76991118, 5.02654825]])
azimuth_angle_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    azimuth_angle,
)

angular_speed = np.array([[104.71975512, 104.71975512, 104.71975512, 104.71975512, 104.71975512],
  [104.71975512, 104.71975512, 104.71975512, 104.71975512, 104.71975512],
  [104.71975512, 104.71975512, 104.71975512, 104.71975512, 104.71975512]])
angular_speed_2 = np.einsum(
    'ijk,jk->ijk',
    np.ones((2,3,5)),
    angular_speed,
)


class TestModel(Model):
    def define(self):
        Re = self.create_input('_re_pitt_peters', val=Reynolds_2)#.reshape(1,3,5))
        c = self.create_input('_chord', val=chord_2)#.reshape(1,3,5))
        pitch = self.create_input('_pitch', val=theta_2)#.reshape(1,3,5))
        self.add_design_variable('_pitch')
        self.add_design_variable('_chord')
        Vt = self.create_input('_tangential_inflow_velocity', val=Vtheta_2)#.reshape(1,3,5))
        R = self.create_input('_rotor_radius', val=Radius_2)#.reshape(1,3,5))
        dr = self.create_input('_dr', val=slice_thickness_2)#.reshape(1,3,5))
        r = self.create_input('_radius', val=radius_2)#.reshape(1,3,5))
        psi = self.create_input('_psi',val=azimuth_angle_2)#.reshape(1,3,5))
        Omega = self.create_input('_angular_speed', val=angular_speed_2)#.reshape(1,3,5))
        
        airfoil = 'NACA_4412_extended_range' 
        interp = get_surrogate_model(airfoil)
        i_vec = np.array([40,40]) 
        RPM_vec = np.array([1000,1000]) 
        V_inf_vec = np.array([40,40])   
        diameter_vec = np.array([2,2])

        num_evaluations = 2
        num_radial = 3
        num_tangential = 5
        
        shape = (num_evaluations,num_radial,num_tangential)
        rotor = get_rotor_dictionary(airfoil,5,1000,3,interp,i_vec,RPM_vec,V_inf_vec,diameter_vec,num_evaluations,num_radial,num_tangential)

        test = csdl.custom(
            Re,
            c,
            pitch,
            Vt,
            dr,
            R,
            r,
            # psi,
            Omega,
            op=PittPetersCustomImplicitOperation(
                shape=shape,
                rotor=rotor,
            ))
        
        self.register_output('_lambda', test)
        self.add_objective('_lambda')

from csdl_om import Simulator
sim = Simulator(TestModel())



startTime = time.time()
sim.run()
TestModel().visualize_sparsity(recursive=True)
exit()

sim.prob.check_partials(compact_print=True, step=1e-5, form='central')
sim.prob.check_totals(compact_print=True, step=1e-5, form='central')

stopTime = time.time()

execution_time = stopTime - startTime
print(execution_time)
# sim.prob.check_totals()


  # lamb_0_sym = sym.Symbol('lamb_0')
        # lamb_c_sym = sym.Symbol('lamb_c')
        # lamb_s_sym = sym.Symbol('lamb_s')

        # M = Matrix(np.array([[128/75/np.pi,0,0],[0,64/45/np.pi,0],[0,0,64/45/np.pi]]))
        # M_inv = M**-1

        # dr_sym = Matrix(symarray('dr', (nr,nt)))
        # c_sym = Matrix(symarray('c', (nr,nt)))
        # Vt_sym = ImmutableMatrix(symarray('Vt',(nr,nt)))
        # r_sym = ImmutableMatrix(symarray('r',(nr,nt)))
        # sym_list = [dr_sym,c_sym,r_sym]



 # for i in range(len(sym_list)):
            # if i == 0:
            #     dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * chord * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi))
            #     dT_sym = matrix_multiply_elementwise(Matrix(dT.reshape(nr,nt)), dr_sym)
            #     dL_sym = matrix_multiply_elementwise(dT_sym, Matrix((r * np.sin(psi)).reshape(nr,nt)))
            #     dM_sym = matrix_multiply_elementwise(dT_sym, Matrix((r * np.cos(psi)).reshape(nr,nt)))
            #     force_moment_mat = BlockDiagMatrix(dT_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2),-dL_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]),dM_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]))
            #     force_moment_mat = ImmutableMatrix(force_moment_mat)
            #     C =  ImmutableMatrix(A) * force_moment_mat *  ImmutableMatrix(b)
            #     lamb_i_vec =  ImmutableMatrix(np.array([self.lamb[0,0],self.lamb[0,1],self.lamb[0,2]]).reshape(3,1))
            #     lamb_i_star = self.lamb[0,0] + self.lamb[0,1] * rcos_mean + self.lamb[0,2] * rsin_mean
            #     lamb = lamb_i_star + mu_z
            #     V_eff = (mu**2 + lamb * (lamb + lamb_i_star)) / (mu**2 + lamb**2)**0.5
            #     Chi = atan(mu/lamb)
            #     L =  1/V_eff *  ImmutableMatrix(np.array([[0.5, -15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5 , 0],
            #         [+15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5, 4 * cos(Chi) / (1+cos(Chi)), 0],
            #         [0,0,4/(1+cos(Chi))]]))
            #     L_inv = L**-1
            #     R_sym = 0.5 * L * M * (M_inv * C - M_inv *  L_inv * lamb_i_vec)
            #     dR_ddr = sym.diff(R_sym,dr_sym)
            #     dR_ddr_array = np.array(dR_ddr).astype(np.float64).reshape(nr*nt,3).T
            
            #     derivatives['_lambda','_dr'] = dR_ddr_array
            # if i == 1:
            #     dT = 0.5 * B * rho * (self.ux**2 + Vt**2) * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
            #     dT_sym = matrix_multiply_elementwise(Matrix(dT.reshape(nr,nt)), c_sym)
            #     dL_sym = matrix_multiply_elementwise(dT_sym, Matrix((r * np.sin(psi)).reshape(nr,nt)))
            #     dM_sym = matrix_multiply_elementwise(dT_sym, Matrix((r * np.cos(psi)).reshape(nr,nt)))
            #     force_moment_mat = BlockDiagMatrix(dT_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2),-dL_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]),dM_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]))
            #     force_moment_mat = ImmutableMatrix(force_moment_mat)
            #     C =  ImmutableMatrix(A) * force_moment_mat *  ImmutableMatrix(b)
            #     lamb_i_vec =  ImmutableMatrix(np.array([self.lamb[0,0],self.lamb[0,1],self.lamb[0,2]]).reshape(3,1))
            #     lamb_i_star = self.lamb[0,0] + self.lamb[0,1] * rcos_mean + self.lamb[0,2] * rsin_mean
            #     lamb = lamb_i_star + mu_z
            #     V_eff = (mu**2 + lamb * (lamb + lamb_i_star)) / (mu**2 + lamb**2)**0.5
            #     Chi = atan(mu/lamb)
            #     L =  1/V_eff *  ImmutableMatrix(np.array([[0.5, -15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5 , 0],
            #         [+15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5, 4 * cos(Chi) / (1+cos(Chi)), 0],
            #         [0,0,4/(1+cos(Chi))]]))
            #     L_inv = L**-1
            #     R_sym = 0.5 * L * M * (M_inv * C - M_inv *  L_inv * lamb_i_vec)
            #     dR_dc = sym.diff(R_sym,c_sym)
            #     dR_dc_array = np.array(dR_dc).astype(np.float64).reshape(nr*nt,3).T
            
            #     derivatives['_lambda','_chord'] = dR_dc_array
            # elif i == 2:
            #     dT1 = ImmutableMatrix((0.5 * B * rho * (self.ux**2) * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr).reshape(nr,nt))
            #     dT2 = matrix_multiply_elementwise(Vt_sym.applyfunc(lambda e: e**2), ImmutableMatrix((0.5 * B * rho * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr).reshape(nr,nt)))
            #     dT_sym = dT1 + dT2
            #     dL_sym = matrix_multiply_elementwise(dT_sym, ImmutableMatrix((r * np.sin(psi)).reshape(nr,nt)))
            #     dM_sym = matrix_multiply_elementwise(dT_sym, ImmutableMatrix((r * np.cos(psi)).reshape(nr,nt)))
            #     force_moment_mat = BlockDiagMatrix(dT_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2),-dL_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]),dM_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]))
            #     force_moment_mat = ImmutableMatrix(force_moment_mat)
            #     C =  ImmutableMatrix(A) * force_moment_mat *  ImmutableMatrix(b)
            #     lamb_i_vec =  ImmutableMatrix(np.array([self.lamb[0,0],self.lamb[0,1],self.lamb[0,2]]).reshape(3,1))
            #     lamb_i_star = self.lamb[0,0] + self.lamb[0,1] * rcos_mean + self.lamb[0,2] * rsin_mean
            #     lamb = lamb_i_star + mu_z
            #     V_eff = (mu**2 + lamb * (lamb + lamb_i_star)) / (mu**2 + lamb**2)**0.5
            #     Chi = atan(mu/lamb)
            #     L =  1/V_eff *  ImmutableMatrix(np.array([[0.5, -15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5 , 0],
            #         [+15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5, 4 * cos(Chi) / (1+cos(Chi)), 0],
            #         [0,0,4/(1+cos(Chi))]]))
            #     L_inv = L**-1
            #     R_sym = 0.5 * L * M * (M_inv * C - M_inv *  L_inv * lamb_i_vec)
            #     dR_dVt = sym.diff(R_sym,Vt_sym)
            #     e1 = lambdify(Vt_sym,dR_dVt)                
            #     evaluated_expression = e1(*Vt.T.flatten())
            #     print(e1(*Vt.flatten()),'evaluated expression')
            #     dR_dVt_array = np.array(evaluated_expression).astype(np.float64).reshape(nr*nt,3).T
            
            #     derivatives['_lambda','_tangential_inflow_velocity'] =
            #     dR_dVt_array
            # if i == 2:
            #     dT = 0.5 * B * rho * chord * (self.ux**2 + Vt**2) * (self.Cl * np.cos(self.phi) - self.Cd * np.sin(self.phi)) * dr
            #     dT_sym = ImmutableMatrix(dT.reshape(nr,nt))
            #     dL_sym = matrix_multiply_elementwise(dT_sym, ImmutableMatrix((r * np.sin(psi)).reshape(nr,nt)))
            #     dM_sym = matrix_multiply_elementwise(dT_sym, ImmutableMatrix((r * np.cos(psi)).reshape(nr,nt)))
            #     force_moment_mat = BlockDiagMatrix(dT_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2),-dL_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]),dM_sym/ (nt * rho * np.pi * R[0,0,0]**2 * (Omega[0,0,0] * R[0,0,0])**2 * R[0,0,0]))
            #     force_moment_mat = ImmutableMatrix(force_moment_mat)
            #     C =  ImmutableMatrix(A) * force_moment_mat *  ImmutableMatrix(b)
            #     lamb_i_vec =  ImmutableMatrix(np.array([self.lamb[0,0],self.lamb[0,1],self.lamb[0,2]]).reshape(3,1))
            #     lamb_i_star = self.lamb[0,0] + self.lamb[0,1] * rcos_mean + self.lamb[0,2] * rsin_mean
            #     lamb = lamb_i_star + mu_z
            #     V_eff = (mu**2 + lamb * (lamb + lamb_i_star)) / (mu**2 + lamb**2)**0.5
            #     Chi = atan(mu/lamb)
            #     L =  1/V_eff *  ImmutableMatrix(np.array([[0.5, -15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5 , 0],
            #         [+15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5, 4 * cos(Chi) / (1+cos(Chi)), 0],
            #         [0,0,4/(1+cos(Chi))]]))
            #     L_inv = L**-1
            #     R_sym = 0.5 * L * M * (M_inv * C - M_inv *  L_inv * lamb_i_vec)
            #     dR_dVt = sym.diff(R_sym,Vt_sym)
            #     e1 = lambdify(Vt_sym,dR_dVt)                
            #     evaluated_expression = e1(*Vt.T.flatten())
            #     print(e1(*Vt.flatten()),'evaluated expression')
            #     dR_dVt_array = np.array(evaluated_expression).astype(np.float64).reshape(nr*nt,3).T
            
            #     derivatives['_lambda','_tangential_inflow_velocity'] = dR_dVt_array
