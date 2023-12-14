from lsdo_rotor.core.pitt_peters.pitt_peters_rotor_parameters import  PittPetersRotorParameters
import numpy as np
import csdl
from scipy.linalg import block_diag

import sympy as sym 
from sympy import *




def get_pitt_peters_rotor_dictionary(airfoil,interp,normal_inflow,in_plane_inflow,n,rotor_radius,ne,nr,nt):
    # -------Angular speed--------- #
    angular_speed = n * 2 * np.pi 

    # -------Azimuth angle-------- #
    v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
    # v = np.linspace(0, np.pi * 2-1e-5, nt)
    theta = np.einsum(
        'ij,k->ijk',
        np.ones((ne, nr)),
        v,  
    )
    
    # -------non-dimensiona quantities ------- #
    mu_z = normal_inflow / angular_speed / csdl.expand(rotor_radius,(ne,))
    mu = in_plane_inflow / angular_speed / csdl.expand(rotor_radius,(ne,))

    # ------- M matrix for Pitt-Peters -------- #
    M_mat = np.zeros((ne,3,3))
    M_inv_mat = np.zeros((ne,3,3))
    # m = np.array([128/75/np.pi, -16/45/np.pi, -16/45/np.pi])
    # M  = np.diag(m)
    # M_inv = np.linalg.inv(M)
    M_block_list = []
    M_inv_block_list = []
    for i in range(ne):
        M_mat[i,0,0] = 128/75/np.pi
        M_mat[i,1,1] = -16/45/np.pi
        M_mat[i,2,2] = -16/45/np.pi
        M_inv_mat[i,:,:] = np.linalg.inv(M_mat[i,:,:]) 

        M_block_list.append(M_mat[i,:,:])
        M_inv_block_list.append(M_inv_mat[i,:,:])
    M_block = block_diag(*M_block_list)
    M_inv_block = block_diag(*M_inv_block_list)

    # -------symbolic derivative of L matrix w.r.t lambda for Pitt-Peters ------- #
    lamb_0 = sym.Symbol('lambda_0')
    lamb_c = sym.Symbol('lambda_c')
    lamb_s = sym.Symbol('lambda_s')

    lamb_i_vec =  ImmutableMatrix(np.array([lamb_0,lamb_c,lamb_s]).reshape(3,1))

    r_cos_psi_mean = sym.Symbol('r_cos_psi_mean')
    r_sin_psi_mean = sym.Symbol('r_sin_psi_mean')

    mu_z_sym = sym.Symbol('mu_z')
    mu_sym = sym.Symbol('mu')

    lamb_i_star = lamb_0 + lamb_c * r_cos_psi_mean + lamb_s * r_sin_psi_mean
    lamb = lamb_i_star + mu_z_sym
    V_eff = (mu_sym**2 + lamb * (lamb + lamb_i_star)) / (mu_sym**2 + lamb**2)**0.5
    Chi = atan(mu_sym/lamb)

    L =  1/V_eff *  ImmutableMatrix(np.array([[0.5, -15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5 , 0],
              [+15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5, 4 * cos(Chi) / (1+cos(Chi)), 0],
              [0,0,4/(1+cos(Chi))]]))

    dL_dlambda = sym.diff(L,lamb_i_vec)
    dL_dlambda_func = lambdify((lamb_0,lamb_c,lamb_s,r_cos_psi_mean,r_sin_psi_mean,mu_z_sym,mu_sym),dL_dlambda,'numpy')

    dL_dmu = sym.diff(L,mu_sym)
    dL_dmu_func = lambdify((lamb_0,lamb_c,lamb_s,r_cos_psi_mean,r_sin_psi_mean,mu_z_sym,mu_sym),dL_dmu,'numpy')
    dL_dmu_z = sym.diff(L,mu_z_sym)
    dL_dmu_z_func = lambdify((lamb_0,lamb_c,lamb_s,r_cos_psi_mean,r_sin_psi_mean,mu_z_sym,mu_sym),dL_dmu_z,'numpy')


    rotor = PittPetersRotorParameters(
        airfoil_name=airfoil,
        interp=interp,
        mu=mu,
        mu_z=mu_z,
        dL_dlambda_function=dL_dlambda_func,
        dL_dmu_function=dL_dmu_func,
        dL_dmu_z_function=dL_dmu_z_func,
        M_block_matrix=M_block,
        M_inv_block_matrix=M_inv_block,
        azimuth_angle=theta,
    )


    return rotor