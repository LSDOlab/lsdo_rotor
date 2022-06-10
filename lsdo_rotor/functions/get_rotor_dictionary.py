from lsdo_rotor.rotor_parameters import RotorParameters
import numpy as np
from scipy.linalg import block_diag

import sympy as sym 
from sympy import *

def get_rotor_dictionary(airfoil_name, blades, altitude, mode, interp, Vx_vec,RPM_vec, Vy_vec, diameter_vec, num_evaluations, num_radial, num_tangential):#, ideal_alpha_ref_chord, ideal_Cl_ref_chord, ideal_Cd_ref_chord, c_ref, beta):
    rotor_radius = diameter_vec/2
    ne = num_evaluations
    nr = num_radial
    nt = num_tangential

    # -------Atmosphere--------- #
    h    = altitude * 1e-3
    # Constants
    L           = 6.5
    R           = 287
    T0          = 288.16
    P0          = 101325
    g0          = 9.81
    mu0         = 1.735e-5
    S1          = 110.4
    # Temperature 
    T           = T0 - L * h

    # Pressure 
    P           = P0 * (T/T0)**(g0/(L * 1e-3)/R)
    # Density
    rho         = P/R/T

    # -------Angular speed--------- #
    angular_speed = RPM_vec * 2 * np.pi / 60 

    # -------Azimuth angle-------- #
    v = np.linspace(0, np.pi * 2 - np.pi * 2 / nt, nt)
    theta = np.einsum(
        'ij,k->ijk',
        np.ones((ne, nr)),
        v,  
    )

    # -------non-dimensiona quantities ------- #
    mu_z = Vx_vec / angular_speed / rotor_radius
    mu = Vy_vec / angular_speed / rotor_radius
    
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

    lamb_i_star = lamb_0 + lamb_c * r_sin_psi_mean + lamb_s * r_cos_psi_mean
    lamb = lamb_i_star + mu_z_sym
    V_eff = (mu_sym**2 + lamb * (lamb + lamb_i_star)) / (mu_sym**2 + lamb**2)**0.5
    Chi = atan(mu_sym/lamb)

    L =  1/V_eff *  ImmutableMatrix(np.array([[0.5, -15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5 , 0],
              [+15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5, 4 * cos(Chi) / (1+cos(Chi)), 0],
              [0,0,4/(1+cos(Chi))]]))

    dL_dlambda = sym.diff(L,lamb_i_vec)
    dL_dlambda_func = lambdify((lamb_0,lamb_c,lamb_s,r_cos_psi_mean,r_sin_psi_mean,mu_z_sym,mu_sym),dL_dlambda,'numpy')

    rotor=RotorParameters(
        airfoil_name=airfoil_name,
        num_blades=blades,
        altitude=altitude,
        mode=mode,
        interp=interp,
        density=rho,
        mu=mu,
        mu_z=mu_z,
        dL_dlambda_function=dL_dlambda_func,
        M_block_matrix=M_block,
        M_inv_block_matrix=M_inv_block,
        azimuth_angle=theta,
    )

    return rotor