from csdl_om import Simulator
import numpy as np
from csdl import Model
import csdl
num_tangential = 5

import time


# nu_0 = np.random.randn(2,)
# # print(nu_0)

# exp_nu0 = np.einsum(
#     'i,ijk->ijk',
#     nu_0,
#     np.ones((2, 40,40)),
    
# )
# print(exp_nu0[0,:,:])
# print(exp_nu0[1,:,:])
# # print(np.broadcast_to(nu_0,(2,4,4)))
# exit()

# t1 = np.array([[1,2,3],
#                [4,5,6],
#                ]
#             )
# print(np.diff(t1))
# # exit()
# # t1 = np.broadcast_to(t1,(4,3,3))

# from scipy.linalg import block_diag
# t2 = np.array([1,2,3])
# t2 = np.broadcast_to(t2.reshape((3,1)),(4,3,1))
# print(t2.shape)

# # print(t1)
# block_list = [t1,t1,t1]
# block_diagonal_mat = block_diag(*block_list)
# print(block_diagonal_mat)
# # print(t2)
# exit()

# t3 = np.array([[1,1,1],
#               [2,2,2],
#               [3,3,3]])
# t3 = np.broadcast_to(t3,(4,3,3))

# print(t1[0,:,:])
# print(t3[0,:,:])

# einsum_test1 = np.einsum(
#     'ijk,ikl->ijl',
#     t1,
#     t3
# )

# print(einsum_test1)

# einsum_test2 = np.einsum(
#     'ijk,ikl->ijl',
#     t1,
#     t2
# )
# # print(einsum_test2)
# # print(np.matmul(t1[0,:,:], t2[0,:,:]))
# # print(einsum_test1)
# print(np.matmul(t1[0,:,:], t3[0,:,:]))


# class ExampleInnerTensorVector(Model):

#     def define(self):
#         ten1 = self.declare_variable('ten1', val = t1)
#         ten2 = self.declare_variable('ten2', val = t2)
#         ten3 = self.declare_variable('ten3', val = t3)

#         # ten1_ten2 = csdl.inner(ten1,ten2, axes = ([0],[1]))
#         ten1_ten2 = csdl.einsum(ten1,ten2, subscripts = 'ijk,ikl->ijl')
#         self.register_output('tensor_product_test_1',ten1_ten2)
#         ten1_ten3 = csdl.einsum(ten1,ten3, subscripts = 'ijk,ikl->ijl')
#         self.register_output('tensor_product_test_2',ten1_ten3)

# sim = Simulator(ExampleInnerTensorVector())
# sim.run()

# # print('tensor_product_test_1', sim['tensor_product_test_1'])
# # print('tensor_product_test_1_shape', sim['tensor_product_test_1'].shape)

# print('tensor_product_test_2', sim['tensor_product_test_2'])
# print('tensor_product_test_2_shape', sim['tensor_product_test_2'].shape)
# # print(sim['a'])
# # print('c', sim['c'].shape)
# # print(sim['c'])
# # print('einsum_inner2', sim['einsum_inner2'].shape)
# # print(sim['einsum_inner2'])


# print('\n')
# print(t1[0,:,:])
# print(t2[0,:,:])
# exit()
# print('\n')
# print(np.matmul(t1[0,:,:], t2[0,:,:]))
# print(t2)


# class TestCustomImplicitOperation(csdl.CustomImplicitOperation):
#     def define(self):
#         self.add_input('a')
#         self.add_input('b')
#         self.add_input('c')
#         self.add_output('x')

#         self.declare_derivatives('x','a')
#         self.declare_derivatives('x','b')
#         self.declare_derivatives('x','c')
#         self.declare_derivatives('x','x')

#     def evaluate_residuals(self, inputs, outputs, residuals):
#         x = outputs['x']
#         a = inputs['a']
#         b = inputs['b']
#         c = inputs['c']
#         residuals['x'] = a * x**2 + b * x + c
#         print(residuals['x'],'evaluate_residuals')

    

#     def compute_derivatives(self, inputs, outputs, derivatives):
#         a = inputs['a']
#         b = inputs['b']
#         self.x = outputs['x']

#         derivatives['x', 'x'] = 2 * a * self.x + b
#         derivatives['x', 'a'] = self.x**2
#         derivatives['x', 'b'] = self.x
#         derivatives['x', 'c'] = 1.0

#         print(outputs['x'],'compute_derivatives')

#         self.inv_jac = 1.0 / (2 * a * self.x + b)
        

#     def solve_residual_equations(self, inputs, outputs):
#         a = inputs['a']
#         b = inputs['b']
#         c = inputs['c']
#         outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
#         print(outputs['x'],'solve_resdiual')

#     def apply_inverse_jacobian( self, d_outputs, d_residuals, mode):
#         if mode == 'fwd':
#             d_outputs['x'] = self.inv_jac * d_residuals['x']
#         elif mode == 'rev':
#             d_residuals['x'] = self.inv_jac * d_outputs['x']


# # model = Model()

# class TestModel(Model):
#     def define(self):
#         a = self.create_input('a', val=0.1)
#         b = self.create_input('b', val=4.3)
#         c = self.create_input('c', val=3)
#         self.add_design_variable('c')
#         self.add_design_variable('b')
    
#         test = csdl.custom(
#             a,
#             b,
#             c,
#             op=TestCustomImplicitOperation())
        
#         self.register_output('x', test)
#         self.register_output('y', b * test)
#         self.add_objective('y')

# # model.add(test,'test')

# sim = Simulator(TestModel())

# sim.run()
# print('\n')

# # sim.prob.check_partials(compact_print=True)
# sim.prob.check_totals()


# coeff = [1, 0.05, 0.01, 0, -0.0006]
# roots = np.roots(coeff)

# # for i in range(4):
# real_roots = roots[np.isreal(roots)]

# positive_roots = real_roots[real_roots>0]
# print(positive_roots)
# print(roots)

import sympy as sym 
from sympy import *

nt = 5
nr = 3
ne = 1

r = np.array([[1/3,1/3,1/3,1/3,1/3],[0.6,0.6,0.6,0.6,0.6],[0.87,0.87,0.87,0.87,0.87]])
psi = np.array([[0,1.25663706, 2.51327412, 3.76991118, 5.02654825],
  [0, 1.25663706, 2.51327412, 3.76991118, 5.02654825],
  [0, 1.25663706, 2.51327412, 3.76991118, 5.02654825]])
Vt = np.array([[34.90658504, 34.90658504, 34.90658504, 34.90658504, 34.90658504],
  [62.83185307, 62.83185307, 62.83185307, 62.83185307, 62.83185307],
  [90.7571211,  90.7571211,  90.7571211,  90.7571211,  90.7571211 ]])
Omega = np.array([[104.71975512, 104.71975512, 104.71975512, 104.71975512, 104.71975512],
  [104.71975512, 104.71975512, 104.71975512, 104.71975512, 104.71975512],
  [104.71975512, 104.71975512, 104.71975512, 104.71975512, 104.71975512]])
Radius = np.array([[1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1.]])
phi = np.array([[0.85720046, 0.85720046, 0.85720046, 0.85720046, 0.85720046],
  [0.57049484, 0.57049484, 0.57049484, 0.57049484, 0.57049484],
  [0.41804603, 0.41804603, 0.41804603, 0.41804603, 0.41804603]])
Cl = np.array([[-0.4237006,  -0.4237006,  -0.4237006,  -0.4237006 , -0.4237006 ],
  [ 0.20481035,  0.20481035,  0.20481035,  0.20481035,  0.20481035],
  [ 0.08454137 , 0.08454137,  0.08454137,  0.08454137,  0.08454137]])
Cd = np.array([[0.03070149, 0.03070149, 0.03070149, 0.03070149, 0.03070149],
  [0.00820729, 0.00820729 ,0.00820729 ,0.00820729, 0.00820729],
  [0.0083878 , 0.0083878,  0.0083878,  0.0083878,  0.0083878 ]])

mu_z_exp = np.array([[0.38197186, 0.38197186, 0.38197186, 0.38197186, 0.38197186],
  [0.38197186, 0.38197186, 0.38197186, 0.38197186, 0.38197186],
  [0.38197186, 0.38197186, 0.38197186, 0.38197186, 0.38197186]])

chord = np.array([[0.1, 0.1, 0.1, 0.1, 0.1],
  [0.1, 0.1, 0.1, 0.1, 0.1],
  [0.1, 0.1, 0.1, 0.1, 0.1]])

# dr = Matrix(np.array([[0.4, 0.4, 0.4, 0.4, 0.4],
#   [0.4, 0.4, 0.4, 0.4, 0.4],
#   [0.4, 0.4, 0.4, 0.4, 0.4]]))

dr = Matrix(symarray('dr', (3, 5)))


C_T = sym.Symbol('C_T')
C_L = sym.Symbol('C_L')
C_M = sym.Symbol('C_M')

lamb_0 = sym.Symbol('lambda_0')
lamb_c = sym.Symbol('lambda_c')
lamb_s = sym.Symbol('lambda_s')

lambda_i = lamb_0 + lamb_c * r * np.cos(psi) * r * np.sin(psi)
ux = (lambda_i + mu_z_exp) * Omega * Radius

B = 5
rho = 1.1

dT = matrix_multiply_elementwise(Matrix(0.5 * B * rho * (ux**2 + Vt**2) * chord * (Cl * np.cos(phi) - Cd * np.sin(phi))), dr)
dL = matrix_multiply_elementwise(dT ,Matrix(r * np.sin(psi)))
dM = matrix_multiply_elementwise(dT , Matrix(r * np.cos(psi)))

from scipy.linalg import block_diag
a = np.ones((1,3))
b = np.ones((15,1))
A = block_diag(*[a,a,a])

# force_moment_mat = block_diag(*[dT/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2),-dL/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0]),dM/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0])])
# force_moment_mat_2 = block_diag(*[dT/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2)/ dT,-dL/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0]) / dT, dM/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0])/dT])

force_moment_mat = BlockDiagMatrix(dT/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2),-dL/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0]),dM/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0]))
force_moment_mat = ImmutableMatrix(force_moment_mat)

# force_moment_mat_2 = BlockDiagMatrix(dT/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2)/dT,-dL/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0])/dT,dM/ (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0])/dT)
# const_vec = np.array([[1 / (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2)],
#                      [-1 / (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0])],
#                      [1 / (nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2 * Radius[0,0])]])


# C = Matrix(np.matmul(np.matmul(A,force_moment_mat),b))
C =  ImmutableMatrix(A) * force_moment_mat *  ImmutableMatrix(b)


lamb_i_vec =  ImmutableMatrix(np.array([lamb_0,lamb_c,lamb_s]).reshape(3,1))

rcos_mean = sym.Symbol('r_1')# np.mean(r * np.cos(psi))
rsin_mean = sym.Symbol('r_2')#np.mean(r * np.sin(psi))

lamb_i_star = lamb_0 + lamb_c * rcos_mean + lamb_s * rsin_mean

mu_z = sym.Symbol('mu_z')#0.38197186
mu = sym.Symbol('mu')#0

lamb = lamb_i_star + mu_z
V_eff = (mu**2 + lamb * (lamb + lamb_i_star)) / (mu**2 + lamb**2)**0.5
Chi = atan(mu/lamb)


L =  1/V_eff *  ImmutableMatrix(np.array([[0.5, -15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5 , 0],
              [+15 * np.pi / 64 * ((1-cos(Chi))/(1+cos(Chi)))**0.5, 4 * cos(Chi) / (1+cos(Chi)), 0],
              [0,0,4/(1+cos(Chi))]]))
print(L.subs([(lamb_0,0.00302828),(lamb_c,0),(lamb_s,0)]))

dL_dlambda = sym.diff(L,lamb_i_vec)


dL_dlambda_func = lambdify((lamb_0,lamb_c,lamb_s,rcos_mean,rsin_mean,mu_z,mu),dL_dlambda,'numpy')
lambdify_start = time.time()
test_L_derivative = np.nan_to_num(np.array([dL_dlambda_func(0.00302828,0,0,np.mean(r * np.cos(psi)),np.mean(r * np.sin(psi)),0.38197186,0)]), copy=False, nan=0.0)
lambdify_stop = time.time()
lambdify_time = lambdify_stop - lambdify_start
print(test_L_derivative.reshape(3,3,3),'lambdify test')
print(lambdify_time,'total lambdify time')
exit()
# pprint(dL_dlambda, use_unicode=True)

dL_dlambda =  dL_dlambda.reshape(3,3,3)
dL_dlambda = dL_dlambda.subs([(lamb_0,0.00302828),(lamb_c,0),(lamb_s,0)])
# print(dL_dlambda)
dL_dlambda = np.array(dL_dlambda).astype(np.float64).reshape(3,9)
print(dL_dlambda)

C_new = np.array([2.35012401e-03, 0,0]).reshape(1,3)
print(C_new)

print(0.5 * np.matmul(C_new,dL_dlambda).reshape(3,3))
# print(C.T.shape)
# C_dL_dlambda =  ImmutableMatrix(np.ones((1,3))) * dL_dlambda
# print(C_dL_dlambda.shape)
exit()

M = Matrix(np.array([[128/75/np.pi,0,0],[0,64/45/np.pi,0],[0,0,64/45/np.pi]]))
L_inv = L**-1
M_inv = M**-1
R = 0.5 * L * M * (M_inv * C - M_inv *  L_inv * lamb_i_vec)
# print(R.subs([(lamb_0,0.00302828),(lamb_c,-2.50071277e-19),(lamb_s,-2.50071277e-19)]))
dR_dlambda = Matrix(sym.diff(R,lamb_i_vec).reshape(3,3))
dR_dlambda_subs = dR_dlambda.subs([(lamb_0,0.00302828),(lamb_c,-2.50071277e-19),(lamb_s,-2.50071277e-19)])
# print(dR_dlambda_subs)
# dR_dlambda_array = np.array(dR_dlambda_subs).astype(np.float64).reshape(3,3).T

dR_ddr = sym.diff(R,dr)
dR_ddr_subs = dR_ddr.subs([(lamb_0,0.00302828),(lamb_c,-2.50071277e-19),(lamb_s,-2.50071277e-19)])
dR_ddr_array = np.array(dR_ddr_subs).astype(np.float64).reshape(3*5,3).T
print(dR_ddr_array)
exit()


dT_dr = (dT / dr).flatten()
# print(np.matmul(L,A).shape)
print(np.array(Matrix(force_moment_mat_2).subs([(lamb_0,0.00302828),(lamb_c,-2.50071277e-19),(lamb_s,-2.50071277e-19)])).astype(np.float64),'Block dT')
print(np.array(Matrix(dT).subs([(lamb_0,0.00302828),(lamb_c,-2.50071277e-19),(lamb_s,-2.50071277e-19)])).astype(np.float64),'dT')
# print((nt * rho * np.pi * Radius[0,0]**2 * (Omega[0,0] * Radius[0,0])**2),'constant')
print(Matrix(block_diag(*[dT,dT,dT])).subs([(lamb_0,0.00302828),(lamb_c,-2.50071277e-19),(lamb_s,-2.50071277e-19)]))
dR_ddr = Matrix(np.matmul(np.matmul(np.matmul(np.matmul(0.5 * L,A),force_moment_mat_2),b),dT_dr.reshape(1,15)))
dR_ddr_subs = dR_ddr.subs([(lamb_0,0.00302828),(lamb_c,-2.50071277e-19),(lamb_s,-2.50071277e-19)])
dR_ddr_array = np.array(dR_ddr_subs).astype(np.float64)
print(dR_ddr_array)
# exit()



# print(derivative_array)




