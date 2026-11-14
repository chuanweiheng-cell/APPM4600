"""
 This script provides you the opprotunity play with different linear algebra solvers.
You will learn about the performance of the different solvers and how to build
test some test matrices to challenge the different methods. 
"""

############################################# 
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
############################################# 


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from scipy.linalg import lu, lu_factor, lu_solve

import time




def driver():

    '''Compare numerical methods for solving linear systems.'''

    for N in [100, 500, 1000, 2000, 4000, 5000]:

        M = N
        A = create_rect(N, M)
        b = np.random.rand(N, 1)

        # Standard SciPy solver
        start_time = time.perf_counter()
        x = scila.solve(A, b)
        t_scipy = time.perf_counter() - start_time

        print(f'scila.solve run time for N={N}: {t_scipy:.4f} seconds')

        # LU factorization
        start_time = time.perf_counter()
        LU, P = lu_factor(A)
        t_factor = time.perf_counter() - start_time

        # LU triangular solves
        start_time = time.perf_counter()
        x_lu = lu_solve((LU, P), b)
        t_solve = time.perf_counter() - start_time

        print(f'LU factorization time for N={N}: {t_factor:.4f} seconds')
        print(f'LU solve time for N={N}: {t_solve:.4f} seconds')
        print(f'LU total time for N={N}: {t_factor + t_solve:.4f} seconds')

        # Break-even number of right-hand sides
        if t_scipy > t_solve:
            n_rhs = int(np.floor(t_factor / (t_scipy - t_solve))) + 1
            print(f'LU becomes faster after {n_rhs} right-hand sides')
        else:
            print('No break-even point from measured timings')

        print()

    # Normal equations and QR factorization

    N = 100
    M = 50

    # Compare original and reduced smallest singular values
    for d_min in [1e-10, 1e-12]:

        A = create_rect(N, M, d_min=d_min)

        # Construct a system with a known exact solution
        x_true = np.ones((M, 1))
        b = A @ x_true

        print(f'\nSmallest singular value: {d_min:.1e}')
        print(f'Condition number: {0.1 / d_min:.2e}')

        # Normal equations
        start_time = time.perf_counter()

        try:
            x_normal = scila.solve(A.T @ A, A.T @ b)

            elapsed_time = time.perf_counter() - start_time

            residual_normal = la.norm(A @ x_normal - b) / la.norm(b)
            error_normal = la.norm(x_normal - x_true) / la.norm(x_true)

            print(f'Normal equations runtime: {elapsed_time:.4f} s')
            print(f'Normal equations residual: {residual_normal:.4e}')
            print(f'Normal equations error: {error_normal:.4e}')

        except la.LinAlgError:
            print('Normal equations failed: singular matrix.')

        # QR factorization
        start_time = time.perf_counter()

        Q, R = la.qr(A, mode='reduced')
        x_qr = scila.solve_triangular(R, Q.T @ b)

        elapsed_time = time.perf_counter() - start_time

        residual_qr = la.norm(A @ x_qr - b) / la.norm(b)
        error_qr = la.norm(x_qr - x_true) / la.norm(x_true)

        print(f'QR factorization runtime: {elapsed_time:.4f} s')
        print(f'QR factorization residual: {residual_qr:.4e}')
        print(f'QR factorization error: {error_qr:.4e}')
        print()
     
def create_rect(N, M, d_min=1e-10):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     d[-1] = d_min
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
