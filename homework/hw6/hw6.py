#%% imports

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

# appm4600 project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})



#%% q1

print('Question 1:')

# test
def jac(x, y):
    return jnp.array([
        [  6*x   ,   8*y ],
        [-24*x**2, 3*y**2]
    ])
    
print(f'\nThe matrix in the question: \n{jnp.linalg.inv(jac(-0.5, 0.25))}\nis the inverse Jacobian matrix evaluated at (x0, y0)=(-0.5, 0.25).')

# part a, fixed point
def fixed_point_iteration(F, p_guess, tol=1e-10, max_iter=1000):
 
    count = 0
    
    f, g = F

    # Convert initial guess to a JAX array
    p_guess = jnp.array(p_guess, dtype=float)

    # Store initial guess
    history = [p_guess]
    
    J = jnp.array([
        [0.016, -0.17],
        [0.52 , -0.26]
    ])

    while count < max_iter:

        x_guess, y_guess = p_guess
        
        vec_new = jnp.array([x_guess, y_guess]) - J @ jnp.array([f(x_guess, y_guess), g(x_guess, y_guess)])

        x_new = vec_new[0]
        y_new = vec_new[1]
        
        p_new = jnp.array([x_new, y_new])

        # Error estimate between successive approximations
        err = jnp.linalg.norm(p_new - p_guess, ord=2)

        # Store new approximation
        history.append(p_new)

        count += 1

        # Check convergence
        if err < tol:

            root = p_new
            break

        # Update approximation for next iteration
        p_guess = p_new

    else:

        raise RuntimeError(
            f'Fixed-point iteration did not converge within {max_iter} iterations.'
        )

    history = jnp.array(history)

    return root, count, history


f = lambda x, y: 3*x**2 + 4*y**2 - 1
g = lambda x, y: y**3 - 8*x**3 - 1

F = (f, g)

tol = 1e-4
p_guess = (-0.5, 0.25)

root, count, history = fixed_point_iteration(F, p_guess, tol)

print(f'\nInitial guess (x0, y0) = {p_guess}')
print(f'Root from fixed-point iteration = {root}')
print(f'Iterations with tol={tol}: {count}')

#Convergence plot

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))

plt.semilogy(iterations, errors[:-1], '-o')

plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^*\|_2$')
plt.title(f'Fixed-Point Iteration Convergence for ($x_0$, $y_0$)={p_guess}')

plt.grid(True, which='both', alpha=0.3)

plt.show()

# part b, Newtons
from numerical_methods.higher_dimensional_methods.nD_newton_method import newtons_method

def F(p):
    x, y = p
    return jnp.array([
        3.0*x**2 + 4.0*y**2 - 1.0,
        y**3 - 8.0*x**3 - 1.0
    ])


p_guess = (-3.0, 3.0)

root, count, history = newtons_method(
    F,
    p_guess,
    tol=tol,
)

print(f'Root = {root}')
print(f'Iterations with tol={tol}: {count}')

# convergence plot
# Error relative to the converged root
errors = jnp.linalg.norm(
    history - root,
    axis=1
)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(
    len(history) - 1
)

plt.figure(figsize=(8, 5))

plt.semilogy(
    iterations,
    errors[:-1],
    '-o'
)

plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^{\,*}\|_2$')
plt.title(f"Newton's Method Convergence for ($x_0$, $y_0$)={p_guess}")

plt.grid(
    True,
    which='both',
    alpha=0.3
)

plt.show()

## Plotting

# f1 = lambda x: ((1 - 3*x**2) / 4)**0.5
# f2 = lambda x: (1 + 8*x**3)**(1/3)

# x_range = jnp.linspace(-13, 13, 1000000)
# x_range_2 = jnp.linspace(-300, 0.5, 100000)

# # plt.plot(x_range, f1(x_range), 'b')
# # plt.plot(x_range, -f1(x_range), 'b')
# plt.plot(x_range_2, f2(x_range_2))
# plt.axis('equal')

# plt.show()

#%% q2

print('question 2:')
print()

"""
 This compares Newton's method, Broyden and Lazy Newton for 
 computing the roots of vector valued functions.
 The function and the Jacobian are stored in subroutines and need 
 to be changed for different problems.  
 
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


import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    x0 = np.array([0.0, 0.0])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] = Broyden(x0, tol,Nmax)     
    elapsed = time.time()-t
    print(xstar)
    print('Broyden: the error message reads:',ier)
    print('Broyden: took this many seconds:',elapsed/20)
    print('Broyden: number of iterations is:',its)
     
def evalF(x): 
# vector function that you want to find the roots of

    F = np.zeros(2)
    
    F[0] = x[0]**2 + x[1]**2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1
 
    return F
    
def evalJ(x): 
# Jacobian of the vector function you want to find the roots of
    
    J = np.array([
        [  2.0*x[0]  , 2*x[1]], 
        [np.exp(x[0]),   1   ], 
        ])

    return J


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]
           
def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
    
def Broyden(x0,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''


    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''
    
    '''initialize with 1 newton step'''
    
    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = evalF(xk)
       '''y_k = F(xk)-F(xk-1)'''
       y = v-w;                   
       '''-A_{k-1}^{-1}y_k'''
       z = -A.dot(y)
       ''' p = s_k^tA_{k-1}^{-1}y_k'''
       p = -np.dot(s,z)                 
       u = np.dot(s,A) 
       ''' A = A_k^{-1} via Morrison formula'''
       tmp = s+z
       tmp2 = np.outer(tmp,u)
       A = A+1./p*tmp2
       ''' -A_k^{-1}F(x_k)'''
       s = -A.dot(v)
       xk = xk+s
       if (norm(s)<tol):
          alpha = xk
          ier = 0
          return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()       


#%% q3

"""
 This program implements steepest descent for finding the 
 root of a non-linear vector function. 
The vector function F and the corresponding Jacobian are hard 
coded and need to be changed for each problem. 
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




#libraries:
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

p_guess = (0.0, 1.0, 1.0)

tol=1e-8

def driver():

    Nmax = 100
    x0= np.array(p_guess)
    tol = 5e-2
    
    [xstar,gval,ier] = SteepestDescent(x0,tol,Nmax)
    print("the steepest descent code found the solution ",xstar)
    print("g evaluated at this point is ", gval)
    print("ier is ", ier	)

###########################################################
#functions:
def evalF(x):
# vector valued function you are trying to find the root of 

    F = np.zeros(3)
    F[0] = x[0] +math.cos(x[0]*x[1]*x[2])-1.
    F[1] = (1.-x[0])**(0.25) + x[1] +0.05*x[2]**2 -0.15*x[2]-1
    F[2] = -x[0]**2-0.1*x[1]**2 +0.01*x[1]+x[2] -1
    return F

def evalJ(x): 
# Jacobian of the vector valued function you are trying to find the root of.

    J =np.array([[1.+x[1]*x[2]*math.sin(x[0]*x[1]*x[2]),x[0]*x[2]*math.sin(x[0]*x[1]*x[2]),x[1]*x[0]*math.sin(x[0]*x[1]*x[2])],
          [-0.25*(1-x[0])**(-0.75),1,0.1*x[2]-0.15],
          [-2*x[0],-0.2*x[1]+0.01,1]])
    return J

def evalg(x):
# The least squares function you are minimizing

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
# Grad g
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]

def F(p):
    x, y, z = p
    return jnp.array([
        x + jnp.cos(x*y*z) - 1,
        (1 - x)**(1/4) + y + 0.05*z**2 - 0.15*z - 1,
        -x**2 - 0.1*y**2 + 0.01*y + z -1 
    ])

root, count, history = newtons_method(
    F,
    p_guess,
    tol=tol,
)

print(f'Newton Root = {root}')
print(f'Iterations with tol={tol}: {count}')
print()

driver()        


# Steepest descent into Newton

def F(p):
    x, y, z = p
    return jnp.array([
        x + jnp.cos(x*y*z) - 1,
        (1 - x)**(1/4) + y + 0.05*z**2 - 0.15*z - 1,
        -x**2 - 0.1*y**2 + 0.01*y + z -1 
    ])

root, count, history = newtons_method(
    F,
    (-0.02218553, 0.08874213, 0.99556289),
    tol=tol,
)

print()
print(f'Steepest descent into Newton Root = {root}')
print(f'Iterations with tol={tol}: {count}')
print()
        
