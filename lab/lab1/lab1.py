#%% imports
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

#%% 3.2 Exercises: The Basics

x = np.linspace(0, 2, 10)
y = np.arange(0, 2.001, 0.21)

print(f'the size of x is {x.size}, the last element is {x[-1]}')
print(f'the size of y is {y.size}, the last element is {y[-1]}')

print(f'the first three entries of x are {x[0:3]}')

w = 10 ** (-np.linspace(1, 10, 10))
x = np.array(range(w.size))

plt.semilogy(x, w, label='w')
plt.xlabel('$n\in \mathbb{{N}}$')
plt.ylabel('$w=10^{{x}}$')
plt.legend()

s = 3 * w
plt.semilogy(x, s, label='s')
plt.ylabel('$s=3w$')
plt.legend()

plt.show()

#%% 4.2 Exercises

"""
 This program is a warm up for coding. You get used to the coding 
format and practice some coding skills. 
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
import numpy.linalg as la
import math

def driver():

     n = 100
     x = np.linspace(0,np.pi,n)

# this is a function handle.  You can use it to define 
# functions instead of using a subroutine like you 
# have to in a true low level language.     
     f = lambda x: np.sin(x)
     g = lambda x: np.cos(x)

     y = f(x)
     w = g(x)

# evaluate the dot product of y and w     
     dp = dotProduct(y,w,n)

# print the output
     print('the dot product is : ', dp)

     return
     
def dotProduct(x,y,n):
#   Computes the dot product of the n x 1 vectors x and y
     dp = 0.
     for j in range(n):
        dp = dp + x[j]*y[j]

     return dp  
     
driver()               

A = np.array((
    [1, 2, 3],
    [3, 4, 5],
    [3, 5, 7]
             ))

def matrix_mult(A, B):
    rows_of_A, help = np.shape(A)
    help, cols_of_B = np.shape(B)
    C = np.zeros((rows_of_A, cols_of_B))
    
    for i in
    
    