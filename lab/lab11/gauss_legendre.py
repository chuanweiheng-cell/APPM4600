"""
 This program creates Gauss Legendre quadrature nodes and weights.
 It is a translation of a matlab script.  The citation and reference are
 provided below.  It is meant to be imported, e.g. by gauss_legendre_test.py.
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


# gauss legendre quadrature nodes and weights via
# Newton iteration
def lgwt(N,a,b):
    """
     Computes the Legendre-Gauss nodes and weights on an interval
     [a,b] with truncation order N.

     Suppose you have a continuous function f(x) which is defined on [a,b]
     which you can evaluate at any x in [a,b]. Simply evaluate it at all of
     the values contained in the x vector to obtain a vector f. Then compute
     the definite integral using np.sum(f*w)

     Written by Greg von Winckel - 02/25/2004
     translated to Python - 10/30/2022
    """
    N = N-1
    N1 = N+1
    N2 = N+2
    eps = np.finfo(float).eps
    xu = np.linspace(-1,1,N1)

    # Initial guess
    y = np.cos((2*np.arange(0,N1)+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)

    # Legendre-Gauss Vandermonde Matrix
    L = np.zeros((N1,N2))

    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0 = 2.
    one = np.ones((N1,))

    # Iterate until new points are uniformly within epsilon of old points
    while np.max(np.abs(y-y0)) > eps:

        L[:,0] = one
        L[:,1] = y
        for k in range(2,N1+1):
            L[:,k] = ((2*k-1)*y*L[:,k-1]-(k-1)*L[:,k-2])/k

        lp = N2*(L[:,N1-1]-y*L[:,N2-1])/(1-y**2)

        y0 = y
        y = y0-L[:,N2-1]/lp

    # Linear map from [-1,1] to [a,b]
    x = (a*(1-y)+b*(1+y))/2

    # Compute the weights
    w = (b-a)/((1-y**2)*lp**2)*(N2/N1)**2

    return x,w
