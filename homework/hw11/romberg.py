"""
Compare Romberg integration based on composite trapezoidal and Simpson rules
for the integral of sin(x) on [0, pi], whose exact value is 2.

Adapted from romberg.py by Adrianna M. Gillman (2026).
"""

#############################################
"""
Copyright (C) 2026  Adrianna M. Gillman

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


def driver():
    f = lambda x: np.sin(x)
    a = 0
    b = np.pi
    I_ex = 2.0

    nrefine = 6

    # Trapezoidal Romberg starts with n = 1; Simpson requires even n.
    # The final errors are compared at the SAME n = 2, 4, ..., 2^(nrefine+1).
    ntrap = 2**np.arange(0, nrefine + 2)
    ntest = 2**np.arange(1, nrefine + 2)

    RomTrap = rombergTrap(a, b, nrefine + 1, f)
    RomSimp = rombergSimp(a, b, nrefine, f)

    # Diagonal entries: highest-order extrapolation available at each n.
    # Skip n = 1 in the trapezoidal table to align the panel counts.
    err_trap = abs(I_ex - np.diag(RomTrap)[1:])
    err_simp = abs(I_ex - np.diag(RomSimp))

    print('  n       Romberg-Trap error      Romberg-Simpson error')
    for n, et, es in zip(ntest, err_trap, err_simp):
        print(f'{n:4d}       {et:.8e}              {es:.8e}')

    print('\nMaximum difference between matched diagonal estimates:')
    print(np.max(abs(np.diag(RomTrap)[1:] - np.diag(RomSimp))))

    # Also compare the underlying fourth- and second-order quadratures.
    err_base_trap = abs(I_ex - RomTrap[1:, 0])
    err_base_simp = abs(I_ex - RomSimp[:, 0])

    plt.figure(figsize=(8, 5))
    plt.loglog(ntest, err_base_trap, 'o-', label='Composite trapezoidal')
    plt.loglog(ntest, err_base_simp, 's-', label='Composite Simpson')
    plt.loglog(ntest, np.maximum(err_trap, np.finfo(float).eps),
               'o--', label='Romberg-Trapezoidal')
    plt.loglog(ntest, np.maximum(err_simp, np.finfo(float).eps),
               'x:', label='Romberg-Simpson')
    plt.xlabel('Number of panels, n')
    plt.ylabel('Absolute error (Romberg values floored at machine epsilon)')
    plt.title(r'Approximating $\int_0^\pi \sin(x)\,dx = 2$')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def rombergTrap(a, b, nrefine, f):
    ntest = 2**np.arange(0, nrefine + 1)
    nn = len(ntest)
    Rom = np.zeros((nn, nn))

    for i in range(nn):
        Rom[i, 0] = CompTrap(a, b, ntest[i], f)

    # Trapezoidal errors: h^2, h^4, h^6, ...
    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j-1] + (Rom[i, j-1] - Rom[i-1, j-1])/(4**j - 1)

    return Rom


def rombergSimp(a, b, nrefine, f):
    ntest = 2**np.arange(1, nrefine + 2)  # n = 2, 4, 8, ...
    nn = len(ntest)
    Rom = np.zeros((nn, nn))

    for i in range(nn):
        Rom[i, 0] = CompSimp(a, b, ntest[i], f)

    # Simpson errors: h^4, h^6, h^8, ...
    # Column j removes the h^(2*j+2) term: denominators 15, 63, 255, ...
    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j-1] + (Rom[i, j-1] - Rom[i-1, j-1])/(4**(j+1) - 1)

    return Rom


def CompTrap(a, b, n, f):
    h = (b-a)/n
    xnode = a + np.arange(n+1)*h
    return h*(0.5*f(xnode[0]) + np.sum(f(xnode[1:-1])) + 0.5*f(xnode[-1]))


def CompSimp(a, b, n, f):
    if n % 2 != 0:
        raise ValueError('Composite Simpson requires an even number of panels.')

    h = (b-a)/n
    xnode = a + np.arange(n+1)*h
    return (h/3)*(f(xnode[0]) + 4*np.sum(f(xnode[1:-1:2]))
                  + 2*np.sum(f(xnode[2:-1:2])) + f(xnode[-1]))


if __name__ == '__main__':
    driver()

    
