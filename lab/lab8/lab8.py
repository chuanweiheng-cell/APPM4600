#%% imports
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy.linalg import inv 

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

#%% 2: Before lab

'''Write a subroutine that constructs and evaluates a line 
that goes through the points (x0, f (x0)) and (x1, f (x1))'''

def make_line(p0, p1):
    p0 = jnp.array(p0)
    p1 = jnp.array(p1)

    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    m = (y1 - y0) / (x1 - x0)

    def line(x):
        return m * (x - x0) + y0

    return line

line = make_line([5, 90], [3, 5])

#%% 3.1 Constructing piecewise linear approximations

def eval_lin_spline(xeval, Neval, a, b, f, Nint):

    # Create the interpolation intervals
    xint = jnp.linspace(a, b, Nint + 1)

    # Initialize evaluation array
    yeval = np.zeros(Neval)

    for jint in range(Nint):

        # Endpoints of the current interval
        a1 = xint[jint]
        b1 = xint[jint + 1]

        fa1 = f(a1)
        fb1 = f(b1)

        # Construct the local interpolation line
        line = make_line((a1, fa1), (b1, fb1))

        # Find evaluation points belonging to this interval
        for kk in range(Neval):

            if a1 <= xeval[kk] <= b1:
                yeval[kk] = line(xeval[kk])

    return yeval

if __name__ == '__main__':

    f = lambda x: jnp.exp(x)
    a, b = 0.0, 1.0
    Neval = 1000
    Nint = 10
    xeval = jnp.linspace(a, b, Neval)
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    fex = f(xeval)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(xeval, fex, 'r', label='$f(x)=e^x$')
    ax[0].plot(xeval, yeval, 'b--', label='Linear spline')
    ax[0].set_ylabel('f(x)')

    ax[0].legend(frameon=False)

    err = jnp.abs(yeval - fex)

    ax[1].semilogy(xeval, err, 'r-', label='Absolute error')

    ax[1].legend(frameon=False)
    plt.show()
    
#%% 3.2 Linear spline experiment

f = lambda x: 1.0 / (1.0 + (10.0*x)**2)

a, b = -1.0, 1.0
Neval = 1000
xeval = np.linspace(a, b, Neval)
fex = f(xeval)

Nint = 10

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(xeval, fex, 'k-', label=r'$f(x)=\frac{{1}}{{1+100x^2}}$')

yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
err = np.abs(yeval - fex)

ax[0].plot(xeval, yeval, label=f'Linear spline, N = {Nint}', color='m')

print(f'N = {Nint}, maximum error = {np.max(err):.6e}')

ax[1].set_xlabel('x')
ax[0].set_ylabel('f(x)')
ax[0].legend(frameon=False)

err = jnp.abs(yeval - fex)

ax[1].semilogy(xeval, err, 'r-', label='Absolute error')

ax[1].legend(frameon=False)
plt.show()


plt.show()

#%% 3.3 Constructing natural cubic splines

def create_natural_spline(yint, xint, N):

    # Widths of interpolation intervals
    h = np.diff(xint)

    # Construct matrix A and right-hand side b
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # Natural boundary conditions
    A[0, 0] = 1.0
    A[N, N] = 1.0

    # Interior continuity equations
    for i in range(1, N):

        A[i, i-1] = h[i-1]
        A[i, i] = 2.0 * (h[i-1] + h[i])
        A[i, i+1] = h[i]

        slope_right = (yint[i+1] - yint[i]) / h[i]
        slope_left = (yint[i] - yint[i-1]) / h[i-1]

        b[i] = 6.0 * (slope_right - slope_left)

    # Solve for the second-derivative coefficients
    M = np.linalg.solve(A, b)

    # Compute remaining coefficients
    C = np.zeros(N)
    D = np.zeros(N)

    for j in range(N):

        C[j] = yint[j] / h[j] - h[j] * M[j] / 6.0
        D[j] = yint[j+1] / h[j] - h[j] * M[j+1] / 6.0

    return M, C, D


def eval_local_spline(x, xi, xip, Mi, Mip, C, D):

    h = xip - xi

    yeval = ((Mi * (xip-x)**3 + Mip * (x-xi)**3) / (6.0*h)
             + C * (xip-x) + D * (x-xi))

    return yeval


def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):

    yeval = np.zeros(Neval)

    for j in range(Nint):

        # Find evaluation points inside the current interval
        ind = (xeval >= xint[j]) & (xeval <= xint[j+1])

        # Evaluate the local cubic polynomial
        yeval[ind] = eval_local_spline(
            xeval[ind], xint[j], xint[j+1],
            M[j], M[j+1], C[j], D[j]
        )

    return yeval


#%% 3.4 Comparing linear and cubic splines

f = lambda x: 1.0 / (1.0 + (10.0*x)**2)

a, b = -1.0, 1.0
Neval = 1000
Nint = 10

# Evaluation points and exact function
xeval = np.linspace(a, b, Neval)
fex = f(xeval)

# Interpolation nodes
xint = np.linspace(a, b, Nint + 1)
yint = f(xint)

# Construct the natural cubic spline coefficients
M, C, D = create_natural_spline(yint, xint, Nint)

# Evaluate the cubic spline
yeval_cubic = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)

# Evaluate the linear spline for comparison
yeval_linear = eval_lin_spline(xeval, Neval, a, b, f, Nint)

# Absolute errors
err_cubic = np.abs(yeval_cubic - fex)
err_linear = np.abs(yeval_linear - fex)

print(f'Cubic spline maximum error = {np.max(err_cubic):.6e}')
print(f'Linear spline maximum error = {np.max(err_linear):.6e}')

# Plot comparisons
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(xeval, fex, 'k-', label=r'$f(x)=\frac{1}{1+100x^2}$')
ax[0].plot(xeval, yeval_cubic, 'm--', label='Natural cubic spline')
ax[0].plot(xeval, yeval_linear, 'b:', label='Linear spline')
ax[0].plot(xint, yint, 'ko', markersize=4, label='Interpolation nodes')

ax[0].set_ylabel('f(x)')
ax[0].legend(frameon=False)

ax[1].semilogy(xeval, err_cubic, 'm-', label='Cubic spline error')
ax[1].semilogy(xeval, err_linear, 'b--', label='Linear spline error')

ax[1].set_xlabel('x')
ax[1].set_ylabel('Absolute error')
ax[1].legend(frameon=False)

plt.show()