#%% imports

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpy as np

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
    'legend.frameon':False,
    'figure.dpi': 300
})

#%% Set up

n1, n2, n3, n4 = 5, 10, 15, 20
Ns = [n1, n2, n3, n4]

f = lambda x: 1 / (1 + x**2)

N_fine = 10000
x_min, x_max = -5.0, 5.0

# Fine grid for evaluation and plotting
x_fine = jnp.linspace(x_min, x_max, N_fine)

# Select interpolation nodes
use_chebyshev = True

def get_nodes(n):

    if use_chebyshev:
        i = jnp.arange(n)
        return (x_min + x_max)/2 - (x_max - x_min)/2*jnp.cos(i*jnp.pi/(n - 1))

    return jnp.linspace(x_min, x_max, n)

#%% part a: Lagrange interpolation

def lagrange_j(x, x_data, j):

    x_data = jnp.asarray(x_data)

    mask = jnp.arange(x_data.size) != j

    num = jnp.where(mask, x - x_data, 1.0)
    den = jnp.where(mask, x_data[j] - x_data, 1.0)

    return jnp.prod(num / den)


# Vectorize over evaluation points
lagrange_j_vec = jax.vmap(lagrange_j, in_axes=(0, None, None))
 
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for k, n in enumerate(Ns):

    x_is = get_nodes(n)    
    f_i = f(x_is)

    P = jnp.zeros_like(x_fine)

    for j in range(x_is.size):

        phi_j = lagrange_j_vec(x_fine, x_is, j)
        P += f_i[j] * phi_j

    ax.flat[k].plot(x_fine, f(x_fine), 'k-', label='Exact')
    ax.flat[k].plot(x_fine, P, 'r--', label='Lagrange')
    ax.flat[k].plot(x_is, f_i, 'ko', markersize=4, label='Nodes')

    ax.flat[k].set_title(f'n = {n} nodes')
    ax.flat[k].set_ylabel('f(x)')
    ax.flat[k].legend()

for a in ax[-1]:
    a.set_xlabel('x')

plt.tight_layout()
plt.show()

#%% part b: Hermite interpolation

def hermite_Q_j(x, x_data, j):

    x_data = jnp.asarray(x_data)
    dLj = jax.grad(lagrange_j, argnums=0)(x_data[j], x_data, j)
    Lj = lagrange_j(x, x_data, j)

    return (1.0 - 2.0*(x - x_data[j])*dLj) * Lj**2

hermite_Q_j_vec = jax.vmap(hermite_Q_j, in_axes=(0, None, None))

def hermite_R_j(x, x_data, j):

    x_data = jnp.asarray(x_data)
    Lj = lagrange_j(x, x_data, j)

    return (x - x_data[j]) * Lj**2

hermite_R_j_vec = jax.vmap(hermite_R_j, in_axes=(0, None, None))

fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

df = jax.vmap(jax.grad(f))

for k, n in enumerate(Ns):
    
    x_is = get_nodes(n)    
    f_i = f(x_is)
    df_i = df(x_is)

    P = jnp.zeros_like(x_fine)

    for j in range(x_is.size):

        Q_j = hermite_Q_j_vec(x_fine, x_is, j)
        R_j = hermite_R_j_vec(x_fine, x_is, j)

        P += f_i[j]*Q_j + df_i[j]*R_j

    ax.flat[k].plot(x_fine, f(x_fine), 'k-', label='Exact')
    ax.flat[k].plot(x_fine, P, 'r--', label='Hermite')
    ax.flat[k].plot(x_is, f_i, 'ko', markersize=4, label='Nodes')

    ax.flat[k].set_title(f'n = {n} nodes')
    ax.flat[k].set_ylabel('f(x)')
    ax.flat[k].legend()

for a in ax[-1]:
    a.set_xlabel('x')

plt.tight_layout()
plt.show()

#%% Part (c): Natural cubic spline interpolation

fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for k, n in enumerate(Ns):

    # Interpolation nodes
    x_is = get_nodes(n)
    f_i = f(x_is)

    # Number and widths of intervals
    N = n - 1
    h = jnp.diff(x_is)

    # Construct the linear system A @ M = b
    A = jnp.zeros((n, n))
    b = jnp.zeros(n)

    # Natural boundary conditions
    A = A.at[0, 0].set(1.0)
    A = A.at[N, N].set(1.0)

    # Interior continuity equations
    for i in range(1, N):

        A = A.at[i, i-1].set(h[i-1])
        A = A.at[i, i].set(2.0*(h[i-1] + h[i]))
        A = A.at[i, i+1].set(h[i])

        slope_right = (f_i[i+1] - f_i[i]) / h[i]
        slope_left = (f_i[i] - f_i[i-1]) / h[i-1]

        b = b.at[i].set(6.0*(slope_right - slope_left))

    # Solve for second-derivative coefficients
    M = jnp.linalg.solve(A, b)

    # Construct the remaining spline coefficients
    C = f_i[:-1]/h - h*M[:-1]/6.0
    D = f_i[1:]/h - h*M[1:]/6.0

    # Initialize the spline
    P = jnp.zeros_like(x_fine)

    # Evaluate the cubic polynomial on each interval
    for j in range(N):

        xi = x_is[j]
        xip = x_is[j+1]

        P_j = ((M[j]*(xip-x_fine)**3 + M[j+1]*(x_fine-xi)**3)/(6.0*h[j])
               + C[j]*(xip-x_fine) + D[j]*(x_fine-xi))

        # Select evaluation points within the current interval
        mask = (x_fine >= xi) & (x_fine <= xip)

        P = jnp.where(mask, P_j, P)

    # Plot exact function and interpolation
    ax.flat[k].plot(x_fine, f(x_fine), 'k-', label='Exact')
    ax.flat[k].plot(x_fine, P, 'r--', label='Natural cubic spline')
    ax.flat[k].plot(x_is, f_i, 'ko', markersize=4, label='Nodes')

    ax.flat[k].set_title(f'n = {n} nodes')
    ax.flat[k].set_ylabel('f(x)')
    ax.flat[k].legend()

for a in ax[-1]:
    a.set_xlabel('x')

plt.tight_layout()
plt.show()

#%% Part (d): Clamped cubic spline interpolation

fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for k, n in enumerate(Ns):

    # Interpolation nodes
    x_is = get_nodes(n)
    f_i = f(x_is)

    # Number and widths of intervals
    N = n - 1
    h = jnp.diff(x_is)

    # Construct the linear system A @ M = b
    A = jnp.zeros((n, n))
    b = jnp.zeros(n)

    # Clamped boundary conditions

    # First derivatives at the endpoints
    df = jax.grad(f)

    df_left = df(x_is[0])
    df_right = df(x_is[-1])

    # Left boundary: S'(x_0) = f'(x_0)
    A = A.at[0, 0].set(2.0*h[0])
    A = A.at[0, 1].set(h[0])

    b = b.at[0].set(
        6.0*((f_i[1] - f_i[0])/h[0] - df_left)
    )

    # Right boundary: S'(x_N) = f'(x_N)
    A = A.at[N, N-1].set(h[N-1])
    A = A.at[N, N].set(2.0*h[N-1])

    b = b.at[N].set(
        6.0*(df_right - (f_i[N] - f_i[N-1])/h[N-1])
    )
    # Interior continuity equations
    for i in range(1, N):

        A = A.at[i, i-1].set(h[i-1])
        A = A.at[i, i].set(2.0*(h[i-1] + h[i]))
        A = A.at[i, i+1].set(h[i])

        slope_right = (f_i[i+1] - f_i[i]) / h[i]
        slope_left = (f_i[i] - f_i[i-1]) / h[i-1]

        b = b.at[i].set(6.0*(slope_right - slope_left))

    # Solve for second-derivative coefficients
    M = jnp.linalg.solve(A, b)

    # Construct the remaining spline coefficients
    C = f_i[:-1]/h - h*M[:-1]/6.0
    D = f_i[1:]/h - h*M[1:]/6.0

    # Initialize the spline
    P = jnp.zeros_like(x_fine)

    # Evaluate the cubic polynomial on each interval
    for j in range(N):

        xi = x_is[j]
        xip = x_is[j+1]

        P_j = ((M[j]*(xip-x_fine)**3 + M[j+1]*(x_fine-xi)**3)/(6.0*h[j])
               + C[j]*(xip-x_fine) + D[j]*(x_fine-xi))

        # Select evaluation points within the current interval
        mask = (x_fine >= xi) & (x_fine <= xip)

        P = jnp.where(mask, P_j, P)

    # Plot exact function and interpolation
    ax.flat[k].plot(x_fine, f(x_fine), 'k-', label='Exact')
    ax.flat[k].plot(x_fine, P, 'r--', label='Clamped cubic spline')
    ax.flat[k].plot(x_is, f_i, 'ko', markersize=4, label='Nodes')

    ax.flat[k].set_title(f'n = {n} nodes')
    ax.flat[k].set_ylabel('f(x)')
    ax.flat[k].legend()

for a in ax[-1]:
    a.set_xlabel('x')

plt.tight_layout()
plt.show()
