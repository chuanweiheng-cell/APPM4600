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
    'legend.frameon': False,
    'figure.dpi': 200
})

#%% question 2

def barycentric_weights(x_data):
    n = len(x_data)
    weights = np.zeros(n)
    for j in range(n):
        product = 1.0
        for i in range(n):
            if i != j:
                product *= x_data[j] - x_data[i]

        weights[j] = 1 / product

    return weights

def barycentric_interpolation(x, x_data, y_data, weights):
    for j in range(len(x_data)):
        if np.isclose(x, x_data[j]):
            return y_data[j]

    numerator = 0.0
    denominator = 0.0

    for j in range(len(x_data)):

        term = weights[j] / (x - x_data[j])

        numerator += term * y_data[j]
        denominator += term

    return numerator / denominator


f = lambda x: 1 / (1 + (10*x)**2)
N = 1000
x_interp = np.array([-1 + (i - 1)*2/(N - 1) for i in range(1, N + 1)])
fx = f(x_interp)
x_fine = np.linspace(-1, 1, 1000)

weights = barycentric_weights(x_interp)

p_fine = np.array([barycentric_interpolation(x, x_interp, fx, weights) for x in x_fine])

plt.figure(figsize=(8, 5))

plt.plot(x_fine, f(x_fine), label=r'$f(x)$')
plt.plot(x_fine, p_fine, ls='--', label=r'$p(x)$')
plt.plot(x_interp, fx, 'o', markersize=4, label='Interpolation points')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(f'for N={N} interpolant nodes')
plt.legend()
plt.grid(alpha=0.3)

# plt.savefig('homework/hw7/q2_N_1000')
plt.show()


#%% question 3

N = 1000
x_chebyshev = np.array([np.cos((2*i - 1)*np.pi/2/N) for i in range(1, N + 1)])
fx = f(x_chebyshev)
x_fine = np.linspace(-1, 1, 1000)

weights = barycentric_weights(x_chebyshev)

p_fine = np.array([barycentric_interpolation(x, x_chebyshev, fx, weights) for x in x_fine])

plt.figure(figsize=(8, 5))

plt.plot(x_fine, f(x_fine), label=r'$f(x)$')
plt.plot(x_fine, p_fine, ls='--', label=r'$p(x)$')
plt.plot(x_chebyshev, fx, 'o', markersize=4, label='Interpolation points')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(f'for N={N} interpolant nodes')
plt.legend()
plt.grid(alpha=0.3)

# plt.savefig('homework/hw7/q3_N_1000')
plt.show()

