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

#%% question 1 part b

def make_matrix(n, x_data):

    V = np.zeros((n, n))
    x_data = np.array(x_data)

    for i in range(n):
        for j in range(n):
            V[i, j] = x_data[i] ** (n - 1 - j)

    return V


def evaluate_polynomial(x, C):

    n = len(C)

    p = np.zeros_like(np.array(x), dtype=float)

    for j in range(n):
        p += C[j] * np.array(x) ** (n - 1 - j)

    return p

f = lambda x: 1 / (1 + (10*x)**2)

x_fine = jnp.linspace(-1, 1, 1001)

N = 10

x_interp = jnp.array([-1 + (i - 1)*2/(N - 1) for i in range(1, N + 1)])

fx = f(x_interp)

V = make_matrix(x_interp.size, x_interp)

C = np.linalg.solve(V, np.array(fx))

p_fine = evaluate_polynomial(x_fine, C)


plt.figure(figsize=(8, 5))

plt.plot(x_fine, f(x_fine), label=r'$f(x)$')
plt.plot(x_fine, p_fine, ls='--', label=r'$p(x)$')
plt.plot(x_interp, fx, 'o', label='Interpolation points')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(f'for N={N} interpolant nodes')
# plt.xlim(-0.75, 0.75)
# plt.ylim(0, 2)

plt.legend()
plt.grid(alpha=0.3)

plt.savefig('homework/hw7/q1_N_10')
plt.show()