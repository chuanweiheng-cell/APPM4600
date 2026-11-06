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

#%% q2

A = np.array([
    [581, 105],
    [105, 454]
])

b = np.array([421, 228])

x = np.linalg.solve(A, b)

print(x)


#%% q4 

# Evaluation grid and exact function
x = np.linspace(0.0, 5.0, 2000)
f_exact = np.sin(x)

# Padé approximations
pade_33 = (x - 7.0*x**3/60.0) / (1.0 + x**2/20.0)
pade_42 = (x - 7.0*x**3/60.0) / (1.0 + x**2/20.0)
pade_24 = x / (1.0 + x**2/6.0 + 7.0*x**4/360.0)

# Sixth-order Maclaurin polynomial
taylor_6 = x - x**3/6.0 + x**5/120.0

# Absolute errors
error_33 = np.abs(f_exact - pade_33)
error_42 = np.abs(f_exact - pade_42)
error_24 = np.abs(f_exact - pade_24)
error_taylor = np.abs(f_exact - taylor_6)

# Plot 1: All approximation errors
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, error_33, label='Padé [3/3] = [4/2]', linewidth=2)
ax.plot(x, error_24, label='Padé [2/4]', linewidth=2)
ax.plot(x, error_taylor, label='Maclaurin (degree 6)', linewidth=2, linestyle='--')

ax.set_xlabel('x')
ax.set_ylabel('Absolute error')
ax.set_title('Approximation Error of sin(x)')
ax.set_xlim(0.0, 5.0)
ax.grid(True, alpha=0.25)
ax.legend()

plt.tight_layout()
plt.show()

