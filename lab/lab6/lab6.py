#%% imports
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

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

#%% pre-lab

from numerical_methods.higher_dimensional_methods.nD_newton_method import newtons_method 
from numerical_methods.higher_dimensional_methods.nD_lazy_newton_method import lazy_newtons_method

def F(p):
    x1, x2 = p
    
    return jnp.array(
        [x1 ** 2 + x2 ** 2 - 2,
        jnp.exp(x1 - 1) + x2 ** 2 - 2]
        )
    
# Newton's for x0 = (2.0, 0.5)    

p_guess = jnp.array([2.0, 0.5])

tol = 1e-6

root, count, history = newtons_method(F, p_guess, tol=tol)

print(f"Root for Newton's = {root}")
print(f'Iterations with tol={tol}: {count}')

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))
plt.semilogy(iterations, errors[:-1],'-o')
plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^{\,*}\|_2$')
plt.title("Newton's Method Convergence")
plt.grid(True, which='both', alpha=0.3)
# plt.savefig('pre_lab_newton.png')
plt.show()

# Lazy Newton's for x0 = (3.0, 5.0)    

p_guess = jnp.array([3.0, 5.0])

root, count, history = lazy_newtons_method(F, p_guess, tol=tol)

print(f"\nRoot for lazy Newton's = {root}")
print(f'Iterations with tol={tol}: {count}')

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))
plt.semilogy(iterations, errors[:-1],'-o')
plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^{\,*}\|_2$')
plt.title("Lazy Newton's Method Convergence")
plt.grid(True, which='both', alpha=0.3)
# plt.savefig('pre_lab_lazy_newton.png')
plt.show()


#%% 3.2 Exercises: Build Slacker Newton

from numerical_methods.higher_dimensional_methods.nD_slacker_newton_method import slacker_newtons_method

def F(p):
    x1, x2 = p
    
    return jnp.array(
        [4 * x1 ** 2 + x2 ** 2 - 4,
        x1 + x2 - jnp.sin(x1 - x2)]
        )

tol = 1e-10

# Slacker Newton's for x0 = (1.0, 0.0)    

p_guess = jnp.array([1.0, 0.0])

root, count, history = slacker_newtons_method(F, p_guess, tol=tol)

print(f"\nRoot for Slacker Newton's = {root}")
print(f'Iterations with tol={tol}: {count}')

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))
plt.semilogy(iterations, errors[:-1],'-o')
plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^{\,*}\|_2$')
plt.title(r"Slacker Newton's Method Convergence, $\mathbf{x_0}=(1, 0)$")
plt.grid(True, which='both', alpha=0.3)
plt.savefig('slacker_newton_1.png')
plt.show()


# Slacker Newton's for x0 = (-1.0, 0.0)    

p_guess = jnp.array([-1.0, 0.0])

root, count, history = slacker_newtons_method(F, p_guess, tol=tol)

print(f"\nRoot for Slacker Newton's = {root}")
print(f'Iterations with tol={tol}: {count}')

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))
plt.semilogy(iterations, errors[:-1],'-o')
plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^{\,*}\|_2$')
plt.title(r"Slacker Newton's Method Convergence, $\mathbf{x_0}=(-1, 0)$")
plt.grid(True, which='both', alpha=0.3)
plt.savefig('slacker_newton_2.png')
plt.show()

#%% Slacker Newton for some random system

def F(p):
    x1, x2, x3 = p
    
    return jnp.array(
        [4 * x1 ** 2 + x2 ** 2 - 4 * x3 ** 0.5,
        x1 * x3 + x2 - jnp.sin(x1 - x2),
        jnp.exp(x3) + x2 ** 3 + 6 * x1]
        )


p_guess = jnp.array([-1.0, 1.0, 1.0])

root, count, history = slacker_newtons_method(F, p_guess, tol=tol)

print(f"\nRoot for Slacker Newton's = {root}")
print(f'Iterations with tol={tol}: {count}')

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))
plt.semilogy(iterations, errors[:-1],'-o')
plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^{\,*}\|_2$')
plt.title(r"Slacker Newton's Method Convergence, $\mathbf{x_0}=(-1, 1, 1)$")
plt.grid(True, which='both', alpha=0.3)
# plt.savefig('slacker_newton_for_random_system.png')
plt.show()

