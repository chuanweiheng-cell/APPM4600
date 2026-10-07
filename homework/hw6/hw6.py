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

#(i): (x0, y0) = (-0.5, 0.25)
p_guess = (-0.5, 0.25)

root, count, history = fixed_point_iteration(F, p_guess, tol)

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
plt.title('Fixed-Point Iteration Convergence')

plt.grid(True, which='both', alpha=0.3)

plt.show()

# #(ii): (x0, y0) = (-1.5, 1.25)
# p_guess = (-1.5, 1.25)

# root, count, history = fixed_point_iteration(F, p_guess, tol)

# print(f'Root from fixed-point iteration = {root}')
# print(f'Iterations with tol={tol}: {count}')

# #Convergence plot

# # Error relative to the converged root
# errors = jnp.linalg.norm(history - root, axis=1)

# # Exclude the final point since its error is exactly zero
# iterations = jnp.arange(len(history) - 1)

# plt.figure(figsize=(8, 5))

# plt.semilogy(iterations, errors[:-1], '-o')

# plt.xlabel('Iteration')
# plt.ylabel(r'$\|\vec{p}_n-\vec{p}^*\|_2$')
# plt.title('Fixed-Point Iteration Convergence')

# plt.grid(True, which='both', alpha=0.3)

# plt.show()

# #(iii): (x0, y0) = (-3, 3)
# p_guess = (-3.0, 3.0)

# root, count, history = fixed_point_iteration(F, p_guess, tol)

# print(f'Root from fixed-point iteration = {root}')
# print(f'Iterations with tol={tol}: {count}')

# #Convergence plot

# # Error relative to the converged root
# errors = jnp.linalg.norm(history - root, axis=1)

# # Exclude the final point since its error is exactly zero
# iterations = jnp.arange(len(history) - 1)

# plt.figure(figsize=(8, 5))

# plt.semilogy(iterations, errors[:-1], '-o')

# plt.xlabel('Iteration')
# plt.ylabel(r'$\|\vec{p}_n-\vec{p}^*\|_2$')
# plt.title('Fixed-Point Iteration Convergence')

# plt.grid(True, which='both', alpha=0.3)

# plt.show()

# part b, Newtons

from numerical_methods.higher_dimensional_methods.nD_newton_method import newtons_method

def F(p):

    x, y = p

    return jnp.array([
        3.0*x**2 + 4.0*y**2 - 1.0,
        y**3 - 8.0*x**3 - 1.0
    ])


# (i): (x0, y0) = (-0.5, 0.25)
p_guess = jnp.array([
    -0.5,
     0.25
])

tol = 1e-8

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
plt.title("Newton's Method Convergence")

plt.grid(
    True,
    which='both',
    alpha=0.3
)

plt.show()

# (ii): (x0, y0) = (-1.5, 1.25)
p_guess = jnp.array([
    -1.5,
     1.25
])

tol = 1e-8

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
plt.title("Newton's Method Convergence")

plt.grid(
    True,
    which='both',
    alpha=0.3
)

plt.show()

# (iii): (x0, y0) = (-3, 3)
p_guess = jnp.array([
    -3.0,
     3.0
])

tol = 1e-8

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
plt.title("Newton's Method Convergence")

plt.grid(
    True,
    which='both',
    alpha=0.3
)

plt.show()