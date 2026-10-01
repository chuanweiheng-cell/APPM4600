#%% imports
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

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


#%% n-dimensional fixed-point iteration

def fixed_point_iteration(G, p_guess, tol=1e-10, max_iter=1000):

    # Convert initial guess to a JAX array
    p_guess = jnp.array(
        p_guess,
        dtype=float
    )

    # Store initial guess
    history = [p_guess]

    count = 0

    while count < max_iter:

        # Fixed-point iteration:
        #
        #     p_(n+1) = G(p_n)
        p_new = jnp.array(
            G(p_guess),
            dtype=float
        )

        # Error estimate between successive approximations
        err = jnp.linalg.norm(
            p_new - p_guess,
            ord=2
        )

        # Store new approximation
        history.append(p_new)

        count += 1

        # Check convergence
        if err < tol:

            fixed_point = p_new
            break

        # Update approximation
        p_guess = p_new

    else:

        raise RuntimeError(
            f'Fixed-point iteration did not converge within '
            f'{max_iter} iterations.'
        )

    history = jnp.array(history)

    return fixed_point, count, history


#%% example

def G(p):

    x, y = p

    return jnp.array([
        jnp.cos(y),
        jnp.sin(x)
    ])


p_guess = jnp.array([
    1.0,
    1.0
])

tol = 1e-8

fixed_point, count, history = fixed_point_iteration(
    G,
    p_guess,
    tol=tol
)

print(f'Fixed point = {fixed_point}')
print(f'Iterations with tol={tol}: {count}')


#%% convergence plot

# Error relative to the converged fixed point
errors = jnp.linalg.norm(
    history - fixed_point,
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
plt.ylabel(r'$\|\vec{p}_{n+1}-\vec{p}_n\|_2$')
plt.title('Fixed-Point Iteration Convergence')

plt.grid(
    True,
    which='both',
    alpha=0.3
)

plt.show()