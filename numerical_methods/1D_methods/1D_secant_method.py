#%% Imports

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


#%% Secant method

def secant_method(
    f,
    x0,
    x1,
    tol=1e-10,
    derivative_tol=1e-14,
    max_iter=1000
):

    count = 0

    # Store initial guesses
    roots = [x0, x1]

    while count < max_iter:

        # Secant approximation to derivative
        derivative = (
            (f(x1) - f(x0))
            / (x1 - x0)
        )

        # Secant method requires a nonzero approximate derivative
        if jnp.abs(derivative) <= derivative_tol:

            raise ZeroDivisionError(
                'Secant slope is too close to zero.'
            )

        # Secant update
        x2 = x1 - f(x1) / derivative

        # Store new approximation
        roots.append(x2)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x2 - x1)

        # Check convergence
        if err <= tol:

            root = x2

            break

        # Update approximations
        x0 = x1
        x1 = x2

    else:

        raise RuntimeError(
            f'Secant method did not converge within '
            f'{max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots


#%% Example problem

# Solve:
#
#     x^3 - x - 2 = 0

f = lambda x: x ** 3 - x - 2

# Initial guesses
x0 = 1.5
x1 = 2.0

# Tolerances
tol = 1e-10
derivative_tol = 1e-14

root, count, roots = secant_method(
    f,
    x0,
    x1,
    tol=tol,
    derivative_tol=derivative_tol,
    max_iter=1000
)

print(f'Root from Secant method = {root}')
print(f'Iterations from Secant method with tol={tol}: {count}')
print(f'f(root) = {f(root):.3e}')


#%% Convergence plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    range(len(roots)),
    roots,
    '-o'
)

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Secant approximation $x_n$')
ax.set_title('Secant Method Convergence Profile')

ax.text(
    0.55,
    0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root:.6f}$',
    transform=ax.transAxes,
    ha='left',
    va='top',
    bbox={
        'facecolor': 'yellow',
        'alpha': 0.4,
        'pad': 5
    }
)

ax.text(
    0.55,
    0.52,
    f'Number of iterations = {count}',
    transform=ax.transAxes,
    ha='left',
    va='top',
    bbox={
        'facecolor': 'yellow',
        'alpha': 0.4,
        'pad': 5
    }
)

plt.show()