#%% Imports

import jax
jax.config.update('jax_enable_x64', True)

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})


#%% Newton's method

def newton_method(f, dfdx, x0, tol=1e-10, max_iter=1000):
    """
    Approximate a root of f(x) using Newton's method.

    Newton's method generates the sequence

        x_(n+1) = x_n - f(x_n) / f'(x_n)

    until two successive approximations are sufficiently close.
    """

    count = 0

    # Store initial guess
    roots = [x0]

    while count < max_iter:

        # Evaluate derivative at current approximation
        derivative = dfdx(x0)

        # Newton's method requires a nonzero derivative
        if jnp.abs(derivative) < 1e-14:
            raise ZeroDivisionError(
                'Derivative is too close to zero for Newton iteration.'
            )

        # Newton update
        x1 = x0 - f(x0) / derivative

        # Store new approximation
        roots.append(x1)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x1 - x0)

        # Check convergence
        if err <= tol:

            root = x1
            break

        # Update approximation for next iteration
        x0 = x1

    else:

        raise RuntimeError(
            f'Newton method did not converge within {max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots


#%% Example problem

# Solve:
#
#     x^3 - x - 2 = 0

f = lambda x: x ** 3 - x - 2

# Derivative using JAX automatic differentiation
dfdx = jax.grad(f)

# Initial guess
x0 = 1.5

tol = 1e-10

root, count, roots = newton_method(
    f,
    dfdx,
    x0,
    tol=tol
)

print(f'Root from Newton method = {root}')
print(f'Iterations from Newton method with tol={tol}: {count}')
print(f'f(root) = {f(root):.3e}')


#%% Convergence plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(roots)), roots, '-o')

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Newton approximation $x_n$')
ax.set_title("Newton's Method Convergence Profile")

ax.text(
    0.55, 0.65,
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
    0.55, 0.52,
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