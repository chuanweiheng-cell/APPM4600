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


#%% Modified Newton's method 1

def modified_newton_method_1(
    f,
    dfdx,
    x0,
    m,
    tol=1e-10,
    derivative_tol=1e-14,
    max_iter=1000
):

    count = 0

    # Store initial guess
    roots = [x0]

    while count < max_iter:

        # Evaluate derivative at current approximation
        derivative = dfdx(x0)

        # For a multiple root, f'(x) approaches zero
        if jnp.abs(derivative) <= derivative_tol:

            root = x0

            break

        # Modified Newton update
        x1 = x0 - m * f(x0) / derivative

        # Store new approximation
        roots.append(x1)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x1 - x0)

        # Check convergence
        if err <= tol:

            root = x1

            break

        # Update approximation
        x0 = x1

    else:

        raise RuntimeError(
            f'Modified Newton Method 1 did not converge within '
            f'{max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots


#%% Example problem

# Solve:
#
#     f(x) = (x - 1)^3 = 0
#
# The root:
#
#     x* = 1
#
# has multiplicity:
#
#     m = 3

f = lambda x: (x - 1.0) ** 3

# Derivative using JAX automatic differentiation
dfdx = jax.grad(f)

# Known multiplicity
m = 3

# Initial guess
x0 = 2.0

# Tolerances
tol = 1e-10
derivative_tol = 1e-14

root, count, roots = modified_newton_method_1(
    f,
    dfdx,
    x0,
    m,
    tol=tol,
    derivative_tol=derivative_tol,
    max_iter=10000
)

print(f'Root from Modified Newton Method 1 = {root}')
print(f'Root multiplicity = {m}')
print(f'Iterations with tol={tol}: {count}')
print(f'f(root) = {f(root):.3e}')


#%% Convergence plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    range(len(roots)),
    roots,
    '-o'
)

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Modified Newton approximation $x_n$')
ax.set_title("Modified Newton's Method 1 Convergence Profile")

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