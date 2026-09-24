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


#%% Hybrid bisection-Newton method

def hybrid_method(f, dfdx, d2fdx2, L_boundary, R_boundary,
                  tol=1e-10, max_iter=1000):

    # Evaluate initial boundaries
    func_L = f(L_boundary)
    func_R = f(R_boundary)

    # Root must initially be bracketed
    if func_L * func_R > 0:
        raise ValueError(
            'Initial interval does not bracket a root.'
        )

    count = 0
    roots = []

    # ---------------------------------------------------------
    # Bisection stage
    # ---------------------------------------------------------

    while count < max_iter:

        # Current midpoint
        mid_point = (L_boundary + R_boundary) / 2

        roots.append(mid_point)

        # First derivative cannot be zero in switching criterion
        derivative = dfdx(mid_point)

        if jnp.abs(derivative) < 1e-14:
            condition = jnp.inf

        else:
            condition = (
                f(mid_point)
                * d2fdx2(mid_point)
                / derivative ** 2
            )

        # Switch to Newton once local contraction condition is met
        if jnp.abs(condition) < 1:
            break

        # Evaluate midpoint
        func_mid = f(mid_point)

        # Determine which half still brackets the root
        if func_L * func_mid < 0:

            R_boundary = mid_point

        else:

            L_boundary = mid_point
            func_L = func_mid

        count += 1

    else:

        raise RuntimeError(
            f'Hybrid method did not switch within {max_iter} iterations.'
        )

    # Record where Newton's method begins
    switch_iteration = count

    # Use final bisection midpoint as Newton initial guess
    x0 = mid_point

    # ---------------------------------------------------------
    # Newton stage
    # ---------------------------------------------------------

    while count < max_iter:

        derivative = dfdx(x0)

        if jnp.abs(derivative) < 1e-14:
            raise ZeroDivisionError(
                'Derivative is too close to zero for Newton iteration.'
            )

        # Newton update
        x1 = x0 - f(x0) / derivative

        roots.append(x1)

        count += 1

        # Check convergence
        if jnp.abs(x1 - x0) <= tol:

            root = x1
            break

        x0 = x1

    else:

        raise RuntimeError(
            f'Hybrid method did not converge within {max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots, switch_iteration


#%% Example problem

# Solve:
#
#     exp(0.5 sin(2x)) + x^4 / 20 - 11 / 5 = 0

f = lambda x: (
    jnp.exp(0.5 * jnp.sin(2 * x))
    + x ** 4 / 20
    - 11 / 5
)

# Derivatives using JAX automatic differentiation
dfdx = jax.grad(f)
d2fdx2 = jax.grad(dfdx)

# Initial bracketing interval
L_boundary = 0.5
R_boundary = 2.5

tol = 1e-10

root, count, roots, switch_iteration = hybrid_method(
    f,
    dfdx,
    d2fdx2,
    L_boundary,
    R_boundary,
    tol=tol
)

print(f'Root from hybrid method = {root}')
print(f'Total iterations with tol={tol}: {count}')
print(f'Switched to Newton at iteration = {switch_iteration}')
print(f'f(root) = {f(root):.3e}')


#%% Convergence plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(roots)), roots, '-o')

# Show where the algorithm switches methods
ax.axvline(
    switch_iteration,
    linestyle='--',
    label='Bisection → Newton'
)

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Root approximation $x_n$')
ax.set_title('Hybrid Bisection-Newton Convergence Profile')

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

ax.legend(frameon=False)

plt.show()