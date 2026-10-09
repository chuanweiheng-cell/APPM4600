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


#%% n-dimensional Broyden's method (Sherman-Morrison)

def broydens_method(F, p_guess, tol=1e-10, max_iter=1000):

    # Convert initial guess to JAX array
    p_guess = jnp.array(
        p_guess,
        dtype=float
    )

    # Construct the initial Jacobian only once
    J = jax.jacfwd(F)(p_guess)

    # Initial inverse Jacobian
    H = jnp.linalg.solve(
        J,
        jnp.eye(p_guess.size)
    )

    # Evaluate nonlinear system
    F_val = F(p_guess)

    # Store initial guess
    history = [p_guess]

    count = 0

    while count < max_iter:

        # Broyden step:
        #
        #     s_n = -H_n F(p_n)
        s = -H @ F_val

        # Update approximation
        p_new = p_guess + s

        # Evaluate nonlinear system at new approximation
        F_new = F(p_new)

        # Error estimate between successive approximations
        err = jnp.linalg.norm(
            p_new - p_guess,
            ord=2
        )

        # Store new approximation
        history.append(p_new)

        count += 1

        # Check for divergence
        if not jnp.all(jnp.isfinite(p_new)) or not jnp.all(jnp.isfinite(F_new)):
            raise RuntimeError(
                'Broyden iteration produced non-finite values.'
            )

        # Check convergence
        if err < tol and jnp.linalg.norm(F_new, ord=2) < tol:

            root = p_new
            break

        # Difference in function evaluations:
        #
        #     y_n = F(p_(n+1)) - F(p_n)
        y = F_new - F_val

        # Intermediate products
        H_y = H @ y

        denominator = s @ H_y

        # Check Sherman-Morrison denominator
        if jnp.abs(denominator) <= (
            1e-12 * jnp.linalg.norm(s) * jnp.linalg.norm(H_y)
        ):
            raise RuntimeError(
                'Sherman-Morrison denominator is too small.'
            )

        # Sherman-Morrison inverse Jacobian update:
        #
        #                  (s - H y)(s^T H)
        #     H_new = H + ------------------
        #                       s^T H y
        H = H + jnp.outer(
            s - H_y,
            s @ H
        ) / denominator

        # Update approximation and function evaluation
        p_guess = p_new
        F_val = F_new

    else:

        raise RuntimeError(
            f'Broyden method did not converge within '
            f'{max_iter} iterations.'
        )

    history = jnp.array(history)

    return root, count, history


#%% example

if __name__ == '__main__':

    def F(p):

        x, y = p

        return jnp.array([
            3.0 * x ** 3 - y ** 2,
            3.0 * x * y ** 2 - jnp.sin(x) ** 3 - 1.0
        ])


    p_guess = jnp.array([
        1.0,
        1.0
    ])

    tol = 1e-8

    root, count, history = broydens_method(
        F,
        p_guess,
        tol=tol
    )

    print(f'Root = {root}')
    print(f'Iterations with tol={tol}: {count}')


#%% convergence plot

if __name__ == '__main__':

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
    plt.title("Broyden's Method Convergence")

    plt.grid(
        True,
        which='both',
        alpha=0.3
    )

    plt.show()