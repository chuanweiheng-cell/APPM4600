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


#%% n-dimensional slacker Newton's method

def slacker_newtons_method(
    F,
    p_guess,
    tol=1e-10,
    tol_slacker=1e-2,
    max_iter=100
):

    # Convert initial guess to JAX array
    p_guess = jnp.array(
        p_guess,
        dtype=float
    )

    # Automatically construct the Jacobian function
    jacobian_F = jax.jacfwd(F)

    # Evaluate the Jacobian once at the initial guess:
    #
    #     J_slacker = J_F(p_0)
    #
    # This Jacobian is reused until the refresh condition is met.
    J_slacker = jacobian_F(p_guess)

    # Store initial guess
    history = [p_guess]

    count = 0

    while count < max_iter:

        # Evaluate nonlinear system at current iterate:
        #
        #     F(p_n)
        F_val = jnp.array(
            F(p_guess),
            dtype=float
        )

        # Slacker Newton step:
        #
        #     J_slacker delta_p = F(p_n)
        #
        # where J_slacker may have been computed at an earlier iterate.
        delta_p = jnp.linalg.solve(
            J_slacker,
            F_val
        )

        # Update:
        #
        #     p_(n+1) = p_n - delta_p
        p_new = p_guess - delta_p

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

            root = p_new
            break

        # Recompute the Jacobian when the step becomes sufficiently small.
        #
        # The new Jacobian is evaluated at p_(n+1), since this is the
        # iterate that will be used in the next step.
        if err < tol_slacker:

            J_slacker = jacobian_F(p_new)

        # Update approximation
        p_guess = p_new

    else:

        raise RuntimeError(
            f'Slacker Newton method did not converge within '
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
    tol_slacker = 1e-2

    root, count, history = slacker_newtons_method(
        F,
        p_guess,
        tol=tol,
        tol_slacker=tol_slacker
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
    plt.title("Slacker Newton's Method Convergence")

    plt.grid(
        True,
        which='both',
        alpha=0.3
    )

    plt.show()