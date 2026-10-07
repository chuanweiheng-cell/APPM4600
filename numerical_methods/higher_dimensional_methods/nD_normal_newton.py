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


#%% n-dimensional normal Newton iteration

def normal_newtons_method(f, p_guess, tol=1e-10, max_iter=100):

    # Convert initial guess to JAX array
    p_guess = jnp.array(
        p_guess,
        dtype=float
    )

    # Automatically construct the gradient:
    #
    #     grad(f)(p)
    grad_f = jax.grad(f)

    # Store initial guess
    history = [p_guess]

    count = 0

    while count < max_iter:

        # Evaluate:
        #
        #     f(p_n)
        f_val = f(p_guess)

        # Evaluate:
        #
        #     grad(f)(p_n)
        grad_val = grad_f(p_guess)

        # Compute:
        #
        #     ||grad(f)(p_n)||_2^2
        grad_norm_squared = jnp.dot(
            grad_val,
            grad_val
        )

        # Protect against division by zero
        if grad_norm_squared < 1e-14:

            raise RuntimeError(
                'Gradient magnitude is too small for the iteration.'
            )

        # Compute:
        #
        #              f(p_n)
        #     d_n = -----------------
        #           ||grad(f)(p_n)||^2
        d = f_val / grad_norm_squared

        # Iteration:
        #
        #     p_(n+1) = p_n - d_n grad(f)(p_n)
        p_new = p_guess - d * grad_val

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

        # Update approximation
        p_guess = p_new

    else:

        raise RuntimeError(
            f'Normal Newton iteration did not converge within '
            f'{max_iter} iterations.'
        )

    history = jnp.array(history)

    return root, count, history


#%% example

if __name__ == '__main__':

    # Example level-set equation:
    #
    #     f(x,y) = x^2 + y^2 - 1 = 0
    #
    # whose zero set is the unit circle.
    def f(p):

        x, y = p

        return (
            x ** 2
            +
            y ** 2
            -
            1.0
        )


    p_guess = jnp.array([
        2.0,
        0.5
    ])

    tol = 1e-8

    root, count, history = normal_newtons_method(
        f,
        p_guess,
        tol=tol
    )

    print(f'Root = {root}')
    print(f'f(root) = {f(root):.4e}')
    print(f'Iterations with tol={tol}: {count}')


#%% convergence plot

if __name__ == '__main__':

    # Error relative to the converged point
    errors = jnp.linalg.norm(
        history - root,
        axis=1
    )

    # Exclude final point since its error is exactly zero
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
    plt.title('Normal Newton Iteration Convergence')

    plt.grid(
        True,
        which='both',
        alpha=0.3
    )

    plt.show()