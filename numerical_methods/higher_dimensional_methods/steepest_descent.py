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


#%% n-dimensional steepest descent

def steepest_descent(F, p_guess, tol=1e-10, max_iter=1000):

    # Convert initial guess to JAX array
    p_guess = jnp.asarray(
        p_guess,
        dtype=float
    )

    # Objective function:
    #
    #     Phi(p) = 1/2 ||F(p)||^2
    def objective(p):
        F_val = F(p)
        return 0.5 * jnp.dot(F_val, F_val)

    # Construct objective and its gradient
    objective_eval = jax.jit(objective)
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    # Store initial guess
    history = [p_guess]

    # Evaluate objective and gradient
    phi, grad_phi = value_and_grad(p_guess)

    # Check initial guess
    if not bool(jnp.isfinite(phi)) or not bool(jnp.all(jnp.isfinite(grad_phi))):
        raise ValueError(
            'Initial guess is outside the real domain, '
            'or gradient is undefined.'
        )

    # Check whether initial guess is already a root
    if float(jnp.sqrt(2.0 * phi)) < tol:
        return p_guess, 0, jnp.stack(history)

    for count in range(1, max_iter + 1):

        # Initial step size
        beta = 1.0

        # Backtracking line search
        for _ in range(50):

            # Steepest descent update:
            #
            #     p_new = p_n - beta * grad(Phi)
            p_new = p_guess - beta * grad_phi

            phi_new = objective_eval(p_new)

            # Armijo sufficient-decrease condition
            if (
                bool(jnp.isfinite(phi_new))
                and bool(
                    phi_new <= phi
                    - 1e-4 * beta * (grad_phi @ grad_phi)
                )
            ):
                break

            beta *= 0.5

        else:
            raise RuntimeError(
                'Line search failed to find a decreasing step.'
            )

        # Store new approximation
        history.append(p_new)

        # Check convergence using ||F(p_new)||
        if float(jnp.sqrt(2.0 * phi_new)) < tol:

            root = p_new
            break

        # Update approximation
        p_guess = p_new

        # Update objective and gradient
        phi, grad_phi = value_and_grad(p_guess)

        if not bool(jnp.isfinite(phi)) or not bool(jnp.all(jnp.isfinite(grad_phi))):
            raise RuntimeError(
                'Encountered an invalid function value or gradient.'
            )

    else:

        raise RuntimeError(
            f'Steepest descent did not converge within '
            f'{max_iter} iterations.'
        )

    return root, count, jnp.stack(history)


#%% example

def F(p):

    x, y, z = p

    return jnp.array([
        x + jnp.cos(x*y*z) - 1,
        (1 - x)**(1/4) + y + 0.05*z**2 - 0.15*z - 1,
        -x**2 - 0.1*y**2 + 0.01*y + z - 1
    ])


p_guess = jnp.array([
    0.5,
    -3.0,
    -1.0
])

tol = 1e-8

if __name__ == '__main__':

    root, count, history = steepest_descent(
        F,
        p_guess,
        tol=tol
    )

    print(f'Root = {root}')
    print(f'Iterations with tol={tol}: {count}')
    print(f'Residual norm = {float(jnp.linalg.norm(F(root))):.3e}')


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
    plt.title('Steepest Descent Convergence')

    plt.grid(
        True,
        which='both',
        alpha=0.3
    )

    plt.show()