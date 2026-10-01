#%% imports
import numpy as np
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

#%% defining n-dimensional Newton's method

def newtons_method(F, p_guess, tol=1e-10, max_iter=1000):

    # Convert initial guess to JAX array
    p_guess = jnp.array(
        p_guess,
        dtype=float
    )

    # Automatically construct the Jacobian
    jacobian_F = jax.jacfwd(F)

    # Store initial guess
    history = [p_guess]

    count = 0

    while count < max_iter:

        # Evaluate nonlinear system
        F_val = jnp.array(
            F(p_guess),
            dtype=float
        )

        # Evaluate Jacobian
        J = jacobian_F(p_guess)

        # Solve:
        #
        #     J_F(p_n) delta_p = F(p_n)
        delta_p = jnp.linalg.solve(
            J,
            F_val
        )

        # Newton update:
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

        # Update approximation
        p_guess = p_new

    else:

        raise RuntimeError(
            f'Newton method did not converge within '
            f'{max_iter} iterations.'
        )

    history = jnp.array(history)

    return root, count, history

#%% defining n-dimensional lazy Newton's method

def lazy_newtons_method(F, p_guess, tol=1e-10, max_iter=100):

    # Convert initial guess to JAX array
    p_guess = jnp.array(
        p_guess,
        dtype=float
    )
    
    # Automatically construct the Jacobian
    jacobian_F = jax.jacfwd(F)
    
    # Evaluate a single Jacobian for lazy Newton's
    J_lazy = jacobian_F(p_guess)

    # Store initial guess
    history = [p_guess]

    count = 0

    while count < max_iter:

        # Evaluate nonlinear system
        F_val = jnp.array(
            F(p_guess),
            dtype=float
        )


        # Solve:
        #
        #     J_F(p_n) delta_p = F(p_n)
        delta_p = jnp.linalg.solve(
            J_lazy,
            F_val
        )

        # Newton update:
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

        # Update approximation
        p_guess = p_new

    else:

        raise RuntimeError(
            f'Newton method did not converge within '
            f'{max_iter} iterations.'
        )

    history = jnp.array(history)

    return root, count, history


#%% pre-lab

def F(p):
    x1, x2 = p
    
    return jnp.array(
        [x1 ** 2 + x2 ** 2 - 2,
        jnp.exp(x1 - 1) + x2 ** 2 - 2]
        )
    
# Newton's for x0 = (2.0, 0.5)    

p_guess = jnp.array([2.0, 0.5])

tol = 1e-6

root, count, history = newtons_method(
    F,
    p_guess,
    tol=tol
)

print(f"Root for Newton's = {root}")
print(f'Iterations with tol={tol}: {count}')

print()
# Lazy Newton's for x0 = (3.0, 5.0)    

p_guess = jnp.array([3.0, 5.0])

root, count, history = lazy_newtons_method(
    F,
    p_guess,
    tol=tol
)

print(f"Root for lazy Newton's = {root}")
print(f'Iterations with tol={tol}: {count}')

#%% 3.2 Exercises: Build Slacker Newton

