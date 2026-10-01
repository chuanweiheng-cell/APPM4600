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

#%% q1

# part a

def fixed_point_iteration(F, p_guess, tol=1e-10, max_iter=1000):
 
    count = 0
    
    f, g = F

    # Convert initial guess to a JAX array
    p_guess = jnp.array(p_guess, dtype=float)

    # Store initial guess
    history = [p_guess]
    
    J = jnp.array([
        [1/6, 1/18],
        [0  , 1/6 ]
    ])

    while count < max_iter:

        x_guess, y_guess = p_guess
        
        vec_new = jnp.array([x_guess, y_guess]) - J @ jnp.array([f(x_guess, y_guess), g(x_guess, y_guess)])

        x_new = vec_new[0]
        y_new = vec_new[1]
        
        p_new = jnp.array([x_new, y_new])

        # Error estimate between successive approximations
        err = jnp.linalg.norm(p_new - p_guess, ord=2)

        # Store new approximation
        history.append(p_new)

        count += 1

        # Check convergence
        if err < tol:

            root = p_new
            break

        # Update approximation for next iteration
        p_guess = p_new

    else:

        raise RuntimeError(
            f'Fixed-point iteration did not converge within {max_iter} iterations.'
        )

    history = jnp.array(history)

    return root, count, history

f = lambda x, y: 3 * x ** 2 - y ** 2
g = lambda x, y: 3 * x * y ** 2 - x ** 3 - 1

F = (f, g)

tol = 1e-8

p_guess = (1, 1)

root, count, history = fixed_point_iteration(F, p_guess, tol)

print(f'Root from fixed-point iteration = {root}')
print(f'Iterations with tol={tol}: {count}')

#Convergence plot

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))

plt.semilogy(iterations, errors[:-1], '-o')

plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^*\|_2$')
plt.title('Fixed-Point Iteration Convergence')

plt.grid(True, which='both', alpha=0.3)

plt.show()

#%% part b

#%% part b

def F_vec(p):

    x, y = p

    return jnp.array([
        f(x, y),
        g(x, y)
    ])


def newtons_method(F, p_guess, tol=1e-10, max_iter=100):

    # Convert initial guess to JAX array
    p_guess = jnp.array(
        p_guess,
        dtype=float
    )

    # Construct Jacobian automatically
    jacobian_F = jax.jacfwd(F)

    history = [p_guess]

    count = 0

    while count < max_iter:

        # Evaluate F(p_n)
        F_val = F(p_guess)

        # Evaluate Jacobian J_F(p_n)
        J = jacobian_F(p_guess)

        # Solve:
        #
        #     J delta_p = F
        delta_p = jnp.linalg.solve(
            J,
            F_val
        )

        # Newton update:
        #
        #     p_(n+1) = p_n - delta_p
        p_new = p_guess - delta_p

        # Error estimate
        err = jnp.linalg.norm(
            p_new - p_guess,
            ord=2
        )

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


root, count, history = newtons_method(
    F_vec,
    p_guess,
    tol
)

print(f'Root from Newton method = {root}')
print(f'Iterations with tol={tol}: {count}')

#Convergence plot

# Error relative to the converged root
errors = jnp.linalg.norm(history - root, axis=1)

# Exclude the final point since its error is exactly zero
iterations = jnp.arange(len(history) - 1)

plt.figure(figsize=(8, 5))

plt.semilogy(iterations, errors[:-1], '-o')

plt.xlabel('Iteration')
plt.ylabel(r'$\|\vec{p}_n-\vec{p}^*\|_2$')
plt.title("Newton's method Convergence")

plt.grid(True, which='both', alpha=0.3)

plt.show()
