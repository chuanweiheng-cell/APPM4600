#%% Imports

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})


#%% Fixed-point iteration

def fixed_point_iteration(g, p_guess, tol=1e-10, max_iter=1000):
    """
    Approximate a fixed point of g(x) using fixed-point iteration.

    Fixed-point iteration generates the sequence

        p_(n+1) = g(p_n)

    until two successive approximations are sufficiently close.

    Parameters
    ----------
    g : function
        Fixed-point mapping.

    p_guess : float
        Initial guess p_0.

    tol : float
        Desired absolute tolerance between successive iterates.

    max_iter : int
        Maximum number of iterations allowed.

    Returns
    -------
    root : float
        Approximation of the fixed point.

    count : int
        Number of fixed-point iterations performed.

    history : ndarray
        Fixed-point approximations from each iteration.
    """

    count = 0

    # Store initial guess
    history = [p_guess]

    while count < max_iter:

        # Fixed-point iteration:
        #
        #     p_(n+1) = g(p_n)

        p_new = g(p_guess)

        # Error estimate between successive approximations
        err = np.abs(p_new - p_guess)

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

    history = np.array(history)

    return root, count, history


#%% Convergence-order estimate

def alpha(error_next, error_current, error_previous):
    """
    Estimate the order of convergence alpha using three successive errors.

    For

        e_(n+1) ≈ C e_n^alpha,

    the estimated convergence order is

        alpha ≈ log(|e_(n+1) / e_n|)
                -----------------------
                log(|e_n / e_(n-1)|)
    """

    numerator = np.log(np.abs(error_next / error_current))
    denominator = np.log(np.abs(error_current / error_previous))

    return numerator / denominator


#%% Example problem

# Fixed-point mapping:
#
#     p = sqrt(10 / (p + 4))

g = lambda x: np.sqrt(10 / (x + 4))

# Known fixed point used only for error analysis
p_star = 1.3652300134140976

# Initial guess
p_guess = 1.5

tol = 1e-10

root, count, history = fixed_point_iteration(
    g,
    p_guess,
    tol=tol
)

print(f'Root from fixed-point iteration = {root}')
print(f'Iterations with tol={tol}: {count}')
print(f'|p - p_star| = {np.abs(root - p_star):.3e}')
print(f'|g(p) - p| = {np.abs(g(root) - root):.3e}')


#%% Convergence plot

iterations = np.arange(history.size)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(iterations, history, '-o')

ax.axhline(
    p_star,
    linestyle='--',
    label=r'$p^*$'
)

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Fixed-point approximation $p_n$')
ax.set_title('Fixed-Point Iteration Convergence Profile')

ax.legend(frameon=False)

plt.show()


#%% Error convergence

errors = np.abs(history - p_star)

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(iterations, errors, '-o')

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Absolute error $|p_n-p^*|$')
ax.set_title('Fixed-Point Iteration Error')

plt.show()


#%% Estimated order of convergence

if errors.size >= 4:

    alpha_estimate = alpha(
        errors[-1],
        errors[-2],
        errors[-3]
    )

    print(f'Estimated order of convergence = {alpha_estimate:.6f}')