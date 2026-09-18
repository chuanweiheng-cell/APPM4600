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


#%% Bisection method

def bisection_method(f, L_boundary, R_boundary, tol=1e-10, max_iter=1000):
    """
    Approximate a root of f(x) using the bisection method.

    Parameters
    ----------
    f : function
        Function whose root is being approximated.

    L_boundary : float
        Left boundary of the initial interval.

    R_boundary : float
        Right boundary of the initial interval.

    tol : float
        Desired absolute error tolerance.

    max_iter : int
        Maximum number of iterations allowed.

    Returns
    -------
    root : float
        Approximation of the root.

    count : int
        Number of bisection iterations performed.

    roots : ndarray
        Midpoint approximation from every iteration.
    """

    # Evaluate function at initial boundaries
    func_L = f(L_boundary)
    func_R = f(R_boundary)

    # Bisection requires the root to be bracketed
    if func_L * func_R > 0:
        raise ValueError(
            'Initial interval does not bracket a root: '
            'f(L_boundary) and f(R_boundary) must have opposite signs.'
        )

    # Check whether either boundary is already a root
    if func_L == 0:
        return L_boundary, 0, np.array([L_boundary])

    if func_R == 0:
        return R_boundary, 0, np.array([R_boundary])

    count = 0
    roots = []

    while count < max_iter:

        # Midpoint of current interval
        mid_point = (L_boundary + R_boundary) / 2

        # Evaluate function at midpoint
        func_mid = f(mid_point)

        # Store midpoint to examine convergence later
        roots.append(mid_point)

        # Determine which half still brackets the root
        if func_L * func_mid < 0:

            R_boundary = mid_point

        else:

            L_boundary = mid_point
            func_L = func_mid

        count += 1

        # Maximum possible absolute error in midpoint approximation
        err = np.abs(R_boundary - L_boundary) / 2

        # Stop once desired accuracy has been reached
        if err < tol:

            root = (L_boundary + R_boundary) / 2
            break

    else:

        raise RuntimeError(
            f'Bisection method did not converge within {max_iter} iterations.'
        )

    roots = np.array(roots)

    return root, count, roots


#%% Example problem

# Solve:
#
#     exp(x^2 + 7x - 30) - 1 = 0
#
# on the interval [2, 4.5]

f = lambda x: np.exp(x ** 2 + 7 * x - 30) - 1

L_boundary = 2.0
R_boundary = 4.5

tol = 1e-10

root, count, roots = bisection_method(
    f,
    L_boundary,
    R_boundary,
    tol=tol
)

print(f'Root from bisection = {root}')
print(f'Iterations from bisection with tol={tol}: {count}')
print(f'f(root) = {f(root):.3e}')


#%% Convergence plot

iterations = np.arange(1, roots.size + 1)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(iterations, roots, '-o')

ax.set_xlabel('Iteration')
ax.set_ylabel('Root approximation')
ax.set_title('Bisection Method Convergence Profile')

ax.text(
    0.55, 0.4,
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
    0.55, 0.3,
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