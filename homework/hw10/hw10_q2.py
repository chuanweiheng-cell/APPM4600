#%% Composite quadrature versus SciPy's adaptive quadrature
# Problem: integrate f(s) = 1 / (1 + s^2) over [-5, 5].
# Choose panel counts from textbook worst-case error estimates,
# then compare computed values and function-evaluation counts.

import math
import numpy as np
from scipy.integrate import quad


def f(s):
    return 1.0 / (1.0 + s**2)


def composite_trapezoidal(a, b, n):
    # n subintervals create n+1 equally spaced function values.
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    value = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return value, len(y)


def composite_simpson(a, b, n):
    # Simpson's rule uses pairs of subintervals, so n must be even.
    if n < 2 or n % 2 != 0:
        raise ValueError('Simpson\'s rule requires a positive even n >= 2.')
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    value = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return value, len(y)


def scipy_quad_with_tolerance(a, b, tol):
    # Set both absolute and relative tolerances explicitly.
    value, estimated_error, info = quad(f, a, b, epsabs=tol, epsrel=tol, full_output=1)
    return value, info['neval'], estimated_error


#%% Error-bound parameters

a = -5.0
b = 5.0
tol = 1.0e-4
exact = 2.0 * math.atan(5.0)

# Maximum absolute derivatives on [-5, 5]:
# f''(s)   = (6s^2 - 2) / (1 + s^2)^3, with maximum |f''| = 2.
# f''''(s) = 24(5s^4 - 10s^2 + 1) / (1 + s^2)^5,
#            with maximum |f''''| = 24.
M2 = 2.0
M4 = 24.0

trap_constant = (b - a)**3 * M2 / 12.0
simp_constant = (b - a)**5 * M4 / 180.0

# Require strict error < tol, so the bounds must be strictly below tol.
n_trap = math.floor(math.sqrt(trap_constant / tol)) + 1
n_simp = 2
while simp_constant / n_simp**4 >= tol:
    n_simp += 2

trap_bound = trap_constant / n_trap**2
simp_bound = simp_constant / n_simp**4


#%% Execute the composite rules and adaptive quadrature

trap_value, trap_neval = composite_trapezoidal(a, b, n_trap)
simp_value, simp_neval = composite_simpson(a, b, n_simp)
quad_1e6_value, quad_1e6_neval, quad_1e6_est = scipy_quad_with_tolerance(a, b, 1.0e-6)
quad_1e4_value, quad_1e4_neval, quad_1e4_est = scipy_quad_with_tolerance(a, b, 1.0e-4)

# SciPy's actual defaults (typically epsabs=epsrel=1.49e-8),
# which differ from the 1e-6 'default' mentioned in the assignment.
quad_default_value, quad_default_est, quad_default_info = quad(f, a, b, full_output=1)

print(f'Exact integral = 2 atan(5) = {exact:.15f}')
print(f'Target absolute error: < {tol:.1e}')
print(f'Trapezoidal: n = {n_trap}, theoretical bound = {trap_bound:.9e}')
print(f'Simpson:     n = {n_simp}, theoretical bound = {simp_bound:.9e}')
print()
print(f'{"Method":<25} {"Integral":>19} {"Absolute error":>17} {"Function values":>16}')
print('-' * 82)
rows = [
    ('Composite trapezoidal', trap_value, trap_neval),
    ('Composite Simpson', simp_value, simp_neval),
    ('quad, tolerance 1e-6', quad_1e6_value, quad_1e6_neval),
    ('quad, tolerance 1e-4', quad_1e4_value, quad_1e4_neval),
    ('quad, SciPy defaults', quad_default_value, quad_default_info['neval']),
]
for label, value, neval in rows:
    print(f'{label:<25} {value:>19.15f} {abs(value - exact):>17.9e} {neval:>16d}')

print('\nSciPy quad estimated absolute errors:')
print(f'  Requested 1e-6: {quad_1e6_est:.9e}')
print(f'  Requested 1e-4: {quad_1e4_est:.9e}')
print(f'  SciPy defaults: {quad_default_est:.9e}')
