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

#%% Problem definition

a = 0.0
b = np.pi
tol = 1e-10

def f(x):
    return np.exp(x) * np.cos(x)

# Exact antiderivative: (e^x / 2) * (sin(x) + cos(x))
I_true = -(np.exp(np.pi) + 1.0) / 2.0


#%% Composite quadrature rules

def composite_trapezoid(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = (b - a) / N
    return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))


def composite_simpson(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = (b - a) / N
    return (h / 3.0) * (f(x[0]) + f(x[-1]) + 4.0 * np.sum(f(x[1:-1:2])) + 2.0 * np.sum(f(x[2:-1:2])))



#%% Double the number of panels until the target relative error is reached

def convergence_data(rule):

    N = 2
    iteration = 0
    max_iter = 25

    rel_error = np.inf

    N_values, h_values, abs_errors, rel_errors = [], [], [], []

    while rel_error > tol and iteration < max_iter:

        I_approx = rule(f, a, b, N)

        h = (b - a) / N
        abs_error = abs(I_approx - I_true)
        rel_error = abs_error / abs(I_true)

        N_values.append(N)
        h_values.append(h)
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)

        N *= 2
        iteration += 1

    if rel_error > tol:
        raise RuntimeError('Maximum iterations reached without convergence.')

    return (np.array(N_values), np.array(h_values), np.array(abs_errors), np.array(rel_errors))

N_trap, h_trap, err_trap, rel_trap = convergence_data(composite_trapezoid)
N_simp, h_simp, err_simp, rel_simp = convergence_data(composite_simpson)

# Estimate the slope of log(error) vs log(h) using the final five points.
slope_trap = np.polyfit(np.log(h_trap[-5:]), np.log(err_trap[-5:]), 1)[0]
slope_simp = np.polyfit(np.log(h_simp[-5:]), np.log(err_simp[-5:]), 1)[0]

print(f'Exact integral: {I_true:.15f}')
print(f'Trapezoidal: N = {N_trap[-1]}, h = {h_trap[-1]:.5e}, '
      f'relative error = {rel_trap[-1]:.3e}, slope = {slope_trap:.4f}')
print(f'Simpson:     N = {N_simp[-1]}, h = {h_simp[-1]:.5e}, '
      f'relative error = {rel_simp[-1]:.3e}, slope = {slope_simp:.4f}')


#%% Plot absolute error versus panel width h

fig, ax = plt.subplots(figsize=(9, 6))

ax.loglog(h_trap, err_trap, 'o-', lw=2, label=f'Trapezoidal (slope = {slope_trap:.2f})')

ax.loglog(h_simp, err_simp, 's-', lw=2, label=f'Simpson (slope = {slope_simp:.2f})')

ax.set_xlabel(r'Subinterval width $h$')
ax.set_ylabel('Absolute error')
ax.set_title('Convergence of composite quadrature rules')
ax.grid(True, which='both', alpha=0.3)
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
