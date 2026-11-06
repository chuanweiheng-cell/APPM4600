#%% APPM 4600 — Lab 11: Adaptive Quadrature
'''
Compare three adaptive integration schemes with five nodes per evaluated
subinterval for the integral of sin(1/x) on [0.1, 2.0].

The instructor's local acceptance test is
    |Q_left + Q_right - Q_parent| <= tol.

Print the integral approximation, its actual absolute error relative to an
accurate SciPy reference, and the number of accepted adaptive intervals.
Figures are saved into lab11_tex/figure/ for the LaTeX write-up.

Run from the lab11 folder:
    python lab11.py

Keep adaptive_quad.py and gauss_legendre.py beside this file.
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from adaptive_quad import (
    adaptive_quad,
    eval_composite_trap,
    eval_composite_simpsons,
    eval_gauss_quad
)

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150
})


#%% Lab 11 parameters and reference integral

a, b = 0.1, 2.0
M = 5                            # Five quadrature nodes PER subinterval
tol = 1e-3                     # Local absolute acceptance threshold

f = lambda x: np.sin(1.0 / x)
I_true, reference_error = quad(f, a, b, epsabs=1e-12, epsrel=1e-12)

METHODS = {
    'Composite trapezoidal': eval_composite_trap,
    'Composite Simpson': eval_composite_simpsons,
    'Gauss-Legendre': eval_gauss_quad
}

FIGURE_DIR = Path(__file__).resolve().parent / 'lab11_tex' / 'figure'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False                # Set True to display figures interactively


#%% Exercise 1: Compare adaptive quadrature methods

results = {}

for name, method in METHODS.items():
    I, nodes, n_intervals, intervals = adaptive_quad(
        a, b, f, tol, M, method, return_intervals=True
    )

    results[name] = {
        'I': I,
        'error': abs(I - I_true),
        'n_intervals': n_intervals,
        'nodes': nodes,
        'intervals': intervals
    }

print('APPM 4600 — Lab 11: Adaptive Quadrature')
print(f'Integral: int_{a:g}^{b:g} sin(1/x) dx')
print(f'M = {M} nodes, tolerance = {tol:.1e}')
print(f'Reference integral (scipy.integrate.quad): {I_true:.13f}')
print()
print('{:<24} {:>18} {:>16} {:>11}'.format('Method', 'Integral', 'Abs. error', 'Intervals'))
print('-' * 74)

for name, data in results.items():
    I_value, error, count = data['I'], data['error'], data['n_intervals']
    print(f'{name:<24} {I_value:>18.12f} {error:>16.3e} {count:>11d}')

best_method = min(results, key=lambda name: results[name]['n_intervals'])
print()
print(f'Fewest adaptive intervals: {best_method}')
print('NOTE: tol controls the local comparison, not a rigorous global error bound.')


#%% Exercise 2: Plot adaptive interval partitions

x_dense = np.linspace(a, b, 3000)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

for ax, (name, data) in zip(axes, results.items()):
    ax.plot(x_dense, f(x_dense), color='#203c52', lw=1.8)
    bounds = np.r_[data['intervals'][:, 0], data['intervals'][-1, 1]]
    ax.vlines(bounds, ymin=-1.1, ymax=1.1, color='#b45309', alpha=0.6, lw=1.0)
    ax.set_ylabel(r'$f(x)$')
    ax.set_ylim(-1.1, 1.1)
    count = data['n_intervals']
    ax.set_title(f'{name}: {count} adaptive intervals', loc='left')
    ax.grid(alpha=0.18)

axes[-1].set_xlabel(r'$x$')
fig.tight_layout()
fig.savefig(FIGURE_DIR / 'adaptive_intervals.png', dpi=200, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)


#%% Exercise 2: Compare absolute errors with the target tolerance

fig, ax = plt.subplots(figsize=(9, 5))
names = list(results)
errors = [results[name]['error'] for name in names]
bars = ax.bar(names, errors, color=['#4c8095', '#64a69a', '#c28a37'], width=0.6)
ax.axhline(tol, color='#b91c1c', lw=1.5, ls='--', label=r'$10^{-3}$ target')
ax.set_yscale('log')
ax.set_ylabel('Absolute error')
ax.set_title('Accuracy of adaptive quadrature at five nodes per interval')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.22)
ax.set_ylim(1e-8, 1e-2)

for bar, value in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.3,
            f'{value:.2e}', ha='center', va='bottom', fontsize=11)

fig.tight_layout()
fig.savefig(FIGURE_DIR / 'absolute_errors.png', dpi=200, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

print(f'\nSaved figures in: {FIGURE_DIR}')
