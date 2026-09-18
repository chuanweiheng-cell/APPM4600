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

#%% 3: Constructing a robust and rapidly convergent root finder

f = lambda x: jnp.exp(x ** 2 + 7 * x - 30) - 1
x = np.linspace(0, 3.1, 1000)

plt.plot(x, f(x), label=r'$f(x)=e^{x^2+7x-30}-1$')
plt.hlines(0, 0, 4, color='k', label='$f(x)=0$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(frameon=False)
plt.savefig('plotting_the_function')
# plt.show()

# 1st derivative
dfdx = jax.grad(f)

# 2nd derivative
d2fdx2 = jax.grad(dfdx)

#Global tolerance variable
tol = 1e-10


#%% 6 Consider the function f (x) = ex2+7x−30 −1. x = 3 is a root of this function. Apply the methods below with the specified initial data

# Bisection:
L_boundary, R_boundary = 2, 4.5      # initial guesses
count = 0
roots = []

while True:

    mid_point = (L_boundary + R_boundary) / 2

    func_L = f(L_boundary)
    func_R = f(R_boundary)
    func_mid = f(mid_point)

    if func_mid * func_L < 0:
        R_boundary = mid_point
    else:
        L_boundary = mid_point

    count += 1

    err = jnp.abs(L_boundary - R_boundary)

    roots.append(mid_point)

    if err < tol:
        root = (L_boundary + R_boundary) / 2
        break

roots = jnp.array(roots)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(roots)), roots, '-o')

ax.set_ylabel('Root')
ax.set_xlabel('Iteration')
ax.set_title('Lab 5, Bisection method convergence profile')

ax.text(
    0.55, 0.4,
    rf'$x_{{\mathrm{{root}}}}\approx {root:.2f}$',
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

plt.savefig('bisection')
# plt.show()

print(f'root from bisection = {root}')
print(f'iterations from bisection with tol={tol}: {count}')

# Newton's method

x0 = 4.5

count = 0
roots = []

while True:

    x1 = x0 - f(x0) / dfdx(x0)

    roots.append(x1)

    if jnp.abs(x1 - x0) <= tol:
        root = x1
        break
    
    count += 1
    x0 = x1

roots = jnp.array(roots)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(roots)), roots, '-o')

ax.set_ylabel('Root')
ax.set_xlabel('Iteration')
ax.set_title("Lab 5, Newton's method convergence profile")

ax.text(
    0.55, 0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root:.2f}$',
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
    0.55, 0.52,
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

plt.savefig('Newton')
# plt.show()

print(f'root from Newton method = {root}')
print(f'iterations from Newton method with tol={tol}: {count}')    

#Hybrid Bisection ---> if x_k in basin --switch--> Newtons
L_boundary, R_boundary = 2, 4.5 # initial guesses

count = 0
roots = []
while True:
    
    mid_point = (L_boundary + R_boundary) / 2
    
    condition = f(mid_point) * d2fdx2(mid_point) / (dfdx(mid_point)) ** 2
    if jnp.abs(condition) < 1:
        break
    func_L = f(L_boundary)
    func_R = f(R_boundary)
    func_mid = f(mid_point)

    if func_mid * func_L < 0:
        R_boundary = mid_point
    else:
        L_boundary = mid_point

    count += 1

    err = jnp.abs(L_boundary - R_boundary)

    roots.append(mid_point)
    
x0 = mid_point
    
while True:

    x1 = x0 - f(x0) / dfdx(x0)

    roots.append(x1)

    if jnp.abs(x1 - x0) <= tol:
        root = x1
        break
    
    count += 1
    x0 = x1

roots = jnp.array(roots)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(roots)), roots, '-o')

ax.set_ylabel('Root')
ax.set_xlabel('Iteration')
ax.set_title("Lab 5, Hybrid method convergence profile")

ax.text(
    0.55, 0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root:.2f}$',
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
    0.55, 0.52,
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

plt.savefig('hybrid')
# plt.show()

print(f'root from Hybrid method = {root}')
print(f'iterations from Hybrid method with tol={tol}: {count}')    