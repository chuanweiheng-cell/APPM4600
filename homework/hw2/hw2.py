#%% imports
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

#%% q2

from scipy.linalg import inv
from scipy.linalg import norm

A = 0.5 * jnp.array([[1,          1],
                     [1 + 1e-10, 1 - 1e-10]])

b = jnp.array([1.0, 1.0])

A_inv = inv(A)

x = A_inv @ b


# part b

A_2_norm = norm(A, ord=2)
A_inv_2_norm = norm(A_inv, ord=2)

cond_num = A_2_norm * A_inv_2_norm

print(f'condition number = {cond_num}')


# part c

delta_b = jnp.array([1e-5, -1e-5])

b_perturbed = b + delta_b

x_perturbed = A_inv @ b_perturbed

delta_x = x_perturbed - x

rel_err = norm(delta_x, ord=2) / norm(x, ord=2)

print(f'relative error = {rel_err}')


#%% q3

# part c

def f(x):
    y = jnp.exp(x)
    return y - 1

x = 9.999999995000000e-10

print(f(x))

# part d

def taylor_f(x):
    return x + x ** 2 / 2 
    
print(taylor_f(x))


#%% q4

# bisection

def func(x):
    return  2 * x - 1 - jnp.sin(x)

tol = 1e-10
count = 0
L_boundary = 3
R_boundary = -3

roots = []

while True:
    
    mid_point = (L_boundary + R_boundary) / 2
    
    func_L = func(L_boundary)
    func_R = func(R_boundary)
    func_mid = func(mid_point)

    if func_mid * func_L < 0:
        R_boundary = mid_point
    else:
        L_boundary = mid_point
    
    count += 1
    err = jnp.abs(L_boundary - R_boundary)
    
    roots.append(mid_point)
    
    if err < tol:
        root = mid_point
        break

fig, ax = plt.subplots(figsize=(8, 5))

plt.plot(range(len(roots)), roots, 'o-', label='$f(x)=2x-1-sin(x)$')
plt.xlabel('Iterations')
plt.ylabel('root')
plt.title('Bisection method')
plt.legend(frameon=False)

ax.text(
    0.35, 0.35,
    f'Root={root:.5f}, 5 significant figures',
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
    0.35, 0.22,
    f'Number of iterations={count}',
    transform=ax.transAxes,
    ha='left',
    va='top',
    bbox={
        'facecolor': 'yellow',
        'alpha': 0.4,
        'pad': 5
    }
)

# plt.savefig('q4c.png')
plt.show()    

print(f'root={root}, iterations={count}')

#%% q5

def func(x):
    return x ** 3 + x -4

tol = 1e-3
count = 0
L_boundary = 3
R_boundary = -3

roots = []

while True:
    
    mid_point = (L_boundary + R_boundary) / 2
    
    func_L = func(L_boundary)
    func_R = func(R_boundary)
    func_mid = func(mid_point)

    if func_mid * func_L < 0:
        R_boundary = mid_point
    else:
        L_boundary = mid_point
    
    count += 1
    err = jnp.abs(L_boundary - R_boundary)
    
    roots.append(mid_point)
    
    if err < tol:
        root = mid_point
        break

print(f'root from bisection={root}')
print(f'iterations from bisction with tol=1e-3: {count}')
print(f'theoretical upper bound for tol=1e-3: {(jnp.log(1e-3) - jnp.log(3)) / jnp.log(1/2)}')

#%% q6

# part a

def f(x):
    return x - 4 * jnp.sin(2 * x) - 3

x = jnp.linspace(-1, 2.35 * jnp.pi, 1000)
# x = jnp.linspace(6.5, 7.5, 1000)
plt.plot(x, f(x))
plt.hlines(0, -0.5 * jnp.pi, 2.5 * jnp.pi, 'k')
# plt.hlines(0, 6, 8, 'k')

plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.title('$f(x)=x-4sin(2x)-3$')

# plt.savefig('q6a.png')
plt.show()

# part b

def g(x):
    return -jnp.sin(2 * x) + 5 * x / 4 - 3 / 4

tol = 1e-10
x_guesses = [1, 2, 3, 4, 5]

for i, x_guess in enumerate(x_guesses):

    x_initial = x_guess
    count = 0

    while True:

        x_new = g(x_guess)

        err = jnp.abs(x_new - x_guess)

        if err < tol:
            break

        x_guess = x_new
        count += 1

    print(f'iteration number {i + 1}')
    print(f'For initial guess x = {x_initial}')
    print(f'root = {x_new}')
    print(f'iterations = {count}')
    print(f'error estimate = {err}')
    print()