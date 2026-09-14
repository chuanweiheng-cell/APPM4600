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

#%% q1

# part b

g = lambda x: 2 * x / 3 + 1 / x ** 2

p_star = 3 ** (1 / 3)

p_guess = p_star + 5
tol = 1e-5
history = []
count = 0

while True:
    func = g(p_guess)
    err = jnp.abs(p_guess - func)
    history.append(p_guess)
    print()
    if err < tol:
        break
    if count > 9999:
        print('exceeded maximum number of interations')
        break
    p_guess = func
    count += 1

history = jnp.array(history)
    
a = history[-1] - p_star
b = history[-2] - p_star
c = history[-3] - p_star

print(a, b, c)

def alpha(a, b, c): 
    return jnp.log(jnp.abs(a / b)) / jnp.log(jnp.abs(b / c))

alpha = alpha(a, b, c)

print(f'fixed point number of iterations: {count}')
print(f'fixed point iteration order of convergence = {alpha}')

fig, ax = plt.subplots(figsize=(8, 5))

plt.plot(range(count + 1), history, '-o')
plt.xlabel('Iteration')
plt.ylabel('g(x)')
plt.title(r'fixed point iteration for $g(x)=\frac{{2x}}{{3}}+\frac{{1}}{{x^2}}$')

ax.text(
    0.5, 0.45,
    rf'$\alpha\approx${alpha:.2f}',
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
    0.5, 0.32,
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

# plt.savefig('q1pb')
plt.show()

# part c

g = lambda x: 12 / (1 + x)

p_star = 3

p_guess = p_star + 5
tol = 1e-5
history = []
count = 0

while True:
    func = g(p_guess)
    err = jnp.abs(p_guess - func)
    history.append(p_guess)
    print()
    if err < tol:
        break
    if count > 9999:
        print('exceeded maximum number of interations')
        break
    p_guess = func
    count += 1

history = jnp.array(history)
    
a = history[-1] - p_star
b = history[-2] - p_star
c = history[-3] - p_star

print(a, b, c)

def alpha(a, b, c): 
    return jnp.log(jnp.abs(a / b)) / jnp.log(jnp.abs(b / c))

alpha = alpha(a, b, c)

print(f'fixed point number of iterations: {count}')
print(f'fixed point iteration order of convergence = {alpha}')

fig, ax = plt.subplots(figsize=(8, 5))

plt.plot(range(count + 1), history, '-o')
plt.xlabel('Iteration')
plt.ylabel('g(x)')
plt.title(r'fixed point iteration for $g(x)=\frac{{12}}{{1+x}}$')

ax.text(
    0.5, 0.55,
    rf'$\alpha\approx${alpha:.2f}',
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
    0.5, 0.42,
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

# plt.savefig('q1pc')
plt.show()


#%% q2

T_i = 20.0                         # deg C
T_s = -15.0                        # deg C
delta_T = T_i - T_s                # deg C

alpha = 0.138e-6                   # m^2/s

t_star = 60.0                      # days
t_star = t_star * 24 * 60 * 60     # s

tol = 1e-13

# part a

from scipy.special import erf

f = lambda x: T_s + delta_T * erf(x / (2 * jnp.sqrt(alpha * t_star)))

L = 3.0                            # m

x_domain = jnp.linspace(0.0, L, 100)
f_vals = f(x_domain)

fig, ax = plt.subplots(figsize=(8, 5))

plt.plot(x_domain, f_vals)
plt.axhline(0, color='k')
plt.xlabel('Depth, x [m]')
plt.ylabel('f(x)')
plt.title(r'$f(x)=\Delta T*erf(x/2\sqrt{60\alpha})$')
# plt.savefig('q2pa.png')
plt.show()

# part b, bisection method

L_boundary, R_boundary = 0, L      # initial guesses

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
ax.set_title('Bisection method convergence profile')

ax.text(
    0.55, 0.55,
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
    0.55, 0.42,
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

print(f'root from bisection = {root}')
print(f'iterations from bisection with tol={tol}: {count}')

#%% q3

x_star = 7 ** (1 / 5)
tol = 1e-10

print(f'\nx_star={x_star}')

# part a

f = lambda x: x * (1 + (7 - x ** 5) / (x ** 2)) ** 3

print(f'\nq3 part a: f(x_star) = {f(x_star)} ~ x_star')
print(f'Thus, x_star = 7^(1/5) is a fixed point of f(x)')

dfdx = jax.grad(f)
print(f"|f'(x_star)|={jnp.abs(dfdx(x_star))}")
print('Therefore, f(x) will not converge for x_star = 7^(1/5)')

# part b

f = lambda x: x - (x ** 5 - 7) / (x ** 2)

print(f'\nq3 part b: f(x_star) = {f(x_star)} ~ x_star')
print(f'Thus, x_star = 7^(1/5) is a fixed point of f(x)')

dfdx = jax.grad(f)
print(f"|f'(x_star)|={jnp.abs(dfdx(x_star))}")
print('Therefore, f(x) will not converge for x_star = 7^(1/5)')

# part c

f = lambda x: x - (x ** 5 - 7) / (5 * x ** 4)

print(f'\nq3 part c: f(x_star) = {f(x_star)} ~ x_star')
print(f'Thus, x_star = 7^(1/5) is a fixed point of f(x)')

dfdx = jax.grad(f)
print(f"|f'(x_star)|={jnp.abs(dfdx(x_star))}")
print('Therefore, f(x) will converge for x_star = 7^(1/5)')

history = []
count = 0

x_guess = 1.0

while True:
    func = f(x_guess)
    err = jnp.abs(x_guess - func)
    history.append(x_guess)
    if err < tol:
        root = func
        break
    if count > 9999:
        print('exceeded maximum number of interations')
        break
    x_guess = func
    count += 1

plt.plot(range(count + 1), history, '-o')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Q3 part c:')

plt.show()

# part d

f = lambda x: x - (x ** 5 - 7) / 12

print(f'\nq3 part d: f(x_star) = {f(x_star)} ~ x_star')
print(f'Thus, x_star = 7^(1/5) is a fixed point of f(x)')

dfdx = jax.grad(f)
print(f"|f'(x_star)|={jnp.abs(dfdx(x_star))}")
print('Therefore, f(x) will converge for x_star = 7^(1/5)')

history = []
count = 0

x_guess = 1.0

while True:
    func = f(x_guess)
    err = jnp.abs(x_guess - func)
    history.append(x_guess)
    if err < tol:
        root = func
        break
    if count > 9999:
        print('exceeded maximum number of interations')
        break
    x_guess = func
    count += 1

plt.plot(range(count + 1), history, '-o')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Q3 part d')

plt.show()
