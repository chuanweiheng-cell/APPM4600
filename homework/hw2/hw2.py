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

plt.savefig('q4c.png')
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
print(f'theoratical upper bound for tol=1e-3: {(jnp.log(1e-3) - jnp.log(3)) / jnp.log(1/2)}')
