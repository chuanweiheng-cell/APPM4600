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


#%% 2.2 Excercises

# 1 
def alpha(a, b, c): 
    return jnp.log(jnp.abs(a / b)) / jnp.log(jnp.abs(b / c))

print('Fixed point iteration')

# 2a
g = lambda x : jnp.sqrt(10 / (x + 4))

p_star = 1.3652300134140976

p_guess = 1.5
tol = 1e-10
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

plt.plot(range(count + 1), history, '-o')
plt.xlabel('Iteration')
plt.ylabel('g(x)')
plt.show()

history = jnp.array(history)

#2b
    
a = history[-1] - p_star
b = history[-2] - p_star
c = history[-3] - p_star

print(f'fixed point number of iterations: {count}')
print(f'fixed point iteration order of convergence = {alpha(a, b, c)}')


#%% 3 Lab day: Exploring order of convergence and creating higher order approximations out of low order approximations

# 3.2 Exercises

#%% Aitken's delta squared acceleration

print("\nAitken's delta squared acceleration technique")

p_n = history[:-2]
p_n1 = history[1:-1]
p_n2 = history[2:]

history_aitkens = (
    p_n
    - (p_n1 - p_n) ** 2
    / (p_n2 - 2 * p_n1 + p_n)
)

plt.plot(range(len(history) - 2), history_aitkens, '-o')
plt.xlabel('Iteration')
plt.ylabel(r"Atkin's $\Delta^2$")
plt.show()

a = history_aitkens[-1] - p_star
b = history_aitkens[-2] - p_star
c = history_aitkens[-3] - p_star

print(f"Aitken's number of iterations: {count}")
print(f"Aitken's order of convergence = {alpha(a, b, c)}")


# 3.4 Exercises 

print("\nSteffenson’s method")

p_guess = 1.5
history_steffensons = []
count = 0

# while True:
#     a = p_guess
#     b = g(p_guess)
#     c = g(b)
    
#     p_steffensons = a - (b - a) ** 2 / (c - 2 * b + a)
    
#     err = jnp.abs
    
#     if    
    
