#%% imports
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

#%% q1

# part (i)
def f1(x):
    return (x - 2) ** 9

x = np.arange(1.92, 2.08, 0.001)
y1 = f1(x)

plt.plot(x, y1, label='$f_{1}(x) = (x - 2)^9$')

# part (ii)
def f2(x):
    return x ** 9 - 18 * x ** 8 + 144 * x ** 7 - 672 * x ** 6 + 2016 * x ** 5 - 4032 * x ** 4 + 5376 * x ** 3 - 4608 * x ** 2 + 2304 * x - 512

y2 = f2(x)
plt.plot(x, y2, label='$f_{2}(x) = x^9 - ... - 512$')
plt.legend()
plt.show()

#%% q5

# part (b)
def f1(x, delta):
    return np.cos(x + delta) - np.cos(x)

def f2(x, delta):
    return -2 * np.sin(x + delta / 2) * np.sin(delta / 2)

x_1 = np.pi
x_2 = 1e6

deltas = np.logspace(-16, 0, 100)
y_1 = f1(x_1, deltas) - f2(x_1, deltas)
y_2 = f1(x_2, deltas) - f2(x_2, deltas)
plt.figure(figsize=(8, 6))
plt.semilogx(deltas, y_1, label='$x = \pi$')
plt.semilogx(deltas, y_2, label=f'$x = {x_2:.1e}$')
plt.xlabel('$\delta$')
plt.ylabel('Difference between $f_1$ and $f_2$')
plt.legend(frameon=False)
plt.show()

# part (c)
def f3(x, delta):
    return -delta * np.sin(x) 

y_3 = f2(x_1, deltas) - f3(x_1, deltas)
y_4 = f2(x_2, deltas) - f3(x_2, deltas)
plt.figure(figsize=(8, 6))
plt.semilogx(deltas, y_3, label='$x = \pi$')
plt.semilogx(deltas, y_4, label=f'$x = {x_2:.1e}$')
plt.xlabel('$\delta$')
plt.ylabel('Difference between $f_2$ and $f_3$')
plt.legend(frameon=False)
plt.show()