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