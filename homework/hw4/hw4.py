#%% imports
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
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

# (i) Newton’s method
def newton_method(f, dfdx, x0, tol=1e-10, max_iter=1000):

    count = 0

    # Store initial guess
    roots = [x0]

    while count < max_iter:

        # Evaluate derivative at current approximation
        derivative = dfdx(x0)

        # Newton's method requires a nonzero derivative
        if jnp.abs(derivative) < 1e-16:
            raise ZeroDivisionError(
                'Derivative is too close to zero for Newton iteration.'
            )

        # Newton update
        x1 = x0 - f(x0) / derivative

        # Store new approximation
        roots.append(x1)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x1 - x0)

        # Check convergence
        if err <= tol:

            root = x1
            break

        # Update approximation for next iteration
        x0 = x1

    else:

        raise RuntimeError(
            f'Newton method did not converge within {max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots

f = lambda x: (jnp.e ** x - 3 * x ** 2) ** 3

# Derivative using JAX automatic differentiation
dfdx = jax.grad(f)

# Initial guess
x0 = 3.5

tol = 1e-10

root, count, roots = newton_method(f, dfdx, x0, tol=tol)

print(f'Root from Newton method = {root}')
print(f'Iterations from Newton method with tol={tol}: {count}')
print(f'f(root) = {f(root):.3e}')

# Convergence plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(roots)), roots, '-o')

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Newton approximation $x_n$')
ax.set_title("Newton's Method Convergence Profile")

ax.text(
    0.55, 0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root:.6f}$',
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

# plt.savefig('q3_newton.png')
plt.show()

# (ii) the modified Newton’s method from class
def modified_newton_method_1(f, dfdx, x0, m, tol=1e-10, max_iter=1000):

    count = 0

    # Store initial guess
    roots = [x0]

    while count < max_iter:

        # Evaluate derivative at current approximation
        derivative = dfdx(x0)

        # Newton's method requires a nonzero derivative
        if jnp.abs(derivative) < 1e-14:

            raise ZeroDivisionError(
                'Derivative is too close to zero for Newton iteration.'
            )

        # Modified Newton update
        x1 = x0 - m * f(x0) / derivative

        # Store new approximation
        roots.append(x1)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x1 - x0)

        # Check whether new approximation is a root
        if jnp.abs(f(x1)) <= tol:

            root = x1

            break

        # Check whether successive approximations are sufficiently close
        if err <= tol:

            root = x1

            break

        # Update approximation for next iteration
        x0 = x1

    else:

        raise RuntimeError(
            f'Modified Newton method did not converge within '
            f'{max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots

# Known multiplicity
m = 3

# Tolerance
tol = 1e-10

root, count, roots = modified_newton_method_1(f, dfdx, x0, m, tol=tol, max_iter=10000)

print(f'Root from Modified Newton Method 1 = {root}')
print(f'Root multiplicity = {m}')
print(f'Iterations with tol={tol}: {count}')
print(f'f(root) = {f(root):.3e}')

#Convergence plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    range(len(roots)),
    roots,
    '-o'
)

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Modified Newton approximation $x_n$')
ax.set_title("Modified Newton's Method 1 Convergence Profile")

ax.text(
    0.55,
    0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root:.6f}$',
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
    0.55,
    0.52,
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

# plt.savefig('q3_newton_1.png')
plt.show()


# (iii) modified Newton’s method in Problem 1
def modified_newton_method_2(h, dhdx, dfdx, x0, tol=1e-10, derivative_tol=1e-14, max_iter=1000):

    count = 0

    # Store initial guess
    roots = [x0]

    while count < max_iter:

        # For a multiple root, f'(x) approaches zero.
        # Stop before evaluating h = f/f' at the singular point.
        if jnp.abs(dfdx(x0)) <= derivative_tol:

            root = x0

            break

        # Derivative of the modified function h
        derivative_modified = dhdx(x0)

        # Newton's method applied to h requires h'(x) != 0
        if jnp.abs(derivative_modified) <= derivative_tol:

            raise ZeroDivisionError(
                "Derivative of modified function h is too close to zero."
            )

        # Modified Newton update
        x1 = x0 - h(x0) / derivative_modified

        # Store new approximation
        roots.append(x1)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x1 - x0)

        # Check convergence
        if err <= tol:

            root = x1

            break

        # Update approximation
        x0 = x1

    else:

        raise RuntimeError(
            f'Modified Newton Method 2 did not converge within '
            f'{max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots

# Modified function:
h = lambda x: f(x) / dfdx(x)

# Derivative of modified function using JAX
dhdx = jax.grad(h)

# Tolerances
tol = 1e-10
derivative_tol = 1e-14

root, count, roots = modified_newton_method_2(
    h,
    dhdx,
    dfdx,
    x0,
    tol=tol,
    derivative_tol=derivative_tol,
    max_iter=10000
)

print(f'Root from Modified Newton Method 2 = {root}')
print(f'Iterations with tol={tol}: {count}')
print(f'f(root) = {f(root):.3e}')


# Convergence plot

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    range(len(roots)),
    roots,
    '-o'
)

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Modified Newton approximation $x_n$')
ax.set_title("Modified Newton's Method 2 Convergence Profile")

ax.text(
    0.55,
    0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root:.6f}$',
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
    0.55,
    0.52,
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

# plt.savefig('q3_newton_2.png')
plt.show()

#%% q4

def secant_method(f, x0, x1, tol=1e-10, derivative_tol=1e-14, max_iter=1000):

    count = 0

    # Store initial guesses
    roots = [x0, x1]
    
    # Store errors
    errs = []

    while count < max_iter:

        # Secant approximation to derivative
        derivative = (
            (f(x1) - f(x0))
            / (x1 - x0)
        )

        # Secant method requires a nonzero approximate derivative
        if jnp.abs(derivative) <= derivative_tol:

            raise ZeroDivisionError(
                'Secant slope is too close to zero.'
            )

        # Secant update
        x2 = x1 - f(x1) / derivative

        # Store new approximation
        roots.append(x2)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x2 - x1)
        
        # Store errors
        errs.append(err)

        # Check convergence
        if err <= tol:

            root = x2
        
            break

        # Update approximations
        x0 = x1
        x1 = x2

    else:

        raise RuntimeError(
            f'Secant method did not converge within '
            f'{max_iter} iterations.'
        )

    roots = jnp.array(roots)
    
    errs = jnp.array(errs)

    return root, count, roots, errs


def newton_method(f, dfdx, x0, tol=1e-10, max_iter=1000):

    count = 0

    # Store initial guess
    roots = [x0]
    
    # Store errors
    errs = []


    while count < max_iter:

        # Evaluate derivative at current approximation
        derivative = dfdx(x0)

        # Newton's method requires a nonzero derivative
        if jnp.abs(derivative) < 1e-14:
            raise ZeroDivisionError(
                'Derivative is too close to zero for Newton iteration.'
            )

        # Newton update
        x1 = x0 - f(x0) / derivative

        # Store new approximation
        roots.append(x1)

        count += 1

        # Error between successive approximations
        err = jnp.abs(x1 - x0)
        
        errs.append(err)        

        # Check convergence
        if err <= tol:

            root = x1
            break

        # Update approximation for next iteration
        x0 = x1

    else:

        raise RuntimeError(
            f'Newton method did not converge within {max_iter} iterations.'
        )

    roots = jnp.array(roots)

    return root, count, roots, errs


f = lambda x: x ** 6 - x - 1

# Initial guesses
x0 = 2.0
x1 = 1.0

# Tolerances
tol = 1e-10
derivative_tol = 1e-14

root_sec, count_sec, roots_sec, errs_sec = secant_method(
    f,
    x0,
    x1,
    tol=tol,
    derivative_tol=derivative_tol,
    max_iter=1000
)

dfdx = jax.grad(f)

root_newt, count_newt, roots_newt, errs_newt = newton_method(
    f,
    dfdx,
    x0,
    tol=tol
)

print(f'Root from Secant method = {root_sec}')
print(f'Iterations from Secant method with tol={tol}: {count_sec}')
print(f'f(root) = {f(root_sec):.3e}')

print(f'Root from Newton method = {root_newt}')
print(f'Iterations from Newton method with tol={tol}: {count_newt}')
print(f'f(root) = {f(root_newt):.3e}')

# Convergence plot

fig, ax = plt.subplots(2, 1, figsize=(8, 11), sharex=True)

ax[0].plot(
    range(len(roots_sec)),
    roots_sec,
    '-o'
)

ax[0].set_ylabel(r'Secant approximation $x_n$')
ax[0].set_title('Secant Method Convergence Profile')

ax[0].text(
    0.55,
    0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root_sec:.6f}$',
    transform=ax[0].transAxes,
    ha='left',
    va='top',
    bbox={
        'facecolor': 'yellow',
        'alpha': 0.4,
        'pad': 5
    }
)

ax[0].text(
    0.55,
    0.52,
    f'Number of iterations = {count_sec}',
    transform=ax[0].transAxes,
    ha='left',
    va='top',
    bbox={
        'facecolor': 'yellow',
        'alpha': 0.4,
        'pad': 5
    }
)

ax[1].plot(range(len(roots_newt)), roots_newt, '-o')

ax[1].set_xlabel('Iteration')
ax[1].set_ylabel(r'Newton approximation $x_n$')
ax[1].set_title("Newton's Method Convergence Profile")

ax[1].text(
    0.55, 0.65,
    rf'$x_{{\mathrm{{root}}}}\approx {root_newt:.6f}$',
    transform=ax[1].transAxes,
    ha='left',
    va='top',
    bbox={
        'facecolor': 'yellow',
        'alpha': 0.4,
        'pad': 5
    }
)

ax[1].text(
    0.55, 0.52,
    f'Number of iterations = {count_newt}',
    transform=ax[1].transAxes,
    ha='left',
    va='top',
    bbox={
        'facecolor': 'yellow',
        'alpha': 0.4,
        'pad': 5
    }
)

# plt.savefig('q4_convergence_plot.png')
plt.show()

# part a

alpha = 1.1347241384015195

errors_sec = jnp.abs(roots_sec - alpha)
errors_newt = jnp.abs(roots_newt - alpha)

print('\nSecant method errors:')

for idx, error in enumerate(errors_sec):

    print(f'Iteration {idx}: 'f'error = {error:.9e}')


print('\nNewton method errors:')

for idx, error in enumerate(errors_newt):

    print(f'Iteration {idx}: 'f'error = {error:.9e}')
        
# part b

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(errors_newt[:-1], errors_newt[1:], 'o-', label='Newton')

ax.loglog(errors_sec[:-1], errors_sec[1:], 's-', label='Secant')

ax.set_xlabel(r'$|x_k-\alpha|$')
ax.set_ylabel(r'$|x_{k+1}-\alpha|$')
ax.set_title('Convergence Order')
ax.legend(frameon=False)

# plt.savefig('q4_pb.png')
plt.show()