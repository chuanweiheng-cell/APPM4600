'''
Adaptive quadrature using composite trapezoidal, composite Simpson,
or Gauss-Legendre rules on each subinterval.

Original adaptive method and Gauss-Legendre interface:
Copyright (C) 2025 Adrianna M. Gillman.

Adapted for APPM 4600 Lab 11 (2026): completed the composite rules,
added input validation and a depth guard, and made the reported adaptive
partition and evaluated nodes consistent with accepted subintervals.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
from gauss_legendre import lgwt


#%% Nonadaptive quadrature rules on one interval

def eval_composite_trap(M, a, b, f):
    '''Composite trapezoidal quadrature with M equally spaced NODES.'''
    if M < 2:
        raise ValueError('Composite trapezoidal quadrature needs M >= 2.')

    x = np.linspace(a, b, M)
    h = (b - a) / (M - 1)             # M points produce M-1 panels

    w = np.full(M, h)
    w[0] *= 0.5                      # Half weight at the left endpoint
    w[-1] *= 0.5                     # Half weight at the right endpoint

    I_hat = np.sum(w * f(x))
    return float(I_hat), x, w


def eval_composite_simpsons(M, a, b, f):
    '''Composite Simpson 1/3 quadrature using M equally spaced NODES.'''
    # Simpson requires an EVEN number of panels, hence an ODD number of nodes.
    if M < 3 or M % 2 == 0:
        raise ValueError('Composite Simpson quadrature needs odd M >= 3.')

    x = np.linspace(a, b, M)
    h = (b - a) / (M - 1)

    w = np.full(M, 2.0)
    w[1:-1:2] = 4.0                  # Odd-numbered interior nodes: 4
    w[0] = 1.0
    w[-1] = 1.0                     # Endpoints: 1
    w *= h / 3.0                    # Interior even nodes: 2

    I_hat = np.sum(w * f(x))
    return float(I_hat), x, w


def eval_gauss_quad(M, a, b, f):
    '''M-point Gauss-Legendre quadrature on [a, b].'''
    if M < 2:
        raise ValueError('Gauss-Legendre quadrature needs M >= 2 here.')

    x, w = lgwt(M, a, b)
    I_hat = np.sum(f(x) * w)
    return float(I_hat), x, w


#%% Adaptive interval bisection

def adaptive_quad(a, b, f, tol, M, method, return_intervals=False):
    '''
    Adaptively integrate f on [a,b] using a user-supplied quadrature rule.

    On each candidate interval [left,right], compare the quadrature on
    the whole interval with the sum of quadratures on its two halves.
    Reject and bisect if their absolute difference exceeds tol.

    Parameters
    ----------
    a, b : float
        Integration bounds, with a < b.
    f : callable
        Function accepting a NumPy array and returning an array.
    tol : float
        Local absolute acceptance threshold (NOT a rigorous global bound).
    M : int
        Quadrature nodes per evaluated subinterval (M=5 in Lab 11).
    method : callable
        One of eval_composite_trap, eval_composite_simpsons, eval_gauss_quad.
    return_intervals : bool
        If True, append the accepted adaptive partition as a (n,2) array.

    Returns
    -------
    I : float
        Integral assembled from the two half-interval approximations
        for each accepted adaptive interval.
    X : ndarray
        Unique quadrature nodes used for accepted half-interval values.
    n_intervals : int
        Number of accepted adaptive (coarse) intervals; this is the same
        count as the supplied instructor routine's nsplit variable.
    intervals : ndarray, optional
        Each accepted adaptive interval as [left,right]. Only returned if
        return_intervals=True. Useful for visualizing adaptive refinement.
    '''
    if not a < b:
        raise ValueError('Require a < b.')
    if tol <= 0.0:
        raise ValueError('Require tol > 0.')

    max_depth = 50
    I_initial, _, _ = method(M, a, b, f)
    pending = [(float(a), float(b), I_initial, 0)]
    accepted = []
    accepted_nodes = []
    I = 0.0
    n_intervals = 1

    while pending:
        left, right, I_parent, depth = pending.pop()
        midpoint = (left + right) / 2.0

        I_left, x_left, _ = method(M, left, midpoint, f)
        I_right, x_right, _ = method(M, midpoint, right, f)
        estimated_error = abs(I_left + I_right - I_parent)

        if estimated_error > tol:
            if depth >= max_depth or midpoint in (left, right):
                raise RuntimeError('Adaptive quadrature reached the depth limit.')

            # Stack uses LIFO; append right first to process left first.
            pending.append((midpoint, right, I_right, depth + 1))
            pending.append((left, midpoint, I_left, depth + 1))
            n_intervals += 1
        else:
            I += I_left + I_right
            accepted.append((left, right))
            accepted_nodes.extend((x_left, x_right))

    accepted.sort(key=lambda pair: pair[0])
    X = np.unique(np.concatenate(accepted_nodes))

    if return_intervals:
        return I, X, n_intervals, np.asarray(accepted)
    return I, X, n_intervals


#%% Named entry points for the three Lab 11 schemes

def adaptive_trapezoidal(a, b, f, tol, M=5, return_intervals=False):
    return adaptive_quad(a, b, f, tol, M, eval_composite_trap, return_intervals)


def adaptive_simpson(a, b, f, tol, M=5, return_intervals=False):
    return adaptive_quad(a, b, f, tol, M, eval_composite_simpsons, return_intervals)


def adaptive_gauss(a, b, f, tol, M=5, return_intervals=False):
    return adaptive_quad(a, b, f, tol, M, eval_gauss_quad, return_intervals)
