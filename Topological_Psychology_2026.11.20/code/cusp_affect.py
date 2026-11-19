"""Cusp-potential model of an affective life space (Lewin 1936 life-space concept, Thom cusp normal form).
State x: affective intensity (one coordinate of the life space; a 1-D reduction).
Control parameters:
  a : internal tension / reactivity (slow; negative values create bistability)
  b : provocation (external input; can change quickly)
Potential V(x; a, b) = x^4/4 + a x^2/2 - b x.  Dynamics dx = -V'(x) dt + sqrt(2D) dW.
Bifurcation set (fold curves): 4 a^3 + 27 b^2 = 0.  Inside the cusp (4a^3+27b^2<0) there are 2 minima and 1 saddle.
Stylised model: it makes no claim that real affect obeys this potential."""
import numpy as np


def dV(x, a, b):
    return x**3 + a * x - b


def equilibria(a, b):
    roots = np.roots([1.0, 0.0, a, -b])
    real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
    stable = 3 * real**2 + a > 0
    return real, stable


def n_minima(a, b):
    return int(equilibria(a, b)[1].sum())


def in_bistable(a, b):
    return 4 * a**3 + 27 * b**2 < 0


def fold_b(a):
    """Upper fold location b*(a) for a < 0."""
    return 2.0 * (-a) ** 1.5 / np.sqrt(27.0)


def langevin(x0, a_fun, b_fun, D, dt, T, rng):
    n = int(T / dt)
    t = np.arange(n + 1) * dt
    x = np.empty(n + 1)
    x[0] = x0
    noise = np.sqrt(2 * D * dt) * rng.standard_normal(n)
    for k in range(n):
        x[k + 1] = x[k] - dV(x[k], a_fun(t[k]), b_fun(t[k])) * dt + noise[k]
    return t, x
