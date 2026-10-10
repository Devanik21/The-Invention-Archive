#!/usr/bin/env python3
"""Reproducible synthetic audit for the heteroplasmy scheduling preprint.

Python >= 3.10; NumPy >= 1.24. No biological data are loaded.
"""
from __future__ import annotations
import math
import sys
import numpy as np

SEED = 20261010


def logit(x: np.ndarray | float) -> np.ndarray | float:
    if np.any(np.asarray(x) <= 0.0) or np.any(np.asarray(x) >= 1.0):
        raise ValueError("logit requires all inputs strictly between 0 and 1")
    return np.log(np.asarray(x) / (1.0 - np.asarray(x)))


def logistic(z: np.ndarray | float) -> np.ndarray | float:
    z = np.asarray(z)
    # Stable for the parameter range used in this audit.
    return 1.0 / (1.0 + np.exp(-z))


def trajectory(x0: float, s: float, delta: float, u: np.ndarray, dt: float) -> np.ndarray:
    """Exact trajectory for piecewise-constant controls sampled on a time grid."""
    u = np.asarray(u, dtype=float)
    if not 0.0 < x0 < 1.0:
        raise ValueError("x0 must be strictly between 0 and 1")
    if dt <= 0 or not np.all(np.isfinite(u)) or np.any(u < 0):
        raise ValueError("dt must be positive and control values finite/nonnegative")
    cumdose = np.concatenate(([0.0], np.cumsum(u) * dt))
    t = np.arange(u.size + 1, dtype=float) * dt
    return logistic(logit(x0) + s * t - delta * cumdose)


def threshold_burden(x: np.ndarray, threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    return float(np.mean(np.maximum(x[:-1] - threshold, 0.0) ** 2))


def feasible_random_schedule(rng: np.random.Generator, n: int, dt: float, dose: float, umax: float) -> np.ndarray:
    """Water-fill positive random weights to satisfy the exact dose and box bound."""
    weights = rng.uniform(0.05, 1.0, size=n)
    lo, hi = 0.0, (umax / weights.min()) * 2.0
    for _ in range(90):
        mid = (lo + hi) / 2
        total = np.minimum(umax, mid * weights).sum() * dt
        if total < dose:
            lo = mid
        else:
            hi = mid
    u = np.minimum(umax, ((lo + hi) / 2) * weights)
    # Correct floating point residue on one unsaturated element.
    residue = dose - u.sum() * dt
    candidates = np.flatnonzero(u < umax - 1e-12)
    if candidates.size:
        i = int(candidates[0])
        u[i] += residue / dt
    assert np.all(u >= -1e-10) and np.all(u <= umax + 1e-10)
    assert abs(float(u.sum() * dt) - dose) < 1e-8
    return np.clip(u, 0.0, umax)


def stochastic_paths(x0: float, s: float, delta: float, u: np.ndarray, dt: float,
                     ne: float, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Euler-Maruyama diffusion approximation, projected onto [0,1]."""
    if ne <= 0 or n_paths < 1:
        raise ValueError("ne and n_paths must be positive")
    xs = np.empty((n_paths, u.size + 1), dtype=float)
    xs[:, 0] = x0
    for k, uk in enumerate(u):
        x = xs[:, k]
        drift = x * (1.0 - x) * (s - delta * uk)
        diffusion = np.sqrt(np.maximum(x * (1.0 - x), 0.0) / ne)
        proposal = x + drift * dt + diffusion * math.sqrt(dt) * rng.normal(size=n_paths)
        # Projection is an explicit numerical convention, not an exact boundary model.
        xs[:, k + 1] = np.clip(proposal, 0.0, 1.0)
    return xs


def main() -> None:
    print("Heteroplasmy phase-control audit: synthetic verification")
    print(f"Python={sys.version.split()[0]}; NumPy={np.__version__}; seed={SEED}")
    rng = np.random.default_rng(SEED)
    x0, s, delta, umax, T, B, threshold = 0.40, 0.15, 0.18, 1.0, 20.0, 5.0, 0.70
    dt = 0.05
    n = int(round(T / dt))
    t = np.arange(n) * dt
    u_front = np.zeros(n)
    u_front[: int(round((B / umax) / dt))] = umax
    u_const = np.full(n, B / T)

    # Exact transformation and equal-dose endpoint invariance.
    x_front = trajectory(x0, s, delta, u_front, dt)
    x_const = trajectory(x0, s, delta, u_const, dt)
    assert abs(float(u_front.sum() * dt) - B) < 1e-10
    assert abs(float(u_const.sum() * dt) - B) < 1e-10
    assert abs(float(x_front[-1] - x_const[-1])) < 1e-12
    assert np.all(x_front <= x_const + 1e-12)
    assert threshold_burden(x_front, threshold) <= threshold_burden(x_const, threshold) + 1e-12

    # Fixed-dose dominance for many independently generated admissible controls.
    for _ in range(250):
        u = feasible_random_schedule(rng, n, dt, B, umax)
        x = trajectory(x0, s, delta, u, dt)
        assert np.all(x_front <= x + 2e-10), "front-loading dominance failed"
        assert abs(float(x[-1] - x_front[-1])) < 2e-10, "equal-dose endpoint invariant failed"
        assert threshold_burden(x_front, threshold) <= threshold_burden(x, threshold) + 1e-10

    # The selectivity-sign edge cases are essential kill checks.
    x_null_front = trajectory(x0, s, 0.0, u_front, dt)
    x_null_const = trajectory(x0, s, 0.0, u_const, dt)
    assert np.allclose(x_null_front, x_null_const, atol=1e-12)
    x_adverse_front = trajectory(x0, s, -delta, u_front, dt)
    x_adverse_const = trajectory(x0, s, -delta, u_const, dt)
    assert np.all(x_adverse_front >= x_adverse_const - 1e-12)

    # Stochastic audit. Equal random seeds and path count make the comparison reproducible.
    n_paths = 6000
    rng_f = np.random.default_rng(SEED + 1)
    rng_c = np.random.default_rng(SEED + 2)
    xs_f = stochastic_paths(x0, s, delta, u_front, dt, ne=300.0, n_paths=n_paths, rng=rng_f)
    xs_c = stochastic_paths(x0, s, delta, u_const, dt, ne=300.0, n_paths=n_paths, rng=rng_c)
    assert np.all((xs_f >= 0.0) & (xs_f <= 1.0))
    assert np.all((xs_c >= 0.0) & (xs_c <= 1.0))
    # Restricted crossing fraction is measured over discrete observation times.
    cross_f = np.any(xs_f >= threshold, axis=1)
    cross_c = np.any(xs_c >= threshold, axis=1)
    burden_f = np.maximum(xs_f[:, :-1] - threshold, 0.0).mean(axis=1)
    burden_c = np.maximum(xs_c[:, :-1] - threshold, 0.0).mean(axis=1)
    var_t_f = np.var(xs_f, axis=0, ddof=1)
    var_t_c = np.var(xs_c, axis=0, ddof=1)
    # Monte Carlo results are descriptive; no claim of pathwise stochastic dominance.
    print(f"deterministic final x (pulse, constant) = ({x_front[-1]:.6f}, {x_const[-1]:.6f})")
    print(f"deterministic threshold burden (pulse, constant) = ({threshold_burden(x_front, threshold):.8f}, {threshold_burden(x_const, threshold):.8f})")
    print(f"stochastic fraction crossing x* by T (pulse, constant) = ({cross_f.mean():.5f}, {cross_c.mean():.5f})")
    print(f"stochastic mean threshold burden (pulse, constant) = ({burden_f.mean():.8f}, {burden_c.mean():.8f})")
    print(f"stochastic variance at T (pulse, constant) = ({var_t_f[-1]:.8f}, {var_t_c[-1]:.8f})")
    print("PASS: analytic transform, equal-dose endpoint, deterministic dominance, selectivity-sign checks, and bounded stochastic simulation")

if __name__ == "__main__":
    main()
