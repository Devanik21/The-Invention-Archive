#!/usr/bin/env python3
"""Deterministic checks for the Gaussian identifiability-budget toy model.

Only the Python standard library is required. This validates algebra and a toy
Monte Carlo experiment; it does not validate any real scientific mechanism.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist

NORMAL = NormalDist()


def f0(u: float) -> float:
    return u


def f1(u: float) -> float:
    return u * u


def kl_equal_variance(mu0: float, mu1: float, sigma: float, n: int = 1) -> float:
    if sigma <= 0 or n < 1:
        raise ValueError("sigma must be positive and n must be >= 1")
    return n * (mu0 - mu1) ** 2 / (2.0 * sigma**2)


def gaussian_pair_error(divergence: float) -> float:
    if divergence < 0:
        raise ValueError("KL divergence cannot be negative")
    return NORMAL.cdf(-math.sqrt(divergence / 2.0))


def reps_for_target(mu0: float, mu1: float, sigma: float, epsilon: float) -> int:
    """Minimum n for exact binary Gaussian error <= epsilon, 0<epsilon<0.5."""
    if not (0.0 < epsilon < 0.5):
        raise ValueError("epsilon must be between 0 and 0.5")
    delta = abs(mu0 - mu1)
    if delta == 0.0:
        raise ValueError("No finite replicate count separates equal means")
    z = NORMAL.inv_cdf(1.0 - epsilon)
    threshold = 2.0 * z * z
    per_rep = delta * delta / (2.0 * sigma * sigma)
    return math.ceil((threshold - 1e-12) / per_rep)


def simulate_error(n: int, true_mean: float, rival_mean: float, sigma: float,
                   trials: int, seed: int) -> float:
    """Likelihood-ratio test for two known equal-variance Gaussian means."""
    rng = random.Random(seed)
    threshold = (true_mean + rival_mean) / 2.0
    wrong = 0
    for _ in range(trials):
        avg = sum(rng.gauss(true_mean, sigma) for _ in range(n)) / n
        if true_mean < rival_mean:
            wrong += avg > threshold
        else:
            wrong += avg < threshold
    return wrong / trials


def main() -> int:
    tol = 1e-12
    # Observational support contains only u=0 and u=1.
    for u in (0.0, 1.0):
        assert abs(f0(u) - f1(u)) < tol, f"candidate means disagree at u={u}"
    d_obs = sum(kl_equal_variance(f0(u), f1(u), 1.0) for u in (0.0, 1.0))
    assert abs(d_obs) < tol, f"expected zero observational KL, got {d_obs}"

    # At u=2, the means are 2 and 4; each replicate contributes KL=2.
    assert f0(2.0) == 2.0 and f1(2.0) == 4.0
    d_one = kl_equal_variance(f0(2.0), f1(2.0), sigma=1.0, n=1)
    assert abs(d_one - 2.0) < tol
    assert abs(kl_equal_variance(2.0, 4.0, 1.0, n=3) - 6.0) < tol

    expected = {1: gaussian_pair_error(2.0),
                2: gaussian_pair_error(4.0),
                3: gaussian_pair_error(6.0)}
    n_required = reps_for_target(2.0, 4.0, 1.0, epsilon=0.05)
    assert n_required == 3, f"expected 3 replicates for 5% error, got {n_required}"
    assert expected[2] > 0.05 and expected[3] < 0.05
    # Verify the quantile/inversion identity used for each n.
    for n, p in expected.items():
        assert 0.0 < p < 0.5
        assert abs(p - NORMAL.cdf(-math.sqrt(n))) < 1e-12

    trials = 60_000
    estimates = {}
    # Two true hypotheses are symmetric; test both to catch direction errors.
    for n in (1, 2, 3):
        p0 = simulate_error(n, true_mean=2.0, rival_mean=4.0,
                            sigma=1.0, trials=trials, seed=1000 + n)
        p1 = simulate_error(n, true_mean=4.0, rival_mean=2.0,
                            sigma=1.0, trials=trials, seed=2000 + n)
        estimates[n] = (p0, p1)
        # Approx. 0.006 covers >99.9% of binomial fluctuations at these rates.
        assert abs(p0 - expected[n]) < 0.006, (n, p0, expected[n])
        assert abs(p1 - expected[n]) < 0.006, (n, p1, expected[n])

    print("Identifiability-budget verification: PASS")
    print("Checks: exact observational equivalence; zero observed KL; intervention means; KL additivity; Gaussian error formula; minimum replicate count; seeded Monte Carlo agreement.")
    print(f"Observed-support KL: {d_obs:.12f}")
    print(f"Single replicate at u=2: KL={d_one:.6f}, analytic error={expected[1]:.6f}")
    for n in (1, 2, 3):
        p0, p1 = estimates[n]
        print(f"n={n}: KL={2*n:.3f}, analytic error={expected[n]:.6f}, empirical M0={p0:.6f}, empirical M1={p1:.6f}")
    print(f"Replicates required for exact binary error <= 0.05: {n_required}")
    print("Interpretation limit: toy statistical verification only; no biological/physical empirical claim is tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
