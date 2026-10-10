#!/usr/bin/env python3
"""Synthetic verification for source-channel identifiability.

Dependencies: Python >=3.10; NumPy >=1.24.
This is a dimensionless binary communication example, not a model of biological aging.
It verifies source-distribution dependence of mutual information and an exact
reference-distribution decomposition using fixed, documented random seed.
"""
from __future__ import annotations
import numpy as np

SEED = 20261010
N = 200_000
TOL = 0.012

def h2(p: float) -> float:
    """Binary entropy in bits, including valid endpoint conventions."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if p in (0.0, 1.0):
        return 0.0
    return float(-p * np.log2(p) - (1.0-p) * np.log2(1.0-p))

def mi_bsc(p: float, eps: float) -> float:
    """Analytical I(X;Y) for X~Bernoulli(p), Y=X XOR N, N~Bernoulli(eps)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("source probability p must lie in [0, 1]")
    if not 0.0 <= eps <= 0.5:
        raise ValueError("crossover probability eps must lie in [0, 0.5]")
    py1 = eps + p * (1.0 - 2.0*eps)
    return h2(py1) - h2(eps)

def empirical_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in mutual information for paired binary arrays."""
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be nonempty one-dimensional arrays of equal length")
    joint = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            joint[a,b] = np.mean((x == a) & (y == b))
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    total = 0.0
    for a in (0,1):
        for b in (0,1):
            q = joint[a,b]
            if q > 0 and px[a] > 0 and py[b] > 0:
                total += q * np.log2(q/(px[a]*py[b]))
    return float(total)

def draw(rng: np.random.Generator, p: float, eps: float, n: int = N):
    x = rng.binomial(1, p, size=n).astype(np.int8)
    noise = rng.binomial(1, eps, size=n).astype(np.int8)
    return x, np.bitwise_xor(x, noise)

def main() -> None:
    # Analytical limiting cases.
    assert h2(0.0) == 0.0 and h2(1.0) == 0.0
    assert abs(h2(0.5) - 1.0) < 1e-12
    assert abs(mi_bsc(0.5, 0.0) - 1.0) < 1e-12
    assert abs(mi_bsc(0.5, 0.5)) < 1e-12
    assert abs(mi_bsc(0.0, 0.1)) < 1e-12
    try:
        mi_bsc(0.4, 0.7)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid channel parameter did not fail")

    eps_y, eps_a = 0.10, 0.20
    p_y, p_a, q = 0.50, 0.05, 0.50
    rng = np.random.default_rng(SEED)
    source_sweep = []
    for p in (0.05, 0.25, 0.50, 0.75, 0.95):
        analytic = mi_bsc(p, eps_y)
        x, y = draw(rng, p, eps_y)
        observed = empirical_mi(x, y)
        estimated_eps = float(np.mean(x != y))
        assert abs(observed - analytic) < TOL, (p, observed, analytic)
        assert abs(estimated_eps - eps_y) < 0.005, (p, estimated_eps, eps_y)
        source_sweep.append((p, analytic, observed, estimated_eps))

    # Same conditional channel, different source: MI changes while crossover stays fixed.
    mi_y_nat = mi_bsc(p_y, eps_y)
    mi_a_source_only = mi_bsc(p_a, eps_y)
    assert mi_a_source_only < mi_y_nat - 0.30

    # Source and channel change together. Decompose exactly at q=Bernoulli(0.5).
    mi_y_nat = mi_bsc(p_y, eps_y)
    mi_a_nat = mi_bsc(p_a, eps_a)
    d_a = mi_bsc(p_a, eps_a) - mi_bsc(q, eps_a)
    c_q = mi_bsc(q, eps_a) - mi_bsc(q, eps_y)
    d_y = mi_bsc(q, eps_y) - mi_bsc(p_y, eps_y)
    natural_delta = mi_a_nat - mi_y_nat
    reconstructed_delta = d_a + c_q + d_y
    assert abs(natural_delta - reconstructed_delta) < 1e-12
    assert c_q < 0.0  # the crossover probability increased under the same source

    # Independent Monte Carlo check of channel errors at the common source q.
    standardized = []
    for eps in (eps_y, eps_a):
        x, y = draw(rng, q, eps)
        estimated_eps = float(np.mean(x != y))
        estimated_std_mi = 1.0 - h2(estimated_eps)
        empirical = empirical_mi(x, y)
        theoretical = mi_bsc(q, eps)
        assert abs(estimated_std_mi - theoretical) < TOL
        assert abs(empirical - theoretical) < TOL
        standardized.append((eps, estimated_eps, estimated_std_mi, empirical, theoretical))

    print("Source-channel identifiability audit: synthetic verification")
    print(f"seed={SEED}; samples_per_scenario={N}; NumPy={np.__version__}")
    print("source_sweep (p, analytical_MI_bits, empirical_MI_bits):")
    for row in source_sweep:
        print(f"  p={row[0]:.2f}: analytic_MI={row[1]:.6f}; empirical_MI={row[2]:.6f}; estimated_error={row[3]:.6f}")
    print(f"fixed_channel_source_shift: I(p={p_y:.2f},eps={eps_y:.2f})={mi_y_nat:.6f}; I(p={p_a:.2f},eps={eps_y:.2f})={mi_a_source_only:.6f}")
    print(f"combined_shift_natural_delta={natural_delta:.6f} bits")
    print(f"decomposition: D_A={d_a:.6f}; C(q)={c_q:.6f}; D_Y={d_y:.6f}; sum={reconstructed_delta:.6f}")
    print("common-source channel comparison (eps, estimated_eps, standardized_MI, empirical_MI, analytical_MI):")
    for row in standardized:
        print("  " + ", ".join(f"{v:.6f}" for v in row))
    print("PASS: source dependence, endpoint cases, empirical convergence, parameter rejection, and exact decomposition")
    print("LIMIT: synthetic finite-alphabet verification only; no biological data or causal inference")

if __name__ == "__main__":
    main()
