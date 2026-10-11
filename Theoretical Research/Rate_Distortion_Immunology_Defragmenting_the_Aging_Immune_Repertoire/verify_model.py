#!/usr/bin/env python3
"""Synthetic verification for Rate-Distortion Immunology.

This script checks exact properties of the defined mathematical model and a small
synthetic candidate-selection problem. It does not analyze human repertoire data,
model real pathogen epidemiology, validate TCR binding, or support clinical removal.
Requires Python >=3.10 and NumPy >=1.24.
"""
from __future__ import annotations
import itertools
import math
import sys
import numpy as np

SEED = 20261011

def miss_probabilities(recognition: np.ndarray, selected: tuple[int, ...] | list[int]) -> np.ndarray:
    """Conditional-independence approximation q_e(S)=prod_i(1-r_ie)."""
    r = np.asarray(recognition, dtype=float)
    if r.ndim != 2 or not np.all(np.isfinite(r)):
        raise ValueError("recognition must be a finite 2D matrix")
    if np.any(r < 0.0) or np.any(r > 1.0):
        raise ValueError("recognition probabilities must lie in [0,1]")
    if any((not isinstance(i, (int, np.integer))) or i < 0 or i >= r.shape[0] for i in selected):
        raise ValueError("selected clone index out of bounds")
    if len(set(selected)) != len(selected):
        raise ValueError("selected clone indices must be unique")
    if not selected:
        return np.ones(r.shape[1], dtype=float)
    return np.prod(1.0 - r[np.asarray(selected, dtype=int), :], axis=0)

def distortion(recognition: np.ndarray, selected: tuple[int, ...] | list[int], prior: np.ndarray, loss: np.ndarray | None = None) -> float:
    pi = np.asarray(prior, dtype=float)
    if pi.ndim != 1 or len(pi) != recognition.shape[1] or not np.all(np.isfinite(pi)):
        raise ValueError("prior dimensions/values invalid")
    if np.any(pi < 0) or not np.isclose(pi.sum(), 1.0, atol=1e-10):
        raise ValueError("prior must be a probability vector")
    ell = np.ones_like(pi) if loss is None else np.asarray(loss, dtype=float)
    if ell.shape != pi.shape or not np.all(np.isfinite(ell)) or np.any(ell < 0):
        raise ValueError("loss weights must be finite, nonnegative, and aligned")
    return float(np.dot(pi * ell, miss_probabilities(recognition, selected)))

def brute_force_optimum(recognition: np.ndarray, prior: np.ndarray, k: int) -> tuple[float, tuple[int, ...]]:
    n = recognition.shape[0]
    if not isinstance(k, int) or k < 0 or k > n:
        raise ValueError("k must be an integer in [0, n_candidates]")
    best = (float("inf"), tuple())
    # Vectorized combinations remain tractable for the explicitly tiny benchmark.
    for subset in itertools.combinations(range(n), k):
        value = distortion(recognition, subset, prior)
        if value < best[0]:
            best = value, subset
    return best

def coverage_utility(recognition: np.ndarray, selected: tuple[int, ...] | list[int], prior: np.ndarray) -> float:
    return 1.0 - distortion(recognition, selected, prior)

def shannon_effective_diversity(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1 or not len(c) or not np.all(np.isfinite(c)) or np.any(c < 0) or c.sum() <= 0:
        raise ValueError("counts must be a nonempty finite nonnegative vector with positive sum")
    p = c[c > 0] / c.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

def synthetic_instance() -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]:
    """Construct 8 observed clonotypes plus 10 idealized reserve candidates."""
    n_epitopes = 36
    r = np.zeros((18, n_epitopes), dtype=float)
    # Five observed clones redundantly recognize the same four epitope classes.
    for i in range(5):
        for e in range(4):
            r[i, e] = 0.86 - 0.025 * i + 0.01 * (e % 2)
    # Three other observed clones have narrow, partially overlapping recognition.
    for i, e in zip(range(5, 8), (4, 5, 6)):
        r[i, e] = 0.88
        if e + 1 < n_epitopes:
            r[i, e + 1] = 0.14
    # Reserve candidates cover distinct blocks; they stand in for feasible candidates,
    # not as evidence that such clones exist or can be generated in a particular person.
    for j in range(10):
        idx = 8 + j
        start = 4 + 3 * j
        for e in range(start, min(start + 3, n_epitopes)):
            r[idx, e] = 0.91
        # small cross-reactive tail keeps the toy less degenerate
        if start > 0:
            r[idx, start - 1] = 0.08
    observed = tuple(range(8))
    uniform = np.full(n_epitopes, 1.0 / n_epitopes)
    focused = np.full(n_epitopes, 0.60 / 4.0)
    focused[4:] = 0.40 / (n_epitopes - 4)
    return r, observed, uniform, focused

def check_submodularity(r: np.ndarray, pi: np.ndarray) -> None:
    # Exhaustively check diminishing returns over selected small sets and candidate j.
    n = r.shape[0]
    universe = range(n)
    def f(S): return coverage_utility(r, tuple(S), pi)
    for a_size in range(min(3, n) + 1):
        for A in itertools.combinations(universe, a_size):
            Aset = set(A)
            for b_size in range(a_size, min(3, n) + 1):
                for B in itertools.combinations(universe, b_size):
                    Bset = set(B)
                    if not Aset.issubset(Bset):
                        continue
                    for j in universe:
                        if j in Bset:
                            continue
                        lhs = f(Aset | {j}) - f(Aset)
                        rhs = f(Bset | {j}) - f(Bset)
                        assert lhs + 1e-12 >= rhs, (A, B, j, lhs, rhs)
                        assert rhs >= -1e-12, (A, B, j, rhs)

def main() -> int:
    rng = np.random.default_rng(SEED)
    print("Rate-Distortion Immunology: synthetic verification")
    print(f"Python={sys.version.split()[0]}; NumPy={np.__version__}; seed={SEED}")
    r, observed, pi, focused = synthetic_instance()
    d_empty = distortion(r, (), pi)
    d_observed = distortion(r, observed, pi)
    opt_uniform, opt_set = brute_force_optimum(r, pi, 8)
    debt = d_observed - opt_uniform
    d_focused_observed = distortion(r, observed, focused)
    d_focused_opt, focused_set = brute_force_optimum(r, focused, 8)
    focused_debt = d_focused_observed - d_focused_opt
    assert math.isclose(d_empty, 1.0, abs_tol=1e-12)
    assert 0.0 <= opt_uniform <= d_observed <= 1.0
    assert debt >= -1e-12 and focused_debt >= -1e-12
    assert len(opt_set) == 8 and len(focused_set) == 8
    # Perfect recognition of every epitope should produce zero distortion.
    perfect = np.ones((1, 5), dtype=float)
    assert distortion(perfect, (0,), np.full(5, .2)) == 0.0
    # Increasing selected sets cannot increase distortion.
    for _ in range(50):
        rr = rng.uniform(0, 0.7, size=(7, 12))
        pp = rng.dirichlet(np.ones(12))
        a = tuple(sorted(rng.choice(7, size=2, replace=False).tolist()))
        j = next(k for k in range(7) if k not in a)
        assert distortion(rr, a + (j,), pp) <= distortion(rr, a, pp) + 1e-12
    check_submodularity(r[:7, :10], np.full(10, .1))
    # An abundance metric depends only on counts, not antigen-recognition labels.
    counts = np.array([40, 20, 12, 8, 6, 5, 5, 4], dtype=float)
    diversity = shannon_effective_diversity(counts)
    assert diversity > 1.0
    # Alter target-prior vector while retaining the identical repertoire and counts;
    # the coverage debt changes, showing that this is not a prior-free biological constant.
    assert not math.isclose(debt, focused_debt, rel_tol=1e-6, abs_tol=1e-6)
    # Input validation checks.
    bad = False
    try:
        miss_probabilities(np.array([[1.1, 0.0]]), (0,))
    except ValueError:
        bad = True
    assert bad
    bad = False
    try:
        distortion(r, observed, np.full(36, 1.0))
    except ValueError:
        bad = True
    assert bad
    print(f"Observed synthetic distortion (uniform prior): {d_observed:.6f}")
    print(f"Exact optimum distortion (uniform prior): {opt_uniform:.6f}")
    print(f"Coverage debt (uniform prior): {debt:.6f}")
    print(f"Observed synthetic distortion (focused prior): {d_focused_observed:.6f}")
    print(f"Exact optimum distortion (focused prior): {d_focused_opt:.6f}")
    print(f"Coverage debt (focused prior): {focused_debt:.6f}")
    print(f"Shannon effective diversity of fixed abundance profile: {diversity:.6f}")
    print("PASS: bounds, exhaustive optimizer, submodularity, monotonicity, prior sensitivity, and invalid-input checks")
    print("LIMIT: synthetic illustration only; no human cohort, TCR binding assay, or clinical outcome tested")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
