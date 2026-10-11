#!/usr/bin/env python3
"""Synthetic mathematical verification for Cohomological Rejuvenation.

Requirements: Python >=3.10, NumPy >=2.0.
This checks finite-dimensional identities and a seeded toy simulation only.
It does NOT analyse an aging atlas or validate a biological mechanism.
"""
from __future__ import annotations
import sys
import numpy as np

SEED = 20261011
TOL = 1e-9

def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL: {name}. {detail}")
    print(f"PASS: {name}" + (f" -- {detail}" if detail else ""))

def incidence(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    B = np.zeros((len(edges), n), dtype=float)
    for i, (u, v) in enumerate(edges):
        B[i, u] = -1.0
        B[i, v] = 1.0
    return B

def main() -> int:
    # Triangle: connected graph, one independent cycle, scalar stalks.
    edges = [(0, 1), (1, 2), (2, 0)]
    B = incidence(3, edges)
    x = np.array([0.7, -0.2, 1.1])
    L0 = B.T @ B
    energy = float(x @ L0 @ x)
    residual_energy = float(np.linalg.norm(B @ x) ** 2)
    eig = np.linalg.eigvalsh(L0)
    check("Laplacian symmetry", np.allclose(L0, L0.T, atol=TOL))
    check("Laplacian positive semidefinite", eig.min() >= -TOL, f"minimum eigenvalue={eig.min():.3e}")
    check("Dirichlet identity x^T L x = ||B x||^2", abs(energy-residual_energy) < TOL,
          f"both={energy:.12g}")
    check("constant section is harmonic for identity restrictions", np.linalg.norm(B @ np.ones(3)) < TOL)

    # Exact cochain has no harmonic/cycle obstruction by construction.
    y_exact = B @ x
    P = B @ np.linalg.pinv(B)
    harmonic_exact = (np.eye(B.shape[0]) - P) @ y_exact
    check("coboundary has zero cycle obstruction", np.linalg.norm(harmonic_exact) < TOL)

    # An arbitrary edge cochain on this graph has a nonzero cycle component.
    y_observed = np.array([1.0, 0.4, -0.1])
    y_projected = P @ y_observed
    y_harmonic = y_observed - y_projected
    check("projection components are orthogonal", abs(float(y_projected @ y_harmonic)) < TOL)
    check("harmonic residual is orthogonal to coboundaries", np.linalg.norm(B.T @ y_harmonic) < TOL)
    check("non-exact edge data can have positive cycle residual", np.linalg.norm(y_harmonic) > 1e-6,
          f"residual norm={np.linalg.norm(y_harmonic):.6f}")
    cycle_rank = B.shape[0] - np.linalg.matrix_rank(B)
    check("graph cycle-rank identity", cycle_rank == 1, f"computed cycle rank={cycle_rank}")

    # Seeded synthetic demonstration: node noise is deliberately increased.
    # We report a contrast from the toy generator, not an empirical aging result.
    rng = np.random.default_rng(SEED)
    n, d = 40, 3
    path_edges = [(i, i+1) for i in range(n-1)]
    Bp = incidence(n, path_edges)
    latent = rng.normal(0.0, 0.15, size=(n, d))
    common = rng.normal(0.0, 1.0, size=(1, d))
    young_scores, old_scores = [], []
    for _ in range(200):
        low = latent + common + rng.normal(0.0, 0.08, size=(n, d))
        high = latent + common + rng.normal(0.0, 0.50, size=(n, d))
        low_edge, high_edge = Bp @ low, Bp @ high
        young_scores.append(float(np.mean(np.sum(low_edge**2, axis=1))))
        old_scores.append(float(np.mean(np.sum(high_edge**2, axis=1))))
    mean_low = float(np.mean(young_scores))
    mean_high = float(np.mean(old_scores))
    check("toy mismatch rises when injected independent noise rises", mean_high > mean_low,
          f"low-noise={mean_low:.6f}; high-noise={mean_high:.6f}; seed={SEED}")
    print("\nSUMMARY")
    print("Mathematical checks passed: 9/9")
    print(f"Synthetic contrast (not biological): ratio={mean_high/mean_low:.3f}")
    print("Not tested: atlas-scale validity, aging association, prediction, frailty/mortality, intervention, or safety.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VERIFICATION ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)