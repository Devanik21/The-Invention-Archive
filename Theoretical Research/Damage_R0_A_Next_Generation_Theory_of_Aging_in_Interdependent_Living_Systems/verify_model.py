#!/usr/bin/env python3
"""Numerically check the Damage R0 threshold on a fixed synthetic network.

Requires Python >=3.10 and NumPy >=1.24. No biological data are loaded.
Run: python verify_model.py
"""
from __future__ import annotations
import sys
import numpy as np

SEED = 20261013

def spectral_radius(matrix: np.ndarray) -> float:
    values = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(values)))

def dominant_growth(matrix: np.ndarray) -> float:
    values = np.linalg.eigvals(matrix)
    return float(np.max(values.real))

def build_network(beta: float):
    # Symmetrized toy topology; rows index receiving damage states.
    A = np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=float)
    # Damage-type coupling: type 1/proteostasis and type 2/mitochondrial stress.
    C = np.array([[1.0, 0.35], [0.25, 0.8]], dtype=float)
    F = beta * np.kron(A, C)
    node_loss = np.array([1.0, 0.9, 1.0, 0.95, 0.9], dtype=float)
    type_loss = np.array([1.0, 0.8], dtype=float)
    rates = np.array([node_loss[i] * type_loss[j] for i in range(5) for j in range(2)])
    V = np.diag(rates)
    return A, C, F, V

def repair_node(V: np.ndarray, node: int, factor: float) -> np.ndarray:
    out = V.copy()
    for typ in range(2):
        out[node * 2 + typ, node * 2 + typ] *= factor
    return out

def main() -> int:
    print(f"Damage R0 synthetic verification | Python={sys.version.split()[0]} | NumPy={np.__version__} | seed={SEED}")
    # Threshold theorem: choose rates clearly on both sides of the boundary.
    _, _, F_sub, V = build_network(0.20)
    R_sub = spectral_radius(F_sub @ np.linalg.inv(V))
    g_sub = dominant_growth(F_sub - V)
    _, _, F_super, V = build_network(0.32)
    R_super = spectral_radius(F_super @ np.linalg.inv(V))
    g_super = dominant_growth(F_super - V)
    assert R_sub < 1.0 and g_sub < 0.0, (R_sub, g_sub)
    assert R_super > 1.0 and g_super > 0.0, (R_super, g_super)
    # F V^-1 and V^-1 F have the same nonzero eigenvalues.
    assert abs(spectral_radius(F_super @ np.linalg.inv(V)) - spectral_radius(np.linalg.inv(V) @ F_super)) < 1e-10
    # Homogeneous-loss special case: rho(beta kron(A,C)/mu) = beta*rho(A)*rho(C)/mu.
    A, C, F_hom, _ = build_network(0.32)
    mu = 1.3
    V_hom = mu * np.eye(F_hom.shape[0])
    R_matrix = spectral_radius(F_hom @ np.linalg.inv(V_hom))
    R_formula = 0.32 * spectral_radius(A) * spectral_radius(C) / mu
    assert abs(R_matrix - R_formula) < 1e-10, (R_matrix, R_formula)
    # A positive continuing source leaves positive steady damage even below threshold.
    source = np.linspace(0.01, 0.03, F_sub.shape[0])
    equilibrium = np.linalg.solve(V - F_sub, source)
    assert np.all(equilibrium > 0.0) and np.all(np.isfinite(equilibrium))
    assert np.linalg.norm((F_sub - V) @ equilibrium + source, ord=np.inf) < 1e-10
    # Hub repair is more effective than an equal relative change in any one leaf here.
    R_base = spectral_radius(F_super @ np.linalg.inv(V))
    V_hub = repair_node(V, 0, 1.8)
    R_hub = spectral_radius(F_super @ np.linalg.inv(V_hub))
    leaf_values = [spectral_radius(F_super @ np.linalg.inv(repair_node(V, node, 1.8))) for node in range(1, 5)]
    assert R_hub < 1.0 and all(value > 1.0 for value in leaf_values), (R_hub, leaf_values)
    # Reducing outgoing causal rates from the central node reduces the threshold.
    F_insulated = F_super.copy()
    F_insulated[:, 0:2] *= 0.5
    R_insulated = spectral_radius(F_insulated @ np.linalg.inv(V))
    assert R_insulated < R_base
    # Deterministic uncertainty propagation: perturb rates only as an illustrative stress test.
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(2000):
        beta = 0.32 * max(0.01, rng.normal(1.0, 0.05))
        _, _, F_draw, _ = build_network(beta)
        multipliers = np.clip(rng.normal(1.0, 0.05, V.shape[0]), 0.75, 1.25)
        V_draw = np.diag(np.diag(V) * multipliers)
        samples.append(spectral_radius(F_draw @ np.linalg.inv(V_draw)))
    samples = np.asarray(samples)
    lo, med, hi = np.quantile(samples, [0.025, 0.5, 0.975])
    assert np.all(np.isfinite(samples)) and np.all(samples > 0.0)
    print(f"Subcritical: R_d={R_sub:.6f}; dominant growth={g_sub:.6f} -> PASS")
    print(f"Supercritical baseline: R_d={R_base:.6f}; dominant growth={g_super:.6f} -> PASS")
    print(f"Homogeneous formula residual: {abs(R_matrix - R_formula):.3e} -> PASS")
    print(f"Source-driven subcritical steady burden range: [{equilibrium.min():.6f}, {equilibrium.max():.6f}] -> PASS")
    print(f"Hub repair x1.8: R_d={R_hub:.6f}; peripheral repair range=[{min(leaf_values):.6f}, {max(leaf_values):.6f}] -> PASS")
    print(f"Hub outgoing propagation x0.5: R_d={R_insulated:.6f} -> PASS")
    print(f"Synthetic rate-uncertainty stress test: median={med:.6f}; 95% interval=[{lo:.6f}, {hi:.6f}]; P(R_d>1)={np.mean(samples > 1):.3f}")
    print("LIMIT: the graph, rates, interventions, and uncertainty distribution are synthetic; no biological atlas, lifespan record, or experiment was analyzed.")
    print("PASS: threshold equivalence, spectral identity, homogeneous limit, forced subcritical state, intervention comparison, and seeded uncertainty checks")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
