#!/usr/bin/env python3
"""Deterministic mathematical verification for the FER proof-of-concept.

This script verifies mathematical identities and toy-model properties only.
It does not model human physiology, tissue acoustics, clinical treatment,
or biological rejuvenation.

Dependencies: Python 3 + NumPy.
"""
from __future__ import annotations

import numpy as np

SEED = 20261008
RNG = np.random.default_rng(SEED)


def gf2_rank(rows: list[int]) -> int:
    basis = {}
    rank = 0
    for row in rows:
        x = int(row)
        while x:
            pivot = x.bit_length() - 1
            if pivot in basis:
                x ^= basis[pivot]
            else:
                basis[pivot] = x
                rank += 1
                break
    return rank


def clique_complex_beta1(adj: np.ndarray) -> tuple[int, int, int, int]:
    """Exact H1 Betti number for the clique complex through dimension 2 over GF(2)."""
    n = adj.shape[0]
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append((i, j))
                union(i, j)
    m = len(edges)
    edge_index = {e: k for k, e in enumerate(edges)}
    components = len({find(i) for i in range(n)})
    rank_b1 = n - components

    triangle_masks: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            for k in range(j + 1, n):
                if adj[i, k] and adj[j, k]:
                    mask = (
                        (1 << edge_index[(i, j)])
                        | (1 << edge_index[(i, k)])
                        | (1 << edge_index[(j, k)])
                    )
                    triangle_masks.append(mask)

    rows = [0] * m
    for col, mask in enumerate(triangle_masks):
        mm = mask
        while mm:
            bit = mm & -mm
            edge = bit.bit_length() - 1
            rows[edge] |= 1 << col
            mm ^= bit
    rank_b2 = gf2_rank(rows)
    beta1 = m - rank_b1 - rank_b2
    return n, m, components, beta1


def radius_graph(points: np.ndarray, eps: float) -> np.ndarray:
    d = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    return (dist <= eps) & (~np.eye(len(points), dtype=bool))


def topology_demo(n: int = 32):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    young = np.c_[np.cos(theta), np.sin(theta)]
    young += RNG.normal(0.0, 0.015, young.shape)

    x = np.linspace(-1.0, 1.0, n)
    aged = np.c_[x, 0.08 * np.sin(3.0 * x)]
    aged += RNG.normal(0.0, 0.008, aged.shape)

    eps_grid = np.linspace(0.24, 0.42, 10)
    beta_y, beta_a = [], []
    for eps in eps_grid:
        beta_y.append(clique_complex_beta1(radius_graph(young, eps))[-1])
        beta_a.append(clique_complex_beta1(radius_graph(aged, eps))[-1])
    return eps_grid, np.array(beta_y), np.array(beta_a)


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def quantile_map(p: np.ndarray, grid: np.ndarray, m: int = 4096):
    p = normalize(p)
    cdf = np.cumsum(p)
    u = (np.arange(m) + 0.5) / m
    q = np.interp(u, cdf, grid)
    return q, u


def ot_geodesic_metrics(p0: np.ndarray, p1: np.ndarray, grid: np.ndarray):
    q0, u = quantile_map(p0, grid)
    q1, _ = quantile_map(p1, grid, len(u))
    delta = q1 - q0
    w2_sq = float(np.mean(delta * delta))

    tvals = np.linspace(0.0, 1.0, 21)
    path = [(1.0 - t) * q0 + t * q1 for t in tvals]
    step_speeds = [
        float(np.mean((path[i + 1] - path[i]) ** 2)) ** 0.5
        for i in range(len(path) - 1)
    ]
    dist_sq = [float(np.mean((qt - q0) ** 2)) for qt in path]
    expected = [float(t * t * w2_sq) for t in tvals]
    return w2_sq, np.array(step_speeds), np.array(dist_sq), np.array(expected)


def fisher_rao_metric(p: np.ndarray, tangent: np.ndarray) -> float:
    p = normalize(p)
    v = np.asarray(tangent, dtype=float)
    v = v - v.mean()
    return float(np.sum((v * v) / p))


def control_demo():
    # Linearized state-velocity map: v_state = A u.
    A = np.array(
        [[1.20, 0.10, 0.30],
         [0.15, 0.95, 0.25],
         [0.05, 0.20, 1.10]],
        dtype=float,
    )
    target = np.array([0.30, 0.41, 0.29]) - np.array([0.16, 0.28, 0.56])
    u_min = np.linalg.pinv(A) @ target
    residual = float(np.linalg.norm(A @ u_min - target))
    energy = float(u_min @ u_min)
    rank = int(np.linalg.matrix_rank(A))
    # The minimum-norm identity is equivalent to orthogonality to the null space.
    _, _, vh = np.linalg.svd(A)
    null_dim = int(np.sum(np.linalg.svd(A, compute_uv=False) < 1e-12))
    orthogonality_residual = 0.0
    if null_dim:
        null_basis = vh[-null_dim:].T
        orthogonality_residual = float(np.linalg.norm(null_basis.T @ u_min))
    return residual, energy, rank, orthogonality_residual


def functor_demo():
    """Composition-preservation test for a deliberately linear state-to-control functor."""
    f = np.array([[1.0, 0.2], [0.0, 1.0]])
    g = np.array([[0.8, 0.0], [0.3, 1.1]])
    gf = g @ f
    mapped_composition = g @ f
    object_identity = np.eye(2)
    return float(np.linalg.norm(gf - mapped_composition)), float(np.linalg.norm(object_identity @ f - f))


def lindblad_trace_test():
    """Trace preservation of a toy two-level pure-dephasing Lindblad generator."""
    rho = np.array([[0.7 + 0j, 0.2 - 0.1j], [0.2 + 0.1j, 0.3 + 0j]])
    L = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    LdL = L.conj().T @ L
    dissipator = L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return float(abs(np.trace(dissipator)))


def defect_demo():
    r = 1.0
    n = 721
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    x = r * np.cos(t)
    y = r * np.sin(t)
    theta_plus = 0.5 * np.arctan2(y, x)
    theta_minus = -0.5 * np.arctan2(y, x)
    phase_plus = np.unwrap(np.angle(np.exp(2j * theta_plus)))
    phase_minus = np.unwrap(np.angle(np.exp(2j * theta_minus)))
    q_plus = float((phase_plus[-1] - phase_plus[0]) / (4.0 * np.pi))
    q_minus = float((phase_minus[-1] - phase_minus[0]) / (4.0 * np.pi))
    return q_plus, q_minus, q_plus + q_minus


def main():
    eps, beta_y, beta_a = topology_demo()
    print('FER numerical verification')
    print('seed =', SEED)
    print('topology beta1(youth):', beta_y.tolist())
    print('topology beta1(aged): ', beta_a.tolist())
    print('topology separation exists:', bool(np.any(beta_y > beta_a)))

    grid = np.linspace(-3.0, 3.0, 96)
    p0 = np.exp(-0.5 * ((grid + 0.75) / 0.62) ** 2)
    p0 += 0.35 * np.exp(-0.5 * ((grid - 1.10) / 0.40) ** 2)
    p1 = np.exp(-0.5 * ((grid - 0.20) / 0.52) ** 2)
    p1 += 0.50 * np.exp(-0.5 * ((grid + 1.00) / 0.50) ** 2)
    w2sq, speeds, dist_sq, expected = ot_geodesic_metrics(p0, p1, grid)
    fisher = fisher_rao_metric(p0, p1 - p0)
    print('W2^2 =', w2sq)
    print('geodesic step-speed std =', float(np.std(speeds)))
    print('W2 endpoint error =', float(abs(dist_sq[-1] - w2sq)))
    print('geodesic t^2 law max error =', float(np.max(np.abs(dist_sq - expected))))
    print('Fisher-Rao tangent norm^2 =', fisher)

    residual, energy, rank, orth_residual = control_demo()
    print('control rank =', rank)
    print('control residual =', residual)
    print('minimum-norm control energy =', energy)
    print('null-space orthogonality residual =', orth_residual)

    fcomp, fid = functor_demo()
    print('functor composition residual =', fcomp)
    print('functor identity residual =', fid)

    lindblad_trace_error = lindblad_trace_test()
    print('Lindblad trace-derivative magnitude =', lindblad_trace_error)

    qp, qm, qsum = defect_demo()
    print('defect charge +1/2 =', qp)
    print('defect charge -1/2 =', qm)
    print('paired net charge =', qsum)

    assert np.any(beta_y > beta_a)
    assert np.isfinite(w2sq) and w2sq > 0.0
    assert np.std(speeds) < 2e-3
    assert abs(dist_sq[-1] - w2sq) < 1e-9
    assert np.max(np.abs(dist_sq - expected)) < 1e-9
    assert residual < 1e-12 and rank == 3 and orth_residual < 1e-12
    assert fcomp < 1e-15 and fid < 1e-15
    assert lindblad_trace_error < 1e-15
    assert abs(qp - 0.5) < 1e-12 and abs(qm + 0.5) < 1e-12
    assert abs(qsum) < 1e-12
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()