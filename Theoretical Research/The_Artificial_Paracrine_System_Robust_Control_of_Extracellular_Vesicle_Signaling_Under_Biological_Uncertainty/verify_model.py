#!/usr/bin/env python3
"""Synthetic, dimensionless control-theory checks for the Artificial Paracrine System.

Dependencies: Python >=3.10 and NumPy >=1.24.
This script does not model a human body, prescribe payloads, or validate a therapy.
It checks a two-state linear plant, a static state-feedback controller, a bounded
uncertainty sweep, and elementary binary-symmetric-channel capacity arithmetic.
"""
import numpy as np

SEED = 20261010

def main():
    # xdot = A x + B u + E w; all quantities dimensionless and illustrative.
    A = np.array([[-1.0, 0.35], [0.20, -0.55]], dtype=float)
    B = np.array([[0.50], [0.80]], dtype=float)
    K = np.array([[1.20, 0.70]], dtype=float)
    Acl = A - B @ K
    eig_nominal = np.linalg.eigvals(Acl)
    assert np.max(np.real(eig_nominal)) < 0.0, f"Nominal closed loop unstable: {eig_nominal}"

    # Robustness screen: bounded multiplicative drift in plant matrix entries.
    rng = np.random.default_rng(SEED)
    worst_real_part = -np.inf
    unstable_cases = 0
    samples = 5000
    for _ in range(samples):
        delta = rng.uniform(-0.10, 0.10, size=A.shape)
        A_unc = A + delta
        ev = np.linalg.eigvals(A_unc - B @ K)
        worst_real_part = max(worst_real_part, float(np.max(np.real(ev))))
        unstable_cases += int(np.max(np.real(ev)) >= 0.0)
    assert unstable_cases == 0, f"Unstable cases in bounded uncertainty screen: {unstable_cases}/{samples}"

    # Controllability of this toy plant (rank is exact up to floating-point tolerance).
    ctrb = np.column_stack([B[:, 0], A @ B[:, 0]])
    rank = int(np.linalg.matrix_rank(ctrb, tol=1e-10))
    assert rank == 2, f"Expected controllable toy system, got rank {rank}"

    # Binary symmetric channel capacity: C=1-H_b(p), bits/use; p in [0, 0.5].
    def h2(p):
        if p == 0.0 or p == 1.0: return 0.0
        return -p*np.log2(p) - (1-p)*np.log2(1-p)
    c0, c1, c2 = 1-h2(0.0), 1-h2(0.1), 1-h2(0.5)
    assert abs(c0-1.0) < 1e-12
    assert abs(c1-(1-h2(0.1))) < 1e-12
    assert abs(c2) < 1e-12
    assert c0 > c1 > c2

    print("Artificial Paracrine System: synthetic verification")
    print(f"seed={SEED}; uncertainty_samples={samples}; uncertainty_bound=±0.10")
    print("nominal_closed_loop_eigenvalues=" + np.array2string(eig_nominal, precision=6))
    print(f"worst_sampled_real_eigenvalue={worst_real_part:.6f}")
    print(f"unstable_sampled_cases={unstable_cases}/{samples}")
    print(f"controllability_rank={rank}/2")
    print(f"binary_channel_capacity_bits_per_use: p=0 -> {c0:.6f}; p=0.1 -> {c1:.6f}; p=0.5 -> {c2:.6f}")
    print("PASS: all synthetic mathematical assertions satisfied")
    print("LIMIT: sampled stability is not a formal H-infinity certificate or biological validation")

if __name__ == '__main__':
    main()
