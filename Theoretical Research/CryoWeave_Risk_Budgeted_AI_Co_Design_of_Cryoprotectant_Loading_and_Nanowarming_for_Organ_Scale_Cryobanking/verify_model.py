#!/usr/bin/env python3
"""Reproducible reduced-order verification for the CryoWeave proposal.

This is a dimensionless proof-of-work benchmark, not a calibrated organ model.
Dependencies: Python >=3.10 and NumPy >=1.24. All parameters and constraints are
synthetic and are labelled as such in the manuscript. No clinical protocol is
produced by this script.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

SEED = 20261011
REGIONS = 3
K0 = np.array([0.48, 0.41, 0.34], dtype=float)  # normalized delivery rates
Q0 = np.array([0.36, 0.32, 0.28], dtype=float)  # normalized particle delivery rates
ETA0 = np.array([1.05, 1.00, 0.94], dtype=float) # normalized heating couplings
L = np.array([[ 1., -1.,  0.], [-1., 2., -1.], [0., -1., 1.]], dtype=float)
D0 = 0.50
DT = 0.015
WARM_HORIZON = 1.5
C_MIN, C_MAX = 0.80, 1.08
N_MIN, N_MAX = 0.50, 1.10
T_MIN, T_MAX = 0.80, 1.25
GRAD_MAX = 0.20

@dataclass(frozen=True)
class Scenario:
    k_scale: float
    q_scale: float
    eta_scale: float
    diffusion_scale: float

# Small, deterministic uncertainty set, not a probability distribution.
SCENARIOS = tuple(
    Scenario(ks, qs, es, ds)
    for ks, qs, es, ds in [
        (0.90, 0.85, 0.90, 0.90),
        (0.90, 1.00, 1.00, 1.00),
        (0.90, 1.10, 1.10, 1.10),
        (1.00, 0.85, 1.10, 0.90),
        (1.00, 1.00, 0.90, 1.10),
        (1.00, 1.10, 1.00, 1.00),
        (1.10, 0.85, 1.00, 1.10),
        (1.10, 1.00, 1.10, 1.00),
        (1.10, 1.10, 0.90, 0.90),
    ]
)

BOUNDS = np.array([
    [4.0, 8.0],    # normalized loading time
    [0.90, 1.10],  # normalized CPA inlet level
    [0.75, 1.15],  # normalized nanoparticle inlet level
    [0.70, 1.60],  # early RF power, dimensionless
    [0.70, 1.60],  # late RF power, dimensionless
], dtype=float)


def loading_state(k: np.ndarray, tau: float, c_in: float) -> tuple[np.ndarray, np.ndarray]:
    """Return final normalized CPA concentration and exact integral of c(t)^2."""
    k = np.asarray(k, dtype=float)
    if tau < 0 or c_in < 0 or np.any(k <= 0):
        raise ValueError("Loading parameters must be nonnegative with positive rates.")
    one_minus = -np.expm1(-k * tau)
    c = c_in * one_minus
    dose = c_in**2 * (tau - 2.0 * one_minus / k + (-np.expm1(-2.0 * k * tau)) / (2.0 * k))
    return c, np.maximum(dose, 0.0)


def particle_state(q: np.ndarray, tau: float, n_in: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if tau < 0 or n_in < 0 or np.any(q <= 0):
        raise ValueError("Particle-loading parameters must be nonnegative with positive rates.")
    return n_in * (-np.expm1(-q * tau))


def warm_trace(n: np.ndarray, eta: np.ndarray, p_early: float, p_late: float,
               diffusion: float = D0) -> np.ndarray:
    """Dimensionless three-region warming trace over a normalized risk window."""
    if min(p_early, p_late, diffusion) < 0:
        raise ValueError("Warming power and diffusion must be nonnegative.")
    steps = int(round(WARM_HORIZON / DT))
    T = np.zeros(REGIONS, dtype=float)
    trace = np.empty((steps + 1, REGIONS), dtype=float)
    trace[0] = T
    split = steps // 2
    for j in range(steps):
        p = p_early if j < split else p_late
        # -L@T is diffusive and conserves the spatial sum; eta*n*p is volumetric heating.
        dT = eta * n * p - diffusion * (L @ T)
        T = T + DT * dT
        trace[j + 1] = T
    return trace


def score(theta: np.ndarray, *, robust: bool = True) -> tuple[float, dict[str, float]]:
    """Worst-case score plus feasibility metrics for a five-variable design.

    theta=(tau, c_in, n_in, p_early, p_late). A finite score below 50 denotes a
    feasible point. Large values penalize violations; the returned metrics are
    synthetic and are not calibrated toxicity or tissue-damage predictions.
    """
    tau, c_in, n_in, p1, p2 = map(float, theta)
    if not (BOUNDS[:, 0] <= theta).all() or not (theta <= BOUNDS[:, 1]).all():
        return 1e6, {"feasible": 0.0}
    scenarios = SCENARIOS if robust else (Scenario(1., 1., 1., 1.),)
    worst_obj = -np.inf
    max_grad = 0.0
    min_c = np.inf
    max_c = -np.inf
    min_n = np.inf
    max_n = -np.inf
    min_final_t = np.inf
    max_final_t = -np.inf
    max_overshoot = 0.0
    tox_vals, risk_vals, grad_rms_vals, energy_vals = [], [], [], []
    for s in scenarios:
        k = K0 * s.k_scale
        q = Q0 * s.q_scale
        eta = ETA0 * s.eta_scale
        c, tox = loading_state(k, tau, c_in)
        n = particle_state(q, tau, n_in)
        tr = warm_trace(n, eta, p1, p2, D0 * s.diffusion_scale)
        spread = np.ptp(tr, axis=1)
        max_grad = max(max_grad, float(spread.max()))
        min_c = min(min_c, float(c.min()))
        max_c = max(max_c, float(c.max()))
        min_n = min(min_n, float(n.min()))
        max_n = max(max_n, float(n.max()))
        tf_min, tf_max = float(tr[-1].min()), float(tr[-1].max())
        min_final_t = min(min_final_t, tf_min)
        max_final_t = max(max_final_t, tf_max)
        max_overshoot = max(max_overshoot, max(0.0, tf_max - 1.0))
        # Smooth proxy for dwell inside normalized hazard window [0.25, 0.75].
        sig_lo = 1.0 / (1.0 + np.exp(-40.0 * (tr - 0.25)))
        sig_hi = 1.0 / (1.0 + np.exp(-40.0 * (0.75 - tr)))
        risk = float(np.mean(np.sum(sig_lo * sig_hi, axis=0)) * DT)
        center = tr.mean(axis=1, keepdims=True)
        grad_rms = float(np.sqrt(np.mean((tr - center) ** 2)))
        tox_sum = float(np.mean(tox))
        energy = float((p1**2 + p2**2) * WARM_HORIZON / 2.0)
        obj = 0.18 * tox_sum + 0.08 * tau + 0.8 * risk + 2.0 * float(np.mean((tr[-1] - 1.0)**2)) + 1.3 * grad_rms**2 + 0.025 * energy
        worst_obj = max(worst_obj, obj)
        tox_vals.append(tox_sum); risk_vals.append(risk); grad_rms_vals.append(grad_rms); energy_vals.append(energy)
    violations = [
        max(0.0, C_MIN - min_c), max(0.0, max_c - C_MAX),
        max(0.0, N_MIN - min_n), max(0.0, max_n - N_MAX),
        max(0.0, T_MIN - min_final_t), max(0.0, max_final_t - T_MAX),
        max(0.0, max_grad - GRAD_MAX),
    ]
    violation = float(np.sum(np.square(violations)))
    value = worst_obj + 1000.0 * violation
    feasible = float(violation < 1e-14)
    metrics = {
        "feasible": feasible, "min_c": min_c, "max_c": max_c, "min_n": min_n,
        "max_n": max_n, "min_final_t": min_final_t, "max_final_t": max_final_t,
        "max_spread": max_grad, "tox_proxy": max(tox_vals), "risk_dwell": max(risk_vals),
        "spatial_rms": max(grad_rms_vals), "energy_proxy": max(energy_vals),
        "violation": violation,
    }
    return float(value), metrics


def matern32_kernel(X1: np.ndarray, X2: np.ndarray, length: float = 0.32) -> np.ndarray:
    """ARD-scaled Matérn-3/2 covariance kernel on unit-scaled design variables."""
    delta = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(delta * delta, axis=2) + 1e-15) / length
    z = math.sqrt(3.0) * r
    return (1.0 + z) * np.exp(-z)


def scale_theta(theta: np.ndarray) -> np.ndarray:
    return (np.asarray(theta) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def bayes_search(seed: int = SEED, initial: int = 20, iterations: int = 100,
                 pool_size: int = 3000) -> tuple[np.ndarray, float, dict[str, float], int]:
    """Small deterministic GP lower-confidence-bound search over safe loading candidates.

    The GP is a prototype of the active-learning layer. Candidate feasibility in
    the final score remains governed by explicit hard constraints, not GP output.
    """
    rng = np.random.default_rng(seed)
    pool = BOUNDS[:, 0] + rng.random((pool_size, 5)) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    # Analytic low-cost screen on all nine rate scenarios for loading feasibility.
    screened = []
    for x in pool:
        tau, c_in, n_in, p1, p2 = x
        c_mins, c_maxs, n_mins, n_maxs = [], [], [], []
        for s in SCENARIOS:
            c, _ = loading_state(K0*s.k_scale, tau, c_in)
            n = particle_state(Q0*s.q_scale, tau, n_in)
            c_mins.append(c.min()); c_maxs.append(c.max()); n_mins.append(n.min()); n_maxs.append(n.max())
        if min(c_mins) >= C_MIN and max(c_maxs) <= C_MAX and min(n_mins) >= N_MIN and max(n_maxs) <= N_MAX:
            screened.append(x)
    if len(screened) < initial:
        raise RuntimeError(f"Only {len(screened)} loading-feasible candidates; expand pool or relax synthetic bounds.")
    candidates = np.asarray(screened)
    # Shuffle candidates and seed the optimizer with a varied initial design.
    rng.shuffle(candidates)
    X, y, met = [], [], []
    start_idx = np.linspace(0, len(candidates)-1, initial, dtype=int)
    evaluated = set()
    for idx in start_idx:
        theta = candidates[idx]
        val, metrics = score(theta, robust=True)
        X.append(scale_theta(theta)); y.append(val); met.append(metrics); evaluated.add(int(idx))
    # Active learning over the screened candidate pool. The acquisition explicitly
    # rewards uncertainty while prioritizing low predicted objective values.
    for _ in range(iterations):
        Xarr = np.asarray(X); yarr = np.asarray(y)
        K = matern32_kernel(Xarr, Xarr) + 1e-6 * np.eye(len(Xarr))
        try:
            chol = np.linalg.cholesky(K)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yarr))
            Kxs = matern32_kernel(scale_theta(candidates), Xarr)
            mu = Kxs @ alpha
            v = np.linalg.solve(chol, Kxs.T)
            var = np.maximum(1.0 - np.sum(v*v, axis=0), 1e-10)
            sd = np.sqrt(var)
            # Lower confidence bound: minimize mu - beta*sigma.
            acquisition = mu - 0.45 * sd
        except np.linalg.LinAlgError:
            acquisition = np.full(len(candidates), np.inf)
        if evaluated:
            acquisition[list(evaluated)] = np.inf
        nxt = int(np.argmin(acquisition))
        theta = candidates[nxt]
        val, metrics = score(theta, robust=True)
        X.append(scale_theta(theta)); y.append(val); met.append(metrics); evaluated.add(nxt)
    feasible_indices = [i for i, m in enumerate(met) if m["feasible"] > 0.5]
    if not feasible_indices:
        raise RuntimeError("Active search did not find any robustly feasible protocol in the synthetic search space.")
    best_i = min(feasible_indices, key=lambda i: y[i])
    theta_best = np.asarray([BOUNDS[j, 0] + X[best_i][j] * (BOUNDS[j, 1] - BOUNDS[j, 0]) for j in range(5)])
    return theta_best, float(y[best_i]), met[best_i], len(evaluated)


def main() -> int:
    # 1) Exact transport-dose formula agrees with numerical quadrature.
    k = np.array([0.2, 0.35, 0.5])
    tau, c_in = 3.7, 0.93
    c, analytic_dose = loading_state(k, tau, c_in)
    ts = np.linspace(0.0, tau, 100_001)
    numerical_dose = np.trapezoid((c_in * (1.0 - np.exp(-k[:, None] * ts[None, :])) )**2, ts, axis=1)
    np.testing.assert_allclose(analytic_dose, numerical_dose, rtol=2e-7, atol=2e-8)

    # 2) Diffusion conserves the sum in the absence of heating.
    rng = np.random.default_rng(SEED)
    T = rng.normal(size=REGIONS)
    d_diff = -D0 * (L @ T)
    assert abs(float(d_diff.sum())) < 1e-12
    assert np.linalg.eigvalsh(L).min() > -1e-12

    # 3) Robust active search runs deterministically and yields explicit hard-feasible design.
    theta, objective, metrics, evaluations = bayes_search()
    assert np.isfinite(objective)
    assert metrics["feasible"] == 1.0
    assert metrics["min_c"] >= C_MIN - 1e-9 and metrics["max_c"] <= C_MAX + 1e-9
    assert metrics["min_n"] >= N_MIN - 1e-9 and metrics["max_n"] <= N_MAX + 1e-9
    assert metrics["min_final_t"] >= T_MIN - 1e-9 and metrics["max_final_t"] <= T_MAX + 1e-9
    assert metrics["max_spread"] <= GRAD_MAX + 1e-9
    theta2, objective2, metrics2, evaluations2 = bayes_search()
    np.testing.assert_allclose(theta, theta2, rtol=0.0, atol=1e-12)
    assert abs(objective-objective2) < 1e-12 and evaluations == evaluations2

    # 4) Compare against a fixed, conservative reference design. The percent change
    # is purely relative to this synthetic objective; it is not a real-world gain.
    baseline_theta = np.array([6.5, 1.05, 1.0, 0.8, 0.8])
    baseline_obj, baseline_metrics = score(baseline_theta, robust=True)
    assert baseline_metrics["feasible"] == 1.0
    assert objective <= baseline_obj + 1e-9
    relative_gain = 100.0 * (baseline_obj - objective) / baseline_obj

    # 5) The safety gate rejects an intentionally unsafe design.
    unsafe_obj, unsafe = score(np.array([8.0, 1.1, 1.15, 1.6, 1.6]), robust=True)
    assert unsafe["feasible"] == 0.0 or unsafe_obj >= 50.0

    # 6) Directional checks: loading concentration rises with inlet level and time.
    c_short, _ = loading_state(k, 2.0, 1.0)
    c_long, _ = loading_state(k, 4.0, 1.0)
    assert np.all(c_long > c_short)
    assert np.all(loading_state(k, tau, 1.1)[0] > c)

    print("CRYOWEAVE REDUCED-ORDER VERIFICATION")
    print("PASS 1: closed-form CPA exposure integral matches 100,001-point quadrature")
    print("PASS 2: graph diffusion is sum-conserving; Laplacian is positive semidefinite")
    print("PASS 3: deterministic GP-guided search returned a robustly feasible design")
    print("PASS 4: robust design beats the fixed feasible baseline on the synthetic objective")
    print(f"      objective: optimized={objective:.6f}, baseline={baseline_obj:.6f}, relative change={relative_gain:.2f}%")
    print("PASS 5: hard safety gate rejects deliberately unsafe design")
    print("PASS 6: concentration is monotone in loading time and inlet level")
    print("SYNTHETIC OPTIMIZER OUTPUT (dimensionless, not a treatment protocol)")
    print("theta [tau, C_in, N_in, p_early, p_late] =", np.array2string(theta, precision=5))
    print(f"worst-case objective = {objective:.6f}; candidate evaluations = {evaluations}")
    for key in ("min_c", "max_c", "min_n", "max_n", "min_final_t", "max_final_t", "max_spread", "tox_proxy", "risk_dwell", "spatial_rms", "energy_proxy"):
        print(f"{key} = {metrics[key]:.6f}")
    print("LIMIT: deterministic toy model only; no experimentally calibrated biology or clinical efficacy is established.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
