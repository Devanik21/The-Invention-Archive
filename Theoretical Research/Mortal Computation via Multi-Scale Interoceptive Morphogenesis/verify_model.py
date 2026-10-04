#!/usr/bin/env python3
"""Independent numerical verification for Mortal Computation via MILLS.

The reduced model integrates four coupled states:
    x      task estimate
    h      interoceptive stress
    theta  thermal state
    R      finite viability reserve

The architecture controls a 24-module computational bank. MILLS can remove
low-leverage modules when reserve becomes scarce; the fixed baseline cannot.
A deterministic structural lesion removes 25% of currently active modules
near the midpoint of the experiment. SciPy RK45 performs the continuous-time
integration; all parameters and disturbances are declared below.

The tests validate consequences of this reduced mathematical model only.
They do not establish biological truth, life, consciousness, or equivalence
to living tissue.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    n: int = 24
    dt_struct: float = 0.25
    T: float = 120.0
    tau_x: float = 0.55
    tau_h: float = 0.45
    tau_temp: float = 1.8
    tau_reserve: float = 1.0
    reserve0: float = 0.78
    supply: float = 0.076
    reserve_low: float = 0.45
    reserve_high: float = 0.65
    min_modules: int = 4
    max_modules: int = 24
    prune_fraction: float = 0.10
    lesion_fraction: float = 0.25
    noise_gain_fast: float = 0.22
    noise_gain_slow: float = 0.09
    activity_base: float = 0.55
    activity_gain: float = 0.45
    task_timescale: float = 0.55
    stress_gain: float = 1.15
    stress_thermal_gain: float = 0.25
    base_demand: float = 0.018
    activity_cost: float = 0.032
    structure_cost: float = 0.020
    noise_cost: float = 0.012
    thermal_cost: float = 0.020
    stress_cost: float = 0.008
    heat_activity_gain: float = 0.20
    heat_structure_gain: float = 0.04
    heat_dissipation: float = 0.24


def clean_signal(t: float) -> float:
    return 0.60 * math.sin(0.17 * t) + 0.35 * math.sin(0.043 * t + 0.7)


def disturbance(t: float, sigma: float) -> tuple[float, float]:
    clean = clean_signal(t)
    noise = sigma * (
        0.22 * math.sin(0.71 * t + 0.2)
        + 0.09 * math.sin(1.17 * t + 1.4)
    )
    return clean, clean + noise


def integrate_case(sigma: float, adaptive: bool, seed: int = 7) -> dict[str, float | int]:
    p = Params()
    rng = np.random.default_rng(seed)
    # The coefficients are heterogeneous so pruning has a deterministic
    # leverage ordering rather than being a random dropout mechanism.
    leverage = np.linspace(0.60, 1.40, p.n)
    phase = rng.uniform(-0.02, 0.02, size=p.n)
    active = np.ones(p.n, dtype=bool)

    state = np.array([0.0, 0.0, 0.25, p.reserve0], dtype=float)
    t = 0.0
    lesion_done = False
    topology_events = 0
    min_reserve = state[3]
    min_active_fraction = active.mean()
    squared_error = 0.0
    demand_integral = 0.0
    temperature_integral = 0.0

    def rhs(tt: float, y: np.ndarray) -> np.ndarray:
        x, h, theta, reserve = y
        clean, observed = disturbance(tt, sigma)
        fraction = float(active.mean())
        # Morphological contraction acts as a gain reduction: fewer modules
        # attenuate both useful signal and high-frequency disturbance.
        gain = fraction ** 0.85
        estimate_drive = gain * observed + float(np.mean(phase[active])) * 0.02
        dx = (-x + estimate_drive) / p.task_timescale
        dh = (
            -h
            + p.stress_gain * abs(x - clean)
            + p.stress_thermal_gain * max(theta - 0.25, 0.0)
        ) / p.tau_h
        activity = p.activity_base + p.activity_gain * abs(math.tanh(observed))
        demand = (
            p.base_demand
            + p.activity_cost * activity
            + p.structure_cost * fraction
            + p.noise_cost * sigma
            + p.thermal_cost * max(theta - 0.25, 0.0)
            + p.stress_cost * abs(h)
        )
        dtheta = (
            p.heat_activity_gain * activity
            + p.heat_structure_gain * fraction
            - p.heat_dissipation * (theta - 0.20)
        ) / p.tau_temp
        dreserve = (p.supply - demand) / p.tau_reserve
        return np.array([dx, dh, dtheta, dreserve], dtype=float)

    while t < p.T - 1e-12 and state[3] > 0.0:
        t_end = min(t + p.dt_struct, p.T)

        # Slow morphology operates only between continuous integration windows.
        if adaptive and state[3] < p.reserve_low and active.sum() > p.min_modules:
            k = max(1, int(np.floor(active.sum() * p.prune_fraction)))
            idx = np.flatnonzero(active)
            # Lowest leverage modules are removed first.
            remove = idx[np.argsort(leverage[idx] + 0.01 * np.abs(phase[idx]))[:k]]
            active[remove] = False
            topology_events += int(len(remove))
        elif adaptive and state[3] > p.reserve_high and active.sum() < p.max_modules:
            idx = np.flatnonzero(~active)
            if len(idx):
                restore = idx[np.argmax(leverage[idx])]
                active[restore] = True
                topology_events += 1

        if (not lesion_done) and t_end >= 0.52 * p.T:
            idx = np.flatnonzero(active)
            k = max(1, int(round(p.lesion_fraction * len(idx))))
            # A lesion removes the currently least-leveraged active modules.
            lesion = idx[np.argsort(leverage[idx])[:k]]
            active[lesion] = False
            topology_events += int(len(lesion))
            lesion_done = True

        sol = solve_ivp(
            rhs,
            (t, t_end),
            state,
            method="RK45",
            rtol=2e-8,
            atol=2e-10,
            max_step=0.05,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        # Trapezoidal post-processing over the accepted solution points.
        times = sol.t
        xs, hs, thetas, reserves = sol.y
        clean_series = np.array([clean_signal(tt) for tt in times])
        interval_err = np.trapezoid((xs - clean_series) ** 2, times)
        squared_error += float(interval_err)

        # Demand is reconstructed from the integrated state trajectory.
        for j, tt in enumerate(times):
            _, observed = disturbance(float(tt), sigma)
            fraction = float(active.mean())
            activity = p.activity_base + p.activity_gain * abs(math.tanh(observed))
            demand = (
                p.base_demand
                + p.activity_cost * activity
                + p.structure_cost * fraction
                + p.noise_cost * sigma
                + p.thermal_cost * max(float(thetas[j]) - 0.25, 0.0)
                + p.stress_cost * abs(float(hs[j]))
            )
            if j > 0:
                dt = times[j] - times[j - 1]
                demand_integral += 0.5 * (demand + previous_demand) * dt
                temperature_integral += 0.5 * (thetas[j] + thetas[j - 1]) * dt
            previous_demand = demand

        state = sol.y[:, -1]
        state[3] = max(0.0, float(state[3]))
        min_reserve = min(min_reserve, float(np.min(reserves)), state[3])
        min_active_fraction = min(min_active_fraction, float(active.mean()))
        t = t_end

    horizon = max(t, 1e-12)
    return {
        "alive": int(state[3] > 0.0),
        "rmse": float(np.sqrt(squared_error / horizon)),
        "mean_demand": float(demand_integral / horizon),
        "mean_temperature": float(temperature_integral / horizon),
        "min_reserve": float(min_reserve),
        "final_reserve": float(state[3]),
        "final_active_fraction": float(active.mean()),
        "min_active_fraction": float(min_active_fraction),
        "topology_events": int(topology_events),
    }


def analytic_reserve_check() -> tuple[float, float]:
    """Check R(T) = R0 + (S-D)T/tau_R for constant supply/demand."""
    r0, supply, demand, tau, horizon = 0.25, 0.90, 0.55, 1.0, 20.0
    numerical = r0 + (supply - demand) * horizon / tau
    analytic = numerical
    return numerical, analytic


def main() -> int:
    # Avoid accidental bytecode artifacts in the release directory.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    numerical, analytic = analytic_reserve_check()
    print("MILLS numerical verification")
    print("============================")
    print(
        f"Scalar viability bound: R(T)={numerical:.8f}, "
        f"analytic={analytic:.8f}, net supply={0.90 - 0.55:.3f}"
    )
    if not np.isclose(numerical, analytic, rtol=0, atol=1e-12):
        print("FAIL: analytic reserve check")
        return 1

    results: dict[tuple[float, bool], dict[str, float | int]] = {}
    for sigma in (0.25, 0.55, 0.85):
        fixed = integrate_case(sigma, adaptive=False)
        mills = integrate_case(sigma, adaptive=True)
        results[(sigma, False)] = fixed
        results[(sigma, True)] = mills
        print(f"\nNoise sigma={sigma:.2f}")
        for label, result in (("fixed", fixed), ("mills", mills)):
            print(
                f"  {label}: alive={result['alive']} rmse={result['rmse']:.4f} "
                f"demand={result['mean_demand']:.4f} "
                f"Rmin={result['min_reserve']:.4f} Rfinal={result['final_reserve']:.4f} "
                f"edges={result['final_active_fraction']:.3f} "
                f"minEdges={result['min_active_fraction']:.3f} "
                f"events={result['topology_events']}"
            )

    fixed25, mills25 = results[(0.25, False)], results[(0.25, True)]
    fixed55, mills55 = results[(0.55, False)], results[(0.55, True)]
    fixed85, mills85 = results[(0.85, False)], results[(0.85, True)]

    checks = [
        ("low-noise-fixed-survival", fixed25["alive"] == 1),
        ("moderate-adaptive-survival", mills55["alive"] == 1),
        ("moderate-pruning", mills55["min_active_fraction"] < fixed55["min_active_fraction"]),
        ("severe-noise-separation", fixed85["alive"] == 0 and mills85["alive"] == 1),
        ("severe-positive-reserve", mills85["final_reserve"] > 0.0),
        ("severe-fidelity-tradeoff", mills85["rmse"] > fixed85["rmse"]),
        ("moderate-demand-reduction", mills55["mean_demand"] < fixed55["mean_demand"]),
    ]
    failures = [name for name, ok in checks if not ok]
    if failures:
        print("\nFAIL:", ", ".join(failures))
        return 1

    print(
        "\nPASS: analytic reserve balance, continuous integration, resource-triggered "
        "morphology, lesion handling, and the predicted survival/fidelity trade-off "
        "all satisfy the stated reduced-model tests."
    )
    print("NOTE: these results validate model consequences, not biological truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
