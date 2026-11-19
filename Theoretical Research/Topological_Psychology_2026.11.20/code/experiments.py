#!/usr/bin/env python3
"""Topological-psychology experiments on the cusp affect model. Seed 20261120.
E1 bifurcation topology: stable-state count versus the analytic cusp set, on a grid.
E2 hysteresis: simulated jump provocations versus analytic fold locations.
E3 escape times: mean first-passage time versus noise, slope tested against Kramers' barrier.
E4 deformation rate: outburst timing versus the speed of provocation build-up.
E5 regulation: reappraisal as a gain reduction on provocation; fraction of time in the high branch."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, HERE)
import cusp_affect as C

rng = np.random.default_rng(20261120)


def E1_topology(n=161):
    a_grid = np.linspace(-2.0, 2.0, n)
    b_grid = np.linspace(-2.0, 2.0, n)
    match, counts, disagree = 0, {}, []
    for a in a_grid:
        for b in b_grid:
            k = C.n_minima(a, b)
            counts[k] = counts.get(k, 0) + 1
            predicted = 2 if C.in_bistable(a, b) else 1
            if k == predicted:
                match += 1
            elif len(disagree) < 5:
                disagree.append((float(a), float(b), k, predicted))
    total = n * n
    return {"grid": n, "points": total, "agreement": match / total,
            "minima_count_histogram": {str(k): v for k, v in sorted(counts.items())},
            "first_disagreements": disagree}


def E2_hysteresis(a_val=-1.0, D=1e-4, dt=0.01):
    bstar = float(C.fold_b(a_val))
    T_half = 900.0
    b_fun = lambda t: np.where(t < T_half,
                               -1.5 * bstar + 3.0 * bstar * t / T_half,
                               1.5 * bstar - 3.0 * bstar * (t - T_half) / T_half)
    a_fun = lambda t: a_val
    t, x = C.langevin(-1.0, a_fun, b_fun, D, dt, 2 * T_half, rng)
    b = b_fun(t)
    up = t < T_half
    idx_up = np.where(up & (x > 0.5))[0]
    jump_up = float(b[idx_up[0]]) if idx_up.size else float("nan")
    idx_dn = np.where((~up) & (x < -0.5))[0]
    jump_dn = float(b[idx_dn[0]]) if idx_dn.size else float("nan")
    return {"a": a_val, "b_fold_analytic": bstar, "b_jump_up_sim": jump_up,
            "b_jump_down_sim": jump_dn, "hysteresis_width_sim": jump_up - jump_dn,
            "hysteresis_width_analytic": 2 * bstar}


def E3_kramers(a_val=-1.0, reps=200, dt=0.01, tmax=3000.0):
    """Two minima at +-1 (for a=-1, b=0) and a saddle at 0. Barrier dV = V(0) - V(1) = 0.25."""
    barrier = 0.25
    Ds = np.array([0.10, 0.125, 0.15, 0.20, 0.25])
    rows = []
    for D in Ds:
        x = np.ones(reps)
        t_abs = np.full(reps, np.nan)
        alive = np.ones(reps, bool)
        steps = int(tmax / dt)
        sd = np.sqrt(2 * D * dt)
        for k in range(steps):
            if not alive.any():
                break
            x[alive] += -(x[alive]**3 + a_val * x[alive]) * dt + sd * rng.standard_normal(alive.sum())
            crossed = alive & (x < -0.2)
            t_abs[crossed] = (k + 1) * dt
            alive &= ~crossed
        mfpt = float(np.nanmean(t_abs))
        rows.append({"D": float(D), "inv_D": float(1 / D), "mfpt": mfpt,
                     "censored_fraction": float(np.mean(np.isnan(t_abs)))})
    invD = np.array([r["inv_D"] for r in rows])
    lnt = np.log([r["mfpt"] for r in rows])
    slope, intercept = np.polyfit(invD, lnt, 1)
    return {"rows": rows, "fitted_slope": float(slope), "barrier_analytic": barrier,
            "relative_error": float(abs(slope - barrier) / barrier)}


def E4_deformation(a_val=-1.0, D=1e-4, dt=0.01):
    bstar = float(C.fold_b(a_val))
    rows = []
    for speed in [0.0005, 0.002, 0.008, 0.032]:
        b_end = 1.5 * bstar
        T = (2 * b_end) / speed
        b_fun = lambda t, s=speed: -b_end + s * t
        a_fun = lambda t: a_val
        t, x = C.langevin(-1.0, a_fun, b_fun, D, dt, T, rng)
        idx = np.where(x > 0.5)[0]
        if idx.size:
            t_jump = float(t[idx[0]]); b_jump = float(b_fun(t[idx[0]]))
        else:
            t_jump = float("nan"); b_jump = float("nan")
        rows.append({"ramp_speed": float(speed), "t_jump": t_jump, "b_jump": b_jump,
                     "overshoot_past_fold": b_jump - bstar})
    return {"b_fold": bstar, "rows": rows}


def E5_regulation(a_val=-0.8, D=0.012, T=3000.0, dt=0.01):
    n = int(T / dt)
    t = np.arange(n) * dt
    sched = np.clip(0.9 * np.sin(2 * np.pi * t / 60.0)
                    + 0.5 * np.sin(2 * np.pi * t / 17.0 + 1.0), -1.2, 1.2)
    rows = []
    for gamma in [1.0, 0.8, 0.6, 0.4]:
        x = np.empty(n); x[0] = -1.0
        noise = np.sqrt(2 * D * dt) * rng.standard_normal(n)
        for k in range(n - 1):
            x[k + 1] = x[k] - (x[k]**3 + a_val * x[k] - gamma * sched[k]) * dt + noise[k]
        rows.append({"gamma": gamma, "fraction_time_high_branch": float(np.mean(x > 0.5))})
    return {"a": a_val, "rows": rows}


def main():
    out = {}
    out["E1"] = E1_topology(); out["E2"] = E2_hysteresis(); out["E3"] = E3_kramers()
    out["E4"] = E4_deformation(); out["E5"] = E5_regulation()
    names = {"E1": "E1_topology", "E2": "E2_hysteresis", "E3": "E3_kramers", "E4": "E4_deformation", "E5": "E5_regulation"}
    for k, v in out.items():
        with open(os.path.join(RES, names[k] + ".json"), "w") as f:
            json.dump(v, f, indent=2)
        print(k, json.dumps(v)[:300], flush=True)


if __name__ == "__main__":
    main()
