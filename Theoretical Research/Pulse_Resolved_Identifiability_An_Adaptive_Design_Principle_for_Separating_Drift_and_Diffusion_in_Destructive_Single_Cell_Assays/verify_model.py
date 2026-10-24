#!/usr/bin/env python3
"""Reproducible synthetic benchmark for pulse-resolved identifiability.

All generated observations are synthetic. The script tests algebraic identities,
Fisher-information structure, finite-sample estimation, and artifact generation;
it does not validate a biological mechanism.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

SEED = 20261024
N_CELLS = 40
N_REPLICATES = 40
PULSE_AMPLITUDE = 0.60
PULSE_DURATION = 0.75
MEASUREMENT_VARIANCE = 0.04
NOMINAL = np.log(np.array([0.80, 1.10, 0.22], dtype=float))
LOG_PRIOR_SD = 0.70
CANDIDATE_TIMES = np.linspace(0.10, 3.00, 30)
PARAM_NAMES = ("log_kappa", "log_beta", "log_q")
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def unpack(theta):
    log_values = np.asarray(theta, dtype=float)
    if log_values.shape != (3,) or not np.all(np.isfinite(log_values)):
        raise ValueError("log parameters must be a finite length-three vector")
    values = np.exp(log_values)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("parameters must be finite and strictly positive")
    return float(values[0]), float(values[1]), float(values[2])


def mean_response(t, theta, u0=PULSE_AMPLITUDE, tau=PULSE_DURATION):
    kappa, beta, _ = unpack(theta)
    t = float(t)
    if t <= tau:
        return beta * u0 / kappa * (1.0 - math.exp(-kappa * t))
    return beta * u0 / kappa * (1.0 - math.exp(-kappa * tau)) * math.exp(-kappa * (t - tau))


def observation_variance(theta, r=MEASUREMENT_VARIANCE):
    kappa, _, q = unpack(theta)
    if kappa <= 0 or q <= 0 or r < 0:
        raise ValueError("kappa and q must be positive and observation variance nonnegative")
    return q / (2.0 * kappa) + r


def fisher_information_at_time(t, theta, n=N_CELLS, r=MEASUREMENT_VARIANCE):
    theta = np.asarray(theta, dtype=float)
    eps = 1e-5
    grad_mean = np.zeros(3)
    grad_logv = np.zeros(3)
    v = observation_variance(theta, r)
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        mp = mean_response(t, theta + step)
        mm = mean_response(t, theta - step)
        vp = observation_variance(theta + step, r)
        vm = observation_variance(theta - step, r)
        grad_mean[j] = (mp - mm) / (2.0 * eps)
        grad_logv[j] = (math.log(vp) - math.log(vm)) / (2.0 * eps)
    return n * (np.outer(grad_mean, grad_mean) / v + 0.5 * np.outer(grad_logv, grad_logv))


def schedule_information(times, theta, n=N_CELLS):
    out = np.zeros((3, 3), dtype=float)
    for t in times:
        out += fisher_information_at_time(float(t), theta, n)
    return (out + out.T) / 2.0


def regularized_logdet(matrix, ridge=1e-8):
    sign, value = np.linalg.slogdet(matrix + ridge * np.eye(matrix.shape[0]))
    return float(value) if sign > 0 else float("-inf")


def matrix_rank(matrix):
    vals = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    threshold = max(float(np.max(np.abs(vals))), 1.0) * 1e-8
    return int(np.sum(vals > threshold))


def nearest_times(requested):
    return [float(CANDIDATE_TIMES[int(np.argmin(np.abs(CANDIDATE_TIMES - x)))]) for x in requested]


def greedy_schedule(theta, grid=CANDIDATE_TIMES, k=4, n=N_CELLS, used=None):
    selected = list(used or [])
    information = schedule_information(selected, theta, n) if selected else np.zeros((3, 3))
    for _ in range(k - len(selected)):
        candidates = [float(t) for t in grid if not any(abs(float(t) - s) < 1e-9 for s in selected)]
        scored = []
        for t in candidates:
            candidate_info = information + fisher_information_at_time(t, theta, n)
            scored.append((regularized_logdet(candidate_info), t))
        _, best = max(scored, key=lambda pair: pair[0])
        selected.append(best)
        information = schedule_information(selected, theta, n)
    return sorted(selected)


def fit_map(observations, initial=NOMINAL):
    """Fit log parameters to independent per-time sample means/variances."""
    if not observations:
        return np.asarray(initial, dtype=float).copy()

    def objective(theta):
        try:
            kappa, beta, q = unpack(theta)
            v = observation_variance(theta)
        except (ValueError, OverflowError):
            return 1e100
        if not np.isfinite(v) or v <= 0:
            return 1e100
        nll = 0.5 * np.sum((theta - NOMINAL) ** 2 / LOG_PRIOR_SD**2)
        for t, n, sample_mean, sample_var in observations:
            mu = mean_response(t, theta)
            nll += 0.5 * n * math.log(v)
            nll += ((n - 1) * sample_var + n * (sample_mean - mu) ** 2) / (2.0 * v)
        return float(nll)

    bounds = [(-2.0, 1.0), (-2.0, 1.5), (-4.0, 0.5)]
    result = minimize(objective, np.asarray(initial, dtype=float), method="L-BFGS-B", bounds=bounds)
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("MAP estimation returned non-finite parameters")
    return np.asarray(result.x, dtype=float)


def sample_summary(t, theta_true, n, rng):
    mu = mean_response(t, theta_true)
    v = observation_variance(theta_true)
    values = rng.normal(mu, math.sqrt(v), size=n)
    return (float(t), int(n), float(np.mean(values)), float(np.var(values, ddof=1)))


def run_policy(policy, theta_true, rng):
    observations = []
    selected = []
    if policy == "uniform":
        selected = nearest_times([0.10, 1.10, 2.00, 3.00])
    elif policy == "early-only":
        selected = nearest_times([0.10, 0.20, 0.40, 0.70])
    elif policy == "random":
        selected = sorted(float(x) for x in rng.choice(CANDIDATE_TIMES, size=4, replace=False))
    elif policy != "adaptive":
        raise ValueError(f"Unknown policy: {policy}")

    if policy != "adaptive":
        for t in selected:
            observations.append(sample_summary(t, theta_true, N_CELLS, rng))
        return fit_map(observations), selected

    theta_hat = NOMINAL.copy()
    selected = []
    for _ in range(4):
        info = schedule_information(selected, theta_hat, N_CELLS) if selected else np.zeros((3, 3))
        candidates = [float(t) for t in CANDIDATE_TIMES if not any(abs(float(t) - s) < 1e-9 for s in selected)]
        scored = []
        for t in candidates:
            scored.append((regularized_logdet(info + fisher_information_at_time(t, theta_hat, N_CELLS)), t))
        _, chosen = max(scored, key=lambda pair: pair[0])
        selected.append(chosen)
        observations.append(sample_summary(chosen, theta_true, N_CELLS, rng))
        theta_hat = fit_map(observations, initial=theta_hat)
    return theta_hat, sorted(selected)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError(f"No rows supplied for {path.name}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.svg")
    plt.close(fig)


def draw_network_graph(graph, positions, stem, title, node_colors=None, edge_widths=None, figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    node_list = list(graph.nodes())
    colors = node_colors or [PALETTE[i % len(PALETTE)] for i in range(len(node_list))]
    nx.draw_networkx_nodes(graph, positions, node_size=1900, node_color=colors, edgecolors="#243746", linewidths=0.7, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="#172B3A", ax=ax)
    widths = edge_widths or [1.2] * graph.number_of_edges()
    nx.draw_networkx_edges(graph, positions, width=widths, edge_color="#526574", arrows=graph.is_directed(), arrowsize=18, connectionstyle="arc3,rad=0.04", ax=ax)
    ax.set_title(title, color="#172B3A", pad=12, weight="bold")
    ax.axis("off")
    save_figure(fig, stem)


def draw_flow(nodes, arrows, positions, stem, title, figsize=(12, 4.3)):
    fig, ax = plt.subplots(figsize=figsize)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    for key, label in nodes.items():
        x, y = positions[key]
        box = FancyBboxPatch((x - 0.78, y - 0.34), 1.56, 0.68, boxstyle="round,pad=0.04,rounding_size=0.07", linewidth=1.1, edgecolor="#526574", facecolor="#E8F2F4")
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.0, color="#172B3A", wrap=True)
    for a, b, label in arrows:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        if x2 < x1:
            start = (x1 - 0.80, y1 + 0.34)
            end = (x2 + 0.80, y2 + 0.34)
            arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.0, color="#007C91", connectionstyle="arc3,rad=0.30")
            label_y = max(y1, y2) + 0.70
        else:
            start = (x1 + np.sign(x2-x1)*0.80, y1 + np.sign(y2-y1)*0.08)
            end = (x2 - np.sign(x2-x1)*0.80, y2 - np.sign(y2-y1)*0.08)
            arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.0, color="#007C91", connectionstyle="arc3,rad=0.0")
            label_y = (y1+y2)/2 + 0.16
        ax.add_patch(arrow)
        if label:
            ax.text((x1+x2)/2, label_y, label, ha="center", va="bottom", fontsize=7.4, color="#526574")
    ax.set_title(title, color="#172B3A", weight="bold", pad=10)
    ax.set_xlim(min(p[0] for p in positions.values()) - 1.1, max(p[0] for p in positions.values()) + 1.1)
    ax.set_ylim(min(p[1] for p in positions.values()) - 0.8, max(p[1] for p in positions.values()) + 0.8)
    ax.axis("off")
    save_figure(fig, stem)


def generate_figures_and_data():
    rng = np.random.default_rng(SEED)
    theta_true = np.log(np.array([0.80, 1.10, 0.22]))
    theta_alt1 = np.log(np.array([0.60, 1.10, 0.165]))
    theta_alt2 = np.log(np.array([1.20, 1.10, 0.33]))
    time_grid = np.linspace(0.0, 3.0, 61)

    # Figure 1: model-generated pulse-response curves and synthetic cohort means.
    rows = []
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for idx, theta in enumerate([theta_alt1, theta_true, theta_alt2]):
        kappa, beta, q = unpack(theta)
        means = np.array([mean_response(t, theta) for t in time_grid])
        label = f"kappa={kappa:.2f}"
        ax.plot(time_grid, means, linewidth=2.0, color=PALETTE[idx], label=label)
        sampled_times = np.array([0.15, 0.40, 0.70, 0.95, 1.40, 2.00, 2.70])
        sample_means = []
        ses = []
        for t in sampled_times:
            vals = rng.normal(mean_response(t, theta), math.sqrt(observation_variance(theta)), size=50)
            sample_means.append(float(np.mean(vals)))
            ses.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            rows.append({"kappa": kappa, "beta": beta, "q": q, "time": float(t), "sample_mean": sample_means[-1], "standard_error": ses[-1], "source": "synthetic model-generated cohort"})
        ax.errorbar(sampled_times, sample_means, yerr=ses, fmt="o", ms=3.5, color=PALETTE[idx], alpha=0.8, capsize=2)
    ax.axvline(PULSE_DURATION, color="#526574", linestyle="--", linewidth=1.0, label="pulse ends")
    ax.set_xlabel("Time (normalized units)")
    ax.set_ylabel("Population mean (normalized state)")
    ax.set_title("Pulse response: relaxation controls the post-pulse slope", color="#172B3A", weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    save_figure(fig, "fig1_mean_response")
    write_csv(RES / "mean_response.csv", rows)

    # Figure 2: variance-only likelihood ridge.
    k_obs, _, q_obs = unpack(theta_true)
    v_obs = q_obs / (2.0 * k_obs) + MEASUREMENT_VARIANCE
    n_variance = 300
    sample_var = v_obs * rng.chisquare(n_variance - 1) / (n_variance - 1)
    kappas = np.linspace(0.35, 1.35, 100)
    qs = np.linspace(0.06, 0.42, 100)
    K, Q = np.meshgrid(kappas, qs)
    V = Q / (2.0 * K) + MEASUREMENT_VARIANCE
    nll = 0.5 * n_variance * np.log(V) + (n_variance - 1) * sample_var / (2.0 * V)
    delta = nll - np.min(nll)
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    levels = [0.02, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0]
    contours = ax.contour(K, Q, delta, levels=levels, colors=[PALETTE[i % len(PALETTE)] for i in range(len(levels))], linewidths=1.1)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2g")
    ax.plot([k_obs], [q_obs], marker="*", markersize=10, color="#D55E00", label="generating parameters")
    ridge_q = 2.0 * (v_obs - MEASUREMENT_VARIANCE) * kappas
    ax.plot(kappas, ridge_q, linestyle="--", color="#172B3A", linewidth=1.2, label="constant-variance ridge")
    ax.set_xlabel("Relaxation rate kappa")
    ax.set_ylabel("Diffusion intensity q")
    ax.set_title("Stationary variance leaves a drift–diffusion ridge", color="#172B3A", weight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    save_figure(fig, "fig2_variance_confounding")
    write_csv(RES / "variance_ridge.csv", [
        {"kappa": float(k), "q": float(q), "predicted_variance": float(q/(2*k)+MEASUREMENT_VARIANCE), "delta_nll": float(d), "source": "synthetic variance-only likelihood grid"}
        for k, q, d in zip(K.ravel(), Q.ravel(), delta.ravel())
    ])

    # Candidate schedules and nominal Fisher information.
    uniform = nearest_times([0.10, 1.10, 2.00, 3.00])
    early = nearest_times([0.10, 0.20, 0.40, 0.70])
    late = nearest_times([1.10, 1.80, 2.40, 3.00])
    greedy = greedy_schedule(NOMINAL)
    schedule_rows = []
    schedule_specs = {"uniform": uniform, "early-only": early, "late-only": late, "greedy-D-optimal": greedy}
    random_rng = np.random.default_rng(SEED + 17)
    random_logdets = []
    for _ in range(100):
        times = sorted(float(x) for x in random_rng.choice(CANDIDATE_TIMES, size=4, replace=False))
        random_logdets.append(regularized_logdet(schedule_information(times, NOMINAL)))
    schedule_specs["random-mean"] = []
    schedule_logdets = {}
    for name, times in schedule_specs.items():
        if name == "random-mean":
            value = float(np.mean(random_logdets))
            spread = float(np.std(random_logdets, ddof=1))
            used_times = []
        else:
            value = regularized_logdet(schedule_information(times, NOMINAL))
            spread = 0.0
            used_times = times
        schedule_logdets[name] = value
        schedule_rows.append({"schedule": name, "times": ";".join(f"{x:.3f}" for x in used_times) if used_times else "100 seeded random schedules", "logdet_fim": value, "sd_logdet": spread, "rank": matrix_rank(schedule_information(times, NOMINAL)) if used_times else "", "source": "synthetic local Fisher information"})
    write_csv(RES / "schedule_fim.csv", schedule_rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    names = list(schedule_logdets)
    values = [schedule_logdets[n] for n in names]
    errors = [float(np.std(random_logdets, ddof=1)) if n == "random-mean" else 0.0 for n in names]
    ax.bar(np.arange(len(names)), values, yerr=errors, capsize=3, color=[PALETTE[i % len(PALETTE)] for i in range(len(names))], edgecolor="#243746", linewidth=0.5)
    ax.set_xticks(np.arange(len(names)), [n.replace("-", "\n") for n in names])
    ax.set_ylabel("Regularized log determinant")
    ax.set_title("Information depends on when cohorts are sampled", color="#172B3A", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "fig3_information_schedule")

    # Monte Carlo comparison. The prior varies around the nominal log parameters.
    policies = ["adaptive", "uniform", "early-only", "random"]
    sim_rng = np.random.default_rng(SEED + 101)
    estimates = []
    for replicate in range(N_REPLICATES):
        theta_true_rep = NOMINAL + sim_rng.normal(0.0, np.array([0.12, 0.12, 0.22]))
        for policy in policies:
            policy_rng = np.random.default_rng(SEED + 10000 + replicate * 101 + policies.index(policy))
            theta_hat, selected = run_policy(policy, theta_true_rep, policy_rng)
            row = {
                "replicate": replicate,
                "policy": policy,
                "true_log_kappa": float(theta_true_rep[0]),
                "true_log_beta": float(theta_true_rep[1]),
                "true_log_q": float(theta_true_rep[2]),
                "estimated_log_kappa": float(theta_hat[0]),
                "estimated_log_beta": float(theta_hat[1]),
                "estimated_log_q": float(theta_hat[2]),
                "selected_times": ";".join(f"{x:.3f}" for x in selected),
                "source": "synthetic Monte Carlo replicate"
            }
            estimates.append(row)
    write_csv(RES / "estimates.csv", estimates)

    rmse_rows = []
    rng_boot = np.random.default_rng(SEED + 202)
    for policy in policies:
        subset = [row for row in estimates if row["policy"] == policy]
        for true_key, est_key, name in [
            ("true_log_kappa", "estimated_log_kappa", "log_kappa"),
            ("true_log_beta", "estimated_log_beta", "log_beta"),
            ("true_log_q", "estimated_log_q", "log_q"),
        ]:
            errors = np.array([row[est_key] - row[true_key] for row in subset], dtype=float)
            rmse = float(np.sqrt(np.mean(errors**2)))
            boot = []
            for _ in range(300):
                sample = rng_boot.choice(errors, size=len(errors), replace=True)
                boot.append(float(np.sqrt(np.mean(sample**2))))
            low, high = np.quantile(boot, [0.025, 0.975])
            rmse_rows.append({"policy": policy, "parameter": name, "rmse_log_scale": rmse, "ci_low": float(low), "ci_high": float(high), "replicates": len(errors), "source": "synthetic Monte Carlo with bootstrap interval"})
    write_csv(RES / "parameter_rmse.csv", rmse_rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(3)
    width = 0.19
    param_order = ["log_kappa", "log_beta", "log_q"]
    for j, policy in enumerate(policies):
        rows = [next(row for row in rmse_rows if row["policy"] == policy and row["parameter"] == p) for p in param_order]
        vals = [row["rmse_log_scale"] for row in rows]
        lows = [row["rmse_log_scale"] - row["ci_low"] for row in rows]
        highs = [row["ci_high"] - row["rmse_log_scale"] for row in rows]
        ax.bar(x + (j - 1.5) * width, vals, width, yerr=np.array([lows, highs]), capsize=2, label=policy, color=PALETTE[j], edgecolor="#243746", linewidth=0.4)
    ax.set_xticks(x, ["log kappa", "log beta", "log q"])
    ax.set_ylabel("RMSE (log-parameter scale)")
    ax.set_title("Finite-sample parameter error across synthetic replicates", color="#172B3A", weight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "fig4_parameter_rmse")

    # Graph 1: dependencies implied by the model equations.
    graph = nx.DiGraph()
    graph_edges = [
        ("pulse u(t)", "mean m(t)"), ("gain beta", "mean m(t)"),
        ("relaxation kappa", "mean m(t)"), ("relaxation kappa", "variance v"),
        ("diffusion q", "variance v"), ("measurement r", "variance v"),
        ("sampling time t", "mean m(t)")
    ]
    graph.add_edges_from(graph_edges)
    pos = {
        "pulse u(t)": (-1.5, 1.0), "gain beta": (-1.5, 0.2), "relaxation kappa": (0.0, 0.6),
        "diffusion q": (-1.5, -0.6), "measurement r": (0.0, -0.6),
        "sampling time t": (-1.5, -1.4), "mean m(t)": (1.6, 0.6), "variance v": (1.6, -0.6)
    }
    node_colors = [("#E8F2F4" if n in {"mean m(t)", "variance v"} else "#F4EEE2") for n in graph.nodes()]
    draw_network_graph(graph, pos, "fig5_model_dependency_graph", "Equation-derived dependency graph", node_colors=node_colors, figsize=(9.5, 5.2))

    # Graph 2: parameter coupling from normalized Fisher information.
    info = schedule_information(greedy, NOMINAL)
    diag = np.sqrt(np.maximum(np.diag(info), 1e-12))
    corr = info / np.outer(diag, diag)
    corr_graph = nx.Graph()
    corr_labels = {"log_kappa": "log kappa", "log_beta": "log beta", "log_q": "log q"}
    corr_graph.add_nodes_from(corr_labels)
    for i, a in enumerate(corr_labels):
        for j, b in enumerate(corr_labels):
            if j > i:
                weight = abs(float(corr[i, j]))
                if weight > 0.03:
                    corr_graph.add_edge(a, b, weight=weight)
    corr_pos = {"log_kappa": (-1.0, 0.5), "log_beta": (1.0, 0.5), "log_q": (0.0, -0.8)}
    edge_widths = [1.0 + 4.0 * corr_graph[a][b]["weight"] for a, b in corr_graph.edges()]
    draw_network_graph(corr_graph, corr_pos, "fig6_information_coupling_graph", "Local parameter-coupling graph", node_colors=["#E8F2F4"] * 3, edge_widths=edge_widths, figsize=(6.5, 4.5))

    # Flow 1: experimental pipeline.
    nodes = {
        "a": "Specify pulse\nand assumptions", "b": "Randomize\nindependent cohorts",
        "c": "Measure at\nselected times", "d": "Estimate means\nand variances",
        "e": "Check rank,\nresiduals, controls", "f": "Report identified\nsets and limits"
    }
    positions = {"a": (0.0, 0.0), "b": (2.1, 0.0), "c": (4.2, 0.0), "d": (6.3, 0.0), "e": (8.4, 0.0), "f": (10.5, 0.0)}
    draw_flow(nodes, [(a, b, "") for a, b in zip(list(nodes)[:-1], list(nodes)[1:])], positions, "fig7_assay_pipeline", "Pulse-resolved assay and analysis pipeline", figsize=(13.2, 3.8))

    # Flow 2: sequential design loop with feedback.
    nodes2 = {
        "a": "Initialize prior\nand constraints", "b": "Choose next time\nby log-det FIM",
        "c": "Collect a new\nindependent cohort", "d": "Update MAP\nparameter estimate",
        "e": "Budget or precision\ntarget reached"
    }
    positions2 = {"a": (0.0, 0.5), "b": (2.5, 0.5), "c": (5.0, 0.5), "d": (7.5, 0.5), "e": (10.0, 0.5)}
    arrows2 = [("a", "b", ""), ("b", "c", ""), ("c", "d", ""), ("d", "e", ""), ("d", "b", "update estimate")]
    draw_flow(nodes2, arrows2, positions2, "fig8_adaptive_loop", "Sequential information-aware sampling", figsize=(12.8, 3.8))

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "model": "controlled scalar Ornstein-Uhlenbeck process",
        "nominal_parameters": {"kappa": float(np.exp(NOMINAL[0]),), "beta": float(np.exp(NOMINAL[1])), "q": float(np.exp(NOMINAL[2]))},
        "pulse_amplitude_normalized": PULSE_AMPLITUDE,
        "pulse_duration_normalized": PULSE_DURATION,
        "measurement_variance": MEASUREMENT_VARIANCE,
        "cells_per_time": N_CELLS,
        "monte_carlo_replicates": N_REPLICATES,
        "policies": policies,
        "candidate_times": CANDIDATE_TIMES.tolist(),
        "all_observations_synthetic": True,
        "external_data_downloaded": False
    }
    (RES / "simulation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    data_manifest = {
        "generator": "verify_model.py",
        "seed": SEED,
        "figures": {
            "fig1_mean_response": ["results/mean_response.csv"],
            "fig2_variance_confounding": ["results/variance_ridge.csv"],
            "fig3_information_schedule": ["results/schedule_fim.csv"],
            "fig4_parameter_rmse": ["results/parameter_rmse.csv", "results/estimates.csv"],
            "fig5_model_dependency_graph": ["verify_model.py model equations"],
            "fig6_information_coupling_graph": ["results/schedule_fim.csv"],
            "fig7_assay_pipeline": ["verify_model.py method definition"],
            "fig8_adaptive_loop": ["verify_model.py greedy policy"]
        }
    }
    (RES / "figure_data_manifest.json").write_text(json.dumps(data_manifest, indent=2) + "\n", encoding="utf-8")
    return {"theta_true": theta_true, "greedy": greedy, "schedules": schedule_specs, "schedule_logdets": schedule_logdets, "rmse_rows": rmse_rows, "estimates": estimates}


def write_generated_results(data):
    rows = []
    rmse_rows = data["rmse_rows"]
    for policy, schedule_name in [
        ("adaptive", "greedy-D-optimal"),
        ("uniform", "uniform"),
        ("early-only", "early-only"),
        ("random", "random-mean")
    ]:
        logdet = data["schedule_logdets"][schedule_name]
        vals = {row["parameter"]: row["rmse_log_scale"] for row in rmse_rows if row["policy"] == policy}
        rows.append((policy.replace("-", "\\texttt{") + ("}" if "-" in policy else ""), logdet, vals["log_kappa"], vals["log_beta"], vals["log_q"]))
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Current-run information and estimation results from the seeded synthetic benchmark. RMSE is measured on the log-parameter scale; values are generated by the released script.}",
        r"\label{tab:results}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Policy & $\log\det(\mathcal{I}+\lambda I)$ & RMSE$_{\log\kappa}$ & RMSE$_{\log\beta}$ & RMSE$_{\log q}$ \\",
        r"\midrule"
    ]
    for name, logdet, a, b, c in rows:
        safe_name = name.replace("adaptive", r"\texttt{adaptive}").replace("uniform", r"\texttt{uniform}").replace("early\\texttt{only}", r"\texttt{early-only}").replace("random", r"\texttt{random}")
        lines.append(f"{safe_name} & {logdet:.3f} & {a:.3f} & {b:.3f} & {c:.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.96\linewidth}\footnotesize The information objective is evaluated at the nominal parameter vector; the adaptive policy updates its estimate during sequential sampling. Random-design information is averaged over 100 seeded schedules. No biological observations are included.\end{minipage}",
        r"\end{table}"
    ]
    (RES / "generated_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_checks(data):
    checks = []
    def record(name, passed, detail):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    theta = np.log(np.array([0.80, 1.10, 0.22]))
    record("zero_initial_mean", abs(mean_response(0.0, theta)) < 1e-12, f"m(0)={mean_response(0.0, theta):.3e}")
    record("post_pulse_decay", mean_response(2.4, theta) < mean_response(1.2, theta), "post-pulse mean decreases for positive parameters")
    # Exact ratio identity.
    t1, t2 = 1.2, 2.4
    m1, m2 = mean_response(t1, theta), mean_response(t2, theta)
    k_recovered = (math.log(m1) - math.log(m2)) / (t2 - t1)
    record("exact_relaxation_ratio", abs(k_recovered - unpack(theta)[0]) < 1e-12, f"true={unpack(theta)[0]:.8f}, recovered={k_recovered:.8f}")
    # Variance-only confounding.
    th_a = np.log(np.array([0.60, 1.10, 0.18]))
    th_b = np.log(np.array([1.20, 1.10, 0.36]))
    va, vb = observation_variance(th_a), observation_variance(th_b)
    record("stationary_variance_confounds_q_and_kappa", abs(va-vb) < 1e-12, f"matched variances={va:.8f}, {vb:.8f}")
    # Fisher information properties.
    fim_single = fisher_information_at_time(1.2, theta)
    fim_multi = schedule_information([1.2, 2.4], theta)
    eig_single = np.linalg.eigvalsh(fim_single)
    eig_multi = np.linalg.eigvalsh(fim_multi)
    record("fisher_information_psd", float(np.min(eig_multi)) >= -1e-8, f"minimum eigenvalue={np.min(eig_multi):.3e}")
    record("single_time_rank_deficient", matrix_rank(fim_single) <= 2, f"rank={matrix_rank(fim_single)} for 3 parameters")
    record("two_post_pulse_times_full_rank", matrix_rank(fim_multi) == 3, f"rank={matrix_rank(fim_multi)} for 3 parameters")
    # Safety constraint in normalized model units.
    max_mean = max(abs(mean_response(t, NOMINAL)) for t in CANDIDATE_TIMES)
    record("normalized_amplitude_constraint", max_mean < 0.80, f"max |m(t)|={max_mean:.4f} < 0.80")
    # Invalid physical inputs must fail.
    invalid_rejected = False
    try:
        observation_variance(np.array([np.log(0.8), np.log(1.1), np.nan]))
    except (ValueError, FloatingPointError):
        invalid_rejected = True
    record("invalid_parameter_rejection", invalid_rejected, "nonpositive diffusion intensity is rejected")
    # Output and estimate checks.
    all_finite = all(np.isfinite(float(row["estimated_log_kappa"])) and np.isfinite(float(row["estimated_log_beta"])) and np.isfinite(float(row["estimated_log_q"])) for row in data["estimates"])
    record("finite_monte_carlo_estimates", all_finite, f"{len(data['estimates'])} fitted replicate-policy rows")
    expected_figs = [f"fig{i}_{stem}.pdf" for i, stem in [
        (1, "mean_response"), (2, "variance_confounding"), (3, "information_schedule"), (4, "parameter_rmse"),
        (5, "model_dependency_graph"), (6, "information_coupling_graph"), (7, "assay_pipeline"), (8, "adaptive_loop")
    ]]
    assets_ok = all((FIG / name).is_file() and (FIG / name).stat().st_size > 1000 for name in expected_figs)
    svg_ok = all((FIG / name.replace(".pdf", ".svg")).is_file() and (FIG / name.replace(".pdf", ".svg")).stat().st_size > 1000 for name in expected_figs)
    record("figure_assets_generated", assets_ok and svg_ok, "eight PDF vectors and eight SVG vectors exist")
    generated_table = RES / "generated_results.tex"
    record("generated_results_table", generated_table.is_file() and generated_table.stat().st_size > 100, "current-run results table exists")
    failed = [x for x in checks if x["status"] != "PASS"]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "seed": SEED,
        "model": "controlled scalar Ornstein-Uhlenbeck process",
        "all_data_synthetic": True,
        "external_data_downloaded": False,
        "checks": checks,
        "failed_checks": failed,
        "schedule": {"greedy_d_optimal_times": data["greedy"]},
        "interpretation": "Passing checks establish only algebraic/numerical consistency of the specified synthetic model, not empirical biological validity."
    }
    (RES / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "networkx": nx.__version__,
        "pypdf": "5.9.0 (PDF QA dependency; version checked by deploy workflow)",
        "seed": SEED,
        "latex": "pdflatex + BibTeX (version recorded by build workflow)",
        "data_source": "synthetic model-generated observations only"
    }
    (RES / "environment_report.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (RES / "verification_summary.txt").write_text(
        f"Status: {report['status']}\nTimestamp UTC: {report['timestamp_utc']}\nSeed: {SEED}\n"
        + "\n".join(f"{x['status']}: {x['name']} — {x['detail']}" for x in checks)
        + "\nLimit: no biological dataset was analyzed; tests do not validate a biological mechanism.\n",
        encoding="utf-8"
    )
    if failed:
        raise SystemExit(1)


def main():
    print("Pulse-Resolved Identifiability — synthetic verification")
    print(f"Seed={SEED}; cells/time={N_CELLS}; Monte Carlo replicates={N_REPLICATES}")
    data = generate_figures_and_data()
    write_generated_results(data)
    run_checks(data)
    print("All required algebraic, numerical, and output checks passed.")
    print("All reported observations are synthetic; no biological dataset was analyzed.")


if __name__ == "__main__":
    main()
