#!/usr/bin/env python3
"""Reproducible mathematical and synthetic-network checks for the MCD framework.

This program does NOT simulate a planarian, ageing, cancer, stem-cell migration,
non-Hermitian skin effects, or a clinical intervention. It verifies algebraic
identities and runs transparent synthetic graph/network experiments.

Dependencies: numpy, scipy, networkx, matplotlib.
Run: python verify_model.py
Outputs are written relative to this file under figures/ and results/.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy
from scipy.optimize import nnls
from scipy.linalg import eigh
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261107
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StructuralBenchmark:
    graph: nx.Graph
    left: tuple[int, ...]
    right: tuple[int, ...]
    bridges: tuple[int, ...]
    source: int
    sink: int


def build_structural_benchmark() -> StructuralBenchmark:
    """Two 15-node cliques connected only through five shared relay vertices.

    The relay vertices connect to every cell in both modules. Source and sink
    are protected measurement anchors (one per module); all other module nodes
    are eligible for dispersed lesions. This is a deliberately stylized,
    highly redundant graph, not a reconstructed biological tissue.
    """
    G = nx.Graph()
    left = tuple(range(15))
    right = tuple(range(15, 30))
    bridges = tuple(range(30, 35))
    G.add_nodes_from(left, compartment="A", kind="cell")
    G.add_nodes_from(right, compartment="B", kind="cell")
    G.add_nodes_from(bridges, compartment="interface", kind="relay")
    G.add_edges_from((u, v) for i, u in enumerate(left) for v in left[i + 1 :])
    G.add_edges_from((u, v) for i, u in enumerate(right) for v in right[i + 1 :])
    for b in bridges:
        for u in left:
            G.add_edge(b, u, weight=1.0, kind="relay-link")
        for v in right:
            G.add_edge(b, v, weight=1.0, kind="relay-link")
    for u, v in G.edges:
        G[u][v].setdefault("weight", 1.0)
        G[u][v].setdefault("kind", "within-module")
    return StructuralBenchmark(G, left, right, bridges, left[0], right[0])


def weighted_vertex_mincut(
    G: nx.Graph,
    source: int,
    sink: int,
    vertex_costs: dict[int, float] | None = None,
) -> tuple[float, set[int]]:
    """Exact node-capacitated s-t cut by node splitting and max-flow/min-cut.

    The source and sink vertices are protected (capacity M). Every other
    vertex v is split into v_in -> v_out with capacity c_v. Each undirected
    edge becomes two directed infinite-capacity arcs. The set of split-node
    arcs crossing the minimum cut yields the removable vertex set.
    """
    if source not in G or sink not in G or source == sink:
        raise ValueError("source and sink must be distinct vertices in G")
    if not nx.has_path(G, source, sink):
        return 0.0, set()
    big = 1e9
    costs = vertex_costs or {}
    H = nx.DiGraph()
    ss, tt = ("__super_source__", "__super_sink__")
    for v in G.nodes:
        c = big if v in (source, sink) else float(costs.get(v, 1.0))
        if not np.isfinite(c) or c <= 0:
            raise ValueError("all vertex capacities must be finite and positive")
        H.add_edge((v, "in"), (v, "out"), capacity=c)
    for u, v in G.edges:
        H.add_edge((u, "out"), (v, "in"), capacity=big)
        H.add_edge((v, "out"), (u, "in"), capacity=big)
    H.add_edge(ss, (source, "in"), capacity=big)
    H.add_edge((sink, "out"), tt, capacity=big)
    value, (reachable, nonreachable) = nx.minimum_cut(H, ss, tt, capacity="capacity")
    cut_nodes: set[int] = set()
    for v in G.nodes:
        if (v, "in") in reachable and (v, "out") in nonreachable:
            if v not in (source, sink):
                cut_nodes.add(v)
    return float(value), cut_nodes


def sheaf_matrix_cycle_test() -> dict[str, float | int | bool]:
    """Check exactness and harmonic residual on a 2D-stalk sheaf over C4."""
    n, d = 4, 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    D = np.zeros((len(edges) * d, n * d), dtype=float)
    for ei, (u, v) in enumerate(edges):
        # Trivial cellular sheaf: both restriction maps are identity maps.
        D[ei*d:(ei+1)*d, u*d:(u+1)*d] = np.eye(d)
        D[ei*d:(ei+1)*d, v*d:(v+1)*d] = -np.eye(d)
    x = np.array([0.2, -0.5, 0.7, 0.1, -0.2, 0.4, 0.8, -0.3])
    z_exact = D @ x
    projector = D @ np.linalg.pinv(D)
    exact_residual = np.linalg.norm((np.eye(D.shape[0]) - projector) @ z_exact)
    rng = np.random.default_rng(SEED + 1)
    z_observed = rng.normal(size=D.shape[0])
    harmonic_residual = (np.eye(D.shape[0]) - projector) @ z_observed
    harmonic_norm = np.linalg.norm(harmonic_residual)
    rank = int(np.linalg.matrix_rank(D))
    h1_dim = int(D.shape[0] - rank)  # no 2-cells: delta^1 = 0
    return {
        "coboundary_rank": rank,
        "first_cohomology_dimension_for_C4_trivial_R2_sheaf": h1_dim,
        "exact_syndrome_harmonic_residual": float(exact_residual),
        "random_observation_harmonic_residual": float(harmonic_norm),
        "exact_cochain_projects_to_zero_harmonic_part": bool(exact_residual < 1e-10),
        "nonexact_observation_has_nonzero_harmonic_part": bool(harmonic_norm > 1e-5),
    }


def pinned_laplacian(G: nx.Graph, anchor: int, weight: str = "weight") -> np.ndarray:
    nodes = list(G.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    L = nx.laplacian_matrix(G, nodelist=nodes, weight=weight).toarray().astype(float)
    P = np.zeros_like(L)
    P[idx[anchor], idx[anchor]] = 1.0
    return L + P


def structural_experiments() -> tuple[dict, list[dict], list[dict]]:
    b = build_structural_benchmark()
    G = b.graph
    mcd, cut_set = weighted_vertex_mincut(G, b.source, b.sink)
    assert abs(mcd - 5.0) < 1e-9, (mcd, cut_set)
    assert set(b.bridges) == cut_set, (cut_set, b.bridges)

    rng = np.random.default_rng(SEED)
    candidates = [v for v in G.nodes if v not in (b.source, b.sink)]
    n_trials = 10_000
    uniform_fail = 0
    balanced_fail = 0
    for _ in range(n_trials):
        # Primary comparison: 20 vertices sampled uniformly from all eligible
        # non-terminal vertices, including the five relay cells.
        damaged = rng.choice(candidates, size=20, replace=False)
        H = G.copy(); H.remove_nodes_from(damaged)
        if not nx.has_path(H, b.source, b.sink):
            uniform_fail += 1
        # Secondary control: 10 lesions per module, with relay vertices spared.
        damaged_left = rng.choice([v for v in b.left if v != b.source], size=10, replace=False)
        damaged_right = rng.choice([v for v in b.right if v != b.sink], size=10, replace=False)
        H2 = G.copy(); H2.remove_nodes_from(list(damaged_left) + list(damaged_right))
        if not nx.has_path(H2, b.source, b.sink):
            balanced_fail += 1
    p_uniform = uniform_fail / n_trials
    p_balanced = balanced_fail / n_trials
    p_exact = math.comb(28, 15) / math.comb(33, 20)

    targeted = G.copy()
    targeted.remove_nodes_from(b.bridges)
    targeted_fail = not nx.has_path(targeted, b.source, b.sink)

    # Wilson score intervals for observed binomial proportions.
    z = 1.959963984540054
    def wilson(p: float) -> list[float]:
        den = 1 + z*z/n_trials
        center = (p + z*z/(2*n_trials))/den
        half = z*math.sqrt(p*(1-p)/n_trials + z*z/(4*n_trials*n_trials))/den
        return [max(0.0, center-half), min(1.0, center+half)]

    damage_rows = [
        {"protocol": "20 uniformly sampled eligible vertices", "trials": n_trials,
         "failures": uniform_fail, "failure_fraction": p_uniform,
         "wilson95_low": wilson(p_uniform)[0], "wilson95_high": wilson(p_uniform)[1],
         "exact_probability": p_exact,
         "interpretation": "synthetic; eligible set includes relay vertices"},
        {"protocol": "20 balanced module-cell lesions (10 per module)", "trials": n_trials,
         "failures": balanced_fail, "failure_fraction": p_balanced,
         "wilson95_low": wilson(p_balanced)[0], "wilson95_high": wilson(p_balanced)[1],
         "exact_probability": 0.0,
         "interpretation": "synthetic; relay vertices and anchors excluded by design"},
        {"protocol": "5 targeted relay-node lesions", "trials": 1,
         "failures": int(targeted_fail), "failure_fraction": float(targeted_fail),
         "wilson95_low": None, "wilson95_high": None, "exact_probability": 1.0,
         "interpretation": "deterministic removal of exact minimum cut"},
    ]

    # Vary the number of removed relay nodes; compute pinned spectral gap.
    spectral_rows = []
    for k in range(6):
        H = G.copy()
        H.remove_nodes_from(b.bridges[:k])
        A = pinned_laplacian(H, b.source)
        eigvals = np.linalg.eigvalsh(A)
        lam = float(max(0.0, eigvals[0]))
        spectral_rows.append({"removed_relays": k, "remaining_relays": 5-k,
                              "lambda_min_pinned_laplacian": lam,
                              "source_sink_connected": nx.has_path(H, b.source, b.sink)})
    weight_sensitivity = []
    for relay_cost in (0.25, 0.5, 1.0, 2.0, 4.0):
        costs = {v: relay_cost for v in b.bridges}
        cap, chosen = weighted_vertex_mincut(G, b.source, b.sink, costs)
        weight_sensitivity.append({"relay_vertex_cost": relay_cost, "mcd_weighted": cap,
                                   "selected_cut_size": len(chosen), "selected_cut": sorted(chosen)})

    metrics = {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "mcd_weighted_vertex_cut": mcd, "minimum_cut_nodes": sorted(cut_set),
        "uniform_20_vertex_failures": uniform_fail, "uniform_trials": n_trials,
        "uniform_20_vertex_failure_fraction": p_uniform,
        "uniform_20_vertex_exact_failure_probability": p_exact,
        "uniform_20_vertex_wilson95": wilson(p_uniform),
        "balanced_module_20_vertex_failures": balanced_fail,
        "balanced_module_failure_fraction": p_balanced,
        "balanced_module_wilson95": wilson(p_balanced),
        "targeted_cut_disconnects": bool(targeted_fail),
        "relay_cost_sensitivity": weight_sensitivity,
        "synthetic_only": True,
    }
    return metrics, damage_rows, spectral_rows


def inference_graph() -> tuple[nx.Graph, list[tuple[int, int]], np.ndarray]:
    """Smaller synthetic graph for identifiable edge-conductance recovery."""
    G = nx.Graph()
    left = list(range(6))
    right = list(range(6, 12))
    bridges = list(range(12, 15))
    G.add_nodes_from(range(15))
    G.add_edges_from((u, v) for i, u in enumerate(left) for v in left[i+1:])
    G.add_edges_from((u, v) for i, u in enumerate(right) for v in right[i+1:])
    for h in bridges:
        for u in left + right:
            G.add_edge(h, u)
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges)
    true_w = np.array([1.0 if (u < 12 and v < 12) else 0.65 for u, v in edges])
    return G, edges, true_w


def design_matrix(states: np.ndarray, n_nodes: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    rows = states.shape[0] * n_nodes
    A = np.zeros((rows, len(edges)), dtype=float)
    for s_idx, x in enumerate(states):
        for node in range(n_nodes):
            row = s_idx*n_nodes + node
            for e_idx, (u, v) in enumerate(edges):
                if node == u:
                    A[row, e_idx] = x[u] - x[v]
                elif node == v:
                    A[row, e_idx] = x[v] - x[u]
    return A


def inverse_experiment() -> tuple[list[dict], dict]:
    """NNLS recovery of conductances from synthetic diffusion snapshots."""
    rng = np.random.default_rng(SEED + 100)
    G, edges, w_true = inference_graph()
    n_nodes = G.number_of_nodes()
    A = design_matrix(np.zeros((1, n_nodes)), n_nodes, edges)  # shape template
    noise_levels = [0.0, 0.01, 0.03, 0.07, 0.12]
    n_snapshots = 80
    replicates = 5
    rows: list[dict] = []
    all_results: dict[str, list[float]] = {}
    for sigma in noise_levels:
        errs, rmses = [], []
        for rep in range(replicates):
            states = rng.normal(size=(n_snapshots, n_nodes))
            X = design_matrix(states, n_nodes, edges)
            y = X @ w_true
            noisy = y + rng.normal(scale=sigma, size=y.shape)
            w_hat, rnorm = nnls(X, noisy, maxiter=20*len(edges))
            rel = float(np.linalg.norm(w_hat-w_true)/np.linalg.norm(w_true))
            rmse = float(np.sqrt(np.mean((w_hat-w_true)**2)))
            errs.append(rel); rmses.append(rmse)
            rows.append({"noise_sd": sigma, "replicate": rep, "n_snapshots": n_snapshots,
                         "n_nodes": n_nodes, "n_edges": len(edges),
                         "relative_parameter_error": rel, "parameter_rmse": rmse,
                         "residual_norm": float(rnorm),
                         "minimum_estimated_conductance": float(np.min(w_hat)),
                         "maximum_estimated_conductance": float(np.max(w_hat))})
        all_results[str(sigma)] = errs
    probe_states = rng.normal(size=(n_snapshots, n_nodes))
    probe_A = design_matrix(probe_states, n_nodes, edges)
    singular_values = np.linalg.svd(probe_A, compute_uv=False)
    design_rank = int(np.linalg.matrix_rank(probe_A))
    design_condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    summary = {"nodes": n_nodes, "edges": len(edges), "design_rank": design_rank,
               "design_dimension": len(edges), "design_condition_number": design_condition,
               "snapshots_per_fit": n_snapshots, "replicates_per_noise": replicates,
               "noise_levels": noise_levels,
               "relative_error_mean_by_noise": {str(s): float(np.mean(all_results[str(s)])) for s in noise_levels},
               "relative_error_sd_by_noise": {str(s): float(np.std(all_results[str(s)], ddof=1)) for s in noise_levels},
               "synthetic_only": True}
    return rows, summary


def recovery_traces() -> list[dict]:
    b = build_structural_benchmark()
    rows: list[dict] = []
    for k in (0, 4, 5):
        H = b.graph.copy()
        H.remove_nodes_from(b.bridges[:k])
        A = pinned_laplacian(H, b.source)
        vals, vecs = eigh(A)
        rng = np.random.default_rng(SEED+500)
        x0 = np.zeros(A.shape[0])
        nodes = list(H.nodes)
        idx = {v:i for i,v in enumerate(nodes)}
        for v in b.right:
            if v in idx:
                x0[idx[v]] = 1.0
        x0 += rng.normal(scale=0.01, size=x0.shape)
        coeff = vecs.T @ x0
        right_idx = [idx[v] for v in b.right if v in idx]
        for t in np.linspace(0, 12, 61):
            xt = vecs @ (np.exp(-vals*t) * coeff)
            right_mean = float(np.mean(xt[right_idx]))
            rows.append({"removed_relays":k,"time":float(t),"right_module_mean_state":right_mean,
                         "right_module_abs_mean":abs(right_mean),"lambda_min":float(max(0.0,vals[0]))})
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def make_figures(metrics: dict, damage_rows: list[dict], spectral_rows: list[dict],
                 inference_rows: list[dict], trace_rows: list[dict], sheaf: dict) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
                         "figure.dpi": 140, "savefig.dpi": 220,
                         "font.family": "DejaVu Sans"})
    # Plot 1: lesion topology / failure rates.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [r["protocol"] for r in damage_rows]
    vals = [r["failure_fraction"] for r in damage_rows]
    ax.bar(["20 uniform\neligible lesions", "20 balanced\nmodule lesions", "5 targeted\nrelay lesions"], vals, width=.55)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction disconnected")
    ax.set_title("Synthetic benchmark: lesion placement dominates cell count")
    ax.text(0.02, .95, "Stylized two-module graph; not biological data", transform=ax.transAxes,
            va="top", color="dimgray")
    for i,v in enumerate(vals): ax.text(i, v+0.025, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(FIG/"fig1_lesion_topology.png"); fig.savefig(FIG/"fig1_lesion_topology.pdf"); plt.close(fig)

    # Plot 2: pinned spectral gap vs bridge deletion count.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ks=[r["removed_relays"] for r in spectral_rows]
    lam=[r["lambda_min_pinned_laplacian"] for r in spectral_rows]
    ax.plot(ks, lam, marker="o", linewidth=2)
    ax.set_xlabel("Relay nodes removed"); ax.set_ylabel(r"Smallest eigenvalue of $L+P$")
    ax.set_title("Loss of the final relay creates an unpinned component")
    ax.set_xticks(range(6)); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/"fig2_pinned_spectral_gap.png"); fig.savefig(FIG/"fig2_pinned_spectral_gap.pdf"); plt.close(fig)

    # Plot 3: NNLS edge conductance estimation uncertainty vs noise.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ns=sorted({float(r["noise_sd"]) for r in inference_rows})
    means=[]; sds=[]
    for s in ns:
        v=[float(r["relative_parameter_error"]) for r in inference_rows if float(r["noise_sd"])==s]
        means.append(float(np.mean(v))); sds.append(float(np.std(v,ddof=1)))
    ax.errorbar(ns,means,yerr=sds,marker="o",capsize=4,linewidth=1.8)
    ax.set_xlabel("Gaussian derivative-noise standard deviation")
    ax.set_ylabel("Relative conductance-estimation error")
    ax.set_title("Physics-constrained NNLS recovery under synthetic noise")
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/"fig3_nnls_noise_sensitivity.png"); fig.savefig(FIG/"fig3_nnls_noise_sensitivity.pdf"); plt.close(fig)

    # Plot 4: recovery trajectories.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k in (0,4,5):
        rr=[r for r in trace_rows if r["removed_relays"]==k]
        ax.plot([r["time"] for r in rr],[r["right_module_abs_mean"] for r in rr],label=f"{k} relay(s) removed")
    ax.set_xlabel("Dimensionless time"); ax.set_ylabel("Absolute mean state in right module")
    ax.set_title("Pinned diffusion: connectivity controls relaxation to the anchor")
    ax.legend(frameon=False); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/"fig4_pinned_diffusion_traces.png"); fig.savefig(FIG/"fig4_pinned_diffusion_traces.pdf"); plt.close(fig)

    # Graph 1: structural network.
    b=build_structural_benchmark(); G=b.graph
    pos={}
    for i,v in enumerate(b.left): pos[v]=(0.0, i)
    for i,v in enumerate(b.right): pos[v]=(4.0, i)
    for i,v in enumerate(b.bridges): pos[v]=(2.0, 2.0+i*2.4)
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color="#aab4c3",width=.65,alpha=.38)
    nx.draw_networkx_nodes(G,pos,nodelist=list(b.left),node_color="#4c78a8",node_size=80,ax=ax,label="Module A")
    nx.draw_networkx_nodes(G,pos,nodelist=list(b.right),node_color="#59a14f",node_size=80,ax=ax,label="Module B")
    nx.draw_networkx_nodes(G,pos,nodelist=list(b.bridges),node_color="#e45756",node_size=170,ax=ax,label="Relay / cut set")
    nx.draw_networkx_nodes(G,pos,nodelist=[b.source,b.sink],node_color="#f2cf5b",node_size=220,edgecolors="black",linewidths=1.2,ax=ax)
    ax.set_title("Stylized two-module graph with a five-node relay cut")
    ax.text(.01,.01,"Complete subgraphs and all-to-all relay links are deliberately idealized.",transform=ax.transAxes,fontsize=8)
    ax.legend(loc="upper center",ncol=4,frameon=False); ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG/"fig5_structural_network.pdf"); fig.savefig(FIG/"fig5_structural_network.png"); plt.close(fig)

    # Graph 2: sheaf stalks, restrictions and observed edge cochain.
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    xy=np.array([[0,0],[2,0],[2,1.7],[0,1.7]])
    for (u,v) in [(0,1),(1,2),(2,3),(3,0)]:
        ax.plot([xy[u,0],xy[v,0]],[xy[u,1],xy[v,1]],color="#6b7280",lw=2)
    for i,(x,y) in enumerate(xy):
        ax.scatter([x],[y],s=900,color="#4c78a8",zorder=3)
        ax.text(x,y,rf"$\mathcal{{F}}(v_{i})$",ha="center",va="center",color="white",fontsize=9,zorder=4)
    ax.text(1,-.27,r"$\mathcal{F}(e_{01})$ with restrictions $R_{0,e},R_{1,e}$",ha="center",fontsize=9)
    ax.text(2.25,.85,r"$z_e$",fontsize=10)
    ax.text(-.3,.85,r"$\delta^0 x$",fontsize=10,rotation=90,va="center")
    ax.set_xlim(-.7,3.6); ax.set_ylim(-.55,2.15); ax.axis("off")
    ax.set_title("Cellular-sheaf bookkeeping: vertex states restrict to edge stalks")
    fig.tight_layout(); fig.savefig(FIG/"fig6_sheaf_stalk_graph.pdf"); fig.savefig(FIG/"fig6_sheaf_stalk_graph.png"); plt.close(fig)

    # Flow 1: computational pipeline.
    fig,ax=plt.subplots(figsize=(8.2,3.4)); ax.axis("off")
    boxes=[("Imaging / perturbation\nmeasurements",.08), ("Infer graph +\nrestriction maps",.29), ("Estimate conductance\nwith constraints",.50), ("Compute cut +\nspectral diagnostics",.71), ("Prospective tests +\nuncertainty",.91)]
    for label,x in boxes:
        ax.text(x,.55,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.55",fc="#eef3f8",ec="#47627a"),transform=ax.transAxes,fontsize=9)
    for i in range(len(boxes)-1):
        ax.annotate("",xy=(boxes[i+1][1]-.065,.55),xytext=(boxes[i][1]+.065,.55),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.5,color="#47627a"))
    ax.text(.5,.12,"Every stage carries calibration, missingness, and model uncertainty forward.",ha="center",transform=ax.transAxes,fontsize=9)
    ax.set_title("Proposed measurement-to-inference workflow")
    fig.tight_layout(); fig.savefig(FIG/"fig7_inference_flow.pdf"); fig.savefig(FIG/"fig7_inference_flow.png"); plt.close(fig)

    # Flow 2: falsifiable future experiment.
    fig,ax=plt.subplots(figsize=(8.2,4.0)); ax.axis("off")
    items=[("Pre-register\nsource / target regions",.50,.86),
           ("Measure baseline\nvoltage + connectivity",.22,.62),
           ("Apply graded, spatially\nregistered perturbations",.78,.62),
           ("Track state recovery\nand anatomy",.50,.37),
           ("Compare measured cut,\nspectral gap, and outcomes",.50,.12)]
    for label,x,y in items:
        ax.text(x,y,label,ha="center",va="center",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.55",fc="#f5f2ea",ec="#88784f"),fontsize=9)
    ax.annotate("",xy=(.32,.54),xytext=(.45,.75),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.3))
    ax.annotate("",xy=(.68,.54),xytext=(.55,.75),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.3))
    ax.annotate("",xy=(.48,.25),xytext=(.30,.52),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.3))
    ax.annotate("",xy=(.52,.25),xytext=(.70,.52),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.3))
    ax.set_title("Prospective biological validation plan (not yet performed)")
    fig.tight_layout(); fig.savefig(FIG/"fig8_validation_flow.pdf"); fig.savefig(FIG/"fig8_validation_flow.png"); plt.close(fig)


def run_checks() -> dict:
    checks = []
    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            raise AssertionError(f"{name}: {detail}")

    # Generic incidence/sheaf Laplacian: symmetry and PSD.
    b = build_structural_benchmark()
    nodes = list(b.graph.nodes); idx = {v:i for i,v in enumerate(nodes)}
    B = np.zeros((b.graph.number_of_edges(), len(nodes)))
    for ei,(u,v) in enumerate(b.graph.edges): B[ei,idx[u]]=1.0; B[ei,idx[v]]=-1.0
    W = np.diag([b.graph[u][v]["weight"] for u,v in b.graph.edges])
    L = B.T @ W @ B
    symmetry_error = float(np.linalg.norm(L-L.T))
    min_eig = float(np.linalg.eigvalsh(L)[0])
    record("weighted_laplacian_symmetry", symmetry_error < 1e-10, f"||L-L^T||_F={symmetry_error:.3e}")
    record("weighted_laplacian_positive_semidefinite", min_eig > -1e-9, f"lambda_min={min_eig:.3e}")
    # Energy identity.
    rng=np.random.default_rng(SEED+7); x=rng.normal(size=len(nodes))
    e1=float(x@L@x); e2=float(np.sum(np.diag(W)*(B@x)**2))
    record("dirichlet_energy_identity", abs(e1-e2)<1e-9, f"quadratic={e1:.12g}; edge_sum={e2:.12g}")

    sh=sheaf_matrix_cycle_test()
    record("exact_syndrome_is_cohomologically_trivial", bool(sh["exact_cochain_projects_to_zero_harmonic_part"]),
           f"harmonic residual={sh['exact_syndrome_harmonic_residual']:.3e}")
    record("observed_nonexact_cochain_has_harmonic_component", bool(sh["nonexact_observation_has_nonzero_harmonic_part"]),
           f"harmonic residual={sh['random_observation_harmonic_residual']:.3e}")
    record("cycle_rank_identity", sh["first_cohomology_dimension_for_C4_trivial_R2_sheaf"]==2,
           f"computed dim H1={sh['first_cohomology_dimension_for_C4_trivial_R2_sheaf']}")

    metrics, damage_rows, spectral_rows = structural_experiments()
    record("exact_minimum_vertex_cut", abs(metrics["mcd_weighted_vertex_cut"]-5.0)<1e-9,
           f"cut capacity={metrics['mcd_weighted_vertex_cut']}; nodes={metrics['minimum_cut_nodes']}")
    record("uniform_random_lesions_match_hypergeometric_baseline",
           abs(metrics["uniform_20_vertex_failure_fraction"]-metrics["uniform_20_vertex_exact_failure_probability"])<0.02,
           f"observed={metrics['uniform_20_vertex_failure_fraction']:.4f}; exact={metrics['uniform_20_vertex_exact_failure_probability']:.4f}")
    record("balanced_module_lesions_preserve_connectivity_in_stylized_model", metrics["balanced_module_20_vertex_failures"]==0,
           f"{metrics['balanced_module_20_vertex_failures']}/{metrics['uniform_trials']} disconnected")
    record("targeted_minimum_cut_disconnects", metrics["targeted_cut_disconnects"], "all five relay vertices removed")
    record("pinned_laplacian_gap_collapses_only_after_all_relays_removed",
           spectral_rows[-1]["lambda_min_pinned_laplacian"]<1e-9 and spectral_rows[-2]["lambda_min_pinned_laplacian"]>1e-9,
           f"lambda(4 removed)={spectral_rows[-2]['lambda_min_pinned_laplacian']:.3e}; lambda(5 removed)={spectral_rows[-1]['lambda_min_pinned_laplacian']:.3e}")

    inference_rows, inference_summary = inverse_experiment()
    errs=inference_summary["relative_error_mean_by_noise"]
    # With the chosen random-state design, clean-data error should be very small
    # and parameter error should generally not improve as noise rises.
    clean=float(errs["0.0"]); noisy=float(errs["0.12"])
    record("nnls_recovers_clean_synthetic_conductances", clean<1e-7, f"relative error={clean:.3e}")
    rank_text = f"rank={inference_summary['design_rank']}/{inference_summary['design_dimension']}; condition={inference_summary['design_condition_number']:.3g}"
    record("nnls_design_matrix_full_column_rank", inference_summary["design_rank"] == inference_summary["design_dimension"], rank_text)
    record("nnls_remains_finite_under_measurement_noise", np.isfinite(noisy) and noisy<0.5, f"relative error at sigma=0.12: {noisy:.3f}")

    trace_rows=recovery_traces()
    make_figures(metrics, damage_rows, spectral_rows, inference_rows, trace_rows, sh)

    save_csv(RES/"dispersed_vs_targeted_lesions.csv",damage_rows)
    save_csv(RES/"spectral_gap_vs_cut_size.csv",spectral_rows)
    save_csv(RES/"weighted_mcd_sensitivity.csv",metrics["relay_cost_sensitivity"])
    save_csv(RES/"nnls_conductance_estimation_trials.csv",inference_rows)
    save_csv(RES/"pinned_diffusion_traces.csv",trace_rows)

    report = {
        "run_status":"PASS", "seed":SEED,
        "scope":"mathematical checks and synthetic graph/network experiments only",
        "not_established":["biological MCD in any organism", "planarian tissue graph structure", "ageing reversal",
                           "cancer normalization", "therapeutic efficacy", "universal QEC principle"],
        "environment":{"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,
                       "scipy":scipy.__version__,"networkx":nx.__version__,"matplotlib":matplotlib.__version__},
        "sheaf_test":sh,"structural_metrics":metrics,"inverse_problem":inference_summary,
        "checks":checks,"passed":len([c for c in checks if c["status"]=="PASS"]),
        "failed":len([c for c in checks if c["status"]=="FAIL"]),
        "outputs":["figures/fig1_lesion_topology.png","figures/fig2_pinned_spectral_gap.png",
                    "figures/fig3_nnls_noise_sensitivity.png","figures/fig4_pinned_diffusion_traces.png",
                    "figures/fig5_structural_network.pdf","figures/fig6_sheaf_stalk_graph.pdf",
                    "figures/fig7_inference_flow.pdf","figures/fig8_validation_flow.pdf",
                    "results/dispersed_vs_targeted_lesions.csv","results/spectral_gap_vs_cut_size.csv",
                    "results/nnls_conductance_estimation_trials.csv","results/pinned_diffusion_traces.csv",
                    "results/weighted_mcd_sensitivity.csv"],
    }
    (RES/"verification_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["MCD MODEL VERIFICATION SUMMARY",f"Status: {report['run_status']}",f"Seed: {SEED}",
           f"Checks passed: {report['passed']}; failed: {report['failed']}","", "CHECKS:"]
    lines += [f"[{c['status']}] {c['name']}: {c['detail']}" for c in checks]
    lines += ["", "SCOPE LIMITATION: All graph and inverse-problem data in this run are synthetic.",
              "No organism, patient, ageing process, tumour, or therapeutic intervention was simulated."]
    (RES/"verification_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Content hashes for all generated numerical/visual outputs; final release manifest
    # is created by deploy.py after the PDF build.
    output_files=sorted(list(FIG.glob("*"))+list(RES.glob("*.csv"))+list(RES.glob("verification*")))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files if p.is_file()}
    (RES/"generated_outputs_sha256.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","checks":report["passed"],"mcd":mcd if False else metrics["mcd_weighted_vertex_cut"],
                      "uniform_20_vertex_failure_fraction":metrics["uniform_20_vertex_failure_fraction"],
                      "uniform_exact_probability":metrics["uniform_20_vertex_exact_failure_probability"],
                      "nnls_clean_relative_error":clean,"nnls_high_noise_relative_error":noisy,
                      "figures":8,"output_dir":str(ROOT)},indent=2))
    return report


def self_test_invalid_inputs() -> None:
    try:
        weighted_vertex_mincut(nx.path_graph(2), 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("same source/sink should be rejected")
    try:
        weighted_vertex_mincut(nx.path_graph(3), 0, 2, {1:0.0})
    except ValueError:
        pass
    else:
        raise AssertionError("zero vertex capacity should be rejected")
    print("PASS: invalid source/sink and non-positive capacity tests")


if __name__ == "__main__":
    self_test_invalid_inputs()
    run_checks()