#!/usr/bin/env python3
"""Reproduce synthetic channel calculations and all eight publication figures.

This is a mathematical illustration, not a biological simulator. All parameter
values are hypothetical and are written to results/model_outputs.csv.
Dependencies: Python >=3.10, numpy, matplotlib, networkx.
Run: python verify_model.py
"""
from __future__ import annotations
import csv, json, math, platform, sys, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
SEED = 20261017
EPS = 1e-12

# Plot style deliberately conservative, legible, and colourblind-aware.
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "svg.fonttype": "none",
})


def h2(p: float) -> float:
    """Binary entropy in bits, with exact endpoint handling."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1.0-p) * math.log2(1.0-p)


def symmetric_channel_capacity(k: int, error: float) -> float:
    """Capacity of a K-ary symmetric channel (uniform-input optimum), bits."""
    if not isinstance(k, int) or k < 2:
        raise ValueError("K must be an integer >= 2")
    if not 0.0 <= error <= 1.0 - 1.0/k:
        raise ValueError("error outside the nonnegative-capacity operating range")
    return math.log2(k) - h2(error) - error * math.log2(k-1)


def compose_kary_errors(e_generator: float, e_biology: float, k: int) -> float:
    """Mismatch probability after two independent K-ary symmetric channels."""
    if not isinstance(k, int) or k < 2:
        raise ValueError("K must be an integer >= 2")
    for p in (e_generator, e_biology):
        if not 0 <= p <= 1 - 1/k:
            raise ValueError("each stage error must be in [0, 1-1/K]")
    return e_generator + e_biology - (k/(k-1))*e_generator*e_biology


def channel_matrix(k: int, error: float) -> np.ndarray:
    """Row-stochastic K-ary symmetric transition matrix."""
    if not isinstance(k, int) or k < 2:
        raise ValueError("K must be an integer >= 2")
    if not 0 <= error <= 1:
        raise ValueError("error must lie in [0, 1]")
    m = np.full((k, k), error/(k-1), dtype=float)
    np.fill_diagonal(m, 1-error)
    return m


def mutual_information_bits(px: np.ndarray, channel: np.ndarray) -> float:
    """I(X;Y) for input probabilities px and row-conditional P(Y|X)."""
    px = np.asarray(px, dtype=float)
    channel = np.asarray(channel, dtype=float)
    if channel.ndim != 2 or px.ndim != 1 or channel.shape[0] != px.size:
        raise ValueError("incompatible input distribution and channel shape")
    if np.any(px < 0) or not np.isclose(px.sum(), 1.0):
        raise ValueError("input probabilities must be nonnegative and sum to 1")
    if np.any(channel < -EPS) or not np.allclose(channel.sum(axis=1), 1.0):
        raise ValueError("channel rows must be valid probability distributions")
    py = px @ channel
    result = 0.0
    for i in range(px.size):
        for j in range(channel.shape[1]):
            joint = px[i] * channel[i, j]
            if joint > 0 and py[j] > 0:
                result += joint * math.log2(channel[i, j] / py[j]) if channel[i, j] > 0 else 0.0
    return float(result)


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_figures_and_data() -> list[dict]:
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    records: list[dict] = []

    # Plot 1: The capacity of idealized K-ary symmetric channels.
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    capacity_rows = []
    for k in (2, 4, 8):
        limit = 1 - 1/k
        es = np.linspace(0, limit, 180)
        cs = [symmetric_channel_capacity(k, float(e)) for e in es]
        ax.plot(es, cs, label=f"K={k}", lw=2)
        for e, c in zip(es, cs):
            capacity_rows.append({"plot":"capacity_vs_error", "K":k, "error":float(e), "capacity_bits":float(c), "data_type":"analytical synthetic channel"})
    ax.set(xlabel="Per-stage symbol error probability", ylabel="Channel capacity (bits/use)", title="Idealized capacity falls as functional mismatch rises")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=.23)
    savefig(fig, "figure01_capacity_vs_error")
    records.extend(capacity_rows)

    # Plot 2: composed generator and realization mismatch, K=4.
    k = 4
    eg = np.linspace(0, 1-1/k, 101)
    eb = np.linspace(0, 1-1/k, 101)
    Z = np.array([[compose_kary_errors(float(x), float(y), k) for x in eg] for y in eb])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=[eg[0], eg[-1], eb[0], eb[-1]], cmap="viridis", vmin=0, vmax=1-1/k)
    ax.set(xlabel="Generative proposal mismatch", ylabel="Biological realization mismatch", title="Errors compound across the design chain (K=4)")
    cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label("Final target mismatch")
    ax.contour(eg, eb, Z, levels=[.1,.2,.3,.4,.5], colors="white", linewidths=.65, alpha=.85)
    savefig(fig, "figure02_composed_error_heatmap")
    for iy in range(0, len(eb), 10):
        for ix in range(0, len(eg), 10):
            records.append({"plot":"composed_error_heatmap", "K":k, "generator_error":float(eg[ix]), "biology_error":float(eb[iy]), "final_error":float(Z[iy,ix]), "data_type":"analytical synthetic channel"})

    # Plot 3: explicit hypothetical context shift. Error parameters are assumptions.
    shifts = np.linspace(0, 1, 101)
    eg_ctx = .04 + .22 * shifts
    eb_ctx = .06 + .24 * shifts
    final_ctx = np.array([compose_kary_errors(float(a), float(b), 2) for a,b in zip(eg_ctx, eb_ctx)])
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(shifts, eg_ctx, label="Proposal-stage mismatch", lw=2)
    ax.plot(shifts, eb_ctx, label="Realization-stage mismatch", lw=2)
    ax.plot(shifts, final_ctx, label="Composed mismatch", lw=2.5, color="#333333")
    ax.set(xlabel="Normalized context shift (assumed 0–1)", ylabel="Mismatch probability", title="Context shift can expose a weakly validated design")
    ax.set_ylim(0, .42); ax.legend(frameon=False); ax.grid(alpha=.23)
    savefig(fig, "figure03_context_shift")
    for a,b,c,d in zip(shifts,eg_ctx,eb_ctx,final_ctx):
        records.append({"plot":"context_shift", "context_shift":float(a), "generator_error":float(b), "biology_error":float(c), "final_error":float(d), "data_type":"hypothetical sensitivity scenario; not measured biology"})

    # Plot 4: Monte Carlo measurement uncertainty under a fixed hypothetical error.
    true_e = .15
    sample_sizes = np.array([20, 40, 80, 160, 320, 640, 1280])
    reps = 1500
    summary_rows = []
    means, low, high = [], [], []
    for n in sample_sizes:
        phat = rng.binomial(int(n), true_e, size=reps) / n
        # empirical 2.5% and 97.5% Monte Carlo quantiles of the estimator
        qlo, qhi = np.quantile(phat, [.025,.975])
        means.append(float(np.mean(phat))); low.append(float(qlo)); high.append(float(qhi))
        summary_rows.append({"plot":"finite_sample_error_estimation", "n_per_assumed_design":int(n), "replicates":reps, "assumed_true_error":true_e, "mean_estimated_error":float(np.mean(phat)), "mc_q025":float(qlo), "mc_q975":float(qhi), "data_type":"synthetic binomial Monte Carlo"})
    fig, ax = plt.subplots(figsize=(6.5,3.7))
    ax.fill_between(sample_sizes, low, high, alpha=.2, label="95% Monte Carlo quantile interval")
    ax.plot(sample_sizes, means, marker="o", lw=2, label="Mean estimated mismatch")
    ax.axhline(true_e, linestyle="--", color="#444444", label="Assumed mismatch = 0.15")
    ax.set_xscale("log", base=2); ax.set(xlabel="Synthetic independent observations, n", ylabel="Estimated mismatch", title="More observations narrow uncertainty; they do not fix model bias")
    ax.legend(frameon=False); ax.grid(alpha=.23)
    savefig(fig, "figure04_finite_sample_uncertainty")
    records.extend(summary_rows)

    # Graph 1: channel architecture (not a flowchart; nodes are random variables).
    G = nx.DiGraph()
    edges = [("T", "G"), ("G", "Y"), ("C", "Y"), ("D", "G"), ("Y", "A")]
    G.add_edges_from(edges)
    pos = {"T":(0,0.7), "G":(1.3,.7), "Y":(2.7,.7), "C":(2.7,-.35), "D":(0,-.35), "A":(1.3,-.35)}
    fig, ax = plt.subplots(figsize=(6.5,3.1))
    nx.draw_networkx_nodes(G,pos,node_size=[1800 if n in ("T","G","Y") else 1450 for n in G.nodes],node_color=["#d9eaf7" if n in ("T","G","Y") else "#f8e8c8" for n in G.nodes],edgecolors="#364152",linewidths=1.0,ax=ax)
    nx.draw_networkx_labels(G,pos,labels={"T":"Target T","G":"Proposed design G","Y":"Realized phenotype Y","C":"Context C","D":"Training data D","A":"Assay / measurement A"},font_size=8,ax=ax)
    nx.draw_networkx_edges(G,pos,edgelist=edges,arrows=True,arrowsize=16,width=1.25,edge_color="#526170",connectionstyle="arc3,rad=0.03",ax=ax)
    ax.set_title("Design chain and context / evidence dependencies", pad=10); ax.axis("off")
    savefig(fig, "figure05_design_channel_graph")

    # Graph 2: schematic confounding/context graph, intentionally a graph not a process flow.
    H = nx.Graph()
    H.add_edges_from([("Prompts","Training support"),("Training support","Proposal quality"),("Proposal quality","Measured phenotype"),("Context","Measured phenotype"),("Assay noise","Measured phenotype"),("Context","Observed training data"),("Observed training data","Training support")])
    pos2={"Prompts":(0,1),"Training support":(1.4,1.2),"Proposal quality":(2.8,1),"Measured phenotype":(4.1,.2),"Context":(2.8,-.45),"Assay noise":(4.1,-.75),"Observed training data":(.8,-.35)}
    fig, ax = plt.subplots(figsize=(7.1,3.5))
    node_colors=["#d9eaf7" if n in ("Prompts","Proposal quality") else "#f8e8c8" if n in ("Context","Assay noise") else "#e3f0df" for n in H.nodes]
    nx.draw_networkx_nodes(H,pos2,node_size=1750,node_color=node_colors,edgecolors="#364152",linewidths=1,ax=ax)
    nx.draw_networkx_labels(H,pos2,font_size=7.2,ax=ax)
    nx.draw_networkx_edges(H,pos2,width=1.2,edge_color="#526170",ax=ax)
    ax.set_title("Potential confounding paths to measured performance",pad=10); ax.axis("off")
    savefig(fig,"figure06_context_dependency_graph")

    # Flow diagram 1: candidate design / validation sequence.
    fig, ax = plt.subplots(figsize=(7.0,3.15)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis("off")
    flow = [(0.3,1.65,1.5,0.85,"Target spec\n+ context"),(2.25,1.65,1.5,.85,"Generative\nproposal"),(4.2,1.65,1.5,.85,"Model-based\nscore"),(6.15,1.65,1.5,.85,"Independent\nvalidation"),(8.1,1.65,1.5,.85,"Release or\nreject")]
    for x,y,w,h,t in flow:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#edf3f8",edgecolor="#364152",linewidth=1.2,joinstyle="round"))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8)
    for i in range(len(flow)-1):
        x=flow[i][0]+flow[i][2]; nx0=flow[i+1][0]
        ax.annotate("",xy=(nx0-.05,2.075),xytext=(x+.03,2.075),arrowprops=dict(arrowstyle="->",lw=1.2,color="#526170"))
    ax.text(5,3.35,"Proposed workflow — each score remains conditional on its evidence",ha="center",fontsize=10,weight="bold")
    ax.text(5,.65,"A high generative score is a triage signal, not experimental confirmation.",ha="center",fontsize=8.5)
    savefig(fig,"figure07_validation_pipeline_flow")

    # Flow diagram 2: gate with an explicit failure return path.
    fig, ax = plt.subplots(figsize=(6.8,3.8)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    boxes=[(3.6,4.05,2.5,.58,"Define target + context"),(3.6,3.0,2.5,.58,"Generate proposal set"),(3.6,1.95,2.5,.58,"Check calibration / robustness"),(3.6,.9,2.5,.58,"Report uncertainty + limits")]
    for x,y,w,h,t in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#edf3f8",edgecolor="#364152",linewidth=1.1))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8)
    for i in range(3):
        ax.annotate("",xy=(4.85,boxes[i+1][1]+boxes[i+1][3]+.02),xytext=(4.85,boxes[i][1]-.02),arrowprops=dict(arrowstyle="->",lw=1.15,color="#526170"))
    ax.add_patch(plt.Rectangle((7.25,2.35),2.0,.85,facecolor="#f8e8c8",edgecolor="#8a5a00",linewidth=1.1))
    ax.text(8.25,2.775,"Gate fails:\nrevise / abstain",ha="center",va="center",fontsize=8)
    ax.annotate("",xy=(7.2,2.75),xytext=(6.15,2.24),arrowprops=dict(arrowstyle="->",lw=1.2,color="#8a5a00",connectionstyle="arc3,rad=-.12"))
    ax.annotate("return",xy=(6.4,3.32),xytext=(8.0,3.55),ha="center",fontsize=7.5,arrowprops=dict(arrowstyle="->",lw=1.0,color="#8a5a00",connectionstyle="arc3,rad=.35"))
    ax.set_title("Evidence gate with a non-success exit",pad=8)
    savefig(fig,"figure08_evidence_gate_flow")

    with (RES / "model_outputs.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in records for key in row})
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(records)

    # Small, explicit, deterministic numerical summary used in paper/README.
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "status": "passed; see verification_report.json for test details",
        "data_scope": "synthetic/analytical only; no biological measurements",
        "kary_symmetric_example": {
            "K": 4, "generator_error": 0.10, "realization_error": 0.15,
            "composed_error": compose_kary_errors(.10,.15,4),
            "generator_capacity_bits": symmetric_channel_capacity(4,.10),
            "realization_capacity_bits": symmetric_channel_capacity(4,.15),
            "composed_capacity_bits": symmetric_channel_capacity(4,compose_kary_errors(.10,.15,4)),
        },
        "binary_example": {
            "generator_error": .10, "realization_error": .15,
            "composed_error": compose_kary_errors(.10,.15,2),
            "generator_capacity_bits": symmetric_channel_capacity(2,.10),
            "realization_capacity_bits": symmetric_channel_capacity(2,.15),
            "composed_capacity_bits": symmetric_channel_capacity(2,compose_kary_errors(.10,.15,2)),
        },
        "figures_generated": sorted(p.name for p in FIG.glob("figure*.svg")),
    }
    (RES / "numerical_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return records


def run_tests() -> dict:
    tests=[]
    def check(name, fn):
        try:
            fn(); tests.append({"name":name,"status":"passed"})
        except Exception as exc:
            tests.append({"name":name,"status":"failed","detail":repr(exc)})
            raise
    check("binary entropy endpoints", lambda: (_ for _ in ()).throw(AssertionError()) if (h2(0)!=0 or h2(1)!=0) else None)
    check("binary entropy symmetry", lambda: (_ for _ in ()).throw(AssertionError()) if abs(h2(.2)-h2(.8))>1e-12 else None)
    check("K=2 channel capacity at zero error", lambda: (_ for _ in ()).throw(AssertionError()) if abs(symmetric_channel_capacity(2,0)-1)>1e-12 else None)
    check("capacity zero at random-guess error", lambda: (_ for _ in ()).throw(AssertionError()) if abs(symmetric_channel_capacity(4,.75))>1e-12 else None)
    check("error composition stays bounded", lambda: (_ for _ in ()).throw(AssertionError()) if not 0 <= compose_kary_errors(.1,.15,4) <= .75 else None)
    check("binary composition identity", lambda: (_ for _ in ()).throw(AssertionError()) if abs(compose_kary_errors(.1,.15,2)-(.1+.15-2*.1*.15))>1e-12 else None)
    check("channel rows sum to one", lambda: (_ for _ in ()).throw(AssertionError()) if not np.allclose(channel_matrix(4,.1).sum(axis=1),1) else None)
    check("uniform-input MI bounded by log K", lambda: (_ for _ in ()).throw(AssertionError()) if not 0 <= mutual_information_bits(np.ones(4)/4,channel_matrix(4,.1)) <= 2+1e-12 else None)
    check("invalid probability rejected", lambda: _expect_value_error(lambda: h2(1.1)))
    check("invalid K rejected", lambda: _expect_value_error(lambda: symmetric_channel_capacity(1,.1)))
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(), "python":sys.version, "platform":platform.platform(), "seed":SEED, "dependency_versions":{"numpy":np.__version__,"matplotlib":matplotlib.__version__,"networkx":nx.__version__}, "tests":tests,"passed":sum(t["status"]=="passed" for t in tests),"failed":sum(t["status"]=="failed" for t in tests),"status":"passed"}


def _expect_value_error(f):
    try: f()
    except ValueError: return
    raise AssertionError("expected ValueError was not raised")


def main() -> int:
    try:
        run_tests()
        generate_figures_and_data()
        report=run_tests()
        report["plots_rebuilt"] = 4
        report["additional_graphs_rebuilt"] = 2
        report["flow_diagrams_rebuilt"] = 2
        report["note"] = "These tests verify the equations and synthetic examples only; no biological or wet-lab claim is tested."
        (RES/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        with (RES/"verify_model_stdout.txt").open("w",encoding="utf-8") as f:
            f.write("Verification report\n")
            f.write(f"Status: {report['status']}\nPassed: {report['passed']}\nFailed: {report['failed']}\n")
            f.write(f"Figures generated: 8 (4 plots, 2 additional graphs, 2 flow diagrams)\n")
            f.write("Data: analytical/synthetic only; no biological measurements.\n")
            for item in report["tests"]: f.write(f"[{item['status'].upper()}] {item['name']}\n")
        print(f"VERIFY_MODEL: {report['status'].upper()} ({report['passed']} passed, {report['failed']} failed)")
        print("Data scope: analytical/synthetic only; no biological measurements.")
        print("Figures: 4 plots, 2 graph-structured schematics, 2 workflow diagrams.")
        print("Results: results/model_outputs.csv, numerical_summary.json, verification_report.json")
        return 0
    except Exception as exc:
        print(f"VERIFY_MODEL: FAILED: {exc}",file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
