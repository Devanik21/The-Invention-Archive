#!/usr/bin/env python3
"""Reproducible synthetic benchmark for adversarial probe selection in repair-response detection.

This script is a model-checking/illustration artifact, not an empirical alien-life detector.
All benchmark worlds and measurements are synthetic and explicitly parameterized below.
Dependencies: numpy, matplotlib, networkx.
"""
from __future__ import annotations
import csv, json, math, os, platform, sys, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
SEED = 20261017
RNG = np.random.default_rng(SEED)

ACTIONS = [
    "stress_response", "drive_gating", "history_challenge", "spatial_locality", "novel_context"
]
ACTION_LABELS = ["Initial recovery", "Drive/gradient gate", "Repeated pulse", "Spatial map", "Novel-context transfer"]
# Dimensionless synthetic observables are scaled approximately to [0,1].
MU_REPAIR = np.array([0.78, 0.72, 0.69, 0.75, 0.71], dtype=float)
DECOY_NAMES = ["Passive relaxor", "Self-healing material", "Driven oscillator", "Scripted controller"]
MU_DECOY = np.array([
    [0.82, 0.18, 0.22, 0.30, 0.24],
    [0.84, 0.48, 0.42, 0.78, 0.44],
    [0.56, 0.74, 0.28, 0.50, 0.25],
    [0.77, 0.62, 0.64, 0.68, 0.39],
], dtype=float)
SIGMA = np.array([0.12, 0.13, 0.14, 0.13, 0.14], dtype=float)
N_BINS = 31
E_MIN, E_MAX = -8.0, 8.0
GAMMA = 0.96
ALPHA = 0.18
EPISODES_ADV = 30000
EPISODES_UNI = 30000


def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


def logsumexp(vals: np.ndarray) -> float:
    m = float(np.max(vals))
    return m + math.log(float(np.sum(np.exp(vals - m))))


def log_likelihood_ratio(x: float, action: int, null_weights: np.ndarray) -> float:
    lr = log_normal_pdf(x, MU_REPAIR[action], SIGMA[action])
    alt = np.array([math.log(max(null_weights[j], 1e-12)) + log_normal_pdf(x, MU_DECOY[j, action], SIGMA[action]) for j in range(len(DECOY_NAMES))])
    return float(lr - logsumexp(alt))


def belief_bin(evidence: float) -> int:
    x = min(E_MAX, max(E_MIN, evidence))
    return int(np.clip(round((x - E_MIN) / (E_MAX - E_MIN) * (N_BINS - 1)), 0, N_BINS - 1))


def sample_observation(label: int, decoy_id: int, action: int, rng: np.random.Generator) -> float:
    mu = MU_REPAIR[action] if label == 1 else MU_DECOY[decoy_id, action]
    return float(np.clip(rng.normal(mu, SIGMA[action]), -0.15, 1.15))


def choose_action(q: np.ndarray, mask: int, eb: int, eps: float, rng: np.random.Generator) -> int:
    # 0..4 are probes; 5 declares abiotic/non-repair; 6 declares repair-like.
    valid = [a for a in range(5) if not (mask & (1 << a))] + [5, 6]
    if rng.random() < eps:
        return int(rng.choice(valid))
    vals = np.array([q[mask, eb, a] for a in valid])
    best = np.flatnonzero(vals == vals.max())
    return int(valid[int(rng.choice(best))])


def run_episode(q: np.ndarray, label: int, decoy_id: int, rng: np.random.Generator,
                null_weights: np.ndarray, epsilon: float = 0.0, train: bool = False,
                alpha: float = ALPHA) -> Tuple[int, float, int, List[int], List[float]]:
    mask, evidence = 0, 0.0
    visited, observations = [], []
    states_actions = []
    total_reward = 0.0
    for t in range(6):
        eb = belief_bin(evidence)
        a = choose_action(q, mask, eb, epsilon if train else 0.0, rng)
        state = (mask, eb)
        if a >= 5:
            pred = 1 if a == 6 else 0
            reward = (1.0 if pred == label else (-1.6 if pred == 1 else -1.0))
            reward -= 0.02 * len(visited)
            total_reward += reward
            if train:
                old = q[mask, eb, a]
                q[mask, eb, a] = old + alpha * (reward - old)
                for (sm, se, sa, r, nm, ne) in reversed(states_actions):
                    target = r + GAMMA * max(q[nm, ne, :].max(), 0.0)
                    q[sm, se, sa] += alpha * (target - q[sm, se, sa])
            return pred, evidence, len(visited), visited, observations
        x = sample_observation(label, decoy_id, a, rng)
        inc = log_likelihood_ratio(x, a, null_weights)
        new_evidence = float(np.clip(evidence + inc, E_MIN, E_MAX))
        new_mask = mask | (1 << a)
        new_state = (new_mask, belief_bin(new_evidence))
        reward = -0.035
        total_reward += reward
        if train:
            states_actions.append((mask, eb, a, reward, new_mask, belief_bin(new_evidence)))
            old = q[mask, eb, a]
            next_best = max(q[new_mask, belief_bin(new_evidence), :].max(), 0.0)
            q[mask, eb, a] = old + alpha * (reward + GAMMA * next_best - old)
        mask, evidence = new_mask, new_evidence
        visited.append(a)
        observations.append(x)
    # Safety fallback when a policy uses the complete probe budget.
    pred = 1 if evidence >= 0 else 0
    reward = (1.0 if pred == label else (-1.6 if pred == 1 else -1.0)) - 0.02 * len(visited)
    if train:
        for (sm, se, sa, r, nm, ne) in reversed(states_actions):
            target = r + GAMMA * max(q[nm, ne, :].max(), reward)
            q[sm, se, sa] += alpha * (target - q[sm, se, sa])
    return pred, evidence, len(visited), visited, observations



def evaluate_policy(q: np.ndarray, episodes_per_world: int, null_weights: np.ndarray,
                    seed: int) -> Tuple[dict, list]:
    rng = np.random.default_rng(seed)
    rows = []
    for lab in (1, 0):
        worlds = [-1] if lab == 1 else list(range(len(DECOY_NAMES)))
        for w in worlds:
            for n in range(episodes_per_world):
                did = 0 if lab == 1 else w
                pred, score, probes, acts, obs = run_episode(q, lab, did, rng, null_weights)
                rows.append({"label": lab, "decoy": "repair-capable" if lab else DECOY_NAMES[w],
                             "decoy_id": did, "pred": pred, "score": score,
                             "probes": probes, "actions": ";".join(ACTIONS[a] for a in acts),
                             "observations": ";".join(f"{x:.6f}" for x in obs)})
    return metrics(rows), rows


def wilson_interval(k: int, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / den
    return [round(max(0.0, center-half), 5), round(min(1.0, center+half), 5)]


def metrics(rows: List[dict]) -> dict:
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    tp = sum(r["pred"] == 1 for r in pos); fn = len(pos) - tp
    fp = sum(r["pred"] == 1 for r in neg); tn = len(neg) - fp
    tpr = tp / max(1, len(pos)); fpr = fp / max(1, len(neg))
    # Empirical AUROC through pairwise Mann-Whitney ranking with tie correction.
    ps = np.array([r["score"] for r in pos]); ns = np.array([r["score"] for r in neg])
    # Small evaluation sets make this O(n*m) calculation inexpensive.
    pair = (ps[:, None] > ns[None, :]) + 0.5 * (ps[:, None] == ns[None, :])
    auc = float(pair.mean()) if pair.size else float("nan")
    decoy_fprs = {}
    for name in DECOY_NAMES:
        vals = [r for r in neg if r["decoy"] == name]
        decoy_fprs[name] = sum(r["pred"] == 1 for r in vals) / max(1, len(vals))
    return {"sensitivity": round(tpr, 5), "sensitivity_wilson_95": wilson_interval(tp, len(pos)),
            "specificity": round(tn / max(1, tn + fp), 5), "specificity_wilson_95": wilson_interval(tn, len(neg)),
            "false_positive_rate": round(fpr, 5), "fpr_wilson_95": wilson_interval(fp, len(neg)),
            "worst_decoy_fpr": round(max(decoy_fprs.values()), 5),
            "balanced_accuracy": round(0.5 * (tpr + tn / max(1, tn + fp)), 5),
            "auroc_score": round(auc, 5), "mean_probes": round(float(np.mean([r["probes"] for r in rows])), 4),
            "n_positive": len(pos), "n_negative": len(neg), "per_decoy_fpr": {k: round(v, 5) for k,v in decoy_fprs.items()},
            "per_decoy_fpr_wilson_95": {name: wilson_interval(sum(r["pred"] == 1 for r in neg if r["decoy"] == name),
                                                                  sum(r["decoy"] == name for r in neg)) for name in DECOY_NAMES}}


def train_agent(episodes: int, adversarial: bool, seed: int) -> Tuple[np.ndarray, list, list]:
    rng = np.random.default_rng(seed)
    q = np.zeros((32, N_BINS, 7), dtype=float)
    weights = np.ones(4, dtype=float) / 4
    history, weight_history = [], []
    for ep in range(episodes):
        # Slowly annealed exploration. A finite probe budget makes long episodes at most 5 probes.
        epsilon = max(0.035, 0.30 * (1.0 - ep / episodes))
        label = 1 if rng.random() < 0.5 else 0
        if label:
            did = 0
        else:
            did = int(rng.choice(4, p=weights if adversarial else np.ones(4) / 4))
        pred, score, nprobe, acts, obs = run_episode(q, label, did, rng, weights if adversarial else np.ones(4)/4,
                                                      epsilon=epsilon, train=True)
        if adversarial and ep > 0 and ep % 5000 == 0:
            # Approximate best-response adversary: estimate each decoy's false-positive rate under current greedy policy,
            # then softmax the rates so future training emphasizes currently confusing abiotic worlds.
            fprs = []
            for j in range(4):
                sample_preds = []
                ev_rng = np.random.default_rng(seed + ep * 7 + j)
                for _ in range(100):
                    p, _, _, _, _ = run_episode(q, 0, j, ev_rng, weights)
                    sample_preds.append(p)
                fprs.append(float(np.mean(sample_preds)))
            scores = np.array(fprs) * 5.0
            scores -= scores.max()
            neww = np.exp(scores); neww /= neww.sum()
            weights = 0.65 * neww + 0.35 * (np.ones(4)/4)
            weights /= weights.sum()
            weight_history.append({"episode": ep, **{DECOY_NAMES[j]: float(weights[j]) for j in range(4)},
                                   **{f"estimated_fpr_{j}": fprs[j] for j in range(4)}})
        if (ep + 1) % 1000 == 0:
            history.append({"episode": ep+1, "epsilon": epsilon, "sample_reward": float(pred == label) - 0.035*nprobe,
                            "mean_q": float(np.mean(q)), "weights": weights.tolist()})
    return q, history, weight_history


def policy_from_fixed_rule(kind: str, episodes_per_world: int, seed: int) -> Tuple[dict, list]:
    rng = np.random.default_rng(seed)
    rows = []
    uniform = np.ones(4)/4
    for lab in (1, 0):
        worlds = [-1] if lab else list(range(4))
        for w in worlds:
            for _ in range(episodes_per_world):
                did = 0 if lab else w
                if kind == "recovery_only":
                    obs = sample_observation(lab, did, 0, rng)
                    # Fixed midpoint of positive and mean abiotic response; deliberately weak baseline.
                    pred = int(obs >= (MU_REPAIR[0] + MU_DECOY[:,0].mean()) / 2)
                    score = log_likelihood_ratio(obs, 0, uniform)
                    nprobes = 1
                    actions = [0]
                    observations = [obs]
                elif kind == "full_panel":
                    actions = list(range(5)); observations = [sample_observation(lab, did, a, rng) for a in actions]
                    score = sum(log_likelihood_ratio(x,a,uniform) for a,x in zip(actions,observations))
                    pred = int(score >= 0)
                    nprobes = 5
                elif kind == "random_order":
                    actions = list(rng.permutation(5)); observations = [sample_observation(lab,did,a,rng) for a in actions]
                    score = sum(log_likelihood_ratio(x,a,uniform) for a,x in zip(actions,observations))
                    pred = int(score >= 0); nprobes = 5
                rows.append({"label":lab,"decoy":"repair-capable" if lab else DECOY_NAMES[w],"decoy_id":did,
                             "pred":pred,"score":score,"probes":nprobes,"actions":";".join(ACTIONS[a] for a in actions),
                             "observations":";".join(f"{x:.6f}" for x in observations)})
    return metrics(rows), rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    # Serialize nested objects for portable CSV.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def savefig(name: str, fig=None) -> None:
    if fig is None: fig = plt.gcf()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(results: dict, eval_rows: dict, adv_history: list, weight_history: list, noise_sensitivity: list) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
                         "figure.dpi": 150, "savefig.facecolor": "white"})
    # Fig 1: ROC curves from held-out score distributions (synthetic evaluation data).
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for name, rows in eval_rows.items():
        pos = np.array([r["score"] for r in rows if r["label"] == 1]); neg = np.array([r["score"] for r in rows if r["label"] == 0])
        thr = np.linspace(min(pos.min(), neg.min())-0.1, max(pos.max(),neg.max())+0.1, 120)
        tprs = [np.mean(pos >= t) for t in thr]; fprs = [np.mean(neg >= t) for t in thr]
        order = np.argsort(fprs)
        ax.plot(np.array(fprs)[order], np.array(tprs)[order], label=f"{name} (AUC {results[name]['auroc_score']:.2f})", linewidth=1.6)
    ax.plot([0,1],[0,1],"--",linewidth=1,color="grey",label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out discrimination across synthetic worlds", xlim=(0,1), ylim=(0,1))
    ax.grid(alpha=.22); ax.legend(fontsize=7, loc="lower right"); savefig("fig1_roc",fig)
    # Fig 2: FPR by abiotic decoy.
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    names = list(results); xs = np.arange(len(DECOY_NAMES)); width = 0.78 / len(names)
    for i, method in enumerate(names):
        vals = [results[method]["per_decoy_fpr"][d] for d in DECOY_NAMES]
        ax.bar(xs + (i-(len(names)-1)/2)*width, vals, width, label=method)
    ax.set_xticks(xs, ["Passive\nrelaxor","Self-healing\nmaterial","Driven\noscillator","Scripted\ncontroller"])
    ax.set(ylabel="False-positive rate", title="Hard-negative audit: each abiotic family evaluated separately", ylim=(0,1))
    ax.grid(axis="y",alpha=.22); ax.legend(fontsize=7, ncols=2); savefig("fig2_decoy_fpr",fig)
    # Fig 3: Recovery curves: synthetic illustrative time series, not observational data.
    t = np.linspace(0, 12, 160)
    fig, ax = plt.subplots(figsize=(6.8,4.2))
    curve = lambda base, amp, tau, osc=0: base + amp*np.exp(-t/tau) + osc*np.sin(2*np.pi*t/3.5)*np.exp(-t/7)
    ax.plot(t, curve(0.08,0.86,2.1), label="Repair-capable model")
    ax.plot(t, curve(0.08,0.87,1.8), label="Passive relaxation decoy")
    ax.plot(t, curve(0.14,0.60,3.6,0.16), label="Driven oscillator decoy")
    ax.axhline(.08, color="grey", linestyle="--", linewidth=1, label="Pre-perturbation baseline")
    ax.set(xlabel="Dimensionless time after perturbation",ylabel="Normalized residual disturbance",title="Recovery alone is deliberately non-diagnostic")
    ax.grid(alpha=.22); ax.legend(fontsize=8); savefig("fig3_recovery_confounds",fig)
    # Fig 4: sensitivity, specificity, balanced accuracy and mean probes.
    fig, ax = plt.subplots(figsize=(7.0,4.3))
    ms = list(results); met = ["sensitivity","specificity","balanced_accuracy"]
    x=np.arange(len(ms)); w=.22
    for j,m in enumerate(met): ax.bar(x+(j-1)*w,[results[n][m] for n in ms],w,label=m.replace("_"," "))
    ax.set_xticks(x,ms,rotation=14,ha="right"); ax.set_ylim(0,1); ax.set_ylabel("Metric (0-1)"); ax.set_title("Aggregate held-out metrics; prevalence is balanced by construction")
    ax.grid(axis="y",alpha=.22); ax.legend(fontsize=8); savefig("fig4_metrics",fig)
    # Fig 5: graph of cross-domain observation process.
    G=nx.DiGraph()
    nodes={"u":"Perturbation","x":"Latent\nstate","r":"Repair-like\nresponse","v":"Drive /\nresource","y1":"Temporal\ntrajectory","y2":"Spatial\nresponse","y3":"History /\ntransfer","agent":"Probe\npolicy","decision":"Evidence /\nuncertainty"}
    G.add_edges_from([("u","x"),("x","r"),("v","r"),("r","y1"),("r","y2"),("r","y3"),("y1","agent"),("y2","agent"),("y3","agent"),("agent","decision")])
    pos={"u":(0,1),"x":(2,1),"r":(4,1),"v":(2,-.45),"y1":(6,2.2),"y2":(6,1),"y3":(6,-.2),"agent":(8,1),"decision":(10.3,1)}
    fig, ax=plt.subplots(figsize=(8.8,3.9))
    # Draw directed edges first so node disks mask edge interiors and labels remain legible.
    nx.draw_networkx_edges(G,pos,arrows=True,arrowstyle="-|>",arrowsize=14,width=1.2,connectionstyle="arc3,rad=0.03",min_source_margin=28,min_target_margin=28,node_size=2500,ax=ax)
    nx.draw_networkx_nodes(G,pos,node_size=2500,node_color="#e9f0fb",edgecolors="#355070",linewidths=1,ax=ax)
    nx.draw_networkx_labels(G,pos,labels=nodes,font_size=6.4,ax=ax)
    ax.set_xlim(-.8,11.2); ax.set_ylim(-1.0,2.8)
    ax.set_title("Causal measurement graph: substrate labels are not inputs to the detector"); ax.axis("off"); savefig("fig5_measurement_graph",fig)
    # Fig 6: signatures versus strongest confounders as an undirected design graph.
    G=nx.Graph()
    sig=["Recovery", "Drive coupling", "History dependence", "Locality", "Cessation", "Transfer"]
    conf=["Passive relaxation", "Self-healing material", "Oscillator", "Scripted controller"]
    G.add_nodes_from(sig,bipartite=0); G.add_nodes_from(conf,bipartite=1)
    links={"Recovery":["Passive relaxation","Self-healing material","Scripted controller"],
           "Drive coupling":["Oscillator","Scripted controller","Self-healing material"],
           "History dependence":["Self-healing material","Scripted controller"],
           "Locality":["Self-healing material","Scripted controller"],
           "Cessation":["Passive relaxation","Self-healing material","Scripted controller"],
           "Transfer":["Scripted controller","Self-healing material"]}
    for s,cs in links.items():
        for c in cs:G.add_edge(s,c)
    pos={**{s:(0, i) for i,s in enumerate(sig)},**{c:(2.7,i+0.7) for i,c in enumerate(conf)}}
    fig,ax=plt.subplots(figsize=(7.5,4.8)); col=["#d8ead1" if n in sig else "#f7dfc4" for n in G.nodes()]
    # Edges must be below nodes/labels to prevent them crossing node text.
    nx.draw_networkx_edges(G,pos,width=1.2,edge_color="#8b8b8b",min_source_margin=23,min_target_margin=23,node_size=1700,ax=ax)
    nx.draw_networkx_nodes(G,pos,node_color=col,node_size=1700,edgecolors="#555",ax=ax)
    display_labels={n:n.replace(" ","\n",1) if " " in n else n for n in G.nodes()}
    nx.draw_networkx_labels(G,pos,labels=display_labels,font_size=6.5,ax=ax)
    ax.set_title("Confounding graph: no single response feature is life-specific"); ax.axis("off"); savefig("fig6_confounding_graph",fig)
    # Fig 7: method flow diagram
    fig,ax=plt.subplots(figsize=(7.3,3.6)); ax.axis("off")
    steps=[("1. Define context", "Safe perturbation limits\n+ sensor resolution"),("2. Probe", "Choose action under\nremaining budget"),("3. Measure", "Time, space, drive,\nhistory, transfer"),("4. Update", "Evidence vs hard-negative\nmodel family"),("5. Decide", "Continue / stop; report\nlikelihood + uncertainty")]
    xpos=np.linspace(.1,.9,len(steps))
    for i,((title,desc),x) in enumerate(zip(steps,xpos)):
        ax.add_patch(plt.Rectangle((x-.085,.39),.17,.28,facecolor="#eef4fb",edgecolor="#355070",linewidth=1.1,transform=ax.transAxes))
        ax.text(x,.58,title,ha="center",va="center",fontsize=8,weight="bold",transform=ax.transAxes)
        ax.text(x,.47,desc,ha="center",va="center",fontsize=6.8,transform=ax.transAxes)
        if i<len(steps)-1: ax.annotate("",xy=(xpos[i+1]-.09,.53),xytext=(x+.09,.53),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.3,color="#355070"))
    ax.set_title("Proposed adaptive assay workflow (design, not flight hardware)",pad=16); savefig("fig7_method_flow",fig)
    # Fig 8: adversarial RL training flow.
    fig,ax=plt.subplots(figsize=(7.1,4.1)); ax.axis("off")
    boxes=[(.08,.62,.23,.20,"Probe policy / Q-table","Select untested probe"),(.39,.62,.23,.20,"Synthetic environment","Repair model or decoy"),(.70,.62,.23,.20,"Observation + reward","Evidence, cost, decision"),(.39,.20,.23,.20,"Hard-negative adversary","Reweight confusing decoys"),(.70,.20,.23,.20,"Held-out stress test","Report worst-decoy FPR")]
    for x,y,w,h,t,sub in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#f0f3f8",edgecolor="#3d5266",transform=ax.transAxes,linewidth=1.1))
        ax.text(x+w/2,y+h*.65,t,ha="center",va="center",fontsize=8,weight="bold",transform=ax.transAxes)
        ax.text(x+w/2,y+h*.34,sub,ha="center",va="center",fontsize=7,transform=ax.transAxes)
    def arr(p1,p2,rad=0): ax.annotate("",xy=p2,xytext=p1,xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.3,color="#355070",connectionstyle=f"arc3,rad={rad}"))
    arr((.31,.72),(.39,.72)); arr((.62,.72),(.70,.72)); arr((.815,.62),(.815,.40)); arr((.70,.30),(.62,.30)); arr((.505,.40),(.505,.62),-.25)
    ax.set_title("Alternating probe-policy learning and hard-negative challenge sampling",pad=16); savefig("fig8_adversarial_training",fig)



    # Fig 9: explicit observation-noise sensitivity, using the fixed learned policy.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    scales = [r["noise_multiplier"] for r in noise_sensitivity]
    ax.plot(scales, [r["balanced_accuracy"] for r in noise_sensitivity], marker="o", linewidth=1.7, label="Balanced accuracy")
    ax.plot(scales, [r["worst_decoy_fpr"] for r in noise_sensitivity], marker="s", linewidth=1.7, label="Worst-decoy FPR")
    ax.set(xlabel="Observation-noise multiplier", ylabel="Metric (0-1)", title="Sensitivity to increased measurement noise", ylim=(0,1))
    ax.grid(alpha=.22); ax.legend(fontsize=8); savefig("fig9_noise_sensitivity",fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    q_adv, hist_adv, w_hist = train_agent(EPISODES_ADV, True, SEED + 1)
    q_uni, hist_uni, _ = train_agent(EPISODES_UNI, False, SEED + 2)
    eval_results, eval_rows = {}, {}
    m, rows = evaluate_policy(q_adv, 500, np.ones(4)/4, SEED+10); eval_results["Adversarial Q-RL"] = m; eval_rows["Adversarial Q-RL"] = rows
    m, rows = evaluate_policy(q_uni, 500, np.ones(4)/4, SEED+11); eval_results["Uniform Q-RL"] = m; eval_rows["Uniform Q-RL"] = rows
    for name,kind,seed in [("Recovery-only", "recovery_only", SEED+12), ("Fixed full panel", "full_panel", SEED+13), ("Random order panel", "random_order", SEED+14)]:
        m, rows = policy_from_fixed_rule(kind, 500, seed); eval_results[name] = m; eval_rows[name] = rows
    # One-factor-at-a-time sensor-noise sensitivity for the already-trained adversarial policy.
    base_sigma = SIGMA.copy()
    noise_sensitivity = []
    for idx, multiplier in enumerate([0.75, 1.0, 1.25, 1.5]):
        SIGMA[:] = base_sigma * multiplier
        sens_metrics, sens_rows = evaluate_policy(q_adv, 200, np.ones(4)/4, SEED + 200 + idx)
        noise_sensitivity.append({"noise_multiplier": multiplier, "balanced_accuracy": sens_metrics["balanced_accuracy"],
                                  "sensitivity": sens_metrics["sensitivity"], "specificity": sens_metrics["specificity"],
                                  "worst_decoy_fpr": sens_metrics["worst_decoy_fpr"],
                                  "per_decoy_fpr": sens_metrics["per_decoy_fpr"],
                                  "n_positive": sens_metrics["n_positive"], "n_negative": sens_metrics["n_negative"]})
    SIGMA[:] = base_sigma
    write_csv(RES/"noise_sensitivity.csv", noise_sensitivity)
    results = {
        "metadata": {"run_started_utc": start, "seed": SEED, "synthetic_only": True,
                     "episodes_per_agent": EPISODES_ADV, "evaluation_episodes_per_positive_world": 500,
                     "evaluation_episodes_per_each_negative_world": 500,
                     "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                     "networkx": nx.__version__, "command": "python verify_model.py"},
        "action_names": ACTIONS, "action_labels": ACTION_LABELS,
        "repair_means": MU_REPAIR.tolist(), "decoy_names": DECOY_NAMES, "decoy_means": MU_DECOY.tolist(), "sensor_sigma": SIGMA.tolist(),
        "metrics": eval_results, "adversary_history": w_hist, "noise_sensitivity": noise_sensitivity,
        "interpretation": "All values are from synthetic benchmark worlds, not observed astrobiology data."
    }
    (RES/"benchmark_summary.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    write_csv(RES/"evaluation_episodes.csv", [r for method,rows in eval_rows.items() for r in [{**row,"method":method} for row in rows]])
    write_csv(RES/"adversarial_training_history.csv",hist_adv)
    write_csv(RES/"uniform_training_history.csv",hist_uni)
    write_csv(RES/"adversary_reweighting.csv",w_hist)
    # A fixed synthetic trajectory set underlying Fig 3, preserved as data.
    t=np.linspace(0,12,160)
    traj=[]
    for model,base,amp,tau,osc in [("repair_capable",.08,.86,2.1,0),("passive_relaxor",.08,.87,1.8,0),("driven_oscillator",.14,.60,3.6,.16)]:
        vals=base+amp*np.exp(-t/tau)+osc*np.sin(2*np.pi*t/3.5)*np.exp(-t/7)
        for tt,v in zip(t,vals):traj.append({"model":model,"time_dimensionless":round(float(tt),6),"residual_disturbance":round(float(v),8),"data_status":"synthetic illustration"})
    write_csv(RES/"synthetic_recovery_trajectories.csv",traj)
    make_figures(eval_results, eval_rows, hist_adv, w_hist, noise_sensitivity)
    # Structural checks: exact dimensions, finite metrics, finite Q and figure completeness.
    assert q_adv.shape == (32,N_BINS,7) and q_uni.shape == (32,N_BINS,7)
    assert np.isfinite(q_adv).all() and np.isfinite(q_uni).all()
    assert len(DECOY_NAMES)==4 and MU_DECOY.shape==(4,5) and len(ACTIONS)==5
    assert all(0.0 <= eval_results[k]["false_positive_rate"] <= 1.0 for k in eval_results)
    assert all(0.0 <= eval_results[k]["sensitivity"] <= 1.0 for k in eval_results)
    required_figs=[f"fig{i}_{suffix}.{ext}" for i,suffix in [(1,"roc"),(2,"decoy_fpr"),(3,"recovery_confounds"),(4,"metrics"),(5,"measurement_graph"),(6,"confounding_graph"),(7,"method_flow"),(8,"adversarial_training"),(9,"noise_sensitivity")] for ext in ("pdf","png")]
    missing=[n for n in required_figs if not (FIG/n).exists() or (FIG/n).stat().st_size==0]
    assert not missing, f"Missing figure files: {missing}"
    # Write a human-readable actual output report.
    report=["# Verification summary", "", f"Run started (UTC): {start}", f"Seed: {SEED}",
            f"Q-learning episodes per agent: {EPISODES_ADV}", "Evaluation design: 500 repair-capable episodes and 500 episodes per each of 4 synthetic abiotic decoys, per method.",
            "", "## Checks", "- PASS: Q tables have expected shape and contain finite values.",
            "- PASS: five synthetic probe actions and four explicit decoy families are present.",
            "- PASS: all evaluated sensitivity and false-positive rates are bounded in [0, 1].",
            "- PASS: nine distinct figure assets generated in PDF and PNG formats.",
            "- PASS: one-factor noise sensitivity evaluated at multipliers 0.75, 1.0, 1.25, and 1.5 with a fixed trained policy.",
            "- PASS: episode-level evaluation data and synthetic trajectory data archived.",
            "- LIMITATION: this is an internally generated synthetic benchmark; no field, spacecraft, terrestrial instrument, or alien sample data were used.",
            "- LIMITATION: success in this simulator is not evidence of extraterrestrial life or a validated flight instrument.", "", "## Metrics", ""]
    report += ["## One-factor observation-noise sensitivity", ""]
    for item in noise_sensitivity:
        report += [f"- Noise multiplier {item['noise_multiplier']}: balanced accuracy {item['balanced_accuracy']}; sensitivity {item['sensitivity']}; specificity {item['specificity']}; worst-decoy FPR {item['worst_decoy_fpr']}; per-decoy FPR \u0060{json.dumps(item['per_decoy_fpr'], sort_keys=True)}\u0060"]
    report += [""]
    for name,m in eval_results.items():
        report += [f"### {name}", f"- Sensitivity: {m['sensitivity']}", f"- Specificity: {m['specificity']}",
                   f"- Sensitivity Wilson 95% interval: {m['sensitivity_wilson_95']}",
                   f"- Balanced accuracy: {m['balanced_accuracy']}", f"- AUROC (score ranking): {m['auroc_score']}",
                   f"- Mean probes: {m['mean_probes']}", f"- Worst-decoy FPR: {m['worst_decoy_fpr']}",
                   f"- Aggregate FPR Wilson 95% interval: {m['fpr_wilson_95']}",
                   f"- Per-decoy FPR: \u0060{json.dumps(m['per_decoy_fpr'], sort_keys=True)}\u0060",
                   f"- Per-decoy FPR Wilson 95% intervals: \u0060{json.dumps(m['per_decoy_fpr_wilson_95'], sort_keys=True)}\u0060", ""]
    (RES/"verification_summary.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    env={"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,"networkx":nx.__version__,
         "executable":sys.executable,"run_started_utc":start,"seed":SEED,"command":"python verify_model.py"}
    (RES/"environment.json").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","seed":SEED,"episodes_per_agent":EPISODES_ADV,"metrics":eval_results,
                      "noise_sensitivity":noise_sensitivity,"figures":len(required_figs),"run_started_utc":start},indent=2))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print(f"VERIFICATION FAILED: {e}", file=sys.stderr)
        raise SystemExit(2)
