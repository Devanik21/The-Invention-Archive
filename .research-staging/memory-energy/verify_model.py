#!/usr/bin/env python3
"""Deterministic synthetic verification and figure generation for the Memory-Energy Frontier release."""
from __future__ import annotations
import csv
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FIG, RES = ROOT / "figures", ROOT / "results"
SEED, TOL = 20261024, 1e-10
GRID = np.linspace(0.0, 1.0, 2001)
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
INK, BLUE, TEAL, ORANGE, RED, GREY, PALE = "#263238", "#3568A8", "#168C86", "#D17A22", "#B44A4A", "#77838B", "#EAF0F4"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
 "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":GREY,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
 "figure.dpi":150,"savefig.bbox":"tight"})

def mse(k, a, q, r):
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(q)) and np.all(np.isfinite(r))):
        raise ValueError("parameters must be finite")
    if np.any(np.abs(a) >= 1) or np.any(np.asarray(q) <= 0) or np.any(np.asarray(r) <= 0):
        raise ValueError("require |a| < 1 and q,r > 0")
    den = 1.0 - np.asarray(a)**2 * (1.0-np.asarray(k))**2
    if np.any(den <= 0):
        raise ValueError("unstable denominator")
    return ((1.0-np.asarray(k))**2*q + np.asarray(k)**2*r)/den

def activity_cost(k, a, alpha, beta):
    if np.any(np.asarray(alpha) <= 0) or np.any(np.asarray(beta) <= 0):
        raise ValueError("cost weights must be positive")
    return alpha*np.asarray(k)**2 + beta*np.asarray(a)**2*(1.0-np.asarray(k))**2

def oracle_gain(a,q,r,alpha,beta,budget):
    feasible = activity_cost(GRID,a,alpha,beta) <= budget+TOL
    if not np.any(feasible):
        raise ValueError("empty feasible gain set")
    return float(GRID[int(np.argmin(np.where(feasible,mse(GRID,a,q,r),np.inf)))])

def project_gain(predicted,a,alpha,beta,budget):
    feasible = GRID[activity_cost(GRID,a,alpha,beta) <= budget+TOL]
    if feasible.size == 0:
        raise ValueError("empty feasible gain set")
    return float(feasible[int(np.argmin(np.abs(feasible-np.clip(predicted,0.0,1.0))))])

def scenario_batch(n,rng,shifted=False):
    if not shifted:
        a=rng.uniform(0.05,0.97,n); q=np.exp(rng.uniform(-2.5,1.0,n)); r=np.exp(rng.uniform(-2.5,1.0,n))
    else:
        a=rng.uniform(0.97,0.999,n); q=np.exp(rng.uniform(-4.0,-1.0,n)); r=np.exp(rng.uniform(1.0,2.5,n))
    alpha=np.exp(rng.uniform(-0.5,0.5,n)); beta=np.exp(rng.uniform(-0.5,0.5,n))
    cgrid=activity_cost(GRID[None,:],a[:,None],alpha[:,None],beta[:,None])
    cmin,cmax=cgrid.min(axis=1),cgrid.max(axis=1)
    frac=rng.uniform(0.15,1.0,n); budget=cmin+frac*(cmax-cmin)
    features=np.column_stack([a,np.log(q/r),np.log(q),np.log(r),np.log(alpha),np.log(beta),budget,frac])
    params={"a":a,"q":q,"r":r,"alpha":alpha,"beta":beta,"budget":budget,"budget_fraction":frac}
    labels=np.array([oracle_gain(ai,qi,ri,al,be,bi) for ai,qi,ri,al,be,bi in zip(a,q,r,alpha,beta,budget)])
    return features,labels,params

def score_policy(model,X,labels,params,split):
    raw=np.clip(model.predict(X),0.0,1.0); n=len(labels)
    projected=np.empty(n); fixed=np.empty(n); oracle_mse=np.empty(n); policy_mse=np.empty(n); fixed_mse=np.empty(n); costs=np.empty(n)
    violations=0
    for i in range(n):
        a,q,r=params["a"][i],params["q"][i],params["r"][i]
        al,be,b=params["alpha"][i],params["beta"][i],params["budget"][i]
        projected[i]=project_gain(raw[i],a,al,be,b); fixed[i]=project_gain(0.5,a,al,be,b)
        oracle_mse[i]=float(mse(labels[i],a,q,r)); policy_mse[i]=float(mse(projected[i],a,q,r)); fixed_mse[i]=float(mse(fixed[i],a,q,r))
        costs[i]=float(activity_cost(projected[i],a,al,be))
        violations += int(costs[i] > b+1e-8)
    regret=(policy_mse-oracle_mse)/oracle_mse; fixed_regret=(fixed_mse-oracle_mse)/oracle_mse
    if min(float(np.min(regret)),float(np.min(fixed_regret))) < -1e-8:
        raise AssertionError("a feasible policy beat the grid oracle beyond numerical tolerance")
    regret=np.maximum(regret,0.0); fixed_regret=np.maximum(fixed_regret,0.0)
    rows=[]
    for i in range(n):
        rows.append({"split":split,"case":i,"a":params["a"][i],"q":params["q"][i],"r":params["r"][i],
          "alpha":params["alpha"][i],"beta":params["beta"][i],"budget":params["budget"][i],
          "budget_fraction":params["budget_fraction"][i],"oracle_gain":labels[i],"raw_predicted_gain":raw[i],
          "projected_gain":projected[i],"fixed_gain_projected":fixed[i],"oracle_mse":oracle_mse[i],
          "policy_mse":policy_mse[i],"fixed_mse":fixed_mse[i],"policy_regret":regret[i],
          "fixed_regret":fixed_regret[i],"proxy_cost":costs[i],"budget_slack":params["budget"][i]-costs[i]})
    metrics={"n":int(n),"mean_regret":float(np.mean(regret)),"median_regret":float(np.median(regret)),
      "p95_regret":float(np.quantile(regret,0.95)),"mean_fixed_regret":float(np.mean(fixed_regret)),
      "raw_gain_mae":float(np.mean(np.abs(raw-labels))),"budget_violations":int(violations),
      "mean_oracle_mse":float(np.mean(oracle_mse)),"mean_policy_mse":float(np.mean(policy_mse))}
    return metrics,rows,regret,fixed_regret

def write_csv(path,rows,fieldnames=None):
    if not rows: raise ValueError(f"refusing to write empty CSV: {path}")
    if fieldnames is None: fieldnames=list(rows[0].keys())
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)

def save_figure(fig,stem):
    fig.savefig(FIG/f"{stem}.pdf",metadata={"Title":stem.replace("_"," ")})
    fig.savefig(FIG/f"{stem}.svg"); fig.savefig(FIG/f"{stem}.png",dpi=180); plt.close(fig)

def make_flow(stem,title,boxes,subtitle):
    fig,ax=plt.subplots(figsize=(8.4,2.35 if len(boxes)<=4 else 2.8)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    n=len(boxes); left,right,gap=0.035,0.965,0.018; width=(right-left-gap*(n-1))/n; y,h=0.33,0.42
    for i,label in enumerate(boxes):
        x=left+i*(width+gap)
        box=FancyBboxPatch((x,y),width,h,boxstyle="round,pad=0.012,rounding_size=0.018",
          linewidth=1.1,edgecolor=BLUE if i%2==0 else TEAL,facecolor=PALE)
        ax.add_patch(box); ax.text(x+width/2,y+h/2,label,ha="center",va="center",fontsize=8.0,wrap=True)
        if i<n-1:
            ax.annotate("",xy=(x+width+gap*0.88,y+h/2),xytext=(x+width+gap*0.12,y+h/2),
              arrowprops=dict(arrowstyle="->",lw=1.2,color=GREY))
    ax.text(0.5,0.92,title,ha="center",va="center",fontsize=11,fontweight="bold")
    ax.text(0.5,0.12,subtitle,ha="center",va="center",fontsize=8,color=GREY)
    save_figure(fig,stem)

def generate_figures(regret_id,fixed_id,regret_ood,fixed_ood):
    k=np.linspace(0,1,501)
    fig,ax=plt.subplots(figsize=(6.4,3.7))
    for a,color in zip([0.1,0.6,0.9],[BLUE,TEAL,ORANGE]):
        ax.plot(k,mse(k,a,0.4,0.7),label=f"a = {a:.1f}",lw=2,color=color)
    ax.set_xlabel("Observation gain k"); ax.set_ylabel("Stationary MSE P(k)"); ax.set_title("Error versus gain")
    ax.legend(frameon=False); ax.grid(alpha=0.2); save_figure(fig,"figure01_error_gain")
    fig,ax=plt.subplots(figsize=(6.4,3.7))
    for a,color in zip([0.1,0.6,0.9],[BLUE,TEAL,ORANGE]):
        ax.plot(k,activity_cost(k,a,1.0,0.7),label=f"a = {a:.1f}",lw=2,color=color)
    ax.set_xlabel("Observation gain k"); ax.set_ylabel("Activity proxy C(k)"); ax.set_title("Declared activity proxy")
    ax.legend(frameon=False); ax.grid(alpha=0.2); save_figure(fig,"figure02_cost_gain")
    a,q,r,alpha,beta=0.88,0.3,0.7,1.0,0.8
    c=activity_cost(k,a,alpha,beta); p=mse(k,a,q,r); order=np.argsort(c); pareto=[]; best=np.inf
    for idx in order:
        if p[idx]<best-1e-12: pareto.append(idx); best=p[idx]
    fig,ax=plt.subplots(figsize=(6.4,3.7)); ax.scatter(c,p,s=13,alpha=0.55,color=GREY,label="All grid gains")
    ax.plot(c[pareto],p[pareto],color=BLUE,lw=2.1,marker="o",ms=3,label="Grid Pareto frontier")
    ax.set_xlabel("Activity proxy C(k)"); ax.set_ylabel("Stationary MSE P(k)"); ax.set_title("Synthetic activity-distortion frontier")
    ax.legend(frameon=False); ax.grid(alpha=0.2); save_figure(fig,"figure03_frontier")
    write_csv(RES/"frontier_data.csv",[{"gain":float(k[i]),"activity_proxy":float(c[i]),"mse":float(p[i]),"pareto_frontier":int(i in pareto)} for i in range(len(k))])
    fig,ax=plt.subplots(figsize=(6.4,3.7)); data=[regret_id,fixed_id,regret_ood,fixed_ood]
    bp=ax.boxplot(data,tick_labels=["MLP / ID","Fixed / ID","MLP / OOD","Fixed / OOD"],patch_artist=True,showfliers=False,widths=0.62)
    for patch,color in zip(bp["boxes"],[BLUE,GREY,TEAL,GREY]): patch.set_facecolor(color); patch.set_alpha(0.65)
    ax.set_ylabel("Relative regret"); ax.set_title("Held-out policy regret"); ax.grid(axis="y",alpha=0.2)
    save_figure(fig,"figure04_policy_regret")
    G=nx.DiGraph()
    G.add_edges_from([("Latent process","Noisy observation"),("Noisy observation","Estimator"),("Persistence a","Estimator"),
      ("Gain k","Estimator"),("Estimator","Error MSE P(k)"),("Gain k","Activity proxy C(k)"),
      ("Persistence a","Activity proxy C(k)"),("Budget B","Feasibility gate"),("Activity proxy C(k)","Feasibility gate"),
      ("Feasibility gate","Selected gain")])
    pos={"Latent process":(0.08,0.72),"Noisy observation":(0.28,0.72),"Persistence a":(0.08,0.32),"Gain k":(0.31,0.32),
      "Estimator":(0.49,0.72),"Error MSE P(k)":(0.69,0.83),"Activity proxy C(k)":(0.52,0.32),
      "Budget B":(0.70,0.12),"Feasibility gate":(0.80,0.38),"Selected gain":(0.94,0.62)}
    fig,ax=plt.subplots(figsize=(7.6,4.0)); colors=[TEAL if n in {"Selected gain","Feasibility gate"} else BLUE if n in {"Estimator","Error MSE P(k)"} else PALE for n in G.nodes]
    nx.draw_networkx_nodes(G,pos,node_size=1850,node_color=colors,edgecolors=INK,linewidths=0.7,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=7.3,font_color=INK,ax=ax)
    nx.draw_networkx_edges(G,pos,arrows=True,arrowsize=13,width=1.0,edge_color=GREY,connectionstyle="arc3,rad=0.04",
      min_source_margin=12,min_target_margin=12,ax=ax)
    ax.set_title("Model dependency graph",fontsize=10,pad=10); ax.axis("off"); save_figure(fig,"figure05_dependency_graph")
    M=nx.DiGraph()
    for i in range(1,7):
        M.add_edge(f"Mode {i}",f"Gain k{i}"); M.add_edge(f"Gain k{i}","Aggregate MSE"); M.add_edge(f"Gain k{i}","Aggregate proxy cost")
    pos={}
    for i in range(1,7): pos[f"Mode {i}"]=(0.08,1-i/7); pos[f"Gain k{i}"]=(0.43,1-i/7)
    pos["Aggregate MSE"]=(0.83,0.70); pos["Aggregate proxy cost"]=(0.83,0.30)
    fig,ax=plt.subplots(figsize=(7.4,4.0)); colors=[BLUE if str(n).startswith("Mode") else TEAL if str(n).startswith("Gain") else ORANGE for n in M.nodes]
    nx.draw_networkx_nodes(M,pos,node_size=1250,node_color=colors,alpha=0.85,edgecolors=INK,linewidths=0.6,ax=ax)
    nx.draw_networkx_labels(M,pos,font_size=7.5,font_color="white",ax=ax)
    nx.draw_networkx_edges(M,pos,arrows=True,arrowsize=11,width=0.8,edge_color=GREY,connectionstyle="arc3,rad=0.02",
      min_source_margin=10,min_target_margin=10,ax=ax)
    ax.set_title("Conceptual mode-wise extension",fontsize=10,pad=10); ax.axis("off"); save_figure(fig,"figure06_mode_graph")
    make_flow("figure07_training_flow","Synthetic policy training and evaluation",
      ["Sample parameters","Grid-search feasible oracle","Train gain surrogate","Project to budget","Evaluate ID and OOD"],
      "Every point is generated from the declared simulator; no empirical observations are used.")
    make_flow("figure08_validation_flow","Proposed empirical validation path",
      ["Choose device or preparation","Calibrate proxy to measured energy","Measure task error and latency","Test shifted conditions","Accept or reject proxy"],
      "Proposed next steps only; none of these empirical stages was performed.")

def write_candidate_and_audit_files():
    candidates=[
      ("Memory-energy frontier for noisy recurrent inference",4.6,"Selected","Clear derivation, reproducible synthetic test, distinct from existing archive papers."),
      ("Morphogenetic error-correcting codes after cell loss",4.4,"Reserve","High potential but overlaps tissue-coherence/control themes and needs biological calibration."),
      ("Spectral repairability of lesioned neural circuits",4.3,"Reserve","Tractable but overlaps network-control and observability lines."),
      ("Active sensing for exoplanet biosignatures under abiotic mimicry",4.2,"Reject this cycle","Adjacent to the existing alien-life release; needs instrument-specific validation."),
      ("Epigenetic reset control under latent cell-state uncertainty",4.1,"Reject this cycle","Overlaps FER and cell-state-control releases; real biology unavailable."),
      ("Mitochondrial heteroplasmy control with adaptive intervention",4.0,"Reject this cycle","Overlaps existing heteroplasmy work and requires empirical parameters."),
      ("Active scientific inference under simulator misspecification",4.8,"Reject this cycle","Strong candidate but too close to the existing Identifiability Budget and information-theoretic papers."),
      ("AI co-design of cryoprotectant loading and nanowarming",4.2,"Reject this cycle","Overlaps CryoWeave; validation requires physical measurements."),
      ("Diatom photonic crystals as paleoclimate sensors",4.1,"Reject this cycle","Overlaps the recent Silica Light Meters release."),
      ("Information-theoretic limits of synthetic gene-circuit design",4.5,"Reject this cycle","Overlaps the recent synthetic gene-circuit information-limit release."),
    ]
    write_csv(RES/"candidate_topics.csv",[{"candidate":t,"heuristic_score_1_to_5":s,"decision":d,"rationale":why} for t,s,d,why in candidates])
    (RES/"author_profile_audit.md").write_text(
      "# Author-profile audit\n\n**Audit date:** 2026-10-24. **Coverage:** targeted public-web and repository checks; not exhaustive.\n\n"
      "- The public GitHub repository Devanik21/The-Invention-Archive is accessible and its Theoretical Research index was inspected before selecting a topic. Existing releases include information-theoretic gene-circuit limits, FER, an identifiability audit, CryoWeave, mitochondrial heteroplasmy, an alien-life synthetic benchmark, and Silica Light Meters. The selected direction was screened against these to reduce duplication.\n"
      "- The public GitHub profile displays self-reported labels including Samsung Fellow @IISc, ISRO Hackathon Winner, AI Researcher, and ECE '26, and links to NIT Agartala and the supplied ORCID. These labels were not independently corroborated and are not used as endorsements or credentials in the manuscript.\n"
      "- The supplied ORCID URL opens a JavaScript application but did not expose machine-readable identity details in this run. It is recorded as author-supplied, not independently verified.\n"
      "- Targeted searches for a confidently matched Zenodo record, OpenAIRE person/publication record, and Google Scholar author profile did not yield a verified match. This is not proof that no such record exists; no DOI, index entry, or citation count is asserted.\n"
      "- No new DOI, preprint record, citation, or formal academic appointment was established. The manuscript uses the supplied NIT Agartala ECE affiliation and does not state that a degree has been awarded.\n",encoding="utf-8")
    (RES/"literature_audit.md").write_text(
      "# Literature audit\n\n**Date:** 2026-10-24. **Scope:** targeted primary/authoritative search on recurrent estimation, energy-constrained filtering, neural energetics, thermodynamic computation, and neuromorphic hardware. Not an exhaustive systematic review.\n\n"
      "1. **Kalman (1960), DOI 10.1115/1.3662552.** Original recursive linear-filtering foundation; supports state-estimation context, not the cost proxy.\n"
      "2. **Hopfield (1982), DOI 10.1073/pnas.79.8.2554.** Recurrent associative memory and error correction; conceptual background only.\n"
      "3. **Landauer (1961), DOI 10.1147/rd.53.0183.** Thermodynamic cost of logically irreversible operations; does not calibrate this proxy.\n"
      "4. **Laughlin et al. (1998), DOI 10.1038/236.** Biophysical estimate of neural information cost in blowfly retina; not a general energy law for arbitrary estimators.\n"
      "5. **Attwell and Laughlin (2001), DOI 10.1097/00004647-200110000-00001.** Grey-matter signaling energy budget; its assumptions and biological scale differ from this scalar model.\n"
      "6. **Nourian et al. (2014), DOI 10.1109/TAC.2014.2319011.** Energy allocation for Kalman filtering over packet-dropping links; demonstrates established energy-aware estimation prior art.\n"
      "7. **Ye, Roy, and Sundaram (2020), arXiv:2006.14036.** Resilient sensor placement for Kalman filtering; relevant budgeted-estimation prior art.\n"
      "8. **Ferreira et al. (2025), DOI 10.3389/fnins.2025.1676570.** Review comparing deep and spiking networks for edge-AI neuromorphic circuits.\n"
      "9. **AbuHamra et al. (2025), DOI 10.1002/aelm.202500250.** Review of memcapacitors in neuromorphic computing; illustrates device-level considerations absent from this proxy.\n\n"
      "Novelty assessment: no absolute priority claim is made. The defensible distinction is the reproducible combination of a closed-form scalar MSE, a clearly labelled internal activity proxy, a learned gain surrogate with hard feasibility projection, and an ID/OOD stress test. Energy-aware Kalman filtering is established; this paper does not claim to supersede it.\n\n"
      "Primary records: https://doi.org/10.1115/1.3662552 ; https://doi.org/10.1073/pnas.79.8.2554 ; https://doi.org/10.1147/rd.53.0183 ; https://doi.org/10.1038/236 ; https://doi.org/10.1097/00004647-200110000-00001 ; https://doi.org/10.1109/TAC.2014.2319011 ; https://arxiv.org/abs/2006.14036 ; https://doi.org/10.3389/fnins.2025.1676570 ; https://doi.org/10.1002/aelm.202500250\n",encoding="utf-8")
    claims=[
      {"claim_id":"C1","claim":"Stationary MSE expression","class":"P1","evidence":"Algebraic derivation from declared scalar state and estimator","test":"Monte Carlo stationary error variance","status":"TO_BE_RUN","limitation":"Independent white Gaussian noise and stable scalar recurrence"},
      {"claim_id":"C2","claim":"Projection satisfies declared activity budget","class":"P1","evidence":"Projection onto finite feasible grid","test":"Assert proxy cost <= budget for all held-out cases","status":"TO_BE_RUN","limitation":"Guarantees only the proxy constraint"},
      {"claim_id":"C3","claim":"Learned policy approximates grid oracle","class":"P3","evidence":"Held-out synthetic relative regret","test":"ID and shifted synthetic sets","status":"TO_BE_RUN","limitation":"Oracle labels come from the same assumed simulator"},
      {"claim_id":"C4","claim":"Proxy may inform neural/neuromorphic design","class":"P4","evidence":"Structural analogy only","test":"Compare proxy ranking with measured energy and task error","status":"NOT_TESTED","limitation":"No empirical calibration or hardware measurements"},
      {"claim_id":"C5","claim":"Memory is a resource traded against correction","class":"P5","evidence":"Qualitative intuition motivating the model","test":"Not a standalone empirical claim","status":"HYPOTHESIS","limitation":"No biological mechanism inferred from the analogy"},
    ]
    write_csv(RES/"claim_evidence_matrix.csv",claims)
    (RES/"research_log.md").write_text(
      "# Research workflow log\n\n- Cycle date: 2026-10-24.\n"
      "- Selected topic: The Memory-Energy Frontier in Noisy Recurrent Inference: A Spectral Rate-Distortion Framework for Neural and Neuromorphic Systems.\n"
      "- Ten directions were compared in candidate_topics.csv. Candidates overlapping existing archive papers or requiring unavailable empirical data were rejected for this cycle.\n"
      "- Literature pass: Kalman filtering, energy-aware state estimation, neural energetic costs, Landauer's principle, and neuromorphic device reviews were checked against primary/authoritative records. Citation details are in paper.bib and literature_audit.md.\n"
      "- Formal core: scalar AR(1) state, additive observation noise, recurrent gain estimator, stationary error variance, quadratic activity proxy, constrained grid optimum, and projection-feasibility proposition.\n"
      "- AI role: an MLP surrogate predicts a grid-oracle gain from synthetic model descriptors; hard projection enforces the declared proxy budget. It is not trained on biological or hardware data.\n"
      "- Validation scope: deterministic analytical/numerical checks and synthetic ID/OOD policy evaluation. No neural recordings, device measurements, ATP measurements, clinical outcomes, or external dataset were used.\n"
      "- Remaining gate: manual page-by-page inspection of the final compiled PDF and independent remote release closure.\n",encoding="utf-8")

def run_checks():
    checks=[]
    def record(name,ok,detail):
        checks.append({"name":name,"status":"PASS" if ok else "FAIL","detail":detail})
        if not ok: raise AssertionError(f"{name}: {detail}")
    q0,r0=0.37,0.82; k0=float(GRID[int(np.argmin(mse(GRID,0.0,q0,r0)))])
    record("a=0 limiting optimum",abs(k0-q0/(q0+r0))<=1.0/(len(GRID)-1),f"grid k={k0:.6f}; analytic k={q0/(q0+r0):.6f}")
    rng=np.random.default_rng(SEED+1); a,q,r,k=0.72,0.35,0.60,0.37; x=0.0; xhat=0.0; errors=[]
    for t in range(260000):
        w=rng.normal(0.0,np.sqrt(q)); v=rng.normal(0.0,np.sqrt(r)); x=a*x+w; y=x+v
        xhat=a*(1.0-k)*xhat+k*y
        if t>=10000: errors.append(x-xhat)
    empirical=float(np.var(errors)); analytic=float(mse(k,a,q,r)); rel=abs(empirical-analytic)/analytic
    record("Monte Carlo stationary variance",rel<0.04,f"analytic={analytic:.6f}; empirical={empirical:.6f}; relative error={rel:.4f}")
    invalid=False
    try: mse(0.5,0.3,-1.0,0.2)
    except ValueError: invalid=True
    record("invalid variance rejected",invalid,"q <= 0 raises ValueError")
    empty=False
    try: oracle_gain(0.8,0.3,0.4,1.0,1.0,-1.0)
    except ValueError: empty=True
    record("empty feasible set rejected",empty,"infeasible budget raises ValueError")
    X_train,y_train,_=scenario_batch(2400,np.random.default_rng(SEED),False)
    X_id,y_id,p_id=scenario_batch(500,np.random.default_rng(SEED+10),False)
    X_ood,y_ood,p_ood=scenario_batch(500,np.random.default_rng(SEED+20),True)
    model=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(48,32),activation="relu",solver="adam",
      max_iter=700,early_stopping=True,n_iter_no_change=30,random_state=SEED,validation_fraction=0.15,tol=1e-5))
    model.fit(X_train,y_train)
    idm,rows_id,regret_id,fixed_id=score_policy(model,X_id,y_id,p_id,"ID")
    oodm,rows_ood,regret_ood,fixed_ood=score_policy(model,X_ood,y_ood,p_ood,"OOD")
    total_violations=idm["budget_violations"]+oodm["budget_violations"]
    record("budget projection",total_violations==0,f"{total_violations} violations across 1000 held-out cases")
    record("finite policy metrics",all(np.isfinite(v) for v in [idm["mean_regret"],oodm["mean_regret"],idm["raw_gain_mae"],oodm["raw_gain_mae"]]),"all reported metrics finite")
    record("nonnegative oracle regret",min(float(np.min(regret_id)),float(np.min(regret_ood)))>=0.0,"regret is nonnegative relative to the same finite-grid oracle")
    write_csv(RES/"model_outputs.csv",rows_id+rows_ood)
    allrows=[("ID",row,float(reg)) for row,reg in zip(rows_id,regret_id)]+[("OOD",row,float(reg)) for row,reg in zip(rows_ood,regret_ood)]
    sens=[]
    for name,lo,hi in [("Tight",0.15,0.4333333334),("Intermediate",0.4333333334,0.7166666667),("Relaxed",0.7166666667,1.0000001)]:
        sub=[(lab,row,reg) for lab,row,reg in allrows if lo<=float(row["budget_fraction"])<hi]
        if not sub: raise AssertionError(f"empty sensitivity bin: {name}")
        vio=sum(float(row["proxy_cost"])>float(row["budget"])+1e-8 for _,row,_ in sub)
        sens.append({"regime":name,"count":len(sub),"mean_policy_regret":float(np.mean([reg for _,_,reg in sub])),
          "mean_oracle_mse":float(np.mean([float(row["oracle_mse"]) for _,row,_ in sub])),"budget_violations":int(vio)})
    sens.append({"regime":"All","count":len(allrows),"mean_policy_regret":float(np.mean([reg for _,_,reg in allrows])),
      "mean_oracle_mse":float(np.mean([float(row["oracle_mse"]) for _,row,_ in allrows])),"budget_violations":total_violations})
    write_csv(RES/"sensitivity_sweep.csv",sens)
    generate_figures(regret_id,fixed_id,regret_ood,fixed_ood); write_candidate_and_audit_files()
    summary={"seed":SEED,"data_scope":"synthetic analytical model only; no measured biological or hardware data",
      "training_cases":int(len(y_train)),"id":idm,"ood":oodm,"budget_violations_total":int(total_violations),
      "mlp_iterations":int(model[-1].n_iter_),"mlp_training_loss":float(model[-1].loss_),
      "gain_grid_points":int(len(GRID)),"gain_grid_spacing":float(GRID[1]-GRID[0]),"checks":checks,
      "figure_inventory":[{"file":f"figures/figure{i:02d}_{name}.pdf","type":kind} for i,(name,kind) in enumerate([
        ("error_gain","plot"),("cost_gain","plot"),("frontier","plot"),("policy_regret","plot"),
        ("dependency_graph","graph"),("mode_graph","graph"),("training_flow","flow"),("validation_flow","flow")],start=1)]}
    (RES/"numerical_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    macros={"MeanIDRegret":idm["mean_regret"],"MedianIDRegret":idm["median_regret"],"P95IDRegret":idm["p95_regret"],
      "MeanOODRegret":oodm["mean_regret"],"MedianOODRegret":oodm["median_regret"],"P95OODRegret":oodm["p95_regret"],
      "PolicyMAE":idm["raw_gain_mae"],"PolicyMAEOOD":oodm["raw_gain_mae"],"TestIDCount":idm["n"],"TestOODCount":oodm["n"],
      "TestAllCount":idm["n"]+oodm["n"],"BudgetViolationCount":total_violations,
      "TightCount":sens[0]["count"],"TightRegret":sens[0]["mean_policy_regret"],"TightViolations":sens[0]["budget_violations"],
      "MidCount":sens[1]["count"],"MidRegret":sens[1]["mean_policy_regret"],"MidViolations":sens[1]["budget_violations"],
      "RelaxedCount":sens[2]["count"],"RelaxedRegret":sens[2]["mean_policy_regret"],"RelaxedViolations":sens[2]["budget_violations"],
      "AllRegret":sens[3]["mean_policy_regret"]}
    tex=[]
    for name,value in macros.items():
        val=str(int(value)) if isinstance(value,(int,np.integer)) else f"{float(value):.5f}"
        tex.append("\\newcommand{\\"+name+"}{"+val+"}")
    (RES/"metrics.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    env={"python":platform.python_version(),"platform":platform.platform(),"numpy":metadata.version("numpy"),
      "matplotlib":metadata.version("matplotlib"),"scikit-learn":metadata.version("scikit-learn"),
      "networkx":metadata.version("networkx"),"PyMuPDF":metadata.version("PyMuPDF"),"seed":SEED}
    (RES/"environment_report.json").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")
    (RES/"verification_report.json").write_text(json.dumps({"status":"PASS","checks_passed":len(checks),"checks_failed":0,
      "checks":checks,"data_scope":summary["data_scope"],"command":"python verify_model.py","seed":SEED},indent=2)+"\n",encoding="utf-8")
    curves=[]
    for a in [0.1,0.6,0.9]:
        for kval in np.linspace(0,1,501):
            curves.append({"a":a,"k":float(kval),"q":0.4,"r":0.7,"mse":float(mse(kval,a,0.4,0.7)),
              "activity_proxy":float(activity_cost(kval,a,1.0,0.7))})
    write_csv(RES/"figure_curve_data.csv",curves)
    print("VERIFY_MODEL: PASS")
    print(f"Checks passed: {len(checks)}; failed: 0")
    print(f"ID mean regret: {idm['mean_regret']:.6f}; ID p95: {idm['p95_regret']:.6f}")
    print(f"OOD mean regret: {oodm['mean_regret']:.6f}; OOD p95: {oodm['p95_regret']:.6f}")
    print(f"Budget violations after projection: {total_violations}")
    print("Data scope: synthetic analytical model only; no measured biological or hardware data.")
    print("Figures: 4 plots, 2 structural graphs, 2 flow diagrams; PDF/SVG/PNG variants.")
    return summary

if __name__ == "__main__":
    run_checks()
