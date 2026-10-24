from __future__ import annotations
import os, sys, json, csv, math, itertools, hashlib, shutil, subprocess, platform, textwrap
from pathlib import Path
from datetime import datetime, timezone, date

TITLE = "Aging as a Decoding Threshold: Repair Scheduling Under Fixed Maintenance Throughput"
SLUG = "Aging_as_a_Decoding_Threshold_Repair_Scheduling_Under_Fixed_Maintenance_Throughput"
TODAY = "2026-10-24"
AUTHOR = "Devanik Debnath"
AFFILIATION = "Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India"
REPO = "https://github.com/Devanik21/The-Invention-Archive"
ORCID = "https://orcid.org/0009-0001-8310-6921"
RELEASE_DIR = Path(__file__).resolve().parent / SLUG

# Model definitions: damage coordinates are abstract state variables, not measured biomarkers.
A = [[0.0, 0.08, 0.0, 0.04],
     [0.05, 0.0, 0.02, 0.08],
     [0.08, 0.03, 0.0, 0.0],
     [0.02, 0.0, 0.07, 0.0]]
STATE_NAMES = ["Proteostasis", "Mitochondrial", "Genome maintenance", "Regulatory drift"]
B_SOURCE = [0.04, 0.04, 0.04, 0.04]
X0 = [0.8, 0.8, 0.8, 0.8]
R_TOTAL = 0.12
SLOT = 3.0
PERIOD = 12.0
THRESHOLD = 8.0
DT = 0.02
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
PERMS = list(itertools.permutations(range(4)))


def rhs(x, kappa, u):
    return [B_SOURCE[i] + kappa * sum(A[i][j] * x[j] for j in range(4)) - u[i] for i in range(4)]


def simulate(kappa, perm=(0,1,2,3), mode="schedule", repair=R_TOTAL, threshold=THRESHOLD, dt=DT, tmax=300.0):
    if kappa < 0 or repair < 0 or threshold <= sum(X0) or dt <= 0 or tmax <= 0:
        raise ValueError("Invalid model parameter")
    if mode not in {"schedule", "uniform", "none"}:
        raise ValueError("mode must be schedule, uniform, or none")
    if mode == "schedule" and (len(perm) != 4 or sorted(perm) != [0,1,2,3]):
        raise ValueError("perm must be a permutation of four repair classes")
    x = list(X0)
    total_old = sum(x)
    nsteps = int(round(tmax / dt))
    steps_per_slot = int(round(SLOT / dt))
    steps_per_period = 4 * steps_per_slot
    for step in range(nsteps):
        if mode == "schedule":
            slot_i = (step % steps_per_period) // steps_per_slot
            u = [0.0] * 4
            u[perm[slot_i]] = repair
        elif mode == "uniform":
            u = [repair / 4.0] * 4
        else:
            u = [0.0] * 4
        k1 = rhs(x, kappa, u)
        k2 = rhs([x[i] + dt*k1[i]/2 for i in range(4)], kappa, u)
        k3 = rhs([x[i] + dt*k2[i]/2 for i in range(4)], kappa, u)
        k4 = rhs([x[i] + dt*k3[i] for i in range(4)], kappa, u)
        xn = [x[i] + dt*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 for i in range(4)]
        if min(xn) < -1e-7:
            raise ArithmeticError(f"Negative damage state encountered: {min(xn)}")
        total_new = sum(xn)
        if total_new >= threshold:
            if total_new == total_old:
                return (step+1)*dt
            frac = (threshold-total_old)/(total_new-total_old)
            return step*dt + dt*max(0.0, min(1.0, frac))
        x = xn
        total_old = total_new
    return float("nan")


def generate_outputs(root: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    figures = root / "figures"
    results = root / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    np.set_printoptions(precision=6, suppress=True)

    # Enumerate all feasible periodic orders at every coupling level.
    rows = []
    summaries = []
    for k in KAPPAS:
        vals = []
        for p in PERMS:
            t = simulate(k, p, "schedule")
            vals.append((t, p))
            rows.append({"kappa": k, "order_indices": "-".join(map(str,p)),
                         "order_names": " -> ".join(STATE_NAMES[i] for i in p),
                         "first_passage_time": t, "total_repair_flux": R_TOTAL,
                         "repair_dose_per_class_per_cycle": R_TOTAL*SLOT})
        ts = np.array([v[0] for v in vals])
        uniform = simulate(k, mode="uniform")
        uniform_plus10 = simulate(k, mode="uniform", repair=R_TOTAL*1.1)
        best_t, best_p = max(vals, key=lambda v: v[0])
        worst_t, worst_p = min(vals, key=lambda v: v[0])
        med = float(np.median(ts))
        summaries.append({"kappa": k, "n_orders": len(vals), "minimum_time": float(ts.min()),
                          "median_time": med, "maximum_time": float(ts.max()),
                          "order_spread": float(ts.max()-ts.min()),
                          "relative_order_spread_pct": float(100*(ts.max()-ts.min())/med),
                          "uniform_same_flux_time": uniform,
                          "uniform_plus_10pct_repair_time": uniform_plus10,
                          "best_order": " -> ".join(STATE_NAMES[i] for i in best_p),
                          "worst_order": " -> ".join(STATE_NAMES[i] for i in worst_p),
                          "best_vs_uniform_pct": float(100*(best_t-uniform)/uniform)})
    with (results/"schedule_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (results/"coupling_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)

    # Sensitivity of repair budget at the primary coupling.
    budget_rows=[]
    for repair in [0.108, 0.12, 0.132, 0.144]:
        vals=[(simulate(1.0,p,"schedule",repair=repair),p) for p in PERMS]
        tv=np.array([v[0] for v in vals])
        budget_rows.append({"kappa":1.0,"repair_flux":repair,"repair_change_pct":100*(repair/R_TOTAL-1),
                            "min_schedule_time":float(tv.min()),"median_schedule_time":float(np.median(tv)),
                            "max_schedule_time":float(tv.max()),"uniform_time":simulate(1.0,mode="uniform",repair=repair),
                            "uniform_time_dt_half":simulate(1.0,mode="uniform",repair=repair,dt=DT/2)})
    with (results/"repair_budget_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(budget_rows[0]));w.writeheader();w.writerows(budget_rows)

    # State trajectories for selected order and uniform control at kappa=1.
    def trajectory(kappa, perm=None, mode="schedule", repair=R_TOTAL, dt=DT, tmax=14.0):
        x=np.array(X0,float); ts=[0.0]; xs=[x.copy()]
        nsteps=int(round(tmax/dt)); nslot=int(round(SLOT/dt)); nperiod=4*nslot
        for s in range(nsteps):
            if mode=="schedule":
                u=np.zeros(4);u[perm[(s%nperiod)//nslot]]=repair
            elif mode=="uniform": u=np.full(4,repair/4)
            else: u=np.zeros(4)
            xx=x.tolist()
            k1=np.array(rhs(xx,kappa,u.tolist()))
            k2=np.array(rhs((x+dt*k1/2).tolist(),kappa,u.tolist()))
            k3=np.array(rhs((x+dt*k2/2).tolist(),kappa,u.tolist()))
            k4=np.array(rhs((x+dt*k3).tolist(),kappa,u.tolist()))
            x=x+dt*(k1+2*k2+2*k3+k4)/6
            ts.append((s+1)*dt);xs.append(x.copy())
            if x.sum()>=THRESHOLD: break
        return np.array(ts),np.array(xs)
    best_at1=max([(simulate(1,p),p) for p in PERMS], key=lambda v:v[0])[1]
    worst_at1=min([(simulate(1,p),p) for p in PERMS], key=lambda v:v[0])[1]
    t_best,x_best=trajectory(1.0,best_at1,"schedule")
    t_worst,x_worst=trajectory(1.0,worst_at1,"schedule")
    t_unif,x_unif=trajectory(1.0,None,"uniform")
    with (results/"trajectory_sample.csv").open("w",newline="",encoding="utf-8") as f:
        cols=["time","best_order_total","worst_order_total","uniform_total"]+[f"best_{n.lower().replace(' ','_')}" for n in STATE_NAMES]
        w=csv.writer(f); w.writerow(cols)
        for idx,t in enumerate(t_best):
            def interp(arr_t, arr_x, t_query):
                if t_query>=arr_t[-1]: return arr_x[-1]
                j=max(0,min(len(arr_t)-2,int(t_query/DT)))
                frac=(t_query-arr_t[j])/(arr_t[j+1]-arr_t[j])
                return arr_x[j]+frac*(arr_x[j+1]-arr_x[j])
            xb=interp(t_best,x_best,t); xw=interp(t_worst,x_worst,t); xu=interp(t_unif,x_unif,t)
            w.writerow([t,xb.sum(),xw.sum(),xu.sum(),*xb.tolist()])

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,"axes.labelsize":9,
                         "legend.fontsize":8,"figure.dpi":150,"savefig.dpi":220,"axes.spines.top":False,"axes.spines.right":False})
    # Figure 1: coupling sweep, schedule bounds vs uniform.
    ks=np.array(KAPPAS)
    mins=np.array([s["minimum_time"] for s in summaries]); meds=np.array([s["median_time"] for s in summaries]); maxs=np.array([s["maximum_time"] for s in summaries]); unis=np.array([s["uniform_same_flux_time"] for s in summaries])
    fig,ax=plt.subplots(figsize=(7.2,4.1),constrained_layout=True)
    ax.fill_between(ks,mins,maxs,alpha=.18,label="Range across 24 orders")
    ax.plot(ks,meds,marker="o",lw=2,label="Median scheduled order")
    ax.plot(ks,unis,marker="s",lw=1.8,ls="--",label="Continuous equal-share control")
    ax.set(xlabel=r"Coupling multiplier $\kappa$ (synthetic parameter)",ylabel="First-passage time (model units; log scale)",title="Interaction strength changes failure-time scale; order spread stays bounded")
    ax.set_yscale("log")
    ax.legend(frameon=False);ax.grid(alpha=.22);fig.savefig(figures/"figure1_coupling_sweep.pdf");fig.savefig(figures/"figure1_coupling_sweep.png");plt.close(fig)
    # Figure 2: relative first-passage-time deviation across the 24 orders.
    data=[]
    for k in KAPPAS:
        group=[r["first_passage_time"] for r in rows if r["kappa"]==k]
        med=float(np.median(group))
        data.append([100*(v/med-1.0) for v in group])
    fig,ax=plt.subplots(figsize=(7.2,4.1),constrained_layout=True)
    ax.boxplot(data,tick_labels=[str(k) for k in KAPPAS],showmeans=True,meanline=True,patch_artist=True,showfliers=False)
    ax.axhline(0,color="black",lw=.8)
    ax.set_ylim(-1.7,1.7)
    ax.set(xlabel=r"Coupling multiplier $\kappa$",ylabel="Deviation from the per-coupling median (%)",title="Order effects are small relative to the threshold-time scale")
    ax.grid(axis="y",alpha=.22);fig.savefig(figures/"figure2_order_distribution.pdf");fig.savefig(figures/"figure2_order_distribution.png");plt.close(fig)
    # Figure 3: budget sensitivity.
    fig,ax=plt.subplots(figsize=(7.2,4.1),constrained_layout=True)
    rr=np.array([r["repair_flux"] for r in budget_rows]);bmin=np.array([r["min_schedule_time"] for r in budget_rows]);bmed=np.array([r["median_schedule_time"] for r in budget_rows]);bmax=np.array([r["max_schedule_time"] for r in budget_rows]);bu=np.array([r["uniform_time"] for r in budget_rows])
    ax.fill_between(100*(rr/R_TOTAL-1),bmin,bmax,alpha=.18,label="Scheduled-order range")
    ax.plot(100*(rr/R_TOTAL-1),bmed,marker="o",lw=2,label="Scheduled median")
    ax.plot(100*(rr/R_TOTAL-1),bu,marker="s",ls="--",lw=1.8,label="Continuous equal-share")
    ax.axvline(0,color="grey",lw=1,ls=":")
    ax.set(xlabel="Change in offered total repair flux (%)",ylabel="First-passage time (model units)",title=r"Repair budget sensitivity at $\kappa=1.0$")
    ax.legend(frameon=False);ax.grid(alpha=.22);fig.savefig(figures/"figure3_budget_sensitivity.pdf");fig.savefig(figures/"figure3_budget_sensitivity.png");plt.close(fig)
    # Figure 4: order effect and uniform +10% benchmark, using two labelled axes.
    spread_pct=np.array([s["relative_order_spread_pct"] for s in summaries]);gain=np.array([s["best_vs_uniform_pct"] for s in summaries]);plus10=np.array([100*(s["uniform_plus_10pct_repair_time"]/s["uniform_same_flux_time"]-1) for s in summaries])
    fig,ax=plt.subplots(figsize=(7.2,4.1),constrained_layout=True)
    l1=ax.plot(ks,spread_pct,marker="o",lw=2,label="Across-order range / median (left axis)")
    l2=ax.plot(ks,gain,marker="s",lw=1.8,label="Best order vs equal-share (left axis)")
    ax.axhline(0,color="black",lw=.8)
    ax.set(xlabel=r"Coupling multiplier $\kappa$",ylabel="Schedule-order effect (%)",title="Schedule effects and total-budget effects are different questions")
    ax2=ax.twinx()
    l3=ax2.plot(ks,plus10,marker="^",lw=1.8,ls="--",label="+10% total repair (right axis)")
    ax2.set_ylabel("Threshold-time increase from +10% repair (%)")
    ax.legend(l1+l2+l3,[l.get_label() for l in l1+l2+l3],frameon=False,loc="upper right")
    ax.grid(alpha=.22);fig.savefig(figures/"figure4_effect_size.pdf");fig.savefig(figures/"figure4_effect_size.png");plt.close(fig)
    # Figure 5: directed damage dependency graph (edge direction i->j if damage i contributes to j).
    G=nx.DiGraph()
    G.add_nodes_from(range(4))
    for i in range(4):
        for j in range(4):
            if A[j][i]>0: G.add_edge(i,j,weight=A[j][i])
    pos={0:(0,1),1:(1,0.2),2:(2,1),3:(1,1.8)}
    fig,ax=plt.subplots(figsize=(7.0,4.5),constrained_layout=True)
    nx.draw_networkx_nodes(G,pos,node_size=2600,node_color="#dbeafe",edgecolors="#334155",linewidths=1.2,ax=ax)
    labels={i:STATE_NAMES[i].replace(" ","\n") for i in range(4)}
    nx.draw_networkx_labels(G,pos,labels,font_size=8,ax=ax)
    nx.draw_networkx_edges(G,pos,ax=ax,arrows=True,arrowstyle="-|>",arrowsize=16,connectionstyle="arc3,rad=0.08",edge_color="#64748b",width=[1+16*G[u][v]["weight"] for u,v in G.edges()])
    edge_labels={(u,v):f"{G[u][v]['weight']:.02f}" for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=7,label_pos=.55,rotate=False,ax=ax)
    ax.set_title("Synthetic directed damage-dependency graph");ax.axis("off");fig.savefig(figures/"figure5_damage_network.pdf");fig.savefig(figures/"figure5_damage_network.png");plt.close(fig)
    # Figure 6: scheduling precedence DAG, explicitly a hypothetical constraint graph.
    D=nx.DiGraph()
    labels6={"signal":"Signal\narrives","mito":"Mitochondrial\nQC","proteo":"Proteostasis\nQC","genome":"Genome\nmaintenance","reg":"Regulatory\nQC","integrate":"Integrate /\nstop"}
    D.add_edges_from([("signal","mito"),("signal","proteo"),("mito","genome"),("proteo","reg"),("genome","integrate"),("reg","integrate")])
    # Expanded, layered layout and short wrapped labels keep node text inside nodes and away from edges.
    pos6={"signal":(0,1.4),"mito":(1.8,2.1),"proteo":(1.8,.7),"genome":(3.9,2.1),"reg":(3.9,.7),"integrate":(5.9,1.4)}
    fig,ax=plt.subplots(figsize=(8.0,3.6),constrained_layout=True)
    nx.draw_networkx_nodes(D,pos6,node_color="#fef3c7",edgecolors="#92400e",node_size=2700,ax=ax)
    nx.draw_networkx_labels(D,pos6,labels=labels6,font_size=7.2,ax=ax)
    nx.draw_networkx_edges(D,pos6,arrows=True,arrowstyle="-|>",arrowsize=16,width=1.4,edge_color="#92400e",ax=ax)
    ax.set_xlim(-0.75,6.65);ax.set_ylim(-0.20,2.85)
    ax.set_title("Max-plus precedence example (conceptual, not inferred biology)");ax.axis("off");fig.savefig(figures/"figure6_precedence_dag.pdf");fig.savefig(figures/"figure6_precedence_dag.png");plt.close(fig)
    # Figure 7: research workflow flow diagram.
    fig,ax=plt.subplots(figsize=(7.2,3.7),constrained_layout=True);ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis("off")
    boxes=[(0.4,3.3,2.0,1.0,"Define state +\nthreshold"),(2.9,3.3,2.0,1.0,"Enumerate 24\nrepair orders"),(5.4,3.3,2.0,1.0,"Simulate matched\nflux"),(7.9,3.3,1.7,1.0,"Compare with\ncontrols"),(2.1,0.7,2.2,1.0,"Numerical +\nedge-case tests"),(5.3,0.7,2.6,1.0,"Report conditional result +\nfalsification gate")]
    for x,y,w,h,txt in boxes: ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc="#eff6ff",ec="#1d4ed8",lw=1.2));ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=8)
    for p1,p2 in [((2.4,3.8),(2.9,3.8)),((4.9,3.8),(5.4,3.8)),((7.4,3.8),(7.9,3.8)),((8.75,3.3),(6.7,1.7)),((5.3,1.2),(4.3,1.2)),((2.1,1.2),(1.4,3.3))]:
        ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=12,color="#475569",lw=1.1,connectionstyle="arc3,rad=0.0"))
    ax.set_title("Reproducible model-audit pipeline",fontsize=11);fig.savefig(figures/"figure7_research_flow.pdf");fig.savefig(figures/"figure7_research_flow.png");plt.close(fig)
    # Figure 8: proposed experiment design flow.
    fig,ax=plt.subplots(figsize=(7.2,4.0),constrained_layout=True);ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis("off")
    nodes=[(2.9,4.7,4.2,.8,"1. Measure pathway reporters and total flux"),(2.9,3.4,4.2,.8,"2. Calibrate interaction + threshold model"),(2.9,2.1,4.2,.8,"3. Randomize schedules at matched flux"),(2.9,.8,4.2,.8,"4. Compare survival / function; preregister failure rule")]
    for x,y,w,h,txt in nodes: ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07",fc="#ecfdf5",ec="#047857",lw=1.2));ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=8.3)
    for y1,y2 in [(4.7,4.2),(3.4,2.9),(2.1,1.6)]: ax.add_patch(FancyArrowPatch((5,y1),(5,y2),arrowstyle="-|>",mutation_scale=13,color="#047857",lw=1.3))
    ax.set_title("Prospective falsification study (not performed)",fontsize=11);fig.savefig(figures/"figure8_experiment_flow.pdf");fig.savefig(figures/"figure8_experiment_flow.png");plt.close(fig)

    # Results and explicit verification checks.
    checks=[]
    checks.append(("matrix_nonnegative", bool(np.all(np.array(A)>=0)), "All cross-damage entries are nonnegative"))
    checks.append(("zero_coupling_order_invariance", max(r["first_passage_time"] for r in rows if r["kappa"]==0)-min(r["first_passage_time"] for r in rows if r["kappa"]==0)<1e-7, "At kappa=0, order cannot change summed damage trajectory under symmetric initial/source/repair dose"))
    checks.append(("all_24_orders_each_coupling", len(rows)==len(KAPPAS)*24, "Every permutation enumerated at every coupling value"))
    checks.append(("fixed_flux_dose_match", all(abs(r["repair_dose_per_class_per_cycle"]-R_TOTAL*SLOT)<1e-12 for r in rows), "Every schedule gives each class equal offered dose per cycle"))
    checks.append(("positive_first_passage_times", all(np.isfinite(r["first_passage_time"]) and r["first_passage_time"]>0 for r in rows), "Every specified synthetic schedule reaches threshold in the simulated horizon"))
    # Euler-independent convergence by RK4 step halving for representative points.
    conv=[]
    for k,p in [(0.0,(0,1,2,3)),(0.5,(2,1,3,0)),(1.0,(2,1,3,0)),(1.25,(0,3,1,2))]:
        t1=simulate(k,p,dt=DT);t2=simulate(k,p,dt=DT/2)
        conv.append({"kappa":k,"order":"-".join(map(str,p)),"dt":DT,"dt_half":DT/2,"time_dt":t1,"time_dt_half":t2,"absolute_difference":abs(t1-t2)})
    with (results/"convergence_checks.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(conv[0]));w.writeheader();w.writerows(conv)
    checks.append(("rk4_step_halving", max(r["absolute_difference"] for r in conv)<0.03, f"Maximum first-passage difference after step halving is {max(r['absolute_difference'] for r in conv):.6f}"))
    # Analytical limiting check at kappa=0: total growth slope = sum(b)-R.
    expected=(THRESHOLD-sum(X0))/(sum(B_SOURCE)-R_TOTAL)
    observed=simulate(0.0,(0,1,2,3))
    checks.append(("analytical_decoupled_limit", abs(observed-expected)<0.03, f"Analytic {expected:.6f}, simulated {observed:.6f}"))
    try:
        simulate(1.0,(0,0,1,2))
        invalid_rejected=False
    except ValueError:
        invalid_rejected=True
    checks.append(("invalid_order_rejected", invalid_rejected, "Malformed order raises ValueError"))
    try:
        simulate(-0.1)
        negative_rejected=False
    except ValueError:
        negative_rejected=True
    checks.append(("negative_coupling_rejected", negative_rejected, "Negative coupling raises ValueError"))
    with (results/"verification_checks.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["check","status","evidence"])
        for n,ok,msg in checks:w.writerow([n,"PASS" if ok else "FAIL",msg])
    report={"status":"PASS" if all(ok for _,ok,_ in checks) else "FAIL","generated_at_utc":datetime.now(timezone.utc).isoformat(),
            "model":"4-state deterministic nonnegative linear damage propagation with periodic allocated repair capacity",
            "interpretation":"synthetic model verification only; no organismal or experimental validation",
            "parameters":{"A":A,"source":B_SOURCE,"initial_state":X0,"total_repair_flux":R_TOTAL,"slot_duration":SLOT,"cycle_duration":PERIOD,"threshold":THRESHOLD,"dt":DT,"kappa_values":KAPPAS},
            "checks_total":len(checks),"checks_passed":sum(ok for _,ok,_ in checks),"checks_failed":sum(not ok for _,ok,_ in checks),
            "checks":[{"name":n,"status":"PASS" if ok else "FAIL","evidence":msg} for n,ok,msg in checks],
            "sweep_points":len(rows),"permutations_per_coupling":24,
            "best_order_at_kappa_1":next(s["best_order"] for s in summaries if s["kappa"]==1.0),
            "worst_order_at_kappa_1":next(s["worst_order"] for s in summaries if s["kappa"]==1.0)}
    (results/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (results/"verification_stdout.txt").write_text("Repair-scheduling model audit\n"+"\n".join(f"{'PASS' if ok else 'FAIL'} {n}: {msg}" for n,ok,msg in checks)+f"\nStatus: {report['status']} ({report['checks_passed']}/{report['checks_total']})\n",encoding="utf-8")
    if not all(ok for _,ok,_ in checks): raise RuntimeError("One or more model verification checks failed")
    return summaries, budget_rows, report


def write_csv(path, headers, rows):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)


def build_release(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    # Clear contents only inside this newly designated local output directory.
    for p in list(root.iterdir()):
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
    summaries,budget_rows,report=generate_outputs(root)
    res=root/"results"
    # Candidate comparison: qualitative triage, based on topical fit and literature overlap, not objective measurements.
    candidates=[
      ("Repair-order effects with propagation thresholds",[5,4,5,4,5,4,4,5,5,5],"Selected; directly tests the supplied hypothesis and is falsifiable in a compact model."),
      ("Heteroplasmy-phase control by front-loaded mitophagy",[4,4,4,4,4,4,4,3,5,4],"Close to an existing archived release; do not duplicate."),
      ("Spectral damage-reproduction threshold",[4,4,4,4,5,4,4,4,5,5],"Close to the existing Damage R0 release; retain as related work."),
      ("Information loss as an aging mechanism",[5,4,3,3,4,3,5,3,4,4],"Substantial existing literature including the 2023 Information Theory of Aging."),
      ("Percolation of mitochondrial damage clusters",[4,3,5,3,4,3,4,3,5,4],"Potential extension; current model has no spatial/cluster data."),
      ("Max-plus scheduling for multi-pathway repair jobs",[3,3,4,5,5,4,3,5,5,5],"Useful engineering abstraction when genuine precedence constraints are measured."),
      ("Ultrametric repair priorities from damage lineage trees",[3,2,5,2,4,3,3,2,4,4],"High novelty risk and weak operational grounding without lineage data."),
      ("Stochastic HJB repair policy with first-passage objective",[5,4,4,4,5,5,5,3,5,5],"Strong future route; full HJB solver and calibration beyond this release."),
      ("Reinforcement-learned cross-species longevity control",[5,3,5,2,4,5,5,2,4,2],"Empirical calibration absent; unsafe to imply actionable longevity benefit."),
      ("Repair flux as a threshold in cellular coding capacity",[5,3,5,3,4,3,4,4,5,5],"The coding analogy is useful only after defining an observable decoder-like threshold."),
    ]
    crit=["importance","gap_evidence","boldness","plausibility","math_core","AI_role","human_benefit","validation_feasibility","falsifiability","safety"]
    with (res/"candidate_topics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["candidate"]+crit+["mean_score_1_to_5","decision","rationale"])
        for i,(name,scores,rationale) in enumerate(candidates):w.writerow([name]+scores+[f"{sum(scores)/len(scores):.2f}","SELECTED" if i==0 else "ALTERNATIVE",rationale])
    (res/"author_profile_audit.md").write_text(f"""# Author profile audit

**Audit date:** {TODAY}. **Coverage:** targeted web and repository audit; not exhaustive.

- Repository identity confirmed from the connected GitHub repository: {REPO} (public repository; connected account reports push/admin capability at audit time).
- Supplied profile links: [GitHub](https://github.com/Devanik21), [LinkedIn](https://www.linkedin.com/in/devanik), [ORCID]({ORCID}), [Kaggle](https://www.kaggle.com/devanikdebnath).
- The GitHub profile contains self-reported distinctions, including Samsung Fellow @IISc and ISRO Hackathon Winner. These were not independently corroborated during this cycle and are omitted from the manuscript.
- An exact-identifier web query did not establish an ORCID page result confidently matching this author, and targeted web searches did not verify a Zenodo, OpenAIRE, or Google Scholar record. This is **not** evidence that no such account or record exists; no citation count or DOI is asserted.
- Author/affiliation as supplied for the manuscript: Devanik Debnath; Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India. No degree completion or external endorsement is implied.
""",encoding="utf-8")
    (res/"literature_audit.md").write_text("""# Literature audit (targeted, non-systematic)

**Audit date:** 2026-10-24. **Queries:** `optimal control of aging in complex networks`; `repair scheduling fixed maintenance throughput aging`; `Information Theory of Aging`; `coordination of mitophagy and mitochondrial biogenesis C. elegans`; `autophagy aging spatiotemporal regulation`; `repair order threshold damage network`; plus exact DOI/title searches for principal sources. Search scope was targeted web search and a review of related releases already present in the destination archive; it was not a systematic review.

## Relevant prior work and overlap

1. Sun, Michaels & Mahadevan (2020) formulated an optimal-control and learning framework for longevity in an interdependent network and derived optimal maintenance schedules. This directly rules out claiming that “optimal repair scheduling for aging networks” is itself new. DOI: <https://doi.org/10.1073/pnas.2006375117>.
2. Murphy & Partridge (2008) discussed aging through control theory, feedback, and interactions across physiological systems. DOI: <https://doi.org/10.1146/annurev.biochem.77.070606.101605>.
3. Lu, Tian & Sinclair (2023) set out the Information Theory of Aging, focused on information stored in the genome/epigenome and loss of youthful epigenetic information. DOI: <https://doi.org/10.1038/s43587-023-00527-6>.
4. Palikaras, Lionaki & Tavernarakis (2015) studied coordination between mitophagy and mitochondrial biogenesis during *C. elegans* aging, demonstrating that turnover and replenishment interact. DOI: <https://doi.org/10.1038/nature14300>.
5. *Spatiotemporal regulation of autophagy during Caenorhabditis elegans aging* (eLife, 2017) describes age- and tissue-dependent autophagy patterns. DOI: <https://doi.org/10.7554/eLife.18459>.
6. Hamazaki & Murata (2024) review the relationships among protein degradation, senescence, and organismal aging. DOI: <https://doi.org/10.1093/jb/mvae016>.
7. The destination archive already contained “Heteroplasmy Phase Control” and “Damage R₀”, both close in motif. The present release explicitly treats these as related, not independent proof or priority.

## Bounded novelty statement

The defensible contribution here is a reproducible, four-state counterexample-oriented audit that (i) enumerates every periodic repair order at fixed offered total flux, (ii) separates schedule-order dispersion from total-budget changes, and (iii) gives limiting-case and step-halving checks. It is **not** a claim that schedule control for aging is new, that tropical algebra or ultrametrics improve biological repair decisions, or that any simulation extends lifespan. Literature coverage can miss neighboring work; expert review remains necessary before submission.
""",encoding="utf-8")
    (res/"research_log.md").write_text(f"""# Research workflow log

- Date: {TODAY}
- Topic: user-supplied “Aging as a decoding threshold, and lifespan extension by repair scheduling”.
- Prior-art gate: passed with narrowed scope after finding prior optimal-control work and two related archive releases.
- Main formulation: deterministic four-state damage propagation with nonnegative off-diagonal dependencies, equal per-state allocated repair dose per 12-unit cycle, 24 period-3 schedules, and threshold first-passage time.
- Generated data: synthetic/model-generated only. No public biological dataset was downloaded; no lifespan/survival cohort was analyzed.
- Numerical method: fixed-step fourth-order Runge–Kutta, dt={DT}; convergence check repeated at dt={DT/2}.
- Validation includes: all schedule permutations across six coupling values, decoupled analytical limiting case, dose matching, malformed parameter rejection, and time-step refinement.
- Excluded from quantitative core: RL policy learning, HJB solution, p-adic/ultrametric clustering, empirical percolation threshold, and comparative-species calibration. Their data/operational prerequisites are absent here.
- Output: paper manuscript, verifier, reproducible plots/graphs/flow diagrams, CSV results, metadata, and deployment validator.
- Important limitation: model variables/parameters are illustrative dimensionless quantities; no biological rates are inferred.
""",encoding="utf-8")
    (res/"claim_evidence_matrix.csv").write_text("claim_id,claim,status,evidence,limitation\nC1,At zero cross-coupling order has no effect on summed threshold trajectory,PASS,results/verification_report.json,Only under symmetric synthetic model setup\nC2,At nonzero coupling some periodic orders have different first-passage times,PASS,results/schedule_sweep.csv,Model-dependent deterministic result\nC3,All periodic orders deliver equal nominal dose per state per cycle,PASS,results/schedule_sweep.csv,Allocated capacity is fixed; biological realized flux not modelled\nC4,Optimal schedule extends real organismal lifespan,NOT_TESTED,No biological dataset or experiment,Lifespan extension is not established\nC5,Ultrametric clustering improves repair choice,HYPOTHESIS,Future-work discussion,No lineage data or quantitative test\nC6,Additional repair capacity can be compared against order effects,PASS,results/repair_budget_sensitivity.csv,Not a matched-budget comparison\n",encoding="utf-8")
    (res/"publication_design_brief.md").write_text("""# Publication design brief

- Audience: mathematical biology, complex systems, and computational longevity researchers.
- Style: restrained two-column-inspired single-column research preprint; serif body/math, deep navy headings, teal accent, no decorative cover.
- Page: US Letter, margins 0.78 in; 10pt body, 1.12 baseline, clearly separated equations.
- Figures: generated as vector PDF from Matplotlib/NetworkX, consistent labels and line styles, embedded as vector where supported.
- Tables: booktabs, consistent notation/units, compact rows and informative captions.
- Headers: running title at left and page number at right.
- Evidence policy: all simulation figures explicitly identify synthetic parameters; no biological or experimental result is implied.
""",encoding="utf-8")
    (res/"release_check_matrix.json").write_text(json.dumps({"release_status":"BLOCKED","generated_at_utc":datetime.now(timezone.utc).isoformat(),"repository":"Devanik21/The-Invention-Archive","target_branch":"main","checks":[{"name":"Model verification","status":"PASS" if report["status"]=="PASS" else "FAIL","evidence":"results/verification_report.json"},{"name":"Citation / metadata syntax","status":"NOT_RUN","evidence":"pending deploy validation"},{"name":"LaTeX compilation and page count","status":"NOT_RUN","evidence":"pending PDF build"},{"name":"Full-page visual review","status":"NOT_RUN","evidence":"pending render"},{"name":"GitHub remote artifact read-back","status":"NOT_RUN","evidence":"pending publication"},{"name":"GitHub Actions exact-SHA run closure","status":"NOT_RUN","evidence":"pending publication"}],"blockers":["Final PDF has not yet been built and visually reviewed","Remote release has not yet been deployed and verified"]},indent=2)+"\n",encoding="utf-8")
    (res/"environment.txt").write_text(f"date={TODAY}\npython={platform.python_version()}\nplatform={platform.platform()}\nnumpy={__import__('numpy').__version__}\nmatplotlib={__import__('matplotlib').__version__}\nnetworkx={__import__('networkx').__version__}\nsolver=classical fixed-step RK4\ndt={DT}\nseed=none (deterministic model)\n",encoding="utf-8")

    # Standalone verifier is this script's simulation/figure-generation logic under the same file.
    this = Path(__file__).resolve()
    shutil.copy2(this, root/"verify_model.py")
    (root/"verify_model.py").chmod(0o755)

    bib=r'''@article{Sun2020OptimalControl,
  author = {Sun, Eric D. and Michaels, Thomas C. T. and Mahadevan, L.},
  title = {Optimal control of aging in complex networks},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {117}, number = {34}, pages = {20404--20410}, year = {2020},
  doi = {10.1073/pnas.2006375117}
}
@article{MurphyPartridge2008Control,
  author = {Murphy, Michael P. and Partridge, Linda},
  title = {Toward a Control Theory Analysis of Aging},
  journal = {Annual Review of Biochemistry}, volume = {77}, pages = {777--798}, year = {2008},
  doi = {10.1146/annurev.biochem.77.070606.101605}
}
@article{Lu2023InformationAging,
  author = {Lu, Yuancheng Ryan and Tian, Xiao and Sinclair, David A.},
  title = {The Information Theory of Aging},
  journal = {Nature Aging}, volume = {3}, pages = {1486--1499}, year = {2023},
  doi = {10.1038/s43587-023-00527-6}
}
@article{Palikaras2015Mitophagy,
  author = {Palikaras, Konstantinos and Lionaki, Eirini and Tavernarakis, Nektarios},
  title = {Coordination of mitophagy and mitochondrial biogenesis during ageing in {C. elegans}},
  journal = {Nature}, volume = {521}, pages = {525--528}, year = {2015},
  doi = {10.1038/nature14300}
}
@article{Autophagy2017Wang,
  title = {Spatiotemporal regulation of autophagy during {Caenorhabditis elegans} aging},
  journal = {eLife}, volume = {6}, pages = {e18459}, year = {2017},
  doi = {10.7554/eLife.18459}
}
@article{HamazakiMurata2024Proteostasis,
  author = {Hamazaki, Jun and Murata, Shigeo},
  title = {Relationships between protein degradation, cellular senescence, and organismal aging},
  journal = {Journal of Biochemistry}, volume = {175}, number = {5}, pages = {473--480}, year = {2024},
  doi = {10.1093/jb/mvae016}
}
@book{Baccelli1992MaxPlus,
  author = {Baccelli, Fran\c{c}ois and Cohen, Guy and Olsder, Geert Jan and Quadrat, Jean-Pierre},
  title = {Synchronization and Linearity: An Algebra for Discrete Event Systems},
  publisher = {Wiley}, year = {1992}
}
@book{StaufferAharony1994Percolation,
  author = {Stauffer, Dietrich and Aharony, Amnon},
  title = {Introduction to Percolation Theory}, edition = {2}, publisher = {Taylor \& Francis}, year = {1994}
}
@book{SuttonBarto2018RL,
  author = {Sutton, Richard S. and Barto, Andrew G.},
  title = {Reinforcement Learning: An Introduction}, edition = {2}, publisher = {MIT Press}, year = {2018}
}
'''
    (root/"paper.bib").write_text(bib,encoding="utf-8")
    # Write manuscript with 6 substantive tables and 8 distinct figure assets.
    tex=r'''\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.78in,headheight=14pt,footskip=24pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{amsmath,mathtools}
\usepackage{graphicx,booktabs,longtable,tabularx,array,multirow}
\usepackage{caption,enumitem,fancyhdr,xcolor,hyperref,float}
\definecolor{ink}{HTML}{172554}
\definecolor{teal}{HTML}{0F766E}
\hypersetup{colorlinks=true,linkcolor=ink,citecolor=teal,urlcolor=teal,pdftitle={Aging as a Decoding Threshold: Repair Scheduling Under Fixed Maintenance Throughput},pdfauthor={Devanik Debnath}}
\captionsetup{font=small,labelfont={bf,color=ink},justification=raggedright,singlelinecheck=false}
\setlength{\parindent}{1em}
\setlength{\parskip}{2pt}
\setlist{nosep,leftmargin=*}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small\textit{Repair Scheduling Under Fixed Throughput}}\fancyhead[R]{\small Devanik Debnath}\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0.35pt}
\title{\vspace{-1.6em}\textbf{\color{ink}Aging as a Decoding Threshold:}\\[0.25em]\large Repair Scheduling Under Fixed Maintenance Throughput}
\author{Devanik Debnath\\\small Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India\\\small \href{https://github.com/Devanik21}{GitHub: Devanik21} \; $\vert$ \; \href{https://orcid.org/0009-0001-8310-6921}{ORCID record supplied by author}}
\date{24 October 2026 $\;\vert\;$ Theoretical and computational preprint}
\begin{document}
\maketitle
\begin{abstract}
Cells accumulate heterogeneous lesions while repair pathways have finite throughput. This paper tests a narrow version of the hypothesis that repair order can alter a threshold-defined failure time even when the offered repair budget is fixed. We construct a deterministic four-state damage-propagation model with a nonnegative interaction matrix and periodically allocate the same repair capacity to each of four abstract damage classes. All 24 class orders are enumerated across six coupling strengths; every order gives each class the same nominal repair dose per cycle. In the decoupled limit, order is analytically irrelevant. Under nonzero coupling, order changes first-passage time, but the effect is modest in the illustrative parameter range and a continuous equal-share policy is generally competitive or better. A 10\% increase in offered repair changes the threshold time more than the difference between schedules in these simulations. The results support a conditional mathematical statement—not a theory of organismal lifespan extension. We distinguish the coding-threshold analogy from literal error-correcting codes, and identify tropical scheduling, percolation, stochastic control, ultrametric clustering, and reinforcement learning as candidate extensions that require their own operational definitions and evidence.
\end{abstract}
\noindent\textbf{Keywords:} aging; damage propagation; repair scheduling; first-passage time; control theory; threshold models; reproducibility

\section{Question and contribution}
Aging is associated with changes across proteostasis, mitochondrial quality control, genome maintenance, and gene-regulatory state. These processes interact, and the timing of interventions can matter when repair is delayed, state-dependent, or coupled to downstream damage. Control-theoretic approaches to aging networks and optimal maintenance schedules already exist; in particular, Sun, Michaels, and Mahadevan studied longevity optimization in interdependent networks and repair allocation \cite{Sun2020OptimalControl}. Control-theoretic perspectives on aging predate that work \cite{MurphyPartridge2008Control}. The broad proposition that maintenance schedules can affect longevity is therefore not novel by itself.

The present paper asks a more constrained question: in a transparent model, does the ordering of repair jobs alter a threshold-crossing time when the total offered capacity and per-class dose per cycle are fixed? We also quantify the comparison against an equal-share policy and a modest increase in repair capacity. The result is intentionally model-first and counterexample-oriented. It does not infer that an organism has a literal biological decoder, or that changing pathway priority will extend lifespan.

The main contributions are: (i) a compact model with explicit variables, threshold, and scheduling constraints; (ii) exhaustive enumeration of 24 periodic repair orders at matched nominal throughput; (iii) a limiting-case result and numerical convergence checks; and (iv) a falsification-oriented interpretation that separates schedule-order effects from total-budget effects. This release makes no empirical claim and uses no animal, patient, cell-line, or comparative-species dataset.

\section{Prior work and scope}
\subsection{Aging control and maintenance}
The network-optimal-control literature directly addresses repair scheduling and longevity trade-offs \cite{Sun2020OptimalControl}. Our contribution is not a replacement for this framework. It narrows the explanatory object to a controlled comparison in which nominal repair throughput is fixed and every class receives identical cumulative offered dose over a cycle. The paper's small system is designed to expose when scheduling order has an effect and when it does not.

\subsection{Biological interactions relevant to calibration}
Mitophagy and mitochondrial biogenesis are coupled in \textit{C. elegans}; perturbing one process can alter the other, so “repair flux” cannot safely be treated as a single independent actuator without measurement \cite{Palikaras2015Mitophagy}. Autophagy regulation varies with age, tissue, and physiological context \cite{Autophagy2017Wang}. Proteostasis, degradation, senescence, and organismal aging also have complex relationships \cite{HamazakiMurata2024Proteostasis}. These studies motivate interactions but do not calibrate the abstract matrix used here.

The Information Theory of Aging discusses loss of youthful epigenetic information as a proposed aging mechanism \cite{Lu2023InformationAging}. The present “decoding threshold” phrasing is a separate operational analogy: it means an externally defined functional boundary in a state space. It does not assert that the cell is a digital communication system or that coding theory derives the biological threshold.

\begin{table}[H]
\centering\small
\caption{Position relative to closely related concepts. Classification is qualitative, based on the cited papers and current archive inspection.}
\begin{tabularx}{\linewidth}{p{0.22\linewidth}p{0.31\linewidth}X}
\toprule
\textbf{Area} & \textbf{Established contribution} & \textbf{What this paper adds / does not claim}\\
\midrule
Aging network control & Optimal control and maintenance-schedule analysis in interdependent networks \cite{Sun2020OptimalControl}. & Exhaustive 24-order matched-offered-flux audit for a four-state toy model; not a new general scheduling theory.\\
Control theory of aging & Feedback and control language applied to interacting physiological variables \cite{MurphyPartridge2008Control}. & Makes one state threshold and one order-vs-budget contrast explicit; no calibrated biological controller.\\
Information theory of aging & Epigenetic information preservation/retrieval hypothesis \cite{Lu2023InformationAging}. & Defines a functional threshold analogue only; no information-theoretic aging mechanism is demonstrated.\\
Mitochondrial maintenance & Mitophagy/biogenesis coupling and longevity relevance in \textit{C. elegans} \cite{Palikaras2015Mitophagy}. & Motivates coupling as a concern; does not infer actual pathway flux or organismal outcomes.\\
\bottomrule
\end{tabularx}
\end{table}

\section{Model and assumptions}
Let $x(t)\in\mathbb{R}_{\geq0}^{4}$ denote abstract burdens in proteostasis ($x_1$), mitochondrial state ($x_2$), genome maintenance ($x_3$), and regulatory drift ($x_4$). Units are dimensionless model units. We define
\begin{equation}
\dot{x}(t)=b+\kappa A x(t)-u(t), \label{eq:dyn}
\end{equation}
where $b_i=0.04$ is a constant independent damage source, $\kappa\geq0$ scales cross-state propagation, and $A$ is the nonnegative off-diagonal matrix shown in Table~\ref{tab:params}. The input $u_i(t)$ is offered repair capacity allocated to class $i$. In the scheduled policy, exactly one class receives capacity $R=0.12$ for a slot of duration $\tau=3$; slots repeat in a cycle of duration $T_c=12$. Each class therefore receives nominal dose $R\tau=0.36$ per cycle, and the offered total flux is $R$ at every time. In the continuous equal-share comparison, every class receives $R/4$ at all times. The two policies match in total offered flux and per-class dose over a complete cycle.

Define the failure proxy as the first time the sum of burdens reaches a threshold $\theta=8$:
\begin{equation}
\tau_f=\inf\{t\geq0:\mathbf{1}^{\top}x(t)\geq\theta\}, \qquad x(0)=(0.8,0.8,0.8,0.8)^{\top}. \label{eq:threshold}
\end{equation}
Here $\tau_f$ is a model first-passage time. Calling it “lifespan” would be unjustified without a map from state variables to a measured organismal endpoint.

\begin{table}[H]
\centering\small
\caption{State variables, units, and modeled interpretation.}
\label{tab:states}
\begin{tabularx}{\linewidth}{p{0.19\linewidth}p{0.18\linewidth}X}
\toprule
\textbf{Coordinate} & \textbf{Symbol} & \textbf{Interpretation and boundary}\\
\midrule
Proteostasis & $x_1$ & Abstract burden proxy for misfolded/aggregation-related dysfunction; not a measured protein-quality variable.\\
Mitochondrial state & $x_2$ & Abstract burden proxy for mitochondrial dysfunction; it does not distinguish mitophagy from biogenesis.\\
Genome maintenance & $x_3$ & Abstract burden proxy for lesions and repair load; mutations are not assumed to be reversible by the input.\\
Regulatory drift & $x_4$ & Abstract burden proxy for regulatory/epigenetic deviation; no specific epigenetic clock is modeled.\\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]
\centering\small
\caption{Synthetic model parameterization; values are chosen for illustration and not fitted to biology.}
\label{tab:params}
\begin{tabular}{lll}
\toprule
\textbf{Quantity} & \textbf{Value} & \textbf{Role}\\
\midrule
$A$ & $\begin{bmatrix}0&.08&0&.04\\.05&0&.02&.08\\.08&.03&0&0\\.02&0&.07&0\end{bmatrix}$ & Directed coupling topology\\
$b_i$ & $0.04$ & Constant source per class\\
$R$ & $0.12$ & Offered total repair capacity\\
$\tau$ & $3$ & Slot duration\\
$T_c$ & $12$ & Repair-order cycle\\
$\theta$ & $8$ & Failure threshold\\
$\kappa$ & $0$ to $1.25$ & Synthetic propagation multiplier\\
$\Delta t$ & $0.02$ & RK4 step size, halved in convergence check\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Exact solution identity and a limiting case}
For any piecewise-constant offered input and constant matrix $M=\kappa A$, Eq.~\ref{eq:dyn} has the variation-of-constants identity
\begin{equation}
x(t)=e^{Mt}x(0)+\int_0^t e^{M(t-s)}b\,ds-\int_0^t e^{M(t-s)}u(s)\,ds.\label{eq:variation}
\end{equation}
The final term weights each repair action by how much time remains for its effect to propagate. Therefore equal integrals $\int_0^{T_c}u(s)ds$ do not generally imply equal states when $M\neq0$. At $\kappa=0$, however, the sum obeys
\begin{equation}
\frac{d}{dt}\mathbf{1}^{\top}x(t)=\mathbf{1}^{\top}b-\mathbf{1}^{\top}u(t),\label{eq:sum}
\end{equation}
which depends only on the total offered repair rate. With a symmetric initial state and equal per-cycle dose, changing the class order cannot change the summed burden; the predicted threshold times are identical. This is an analytic limiting-case check, not a universal theorem about real repair systems.

\section{Numerical protocol}
We enumerate every permutation $\pi\in S_4$ for $\kappa\in\{0,0.25,0.5,0.75,1,1.25\}$. The system is integrated using fixed-step fourth-order Runge--Kutta with $\Delta t=0.02$; this step divides the 3-unit slot exactly. First-passage time is linearly interpolated across the threshold crossing. We compare (i) all 24 scheduled orders, (ii) continuous equal-share repair at the same offered total flux, and (iii) continuous equal-share repair with a 10\% larger total offered flux. The third comparison is deliberately not a matched-budget comparison; it measures whether a small budget increase can dominate the scheduling effect.

\begin{table}[H]
\centering\small
\caption{Model checks and evidence produced by the executable verifier.}
\label{tab:checks}
\begin{tabularx}{\linewidth}{p{0.25\linewidth}p{0.18\linewidth}X}
\toprule
\textbf{Check} & \textbf{Status} & \textbf{Interpretation}\\
\midrule
Nonnegative coupling matrix & \texttt{PASS} & Off-diagonal entries are nonnegative by construction.\\
All schedule orders & \texttt{PASS} & 24 orders are evaluated for each of six coupling values (144 runs).\\
Equal offered dose & \texttt{PASS} & Each class is allocated $R\tau=0.36$ per cycle.\\
Decoupled limiting case & \texttt{PASS} & Numerical result agrees with the analytic summed-state threshold time within tolerance.\\
Time-step refinement & \texttt{PASS} & Representative trajectories are re-evaluated with half the step size; maximum difference is reported in \texttt{results/convergence\_checks.csv}.\\
Experimental validity & \texttt{NOT TESTED} & The model contains no measured biological rates or organismal observations.\\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[H]\centering\includegraphics[width=0.86\linewidth]{figures/figure1_coupling_sweep.pdf}\caption{First-passage time as cross-state coupling multiplier varies. Shaded band covers all 24 periodic orders; the dashed curve is continuous equal-share repair at the same offered total flux. The coupling multiplier is not an empirical correlation coefficient.}\label{fig:sweep}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=0.86\linewidth]{figures/figure2_order_distribution.pdf}\caption{Distribution of threshold times across all 24 repair orders at each coupling value. This is a deterministic enumeration of schedules, not a sample of organisms.}\label{fig:distribution}\end{figure}

\section{Results}
The exhaustive sweep verifies that schedule order is irrelevant in the decoupled limit under the symmetric set-up, but the first-passage time depends on order once the off-diagonal coupling terms are active. The spread is small relative to the absolute first-passage scale in the tested range. At $\kappa=1$, the simulated schedule-order range is reported in Table~\ref{tab:summary} from the generated CSV; the exact best and worst orders are preserved in the machine-readable outputs. A continuous equal-share policy provides a useful baseline and is close to the median scheduled outcome. In this configuration, some orders are slightly better and some slightly worse than the equal-share policy. This is precisely why “repair order matters” is too weak to imply “a learned schedule is superior.”

\begin{table}[H]
\centering\small
\caption{Summary of fixed-throughput sweep. Values are generated by \texttt{verify\_model.py}; units are abstract model time.}
\label{tab:summary}
\begin{tabular}{rrrrr}
\toprule
$\kappa$ & Min order & Median order & Max order & Equal-share\\
\midrule
0.00 & 120.00 & 120.00 & 120.00 & 120.00\\
0.25 & 24.23 & 24.48 & 24.71 & 24.47\\
0.50 & 13.56 & 13.71 & 13.84 & 13.70\\
0.75 & 9.40 & 9.52 & 9.63 & 9.52\\
1.00 & 7.20 & 7.29 & 7.38 & 7.29\\
1.25 & 5.84 & 5.91 & 5.97 & 5.91\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering\includegraphics[width=0.86\linewidth]{figures/figure3_budget_sensitivity.pdf}\caption{Changing offered repair capacity at $\kappa=1$. The shaded range represents schedule-order variation, while the dashed line represents continuous equal-share repair at each capacity. This comparison changes the total budget and must not be mistaken for an equal-budget test.}\label{fig:budget}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=0.86\linewidth]{figures/figure4_effect_size.pdf}\caption{Relative schedule dispersion, best-order comparison, and a 10\% total-repair increase. The budget increase generally changes the threshold time more than choosing among the 24 orders in this toy model.}\label{fig:effects}\end{figure}

Table~\ref{tab:summary} uses a rounded display of the deterministic output. Full precision, order labels, and all 144 runs are available in \texttt{results/schedule\_sweep.csv}. The budget experiment shows that a uniform increase in offered repair can dominate the matched-budget schedule-order range near the threshold. This result is not universal: it depends on the chosen $A$, source, initial state, threshold, and repair actuator. It does, however, contradict any unqualified claim that order alone should outperform a modest increase in repair capacity.

\begin{table}[H]
\centering\small
\caption{Interdisciplinary ingredients: operational role and decision to include or defer.}
\label{tab:fields}
\begin{tabularx}{\linewidth}{p{0.21\linewidth}p{0.35\linewidth}X}
\toprule
\textbf{Field} & \textbf{Concrete role} & \textbf{Status in this release}\\
\midrule
Coding theory & A threshold is analogous to an uncorrectable functional error pattern; define the boundary through an observable. & Analogy only; no codeword or decoder is modeled.\\
Max-plus / tropical scheduling & Precedence completion: $C_j=p_j+\max_{i\in\mathrm{Pred}(j)}C_i$. & Illustrative precedence DAG only; no measured pathway precedence constraints.\\
Percolation & Potential cluster/connectivity-based failure criterion for spatially localized damage. & Deferred; no spatial graph, cluster distribution, or percolation threshold is fitted.\\
Stochastic optimal control & Minimize expected first-passage time subject to a capacity constraint. & Motivation only; the present solver is deterministic and enumerative, not an HJB solver.\\
Ultrametric / p-adic analysis & Could cluster damage lineages if distances derive from a measured branching hierarchy. & Excluded from the core; no lineage data or calibrated ultrametric is available.\\
Reinforcement learning & Learn policy under a validated simulator and compare to exact/heuristic baselines. & Not run. Adding RL without a calibrated objective would add complexity, not evidence.\\
Comparative biology & Test parameter transfer and external validity across species. & Not performed; no interspecies data were downloaded or analyzed.\\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[H]\centering\includegraphics[width=0.72\linewidth]{figures/figure5_damage_network.pdf}\caption{Directed dependency graph used by the synthetic linear model. Arrow weights are the entries of the illustrative matrix $A$; they are not measured biological interaction strengths.}\label{fig:network}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=0.82\linewidth]{figures/figure6_precedence_dag.pdf}\caption{A conceptual max-plus precedence graph. Edges are hypothetical constraints for explaining scheduling algebra and are not inferred from biology or used as measured causal edges in the simulation.}\label{fig:precedence}\end{figure}

\section{Interpretation and theoretical boundaries}
\subsection{The decoding-threshold analogy}
The term “decoding threshold” can be useful only if the failure criterion is operational. In this work it is the scalar boundary $\mathbf{1}^{\top}x=\theta$, not a threshold theorem for biological codes. Classical error correction depends on a defined code, channel, error model, and decoder; none is identified here. A biological application would need a measurable functional state, a task-specific criterion for loss of function, and evidence that distinct lesions combine nonlinearly around that criterion.

\subsection{Max-plus and tropical geometry}
Max-plus algebra is natural for discrete-event systems with precedence constraints, in which completion of a job depends on the maximum completion time among its predecessors \cite{Baccelli1992MaxPlus}. It predicts a scheduling bottleneck only if pathway or repair-task dependencies can be written as actual prerequisites with identifiable processing times. Figure~\ref{fig:precedence} is therefore a vocabulary illustration, not a biological discovery. A falsifiable quantitative prediction would be that the measured critical path predicts repair completion delay better than a sum-of-times baseline, out of sample.

\subsection{Percolation and damage correlation}
Percolation theory offers tools to study when connected damaged regions lose connectivity \cite{StaufferAharony1994Percolation}. But the current model has no spatial lattice or measured damage clusters; $\kappa$ scales causal cross-state propagation and is not a correlation coefficient. A next study could compare threshold time with a cluster-percolation order parameter on image-registered mitochondrial networks or chromatin domains. It should test whether cluster statistics add predictive value beyond total damage and local burden.

\subsection{Stochastic control and reinforcement learning}
An optimal-control formulation could minimize expected first-passage time, $\mathbb{E}[\tau_f]$, while limiting total repair capacity. A proper stochastic model would require process noise, state-dependent repair effectiveness, observation noise, and a justified policy constraint. Reinforcement learning is relevant only after a calibrated environment exists and must be compared with a non-learning policy, the uniform baseline, and an exact small-state solution where available \cite{SuttonBarto2018RL}. No RL training result is claimed in this paper.

\subsection{Ultrametrics and damage lineages}
An ultrametric distance might be appropriate for damage states that descend through a branching lineage, but only if a lineage tree or ancestry-aware distance is measured rather than imposed. P-adic language is not used as decorative machinery here because no biological observation currently makes that extra structure necessary. A meaningful future test would ask whether lineage-derived ultrametric clustering predicts the best repair priority across held-out perturbations after controlling for burden, cell identity, and lineage depth.

\section{Falsification and experimental plan}
A realistic validation programme should begin with a tractable system such as \textit{C. elegans} or yeast, but only after directly measuring the candidate actuator and total repair activity. The proposed experiment is not performed in this release. A minimal study would: (1) characterize autophagy and mitophagy reporters and their response delays; (2) estimate the interaction matrix and threshold using independently measured perturbations; (3) randomize two or more pathway-priority schedules while matching integrated repair flux and per-pathway dose; (4) pre-register a functional endpoint and a minimum relevant effect size; and (5) report survival curves, uncertainty intervals, flux matching error, and off-target effects. A reporter-derived flux proxy is not automatically the amount of functional damage removed.

The hypothesis that schedule order helps in this context is falsified if order effects are absent within a pre-specified equivalence margin despite sufficient temporal resolution and verified flux matching, or if any apparent benefit disappears after controlling for total activity, delay, toxicity, and baseline damage. A null result with wide uncertainty is inconclusive. If continuous equal-share or a simple rule-based schedule equals or beats a learned policy on held-out data, the claimed advantage of AI scheduling fails. In multicellular models, lineage-level damage tracing and species-specific calibration would remain major sources of uncertainty.

\begin{figure}[H]\centering\includegraphics[width=0.9\linewidth]{figures/figure7_research_flow.pdf}\caption{Actual computational workflow in this release: model specification, exhaustive order sweep, matched-flux comparison, and verification.}\label{fig:researchflow}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=0.8\linewidth]{figures/figure8_experiment_flow.pdf}\caption{Prospective experimental sequence. It is a proposed validation design, not a completed experiment or protocol for lifespan intervention.}\label{fig:experiment}\end{figure}

\begin{table}[H]
\centering\small
\caption{Failure modes, observable consequences, and the result that would weaken the central claim.}
\label{tab:falsification}
\begin{tabularx}{\linewidth}{p{0.22\linewidth}p{0.33\linewidth}X}
\toprule
\textbf{Failure mode} & \textbf{Observable / diagnostic} & \textbf{Decision rule}\\
\midrule
No true interaction & $\kappa\approx0$ or empirical off-diagonal effects near zero. & Order advantage should vanish; any estimated persistent effect suggests model/measurement misspecification.\\
Repair actuator delayed & Measured response-time distribution by pathway. & Add pathway state and delay; reject zero-delay ordering predictions if the new model does not reproduce held-out responses.\\
Flux mismatch & Integrated reporter/turnover activity per pathway. & Do not attribute survival differences to order until equivalence bounds on total and per-class flux are met.\\
Threshold arbitrary & Change-point/functional endpoint uncertainty. & Report sensitivity to the threshold definition; reject claims that depend on one unsupported cutoff.\\
Compensatory turnover & Biogenesis, growth, or replenishment changes. & Extend the state vector and re-test; current result is not valid if omitted feedback reverses rankings.\\
No external validity & Held-out perturbations or species. & If order rankings do not transfer, limit claim to the calibrated system.\\
\bottomrule
\end{tabularx}
\end{table}

\section{Limitations}
The model is deliberately small and deterministic. Damage is assumed to separate into four nonnegative classes with constant source and linear interactions, while the repair input has no observation noise, delays, toxicity, saturation, or state-dependent effectiveness. The failure threshold is a scalar sum and omits compensatory mechanisms, cell turnover, organismal physiology, and competing risks. The matrix and all numerical constants are synthetic, dimensionless choices. Thus the numerical range is a proof-of-implementation for this toy model, not a prediction of days of life, percent lifespan extension, or treatment effect. Even the “matched flux” comparator matches offered repair capacity; it does not represent measured realized biochemical flux under a real pathway.

The tropical, percolation, ultrametric, HJB, and RL concepts are not integrated into the numerical core because doing so without data or independently validated assumptions would be mathematical decoration. The paper gives each one an operational prerequisite and a falsification route instead. The topic's broader claim that optimal scheduling can extend lifespan remains open and requires experiments and model calibration that are outside this release.

\section{Conclusion}
Repair order can affect threshold-crossing time in an interacting damage model even when the offered total repair rate and dose per class per cycle are fixed. In the specified four-state example, however, the observed order effect is conditional and comparatively modest; a continuous equal-share policy is competitive, and a 10\% budget increase can produce a larger change. The decoupled limit provides an exact null case in which order is irrelevant. These results do not prove that repair scheduling extends lifespan. They motivate a more restrictive experimental question: after verifying interactions, delay, and matched pathway activity, does repair order improve a predefined functional threshold or survival endpoint over a simple equal-share baseline?

\section*{Research-process disclosure}
Co-equal research-process credit is given to Devanik Debnath and Xylia, the autonomous multi-model research engine coordinating AI-assisted discovery, synthesis, and verification. Devanik Debnath retains responsibility for the final manuscript, scientific claims, citations, and released artifacts.

\bibliographystyle{unsrt}
\bibliography{paper}
\end{document}
'''
    (root/"paper.tex").write_text(tex,encoding="utf-8")

    readme=f'''# {TITLE}

**Author:** Devanik Debnath (Devanik21)
**Affiliation:** {AFFILIATION}
**Date:** {TODAY}
**Status:** Theoretical/modeling preprint; no empirical lifespan extension or clinical efficacy is claimed.

## Main question

At fixed offered repair throughput, can changing the order of repair tasks change a threshold-defined failure time in a coupled damage model? The paper uses “decoding threshold” as an operational analogy for a functional boundary, not a literal biological error-correcting code.

## Model

For the four-dimensional burden vector $x(t)$, constant source vector $b$, synthetic nonnegative coupling matrix $A$, coupling multiplier $\\kappa$, and allocated repair input $u(t)$:

$$
\\dot{{x}}(t) = b + \\kappa A x(t) - u(t).
$$

The modeled failure proxy is the first-passage time to the threshold $\\theta=8$:

$$
\\tau_f = \\inf\\{{t \\geq 0 : \\mathbf{{1}}^\\top x(t) \\geq \\theta\\}}.
$$

The scheduled policy allocates $R=0.12$ to one of four damage classes for three time units per slot, then cycles to the next. All 24 orders are enumerated. Each class receives the same nominal $0.36$ offered dose per 12-unit cycle. Comparison is against continuous equal-share repair at the same total offered capacity, plus a deliberately non-matched 10% budget increase.

## Interpretation boundary

- Every parameter and model output is synthetic; no biological rate was fitted.
- $\\kappa$ is an interaction-strength multiplier, not a measured correlation coefficient.
- The threshold time is not organismal lifespan.
- Reinforcement learning and stochastic HJB control were not run; ultrametric clustering and percolation are future hypotheses, not results.
- Prior work already includes optimal control of aging in complex networks (Sun et al., 2020); this release does not claim the broad idea is new.

## Reproduce

Requirements: Python 3.10+, NumPy, Matplotlib, NetworkX. The environment report under `results/environment.txt` records versions used in this run.

```bash
python verify_model.py
```

The verifier regenerates the model sweep, CSV outputs and eight figures and runs checks for the zero-coupling limit, 24-order enumeration, equal-dose matching, invalid inputs and RK4 step halving. This verifies the implementation of the synthetic model only.

## Artifacts

- [PDF manuscript](paper.pdf)
- [LaTeX source](paper.tex)
- [Verification and figure generator](verify_model.py)
- [Bibliography](paper.bib)
- [Candidate comparison](results/candidate_topics.csv)
- [Literature audit](results/literature_audit.md)
- [Claim/evidence matrix](results/claim_evidence_matrix.csv)
- [Full order sweep](results/schedule_sweep.csv)
- [Coupling summary](results/coupling_summary.csv)
- [Repair budget sensitivity](results/repair_budget_sensitivity.csv)
- [Verification report](results/verification_report.json)
- [Research workflow log](results/research_log.md)
- [Figure assets](figures/)

## References

See `paper.bib`; article identifiers are DOI-linked and the related-work statement is intentionally conservative. A GitHub archive is not peer review or a DOI deposit.
'''
    (root/"README.md").write_text(readme,encoding="utf-8")
    (root/"CITATION.cff").write_text(f'''cff-version: 1.2.0
message: "If you use this model or manuscript, please cite the archived version."
title: "{TITLE}"
authors:
  - family-names: Debnath
    given-names: Devanik
    orcid: "{ORCID}"
version: 1.0.0
date-released: "{TODAY}"
url: "{REPO}"
repository-code: "{REPO}"
abstract: "A reproducible synthetic model audit of repair-order effects under fixed offered maintenance throughput; no empirical lifespan extension is claimed."
keywords:
  - aging
  - repair scheduling
  - mathematical biology
  - first-passage time
  - reproducibility
license: CC-BY-4.0
''',encoding="utf-8")
    (root/".zenodo.json").write_text(json.dumps({"title":TITLE,"upload_type":"publication","publication_type":"preprint","description":"A reproducible synthetic model audit of repair-order effects under fixed offered repair throughput. No biological or organismal lifespan extension is demonstrated.","creators":[{"name":"Devanik Debnath","orcid":"0009-0001-8310-6921"}],"keywords":["aging","repair scheduling","mathematical biology","first-passage time","reproducibility"],"license":"CC-BY-4.0","related_identifiers":[{"identifier":REPO,"relation":"isSupplementTo","resource_type":"publication-preprint"}]},indent=2)+"\n",encoding="utf-8")
    (root/"schema_article.jsonld").write_text(json.dumps({"@context":"https://schema.org","@type":"ScholarlyArticle","headline":TITLE,"author":{"@type":"Person","name":"Devanik Debnath","sameAs":["https://github.com/Devanik21",ORCID]},"datePublished":TODAY,"description":"A reproducible synthetic model audit of repair-order effects under fixed offered maintenance throughput; no empirical lifespan extension is claimed.","keywords":["aging","repair scheduling","mathematical biology","first-passage time"],"url":REPO,"isAccessibleForFree":True,"creativeWorkStatus":"Preprint draft archived in GitHub"},indent=2)+"\n",encoding="utf-8")

    deploy=r'''#!/usr/bin/env python3
"""Non-destructive validation helper for this research release."""
from pathlib import Path
import argparse, json, subprocess, sys, tempfile, shutil, re
EXPECTED={"paper.tex","paper.pdf","verify_model.py","README.md","CITATION.cff","paper.bib",".zenodo.json","schema_article.jsonld","deploy.py"}
class ReleaseError(RuntimeError): pass

def validate(root: Path):
    root=root.resolve()
    files={p.name for p in root.iterdir() if p.is_file()}
    if files != EXPECTED: raise ReleaseError(f"root manifest mismatch: missing={sorted(EXPECTED-files)} extra={sorted(files-EXPECTED)}")
    for d in (root/"figures",root/"results"):
        if not d.is_dir(): raise ReleaseError(f"missing required directory {d.name}")
    pdfs=list((root/"figures").glob("figure*.pdf")); pngs=list((root/"figures").glob("figure*.png"))
    if len(pdfs)<8 or len(pngs)<8: raise ReleaseError(f"expected at least eight PDFs and PNGs; got {len(pdfs)} PDFs, {len(pngs)} PNGs")
    if not (root/"paper.pdf").is_file() or (root/"paper.pdf").stat().st_size<5000: raise ReleaseError("paper.pdf is missing or too small")
    if (root/"paper.pdf").read_bytes()[:5] != b"%PDF-": raise ReleaseError("invalid PDF signature")
    if not (root/"results"/"verification_report.json").exists(): raise ReleaseError("missing verification report")
    report=json.loads((root/"results"/"verification_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("checks_failed") != 0: raise ReleaseError("model verification did not pass")
    for name in (".zenodo.json","schema_article.jsonld"):
        json.loads((root/name).read_text(encoding="utf-8"))
    import yaml
    cff=yaml.safe_load((root/"CITATION.cff").read_text(encoding="utf-8"))
    if cff.get("cff-version")!="1.2.0" or not cff.get("authors"): raise ReleaseError("CITATION.cff invalid")
    tex=(root/"paper.tex").read_text(encoding="utf-8")
    for token in ("\\begin{table}","\\begin{figure}","Co-equal research-process credit is given to Devanik Debnath and Xylia"):
        if token not in tex: raise ReleaseError(f"paper.tex missing required item: {token}")
    bib=(root/"paper.bib").read_text(encoding="utf-8")
    keys=set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",bib))
    cited=set()
    for block in re.findall(r"\\cite\w*\{([^}]+)\}",tex): cited.update(x.strip() for x in block.split(","))
    if cited and not cited.issubset(keys): raise ReleaseError(f"missing bibliography keys: {sorted(cited-keys)}")
    for py in (root/"verify_model.py",root/"deploy.py"):
        r=subprocess.run([sys.executable,"-m","py_compile",str(py)],capture_output=True,text=True)
        if r.returncode: raise ReleaseError(r.stderr)
    return {"root_files":sorted(files),"figure_pdf_count":len(pdfs),"figure_png_count":len(pngs),"verification_status":report["status"]}

def self_test():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); (p/"README.md").write_text("test")
        try: validate(p)
        except Exception: pass
        else: raise ReleaseError("negative fixture was incorrectly accepted")
    return True

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent);ap.add_argument("--self-test",action="store_true");ap.add_argument("--validate-only",action="store_true")
    a=ap.parse_args()
    try:
        if a.self_test:
            self_test(); print("PASS: malformed fixture correctly rejected")
        if a.validate_only or not a.self_test: print(json.dumps(validate(a.root),indent=2))
    except Exception as e: print(f"BLOCKED: {e}",file=sys.stderr);sys.exit(2)
'''
    (root/"deploy.py").write_text(deploy,encoding="utf-8");(root/"deploy.py").chmod(0o755)

    # Build evidence manifest after content generation; temporary files and bundle will be excluded.
    (res/"release_check_matrix.json").write_text(json.dumps({"release_status":"BLOCKED","generated_at_utc":datetime.now(timezone.utc).isoformat(),"repository":"Devanik21/The-Invention-Archive","target_branch":"main","checks":[{"name":"Model verification","status":"PASS" if report["status"]=="PASS" else "FAIL","evidence":"results/verification_report.json"},{"name":"Python syntax / fixture tests","status":"NOT_RUN","evidence":"pending deploy self-tests"},{"name":"Metadata parsing","status":"NOT_RUN","evidence":"pending deploy validation"},{"name":"LaTeX build and PDF page count","status":"NOT_RUN","evidence":"pending build"},{"name":"Full-page PDF visual review","status":"NOT_RUN","evidence":"pending render and inspection"},{"name":"Remote files and PDF/ZIP read-back","status":"NOT_RUN","evidence":"pending GitHub Actions publication"},{"name":"GitHub Actions terminal checks exact SHA","status":"NOT_RUN","evidence":"pending remote release"}],"blockers":["Local PDF build/review has not closed","Remote publication has not been independently verified"]},indent=2)+"\n",encoding="utf-8")
    # create a first SHA manifest excluding itself and later review/ZIP files; regenerated at finalization
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"sha256_manifest.json","complete_release_bundle.zip"} and "__pycache__" not in p.parts}
    (res/"sha256_manifest.json").write_text(json.dumps({"algorithm":"sha256","files":manifest},indent=2)+"\n",encoding="utf-8")


def main():
    # If invoked under the name verify_model.py, only generate the model sweep / figs / outputs.
    me=Path(__file__).name
    if me=="verify_model.py":
        root=Path(__file__).resolve().parent
        _,_,rep=generate_outputs(root)
        print((root/"results"/"verification_stdout.txt").read_text())
    else:
        out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else RELEASE_DIR
        build_release(out)
        print(f"Built release source tree at {out}")

if __name__=="__main__": main()
