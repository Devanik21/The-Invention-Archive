from __future__ import annotations
import csv, itertools, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np

ROOT=Path(__file__).resolve().parent
FIG=ROOT/"figures"; RES=ROOT/"results"
SEED=20261025
P={"kappa":0.25,"alpha":0.75,"rho":0.02,"beta":0.40,"sigma_clock":0.18,
   "sigma_function":0.22,"sigma_state":0.03,"baseline_a_sd":0.30,"baseline_m_sd":0.35,
   "marker_dose_key":0.40,"clock_only_threshold":0.20,"n_key":200,"replicates_key":400}
C={"ink":"#17324D","teal":"#0F766E","gold":"#B98528","coral":"#B94A48","blue":"#4C78A8","gray":"#667085","light":"#EEF4F6"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
 "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,"axes.spines.right":False,
 "savefig.bbox":"tight","savefig.pad_inches":0.08})

def write_csv(path, rows):
    if not rows: path.write_text("",encoding="utf-8"); return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def sim(n,u,q,rng,rho=None):
    rho=P["rho"] if rho is None else rho
    a=rng.normal(1,P["baseline_a_sd"],n); m=rng.normal(0,P["baseline_m_sd"],n)
    g=rng.binomial(1,0.5,n)
    c0=a+P["alpha"]*m+P["beta"]*g+rng.normal(0,P["sigma_clock"],n)
    f0=a+rng.normal(0,P["sigma_function"],n)
    a1=a-P["kappa"]*u+rng.normal(0,P["sigma_state"],n)
    m1=m-q+rng.normal(0,P["sigma_state"],n)
    c1=a1+P["alpha"]*m1+P["beta"]*g+rng.normal(0,P["sigma_clock"],n)
    f1=a1+rho*q+rng.normal(0,P["sigma_function"],n)
    return {"dc":c0-c1,"df":f0-f1,"g":g,"c0":c0,"f0":f0}

def reps(n,u,q,nrep,rng,rho=None):
    out=np.empty((nrep,2))
    for i in range(nrep):
        d=sim(n,u,q,rng,rho=rho); out[i]=[d["dc"].mean(),d["df"].mean()]
    return out

def rows_for(arms):
    rows=[]
    for u,q in arms:
        rows.append([u,q,0,1,0])
        rows.append([u,0,-q,0,1])
    return np.asarray(rows,float)

def info(arms):
    x=rows_for(arms)
    w=np.tile([1/P["sigma_clock"]**2,1/P["sigma_function"]**2],len(arms))
    return x.T@np.diag(w)@x

def logdet(a):
    sign,val=np.linalg.slogdet(a+1e-6*np.eye(a.shape[0]))
    return float(val) if sign>0 else float("-inf")

def cost(arms): return float(sum(u+q for u,q in arms))
def risk(arms): return float(sum(.2*u+.6*q for u,q in arms))

def choose_design():
    candidates=[(0.0,0.0)]
    for u in (0.25,0.5,0.75,1.0):
        for q in (0.0,0.25,0.5,0.75,1.0):
            if u+q<=1.00001 and .2*u+.6*q<=.45001:
                pair=(round(u,2),round(q,2))
                if pair not in candidates:candidates.append(pair)
    budget_c,budget_r,k=2.4,1.10,4
    selected=[(0.0,0.0)]
    while len(selected)<k:
        options=[]
        for arm in candidates:
            if arm in selected: continue
            trial=selected+[arm]
            if cost(trial)>budget_c or risk(trial)>budget_r: continue
            score=logdet(info(trial))-.025*cost(trial)-.08*risk(trial)
            options.append((score,arm,trial))
        if not options: break
        options.sort(key=lambda z:z[0],reverse=True); selected=options[0][2]
    if np.linalg.matrix_rank(rows_for(selected),tol=1e-9)<5:
        best=None
        for combo in itertools.combinations(candidates,k):
            if (0.0,0.0) not in combo or cost(combo)>budget_c or risk(combo)>budget_r: continue
            if np.linalg.matrix_rank(rows_for(combo),tol=1e-9)<5: continue
            score=logdet(info(combo))-.025*cost(combo)-.08*risk(combo)
            if best is None or score>best[0]: best=(score,combo)
        if best: selected=list(best[1])
    return selected,candidates,budget_c,budget_r

def save(fig,stem):
    fig.savefig(FIG/f"{stem}.pdf",format="pdf"); fig.savefig(FIG/f"{stem}.svg",format="svg"); plt.close(fig)

def flow(ax,boxes,arrows,title):
    ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off"); ax.set_title(title,color=C["ink"],weight="bold",pad=8)
    centers={}
    for key,(x,y,w,h,label,col) in boxes.items():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12,rounding_size=.12",
                     linewidth=1.1,edgecolor=col,facecolor="white"))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=8,color=C["ink"])
        centers[key]=(x+w/2,y+h/2)
    for s,d,label,col in arrows:
        x1,y1=centers[s]; x2,y2=centers[d]
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,
                    linewidth=1.2,color=col,shrinkA=26,shrinkB=26))
        if label: ax.text((x1+x2)/2,(y1+y2)/2+.14,label,ha="center",fontsize=7,color=col,
                          bbox=dict(facecolor="white",edgecolor="none",pad=1))

def figures(rng,marker,repair,mixed):
    inv=[]
    def add(stem,kind,purpose):
        inv.append({"figure_id":f"Figure {int(stem[3:5])}","kind":kind,"filename":stem+".pdf",
                    "purpose":purpose,"source":"verify_model.py and archived results CSV"})
    fig,ax=plt.subplots(figsize=(8.2,4.3))
    boxes={"in":(.3,2.2,1.5,1.1,"Randomized inputs\nu, q",C["teal"]),
           "state":(2.3,2.2,1.7,1.1,"Latent states\nfunction a\nmarker m",C["ink"]),
           "clock":(4.8,3.6,1.6,1.0,"Clock score C",C["gold"]),
           "func":(4.8,2.0,1.6,1.0,"Functional F",C["teal"]),
           "id":(4.8,.4,1.6,1.0,"Identity / safety I",C["coral"]),
           "design":(7.4,2.2,2.1,1.1,"Information +\nrisk-constrained\ndesign",C["ink"])}
    flow(ax,boxes,[("in","state","intervention",C["teal"]),("state","clock","shared state",C["gold"]),
         ("state","func","functional path",C["teal"]),("in","id","risk path",C["coral"]),
         ("clock","design","contrast",C["gold"]),("func","design","independent check",C["teal"]),
         ("id","design","constraint",C["coral"])],"From intervention to identifiable evidence")
    save(fig,"fig01_framework"); add("fig01_framework","flow","Orientation flow from inputs to latent states and independent readouts")

    g=nx.DiGraph()
    g.add_edges_from([("Repair input u","Functional state a"),("Marker input q","Marker state m"),
      ("Functional state a","Clock C"),("Marker state m","Clock C"),("Genetic set point g","Clock C"),
      ("Functional state a","Function F"),("Marker input q","Function F"),("Repair input u","Identity I"),
      ("Marker input q","Identity I"),("Identity state i","Identity I")])
    pos={"Repair input u":(0,2),"Marker input q":(0,.7),"Genetic set point g":(0,-.8),
         "Functional state a":(2,2),"Marker state m":(2,.7),"Identity state i":(2,-.8),
         "Clock C":(4,1.8),"Function F":(4,.3),"Identity I":(4,-1.1)}
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    nodecols=[C["gold"] if "Clock" in n else C["teal"] if "Function" in n else C["coral"] if "Identity" in n
              else "#DCEBE7" if "input" in n else "#E7EDF3" for n in g.nodes]
    nx.draw_networkx_nodes(g,pos,node_color=nodecols,node_size=2050,edgecolors=C["ink"],linewidths=.8,ax=ax)
    nx.draw_networkx_labels(g,pos,font_size=8,font_color=C["ink"],ax=ax)
    nx.draw_networkx_edges(g,pos,edge_color=C["gray"],arrows=True,arrowsize=15,width=1.1,
                           connectionstyle="arc3,rad=.04",min_source_margin=18,min_target_margin=18,ax=ax)
    ax.set_title("Causal graph of the stylized observation model",color=C["ink"],weight="bold"); ax.axis("off")
    save(fig,"fig02_causal_dag"); add("fig02_causal_dag","graph","Causal graph separates direct marker, function, genotype, and identity paths")

    h=nx.Graph(); latent=["Functional state","Marker state","Inherited set point","Identity state"]; obs=["Clock C","Function F","Identity I"]
    h.add_nodes_from(latent,bipartite=0); h.add_nodes_from(obs,bipartite=1)
    h.add_edges_from([("Functional state","Clock C"),("Marker state","Clock C"),("Inherited set point","Clock C"),
                      ("Functional state","Function F"),("Identity state","Identity I")])
    pos2={"Functional state":(0,2.1),"Marker state":(0,1),"Inherited set point":(0,-.1),"Identity state":(0,-1.2),
          "Clock C":(3,1.6),"Function F":(3,.4),"Identity I":(3,-.8)}
    fig,ax=plt.subplots(figsize=(7.7,4.2))
    nx.draw_networkx_nodes(h,pos2,nodelist=latent,node_color="#E7EDF3",node_size=2200,edgecolors=C["ink"],ax=ax)
    nx.draw_networkx_nodes(h,pos2,nodelist=obs,node_color=[C["gold"],C["teal"],C["coral"]],node_size=1900,edgecolors=C["ink"],ax=ax)
    nx.draw_networkx_labels(h,pos2,font_size=8,font_color=C["ink"],ax=ax); nx.draw_networkx_edges(h,pos2,edge_color=C["gray"],width=1.4,ax=ax)
    ax.set_title("Latent-to-observation measurement network",color=C["ink"],weight="bold"); ax.axis("off")
    save(fig,"fig03_measurement_network"); add("fig03_measurement_network","graph","Bipartite graph maps latent coordinates to observation channels")

    repeat=[(.5,.5)]*4; varied=[(0,0),(1,0),(0,1),(.5,.5)]
    fig,ax=plt.subplots(figsize=(7.8,4.2))
    for name,arms in [("Repeated mixture",repeat),("Varied + two channels",varied)]:
        x=rows_for(arms); s=np.linalg.svd(x,compute_uv=False); s=np.pad(s,(0,max(0,5-len(s))))[:5]
        ax.plot(np.arange(1,6),s,"o-",label=f"{name} (rank {np.linalg.matrix_rank(x)})",lw=1.8,ms=5)
    ax.set_yscale("symlog",linthresh=1e-5); ax.set_xlabel("Singular-value index"); ax.set_ylabel("Singular value")
    ax.set_title("Design-matrix spectrum exposes weakly identified directions",color=C["ink"],weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.2); save(fig,"fig04_identifiability_spectrum")
    add("fig04_identifiability_spectrum","plot","Singular values diagnose structural and practical identifiability")

    labels=["Marker only","Repair only","Mixed"]; groups=[marker,repair,mixed]
    mc=[x[:,0].mean() for x in groups]; mf=[x[:,1].mean() for x in groups]
    ec=[x[:,0].std(ddof=1)/np.sqrt(len(x)) for x in groups]; ef=[x[:,1].std(ddof=1)/np.sqrt(len(x)) for x in groups]
    fig,ax=plt.subplots(figsize=(7.8,4.2)); x=np.arange(3); w=.34
    ax.bar(x-w/2,mc,w,yerr=ec,capsize=3,label="Clock improvement",color=C["gold"])
    ax.bar(x+w/2,mf,w,yerr=ef,capsize=3,label="Functional improvement",color=C["teal"])
    ax.axhline(0,color=C["ink"],lw=.7); ax.set_xticks(x,labels); ax.set_ylabel("Mean paired improvement (model units)")
    ax.set_title("Clock and functional outcomes can disagree",color=C["ink"],weight="bold")
    ax.legend(frameon=False); ax.grid(axis="y",alpha=.2); save(fig,"fig05_clock_vs_function")
    add("fig05_clock_vs_function","plot","Compares synthetic clock and functional changes across intervention classes")

    qvals=np.linspace(0,.8,9); bias=[]
    for qv in qvals:
        r=reps(200,0,float(qv),250,rng); bias.append((qv,r[:,0].mean(),r[:,0].std(ddof=1),P["alpha"]*qv))
    fig,ax=plt.subplots(figsize=(7.8,4.2)); qx=np.array([r[0] for r in bias])
    ax.plot(qx,[r[3] for r in bias],color=C["ink"],lw=2,label="Analytical expectation")
    ax.errorbar(qx,[r[1] for r in bias],yerr=[r[2]/np.sqrt(250) for r in bias],fmt="o",capsize=3,color=C["gold"],label="Monte Carlo mean ± SE")
    ax.set_xlabel("Marker-targeting dose q"); ax.set_ylabel("Clock-only apparent improvement")
    ax.set_title("Direct marker displacement creates attribution bias",color=C["ink"],weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.2); save(fig,"fig06_marker_bias")
    add("fig06_marker_bias","plot","Monte Carlo clock-only bias versus direct marker-targeting dose")

    active,candidates,bc,br=choose_design()
    design_sets={"Repeated mixture":repeat,"Repair axis":[(0,0),(.5,0),(.75,0),(1,0)],
                 "Factorial":factorial_arms(),"Active design":active}
    design_data=[]
    for name,arms in design_sets.items():
        x=rows_for(arms); s=np.linalg.svd(x,compute_uv=False); nz=s[s>1e-8]
        design_data.append({"design":name,"arms":json.dumps(arms),"logdet_regularized":logdet(info(arms)),
          "rank":int(np.linalg.matrix_rank(x,tol=1e-9)),
          "condition_number_nonzero":float(nz.max()/nz.min()) if len(nz) else float("inf"),
          "cost":cost(arms),"risk":risk(arms),"within_budget":cost(arms)<=bc+1e-9 and risk(arms)<=br+1e-9})
    fig,ax=plt.subplots(figsize=(8,4.3)); names=[r["design"] for r in design_data]; vals=[r["logdet_regularized"] for r in design_data]
    bars=ax.bar(np.arange(len(names)),vals,color=[C["gray"] if r["rank"]<5 else C["teal"] for r in design_data])
    ax.set_xticks(np.arange(len(names)),names,rotation=12,ha="right"); ax.set_ylabel("Regularized log-det information score")
    ax.set_title("Information score must be read with unregularized rank",color=C["ink"],weight="bold")
    for bar,r in zip(bars,design_data): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"rank {r['rank']}",ha="center",va="bottom",fontsize=8)
    ax.grid(axis="y",alpha=.2); save(fig,"fig07_information_design")
    add("fig07_information_design","plot","Information scores and unregularized ranks for candidate designs")

    rmse=[]
    for n in [20,40,80,160,320]:
        t=reps(n,1,P["marker_dose_key"],250,rng)
        naive=t[:,0]; corr=t[:,1]+P["rho"]*P["marker_dose_key"]; truth=P["kappa"]
        rmse.append({"n":n,"rmse_clock_only":float(np.sqrt(np.mean((naive-truth)**2)),
                    "rmse_function_adjusted":float(np.sqrt(np.mean((corr-truth)**2)),
                    "bias_clock_only":float(np.mean(naive-truth)),"bias_function_adjusted":float(np.mean(corr-truth))})
    fig,ax=plt.subplots(figsize=(7.8,4.2)); ns=[r["n"] for r in rmse]
    ax.plot(ns,[r["rmse_clock_only"] for r in rmse],"o-",color=C["coral"],label="Clock-only attribution")
    ax.plot(ns,[r["rmse_function_adjusted"] for r in rmse],"o-",color=C["teal"],label="Functional contrast (rho known)")
    ax.set_xscale("log",base=2); ax.set_xlabel("Subjects per synthetic arm"); ax.set_ylabel("RMSE for kappa")
    ax.set_title("More subjects reduce noise, not structural attribution bias",color=C["ink"],weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.2); save(fig,"fig08_rmse_sample_size")
    add("fig08_rmse_sample_size","plot","RMSE scaling separates sampling variance from structural bias")

    coverage=[]
    for rho in [0,.03,.06,.10,.15]:
        covered=0; nr=300
        for _ in range(nr):
            vals=sim(100,0,P["marker_dose_key"],rng,rho=rho)["df"]
            mean=float(vals.mean()); se=float(vals.std(ddof=1)/np.sqrt(len(vals)))
            covered+=int(mean-1.96*se<=0<=mean+1.96*se)
        coverage.append({"rho_direct_function_effect":rho,"coverage_nominal_95":covered/nr})
    fig,ax=plt.subplots(figsize=(7.8,4.2))
    ax.plot([r["rho_direct_function_effect"] for r in coverage],[r["coverage_nominal_95"] for r in coverage],"o-",color=C["coral"],lw=1.8)
    ax.axhline(.95,color=C["ink"],ls="--",lw=1,label="Nominal 0.95"); ax.set_ylim(0,1.02)
    ax.set_xlabel("Direct functional effect rho"); ax.set_ylabel("Empirical interval coverage")
    ax.set_title("Coverage degrades when the endpoint model omits a direct effect",color=C["ink"],weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.2); save(fig,"fig09_coverage_misspecification")
    add("fig09_coverage_misspecification","plot","Coverage stress test for a misspecified functional-endpoint model")

    genotype=[]
    for delta in np.linspace(-.4,.4,9):
        p0=.5; p1=float(np.clip(p0+delta,.01,.99)); actual=p1-p0
        genotype.append({"genotype_prevalence_difference":actual,"expected_clock_difference":.5+P["beta"]*actual,
                         "true_function_difference":.5,"clock_bias":P["beta"]*actual,"paired_additive_setpoint_bias":0.0})
    fig,ax=plt.subplots(figsize=(7.8,4.2)); gx=[r["genotype_prevalence_difference"] for r in genotype]
    ax.plot(gx,[r["expected_clock_difference"] for r in genotype],"o-",color=C["gold"],label="Cross-sectional clock contrast")
    ax.plot(gx,[r["true_function_difference"] for r in genotype],"--",color=C["teal"],label="True functional difference")
    ax.set_xlabel("Difference in genotype prevalence"); ax.set_ylabel("Group contrast (model units)")
    ax.set_title("Between-group set-point imbalance can shift clock contrasts",color=C["ink"],weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.2); save(fig,"fig10_genotype_sensitivity")
    add("fig10_genotype_sensitivity","plot","Sensitivity of cross-sectional clock contrasts to inherited set-point imbalance")

    us=np.linspace(0,1,51); qs=np.linspace(0,1,51); U,Q=np.meshgrid(us,qs)
    clock=P["kappa"]*U+P["alpha"]*Q; function=P["kappa"]*U-P["rho"]*Q; rr=.2*U+.6*Q
    response={"u":us.tolist(),"q":qs.tolist(),"clock_improvement":clock.tolist(),
              "functional_improvement":function.tolist(),"identity_risk_score":rr.tolist()}
    fig,axs=plt.subplots(1,3,figsize=(10,3.5))
    for ax,data,title,cmap in [(axs[0],clock,"Clock response","YlGnBu"),(axs[1],function,"Functional response","GnBu"),(axs[2],rr,"Identity-risk score","OrRd")]:
        im=ax.imshow(data,origin="lower",extent=[0,1,0,1],aspect="auto",cmap=cmap)
        ax.set_xlabel("Repair input u"); ax.set_ylabel("Marker input q"); ax.set_title(title,fontsize=9); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    fig.suptitle("Synthetic response surfaces: objectives are not interchangeable",color=C["ink"],weight="bold")
    save(fig,"fig11_response_surface"); add("fig11_response_surface","plot","Response surfaces expose clock/function/risk trade-offs")

    fig,ax=plt.subplots(figsize=(8.2,4.3))
    boxes={"q":(.2,2.2,1.5,1,"Define causal\nquestion",C["ink"]),
      "d":(2.2,2.2,1.7,1,"Select orthogonal\ninputs + controls",C["teal"]),
      "m":(4.4,2.2,1.7,1,"Measure clock,\nfunction, identity",C["gold"]),
      "t":(6.6,2.2,1.3,1,"Test rank,\nfit, safety",C["coral"]),
      "p":(8.4,2.2,1.4,1,"Revise claim\nor proceed",C["ink"]),
      "f":(6.6,.35,1.3,.9,"Stop / redesign",C["coral"])}
    flow(ax,boxes,[("q","d","estimand",C["ink"]),("d","m","randomize",C["teal"]),("m","t","multi-view",C["gold"]),
                   ("t","p","passes",C["teal"]),("t","f","fails",C["coral"])],"Proposed experimental decision pipeline")
    save(fig,"fig12_validation_pipeline"); add("fig12_validation_pipeline","flow","Proposed validation pipeline with explicit stop/redesign branch")

    write_csv(RES/"figure_inventory.csv",inv)
    write_csv(RES/"rmse_by_sample_size.csv",rmse); write_csv(RES/"coverage_misspecification.csv",coverage)
    write_csv(RES/"genotype_sensitivity.csv",genotype); write_csv(RES/"design_comparison.csv",design_data)
    write_csv(RES/"synthetic_source_data.csv",[
        {"scenario":name,"replicate":i,"n_per_arm":P["n_key"],"u":u,"q":q,
         "clock_improvement":float(v[0]),"function_improvement":float(v[1])}
        for name,arr,u,q in [("marker_only",marker,0,P["marker_dose_key"]),("repair_only",repair,1,0),("mixed",mixed,1,P["marker_dose_key"])]
        for i,v in enumerate(arr)])
    sens=[]
    for r in rmse:
        sens += [{"analysis":"rmse","setting":r["n"],"metric":"rmse_clock_only","value":r["rmse_clock_only"]},
                 {"analysis":"rmse","setting":r["n"],"metric":"rmse_function_adjusted","value":r["rmse_function_adjusted"]}]
    for r in coverage:sens.append({"analysis":"coverage","setting":r["rho_direct_function_effect"],"metric":"coverage_nominal_95","value":r["coverage_nominal_95"]})
    for r in genotype:sens.append({"analysis":"genotype","setting":r["genotype_prevalence_difference"],"metric":"clock_bias","value":r["clock_bias"]})
    write_csv(RES/"sensitivity_results.csv",sens)
    return inv,rmse,coverage,genotype,design_data,response,active,candidates,bc,br

def factorial_arms():
    return [(0,0),(1,0),(0,1),(.5,.5)]

def records(summary,active,candidates,bc,br,inv,rmse,coverage,genotype,design_data):
    topics=[
      ("Rejuvenation Mirage: clock reset versus functional repair",[5,5,4,5,5,4,5,5,5,5],"Selected: precise identifiability result, recent intervention evidence, falsifiable synthetic test."),
      ("Safe TF sequencing for partial reprogramming",[5,4,5,4,5,5,5,4,5,4],"Strong but overlaps the existing cell-fate-control release."),
      ("Genotype-conditioned epigenetic-clock calibration",[4,5,3,5,5,4,4,5,5,5],"Well grounded but narrower and close to an audit-style paper."),
      ("Topological drift in chromatin organization",[4,3,5,3,4,4,5,3,4,4],"Conceptually rich; direct empirical mapping remains unclear."),
      ("Acoustic mechanotransduction for cell fate control",[4,2,5,2,4,3,4,2,3,2],"High spectacle but insufficiently grounded for a strong mechanism claim."),
      ("Mitochondrial heteroplasmy phase control",[4,4,4,4,4,4,5,4,4,4],"Plausible but overlaps an existing archive topic."),
      ("Quantum coherence as a driver of epigenetic aging",[3,1,5,1,3,2,2,1,2,2],"Rejected: no justified quantum mechanism."),
      ("Causal AI target prioritization across aging diseases",[5,3,3,4,4,4,5,4,4,4],"Important but crowded and close to existing multi-omic work."),
      ("AI detection of alien self-repair dynamics",[4,3,5,4,4,5,4,4,5,4],"Interesting but overlaps an existing archive topic."),
      ("Information-bottleneck design of synthetic gene circuits",[4,3,3,5,5,4,4,5,5,5],"Feasible but overlaps a recent archive release.")
    ]
    names=["importance","gap_evidence","boldness","plausibility","math_core","AI_role","human_benefit","validation_feasibility","falsifiability","safety"]
    tr=[]
    for title,sc,why in topics:
        r={"candidate":title}; r.update(dict(zip(names,sc))); r["mean_score"]=round(float(np.mean(sc)),3); r["selection_rationale"]=why; tr.append(r)
    tr.sort(key=lambda x:x["mean_score"],reverse=True); write_csv(RES/"candidate_topic_matrix.csv",tr)
    lit=[
      ("Horvath 2013; DOI 10.1186/gb-2013-14-10-r115","primary research","Methylation patterns predict age across tissues","Prediction does not establish causal response."),
      ("Levine et al. 2018; DOI 10.18632/aging.101414","primary research","Epigenetic biomarker associated with lifespan/healthspan","Association is not direct proof of rejuvenation."),
      ("Lu et al. 2019; DOI 10.18632/aging.101684","primary research","GrimAge predicts lifespan/healthspan-associated outcomes","Manipulating the score is not shown to change lifespan."),
      ("Liesenfelder et al. 2025; DOI 10.1038/s43587-025-00841-1","primary research","Targeted CpG editing can produce bystander changes and perturb clocks","Context-specific evidence; not a universal effect size."),
      ("Sengstack et al. 2026; DOI 10.1073/pnas.2515183123","primary research","TF perturbation screen sought rejuvenation without dedifferentiation","Does not validate this synthetic model."),
      ("Lim 2026; arXiv:2609.17801","preprint","Preliminary analysis of germline-regulated clock CpGs","Preprint; treated as preliminary."),
      ("Li et al. 2025; DOI 10.1038/s41591-025-03856-8","primary research","LLM-based biological age prediction at population scale","Predictive utility does not imply causal identifiability."),
      ("Haghani et al. 2025; DOI 10.1007/s11357-025-01808-1","primary research","Multi-clock ensemble strategy","Correlated clocks may share direct marker sensitivity."),
      ("Chaloner & Verdinelli 1995; DOI 10.1214/ss/1177009939","methods review","Bayesian experimental-design principles","Risk model remains application-specific."),
      ("MacKay 1992; DOI 10.1162/neco.1992.4.4.590","methods paper","Information-based active data selection","Not a biological validation study.")
    ]
    write_csv(RES/"literature_audit.csv",[{"source":a,"type":b,"supports":c,"limitation":d,"verified_on":"2026-10-25"} for a,b,c,d in lit])
    claims=[
      ("C1","Fixed scalar contrast cannot identify kappa and alpha separately at fixed nonzero inputs","P1","Appendix A algebra; executable assertion","Only the stated reduced linear model."),
      ("C2","Full local identification requires rank 3 for kappa, alpha, rho","P1","Rank test in verify_model.py","Full rank does not validate causal structure."),
      ("C3","Marker-only input can lower synthetic clock while function is unchanged or slightly worse","P2","Deterministic synthetic simulation and archived CSV","No empirical biological parameters."),
      ("C4","Finite-set information-gain planner can select a full-rank design","P2","Candidate design matrix and rank output","Not globally optimal for arbitrary nonlinear designs."),
      ("C5","Discordance is common across biological systems","P3","Not tested","No prevalence claim is made."),
      ("C6","Latent coordinates map to distinct biological pathways","P4","Conceptual mapping motivated by literature","Requires system-specific validation."),
      ("C7","Clock-only reset can resemble repair without functional change","P5/P2","Explicit synthetic counterexample","Not a universal claim about clocks.")
    ]
    write_csv(RES/"claim_evidence_matrix.csv",[{"claim_id":a,"claim":b,"epistemic_level":c,"evidence":d,"limitation":e} for a,b,c,d,e in claims])
    (RES/"source_queries.json").write_text(json.dumps({"audit_date":"2026-10-25","queries":[
      "epigenetic clock targeted CpG editing bystander effects Nature Aging 2025",
      "systematic identification single transcription factor perturbations cellular tissue rejuvenation PNAS 2026",
      "GIA Germline-Informed Aging AlphaGenome genetically regulated CpGs arXiv 2609.17801",
      "large language model biological age prediction large-scale populations Nature Medicine 2025",
      "EnsembleAge multi-clock framework epigenetic age assessment 2025",
      "Horvath 2013 DNA methylation age DOI; Levine 2018 epigenetic biomarker DOI",
      "Chaloner Verdinelli Bayesian experimental design review DOI; MacKay active data selection DOI",
      "Devanik Debnath Devanik21 ORCID Zenodo OpenAIRE Google Scholar"
    ],"coverage":"Focused search, not exhaustive."},indent=2)+"\n",encoding="utf-8")
    (RES/"author_profile_audit.md").write_text(
      "# Author-profile audit — 2026-10-25\n\n"
      "## Public records reviewed\n"
      "- GitHub profile Devanik21 identifies Devanik Debnath, NIT Agartala, ECE '26, and links ORCID 0009-0001-8310-6921, LinkedIn, and two Kaggle profiles.\n"
      "- LinkedIn profile linkedin.com/in/devanik describes a final-year ECE undergraduate and lists a Samsung Fellowship and an ISRO Space Hackathon award. These are profile-reported, not independently audited certificates.\n"
      "- The ORCID identifier is linked from the author's GitHub profile; this search pass did not establish a complete public works list attributable to it.\n\n"
      "## Unresolved in this pass\n"
      "- Zenodo: no unambiguous author record was independently verified by the targeted query.\n"
      "- OpenAIRE and Google Scholar: no unambiguous matching author record was confirmed. No citation counts, publication totals, or index status are claimed.\n"
      "- This is a focused snapshot, not an exhaustive identity search. Name-only matches are not treated as proof.\n\n"
      "Only the supplied name, institution, department, ECE '26 designation, and ORCID URL are used in the manuscript. No completed degree or external endorsement is claimed.\n",encoding="utf-8")
    (RES/"public_data_status.md").write_text(
      "# Public-data status\n\nNo public biological dataset was downloaded or analyzed. All numerical outputs are from the synthetic benchmark. A future empirical study must specify data access, preprocessing, inclusion criteria, and endpoints before reporting results.\n",encoding="utf-8")
    (RES/"research_log.md").write_text(
      "# Research log — 2026-10-25\n\n"
      "1. Compared ten candidate directions across aging clocks, partial reprogramming, genetics, mechanobiology, mitochondria, multi-omics, astrobiology, and gene circuits.\n"
      "2. Scored candidates on importance, gap evidence, boldness, plausibility, mathematical core, AI role, human benefit, validation feasibility, falsifiability, and safety. Scores are internal triage judgments, not measured evidence.\n"
      "3. Selected the Rejuvenation Mirage topic because targeted CpG editing and germline-informed clock work motivate a concrete identifiability problem with a falsifiable analytical/synthetic first test.\n"
      "4. Checked primary/publisher records for clocks, targeted methylation editing, TF perturbation, LLM age prediction, and experimental design. The GIA record is explicitly treated as a preprint.\n"
      "5. Derived the scalar observational-equivalence result and rank condition; implemented independent numerical checks and synthetic outputs.\n"
      "6. Generated vector figures from code and archived the data feeding the plots. No wet-lab data or public biological dataset was used.\n"
      "7. Design system: A4, 10 pt NewTX text/math, restrained navy/teal/gold/coral palette, running heads, contents page, numbered equations, booktabs tables, vector figures, and appendices.\n"
      "8. Full visual review is complete only after every rendered final-PDF page is inspected; rendering alone is not a visual pass.\n\n"
      "Queries and source claims are archived in source_queries.json, literature_audit.csv, and claim_evidence_matrix.csv.\n",encoding="utf-8")
    (RES/"publication_design_brief.md").write_text(
      "# Publication design brief\n\n"
      "- Audience: computational biology, aging-biomarker, and experimental-design researchers.\n"
      "- Venue style: restrained journal-style A4 preprint; no submission or acceptance is implied.\n"
      "- Page system: A4, 0.82-inch margins, 10 pt NewTX text/math, 1.06 line spacing, running header and centered page numbers.\n"
      "- Hierarchy: title/subtitle, abstract, scope callout, contents, numbered sections, appendices, references.\n"
      "- Equations: aligned displays with prose definitions and explicit assumptions.\n"
      "- Tables: booktabs rules, concise captions, consistent notation, no vertical rules.\n"
      "- Figures: vector PDF/SVG, common typography, navy/teal/gold/coral palette, captions distinguish synthetic and conceptual material.\n"
      "- Release gate: compile current source, inspect every rendered page, fix critical/major defects, and bind review to the final PDF hash.\n",encoding="utf-8")

def run_checks(marker,repair,mixed,active,bc,br):
    checks=[]
    def ck(name,ok,detail):
        checks.append({"name":name,"status":"PASS" if ok else "FAIL","detail":detail})
        if not ok: raise AssertionError(name+": "+detail)
    y1=.25*1+.45*1; y2=.40*1+.30*1
    ck("observational_equivalence",abs(y1-y2)<1e-12,f"two parameter pairs both give {y1:.6f}")
    ck("rank_failure_detected",np.linalg.matrix_rank(rows_for([(.5,.5)]*4))<5,"repeated mixture is rank deficient")
    ck("full_rank_design",np.linalg.matrix_rank(rows_for([(0,0),(1,0),(0,1),(.5,.5)]))==5,"varied multi-view design has full rank")
    ec=P["alpha"]*P["marker_dose_key"]; ef=-P["rho"]*P["marker_dose_key"]
    mc=float(marker[:,0].mean()); mf=float(marker[:,1].mean())
    ck("clock_expectation",abs(mc-ec)<.035,f"Monte Carlo {mc:.5f}; analytic {ec:.5f}")
    ck("function_expectation",abs(mf-ef)<.025,f"Monte Carlo {mf:.5f}; analytic {ef:.5f}")
    ck("marker_only_counterexample",mc>.15 and abs(mf)<.08,f"clock {mc:.5f}; function {mf:.5f}")
    ck("active_design_cost",cost(active)<=bc+1e-9,f"cost {cost(active):.3f} <= {bc:.3f}")
    ck("active_design_risk",risk(active)<=br+1e-9,f"risk {risk(active):.3f} <= {br:.3f}")
    ck("active_design_identifiability",np.linalg.matrix_rank(rows_for(active))==5,f"rank {np.linalg.matrix_rank(rows_for(active))} of 5")
    ck("nonempty_simulations",len(marker)>0 and len(repair)>0 and len(mixed)>0,"all key scenarios have replicates")
    return checks

def main():
    FIG.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED); n=P["n_key"]; q=P["marker_dose_key"]; nr=P["replicates_key"]
    marker=reps(n,0,q,nr,rng); repair=reps(n,1,0,nr,rng); mixed=reps(n,1,q,nr,rng)
    inv,rmse,coverage,genotype,design_data,response,active,candidates,bc,br=figures(rng,marker,repair,mixed)
    mc=float(marker[:,0].mean()); mf=float(marker[:,1].mean()); fpr=float(np.mean(marker[:,0]>P["clock_only_threshold"]))
    best=max(design_data,key=lambda r:r["logdet_regularized"])
    summary={"schema_version":"1.0","execution_timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,
      "parameters":P,"synthetic_only":True,"key_results":{"marker_only_clock_improvement_mean":mc,
      "marker_only_function_improvement_mean":mf,"clock_only_threshold":P["clock_only_threshold"],
      "clock_only_attribution_rate":fpr,"expected_marker_only_clock_improvement":P["alpha"]*q,
      "expected_marker_only_function_improvement":-P["rho"]*q,"active_design_arms":active,
      "active_design_cost":cost(active),"active_design_risk":risk(active),"best_design_by_logdet":best["design"],
      "best_design_logdet":best["logdet_regularized"],"best_design_rank":best["rank"]},
      "interpretation":"Synthetic model outputs only; not biological measurements or evidence of intervention efficacy."}
    (RES/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (RES/"generated_macros.tex").write_text(
      f"\\newcommand{{\\ClockOnlyMean}}{{{mc:.3f}}}\n\\newcommand{{\\FunctionalOnlyMean}}{{{mf:.3f}}}\n"
      f"\\newcommand{{\\FalsePositiveRate}}{{{fpr:.3f}}}\n\\newcommand{{\\BestDesignLogDet}}{{{best['logdet_regularized']:.3f}}}\n"
      f"\\newcommand{{\\BestDesignRank}}{{{best['rank']}}}\n",encoding="utf-8")
    records(summary,active,candidates,bc,br,inv,rmse,coverage,genotype,design_data)
    env={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,
         "networkx":nx.__version__,"seed":SEED,"execution_command":"python verify_model.py","network_required":False,
         "data_source":"synthetic model generated in this script"}
    (RES/"environment_report.json").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")
    checks=run_checks(marker,repair,mixed,active,bc,br)
    vr={"schema_version":"1.0","status":"PASS","execution_timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "command":"python verify_model.py","seed":SEED,"checks":checks,"failed_checks":[],
        "scope":"Mathematical and synthetic computational checks only; no empirical biological validation."}
    (RES/"verification_report.json").write_text(json.dumps(vr,indent=2)+"\n",encoding="utf-8")
    (RES/"verification_summary.md").write_text(
      "# Verification summary\n\n**Model-check status: PASS** for the current deterministic synthetic implementation.\n\n"
      f"- Seed: {SEED}\n- Marker-only mean clock improvement: {mc:.5f} model units\n"
      f"- Marker-only mean functional improvement: {mf:.5f} model units\n"
      f"- Clock-only threshold-attribution rate: {fpr:.5f} at threshold {P['clock_only_threshold']}\n"
      f"- Active design: {json.dumps(active)}\n- Active design rank: {np.linalg.matrix_rank(rows_for(active))} of 5 local parameters\n\n"
      "All values are synthetic and reproducible from verify_model.py. Passing checks do not establish that any biological intervention rejuvenates cells or extends lifespan.\n",encoding="utf-8")
    (RES/"verify_model_stdout.txt").write_text(
      f"verify_model.py\nseed={SEED}\nmarker_only_clock_mean={mc:.8f}\nmarker_only_function_mean={mf:.8f}\n"
      f"clock_only_attribution_rate={fpr:.8f}\nactive_design={json.dumps(active)}\n"
      f"active_design_rank={np.linalg.matrix_rank(rows_for(active))}\nstatus=PASS\nscope=synthetic model only\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","seed":SEED,"marker_only_clock_mean":mc,"marker_only_function_mean":mf,
      "clock_only_attribution_rate":fpr,"active_design":active,"active_design_rank":int(np.linalg.matrix_rank(rows_for(active))),
      "figures":len(inv),"checks":len(checks),"scope":"synthetic model only"},indent=2))

if __name__=="__main__": main()
