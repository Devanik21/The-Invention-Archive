#!/usr/bin/env python3
"""Reproducible synthetic demonstration for the abiotic-intelligence framework.

All generated values are synthetic and pedagogical. They are not observations,
priors inferred from nature, or calibrated exoplanet/technosignature distributions.
The code illustrates (i) functional equivalence between natural and constructed
intelligence hypotheses and (ii) how a separately informative provenance channel
can, in principle, improve origin discrimination.
"""
from __future__ import annotations
import csv, json, hashlib, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
SEED = 20261030
N_PER_CLASS = 10000
CLASSES = ["Direct abiotic emergence", "Constructed non-biological system", "Non-intelligent abiotic decoy"]
# Illustrative distributions, chosen to encode the identifiability question.
# Functional capability is intentionally matched for the first two classes.
PARAMS = {
    "Direct abiotic emergence": {"F": (0.82, 0.10), "P": (0.15, 0.16), "R": (0.76, 0.13)},
    "Constructed non-biological system": {"F": (0.82, 0.10), "P": (0.82, 0.16), "R": (0.73, 0.13)},
    "Non-intelligent abiotic decoy": {"F": (0.23, 0.15), "P": (0.12, 0.16), "R": (0.30, 0.16)},
}


def logpdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd * np.sqrt(2.0 * np.pi))


def sample_truncated_normal(rng: np.random.Generator, mu: float, sd: float, n: int) -> np.ndarray:
    """Sample a Gaussian restricted to [0,1] without boundary point-mass clipping."""
    out = rng.normal(mu, sd, n)
    invalid = (out < 0.0) | (out > 1.0)
    while np.any(invalid):
        out[invalid] = rng.normal(mu, sd, int(np.sum(invalid)))
        invalid = (out < 0.0) | (out > 1.0)
    return out


def make_figures(data: dict[str, dict[str, np.ndarray]], metrics: dict) -> None:
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":11,
                         "axes.labelsize":9, "figure.dpi":150, "savefig.dpi":220,
                         "savefig.facecolor":"white", "axes.spines.top":False,
                         "axes.spines.right":False})
    palette = {CLASSES[0]:"#3478A8", CLASSES[1]:"#D28B2D", CLASSES[2]:"#6C7A89"}

    # Figure 1: causal/evolutionary pathways.
    fig, ax = plt.subplots(figsize=(10, 4.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
    def box(x,y,w,h,title,sub,color):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",fc=color,ec="#243447",lw=1.0)
        ax.add_patch(p); ax.text(x+w/2,y+h*0.66,title,ha='center',va='center',weight='bold',fontsize=9,color="#142333")
        ax.text(x+w/2,y+h*0.29,sub,ha='center',va='center',fontsize=7.5,color="#243447",wrap=True)
    def arrow(x1,y1,x2,y2,label=""):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.3,color="#34495e"))
        if label: ax.text((x1+x2)/2,(y1+y2)/2+0.16,label,ha='center',fontsize=7.5,color="#34495e")
    box(.25,2.08,1.7,.95,"Natural processes","chemistry · geology · fields","#E5EDF3")
    box(3.0,3.72,2.05,.9,"Abiotic computation","physical information processing","#DCEBF6")
    box(6.55,3.72,2.65,.9,"Candidate abiotic agency","adaptive control without biological lineage","#D7E9DC")
    arrow(1.95,2.75,3.0,4.05,"Path A")
    arrow(5.05,4.16,6.55,4.16)
    box(3.0,2.02,2.05,.9,"Life","self-maintaining evolution","#F5E7CE")
    box(6.55,2.02,2.65,.9,"Biological intelligence","Earth's familiar route","#F5E7CE")
    arrow(1.95,2.52,3.0,2.47,"Path B"); arrow(5.05,2.47,6.55,2.47)
    box(6.55,.25,2.65,.9,"Constructed AI","non-biological system built by agents","#F4DFD8")
    arrow(7.85,2.02,7.85,1.15,"Path C")
    ax.text(5,4.82,"Three causal pathways to non-biological intelligent systems",ha='center',weight='bold',fontsize=12)
    ax.text(5,.0,"The diagram distinguishes origin history; it does not assert that Path A has occurred.",ha='center',fontsize=8,color="#5b6570")
    fig.tight_layout(); fig.savefig(FIG/"fig1_pathways.png",bbox_inches="tight"); fig.savefig(FIG/"fig1_pathways.pdf",bbox_inches="tight"); plt.close(fig)

    # Figure 2: hierarchy from physical dynamics to cautious functional intelligence.
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0,10); ax.set_ylim(0,4.6); ax.axis('off')
    labels=[("0 · Dynamics","State changes under physical laws", "#EDF0F2"),
            ("1 · Computation","Reliable input–output transformation", "#DCEBF6"),
            ("2 · Adaptive computation","History-dependent improvement on held-out tasks", "#E5E4F3"),
            ("3 · Minimal agency","Intervention-sensitive goal-directed control", "#D7E9DC")]
    xs=[.2,2.7,5.2,7.7]
    for i,(t,s,c) in enumerate(labels):
        p=FancyBboxPatch((xs[i],1.55),2.05,1.5,boxstyle="round,pad=.04,rounding_size=.12",fc=c,ec="#243447",lw=1)
        ax.add_patch(p); ax.text(xs[i]+1.025,2.55,t,ha='center',weight='bold',fontsize=8.3)
        ax.text(xs[i]+1.025,2.05,s,ha='center',va='center',fontsize=7.2,wrap=True)
        if i<3: ax.add_patch(FancyArrowPatch((xs[i]+2.05,2.3),(xs[i+1],2.3),arrowstyle='-|>',mutation_scale=12,color="#34495e"))
    ax.text(5,4.1,"A capability ladder, not a claim that every level implies the next",ha='center',fontsize=11,weight='bold')
    ax.text(5,.75,"Required controls: passive relaxation · driven oscillation · self-healing material · scripted controller · trained software",ha='center',fontsize=8,color="#5b6570")
    ax.text(5,.28,"Consciousness, language, self-replication, and Earth-like chemistry are not assumed as requirements.",ha='center',fontsize=8,color="#5b6570")
    fig.tight_layout(); fig.savefig(FIG/"fig2_capability_ladder.png",bbox_inches='tight'); fig.savefig(FIG/"fig2_capability_ladder.pdf",bbox_inches='tight'); plt.close(fig)

    # Figure 3: synthetic distributions show origin non-identifiability from function alone.
    fig, axes = plt.subplots(1,2,figsize=(10,4.1))
    for name in CLASSES:
        axes[0].hist(data[name]["F"],bins=55,density=True,histtype='step',lw=1.8,label=name,color=palette[name])
    axes[0].set_title("Functional capability channel F"); axes[0].set_xlabel("Illustrative normalized functional score"); axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=7,frameon=False)
    for name in CLASSES:
        axes[1].hist(data[name]["P"],bins=55,density=True,histtype='step',lw=1.8,label=name,color=palette[name])
    axes[1].set_title("Provenance channel P"); axes[1].set_xlabel("Illustrative provenance evidence score")
    axes[1].text(.03,.95,"D and C are identical on F by construction; P separates them only in this toy model.",transform=axes[1].transAxes,va='top',fontsize=7.5,bbox=dict(boxstyle='round',fc='white',ec='#cccccc',alpha=.9))
    fig.suptitle("Synthetic identifiability demonstration — not observational data",fontsize=11,weight='bold')
    fig.tight_layout(); fig.savefig(FIG/"fig3_identifiability.png",bbox_inches='tight'); fig.savefig(FIG/"fig3_identifiability.pdf",bbox_inches='tight'); plt.close(fig)

    # Figure 4: staged inference pipeline.
    fig, ax = plt.subplots(figsize=(10,4.6)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis('off')
    stages=[("Stage I","Is it computing?","input–output tests"),
            ("Stage II","Is it adapting?","history · transfer · control"),
            ("Stage III","What is its origin?","natural · constructed · unresolved"),
            ("Stage IV","Can the claim survive?","decoys · priors · independent replication")]
    xs=[.25,2.7,5.15,7.6]
    for i,(a,b,c) in enumerate(stages):
        p=FancyBboxPatch((xs[i],1.8),2.1,1.6,boxstyle='round,pad=.04,rounding_size=.12',fc=['#E5EDF3','#D7E9DC','#F5E7CE','#E5E4F3'][i],ec='#243447')
        ax.add_patch(p); ax.text(xs[i]+1.05,2.95,a,ha='center',weight='bold',fontsize=8); ax.text(xs[i]+1.05,2.5,b,ha='center',weight='bold',fontsize=8); ax.text(xs[i]+1.05,2.12,c,ha='center',fontsize=7.2)
        if i<3: ax.add_patch(FancyArrowPatch((xs[i]+2.1,2.6),(xs[i+1],2.6),arrowstyle='-|>',mutation_scale=12,color='#34495e'))
    ax.text(5,4.45,"Recommended order of inference",ha='center',weight='bold',fontsize=12)
    ax.text(5,.95,"Do not infer origin from capability alone. Do not infer intelligence from complexity alone.",ha='center',fontsize=9,color='#5b6570')
    ax.text(5,.45,"Each stage reports uncertainty and alternative explanations before proceeding.",ha='center',fontsize=8,color='#5b6570')
    fig.tight_layout(); fig.savefig(FIG/"fig4_inference_pipeline.png",bbox_inches='tight'); fig.savefig(FIG/"fig4_inference_pipeline.pdf",bbox_inches='tight'); plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    rng=np.random.default_rng(SEED)
    data={}
    rows=[]
    for name in CLASSES:
        data[name]={}
        for key in ["F","P","R"]:
            mu,sd=PARAMS[name][key]
            vals=sample_truncated_normal(rng,mu,sd,N_PER_CLASS)
            data[name][key]=vals
        # Save a compact preview; full synthetic arrays are regenerated deterministically.
        for i in range(min(100, N_PER_CLASS)):
            rows.append({"class":name,"functional_score_F":float(data[name]["F"][i]),
                         "provenance_score_P":float(data[name]["P"][i]),
                         "response_transfer_score_R":float(data[name]["R"][i])})

    # Direct-vs-constructed discrimination using F alone and F+P.
    d=data[CLASSES[0]]; c=data[CLASSES[1]]
    # Bayes rule under the stipulated Gaussian generative model, equal priors.
    def predict_binary(a,b,features):
        x0=np.column_stack([data[a][f] for f in features]); x1=np.column_stack([data[b][f] for f in features])
        X=np.vstack([x0,x1]); y=np.r_[np.zeros(len(x0),dtype=int),np.ones(len(x1),dtype=int)]
        scores=[]
        for name in [a,b]:
            s=np.zeros(len(X))
            for j,f in enumerate(features):
                mu,sd=PARAMS[name][f]; s += logpdf(X[:,j],mu,sd)
            scores.append(s)
        pred=(scores[1]>scores[0]).astype(int)
        acc=float(np.mean(pred==y))
        return acc, int(np.sum((pred[:len(x0)]==0))), int(np.sum((pred[:len(x0)]==1))), int(np.sum((pred[len(x0):]==0))), int(np.sum((pred[len(x0):]==1)))
    acc_F,*cmF=predict_binary(CLASSES[0],CLASSES[1],["F"])
    acc_FP,*cmFP=predict_binary(CLASSES[0],CLASSES[1],["F","P"])
    # Capability separation: D+C versus abiotic non-intelligent decoy, F only.
    cap_names=[CLASSES[0],CLASSES[1]]; dec=CLASSES[2]
    # Illustrative threshold; not optimized on a validation dataset.
    pos=np.concatenate([data[n]["F"] for n in cap_names]); neg=data[dec]["F"]
    threshold=0.525
    sens=float(np.mean(pos>=threshold)); spec=float(np.mean(neg<threshold))
    summary={
      "schema_version":"1.0","generated_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,
      "synthetic_only":True,"n_per_class":N_PER_CLASS,
      "csv_note":"CSV contains a 100-row-per-class preview; the full sample is regenerated by this script",
      "parameterization":{k:{f:{"mean":v[f][0],"sd":v[f][1]} for f in v} for k,v in PARAMS.items()},
      "origin_discrimination_direct_vs_constructed":{
         "functional_only_accuracy":round(acc_F,5),"functional_plus_provenance_accuracy":round(acc_FP,5),
         "functional_only_confusion_counts":{"direct_correct":cmF[0],"direct_as_constructed":cmF[1],"constructed_as_direct":cmF[2],"constructed_correct":cmF[3]},
         "functional_plus_provenance_confusion_counts":{"direct_correct":cmFP[0],"direct_as_constructed":cmFP[1],"constructed_as_direct":cmFP[2],"constructed_correct":cmFP[3]},
         "interpretation":"Because F has identical stipulated distributions under the two origin hypotheses, F alone is non-identifying. P is stipulated to differ; the gain is a property of the illustrative assumptions, not evidence that real provenance traces are always observable."
      },
      "capability_vs_nonintelligent_decoy":{"threshold_F":threshold,"sensitivity":round(sens,5),"specificity":round(spec,5)},
      "warning":"No values are empirical. Distribution parameters are illustrative and chosen to demonstrate identifiability, not estimated from physical systems or SETI observations."
    }
    with (RES/"synthetic_origin_demo.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (RES/"synthetic_origin_demo_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    make_figures(data,summary)
    (RES/"model_run.txt").write_text("Synthetic origin-identifiability demonstration\n"+json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","seed":SEED,"synthetic_only":True,"n_per_class":N_PER_CLASS,
      "csv_note":"CSV contains a 100-row-per-class preview; the full sample is regenerated by this script",
                      "functional_only_accuracy":round(acc_F,5),"functional_plus_provenance_accuracy":round(acc_FP,5),
                      "capability_sensitivity":round(sens,5),"capability_specificity":round(spec,5)},indent=2))

if __name__=="__main__": main()
