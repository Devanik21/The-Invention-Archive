#!/usr/bin/env python3
"""Reduced perturbative plasma-echo benchmark; not a Vlasov-Poisson solver."""
from pathlib import Path
import json, platform
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
FIG=ROOT/"figures"; RES=ROOT/"results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SEED=20261110

def echo_signal(t,k1=1.0,k2=2.0,tau=3.0,nu=0.0,a1=.04,a2=.04):
    if k1<=0 or k2<=k1 or tau<=0 or nu<0: raise ValueError("require k2>k1>0, tau>0, nu>=0")
    k3=k2-k1; te=k2*tau/k3
    amp=a1*a2*(k1/k3)*np.exp(-nu*k1*k1*tau**3/3.0)
    width=1.0/k3
    return amp*np.exp(-.5*((np.asarray(t)-te)/width)**2),te,amp

def phase_proxy(t,m=np.arange(100),k=1.0,nu=0.0):
    center=.5*(k*t)**2; width=max(1.0,.18*center+1)
    p=np.exp(-.5*((m-center)/width)**2)*np.exp(-nu*m*max(t,0))
    return p/(p.sum()+1e-30)

def memory_benchmark(seed=SEED,n=2400,lag=24):
    rng=np.random.default_rng(seed); dt=.05; t=np.arange(n)*dt
    base=np.exp(-.012*t)*np.sin(.72*t+.12*np.sin(.11*t))
    y=base+.38*np.exp(-.5*((t-55)/1.8)**2)+.015*rng.normal(size=n)
    X=np.array([y[i-lag+1:i+1] for i in range(lag,n-1)])
    Y=y[lag+1:]; cut=int(.7*len(Y))
    def fit(A,b,lam=.01):
        A=np.c_[np.ones(len(A)),A]
        return np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@b)
    def pred(A,c): return np.c_[np.ones(len(A)),A]@c
    c0=fit(X[:cut,-1:],Y[:cut]); c1=fit(X[:cut],Y[:cut])
    p0=pred(X[cut:,-1:],c0); p1=pred(X[cut:],c1); truth=Y[cut:]
    rm0=float(np.sqrt(np.mean((p0-truth)**2))); rm1=float(np.sqrt(np.mean((p1-truth)**2)))
    return t,y,truth,p0,p1,{"memoryless_rmse":rm0,"delay_rmse":rm1,"improvement_fraction":1-rm1/rm0,"train_n":cut,"test_n":len(truth),"lag":lag,"seed":seed}

def save(fig,stem):
    fig.tight_layout()
    fig.savefig(FIG/(stem+".png"),dpi=220)
    fig.savefig(FIG/(stem+".pdf"))
    fig.savefig(FIG/(stem+".svg"))
    plt.close(fig)

def make_figures():
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25})
    t=np.linspace(0,10,1200)
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    for nu in [0,.02,.06,.12]:
        s,te,_=echo_signal(t,nu=nu); ax.plot(t,s,label=f"nu={nu:g}")
    ax.axvline(6,color="black",ls="--",label="predicted echo time 6")
    ax.set(xlabel="time (normalized)",ylabel="echo amplitude (model units)",title="Reduced two-pulse echo"); ax.legend()
    save(fig,"fig1_echo_recovery")
    taus=np.linspace(1,5,70); nus=np.linspace(0,.15,70)
    Z=np.array([[np.exp(-n*tau**3/3) for tau in taus] for n in nus])
    fig,ax=plt.subplots(figsize=(7.2,3.8)); im=ax.imshow(Z,origin="lower",aspect="auto",extent=[1,5,0,.15],vmin=0,vmax=1)
    ax.set(xlabel="pulse separation tau",ylabel="reduced collision parameter nu",title="Declared attenuation factor"); fig.colorbar(im,ax=ax)
    save(fig,"fig2_attenuation_map")
    m=np.arange(100); S=np.array([phase_proxy(x,m=m) for x in np.linspace(0,8,100)])
    fig,ax=plt.subplots(figsize=(7.2,3.8)); im=ax.imshow(np.log10(S.T+1e-12),origin="lower",aspect="auto",extent=[0,8,0,99])
    ax.set(xlabel="time",ylabel="Hermite-like index proxy",title="Synthetic phase-mixing diagnostic"); fig.colorbar(im,ax=ax)
    save(fig,"fig3_phase_mixing_proxy")
    t,y,truth,p0,p1,metrics=memory_benchmark()
    fig,ax=plt.subplots(figsize=(7.2,3.8)); sl=slice(0,220); ax.plot(truth[sl],label="target"); ax.plot(p0[sl],label="memoryless"); ax.plot(p1[sl],label="24-lag delay")
    ax.set(xlabel="held-out sample",ylabel="next-step signal",title="Synthetic chronological holdout"); ax.legend()
    save(fig,"fig4_memory_forecast")
    fig,ax=plt.subplots(figsize=(7.2,3.5)); ax.axis("off")
    pts={"resolved moments":(.12,.55),"velocity filaments":(.50,.82),"memory state":(.50,.28),"echo observable":(.87,.55)}
    for a,b in [("resolved moments","velocity filaments"),("velocity filaments","memory state"),("memory state","resolved moments"),("velocity filaments","echo observable"),("resolved moments","echo observable")]:
        ax.annotate("",pts[b],pts[a],arrowprops={"arrowstyle":"->"})
    for label,(x,y) in pts.items(): ax.text(x,y,label,ha="center",va="center",bbox={"boxstyle":"round","fc":"white"})
    ax.set_title("Proposed information-flow graph"); save(fig,"graph1_information_flow")
    fig,ax=plt.subplots(figsize=(7.2,3.5)); ax.axis("off")
    nodes=[(.10,.5,"pulse 1"),(.34,.78,"phase mix"),(.34,.22,"pulse 2"),(.67,.5,"rephase"),(.90,.5,"echo")]
    for i,j in [(0,1),(2,3),(1,3),(3,4)]: ax.annotate("",nodes[j][:2],nodes[i][:2],arrowprops={"arrowstyle":"->"})
    for x,y,label in nodes: ax.text(x,y,label,ha="center",va="center",bbox={"boxstyle":"circle","fc":"white"})
    ax.set_title("Two-pulse echo causal sequence"); save(fig,"graph2_echo_causality")
    for stem,title,labels in [
      ("flow1_pipeline","Research pipeline",["Define reduced model","Derive echo time","Sweep parameters","Check assumptions","Archive evidence"]),
      ("flow2_validation","Validation sequence",["Train pulse families","Hold out protocols","Compare baselines","Check echo event","Reject on failure"])]:
        fig,ax=plt.subplots(figsize=(7.2,4)); ax.axis("off"); ys=np.linspace(.88,.12,len(labels))
        for i,(yy,label) in enumerate(zip(ys,labels)):
            ax.text(.5,yy,label,ha="center",va="center",bbox={"boxstyle":"round","fc":"white"})
            if i<len(labels)-1: ax.annotate("",(.5,ys[i+1]+.05),(.5,yy-.05),arrowprops={"arrowstyle":"->"})
        ax.set_title(title); save(fig,stem)
    np.savetxt(RES/"attenuation_grid.csv",np.c_[nus.repeat(len(taus)),np.tile(taus,len(nus)),Z.ravel()],delimiter=",",header="nu,tau,attenuation",comments="")
    np.savetxt(RES/"memory_predictions.csv",np.c_[truth,p0,p1],delimiter=",",header="target,memoryless_prediction,delay_prediction",comments="")
    return metrics

def main():
    metrics=make_figures(); checks=[]
    _,te,_=echo_signal(np.array([0.])); checks.append(("echo_rephasing_time",abs(te-6)<1e-12,te))
    _,_,a=echo_signal(np.array([6.]),a1=.02,a2=.03); _,_,b=echo_signal(np.array([6.]),a1=.04,a2=.03); checks.append(("bilinear_scaling",abs(b/a-2)<1e-12,b/a))
    amps=[echo_signal(np.array([6.]),nu=n)[2] for n in [0,.02,.04,.08]]; checks.append(("attenuation_monotonicity",all(amps[i]>=amps[i+1] for i in range(3)),amps))
    checks.append(("proxy_normalization",abs(phase_proxy(4).sum()-1)<1e-12,float(phase_proxy(4).sum())))
    try: echo_signal(np.array([1.]),k1=2,k2=1); ok=False
    except ValueError: ok=True
    checks.append(("invalid_parameters_rejected",ok,"k2<=k1"))
    checks.append(("delay_embedding_beats_memoryless",metrics["delay_rmse"]<metrics["memoryless_rmse"],metrics))
    report={"model_scope":"reduced perturbative free-streaming echo model; not self-consistent Vlasov-Poisson","seed":SEED,"checks":[{"name":n,"passed":bool(p),"observed":v} for n,p,v in checks],"memory_metrics":metrics,"environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}}
    (RES/"verification.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps(report,indent=2,default=str))
    if not all(p for _,p,_ in checks): raise SystemExit(1)
if __name__=="__main__": main()
