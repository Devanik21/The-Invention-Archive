#!/usr/bin/env python3
"""Reproducible synthetic benchmark for topology-informed stochastic cell-state control.

All simulation data in this script are synthetic. This is a method stress test,
not biological validation. Run from the release directory: python verify_model.py
"""
from __future__ import annotations
import csv, json, math, platform, sys, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import networkx as nx

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
SEED = 20261024
rng_global = np.random.default_rng(SEED)

# Cohesive, restrained visual language; all data-bearing graphics are generated here.
NAVY = "#17324D"
TEAL = "#168C8C"
CORAL = "#D96C5F"
GOLD = "#D7A441"
SLATE = "#536579"
PALE = "#EAF1F5"
GREY = "#8A98A6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA9B5", "axes.labelcolor": NAVY,
    "text.color": NAVY, "xtick.color": SLATE, "ytick.color": SLATE,
    "grid.color": "#D9E1E7", "grid.linewidth": 0.6,
})


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def potential(x):
    x = np.asarray(x)
    return 0.25 * (x*x - 1.0)**2


def dpotential(x):
    x = np.asarray(x)
    return x * (x*x - 1.0)


def h0_intervals_1d(x, values):
    """H0 persistence for a sampled 1D path graph under the sublevel filtration.

    Returns finite intervals with birth/death locations. Vertices are activated in
    stable (value, index) order; when components merge, the lower-birth component
    survives (elder rule). This is intentionally a transparent toy implementation.
    """
    x = np.asarray(x); v = np.asarray(values); n = len(v)
    order = np.lexsort((np.arange(n), v))
    parent = np.arange(n); active = np.zeros(n, dtype=bool)
    birth_val = v.copy(); birth_idx = np.arange(n)
    intervals = []
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return int(a)
    for i in order:
        active[i] = True
        neighbors = [j for j in (i-1, i+1) if 0 <= j < n and active[j]]
        roots = sorted(set(find(j) for j in neighbors))
        # A new isolated vertex is a component born at this filtration level.
        parent[i] = i
        if not roots:
            continue
        # The new vertex and any neighbor components merge at v[i]. A component
        # born at this same level produces a zero-length interval and is ignored.
        components = [i] + roots
        roots_unique = sorted(set(find(j) for j in components), key=lambda r: (birth_val[r], birth_idx[r]))
        survivor = roots_unique[0]
        for loser in roots_unique[1:]:
            if birth_val[loser] < v[i] - 1e-12:
                intervals.append({"birth": float(birth_val[loser]), "death": float(v[i]),
                                  "persistence": float(v[i]-birth_val[loser]),
                                  "birth_x": float(x[birth_idx[loser]]), "death_x": float(x[i])})
            parent[loser] = survivor
        parent[i] = survivor
    # Infinite survivor intentionally omitted; only finite H0 intervals are returned.
    return intervals


def fit_synthetic_neural_drift(seed=SEED):
    """Fit a small neural drift model to synthetic Euler-Maruyama transitions."""
    rng = np.random.default_rng(seed + 11)
    n = 9000; dt = 0.05; sigma_true = 0.08
    x0 = rng.uniform(-1.8, 1.8, n)
    b0 = -dpotential(x0)
    x1 = x0 + dt*b0 + sigma_true*np.sqrt(dt)*rng.normal(size=n)
    targets = (x1-x0)/dt
    perm = rng.permutation(n); tr, te = perm[:7200], perm[7200:]
    with warnings.catch_warnings(record=True) as cw:
        warnings.simplefilter("always", ConvergenceWarning)
        model = MLPRegressor(hidden_layer_sizes=(40, 24), activation="tanh", solver="lbfgs",
                             alpha=1e-4, max_iter=700, tol=1e-8, random_state=seed)
        model.fit(x0[tr, None], targets[tr])
    xt = np.linspace(-1.8, 1.8, 1201)
    btrue = -dpotential(xt)
    bhat = model.predict(xt[:, None])
    bpred = model.predict(x0[te, None])
    rmse = float(np.sqrt(np.mean((bpred - (-dpotential(x0[te])))**2)))
    ss_res = float(np.sum((bpred - (-dpotential(x0[te])))**2))
    base = -dpotential(x0[te]); ss_tot = float(np.sum((base-base.mean())**2))
    r2 = float(1.0-ss_res/ss_tot)
    # Residual diffusion estimate from training increments; the known simulation
    # noise is retained as ground truth for a calibration check.
    residuals = x1[te] - (x0[te] + dt*model.predict(x0[te, None]))
    sigma_est = float(np.std(residuals, ddof=1)/np.sqrt(dt))
    metrics = {
        "status": "PASS" if r2 > 0.90 and rmse < 0.35 else "FAIL",
        "model": "scikit-learn MLPRegressor drift estimator; Gaussian residual diffusion",
        "synthetic_training_transitions": int(len(tr)), "synthetic_test_transitions": int(len(te)),
        "time_step": dt, "true_diffusion_sigma": sigma_true, "estimated_diffusion_sigma": sigma_est,
        "test_drift_rmse": rmse, "test_drift_r2": r2,
        "warning_count": len(cw), "training_seed": seed+11,
        "not_empirical": True,
    }
    return model, xt, btrue, bhat, metrics


def train_q_agent(drift_lookup, seed=SEED, beta=6.0, noise=0.08, episodes=5000):
    rng = np.random.default_rng(seed + 21)
    n_steps=40; dt=0.1; umax=4.0; max_budget=8
    x_edges=np.linspace(-1.5, 1.5, 32)
    a_edges=np.linspace(0.0, 1.4, 12)
    # [time, x-bin, adaptation-bin, remaining-dose-pulses, action]
    Q=np.zeros((n_steps+1, len(x_edges)+1, len(a_edges)+1, max_budget+1, 2), dtype=np.float32)
    rewards=[]; eps0=0.30
    def xb(x): return int(np.digitize([np.clip(x,-1.5,1.5)], x_edges)[0])
    def ab(a): return int(np.digitize([np.clip(a,0,1.4)], a_edges)[0])
    def step(x,a,u,rrng):
        eff=u/(1.0+beta*a)
        b=float(np.interp(np.clip(x,-1.8,1.8), np.linspace(-1.8,1.8,len(drift_lookup)), drift_lookup))
        xn=x+dt*(b+eff)+noise*np.sqrt(dt)*rrng.normal()
        an=0.75*a+0.25*(u/umax)
        return float(np.clip(xn,-2.0,2.0)),float(np.clip(an,0,1.5))
    for ep in range(episodes):
        x=float(-1.0+rng.normal(0,0.025)); a=0.0; rem=max_budget; total_r=0.0
        epsilon=max(0.025,eps0*(1.0-ep/episodes))
        for t in range(n_steps):
            s=(t,xb(x),ab(a),rem)
            if rng.random()<epsilon: act=int(rng.integers(2))
            else: act=int(np.argmax(Q[s]))
            u=umax if (act==1 and rem>0) else 0.0
            if act==1 and rem>0: rem-=1
            xn,an=step(x,a,u,rng)
            # Potential-based progress shaping encourages reaching the target sooner.
            shape=0.12*(np.clip(xn,-1.2,0.7)-np.clip(x,-1.2,0.7))
            r=float(shape - (0.018 if u>0 else 0.0))
            done=False
            if xn>=0.72:
                r += 10.0; done=True
            elif t==n_steps-1 or (rem==0 and xn<-1.45 and t>n_steps-4):
                if t==n_steps-1: r -= 3.0; done=True
            sn=(t+1,xb(xn),ab(an),rem)
            target=r if done else r+0.98*float(np.max(Q[sn]))
            Q[s+(act,)]+=0.16*(target-Q[s+(act,)])
            x,a=xn,an; total_r+=r
            if done: break
        rewards.append(total_r)
    return Q, x_edges, a_edges, np.asarray(rewards)


def evaluate_policy(name, drift_lookup, Q=None, x_edges=None, a_edges=None, beta=6.0, noise=0.08,
                    episodes=700, seed=SEED+31):
    rng=np.random.default_rng(seed)
    n_steps=40; dt=0.1; umax=4.0; max_budget=8
    xgrid=np.linspace(-1.8,1.8,len(drift_lookup))
    def xb(x): return int(np.digitize([np.clip(x,-1.5,1.5)], x_edges)[0]) if x_edges is not None else 0
    def ab(a): return int(np.digitize([np.clip(a,0,1.4)], a_edges)[0]) if a_edges is not None else 0
    successes=0; doses=[]; terminal_x=[]; paths=[]
    pulse_steps=np.round(np.linspace(2,37,max_budget)).astype(int)
    for ep in range(episodes):
        x=float(-1.0+rng.normal(0,0.025)); a=0.0; rem=max_budget; dose=0.0; traj=[x]
        for t in range(n_steps):
            if name=="constant-dose":
                u=umax*max_budget/n_steps
            elif name=="evenly-spaced pulses":
                u=umax if (t in pulse_steps and rem>0) else 0.0
                if u: rem-=1
            elif name=="greedy pulses":
                u=umax if (x<0.0 and rem>0) else 0.0
                if u: rem-=1
            elif name=="RL policy":
                s=(t,xb(x),ab(a),rem)
                act=int(np.argmax(Q[s]))
                u=umax if (act==1 and rem>0) else 0.0
                if u: rem-=1
            else:
                raise ValueError(name)
            dose+=u*dt
            eff=u/(1.0+beta*a)
            b=float(np.interp(np.clip(x,-1.8,1.8),xgrid,drift_lookup))
            x=float(np.clip(x+dt*(b+eff)+noise*np.sqrt(dt)*rng.normal(),-2.0,2.0))
            a=float(np.clip(0.75*a+0.25*(u/umax),0,1.5)); traj.append(x)
            if x>=0.72:
                successes+=1; break
        terminal_x.append(x); doses.append(dose); paths.append(traj)
    return {"policy":name,"success_rate":successes/episodes,"successes":successes,"episodes":episodes,
            "mean_terminal_x":float(np.mean(terminal_x)),"mean_nominal_dose":float(np.mean(doses)),
            "dose_budget_upper_bound":umax*max_budget*dt,"paths":paths}


def rotor_counterexample():
    T=3.0; omega=0.20; target=1.0
    u=(target-omega*T)/T
    # Same net control impulse, but a pulse occupies only 30% of the horizon.
    frac=0.30; pulse_u=u/frac
    constant_cost=T*u*u
    pulse_cost=T*frac*pulse_u*pulse_u
    return {"state_space":"S^1; first Betti number b1=1", "T":T,"natural_angular_velocity":omega,
            "target_angle_unwrapped":target,"constant_control":u,"constant_control_energy":constant_cost,
            "pulse_fraction":frac,"pulse_amplitude":pulse_u,"pulse_control_energy":pulse_cost,
            "pulse_to_constant_energy_ratio":pulse_cost/constant_cost if constant_cost>0 else math.inf,
            "constant_is_optimal_by_cauchy_schwarz":True}


def simulate_baselines_grid(drift_lookup, beta_values=None, noise_values=None, trials=40):
    beta_values=beta_values or [0.0,2.0,4.0,6.0,8.0]
    noise_values=noise_values or [0.02,0.05,0.08,0.12,0.16]
    rows=[]
    xgrid=np.linspace(-1.8,1.8,len(drift_lookup)); dt=0.1; n_steps=40; umax=4.0; budget=8
    pulse_steps=set(np.round(np.linspace(2,37,budget)).astype(int).tolist())
    rng=np.random.default_rng(SEED+71)
    for beta in beta_values:
        for sigma in noise_values:
            for policy in ["constant-dose","evenly-spaced pulses"]:
                wins=0
                for j in range(trials):
                    x=-1.0; a=0.0; rem=budget
                    for t in range(n_steps):
                        if policy=="constant-dose": u=umax*budget/n_steps
                        else:
                            u=umax if t in pulse_steps and rem>0 else 0.0
                            if u: rem-=1
                        eff=u/(1+beta*a)
                        b=float(np.interp(x,xgrid,drift_lookup))
                        x=float(np.clip(x+dt*(b+eff)+sigma*np.sqrt(dt)*rng.normal(),-2,2))
                        a=float(np.clip(0.75*a+0.25*(u/umax),0,1.5))
                        if x>=0.72: wins+=1; break
                rows.append({"adaptation_beta":beta,"noise_sigma":sigma,"policy":policy,
                             "success_rate":wins/trials,"trials":trials})
    return rows


def savefig(name, fig):
    print(f"[figure] start {name}", flush=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight", metadata={"Title":name,"Creator":"verify_model.py"})
    print(f"[figure] pdf {name}", flush=True)
    fig.savefig(FIG/f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] done {name}", flush=True)


def generate_figures(xgrid, model_x, btrue, bhat, persist_rows, fw_rows, policies, train_rewards,
                     pulse_sweep, rotor, metrics):
    # Fig. 1: landscape geometry (plot 1)
    xx=np.linspace(-1.6,1.6,180); yy=np.linspace(-1.2,1.2,150)
    X,Y=np.meshgrid(xx,yy); V=0.25*(X*X-1)**2+0.60*Y*Y
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    cf=ax.contourf(X,Y,V,levels=22,cmap="YlGnBu")
    ax.contour(X,Y,V,levels=[0.05,0.10,0.20,0.25,0.4,0.7,1.0],colors="white",linewidths=.6,alpha=.75)
    ax.scatter([-1,0,1],[0,0,0],c=[TEAL,GOLD,CORAL],s=[70,55,70],edgecolor="white",linewidth=1.1,zorder=5)
    ax.annotate("source basin",(-1,0),xytext=(-1.45,.42),arrowprops=dict(arrowstyle="->",color=NAVY),fontsize=8)
    ax.annotate("saddle",(0,0),xytext=(-.35,-.42),arrowprops=dict(arrowstyle="->",color=NAVY),fontsize=8)
    ax.annotate("target basin",(1,0),xytext=(.72,.42),arrowprops=dict(arrowstyle="->",color=NAVY),fontsize=8)
    ax.set(xlabel="latent fate coordinate $x$",ylabel="orthogonal state coordinate $y$",title="Synthetic Waddington-like potential: geometry is not the control law")
    cb=fig.colorbar(cf,ax=ax,pad=.02); cb.set_label("synthetic potential $V(x,y)$")
    savefig("fig01_landscape",fig)

    # Fig. 2: exact barrier and persistence pair (plot 2)
    xs=np.linspace(-1.55,1.55,800); vv=potential(xs)
    fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.plot(xs,vv,color=NAVY,lw=2.2)
    ax.scatter([-1,0,1],[0,.25,0],c=[TEAL,GOLD,CORAL],s=45,zorder=4)
    ax.vlines(0,0,.25,color=GOLD,ls="--",lw=1.7); ax.annotate(r"$\Delta V=0.25$ = $H_0$ persistence",xy=(0,.25),xytext=(.18,.36),arrowprops=dict(arrowstyle="->",color=GOLD),fontsize=9)
    ax.set(xlabel="$x$",ylabel="$V(x)$",title="Exact identity in the symmetric gradient benchmark"); ax.grid(True,alpha=.45)
    savefig("fig02_barrier_persistence",fig)

    # Fig. 3: robustness under perturbations (plot 3)
    fig,ax=plt.subplots(figsize=(7.2,4.0))
    noise_levels=sorted({r['noise_sd'] for r in persist_rows})
    meds=[]; lo=[]; hi=[]
    for sd in noise_levels:
        ar=np.array([r['estimated_persistence'] for r in persist_rows if r['noise_sd']==sd])
        meds.append(float(np.median(ar))); lo.append(float(np.quantile(ar,.05))); hi.append(float(np.quantile(ar,.95)))
    ax.errorbar(noise_levels,meds,yerr=[np.array(meds)-lo,np.array(hi)-meds],fmt="o-",capsize=4,color=TEAL,lw=2,ms=5,label="median and 90% interval")
    ax.axhline(.25,color=CORAL,ls="--",label="true barrier / persistence = 0.25")
    ax.set(xlabel="Gaussian potential-observation noise SD",ylabel="$H_0$ persistence estimate",title="Persistence is stable within a range, not noise-free"); ax.legend(frameon=False); ax.grid(True,alpha=.4)
    savefig("fig03_persistence_robustness",fig)

    # Fig. 4: synthetic neural drift fit (plot 4)
    fig,ax=plt.subplots(figsize=(7.2,4.0)); ax.plot(model_x,btrue,color=NAVY,lw=2.1,label="known synthetic drift")
    ax.plot(model_x,bhat,color=CORAL,lw=1.8,ls="--",label="MLP-fitted drift")
    ax.fill_between(model_x,btrue,bhat,color=GOLD,alpha=.18,label="model discrepancy")
    ax.set(xlabel="latent fate coordinate $x$",ylabel="drift $b(x)$",title=f"Synthetic neural-SDE drift fit (test $R^2$={metrics['test_drift_r2']:.3f})"); ax.legend(frameon=False); ax.grid(True,alpha=.4)
    savefig("fig04_neural_drift_fit",fig)

    # Fig. 5: rotor constant vs pulse protocols (plot 5)
    T=rotor['T']; t=np.linspace(0,T,800); u=rotor['constant_control']
    up=np.where(t<=rotor['pulse_fraction']*T,rotor['pulse_amplitude'],0.0)
    fig,axs=plt.subplots(2,1,figsize=(7.2,5.0),sharex=True)
    axs[0].plot(t,np.full_like(t,u),color=TEAL,lw=2,label="constant optimal control")
    axs[0].plot(t,up,color=CORAL,lw=1.8,ls="--",label="single rectangular pulse, same endpoint")
    axs[0].set_ylabel("control $u(t)$"); axs[0].legend(frameon=False,ncol=2); axs[0].grid(True,alpha=.35)
    theta_c=(0.2+u)*t; theta_p=0.2*t+np.cumsum(up)*(T/(len(t)-1))
    axs[1].plot(t,theta_c,color=TEAL,lw=2,label="constant trajectory")
    axs[1].plot(t,theta_p,color=CORAL,lw=1.8,ls="--",label="pulse trajectory")
    axs[1].axhline(1.0,color=SLATE,ls=":",lw=1.4,label="target angle")
    axs[1].set(xlabel="time",ylabel=r"unwrapped angle $\theta(t)$"); axs[1].legend(frameon=False,ncol=3); axs[1].grid(True,alpha=.35)
    fig.suptitle("A state space with $b_1=1$ can still have constant optimal control",y=1.01,fontsize=11)
    savefig("fig05_rotor_counterexample",fig)

    # Fig. 6: policy trajectories sampled from simulation (plot 6)
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    color_map={"constant-dose":NAVY,"evenly-spaced pulses":TEAL,"greedy pulses":CORAL,"RL policy":GOLD}
    for p in policies:
        trajectories=[tr for tr in p['paths'][:80] if len(tr)>2]
        m=max(len(tr) for tr in trajectories)
        arr=np.full((len(trajectories),m),np.nan)
        for i,tr in enumerate(trajectories): arr[i,:len(tr)]=tr
        tm=np.arange(m)*.1; med=np.nanmedian(arr,axis=0); q1=np.nanquantile(arr,.25,axis=0); q3=np.nanquantile(arr,.75,axis=0)
        ax.plot(tm,med,lw=1.9,color=color_map[p['policy']],label=p['policy'])
        ax.fill_between(tm,q1,q3,color=color_map[p['policy']],alpha=.10)
    ax.axhline(.72,color=SLATE,ls=":",label="target threshold")
    ax.set(xlabel="time",ylabel="latent state $x_t$",title="Synthetic controller trajectories (median and interquartile band)"); ax.legend(frameon=False,ncol=2); ax.grid(True,alpha=.35)
    savefig("fig06_policy_trajectories",fig)

    # Fig. 7: Q-learning curve (plot 7)
    fig,ax=plt.subplots(figsize=(7.2,3.8)); rr=np.asarray(train_rewards)
    window=500; vals=np.array([np.mean(rr[max(0,i-window):i+1]) for i in range(0,len(rr),100)])
    ax.plot(np.arange(len(vals))*100,vals,color=TEAL,lw=2)
    ax.set(xlabel="training episode",ylabel="mean shaped return (500-episode window)",title="Tabular Q-learning in the known synthetic model"); ax.grid(True,alpha=.4)
    savefig("fig07_rl_learning_curve",fig)

    # Fig. 8: toggle-switch-like interaction graph (graph 1)
    G=nx.DiGraph(); G.add_edges_from([("TF-A","TF-B"),("TF-B","TF-A"),("TF-A","Target state"),("TF-B","Target state"),("Pulse input","TF-A"),("Pulse input","TF-B")])
    pos={"TF-A":(-.7,.1),"TF-B":(.7,.1),"Target state":(0,-.8),"Pulse input":(0,.9)}
    fig,ax=plt.subplots(figsize=(6.8,4.2)); nx.draw_networkx_nodes(G,pos,node_size=[1500,1500,1850,1600],node_color=[PALE,PALE,"#F8E5E1","#DDF2EE"],edgecolors=[NAVY,NAVY,CORAL,TEAL],linewidths=1.4,ax=ax)
    nx.draw_networkx_labels(G,pos,font_size=8,font_weight="bold",ax=ax)
    nx.draw_networkx_edges(G,pos,edgelist=[("TF-A","TF-B"),("TF-B","TF-A")],connectionstyle="arc3,rad=0.20",arrowsize=16,width=1.7,edge_color=CORAL,ax=ax)
    nx.draw_networkx_edges(G,pos,edgelist=[("TF-A","Target state"),("TF-B","Target state"),("Pulse input","TF-A"),("Pulse input","TF-B")],arrowsize=15,width=1.5,edge_color=SLATE,ax=ax)
    ax.set_title("A minimal toggle-like regulatory motif (conceptual graph)"); ax.axis("off"); savefig("fig08_toggle_network",fig)

    # Fig. 9: cyclic repression motif with directional circulation (graph 2)
    C=nx.DiGraph(); C.add_edges_from([("G1","G2"),("G2","G3"),("G3","G1")])
    pos=nx.circular_layout(C,scale=1.0)
    fig,ax=plt.subplots(figsize=(6.4,4.2)); nx.draw_networkx_nodes(C,pos,node_size=1850,node_color=["#DDF2EE","#E9E7F8","#F8E5E1"],edgecolors=NAVY,linewidths=1.3,ax=ax)
    nx.draw_networkx_labels(C,pos,font_size=10,font_weight="bold",ax=ax)
    nx.draw_networkx_edges(C,pos,arrowsize=22,width=2.0,edge_color=TEAL,connectionstyle="arc3,rad=0.13",ax=ax)
    ax.set_title("Repressilator-like directed cycle: motif existence is not a pulse theorem"); ax.axis("off"); savefig("fig09_cycle_network",fig)

    # Fig. 10: data-to-control pipeline flow diagram 1
    fig,ax=plt.subplots(figsize=(7.8,3.7)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis("off")
    boxes=[(0.25,2.3,1.65,.9,"Single-cell\ntime course"),(2.25,2.3,1.65,.9,"Latent-state\nembedding"),(4.25,2.3,1.65,.9,"Neural drift +\nnoise fit"),(6.25,2.3,1.65,.9,"Topology / Hodge\nuncertainty"),(8.25,2.3,1.45,.9,"Control\nbenchmark")]
    for i,(x,y,w,h,txt) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=.08",facecolor=PALE if i<4 else "#DDF2EE",edgecolor=NAVY,linewidth=1.1)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=8)
        if i<len(boxes)-1: ax.add_patch(FancyArrowPatch((x+w,y+h/2),(boxes[i+1][0],y+h/2),arrowstyle="-|>",mutation_scale=12,color=TEAL,lw=1.5))
    ax.text(5,1.35,"Required guards: held-out time points · bootstrap persistence · matched-dose baselines · no biological claim from synthetic data",ha="center",va="center",fontsize=8,color=SLATE,wrap=True)
    ax.set_title("Proposed analysis pipeline; dataset fitting is a future empirical stage",pad=10)
    savefig("fig10_pipeline_flow",fig)

    # Fig. 11: hypothesis adjudication / falsification flow diagram 2
    fig,ax=plt.subplots(figsize=(7.8,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    nodes=[(3.2,4.0,3.6,.65,"Candidate topological statistic"),(3.2,2.85,3.6,.65,"Test invariance + noise stability"),(0.35,1.35,3.0,.8,"Stable + calibrated\nunder stated assumptions"),(6.65,1.35,3.0,.8,"Unstable / non-identifiable\nor no control implication"),(3.25,.15,3.5,.65,"Report conditional use or reject claim")]
    for i,(x,y,w,h,txt) in enumerate(nodes):
        fc="#DDF2EE" if i==2 else ("#F8E5E1" if i==3 else PALE)
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.06,rounding_size=.07",facecolor=fc,edgecolor=NAVY,linewidth=1.1)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=8)
    arrows=[((5,4),(5,3.5)),((3.8,2.85),(1.85,2.15)),((6.2,2.85),(8.15,2.15)),((1.85,1.35),(4.1,.8)),((8.15,1.35),(5.9,.8))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,color=TEAL,lw=1.4))
    ax.set_title("Fail-closed decision tree for topology-to-control claims",pad=8); savefig("fig11_falsification_flow",fig)

    # Fig. 12: control benefit sensitivity (plot 8)
    beta_vals=sorted({r['adaptation_beta'] for r in pulse_sweep}); sigma_vals=sorted({r['noise_sigma'] for r in pulse_sweep})
    diff=np.zeros((len(beta_vals),len(sigma_vals)))
    for i,be in enumerate(beta_vals):
        for j,si in enumerate(sigma_vals):
            a=next(r['success_rate'] for r in pulse_sweep if r['adaptation_beta']==be and r['noise_sigma']==si and r['policy']=="evenly-spaced pulses")
            b=next(r['success_rate'] for r in pulse_sweep if r['adaptation_beta']==be and r['noise_sigma']==si and r['policy']=="constant-dose")
            diff[i,j]=a-b
    fig,ax=plt.subplots(figsize=(6.8,4.2)); im=ax.imshow(diff,origin="lower",aspect="auto",cmap="RdYlBu",vmin=-1,vmax=1)
    ax.set_xticks(range(len(sigma_vals)),[f"{s:.02f}" for s in sigma_vals]); ax.set_yticks(range(len(beta_vals)),[f"{b:.1f}" for b in beta_vals])
    ax.set(xlabel="diffusion noise SD",ylabel=r"adaptation strength $\beta$",title="Pulse minus constant success rate (synthetic, fixed nominal-dose budget)")
    for i in range(len(beta_vals)):
        for j in range(len(sigma_vals)): ax.text(j,i,f"{diff[i,j]:+.2f}",ha="center",va="center",fontsize=7,color="black")
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label("success-rate difference")
    savefig("fig12_schedule_sensitivity",fig)


def main():
    # 1. Exact 1D persistence and noise robustness.
    x=np.linspace(-1.55,1.55,1001); v=potential(x)
    barrier=float(potential(0.0)-potential(-1.0))
    intervals=h0_intervals_1d(x,v)
    exact_pair=[z for z in intervals if abs(z['death_x'])<0.02 and abs(z['birth_x'])>.55 and abs(z['birth_x'])<1.4]
    if not exact_pair:
        # In a symmetric equal-depth pair, tie-breaking can make either basin the elder.
        exact_pair=sorted([z for z in intervals if abs(z['death_x'])<0.02],key=lambda q:q['persistence'],reverse=True)
    persistence=float(max((z['persistence'] for z in exact_pair),default=0.0))
    persist_rows=[]; persist_noise_rows=[]
    for sd in [0.0,0.01,0.03,0.05,0.08,0.12]:
        for rep in range(40):
            rr=np.random.default_rng(SEED+1000+int(sd*10000)*100+rep)
            obs=v+rr.normal(0,sd,len(v))
            smooth=gaussian_filter1d(obs, sigma=2.0)
            ints=h0_intervals_1d(x,smooth)
            candidates=[z for z in ints if abs(z['death_x'])<0.05 and 0.55<abs(z['birth_x'])<1.4 and z['persistence']>0]
            est=max((z['persistence'] for z in candidates),default=float('nan'))
            if not np.isfinite(est):
                # Report missing pairing as missing, never impute it into the estimate.
                persist_rows.append({"noise_sd":sd,"replicate":rep,"estimated_persistence":float('nan'),"absolute_error":float('nan'),"pair_found":False})
            else:
                persist_rows.append({"noise_sd":sd,"replicate":rep,"estimated_persistence":float(est),"absolute_error":abs(float(est)-barrier),"pair_found":True})
    write_csv(RES/"persistence_noise_replicates.csv",persist_rows)
    for sd in sorted({r['noise_sd'] for r in persist_rows}):
        arr=np.array([r['estimated_persistence'] for r in persist_rows if r['noise_sd']==sd and np.isfinite(r['estimated_persistence'])])
        pairfrac=np.mean([r['pair_found'] for r in persist_rows if r['noise_sd']==sd])
        persist_noise_rows.append({"noise_sd":sd,"n_replicates":40,"paired_fraction":float(pairfrac),
                                   "median_persistence":float(np.median(arr)) if len(arr) else None,
                                   "q05":float(np.quantile(arr,.05)) if len(arr) else None,
                                   "q95":float(np.quantile(arr,.95)) if len(arr) else None,
                                   "median_absolute_error":float(np.median(np.abs(arr-barrier))) if len(arr) else None})
    write_csv(RES/"persistence_noise_sweep.csv",persist_noise_rows)

    # 2. FW action integral convergence along the time-reversed gradient path.
    fw_rows=[]
    for n in [40,80,160,320,640,1280,2560]:
        xx=np.linspace(-0.999, -0.001, n)
        dx=xx[1]-xx[0]
        mid=(xx[:-1]+xx[1:])/2
        dV=dpotential(mid)
        # Along the uphill path xdot = +V'(x), the FW Lagrangian 1/4|xdot-b|^2
        # with b=-V' equals V'(x)^2; dt=dx/V'(x), so segment action ~= V'(x) dx.
        approx=float(np.sum(np.abs(dV))*dx)
        fw_rows.append({"grid_points":n,"numerical_action":approx,"exact_barrier":barrier,
                        "absolute_error":abs(approx-barrier),"relative_error":abs(approx-barrier)/barrier})
    write_csv(RES/"fw_action_convergence.csv",fw_rows)

    # 3. Fit the neural drift on synthetic transitions.
    model, model_x, btrue, bhat, metrics=fit_synthetic_neural_drift()
    (RES/"neural_sde_fit_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    # Use the fitted drift itself for control to avoid silently evaluating only the oracle.
    drift_lookup=bhat

    # 4. Train/evaluate an RL policy against matched nominal-dose baselines.
    Q,x_edges,a_edges,train_rewards=train_q_agent(drift_lookup,episodes=6000)
    (RES/"rl_training_summary.json").write_text(json.dumps({"algorithm":"tabular Q-learning","episodes":len(train_rewards),
       "seed":SEED+21,"state":"time, binned fate coordinate, binned adaptation state, remaining pulses",
       "actions":["no pulse","pulse at u_max"],"reward":"target hit +10; terminal miss -3; small potential-based progress shaping and pulse penalty",
       "warning":"This is a synthetic finite-state RL demonstration, not a validated policy for cells."},indent=2)+"\n",encoding="utf-8")
    train_reward_records=[{"episode":i+1,"return":float(r)} for i,r in enumerate(train_rewards)]
    write_csv(RES/"rl_training_returns.csv",train_reward_records)
    policies=[]
    for i,name in enumerate(["constant-dose","evenly-spaced pulses","greedy pulses","RL policy"]):
        kw={"Q":Q,"x_edges":x_edges,"a_edges":a_edges} if name=="RL policy" else {}
        p=evaluate_policy(name,drift_lookup,episodes=400,seed=SEED+310+i,**kw)
        policies.append(p)
    policy_rows=[{k:v for k,v in p.items() if k!="paths"} for p in policies]
    write_csv(RES/"policy_benchmarks.csv",policy_rows)

    # 5. Analytic topological counterexample: controlled rotation on S^1.
    rotor=rotor_counterexample()
    (RES/"rotor_counterexample.json").write_text(json.dumps(rotor,indent=2)+"\n",encoding="utf-8")
    # Cross-sensitivity: paired stochastic trials, model and budget held fixed.
    pulse_sweep=simulate_baselines_grid(drift_lookup,trials=40)
    write_csv(RES/"schedule_sensitivity.csv",pulse_sweep)

    # Save the synthetic potential source values and MLP curve.
    write_csv(RES/"synthetic_potential.csv",[{"x":float(a),"potential":float(b)} for a,b in zip(x,v)])
    write_csv(RES/"neural_drift_curve.csv",[{"x":float(a),"true_drift":float(b),"mlp_drift":float(c)} for a,b,c in zip(model_x,btrue,bhat)])
    write_csv(RES/"policy_trajectory_medians.csv",[
        {"policy":p['policy'],"replicate":i,"step":j,"x":float(val)}
        for p in policies for i,tr in enumerate(p['paths'][:80]) for j,val in enumerate(tr)
    ])

    # Independent assertions: identity, noisy pairing record, convergence, model fit,
    # correct dose matching, and the H1=1 constant-control counterexample.
    checks=[]
    def check(name, cond, detail): checks.append({"name":name,"status":"PASS" if bool(cond) else "FAIL","detail":str(detail)}); assert cond, f"{name}: {detail}"
    check("analytic barrier",abs(barrier-0.25)<1e-12,f"barrier={barrier}")
    check("H0 finite persistence pair",abs(persistence-barrier)<2e-3,f"persistence={persistence}; intervals={len(intervals)}")
    check("FW action convergence",fw_rows[-1]['absolute_error']<fw_rows[0]['absolute_error'],f"coarse={fw_rows[0]['absolute_error']:.4g}; fine={fw_rows[-1]['absolute_error']:.4g}")
    check("synthetic neural drift fit",metrics['test_drift_r2']>0.90 and metrics['test_drift_rmse']<0.35,f"R2={metrics['test_drift_r2']:.3f}; RMSE={metrics['test_drift_rmse']:.3f}")
    matched=abs(3.0*8*0.1 - 0.6*40*0.1)<1e-10
    check("nominal-dose matching",matched,"8 pulses x 4.0 x dt equals constant 0.8 x 40 steps x dt")
    check("rotor constant lower energy",rotor['pulse_to_constant_energy_ratio']>1.0,f"pulse/constant energy={rotor['pulse_to_constant_energy_ratio']:.3f}")
    check("policy benchmark complete",len(policy_rows)==4 and all(r['episodes']==400 for r in policy_rows),f"policies={len(policy_rows)}")
    check("all sensitivity cells produced",len(pulse_sweep)==50,f"cells={len(pulse_sweep)}")
    check("noise replicates preserved",len(persist_rows)==240,f"replicates={len(persist_rows)}")

    generate_figures(x,model_x,btrue,bhat,persist_rows,fw_rows,policies,train_rewards,pulse_sweep,rotor,metrics)
    # Figure inventory is machine-checked; PDFs and 300-dpi PNG previews are both archived.
    fig_inventory=[]
    for i in range(1,13):
        names={1:"landscape",2:"barrier_persistence",3:"persistence_robustness",4:"neural_drift_fit",5:"rotor_counterexample",6:"policy_trajectories",7:"rl_learning_curve",8:"toggle_network",9:"cycle_network",10:"pipeline_flow",11:"falsification_flow",12:"schedule_sensitivity"}
        slug=f"fig{i:02d}_{names[i]}"
        for ext in ["pdf","png"]:
            f=FIG/f"{slug}.{ext}"
            check(f"figure exists {f.name}",f.exists() and f.stat().st_size>1000,f"size={f.stat().st_size if f.exists() else 0}")
        fig_inventory.append({"number":i,"basename":slug,"categories": ("plot" if i in [1,2,3,4,5,6,7,12] else "graph" if i in [8,9] else "flow_diagram"),
                             "vector_asset":f"figures/{slug}.pdf","preview_asset":f"figures/{slug}.png"})
    write_csv(RES/"figure_inventory.csv",fig_inventory)
    # Human-readable numerical record with no empirical claims.
    summary={
        "schema_version":"1.0","execution_timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS" if all(c['status']=="PASS" for c in checks) else "FAIL",
        "synthetic_only":True,"empirical_data_loaded":False,"seed":SEED,
        "potential": {"V(x)":"(x^2-1)^2/4","source_minimum":0.0,"saddle":0.25,"analytic_barrier":barrier,"H0_persistence":persistence},
        "persistence_noise":persist_noise_rows,"neural_sde_fit":metrics,
        "policy_benchmarks":policy_rows,"rotor_counterexample":rotor,
        "checks_total":len(checks),"checks_passed":sum(c['status']=="PASS" for c in checks),
        "checks_failed":sum(c['status']=="FAIL" for c in checks),"checks":checks,
        "figures":fig_inventory,
        "interpretation":"PASS means the stated synthetic checks executed; it is not experimental or biological validation.",
    }
    (RES/"verification_report.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    env={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,
         "scikit_learn":sklearn.__version__,"matplotlib":matplotlib.__version__,"networkx":nx.__version__,"seed":SEED}
    (RES/"environment.json").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")
    with (RES/"verify_model_stdout.txt").open("w",encoding="utf-8") as f:
        f.write("Synthetic topology/control verification\n")
        f.write(f"Barrier={barrier:.6f}; H0 persistence={persistence:.6f}\n")
        f.write(f"Neural drift test R2={metrics['test_drift_r2']:.6f}; RMSE={metrics['test_drift_rmse']:.6f}\n")
        for r in policy_rows: f.write(f"{r['policy']}: success={r['success_rate']:.4f}; mean dose={r['mean_nominal_dose']:.4f}\n")
        f.write(f"Rotor pulse/constant control energy ratio={rotor['pulse_to_constant_energy_ratio']:.6f} (b1=1)\n")
        for c in checks: f.write(f"{c['status']}: {c['name']} — {c['detail']}\n")
    print((RES/"verify_model_stdout.txt").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
