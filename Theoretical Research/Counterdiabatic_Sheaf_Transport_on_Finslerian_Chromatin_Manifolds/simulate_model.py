#!/usr/bin/env python3
"""Synthetic-only demonstration for the proposed identity-constrained control framework.
No biological dataset or experimentally fitted parameter is used.
Requires numpy and matplotlib.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets"
ASSET.mkdir(exist_ok=True)
SEED = 20261014
DT = 0.01
T_END = 12.0
N = int(T_END / DT) + 1
T = np.linspace(0, T_END, N)

# State x = [a, ell, r, phi]; dimensionless placeholders, not measured biomarkers.
# a: abstract age-associated-state coordinate (lower target is called "younger").
# ell: lineage-identity deviation proxy; r: oncogenic-stress proxy; phi: phase decoherence.
BASE = dict(aging_drift=0.018, phase_to_age=0.040, age_relax=0.10, rejuvenation_gain=0.24,
            identity_relax=0.40, rejuvenation_identity=0.55, repair_identity=0.45,
            risk_relax=0.38, rejuvenation_risk=0.45, repair_risk=0.35,
            phase_relax=0.20, input_phase=0.25)

def rhs(x, u, v, p):
    a, ell, risk, phi = x
    da = p['aging_drift'] + p['phase_to_age'] * phi - p['age_relax'] * a - p['rejuvenation_gain'] * u
    dell = -p['identity_relax'] * ell + p['rejuvenation_identity'] * u * u - p['repair_identity'] * v
    drisk = -p['risk_relax'] * risk + p['rejuvenation_risk'] * u * u - p['repair_risk'] * v
    dphi = -p['phase_relax'] * phi + p['input_phase'] * u
    return np.array([da, dell, drisk, dphi])

def run(policy, p=None, noise_seed=None):
    p = dict(BASE if p is None else p)
    rng = np.random.default_rng(noise_seed)
    X = np.zeros((N, 4)); X[0] = [1.0, 0.02, 0.01, 0.18]
    U = np.zeros(N); V = np.zeros(N)
    for k in range(N-1):
        t=T[k]; a,ell,risk,phi=X[k]
        if policy == 'baseline':
            u, v = 0.0, 0.0
        elif policy == 'aggressive':
            u, v = (0.90 if 0.4 <= t <= 4.4 else 0.0), 0.0
        elif policy == 'constrained':
            # Bounded pulses are enabled only while identity/risk proxies are below
            # explicit synthetic thresholds; repair responds smoothly to excursions.
            pulse = ((t % 1.0) < 0.30)
            safe = (ell < 0.075) and (risk < 0.050)
            u = 0.62 if (pulse and safe and t >= 0.4 and t <= 7.5) else 0.0
            v = min(1.0, max(0.0, 6.0*max(0.0,ell-0.025) + 5.0*max(0.0,risk-0.018)))
        else: raise ValueError(policy)
        dx=rhs(X[k],u,v,p)
        if noise_seed is not None:
            # Arbitrary normalized process noise for stress-testing, not calibrated biology.
            dx = dx + rng.normal(0,[0.012,0.010,0.012,0.018])
        X[k+1]=X[k]+DT*dx
        # guard only against numerical negative artifacts in deviation-type states
        X[k+1,1:3]=np.maximum(X[k+1,1:3],0.0)
        U[k]=u; V[k]=v
    U[-1]=U[-2]; V[-1]=V[-2]
    return X,U,V

# Representative deterministic trajectories.
trajectories={}
for policy in ['baseline','aggressive','constrained']:
    trajectories[policy]=run(policy)

# Figure 1: framework pipeline diagram.
fig, ax=plt.subplots(figsize=(12,3.4)); ax.set_axis_off()
boxes=[(0.02,0.28,0.17,0.48,'Multimodal data','RNA / ATAC / Hi-C\nmetabolites / function'),
       (0.22,0.28,0.17,0.48,'State geometry','Fisher metric +\ndirectional drift'),
       (0.42,0.28,0.17,0.48,'Identity sheaf','consistency residuals\n+ lineage markers'),
       (0.62,0.28,0.17,0.48,'pH + control','dissipation, reachability,\nrobust constraints'),
       (0.82,0.28,0.16,0.48,'Decision gate','stop / revise / test\n(no clinical advice)')]
for x,y,w,h,title,sub in boxes:
    rect=plt.Rectangle((x,y),w,h,linewidth=1.4,edgecolor='#263746',facecolor='#eef3f6',transform=ax.transAxes)
    ax.add_patch(rect); ax.text(x+w/2,y+h*0.67,title,ha='center',va='center',fontsize=11,fontweight='bold',transform=ax.transAxes)
    ax.text(x+w/2,y+h*0.34,sub,ha='center',va='center',fontsize=9,transform=ax.transAxes)
for i in range(4):
    x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
    ax.annotate('',xy=(x2-0.006,0.52),xytext=(x1+0.006,0.52),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.6,color='#263746'))
ax.text(0.5,0.08,'Proposed inference-to-control workflow; every biological module remains an empirical hypothesis.',ha='center',fontsize=10,transform=ax.transAxes)
fig.tight_layout(); fig.savefig(ASSET/'framework_pipeline.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 2: synthetic trajectories.
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
styles={'baseline':'--','aggressive':'-','constrained':'-.'}
labels=['Age-associated coordinate a (lower is target)','Identity-deviation proxy ell','Oncogenic-stress proxy r','Phase-decoherence proxy phi']
for ax,j in zip(axs.ravel(),range(4)):
    for policy,(X,U,V) in trajectories.items():
        ax.plot(T,X[:,j],linestyle=styles[policy],linewidth=2,label=policy.capitalize())
    ax.set_ylabel(labels[j]); ax.set_xlabel('Dimensionless model time'); ax.grid(alpha=.25)
    if j==1: ax.axhline(.075,color='gray',lw=1,ls=':',label='illustrative guard')
    if j==2: ax.axhline(.050,color='gray',lw=1,ls=':',label='illustrative guard')
handles,leglabels=axs[0,0].get_legend_handles_labels(); fig.legend(handles,leglabels,loc='upper center',bbox_to_anchor=(0.5,0.99),ncol=3,frameon=False)
fig.suptitle('Synthetic demonstration only: trajectories under selected toy dynamics',y=1.08,fontsize=14)
fig.tight_layout(rect=[0,0,1,.88]); fig.savefig(ASSET/'synthetic_trajectories.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Monte Carlo uncertainty stress-test over synthetic coefficients and noise.
rng=np.random.default_rng(SEED)
mc={p:[] for p in ['baseline','aggressive','constrained']}
for rep in range(300):
    p={k: float(v*rng.uniform(.8,1.2)) for k,v in BASE.items()}
    for policy in mc:
        X,U,V=run(policy,p,noise_seed=SEED+rep*7+len(policy))
        mc[policy].append([X[-1,0],np.max(X[:,1]),np.max(X[:,2]),np.max(X[:,3]),np.trapezoid(U*U,dx=DT),np.trapezoid(V*V,dx=DT)])

summary={}
for policy,rows in mc.items():
    arr=np.asarray(rows)
    summary[policy]={}
    for idx,name in enumerate(['endpoint_age_coordinate','max_identity_deviation','max_risk_proxy','max_phase_decoherence','integrated_rejuvenation_effort','integrated_repair_effort']):
        summary[policy][name]={
            'median':float(np.median(arr[:,idx])),
            'q05':float(np.quantile(arr[:,idx],.05)),
            'q95':float(np.quantile(arr[:,idx],.95))}

# plot Monte Carlo distributions.
fig,axs=plt.subplots(1,3,figsize=(11,3.6))
for ax,idx,title in zip(axs,[0,1,2],['Endpoint age coordinate','Peak identity deviation','Peak risk proxy']):
    ax.boxplot([np.asarray(mc[p])[:,idx] for p in ['baseline','aggressive','constrained']],tick_labels=['Baseline','Aggressive','Constrained'],showfliers=False)
    ax.set_title(title); ax.grid(axis='y',alpha=.25)
fig.suptitle('Monte Carlo parameter/noise stress test (n=300; all parameters synthetic)',y=1.02,fontsize=13)
fig.tight_layout(); fig.savefig(ASSET/'synthetic_monte_carlo.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Write machine-readable outputs and compact verification checks.
for policy,(X,U,V) in trajectories.items():
    assert np.isfinite(X).all() and X.shape==(N,4)
    assert (X[:,1:3]>=0).all()
assert trajectories['aggressive'][0][0,0] == 1.0
assert max(trajectories['aggressive'][0][:,1]) > max(trajectories['baseline'][0][:,1])
assert max(trajectories['aggressive'][0][:,2]) > max(trajectories['baseline'][0][:,2])
assert summary['aggressive']['max_identity_deviation']['median'] > summary['baseline']['max_identity_deviation']['median']
(ASSET/'synthetic_summary.json').write_text(json.dumps({'disclaimer':'Synthetic demonstration only. No biological datasets or experimentally calibrated parameters were used.', 'seed':SEED,'n_replicates':300,'dt':DT,'T_end':T_END,'base_parameters':BASE,'summary':summary},indent=2))
print(json.dumps({'seed':SEED,'n_replicates':300,'base_parameters':BASE,'deterministic_endpoints':{k: { 'a':float(v[0][-1,0]),'identity_max':float(v[0][:,1].max()),'risk_max':float(v[0][:,2].max()),'phi_max':float(v[0][:,3].max()),'rejuvenation_effort':float(np.trapezoid(v[1]**2,dx=DT)),'repair_effort':float(np.trapezoid(v[2]**2,dx=DT))} for k,v in trajectories.items()},'monte_carlo':summary,'all_checks':'PASS'},indent=2))
