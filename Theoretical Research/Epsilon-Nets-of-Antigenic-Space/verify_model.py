#!/usr/bin/env python3
"""Deterministic synthetic verification for Epsilon-Nets of Antigenic Space.

Not empirical immunology. Tested with Python 3.13.5, NumPy 2.3.5,
Matplotlib 3.10.8, NetworkX 3.6.1. Seed 20261101.
"""
from __future__ import annotations
import csv, json, math, platform, datetime
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
import matplotlib.pyplot as plt
import networkx as nx

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; RES=ROOT/'results'
SEED=20261101; EPS=.20; K=2

def hill_neutralization(distance, radius, kappa=8.0, hill=1.4, concentration=1.0):
    distance=np.asarray(distance,dtype=float); radius=np.asarray(radius,dtype=float)
    if np.any(distance<0) or np.any(radius<=0) or hill<=0 or concentration<=0:
        raise ValueError('distance must be nonnegative and response parameters positive')
    z=hill*(math.log(concentration)-kappa*(distance-radius))
    return np.where(z>=0,1/(1+np.exp(-np.minimum(z,700))),np.exp(np.maximum(z,-700))/(1+np.exp(np.maximum(z,-700))))

def greedy_multicover(incidence,k):
    incidence=np.asarray(incidence,dtype=bool)
    deficit=np.full(incidence.shape[0],int(k),dtype=int); remaining=np.ones(incidence.shape[1],bool); selected=[]
    while np.any(deficit>0):
        avail=np.flatnonzero(remaining)
        if not len(avail): raise RuntimeError('infeasible finite multi-cover')
        gain=(incidence[:,avail] & (deficit[:,None]>0)).sum(axis=0)
        if not gain.size or gain.max()==0: raise RuntimeError('infeasible finite multi-cover')
        j=int(avail[np.argmax(gain)]); selected.append(j); remaining[j]=False
        deficit[incidence[:,j]]=np.maximum(deficit[incidence[:,j]]-1,0)
    return np.array(selected,dtype=int)

def savefig(name):
    plt.tight_layout(); fixed=datetime(2026,11,1,tzinfo=timezone.utc); plt.savefig(FIG/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':fixed,'ModDate':fixed}); plt.savefig(FIG/f'{name}.svg',bbox_inches='tight',metadata={'Date':'2026-11-01'}); plt.close()

def run():
    FIG.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED); na,nv=4000,72
    A=rng.uniform(0,1,(na,2)); X=rng.uniform(.2,.8,(nv,2)); radii=rng.uniform(.315,.375,nv)
    D=np.linalg.norm(A[:,None,:]-X[None,:,:],axis=2)
    N=hill_neutralization(D,radii[None,:]); C=N>=.5; mass=C.mean(axis=0); heavy=mass>=EPS
    if not heavy.all(): raise AssertionError('synthetic design must keep declared ranges epsilon-heavy')
    H=C[:,heavy].T
    one=greedy_multicover(H,1); two=greedy_multicover(H,K)
    counts=H[:,two].sum(axis=1)
    sizes=[len(greedy_multicover(H,k)) for k in range(1,5)]
    ns=np.array([8,12,16,24,32,48,64,96,128,192]); trials=250; success=[]
    for n in ns:
        ok=0
        for _ in range(trials):
            pick=rng.choice(na,size=int(n),replace=False); ok+=int(np.all(C[pick][:,heavy].sum(axis=0)>=1))
        success.append(ok/trials)
    success=np.array(success)
    rhos=np.linspace(0,.12,7); robust=[]
    offsets=lambda r: np.array([[r,0],[-r,0],[0,r],[0,-r]])
    for rho in rhos:
        good=np.ones((na,nv),dtype=bool)
        for off in offsets(rho):
            dd=np.linalg.norm(A[:,None,:]-(X[None,:,:]+off[None,None,:]),axis=2)
            good &= dd<=radii[None,:]
        robust.append(good.mean(axis=0))
    robust=np.array(robust)
    thresholds=np.array([.5,.6,.7,.8,.9]); tm=np.stack([(N>=q).mean(axis=0) for q in thresholds])
    # output data
    with (RES/'synthetic_antibody_antigen_features.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['antibody_id','x','y']); w.writerows([[f'A{i:04d}',*np.round(a,8)] for i,a in enumerate(A)])
    with (RES/'synthetic_antigen_targets.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['target_id','x','y','radius','range_mass','epsilon_heavy']); w.writerows([[f'T{j:03d}',X[j,0],X[j,1],radii[j],mass[j],int(heavy[j])] for j in range(nv)])
    with (RES/'sampling_success.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['sample_size','trials','all_ranges_hit_fraction']); w.writerows([[int(n),trials,p] for n,p in zip(ns,success)])
    with (RES/'robust_neighbourhood.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['rho','mean_robust_mass','fraction_epsilon_heavy']); w.writerows([[r,robust[i].mean(),(robust[i]>=EPS).mean()] for i,r in enumerate(rhos)])
    with (RES/'threshold_sensitivity.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['threshold','mean_candidate_mass','fraction_epsilon_heavy']); w.writerows([[q,tm[i].mean(),(tm[i]>=EPS).mean()] for i,q in enumerate(thresholds)])
    # 1 Hill curves
    x=np.linspace(0,.8,250); plt.figure(figsize=(6.8,4.2))
    for h in [.8,1.4,2.5]: plt.plot(x,hill_neutralization(x,.35,hill=h),label=f'Hill slope h={h}')
    plt.axhline(.5,ls='--',lw=1,c='0.3'); plt.axvline(.35,ls=':',lw=1,c='0.3'); plt.xlabel('Feature-space distance'); plt.ylabel('Neutralization response'); plt.title('Hill response and threshold geometry'); plt.legend(frameon=False); savefig('fig01_hill_response')
    # 2 incidence heatmap
    idx=np.argsort(mass)[::-1][:24]; plt.figure(figsize=(7,4.2)); plt.imshow(C[:80,idx].T,aspect='auto',interpolation='nearest',cmap='Greys'); plt.xlabel('First 80 candidate antibodies'); plt.ylabel('24 targets ordered by range mass'); plt.title('Synthetic neutralization incidence matrix'); savefig('fig02_incidence_heatmap')
    # 3 theoretical scaling curve
    e=np.linspace(.05,.5,120); plt.figure(figsize=(6.5,4)); plt.plot(e,3/e*np.log(1/e),lw=2); plt.xlabel(r'Coverage threshold $\epsilon$'); plt.ylabel(r'Scaling index $3\epsilon^{-1}\log(1/\epsilon)$'); plt.title('VC epsilon-net size scaling (constant omitted)'); savefig('fig03_epsilon_scaling')
    # 4 sampling success
    plt.figure(figsize=(6.5,4)); plt.plot(ns,success,'o-',lw=2); plt.ylim(-.03,1.03); plt.xlabel('Uniform sample size'); plt.ylabel('Fraction of trials hitting all heavy ranges'); plt.title(f'Finite synthetic sampling diagnostic ({trials} trials/size)'); savefig('fig04_sampling_success')
    # 5 robust neighbourhood
    plt.figure(figsize=(6.5,4)); plt.plot(rhos,robust.mean(axis=1),'o-',label='Mean robust candidate mass'); plt.plot(rhos,(robust>=EPS).mean(axis=1),'s--',label='Fraction targets remaining epsilon-heavy'); plt.axhline(EPS,ls=':',c='0.25',label='epsilon threshold'); plt.xlabel('Neighbourhood radius rho'); plt.ylabel('Mass / target fraction'); plt.title('Robust mass under sampled antigenic perturbations'); plt.legend(frameon=False); savefig('fig05_escape_neighbourhood')
    # 6 k-cover size
    plt.figure(figsize=(6.5,4)); plt.plot(np.arange(1,5),sizes,'o-',lw=2); plt.xlabel('Required distinct hits k'); plt.ylabel('Greedy finite cover size'); plt.title('Finite synthetic multi-cover cost (not optimum)'); savefig('fig06_multicover_cost')
    # 7 bipartite network
    G=nx.Graph(); ai=np.linspace(0,na-1,12,dtype=int); vi=np.linspace(0,nv-1,12,dtype=int)
    for i in ai:G.add_node(f'A{i:04d}',bipartite=0)
    for j in vi:G.add_node(f'T{j:03d}',bipartite=1)
    for i in ai:
        for j in vi:
            if C[i,j]:G.add_edge(f'A{i:04d}',f'T{j:03d}')
    pos={f'A{i:04d}':(0,1-q/11) for q,i in enumerate(ai)}; pos.update({f'T{j:03d}':(1,1-q/11) for q,j in enumerate(vi)})
    labels={f'A{i:04d}':f'A{q+1}' for q,i in enumerate(ai)}; labels.update({f'T{j:03d}':f'T{q+1}' for q,j in enumerate(vi)})
    plt.figure(figsize=(7.4,4.8)); nx.draw_networkx_nodes(G,pos,nodelist=[n for n in G if n.startswith('A')],node_size=260,node_color='#355c7d'); nx.draw_networkx_nodes(G,pos,nodelist=[n for n in G if n.startswith('T')],node_size=260,node_color='#c06c84'); nx.draw_networkx_edges(G,pos,alpha=.32,width=.8,edge_color='#64748b'); nx.draw_networkx_labels(G,pos,labels=labels,font_size=7,font_color='white',font_weight='bold'); plt.xlim(-.22,1.22); plt.ylim(-.1,1.1); plt.axis('off'); plt.title('Bipartite antibody-target coverage graph (synthetic)'); savefig('fig07_bipartite_graph')
    # 8 overlap graph
    ids=np.arange(0,120,3); sub=C[ids]; J=np.zeros((len(ids),len(ids)))
    for i in range(len(ids)):
        for j in range(len(ids)):
            un=np.logical_or(sub[i],sub[j]).sum(); J[i,j]=np.logical_and(sub[i],sub[j]).sum()/un if un else 0
    G2=nx.Graph(); G2.add_nodes_from([f'A{i}' for i in range(len(ids))])
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            if J[i,j]>=.45:G2.add_edge(f'A{i}',f'A{j}',weight=J[i,j])
    pos2=nx.spring_layout(G2,seed=SEED); plt.figure(figsize=(6.6,5)); nx.draw_networkx(G2,pos2,node_size=95,with_labels=False,node_color='#6c5b7b',edge_color='#9aa0aa',width=.8); plt.title('Candidate neutralization-overlap graph (synthetic)'); plt.axis('off'); savefig('fig08_overlap_graph')
    # 9 flow diagram 1
    plt.figure(figsize=(9,2.6)); ax=plt.gca(); ax.axis('off'); boxes=[('Candidate repertoire\nwith measure mu',.08),('Target-indexed\nranges R_x',.32),('VC / definable\ncomplexity d',.56),('Sample an\nepsilon-net',.82)]
    for txt,xx in boxes: ax.text(xx,.57,txt,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#edf2f5',ec='#355c7d'),fontsize=9)
    for a,b in [(.18,.22),(.42,.46),(.66,.72)]: ax.annotate('',(b,.57),(a,.57),arrowprops=dict(arrowstyle='->',lw=1.4))
    ax.text(.5,.13,'Certificate only for ranges whose candidate mass is at least epsilon',ha='center',fontsize=8.5); plt.title('Conditional epsilon-net certificate'); savefig('fig09_epsilon_pipeline')
    # 10 active assay flow
    plt.figure(figsize=(9,3.2)); ax=plt.gca(); ax.axis('off'); steps=[('Posterior response\nmodel',.12),('Choose assay by\nuncertainty and cost',.37),('Measure IC50 / IC80\nfor selected pair',.63),('Update lower\ncoverage margin',.88)]
    for txt,xx in steps: ax.text(xx,.66,txt,ha='center',va='center',bbox=dict(boxstyle='round,pad=.55',fc='#f2ecef',ec='#6c5b7b'),fontsize=8.5)
    for a,b in [(.23,.26),(.49,.52),(.75,.78)]: ax.annotate('',(b,.66),(a,.66),arrowprops=dict(arrowstyle='->',lw=1.3))
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((.88,.43),(.12,.43),connectionstyle='arc3,rad=-.32',arrowstyle='-|>',mutation_scale=11,lw=1.2,color='#6c5b7b')); ax.text(.5,.08,'Repeat until certified, or abstain and request more assays',ha='center',fontsize=8.3); plt.title('Proposed AI-assisted assay-selection loop'); savefig('fig10_active_assay_loop')
    # 11 histogram
    plt.figure(figsize=(6.5,4)); plt.hist(mass,bins=18,edgecolor='white'); plt.axvline(EPS,ls='--',c='#c06c84',label=f'epsilon={EPS:.2f}'); plt.xlabel('Empirical candidate-repertoire mass'); plt.ylabel('Number of target variants'); plt.title('Synthetic target-range mass distribution'); plt.legend(frameon=False); savefig('fig11_range_mass_distribution')
    # 12 threshold sensitivity
    plt.figure(figsize=(6.5,4)); plt.plot([.5,.6,.7,.8,.9],tm.mean(axis=1),'o-',label='Mean candidate mass'); plt.plot([.5,.6,.7,.8,.9],(tm>=EPS).mean(axis=1),'s--',label='Fraction epsilon-heavy'); plt.xlabel('Neutralization threshold'); plt.ylabel('Mass / target fraction'); plt.title('Coverage sensitivity to threshold'); plt.legend(frameon=False); savefig('fig12_threshold_sensitivity')
    # checks
    assert abs(float(hill_neutralization(.35,.35))-.5)<1e-12
    assert np.all(np.diff(hill_neutralization(np.linspace(0,.8,100),.35))<0)
    assert np.array_equal(C,D<=radii[None,:])
    assert np.all(counts>=K) and len(np.unique(two))==len(two)
    assert np.all((mass>=0)&(mass<=1))
    try: hill_neutralization(-.1,.2); raise AssertionError('invalid distance accepted')
    except ValueError: pass
    try: greedy_multicover(np.zeros((1,2),bool),1); raise AssertionError('infeasible cover accepted')
    except RuntimeError: pass
    assert np.random.default_rng(SEED).integers(2**31)==np.random.default_rng(SEED).integers(2**31)
    oldout=RES/'numerical_outputs.json'; prior=json.loads(oldout.read_text()).get('timestamp_utc') if oldout.exists() else None
    result={'timestamp_utc':prior or datetime.now(timezone.utc).isoformat(),'seed':SEED,'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'networkx':nx.__version__},'design':{'antibody_candidates':na,'target_variants':nv,'epsilon':EPS,'k':K,'range':'variable-radius Euclidean disks in 2D','hill_slope':1.4},'coverage':{'fraction_epsilon_heavy':float(heavy.mean()),'mean_range_mass':float(mass.mean()),'median_range_mass':float(np.median(mass)),'minimum_range_mass':float(mass.min())},'greedy_finite_cover':{'single':int(len(one)),'double':int(len(two)),'min_double_multiplicity':int(counts.min()),'k_1_to_4_sizes':sizes,'optimality':'not established'},'sampling':{'trials':trials,'sample_sizes':ns.tolist(),'hit_fractions':success.tolist()},'robust_neighbourhood':{'rho':rhos.tolist(),'mean_mass':robust.mean(axis=1).tolist(),'epsilon_heavy_fraction':(robust>=EPS).mean(axis=1).tolist(),'test':'four cardinal offsets only; not continuum worst-case'},'checks':[{'name':n,'status':'PASS'} for n in ['Hill IC50 threshold equals 0.5','monotone Hill response','threshold equals Euclidean disk','finite double-cover feasibility','unique selected antibodies','range masses lie in [0,1]','invalid negative-distance rejected','infeasible multicover rejected','deterministic seed']],'empirical_data':'No CATNAP or antivenom table downloaded or analyzed; all numeric outputs synthetic.'}
    (RES/'numerical_outputs.json').write_text(json.dumps(result,indent=2)+'\n')
    (RES/'verification_summary.txt').write_text('\n'.join(['SYNTHETIC VERIFICATION SUMMARY','==============================',f'Seed: {SEED}',f'Candidate antibodies: {na}',f'Target variants: {nv}',f'epsilon: {EPS}',f'epsilon-heavy target fraction: {heavy.mean():.6f}',f'mean/minimum range mass: {mass.mean():.6f} / {mass.min():.6f}',f'greedy single cover: {len(one)}',f'greedy double cover: {len(two)}',f'minimum double-cover multiplicity: {counts.min()}',f'random-sampling hit fractions: {dict(zip(ns.tolist(),success.round(3).tolist()))}','Checks: PASS 9/9','Scope: synthetic code and finite-instance verification only; not empirical or clinical validation.'])+'\n')
    print((RES/'verification_summary.txt').read_text())
    return result
if __name__=='__main__': run()