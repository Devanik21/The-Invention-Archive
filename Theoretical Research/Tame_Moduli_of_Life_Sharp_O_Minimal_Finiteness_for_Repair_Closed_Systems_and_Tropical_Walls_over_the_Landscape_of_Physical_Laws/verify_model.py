#!/usr/bin/env python3
"""Reproducible mathematical checks and synthetic figures for the tame-life toy model.

Dependencies: numpy, scipy, matplotlib, networkx, pandas (only numpy/scipy/matplotlib/networkx used).
The model is synthetic and serves as a proof-of-concept for definable parameter regions.
It is not an empirical model of life, origin of life, extraterrestrial biology, or physical law.
"""
from __future__ import annotations
import csv, json, math, platform, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SEED = 20261031
EPS = 0.05
C0 = 0.04
N = 801

# One deliberately small, semialgebraic test family.
def life_region(a: np.ndarray, b: np.ndarray, c: float = C0) -> np.ndarray:
    return (a >= EPS) & (b >= EPS) & (a <= 1-EPS) & (b <= 1-EPS) & ((a+b) <= 1.0) & ((a*b) >= c)

def repair_map(x: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    if not (0.0 <= lam <= 1.0):
        raise ValueError("repair fraction must lie in [0,1]")
    return (1.0-lam)*x + lam*target

def damage(x: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum((x-target)**2))

def component_count(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask, structure=np.ones((3,3), dtype=int))
    return int(n)

def write_csv(path: Path, header: list[str], rows: list[list[float|int|str]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG/f"{name}.pdf", bbox_inches='tight')
    plt.savefig(FIG/f"{name}.png", dpi=220, bbox_inches='tight')
    plt.close()

# Grid evaluation; grid approximates the semialgebraic region, not its proof.
a=np.linspace(EPS,1-EPS,N); b=np.linspace(EPS,1-EPS,N)
A,B=np.meshgrid(a,b,indexing='xy')
M=life_region(A,B,C0)
components=component_count(M)
cell_area=((1-2*EPS)/(N-1))**2
area_est=float(M.sum()*cell_area)
# Boundary intersections ab=c and a+b=1, analytically solved.
disc=1.0-4.0*C0
if disc < 0: raise AssertionError('toy life region should have a nonempty analytic boundary')
rlo=(1.0-math.sqrt(disc))/2.0
rhi=(1.0+math.sqrt(disc))/2.0
# A feasible witness and positive autocatalytic flux for the fixed two-species core.
witness=np.array([0.25,0.25])
assert witness.sum() <= 1 and witness.prod() >= C0
S=np.eye(2); flux=np.array([1.0,1.0]); production=S@flux
assert np.all(production > 0)
# Exact repair contraction identity: V(R_lambda(x))=(1-lambda)^2 V(x).
x=np.array([0.8,0.15]); target=np.array([0.25,0.25])
repair_rows=[]
for lam in np.linspace(0,1,21):
    xr=repair_map(x,target,float(lam)); v0=damage(x,target); v1=damage(xr,target)
    assert v1 <= v0 + 1e-12
    assert abs(v1-(1-lam)**2*v0) < 1e-12
    repair_rows.append([float(lam),v0,v1])
write_csv(RES/'repair_contraction.csv',['lambda','damage_before','damage_after'],repair_rows)
# Tropical wall is exact for the binomial ab-c in log coordinates.
trop_rows=[]
for aa in np.linspace(EPS,1-EPS,200):
    bb=C0/aa
    if EPS <= bb <= 1-EPS:
        trop_rows.append([float(aa),float(bb),float(math.log(aa)+math.log(bb)-math.log(C0))])
assert len(trop_rows)>0 and max(abs(r[2]) for r in trop_rows)<1e-12
write_csv(RES/'tropical_wall_points.csv',['a','b','log_residual'],trop_rows)
# Polynomial sign sets: explicit simple roots show degree-sensitive component growth.
complexity_rows=[]
for degree in range(2,22,2):
    roots=np.linspace(-0.85,0.85,degree)
    grid=np.linspace(-1,1,20001)
    p=np.ones_like(grid)
    for root in roots: p *= (grid-root)
    pos=p>=0
    ncomp=component_count(pos.reshape(1,-1))
    expected=degree//2+1
    assert ncomp==expected, (degree,ncomp,expected)
    complexity_rows.append([degree,ncomp,expected])
write_csv(RES/'degree_component_demo.csv',['polynomial_degree','observed_components','expected_components'],complexity_rows)
# Grid resolution convergence.
conv=[]
for n in [101,201,401,801]:
    aa=np.linspace(EPS,1-EPS,n); AA,BB=np.meshgrid(aa,aa)
    mm=life_region(AA,BB,C0)
    conv.append([n,int(mm.sum()),component_count(mm),float(mm.sum()*(((1-2*EPS)/(n-1))**2))])
write_csv(RES/'grid_convergence.csv',['grid_n','feasible_nodes','components_8_connected','area_estimate'],conv)
# Save grid and a small deterministic numerical summary.
write_csv(RES/'toy_landscape_grid.csv',['a','b','life_indicator','repair_margin'],
          [[float(A[i,j]),float(B[i,j]),int(M[i,j]),float(A[i,j]*B[i,j]-C0)]
           for i in range(0,N,8) for j in range(0,N,8)])
summary={
  'model':'synthetic finite-dimensional semialgebraic toy family',
  'seed':SEED,'grid_n':N,'epsilon':EPS,'damage_threshold_c':C0,
  'connected_components_8_connected':components,'grid_area_estimate':area_est,
  'boundary_intersections_with_a_plus_b_eq_1':[rlo,rhi],
  'feasible_witness':[float(witness[0]),float(witness[1])],
  'stoichiometric_matrix':S.tolist(),'positive_flux':flux.tolist(),'net_production':production.tolist(),
  'repair_contraction_identity_tested':True,
  'tropical_binomial_log_residual_max':max(abs(r[2]) for r in trop_rows),
  'complexity_demo':complexity_rows,
  'grid_convergence':conv,
  'interpretation':'Toy mathematical checks only; no empirical or universal biological inference.'
}
(RES/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')

# Figure 1: exact semialgebraic life region on a parameter slice.
plt.figure(figsize=(6.8,5.0)); plt.contourf(A,B,M.astype(int),levels=[-.5,.5,1.5],alpha=.8)
plt.contour(A,B,A*B-C0,levels=[0],linewidths=1.8)
plt.contour(A,B,A+B-1,levels=[0],linestyles='--',linewidths=1.6)
plt.scatter([witness[0]],[witness[1]],marker='*',s=100,label='feasible witness')
plt.xlabel('law parameter a'); plt.ylabel('law parameter b'); plt.title('Synthetic repair-admissible region (c = 0.04)'); plt.legend(); savefig('fig01_life_region')
# Figure 2: repair margin in parameter space.
plt.figure(figsize=(6.8,5.0)); im=plt.imshow(A*B-C0,origin='lower',extent=[EPS,1-EPS,EPS,1-EPS],aspect='equal')
plt.colorbar(im,label='repair margin ab − c'); plt.contour(A,B,(A+B<=1),levels=[.5],colors='black',linewidths=1.2)
plt.xlabel('a'); plt.ylabel('b'); plt.title('Margin field and resource-budget boundary'); savefig('fig02_margin_field')
# Figure 3: repair contraction.
lam=np.linspace(0,1,101); v0=damage(x,target); vafter=np.array([damage(repair_map(x,target,float(l)),target) for l in lam])
plt.figure(figsize=(6.8,4.5)); plt.plot(lam,np.full_like(lam,v0),label='before repair'); plt.plot(lam,vafter,label='after repair'); plt.plot(lam,(1-lam)**2*v0,'--',label='exact contraction law')
plt.xlabel('repair fraction λ'); plt.ylabel('quadratic damage V'); plt.title('One-step repair map contracts a Lyapunov proxy'); plt.legend(); savefig('fig03_repair_contraction')
# Figure 4: tropical binomial wall in logarithmic coordinates.
plt.figure(figsize=(6.8,5.0)); aa=np.geomspace(EPS,1-EPS,300); bb=C0/aa; ok=(bb>=EPS)&(bb<=1-EPS)
plt.plot(np.log(aa[ok]),np.log(bb[ok]),lw=2,label='log a + log b = log c')
plt.scatter(np.log(aa[ok][::20]),np.log(bb[ok][::20]),s=15,label='numerical wall samples')
plt.xlabel('log a'); plt.ylabel('log b'); plt.title('Tropical wall for the binomial ab − c'); plt.legend(); savefig('fig04_tropical_wall')
# Figure 5: degree versus number of connected sign components.
cr=np.array(complexity_rows)
plt.figure(figsize=(6.8,4.5)); plt.plot(cr[:,0],cr[:,1],marker='o',label='observed'); plt.plot(cr[:,0],cr[:,2],'--',label='D/2 + 1');
plt.xlabel('polynomial degree D'); plt.ylabel('components of {p_D ≥ 0}'); plt.title('One-dimensional illustrative degree-complexity relation'); plt.legend(); savefig('fig05_degree_components')
# Figure 6: robustness by damage threshold.
cs=np.linspace(.005,.20,50); feasible=[]
for cc in cs: feasible.append(float(life_region(A,B,float(cc)).sum()/A.size))
plt.figure(figsize=(6.8,4.5)); plt.plot(cs,feasible,lw=2); plt.axvline(C0,ls='--',label='reference c = 0.04')
plt.xlabel('damage threshold c'); plt.ylabel('feasible grid fraction'); plt.title('Sensitivity of the synthetic admissible region'); plt.legend(); savefig('fig06_threshold_sensitivity')
# Figure 7: two-species autocatalytic reaction graph.
G=nx.DiGraph(); G.add_nodes_from(['A','B']); G.add_edge('A','A',label='A autocatalysis'); G.add_edge('B','B',label='B autocatalysis'); G.add_edge('A','B',label='B as co-reactant'); G.add_edge('B','A',label='A as co-reactant')
pos={'A':(-1,0),'B':(1,0)}
plt.figure(figsize=(6.8,4.5)); nx.draw_networkx_nodes(G,pos,node_size=1800,node_color='#dce8ef',edgecolors='#34495e'); nx.draw_networkx_labels(G,pos,font_size=14)
nx.draw_networkx_edges(G,pos,edgelist=[('A','A'),('B','B')],connectionstyle='arc3,rad=0.45',arrows=True,arrowsize=20,width=2)
nx.draw_networkx_edges(G,pos,edgelist=[('A','B'),('B','A')],connectionstyle='arc3,rad=0.1',arrows=True,arrowsize=20,width=1.5,style='dashed')
plt.title('Conceptual two-species catalytic core used in the toy model'); plt.axis('off'); savefig('fig07_autocatalytic_graph')
# Figure 8: state-space graph for repair map (discretized illustrative network).
H=nx.DiGraph(); vals=np.linspace(0,1,5)
for i,u in enumerate(vals):
 for j,v in enumerate(vals):
  node=f'{i},{j}'; H.add_node(node,xy=(u,v))
  ni=int(round((i+2)/2)); nj=int(round((j+1)/2))
  dest=f'{ni},{nj}'
  if dest in H: H.add_edge(node,dest)
xy={n:d['xy'] for n,d in H.nodes(data=True)}
plt.figure(figsize=(6.8,5.0)); nx.draw_networkx_nodes(H,xy,node_size=350,node_color='#dce8ef',edgecolors='#34495e'); nx.draw_networkx_edges(H,xy,arrows=True,arrowsize=12,alpha=.7);
plt.xlabel('state coordinate x₁'); plt.ylabel('state coordinate x₂'); plt.title('Discretized repair-state transition graph (illustrative)'); savefig('fig08_repair_state_graph')
# Figure 9: conceptual model pipeline flow.
def flow_figure(name, title, boxes):
    fig,ax=plt.subplots(figsize=(8.2,2.5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    xs=np.linspace(.08,.92,len(boxes)); w=min(.18,.82/len(boxes))
    for i,(xx,label) in enumerate(zip(xs,boxes)):
        patch=FancyBboxPatch((xx-w/2,.36),w,.3,boxstyle='round,pad=0.02',fc='#e8eef2',ec='#34495e',lw=1.2); ax.add_patch(patch)
        ax.text(xx,.51,label,ha='center',va='center',fontsize=8,wrap=True)
        if i<len(boxes)-1: ax.add_patch(FancyArrowPatch((xx+w/2,.51),(xs[i+1]-w/2,.51),arrowstyle='->',mutation_scale=14,lw=1.2))
    ax.set_title(title,fontsize=11); fig.tight_layout(); fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight'); fig.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight'); plt.close(fig)
flow_figure('fig09_model_flow','From finite law parameters to a repair-admissibility predicate',['law parameters θ','finite reaction core','repair map + invariant set','definable predicate L(θ)','tame component bound'])
flow_figure('fig10_ai_audit_flow','Proposed AI-assisted symbolic audit pipeline',['mechanistic model','symbolic AST','format / degree ledger','proof + tropical checks','human review'])
# Figure 11: grid convergence in area and component count.
cv=np.array(conv)
plt.figure(figsize=(6.8,4.5)); plt.plot(cv[:,0],cv[:,3],marker='o');
plt.xlabel('grid resolution n'); plt.ylabel('estimated feasible area'); plt.title('Grid approximation convergence (not a proof of finiteness)'); savefig('fig11_grid_convergence')
# Figure 12: tropical error residuals and exact arithmetic check.
resids=np.array([r[2] for r in trop_rows])
zero_count=int(np.sum(resids==0.0)); roundoff_count=int(np.sum((np.abs(resids)>0.0)&(np.abs(resids)<=5e-16)))
plt.figure(figsize=(6.8,4.5)); plt.bar(['exact floating-point zero','nonzero roundoff (≤ 5×10⁻¹⁶)'],[zero_count,roundoff_count],width=0.62)
plt.ylabel('number of sampled wall points'); plt.title('Floating-point residual classes on the exact binomial wall')
plt.xticks(rotation=8,ha='right'); savefig('fig12_tropical_residual')

# Manifest of generated figure/data outputs (updated by deployment script after PDF build).
report={
 'status':'PASS' if components==1 and len(trop_rows)>0 else 'FAIL',
 'assertions':{
  'feasible_witness_satisfies_budget_and_repair':bool(witness.sum()<=1 and witness.prod()>=C0),
  'stoichiometric_positive_flux':bool(np.all(production>0)),
  'repair_map_contracts_damage_for_all_tested_lambdas':True,
  'tropical_binomial_log_wall_residual_below_1e-12':bool(max(abs(r[2]) for r in trop_rows)<1e-12),
  'degree_component_demo_matches_exact_expected_count':True,
  'grid_connected_components_detected':components,
  'all_12_scientific_figure_assets_generated':len(list(FIG.glob('fig*.pdf'))) == 12,
 },
 'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'matplotlib':matplotlib.__version__,'networkx':nx.__version__},
 'limitations':['synthetic model only','grid component count is numerical illustration, not theorem','no empirical biology or physical-law inference']
}
(RES/'model_test_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'components':components,'area_estimate':area_est,'figures':len(list(FIG.glob('fig*.pdf'))),'tropical_residual':summary['tropical_binomial_log_residual_max']},indent=2))
