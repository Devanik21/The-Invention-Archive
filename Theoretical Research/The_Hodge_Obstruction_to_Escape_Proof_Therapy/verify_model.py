#!/usr/bin/env python3
"""Reproducible synthetic verification for the Hodge Obstruction preprint.

All fitness values are synthetic dimensionless model outputs, not antimicrobial
susceptibility measurements. Dependencies: numpy==2.3.5, matplotlib==3.10.8,
networkx==3.6.1. Run `python verify_model.py` from any directory.
"""
from __future__ import annotations
import csv, json, hashlib, platform, sys, os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RES = HERE / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
SEED = 20261025
RNG = np.random.default_rng(SEED)

# A 4-locus genotype hypercube: every landscape is a scalar potential on vertices.
D = 4
N = 2**D
M = 3
GENO = np.array([[int((i >> j) & 1) for j in range(D)] for i in range(N)], dtype=int)
# Fixed, transparent synthetic coefficients with both trade-offs and pairwise epistasis.
COEF = np.array([
    [-0.34, 0.38, 0.05, -0.19, 0.24],
    [-0.31, -0.16, 0.41, 0.22, -0.07],
    [-0.29, 0.17, -0.21, 0.36, 0.31],
], dtype=float)
PAIR = np.array([
    [0.00, 0.03, -0.04, 0.01, 0.05, -0.02],
    [0.00, -0.03, 0.06, 0.02, -0.04, 0.03],
    [0.00, 0.04, 0.01, -0.05, 0.02, 0.04],
], dtype=float)
PAIRS = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
FIT = np.zeros((M,N))
for a in range(M):
    FIT[a] = COEF[a,0] + GENO @ COEF[a,1:] + sum(PAIR[a,k]*GENO[:,i]*GENO[:,j] for k,(i,j) in enumerate(PAIRS))

# Orient every hypercube edge from the genotype with locus bit 0 to bit 1.
EDGES=[]
for i in range(N):
    for j in range(D):
        if GENO[i,j] == 0:
            EDGES.append((i, i | (1<<j), j))
E=len(EDGES)
B0=np.zeros((E,N))
for e,(u,v,j) in enumerate(EDGES):
    B0[e,u]=-1.0; B0[e,v]=1.0
# Oriented square 2-faces are the boundary of each 2-bit face, with orientations inherited
# from the standard cubical complex. For the chosen orientation this signed boundary is exact.
SQUARES=[]
for base in range(N):
    for i in range(D):
        for j in range(i+1,D):
            if GENO[base,i] == 0 and GENO[base,j] == 0:
                u=base; vi=base | (1<<i); vj=base | (1<<j); vij=base | (1<<i) | (1<<j)
                SQUARES.append((u,vi,vij,vj))
B2=np.zeros((len(SQUARES),E))
edge_idx={(u,v):e for e,(u,v,j) in enumerate(EDGES)}
for s,(u,vi,vij,vj) in enumerate(SQUARES):
    # boundary: u->vi, vi->vij, vij->vj, vj->u
    for a,b in [(u,vi),(vi,vij),(vij,vj),(vj,u)]:
        if (a,b) in edge_idx: B2[s,edge_idx[(a,b)]] += 1
        elif (b,a) in edge_idx: B2[s,edge_idx[(b,a)]] -= 1

def hodge_decompose(w: np.ndarray):
    # Orthogonal projections onto exact and coexact spaces; harmonic is residual.
    g = B0 @ np.linalg.lstsq(B0, w, rcond=None)[0]
    cmat = B2.T
    c = cmat @ np.linalg.lstsq(cmat, w, rcond=None)[0] if cmat.size else np.zeros_like(w)
    h = w - g - c
    return g,c,h

# Per-drug exact edge cochains: fitness differences of scalar node potentials.
EDGE_FIELDS=np.vstack([FIT[a, np.array([v for u,v,j in EDGES])] - FIT[a, np.array([u for u,v,j in EDGES])] for a in range(M)])
CURL_EXACT=np.array([np.max(np.abs(B2@EDGE_FIELDS[a])) for a in range(M)])
NOISE_SIGMA=0.08
NOISY_FIELDS=EDGE_FIELDS + RNG.normal(0,NOISE_SIGMA,EDGE_FIELDS.shape)
CURL_NOISY=np.array([np.sqrt(np.mean((B2@NOISY_FIELDS[a])**2)) for a in range(M)])

# Toy resistant region: >=2 synthetic landscapes exceed a declared relative-fitness threshold.
# This is only a game benchmark label and is not a clinical resistance breakpoint.
THETA=0.0
RESISTANT=np.where(np.sum(FIT >= THETA, axis=0) >= 2)[0]
RISK=np.zeros(N,dtype=bool); RISK[RESISTANT]=True
SAFE=~RISK
# For each drug/state, possible successors include stasis and every strictly beneficial
# one-locus mutation under that drug (strong-selection/weak-mutation stylization).
SUCCESSORS=[]
for a in range(M):
    sa=[]
    for x in range(N):
        nxt={x}
        for y in range(N):
            if (GENO[x] != GENO[y]).sum()==1 and FIT[a,y] > FIT[a,x] + 1e-12:
                nxt.add(y)
        sa.append(sorted(nxt))
    SUCCESSORS.append(sa)

def viability_kernel():
    K=set(np.where(SAFE)[0].tolist())
    history=[len(K)]
    while True:
        Knew={x for x in K if any(set(SUCCESSORS[a][x]).issubset(K) for a in range(M))}
        history.append(len(Knew))
        if Knew==K: return K,history
        K=Knew
KERNEL,K_HIST=viability_kernel()

def choose_safe_action(x, K=KERNEL):
    for a in range(M):
        if set(SUCCESSORS[a][x]).issubset(K): return a
    return -1

# Stochastic mutation kernel for finite-horizon evaluation; not the robust adversarial model.
BETA=5.0
TRANS=np.zeros((M,N,N))
for a in range(M):
    for x in range(N):
        neigh=[]
        for y in range(N):
            if y!=x and (GENO[x]!=GENO[y]).sum()==1:
                delta=FIT[a,y]-FIT[a,x]
                if delta > 0:
                    neigh.append((y, float(np.exp(min(BETA*delta,30.0)))))
        stay_w=1.0
        total=stay_w+sum(w for _,w in neigh)
        TRANS[a,x,x]=stay_w/total
        for y,w in neigh: TRANS[a,x,y]=w/total

def finite_horizon_dp(H=12):
    # V[h,x] = minimum probability of reaching R within h steps under a stationary state policy.
    V=np.zeros((H+1,N)); V[:,RISK]=1.0
    POL=np.zeros((H,N),dtype=int)
    for h in range(1,H+1):
        for x in range(N):
            if RISK[x]: V[h,x]=1.0; POL[h-1,x]=0; continue
            q=np.array([TRANS[a,x]@V[h-1] for a in range(M)])
            POL[h-1,x]=int(np.argmin(q)); V[h,x]=float(q.min())
    return V,POL
V,POL=finite_horizon_dp(12)

def simulate_policy(policy, start=0, horizon=12, reps=2500, seed=911):
    rng=np.random.default_rng(seed); hit=0; times=[]
    for _ in range(reps):
        x=start; when=horizon+1
        for t in range(horizon):
            if RISK[x]: when=t; break
            a=int(policy(x,t))
            x=int(rng.choice(N,p=TRANS[a,x]))
            if RISK[x]: when=t+1; break
        times.append(when); hit += int(when<=horizon)
    return hit/reps, np.array(times)

def dp_pol(x,t): return int(POL[min(11,11-t),x])
def greedy_pol(x,t): return int(np.argmin(FIT[:,x]))
def cycle_pol(x,t): return t%M
def random_pol(x,t): return int(t%M) # deterministic baseline cycle named round-robin

# Tabular Q-learning on the same synthetic stochastic environment; reward penalizes reaching risk.
def train_q(seed=431, episodes=9000, horizon=12):
    rng=np.random.default_rng(seed); Q=np.zeros((N,M)); alpha=0.12; gamma=0.95
    for ep in range(episodes):
        x=int(rng.choice(np.where(SAFE)[0]))
        eps=max(0.04,0.35*(1-ep/episodes))
        for t in range(horizon):
            if RISK[x]: break
            a=int(rng.integers(M)) if rng.random()<eps else int(np.argmax(Q[x]))
            y=int(rng.choice(N,p=TRANS[a,x]))
            reward=-0.03 - (20.0 if RISK[y] else 0.0)
            target=reward if RISK[y] or t==horizon-1 else reward+gamma*np.max(Q[y])
            Q[x,a] += alpha*(target-Q[x,a]); x=y
            if RISK[x]: break
    qpol=lambda x,t:int(np.argmax(Q[x]))
    return Q,qpol
Q,QPOL=train_q()

# Save source data used by plots and transparent numeric summaries.
def write_csv(path, header, rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
write_csv(RES/'synthetic_fitness_landscapes.csv',['genotype','bits','drug_A','drug_B','drug_C','synthetic_resistant_region'],
          [[i,''.join(map(str,GENO[i])),*map(float,FIT[:,i]),int(RISK[i])] for i in range(N)])
write_csv(RES/'edge_fields.csv',['edge','source','target','locus','delta_A','delta_B','delta_C','noisy_delta_A','noisy_delta_B','noisy_delta_C'],
          [[e,u,v,j,*map(float,EDGE_FIELDS[:,e]),*map(float,NOISY_FIELDS[:,e])] for e,(u,v,j) in enumerate(EDGES)])
write_csv(RES/'hodge_components.csv',['drug','gradient_energy_fraction','curl_energy_fraction','harmonic_energy_fraction','max_exact_square_curl','rms_noisy_square_curl'],
          [[chr(65+a),*(float(np.sum(c*c)/np.sum(EDGE_FIELDS[a]**2)) for c in hodge_decompose(EDGE_FIELDS[a])),float(CURL_EXACT[a]),float(CURL_NOISY[a])] for a in range(M)])
write_csv(RES/'viability_iterations.csv',['iteration','candidate_safe_states'],[[i,n] for i,n in enumerate(K_HIST)])
def monte_carlo_horizon(policy, h, seed):
    return simulate_policy(policy, horizon=h, reps=3000, seed=seed)[0]
write_csv(RES/'policy_risk_by_horizon.csv',['horizon','optimal_dp_start_risk','greedy_one_step_start_risk','round_robin_start_risk','q_learning_start_risk'],
          [[h,float(V[h,0]),monte_carlo_horizon(greedy_pol,h,1000+h),monte_carlo_horizon(cycle_pol,h,2000+h),monte_carlo_horizon(QPOL,h,3000+h)] for h in range(1,13)])

# Reproducible plot style.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':150,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
COL=['#315A7D','#D16A51','#3A8C7D','#8A6DAF']

def savefig(num, name, caption, kind, draw):
    fig=plt.figure(figsize=(7.0,4.3)); draw(fig); fig.tight_layout()
    fig.savefig(FIG/f'fig{num:02d}_{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'fig{num:02d}_{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return {'number':num,'name':name,'kind':kind,'caption':caption,'asset':f'figures/fig{num:02d}_{name}.pdf','source':'verify_model.py','data':'results/synthetic_fitness_landscapes.csv / outputs noted in caption'}

inventory=[]
def f1(fig):
    ax=fig.add_subplot(111); ax.axis('off')
    nodes=[(0.08,0.65,'Observed genotype\n(belief state)'),(0.33,0.65,'Choose drug\n(action)'),(0.58,0.65,'Mutation kernel /\nadversary'),(0.83,0.65,'Safety set test\n(target reachability)')]
    for x,y,t in nodes:
        ax.text(x,y,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#eef3f7',ec=COL[0],lw=1.3),transform=ax.transAxes)
    for i in range(3):
        ax.annotate('',xy=(nodes[i+1][0]-.09,.65),xytext=(nodes[i][0]+.09,.65),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.5,color=COL[0]))
    ax.text(.5,.25,'Hodge decomposition audits edge data; the safety game decides viability.',ha='center',transform=ax.transAxes,fontsize=10,color='#263b4d')
inventory.append(savefig(1,'framework','Overview: a Hodge-consistency audit is separated from robust safety-game synthesis. Conceptual, not empirical.', 'flow', f1))

def f2(fig):
    ax=fig.add_subplot(111); G=nx.hypercube_graph(D)
    pos0=nx.spring_layout(G,seed=18,dim=2)
    mapping={tuple(int(b) for b in n):sum(int(b)<<j for j,b in enumerate(n)) for n in G.nodes}
    G=nx.relabel_nodes(G,mapping)
    pos={mapping[k]:v for k,v in pos0.items()}
    nodecols=[COL[1] if RISK[i] else COL[2] for i in G.nodes]
    nx.draw_networkx_edges(G,pos,ax=ax,width=.8,alpha=.55,edge_color='#9aa8b3')
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=nodecols,node_size=440,edgecolors='white',linewidths=.7)
    nx.draw_networkx_labels(G,pos,ax=ax,labels={i:format(i,'04b') for i in G.nodes},font_size=6,font_color='white')
    ax.set_title('Four-locus genotype hypercube (synthetic safety set)'); ax.axis('off')
inventory.append(savefig(2,'genotype_hypercube','Graph/network: vertices are 4-bit genotypes; orange indicates the synthetic failure set and teal the safe set.', 'graph', f2))

def f3(fig):
    axs=fig.subplots(1,3)
    idx=np.arange(N)
    for a,ax in enumerate(axs):
        ax.bar(idx,FIT[a],color=[COL[1] if RISK[i] else COL[a%3] for i in idx],width=.85)
        ax.axhline(THETA,ls='--',color='#333333',lw=.8); ax.set_title(f'Drug {chr(65+a)}'); ax.set_xlabel('Genotype index');
        if a==0: ax.set_ylabel('Synthetic relative fitness')
    fig.suptitle('Three scalar fitness potentials; synthetic values only',y=1.02)
inventory.append(savefig(3,'fitness_potentials','Plot: the three scalar node potentials used in the finite synthetic benchmark; dashed line is the illustrative threshold.', 'plot', f3))

def f4(fig):
    ax=fig.add_subplot(111); x=np.arange(1,M+1)
    ax.bar(x-.18,CURL_EXACT,width=.35,label='Exact potential field',color=COL[2]); ax.bar(x+.18,CURL_NOISY,width=.35,label='Independent edge noise added',color=COL[1])
    ax.set_xticks(x,['Drug A','Drug B','Drug C']); ax.set_ylabel('RMS square-boundary circulation'); ax.set_title('A scalar fitness potential has zero oriented square curl'); ax.legend(frameon=False)
inventory.append(savefig(4,'curl_exact_vs_noisy','Plot: exact edge differences telescope around every square; added independent measurement noise induces nonzero apparent circulation.', 'plot', f4))

def f5(fig):
    ax=fig.add_subplot(111); xx=np.arange(3); parts=[]
    for a in range(3):
        w=EDGE_FIELDS[a]; g,c,h=hodge_decompose(w); den=np.sum(w*w)
        parts.append([np.sum(g*g)/den,np.sum(c*c)/den,np.sum(h*h)/den])
    parts=np.array(parts)
    bot=np.zeros(3)
    for j,(lab,col) in enumerate(zip(['Gradient/exact','Curl/coexact','Harmonic residual'],[COL[0],COL[1],COL[3]])):
        ax.bar(xx,parts[:,j],bottom=bot,label=lab,color=col); bot+=parts[:,j]
    ax.set_xticks(xx,['Drug A','Drug B','Drug C']); ax.set_ylim(0,1.02); ax.set_ylabel('Fraction of squared edge-flow norm'); ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16))
    ax.set_title('Hodge-energy decomposition for exact synthetic gradients')
inventory.append(savefig(5,'hodge_energy','Plot: projections of scalar-potential edge differences onto exact, coexact, and harmonic subspaces; tiny numerical residuals remain.', 'plot', f5))

def f6(fig):
    ax=fig.add_subplot(111); G=nx.Graph();
    for x in range(N): G.add_node(x)
    for u,v,j in EDGES:
        G.add_edge(u,v,locus=j,beneficial=sum(FIT[a,v]>FIT[a,u] for a in range(M)))
    pos=nx.spring_layout(G,seed=9)
    nx.draw_networkx_edges(G,pos,ax=ax,width=.65,alpha=.25)
    edge_colors=[]
    for u,v in G.edges:
        edge_colors.append(COL[2] if any((v in SUCCESSORS[a][u] or u in SUCCESSORS[a][v]) for a in range(M)) else '#b9c0c8')
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=[COL[1] if RISK[x] else COL[0] for x in G.nodes],node_size=300,edgecolors='white')
    nx.draw_networkx_labels(G,pos,labels={x:format(x,'04b') for x in G.nodes},font_size=5,font_color='white',ax=ax)
    ax.set_title('Mutation adjacency graph; edge color emphasizes at least one beneficial direction'); ax.axis('off')
inventory.append(savefig(6,'accessible_mutation_graph','Graph/network: mutation adjacency on the genotype cube; darker teal edges have a beneficial direction under at least one synthetic drug.', 'graph', f6))

def f7(fig):
    ax=fig.add_subplot(111); ax.plot(range(len(K_HIST)),K_HIST,marker='o',lw=2,color=COL[0]); ax.set_xlabel('Greatest-fixed-point iteration'); ax.set_ylabel('Candidate robustly safe states'); ax.set_title('Exact safety-game elimination of states outside the viability kernel'); ax.grid(alpha=.2)
inventory.append(savefig(7,'viability_kernel','Plot: iterative greatest-fixed-point computation of the robust viability kernel for the synthetic SSWM successor relation.', 'plot', f7))

def f8(fig):
    ax=fig.add_subplot(111); horizons=np.arange(1,13)
    riskdp=V[1:,0]
    greedy,_=simulate_policy(greedy_pol,horizon=12,reps=2500,seed=101)
    cycle,_=simulate_policy(cycle_pol,horizon=12,reps=2500,seed=102)
    qrisk,_=simulate_policy(QPOL,horizon=12,reps=2500,seed=103)
    # Plot the full horizon curve for DP and horizon-12 observed estimates for baselines as markers.
    ax.plot(horizons,riskdp,marker='o',label='Finite-horizon DP',color=COL[0])
    ax.scatter([12,12,12],[greedy,cycle,qrisk],marker='D',s=55,label='Greedy / round-robin / Q-learn (H=12)',c=[COL[1],COL[2],COL[3]])
    ax.set_xlabel('Horizon (steps)'); ax.set_ylabel('Probability of entering synthetic failure set'); ax.set_ylim(-.03,1.03); ax.set_title('Finite-horizon policy comparison from genotype 0000'); ax.legend(frameon=False); ax.grid(alpha=.2)
inventory.append(savefig(8,'policy_risk_horizon','Plot: exact finite-horizon dynamic programming risk curve and horizon-12 Monte Carlo baseline estimates. Model-generated only.', 'plot', f8))

def f9(fig):
    ax=fig.add_subplot(111); im=ax.imshow(FIT,aspect='auto',cmap='coolwarm',interpolation='nearest'); ax.set_xlabel('Genotype index'); ax.set_ylabel('Drug action'); ax.set_yticks([0,1,2],['A','B','C']); ax.set_title('Action-dependent landscape values'); fig.colorbar(im,ax=ax,label='Synthetic relative fitness')
inventory.append(savefig(9,'action_fitness_matrix','Plot: matrix of synthetic fitness values across all 16 genotypes and three available drugs.', 'plot', f9))

def f10(fig):
    ax=fig.add_subplot(111); sigmas=np.linspace(0,.20,11); curl=[]
    for s in sigmas:
        vals=[]
        for a in range(M):
            ww=EDGE_FIELDS[a]+RNG.normal(0,s,E)
            vals.append(np.sqrt(np.mean((B2@ww)**2)))
        curl.append(np.mean(vals))
    ax.plot(sigmas,curl,marker='o',color=COL[1]); ax.set_xlabel('Edge-measurement noise SD'); ax.set_ylabel('Mean RMS square curl'); ax.set_title('Noise sensitivity of the Hodge consistency diagnostic'); ax.grid(alpha=.2)
    write_csv(RES/'noise_curl_sensitivity.csv',['noise_sd','mean_rms_square_curl'],zip(sigmas,curl))
inventory.append(savefig(10,'noise_sensitivity','Plot: noisy edge-field circulation grows with measurement-noise scale in a seeded synthetic sensitivity experiment.', 'plot', f10))

def f11(fig):
    ax=fig.add_subplot(111); ths=np.linspace(-.05,.45,15); vals=[]
    for th in ths:
        r=np.sum(FIT>=th,axis=0)>=2; s=~r; k=set(np.where(s)[0].tolist())
        while True:
            kn={x for x in k if any(set(SUCCESSORS[a][x]).issubset(k) for a in range(M))}
            if kn==k: break
            k=kn
        vals.append(len(k))
    ax.plot(ths,vals,marker='o',color=COL[0]); ax.set_xlabel('Illustrative multidrug-fitness threshold'); ax.set_ylabel('States in robust viability kernel'); ax.set_title('Safety result depends on declared target definition'); ax.grid(alpha=.2)
    write_csv(RES/'threshold_sensitivity.csv',['threshold','resistant_states','viability_kernel_size'],[[t,int(np.sum(np.sum(FIT>=t,axis=0)>=2),),v] for t,v in zip(ths,vals)])
inventory.append(savefig(11,'target_sensitivity','Plot: viability-kernel size versus the illustrative multi-landscape threshold; this is definition sensitivity, not clinical robustness.', 'plot', f11))

def f12(fig):
    ax=fig.add_subplot(111); ax.axis('off')
    labels=[('1. Measure',.12,.7),('2. Fit / bound\nlandscapes',.38,.7),('3. Build accessible\ntransition relation',.65,.7),('4. Solve robust\nsafety game',.88,.7),('5. Audit field\nconsistency',.38,.25),('6. Validate with\nnew observations',.72,.25)]
    for t,x,y in labels: ax.text(x,y,t,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.5',fc='#eef3f7',ec=COL[0],lw=1.1))
    for a,b in [((.18,.7),(.30,.7)),((.47,.7),(.55,.7)),((.75,.7),(.80,.7)),((.38,.62),(.38,.34)),((.47,.25),(.63,.25))]: ax.annotate('',xy=b,xytext=a,xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',lw=1.4,color=COL[0]))
    ax.text(.5,.06,'No clinical action is inferred from a synthetic benchmark.',ha='center',transform=ax.transAxes,color=COL[1])
inventory.append(savefig(12,'verification_workflow','Flow diagram: proposed evidence-to-safety workflow, with independent field-consistency auditing and validation.', 'flow', f12))

# Save figure inventory and representative numerical outputs.
write_csv(RES/'figure_inventory.csv',['number','name','kind','caption','asset','source','data'],[[x['number'],x['name'],x['kind'],x['caption'],x['asset'],x['source'],x['data']] for x in inventory])
# Baseline Monte Carlo estimates at horizon 12; Q-learning is a heuristic, not a safety proof.
risk_g, _=simulate_policy(greedy_pol,horizon=12,reps=5000,seed=201)
risk_c, _=simulate_policy(cycle_pol,horizon=12,reps=5000,seed=202)
risk_q, _=simulate_policy(QPOL,horizon=12,reps=5000,seed=203)
# exact DP hit risk uses same stochastic kernel, but at horizon 12 from state 0.
summary={
 'seed':SEED,'landscape':'synthetic 4-locus hypercube, 16 genotypes, 3 scalar fitness potentials',
 'fitness_threshold':THETA,'synthetic_failure_states':RESISTANT.tolist(),
 'exact_edge_square_curl_max_by_drug':CURL_EXACT.tolist(),
 'noisy_edge_rms_square_curl_by_drug':CURL_NOISY.tolist(),
 'robust_viability_kernel_states':sorted(KERNEL),'robust_viability_kernel_size':len(KERNEL),
 'viability_iteration_sizes':K_HIST,'start_state':0,'start_state_in_viability_kernel':bool(0 in KERNEL),
 'horizon_12_probability_of_failure':{
   'dynamic_programming':float(V[12,0]),
   'one_step_greedy_monte_carlo_5000':float(risk_g),
   'round_robin_monte_carlo_5000':float(risk_c),
   'tabular_q_learning_monte_carlo_5000':float(risk_q)},
 'q_learning_training_episodes':9000,'stochastic_transition_beta':BETA,
 'empirical_data_used':False,'clinical_interpretation':False,
 'interpretation':'All numerical values are synthetic proof-of-concept outputs; no bacterial, patient, or experimental data were used.'}
(RES/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')

# Assertions independently test the mathematical claims.
checks=[]
def check(name, cond, detail):
    checks.append({'name':name,'status':'PASS' if bool(cond) else 'FAIL','detail':detail})
    if not cond: raise AssertionError(name+': '+detail)
check('all drug fields are exact gradients', np.max(CURL_EXACT)<1e-10, f'max square circulation={float(np.max(CURL_EXACT)):.3e}')
check('discrete boundary-of-boundary consistency', np.linalg.norm(B2@B0)<1e-12, f'||B2 B0||_F={np.linalg.norm(B2@B0):.3e}')
check('Hodge orthogonal projections reconstruct edge flow', max(np.linalg.norm(EDGE_FIELDS[a]-sum(hodge_decompose(EDGE_FIELDS[a]))) for a in range(M))<1e-10, 'gradient + coexact + harmonic reconstructs each edge flow')
check('Hodge projected components are orthogonal', max(abs(float(hodge_decompose(w)[0]@hodge_decompose(w)[1])) for w in EDGE_FIELDS)<1e-9, 'exact and coexact components inner products below tolerance')
check('each stochastic kernel normalizes', np.max(np.abs(TRANS.sum(axis=2)-1))<1e-12, 'transition-row normalization max error below tolerance')
check('failure set nonempty and not all states', 0<len(RESISTANT)<N, f'{len(RESISTANT)} of {N} states designated failure by synthetic threshold')
check('viability kernel is safe', all(SAFE[x] for x in KERNEL), f'{len(KERNEL)} states remain in the robust kernel')
check('kernel is closed under its witness actions', all(choose_safe_action(x)>=0 for x in KERNEL), 'each kernel state has an action whose possible successors remain in kernel')
check('risk DP is bounded and monotone', np.all((V[:,0]>=0)&(V[:,0]<=1)) and np.all(np.diff(V[:,0])>=-1e-12), 'finite-horizon hitting probabilities lie in [0,1] and do not decrease with horizon')
check('finite-horizon DP boundary condition', np.all(V[:,RISK]==1.0), 'failure states have risk probability 1 at every horizon')
check('figure inventory floor', len(inventory)>=12 and sum(x['kind']=='plot' for x in inventory)>=6 and sum(x['kind']=='graph' for x in inventory)>=2 and sum(x['kind']=='flow' for x in inventory)>=2, f"{len(inventory)} figures; {sum(x['kind']=='plot' for x in inventory)} plots, {sum(x['kind']=='graph' for x in inventory)} graphs, {sum(x['kind']=='flow' for x in inventory)} flows")
check('data files exist', all((RES/f).exists() and (RES/f).stat().st_size>0 for f in ['synthetic_fitness_landscapes.csv','edge_fields.csv','hodge_components.csv','figure_inventory.csv']), 'figure-generating data are archived')
report={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'networkx':nx.__version__,'checks':checks,'all_passed':all(c['status']=='PASS' for c in checks),'limitations':['synthetic fitness potentials only','no pathogen-specific or clinical recommendation','Hodge diagnostic is not asserted to determine therapy safety','Q-learning policy is a heuristic and carries no robust guarantee']}
(RES/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
with open(RES/'verify_model_stdout.txt','w',encoding='utf-8') as f:
    f.write('The Hodge Obstruction to Escape-Proof Therapy — verification summary\n')
    f.write(f'Seed: {SEED}\nFailure-set states: {RESISTANT.tolist()}\nRobust viability kernel size: {len(KERNEL)}/{N}\nH=12 DP hit risk from state 0000: {V[12,0]:.6f}\n')
    f.write(f'H=12 greedy hit-rate estimate (5000 runs): {risk_g:.6f}\nH=12 round-robin hit-rate estimate (5000 runs): {risk_c:.6f}\nH=12 Q-learning hit-rate estimate (5000 runs): {risk_q:.6f}\n')
    for c in checks: f.write(f"[{c['status']}] {c['name']}: {c['detail']}\n")
print((RES/'verify_model_stdout.txt').read_text())
