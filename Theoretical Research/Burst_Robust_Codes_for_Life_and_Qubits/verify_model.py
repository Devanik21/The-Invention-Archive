#!/usr/bin/env python3
from pathlib import Path
import json,math
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/"results"; F=Path(__file__).resolve().parent/"figures"
R.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
N=24; SEED=20261106; REPS=4000; GRID=np.arange(.20,.651,.05)
def spans(m):
 seen=np.zeros_like(m,dtype=bool)
 for y,x in zip(*np.where(m)):
  if seen[y,x]: continue
  q=[(y,x)];seen[y,x]=1;a=b=x;c=d=y
  while q:
   yy,xx=q.pop()
   for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
    u,v=yy+dy,xx+dx
    if 0<=u<N and 0<=v<N and m[u,v] and not seen[u,v]:
     seen[u,v]=1;q.append((u,v));a=min(a,v);b=max(b,v);c=min(c,u);d=max(d,u)
  if (a==0 and b==N-1) or (c==0 and d==N-1):return True
 return False
def track(p,r):
 m=np.zeros((N,N),bool);lam=-math.log(max(1e-9,1-p))*N*N/5.5;dirs=((1,0),(0,1),(1,1),(1,-1))
 for _ in range(r.poisson(lam)):
  y=int(r.integers(N));x=int(r.integers(N));dy,dx=dirs[int(r.integers(4))]
  for k in range(-2,3):
   u,v=y+k*dy,x+k*dx
   if 0<=u<N and 0<=v<N:m[u,v]=1
 return m
def main():
 r=np.random.default_rng(SEED);rows=[]
 for p in GRID:
  a=b=0
  for _ in range(REPS):
   a+=spans(r.random((N,N))<p);b+=spans(track(float(p),r))
  rows.append(dict(p=round(float(p),3),iid_failures=a,track_failures=b,trials=REPS,iid_rate=a/REPS,track_rate=b/REPS))
 df=pd.DataFrame(rows);df.to_csv(R/"spanning_sweep.csv",index=False)
 fig,ax=plt.subplots(figsize=(7.2,4.4))
 ax.plot(df.p,df.iid_rate,marker="o",label="IID erasures");ax.plot(df.p,df.track_rate,marker="s",label="Track-burst erasures")
 ax.set(xlabel="Matched marginal site-erasure probability",ylabel="Spanning-failure frequency",title="Correlated tracks shift the finite-size failure curve")
 ax.set_ylim(-.02,1.02);ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
 fig.savefig(F/"figure1_spanning_threshold.pdf");fig.savefig(F/"figure1_spanning_threshold.png",dpi=180);plt.close(fig)
 m=track(.27,np.random.default_rng(SEED));fig,ax=plt.subplots(figsize=(6,5));ax.imshow(m,origin="lower")
 ax.set(title="Illustrative correlated track footprints",xlabel="Lattice site",ylabel="Lattice site");fig.tight_layout()
 fig.savefig(F/"figure2_track_realization.pdf");fig.savefig(F/"figure2_track_realization.png",dpi=180);plt.close(fig)
 report={"status":"PASS","model":"24x24 4-neighbour lattice connectivity surrogate","seed":SEED,"trials_per_probability":REPS,"results":rows,"limitations":["Not a physical stabilizer-code decoder or holographic tensor-network simulation.","Phenomenological track generator; not Geant4/PARTRAC track structure.","Synthetic outputs, not quantum-hardware or biological measurements."]}
 (R/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n")
 assert len(rows)==10
 print("PASS: 10 probabilities x 4,000 trials per channel; seed",SEED)
if __name__=="__main__":main()