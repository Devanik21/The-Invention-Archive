#!/usr/bin/env python3
"""Reproduce the finite-lattice iid-versus-track-burst benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
R=ROOT/"results"; F=ROOT/"figures"
R.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
N=24; SEED=20261106; REPS=4000
GRID=np.arange(.20,.651,.05)
DIRS=((1,0),(0,1),(1,1),(1,-1))

def site_hit_probabilities():
    """Probability that one uniformly placed/oriented track hits each site."""
    counts=np.zeros((N,N),dtype=np.int64)
    placements=N*N*len(DIRS)
    for y in range(N):
        for x in range(N):
            for dy,dx in DIRS:
                for k in range(-2,3):
                    u,v=y+k*dy,x+k*dx
                    if 0<=u<N and 0<=v<N:
                        counts[u,v]+=1
    return counts/placements

HIT_PROB=site_hit_probabilities()

def expected_mean_occupancy(intensity):
    # Poisson thinning: P(site i is hit) = 1 - exp(-lambda * q_i).
    return float(np.mean(-np.expm1(-float(intensity)*HIT_PROB)))

def calibrated_intensity(p):
    """Solve E[mean occupied sites] = p, including finite-edge clipping."""
    lo,hi=0.0,1.0
    while expected_mean_occupancy(hi)<p:
        hi*=2.0
    for _ in range(80):
        mid=(lo+hi)/2.0
        if expected_mean_occupancy(mid)<p:
            lo=mid
        else:
            hi=mid
    value=(lo+hi)/2.0
    if abs(expected_mean_occupancy(value)-p)>1e-10:
        raise RuntimeError(f"marginal calibration failed at p={p}")
    return value

LAMBDA_BY_P={round(float(p),3):calibrated_intensity(float(p)) for p in GRID}

def boundary_spans(mask,axis):
    """Test reachability from one boundary to its opposite in a 4-neighbour graph."""
    seen=np.zeros_like(mask,dtype=bool)
    if axis==0:  # left to right
        stack=[(y,0) for y in range(N) if mask[y,0]]
        target=lambda y,x: x==N-1
    else:  # top to bottom
        stack=[(0,x) for x in range(N) if mask[0,x]]
        target=lambda y,x: y==N-1
    for y,x in stack:
        seen[y,x]=True
    while stack:
        y,x=stack.pop()
        if target(y,x):
            return True
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            u,v=y+dy,x+dx
            if 0<=u<N and 0<=v<N and mask[u,v] and not seen[u,v]:
                seen[u,v]=True
                stack.append((u,v))
    return False

def spans(mask):
    return boundary_spans(mask,0) or boundary_spans(mask,1)

def track(p,rng):
    """Poisson superposition of finite five-site line tracks."""
    mask=np.zeros((N,N),bool)
    p_key=round(float(p),3)
    intensity=LAMBDA_BY_P[p_key] if p_key in LAMBDA_BY_P else calibrated_intensity(float(p))
    for _ in range(rng.poisson(intensity)):
        y=int(rng.integers(N)); x=int(rng.integers(N))
        dy,dx=DIRS[int(rng.integers(len(DIRS)))]
        for k in range(-2,3):
            u,v=y+k*dy,x+k*dx
            if 0<=u<N and 0<=v<N:
                mask[u,v]=True
    return mask

def main():
    rng=np.random.default_rng(SEED)
    rows=[]
    for p in GRID:
        a=b=0
        for _ in range(REPS):
            a+=spans(rng.random((N,N))<p)
            b+=spans(track(float(p),rng))
        p_key=round(float(p),3)
        rows.append({
            "p":p_key,
            "iid_failures":int(a),
            "track_failures":int(b),
            "trials":REPS,
            "iid_rate":a/REPS,
            "track_rate":b/REPS,
            "track_poisson_intensity":LAMBDA_BY_P[p_key],
            "target_mean_occupancy":p_key,
        })
    df=pd.DataFrame(rows)
    df.to_csv(R/"spanning_sweep.csv",index=False)

    fig,ax=plt.subplots(figsize=(7.2,4.4))
    ax.plot(df.p,df.iid_rate,marker="o",label="IID erasures")
    ax.plot(df.p,df.track_rate,marker="s",label="Track-burst erasures")
    ax.set(xlabel="Lattice-wide mean site-erasure probability",
           ylabel="Spanning-failure frequency",
           title="Correlated tracks shift the finite-size failure curve")
    ax.set_ylim(-.02,1.02); ax.grid(True,alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(F/"figure1_spanning_threshold.pdf")
    fig.savefig(F/"figure1_spanning_threshold.png",dpi=180)
    plt.close(fig)

    m=track(.27,np.random.default_rng(SEED))
    fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(m,origin="lower")
    ax.set(title="Illustrative correlated track footprints",
           xlabel="Lattice site",ylabel="Lattice site")
    fig.tight_layout()
    fig.savefig(F/"figure2_track_realization.pdf")
    fig.savefig(F/"figure2_track_realization.png",dpi=180)
    plt.close(fig)

    occupancy_checks={
        f"{p:.2f}":expected_mean_occupancy(LAMBDA_BY_P[round(float(p),3)])
        for p in GRID
    }
    report={
        "status":"PASS",
        "model":"24x24 4-neighbour lattice connectivity surrogate",
        "seed":SEED,
        "trials_per_probability":REPS,
        "calibration":{
            "method":"bisection on the exact Poisson-thinning expectation",
            "target":"lattice-wide mean occupancy equals p",
            "sitewise_note":"individual site marginals vary near boundaries because tracks are clipped",
            "poisson_intensity_by_p":{
                f"{p:.2f}":LAMBDA_BY_P[round(float(p),3)] for p in GRID
            },
            "expected_mean_occupancy_by_p":occupancy_checks,
        },
        "results":rows,
        "limitations":[
            "Not a physical stabilizer-code decoder or holographic tensor-network simulation.",
            "Phenomenological track generator; not Geant4/PARTRAC track structure.",
            "Synthetic outputs, not quantum-hardware or biological measurements.",
        ],
    }
    (R/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n")
    assert len(rows)==10
    assert all(abs(occupancy_checks[f"{p:.2f}"]-float(p))<1e-10 for p in GRID)
    print("PASS: 10 probabilities x 4,000 trials per channel; exact lattice-mean occupancy calibration; seed",SEED)

if __name__=="__main__":
    main()
