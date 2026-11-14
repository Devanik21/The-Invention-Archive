#!/usr/bin/env python3
"""Step-size sweep for the allocation rate on digits (run on the test split; not validated)."""
import numpy as onp, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_alien as RA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True); y = y.astype(int)
out = {}
for eta in [0.05, 0.5]:
    RA.ETA_B = eta
    for cfg in [(1, 0, 1), (1, 1, 1)]:
        accs, dev = [], []
        for seed in range(3):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            Xtr, Xte = RA.minmax(Xtr, Xte)
            acc, mv = RA.train_alien(Xtr, ytr, Xte, yte, 16, 10, *cfg, 40, seed)
            accs.append(acc); dev.append(mv)
        out[f"eta={eta} cfg={cfg}"] = {"acc_mean": float(onp.mean(accs)), "acc_sd": float(onp.std(accs)), "max_share": float(max(dev) + 1)}
        print(eta, cfg, out[f"eta={eta} cfg={cfg}"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "eta_sweep_digits.json"), "w"), indent=2)
