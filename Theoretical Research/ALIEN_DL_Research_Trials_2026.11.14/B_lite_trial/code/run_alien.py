#!/usr/bin/env python3
"""ALIEN-DL lite experiment: eight coupling configurations (R, P, A) versus plain and
state-matched MLP baselines on small sklearn datasets. Writes results/<dataset>.json."""
import json, os, sys, time
import numpy as onp
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import alien as A

LOADERS = {"wine": load_wine, "breast_cancer": load_breast_cancer, "digits": load_digits}
CONFIGS = [(R, P, Ac) for R in (0, 1) for P in (0, 1) for Ac in (0, 1)]
LR, BATCH, ETA_B = 0.01, 64, 0.5


def minmax(Xtr, Xte):
    lo, hi = Xtr.min(0), Xtr.max(0)
    span = onp.where(hi - lo > 0, hi - lo, 1.0)
    return onp.clip((Xtr - lo) / span, 0, 1), onp.clip((Xte - lo) / span, 0, 1)


def adam_init(p):
    return {k: onp.zeros_like(v) for k, v in p.items()}, {k: onp.zeros_like(v) for k, v in p.items()}


def train_alien(Xtr, ytr, Xte, yte, H, K, R, P, Ac, epochs, seed):
    rng = onp.random.default_rng(seed)
    d = Xtr.shape[1]
    p = A.init(d, H, K, rng)
    logits = onp.zeros((d, H))
    m, v = adam_init(p)
    onehot = onp.eye(K)[ytr]
    t = 0
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            pi = A.shares(logits) if Ac else onp.ones((d, H))
            gp = A.grad_params(p, pi, Xtr[b], onehot[b], R)
            if P:
                gp = dict(gp); gp["theta"] = gp["theta"] * pi
            t += 1
            for k in p:
                m[k] = 0.9 * m[k] + 0.1 * gp[k]
                v[k] = 0.999 * v[k] + 0.001 * gp[k] ** 2
                mh = m[k] / (1 - 0.9 ** t); vh = v[k] / (1 - 0.999 ** t)
                p[k] = p[k] - LR * mh / (onp.sqrt(vh) + 1e-8)
            if Ac:
                gpi = A.grad_pi(p, pi, Xtr[b], onehot[b], R)
                gpi = gpi / (onp.mean(onp.abs(gpi)) + 1e-12)   # scale-free: unit mean magnitude
                logits = logits - ETA_B * gpi                  # exponentiated gradient on the budget simplex
    pi_final = A.shares(logits) if Ac else onp.ones((d, H))
    acc = float(onp.mean(A.predict(p, pi_final, Xte, R) == yte))
    return acc, float(onp.max(onp.abs(pi_final - 1.0)))


def run_dataset(name, seeds, epochs_alien, H=16):
    X, y = LOADERS[name](return_X_y=True)
    y = y.astype(int); K = int(y.max() + 1)
    out = {"dataset": name, "n": int(len(y)), "d": int(X.shape[1]), "K": K, "H": H,
           "epochs": epochs_alien, "alien": {}, "mlp_plain": [], "mlp_state_matched": []}
    accs_cfg = {str(c): [] for c in CONFIGS}; share_move = {str(c): [] for c in CONFIGS}
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        Xtr, Xte = minmax(Xtr, Xte)
        for c in CONFIGS:
            acc, mv = train_alien(Xtr, ytr, Xte, yte, H, K, *c, epochs_alien, seed)
            accs_cfg[str(c)].append(acc); share_move[str(c)].append(mv)
        mlp = MLPClassifier(hidden_layer_sizes=(H,), max_iter=400, random_state=seed, learning_rate_init=LR)
        mlp.fit(Xtr, ytr); out["mlp_plain"].append(float(mlp.score(Xte, yte)))
        mlp2 = MLPClassifier(hidden_layer_sizes=(2 * H,), max_iter=400, random_state=seed, learning_rate_init=LR)
        mlp2.fit(Xtr, ytr); out["mlp_state_matched"].append(float(mlp2.score(Xte, yte)))
        print(name, "seed", seed, "done", flush=True)
    out["alien"] = {k: {"acc": v, "mean": float(onp.mean(v)), "sd": float(onp.std(v)),
                        "max_share_deviation": share_move[k]} for k, v in accs_cfg.items()}
    return out


if __name__ == "__main__":
    name = sys.argv[1]; epochs = int(sys.argv[2]); nseeds = int(sys.argv[3])
    t0 = time.time()
    res = run_dataset(name, list(range(nseeds)), epochs)
    res["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"{name}.json"), "w"), indent=2)
    print(json.dumps({k: (v["mean"], v["sd"]) for k, v in res["alien"].items()}, indent=1))
    print("plain MLP", onp.mean(res["mlp_plain"]), "state-matched MLP", onp.mean(res["mlp_state_matched"]), "runtime", round(res["runtime_s"], 1))
