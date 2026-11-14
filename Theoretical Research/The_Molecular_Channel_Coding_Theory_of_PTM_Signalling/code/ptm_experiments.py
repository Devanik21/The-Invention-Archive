#!/usr/bin/env python3
"""PTM channel experiments on a 4-site proteoform system (16 inputs, 16 outputs).
E2 capacity vs observation time and turnover; E3 crosstalk; E4 dissipation-capacity frontier;
E5 codebook error versus minimum distance; E6 learned neural decoder versus exact Bayes limit.
Seed 20261112. Writes results/*.json."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
sys.path.insert(0, HERE)
import ptm_channel as P

N_SITES = 4
ALPHA, BETA = 1.0, 0.5
INPUTS = P.all_inputs(N_SITES)
rng = np.random.default_rng(20261112)


def cap_at(t, kappa=1.0, c=0.0):
    W = P.channel_matrix(N_SITES, INPUTS, ALPHA, BETA, kappa, c, t)
    C, p = P.blahut_arimoto(W)
    return C, W


def E2_time_turnover():
    ts = np.logspace(-1, 1.2, 9)
    out = {"t": ts.tolist(), "kappa": [0.5, 1.0, 2.0], "C": {}}
    for k in out["kappa"]:
        out["C"][str(k)] = [cap_at(t, kappa=k)[0] for t in ts]
    out["C_max_bits"] = float(np.log2(len(INPUTS)))
    return out


def E3_crosstalk():
    cs = [0.0, 0.5, 1.0, 2.0]
    out = {"c": cs, "C": [], "t": 2.0}
    for c in cs:
        out["C"].append(cap_at(2.0, kappa=1.0, c=c)[0])
    return out


def E4_energy_frontier():
    kappas = [0.25, 0.5, 1.0, 2.0, 4.0]
    ts = [0.5, 1.0, 2.0, 4.0]
    rows = []
    for t in ts:
        for k in kappas:
            C, W = cap_at(t, kappa=k)
            cyc = np.mean([P.expected_cycles(N_SITES, s, ALPHA, BETA, k, 0.0, t, steps=200) for s in INPUTS])
            rows.append({"t": t, "kappa": k, "C": C, "cycles_mean": float(cyc),
                         "C_per_cycle": float(C / cyc) if cyc > 0 else None})
    return {"rows": rows}


def E5_codes():
    W = P.channel_matrix(N_SITES, INPUTS, ALPHA, BETA, 1.0, 0.0, 2.0)
    W1 = P.channel_matrix(N_SITES, INPUTS, ALPHA, BETA, 1.0, 1.0, 2.0)
    out = {"rows": []}
    for name, book in [("uncoded M=2", [1, 2]), ("coded M=2", [0, 15]),
                       ("uncoded M=4", [1, 2, 4, 8]), ("coded M=4", [0, 3, 5, 6]),
                       ("coded M=4 alt", [0, 7, 11, 13])]:
        d = P.min_distance(book)
        out["rows"].append({"name": name, "M": len(book), "d": int(d) if d != np.inf else None,
                            "Pe_c0": P.map_error(W, book), "Pe_c1": P.map_error(W1, book)})
    return out


def train_mlp(X, y, n_classes, hidden=32, epochs=400, lr=0.05):
    d = X.shape[1]
    W1 = rng.normal(0, 0.3, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.3, (hidden, n_classes)); b2 = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    for _ in range(epochs):
        h = np.tanh(X @ W1 + b1)
        z = h @ W2 + b2
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = (p - Y) / len(y)
        gW2 = h.T @ g; gb2 = g.sum(0)
        gh = (g @ W2.T) * (1 - h ** 2)
        gW1 = X.T @ gh; gb1 = gh.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return (W1, b1, W2, b2)


def predict(model, X):
    W1, b1, W2, b2 = model
    z = np.tanh(X @ W1 + b1) @ W2 + b2
    return z.argmax(1)


def E6_learned_decoder():
    W = P.channel_matrix(N_SITES, INPUTS, ALPHA, BETA, 1.0, 0.0, 2.0)
    bayes = float(np.sum(np.max(W, axis=0)) / len(INPUTS))
    out = {"bayes_accuracy_uniform_prior": bayes, "rows": []}
    M = len(INPUTS)
    for N in [200, 1000, 5000, 20000]:
        x = rng.integers(0, M, N)
        y = np.array([rng.choice(W.shape[1], p=W[xi]) for xi in x])
        X = np.eye(W.shape[1])[y]
        xt = rng.integers(0, M, 4000)
        yt = np.array([rng.choice(W.shape[1], p=W[xi]) for xi in xt])
        Xt = np.eye(W.shape[1])[yt]
        accs = []
        for seed in range(3):
            model = train_mlp(X, x, M)
            accs.append(float(np.mean(predict(model, Xt) == xt)))
        out["rows"].append({"N": N, "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs))})
    return out


def E1_validation():
    a, b, t = 1.3, 0.7, 0.9
    W = P.channel_matrix(1, [np.array([1.0])], a, b, 1.0, 0.0, t)
    ana = a / (a + b) * (1 - np.exp(-(a + b) * t))
    W1 = P.channel_matrix(1, [np.array([0.0]), np.array([1.0])], a, b, 1.0, 0.0, t)
    q = W1[1, 1]; p = 1 - q
    C_closed = float(np.log2(1 + (1 - p) * p ** (p / (1 - p))))
    C_ba = P.blahut_arimoto(W1)[0]
    return {"two_state_cme": float(W[0, 1]), "two_state_analytic": float(ana),
            "z_channel_BA_bits": float(C_ba), "z_channel_closed_bits": C_closed}


def main():
    res = {"E1": E1_validation()}
    json.dump(res["E1"], open(os.path.join(RES, "E1_validation.json"), "w"), indent=2)
    res["E2"] = E2_time_turnover(); json.dump(res["E2"], open(os.path.join(RES, "E2_time_turnover.json"), "w"), indent=2)
    res["E3"] = E3_crosstalk(); json.dump(res["E3"], open(os.path.join(RES, "E3_crosstalk.json"), "w"), indent=2)
    res["E4"] = E4_energy_frontier(); json.dump(res["E4"], open(os.path.join(RES, "E4_energy_frontier.json"), "w"), indent=2)
    res["E5"] = E5_codes(); json.dump(res["E5"], open(os.path.join(RES, "E5_codes.json"), "w"), indent=2)
    res["E6"] = E6_learned_decoder(); json.dump(res["E6"], open(os.path.join(RES, "E6_learned_decoder.json"), "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
