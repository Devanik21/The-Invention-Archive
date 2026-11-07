"""Experimental protocols: nested tuning, benchmarks, statistics, budget curves, timing."""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .baselines import ClippedOLS, OLS, make_gpr, make_huber, make_knn, make_quadratic
from .data import REAL, TOYS, real_split, toy_split
from .model import ThinkingLinearRegression as TLR

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
KAPPAS = [0.5, 1.0, 2.0, 4.0]
ETAS = [0.25, 0.5, 1.0]
TS = [1, 3, 10, 30]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def tune_tlr(x, y, anchor, kappas=KAPPAS, Ts=TS):
    """Nested tuning on a simulated shift inside the training data: fit on the lower
    80% of x, score on the top 20%, which is genuine extrapolation. No test data used."""
    cut = np.quantile(x, 0.8)
    tr, va = x <= cut, x > cut
    best_err, best = np.inf, None
    for g, k, e, T in itertools.product(GAMMAS, kappas, ETAS, Ts):
        m = TLR(thought_steps=T, eta=e, kappa=k, gamma=g, anchor=anchor).fit(x[tr], y[tr])
        err = rmse(m.predict(x[va]), y[va])
        if err < best_err:
            best_err, best = err, dict(gamma=g, kappa=k, eta=e, thought_steps=T)
    return best


def method_predictions(split):
    """Fit every method on the training split. Returns {name: (pred_in, pred_ext, params)}."""
    xtr, ytr = split["xtr"], split["ytr"]
    out = {}

    def run(name, model, params=None):
        model.fit(xtr, ytr)
        out[name] = (model.predict(split["xin"]), model.predict(split["xext"]), params)

    run("OLS", OLS())
    run("Clipped OLS", ClippedOLS())
    run("Huber", make_huber())
    run("Quadratic", make_quadratic())
    run("kNN (k=10)", make_knn(10))
    run("GPR", make_gpr())
    specs = [
        ("TLR (extrap, tuned)", "extrap", KAPPAS, TS),
        ("TLR (residual, tuned)", "residual", KAPPAS, TS),
        ("TLR one-shot (T=1)", "extrap", KAPPAS, [1]),
        ("TLR binary critic", "extrap", [np.inf], TS),
    ]
    for label, anchor, kappas, Ts in specs:
        p = tune_tlr(xtr, ytr, anchor, kappas=kappas, Ts=Ts)
        run(label, TLR(anchor=anchor, **p), params=p)
    return out


def benchmark(kind, names, n_seeds):
    rows, params_rows = [], []
    for name in names:
        for seed in range(n_seeds):
            split = toy_split(name, seed) if kind == "toy" else real_split(name, seed)
            preds = method_predictions(split)
            ols_in = preds["OLS"][0]
            # Points inside the *realised* training range: the in-support identity
            # (TLR == OLS) must hold on exactly these points.
            lo_r, hi_r = split["xtr"].min(), split["xtr"].max()
            inside = (split["xin"] >= lo_r) & (split["xin"] <= hi_r)
            frac_inside = float(inside.mean())
            for method, (p_in, p_ext, params) in preds.items():
                diff = float(np.max(np.abs(p_in[inside] - ols_in[inside]))) if inside.any() else 0.0
                rows.append(dict(dataset=name, kind=kind, seed=seed, method=method,
                                 rmse_in=rmse(p_in, split["yin"]),
                                 rmse_ext=rmse(p_ext, split["yext"]),
                                 frac_in_realised_support=frac_inside,
                                 max_abs_in_diff_vs_ols=diff))
                if params is not None:
                    params_rows.append(dict(dataset=name, kind=kind, seed=seed,
                                            method=method, **params))
    return pd.DataFrame(rows), pd.DataFrame(params_rows)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare_to_ols(df, metric="rmse_ext", base="OLS", B=10000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for name, sub in df.groupby("dataset"):
        piv = sub.pivot(index="seed", columns="method", values=metric)
        base_mean = piv[base].mean()
        rows = []
        for m in piv.columns:
            if m == base:
                continue
            d = (piv[m] - piv[base]).to_numpy()
            idx = rng.integers(0, d.size, (B, d.size))
            ci = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            try:
                p = float(stats.wilcoxon(d).pvalue)
            except ValueError:
                p = 1.0
            rows.append(dict(dataset=name, method=m, metric=metric,
                             mean_base=base_mean, mean_method=piv[m].mean(),
                             mean_diff=d.mean(), ci_lo=ci[0], ci_hi=ci[1],
                             rel_change_pct=100 * d.mean() / base_mean,
                             win_rate=float((d < 0).mean()), p_wilcoxon=p))
        if rows:
            for r, a in zip(rows, holm([r["p_wilcoxon"] for r in rows])):
                r["p_holm"] = float(a)
            out.extend(rows)
    return pd.DataFrame(out)


def budget_curve(params, kind, name, Ts=(0, 1, 2, 3, 5, 10, 20, 50, 100)):
    res = []
    sub = params[(params.dataset == name) & (params.method == "TLR (extrap, tuned)")]
    for _, row in sub.iterrows():
        seed = int(row.seed)
        split = toy_split(name, seed) if kind == "toy" else real_split(name, seed)
        for T in Ts:
            m = TLR(thought_steps=T, eta=row.eta, kappa=row.kappa, gamma=row.gamma,
                    anchor="extrap").fit(split["xtr"], split["ytr"])
            res.append(dict(dataset=name, seed=seed, T=T,
                            rmse_ext=rmse(m.predict(split["xext"]), split["yext"])))
    d = pd.DataFrame(res)
    return d.groupby(["dataset", "T"])["rmse_ext"].agg(["mean", "std"]).reset_index()


def gamma_kappa_heatmap(name="saturating", n_seeds=30, eta=0.5, T=10):
    rows = []
    for seed in range(n_seeds):
        split = toy_split(name, seed)
        for g, k in itertools.product(GAMMAS, KAPPAS):
            m = TLR(thought_steps=T, eta=eta, kappa=k, gamma=g, anchor="extrap")
            m.fit(split["xtr"], split["ytr"])
            rows.append(dict(seed=seed, gamma=g, kappa=k,
                             rmse_ext=rmse(m.predict(split["xext"]), split["yext"])))
    return pd.DataFrame(rows).groupby(["gamma", "kappa"])["rmse_ext"].mean().reset_index()


def timing(n_points=20000, T=10):
    x = np.linspace(0, 8, 200)
    m = TLR(thought_steps=T, eta=0.5, kappa=1.0, gamma=0.5).fit(x, 2 * x + 1)
    xs = np.linspace(0, 16, n_points)
    t0 = time.perf_counter(); m.predict(xs); closed = time.perf_counter() - t0
    t0 = time.perf_counter(); m.predict_iterative(xs); loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    for v in xs[:2000]:
        m.trace(v)
    per_trace = (time.perf_counter() - t0) / 2000
    return dict(n_points=n_points, T=T, closed_form_seconds=closed,
                iterative_seconds=loop, per_point_trace_seconds=per_trace)


def run_all(n_seeds=30, out_dir="results"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    toy_df, toy_p = benchmark("toy", list(TOYS), n_seeds)
    real_df, real_p = benchmark("real", list(REAL), n_seeds)
    main = pd.concat([toy_df, real_df], ignore_index=True)
    params = pd.concat([toy_p, real_p], ignore_index=True)
    main.to_csv(out / "main_results.csv", index=False)
    params.to_csv(out / "tuned_params.csv", index=False)
    pd.concat([compare_to_ols(main, "rmse_ext"), compare_to_ols(main, "rmse_in")],
              ignore_index=True).to_csv(out / "statistics.csv", index=False)
    curves = [budget_curve(params, "toy", n) for n in TOYS]
    curves += [budget_curve(params, "real", n) for n in REAL]
    pd.concat(curves, ignore_index=True).to_csv(out / "budget_curves.csv", index=False)
    gamma_kappa_heatmap(n_seeds=n_seeds).to_csv(out / "heatmap_gamma_kappa.csv", index=False)
    json.dump(timing(), open(out / "timing.json", "w"), indent=2)
    return main, params


if __name__ == "__main__":
    run_all()
