"""Figures and LaTeX tables, generated only from the saved results (no hand-typed numbers)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .baselines import ClippedOLS, OLS, make_quadratic  # noqa: E402
from .data import REAL, TOYS, load_real, real_split, toy_split  # noqa: E402
from .model import ThinkingLinearRegression as TLR  # noqa: E402

PRIMARY = "TLR (extrap, tuned)"
COLORS = {
    "OLS": "#7f7f7f", "Clipped OLS": "#bcbd22", "Huber": "#17becf", "Quadratic": "#9467bd",
    "kNN (k=10)": "#8c564b", "GPR": "#e377c2", PRIMARY: "#d62728",
    "TLR (residual, tuned)": "#ff9896", "TLR one-shot (T=1)": "#ffbb78",
    "TLR binary critic": "#2ca02c",
}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.svg")
    plt.close(fig)


def _prm(params, dataset, method=PRIMARY, seed=0):
    row = params[(params.dataset == dataset) & (params.method == method) & (params.seed == seed)].iloc[0]
    return dict(thought_steps=int(row.thought_steps), eta=float(row.eta),
                kappa=float(row.kappa), gamma=float(row.gamma))


class TLRSpec:
    def __init__(self, name):
        self.f = TOYS[name]["f"]

    def __call__(self, x):
        return self.f(np.asarray(x, float))


def fig_mechanism(out, params):
    """Figure 1: the loop on one extrapolated input, plus the anchor and critic."""
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    split = toy_split("saturating", 0)
    p = _prm(params, "saturating")
    m = TLR(anchor="extrap", **p).fit(split["xtr"], split["ytr"])
    xs = np.linspace(0, 16, 300)
    ols = OLS().fit(split["xtr"], split["ytr"])
    ax[0].scatter(split["xtr"], split["ytr"], s=7, c="#1f77b4", alpha=.45, label="training data")
    ax[0].plot(xs, TLRSpec("saturating")(xs), color="k", lw=1.2, label="true mean f(x)")
    ax[0].plot(xs, ols.predict(xs), "--", color=COLORS["OLS"], label="OLS (System 1)")
    ax[0].plot(xs, m.predict(xs), color=COLORS[PRIMARY], lw=1.8, label=f"TLR, T={p['thought_steps']}")
    ax[0].axvspan(split["xtr"].min(), split["xtr"].max(), color="0.93", zorder=0)
    ax[0].set(xlabel="x", ylabel="y", title="(a) Training support (shaded) and extrapolation")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    x0 = 13.0
    traj, A, v, a = m.trace(x0)
    ax[1].plot(range(len(traj)), traj, "o-", color=COLORS[PRIMARY], ms=3, label=r"thought $y_t$")
    ax[1].axhline(A, ls=":", color="k", label=r"anchor $A(x)$")
    ax[1].axhline(ols.predict(np.array([x0]))[0], ls="--", color=COLORS["OLS"], label="OLS $y_0$")
    ax[1].set(xlabel="thought step t", ylabel="prediction",
              title=f"(b) Trajectory at x={x0:g}  (v={v:.2f}, gain a={a:.2f})")
    ax[1].legend(frameon=False, fontsize=7)
    _save(fig, out, "fig1_mechanism")


def fig_schematic(out):
    """Figure 2: the thinking loop as a flow diagram."""
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    boxes = [
        (1, 12, 14, 12, "input x*"),
        (19, 12, 17, 12, "System 1\ny0 = w x* + b"),
        (40, 12, 17, 12, "Critic\nv = distance outside\nsupport;  a = 1-exp(-kv)"),
        (61, 12, 17, 12, "Anchor\nA(x*) from boundary\ndata (fixed per x*)"),
        (83, 12, 16, 12, "Thought\ny <- y + eta a (A - y)\nrepeat T times"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#fdf1f0" if "Thought" in t else "#f4f4f4",
                                   ec="#444", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.4)
    for x0, x1 in [(15, 19), (36, 40), (57, 61), (78, 83)]:
        ax.annotate("", xy=(x1, 18), xytext=(x0, 18), arrowprops=dict(arrowstyle="->", lw=.9))
    ax.annotate("", xy=(91, 6.5), xytext=(91, 12), arrowprops=dict(arrowstyle="->", lw=.9))
    ax.text(50, 3, "output y_T  (closed form:  y_T = A + (1 - eta a)^T (y0 - A))",
            ha="center", fontsize=8, style="italic")
    ax.text(50, 27.5, "Thinking Linear Regression: the loop changes nothing inside the data",
            ha="center", fontsize=8.5, weight="bold")
    _save(fig, out, "fig2_schematic")


def fig_benchmark(out, main):
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.2))
    datasets = list(TOYS) + list(REAL)
    for ax, ds in zip(axes.ravel(), datasets):
        sub = main[(main.dataset == ds)]
        g = sub.groupby("method")["rmse_ext"]
        mu, sd, n = g.mean(), g.std(), g.count()
        order = mu.sort_values().index
        yy = np.arange(len(order))
        ax.barh(yy, mu[order], xerr=1.96 * sd[order] / np.sqrt(n[order]),
                color=[COLORS.get(m, "#999") for m in order], capsize=2, height=.7)
        ax.set_yticks(yy); ax.set_yticklabels(order, fontsize=6.3)
        title = TOYS[ds]["title"] if ds in TOYS else ("Engel curve (real)" if ds == "engel" else "Diabetes BMI (real)")
        ax.set_title(title, fontsize=8); ax.set_xlabel("extrapolation RMSE", fontsize=7)
        ax.invert_yaxis()
    axes.ravel()[-1].axis("off")
    fig.suptitle("Extrapolation error by method (mean and 95% CI over 30 seeds; lower is better)", fontsize=9)
    _save(fig, out, "fig3_benchmark")


def fig_forest(out, stats):
    s = stats[(stats.metric == "rmse_ext")]
    datasets = list(TOYS) + list(REAL)
    fig, axes = plt.subplots(1, len(datasets), figsize=(12.5, 3.6), sharey=True)
    methods = [m for m in COLORS if m != "OLS"]
    for ax, ds in zip(axes, datasets):
        sub = s[s.dataset == ds].set_index("method").reindex(methods).dropna(how="all")
        for i, (m, r) in enumerate(sub.iterrows()):
            c = COLORS[m]
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=1.5)
            ax.plot(r.mean_diff, i, "o", color=c, ms=3.5)
        ax.axvline(0, color="k", lw=.7)
        ax.set_yticks(range(len(sub.index))); ax.set_yticklabels(sub.index, fontsize=6.5)
        ax.invert_yaxis(); ax.set_title(ds, fontsize=8); ax.set_xlabel("RMSE minus OLS", fontsize=7)
    fig.suptitle("Paired difference from OLS on extrapolation (left of zero = better than OLS); 95% bootstrap CI",
                 fontsize=9)
    _save(fig, out, "fig4_forest_vs_ols")


def fig_budget(out, curves):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for ds in list(TOYS) + list(REAL):
        c = curves[curves.dataset == ds]
        ax.plot(np.maximum(c["T"], 0.5), c["mean"], "o-", ms=3, lw=1.1, label=ds)
    ax.set_xscale("log"); ax.set_xlabel("thinking budget T (log scale; T=0 plotted at 0.5)")
    ax.set_ylabel("extrapolation RMSE (mean over seeds)")
    ax.set_title("Accuracy versus thinking budget (tuned gamma, kappa, eta per seed)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    _save(fig, out, "fig5_budget_curve")


def fig_heatmap(out, hm):
    piv = hm.pivot(index="gamma", columns="kappa", values="rmse_ext")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(piv.values, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    ax.set_xlabel("critic sharpness kappa"); ax.set_ylabel("slope carried into anchor gamma")
    fig.colorbar(im, ax=ax, shrink=.85, label="extrapolation RMSE")
    ax.set_title("Saturating toy, eta=0.5, T=10: sensitivity map", fontsize=8.5)
    _save(fig, out, "fig6_heatmap_gamma_kappa")


def fig_regime(out, params):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, ds, title in [(ax[0], "regime_shift", "(a) Regime shift toy (Zillow-style)"),
                         (ax[1], "engel", "(b) Engel curve (real): income extrapolation")]:
        if ds == "regime_shift":
            split = toy_split(ds, 0)
            xs = np.linspace(TOYS[ds]["train"][0], TOYS[ds]["ext"][1], 300)
            truth = TOYS[ds]["f"](xs)
        else:
            split = real_split(ds, 0)
            xs = np.linspace(split["xtr"].min(), split["xext"].max(), 300)
            truth = None
        p = _prm(params, ds)
        tlr = TLR(anchor="extrap", **p).fit(split["xtr"], split["ytr"])
        ols = OLS().fit(split["xtr"], split["ytr"])
        clip = ClippedOLS().fit(split["xtr"], split["ytr"])
        quad = make_quadratic().fit(split["xtr"], split["ytr"])
        a.scatter(split["xtr"], split["ytr"], s=6, c="#1f77b4", alpha=.4, label="train (support)")
        a.scatter(split["xext"], split["yext"], s=6, c="#ff7f0e", alpha=.5, label="extrapolation")
        if truth is not None:
            a.plot(xs, truth, "k", lw=1.1, label="truth")
        a.plot(xs, ols.predict(xs), "--", color=COLORS["OLS"], label="OLS")
        a.plot(xs, clip.predict(xs), ":", color=COLORS["Clipped OLS"], lw=1.4, label="Clipped OLS")
        a.plot(xs, quad.predict(xs), "-.", color=COLORS["Quadratic"], lw=1, label="Quadratic")
        a.plot(xs, tlr.predict(xs), color=COLORS[PRIMARY], lw=1.8, label="TLR")
        a.set_title(title, fontsize=8.5); a.set_xlabel("x"); a.set_ylabel("y")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    _save(fig, out, "fig7_shift_cases")


def fig_inside_identity(out, main):
    sub = main[main.method == PRIMARY]
    g = sub.groupby("dataset")["max_abs_in_diff_vs_ols"].max()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(range(len(g)), np.maximum(g.values, 1e-16), color="#d62728")
    ax.set_yscale("log"); ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=25, ha="right", fontsize=7)
    ax.axhline(1e-10, color="k", ls=":", lw=.8)
    ax.set_ylabel("max |TLR - OLS| inside support")
    ax.set_title("In-support identity (dotted: numerical tolerance 1e-10)", fontsize=8.5)
    _save(fig, out, "fig8_inside_identity")


# ---------------------------------------------------------------- tables
def _fmt(m, s, d=2):
    return f"{m:.{d}f}\\,{{\\scriptsize$\\pm${s:.{d}f}}}"


def table_main(out, main):
    rows = []
    for ds in list(TOYS) + list(REAL):
        sub = main[main.dataset == ds].groupby("method")["rmse_ext"].agg(["mean", "std"])
        for m in sub.sort_values("mean").index:
            rows.append(f"{ds.replace('_', chr(92)+'_')} & {m.replace('&','\\&')} & "
                        f"{_fmt(sub.loc[m,'mean'], sub.loc[m,'std'])} \\\\")
    body = "\n".join(rows)
    tex = ("\\begin{longtable}{llr}\n\\caption{Extrapolation RMSE by method, mean and SD over 30 seeds. Lower is better.}\\label{tab:main}\\\\\n\\toprule\nDataset & Method & Extrapolation RMSE \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{longtable}\n")
    (out / "tab_main.tex").write_text(tex)


def table_stats(out, stats):
    s = stats[stats.metric == "rmse_ext"]
    rows = []
    for _, r in s.iterrows():
        rows.append(f"{r.dataset.replace('_', chr(92)+'_')} & {r.method.replace('&','\\&')} & "
                    f"{r.mean_diff:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & "
                    f"{r.rel_change_pct:+.1f}\\% & {100*r.win_rate:.0f}\\% & {r.p_holm:.3g} \\\\")
    tex = ("\\begin{longtable}{llrrrrr}\n\\toprule\nDataset & Method & Mean diff. vs OLS & 95\\% CI & Rel. & Win & Holm $p$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
    (out / "tab_stats.tex").write_text(tex)


def table_params(out, params):
    rows = []
    for ds in list(TOYS) + list(REAL):
        sub = params[(params.dataset == ds) & (params.method == PRIMARY)]

        def mode(c):
            return sub[c].astype(str).value_counts().index[0]
        rows.append(f"{ds.replace('_', chr(92)+'_')} & {mode('thought_steps')} & {mode('eta')} & "
                    f"{mode('kappa')} & {mode('gamma')} \\\\")
    tex = ("\\begin{tabular}{lrrrr}\n\\toprule\nDataset & Modal $T$ & Modal $\\eta$ & Modal $\\kappa$ & Modal $\\gamma$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_params.tex").write_text(tex)


def make_all(root):
    root = Path(root)
    res, figs, tabs = root / "results", root / "figures", root / "paper" / "tables"
    figs.mkdir(exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(res / "main_results.csv")
    params = pd.read_csv(res / "tuned_params.csv")
    stats = pd.read_csv(res / "statistics.csv")
    curves = pd.read_csv(res / "budget_curves.csv")
    hm = pd.read_csv(res / "heatmap_gamma_kappa.csv")
    fig_schematic(figs)
    fig_mechanism(figs, params)
    fig_benchmark(figs, main)
    fig_forest(figs, stats)
    fig_budget(figs, curves)
    fig_heatmap(figs, hm)
    fig_regime(figs, params)
    fig_inside_identity(figs, main)
    table_main(tabs, main)
    table_stats(tabs, stats)
    table_params(tabs, params)
    return json.loads((res / "timing.json").read_text())
