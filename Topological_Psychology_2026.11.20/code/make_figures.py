"""Generate the topological-psychology figures from saved results. Writes figures/ as PNG and SVG."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)
import cusp_affect as C
C1, C2, C3, C4 = "#1f4e79", "#c0504d", "#4f8f3a", "#8064a2"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 110})


def save(fig, n):
    fig.savefig(os.path.join(FIG, n + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, n + ".svg"), bbox_inches="tight")
    plt.close(fig)


L = lambda n: json.load(open(os.path.join(RES, n)))
E1, E2, E3, E4, E5 = (L("E1_topology.json"), L("E2_hysteresis.json"), L("E3_kramers.json"),
                      L("E4_deformation.json"), L("E5_regulation.json"))

# F1 potential landscapes as provocation changes (a = -1)
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
x = np.linspace(-1.6, 1.6, 400)
for i, (b_, col) in enumerate([(-0.5, C1), (0.0, C3), (0.5, C2)]):
    V = x**4 / 4 - 1.0 * x**2 / 2 - b_ * x
    ax[i].plot(x, V, color=col); ax[i].set_title(f"b = {b_:+.1f}, a = -1")
    ax[i].set_xlabel("affective intensity x"); ax[i].set_yticks([])
ax[0].set_ylabel("V(x; a, b)")
fig.tight_layout(); save(fig, "fig01_potential_landscapes")

# F2 cusp bifurcation diagram and equilibria
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
a_ = np.linspace(-2, 0, 300)
bf = C.fold_b(a_)
ax[0].plot(a_, bf, color="k", lw=1); ax[0].plot(a_, -bf, color="k", lw=1)
ax[0].fill_between(a_, -bf, bf, color=C2, alpha=0.2, label="bistable (2 minima)")
ax[0].set_xlabel("tension a"); ax[0].set_ylabel("provocation b"); ax[0].set_title("(a) Cusp bifurcation set"); ax[0].legend(fontsize=6.5)
bb = np.linspace(-0.8, 0.8, 400)
xs_s, xs_u = [], []
for b in bb:
    r, s = C.equilibria(-1.0, b)
    xs_s.append([v if st else np.nan for v, st in zip(r, s)] + [np.nan] * (3 - len(r)))
    xs_u.append([v if not st else np.nan for v, st in zip(r, s)] + [np.nan] * (3 - len(r)))
xs_s = np.array(xs_s); xs_u = np.array(xs_u)
ax[1].plot(bb, xs_s, color=C1, lw=1.2); ax[1].plot(bb, xs_u, color=C2, lw=0.9, ls="--")
ax[1].set_xlabel("provocation b (a = -1)"); ax[1].set_ylabel("equilibrium x"); ax[1].set_title("(b) Equilibria: stable (solid), saddle (dashed)")
fig.tight_layout(); save(fig, "fig02_cusp_bifurcation")

# F3 stable-state count map
fig, ax = plt.subplots(figsize=(4.2, 3.4))
aa = np.linspace(-2, 2, 161); bb2 = np.linspace(-2, 2, 161)
M = np.array([[C.n_minima(a, b) for b in bb2] for a in aa])
im = ax.imshow(M.T, origin="lower", extent=[-2, 2, -2, 2], cmap="viridis", aspect="auto")
ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_title(f"Stable-state count (agreement {E1['agreement']:.4%})")
fig.colorbar(im, ax=ax, fraction=0.046, label="minima")
fig.tight_layout(); save(fig, "fig03_minima_count_map")

# F4 hysteresis
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.axvline(E2["b_fold_analytic"], color="grey", ls=":", lw=0.9); ax.axvline(-E2["b_fold_analytic"], color="grey", ls=":", lw=0.9)
ax.axvline(E2["b_jump_up_sim"], color=C2, lw=1.0, label=f"upward jump {E2['b_jump_up_sim']:.3f}")
ax.axvline(E2["b_jump_down_sim"], color=C1, lw=1.0, label=f"downward jump {E2['b_jump_down_sim']:.3f}")
ax.set_xlabel("provocation b"); ax.set_yticks([]); ax.set_title(f"Hysteresis, a = -1: width {E2['hysteresis_width_sim']:.3f} vs analytic {E2['hysteresis_width_analytic']:.3f}")
ax.legend(fontsize=6.8); fig.tight_layout(); save(fig, "fig04_hysteresis")

# F5 Kramers scaling
fig, ax = plt.subplots(figsize=(5.2, 3.2))
invD = np.array([r["inv_D"] for r in E3["rows"]]); lnt = np.log([r["mfpt"] for r in E3["rows"]])
ax.plot(invD, lnt, "o", color=C1, label="simulated mean first-passage time")
fit = E3["fitted_slope"] * invD + (lnt.mean() - E3["fitted_slope"] * invD.mean())
ax.plot(invD, fit, color=C1, lw=0.9, label=f"fit slope {E3['fitted_slope']:.3f}")
ax.plot(invD, E3["barrier_analytic"] * invD + (lnt.mean() - E3["barrier_analytic"] * invD.mean()), color=C2, ls="--", lw=0.9, label="analytic barrier 0.25")
ax.set_xlabel("1 / D"); ax.set_ylabel("ln(mean first-passage time)"); ax.set_title("Kramers scaling inside the cusp")
ax.legend(fontsize=6.8); fig.tight_layout(); save(fig, "fig05_kramers")

# F6 deformation rate
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
sp = [r["ramp_speed"] for r in E4["rows"]]
ov = [r["overshoot_past_fold"] for r in E4["rows"]]
ok = [(s, o) for s, o in zip(sp, ov) if o == o]
ax[0].loglog([s for s, _ in ok], [o for _, o in ok], "o-", color=C1)
ax[0].set_xlabel("provocation ramp speed"); ax[0].set_ylabel("overshoot past fold b*")
ax[0].set_title("(a) Delay grows with build-up speed")
for s in [s for s, o in zip(sp, ov) if o != o]:
    ax[0].axvline(s, color=C2, ls=":", lw=0.9)
ax[0].text(0.5, 0.92, "dotted: no jump within window", transform=ax[0].transAxes, fontsize=6.5, ha="center")
ax[1].bar([str(s) for s in sp], [r["t_jump"] if r["t_jump"] == r["t_jump"] else 0 for r in E4["rows"]], color=C4)
ax[1].set_xlabel("ramp speed"); ax[1].set_ylabel("time of outburst"); ax[1].set_title("(b) Outburst time (0 = censored)")
fig.tight_layout(); save(fig, "fig06_deformation_rate")

# F7 regulation
fig, ax = plt.subplots(figsize=(4.8, 2.8))
g = [r["gamma"] for r in E5["rows"]]; f = [r["fraction_time_high_branch"] for r in E5["rows"]]
ax.bar([str(v) for v in g], f, color=C3)
ax.set_xlabel("reappraisal gain factor gamma"); ax.set_ylabel("fraction of time, high branch")
ax.set_title("Regulation as provocation gain (single run per gamma)")
fig.tight_layout(); save(fig, "fig07_regulation")
print("figures:", sorted(os.listdir(FIG)))
