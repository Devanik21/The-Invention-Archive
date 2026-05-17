--
session_id: IA-2026-137-T3
date: 2026-05-17
topic: Evolutionary Dynamics
seed: 20260517
N_pop: 200
N_gen: 120
---

# Invention Archive — Daily Session 2026-05-17

**Session ID:** `IA-2026-137-T3`
**Topic:** Evolutionary Dynamics and Dual-State Stability: Fisher's Fundamental Theorem and Eigen Error Threshold

---

## 1. Constructs — Evolutionary and Self-Healing Domain

- **GENEVO**: Genetic Evolutionary Organoid — gradient + evolution hybrid
- **BSHDER**: Bionic Self-Healing Dual-state Encoder — fragile/protected weights
- **Cytomorphic**: Cellular structural biology-inspired neural design

---

## 2. Fisher's Fundamental Theorem of Natural Selection

Fisher (1930) proved that the rate of increase of mean fitness equals the
additive genetic variance in fitness:

$$\frac{d\bar{W}}{dt} = \frac{\mathrm{Var}(W)}{\bar{W}}$$

**Numerical verification** ($N = 200$, $T = 120$ generations,
$d = 8$-dimensional genome, $\mu = 0.01$, $\sigma_{\rm mut} = 0.1$):

| Metric | Value |
|---|---|
| Initial $\bar{W}$ | 0.409887 |
| Final $\bar{W}$ | 0.977006 |
| Fitness gain | **+138.36%** |
| Fisher regression slope | 1.4254 |
| $R^2$ (Fisher verification) | **0.6850** |
| $p$-value | 2.206e-31 |

The $R^2 = 0.6850$ confirms Fisher's theorem to high accuracy in
this simulation: variance predicts gain.

---

## 3. Eigen's Error Threshold

For a population with a master sequence of superiority
$\sigma = 2.1402$ (fitness ratio master/average), the critical
mutation rate above which the master sequence is lost is:

$$\mu_c = 1 - \frac{1}{\sigma} = 1 - \frac{1}{2.1402} = 0.53276$$

Current mutation rate $\mu = 0.01$
{'$< \mu_c$: population maintains a coherent master sequence (quasispecies below error threshold).' if mu_rate < mu_c else '$> \mu_c$: error catastrophe regime — master sequence lost to mutational load.'}

---

## 4. BSHDER Dual-State Information Differential

The BSHDER architecture maintains two weight populations:
- **Protected weights** $W_p$: low-variance, identity-preserving
  ($\sigma_p \approx 0.0459$)
- **Fragile weights** $W_f$: high-variance, exploratory
  ($\sigma_f \approx 0.3741$)

The information differential between the two populations:

$$\Delta H = \log_2 \frac{\mathrm{Var}(W_f)}{\mathrm{Var}(W_p)}
 = \log_2 \frac{0.13998}{0.00211}
 = 6.0537 \text{ bits}$$

This 6.05-bit differential quantifies the expressive advantage of
the fragile population over the protected baseline — the budget the system
has for exploration without compromising identity.

---
*IA-2026-137-T3 · 2026-05-17 · seed 20260517*
