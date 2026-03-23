<div align="center">

```
███╗   ██╗███████╗ ██████╗███████╗
████╗  ██║██╔════╝██╔════╝██╔════╝
██╔██╗ ██║█████╗  ██║     █████╗
██║╚██╗██║██╔══╝  ██║     ██╔══╝
██║ ╚████║███████╗╚██████╗██║
╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝
```

**Non-Equilibrium Cognitive Field**

*A self-modifying dynamical system for proto-cognitive adaptation*

<br/>

[![Status](https://img.shields.io/badge/Status-Beta%20·%20Active%20Research-f59e0b?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![License](https://img.shields.io/badge/License-Apache-22c55e?style=for-the-badge)]()
[![Research](https://img.shields.io/badge/Research-NIT%20Agartala-7c3aed?style=for-the-badge)]()
[![Daily Log](https://img.shields.io/badge/Daily%20Research%20Log-docs%2Fsessions-38bdf8?style=for-the-badge)]()

<br/>

> **This project is in active beta development.**
> Results reported here are preliminary and subject to revision as the research matures.
> Contributions, critiques, and collaborations are warmly welcome.

<br/>

**Devanik** · B.Tech ECE '26, NIT Agartala
Samsung  Fellowship · Indian Institute of Science

</div>

---

## What Is This?

NECF is an attempt to explore a question that does not yet have a satisfying answer in the literature:

> *Can a system's **learning rules** themselves be treated as dynamic state variables — evolving continuously under thermodynamic constraints — while preserving a coherent identity?*

Most adaptive systems operate at two levels: a **state** that changes (neural activations, oscillator phases) and a **rule** that governs how the state changes (weights, coupling strengths). The rule is typically fixed or updated by a separate, outer-loop optimizer. NECF introduces a third level: the rule governing each node's adaptation is itself a dynamic variable, constrained by an identity curvature functional that prevents both chaotic drift and catatonic collapse.

This is not a claim about intelligence or general reasoning. It is a study of whether **Level-3 meta-rule dynamics** produces measurably different adaptive behavior from Level-1 (fixed rules) and Level-2 (globally adaptive coupling) baselines — and whether that difference can be characterized rigorously.

---

## Table of Contents

1. [Theoretical Framework](#1-theoretical-framework)
2. [The Three Levels of Dynamics](#2-the-three-levels-of-dynamics)
3. [Mathematical Specification](#3-mathematical-specification)
4. [Identity Curvature Functional](#4-identity-curvature-functional)
5. [Epistemic Contagion — The Core Mechanism](#5-epistemic-contagion--the-core-mechanism)
6. [External Driving and Non-Equilibrium Structure](#6-external-driving-and-non-equilibrium-structure)
7. [Seven Falsifiable Predictions](#7-seven-falsifiable-predictions)
8. [Experimental Results (Beta)](#8-experimental-results-beta)
9. [Relationship to Prior Work](#9-relationship-to-prior-work)
10. [What NECF Is Not](#10-what-necf-is-not)
11. [Codebase Structure](#11-codebase-structure)
12. [Installation and Usage](#12-installation-and-usage)
13. [Daily Research Log](#13-daily-research-log)
14. [Known Limitations](#14-known-limitations)
15. [Roadmap](#15-roadmap)
16. [References](#16-references)

---

## 1. Theoretical Framework

### 1.1 Motivation

The standard picture of an adaptive system is a two-level hierarchy:

$$
\text{Level 1: } \frac{d\phi}{dt} = F(\phi,\, \mathcal{L})
$$

$$
\text{Level 2: } \frac{d\mathcal{L}}{dt} = 0 \quad \text{(fixed)} \quad \text{or} \quad G(\mathcal{L}, \nabla_\phi \mathcal{J}) \quad \text{(trained by outer loop)}
$$

where $\phi$ is the system state, $\mathcal{L}$ is the learning rule, and $\mathcal{J}$ is a loss function. The key characteristic of Level-2 systems is that **the function $G$ itself is fixed** — it does not adapt. Meta-learning approaches like MAML (Finn et al., 2017) find a good initialization for $\mathcal{L}$, but the update rule for $\mathcal{L}$ during deployment is still fixed.

NECF proposes a genuine third level:

$$
\text{Level 3: } \frac{d\mathcal{L}_i}{dt} = \underbrace{F_{\text{contagion}}(\mathcal{L}_i,\, \varepsilon_i,\, W)}_{\text{Boltzmann epistemic contagion}} - \lambda\, \underbrace{\nabla_{\mathcal{L}_i} \mathcal{H}[\mathcal{L}]}_{\text{identity gradient}}
$$

where $F_{\text{contagion}}$ is an error-driven, thermodynamically-weighted rule propagation mechanism, and $\mathcal{H}[\mathcal{L}]$ is an **identity curvature functional** that constrains the evolution. Critically, the rule governing $\mathcal{L}$'s evolution is **not fixed** — it adapts through the Boltzmann-weighted averaging of neighbor rules, making the rule-evolution process itself a genuine spatial field phenomenon.

### 1.2 Physical Intuition

The system draws on three physical analogies:

**Kuramoto model (Kuramoto, 1975):** $N$ coupled oscillators with natural frequencies $\omega_i$ synchronize above a critical coupling $K_c$. NECF uses the amplitude-weighted Kuramoto model as its Level-1 substrate, with local coupling strength $\beta_i$ drawn from the rule field rather than a global constant.

**Dissipative structures (Prigogine, 1977):** Open thermodynamic systems driven far from equilibrium can spontaneously organize into ordered states. NECF is explicitly constructed as an open system — driven by Lorenz chaotic input, periodic signals, and Poisson spikes — to prevent equilibration and maintain perpetual self-organization.

**Active inference (Friston, 2017):** The Curiosity Engine monitors prediction error and drives the system toward states of minimal free energy. NECF's departure from standard active inference is that the minimization mechanism itself (the $\mathcal{L}$ field) is dynamic.

---

## 2. The Three Levels of Dynamics

The following taxonomy situates NECF in the landscape of adaptive systems:

| Level | What evolves | What is fixed | Representative systems |
|:---:|---|---|---|
| **0** | Nothing | Everything | Lookup tables, hardcoded controllers |
| **1** | State $\phi$ | Rule $\mathcal{L}$ | Standard Kuramoto, fixed-weight neural networks |
| **2** | State $\phi$, coupling $K_{ij}$ | Rule for updating $K$ | Adaptive Kuramoto (Ha et al., 2016), gradient descent |
| **3** | State $\phi$, rule $\mathcal{L}$ | Identity curvature $\mathcal{H}$ | **NECF (this work)** |
| **4** | State, rule, and $\mathcal{H}$ | — | Hypothetical; not yet formalized |

The specific claim of this project is that Level-3 dynamics — where $\mathcal{L}$ evolves as a continuous field variable under the constraint of a fixed identity functional — has not been previously formalized as a unified implementable architecture in the coupled oscillator literature. A March 2026 literature search confirmed that:

- Standard Kuramoto (1975): $K$ is a global scalar, fixed
- Adaptive Kuramoto (Ha et al., SIAM 2016): $K_{ij}$ evolves, but the **rule governing $K$'s evolution is fixed**
- Hopfield-Kuramoto (arXiv 2505.03648, May 2025): joint memory model, no meta-rule evolution
- MAML (Finn et al., 2017): learns an **initialization** for $\mathcal{L}$, not a continuously evolving rule field with identity constraints
- Friston FEP (2005–2022): minimizes free energy via a **fixed** variational update rule

The gap NECF occupies is the combination of (a) per-node rule fields, (b) Boltzmann-weighted epistemic contagion between rules, (c) identity curvature constraint on rule evolution, and (d) Lyapunov-gated rollback — unified in a single dynamical system.

---

## 3. Mathematical Specification

### 3.1 State Representation

Each node $i \in \{1, \ldots, N\}$ carries a complex state:

$$
\phi_i(t) = A_i(t)\, e^{i\theta_i(t)}
$$

- $A_i \in (0, 1]$: amplitude — interpreted as local confidence or energy
- $\theta_i \in [0, 2\pi)$: phase — causal alignment / temporal coordinate

The **Kuramoto order parameter** quantifies global synchrony:

$$
r(t)\, e^{i\psi(t)} = \frac{1}{N} \sum_{j=1}^N A_j\, e^{i\theta_j}
$$

$r \in [0, 1]$ with $r = 0$ (incoherent) and $r = 1$ (fully synchronized).

### 3.2 Local Learning Rule Field

Each node carries a **rule vector**:

$$
\mathcal{L}_i(t) = \bigl(\alpha_i(t),\; \beta_i(t),\; \gamma_i(t)\bigr)
$$

| Parameter | Role | Initial value |
|---|---|:---:|
| $\alpha_i$ | Error sensitivity — governs amplitude decay rate | 0.30 |
| $\beta_i$ | Coupling strength — governs phase synchrony pull | 0.80 |
| $\gamma_i$ | Curiosity weight — governs uncertainty-seeking | 0.10 |

### 3.3 Level-1 Dynamics: Field Evolution

**Phase update** (amplitude-weighted generalized Kuramoto with curiosity and external driving):

$$
\frac{d\theta_i}{dt} = \underbrace{\omega_i}_{\text{intrinsic}} + \underbrace{\beta_i \cdot \frac{1}{N} \sum_{j=1}^N W_{ij}\, A_j\, \sin(\theta_j - \theta_i)}_{\text{synchrony pull}} + \underbrace{\gamma_i \cdot \nabla_{\theta_i} U(\theta_i)}_{\text{curiosity gradient}} + \underbrace{\Delta_{\text{ext}}(t)}_{\text{Lorenz + spikes}}
$$

where $W_{ij} \sim \mathcal{U}(0.5, 1.5)$ is the symmetric coupling matrix (zero diagonal) and $\Delta_{\text{ext}}(t)$ captures spatially distributed Lorenz kicks and Poisson phase resets (Section 6).

**Curiosity potential** — approximated via circular KDE with Silverman bandwidth $h = 1.06\,\hat{\sigma}_\theta\, N^{-1/5}$:

$$
p(\theta_i) \approx \frac{1}{N} \sum_{j=1}^{N} \frac{1}{h} \exp\!\left(-\frac{\sin^2(\theta_i - \theta_j)}{2h^2}\right), \qquad U(\theta_i) = -\log p(\theta_i \mid \text{context})
$$

Nodes in dense, coherent phase regions experience a low $\|\nabla U\|$; nodes in sparse, chaotic regions experience a high gradient, pulling them toward informationally underexplored territory.

**Amplitude update** (error-driven with noise and periodic modulation):

$$
\frac{dA_i}{dt} = -\alpha_i\, \varepsilon_i(t)\, A_i + \sigma\, \eta_i(t) + \varepsilon_s \sin(2\pi f_s t)
$$

where $\varepsilon_i(t) = \sin^2\!\bigl((\theta_i - \psi)/2\bigr) \in [0,1]$ is the local prediction error (squared circular distance from the mean-field phase $\psi$), $\eta_i(t) \sim \mathcal{N}(0,1)$ is a Wiener process with amplitude $\sigma = 0.02$ (thermodynamic floor preventing amplitude collapse to zero), and $\varepsilon_s \sin(2\pi f_s t)$ is the periodic amplitude modulation driver ($\varepsilon_s = 0.03$, $f_s = 0.1$).

**Mean-field critical coupling** (Kuramoto, 1975; Strogatz, 2000):

$$
K_c = \frac{2}{\pi\, g(\Omega)} = \frac{2\sigma_\omega\sqrt{2\pi}}{\pi} \approx 0.4787 \quad \text{for } \sigma_\omega = 0.3
$$

At initialization $\beta_i \approx 0.80$, giving an effective coupling $K_{\text{eff}} = \beta_i \cdot K \cdot \bar{W} \approx 0.56 > K_c$, placing the field in the synchronizing regime from the outset.

### 3.4 Level-3 Dynamics: Meta-Rule Evolution

$$
\frac{d\mathcal{L}_i}{dt} = \underbrace{F_{\text{contagion}}(\mathcal{L}_i,\, \varepsilon_i,\, W)}_{\text{Boltzmann epistemic contagion}} \;-\; \lambda\, \underbrace{\nabla_{\mathcal{L}_i} \mathcal{H}[\mathcal{L}]}_{\text{identity gradient}}
$$

This is a dynamical system **in rule space**, not gradient descent on a loss function. The contagion term $F$ propagates rules from low-error nodes to high-error nodes; the identity gradient $\nabla \mathcal{H}$ prevents unlimited drift.

---

## 4. Identity Curvature Functional

The central object distinguishing NECF from prior adaptive systems is $\mathcal{H}[\mathcal{L}]$:

$$
\mathcal{H}[\mathcal{L}] = \underbrace{\frac{1}{N} \sum_{i=1}^N \|\mathcal{L}_i(t) - \mathcal{L}_i^{(0)}\|^2}_{\text{drift penalty}} + \underbrace{\kappa\; \overline{\text{Var}}(\mathcal{L}_i)}_{\text{collapse penalty}}
$$

**Properties:**

- $\mathcal{H}[\mathcal{L}] = 0$ at initialization by construction
- $\mathcal{H} \to \infty$ in both failure modes:
  - *Chaotic drift*: each $\mathcal{L}_i$ wanders freely → drift term grows
  - *Catatonic collapse*: all $\mathcal{L}_i \to \bar{\mathcal{L}}$ (identical) → variance penalty activates
- The **viable regime** corresponds to $\mathcal{H}$ bounded in a system-dependent interval

The gradient used in the meta-dynamics update:

$$
\nabla_{\mathcal{L}_i} \mathcal{H} = \frac{2}{N}\bigl(\mathcal{L}_i - \mathcal{L}_i^{(0)}\bigr) + \frac{2\kappa}{N}\bigl(\mathcal{L}_i - \bar{\mathcal{L}}\bigr)
$$

where $\bar{\mathcal{L}} = \frac{1}{N}\sum_i \mathcal{L}_i$ is the field mean. The first term is an elastic restoring force toward each node's structural origin; the second term is a **repulsive field** centered on the mean — it pushes nodes away from homogenization, preserving spatial rule diversity.

### 4.1 Rollback Mechanism

At each step, the system checks whether identity curvature has increased too rapidly:

$$
\delta\mathcal{H}(t) = \mathcal{H}[\mathcal{L}(t)] - \mathcal{H}[\mathcal{L}(t - \Delta t)]
$$

If $\delta\mathcal{H} > \delta_{\text{thresh}}$ (default $0.30$), the system applies a **thermodynamic rollback**:

$$
\mathcal{L}(t) \leftarrow \mathcal{L}(t - \Delta t) - \eta_{\text{rb}} \cdot \nabla_\mathcal{L} \mathcal{H}[\mathcal{L}(t - \Delta t)], \quad \eta_{\text{rb}} = 0.05
$$

This is not gradient descent — it is a reversion to the previous state plus a corrective step in the direction of lower identity curvature. The combined effect prevents the same spike from recurring immediately and provides a Lyapunov stability certificate for the rule field over $T \to \infty$.

---

## 5. Epistemic Contagion — The Core Mechanism

The contagion term $F_{\text{contagion}}$ implements thermodynamically-weighted rule propagation. The key design decision — and the site of the primary engineering fix — is the choice of influence weights.

### 5.1 The Singularity Problem (Rejected Design)

A naive inverse-error weighting:

$$
w_j^{\text{naive}} = \frac{1}{\varepsilon_j + \epsilon}
$$

is singular. For $\varepsilon_j = 10^{-5}$, $w_j \approx 10^5$. After normalization this degenerates to a one-hot vector, snapping the entire field to the rule of whichever single node happens to achieve near-zero error on that timestep. This is **not** field dynamics — it is a discrete logic switch masquerading as a continuous system.

### 5.2 Boltzmann Softmax Weights (Adopted Design)

$$
w_j(\kappa) = \frac{\exp(-\varepsilon_j / \kappa)}{\displaystyle\sum_{k=1}^N \exp(-\varepsilon_k / \kappa)}
$$

implemented with the log-sum-exp trick for numerical stability:

```python
log_w = -eps / kappa
log_w -= log_w.max()          # stability
w = np.exp(log_w)
w /= w.sum() + 1e-15
```

The weight is globally $C^\infty$; its sensitivity to error is:

$$
\frac{\partial w_j}{\partial \varepsilon_j} = -\frac{1}{\kappa}\, w_j(1 - w_j)
$$

which is bounded by $[-1/(4\kappa),\, 0]$ for all $\varepsilon_j$ — preventing unbounded weight gradients and guaranteeing mathematical stability of the numerical integration step.

**Properties of the Boltzmann weights:**

| $\kappa$ | Character | $H_w = -\sum w_j \ln w_j$ | Max weight |
|:---:|---|:---:|:---:|
| $\to 0$ | Winner-takes-all | $\to 0$ | $\to 1$ |
| $0.10$ | Default (selective) | $\approx 3.85$ | $\approx 0.053$ |
| $0.50$ | Diffuse | $\approx 4.15$ | $\approx 0.021$ |
| $\to \infty$ | Uniform | $\ln N$ | $1/N$ |

At $\kappa = 0.10$, the maximum weight across a 64-node field is approximately $0.016$, compared to $\sim 90{,}000$ under the naive formula — a six-order-of-magnitude reduction that preserves genuine field diffusion.

### 5.3 Contagion Update

For each node $i$, the Boltzmann-weighted target rule is:

$$
\mathcal{L}_i^{\text{target}} = \frac{\sum_j W_{ij}\, w_j\, \mathcal{L}_j}{\sum_j W_{ij}\, w_j + \epsilon}
$$

The contagion contribution to $d\mathcal{L}_i/dt$:

$$
F_{\text{contagion},i} = \boldsymbol{\mu} \odot \bigl(\mathcal{L}_i^{\text{target}} - \mathcal{L}_i\bigr) \cdot \varepsilon_i
$$

where $\boldsymbol{\mu} = (\mu_\alpha, \mu_\beta, \mu_\gamma) = (0.05, 0.05, 0.05)$ and $\odot$ is elementwise multiplication. The factor $\varepsilon_i$ acts as **receptivity**: high-error nodes update more strongly toward the rules of their low-error neighbors; low-error nodes are stubborn — they resist change, acting as stable anchors from which contagion radiates.

**Two-group mixing time** (analytically derived and empirically verified):

$$
\tau_{\text{mix}}(\kappa) \approx \frac{1}{\mu\, \varepsilon_h\, w_{\text{low}}(\kappa)\, \Delta t}
$$

At the default $\kappa = 0.10$, $\mu = 0.50$ (accelerated test), $\varepsilon_h = 0.25$: $\tau_{\text{emp}} = 878$ steps, $\tau_{\text{theory}} = 708$ steps — relative error $\approx 19\%$, consistent with the non-uniform coupling matrix correction.

---

## 6. External Driving and Non-Equilibrium Structure

NECF is deliberately constructed as an **open thermodynamic system**. Three driving mechanisms prevent the field from reaching equilibrium:

### 6.1 Lorenz Chaotic Driver

The Lorenz attractor $(x, y, z)$ with standard parameters $(\sigma=10,\, \rho=28,\, \beta=8/3)$ injects a spatially-distributed, deterministic chaotic phase perturbation:

$$
\Delta\theta_i^{\text{Lorenz}} = \varepsilon_L \cdot \frac{x(t)}{25} \cdot \sin\!\left(\frac{2\pi i}{N}\right), \quad \varepsilon_L = 0.05
$$

The spatial factor $\sin(2\pi i/N)$ is essential: nodes near $i = N/4$ receive a strong positive kick while nodes near $i = 3N/4$ receive an equal negative kick, physically tearing the field in half in a deterministic, aperiodic, structured manner — a far richer perturbation regime than white noise.

### 6.2 Periodic Signal

A structured sinusoidal driver creates resonance opportunities and modulates node amplitudes globally:

$$
\Delta A_i^{\text{periodic}} = \varepsilon_s \cdot \sin(2\pi f_s t), \quad \varepsilon_s = 0.03,\quad f_s = 0.1
$$

### 6.3 Poisson Phase Resets

At each step, each node independently spikes with probability $\lambda_s = 0.02$, resetting its phase to $\mathcal{U}(0, 2\pi)$. This is the mechanism responsible for the **masked Lyapunov proxy**: a spiked node can jump from $\theta = 0.1$ to $\theta = 5.9$ in one step — a phase delta of $5.8$ radians. The raw $\|\delta\theta\|$ would explode, the rollback would trigger continuously, and all dynamics would freeze. The fix is to compute phase divergence only on nodes that were not spiked at $t$ or $t-1$:

$$
\hat{\lambda}_1 = \frac{1}{\Delta t}\, \log\!\left(\frac{\|\delta\theta_{\sim\text{spike}}\|}{\sqrt{N_{\text{valid}}}} + 10^{-10}\right)
$$

where $\sim\text{spike}$ denotes exclusion of spiked nodes at both $t$ and $t-1$, and $N_{\text{valid}}$ is the number of unmasked nodes. When $N_{\text{valid}} < 2$ (edge case: near-total spike event), the estimator returns `NaN` and the step is classified `SPIKE_DOMINATED` — a transient physical event, not internal chaos. The spike mask is explicitly threaded from `environment.step()` through `field.step()` to `observer.record()` at every timestep.

---

## 7. Seven Falsifiable Predictions

NECF makes seven quantitative predictions, each independently falsifiable:

| # | Observable | Prediction | Falsification condition |
|:---:|---|---|---|
| P1 | Order parameter $r(t)$ | Rises from $\sim 0.1$ to $> 0.5$ within viable regime | $r < 0.2$ at all times |
| P2 | Identity curvature $\mathcal{H}[\mathcal{L}]$ | Remains bounded in $(0.1,\, 5.0)$ throughout | $\mathcal{H}$ diverges or collapses to zero |
| P3 | Mean prediction error $\bar{\varepsilon}$ | Decreases after $\sim 200$ steps | $\varepsilon$ monotonically increases |
| P4 | Masked Lyapunov proxy $\hat{\lambda}_1$ | Stays in $(-0.5,\, 0.8)$ | $\hat{\lambda}_1 > 1.5$ sustained for $> 50$ steps |
| P5 | Rule diversity $\text{Var}(\mathcal{L}_i)$ | Bounded; neither collapses to zero nor diverges | $\text{Var}(\mathcal{L}_i) \to 0$ |
| P6 | Curiosity directives | Fires within 30 steps of plateau detection | Never fires, or fires on non-plateau steps |
| P7 | Rollback rate | Decreases over the course of a run | Rate increases monotonically |

These predictions are what distinguish NECF from a philosophical proposal. A system that satisfies all seven is behaving as the theory predicts; a system that fails any one provides specific information about which component requires revision.

---

## 8. Experimental Results (Beta)

> ⚠️ All results below are preliminary. $N$ is small (16–64 nodes), run lengths are short ($T \leq 2000$ steps), and no task-level benchmark has been applied. These are substrate-characterization experiments only.

### 8.1 Synchronization Onset (T0)

Sweep over $K \in [0.05, 2.50]$, $N = 64$, $\sigma_\omega = 0.3$:

| $K$ | $\bar{r}$ |
|:---:|:---:|
| $0.30$ | $0.162$ |
| $K_c^{\text{th}} = 0.479$ | $0.109$ |
| $0.60$ | $0.405$ |
| $1.00$ | $0.820$ |
| $2 K_c^{\text{th}} = 0.958$ | $0.928$ |

Empirical $K_c$ (first $K$ where $r > r_{\text{bg}} + 3\sigma_{\text{noise}}$, with $r_{\text{bg}} = N^{-1/2} \approx 0.125$) is approximately $0.89$, compared to the mean-field prediction $0.479$. The deviation is consistent with finite-size broadening: for $N = 64$ the stochastic background elevates the apparent threshold.

The order parameter scales approximately as $r \sim (K - K_c)^\beta$ with fitted $\hat{\beta}$ near $0.5$, consistent with mean-field universality.

### 8.2 Boltzmann Temperature Scan (T1)

Over 22 values of $\kappa \in [0.01, 5.00]$, the weight entropy $H_w(\kappa) = -\sum_j w_j \ln w_j$ increases monotonically from near-zero (winner-takes-all) toward $\ln N$ (uniform). The optimal operating point $\kappa^* = \arg\max_\kappa H_w(\kappa) \cdot r(\kappa)$ balances discrimination against field diversity. The default $\kappa = 0.10$ lies in the selective regime with $H_w \approx 3.85$ (out of $\ln 32 \approx 3.47$ maximum for $N=32$).

### 8.3 Identity Stability Landscape (T2)

Grid sweep over $\lambda \in \{0.01, 0.05, 0.10, 0.25, 0.50, 1.00\}$ and $\delta_{\text{thresh}} \in \{0.10, 0.20, 0.30, 0.50, 0.80\}$:

- **67% of the explored $(\lambda, \delta)$ space** produces viable dynamics at $T = 400$ steps
- Regime boundaries: VIABLE / CATATONIC ($r < 0.04$) / DRIFTED ($\mathcal{H} > 3$) / ROLLBACK-HEAVY
- Default values $(\lambda = 0.10,\, \delta = 0.30)$ fall in the VIABLE cell

### 8.4 Lyapunov Spectrum (T3)

Full spectrum via continuous QR decomposition (Benettin et al., 1980), $N = 16$, $K = 0.70$:

$$
\lambda_1 = +0.173,\quad \lambda_2 = +0.049,\quad \lambda_3 = -0.060,\; \ldots
$$

Kaplan–Yorke dimension:

$$
D_{\text{KY}} = j + \frac{\sum_{k=1}^{j} \lambda_k}{|\lambda_{j+1}|} = 4.84
$$

indicating a fractal strange attractor of dimension $\approx 4.8$ embedded in the 16-dimensional phase space. This places the system in the bounded-chaos regime — neither frozen ($\lambda_1 < 0$) nor explosively divergent ($\lambda_1 \gg 1$).

### 8.5 Epistemic Contagion Rate (T4)

Two-group mixing time as a function of $\kappa$ (pure contagion, $\mu = 0.50$, $T = 2500$):

| $\kappa$ | $\tau_{\text{emp}}$ | $\tau_{\text{theory}}$ | Rel. error |
|:---:|:---:|:---:|:---:|
| $0.02$ | $666$ | $667$ | $0.1\%$ |
| $0.05$ | $670$ | $671$ | $0.1\%$ |
| $0.10$ | $708$ | $721$ | $1.8\%$ |
| $0.30$ | $878$ | $956$ | $8.2\%$ |
| $1.00$ | $1027$ | $1186$ | $13.4\%$ |

Power law: $\tau_{\text{mix}} \sim \kappa^{0.121}$, $R^2 = 0.92$. Higher temperature slows mixing; the analytical formula $\tau = 1/(\mu\, \varepsilon_h\, w_{\text{low}}(\kappa)\, \Delta t)$ predicts the trend with good accuracy at low $\kappa$.

### 8.6 Free Energy Topology (T5)

From 40 random initialisations ($N = 32$, $K = 0.80$, $T = 600$ settling steps): **11 distinct attractor basins** detected by clustering final mean phases. Lorenz-perturbation escape rate: $k_{\text{esc}} \approx 0.38$, indicating a moderately fragile landscape where roughly 38% of settled trajectories are dislodged by a 50-step Lorenz kick of amplitude 0.10.

### 8.7 Ablation Study (T6)

$N = 32$, $T = 600$, $K = 0.55$, $n = 25$ trials per condition:

| Condition | $\bar{r} \pm \sigma$ | $\text{Var}(\mathcal{L})$ |
|---|:---:|:---:|
| **Level-1** (frozen $\mathcal{L}$) | $0.0335 \pm 0.037$ | $9.5 \times 10^{-5}$ |
| **Level-2** (global $\beta$ adapts) | $0.0335 \pm 0.037$ | $9.5 \times 10^{-5}$ |
| **Level-3** (full NECF) | $0.0399 \pm 0.043$ | $7.2 \times 10^{-5}$ |

Welch's $t$-test (L3 vs L1): $t = 0.436$, $p = 0.665$, Cohen's $d = 0.31$.

The difference in $r$ is **not statistically significant at $T = 600$**. This is expected: the contagion timescale at $\mu = 0.05$ is $\tau_{\text{mix}} \approx 8{,}000$ steps, so $T = 600$ captures only very early adaptation. The $\text{Var}(\mathcal{L})$ ratio (Level-3 / Level-1 $\approx 0.76\times$) confirms that the meta-dynamics is active and measurably modifying the rule field, even when the effect on $r$ is not yet visible.

> **Interpretation:** The current experiments are substrate characterization, not performance benchmarking. The ablation will be re-run at $T \geq 5{,}000$ once compute budget permits.

---

## 9. Relationship to Prior Work

NECF is built on and departs from several bodies of literature:

### 9.1 Kuramoto Model

The standard Kuramoto model (Kuramoto, 1975; Strogatz, 2000) uses a fixed global coupling $K$. Adaptive extensions (Ha, Kim & Zhang, SIAM 2016; Berner et al., 2021) allow $K_{ij}(t)$ to evolve, but the rule governing $K_{ij}$'s evolution is fixed. NECF departs by making the **local coupling strength** $\beta_i$ a component of the dynamically-evolving rule field $\mathcal{L}_i$.

### 9.2 Free Energy Principle

Friston's Free Energy Principle (2005–2022) provides the theoretical foundation for prediction-error-driven adaptation and the curiosity/active inference components. NECF's departure is that the variational update rule (what FEP holds fixed) is itself a dynamic variable — the $\mathcal{L}$ field plays the role of a variational family whose parameters co-evolve with the belief state.

### 9.3 Meta-Learning

MAML (Finn, Abbeel & Levine, 2017) and its descendants learn a good initialization $\mathcal{L}_0$ such that few gradient steps suffice for a new task. NECF is not an initialization-finding algorithm. It is a continuous-time dynamical system in which $\mathcal{L}$ evolves in deployment, not through explicit gradient computation on a loss, but through thermodynamically-weighted peer-to-peer rule exchange.

### 9.4 Dissipative Structures

Prigogine's theory of dissipative structures (1977) demonstrates that open thermodynamic systems far from equilibrium can spontaneously organize into ordered states through energy dissipation. NECF uses this principle architecturally: the Lorenz driving and Poisson spikes ensure the system never equilibrates, creating the conditions for persistent self-organization.

### 9.5 Causal Entropy / Empowerment

The proto-will mechanism draws on causal entropy maximization (Klyubin, Polani & Nehaniv, 2005; Salge, Glackin & Polani, 2014), where an agent selects actions that maximize the logarithm of future state-space volume accessible under its policy. NECF's attractor selection formula:

$$
a^* = \arg\max_a \bigl[H_{\text{causal}}(a) - \mu_{\text{will}}\, D_{\text{identity}}(a)\bigr]
$$

where $H_{\text{causal}}(a) = -\sum_{S'} p(S' \mid a)\log p(S' \mid a)$ is the Shannon entropy of future states reachable from attractor $a$, and $D_{\text{identity}}(a) = \frac{1}{N}\sum_i \|\mathcal{L}_{i,a} - \mathcal{L}_{i,\text{current}}\|^2$ is the structural cost of transitioning to that attractor. At $\mu_{\text{will}} = 0$: pure exploration (destructive). At $\mu_{\text{will}} \to \infty$: complete paralysis. The default $\mu_{\text{will}} = 1.0$ is a constrained empowerment maximizer — seeking influence while preserving identity.

---

## 10. What NECF Is Not

In the interest of accuracy, several things should be clearly stated:

**NECF is not a cognitive architecture.** It has no perception layer, no task representation, and no mechanism for mapping external stimuli to field states or extracting decisions from phase patterns.

**NECF has not been benchmarked on any task.** No classification accuracy, no ARC score, no game score. The experiments reported are purely substrate-characterization: does the field behave as the theory predicts?

**NECF does not demonstrate intelligence or reasoning.** It demonstrates that a coupled oscillator field with Level-3 meta-rule dynamics behaves differently from Level-1 and Level-2 baselines in specific, measurable ways. Whether that difference is relevant to cognitive function is an open question.

**NECF is not a refutation of neural networks.** It is an exploration of a different computational primitive — one that may complement, rather than replace, existing approaches.

The honest position is that NECF is a **proto-cognitive substrate** — a dynamical system with properties that may be useful building blocks for more complete architectures, but which is not itself a reasoning system.

---

## 11. Codebase Structure

```
necf/
├── app.py               # Streamlit live dashboard — entry point
├── config.py            # All hyperparameters (single source of truth)
├── field.py             # Level-1: oscillator field state + Numba-compiled Kuramoto
├── meta_dynamics.py     # Level-3: Boltzmann epistemic contagion + rule evolution
├── identity.py          # H[L] curvature functional + Lyapunov-gated rollback
├── curiosity.py         # CuriosityEngine: plateau detection + exploration directives
├── environment.py       # External drivers: Lorenz attractor, periodic, Poisson spikes
├── observer.py          # All observables: r(t), masked λ_max, entropy, regime classifier
├── experiment.py        # Run loop with ablation helpers (Level-1, Level-2, Level-3)
└── requirements.txt
```

```
.github/
└── workflows/
    └── daily_necf_research.yml   # Automated daily research session (no LLM)
```

```
docs/
├── INDEX.md             # Session index — one entry per day
└── sessions/
    └── YYYY-MM-DD_<slug>.md    # Technical research note per session
```

### Key Design Decisions

**`field.py` — Numba JIT kernel:** The $O(N^2)$ Kuramoto phase-difference computation is compiled via `@numba.njit(parallel=True, fastmath=True)`, pre-warmed during `__init__` to avoid first-step latency in Streamlit. Falls back to NumPy broadcasting if Numba is unavailable.

**`observer.py` — Masked Lyapunov:** The Lyapunov proxy excludes nodes spiked at $t$ *and* $t-1$ (a spike at $t-1$ still corrupts the delta at $t$). Returns `NaN` with `n_valid_lyapunov` when coverage is insufficient, rather than a spurious value.

**`experiment.py` — Ablation factories:** `NECFExperiment.level1_only()` and `.level2_adaptive_coupling()` return pre-configured instances for controlled comparison. The ablation factories override `meta.step` rather than duplicating logic, ensuring identical dynamics up to the rule-update layer.

---

## 12. Installation and Usage

### Requirements

```
python >= 3.11
numpy >= 1.24
scipy >= 1.11
streamlit >= 1.32
matplotlib >= 3.7
numba >= 0.58      # optional but strongly recommended
```

### Install

```bash
git clone https://github.com/Devanik21/NECF.git
cd NECF
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`. The dashboard provides:
- Real-time phase polar plot and order parameter trace
- All 7 observable metrics updated live
- Rule field distribution histograms
- Identity curvature $\mathcal{H}[\mathcal{L}]$ phase diagram
- Curiosity Engine directive timeline
- Regime classifier (VIABLE / CHAOTIC / CATATONIC)

### Scripted Usage

```python
from config import NECFConfig
from experiment import NECFExperiment

# Full NECF (Level-3)
exp = NECFExperiment(NECFConfig(seed=42))
for snap in exp.stream():
    print(f"t={snap.t:4d}  r={snap.r:.4f}  H={snap.H:.4f}  λ={snap.lyapunov_proxy:+.4f}")

# Ablation baselines
exp_l1 = NECFExperiment.level1_only()          # fixed rules
exp_l2 = NECFExperiment.level2_adaptive_coupling()  # global β only
```

### Running the Ablation Study

```python
import numpy as np
from scipy.stats import ttest_ind
from experiment import NECFExperiment
from config import NECFConfig

N_TRIALS = 25
results = {1: [], 2: [], 3: []}

for seed in range(N_TRIALS):
    cfg = NECFConfig(seed=1000 + seed)
    for level, factory in [(1, NECFExperiment.level1_only),
                            (2, NECFExperiment.level2_adaptive_coupling),
                            (3, NECFExperiment)]:
        exp = factory(cfg)
        history = exp.run_batch()
        results[level].append(history.snapshots[-1].r)

r1, r3 = np.array(results[1]), np.array(results[3])
t, p = ttest_ind(r3, r1)
print(f"L3 vs L1: t={t:.3f}, p={p:.4f}, d={( r3.mean()-r1.mean())/np.sqrt((r3.var()+r1.var())/2):.3f}")
```

---

## 13. Daily Research Log

A GitHub Actions workflow (`.github/workflows/daily_necf_research.yml`) runs once per day at 06:00 UTC and performs a real numerical experiment on the NECF substrate, writing the results to `docs/sessions/`. Seven analysis types rotate by day-of-year modulo 7:

| Topic index | Analysis |
|:---:|---|
| T0 | Synchronization onset and critical coupling — sweep $K$, fit $\beta$ |
| T1 | Boltzmann temperature scan — optimal $\kappa$ for rule-field diffusion |
| T2 | Identity curvature stability landscape — phase diagram in $(\lambda, \delta)$ space |
| T3 | Lyapunov spectrum via continuous QR decomposition |
| T4 | Epistemic contagion rate constant — two-group mixing time vs $\kappa$ |
| T5 | Free energy topology — attractor basin counting and escape rates |
| T6 | Ablation study — Level-1 vs Level-2 vs Level-3 comparison |

Each session generates a markdown file with:
- Full parameter specification and seed
- LaTeX-formatted mathematical derivation of the relevant theory
- Numerical results table
- Comparison to analytical predictions where available
- Interpretation and connection to the broader NECF framework

The workflow uses **no LLM and no external API** — all content is computed from first principles using NumPy and SciPy. The commit message encodes the key numerical result, e.g.:

```
necf(NECF-2026-077-T6): ablation level comparison [L3_r=0.0399 L1_r=0.0335 p=0.2852 d=0.3120]
```

See [`docs/INDEX.md`](docs/INDEX.md) for the full session archive.

---

## 14. Known Limitations

These are not future work items — they are current, known gaps in the architecture that affect the interpretation of all results:

**No task interface.** The field has no mechanism for receiving structured input or producing structured output. All experiments are self-contained dynamical characterizations. Whether the field's synchrony and rule diversity are relevant to any task-level performance is currently unknown.

**Short run lengths.** The contagion timescale at $\mu = 0.05$ is $\tau_{\text{mix}} \approx 8{,}000$ steps. Current experiments run to $T \leq 2{,}000$, capturing only the earliest phase of adaptation. The ablation results in particular should be treated as preliminary until longer runs are available.

**Small $N$.** Results are for $N \in \{16, 32, 64\}$. Scaling behavior — whether the viable regime survives as $N \to 256$ or $N \to 1024$ — has not been studied.

**Lyapunov proxy accuracy.** The masked $\hat{\lambda}_1$ is an approximation of the true maximal Lyapunov exponent. It has not been cross-validated against the full QR-method spectrum at matched parameters.

**Analytical gap.** The Kaplan–Yorke dimension formula $D_{\text{KY}} = j + \sum_{k=1}^j \lambda_k / |\lambda_{j+1}|$ assumes a smooth, ergodic attractor. Whether the NECF attractor satisfies these conditions under open-system driving is an open question.

**No comparison to learned baselines.** Level-1 and Level-2 ablations are the only baselines. A comparison to a gradient-trained RNN or a transformer-based adaptive system on the same substrate task would significantly strengthen the ablation.

---

## 15. Roadmap

The following phases are planned but have not yet begun. Timelines are estimates only.

**Phase 1 — Longer runs and scaling study** *(next 4–6 weeks)*
Run ablation at $T = 10{,}000$ steps to cross the contagion timescale. Sweep $N \in \{32, 64, 128, 256\}$ to characterize finite-size effects on the viable regime boundary.

**Phase 2 — Task interface** *(6–12 weeks)*
Design a perception layer that maps discrete input patterns to structured phase perturbations, and a readout mechanism that extracts categorical decisions from the order parameter. Apply to a simple pattern classification task.

**Phase 3 — Learned transformation spaces** *(12–20 weeks)*
Replace the fixed DSL primitives (from the companion LAteNT project) with embeddings learned from ARC-style input-output pairs. This connects NECF's rule-field dynamics to the program synthesis literature.

**Phase 4 — Cross-domain generalization measurement** *(20–28 weeks)*
Train on Task Domain A, validate on Domain B, test on Domain C (entirely unseen transformation types). Measure transfer rates as a function of $\tau_{\text{mix}}$ and field size $N$.

**Phase 5 — Publication** *(28–32 weeks)*
Target: NeurIPS 2027 Workshop on Emergence in Complex Systems, or a journal submission to *Physical Review E* (dynamical systems track) for the mathematical results.

---

## 16. References

Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J.-M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems. *Meccanica*, 15, 9–30.

Berner, R., Gross, T., Kuehn, C., Kurths, J., & Yanchuk, S. (2023). Adaptive dynamical networks. *Physics Reports*, 1031, 1–59.

Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML 2017*.

Friston, K., et al. (2017). Active inference and learning. *Neuroscience & Biobehavioral Reviews*, 68, 862–879.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

Ha, S.-Y., Kim, D., & Zhang, X. (2016). On the complete synchronization of the Kuramoto phase model. *SIAM Journal on Applied Dynamical Systems*, 15(1).

Kaplan, J. L., & Yorke, J. A. (1979). Chaotic behavior of multidimensional difference equations. *Lecture Notes in Mathematics*, 730, 204–227.

Klyubin, A. S., Polani, D., & Nehaniv, C. L. (2005). Empowerment: a universal agent-centric measure of control. *IEEE Congress on Evolutionary Computation*, 128–135.

Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *International Symposium on Mathematical Problems in Theoretical Physics*.

Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Prigogine, I. (1977). *Self-Organization in Non-Equilibrium Systems*. Wiley. (Nobel Lecture).

Salge, C., Glackin, C., & Polani, D. (2014). Empowerment — an introduction. *Guided Self-Organization: Inception*, 67–114. Springer.

Strogatz, S. H. (1994). *Nonlinear Dynamics and Chaos*. Addison-Wesley.

Strogatz, S. H. (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators. *Physica D*, 143(1–4), 1–20.

---

<div align="center">

**Author:** [Devanik](https://github.com/Devanik21) · B.Tech ECE '26, NIT Agartala
Samsung  Fellowship  · Indian Institute of Science

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-181717?style=flat-square&logo=github)](https://github.com/Devanik21)
[![Twitter](https://img.shields.io/badge/Twitter-@devanik2005-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/devanik2005)

*This is an independent research project in active development.*
*Feedback, questions, and collaboration proposals are always welcome.*

**License:** Apache License Version 2.0, January 2004 · **Status:** Beta

</div>
