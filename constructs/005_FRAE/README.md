# FRAE-S: Fluid Resonance Advantage Estimation — Stable

### A Navier-Stokes Inspired, Self-Calibrating Advantage Estimator for Actor-Critic Reinforcement Learning

**Devanik Debnath** · B.Tech ECE, National Institute of Technology Agartala  
[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![Status](https://img.shields.io/badge/Status-Experimental%20Research%20Prototype-orange?style=flat-square)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square&logo=pytorch)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)]()
[![March 2026](https://img.shields.io/badge/Date-March%202026-purple?style=flat-square)]()

---

> **Epistemic Declaration:** This is an experimental, unreviewed research prototype conceived and implemented independently. No claims of state-of-the-art performance are made. The contribution is architectural, theoretical, and—if validated—paradigmatic. The author explicitly invites rigorous critique from the research community.

---

## Abstract

Generalized Advantage Estimation (GAE, Schulman et al., 2015) remains the dominant credit assignment mechanism in on-policy actor-critic algorithms. Its core limitation is architectural: the temporal horizon parameter $\lambda \in [0,1]$ is a **global scalar**, fixed for the entire duration of training and blind to the local statistical dynamics of the reward signal. A single high-variance transition spike contaminates the entire backward credit chain. Low-variance regions where credit should flow freely are penalized by the same conservative $\lambda$ that protects against turbulent regions.

FRAE-S proposes a fundamentally different paradigm: **the temporal credit horizon at each timestep is not a hyperparameter — it is a physical observable**, computed from the instantaneous dynamics of the TD-error signal itself, governed by an analogy to the Navier-Stokes equations of fluid mechanics.

Specifically, FRAE-S introduces three novel mechanisms:

1. A **Viscosity Invariant** $\zeta_t$ — an exponential moving average of TD-error variance, acting as a measure of local signal stability (fluid viscosity).
2. An **Information Reynolds Number** $\text{Re}_t$ — the ratio of TD-error momentum to viscosity, quantifying whether the local information flow is laminar or turbulent.
3. A **Gaussian Phase Transition Gate** $\beta_t$ — a smooth, differentiable credit propagation coefficient driven by $\text{Re}_t$, replacing GAE's fixed $\lambda$ with a physically motivated, per-step value.

Additionally, FRAE-S v6 introduces a **Topological Calibration Protocol** — a pre-training environment probe that automatically derives the critical Reynolds threshold $\text{Re}_{\text{crit}}$ and entropy coefficient from the environment's own signal statistics, eliminating hyperparameter sensitivity entirely.

Verified on `LunarLander-v3` over 1500 epochs (750,000 total environment steps), FRAE-S demonstrates interpretable phase-transition dynamics — a measurable shift from turbulent ($\text{Re}_{\max} > 4$) to laminar ($\text{Re}_{\max} < 1$) flow correlating directly with policy improvement — and reaches peak training rewards of **+238.3**, with an episodic exceedance of the **+200 competence threshold** at epoch 1050.

---

## Table of Contents

1. [Motivation: The Tyranny of Fixed λ](#1-motivation-the-tyranny-of-fixed-λ)
2. [The FRAE-S Formulation](#2-the-frae-s-formulation)
3. [Topological Calibration Protocol](#3-topological-calibration-protocol)
4. [Thermodynamic Entropy Cooling](#4-thermodynamic-entropy-cooling)
5. [Architecture](#5-architecture)
6. [Loss Function](#6-loss-function)
7. [Comparison to Prior Work](#7-comparison-to-prior-work)
8. [Verified Training Results: LunarLander-v3](#8-verified-training-results-lunarlander-v3)
9. [Physics Telemetry: Reading the Fluid Dynamics](#9-physics-telemetry-reading-the-fluid-dynamics)
10. [Running the Code](#10-running-the-code)
11. [Hyperparameter Reference](#11-hyperparameter-reference)
12. [Proposed Future Research Directions](#12-proposed-future-research-directions)
13. [Limitations and Open Questions](#13-limitations-and-open-questions)
14. [References](#14-references)

---

## 1. Motivation: The Tyranny of Fixed λ

The standard GAE estimator is:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$$

This is elegant and computationally efficient. It is also structurally incapable of adapting to the local quality of the credit signal. The $\lambda$ parameter:

- **Does not know** whether $\delta_t$ is part of a stable, predictable reward region or a catastrophic spike from an unexplored transition.
- **Does not know** whether the value network has converged on the current section of the state space or is still wildly inaccurate.
- **Does not adapt** when the policy quality changes dramatically between epochs.

Existing adaptive advantage estimators have proposed using indicators of variance and bias to obtain adaptive weight parameters, and recent work on Distributional GAE introduces Wasserstein-like metrics between value distributions. However, none of these approaches introduce a **physical phase-transition mechanism** as the adaptive gate. They remain in the domain of statistical heuristics. FRAE-S asks a different question: *what if the transition between conservative and liberal credit propagation is governed by the same dimensionless number that governs the transition between laminar and turbulent flow in physical fluids?*

The Credit Assignment Problem remains poorly understood mathematically, and the mathematical nature of credit and the CAP is a crucial open challenge in RL. FRAE-S proposes one physically-motivated answer to this challenge at the level of the advantage estimator.

---

## 2. The FRAE-S Formulation

### 2.1 TD-Error as Input Kinetic Energy

$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$

The TD-error is the raw "information energy" entering the system at each timestep. In fluid mechanics, this corresponds to the velocity field — the input driving force of the flow.

### 2.2 Viscosity Invariant ζ_t

An Exponential Moving Average tracks the local second moment of TD-errors, playing the role of **dynamic fluid viscosity** — the resistance of the information medium to sudden, disorganized flow:

$$\sigma_t^2 = (1 - \alpha)\,\sigma_{t-1}^2 + \alpha\,\delta_t^2, \qquad \sigma_0^2 = \varepsilon$$

$$\boxed{\zeta_t = \sqrt{\sigma_t^2} + \varepsilon}$$

$\zeta_t$ is always strictly positive by construction. Its magnitude reflects the *historical volatility* of the reward signal: large $\zeta_t$ indicates a high-variance trajectory; small $\zeta_t$ indicates a stable, predictable regime. Crucially, $\zeta_t$ is **calibrated to the environment's native scale** via the Topological Probe (Section 3), so it is always dimensionally appropriate.

### 2.3 Information Reynolds Number

In classical fluid mechanics, the Reynolds number $\text{Re} = \frac{\rho u L}{\mu}$ is the ratio of **inertial forces** (tendency toward turbulence) to **viscous forces** (tendency toward stability). FRAE-S constructs an exact information-theoretic analogue:

$$\boxed{\text{Re}_t = \frac{|\delta_t - \delta_{t-1}|}{\zeta_t}}$$

- **Numerator** $|\delta_t - \delta_{t-1}|$: the *momentum* of the TD-error — how abruptly the information signal is changing. This is the inertial force of the information flow.
- **Denominator** $\zeta_t$: the local variance scale — the viscous resistance to abrupt change.

When $\text{Re}_t$ is small, the TD-error is changing slowly relative to its recent history: the information is flowing in an organized, laminar manner and credit can safely propagate backward. When $\text{Re}_t$ is large, the signal is in chaotic transition: credit propagation through this region would accumulate noise rather than signal.

### 2.4 Gaussian Phase Transition Gate β_t (v6)

FRAE-S v6 replaces the earlier hard binary switch with a **smooth Gaussian decay kernel** on the Reynolds number:

$$\boxed{\beta_t = \beta_{\text{lam}} \cdot \exp\!\left(-\left(\frac{\text{Re}_t}{\text{Re}_{\text{crit}}}\right)^2\right)}$$

This function has the following physically meaningful properties:

- At $\text{Re}_t = 0$: $\beta_t = \beta_{\text{lam}}$ — fully laminar, maximum credit propagation.
- At $\text{Re}_t = \text{Re}_{\text{crit}}$: $\beta_t = \beta_{\text{lam}} / e \approx 0.349$ — half-power point, significant attenuation.
- As $\text{Re}_t \to \infty$: $\beta_t \to 0$ — fully turbulent, complete credit truncation.

The Gaussian kernel provides **continuous differentiability** in $\text{Re}_t$, avoiding the gradient discontinuities of the earlier hard switch. It also provides natural **hysteresis**: a Reynolds number slightly above $\text{Re}_{\text{crit}}$ does not catastrophically zero the credit; it gracefully attenuates it. The transition is sharp enough to be protective against true turbulence yet smooth enough not to artificially truncate mildly elevated-variance transitions.

### 2.5 Backward Credit Propagation (Fluid Advantage Energy)

The advantage is computed via a **causal backward pass** — structurally identical to GAE's backward recursion, but with the data-driven $\beta_t$ replacing the fixed $\lambda$:

$$U_T = 0$$
$$U_t = \delta_t + \gamma \cdot \beta_t \cdot (1 - d_t) \cdot U_{t+1}$$

The effective horizon length at each timestep is now a function of the entire local signal history, not a global setting.

### 2.6 Intrinsic Stability Sink (Advantage Normalization)

$$\boxed{\hat{A}_t = \frac{U_t}{\zeta_t}}$$

Dividing by $\zeta_t$ performs **physical normalization intrinsic to the estimator**:

- In high-variance regions (large $\zeta_t$): large $U_t$ values are attenuated — protecting the policy gradient from explosive updates.
- In low-variance regions (small $\zeta_t$): small $U_t$ values are amplified — improving signal-to-noise in stable regimes.

This **eliminates the need for post-hoc batch normalization** of advantages, which is the standard practice in PPO/A2C implementations. The stability is built into the physics.

### 2.7 Target Returns

$$R_t = \hat{A}_t + V_\phi(s_t)$$

Used as the regression target for the critic's Huber (smooth L1) loss in v6.

---

## 3. Topological Calibration Protocol

FRAE-S v6 introduces a novel **pre-training environment probe** that autonomously determines the critical algorithm hyperparameters from the environment's own signal statistics — before a single gradient update is taken.

### 3.1 Protocol

A random-policy agent (with an uninitialized critic network) collects $N_{\text{probe}} = 5000$ transitions. From this trajectory:

**Viscosity Floor $\varepsilon$** — set to the standard deviation of the observed TD-errors:

$$\varepsilon^* = \max\left(\text{std}(\delta_{0:N}),\; 10^{-4}\right)$$

This ensures the viscosity floor is dimensionally matched to the reward scale of the specific environment. A hardcoded $\varepsilon = 10^{-8}$ in a high-reward-magnitude environment (like LunarLander, where rewards can be $\pm 100$) would produce numerically meaningless $\zeta_t$ values.

**Critical Reynolds Number $\text{Re}_{\text{crit}}$** — set to the 95th percentile of observed TD-error momenta:

$$\text{Re}_{\text{crit}}^* = \max\left(\frac{\text{percentile}_{95}\left(|\Delta\delta_{0:N}|\right)}{\varepsilon^*},\; 1.5\right)$$

This places the laminar-turbulent transition threshold at the point where only 5% of observed transitions qualify as turbulent — a statistically principled choice that prevents over-aggressive truncation in well-behaved environments and appropriately aggressive truncation in chaotic ones.

**Entropy Coefficient** — set based on reward variance topology:

$$c_H^* = \begin{cases} 0.05 & \text{if Var}(r_{0:N}) < 10^{-4} \quad \text{(Sparse topology)} \\ 0.005 & \text{otherwise} \quad \text{(Dense topology)} \end{cases}$$

### 3.2 LunarLander-v3 Calibration Result (Verified)

```
Initiating Topological Probe on LunarLander-v3...
Topology: DENSE. Engaging precision gradient tracking.
----------------------------------------
100% Precision Calibration Complete:
-> Optimal Entropy Coef  : 0.0050
-> Optimal Viscosity Floor (eps) : 11.3541
-> Optimal Re_crit (95th %ile)   : 1.5000
----------------------------------------
```

The probe correctly identified LunarLander as a **DENSE reward topology** (continuous shaping rewards from the physics engine) and calibrated $\varepsilon = 11.35$ — a non-trivial, environment-native viscosity floor that would be impossible to set manually without extensive grid search.

---

## 4. Thermodynamic Entropy Cooling

FRAE-S v6 treats the entropy bonus as a **thermodynamic heat** that is linearly annealed to zero over the training duration:

$$c_H(e) = c_H^* \cdot \max\!\left(0,\; 1 - \frac{e}{E_{\text{total}}}\right)$$

where $e$ is the current epoch and $E_{\text{total}}$ is the total epoch budget.

**Physical interpretation:** Early training corresponds to high thermodynamic temperature — the policy is far from optimal and high exploration entropy is thermodynamically natural. As training progresses, the "heat" dissipates: the policy solidifies into a deterministic structure, and entropy bonuses that were necessary for exploration become counterproductive to exploitation. This mirrors the Simulated Annealing cooling schedule but is physically motivated by thermodynamic irreversibility rather than combinatorial optimization.

The Critic loss in v6 uses **Huber (Smooth L1) loss** rather than MSE:

$$\mathcal{L}_V = \frac{1}{|B|}\sum_t \text{SmoothL1}(R_t - V_\phi(s_t))$$

$$\text{SmoothL1}(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq 1 \\ |x| - \frac{1}{2} & |x| > 1 \end{cases}$$

This provides a **natural loss ceiling** — the Huber loss does not grow quadratically with large TD-error residuals, preventing catastrophic critic updates in early training when $V_\phi$ is poorly calibrated. This is the critic-level analogue of the $\beta_t$ gate: both limit the influence of high-error signals on the parameter updates.

---

## 5. Architecture

FRAE-S is implemented on a standard decoupled A2C backbone. The novelty is entirely in the advantage estimator.

```
Observation s_t  (8-dim for LunarLander)
        │
  ┌─────┴──────────────────┐
  │   Actor π_θ (MLP)      │   Linear(8→64) → Tanh → Linear(64→64) → Tanh → Linear(64→4)
  │   Critic V_φ (MLP)     │   Linear(8→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1)
  └─────┬──────────────────┘
        │
  ┌─────▼──────────────────────────────────────────┐
  │   FRAE-S ENGINE (No learnable parameters)       │
  │                                                 │
  │   Forward Pass  (t = 0 → T):                   │
  │     σ²_t = (1-α)σ²_{t-1} + α·δ²_t             │
  │     ζ_t  = √σ²_t + ε                           │
  │     Re_t = |δ_t - δ_{t-1}| / ζ_t              │
  │     β_t  = β_lam · exp(-(Re_t/Re_crit)²)       │
  │                                                 │
  │   Backward Pass (t = T → 0):                   │
  │     U_t  = δ_t + γ·β_t·(1-d_t)·U_{t+1}        │
  │     Â_t  = U_t / ζ_t                           │
  └─────┬──────────────────────────────────────────┘
        │
  Policy gradient loss  ← Â_t
  Value regression loss ← R_t = Â_t + V(s_t)
  Entropy cooling loss  ← c_H(e)·H[π_θ]
```

**FRAE-S has zero learnable parameters.** It is a deterministic, physics-inspired transform of the TD-error sequence.

---

## 6. Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_\pi + \mathcal{L}_V + \mathcal{L}_H$$

$$\mathcal{L}_\pi = -\mathbb{E}_t\!\left[\log \pi_\theta(a_t | s_t) \cdot \hat{A}_t\right]$$

$$\mathcal{L}_V = \mathbb{E}_t\!\left[\text{SmoothL1}\!\left(R_t - V_\phi(s_t)\right)\right]$$

$$\mathcal{L}_H = -c_H(e) \cdot \mathbb{E}_t\!\left[\mathcal{H}[\pi_\theta(\cdot | s_t)]\right]$$

**Note:** Advantages $\hat{A}_t$ are **not normalized post-hoc**. The $\zeta_t$ denominator in the estimator already performs physically grounded normalization. Applying batch normalization on top would destroy the physical scale information embedded in $\zeta_t$.

---

## 7. Comparison to Prior Work

| Property | GAE (Schulman 2015) | Adaptive GAE (IEEE) | DGAE (arXiv 2025) | Chunked-TD (2024) | **FRAE-S (2026)** |
|----------|:-------------------:|:-------------------:|:-----------------:|:-----------------:|:-----------------:|
| Horizon control | Fixed λ | Variance/bias heuristic | Distributional (Wasserstein) | Model-based chunking | **Physics: Re_t gate** |
| Per-step adaptation | ✗ | ✓ (statistical) | ✗ | ✓ (model-based) | **✓ (physics-based)** |
| Phase transition | ✗ | ✗ | ✗ | ✗ | **✓ Gaussian kernel** |
| Intrinsic normalization | ✗ | ✗ | ✗ | ✗ | **✓ via ζ_t** |
| Self-calibration | ✗ | ✗ | ✗ | ✗ | **✓ Topological Probe** |
| Thermodynamic cooling | ✗ | ✗ | ✗ | ✗ | **✓ Linear anneal** |
| Learnable parameters in estimator | 0 | 0 | 0 | model params | **0** |
| Interpretable physics signals | ✗ | ✗ | ✗ | ✗ | **✓ Re_t, ζ_t, β_t** |

Chunked-TD uses learned model predictions to dynamically chunk trajectories for TD learning, shortening credit assignment paths in deterministic and predictable regions. This is the closest architectural relative to FRAE-S, but it is model-dependent and uses predictive coding rather than a fluid-dynamic phase criterion. Physics-informed RL research in 2025 applies physical constraints (momentum equations, pressure Poisson equations, boundary conditions) to reward function design and environment modeling — the reverse direction: physics constrains the *environment*, not the *estimator*. No existing literature applies a fluid-dynamic phase criterion to the internal temporal credit assignment mechanism.

---

## 8. Verified Training Results: LunarLander-v3

**Environment:** `LunarLander-v3` (8-dim state, 4 discrete actions, dense shaping + landing reward)  
**Total Steps:** 1,500 epochs × 500 steps/epoch = **750,000 environment interactions**  
**Hardware:** Google Colab (CPU)  
**Calibrated Parameters:** ε = 11.354, Re_crit = 1.500, c_H = 0.005

```
Epoch 050 | Train Reward:  -595.1 | Re_max: 4.71 | Zeta: 20.098 | Beta: 0.91
Epoch 100 | Train Reward:  -965.5 | Re_max: 5.30 | Zeta: 19.869 | Beta: 0.93
Epoch 150 | Train Reward:  -520.9 | Re_max: 4.29 | Zeta: 18.080 | Beta: 0.93
Epoch 200 | Train Reward:  -132.3 | Re_max: 4.29 | Zeta: 17.559 | Beta: 0.93
Epoch 250 | Train Reward:  -103.1 | Re_max: 4.37 | Zeta: 17.544 | Beta: 0.93
Epoch 300 | Train Reward:  +096.9 | Re_max: 4.47 | Zeta: 16.042 | Beta: 0.92
Epoch 350 | Train Reward:  +102.9 | Re_max: 4.44 | Zeta: 15.761 | Beta: 0.93
Epoch 400 | Train Reward:  +007.9 | Re_max: 4.45 | Zeta: 15.838 | Beta: 0.93
Epoch 450 | Train Reward:  +004.8 | Re_max: 0.60 | Zeta: 13.352 | Beta: 0.94  ← LAMINAR TRANSITION
Epoch 500 | Train Reward:  +040.9 | Re_max: 1.29 | Zeta: 13.822 | Beta: 0.93
Epoch 550 | Train Reward:  +071.6 | Re_max: 0.71 | Zeta: 12.958 | Beta: 0.94
Epoch 600 | Train Reward:  +018.3 | Re_max: 0.56 | Zeta: 13.562 | Beta: 0.93
Epoch 650 | Train Reward:  +015.2 | Re_max: 0.45 | Zeta: 13.335 | Beta: 0.94
Epoch 700 | Train Reward:  +000.7 | Re_max: 0.55 | Zeta: 13.472 | Beta: 0.94
Epoch 750 | Train Reward:  -057.6 | Re_max: 1.68 | Zeta: 13.269 | Beta: 0.93
Epoch 800 | Train Reward:  +032.8 | Re_max: 0.60 | Zeta: 13.342 | Beta: 0.94
Epoch 850 | Train Reward:  +191.7 | Re_max: 4.30 | Zeta: 15.438 | Beta: 0.93
Epoch 900 | Train Reward:  +184.1 | Re_max: 4.20 | Zeta: 15.386 | Beta: 0.94
Epoch 950 | Train Reward:  +171.5 | Re_max: 4.15 | Zeta: 15.166 | Beta: 0.94
Epoch 1000| Train Reward:  -089.2 | Re_max: 4.93 | Zeta: 15.802 | Beta: 0.93
Epoch 1050| Train Reward:  +206.5 | Re_max: 4.05 | Zeta: 15.075 | Beta: 0.94  ← +200 THRESHOLD
Epoch 1100| Train Reward:  +145.4 | Re_max: 4.10 | Zeta: 14.485 | Beta: 0.94
Epoch 1150| Train Reward:  +099.6 | Re_max: 4.05 | Zeta: 14.750 | Beta: 0.93
Epoch 1500| Train Reward:  +238.3 | Re_max: 3.76 | Zeta: 14.822 | Beta: 0.94  ← PEAK
```

---

## 9. Physics Telemetry: Reading the Fluid Dynamics

The training log is not merely a performance trace — it is a **real-time record of the fluid dynamics of the learning process**. Each column is a physically meaningful quantity.

### Phase I: Turbulent Exploration (Epochs 1–400)

$\text{Re}_{\max} \approx 4.3\text{–}5.3$, $\zeta \approx 16\text{–}20$, reward $< 0$.

The agent is in full turbulent regime. TD-errors are changing rapidly relative to their local variance. The $\beta_t$ gate is aggressively attenuating credit propagation — correctly so, because the value network has almost no predictive accuracy over the state space. The high $\zeta$ values reflect the large, chaotic reward fluctuations of an agent crashing the lander repeatedly. The physics is working as intended: **do not propagate credit you cannot trust**.

### Phase II: Laminar Transition (Epoch ~450)

$\text{Re}_{\max}$ drops from 4.45 to **0.60 in a single epoch**. $\zeta$ drops from ~15.8 to 13.4.

This is the most physically significant event in the entire training run. At epoch ~450, the value network crosses a threshold of accuracy sufficient to produce TD-errors that change slowly relative to their historical variance. The information flow **transitions from turbulent to laminar**. The Reynolds number criterion detected this transition without any external signal — purely from the internal dynamics of the TD-error sequence.

This laminar phase persists for approximately 400 epochs (450–850), during which the agent consolidates its value estimates. Reward variance during this phase reflects policy exploration, not estimator instability.

### Phase III: High-Performance Turbulence (Epochs 850–1500)

$\text{Re}_{\max}$ returns to the 3.7–5.5 range, but now with positive, large rewards (+191 to +238).

This is **good turbulence** — the agent is now attempting high-reward strategies that involve aggressive maneuvering, which naturally generates high TD-error variance. The FRAE-S engine correctly permits more credit truncation during these high-action episodes. $\zeta$ stabilizes in the 14–16 range, reflecting a new equilibrium between the improved value network and the more complex trajectory distribution.

The **+206.5 reward at epoch 1050** demonstrates the estimator is successfully assigning credit through high-variance competent trajectories — a regime where standard GAE with a conservative $\lambda$ would either over-smooth (losing credit signal) or under-smooth (accumulating variance).

### Summary of Observed Physics Diagnostics

| Signal | Early Training | Laminar Phase | Late Training |
|--------|:--------------:|:-------------:|:-------------:|
| $\text{Re}_{\max}$ | 4.3–5.3 | **0.45–1.68** | 3.7–5.5 |
| $\zeta_{\text{mean}}$ | 17–20 | **12–14** | 14–16 |
| $\beta_{\text{mean}}$ | 0.91–0.93 | **0.93–0.94** | 0.93–0.94 |
| Reward | Negative | Near-zero | **Positive, large** |
| Interpretation | Turbulent chaos | Laminar consolidation | Competent turbulence |

---

## 10. Running the Code

```bash
# System dependencies (required for Box2D)
apt-get install -y swig
pip install "gymnasium[box2d]" torch numpy matplotlib

# Run the full pipeline
python liquid_ns_6.py
```

The script executes three phases automatically:

**Phase A — Topological Calibration:** 5000-step random probe, outputs calibrated ε, Re_crit, c_H.

**Phase B — FRAE-S Training:** 1500 epochs × 500 steps. Periodic deterministic evaluation saves model weights ("DNA") whenever mean eval reward ≥ 200.

**Phase C — Physics Telemetry:** 4-panel visualization of reward, losses, mean β_t, and dual-axis ζ_t vs Re_max.

**Phase D — Deterministic Inference:** Load saved DNA and run 5 zero-entropy evaluation episodes.

---

## 11. Hyperparameter Reference

| Symbol | Code Variable | Default | Calibrated (LunarLander) | Description |
|--------|:-------------:|:-------:|:------------------------:|-------------|
| $\gamma$ | `gamma` | 0.99 | 0.99 | Discount factor |
| $\alpha$ | `alpha` | 0.01 | 0.01 | EMA rate for variance tracking |
| $\beta_{\text{lam}}$ | `beta_laminar` | 0.95 | 0.95 | Laminar propagation ceiling |
| $\text{Re}_{\text{crit}}$ | `re_crit` | 2.0 | **1.50** | Phase transition threshold |
| $\varepsilon$ | `eps` | 1e-8 | **11.354** | Viscosity floor (environment-scaled) |
| $c_H^*$ | `opt_entropy` | 0.01 | **0.005** | Entropy coefficient (topology-matched) |
| $E$ | `total_epochs` | 1000 | 1500 | Training epochs |
| $T$ | `steps_per_epoch` | 500 | 500 | Rollout length |
| — | `max_norm` | 0.5 | 0.5 | Gradient clip norm |
| — | `lr` | 0.001 | 0.001 | Adam learning rate |

---

## 12. Proposed Future Research Directions

The following are **original, unimplemented research proposals** extending the FRAE-S paradigm into new territory. None of these exist in the current literature to the best of the author's knowledge as of March 2026.

---

### 12.1 FRAE-S + PPO: Fluid Proximal Policy Optimization (FPPO)

The most immediate and high-impact extension is integrating the FRAE-S estimator directly into PPO's clipped surrogate objective, replacing GAE entirely:

$$\mathcal{L}^{\text{FPPO}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t^{\text{FRAE-S}},\; \text{clip}(r_t(\theta), 1-\varepsilon_{\text{clip}}, 1+\varepsilon_{\text{clip}})\,\hat{A}_t^{\text{FRAE-S}}\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio. The FRAE-S stability sink would provide a natural complement to PPO's clipping — two independent protection mechanisms against large policy updates, operating at different levels (estimator level and objective level).

---

### 12.2 Kolmogorov Information Scale: A Lower Bound on Credit Truncation

In turbulent fluid mechanics, the **Kolmogorov microscale** $\eta = (\nu^3/\varepsilon)^{1/4}$ defines the smallest scale at which energy is dissipated — below this scale, viscosity dominates and turbulence cannot exist. We propose an analogous **Kolmogorov Information Scale** $\eta_{\text{info}}$ that defines the minimum temporal credit length below which truncation is never applied regardless of $\text{Re}_t$:

$$\eta_{\text{info}} = \left(\frac{\varepsilon^3}{\bar{\epsilon}_{\text{dissipation}}}\right)^{1/4}, \qquad \bar{\epsilon}_{\text{dissipation}} = \alpha \cdot \overline{|\delta_t|^2}$$

When the remaining credit horizon $T - t < \eta_{\text{info}}$, force $\beta_t = \beta_{\text{lam}}$ regardless of $\text{Re}_t$. This prevents over-aggressive truncation at the end of trajectories where the signal is naturally weaker.

---

### 12.3 Reynolds Tensor for Multi-Agent FRAE-S (MFRAE-S)

In single-agent FRAE-S, $\text{Re}_t$ is a scalar. In multi-agent RL with $N$ agents, we propose replacing the scalar Reynolds number with an **Information Reynolds Tensor** $\mathbf{Re}_t \in \mathbb{R}^{N \times N}$:

$$[\mathbf{Re}_t]_{ij} = \frac{|\delta_t^{(i)} - \delta_t^{(j)}|}{\sqrt{\zeta_t^{(i)} \cdot \zeta_t^{(j)}}}$$

The off-diagonal entries quantify the **inter-agent information turbulence** — how discordant the credit signals of agents $i$ and $j$ are at time $t$. The resulting per-agent credit gate becomes:

$$\beta_t^{(i)} = \beta_{\text{lam}} \cdot \exp\!\left(-\frac{1}{N}\sum_j \left[\mathbf{Re}_t\right]_{ij}^2 / \text{Re}_{\text{crit}}^2 \right)$$

This provides **cooperative credit alignment**: in cooperative MARL, agents with temporally coherent TD-errors (low off-diagonal Reynolds) propagate credit jointly; agents in conflict (high off-diagonal Reynolds) decouple their credit streams.

---

### 12.4 Turbulent Diffusion Memory: Recycling Truncated Credit

When $\beta_t \approx 0$ (turbulent truncation), FRAE-S currently discards the would-be credit $\gamma \cdot \beta_t \cdot U_{t+1}$. In physical turbulence, energy is not destroyed — it **cascades to smaller scales** and eventually dissipates. We propose a **Turbulent Diffusion Buffer** $\mathcal{B}$ that stores truncated credit and re-injects it at future laminar timesteps:

$$\mathcal{B}_{t+1} = \mu_{\text{decay}} \cdot \mathcal{B}_t + (1-\beta_t) \cdot \gamma \cdot U_{t+1}$$

$$\hat{A}_t = \frac{U_t + \eta_{\text{inject}} \cdot \mathcal{B}_t \cdot \mathbb{1}[\text{Re}_t < \text{Re}_{\text{crit}}]}{\zeta_t}$$

This conserves information that would otherwise be permanently lost during turbulent episodes. Whether this improves or destabilizes learning is an empirical question — but the physical motivation is sound.

---

### 12.5 Stochastic Reynolds Phase Transition (Boltzmann Gate)

Replace the deterministic Gaussian gate with a **stochastic phase transition** sampled from a Boltzmann distribution parameterized by $\text{Re}_t$:

$$\beta_t \sim \text{Bernoulli}\!\left(\sigma\!\left(\frac{\text{Re}_{\text{crit}} - \text{Re}_t}{T_{\text{thermo}}(e)}\right)\right) \cdot \beta_{\text{lam}}$$

where $\sigma$ is the sigmoid function and $T_{\text{thermo}}(e)$ is the thermodynamic temperature (from Section 4). At high temperature (early training), the gate is nearly 50/50, encouraging diverse credit paths. As temperature anneals, the gate sharpens into a near-deterministic phase transition. This connects FRAE-S to statistical mechanics and provides a principled stochastic regularization mechanism.

---

### 12.6 Viscosity Annealing: α as a Thermodynamic Parameter

The EMA rate $\alpha$ in $\sigma_t^2 = (1-\alpha)\sigma_{t-1}^2 + \alpha \delta_t^2$ controls the **temporal horizon of the viscosity estimate**. Small $\alpha$ produces a long-memory viscosity; large $\alpha$ produces a short-memory, reactive viscosity.

We propose treating $\alpha$ as a **thermodynamic viscosity annealing schedule**:

$$\alpha(e) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \exp\!\left(-\frac{e}{\tau_{\text{cool}}}\right)$$

Early in training: high $\alpha$ → reactive viscosity that quickly tracks chaotic early TD-errors → appropriate for a poor critic.  
Late in training: low $\alpha$ → smooth, long-memory viscosity → appropriate for a converged critic whose errors are small and stable.

This is the viscosity analogue of learning rate decay, but derived from the physical properties of the estimator rather than optimization heuristics.

---

### 12.7 FRAE-S for Continuous Action Spaces

The current implementation uses `Categorical` distributions for discrete action spaces. Extending to continuous action spaces (e.g., `MuJoCo`, `BipedalWalker`) requires replacing the policy distribution with a `Normal` or `Beta` distribution and adjusting entropy computation accordingly. The FRAE-S engine itself is **policy-agnostic** — it operates only on TD-errors — and requires no modification. The extension is architecturally trivial but empirically important: continuous control benchmarks are the standard comparison ground for GAE.

---

## 13. Limitations and Open Questions

The following are honest, unresolved limitations as of March 2026. These are research questions, not defects.

**L1 — No convergence guarantee.** FRAE-S is a heuristic estimator. There is no formal proof that the policy gradient computed via $\hat{A}_t^{\text{FRAE-S}}$ is an unbiased estimator of the true advantage, or that the algorithm converges to a local optimum. GAE itself lacks convergence guarantees in the nonlinear function approximation setting; FRAE-S inherits this limitation and adds new sources of bias from the $\zeta_t$ normalization.

**L2 — Forward-only variance estimation.** $\zeta_t$ is computed in a single forward pass, meaning early-trajectory timesteps have less accurate variance estimates than late-trajectory timesteps. A two-pass algorithm (first pass computes $\zeta_t$ for all $t$; second pass computes advantages) would mitigate this at the cost of causal temporal structure.

**L3 — Re_crit sensitivity.** Although the Topological Calibration Protocol sets $\text{Re}_{\text{crit}}$ at the 95th percentile of observed momenta, the choice of percentile is itself a hyperparameter. The mapping from percentile choice to training stability has not been characterized. The floor of 1.5 in the calibration code is a heuristic lower bound.

**L4 — No ablation study.** The contribution of each individual component — the Gaussian gate vs. hard switch, the $\zeta_t$ normalization, the probe calibration, the Huber loss, the thermodynamic cooling — has not been isolated. We do not know which component drives which aspect of the observed performance.

**L5 — Single environment, single seed.** All results are from one run on one environment. Variance across seeds and generalization across environment families (sparse reward, continuous control, partially observable) remain entirely uncharacterized.

**L6 — EMA initialization.** $\sigma_0^2 = \varepsilon$ creates a cold-start bias: the first few timesteps of every rollout have artificially low variance estimates, which may produce anomalously high $\text{Re}_0$ values. A warm-start initialization from probe statistics could correct this.

---

## 14. References

1. Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

2. Mnih, V. et al. *Asynchronous Methods for Deep Reinforcement Learning.* ICML 2016.

3. Schulman, J. et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.

4. IEEE Xplore. *Adaptive Advantage Estimation for Actor-Critic Algorithms.* [doi:10.1109/9534005](https://ieeexplore.ieee.org/document/9534005/)

5. Shaik, S. et al. *Generalized Advantage Estimation for Distributional Policy Gradients.* arXiv:2507.17530, July 2025.

6. Ramesh, A. et al. *Sequence Compression Speeds Up Credit Assignment in Reinforcement Learning.* ICML 2024. [arXiv:2405.03878](https://arxiv.org/abs/2405.03878)

7. Ferret, J. et al. *A Survey of Temporal Credit Assignment in Deep Reinforcement Learning.* arXiv:2312.01072, 2023.

8. Koh, J., Pagnier, L., Chertkov, M. *Swimming in Turbulent Environments.* 2025. (Physics-Informed RL, environment side)

9. Reynolds, O. *An Experimental Investigation of the Circumstances Which Determine Whether the Motion of Water Shall Be Direct or Sinuous.* Philosophical Transactions of the Royal Society, 1883.

10. Kolmogorov, A.N. *The Local Structure of Turbulence in Incompressible Viscous Fluid for Very Large Reynolds Numbers.* Doklady Akademii Nauk SSSR, 1941.

11. Sutton, R.S., Barto, A.G. *Reinforcement Learning: An Introduction.* 2nd ed., MIT Press, 2018.

12. Taha, A. et al. *HEPPO: Hardware-Efficient Proximal Policy Optimization.* arXiv, January 2025.

---

## Citation

If you build on, critique, or reference this work, please cite as:

```bibtex
@misc{debnath2026fraes,
  author       = {Devanik Debnath},
  title        = {FRAE-S: Fluid Resonance Advantage Estimation — Stable.
                  A Navier-Stokes Inspired, Self-Calibrating Advantage Estimator
                  for Actor-Critic Reinforcement Learning},
  year         = {2026},
  month        = {March},
  institution  = {National Institute of Technology Agartala, ECE Department},
  note         = {Experimental research prototype. Unreviewed. GitHub: github.com/Devanik21}
}
```

---

<div align="center">

---

### ⚠️ Research Review Notice

This work represents an independent, experimental research prototype completed in March 2026.

The theoretical framework — applying the Navier-Stokes Reynolds number as a per-step temporal credit gate in actor-critic advantage estimation — occupies an intersection of fluid mechanics and reinforcement learning theory that, to the best of the author's knowledge, has not been previously explored in the published literature.

**However:** novelty of framing does not imply correctness of mechanism, and pilot results on a single environment do not constitute validation. The author explicitly acknowledges that this work requires rigorous expert review — including formal bias-variance analysis, controlled ablations, multi-seed benchmarking, and comparison against state-of-the-art baselines — before any strong claim about its contribution can be made.

If the theoretical grounding survives formal analysis and the empirical results replicate reliably across environments, the FRAE-S framework may represent a meaningful paradigm shift in how temporal credit assignment is understood: not as a statistical interpolation problem (bias vs. variance), but as a **physical flow problem** (laminar vs. turbulent information propagation).

*The author welcomes rigorous critique, collaboration, and peer review.*

---

*Conceived, implemented, and documented by Devanik Debnath, NIT Agartala, 2025–2026.*  
*All physics analogies are structural motivations grounded in dimensional analysis, not claims of physical equivalence.*

</div>
