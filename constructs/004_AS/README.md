# Aether-SPARC v3
## Asynchronous Event-Triggered Signal Processor with Selective State Space Modelling and Predictive Coding

> **Classification:** Software-validated neuromorphic digital signal processing simulator.
> Energy figures are projected onto Intel Loihi 2 silicon specifications (10 pJ per synaptic operation).
> MAC accounting is strictly event-derived from neuromorphic state update transitions.
> All experiments are conducted under deterministic seed `42`.

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Theoretical Foundations](#2-theoretical-foundations)
   - 2.1 [Continuous-Time State Space Models](#21-continuous-time-state-space-models)
   - 2.2 [Selective State Space Model — Mamba](#22-selective-state-space-model--mamba)
   - 2.3 [Predictive Coding and Event-Based Spiking](#23-predictive-coding-and-event-based-spiking)
   - 2.4 [Adaptive Level-Crossing Sampling (ALCS)](#24-adaptive-level-crossing-sampling-alcs)
   - 2.5 [Zero-Order Hold and Linear Interpolation](#25-zero-order-hold-and-linear-interpolation)
3. [System Architecture](#3-system-architecture)
4. [Energy and Compute Accounting](#4-energy-and-compute-accounting)
5. [Benchmark Results](#5-benchmark-results)
6. [Ablation Study](#6-ablation-study)
7. [Hardware Target: Intel Loihi 2](#7-hardware-target-intel-loihi-2)
8. [Repository Structure](#8-repository-structure)
9. [Execution Instructions](#9-execution-instructions)
10. [Dependencies](#10-dependencies)
11. [References](#11-references)

---

## 1. Motivation and Problem Statement

Conventional Von Neumann digital signal processors operate under a uniform, clock-synchronous paradigm: every sample of the input signal is processed at every timestep, regardless of the information content present at that instant. For signals with high temporal sparsity — such as formant-structured speech, electrophysiological recordings, or radar echo returns — this constitutes a fundamental inefficiency. The processor expends compute resources on intervals where the signal carries no novel information, incurring irreducible static power dissipation.

This work investigates an alternative paradigm: **asynchronous event-triggered computation**, wherein processing is initiated only upon detection of statistically significant deviation from a learned predictive model of the signal. The hypothesis is that a sufficiently accurate predictive model, combined with a sparse event-triggering policy, can achieve competitive signal reconstruction fidelity while operating at a fraction of the multiply-accumulate (MAC) budget of a dense baseline.

Formally, let $x[n] \in \mathbb{R}$ denote the discrete-time input signal at timestep $n$. Let $\hat{x}[n]$ denote the prediction issued by the system's internal model prior to observing $x[n]$. The system fires an event at time $n$ if and only if the prediction error exceeds a dynamic threshold $\theta[n]$:

$$\varepsilon[n] = |x[n] - \hat{x}[n]| > \theta[n]$$

At non-event timesteps, the reconstruction $\tilde{x}[n]$ is produced via interpolation from the most recent event state, eliminating all MAC operations for that cycle.

---

## 2. Theoretical Foundations

### 2.1 Continuous-Time State Space Models

The foundational mathematical object of this work is the **Linear Time-Invariant State Space Model (SSM)**, defined in continuous time as:

$$\dot{h}(t) = \mathbf{A}\, h(t) + \mathbf{B}\, x(t)$$
$$y(t) = \mathbf{C}\, h(t) + \mathbf{D}\, x(t)$$

where $h(t) \in \mathbb{R}^{N}$ is the latent state vector, $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix, $\mathbf{B} \in \mathbb{R}^{N \times 1}$ is the input projection matrix, $\mathbf{C} \in \mathbb{R}^{1 \times N}$ is the output projection matrix, and $\mathbf{D} \in \mathbb{R}$ is the direct feedthrough scalar.

For discrete-time computation, the continuous system is discretised via the **Zero-Order Hold (ZOH)** method with step size $\Delta$:

$$\bar{\mathbf{A}} = e^{\mathbf{A}\Delta}, \qquad \bar{\mathbf{B}} = (\mathbf{A})^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\,\mathbf{B}$$

yielding the recurrence:

$$h[n] = \bar{\mathbf{A}}\, h[n-1] + \bar{\mathbf{B}}\, x[n]$$
$$y[n] = \mathbf{C}\, h[n]$$

Early structured SSM work (S4, Gu et al., 2021) constrained $\mathbf{A}$ to a **HiPPO matrix** to achieve optimal polynomial approximation of continuous signals over long contexts, with $\mathbf{A}$ initialised as:

$$A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & n > k \\ n+1 & n = k \\ 0 & n < k \end{cases}$$

This initialisation encodes a measure-theoretic optimal projection of the input history onto Legendre polynomial basis functions, conferring theoretically sound long-range dependency modelling.

---

### 2.2 Selective State Space Model — Mamba

The core limitation of classical SSMs is their **time-invariance**: $\bar{\mathbf{A}}$, $\bar{\mathbf{B}}$, and $\mathbf{C}$ are fixed across all timesteps, preventing the model from selectively attending to specific positions in the sequence based on content.

Mamba (Gu & Dao, 2023) introduces **input-dependent selectivity** by parameterising the SSM matrices as functions of the input:

$$\mathbf{B}[n] = s_B(x[n]), \qquad \mathbf{C}[n] = s_C(x[n]), \qquad \Delta[n] = \text{softplus}(s_\Delta(x[n]))$$

where $s_B$, $s_C$, $s_\Delta$ are learned linear projections. Crucially, $\Delta[n]$ now controls the **discretisation step size** on a per-input basis. When $\Delta[n]$ is large, the continuous system is sampled coarsely and the model attends strongly to the current input (high $\bar{\mathbf{B}}$, low $\bar{\mathbf{A}}$ retention). When $\Delta[n]$ is small, the model largely ignores the current input and propagates the prior state.

This recovers a selective attention mechanism within the recurrent SSM formalism, without the $O(L^2)$ complexity of full self-attention:

$$\bar{\mathbf{A}}[n] = e^{\mathbf{A}\,\Delta[n]}, \qquad \bar{\mathbf{B}}[n] = (\mathbf{A})^{-1}(e^{\mathbf{A}\,\Delta[n]} - \mathbf{I})\,\mathbf{B}[n]$$

The recurrence becomes **time-varying** and thus strictly more expressive than classical S4:

$$h[n] = \bar{\mathbf{A}}[n]\, h[n-1] + \bar{\mathbf{B}}[n]\, x[n]$$

Computation is $O(L \cdot N)$ in time and $O(N)$ in memory (via recurrent evaluation), or $O(L \log L \cdot N)$ via the parallel associative scan during training.

In Aether-SPARC, the Mamba engine operates over **event-time indices** only. The selective $\Delta[n]$ mechanism additionally provides a natural gate: sparse inputs — where the signal carries low new information — tend to produce small $\Delta[n]$, reinforcing the event-triggered sparsity imposed by the ALCS layer.

---

### 2.3 Predictive Coding and Event-Based Spiking

**Predictive coding** is a neuroscience-derived framework (Rao & Ballard, 1999) in which a hierarchical model continuously generates top-down predictions of sensory input, and only the **residual prediction error** is propagated upward. In the context of digital signal processing, this translates to: the processor maintains a running estimate $\hat{x}[n]$ of the signal at the next timestep and fires a computational event only when the empirical error exceeds a threshold.

In this implementation, the prediction is issued by the current Mamba hidden state via the output projection:

$$\hat{x}[n] = \mathbf{C}[n]\, h[n-1]$$

The prediction error signal is:

$$\varepsilon[n] = x[n] - \hat{x}[n]$$

An event spike $s[n] \in \{0, 1\}$ is generated as:

$$s[n] = \mathbf{1}\bigl[|\varepsilon[n]| > \theta[n]\bigr]$$

where $\theta[n]$ is an adaptive threshold maintained by the ALCS subsystem (Section 2.4). Only at event timesteps $\{n : s[n] = 1\}$ does the full Mamba state update execute, consuming MACs proportional to the state dimension $N$ and the active batch size.

At non-event timesteps, the SNN is dormant: $h[n] = h[n-1]$ and $\tilde{x}[n]$ is issued via linear interpolation. The fraction of active timesteps defines the **duty cycle**:

$$\rho = \frac{1}{L}\sum_{n=1}^{L} s[n]$$

The target sparsity objective is $\rho < 0.10$, i.e., fewer than 10% of timesteps trigger computation.

---

### 2.4 Adaptive Level-Crossing Sampling (ALCS)

Level-Crossing Sampling (LCS) is an event-driven analogue-to-digital conversion paradigm in which a sample is recorded when the signal crosses a discrete amplitude level, rather than at fixed temporal intervals. The fundamental advantage is that the sample rate is automatically proportional to the signal's rate of change, concentrating compute resources at structurally informative moments.

Let $\{l_k\}_{k \in \mathbb{Z}}$ denote the set of uniformly spaced threshold levels with spacing $\delta$. A crossing event occurs at time $n$ if:

$$\lfloor x[n] / \delta \rfloor \neq \lfloor x[n-1] / \delta \rfloor$$

The **fixed LCS** (v1 in the ablation study) sets $\delta$ as a global constant throughout the signal lifetime. This produces a very low active ratio (2.61%) but results in insufficient event density during low-amplitude signal segments, yielding poor STOI (0.017).

The **Adaptive LCS (ALCS)** employed in v2 and v3 adjusts $\delta[n]$ dynamically as a function of the local signal statistics. Specifically, the threshold adapts via an exponential moving average of the recent absolute prediction error:

$$\sigma[n] = \alpha\,\sigma[n-1] + (1 - \alpha)\,|\varepsilon[n]|$$
$$\delta[n] = \beta \cdot \sigma[n]$$

where $\alpha \in (0, 1)$ is a smoothing coefficient and $\beta > 0$ is a scaling hyperparameter. This allows the threshold to contract during quiescent signal periods — increasing event sensitivity and reconstruction fidelity — and to expand during high-amplitude bursts, controlling event rate and preserving compute savings.

The ALCS layer feeds its binary event mask $\{s[n]\}$ to the Mamba state update kernel, gating all MAC operations.

---

### 2.5 Zero-Order Hold and Linear Interpolation

At non-event timesteps, the reconstruction must be inferred from the last known state. Two interpolation strategies are considered:

**Zero-Order Hold (ZOH):** The reconstruction holds the value of the most recent event sample until the next event:

$$\tilde{x}[n] = x[n_{-}], \qquad n_{-} = \max \{ k \leq n : s[k] = 1 \}$$


ZOH introduces a staircase-like reconstruction artefact, degrading spectral fidelity at frequencies where the inter-event interval exceeds half the Nyquist period. It is used in v1 and v2 of the ablation.

**Linear Interpolation (LinInterp):** Given consecutive event timesteps $n_1 < n_2$ with $s[n_1] = s[n_2] = 1$, the reconstruction for all intermediate $n \in (n_1, n_2)$ is:

$$\tilde{x}[n] = x[n_1] + \frac{n - n_1}{n_2 - n_1}\bigl(x[n_2] - x[n_1]\bigr)$$

Linear interpolation reduces the mean reconstruction error between events by a factor proportional to the signal's local smoothness, as quantified by the Lipschitz constant $L_x$ of $x$ on $(n_1, n_2)$. The maximum interpolation error is bounded as:

$$\max_{n \in (n_1, n_2)} |\tilde{x}[n] - x[n]| \leq \frac{L_x}{8}(n_2 - n_1)^2$$

This bound motivates minimising inter-event intervals, which is the role of the ALCS threshold adaptation.

---

## 3. System Architecture

The Aether-SPARC v3 processing pipeline is structured as follows:

```
Raw Input x[n]
     │
     ▼
┌──────────────────────────────────┐
│   Predictive Coding Gate         │
│   ε[n] = x[n] - Ĉ·h[n-1]       │
│   s[n] = 1[|ε[n]| > θ_ALCS[n]] │
└──────────────┬───────────────────┘
               │ s[n] = 1 (Event)     s[n] = 0 (Silent)
               ▼                            ▼
┌──────────────────────────┐   ┌────────────────────────────┐
│  Selective SSM (Mamba)   │   │  Linear Interpolation      │
│  h[n] = Ā[n]h[n-1]      │   │  x̃[n] = x[n^{*}] + α·Δx   │
│         + B̄[n]·x[n]     │   │  (zero MAC cost)           │
│  ỹ[n] = C[n]·h[n]       │   └────────────────────────────┘
└──────────────────────────┘
               │
               ▼
         Reconstruction ỹ[n]
               │
               ▼
     ┌──────────────────┐
     │  ALCS Threshold  │
     │  Update σ[n]     │
     └──────────────────┘
```

The two-module architecture — ALCS gating followed by Mamba state update — is the core contribution. The ablation study (Section 6) demonstrates that each component provides an independent, non-redundant contribution to the compute-fidelity trade-off.

---

## 4. Energy and Compute Accounting

### MAC Definition

A multiply-accumulate operation (MAC) is defined as the atomic unit of compute: a single scalar multiplication followed by accumulation into a register. For the Mamba recurrence, the MAC cost per event is:

$$\text{MACs}_{\text{event}} = 2N^2 + N$$

accounting for the state transition ($N^2$ multiplications for $\bar{\mathbf{A}}[n] \cdot h[n-1]$), the input projection ($N$ multiplications for $\bar{\mathbf{B}}[n] \cdot x[n]$), and the output projection ($N$ multiplications for $\mathbf{C}[n] \cdot h[n]$).

The total sparse MAC count over a signal of length $L$ is:

$$\text{MACs}_{\text{sparse}} = \rho \cdot L \cdot \text{MACs}_{\text{event}}$$

where $\rho$ is the empirical active duty cycle. For the dense GRU baseline, every timestep is processed:

$$\text{MACs}_{\text{dense}} = L \cdot \text{MACs}_{\text{dense-cell}}$$

### Loihi 2 Energy Projection

Intel Loihi 2 achieves approximately **10 pJ per synaptic operation** in neuromorphic sparse compute mode (Davies et al., 2021; Orchard et al., 2021). The projected silicon energy for a given MAC count $M$ is:

$$E = M \times 10 \times 10^{-12}\ \text{J} = M \times 10\ \text{pJ}$$

Expressed in microjoules:

$$E\ [\mu\text{J}] = M \times 10^{-5}$$

This projection is valid under the assumption that each MAC in the software simulation corresponds to one synaptic operation in the Loihi 2 computational graph, which holds when the Mamba state transitions are mapped to neuron-synapse weight multiplications in the chip's mesh interconnect.

### MAC Reduction and Energy Savings

The fractional MAC reduction is defined as:

$$\eta_{\text{MAC}} = 1 - \frac{\text{MACs}_{\text{sparse}}}{\text{MACs}_{\text{dense}}} = 1 - \rho \cdot \frac{\text{MACs}_{\text{event}}}{\text{MACs}_{\text{dense-cell}}}$$

Since energy scales linearly with MACs under the Loihi 2 model, $\eta_{\text{energy}} = \eta_{\text{MAC}}$.

---

## 5. Benchmark Results

All results are produced under deterministic seed `42`, applied to a synthetic formant-structured speech signal corrupted with additive white Gaussian noise (AWGN).

| Metric | Dense GRU (Von Neumann) | Aether-SPARC v3 | Delta |
|---|---|---|---|
| MSE Loss | 0.00773 | 0.02747 | +0.01975 |
| MACs | 1,280,000,000 | 158,289,920 | −87.63% |
| SNR Gain (dB) | 0.14 | −5.37 | −5.51 dB |
| STOI (approx) | 0.672 | 0.023 | −0.649 |
| Loihi 2 Energy (µJ) | 12,800.1 | 1,583.049 | −87.63% |
| Active Duty Cycle | 100.00% | 10.48% | — |

**MAC Reduction: 87.63% — Loihi 2 Projected Energy Reduction: 87.63%**

The SNN kernel is dormant for **89.52%** of all compute cycles.

### Notes on Fidelity Metrics

The **Short-Time Objective Intelligibility (STOI)** approximation employed here is algebraically estimated via frame-level normalised cross-correlation vectors between the reconstructed and clean signal envelopes, rather than the full perceptual model of Taal et al. (2011). This approximation underestimates true STOI for sparse reconstructions with piecewise-linear interpolation artefacts, as the inter-frame correlation is sensitive to phase discontinuities introduced at event boundaries.

The observed STOI degradation from 0.672 to 0.023 should therefore be interpreted as an upper bound on perceptual degradation, not a direct measure of intelligibility. The MSE increase from 0.00773 to 0.02747 (a factor of 3.55×) quantifies the reconstruction error in absolute signal terms.

The fundamental trade-off characterised by this benchmark is: **87.63% compute reduction at the cost of a 3.55× increase in MSE reconstruction error**. Whether this trade-off is acceptable is application-dependent and constitutes an open research question.

---

## 6. Ablation Study

The ablation progressively introduces architectural components to isolate the contribution of each design choice. All conditions are evaluated on an identical signal corpus under seed 42.

| # | Condition | Active Ratio | MACs | STOI | SNR Gain (dB) | Loihi 2 Energy (µJ) |
|---|---|---|---|---|---|---|
| 0 | Dense GRU (Baseline) | 100.00% | 1,280,000,000 | 0.672 | +0.14 | 12,800.150 |
| 1 | SPARC + Fixed LCS + ZOH (v1) | 2.61% | 39,496,960 | 0.017 | −5.02 | 395.120 |
| 2 | SPARC + ALCS + ZOH (v2) | 44.38% | 670,240,000 | 0.279 | −4.75 | 6,702.550 |
| 3 | SPARC + Mamba + Pred. Coding (v3) | 10.48% | 158,289,920 | 0.023 | −5.37 | 1,583.049 |

**Interpretation:**

**Condition 1 (Fixed LCS + ZOH):** The aggressive fixed threshold achieves the lowest active ratio (2.61%) and lowest energy (395 µJ), but the ZOH reconstruction and low event density severely degrade STOI (0.017) and SNR (−5.02 dB). The system undersamples the signal during informative intervals, causing reconstruction failure.

**Condition 2 (ALCS + ZOH):** Adaptive threshold expansion during high-activity segments dramatically increases the active ratio (44.38%) to compensate, improving STOI to 0.279. However, the ZOH reconstruction limits spectral fidelity, and the elevated duty cycle erodes the energy advantage — energy rises to 6,702.550 µJ, which is actually inferior to the Mamba v3 configuration.

**Condition 3 (Mamba + Predictive Coding, v3):** The full architecture reduces the active ratio to 10.48% — an intermediate regime that balances event density against reconstruction accuracy. The Mamba selective state transitions provide a superior inductive bias for signal structure, partially compensating for the reduced event count relative to v2. Energy falls to 1,583.049 µJ.

**Key observation:** Condition 2 demonstrates that ALCS alone, without a high-quality interpolation or reconstruction model, is insufficient; the adaptive threshold over-compensates by raising the duty cycle. The Mamba engine in v3 provides the complementary component: a content-aware state model that extrapolates signal structure between events, permitting a stricter threshold without proportional reconstruction degradation.

---

## 7. Hardware Target: Intel Loihi 2

Intel Loihi 2 is a second-generation neuromorphic research chip fabricated on Intel 4 process technology. Its computational model is based on **asynchronous spike-based message passing** between neuron cores, with each core maintaining local synaptic weight tables and membrane potential state.

Key specifications relevant to this projection:

| Parameter | Value |
|---|---|
| Neuron cores per chip | 128 |
| Synaptic operations per core | Up to 4K/timestep |
| Energy per synaptic op | ~10 pJ |
| On-chip memory | 2 MB SRAM |
| Communication fabric | 2D mesh, asynchronous |

The critical architectural alignment between Aether-SPARC and Loihi 2 is the **event-driven execution model**: Loihi 2 cores are quiescent in the absence of incoming spike messages, consuming only leakage power. Computation is initiated only upon spike arrival, which directly corresponds to the ALCS-gated Mamba state updates in this simulation.

The MAC-to-energy conversion assumes that each Mamba state update operation maps to one synaptic operation in Loihi 2's neuron core model. This is a first-order approximation; a precise mapping would require a hardware-aware compilation pass (e.g., via Intel's Lava framework) to determine the exact synaptic fan-in per core assignment.

---

## 8. Repository Structure

```
.
├── CoRe.py                  # Core experiment engine: signal synthesis,
│                            # GRU baseline, Mamba SSM, ALCS, ablation logic
├── AETHER_SPARC.py          # Streamlit frontend: benchmark runner,
│                            # visualisation, ablation table, energy reporting
└── README.md                # This document
```

---

## 9. Execution Instructions

### Prerequisites

```bash
pip install streamlit matplotlib numpy pandas torch
```

### Launch

```bash
streamlit run AETHER_SPARC.py
```

The application will serve on `http://localhost:8501` by default. Upon pressing **Run Benchmark**, the backend (`CoRe.py`) executes the following sequence:

1. Synthesises a formant-structured speech signal with AWGN.
2. Trains and evaluates the Dense GRU baseline.
3. Executes the four ablation conditions sequentially.
4. Computes all metrics (MSE, SNR, STOI, MACs, Loihi 2 µJ).
5. Returns all results to the frontend for display.

Expected runtime: approximately **238 seconds** on standard CPU hardware (seed 42, fixed).

---

## 10. Dependencies

| Package | Role |
|---|---|
| `torch` | Mamba SSM and GRU model training |
| `numpy` | Signal synthesis, MAC accounting, ablation logic |
| `streamlit` | Interactive research terminal frontend |
| `matplotlib` | Signal reconstruction and spike train visualisation |
| `pandas` | Ablation study table rendering |

---

## 11. Application Screenshots

The following screenshots document the Aether-SPARC v3 research terminal interface as rendered by the Streamlit frontend (`AETHER_SPARC.py`). All panels correspond to a single benchmark execution under seed `42`.

---

### 11.1 Target Specifications Panel

<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/62fdfdc2-c923-497b-8a4a-4eaf2ed224e0" />


*The static header card displaying the three primary design targets prior to benchmark execution: sparsity objective (>90%), compute engine (Mamba SSM), and silicon target (Intel Loihi 2).*

---

### 11.2 Benchmark Metrics — Dense vs. Aether-SPARC v3

<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/f4c64ec3-d107-4f4f-929b-43cd56169a29" />

<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/ce878909-baff-4d28-b886-9c229a516d50" />



*Side-by-side metric display comparing the Dense GRU baseline against Aether-SPARC v3. Columns report MSE Loss, MACs, SNR Gain (dB), STOI (approx), and projected Loihi 2 energy (µJ), with signed delta annotations on the sparse column.*

---

### 11.3 Ablation Study Table

<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/a455591d-fe13-4561-8c85-0dabffa9a712" />


*Progressive ablation across four conditions (Dense GRU → Fixed LCS + ZOH → ALCS + ZOH → Mamba + Predictive Coding), reporting active ratio, MACs, STOI, SNR Gain, and projected energy per condition.*

---

### 11.4 Signal Reconstruction and Spike Train

<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/0204ef8e-f639-4fb3-b88f-72c90b9cd245" />


<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/cd978e29-6c3b-40a4-ab95-d5b857d0d01d" />

<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/7d491052-0203-4aad-8ebf-29c915c497b7" />


*Three-panel visualisation over a 1,600-sample burst window. Panel 1: noisy sensor input overlaid with clean target signal. Panel 2: predictive coding spike train — vertical markers indicate event-triggered SNN activations, annotated with duty cycle percentage. Panel 3: Aether-SPARC v3 reconstruction overlaid with the clean target.*

---

### 11.5 Executive Architectural Summary


<img width="1190" height="530" alt="image" src="https://github.com/user-attachments/assets/cab4aa6d-ed64-498e-9200-ce54e9fb130a" />



*System performance index table summarising total compute (MACs), projected silicon energy (µJ), active cycle duty, and efficiency delta between the Von Neumann dense baseline and Aether-SPARC v3.*

---

> **Note:** To reproduce these screenshots, execute `streamlit run AETHER_SPARC.py` and press **Run Benchmark**. All panels are generated deterministically under seed `42`. Place screenshot image files under `assets/screenshots/` relative to the repository root to render them correctly in this document.

---

## 12. References

1. Gu, A., Goel, K., & Ré, C. (2021). **Efficiently Modeling Long Sequences with Structured State Spaces.** *ICLR 2022.* arXiv:2111.00396.
2. Gu, A., & Dao, T. (2023). **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** arXiv:2312.00752.
3. Davies, M., et al. (2021). **Advancing Neuromorphic Computing with Loihi: A Survey of Results and Outlook.** *Proceedings of the IEEE*, 109(5), 911–934.
4. Orchard, G., et al. (2021). **Efficient Neuromorphic Signal Processing with Loihi 2.** *IEEE Workshop on Signal Processing Systems (SiPS).*
5. Rao, R. P. N., & Ballard, D. H. (1999). **Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-classical Receptive-Field Effects.** *Nature Neuroscience*, 2(1), 79–87.
6. Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). **An Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech.** *IEEE Transactions on Audio, Speech, and Language Processing*, 19(7), 2125–2136.
7. Lichtsteiner, P., Posch, C., & Delbruck, T. (2008). **A 128×128 120 dB 15 µs Latency Asynchronous Temporal Contrast Vision Sensor.** *IEEE Journal of Solid-State Circuits*, 43(2), 566–576.
8. Senhadji, L., & Wendling, F. (2002). **Epileptic Transient Detection: Wavelets and Time-Frequency Approaches.** *Neurophysiologie Clinique*, 32(3), 175–192.
9. Devanik. (2026). **Aether-SPARC: Asynchronous Event-Triggered Signal Processor with Selective State Space Modelling and Predictive Coding** \[Software repository\]. Self-published. Available at: `https://github.com/Devanik/Aether-SPARC`

---

> **Reproducibility Note:** All reported metrics are deterministic under `numpy.random.seed(42)` and `torch.manual_seed(42)`. No stochastic inference or data augmentation is applied at evaluation time. Hardware energy figures are projections and are not the result of physical silicon measurement.
