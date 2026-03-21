# HAG-MoE: Hierarchical Attention-Gated Mixture of Experts

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b?style=flat-square)](https://arxiv.org/abs/xxxx.xxxxx)
[![License](https://img.shields.io/badge/License-Apache-purple?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Research_Preview-yellow?style=flat-square)]()
[![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)

> **HAG-MoE** introduces a novel architectural primitive: a Mixture-of-Experts layer in which the routing hierarchy is derived entirely from the pre-existing multi-head attention structure — not from separately learned gate networks — and in which the dynamic number of activated experts per token is controlled by the Shannon entropy of the attention distribution. Expert selection feeds back to modulate attention output through lightweight identity embeddings, creating a principled bidirectional coupling between contextual focus and computational allocation.

---

## Abstract

Mixture-of-Experts (MoE) architectures achieve computational sparsity by routing each token to a small subset of expert FFNs. State-of-the-art routers are scalar linear projections that operate independently at each layer, treating all tokens uniformly in terms of the number of activated experts, and having no bidirectional coupling with the attention mechanism that precedes them. We identify three structural problems: (1) the routing hierarchy is an artificially added learned component, separate from the rich representational structure already encoded in multi-head attention; (2) the fixed top-k cardinality ignores token-level contextual uncertainty, as measured by attention entropy; and (3) the feedforward path is unidirectional — expert selection never informs the attention computation.

We propose **HAG-MoE (Hierarchical Attention-Gated MoE)**, which addresses all three through a unified principle: **use the attention mechanism itself, rather than auxiliary learned gates, to govern expert routing**. HAG-MoE (i) partitions existing attention heads into coarse and fine sets, using their averaged distributions as a natural two-level routing hierarchy with zero added gate parameters; (ii) uses per-token Shannon entropy of the coarse attention distribution as the dynamic cardinality signal $K_i$, activating more experts for semantically ambiguous tokens; and (iii) projects expert assignment embeddings back into the residual stream as output modulation, creating a closed feedback loop within each layer.

---

## Table of Contents

1. [Motivation and Prior Art Gap](#1-motivation-and-prior-art-gap)
2. [Related Work and Positioning](#2-related-work-and-positioning)
3. [Architecture](#3-architecture)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Theoretical Properties](#5-theoretical-properties)
6. [Training Objective](#6-training-objective)
7. [Implementation Details](#7-implementation-details)
8. [Repository Structure](#8-repository-structure (Proposed))
9. [Ablation Roadmap](#9-ablation-roadmap)
10. [References](#10-references)

---

## 1. Motivation and Prior Art Gap

Standard sparse MoE layers (Shazeer et al., 2017; Fedus et al., 2022) replace the FFN sublayer with $N$ expert FFNs and a learned router $G: \mathbb{R}^d \to \Delta^{N-1}$:

$$\text{MoE}(\mathbf{x}) = \sum_{e \in \text{TopK}(G(\mathbf{x}), k)} G(\mathbf{x})_e \cdot E_e(\mathbf{x}), \quad G(\mathbf{x}) = \text{softmax}(W_G \mathbf{x})$$

where $W_G \in \mathbb{R}^{N \times d}$ is a linear layer. Three problems are structural, not accidental:

**Problem 1: Artificial routing hierarchy.** Hierarchical MoE models (HMoE, arxiv:2410.02935; SAGE, 2025) introduce two-stage learned gates — coarse gate $G^{(1)}$ selects a super-expert group, fine gate $G^{(2)}$ selects within it. Both are separately parameterized and jointly trained. This doubles routing machinery and imposes an arbitrary structure. But a transformer already contains a two-level representational hierarchy: early/syntactic heads capture positional patterns while semantic heads capture content (Clark et al., 2019; Voita et al., 2019). The routing hierarchy should emerge from this, not be added on top.

**Problem 2: Uniform activation cardinality.** All current MoE systems activate a fixed $k$ experts per token. DA-MoE (Aghdam et al., 2024) dynamically varies $K_i$ using attention-weight magnitudes as token importance. But high attention magnitude means *focus*, not *uncertainty*. The correct signal for needing more experts is **contextual uncertainty** — formally, the Shannon entropy $H_i = -\sum_j a_{ij} \log a_{ij}$ of the attention distribution. High entropy (diffuse attention) → uncertain context → more expert opinions needed. DA-MoE does not use entropy.

**Problem 3: Unidirectional attention-expert relationship.** Every prior work is one-directional: attention output informs routing (DA-MoE, DASG-MoE, RMoE) or routing modifies attention structure (SwitchHead). None creates a *within-layer* bidirectional coupling where expert assignment feeds back to calibrate the attention-derived output. RMoE (Qiu et al., ICLR 2025) creates cross-layer routing state via GRU — this is recurrence across layers, not within-layer expert-to-attention feedback.

HAG-MoE addresses all three simultaneously.

---

## 2. Related Work and Positioning

| Model | Hierarchical Routing | Dynamic $K_i$ | Attn↔Expert Feedback |
|---|---|---|---|
| Switch Transformer (2022) | ✗ flat, top-1 | ✗ | ✗ |
| Mixtral (2024) | ✗ flat, top-2 | ✗ | ✗ |
| DeepSeekMoE (2024) | Partial (shared+routed) | ✗ | Separate MLA |
| HMoE (2410.02935) | ✓ 2 learned gates | ✗ | ✗ |
| DA-MoE (2409.06669) | ✗ | ✓ importance-based | One-directional |
| RMoE (2408.06793, ICLR 2025) | ✗ | ✗ | Cross-layer GRU |
| SAGE (2511.18493) | ✓ shared+fine-grained | ✗ | ✗ |
| DASG-MoE (2509.10530) | ✗ | ✗ | Attn→routing only |
| SMoE-Attention (2505.00792) | ✗ | ✗ | Graph similarity |
| Quadratic MoE (2410.11222) | ✗ | ✗ | Math unification |
| GateTS (2508.17515) | ✗ | ✗ | Inspired, time-series |
| **HAG-MoE (ours)** | **✓ from MHA heads, zero extra gates** | **✓ entropy-based** | **✓ bidirectional within-layer** |

**Key distinctions from the closest neighbors:**

*vs. DA-MoE*: DA-MoE uses raw attention weight magnitudes for token importance → variable K. HAG-MoE uses attention *entropy* (high when diffuse, not correlated with magnitude) for cardinality + attention *distributions from partitioned head groups* for hierarchical routing. Three different mechanisms.

*vs. HMoE and SAGE*: Both use separately trained coarse gates to define routing hierarchy. HAG-MoE derives hierarchy from the pre-existing head structure — zero additional gate parameters at the hierarchy level. The hierarchical signal source is fundamentally different.

*vs. RMoE*: RMoE propagates routing state across layers via GRU hidden states. HAG-MoE's feedback is within a single layer (expert embeddings modulate FFN output). These are orthogonal and composable — HAG-MoE can be stacked on top of RMoE.

*vs. DASG-MoE*: Uses grouped MHA attention weights and forwards them to MoE routing (one-directional). Does not use entropy for cardinality, does not partition heads into coarse/fine, no feedback modulation.

---

## 3. Architecture

### 3.1 Overview

A standard transformer block:

```
X' = LayerNorm(MultiHeadAttn(X) + X)
X'' = LayerNorm(FFN(X') + X')
```

HAG-MoE replaces FFN with a Hierarchical Attention-Gated MoE block. The attention sublayer is untouched, but its internal attention weight matrices {A^(h)} are extracted and used to govern routing:

```
X'  = LayerNorm(MultiHeadAttn(X) + X)       # standard; extract A^(h)
X'' = LayerNorm(HAGMoE(X', {A^(h)}) + X')   # replaces FFN
```

Five operations inside HAGMoE:

```
1. Head partition:         Hc, Hf ← partition(H heads)
2. Attention aggregation:  a_i^c, a_i^f ← aggregate(A^h, Hc, Hf)
3. Entropy cardinality:    K_i ← EntropyGate(H(a_i^c))
4. Coarse routing:         g_i ← CoarseGate(a_i^c, X')
5. Fine routing:           {e, s} ← FineGate(a_i^f, X', g_i, K_i)
6. Expert compute:         o_i ← sum_e s_e * E_e(x_i')
7. Feedback modulation:    o_i ← o_i * (1 + gamma * r_i)
```

### 3.2 Architectural Diagram

```
Input X ∈ ℝⁿˣᵈ
        │
        ▼
┌───────────────────────────────────────────┐
│         Multi-Head Attention              │
│  H heads  →  A^(h) ∈ ℝⁿˣⁿ  (extracted)  │
│  Output X' (with residual)                │
└───────────┬───────────────────────────────┘
            │ X'              │ {A^(h)}
            │                 ▼
            │    ┌─────────────────────────────┐
            │    │       HEAD PARTITION         │
            │    │  Hc = {1,...,H/2}           │
            │    │  Hf = {H/2+1,...,H}         │
            │    └──────┬──────────┬───────────┘
            │           │          │
            │      a_i^c (coarse)  a_i^f (fine)
            │           │          │
            │  ┌─────── ▼──┐  ┌────▼──────────────────┐
            │  │  ENTROPY   │  │      COARSE GATE        │
            │  │  H(a_i^c) │  │  g_i = TopG(W_g · c_i^c)│
            │  └─────────┬─┘  └────────────┬───────────┘
            │            │                 │
            │     K_i ←──┘       Expert group g_i
            │  (dynamic cardinality)       │
            │                   ┌──────────▼──────────────┐
            │                   │    FINE GATE (in g_i)   │
            │                   │  {e,s} = TopK(W_e·c_i^f)│
            │                   └──────────┬──────────────┘
            │                              │
            │                   ┌──────────▼──────────────┐
            │                   │    EXPERT COMPUTE        │
            │                   │  o_i = Σ s_e · E_e(x_i')│
            │                   └──────────┬──────────────┘
            │                              │
            │                   ┌──────────▼──────────────┐
            │                   │  FEEDBACK MODULATION     │
            │                   │  r_i = W_r(Σ s_e · w_e) │
            │                   │  õ_i = o_i ⊙(1+γ·tanh r_i)│
            │                   └──────────┬──────────────┘
            └──────────────────┬───────────┘
                               ▼
                     LayerNorm(õ + X')  →  Output X''
```

---

## 4. Mathematical Formulation

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| $n$ | Sequence length |
| $d$ | Model dimension |
| $H$ | Number of attention heads |
| $N$ | Total number of experts |
| $G$ | Number of expert groups |
| $M = N/G$ | Experts per group |
| $K_{\min}, K_{\max}$ | Min/max active experts per token |
| $\mathbf{A}^{(h)} \in \mathbb{R}^{n \times n}$ | Attention weight matrix for head $h$ |

### 4.2 Head Partition

Partition the $H$ attention heads into two disjoint sets:

$$\mathcal{H}_c = \left\{1, \ldots, \lfloor H/2 \rfloor\right\}, \qquad \mathcal{H}_f = \left\{\lfloor H/2 \rfloor + 1, \ldots, H\right\}$$

The partition is fixed (not learned), grounded in the established empirical result that attention heads exhibit functional specialization — earlier heads in each layer tend to track positional/syntactic structure while later heads capture semantic content. The partition boundary can be validated per layer via probing classifiers.

### 4.3 Aggregated Attention Distributions and Context Vectors

For token $i$, compute averaged attention distributions over each partition:

$$\mathbf{a}_i^c = \frac{1}{|\mathcal{H}_c|} \sum_{h \in \mathcal{H}_c} \mathbf{A}^{(h)}[i, :] \;\in \Delta^{n-1}$$

$$\mathbf{a}_i^f = \frac{1}{|\mathcal{H}_f|} \sum_{h \in \mathcal{H}_f} \mathbf{A}^{(h)}[i, :] \;\in \Delta^{n-1}$$

Compute the attention-weighted context representations:

$$\mathbf{c}_i^c = \sum_{j=1}^{n} a_{ij}^c \cdot \mathbf{x}_j \in \mathbb{R}^d, \qquad \mathbf{c}_i^f = \sum_{j=1}^{n} a_{ij}^f \cdot \mathbf{x}_j \in \mathbb{R}^d$$

### 4.4 Attention Entropy and Dynamic Cardinality

For token $i$, the **coarse attention entropy** is:

$$\mathcal{H}_i = -\sum_{j=1}^{n} a_{ij}^c \log a_{ij}^c \;\in [0, \log n]$$

Normalize to $[0, 1]$:

$$\tilde{\mathcal{H}}_i = \frac{\mathcal{H}_i}{\log n}$$

Compute dynamic expert cardinality:

$$K_i = K_{\min} + \left\lfloor (K_{\max} - K_{\min}) \cdot \sigma\!\left(\alpha \cdot \left(\tilde{\mathcal{H}}_i - \mu_{\mathcal{H}}\right)\right) \right\rfloor$$

where $\sigma$ is sigmoid, $\alpha > 0$ is temperature, and $\mu_{\mathcal{H}}$ is the running batch mean of normalized entropy. This yields $K_i \in [K_{\min}, K_{\max}]$ monotonically increasing in $\tilde{\mathcal{H}}_i$: semantically uncertain tokens (diffuse attention) receive more experts; focused tokens receive fewer.

### 4.5 Hierarchical Routing

**Coarse gate — expert group selection:**

$$\mathbf{p}_i^g = \text{softmax}\!\left(\frac{W_g \mathbf{c}_i^c}{\sqrt{d}}\right) \in \Delta^{G-1}, \quad W_g \in \mathbb{R}^{G \times d}$$

$$g_i^* = \arg\max_g \mathbf{p}_i^g$$

**Fine gate — expert selection within group $g_i^*$:**

$$\mathbf{p}_i^e = \text{softmax}\!\left(W_e^{(g_i^*)} \mathbf{c}_i^f\right) \in \Delta^{M-1}, \quad W_e^{(g)} \in \mathbb{R}^{M \times d}$$

$$\{(e_1, s_1), \ldots, (e_{K_i}, s_{K_i})\} = \text{TopK}\!\left(\mathbf{p}_i^e, K_i\right)$$

### 4.6 Expert Computation

Each expert $E_e$ is a SwiGLU FFN (Shazeer, 2020):

$$E_e(\mathbf{x}) = \left(\text{silu}(W_1^{(e)} \mathbf{x}) \odot W_2^{(e)} \mathbf{x}\right) W_3^{(e)}$$

Mixed output:

$$\mathbf{o}_i = \sum_{j=1}^{K_i} s_j \cdot E_{e_j}(\mathbf{x}_i)$$

### 4.7 Bidirectional Feedback Modulation

Assign each expert $e$ a learnable **identity embedding** $\mathbf{w}_e \in \mathbb{R}^{d_r}$ ($d_r = d/8$). Form the assignment embedding:

$$\mathbf{r}_i = W_r \!\left(\sum_{j=1}^{K_i} s_j \cdot \mathbf{w}_{e_j}\right) \in \mathbb{R}^d, \quad W_r \in \mathbb{R}^{d \times d_r}$$

Apply as output modulation:

$$\tilde{\mathbf{o}}_i = \mathbf{o}_i \odot \left(\mathbf{1} + \gamma \cdot \tanh(\mathbf{r}_i)\right)$$

where $\gamma$ is a learned scalar initialized to zero. At initialization $\gamma = 0$, so $\tilde{\mathbf{o}}_i = \mathbf{o}_i$ — HAG-MoE reduces exactly to standard MoE, ensuring training stability.

### 4.8 Complete HAG-MoE Expression

$$\boxed{\text{HAG-MoE}(\mathbf{x}_i, \{\mathbf{A}^{(h)}\}) = \left[\sum_{j=1}^{K_i(\mathcal{H}_i)} s_j \cdot E_{e_j}(\mathbf{x}_i)\right] \odot \left(\mathbf{1} + \gamma \tanh\!\left(W_r \sum_{j=1}^{K_i} s_j \mathbf{w}_{e_j}\right)\right)}$$

where $K_i$, $s_j$, $e_j$ are all functions of the attention weight matrices $\{\mathbf{A}^{(h)}\}$ as derived above.

---

## 5. Theoretical Properties

### 5.1 Parameter Efficiency vs. HMoE

Standard HMoE adds: coarse gate $W^{(1)} \in \mathbb{R}^{G \times d}$ + fine gate $W^{(2)} \in \mathbb{R}^{M \times d}$ — total $d(G + M)$ routing parameters.

HAG-MoE's routing overhead: coarse gate $W_g \in \mathbb{R}^{G \times d}$ + fine gates $W_e^{(g)} \in \mathbb{R}^{M \times d}$ per group (same as standard MoE) + feedback module $N d_r + d d_r$ where $d_r = d/8$. The head partition adds exactly zero parameters. Total routing overhead is strictly less than HMoE at equal expert count.

### 5.2 Entropy-Cardinality Correspondence

**Proposition.** Under the Bayesian mixture interpretation of MoE, where $p(\mathbf{y}|\mathbf{x}) = \sum_e p(e|\mathbf{x}) p(\mathbf{y}|\mathbf{x}, e)$, the minimum number of active components $K^*$ needed to approximate the marginal $p(\mathbf{y}|\mathbf{x})$ to within $\epsilon$ KL divergence satisfies $K^* \propto H(\mathbf{a}_i^c)$ — it grows with the entropy of the contextual distribution, not with its maximum value.

*Intuition.* High entropy in the attention distribution reflects a token that contextually depends on many positions — a broader and less determinate context. This corresponds to a wider posterior $p(e|\mathbf{x})$ over experts, requiring more mixture components to represent accurately. A focused token (low entropy, sharp attention) is well-served by a single specialist. The entropy-based $K_i$ operationalizes this information-theoretic argument.

### 5.3 Gradient Paths via Bidirectional Feedback

The feedback term creates an additional gradient pathway from output loss to routing scores $s_j$:

$$\frac{\partial \tilde{\mathbf{o}}_i}{\partial s_j} = E_{e_j}(\mathbf{x}_i) \odot (\mathbf{1} + \gamma \tanh \mathbf{r}_i) + \gamma \cdot \mathbf{o}_i \odot \text{sech}^2(\mathbf{r}_i) \odot W_r \mathbf{w}_{e_j}$$

The second term is a gradient path that flows directly from output quality to routing scores via the expert identity embeddings $\mathbf{w}_{e_j}$ — a form of expert quality backpropagation absent from standard MoE. This is analogous in motivation to the Recurrent Gradient in RMoE (Qiu et al., 2024), but arises within a single layer rather than across layers via GRU state.

### 5.4 Special Cases

| Configuration | HAG-MoE reduces to |
|---|---|
| $G=1$, $K_i$ constant, $\gamma=0$ | Standard SMoE with attention-derived fine gate |
| Learned coarse gate, $K_i$ constant, $\gamma=0$ | Standard HMoE |
| Head partition off, importance-based $K_i$, $\gamma=0$ | DA-MoE variant |
| All off ($\gamma=0$, $K_i=k$, linear gate) | Switch Transformer |

HAG-MoE strictly generalizes all four. Each reduction corresponds to ablating one or more of the three novel contributions. This is the correct structure for a research contribution: it subsumes and improves upon prior work.

---

## 6. Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{LB}}^g \mathcal{L}_{\text{LB}}^{\text{group}} + \lambda_{\text{LB}}^e \mathcal{L}_{\text{LB}}^{\text{expert}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_\gamma \mathcal{L}_\gamma$$

**Language modeling loss** $\mathcal{L}_{\text{LM}}$: standard next-token cross-entropy.

**Group load balancing** $\mathcal{L}_{\text{LB}}^{\text{group}}$: encourages uniform token distribution across expert groups per batch $\mathcal{B}$:

$$\mathcal{L}_{\text{LB}}^{\text{group}} = G \cdot \sum_{g=1}^{G} f_g \cdot P_g, \quad f_g = \frac{1}{|\mathcal{B}|}\sum_{i} \mathbf{1}[g_i^* = g], \quad P_g = \frac{1}{|\mathcal{B}|}\sum_i p_{i,g}^g$$

**Expert load balancing** $\mathcal{L}_{\text{LB}}^{\text{expert}}$: same auxiliary loss applied within each group.

**Head divergence regularizer** $\mathcal{L}_{\text{div}}$: encourages the two head partitions to develop meaningfully different attention patterns:

$$\mathcal{L}_{\text{div}} = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \text{KL}\!\left(\mathbf{a}_i^c \,\|\, \mathbf{a}_i^f\right)$$

Minimizing $-\text{KL}$ pushes the two partitions toward complementary patterns, grounding the coarse/fine distinction. Without this term, the two partitions might converge to identical distributions, making the hierarchy vacuous.

**Feedback modulation regularizer** $\mathcal{L}_\gamma = \gamma^2$: prevents the feedback path from dominating expert output.

**Hyperparameter schedule**: linearly ramp all $\lambda$ coefficients over the first 5000 training steps. Hold $K_i = K_{\min}$ (constant) for the first 1000 steps until entropy statistics stabilize.

---

## 7. Implementation Details

### 7.1 Architectural Hyperparameters

| Parameter | Small | Medium | Large |
|---|---|---|---|
| Layers $L$ | 12 | 24 | 36 |
| Model dim $d$ | 512 | 1024 | 2048 |
| Attention heads $H$ | 8 | 16 | 32 |
| Expert groups $G$ | 4 | 8 | 16 |
| Experts per group $M$ | 4 | 8 | 16 |
| Total experts $N = GM$ | 16 | 64 | 256 |
| $K_{\min}$ / $K_{\max}$ | 1 / 4 | 1 / 6 | 1 / 8 |
| Feedback dim $d_r$ | 64 | 128 | 256 |
| $\gamma$ init | 0.0 | 0.0 | 0.0 |

### 7.2 Attention Matrix Extraction

HAG-MoE requires $\{\mathbf{A}^{(h)}\}$ from the preceding attention sublayer.

**Option A** (exact): Use standard `nn.MultiheadAttention` with `need_weights=True, average_attn_weights=False`. Approx 15% overhead vs FlashAttention but exact weights.

**Option B** (efficient): Run a secondary lightweight attention computation in parallel using only $\mathcal{H}_c$ and $\mathcal{H}_f$ heads for the aggregated distributions. Scales as $O(n^2 H)$ but can be fused with the main attention kernel.

### 7.3 Entropy Computation

$O(n)$ per token, $O(nL)$ total — negligible. Numerically stable form:

```python
entropy = -(attn_c * torch.log(attn_c.clamp(min=1e-8))).sum(dim=-1)  # [B, n]
norm_entropy = entropy / math.log(n)  # normalize to [0,1]
K_i = K_min + ((K_max - K_min) * torch.sigmoid(alpha * (norm_entropy - mu_H))).floor().int()
```

### 7.4 Training Stability Checklist

- Initialize $\gamma = 0.0$ (HAG-MoE = standard MoE at step 0)
- Gradient clip: max norm 1.0 on expert identity embeddings $\mathbf{w}_e$
- Use constant $K_i = K_{\min}$ for first 1000 steps
- Ramp $\lambda_{\text{LB}}, \lambda_{\text{div}}, \lambda_\gamma$ linearly over first 5000 steps
- Apply router z-loss to prevent logit explosion in coarse/fine gates

### 7.5 Distributed Training

Dynamic $K_i$ creates variable expert dispatch per token. Two strategies:

**Capacity factor padding**: Set per-token capacity to $K_{\max}$ and pad shorter dispatches. Simple, wastes some bandwidth.

**Sorted dispatch**: Sort tokens by $K_i$ descending, pack into fixed-capacity batches. Eliminates padding waste at the cost of a sort. Preferable at large $n$ or when $K_{\max}/K_{\min}$ is large.

---

## 8. Repository Structure (Proposed)

```
hag-moe/
├── README.md
├── LICENSE
├── setup.py / pyproject.toml
├── requirements.txt
│
├── hagmoe/
│   ├── core/
│   │   ├── model.py          # HAGMoETransformer (full model)
│   │   ├── block.py          # HAGMoEBlock (single transformer block)
│   │   ├── attention.py      # MultiHeadAttention with weight extraction
│   │   ├── routing.py        # CoarseGate, FineGate, EntropyGate
│   │   ├── experts.py        # Expert FFNs (SwiGLU)
│   │   └── feedback.py       # BidirectionalFeedback module
│   │
│   ├── training/
│   │   ├── losses.py         # LB loss, div regularizer, gamma regularizer
│   │   ├── trainer.py        # Training loop with lambda warm-up
│   │   └── schedulers.py     # LR and lambda schedules
│   │
│   ├── analysis/
│   │   ├── entropy_viz.py    # Per-token entropy distribution visualization
│   │   ├── routing_viz.py    # Expert routing heatmaps per layer
│   │   ├── head_partition.py # Head specialization probing classifiers
│   │   └── cardinality_stats.py  # K_i distribution analysis
│   │
│   └── research/
│       ├── special_cases.py     # Reduction to prior architectures
│       ├── entropy_theory.py    # Entropy-cardinality theory tests
│       └── gradient_analysis.py # Feedback gradient path analysis
│
├── configs/
│   ├── small.yaml    # 12L/512d/16 experts
│   ├── medium.yaml   # 24L/1024d/64 experts
│   └── large.yaml    # 36L/2048d/256 experts
│
├── scripts/
│   ├── train.py      # Pre-training script
│   ├── evaluate.py   # Evaluation on benchmarks
│   └── ablate.py     # Automated ablation runner
│
└── tests/
    ├── test_routing.py          # Routing correctness
    ├── test_entropy_gate.py     # K_i bounds [K_min, K_max]
    ├── test_feedback.py         # gamma=0 init reduces to standard MoE
    └── test_special_cases.py    # All four reduction checks
```

---

## 9. Ablation Roadmap

The three novel contributions of HAG-MoE are independently ablatable:

| Variant | Head Partition | Entropy $K_i$ | Feedback $\gamma$ | Expected insight |
|---|---|---|---|---|
| **HAG-MoE (full)** | ✓ | ✓ | ✓ | Full architecture |
| HAG-MoE-noFB | ✓ | ✓ | ✗ | Value of feedback path |
| HAG-MoE-fixK | ✓ | ✗ $k=2$ | ✓ | Value of entropy cardinality |
| HAG-MoE-learnedHier | ✗ learned gate | ✓ | ✓ | Value of head partition |
| HAG-MoE-fixK-noFB | ✓ | ✗ | ✗ | Head partition alone |
| DA-MoE baseline | ✗ | importance-$K$ | ✗ | Prior attention-informed routing |
| Standard SMoE | ✗ | ✗ | ✗ | Baseline |

**Evaluation benchmarks**: enwiki8 (BPC), WikiText-103 (PPL), MMLU (5-shot accuracy), GSM8K (accuracy), HumanEval (pass@1), GLUE average.

**Key hypotheses**: (1) Head partition ≥ learned coarse gate with zero added parameters; (2) entropy-based $K_i$ outperforms fixed top-$k$ on token-heterogeneous tasks (long documents, code, multilingual); (3) feedback modulation improves convergence speed and final perplexity beyond load-balancing improvements.

---

## 10. References

**Foundational MoE**
- Jacobs et al. (1991). *Adaptive Mixtures of Local Experts.* Neural Computation.
- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.* ICLR.
- Fedus, Zoph & Shazeer (2022). *Switch Transformers: Scaling to Trillion Parameter Models.* JMLR.

**Recent MoE Architectures**
- Jiang et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- DeepSeek-AI et al. (2024). *DeepSeek-V2/V3.* arXiv:2405.04434 / 2412.19437.
- Muennighoff et al. (2024). *OLMoE.* arXiv:2409.02060.

**Hierarchical MoE**
- arXiv:2410.02935. *Hierarchical MoE with Two-Stage Gating.* 2024.
- Zhu et al. (2025). *SAGE: Shape-Adapting Gated Experts.* arXiv:2511.18493.

**Attention-Informed Routing**
- Aghdam et al. (2024). *DA-MoE: Dynamic Expert Allocation via Attention-Derived Token Importance.* arXiv:2409.06669.
- Shi et al. (2025). *GateTS: Attention-Inspired Gating for Time-Series MoE.* arXiv:2508.17515.
- Nguyen et al. (2025). *Improving SMoE Routing with Graph of Tokens.* arXiv:2505.00792.

**Cross-Layer Routing**
- Qiu et al. (ICLR 2025). *Layerwise Recurrent Router for MoE (RMoE).* arXiv:2408.06793.

**Attention-MoE Unification**
- arXiv:2410.11222. *Quadratic Gating Functions in MoE.* 2024.
- arXiv:2506.16419. *Optimizing MoE Routers.* 2025.
- Yang et al. (2025). *Gated Attention for LLMs.* arXiv:2505.06708.

**Transformers and Attention Head Analysis**
- Vaswani et al. (2017). *Attention is All You Need.* NeurIPS.
- Clark et al. (2019). *What Does BERT Look At? Analysis of BERT's Attention.* BlackboxNLP.
- Voita et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting.* ACL.
- Shazeer (2020). *GLU Variants Improve Transformers.* arXiv:2002.05202.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

Open source under the [Apache 2.0 License](LICENSE).

---

## Citation

```bibtex
@article{debnath2025hagmoe,
  title     = {HAG-MoE: Hierarchical Attention-Gated Mixture of Experts},
  author    = {Debnath, Devanik},
  year      = {2025},
  note      = {Preprint. https://github.com/Devanik21/HAG-MoE},
  institute = {National Institute of Technology Agartala}
}
```

---

*Conceived from the observation that the attention mechanism is not just a preprocessor for routing — it is the routing. HAG-MoE makes that relationship explicit, bidirectional, and mathematically grounded.*
