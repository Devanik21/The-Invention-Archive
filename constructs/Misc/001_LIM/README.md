# Latent Inference Manifold

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Paradigm-Multi--Agent_RL-6d28d9?style=flat-square"/>
  <img src="https://img.shields.io/badge/Target-ARC--AGI--2-fbbf24?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active_Research-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github"/>
  <img src="https://img.shields.io/github/stars/Devanik21/Latent-Inference-Manifold?style=flat-square&color=facc15"/>
  <img src="https://img.shields.io/github/forks/Devanik21/Latent-Inference-Manifold?style=flat-square&color=38bdf8"/>
</p>

> *A neuro-symbolic multi-agent system that discovers transformation abstractions through pure induction — no hardcoded rules, no DSL primitives, no IF-THEN logic. Nine specialized cognitive agents debate over a shared latent space to solve ARC-AGI-2 tasks, learning reusable transformation programs entirely from examples.*

---

**Research Topics:** `arc-agi-2` · `inference-time compute` · `latent-space geometry` · `meta-learning` · `multi-agent reinforcement learning` · `program synthesis` · `online dictionary learning` · `Bayesian priors` · `free energy minimization` · `neuro-symbolic AI`

---

## Table of Contents

1. [Abstract](#abstract)
2. [Theoretical Foundations](#theoretical-foundations)
3. [System Architecture](#system-architecture)
4. [Mathematical Formulation](#mathematical-formulation)
5. [The Council of Nine Agents](#the-council-of-nine-agents)
6. [Module Breakdown](#module-breakdown)
7. [Tech Stack](#tech-stack)
8. [Getting Started](#getting-started)
9. [Usage](#usage)
10. [Configuration](#configuration)
11. [Project Structure](#project-structure)
12. [Research Roadmap](#research-roadmap)
13. [Contributing](#contributing)
14. [Author](#author)
15. [License](#license)

---

## Abstract

**Latent Inference Manifold (LIM)** is a research system that investigates whether *general* visual-logical reasoning can emerge from a collection of specialized cognitive agents operating entirely in a continuous, learned latent space — without ever being told what operations to perform.

The central hypothesis is that any visual transformation `T : X → Y` (where `X, Y ∈ ℤ₀₋₉^{H×W}` are ARC-AGI-2 grids) can be encoded as a sparse coefficient vector `z ∈ ℝ⁶⁴` over a set of learned basis transformations `{dᵢ}ᵢ₌₁⁶⁴ ⊂ ℝ^{225}`, such that:

```
δ(X, Y) ≈ Σᵢ zᵢ · dᵢ     where z ≥ 0, ‖z‖₀ ≪ 64
```

where `δ(X, Y) = flatten(Y) − flatten(X) + 9` is the non-negative transformation delta vector. The basis `D = [d₁ | d₂ | … | d₆₄]` is not specified by the programmer — it is discovered incrementally via **online Non-Negative Matrix Factorization** as the system solves more tasks. Each new solved episode enriches the dictionary. The system thus *learns its own primitives* from the data.

Nine autonomous agents — Perceiver, Dreamer, Scientist, Skeptic, Philosopher, CausalReasoner, CuriosityEngine, Metacognitor, and Archivist — run a Socratic debate loop over a shared **Blackboard** working memory. A **MetaLearner** component maintains a Bayesian prior over the 64 latent dimensions, biasing future searches toward subspaces that historically led to correct solutions. This meta-reasoning allows the system to improve its own discovery process across episodes without programmer intervention.

The project targets the **ARC-AGI-2** benchmark, widely regarded as a key test for general visual-logical reasoning. Tasks are generated procedurally via a zero-cheat `Universe` module that synthesizes novel transformations from Core Knowledge Priors (symmetry, objectness, containment, gravity, numerosity, causality, goal-directedness) — ensuring no two episodes share the same transformation fingerprint.

---

## Theoretical Foundations

### 1. The Manifold Hypothesis in Latent Transformation Space

The manifold hypothesis, formalized by Bengio et al. (2013) and Fefferman et al. (2016), asserts that natural data distributions concentrate near low-dimensional manifolds embedded in high-dimensional ambient spaces. LIM extends this to the *transformation* domain: the space of all meaningful ARC grid transformations, though formally infinite-dimensional, is hypothesized to lie near a low-dimensional manifold `M ⊂ ℝ^{225}` spanned by a small number of compositional basis operations.

This justifies the dimensionality choice of 64 latent components for a 225-dimensional delta space. The intrinsic dimensionality of the transformation manifold — measurable via TWO-NN estimation (Facco et al., 2017) — is expected to be far lower than 64, confirming that the learned dictionary is overparameterized with respect to the true manifold, which aids generalization.

### 2. Online Non-Negative Matrix Factorization

The dictionary learning mechanism follows the online NMF framework of Mairal et al. (2010). Given an incrementally growing matrix of delta vectors `Δ = [δ₁ | δ₂ | … | δₙ] ∈ ℝ^{225×n}`, the system solves:

```
min_{D,Z}  ‖Δ − D·Z‖²_F     subject to:   D ≥ 0,  Z ≥ 0,  ‖dᵢ‖₂ ≤ 1  ∀i
```

Rather than batch recomputation, each new pair `(δₜ, zₜ)` triggers a coordinate descent update over `D` using accumulated sufficient statistics `A = Σ zₜzₜᵀ` and `B = Σ δₜzₜᵀ`. This keeps memory bounded at `O(FLAT_DIM × LATENT_DIM) = O(225 × 64)` regardless of the number of episodes seen — critical for deployment on memory-constrained environments like Streamlit Cloud.

### 3. Free Energy Minimization and Active Inference

The agent council embodies the Free Energy Principle (Friston, 2010). Each agent acts to minimize the variational free energy `F = E_q[log q(z) − log p(z, X)]`, where `q(z)` is the agent's current belief over the latent transformation and `p(z, X)` is the generative model. The CuriosityEngine concretely measures **surprise** as the `ℓ₁` pixel error between the best current hypothesis and the ground truth, and issues exploration directives that drive the Dreamer and Scientist toward lower-surprise regions of the latent space.

### 4. Bayesian Meta-Learning over the Latent Prior

The MetaLearner maintains a prior vector `π ∈ [0.05, 1.0]⁶⁴`, where `πᵢ` is the empirical probability that latent dimension `i` is actively used in successful solutions. Upon observing a winning `z*` in a task solved in `r` rounds:

```
πᵢ ← (1 − α·η) · πᵢ  +  α·η · (z*ᵢ / max z*)
```

where `η = max(0.2, 1/r)` is an efficiency signal (faster solutions apply stronger updates) and `α = 0.1` is the base learning rate. This is an exponential moving average with an adaptive step, equivalent to maintaining a Beta posterior with moment matching under the assumption that `z*ᵢ / max z*` is a sufficient statistic for dimension relevance. The result is a prior `π` that the Dreamer and Scientist use to bias sampling toward historically productive subspaces — a form of **amortized inference acceleration** across episodes.

### 5. ARC-AGI-2 as a Benchmark for Abstraction

The Abstraction and Reasoning Corpus (ARC-AGI) was designed by François Chollet to require genuine human-like abstraction — the ability to identify a rule from just 2–5 examples and apply it to a novel test case. ARC-AGI-2 extends this with harder compositional rules. Crucially, LIM generates tasks *procedurally* using `universe.py`, which synthesizes transformations by composing Core Knowledge Priors (topological, spatial, abstract-logical) without repetition of fingerprints. This ensures that neither the agent council nor the dictionary can overfit to a fixed task distribution.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Latent Inference Manifold                         │
│                       LAteNT.py — Streamlit Dashboard                   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
               ┌────────────────────▼──────────────────────┐
               │         Universe.py — Task Generator       │
               │  Procedural ARC-AGI-2 synthesis via        │
               │  Core Knowledge Priors (7 priors,          │
               │  3 domains A/B/C, 5 difficulty levels)     │
               └────────────────────┬──────────────────────┘
                                    │  ARCTask
               ┌────────────────────▼──────────────────────┐
               │              Council.py                     │
               │         The Council of 9 Agents            │
               │                                             │
               │  ┌──────────────────────────────────────┐  │
               │  │         Blackboard (memory.py)        │  │
               │  │  SharedWorkingMemory: WorldState,     │  │
               │  │  HypothesisStack, ContradictionLog,   │  │
               │  │  AgentCallLog, MeetingAgenda          │  │
               │  └───────────────────┬──────────────────┘  │
               │                      │                      │
               │  ┌───────────────────▼────────────────┐    │
               │  │  Phase 0: Orientation               │    │
               │  │    Perceiver → WorldState           │    │
               │  │    Archivist → inject Prior Art     │    │
               │  │    LatentDictionary → pre-warm      │    │
               │  └───────────────────┬────────────────┘    │
               │                      │                      │
               │  ┌───────────────────▼────────────────┐    │
               │  │  Phase 1: First Imagination         │    │
               │  │    Dreamer → K=8 z-sampled          │    │
               │  │             hypotheses              │    │
               │  └───────────────────┬────────────────┘    │
               │                      │                      │
               │  ┌───────────────────▼────────────────┐    │
               │  │  Phase 2–4: Debate Loop (≤30 rounds)│    │
               │  │    Metacognitor → sets Agenda       │    │
               │  │    Scientist   → z-vector search    │    │
               │  │    Skeptic     → adversarial probe  │    │
               │  │    CausalReasoner → counterfactual  │    │
               │  │    Dreamer     → re-imagination     │    │
               │  │    Philosopher → ontological reframe│    │
               │  │    CuriosityEngine → surprise signal│    │
               │  └───────────────────┬────────────────┘    │
               │                      │                      │
               │  ┌───────────────────▼────────────────┐    │
               │  │  Phase 5: Archival                  │    │
               │  │    Archivist → EpisodeMemory        │    │
               │  │    MetaLearner → prior update (π)   │    │
               │  │    LatentDictionary → NMF update    │    │
               │  │    SkillLibrary → z* extraction     │    │
               │  └─────────────────────────────────────┘    │
               └───────────────────────────────────────────┘
                                    │
               ┌────────────────────▼──────────────────────┐
               │    Cross-Domain Generalization Test        │
               │    Domain A (Spatial) → Domain B (Topo)   │
               │    → Domain C (Abstract) — zero-shot      │
               └────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Grid Representation

Every ARC grid `G ∈ ℤ₀₋₉^{H×W}` is padded to a canonical `15×15` canvas and flattened into a vector:

```
φ(G) = flatten(pad₁₅ₓ₁₅(G))  ∈  ℝ^{225}
```

### Transformation Delta Encoding

For a training pair `(Xᵢ, Yᵢ)`, the non-negative transformation delta is:

```
δᵢ = φ(Yᵢ) − φ(Xᵢ) + 9  ∈  [0, 18]^{225}
```

The shift `+9` renders the delta non-negative, satisfying the NMF constraint, since ARC pixel values lie in `[0, 9]` and their differences lie in `[−9, +9]`.

### Dictionary Learning Objective

The online NMF seeks a dictionary `D ∈ ℝ^{225×64}` and code matrix `Z ∈ ℝ^{64×n}` such that:

```
(D*, Z*) = argmin_{D,Z≥0}  (1/n) Σᵢ ‖δᵢ − D·zᵢ‖₂²   +  λ‖Z‖₁
```

The `ℓ₁` regularizer on `Z` encourages sparse codes — a desired property since most tasks require only a small number of active transformation components. The online update for dictionary atom `dⱼ` uses accumulated statistics:

```
A ∈ ℝ^{64×64}:  A ← A + zₜzₜᵀ
B ∈ ℝ^{225×64}: B ← B + δₜzₜᵀ

dⱼ ← max(0, (Bⱼ − D·Aⱼ + Aⱼⱼ·dⱼ) / Aⱼⱼ),  then normalize: dⱼ ← dⱼ/‖dⱼ‖₂
```

### Hypothesis Confidence

The Dreamer generates `K = 8` hypotheses per round. The confidence of hypothesis `hₖ` with latent code `zₖ` is:

```
conf(hₖ) = 1 / (1 + 0.5·‖zₖ‖₂ + 0.05·k)
```

The penalty `0.5·‖zₖ‖₂` encodes an **MDL (Minimum Description Length) prior**: simpler programs (sparser `z`) are preferred. The `0.05·k` term provides tiebreaking across hypothesis indices.

### Curiosity and Surprise

The CuriosityEngine measures pixel-level prediction error against the ground truth:

```
surprise(ĥ, Y*) = ‖flatten(ĥ) − flatten(Y*)‖₁ / (H·W)
```

This normalized surprise drives exploration directives:  
- `surprise > 0.6` → `DREAMER_EXPLORE_LOW_CONFIDENCE` (explore new regions of latent space)  
- `surprise > 0.3` → `SCIENTIST_EXTEND_SEARCH` (gradient search with more iterations)  
- Otherwise → `CAUSAL_VERIFY` (test causal consistency)

### MetaLearner Prior Update

Let `z* ∈ ℝ⁶⁴` be the winning latent code and `r` the number of rounds to solution:

```
η = max(0.2,  1/r)                    # efficiency signal
α_eff = α · η                         # effective learning rate
π ← clip((1 − α_eff)·π + α_eff·(z*/max z*),  0.05, 1.0)
```

This is a moment-matched EMA approximating a Beta-posterior update over dimension relevance. Dimensions unused in successful solutions decay slowly toward the exploration floor of 0.05, preventing complete latent dimension collapse.

---

## The Council of Nine Agents

Each agent has a distinct cognitive role, implemented as a Python class with a primary method returning a typed `AgentResult` dataclass. Agents communicate exclusively through the shared **Blackboard** — they have no direct inter-agent channels, which preserves the independence of their reasoning and prevents groupthink.

| # | Agent | Role | Primary Method | Mathematical Operation |
|---|-------|------|----------------|----------------------|
| 1 | **Perceiver** | Object segmentation | `perceive(grid, bb)` | Connected-component labeling → `WorldState` |
| 2 | **Dreamer** | Latent hypothesis generation | `imagine(task, bb, skill_lib, latent_dict, meta_learner)` | `z ~ p(z; π)`, decode via `D·z + φ(X) − 9` |
| 3 | **Scientist** | Transformation discovery | `synthesize(task, bb, ...)` | Gradient search in `z`-space: `z* = argmin_z ‖D·z + φ(X) − φ(Y)‖₂` |
| 4 | **Skeptic** | Adversarial falsification | `challenge(task, bb, latent_dict)` | Apply mutations `μ(X)` to training inputs; verify `h(μ(X)) ≈ μ(Y)` |
| 5 | **Philosopher** | Ontological reframing | `reframe(grid, bb, revision, latent_dict)` | Basis rotation / background reinterpretation |
| 6 | **CausalReasoner** | Counterfactual testing | `verify(task, bb, latent_dict)` | Interventional consistency: `do(X=X')` → verify `Y'` matches predicted `h(X')` |
| 7 | **CuriosityEngine** | Active-inference exploration | `observe(pred, target, bb)` | `surprise = ‖ĥ − Y*‖₁/(H·W)` → exploration directive |
| 8 | **Metacognitor** | Session chair & convergence | `arbitrate(bb, curiosity_directive)` | Round-budget tracking; convergence vote; agenda composition |
| 9 | **Archivist** | Episodic memory & skill extraction | `archive(task, bb)` → `inject_hints(task, bb)` | Store `z*` in `LatentSkillLibrary`; feed `(X,Y)` to `LatentDictionary.register_pair()` |

### Council Meeting Protocol

The Council runs a **Socratic debate loop** over at most `MAX_ROUNDS = 30` rounds per task. Each round is orchestrated by the Metacognitor, which constructs a `meeting_agenda` — an ordered list of agents to call this round — based on the current state of the Blackboard (number of pending hypotheses, contradiction log depth, rounds elapsed, curiosity directive). The loop terminates when:

1. The CausalReasoner promotes a hypothesis to `CAUSAL_LAW` status, triggering acceptance and `declare_answer()`; or  
2. The round budget `MAX_ROUNDS` is exhausted, in which case the best non-falsified hypothesis is declared by best-effort.

Every agent action is appended to `bb.agent_call_log`, enabling the Streamlit dashboard to stream the live debate as an HTML log with per-agent color coding.

---

## Module Breakdown

### `universe.py` — Procedural Task Generator

Synthesizes novel ARC-AGI-2 tasks from scratch using **Core Knowledge Priors** drawn from developmental psychology (Spelke & Kinzler, 2007). No task seen in one session can repeat a transformation fingerprint — the fingerprint is a `SHA-256` hash of the composition of priors and their parameters. Tasks are stratified into three cross-generalization domains:

| Domain | Priors | Character |
|--------|--------|-----------|
| **A — Spatial/Geometric** | `SYMMETRY` | Reflections, rotations, 2D transforms |
| **B — Topological/Physical** | `OBJECTNESS`, `CONTAINMENT`, `GRAVITY` | Object movement, filling, gravity simulation |
| **C — Abstract/Logical** | `NUMEROSITY`, `CAUSALITY`, `GOAL_DIRECTEDNESS` | Counting, causal chains, goal inference |

Difficulty levels L1–L5 correspond to compositions of 1 through 4+ priors respectively. `GridTransforms` provides the atomic building blocks (`rotate90`, `flip_h`, `flip_v`, `gravity_down`, `flood_fill`, and more) that are composed procedurally by the generator.

### `council.py` — The Council of Minds (~940 lines)

Implements all nine agent classes and the `Council` orchestrator. Key design decisions:

- **No inter-agent channels.** All state passes through `Blackboard`. Agents are stateless with respect to a single task (their persistent state lives in the shared `LatentDictionary`, `LatentSkillLibrary`, and `MetaLearner`).
- **Generator-based streaming.** `Council.solve()` is a Python generator that `yield`s a `Blackboard.snapshot()` after every agent action, enabling the Streamlit dashboard to render live updates without blocking.
- **Typed `AgentResult` dataclass.** Every agent returns `AgentResult(agent, success, message, data)`, providing a uniform interface for the orchestrator.

### `latent_dictionary.py` — Online Transformation Dictionary Learning (~501 lines)

The intellectual heart of the system. Implements an online NMF-based dictionary learner with the following API:

```python
# Register a solved (input, output) pair to grow the dictionary
latent_dict.register_pair(inp_grid, out_grid, task_id="t001", label="gravity_fill")

# Sample novel z coefficients biased by the MetaLearner's prior
z_samples = latent_dict.sample_z(n=8, temperature=1.0, prior_z=π)

# Decode a z vector back to a predicted output grid
predicted_grid = latent_dict.decode_z(z, test_input)

# Search for z* that best explains a training pair
z_star = latent_dict.search_z(inp_grid, out_grid, n_iter=NMF_ITERS)
```

Memory usage is bounded: at most `MAX_PAIRS = 2000` delta vectors are retained (FIFO), and the sufficient statistics `A, B` are updated incrementally. Total RAM footprint is under 50 MB regardless of session length.

### `memory.py` — The Shared Free Energy Substrate (~543 lines)

Provides the core data structures:

- **`Blackboard`** — Structured working memory. Holds `hypothesis_stack` (max 50 `Hypothesis` objects), `contradiction_log`, `agent_call_log`, `world_state`, `meeting_agenda`, `prior_art_hints`, and the `final_verdict`. Exposes `snapshot()` for dashboard streaming.
- **`EpisodeMemory`** — Persistent store of `EpisodeRecord` objects (max 500). Tracks task IDs, outcomes, rounds to solve, and winning programs. Computes `get_generalization_series()` for the cross-domain transfer curve.
- **`LatentSkillLibrary`** — A growing library of named `LatentSkill` objects, each containing the winning `z*` vector from a solved episode. The Archivist deposits skills; the Dreamer borrows hints from them for new tasks.
- **`SurpriseTracker`** — A circular buffer of recent prediction errors, consumed by the CuriosityEngine.

### `meta_learner.py` — Bayesian Prior Updater (~90 lines)

A lightweight module maintaining `π ∈ ℝ⁶⁴` as a relevance prior over latent dimensions. Designed to be called once per solved episode via `update(winning_z, rounds_to_solve)`. The `get_prior_z()` method returns the current prior for use by the Dreamer's sampling procedure and the Scientist's search initialization.

### `LAteNT.py` — Streamlit Dashboard (~800+ lines)

The live scientific dashboard. Features include:

- **ARC grid renderer** using a custom 10-color `ListedColormap` matching the official ARC palette, with `BoundaryNorm` for integer cell values.
- **Real-time Council log** streaming HTML-colored agent messages in a scrollable `<div>` with agent-specific CSS classes.
- **Cross-domain experiment mode** that runs sequential Domain A → B → C episodes and plots the generalization transfer curve.
- **Latent dictionary visualizer** displaying the top-k learned basis components as 15×15 heatmaps.
- **MetaLearner radar chart** showing the current 64-dim prior vector `π`.
- **Session statistics** with solve rate, average rounds, skill library size, and meta-learner health indicators.

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| **Python** | 3.11+ | Core language |
| **Streamlit** | ≥1.32.0 | Live scientific dashboard and session state |
| **NumPy** | ≥1.24.0 | Grid arithmetic, NMF updates, latent vector operations |
| **SciPy** | ≥1.11.0 | Sparse solvers, connected-component labeling (`ndimage`), graph algorithms |
| **Matplotlib** | ≥3.7.0 | ARC grid rendering, basis component visualization, bar/line plots |
| **Pandas** | ≥2.0.0 | Episode records, statistics tables, export logs |

**Design constraint:** The entire system runs with zero GPU requirement and within Streamlit Community Cloud's 1 GB RAM limit. All heavy linear algebra (NMF, eigendecomposition) uses NumPy's BLAS-backed routines on CPU. FAISS or Ripser are intentionally not in the current requirements — the system demonstrates that meaningful latent-space reasoning does not require specialized geometric libraries.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher (3.11 recommended)
- A virtual environment manager (`venv` or `conda`)
- ~200 MB disk space for dependencies
- No GPU required

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Devanik21/Latent-Inference-Manifold.git
cd Latent-Inference-Manifold

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run LAteNT.py
```

The dashboard will open at `http://localhost:8501`. On first launch, the `Universe`, `Council`, `LatentDictionary`, and `MetaLearner` are initialized with a random seed displayed in the sidebar.

### Verifying the Installation

```bash
# Run the included smoke test
python test_cdg.py

# Expected output: cross-domain generalization test report
# with Domain A, B, C solve rates and transfer curve
```

---

## Usage

### Single Task Inference

Click **"Generate New Task"** in the sidebar to synthesize a fresh ARC-AGI-2 task from the Universe. Select a difficulty level (L1–L5) and domain (A/B/C or random). Click **"Convene Council"** to start the debate. The Council Log tab streams real-time agent messages as the 9 agents reason over the task.

The **Task Visualization** tab renders the training pairs and test input as colored ARC grids. The **Hypothesis Panel** shows the current top hypothesis overlaid on the ground truth, with pixel-level diff highlighting.

### Cross-Domain Generalization Experiment

Enable **"Cross-Domain Mode"** in the sidebar. The system runs 30 episodes on Domain A, then 30 on Domain B, then 30 on Domain C — tracking solve rates at each stage. The **Generalization Curve** tab plots the transfer performance, measuring whether abstractions learned in Domain A generalize to the unseen structure of Domain C.

A solve rate above **50% on Domain C** with no Domain C training is the primary success criterion defined in `plan.md`.

### Inspecting the Latent Dictionary

Navigate to the **Latent Space** tab after several episodes. The dictionary basis components `{dᵢ}` are displayed as 15×15 heatmaps. Components that have received many updates cluster around common transformation patterns (e.g., edge-filling operations, color-propagation patterns). The **MetaLearner Prior** radar shows which of the 64 dimensions are currently most trusted.

### Programmatic API

```python
from universe import Universe, DifficultyLevel, TaskDomain
from council import Council

# Initialize
universe = Universe(seed=42)
council  = Council(seed=42)

# Generate a task
task = universe.generate_task(
    difficulty=DifficultyLevel.L3,
    domain=TaskDomain.B_TOPOLOGICAL
)

# Run the Council (generator interface for streaming)
for snapshot in council.solve(task, stream=True):
    agent = snapshot["last_agent"]
    msg   = snapshot["last_message"]
    print(f"[{agent}] {msg}")

# Final statistics
stats = council.stats()
print(f"Skill library size: {stats['skill_library_size']}")
print(f"Active latent dims: {stats['meta_learner']['active_dimensions']}")
```

---

## Configuration

All hyperparameters are defined as module-level constants and can be overridden before import or via `.env` if you add a `python-dotenv` loader.

| Parameter | Module | Default | Description |
|-----------|--------|---------|-------------|
| `LATENT_DIM` | `latent_dictionary.py` | `64` | Number of NMF dictionary atoms (basis vectors) |
| `FLAT_DIM` | `latent_dictionary.py` | `225` | Flattened grid dimension (15×15) |
| `MAX_GRID_DIM` | `latent_dictionary.py` | `15` | Maximum grid height/width (padding target) |
| `LEARNING_RATE` | `latent_dictionary.py` | `0.005` | Online NMF step size |
| `NMF_ITERS` | `latent_dictionary.py` | `30` | Coordinate descent iterations per online update |
| `MAX_PAIRS` | `latent_dictionary.py` | `2000` | Max retained delta vectors (FIFO eviction) |
| `MIN_PAIRS_FIT` | `latent_dictionary.py` | `5` | Min pairs before dictionary is usable |
| `Dreamer.K` | `council.py` | `8` | Hypotheses generated per Dreamer call |
| `Council.MAX_ROUNDS` | `council.py` | `30` | Maximum debate rounds per task |
| `MetaLearner.learning_rate` | `meta_learner.py` | `0.1` | Base EMA rate for prior update |
| `MAX_HYPOTHESIS_STACK` | `memory.py` | `50` | Maximum concurrent hypotheses on Blackboard |
| `MAX_EPISODE_MEMORY` | `memory.py` | `500` | Maximum stored episode records |

---

## Project Structure

```
Latent-Inference-Manifold/
│
├── LAteNT.py               # Streamlit dashboard — entry point
│                           # (≈800 lines: page config, CSS, session state,
│                           #  grid rendering, Council log, cross-domain UI)
│
├── council.py              # The Council of 9 Agents + orchestration loop
│                           # (≈940 lines: Perceiver, Dreamer, Scientist,
│                           #  Skeptic, Philosopher, CausalReasoner,
│                           #  CuriosityEngine, Metacognitor, Archivist)
│
├── universe.py             # Procedural ARC-AGI-2 task generator
│                           # (≈526 lines: Prior enum, TaskDomain, DifficultyLevel,
│                           #  GridTransforms, ARCTask, Universe)
│
├── memory.py               # Shared working memory and persistent stores
│                           # (≈543 lines: Blackboard, EpisodeMemory,
│                           #  LatentSkillLibrary, SurpriseTracker,
│                           #  Hypothesis, ContradictionEntry, WorldState)
│
├── latent_dictionary.py    # Online NMF-based transformation dictionary
│                           # (≈501 lines: register_pair, sample_z,
│                           #  decode_z, search_z, online NMF update)
│
├── meta_learner.py         # Bayesian prior updater over latent dimensions
│                           # (≈90 lines: MetaLearner, update, get_prior_z)
│
├── test_cdg.py             # Cross-domain generalization smoke test
│
├── plan.md                 # Research plan: 5 phases, 15–17 week timeline
│
├── requirements.txt        # Minimal dependency set (6 packages)
├── .gitignore
└── LICENSE                 # MIT License
```

---

## Research Roadmap

The plan is structured in five phases targeting a 15–17 week timeline toward demonstrating genuine cross-domain abstraction transfer.

**Phase 1 — Latent Transformation Learning (Weeks 1–4)**  
Expand the `LatentDictionary` training pipeline. Collect 1000+ diverse `(input → output)` pairs across all three domains and pre-train the NMF basis offline, then continue online learning during inference. Evaluate intrinsic dimensionality of the learned transformation manifold via TWO-NN estimation on the accumulated delta matrix.

**Phase 2 — Abstraction Discovery Through Pure Induction (Weeks 5–9)**  
Remove all symbolic primitives from `universe.py`'s transformation composer and replace with a purely learned program synthesis path in the Scientist agent. The Scientist will search the latent space via gradient descent (`z* = argmin_z ‖D·z + φ(X) − φ(Y)‖₂²`) rather than matching against a symbolic DSL. The goal is that the system discovers what "rotation-like" behavior is, without being told that `rotate90` is a valid primitive.

**Phase 3 — Cross-Domain Generalization Benchmark (Weeks 10–12)**  
Formal evaluation: train on 30 Domain-A tasks, validate on 30 Domain-B tasks, test zero-shot on 30 Domain-C tasks. **Success threshold: ≥ 50% solve rate on Domain C.** Plot the generalization curve and compare against a rule-based baseline that has access to the full DSL.

**Phase 4 — Meta-Learning Acceleration (Weeks 13–16)**  
Extend the `MetaLearner` from a flat prior over dimensions to a recurrent model that conditions on episode history (task fingerprints, failure modes, hypothesis trajectories). This meta-learner shapes both the Dreamer's sampling distribution and the Scientist's search initialization without explicit rules — purely from observed episode patterns. Target metric: reduction in mean rounds-to-solve across a session of 100 tasks.

**Phase 5 — Emergence Analysis and Publication (Weeks 17)**  
Measure emergent properties: does the skill library contain genuinely novel z-compositions not present in any individual training example? Do similar transformations cluster in latent space (visualized via UMAP of the dictionary atoms)? Ablate the MetaLearner to quantify its contribution to solve rate and efficiency. Target: a short technical report titled *"Learning Transformation Abstractions Without Explicit Rules"* with latent space visualizations and cross-domain transfer metrics.

**Future Directions**
- [ ] Riemannian geometry of the learned transformation manifold (sectional curvature, geodesic distances)
- [ ] Neural collapse analysis in the latent space at convergence
- [ ] Cross-architecture comparison: does a different agent topology discover different geometric structure?
- [ ] Theoretical connection: empirical validation of manifold dimension vs. sample complexity bounds
- [ ] Multi-modal extension: apply the same framework to symbolic (text) transformations, testing whether the same latent dictionary can unify visual and symbolic abstraction

---

## Contributing

Contributions, issues, and research discussions are warmly welcome. This is an open research project, and interesting ideas often come from unexpected directions.

```bash
# Fork, then clone your fork
git clone https://github.com/<your-username>/Latent-Inference-Manifold.git
cd Latent-Inference-Manifold

# Create a feature branch
git checkout -b feature/your-idea

# Make your changes, then commit using conventional commits
git commit -m "feat: add TWO-NN intrinsic dimension estimator to LatentDictionary"
git commit -m "fix: prevent MetaLearner prior collapse on long sessions"
git commit -m "docs: add mathematical derivation for NMF update step"

# Push and open a Pull Request
git push origin feature/your-idea
```

**Good first contributions:**
- Adding an intrinsic dimensionality estimator (TWO-NN or correlation dimension) to `latent_dictionary.py` to track manifold dimension as the dictionary grows
- Adding persistent homology (via `ripser`) to characterize the topology of the hypothesis space
- Replacing the flat EMA `MetaLearner` with a small LSTM that conditions on episode fingerprints
- Writing a proper unit test suite for `universe.py`'s transformation primitives

Please follow conventional commit messages, include docstrings for new classes and methods, and add a brief entry to `plan.md` if your contribution aligns with one of the research phases.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [Apache-2.0 License](LICENSE).

---

*Built with curiosity, mathematical depth, and the quiet conviction that abstraction can be learned — not programmed.*
