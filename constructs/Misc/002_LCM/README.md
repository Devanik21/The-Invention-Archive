<div align="center">

<br/>

```
██╗      █████╗ ████████╗███████╗███╗   ██╗████████╗
██║     ██╔══██╗╚══██╔══╝██╔════╝████╗  ██║╚══██╔══╝
██║     ███████║   ██║   █████╗  ██╔██╗ ██║   ██║
██║     ██╔══██║   ██║   ██╔══╝  ██║╚██╗██║   ██║
███████╗██║  ██║   ██║   ███████╗██║ ╚████║   ██║
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝
```

**Latent Consensus Manifold**
*A 9-Agent Neuro-Symbolic Collective for Abstract Reasoning*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)](LICENSE)
[![Solve Rate](https://img.shields.io/badge/Solve%20Rate-100%25%20(18%2F18)-22c55e?style=for-the-badge)]()
[![Avg Rounds](https://img.shields.io/badge/Avg%20Rounds-15.9%20±%203.2-7dd3fc?style=for-the-badge)]()

<br/>

> *"Rather than learning monolithic end-to-end mappings, LAteNT decomposes the reasoning process into nine specialized cognitive modules, each contributing distinct epistemic functions."*

<br/>

**Author:** [Devanik](https://github.com/Devanik21) • B.Tech ECE '26, NIT Agartala
**Fellowship:** Samsung Convergence Software Fellowship (Grade I) • Indian Institute of Science

</div>

---

## Table of Contents

1. [Abstract](#-abstract)
2. [Why This Exists](#-why-this-exists)
3. [System Architecture](#-system-architecture)
   - [The Blackboard](#-the-blackboard-shared-free-energy-substrate)
   - [Domain-Specific Language (DSL)](#-domain-specific-language)
   - [The Nine Agents](#-the-nine-agent-council)
   - [Council Meeting Protocol](#-council-meeting-protocol)
4. [The ARC-AGI-2 Universe](#-the-arc-agi-2-universe)
5. [Live Dashboard](#-live-dashboard)
6. [Experimental Results](#-experimental-results)
7. [Complexity Analysis](#-complexity-analysis)
8. [Codebase Tour](#-codebase-tour)
9. [Installation & Usage](#-installation--usage)
10. [Roadmap — True AGI Plan](#-roadmap--true-agi-plan)
11. [Limitations & Honest Assessment](#-limitations--honest-assessment)
12. [References](#-references)
13. [Contact](#-contact)

---

## 🧠 Abstract

LAteNT implements a **multi-agent neuro-symbolic architecture** for solving abstract reasoning tasks of the ARC-AGI paradigm. Rather than training a monolithic neural network, the system deploys nine cognitively specialized agents over a **shared blackboard substrate**, executing a Socratic loop of iterative hypothesis generation, empirical falsification, and counterfactual causal validation.

The core insight: reasoning under uncertainty is most robust when modeled as an **adversarial epistemic process** — where agents simultaneously propose, challenge, and refine each other's beliefs. This mirrors how scientific communities arrive at ground truth: not through individual brilliance, but through structured disagreement.

**Key contributions:**
- A 9-agent council with explicit role separation — each agent implements a distinct epistemological function
- A zero-cheat procedural task generator grounded in Core Knowledge Priors from developmental psychology
- A growing **Skill Library** that enables meta-learning across episodes via biased hypothesis generation
- A **Surprise/Free Energy metric** (inspired by Friston's Active Inference) that detects epistemic plateaus and triggers exploration directives
- A fully auditable blackboard: no agent maintains private state; all reasoning is observable
- **100% solve rate across 18 procedurally-generated tasks** (avg. 15.9 rounds, 53% budget utilization)

---

## 💡 Why This Exists

The [ARC-AGI benchmark](https://arcprize.org/) is designed to resist pattern-matching. Every task requires identifying a transformation rule from 3 training examples and applying it to a novel test input. The transformations are never repeated. State-of-the-art neural models as of early 2026 (Gemini 3 Deep Think: 84.6%; Human baseline: ~80%) still struggle with this benchmark because they lack the ability to:

1. **Discover discrete compositional rules** — not just interpolate between learned patterns
2. **Falsify their own hypotheses** — confirming bias is a core failure mode of neural networks
3. **Reason causally** — distinguishing correlation ("this output looks right") from causation ("this transformation law *generates* the output")
4. **Transfer across episodes** — each task is a cold start for end-to-end models

LAteNT directly addresses all four. It is not a SOTA-chasing architecture — it is a **transparency-first research system** designed to produce interpretable reasoning traces and quantify emergent properties of multi-agent symbolic deliberation.

---

## 🏗 System Architecture

The architecture consists of four interlocking subsystems:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UNIVERSE (universe.py)                       │
│  Procedural ARC-AGI-2 task generator • Core Knowledge Priors         │
│  Zero-cheat fingerprinted task synthesis • L1–L5 difficulty          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ ARCTask
┌──────────────────────────────▼──────────────────────────────────────┐
│                       BLACKBOARD (memory.py)                         │
│  Shared working memory • Hypothesis stack (max 50) • Surprise history│
│  Contradiction log (max 100) • Agent call log • Meeting agenda       │
└──┬───────────────────────────────────────────────────────────────┬──┘
   │ reads/writes                                              reads/writes
┌──▼──────────────────────────────────────────────────────────────▼──┐
│                       COUNCIL (council.py)                           │
│                                                                      │
│  Perceiver → Dreamer → Scientist → Skeptic → Philosopher             │
│  CausalReasoner → CuriosityEngine → Metacognitor → Archivist        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ snapshots
┌──────────────────────────────▼──────────────────────────────────────┐
│                      DASHBOARD (LAteNT.py)                           │
│  Streamlit live dashboard • 7 tabs • 40+ scientific visualizations   │
│  Real-time agent dialogue • Surprise metric • Skill library browser  │
└─────────────────────────────────────────────────────────────────────┘
```

### 🗄 The Blackboard: Shared Free Energy Substrate

`memory.py` implements the central knowledge substrate that all agents read from and write to. This is not message-passing — it is a **structured shared memory model** (Newell's Blackboard Architecture, 1962), extended with typed constraints and convergence signals.

**Core data structures:**

```python
class HypothesisStatus(Enum):
    PENDING    → TESTING → FALSIFIED
                         → CAUSAL_LAW → ACCEPTED
                         → COINCIDENCE
```

| Structure | Description | Capacity |
|-----------|-------------|----------|
| `Hypothesis` | Candidate DSL program + predicted grid + confidence + MDL score + causal verdict | Stack of 50 |
| `ContradictionEntry` | Falsification event: counter-example, failure mode, originating agent | Log of 100 |
| `WorldState` | Perceiver output: segmented objects, bounding boxes, grid shape | Per-round |
| `SurpriseTracker` | Rolling prediction error L2 distance; plateau detection | Real-time |
| `EpisodeRecord` | Archived task: priors, winning program, rounds, verdict, dialogue log | 500 max |
| `DSLSkillLibrary` | Indexed primitive library with usage counts and success rates | Unbounded |

**Hypothesis ranking composite score:**
```
score(h) = confidence(h) × (1 − false_positive_rate(h)) × (1 + causal_bonus(h))
where causal_bonus = 1.0 if h.status == CAUSAL_LAW else 0.0
```

**Blackboard invariants enforced at runtime:**
- No agent reads from another agent's private state (none exists)
- All mutations are timestamped and logged to `agent_call_log`
- Budget is decremented by the blackboard, not by agents
- `final_verdict` transitions are irreversible once set

---

### 🔧 Domain-Specific Language

The DSL (`council.py`, ~100 lines) defines 10 reversible, compositional transformation primitives over integer grids in the 10-color ARC color space:

```python
DSL.PRIMITIVES = {
    "rotate90":         np.rot90(g, 1),
    "rotate180":        np.rot90(g, 2),
    "rotate270":        np.rot90(g, 3),
    "mirror_h":         np.fliplr(g),
    "mirror_v":         np.flipud(g),
    "gravity_down":     # non-bg cells settle downward per column
    "gravity_up":       # non-bg cells settle upward per column
    "majority_recolor": # all non-bg → most frequent color
    "sort_by_size":     # objects reordered left→right by ascending cell count
    "identity":         # no-op baseline
}
```

Programs are sequences of primitive names serialized as `"prim_a → prim_b → prim_c"`. The **MDL score** is simply the program length — shortest programs that generalize win. This implements the **Minimum Description Length principle**: if two programs fit all training examples equally well, prefer the simpler one.

```python
DSL.execute(grid, ["gravity_down", "mirror_h"])  # returns np.ndarray
DSL.mdl_score(["gravity_down", "mirror_h"])       # returns 2.0
```

**Design rationale for primitives:**
Each primitive preserves semantic content (colors, cell counts) while transforming spatial structure. This ensures that the program search space is tractable (10^6 programs of length ≤6), verifiable in O(H×W) per training pair, and interpretable to humans.

---

### 🤖 The Nine-Agent Council

Each agent implements a distinct epistemological function. No agent duplicates another's role.

---

#### Agent 1 — Perceiver 👁️
**Role:** Object segmentation → WorldState

Performs connected-component analysis on raw integer grids to extract discrete objects. Each object is characterized by `{id, color, cells:[(r,c),...], bbox:(r0,c0,r1,c1), size}`. The Perceiver is the only agent with direct read access to the raw grid; all subsequent agents operate on the structured `WorldState`.

**Invariant:** Invoked first in every episode; re-invoked after Philosopher reframing.

---

#### Agent 2 — Dreamer 💭
**Role:** Stochastic hypothesis generation

Generates `K=8` candidate output grids per invocation by sampling DSL primitive compositions:

1. Sample program length uniformly from `{1, 2, 3}`
2. With 50% probability, prepend a primitive biased by prior art hints from the Archivist
3. Fill remaining slots with uniform samples from `DSL.PRIMITIVES`
4. Deduplicate consecutive identical primitives
5. Execute on all training inputs; compute confidence as fraction of training pairs correctly produced
6. Push to Blackboard if confidence ≥ 0.3

**Exploration bias:** Primitives in the skill library receive 10× higher sampling weight.

---

#### Agent 3 — Scientist 🔬
**Role:** MDL-optimal program synthesis

Runs `MCTS_ROLLOUTS=60` random program searches per invocation. For each candidate program, checks generalization across all training pairs. If a program perfectly generalizes, scores it with MDL. Attaches the shortest generalizing program to the top hypothesis and marks it `TESTING`.

```python
def _generalizes(program, task) -> bool:
    for inp, expected in task.train_pairs:
        produced = DSL.execute(inp, program)
        if produced.shape != expected.shape: return False
        if not np.array_equal(produced, expected): return False
    return True
```

**Key insight:** The Scientist inverts typical program synthesis. Rather than searching for outputs given a program, it searches for a program that explains the Dreamer's already-imagined output. This dramatically narrows the search space.

---

#### Agent 4 — Skeptic 🔴
**Role:** Adversarial falsification (Popperian)

Applies `MUTATION_COUNT=12` structural mutations to training inputs and checks program stability. Mutations include: color swaps, noise injection, grid shifts. A single failing mutation falsifies the hypothesis and logs a `ContradictionEntry`.

```python
# If mutation of a well-defined input causes shape mismatch under a supposedly general program:
entry = ContradictionEntry(
    hypothesis_id=top_h.id,
    failure_mode="shape_mismatch_under_mutation",
    agent="Skeptic"
)
```

The Skeptic embodies Karl Popper's demarcation criterion: a hypothesis is scientific only if it is falsifiable and has survived falsification attempts.

---

#### Agent 5 — Philosopher 🏛️
**Role:** Ontological reframing

When the Curiosity Engine detects persistent failure, the Philosopher challenges the Perceiver's fundamental object decomposition. It proposes two alternative segmentation schemes:

- **Revision 0 → 1:** Include the background color as an explicit object (relevant when the task manipulates "holes" or "enclosed regions")
- **Revision 1 → 2:** Merge all cells of the same color into a single object regardless of connectivity (relevant for color-based counting tasks)

After reframing, the Perceiver is immediately re-invoked with the new `WorldState`. This allows the council to discover that the task requires treating "colors as objects" rather than "shapes as objects."

---

#### Agent 6 — CausalReasoner 🕸️
**Role:** Counterfactual causal verification

Tests whether the Scientist's program is a **causal law** or a **spurious coincidence** via `COUNTERFACTUAL_COUNT=8` single-variable interventions:

```python
def _intervene(grid) -> np.ndarray:
    # Single-variable counterfactual: change one cell's color
    out[r, c] = random_color_not_equal_to(out[r, c])
    return out
```

**Verdict logic:**
- If `program(intervened_input) == program(original_input)` despite `intervened_input ≠ original_input`, the program is **insensitive** to its inputs → `COINCIDENCE`
- Verdict `CAUSAL_LAW` requires sensitivity in fewer than half of counterfactual tests

This prevents the classic failure mode where a program accidentally satisfies all training examples by returning a constant or near-constant output.

---

#### Agent 7 — CuriosityEngine ⚡
**Role:** Active Inference / surprise monitoring

Computes prediction error as normalized L2 distance between the top hypothesis's predicted grid and the ground-truth test output. Tracks error over time via `SurpriseTracker`:

```python
error_t = ‖predicted - actual‖₂ / (H × W × 9)   # normalized to [0, 1]
```

**Plateau detection:** If `|error[t] - error[t-2]| < ε` for two consecutive observations, the tracker declares `is_plateauing=True`.

**Directives issued on plateau:**

| Condition | Directive |
|-----------|-----------|
| ≥3 falsified hypotheses | `PHILOSOPHER_REFRAME` |
| Odd intervention count | `DREAMER_EXPLORE_LOW_CONFIDENCE` |
| Even intervention count | `SCIENTIST_EXTEND_SEARCH` |

This implements **active inference** in the sense of Friston: the agent drives itself toward states of minimal free energy (prediction error), not by passively waiting for better hypotheses, but by actively redirecting the council's exploration.

---

#### Agent 8 — Metacognitor 🧭
**Role:** Meeting chair, agenda setter, convergence arbiter

The Metacognitor has no domain knowledge. Its sole function is **meta-cognitive orchestration**:

1. Reads `curiosity_directive` from the CuriosityEngine
2. Reads hypothesis stack state from the Blackboard
3. Constructs an ordered `agenda` of agents to invoke this round
4. When budget is critical, triggers the convergence vote:

```python
# Convergence vote mechanism
winner = max(candidates,
    key=lambda h: h.confidence * (1.0 if h.causal_verdict == "CAUSAL_LAW" else 0.5))

if winner.confidence >= 0.30:
    bb.declare_answer(winner.grid, "solved", "Council")
```

**Agenda examples:**

| State | Agenda |
|-------|--------|
| No hypotheses | `[Dreamer, Scientist, Skeptic, CausalReasoner]` |
| Top hypothesis PENDING | `[Scientist, Skeptic, CausalReasoner]` |
| Top hypothesis FALSIFIED | `[Dreamer, Scientist, Skeptic, CausalReasoner]` |
| Curiosity says PHILOSOPHER_REFRAME | `[Philosopher, Perceiver, Dreamer, Scientist, Skeptic, CausalReasoner]` |

---

#### Agent 9 — Archivist 📚
**Role:** Episodic memory, skill extraction, prior art injection

The Archivist bridges episodes. At episode start, it retrieves `k=3` most similar past episodes (matched by prior overlap) and injects their winning programs as hints for the Dreamer. At episode end, it archives the full episode record and extracts skill primitives:

```python
for prim_name in winning_program.split(" → "):
    skill_lib.add_skill(SkillPrimitive(
        name=prim_name,
        origin_task_id=task.task_id,
        description=f"Used to solve {task.transformation_description}"
    ))
```

The skill library implements a form of **Bayesian program learning**: primitives that appear in successful programs are biased toward in future hypothesis generation, producing a soft meta-learning signal without gradient descent.

---

### 🔄 Council Meeting Protocol

The main loop (`council.py → Council.solve()`) is a **streaming generator** — every agent action yields a `Blackboard.snapshot()` for real-time dashboard rendering:

```
PHASE 0: ORIENTATION
  └─ Perceiver.perceive(test_input)
  └─ Archivist.inject_hints()

PHASE 1: FIRST IMAGINATION
  └─ Dreamer.imagine(K=8 hypotheses)

PHASE 2: MAIN DEBATE LOOP [repeat until solved or MAX_ROUNDS=30]
  └─ Metacognitor.arbitrate() → sets agenda
  └─ For each agent in agenda:
      ├─ Scientist.synthesize()   → attaches program to top hypothesis
      ├─ Skeptic.challenge()      → falsifies or passes
      │   └─ if fail: CuriosityEngine.observe() → sets directive → break round
      ├─ CausalReasoner.verify()  → CAUSAL_LAW or COINCIDENCE
      │   └─ if CAUSAL_LAW: declare_answer("solved") → exit
      ├─ Dreamer.imagine()        → if directed
      └─ Philosopher.reframe()    → if directed → re-invoke Perceiver
  └─ CuriosityEngine.observe() [end-of-round]

PHASE 3: ARCHIVAL
  └─ Archivist.archive()
```

**Budget accounting:** Each round costs 1 budget unit. Budget runs from 0 to 100; `budget_critical` triggers at 85. This forces convergence under uncertainty.

---

## 🌌 The ARC-AGI-2 Universe

`universe.py` implements a **zero-cheat procedural task generator**. "Zero-cheat" means no task is ever repeated — each task receives a unique fingerprint (`SHA-256(transformation_composition)`), and the Blackboard enforces that the same fingerprint cannot appear twice in a session.

### Core Knowledge Priors

The task generator is grounded in **Spelke's Core Knowledge Theory** — the set of innate concepts human infants possess by ~6 months of age:

| Prior | Enum | Description |
|-------|------|-------------|
| OBJECTNESS | `Prior.OBJECTNESS` | Discrete objects persist through transformation |
| NUMEROSITY | `Prior.NUMEROSITY` | Quantities are conserved or predictably altered |
| SYMMETRY | `Prior.SYMMETRY` | Spatial invariances constrain valid transformations |
| CAUSALITY | `Prior.CAUSALITY` | Transformations exhibit consistent causal structure |
| CONTAINMENT | `Prior.CONTAINMENT` | Objects may contain other objects |
| GRAVITY | `Prior.GRAVITY` | Non-background elements settle toward edges |
| GOAL_DIRECTEDNESS | `Prior.GOAL` | Transformations optimize toward target configurations |

These priors are combined compositionally at difficulty level `L` by selecting `L` priors and chaining their corresponding transformation primitives.

### Difficulty Levels

| Level | Priors | Task Example | Complexity |
|-------|--------|-------------|------------|
| L1 | 1 | `mirror_v` | Trivially generalizable |
| L2 | 2 | `gravity_down → mirror_h` | Requires composition |
| L3 | 3 | `rotate90 → sort_by_size → majority_recolor` | Multi-step reasoning |
| L4 | 4 | 4-primitive chain | Expert-level ARC difficulty |
| L5 | 4+ | Chained dependencies | Frontier-level |

### Task Synthesis Pipeline

```python
1. Sample priors(difficulty_level)          # e.g., [GRAVITY, SYMMETRY]
2. Compose transform_fn = T₁ ∘ T₂ ∘ ... Tₙ
3. Generate n_train=3 random input grids    # 5×5 to 30×30, 1–15 objects
4. Produce outputs = [transform_fn(inp) for inp in inputs]
5. Generate test_input (same prior structure, different grid)
6. Compute test_output = transform_fn(test_input)   # ground truth, hidden from agents
7. Fingerprint = SHA-256(repr(transform_fn))
```

Grid specifications:
- **Dimensions:** 5×5 to 30×30 (procedurally bounded by prior requirements)
- **Colors:** 10-color ARC standard palette (`{0:black, 1:blue, 2:red, ..., 9:purple}`)
- **Objects per grid:** 1 to 15
- **Training pairs:** 3 examples per task

---

## 📊 Live Dashboard

The Streamlit dashboard (`LAteNT.py`, ~1880 lines) provides a real-time scientific interface with **7 tabs** and **40+ visualizations** across **10 observatory sections**:

| Tab | Contents |
|-----|----------|
| 🏛️ **Council Chamber** | Live agent dialogue stream • Answer comparison grid • Cell accuracy badge • Hypothesis breakdown |
| ⚡ **Surprise Metric** | Prediction error curve • Resolution stats • Convergence status |
| 🔬 **Program Inspector** | Discovered DSL rule • MDL score • Step-by-step execution trace • Applied to training example |
| 🔴 **Skeptic's Dossier** | Contradiction log • All hypotheses table • Falsification breakdown |
| 📉 **Generalization Curve** | Rounds-to-solve over time • Rolling mean • Solve rate trend |
| 📚 **Skill Library** | Usage bar chart (builtin vs discovered) • Full skill dataframe with pseudocode |
| 🔭 **Observatory** | 10 sections, 40+ charts (full breakdown below) |

### Observatory Sections

```
A — Hypothesis Manifold
    A1 Confidence Cascade (horizontal bar, plasma colormap)
    A2 Status Mosaic (donut chart)
    A3 MDL Score Waterfall (cool colormap)
    A4 Contradiction Pressure (polar bar chart)
    A5 Confidence × Age Heatmap (inferno scatter)

B — Free Energy & Surprise
    B1 Free Energy Landscape (segmented gradient fill)
    B2 Surprise Gradient dE/dt (green/red bars)
    B3 Active Inference Phase Space (E[t-1] vs E[t])
    B4 Resolution Speedometer (polar gauge)
    B5 Entropy Reduction Timeline (cumulative area)

C — Agent Council Activity
    C1 Agent Brain Heatmap (9 agents × N rounds)
    C2 Council Speaking Clock (polar bar)
    C3 Agent Activation Gantt (horizontal timeline)
    C4 Dialogue Density Wave (stacked area)
    C5 Agent Co-activation Matrix (viridis)

D — Skill Meme Grid
    D1 Skill Meme Grid (HSV pixel mosaic, brightness ∝ usage)
    D2 Skill Usage Heatmap (YlOrRd)
    D3 Success Rate Radar (top 8 skills)
    D4 Discovery Timeline (builtin vs emergent)
    D5 Skill Gravity Well (bubble chart)

E — Program Structure Analysis
    E1 Program Length Distribution
    E2 MDL vs Confidence Scatter (magma)
    E3 Primitive Co-occurrence Matrix (RdYlGn)
    E4 Winning Program Spotlight (colorized pipeline)
    E5 Confidence Distribution (histogram)

F — Causal Reasoning Engine
    F1 Causal Law vs Coincidence bar chart
    F2 Falsification Heatmap (Agent × failure_mode)
    F3 Causal Confidence Scatter
    F4 Skeptic Contradiction Spiral (polar)
    F5 Causal Law Rate Over Hypotheses (cumulative %)

G — World State & Perception
    G1 Object Color Distribution (polar)
    G2 Object Size Histogram
    G3 Color Transition Matrix Input→Output (plasma, annotated)
    G4 Philosopher Revision Depth Gauge
    G5 Object Bounding Box Map

H — Multi-Episode Meta-Learning
    H1 Rounds to Solve Learning Curve
    H2 Cumulative Solve Rate
    H3 Difficulty vs Rounds Scatter
    H4 Final Surprise per Episode
    H5 Budget Efficiency per Episode

I — Curiosity Engine Deep Dive
    I1 Directive Frequency
    I2 Plateau Detection on Surprise
    I3 Surprise Spectrum (1D heartbeat colorbar)
    I4 Curiosity Engine Stats box
    I5 Free Energy Convergence (log scale)

J — Emergent Intelligence Metrics
    J1 GI Progress Multi-Ring Gauge (5 metrics)
    J2 Intelligence Fingerprint Radar (8 dimensions)
    J3 Metacognitor Activity Heatmap
    J4 Council Consensus Heat
    J5 System Complexity Score Timeline + Composite GI Score
```

**Composite General Intelligence Score** (Section J5) is a session-level aggregate:
```
GI_score = mean([solve_rate, skill_reuse, surprise_decay,
                 causal_law_rate, budget_efficiency, round_efficiency])
```

---

## 📈 Experimental Results

All results reproducible with `seed=11290`.

### Overall Performance

| Metric | Value |
|--------|-------|
| **Solve Rate** | **18/18 (100%)** |
| Avg. Rounds to Solve | 15.9 ± 3.2 |
| Budget Utilization | 53% (15.9 / 30) |
| Worst Case | 23 rounds (T0013, L3) |
| Best Case | 14 rounds (T0001, T0005, T0006, T0007) |

### Per-Task Results

```
Task ID                       Difficulty  Rounds  Verdict
───────────────────────────────────────────────────────────
T0000_7046e3eef9c38598        L1          15      ✅ SOLVED
T0001_f0ff7e211c60a023        L1          14      ✅ SOLVED
T0002_1e94c74b1c4cd52a        L2          15      ✅ SOLVED
T0003_44f0bbbd3ae17296        L2          15      ✅ SOLVED
T0004_a633ef1a1bbaadbb        L2          15      ✅ SOLVED
T0005_a31a626e619c8024        L2          14      ✅ SOLVED
T0006_9224a01a0b0e5d79        L2          14      ✅ SOLVED
T0007_3eb91af5049b67fe        L2          14      ✅ SOLVED
T0008_efb451d8fa4c9405        L3          15      ✅ SOLVED
T0009_5b46400abc669d5a        L3          15      ✅ SOLVED
T0010_d4dad90e496df51f        L3          18      ✅ SOLVED
T0011_4689eee23a368e7b        L3          17      ✅ SOLVED
T0012_63a7c7a103865c22        L3          15      ✅ SOLVED
T0013_4fba135a50799a8e        L3          23      ✅ SOLVED  ← hardest
T0014_35ae0bce04df14d6        L3          19      ✅ SOLVED
T0015_ca91975b5d86d463        L3          18      ✅ SOLVED
T0016_7bde9a0ea5fce54e        L3          16      ✅ SOLVED
T0017_a8005ea73b232c2d        L3          15      ✅ SOLVED
```

### Skill Library Growth

| Metric | Value |
|--------|-------|
| Initial builtin primitives | 10 |
| Total skills after 18 episodes | 17 |
| Emergent (discovered) skills | 7 |
| Transfer rate (emergent → reused) | 5/7 (71%) |

**Top 5 skills by usage:**
1. `majority_recolor` — 6 uses, **100% success rate**
2. `gravity_down` — 4 uses, **100% success rate**
3. `gravity_up` — 4 uses, **100% success rate**
4. `mirror_v` — 3 uses, **100% success rate**
5. `sort_by_size` — 3 uses, **100% success rate**

### Agent Contribution (avg invocations/task)

| Agent | Avg. Invocations | Function |
|-------|-----------------|----------|
| Dreamer | 6.2 | Hypothesis generation |
| Metacognitor | 6.2 | Agenda + convergence |
| Scientist | 6.1 | Program synthesis |
| Skeptic | 6.0 | Falsification |
| CuriosityEngine | 6.0 | Surprise tracking |
| CausalReasoner | 5.8 | Causal validation |
| Philosopher | 1.8 | Ontological reframing |
| Perceiver | 1.4 | Segmentation |
| Archivist | 1.0 | End-of-task archival |

---

## ⚙️ Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Perceiver (segmentation) | O(H × W) | O(n_objects) |
| Dreamer (hypothesis gen) | O(K × n_train × L) | O(K) |
| Scientist (MDL search) | O(\|DSL\|^L × n_train) | O(\|DSL\|^L) |
| Skeptic (falsification) | O(n_hyp × n_train) | O(1) streaming |
| CausalReasoner | O(C × L) | O(C) |
| **Session total** | O(MAX_ROUNDS × \|DSL\|^L × n_train) | O(MAX_HYPO + EPISODE_CAP) |

For typical parameters (`MAX_ROUNDS=30, |DSL|=10, L=6, n_train=3`):
~100M primitive operations per task, all in NumPy vectorized ops.

---

## 📁 Codebase Tour

```
Latent-Consensus-Manifold/
├── LAteNT.py            # Main Streamlit dashboard (~1880 lines)
│   ├── ARC_CMAP         # ARC 10-color palette → ListedColormap
│   ├── _render_grid()   # Dark-themed ARC grid renderer
│   ├── _grid_fig()      # Multi-panel figure builder
│   ├── _agent_html()    # Colored agent dialogue HTML
│   ├── _verdict_badge() # Status badge renderer
│   ├── _winning_program() # Extracts accepted program from snapshot
│   ├── _answer_grid()   # Re-executes program on test input
│   ├── Sidebar          # Difficulty selector + session stats + export
│   ├── Tab 1–6          # Core analysis tabs
│   └── Tab 7            # Observatory (sections A–J)
│
├── council.py           # The 9-agent council (~962 lines)
│   ├── DSL              # Primitive interpreter + program execution
│   ├── AgentResult      # Typed return value dataclass
│   ├── Perceiver        # Connected-component segmentation
│   ├── Dreamer          # Stochastic hypothesis generator
│   ├── Scientist        # MDL program synthesizer (MCTS-style)
│   ├── Skeptic          # Adversarial falsifier (Popperian)
│   ├── Philosopher      # Ontological reframer
│   ├── CausalReasoner   # Counterfactual verifier
│   ├── CuriosityEngine  # Active inference / surprise tracker
│   ├── Metacognitor     # Meeting chair + convergence vote
│   ├── Archivist        # Episode memory + skill extraction
│   └── Council          # Main orchestration loop (streaming generator)
│
├── memory.py            # Shared state substrate (~565 lines)
│   ├── HypothesisStatus # Enum: PENDING→TESTING→{FALSIFIED,CAUSAL_LAW,ACCEPTED}
│   ├── Hypothesis       # Candidate program + grid + metadata
│   ├── ContradictionEntry # Falsification event record
│   ├── WorldState       # Perceiver output: segmented objects
│   ├── Blackboard       # Central mutable state (all agent I/O)
│   ├── EpisodeMemory    # Persistent episode archive (FIFO, 500 cap)
│   ├── DSLSkillLibrary  # Primitive library with usage tracking
│   └── SurpriseTracker  # Rolling L2 error + plateau detector
│
├── universe.py          # Procedural task generator (~503 lines)
│   ├── Prior            # Enum: 7 core knowledge priors
│   ├── DifficultyLevel  # Enum: L1–L5
│   ├── GridObject       # Segmented object dataclass
│   ├── ARCTask          # Task specification + train/test pairs
│   ├── GridTransforms   # All atomic transformation implementations
│   └── Universe         # Zero-cheat procedural task factory
│
├── meta_learner.py      # (in development)
├── latent_dictionary.py # (in development)
│
├── requirements.txt
├── plan.md              # True AGI research roadmap
├── LICENSE              # Apache 2.0
└── Results Archive/
    ├── agi_session_11290.json          # 18-task session (seed 11290)
    ├── agi_session_11290_latest.json
    ├── agi_session_70290.json          # 18-task session (seed 70290)
    ├── general_intelligence_session_6262.json
    └── readme.md                       # Full technical paper
```

---

## 🚀 Installation & Usage

### Requirements

```
Python >= 3.10
numpy >= 1.24.0
scipy >= 1.11.0
streamlit >= 1.32.0
matplotlib >= 3.7.0
pandas >= 2.0.0
```

### Install

```bash
git clone https://github.com/Devanik21/Latent-Consensus-Manifold.git
cd Latent-Consensus-Manifold
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run LAteNT.py
```

Open `http://localhost:8501`. Select a difficulty level in the sidebar and click **⚡ Run Council**.

### Scripted Usage

```python
from universe import Universe, DifficultyLevel
from council import Council

# Reproducible session
universe = Universe(seed=11290)
council = Council(seed=11290)

# Generate and solve a task
task = universe.generate_task(DifficultyLevel.L2)
print(f"Task: {task.task_id}")
print(f"Rule (hidden from agents): {task.transformation_description}")

# council.solve() is a streaming generator
final_snapshot = None
for snapshot in council.solve(task):
    final_snapshot = snapshot
    print(f"[Round {snapshot['round']}] {snapshot['final_verdict']}")

print(f"\nVerdict : {final_snapshot['final_verdict']}")
print(f"Rounds  : {final_snapshot['round']}")
print(f"Budget  : {final_snapshot['budget_used']}/100")

# Session-level meta-learning stats
stats = council.stats()
print(f"Skills discovered: {stats['skill_library_size']}")
print(f"Avg rounds: {stats['avg_rounds']}")
```

### Export Session Data

The dashboard provides a **💾 Download Session Data** button in the sidebar after ≥1 task run. The exported JSON contains:

```json
{
  "seed": 11290,
  "tasks_run": 18,
  "solved": 18,
  "avg_rounds": 15.9,
  "skills": [...],
  "generalization": [...],
  "cumulative_dialogue_logs": [...]
}
```

### Self-Test

```bash
python council.py
# Runs 3 tasks (L1, L2, L1) and prints per-task results
```

---

## 🗺 Roadmap — True AGI Plan

The current system is an explicitly symbolic baseline. The research roadmap targets a system where **nothing is defined — agents must discover everything from data alone**.

### Phase 1 — Latent Transformation Learning *(3–4 weeks)*
Replace the handcrafted DSL with **learned transformation embeddings**:
- Collect 1000+ `(input→output)` pairs across diverse domains
- Train a transformation autoencoder: each transformation → point in continuous latent space
- Agents sample from this space rather than from a fixed primitive list
- Result: agents discover "rotation-like" operations without being told what rotation is

### Phase 2 — Abstraction Discovery Through Pure Induction *(4–5 weeks)*
Remove the DSL interpreter entirely. Replace program synthesis with **latent space search**:
- The Scientist searches the learned transformation latent space (no discrete primitives)
- The Dreamer samples latent transformations and imagines hypothetical outputs
- Discovery of novel compositions through interpolation in latent space
- No `IF-THEN`. No hardcoded logic. Pure causal inference in continuous space.

### Phase 3 — Cross-Domain Generalization Test *(3 weeks)*
The real measure of AGI:
- Train on Task Domain A (30 tasks)
- Validate on Domain B (30 tasks, different visual properties)
- Test on Domain C (30 tasks, completely unseen transformation types)
- Target: **50%+ on Domain C** without having seen that domain's pattern types

### Phase 4 — Meta-Learning (Learning to Learn Faster) *(3–4 weeks)*
- Track episode-level discovery speed
- Train a meta-learner to predict which hypothesis directions are most promising
- The meta-learner shapes the Dreamer and Scientist purely from episode history
- Result: system improves its own discovery process through meta-reasoning, not programmer-defined improvements

### Phase 5 — Scientific Validation & Emergence Analysis *(2 weeks)*
- Does the system discover novel transformation types not in training data?
- Do agents develop implicit strategies without explicit programming?
- Can the learned transformation latent space be visualized? Do similar operations cluster?
- Publish findings: *"Learning Transformation Abstractions Without Explicit Rules"*

**Timeline:** 15–17 weeks to top-0.1% territory.
**Success threshold:** 50%+ cross-domain transfer + demonstrated meta-learning improvement + published emergence analysis.

**Ultimate goal:** A true AGI capable of solving any problem a human can solve — with infinite tools, skills, knowledge, and memories discovered autonomously.

---

## ⚠️ Limitations & Honest Assessment

This system achieves 100% solve rate on a task distribution it was designed for. That is not the same as general reasoning capability.

**1. DSL Expressiveness Ceiling**
The 10 builtin primitives cover a restricted subset of abstract reasoning. Tasks requiring conditional logic, counting, connectivity-based transformations, or arbitrary color mappings cannot be solved by construction. The 100% solve rate reflects task-distribution alignment.

**2. Program Search Tractability**
Scientist search is capped at programs of length ≤6. Solutions requiring longer compositions will not be found within budget.

**3. Hypothesis Stack Saturation**
Under sustained Dreamer output (8 hypotheses × 20 rounds = 160 generated vs. stack cap of 50), older hypotheses with initially low confidence are discarded. This may cause premature elimination of correct-but-initially-uncertain hypotheses.

**4. Zero-Shot Regime**
Without training examples, confidence computation is undefined. The system would degrade to random hypothesis sampling.

**5. Transfer Scope**
The skill library enables transfer within the same task distribution. Generalization to fundamentally different task types (e.g., trained on rotation-type tasks, tested on connectivity-based tasks) is constrained by DSL expressiveness.

**6. Causal Reasoning Depth**
Single-variable counterfactual testing (changing one cell) is a weak causal intervention. Real causal graph discovery requires structured interventions at the object level, which the current CausalReasoner does not implement.

---

## 📖 References

**Program Synthesis**
- Gulwani, S. (2015). Dimensions in Program Synthesis. *PPLJ*.
- Solar-Lezama, A. (2008). Program Synthesis by Sketching. *PhD dissertation, UC Berkeley*.

**Multi-Agent Systems**
- Stone, P., & Veloso, M. (2000). Multiagent systems: A survey from an AI perspective. *Autonomous Robots, 8(3)*, 345–383.
- Newell, A. (1962). Some problems of basic organization in problem-solving programs. *Self-Organizing Systems*.

**Active Inference & Free Energy**
- Friston, K., et al. (2017). Active inference and learning. *Neuroscience & Biobehavioral Reviews*.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience, 11(2)*, 127–138.

**Causal Inference**
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference. *Cambridge University Press*.
- Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference. *MIT Press*.

**Minimum Description Length**
- Rissanen, J. (1978). Modeling by shortest data description. *Automatica, 14(5)*, 465–471.
- Grünwald, P. (2007). The Minimum Description Length Principle. *MIT Press*.

**ARC Benchmark**
- Chollet, F. (2019). On the Measure of Intelligence. *arXiv:1911.01547*.

**Core Knowledge Theory**
- Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge. *Developmental Science, 10(1)*, 89–96.

**Falsificationism**
- Popper, K. (1959). The Logic of Scientific Discovery. *Hutchinson & Co*.

---

## 🔗 Contact

<div align="center">

**Devanik**
B.Tech ECE '26 • National Institute of Technology Agartala
Samsung Convergence Software Fellowship (Grade I) • Indian Institute of Science

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-181717?style=for-the-badge&logo=github)](https://github.com/Devanik21)
[![Twitter](https://img.shields.io/badge/Twitter-@devanik2005-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/devanik2005)
[![Email](https://img.shields.io/badge/Email-devanik%40iisertirupati.ac.in-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:devanik@iisertirupati.ac.in)

<br/>

*This work represents independent research conducted during the Samsung Convergence Software Fellowship at the Indian Institute of Science. All code, experimental data, and analysis are made available for academic and research purposes.*

</div>

---

<div align="center">

**License:** [Apache 2.0](LICENSE) • **Last Updated:** March 2026

*Built with deliberate constraints — not to beat the leaderboard, but to understand reasoning itself.*

</div>
