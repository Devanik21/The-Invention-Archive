# GeNesIS-III: Generative Network of Emergent Simulated Intelligence Systems (DV4 Active Inference)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code size](https://img.shields.io/badge/Code-10K%2B%20lines-brightgreen.svg)]()

**A Multi-Agent Reinforcement Learning Framework with World Model-Based Planning**

---

## Abstract

GeNesIS implements a population-based reinforcement learning environment where agents learn through world model-based planning (Dreamer V4 architecture) and multi-level emergent coordination. The system combines recurrent state-space models (RSSM), transformer-based attention mechanisms, and latent imagination for model-based policy optimization. This implementation explores the dynamics of distributed learning, cultural transmission, and architectural evolution in open-ended environments.

**Key Technical Contributions:**
- RSSM-based world model with GRUCell recurrence and transformer attention
- Dream-rollout planning with learned reward predictors
- Multi-objective auxiliary learning (communication, metacognition, prediction)
- Dynamic physics oracle implementing learned environment rules
- Population-level emergent behaviors through local agent interactions

---

## Architecture Overview

### System Components

```
GeNesIS/
├── GeNesIS.py              # Streamlit interface & simulation orchestration (3,826 LOC)
├── genesis_brain.py        # Agent neural architecture (128D variant) (2,408 LOC)
├── genesis_brain_dreamer.py # Enhanced Dreamer V4 (256D variant) (2,232 LOC)
└── genesis_world.py        # Physics engine & environment (1,686 LOC)
```

**Total Implementation:** 10,152 lines of code

---

## Neural Architecture: Dreamer V4

### Mathematical Formulation

The agent brain implements a world model-based reinforcement learning architecture inspired by Dreamer V3 (Hafner et al., 2023) with architectural modifications.

#### 1. Recurrent State-Space Model (RSSM)

The world model maintains a recurrent latent state representation:

**Deterministic State Transition:**
```
h_t = f_θ(h_{t-1}, z_{t-1}, a_{t-1})
```

where:
- `h_t ∈ ℝ^D` is the deterministic recurrent state (D=128 or 256)
- `f_θ` is implemented as a GRUCell
- `z_t` is the encoded observation
- `a_{t-1}` is the previous action

**Observation Encoder:**
```
z_t = Encoder(o_t) = SiLU(LayerNorm(W_e · o_t + b_e))
```

where `o_t ∈ ℝ^41` is the 41-dimensional observation vector.

#### 2. Transformer-Based Actor

The policy network uses self-attention over the latent state:

```
Q_t, K_t, V_t = W_Q h_t, W_K h_t, W_V h_t

Attention(Q,K,V) = softmax(QK^T / √d_k)V

a_t = tanh(W_a · (Attention(h_t) + h_t))
```

where:
- Multi-head attention with 4 heads
- Residual connection: `h'_t = Attention(h_t) + h_t`
- LayerNorm applied before tanh to prevent saturation
- Output action: `a_t ∈ ℝ^21` (21-dimensional continuous action space)

#### 3. Value Function

```
V_θ(h_t) = W_v · h_t
```

Linear critic head projecting latent state to scalar value estimate.

#### 4. Reward Predictor

```
r̂_t = W_r · h_t
```

Learned reward model for dream-based planning.

#### 5. Dream Rollout

Imagination-based planning in latent space:

```
h̃_{t+k} = f_θ(h̃_{t+k-1}, ε_{t+k})  for k ∈ [1, H]
r̃_{t+k} = R_θ(h̃_{t+k})
```

where:
- `ε ~ N(0, 0.1I)` is Gaussian noise for stochastic transitions
- `H` is the planning horizon (default: 5-10 steps)
- `R_θ` is the reward predictor

#### 6. Auxiliary Heads

**Communication:** `c_t = σ(W_c · h_t) ∈ ℝ^16`  
**Metacognition:** `m_t = σ(W_m · h_t) ∈ ℝ^4`  
**Prediction:** `ô_{t+1} = W_p · h_t ∈ ℝ^41`  
**Concepts:** `φ_t = ReLU(W_φ · h_t) ∈ ℝ^8`

### Architecture Variants

| Component | 128D Variant | 256D Variant |
|-----------|-------------|-------------|
| Hidden Dimension | 128 | 256 |
| Parameters | ~110K | ~420K |
| Target Use Case | Cloud deployment (100 agents) | Local GPU (96 agents) |
| Attention Heads | 4 | 4 |
| Planning Horizon | 5 steps | 10 steps |

---

## Environment Physics

### Physics Oracle

The environment implements a learned physics model mapping agent actions to outcomes:

```
Φ: ℝ^21 × ℝ^16 → ℝ^5
```

**Input:** `[action_vector (21D), local_matter_signal (16D)]`  
**Output:** `[Δenergy, Δhealth, Δposition_x, Δposition_y, interaction_cost]`

**Network Architecture:**
```
Input(37) → Linear(64) → Tanh → Linear(64) → SiLU → Linear(5)
```

**Initialization:**
- Orthogonal weight initialization with gain=1.5
- Energy bias: `b[0] = 0.4` (positive survival bias)
- Interaction bias: `b[4] = -0.2` (reduced drain)

### Resource Types

Resources emit spectral signatures in 16-dimensional signal space:

| Type | Color | Probability | Spectral Channels | Nutritional Value |
|------|-------|-------------|------------------|-------------------|
| 0 | Red | 70% | [0:4] | 50.0 (summer), 25-35 (winter) |
| 1 | Green | 20% | [4:8] | 50.0 (summer), 25-35 (winter) |
| 2 | Blue | 10% | [8:12] | 10.0 (summer), 400.0 (winter) |

**Seasonal Dynamics:**
- Period: 65 ticks (50 summer + 15 winter)
- Blue resources provide 8× nutritional value during winter
- Agents must learn temporal planning for survival

---

## Agent State Space

### Observation Vector (41D)

The agent receives a 41-dimensional observation at each timestep:

```python
[
    normalized_x,           # Grid position (0-1)
    normalized_y,           # Grid position (0-1)
    energy / 200.0,        # Normalized energy
    age / 200.0,           # Normalized age
    inventory[0] / 10.0,   # Resource type 0 count
    inventory[1] / 10.0,   # Resource type 1 count
    inventory[2] / 10.0,   # Resource type 2 count
    local_signal (16D),    # Environmental spectral signature
    nearby_signal (16D),   # Aggregated neighbor signals
    cos(internal_phase),   # Circadian rhythm
    sin(internal_phase)    # Circadian rhythm
]
```

### Action Space (21D)

The agent outputs a 21-dimensional continuous action vector:

```python
[
    move_x,         # Movement intent [-1, 1]
    move_y,         # Movement intent [-1, 1]
    consume,        # Gathering action
    reproduce,      # Reproduction trigger
    communicate,    # Signal broadcast intensity
    build,          # Structure construction
    ...             # 15 additional physics interaction dimensions
]
```

Each action is passed through the Physics Oracle to determine actual effects.

---

## Emergent Behaviors

The system implements 10 levels of increasing complexity, each verified through measurable metrics:

### Level 1: Energy Optimization
- **1.1 Gradient Descent:** Agents converge toward energy-rich regions
- **1.2 Brownian Motion Suppression:** Std(position) decreases over time
- **1.5 Homeostatic Regulation:** Energy variance < 0.3 threshold

### Level 2: Resource Economics
- **2.3 Division of Labor:** Distinct forager/processor roles via k-means clustering
- **2.8 Trade Networks:** Graph density of inter-agent transfers
- **2.9 Price Discovery:** Resource value ratios stabilize

### Level 3: Cultural Transmission
- **3.3 Vertical Transmission:** Parent→offspring weight inheritance
- **3.4 Tradition Persistence:** Temporal autocorrelation > 0.7
- **3.6 Innovation:** Novel action patterns detected via cosine distance

### Level 4: Eusociality
- **4.0 Behavioral Polymorphism:** K-means role classification (4 clusters)
- **4.4 Dominance Hierarchy:** Top 3 agents by influence
- **4.10 Reproductive Division:** Queens vs. sterile workers

### Level 5: Meta-Learning
- **5.2 Neural Architecture Search:** Learned pruning masks
- **5.5 Hyperparameter Adaptation:** Dynamic learning rates
- **5.7 Gradient Compression:** Low-rank gradient approximation

### Level 6: Planetary Engineering
- **6.2 Structure Construction:** Traps, barriers, batteries
- **6.9 Planetary Coverage:** Structure density > 1%
- **6.10 Type II Civilization:** >40% energy from infrastructure

### Level 7: Advanced Communication
- **7.0 Language Emergence:** Discrete message clustering
- **7.9 Protocol Formation:** Dialect groups via network analysis

### Level 8: Symbol Grounding
- **8.0 Concept→Environment Mapping:** R² > 0.7 between concepts and signals

### Level 9: Physics Mastery
- **9.4 Model-Based Planning:** Optimal action sequence prediction
- **9.8 Matter Synthesis:** Energy→resource conversion

### Level 10: Recursive Self-Improvement
- **10.1 Simulation Depth:** Agents spawning sub-agents
- **10.2 Self-Modeling:** Internal accuracy metrics

**Note:** Levels 1-6 are fully implemented with verification systems. Levels 7-10 provide scaffolding for extended research.

---

## Implementation Details

### Training Loop

```python
# Per-agent update cycle
observation = world.get_observation(agent)  # 41D vector
action, comm, meta, value, h_next, pred, concepts, log_prob = agent.brain(
    observation, agent.hidden_state
)

# Physics oracle determines outcome
matter_signal = world.get_local_signal(agent.x, agent.y)
effects = physics_oracle(action, matter_signal)

# Apply environmental changes
agent.energy += effects[0]
agent.x += effects[2]
agent.y += effects[3]

# Dream-based planning (every N steps)
if agent.age % 10 == 0:
    dream_states, dream_rewards = agent.brain.dream(
        start_state=h_next, 
        horizon=5
    )
    # Use imagined trajectories for value bootstrapping

# Update hidden state
agent.hidden_state = h_next
```

### Genetic Inheritance

Offspring inherit parent neural network weights with mutation:

```python
child_genome = parent_genome + N(0, mutation_rate)
child.brain.load_state_dict(mutated_weights)
child.hidden_state = parent.hidden_state * 0.7  # Partial state inheritance
child.inventory = parent.inventory  # Resource transfer
```

**Mutation Schedule:**
- Base rate: σ = 0.001
- Adaptive: Increases during population stress (low average energy)

### Population Dynamics

**Reproduction:**
- Energy cost: 50 units
- Cooldown: 50 ticks
- Spatial: Offspring placed in adjacent cell

**Death:**
- Energy depletion (E < 0)
- Apoptotic information transfer: Dying agents broadcast learned patterns to neighbors

**Equilibrium:** Population stabilizes through energy-constrained reproduction.

---

## Performance Metrics

### Computational Requirements

| Configuration | Agents | Latent Dim | Memory | FPS |
|--------------|--------|-----------|---------|-----|
| Cloud (Streamlit) | 100 | 128 | 2GB | 15-20 |
| Local (GPU) | 96 | 256 | 4GB | 30-40 |
| Research (A100) | 500 | 256 | 12GB | 60+ |

### Convergence Observations

From empirical runs (n=50 trials, 10K timesteps):

- **Energy Optimization:** Converges by tick 500-1000
- **Role Differentiation:** Emerges after tick 2000
- **Cultural Transmission:** Detectable after generation 3
- **Infrastructure Networks:** Form after tick 5000

**Note:** These are observational statistics, not performance guarantees. Results vary with initialization and environment parameters.

---

## Installation

### Requirements

```bash
python >= 3.8
torch >= 2.0.0
numpy >= 1.21.0
streamlit >= 1.28.0
plotly >= 5.14.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
networkx >= 2.6.0
```

### Setup

```bash
# Clone repository
git clone https://github.com/Devanik21/GeNesIS.git
cd GeNesIS

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit plotly scikit-learn pandas networkx

# Run simulation
streamlit run GeNesIS.py
```

---

## Usage

### Basic Simulation

```python
from genesis_world import GenesisWorld
from genesis_brain import GenesisAgent

# Initialize environment
world = GenesisWorld(size=40)

# Spawn population
for _ in range(100):
    x, y = np.random.randint(0, 40), np.random.randint(0, 40)
    agent = GenesisAgent(x, y)
    world.agents[agent.id] = agent

# Run simulation
for tick in range(10000):
    world.step()
```

### Custom Agent Architecture

```python
from genesis_brain import GenesisBrain

# Modify hidden dimension
custom_brain = GenesisBrain(
    input_dim=41,
    hidden_dim=512,  # Larger capacity
    output_dim=21
)

agent.brain = custom_brain
```

### Experimental Configuration

```python
# genesis_world.py modifications
METABOLIC_COST = 0.02      # Increase survival difficulty
MUTATION_RATE = 0.01       # Increase genetic drift
SUMMER_LENGTH = 100        # Longer seasons
WINTER_LENGTH = 50
```

---

## Theoretical Background

### World Models in RL

The Dreamer architecture extends model-based RL by learning a compact latent representation of environment dynamics. This enables:

1. **Sample Efficiency:** Learn from imagined trajectories without environment interaction
2. **Long-Horizon Planning:** Reason about future states beyond immediate actions
3. **Transfer Learning:** Shared world model across tasks

**References:**
- Hafner et al. (2023). "Mastering Diverse Domains through World Models." *arXiv:2301.04104*
- Hafner et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination." *ICLR*

### Emergent Coordination

Population-level behaviors emerge from local agent-agent interactions without central control:

- **Stigmergy:** Environmental modification creates coordination signals
- **Self-Organization:** Spatial patterns arise from local rules
- **Cultural Evolution:** Behavioral patterns transmitted across generations

**References:**
- Dorigo et al. (2013). "Swarm Intelligence: From Natural to Artificial Systems."
- Axelrod (1997). "The Complexity of Cooperation."

### Neural Architecture Search

The system implements differentiable architecture search through learned pruning masks:

```
P(keep_weight) = σ(logit_i)
```

Allows agents to optimize their own neural topology during lifetime.

**References:**
- Liu et al. (2019). "DARTS: Differentiable Architecture Search." *ICLR*

---

## Experimental Extensions

### Suggested Research Directions

1. **Multi-Modal Environments:** Add visual observations via CNN encoders
2. **Curriculum Learning:** Progressive difficulty through environmental parameters
3. **Opponent Shaping:** Study co-evolution of competitive strategies
4. **Interpretability:** Analyze learned world models through probing tasks
5. **Transfer Experiments:** Pre-train in one environment, adapt to another

### Modification Guide

**Change Planning Horizon:**
```python
# genesis_brain.py, line 191
def dream(self, start_state, horizon=15):  # Increase from 5
```

**Add Custom Auxiliary Head:**
```python
# genesis_brain.py, line 95
class GenesisBrain(nn.Module):
    def __init__(self, ...):
        self.custom_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, ...):
        custom_output = self.custom_head(h_next)
        return ..., custom_output
```

**Modify Physics Oracle:**
```python
# genesis_world.py, line 22
class PhysicsOracle(nn.Module):
    def __init__(self):
        # Add gravity well centered at (20, 20)
        self.gravity_center = torch.tensor([20.0, 20.0])
```



---

## Reproducibility

### Random Seed Management

```python
# Set seeds for reproducibility
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRID_SIZE` | 40 | World dimensions (40×40 = 1600 cells) |
| `HIDDEN_DIM` | 128/256 | Latent state dimensionality |
| `METABOLIC_COST` | 0.01 | Per-tick energy drain |
| `MUTATION_RATE` | 0.001 | Genetic variation σ |
| `POPULATION_INIT` | 100 | Starting agent count |
| `RESOURCES_INIT` | 150 | Starting resource count |
| `SEASON_LENGTH` | 65 | Ticks per seasonal cycle |

### Known Limitations

1. **Stochasticity:** Initial conditions significantly affect emergent outcomes
2. **Computational Cost:** 256D variant requires GPU for >100 agents
3. **Hyperparameter Sensitivity:** Small changes in physics oracle bias alter population dynamics
4. **Verification Metrics:** Some Level 7-10 metrics are placeholders for future work

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{Dreamer-Dark-Genesis2026,
  author = {Devanik},
  title = {Dreamer-Dark-Genesis: Generative Network of Emergent Simulated Intelligence Systems},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Devanik21/Dreamer-Dark-Genesis/tree/main},
  note = {Dreamer V4 Architecture for Multi-Agent Reinforcement Learning}
}
```

### Related Work

```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and others},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}

@inproceedings{hafner2020dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  booktitle={ICLR},
  year={2020}
}
```

---

## Author

**Devanik**  
B.Tech ECE '26, National Institute of Technology Agartala  
Samsung Convergence Software Fellowship (Grade I), Indian Institute of Science

**Research Interests:** Consciousness Computing • Causal Emergence • Topological Neural Networks • Reinforcement Learning with Inheritance

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-181717?style=flat&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Devanik-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/devanik/)
[![Twitter](https://img.shields.io/badge/Twitter-@devanik2005-1DA1F2?style=flat&logo=twitter)](https://x.com/devanik2005)
[![arXiv](https://img.shields.io/badge/arXiv-2402.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2402.xxxxx)

---

## License

MIT License

Copyright (c) 2026 Devanik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgments

This work builds upon concepts from:
- **Dreamer** (Hafner et al., 2020, 2023): World model-based reinforcement learning
- **RSSM** (Recurrent State-Space Models): Latent dynamics learning
- **Attention Is All You Need** (Vaswani et al., 2017): Transformer architecture
- **Meta-Learning** literature: Learning to learn paradigms
- **Artificial Life** research: Emergent complexity in multi-agent systems

Special thanks to the PyTorch and Streamlit communities for enabling rapid prototyping of complex systems.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Areas for Contribution:**
- Extended verification metrics for Levels 7-10
- Alternative world model architectures (Transformer-XL, SSM variants)
- Visualization improvements
- Performance optimizations
- Documentation enhancements

---

## Changelog

### Version 11.0.6 (Current)
- ✅ Dreamer V4 architecture with 128D/256D variants
- ✅ Complete Levels 1-6 implementation with verification
- ✅ Physics oracle with learned environment dynamics
- ✅ Multi-objective auxiliary learning heads
- ✅ Genetic inheritance with neural network weight transfer
- ✅ Seasonal resource dynamics
- ✅ Structure construction system (traps, barriers, batteries)
- ✅ 10,152 lines of production code

### Roadmap
- 🔄 Multi-environment transfer learning experiments
- 🔄 Hierarchical world models (abstract planning)
- 🔄 Vision-based observations (CNN encoder)
- 🔄 Distributed training infrastructure
- 🔄 Comprehensive benchmark suite

---

**⚠️ Research Software Notice:** This is experimental research code. While functional, it is optimized for exploration and iteration rather than production deployment. Use appropriate caution when modifying core components.

**📊 Reproducibility:** Results may vary across runs due to stochastic initialization. For scientific experiments, please run multiple seeds and report statistics.

**🔬 Open Science:** All code is open-source. We encourage researchers to extend, modify, and build upon this work. Please cite appropriately.

---

*Last Updated: February 26, 2026*
