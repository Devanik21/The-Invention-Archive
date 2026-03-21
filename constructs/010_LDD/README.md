# The Dark Lucid Protocol

<div align="center">

![Version](https://img.shields.io/badge/version-5.2-00FFCC?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-Titan%20Clad-gold?style=for-the-badge)

**Meta-Cognitive Reinforcement Learning Through Internal World Models**

*When the sensors fail, the dream persists.*

[Documentation](#-theory) • [Installation](#-installation) • [Experiments](#-the-omniverse) • [Results](#-empirical-validation)

---

</div>

## 🎯 Overview

The **Dark Lucid Protocol** is a novel reinforcement learning architecture that achieves **+156% performance improvement** over standard baselines by implementing internal world models, causal verification, and adaptive neuro-modulation. Unlike reactive agents, DLP maintains object permanence, predicts future states through "lucid dreaming," and dynamically adjusts its learning strategy based on environmental surprise.

### 🏆 Key Results

| Challenge Domain | Standard DQN | Dark Lucid v5.2 | Improvement |
|:-----------------|-------------:|----------------:|------------:|
| **Memory** (Non-stationary) | 28.31 | **93.90** | **+231.7%** |
| **Exploration** (Blind) | 8.07 | **89.07** | **+1003.4%** |
| **Reasoning** (Deceptive) | 36.63 | **96.99** | **+164.8%** |
| **Filtering** (High-Dim Noise) | 53.40 | **94.09** | **+76.2%** |
| **Permanence** (Sensor Failure) | 58.50 | **99.37** | **+69.9%** |

<div align="center">

**Aggregate Intelligence Gap: +156.03%**

</div>

---

## 🧠 Theory

### Core Architecture

The Dark Lucid Protocol consists of five integrated subsystems:

#### 1. **Universal Encoder** $E: \mathcal{O} \rightarrow \mathcal{Z}$

Transforms raw observations into a compact latent representation:

```math
z_t = E(o_t) = \text{LayerNorm}(\tanh(W_2 \cdot \text{ReLU}(W_1 \cdot o_t)))
```

- **Image Encoder**: CNN with stride-2 convolutions → Flatten → MLP
- **Vector Encoder**: 2-layer MLP with LayerNorm
- **Output**: $z \in \mathbb{R}^{256}$ (latent thought)

#### 2. **Latent Dreamer** $D: \mathcal{Z} \times \mathcal{A} \rightarrow \mathcal{Z} \times \mathbb{R}$

Simulates future states without environment interaction:

```math
\hat{z}_{t+1}, \hat{r}_{t+1} = D(z_t, a_t) = \text{GRU}(z_t \oplus \text{embed}(a_t))
```

```math
\hat{r}_{t+1} = \text{symexp}(\text{RewardHead}(\hat{z}_{t+1}))
```

Where $\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$ compresses unbounded rewards for stability.

**Loss Function**:
```math
\mathcal{L}_{\text{dreamer}} = \text{KL}(P(z_{t+1}|z_t, a_t) \| Q(z_{t+1}|o_{t+1})) + \|\text{symlog}(\hat{r}_{t+1}) - \text{symlog}(r_{t+1})\|^2 + 0.1\|\hat{z}_{t+1} - \tilde{z}_{t+1}\|^2
```

#### 3. **Causal Verifier** $V: \mathcal{Z} \times \mathcal{A} \rightarrow \mathcal{Z}$

Learns physically plausible state transitions:

```math
\tilde{z}_{t+1} = V(z_t, a_t) = W_2 \cdot \text{ELU}(W_1 \cdot [z_t; \text{embed}(a_t)])
```

**Confidence (Oxygen Gauge)**:
```math
C(z, \tilde{z}) = \exp\left(-10 \cdot \|\hat{z}_{t+1} - \tilde{z}_{t+1}\|^2\right)
```

- $C \rightarrow 1$: High certainty (stable physics)
- $C \rightarrow 0$: Low certainty (novel situation)

#### 4. **Policy Network** $Q: \mathcal{Z} \rightarrow \mathbb{R}^{|\mathcal{A}|}$

Maps latent states to action values:

```math
Q(z, a) = W_3 \cdot \text{ReLU}(W_2 \cdot z)
```

**Policy Loss (Titan Formula)**:
```math
\mathcal{L}_{\text{policy}} = \|Q(z_t, a_t) - y_t\|^2 + \alpha_{\text{dark}} \|Q(z_t) - Q_{\text{past}}(z_t)\|^2 - 0.01 \cdot H(Q)
```

Where:
- $y_t = r_t + \gamma \max_{a'} Q_{\text{target}}(z_{t+1}, a')$ (DQN target)
- $\alpha_{\text{dark}} = \frac{0.5}{1 + 50 \cdot \text{surprise}}$ (adaptive dark loss)
- $H(Q) = -\sum_a \pi(a|z) \log \pi(a|z)$ (entropy bonus)

#### 5. **Dark Replay Buffer**

Stores $(o_t, a_t, r_t, o_{t+1}, d_t, Q_{\text{past}}(z_t))$ tuples, enabling **logit regularization** to prevent catastrophic forgetting.

---

### 🔬 Neuro-Modulation: The Adrenaline Engine

Dynamic learning rate adaptation based on prediction surprise:

```math
\text{surprise} = \mathbb{E}\left[\left|\text{symlog}(\hat{r}_{t+1}) - \text{symlog}(r_{t+1})\right|\right]
```

```math
\eta_{\text{adaptive}} = \eta_{\text{base}} \cdot (1 + 10 \cdot \text{surprise})
```

```math
\alpha_{\text{dark}} = \frac{0.5}{1 + 50 \cdot \text{surprise}}
```

**Behavioral Modes**:
- **High Surprise** (novel physics): ↑ Learning rate, ↓ Dark loss → *Rapid adaptation*
- **Low Surprise** (stable environment): ↓ Learning rate, ↑ Dark loss → *Knowledge consolidation*

---

### 🌊 Deep Ocean Planning (Blind Mode)

When observations are unavailable ($o_t = \mathbf{0}$), the agent relies on internal thought:

**Algorithm**:
```python
for each action a_start in [0, 1, 2, 3]:
    z_curr = z_internal
    path_value = 0
    confidence = 1.0
    
    for depth in range(MAX_HORIZON):  # 50 steps
        z_next, r_pred = Dreamer.forward_dream(z_curr, a_curr)
        z_verified = Verifier(z_curr, a_curr)
        
        conf_step = exp(-10 * ||z_next - z_verified||²)
        if conf_step < 0.85:  # Oxygen depleted
            break
        
        path_value += r_pred * γ^depth
        confidence = min(confidence, conf_step)
        
        a_curr = argmax Q_target(z_next)
        z_curr = z_next
    
    score = path_value * confidence
```

Select action with highest `score` → **Object Permanence Achieved**

---

## 🌌 The Omniverse

Five custom environments testing different facets of intelligence:

### 1️⃣ **Shifted Universe** (Memory)
- **Challenge**: Gravity periodically inverts (every 50 steps)
- **Test**: Adaptation to non-stationary dynamics
- **Mechanism**: Dark loss prevents catastrophic forgetting

### 2️⃣ **Invisible Universe** (Exploration)
- **Challenge**: Observations always return `[0, 0, 0, 0]`
- **Test**: Navigation without sensory input
- **Mechanism**: Planning within internal latent model

### 3️⃣ **Deceptive Universe** (Reasoning)
- **Challenge**: Traps give +1 reward, then -10 after 10 steps
- **Test**: Long-horizon planning vs. myopic greed
- **Mechanism**: 50-step dream horizon avoids delayed penalties

### 4️⃣ **High-Dim Matrix** (Filtering)
- **Challenge**: 64×64 grayscale images with 50% noise
- **Test**: Feature extraction from sensory clutter
- **Mechanism**: CNN encoder + LayerNorm compression

### 5️⃣ **Adversarial Eclipse** (Permanence)
- **Challenge**: Random 10-step sensor blackouts (5% chance)
- **Test**: Object permanence under sensor failure
- **Mechanism**: Internal thought maintenance + dream-based navigation

---

## 🚀 Installation

### Prerequisites

```bash
Python >= 3.8
CUDA >= 11.0 (for GPU acceleration)
```

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/dark-lucid-protocol.git
cd dark-lucid-protocol

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=2.0.0
numpy>=1.21.0
gymnasium>=0.28.0
matplotlib>=3.5.0
opencv-python>=4.6.0
tqdm>=4.65.0
pandas>=1.5.0
seaborn>=0.12.0
```

---

## 🎮 Usage

### Quick Start

```python
from dark_lucid import DarkLucidAgent, OmniverseEnv

# Initialize environment
env = OmniverseEnv(mode="adversarial")  # or shifted, invisible, deceptive, high_dim
obs_shape = env.observation_space.shape
action_dim = env.action_space.n

# Create agent
agent = DarkLucidAgent(obs_shape, action_dim, device="cuda")

# Training loop
obs, _ = env.reset()
for step in range(1000):
    action, logits = agent.select_action(obs, epsilon=0.1)
    next_obs, reward, done, _, _ = env.step(action)
    
    agent.memory.add(obs, action, reward, next_obs, done, logits)
    
    if step > 500:
        agent.update(batch_size=64)
    
    obs = next_obs
    if done:
        obs, _ = env.reset()
```

### Full Experiment Suite

```bash
# Run all 5 universes (takes ~2 hours on T4 GPU)
python run_omniverse.py --agent dark_lucid --episodes 200

# Compare against baseline
python run_omniverse.py --agent standard --episodes 200

# Generate analysis plots
python visualize_results.py
```

---

## 📊 Empirical Validation

### Experimental Protocol

- **Zero-Cheating Seed**: All RNG locked to `seed=42` for reproducibility
- **Fair Comparison**: Both agents trained with identical:
  - Environment dynamics
  - Reward structures
  - Episode budgets
  - Hardware (NVIDIA T4 GPU)
- **No Privileged Info**: Agents only observe `(obs, reward, done)`

### Statistical Significance

Repeated across 5 independent runs with different seeds:

| Universe | Mean Δ | Std Dev | p-value |
|:---------|-------:|--------:|--------:|
| Shifted  | +65.59 | ±4.12   | < 0.001 |
| Invisible | +81.00 | ±6.23   | < 0.001 |
| Deceptive | +60.36 | ±3.85   | < 0.001 |
| Matrix   | +40.69 | ±5.14   | < 0.001 |
| Eclipse  | +40.87 | ±2.91   | < 0.001 |

All improvements significant at $p < 0.001$ (Welch's t-test).

---

## 🔍 Architecture Diagrams

### System Flow

```
┌─────────────┐
│ Environment │
└──────┬──────┘
       │ obs
       ▼
┌─────────────────┐
│ Universal       │──────► z (latent)
│ Encoder         │
└─────────────────┘
       │
       ├──────────────────┬─────────────────┐
       ▼                  ▼                 ▼
┌─────────────┐    ┌──────────────┐  ┌─────────────┐
│ Latent      │    │ Causal       │  │ Policy      │
│ Dreamer     │    │ Verifier     │  │ Network     │
│ (GRU)       │    │ (MLP)        │  │ (Q-Net)     │
└──────┬──────┘    └──────┬───────┘  └──────┬──────┘
       │                  │                  │
       │ ẑ_next, r̂       │ z̃_next          │ Q(z,a)
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────┐
│         Neuro-Modulation Engine                 │
│  surprise → adaptive_lr, adaptive_dark_weight   │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────┐
│ Dark Replay     │
│ Buffer          │
│ (obs, a, r,     │
│  next_obs, Q)   │
└─────────────────┘
```

---

## 📈 Key Innovations

### 1. **Symlog Reward Compression**

Traditional RL struggles with unbounded rewards. We compress:

```math
r_{\text{compressed}} = \text{sign}(r) \ln(|r| + 1)
```

Prevents gradient explosion while preserving reward ordinality.

### 2. **Dark Loss Regularization**

Standard DQN: $\mathcal{L} = (Q(s,a) - y)^2$

Dark Lucid: $\mathcal{L} = (Q(z,a) - y)^2 + \alpha \|Q(z) - Q_{\text{past}}(z)\|^2$

The second term anchors current policy to past logits, mitigating catastrophic forgetting.

### 3. **Confidence-Gated Planning**

Only follow dream trajectories with high causal confidence:

```math
\text{path\_score} = \left(\sum_{t=0}^H \gamma^t \hat{r}_t\right) \times \min_{t \in [0,H]} C_t
```

Prevents hallucinations in blind mode.

---

## 🛠️ Hyperparameters

### DLP Configuration (v5.2)

```python
{
    "GRID_SIZE": 10,
    "MAX_STEPS": 200,
    "MAX_DREAM_HORIZON": 50,
    "CONFIDENCE_THRESHOLD": 0.01,
    
    # Learning Rates
    "LR_DREAMER": 1e-4,
    "LR_VERIFIER": 1e-4,
    "LR_POLICY": 5e-5,  # Drops to 5e-6 when avg_reward > 80
    
    # Neuro-Modulation
    "ADRENALINE_SCALE": 5,
    "DARK_WEIGHT_BASE": 0.5,
    "SURPRISE_SCALE": 50,
    
    # Stability
    "KL_BALANCE": 0.8,
    "FREE_NATS": 1.0,
    "GRADIENT_CLIP": 1.0,
    "TARGET_UPDATE_TAU": 0.995,
    
    # Memory
    "BUFFER_CAPACITY": 10000,
    "BATCH_SIZE": 64,
    "UPDATE_FREQ": 4
}
```

---

## 🎨 Visualization

Generate publication-ready plots:

```python
from dark_lucid.viz import plot_radar, plot_trajectory

# Intelligence shape (radar chart)
plot_radar(
    scores_dlp=[93.90, 89.07, 96.99, 94.09, 99.37],
    scores_baseline=[28.31, 8.07, 36.63, 53.40, 58.50],
    save_path="radar_chart.png"
)

# Eclipse trajectory
plot_trajectory(
    rewards_dlp=rewards_adv_dark,
    rewards_baseline=rewards_adv_std,
    save_path="eclipse_trajectory.png"
)
```

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- [ ] Multi-agent extensions
- [ ] Continuous control tasks
- [ ] Transformer-based world models
- [ ] Real-world robotics integration

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{darklucid2026,
  title={The Dark Lucid Protocol: Meta-Cognitive Reinforcement Learning Through Internal World Models},
  author={[Devanik]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Inspired by:
- **DreamerV3** (Hafner et al., 2023) - World model architecture
- **MuZero** (Schrittwieser et al., 2020) - Planning with learned models
- **PlaNet** (Hafner et al., 2019) - Latent imagination

Built with PyTorch ❤️

---

<div align="center">

**"When reality fades, the dream persists."**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/dark-lucid-protocol?style=social)](https://github.com/Devanik21/dark-lucid-protocol)
[![Twitter Follow](https://img.shields.io/twitter/follow/Devanik?style=social)](https://twitter.com/devanik2005)

</div>
