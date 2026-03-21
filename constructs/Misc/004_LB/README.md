# 🏛️ LIFE BEYOND: The Museum of Universal Life

**An Interactive Laboratory for Simulating Exotic Biologies and Emergent Ecosystems**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research_Active-success?style=for-the-badge)

---

### 📜 Abstract

`OmNIvErZe.py` is a computational framework designed to simulate the emergence of life in arbitrary physical and chemical environments. Unlike traditional sandbox simulations which focus on user-defined creation, this system operates as a **Museum of Universal Life**. The user acts as a Curator, defining the fundamental constants of a "gallery"—such as gravitational constraints, atmospheric density, and available chemical substrates.

The system utilizes a **Genetic Regulatory Network (GRN)** to govern the developmental biology of organisms. Lifeforms initiate as single cellular automata and evolve complex morphologies and behaviors driven by environmental selection pressures. The simulation supports substrate-independent life, allowing for the emergence of biology based on Carbon, Silicon, Plasma, Quantum states, and Void matter.

---

### 🧠 Theoretical Framework

#### 1. Genetic Regulatory Networks (GRN) as Generative Engines
The core simulation logic moves beyond simple genotype-phenotype maps. Instead, it employs a dynamic GRN model.
*   **State Vector**: Each cell maintains a state vector representing the concentration of internal proteins/messengers.
*   **Differential Dynamics**: Cell states evolve according to interaction matrices derived from the genetic code.
*   **Phenotypic Expression**: Morphological traits (e.g., structural integrity, motility organs) are the result of threshold-based activations within the network, rather than hard-coded features.

#### 2. The Chemical Base Registry
The simulation defines life not by specific atomic elements, but by **property archetypes**. The `CHEMICAL_BASES_REGISTRY` maps distinct chemical bases to physical parameters:

$$
P_{chem} = \{ M, S, E, C, \dots \}
$$

Where:
*   $M \in [m_{min}, m_{max}]$: Mass range probability density.
*   $S$: Structural integrity multiplier (affecting armor/fragility).
*   $E$: Energy storage coefficient (metabolic efficiency).
*   $C$: Conductance bias (neural/signal processing speed).

This abstraction allows for the simulation of **Exotic Biologies**, such as:
*   **Aether-based life**: High conductance, near-zero mass, computationally dense.
*   **Void-based life**: Negative thermodynamic biases, entropy-reversal mechanisms.
*   **Quantum substrates**: Non-deterministic state collapses driving behavioral outcomes.

#### 3. Meta-Innovation and Sensor Evolution
The system implements a recursive meta-learning layer. Organisms are not limited to pre-defined sensors. The simulation monitors the informational entropy of the environment. If an environmental gradient provides actionable information, the evolutionary algorithm may splice new sensing nodes into the GRN (e.g., `sense_neighbor_complexity`), expanding the organism's observable universe.

---

### ⚙️ Architectural Overview

The codebase is structured into three primary pillars:

#### I. The Curator's Console
A high-dimensional parameter space controller (approx. 4,200 lines of logic) allowing the configuration of:
*   **Physics Engine**: Gravity, friction, fluid dynamics coefficients.
*   **Energy Gradients**: Light intensity, thermal vents, mineral deposits.
*   **Selection Pressure**: Predation rates, resource scarcity, environmental decay.

#### II. The Exhibit Hall Manager
Utilizes `TinyDB` for persistent local storage of simulation states. This allows for the serialization of entire universes, preserving the genomic history of successful evolutionary branches.

#### III. The Biochemistry Engine
A probabilistic engine that interprets the `CHEMICAL_BASES_REGISTRY`.
*   **Mutation Operators**: Algorithms that perturb the HSV color space, mass, and bias values to create new hybrid substrates (e.g., `Psionic-Carbon-Shell`).
*   **Fitness Calculation**: A multi-objective function evaluating survival, reproduction, and energy efficiency.

---

### 🛠️ Technical Implementation

**Dependencies:**
The system relies on high-performance numerical and visualization libraries to handle real-time evolution rendering.

```python
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.special import softmax
import networkx as nx
```

**Key Data Structures:**

*   **Organism State**: Represented as a dynamic graph $G = (V, E)$ where nodes $V$ are cellular units and edges $E$ represent structural or signal传导 connections.
*   **Chemical Registry**: Implemented as a dictionary of dictionaries, mapping keys like `'Carbon'` or `'Quantum'` to parameter ranges.

```python
# Example: Parameter Definition for a Silicon Base
'Silicon': {
    'name': 'Silicon',
    'mass_range': (1.0, 2.5),
    'structural_mult': (1.5, 3.0), # Higher structural integrity than Carbon
    'compute_bias': 0.3,           # Moderate computational potential
    'chemosynthesis_bias': 0.4     # Tendency for mineral-based metabolism
}
```

---

### 📊 Visualization Pipeline

The simulation uses `Plotly` and `Matplotlib` to render the state of the exhibit.
1.  **Topological Mapping**: Visualizing the neural/structural graph of complex organisms.
2.  **Population Dynamics**: Real-time plotting of species survival rates and genetic diversity (using entropy measures).
3.  **Environment Heatmaps**: Rendering energy gradients and resource distribution across the 2D/3D grid.

---

### 🚀 Installation & Execution

To run the simulation locally:

```bash
git clone https://github.com/Devanik21/LIFE_BEYOND.git
cd LIFE_BEYOND
pip install -r requirements.txt
streamlit run OmNIvErZe.py
```

Upon launch, the dashboard presents the **Curator's Console**. Select a Chemical Base or generate a random seed to initiate a new Exhibit.

---

### 👤 Researcher Profile

This project aligns with broader research into **Scalable Neuro-Symbolic Architectures** and **Autonomous Agent Systems**.

**DΞVΛΠIK**
*B.Tech Electronics & Communication Engineering '26 | NIT Agartala*

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-181717?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Devanik-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)
[![Twitter](https://img.shields.io/badge/Twitter-@devanik2005-1DA1F2?style=flat-square&logo=twitter)](https://x.com/devanik2005)

**Research Focus:**
*   Investigating internal self-model formation in high-dimensional state spaces.
*   Application of Information Geometry to evolutionary fitness landscapes.
*   Topological Data Analysis (TDA) in studying morphological convergence in simulated biologies.

---
*"The fitness of an organism is its ability to survive and thrive within the harsh physics of its simulated environment."*
