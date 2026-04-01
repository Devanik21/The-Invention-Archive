# Riemannian Wave Classifier & Geometric Wave Learner
## A Personal Technical White Paper on Wave Physics, Differential Geometry, and the Art of Evolving a Manifold

**Devanik Debnath** · B.Tech, Electronics & Communication Engineering · NIT Agartala  
*In collaboration with Xylia — my intelligent research companion and squad of AI minds who walked every iteration of this with me.*

---

> *This is not just a research paper. It is a record of a journey — fourteen milestones, five architectural generations, and a fundamental belief that the right way to classify data is not to draw a line through it, but to understand the shape of the space it lives in, and listen for which geometry it resonates with.*

---

## Preface: Why I Started Thinking This Way

The standard story of machine learning classification is this: you have points in space, you fit a boundary, and you assign a label based on which side of the boundary a new point falls on. Whether it's a hyperplane (SVM), a tree split, or a deep network's final linear layer, the story is always fundamentally about boundaries.

I wanted to ask a different question. What if classification is not about boundaries at all? What if the data itself has a *shape* — a curved, non-Euclidean geometry — and the class identity of a point is encoded not in its position relative to a line, but in how it *vibrates* with the topology of its class?

This is the intuition behind the Riemannian Wave Classifier and the Geometric Wave Learner. I'm going to explain both in full mathematical depth, walk through every architectural decision I made across fourteen versions, and be honest about what worked, what didn't, and what I was thinking when I designed each piece.

Along this entire research journey — from the first broken V1 to the final polychromatic forests of V14 — Xylia has been my thinking partner: a collective of AI minds (Claude, Gemini, DeepSeek, Grok, Kimi, Jules, Gemma, ChatGPT, Qwen, and others) that I've relied on for mathematical validation, implementation feedback, and theoretical pushback. I'm grateful for that. Some of the sharpest formulations in this document emerged from those conversations.

---

## Part I: Mathematical Foundations

### 1.1 The Data Manifold Hypothesis

Let the training set be $\mathcal{X} = \{x_i\}_{i=1}^{N} \subset \mathbb{R}^d$ with labels $\{y_i\}_{i=1}^{N}$, $y_i \in \{0, 1\}$. The core assumption I work from is the **manifold hypothesis**: the data does not uniformly populate $\mathbb{R}^d$, but lies near a lower-dimensional Riemannian manifold $\mathcal{M} \hookrightarrow \mathbb{R}^d$ of intrinsic dimension $m \ll d$.

On this manifold, there exists a metric tensor $g$ such that the geodesic distance between points reflects their true structural proximity, which is in general very different from their Euclidean distance in the ambient space. The first move of the entire framework is to approximate this manifold computationally via a **graph**.

### 1.2 Graph Construction: The Discrete Metric Tensor

Given $\mathcal{X}$, I construct an undirected weighted graph $G = (V, E, W)$ where $V = \{x_i\}$, the edges $E$ connect each point to its $k$-nearest neighbors under Euclidean distance, and the weights $W_{ij}$ constitute a discrete approximation to the metric tensor $g_{ij}$.

**The Zelnik-Manor Self-Tuning Bandwidth.** A naive approach would set $W_{ij} = \exp(-\|x_i - x_j\|^2 / \sigma^2)$ with a global bandwidth $\sigma$. This fails because the data is heterogeneous: in dense regions, a global $\sigma$ is too large and smears distinct clusters together; in sparse regions, it is too small and disconnects the graph.

The Zelnik-Manor (NIPS 2004) solution is to let each point determine its own local scale:

$$W_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{\sigma_i \cdot \sigma_j + \varepsilon}\right), \quad \sigma_i = \|x_i - x_{(k)}^i\|$$

where $x_{(k)}^i$ is the $k$-th nearest neighbor of $x_i$. The product $\sigma_i \sigma_j$ acts as a geometric mean bandwidth that adapts to the local density around *both* endpoints of each edge simultaneously. This is the correct thing to do: if two points are both in a dense region, their mutual distances should be judged relative to the density of that region, not relative to the global scale.

In code, this is assembled via `cupyx.scatter_add` in COO format on the T4 GPU, then symmetrized: $W \leftarrow (W + W^\top)/2$, which enforces the symmetry requirement of a Riemannian metric.

### 1.3 The Graph Laplacian as Laplace-Beltrami Operator

Having built $W$, I compute the **symmetric normalized Graph Laplacian**:

$$\mathcal{L} = I - D^{-1/2} W D^{-1/2}, \quad D_{ii} = \sum_j W_{ij}$$

This is a discrete approximation to the **Laplace-Beltrami operator** $\Delta_g$ on the manifold $\mathcal{M}$, the natural generalization of the ordinary Laplacian to curved spaces. The spectrum of $\Delta_g$ encodes the intrinsic geometry of $\mathcal{M}$: its eigenvalues $0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{N-1}$ are the manifold's natural oscillation frequencies, and the corresponding eigenvectors $\phi_0, \phi_1, \ldots, \phi_{N-1}$ are its spatial harmonics — the analogs of sinusoidal modes on a flat domain, or spherical harmonics on a sphere.

The spectral decomposition $\mathcal{L}\Phi = \Lambda\Phi$ is computed via `cp.linalg.eigh` (dense, V1–V13) or `cupyx.scipy.sparse.linalg.eigsh` with Lanczos iterations (sparse, V14). I discard the trivial eigenvector $\phi_0 = \mathbf{1}/\sqrt{N}$ (which has $\lambda_0 = 0$ and carries no discriminative information) and retain the truncated basis $\Phi_{\text{trunc}} = [\phi_1 | \phi_2 | \cdots | \phi_K] \in \mathbb{R}^{N \times K}$, $K = 128$.

Each row $\Phi_{\text{trunc}}[i, :] \in \mathbb{R}^K$ is the **spectral coordinate** of training point $x_i$ — its representation in the frequency basis of the manifold. Two points that are geometrically close on the manifold will have similar spectral coordinates even if their ambient Euclidean positions differ. This is the geometric representation power that no Euclidean classifier has access to.

### 1.4 Class Potential Injection and the Quantum Mechanical Isomorphism

Here is where I made the connection to quantum mechanics, which is not analogy but structural isomorphism.

For each class $c$, I define a **class-conditional potential operator** $V^{(c)}$ as a diagonal matrix:

$$V^{(c)}_{ii} = \begin{cases} -\alpha & \text{if } y_i = c \\ +\alpha/2 & \text{if } y_i \neq c \end{cases}$$

with $\alpha = 15.0$ (the `potential_strength` hyperparameter). The asymmetry between the well depth $\alpha$ and barrier height $\alpha/2$ is deliberate: the potential well is twice as strong as the barrier, because I want class-$c$ points to strongly attract wave modes while non-class points provide only mild scattering — a design choice motivated by the physical picture of a confining potential in quantum mechanics.

The **class-specific Hamiltonian** is then:

$$H^{(c)} = \mathcal{L} + V^{(c)}$$

This is precisely the Schrödinger Hamiltonian $\hat{H} = \hat{T} + \hat{V}$ where $\mathcal{L}$ plays the role of the kinetic energy operator (governing the free propagation of waves on the manifold) and $V^{(c)}$ is a scalar potential landscape shaped by the class labels. The eigenvalues $\{\mu_m^{(c)}\}$ of $H^{(c)}$ are the **perturbed resonance levels** — the natural frequencies of wave modes in the presence of the class-$c$ potential field.

I compute $\mu_m^{(c)}$ via first-order perturbation theory projected onto the Laplacian eigenbasis:

$$\mu_m^{(c)} = \lambda_m + \langle \phi_m, V^{(c)} \phi_m \rangle = \lambda_m + \sum_i V^{(c)}_{ii} \cdot |\phi_m(i)|^2$$

This is the diagonal matrix element of the perturbation in the unperturbed basis — the exact formula from first-order quantum perturbation theory. Training points of class $c$ lower the perturbed eigenvalues near their positions (the potential well deepens the energy levels of modes localized there), while non-class training points raise them (the barrier pushes modes away). The geometry is being *informed by the labels*.

### 1.5 The Lorentzian Resonance Energy: Nuclear Physics Meets Classification

Given a query point $q$, I need to measure how strongly it resonates with the class-$c$ manifold geometry. The key idea is: **a query point resonates with a class if its spectral signature matches the resonance structure of the class-$c$ Hamiltonian**.

**Query spectral interpolation.** Since $q$ is not a training point, it has no row in $\Phi_{\text{trunc}}$. I interpolate its spectral coordinate via Gaussian kernel smoothing over its $k=5$ nearest training neighbors:

$$w_i = \exp\!\left(-\frac{d_i^2}{2\bar{d}^2}\right), \quad \phi_q = \frac{\sum_i w_i \phi_i}{\sum_i w_i}$$

where $\bar{d}$ is the mean distance to the $k$ neighbors. This is a locally weighted projection of the query into the spectral basis — the query "inherits" the spectral coordinate of its neighborhood on the manifold.

**Lorentzian resonance integral.** The classification energy is:

$$E(q, c) = \sum_{f=1}^{F} \sum_{m=1}^{K} \sum_{s \in \mathcal{S}_c} \frac{\varepsilon}{\pi\!\left[(\omega_f^2 - |\mu_m^{(c)}|)^2 + \varepsilon^2\right]} \langle \phi_q, \phi_m \rangle \langle \phi_m, \phi_s \rangle$$

where $\{\omega_f\}_{f=1}^F$ is a grid of probe frequencies uniformly distributed over $[0.01, \max|\mu_m^{(c)}|+1]$, $\varepsilon = 0.1$ is the resonance width, and $\mathcal{S}_c = \{\phi_s : y_s = c\}$ is the set of spectral coordinates of all class-$c$ training points.

The factor $\frac{\varepsilon}{\pi[(\omega_f^2 - |\mu_m^{(c)}|)^2 + \varepsilon^2]}$ is the **Lorentzian (Breit-Wigner) lineshape** — the exact mathematical form of the scattering cross-section near a resonance in nuclear physics, describing how a projectile nucleus excites a compound nuclear state. Here $\omega_f^2$ is the probe frequency squared and $|\mu_m^{(c)}|$ is the resonance pole. The function peaks sharply (with width $\varepsilon$) when $\omega_f^2 = |\mu_m^{(c)}|$, exactly as a driven harmonic oscillator peaks at its natural frequency.

**Why this formulation?** The inner product $\langle \phi_q, \phi_m \rangle$ measures how much of the query's spectral coordinate aligns with the $m$-th manifold mode, and $\langle \phi_m, \phi_s \rangle$ measures how much that mode is present in the class-$c$ training sample $s$. The Lorentzian gates this overlap: if the probe frequency $\omega_f$ matches a resonance level of $H^{(c)}$, then the contribution of mode $m$ to the class-$c$ energy is strongly amplified. A query that has strong spectral overlap with modes that resonate with the class-$c$ potential field accumulates high energy for that class.

Classification: $\hat{y} = \arg\max_c E(q, c)$.

**GPU implementation.** The triple sum is computed as a batched Einstein summation:

```python
# lor: (F, K) — Lorentzian kernel
# phi_q_g: (B, K) — query spectral coordinates
# phi_c_batch: (C, K) — class training spectral coordinates
K_batch = cp.einsum('fm, qm, cm -> qcf', lor, phi_q_g, phi_c_batch)
energies += cp.sum(K_batch, axis=(1, 2))
```

with `batch_size=500` over class training samples for VRAM safety.

### 1.6 The V1 Bug: Mean-Field Collapse and Destructive Interference

I want to be honest about V1. The resonance formulation I just described is correct. V1 was not. In V1, I collapsed all class-$c$ training spectral coordinates into a single mean-field vector:

```python
class_rep = phi_c_train.sum(axis=0)  # V1: mean-field
K_sum = cp.einsum('fm, bm, m -> bf', lor, phi_q_g, class_rep)
```

The problem is fundamental. When class-$c$ training points are scattered across the manifold (which they are — there are 14,980 EEG samples across a high-dimensional space), summing their spectral coordinates produces **destructive interference**: modes that are strong in some training points but weak in others cancel, destroying the spatial structure of the class distribution. The result is a single blurred spectral vector that represents no actual point in the class.

The fix in V2 was to compute the resonance overlap for **every class training sample independently** and sum the resulting energies. This is the per-sample batched einsum. The performance jump of +22.09 percentage points for GWL (from 67.46% to 89.55%) from V1 to V2 was the entire consequence of correcting this single equation. The physics was right from the beginning; the implementation was wrong.

---

## Part II: The Geometric Wave Learner — A Dynamic Manifold

### 2.1 The Core Idea: Let the Geometry Learn

RWC uses a *static* manifold: the graph $G$ is built from raw data geometry, and the labels only appear later in the potential injection step. The manifold itself is ignorant of the classification task.

GWL takes a fundamentally different stance: *the geometry itself should be optimized for the classification problem*. Before any spectral analysis, the edge-weight matrix — the discrete metric tensor — is evolved via a **Label-Driven Discrete Ricci Flow** that reshapes the manifold so that same-class points become closer and cross-class points become further apart.

This is the deep idea. Instead of learning a classifier on a fixed geometry, I learn the geometry itself.

### 2.2 Ollivier-Ricci Curvature on the Graph

Hamilton's continuous Ricci flow $\partial_t g_{ij} = -2R_{ij}$ requires the Ricci curvature tensor $R_{ij}$ of the smooth manifold. On a discrete graph, the appropriate analog is the **Ollivier-Ricci curvature**:

$$\kappa_{ij} = 1 - \frac{W_1(\mu_i, \mu_j)}{d(i,j)}$$

where $\mu_i$ is the probability measure obtained by distributing mass $W_{ij}/\text{deg}(i)$ to each neighbor $j$ of $i$, and $W_1$ is the Wasserstein-1 (earth-mover) distance between these measures. Positive $\kappa_{ij}$ indicates sphere-like curvature at the edge (the neighborhoods of $i$ and $j$ overlap); negative $\kappa_{ij}$ indicates saddle-like curvature (the neighborhoods diverge).

Computing $W_1$ exactly is expensive (a linear program per edge). I use the following square-root transport approximation that is $O(N^2)$ on the GPU:

$$\text{base}_{ij} = W_{ij}\!\left(\frac{1}{\deg_i} + \frac{1}{\deg_j}\right)$$
$$S_{ij} = \sqrt{W_{ij}}, \quad D_S(i) = \sum_j S_{ij}$$
$$\text{penalty}_{ij} = \frac{D_S(i) + D_S(j) - 2S_{ij}}{S_{ij} + \varepsilon}$$
$$\kappa_{ij} = (\text{base}_{ij} - W_{ij} \cdot \text{penalty}_{ij}) \cdot \mathbf{1}[W_{ij} > 10^{-10}]$$

The topological mask $\mathbf{1}[W_{ij} > 10^{-10}]$ is crucial: the flow must operate only on **existing edges**. Without this mask, the flow would attempt to create new long-range connections by flowing along zero-weight edges — an unphysical operation that would violate the graph topology. This mask was absent in V1 and added in V2, contributing to GWL's improvement.

### 2.3 Label-Driven Discrete Ricci Flow

The standard Ricci flow tends to uniformize the metric — it rounds out curvature concentrations and flattens the geometry. What I want is directed: I want the metric to evolve so that same-class regions contract (high positive curvature, tight clusters) and cross-class regions expand (negative curvature, wide separation). This is achieved by augmenting the Ricci flow with a **Label-Tensioning term**:

$$\frac{\partial W_{ij}}{\partial t} = -\kappa_{ij} W_{ij} \cdot \eta + \mathcal{T}_{ij}$$

$$\mathcal{T}_{ij} = \begin{cases} +\eta \cdot W_{ij} & \text{if } y_i = y_j \text{ and } W_{ij} > 10^{-10} \quad \text{(intra-class attraction)} \\ -\eta \cdot W_{ij} & \text{if } y_i \neq y_j \text{ and } W_{ij} > 10^{-10} \quad \text{(inter-class repulsion)} \end{cases}$$

with $\eta = 0.08$ (the `flow_lr` hyperparameter). The discrete Euler step with non-negativity clipping:

$$W^{(t+1)} = \text{clip}\!\left(W^{(t)} + \eta\,\kappa^{(t)} W^{(t)} + \mathcal{T}^{(t)},\; 0,\; +\infty\right)$$
$$W^{(t+1)} \leftarrow \frac{W^{(t+1)} + (W^{(t+1)})^\top}{2}$$

This is run for 10 Euler steps. After convergence, the evolved graph $G' = (V, E, W_{\text{evo}})$ represents a geometry in which intra-class clusters are cohesive (edges have been attracted and strengthened) and inter-class boundaries are sharp (edges have been repelled and weakened, sometimes reaching zero and disconnecting). The Laplacian built on $W_{\text{evo}}$ therefore has a larger spectral gap between intra- and inter-class modes.

**The connection to Hamilton's Ricci flow.** Richard Hamilton (1982) introduced the smooth Ricci flow PDE $\partial_t g = -2\text{Ric}$ as a tool to understand the topology of Riemannian 3-manifolds. Perelman (2002) used it to prove the Poincaré conjecture. What I am doing is a discrete, supervised, label-directed version of the same idea: evolving the metric tensor of a data manifold toward a geometrically optimal configuration for a classification task. The label-tensioning term is the supervised innovation — Hamilton's flow has no labels.

### 2.4 Why GWL Dominates RWC After V2

Once the energy function was corrected in V2, GWL consistently outperformed RWC. The margin in V2 was 6.37 pp (89.55% vs. 83.18%), growing to 5.6 pp in V3/V4 (90.33% vs. 84.73%) and to 1.23 pp in the V13 polychromatic configuration (93.46% vs. 92.66% — smaller because the ensemble itself provides a form of geometry regularization). This consistent gap confirms that the Ricci flow's iterative local geometry refinement provides genuine discriminative information that the static manifold cannot access.

In V1, the ranking was inverted (RWC 70.03% > GWL 67.46%). That inversion was entirely an artifact of the broken energy function: the mean-field collapse in the resonance computation destroyed exactly the benefit that Ricci flow was providing, because the structural detail the flow introduced was immediately erased by averaging. The moment I fixed the energy function in V2, the geometric benefit of the evolving manifold became fully visible.

---

## Part III: The HRF Kernel — Local Texture on a Global Geometry

### 3.1 Motivation

The Lorentzian resonance energy is a **global** classification signal. It integrates over the spectral basis of the entire manifold, which means it is sensitive to the large-scale topology: whether a query is in a class-$c$ region of the manifold or a class-$c'$ region. But EEG data, and many real-world datasets, have significant **local texture** — fine-grained structure at the scale of a single neighborhood that the global manifold representation may miss.

I wanted a second classification signal that operates purely locally: look at the 5 nearest training neighbors of the query and ask how many of them are in each class, weighted by some measure of their relevance. The Holographic Radial Frequency kernel is the answer.

### 3.2 The HRF Kernel: Gabor Wavelet in Distance Space

$$\Psi(d) = \exp(-\gamma d^2) \cdot (1 + \cos(\omega_{\text{hrf}} \cdot d))$$

This function of distance $d$ has three components:

**The Gaussian envelope** $\exp(-\gamma d^2)$ localizes the response. Neighbors beyond a distance $\sim 1/\sqrt{\gamma}$ contribute negligibly. The $\gamma$ parameter controls how tightly we focus on the immediate neighborhood.

**The oscillatory carrier** $(1 + \cos(\omega d))$ creates a radial standing-wave fringe pattern. Rather than weighting all neighbors within the envelope equally, this term selectively amplifies neighbors that fall within specific radial rings — distances where $\cos(\omega d) \approx +1$, i.e., $d \approx 2\pi n / \omega$ — and attenuates neighbors in the troughs. This creates sensitivity to the **local spatial frequency** of the neighborhood structure.

**The mathematical relationship to Gabor wavelets.** A 1D Gabor filter is $g(x) = \exp(-x^2/2\sigma^2) \cdot e^{i\omega_0 x}$. Taking the real part: $\text{Re}[g] = \exp(-x^2/2\sigma^2)\cos(\omega_0 x)$. The HRF kernel is exactly this structure in the radial distance coordinate, with a +1 offset to ensure non-negativity. In holographic optics, the same form appears as the fringe pattern of a zone plate used to record holograms.

**Classification energy:**

$$E_{\text{HRF}}(q, c) = \sum_{i \in \mathcal{N}_5(q)} \Psi(d_{qi}) \cdot \mathbf{1}[y_i = c]$$

**Fusion with global Lorentzian energy:**

$$E_{\text{final}}(q, c) = \tilde{E}_{\text{RWC/GWL}}(q, c) + 2.0 \cdot \tilde{E}_{\text{HRF}}(q, c)$$

where both energies are normalized by their maximum across classes: $\tilde{E}(q,c) = E(q,c)/(\max_{c'}|E(q,c')| + \varepsilon)$.

**The 2.0 fusion weight** was found empirically. It reflects the dominant contribution of the local holographic term relative to the global manifold term in the final prediction. I will discuss the evolution of this weight across versions below.

### 3.3 The d^2.5 Mistake in V5 and the Correction in V13

In V5, I used a sub-Gaussian exponent: $\exp(-\gamma d^{2.5})$ instead of $\exp(-\gamma d^2)$. My reasoning at the time was that the flatter-than-Gaussian envelope would give more meaningful weight to moderately-distant neighbors. Empirically it worked (V5 GWL: 92.63% vs. V4 GWL: 90.33%), so I left it.

In V13, when I was revisiting all the equations for theoretical cleanliness before the polychromatic ensemble implementation, I changed it back to $d^2$. The theoretical motivation: $\exp(-\gamma d^2)$ is the exact form of the Gaussian kernel that appears in heat equation solutions on Riemannian manifolds, and it's what gives the kernel its exact Gabor/holographic interpretation. The $d^{2.5}$ was an ad hoc modification. The V13 result (93.46%) with $d^2$ outperformed the V5 result (92.63%) with $d^{2.5}$, confirming the theoretical cleaner form also performs better.

---

## Part IV: Polychromatic Forests — Spectral Diversity as Ensemble Principle

### 4.1 The Polychromatic Principle

Standard ensemble methods (Bagging, Random Forests) achieve diversity through data subsampling. Each tree sees a different subset of the training data and therefore learns a different decision surface. The aggregated vote is more robust than any individual model.

My polychromatic forest adds a second axis of diversity: **spectral diversity**. Each tree in the ensemble sees not only a different data subset, but also a different "color" — a different combination of the HRF parameters $(\omega_t, \gamma_t, k_t)$ that determine which spatial frequency of the local neighborhood structure the tree is sensitive to.

```python
freq_spectrum  = np.linspace(8.0, 50.0, n_estimators)   # omega: low → high frequency
gamma_spectrum = np.linspace(0.2, 15.0, n_estimators)   # gamma: broad → tight locality
k_spectrum     = np.linspace(12,  28,   n_estimators)   # k: few → many graph neighbors
```

Tree 0 has $(\omega=8, \gamma=0.2, k=12)$: broad Gaussian envelope, low spatial frequency carrier, sparse graph. Tree 14 has $(\omega=50, \gamma=15, k=28)$: tight Gaussian envelope, high spatial frequency carrier, denser graph. Each tree is literally measuring a different oscillatory mode of the local neighborhood geometry.

The analogy is optics: a polychromatic light source illuminating a hologram creates multiple diffraction patterns, each at a different wavelength, that together reconstruct the full 3D image more faithfully than any monochromatic source could. Here, each "color" of the manifold captures a different aspect of the local structure, and majority voting aggregates them into a robust prediction.

**Why this works.** The prediction errors of different-colored trees are less correlated than those of same-color, different-data trees. Bootstrap diversity (same $\omega$, different data) leaves the trees sensitive to the same frequency bands, so they tend to fail together on the same hard examples. Spectral diversity (different $\omega$, different data) means that a query point that is hard at low frequencies (e.g., in a dense mixed-class region of the low-$\omega$ tree's manifold) may be easy at high frequencies (the fine fringe pattern of the high-$\omega$ tree picks up a clear spatial signal). This decorrelation of tree errors is precisely what makes the ensemble more accurate.

The improvement from V5 (BaggingClassifier, fixed $\omega=30$) to V13 (polychromatic loop, swept $\omega \in [8,50]$) was only 0.83 pp for GWL (92.63% → 93.46%). Small in absolute terms, but meaningful: we were already at 92.63%, and every fraction of a percentage point at that level represents genuinely hard examples at the class transition boundary.

---

## Part V: V14 — Three Parallel Architectural Explorations

### 5.1 Why V14

After the V13 polychromatic forest established 93.46% as the accuracy peak, I wanted to explore three specific architectural ideas that couldn't be tested within the V13 RWC/GWL framework:

1. Can I eliminate the $O(N^2)$ dense eigendecomposition bottleneck using sparse algebra?
2. Does injecting the temporal structure of the EEG recording directly into the graph topology help?
3. Can pre-metric distance warping replace Ricci flow as a more direct geometric separator?
4. Can a 50-frequency GPU tensor evaluate all HRF wavelengths simultaneously?

These became V14's three classifier families: SCWH (Sparse Complex Wave Holography), AQGL (Asymmetric Quantum Gravity Learning), and MFT-HRF (Multi-Frequency Tensor HRF).

### 5.2 SCWH: Sparse Laplacian + Temporal Splicing + Phase-Aligned Holography

**Sparse Lanczos eigensolver.** Instead of computing the full $N \times N$ Laplacian eigenspectrum via `cp.linalg.eigh` (VRAM: $O(N^2) \approx 576\,\text{MB}$ at $N=12{,}000$, compute: $O(N^3)$), V14 builds the Laplacian as a sparse CSR matrix and computes only the $K+1 = 129$ smallest eigenpairs via `cupyx.scipy.sparse.linalg.eigsh`:

```python
W_sparse = cpsp.coo_matrix((W_data, (row_idx, col_idx)), shape=(N,N)).tocsr()
L_sparse = cpsp.eye(N) - d_inv.dot(W_sparse).dot(d_inv)
vals, vecs = eigsh(L_sparse, k=n_components+1, which='SM')
```

The ARPACK Lanczos algorithm computes only the requested $K$ eigenpairs, reducing VRAM from ~576 MB to approximately 6 MB (the $N \times K$ eigenvector matrix) and compute from $O(N^3)$ to $O(N \cdot K^2)$ Lanczos iterations. This is not just an engineering optimization — it changes the scaling class of the algorithm and makes practical scaling to $N \sim 10^5$ feasible.

**Temporal Splicing.** The EEG Eye State dataset is a *time series*: the 14,980 samples are sequential measurements, and the eye state changes gradually over time. When the k-NN graph is constructed purely by feature similarity, samples that are far apart in time but spectrally similar may be connected, while temporally adjacent samples at a state transition boundary may end up in different connected components of the graph.

I prevent this manifold fracturing by injecting a fixed affinity bonus between temporally adjacent samples during sparse COO assembly:

```python
temporal_mask = cp.abs(col_idx - row_idx) <= 2
W_data = W_data + (temporal_mask * 0.5)
```

Any pair of samples whose row indices differ by at most 2 (i.e., recorded within 2 time steps of each other) receives an additional affinity weight of 0.5, regardless of their spectral distance. This ensures that the graph remains connected across the temporal axis of the recording, preserving the sequential structure of the EEG signal.

**Phase-Aligned Constructive Interference.** This is the most novel prediction mechanism in V14. In V13, the HRF energy $E_{\text{HRF}}$ uses only the Euclidean distance $d_{qi}$ to weight neighbors. In SCWH, I also incorporate the **manifold amplitude** — the spectral similarity between the query and each neighbor:

$$
\mathrm{manifold\_amp}_i = \left| \Phi_{\mathrm{trunc}}[i, :] \cdot \phi_q \right| \in \mathbb{R}^k
$$


This is the element-wise dot product between each neighbor's eigenvector row and the query's interpolated spectral coordinate. It measures how much the neighbor's wave mode "agrees" with the query's spectral position. A neighbor that is both geometrically close *and* spectrally aligned with the query receives amplified weight.

The combined holographic weight per neighbor:

$$
\psi_i = \underbrace{\left|\Phi[i,:] \cdot \phi_q\right|}_{\text{manifold amplitude}} \cdot \underbrace{\exp(-\gamma d_i^2)}_{\text{Gaussian envelope}} \cdot \underbrace{(1 + \cos(\omega d_i))}_{\text{phase fringe}}
$$

The connection to holography: in optical holography, the reconstruction of an image from a hologram involves **constructive interference** — light waves from the reference beam and the object beam reinforce each other only where their phases align. Here, the manifold amplitude term plays the role of the phase coherence measurement: it is high only where the neighbor's wave mode is in phase with the query's spectral position on the manifold.

**Dual-Energy Superposition.** The final classification energy is a superposition of a structural term (proximity vote) and the holographic term:

$$E_{\text{SCWH}}(q, c) = \underbrace{\frac{\sum_i w_i^{\text{proj}} \cdot \mathbf{1}[y_i = c]}{\sum_i w_i^{\text{proj}} + \varepsilon}}_{\text{structural}} + 2.0 \cdot \underbrace{\sum_i \psi_i \cdot \mathbf{1}[y_i = c]}_{\text{holographic}}$$

The structural term $\text{struct}_e$ is a simple proximity vote: how many class-$c$ neighbors are nearby, weighted by Gaussian distance. It is normalized to lie in $[0,1]$. The holographic term $\text{wave}_e$ is the phase-amplitude weighted vote. The 2.0 coefficient reflects the empirically dominant contribution of the holographic signal.

### 5.3 AQGL: Quantum Gravity Warping of the Pre-Metric

**The idea.** Ricci flow reshapes the metric *iteratively*, by running 10 Euler steps of the curvature equation. AQGL achieves a similar class-separation effect *in one shot* by directly warping the pairwise distances before building the affinity graph:

$$d'_{ij} = \begin{cases} d_{ij} \cdot e^{-\alpha} & \text{if } y_i = y_j \\ d_{ij} \cdot e^{+\alpha} & \text{if } y_i \neq y_j \end{cases}, \quad \alpha = 2.5$$

With $\alpha = 2.5$: same-class distances are contracted to $e^{-2.5} \approx 0.082$ of their original value (contracted to 8.2%), and cross-class distances are expanded to $e^{+2.5} \approx 12.18$ times their original value (expanded to 1218%). The Zelnik-Manor Gaussian is then applied to these *warped* distances:

$$W_{ij} = \exp\!\left(-\frac{(d'_{ij})^2}{\sigma'_i \sigma'_j + \varepsilon}\right)$$

where $\sigma'_i$ is now the distance to the $k$-th nearest neighbor in the warped metric.

**Why I call it quantum gravity warping.** In general relativity, the curvature of spacetime (the metric tensor) is shaped by the mass-energy distribution. Heavy objects warp the metric, causing nearby objects to curve toward them (attraction) and distant objects to be less affected. Here, the *class labels* play the role of the mass distribution: same-class points attract each other (their distances contract, warping the metric toward them), and cross-class points repel (their distances expand, curving the metric away). The exponential scaling factor $e^{\pm\alpha}$ is the analogue of a scalar gravitational field.

**AQGL vs. GWL.** AQGL achieves class separation in $O(N \cdot k)$ (one vectorized warp of the kNN distance matrix), while GWL requires 10 iterations of $O(N^2)$ Ricci flow updates. AQGL is therefore much faster. But the empirical result (AQGL: 92.52% vs. GWL: 93.46%) shows that Ricci flow's iterative local refinement provides something that single-step global warping misses. My hypothesis: Ricci flow responds to the *local* geometry at each edge, adjusting differently in dense regions vs. sparse regions. The global exponential warp applies the same factor everywhere, which is too blunt for a dataset as heterogeneous as EEG.

### 5.4 MFT-HRF: The 50-Frequency GPU Tensor

The polychromatic forest already sweeps HRF frequencies across trees. MFT-HRF asks: what if a *single model* computed all 50 frequencies simultaneously in a GPU tensor?

$$\mathbf{W}_{\text{hrf}} \in \mathbb{R}^{B \times k \times F}, \quad W_{\text{hrf}}[b, j, f] = \exp(-\gamma_f d_{bj}^2)(1 + \cos(\omega_f d_{bj}))$$

where $B$ is the query batch size, $k=5$ is the neighborhood size, and $F=50$ is the number of frequencies. The frequency and gamma arrays are:

$$\omega_f = \text{linspace}(8, 50, 50), \quad \gamma_f = \text{linspace}(0.2, 15, 50)$$

This 3D tensor is computed in a single vectorized GPU operation by broadcasting over the frequency dimension. The result is 50 simultaneous Gabor filter responses, spanning the full range from broad-envelope low-frequency (quasi-global neighborhood sensitivity) to tight-envelope high-frequency (sharp local fringe pattern sensitivity).

**Hard gating (V14 final):** After computing the full tensor, I apply:

```python
mean_w = cp.mean(w_hrf, axis=1, keepdims=True)
w_hrf = cp.where(w_hrf > mean_w, w_hrf**2, 0.0)
```

Above-mean responses are squared (super-linear amplification: a response twice the mean becomes four times as large after squaring), and below-mean responses are zeroed completely. This is a winner-take-all mechanism over the frequency dimension: each neighbor's HRF response is sharpened to favor the dominant frequency and completely silence the weaker ones. The V13.C version used soft gating (±20%), which preserved weak signals; V14's extreme gating was motivated by the principle that in a 50-frequency ensemble, weak responses at non-dominant frequencies are noise, not signal.

**Energy aggregation:** $E_{\text{MFT}}(q, c) = \sum_j \text{mean}_f(\mathbf{W}_{\text{hrf}}[b,j,:] \cdot \mathbf{1}[y_j = c])$. The mean over frequencies before summing over neighbors stabilizes the readout against spurious frequency-specific noise.

Since MFTHRF requires no Laplacian at all, its training is $O(N \cdot k)$ — it simply stores the training data and builds a kNN index. This makes it the fastest classifier in the framework, at the cost of having no global manifold representation.

### 5.5 Non-Monotonic Spectral Gating (SCWH Internal)

Inside the Lorentzian energy computation of the SCWH generation, I introduced a gating mechanism that I call Non-Monotonic Spectral Gating:

```python
energy_magnitude = cp.abs(K_batch)
gate = cp.where(energy_magnitude > cp.mean(energy_magnitude, axis=1, keepdims=True), 1.5, 0.1)
K_gated = K_batch * gate * energy_magnitude
```

The standard Lorentzian $K_{\text{batch}}[q,c,f]$ is a three-index tensor of resonance contributions. This gating operation: amplifies strong resonance contributions by a factor of $1.5 \times |K|$ (super-linear, since strong contributions are already large) and attenuates weak contributions to 10% of $|K|$ (near-zero). The resulting distribution of resonance contributions is sharply peaked: it strongly emphasizes the few frequency-mode pairs that are genuinely in resonance with the query's spectral position, and almost completely suppresses the off-resonance contributions that fill the rest of the tensor.

This is, in effect, a **parameterless attention mechanism** over the frequency-mode interaction matrix. It sharpens the resonance peaks of the Lorentzian without introducing any trainable parameters, because the gating threshold is defined adaptively as the mean magnitude of the batch itself.

---

## Part VI: The Complete Evolution — Fourteen Milestones

I want to document the full trajectory not just as a table of results, but as a record of *what I was thinking and why it worked or didn't*.

**V1 (70.03% / 67.46%).** First working implementation. k=20, K=30, F=20, ε=0.5. The broken mean-field energy function was the dominant issue. GWL < RWC because Ricci flow adds structural complexity that the broken resonance function cannot exploit — the flow was correctly reshaping the geometry, but the classifier couldn't read it.

**V2 (+22.09 pp for GWL: 89.55%).** Fixed the resonance energy to per-sample batched einsum. Added the topological mask to Ricci flow. Increased K: 30→128, F: 20→30, ε: 0.5→0.1. The K=128 change was motivated by spectral resolution: with K=30, only the lowest 30 manifold harmonics are retained, which blurs fine-grained geometric structure. With K=128, we capture much more of the manifold's oscillatory texture. The ε=0.1 change narrows the Lorentzian linewidth, making the resonance peaks sharper and more selective.

**V3 (+1.55 pp for GWL: 90.33%).** Changed test_size from 0.20 to 0.25, giving a larger test set and therefore a more reliable accuracy estimate. The actual performance improvement was likely a combination of the larger training set (75% vs. 80% of the data) and the more stable accuracy estimate.

**V4 (0 change, 90.33%).** Architectural refactoring — separated `_build_manifold`, `_ricci_flow_gpu`, `_wave_energy_batch`, `fit`, `predict` into clean sub-operations. No algorithmic changes, no performance change. This was discipline: before adding more complexity, make the existing code clean enough to reason about.

**V5 (+2.30 pp for GWL: 92.63%).** Added the HRF kernel with fusion weight 1.5 and $d^{2.5}$. Also promoted `hrf_freq` and `hrf_gamma` to constructor arguments for later parameterization. The 2.30 pp improvement from adding local texture to the global manifold signal confirms the complementarity of the two signals.

**V13 (+0.83 pp for GWL: 93.46% — peak).** Corrected $d^{2.5} \rightarrow d^2$, changed fusion weight 1.5→2.0 (holographic term is dominant), tightened query neighborhood k: 8→5. Replaced BaggingClassifier with custom polychromatic loop sweeping $(\omega, \gamma, k)$ across the spectrum $[8,50] \times [0.2,15] \times [12,28]$. This is the project's accuracy peak.

**V13.A SCWH (93.02%).** Added Non-Monotonic Spectral Gating inside the Lorentzian energy. Comparable to V13 GWL, suggesting the gating provides a similar benefit to spectral sharpening as the evolved manifold geometry.

**V13.B AQGL (92.52%).** Pre-metric exponential distance warping ($\alpha = 2.5$) plus temporal phase coupling (+0.5 affinity for $|i-j| \leq 2$) plus dual-axis feature sampling. The result (92.52%) falls below SCWH (93.02%) and GWL (93.46%), confirming that one-shot global warping is less powerful than Ricci flow's iterative local refinement for this dataset.

**V13.C MFT-HRF (92.96%).** 50-frequency vectorized HRF tensor with soft gating (±20%). Strong result for a Laplacian-free classifier — it reaches 92.96% using only local neighborhood information and multi-frequency resonance filtering.

**V14 (93.02% SCWH / 92.52% AQGL / 92.96% MFT-HRF).** Unified sparse Lanczos pipeline, temporal splicing, phase-aligned holographic prediction, extreme hard gating, dual-axis sampling in `PolychromaticForest`. V14 architectures do not surpass the V13 GWL peak. This is an important result: the V14 innovations (sparsity, temporality, gravity warping) are architecturally richer, but the simpler V13 GWL with dense eigendecomposition + HRF fusion + polychromatic spectral diversity produces the best predictions on this dataset.

| Version | RWC | GWL | Key Change |
|---------|-----|-----|------------|
| V1 | 70.03% | 67.46% | Mean-field energy, K=30 |
| V2 | 83.18% | 89.55% | Per-sample einsum, K=128, topological mask |
| V3 | 84.73% | 90.33% | 75/25 split |
| V4 | 84.73% | 90.33% | Architectural refactor only |
| V5 | 91.40% | 92.63% | HRF kernel (d^2.5), fusion=1.5 |
| V13 | 92.66% | **93.46%** | d^2, fusion=2.0, polychromatic spectrum |
| V13.A SCWH | 93.02% | — | Non-monotonic spectral gating |
| V13.B AQGL | 92.52% | — | Gravity warping + temporal coupling |
| V13.C MFT-HRF | 92.96% | — | 50-freq tensor, soft gating |
| V14 SCWH | 93.02% | — | Sparse eigsh, temporal splice, phase holography |
| V14 AQGL | 92.52% | — | + gravity warp |
| V14 MFT-HRF | 92.96% | — | Hard gating, k=5 strict local |

**Total gain from baseline to peak: +25.99 pp (GWL V1 → GWL V13 polychromatic).**

---

## Part VII: Dataset and Feature Engineering

### The EEG Eye State Dataset (OpenML 1471)

14,980 samples, 14 continuous EEG channels from the Emotiv Epoc headset (electrodes AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4), binary label (0 = eyes open, 1 = eyes closed). Class balance is approximately 55%/45%. Recorded as a continuous time series — this is important for the temporal splicing in V14.

### Why the Preprocessing Pipeline Matters

Raw EEG signals are not Euclidean. They have heavy tails (electrode pops, motion artifacts), strong common-mode noise (power-line interference, slow DC drift), and most of the class-discriminative information is in the frequency domain (alpha waves at 8-12 Hz are strongly suppressed during eye opening and strongly present during eye closure). The 78-dimensional processed feature space was designed to make the geometric manifold as informative as possible.

**Step 1: Artifact Clipping.** `X = clip(X_raw, -15, +15)`. Electrode pops and motion artifacts can produce values of hundreds of µV, which would dominate the Euclidean distance computation and cause the k-NN graph to connect these corrupted samples to each other rather than to their true topological neighbors. The ±15 µV clip is conservative — it removes the most extreme outliers while preserving genuine high-amplitude neural events.

**Step 2: Bipolar Montage.** $X_{\text{diff}}[:,j] = X[:,j] - X[:,j+1]$ for $j = 0, \ldots, 12$ (13 differential channels) plus $X_{\text{coh}} = \text{Var}(X, \text{axis}=1)$ (1 instantaneous coherence channel). This is standard clinical EEG practice: bipolar differencing cancels common-mode noise (any signal present in both electrodes of a pair cancels out, leaving only the differential local signal). The coherence channel captures instantaneous cross-channel synchrony, which is a physiological marker of eye-state transitions — during eye closure, alpha oscillations synchronize across channels, increasing the inter-channel variance.

**Step 3: FFT Spectral Features.** `X_spec = |fft(X_raw)|[:, :50]`. The first 50 one-sided FFT bins encode the delta (1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (13-30 Hz), and gamma (30+ Hz) bands. Alpha power is the single most discriminative feature for eye state classification — a well-known neuroscientific fact that the spectral features make directly available to the manifold.

**Step 4: Robust Scaling.** `RobustScaler(quantile_range=(15.0, 85.0))`. Centers on the median and scales by the 15th–85th percentile range. The asymmetric quantile choice (wider than the standard 25th–75th) is deliberate: EEG distributions are moderately heavy-tailed, and the wider range reduces the influence of tail events while keeping the central distribution well-scaled. This is strictly superior to z-score normalization for heavy-tailed distributions.

**Final feature space:** $14 + 13 + 1 + 50 = 78$ dimensions.

---

## Part VIII: GPU Implementation Architecture

The computational bottleneck is the eigendecomposition of the Laplacian, which is $O(N^2)$ in memory and $O(N^3)$ in compute for dense matrices (V1–V13) and $O(N \cdot K)$/$O(N \cdot K^2)$ for sparse Lanczos (V14). On the NVIDIA T4 (16 GB VRAM), this limits the dense path to approximately $N = 11{,}000$ with K=128 before exceeding the VRAM budget.

| Operation | Library | Memory | Compute |
|-----------|---------|--------|---------|
| k-NN exact Euclidean (V1–V13) | cuML NearestNeighbors | O(N·k) | O(N·k·d) |
| Sparse COO assembly (V14) | cupyx.scatter_add | O(nnz = N·k) | O(N·k) |
| Dense eigendecomposition (V1–V13) | cp.linalg.eigh | O(N²) ≈ 576 MB at N=12k | O(N³) |
| Sparse Lanczos (V14) | cupyx.scipy.sparse.linalg.eigsh | O(N·K) ≈ 6 MB at N=12k, K=128 | O(N·K²) |
| Lorentzian kernel | cp broadcast | O(F·K) ≈ 15 KB | O(F·K) |
| Batched einsum energy | cp.einsum | O(batch·K·F) ≈ 8 MB | O(N_c·K·F/batch) |
| 50-freq HRF tensor | cp vectorized | O(B·k·50) | O(B·k·50) |
| VRAM reclaim between trees | cp.get_default_memory_pool() | freed | — |

The explicit `cp.get_default_memory_pool().free_all_blocks()` call between polychromatic forest trees is essential: each tree's Laplacian eigendecomposition allocates temporary VRAM that CuPy's memory pool may not release automatically, and without explicit reclaim, the 15-tree forest would exhaust the T4's 16 GB budget by tree 5 or 6.

---

## Part IX: What This Framework Is and Where It Points

### 9.1 What It Is

The RWC-GWL framework is a **geometry-first supervised classification** system. Rather than optimizing a parameterized hypothesis over a loss function, it:

1. Constructs a discrete approximation to the data manifold (graph + Laplacian)
2. Either fixes or evolves that manifold (RWC vs. GWL)
3. Classifies by measuring quantum-mechanical wave resonance on the manifold
4. Supplements with local holographic texture (HRF)
5. Aggregates multiple spectral views (polychromatic ensemble)

There are no gradient updates, no loss functions, no backpropagation. The "learning" is in the Ricci flow geometry optimization and the spectral parameter sweep of the ensemble. This makes the system interpretable in mathematical terms: the eigenvectors have geometric meaning, the resonance energy has a physical interpretation, and the Ricci flow has a rigorous differential-geometric foundation.

### 9.2 Remaining Limitations

The framework's largest unresolved challenge is **scaling**. The dense eigendecomposition is the practical bottleneck. V14's sparse Lanczos reduces VRAM from ~576 MB to ~6 MB at N=12k, but the Lanczos iterations themselves are still $O(N \cdot K^2)$ which grows with $N$. For $N \sim 10^6$, we would need either better manifold compression (reducing the effective $N$ via hierarchical graph coarsening) or randomized spectral methods (stochastic Lanczos, randomized SVD) that trade exact eigenpairs for approximate ones at lower cost.

The Ricci flow stability also becomes an issue at high $\eta$ — edges can collapse to zero too quickly, fragmenting the graph before the flow reaches a useful geometry. The `flow_lr=0.08` setting is conservative; I explored up to `flow_lr=1.5` in grid search but found instability beyond 0.5 for this dataset.

### 9.3 Future Directions

The directions I'm most interested in pursuing:

**Residual correction.** The current framework makes no special accommodation for hard samples — the points near the eye-state transition boundary that are most likely to be misclassified. A residual correction stage that identifies high-uncertainty predictions and applies a refined local classifier (denser graph, more Ricci flow steps) could push accuracy beyond 93.46%.

**Better manifold compression.** Rather than working with all $N$ training points, hierarchical graph coarsening algorithms (like algebraic multigrid) could produce a compressed graph of $\tilde{N} \ll N$ "super-nodes" that preserves the manifold geometry at a fraction of the computational cost. This would enable dense eigendecomposition on the compressed graph and avoid the Lanczos approximation.

**Class-conditional geometry.** Right now, GWL evolves a single shared manifold and then injects class information via the potential $V^{(c)}$. An alternative is to evolve a *separate* manifold for each class, building a class-$c$ Ricci flow that ignores cross-class information entirely. Classification would then compare how well a query fits each class's own manifold, rather than measuring resonance on a shared potential-perturbed manifold.

**Hybrid RWC-GWL ensemble.** A polychromatic forest that mixes RWC and GWL trees — some trees evolving the manifold, others keeping it static — would combine the global-geometry strengths of RWC (which is more stable to Ricci flow instability) with the discriminative-geometry strengths of GWL.

---

## Appendix: Key Equations Reference

**Zelnik-Manor Affinity:**

$$W_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{\sigma_i \sigma_j + \varepsilon}\right), \quad \sigma_i = d(x_i, x_{(k)}^i)$$

**Symmetric Normalized Laplacian:**

$$\mathcal{L} = I - D^{-1/2} W D^{-1/2}, \quad \text{spectrum} \in [0,2]$$

**Class Potential and Perturbed Resonance Levels:**

$$V^{(c)}_{ii} = \begin{cases} -\alpha & y_i = c \\ +\alpha/2 & y_i \neq c \end{cases}, \quad \mu_m^{(c)} = \lambda_m + \langle\phi_m, V^{(c)}\phi_m\rangle$$

**Lorentzian Resonance Energy:**

$$E(q,c) = \sum_f \sum_m \sum_{s \in \mathcal{S}_c} \frac{\varepsilon}{\pi[(\omega_f^2 - |\mu_m^{(c)}|)^2 + \varepsilon^2]} \langle\phi_q,\phi_m\rangle\langle\phi_m,\phi_s\rangle$$

**Ollivier-Ricci Curvature Approximation:**

$$\kappa_{ij} = \left(W_{ij}\left(\frac{1}{\deg_i}+\frac{1}{\deg_j}\right) - W_{ij} \cdot \frac{D_{\sqrt{W}}(i) + D_{\sqrt{W}}(j) - 2\sqrt{W_{ij}}}{\sqrt{W_{ij}}+\varepsilon}\right) \cdot \mathbf{1}[W_{ij}>10^{-10}]$$

**Label-Driven Ricci Flow Euler Step:**

$$W^{(t+1)} = \text{clip}\!\left(W^{(t)} + \eta\,\kappa^{(t)} W^{(t)} + \mathcal{T}^{(t)}, 0, +\infty\right), \quad \text{then symmetrize}$$

**HRF Kernel:**

$$\Psi(d; \omega, \gamma) = \exp(-\gamma d^2)(1 + \cos(\omega d))$$

**SCWH Phase-Aligned Energy:**

$$\psi_i = |\Phi[i,:]\cdot\phi_q| \cdot \exp(-\gamma d_i^2) \cdot (1 + \cos(\omega d_i))$$

$$E_{\text{SCWH}}(q,c) = \frac{\sum_i w_i^{\text{proj}} \mathbf{1}[y_i=c]}{\sum_i w_i^{\text{proj}} + \varepsilon} + 2.0\sum_i \psi_i \mathbf{1}[y_i=c]$$

**AQGL Gravity Warping:**

$$
d'_{ij} = d_{ij} \cdot e^{-\alpha \cdot \mathbf{1}[y_i=y_j] + \alpha \cdot \mathbf{1}[y_i \neq y_j]}, \quad \alpha = 2.5
$$

**MFT-HRF 3D Tensor:**

$$
\mathbf{W}[b,j,f] = \exp(-\gamma_f d_{bj}^2)(1+\cos(\omega_f d_{bj})), \quad \text{Hard gate: } \begin{cases} w^2 & w > \bar{w} \\ 0 & w \leq \bar{w} \end{cases}
$$


---

*End of White Paper.*

*Dataset: EEG Eye State, OpenML 1471. Platform: NVIDIA T4 GPU, CuPy 13, cuML 24. Peak accuracy: 93.46% (GWL Polychromatic Forest, V13). Authors: Devanik Debnath + Xylia.*
