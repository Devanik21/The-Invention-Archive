https://github.com/Devanik21/Riemannian-Wave-Geometry


# Riemannian Wave Classifier & Geometric Wave Learner

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accelerator-NVIDIA_T4_GPU-76b900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-CuPy_%7C_cuML-FF6F00?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-EEG_Eye_State_(OpenML_1471)-6d28d9?style=flat-square"/>
  <img src="https://img.shields.io/badge/Peak_Accuracy-93.46%25_(GWL_V13)-FFD700?style=flat-square"/>
  <img src="https://img.shields.io/badge/Versions-V1_%E2%86%92_V14_(14_Milestones)-fbbf24?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Authors-Devanik_Debnath_%7C_Xylia-black?style=flat-square&logo=github"/>
</p>

> *Two novel GPU-accelerated classifiers — RWC and GWL — that treat machine learning as a problem of wave physics on a Riemannian manifold. Classification is performed not by learning a decision boundary, but by measuring quantum-mechanical resonance energies on a continuously evolving geometric surface sculpted by discrete Ricci flow. Across 14 iterative milestones, five distinct architectural generations are explored — with the **GWL Polychromatic Forest (V13, 93.46% accuracy)** representing the project's accuracy peak, and the V14 generation (SCWH, AQGL, MFT-HRF) constituting a parallel architectural exploration — a +26.00 percentage-point absolute gain from the 67.46% baseline, all on a single NVIDIA T4 GPU.*

---

**Research Intersection:** `Riemannian Differential Geometry` · `Spectral Graph Theory` · `Discrete Ricci Flow (Hamilton)` · `Quantum Scattering Mechanics (Breit-Wigner)` · `Holographic Optics / Gabor Wavelets` · `Temporal Graph Signal Processing` · `Asymmetric Metric Warping` · `Polychromatic Ensemble Learning` · `EEG Neuroscience` · `CUDA Scientific Computing`

---

# The Benchmarks

*Version 1*

<img width="1667" height="1067" alt="image" src="https://github.com/user-attachments/assets/ab0acef9-6955-477f-b115-4fb57ed1b305" />

---

*Version 2*
<img width="2085" height="1385" alt="image" src="https://github.com/user-attachments/assets/8197cede-faa9-4682-8f49-3af401e07cab" />

---
## Table of Contents

1. [Abstract](#abstract)
2. [Why This Work Is Uniquely Positioned](#why-this-work-is-uniquely-positioned)
3. [Dataset and Preprocessing Pipeline](#dataset-and-preprocessing-pipeline)
4. [Core Mathematical Framework](#core-mathematical-framework)
   - [4.1 Graph Construction and Zelnik-Manor Bandwidth](#41-graph-construction-and-zelnik-manor-bandwidth)
   - [4.2 Symmetric Normalized Graph Laplacian](#42-symmetric-normalized-graph-laplacian)
   - [4.3 Class Potential Injection and Perturbed Hamiltonian](#43-class-potential-injection-and-perturbed-hamiltonian)
   - [4.4 Lorentzian Wave Resonance Energy](#44-lorentzian-wave-resonance-energy)
   - [4.5 Discrete Ollivier-Ricci Curvature](#45-discrete-ollivier-ricci-curvature)
   - [4.6 Label-Driven Discrete Ricci Flow](#46-label-driven-discrete-ricci-flow)
   - [4.7 Holographic Radial Frequency Kernel](#47-holographic-radial-frequency-kernel)
5. [Extended Architectures: V13 Generation and V14](#extended-architectures-v13-generation-and-v14)
   - [5.1 Non-Monotonic Spectral Gating (SCWH)](#51-non-monotonic-spectral-gating-scwh)
   - [5.2 Temporal Phase Coupling](#52-temporal-phase-coupling)
   - [5.3 Asymmetric Quantum Gravity Warping (AQGL)](#53-asymmetric-quantum-gravity-warping-aqgl)
   - [5.4 Vectorized Multi-Frequency HRF Tensor (MFT-HRF)](#54-vectorized-multi-frequency-hrf-tensor-mft-hrf)
   - [5.5 Phase-Aligned Constructive Interference (SCWH predict)](#55-phase-aligned-constructive-interference)
   - [5.6 Sparse Laplacian Eigensolver](#56-sparse-laplacian-eigensolver)
   - [5.7 Dual-Axis Feature Subspace Sampling](#57-dual-axis-feature-subspace-sampling)
6. [Complete Iteration History: V1 to V14](#complete-iteration-history-v1-to-v14)
7. [Performance Results](#performance-results)
8. [System Architecture and Class Hierarchy](#system-architecture-and-class-hierarchy)
9. [GPU Implementation Details](#gpu-implementation-details)
10. [Hyperparameter Reference](#hyperparameter-reference)
11. [Getting Started](#getting-started)
12. [Usage](#usage)
13. [Requirements](#requirements)
14. [Authors](#authors)
15. [License](#license)

---

## Abstract

The **Riemannian Wave Classifier (RWC)** and **Geometric Wave Learner (GWL)** are two original classification algorithms that reframe supervised learning as a problem of wave propagation on a discrete, curved manifold. Rather than optimizing a parameterized hypothesis over a loss surface, both algorithms ask a fundamentally different question: given a geometric representation of the training data, does a query point *resonate* more strongly with the wave modes of one class or another?

In RWC the manifold is **static**. A symmetric normalized Graph Laplacian `L` is constructed from the k-NN affinity graph using Zelnik-Manor self-tuning bandwidths. Its eigendecomposition yields spatial harmonics `{phi_m}` and structural frequencies `{lambda_m}` encoding the data manifold's geometry. A class-conditional potential `V^(c)` is injected to form a perturbed Hamiltonian `H^(c) = L + V^(c)`, and query points are classified via a **Lorentzian (Breit-Wigner) resonance integral** — the same mathematical structure that governs scattering cross-sections in nuclear physics.

In GWL the manifold is **dynamic**. Before spectral analysis, the edge-weight matrix — the discrete metric tensor — undergoes a **Label-Driven Discrete Ricci Flow**: same-class edges are attracted and cross-class edges are repelled, warping the feature space into disjoint class clusters and maximizing the spectral gap between classes.

The project then explores five architectural generations beyond this foundation:

- **V13 Polychromatic Forests** — ensembles where each tree samples a unique spectral "color" `(omega, gamma, k)` of the manifold response
- **V13.A SCWH** — Sparse Complex Wave Holography with Non-Monotonic Spectral Gating and Phase-Aligned Constructive Interference
- **V13.B AQGL** — Asymmetric Quantum Gravity Learning with pre-metric distance warping and Temporal Phase Coupling
- **V13.C MFT-HRF** — Multi-Frequency Tensor HRF with a fully vectorized 50-frequency GPU tensor
- **V14 Dynamic Adaptive Manifold** — the final architecture unifying sparse eigensolver, temporal splicing, phase holography, asymmetric gravity warping, extreme spectral gating, and dual-axis polychromatic forests into three distinct classifiers (SCWH, AQGL, MFTHRF) under a generic `PolychromaticForest` wrapper

Evaluated on the EEG Eye State dataset (OpenML ID 1471, N = 14,980), the **GWL Polychromatic Forest (V13)** achieves **93.46% test accuracy** — rising from a 67.46% baseline, a **+26.00 percentage-point absolute gain** across 14 documented milestones. The V14 generation (SCWH: 93.02%, AQGL: 92.52%, MFT-HRF: 92.96%) constitutes a parallel architectural exploration introducing sparse eigensolvers, temporal phase coupling, asymmetric gravity warping, and multi-frequency HRF tensors.

---

## Why This Work Is Uniquely Positioned

This project occupies a genuinely rare intersection of disciplines where each field contributes a structurally irreplaceable component. No prior classifier, to our knowledge, simultaneously incorporates all of the following.

**Riemannian Differential Geometry.** Edge weights `W_ij` constitute the discrete metric tensor `g_ij`. The Laplace-Beltrami operator, Ollivier-Ricci curvature, and Hamilton's Ricci flow are computed as genuine differential-geometric quantities on this structure — not as analogy but as direct application.

**Quantum Mechanics (Structural Isomorphism).** The perturbed Hamiltonian `H^(c) = L + V^(c)` is mathematically identical to a Schrodinger Hamiltonian with a class-dependent scalar potential. The Lorentzian resonance integral mirrors the Breit-Wigner scattering amplitude of nuclear physics. Spectral interpolation of query points is the quantum-mechanical projection onto eigenstates of a potential operator.

**Discrete Ricci Flow (Modern Topology).** Hamilton's `dg/dt = -2R` — the PDE used by Perelman to prove the Poincare conjecture — is discretized and repurposed as a supervised metric optimization step. The label-tensioning term directs it toward class separation, an application of one of the deepest results in modern geometry to a machine learning preprocessing problem.

**Asymmetric Gravitational Metric Warping (AQGL).** The AQGL architecture pre-warps the pairwise distance matrix before graph construction using exponential scaling — same-class distances are contracted by `exp(-alpha)` while cross-class distances are expanded by `exp(+alpha)`. This "quantum gravity" field reshapes the metric tensor directly, achieving in one deterministic step what Ricci flow achieves iteratively.

**Holographic Optics and Gabor Wavelet Theory.** The HRF kernel `Psi(d) = exp(-gamma*d^2)*(1+cos(omega*d))` has the exact mathematical form of a 1D Gabor filter or holographic fringe pattern. The SCWH architecture further introduces **phase-aligned constructive interference**: the manifold amplitude `|phi[neighbors] . phi_q|` modulates the HRF envelope, producing a readout that mirrors the coherence measurement of an optical hologram.

**Temporal Graph Signal Processing.** The Temporal Phase Coupling mechanism — injecting +0.5 affinity between graph nodes whose row indices differ by at most 2 — explicitly encodes the chronological structure of the EEG recording into the graph topology. This prevents manifold fracturing at rapid eye-state transitions, a domain-specific innovation grounded in the sequential nature of EEG time series.

**Non-Monotonic Spectral Gating.** A novel non-linearity applied inside the resonance energy computation: `K_gated = K * where(|K| > mean(|K|), 1.5, 0.1) * |K|`. This amplifies strong resonance contributions and suppresses weak ones, acting as a learned attention mechanism over the frequency-mode interaction matrix — without any trainable parameters.

**GPU-Accelerated Sparse Spectral Algebra.** V14 replaces the dense `cp.linalg.eigh` with `cupyx.scipy.sparse.linalg.eigsh`, computing only the `K+1 = 129` smallest eigenvectors via Lanczos iteration rather than the full `N x N` spectrum. This reduces VRAM footprint from `O(N^2)` to `O(N*K)` for the eigendecomposition, enabling practical scaling to larger manifolds.

**EEG Neuroscience.** Bipolar montage re-referencing, spectral FFT features, and coherence estimation are domain-specific preprocessing steps grounded in clinical EEG methodology, not generic feature engineering.

---

## Dataset and Preprocessing Pipeline

### EEG Eye State (OpenML ID 1471)

| Property | Value |
|----------|-------|
| Source | UCI / OpenML Dataset #1471 |
| Samples | 14,980 |
| Raw Features | 14 continuous EEG channels (AF3–AF4 Emotiv Epoc headset) |
| Target | Binary: 0 = eyes open, 1 = eyes closed |
| Class Balance | ~55% / 45% |
| Evaluation Protocol (V1–V13) | StratifiedShuffleSplit, test_size=0.25, random_state=42 |
| Evaluation Protocol (V14) | StratifiedShuffleSplit, test_size=0.20, random_state=42 |

### Preprocessing Pipeline

**Step 1 — Artifact Clipping.**

```python
X = np.clip(X_raw, -15, +15)
```

Suppresses electrode pop events and motion artifacts exceeding ±15 µV.

**Step 2 — Bipolar Montage (Spatial Derivative).**

```
X_diff[:, j] = X[:, j] - X[:, j+1]    for j = 0, ..., 12   (13 differential channels)
X_coh = Var(X, axis=1, keepdims=True)                         (1 coherence channel)
```

Bipolar differencing cancels common-mode noise (power-line interference, slow drift). The coherence feature captures instantaneous cross-channel synchrony, a physiological marker of cognitive state transitions.

**Step 3 — Spectral Magnitude Features.**

```python
X_spec = np.abs(np.fft.rfft(X_raw, axis=1))[:, :50]
```

The first 50 one-sided FFT magnitude bins encode the clinically relevant delta (1-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (13-30 Hz), and gamma (30+ Hz) bands.

**Final processed feature dimensionality:** `14 + 13 + 1 + 50 = 78`

**Step 4 — Robust Scaling.**

```python
RobustScaler(quantile_range=(15.0, 85.0))
```

Centers on the median; scales by the 15th–85th percentile range. Strictly superior to z-score normalization for heavy-tailed EEG distributions.

---

## Core Mathematical Framework

### 4.1 Graph Construction and Zelnik-Manor Bandwidth

For training set `X ∈ R^{N x d}`, a symmetric k-NN affinity graph `G = (V, E, W)` is constructed. Edge weights use the **Zelnik-Manor self-tuning bandwidth** (NIPS 2004), which adapts the Gaussian kernel to local data density:

```
W_ij = exp(-d^2_ij / (sigma_i * sigma_j + epsilon))    if j in N_k(i)
     = 0                                                 otherwise
```

where `sigma_i = d(x_i, x_{k(i)})` is the distance to `x_i`'s k-th nearest neighbor. This prevents both the false-closeness artifact in sparse regions and the false-distance artifact in dense regions that a global bandwidth would cause. After GPU sparse assembly via `cupyx.scatter_add`, the matrix is symmetrized: `W <- (W + W^T) / 2`.

### 4.2 Symmetric Normalized Graph Laplacian

The symmetric normalized Graph Laplacian:

```
L = I - D^{-1/2} W D^{-1/2}

where  D_ii = sum_j W_ij,   (D^{-1/2})_ii = 1/sqrt(D_ii)
```

has spectrum in `[0, 2]` and yields real orthonormal eigenvectors. The spectral decomposition `L Phi = Lambda Phi` produces spatial harmonics `phi_m` (analogous to spherical harmonics) and structural frequencies `lambda_m` encoding the manifold's oscillation geometry. The trivial eigenvector `phi_0` with `lambda_0 = 0` is discarded; the retained basis is `Phi_trunc = [phi_1 | ... | phi_K]`, `K = n_components = 128`.

Dense computation uses `cp.linalg.eigh(L)`. V14 replaces this with sparse `eigsh` (Section 5.6).

### 4.3 Class Potential Injection and Perturbed Hamiltonian

A class-conditional potential operator `V^(c)` is injected as a diagonal perturbation:

```
V^(c)_ii = -potential_strength          if y_i = c     (potential well: attraction)
            +0.5 * potential_strength    if y_i != c    (potential barrier: repulsion)
```

Projected onto the spectral basis, this yields perturbed resonance levels:

```
mu_m^(c) = lambda_m + sum_i V^(c)_ii * |phi_m(i)|^2 = lambda_m + <phi_m, diag(V^(c)) phi_m>
```

The operator `H^(c) = L + V^(c)` is a quantum Hamiltonian: `L` is the kinetic energy operator and `V^(c)` is a scalar potential landscape shaped by class labels. Training points of class `c` create deep wells (lowering eigenvalues and trapping wave modes near those points), while non-class points create barriers (scattering wave modes away).

### 4.4 Lorentzian Wave Resonance Energy

Query points are mapped into the spectral domain via Gaussian kernel interpolation over k=5 nearest training neighbors:

```
w_i = exp(-d^2_i / (2 * d_bar^2))
phi_q = sum_i (w_i / sum_j w_j) * Phi_trunc[i, :]
```

The classification energy is a **Lorentzian (Breit-Wigner) resonance integral**:

```
E(q, c) = sum_f sum_{m,c'} [epsilon / (pi * ((omega_f^2 - |mu_m^(c)|)^2 + epsilon^2))]
           * <phi_q, phi_m> * <phi_m, phi_{c'}>
```

The Lorentzian factor peaks sharply when `omega_f^2 ≈ |mu_m^(c)|` — exactly as a driven oscillator resonates at its natural frequency, or as a nucleus has peak scattering cross-section at its resonance energy.

GPU implementation via batched Einstein summation (batch_size=500 for VRAM safety):

```python
K_batch = cp.einsum('fm, qm, cm -> qcf', lor, phi_q_g, phi_c_batch)
energies += cp.sum(K_batch, axis=(1, 2))
```

Classification: `y_hat = argmax_c E(q, c)`.

**Correctness note (V1 vs V2+):** V1 collapsed all class training points to a single spectral vector `phi_c_train.sum(axis=0)`, producing destructive interference between distant same-class points. The V2 per-sample einsum fix — the largest single improvement in the project at +22 pp — correctly accumulates the resonance overlap from every class sample independently.

### 4.5 Discrete Ollivier-Ricci Curvature

The Ollivier-Ricci curvature `kappa_ij = 1 - W_1(mu_i, mu_j) / d(i,j)` (positive = sphere-like, negative = saddle-like) is approximated via a square-root transport construction:

```
base_ij    = W_ij * (d^{-1}_i + d^{-1}_j)
S_ij       = sqrt(W_ij)
D_S_i      = sum_j S_ij
penalty_ij = (D_S_i + D_S_j - 2*S_ij) / (S_ij + eps)
kappa_ij   = (base_ij - W_ij * penalty_ij) * mask_ij
```

The binary `mask = (W > 1e-10)` restricts flow to existing edges — the topological fabric constraint that prevents creation of spurious long-range connections.

### 4.6 Label-Driven Discrete Ricci Flow

Hamilton's continuous Ricci flow `dg_ij/dt = -2 R_ij` is discretized on the edge-weight matrix, augmented by a Label-Tensioning term:

```
dW_ij/dt = -kappa_ij * W_ij * flow_lr  +  T_ij

where  T_ij = +flow_lr * W_ij    if y_i = y_j  AND mask_ij    (intra-class attraction)
              -flow_lr * W_ij    if y_i != y_j AND mask_ij    (inter-class repulsion)
```

Euler step with non-negativity clipping:

```
W(t+1) = clip(W(t) + flow_lr * kappa(t) * W(t) + T(t), 0, +inf)
W(t+1) = (W(t+1) + W(t+1)^T) / 2
```

After `flow_steps = 10` iterations, `W_evolved` encodes the label-warped geometry. Intra-class clusters are cohesive (high affinity), inter-class separations are widened (collapsed edges), maximizing the spectral gap between classes.

### 4.7 Holographic Radial Frequency Kernel

The HRF kernel is a modulated Gaussian applied to local neighbor distances:

```
Psi(d) = exp(-gamma * d^2) * (1 + cos(omega_hrf * d))
```

The Gaussian envelope `exp(-gamma*d^2)` localizes the response to the immediate neighborhood. The oscillatory carrier `(1 + cos(omega*d))` creates a radial standing-wave fringe pattern — sensitive to whether a neighbor falls within a specific radial band. This is precisely the structure of holographic fringe patterns in optics and Gabor wavelets in signal processing.

HRF classification energy: `E_HRF(q, c) = sum_{i in N_local(q)} Psi(d_qi) * 1[y_i = c]`

V13 final prediction fuses global and local signals:

```
E_final(q, c) = E_GWL_norm(q, c) + 2.0 * E_HRF_norm(q, c)

where  E_norm(q, c) = E(q, c) / (max_c |E(q, c)| + eps)
```

The weight 2.0 (raised from 1.5 in V5) reflects that local oscillatory texture is the dominant discriminant for the EEG eye-state manifold.

---

## Extended Architectures: V13 Generation and V14

### 5.1 Non-Monotonic Spectral Gating (SCWH)

Introduced in V13.A (Cell 22). A novel non-linearity applied inside `_wave_energy_batch` after the standard Lorentzian computation:

```python
energy_magnitude = cp.abs(K_batch)
gate = cp.where(energy_magnitude > cp.mean(energy_magnitude, axis=1, keepdims=True),
                1.5, 0.1)
K_gated = K_batch * gate * energy_magnitude
```

For each element of the `(query, class_sample, frequency)` energy tensor `K_batch`:
- Elements whose absolute value exceeds the row mean receive a **1.5x amplification** and are additionally scaled by their magnitude (super-linear boosting of strong resonances)
- Elements below the row mean receive a **0.1x attenuation** (suppression of weak off-resonance contributions)

This is a hard attention mechanism over the frequency-mode interaction matrix, operating without trainable parameters. Mathematically it implements:

```
K_gated_ij = K_ij * |K_ij| * (1.5 * 1[|K_ij| > mean_j |K_ij|] + 0.1 * 1[|K_ij| <= mean_j |K_ij|])
```

The resulting energy accumulation is no longer a linear sum over frequencies but a power-amplified sum that sharpens the resonance peaks — analogous to applying a non-linear activation to the spectral response of a resonator cavity.

### 5.2 Temporal Phase Coupling

Introduced in V13.B's GWL `fit()` (Cell 25) and carried into all V14 architectures. The EEG recording in OpenML 1471 is stored in temporal order (row index = recording timestep). Adjacent temporal frames share high physiological coherence — the brain state does not change instantaneously between consecutive EEG samples.

The coupling injects a constant +0.5 affinity between all graph nodes whose row indices differ by at most 2:

```python
row_indices = cp.arange(N)
temporal_mask = cp.abs(row_indices[:, None] - row_indices[None, :]) <= 2
W = W + (temporal_mask * 0.5)
```

This **prevents manifold fracturing**: without temporal coupling, the Ricci flow can sometimes collapse the transition boundary between eye-open and eye-closed states into a topological discontinuity — a zero-affinity barrier that creates spectral eigenvectors with artificial sharp boundaries at state transitions. The +0.5 temporal coupling ensures that adjacent-timestep nodes remain connected through the flow iterations, preserving the smooth topology of the EEG time-stream.

In sparse format (V14), the temporal coupling is applied at the sparse COO matrix level:

```python
temporal_mask = cp.abs(col_idx - row_idx) <= 2
W_data = W_data + (temporal_mask * 0.5)
```

### 5.3 Asymmetric Quantum Gravity Warping (AQGL)

The central innovation of V13.B and V14's `AQGL_Classifier`. Rather than iteratively deforming the metric via Ricci flow (GWL) or constructing an undirected affinity graph (RWC), AQGL applies **pre-metric asymmetric distance warping** before any graph construction:

```python
row_labels = cp.repeat(y_gpu[:, None], k_neighbors_train, axis=1)
col_labels = y_gpu[indices_gpu]
same_class = (row_labels == col_labels)

warped_dists = cp.where(same_class,
                        dists_gpu * cp.exp(-gravity_alpha),    # same-class: contract
                        dists_gpu * cp.exp(+gravity_alpha))    # cross-class: expand
```

With `gravity_alpha = 2.5`:
- Same-class pairwise distances are multiplied by `exp(-2.5) ≈ 0.082` — contracted to approximately 8.2% of their Euclidean value
- Cross-class pairwise distances are multiplied by `exp(+2.5) ≈ 12.18` — expanded to ~12× their Euclidean value

The Zelnik-Manor Gaussian affinity is then computed on these warped distances, yielding a graph where same-class connections have dramatically higher weight. The graph Laplacian of this warped affinity has a fundamentally different spectral structure from its un-warped counterpart: the lowest eigenvectors are almost entirely intra-class, providing extreme spectral class separation before any potential injection.

This is formally equivalent to placing a class-conditional gravitational field in the metric space: points belonging to the same class are attracted (geodesic distance shrunk) and points of different classes are repelled (geodesic distance expanded), just as masses in a gravitational field warp the surrounding spacetime geometry.

Compared to GWL's iterative Ricci flow, AQGL achieves the class separation in a single vectorized operation — more computationally efficient and with a precisely controlled warp factor `exp(alpha)` per unit distance.

### 5.4 Vectorized Multi-Frequency HRF Tensor (MFT-HRF)

Introduced in V13.C (Cell 28) and V14's `MFTHRF_Classifier`. Rather than computing HRF for one `(omega, gamma)` pair at a time (as in the polychromatic forest's per-tree approach), MFT-HRF computes the full 50-frequency response simultaneously on a GPU tensor:

```python
freq_tensor  = cp.linspace(8.0, 50.0, n_frequencies)    # shape: (50,)
gamma_tensor = cp.linspace(0.2, 15.0, n_frequencies)    # shape: (50,)
dists_exp    = dists_g[:, :, None]                       # shape: (B, k, 1)

w_hrf = cp.exp(-gamma_tensor * dists_exp**2) * (1.0 + cp.cos(freq_tensor * dists_exp))
# w_hrf shape: (B, k, 50)  — B queries, k neighbors, 50 frequencies
```

This produces a 3D tensor of HRF responses where axis 2 is the frequency dimension. The 50 simultaneous responses span the full spectral range from coarse (8 Hz-analog) to fine (50 Hz-analog) in one GPU kernel, equivalent to running 50 parallel Gabor filters at different spatial frequencies.

In V13.C, **soft gating** is applied: `w_hrf = where(w > mean_w, w * 1.2, w * 0.8)` — a mild ±20% modulation.

In V14, this is replaced by **extreme hard gating**: `w_hrf = where(w > mean_w, w**2, 0.0)` — squaring above-threshold values (super-linear amplification of strong resonances) and zeroing below-threshold values (complete silence of weak responses). This winner-take-all frequency gating creates highly peaked spectral responses that are more discriminative than the soft version.

Energy aggregation: `E_HRF(q, c) = sum_{k} mean_{f}(w_hrf[q, k, :] * 1[y_k = c])` — the mean across the frequency tensor stabilizes the readout against spurious frequency-specific noise.

Since `MFTHRF_Classifier` requires no spectral eigendecomposition (`fit()` only stores the training data and builds a kNN index), its training cost is `O(N * k)` versus `O(N^2)` for the Laplacian-based classifiers.

### 5.5 Phase-Aligned Constructive Interference

The signature innovation of V14's `SCWH_Classifier` prediction step. Unlike the original HRF kernel which uses only the Euclidean distance `d` to the neighbor, SCWH incorporates the **manifold amplitude** — the projection similarity between the query's interpolated spectral coordinate and the spectral coordinates of its neighbors:

```python
# Manifold amplitude: how much the neighbor's spectral response aligns with the query
manifold_amp = np.abs(np.dot(self.phi_[idx[i]], phi_q[i]))   # shape: (k,)

# Phase alignment wave
phase_alignment = 1.0 + np.cos(self.hrf_freq * dists[i])

# Combined holographic weight
w_hrf = manifold_amp * np.exp(-self.hrf_gamma * dists[i]**2) * phase_alignment
```

The term `manifold_amp = |phi[neighbors] . phi_q|` is the element-wise inner product between the neighbor's eigenvector row vectors and the query's interpolated spectral coordinate — it measures how much each neighbor's wave mode "agrees" with the query's spectral position. Neighbors that are not only geometrically close but also spectrally aligned with the query receive amplified weight.

The final energy computation uses **Dual-Energy Superposition** — a linear combination of a structural term (normalized distance-weighted vote) and the holographic term:

```python
struct_e = np.sum(w_proj * mask) / (w_proj.sum() + 1e-12)
wave_e   = np.sum(w_hrf * mask)
energy[i, ci] = struct_e + 2.0 * wave_e
```

This fuses the two fundamentally different classification signals: `struct_e` measures simple proximity (how many class-`c` neighbors are close) and `wave_e` measures spectral-holographic coherence (how strongly the query resonates with class-`c` neighbors at its specific phase alignment). The 2.0 fusion weight reflects the empirically dominant contribution of the holographic term.

### 5.6 Sparse Laplacian Eigensolver

All V14 architecture classes replace the dense `cp.linalg.eigh(L)` — which computes the full `N x N` eigenspectrum — with a sparse Lanczos-based partial eigendecomposition:

```python
from cupyx.scipy.sparse.linalg import eigsh
W_sparse = cpsp.coo_matrix((W_data, (row_idx, col_idx)), shape=(N, N)).tocsr()
L_sparse = cpsp.eye(N) - d_inv.dot(W_sparse).dot(d_inv)
vals, vecs = eigsh(L_sparse, k=n_components+1, which='SM')  # only K+1 = 129 eigenpairs
```

The ARPACK Lanczos algorithm `eigsh` computes only the `k+1` smallest eigenpairs rather than the full spectrum, reducing:
- VRAM from `O(N^2)` (dense matrix) to `O(N*k + nnz(W))` where `nnz` is the number of non-zero edges
- Compute from `O(N^3)` (full eigendecomposition) to approximately `O(N * k^2)` (Lanczos iterations)

A CPU fallback (`scipy.sparse.linalg.eigsh`) is invoked if the GPU sparse solver fails, ensuring robustness across different RAPIDS versions.

The sparse affinity matrix is constructed using COO (coordinate) format, retaining only `N * k` non-zero entries rather than the full `N x N` dense matrix, enabling larger training sets with the same VRAM budget.

### 5.7 Dual-Axis Feature Subspace Sampling

Introduced in V13.C's ensemble wrappers (Cell 28) and made canonical in V14's `PolychromaticForest`. Extends the polychromatic spectral diversity with a second axis of randomization: feature subspace sampling.

Each tree receives not only a random row subset but also a random column (feature) subset:

```python
row_indices  = np.random.choice(N, n_samples, replace=False)
feat_indices = np.random.choice(F, n_features, replace=False)  # n_features = max_features * F

X_sub = X[np.ix_(row_indices, feat_indices)]    # train on subspace
model.predict(X[:, f_mask])                      # predict on same subspace
```

This creates three simultaneous diversity axes in the ensemble:
1. **Spectral diversity** — different `(omega_t, gamma_t)` HRF parameters per tree
2. **Topological diversity** — different `k_t` neighborhood sizes per tree
3. **Feature subspace diversity** — different random projections into 78-dimensional feature space per tree

The third axis is particularly powerful for EEG data where different feature subsets (e.g., spectral features vs. bipolar montage features) may capture orthogonal aspects of the eye-state transition, and different trees specializing in different feature regimes will produce decorrelated predictions that, when aggregated, are more robust than any single-feature-space prediction.

The `PolychromaticForest` in V14 is fully generic — it accepts `base_estimator_class` as a constructor argument and dynamically injects the appropriate spectral parameters at fit time based on class identity:

```python
if issubclass(base_estimator_class, (SCWH_Classifier, AQGL_Classifier)):
    kwargs['hrf_freq']          = freq_spectrum[i]
    kwargs['hrf_gamma']         = gamma_spectrum[i]
    kwargs['k_neighbors_train'] = k_train_spectrum[i]
elif issubclass(base_estimator_class, MFTHRF_Classifier):
    kwargs['k_neighbors_test']  = k_test_spectrum[i]
```

---

## Complete Iteration History: V1 to V14

### V1 — Baseline (Cells 4–6)

**Hyperparameters:** `n_components=30, k=20, n_freq=20, epsilon=0.5, potential_strength=10.0, flow_lr=0.08, n_est=15, split=80/20`

Mean-field energy collapse (`phi_c_train.sum(axis=0)`) produces destructive interference across same-class samples. Label tensioning lacks the topological mask. GWL underperforms RWC because Ricci flow adds complexity that the broken energy function cannot leverage.

**Results:** RWC: **70.03%** | GWL: **67.46%**

---

### V2 — Lorentzian Energy Fix (Cells 7–9)

**Changes:** `n_components 30→128`, `k 20→15`, `n_freq 20→30`, `epsilon 0.5→0.1`, `potential_strength 10→15`. Core fix: per-sample batched einsum `'fm,qm,cm->qcf'` replaces mean-field collapse. Topological mask `(W > 1e-10)` introduced for Ricci flow.

The einsum fix is the project's single most impactful change. It computes the correct resonance overlap for every class sample independently, enabling constructive accumulation rather than destructive cancellation. The inversion of ranking (GWL now dominates by +6.37 pp) confirms that Ricci flow was providing genuine geometric benefit all along.

**Results:** RWC: **83.18%** (+13.15 pp) | GWL: **89.55%** (+22.09 pp)

---

### V3 — Evaluation Protocol Correction (Cells 10–12)

**Change:** `test_size 0.20 → 0.25`. Canonical evaluation protocol established.

**Results:** RWC: **84.73%** | GWL: **90.33%**

---

### V4 — Architectural Cleanup (Cells 13–18)

No algorithmic changes. `fit()` and `predict()` refactored into explicit sub-operations; `BaggingClassifier` wrappers unified. Zero regression confirmed.

**Results:** RWC: **84.73%** | GWL: **90.33%**

---

### V5 — HRF Integration (Cells 19–21)

**Changes:** `y_train_` stored in `fit()`. HRF kernel introduced in `predict()`: `w_hrf = exp(-gamma * d^2.5) * (1 + cos(omega * d))`. Fusion: `final = e_gwl_norm + 1.5 * e_hrf_norm`. Query interpolation k=8 unchanged.

The sub-Gaussian `d^2.5` exponent provides broader support than a standard Gaussian, giving meaningful weight to moderately-distant neighbors. Empirically effective — V13 later corrects to `d^2` for theoretical cleanliness.

**Results:** RWC: **91.40%** (+6.67 pp) | GWL: **92.63%** (+2.30 pp)

---

### V13 — Polychromatic Forests + Final HRF (Cells 22–24)

**Changes:** HRF exponent corrected `d^2.5 → d^2`. Fusion weight `1.5 → 2.0`. Query k `8 → 5`. `hrf_freq` and `hrf_gamma` promoted to constructor arguments. `BaggingClassifier` replaced by custom polychromatic loop sweeping `freq_spectrum = linspace(8, 50, n_est)`, `gamma_spectrum = linspace(0.2, 15, n_est)`, `k_spectrum = linspace(12, 28, n_est)`. Each tree receives a unique spectral color `(omega_t, gamma_t, k_t)`. Explicit VRAM reclaim between trees.

**Results:** RWC: **92.66%** | GWL: **93.46%** ← *Project accuracy peak*

This is the polychromatic principle: spectral filter diversity (not merely bootstrap diversity) produces decorrelated tree predictions that aggregate into a robust consensus.

---

### V13.A — SCWH Forest: Complex Holography (Cells 25–27)

**New mechanism:** Non-Monotonic Spectral Gating (Section 5.1) applied inside `_wave_energy_batch`. After computing the standard Lorentzian K_batch tensor:

```python
energy_magnitude = cp.abs(K_batch)
gate = cp.where(energy_magnitude > cp.mean(energy_magnitude, axis=1, keepdims=True), 1.5, 0.1)
K_gated = K_batch * gate * energy_magnitude
```

Strong resonance contributions receive super-linear amplification (`K * 1.5 * |K|`); weak off-resonance contributions are attenuated to 10%. This sharpens the resonance peaks in the energy accumulation without any trainable parameters. The ensemble is otherwise identical to the V13 polychromatic forest.

**Results:** SCWH-RWC: **93.02%** | SCWH-GWL: **93.02%**

---

### V13.B — AQGL Forest: Asymmetric Quantum Gravity (Cells 28–30)

**New mechanism — Distance Pre-Warping:** Before building the affinity graph, pairwise kNN distances are asymmetrically warped (Section 5.3):

```python
warped_dists = cp.where(same_class, dists_gpu * cp.exp(-2.5), dists_gpu * cp.exp(+2.5))
```

Same-class distances contracted to 8.2%; cross-class distances expanded to 1218%. The Zelnik-Manor Gaussian is then applied to these warped distances, producing an affinity graph that is geometrically restructured before any Laplacian is built.

**New mechanism — Temporal Phase Coupling (GWL):** +0.5 affinity injected between temporally adjacent samples `(|row_i - row_j| <= 2)` to prevent manifold fracturing (Section 5.2).

**New mechanism — Dual-Axis Sampling in ensemble:** Each tree now samples both a random row subset and a random feature subset:

```python
feat_indices = np.random.choice(F, n_features, replace=False)
X_sub = X[np.ix_(row_indices, feat_indices)]
```

**Results:** AQGL-RWC: **92.52%** | AQGL-GWL: **92.52%**

Note: AQGL achieves slightly lower accuracy than SCWH, suggesting that the pre-metric warping alone (without the Ricci flow's iterative local refinement) produces a less discriminative manifold for this dataset, despite its more aggressive global class separation.

---

### V13.C — MFT-HRF: Multi-Frequency Tensor (Cells 31–33)

**New mechanism — Vectorized 50-frequency HRF tensor (Section 5.4):**

```python
freq_tensor  = cp.linspace(8.0, 50.0, n_frequencies)
gamma_tensor = cp.linspace(0.2, 15.0, n_frequencies)
dists_exp    = dists_g[:, :, None]
w_hrf = cp.exp(-gamma_tensor * dists_exp**2) * (1.0 + cp.cos(freq_tensor * dists_exp))
# shape: (B, k, 50)
```

Soft gating: `w_hrf = where(w > mean_w, w * 1.2, w * 0.8)`. No Laplacian; `fit()` only stores training data and builds kNN index.

**Results:** MFT-HRF: **92.96%**

---

### V14 — Dynamic Adaptive Manifold (Cell 31 Master Execution)

The final version unifies all preceding innovations into three production-grade architecture classes under a generic `PolychromaticForest` wrapper. Uses strict 80/20 split. Three distinct classifiers:

**`SCWH_Classifier` (Sparse Complex Wave Holography):**
- Sparse Laplacian via `cupyx.scipy.sparse.linalg.eigsh` (K+1 eigenpairs, Lanczos)
- Temporal splicing: `W_data += (temporal_mask * 0.5)` at sparse COO assembly
- Phase-aligned constructive interference with dual-energy superposition: `struct_e + 2.0 * wave_e`
- k_train=20, predict k=5, `hrf_freq` and `hrf_gamma` swept per tree

**`AQGL_Classifier` (Asymmetric Quantum Gravity Learning):**
- Inherits SCWH sparse Laplacian + temporal splicing
- Additionally applies pre-metric exponential distance warping with `gravity_alpha=2.5`
- Same phase-aligned prediction as SCWH

**`MFTHRF_Classifier` (Multi-Frequency Tensor HRF):**
- No Laplacian at all — purely local, `fit()` stores data only
- k=5 strict local search, 50-frequency vectorized HRF tensor
- Extreme hard gating: `w_hrf = where(w > mean_w, w**2, 0.0)` (squaring + zeroing)
- Energy: `sum_k mean_f(w_hrf * mask_c)` for each class `c`

**`PolychromaticForest`:** Generic wrapper with dual-axis sampling (rows + features) and dynamic per-tree spectral parameter injection. Sweeps `k_train in [12, 28]`, `freq in [8, 50]`, `gamma in [0.2, 15]`.

**Results:** SCWH-Forest: **93.02%** | AQGL-Forest: **92.52%** | MFT-HRF-Forest: **92.96%**

Note: V14 architectures represent a parallel exploration of sparse, temporal, and gravity-warped manifold designs. They do not surpass the V13 GWL Polychromatic Forest's accuracy peak of 93.46%.

---

## Performance Results

### Complete 14-Milestone Accuracy Registry

| Milestone | Architecture Name | Description | Accuracy |
|-----------|------------------|-------------|---------|
| **V1.1** | RWC Baseline | Static manifold, mean-field energy, K=30 | 70.03% |
| **V1.2** | GWL Baseline | Basic Ricci Flow, mean-field energy | 67.46% |
| **V2.1** | RWC Lorentzian | Per-sample einsum, K=128, epsilon=0.1 | 83.18% |
| **V2.2** | GWL Label-Driven | Ricci flow directed by ground-truth labels | 89.55% |
| **V3.1** | RWC Spectral Shift | 75/25 evaluation split | 84.73% |
| **V3.2** | GWL Curvature Fix | Topological mask + 75/25 split | 90.33% |
| **V4.1** | RWC Zelnik-Manor | Code refactor (no algorithm change) | 84.73% |
| **V4.2** | GWL Metric Warp | Code refactor (no algorithm change) | 90.33% |
| **V5.1** | RWC Ensemble V13 | HRF kernel (d^2.5), fusion 1.5, k_q=8 | 91.40% |
| **V5.2** | GWL Ensemble V13 | HRF kernel (d^2.5), Ricci + HRF, fusion 1.5 | 92.63% |
| **V13.A** | SCWH Forest | Non-Monotonic Spectral Gating + Polychromatic | 93.02% |
| **V13.B** | AQGL Forest | Asymmetric Gravity Warping + Temporal Coupling | 92.52% |
| **V13.C** | MFT-HRF | Multi-Frequency Tensor, 50-freq vectorized HRF | 92.96% |
| **V13** | GWL Polychromatic Forest | d² HRF, fusion 2.0, k_q=5, spectrum sweep k∈[12,28] | **93.46%** |
| **V13** | RWC Polychromatic Forest | Same spectrum sweep, static manifold (no Ricci) | 92.66% |
| **V14** | SCWH/AQGL/MFT-HRF (parallel) | Sparse eigsh, temporal splicing, gravity warp | 93.02% (best) |

### Cumulative Gain Summary

| Metric | Value |
|--------|-------|
| Absolute gain: baseline → GWL V13 polychromatic | **+26.00 pp** (67.46% → 93.46%) |
| Absolute gain: baseline → RWC V13 polychromatic | +25.20 pp (67.46% → 92.66%) |
| Largest single-step gain | V1→V2 GWL: +22.09 pp (energy function fix) |
| HRF contribution (V4→V5, GWL) | +2.30 pp |
| V13 polychromatic over V5 GWL | +0.83 pp (92.63% → 93.46%) |
| V14 best (SCWH 93.02%) vs V13 GWL peak | −0.44 pp (V14 did not surpass V13 GWL) |

### Benchmark Chart Color Tiers

| Tier | Threshold | Models |
|------|-----------|--------|
| Gold — Project Peak | >= 93.00% | **GWL V13 Polychromatic (93.46%)**, V13.A SCWH (93.02%), RWC V13 (92.66%→ near gold) |
| Orange — Excellent | >= 92.00% | RWC V13 Polychromatic (92.66%), V13.B AQGL (92.52%), V13.C MFT-HRF (92.96%), V5.2 GWL (92.63%) |
| Cyan — Very Good | >= 90.00% | V5.1 RWC (91.40%), V3.2 GWL (90.33%), V4.2 GWL (90.33%) |
| Blue — Research | < 90.00% | V1–V2 versions |

---

## System Architecture and Class Hierarchy

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                   EEG Eye State Dataset (OpenML 1471)                         │
│           N=14,980 samples · 14 EEG channels · Binary label                   │
└──────────────────────────────┬────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────────────────┐
│                      Preprocessing Pipeline                                    │
│   clip(±15) → bipolar_montage(13 diff + 1 coh) → rfft(50 bins)               │
│   → RobustScaler(q=(15,85)) → X_processed: (14980, 78)                        │
└──────────────────────────────┬────────────────────────────────────────────────┘
                               │
              ┌────────────────┴──────────────────────────────┐
              │                                               │
  ┌───────────▼──────────────────────┐   ┌────────────────────▼────────────────┐
  │  RiemannianWaveClassifier (V1-V13)│   │  GeometricWaveLearner (V1-V13)      │
  │  _build_manifold()               │   │  _ricci_flow_gpu()                  │
  │  ├─ Zelnik-Manor W               │   │  ├─ kappa curvature (masked)        │
  │  ├─ dense eigh(L)                │   │  ├─ label tensioning T_ij            │
  │  └─ K=128 eigenpairs             │   │  ├─ 10 Euler steps                  │
  │                                  │   │  └─ Temporal coupling (V13.B+)      │
  │  predict(): k=5 spectral interp  │   │                                     │
  │  + Lorentzian batched einsum     │   │  fit(): Ricci(W,y) → L_evo → Phi    │
  │  + HRF kernel (V5+)              │   └─────────────────────────────────────┘
  │  + NM Spectral Gating (V13.A+)   │
  └──────────────┬───────────────────┘
                 │
     ┌───────────┴──────────────────────────────┐
     │                                          │
┌────▼─────────────────┐   ┌────────────────────▼──────────────────────────────┐
│  V13 Polychromatic   │   │  V14 PolychromaticForest (generic)                 │
│  RWCEnsemble /       │   │  base_estimator_class: SCWH | AQGL | MFTHRF       │
│  GWLEnsemble         │   │  Dual-axis sampling: rows + features               │
│  freq ∈ [8, 50]      │   │  Dynamic per-tree kwargs injection                 │
│  gamma ∈ [0.2, 15]   │   └───────┬──────────────────────────────────────────┘
│  k ∈ [12, 28]        │           │
└──────────────────────┘   ┌───────▼───────────────────────────────────────────┐
                           │                                                    │
               ┌───────────▼──────────┐  ┌──────────▼───────┐  ┌──────────────▼──┐
               │  SCWH_Classifier     │  │  AQGL_Classifier │  │ MFTHRF_Classifier│
               │  ├─ sparse eigsh     │  │  ├─ inherits SCWH│  │ ├─ no Laplacian  │
               │  ├─ temporal splice  │  │  ├─ pre-metric   │  │ ├─ k=5 kNN only  │
               │  ├─ phase holography │  │  │  gravity warp │  │ ├─ 50-freq tensor│
               │  └─ dual-energy:     │  │  │  exp(±2.5)    │  │ └─ hard gating   │
               │     struct + 2*wave  │  │  └─ temporal spl.│  │    w^2 / 0       │
               └──────────────────────┘  └──────────────────┘  └─────────────────┘
```

---

## GPU Implementation Details

| Operation | Library | Memory | Notes |
|-----------|---------|--------|-------|
| k-NN graph (exact Euclidean) | `cuml.neighbors.NearestNeighbors` | O(N*k) | V1–V13 |
| Sparse k-NN assembly | `cupyx.scatter_add` | O(nnz = N*k) | COO format, V14 |
| Asymmetric gravity warp | `cp.where` broadcast | O(N*k) | AQGL only |
| Dense Laplacian eigh | `cp.linalg.eigh` | O(N^2) ≈ 576 MB@N=12k | V1–V13 |
| Sparse Laplacian eigsh | `cupyx.scipy.sparse.linalg.eigsh` | O(N*K) ≈ 6 MB@N=12k,K=128 | V14 |
| Lorentzian factor | `cp` broadcast | O(n_freq * K) = ~15 KB | Float32 |
| Batched einsum energy | `cp.einsum('fm,qm,cm->qcf')` batch=500 | O(500*K*n_freq) ≈ 8 MB | VRAM safe |
| HRF 50-freq tensor | `cp` vectorized ops | O(B * k * 50) | MFTHRF only |
| VRAM reclaim | `cp.get_default_memory_pool().free_all_blocks()` | Freed | Between trees |
| CPU fallback | `scipy.sparse.linalg.eigsh` | — | If cuSPARSE fails |

**V14 memory efficiency:** Replacing dense `eigh` with sparse `eigsh` reduces the eigendecomposition VRAM footprint from ~576 MB (dense `N x N` matrix at N=12,000) to approximately 6 MB (the `N x K` eigenvector matrix), enabling practical scaling to larger manifolds within the T4's 16 GB budget.

---

## Hyperparameter Reference

### Base Classifier Parameters

| Parameter | V1 | V2–V4 | V5 | V13 | V14 (per-tree range) |
|-----------|-----|-------|-----|-----|---------------------|
| `n_components` (K) | 30 | 128 | 128 | 128 | 128 |
| `k_neighbors` (train) | 20 | 15 | 15 | 12–28 | 12–28 |
| `n_freq` | 20 | 30 | 30 | 30 | 30 |
| `epsilon` | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| `potential_strength` | 10.0 | 15.0 | 15.0 | 15.0 | 15.0 |

### HRF Parameters

| Parameter | V5 | V13 | V14 |
|-----------|-----|-----|-----|
| `hrf_freq` (omega) | 30.0 fixed | 8.0–50.0 swept | 8.0–50.0 swept |
| `hrf_gamma` (gamma) | 10.0 fixed | 0.2–15.0 swept | 0.2–15.0 swept |
| HRF exponent | `d^2.5` | `d^2` | `d^2` |
| Fusion weight | 1.5 | 2.0 | 2.0 (struct + 2*wave) |
| Query k | 8 | 5 | 5 |

### GWL / Ricci Flow Parameters

| Parameter | All Versions | Golden Grid Range |
|-----------|-------------|------------------|
| `flow_steps` | 10 | fixed |
| `flow_lr` | 0.08 | [0.3, 0.6, 1.0, 1.5] (grid search) |

### AQGL-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gravity_alpha` | 2.5 | Exponential distance warp factor. Same-class: `d * exp(-alpha)`. Cross-class: `d * exp(+alpha)` |

### MFTHRF-Specific Parameters

| Parameter | V13.C | V14 |
|-----------|-------|-----|
| `n_frequencies` | 50 | 50 |
| `k_neighbors_test` | 256 | 5 |
| Gating | Soft (±20%) | Hard (`w^2` or 0) |

### Ensemble Parameters

| Parameter | V1–V4 | V5 | V13 | V14 |
|-----------|-------|-----|-----|-----|
| `n_estimators` | 15 | 15 | 15 | 15 |
| `max_samples` | 0.75 | 0.75 | 0.75 | 0.75 |
| `max_features` | 1.0 (implicit) | 1.0 | 1.0 | 1.0 (available) |
| Ensemble type | `BaggingClassifier` | `BaggingClassifier` | Custom spectral loop | `PolychromaticForest` |
| Diversity axes | Bootstrap | Bootstrap | Bootstrap + spectral | Bootstrap + spectral + feature |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- NVIDIA CUDA-compatible GPU (T4 or equivalent, >= 8 GB VRAM recommended)
- CUDA Toolkit 12.x
- Conda (recommended for RAPIDS installation)

### Installation

```bash
# Clone the repository
git clone https://github.com/Devanik21/Riemannian-Wave-Geometry.git
cd Riemannian-Wave-Geometry

# Create conda environment
conda create -n rwc-gwl python=3.11 -y
conda activate rwc-gwl

# Install RAPIDS (cuML + CuPy) for CUDA 12.x
pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# Install remaining dependencies
pip install openml scikit-learn scipy numpy matplotlib seaborn

# Launch the notebook
jupyter notebook RWC_GWL_Master.ipynb
```

### CPU Fallback

Replace `import cupy as cp` with `import numpy as cp` and `cuml.neighbors.NearestNeighbors` with `sklearn.neighbors.NearestNeighbors`. All `cp.asnumpy()` calls become no-ops. The sparse eigensolver fallback to `scipy.sparse.linalg.eigsh` is already built into V14's architecture classes. Runtime will be 10–50× slower at full N but functionally identical.

---

## Usage

### V14 Polychromatic Forest (Final Architecture)

```python
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# After running preprocessing to obtain X_processed, y_raw
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for tr_idx, te_idx in sss.split(X_processed, y_raw):
    X_tr, X_te = X_processed[tr_idx], X_processed[te_idx]
    y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]

# Three V14 architecture variants
scwh_forest  = PolychromaticForest(SCWH_Classifier,  n_estimators=15, n_components=128)
aqgl_forest  = PolychromaticForest(AQGL_Classifier,  n_estimators=15, n_components=128, gravity_alpha=2.5)
mfthrf_forest = PolychromaticForest(MFTHRF_Classifier, n_estimators=15, n_frequencies=50)

for name, model in [("SCWH", scwh_forest), ("AQGL", aqgl_forest), ("MFT-HRF", mfthrf_forest)]:
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"{name}: {acc*100:.2f}%")
```

### V13 Polychromatic Baseline

```python
rwc_forest = RWCEnsemble(n_estimators=15, max_samples=0.75)
gwl_forest = GWLEnsemble(n_estimators=15, max_samples=0.75)

rwc_forest.fit(X_tr, y_tr)
gwl_forest.fit(X_tr, y_tr)

print(f"RWC Polychromatic: {accuracy_score(y_te, rwc_forest.predict(X_te))*100:.2f}%")
print(f"GWL Polychromatic: {accuracy_score(y_te, gwl_forest.predict(X_te))*100:.2f}%")
```

### Confusion Matrix Visualization

```python
# plot_dark_confusion_matrix is defined in Cell 3 of the notebook
plot_dark_confusion_matrix(y_te, model.predict(X_te), title="V14 SCWH — Final Benchmark")
```

---

## Requirements

```
# Core (CPU mode fallback supported)
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
openml>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0

# GPU acceleration (CUDA 12.x)
cupy-cuda12x>=13.0.0
cuml-cu12>=24.0.0
cudf-cu12>=24.0.0
```

---

## Authors

**Devanik Debnath** — *Manifold architecture, Ricci flow design, HRF kernel, SCWH/AQGL/MFTHRF design, V14 sparse pipeline, polychromatic ensemble, GPU optimization*
B.Tech, Electronics & Communication Engineering
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

**Xylia** — *The Artificially Intelligent Squad*

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

You are free to use, modify, and distribute this software for any purpose — commercial or non-commercial — with or without modification, subject to the conditions of the Apache 2.0 License. Attribution to the original authors is required in derivative works.

---

*This work demonstrates that the language of differential geometry — curvature, flow, spectral harmonics, resonance, holographic phase alignment — is not metaphor but a precise, implementable, and empirically powerful framework for structured classification on real-world sensor data.*
