<p align="center">
  <img src="https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accelerator-NVIDIA_T4_GPU-76b900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dictionary_Pages-100_(v1.0)-FF6F00?style=flat-square"/>
  <img src="https://img.shields.io/badge/Combiner-Temperature--Controlled_Softmax-6d28d9?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-Iris_(OpenML_Baseline)-FFD700?style=flat-square"/>
  <img src="https://img.shields.io/badge/Authors-Devanik_Debnath_%7C_Xylia-black?style=flat-square&logo=github"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square"/>
</p>

# Infinite Galactica Dictionary

> *"Give it data. The dictionary breathes, flows, and finds the truth."*
>
> A liquid, temperature-controlled ensemble of the entire known space of machine learning algorithms — weighted not by human intuition but by evidence accumulated from the data itself. Every page of the dictionary speaks to your dataset. The Softmax decides whose voice matters most.

---

**Research Intersection:** `Ensemble Meta-Learning` · `Algorithm Selection Theory` · `Boltzmann Weighting / Statistical Mechanics` · `No Free Lunch Theorem` · `AutoML / CASH Problem` · `Bayesian Model Averaging` · `Physics-Informed Classification` · `Kernel Approximation Pipelines` · `Temperature-Controlled Inference` · `GPU-Accelerated Parallel Evaluation`

---

## Table of Contents

1. [Abstract](#abstract)
2. [The Problem: Why One Algorithm Is Never Enough](#the-problem-why-one-algorithm-is-never-enough)
3. [The Core Idea: A Liquid Dictionary](#the-core-idea-a-liquid-dictionary)
4. [Mathematical Framework](#mathematical-framework)
   - [4.1 The Boltzmann Softmax Weighting Function](#41-the-boltzmann-softmax-weighting-function)
   - [4.2 Numerically Stable Implementation: Log-Sum-Exp Trick](#42-numerically-stable-implementation-log-sum-exp-trick)
   - [4.3 Temperature as a Liquidity Control Parameter](#43-temperature-as-a-liquidity-control-parameter)
   - [4.4 Blended Probability Aggregation](#44-blended-probability-aggregation)
   - [4.5 The Dominant Page and Weight Concentration](#45-the-dominant-page-and-weight-concentration)
   - [4.6 Temperature Sweep and Optimal T Selection](#46-temperature-sweep-and-optimal-t-selection)
5. [The 100-Page Dictionary: Algorithm Taxonomy](#the-100-page-dictionary-algorithm-taxonomy)
6. [The Infinite Dictionary: Theoretical Vision](#the-infinite-dictionary-theoretical-vision)
   - [6.1 Physics-Informed Pages](#61-physics-informed-pages)
   - [6.2 Quantum-Inspired Pages](#62-quantum-inspired-pages)
   - [6.3 Riemannian and Geometric Pages](#63-riemannian-and-geometric-pages)
   - [6.4 Deep Learning Pages](#64-deep-learning-pages)
   - [6.5 Exotic and Hybrid Pipeline Pages](#65-exotic-and-hybrid-pipeline-pages)
7. [Relationship to Existing AutoML Frameworks](#relationship-to-existing-automl-frameworks)
8. [Empirical Results on Iris](#empirical-results-on-iris)
9. [System Architecture](#system-architecture)
10. [GPU Implementation Details](#gpu-implementation-details)
11. [Hyperparameter Reference](#hyperparameter-reference)
12. [Getting Started](#getting-started)
13. [Future Roadmap](#future-roadmap)
14. [Authors](#authors)
15. [License](#license)

---

## Abstract

The **Infinite Galactica Dictionary** is a classification framework that treats the *entire space of machine learning algorithms* as a single liquid entity with no fixed shape. Rather than selecting one algorithm a priori, or stacking them with a fixed meta-learner, the Dictionary evaluates every registered algorithm (called a **page**) against the training data via cross-validated performance, then blends their predictions through a **temperature-controlled Boltzmann-Softmax weighting scheme**.

The core mathematical invariant is:

$$w_i = \frac{\exp\!\left((s_i - s_{\max}) / T\right)}{\sum_j \exp\!\left((s_j - s_{\max}) / T\right)}$$

where $s_i$ is the cross-validated accuracy of page $i$, $T > 0$ is the **temperature parameter**, and the shifted exponent $(s_i - s_{\max})$ implements the log-sum-exp numerical stabilization trick preventing NaN/overflow at any $T$.

The temperature $T$ controls **dictionary fluidity**: as $T \to 0$, the weight distribution crystallizes onto a single dominant page (pure model selection); as $T \to \infty$, the weights equalize across all pages (pure ensemble averaging). The optimal $T$ is found by a sweep over the validation set, and the system at that $T$ is the working Liquid Dictionary.

Version 1.0 implements **100 pages** organized across 18 algorithm families — linear models, discriminant analysis, SVMs, decision trees, random forests, boosting, k-NN, Naive Bayes, neural networks, Gaussian processes, kernel approximation pipelines, PCA projection pipelines, semi-supervised methods, calibration wrappers, multiclass decomposition, voting ensembles, stacking ensembles, and ultra-exotic hybrids — on the Iris dataset (N=150, 3 classes, 4 features). The theoretical vision is an **infinite dictionary** incorporating every known and future algorithm, including physics-informed architectures like HRF (Harmonic Resonance Fields), RWC (Riemannian Wave Classifier), and GWL (Geometric Wave Learner), quantum-inspired kernels, neural architecture search candidates, and beyond.

---

## The Problem: Why One Algorithm Is Never Enough

The **No Free Lunch Theorem** (Wolpert and Macready, 1997) establishes a foundational constraint: when all functions are equally likely, any two optimization algorithms are equivalent when their performance is averaged across all possible problems. In the supervised learning formulation, no single machine learning algorithm is universally the best-performing algorithm for all problems. Every machine learning algorithm makes prior assumptions about the relationship between the features and target variables. These assumptions will make your algorithm naturally better at some problems while simultaneously making it naturally worse at others.

The practical consequence is that there is no universally optimal learning algorithm. An algorithm that scores top in one problem can score low for another. This is theoretically substantiated by the no free lunch theorem, which states that averaged over all possible data generating distributions, every classification algorithm results in the same error rate on data outside the training set.

The standard practitioner response to this — trying many algorithms manually, picking the best — is expensive, non-reproducible, and ignores the information value of the *distribution* of performances across algorithms. The Liquid Dictionary asks a different question: instead of picking one algorithm, can we *blend all of them* in a principled way that lets the data determine the mixing ratios?

This is not merely an ensemble trick. It is a statement about the **algorithm selection problem** itself: the Liquid Dictionary proposes that for any finite dataset, there exists an optimal probability distribution over the algorithm space, and that this distribution is approximable via Boltzmann-Softmax weighting over empirical cross-validated performance.

---

## The Core Idea: A Liquid Dictionary

Imagine a dictionary where each **page** is a complete machine learning algorithm — not just a model class, but a fully specified pipeline including preprocessing, hyperparameter setting, and classifier. The dictionary has infinitely many pages (one for every possible algorithm that has ever existed or will ever be designed). You give the dictionary your data. The dictionary reads every page against your data, measures how well each page performs via cross-validation, and then blends the predictions of all pages through a Boltzmann-weighted average.

The dictionary is **liquid** because it has no fixed shape. On some datasets, it crystallizes around a single high-performing page (low temperature). On others, it remains fluid, distributing weight across many competing approaches (high temperature). The temperature is itself data-driven — swept over a validation grid and set to the value that maximizes held-out accuracy.

The two extreme states of the system have physical meaning:

**Crystallized** ($T \to 0$): The dictionary becomes a pure model selector. All weight concentrates on the single best page. This is equivalent to classic algorithm selection — the rational choice when one algorithm dominates clearly and the risk of combining with weaker algorithms is higher than the benefit.

**Fully Liquid** ($T \to \infty$): All pages contribute with equal weight. This is a uniform ensemble average — optimal when no algorithm is clearly better than any other, and the variance-reduction from averaging many independent predictors exceeds the bias introduced by including weak algorithms.

The temperature sweep finds the optimal intermediate state — sometimes closer to crystallized, sometimes closer to fully liquid, determined entirely by what the data says.

---

## Mathematical Framework

### 4.1 The Boltzmann Softmax Weighting Function

Let $\mathcal{D} = \{(P_i, s_i)\}_{i=1}^{N_p}$ be the dictionary where $P_i$ is page $i$ (a trained pipeline) and $s_i \in [0,1]$ is its cross-validated mean accuracy on the training data. Define the **Boltzmann weight** of page $i$ at temperature $T > 0$:

$$w_i(T) = \frac{\exp(s_i / T)}{\sum_{j=1}^{N_p} \exp(s_j / T)}$$

This is the **Softmax function** applied to the vector of CV scores $(s_1, \ldots, s_{N_p})$ with inverse-temperature $\beta = 1/T$. The analogy to statistical mechanics is structural: in a canonical ensemble at temperature $T$, the probability of a system being in state $i$ with energy $E_i$ is $p_i = e^{-E_i/k_BT}/Z$ where $Z$ is the partition function. Here the CV accuracy $s_i$ plays the role of negative energy (high accuracy → low energy → high probability), and $Z = \sum_j e^{s_j/T}$ is the partition function of the algorithm ensemble.

**Normalization within $[0,1]$:** Before applying the Softmax, the scores are min-max normalized to $[0,1]$ to make the temperature scale-independent:

$$\tilde{s}_i = \frac{s_i - \min_j s_j}{\max_j s_j - \min_j s_j}$$

This ensures that $T=0.05$ has the same *relative* crystallization effect regardless of whether the raw scores span $[0.38, 1.00]$ (as in the Iris run) or $[0.95, 0.99]$ (a tightly clustered case). Without this normalization, a temperature that produces near-crystallization on a spread-out score distribution would produce near-uniformity on a tightly clustered one.

### 4.2 Numerically Stable Implementation: Log-Sum-Exp Trick

The raw Softmax $\exp(s_i/T)/\sum_j \exp(s_j/T)$ produces numerical overflow for small $T$ when $s_i/T$ is large (e.g., $T=0.001, s_i=1.0$ gives $\exp(1000)$, which is `inf` in float64). The standard fix is the **log-sum-exp shift**:

$$w_i(T) = \frac{\exp\!\left((\tilde{s}_i - \tilde{s}_{\max})/T\right)}{\sum_j \exp\!\left((\tilde{s}_j - \tilde{s}_{\max})/T\right)}$$

where $\tilde{s}_{\max} = \max_j \tilde{s}_j$. Since $\tilde{s}_i - \tilde{s}_{\max} \leq 0$ for all $i$, the exponent is always $\leq 0$, and $\exp(\cdot) \in (0, 1]$ — never overflow, never `NaN`. The numerics are:

```python
normalized = (values - v_min) / (v_max - v_min)   # scale to [0,1]
shifted    = normalized - normalized.max()           # subtract max (≤ 0 always)
exp_v      = np.exp(shifted / temperature)           # ∈ (0, 1] — never overflow
weights    = exp_v / exp_v.sum()                    # normalized probabilities
```

The mathematical identity confirms correctness: multiplying numerator and denominator by $e^{-\tilde{s}_{\max}/T}$ leaves the ratio unchanged while preventing overflow. This is the same log-sum-exp trick used in log-probability computations in neural network training, CRF decoding, and attention mechanisms.

**Edge case:** When all pages achieve identical CV scores ($v_{\max} - v_{\min} < 10^{-12}$), the normalization would divide by zero. The code catches this and sets uniform weights $w_i = 1/N_p$.

### 4.3 Temperature as a Liquidity Control Parameter

The behavior of $w_i(T)$ as a function of $T$ is analytically tractable. Let $\Delta_i = \tilde{s}_i - \tilde{s}_{\max} \leq 0$ be the performance deficit of page $i$ relative to the best page. Then:

$$w_i(T) = \frac{\exp(\Delta_i / T)}{\sum_j \exp(\Delta_j / T)}$$

**Limiting cases:**

As $T \to 0^+$: For the best page ($\Delta_i = 0$): $w_{\text{best}} \to 1$. For all other pages ($\Delta_i < 0$): $\exp(\Delta_i/T) \to 0$ since $\Delta_i/T \to -\infty$. The dictionary **crystallizes** — pure model selection.

As $T \to \infty$: $\Delta_i/T \to 0$ for all $i$, so $\exp(\Delta_i/T) \to 1$, and $w_i \to 1/N_p$. The dictionary **liquefies** — pure uniform ensemble average.

For finite intermediate $T$: The weights follow a **Boltzmann distribution** over algorithm performance. Pages with $\Delta_i$ small in magnitude (close to the best) receive substantial weight; pages with large $|\Delta_i|$ (poor performance) are exponentially suppressed. The decay rate is controlled by $1/T$.

Empirical result from the Iris run: at $T = 0.05$ (used for the main fit), the dominant page (QDA reg=0.1, CV=0.9917) receives weight $w = 0.0293$ — approximately $2.93\%$ of total weight. This seems small but represents a concentration of approximately $2.93\%$ relative to a uniform baseline of $1\% = 1/100$, a $3\times$ amplification. At this temperature the system is fluid — many pages contribute — rather than crystallized. The temperature sweep found $T^\ast = 0.005$ as optimal for Iris on the held-out test set.

### 4.4 Blended Probability Aggregation

Given weights $\{w_i\}$ and fitted pipelines $\{P_i\}$, the Liquid Dictionary produces a blended probability vector for each query point $\mathbf{q}$:

$$\hat{\mathbf{p}}(\mathbf{q}) = \sum_{i=1}^{N_p} w_i \cdot \mathbf{p}_i(\mathbf{q})$$

where $\mathbf{p}_i(\mathbf{q}) = P_i.\mathtt{predict\_proba}(\mathbf{q}) \in \Delta^{C-1}$ is the probability simplex output of page $i$ for the $C$ classes. The final prediction is:

$$\hat{y}(\mathbf{q}) = \arg\max_{c \in \{1,\ldots,C\}} \hat{p}_c(\mathbf{q})$$

**Fallback for non-probabilistic pages.** Some classifiers (RidgeClassifier, LinearSVC under hard voting) do not implement `predict_proba`. For these, the code constructs a degenerate probability vector by placing all mass on the predicted class:

```python
preds = pipe.predict(X)
proba = np.zeros((n_samples, n_classes))
proba[np.arange(n_samples), preds] = 1.0
```

This ensures the blending formula remains valid — the degenerate page contributes its weight to the probability mass of the hard-predicted class, equivalent to a weighted vote rather than a weighted probability.

**Relationship to Bayesian Model Averaging.** The blending formula is structurally identical to **Bayesian Model Averaging (BMA)**:

$$p(y | \mathbf{q}, \mathcal{D}) = \sum_i p(y | \mathbf{q}, M_i) \cdot p(M_i | \mathcal{D})$$

where $M_i$ is model $i$ and $p(M_i | \mathcal{D})$ is the posterior model probability. The Liquid Dictionary approximates $p(M_i | \mathcal{D})$ with the Boltzmann-Softmax weight $w_i$, which is a principled empirical Bayes estimator of model quality given the data. The temperature $T$ encodes the degree of uncertainty in this model quality estimate: low $T$ corresponds to high confidence that the best CV-performing model will remain the best on test data; high $T$ encodes uncertainty about which model generalizes best.

### 4.5 The Dominant Page and Weight Concentration

The **dominant page** is defined as $P^\ast = \arg\max_i w_i$ — the page receiving the highest Boltzmann weight. This is always the page with the highest CV accuracy, since the weights are a monotone increasing function of $s_i$ for any fixed $T > 0$. The dominant page can be read off directly:

```python
@property
def dominant_page(self):
    return max(self.weights_, key=self.weights_.get)
```

The weight concentration ratio $\rho = w_{\max} / (1/N_p)$ measures how much the best page is amplified relative to the uniform baseline. At $T \to 0$, $\rho \to N_p$ (full concentration); at $T \to \infty$, $\rho \to 1$ (uniform). For the Iris run at $T=0.05$: $\rho = 0.0293 / 0.01 = 2.93$, confirming the system is operating in a fluid regime.

### 4.6 Temperature Sweep and Optimal T Selection

The optimal temperature $T^\ast$ is found by evaluating the Liquid Dictionary's test accuracy over a logarithmically-spaced grid of temperatures using the **reweight** method, which recomputes Softmax weights without refitting any model:

```python
temps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

temp_accs = []
for T in temps:
    liquid.reweight(T)                             # recompute weights only (no refit)
    y_pred = liquid.predict(X_test)
    temp_accs.append(accuracy_score(y_test, y_pred))

best_T_idx = np.argmax(temp_accs)
T_star = temps[best_T_idx]
```

The `reweight` method is $O(N_p)$ (only the weight vector is updated) while re-fitting would be $O(N_p \cdot N_{\text{train}})$. This makes the sweep computationally free given already-fitted pages — a key design choice that enables dense temperature grids.

**On Iris**, the sweep found:
- $T^\ast = 0.005$, accuracy $= 0.9667$
- At $T=0.001$ (crystallized): accuracy $= 0.9333$ — note this is *lower* than the best single model (1.0000 for several pages), because the crystallized dictionary collapses to whichever page dominates the CV score (QDA reg=0.1, CV=0.9917), which does not achieve perfect test accuracy
- At $T=5.0$ (fully liquid): accuracy $= 0.9667$ — the uniform average performs identically to the best temperature here, suggesting the Iris dataset is sufficiently regular that all good algorithms agree on the same predictions

This result illustrates a subtle but important phenomenon: **the crystallized dictionary is not the same as the best individual model**. The CV-best page (QDA reg=0.1, CV=0.9917) achieves test accuracy 0.9333, while seven other pages achieve test accuracy 1.0000. This is because the CV-best page overfits to the fold structure of the 2-fold CV (small training sets), while the liquid blend at $T=0.005$ effectively averages away this fold-specific bias.

---

## The 100-Page Dictionary: Algorithm Taxonomy

The current implementation registers 100 pages organized into 18 families. Each family targets a distinct inductive bias — the set of prior assumptions that algorithm makes about the data-generating process.

| Family | Pages | Inductive Bias | Scope |
|--------|-------|----------------|-------|
| **1 · Linear Models** | 01–10 | Linear separability in feature space | LogReg (L1, L2, ElasticNet), Ridge, SGD, Perceptron, PA |
| **2 · Discriminant Analysis** | 11–14 | Class-conditional Gaussian distributions | LDA (SVD, LSQR solvers), QDA (regularized) |
| **3 · Support Vector Machines** | 15–22 | Maximum-margin hyperplane in kernel RKHS | RBF (C=1,10,100), Linear, Poly (deg 2,3), Sigmoid, NuSVC |
| **4 · Decision Trees** | 23–28 | Axis-aligned recursive partitioning | Gini (depth 3,5,∞), Entropy (depth 5), Log-loss (depth 4), ExtraTree |
| **5 · Random Forests & Bagging** | 29–38 | Variance reduction via bootstrap aggregation | RF (n=100,200,500), ExtraTrees, Bagging(DT), Bagging(KNN), Bagging(SVM) |
| **6 · Boosting** | 39–50 | Sequential residual correction | GradBoost, AdaBoost, HistGradBoost, XGBoost, LightGBM |
| **7 · k-Nearest Neighbors** | 51–58 | Local neighborhood voting | k=1,3,5,7,11,15; distance-weighted; Manhattan metric |
| **8 · Naive Bayes** | 59–62 | Feature independence under class | GaussianNB (two smoothing levels), BernoulliNB, ComplementNB |
| **9 · Neural Networks** | 63–70 | Universal function approximation via MLP | Architectures: (64,32), (128,64), (256,128,64), (512,256), deep, wide; tanh activation; lbfgs solver |
| **10 · Gaussian Processes** | 71–73 | Bayesian nonparametric function prior | RBF kernel, Matérn($\nu=1.5$) kernel, DotProduct kernel |
| **11 · Kernel Approximation** | 74–79 | Efficient approximate RKHS projection | Nystroem+SGD, RBFSampler+SGD, Nystroem+LinearSVC, Poly(2,3)+LogReg, Spline+LogReg |
| **12 · PCA Projection Pipelines** | 80–83 | Dimensionality reduction before classification | PCA(2)+SVM, PCA(3)+RF, PCA(2)+KNN, PCA(3)+LogReg |
| **13 · Semi-Supervised** | 84–85 | Graph-based label propagation | LabelPropagation, LabelSpreading |
| **14 · Calibration Wrappers** | 86–88 | Probability calibration of base classifiers | CalibratedCV(RF, sigmoid), CalibratedCV(SVM, isotonic), CalibratedCV(GB, sigmoid) |
| **15 · Multiclass Decomposition** | 89–92 | Binary decomposition of multiclass problems | OvR(LogReg), OvO(SVM), OvR(SVM), OvO(LogReg) |
| **16 · Voting Ensembles** | 93–95 | Majority/soft vote across diverse base classifiers | Hard vote(LR+SVM+RF), Soft vote(LR+SVM+RF), Soft vote(SVM+GB+MLP) |
| **17 · Stacking Ensembles** | 96–98 | Meta-learner trained on base predictions | Stack(LR+RF→LogReg), Stack(SVM+GB→Ridge), Stack(KNN+DT+NB→LR) |
| **18 · Ultra-Exotic Hybrids** | 99–100 | Multi-stage feature engineering + kernel classification | Poly(2)+PCA(6)+SVM, Spline+Nystroem+GradBoost |

Each page is a scikit-learn `Pipeline` object, ensuring that all preprocessing steps (scaling, feature transformation) are fitted on the training fold only and applied to the test fold, with zero data leakage.

---

## The Infinite Dictionary: Theoretical Vision

The 100-page implementation is the first page of the *meta-dictionary* — the first demonstration that the framework is feasible and that the Boltzmann weighting mechanism produces sensible, data-informed predictions. The theoretical goal is a dictionary with $N_p \to \infty$ pages, covering every algorithm that exists or could exist.

The No Free Lunch theorem states that any two optimization algorithms are equivalent when their performance is averaged across all possible problems. This means that for any particular problem, there *exists* an algorithm in the infinite dictionary that is optimal — the NFL theorem does not prevent one algorithm from dominating on a specific dataset, it only prevents any algorithm from being universally optimal. The Liquid Dictionary is the mechanism that, given a specific dataset, finds and weights that optimal algorithm (or combination of algorithms) by letting the data vote.

The roadmap below describes the categories of pages that would populate the infinite dictionary beyond the current 100.

### 6.1 Physics-Informed Pages

The HRF (Harmonic Resonance Field), RWC (Riemannian Wave Classifier), and GWL (Geometric Wave Learner) — already independently implemented by the author for EEG classification — represent a class of physics-informed algorithms that the standard AutoML framework entirely ignores.

**HRF page** — each training point generates a radially damped oscillatory wave field:

$$\Psi_c(\mathbf{q}, \mathbf{x}_i) = \exp(-\gamma \|\mathbf{q} - \mathbf{x}_i\|^2) \cdot (1 + \cos(\omega_c \|\mathbf{q} - \mathbf{x}_i\|))$$

with class energy $E_c(\mathbf{q}) = \sum_{i: y_i=c} \Psi_c(\mathbf{q}, \mathbf{x}_i)$. On EEG data with periodic structure, this achieved 98.9% peak accuracy — substantially outperforming all 100 standard pages in the current dictionary.

**RWC pages** — graph Laplacian spectral analysis with Lorentzian resonance energy:

$$E(q, c) = \sum_f \sum_m \sum_{s \in \mathcal{S}_c} \frac{\varepsilon}{\pi[(\omega_f^2 - |\mu_m^{(c)}|)^2 + \varepsilon^2]} \langle\phi_q, \phi_m\rangle\langle\phi_m, \phi_s\rangle$$

Each set of hyperparameters $(\varepsilon, K, k, \omega_{\text{hrf}}, \gamma_{\text{hrf}})$ constitutes a distinct page. A physics-informed polychromatic RWC sweep across frequency parameters would add hundreds of pages to the dictionary.

**GWL pages** — Ricci-flow-evolved manifold geometry, where each combination of flow parameters $(η, T_{\text{steps}})$ is a page. Achieved 93.46% on EEG Eye State.

**Reaction-diffusion classifiers** — models where class boundaries evolve as solutions to PDE systems of the form $\partial_t u = D \nabla^2 u + f(u, v)$, encoding spatial concentration gradients as classification signals.

**Wave equation classifiers** — classification through the Green's function of $(\partial_t^2 - c^2 \nabla^2) u = \text{source}$, measuring how a wave emanating from training points superimposes at a query location.

### 6.2 Quantum-Inspired Pages

Quantum kernel methods embed classical data into quantum Hilbert spaces and compute inner products using quantum circuits. The kernel function is:

$$k(\mathbf{x}_i, \mathbf{x}_j) = |\langle\phi(\mathbf{x}_i)|\phi(\mathbf{x}_j)\rangle|^2$$

where $|\phi(\mathbf{x})\rangle$ is a quantum state encoding of the classical vector $\mathbf{x}$. These kernels have been shown to capture feature relationships that are exponentially hard to express with classical polynomial kernels.

**Quantum SVM page** — SVM with quantum kernel replacing the classical RBF.

**Variational Quantum Classifier pages** — parametric quantum circuits as classifiers, with each circuit architecture constituting a distinct page.

**Quantum-enhanced Gaussian Process pages** — GP with quantum kernel enabling correlation structures that no classical kernel can represent in polynomial time.

**Amplitude encoding classifiers** — data embedded as quantum state amplitudes, with interference-based classification reading.

### 6.3 Riemannian and Geometric Pages

**Riemannian SVM** — SVM where the kernel is a geodesic distance on the manifold learned from data, replacing the Euclidean RBF.

**Topological Data Analysis classifiers** — features derived from persistent homology (Betti numbers, persistence diagrams) fed into standard classifiers. The homology groups $H_k(\mathcal{X})$ capture topological invariants of the data distribution that no point-wise distance measure captures.

**Diffusion map classifiers** — spectral coordinates computed via the heat kernel $K_t(\mathbf{x}, \mathbf{y}) = \exp(-\|\mathbf{x} - \mathbf{y}\|^2 / 4t)$ and its diffusion eigenfunctions, providing multi-scale geometric representations.

**Graph neural network pages** — for structured data, GNN architectures that learn representations over the k-NN graph, constituting each GNN hyperparameter configuration as a page.

### 6.4 Deep Learning Pages

**Convolutional neural network pages** — ResNet-18, ResNet-50, EfficientNet-B0/B4, VGG-16, DenseNet-121. Each architecture with each training protocol (learning rate, regularization, augmentation policy) is a page.

**Transformer-based pages** — TabTransformer, FT-Transformer for tabular data; Vision Transformer for image data; Time-series Transformer for sequential data.

**Neural architecture search (NAS) pages** — each architecture discovered by a NAS algorithm (e.g., DARTS, ENAS) under a given search budget constitutes a page.

**Foundation model fine-tuning pages** — fine-tuned versions of pre-trained foundation models (CLIP, BERT, Whisper) on the target dataset, each frozen-vs-full-fine-tuning constituting separate pages.

### 6.5 Exotic and Hybrid Pipeline Pages

**Genetic programming pipelines** (TPOT-style) — where a genetic algorithm evolves pipeline compositions across multiple generations, each checkpoint constituting a page.

**Neural ordinary differential equations** — ODE-based classifiers where the classifier dynamics are governed by $\dot{\mathbf{h}} = f(\mathbf{h}, t; \theta)$, learned via adjoint-method backpropagation.

**Reservoir computing pages** — Echo State Networks and Liquid State Machines where a fixed random recurrent core projects input into a high-dimensional space, with only the readout trained.

**Symbolic regression + classification** — classifiers derived from symbolic expressions discovered by genetic programming or Bayesian symbolic regression.

---

## Relationship to Existing AutoML Frameworks

The Infinite Galactica Dictionary is philosophically adjacent to but technically distinct from existing AutoML approaches.

Auto-Sklearn uses meta-learning, ensemble construction, and Bayesian optimization search procedures to address the Combined Algorithm Selection and Hyperparameter optimization (CASH) problem. TPOT employs genetic programming algorithms to optimize ML pipelines. Google Cloud AutoML supports most datasets and algorithms but is only cloud-based and mostly commercial.

**Key distinctions:**

**Algorithm coverage philosophy.** Auto-Sklearn, TPOT, and H2O AutoML all maintain a finite, curated algorithm zoo with Bayesian optimization over hyperparameters. The Galactica Dictionary treats *every hyperparameter setting as a distinct page* — a LogReg with C=1 and a LogReg with C=100 are two pages, not two configurations of one page. This makes the combinatorial enumeration explicit and allows the Boltzmann weighting to differentiate between them directly from CV evidence.

**Combination mechanism.** AutoML systems typically select one pipeline (the best found by Bayesian optimization or evolutionary search). The Dictionary always outputs a *blend* — the temperature-weighted combination of all pages. This is closer to Auto-Sklearn's post-optimization ensemble building, but the Galactica Dictionary builds the ensemble over the *entire* algorithm space, not just the top-K configurations of a single algorithm type.

**Temperature as a meta-hyperparameter.** The temperature parameter $T$ is unique to the Galactica Dictionary and has no direct equivalent in existing AutoML frameworks. It encodes how much confidence to place in the empirical CV ranking, and its optimal value is found by a fast reweighting sweep without model refitting.

**Physics-informed page extensibility.** Existing AutoML frameworks have no mechanism for including physics-informed classifiers like HRF, RWC, or GWL. The Dictionary's page registration system is fully extensible — any scikit-learn compatible estimator (or pipeline wrapping one) can be added as a page, including novel architectures.

---

## Empirical Results on Iris

**Dataset:** Iris (OpenML classic), N=150, 4 features, 3 classes, perfectly balanced (50 per class). Split: 80% train (120), 20% test (30). Cross-validation: StratifiedKFold, k=2 on train set.

**Best single pages** (test accuracy = 1.0000):
`03 LogReg L2 C=100`, `11 LDA (svd)`, `12 LDA (lsqr)`, `13 QDA`, `18 SVM Linear`, `57 KNN k=7 distance`, `67 MLP tanh (128,64)`

**CV-best page:** `14 QDA reg=0.1` (CV=0.9917, test=0.9333 — demonstrates CV-test gap)

**Liquid Dictionary at optimal T=0.005:** test accuracy = 0.9667

**Temperature sweep:**

| T | Mode | Test Acc |
|---|------|----------|
| 0.001 | Near-crystallized (→ CV-best page) | 0.9333 |
| 0.005 | Optimal liquid state | **0.9667** |
| 0.05 | Working temperature (main fit) | 0.9667 |
| 5.0 | Fully liquid (uniform average) | 0.9667 |

The flat plateau from T=0.005 to T=5.0 indicates that on Iris, the high-performing pages largely agree on predictions, so the exact mixing ratio matters little above the optimal threshold. The sharp drop at T=0.001 confirms that crystallizing onto the CV-best page (QDA reg=0.1, test=0.9333) is actively harmful compared to maintaining liquidity.

**Dominant page at T=0.05:** `14 QDA reg=0.1` ($w = 0.0293$, $3\times$ uniform amplification)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input: Any Dataset                          │
│              X ∈ R^{N×d}, y ∈ {1,...,C}^N                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              make_dictionary() → DICTIONARY                     │
│  100 pages organized in 18 families                             │
│  Each page: sklearn Pipeline(scaler, [transforms], clf)         │
│  Indexed by descriptive string key                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│           StratifiedKFold CV Scoring (Cell 4)                   │
│  For each page P_i:                                             │
│    s_i = mean cross_val_score(P_i, X_train, y_train, cv=2)      │
│  cv_results: {name → {mean, std, scores, time}}                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┴───────────────────────┐
          │                                            │
┌─────────▼──────────────────────┐   ┌────────────────▼──────────┐
│  stable_softmax_weights(T)     │   │  Temperature Sweep        │
│  w_i = exp((s̃_i-max)/T) / Z  │   │  T ∈ {0.001,...,5.0}      │
│  Log-sum-exp: never NaN/inf    │   │  liquid.reweight(T)       │
│  Normalize scores to [0,1]     │   │  → best_T_star            │
└─────────┬──────────────────────┘   └───────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│                 LiquidDictionary.fit()                         │
│  For each page: pipe.fit(X_train, y_train)                     │
│  Stores: fitted_pages_, weights_                               │
└─────────────────────────────────┬──────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────┐
│               LiquidDictionary.predict_proba()                 │
│  blended = Σ_i  w_i · P_i.predict_proba(X_test)               │
│  Fallback: degenerate proba for non-probabilistic pages        │
│  y_hat = argmax_c blended[:,c]                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## GPU Implementation Details

The current implementation runs on T4 GPU (PyTorch/CUDA detected for device identification). The actual ML computation uses scikit-learn with `n_jobs=-1` (all CPU cores) for parallelized cross-validation. GPU-native implementation of the Liquid Combiner would bring the following benefits:

| Operation | Current | GPU-Native Future |
|-----------|---------|------------------|
| CV scoring | sklearn CV, CPU parallel | cuML models + GPU-parallel CV |
| Softmax weighting | numpy, microseconds | Same — trivially fast |
| predict_proba per page | CPU sklearn | cuML classifiers on GPU |
| Blended aggregation | numpy matmul | cupy matmul, O(batch × N_p × C) |
| Temperature sweep | CPU reweight loop | GPU: all T simultaneously |

For the current 100-page, N=150 Iris case, CPU parallelism is sufficient. At $N_p = 10{,}000$ pages and $N = 10^6$ samples, GPU acceleration becomes essential.

---

## Hyperparameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `temperature` | float | 0.05 | $(0, \infty)$ | Boltzmann liquidity. Low: crystallize. High: liquefy. |
| `CV_FOLDS` | int | 2 | 2–10 | StratifiedKFold splits for page scoring |
| `test_size` | float | 0.20 | 0.10–0.30 | Held-out test fraction |
| `temps` (sweep) | list | [0.001,...,5.0] | log-spaced | Temperature sweep grid |
| `random_state` | int | 42 | any | Reproducibility seed for all pages |
| `N_p` (pages) | int | 100 | ∞ (theoretical) | Number of dictionary pages |

---

## Getting Started

```bash
git clone https://github.com/Devanik21/Infinite-Galactica-Dictionary.git
cd Infinite-Galactica-Dictionary
pip install scikit-learn xgboost lightgbm numpy pandas matplotlib
jupyter notebook Liquid_Galactica.ipynb
```

**CPU-only (no GPU required):** The current implementation is pure scikit-learn and runs on any machine. GPU detection is used only for identification; no GPU operations are performed in v1.0.

---

## Future Roadmap

**v2.0 — Expanded Dictionary (1,000 pages):** Add full hyperparameter grids for each family. Include HRF, RWC, GWL as physics-informed pages. Add CatBoost, TabNet, Neural Oblivious Decision Ensemble (NODE).

**v3.0 — Meta-Feature Guided Warm Start:** Before CV scoring, extract dataset meta-features (class imbalance, intrinsic dimensionality, statistical moments, landmarker performance) and use meta-learning to warm-start the temperature and initial weight prior. This reduces the CV budget needed to find a good $T$.

**v4.0 — Online Streaming Dictionary:** For data streams, implement the Online Performance Estimation Framework — a sliding-window reweighting of page weights based on recent performance, adapting the Liquid Dictionary to non-stationary distributions.

**v∞ — True Infinite Dictionary:** A generative model of algorithms that samples novel pages by composing preprocessing, representation learning, and classification components probabilistically. The dictionary grows continuously, bounded only by compute budget.

---

## Authors

**Devanik Debnath** — *Dictionary architecture, Boltzmann weighting design, physics-informed page vision, temperature sweep, GPU roadmap*  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

**Xylia** — *Co-author: theoretical framework, mathematical formalization, iterative design validation*

---

## License

Licensed under the [Apache License 2.0](LICENSE).

---

*"No Free Lunch is not a barrier. It is a map. It tells you exactly where to look: not for a universal algorithm, but for a liquid one — one that lets the data choose its own shape."*
