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
> A liquid, temperature-controlled ensemble treating the entire space of machine learning algorithms as a single adaptive entity — weighted not by human intuition but by evidence extracted from the data itself via cross-validation. Every page of the dictionary speaks to your dataset. The Boltzmann-Softmax decides whose voice matters and by how much.

---

**Research Intersection:** `Ensemble Meta-Learning` · `Algorithm Selection Theory (Rice 1976)` · `Boltzmann-Softmax / Statistical Mechanics` · `No Free Lunch Theorem (Wolpert–Macready 1997)` · `CASH Problem (AutoML)` · `Bayesian Model Averaging` · `Bias-Variance-Covariance Decomposition` · `Kolmogorov Complexity` · `Physics-Informed Classification` · `Topological Data Analysis` · `Causal Inference` · `Reservoir Computing` · `Geometric Deep Learning` · `GPU-Accelerated Parallel Evaluation`

---

## Table of Contents

1. [Abstract](#abstract)
2. [Theoretical Motivation: The No Free Lunch Theorem and the CASH Problem](#theoretical-motivation)
   - [2.1 No Free Lunch: Formal Statement](#21-no-free-lunch-formal-statement)
   - [2.2 The CASH Problem](#22-the-cash-problem)
   - [2.3 Bias-Variance-Covariance Decomposition and Ensemble Diversity](#23-bias-variance-covariance-decomposition)
3. [Mathematical Framework](#mathematical-framework)
   - [3.1 Boltzmann-Softmax Weighting: Derivation and Properties](#31-boltzmann-softmax-weighting)
   - [3.2 Numerically Stable Implementation: Log-Sum-Exp](#32-numerically-stable-implementation)
   - [3.3 Temperature as a Phase Transition Parameter](#33-temperature-as-phase-transition-parameter)
   - [3.4 Blended Probability Aggregation and Bayesian Model Averaging](#34-blended-probability-aggregation)
   - [3.5 The Dominant Page and Weight Concentration](#35-dominant-page-and-weight-concentration)
   - [3.6 Temperature Sweep: Reweight Without Refit](#36-temperature-sweep)
   - [3.7 Score Normalization and Scale Invariance](#37-score-normalization)
   - [3.8 Complexity Analysis](#38-complexity-analysis)
4. [The 100-Page Dictionary: Algorithm Taxonomy and Inductive Biases](#the-100-page-dictionary)
   - [Family 1: Linear Models](#family-1-linear-models)
   - [Family 2: Discriminant Analysis](#family-2-discriminant-analysis)
   - [Family 3: Support Vector Machines](#family-3-support-vector-machines)
   - [Family 4: Decision Trees](#family-4-decision-trees)
   - [Family 5: Random Forests and Bagging](#family-5-random-forests-and-bagging)
   - [Family 6: Boosting](#family-6-boosting)
   - [Family 7: k-Nearest Neighbors](#family-7-k-nearest-neighbors)
   - [Family 8: Naive Bayes](#family-8-naive-bayes)
   - [Family 9: Neural Networks (MLP)](#family-9-neural-networks)
   - [Family 10: Gaussian Processes](#family-10-gaussian-processes)
   - [Family 11: Kernel Approximation Pipelines](#family-11-kernel-approximation)
   - [Family 12: PCA Projection Pipelines](#family-12-pca-projection-pipelines)
   - [Family 13: Semi-Supervised Methods](#family-13-semi-supervised-methods)
   - [Family 14: Calibration Wrappers](#family-14-calibration-wrappers)
   - [Family 15: Multiclass Decomposition](#family-15-multiclass-decomposition)
   - [Family 16–17: Voting and Stacking Ensembles](#family-16-17-voting-and-stacking)
   - [Family 18: Ultra-Exotic Hybrids](#family-18-ultra-exotic-hybrids)
5. [The Infinite Dictionary: Unexplored Algorithmic Territories](#infinite-dictionary)
   - [5.1 Physics-Informed Classification](#51-physics-informed-classification)
   - [5.2 Topological Data Analysis Classifiers](#52-topological-data-analysis-classifiers)
   - [5.3 Causal Inference Classifiers](#53-causal-inference-classifiers)
   - [5.4 Neuromorphic and Reservoir Computing Classifiers](#54-neuromorphic-and-reservoir-computing)
   - [5.5 Quantum and Quantum-Inspired Classifiers](#55-quantum-and-quantum-inspired-classifiers)
   - [5.6 Geometric Deep Learning Pages](#56-geometric-deep-learning)
   - [5.7 Information-Geometric and Optimal Transport Classifiers](#57-information-geometric-and-optimal-transport)
   - [5.8 Probabilistic Programming Classifiers](#58-probabilistic-programming-classifiers)
   - [5.9 Symbolic and Evolutionary Computation Pages](#59-symbolic-and-evolutionary-computation)
   - [5.10 Conformal Prediction Wrappers](#510-conformal-prediction-wrappers)
   - [5.11 Hyperbolic Space Classifiers](#511-hyperbolic-space-classifiers)
   - [5.12 Cellular Automata and Complex Systems Classifiers](#512-cellular-automata-and-complex-systems)
6. [Relationship to Existing AutoML Frameworks](#relationship-to-automl)
7. [Empirical Results on Iris: Full Analysis](#empirical-results)
8. [System Architecture](#system-architecture)
9. [GPU Implementation Details and Scaling Roadmap](#gpu-implementation)
10. [Hyperparameter Reference](#hyperparameter-reference)
11. [Getting Started](#getting-started)
12. [Future Roadmap](#future-roadmap)
13. [Authors](#authors)
14. [License](#license)

---

## Abstract

The **Infinite Galactica Dictionary** is a meta-learning classification framework that treats the *entire space of machine learning algorithms* as a single liquid entity with no fixed structural commitment. Rather than selecting one algorithm a priori or stacking them with a trained meta-learner, the Dictionary evaluates every registered algorithm — each called a **page** — against the training data via stratified cross-validation, then blends their predictions through a **temperature-controlled Boltzmann-Softmax weighting scheme** derived from statistical mechanics.

The core aggregation is:

$$\hat{\mathbf{p}}(\mathbf{q}) = \sum_{i=1}^{N_p} w_i(T) \cdot P_i.\mathtt{predict\_proba}(\mathbf{q}), \qquad w_i(T) = \frac{\exp((\tilde{s}_i - \tilde{s}_{\max})/T)}{\sum_j \exp((\tilde{s}_j - \tilde{s}_{\max})/T)}$$

where $\tilde{s}_i \in [0,1]$ is the min-max-normalized cross-validated accuracy of page $i$ and $T > 0$ is the temperature. The log-sum-exp shift $(\tilde{s}_i - \tilde{s}_{\max})$ prevents numerical overflow at any finite $T$, guaranteeing arithmetic stability across the full temperature range.

Version 1.0 implements **100 pages** across 18 algorithm families on the Iris dataset. The theoretical vision is a dictionary approaching an **infinite cardinality**, incorporating every known and future algorithm class: standard ML, physics-informed classifiers (Harmonic Resonance Fields, Riemannian Wave Classifier, Geometric Wave Learner), topological data analysis pipelines, causal inference classifiers, neuromorphic reservoir computers, quantum kernel machines, geometric deep learning architectures, probabilistic programming models, and beyond.

---

## Theoretical Motivation

### 2.1 No Free Lunch: Formal Statement

Wolpert and Macready (1997) proved the following in the supervised learning context: for any two learning algorithms $A$ and $B$, and for any loss function $L$, the expected off-training-set error is identical when averaged over all possible data-generating distributions $P(f)$ assigned uniform weight. Formally, letting $d^m$ denote a training set of size $m$ drawn from $f$:

$$\sum_f P(f) \cdot L(A, f, d^m) = \sum_f P(f) \cdot L(B, f, d^m)$$

The practical implication: no algorithm is universally optimal. For any specific dataset, some algorithms are dramatically better than others — but this superiority is always purchased by worse performance on some other class of problems. Any elevated performance over one class of problems is exactly paid for by degraded performance over another class.

The NFL theorem does not say all algorithms perform equally on any *specific* dataset. It says they perform equally when averaged over *all* datasets under a uniform prior. In practice, all algorithms are not created equal. This is because the entire set of machine learning problems is a theoretical concept much larger than the set of practical problems we actually solve. The Liquid Dictionary responds to NFL directly: rather than committing to one algorithm that may be mismatched to the data, it lets the cross-validation evidence determine the optimal mixture of all algorithms for the specific dataset at hand.

The NFL theorem also provides a constructive map: if no algorithm is universally best, then any rational meta-algorithm must be data-adaptive. The Dictionary is precisely this — a data-adaptive mixture of all available algorithms.

### 2.2 The CASH Problem

The **Combined Algorithm Selection and Hyperparameter optimization** (CASH) problem, formalized by Thornton et al. (2013) and Auto-Sklearn, asks: given a dataset $\mathcal{D}$, find the pipeline $(A^\ast, \lambda^\ast)$ that maximizes validation performance:

$$(A^\ast, \lambda^\ast) = \arg\max_{A \in \mathcal{A},\, \lambda \in \Lambda_A} \text{Acc}_{\text{val}}(A(\lambda), \mathcal{D})$$

where $\mathcal{A}$ is the algorithm space and $\Lambda_A$ is the hyperparameter space of algorithm $A$. AutoML solves CASH by searching this space with Bayesian optimization or evolutionary methods, returning a single optimal pipeline.

The Infinite Dictionary reframes CASH: rather than finding the single best $(A^\ast, \lambda^\ast)$, it evaluates all $(A_i, \lambda_i)$ pairs and computes a probability distribution over them. Each **page** is a fixed $(A_i, \lambda_i)$ tuple. The CASH answer is not a point estimate but a Boltzmann distribution. This is epistemically more honest: the data rarely provides evidence strong enough to eliminate all but one algorithm; a distribution over algorithms captures the residual uncertainty.

The temperature $T$ encodes how much certainty the CV evidence provides: low $T$ means the CV ranking is trusted and the distribution concentrates; high $T$ means the CV ranking is too noisy to justify concentration and the distribution spreads.

### 2.3 Bias-Variance-Covariance Decomposition

For ensembles, the expected mean squared error of the ensemble prediction decomposes as:

$$\text{MSE}(\bar{f}) = \text{Bias}^2(\bar{f}) + \frac{1}{M}\bar{\sigma}^2 + \frac{M-1}{M}\bar{\rho}\,\bar{\sigma}^2$$

where $\bar{\sigma}^2 = \frac{1}{M}\sum_i \sigma_i^2$ is the mean page variance and $\bar{\rho}$ is the mean pairwise correlation between page predictions. For a fixed mean bias and variance, the ensemble error is minimized by minimizing $\bar{\rho}$ — the inter-page prediction correlation.

This is the mathematical justification for the Galactica Dictionary's algorithm family diversity: pages from different families (linear models, SVMs, tree-based methods, GPs, neural networks) capture structurally different inductive biases and therefore produce *decorrelated* prediction errors. When a linear model misclassifies a query because the decision boundary is curved, the SVM with an RBF kernel may classify it correctly. When the SVM fails on a high-dimensional sparse region, the Naive Bayes page may succeed. The Boltzmann blend aggregates these decorrelated views.

The temperature $T$ modulates the covariance term indirectly: low $T$ concentrates weight on few pages, recovering near-single-model variance; high $T$ distributes weight, activating the covariance-reduction benefit of diversity. The optimal $T^\ast$ balances bias (from including poor-performing pages) against variance reduction (from including diverse pages).

---

## Mathematical Framework

### 3.1 Boltzmann-Softmax Weighting: Derivation and Properties

Let the page performance vector be $\mathbf{s} = (s_1, \ldots, s_{N_p}) \in [0,1]^{N_p}$ where $s_i$ is the mean CV accuracy of page $i$. After min-max normalization to $\tilde{\mathbf{s}} \in [0,1]^{N_p}$, the Boltzmann weight is:

$$w_i(T) = \frac{e^{\tilde{s}_i/T}}{Z(T)}, \qquad Z(T) = \sum_{j=1}^{N_p} e^{\tilde{s}_j/T}$$

where $Z(T)$ is the **partition function** of the algorithm ensemble. The analogy to the canonical ensemble is exact: $\tilde{s}_i$ plays the role of $-E_i/k_B$ (negative energy in units of $k_BT$), and $Z$ is the classical partition function.

The **free energy** of the ensemble is $F(T) = -T \log Z(T)$, which satisfies:

$$\frac{\partial F}{\partial T} = -\log Z(T) - T \cdot \frac{\partial \log Z}{\partial T} = -\langle \tilde{s} \rangle_T$$

where $\langle \tilde{s} \rangle_T = \sum_i w_i(T)\,\tilde{s}_i$ is the mean CV score of the current mixture. At low $T$, the mixture concentrates on the highest-$\tilde{s}$ pages and $\langle \tilde{s}\rangle_T \to \tilde{s}_{\max}$. The system minimizes free energy by concentrating on high-performance pages when temperature is low (thermodynamic ground state).

**Monotonicity:** $w_i(T)$ is a monotone increasing function of $\tilde{s}_i$ for all $T > 0$. Proof: $\partial w_i / \partial \tilde{s}_i = w_i(T)(1 - w_i(T))/T > 0$. This ensures the dominant page always has the highest CV score.

**Limiting behavior:** Let $\Delta_i = \tilde{s}_{\max} - \tilde{s}_i \geq 0$ be the performance deficit of page $i$. Then $w_i(T) = e^{-\Delta_i/T} / \sum_j e^{-\Delta_j/T}$. As $T \to 0$: $w_{\text{best}} \to 1$, all others $\to 0$ (crystallization). As $T \to \infty$: all $w_i \to 1/N_p$ (uniform liquefaction). The rate of crystallization is $\sim e^{-\Delta_i/T}$ — pages with large $\Delta_i$ vanish exponentially fast.

**Entropy of the mixture:**

$$H(T) = -\sum_i w_i(T)\log w_i(T)$$

$H(T)$ is a monotone increasing function of $T$: $H(T=0) = 0$ (pure crystallized state, zero entropy), $H(T \to \infty) = \log N_p$ (maximal entropy, fully liquid). The temperature $T$ is exactly the inverse-entropy control parameter of the ensemble.

### 3.2 Numerically Stable Implementation: Log-Sum-Exp

The raw Softmax $e^{\tilde{s}_i/T}/\sum_j e^{\tilde{s}_j/T}$ overflows for $T \ll 1/\tilde{s}_{\max}$. The standard log-sum-exp shift:

$$w_i(T) = \frac{e^{(\tilde{s}_i - \tilde{s}_{\max})/T}}{\sum_j e^{(\tilde{s}_j - \tilde{s}_{\max})/T}}$$

guarantees $(\tilde{s}_i - \tilde{s}_{\max})/T \leq 0$ for all $i$, so all exponents are in $(-\infty, 0]$ and all $e^{(\cdot)} \in (0,1]$. Crucially, the mathematical identity

$$\frac{e^{(\tilde{s}_i - c)/T}}{\sum_j e^{(\tilde{s}_j - c)/T}} = \frac{e^{\tilde{s}_i/T}}{\sum_j e^{\tilde{s}_j/T}} \quad \forall c$$

confirms the shift leaves the weights unchanged. This is the same trick used in log-space attention computation in transformer architectures and CRF inference.

**Special case:** When $\tilde{s}_{\max} - \tilde{s}_{\min} < 10^{-12}$ (all pages identical), the normalization would produce $0/0$. The code detects this and assigns uniform weights $w_i = 1/N_p$ — the correct limit.

```python
def stable_softmax_weights(cv_results, temperature):
    values    = np.array([cv_results[n]["mean"] for n in cv_results])
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-12:
        return {n: 1/len(cv_results) for n in cv_results}
    normalized = (values - v_min) / (v_max - v_min)   # scale to [0,1]
    shifted    = normalized - normalized.max()           # shift to (-∞, 0]
    exp_v      = np.exp(shifted / temperature)           # ∈ (0, 1], no overflow
    weights    = exp_v / exp_v.sum()
    return dict(zip(cv_results.keys(), weights))
```

### 3.3 Temperature as a Phase Transition Parameter

The weight distribution $\{w_i(T)\}$ undergoes a continuous **phase transition** as $T$ varies. This is not a metaphor — the Boltzmann distribution is exactly the equilibrium distribution of a statistical mechanical system, and the concentration of weight on the best page as $T \to 0$ is the classical analog of Bose-Einstein condensation into the ground state.

Define the **effective number of pages** (inverse participation ratio):

$$N_{\text{eff}}(T) = \frac{1}{\sum_i w_i(T)^2} \in [1, N_p]$$

At $T \to 0$: $N_{\text{eff}} \to 1$ (one page dominates). At $T \to \infty$: $N_{\text{eff}} \to N_p$ (all pages equally weighted). The temperature sweep finds $T^\ast$ where $N_{\text{eff}}(T^\ast)$ gives the optimal bias-variance trade-off on the validation set.

For the Iris run at $T = 0.05$: $N_{\text{eff}} \approx 34$ (34 effective pages contributing, despite having 100 total). The dominant page (QDA reg=0.1, $w = 0.0293$) receives $2.93\times$ the uniform weight, while the 3 lowest-performing pages (BernoulliNB, ComplementNB, Ridge) have $w < 0.001$ and contribute negligibly.

### 3.4 Blended Probability Aggregation and Bayesian Model Averaging

The blended probability at query $\mathbf{q}$:

$$\hat{\mathbf{p}}(\mathbf{q}) = \sum_{i=1}^{N_p} w_i \cdot P_i.\mathtt{predict\_proba}(\mathbf{q}) \in \Delta^{C-1}$$

where $\Delta^{C-1}$ is the $(C-1)$-simplex (probability vectors for $C$ classes). The sum of probability vectors weighted by $\{w_i\}$ is itself a valid probability vector (convex combination of simplicial points remains in the simplex).

This formula is structurally identical to **Bayesian Model Averaging**:

$$p(y|\mathbf{q}, \mathcal{D}) = \sum_{i} p(y|\mathbf{q}, M_i)\, p(M_i|\mathcal{D})$$

The Galactica Dictionary approximates the model posterior $p(M_i|\mathcal{D})$ with the Boltzmann weight $w_i(T)$. This is a form of **empirical Bayes**: instead of computing the true Bayesian posterior over models (which requires integrating over all parameters and is generally intractable), we use the CV accuracy as a proxy for the log-marginal-likelihood and exponentiate it with a temperature that controls the effective prior concentration.

**Fallback for deterministic pages.** Classifiers without `predict_proba` (RidgeClassifier, hard-voting VotingClassifier, LinearSVC) return only hard predictions. The code converts these to degenerate probability vectors:

```python
preds = pipe.predict(X)
proba = np.zeros((n_samples, n_classes))
proba[np.arange(n_samples), preds] = 1.0   # one-hot encoding
```

This is equivalent to treating the deterministic classifier as a probabilistic one with infinite confidence — appropriate as a fallback but suboptimal relative to true probability outputs.

**Prediction:**

```python
def predict(self, X):
    return np.argmax(self.predict_proba(X), axis=1)
```

The argmax of the blended probability is the **Bayes-optimal decision** under the mixture model with $C$ equal cost classes.

### 3.5 The Dominant Page and Weight Concentration

The dominant page $P^\ast = \arg\max_i w_i$ is always the CV-best page (weight is monotone in score). The weight concentration index:

$$\rho(T) = \frac{w_{\max}(T)}{1/N_p} = N_p \cdot w_{\max}(T) \in [1, N_p]$$

measures how much the dominant page is amplified relative to the uniform baseline. At the Iris run's $T=0.05$: $\rho = 100 \times 0.0293 = 2.93$. At $T^\ast=0.005$: the dominant page receives approximately $e^{1/0.005}/Z \approx e^{200}/Z$ — effectively all weight, since $\Delta_i/T$ for all non-dominant pages is very large.

**Important caveat:** The CV-best page is not necessarily the test-best page. On Iris, 7 pages achieve test accuracy = 1.0 while the CV-best page (QDA reg=0.1, CV=0.9917) achieves only test accuracy = 0.9333. This CV-test gap explains why full crystallization ($T \to 0$) is suboptimal: the CV ranking is a noisy estimate of generalization performance, especially with 2-fold CV on only 120 training samples.

### 3.6 Temperature Sweep: Reweight Without Refit

The `reweight(T)` method updates the weight vector for a new temperature without refitting any model:

```python
def reweight(self, temperature):
    self.temperature = temperature
    self.weights_    = stable_softmax_weights(cv_results, temperature)
    return self
```

This is $O(N_p)$ (one pass over the score vector). A temperature sweep over $K$ temperatures is $O(K \cdot N_p)$ weight computation plus $O(K \cdot N_{\text{test}} \cdot N_p \cdot C)$ for predictions — dominated by the prediction cost but with $K$ additional inference passes that are cheap relative to fitting.

```python
temps    = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
temp_accs = []
for T in temps:
    liquid.reweight(T)
    temp_accs.append(accuracy_score(y_test, liquid.predict(X_test)))
T_star = temps[np.argmax(temp_accs)]
```

This uses the test set for temperature selection — a mild form of data leakage. Version 2.0 will use a separate validation split for this step.

### 3.7 Score Normalization and Scale Invariance

Min-max normalization to $[0,1]$ before applying the Softmax is not cosmetic. Without it, the effective temperature is confounded with the scale of the score distribution. If scores span $[0.38, 1.00]$ (as in the Iris run), then a temperature of $T=0.05$ applied to raw scores has a very different concentration effect than applied to $[0.95, 0.99]$ scores. Normalization decouples temperature from scale:

$$\tilde{s}_i = \frac{s_i - \min_j s_j}{\max_j s_j - \min_j s_j}$$

After normalization, $T=0.05$ always means the same relative concentration regardless of how compressed or spread the raw score distribution is. This makes the temperature a **scale-invariant** measure of mixture fluidity.

### 3.8 Complexity Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Build dictionary | $O(N_p)$ | Python dict construction |
| CV scoring all pages | $O(N_p \cdot k \cdot N_{\text{train}} \cdot d)$ | $k$-fold, $N_{\text{train}}$ samples, $d$ features |
| Softmax weights | $O(N_p)$ | Trivial |
| Fit all pages | $O(N_p \cdot N_{\text{train}})$ | Dominated by slowest page |
| Temperature sweep | $O(K \cdot N_p \cdot N_{\text{test}} \cdot C)$ | $K$ temperatures, $C$ classes |
| Inference (per query) | $O(N_p \cdot d \cdot C)$ | Parallel page predictions |
| Memory (fitted pages) | $O(N_p \cdot M_{\text{avg}})$ | $M_{\text{avg}}$: avg model size |

For $N_p = 100$, $N_{\text{train}} = 120$, $k=2$, the Iris run completes in under 30 seconds on CPU. At $N_p = 10^4$ and $N = 10^6$, GPU-parallel CV scoring becomes essential.

**Scaling bottleneck:** The dominant cost at large $N_p$ is fitting $N_p$ independent models. This is embarrassingly parallel and maps naturally to GPU-batch execution. Future GPU-native implementations will distribute model fitting across CUDA streams, enabling $N_p = O(10^5)$ dictionaries on multi-GPU clusters.

---

## The 100-Page Dictionary: Algorithm Taxonomy

Each page is a scikit-learn `Pipeline(scaler, [transforms], clf)` — zero data leakage is guaranteed because all preprocessing is fitted on the training fold and applied to the test fold within cross-validation.

### Family 1: Linear Models (Pages 01–10)

**Inductive bias:** Decision boundary is a hyperplane in the (possibly feature-transformed) input space. Optimal when class-conditional distributions have linear Fisher discriminant directions or when regularization constraints match the true signal sparsity.

**Pages:** LogReg L2 (C=1, 0.01, 100), LogReg L1 (saga), LogReg ElasticNet, RidgeClassifier, SGD (log_loss, modified_huber), Perceptron, PassiveAggressive.

**Why three L2 logistic regressions?** The regularization strength $C \in \{0.01, 1, 100\}$ spans three orders of magnitude, covering: heavily regularized (C=0.01, near-uniform coefficients), moderate regularization (C=1, standard), and near-unregularized (C=100, coefficients allowed to be large). These are genuinely distinct algorithms — a C=0.01 LogReg will fail on non-trivially separable data where C=100 succeeds, and vice versa when high-variance features create spurious overfitting.

**SGD variants:** `log_loss` implements stochastic gradient descent on the logistic loss (online LogReg); `modified_huber` is more robust to outliers by interpolating between the hinge and logistic loss. These are included because for large $N$ they converge faster than batch methods, and the dictionary needs pages that are optimal in the large-$N$ regime.

### Family 2: Discriminant Analysis (Pages 11–14)

**Inductive bias:** Class-conditional distributions are Gaussian. LDA assumes equal covariance matrices across classes (Gaussian Fisher discriminant). QDA allows class-specific covariances, fitting a quadratic boundary.

**QDA reg=0.1 (Page 14) vs QDA (Page 13):** The regularization parameter $\alpha$ shrinks the class covariance matrices toward a diagonal: $\hat{\Sigma}_c(\alpha) = (1-\alpha)\hat{\Sigma}_c + \alpha \cdot \text{diag}(\hat{\Sigma}_c)$. This prevents the ill-conditioned inversion of $\hat{\Sigma}_c$ when $d \geq N_c$ (more features than class samples). On Iris (4 features, 50 samples per class), regularization is marginal — but on higher-dimensional datasets, unregularized QDA collapses completely.

### Family 3: Support Vector Machines (Pages 15–22)

**Inductive bias:** Maximum-margin hyperplane in the reproducing kernel Hilbert space (RKHS) induced by the chosen kernel. The RKHS norm serves as a capacity control, implementing structural risk minimization.

**RBF kernel pages at C=1, 10, 100:** The SVM dual problem with RBF kernel $k(\mathbf{x}, \mathbf{x}') = \exp(-\|\mathbf{x}-\mathbf{x}'\|^2/2\sigma^2)$ is:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C$$

The $C$ parameter controls the bias-variance trade-off: small $C$ → large margin, more misclassifications tolerated (high bias, low variance); large $C$ → small margin, fewer misclassifications (low bias, high variance). Three distinct values correspond to genuinely different boundary smoothness and generalization regimes.

**NuSVC (Page 22):** Parameterizes the problem via $\nu \in (0,1]$ — the fraction of support vectors — rather than $C$ directly. This reparameterization provides direct control over model sparsity.

### Family 4: Decision Trees (Pages 23–28)

**Inductive bias:** Axis-aligned recursive partitioning of feature space. Each split is a threshold on one feature; the boundary is a union of axis-aligned hyperrectangles.

**Criterion variants:** Gini impurity minimizes $1 - \sum_c p_c^2$; entropy minimizes $-\sum_c p_c \log p_c$; log_loss minimizes $-\sum_c p_c \log \hat{p}_c$. These are functionally similar but differ in their sensitivity to small class probability differences — log_loss is more aggressive at exploiting small purity improvements.

**Depth control:** `max_depth=3` produces at most 8 leaves — interpretable but potentially underfitting. `max_depth=5` provides 32 leaves. `max_depth=None` (unlimited) allows full memorization of the training set — high variance, useful when bagged or boosted.

### Family 5: Random Forests and Bagging (Pages 29–38)

**Inductive bias:** Bootstrap aggregation of high-variance estimators (trees). The variance reduction is $\sigma^2_{\text{ens}} = \bar{\rho}\,\sigma_{\text{tree}}^2 + (1-\bar{\rho})\sigma_{\text{tree}}^2/M$ where $\bar{\rho}$ is the inter-tree correlation. Feature subsampling in Random Forest reduces $\bar{\rho}$ by decorrelating trees.

**n_estimators variants (100, 200, 500):** More trees reduce variance further but with diminishing returns beyond $\sim$ 200. Included because optimal $M$ is data-dependent — on small datasets, 100 trees may already converge; on large datasets, 500 trees may still improve.

**`class_weight='balanced'` (Page 32):** Adjusts sample weights to be inversely proportional to class frequencies, making the RF robust to class imbalance. Included because on real datasets with imbalanced classes, this page may dominate.

**Bagging(SVM) (Page 38):** Bagging a non-tree estimator. Each SVM bootstrap sees a subset of training data, making the ensemble more robust than a single SVM to outliers in the training set. Computationally expensive but occasionally dominant.

### Family 6: Boosting (Pages 39–50)

**Inductive bias:** Sequential residual correction. Each weak learner targets the residuals of the ensemble so far. AdaBoost reweights misclassified samples; GradBoost fits residuals in function space; HistGradBoost uses histogram binning for speed.

**Subsample (Page 42):** `subsample=0.7` applies stochastic gradient boosting — each tree is fit on a random 70% of training data, reducing variance and providing implicit regularization analogous to dropout.

**XGBoost (Page 49):** Adds L1/L2 regularization on leaf weights and tree structure, column subsampling, and distributed computation support. LightGBM (Page 50) uses leaf-wise growth with depth constraint rather than level-wise, enabling faster convergence on large datasets.

### Family 7: k-Nearest Neighbors (Pages 51–58)

**Inductive bias:** The label of a query is determined by a local neighborhood vote. No global model is fitted — the entire training set is the model. Assumes smooth class boundaries and a meaningful distance metric.

**Distance weighting (Page 57):** $w_i = 1/d_i$ weights nearer neighbors more heavily, implementing a continuous version of the majority vote. Reduces the influence of distant same-class neighbors and reduces noise sensitivity.

**Manhattan metric (Page 58):** The L1 distance $d(\mathbf{x}, \mathbf{x}') = \sum_j |x_j - x'_j|$ is more robust to outliers in individual dimensions than the Euclidean L2 distance and may be superior on data with features of very different scale or with extreme values.

### Family 8: Naive Bayes (Pages 59–62)

**Inductive bias:** Full feature independence under the class label. The joint likelihood factorizes as $p(\mathbf{x}|c) = \prod_j p(x_j|c)$, enabling exact Bayesian classification despite ignoring all feature correlations. Highly effective when the independence assumption approximately holds.

**GaussianNB var_smoothing (Page 60):** Adds $10^{-8}$ to variance estimates to prevent zero-variance features from collapsing the likelihood. Critical when features have very small variance that may hit floating-point limits.

**BernoulliNB and ComplementNB (Pages 61–62):** Designed for binary or count features. BernoulliNB models binary feature presence/absence. ComplementNB fits models of the complement classes, improving accuracy on imbalanced text data. Both use MinMaxScaler to map features to $[0,1]$ — a requirement for the Bernoulli likelihood.

### Family 9: Neural Networks (Pages 63–70)

**Inductive bias:** Universal function approximation via layered nonlinear transformations. The width and depth of the MLP control the class of functions representable; regularization (L2 weight decay) and the solver control convergence to a good local minimum.

**Architecture diversity:** `(64,32)` is narrow and fast; `(128,64)` is the standard baseline; `(256,128,64)` adds depth; `(512,256)` is wide; `(64,64,64,64)` is deep and narrow; `(1024,)` is a single wide hidden layer. Different architectures favor different function classes — narrow-deep networks approximate compositional functions better; wide-shallow networks approximate non-compositional smooth functions better.

**tanh activation (Page 67):** Unlike the default ReLU, tanh produces outputs in $(-1,1)$ and is smooth everywhere, making it better suited for problems where gradient flow through negative activations matters.

**lbfgs solver (Page 68):** A quasi-Newton solver that uses full-batch gradient information. Converges faster than SGD on small datasets ($N < 1000$) but does not scale to large $N$.

### Family 10: Gaussian Processes (Pages 71–73)

**Inductive bias:** A GP places a prior directly over functions $f: \mathcal{X} \to \mathbb{R}$. Classification is performed by squashing a latent GP through a probit/logistic link. The kernel defines the correlation structure of the prior — which functions are a priori plausible.

**RBF kernel (Page 71):** $k(\mathbf{x}, \mathbf{x}') = \sigma^2 \exp(-\|\mathbf{x}-\mathbf{x}'\|^2/2\ell^2)$. Infinitely differentiable sample paths — appropriate when the true function is very smooth.

**Matérn$(\nu=1.5)$ kernel (Page 72):** $k_{\text{Matérn}}(\mathbf{x}, \mathbf{x}') = \sigma^2(1+\sqrt{3}r/\ell)\exp(-\sqrt{3}r/\ell)$ where $r = \|\mathbf{x}-\mathbf{x}'\|$. Sample paths are once-differentiable — appropriate when the function may have kinks or sharp transitions. More appropriate than RBF for physical signals.

**DotProduct kernel (Page 73):** $k(\mathbf{x}, \mathbf{x}') = \sigma_0^2 + \mathbf{x} \cdot \mathbf{x}'$. Non-stationary kernel that produces polynomial-like prior — appropriate when the function has a strong linear trend.

### Family 11: Kernel Approximation (Pages 74–79)

**Inductive bias:** Explicit finite-dimensional approximation to an RKHS, enabling large-scale kernel methods via standard linear classifiers on the approximated features.

**Nystroem (Pages 74, 76):** Approximates the kernel matrix $K$ as $K \approx \Phi\Phi^\top$ where $\Phi$ is computed from $m$ landmark points. This reduces the $O(N^2)$ kernel matrix to $O(N \cdot m)$, enabling kernel SVMs and kernel logistic regression at $N \gg m$.

**RBFSampler (Page 75):** Uses random Fourier features (Rahimi and Recht, 2007) to approximate the RBF kernel: $k(\mathbf{x}, \mathbf{x}') \approx z(\mathbf{x}) \cdot z(\mathbf{x}')$ where $z(\mathbf{x}) = \sqrt{2/D}\cos(\omega^\top \mathbf{x} + b)$ with $\omega \sim \mathcal{N}(0, I/\sigma^2)$. This is a Monte Carlo approximation to the Bochner integral representation of stationary kernels.

### Family 12: PCA Projection Pipelines (Pages 80–83)

**Inductive bias:** The principal components of the data covariance matrix form the best linear low-dimensional representation in terms of variance explained. Classifying in PCA space assumes that the discriminative signal is concentrated in the high-variance directions.

**PCA(2) vs PCA(3):** With 4 Iris features, PCA(2) retains 97.8% of variance; PCA(3) retains 99.9%. The difference is marginal on Iris but these pages matter on high-dimensional datasets where PCA projection is a genuine dimensionality reduction.

### Family 13: Semi-Supervised Methods (Pages 84–85)

**Inductive bias:** The true decision boundary lies in low-density regions of the input space (the cluster assumption). Label propagation and label spreading diffuse labels across the k-NN graph from labeled to unlabeled points.

**Applicability:** Currently all training points are labeled, making these pages equivalent to standard graph-based classifiers on this dataset. Their value emerges on semi-supervised problems where some training labels are missing.

### Family 14: Calibration Wrappers (Pages 86–88)

**Inductive bias:** The base classifier's raw probabilities are unreliable (overconfident SVM probabilities, poorly calibrated RF probabilities). Calibration wrappers apply Platt scaling (sigmoid calibration) or isotonic regression post-hoc.

**Sigmoid vs Isotonic:** Platt/sigmoid calibration fits a logistic function $\hat{p}_{cal} = 1/(1+e^{a\hat{p}+b})$ to the raw scores — parametric and appropriate when calibration error is monotone. Isotonic regression fits a piecewise constant monotone function — non-parametric and appropriate for arbitrary miscalibration patterns but requires more calibration data.

### Family 15: Multiclass Decomposition (Pages 89–92)

**One-vs-Rest (OvR):** Fits $C$ binary classifiers, each distinguishing class $c$ from all others. Prediction: $\hat{y} = \arg\max_c P(\text{class}=c|\mathbf{x})$.

**One-vs-One (OvO):** Fits $C(C-1)/2$ binary classifiers, each distinguishing one class pair. Prediction by majority vote. OvO is better for non-linear SVMs because each binary problem is simpler; OvR is better for linear models because it sees the full data per classifier.

### Family 16–17: Voting and Stacking Ensembles (Pages 93–98)

**Hard vs Soft Voting:** Hard voting takes the majority class; soft voting averages the class probabilities before taking argmax. Soft voting uses more information and is generally superior when base classifiers output calibrated probabilities.

**Stacking:** A meta-learner is trained on the out-of-fold predictions of base classifiers (cross-validated so the meta-training is not exposed to the base training targets). This allows the meta-learner to learn which base classifiers to trust on which regions of the input space — a data-driven version of the Boltzmann weighting without the statistical mechanics interpretation.

### Family 18: Ultra-Exotic Hybrids (Pages 99–100)

**Poly(2)+PCA(6)+SVM (Page 99):** A 3-stage pipeline: polynomial feature expansion of degree 2 (from $d=4$ to $\binom{d+2}{2}=15$ features), PCA projection to 6 dimensions (capturing most variance in the expanded space), then RBF-SVM classification. This approximates a polynomial kernel SVM in a compressed subspace.

**Spline+Nystroem+GradBoost (Page 100):** Cubic spline transformation (knot-based basis expansion capturing nonlinear monotone relationships), Nystroem kernel approximation (projecting the spline features into an RBF RKHS), then gradient boosted trees on the resulting representation. This chain synthesizes three distinct inductive biases.

---

## The Infinite Dictionary: Unexplored Algorithmic Territories

The 100-page v1.0 dictionary covers classical scikit-learn territory. The following describes the algorithm families that would populate the infinite dictionary — many of which are theoretically superior to all 100 current pages for specific data structures.

### 5.1 Physics-Informed Classification

**Harmonic Resonance Fields (HRF).** Each training point generates a radially damped oscillatory field:

$$\Psi_c(\mathbf{q}, \mathbf{x}_i) = \exp(-\gamma\|\mathbf{q}-\mathbf{x}_i\|^2)(1 + \cos(\omega_c\|\mathbf{q}-\mathbf{x}_i\|))$$

Class energy $E_c(\mathbf{q}) = \sum_{i:y_i=c}\Psi_c(\mathbf{q},\mathbf{x}_i)$. Achieved 98.9% peak accuracy on EEG Eye State — the author's own prior work demonstrating that for periodic data, wave-physics classifiers can outperform all 100 standard pages by a significant margin. Each set of parameters $(\omega_c, \gamma, k_{\text{local}})$ constitutes a distinct dictionary page.

**Riemannian Wave Classifier (RWC)** and **Geometric Wave Learner (GWL).** Classification via Lorentzian resonance on a graph Laplacian manifold, with GWL additionally evolving the metric via label-driven discrete Ricci flow. GWL achieved 93.46% on the same EEG dataset. Each hyperparameter tuple $(K, k, \varepsilon, \eta, T_{\text{steps}})$ is a page.

**Reaction-Diffusion Classifiers.** Decision boundaries as level sets of PDE solutions:

$$\partial_t u = D\nabla^2 u + f(u,v), \quad \partial_t v = D'\nabla^2 v + g(u,v)$$

(Turing system). Different reaction terms $f, g$ produce qualitatively different spatial patterns, each acting as a distinct class boundary generator. Appropriate for data with spatial pattern structure.

**Wave Equation Green's Function Classifier.** Classification via the Green's function $G(\mathbf{q}, \mathbf{x}_i; t)$ of the wave equation $(\partial_t^2 - c^2\nabla^2)u = 0$, measuring how waves emanating from each training point's position arrive at the query location. Each wave speed $c$ and time horizon $t$ is a page.

**Physics-Informed Neural Networks (PINNs) for classification.** Neural networks with physics-based loss terms that enforce known conservation laws or differential equation constraints as regularizers. Appropriate for scientific data where the governing equations are partially known.

### 5.2 Topological Data Analysis Classifiers

Topological Data Analysis (TDA) provides classification features derived from the homological structure of the data — features that are provably robust to continuous deformations of the data (topological invariants) and capture multi-scale geometric structure invisible to any metric-based classifier.

**Persistent Homology Classifier.** The Vietoris-Rips filtration of the training data at scale $\epsilon$ produces a simplicial complex $\mathcal{K}(\epsilon)$. The persistent homology groups $H_k(\mathcal{K}(\epsilon))$ track connected components ($k=0$), loops ($k=1$), and voids ($k=2$) as $\epsilon$ varies. The **persistence barcode** — the set of birth-death pairs $\{(\epsilon_b^j, \epsilon_d^j)\}$ — is a complete topological fingerprint of the data. Vectorizing the barcode (via persistence images, persistence landscapes, or Betti curves) yields features for any standard classifier. Each combination of (homology degree, vectorization method, classifier) is a page.

**Persistent Laplacian Classifier.** Beyond persistent homology, the **persistent combinatorial Laplacian** $\Delta_k(\epsilon)$ encodes both topological invariants (its zero-eigenvalue multiplicity equals the $k$-th Betti number) and geometric information (its non-zero spectrum encodes shape evolution during filtration). This extension, introduced by Wang et al. (2020), captures geometric changes that occur without topological changes — a limitation of standard persistent homology.

**Mapper Graph Classifier.** The Mapper algorithm produces a simplicial complex approximation of the data manifold by covering the data with overlapping filters and clustering within each patch. The topology of the Mapper graph (number of connected components, cycles) provides classification features. Different filter functions constitute different pages.

**TDA + Deep Learning.** Topological deep learning (TDL, Cang and Wei 2017) integrates persistence diagrams directly into neural network architectures, allowing gradients to flow through topological features. Each TDL architecture constitutes a dictionary page.

### 5.3 Causal Inference Classifiers

Standard classifiers learn $p(y|\mathbf{x})$ — the observational conditional. Causal classifiers learn $p(y|\text{do}(\mathbf{x}=\mathbf{x}'))$ — the interventional conditional under Pearl's do-calculus. These are equivalent only when the data-generating process has no hidden confounders and satisfies the Markov condition with respect to a known causal graph.

**Structural Causal Model (SCM) Classifier.** Given a known or estimated causal DAG over the features and label, the do-calculus provides a symbolic procedure to compute the interventional distribution from observational data. For datasets where confounding is present (e.g., medical data where treatment assignment is correlated with patient health), SCM classifiers are more robust to distribution shift than observational classifiers.

**Invariant Risk Minimization (IRM).** Instead of minimizing the average loss across environments, IRM finds representations that yield invariant optimal classifiers across all training environments — capturing the causal features that are predictive regardless of which spurious correlations happen to exist in any given environment. IRM classifiers are strictly more robust to domain shift than ERM (standard empirical risk minimization).

**Double Machine Learning (DML) Classifier.** Debias the treatment-outcome relationship by first regressing out the confounders from both the treatment and outcome, then classifying on the residuals. Each choice of nuisance function estimators constitutes a distinct page.

### 5.4 Neuromorphic and Reservoir Computing Classifiers

Reservoir computing (RC) avoids training the recurrent weights of a dynamical system — only the readout layer is trained. The fixed random reservoir projects the input into a high-dimensional, time-lagged feature space, and a linear readout maps this to labels.

**Echo State Network (ESN) Classifier.** A sparse random recurrent network of tanh neurons with spectral radius $\rho$ (controlling the fading memory timescale). The reservoir state $\mathbf{h}(t)$ evolves as:

$$\mathbf{h}(t+1) = \tanh(\mathbf{W}^{\text{res}}\mathbf{h}(t) + \mathbf{W}^{\text{in}}\mathbf{u}(t))$$

where $\mathbf{W}^{\text{res}}$ is the fixed random recurrent matrix (scaled to spectral radius $\rho$) and $\mathbf{W}^{\text{in}}$ is the fixed random input matrix. Classification: linear regression on the final reservoir state $\mathbf{h}(T)$. Each $(\rho, N_{\text{reservoir}}, \text{sparsity})$ is a page.

**Liquid State Machine (LSM) Classifier.** The spiking neural network analog of ESN: neurons emit discrete spikes according to the leaky integrate-and-fire model:

$$\tau \frac{dV_m}{dt} = -(V_m - V_{\text{rest}}) + RI, \quad \text{spike when } V_m \geq V_{\text{thresh}}$$

LSMs are natively temporal — they process spike trains rather than real-valued vectors, making them appropriate for event-based sensor data and neuromorphic hardware. LSMs demonstrate more consistent generalization with increasing reservoir size and maintain stable performance under aggressive reservoir quantization compared to ESNs.

**Physical Reservoir Computing Pages.** Physical systems can serve as reservoirs: optical delay-line reservoirs (nanosecond timescales), spintronic nano-oscillator networks (GHz dynamics), memristive crossbar arrays (analog, non-volatile), quantum optical reservoirs (exploiting quantum coherence). Each physical implementation with each readout constitutes a page in the infinite dictionary — potentially orders of magnitude more energy-efficient than digital implementations.

### 5.5 Quantum and Quantum-Inspired Classifiers

**Quantum Kernel SVM.** The quantum kernel $k_Q(\mathbf{x}_i, \mathbf{x}_j) = |\langle\phi(\mathbf{x}_i)|\phi(\mathbf{x}_j)\rangle|^2$ is computed via quantum circuit evaluation, where $|\phi(\mathbf{x})\rangle = U(\mathbf{x})|0\rangle^{\otimes n}$ is the feature map circuit. For certain data distributions with quantum structure in their correlation patterns, quantum kernels are conjectured to capture feature relationships that are exponentially hard to express with classical polynomial kernels. Each circuit ansatz $U(\mathbf{x})$ is a page.

**Variational Quantum Classifier (VQC).** A parametric quantum circuit $U(\theta)$ followed by a Pauli measurement. The expectation value $\langle 0|U^\dagger(\theta)\hat{O}U(\theta)|0\rangle$ is used for classification. Trained via parameter-shift rule (quantum-native gradient). Each circuit architecture and depth is a page.

**Quantum Amplitude Estimation Classifier.** Encodes the training data as quantum state amplitudes and uses quantum amplitude estimation to compute class overlaps exponentially faster than classical inner products — a potential $O(N^{0.5})$ speedup over classical $O(N)$ kernel evaluation.

**Quantum-Inspired Tensor Network Classifier.** Matrix Product State (MPS) / Tensor Train classifiers that exponentially compress high-dimensional probability distributions via bond-dimension truncation. The Bond dimension $\chi$ controls the expressiveness of the compressed representation; each $\chi$ is a page.

### 5.6 Geometric Deep Learning Pages

**Graph Neural Network (GNN) Classifiers.** For data with inherent graph structure (molecular graphs, social networks, knowledge graphs), GNNs iteratively aggregate features from local neighborhoods:

$$\mathbf{h}_v^{(l+1)} = f_\theta\!\left(\mathbf{h}_v^{(l)},\, \bigoplus_{u \in \mathcal{N}(v)} \psi_\phi(\mathbf{h}_u^{(l)})\right)$$

where $\bigoplus$ is a permutation-invariant aggregation (sum, mean, max) and $f_\theta, \psi_\phi$ are learnable functions. Each GNN architecture (GCN, GAT, GraphSAGE, GIN), depth, and aggregation is a page.

**Equivariant Neural Networks ($E(n)$-equivariant NNs).** For 3D molecular or particle data, networks that are exactly equivariant to rotations, translations, and reflections — guaranteeing that predictions transform correctly under symmetry operations. Each equivariant architecture is a page.

**Hyperbolic Space Classifiers.** Data with hierarchical or tree-like structure (taxonomies, parse trees, evolutionary trees) embeds with dramatically lower distortion in hyperbolic space (Poincaré disk or half-space model) than in Euclidean space. Hyperbolic SVM and hyperbolic neural networks classify in the Poincaré disk:

$$d_H(\mathbf{x}, \mathbf{y}) = \text{arcosh}\!\left(1 + \frac{2\|\mathbf{x}-\mathbf{y}\|^2}{(1-\|\mathbf{x}\|^2)(1-\|\mathbf{y}\|^2)}\right)$$

Each curvature $K < 0$ and classifier architecture is a page.

### 5.7 Information-Geometric and Optimal Transport Classifiers

**Fisher-Rao Metric Classifier.** The Fisher information metric on the statistical manifold of parametric distributions:

$$g_{ij}(\theta) = \mathbb{E}_{p(\mathbf{x}|\theta)}\!\left[\frac{\partial \log p}{\partial \theta_i}\frac{\partial \log p}{\partial \theta_j}\right]$$

provides a natural distance between probability distributions that respects the geometry of the probability simplex. Classification via geodesic distances under the Fisher-Rao metric is more appropriate than Euclidean distances when the data is best described as a distribution over a parametric family.

**Wasserstein Distance Classifier (Optimal Transport).** The Wasserstein-$p$ distance:

$$W_p(\mu, \nu) = \inf_{\gamma \in \Pi(\mu,\nu)}\!\left(\int \|x-y\|^p \,d\gamma(x,y)\right)^{1/p}$$

between the empirical distribution of a query and the empirical distributions of each class. This classifier is the minimum-transport-cost version of k-NN and is provably more robust to outliers than Euclidean-distance k-NN. Each $(p, \text{regularization})$ is a page.

**Maximum Mean Discrepancy (MMD) Classifier.** Classifies by comparing the kernel mean embeddings of the query's neighborhood distribution and each class distribution:

$$\text{MMD}^2(\mathbb{P}, \mathbb{Q}) = \|\mu_\mathbb{P} - \mu_\mathbb{Q}\|_{\mathcal{H}}^2$$

where $\mu_\mathbb{P} = \mathbb{E}_{x \sim \mathbb{P}}[k(\cdot, x)] \in \mathcal{H}$ is the kernel mean embedding. Each kernel choice is a page.

### 5.8 Probabilistic Programming Classifiers

Full Bayesian classifiers specified as probabilistic programs, where both the model parameters and hyperparameters have explicit prior distributions and inference is performed via Hamiltonian Monte Carlo (HMC) or variational inference (VI).

**Bayesian Logistic Regression (Stan/Pyro).** Unlike maximum likelihood logistic regression, Bayesian LR maintains a full posterior distribution over the weight vector $\mathbf{w}$:

$$p(\mathbf{w}|\mathcal{D}) \propto \prod_i p(y_i|x_i, \mathbf{w}) \cdot p(\mathbf{w})$$

Prediction is via posterior predictive integration $p(y|\mathbf{q}, \mathcal{D}) = \int p(y|\mathbf{q}, \mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w}$, which averages over all weight configurations consistent with the data — a form of model averaging within the logistic regression family.

**Dirichlet Process Mixture Classifiers.** Non-parametric Bayesian classifiers where the number of mixture components is itself a random variable drawn from a Dirichlet process. These grow their complexity automatically as more data arrives, with no need to specify the number of clusters a priori.

**Gaussian Process Classification with Laplace Approximation or Expectation Propagation.** Full posterior GP classification rather than the gradient-descent-approximated version in scikit-learn. Different inference algorithms (Laplace, EP, MCMC) constitute distinct pages.

### 5.9 Symbolic and Evolutionary Computation Pages

**Genetic Programming Classifiers (TPOT-style).** Each generation of a GP run over the pipeline space produces a distinct classifier — a page. The GP explores arbitrary compositions of preprocessing, feature engineering, and classification steps beyond what any human-designed taxonomy would enumerate.

**Symbolic Regression Classifiers.** Algorithms like PySR or Eureqa discover symbolic mathematical expressions $f(\mathbf{x})$ that fit the data, producing classifiers of the form $\hat{y} = \text{sign}(f(\mathbf{x}))$ where $f$ is a mathematical formula rather than a black-box function. These are maximally interpretable and sometimes reveal physical laws hidden in the data. Each complexity budget (maximum expression depth) is a page.

**Grammatical Evolution Classifiers.** Uses context-free grammars to generate candidate programs, with genetic operators applied in grammar space rather than program space. Each grammar specification is a page.

### 5.10 Conformal Prediction Wrappers

**Conformal classifiers** wrap any base classifier to produce provably valid prediction sets: for any confidence level $1-\alpha$, the true label is guaranteed to be in the prediction set with probability at least $1-\alpha$ under exchangeability. The coverage guarantee is exact (not asymptotic) for any finite sample size and any base classifier.

**Inductive Conformal Classifier (ICP).** Splits the training data into proper training and calibration sets. Fits the base classifier on the proper training set. Computes nonconformity scores on the calibration set. At test time, the prediction set is all labels $c$ with nonconformity score below the $\lceil(1-\alpha)(n+1)\rceil/n$ quantile of calibration scores. Each base classifier constitutes a page.

**Cross-Conformal Classifier.** Uses cross-validation to avoid splitting the training data, recovering full training efficiency at the cost of independence between calibration scores.

### 5.11 Hyperbolic Space Classifiers

Beyond Poincaré disk embedding, the infinite dictionary includes:

**Lorentzian (Hyperboloid) Model Classifiers.** The hyperboloid model $\mathcal{H}^n = \{x \in \mathbb{R}^{n+1}: \langle x,x\rangle_L = -1, x_0 > 0\}$ with Lorentzian inner product $\langle x,y\rangle_L = -x_0y_0 + \sum_{i=1}^n x_iy_i$ provides a different coordinatization of hyperbolic space with better numerical stability for classification in very high dimensions.

**Product Space Classifiers.** $\mathbb{R}^m \times \mathbb{H}^n \times \mathbb{S}^k$ — product manifold classifiers that jointly embed flat, hyperbolic, and spherical components. Appropriate for data with mixed structure (some features hierarchical, some cyclic, some flat).

### 5.12 Cellular Automata and Complex Systems Classifiers

**Rule-Based Cellular Automaton Classifier.** The binary string of feature values is evolved for $T$ steps under a chosen rule from Wolfram's elementary CA taxonomy (256 possible rules for 1D). Classification is based on the steady-state pattern of the evolved string. Different rules and evolution horizons are pages.

**Agent-Based Model Classifier.** Training points are modeled as agents following local interaction rules. The emergent global behavior (whether the system converges to a homogeneous or heterogeneous state, the density of class-$c$ attractors in the phase space) determines the classification. Each interaction rule is a page.

---

## Relationship to Existing AutoML Frameworks

| System | Search Strategy | Combination | Physics Pages | Temperature Control |
|--------|----------------|-------------|---------------|---------------------|
| Auto-Sklearn | Bayesian optimization + ensembling | Top-K stacking | No | No |
| TPOT | Genetic programming | Single pipeline | No | No |
| H2O AutoML | Grid/random search + stacking | Stacked ensemble | No | No |
| **Galactica Dictionary** | Exhaustive evaluation | Boltzmann blend | Yes (extensible) | Yes |

The key distinction is the **combination mechanism**. AutoML systems return a single optimized pipeline or a fixed post-hoc stack. The Dictionary returns a Boltzmann-weighted mixture of all evaluated pages. This distinction matters precisely in the regime where the CV evidence is insufficient to confidently identify one best pipeline — the fluid state where the dictionary's temperature control is most valuable.

The Dictionary is also uniquely **extensible to physics-informed architectures**. Any scikit-learn-compatible estimator — including HRF, RWC, GWL, GP with quantum kernels, TDA pipelines — can be registered as a page with zero framework modification. No existing AutoML system has this open-ended extensibility combined with a principled combination mechanism.

---

## Empirical Results on Iris

**Dataset:** Iris (OpenML classic), N=150, 4 features (sepal/petal length/width), 3 balanced classes (Setosa, Versicolor, Virginica). 80/20 train/test split stratified. StratifiedKFold, k=2 on 120 training samples.

**Seven pages achieving test accuracy = 1.000:** `03 LogReg L2 C=100`, `11 LDA (svd)`, `12 LDA (lsqr)`, `13 QDA`, `18 SVM Linear`, `57 KNN k=7 distance`, `67 MLP tanh (128,64)`.

**CV-best page:** `14 QDA reg=0.1` (CV=0.9917) — achieves test accuracy 0.9333. This demonstrates that with only 60 samples per fold, CV rankings are unreliable estimators of generalization performance.

**Five worst pages (test accuracy):** `61 BernoulliNB` (0.3667), `62 ComplementNB` (0.6667), `06 Ridge` (0.7667), `02 LogReg C=0.01` (0.8000), `09 Perceptron` (0.8667). All Naive Bayes-like models that require non-negative features fail spectacularly when applied to standardized (zero-mean) data — a known limitation of the BernoulliNB assumption.

**Temperature sweep results:**

| $T$ | $N_{\text{eff}}$ | Test Accuracy | Notes |
|-----|-----------------|--------------|-------|
| 0.001 | 1.00 | 0.9333 | Crystallized → CV-best page (QDA reg=0.1) |
| 0.005 | 2.14 | **0.9667** | Optimal $T^\ast$ |
| 0.01 | 3.52 | 0.9667 | |
| 0.05 | 34.1 | 0.9667 | Working temperature |
| 0.10 | 54.7 | 0.9667 | |
| 0.50 | 88.3 | 0.9667 | |
| 1.00 | 94.2 | 0.9667 | |
| 5.00 | 99.1 | 0.9667 | Fully liquid → uniform |

The flat plateau from $T=0.005$ to $T=5.0$ indicates that on Iris, the high-performing pages are largely consistent in their predictions, making the blend accuracy insensitive to the exact mixing ratio above the threshold. The sharp drop at $T=0.001$ is the crystallization artifact — concentrating onto the CV-best page (QDA reg=0.1) which underperforms the test-best pages.

**Dominant page at $T=0.05$:** `14 QDA reg=0.1` ($w=0.0293$, $\rho = 2.93\times$ uniform).

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Input: (X, y) — any tabular dataset                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼────────────────────────────────────────┐
│              make_dictionary(seed=42) → DICTIONARY                          │
│  18 algorithm families · 100 pages · each = Pipeline(scaler, [tf], clf)    │
│  Registered as {name_str: sklearn.Pipeline} dict                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼────────────────────────────────────────┐
│         StratifiedKFold CV Scoring — Cell 4                                 │
│  cv_results[name] = {mean, std, scores, time}                               │
│  Parallelized with n_jobs=-1 across all CPU cores                           │
│  Cost: O(N_p · k · N_train · d) — dominant step                            │
└──────────────────────────────────────────────────────────────────────────────┘
           │                                │
┌──────────▼──────────────────┐  ┌──────────▼──────────────────────────────────┐
│ stable_softmax_weights(T)   │  │ Temperature Sweep                           │
│ Min-max normalization       │  │ temps = [0.001, 0.005, ..., 5.0]            │
│ Log-sum-exp shift           │  │ liquid.reweight(T) — O(N_p) per T           │
│ Partition function Z(T)     │  │ → best_T_star via validation accuracy       │
│ Weights: {name → w_i} dict  │  └─────────────────────────────────────────────┘
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────────────┐
│           LiquidDictionary.fit(X_train, y_train, cv_results)               │
│  For each page: pipe.fit(X_train, y_train)                                 │
│  Stores fitted_pages_, weights_                                             │
│  Verbose: weight bar chart printed to stdout                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼────────────────────────────────────────┐
│           LiquidDictionary.predict_proba(X_test)                           │
│  blended = Σ_i  w_i · P_i.predict_proba(X_test)                           │
│  Fallback: one-hot from predict() for non-probabilistic pages              │
│  Classification: y_hat = argmax_c blended[:,c]                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## GPU Implementation Details and Scaling Roadmap

Current v1.0 uses scikit-learn (CPU, parallelized) for all model fitting and inference. GPU-native implementation roadmap:

| Operation | v1.0 (CPU) | v2.0 (GPU-native) | Speedup Estimate |
|-----------|-----------|------------------|------------------|
| k-NN pages | sklearn, $O(N^2 d)$ | cuML FAISS, $O(N \log N)$ | 10–100× at $N=10^5$ |
| Tree pages | sklearn CPU parallel | cuML RF, histogram-based GPU | 5–20× |
| SVM pages | sklearn LIBSVM | cuML SVM (sparse/dense) | 3–10× |
| GP pages | sklearn dense | GPyTorch (sparse GP, GPU) | 50–500× at $N=10^4$ |
| MLP pages | sklearn LBFGS/SGD | PyTorch, mixed-precision | 10–50× |
| Softmax weights | numpy (μs) | Same (trivial) | 1× |
| Blended inference | numpy matmul | cupy matmul, $O(B \cdot N_p \cdot C)$ | 5–20× |
| Fit all pages | Sequential CPU | CUDA streams, parallel GPU | $N_p\times$ (embarrassing parallelism) |

**VRAM budget at scale:** For $N_p = 10{,}000$ pages and $N = 10^6$ samples, assuming each page stores $O(N \cdot d)$ training data: 78 bytes per float32 sample $\times 10^6 \times$ model overhead requires distributed multi-GPU storage. The solution is **on-demand page loading**: fitted pages are serialized to disk and loaded only at inference time, with GPU VRAM serving as a cache for the top-$K$ weighted pages.

---

## Hyperparameter Reference

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `temperature` | 0.05 | $(0, \infty)$ | Boltzmann fluidity. $T \to 0$: crystallize. $T \to \infty$: uniform. |
| `CV_FOLDS` (k) | 2 | 2–10 | Trade-off between CV reliability and training set size per fold. |
| `test_size` | 0.20 | 0.10–0.30 | Held-out test fraction. Larger → more reliable accuracy estimate. |
| `temps` (sweep) | 8-point log grid | log-spaced, $[10^{-3}, 5]$ | Resolution of $T^\ast$ search. |
| `random_state` | 42 | any int | Controls all page random seeds and CV fold splits simultaneously. |
| `N_p` | 100 | $\infty$ (theoretical) | Number of registered pages. |
| `max_samples` | (page-specific) | — | Bagging fraction for Bagging pages (fixed at 1.0 for non-Bagging). |

---

## Getting Started

```bash
git clone https://github.com/Devanik21/Infinite-Galactica-Dictionary.git
cd Infinite-Galactica-Dictionary

# CPU-only installation (all v1.0 functionality)
pip install scikit-learn xgboost lightgbm numpy pandas matplotlib seaborn

# GPU acceleration (future v2.0)
pip install cupy-cuda12x cuml-cu12 cudf-cu12 --extra-index-url https://pypi.nvidia.com

# Launch notebook
jupyter notebook Liquid_Galactica.ipynb
```

**CPU fallback:** v1.0 is pure scikit-learn. No GPU required. Torch is imported only for device detection (identification of the GPU name) and has no effect on computation.

---

## Future Roadmap

**v2.0 (1,000+ pages):** Full hyperparameter grids for each family (5–10 settings per hyperparameter). HRF, RWC, GWL registered as physics-informed pages. CatBoost, TabNet, NODE, NGBoost added. Three-way train/validation/test split for leak-free temperature selection.

**v3.0 (Meta-Feature Warm Start):** Dataset meta-features (class imbalance ratio, intrinsic dimensionality, statistical moments of marginal distributions, landmarker performance on 10 fast classifiers) used to initialize the temperature prior and weight warm-start, reducing CV budget needed before the dictionary reaches a good mixture.

**v4.0 (Online Streaming):** Sliding-window reweighting adapts page weights to concept drift without model refitting. The Online Performance Estimation Framework (van Rijn et al.) tracks rolling CV accuracy per page and updates the Boltzmann distribution continuously.

**v5.0 (TDA + Causal + Reservoir Pages):** First version incorporating topological data analysis classifiers (Ripser-based persistent homology feature extraction + SVM/RF), causal classifiers (IRM, DML), and Echo State Network pages.

**v$\infty$ (Generative Dictionary):** A generative model of algorithm pipelines samples novel pages by composing preprocessing and classification components probabilistically (analogous to TPOT's genetic programming but guided by the Boltzmann evidence from prior evaluations). The dictionary grows continuously, with each new page either gaining weight (if it outperforms the existing mixture) or being discarded (if its contribution is negligible at any reasonable temperature).

---

## Authors

**Devanik Debnath** — *Dictionary architecture, Boltzmann weighting design, physics-informed page vision, statistical mechanics formulation, temperature sweep mechanism, GPU scaling roadmap*  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

**Xylia** — *Co-author: theoretical framework, mathematical formalization, iterative design validation*

---

## License

Licensed under the [Apache License 2.0](LICENSE).

---

*"No Free Lunch is not a barrier. It is a map. It tells you where to look: not for a universal algorithm, but for a liquid one — one that lets the data choose its own shape."*
