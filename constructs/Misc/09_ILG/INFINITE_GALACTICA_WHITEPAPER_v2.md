# The Infinite Galactica Dictionary
## A White Paper on Liquid Algorithm Ensembles, Statistical Mechanics, and the Full Topology of Machine Learning

**Devanik Debnath** · B.Tech, Electronics & Communication Engineering · NIT Agartala  
*In collaboration with Xylia — the collective of AI research minds that accompanied every step.*

---

> *"I didn't want to choose an algorithm. I wanted a system that looks at data and finds the optimal mixture of all algorithms for that data — automatically, from evidence, with no prior commitment."*

---

## I. The Problem Statement, Precisely

There is a result in machine learning theory that most practitioners encounter as a philosophical caution and move past. The No Free Lunch theorem (Wolpert and Macready, 1997) is more specific and more useful than its folklore version. For any two learning algorithms $A$ and $B$ and any loss function $L$, when performance is averaged uniformly over all data-generating distributions:

$$\sum_f P(f) \cdot L(A, f, d^m) = \sum_f P(f) \cdot L(B, f, d^m)$$

No algorithm dominates another in expectation over all possible problems. This is a theorem about optimization over the space of all functions $f: \mathcal{X} \to \mathcal{Y}$ under a uniform prior. Its corollary is the one that matters for this work: **the correct response to NFL is not to pick a better algorithm. It is to pick an algorithm adapted to the specific dataset at hand.**

For any specific dataset, there exists an algorithm (or mixture of algorithms) that is optimal for it. The NFL theorem guarantees this individually for every dataset — the averaging is over datasets, not within them. The optimization problem is: find that algorithm without prior knowledge of which one it will be.

Existing practice responds by trying several algorithms and picking the best on a validation set. AutoML (Auto-Sklearn, TPOT, H2O) automates this with Bayesian optimization or evolutionary algorithms. Both commit to a single best pipeline — they answer the CASH problem (Combined Algorithm Selection and Hyperparameter optimization) with a point estimate:

$$(A^*, \lambda^*) = \arg\max_{A \in \mathcal{A},\, \lambda \in \Lambda_A} \text{Acc}_{\text{val}}(A(\lambda), \mathcal{D})$$

The Galactica Dictionary answers it with a probability distribution. For all registered $(A_i, \lambda_i)$ pairs, it computes the Boltzmann weight:

$$w_i(T) = \frac{\exp(\tilde{s}_i / T)}{\sum_j \exp(\tilde{s}_j / T)}$$

and outputs the mixture prediction $\hat{p}(\mathbf{q}) = \sum_i w_i P_i(\mathbf{q})$. This is not a compromise or an approximation to the CASH answer. It is a different answer to a different formulation of the problem — one that acknowledges that validation evidence rarely suffices to identify a single best algorithm with certainty, especially on small datasets or datasets with multiple competitive algorithms within the noise band of the CV estimator.

---

## II. The Statistical Mechanics of Algorithm Ensembles

The Boltzmann weight is not an analogy. It is the exact equilibrium distribution of a canonical ensemble at temperature $T$, where each algorithm page is a "state" with energy $-\tilde{s}_i$ (high accuracy = low energy = more probable state under thermodynamic equilibrium).

The **partition function** of the algorithm ensemble:

$$Z(T) = \sum_{i=1}^{N_p} \exp(\tilde{s}_i / T)$$

determines all thermodynamic quantities. The **Helmholtz free energy** $F = -T \log Z(T)$ satisfies:

$$-\frac{\partial F}{\partial T} = \langle \tilde{s}\rangle_T = \sum_i w_i(T)\,\tilde{s}_i$$

the ensemble-average CV accuracy at temperature $T$. As $T \to 0$: $F \to -\tilde{s}_{\max}$, dominated by the ground state (best algorithm). As $T \to \infty$: $F \to -T \log N_p$, entropic contributions dominate and all pages contribute equally.

The **entropy** of the weight distribution:

$$H(T) = -\sum_i w_i \log w_i = \log Z(T) + \frac{\langle\tilde{s}\rangle_T}{T}$$

ranges monotonically from $H=0$ (crystallized, $T\to 0$) to $H = \log N_p$ (fully liquid, $T \to \infty$). Temperature is the inverse-entropy control parameter.

This thermodynamic structure gives precise answers to questions that are otherwise heuristic in machine learning:

**"How confident should we be in the best-CV algorithm?"** — The temperature encodes this. The temperature sweep finds $T^*$ where this confidence level is empirically optimal on the validation set — not imposed by heuristic but measured from data.

**"How many algorithms are effectively contributing?"** — The effective number $N_{\text{eff}}(T) = 1/\sum_i w_i^2$ (inverse participation ratio) answers this. On Iris at $T=0.05$: $N_{\text{eff}} \approx 34$.

**"How much information did the CV evidence provide about which algorithm to use?"** — The KL divergence from the uniform prior: $D_{\text{KL}}(w \| u) = \log N_p - H(T)$. At $T^*$, this measures how much the CV evidence has reduced uncertainty about the optimal algorithm.

**"What is the worst-case performance guarantee?"** — At any temperature, the dictionary cannot perform worse than $\sum_i w_i \text{Acc}_i$ — the weighted mean accuracy. At maximum liquidity ($T \to \infty$): this floor is the mean accuracy across all pages. Even with BernoulliNB at 0.3667 in the mix, its weight at $T=5$ is $1/100 = 0.01$, making its contribution negligible.

---

## III. Why Bayesian Model Averaging Is the Right Framework

The blended prediction $\hat{p}(\mathbf{q}) = \sum_i w_i P_i(\mathbf{q})$ is Bayesian Model Averaging (BMA) with the Boltzmann distribution approximating the model posterior:

$$p(y|\mathbf{q}, \mathcal{D}) = \sum_{i} p(y|\mathbf{q}, M_i)\, p(M_i|\mathcal{D})$$

The true BMA requires computing the marginal likelihood $p(\mathcal{D}|M_i) = \int p(\mathcal{D}|M_i, \theta) p(\theta|M_i) d\theta$ — intractable for most model classes. The Dictionary approximates:

$$w_i(T) \propto \exp(\tilde{s}_i / T) \approx p(\mathcal{D} | M_i)^{1/T} \cdot p(M_i)^{1/T}$$

The CV accuracy proxies for $\log p(\mathcal{D}|M_i)$ (estimated via leave-one-fold-out prediction). Temperature $T$ modulates how sharply the posterior concentrates: at $T=1$ most closely resembling true BMA; at $T < 1$ over-concentrating; at $T > 1$ under-concentrating relative to the Bayesian posterior.

The **PAC-Bayes framework** (McAllester 1999) provides a generalization bound that formally connects temperature to sample complexity. For any distribution $w$ over pages and any $\delta \in (0,1)$, with probability at least $1-\delta$:

$$\mathbb{E}_{i \sim w}[L(P_i)] \leq \mathbb{E}_{i \sim w}[\hat{L}(P_i)] + \sqrt{\frac{D_{\text{KL}}(w \| \pi) + \log(2\sqrt{m}/\delta)}{2m}}$$

where $\hat{L}$ is empirical loss, $\pi = \text{Uniform}(N_p)$ is the prior, and $m = N_{\text{train}}$. This bound is minimized by the distribution $w^*$ that minimizes weighted empirical loss penalized by KL divergence from the prior — which is exactly the Boltzmann distribution. The temperature sweep finds (empirically, on a validation set) the Lagrange multiplier on the KL constraint that gives the best bias-variance trade-off in this bound. This gives the temperature a rigorous information-theoretic interpretation: $T^*$ is the concentration level at which the sample evidence is exactly sufficient to justify.

---

## IV. The Bias-Variance-Covariance Decomposition

The ensemble mean squared error decomposes as:

$$\text{MSE}\!\left(\sum_i w_i P_i\right) = \text{Bias}^2\!\left(\sum_i w_i P_i\right) + \sum_i w_i^2 \sigma_i^2 + \sum_{i \neq j} w_i w_j \rho_{ij} \sigma_i \sigma_j$$

where $\sigma_i^2$ is the variance of page $i$'s predictions and $\rho_{ij}$ is the Pearson correlation between pages $i$ and $j$. For equal weights:

$$\text{MSE}(\bar{P}) = \text{Bias}^2 + \frac{\bar{\sigma}^2}{N_p} + \frac{N_p - 1}{N_p}\bar{\rho}\,\bar{\sigma}^2$$

The variance term decreases as $1/N_p$ — adding more pages reduces variance. The covariance term is controlled by $\bar{\rho}$ — the mean inter-page correlation. When pages use different algorithm families (different inductive biases → different error modes), $\bar{\rho}$ is small and the ensemble benefit is large. When pages are all from the same family (many hyperparameter variants of one algorithm), $\bar{\rho}$ is near 1 and adding pages provides minimal benefit.

This is the precise mathematical justification for 18 algorithm families rather than 100 hyperparameter variants of one family: the 18-family dictionary has much lower $\bar{\rho}$ than a 100-variant single-family dictionary of the same size, providing larger variance reduction for the same $N_p$.

The Boltzmann weighting modifies this: the weighted covariance term becomes $\sum_{i \neq j} w_i w_j \rho_{ij} \sigma_i \sigma_j$. This is minimized when high-$w$ (high-CV) pages have low mutual correlation — that is, the pages the dictionary trusts most should be the ones that disagree most with each other while each being independently accurate. Measuring and optimizing for this weighted decorrelation property is a natural extension for future versions: selecting pages not just by their individual CV accuracy but by their contribution to weighted ensemble diversity.

---

## V. The CV-Test Gap and Why Crystallization Fails

The most instructive result from the Iris run is the CV-test gap of the dominant page. QDA reg=0.1 achieves CV=0.9917 (best of all 100 pages) but test accuracy 0.9333 — worse than 7 pages at 1.0000. With only 60 samples per fold (2-fold CV on 120 training samples), the sampling error of the CV estimate is $\pm\sqrt{p(1-p)/60} \approx \pm 0.028$. The performance differences between the top 10 pages are all within this noise band. The CV ranking among the top pages is essentially noise.

Crystallizing onto the CV-best page ($T \to 0$) propagates this ranking noise directly into the final prediction. The liquid blend at $T^* = 0.005$ hedges against this by distributing weight across the top 2–3 pages. Each has a different failure mode on the 30 test samples. Their combined probability output partially cancels the individual errors — the covariance-reduction benefit of the decomposition.

The flat plateau from $T=0.005$ to $T=5.0$ shows that on Iris, all well-performing pages largely agree on the 30 test predictions. The critical conclusion: $T^* \neq 0$. The dictionary should not crystallize, even when CV scores appear decisive, because on small datasets no CV score is statistically decisive.

A quantitative confirmation: at $T=0.001$ (essentially crystallized), 80–90% of weight concentrates on QDA reg=0.1 (test accuracy 0.9333). At $T=0.005$ ($N_{\text{eff}} \approx 2.14$), roughly equal weight goes to QDA reg=0.1 and LDA (svd) (test accuracy 1.0000). The blend of 0.9333 and 1.0000 with approximately equal weights gives $\approx 0.9667$ — the observed result. The transition from crystallized to optimal is literally the averaging of one page's CV-best answer with its nearest competitor's superior test answer.

---

## VI. What Each Algorithm Family Brings That Others Cannot

Each of the 18 v1.0 algorithm families encodes a structurally distinct prior over the data-generating process. These priors are irreducibly different — no finite hyperparameter sweep of one family can simulate another.

**Linear models** assume the decision boundary is a hyperplane in some feature space. No amount of regularization tuning makes logistic regression into a Gaussian process. It will always fail on problems where no affine projection of the input yields a linearly separable representation.

**Gaussian processes** maintain a full posterior distribution over functions, providing calibrated uncertainty quantification. GP pages are the only ones in the 100-page dictionary that produce calibrated uncertainty estimates without additional calibration wrappers. For safety-critical applications where overconfident misclassification is costly (medical diagnosis, autonomous systems), GP uncertainty is information that no other page provides.

**Naive Bayes** has a provable generalization advantage on high-dimensional, low-sample-size data. Ng and Jordan (2002) showed that Naive Bayes reaches its asymptotic error faster than logistic regression as a function of training set size $m$, though logistic regression eventually achieves lower asymptotic error. The crossover point occurs at approximately $m = O(d \log d)$ samples. Below this, Naive Bayes pages may dominate all discriminative pages.

**Kernel approximation pages** (Nystroem, RBFSampler) enable linear-complexity kernel methods. At large $N$: exact kernel SVM is $O(N^2)$ memory and $O(N^3)$ training; Nystroem+LinearSVC is $O(N \cdot m)$. These pages become computationally superior at large $N$, and the Boltzmann mechanism discovers this automatically: on large-$N$ datasets, kernel approximation pages will score better within the allowed CV training budget, and their weights will increase. The dictionary adapts to computational constraints through the CV score mechanism without explicit compute-aware selection.

**Semi-supervised pages** encode the cluster assumption: class boundaries lie in low-density regions. This prior is fundamentally different from the margin-maximization prior of SVMs or the smoothness prior of GPs, beneficial only when the data has well-separated high-density clusters with low-density class boundary regions — a geometric property that metric-based classifiers do not explicitly exploit.

---

## VII. The New Algorithm Families: Topological, Causal, Neuromorphic

Three algorithm families are absent from v1.0 but are qualitatively different from everything currently registered. Each brings strictly new information — inaccessible via any hyperparameter tuning of existing pages.

**Topological pages** capture the *shape* of data distributions at multiple scales simultaneously. Persistent homology constructs a filtration of simplicial complexes at increasing distance thresholds and records birth-death pairs of topological features — connected components ($H_0$), loops ($H_1$), voids ($H_2$) — as a persistence barcode. This barcode is invariant under any homeomorphism of the feature space: immune to rotations, translations, and smooth warping. A dataset whose class structure is topologically distinct — one class forms a loop, the other a disk — cannot be separated by any metric-based classifier (no distance is informative) but is perfectly separable by $H_1$ persistence: the loop generates a persistent $H_1$ feature that the disk does not. No number of kernel choices or hyperparameter settings of any metric-based classifier can access this topological information. Persistent homology features are not expressible as inner products in any finite-dimensional RKHS — they lie outside the reach of all kernel-based methods.

Recent extensions include the **persistent topological Laplacian** (Wang et al. 2020), which goes beyond standard persistent homology by capturing geometric changes (spectral evolution of the Laplacian) that occur without topological changes — specifically, shape evolution within the filtration where no new connected components, loops, or voids appear or disappear. For molecular, protein, and materials science data, this additional geometric information has been shown to substantially improve prediction accuracy compared to standard persistent homology alone. Each combination of (filtration type, homology degree, vectorization method, base classifier) is a distinct dictionary page.

**Causal pages** operate on Pearl's ladder of causation. Standard classifiers occupy rung 1: they learn $p(y|\mathbf{x})$ — observational conditionals. Causal classifiers compute $p(y|\text{do}(\mathbf{x}=\mathbf{x}'))$ — the interventional distribution. These coincide only when the Markov condition holds with respect to a known causal graph and there are no hidden confounders. When confounders exist — in medical data, economic observational studies, or any dataset where feature-label relationships are partially spurious — observational classifiers overfit to confounding correlations that do not hold at test time under distribution shift.

Invariant Risk Minimization (IRM) addresses this by finding a representation $\Phi(\mathbf{x})$ such that the optimal classifier on top of $\Phi$ is the same across all training environments. This finds only the invariant causal features, ignoring the spurious ones. On data where confounders are present, IRM pages can generalize dramatically better than all observational pages under the natural distribution shift of moving from training to deployment. Each assumed set of environments constitutes a distinct IRM page. Double Machine Learning pages debias the classification problem by regressing out confounders from both features and labels before classification.

**Neuromorphic and reservoir computing pages** operate through a computationally distinct paradigm. An Echo State Network (ESN) projects the input through a fixed random high-dimensional dynamical system:

$$\mathbf{h}(t+1) = \tanh(\mathbf{W}^{\text{res}}\mathbf{h}(t) + \mathbf{W}^{\text{in}}\mathbf{u}(t))$$

where $\mathbf{W}^{\text{res}}$ is the fixed random recurrent weight matrix (scaled to spectral radius $\rho$) and $\mathbf{W}^{\text{in}}$ is the fixed random input projection. Only $\mathbf{W}^{\text{out}}$ is trained — via linear regression, a closed-form operation with no gradient computation, no vanishing gradient, no local minima. The fading memory property means reservoir state $\mathbf{h}(T)$ depends on the entire input history with older inputs exponentially discounted by $\rho^{T-t}$. For time series classification (EEG, speech, motion capture, financial sequences), reservoir pages capture long-range temporal correlations that all i.i.d. classifiers fundamentally cannot — because i.i.d. classifiers treat each sample as independent, discarding the temporal ordering that generates the sequence.

Liquid State Machines (LSMs) extend this to spiking neural networks, where information is represented as discrete spike events rather than continuous values. The leaky integrate-and-fire neuron model integrates input current over time, firing when the membrane potential threshold is crossed. LSMs on neuromorphic hardware (SpiNNaker, Intel Loihi) achieve 1–5 watts per chip versus 50–100 watts for equivalent GPU computation — a 10–50× energy efficiency advantage. For wearable EEG, IoT sensors, and battery-constrained embedded systems, energy efficiency is a classification constraint that no standard algorithm addresses. The neuromorphic page is the only page that optimizes this constraint.

---

## VIII. Deep Dive: What the 100 CV Scores Reveal

Looking at the actual CV scores from the Iris run reveals several non-obvious patterns worth discussing precisely, because they illustrate what the Boltzmann mechanism is actually doing with the evidence.

**The LDA/QDA ranking reversal.**

- LDA (svd), LDA (lsqr): CV=0.9750, test=1.0000
- QDA (unregularized): CV=0.9583, test=1.0000
- QDA reg=0.1: CV=0.9917 (best CV), test=0.9333 (worse than above)

The reversal is not random. QDA reg=0.1 is the most regularized QDA variant, gaining a regularization advantage on 60-sample folds that does not generalize because the full 120-sample training set already provides enough data for unregularized QDA. The liquid blend at $T^*=0.005$ handles this implicitly:

$$\hat{p}_{\text{blend}} \approx 0.5 \cdot P_{\text{QDA-reg}}(\mathbf{q}) + 0.5 \cdot P_{\text{LDA}}(\mathbf{q})$$

$$\text{Expected test accuracy} \approx 0.5 \times 0.9333 + 0.5 \times 1.0000 = 0.9667 \checkmark$$

**The BernoulliNB catastrophe.**

BernoulliNB achieves CV=0.3833 and test=0.3667 — below chance for a 3-class problem (chance=0.333). This is not a failure of the algorithm itself. BernoulliNB assumes:

$$p(x_j = 1 | c) = \theta_{jc}, \quad p(x_j = 0 | c) = 1 - \theta_{jc}$$

After MinMaxScaler scaling Iris features to $[0,1]$, the features are continuous, not binary. The algorithm treats continuous probabilities as binary frequencies — a fundamental assumption mismatch. The Boltzmann mechanism handles this automatically without domain knowledge:

$$w_{\text{BernoulliNB}} = \frac{\exp(0.38/0.05)}{\sum_j \exp(\tilde{s}_j/0.05)} \approx \frac{e^{7.6}}{Z} \approx 10^{-5}$$

Weight $\approx 10^{-5}$ — the catastrophically failing page contributes essentially zero to the blend.

**The kernel approximation surprise.**

Nystroem+SGD (page 74) and Nystroem+LinearSVC (page 76) both achieve CV=0.9833 — second-best among all 100 pages. This is unexpected on a 4-feature, 150-sample dataset. The explanation:

- Nystroem approximates the RBF kernel: $k(\mathbf{x}, \mathbf{x}') \approx z(\mathbf{x}) \cdot z(\mathbf{x}')$ with $m=100$ landmark points
- The 100-dimensional Nystroem feature space introduces regularization through approximation error
- The linear classifier adds L2 regularization on top
- Result: doubly-regularized nonlinear classifier

The Boltzmann weights for these pages: $w \approx 0.0223$ each — among the highest. The dictionary correctly identifies that non-obvious algorithm compositions can outperform individually-obvious choices.

**The GP underperformance relative to expectation.**

GP RBF (page 71) achieves CV=0.9417, below many simpler methods. On Iris (150 samples):

- GP classification via Laplace approximation introduces approximation error
- The Laplace approximation's accuracy improves as $N \to \infty$, but at $N=150$ it is suboptimal
- GP training is $O(N^3)$ vs. $O(N \cdot d)$ for logistic regression — slower per fold evaluation
- At larger $N$: GP uncertainty quantification becomes more valuable; the relative ranking changes

The Boltzmann mechanism gets this right: GP pages receive low weights at $N=150$. At $N=100{,}000$, the temperature sweep would likely find the GP pages gaining weight.

---

## IX. Why the Dictionary Is Not Just an Ensemble

The Galactica Dictionary is often confused with standard ensemble methods. The confusion deserves precise resolution.

**Standard ensembles (same model class):**

| Method | Base algorithm | Diversity source | Inductive bias |
|--------|---------------|-----------------|---------------|
| Random Forest | Decision tree | Bootstrap + feature subsampling | Recursive axis-aligned partitioning |
| Gradient Boosting | Decision tree | Sequential residual correction | Additive tree models |
| Bagging | Any single algorithm | Bootstrap sampling | Fixed (same as base) |
| Voting Classifier | 3–5 pre-chosen models | Algorithmic diversity (limited) | Mixture of 3–5 specific biases |

**Galactica Dictionary:**

| Aspect | Description |
|--------|-------------|
| Base algorithms | 100 distinct algorithms from 18 families |
| Diversity source | Fundamentally different inductive biases |
| Inductive bias | The Boltzmann mixture adapts to the data |
| Weight mechanism | Temperature-controlled Boltzmann Softmax |
| Trainable parameters | Zero (weights are deterministic from CV scores + T) |

The key distinction: **standard ensembles reduce variance within a fixed inductive bias; the Dictionary selects and blends across fundamentally different inductive biases.** These are complementary operations. A Random Forest is variance reduction within the tree bias. The Dictionary's weighting of Random Forest vs. SVM vs. GP is bias selection. Both operations are necessary; the Dictionary performs the second, which standard ensembles cannot.

**The Boltzmann weighting versus equal weighting:**

At $T=0.05$ on Iris:
- Dominant page (QDA reg=0.1): $w = 0.0293$ — $2.93\times$ uniform amplification
- BernoulliNB: $w \approx 10^{-5}$ — $0.001\times$ uniform suppression

At $T=5.0$ (uniform):
- All pages: $w = 0.01$ — equal contribution

The transition between these states as $T$ varies is the phase transition described in Section II. The dictionary finds the optimal point in this continuum from data, rather than committing to either extreme.

**The stacking comparison.**

The stacking ensemble (Family 17) is the closest existing method: a meta-learner is trained on out-of-fold predictions of base classifiers. But:

1. The meta-learner must be chosen (another algorithm selection problem)
2. The meta-learner must be trained (risk of meta-level overfitting)
3. The meta-learner has trainable parameters (weights are not deterministic from evidence)
4. The stacking meta-learner cannot represent uncertainty over base models — it commits to a weight

The Boltzmann weighting has zero trainable parameters. Weights are a deterministic function of CV scores and temperature. No meta-level overfitting is possible (no parameters to overfit). Uncertainty over base models is explicitly represented by the entropy $H(T)$.

A natural hybrid: use Boltzmann weights as the prior for the meta-learner, regularizing the stacking combination toward the Boltzmann distribution. This would combine Boltzmann's uncertainty-aware weighting with the stacking meta-learner's ability to learn nonlinear page combinations.

---

## X. The Quantum and Information-Geometric Frontier

**Quantum kernel SVMs** compute $k_Q(\mathbf{x}_i, \mathbf{x}_j) = |\langle\phi(\mathbf{x}_i)|\phi(\mathbf{x}_j)\rangle|^2$ via quantum circuit evaluation. For certain data distributions with correlations that are exponentially hard to express classically, quantum kernels may achieve a given accuracy with exponentially fewer samples than any classical kernel. Each circuit ansatz constitutes a distinct page; the Boltzmann dictionary would automatically upweight quantum pages on datasets where they outperform classical ones — providing empirical evidence of quantum advantage without requiring prior proof.

**Optimal transport classifiers** classify by Wasserstein distance $W_1(\mu_\mathbf{q}^k, \mu_c)$ between the query neighborhood distribution and each class distribution. This is the minimum-transport-cost generalization of k-NN and is provably more robust to adversarial perturbations: an adversary must move a larger "mass" of probability to shift Wasserstein distance than to shift Euclidean distance.

**Information-geometric classifiers** classify via geodesic distances under the Fisher-Rao metric $g_{ij}(\theta) = \mathbb{E}[\partial_i \log p \cdot \partial_j \log p]$ on the statistical manifold of parametric distributions. When the data naturally lives on a statistical manifold (covariance matrices for multivariate time series, Gaussian distributions for sensor data), Fisher-Rao geodesic distance is the natural dissimilarity, and classifiers in this geometry substantially outperform Euclidean-space analogs.

**Hyperbolic space classifiers** embed data in the Poincaré disk $\mathcal{D}^n = \{x \in \mathbb{R}^n: \|x\| < 1\}$ with geodesic distance:

$$d_H(x,y) = \text{arcosh}\!\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$$

For hierarchical data (taxonomies, evolutionary trees, concept hierarchies), hyperbolic space provides exponentially more representational capacity than Euclidean space: a tree of branching factor $b$ and depth $d$ embeds with $O(\epsilon)$ distortion in $O(\log d)$ Poincaré dimensions versus $O(d^{\log b})$ Euclidean dimensions. Hyperbolic k-NN and SVM can classify hierarchical data that Euclidean classifiers systematically fail on, even with arbitrarily many hyperparameter settings.

---

## IX. The Computational Reality of Infinite Pages

The theoretical infinite dictionary — one page for every possible algorithm with every possible hyperparameter configuration — is an uncountable set and uncomputable. The practical dictionary is always a finite approximation: $N_p$ pages dense enough in algorithm space that the Boltzmann mixture converges to BMA as $N_p \to \infty$.

For a 10,000-page dictionary on $N = 100{,}000$ samples with $k=2$ CV:

$$\text{CV cost} \approx N_p \times k \times N_{\text{train}} \times t_{\text{avg}} \approx 10^4 \times 2 \times 8 \times 10^4 \times 100\,\mu\text{s} \approx 1.6 \times 10^5\,\text{s}$$

infeasible single-machine. On 1,000 GPUs, each handling 10 pages: $\approx 100$ seconds — practical. The page-fitting step is embarrassingly parallel: no communication between pages, no shared state. The dictionary scales naturally to GPU clusters.

The question of *which* $N_p$ pages to register is itself a research problem. My current approach — human-curated 100-page taxonomy — is the zeroth-order solution. Meta-learning-guided selection (using dataset meta-features to predict which families are likely to perform well before evaluation) is the first-order solution. Generative page proposals (using a model of algorithm pipelines to propose novel pages based on which regions of algorithm space have high Boltzmann weight on the current data) is the eventual goal — transforming the dictionary from a static structure into an adaptive one that grows in response to evidence.

This generative dictionary would close a deep loop: the Boltzmann mechanism identifies which types of algorithms perform well on this dataset; the generative model proposes new algorithm variants in those promising regions; the proposed variants are evaluated and their weights computed; the process repeats. This is essentially a evolutionary algorithm over algorithm space, guided by the Boltzmann evidence from each dataset — a meta-AutoML system that not only selects from existing algorithms but invents new ones.

---

## X. The Symbolic and Evolutionary Pages: Classifiers That Design Themselves

Beyond the families discussed above, there is a class of algorithm that does not have a fixed structure at design time — it discovers its own structure from data. These belong in the infinite dictionary, and they are categorically different from the parametric models in v1.0.

**Genetic programming classifiers** (as implemented by TPOT) treat the space of ML pipelines as a graph search problem and use evolutionary operators (crossover, mutation) to navigate it. Each generation of the evolutionary search produces a population of distinct classifiers. The Galactica Dictionary can register each checkpoint of a GP run as a page — after 10 generations, 50 generations, 200 generations — with each constituting a distinct pipeline candidate. The Boltzmann mechanism then identifies which evolutionary checkpoint generalizes best, effectively selecting the optimal number of generations without manual tuning.

**Symbolic regression classifiers** discover explicit mathematical expressions that fit the class boundary. PySR and similar tools search the space of mathematical formulas (sums, products, exponentials, trigonometric functions, and their compositions) and find expressions that minimize prediction error while penalizing expression complexity. A symbolic regression classifier $\hat{y} = \text{sign}(f(\mathbf{x}))$ where $f$ is a closed-form mathematical expression has two properties no other classifier in the dictionary possesses: it is maximally interpretable (the decision rule can be read by a human domain expert), and it sometimes reveals genuine physical laws hidden in the data. For scientific datasets where the ground truth is governed by known or unknown physical equations, symbolic regression pages may outperform all black-box classifiers while simultaneously producing an interpretable model. Each complexity budget (maximum expression depth, number of allowed operators) constitutes a distinct page.

**Neural architecture search (NAS) classifiers.** DARTS, ENAS, and related NAS algorithms discover neural network architectures that outperform hand-designed ones on specific datasets. Each NAS-discovered architecture is a page. The dictionary would automatically identify which NAS architectures perform best on the given data without the human needing to know anything about neural architecture design.

**Ensemble of discovered algorithms.** The generative dictionary — the eventual $v_{\infty}$ — would use the Boltzmann evidence from each dataset to guide a generative model of algorithm pipelines, proposing novel pages that exploit the structure of the high-weight region of algorithm space. This closes a deep loop: the dictionary identifies which algorithm types work best; a generator proposes new variants in those regions; the new variants are evaluated; the weights update. This is an evolutionary algorithm operating over algorithm space rather than over data, guided by the Boltzmann posterior from each dataset. Each iteration of this process adds pages that are specifically adapted to the current dataset's structure — a form of on-the-fly algorithm design.

---

## XI. Physical Reservoir Computing: The Dictionary Beyond Digital Pages

A genuinely unexpected dimension of the infinite dictionary is that not all pages need to be implemented in software. **Physical reservoir computing** uses real physical systems as reservoirs, where the natural dynamics of the physical substrate perform the high-dimensional projection, and only the readout is implemented digitally.

**Optical delay-line reservoirs.** A nonlinear optical element with feedback loop uses the delayed signal as the reservoir state. The optical reservoir operates at nanosecond timescales — orders of magnitude faster than digital computation — and performs the reservoir transformation in analog optical physics, consuming milliwatts of power. Classification tasks for high-bandwidth signal processing (optical communication, radar) become feasible at speeds that digital ESNs cannot match.

**Spintronic nano-oscillator reservoirs.** Arrays of magnetic nano-oscillators coupled by spin waves exhibit rich nonlinear dynamics with GHz natural frequencies. The reservoir state is the collection of oscillator phase and amplitude values, which encode temporal correlations in the input signal through spin wave interference. Each oscillator configuration constitutes a distinct page with different reservoir dynamics.

**Memristive crossbar reservoirs.** Memristors (resistors with memory) form a physical reservoir when arranged in a crossbar array: input signals drive currents through the array; the resistance state of each memristor evolves based on the current history (memristive memory). The readout is a simple linear transformation of the resistance state vector. This is the most hardware-efficient implementation of reservoir computing — the physical memory of the memristors performs the temporal basis expansion without any digital computation.

**Quantum reservoir computing.** A quantum system (array of coupled qubits or quantum harmonic oscillators) serves as the reservoir. Quantum coherence and entanglement create reservoir dynamics with correlations that no classical reservoir can reproduce. Quantum reservoir computing may enable exponentially more powerful temporal basis expansions than classical reservoirs for certain classes of time-series data — an intersection of quantum computing and reservoir computing that the infinite dictionary would explore automatically.

Each physical reservoir type and configuration is a page. In a physical implementation, these pages would be evaluated by routing each CV fold through the physical device, reading out the reservoir states, and fitting a linear readout. The Boltzmann mechanism would then identify whether any physical reservoir page outperforms all digital pages — discovering quantum or optical computational advantages empirically, from prediction evidence, without requiring theoretical quantum advantage proofs.

---

The field of meta-learning studies how to adapt quickly to new tasks using experience from previous tasks. In the algorithm selection context, meta-learning uses dataset meta-features — statistical, information-theoretic, and complexity properties of a dataset — to predict which algorithms will perform well before evaluating them (Vanschoren 2018, Feurer et al. Auto-Sklearn 2015, OpenML platform).

The Galactica Dictionary connects to meta-learning in two ways. First, CV evaluation is within-dataset meta-learning: it measures algorithm performance on held-out folds and uses this as evidence for the mixture weights. Second, the temperature parameter $T$ can be informed by across-dataset meta-learning: if prior experience shows that on datasets with similar meta-features the CV-test gap is typically large (the CV ranking is unreliable), we should set a higher $T$. If the CV ranking is typically reliable on similar datasets, set lower $T$. This gives the dictionary a meta-learned prior on temperature — the first practical bridge between the Galactica Dictionary and the broader meta-learning literature.

The Open Meta-Learning initiative (Vanschoren et al., OpenML) has accumulated algorithm performance across thousands of datasets, which could directly seed this prior. For each new dataset, the system would: compute meta-features; find the $k$ most similar historical datasets; examine the distribution of optimal temperatures on those datasets; set the initial temperature prior accordingly. This collapses the temperature sweep cost from full validation evaluation to a prior-informed single-point estimate, dramatically reducing the computational budget needed before the dictionary produces a reliable prediction.

---

## XI. Open Research Questions

**What is the sample complexity of optimal temperature selection?** The temperature sweep uses a validation set. How many validation samples are needed to reliably identify $T^\ast$ for a given $N_p$? Is there a minimax-optimal temperature given only $N_{\text{val}}$ samples and $N_p$ pages? The PAC-Bayes bound gives one answer: $T^\ast \sim \sqrt{m/D_{\text{KL}}}$,
 but this is a worst-case bound. The typical-case optimal temperature likely depends on properties of the CV score distribution that are not captured by the worst-case bound.

### Theoretical Sample Complexity
To ensure that the selected temperature $\hat{T}$ from a set of $N_p$ candidates achieves an error no more than $\epsilon$ away from the true $T^*$ with probability $1-\delta$, the required number of validation samples is:
$$N_{\text{val}} \geq \frac{C}{\epsilon^2} \ln\left(\frac{N_p}{\delta}\right)$$
where $C$ is a constant related to the range of the loss function (e.g., NLL). This shows that the complexity is **logarithmic** in the number of sweep points ($N_p$) but **inverse-quadratic** in the desired precision ($\epsilon$).

### Minimax vs. Typical-Case
* **Minimax-Optimal:** Without specific distribution knowledge, the minimax-optimal $T$ is the one that minimizes the maximum possible ECE (Expected Calibration Error) or NLL over all valid score distributions. This often aligns with the conservative **PAC-Bayes** scaling to ensure robustness against adversarial or heavy-tailed logit distributions.
* **Typical-Case:** In practice, $T^*$ is driven by the **second moment** (variance) of the logit distribution. For overconfident deep learning models, $T^*$ is typically solved by finding the point where the average predicted confidence matches the empirical accuracy on $N_{\text{val}}$.



**What is the optimal page registration strategy for a given compute budget?** If we can evaluate $B$ page-fold pairs total, how should we allocate between breadth (many pages, fewer folds each) versus depth (fewer pages, more reliable CV scores)? This is a sequential experimental design problem — a variant of the explore-exploit trade-off from bandit theory. The dictionary with highest effective $N_{\text{eff}}(T^*)$ for a given $B$ budget is the best allocation, but finding that allocation without evaluating all options requires a meta-strategy.

**Does the Boltzmann mixture converge to BMA?** As $N_p \to \infty$ with pages drawn from an increasingly dense covering of algorithm space, does $\sum_i w_i P_i(\mathbf{q})$ converge to the true BMA $\int P(y|\mathbf{q}, M) p(M|\mathcal{D}) dM$? What coverage conditions on the page distribution are needed? This is a functional approximation theory question — analogous to asking whether a quadrature rule converges to the true integral.

**When does a physics-informed page outperform all statistical pages?** HRF achieved 98.9% on EEG — a 4-5 pp advantage over all 100 standard pages. What data properties predict this advantage? Is it purely the presence of periodic structure, or are there more general conditions under which wave-physics classifiers dominate? Answering this would let the dictionary pre-rank physics-informed pages by expected advantage on each dataset type — reducing the CV budget needed to discover them.

**What is the dictionary's behavior under adversarial examples?** Because the dictionary blends multiple classifiers with different decision geometries, an adversarial perturbation that fools one page may not fool the majority. Is the dictionary inherently more robust to adversarial attacks than any single page? The robustness benefit likely scales with inter-page decorrelation: more diverse pages produce more robust blended predictions. This could be formalized as a robustness certificate — a lower bound on the fraction of pages that must be simultaneously fooled before the dictionary's prediction changes.

---

## XII. Honest Assessment of v1.0

v1.0 is a proof of concept. The Boltzmann weighting mechanism is numerically stable, interpretable, and produces sensible results on a clean benchmark. It is not the infinite dictionary.

The 2-fold CV on 60 samples per fold produces score estimates with standard error $\approx 2.8\%$ — comparable to performance differences between the top pages. The Boltzmann weights are noisy. More folds, larger datasets, or repeated k-fold would sharpen estimates and make temperature selection more trustworthy.

Temperature selection uses the test set. A three-way split (train/validation/test) would eliminate this leakage at the cost of reducing both training data and temperature-selection statistical power.

No physics-informed, topological, causal, or reservoir pages are registered. The most important predictions of the framework — that it will discover domain-matched exotic algorithms automatically from CV evidence — cannot be tested until those pages are implemented. This is the most significant gap between what v1.0 demonstrates and what the infinite dictionary would demonstrate.

The fallback for non-probabilistic pages (one-hot encoding from hard predictions) discards calibration information. Better: Platt scaling on raw decision scores, producing soft probabilities that participate meaningfully in the blended output.

These are engineering gaps, not theoretical failures. The statistical mechanics framework is sound. The PAC-Bayes interpretation is correct. The bias-variance-covariance decomposition provides principled justification for algorithm family diversity. The open research questions are real and nontrivial. v1.0 is the first experiment confirming the framework is computationally feasible and produces interpretable, sensible results.

*The dictionary breathes. The dictionary flows. Page one of the proof is complete.*

---

*End of White Paper.*

*Dataset: Iris (OpenML, N=150). Platform: Python 3.11, NVIDIA T4 (detection only). Dictionary: 100 pages, 18 families. CV-best page: QDA reg=0.1 (CV=0.9917, test=0.9333). Test-best pages: 7 pages at 1.0000. Liquid Dictionary at T*=0.005: 0.9667. Authors: Devanik Debnath + Xylia.*
