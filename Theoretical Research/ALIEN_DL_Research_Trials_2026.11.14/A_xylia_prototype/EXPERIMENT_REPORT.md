# ALIEN-DL exploratory benchmark, first test (Xylia prototype, trial A)

**Run date:** 2026-11-14
**Purpose:** first small-data feasibility comparison; not a claim of general superiority.

## Prototype implemented

ALIEN-DL uses eight dendritic-style compartments. Each incoming connection has a trainable synaptic efficacy, a share in a per-compartment resource simplex, short-term resource/facilitation state, and a structural-persistence gate. The three independently switched couplings are:

- **R:** allocation share controls synaptic resource recovery.
- **P:** allocation share gates the synaptic efficacy update.
- **A:** allocation share updates a structural-persistence gate, with thresholded pruning available.

The allocation signal is a normalized first-order contribution heuristic, `-q * dL/dq`, updated once per epoch using a multiplicative-weights step. It is not the exact gradient of loss with respect to the resource budget.

## Main comparison

Train/validation/test split: 70% / 15% / 15%, stratified. Standardization was fitted on the training partition only. Three independent split/initialization seeds were used: 7, 21, 42. The best validation-accuracy checkpoint was evaluated on the untouched test partition. The MLP parameter-matched baseline was selected to have approximately the same number of trainable parameters as ALIEN-DL.

| Dataset | Model | Test accuracy (mean ± SD) | Macro-F1 (mean ± SD) | Trainable parameters | Mean train time (s) |
|---|---|---:|---:|---:|---:|
| Wine | ALIEN-DL | 0.8519 ± 0.0741 | 0.8414 ± 0.0828 | 243 | 0.164 |
| Wine | MLP, parameter-matched | 0.9259 ± 0.0370 | 0.9266 ± 0.0370 | 241 | 0.046 |
| Wine | MLP, width 32 | 0.9012 ± 0.0771 | 0.9025 ± 0.0761 | 547 | 0.042 |
| Breast Cancer | ALIEN-DL | 0.9845 ± 0.0134 | 0.9833 ± 0.0145 | 506 | 0.599 |
| Breast Cancer | MLP, parameter-matched | 0.9845 ± 0.0134 | 0.9836 ± 0.0142 | 497 | 0.157 |
| Breast Cancer | MLP, width 32 | 0.9845 ± 0.0134 | 0.9836 ± 0.0142 | 1,058 | 0.120 |
| Digits | ALIEN-DL | 0.9383 ± 0.0113 | 0.9385 ± 0.0117 | 1,122 | 2.678 |
| Digits | MLP, parameter-matched | 0.9605 ± 0.0130 | 0.9604 ± 0.0130 | 1,135 | 0.424 |
| Digits | MLP, width 32 | 0.9679 ± 0.0057 | 0.9678 ± 0.0057 | 2,410 | 0.461 |

## 2x2x2 coupling ablation on Wine

R, P and A were independently enabled or disabled. For each seed, all eight configurations used the same initialization and minibatch order, making the comparisons paired. Accuracy is the mean across three seeds.

| R | P | A | Mean test accuracy | SD |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.9136 | 0.0855 |
| 0 | 0 | 1 | 0.9012 | 0.1069 |
| 0 | 1 | 0 | 0.9259 | 0.0980 |
| 0 | 1 | 1 | 0.9136 | 0.1191 |
| 1 | 0 | 0 | 0.9136 | 0.0855 |
| 1 | 0 | 1 | 0.8889 | 0.0980 |
| 1 | 1 | 0 | 0.9259 | 0.0980 |
| 1 | 1 | 1 | 0.9136 | 0.1191 |

Paired interaction contrasts were: R x P = 0.0000, R x A = -0.0123, P x A = 0.0000. The three-way contrast was approximately +0.0123 (SD 0.0214 across three seed-level contrasts). With three seeds and a 27-example test set, these values are descriptive only and do not establish a positive interaction or support superadditivity.

## Diagnostics and interpretation

- Allocation competition became non-uniform. Mean allocation Gini for the selected full-ALIEN checkpoints was approximately 0.22 (Wine), 0.35 (Breast Cancer) and 0.64 (Digits).
- The selected checkpoints in the main comparison retained all candidate connections (`active_fraction = 1.0`). In the Wine factorial, some A-enabled configurations pruned a small fraction of edges, while the full R=P=A=1 condition retained all edges. Pruning is implemented but not yet a consistent outcome of the full architecture.
- ALIEN matched the parameter-matched MLP on Breast Cancer, and trailed it on Wine and Digits.
- ALIEN was approximately 3.6x, 3.8x and 6.3x slower than the parameter-matched MLP on Wine, Breast Cancer and Digits, in this CPU run.
- No energy, GPU or deployment-efficiency claim is supported.

## Limitations

1. Three seeds and one fixed protocol per dataset; uncertainty is high, especially on Wine.
2. A compact tabular prototype, not a full biological neural simulation.
3. Synaptic r and u states are reset for each independent example. No long-horizon sequence memory or online continual learning is tested.
4. The micro-steps apply a static input repeatedly; they are not a temporal stimulus benchmark.
5. Structural births are not implemented, and hard pruning did not occur in the selected checkpoints.
6. The allocation utility is a heuristic contribution score.
7. Optimizer and initialization are not exhaustively tuned.

## Current conclusion

The resource-allocation state can become non-uniform and can be integrated into a differentiable feedforward classifier. It does not yet outperform a parameter-matched MLP consistently, does not validate superadditive coupling, and incurs extra runtime.
