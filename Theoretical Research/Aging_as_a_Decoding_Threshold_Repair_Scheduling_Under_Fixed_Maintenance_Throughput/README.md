# Aging as a Decoding Threshold: Repair Scheduling Under Fixed Maintenance Throughpu

**Author:** Devanik Debnath (Devanik21)<br>
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India<br>
**Date:** 2026-10-24<br>
**Status:** Theoretical/modeling preprint; no empirical lifespan extension or clinical efficacy is claimed.

## Main question

At fixed offered repair throughput, can changing the order of repair tasks change a threshold-defined failure time in a coupled damage model? The paper uses “decoding threshold” as an operational analogy for a functional boundary, not a literal biological error-correcting code.

## Model

For the four-dimensional burden vector $x(t)$, constant source vector $b$, synthetic nonnegative coupling matrix $A$, coupling multiplier $\kappa$, and allocated repair input $u(t)$:

$$
\dot{x}(t) = b + \kappa A x(t) - u(t).
$$

The modeled failure proxy is the first-passage time to the threshold $\theta=8$:

$$
\tau_f = \inf\{t \geq 0 : \mathbf{1}^\top x(t) \geq \theta\}.
$$

The scheduled policy allocates $R=0.12$ to one of four damage classes for three time units per slot, then cycles to the next. All 24 orders are enumerated. Each class receives the same nominal $0.36$ offered dose per 12-unit cycle. Comparison is against continuous equal-share repair at the same total offered capacity, plus a deliberately non-matched 10% budget increase.

## Interpretation boundary

- Every parameter and model output is synthetic; no biological rate was fitted.
- $\kappa$ is an interaction-strength multiplier, not a measured correlation coefficient.
- The threshold time is not organismal lifespan.
- Reinforcement learning and stochastic HJB control were not run; ultrametric clustering and percolation are future hypotheses, not results.
- Prior work already includes optimal control of aging in complex networks (Sun et al., 2020); this release does not claim the broad idea is new.

## Reproduce

Requirements: Python 3.10+, NumPy, Matplotlib, NetworkX. The environment report under `results/environment.txt` records versions used in this run.

```bash
python verify_model.py
```

The verifier regenerates the model sweep, CSV outputs and eight figures and runs checks for the zero-coupling limit, 24-order enumeration, equal-dose matching, invalid inputs and RK4 step halving. This verifies the implementation of the synthetic model only.

## Artifacts

- [PDF manuscript](paper.pdf)
- [LaTeX source](paper.tex)
- [Verification and figure generator](verify_model.py)
- [Bibliography](paper.bib)
- [Candidate comparison](results/candidate_topics.csv)
- [Literature audit](results/literature_audit.md)
- [Claim/evidence matrix](results/claim_evidence_matrix.csv)
- [Full order sweep](results/schedule_sweep.csv)
- [Coupling summary](results/coupling_summary.csv)
- [Repair budget sensitivity](results/repair_budget_sensitivity.csv)
- [Verification report](results/verification_report.json)
- [Research workflow log](results/research_log.md)
- [Figure assets](figures/)

## References

See `paper.bib`; article identifiers are DOI-linked and the related-work statement is intentionally conservative. A GitHub archive is not peer review or a DOI deposit.
