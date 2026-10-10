# Heteroplasmy Phase Control: Threshold-Aware Mitophagy Scheduling as a Stochastic Control Problem for Mitochondrial Aging

**Author:** Devanik Debnath (Devanik21)  
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India  
**Release date:** 10 October 2026  
**Status:** Theoretical methods preprint with synthetic numerical checks; not peer-reviewed by virtue of archival.

## Abstract

The manuscript asks whether the timing of a hypothetical mutant-selective mitochondrial quality-control actuator can reduce time above a measured heteroplasmy threshold at fixed total dose. In a reduced selection model, an exact log-odds transform proves that a maximally front-loaded schedule weakly minimizes any time-integrated nondecreasing heteroplasmy penalty, provided the effective selectivity is constant and positive. The deterministic endpoint is unchanged by schedule when cumulative dose is fixed. Stochastic diffusion and first-passage endpoints are proposed for empirical testing. The work does not establish a biological actuator, universal pulse advantage, restored mitochondrial function, or lifespan extension.

## Main equations

The reduced deterministic dynamics are

$$
\dot{x}=x(1-x)(s-\Delta u), \quad 0\leq u(t)\leq U.
$$

The log-odds transformation makes cumulative dose explicit:

$$
\operatorname{logit}x(t)=\operatorname{logit}x(0)+st-\Delta\int_0^t u(\tau)\,d\tau.
$$

A stochastic approximation used for synthetic auditing is

$$
dX_t=X_t(1-X_t)(s-\Delta u(t))dt+\sqrt{\frac{X_t(1-X_t)}{N_e}}\,dW_t.
$$

## Key result and scope

At equal cumulative input dose $B$, the front-loaded control $u_F(t)=U$ for $0\leq t<B/U$ and zero afterward maximizes cumulative dose at every intermediate time. If $\Delta>0$ is constant, the theorem proves $x_F(t)\leq x_u(t)$ for every competing schedule, equality of deterministic final states, and no greater integrated penalty for any nondecreasing loss. The result depends on the model's restrictive assumptions. It does not imply that pulsed mitophagy works in vivo; toxicity, delayed pharmacodynamics, compensatory biogenesis, state-dependent selectivity, or negative/zero selectivity can invalidate or reverse the ranking.

## Reproducibility

Requirements: Python 3.10 or later and NumPy 1.24 or later.

```bash
python verify_model.py
```

The random seed is `20261010`. The script checks the exact logit solution, endpoint invariance at equal dose, front-loaded dominance across 250 random feasible schedules, the zero- and negative-selectivity edge cases, threshold-burden ordering, and boundedness of 6,000 stochastic trajectories. The stochastic step uses Euler-Maruyama with projection onto `[0,1]`; this boundary convention is disclosed and is not an exact biochemical model. The printed stochastic statistics are synthetic numerical diagnostics and may not be interpreted as experimental evidence.

## P1-P5 claim classification

| Class | Statement | Status |
|---|---|---|
| P1 | Log-odds transformation and fixed-dose dominance theorem | Exact within the declared deterministic model |
| P2 | Mutant-selective mitochondrial clearance is feasible in a target tissue | Unknown; requires direct measurement of selectivity and safety |
| P3 | Rising cross-cell variance is a universal early-warning signal | Not claimed; proposed as a context-specific, testable prediction |
| P4 | Threshold-aware scheduling may improve time-above-threshold metrics | Conditional theoretical consequence; requires actuator validity and external validation |
| P5 | Pulsed mitophagy slows aging or extends lifespan | Unsupported and not claimed |

## Experimental gates and falsification

First estimate mutant-versus-wild-type selection under the proposed actuator; total mitophagy flux is not a substitute. If the estimated effect `Delta` is nonpositive or indistinguishable from zero, the actuator fails the model's central gate. Next calibrate a mutation- and cell-specific functional threshold, then compare constant and front-loaded schedules at matched cumulative input in randomized cybrid experiments. Primary outcomes should include threshold burden, final heteroplasmy, and a prespecified oxidative-phosphorylation endpoint, while monitoring mtDNA copy number, membrane potential, mitochondrial mass, cell survival, and collateral loss of wild-type genomes. The pulsed schedule is falsified in a context if the prespecified threshold-burden reduction is absent with uncertainty narrow enough to rule out the minimum relevant effect. A null result with low power is inconclusive.

In vivo work is downstream of cell-level actuator validation. PolgA D257A mice model elevated mtDNA mutagenesis and premature-aging phenotypes, but are not automatically a model of one mutant heteroplasmy or a genotype-selective clearance mechanism. Any animal protocol requires independent ethical review and formal power analysis. This release does not give a dosing protocol or claim animal validation.

## What would change the conclusion?

A time-delayed actuator state, state-dependent `Delta`, explicit mtDNA biogenesis, toxicity, or competing functional costs may make continuous or feedback schedules preferable. A separate bistable potential model with measured parameters would be needed for Kramers-style escape-time arguments. The proposed variance warning must add out-of-sample predictive value beyond mean heteroplasmy, mtDNA copy number, cell type, sequencing depth, and technical batch. Failure of any of these tests would narrow or reject the corresponding claim.

## Bibliography and novelty audit

The paper cites recent reviews of heteroplasmy dynamics and biochemical thresholds; single-cell evidence for age-associated somatic mutations; work on the limitations of potential-dependent mitochondrial quality control; the PolgA D257A mutator model; DdCBE mitochondrial base editing; and adaptive therapy as a distant analogy for schedule design. A targeted search used “heteroplasmy threshold control,” “mitophagy optimal control,” “mitochondrial heteroplasmy dynamics,” and “adaptive therapy mitochondria,” including PubMed/arXiv-oriented queries. Related work exists; the search does not prove originality or priority. An updated systematic search and domain-expert review are necessary before submission.

## Citation and archival status

A `CITATION.cff`, BibTeX file, Zenodo-ready metadata template, and Schema.org record are supplied. These are metadata files, not evidence that a DOI has been minted, a Zenodo deposit exists, or the manuscript has been indexed. No external submission or publication is claimed.
