# Mortal Computation via Multi-Scale Interoceptive Morphogenesis

**Author:** Devanik Debnath (Devanik21)  
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology, India  
**Role:** Lead AGI and Longevity Researcher  
**ORCID:** https://orcid.org/0009-0001-8310-6921  
**Release date:** 2026-10-04  
**Archive:** https://github.com/Devanik21/The-Invention-Archive

## Abstract

This preprint proposes MILLS (Multi-timescale Interoceptive Learning and Lifespan Selection): an AI architecture in which variational inference, parameter learning, and topology are coupled to a finite internal viability reserve. The agent can prune or regrow computational structure according to predictive utility, interoceptive stress, and resource pressure. The model is explicitly framed as a theoretical engineering hypothesis, not as a claim of biological equivalence or consciousness.

## Core architecture

```text
                         Thermodynamic / noise environment
                                      |
                                      v
                            Interoceptive state z
                                      |
                         +------------+------------+
                         |                         |
                   fast inference             reserve R
                      belief mu                 /   \
                         |                    /       \
                         v                   v         \
                      W learning <------ morphology M
                         |                         |
                         +-----------+-------------+
                                     |
                              top-down goal g
```

The central objective is a weighted combination of variational free energy, homeostatic stress, computational/metabolic demand, connection count, and topology turnover. A separate reserve state integrates resource supply minus demand. Reserve depletion raises the marginal cost of structural complexity.

## Main hypotheses

1. Resource pressure should induce sparsification once the reserve multiplier crosses a structural threshold.
2. Survival margin and task fidelity should form a Pareto trade-off under severe disturbance.
3. Slow topology adaptation should improve resilience to structural lesions when dormant capacity exists.
4. High-level goals should alter lower-level setpoints and therefore topology selection.
5. Violating the timescale separation should increase topology chasing and turnover.

## Epistemological classification

| Level | Role in this release |
|---|---|
| P1 | Exact definitions, variational objective, reserve balance, local edge-selection inequality. |
| P2 | Dynamical and variational selection of weights and topology. |
| P3 | Falsifiable sparsity/survival scaling hypotheses; not assumed universal. |
| P4 | Formal mapping across active inference, morphology, neuromorphic hardware, and stochastic thermodynamics. |
| P5 | Visual resemblance to biological systems is explicitly excluded as evidence. |

## Numerical verification

`verify_model.py` integrates a four-state reduced dynamical system with NumPy/SciPy. A deterministic 120-time-unit sweep is run at three disturbance levels and includes a 25% structural lesion. At `sigma=0.85`, the fixed-topology baseline reaches the absorbing zero-reserve boundary, while MILLS remains viable with `R=0.4926` after contracting to 16.7% active modules at the end of the horizon; the minimum active fraction is 12.5%. Mean modeled demand is `0.0784` versus `0.0870` for the fixed condition, while RMSE rises from `0.1407` to `0.2737`.

These numbers validate the stated reduced model only. They are not measurements of biological energy use and do not establish that mortality produces consciousness or life.

## Repository payload

The release is intentionally isolated under:

`The-Invention-Archive/Theoretical Research/Mortal Computation via Multi-Scale Interoceptive Morphogenesis/`

The directory contains exactly the 11 required core files:

`paper.tex` · `paper.pdf` · `verify_model.py` · `README.md` · `medium_story.md` · `x_thread.md` · `CITATION.cff` · `paper.bib` · `.zenodo.json` · `schema_article.jsonld` · `deploy.py`

## Literature position

The manuscript treats mortal computation as an existing research theme, especially following Ororbia and Friston (2023), and builds a narrower executable framework around multi-timescale interoceptive topology selection. Recent 2026 commentary by Friston and Hohwy makes the mortality/self-evidencing discussion especially timely. Structural pruning in neuromorphic hardware provides an engineering bridge, while morphogenetic homeostasis work motivates the multi-scale architecture.

## Indexing note

`schema_article.jsonld` supplies structured `ScholarlyArticle` metadata for web crawlers. It improves machine-readable provenance but does not guarantee Google Scholar or Semantic Scholar indexing.

## Provenance

The release uses the semantic tag `v2026.10.04` in the deployment script. A cryptographic digest manifest is generated by `deploy.py` for every file except this README, which is intentionally excluded to avoid a self-referential hash cycle.

## Links

- Archive: https://github.com/Devanik21/The-Invention-Archive
- Author ORCID: https://orcid.org/0009-0001-8310-6921
- Medium: https://medium.com/@devanik2005
- Kaggle: https://www.kaggle.com/devanik

**LinkedIn is intentionally excluded from this release and from the deployment pipeline.**

## File SHA-256 manifest

| File | SHA-256 |
|---|---|
| `paper.tex` | `e6065bb9b0e663447224430fd29c89138e1a8ddedfd637f9d813c606f044cd4a` |
| `paper.pdf` | `32274d71376bb834e1fd4ded2e797c75ac724ad1e8e1e4377b2b370342715e32` |
| `verify_model.py` | `9211283cee679ec73335a8e3aa1835faa295d967613395665681b6e8a25a52e1` |
| `medium_story.md` | `cad1b00b897dad9a02db7a16b049aa522099068b2100cffa8f2a1da362db7479` |
| `x_thread.md` | `f2df4e204001dadc1a79e7faaba4fb007276baf33db15992d3eb0b952804f2be` |
| `CITATION.cff` | `00c666fc8938da9dfec39620d03f3e933ffb650193683dda6b64c7b6312804e0` |
| `paper.bib` | `3d3c0ef704e22902504765f64da78acc3e1d7ddc4cf9375872472144cafcdceb` |
| `.zenodo.json` | `b8bb11aaeefed8fad4b25585215e7f96b90493f5a4de0501f1f0e8e42b9dea40` |
| `schema_article.jsonld` | `205e0660b5d4328be389474766465e9997b2f298555a4f885da7d66ebc11702e` |
| `deploy.py` | `1c1f5a6201609f58d29d2f980ceba458090ab6cbff85c641926030460e898a7c` |
