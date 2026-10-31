# Tame Moduli of Life

**Sharp O-Minimal Finiteness for Repair-Closed Systems and Tropical Walls over the Landscape of Physical Laws**

**Preprint v1.0 · 31 October 2026 · Devanik Debnath**
Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India · ORCID: [0009-0001-8310-6921](https://orcid.org/0009-0001-8310-6921)

[Read the PDF](paper.pdf) · [LaTeX source](paper.tex) · [Rebuild and verify](verify_model.py) · [Bibliography](paper.bib)

## Central question

Can a museum of every possible life form, in every possible universe, be finite?

The paper gives a carefully delimited mathematical answer. For a **fixed finite-dimensional family** of candidate laws, if the repair-admissibility predicate is definable in a fixed o-minimal structure, the admissible parameter region has finitely many connected components. In a suitable sharply o-minimal structure, a format--degree ledger can yield a structure-dependent polynomial bound on topological complexity.

This means finitely many connected regions or cells, **not finitely many points, species, organisms, or possible universes**. A single connected region may contain a continuum of models. The result does not cover arbitrary infinite-dimensional or non-definable families of physical laws.

## Mathematical core

A finite reaction network is represented by a stoichiometric matrix `S`. A positive production witness is a nonnegative flux vector `v` with strictly positive net production on the designated species:

$$
Sv \gg 0, \qquad v \geq 0.
$$

For a finite-dimensional law parameter `theta`, a state set `K_theta`, repair map `R_theta,j`, and damage functional `V_theta`, repair closure is expressed by:

$$
R_{\theta,j}(K_\theta) \subseteq K_\theta,
\qquad
V_\theta(R_{\theta,j}(x)) \leq V_\theta(x)
\quad \text{for all }x\in K_\theta.
$$

When all primitive maps and constraints are definable in the same o-minimal structure, Boolean operations and quantifier projection preserve definability. O-minimal cell decomposition then gives finitely many connected components of the parameter region. Sharp o-minimality can refine this to a format--degree complexity bound, provided the exact structure and complexity propagation rules are specified.

For a finite-support polynomial balance

$$
P(x)=\sum_{\alpha\in A} c_\alpha x^\alpha,
$$

tropicalization gives a piecewise-linear dominant-balance skeleton. Its cells are finite for finite support, but a tropical wall is not automatically an exact positive-real biological phase boundary; sign conditions and lifting hypotheses matter.

## Main result and scope

The formal result is conditional: definability of the model family is the hypothesis, and finite connected-component structure is the consequence. The paper combines this result with an explicit repair-closure predicate, a finite stoichiometric witness, and a tropical overlay. The literature review includes the string-landscape finiteness precedent, sharp o-minimality, autocatalytic-core classifications, o-minimal invariants for discrete-time dynamics, and tropical biochemical methods.

The paper does **not** claim a universal definition of life, a proof that all possible universes form a finite set, an exhaustive novelty result, or an empirical discovery. The MathSciNet and zbMATH record-level negative searches could not be independently completed in this run; no absolute priority claim is made.

## Synthetic model and verification

The released verifier implements a semialgebraic toy family with parameters `a`, `b`, and `c`:

$$
a+b\leq1, \qquad ab\geq c.
$$

It checks a positive stoichiometric witness, a convex repair map with a quadratic contraction identity, a logarithmic binomial wall, a one-dimensional degree/component illustration, and grid-resolution sensitivity. All generated figures and numerical data are synthetic and are included as reproducibility evidence, not as biological measurements.

### Reproduce

Requires Python 3.11+ and `numpy`, `scipy`, `matplotlib`, and `networkx`.

```bash
python verify_model.py
python deploy.py --self-test
python deploy.py --validate-only
```

To rebuild `paper.pdf` from the current `paper.tex` and `paper.bib`, install `pdflatex`, `bibtex8`, `pdfinfo`, and `pdffonts`, then run:

```bash
python deploy.py --build
```

The build is fail-closed: it does not copy a stale PDF when source compilation fails. The verifier's numerical checks establish only the exact identities and synthetic calculations implemented in the code. They do not establish a physical or biological mechanism.

## Figures and evidence

The twelve numbered scientific figures are stored in [`figures/`](figures/). Underlying data, model outputs, literature audit, claims/evidence matrix, environment report, build records, and the publication-quality/visual-review reports are stored in [`results/`](results/).

The visual set includes six or more plots, two network/state graphs, two process diagrams, and additional numerical/sensitivity views. Every figure is generated from `verify_model.py`; the figures are vector PDF with raster previews.

## Files

- `paper.pdf` — compiled 29-page theoretical preprint (see current build and visual-review records).
- `paper.tex` and `paper.bib` — manuscript and compiled bibliography source.
- `verify_model.py` — mathematical checks, data generation, and figure generation.
- `deploy.py` — fail-closed build, validation, and self-test helper.
- `CITATION.cff`, `.zenodo.json`, `schema_article.jsonld` — citation and discovery metadata. No DOI has been minted and no Zenodo deposit is claimed.
- `figures/` — twelve vector figures and raster previews.
- `results/` — reproducibility and review evidence.

## Status

Theoretical preprint; not peer reviewed and not submitted. No external indexing, publication acceptance, or experimental validation is claimed. The research-process credit in the manuscript identifies Xylia as the autonomous multi-model research engine and retains final scientific responsibility with Devanik Debnath.
