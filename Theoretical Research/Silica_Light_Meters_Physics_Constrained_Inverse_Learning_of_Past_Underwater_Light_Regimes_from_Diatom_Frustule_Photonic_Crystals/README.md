# Silica Light Meters

**Physics-Constrained Inverse Learning of Past Underwater Light Regimes from Diatom Frustule Photonic Crystals**  
**Author:** Devanik Debnath · National Institute of Technology Agartala, India  
**Research-process coordination:** Xylia  
**Release date:** 2026-10-18 · **Version:** v2026.10.18  
**Status:** theoretical research proposal plus synthetic pipeline demonstration; not peer reviewed

> **Evidence boundary.** The current release includes no empirical diatom preference data, measured frustule spectra, fitted molecular phylogeny, real sediment core, or paleoclimate reconstruction. `verify_model.py` uses synthetic data and a Gaussian-resonance spectral surrogate, **not RCWA or FDTD**. A successful code run validates the declared software checks only.

## Research question

Does a physics-derived optical-response descriptor predict measured diatom light preference better than raw pore size after pre-specified phylogenetic and habitat controls, using held-out-taxon evaluation? Only if that first test succeeds should an uncertainty-aware inverse model be evaluated on sediment cores with independent light-history constraints.

## Why the gap is narrower than “pore size correlates with light”

Diatom frustules have demonstrated wavelength-dependent optical effects. A 2020 University of Nebraska–Lincoln thesis examined `Aulacoseira crenulata` at one Amazon lake and reported an indirect pore-size/light relationship. Studies of `Fragilariopsis kerguelensis` have already linked morphotypes to temperature-related environmental gradients. Therefore, optical properties and morphology-based environmental inference are not claimed as new. The proposed, still-unproven contribution is the calibrated bridge from measured geometry to spectral response to ecologically measured preference, with explicit phylogenetic control, held-out evaluation, and inverse non-identifiability checks.

## Key equations

A candidate optical forward model maps measured geometry `g_s`, wavelength `λ`, orientation `θ`, and optical constants to frustule response:

$$
\mathcal{T}_s(\lambda,\theta)=\mathcal{M}(g_s,n(\lambda),\theta;\lambda).
$$

A reduced water-column model is:

$$
\mathcal{E}(\lambda,z,t)=\mathcal{E}_0(\lambda,t)\exp[-K_d(\lambda,t)z].
$$

The candidate exposure index is:

$$
Q_s(t,z)=\int \mathcal{E}(\lambda,z,t)\mathcal{T}_s(\lambda,\theta)A_s(\lambda)\,d\lambda.
$$

These expressions define the proposed modelling chain; `verify_model.py` does **not** implement the Maxwell solver or estimate the formulas from real samples.

## Synthetic pipeline snapshot

Running `python verify_model.py` generates hypothetical taxa, surrogate spectra, synthetic water attenuation, three validation worlds (optical signal, phylogenetic confounding, and no direct signal), and eight reproducible vector/raster visual assets. The actual current-run metrics are in [`results/benchmark_summary.json`](results/benchmark_summary.json), the row-level synthetic outputs are in [`results/synthetic_cv_results.csv`](results/synthetic_cv_results.csv), and all assumptions are in [`results/synthetic_generation_spec.json`](results/synthetic_generation_spec.json).

Do not interpret these outputs as measurements or predictions for real diatoms. The clade control is a simulated fixed effect, not a phylogeny. The spectra are a feature-generator test, not a Maxwell-equation calculation.

The fixed-seed simulation runs 90 replicates in each synthetic world. Mean held-out $\Delta R^2$ (spectral surrogate minus raw-pore baseline) was **+0.517** in the intentionally positive optical-signal world, **-0.001** in the phylogeny-confounded world, and **-0.002** in the no-direct-signal world. The associated 2.5th–97.5th replicate quantiles were [+0.281, +0.751], [-0.105, +0.094], and [-0.124, +0.122], respectively. These are synthetic repeat-run quantiles, not empirical confidence intervals. The positive case is built into the simulator by design; only the loss of the advantage under the negative controls is a useful software sanity check here.

## Visual index

| Figure | Purpose | Assets |
|---|---|---|
| 1 | Synthetic geometry-to-spectrum features | [`PDF`](figures/fig01_surrogate_spectra.pdf) · [`PNG`](figures/fig01_surrogate_spectra.png) |
| 2 | Synthetic water-column spectral attenuation | [`PDF`](figures/fig02_attenuation.pdf) · [`PNG`](figures/fig02_attenuation.png) |
| 3 | Held-out predictive comparison under synthetic worlds | [`PDF`](figures/fig03_incremental_prediction.pdf) · [`PNG`](figures/fig03_incremental_prediction.png) |
| 4 | Inverse ambiguity surface | [`PDF`](figures/fig04_inverse_nonidentifiability.pdf) · [`PNG`](figures/fig04_inverse_nonidentifiability.png) |
| 5 | Causal dependency graph | [`PDF`](figures/fig05_causal_graph.pdf) · [`PNG`](figures/fig05_causal_graph.png) |
| 6 | Feature and model dependency graph | [`PDF`](figures/fig06_feature_graph.pdf) · [`PNG`](figures/fig06_feature_graph.png) |
| 7 | Conditional empirical-validation pipeline | [`PDF`](figures/fig07_validation_flow.pdf) · [`PNG`](figures/fig07_validation_flow.png) |
| 8 | Fail-closed posterior/validation gate | [`PDF`](figures/fig08_fail_closed_flow.pdf) · [`PNG`](figures/fig08_fail_closed_flow.png) |

## Reproduce

Requirements: Python 3.10+ with NumPy, Matplotlib and NetworkX; and a LaTeX toolchain (`pdflatex`, `biber`, and `latexmk` or equivalent) to rebuild the manuscript.

```bash
python -m py_compile verify_model.py deploy.py
python verify_model.py
python deploy.py --self-test
python deploy.py --validate-only
```

To rebuild the PDF from the current `paper.tex` and `paper.bib`, run `python deploy.py --build`. The validator refuses to claim a successful release when required files, source references, metadata, figure inventory, PDF pages, or code checks are missing. A syntactically valid PDF does not establish the underlying paleoproxy hypothesis.

## Proposed confirmatory comparison

- **M0:** habitat covariates plus phylogenetic effect.
- **M1:** M0 plus raw pore size, porosity and valve dimensions.
- **M2:** M0 plus pre-registered optical-response features.

The primary contrast is M2 versus M1 in nested held-out-taxon validation, with an external held-out-site/region analysis. The morphology response must be measured independently under controlled light gradients. The phylogeny and feature selection must be frozen before outer test evaluation.

## Falsification conditions

The project should stop or narrow its claim if optical forward simulations fail calibration against measured spectra; optical features do not improve held-out predictive distributions over raw morphology; improvement vanishes after phylogenetic/habitat controls; the result fails held-out-region transfer; posterior estimates vary sharply with plausible attenuation or preservation assumptions; or independent light-history constraints fall outside calibrated predictive intervals. In those cases, report the null result or withhold the core reconstruction.

## Literature and novelty boundary

Key precedents include measured/simulated frustule optics (Ferrara et al., 2014; Chen et al., 2015), wavelength-dependent light capture (2015), the single-lake `A. crenulata` thesis (Welchans, 2020), the `F. kerguelensis` morphotype studies (Kloster et al., 2018; Glemser et al., 2019), diatom depth/light calibration in paleolimnology (Hofmann et al., 2020), and aquatic spectral attenuation models (Weiskerger et al., 2018). The current literature audit is targeted rather than exhaustive; all references should be rechecked against publisher records before any formal submission.

## Main artifacts

- [Compiled paper](paper.pdf)
- [LaTeX source](paper.tex)
- [Synthetic model and figure generator](verify_model.py)
- [Deployment validator](deploy.py)
- [BibTeX references](paper.bib)
- [Citation metadata](CITATION.cff)
- [Zenodo metadata draft — no deposit implied](.zenodo.json)
- [Schema.org article metadata](schema_article.jsonld)
- [Figures](figures/)
- [Research results and audit trail](results/)

## Research-process credit and AI disclosure

Co-equal research-process credit is given to Devanik Debnath and Xylia, the autonomous multi-model research engine coordinating AI-assisted discovery, synthesis, and verification. Devanik Debnath retains responsibility for the final manuscript, scientific claims, citations, and released artifacts.

## Cite this preprint

```bibtex
@misc{Debnath2026SilicaLightMeters,
  author = {Debnath, Devanik},
  title = {Silica Light Meters: Physics-Constrained Inverse Learning of Past Underwater Light Regimes from Diatom Frustule Photonic Crystals},
  year = {2026},
  howpublished = {The Invention Archive, theoretical and synthetic methods preprint},
  url = {https://github.com/Devanik21/The-Invention-Archive}
}
```

No DOI, journal submission, peer review, external deposit, or external indexing is claimed.
