# Rate-Distortion Immunology: Defragmenting the Aging Immune Repertoire

**Author:** Devanik Debnath (Devanik21)

**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India

**Release date:** 11 October 2026

**Status:** Theoretical/methods preprint with synthetic numerical verification; not peer-reviewed by virtue of GitHub archival.

## Abstract

This paper proposes a prior-sensitive framework for measuring the gap between an observed T-cell receptor (TCR) repertoire and a best-coverage selection from an explicitly defined candidate universe under a fixed representational budget. The score is conditional on a target-antigen prior, HLA context, receptor-recognition estimates, candidate pool and protection constraints. The manuscript derives a monotone-submodular baseline under an independence approximation and proposes an AI evaluation protocol. No human cohort was downloaded or analyzed; all numerical results are from a synthetic finite instance. The main empirical claim remains untested: coverage debt should predict vaccine response or infection outcomes beyond standard repertoire diversity and clonality metrics.

## Mathematical core

For candidate receptor set \(S\), target epitopes \(e\), and calibrated recognition probabilities \(r_{ie}\), the conditional-independence approximation gives the target miss probability:

$$
q_e(S)=\prod_{i\in S}(1-r_{ie}).
$$

For target prior \(\pi_e\) and declared loss weights \(\ell_e\), modeled distortion is:

$$
D_\pi(S)=\sum_e \pi_e\ell_e q_e(S).
$$

At a fixed codebook budget \(K\), the best achievable distortion within candidate universe \(\mathcal{C}\) is:

$$
D^*_{\pi}(K;\mathcal{C})=\min_{S\subseteq\mathcal{C},\ |S|\leq K}D_\pi(S).
$$

For an observed baseline panel \(R_K\), the candidate-space-conditional coverage debt is:

$$
\mathrm{CD}_{\pi}=D_\pi(R_K)-D^*_{\pi}(K;\mathcal{C}).
$$

The quantity is **not prior-free or an intrinsic biological constant**. If the pathogen prior, candidate universe or recognition model changes, the value can change. The codebook size \(K\) is not claimed to equal a fixed physiological memory-cell capacity.

## Main contribution and limitations

The formal contribution is a measurable, falsifiable coverage-gap construct, prior sensitivity analysis, a submodular baseline optimizer, and an AI evaluation protocol with donor-/epitope-disjoint testing. Existing work already addresses repertoire diversity, clone expansion, TCR prediction and maximum-entropy models; the novelty claim is limited to this proposed synthesis and is not a claim of absolute priority.

CMV-expanded clones are not assumed wasteful, and clone abundance is never used alone to justify deletion. Human cohort analysis and antigen-specific functional validation have not been performed. The paper is not a clinical intervention proposal.

## Reproducibility

Requirements: Python 3.10 or later and NumPy 1.24 or later.

```bash
python verify_model.py
```

The seed is `20261011`. The script checks the product-form miss model, probability bounds, monotonicity and diminishing returns, exhaustive optimization for a tiny synthetic candidate universe, prior sensitivity, and invalid inputs. It prints the actual synthetic distortion and debt for uniform and focused target priors.

The simulation uses constructed recognition probabilities. It does not simulate human T-cell development, CMV infection, HLA population frequencies, vaccine efficacy, or clinical outcomes. No raw human repertoire files were successfully downloaded in this release.

## Public-data path for an empirical next step

Candidate resources include the AIRR Data Commons / AIRR Knowledge ecosystem, standardized receptor--epitope records from IEDB, and the public dataset “The T cell receptor repertoire captures healthy aging and CMV independently from epigenetic clocks” (Zenodo record 20545037). That 2026 dataset includes CD4/CD8, TRA/TRB clonotype data and older-adult frailty/CMV metadata; it is not represented here as containing vaccine-response outcomes. A suitable independent cohort with measured vaccine or infection endpoints is needed for the main predictive test.

## Falsification conditions

The framing fails as a useful predictive advance if debt does not improve held-out prediction beyond age, sex, CMV status, HLA, sequencing depth, repertoire size, Hill diversity and top-clone mass; if scores are unstable under plausible pathogen priors; or if apparent performance disappears in donor-disjoint, study-disjoint and epitope-disjoint evaluation.

## Release artifacts

- `paper.tex` — mathematical manuscript and literature review.
- `paper.pdf` — compiled version of the checked-in source.
- `verify_model.py` — executable synthetic verification.
- `medium_story.md` — 1,500–2,000-word technical essay draft.
- `x_thread.md` — 12-post academic thread draft.
- `paper.bib`, `CITATION.cff`, `.zenodo.json`, `schema_article.jsonld` — source and discovery metadata.
- `deploy.py` — local verification/build/index/commit/tag helper.

## Research-process credit

Co-equal research-process credit is given to Devanik Debnath and Xylia, the autonomous multi-model research engine coordinating AI-assisted discovery, synthesis, and verification. Devanik Debnath retains responsibility for the final manuscript, scientific claims, citations, and released artifacts.

## Selected references

- Zahid et al. (2026), TCR diversity, repertoire size and clonal expansion, [Frontiers in Immunology](https://doi.org/10.3389/fimmu.2025.1707727).
- IEDB revised adaptive immune receptor data (2026), [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC13252180/).
- Breëns et al. (2026), aging/CMV TCR dataset, [Zenodo](https://zenodo.org/records/20545037).
- Weber et al. (2024), machine learning for TCR binding prediction, [ImmunoInformatics](https://doi.org/10.1016/j.immuno.2024.100040).

No DOI has been minted for this manuscript; the Zenodo DOI cited above identifies an external dataset, not this release.
