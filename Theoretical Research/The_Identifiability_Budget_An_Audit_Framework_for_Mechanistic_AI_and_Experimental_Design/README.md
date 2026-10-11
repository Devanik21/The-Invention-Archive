# The Identifiability Budget: An Audit Framework for Mechanistic AI and Experimental Design

**Release date:** 2026-10-11  
**Status:** Research preprint; not peer reviewed  
**Author:** Devanik Debnath (NIT Agartala, ECE; affiliation reported by the author)  
**Research-process credit:** Xylia coordinated AI-assisted discovery, synthesis, and verification; the author retains responsibility for the final claims and artifacts.

## Abstract

Scientific AI can achieve strong predictive accuracy without identifying a unique mechanism. This release formalizes observational equivalence and proposes an identifiability-budget audit: under a finite candidate set and independent Gaussian observations with known, mechanism-invariant covariance, choose intervention replicates so pairwise KL divergences are sufficient for a finite-candidate model-selection error bound. A toy example uses \(f_0(u)=u\) and \(f_1(u)=u^2\), which agree on the observed settings \(u\in\{0,1\}\) but disagree at \(u=2\). The contribution is a bounded audit framework and executable demonstration, not a claim to have invented active experimental design or observational equivalence.

## Mathematical core

For two candidate mechanisms whose predicted mean observations are \(\mu_i(u)\) and \(\mu_j(u)\), with common known observation covariance \(\Sigma_k\), the cumulative pairwise information from experiment plan \(E=(u_1,\ldots,u_N)\) is

$$
D_{\mathrm{KL},ij}(E)=\frac12\sum_{k=1}^N [\mu_i(u_k)-\mu_j(u_k)]^\top \Sigma_k^{-1}[\mu_i(u_k)-\mu_j(u_k)].
$$

Under the stated Gaussian assumptions, the equal-prior pairwise error is

$$
P_e(i,j\mid E)=\Phi\left(-\sqrt{D_{\mathrm{KL},ij}(E)/2}\right).
$$

For \(K\) candidate mechanisms, a sufficient condition for maximum-likelihood selection error at most \(\epsilon\), uniformly over the finite candidate class, is

$$
D_{\mathrm{KL},ij}(E)\ge 2\left[\Phi^{-1}\left(1-\frac{\epsilon}{K-1}\right)\right]^2
\quad\text{for every } i\ne j.
$$

This is a finite-model statistical guarantee, not evidence that the candidate class includes the true mechanism. It can be invalidated by wrong candidate means, covariance mismatch, dependent samples, or candidate-set misspecification.

## Numerical example

The toy responses are \(f_0(u)=u\) and \(f_1(u)=u^2\) with Gaussian noise of standard deviation 1. On \(u=0\) and \(u=1\), the means match exactly, so observations from that regime cannot discriminate the mechanisms. At \(u=2\), the means are 2 and 4; each independent replicate contributes KL divergence 2. Three replicates yield total divergence 6 and analytic equal-prior error \(\Phi(-\sqrt3)\approx 0.0416\). The script checks these identities and runs deterministic Monte Carlo tests.

## Reproduce

Requirements: Python 3.10+ (verification uses only the standard library); `pdflatex` and `pdfinfo` are used to rebuild and inspect the PDF. The reference list is embedded in `paper.tex` for a reproducible build without a BibTeX executable; `paper.bib` is supplied as a reusable bibliography source.

```bash
python verify_model.py
python deploy.py --check-only
```

To perform a full local-clone deployment (after reviewing the changes), run:

```bash
python deploy.py
```

The deployment script checks all 11 core files, runs the numerical verification, recompiles the PDF, validates JSON metadata, updates the research index without deleting other entries, and stages only this release and the index. It refuses an existing destination directory and uses a non-force push. Run it from a clean clone of `https://github.com/Devanik21/The-Invention-Archive` on an up-to-date `main` branch.

## Files

- `paper.tex`, `paper.pdf`: manuscript and compiled preprint.
- `verify_model.py`: independent analytical and Monte Carlo verification.
- `medium_story.md`: 1,500–2,000-word publication draft.
- `x_thread.md`: 12-post dissemination draft.
- `paper.bib`: bibliography used for the manuscript.
- `CITATION.cff`, `.zenodo.json`, `schema_article.jsonld`: citation and archival metadata, without a minted DOI.
- `deploy.py`: local validation and safe deployment helper.

## Epistemic status and limitations

The observational-equivalence proposition is exact. The Gaussian error formula and the finite-candidate union bound are exact only under their declared assumptions. The identifiability budget is an operationalization proposed in this preprint. The simulated example validates the calculation, not any biological, physical, or clinical mechanism. No wet-lab or clinical experiments were performed, and no external indexing or DOI is claimed.

## Literature screened

- McCormick (2026), *Position: Prioritize Identifying Structure, Not Complex Models, for Scientific Discovery*, ICML / PMLR 306: [paper](https://proceedings.mlr.press/v306/mccormick26a.html).
- Zhang et al. (2023), *Active learning for optimal intervention design in causal models*, Nature Machine Intelligence: [paper](https://www.nature.com/articles/s42256-023-00719-0).
- Liu, Maini, and Baker (2026), *Optimal experiment design for practical parameter identifiability and model discrimination*, Mathematical Biosciences 399, 109710: [DOI](https://doi.org/10.1016/j.mbs.2026.109710).
- Liu, Wanika, Chappell, and Branke (2025), *Efficient data collection for establishing practical identifiability via active learning*, Computational and Structural Biotechnology Journal 27, 4992–5006: [DOI](https://doi.org/10.1016/j.csbj.2025.10.058).
- Pearl (2009), *Causality: Models, Reasoning, and Inference*, 2nd edition.

## Author profile audit (2026-10-11)

The public GitHub profile and linked Medium account were located. The GitHub profile lists National Institute of Technology Agartala and describes ECE '26; those are treated as self-reported profile details, not as independently verified degree completion. The profile also self-reports a Samsung fellowship and ISRO hackathon award, but those credentials are not needed for this manuscript and are not asserted here. Searches did not resolve unambiguous Zenodo, OpenAIRE, or Google Scholar records for this author in this audit; no DOI, citation count, indexing status, or publication record is claimed. An ORCID link appears on the GitHub profile, but the public record could not be independently resolved in this pass, so the identifier is omitted from the citation metadata.

## Bibliography and citation integrity

Bibliographic details were checked against the public paper pages and DOI-linked records available during the 2026-10-11 search. The final author list for the active-learning paper was checked against PubMed and the journal-linked record. Any future submission should still re-verify every bibliography field and applicable AI-disclosure policy.
