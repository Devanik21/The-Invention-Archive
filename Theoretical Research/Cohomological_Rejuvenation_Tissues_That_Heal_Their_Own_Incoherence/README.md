# Cohomological Rejuvenation: Tissues That Heal Their Own Incoherence

**A falsifiable framework for separating local tissue mismatch from global compatibility obstruction**

**Author:** Devanik Debnath (Devanik21)  
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India  
**ORCID:** https://orcid.org/0009-0001-8310-6921  
**Release date:** 2026-10-11  
**Release:** v2026.10.11.1

## Status and contribution

This is a theoretical and computational research programme, not an empirical aging result. No spatial atlas was processed and no biomarker performance is reported. The literature search found a directly relevant 2026 sheaf-theoretic model of aging and repair failure (Segura, *BioSystems*, DOI: [10.1016/j.biosystems.2026.105924](https://doi.org/10.1016/j.biosystems.2026.105924)) and a 2026 preprint applying hierarchical sheaf spectral features to single-cell RNA-seq ([arXiv:2603.26858](https://arxiv.org/abs/2603.26858)). This release does **not** claim that the general combination of sheaves and aging is novel.

The narrower contribution is to formally distinguish: (1) local mismatch measured by a sheaf Dirichlet energy; (2) a reducible least-squares component; and (3) an irreducible, cycle-level compatibility obstruction represented by an appropriate cohomology class. An observed non-zero sheaf energy is not itself first cohomology.

## Mathematical core

For cell states `x` on a spatial graph and learned incidence-compatibility matrix `B`, with nonnegative edge weights `W`, define the degree-zero sheaf Laplacian:

$$
L_0 = B^\top W B \succeq 0.
$$

A normalized local-mismatch statistic is

$$
D_{\mathrm{loc}} = \frac{\|W^{1/2}Bx\|_2^2}{\sum_e w_e + \epsilon}.
$$

For an observed edge cochain `y` on a graph-only complex, define `A = W^{1/2}B`, `P_A = A A^\dagger`, and the cycle-level residual

$$
D_{\mathrm{cyc}} = \|(I-P_A)W^{1/2}y\|_2^2.
$$

`D_loc` and `D_cyc` are different statistics. `D_loc` measures local compatibility relative to learned maps; `D_cyc` measures the part of observed edge data that cannot be represented as a coboundary under those maps. The latter has a cohomological interpretation only when the cochain complex and closure conditions support it. Neither metric is intrinsically a measure of biological age.

For a cochain complex, the first cohomology is the quotient of closed edge cochains by exact coboundaries:

$$
H^1(K;\mathcal{F}) = \ker(\delta^1) / \mathrm{im}(\delta^0).
$$

This quotient is not the same object as the local Dirichlet energy.

## Research hypothesis

Does either statistic add predictive information about age beyond cell type, region, sex, batch, spatial density, and cell-intrinsic measurements when evaluated on held-out animals/donors? A candidate first dataset is the mouse-brain MERFISH atlas behind Sun et al., *Nature* (2025), DOI [10.1038/s41586-024-08334-8](https://doi.org/10.1038/s41586-024-08334-8); dataset deposit: [Zenodo 10.5281/zenodo.13883176](https://doi.org/10.5281/zenodo.13883176); an independent spatial brain aging atlas is described by Wu et al., *Aging Cell* (2024), DOI [10.1111/acel.14109](https://doi.org/10.1111/acel.14109). This release has not downloaded or analysed either dataset.

## Verification

Requirements: Python 3.10+ and NumPy 1.24 or later (tested with NumPy 2.x).

```bash
python verify_model.py
```

The script checks positive semidefiniteness, the Dirichlet identity, exact/cycle projection, orthogonality, graph cycle rank, and a seeded toy contrast where independent node noise is deliberately increased. The synthetic contrast verifies only that the statistic responds to a pattern inserted by the generator; it is **not** evidence that aging increases tissue incoherence.

## Falsification criteria

Reject or narrow the biomarker hypothesis if the age association disappears after adjustment; if random or shuffled maps perform similarly; if the signal is unstable to graph and normalization choices; if it fails donor-held-out evaluation; or if it simply recapitulates cell composition, graph density, technical batch, or cycle count. An association with chronological age does not establish prediction of frailty, mortality, rejuvenation, or function.

## Translational boundary

Schrödinger bridges minimize path-space relative entropy to a specified reference measure; they are not automatically minimum-entropy-production paths. Partial epigenetic reprogramming is an existing research area but carries cell-identity and safety challenges. SynNotch systems sense specific contact-dependent ligands; they do not directly read a high-dimensional sheaf residual. A molecular proxy, its calibration, delivery, cell identity preservation, and safety would all require independent evidence. No therapy or laboratory protocol is proposed.

## Artifact inventory

- `paper.tex`, `paper.pdf`: manuscript and compiled preprint.
- `verify_model.py`: executable mathematical checks and synthetic demonstration.
- `paper.bib`: machine-readable references.
- `medium_story.md`: technical essay draft (not published).
- `x_thread.md`: 12-post X draft (not published).
- `CITATION.cff`, `.zenodo.json`, `schema_article.jsonld`: archival metadata drafts; no DOI has been minted.
- `deploy.py`: local validation and deployment helper.

## References

- Segura, J. J. (2026). [Aging as cross-hallmark obstruction amplification](https://doi.org/10.1016/j.biosystems.2026.105924). *BioSystems*.
- Wang, X. X. & Wei, G.-W. (2026). [A hierarchical sheaf spectral embedding framework for single-cell RNA-seq analysis](https://arxiv.org/abs/2603.26858). arXiv.
- Hansen, J. & Ghrist, R. (2019). [Toward a spectral theory of cellular sheaves](https://doi.org/10.1007/s41468-019-00038-7). *Journal of Applied and Computational Topology*.
- Sun, E. D. et al. (2025). [Spatial transcriptomic clocks reveal cell proximity effects in brain ageing](https://doi.org/10.1038/s41586-024-08334-8). *Nature*.
- Yücel, A. D. & Gladyshev, V. N. (2024). [The long and winding road of reprogramming-induced rejuvenation](https://doi.org/10.1038/s41467-024-46020-5). *Nature Communications*.