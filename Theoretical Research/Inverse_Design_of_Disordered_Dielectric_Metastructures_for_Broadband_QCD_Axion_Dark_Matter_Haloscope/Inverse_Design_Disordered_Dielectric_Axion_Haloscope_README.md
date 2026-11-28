# Inverse Design of Disordered Dielectric Metastructures for Broadband QCD Axion Dark Matter Haloscopes

**Release status:** research package prepared for manual archival; local reduced-order checks passed; full-wave and experimental validation not performed.  
**Author:** Devanik Debnath (Devanik21), Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India.  
**Date:** 2026-11-28  
**Manuscript:** [`paper.pdf`](paper.pdf) · [`paper.tex`](paper.tex) · [`paper.bib`](paper.bib)

## Research contribution

This release does **not** claim discovery of disordered dielectric interface conversion. Koppell et al., *Phys. Rev. D* **113**, 083045 (2026), DOI [10.1103/k8l8-sqg2](https://doi.org/10.1103/k8l8-sqg2), already study disordered dielectric absorbers, 2D/3D targets and a proposed DPHaSE architecture. The present research contribution is a proposed inverse-design formulation that treats the axion-induced source, broadband conversion, transport, absorption, collection, detector response, disorder risk and fabrication constraints as a coupled problem.

The manuscript derives a source-aware adjoint identity and proposes a CVaR-based robust objective. It also reports a **reduced-order dimensionless screening model**, not a full-wave Maxwell result. In the current seeded surrogate run, the optimizer selected median radius 5237.3 nm, lognormal width 0.3608, and dielectric fraction 0.3119. The normalized band-response proxy was 0.0090654 for the reference design and 0.019154 for the screened design (ratio 2.113). This ratio is conditional on explicit illustrative optical constants, transport proxies, and a heuristic void-accessibility penalty. It is **not** an improvement in axion coupling reach, predicted signal power, or an experimental result.

## Contents

The release follows the current publisher architecture: exactly nine root-level core files, plus `figures/` and `results/`. The metadata uses the parent repository’s MIT license.

| File | Purpose |
|---|---|
| `paper.tex` | Complete LaTeX manuscript. |
| `paper.pdf` | Compiled research paper. |
| `verify_model.py` | Executable reduced-order numerical model, checks, and figure regeneration. |
| `README.md` | This research and reproducibility guide. |
| `CITATION.cff` | GitHub citation metadata. |
| `paper.bib` | Bibliography. |
| `.zenodo.json` | Draft archival metadata only; no deposition or DOI claimed. |
| `schema_article.jsonld` | Schema.org ScholarlyArticle metadata. |
| `deploy.py` | Local validation/build/self-test helper; does not push to GitHub. |

`figures/` contains eight reproducible PDF vector figures and PNG previews: four plots, two structural/transport graphs, and two flow diagrams. `results/` contains numeric outputs, figure source data, the environment record, literature and evidence audits, candidate screening, design brief, local test outputs, and release review records.

## Reproduce the reduced-order run

From this directory:

```bash
python verify_model.py
```

Dependencies: NumPy, SciPy and Matplotlib. Tested package versions are recorded in [`results/environment.txt`](results/environment.txt). The script uses a fixed random seed (`20261128`) and regenerates all eight figures and the primary output files.

The tests check the exact monodisperse sphere area identity, the Rayleigh $x^4$ leading term of the interpolating kernel, positivity and bounds of the implemented transport quantities, optimizer feasibility, and generation of all figure files. A passing run validates these code paths only.

## Important limitations

- No Maxwell/FDTD/FEM solver was run. The forward Maxwell/adjoint method is derived and proposed, not implemented in this release.
- The conversion crossover is a smooth interpolation of published asymptotic limits, not an exact Mie calculation.
- The anisotropy, effective diffusion, path attenuation, sensor efficiency, and porosity-accessibility terms are screening assumptions.
- The nominal refractive index and extinction coefficient are illustrative; they are not asserted to represent a named material at cryogenic temperature.
- No absolute conversion power, photon count rate, QCD axion coupling reach, sensor dark-count model, or experimental sensitivity is reported.
- The DPHaSE concept is cited as a published proposal/projection, not as a demonstrated experiment.

The manuscript sets out the full-wave validation ladder required before a device-level claim: isolated-sphere/Mie benchmarks, multi-particle solver cross-checks, source-aware gradient finite-difference tests, collection-flux and energy-balance checks, measured cryogenic dielectric constants, and a background-characterized integrated prototype.

## Repository destination (manual upload requested)

Place this entire release directory under:

`Theoretical Research/Inverse_Design_of_Disordered_Dielectric_Metastructures_for_Broadband_QCD_Axion_Dark_Matter_Haloscope/`

Repository: <https://github.com/Devanik21/The-Invention-Archive>  
Parent index: `Theoretical Research/Readme.md` (append the supplied index line in `results/parent_index_entry.md`).

No GitHub files were changed by this release-preparation run. Remote CI, remote file hashes, branch/tag state, and GitHub rendering are therefore not verified. `deploy.py` is local-only and deliberately does not push.
