# Speleothems as Noisy Channels

**Working title:** *Speleothems as Noisy Channels: Generative Inverse Decoding of Paleoclimate Under Microbial and Diagenetic Corruption*  
**Author:** Devanik Debnath  
**Release status:** Research package locally verified; GitHub source archive incomplete in this commit.

## Research question

Can speleothem proxy records be treated as observations transmitted through a potentially altered geochemical channel, with fabric information acting as imperfect side information? The proposed framework separates reconstruction error from recoverability and uncertainty calibration.

## Scope and evidence disclosure

The benchmark in this release is **synthetic**. No SISAL rows, cave-monitoring records, laboratory measurements, or field observations were analyzed. The results are not empirical paleoclimate estimates. Literature coverage is bounded and does not establish absolute priority or prove novelty.

The current synthetic benchmark compares a single-proxy baseline, a multi-proxy model, and a multi-proxy model with noisy fabric-class side information. Under shifted alteration, the channel-aware model modestly improves synthetic RMSE, but its nominal 90% interval coverage is only about 84.7%; uncertainty calibration remains inadequate.

## Current local validation

The locally built package was checked for reproducibility, PDF rendering, bibliography consistency, metadata, and fail-closed deployment behavior. The 16-page manuscript contains 8 tables and 8 figures. The full ZIP package is available in the associated ChatGPT conversation, but the binary PDF/package has **not** been uploaded to this repository by this commit.

## Intended next steps

1. Upload the complete source, figure assets, results, and compiled PDF to this directory.
2. Re-run reproducibility and PDF validation from a clean checkout.
3. Verify the exact GitHub Actions run and commit SHA.
4. Only then create a version tag and consider a Zenodo deposit. No DOI is claimed.

## Literature anchors

- Kaushal et al. (2024), SISALv3, *Earth System Science Data*, DOI: [10.5194/essd-16-1933-2024](https://doi.org/10.5194/essd-16-1933-2024).
- Frisia (2015), calcite fabric logging, DOI: [10.5038/1827-806X.44.1.1](https://doi.org/10.5038/1827-806X.44.1.1).
- Scholz et al. (2014), diagenesis and U-series age accuracy, DOI: [10.1016/j.chemgeo.2014.08.005](https://doi.org/10.1016/j.chemgeo.2014.08.005).
- Owen, Day & Henderson (2018), CaveCalc, DOI: [10.1016/j.cageo.2018.06.011](https://doi.org/10.1016/j.cageo.2018.06.011).
- Hu et al. (2021), cave dripwater isotope proxy-system models, DOI: [10.1016/j.quascirev.2021.106799](https://doi.org/10.1016/j.quascirev.2021.106799).
- Cahill et al. (2023), Bayesian multi-proxy hydroclimate time series, DOI: [10.1002/env.2786](https://doi.org/10.1002/env.2786).
- Kalpana et al. (2021), fatty acids in Krem Mawmluh speleothems, DOI: [10.3389/feart.2021.687376](https://doi.org/10.3389/feart.2021.687376).
- McDermott (2004), speleothem isotope review, DOI: [10.1016/j.quascirev.2003.06.021](https://doi.org/10.1016/j.quascirev.2003.06.021).
- Treble et al. (2026), SISAL_monv1 monitoring preprint, DOI: [10.5194/essd-2026-31](https://doi.org/10.5194/essd-2026-31). Underlying aligned records and licensing were not verified.

## Reproducibility status

The complete local package passed Python compilation, 10 model assertions, 3/3 fail-closed fixtures, structural/metadata/PDF preflight checks, and a visual review of all 16 PDF pages. These are local checks; remote GitHub Actions and deployment are not implied by them.
