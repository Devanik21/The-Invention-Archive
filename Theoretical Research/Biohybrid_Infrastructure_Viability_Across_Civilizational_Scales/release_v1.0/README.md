# Biohybrid Infrastructure Viability Across Civilizational Scales — release v1.0

**Preprint, not peer reviewed · 21 November 2026 · Devanik Debnath and Xylia**

Theoretical and computational. In silico and stylised only. No claim about human lifespan, clinical use, or any living organism.

This folder is the completed-paper release for the programme in the parent folder. It follows the archive's nine-core-file standard. See `MANIFEST.md` for exactly what is present in this push and what is held in the bundle.

## What is here

| File | Role |
|---|---|
| `verify_model.py` | Independently recomputes 38 analytic and small-scale claims (capture law, radiative and corrected shell temperatures, radiator area, Petela limit, light time, Murray exponent, lifespan exponent, reversible-work floors, biophotovoltaic gap, biocementation, circularity, gravity penalty, atlas tag counts). Writes `results/summary.json`. |
| `CITATION.cff` | Citation metadata. |
| `paper.bib` | BibTeX for the 22 references. |
| `.zenodo.json` | Deposit metadata, prepared but not deposited. No DOI is claimed. |
| `schema_article.jsonld` | schema.org description of the article. |
| `deploy.py` | Local release packager: checks core files, runs verification, writes a bundle zip. Does not push anywhere. |

The full research README (about 15,500 words, with the equations, the 110-concept atlas, the worked calculations and the errata) and the paper PDF are in the release bundle, not in this push.

## Read before citing

Section 24 of the full README lists errata. The most important is **E1**: Proposition 2 applies a planet-geometry factor (one quarter) to a shell that encloses the star. The scaling laws are unaffected. The temperature constants are not. The corrected values are 393.6 K or 331 K for a bare shell at 1 AU, not 278.3 K. A v1.1 correction is planned. `verify_model.py` checks both the printed and corrected forms.

## Reproduce

```bash
python3 verify_model.py         # 38 analytic checks, writes results/summary.json
python3 deploy.py               # core-file check, verification, local bundle zip
```

The Monte Carlo architecture study, the swarm study and the robust-viability kernel are reproduced by `analysis.py` (seed 20261121). That script is in the release bundle.

## Parent programme

`../README.md` describes the research question, scope and scientific boundaries of the programme. `../PROJECT_STRUCTURE.md` lists the programme tree.
