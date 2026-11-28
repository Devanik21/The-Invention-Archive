# Database-Free Agnostic Biosignature Inference via Neuro-Symbolic Assembly Theory

**Research preprint | 28 November 2026 | Proposal + synthetic software verification**

This release proposes a posterior molecular-graph inference framework for tandem mass spectra. It combines a transformer-style spectrum encoder, formula/chemistry-conditioned graph generation, symbolic feasibility constraints, fragmentation-DAG inference, graph-level molecular assembly, posterior uncertainty, and resource-aware active acquisition.

> **Epistemic status.** This is a research architecture and a deterministic synthetic inference abstraction. It is not a trained graph transformer, not a real-spectrum benchmark, not a calculation of molecular assembly on observed molecules, and not a validated biosignature or flight-ready instrument.

## Why the proposal was revised

A relevant PNAS paper published in August 2026 already predicts molecular assembly from standardized MS1 spectra with machine learning. Earlier work used MS/MS to estimate molecular assembly in mixtures. DiffMS already performs formula-restricted graph diffusion conditioned on spectra, while MSNovelist performs de novo candidate generation without structure-library retrieval at inference. The contribution here is therefore not “the first database-free MS/MS predictor”. It is the proposed integration of a **posterior over molecular graphs, fragmentation explanations and chemistry hypotheses**, with assembly uncertainty, an environment-conditioned abiotic null, and active measurement selection.

The often-quoted MA≈15 boundary is not an impossibility theorem. The original study reports a sample-set separation and a model-dependent random-assembly argument; it also records biological false negatives. The release therefore does not classify MA>15 as proof of life.

## Model in one view

For spectrum episode D, candidate molecular graph G, fragmentation DAG T, chemistry grammar C, and instrument nuisance parameters θ:

`p(G,T,C,θ|D) ∝ p(D|G,T,C,θ,e) p(T|G,C,θ) p(G|C) p(C|e) p(θ|c,u).`

Assembly evidence is a posterior quantity, not a single hard prediction:

`qτ(D)=Pr(MA(G;B_C)≥τ|D).`

A life-relevant evidence ratio compares a specified directed/templated selection family with a specified environment-dependent abiotic family. It cannot rule out every possible abiotic mechanism.

## Synthetic verification snapshot

With seed `20261128` and nominal synthetic feature-noise SD `0.120`, the abstract five-class posterior test returned balanced accuracy `0.9767`, AUC `0.9989`, Brier score `0.0134`, and ten-bin ECE `0.0143` across 1,200 generated examples. These are software-smoke-test outputs from hand-specified four-dimensional feature prototypes; the score labels `[9, 12, 16, 19, 23]` are not molecular assembly indices. They must not be interpreted as MS/MS, molecular-graph, biological, or mission performance.

## Files

The full local release bundle contains the compiled manuscript, LaTeX source, bibliography, synthetic verification code, deployment preflight, figures, validation logs, literature audit, and metadata. The manuscript and supporting files are being staged for repository publication; this README alone does not constitute a complete release.

## Critical limitations

1. “Database-free” means no named-compound or reference-spectrum retrieval at inference. It does not mean data-free or independent of chemical priors.
2. Chemistry-agnostic transfer requires broadening the elemental/valence grammar. A fixed CHNOPS-only model cannot substantiate universal chemistry claims.
3. Distinct isomers or chemistry hypotheses can be observationally equivalent under a given set of scans. The proper output may be an abstention.
4. Molecular assembly requires a declared graph representation and building-block convention. Exact computation may exceed a payload budget; only verified bounds should then be reported.
5. An MA score alone is not a proof of biogenesis. The abiotic model, abundance, environmental context, contamination and instrument calibration matter.

## Selected sources

- Marshall et al. (2021), *Nature Communications*, DOI: 10.1038/s41467-021-23258-x.
- Rutter et al. (2026), *PNAS*, DOI: 10.1073/pnas.2619198123.
- Bohde et al. (2025), DiffMS, PMLR 267, https://proceedings.mlr.press/v267/bohde25a.html.
- Stravs et al. (2022), MSNovelist, DOI: 10.1038/s41592-022-01486-3.
- Seet et al. (2025), graph assembly algorithm, DOI: 10.1021/acs.jcim.5c01964.
- Schneider et al. (2026), de novo MS/MS review, DOI: 10.3390/molecules31050769.
- NASA/JPL, Ocean Worlds Life Surveyor: https://www.jpl.nasa.gov/go/owls/.

## Research-process credit and responsibility

Co-equal research-process credit is given to Devanik Debnath and Xylia, the autonomous multi-model research engine coordinating AI-assisted discovery, synthesis, and verification. Devanik Debnath retains responsibility for the final manuscript, scientific claims, citations, and released artifacts. Xylia is an AI research-orchestration system, not a human author.
