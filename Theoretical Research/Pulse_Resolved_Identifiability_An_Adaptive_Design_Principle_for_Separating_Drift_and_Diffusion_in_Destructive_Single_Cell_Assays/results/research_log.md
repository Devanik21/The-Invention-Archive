# Research-cycle log — 2026-10-24

## Run identity
- Date: 2026-10-24.
- Engine: Xylia, AI-assisted research orchestration.
- Selected title: Pulse-Resolved Identifiability: An Adaptive Design Principle for Separating Drift and Diffusion in Destructive Single-Cell Assays.
- Scope: analytical identifiability result plus a synthetic controlled-OU benchmark. No empirical dataset was downloaded or analyzed.

## Discovery and screening
Ten candidates were scored on problem importance, evidence of a gap, conceptual depth, mechanistic plausibility, mathematical core, substantive AI role, plausible human benefit, first-validation feasibility, falsifiability, and safety. Scores and rejection reasons are in topic_candidates.csv. The selected direction scored highest because its core claim is exact under explicit assumptions, its counterexamples are executable, and it can be tested without inventing biological evidence.

## Search coverage and queries
Searches were run on 2026-10-24 for:
- Devanik Debnath / Devanik21 across GitHub, ORCID, Zenodo, OpenAIRE, Google Scholar, LinkedIn, and public author records.
- Single-cell perturbation response modelling and neural optimal transport.
- scPerturb dataset harmonization and benchmarking.
- Single-cell experiment design under temporally fluctuating environments.
- Fisher information, stochastic chemical kinetics, and identifiability.
- Controlled Ornstein--Uhlenbeck experimental design.
- Recent 2026 snapshot-based stochastic dynamics and lineage-aware transport preprints.

The literature audit records the verified sources and their relevance. The search was broad but not exhaustive; absence of an exact title match is not treated as proof of novelty.

## Scientific posture
The key identity is derived directly from the post-pulse solution of a scalar linear stochastic differential equation. The numerical script checks the identity, stationary variance confounding, Fisher-information rank, positivity, amplitude bounds, and output generation. Synthetic model results are not empirical validation.

## Novelty decision
The literature already contains information-based single-cell sample allocation, stochastic Fisher-information analysis, neural optimal transport, and controlled diffusion-process design. The claim is therefore limited to an explicit, executable identifiability certificate for the stated destructive-snapshot moment model, paired with a transparent local greedy schedule and adversarial confounding tests. No priority claim is made for optimal design or pulse-based system identification generally.

## Remaining uncertainty
The model omits donor effects, cell-state mixtures, cell proliferation/death, nonlinear dynamics, batch drift, and pulse-dependent noise. A real-data benchmark is the next validation step. The paper should be narrowed or rejected if the exponential post-pulse relation fails on held-out time points or if estimated diffusion tracks plausible measurement-noise changes.
