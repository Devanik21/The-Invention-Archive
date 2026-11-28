# Noncommutative Lie-Bracket Control of Stochastic Multi-Organ Aging Manifolds

Preprint v1.0 (not peer reviewed). Authors: Devanik Debnath; Xylia.

## What this is
A stylised, dimensionless, in-silico model of two intervention fields (senolysis A and
partial reprogramming B) acting on four coupled aging variables (epigenetic excess E,
senescent burden S, fibrosis F, oxidative stress O). It tests whether order matters,
decomposes the non-commutativity into control-level and drift-level brackets, and
compares eight schedules at equal per-control exposure.

## Scope
- All numbers come from analysis.py with seed 20261128. No parameter is fitted to data.
- No dose, schedule, or sequence is recommended for any person. No lifespan prediction for humans.
- The premise that a 2026 Aging report demonstrated non-commuting interventions was not
  located in the search. Non-commutativity in biology is treated as a hypothesis.

## Contents
- Noncommutative_Lie_Bracket_Aging_Control_v1.0.pdf   paper (16 pages)
- analysis.py          model, brackets, BCH scaling, Chow rank, stochastic schedules, figures
- build_pdf.py         typesets the paper from results/results.json
- figs/                eight figures (PNG)
- results/results.json full numerical output
- results/summary.json condensed output
- requirements.txt, CITATION.cff, .zenodo.json

## Reproduce
    pip install -r requirements.txt
    python3 analysis.py      # writes results/ and figs/, a few minutes on one core
    python3 build_pdf.py     # writes the PDF

## Status
Proved: zero control bracket at zero coupling (Proposition 1). Numerical: BCH scaling
(slope 1.96), drift-level order dependence at zero coupling, schedule comparison.
Model-dependent: ordering results depend on the declared couplings and identity floor.
Untested: any order effect in a living system (see Section 12 of the paper).
