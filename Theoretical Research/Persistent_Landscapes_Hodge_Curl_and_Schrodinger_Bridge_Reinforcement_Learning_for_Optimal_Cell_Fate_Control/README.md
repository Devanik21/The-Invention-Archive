# Persistent Landscapes, Hodge Curl, and Schrödinger-Bridge Reinforcement Learning for Optimal Cell-Fate Control

**Subtitle:** Topological Obstructions to Cellular Reprogramming
**Author:** Devanik Debnath (Devanik21)
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India
**Version/date:** v2026.10.24.1 — 24 October 2026
**Status:** Theory-plus-simulation preprint. Synthetic dynamics only; no single-cell data were loaded, no wet-lab experiments were run, and no biological efficacy is claimed.

## Main conclusion

The barrier--persistence equality is valid under restrictive conditions: the scalar potential and the Freidlin--Wentzell gradient-diffusion assumptions must match, and the H₀ minimum--saddle persistence pairing must represent the transition of interest. It is not a universal rule for non-gradient dynamics or arbitrary learned landscapes.

An analytic counterexample on the circle S¹ (first Betti number 1) shows that a constant control can minimize quadratic effort: the tested 30%-duration pulse costs 3.333 times the constant control for the same endpoint. In the separate 400-rollout synthetic policy benchmark, evenly spaced pulses achieve 93/400 successes (23.25%), tabular RL 31/400 (7.75%), greedy pulses 11/400 (2.75%), and constant dosing 0/400. Therefore, **this run does not support RL superiority**, and it falsifies the unconditional claim that a topological cycle forces pulsing.

## Model and reproduction

The known quartic potential is

$$
V(x)=\frac{(x^2-1)^2}{4}.
$$

Under the matched gradient SDE convention, its analytic barrier is 0.25. The grid-estimated H₀ persistence is 0.249998, a synthetic unit check rather than empirical evidence. Noise tests show the pairing and calibration can deteriorate as noise increases.

Requirements and observed versions are recorded in [`results/environment.json`](results/environment.json). Execute from this directory:

```bash
python verify_model.py
```

The script rebuilds the synthetic result tables and figure files and runs 33 checks. Reported `PASS` means implemented synthetic checks executed; it does not mean a cell-fate mechanism has been validated. For PDF compilation use `latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex` with a TeX Live installation containing the listed packages.

## Figures

The 12 numbered figures are available as vector PDFs plus PNG previews in [`figures/`](figures/):

1. [Synthetic landscape](figures/fig01_landscape.pdf)
2. [Barrier and persistence check](figures/fig02_barrier_persistence.pdf)
3. [Persistence under noise](figures/fig03_persistence_robustness.pdf)
4. [Synthetic neural drift fit](figures/fig04_neural_drift_fit.pdf)
5. [Circle counterexample](figures/fig05_rotor_counterexample.pdf)
6. [Policy trajectory comparison](figures/fig06_policy_trajectories.pdf)
7. [RL training-return diagnostic](figures/fig07_rl_learning_curve.pdf)
8. [Toggle-switch network schematic](figures/fig08_toggle_network.pdf)
9. [Cycle network schematic](figures/fig09_cycle_network.pdf)
10. [Workflow diagram](figures/fig10_pipeline_flow.pdf)
11. [Falsification diagram](figures/fig11_falsification_flow.pdf)
12. [Schedule sensitivity](figures/fig12_schedule_sensitivity.pdf)

Graph schematics are conceptual only. All data-bearing figures are reproducibly generated from the released script and model-generated outputs.

## Evidence and limitations

- [Verification report](results/verification_report.json)
- [Verification console output](results/verify_model_stdout.txt)
- [Policy benchmark CSV](results/policy_benchmarks.csv)
- [Persistence-noise sweep](results/persistence_noise_sweep.csv)
- [Schedule sensitivity](results/schedule_sensitivity.csv)
- [Literature audit](results/literature_audit.md)
- [Claim/evidence matrix](results/claim_evidence_matrix.csv)
- [Author profile audit](results/author_profile_audit.md)
- [Publication design brief](results/publication_design_brief.md)
- [PDF visual review](results/pdf_visual_review.json)
- [Release check matrix](results/release_check_matrix.json)

GEO GSE242423/GSE242424 are identified as candidate future data sources; no matrix has been downloaded or analyzed in this release. The public accession record notes restrictions on raw data. Population snapshots do not by themselves identify unique cell lineages. Any eventual model fit must respect access conditions and be evaluated on held-out times/experiments.

## Files

- [`paper.pdf`](paper.pdf) — compiled full manuscript
- [`paper.tex`](paper.tex) — LaTeX source
- [`verify_model.py`](verify_model.py) — executable synthetic verification and figures
- [`paper.bib`](paper.bib) — bibliography
- [`CITATION.cff`](CITATION.cff) — citation metadata
- [`.zenodo.json`](.zenodo.json) — metadata payload only; no deposition or DOI claimed
- [`schema_article.jsonld`](schema_article.jsonld) — structured metadata
- [`deploy.py`](deploy.py) — safe validation and self-tests
- [`results/`](results/) — generated evidence, outputs, and QA records

A GitHub archive is not peer review, a Zenodo deposit, or proof of biological validity. The integrated novelty claim remains provisional and is limited to a falsifiable synthetic stress test; related work already covers landscape barriers, topology, stochastic bridges, and RL-based reprogramming. The 2026 GATTACA framework also combines graph neural networks and deep RL for Boolean biological-network control; RL control is not claimed as new (DOI: [10.1609/aaai.v40i2.37055](https://doi.org/10.1609/aaai.v40i2.37055)).
