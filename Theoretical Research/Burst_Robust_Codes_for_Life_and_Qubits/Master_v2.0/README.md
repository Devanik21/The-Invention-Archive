# Burst-Robust Codes for Life and Qubits

**Correlated-Event Reliability, Holographic Erasure Recovery, and Finite Repair Capacity**  
**Author:** Devanik Debnath · National Institute of Technology Agartala, India  
**Version:** 2.0 · **Date:** 6 November 2026  
**Status:** Theoretical and synthetic-computational preprint. Not peer reviewed; no DOI or external repository deposit is claimed.

## Master manuscript

- **[Read the 31-page master paper](paper.pdf)**
- [LaTeX source](paper.tex) · [Bibliography](paper.bib) · [Reproducible model and figure generator](verify_model.py)
- [Verification report](results/verification_report.json) · [Complete numerical sweep](results/spanning_sweep.csv)
- [Figure inventory](results/figure_inventory.csv) · [Visual review record](results/pdf_visual_review.json)
- [Publication-quality report](results/publication_quality_report.md) · [SHA-256 manifest](results/sha256_manifest.json)

The release contains a full mathematical treatment of marked space-time disturbance processes, substrate-specific transformations, code/decoder recovery, finite biological repair workload, queue stability, Poisson thinning, and model-conditional risk envelopes. It also includes a reproducible synthetic benchmark, sensitivity analyses, uncertainty intervals, experimental protocols, falsification criteria, evidence tables, and technical appendices.

## Central scientific distinction

The paper proposes a shared *statistical vocabulary*, not a shared microscopic mechanism. A superconducting-qubit device and a living cell have different state spaces, physical dynamics, measurement channels, failure definitions, and recovery mechanisms. Holographic quantum error correction is not a model of DNA repair. The neutrino fog is discussed only as an inference analogy from direct dark-matter detection; it is not a biological damage mechanism or a repair floor.

## What the synthetic benchmark does

The main experiment compares independent erasures against a Poisson superposition of finite five-site line tracks on a 24 × 24 square lattice. It counts 4-neighbour connected erased components that span either pair of opposite boundaries. Track intensity is solved by bisection so the **expected lattice-wide mean site occupancy** matches the IID target probability. Because tracks are clipped at lattice boundaries, site-specific marginals remain heterogeneous; this is not a sitewise-marginal-matched experiment.

The primary grid contains ten probabilities from 0.20 to 0.65, 4,000 trials per channel and point, and fixed seed `20261106`. At target occupancy `p = 0.40`, the benchmark observed 0/4,000 IID spanning events and 705/4,000 track-channel events (17.625%). At `p = 0.50`, it observed 219/4,000 IID events (5.475%) and 2,737/4,000 track events (68.425%). The zero IID count at `p = 0.40` is not a zero-risk claim; the rule-of-three gives an approximate 95% upper bound of 3/4,000 = 0.075%.

**Interpretation boundary:** these are finite-size results for this selected lattice, graph, and track law. They are not physical surface-code or holographic-code thresholds, device measurements, Geant4/PARTRAC transport, DNA lesion predictions, or biological repair probabilities. The ranking is not asserted as a universal law.

## Mathematical framework

A marked event is represented as `Z_k = (T_k, X_k, Θ_k, L_k, W_k, M_k)`: arrival time, location, orientation, extent, temporal response, and severity/workload marks. Event geometry induces a footprint; a substrate-specific map translates that footprint into a quantum error field or biological lesion field; recovery is then described by a code/decoder or by repair dynamics. The queue condition `ρ = λ E[S] / μ < 1` is a stability condition only for the stated stationary single-server workload abstraction. The Poisson-thinning horizon risk `1 − exp(−λ q_F T)` assumes the stated independent marked-arrival construction. Neither is a universal physical threshold.

## Evidence and domain limits

The manuscript distinguishes published observations, mathematical consequences of model assumptions, synthetic outputs, and proposed experiments. In particular:

- Radiation-associated correlated qubit disturbances are motivated by detector-synchronized and correlated-error studies; no universal “one particle destroys a computer” claim is made.
- High-LET ion tracks motivate clustered DNA-damage models, but not every track overwhelms repair.
- dCas9–KillerRed is treated as a controllable localized oxidative-damage perturbation/calibration platform, not as a heavy-ion equivalent.
- AlphaQubit 2 is discussed as a candidate decoder technology; it is not claimed to have been validated on this paper’s burst law.
- Engineered protein spin resonance is quantum biosensing, not a fault-tolerant quantum computer inside a cell.

## Reproduce

Python 3.12 is used in the release workflow. Install NumPy, Matplotlib, NetworkX, PyMuPDF, and Pillow; then run:

```bash
python verify_model.py
python deploy.py --self-test
python deploy.py --build
python deploy.py --validate-only
```

The main benchmark uses 4,000 trials per channel per point; additional finite-size and track-length sensitivity analyses are smaller and are identified as synthetic. Exact library versions and operating environment are written to `results/environment_report.json`. `deploy.py` enforces the 28–40-page window, at least 18 substantive main-text pages, the table/figure floors, and package structure.

## Visual and reproducibility evidence

The paper includes 12 numbered scientific figures: 8 plots, 2 graphs, and 2 flow diagrams. Seven substantive tables appear in the manuscript. Every PDF page is rendered at 160 dpi and retained under `results/pdf_pages/`; four contact sheets are retained under `results/`. The visual-review record identifies the exact PDF SHA-256 and the inspected page range. Contact-sheet inspection is not a substitute for independent scholarly proofreading or peer review.

## Citation

Debnath, Devanik. *Burst-Robust Codes for Life and Qubits: Correlated-Event Reliability, Holographic Erasure Recovery, and Finite Repair Capacity*. Theoretical and synthetic-computational preprint, version 2.0 (6 November 2026). No DOI is claimed.
