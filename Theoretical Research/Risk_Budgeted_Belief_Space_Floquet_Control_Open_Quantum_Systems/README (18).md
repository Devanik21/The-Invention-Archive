# Risk-Budgeted Belief-Space Floquet Control for Open Quantum Systems

**Version:** 1.0.0  
**Release date:** 28 November 2026  
**Status:** Research preprint; not peer reviewed; no DOI assigned  
**Manuscript:** `paper.pdf` (15 pages, A4)  
**Source:** `paper.tex`; bibliography metadata: `paper.bib`

## Summary

This preprint develops a proposed control architecture for periodically driven open quantum systems under uncertain detuning and dissipation, finite actuator bandwidth, hardware constraints, and an explicit heating/error budget. Its distinguishing formulation is **risk-budgeted belief-space Floquet control (RBB-FC)**, which connects four elements:

1. a Lindblad model for the quantum plant;
2. a Neural-SDE belief model for latent classical drift and hardware uncertainty;
3. a physics-constrained policy acting in a finite Fourier waveform family, including an actuator transfer model and amplitude/slew limits; and
4. a tail-risk-sensitive objective with a hard safety interlock.

The manuscript proves a finite-dimensional, time-dependent Lindblad propagator perturbation bound using a Duhamel argument. Under its stated assumptions, the diamond-norm deviation between two CPTP propagators is bounded by the time integral of the diamond-norm generator mismatch. It then distinguishes the objectives for bosonic-code preparation, prethermal heating suppression, and dissipative time-crystalline order.

## What was actually tested

The released numerical study is a deterministic, two-level open-qubit synthetic benchmark. It tests only a constrained two-harmonic Fourier pulse and a first-order low-pass actuator model with quasi-static detuning, relaxation, and pure dephasing. It does **not** implement the proposed Neural SDE or a deep reinforcement-learning agent.

| Metric | Constant resonant pulse | Risk-weighted two-harmonic candidate |
|---|---:|---:|
| Mean final target-state population over 29 held-out detunings | 0.8900145 | 0.8910273 |
| Worst value on the same finite detuning grid | 0.7340518 | 0.7361297 |

The absolute improvements are small (about 0.10 and 0.21 percentage points, respectively). They are not evidence of statistical significance or a general quantum-control advantage. The benchmark passes **9/9 implemented numerical checks**, including generator trace preservation, Hermiticity preservation, state positivity to numerical tolerance, command amplitude/slew envelopes, and time-step convergence (absolute fidelity difference approximately \(1.11\times10^{-12}\)).

## Scientific scope and limitations

- The proposed Neural-SDE/RL controller is an architecture and evaluation programme, not a trained or experimentally validated system in this release.
- The numerical test is a two-level Lindblad model, not a many-body calculation, bosonic code, discrete time crystal, or superconducting-cavity experiment.
- The reported synthetic benchmark does not demonstrate room-temperature quantum sensing, removal of cryogenic requirements, or fault-tolerant computation.
- The propagator certificate assumes bounded time-dependent Lindblad generators on a common finite-dimensional Markovian model. It does not cover omitted leakage levels, non-Markovian baths, or model classes outside those assumptions.
- The finite-grid worst-case metric is not a continuum worst-case guarantee.

## Reproduction

Requirements: Python 3.10 or later, NumPy, SciPy, and Matplotlib.

```bash
python -m py_compile verify_model.py
python verify_model.py
```

The reported environment was Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8. Running the script regenerates the figures and the CSV/JSON numerical records in `figures/` and `results/`.

The main evidence files are:

- `results/verification_metrics.json` — parameters, optimizer outcome, numerical metrics, and all nine checks;
- `results/verification_stdout.txt` — captured verification output;
- `results/detuning_sweep.csv`, `results/dephasing_sweep.csv`, and `results/waveform.csv` — numerical traces used in the plots.

## Structure

```text
Floquet_Belief_Space_Release/
├── paper.pdf
├── paper.tex
├── paper.bib
├── verify_model.py
├── README.md
├── CITATION.cff
├── .zenodo.json
├── schema_article.jsonld
├── deploy.py
├── release_manifest.txt
├── figures/
│   ├── fig01_architecture.{png,pdf}
│   ├── fig02_filtered_waveform.{png,pdf}
│   ├── fig03_robustness.{png,pdf}
│   └── fig04_dephasing_sweep.{png,pdf}
└── results/
    ├── verification_metrics.json
    ├── verification_stdout.txt
    ├── detuning_sweep.csv
    ├── dephasing_sweep.csv
    ├── waveform.csv
    ├── release_check_matrix.json
    └── pdf_visual_review.json
```

`deploy.py` validates the local release and creates `Floquet_Belief_Space_Release.zip`. It deliberately performs **no network operations and no GitHub upload**; the repository owner can upload the resulting ZIP when ready.

## Prior work and references

The manuscript discusses, among other sources, the 2026 published study on reinforcement-learning-assisted Floquet preparation of bosonic codes (DOI: [10.1038/s42005-026-02875-2](https://doi.org/10.1038/s42005-026-02875-2)); a separate 2025 study of reinforcement learning under physical constraints; Floquet prethermalization and heating; dissipative time crystals; Neural SDEs; and bosonic quantum error correction. Full bibliographic entries are in `paper.bib` and in the reference section of `paper.pdf`.

## Citation

See `CITATION.cff`. No DOI or peer-reviewed publication record is claimed for this preprint.
