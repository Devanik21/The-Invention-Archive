# The Recoverability Frontier of Damaged Information-Bearing Systems

**Release 2026.11.08 · Preprint v1.0 · Devanik Debnath and Xylia**

Not peer reviewed. Numerical checks are not biological validation. No DOI or Zenodo deposit is claimed.

## Summary

A stylised framework that separates five questions about a damaged information-bearing system:

1. **Coding:** does the information survive, given a redundant erasure code?
2. **Topology:** does the organisation survive, measured by persistent clusters of a point cloud?
3. **Control:** can a controller reach a functional state, and at what minimum energy?
4. **Estimation:** how much information do partial observations carry?
5. **Thermodynamics:** what is the physical floor on restoring that information?

Six propositions are proved (Section 4 and Appendix A). Eight hypotheses are tracked with their status, including two that failed.

## Scope

All systems are abstract and dimensionless. This release does not study the reversal of death, the revival of any organism or person, personal identity, or any clinical procedure. It makes no biological claim. The word "recoverability" refers only to the recovery of an abstract information state in a simulated system.

## Contents

| Path | Description |
|---|---|
| `paper/Recoverability_Frontier_of_Damaged_Information_Systems.pdf` | 12-page preprint |
| `code/recoverability_experiments.py` | Experiments E1 to E6, fixed seed 20261108 |
| `code/build_paper.py` | Builds the PDF from the results |
| `results/*.json` | Raw outputs per experiment, plus `summary.json` |
| `figures/fig01` to `fig08` (`.png` and `.svg`) | Figures in the paper |
| `requirements.txt` | Python dependencies |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata |

## Reproduce

```bash
pip install -r requirements.txt
python code/recoverability_experiments.py
python code/build_paper.py
```

Runtime is a few minutes on one CPU core. The same seed gives identical output.

## Headline numerical results

- **Coding.** The 50%-recovery erasure fractions are 0.730, 0.476 and 0.230 for code rates 0.25, 0.5 and 0.75, against the Shannon limits 1 − R = 0.75, 0.50 and 0.25. Monte Carlo agrees with the exact closed form (Proposition 2) with max deviation 0.049.
- **Topology.** The three-cluster signature survives deletion of up to 87% of points. Adding jitter of 0.3 breaks it even at zero deletion (preservation 0.513). Fixed-scale robustness to noise was falsified.
- **Control.** Reachability has a cliff at m = ⌈n/T⌉ = 3 active actuators. The median minimum energy at m = 3 is about 1318 times the full-actuation reference.
- **Estimation.** The observed fraction needed for normalised error D ≤ 0.2 falls from 0.90 at correlation ρ = 0.2, to 0.40 at ρ = 0.9, to 0.10 at ρ = 0.98.
- **Thermodynamic floor.** For D = 0.2 the requirement is 1.161 bits per coordinate. At 310 K, k_B T ln 2 = 2.97 × 10⁻²¹ J. This floor does not constrain restoration at these scales.
- **Composite frontier.** Under an independence assumption, the control layer acts as a cliff. No damage level is safe once actuator loss reaches 0.3. The composite is not tested end to end.

## Hypothesis status

| # | Hypothesis | Status |
|---|---|---|
| H1 | Random binary code recovery matches the closed form | Supported |
| H2 | Three-cluster topology survives moderate deletion | Supported to about 87% deletion |
| H3 | Fixed-scale topology is robust to measurement jitter | Falsified |
| H4 | Reachability requires m ≥ ⌈n/T⌉ | Supported |
| H5 | Energy cost rises steeply as actuators are lost | Supported |
| H6 | Structure lowers the observed fraction needed for estimation | Supported |
| H7 | Composite frontier governs joint recoverability | Untested end to end |
| H8 | Landauer-type floor constrains restoration cost | Stated as a bound; untested physically |

## Limitations

Stylised, dimensionless models. Gaussian and linear assumptions in the estimation and control layers. The composite frontier multiplies probabilities from separate simulations. Monte Carlo sample sizes of 300 to 400 trials per point. The literature search was not exhaustive, so novelty is claimed only to the authors' knowledge.

## Citation

See `CITATION.cff`. No licence file is included yet. The authors should choose a licence before reuse.
