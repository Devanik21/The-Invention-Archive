# Pulse-Resolved Identifiability

**An Adaptive Design Principle for Separating Drift and Diffusion in Destructive Single-Cell Assays**

Author: Devanik Debnath (Devanik21)  
Affiliation: Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India  
Academic designation: ECE '26  
ORCID: https://orcid.org/0009-0001-8310-6921  
Release date: 2026-10-24  
Status: theoretical and computational preprint; no empirical biological dataset was analyzed.

## Summary

Destructive single-cell assays usually produce unpaired population snapshots rather than trajectories of the same cells. This release studies a controlled Ornstein--Uhlenbeck model to make the identifiability limits explicit. Stationary variance alone constrains a drift--diffusion ratio. One post-pulse mean plus one variance cannot identify three unknown parameters. Two distinct post-pulse means identify the relaxation rate under the stated assumptions; calibrated observation noise then permits recovery of input gain and diffusion intensity.

The contribution is a reproducible mathematical benchmark and adaptive sampling rule, not a new general theory of optimal design, a validated biological mechanism, or a clinical claim. The closest prior work already includes information-based single-cell experiment design, Fisher-information analysis of stochastic kinetics, neural optimal transport, and stochastic trajectory inference.

## Mathematical core

The latent state follows a controlled stochastic differential equation.

$$
dx(t)=\left[-\kappa x(t)+\beta u(t)\right]dt+\sqrt{q}\,dW_t.
$$

For a rectangular pulse of known amplitude and duration, the post-pulse population mean decays exponentially. If two nonzero post-pulse means are observed at distinct times, the relaxation rate is identified by the ratio relation.

$$
\kappa=\frac{\log m(t_1)-\log m(t_2)}{t_2-t_1}.
$$

Under stationary initialization, the observed variance is

$$
v=\frac{q}{2\kappa}+r,
$$

where $r$ is observation-noise variance. Therefore the diffusion intensity is recoverable only if $\kappa$ is identified and $r$ is independently calibrated.

For a schedule $\mathcal{T}$, the design policy maximizes a regularized log-determinant Fisher-information objective.

$$
\mathcal{T}_{k+1}=\mathcal{T}_k\cup\left\{\arg\max_{t\in\mathcal{G}\setminus\mathcal{T}_k}
\log\det\left[\mathcal{I}_{\mathcal{T}_k}(\widehat{\theta}_k)+\mathcal{I}_t(\widehat{\theta}_k)+\lambda I\right]\right\}.
$$

The schedule is a local greedy design under a declared sampling budget and state-amplitude constraint. It is not guaranteed globally optimal.

## Reproducibility

Requirements: Python 3.12; NumPy 2.3.5; SciPy 1.17.0; Matplotlib 3.10.8; NetworkX 3.6.1; pypdf 5.9.0; PyYAML 6.0.3; plus a LaTeX/BibTeX toolchain for rebuilding the PDF.

From this directory run:

~~~bash
python -m py_compile verify_model.py deploy.py
python verify_model.py
python deploy.py --self-test
python deploy.py --validate
~~~

The verification script regenerates the numerical data, four plots, two equation-derived graphs, two flow diagrams, and the generated results table. Underlying data are archived in the results directory. Synthetic data are explicitly labelled; no public single-cell dataset was downloaded or analyzed.

## Visual index

- [Compiled manuscript](paper.pdf)
- [LaTeX source](paper.tex)
- [Executable verification](verify_model.py)
- [Bibliography](paper.bib)
- [Citation metadata](CITATION.cff)
- [Zenodo metadata draft](.zenodo.json)
- [Schema.org metadata](schema_article.jsonld)
- [Deployment and validation helper](deploy.py)
- [Mean-response plot](figures/fig1_mean_response.svg)
- [Variance-confounding plot](figures/fig2_variance_confounding.svg)
- [Information-schedule plot](figures/fig3_information_schedule.svg)
- [Parameter-error plot](figures/fig4_parameter_rmse.svg)
- [Model dependency graph](figures/fig5_model_dependency_graph.svg)
- [Information-coupling graph](figures/fig6_information_coupling_graph.svg)
- [Assay pipeline diagram](figures/fig7_assay_pipeline.svg)
- [Adaptive-design loop](figures/fig8_adaptive_loop.svg)
- [Research log](results/research_log.md)
- [Author-profile audit](results/author_profile_audit.md)
- [Topic-candidate comparison](results/topic_candidates.csv)
- [Literature audit](results/literature_audit.md)
- [Claim/evidence matrix](results/claim_evidence_matrix.csv)
- [Verification report](results/verification_report.json)
- [Numerical outputs](results/parameter_rmse.csv)
- [Environment report](results/environment_report.json)
- [SHA-256 manifest](results/sha256_manifest.json)
- [PDF visual-review record](results/pdf_visual_review.json)
- [Release check matrix](results/release_check_matrix.json)

## Evidence interpretation

The figures and numerical outputs are synthetic/model-generated. The tests establish consistency of the implemented equations and the selected design calculations. They do not validate the model against cell measurements, demonstrate a biological mechanism, or show clinical efficacy. The ratio identity becomes numerically unstable when the mean response is close to zero. Unknown measurement noise, donor heterogeneity, cell-state mixtures, batch drift, nonlinear dynamics, or pulse-dependent variance require additional models and controls.

## Falsification and next experiment

The proposed design principle should be narrowed or rejected if the exponential post-pulse relation fails on held-out time points, if estimated parameters change materially under plausible measurement-noise calibration errors, if a nonlinear or mixture model explains the data equally well, or if the adaptive schedule fails to improve held-out parameter recovery against simple baselines. The next step is a pre-registered benchmark on a public perturbation time course with donor-aware validation and comparison against established trajectory and perturbation-prediction methods.

No DOI has been minted. The Zenodo metadata file is a draft only; no deposit or external indexing is implied.
