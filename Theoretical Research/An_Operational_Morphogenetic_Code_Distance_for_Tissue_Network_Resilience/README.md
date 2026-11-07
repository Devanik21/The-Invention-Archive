# An Operational Morphogenetic Code Distance for Tissue-Network Resilience

**Research preprint · 7 November 2026 · Version 1.0.0**  
**Author:** Devanik Debnath · Electronics and Communication Engineering, National Institute of Technology Agartala, India  
**Archive:** [Theoretical Research/An_Operational_Morphogenetic_Code_Distance_for_Tissue_Network_Resilience](https://github.com/Devanik21/The-Invention-Archive/tree/main/Theoretical%20Research/An_Operational_Morphogenetic_Code_Distance_for_Tissue_Network_Resilience)

> **Scope statement.** This release defines a model-relative graph-theoretic quantity for source–target communication resilience. Its numerical studies are synthetic computational experiments. They are not planarian experiments, measured tissue networks, clinical evidence, or a demonstration that living tissue is a quantum error-correcting code.

## Abstract

Planarian research shows that neural, gap-junction-mediated and bioelectric signals can influence regenerative polarity, and that brief physiological perturbations can have persistent consequences. We ask whether the structural vulnerability of a putative morphogenetic communication task can be quantified by the least costly set of cells whose loss disconnects a prescribed source region from a target region.

We define **Morphogenetic Code Distance (MCD)** as a weighted minimum vertex cut on an explicitly specified communication graph. This is a network-resilience measure, not the quantum code distance of a stabilizer code. The framework keeps three diagnostics separate: graph-cut cost, cellular-sheaf compatibility/Hodge residuals, and the spectral gap of an anchored diffusion operator. A finite-dimensional theorem characterizes when an anchor penalty removes all neutral compatible modes. A transparent synthetic two-module graph has an exact five-relay minimum cut. Under one specified uniform 20-node deletion protocol, 612 of 10,000 trials disconnect the anchors (6.12%), compared with an exact hypergeometric probability of 6.5325%; a separately defined module-only control has 0/10,000 disconnections. In a distinct synthetic NNLS inverse problem, a 66-parameter design matrix is full column rank and the mean relative error at derivative-noise SD 0.12 is 1.10% under the stated synthetic model.

These numbers validate code paths and mathematical identities only. No biological MCD has been measured. The paper supplies a preregisterable experimental design, observables, alternatives, and falsification criteria rather than claiming that graph cuts already explain regeneration.

## Main contributions

1. An operational, weighted source–target vertex-cut definition with explicit terminals, eligible nodes, and intervention costs.
2. A node-splitting max-flow/min-cut construction and proof for finite graphs with positive costs and sufficiently large protected-arc capacity.
3. A cellular-sheaf formulation that distinguishes exact coboundaries from non-exact cycle-level residuals.
4. A pinned-Laplacian theorem: for a symmetric positive-semidefinite compatibility operator, anchor penalties remove all neutral modes precisely when the compatible kernel intersects the unpinned subspace only at zero.
5. A reproducible synthetic Monte Carlo benchmark, spectral analysis, and a physics-constrained NNLS inverse-estimation stress test.
6. A prospective biological measurement and falsification plan, explicitly separated from performed experiments.

## Mathematical objects

For a graph $G=(V,E)$, prescribed source and target sets $S,T$, eligible deletion vertices $U$, and strictly positive costs $c_v$, define

$$
\mathrm{MCD}(G,S,T,c)=\min_{C\subseteq U:\,no S--T path\text{ in }G-C}\sum_{v\in C}c_v.
$$

For a degree-zero cellular-sheaf coboundary $\delta^0:C^0(K;\mathcal{F})\to C^1(K;\mathcal{F})$ and diagonal positive edge weights $W$,

$$
L_0=(\delta^0)^\top W\delta^0\succeq0.
$$

The exact syndrome $\sigma=\delta^0x$ is a coboundary and hence represents the zero class in $H^1$ whenever the cochain complex is valid. A nonzero harmonic residual is a different object: it is the component of measured edge data not explained by an exact coboundary under the chosen model.

For an anchor selector $P\succeq0$ and gain $\alpha>0$, the pinned operator is

$$
A=L_0+\alpha P.
$$

It is positive definite exactly when $\ker(L_0)\cap\ker(P)=\{0\}$.

## Selected synthetic results

| Test | Observed result | Interpretation |
|---|---:|---|
| Stylized graph | 35 vertices, 360 edges | Synthetic, deliberately structured two-module graph |
| Weighted MCD | 5.0 | Five unit-cost relay vertices form an exact minimum separator |
| Uniform deletion | 612 / 10,000 = 6.12% | Compared with exact 6.5325% hypergeometric probability |
| Balanced module-only control | 0 / 10,000 | Control excludes the relays; it is not directly comparable to the uniform protocol |
| Pinned diffusion | $\lambda_{\min}=0$ after all five relays are removed | The target-side component is then unpinned and disconnected |
| Sheaf test | $\dim H^1=2$ for the specified $C_4$ with trivial $\mathbb{R}^2$ stalks | Finite-dimensional cochain-complex check, not a tissue measurement |
| NNLS inverse model | rank 66/66; condition number 4.2338 | Identifiable for the synthetic design matrix under the specified forward model |
| NNLS at noise SD 0.12 | mean relative error 1.1035% | Five synthetic replicates at that noise level |

All details, seeds, trial counts, CSV inputs/outputs, and assertions are in `verify_model.py` and `results/`.

## Reproducibility

The verification script runs from any working directory because output paths are resolved relative to the script:

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 networkx==3.6.1 matplotlib==3.10.8
python verify_model.py
```

The PDF is built from `paper.tex` and the vector figures with two or more `pdflatex` passes (the manuscript contains a manually curated bibliography; `paper.bib` is provided as a parallel machine-readable citation file). A Linux build requires a LaTeX installation with `amsmath`, `mathtools`, `booktabs`, `tabularx`, `longtable`, `graphicx`, `hyperref`, `natbib`, `cleveref`, `siunitx`, `listings`, `pdflscape`, `microtype`, and `lmodern`, plus Poppler utilities for inspection.

```bash
python -m py_compile verify_model.py deploy.py
python verify_model.py
python deploy.py --self-test
python deploy.py --validate-only
```

The Actions build workflow regenerates figures, numerical outputs, and `paper.pdf` from the checked-in source and commits those artifacts. The companion verification workflow checks the resulting artifact commit.

## Visual and evidence index

- [Compiled manuscript](paper.pdf)
- [LaTeX source](paper.tex)
- [Python verification and figure generator](verify_model.py)
- [BibTeX references](paper.bib)
- [Verification report](results/verification_report.json)
- [Verification summary](results/verification_summary.txt)
- [Candidate topic screen](results/candidate_screen.csv)
- [Literature audit](results/literature_audit.csv)
- [Claim–evidence matrix](results/claim_evidence_matrix.csv)
- [Author-profile audit](results/author_profile_audit.md)
- [Workflow and provenance log](results/workflow_log.md)
- [PDF audit and visual review](results/pdf_visual_review.json)
- [Figure 1: lesion topology](figures/fig1_lesion_topology.png)
- [Figure 2: pinned spectral gap](figures/fig2_pinned_spectral_gap.png)
- [Figure 3: NNLS noise sensitivity](figures/fig3_nnls_noise_sensitivity.png)
- [Figure 4: pinned diffusion trajectories](figures/fig4_pinned_diffusion_traces.png)
- [Figure 5: structural network](figures/fig5_structural_network.png)
- [Figure 6: cellular-sheaf stalk complex](figures/fig6_sheaf_stalk_graph.png)
- [Figure 7: inference pipeline](figures/fig7_inference_flow.png)
- [Figure 8: prospective validation flow](figures/fig8_validation_flow.png)

Each figure is stored as both vector PDF and PNG preview; the underlying numerical series is saved in `results/`.

## Limitations and non-claims

- The synthetic graph is not a reconstruction of planarian anatomy, innexin distribution, voltage propagation, or measured cellular connectivity.
- MCD is model-relative and depends on graph inference, source/target choice, eligible vertices, and cost calibration.
- A graph cut is not sufficient to characterize dynamic controllability, redundancy in unmeasured pathways, delays, nonlinear feedback, or repair.
- The NNLS experiment assumes a correct known candidate graph, exact state derivatives, independent Gaussian noise, and an identifiable linear diffusion model.
- No ageing model, cancer model, therapy, treatment parameter, wet-lab experiment, or human/animal outcome is reported.
- The QEC vocabulary is only an analogy for disruption of protected function; no biological stabilizer code, logical operator, anyon, or code space is defined.
- The repository archive is a public research preprint record, not peer review, journal publication, DOI registration, or institutional endorsement.

## Citation

Devanik Debnath (2026). *An Operational Morphogenetic Code Distance for Tissue-Network Resilience*. Version 1.0.0. GitHub research archive, `Devanik21/The-Invention-Archive`.

See [`CITATION.cff`](CITATION.cff). No DOI has been issued or claimed.

## Research-process disclosure

Co-equal research-process credit is given to Devanik Debnath and Xylia, the autonomous multi-model research engine coordinating AI-assisted discovery, synthesis, and verification. Devanik Debnath retains responsibility for the final manuscript, scientific claims, citations, and released artifacts.