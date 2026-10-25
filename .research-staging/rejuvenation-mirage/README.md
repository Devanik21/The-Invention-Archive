# The Rejuvenation Mirage

**Identifiability limits of epigenetic-clock reset and an active-design rule for functional repair**

**Author:** Devanik Debnath · National Institute of Technology Agartala, India · ECE '26  
**Status:** Theoretical + synthetic-computational preprint, v1.0 · 25 October 2026  
**ORCID:** https://orcid.org/0009-0001-8310-6921

> **Scope boundary:** This release contains an explicit mathematical model and a deterministic synthetic benchmark. No biological dataset, wet-lab experiment, clinical intervention, or organismal lifespan assay was performed. The paper does not establish that any intervention rejuvenates cells or extends lifespan.

## Central question

If an intervention lowers an epigenetic clock, what identifies a change in functional state rather than a direct change to the molecular features used by the clock?

The model separates a functional repair input \(u\), a marker-targeting input \(q\), a clock score \(C\), a functional endpoint \(F\), and an identity/safety endpoint \(I\). The local synthetic contrasts are:

$$
\Delta C = \kappa u + \alpha q + \epsilon_C
$$

$$
\Delta F = \kappa u - \rho q + \epsilon_F
$$

Here, \(\kappa\) is the functional response of interest, \(\alpha\) is a direct marker-to-clock effect, and \(\rho\) permits the marker-targeting input to affect the functional readout directly. These are model parameters, not measured biological coefficients.

The paper proves a scalar observational-equivalence result, derives a rank condition for local identification, and uses a constrained information-gain planner to choose informative synthetic experiment arms. The planner is an auditable design optimizer; it does not infer molecular mechanisms or recommend biological targets.

## Current reproducible benchmark

<!-- GENERATED_RESULTS_START -->
The current-run numbers will be populated from the deterministic simulation during the release build.
<!-- GENERATED_RESULTS_END -->

The numbers are synthetic model outputs, not biological observations. The exact seed, parameters, simulation summary, numerical checks, and data feeding the figures are archived under results/.

## Main findings and limits

- A fixed scalar contrast can confound a functional response with a direct marker effect. More samples reduce variance but do not remove structural non-identifiability.
- Within-subject differences cancel a time-invariant additive inherited set point under the model assumptions; they do not eliminate genotype-by-treatment interactions or cell-composition shifts.
- Independent perturbation variation and an independent functional endpoint can improve identification, provided their measurement models are defensible.
- Cell identity and safety are separate outcomes. A clock reduction is not a safety guarantee.
- The active-design result is exact only for the declared finite candidate set and local linear model. It is not a biological efficacy or safety result.

## Reproduce

From this directory, using Python 3.12 and the pinned dependencies in the workflow:

~~~bash
python verify_model.py
python deploy.py --self-test
python deploy.py --build
python deploy.py --validate-only
~~~

The verification script has no network or credential requirement. It regenerates the synthetic source data, vector figures, model checks, candidate-topic scores, literature audit, and generated LaTeX macros. The deployment script compiles the current paper.tex and paper.bib in a temporary build directory and blocks on failed checks. A final visual-review pass is separate from automated compilation and preflight.

## Paper and artifacts

- [Compiled preprint](paper.pdf)
- [LaTeX source](paper.tex)
- [Bibliography](paper.bib)
- [Model verification and figure generator](verify_model.py)
- [Deployment and validation script](deploy.py)
- [Citation metadata](CITATION.cff)
- [Zenodo-compatible metadata payload; no deposit performed](.zenodo.json)
- [Schema.org ScholarlyArticle metadata](schema_article.jsonld)
- [Figure inventory](results/figure_inventory.csv)
- [Synthetic source data](results/synthetic_source_data.csv)
- [Verification report](results/verification_report.json)
- [Claim/evidence matrix](results/claim_evidence_matrix.csv)
- [Candidate topic comparison](results/candidate_topic_matrix.csv)
- [Literature audit](results/literature_audit.csv)
- [Publication design brief](results/publication_design_brief.md)
- [Publication-quality report](results/publication_quality_report.md)
- [PDF visual-review record](results/pdf_visual_review.json)
- [Release check matrix](results/release_check_matrix.json)
- [SHA-256 manifest](results/sha256_manifest.json)

The twelve numbered figures are available as vector PDFs and SVG files in the [figures directory](figures/). Start with the [framework flow](figures/fig01_framework.pdf), [causal graph](figures/fig02_causal_dag.pdf), [identifiability spectrum](figures/fig04_identifiability_spectrum.pdf), and [validation pipeline](figures/fig12_validation_pipeline.pdf). They include eight quantitative plots, two graph-structured views, and two flow diagrams. Every data-bearing figure is generated from code, with source values archived in results/.

## Selected literature

1. Horvath (2013), *DNA methylation age of human tissues and cell types*, Genome Biology. DOI: 10.1186/gb-2013-14-10-r115.
2. Liesenfelder et al. (2025), *Epigenetic editing at individual age-associated CpGs affects the genome-wide epigenetic aging landscape*, Nature Aging. DOI: 10.1038/s43587-025-00841-1.
3. Sengstack et al. (2026), *Systematic identification of single transcription factor perturbations that drive cellular and tissue rejuvenation*, PNAS. DOI: 10.1073/pnas.2515183123.
4. Lim (2026), *GIA: Germline-Informed Aging with AlphaGenome Finds Genetically Regulated CpGs*, arXiv:2609.17801. Preprint; treated as preliminary.
5. Chaloner and Verdinelli (1995), *Bayesian experimental design: a review*, Statistical Science. DOI: 10.1214/ss/1177009939.

Full references and verified bibliographic fields are in paper.bib. The literature audit is focused, not exhaustive; the paper makes no absolute priority claim.

## Research-process credit and responsibility

Co-equal research-process credit is given to Devanik Debnath and Xylia, the autonomous multi-model research engine coordinating AI-assisted discovery, synthesis, and verification. Devanik Debnath retains responsibility for the final manuscript, scientific claims, citations, and released artifacts.

## Publication status

This is a GitHub-archived preprint, not a peer-reviewed or journal-accepted paper. .zenodo.json is prepared metadata only; no external deposit or DOI minting is implied.
