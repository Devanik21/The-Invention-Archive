# Workflow and provenance log — 2026-11-07

## Workflow source

The complete 702-line `Xylia_Research_Workflow_Run_Closure_Enhanced.txt` workflow was retrieved from the Library and read in full in consecutive ranges. The applicable requirements were applied: author-profile audit; primary literature; 8–12 candidate screen; P1–P5 classification; mathematical derivations; reproducible verification; minimum 5 tables, 4 plots, 2 structural graphs and 2 flow diagrams; >=25-page user requirement; nine root-level core files plus `figures/` and `results/`; CFF/BibTeX/Zenodo JSON/JSON-LD; local deployment tests; parent index preservation; real GitHub commit; and exact-SHA remote verification.

## Topic-selection strategy

The original proposal's wide scope (quantum stabilizer codes, sheaf cohomology, non-Hermitian skin effect, thermodynamics, renormalization, AdS/CFT, spin glasses, anyons, game theory and sliding-mode control) was screened as an initial hypothesis set, not taken as established biology. Research-creativity breadth was set to 0.90 as a search/prompting strategy; no model sampling-temperature control was exposed, so no temperature change is claimed. Ten candidate directions were compared in `candidate_screen.csv`. The operational weighted vertex-cut definition survived because it has a finite mathematical object, exact algorithm, transparent counterexamples and a falsifiable prospective biological test.

## Revisions made to the supplied draft

1. Replaced the original broad “universal fault-tolerant persistence principle” title with a model-relative network-resilience paper.
2. Removed claims that five or twenty cells demonstrate a universal tissue code distance; the new MCD is explicitly a weighted minimum vertex cut in a graph.
3. Corrected the cohomology distinction: if a syndrome is defined as `sigma = delta_0 x`, it is exact and its class in `H^1` is zero (provided `delta_1 delta_0 = 0`). Non-exact measured edge residuals are analyzed separately.
4. Removed the unsupported claim of a cellular face/plaquette operator from a graph-only model; a separate four-cycle cellular-sheaf example explicitly constructs a valid cochain complex.
5. Removed hard-coded Ollivier–Ricci curvatures and the claim that a five-cell lesion represented a universal homological cut.
6. Removed the imposed ageing/SMC rejuvenation results and cancer/anyon re-confinement outputs. The supplied script set many cohort values, treatment outcomes, growth thresholds and percentages directly; they were not inferred by dynamical simulations.
7. Replaced forced phenotype outcomes with a minimum-cut algorithm, exact hypergeometric reference, Monte Carlo lesion protocols, pinned-Laplacian spectrum, sheaf projection check and NNLS recovery under explicitly generated noise.

## Reproducible computation

- Seed: `20261107`.
- Main script: `verify_model.py`; dependencies are NumPy, SciPy, NetworkX, Matplotlib.
- Local command: `python -m py_compile verify_model.py && python verify_model.py`.
- The script was executed successfully from an unrelated working directory. It reported 14/14 assertions passed, MCD=5.0 on the stylized graph, 612/10,000 uniform deletion failures, exact probability 0.0653251, 66/66 NNLS design rank and eight regenerated visual assets.
- The numeric outcomes in this log are synthetic model outputs, not empirical biological observations.

## Document production

- A4 LaTeX manuscript: `paper.tex`; bibliography: `paper.bib` plus manually rendered reference list in the source.
- Compiled using `pdflatex -interaction=nonstopmode -halt-on-error` twice after copying the source and vector figures to a clean build directory.
- Current local PDF: 45 pages A4.
- Visual suite: four numerical plots, two structural graph figures and two flow diagrams, each preserved as PDF and PNG; all are referenced in the manuscript.

## Scope boundaries

No biological datasets were downloaded, no laboratory work was performed, no treatments were proposed for human use, and no ageing/cancer clinical effect is claimed. The AI/data-driven contribution is a transparent physics-constrained inverse-estimation pipeline; the NNLS solver is a known baseline and no new neural architecture is claimed.

## GitHub publication and final verification

- Source/workflow commit: `125cfde7b1bf67bdcc63e6499795f460e3c7b80e`.
- Successful build and verification run: [42357164937](https://github.com/Devanik21/The-Invention-Archive/actions/runs/42357164937).
- Artifact commit: `cb805b455863577743f8d7d2a0ebfba1f21e4a32`; exact commit archive verification passed for the nine root files, manifest, PDF, figure assets, contact sheets, and 14/14 synthetic numerical checks.
- Exact-verification audit commit: `0caebc40d167cdbdabccfd392749f9d08aec1797`; the workflow independently checked its final archive before reporting success.
- GitHub PDF: 45 pages; SHA-256 `4c5e01fe85b0e83018f2e0fc192c840653d52de53abc164a5d0bb83125cbe4fc`.
- Five contact sheets covering pages 1-45 were generated from the exact GitHub-built PDF and visually reviewed. No apparent clipping, overlap, or page-boundary truncation was found.
- Earlier CI attempts exposed a missing TeX `siunitx` dependency and a concurrent-push race; both were corrected in the final successful workflow. The final build installed `texlive-science` and used a non-force-push fetch/rebase retry.
- Separate repository workflows triggered by the parent-index commit were unrelated release jobs; their status and distinct causes were not used as evidence for this paper's validation.
- Branch-protection/ruleset state remains unknown (HTTP 403); no annotated tag was created. The artifact commit SHA is the immutable release identifier.

## Evidence limits

All results remain mathematical/synthetic. No biological MCD, treatment outcome, ageing reversal, or cancer normalization was measured. The release is a preprint, not peer-reviewed, and no DOI is claimed.
