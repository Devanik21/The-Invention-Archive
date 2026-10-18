# Research-cycle log — 2026-10-18

## Selected question
**Decoding Deep Time Through a Diagenetic Channel: An Information-Theoretic Recoverability Framework for Speleothem Climate Archives**

The candidate was supplied by the author. Ten candidate directions were compared to satisfy the workflow's autonomous-topic check; the submitted candidate had the strongest blend of significance, conceptual integration, formal tractability, and a feasible first verification. Scores are heuristic structured judgments, not an objective ranking or proof of novelty.

## Work performed
1. Audited the public author footprint and repository identity; unresolved records are marked explicitly in `author_profile_audit.md`.
2. Searched relevant literature on the current SISALv3 compilation, calcite fabric logging, diagenetic age effects, cave proxy-system models, Bayesian multi-proxy reconstruction, and microbial fatty-acid sources. Search log and scope are in `literature_audit.md`.
3. Established a bounded novelty claim: recoverability/identifiability gate with imperfect fabric side information, tested for accuracy and interval coverage under synthetic channel shift. No absolute priority claim.
4. Implemented a transparent three-proxy synthetic forward model with three OLS baselines (single proxy, multi-proxy, multi-proxy + noisy fabric class) and two held-out scenarios.
5. Executed `python verify_model.py`: PASS, 10 assertions, deterministic seed 20261018, Python 3.13.5, NumPy 2.3.5, Matplotlib 3.10.8. It emitted the tables, time series, sensitivity sweep, and eight vector figures. No SISAL rows or measured cave records were imported.
6. Drafted a 16-page manuscript with 7 substantive tables, 4 data-bearing synthetic plots, 2 conceptual/structural graph visuals, and 2 flow diagrams. The required disclosure is before references.
7. Local PDF rendering, per-page visual inspection, fonts/layout, metadata, isolated deployment self-tests, manifest hashes, and remote release checks are separate closure items; each is recorded in `release_check_matrix.json` after actual outcome.

## Scientific stop rule
A real-data claim is blocked until at least one suitable dataset jointly aligns proxy records, chronology, per-sample fabric/alteration metadata, and a defensible instrumental/monitoring target. If this alignment is unavailable, results must remain feasibility analysis or synthetic stress testing.

## Result that motivates caution
The channel-aware OLS estimator lowers synthetic RMSE relative to the multi-proxy baseline by a small amount: 0.2739 to 0.2686 in-distribution, and 0.3132 to 0.3007 in the defined shift. Its shifted 90% interval coverage is 0.8471, not the nominal 0.90. This is not evidence of calibration under distribution shift; it is a reason to include calibration as an independent gate.

7. Extended the literature audit to the 2026 SISAL_monv1 cave-monitoring discussion preprint (DOI 10.5194/essd-2026-31). Its linked repository returned an anti-bot proof-of-work challenge in this execution; therefore the underlying rows and overlap with sample-level fabric/proxy/chronology data were not analyzed. Updated the manuscript, bibliography, and README to identify this as a candidate resource without implying dataset access or empirical results.

8. Rebuilt the current manuscript from `paper.tex` and `paper.bib` with the deployment helper; the resulting PDF has 16 pages.
9. Reviewed every rendered page after the figure-layout fixes, recorded the final PDF SHA-256, and confirmed all listed PDF fonts are embedded with no Type 3 fonts.
10. Re-ran syntax checks, all 10 synthetic model assertions, 3/3 fail-closed deployment fixtures, and the metadata/link/PDF preflight. Local validation passes; the empirical-data gate remains blocked and remote publication is not claimed.
