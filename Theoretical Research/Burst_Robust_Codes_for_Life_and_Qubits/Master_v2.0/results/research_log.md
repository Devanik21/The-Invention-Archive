# Research and build log

**Release candidate:** version 2.0 · 6 November 2026

1. Re-opened the existing GitHub publisher. The previous workflow compiled Markdown with a five-page minimum, which was not adequate for the requested master manuscript.
2. Reframed the title and abstract to describe the shared event-statistics/recovery abstraction without presenting the neutrino fog as a biological repair floor.
3. Expanded the source into 17 numbered sections and six appendices, including a formal model, conditional propositions, methods, primary results, sensitivity analyses, experimental protocols, falsification criteria, and scope limitations.
4. Reproduced the original 24×24 benchmark with seed `20261106`, 10 target probabilities, and 4,000 trials per channel/point. Calibration solves for Poisson track intensity using the exact expected lattice-wide mean occupancy under edge clipping.
5. Added finite-size and track-length synthetic sensitivity sweeps, analytic pair covariance, connected-component distributions, and Wilson binomial intervals. These secondary sweeps are explicitly separate from the primary benchmark.
6. Generated twelve numbered vector scientific figures with PNG previews and a typed figure inventory: eight plots, two graphs, two flow diagrams.
7. Ran the verifier: 11 checks passed, including calibration, trial count, probability-grid size, vector/PNG inventory, finite-size/track-length tables, monotone calibration, figure-class counts, Wilson interval validity, and a dependence counterexample.
8. Compiled the LaTeX manuscript with XeLaTeX and BibTeX, resolved citations and cross-references, and confirmed 31 A4 pages. Main text occupies approximately pages 3–28 (26 pages, including the page where the conclusion ends and appendices begin); the manuscript exceeds the requested 28-page minimum.
9. Rendered every final PDF page at 160 dpi. All 31 pages were visible in four contact sheets; selected dense pages were opened at full page size. The review found no obvious clipping, missing glyphs, overflow, or empty/stranded pages.
10. Remaining caveat: this visual pass is AI-assisted layout review. It is not independent human proofreading, journal peer review, hardware validation, or biological validation.

## Reproduction

```bash
python verify_model.py
python deploy.py --self-test
python deploy.py --build
python deploy.py --validate-only
```
