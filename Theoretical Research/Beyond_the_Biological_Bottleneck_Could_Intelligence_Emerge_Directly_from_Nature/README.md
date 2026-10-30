# Beyond the Biological Bottleneck

**A Framework for Natural Computation, Abiotic Agency, and Distinguishing Emergence from Construction**
Research preprint · 30 October 2026 · theoretical framework + synthetic identifiability demonstration

## Abstract

Earth provides one observed route from natural chemistry through biological evolution to technological intelligence. This paper asks whether that route establishes a universal biological bottleneck. It distinguishes physical dynamics, computation, adaptive computation, minimal agency, and functional intelligence; reviews evidence for computation in non-neural physical substrates; and separates the substrate of a system from its causal origin. Three origin hypotheses are defined: direct abiotic emergence, construction by an earlier biological or artificial intelligence, and a non-intelligent natural decoy. A simple proposition shows that if direct-emergence and constructed-origin hypotheses induce identical likelihoods for all observed functional behaviours, then the Bayes factor is one and those behaviours cannot update their relative origin probabilities. A reproducible synthetic demonstration illustrates the distinction between capability detection and origin inference. The work does not claim that abiotic intelligence exists, that any candidate exotic substrate is intelligent, or that biology is unnecessary. It proposes a staged, falsifiable programme for testing these questions.

## Main artifacts

- `paper.pdf` — compiled manuscript.
- `paper.tex` — LaTeX source.
- `paper.bib` — bibliography metadata.
- `verify_model.py` — deterministic synthetic demonstration and figure generator.
- `deploy.py` — build and release preflight.
- `literature_review.md` — scoped evidence map, gaps, and source notes.
- `CITATION.cff` — citation metadata.
- `.zenodo.json` — draft deposition metadata; no DOI or deposit is implied.
- `schema_article.jsonld` — ScholarlyArticle metadata.
- `figures/` — source-generated vector and raster diagrams.
- `results/` — synthetic demonstration summary, compact sample preview, PDF validation and hashes.

## Principal contribution

The paper's main contribution is not the claim that chemistry, plasma, minerals, or planets are intelligent. It is a disciplined separation of two inference tasks:

1. **Capability inference:** does a system perform reproducible computation, learn from interaction, transfer to novel conditions, and exert intervention-sensitive control?
2. **Origin inference:** did the system emerge directly from abiotic processes, was it built by an earlier intelligence, or is it a natural non-intelligent phenomenon that mimics intelligence-like signatures?

Functional competence and causal provenance are different variables. Where their observable likelihoods coincide, behaviour alone cannot identify origin.

## Reproduce

Python 3.10+ with NumPy and Matplotlib; LaTeX with `latexmk` and `pdflatex` for the PDF build.

```bash
python verify_model.py
python deploy.py --self-test
python deploy.py --build
python deploy.py --validate-only
```

The script uses fixed seed `20261030` and generates a three-class synthetic dataset with 10,000 samples per class. The CSV is a compact 100-row-per-class preview; the complete synthetic arrays are regenerated on execution. All distribution parameters are hand-specified for an identifiability illustration. Results do not estimate real-world detection performance, priors, false-positive rates, or the frequency of abiotic intelligence.

## Scope and epistemic status

- Physical/chemical computation has experimental precedent, including chemical reservoir computing.
- Such computation does not establish learning, agency, intelligence, consciousness, or natural emergence of an autonomous intelligent entity.
- Plasma, mineral, atmospheric, and planetary intelligence are treated as research hypotheses or speculative candidate media, not demonstrated examples.
- The origin-identifiability proposition is conditional and elementary: it follows from equal likelihoods under competing hypotheses.
- The literature search is targeted, not systematic. No priority or exhaustive novelty claim is made.
- No empirical extraterrestrial observations, biological samples, or mission data were used.

## Relationship to the companion preprint

The repository's *Interrogating Alien Life* preprint addresses active probing of repair-like dynamics against abiotic decoys. The present paper addresses a different layer: whether evidence of non-biological adaptive capability can establish how that capability originated. The two frameworks are complementary, not interchangeable.

## Citation

Debnath, D. (2026). *Beyond the Biological Bottleneck: Could Intelligence Emerge Directly from Nature? A Framework for Natural Computation, Abiotic Agency, and Distinguishing Emergence from Construction*. Research preprint, 30 October 2026.
