# Publication design brief

- **Audience:** mathematical biology, dynamical systems, mathematical physics, and tame geometry readers.
- **Status:** theoretical research preprint; not peer reviewed and not submitted to a journal.
- **Page system:** A4, restrained margins, running header, page numbers; 29 pages in the current build.
- **Typography:** TeX Gyre Termes body family with NewTX mathematical fonts; embedded subset fonts confirmed in the generated PDF. Body serif, sans-serif figure labels, mono for code-like tokens.
- **Hierarchy:** compact title block, abstract/keywords/status notes, two-page contents, numbered section hierarchy, clearly captioned theorem/table/figure blocks, technical appendices, references.
- **Color:** muted green/blue/orange accents only in figures; body monochrome for print legibility.
- **Equations:** numbered LaTeX displays with notation defined in surrounding prose.
- **Tables:** five tables with light rules and no unnecessary vertical grid; captions above.
- **Figures:** twelve reproducible vector PDF assets with PNG previews; captions distinguish synthetic data, exact algebraic identities, and conceptual diagrams.
- **Whitespace/layout:** float placement uses fixed, local figure/table placements for reading order; section hierarchy and consistent running head. The title/contents pages use controlled whitespace rather than a decorative cover.
- **Build:** `pdflatex`, `bibtex8`, then two further `pdflatex` passes. The build output and PDF checks are recorded in `results/`.
