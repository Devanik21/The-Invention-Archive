# PDF build and visual audit

- Source: `paper.tex`; figures: `figures/*.pdf`.
- Build: clean temporary directory, `pdflatex -interaction=nonstopmode -halt-on-error paper.tex` twice; the second pass resolves cross-references.
- Paper format: A4, 45 pages (checked with `pdfinfo`); text extraction checked with `pdftotext`.
- Exact PDF SHA-256: `635f1dc75d86d808bb045fe571234083eb54cfedc59c89aa1bd96b29c354083e`.
- Required visuals: Tables 1–13 (at least five distinct substantive tables); Figures 1–8 (four plots, two network/structure diagrams, two flow diagrams).
- Inspection: all 45 pages were rendered at 120 dpi and reviewed in six contact sheets (pages 1–8, 9–16, 17–24, 25–32, 33–40, 41–45). The title page, equations, tables, plots, network diagrams, flow diagrams, disclosure, discussion, and bibliography were inspected. No clipping, missing glyphs, unresolved citations, or empty sections were observed.
- Scope limitation: this is a review of the locally compiled PDF. A separately rebuilt GitHub Actions PDF must be fetched and reviewed against its own SHA-256 before remote visual verification can pass.