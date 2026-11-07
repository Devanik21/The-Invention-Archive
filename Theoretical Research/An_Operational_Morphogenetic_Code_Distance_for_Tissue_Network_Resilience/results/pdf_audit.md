# PDF build and visual audit

- Source: `paper.tex`; figures: `figures/*.pdf`.
- Build: clean temporary directory, `pdflatex -interaction=nonstopmode -halt-on-error paper.tex` twice; the second pass resolves cross-references.
- Paper format: A4, 45 pages (checked with `pdfinfo`); text extraction checked with `pdftotext`.
- Exact PDF SHA-256: `635f1dc75d86d808bb045fe571234083eb54cfedc59c89aa1bd96b29c354083e`.
- Required visuals: Tables 1–13 (at least five distinct substantive tables); Figures 1–8 (four plots, two network/structure diagrams, two flow diagrams).
- Inspection: all 45 pages were rendered at 120 dpi and reviewed in six contact sheets (pages 1–8, 9–16, 17–24, 25–32, 33–40, 41–45). The title page, equations, tables, plots, network diagrams, flow diagrams, disclosure, discussion, and bibliography were inspected. No clipping, missing glyphs, unresolved citations, or empty sections were observed.
- Scope limitation: this is a review of the locally compiled PDF. A separately rebuilt GitHub Actions PDF must be fetched and reviewed against its own SHA-256 before remote visual verification can pass.
## GitHub Actions build

- Source commit: `778a9dbb74a1c12eb25e8a60a42b86322e75d774`.
- Rebuilt PDF pages: 45.
- Rebuilt PDF SHA-256: `bfd863e8fb8568c5c1b80a2b7af5c62cfce0ca8b2e09e5b5612a134409b4376b`.
- Remote page-by-page visual review: **PENDING**; no remote visual pass is claimed.

## GitHub Actions build

- Source commit: `125cfde7b1bf67bdcc63e6499795f460e3c7b80e`.
- Rebuilt PDF pages: 45.
- Rebuilt PDF SHA-256: `4c5e01fe85b0e83018f2e0fc192c840653d52de53abc164a5d0bb83125cbe4fc`.
- Remote page-by-page visual review: **PENDING**; no remote visual pass is claimed.

## Exact GitHub commit verification

- Artifact commit: `cb805b455863577743f8d7d2a0ebfba1f21e4a32`.
- The workflow independently exported this commit with `git archive` and rechecked the PDF digest, PDF page count, manifest, all nine root files, generated figure assets, and the 14/14 verification record.

## GitHub-built PDF — final visual review

- Build workflow: [run 42357164937](https://github.com/Devanik21/The-Invention-Archive/actions/runs/42357164937), completed successfully.
- Source commit: `125cfde7b1bf67bdcc63e6499795f460e3c7b80e`.
- Artifact commit: `cb805b455863577743f8d7d2a0ebfba1f21e4a32`.
- PDF page count: 45.
- PDF SHA-256: `4c5e01fe85b0e83018f2e0fc192c840653d52de53abc164a5d0bb83125cbe4fc`.
- Visual review: **PASS**. All 45 pages were inspected using five SHA-bound contact sheets generated from the committed PDF and stored in `results/remote_pdf_review/`.
- Review covered page geometry, content continuity, section hierarchy, tables, equations, figures, appendices, and references. No apparent clipping, overlap, or page-boundary truncation was found.
- Exact commit archive, manifest, required file layout, figure set, and 14/14 synthetic verification report passed in the same successful workflow run.

The local and GitHub-built PDFs have different byte hashes because they were separately compiled; this remote review is bound to the GitHub PDF SHA above, not the local PDF hash.
