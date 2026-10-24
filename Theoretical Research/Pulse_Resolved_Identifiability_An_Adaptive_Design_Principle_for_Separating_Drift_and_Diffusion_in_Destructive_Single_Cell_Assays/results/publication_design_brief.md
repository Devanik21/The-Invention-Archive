# Publication design brief

## Target reader and status
Audience: computational biology, stochastic modelling, system identification, and experimental-design researchers. Format: restrained research preprint; no submission or peer-review status is claimed. The mathematical result is intentionally scoped to a scalar controlled stochastic model.

## Page architecture
- A4 page, 0.82-inch margins, 11-point body text, compact but readable paragraph spacing.
- First page contains title, author metadata, date/status, abstract, and keywords without a decorative cover.
- Running header uses a muted short title and date; centered page numbers appear in the footer.
- Main text uses a consistent section hierarchy and a single restrained accent color for links and heading accents.

## Typography and equations
- Latin Modern text/math with T1 font encoding and microtype.
- Numbered equations for the governing SDE, pulse, moments, Fisher information, and design rule.
- Symbols are introduced before use; assumptions and units are stated near the model.
- Propositions are limited to direct consequences of the stated model; proofs follow each proposition.

## Tables and figures
- Tables use booktabs-style rules, no vertical gridlines, compact type, and explicit captions.
- Eight vector figures share DejaVu Sans labels, restrained colorblind-aware colors, consistent line weights, and informative captions.
- Four plots use only synthetic/model-generated numerical outputs; two graphs show equation-derived structure; two flow diagrams show the assay and adaptive-design process.
- Figure source code is in verify_model.py; plot source data are archived in results/.

## PDF acceptance checks
- Compile from current paper.tex and paper.bib in a clean temporary directory.
- Confirm at least seven pages, embedded fonts, valid references/citations, eight figure assets, and at least five substantive tables.
- Render every page; inspect a contact sheet and every page at readable resolution.
- Bind the visual-review record to the final PDF SHA-256. Critical/major defects block PASS.
