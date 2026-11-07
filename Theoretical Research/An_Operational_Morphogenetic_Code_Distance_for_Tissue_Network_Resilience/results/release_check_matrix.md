# Release check matrix — 2026-11-07

**Release status:** `PUBLISHED_WITH_CONTROL_LIMITATIONS`

| Gate | Status | Evidence |
|---|---|---|
| Full Saturday workflow | PASS | Full 702-line workflow retrieved and read. |
| Source and author audit | PASS | Public profile and source provenance checked; limitations documented. |
| Literature and topic screening | PASS | Twelve foundational/primary sources; ten candidate directions screened. |
| Manuscript | PASS | 45-page A4 preprint; 13 tables and 8 figures. |
| Mathematical/numerical verification | PASS | 14/14 deterministic mathematical and synthetic checks passed. |
| Deployment self-test | PASS | Fail-closed missing/extra-file checks passed. |
| Metadata | PASS | CITATION.cff, BibTeX, Zenodo metadata draft, and JSON-LD included; no DOI claimed. |
| Local PDF visual review | PASS | All 45 locally built pages reviewed; local SHA-256 `635f1dc75d86d808bb045fe571234083eb54cfedc59c89aa1bd96b29c354083e`. |
| GitHub build | PASS | Actions run [42357164937](https://github.com/Devanik21/The-Invention-Archive/actions/runs/42357164937); 45-page PDF SHA-256 `4c5e01fe85b0e83018f2e0fc192c840653d52de53abc164a5d0bb83125cbe4fc`. |
| Exact artifact commit verification | PASS | Artifact commit `cb805b455863577743f8d7d2a0ebfba1f21e4a32`; manifest, PDF, 9 root files, figures, and 14/14 report checked from a fresh Git archive. |
| Remote PDF visual review | PASS | Pages 1–45 inspected in five SHA-bound contact sheets under `results/remote_pdf_review/`. |
| Parent research index | PASS | New release row committed in `Theoretical Research/Readme.md`. |
| SHA-256 manifest | PASS | Manifest refreshed after the final review and audit records. |
| Branch protection/ruleset audit | BLOCKED | Administrative API returned HTTP 403; actual repository setting is unknown. |
| Semantic release tag | NOT RUN | No tag-write operation was available; immutable commit SHA is recorded above. |

**GitHub PDF:** 45 pages; SHA-256 `4c5e01fe85b0e83018f2e0fc192c840653d52de53abc164a5d0bb83125cbe4fc`.

**Source commit:** `125cfde7b1bf67bdcc63e6499795f460e3c7b80e`  
**Artifact commit:** `cb805b455863577743f8d7d2a0ebfba1f21e4a32`  
**Exact-verification audit commit:** `0caebc40d167cdbdabccfd392749f9d08aec1797`

## Scope of evidence

All numerical results are synthetic benchmark outputs from the declared graph and inverse-problem model. No biological tissue MCD was measured, no wet-lab or patient experiment was performed, and no therapeutic, ageing-reversal, or cancer-normalization claim is made. The manuscript is a research preprint, not a peer-reviewed publication; no DOI is assigned.

## Remaining repository-control limitations

- Branch protection and ruleset state remains unknown because the connected integration cannot access the administrative endpoint (HTTP 403).
- No annotated semantic tag was created. The artifact commit SHA above is the immutable release identifier.
