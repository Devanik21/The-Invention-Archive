# Release check matrix — initial local closure

| Gate | Status | Evidence |
|---|---|---|
| Full Saturday workflow read | PASS | Full 702-line workflow read. |
| Literature, topic and author audit | PASS | Source audit, ten-candidate screen and 12-record bibliography. |
| Mathematical and synthetic verification | PASS | 14 deterministic checks passed; no biological validation claimed. |
| 45-page manuscript and visual minimums | PASS | 13 tables and eight figures (four plots, two network diagrams, two flow diagrams). |
| Local PDF visual inspection | PASS | All 45 pages rendered and reviewed; exact SHA-256 is recorded in `pdf_visual_review.json`. |
| Metadata and deployment self-test | PASS | CFF, Zenodo JSON, JSON-LD, bibliography and self-test. |
| GitHub build / exact-SHA verification | PENDING | Dedicated workflows must execute and be reconciled against actual SHAs. |
| GitHub branch-protection / ruleset audit | BLOCKED | API returned 403; required controls could not be inspected. |
| Semantic tag | NOT_RUN | Tagging is deferred until release verification is complete. |

**Current release state: PENDING_REMOTE_VERIFICATION.** A successful local build is not evidence of successful remote publication or empirical biological validation.
- Remote exact-commit archive verification: **PASS** (`cb805b455863577743f8d7d2a0ebfba1f21e4a32`).
- Remote PDF page-render review: **PENDING** (contact sheets generated from exact committed PDF).
