# Release check matrix — final local state

This matrix records observed checks. It does not claim real-data validation or remote publication.

| Check | Status | Evidence |
|---|---|---|
| Literature audit and scoped novelty claim | **PASS** | `results/literature_audit.md`; no absolute priority claim. |
| Public multi-proxy dataset paired with fabric labels and modern target | **BLOCKED** | No SISAL rows or aligned monitoring data downloaded or analyzed; empirical claim remains blocked. |
| Synthetic model numerical verification | **PASS** | `results/verification_report.json`; fixed-seed run, 10 assertions. |
| Python syntax checks | **PASS** | `python -m py_compile verify_model.py deploy.py`. |
| Fail-closed deployment self-tests | **PASS** | `python deploy.py --self-test`; 3/3 negative fixtures rejected. |
| CFF/JSON syntax, citation keys, README links, source structure | **PASS** | `python deploy.py --validate-only`; 9 BibTeX keys, 8 cited keys, 33 README links. |
| Current TeX/Bib source compiled to PDF | **PASS** | `python deploy.py`; fresh 16-page build. |
| Full-page visual QA tied to final PDF hash | **PASS** | `results/pdf_visual_review.json`; pages 1–16 reviewed, SHA-256 recorded. |
| SHA-256 manifest covers release tree | **PASS (local)** | `results/sha256_manifest.txt`; remote hash equality still pending. |
| Latest remote branch head, workflows, collision, index inspected | **PASS** | GitHub connector audit 2026-10-18; branch protection read unavailable (403). |
| Core files, figures, results, and index remotely present and content-consistent | **NOT RUN** | No release files published yet. |
| Applicable workflows terminal for exact final SHA | **NOT RUN** | Requires deployment and exact-SHA Actions checks. |
| Release tag points at intended commit and verified remotely | **NOT RUN** | Tag not created. |
