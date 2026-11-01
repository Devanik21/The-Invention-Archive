# Release-check summary

This release is a conditional mathematical framework for antibody-cocktail coverage. It is not a universal vaccine result, an empirical HIV neutralization analysis, or an antivenom efficacy study.

Local checks and evidence are recorded in `release_check_matrix.json`. The PDF build, source identity, numerical tests, metadata, visual review, and artifact manifest are bound to the SHA-256 values listed in `sha256_manifest.json` and `pdf_visual_review.json`.

Remote GitHub checks are not PASS until the complete release is committed and independently fetched. A green run on an earlier commit does not validate a later tree. The final deployment report must identify the exact artifact commit, current `main` head, run IDs/URLs, and tag target.