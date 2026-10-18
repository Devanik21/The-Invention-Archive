# Release validation summary

- Research form: theoretical/information-theoretic framework plus a deterministic synthetic stress test; not a real-data paleoclimate reconstruction.
- Model result: channel-aware OLS improves synthetic RMSE only modestly relative to a multi-proxy OLS comparator; shifted-case nominal-90% interval coverage is 0.8471, so uncertainty calibration under the tested shift fails.
- Data gate: no public SISAL rows or aligned cave-monitoring data analyzed. Do not interpret the simulated metrics as empirical results.
- Visual gate: final PDF rebuilt from current sources; 16 pages reviewed after fixing the dependency graph and method-flow layout. Final PDF SHA-256 is recorded in `results/pdf_visual_review.json`.
- Local release gate: Python compilation, 10 deterministic model assertions, 3/3 fail-closed deployment self-tests, metadata/link checks, and a fresh 16-page PDF build passed. Remote deployment, tag, and remote Actions closure remain pending until verified.
