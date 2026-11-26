# TH-2026-END-0001: Sub-Extremal Kerr Stability

Date: 2026-11-26. Authors: Devanik Debnath, Xylia (AI Scientist), Gemini 4 Neon High.

The eight documents of the sub-extremal Kerr series (0 <= |a| < M), with the status each document claims for itself. None of them is peer-reviewed.

| Document | Subject | Claimed status | Main result |
| --- | --- | --- | --- |
| Theorem 1 | Trapping and superradiance separation | Proved | (omega - m omega_+)/omega = [s^2 + ds + Md(1 - d/(2s+d))]/(2M r_+) > 0 |
| Theorem 2 | High-spin triple collision law | Proved (sigma -> 0) | kappa ~ sigma/(2M); lambda_L/kappa -> 1 |
| Theorem 3 | Exact spin ceilings of canonical Carter multiplier | Proved (sharp) | \|a\|/M <= 1/2 (3D); \|a\|/M <= 0.786 (axisymmetric) |
| Theorem 4 | Global perfect-square Carter-Lagrangian positivity | Proved, all \|a\| < M | Riccati barrier 0 <= rho <= 1/(r - r_-) |
| Paper 5 | Strict ellipticity, axisymmetric energy, scalar ILED | Partial | Gaps 2A (low-frequency Blot) and 2B (m != 0 superradiant flux) open |
| Paper 6 | Spin-2 Teukolsky to Regge-Wheeler hierarchy | Proved (self-reported algebra) | Exact GRW equation; sigma-uniform transport bounds |
| Paper 7 | Quasilinear Carter commutator; trapping gauge | Partial | Core trapped-set derivative loss open; stability-radius collapse is conditional |
| Paper 8 | Master synthesis and verification protocol | Synthesis | Scripts planned, not yet run |

## Issues found on review

1. Theorem 3, eq. (7): the sigma form of the axisymmetric ceiling is wrong. |a|/M <= sqrt(phi^-1) is equivalent to sigma >= phi^-1 = (sqrt5 - 1)/2 (about 0.618), not sigma >= sqrt((sqrt5 - 1)/2). The bound |a|/M <= 0.786 is correct.
2. Theorem 4, Step 3: P(3M) should be 27M^3 - M a^2, not 27M^3 - 3M a^2. The inequality still holds.
3. Cross-references are ambiguous. Several documents cite "Theorem 1" or "Theorem 4" for different results. Use paper-qualified labels.
4. Paper 7, Theorem 1: the lower bounds C1 and C2 come from heuristic arguments (Aretakis-type growth and quasinormal damping), not rigorous estimates. The collapse result is conditional on them.
5. Paper 7, Proposition 1: the commutator identity is schematic ("up to lower-order terms"), not a verified identity.
6. Paper 8: the SymPy and SciPy verification stages have not been run, so the algebra in Papers 1, 2, 4 and 6 is not yet machine-checked.

## Files not in this folder

The eight PDFs are not committed. The archive allows only .md, .json, .txt and .zip outside snapshot/, and this connector can only write text. To archive them, zip them and upload the zip to snapshot/ by hand, or send the .tex sources and they can be added as text.
