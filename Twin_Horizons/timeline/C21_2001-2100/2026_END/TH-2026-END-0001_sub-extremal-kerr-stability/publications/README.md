# Publications: TH-2026-END-0001 (Sub-Extremal Kerr Stability)

This folder holds submission-ready papers from the physical-space Carter multiplier program on sub-extremal Kerr exteriors. The source of truth is the closing report dated 27 November 2026 (see the parent README). Nothing in this folder may claim more than that report establishes.

---

## 1. What is ready and what is not

The earlier eight-document series (Theorems 1 to 4 and Papers 5 to 8) must **not** be submitted as they stand. The closing report withdraws or downgrades several of their central claims, including the spin-2 transport bound, the redshift closure, the collapse theorem, and the novelty audit. Any manuscript drawn from those documents must be rewritten to carry the report's status labels.

The main obstacle to submission is not the mathematics alone. The report did not perform a literature search, so every claim about prior work is still [C]. A referee will expect those claims to be checked first.

---

## 2. Candidate papers

Each candidate below is a separate paper. They can be submitted independently.

### Paper A: Obstructions to physical-space integrated local energy decay on sub-extremal Kerr (primary candidate)

**Thesis.** A physical-space Carter multiplier of the perfect-square type does not, by itself, yield uniform integrated local energy decay for all sub-extremal spins. The paper identifies the precise obstructions and what would be needed to remove them.

**Contents, drawn from the report:**
- Trapping-superradiance gap (Thm 3.1, [V-S][V-H]) as a kinematic lemma.
- Near-extremal scalings of the photon sphere and Lyapunov exponent (Prop 4.1).
- Horizon compatibility obstruction within the barrier class, valid for sigma < 1/4 (Prop 6.8, Cor 6.9).
- Gap 2A numerical audit with the referee table, convergence study, cutoff sensitivity, and the 54-configuration sweep (Section 9, Tables 3, 4, 5, 9).
- Redshift patch test showing collar dependence (Section 10, Table 6), reported as a negative result.
- Spin-2 transport sign error and the corrected identity (6), with the horizon-trace open problem O1 (Section 11).
- The open-problem list O1 to O6 with the missing ingredient for each.

**Why it is a publishable contribution.** Negative results and precise obstructions are useful to the field when stated with exact status labels and reproducible numerics. The report already has this form.

**Limits to state in the paper.** The numerics test one multiplier family at two extreme spins in full and one parameter setting elsewhere. The Gap 2A window [10^-3, 0.15] is a single-setting observation, not a theorem (E3).

### Paper B: Exact algebraic identities for the spin-2 Teukolsky to Regge-Wheeler reduction (secondary, short)

**Thesis.** The Level-0, Level-1, and Level-2 identities of the physical-space hierarchy hold with zero remainder, with the potential and coupling coefficients stated in closed form.

**Scope limit.** The paper may claim only the algebra, which is [V-S]. It must state that the master equation, the operator K_{-2}, and the physical-space identities (3) and (4) of the drafts were taken as inputs and were not independently verified (Prop 11.1). It must not repeat the transport bound, which is [F] (E1).

**Pre-submission requirement.** Independently re-verify the identities from the Teukolsky master equation in standard form, outside the drafts' conventions, before submitting. The report's check takes the master equation as given.

### Paper C: Closing record and errata (supplementary, not a standalone submission)

The full closing report, with its errata table E1 to E13, status table, and reproducibility appendix, can be deposited as a technical report or an arXiv preprint alongside Paper A. It is the complete audit trail.

---

## 3. Claims that must not appear in any submission

These statements from the earlier series are withdrawn. A manuscript repeating any of them will be rejected or retracted.

- The spin-2 transport bounds (drafts' (30)-(31)). Sign error (E1, [F]).
- A complete physical-space proof of ILED for all sub-extremal spins.
- Gap 2A closes on sigma in [10^-3, 0.15] as a general claim (E3, [F]).
- The redshift patch closes the moderate-spin failure (E11, [F]).
- The nonlinear bootstrap threshold collapses as sigma^(p+q) as a theorem (E10, [F]).
- The commuted energy with K^(1/2) (E8, [F]).
- Any novelty claim relative to the literature (Section 17, withdrawn in full).
- The Affective Cusp statement that a faster build-up cannot produce an earlier outburst (Section 18, [F]). This is separate from the Kerr program.
- Strict ellipticity with constants constructed for all sigma (E9, [C]).
- Any statement that the physical-space multiplier "encodes the structural reason" linear instabilities are absent (Remark 3.5, [C]).

---

## 4. Pre-submission checklist

No paper leaves this folder until every item is done.

1. **Literature search.** Verify every cited work in Table 8 of the report against its publisher page or preprint server, including the 2026 arXiv identifiers. Locate and read the published physical-space and ILED literature on sub-extremal Kerr that the report did not survey. Then decide whether Paper A's obstructions are already known. Record the outcome in the paper's related-work section.
2. **Reference-check the [C] items** that the paper relies on. Either verify them or remove them. For Paper A, that includes the quasinormal-mode identification of lambda_L (Remark 4.3).
3. **Fix the two items from the parent README's review section.** Correct the sigma form of the axisymmetric ceiling to sigma >= (sqrt(5) - 1)/2, and check the equivalence in Theorem 5.2(a) against Lemma 5.1, which gives sigma^2 + sigma - 1 >= 0.
4. **Rerun every script** listed in Section 14 of the parent README on a clean environment. Record the environment (Python, NumPy, SciPy, SymPy 1.14, mpmath 1.3) and the exact outputs. Every number in the paper must trace to a stored output.
5. **Archive the code and data.** Put the scripts, the outputs, and a README for them in a zip under `snapshot/` in the parent entry. Cite the archive ID in the paper.
6. **Check the status label on every claim** in the manuscript. Any statement without a label, or with a label stronger than the check performed, must be fixed.
7. **Confirm the venue's authorship policy.** The report names Xylia (AI research system) as an author. Journals and preprint servers have differing rules for AI systems, and some do not permit them to be listed as authors. Check the venue's policy and disclose the AI system's role as the policy requires.
8. **Peer review of the mathematics.** Have at least one researcher in mathematical general relativity or PDE analysis check the derivations marked [V-H] before submission.

---

## 5. Suggested submission order

1. Complete the checklist for Paper A.
2. Post Paper A as a preprint, with the closing report as a supplementary technical report (Paper C).
3. Submit Paper A to a journal that publishes negative results and careful analyses in mathematical general relativity.
4. Once Paper B's independent re-verification is done, submit it separately, so that it is not read as part of the withdrawn program.

---

## 6. What this folder contains

- This README.
- Final manuscripts, one per paper, named by candidate letter and archive ID (for example `PaperA_TH-2026-END-0001.pdf`). Only final versions are added.
- A `snapshot/` zip with code and outputs, once the checklist item 5 is done. It lives in the parent entry, not here.

Drafts and working versions stay in the parent entry.
