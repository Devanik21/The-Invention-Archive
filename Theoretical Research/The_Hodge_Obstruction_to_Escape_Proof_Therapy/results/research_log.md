# Research log — 2026-10-25

## Scope and selection
Working question: can a Hodge-curl component of drug-specific collateral-effect fields obstruct the existence of an adaptive safety policy for antimicrobial-resistance landscapes?

Selected direction after considering alternatives around antimicrobial sequencing, resistance trade-offs, robust control, mutation-graph topology, stochastic escape, and learned landscape completion. The distinctive angle is a rigorous test of the proposed Hodge obstruction itself followed by an exact finite-state viability game. It is deliberately a corrective no-go result if the original conjecture does not survive, rather than a forced novelty narrative.

## Literature search and coverage
Targeted queries covered sequential antibiotic therapy, collateral sensitivity, resistance landscapes, reinforcement learning for treatment sequencing, combinatorial Hodge decomposition, and discrete topology. Primary sources are recorded in `literature_audit.csv`. Search coverage is targeted and not exhaustive; adjacent work and newly published results may still have been missed. No dataset was downloaded; all numerical examples in this release are generated synthetically.

## Mathematical decision
Define a drug-specific edge field as the difference of a scalar fitness potential between endpoint genotypes. The field is exact, so its circulation around any closed loop telescopes to zero. With the standard cubical boundary-of-boundary identity, a nonzero curl cannot be a structural property of a correctly measured static potential. Thus the original conjecture—nonzero curl deciding whether escape-proof schedules exist and lower-bounding the drug count—is rejected under those definitions.

Nonzero empirical circulation may be useful diagnostically, but it must be interpreted as possible measurement noise, incompatible measurements, environment dependence, history dependence, or a non-potential edge statistic. Therapy safety is instead solved from an explicitly stated transition relation, target failure set, and observation model.

## Numerical model and provenance
The benchmark is a synthetic 4-locus hypercube (16 genotypes) with 3 disclosed scalar potentials. The code fixes seed 20261025, generates edge fields and seeded noise, builds the robust predecessor operator, computes a greatest fixed point and a finite-horizon probabilistic risk calculation, and writes all source data and vector graphics. It is not based on the TEM beta-lactamase dataset or any wet-lab measurements.

## Validation results, from actual code execution
`verify_model.py` passed 12 assertions. Max absolute exact square circulation: `1.1102230246251565e-16`; `||B2 B0||_F`: `0`; declared synthetic failure states: 6/16; robust viability kernel: states `[0, 2, 4, 6]` (4/16); fixed-point candidate sizes `[10, 6, 4, 4]`. All claims remain conditional on the artificial fitness coefficients, threshold, and successor relation.

## Open risks and falsifiers
The static-potential result is tautological if edge effects are defined by endpoint differences; the empirical challenge is how the edge field is actually measured. The robust kernel fails as a real guarantee if transition support omits a biologically plausible mutation or context. Any Hodge-based benefit must beat simpler baselines on held-out, genuinely measured landscapes with calibration and uncertainty checks. No claims of clinical utility or escape-proof real therapy are made.
