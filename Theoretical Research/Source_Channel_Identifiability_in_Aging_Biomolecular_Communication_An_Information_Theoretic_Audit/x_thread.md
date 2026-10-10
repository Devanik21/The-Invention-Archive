# X thread draft — Source-Channel Identifiability in Aging Biomolecular Communication: An Information-Theoretic Audit

**1/12** Aging-related biomolecular communication is increasingly studied with information theory. But when mutual information declines, have we observed a noisier channel—or a different distribution of inputs? The distinction matters for biological interpretation.

**2/12** Let X be a regulatory input and Y its measured response. The source p(x) tells us how often input states occur; the conditional law W(y|x) tells us the response distribution given an input.

**3/12** Mutual information depends on both objects: Iₚ(W) = Σₓᵧ p(x)W(y|x) log₂[W(y|x)/pᵧ(y)]. A change in I alone cannot identify which object changed.

**4/12** A recent 2026 study reports that input-distribution mismatch—not channel corruption—drove declining MI in its analyzed aging transcriptional-regulation system. This is a system-specific result, not a universal law. https://doi.org/10.1016/j.xcrp.2026.103516

**5/12** Counterexample: a binary symmetric channel flips each bit with probability ε = 0.10. Keep ε fixed, but change P(X=1) from 0.50 to 0.05. Mutual information falls even though the conditional channel is unchanged.

**6/12** The exact expression is Iₚ(W) = H₂(ε + p(1−2ε)) − H₂(ε). For p = 0.50, I ≈ 0.531 bits/use; for p = 0.05, it is ≈ 0.115 bits/use.

**7/12** The lesson is not that gene regulation is a binary channel. It is that “MI fell, therefore molecular communication degraded” is not an identified inference without assumptions or additional measurements.

**8/12** A practical audit estimates the conditional response law in each cohort, then evaluates it under a common reference input distribution q(x), restricted to support observed in both cohorts.

**9/12** This produces a standardized contrast I_q(W_aged) − I_q(W_young). It is reference-dependent and observational conditional laws are not automatically causal—but it isolates one important source of ambiguity.

**10/12** The accompanying preprint gives an exact telescoping decomposition of the natural MI contrast and a deterministic NumPy script. The model is synthetic: no biological data, molecular payload design, or treatment claims.

**11/12** An empirical audit should also report common-support counts, cell-composition effects, sequencing depth/dropout, batch sensitivity, estimator uncertainty, and prespecified alternative reference distributions.

**12/12** Contribution: a compact source-channel identifiability audit, not a new theory of aging. Paper, PDF, code, and references: https://github.com/Devanik21/The-Invention-Archive/tree/main/Theoretical%20Research/Source_Channel_Identifiability_in_Aging_Biomolecular_Communication_An_Information_Theoretic_Audit
