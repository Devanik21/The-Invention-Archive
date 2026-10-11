# Rate-Distortion Immunology: What If Immune Aging Is a Coverage Problem?

*Technical essay draft — not yet published*

## Diversity is not the same as protection

Immune aging is often told as a story about numbers: the thymus involutes, fewer naive T cells enter the periphery, long-lived memory clones expand, and the distribution of T-cell receptor (TCR) frequencies becomes less even. These changes matter. But a repertoire can have many unique receptors and still miss a particular pathogen; it can also contain a strikingly dominant memory clone that is essential for controlling a persistent infection. Richness, Shannon entropy and clonality summarize the distribution of clone abundance. They do not directly measure the probability of recognizing a future pathogen epitope.

That distinction motivates a mathematical question: if the immune system can maintain only a limited set of recognition types within a defined memory budget, how much expected protection is being lost relative to the best feasible selection from a specified candidate pool?

Rate-distortion theory offers a language for asking this carefully. In information theory, a representation is judged by the distortion it incurs under a defined distribution of inputs. Applied to immunology, the “input” is a declared panel of peptide–HLA targets, and the distortion is the modeled probability that all selected receptors fail to respond to a target drawn from a specified prior. The important words are *declared*, *modeled* and *specified*. The immune system is not literally a codec, and its pathogen universe is neither fixed nor known.

## A coverage-aware objective

Let \(\mathcal{E}\) denote a finite set of peptide–HLA targets. The probability that target \(e\) is missed by a selected repertoire subset \(S\) can be approximated as

$$
q_e(S)=\prod_{i\in S}(1-r_{ie}),
$$

where \(r_{ie}\) is the calibrated probability that receptor \(i\) responds to target \(e\) under a declared assay. This product formula assumes conditional independence across clones; correlated receptor responses require a joint miss-probability model instead.

Let \(\pi_e\) be the prior probability assigned to target \(e\) and \(\ell_e\) a consequence weight. Expected modeled distortion is

$$
D_\pi(S)=\sum_{e\in\mathcal{E}}\pi_e\ell_e q_e(S).
$$

The weights change the meaning of the score. With all \(\ell_e=1\), it is an estimated probability of missing a random target from the panel. With severity weights, it becomes expected weighted loss—not a probability. Those should never be conflated.

Now choose a candidate universe \(\mathcal{C}\) and a codebook budget \(K\), measured in distinct receptor types in the computational benchmark. The best achievable distortion is

$$
D^*_\pi(K;\mathcal{C})=\min_{S\subseteq\mathcal{C},\ |S|\leq K}D_\pi(S).
$$

The proposed coverage debt compares an observed baseline panel \(R_K\) with this optimum:

$$
\mathrm{CD}_\pi=D_\pi(R_K)-D^*_\pi(K;\mathcal{C}).
$$

This quantity is nonnegative when the observed baseline is feasible in the candidate universe. It is also conditional on everything used to define the universe, target prior, HLA context, recognition score, and budget. That conditionality is a feature rather than an inconvenience: it makes assumptions testable. But it prevents a single score from being called an intrinsic measure of biological age or immune quality.

The codebook budget \(K\) is not the same as total T-cell number, sequencing read depth, or a proven hard limit on memory-cell occupancy. Sequence count and cell count answer different questions. A future model could include an experimentally grounded niche-occupancy cost, but the biological interpretation would need to be developed and measured separately.

## Why ordinary diversity is a strong baseline

This idea must not be built on a caricature of existing immunology. Repertoire diversity, selection, maximum-entropy models and machine-learning prediction of TCR specificity have substantial literatures. One analysis of approximately 30,000 adult TCR-beta repertoires reported that repertoire size and the frequency contributed by highly expanded clones together explained around 96% of observed diversity variance in that cohort. That means a new score must prove it adds information beyond repertoire size and large-clone abundance, not merely rename them.

Nor is a large clone automatically waste. Human studies of cytomegalovirus (CMV) have reported pronounced expansion of antigen-experienced clones, yet one study found that the underlying CD8 repertoire diversity could be preserved. Persistent memory can be protective. If a coverage objective labels every expansion as an inefficiency, the objective—not biology—has failed.

The right question is therefore not “How do we eliminate large clones?” It is “For a defined panel of targets, does the existing repertoire contain measurable, avoidable gaps in recognition coverage, and can that measurement improve prediction of a real outcome?”

## A tractable mathematical result

Under the independent-miss model, define covered utility as one minus distortion. The gain from adding clone \(j\) to set \(S\) is

$$
\Delta_j(S)=\sum_e\pi_e\ell_e r_{je}\prod_{i\in S}(1-r_{ie}).
$$

Every factor is nonnegative. Adding more clones can only shrink the remaining miss probability, so the marginal value of a new clone cannot increase as the selected set grows. This is the defining diminishing-returns property of a monotone submodular objective. For a fixed-cardinality unconstrained selection problem, a standard greedy algorithm consequently achieves at least \(1-1/e\) of optimal covered utility.

This result is useful because it gives a transparent baseline optimizer that can be checked independently. It is also limited. The guarantee does not automatically extend to correlated recognition, distributionally robust optimization, or arbitrary clinical protection constraints. The mathematics is a property of the declared objective—not evidence that the natural immune system evolved to optimize it.

## The pathogen prior is the weak point

The prior \(\pi\) is not a neutral technical detail. Known epitope databases overrepresent pathogens, HLA alleles, demographic groups, and assays that researchers have studied. An emerging pathogen or locally common infection could be assigned too little weight. A vaccine-target prior and an epidemiological prior can produce different optimal subsets.

For that reason, the paper proposes a family of plausible priors rather than one definitive target distribution. A robust version minimizes the worst expected loss over that family. At minimum, results should be reported under a surveillance-weighted prior, a uniform prior over a predeclared target panel, vaccine-specific weighting where relevant, and stress tests that upweight poorly represented target families. If a person's ranking changes dramatically across these alternatives, the honest result is “prior-sensitive,” not a stable biological score.

Recognition predictions carry another uncertainty layer. Peptide binding, TCR binding, cell activation, and clinical protection are not equivalent. A protein-language model may learn sequence similarities or database artifacts instead of response probabilities that generalize to a new donor. Its scores must be calibrated on held-out studies and must account for HLA restrictions. A sequence generated by a VDJ model may be biologically plausible yet fail thymic selection, surface expression, antigen binding, or activation.

AI's role is therefore to estimate a recognition matrix with uncertainty—not to divine which cells are disposable. Models should split validation data by donor and epitope, deduplicate related pairs, and test on entire held-out studies or antigen families. Report calibration and error metrics, not just AUROC. When support is inadequate, the correct output is uncertainty or abstention.

## A realistic computational first step

The first study can be retrospective and cheap compared with a biological intervention. Public infrastructure such as the AIRR Data Commons contains repertoires from many studies, and IEDB supplies experimentally described immune epitopes and receptor-specificity records. A 2026 public dataset also contains CD4/CD8, TRA/TRB clonotype-level repertoires from healthy and frail older adults, with CMV status and epigenetic-age metadata. That dataset can support a limited aging/CMV analysis; it should not be described as a vaccine-response cohort unless such an endpoint is actually present.

A proper benchmark has to lock the outcome and cohort before computing debt. It should harmonize sequencing depth and cell subset, define a target panel with HLA restrictions, train receptor–epitope prediction without donor or epitope leakage, estimate the coverage gap, and compare prediction against strong baselines: age, sex, CMV status, HLA, repertoire size, Hill diversity, top-clone mass, clonality, cell composition and technical covariates.

For vaccine response, the validation cohort must contain measured outcomes such as antibody titre, antigen-specific T-cell response, or a prespecified clinical endpoint. The aging/CMV dataset alone cannot establish that connection. Use participant-level splits and external study validation; report discrimination and calibration with confidence intervals.

The primary falsification criterion is simple: if debt does not improve out-of-cohort prediction over ordinary diversity and clonality metrics, the rate-distortion framing has failed as a useful predictive advance. A second failure condition is instability under reasonable pathogen priors. A third is an apparent signal that disappears after correcting recognition-model calibration or sequencing batch effects.

For this release, I could verify the public record for the aging/CMV dataset and the broader AIRR/IEDB resources, but I could not download the raw human repertoire files into the available environment. The numerical results in the paper are therefore synthetic. They validate the equations and a small optimization example, not the biology.

## Protection must be a hard constraint

The most dangerous leap would be from computational redundancy to clone depletion. A clone that covers no target in the selected panel may recognize an unmeasured pathogen variant or have a role in a different tissue. Memory clones may be durable protective assets. Sequence alone cannot rule out auto-reactivity, tissue residency, exhaustion, or unmeasured functions.

Any optimization framework should include protected epitopes and mandatory retained clones, and it should declare the result infeasible rather than silently relax a safety threshold. Even that is only a computational gate. No clinical use would be justified without independent functional tests, self-reactivity screening, proof of selective removal, and a safe, realistic mechanism for replenishing the repertoire. The current paper does not propose a human depletion protocol.

## What would make this matter?

If validated, coverage debt could become an interpretable research metric for comparing repertoire quality, prioritizing experiments, or identifying where future diversity renewal might be most useful. That could benefit older adults indirectly by helping scientists understand why two repertoires with similar richness differ in response to a new vaccine or infection. The path to that benefit is long: robust prediction first, causal mechanism second, safe intervention much later if at all.

The more important outcome of this proposal may be methodological discipline. It gives “compression” a precise target distribution, distortion function, candidate universe, capacity measure, uncertainty interval and failure criteria. It also leaves room for a negative result. If existing metrics predict outcomes just as well, the community should keep using the simpler metrics.

The point is not to defragment immunity by deleting memory. It is to test whether a coverage-aware description sees something real that abundance and diversity statistics miss—and to stop if it does not.

## Selected references

- Zahid et al. (2026), “A fundamental relationship between TCR diversity, repertoire size and systemic clonal expansion,” *Frontiers in Immunology*. https://doi.org/10.3389/fimmu.2025.1707727
- Weber et al. (2024), “T-cell receptor binding prediction: A machine learning revolution,” *ImmunoInformatics*. https://doi.org/10.1016/j.immuno.2024.100040
- Lindau et al. (2014), “Cytomegalovirus exposure in the elderly does not reduce CD8 T cell repertoire diversity,” *Journal of Immunology*. https://doi.org/10.4049/jimmunol.1800217
- Breëns et al. (2026), aging/CMV TCR repertoire dataset. https://doi.org/10.5281/zenodo.20545037

**Author:** Devanik Debnath (Devanik21), Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India. Research archive: https://github.com/Devanik21/The-Invention-Archive. This essay is a publication draft; it has not been posted to Medium.
