# Heteroplasmy Phase Control: Can the Timing of Mitochondrial Quality Control Matter?

*Devanik Debnath · Theoretical research note · 10 October 2026*

## A threshold is a biological measurement, not a universal constant

Mitochondrial DNA (mtDNA) heteroplasmy describes the fraction of mitochondrial genomes in a cell that carry a particular variant. Heteroplasmy can change as genomes replicate and are degraded, as mitochondrial material is partitioned, and as selection favors some genomes or organelles over others. For many pathogenic variants, the amount of mutant mtDNA matters: a biochemical phenotype may remain buffered until the mutant fraction crosses a threshold. But there is no single threshold that applies to every mutation and every tissue. The 2024 systematic review by Smith and colleagues documents the diversity of biochemical thresholds across mitochondrial genetic variants ([Genome Research, 2024](https://doi.org/10.1101/gr.278200.123)).

This creates a control question. Suppose an intervention can shift the relative abundance of mutant and wild-type mtDNA. Does its timing matter, even when total exposure is held fixed? Could a short, early pulse reduce the time a cell spends above a functional threshold compared with a constant schedule?

The interesting part is not to assume that pulsing works. It is to state exactly when a scheduling advantage follows from a model, identify which biological assumptions the result needs, and define observations that would reject it.

## Why the actuator is the hard part

Mitophagy is one pathway involved in mitochondrial quality control. Yet clearing damaged organelles is not the same thing as selectively eliminating every harmful mtDNA variant. Potential-dependent quality control relies on properties such as mitochondrial membrane potential, and genotype does not map cleanly onto potential across all mutations. Reviews of intracellular mtDNA quality control emphasize both supporting evidence and important limitations ([Seabright and colleagues, 2020](https://doi.org/10.1098/rstb.2019.0196)). The recent synthesis by Ryall, Chinnery and van den Ameele likewise treats heteroplasmy changes as the joint outcome of stochastic processes and context-dependent selection ([Annual Review of Genomics and Human Genetics, 2026](https://doi.org/10.1146/annurev-genom-120324-032239)).

That distinction defines the central experimental gate. Raising bulk mitophagy flux does not, by itself, establish that mutant genomes are being removed faster than wild-type genomes. An intervention might clear mitochondria indiscriminately, harm functional mitochondrial mass, or even favor a variant whose bioenergetic phenotype does not produce lower membrane potential. In the mathematics below, all of that biology is compressed into one effective selectivity parameter. This is a simplifying assumption to measure, not a mechanism to presume.

## A minimal model makes the assumptions visible

Let \(x(t)\) be the mutant fraction in one cell, \(s\) the net relative replication advantage of the mutant in the absence of control, and \(u(t)\) the intervention intensity. Let \(\Delta\) measure how strongly the intervention shifts relative selection against the mutant. The baseline equation is

\[
\dot{x}=x(1-x)(s-\Delta u),\qquad 0\leq u(t)\leq U.
\]

This is a reduced selection model. It is not a detailed model of mitochondrial fission, fusion, nucleoid organization, autophagosome recruitment, or mtDNA replication. In particular, it is not bistable: the sign of the drift is controlled by \(s-\Delta u\), and the equation does not generate a double-well landscape. That matters because a Kramers escape-time formula requires a properly specified metastable barrier. It cannot be inferred merely by adding noise to this equation.

The model becomes especially transparent after transforming heteroplasmy to log-odds, \(z=\log(x/(1-x))\). Integrating the equation gives

\[
z(t)=z(0)+st-\Delta\int_0^t u(\tau)\,d\tau.
\]

The integral is cumulative dose. With a fixed total dose, every schedule reaches the same final deterministic heteroplasmy, because the final log-odds depend on total dose rather than its order. Schedules can nevertheless differ at intermediate times because they build cumulative dose at different speeds.

## The theorem: under narrow assumptions, front-loading wins

Fix a total dose \(B\) over a horizon \(T\), with \(0\leq u(t)\leq U\). Consider the most front-loaded feasible policy: use intensity \(U\) from time zero until dose \(B\) has been delivered, then use zero. Its cumulative dose at time \(t\) is \(A_F(t)=\min(Ut,B)\). No other admissible schedule can have delivered more dose at any time \(t\), because it is constrained by both the maximum intensity and the same total budget.

When \(\Delta>0\) is constant, the log-odds equation implies that this policy has heteroplasmy no higher than any competing schedule at every intermediate time. If the loss function increases with heteroplasmy—for example, a squared penalty for exceeding a calibrated threshold—then the front-loaded policy also minimizes the integrated loss. Yet its final heteroplasmy is the same as that of any other equal-dose schedule.

This is the central formal result. It is a cumulative-dose dominance theorem, not proof that pulsed mitophagy is a good treatment. It also says something less exciting but more honest than a generic “bang-bang” assertion: given these simplified dynamics and a monotone loss, the optimal timing is determined by the assumptions. The result does not need a claimed singular arc, a Kramers barrier, or an unmeasured dynamical phase transition.

The limits are just as important. If \(\Delta=0\), timing has no effect. If \(\Delta<0\), the ordering reverses. A pharmacological delay, changing selectivity, compensatory mitochondrial biogenesis, toxicity, depleted wild-type mtDNA, or a state-dependent recovery cost can make the front-loaded policy inferior. Those effects must be added explicitly, not hidden under the word “realistic.”

## Where stochasticity enters

Real cells are heterogeneous. Genetic drift, copy-number differences, replication dynamics, cell state, and measurement noise can all affect observed heteroplasmy. A first approximation is a Wright–Fisher-type diffusion:

\[
dX_t=X_t(1-X_t)(s-\Delta u(t))dt+\sqrt{\frac{X_t(1-X_t)}{N_e}}\,dW_t.
\]

Here \(N_e\) is an effective population-size parameter and \(W_t\) is Brownian motion. This is a diffusion approximation, not a literal molecular account. It must be fitted or stress-tested against lineage-resolved or single-cell data, and boundary behavior must be specified.

The key endpoint is the first time a trajectory crosses a measured threshold, \(\tau^*=\inf\{t:X_t\geq x^*\}\). Useful population outcomes include the fraction of cells crossing by a fixed horizon, the distribution of first-passage times, integrated time-above-threshold burden, and cross-cell variance. The idea that rising variance could warn of future functional decline is a hypothesis to test in particular cell types—not a universal law. Variance can increase through neutral drift, sampling depth, or changes in cell composition, so it must predict future threshold burden beyond the mean, copy number, tissue identity, sequencing quality, and batch effects.

The accompanying verifier runs a deterministic equal-dose comparison and synthetic stochastic paths using a fixed random seed. These calculations verify implementation and illustrate consequences of the assumed model. They are not calibrated against biological data, do not identify a real drug’s selectivity, and do not support a lifespan claim.

## A falsifiable experimental sequence

The first experiment should not start with mice or with a headline claim about aging. Start with heteroplasmic cybrid lines or another well-characterized cell model and ask whether the candidate intervention produces positive mutant-versus-wild-type selection in the target variant and cell type. Measure heteroplasmy over time alongside mtDNA copy number, membrane potential, mitophagy flux, mitochondrial mass, cell survival, and oxidative-phosphorylation function. A change in total flux alone is not enough.

Next, calibrate the biochemical threshold before comparing schedules. Randomize replicate cultures to vehicle, continuous exposure, and a front-loaded or pulsed schedule with equal cumulative input. Pre-register time above threshold and integrated threshold burden as primary outcomes, along with one functional endpoint. Control handling and washout, track final heteroplasmy, and include independent clones or culture units. Estimate selectivity and actuator kinetics in a training set and evaluate predictions in held-out clones. A pulse that does not improve the preregistered threshold endpoint at matched dose falsifies the baseline scheduling prediction for that context, provided the experiment can exclude a minimum relevant effect. A low-powered null result is inconclusive, not a victory for either side.

The kill criterion comes even earlier: if the candidate actuator’s measured selectivity is zero, negative, or too uncertain to distinguish from zero, the proposed control term has no demonstrated biological counterpart. Optimization cannot repair a missing actuator.

Only after target engagement, efficacy and safety are demonstrated in cells should an in vivo program be considered. The PolgA D257A mouse provides a well-known model of elevated mtDNA mutagenesis and premature-aging phenotypes ([Trifunovic et al., Nature, 2004](https://doi.org/10.1038/nature02517)), but a broad mutator model is not automatically a model of one mutant heteroplasmy or selective clearance. It should be a secondary context after specifying the variant, mechanism, tissue, endpoint, and power analysis. No dosing regimen or animal experiment is claimed here.

## How this fits with existing work

Mitochondrial base editing is a distinct approach: DdCBE demonstrated targeted C-to-T conversions in human mtDNA, enabling controlled mutation models and potential genetic interventions ([Mok et al., Nature, 2020](https://doi.org/10.1038/s41586-020-2477-4)). Adaptive therapy in oncology provides an analogy for considering treatment timing in evolving populations, but tumor competition and mitochondrial genome selection are different biological problems ([Gatenby et al., Cancer Research, 2009](https://doi.org/10.1158/0008-5472.CAN-08-3658)). Neither body of work establishes that pulsed mitophagy will work.

A preliminary search using combinations of “heteroplasmy threshold control,” “mitophagy optimal control,” “mitochondrial heteroplasmy dynamics,” and “adaptive therapy mitochondria,” including PubMed- and arXiv-oriented queries, found related work on mtDNA drift, selection, threshold effects, and editing. It did not establish priority for this exact formulation. That is not proof of novelty: a publication-grade priority claim needs an updated systematic literature search and specialist review.

## The point of the proposal

The useful claim is conditional and testable. In a simple model with constant positive mutant selectivity and fixed cumulative dose, front-loading control lowers intermediate heteroplasmy and reduces any monotone integrated burden, but leaves the deterministic endpoint unchanged. The next question is whether real mitochondrial interventions satisfy those assumptions once delay, toxicity, selection, and recovery are measured.

This preprint has not tested an intervention, used a biological dataset, established an aging mechanism, or shown that any schedule extends lifespan. Its value is a mathematical baseline and a set of explicit failure conditions. A measured absence of selectivity should end the proposed actuator pathway; a failed equal-dose comparison should reject the baseline schedule result in the tested context. Anything stronger must come from experiments, not rhetoric.
