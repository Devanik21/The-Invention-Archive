# Functorial Epigenetic Realignment: Can Aging Be Treated as a Geometric Control Problem?

*Technical research essay by Devanik Debnath (Devanik21)*

Aging is usually introduced as a collection of mechanisms: DNA damage, epigenetic drift, mitochondrial dysfunction, senescence, inflammation, loss of proteostasis, altered nutrient sensing, and many others. That description is necessary because aging is genuinely multicausal. It also leaves an interesting mathematical question open.

What if part of the useful information in an aging cell is not contained in individual molecular values, but in the geometry of the relationships among them?

That is the starting point of Functorial Epigenetic Realignment, or FER.

The proposal is deliberately narrower than the statement that aging is “really topology.” Current evidence does not justify replacing molecular biology with a geometric explanation. FER instead asks whether a geometric layer can be measured well enough to become an additional state variable, and whether that state variable can guide controlled perturbation.

## A cell as a distribution, not a point

Consider a cell measured through transcriptomics, chromatin accessibility, methylation, histone modifications, proteomics, nuclear morphology, and three-dimensional chromatin features. The resulting vector can contain thousands of coordinates. A population of cells is therefore better represented as a probability distribution than as a single average point.

One natural local geometry is supplied by information geometry. On a probability simplex, the Fisher metric is

`gₚ(u,v) = Σᵢ uᵢvᵢ / pᵢ`

The equation matters because it gives state changes a non-Euclidean local structure. Moving a low-probability component can be expensive in Fisher geometry even when its raw Euclidean displacement looks modest.

This does not mean the Fisher metric is “the biological metric.” It is a candidate geometry whose usefulness must be tested empirically.

A second geometry becomes relevant when the question changes from “how far apart are these populations?” to “how could one population move toward the other?”

## The transport question

Optimal transport provides a principled language for comparing distributions. The Wasserstein-2 distance can be written as

`W₂²(pₐ,pᵧ) = min_{π∈Π(pₐ,pᵧ)} Σᵢⱼ cᵢⱼπᵢⱼ`

where `pₐ` is an aged reference, `pᵧ` is a young reference, and the coupling `π` specifies how mass is transported between states.

The important point is that this is a geometry-induced path, not a recipe for treatment. In one dimension, the optimal interpolation has an especially clean form in quantile space:

`Qₜ = (1−t)Q₀ + tQ₁`

This gives a constant-speed Wasserstein geodesic under the usual construction. In FER, that geodesic becomes a candidate target direction against which real biological trajectories can be compared.

The target is not “make the biomarker younger.” The target is “move the distribution toward a pre-defined reference state while preserving identity and avoiding damage.”

## Why topology enters

A high-dimensional biological state is not only a collection of values. It also contains relationships. Gene-regulatory networks have connectivity structure. Chromatin has loops, domains, compartments, and contacts. Cellular populations occupy nontrivial shapes in learned feature spaces.

Persistent homology gives a way to ask whether some of those structures persist across scales. A topological signature can be written as a collection of Betti curves across a filtration:

`τ = {β₀(Kε), β₁(Kε), …}`

The appeal is that persistent features can be summarized without selecting one arbitrary scale in advance. The danger is equally important: topology is representation-dependent. The metric, filtration, sampling density, preprocessing, and feature scaling all matter.

That means a topological signature only becomes scientifically meaningful if it survives held-out validation. FER therefore requires donor-level and batch-level validation and explicit null models.

There is already precedent for this direction. Topological data analysis has been proposed specifically as a way to study aging and to search for a “topological shape of ageing.” The novelty of FER is not the existence of TDA, but its placement inside a control architecture.

## The composite state objective

FER therefore uses a combined objective:

`D(z,zᵧ) = αW₂²(p_z,pᵧ) + βd_top²(τ_z,τᵧ) + γR_id(z,z₀) + λR_stress(z)`

The last two terms are not decorative.

`R_id` penalizes loss of lineage identity. A cell that looks younger because it has partially dedifferentiated is not an acceptable rejuvenation result.

`R_stress` penalizes nuclear deformation, DNA damage, pathological mechanotransduction, uncontrolled proliferation, or other experimentally defined safety violations.

This turns rejuvenation into a constrained optimization problem rather than a race toward the lowest possible age score.

## The role of category theory

The word “Functorial” can easily become ornamental. FER tries to avoid that.

Imagine a category whose objects are experimentally admissible biological states and whose morphisms are admissible transitions between them. Construct another category for physical controls, where objects are controllable actuator states and morphisms are calibrated control transformations.

The proposed translation is a functor `F` satisfying

`F(id_z) = id_F(z)`

and

`F(g ∘ f) = F(g) ∘ F(f)`

The practical interpretation is limited but useful. If two biological transitions compose in a defined way, the control representation should compose consistently too.

This does not make the AI magically “obey biology.” The actual constraints remain physical and experimental. Category theory supplies a structure for checking whether a controller has been specified consistently.

## The physical coupling problem

The next question is much harder.

How would a geometric state target be connected to something physical?

Mechanobiology provides a plausible route. Mechanical signals can propagate from the cytoskeleton through nuclear structures and influence chromatin organization. The LINC complex is one part of that mechanical coupling. The nucleus is not an inert container; its organization responds to force, deformation, and material state.

A coarse model is

`ẋ = f(z) + G(z)u + η`

where `u` is an abstract mechanical control coordinate and `G(z)` is the local transfer operator from control to state velocity.

The crucial word is “abstract.” FER does not assume that `G` is known. It must be identified experimentally.

## A warning from mechanobiology

This is where the theory must be more conservative than the original intuition.

Mechanical stimulation can be adaptive, but it can also be harmful. Recent work has linked compressive stress to Piezo1 activation, Rho-ROCK signaling, histone modification, and persistent epigenetic mechanical memory in cancer models. Reviews of nuclear mechanics also emphasize the possibility of force-induced DNA damage.

That makes the simple intuition “apply the right resonance and unfold the epigenome” scientifically unjustified at present.

The more defensible hypothesis is that some mechanical perturbations might move a cellular state in a direction that can be measured in the same geometric coordinates used to define the target.

That is a much harder claim to test, but it is also a real research question.

## Why acoustic holography is interesting

Acoustic holography supplies the proposed field-synthesis layer.

An abstract acoustic field can be represented as a superposition of controlled waves:

`p(r,t) = Re{Σₘ aₘ(r) exp[i(kₘ·r − ωₘt + φₘ)]}`

Recent biomedical work has demonstrated the ability to shape acoustic fields spatially, including work on cultured-cell applications. This suggests a useful engineering interface: instead of searching directly over biological outcomes, first design a physical field whose mechanical effect is experimentally calibrated.

The scale problem remains severe. Spatially shaped ultrasound does not imply locus-specific chromatin manipulation. A field can be precisely shaped at one scale while its subcellular transfer function is unknown.

FER therefore puts acoustic holography late in the pipeline.

First identify a biological target direction.

Then identify the mechanical observable correlated with that direction.

Then ask whether an external field can reproduce that mechanical observable.

This ordering prevents the project from becoming a search through arbitrary frequencies.

## The proposed experimental sequence

The first experiments should be entirely in vitro. Young and aged reference states would be constructed in a controlled cell system, with single-cell transcriptomics, chromatin accessibility, methylation, selected histone marks, nuclear morphology, and, where possible, three-dimensional chromatin measurements. A pre-registered filtration would define the topological analysis, followed by donor-level validation. The aged and reference populations would then be embedded in the chosen geometry and the Wasserstein direction estimated. Finally, controlled mechanical perturbations would be used to estimate `G(z)` from measured nuclear, chromatin, and mechanosensitive responses. Only then would inverse control become meaningful.

## What would count as a positive result?

A strong positive result would not be “the cells look younger.” It would be a chain of independent observations.

A topological signature should generalize to held-out donors.

A transport direction should predict an observed trajectory.

Mechanical variables should explain state movement beyond exposure magnitude alone.

The controller should preserve lineage identity and genomic integrity.

Finally, a field-synthesis system should reproduce the relevant mechanical observable within the admissible experimental range.

Each link is separately testable.

A convenient alignment statistic is

`A_OT = ⟨v_obs,v_OT⟩ / (||v_obs|| ||v_OT||)`

where `v_OT` is the predicted transport direction and `v_obs` is the measured state velocity.

The value is useful only against blocked-label and permutation null models.

## And what would falsify FER?

A rigorous theory needs an escape route.

Suppose the topological signature does not generalize across donors. Then the topological hypothesis should be rejected or narrowed.

Suppose the transport geodesic does not predict experimental trajectories. Then optimal transport is only a descriptive geometry, not a control geometry.

Suppose mechanical perturbation changes the nucleus but does not move molecular state in the predicted direction. Then the coupling model fails.

Suppose a controller can move state but only by compromising identity or genomic integrity. Then the constrained formulation has failed its purpose.

Suppose the mechanical transfer is real but cannot be reproduced through external acoustic fields. Then acoustic holography is not the correct actuator.

These are scientifically useful outcomes. They tell us which layer is wrong.

## A more disciplined interpretation of the “crumpled paper”

The crumpled-paper analogy remains useful, but only as intuition.

A sheet of paper can be unfolded while retaining the same chemical composition because geometry describes its physical configuration. A cell is more complicated. Its state depends on chemistry, molecular interactions, energy flux, repair systems, history, and mechanical organization. Some molecular state is reversible, some is not, and many variables are coupled.

So the strong statement “aging is geometry” is unnecessary.

The more interesting statement is:

> Some aspects of aging may admit a geometry that is predictive enough to guide control.

That hypothesis is ambitious without pretending to be established.

## Why the combination may still matter

Each major ingredient of FER already exists in a mature field: topological data analysis, information geometry, optimal transport, mechanobiology, active-matter theory, and acoustic field synthesis. The scientific opportunity lies in making the interfaces between them measurable.

That is the proposal: a control stack in which each interface has to earn its place experimentally.

## Where this leaves rejuvenation research

Partial epigenetic reprogramming remains an important contemporary direction, with current work focused on resetting age-associated states while retaining cell identity. First-in-human testing of a cellular rejuvenation approach has begun in 2026, which makes careful measurement of identity, safety, and mechanism increasingly important.

FER does not compete with that program conceptually. It asks whether physical state could become an additional control coordinate.

It is possible that the answer will be no.

A negative result would still identify a route that does not improve biological control.

It is also possible that some mechanical variables turn out to be surprisingly predictive. In that case, geometry could become more than a visualization layer. It could become part of the intervention design problem.

For now, that possibility is a hypothesis.

## Closing

The long-term value of this idea does not depend on calling it revolutionary. It depends on whether its variables can be measured, its assumptions can be challenged, and its failures can be localized.

FER is therefore best understood as a research program: map biological state, measure topology, define transport geometry, identify mechanical coupling, construct constrained controls, and test every link independently.

The intended outcome is not a promise of lifespan extension. It is a sharper question about whether biological aging contains a controllable geometry.

That question can be answered experimentally.

### Research footprint

GitHub: https://github.com/Devanik21/The-Invention-Archive  
Medium: https://medium.com/@devanik2005  
X: https://x.com/devanik2005  
LinkedIn: https://www.linkedin.com/in/devanik  
ORCID: https://orcid.org/0009-0001-8310-6921  
Kaggle: https://www.kaggle.com/devanikdebnath