**1/12** mtDNA heteroplasmy changes through replication, degradation, drift, and selection. Since function can change sharply near a variant-specific threshold, a key question is whether intervention timing matters at equal total dose.

**2/12** The difficult assumption is not the math. It is the actuator: increasing bulk mitophagy flux does not prove preferential removal of mutant genomes. Selectivity must be measured for each variant and target cell context.

**3/12** Baseline model: x is mutant fraction, s is net mutant selection, u(t) is control intensity, and Delta is effective selectivity against the mutant:

ẋ = x(1−x)(s−Delta·u),  0 ≤ u ≤ U.

**4/12** This ODE is not bistable. Its log-odds transform is exact: logit x(t)=logit x(0)+st−Delta·∫u. So final deterministic heteroplasmy depends on total dose, while intermediate states depend on when the dose is delivered.

**5/12** Theorem: for a fixed dose B and bounded control u≤U, front-loading at U until B is spent maximizes cumulative dose at every time t. If Delta>0 is constant, it therefore minimizes x(t) pointwise versus any equal-dose schedule.

**6/12** Consequence: any time-integrated loss that is nondecreasing in heteroplasmy is no worse under the front-loaded schedule. Final x(T) remains equal. This is a model-conditional dominance result—not evidence that pulsed mitophagy is effective.

**7/12** The theorem fails as a favorable ranking if Delta=0 (no schedule effect) or reverses if Delta<0. Delay, state-dependent selectivity, toxicity, compensatory biogenesis, or wild-type depletion can also change the result.

**8/12** A stochastic extension defines threshold first-passage time and cross-cell variance. But the baseline model has no double-well barrier, so a Kramers escape-time formula cannot simply be asserted. Any bistable model needs its own evidence.

**9/12** “Rising variance warns of decline” is a hypothesis, not a law. Test whether variance predicts future threshold burden out-of-sample beyond mean heteroplasmy, mtDNA copy number, cell type, sequencing depth, batch, and lineage effects.

**10/12** First falsification gate: estimate mutant-versus-wild-type selection directly in heteroplasmic cybrids. If Delta is zero, negative, or unresolved, greater mitophagy flux cannot rescue the proposed genotype-selective mechanism.

**11/12** Then compare vehicle, continuous, and front-loaded exposure at matched cumulative input. Pre-register threshold burden and an OXPHOS endpoint; measure copy number, membrane potential, mitochondrial mass, cell survival, and collateral wild-type loss.

**12/12** Synthetic code verifies the algebra; no biological dataset or intervention was tested. No aging reversal or lifespan claim follows. The contribution is a transparent theorem, stochastic endpoints, and experiments designed to kill the hypothesis if selectivity is absent.
