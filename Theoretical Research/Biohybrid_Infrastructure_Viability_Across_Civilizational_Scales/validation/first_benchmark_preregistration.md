# First Benchmark — Pre-registration

**Status:** Planned. No benchmark has been executed and no result is reported.

## Research question

Does an integrated biological or biohybrid life-support architecture reduce total external resupply mass or improve resilience relative to an abiotic reference while meeting identical air, water and nutrition service constraints?

## Primary comparison

A) Abiotic reference: conventional physical-chemical regeneration and imported food/consumables as needed.

B) Biological reference: regenerative plant/algal/microbial compartments with conventional support equipment.

C) Biohybrid candidate: biological compartments plus abiotic pretreatment, polishing, safety isolation and control.

The functional unit is one defined crew-equivalent operating day at a declared habitat demand profile. The same requirements, initial inventories, mission duration and safety thresholds apply to all architectures.

## Primary outcomes

1. Total external makeup mass per crew-equivalent day.
2. Electrical energy per crew-equivalent day.
3. Required buffer inventory and reserve mass.
4. Fraction of simulation time all service constraints are satisfied.
5. Recovery time after a declared single-component failure.

## Secondary outcomes

Water, carbon, nitrogen and phosphorus recovery; waste/byproduct mass; heat rejection; crew maintenance time; sensitivity to uncertain biological rates.

## Pre-registered failure conditions

The hybrid-nondominance hypothesis is not supported if:
- the candidate violates any service or safety constraint;
- its makeup mass and energy are both no better than the best feasible reference;
- its Pareto advantage disappears over plausible parameter uncertainty;
- or it depends on omitting reactor, lighting, feedstock, purification, containment or replacement burdens.

## Data provenance

Parameters must come from cited public sources or explicitly designated engineering assumptions. Each value records units, source, experimental setting, uncertainty and scale-transfer caveat. Synthetic parameters are allowed for software tests only and must not be described as empirical.

## Statistical protocol

Use a declared ensemble of parameter samples with a published seed and sampling distributions. Report Monte Carlo convergence, sensitivity to distribution choice and uncertainty in comparative metrics. Do not tune parameters after seeing the desired result.

## Reproducibility gate

Before execution, commit the model equations, parameters, random seed, outputs and assertions. Record software versions and stdout/stderr. Negative results are retained.
