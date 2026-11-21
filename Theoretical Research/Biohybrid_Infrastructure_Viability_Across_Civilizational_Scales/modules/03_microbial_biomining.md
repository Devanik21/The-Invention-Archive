# Microbial and Fungal Biomining / Resource Refineries

**Classification:** B/H; terrestrial bioleaching is established, while asteroidal material has ISS proof of concept but no industrial validation

## Research objective
Determine when microbial or fungal leaching improves recovery per unit reactor mass, power, water, reagent, time and downstream purification.

## Biological principle
Acid-producing bacteria and fungi can mobilize selected elements through acidification, redox chemistry, complexation and mineral-surface interactions. Organism and substrate specificity is central: one organism will not improve extraction of every element.

## Engineering scope and system boundary
Model characterized mineralogy, particle size, accessible surface area, organism growth, lixiviant composition, pH, temperature, gas transfer and gravity regime. Separate primary extraction from filtration, concentration, solvent extraction, electrorefining and waste treatment.

## Mathematical formulation

For element i, model dissolved inventory m_i(t), accessible solid fraction s_i(t) and solution volume V. A kinetic form may be dot(m_i)=k_i(X,pH,T,g,a_s)A_s-loss_i, with k_i calibrated from specific data. Recovery yield is recovered mass divided by initial element mass, not simply dissolved concentration. Include reagent and water balances.

## Parameters requiring evidence
Feedstock mineralogy, element grade, particle distribution, surface area, organism identity, growth rate, leaching kinetics, pH, temperature, redox potential, gravity, reactor mass, liquid/solid ratio and downstream recovery yield.

## First validation protocol
Use the BioAsteroid ISS findings as motivation, not a universal performance coefficient. Reanalyse element-specific data and uncertainty; compare biological, abiotic and hybrid pretreatment on the same mineral feed and downstream refining route.

## Falsification criteria
Reject biological assignment for an element if lifecycle recovered mass, usable purity, power, mass or safety is not better than the abiotic baseline within the target mission envelope.

## Limitations and safety boundary
The ISS work used small sample volumes and reported high variability. Asteroidal material is heterogeneous. One palladium result does not generalize to every element, rock or process scale.

## Initial source direction
Microbial biomining from asteroidal material onboard the ISS, npj Microgravity 12, 23 (2026), DOI 10.1038/s41526-026-00567-3.

**Status:** Proposed module. No experiment or numerical validation is claimed here.
