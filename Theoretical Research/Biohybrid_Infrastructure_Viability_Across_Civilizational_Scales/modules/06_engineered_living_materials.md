# Engineered Living Materials with Bounded Autonomous Repair

**Classification:** H; laboratory research, with application-specific durability and containment open

## Research objective
Test whether a contained living-material component can deliver measurable repair or sensing benefits without unacceptable containment, stability or maintenance costs.

## Biological principle
Engineered living materials couple viable cells with a supporting matrix. Cells may sense changes, produce polymers/minerals, modify local chemistry or emit a signal. The matrix and living component interact.

## Engineering scope and system boundary
Begin with sealed non-pathogenic models and small noncritical coupons. Investigate crack-localized repair, moisture sensing or contaminant detection. Do not place viable cells in exposed vacuum structures or life-critical hulls.

## Mathematical formulation

A minimal reaction–diffusion model couples substrate c and biomass X: partial_t c=D_c Laplacian(c)-k_u cX+s_c; partial_t X=D_X Laplacian(X)+mu(c,T,pH)X-delta X. A damage field d may obey partial_t d=lambda(r,t)-k_r X f(c,T,pH). All rates require strain/matrix-specific measurements.

## Parameters requiring evidence
Cell viability, substrate diffusion, reaction rate, nutrient delivery, moisture, pH, matrix permeability, containment integrity, repair-product chemistry, strength recovery, false-alarm rate and long-term genetic/phenotypic stability.

## First validation protocol
Use non-pathogenic contained models. Introduce controlled defects; measure repair depth, strength recovery, permeability, sensor accuracy and viability. Include inactive-cell and no-cell controls, leakage detection and failure containment.

## Falsification criteria
Reject an autonomous-repair claim if strength/sealing recovery is not reproducible, viability collapses within the claimed window, escape cannot be contained, or maintaining viability costs more than conventional repair.

## Limitations and safety boundary
An engineered living material is a coupled biological-material system, not automatically self-healing. A function must be constrained, inspectable, fail-safe and application-specific.

## Initial source direction
2025 review literature on adaptive/evolving engineered living materials; primary studies to be added during the full literature audit.

**Status:** Proposed module. No experiment or numerical validation is claimed here.
