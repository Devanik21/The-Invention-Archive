# Mathematical Framework

## 1. Multilayer infrastructure network

Represent an architecture as a directed multilayer graph \(G=(V,E,\mathcal L)\), where nodes are process units or stores, edges represent material, energy, thermal or information transfers, and \(\mathcal L\) labels the physical layer. Example nodes include photobioreactors, electrolyzers, separators, mines, storage vessels, habitats, collectors, radiators and robotic maintenance units.

Let \(x(t)\in\mathbb R_{\geq0}^{m}\) be conserved material inventories. Each state coordinate must use one declared unit (e.g. mol or kg). Let \(v_B,v_H,v_A\) be process-rate vectors, and \(S_B,S_H,S_A\) their signed stoichiometric / transformation matrices. A candidate open-system balance is

\[
\dot x=S_Bv_B+S_Hv_H+S_Av_A+b_{\rm ext}-Lx.
\tag{1}
\]

Here \(b_{\rm ext}\) is external supply and \(Lx\) is explicitly modelled loss or withdrawal. Each reaction column must close elemental balances or identify a declared external reservoir. Transport and reaction must not be double-counted.

For compartment transport, with graph incidence matrix \(B_G\) and edge flow \(q\),

\[
\dot x=Sv+B_Gq+b_{\rm ext}-\ell(x,w).
\tag{2}
\]

The sign convention for every edge is recorded.

## 2. Resource recovery

For resource \(i\), define \(D_i\) as gross process demand during a fixed interval and boundary, and \(R_i\) as returned material meeting the reuse quality specification. Then

\[
\rho_i=\frac{R_i}{D_i},\qquad 0\leq \rho_i\leq1.
\tag{3}
\]

Net external makeup is computed from a single consistent inventory balance. If losses are already included in the unrecovered portion \(D_i-R_i\), they must not be added twice. Recovery fraction alone is inadequate: concentration, purification energy, contamination, storage and replacement logistics are also counted.

## 3. Energy and exergy

Separate available power from consumed power. A feasible operating point requires

\[
P_{\rm load}=P_B+P_H+P_A+P_{\rm aux}\leq P_{\rm available}.
\tag{4}
\]

All terms must be mutually exclusive loads at the chosen boundary. Photosynthetic systems transform supplied optical energy; they are not free energy sources. Where useful-work quality matters, account for source exergy, exergy destruction and product exergy separately.

## 4. Thermal constraints

For an idealized grey radiating surface,

\[
Q_{\rm rad}=\epsilon\sigma A_{\rm rad}(T_{\rm rad}^{4}-T_{\rm bg}^{4}),
\tag{5}
\]

where emissivity, area, radiator temperature and background temperature are specified. This is a first-order bound, not a complete spacecraft thermal model. Real design must include view factors, spectral emissivity, absorbed radiation, degradation and component temperature limits.

A dynamic thermal node can be written as

\[
C_{\rm th}\dot T=P_{\rm diss}+Q_{\rm absorbed}-Q_{\rm rejected}.
\tag{6}
\]

## 5. Damage, repair and reliability

For component \(j\), a reduced damage state is

\[
\dot d_j=\lambda_j(w,\theta)-r_j(x,u,\theta),
\tag{7}
\]

where \(\lambda_j\) is disturbance-dependent damage accumulation and \(r_j\) is measured/modelled repair or maintenance. If a biological repair mechanism has not been validated for the operating envelope, set its credited repair rate to zero.

A time-varying hazard model gives component survival

\[
R_j(t)=\exp\left[-\int_0^t h_j(s)\,ds\right].
\tag{8}
\]

Network reliability is not generally the product of component reliabilities unless independence and topology assumptions are justified.

## 6. Robust viability

Let \(\mathcal X\) be admissible inventories, temperatures, pressures, damage, contamination and safety states. Let \(\mathcal W\) be a bounded disturbance set. Define

\[
\mathcal K=\left\{x_0\in\mathcal X:\exists\pi\;\forall w(\cdot)\in\mathcal W,\;x(t;x_0,\pi,w)\in\mathcal X\right\}.
\tag{9}
\]

The result depends on specified dynamics, controls, observability and boundary conditions. A viable state in an ideal model does not prove that physical sensors and actuators can realize the policy.

## 7. Multiobjective architecture selection

For architecture \(a\) and policy \(\pi\), use an objective vector

\[
\mathbf J(a,\pi)=
[M_{\rm import},E_{\rm operating},C_{\rm lifecycle},
P_{\rm failure},W_{\rm hazardous},\tau_{\rm recovery}].
\tag{10}
\]

The default output is a Pareto set under common mission constraints. If objectives are scalarized, publish weights and sensitivity. Do not quietly combine mass, risk, time and cost into one undefined efficiency.

Compare B-only, H-only, M-only where appropriate, A-only and mixed architectures under the same system boundary and functional demand.

## 8. AI role

AI may support surrogate modelling, Bayesian parameter estimation, uncertainty propagation, constrained multiobjective search, fault diagnosis, robust model-predictive control and experiment selection. It is not credited with discovering a physical law merely by producing a numerical optimum. Every candidate must pass conservation checks, independent baselines, held-out conditions and explicit failure tests.

## 9. Mandatory checks

- \(v_B=v_H=0\) recovers the abiotic limiting model.
- No-feedstock conditions remove dependent product formation.
- Zero recovery produces the declared makeup requirement.
- Zero damage produces no repair requirement.
- Zero emissivity or zero radiator area produces zero idealized radiative rejection.
- Elemental balances close within declared numerical tolerance.
- Units and dimensions match in every term.

Classify claims as P1 exact mathematical identity, P2 conditional optimization/dynamics, P3 statistical scaling, P4 structural correspondence or P5 analogy. Synthetic model output is never empirical validation.
