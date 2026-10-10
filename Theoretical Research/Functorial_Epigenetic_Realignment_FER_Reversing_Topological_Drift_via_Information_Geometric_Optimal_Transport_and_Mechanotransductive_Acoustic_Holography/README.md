# Functorial Epigenetic Realignment (FER)

**Reversing Topological Drift via Information-Geometric Optimal Transport and Mechanotransductive Acoustic Holography**

**Author:** Devanik Debnath (Devanik21)  
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India  
**ORCID:** https://orcid.org/0009-0001-8310-6921  
**Release date:** 2026-10-10  
**Release:** v2026.10.10

## Research posture

FER is a theoretical and computational proposal. It does not establish that aging is caused by topology alone, that a youthful topological state is uniquely defined, that acoustic forcing can restore chromatin organization, or that any human treatment is safe or effective.

The contribution is an explicit, falsifiable composition of mathematical and physical layers:

**state representation -> topology -> information geometry -> optimal transport -> constrained control -> mechanotransduction -> acoustic field synthesis**

The paper deliberately separates exact mathematical identities from empirical hypotheses and from cross-domain structural analogies.

## Why this problem is worth testing

The modern aging literature treats aging as a network of interconnected processes rather than a single defect. Epigenetic alterations are one component of that system. At the same time, 3D chromatin organization, mechanotransduction, cellular trajectories, and topological data analysis now provide quantitative descriptions of state and structure.

FER asks a specific question:

> Can an age-associated cellular state be represented by a compact geometric signature, and can a calibrated mechanical perturbation move that state along a low-cost trajectory toward a defined reference state without violating lineage identity or genomic-integrity constraints?

This question is narrower than the statement that aging is "really geometry," and it is experimentally falsifiable.

## Core mathematical objects

### 1. Information geometry

For a probability vector p and tangent directions u and v:

$$
g_p(u,v)=\sum_i \frac{u_i v_i}{p_i}.
$$

The Fisher metric supplies a local geometry on the probability simplex.

### 2. Wasserstein transport

For an aged distribution p_a and a reference distribution p_y:

$$
W_2^2(p_a,p_y)=\min_{\pi\in\Pi(p_a,p_y)}\sum_{i,j} c_{ij}\pi_{ij}.
$$

In one dimension, the quantile interpolation gives an exact constant-speed W2 geodesic under the usual regularity assumptions.

### 3. Topological signature

Given a filtration K_epsilon:

$$
\tau = \left\{\beta_0(K_\epsilon),\beta_1(K_\epsilon),\ldots\right\}_{\epsilon\in\mathcal E}.
$$

The proposed topological representation is multiscale. It is not assumed in advance that a single Betti number is a biomarker of age.

### 4. Composite geometric-control objective

$$
\mathcal D(z,z_y)=\alpha W_2^2(p_z,p_y)+\beta d_{\mathrm{top}}^2(\tau_z,\tau_y)+\gamma R_{\mathrm{id}}(z,z_0)+\lambda R_{\mathrm{stress}}(z).
$$

The identity term prevents a controller from labeling dedifferentiation or loss of lineage as successful rejuvenation. The stress term makes mechanical safety part of the optimization rather than an afterthought.

### 5. State dynamics

$$
\dot z=f(z)+G(z)u+\eta.
$$

The matrix or operator G must be measured or calibrated experimentally. FER does not infer G from the existence of LINC, Piezo1, or any other mechanosensor.

### 6. Functorial constraint

Let E denote the category of admissible biological state transitions and A the category of admissible physical controls. FER requires a structure-preserving map F such that:

$$
F(\mathrm{id}_z)=\mathrm{id}_{F(z)}.
$$

and

$$
F(g\circ f)=F(g)\circ F(f).
$$

This is a compositional consistency condition. Category theory does not magically enforce physical law; the physical constraints still reside in the object definitions, dynamics, and calibration data.

## Biological coupling hypothesis

The candidate physical chain is:

**external mechanical field -> membrane/cytoskeleton -> LINC/nuclear lamina -> nuclear deformation -> chromatin organization -> transcriptional state**

The literature supports mechanical coupling between cytoskeletal structures and nuclear/chromatin organization, but it also documents contexts in which excessive mechanical stress damages DNA or produces pathological epigenetic memory. FER therefore treats the coupling as sign-indeterminate until measured.

## Acoustic holography hypothesis

Acoustic holography can synthesize spatially shaped ultrasound fields. FER uses that capability as an actuator-design abstraction, not as evidence that chromatin can currently be addressed at the required subcellular resolution through an intact human body.

The acoustic field is represented abstractly as:

$$
p(\mathbf r,t)=\mathrm{Re}\left\{\sum_m a_m(\mathbf r)e^{i(\mathbf k_m\cdot\mathbf r-\omega_m t+\phi_m)}\right\}.
$$

No therapeutic frequency, pressure, intensity, duty cycle, or exposure schedule is provided.

## Epistemic hierarchy: P1-P5

| Level | Meaning | Treatment in FER |
|---|---|---|
| P1 | Algebraic / topological identity | Definitions and exact identities for Betti numbers, Wasserstein geometry, Fisher geometry, functor composition, and defect charge |
| P2 | Dynamical / variational selection | Minimum-action transport and constrained control trajectories |
| P3 | Statistical / universality hypothesis | Stable topological signatures across donors, batches, scales, and independent datasets |
| P4 | Cross-domain structural mapping | Linking state geometry to calibrated mechanics and field synthesis without asserting a common physical field |
| P5 | Visual analogy | Crumpled-paper intuition; explicitly non-probative |

## What the novelty claim actually is

Topological data analysis has already been proposed as a tool for studying aging. Optimal transport has already been used to reconstruct cellular trajectories and reprogramming dynamics. 3D chromatin changes in aging and senescence are established areas of study. Mechanical signaling through the cytoskeleton, LINC complex, and nucleus is established, and acoustic holography is an active biomedical field.

Therefore FER does **not** claim that these individual ideas are new. The contribution is the explicit control architecture that treats topology and transport as a target geometry, maps geometric state velocities into admissible mechanical controls, and defines experimental falsification criteria for the combined mechanism.

A global claim that no equivalent combination has ever been conceived cannot be proven by a finite literature search. The paper instead defines novelty operationally through a reproducible mathematical specification.

## Numerical verification

Run:

```bash
python3 verify_model.py
```

The script contains no fitted biological parameters and no external data. It verifies:

1. An exact H1 calculation for a toy clique complex over GF(2), used only as a topological proxy.
2. The one-dimensional W2 geodesic and its constant-speed and t-squared distance identities.
3. Fisher-Rao tangent-norm calculation on a probability simplex.
4. Minimum-norm recovery of a linearized control through the Moore-Penrose pseudoinverse.
5. Functorial identity and composition in a deliberately linear toy category.
6. Trace preservation of a toy Lindblad dissipator.
7. Exact +1/2 and -1/2 nematic defect winding and neutral pair charge.

Passing these checks proves only the toy mathematics.

## Proposed experimental program

### Phase I - State atlas

Use donor-matched cultured cells with independently defined young and aged states. Collect single-cell transcriptomics, chromatin accessibility, methylation, selected histone marks, nuclear morphology, and where feasible 3D chromatin contacts. Establish train/validation/test partitions by donor and batch.

### Phase II - Topology and transport

Construct regulatory or multi-omic point-cloud filtrations. Learn persistence diagrams, Betti curves, persistence landscapes or persistence images. Estimate an age-conditioned distribution p_y and a candidate aged distribution p_a. Compare geometric distances with shuffled-feature, degree-preserving graph, batch-only, and cell-composition null models.

### Phase III - Mechanical system identification

Measure nuclear deformation, cytoskeletal state, mechanosensitive signaling, and chromatin-state changes under controlled mechanical perturbations. Estimate the local input-response operator G and quantify uncertainty rather than treating it as known.

### Phase IV - Inverse control

Predict the target state velocity from the OT geodesic and solve the constrained control problem. Compare the learned controller with energy-matched random controls and non-topological baselines.

### Phase V - Field synthesis

Only after in-vitro mechanical coupling has been measured should the team ask whether an external acoustic field can reproduce a calibrated mechanical observable. The first endpoint should be field fidelity, not rejuvenation.

## Primary falsification metrics

### Topological reproducibility

A topological signature must generalize to held-out donors and batches.

### Transport alignment

$$
A_{\mathrm{OT}}=\frac{\langle v_{\mathrm{obs}},v_{\mathrm{OT}}\rangle}{\|v_{\mathrm{obs}}\|\,\|v_{\mathrm{OT}}\|}.
$$

A high alignment must beat blocked-label and permutation null distributions.

### Identity preservation

Cell-type identity, proliferation control, genomic integrity, and function must remain within pre-registered admissibility regions.

### Mechanical specificity

A mechanical observable must explain state displacement beyond energy or exposure duration alone.

### Null outcome

If mechanically induced trajectories do not align with the predicted geometric direction despite calibrated field delivery and measurable mechanotransduction, the corresponding FER coupling hypothesis is rejected.

## Safety and scope boundary

FER is not a treatment protocol. Ultrasound and other mechanical perturbations can produce heating, cavitation, membrane stress, tissue strain, and DNA damage. Recent work also shows that pathological mechanical loading can produce persistent epigenetic effects in disease models. The framework therefore treats damage, identity loss, and off-target state transitions as explicit constraints.

No human experiment is implied by this release.

## Digital research footprint

GitHub: https://github.com/Devanik21/The-Invention-Archive  
Medium: https://medium.com/@devanik2005  
X: https://x.com/devanik2005  
LinkedIn: https://www.linkedin.com/in/devanik  
ORCID: https://orcid.org/0009-0001-8310-6921  
Kaggle: https://www.kaggle.com/devanikdebnath

Zenodo, OpenAIRE, and Google Scholar should be linked to the final persistent record only after an actual record exists. Metadata alone is not evidence of indexing, discoverability, or citation.

## Archive package

This release is isolated under:

```text
Theoretical Research/Functorial Epigenetic Realignment (FER): Reversing Topological Drift via Information-Geometric Optimal Transport and Mechanotransductive Acoustic Holography/
```

The directory contains exactly 11 core files:

1. `paper.tex`
2. `paper.pdf`
3. `verify_model.py`
4. `README.md`
5. `medium_story.md`
6. `x_thread.md`
7. `CITATION.cff`
8. `paper.bib`
9. `.zenodo.json`
10. `schema_article.jsonld`
11. `deploy.py`