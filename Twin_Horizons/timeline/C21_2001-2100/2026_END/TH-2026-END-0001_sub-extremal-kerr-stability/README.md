# TH-2026-END-0001: Physical-Space Carter Multipliers on Sub-Extremal Kerr

**Closing report:** "Physical-Space Carter Multipliers on Sub-Extremal Kerr: Verified Results, Failed Claims, and Open Problems"
**Report date:** 27 November 2026. **Entry created:** 26 November 2026.
**Authors:** Devanik Debnath; Xylia (AI research system).
**Peer review:** none. **Literature search:** not performed in the audit.

This entry is the closing account of a physical-space program on sub-extremal Kerr exteriors, 0 <= |a| < M. It replaces the earlier eight-document series. Where the earlier series claimed more than the audit supports, this README uses the report's status.

---

## 1. Bottom line

The program did **not** establish integrated local energy decay (ILED) for all sub-extremal spins using physical-space multipliers. Several headline claims of the earlier series are withdrawn or downgraded.

What survives is a set of exact kinematic and algebraic results: the trapping-superradiance gap, the near-extremal scalings of the photon sphere, the spin ceilings for the canonical multiplier (within its stated class), the discriminant and Riccati-barrier algebra on the inner region, and the algebra of the spin-2 Regge-Wheeler hierarchy.

What is established as an obstruction is more useful: the horizon-trace problem for transport, the lower-order Hardy absorption (Gap 2A), the superradiant horizon flux (Gap 2B), and the trapped-set derivative loss. The report recommends that future work start from those obstructions rather than from the earlier claims.

The report makes no novelty claim, and no result in it should be described as new on its basis.

---

## 2. Status labels

Every statement in the report carries one of six labels. A label records the check actually performed, and statements are never upgraded by rhetoric.

| Label | Meaning | Check performed |
| --- | --- | --- |
| [V-S] | Symbolically verified | Exact SymPy identity, zero remainder |
| [V-N] | Numerically verified | Reproduced by the scripts in Section 14 |
| [V-H] | Hand-derived, re-checked | Re-derived by hand; not machine-checked |
| [C] | Claimed, not checked | From earlier drafts; not re-derived here |
| [F] | Failed or unsupported | Shown false, unproved, or retracted |
| [O] | Open | Obstruction with the missing ingredient stated |

---

## 3. Final ledger of principal claims

This table is the authoritative status of each claim. No other section should be read as stronger.

| Claim | Status | Note |
| --- | --- | --- |
| Trapping-superradiance gap identity (Thm 3.1) | [V-S][V-H] | Kinematic; no energy estimate follows by itself |
| Near-extremal scalings (Prop 4.1) | [V-H][V-S] | Limit checked symbolically; ZDM identification [C] |
| Spin ceilings 1/2 and phi^(-1/2) (Thm 5.2) | [V-S][V-H] | Within the stated class; scaled-symbol step [C] |
| Discriminant and root formulas (Lem 6.1, 6.2) | [V-S][V-H] | Factorization symbolic; root numeric |
| Barrier on [r+, 3M] (Thm 6.5) | [V-N][V-H] | Strict, but degenerate as sigma -> 0 |
| Barrier on [3M, infinity) | [C] | Admissibility of the interpolation not verified |
| Horizon compatibility, sigma < 1/4 (Prop 6.8) | [V-H] | Conditional on the barrier |
| Far-field Hardy identities (Prop 8.1) | [V-S] | Algebraic identity only |
| Horizon Hardy threshold (Prop 8.2) | [V-S][V-H] | Local; does not predict the throat failure |
| Energy coefficient identity (Lem 7.1) | [V-H] | Classical; no novelty claimed |
| Commuted energies with K^(1/2) | [F] | Nonlocal operator; unsupported |
| Gap 2A uniform closure on [10^-3, 0.15] | [F] | One parameter setting; not swept |
| Gap 2A failure at sigma = 0.5 and 10^-4 | [V-N] | Tested multiplier family only |
| Redshift closure of moderate-spin failure | [F] | Collar-dependent; boundary-localized |
| Spin-2 GRW algebra (Prop 11.1) | [V-S] | Algebra only; master equation taken as input |
| Spin-2 transport bound (drafts' (30)-(31)) | [F] | Sign error; correct identity is (6) below |
| Horizon-trace control for transport | [O] | The genuine open ingredient |
| Nonlinear trapped-set loss | [O] | Correctly identified in the drafts |
| Bootstrap threshold collapse | [F] | Conditional algebra only; scalings heuristic |
| Superradiant flux sign, Gap 2B | [V-H][O] | Band identified; control open |

---

## 4. Verified and checked results

### 4.1 Trapping-superradiance gap (Theorem 3.1)

For 0 < a < M, a trapped exterior radius r0 = r+ + s (s > 0) with omega > 0, m > 0 satisfies omega > m omega+, and

(omega - m omega+)/omega = [s^2 + d s + M d (1 - d/(2s+d))] / (2 M r+) > 0, with d = 2 M sigma.

The identity was re-derived by hand [V-H] and the factorization of H(r) = Delta'(r)(r^2 - r+^2) - 4 r Delta(r) was checked with SymPy [V-S] (check T1). The result is kinematic. It says trapped frequencies avoid the superradiant band. It does not yield an energy estimate or a decay rate. The draft claim that this gap encodes the structural reason linear instabilities are absent is [C] and is not supported by this result alone.

The gap is strictly increasing in s (Corollary 3.3), so its minimum over trapped radii is at the equatorial prograde photon sphere. As sigma -> 0, the gap tends to s^2/(2M^2) (Corollary 3.4).

### 4.2 Near-extremal scalings (Proposition 4.1)

As sigma -> 0+:

- Photon sphere offset: r_ph,+ - r+ = (2/sqrt(3) - 1) M sigma + O(sigma^2) [V-H]
- Relative spectral gap: (omega_ph - m omega+)/omega_ph = (1 - sqrt(3)/2) sigma + O(sigma^2) [V-H]
- Radial Lyapunov exponent: lambda_L = sigma/(2M) + O(sigma^2), with lambda_L/kappa -> 1 [V-H] and [V-S] for the limit (check T2)
- Surface gravity: kappa = sigma / (2M(1 + sigma)) [V-H]

The identification of lambda_L with the damping rate of long-lived near-extremal quasinormal modes is [C]. The report did not re-check the quasinormal-mode literature.

### 4.3 Spin ceilings of the canonical multiplier (Theorem 5.2)

For the canonical multiplier g = w R' with w > 0, within the class of weights used in the drafts:

- Axisymmetric waves (m = 0): nonnegativity on the exterior iff P(r+) >= 0, equivalently |a|/M <= phi^(-1/2) = (sqrt(5) - 1)/2 ... precisely |a|/M <= sqrt((sqrt(5)-1)/2) = 0.78615 (phi = golden ratio).
- General 3D waves: nonnegativity iff P(r+)^2 >= M^2 a^2 (6 r+^2 + a^2), equivalently |a|/M <= 1/2.

where P(r) = 2r^3 - 3M r^2 - M a^2. The horizon factorizations of Lemma 5.1 were checked symbolically [V-S] (checks S1, S2). The reduction to a quadratic in x, the scaled-symbol formula (equation (14) in the drafts), and the monotonicity argument for P^2/(6r^2 + a^2) were not independently re-derived symbolically [C].

Remark 5.3 corrects a typo in the drafts: P(3M) = 27M^3 - M a^2, not 27M^3 - 3M a^2. The conclusion is unaffected.

Remark 5.4: these thresholds are necessary and sufficient only within the stated class of weights and reduction. They do not bound what the perfect-square multiplier in Section 6 can achieve.

### 4.4 Perfect-square multiplier and the Riccati barrier (Sections 6.1 to 6.7)

The multiplier is g = w(R' - rho R) with the coupling 2q/Delta = rho g. The coefficients satisfy -rho' = gamma - beta rho + rho^2 identically [V-H].

- **Discriminant factorization (Lemma 6.1):** B^2 - M^2 Delta^2 = 4 r (r - M)^2 [(r - M)^3 + M^3 sigma^2] > 0 for r > r+ [V-S] (check S3).
- **Smaller root (Lemma 6.2):** gamma_-(r) = (2 M^2 / r)(B + S), with S = 2(r - M) sqrt(r((r-M)^3 + M^3 sigma^2)), and gamma_- > 0 [V-H]. Numerical check at r = 2M, sigma = 1/2 gives gamma_- ~ 0.1560 [V-H].
- **Barrier reduction (Lemma 6.4):** the supersolution rho_bar = 1/(r - r-) reduces to beta rho_bar >= gamma, and for alpha = 0 to gamma(r) <= 2/(r - r-)^2 [V-S] (check S4).
- **Barrier on [r+, 3M] (Theorem 6.5):** 0 <= rho <= 1/(r - r-) on [r+, 3M], conditional on the boundary value 0 <= rho(3M) <= 1/(3M - r-). Verified [V-N][V-H].
- **Degeneration (Remark 6.6):** the ratio gamma_- (r - r-)^2 / 2 approaches 1 as sigma -> 0 at r -> r+. The inequality is strict for every sigma tested, but the margin is about 10^-4 at sigma = 10^-4. The drafts' phrase "strict margin for every sigma" should be read as strict but degenerate.
- **Outer region (Remark 6.7):** the interpolation to the far-field pair on [3M, 4M] and the boundary bound at 3M were not verified. The extension of the barrier to all of [r+, infinity) is [C].

### 4.5 Horizon compatibility obstruction (Proposition 6.8 and Corollary 6.9)

Within the barrier class, a horizon Hardy-positivity condition of the form rho(r+) > 2/M is compatible with rho(r+) <= 1/(2 M sigma) if and only if 4 sigma < 1, i.e. sigma < 1/4 [V-H]. Equivalently, the horizon-end contribution of the omega^2 channel cannot be nonnegative for sigma >= 1/4, which is |a|/M <= sqrt(15)/4 ~ 0.968.

Limits (Remark 6.10): this is a local boundary condition, not the sign of the global eigenvalue. It is necessary, not sufficient. It applies only to multipliers obeying the barrier, which the drafts construct but do not show to be necessary. Its significance is that removing the moderate-spin failure must involve a mechanism outside the barrier class, or a violation of the barrier.

### 4.6 Axisymmetric energy (Section 7)

- The coefficient identity (r^2 + a^2)^2 - a^2 Delta sin^2(theta) = Sigma (r^2 + a^2) + 2 M r a^2 sin^2(theta) > 0 outside the horizon [V-H] (Lemma 7.1).
- The positivity and monotonicity of the Killing energy for m = 0 are classical [C]. No novelty is claimed.
- **Commuted energies with K^(1/2) are unsupported [F].** K^(1/2) is a nonlocal pseudo-differential operator. Its energy positivity, the commutation with the wave operator, and the claimed control of all t-boundary terms were not established. Only the local energy for the time derivative and the stated Killing commutations are retained as [C].

### 4.7 Far-field and horizon Hardy identities (Section 8)

- **Far field (Proposition 8.1, [V-S]):** for w ~ r^(-(1+alpha)) with alpha in (0, 2) and rho = O(r^-3), Hardy_omega^2 + L_lot,omega^2 = 2(alpha - 4)(alpha - 2) > 0 and Hardy_K + L_lot,K = alpha (2 - alpha) > 0. These are algebraic identities only.
- **Horizon (Proposition 8.2, [V-S]):** Hardy_hor + L_lot(r+) = -M^2 sigma^2 rho(r+) g(r+), with g_omega^2(r+) = 4 M r+^2 w(r+) (2 - M rho(r+)) and g_K(r+) = -2 M sigma w(r+) < 0. The K-channel horizon term is positive for every rho >= 0. The omega^2-channel term is nonnegative iff rho(r+) >= 2/M or rho(r+) = 0.

Remark 8.3: this is a local statement at the horizon. It explains the sign pattern at moderate spin. It does not explain the failure at sigma = 10^-4, where rho(r+) ~ 4965 >> 2/M and the eigenfunction sits in an intermediate throat near r ~ 1.13M.

---

## 5. Numerical audit of the lower-order operator (Gap 2A, Section 9)

### 5.1 Operator and discretization

The audit tests the **pure quadratic** Sturm-Liouville operator on each channel:

H_true = -d/dr (2 Delta^2 B_rad d/dr) + L_lot, with B_rad = w (R'' - beta R' + gamma R).

It omits the quartic bulk B_pot = w (R' - rho R)^2 that the earlier preprint included. The earlier CLOSES verdicts were produced with B_pot present and are not verdicts about this operator (Section 5.7).

Discretization: second-order finite differences on the logarithmic grid x = log(r - r+) on [log(x_min sigma), log(r_max - r+ - 1)], r_max = 300 M, with symmetric scaling M^(-1/2) H M^(-1/2) and Jacobian dr/dx = r - r+. The reported eigenvalue is the lowest one. Defaults: N = 1000, x_min = 10^-10, gamma_frac = 0.55, rscale = 4, ptrans = 3. Radau integration with relative tolerance 10^-12 and absolute tolerance 10^-14.

### 5.2 Referee run over the spin range (Table 3, [V-N])

| sigma | abs(a)/M | rho(r+) | rho/rho_bar | K-eig | omega^2-eig | Script status |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.8660254 | 0.45 | 0.44921 | +5.931e-1 | -9.389 | OPEN |
| 0.2500 | 0.9682458 | 1.13 | 0.56398 | +4.711e-1 | -4.615 | OPEN |
| 0.1500 | 0.9886860 | 2.14 | 0.64111 | +3.935e-1 | +5.344 | CLOSES |
| 0.1000 | 0.9949874 | 3.48 | 0.69652 | +3.288e-1 | +1.991e1 | CLOSES |
| 0.0500 | 0.9987492 | 7.78 | 0.77753 | +2.378e-1 | +5.022e1 | CLOSES |
| 0.0100 | 0.9999500 | 45.04 | 0.90081 | +1.120e-1 | +6.268e1 | CLOSES |
| 0.0010 | 0.9999995 | 486.30 | 0.97260 | +3.463e-2 | +2.228e1 | CLOSES |
| 0.0001 | 0.9999999950 | 4965.48 | 0.99310 | +1.048e-2 | -2.096e2 | OPEN |

The "CLOSES" rows hold at one parameter setting only and are not swept, so the window [10^-3, 0.15] is [F] as a general claim (E3).

### 5.3 Convergence (Table 4, [V-N])

- **Sign:** stable under grid size and horizon cutoff at every tested sigma.
- **Magnitude in N:** stable, second order (Figure 7).
- **Magnitude in x_min:** not stable at sigma = 0.5 and 0.25, where it changes by 16% to 30% across cutoffs from 10^-8 to 10^-12. At sigma = 0.5 the eigenvalue goes from -8.326 (x_min = 10^-8) to -9.964 (x_min = 10^-12) at N = 2000 and up. This is the signature of an eigenfunction concentrated against the horizon end of the grid.

Earlier summaries described the eigenvalues as converged. That was true for N only (E2).

### 5.4 The disputed 3D discriminant at sigma = 10^-4 (Table 5, [V-N])

An earlier draft attributed a negative three-dimensional discriminant at sigma = 10^-4, gamma_frac = 0.20 to float64 cancellation. The float64 minimum is -1.0223e5. In 60-digit arithmetic, all nine (rscale, ptrans) configurations give a positive minimum, between +15.34 and +26.02, with zero negative points. The earlier attribution is confirmed (E12). At gamma_frac = 0.55 and 0.90 the two arithmetics agree to all printed digits, so the rounding error is confined to gamma ~ 10^8 at s ~ 10^-12.

### 5.5 The 54-configuration sweep (Table 9, [V-N])

At sigma = 0.5 (27 configurations): rho(r+) in [0.27, 0.64], K-channel eigenvalue in [0.100, 1.475], omega^2-channel eigenvalue in [-15.98, -5.90]. Zero of 27 are positive.

At sigma = 10^-4 (27 configurations): rho(r+) in [4945, 4988], K-channel eigenvalue in [0.0056, 0.0302], omega^2-channel eigenvalue in [-481, -152]. Zero of 27 are positive.

Per spin the counts are 27/27 positive K-channel and 0/27 positive omega^2-channel. The earlier pooled count "54/54 and 0/54" is clarified (E4).

### 5.6 What the audit shows and does not show

It shows that the sign of the omega^2 eigenvalue is stable under grid and horizon cutoff, and across the nine-parameter family at the two extreme spins. It shows the sign pattern is consistent with the horizon threshold of Section 4.5 at moderate spin. It identifies the 3D discriminant artifact as a float64 effect, the only numerical anomaly in the sweep.

It does **not** show the sign at any spin other than the two tested extremes. It does not establish a lower bound on the omega^2 eigenvalue, does not identify the eigenfunction analytically, and does not test a different multiplier family. It does not show that Gap 2A is open for every admissible multiplier. It shows only that the tested family gives a negative eigenvalue at sigma = 0.5 and sigma = 10^-4.

### 5.7 Retraction of the earlier CLOSES verdicts (Section 9.7)

The first audit script (kerr_stability_verify.py, Part 2) reported CLOSES at all six spins. That run included the quartic bulk B_pot, used a different C-infinity multiplier, and took r_max = 40M. It tested a different operator from the one defining Gap 2A. Those verdicts are withdrawn as evidence about Gap 2A.

---

## 6. Horizon redshift patch (Section 10, Table 6)

The drafts propose closing the moderate-spin failure with a Dafermos-Rodnianski-type redshift term. The test added V_red(r) = Lambda_red kappa w(r+) chi((r - r+ - delta)/delta), with a C-infinity cutoff chi, and increased Lambda_red by bisection until the lowest eigenvalue became positive. Two collar choices were tested.

| Collar | sigma | Unpatched omega^2 | Patch covers eigenfunction (collar%) | Smallest closing Lambda | Result |
| --- | --- | --- | --- | --- | --- |
| fixed, delta = 0.05M | 0.5000 | -9.389 | 99.7% | 7.875 | closed, boundary-localized |
| fixed, delta = 0.05M | 0.2500 | -4.615 | 99.4% | 2.741 | closed, boundary-localized |
| fixed, delta = 0.05M | 0.0001 | -209.6 | 51.2% | 13.838 | closed |
| scaled, delta = 0.1 sigma M | 0.5000 | -9.389 | 99.7% | 7.875 | closed, boundary-localized |
| scaled, delta = 0.1 sigma M | 0.2500 | -4.615 | 98.7% | 2.771 | closed, boundary-localized |
| scaled, delta = 0.1 sigma M | 0.0001 | -209.6 | 0.0% | none up to 10^8 | not closed (eig = -8.45) |

Findings: the closure at sigma = 10^-4 depends on where the collar is placed. The moderate-spin closures sit at the first grid points next to the horizon, with eigenvalues of order 10^-7 and 10^-6. That is the signature of a boundary-trapped discrete mode, not a demonstrated bound state. The patch is a potential added with a coupling fixed by bisection. It does not implement the Dafermos-Rodnianski positive bulk term.

**Verdict: [F].** The data do not establish that a redshift term closes Gap 2A at any spin.

---

## 7. Spin-2 Teukolsky-Regge-Wheeler hierarchy (Section 11)

### 7.1 Algebra (Proposition 11.1, [V-S])

The Level-0, Level-1, and Level-2 identities of the hierarchy, including the potential V_RW = 8a^2/w + (8M r^3 - 24 M a^2 r)/w^2 and the coefficients gamma_1, gamma_0, hold with zero remainder (SymPy, Theorem 6 check). This verifies the algebra from the master equation to the GRW equation. It does **not** verify the master equation, the operator K_{-2}, or the physical-space identities (3) and (4) of the drafts independently.

At a = 0, S_spin vanishes and V_RW = 8M/r, consistent with the Regge-Wheeler equation [V-H].

### 7.2 Transport sign error (E1, the most consequential correction)

The drafts' Theorem 2(1) claims uniform transport bounds (equations (30)-(31)) for u1 and u0 with constants independent of sigma. The proof multiplies the transport equation by 2u/r and integrates. The bulk term carries the wrong sign. Integrating d/dr(1/r) = -1/r^2 by parts gives a + sign on the bulk term. The correct identity is

int_{r+}^{infinity} ||u||^2 / r^2 dr = ||u(r+)||^2 / r+ + 2 Re int_{r+}^{infinity} <u, F> / (r w) dr. (6)

The integration-by-parts sign was verified symbolically (check S5). The drafts' inequality (30) does not follow from (6). For any solution with F = 0 and nonzero horizon norm, the left side of the drafts' inequality is strictly positive while the right side vanishes, so the stated bound is false as written [F].

This removes the claimed sigma-uniform transport bounds from the program.

### 7.3 The genuine open point

Equation (6) converts the transport problem into a horizon-trace problem. The bulk weighted L^2 norm of u is controlled if and only if one can control ||u(r+)|| by data and forcing. This trace control is not established and is the actual open ingredient of a sigma-uniform transport bound [O] (O1).

### 7.4 Axisymmetric decoupling and high-frequency closure

For m = 0 the source S_spin reduces to -w^(-1/2) a^2 (gamma_1 u1 + gamma_0 u0) [V-H]. The further claim that the bounded-frequency axisymmetric problem is therefore closed is [C], because it depends on the transport bound with the sign error. The claim that the spin-coupling error is strictly sub-principal on the trapped set is a derivative-counting heuristic with no estimate given [C].

---

## 8. Nonlinear obstructions (Section 12)

- **Quasilinear commutator (Proposition 1 of the earlier series):** the principal commutator identity between h^{mu nu} nabla nabla and the physical-space Carter operator, with the Schouten-Nijenhuis bracket as principal part, was not re-derived, and the bracket's normalization was not checked [C].
- **Trapped-set derivative loss:** the multiplier controls D_rho Phi and the first radial derivative but degenerates on second derivatives tangent to the trapped set. Absorbing the quasilinear error requires pointwise decay of nabla h on the trapped set plus a dynamical trapping gauge. The report agrees this is the right open problem [O] (O5).
- **Bootstrap threshold collapse (Theorem 1 of the earlier Paper 7):** if Delta B <= C1 epsilon + C2 Delta B^2 with C1 >~ sigma^(-p) and C2 >~ sigma^(-q), then closing the bootstrap requires epsilon <= 1/(4 C1 C2). The algebra is correct conditionally. The scaling assumptions on C1 and C2 were argued heuristically from Aretakis-type growth and the decay of a single functional, and do not follow from the drafts' arguments. **Status: [F] as a theorem; [C] as a conditional heuristic.** Even a rigorous version would constrain one bootstrap scheme, not every proof (Remark 12.1). O6 asks for the scalings to be derived for a specified scheme.

---

## 9. Superradiant horizon flux, Gap 2B (Section 13)

For K_+ = d/dt + omega_+ d/dphi generating H+, the time-energy flux density on a mode is omega(omega - m omega+)|psi|^2. It is negative exactly in the band 0 < omega < m omega+, which is nonempty for m > 0 and a > 0 [V-H]. The sign structure is standard [C].

Gap 2B is the statement that no positive-definite physical-space multiplier controls this flux in the superradiant band at |a| ~ M. Theorem 3.1 shows trapped frequencies avoid that band, but trapping is not the only route to the horizon, and the non-axisymmetric problem remains open. The drafts' proposal to use a generalized Whiting transform from a 2026 preprint was not checked [C].

Why it is hard (Remark 13.1): the time-energy is the only conserved positive energy for the stationary symmetry, and its horizon flux is indefinite in the superradiant band. Any positive local estimate must use a different multiplier, such as one involving the azimuthal derivative or a Teukolsky-Starobinsky identity relating spin +2 and spin -2. Neither is established [O] (O4).

---

## 10. Errata and corrections (Section 16)

| ID | Location | Correction | Status |
| --- | --- | --- | --- |
| E1 | Paper 6, Thm 2(1), (30)-(31) | Transport bulk term has the wrong sign; stated bounds do not follow; correct identity (6) | [F] |
| E2 | Gap 2A eigenvalues | Converged in N, but change 16-30% with x_min at sigma = 0.25, 0.5 | corrected |
| E3 | Gap 2A window [10^-3, 0.15] | Based on one parameter setting; not swept | [F] as general claim |
| E4 | Sweep counts | Pooled 54/54 and 0/54 is per-spin 27/27 and 0/27 | clarified |
| E5 | Redshift collar | r05 = rpeak = r+; boundary-localization signature | corrected |
| E6 | Paper 4, Step 3 | P(3M) = 27M^3 - M a^2, not 27M^3 - 3M a^2 | corrected |
| E7 | Sign conventions (Papers 1 vs 3-5) | Same potential on modes; reconciled | no result affected |
| E8 | Paper 5, Prop 1 | Commuted energy with K^(1/2): nonlocal, not established | [F] |
| E9 | Paper 5, Thm 1 | Constants delta(sigma), alpha_0(sigma) not constructed; sketch | [C] |
| E10 | Paper 7, Thm 1 | Scaling assumptions heuristic; conditional | [F] as theorem |
| E11 | Redshift patch | Closure at sigma = 10^-4 depends on collar | [F] |
| E12 | Discriminant at sigma = 10^-4 | Float64 artifact confirmed by 60-digit recomputation | [V-N], confirmed |
| E13 | Paper 4, Step 4 | w(r+) ~ sigma^(-1/2) not verified; rho(r+) <= 1/(2 M sigma) verified [V-H] | [C] / [V-H] |

---

## 11. Open problems: derivation targets (Section 15)

Each is stated as a precise target with the missing ingredient. A solution to any would be a genuine result.

- **O1, horizon trace for transport.** Show ||u1(r+)|| is bounded by initial and forcing data, uniformly in sigma. Equivalent, by (6), to a sigma-uniform bound on the bulk weighted norm. *Missing:* an outgoing energy estimate on the horizon for the transport flow.
- **O2, Gap 2A.** Construct a multiplier family, outside the barrier class of Section 6, for which the omega^2-channel lower-order term is absorbed uniformly on sigma in (0, 1]; or prove none exists. The conditional obstruction of Proposition 6.8 shows a counterexample to uniform closure already exists within the barrier class for sigma >= 1/4. *Missing:* a monotone quantity controlling the throat eigenfunction as sigma -> 0.
- **O3, throat failure at near-extremal spin.** Characterize the sigma = 10^-4 eigenfunction at r ~ 1.13M, outside any O(sigma) collar. Determine whether the sign is controlled by the approach of rho/rho_bar to 1 (reached 0.9976 in the tests). *Missing:* a uniform lower bound on the throat contribution.
- **O4, superradiant flux.** Identify a positive-definite physical-space multiplier, or a Teukolsky-Starobinsky-type identity, controlling the time-flux in 0 < omega < m omega+ at |a| ~ M.
- **O5, trapped-set derivative loss.** Construct a dynamical trapping gauge whose approximate hidden Killing tensor keeps the trapped set aligned with the multiplier to the required order.
- **O6, threshold scaling.** Derive, not assume, the scaling of C1(sigma) and C2(sigma) in the bootstrap inequality for a specified scheme.

---

## 12. Literature and novelty status (Section 17)

No literature search was performed in the audit. Statements about published and preprint work in the earlier drafts are carried as [C] and were not checked against sources. This includes the claims attributed to Dafermos and Rodnianski, Andersson and Blue (2015), Yang et al. (2013), Cardoso et al. (2009), Aretakis, Stogin, Brill, Starobinsky, Bardeen, Press and Teukolsky (1972), Giorgi and Wan (2024), Hintz (arXiv:2606.28253), Szeftel (arXiv:2609.40068), and He and Klainerman (arXiv:2607.08958, arXiv:2609.06344).

**The earlier novelty audit (its Tables 1 and 2) is withdrawn in full.** The previous README's claim that Paper 4 was a primary novelty candidate is withdrawn. No result here should be described as new or original on the basis of this document. The status labels record what was checked, which is a narrower thing.

---

## 13. Companion preprint: Affective Cusp Model (separate study, not Kerr)

The companion preprint "Topological Psychology of the Affective Life Space" (20 November 2026) is a separate study with its own status. It is a one-coordinate cusp model with explicit hypotheses, no fitted data, and a stated ethical boundary. Its bifurcation topology, hysteresis width, and Kramers scaling were reproduced in its own numerical experiments. Its results are model statements and are not validated against human data.

Its Section 5.4 states that "a faster build-up does not produce an earlier outburst, it can delay or prevent one within a fixed time window." Its own Table 4 contradicts the first part. Among the three ramps that produced an outburst, the jump times are 1943.6, 496.9, and 130.3 for speeds 0.0005, 0.0020, and 0.0080, so faster ramps produced earlier outbursts. Only the fastest ramp, 0.032, was censored in the window. **Status: [F] for the original wording.** The corrected statement is that faster ramps produced earlier outbursts with larger overshoot up to the speed tested, and the fastest tested ramp produced no outburst in the window.

This study does not belong to the Kerr program and may warrant its own archive entry.

---

## 14. Reproducibility (Appendix B)

Each script was executed in the audit environment: Python 3, NumPy, SciPy, SymPy 1.14, mpmath 1.3. Full outputs were inspected.

- **kerr_stability_verify.py.** Part 1: symbolic checks of the trapping gap (T1), the triple-collision limit (T2), the perfect-square Riccati identity, and the spin-2 Teukolsky-GRW hierarchy (Levels 0, 1, 2). All passed with zero remainder. Part 2: the first Gap 2A table, which included B_pot and is not a verdict on the pure quadratic operator (Section 5.7).
- **kerr_referee_gap2a.py.** Symbolic far-field and horizon Hardy identities (F1, F2, passed). Pure quadratic Gap 2A referee table (Table 3).
- **kerr_convergence_sweep.py.** Grid and cutoff scan (Table 4).
- **kerr_multiplier_sweep.py.** 54-configuration sweep (Table 9).
- **kerr_redshift_patch.py.** Redshift patch in two collar modes (Table 6).
- **closing_audit_checks.py.** Checks S1 to S5 (SymPy); Riccati barrier maximum ratio (T5); 60-digit recheck of the discriminant at sigma = 10^-4, gamma = 0.20 (Table 5).
- **extra_checks.py.** 60-digit recheck of all nine gamma_frac = 0.20 configurations (Table 5); horizon compatibility window.

Check identifiers: T1 trapping-gap identity; T2 triple-collision limit; S1 P(r+) factorization; S2 3D ceiling factorization; S3 discriminant factorization; S4 barrier reduction identity; S5 integration-by-parts sign for the transport identity; T5 maximum barrier ratio; F1, F2 far-field and horizon Hardy identities.

Numerics: Radau with relative tolerance 10^-12 and absolute tolerance 10^-14; default grid N = 1000 unless stated; 60-digit recheck via mpmath at precision 60.

The earlier series described these scripts as not yet run. The closing report states they were run.

---

## 15. Corrections to the previous version of this README

The previous README overstated several results. The ledger in Section 3 supersedes it. Specifically:

- Theorem 3 (spin ceilings) was listed as "Proved (sharp)". It is [V-S][V-H] within the stated class, with the scaled-symbol step [C].
- Paper 2 (triple collision) was listed as "Proved". The scalings are [V-H] with the limit [V-S]; the zero-damped-mode identification is [C].
- Paper 4 (perfect-square barrier) was listed as "Proved, all |a| < M". The barrier on [r+, 3M] is [V-N][V-H] and degenerates as sigma -> 0; the outer interpolation is [C]. The global claim is not established.
- Paper 5 (strict ellipticity) was listed as "Partial". Its constants were never constructed (E9, [C]), and the commuted energy with K^(1/2) is [F] (E8).
- Paper 6 (spin-2 hierarchy) was listed as "Proved (self-reported algebra)". The algebra is [V-S]; the transport bound is [F] (E1).
- Paper 7 (nonlinear) was listed as "Partial". The collapse theorem is [F] as a theorem and [C] as a heuristic; the trapped-set loss is [O].
- Paper 8's verification "not yet run" is superseded by Section 14.
- The byline was given as "Gemini 4 Neon High". The closing report names the authors as Devanik Debnath and Xylia (AI research system).

Two items from my own earlier review are not in the closing report and should still be checked:

- In Theorem 3 of the earlier series, the sigma form of the axisymmetric ceiling appears as sigma >= sqrt((sqrt(5) - 1)/2). From |a|/M <= phi^(-1/2) and sigma^2 = 1 - a^2/M^2, the correct form is sigma >= (sqrt(5) - 1)/2 ~ 0.618.
- In the report's Theorem 5.2(a), the equivalent form reads "sigma sqrt(2) + sigma - 1 >= 0" in the PDF text. Lemma 5.1 gives P(r+) = 2M^3 (sigma + 1)(sigma^2 + sigma - 1), so the intended form is sigma^2 + sigma - 1 >= 0. This is probably a text-extraction error, but the source should be checked.

---

## 16. Files

- Kerr_Closing_Report.pdf: the closing report. The PDF is binary and is not committed in this folder.
- Kerr_Closing_Report.tex: the source. The archive permits only .md, .json, .txt, and .zip outside snapshot/, so it should be included in the snapshot zip with the other PDFs.
- The eight earlier PDFs are described in Section 15 and the entry's earlier version.
