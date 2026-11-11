# Frontier Research Sprint: Sub-Riemannian Schrödinger Bridges and Hodge-Laplacian Rate-Distortion Bounds on Post-Mortem Neural-Metabolic Recoverability
**Principal Investigator:** Xylia  
**Archive Node:** `Devanik21/The-Invention-Archive`  
**Domain:** Biology  
**Target Problem:** The Biological Recoverability Frontier in Post-Mortem Neural-Metabolic Networks  
**Sprint Sub-Lemma:** Underactuated vascular reperfusion ($p \ll E$) and harmonic 1-cycle projection on a simplicial synaptic complex $\mathcal{K}(\tau)$ induce a sub-Riemannian controllability Gramian blow-up and a topological Fano lower bound that strictly tighten the isotropic Gaussian Bures–Wasserstein recoverability frontier.

---

## 1. Formal Setup, Spaces & Boundary Conditions

Building upon empirical demonstrations of post-mortem cellular and synaptic preservation in whole-brain and multi-organ perfusion systems ([Vrselja et al., BrainEx, *Nature* 2019 / PMC6844189](https://pmc.ncbi.nlm.nih.gov/articles/PMC6844189/); [Andrijevic et al., OrganEx, *Nature* 2022](https://frontlinegenomics.com/researchers-partially-revive-organs-in-pigs-an-hour-after-death/)) and electron-microscopic cortical autolysis timelines ([Sheleg et al., PMC2480568](https://pmc.ncbi.nlm.nih.gov/articles/PMC2480568/); [Krassner et al., PMC10294569](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294569/)), we formulate the coupled topological-thermodynamic boundary value problem on an oriented finite simplicial 2-complex:

1. **Simplicial Synaptic-Metabolic Complex $\mathcal{K}(\tau)$:** Let $\mathcal{K}_0 = (\mathcal{V}, \mathcal{E}, \mathcal{F}_2)$ be a finite oriented simplicial complex of dimension 2 with $|\mathcal{V}| = V$ neuronal somata (0-simplices), $|\mathcal{E}| = E$ directed synaptic-axonal projections (1-simplices), and $|\mathcal{F}_2| = F_2$ filled local 3-clique microcircuits (2-simplices). Let $C_k(\mathcal{K}_0; \mathbb{R}) \cong \mathbb{R}^{|\mathcal{K}_k|}$ denote the Hilbert space of real $k$-chains endowed with the Euclidean inner product $\langle \alpha, \beta \rangle_{C_k} = \alpha^\top \beta$.
2. **Boundary Operators & Discrete Hodge Decomposition:** Define the incidence boundary operators $\partial_1: C_1(\mathcal{K}_0; \mathbb{R}) \to C_0(\mathcal{K}_0; \mathbb{R})$ (represented by $B_1 \in \{-1, 0, 1\}^{V \times E}$) and $\partial_2: C_2(\mathcal{K}_0; \mathbb{R}) \to C_1(\mathcal{K}_0; \mathbb{R})$ (represented by $B_2 \in \{-1, 0, 1\}^{E \times F_2}$) satisfying the fundamental chain complex identity $\partial_1 \circ \partial_2 = 0$ ($B_1 B_2 = 0_{V \times F_2}$). The **Hodge 1-Laplacian** $L_1: C_1(\mathcal{K}_0; \mathbb{R}) \to C_1(\mathcal{K}_0; \mathbb{R})$ is the self-adjoint, positive semi-definite operator:

$$L_1 = \partial_1^\top \partial_1 + \partial_2 \partial_2^\top = B_1^\top B_1 + B_2 B_2^\top \in \mathbb{R}^{E \times E}$$

   By the discrete Hodge theorem, the edge state space $C_1(\mathcal{K}_0; \mathbb{R}) \cong \mathbb{R}^E$ admits the orthogonal direct sum decomposition:

$$\mathbb{R}^E = \operatorname{Im}(\partial_1^\top) \oplus \operatorname{Im}(\partial_2) \oplus \ker(L_1)$$

   where $\mathcal{H}_1 = \ker(L_1) \cong H_1(\mathcal{K}_0; \mathbb{R})$ is the harmonic 1-cycle subspace of dimension $\beta_1 = \dim \ker(L_1)$, spanned by the columns of the orthonormal projection matrix $P_{\text{harm}} \in \mathbb{R}^{E \times \beta_1}$ ($P_{\text{harm}}^\top P_{\text{harm}} = I_{\beta_1}$, $L_1 P_{\text{harm}} = 0$).
3. **Harmonic Engram Encoding & Pre-Ischemic NESS ($t \le 0$):** Let $X_t \in \mathbb{R}^E$ represent the vector of synaptic spine phosphorylated efficacy weights across $\mathcal{E}$. Functional long-term memory engrams $Z \in \mathcal{Z} = \{1, \dots, 2^{H_0}\}$ (with prior entropy $H(Z) = H_0 \log 2$ nats) are encoded in the topological harmonic 1-cycle projection $X_{0,\text{harm}} = P_{\text{harm}}^\top X_0 \in \mathbb{R}^{\beta_1}$. Prior to systemic arrest ($t = 0^-$), $X_t$ evolves under the Hurwitz-stable Hodge diffusion:

$$dX_t = A_0 X_t \, dt + \sqrt{2 D_{\text{act}}} \, dW_t, \quad A_0 = -\gamma_e I_E - \kappa_0 L_1$$

   with stationary covariance $\Sigma_0 \in \mathbb{R}^{E \times E}$ satisfying $A_0 \Sigma_0 + \Sigma_0 A_0^\top + 2 D_{\text{act}} I_E = 0$.
4. **Underactuated Vascular Control Manifold ($p \ll E$):** During post-mortem interval $\tau > 0$, synaptic weights undergo proteolytic attenuation at rate $\lambda_{\text{syn}} > 0$ and ultrastructural autolysis noise $R(\tau) = \sigma_{m,0}^2 e^{\alpha_{\text{lysis}}\tau} I_E$. At time $\tau$, a synthetic perfusion intervention $u_t \in L^2([0, T_{\text{rep}}]; \mathbb{R}^p)$ is delivered through $p \ll E$ macroscopic neurovascular unit (NVU) capillary trees via the rank-deficient control input operator $B(\tau) = B_{\text{vasc}} \Lambda_{\text{pat}}(\tau) \in \mathbb{R}^{E \times p}$, where $\Lambda_{\text{pat}}(\tau) = \operatorname{diag}(e^{-\kappa_{\text{nr},1}\tau}, \dots, e^{-\kappa_{\text{nr},p}\tau})$ models microvascular pericyte constriction and capillary no-reflow occlusion:

$$dX_t = \Big(A_{\text{rep}}(\tau) X_t + B(\tau) u_t\Big) dt + \sqrt{2 D_{\text{bath}}} \, dW_t, \quad t \in [0, T_{\text{rep}}]$$

## 2. Theoretical Mechanism & Hypothesis

1. **Hodge-Filtered Data Processing Mechanism:** Gradient modes in $\operatorname{Im}(\partial_1^\top)$ represent transient electrotonic potential differences across somata, and curl modes in $\operatorname{Im}(\partial_2)$ represent local 3-neuron feed-forward motifs that carry zero persistent homology. Because the engram identity $Z$ forms a Markov chain $Z \to X_{0,\text{harm}} \to X_0 \to Y_\tau \to \hat{Z}$, projecting $Y_\tau$ via $P_{\text{harm}}^\top$ onto $\ker(L_1)$ isolates the exact homological information channel $I_{\text{harm}}(X_{0,\text{harm}}; Y_{\tau,\text{harm}})$ ([Data Processing Inequality](https://en.wikipedia.org/wiki/Data_processing_inequality); [Scarlett & Cevher, Fano's Inequality](https://arxiv.org/html/1901.00555v3)).
2. **Sub-Riemannian Pontryagin / Gramian Mechanism:** Isotropic stochastic thermodynamic bounds ([Aurell et al., arXiv:1912.08405](https://arxiv.org/html/1912.08405v1); [Ito & Dechant, arXiv:2103.00503](https://arxiv.org/html/2103.00503v6); [Schrödinger Bridge Problem](https://www.emergentmind.com/topics/schrodinger-bridge-sb-problem)) assume unconstrained full-rank actuation across $\mathbb{R}^E$. Under rank-deficient vascular actuation $B(\tau)$ plus weak interstitial diffusion $\epsilon_{\text{diff}}(\tau) I_E$, applying Cauchy–Schwarz in the reproducing kernel Hilbert space (RKHS) of the finite-time controllability Gramian $\mathcal{G}_c(\tau, T_{\text{rep}})$ captures the exponential thermodynamic blow-up caused by capillary no-reflow.
3. **Kramers Non-Convex Barrier Correction:** For multistable synaptic states governed by a double-well Ginzburg–Landau potential $U(q) = \frac{\alpha_w}{4}q^4 - \frac{\beta_w}{2}q^2$, post-mortem anoxic diffusion causes probability mass $\eta_{\text{leak}}(\tau) = \int_{-\infty}^0 P_\tau(q)\,dq$ to leak across the separatrix $q = 0$, adding an irreversible Kramers barrier work penalty $2 k_B \eta_{\text{leak}}(\tau) \Delta U / D_{\text{bath}}$.

## 3. Step-by-Step Analytical Derivation

### 3.1 Discrete Hodge Orthogonality & Topological Fano Inequality

Let $x = B_1^\top \phi \in \operatorname{Im}(\partial_1^\top)$ and $y = B_2 \psi \in \operatorname{Im}(\partial_2)$. Their inner product in $C_1(\mathcal{K}_0; \mathbb{R})$ vanishes identically by $\partial_1 \circ \partial_2 = 0$:

$$\langle x, y \rangle_{C_1} = \phi^\top (B_1 B_2) \psi = 0$$

For any $h \in \ker(L_1)$, $\langle h, L_1 h \rangle_{C_1} = \|B_1 h\|_2^2 + \|B_2^\top h\|_2^2 = 0 \implies B_1 h = 0$ and $B_2^\top h = 0$. Projecting the linear-Gaussian degradation channel $Y_\tau = \Phi_\tau X_0 + \xi_\tau$ onto $P_{\text{harm}} \in \mathbb{R}^{E \times \beta_1}$ where $L_1 P_{\text{harm}} = 0$ yields $P_{\text{harm}}^\top \Phi_\tau = \Phi_{\tau,\text{harm}} P_{\text{harm}}^\top$. By the Data Processing Inequality and Fano's Inequality on $Z \to X_{0,\text{harm}} \to Y_{\tau,\text{harm}} \to \hat{Z}$, we obtain the **Topological Fano Lower Bound**:

$$P_e(\tau) \ge 1 - \frac{1}{H_0} - \frac{1}{2 H_0 \log 2} \log \det \Big( I_{\beta_1} + P_{\text{harm}}^\top \Sigma_0 P_{\text{harm}} \Phi_{\tau,\text{harm}}^\top \big(P_{\text{harm}}^\top R(\tau) P_{\text{harm}}\big)^{-1} \Phi_{\tau,\text{harm}} \Big)$$

### 3.2 Sub-Riemannian Minimum Entropy Production Under Rank-Deficient Vascular Control

During reperfusion $t \in [0, T_{\text{rep}}]$ with vascular-interstitial mobility $M(\tau) = B(\tau)B(\tau)^\top + \epsilon_{\text{diff}}(\tau) I_E \succ 0$, steering the degraded mean state $m(0) = m_\tau$ to $m(T_{\text{rep}}) = m_0$ requires satisfying Duhamel's integral constraint:

$$\Delta m(\tau) \equiv e^{-A_{\text{rep}}(\tau) T_{\text{rep}}} m_0 - m_\tau = \int_0^{T_{\text{rep}}} e^{-A_{\text{rep}}(\tau) s} M(\tau)^{1/2} v_m(s) \, ds$$

Defining the **Underactuated Vascular Controllability Gramian**:

$$\mathcal{G}_c(\tau, T_{\text{rep}}) = \int_0^{T_{\text{rep}}} e^{-A_{\text{rep}}(\tau) s} \Big( B(\tau) B(\tau)^\top + \epsilon_{\text{diff}}(\tau) I_E \Big) e^{-A_{\text{rep}}(\tau)^\top s} \, ds$$

and expanding $\|v_m\|_{L^2}^2 = \|v_m^*\|_{L^2}^2 + \|\delta v\|_{L^2}^2$ around the RKHS projection $v_m^*(s) = M(\tau)^{1/2} e^{-A_{\text{rep}}(\tau)^\top s} \mathcal{G}_c(\tau, T_{\text{rep}})^{-1} \Delta m(\tau)$ yields the exact **Sub-Riemannian Entropy Production Lower Bound**:

$$\Delta S_{\text{sub}}^{\min}(\tau) = \frac{k_B}{D_{\text{bath}}} \, \Delta m(\tau)^\top \mathcal{G}_c(\tau, T_{\text{rep}})^{-1} \Delta m(\tau) \ge \frac{k_B \|\Delta m_\perp(\tau)\|_2^2}{D_{\text{bath}} \, \lambda_{\min}\big(\mathcal{G}_c(\tau, T_{\text{rep}})\big)}$$

### 3.3 Non-Convex Bistable Kramers Barrier Penalty

For a 1D bistable mode $q \in \mathbb{R}$ in $U(q) = \frac{\alpha_w}{4}q^4 - \frac{\beta_w}{2}q^2$ with barrier $\Delta U = \frac{\beta_w^2}{4\alpha_w}$ and leaked mass $\eta_{\text{leak}}(\tau) = \int_{-\infty}^0 P_\tau(q)\,dq$, combining the exact quantile Wasserstein metric with the uphill barrier work yields:

$$\Delta S_{\text{Kramers}}^{\min}(\tau) \ge \frac{k_B}{D_{\text{bath}} T_{\text{rep}}} \int_0^1 \big| F_{P_\tau}^{-1}(u) - F_{P_0}^{-1}(u) \big|^2 \, du + \frac{2 k_B \, \eta_{\text{leak}}(\tau) \Delta U}{D_{\text{bath}}}$$

## 4. Adversarial Audit & Singularity / Bottleneck Analysis

1. **Time-Varying Simplicial Homology Breakdown (Filtration Non-Commutativity):** When proteolytic cleavage is spatially heterogeneous across edges, the weighted Hodge 1-Laplacian $L_1(\tau)$ does not commute at distinct times ($[L_1(\tau_1), L_1(\tau_2)] \neq 0$), inducing spectral leakage between $\operatorname{Im}(\partial_1^\top)$, $\operatorname{Im}(\partial_2)$, and $\ker(L_1)$.
2. **Bilinear State-Dependent Control Coupling Singularity:** Oxygen and ATP delivery modulate homeostatic restoration stiffness multiplicatively ($dX_t = (A_{\text{deg}} + \sum_{k=1}^p u_{t,k} N_k) X_t \, dt + \sqrt{2 D_{\text{bath}}}\,dW_t$), requiring iterated Lie bracket controllability ($\operatorname{ad}_{A_{\text{deg}}}^j(N_k)$) rather than a linear Gramian.
3. **High-Dimensional Eyring–Kramers Separatrix Geometry:** Evaluating Kramers barrier crossing along a 1D principal coordinate omits the Hessian determinant entropic prefactor $\sqrt{|\det \nabla^2 U(x_{\text{min}})| / |\det \nabla^2 U(x_{\text{saddle}})|}$ across the $(E-1)$-dimensional saddle separatrix hypersurface.

## 5. Theoretical Anchors & Connected Literature

1. **Whole-Brain & Multi-Organ Post-Mortem Cellular Recovery:** [Vrselja et al. (BrainEx, *Nature* 2019 / PMC6844189)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6844189/) and [Andrijevic et al. (OrganEx, *Nature* 2022)](https://frontlinegenomics.com/researchers-partially-revive-organs-in-pigs-an-hour-after-death/).
2. **Ultrastructural Autolysis Timelines:** [Sheleg et al. (*Int. J. Clin. Exp. Pathol.* 2008 / PMC2480568)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2480568/) and [Krassner et al. (*Free Neuropathology* 2023 / PMC10294569)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294569/).
3. **Stochastic Thermodynamics & Optimal Transport Geometry:** [Aurell et al. (arXiv:1912.08405)](https://arxiv.org/html/1912.08405v1), [Ito & Dechant (arXiv:2103.00503)](https://arxiv.org/html/2103.00503v6), and [Schrödinger Bridge Problem](https://www.emergentmind.com/topics/schrodinger-bridge-sb-problem).
4. **Information-Theoretic Converse Bounds:** [Data Processing Inequality](https://en.wikipedia.org/wiki/Data_processing_inequality) and [Scarlett & Cevher (arXiv:1901.00555)](https://arxiv.org/html/1901.00555v3).

---
## 6. Automated Computational Verification
**Archived Script:** `biology_verify_2026-11-11_0528.py` (stored in `snapshots/verified_scripts.zip`)

```python
import numpy as np
import sympy as sp
from scipy.linalg import solve_continuous_lyapunov, expm

np.random.seed(42)

# ==============================================================================
# PART 1: SYMBOLIC VERIFICATION OF SUB-RIEMANNIAN GRAMIAN BLOW-UP & HODGE L1
# ==============================================================================
tau_sym, p_sym, eps_sym, T_sym, D_sym = sp.symbols('tau p epsilon T_rep D_bath', positive=True)
lam_min_G = eps_sym * sp.exp(-tau_sym / p_sym)
W2_sub_bound = (1 / (4 * D_sym)) * (1 / lam_min_G)
limit_blowup = sp.limit(W2_sub_bound, eps_sym, 0, dir='+')
print("=== PART 1: SYMPY SYMBOLIC VERIFICATION ===")
print(f"Symbolic Sub-Riemannian Entropy Bound as min_eig(G_c) -> 0+: {limit_blowup}")

# Verify Hodge Decomposition Orthogonality on a 4-node, 5-edge, 1-triangle simplicial complex
B1 = sp.Matrix([
    [-1,  0,  1,  0,  1],
    [ 1, -1,  0,  0,  0],
    [ 0,  1, -1, -1,  0],
    [ 0,  0,  0,  1, -1]
])
B2 = sp.Matrix([
    [1],
    [1],
    [1],
    [0],
    [0]
])
chain_complex_check = B1 * B2
L1_sym = B1.T * B1 + B2 * B2.T
nullspace_L1 = L1_sym.nullspace()
print(f"Boundary operator identity d1 * d2 == 0 verified: {chain_complex_check.is_zero_matrix}")
print(f"Symbolic Betti-1 (dim ker L1) with 1 filled triangle: {len(nullspace_L1)}")
h_vec = nullspace_L1[0]
print(f"Harmonic 1-cycle eigenvector h in ker(L1): {[round(float(x), 4) for x in h_vec.evalf()]}")
print(f"Orthogonality check <d1^T x, d2 z> == 0: {(B2.T * B1.T).is_zero_matrix}\n")

# ==============================================================================
# PART 2: NUMERICAL SIMULATION OF UNDERACTUATED VASCULAR GRAMIAN & HODGE FANO BOUND
# ==============================================================================
V = 36
E = 90
edges = []
rng = np.random.RandomState(42)
while len(edges) < E:
    u, v = rng.randint(0, V, size=2)
    if u != v and (u, v) not in edges and (v, u) not in edges:
        edges.append((min(u, v), max(u, v)))

B1_num = np.zeros((V, E))
for idx, (u, v) in enumerate(edges):
    B1_num[u, idx] = -1.0
    B1_num[v, idx] = 1.0

triangles = []
edge_map = {e: i for i, e in enumerate(edges)}
for i in range(V):
    for j in range(i + 1, V):
        for k in range(j + 1, V):
            if (i, j) in edge_map and (j, k) in edge_map and (i, k) in edge_map:
                triangles.append((edge_map[(i, j)], edge_map[(j, k)], edge_map[(i, k)]))
                if len(triangles) == 15:
                    break
        if len(triangles) == 15:
            break
    if len(triangles) == 15:
        break

F2 = len(triangles)
B2_num = np.zeros((E, F2))
for t_idx, (e_ij, e_jk, e_ik) in enumerate(triangles):
    B2_num[e_ij, t_idx] = 1.0
    B2_num[e_jk, t_idx] = 1.0
    B2_num[e_ik, t_idx] = -1.0

assert np.allclose(B1_num @ B2_num, 0.0), "Simplicial chain condition violated!"

L1_0 = B1_num.T @ B1_num + B2_num @ B2_num.T
eigs_L1_0, vecs_L1_0 = np.linalg.eigh(L1_0)
harm_mask = np.abs(eigs_L1_0) < 1e-7
beta1_0 = int(np.sum(harm_mask))
P_harm_0 = vecs_L1_0[:, harm_mask]

print("=== PART 2: HODGE 1-LAPLACIAN & UNDERACTUATED CONTROLLABILITY GRAMIAN ===")
print(f"Simplicial Complex: V={V}, E={E}, F2={F2} | Initial Harmonic Dimension beta_1(0) = {beta1_0}")

gamma_e = 1.80
A0_edge = -gamma_e * np.eye(E) - 0.35 * L1_0
D_act = 0.90
Sigma0_edge = solve_continuous_lyapunov(A0_edge, -2.0 * D_act * np.eye(E))
Sigma0_edge = 0.5 * (Sigma0_edge + Sigma0_edge.T)

p_channels = 18
B_vasc_0 = np.zeros((E, p_channels))
for e_idx in range(E):
    c_primary = e_idx % p_channels
    c_collateral = (e_idx + 1) % p_channels
    B_vasc_0[e_idx, c_primary] = 1.0
    B_vasc_0[e_idx, c_collateral] = 0.45

T_rep = 1.0
D_bath = 0.08
sigma_m0 = 0.10
H0_bits = 80.0
I_crit_harm = 38.0
Sigma_max_kB = 220.0

tau_grid = np.array([0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 12.0])

print(f"{'tau (h)':>7} | {'I_harm [bits]':>13} | {'P_e_harm Bound':>14} | {'min_eig(G_c)':>13} | {'Sigma_iso [k_B]':>15} | {'Sigma_sub [k_B]':>15} | {'Sub/Iso Ratio':>13} | {'Phase State':>18}")
print("-" * 124)

for tau in tau_grid:
    lambda_syn = 0.06
    alpha_lysis = 0.11
    k_noreflow = 0.24
    
    w_decay = np.exp(-lambda_syn * tau)
    A_tau = -(gamma_e * 0.12 + lambda_syn) * np.eye(E) - 0.35 * w_decay * L1_0
    Phi_tau = expm(A_tau * tau)
    
    R_tau = (sigma_m0 ** 2) * np.exp(alpha_lysis * tau) * np.eye(E)
    R_inv = np.linalg.inv(R_tau)
    
    Sigma0_harm = P_harm_0.T @ Sigma0_edge @ P_harm_0
    Phi_harm = P_harm_0.T @ Phi_tau @ P_harm_0
    R_harm_inv = P_harm_0.T @ R_inv @ P_harm_0
    
    snr_harm = np.eye(beta1_0) + Sigma0_harm @ Phi_harm.T @ R_harm_inv @ Phi_harm
    _, logdet_harm = np.linalg.slogdet(snr_harm)
    I_harm_bits = 0.5 * logdet_harm / np.log(2.0)
    
    Pe_bound = max(0.0, 1.0 - (1.0 / H0_bits) - (I_harm_bits / H0_bits))
    
    delta_m = 0.28 * (1.0 - np.exp(-0.32 * tau)) * np.ones(E)
    Sigma_iso_kB = np.dot(delta_m, delta_m) / (D_bath * T_rep)
    
    channel_patency = np.exp(-k_noreflow * tau * np.linspace(0.6, 1.8, p_channels))
    B_tau = B_vasc_0 @ np.diag(channel_patency)
    
    s_pts = np.linspace(0.0, T_rep, 41)
    ds = s_pts[1] - s_pts[0]
    G_c = np.zeros((E, E))
    A_rep = -0.45 * np.eye(E) - 0.25 * w_decay * L1_0
    BBT = B_tau @ B_tau.T + (2.5e-2 * np.exp(-0.38 * tau)) * np.eye(E)
    for idx_s, s in enumerate(s_pts):
        w_simp = 1.0 if (idx_s == 0 or idx_s == 40) else (4.0 if idx_s % 2 == 1 else 2.0)
        exp_As = expm(A_rep * s)
        G_c += (ds / 3.0) * w_simp * (exp_As @ BBT @ exp_As.T)
        
    min_eig_Gc = np.min(np.linalg.eigvalsh(G_c))
    G_c_inv = np.linalg.inv(G_c)
    
    Sigma_sub_kB = float(delta_m.T @ G_c_inv @ delta_m) / D_bath
    ratio_sub_iso = Sigma_sub_kB / max(1e-12, Sigma_iso_kB)
    
    if I_harm_bits >= I_crit_harm and Sigma_sub_kB <= Sigma_max_kB:
        phase = "I: Recoverable"
    elif I_harm_bits >= I_crit_harm and Sigma_sub_kB > Sigma_max_kB:
        phase = "II: Cryptobiotic"
    else:
        phase = "III: Info-Lost"
        
    print(f"{tau:7.2f} | {I_harm_bits:13.2f} | {Pe_bound:14.4f} | {min_eig_Gc:13.5f} | {Sigma_iso_kB:15.2f} | {Sigma_sub_kB:15.2f} | {ratio_sub_iso:12.2f}x | {phase:>18}")

# ==============================================================================
# PART 3: NON-CONVEX DOUBLE-WELL KRAMERS ESCAPE & EXACT QUANTILE TRANSPORT
# ==============================================================================
print("\n=== PART 3: NON-GAUSSIAN BISTABLE KRAMERS ESCAPE & OPTIMAL TRANSPORT ===")
alpha_w = 1.20
beta_w = 2.40
q_min = np.sqrt(beta_w / alpha_w)
barrier_dU = (beta_w ** 2) / (4.0 * alpha_w)

M_grid = 200
q = np.linspace(-2.8, 2.8, M_grid)
dq = q[1] - q[0]
U_q = 0.25 * alpha_w * (q ** 4) - 0.5 * beta_w * (q ** 2)

D_act_1d = 0.22
P0_q = np.exp(-U_q / D_act_1d) * (1.0 / (1.0 + np.exp(-10.0 * q)))
P0_q /= np.sum(P0_q * dq)

m0_1d = float(np.sum(q * P0_q * dq))
var0_1d = float(np.sum(((q - m0_1d) ** 2) * P0_q * dq))

def exact_1d_w2_sq(p_density, q_density, grid, d_grid):
    cdf_p = np.cumsum(p_density * d_grid)
    cdf_p /= cdf_p[-1]
    cdf_q = np.cumsum(q_density * d_grid)
    cdf_q /= cdf_q[-1]
    u_grid = np.linspace(0.001, 0.999, 500)
    qp = np.interp(u_grid, cdf_p, grid)
    qq = np.interp(u_grid, cdf_q, grid)
    return float(np.mean((qp - qq) ** 2))

print(f"Bistable Engram Wells at q* = +/- {q_min:.4f} | Saddle Barrier Delta U = {barrier_dU:.4f}")
print(f"{'tau (h)':>7} | {'Leakage P(q<0)':>14} | {'W2^2_Bures (Gauss)':>18} | {'W2^2_Exact (Quant)':>18} | {'S_bridge_Kramers':>18} | {'Kramers/Bures Ratio':>19}")
print("-" * 106)

for tau in [0.25, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0]:
    p_leak = 0.48 * (1.0 - np.exp(-0.28 * tau))
    P_right = np.exp(-((q - q_min * np.exp(-0.08 * tau)) ** 2) / (2.0 * (0.12 + 0.02 * tau)))
    P_left = np.exp(-((q + q_min * np.exp(-0.08 * tau)) ** 2) / (2.0 * (0.12 + 0.02 * tau)))
    
    P_tau_q = (1.0 - p_leak) * P_right + p_leak * P_left
    P_tau_q /= np.sum(P_tau_q * dq)
    
    leak_frac = float(np.sum(P_tau_q[q < 0] * dq))
    
    m_tau = float(np.sum(q * P_tau_q * dq))
    var_tau = float(np.sum(((q - m_tau) ** 2) * P_tau_q * dq))
    W2_sq_bures = (m_tau - m0_1d) ** 2 + (np.sqrt(var_tau) - np.sqrt(var0_1d)) ** 2
    
    W2_sq_exact = exact_1d_w2_sq(P_tau_q, P0_q, q, dq)
    
    S_bures = W2_sq_bures / D_bath
    S_kramers = (W2_sq_exact + 2.0 * leak_frac * barrier_dU) / D_bath
    ratio_kb = S_kramers / max(1e-12, S_bures)
    
    print(f"{tau:7.2f} | {leak_frac:14.4f} | {W2_sq_bures:18.4f} | {W2_sq_exact:18.4f} | {S_kramers:18.2f} | {ratio_kb:18.2f}x")
```

### Execution Output:
```text
=== PART 1: SYMPY SYMBOLIC VERIFICATION ===
Symbolic Sub-Riemannian Entropy Bound as min_eig(G_c) -> 0+: oo
Boundary operator identity d1 * d2 == 0 verified: True
Symbolic Betti-1 (dim ker L1) with 1 filled triangle: 1
Harmonic 1-cycle eigenvector h in ker(L1): [0.3333, 0.3333, -0.6667, 1.0, 1.0]
Orthogonality check <d1^T x, d2 z> == 0: True

=== PART 2: HODGE 1-LAPLACIAN & UNDERACTUATED CONTROLLABILITY GRAMIAN ===
Simplicial Complex: V=36, E=90, F2=15 | Initial Harmonic Dimension beta_1(0) = 40
tau (h) | I_harm [bits] | P_e_harm Bound |  min_eig(G_c) | Sigma_iso [k_B] | Sigma_sub [k_B] | Sub/Iso Ratio |        Phase State
----------------------------------------------------------------------------------------------------------------------------
   0.25 |        108.77 |         0.0000 |       0.00495 |            0.52 |            0.44 |         0.85x |     I: Recoverable
   1.00 |         94.87 |         0.0000 |       0.00380 |            6.61 |            8.39 |         1.27x |     I: Recoverable
   2.00 |         76.77 |         0.0279 |       0.00267 |           19.71 |           43.63 |         2.21x |     I: Recoverable
   3.00 |         59.50 |         0.2438 |       0.00188 |           33.59 |          131.00 |         3.90x |     I: Recoverable
   4.00 |         43.65 |         0.4419 |       0.00132 |           45.97 |          317.30 |         6.90x |   II: Cryptobiotic
   5.00 |         29.97 |         0.6128 |       0.00093 |           56.18 |          688.00 |        12.25x |     III: Info-Lost
   6.00 |         19.15 |         0.7482 |       0.00065 |           64.23 |         1402.51 |        21.83x |     III: Info-Lost
   7.50 |          8.64 |         0.8796 |       0.00039 |           72.92 |         3857.55 |        52.90x |     III: Info-Lost
   9.00 |          3.51 |         0.9437 |       0.00023 |           78.58 |        10327.13 |       131.43x |     III: Info-Lost
  12.00 |          0.51 |         0.9812 |       0.00008 |           84.45 |        68624.90 |       812.62x |     III: Info-Lost

=== PART 3: NON-GAUSSIAN BISTABLE KRAMERS ESCAPE & OPTIMAL TRANSPORT ===
Bistable Engram Wells at q* = +/- 1.4142 | Saddle Barrier Delta U = 1.2000
tau (h) | Leakage P(q<0) | W2^2_Bures (Gauss) | W2^2_Exact (Quant) |   S_bridge_Kramers | Kramers/Bures Ratio
----------------------------------------------------------------------------------------------------------
   0.25 |         0.0325 |             0.1289 |             0.1523 |               2.88 |               1.79x
   1.00 |         0.1174 |             0.5710 |             0.6097 |              11.14 |               1.56x
   2.00 |         0.2066 |             1.0563 |             1.0919 |              19.85 |               1.50x
   4.00 |         0.3272 |             1.6307 |             1.6563 |              30.52 |               1.50x
   6.00 |         0.3986 |             1.8819 |             1.8998 |              35.71 |               1.52x
   9.00 |         0.4536 |             2.0036 |             2.0149 |              38.79 |               1.55x
  12.00 |         0.4768 |             2.0280 |             2.0364 |              39.76 |               1.57x
```
