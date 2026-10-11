# X thread — The Identifiability Budget

**1/12** Scientific AI can predict data accurately without identifying the mechanism that generated them. Prediction and mechanism identification are different claims—and they need different evidence.

**2/12** Suppose two candidate mechanisms produce the same distribution under every observed condition. No model architecture can distinguish them from those observations alone. With equal prior odds, best possible correctness remains 50%, even with more samples from the same regime.

**3/12** The practical question is: what experiment would make the competing mechanisms disagree, how much information would it provide, and what would it cost? That is an experimental-design problem, not a prompting problem.

**4/12** I formalize a finite-candidate “identifiability budget” for independent Gaussian observations with known, mechanism-invariant covariance. Its assumptions are deliberately explicit.

**5/12** For candidate means μᵢ(u), μⱼ(u) and covariance Σ, an experiment’s pairwise information is D = ½(μᵢ−μⱼ)ᵀΣ⁻¹(μᵢ−μⱼ). Across independent measurements, these divergences add.

**6/12** Under that Gaussian model, equal-prior pairwise error is Φ(−√(D/2)). For K candidates, a union bound yields a conservative sufficient separation threshold for a declared model-selection error target.

**7/12** The budget minimizes experiment cost while meeting pairwise information thresholds. If no allowed intervention separates a decision-relevant pair, the budget is infinite: the right output is unresolved, not a confident story.

**8/12** Toy example: f₀(u)=u and f₁(u)=u². At u=0 and u=1, both models agree exactly. Repeating observations only there cannot identify the mechanism.

**9/12** At u=2, predictions are 2 vs 4. With unit Gaussian noise, each replicate adds D=2. Three repeats give D=6 and analytic pairwise error Φ(−√3)≈4.16%; two repeats remain above 5%.

**10/12** This is a mathematical toy—not evidence about biology or any real physical mechanism. A guarantee fails if candidate means, covariance, independence, or candidate-set assumptions are wrong.

**11/12** AI can propose candidate mechanisms, search interventions, and approximate expensive solvers. But the audit must validate surrogate predictions, account for uncertainty, disclose excluded candidates, and abstain when evidence cannot discriminate.

**12/12** The contribution is a transparent audit framework, not a claim to invent causal identifiability or active experimental design. Preprint, code, bibliography, and limitations: https://github.com/Devanik21/The-Invention-Archive/tree/main/Theoretical%20Research/The_Identifiability_Budget_An_Audit_Framework_for_Mechanistic_AI_and_Experimental_Design
