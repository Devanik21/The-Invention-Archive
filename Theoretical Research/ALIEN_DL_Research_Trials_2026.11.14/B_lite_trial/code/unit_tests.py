"""Mathematical unit tests for the ALIEN-DL lite implementation."""
import numpy as onp
import alien as A

rng = onp.random.default_rng(0)

# T1: budget conservation: sum_i pi_ij = d for every hidden unit, before and after an exponentiated-gradient step
d, H = 6, 4
logits = rng.normal(0, 1, (d, H))
pi = A.shares(logits)
assert onp.allclose(pi.sum(axis=0), d), "initial budget not conserved"
g = rng.normal(0, 1, (d, H))
logits2 = logits - 0.7 * g
pi2 = A.shares(logits2)
assert onp.allclose(pi2.sum(axis=0), d), "budget not conserved after update"
print("T1 budget conservation: PASS  (max |sum-d| =", float(onp.max(onp.abs(pi2.sum(axis=0) - d))), ")")

# T2: positivity and boundedness of shares
assert onp.all(pi2 > 0)
print("T2 positivity of shares: PASS")

# T3: reduction to a linear weighted-sum network (Proposition 3) when recovery is saturated and facilitation off
p = A.init(d, H, 3, rng)
A.RHO, A.ALPHA_R, A.ALPHA_U = 1e3, 0.0, 0.0
x = rng.uniform(0, 1, (5, d))
pi_u = onp.ones((d, H))
hidden = A.forward_hidden(p, pi_u, x, R=1)
expected_drive = A.T_STEPS * (x @ (p["theta"] * A.U0)) / d
onp.testing.assert_allclose(hidden, onp.tanh(expected_drive + p["b1"]), rtol=1e-6, atol=1e-8)
print("T3 reduction to weighted-sum network: PASS")
A.RHO, A.ALPHA_R, A.ALPHA_U = 0.3, 0.5, 0.3
