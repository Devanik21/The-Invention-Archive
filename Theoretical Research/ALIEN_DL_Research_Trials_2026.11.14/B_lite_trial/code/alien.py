import autograd.numpy as np
from autograd import grad

RHO, ALPHA_R, ALPHA_U, U0, TAU_U, DT, T_STEPS = 0.3, 0.5, 0.3, 0.3, 3.0, 1.0, 5


def init(d, H, K, rng):
    p = {"theta": rng.normal(0, 0.5, (d, H)), "b1": np.zeros(H),
         "W2": rng.normal(0, 0.3, (H, K)), "b2": np.zeros(K)}
    return p


def shares(logits):
    """pi_ij = d * softmax over synapses i within each hidden unit j (sums to d per unit)."""
    z = logits - np.max(logits, axis=0, keepdims=True)
    e = np.exp(z)
    return logits.shape[0] * e / np.sum(e, axis=0, keepdims=True)


def forward_hidden(p, pi, x, R=1):
    """x: (B, d) with values in [0,1]. Returns hidden activations (B, H)."""
    B, d = x.shape
    H = p["theta"].shape[1]
    pi_eff = pi if R else np.ones_like(pi)
    pre = x[:, :, None]
    r = np.ones((B, d, H)) * 1.0
    u = np.ones((B, d, H)) * U0
    acc = np.zeros((B, d, H))
    for _ in range(T_STEPS):
        rec = RHO * pi_eff[None] * (1.0 - r) - ALPHA_R * u * r * pre
        r = np.minimum(np.maximum(r + DT * rec, 0.0), 1.0)
        du = (U0 - u) / TAU_U + ALPHA_U * (1.0 - u) * pre
        u = np.minimum(np.maximum(u + DT * du, 0.0), 1.0)
        acc = acc + p["theta"][None] * u * r * pre
    drive = np.sum(acc, axis=1) / d
    return np.tanh(drive + p["b1"])


def logits_fn(p, pi, x, R=1):
    h = forward_hidden(p, pi, x, R)
    return h @ p["W2"] + p["b2"]


def loss_fn(p, pi, x, y_onehot, R=1, l2=1e-4):
    z = logits_fn(p, pi, x, R)
    z = z - np.max(z, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    ce = -np.mean(np.sum(y_onehot * logp, axis=1))
    return ce + l2 * np.sum(p["theta"] ** 2)


grad_params = grad(loss_fn, 0)
grad_pi = grad(loss_fn, 1)


def predict(p, pi, x, R=1):
    return np.argmax(logits_fn(p, pi, x, R), axis=1)
