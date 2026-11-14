import numpy as np
from scipy.linalg import expm

# Exact chemical-master-equation (CME) channel for n-site PTM systems.
# Input s in {0,1}^n selects kinase-driven sites. On-rate of site i: alpha*kappa*s_i*(1+c*sum_{j!=i} z_j).
# Off-rate of a modified site: beta*kappa. Output: full proteoform state z (bitmask).


def generator(n, s, alpha, beta, kappa, c):
    N = 2 ** n
    Q = np.zeros((N, N))
    for z in range(N):
        for i in range(n):
            bit = (z >> i) & 1
            if bit == 0:
                nb = sum(((z >> j) & 1) for j in range(n) if j != i)
                r = alpha * kappa * s[i] * (1.0 + c * nb)
                zt = z | (1 << i)
            else:
                r = beta * kappa
                zt = z & ~(1 << i)
            if r > 0:
                Q[zt, z] += r
                Q[z, z] -= r
    return Q


def channel_matrix(n, inputs, alpha, beta, kappa, c, t):
    """W[x, z] = P(Z(t) = z | input x), starting from the fully unmodified state."""
    N = 2 ** n
    W = np.zeros((len(inputs), N))
    p0 = np.zeros(N)
    p0[0] = 1.0
    for k, s in enumerate(inputs):
        W[k] = expm(generator(n, s, alpha, beta, kappa, c) * t) @ p0
    return W


def expected_cycles(n, s, alpha, beta, kappa, c, t, steps=400):
    """Expected number of modification/removal events in [0, t]."""
    N = 2 ** n
    grid = np.linspace(0.0, t, steps + 1)
    Q = generator(n, s, alpha, beta, kappa, c)
    exit_rate = -np.diag(Q)
    p0 = np.zeros(N)
    p0[0] = 1.0
    vals = [float(exit_rate @ (expm(Q * tau) @ p0)) for tau in grid]
    return float(np.trapezoid(vals, grid))


def blahut_arimoto(W, iters=3000, tol=1e-11):
    """Capacity (bits) of discrete channel W[x, y] by Blahut-Arimoto."""
    M = W.shape[0]
    p = np.ones(M) / M
    eps = 1e-300
    for _ in range(iters):
        q = p @ W + eps
        D = np.sum(W * np.log(W / q[None, :] + eps), axis=1)
        p_new = p * np.exp(D)
        p_new /= p_new.sum()
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    q = p @ W + eps
    C = np.sum(p[:, None] * W * np.log2((W + eps) / q[None, :]))
    return float(C), p


def all_inputs(n):
    return [np.array([(x >> i) & 1 for i in range(n)], float) for x in range(2 ** n)]


def map_error(W, codebook_idx):
    """MAP error probability under uniform prior over the codebook (input indices)."""
    sub = W[codebook_idx]
    M = len(codebook_idx)
    best = np.max(sub, axis=0)
    return float(1.0 - np.sum(best) / M)


def min_distance(words):
    d = [bin(a ^ b).count("1") for i, a in enumerate(words) for b in words[i + 1:]]
    return min(d) if d else np.inf
