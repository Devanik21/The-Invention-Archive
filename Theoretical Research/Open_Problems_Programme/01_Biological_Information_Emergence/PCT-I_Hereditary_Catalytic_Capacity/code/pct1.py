"""
PCT-I minimal model (first version, unmasked function). Kept for history.
- Polymers: binary strings of length 1..L (food = length-1 strings, not catalytic).
- Reactions: ligation (a,b) -> a+b when |a+b| <= L. Forward only.
- Catalysis: cat[s][r] ~ Bernoulli(p) for non-food polymers s and reactions r.
- RAF detection: standard fixed-point algorithm (Hordijk-Steel style).
- Organisms: length-L strings x. Offspring copy: each symbol flips w.p. eps.
- Offspring catalytic function: F = phi(y) = full catalysis row of y.
- NOTE: organisms restricted to length L. This produced an artifact (see RESULT_NOTE). Superseded by final_test.py.
"""
import itertools, math, random
import numpy as np

L = 4
FOOD = ['0', '1']

def all_polymers(L):
    out = []
    for n in range(1, L + 1):
        out += [''.join(t) for t in itertools.product('01', repeat=n)]
    return out

POLY = all_polymers(L)
NONFOOD = [s for s in POLY if s not in FOOD]
ORGS = [''.join(t) for t in itertools.product('01', repeat=L)]

REACTIONS = [(a, b) for a in POLY for b in POLY if len(a) + len(b) <= L]
R_IDX = {r: i for i, r in enumerate(REACTIONS)}

def sample_network(p, rng):
    cat = {s: np.array([rng.random() < p for _ in REACTIONS]) for s in NONFOOD}
    return cat

def closure(reactions):
    M = set(FOOD)
    changed = True
    while changed:
        changed = False
        for i in reactions:
            a, b = REACTIONS[i]
            if a in M and b in M and (a + b) not in M:
                M.add(a + b); changed = True
    return M

def has_raf(cat):
    R = set(range(len(REACTIONS)))
    while True:
        M = closure(R)
        Rn = set()
        for i in R:
            a, b = REACTIONS[i]
            if a in M and b in M and any(cat[s][i] for s in M if s in cat):
                Rn.add(i)
        if Rn == R:
            break
        R = Rn
    return len(R) > 0

def copy_channel(eps, L):
    n = len(ORGS)
    C = np.zeros((n, n))
    for i, x in enumerate(ORGS):
        for j, y in enumerate(ORGS):
            d = sum(a != b for a, b in zip(x, y))
            C[i, j] = eps**d * (1 - eps)**(L - d)
    return C

def blahut_arimoto(W, iters=2000, tol=1e-12):
    n = W.shape[0]
    p = np.ones(n) / n
    for _ in range(iters):
        q = p @ W
        with np.errstate(divide='ignore', invalid='ignore'):
            logratio = np.where(W > 0, np.log(W / q), 0.0)
        D = (W * logratio).sum(axis=1)
        p_new = p * np.exp(D)
        p_new /= p_new.sum()
        if np.abs(p_new - p).max() < tol:
            p = p_new; break
        p = p_new
    q = p @ W
    with np.errstate(divide='ignore', invalid='ignore'):
        logratio = np.where(W > 0, np.log(W / q), 0.0)
    return float((p * (W * logratio).sum(axis=1)).sum() / math.log(2))

def function_channel(cat, eps, g):
    C = np.linalg.matrix_power(copy_channel(eps, L), g)
    rows = {}
    for j, y in enumerate(ORGS):
        key = tuple(cat[y].astype(int)) if y in cat else None
        rows.setdefault(key, []).append(j)
    keys = list(rows)
    W = np.zeros((len(ORGS), len(keys)))
    for k, key in enumerate(keys):
        W[:, k] = C[:, rows[key]].sum(axis=1)
    return W

def capacity_g(cat, eps, g):
    return blahut_arimoto(function_channel(cat, eps, g))
