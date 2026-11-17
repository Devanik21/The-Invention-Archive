"""
PCT-I final test (pre-specified).
Changes from the draft model:
 - organisms = ALL polymers (lengths 1..4), copies preserve length
 - operational function F(y) = catalysed reactions restricted to the RAF
 - generator A (training-style): Bernoulli(p) catalysis per (polymer, reaction)
 - generator B (held-out): each reaction gets exactly k random catalysts, k matched to density p
 - controls: decoupled channel (F independent of parent), capacity must be 0
Pre-specified criteria (decided BEFORE running):
 H = heredity present iff C1 > 0.10 bits.
 K1 (fail heredity claim): if P(H | RAF) < 0.5 in generator A OR the decoupled control > 0.
 K2 (distinction holds): RAF networks with C1 <= 0.10 bits make up >= 10% of RAF networks
     in BOTH generators, AND the zero cases are NOT explained by the artifact
     (artifact := no organism catalyses any RAF reaction).
 K3 (distinction fails): artifact explains >= 90% of zero cases, or P(H|RAF) > 0.9 with no genuine zeros.
"""
import itertools, math, random, sys
import numpy as np

L = 4
FOOD = ['0', '1']
def polys(L):
    out = []
    for n in range(1, L+1):
        out += [''.join(t) for t in itertools.product('01', repeat=n)]
    return out
POLY = polys(L)
NONFOOD = [s for s in POLY if s not in FOOD]
REACTIONS = [(a, b) for a in POLY for b in POLY if len(a)+len(b) <= L]
NR = len(REACTIONS)
PI = {s: i for i, s in enumerate(POLY)}
THRESH = 0.10

def copy_matrix(eps):
    n = len(POLY); C = np.zeros((n, n))
    for i, x in enumerate(POLY):
        for j, y in enumerate(POLY):
            if len(x) != len(y): continue
            d = sum(a != b for a, b in zip(x, y))
            C[i, j] = eps**d * (1-eps)**(len(x)-d)
    return C

def closure(R):
    M = set(FOOD); ch = True
    while ch:
        ch = False
        for i in R:
            a, b = REACTIONS[i]
            if a in M and b in M and a+b not in M:
                M.add(a+b); ch = True
    return M

def raf_reactions(cat):
    """cat: bool matrix [poly, reaction]. Returns set of reactions in the maximal RAF (may be empty)."""
    R = set(range(NR))
    while True:
        M = closure(R)
        Rn = {i for i in R if REACTIONS[i][0] in M and REACTIONS[i][1] in M
              and any(cat[PI[s], i] for s in M if s in PI and s not in FOOD)}
        if Rn == R: break
        R = Rn
    return R

def gen_A(p, rng):
    cat = np.zeros((len(POLY), NR), dtype=bool)
    for s in NONFOOD:
        for i in range(NR):
            cat[PI[s], i] = rng.random() < p
    return cat

def gen_B(p, rng):
    cat = np.zeros((len(POLY), NR), dtype=bool)
    k = max(1, round(p*len(NONFOOD)))
    for i in range(NR):
        for s in rng.sample(NONFOOD, k):
            cat[PI[s], i] = True
    return cat

def blahut_arimoto(W, iters=3000, tol=1e-10):
    n = W.shape[0]; p = np.ones(n)/n
    for _ in range(iters):
        q = p @ W
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.where(W > 0, np.log(W/np.where(q>0,q,1)), 0.0)
        D = (W*lr).sum(axis=1)
        pn = p*np.exp(D); pn /= pn.sum()
        if np.abs(pn-p).max() < tol: p = pn; break
        p = pn
    q = p @ W
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.where(W > 0, np.log(W/np.where(q>0,q,1)), 0.0)
    return float((p*(W*lr).sum(axis=1)).sum()/math.log(2))

def function_channel(cat, Cg, mask_reactions):
    mask = np.zeros(NR, dtype=bool)
    for i in mask_reactions: mask[i] = True
    rows = {}
    for j in range(len(POLY)):
        key = tuple((cat[j] & mask).astype(np.uint8))
        rows.setdefault(key, []).append(j)
    W = np.zeros((len(POLY), len(rows)))
    for k, idx in enumerate(rows.values()):
        W[:, k] = Cg[:, idx].sum(axis=1)
    return W

def run_cell(gen, p, eps, n, rng):
    C1 = copy_matrix(eps)
    res = []
    for _ in range(n):
        cat = gen(p, rng)
        R = raf_reactions(cat)
        hasraf = len(R) > 0
        c1 = blahut_arimoto(function_channel(cat, C1, R))
        org_cat = any(cat[PI[s], i] for s in POLY for i in R)
        res.append((hasraf, c1, org_cat))
    return res

def summarize(label, res):
    raf = [r for r in res if r[0]]
    H = [r for r in raf if r[1] > THRESH]
    zeros = [r for r in raf if r[1] <= THRESH]
    art = [r for r in zeros if not r[2]]
    gen_zeros = [r for r in zeros if r[2]]
    pH = len(H)/len(raf) if raf else float('nan')
    print(f"  {label}: nets={len(res)} P(RAF)={len(raf)/len(res):.3f} "
          f"P(H|RAF)={pH:.3f} zeros|RAF={len(zeros)/max(1,len(raf)):.3f} "
          f"artifact_zeros={len(art)} genuine_zeros={len(gen_zeros)} "
          f"C1|RAF median={np.median([r[1] for r in raf]) if raf else float('nan'):.3f}")
    return dict(pH=pH, zero_frac=len(zeros)/max(1,len(raf)), art=len(art), gen=len(gen_zeros), n_raf=len(raf))

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = random.Random(2026)
    cells = [(p, eps) for p in [0.05, 0.08, 0.12] for eps in [0.01, 0.05, 0.10]]
    print(f"polymers={len(POLY)} reactions={NR} networks/cell={N}  threshold={THRESH} bits")
    print("== Generator A (training-style) ==")
    A = {}
    for p, eps in cells:
        res = run_cell(gen_A, p, eps, N, rng)
        print(f" p={p} eps={eps}")
        A[(p, eps)] = summarize("A", res)
    print("== Generator B (held-out, reaction-centric) ==")
    B = {}
    for p, eps in cells:
        res = run_cell(gen_B, p, eps, N, rng)
        print(f" p={p} eps={eps}")
        B[(p, eps)] = summarize("B", res)
    print("== Decoupled control (F independent of parent): capacity ==")
    W = np.ones((len(POLY), 3))/3
    print(f"  capacity = {blahut_arimoto(W):.6f} bits (must be 0)")
