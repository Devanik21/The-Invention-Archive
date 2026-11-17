"""
RAF-masked exploration (superseded by final_test.py). Imports pct1 for shared functions.
Note: pct1 uses length-L organisms, which is the artifact described in RESULT_NOTE.
"""
import random, numpy as np
import pct1 as M

def reachable_mask(cat):
    Mset = M.closure(set(range(len(M.REACTIONS))))
    return np.array([(a in Mset and b in Mset) for (a, b) in M.REACTIONS])

def raf_mask(cat):
    R = set(range(len(M.REACTIONS)))
    while True:
        Mset = M.closure(R)
        Rn = {i for i in R if M.REACTIONS[i][0] in Mset and M.REACTIONS[i][1] in Mset
              and any(cat[s][i] for s in Mset if s in cat)}
        if Rn == R: break
        R = Rn
    mask = np.zeros(len(M.REACTIONS), dtype=bool)
    for i in R: mask[i] = True
    return mask

def fchan(cat, eps, g, mask):
    C = np.linalg.matrix_power(M.copy_channel(eps, M.L), g)
    rows = {}
    for j, y in enumerate(M.ORGS):
        key = tuple((cat[y] & mask).astype(int))
        rows.setdefault(key, []).append(j)
    keys = list(rows)
    W = np.zeros((len(M.ORGS), len(keys)))
    for k, key in enumerate(keys):
        W[:, k] = C[:, rows[key]].sum(axis=1)
    return W
