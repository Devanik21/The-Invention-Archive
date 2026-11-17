"""
PCT-I persistence test (pre-registered BEFORE running).
Question: do RAF networks with one-generation heredity (C1 > 0.10 bits) lose it over generations?
Measure: C_g = capacity of X -> phi(copy^g(X)), operational function (RAF-masked), g = 2, 3.
Criteria (fixed in advance):
 - Loss event for a network: C1 > 0.10 bits AND C3 <= 0.10 bits.
 - SEPARATION SUPPORTED: loss events are >= 10% of RAF networks with C1 > 0.10, in BOTH generator A
   and held-out generator B, in at least 6 of 9 cells, and the decoupled control stays at 0.
 - NOT SUPPORTED otherwise.
Requires final_test.py in the same folder.
"""
import random, numpy as np
import final_test as F

def run(gen, p, eps, n, rng):
    Cm = {g: np.linalg.matrix_power(F.copy_matrix(eps), g) for g in (1, 2, 3)}
    out = []
    for _ in range(n):
        cat = gen(p, rng)
        R = F.raf_reactions(cat)
        if not R:
            continue
        c = {g: F.blahut_arimoto(F.function_channel(cat, Cm[g], R)) for g in (1, 2, 3)}
        out.append(c)
    return out

if __name__ == '__main__':
    N = 500
    rng = random.Random(4242)
    cells = [(p, e) for p in [0.05, 0.08, 0.12] for e in [0.01, 0.05, 0.10]]
    summary = {}
    for name, gen in [('A', F.gen_A), ('B', F.gen_B)]:
        print(f"== Generator {name} ==")
        ok_cells = 0
        for p, e in cells:
            res = run(gen, p, e, N, rng)
            h = [r for r in res if r[1] > F.THRESH]
            loss = [r for r in h if r[3] <= F.THRESH]
            frac = len(loss)/len(h) if h else float('nan')
            med = {g: np.median([r[g] for r in res]) for g in (1,2,3)} if res else {}
            print(f" p={p} eps={e}: RAF={len(res)} H1={len(h)} loss(C1>0.1,C3<=0.1)={len(loss)} "
                  f"frac_loss={frac:.3f}  median C1/C2/C3={med[1]:.3f}/{med[2]:.3f}/{med[3]:.3f}")
            if h and frac >= 0.10: ok_cells += 1
        summary[name] = ok_cells
    print(f"Cells with loss fraction >= 10%: A={summary['A']}/9  B={summary['B']}/9")
    verdict = "SUPPORTED" if summary['A'] >= 6 and summary['B'] >= 6 else "NOT SUPPORTED"
    print(f"PRE-REGISTERED VERDICT: persistence distinction {verdict}")
    print(f"Decoupled control capacity: {F.blahut_arimoto(np.ones((30,3))/3):.6f} bits")
