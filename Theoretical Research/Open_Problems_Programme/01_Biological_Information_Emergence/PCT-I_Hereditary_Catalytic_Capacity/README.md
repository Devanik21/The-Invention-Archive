# PCT-I: Hereditary Catalytic Capacity

Status: tested (negative). Result note: [RESULT_NOTE.md](RESULT_NOTE.md).

Question. Can a catalytic network be RAF (catalytically self-maintaining) without carrying heritable catalytic information, and does that heredity persist over generations?

Channel. Parental sequence X is copied with per-symbol error eps to give offspring Y; the offspring's operational function is phi(Y), the set of reactions catalysed by Y restricted to the maximal RAF. The capacity C_g = max over input distributions of I(X; phi(copy^g(X))), computed exactly with the Blahut-Arimoto algorithm, in bits.

Tests and outcomes (all pre-registered, see the code docstrings):

| Test | Criterion | Outcome |
|---|---|---|
| 1. One-generation heredity | C1 > 0.10 bits for RAF networks; separation requires genuine zeros | P(heredity given RAF) = 1.000 in all 18 cells across two generators. Not supported. |
| 2. Multigenerational persistence | Loss = C1 > 0.10 and C3 <= 0.10, in at least 10% of RAF networks in >= 6 of 9 cells | 0 loss events in 18 cells. Not supported. |
| Decoupled control | Function independent of parent | capacity = 0.000000 bits |

Correction history. An earlier version gave apparent zero-capacity RAF networks. These came from restricting organisms to length-4 strings, so some RAF catalysts were never carried. The corrected model (all polymers as organisms) removed them. The earlier result is kept in the history of this folder and is not a finding.

Limitations. Binary alphabet; polymers up to length 4; 68 reactions; a single copying scheme; a loss threshold chosen in advance. In the masked model, RAF necessity is built into the definition of the function, so it is not a finding.

Files. code/pct1.py (model and first unmasked run), code/variant.py (RAF-masked exploration), code/final_test.py (test 1), code/persistence.py (test 2). Reproduce with `python3 final_test.py 300` and `python3 persistence.py` (requires numpy).
