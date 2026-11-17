# PCT-I: Hereditary Catalytic Capacity - Pre-registered Test Result

Status: negative result. Minimal in-silico model only. Not peer reviewed.

## Question
Can a chemical network be RAF (catalytically self-maintaining) without carrying heritable catalytic information, and does that heredity persist over generations?

## Model (final)
- Polymers: binary strings, length 1-4 (length-1 strings are food). 30 polymers, 68 ligation reactions.
- Organisms: all polymers. Replication: each symbol flips with probability eps; length preserved.
- Operational function: catalysed reactions restricted to the maximal RAF (empty if no RAF).
- Channel: X -> phi(copy^g(X)); capacity computed exactly by Blahut-Arimoto (bits).
- Generators: A = Bernoulli(p) catalysis; B = held-out, each reaction gets k catalysts (k matched to p).
- Grid: p in {0.05, 0.08, 0.12}, eps in {0.01, 0.05, 0.10}; 300 networks/cell (test 1), 500/cell (test 2).

## Test 1 - one-generation heredity (pre-registered)
Heredity present iff C1 > 0.10 bits. Distinction supported only if RAF networks without heredity make up a genuine fraction.
- P(heredity | RAF) = 1.000 in all 18 cells (A and B).
- Zero-capacity RAF networks: 0 genuine (earlier zeros were a length-restriction artifact).
- Verdict: not supported.

## Test 2 - multigenerational persistence (pre-registered)
Loss event: C1 > 0.10 bits and C3 <= 0.10 bits. Supported if loss >= 10% in >= 6 of 9 cells, for both generators.
- Loss events: 0 in all 18 cells.
- Capacity decays with generations (example, p=0.05, eps=0.05, median): C1 = 3.82, C2 = 3.30, C3 = 2.95 bits.
- Verdict: not supported. Persistence is graded decay, not loss within three generations.

Decoupled control (F independent of parent): capacity = 0.000000 bits in both tests.

## Interpretation
In this minimal model, operational heredity coincides with RAF membership, and it decays gradually with copying error rather than vanishing. The separation that motivated PCT-I is not present here.

## Limitations
- Narrow model: binary alphabet, length <= 4, 68 reactions, one fixed copying scheme.
- Loss threshold (0.10 bits) was chosen in advance and is arbitrary; a different threshold changes the loss count but not the decay pattern.
- Test 2 used 500 networks per cell; test 1 used 300.
- RAF necessity is built into the masked function, so it is a definition, not a finding.

## Reproduce
python3 code/final_test.py 300
python3 code/persistence.py
