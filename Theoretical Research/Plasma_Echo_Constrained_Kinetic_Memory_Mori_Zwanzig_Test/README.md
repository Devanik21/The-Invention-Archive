# Plasma-Echo-Constrained Kinetic Memory: A Mori–Zwanzig Memory Test for Separating Reversible Phase Mixing from Collisional Loss

**Release 2026.11.10 · Preprint v1.0 · Devanik Debnath and Xylia**

Not peer reviewed. A numerical study of a one-dimensional, one-velocity Vlasov–Poisson model in normalised units. No laboratory, space or observational data are analysed. No DOI or Zenodo deposit is claimed.

## Author

Electronics and Communication Engineering graduate and independent AI researcher exploring how physical and mathematical structure can improve learning and modeling. Work spans gamma-ray burst reconstruction at the National Astronomical Observatory of Japan (NAOJ), physics-inspired signal classification, reinforcement learning, neural quantum-state methods, causal emergence, morphogenetic neural architectures, and computational artificial life. The research style combines algorithm design with explicit assumptions, code-based checks, and disclosure of unresolved limitations.

Devanik Debnath is a Scientific Machine Learning and AI Architectures researcher; B.Tech, Electronics and Communication Engineering, National Institute of Technology Agartala (2026); Samsung Grade I Fellowship, Cohort 7. Links: github.com/Devanik21, orcid.org/0009-0001-8310-6921.

## What the study tests

1. Whether a semi-Lagrangian Vlasov–Poisson solver reproduces the Landau damping benchmark (it does: fitted rate 0.1545 against 0.1533, relative error 0.8%; invariant C2 drift at most 0.1%).
2. Whether a two-pulse density echo appears and how its amplitude scales with pulse strength (fitted log-log slope 1.72 against 2, consistent with a second-order effect).
3. Whether weak BGK collisions suppress the echo monotonically (echo amplitude falls to 0.58 of the collisionless value at collision rate 0.1, with timing nearly unchanged).
4. Whether a delay-embedding memory model predicts held-out echo-window behaviour better than a memoryless closure (large one-step error reduction at all three cadences; one-step only).

## Caveats stated in the paper

- The echo timing is ambiguous. The difference signal has several maxima from beat structure. The reported time is the largest maximum, about two time units after 2 tau, and is not a validated echo time.
- The memory result is for one-step prediction. Multi-step rollout, nonlinear neural operators and an observational MMS test were not carried out.
- The collisional suppression is a property of the BGK model. It is not shown to be thermodynamic irreversibility, and no grid-convergence study at fixed collision rate was done.
- External literature mentioned in the research note was not verified.

## Files

- paper/Plasma_Echo_Constrained_Kinetic_Memory.pdf (11 pages; in the release bundle)
- code/vlasov.py, code/echo_experiments.py, code/memory_experiment.py, code/make_figures.py, code/build_paper.py (in the release bundle)
- results/ and figures/ (in the release bundle)
- CITATION.cff, .zenodo.json, requirements.txt (this folder)

## Reproduce

```bash
pip install -r requirements.txt
python code/echo_experiments.py
python code/memory_experiment.py
python code/make_figures.py
python code/build_paper.py
```

No licence file is included. The authors should choose one before reuse.
