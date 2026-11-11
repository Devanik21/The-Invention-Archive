# Plasma-Echo-Constrained Kinetic Memory: A Mori–Zwanzig Memory Test for Separating Reversible Phase Mixing from Collisional Loss

**Release 2026.11.10 · Preprint v1.0 (12 pages) · Devanik Debnath and Xylia**

Not peer reviewed. A numerical study of a one-dimensional, one-velocity Vlasov–Poisson model in normalised units. No laboratory, space or observational data are analysed. No DOI or Zenodo deposit is claimed.

## Author

Electronics and Communication Engineering graduate and independent AI researcher exploring how physical and mathematical structure can improve learning and modeling. Work spans gamma-ray burst reconstruction at the National Astronomical Observatory of Japan (NAOJ), physics-inspired signal classification, reinforcement learning, neural quantum-state methods, causal emergence, morphogenetic neural architectures, and computational artificial life. The research style combines algorithm design with explicit assumptions, code-based checks, and disclosure of unresolved limitations.

Devanik Debnath is a Scientific Machine Learning and AI Architectures researcher; B.Tech, Electronics and Communication Engineering, National Institute of Technology Agartala (2026); Samsung Grade I Fellowship, Cohort 7. Links: github.com/Devanik21, orcid.org/0009-0001-8310-6921.

## Results

1. Landau damping benchmark reproduced: fitted rate 0.1545 against 0.1533 (relative error 0.8%); invariant C2 drift at most 0.1% over 40 time units.
2. Two-pulse density echo appears; amplitude scales with pulse amplitude with fitted log-log slope 1.72 (second-order expectation 2).
3. Weak BGK collisions suppress the echo monotonically: amplitude falls to 0.58 of the collisionless value at collision rate 0.1, with timing nearly unchanged.
4. One-step memory test: the delay-embedding memory model beats the memoryless closure by a large factor at all three cadences on held-out cases, including a collisional case not seen in training.
5. Autonomous rollout test (added in this version): the memory model keeps skill in two of three held-out cases at fine cadence (NRMSE 0.20 to 0.28). It fails for the largest-amplitude case (3.5, 0.6, 0), with NRMSE 0.88 to 1.30. The memoryless closure and persistence have no rollout skill (NRMSE near 1). The one-step advantage therefore overstates rollout ability.

## Caveats stated in the paper

- The echo timing is ambiguous. The difference signal has several maxima from beat structure. The reported time is the largest maximum, about two time units after 2 tau, and is not a validated echo time.
- The memory model is linear. A nonlinear neural operator was not built.
- The collisional suppression is a property of the BGK model. It is not shown to be thermodynamic irreversibility, and no grid-convergence study at fixed collision rate was done.
- No observational (MMS) analysis was carried out.
- External literature mentioned in the research note was not verified.

## Files

- paper/Plasma_Echo_Constrained_Kinetic_Memory.pdf (12 pages; in the release bundle)
- code/vlasov.py, echo_experiments.py, memory_experiment.py, rollout_experiment.py, make_figures.py, build_paper.py (in the release bundle)
- results/ and figures/ (in the release bundle)
- CITATION.cff, .zenodo.json, requirements.txt (this folder)

## Reproduce

```bash
pip install -r requirements.txt
python code/echo_experiments.py
python code/memory_experiment.py
python code/rollout_experiment.py
python code/make_figures.py
python code/build_paper.py
```

No licence file is included. The authors should choose one before reuse.
