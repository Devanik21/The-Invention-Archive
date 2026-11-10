# The Non-Abelian Geometry of Morphogenesis: Holonomic Cell-Fate Transport through Active p-Atic Topological Defects and Causal Neural Operators

**Release 2026.11.09 · Preprint v1.0 · Devanik Debnath and Xylia**

Not peer reviewed. The work is a theory-plus-simulation study of abstract models. It makes no biological claim, and no experiment on living tissue is reported. No DOI or Zenodo deposit is claimed.

## Author

Electronics and Communication Engineering graduate and independent AI researcher exploring how physical and mathematical structure can improve learning and modeling. Work spans gamma-ray burst reconstruction at the National Astronomical Observatory of Japan (NAOJ), physics-inspired signal classification, reinforcement learning, neural quantum-state methods, causal emergence, morphogenetic neural architectures, and computational artificial life. The research style combines algorithm design with explicit assumptions, code-based checks, and disclosure of unresolved limitations.

Devanik Debnath is a Scientific Machine Learning and AI Architectures researcher; B.Tech, Electronics and Communication Engineering, National Institute of Technology Agartala (2026); Samsung Grade I Fellowship, Cohort 7. Links: github.com/Devanik21, orcid.org/0009-0001-8310-6921.

## Summary

A matrix-valued connection on the space of intervention histories gives an intervention-order effect that is a curvature. The paper proves:

1. The order effect between two intervention orders is the commutator of the generators, and scales as the square of the step size (Theorem 1).
2. The holonomy of a rectangular loop deviates from the identity by its area times the curvature (Theorem 2).
3. Plaquette holonomy traces are gauge invariant (Theorem 3).
4. Integer holonomies of a p-atic field satisfy a discrete Stokes identity (Proposition 4).
5. Under balanced exposure, the order-dependent part of the second signature level is the Lévy area (Proposition 5), and reversal flips its sign (Proposition 6).

A central correction: the p-atic sector alone is Abelian (holonomy group Z_p), so noncommuting holonomy must come from the transport generators or from a dihedral extension. The dihedral case is stated as Conjecture 1.

## Key numerical results (results/summary.json, seed 20261110)

- Order effect exponent 1.999 (theory 2); loop deviation exponent 1.000 (theory 1); area coefficient 0.7052 against curvature norm 0.7071.
- Gauge-invariance error 7e-16 over 400 plaquettes; median curvature signal 1.18.
- Discrete Stokes identity: 294 of 294 planted-defect loops recovered exactly, for p = 2, 3, 4, 6.
- Fate under reversal, chosen pair: 91% of histories flip. Commuting control: 0%.
- Held-out AUC at N = 6400 (labels with 10% noise, Bayes ceiling about 0.90): Lévy area 0.89, positional 0.90, bigram 0.78, exposure 0.50.
- Counterfactual reversal accuracy: Lévy area 0.95, positional 0.95, bigram 0.74, exposure 0.55.
- Sensitivity sweep over 300 generic pairs: median flip fraction 0.10 (10th to 90th percentile 0.00 to 0.61); sign-of-Lévy rule median accuracy 0.50. The strong effect of the chosen pair is therefore not typical of the family, and Conjecture 2 holds only for that pair.

## Scope and limitations

Two fixed generators in su(2). Logistic order-aware baselines, not a trained neural operator. Single lattice size for the p-atic experiments. Label noise caps AUC. The literature search was not exhaustive.

## Falsification programme

Section 8 of the paper states five pre-specified predictions for epithelial perturbation data, with failure conditions. No such data have been analysed.

## Files

- paper/The_Non_Abelian_Geometry_of_Morphogenesis.pdf (19 pages; in the release bundle)
- code/morpho_experiments.py, code/build_paper.py (in the release bundle)
- results/ and figures/ (in the release bundle)
- CITATION.cff, .zenodo.json, requirements.txt (this folder)

## Reproduce

```bash
pip install -r requirements.txt
python code/morpho_experiments.py
python code/build_paper.py
```

No licence file is included. The authors should choose one before reuse.
