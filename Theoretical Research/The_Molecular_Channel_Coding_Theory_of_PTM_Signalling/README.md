# The Molecular Channel-Coding Theory of Life: Information-Theoretic Limits of Combinatorial Proteoform Signalling Under Stochastic Biochemical Noise

**Release 2026.11.12 · Preprint v1.0 (12 pages) · Devanik Debnath and Xylia**

Not peer reviewed. Theory-plus-computation study of a stylised 4-site post-translational modification channel. The chemical master equation is solved exactly. No measured biological data are used. No claim is made about any living cell, protein or organism. No DOI or Zenodo deposit is claimed.

## Author

Electronics and Communication Engineering graduate and independent AI researcher exploring how physical and mathematical structure can improve learning and modeling. Work spans gamma-ray burst reconstruction at the National Astronomical Observatory of Japan (NAOJ), physics-inspired signal classification, reinforcement learning, neural quantum-state methods, causal emergence, morphogenetic neural architectures, and computational artificial life. The research style combines algorithm design with explicit assumptions, code-based checks, and disclosure of unresolved limitations.

Devanik Debnath is a Scientific Machine Learning and AI Architectures researcher; B.Tech, Electronics and Communication Engineering, National Institute of Technology Agartala (2026); Samsung Grade I Fellowship, Cohort 7. Links: github.com/Devanik21, orcid.org/0009-0001-8310-6921.

## Model

Each input is a binary kinase pattern s in {0,1}^4. Each site is modified or not. Modification on-rate: alpha * kappa * s_i * (1 + c * sum of other modified sites). Removal rate: beta * kappa. The output is the full proteoform state at time t, from the exact generator exp(t Q_s).

## Results (from the run, seed 20261112)

- Validation: two-state CME matches the analytic formula to ten digits. Blahut-Arimoto matches the closed-form Z-channel capacity to ten digits.
- Scaling law (proved and checked): capacity depends only on the product kappa*t. At kappa*t = 0.5, capacity is 0.847703 bits for three different (kappa, t) pairs.
- Dissipation collapse (proved): expected cycles also depend only on kappa*t, so capacity is a single function of dissipation. Information per cycle falls as turnover rises.
- Stationary ceiling (proved, checked): 4 * log2(1 + (2/3) * (1/3)^(1/2)) = 1.879128 bits, matched numerically to six digits.
- Crosstalk: capacity at t = 2 rises from 1.751 (c = 0) to 2.634 (c = 2).
- Codebook error, MAP, uniform prior: two codewords at distance 2 give 0.183; at distance 4 give 0.009 without crosstalk and 0.0015 with crosstalk c = 1.
- Bhattacharyya bound (Proposition 7) is an upper bound in both checked cases, tight at distance 2, loose by about 7x at distance 4.
- Learned decoder (one hidden layer, tanh): 0.438 accuracy from 1000 samples, against an exact Bayes limit of 0.445.

## Caveats stated in the paper

- Four sites, binary states, full-state readout. Real PTM systems are larger and richer.
- Energy is counted as expected cycles, not free energy.
- Rates are not fitted to any data.
- The Molecular Information Coding Network proposed in the research note is not built. The learned component is a small decoder.
- The validation plan (PhosphoSitePlus, PRIDE, UniProt, AlphaFold DB, STRING) is not carried out.
- Literature claims from the research note (Gunawardena 2012, DeepPCT 2024, DeepMVP 2025) were not verified.

## Files

- paper/Molecular_Channel_Coding_Theory_of_PTM_Signalling.pdf (12 pages; in the release bundle)
- code/ptm_channel.py and code/ptm_experiments.py (in this folder)
- code/make_figures.py and code/build_paper.py (in the release bundle)
- results/ and figures/ (in the release bundle)
- CITATION.cff, .zenodo.json, requirements.txt (this folder)

## Reproduce

```bash
pip install -r requirements.txt
python code/ptm_experiments.py
python code/make_figures.py
python code/build_paper.py
```

No licence file is included. The authors should choose one before reuse.
