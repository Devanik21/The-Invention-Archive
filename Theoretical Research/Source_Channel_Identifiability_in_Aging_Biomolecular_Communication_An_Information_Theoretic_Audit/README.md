# Source-Channel Identifiability in Aging Biomolecular Communication: An Information-Theoretic Audit

**Author:** Devanik Debnath (Devanik21)  
**Affiliation:** Department of Electronics and Communication Engineering, National Institute of Technology Agartala, India  
**Release date:** 10 October 2026  
**Status:** Theoretical and computational methods preprint; not peer-reviewed by virtue of repository archival.

## Abstract

Mutual information between a regulatory input and a molecular output can change because the input distribution changes, because the conditional response channel changes, or because both change. The paper gives an exact reference-distribution decomposition and verifies it in a synthetic binary channel. The central caution is that a cohort-level mutual-information decline alone does not identify degradation of the conditional response law.

## Main equations

For a finite source distribution \(p(x)\) and conditional response channel \(W(y\mid x)\), the mutual information is:

$$
I_p(W)=\sum_{x,y}p(x)W(y\mid x)\log_2\frac{W(y\mid x)}{\sum_{x'}p(x')W(y\mid x')}.
$$

For a binary symmetric channel with error probability \(\epsilon\) and Bernoulli source probability \(p\),

$$
I_p(W)=H_2\bigl(\epsilon+p(1-2\epsilon)\bigr)-H_2(\epsilon).
$$

For a chosen reference input law \(q\), the natural cohort contrast has the exact decomposition:

$$
\Delta_{\mathrm{nat}}=\underbrace{I_{p_A}(W_A)-I_q(W_A)}_{D_A(q)}+\underbrace{I_q(W_A)-I_q(W_Y)}_{C(q)}+\underbrace{I_q(W_Y)-I_{p_Y}(W_Y)}_{D_Y(q)}.
$$

The middle term compares the estimated conditional laws under the same reference input distribution. The full decomposition is reference-dependent and is not a unique causal attribution.

## Research scope and contribution

This release formalizes a measurement-identifiability issue raised by recent information-theoretic work on aging-related transcriptional regulation. The contribution is a compact source-standardized audit and a reproducible synthetic counterexample. It does not claim a new biological mechanism, reproduce published biological data, or establish that aging is reducible to information loss.

## Reproducibility

Requirements: Python 3.10 or later and NumPy 1.24 or later.

Run:

```bash
python verify_model.py
```

The random seed is `20261010`; each binary-channel scenario uses 200,000 synthetic observations. The script checks analytical limits, source-distribution dependence under a fixed channel, empirical estimates against the analytical solution, invalid parameter rejection, common-source channel comparison, and the exact three-term decomposition. Expected output begins with `Source-channel identifiability audit: synthetic verification` and ends with `PASS` if all assertions pass.

The simulation is a synthetic mathematical check. It does not simulate genes, cells, organisms, an aging process, extracellular vesicles, or an intervention. It offers no clinical evidence and makes no causal claim.

## P1-P5 classification

| Class | Claim | Status |
|---|---|---|
| P1 | Mutual-information definition, source dependence, telescoping decomposition | Exact under the declared finite-alphabet model |
| P2 | Intervention could improve a biological output | Not established; requires identified dynamics and safety evidence |
| P3 | Universal aging-related communication scaling law | Not claimed |
| P4 | Regulatory input/output as a noisy channel | Useful mathematical correspondence, not a full biological mechanism |
| P5 | Aging is only “software failure” or declining MI proves channel corruption | Unsupported analogy/inference; rejected |

## Validation and falsification

The audit is useful only when inputs and outputs are consistently defined and cohorts share adequate support. Repeat the standardized comparison across prespecified reference laws, inspect per-input sample counts, match technical quality, account for cell composition, and quantify estimation uncertainty. If the conditional response comparison is unstable across reasonable reference laws or driven by assay preprocessing, the claimed inference should be weakened. A persistent standardized difference is a conditional-distribution difference under the chosen variables, not proof of a specific molecular mechanism.

## Literature and citation

The manuscript cites the 2026 *Cell Reports Physical Science* study on source-distribution mismatch in aging-related transcriptional regulation, the 2025 study of information-theoretic communication in aging muscle, and established work on information theory in biochemical signaling. Full metadata are in `paper.bib`.

## Author and repository audit (10 October 2026)

The current public GitHub profile was checked and links the supplied ORCID, Medium, LinkedIn, and Kaggle accounts. The repository’s latest visible session at audit time was `docs/sessions/2026-10-10_genomic_information_theory.md`, which motivates this paper’s careful treatment of Shannon entropy and biological interpretation. The GitHub profile self-describes distinctions including Samsung Fellow @IISc and ISRO Hackathon Winner; those are self-reported profile statements, were not independently corroborated in this audit, and are not used as evidence for the paper’s contribution or institutional endorsement.

Searches for author-specific Zenodo, OpenAIRE, and Google Scholar records did not produce a record confidently matched to the supplied ORCID and repository. Their absence from this release is intentional; no DOI, indexing status, or citation metric is asserted.

## Author links

- [GitHub profile](https://github.com/Devanik21)
- [Research archive](https://github.com/Devanik21/The-Invention-Archive)
- [ORCID](https://orcid.org/0009-0001-8310-6921)
- [Medium](https://medium.com/@devanik2005)
- [X](https://x.com/devanik2005)
- [LinkedIn profile](https://www.linkedin.com/in/devanik)
- [Kaggle](https://www.kaggle.com/devanikdebnath)
- [Additional Kaggle profile](https://www.kaggle.com/devanik)

Zenodo, OpenAIRE, and Google Scholar author records were not independently verified in the audit; no DOI or indexing result is claimed.

## Files

- `paper.tex` — LaTeX manuscript.
- `paper.pdf` — compiled preprint.
- `verify_model.py` — deterministic synthetic verification.
- `README.md` — overview, equations, validation boundaries, and reproducibility.
- `medium_story.md` — long-form technical essay draft; not published.
- `x_thread.md` — 12-post academic dissemination draft; not posted.
- `CITATION.cff` — GitHub citation metadata.
- `paper.bib` — verified bibliographic records.
- `.zenodo.json` — deposit metadata template; no DOI minted.
- `schema_article.jsonld` — structured article metadata; no indexing guarantee.
- `deploy.py` — local validation, build, index, commit, tag, push, and remote-check script.
