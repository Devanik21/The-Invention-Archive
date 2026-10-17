# Literature audit — 2026-10-17

## Research questions searched

1. What information-theoretic results bound communication through gene-regulatory systems under molecular noise?
2. What prior work designs synthetic circuits and catalogues their compositional failure modes?
3. Has generative AI already been applied to whole synthetic gene-circuit topologies, and what validation level was used?
4. What mathematical ingredients can rigorously separate an upstream proposal channel from a downstream realization channel?

## Primary and authoritative sources verified for manuscript use

| Source | Bibliographic identity | Finding relevant to this paper | Use / limitation |
|---|---|---|---|
| Tkačik, Callan & Bialek (2008) | “Information capacity of genetic regulatory elements,” *Physical Review E* 78, 011910. DOI: https://doi.org/10.1103/PhysRevE.78.011910 | Formalizes information capacity in noisy genetic regulatory elements. | Establishes the biological information-capacity foundation; does not model the generative AI proposal stage. |
| Brophy & Voigt (2014) | “Principles of genetic circuit design,” *Nature Methods* 11, 508–520. DOI: https://doi.org/10.1038/nmeth.2926 | Reviews circuit-design tools, tuning, and failure modes under cellular constraints. | Establishes context/constructability limitations. |
| Elowitz & Leibler (2000) | “A synthetic oscillatory network of transcriptional regulators,” *Nature* 403, 335–338. DOI: https://doi.org/10.1038/35002125 | Demonstrates the repressilator synthetic oscillator. | Historical engineered-circuit precedent, not an AI result. |
| Gardner, Cantor & Collins (2000) | “Construction of a genetic toggle switch in Escherichia coli,” *Nature* 403, 339–342. DOI: https://doi.org/10.1038/35002131 | Demonstrates a genetic toggle switch. | Historical functional-circuit precedent. |
| Wall, Hlavacek & Savageau (2004) | “Design of gene circuits: lessons from bacteria,” *Nature Reviews Genetics* 5, 34–42. DOI: https://doi.org/10.1038/nrg1244 | Discusses circuit design principles and control/robustness relationships. | Background for circuit dynamics and design constraints. |
| Chakraborty et al. (2022 review) | “Designing Biological Circuits: From Principles to Applications,” *ACS Synthetic Biology* 11(4), 1377–1388. DOI: https://doi.org/10.1021/acssynbio.1c00557 | Surveys design methods, models, functions, robustness, and open challenges. | Broad methods review; used as contextual evidence only. |
| Gallup & Steel (2026) | “Generative design of synthetic gene circuits for functional and evolutionary properties,” *npj Systems Biology and Applications* 12, article 62. DOI: https://doi.org/10.1038/s41540-026-00683-6 | Uses a conditional variational autoencoder to generate RNA-circuit topologies and evaluates simulated dynamic/evolutionary properties. | Closest direct precedent. It establishes that generative circuit design is not a new premise. Study is in silico; the paper here does not reproduce its model/data. |
| Shannon (1948) | “A Mathematical Theory of Communication,” *Bell System Technical Journal* 27, 379–423. DOI: https://doi.org/10.1002/j.1538-7305.1948.tb01338.x | Foundational channel-capacity framework. | Mathematical background for channel abstraction. |

## Searches and audit trace

Representative queries included:

- `information-theoretic limits generative models genetic circuit design DNA language models synthetic gene circuits`
- `generative AI design genetic circuits synthetic biology foundation model sequence design experimental validation 2024 2025`
- `information theory genetic circuit design mutual information capacity gene regulation circuits review`
- `AI generated genetic circuits design closed-loop synthetic biology gene circuit design 2025`
- `generative design synthetic gene circuits functional evolutionary properties`
- `information capacity genetic regulatory elements gene circuit mutual information`

The March 2026 paper by Gallup and Steel is highly relevant, so the literature gap is deliberately narrow. The framework in the current manuscript is presented as a formal synthesis and evaluation proposal, not an unqualified novelty claim. No broad database-by-database systematic-review protocol was completed; no claim of exhaustive coverage is made. Patent landscape searching and direct raw-data reproduction remain outside this cycle.

## Bibliographic checks

All cited references have identifiable publisher or DOI landing pages; dates, titles, author strings, volumes/pages/article number, and DOI strings were cross-checked against search-result records. A complete automated DOI resolver audit was not run. Any future venue submission should independently recheck every field against the live publisher record.
