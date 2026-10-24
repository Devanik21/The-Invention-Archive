# Literature and novelty audit

**Search date:** 2026-10-24. **Coverage:** targeted web searches of primary literature / publisher, PubMed, PMC, GEO and source repositories. This is not an exhaustive patent or all-index audit.

## Key sources and relevance

1. Zhou et al., “Quantifying Waddington landscapes and paths of non-adiabatic cell fate decisions for differentiation, reprogramming and transdifferentiation,” *Journal of the Royal Society Interface* 10 (2013), DOI [10.1098/rsif.2013.0787](https://doi.org/10.1098/rsif.2013.0787). This directly covers landscape barriers, transition rates and biological reprogramming; the broad barrier/landscape connection is not novel.
2. Edelsbrunner, Letscher & Zomorodian, “Topological persistence and simplification,” *Discrete & Computational Geometry* 28 (2002), DOI [10.1007/s00454-002-2885-2](https://doi.org/10.1007/s00454-002-2885-2). Foundational persistence.
3. Cohen-Steiner, Edelsbrunner & Harer, “Stability of persistence diagrams,” *Discrete & Computational Geometry* 37 (2007), DOI [10.1007/s00454-006-1276-5](https://doi.org/10.1007/s00454-006-1276-5). Stability bounds apply to perturbations of functions under their stated metric; they do not ensure biological potential identification.
4. Freidlin & Wentzell, *Random Perturbations of Dynamical Systems*, 3rd ed. Springer (2012), DOI [10.1007/978-3-642-25847-3](https://doi.org/10.1007/978-3-642-25847-3), and Kramers (1940), DOI [10.1016/S0031-8914(40)90098-2]. Large-deviation action and escape-rate foundations.
5. Jiang, Lim, Yao & Ye, “Statistical ranking and combinatorial Hodge theory,” *Mathematical Programming* 127 (2011), DOI [10.1007/s10107-010-0419-x](https://doi.org/10.1007/s10107-010-0419-x). Relevant to discrete Hodge ideas; not evidence that gene-expression curl equals H1.
6. Zarzycki et al., “pbn-STAC: Deep reinforcement learning-based framework for cellular reprogramming,” *Theoretical Computer Science* 1049 (2025), article 115382, DOI [10.1016/j.tcs.2025.115382](https://doi.org/10.1016/j.tcs.2025.115382), code [repository](https://github.com/jakub-zarzycki2022/pbn-stac). Very close prior work for RL-based reprogramming in Boolean and probabilistic Boolean regulatory networks.
7. Fournié et al., “Cell trajectory inference based on schrödinger problem and a mechanistic model of stochastic gene expression,” *npj Systems Biology and Applications* 12, 120 (2026), DOI [10.1038/s41540-026-00743-x](https://doi.org/10.1038/s41540-026-00743-x). Recent related work in Schrödinger-problem trajectory inference with a mechanistic bursty-expression reference process.
8. “Deciphering cell-fate trajectories using spatiotemporal single-cell transcriptomic data,” *npj Systems Biology and Applications* (2025), DOI [10.1038/s41540-025-00624-9](https://doi.org/10.1038/s41540-025-00624-9). Discusses Schrödinger bridges for stochastic single-cell trajectory inference. Full author metadata should be checked at the publisher before formal submission; the bibliography marks this record as a comparator rather than a fully audited citation.
9. NCBI GEO **GSE242423** [record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242423) and **GSE242424** [SuperSeries](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242424). Public records describe a 14-day OSKM-induced human fibroblast reprogramming time course; raw data restrictions are noted on the GEO page. No data were downloaded or analyzed in this run.

## Novelty conclusion

The broad use of landscape barriers in reprogramming, persistent homology, Hodge decomposition, Schrödinger bridges, and RL for regulatory-network control all have prior art. This release makes no “first ever” claim. Its defensible contribution is an integrated synthetic stress test and an explicit counterexample to the unconditional hypothesis that a nontrivial cycle forces pulsed control, with a reproducible negative RL-vs-heuristic result. Whether this exact integration is novel remains uncertain pending a broader systematic search.

## Central claim audit

- **Barrier = H0 persistence:** conditionally valid when using the same smooth scalar potential and the matched local-minimum / index-one-saddle persistence pair. In isotropic gradient diffusions under the paper's normalization, the FW action barrier equals the potential rise. Not generally true for non-gradient drift, state-dependent diffusion, arbitrary learned scores, or an incorrect pairing.
- **Cycles force pulsed protocols:** falsified as an unconditional statement by the analytic S1 control example. The Betti number alone does not specify endpoint control cost. A narrower claim about dynamic loop/curl features predicting schedule benefit remains untested.
- **RL finds better protocols:** not supported by current benchmark. The learned RL policy succeeds in 7.75% of evaluation episodes, below the 23.25% evenly-spaced pulse baseline.
- **Real-data applicability:** not tested; GEO records are only candidate data sources.


## Final pre-publication update (24 October 2026)

A final current-literature pass added the 2026 AAAI paper by Mizera and Zarzycki, *The GATTACA Framework: Graph Neural Network-Based Reinforcement Learning for Controlling Biological Networks* (DOI: 10.1609/aaai.v40i2.37055). It combines graph neural networks and deep reinforcement learning to control Boolean biological networks and reinforces that RL for biological network control is established prior art. The manuscript now cites it explicitly and narrows its contribution to continuous-state synthetic stress tests and counterexample-led evaluation of topology/control assumptions; it does not claim RL control itself is novel. Publisher record: https://ojs.aaai.org/index.php/AAAI/article/view/37055 (published 14 March 2026).

The publisher record for *Deciphering cell-fate trajectories using spatiotemporal single-cell transcriptomic data* gives online publication on 4 December 2025 and volume 12, article 2 (2026). The BibTeX metadata was corrected to reflect that publication history and the visible author list rather than leaving an `Authors, Various` placeholder. The public-data candidate GSE242423/GSE242424 remains unevaluated in this paper; availability was recorded but no data were downloaded or fitted.
