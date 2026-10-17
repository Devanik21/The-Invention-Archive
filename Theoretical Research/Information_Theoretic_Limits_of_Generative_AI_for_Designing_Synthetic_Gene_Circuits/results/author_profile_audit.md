# Author-profile audit — 2026-10-17

## Scope and method

Queries were run for `Devanik Debnath` and `Devanik21`, including the requested GitHub, ORCID, LinkedIn, Kaggle, Zenodo, OpenAIRE, and Google Scholar sources. This audit is deliberately coverage-limited: it is not represented as exhaustive. Search engine results can be stale, incomplete, or conflate people with similar names.

## Results

| Source | Observation from current audit | Treatment in release |
|---|---|---|
| GitHub profile, https://github.com/Devanik21 | Public profile search result resolves to `Devanik21 (Devanik)`, shows the name Devanik Debnath, NIT Agartala, ECE '26, and links to the supplied ORCID and Kaggle profiles. The biography contains self-descriptions including Samsung Fellow @IISc and ISRO Hackathon Winner. | Uses author name and institution only. Fellowship/hackathon claims are treated as self-reported and not included as verified credentials. |
| GitHub archive, https://github.com/Devanik21/The-Invention-Archive | Repository exists, is public, default branch is `main`, and linked GitHub integration reports write permission. Existing release directories and index were inspected. | Used for archive planning; actual new release is only complete if the later remote verification succeeds. |
| ORCID, https://orcid.org/0009-0001-8310-6921 | Supplied identifier was searched. No matching publication record was independently resolved in this search pass; this does not establish absence of an ORCID record. | Identifier included as supplied, with no publication counts or record claims. |
| LinkedIn, https://www.linkedin.com/in/devanik | Exact handle could not be independently resolved from the returned search results. A result for a different `Dev Debnath` profile was not treated as this author's profile. | No employment or affiliation claims sourced from ambiguous results. |
| Kaggle, https://www.kaggle.com/devanikdebnath and https://www.kaggle.com/devanik | URLs are listed on the author's public GitHub profile. Profile details were not independently audited in depth. | Links recorded; no publication or achievement claims inferred. |
| Zenodo, OpenAIRE, Google Scholar | Queries did not yield a reliably matched record under the supplied name/handle during this pass. Search coverage is incomplete. | Status marked unresolved, not “no record exists”; no DOI, citation count, or index status asserted. |

## Identity and credential decision

The manuscript identifies Devanik Debnath and the provided NIT Agartala affiliation. It does not state that a degree has been awarded. It excludes the publicly self-described fellowship and hackathon award because those details were not independently verified from authoritative sources during this cycle. No same-name publication was attributed without corroboration.

## Coverage and limitations

Audit date: 2026-10-17. Sources included the public GitHub profile and repository metadata, search results for author name/handle and supplied profile URLs, and the repository's current directory/index structures. Authentication-gated profile pages and indexing services may not expose complete information. Re-run this audit in later releases.
