# Open Problems Programme

A single home for the open research programme of Devanik Debnath with Xylia (AI research scientist, multi-agent research system). Every problem has its own folder. Every sub-project has a status, a pre-registration where one exists, its code, and its results, including negative results.

Status values: `not started`, `in progress`, `tested (negative)`, `tested (positive)`, `paused`.

Numerical results here come from stylised, dimensionless models. They are not biological validation, and nothing here makes a claim about human lifespan or medicine.

## The three problems

| # | Problem | Folder | Status |
|---|---|---|---|
| 1 | Emergence of biological information from nonliving chemistry: how do chemical networks acquire heritable, information-bearing function? | [01_Biological_Information_Emergence](01_Biological_Information_Emergence/) | in progress |
| 2 | Genotype-phenotype mapping: can we characterise a causal, non-injective mapping from genome to phenotype, and bound what a genome can tell us about phenotype? | [02_Genotype_Phenotype_Mapping](02_Genotype_Phenotype_Mapping/) | not started |
| 3 | Evolutionary reachability: which evolutionary outcomes are accessible in rugged, epistatic landscapes, and which are not? | [03_Evolutionary_Reachability](03_Evolutionary_Reachability/) | not started |

## Sub-projects under problem 1

| Sub-project | Question | Status |
|---|---|---|
| [PCT-I Hereditary Catalytic Capacity](01_Biological_Information_Emergence/PCT-I_Hereditary_Catalytic_Capacity/) | Can a catalytic network be self-maintaining without carrying heritable catalytic information? | tested (negative) |
| [PCT-II Hierarchical Information Architecture](01_Biological_Information_Emergence/PCT-II_Hierarchical_Information_Architecture/) | Does a two-level encoding architecture carry more heritable information than a single level at matched noise? | not started |
| [PCT-III Multigenerational Persistence](01_Biological_Information_Emergence/PCT-III_Multigenerational_Persistence/) | Does heritable catalytic information persist across generations, or is it lost? | tested within PCT-I (no loss observed) |

## Rules for this folder

1. A test is pre-registered before it is run. Criteria are written into the code docstring and the README before any result is seen.
2. Negative results are kept and reported with the same prominence as positive ones.
3. Each sub-project folder contains only: README (status and question), RESULT_NOTE or RESULTS (what was run), and code/ (reproducible scripts).
4. Artifacts from an earlier version are never overwritten; a changed model gets a new named file, and the change is documented.

## Disclosure

AI assistance: the simulations, code and written text were produced with the assistance of Claude (Anthropic) under the direction of the authors, who define the research questions and are responsible for the content. Xylia is the authors' AI research scientist. Competing interests: none declared. Funding: none. Nothing here is peer reviewed.
