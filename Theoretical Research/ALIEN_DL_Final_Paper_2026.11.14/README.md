# ALIEN-DL final research paper (v1.0, 14 November 2026)

Authors: Devanik Debnath and Xylia (AI research scientist, multi-agent research system).
Status: preprint, not peer reviewed. Exploratory evidence only; no superiority or biological-fidelity claim.

## Contents

- `ALIEN_DL_Final_Research_Paper.pdf`: 14-page paper covering both trials. Binary; upload from `ALIEN_DL_Final_Bundle.zip`.
- `code/build_paper.py`: builds the PDF from `data/` and `figures/`. In the bundle.
- `code/make_figures.py`: regenerates figures. In the bundle.
- `data/`: Trial A per-seed, summary and factorial CSVs (also in `A_xylia_prototype/` of the research-trials folder).
- `figures/`: all figures as PNG and SVG. In the bundle.

## Result in one paragraph

Neither trial shows a superadditive benefit from the three-way coupling. ALIEN does not beat parameter-matched or state-matched MLPs on the datasets tested. Allocation becomes non-uniform, and in Trial B it collapses at step size 0.5. The pre-specified kill criterion from the v2 specification is triggered for the settings tested.

## What changed from the trials report

- Interaction contrasts are stated at the fixed level A = 0, which reproduces the report's values exactly. Every nonzero seed contrast equals one Wine test example (1/27).
- Prior-art positioning (Table 11) and the temporal order benchmark (Section 14) are included from the report.
- Section 15 states which references were checked by the authors and which were not. Several 2025 and 2026 references, and the simplex-optimisation reference (Chok and Vasil 2025), are cited as given in the report and are not verified here.

## Caveats

- Three seeds; test sets of 27, 171 and 540 examples.
- The Trial B step-size sweep was selected on the test split and is descriptive only.
- The Trial B state-matched MLP (width 32) has about twice the parameters of the ALIEN hidden-16 model. The plain width-16 MLP is the closer parameter comparator.

## Reproduce

```bash
pip install numpy pandas matplotlib reportlab pypdf
python code/make_figures.py
python code/build_paper.py
```

The build script reads CSVs from `data/` in the bundle. If you move files, update the path in `build_paper.py`.
