# ALIEN-DL exploratory prototype

This is a compact, inspectable PyTorch prototype for an initial feasibility test.

## Source files
- `alien_dl.py` - resource-coupled synaptic model and standard MLP baseline.
- `run_benchmark.py` - train/test comparison and Wine 2x2x2 coupling ablation.
- `test_alien_dl.py` - mathematical/numerical smoke tests.

## Results
- `results/benchmark_results.csv` - per-seed main comparison.
- `results/benchmark_summary.csv` - means and standard deviations.
- `results/factorial_results.csv` - per-seed R/P/A ablation.
- `results/factorial_summary.csv` - factorial summary.
- `EXPERIMENT_REPORT.md` - methods, measurements, interpretation and limitations.
- `run_metadata.json` - software versions and run configuration.

## Run

```bash
pip install -r requirements.txt
python run_benchmark.py
python -m unittest -v test_alien_dl.py
```

Optional quicker run:

```bash
python run_benchmark.py --epochs 60 --seeds 7 21 --factorial-seeds 7 21
```

## Scope and caveats

This is an exploratory model, not a biologically faithful neural simulator. It uses static tabular samples with short internal synaptic-state relaxation; transient states reset for each sample. Structural births are not implemented, and hard pruning did not occur in the selected validation-best checkpoints in the reported run. The three-seed results are not confirmatory and do not show consistent superiority over parameter-matched MLPs.
