# ALIEN-DL research trials, 2026-11-14

Two independent first-pass implementations of the same idea: a resource-coupled synaptic classifier with three switchable couplings (R: allocation to recovery, P: allocation to plasticity, A: allocation to structure). Neither trial supports a superiority claim. Both report negative or inconclusive first results.

## Trial A: Xylia prototype (PyTorch)

Folder: A_xylia_prototype/. Files: EXPERIMENT_REPORT.md, README__17_.md, benchmark_results.csv, benchmark_summary.csv, factorial_results.csv, test_alien_dl.py, alien_dl.py, run_benchmark.py.

- Full PyTorch model with dendritic compartments, short-term state, structural gate, 2x2x2 ablation.
- 3 seeds (7, 21, 42), 70/15/15 splits, parameter-matched and width-32 MLP baselines.
- Result: ALIEN 85.2% (Wine), 98.5% (Breast Cancer), 93.8% (Digits); parameter-matched MLP 92.6%, 98.5%, 96.1%.
- Factorial on Wine: no superadditive interaction (three-way contrast +0.012, SD 0.021). ALIEN 3.6 to 6.3 times slower.
- Limitations stated in its own report: allocation signal is a heuristic; structural births not implemented; pruning did not occur in the main checkpoints; temporal synaptic states reset per example.

## Trial B: lite trial (NumPy + autograd)

Folder: B_lite_trial/. Files: code/alien.py, code/unit_tests.py, code/run_alien.py, code/eta_sweep.py. Results (JSON) and logs are in the zip bundle.

- Compact model with the same R/P/A switches, a budget-conserving exponentiated-gradient allocation step on dL/dpi, and three unit tests (budget conservation, positivity, reduction to a weighted-sum network). All three pass.
- Breast Cancer and Digits, 3 seeds, 40 epochs, hidden width 16, 70/30 split.
- Digits: plain MLP 0.962, state-matched MLP 0.969, frozen-allocation ALIEN 0.959, transmission plus allocation 0.950, all three couplings 0.932 (final config: rho = 0.3, eta_b = 0.5).
- Allocation collapse: at eta_b = 0.5 one synapse reached share 63 of 64 (winner-take-all). At eta_b = 0.05 the largest share stayed at 6.8 and accuracy recovered to 0.960 for transmission plus allocation.
- Breast Cancer: frozen and most coupled configurations 0.961; full coupling 0.949.
- Caveats: the eta sweep selected on the test split (not validated). The full-coupling cell at eta_b = 0.5 did not finish (background process terminated) and is absent from the sweep results.
- results/wine.json comes from the first timing run, before the recovery-rate and step-size fixes. It is not comparable to the other results.

## Comparison

| | Trial A (Xylia) | Trial B (lite) |
|---|---|---|
| Framework | PyTorch | NumPy + autograd |
| Datasets | Wine, Breast Cancer, Digits | Breast Cancer, Digits (Wine saturated) |
| ALIEN vs matched MLP | Equal on Breast Cancer; -7.4 pt Wine; -2.2 pt Digits | Digits: -0.6 pt vs plain MLP (frozen ALIEN) |
| Allocation non-uniform | yes (Gini 0.24 Wine, 0.35 Breast Cancer, 0.64 Digits) | yes; collapses without step-size control |
| Superadditive interaction | not found | not found |

## Shared conclusion

Neither trial shows that coupling the three mechanisms through one budget helps. Both expose allocation instability and computational overhead. The next test is a temporal benchmark where synaptic state persists across a sequence, which neither trial implements.

## Corrections found on review (2026-11-14)

- Wine gap to the parameter-matched MLP is -7.4 points (0.852 vs 0.926). An earlier draft said -0.8; that was wrong.
- Wine full-ALIEN allocation Gini averages 0.24 across seeds (0.217, 0.203, 0.292), not 0.22.
- EXPERIMENT_REPORT.md states pruned active fractions of about 0.962 and 0.936 in the Wine factorial. The factorial CSV shows 0.885 (R=0, P=0, A=1, seed 42) and 0.808 (R=1, P=0, A=1, seed 42). The CSV is authoritative.
- The first push of benchmark_summary.csv was header-only. The corrected full file is in this commit.

The complete bundle (both trials, all uploaded files, the original zip, and the JSON results and logs) is ALIEN_DL_Research_Trials.zip, which is binary and must be uploaded separately.
