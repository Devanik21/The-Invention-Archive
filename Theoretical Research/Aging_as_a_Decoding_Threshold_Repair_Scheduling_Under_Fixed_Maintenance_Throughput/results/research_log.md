# Research workflow log

- Date: 2026-10-24
- Topic: user-supplied “Aging as a decoding threshold, and lifespan extension by repair scheduling”.
- Prior-art gate: passed with narrowed scope after finding prior optimal-control work and two related archive releases.
- Main formulation: deterministic four-state damage propagation with nonnegative off-diagonal dependencies, equal per-state allocated repair dose per 12-unit cycle, 24 period-3 schedules, and threshold first-passage time.
- Generated data: synthetic/model-generated only. No public biological dataset was downloaded; no lifespan/survival cohort was analyzed.
- Numerical method: fixed-step fourth-order Runge–Kutta, dt=0.02; convergence check repeated at dt=0.01.
- Validation includes: all schedule permutations across six coupling values, decoupled analytical limiting case, dose matching, malformed parameter rejection, and time-step refinement.
- Excluded from quantitative core: RL policy learning, HJB solution, p-adic/ultrametric clustering, empirical percolation threshold, and comparative-species calibration. Their data/operational prerequisites are absent here.
- Output: paper manuscript, verifier, reproducible plots/graphs/flow diagrams, CSV results, metadata, and deployment validator.
- Important limitation: model variables/parameters are illustrative dimensionless quantities; no biological rates are inferred.
