# Burst-Robust Codes for Life and Qubits

Theory and synthetic-computation preprint by Devanik Debnath, 6 November 2026.

The paper separates spatial burst statistics, code-specific recoverability, and finite repair workload. For a stationary ergodic single-server fluid queue, finite-mean stationary workload requires lambda E[S] / mu < 1. For Poisson arrivals, catastrophic-footprint risk over horizon T is 1 - exp(-lambda q_F T). These are conditional queueing/thinning statements, not a universal biological or quantum threshold.

Synthetic benchmark: 24x24 4-neighbour lattice; iid erasures versus finite five-site line-track bursts; exactly matched expected lattice-wide mean occupancy (individual site marginals vary near clipped boundaries); 4,000 trials per point; seed 20261106. At p=0.40: iid 0/4000, track 705/4000 (17.625%). At p=0.50: iid 219/4000 (5.475%), track 2737/4000 (68.425%).

Boundary: geometric connectivity surrogate only. Not a surface-code decoder, holographic tensor-network simulation, radiation-transport calculation, or biological experiment.

Reproduce with Python 3.10+, NumPy, pandas, Matplotlib: python verify_model.py.

Holographic reconstruction is a quantum-code property, not a biological mechanism. Not every heavy-ion track overwhelms repair. dCas9-KillerRed can generate localized ROS damage and can repair rapidly under tested conditions. AlphaQubit 2 is not claimed to be validated on this specific radiation-burst channel. Protein spin sensors are not fault-tolerant cellular computers. The neutrino fog is a dark-matter detector background, not a biological repair floor.

## Release files

- [Manuscript PDF](paper.pdf)
- [Markdown source](paper.md)
- [Reproducible benchmark](verify_model.py)
- [Full numerical sweep](results/spanning_sweep.csv)
- [Verification report and calibration record](results/verification_report.json)
- [Spanning-failure figure](figures/figure1_spanning_threshold.png)
- [Example track realization](figures/figure2_track_realization.png)
- [Complete release bundle](../Burst_Robust_Codes_for_Life_and_Qubits_complete_bundle.zip)
