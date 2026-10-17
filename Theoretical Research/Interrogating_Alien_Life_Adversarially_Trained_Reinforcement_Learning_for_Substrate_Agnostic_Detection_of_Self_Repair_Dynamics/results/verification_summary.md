# Verification summary

Run started (UTC): 2026-10-17T08:36:43.340722+00:00
Seed: 20261017
Q-learning episodes per agent: 30000
Evaluation design: 500 repair-capable episodes and 500 episodes per each of 4 synthetic abiotic decoys, per method.

## Checks
- PASS: Q tables have expected shape and contain finite values.
- PASS: five synthetic probe actions and four explicit decoy families are present.
- PASS: all evaluated sensitivity and false-positive rates are bounded in [0, 1].
- PASS: nine distinct figure assets generated in PDF and PNG formats.
- PASS: one-factor noise sensitivity evaluated at multipliers 0.75, 1.0, 1.25, and 1.5 with a fixed trained policy.
- PASS: episode-level evaluation data and synthetic trajectory data archived.
- LIMITATION: this is an internally generated synthetic benchmark; no field, spacecraft, terrestrial instrument, or alien sample data were used.
- LIMITATION: success in this simulator is not evidence of extraterrestrial life or a validated flight instrument.

## Metrics

## One-factor observation-noise sensitivity

- Noise multiplier 0.75: balanced accuracy 0.975; sensitivity 0.98; specificity 0.97; worst-decoy FPR 0.085; per-decoy FPR `{"Driven oscillator": 0.0, "Passive relaxor": 0.0, "Scripted controller": 0.085, "Self-healing material": 0.035}`
- Noise multiplier 1.0: balanced accuracy 0.92625; sensitivity 0.93; specificity 0.9225; worst-decoy FPR 0.235; per-decoy FPR `{"Driven oscillator": 0.0, "Passive relaxor": 0.0, "Scripted controller": 0.235, "Self-healing material": 0.075}`
- Noise multiplier 1.25: balanced accuracy 0.87062; sensitivity 0.81; specificity 0.93125; worst-decoy FPR 0.165; per-decoy FPR `{"Driven oscillator": 0.015, "Passive relaxor": 0.0, "Scripted controller": 0.165, "Self-healing material": 0.095}`
- Noise multiplier 1.5: balanced accuracy 0.84813; sensitivity 0.775; specificity 0.92125; worst-decoy FPR 0.21; per-decoy FPR `{"Driven oscillator": 0.005, "Passive relaxor": 0.0, "Scripted controller": 0.21, "Self-healing material": 0.1}`

### Adversarial Q-RL
- Sensitivity: 0.932
- Specificity: 0.9375
- Sensitivity Wilson 95% interval: [0.90648, 0.95093]
- Balanced accuracy: 0.93475
- AUROC (score ranking): 0.97851
- Mean probes: 3.7516
- Worst-decoy FPR: 0.168
- Aggregate FPR Wilson 95% interval: [0.05271, 0.07397]
- Per-decoy FPR: `{"Driven oscillator": 0.002, "Passive relaxor": 0.002, "Scripted controller": 0.168, "Self-healing material": 0.078}`
- Per-decoy FPR Wilson 95% intervals: `{"Driven oscillator": [0.00035, 0.01124], "Passive relaxor": [0.00035, 0.01124], "Scripted controller": [0.13779, 0.20328], "Self-healing material": [0.05758, 0.10485]}`

### Uniform Q-RL
- Sensitivity: 0.942
- Specificity: 0.919
- Sensitivity Wilson 95% interval: [0.91794, 0.95932]
- Balanced accuracy: 0.9305
- AUROC (score ranking): 0.98295
- Mean probes: 3.5076
- Worst-decoy FPR: 0.244
- Aggregate FPR Wilson 95% interval: [0.06983, 0.09378]
- Per-decoy FPR: `{"Driven oscillator": 0.0, "Passive relaxor": 0.0, "Scripted controller": 0.244, "Self-healing material": 0.08}`
- Per-decoy FPR Wilson 95% intervals: `{"Driven oscillator": [0.0, 0.00762], "Passive relaxor": [0.0, 0.00762], "Scripted controller": [0.2084, 0.28351], "Self-healing material": [0.0593, 0.10711]}`

### Recovery-only
- Sensitivity: 0.552
- Specificity: 0.496
- Sensitivity Wilson 95% interval: [0.50818, 0.59503]
- Balanced accuracy: 0.524
- AUROC (score ranking): 0.60445
- Mean probes: 1.0
- Worst-decoy FPR: 0.746
- Aggregate FPR Wilson 95% interval: [0.4821, 0.52588]
- Per-decoy FPR: `{"Driven oscillator": 0.048, "Passive relaxor": 0.69, "Scripted controller": 0.532, "Self-healing material": 0.746}`
- Per-decoy FPR Wilson 95% intervals: `{"Driven oscillator": [0.03246, 0.07043], "Passive relaxor": [0.64814, 0.72896], "Scripted controller": [0.48819, 0.57533], "Self-healing material": [0.70607, 0.78218]}`

### Fixed full panel
- Sensitivity: 0.976
- Specificity: 0.876
- Sensitivity Wilson 95% interval: [0.95852, 0.98622]
- Balanced accuracy: 0.926
- AUROC (score ranking): 0.98274
- Mean probes: 5.0
- Worst-decoy FPR: 0.328
- Aggregate FPR Wilson 95% interval: [0.11027, 0.13917]
- Per-decoy FPR: `{"Driven oscillator": 0.0, "Passive relaxor": 0.0, "Scripted controller": 0.328, "Self-healing material": 0.168}`
- Per-decoy FPR Wilson 95% intervals: `{"Driven oscillator": [0.0, 0.00762], "Passive relaxor": [0.0, 0.00762], "Scripted controller": [0.2883, 0.37033], "Self-healing material": [0.13779, 0.20328]}`

### Random order panel
- Sensitivity: 0.972
- Specificity: 0.8945
- Sensitivity Wilson 95% interval: [0.95355, 0.98325]
- Balanced accuracy: 0.93325
- AUROC (score ranking): 0.98443
- Mean probes: 5.0
- Worst-decoy FPR: 0.288
- Aggregate FPR Wilson 95% interval: [0.09278, 0.11973]
- Per-decoy FPR: `{"Driven oscillator": 0.008, "Passive relaxor": 0.0, "Scripted controller": 0.288, "Self-healing material": 0.126}`
- Per-decoy FPR Wilson 95% intervals: `{"Driven oscillator": [0.00312, 0.02039], "Passive relaxor": [0.0, 0.00762], "Scripted controller": [0.25004, 0.32919], "Self-healing material": [0.09973, 0.15797]}`

