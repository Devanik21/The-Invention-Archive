# Verification summary

Status: **PASS for declared synthetic-code checks only**.

- `PASS` — surrogate_bounds: All surrogate transmission values are within [0,1].
- `PASS` — nonnegative_attenuation_clear: Synthetic coefficients are nonnegative.
- `PASS` — depth_monotonicity_clear: For nonnegative K, deeper irradiance is no larger at same wavelength.
- `PASS` — nonnegative_attenuation_mixed: Synthetic coefficients are nonnegative.
- `PASS` — depth_monotonicity_mixed: For nonnegative K, deeper irradiance is no larger at same wavelength.
- `PASS` — nonnegative_attenuation_turbid: Synthetic coefficients are nonnegative.
- `PASS` — depth_monotonicity_turbid: For nonnegative K, deeper irradiance is no larger at same wavelength.
- `PASS` — all_three_worlds: Positive-signal and negative-control worlds are present.
- `PASS` — heldout_predictions_finite: Held-out metrics are finite.
- `PASS` — expected_csvs: Data tables needed for plots exist and are nonempty.
- `PASS` — eight_figure_sources: Eight plot/graph/flow assets were generated in PDF and PNG.

No empirical biological or paleolimnological validation was performed. The spectral generator is not RCWA/FDTD.
