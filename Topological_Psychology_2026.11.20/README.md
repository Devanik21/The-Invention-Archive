# Topological Psychology of the Affective Life Space (v1.0, 20 November 2026)

Authors: Devanik Debnath and Xylia (AI research scientist, multi-agent research system).
Status: preprint, not peer reviewed. A mathematical model of an analogy with no data behind its parameters.
Not for diagnosis, prediction or assessment of any individual.

## Concept

Kurt Lewin (1936, Principles of Topological Psychology) proposed that behaviour is a function of the psychological life space. Here the affective state is a point in a space shaped by parameters: tension a (slow) and provocation b (fast). The potential V(x; a, b) = x^4/4 + a x^2/2 - b x has the cusp bifurcation set 4a^3 + 27b^2 = 0. Crossing the fold is read as an outburst. Regulation is a gain on provocation.

## Results (seed 20261120)

- E1 topology: stable-state count matches the cusp set at 99.99 per cent of a 161 x 161 grid (25,921 points). The three residual points lie on the fold or at the cusp point.
- E2 hysteresis: loop width 0.820 in simulation against 0.770 analytic (one realisation). The excess is a rate-dependent delay.
- E3 Kramers: fitted slope of ln(mean first-passage time) against 1/D is 0.287, against an analytic barrier of 0.25 (15 per cent; barrier-to-noise ratio only 1 to 2.5). 200 replicates per D.
- E4 deformation: overshoot past the fold grows with ramp speed; log-log exponent about 0.77 over three speeds. At speed 0.032 no outburst occurs within the window. Exponent is a conjecture, not an established law.
- E5 regulation: reducing provocation gain from 1.0 to 0.4 changes the fraction of time in the high branch by about one percentage point. Null result, one run per gain.

## Scope and limitations

- One-dimensional affect coordinate. Valence, arousal and control are not separated.
- No empirical data. Parameters are not fitted to measurements.
- Single stochastic realisations for E2, E4 and E5.
- The parameter-to-emotion mapping is a hypothesis. Section 7 of the paper lists five falsifiable predictions and what would count against each.
- Citations were not re-verified against publisher pages, except Lewin (1936) against a bibliographic record.

## Folder layout

- code/cusp_affect.py: potential, equilibria, fold locations, Langevin integrator.
- code/experiments.py: experiments E1 to E5.
- results/: JSON output of each experiment. E4 contains NaN for the censored speed, which is valid Python output but not strict JSON.
- paper/: the 11-page PDF is binary and is in the bundle, Topological_Psychology_Bundle.zip.
- figures/: PNG and SVG figures are in the bundle.

## Reproduce

```bash
pip install numpy matplotlib reportlab pypdf
python code/experiments.py
```

The figure and paper scripts are in the bundle.
