"""
verify_model.py -- independent checks for the Biohybrid Infrastructure Viability preprint v1.0.

Recomputes the closed-form and small numerical claims in the paper from first principles
and compares them with the values stated in the text. It does NOT rerun the Monte Carlo
architecture study, the swarm study or the viability kernel; those are reproduced by
analysis.py (seed 20261121).

Writes results/summary.json. Exit status 0 if every check passes, 1 otherwise.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.special import erfcinv

HERE = os.path.dirname(os.path.abspath(__file__))

# Constants (paper, Appendix D)
SIGMA = 5.670374e-8          # W m^-2 K^-4
L_SUN = 3.828e26             # W
AU = 1.495978707e11          # m
C_LIGHT = 2.998e8            # m s^-1
S0 = L_SUN / (4 * math.pi * AU**2)   # ~1361 W m^-2 at 1 AU
T_SUN = 5772.0               # K
T_ENV = 300.0                # K

checks = []


def check(name, value, expected, tol, source):
    ok = abs(value - expected) <= tol
    checks.append({"check": name, "value": float(value), "expected": float(expected),
                   "tolerance": float(tol), "pass": bool(ok), "source": source})
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}: value={value:.6g} expected={expected:.6g} (tol {tol:g})  [{source}]")


# 1. Capture-geometry law, Proposition 1
req_type2 = 1e26 / L_SUN
check("Type II requirement eta*f_cov", req_type2, 0.2612, 5e-5, "Prop. 1, Sec. 6")
check("Type III in solar luminosities (x1e-9)", 1e36 / L_SUN / 1e9, 2.612, 5e-4, "Sec. 6.2")
check("Sagan K for present power 1.9e13 W", (math.log10(1.9e13) - 6) / 10, 0.728, 5e-3, "Sec. 6.3")

# 2. Area of a one-AU sphere and Type II collector area at eta = 0.20
sphere = 4 * math.pi * AU**2
check("1 AU sphere area (x1e23 m^2)", sphere / 1e23, 2.81, 5e-3, "Sec. 6.4")
check("Type II collector area, eta=0.20 (x1e23 m^2)", 1e26 / (0.20 * S0) / 1e23, 3.67, 5e-3, "Sec. 6.4")
check("Collector area / sphere ratio", (1e26 / (0.20 * S0)) / sphere, 1.31, 5e-3, "Sec. 6.4")

# 3. Radiative equilibrium: PDF formula (quarter-disk factor) and corrected enclosing-shell forms
def t_pdf(alpha, eps, eta):
    return (alpha * S0 * (1 - eta) / (4 * eps * SIGMA)) ** 0.25

def t_shell(alpha, eps, eta, n):
    return (alpha * S0 * (1 - eta) / (n * eps * SIGMA)) ** 0.25

check("PDF bare black body T at 1 AU (K)", t_pdf(1.0, 1.0, 0.0), 278.3, 0.1, "Prop. 2 (as printed)")
check("PDF shell T, eta=0.26 (K)", t_pdf(0.95, 0.90, 0.26), 261.7, 0.1, "Prop. 2 (as printed)")
check("Corrected shell T, n=1, eta=0 (K)", t_shell(1.0, 1.0, 0.0, 1), 393.6, 0.2, "Sec. 7.3 (corrected)")
check("Corrected shell T, n=2, eta=0 (K)", t_shell(1.0, 1.0, 0.0, 2), 331.0, 0.2, "Sec. 7.3 (corrected)")
check("Corrected shell T, n=1, eta=0.26 (K)", t_shell(0.95, 0.90, 0.26, 1), 370.0, 0.5, "Sec. 7.3 (corrected)")
check("Corrected shell T, n=2, eta=0.26 (K)", t_shell(0.95, 0.90, 0.26, 2), 311.2, 0.5, "Sec. 7.3 (corrected)")
# Scaling law T ∝ r^(-1/2): compare shell temperatures at 2 AU and 1 AU with S(r) = S0 / r^2
def t_shell_at(r_au, alpha, eps, eta, n):
    s_r = S0 / r_au**2
    return (alpha * s_r * (1 - eta) / (n * eps * SIGMA)) ** 0.25
check("Scaling T(2 AU)/T(1 AU) = 2^(-1/2)", t_shell_at(2, 1, 1, 0, 1) / t_shell_at(1, 1, 1, 0, 1),
      2 ** -0.5, 1e-12, "Sec. 7.1 scaling")
check("Scaling (1-eta)^(1/4): T(eta=0.5)/T(eta=0)", t_shell(1, 1, 0.5, 1) / t_shell(1, 1, 0.0, 1),
      0.5 ** 0.25, 1e-12, "Sec. 7.1 scaling")


# 4. Radiator area, Section 7.4
A_rad = 1e9 / (0.9 * SIGMA * T_ENV**4)
check("Radiator area 1 GW at 300 K (x1e6 m^2)", A_rad / 1e6, 2.42, 5e-3, "Eq. (11), Sec. 7.4")

# 5. Petela solar exergy limit, Section 9.1
r = T_ENV / T_SUN
petela = 1 - 4 / 3 * r + 1 / 3 * r**4
check("Petela limit", petela, 0.9307, 5e-5, "Sec. 9.1")

# 6. Light time, latency, Proposition 6
tau_1au = 2 * AU / C_LIGHT
check("Round-trip light time 1 AU (min)", tau_1au / 60, 16.6, 0.05, "Sec. 8.1")
check("Minimum control period 6*tau_rt (h)", 6 * tau_1au / 3600, 1.66, 0.01, "Prop. 6, Sec. 8.2")

# 7. Murray exponent (numerical fit of cost minimisation), Proposition 4
Q = np.logspace(-2, 2, 60)
rs = np.logspace(-4, 4, 4000)
a, b = 1.0, 1.0
r_star = np.array([rs[np.argmin(a * q**2 / rs**4 + b * rs**2)] for q in Q])
slope = np.polyfit(np.log(Q), np.log(r_star), 1)[0]
check("Murray exponent (fitted)", slope, 1 / 3, 2e-3, "Prop. 4 (paper: 0.3334)")
check("Murray closed form (2a/b)^(1/6) at a=b=1", (2 * a / b) ** (1 / 6), 1.1225, 1e-3, "Prop. 4")

# 8. Lifespan exponent Phi, Proposition 3 (lambda=0.25, mu=1, theta=0.4, b=0.3)
lam, mu, theta, bb = 0.25, 1.0, 0.4, 0.3
xstar = lam / (mu + lam)
xs = np.linspace(xstar, theta, 200001)
integrand = np.log(mu * np.minimum(xs, bb) / (lam * (1 - xs)))
Phi = np.trapezoid(integrand, xs)
check("Lifespan exponent Phi", Phi, 0.0896, 5e-5, "Prop. 3, Table 5")

# 9. Reversible work floors, Table 5a (magnitudes)
check("H2 work floor (MJ/kg)", 237.1e3 / 2.016e-3 / 1e6, 117.6, 0.05, "Table 5a")
check("H2 work floor (kWh/kg)", 237.1e3 / 2.016e-3 / 3.6e6, 32.7, 0.05, "Table 5a")
check("Glucose work floor (MJ/kg, magnitude)", 2879.0e3 / 180.16e-3 / 1e6, 15.98, 0.01, "Table 5a")
check("O2 work floor (MJ/kg)", 474.2e3 / 32.00e-3 / 1e6, 14.8, 0.05, "Table 5a")
check("CH4 work floor (MJ/kg, magnitude)", 818.1e3 / 16.04e-3 / 1e6, 51.0, 0.05, "Table 5a")

# 10. Biophotovoltaic gap, Section 9.3
ideal_pv = 0.20 * S0 * 1e3              # mW m^-2
check("Ideal PV at 20% (x1e5 mW m^-2)", ideal_pv / 1e5, 2.72, 5e-3, "Sec. 9.3")
check("BPV gap, best reported (1e3 mW m^-2)", ideal_pv / 1e3, 272.0, 1.0, "Sec. 9.3")

# 11. Diffusion-limited biocementation, Proposition 7
c_star = 0.05
pref = 2 * float(erfcinv(c_star))
check("MICP prefactor 2*erfc^-1(0.05)", pref, 2.772, 5e-4, "Prop. 7")
t10 = (0.10)**2 / (4 * float(erfcinv(c_star))**2 * 1e-9)
check("Time to 10 cm at D=1e-9 (days)", t10 / 86400, 15.1, 0.05, "Sec. 12.3")

# 12. Circularity, Section 10.3 (four-person crew, 2.50 kg/day)
D_year = 4 * 2.50 * 365
check("Crew water demand (kg/yr)", D_year, 3650.0, 1e-9, "Sec. 10.3")
check("Replacement at 98% recovery (kg/yr)", (1 - 0.98) * D_year, 73.0, 1e-6, "Table 3b")
check("Replacement at 99.9% recovery (kg/yr)", (1 - 0.999) * D_year, 3.65, 1e-6, "Table 3b")

# 13. Gravity penalty from press-reported BioAsteroid means, Section 11.1
check("Abiotic gravity penalty from rounded means", 29.5 / 2.2, 13.4, 0.05, "Sec. 11.1 (paper: 13.4 from rounded means; 13.6 reported)")
check("Fungal / abiotic microgravity ratio", 11.9 / 2.2, 5.4, 0.05, "Sec. 11.1")

# 14. Atlas counts from Table B1 tags (all 110 concepts), Section 16.2
PHASE_E = {1, 5, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 30}
PHASE_L = {3, 8, 32, 35, 50, 52}
check("Concepts with an E tag", len(PHASE_E), 18, 0, "Table 9 (E=18 tags)")
check("Concepts with an L tag", len(PHASE_L), 6, 0, "Table 9 (L=6 tags)")
check("Concepts with E or L tag", len(PHASE_E | PHASE_L), 24, 0, "Sec. 16.2")

passed = sum(c["pass"] for c in checks)
total = len(checks)
print(f"\n{passed}/{total} checks passed")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "summary.json"), "w", encoding="utf-8") as fh:
    json.dump({"paper": "Biohybrid Infrastructure Viability Across Civilizational Scales v1.0",
               "passed": passed, "total": total, "checks": checks,
               "note": "Analytic and small-scale checks only. Monte Carlo, swarm and viability-kernel results are in analysis.py / results.json."},
              fh, indent=2)
sys.exit(0 if passed == total else 1)
