#!/usr/bin/env python3
"""Limited exact-algebra checks accompanying Kerr_Final_Research_Closure.tex.

This script checks displayed algebraic identities only. It does NOT reproduce
any finite-difference spectrum, Radau integration, redshift-patch experiment,
Teukolsky master equation, or PDE estimate reported in the manuscript.
Requires: SymPy.
"""

from __future__ import annotations

import sympy as sp

M, sigma, r, s = sp.symbols("M sigma r s", positive=True, finite=True)
a2 = M**2 * (1 - sigma**2)
rp = M * (1 + sigma)
rm = M * (1 - sigma)
Delta = (r - rp) * (r - rm)


def assert_zero(expr: sp.Expr, label: str) -> None:
    simplified = sp.factor(sp.simplify(expr))
    if simplified != 0:
        raise AssertionError(f"{label} failed: {simplified}")
    print(f"PASS  {label}")


def main() -> None:
    # (S1) Horizon polynomial P(r_+).
    P = 2 * r**3 - 3 * M * r**2 - M * a2
    p_hor_expected = 2 * M**3 * (sigma + 1) * (sigma**2 + sigma - 1)
    assert_zero(P.subs(r, rp) - p_hor_expected, "horizon polynomial factorisation")

    # (S2) 3D canonical-multiplier horizon discriminant.
    d3 = P.subs(r, rp) ** 2 - M**2 * a2 * (6 * rp**2 + a2)
    d3_expected = M**6 * (sigma + 1) ** 4 * (4 * sigma**2 - 3)
    assert_zero(d3 - d3_expected, "canonical multiplier horizon discriminant")

    # (S3) Carter--Lagrangian discriminant factorisation.
    B = (2 * r - M) * Delta + 2 * M**2 * sigma**2 * r
    bm = 2 * (r - M) ** 3 + 2 * M**3 * sigma**2
    bp = 2 * r * (r - M) ** 2
    assert_zero((B - M * Delta) - bm, "B minus M Delta factor")
    assert_zero((B + M * Delta) - bp, "B plus M Delta factor")
    disc_expected = 4 * r * (r - M) ** 2 * ((r - M) ** 3 + M**3 * sigma**2)
    assert_zero(B**2 - M**2 * Delta**2 - disc_expected, "Carter--Lagrangian discriminant")

    # (S4) Exact trapping-gap factorisation in r_0 = r_+ + s.
    r0 = rp + s
    Delta0 = s * (s + 2 * M * sigma)
    d = 2 * M * sigma
    H = sp.diff(Delta, r).subs(r, r0) * (r0**2 - rp**2) - 4 * r0 * Delta0
    H_factored = -s * (2 * s**2 + 3 * d * s + 2 * rp * d)
    assert_zero(H - H_factored, "trapping-gap numerator factorisation")
    gap_bracket = s**2 + d * s + M * d * (1 - d / (2 * s + d))
    assert_zero(-H / (2 * s + d) - gap_bracket, "closed-form trapping-gap identity")

    # (S5) Riccati coefficient identity.
    w, rho = sp.symbols("w rho", positive=True, cls=sp.Function)
    wr, rr = w(r), rho(r)
    beta = -sp.diff(wr, r) / wr + rr / 2
    gamma = -sp.diff(rr, r) - rr * sp.diff(wr, r) / wr - rr**2 / 2
    assert_zero(gamma - beta * rr + rr**2 + sp.diff(rr, r), "unit-coefficient Riccati identity")

    # (S6) Barrier midpoint identity.
    assert_zero((r - M) * (r - rm) - Delta / 2 - (r - rm) ** 2 / 2,
                "Riccati barrier midpoint identity")

    # (S7) Spin-2 transport integration-by-parts sign identity:
    # d(f/r)/dr = f'/r - f/r^2, hence integral f'/r = [f/r] + integral f/r^2.
    f = sp.Function("f")
    assert_zero(sp.diff(f(r) / r, r) - sp.diff(f(r), r) / r + f(r) / r**2,
                "transport integration-by-parts sign")

    # (S8) Axisymmetric energy coefficient identity.
    theta = sp.symbols("theta", real=True)
    Sigma = r**2 + a2 * sp.cos(theta) ** 2
    lhs = (r**2 + a2) ** 2 - a2 * Delta * sp.sin(theta) ** 2
    rhs = Sigma * (r**2 + a2) + 2 * M * r * a2 * sp.sin(theta) ** 2
    assert_zero(lhs - rhs, "axisymmetric energy coefficient identity")

    # (S9) Leading photon-orbit displacement from the Bardeen--Press--Teukolsky cubic.
    c = 1 / sp.sqrt(3)
    eta = c * sigma
    cubic = (1 + eta) ** 3 - 3 * (1 + eta) + 2 * sp.sqrt(1 - sigma**2)
    # The cubic vanishes through O(sigma^1); the sigma^2 coefficient vanishes for c=1/sqrt(3).
    assert_zero(sp.expand(cubic.series(sigma, 0, 3).removeO()).coeff(sigma, 1),
                "photon-orbit cubic linear coefficient")
    assert_zero(sp.simplify(sp.expand(cubic.series(sigma, 0, 3).removeO()).coeff(sigma, 2)),
                "photon-orbit cubic quadratic coefficient")

    # (S10) Leading trapping-superradiance gap at the prograde equatorial photon orbit.
    c_gap = 2 / sp.sqrt(3) - 1
    s_ph = c_gap * sigma
    gap = (s_ph**2 + 2 * sigma * s_ph + 2 * sigma * (1 - 2 * sigma / (2 * s_ph + 2 * sigma))) / (2 * (1 + sigma))
    gap_leading = sp.simplify(sp.limit(gap / sigma, sigma, 0, dir="+") - (1 - sp.sqrt(3) / 2))
    assert_zero(gap_leading, "leading photon-orbit trapping--superradiance gap")

    print("\nAll limited symbolic checks passed.")
    print("Scope: algebraic identities only; no numerical sweep or PDE estimate was tested.")


if __name__ == "__main__":
    main()
