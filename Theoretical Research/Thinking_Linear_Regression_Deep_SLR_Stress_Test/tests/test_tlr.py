"""Verification tests for the TLR closed form and its guarantees.
Run:  python tests/test_tlr.py    (or)    python -m pytest -q
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from deep_slr import ThinkingLinearRegression as TLR  # noqa: E402
from deep_slr.baselines import OLS  # noqa: E402

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, 150)
Y = 3 * X - 2 + rng.normal(0, 1, 150)


def test_inside_support_equals_ols_exactly():
    xin = rng.uniform(X.min(), X.max(), 500)
    ols = OLS().fit(X, Y).predict(xin)
    for anchor in ("extrap", "residual"):
        for T in (1, 5, 30):
            m = TLR(thought_steps=T, anchor=anchor, eta=0.7, kappa=2.0).fit(X, Y)
            assert np.allclose(m.predict(xin), ols, atol=1e-12)


def test_zero_steps_equals_ols_everywhere():
    xs = np.linspace(-5, 20, 300)
    assert np.allclose(TLR(thought_steps=0).fit(X, Y).predict(xs),
                       OLS().fit(X, Y).predict(xs), atol=1e-12)


def test_closed_form_equals_iterative_loop():
    xs = np.linspace(-6, 18, 400)
    for anchor in ("extrap", "residual"):
        for kappa in (0.5, 2.0, np.inf):
            for eta in (0.25, 1.0):
                for T in (1, 3, 17):
                    m = TLR(thought_steps=T, eta=eta, kappa=kappa, gamma=0.4,
                            anchor=anchor).fit(X, Y)
                    assert np.allclose(m.predict(xs), m.predict_iterative(xs), atol=1e-10)


def test_trace_converges_to_anchor():
    m = TLR(thought_steps=200, eta=0.5, kappa=1.0, gamma=0.3).fit(X, Y)
    traj, A, v, a = m.trace(14.0)
    assert v > 0 and 0 < a < 1
    assert abs(traj[-1] - m.predict(np.array([14.0]))[0]) < 1e-10
    assert abs(traj[-1] - A) < 1e-6 * max(1.0, abs(A))


def test_gap_to_anchor_is_monotone_in_thoughts():
    gaps = []
    for T in range(0, 12):
        m = TLR(thought_steps=T, eta=0.5, kappa=1.0, gamma=0.3).fit(X, Y)
        traj, A, _, _ = m.trace(14.0)
        gaps.append(abs(traj[-1] - A))
    assert all(g2 <= g1 + 1e-12 for g1, g2 in zip(gaps, gaps[1:]))


def test_binary_critic_is_all_or_nothing():
    m = TLR(thought_steps=3, kappa=np.inf, eta=1.0).fit(X, Y)
    ols = OLS().fit(X, Y)
    assert abs(m.predict(np.array([5.0]))[0] - ols.predict(np.array([5.0]))[0]) < 1e-12
    assert m.critic(np.array([12.0]))[3][0] == 1.0


def test_parameter_validation():
    for bad in (dict(eta=0.0), dict(eta=1.5), dict(gamma=2.0), dict(kappa=-1.0),
                dict(anchor="nope"), dict(thought_steps=-1)):
        try:
            TLR(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all tests passed")
