"""Toy generators with known ground truth, and real datasets bundled offline."""
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_diabetes

# Toy specs. Scoring uses the noise-free f(x), so the metric is error to the true mean.
TOYS = {
    "linear": dict(f=lambda x: 2 * x + 1, sigma=1.0, train=(0, 10), ext=(10, 15),
                   title="Linear (control)"),
    "saturating": dict(f=lambda x: 10 * (1 - np.exp(-x / 3)), sigma=0.5, train=(0, 8),
                       ext=(8, 16), title="Saturating growth"),
    "logistic": dict(f=lambda x: 12 / (1 + np.exp(-0.8 * (x - 6))), sigma=0.4,
                     train=(0, 8), ext=(8, 14), title="Logistic (sigmoidal)"),
    "concave": dict(f=lambda x: 3 * x - 0.15 * x ** 2, sigma=0.5, train=(0, 8),
                    ext=(8, 12), title="Concave quadratic (peak beyond data)"),
    "regime_shift": dict(f=lambda x: np.where(x <= 6, x, 6 + 0.3 * (x - 6)), sigma=0.3,
                         train=(0, 6), ext=(6, 12), title="Regime shift (slope 1 to 0.3)"),
}

REAL = {
    "engel": "Engel curve: food expenditure vs income (statsmodels, n=235)",
    "diabetes_bmi": "Diabetes progression vs standardised BMI (scikit-learn, n=442)",
}


def toy_split(name, seed, n_train=200, n_eval=200):
    spec = TOYS[name]
    rng = np.random.default_rng([list(TOYS).index(name), seed])
    xtr = rng.uniform(*spec["train"], n_train)
    ytr = spec["f"](xtr) + rng.normal(0.0, spec["sigma"], n_train)
    xin = rng.uniform(*spec["train"], n_eval)
    xext = rng.uniform(*spec["ext"], n_eval)
    return dict(xtr=xtr, ytr=ytr, xin=xin, yin=spec["f"](xin),
                xext=xext, yext=spec["f"](xext))


def load_real(name):
    if name == "engel":
        d = sm.datasets.engel.load_pandas().data
        return d["income"].to_numpy(float), d["foodexp"].to_numpy(float)
    if name == "diabetes_bmi":
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        return X["bmi"].to_numpy(float), y.to_numpy(float)
    raise KeyError(name)


def real_split(name, seed, q=0.8):
    """Train on the lower q-quantile of x (the support). 20% of the support is held out
    for an in-distribution score; the top (1-q) of x is the extrapolation test."""
    x, y = load_real(name)
    cut = np.quantile(x, q)
    pool = x <= cut
    xp, yp = x[pool], y[pool]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(xp.size)
    k = int(0.8 * xp.size)
    tr, hold = idx[:k], idx[k:]
    ext = x > cut
    return dict(xtr=xp[tr], ytr=yp[tr], xin=xp[hold], yin=yp[hold],
                xext=x[ext], yext=y[ext])
