"""Baseline predictors. All expose fit(x, y) and predict(x)."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class OLS:
    def fit(self, x, y):
        x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
        design = np.column_stack([np.ones_like(x), x])
        (self.b, self.w), *_ = np.linalg.lstsq(design, y, rcond=None)
        return self

    def predict(self, x):
        return self.b + self.w * np.asarray(x, float).ravel()


class ClippedOLS(OLS):
    """OLS whose predictions are clipped to the training target range."""
    def fit(self, x, y):
        super().fit(x, y)
        y = np.asarray(y, float)
        self.lo, self.hi = float(y.min()), float(y.max())
        return self

    def predict(self, x):
        return np.clip(super().predict(x), self.lo, self.hi)


class SklearnWrap:
    """Standardises x on training data, then delegates to a scikit-learn estimator."""
    def __init__(self, est):
        self.est = est

    def fit(self, x, y):
        x = np.asarray(x, float).ravel()
        self.mu_, self.sd_ = x.mean(), x.std(ddof=1)
        self.est.fit(self._z(x).reshape(-1, 1), np.asarray(y, float))
        return self

    def _z(self, x):
        return (np.asarray(x, float).ravel() - self.mu_) / self.sd_

    def predict(self, x):
        return np.asarray(self.est.predict(self._z(x).reshape(-1, 1))).ravel()


def make_huber():
    return SklearnWrap(HuberRegressor(max_iter=1000))


def make_quadratic():
    return SklearnWrap(make_pipeline(PolynomialFeatures(2), LinearRegression()))


def make_knn(k=10):
    return SklearnWrap(KNeighborsRegressor(n_neighbors=k))


def make_gpr():
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-5, 1e1))
    return SklearnWrap(GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                random_state=0))
