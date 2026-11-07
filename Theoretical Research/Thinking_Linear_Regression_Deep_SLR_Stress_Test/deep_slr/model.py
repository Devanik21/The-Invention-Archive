"""Thinking Linear Regression (TLR): simple linear regression with a budgeted,
support-aware refinement loop at inference time.

Notation (paper, Section 3):
    y0(x)   = w*x + b                          System-1 intuition (OLS)
    v(x)    = distance outside the training support, in units of sd(x_train)
    a(x)    = 1 - exp(-kappa * v(x))           critic gain in [0, 1)  (kappa=inf: binary)
    A(x)    = anchor, a function of x only     target the loop moves toward
    y_{t+1} = y_t + eta * a(x) * (A(x) - y_t),  t = 0, ..., T-1

Because A does not depend on y_t, the loop has a closed form:
    y_T = A + (1 - eta*a)^T * (y0 - A).
Inside the training support v = 0, so a = 0 and y_T = y0 exactly.
"""
import math

import numpy as np


class ThinkingLinearRegression:
    """Linear regression plus a T-step inference-time critic loop.

    thought_steps : T, the thinking budget (T = 0 reduces to OLS).
    eta           : step size of each thought, in (0, 1].
    kappa         : critic sharpness; np.inf gives a binary critic.
    gamma         : fraction of the global slope carried into the anchor, in [0, 1].
    anchor        : "extrap"   A = boundary mean y + gamma*w*(x - boundary mean x)
                    "residual" A = y0 + boundary mean residual
    boundary_frac, min_boundary : size of the boundary neighbourhood used for anchors.
    """

    def __init__(self, thought_steps=5, eta=0.5, kappa=1.0, gamma=0.5,
                 anchor="extrap", boundary_frac=0.05, min_boundary=10):
        if not (0.0 < eta <= 1.0):
            raise ValueError("eta must lie in (0, 1]")
        if anchor not in ("extrap", "residual"):
            raise ValueError("anchor must be 'extrap' or 'residual'")
        if int(thought_steps) < 0:
            raise ValueError("thought_steps must be >= 0")
        if kappa < 0:
            raise ValueError("kappa must be >= 0")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must lie in [0, 1]")
        self.thought_steps = int(thought_steps)
        self.eta = float(eta)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.anchor = anchor
        self.boundary_frac = float(boundary_frac)
        self.min_boundary = int(min_boundary)

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.size != y.size or x.size < 3:
            raise ValueError("need at least 3 paired observations")
        design = np.column_stack([np.ones_like(x), x])
        (self.b_, self.w_), *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - (self.b_ + self.w_ * x)

        self.x_min_, self.x_max_ = float(x.min()), float(x.max())
        self.sd_x_ = float(x.std(ddof=1))
        if self.sd_x_ == 0.0:
            raise ValueError("x has zero variance")

        m = max(self.min_boundary, int(math.ceil(self.boundary_frac * x.size)))
        m = min(m, x.size)
        order = np.argsort(x)
        lo, hi = order[:m], order[-m:]
        self.xbar_lo_, self.xbar_hi_ = float(x[lo].mean()), float(x[hi].mean())
        self.ybar_lo_, self.ybar_hi_ = float(y[lo].mean()), float(y[hi].mean())
        self.rbar_lo_, self.rbar_hi_ = float(resid[lo].mean()), float(resid[hi].mean())
        self.n_train_ = int(x.size)
        self.is_fitted_ = True
        return self

    def critic(self, x):
        """Return (y0, A, v, a) for an array of inputs. Does not modify state."""
        self._check_fitted()
        x = np.asarray(x, dtype=float).ravel()
        y0 = self.b_ + self.w_ * x
        v = (np.maximum(x - self.x_max_, 0.0) + np.maximum(self.x_min_ - x, 0.0)) / self.sd_x_
        hi_side = x > self.x_max_
        if self.anchor == "extrap":
            ybar = np.where(hi_side, self.ybar_hi_, self.ybar_lo_)
            xbar = np.where(hi_side, self.xbar_hi_, self.xbar_lo_)
            A = ybar + self.gamma * self.w_ * (x - xbar)
        else:
            rbar = np.where(hi_side, self.rbar_hi_, self.rbar_lo_)
            A = y0 + rbar
        if np.isinf(self.kappa):
            a = np.where(v > 0.0, 1.0, 0.0)
        else:
            a = 1.0 - np.exp(-self.kappa * v)
        return y0, A, v, a

    def predict(self, x):
        """Closed-form prediction after T thoughts (vectorised)."""
        y0, A, _, a = self.critic(x)
        g = np.clip(self.eta * a, 0.0, 1.0)
        return A + np.power(1.0 - g, self.thought_steps) * (y0 - A)

    def predict_iterative(self, x):
        """Explicit T-step loop, used to verify the closed form."""
        y0, A, _, a = self.critic(x)
        g = np.clip(self.eta * a, 0.0, 1.0)
        y = y0.copy()
        for _ in range(self.thought_steps):
            y = y + g * (A - y)
        return y

    def trace(self, x_scalar):
        """Thought trajectory for one input: returns (trajectory, anchor, v, a)."""
        y0, A, v, a = self.critic(np.array([x_scalar]))
        g = min(self.eta * float(a[0]), 1.0)
        y = float(y0[0])
        traj = [y]
        for _ in range(self.thought_steps):
            y = y + g * (float(A[0]) - y)
            traj.append(y)
        return np.asarray(traj), float(A[0]), float(v[0]), float(a[0])

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("call fit() before predict()")
