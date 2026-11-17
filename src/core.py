"""Budget-aware evaluator shared by all optimisers."""
from __future__ import annotations

import math
import time

import numpy as np


class Evaluator:
    """Counts function evaluations (FEs), tracks the best-so-far solution and
    stops exactly at ``max_fes``: rows beyond the remaining budget are not
    evaluated and receive +inf."""

    def __init__(self, fun, max_fes: int):
        self.fun = fun
        self.max = int(max_fes)
        self.used = 0
        self.best_f = math.inf
        self.best_x = None
        self.trace_fe = [0]
        self.trace_f = [math.inf]
        self.t0 = time.perf_counter()
        self.trace_t = [0.0]
        self.dp_fes = 0      # FEs charged to fusion operators (for reporting)

    @property
    def remaining(self) -> int:
        return self.max - self.used

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max

    @property
    def frac(self) -> float:
        return min(1.0, self.used / self.max)

    def __call__(self, X):
        X = np.atleast_2d(X)
        f = np.full(len(X), np.inf)
        k = min(len(X), self.remaining)
        if k <= 0:
            return f
        f[:k] = self.fun(X[:k])
        self.used += k
        i = int(np.argmin(f[:k]))
        if f[i] < self.best_f:
            self.best_f = float(f[i])
            self.best_x = X[i].copy()
        self.trace_fe.append(self.used)
        self.trace_f.append(self.best_f)
        self.trace_t.append(time.perf_counter() - self.t0)
        return f

    def charge(self, fes: int, x=None, fx=math.inf):
        """Account for work done outside __call__ (e.g. the DP fusion)."""
        self.used += int(fes)
        self.dp_fes += int(fes)
        if x is not None and fx < self.best_f:
            self.best_f = float(fx)
            self.best_x = np.array(x, float).copy()
        self.trace_fe.append(self.used)
        self.trace_f.append(self.best_f)
        self.trace_t.append(time.perf_counter() - self.t0)

    def _at(self, values, n_points):
        grid = np.linspace(0, self.max, n_points)
        fe = np.asarray(self.trace_fe)
        idx = np.searchsorted(fe, grid, side="right") - 1
        return np.asarray(values)[np.clip(idx, 0, len(fe) - 1)]

    def curve(self, n_points: int = 101):
        """Best-so-far cost at evenly spaced FE checkpoints 0..max."""
        return self._at(self.trace_f, n_points)

    def time_curve(self, n_points: int = 101):
        """Wall-clock time (s) at which each FE checkpoint was reached."""
        return self._at(self.trace_t, n_points)


def levy(rng, shape, beta=1.5):
    """Mantegna's algorithm for Levy-stable steps (as in HHO reference code)."""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(shape) * sigma
    v = rng.standard_normal(shape)
    return 0.01 * u / np.abs(v) ** (1 / beta)
