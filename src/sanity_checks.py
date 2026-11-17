"""Correctness checks for the cost decomposition, the DP fusion operator and
all optimisers (run before any experiment).

1. DP cost equals the full-path evaluation of the returned path.
2. DP result equals exhaustive search over all candidate combinations (n = 5).
3. DP result is never worse than the best parent.
4. Pruned DP is bit-identical to the dense DP.
5. GLS incremental costs equal full re-evaluation.
6. Every optimiser runs end-to-end and uses exactly the FE budget.
"""
import itertools
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from algorithms import ALGOS
from core import Evaluator
from dpf import fuse_dp, layers_from
from env import CostParams, PathProblem, get_scenario


def check_dp(prob, rng, n_par=3, trials=10):
    worst_gap = 0.0
    for _ in range(trials):
        X = prob.lb + rng.random((n_par, prob.dim)) * (prob.ub - prob.lb)
        fX = prob.evaluate(X)
        layers = layers_from(prob, X)
        x, J = fuse_dp(prob, layers, prune=False)
        x2, J2 = fuse_dp(prob, layers, prune=True)
        assert J == J2 and np.array_equal(x, x2)
        Jfull = float(prob.evaluate(x[None])[0])
        assert abs(J - Jfull) <= 1e-6 * max(1.0, abs(Jfull)), (J, Jfull)
        assert J <= fX.min() + 1e-6 * max(1.0, abs(fX.min())), (J, fX.min())
        if prob.n <= 6:
            combos = np.array(list(itertools.product(*[range(len(L)) for L in layers])))
            Xall = np.empty((len(combos), prob.dim))
            for j, L in enumerate(layers):
                Xall[:, prob.var_idx[j]] = L[combos[:, j]]
            brute = prob.evaluate(Xall).min()
            assert abs(brute - J) <= 1e-6 * max(1.0, abs(brute)), (brute, J)
        worst_gap = max(worst_gap, abs(J - Jfull))
    return worst_gap


class CheckingEvaluator(Evaluator):
    """Verifies that every best solution reported via charge() has the cost
    claimed by the optimiser (catches errors in incremental evaluation)."""

    def __init__(self, fun, max_fes, prob):
        super().__init__(fun, max_fes)
        self.prob = prob
        self.checked = 0

    def charge(self, fes, x=None, fx=float("inf")):
        if x is not None and np.isfinite(fx):
            true = float(self.prob.evaluate(np.asarray(x)[None])[0])
            assert abs(true - fx) <= 1e-6 * max(1.0, abs(true)), (true, fx)
            self.checked += 1
        super().charge(fes, x, fx)


def main():
    rng = np.random.default_rng(0)
    band = "--band" in sys.argv          # altitude band enforced along segments
    params = CostParams(band_along=band)
    if band:
        print("cost model variant: altitude band enforced along every segment")
    for name in ["S1", "S2", "S3", "S4", "S5", "S6", "H03", "C1"]:
        for n in (5, 10, 20):
            prob = PathProblem(get_scenario(name), n, params)
            gap = check_dp(prob, rng, n_par=3 if n == 5 else 6, trials=6)
            print(f"{name} n={n:2d}: DP == full evaluation (max abs gap {gap:.2e}); "
                  f"{'brute-force verified' if n <= 6 else 'parent bound verified'}; pruned == dense")
    if band:
        for name, (fn, enc, kwf) in ALGOS.items():
            prob = PathProblem(get_scenario("S4"), 10, params)
            fun = prob.evaluate if enc != "sph" else prob.evaluate_sph
            ev = CheckingEvaluator(fun, 6000, prob) if enc != "sph" else Evaluator(fun, 6000)
            fn(prob, ev, np.random.default_rng(1), **kwf(prob))
            assert ev.used == ev.max, (name, ev.used)
            print(f"{name:15s} best={ev.best_f:10.2f}  FEs={ev.used}  checked={getattr(ev, 'checked', 0)}")
        return

    prob = PathProblem(get_scenario("S2"), 10)
    X = prob.lb + rng.random((50, prob.dim)) * (prob.ub - prob.lb)
    t = time.perf_counter()
    for _ in range(50):
        prob.evaluate(X)
    print(f"evaluate 50 paths (n=10): {(time.perf_counter() - t) / 50 * 1e3:.3f} ms")

    for name, (fn, enc, kwf) in ALGOS.items():
        prob = PathProblem(get_scenario("S4"), 10)
        fun = prob.evaluate if enc != "sph" else prob.evaluate_sph
        ev = CheckingEvaluator(fun, 6000, prob) if enc != "sph" else Evaluator(fun, 6000)
        t = time.perf_counter()
        fn(prob, ev, np.random.default_rng(1), **kwf(prob))
        assert ev.used == ev.max, (name, ev.used)
        chk = getattr(ev, "checked", 0)
        print(f"{name:15s} best={ev.best_f:10.2f}  FEs={ev.used}  charged={ev.dp_fes:5d}  "
              f"checked={chk:5d}  time={time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
