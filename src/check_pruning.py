"""The pruned DP must return bit-identical results to the dense DP."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from dpf import fuse_dp, layers_from
from env import PathProblem, build_scenarios


def main():
    sc = build_scenarios()
    rng = np.random.default_rng(42)
    n_cases = 0
    for name in sc:
        for n in (10, 20):
            prob = PathProblem(sc[name], n)
            for K in (6, 11, 30, 100):
                for _ in range(3 if K == 100 else 8):
                    X = prob.lb + rng.random((K, prob.dim)) * (prob.ub - prob.lb)
                    if K <= 11:   # local-fusion-like: perturbations of one path
                        X = np.clip(X[0] + rng.normal(0, 0.05, X.shape) * (prob.ub - prob.lb),
                                    prob.lb, prob.ub)
                    L = layers_from(prob, X)
                    x1, J1 = fuse_dp(prob, L, prune=False)
                    x2, J2 = fuse_dp(prob, L, prune=True)
                    assert J1 == J2 and np.array_equal(x1, x2), (name, n, K, J1, J2)
                    n_cases += 1
    print(f"pruned DP bit-identical to dense DP in {n_cases} cases")
    bench = dict(n_cases=n_cases)
    for K in (11, 100):
        res = {False: [], True: []}
        for name in sc:                       # average over all scenarios
            prob = PathProblem(sc[name], 20)
            X = prob.lb + rng.random((K, prob.dim)) * (prob.ub - prob.lb)
            L = layers_from(prob, X)
            reps = 3 if K == 100 else 30
            for prune in (False, True):
                t = time.perf_counter()
                for _ in range(reps):
                    fuse_dp(prob, L, prune=prune)
                res[prune].append((time.perf_counter() - t) / reps)
        for prune in (False, True):
            print(f"n=20 K={K:3d} prune={prune!s:5s}: {np.mean(res[prune]) * 1e3:8.1f} ms per DP "
                  f"(mean over {len(sc)} scenarios, single process)")
        bench[f"K{K}_dense_s"] = float(np.mean(res[False]))
        bench[f"K{K}_pruned_s"] = float(np.mean(res[True]))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes",
                       "pruning_benchmark.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(bench, fh, indent=1)


if __name__ == "__main__":
    main()
