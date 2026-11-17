"""Pilot comparison of design variants (10 runs, hard instances)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from core import Evaluator
from env import PathProblem, build_scenarios
from variants import VARIANTS, dp_only, dpf_pso2

_SC = None


def job(t):
    global _SC
    inst, var, r, fes = t
    if _SC is None:
        _SC = build_scenarios()
    name, n = inst.split("_n")
    prob = PathProblem(_SC[name], int(n))
    ev = Evaluator(prob.evaluate, fes)
    rng = np.random.default_rng(20270201 + r)
    if var == "DP-only":
        dp_only(prob, ev, rng)
    else:
        dpf_pso2(prob, ev, rng, **VARIANTS[var])
    P, h = prob.plane_to_points(ev.best_x[None])
    feas = prob.components(P[0], h[0])["n_viol"] == 0
    return inst, var, ev.best_f, feas


def main():
    insts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["S2_n20", "S3_n10", "S3_n20", "S4_n20"]
    vars_ = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS) + ["DP-only"]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    tasks = [(i, v, r, 30000 if i.endswith("n10") else 60000) for i in insts for v in vars_ for r in range(runs)]
    t0 = time.time()
    res = {}
    with Pool(20) as pool:
        for inst, var, bf, feas in pool.imap_unordered(job, tasks):
            res.setdefault((inst, var), []).append((bf, feas))
    for inst in insts:
        print("==", inst)
        for v in vars_:
            a = np.array([x[0] for x in res[(inst, v)]])
            fe = np.mean([x[1] for x in res[(inst, v)]]) * 100
            print(f"  {v:24s} mean={a.mean():9.1f} std={a.std():8.1f} best={a.min():8.1f} "
                  f"median={np.median(a):8.1f} feas={fe:5.1f}%")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
