"""CMA-ES restart settings (check that selected the setting used in the paper).

The first CMA-ES runs used pycma's strict default tolerances, so within
3000n FEs the first run never stopped and IPOP never restarted.  This script
compares stopping rules that are scaled to the cost range on S1_n10, S3_n10
and S3_n20 with four tuning seeds; the selected setting ("tolx 1e-3,
tolfun 1, stag 200") is the default of cmaes() in src/algorithms.py.
Output: notes/cmaes_restart_check.json
"""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

from core import Evaluator
from env import make_problem


def cma_run(prob, ev, rng, sigma0=0.3, tolx=1e-12, tolfun=1e-11, tolfunhist=1e-12, stag=None,
            inc=2.0, bipop=False):
    import cma
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    lam0 = 4 + int(3 * np.log(D))
    lam = lam0
    restarts = 0
    while not ev.exhausted:
        opts = {"bounds": [0.0, 1.0], "popsize": lam, "seed": int(rng.integers(1, 2 ** 31 - 1)),
                "verbose": -9, "verb_log": 0, "verb_disp": 0, "maxfevals": int(ev.remaining),
                "tolfun": tolfun, "tolx": tolx, "tolfunhist": tolfunhist}
        if stag is not None:
            opts["tolstagnation"] = stag
        es = cma.CMAEvolutionStrategy(rng.random(D), sigma0, opts)
        while not es.stop() and not ev.exhausted:
            Y = np.asarray(es.ask())
            f = ev(lb + np.clip(Y, 0.0, 1.0) * (ub - lb))
            if not np.all(np.isfinite(f)):
                break
            es.tell(list(Y), f.tolist())
        restarts += 1
        lam = int(lam * inc)
    return restarts


VARIANTS = {
    "pycma default tolerances (first runs, discarded)": dict(),
    "tolx 1e-4, tolfun 1e-1": dict(tolx=1e-4, tolfun=1e-1, tolfunhist=1e-1),
    "tolx 1e-3, tolfun 1, stag 200": dict(tolx=1e-3, tolfun=1.0, tolfunhist=1.0, stag=200),
    "tolx 1e-4, tolfun 1e-1, sigma0 0.1": dict(tolx=1e-4, tolfun=1e-1, tolfunhist=1e-1, sigma0=0.1),
}


def job(t):
    inst, name, r = t
    prob = make_problem(inst)
    ev = Evaluator(prob.evaluate, 3000 * prob.n)
    k = cma_run(prob, ev, np.random.default_rng(30270000 + r), **VARIANTS[name])
    return inst, name, r, ev.best_f, k


def main():
    import json
    insts = ("S1_n10", "S3_n10", "S3_n20")
    tasks = [(i, v, r) for i in insts for v in VARIANTS for r in range(4)]
    with Pool(4) as p:
        res = p.map(job, tasks)
    out = {}
    for inst in insts:
        for v in VARIANTS:
            x = [f for i, n, r, f, k in res if i == inst and n == v]
            ks = [k for i, n, r, f, k in res if i == inst and n == v]
            out.setdefault(inst, {})[v] = dict(median=float(np.median(x)), runs=[float(f) for f in x], starts=ks)
            print(f"{inst} {v:38s} median {np.median(x):8.0f} runs {np.round(x).astype(int).tolist()} starts {ks}")
    out["selected"] = "tolx 1e-3, tolfun 1, stag 200"
    out["seeds"] = [30270000 + r for r in range(4)]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmaes_restart_check.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
