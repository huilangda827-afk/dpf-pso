"""Equal-budget parameter tuning on the design scenarios only.

Every tuned algorithm gets the same budget: the default configuration plus 24
configurations drawn at random from its parameter space, each run with 5
seeds on the 8 design instances S1-S4 x n in {10, 20} (1,000 runs per
algorithm).  Score: rank of the median cost among the 25 configurations on
each instance, averaged over the 8 instances; the lowest mean rank wins (ties:
lower mean normalised median).  Held-out scenarios are never used.

    python src/tune.py --workers 20
Output: results/tuning/<algo>.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import rankdata

from algorithms import ALGOS, config_factory
from core import Evaluator
from env import CostParams, PathProblem, get_scenario, parse_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = [f"S{k}_n{n}" for n in (10, 20) for k in (1, 2, 3, 4)]
TUNE_SEED = 30270000           # distinct from pilot (20270201+) and final (20271101+) seeds
N_RANDOM, N_SEEDS = 24, 5


def u(a, b):
    return ("uniform", a, b)


def lu(a, b):
    return ("loguniform", a, b)


def ch(*vals):
    return ("choice",) + vals


SPACES = {
    "DE": dict(strategy=ch("rand1bin", "best1bin", "ctb1bin"), F=u(0.3, 1.0), CR=u(0.0, 1.0),
               NP=ch(20, 50, 100, "5D", "10D")),
    "LSHADE": dict(r_ninit=ch(5.0, 10.0, 18.0, 25.0), p_best=u(0.05, 0.25), H=ch(5, 6, 10, 20),
                   r_arc=u(1.0, 3.0)),
    "SPSO": dict(N=ch(30, 50, 100, 200, 500), c1=u(1.0, 2.5), c2=u(1.0, 2.5), w0=u(0.5, 1.2),
                 w_end=lu(0.005, 0.5), alpha=u(0.1, 1.0)),
    "IDP": dict(K=ch(10, 15, 20, 30, 40), r_min=lu(1e-4, 1e-1), N0=ch(20, 50, 100)),
    "DPF-PSO": dict(g_random=ch(0, 25, 50, 100), g_max=ch(25, 50), g_every=ch(0, 2500, 5000, 10000),
                    l_every=ch(100, 250, 500, 1000), n_elite=ch(2, 5, 10), n_perturb=ch(3, 6, 12),
                    sigma0=lu(0.01, 0.2), adapt=ch(True, False)),
    # added in the third revision (same budget and protocol)
    "CMAES": dict(sigma0=lu(0.05, 0.6), pop_mult=ch(1.0, 2.0, 4.0, 8.0), tolx=lu(1e-4, 1e-2),
                  tolfun=lu(0.1, 10.0), tolstagnation=ch(100, 200, 400)),
}
DEFAULTS = {
    "DE": dict(strategy="rand1bin", F=0.5, CR=0.9, NP=50),
    "LSHADE": dict(r_ninit=18.0, p_best=0.11, H=6, r_arc=2.6),
    "SPSO": dict(N=50, c1=1.5, c2=1.5, w0=1.0, w_end=0.98 ** 200, alpha=0.5),
    "IDP": dict(K=20, r_min=1e-3, N0=50),
    "DPF-PSO": dict(g_random=50, g_max=50, g_every=5000, l_every=250, n_elite=5, n_perturb=6,
                    sigma0=0.05, adapt=True),
    "CMAES": dict(sigma0=0.3, pop_mult=1.0, tolx=1e-3, tolfun=1.0, tolstagnation=200),
}


def sample(space, rng):
    out = {}
    for k, spec in space.items():
        if spec[0] == "uniform":
            out[k] = float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == "loguniform":
            out[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            v = spec[1:][int(rng.integers(len(spec) - 1))]
            out[k] = v.item() if hasattr(v, "item") else v
    return out


def configs_for(algo):
    rng = np.random.default_rng(abs(hash(algo)) % (2 ** 31) if False else
                                {"DE": 11, "LSHADE": 12, "SPSO": 13, "IDP": 14, "DPF-PSO": 15, "CMAES": 16}[algo])
    return [DEFAULTS[algo]] + [sample(SPACES[algo], rng) for _ in range(N_RANDOM)]


def job(t):
    algo, ci, cfg, inst, r = t
    name, n = parse_instance(inst)
    prob = PathProblem(get_scenario(name), n, CostParams())
    fn, enc, _ = ALGOS[algo]
    kw = config_factory(algo, cfg)(prob)
    ev = Evaluator(prob.evaluate if enc != "sph" else prob.evaluate_sph, 3000 * n)
    fn(prob, ev, np.random.default_rng(TUNE_SEED + r), **kw)
    return algo, ci, inst, r, ev.best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--algos", nargs="+", default=list(SPACES))
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "results", "tuning")
    os.makedirs(outdir, exist_ok=True)
    cfgs = {a: configs_for(a) for a in args.algos}
    tasks = [(a, ci, cfg, inst, r) for a in args.algos for ci, cfg in enumerate(cfgs[a])
             for inst in DESIGN for r in range(N_SEEDS)]
    tasks.sort(key=lambda t: -(int(t[3].split("_n")[1]) * (3 if t[0] in ("DPF-PSO", "IDP") else 1)))
    res = {}
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for k, (a, ci, inst, r, bf) in enumerate(pool.imap_unordered(job, tasks, chunksize=1), 1):
            res[(a, ci, inst, r)] = bf
            if k % 250 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"{k}/{len(tasks)} runs, {el:.0f}s, ETA {el / k * (len(tasks) - k):.0f}s", flush=True)
    for a in args.algos:
        C = len(cfgs[a])
        med = np.array([[np.median([res[(a, ci, inst, r)] for r in range(N_SEEDS)]) for inst in DESIGN]
                        for ci in range(C)])                      # (C, 8)
        ranks = np.array([rankdata(med[:, j]) for j in range(len(DESIGN))]).T
        mrank = ranks.mean(axis=1)
        norm = (med / med.min(axis=0)).mean(axis=1)
        best = int(np.lexsort((norm, mrank))[0])
        out = dict(algo=a, space={k: list(v) for k, v in SPACES[a].items()}, configs=cfgs[a],
                   instances=DESIGN, seeds=[TUNE_SEED + r for r in range(N_SEEDS)],
                   median_cost=med.tolist(), mean_rank=mrank.tolist(), norm_median=norm.tolist(),
                   selected_index=best, selected=cfgs[a][best], default_rank=float(mrank[0]),
                   selected_rank=float(mrank[best]))
        with open(os.path.join(outdir, f"{a}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"{a}: selected config {best} (mean rank {mrank[best]:.2f}; default {mrank[0]:.2f}): "
              f"{cfgs[a][best]}")


if __name__ == "__main__":
    main()
