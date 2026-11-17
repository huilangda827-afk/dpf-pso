"""Sanity check of the CMA-ES baseline (not part of the paper pipeline):
does another initial step size, no restarts or a larger population change
its ranking?  Uses tuning seeds (30270000 + r) on two design instances.

    python notes/check_cmaes.py
"""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

from algorithms import cmaes, de
from core import Evaluator
from env import make_problem

VARIANTS = {"default (sigma0 0.3, IPOP)": dict(),
            "sigma0 0.1, IPOP": dict(sigma0=0.1),
            "sigma0 0.5, IPOP": dict(sigma0=0.5),
            "sigma0 0.3, no restarts": dict(ipop=False),
            "sigma0 0.3, popsize 50, IPOP": dict(popsize=50)}


def job(t):
    inst, name, r = t
    prob = make_problem(inst)
    ev = Evaluator(prob.evaluate, 3000 * prob.n)
    rng = np.random.default_rng(30270000 + r)
    if name == "DE":
        de(prob, ev, rng)
    else:
        cmaes(prob, ev, rng, **VARIANTS[name])
    return inst, name, r, ev.best_f


def main():
    tasks = [(i, v, r) for i in ("S1_n10", "S3_n10") for v in list(VARIANTS) + ["DE"] for r in range(4)]
    with Pool(4) as p:
        res = p.map(job, tasks)
    for inst in ("S1_n10", "S3_n10"):
        for v in list(VARIANTS) + ["DE"]:
            x = [f for i, n, r, f in res if i == inst and n == v]
            print(f"{inst} {v:32s} median {np.median(x):8.0f}  runs {np.round(x).astype(int).tolist()}")


if __name__ == "__main__":
    main()
