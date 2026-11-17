"""Controlled timing with convergence curves (journal version; design log,
section 16).

Same set-up as timing_benchmark.py: one worker per performance core, each
pinned to one logical processor of a different P-core, every worker handles
whole instances and runs all methods in interleaved order.  In addition to
the total CPU time, every run stores the best-so-far cost and the CPU time
(process time of the worker) at the 101 FE checkpoints, so that convergence
can be compared by CPU time and not only by evaluations.

    python src/timing_curves.py --tag timing_curves --runs 5

Output: results/<tag>/<instance>__<method>.npz with best_f, curves
(best-so-far cost per checkpoint), ccurves (CPU seconds per checkpoint),
cpu and wall (seconds per run).  Run on an otherwise idle machine.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import sys
import time
from multiprocessing import Manager, Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import psutil

from algorithms import algo_spec
from core import Evaluator
from cpuinfo_win import performance_cores
from env import CostParams, PathProblem, get_scenario, parse_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTS = [f"H{k:02d}_n{n}" for n in (10, 20) for k in range(1, 9)]
METHODS = ["SPSO", "DE", "LSHADE", "CMAES", "IDP", "DPF-PSO"]
SEED = 50270000          # fresh seed family for the timing-curve runs


class CpuEvaluator(Evaluator):
    """Evaluator that also records the process CPU time of every checkpoint."""

    def __init__(self, fun, max_fes):
        super().__init__(fun, max_fes)
        self.c0 = time.process_time()
        self.trace_c = [0.0]

    def __call__(self, X):
        f = super().__call__(X)
        if len(self.trace_c) < len(self.trace_fe):
            self.trace_c.append(time.process_time() - self.c0)
        return f

    def charge(self, fes, x=None, fx=math.inf):
        super().charge(fes, x, fx)
        self.trace_c.append(time.process_time() - self.c0)

    def cpu_curve(self, n_points=101):
        return self._at(self.trace_c, n_points)


def _pin(q):
    lp = q.get()
    psutil.Process().cpu_affinity([lp])
    os.environ["PINNED_LP"] = str(lp)


def bench_instance(args):
    inst, algos, runs, outdir = args
    name, n = parse_instance(inst)
    prob = PathProblem(get_scenario(name), n, CostParams())
    recs = {a: dict(best_f=[], curves=[], ccurves=[], cpu=[], wall=[]) for a in algos}
    for r in range(runs):
        for a in algos:
            fn, enc, kwf = algo_spec(a)
            ev = CpuEvaluator(prob.evaluate if enc != "sph" else prob.evaluate_sph, 3000 * n)
            rng = np.random.default_rng(SEED + r)
            t0, c0 = time.perf_counter(), time.process_time()
            fn(prob, ev, rng, **kwf(prob))
            recs[a]["wall"].append(time.perf_counter() - t0)
            recs[a]["cpu"].append(time.process_time() - c0)
            recs[a]["best_f"].append(ev.best_f)
            recs[a]["curves"].append(ev.curve(101))
            recs[a]["ccurves"].append(ev.cpu_curve(101))
    for a in algos:
        np.savez(os.path.join(outdir, f"{inst}__{a}.npz"),
                 best_f=np.array(recs[a]["best_f"]), curves=np.stack(recs[a]["curves"]),
                 ccurves=np.stack(recs[a]["ccurves"]), cpu=np.array(recs[a]["cpu"]),
                 wall=np.array(recs[a]["wall"]), seed0=SEED, pinned_lp=int(os.environ.get("PINNED_LP", -1)))
    return inst, int(os.environ.get("PINNED_LP", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="timing_curves")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--algos", nargs="+", default=METHODS)
    ap.add_argument("--instances", nargs="+", default=INSTS)
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    lps, info = performance_cores()
    mgr = Manager()
    q = mgr.Queue()
    for lp in lps:
        q.put(lp)
    insts = [i for i in args.instances
             if not all(os.path.exists(os.path.join(outdir, f"{i}__{a}.npz")) for a in args.algos)]
    tasks = [(i, args.algos, args.runs, outdir) for i in sorted(insts, key=lambda s: -int(s.split("_n")[1]))]
    t0 = time.time()
    with open(os.path.join(outdir, "configs.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(vars(args), workers=len(lps), pinned=lps, cpu=info,
                                 started=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    with Pool(len(lps), initializer=_pin, initargs=(q,)) as pool:
        for inst, lp in pool.imap_unordered(bench_instance, tasks):
            print(f"{inst} done on logical processor {lp} ({time.time() - t0:.0f}s)", flush=True)
    print("written", outdir)


if __name__ == "__main__":
    main()
