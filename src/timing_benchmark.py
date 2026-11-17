"""Controlled CPU-time benchmark.

One worker process per performance core (P-core), each pinned to one logical
processor of a different P-core (no SMT sharing, no efficiency cores).  A
worker handles whole instances and runs all methods in interleaved order
(run r of every method before run r+1), so thermal drift affects all methods
alike.  Output: results/<tag>/timing.json with mean and standard deviation of
the CPU time per run, and results/<tag>/multipliers.json with the FE-budget
multipliers T(DPF-PSO)/T(method) used for the equal-time comparison.

    python src/timing_benchmark.py --tag timing --runs 5
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from multiprocessing import Manager, Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import psutil

from algorithms import MAIN, algo_spec
from core import Evaluator
from cpuinfo_win import performance_cores
from env import CostParams, PathProblem, get_scenario, parse_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTS = [f"S{k}_n{n}" for n in (10, 20) for k in (1, 2, 3, 4, 5, 6)]
SEED = 40270000


def _pin(q):
    lp = q.get()
    psutil.Process().cpu_affinity([lp])
    os.environ["PINNED_LP"] = str(lp)


def bench_instance(args):
    inst, algos, runs = args
    name, n = parse_instance(inst)
    prob = PathProblem(get_scenario(name), n, CostParams())
    out = {a: {"cpu": [], "wall": []} for a in algos}
    for r in range(runs):
        for a in algos:
            fn, enc, kwf = algo_spec(a)
            ev = Evaluator(prob.evaluate if enc != "sph" else prob.evaluate_sph, 3000 * n)
            rng = np.random.default_rng(SEED + r)
            t0, c0 = time.perf_counter(), time.process_time()
            fn(prob, ev, rng, **kwf(prob))
            out[a]["wall"].append(time.perf_counter() - t0)
            out[a]["cpu"].append(time.process_time() - c0)
    return inst, int(os.environ.get("PINNED_LP", -1)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="timing")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--algos", nargs="+", default=MAIN)
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    lps, info = performance_cores()
    mgr = Manager()
    q = mgr.Queue()
    for lp in lps:
        q.put(lp)
    tasks = [(i, args.algos, args.runs) for i in sorted(INSTS, key=lambda s: -int(s.split("_n")[1]))]
    res, where = {}, {}
    t0 = time.time()
    with Pool(len(lps), initializer=_pin, initargs=(q,)) as pool:
        for inst, lp, out in pool.imap_unordered(bench_instance, tasks):
            res[inst], where[inst] = out, lp
            print(f"{inst} done on logical processor {lp} ({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for n in ("10", "20"):
        ins = [i for i in INSTS if i.endswith(f"n{n}")]
        summary[n] = {}
        for a in args.algos:
            per_inst = [np.mean(res[i][a]["cpu"]) for i in ins]
            # spread: std over runs, averaged over instances (relative)
            rel_sd = [np.std(res[i][a]["cpu"], ddof=1) / np.mean(res[i][a]["cpu"]) for i in ins]
            summary[n][a] = dict(cpu=float(np.mean(per_inst)), cpu_rel_sd=float(np.mean(rel_sd)))
    mult = {n: {a: round(max(1.0, summary[n]["DPF-PSO"]["cpu"] / summary[n][a]["cpu"]), 1)
                for a in args.algos if a != "DPF-PSO"} for n in ("10", "20")}
    meta = dict(runs=args.runs, workers=len(lps), pinned=lps, cpu=info, instance_lp=where)
    with open(os.path.join(outdir, "timing.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary, per_instance=res, meta=meta), fh, indent=1)
    with open(os.path.join(outdir, "multipliers.json"), "w", encoding="utf-8") as fh:
        json.dump(mult, fh, indent=1)
    for n in ("10", "20"):
        print(f"n={n}: " + "  ".join(f"{a}={summary[n][a]['cpu']:.2f}s(+-{100 * summary[n][a]['cpu_rel_sd']:.0f}%)"
                                     for a in args.algos))
    print("multipliers", mult)


if __name__ == "__main__":
    main()
