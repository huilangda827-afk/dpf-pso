"""Parallel experiment runner.

Examples:
    python src/run_experiments.py --tag main --runs 30 --instances S1_n10 S1_n20 --algos PSO DPF-PSO
    python src/run_experiments.py --tag b2 --params '{"b2": 10}' ...
    python src/run_experiments.py --tag eq --mult results/timing/multipliers.json ...

Budget: 3000 * n FEs (30,000 for n = 10, 60,000 for n = 20, 90,000 for n = 30),
times --budget-mult and, with --mult, times a per-(n, algorithm) multiplier.
Each (instance, algorithm) pair is saved to results/<tag>/<instance>__<algo>.npz.
Run r of every algorithm uses seed --seed + r (common random numbers).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from algorithms import algo_spec
from core import Evaluator
from env import COMP_KEYS, CostParams, PathProblem, get_scenario, parse_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20271101               # fresh seed family for all final experiments
DPF_KEYS = ("n_global", "n_local", "fe_global", "fe_local", "imp_global", "imp_local",
            "k_global", "k_local")


def budget(n, mult=1.0):
    return int(round(3000 * n * mult))


def run_one(task):
    inst, algo, run, seed, max_fes, params = task
    name, n = parse_instance(inst)
    prob = PathProblem(get_scenario(name), n, CostParams(**params))
    fn, enc, kwf = algo_spec(algo)
    kw = kwf(prob)
    fun = prob.evaluate if enc != "sph" else prob.evaluate_sph
    ev = Evaluator(fun, max_fes)
    rng = np.random.default_rng(seed)
    t0, c0 = time.perf_counter(), time.process_time()
    fn(prob, ev, rng, **kw)
    elapsed, cpu = time.perf_counter() - t0, time.process_time() - c0
    if enc != "sph":
        P, h = prob.plane_to_points(ev.best_x[None])
    else:
        P, h = prob.sph_to_points(ev.best_x[None])
    comps = prob.components(P[0], h[0])
    path = prob.full_path(P)[0]
    st = kw["dpf"].stats if "dpf" in kw else {}
    return dict(inst=inst, algo=algo, run=run, best_f=ev.best_f, curve=ev.curve(101),
                tcurve=ev.time_curve(101), path=path, comps=[comps[k] for k in COMP_KEYS],
                time=elapsed, cpu=cpu, dp_fes=ev.dp_fes, used=ev.used,
                dpf=[float(st.get(k, 0)) for k in DPF_KEYS])


def save_group(outdir, inst, algo, recs):
    recs = sorted(recs, key=lambda r: r["run"])
    np.savez_compressed(
        os.path.join(outdir, f"{inst}__{algo}.npz"),
        best_f=np.array([r["best_f"] for r in recs]),
        curves=np.stack([r["curve"] for r in recs]),
        tcurves=np.stack([r["tcurve"] for r in recs]),
        paths=np.stack([r["path"] for r in recs]),
        comps=np.array([r["comps"] for r in recs]),
        comp_keys=np.array(COMP_KEYS),
        times=np.array([r["time"] for r in recs]),
        cpu=np.array([r["cpu"] for r in recs]),
        dp_fes=np.array([r["dp_fes"] for r in recs]),
        used=np.array([r["used"] for r in recs]),
        dpf=np.array([r["dpf"] for r in recs]),
        dpf_keys=np.array(DPF_KEYS),
        runs=np.array([r["run"] for r in recs]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--algos", nargs="+", required=True)
    ap.add_argument("--budget-mult", type=float, default=1.0)
    ap.add_argument("--mult", type=str, default="", help="JSON file {n: {algo: multiplier}}")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--params", type=str, default="{}", help="JSON overrides of CostParams")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    outdir = os.path.join(ROOT, "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    params = json.loads(args.params)
    mult = {}
    if args.mult:
        with open(args.mult, encoding="utf-8") as fh:
            mult = json.load(fh)
    tasks = []
    for inst in args.instances:
        n = parse_instance(inst)[1]
        for algo in args.algos:
            if args.skip_existing and os.path.exists(os.path.join(outdir, f"{inst}__{algo}.npz")):
                continue
            m = args.budget_mult * float(mult.get(str(n), {}).get(algo, 1.0))
            for r in range(args.runs):
                tasks.append((inst, algo, r, args.seed + r, budget(n, m), params))
    # long tasks first (DP-heavy methods, large n) for better load balance
    weight = {"DPF": 3.0, "IDP": 3.0, "LSHADE": 1.5, "GLS": 1.5}
    tasks.sort(key=lambda t: -t[4] * max([v for k, v in weight.items() if k in t[1]] + [1.0])
               * (3.0 if t[0].startswith("C1") else 1.0))
    # one record per invocation (resumed or partial runs append a line)
    with open(os.path.join(outdir, "configs.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(vars(args), n_tasks=len(tasks),
                                 started=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")

    groups = defaultdict(list)
    expected = defaultdict(int)
    for t in tasks:
        expected[(t[0], t[1])] += 1
    t0 = time.time()
    done = 0
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for rec in pool.imap_unordered(run_one, tasks, chunksize=1):
            key = (rec["inst"], rec["algo"])
            groups[key].append(rec)
            done += 1
            if len(groups[key]) == expected[key]:
                save_group(outdir, key[0], key[1], groups.pop(key))
            if done % 200 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"{done}/{len(tasks)} runs, {el:.0f}s elapsed, "
                      f"ETA {el / done * (len(tasks) - done):.0f}s", flush=True)
    print("done", args.tag)


if __name__ == "__main__":
    main()
