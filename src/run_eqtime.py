"""Equal-CPU-time comparison.

Each baseline gets its FE budget multiplied by T(DPF-PSO) / T(baseline), where
T is the mean wall-clock time per run measured in results/<ref_tag> (same
machine, same 20-way parallel load), separately for n = 10 and n = 20.

    python src/run_eqtime.py --ref main_v2 --tag eqtime --workers 20
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

from run_experiments import ROOT, run_one, save_group

BASE = ["PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE"]
INSTS = [f"S{k}_n{n}" for n in (10, 20) for k in (1, 2, 3, 4, 5, 6)]


def multipliers(ref):
    """FE-budget multipliers T(DPF-PSO)/T(baseline) per n.

    If results/<ref>/timing.json exists (controlled benchmark), its CPU
    seconds are used; otherwise the mean wall-clock time per run stored in
    results/<ref>/*.npz (20-process pool of the main experiment)."""
    mult = {}
    tj = os.path.join(ROOT, "results", ref, "timing.json")
    if os.path.exists(tj):
        with open(tj, encoding="utf-8") as fh:
            summ = json.load(fh)["summary"]
        times = {n: {a: summ[n][a]["cpu"] for a in BASE + ["DPF-PSO"]} for n in ("10", "20")}
        mult["source"] = "controlled benchmark (timing.json, CPU time)"
    else:
        times = {}
        for n in ("10", "20"):
            ins = [i for i in INSTS if i.endswith(f"n{n}")]
            times[n] = {a: float(np.mean([np.load(os.path.join(ROOT, "results", ref, f"{i}__{a}.npz"))
                                          ["times"].mean() for i in ins])) for a in BASE + ["DPF-PSO"]}
        mult["source"] = f"pool wall-clock times of results/{ref}"
    for n in ("10", "20"):
        t = times[n]
        mult[n] = {a: round(max(1.0, t["DPF-PSO"] / t[a]), 1) for a in BASE}
        mult[f"time_{n}"] = t
    return mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="timing")
    ap.add_argument("--tag", default="eqtime")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20270201)
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    mult = multipliers(args.ref)
    with open(os.path.join(outdir, "multipliers.json"), "w", encoding="utf-8") as fh:
        json.dump(mult, fh, indent=1)
    print(json.dumps({k: v for k, v in mult.items() if not k.startswith("time")}, indent=1))
    tasks = []
    for inst in INSTS:
        n = int(inst.split("_n")[1])
        base = 30000 if n == 10 else 60000
        for a in BASE:
            fes = int(round(base * mult[str(n)][a]))
            for r in range(args.runs):
                tasks.append((inst, a, r, args.seed + r, fes, {}))
    # longest tasks first for better load balance
    tasks.sort(key=lambda t: -t[4])
    groups = defaultdict(list)
    expected = defaultdict(int)
    for t in tasks:
        expected[(t[0], t[1])] += 1
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for rec in pool.imap_unordered(run_one, tasks, chunksize=1):
            key = (rec["inst"], rec["algo"])
            groups[key].append(rec)
            done += 1
            if len(groups[key]) == expected[key]:
                save_group(outdir, key[0], key[1], groups.pop(key))
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"{done}/{len(tasks)} runs, {el:.0f}s elapsed, ETA {el / done * (len(tasks) - done):.0f}s",
                      flush=True)
    print("done", outdir)


if __name__ == "__main__":
    main()
