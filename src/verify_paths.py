"""Independent safety check of every final path.

Uses code that is separate from the optimiser's cost function:
* threats: analytic point-to-segment distance in a plain loop;
* terrain: exact minimum clearance from clearance_ref (cell-by-cell
  quadratic reconstruction).
A path is independently feasible if every segment keeps a horizontal
distance >= R + D from every threat centre and an exact ground clearance
>= h_clear; for experiments run with the altitude-band variant (band_along in
the run log) the height above ground must also stay in [h_min, h_max] along
every segment.  The result is compared with the feasibility reported by the
cost function (n_viol == 0); any disagreement is listed together with the
cost function's own clearance, threat margin and maximum height.

    python src/verify_paths.py main heldout public b2 heldout_band --workers 8
"""
from __future__ import annotations

import glob
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from clearance_ref import path_min_clearance
from env import BAND_TOL, COMP_KEYS, CostParams, Terrain, get_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-6
# Altitude-band variant: the cost function ignores band violations up to
# BAND_TOL, and optimizers end exactly on that tolerance (clearance
# 100 - 1e-6 m).  Two independent floating-point implementations cannot agree
# on which side of an identical threshold such a path lies, so the check
# allows BAND_TOL plus 1e-9 m for rounding (the implementations differ by
# < 1e-12 m, notes/clearance_check.json, notes/ceiling_check.json) and records
# the largest band violation of the paths the cost function accepted.
TOL_BAND = BAND_TOL + 1e-9


def seg_point_dist(a, b, c):
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 == 0 else min(1.0, max(0.0, float((c - a) @ d / L2)))
    return float(np.hypot(*(a + t * d - c)))


def check_task(task):
    tag, f, band = task
    return [dict(r, tag=tag) for r in check_file(f, band)]


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


def tag_params(tag):
    """Cost-model overrides used for an experiment (from its run log)."""
    f = os.path.join(ROOT, "results", tag, "configs.jsonl")
    if not os.path.exists(f):
        return {}
    with open(f, encoding="utf-8") as fh:
        return json.loads(json.loads(fh.readline())["params"])


_MIRROR = {}


def check_file(f, band=False):
    """band=True: the altitude band [h_min, h_max] must also hold along every
    segment (sensitivity variant); checked with the reference implementation
    on the terrain and on the mirrored terrain."""
    inst, algo = os.path.basename(f)[:-4].split("__")
    name = inst.split("_")[0]
    s = get_scenario(name)
    p = CostParams()
    with np.load(f) as z:
        paths, comps = z["paths"], z["comps"]
    keys = list(COMP_KEYS)

    def col(k):            # cost-function diagnostics (max_agl is missing in older result files)
        i = keys.index(k)
        return comps[:, i] if i < comps.shape[1] else np.full(len(comps), np.nan)

    nv, cf_clear, cf_margin, cf_top = col("n_viol"), col("clearance"), col("thr_margin"), col("max_agl")
    rows = []
    for k, P in enumerate(paths):
        margin = min(seg_point_dist(P[j, :2], P[j + 1, :2], np.array([cx, cy])) - R - p.drone
                     for cx, cy, R in s.threats for j in range(len(P) - 1))
        clear, _ = path_min_clearance(s.terrain, P)
        indep = (margin >= -TOL) and (clear >= p.h_clear - TOL)
        top = bandv = None
        if band:
            if name not in _MIRROR:
                _MIRROR[name] = Terrain(-s.terrain.Z, s.terrain.res)
            Pn = P.copy()
            Pn[:, 2] *= -1.0
            top = float(-path_min_clearance(_MIRROR[name], Pn)[0])
            bandv = float(max(0.0, p.h_min - clear, top - p.h_max))
            indep = indep and (bandv <= TOL_BAND)
        rows.append(dict(inst=inst, algo=algo, run=int(k), margin=float(margin), clearance=float(clear), top=top,
                         band_violation=bandv, cost_feasible=bool(nv[k] == 0), check_feasible=bool(indep),
                         cf_clearance=_num(cf_clear[k]), cf_margin=_num(cf_margin[k]), cf_top=_num(cf_top[k])))
    return rows


def main(tags, workers=20):
    band = {t: bool(tag_params(t).get("band_along", False)) for t in tags}
    tasks = [(t, f, band[t]) for t in tags for f in sorted(glob.glob(os.path.join(ROOT, "results", t, "*.npz")))]
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(check_task, tasks, chunksize=4) for r in part]
    out = {}
    for t in tags:
        rr = [r for r in rows if r["tag"] == t]
        mism = [r for r in rr if r["cost_feasible"] != r["check_feasible"]]
        # dangerous: reported feasible by the cost function, infeasible in the check
        unsafe = [r for r in mism if r["cost_feasible"] and not r["check_feasible"]]
        feas = [r for r in rr if r["check_feasible"]]
        per = {}
        for a in sorted({r["algo"] for r in rr}):
            fa = [r for r in feas if r["algo"] == a]
            per[a] = dict(n=sum(r["algo"] == a for r in rr), feasible=len(fa),
                          min_clear=min((r["clearance"] for r in fa), default=None),
                          max_top=max((r["top"] for r in fa if r["top"] is not None), default=None),
                          tangent=sum(abs(r["margin"]) <= 1e-3 for r in fa),
                          min_margin=min((r["margin"] for r in fa), default=None))
        rec_keys = ("algo", "inst", "run", "margin", "clearance", "top", "band_violation", "cost_feasible",
                    "check_feasible", "cf_clearance", "cf_margin", "cf_top")
        bv = [r["band_violation"] for r in rr if r["cost_feasible"] and r["band_violation"] is not None]
        out[t] = dict(band_along=band[t], n_paths=len(rr), n_feasible=len(feas), n_mismatch=len(mism),
                      tol=TOL, tol_band=TOL_BAND if band[t] else None,
                      max_band_violation_reported_feasible=max(bv) if bv else None,
                      n_at_band_tolerance=sum(x > TOL for x in bv),
                      n_reported_feasible_but_infeasible=len(unsafe),
                      n_rejected_but_feasible=len(mism) - len(unsafe),
                      mismatches=[f"{r['algo']} {r['inst']} run {r['run']}: margin {r['margin']:.3e} m, clearance "
                                  f"{r['clearance']:.12f} m"
                                  + ("" if r["top"] is None else f", max. height {r['top']:.12f} m")
                                  + f", cost function {'feasible' if r['cost_feasible'] else 'infeasible'}"
                                  for r in mism][:50],
                      mismatch_records=[{k: r[k] for k in rec_keys} for r in mism][:50],
                      min_clear=min((r["clearance"] for r in feas), default=None),
                      max_top=max((r["top"] for r in feas if r["top"] is not None), default=None),
                      per_method=per)
        print(f"{t}: {len(rr)} paths, {len(feas)} independently feasible, {len(mism)} disagreements with the "
              f"cost function ({len(unsafe)} reported feasible but infeasible); min exact clearance of feasible "
              f"paths {out[t]['min_clear']:.4f} m" if feas else f"{t}: no feasible paths", flush=True)
        for line in out[t]["mismatches"]:
            print("   ", line)
    dst = os.path.join(ROOT, "results", "analysis", "safety_check.json")
    old = {}
    if os.path.exists(dst):                       # keep the results of other tags
        try:
            with open(dst, encoding="utf-8") as fh:
                old = json.load(fh)
        except ValueError:
            old = {}
    old.update(out)
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(old, fh, indent=1)
    os.replace(tmp, dst)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results", "analysis"), exist_ok=True)
    args = sys.argv[1:]
    w = 20
    if "--workers" in args:
        i = args.index("--workers")
        w = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    main(args or ["main"], workers=w)
