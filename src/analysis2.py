"""Analysis of the second-revision experiments.

    python src/analysis2.py

Reads results/{main,heldout,public,eqtime,n30,budget5,b2,timing,tuning} and
writes results/analysis/summary.json, results/analysis/tables/*.md and the
figures in figures/.  Every number used in the paper and the supplement is
produced here.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scipy.stats import rankdata

import stats2 as S
from algorithms import ABLATION, MAIN, PLUGIN, TUNED
from env import get_scenario, parse_instance

ROOT = S.ROOT
OUT = os.path.join(ROOT, "results", "analysis")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(ROOT, "figures")
REF = "DPF-PSO"
BASE = [m for m in MAIN if m != REF]
DESIGN = [f"S{k}_n{n}" for n in (10, 20) for k in (1, 2, 3, 4, 5, 6)]
LAB = {"LSHADE": "L-SHADE", "CMAES": "CMA-ES", "LSHADE+DPF": "L-SHADE+DPF",
       "DE-T": "DE (tuned)", "LSHADE-T": "L-SHADE (tuned)", "SPSO-T": "SPSO (tuned)",
       "IDP-T": "IDP (tuned)", "DPF-PSO-T": "DPF-PSO (tuned)", "CMAES-T": "CMA-ES (tuned)",
       "DPF-PSO_noG": "w/o global fusion", "DPF-PSO_noR": "w/o random candidates",
       "DPF-PSO_noL": "w/o local fusion", "DPF-PSO_noA": "fixed σ schedule",
       "DPF-PSO_greedy": "greedy instead of DP", "DPF-only": "fusion only (no PSO moves)",
       "DE-NP5D": "DE, NP = 5D", "DE-NP10D": "DE, NP = 10D", "SPSO-N100": "SPSO, 100 particles",
       "SPSO-N500": "SPSO, 500 particles"}
SHORT = {"LSHADE": "L-SH.", "CMAES": "CMA", "DPF-PSO": "DPF-PSO"}
COLORS = {"PSO": "#7f7f7f", "SPSO": "#1f77b4", "GWO": "#8c564b", "WOA": "#9467bd", "HHO": "#e377c2",
          "DE": "#2ca02c", "LSHADE": "#bcbd22", "CMAES": "#17becf", "IDP": "#ff7f0e", "GLS": "#000000",
          "DPF-PSO": "#d62728"}
STYLES = {"PSO": ":", "SPSO": "--", "GWO": "-.", "WOA": (0, (3, 1, 1, 1)), "HHO": (0, (5, 2)),
          "DE": "-.", "LSHADE": (0, (1, 1)), "CMAES": (0, (4, 1, 1, 1, 1, 1)), "IDP": (0, (6, 2)),
          "GLS": (0, (2, 2)), "DPF-PSO": "-"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "mathtext.fontset": "stix",
                     "savefig.dpi": 600})


def lab(a):
    return LAB.get(a, a)


def ilab(inst):
    k, n = inst.split("_n")
    return f"{k}, {n}"


def pct(x, d=1):
    return f"{100 * x:+.{d}f}%".replace("-", "−")


def save_md(name, lines):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def has_all(d, insts, methods):
    return all(i in d and all(m in d[i] for m in methods) for i in insts)


# ----------------------------------------------------------------------------
def design_results(d, methods, tagname, ref=REF, family=None):
    """Per-instance medians, Holm-corrected signs (family = the other methods
    of the table on that instance), A12 and gaps."""
    others = [m for m in methods if m != ref]
    res = {}
    for inst in sorted(d, key=lambda s: (parse_instance(s)[1], s)):
        if not all(m in d[inst] for m in methods):
            continue
        cmp_ = S.compare_instance(d, inst, ref, others)
        res[inst] = dict(median={m: S.median(d[inst][m]) for m in methods},
                         mean={m: float(d[inst][m]["best_f"].mean()) for m in methods},
                         sd={m: float(d[inst][m]["best_f"].std(ddof=1)) for m in methods},
                         feas={m: S.feasible_rate(d[inst][m]) for m in methods},
                         cmp=cmp_)
    insts = list(res)
    meds = np.array([[res[i]["median"][m] for m in methods] for i in insts])
    ranks = S.friedman_ranks(meds)
    w = {o: S.wtl([res[i]["cmp"][o]["sign"] for i in insts]) for o in others}
    best_count = int(sum(np.argmin(meds[k]) == methods.index(ref) for k in range(len(insts))))
    gap_best = [S.gap(min(res[i]["median"][o] for o in others), res[i]["median"][ref]) for i in insts]
    return dict(tag=tagname, methods=methods, instances=insts, per_instance=res, ranks=ranks.tolist(),
                wtl=w, best_count=best_count, gap_best=gap_best)


def table_design(R, name, caption_cols=None):
    methods = R["methods"]
    hdr = "| Inst. | " + " | ".join(SHORT.get(m, m) for m in methods) + " |"
    lines = [hdr, "|:--" + "|:-:" * len(methods) + "|"]
    for inst in R["instances"]:
        r = R["per_instance"][inst]
        meds = [r["median"][m] for m in methods]
        b = int(np.argmin(meds))
        cells = []
        for k, m in enumerate(methods):
            s = "" if m == REF else " " + {"+": "+", "=": "≈", "-": "−"}[r["cmp"][m]["sign"]]
            txt = f"{meds[k]:.0f}{s}"
            if r["feas"][m] < 1.0:
                txt += f"^{int(round(100 * (1 - r['feas'][m])))}%^"
            cells.append(f"**{txt}**" if k == b else txt)
        tag = "*" if inst.split("_")[0] in ("S5", "S6") else ""
        lines.append(f"| {ilab(inst)}{tag} | " + " | ".join(cells) + " |")
    lines.append("| +/≈/− | " + " | ".join("—" if m == REF else "/".join(map(str, R["wtl"][m])) for m in methods) + " |")
    lines.append("| Rank | " + " | ".join(f"{x:.2f}" for x in R["ranks"]) + " |")
    save_md(name, lines)


# ----------------------------------------------------------------------------
def heldout_results(dh, methods, ref=REF, ns=(10, 20), scen=None):
    others = [m for m in methods if m != ref]
    scen = list(scen) if scen is not None else [f"H{k:02d}" for k in range(1, 41)]
    out = dict(methods=methods, n=[], tests={}, scenarios=scen)
    pvals, keys = [], []
    for n in ns:
        insts = [f"{s}_n{n}" for s in scen if f"{s}_n{n}" in dh and all(m in dh[f"{s}_n{n}"] for m in methods)]
        if len(insts) < 3:              # incomplete data (only while the pipeline runs)
            print(f"held-out n={n}: only {len(insts)} complete scenarios, skipped")
            continue
        out["n"].append(n)
        med = {m: np.array([S.median(dh[i][m]) for i in insts]) for m in methods}
        feas = {m: float(np.mean([S.feasible_rate(dh[i][m]) for i in insts])) for m in methods}
        allfail = int(sum(all(S.feasible_rate(dh[i][m]) == 0 for m in methods) for i in insts))
        meds = np.stack([med[m] for m in methods], axis=1)
        ranks = S.friedman_ranks(meds)
        best = int(sum(np.argmin(meds[k]) == methods.index(ref) for k in range(len(insts))))
        out[f"n{n}"] = dict(n_scen=len(insts), feas=feas, all_fail=allfail, ranks=ranks.tolist(),
                            best_count=best, medians={m: med[m].tolist() for m in methods},
                            instances=insts,
                            n_infeasible={m: int(sum(int(np.sum(S.comp(dh[i][m], "n_viol") > 0)) for i in insts))
                                          for m in methods},
                            n_runs={m: int(sum(len(dh[i][m]["best_f"]) for i in insts)) for m in methods})
        # within-scenario rank-sum tests, Holm over the other methods of the family
        wcmp = {i: S.compare_instance(dh, i, ref, others) for i in insts}
        for o in others:
            x, y = med[ref], med[o]
            g = (y - x) / y
            within = [wcmp[i][o]["sign"] for i in insts]
            a = [S.a12(dh[i][ref]["best_f"], dh[i][o]["best_f"]) for i in insts]
            p = S.signed_rank_p(x, y)
            dd = y - x                                   # > 0: DPF-PSO better on that scenario
            nz = dd[dd != 0]
            rk = rankdata(np.abs(nz))
            w_plus, w_minus = float(rk[nz > 0].sum()), float(rk[nz < 0].sum())
            out["tests"][f"{o}|{n}"] = dict(win=float(np.mean(x < y)), loss=float(np.mean(x > y)),
                                            gap_median=float(np.median(g)), gap_q1=float(np.percentile(g, 25)),
                                            gap_q3=float(np.percentile(g, 75)), p=p, a12_median=float(np.median(a)),
                                            within_wtl=S.wtl(within), gaps=g.tolist(),
                                            w_plus=w_plus, w_minus=w_minus)
            pvals.append(p)
            keys.append(f"{o}|{n}")
    adj = S.holm(pvals)
    for k, pa in zip(keys, adj):
        t = out["tests"][k]
        t["p_holm"] = float(pa)
        # direction of a significant result: larger signed-rank sum
        t["sig"] = "=" if pa >= 0.05 else ("+" if t["w_plus"] > t["w_minus"] else "-")
    return out


def alias_tuned(d):
    """If tuning selected the default configuration of a method, its tuned
    variant is identical (same configuration, same seeds; checked on 8
    instances for CMA-ES) and the default runs are used for it."""
    for t in TUNED:
        f = os.path.join(ROOT, "results", "tuning", f"{t}.json")
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            if json.load(fh)["selected_index"] != 0:
                continue
        for i in d:
            if t in d[i] and f"{t}-T" not in d[i]:
                d[i][f"{t}-T"] = d[i][t]
    return d


def holm_family(*results):
    """Re-apply the Holm correction jointly over the tests of several
    heldout_results() outputs (one family); updates p_holm and sig in place."""
    keys = [(R, k) for R in results for k in R["tests"]]
    adj = S.holm([R["tests"][k]["p"] for R, k in keys])
    for (R, k), pa in zip(keys, adj):
        t = R["tests"][k]
        t["p_holm"] = float(pa)
        t["sig"] = "=" if pa >= 0.05 else ("+" if t["w_plus"] > t["w_minus"] else "-")
    for R in results:
        R["family_size"] = len(keys)


def agl_stats(d, methods, insts=None, step=1.0):
    """Height above ground along the final paths (1 m sampling; statistics
    only, not a safety check): length-weighted share below h_min and above
    h_max, and the minimum and maximum height of each path."""
    out = {}
    insts = insts or sorted(d)
    for m in methods:
        below, above, mn, mx = [], [], [], []
        for inst in insts:
            if m not in d.get(inst, {}):
                continue
            T = get_scenario(parse_instance(inst)[0]).terrain
            for P in d[inst][m]["paths"]:
                b = a = tot = 0.0
                lo, hi = np.inf, -np.inf
                for k in range(len(P) - 1):
                    A, B = P[k], P[k + 1]
                    L = float(np.hypot(B[0] - A[0], B[1] - A[1]))
                    if L < 1e-9:
                        continue
                    t = np.linspace(0.0, 1.0, max(2, int(np.ceil(L / step)) + 1))
                    g = A[2] + t * (B[2] - A[2]) - T.height(A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]))
                    w = np.full(len(t), L / (len(t) - 1))
                    w[0] *= 0.5
                    w[-1] *= 0.5
                    b += float(w[g < 100.0].sum())
                    a += float(w[g > 200.0].sum())
                    tot += L
                    lo, hi = min(lo, float(g.min())), max(hi, float(g.max()))
                below.append(b / tot)
                above.append(a / tot)
                mn.append(lo)
                mx.append(hi)
        if below:
            out[m] = dict(below_mean=float(np.mean(below)), above_mean=float(np.mean(above)),
                          outside_mean=float(np.mean(np.add(below, above))),
                          min_median=float(np.median(mn)), max_median=float(np.median(mx)),
                          n_paths=len(below))
    return out


def anytime(dh, methods, ref=REF, ns=(10, 20), fracs=(0.25, 0.5, 1.0), scen=None):
    """Held-out scenarios: gap of the reference to each method at fractions of
    the budget (from the best-so-far curves of the 1x runs), and the budget
    fraction at which the reference's median curve first reaches the final
    median cost of the method (median over scenarios; > 1 if never)."""
    out = {}
    scen = list(scen) if scen is not None else [f"H{k:02d}" for k in range(1, 41)]
    for n in ns:
        insts = [f"{s}_n{n}" for s in scen if all(m in dh.get(f"{s}_n{n}", {}) for m in methods)]
        curves = {m: np.array([np.median(dh[i][m]["curves"], axis=0) for i in insts]) for m in methods}
        grid = np.linspace(0.0, 1.0, curves[ref].shape[1])
        for o in [m for m in methods if m != ref]:
            res = {}
            for f in fracs:
                c = int(round(f * (len(grid) - 1)))
                x, y = curves[ref][:, c], curves[o][:, c]
                ok = np.isfinite(x) & np.isfinite(y)
                g = (y[ok] - x[ok]) / y[ok]
                res[f"gap_{f:g}"] = float(np.median(g)) if len(g) else float("nan")
                res[f"win_{f:g}"] = float(np.mean(x[ok] < y[ok])) if len(g) else float("nan")
            target = curves[o][:, -1]
            reach = []
            for s in range(len(insts)):
                hit = np.flatnonzero(curves[ref][s] <= target[s])
                reach.append(grid[hit[0]] if len(hit) else np.inf)
            reach = np.array(reach)
            res["reach_median"] = float(np.median(reach))
            res["reach_share"] = float(np.mean(np.isfinite(reach)))
            out[f"{o}|{n}"] = res
    return out


def table_heldout(H, name, ref=REF):
    methods = H["methods"]
    lines = ["| Method | Win (10) | Δ̃ (10) | p_H (10) | Win (20) | Δ̃ (20) | p_H (20) |",
             "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for o in [m for m in methods if m != ref]:
        cells = []
        for n in H["n"]:
            t = H["tests"][f"{o}|{n}"]
            p = t["p_holm"]
            ptxt = "<0.001" if p < 0.001 else f"{p:.3f}"
            cells += [f"{100 * t['win']:.0f}%", pct(t["gap_median"]), ptxt + ("" if t["sig"] == "=" else f" {t['sig']}".replace('-', '−'))]
        lines.append(f"| {lab(o)} | " + " | ".join(cells) + " |")
    cells = []
    for n in H["n"]:
        cells += [f"{H[f'n{n}']['ranks'][methods.index(ref)]:.2f}", "", ""]
    lines.append(f"| Mean rank of {lab(ref)} | " + " | ".join(cells) + " |")
    save_md(name, lines)


# ----------------------------------------------------------------------------
def plugin_results(d):
    out = {}
    for host, hd in PLUGIN:
        insts = [i for i in DESIGN if i in d and host in d[i] and hd in d[i]]
        ps, gaps = [], []
        for i in insts:
            x, y = d[i][hd]["best_f"], d[i][host]["best_f"]
            ps.append(S.ranksum_p(x, y))
            gaps.append(S.gap(np.median(y), np.median(x)))
        adj = S.holm(ps)
        signs = []
        for i, pa in zip(insts, adj):
            e = S.a12(d[i][hd]["best_f"], d[i][host]["best_f"])
            signs.append("=" if pa >= 0.05 else ("+" if e > 0.5 else "-"))
        st = np.array([d[i][hd]["dpf"].mean(axis=0) for i in insts]) if insts else None
        out[host] = dict(insts=insts, gaps=gaps, signs=signs, wtl=S.wtl(signs),
                         gap_mean=float(np.mean(gaps)), gap_median=float(np.median(gaps)),
                         n_global=float(st[:, 0].mean()) if st is not None else 0.0,
                         n_local=float(st[:, 1].mean()) if st is not None else 0.0)
    return out


def plugin_spread(d):
    """Relative spread (max - min) / min of the medians of the six hosts on
    each instance, with DPF and without it."""
    hd = [x for _, x in PLUGIN]
    hs = [x for x, _ in PLUGIN]
    insts = [i for i in DESIGN if i in d and all(h in d[i] for h in hd + hs)]
    sp, sp0 = [], []
    for i in insts:
        m = [S.median(d[i][h]) for h in hd]
        m0 = [S.median(d[i][h]) for h in hs]
        sp.append((max(m) - min(m)) / min(m))
        sp0.append((max(m0) - min(m0)) / min(m0))
    return dict(per_instance=dict(zip(insts, sp)), median=float(np.median(sp)), mean=float(np.mean(sp)),
                max=float(np.max(sp)), median_without=float(np.median(sp0)))


def ablation_results(d):
    variants = [v for v in ABLATION if v != REF]
    insts = [i for i in DESIGN if i in d and all(v in d[i] for v in ABLATION)]
    per = {v: [] for v in variants}
    for i in insts:
        c = S.compare_instance(d, i, REF, variants)
        for v in variants:
            per[v].append(c[v])
    meds = np.array([[S.median(d[i][v]) for v in ABLATION] for i in insts])
    ranks = S.friedman_ranks(meds)
    out = {}
    for k, v in enumerate(ABLATION):
        if v == REF:
            out[v] = dict(rank=float(ranks[k]))
            continue
        signs = [c["sign"] for c in per[v]]
        out[v] = dict(rank=float(ranks[k]), gap_mean=float(np.mean([c["gap"] for c in per[v]])),
                      wtl=S.wtl(signs), worse20=int(sum(s == "+" and i.endswith("n20") for s, i in zip(signs, insts))))
    return out


# ----------------------------------------------------------------------------
def fe_composition(d):
    out = {}
    for n in (10, 20):
        ins = [i for i in DESIGN if i.endswith(f"n{n}") and REF in d.get(i, {})]
        st = np.concatenate([d[i][REF]["dpf"] for i in ins])        # runs x keys
        keys = list(d[ins[0]][REF]["dpf_keys"])
        used = np.concatenate([d[i][REF]["used"] for i in ins])
        dp = np.concatenate([d[i][REF]["dp_fes"] for i in ins])
        g = st[:, keys.index("n_global")]
        l_ = st[:, keys.index("n_local")]
        out[str(n)] = dict(budget=3000 * n, host_share=float(np.mean((used - dp) / used)),
                           global_share=float(np.mean(st[:, keys.index("fe_global")] / used)),
                           local_share=float(np.mean(st[:, keys.index("fe_local")] / used)),
                           n_global=float(g.mean()), n_local=float(l_.mean()),
                           k_global=float(np.mean(st[:, keys.index("k_global")] / np.maximum(g, 1))),
                           k_local=float(np.mean(st[:, keys.index("k_local")] / np.maximum(l_, 1))),
                           imp_global=float(np.mean(st[:, keys.index("imp_global")])),
                           imp_local=float(np.mean(st[:, keys.index("imp_local")])))
    return out


def encoding_stats(d, methods=("SPSO", "SPSO-T", "DPF-PSO", "DPF-PSO-T"), samples=200):
    """Shape of the final paths: number of merged waypoints (zero horizontal
    segment length, possible only in the spherical encoding of SPSO), longest
    horizontal segment and largest height above ground along the path."""
    out = {}
    for inst in d:
        s = get_scenario(parse_instance(inst)[0])
        for m in methods:
            if m not in d[inst]:
                continue
            P = d[inst][m]["paths"]                                  # (runs, n+2, 3)
            seg = np.diff(P, axis=1)
            hl = np.hypot(seg[..., 0], seg[..., 1])
            t = np.linspace(0.0, 1.0, samples)
            pts = P[:, :-1, None, :] + t[None, None, :, None] * seg[:, :, None, :]
            agl = pts[..., 2] - s.terrain.height(pts[..., 0], pts[..., 1])
            out.setdefault(inst, {})[m] = dict(merged=float(np.mean((hl < 1e-6).sum(axis=1))),
                                               max_seg=float(np.median(hl.max(axis=1))),
                                               max_agl=float(np.median(agl.reshape(len(P), -1).max(axis=1))),
                                               min_clear=float(np.median(S.comp(d[inst][m], "clearance"))))
    return out


def merge_analysis(dh, enc, a="SPSO-T", b="DPF-PSO-T"):
    """Held-out scenarios: does the advantage of tuned SPSO over tuned DPF-PSO
    go with merged waypoints?  Spearman correlation between the mean number
    of merged waypoints of a's final paths and the gap (J_a - J_b) / J_a
    (negative: a better)."""
    from scipy.stats import spearmanr
    out = {}
    for n in (10, 20):
        ins = [f"H{k:02d}_n{n}" for k in range(1, 41) if f"H{k:02d}_n{n}" in enc
               and a in enc[f"H{k:02d}_n{n}"] and b in enc[f"H{k:02d}_n{n}"]]
        if len(ins) < 3:
            continue
        g = np.array([(S.median(dh[i][a]) - S.median(dh[i][b])) / S.median(dh[i][a]) for i in ins])
        mg = np.array([enc[i][a]["merged"] for i in ins])
        r = spearmanr(mg, g)
        out[str(n)] = dict(n_scen=len(ins), a_better=int((g < 0).sum()), merged_median=float(np.median(mg)),
                           rho=float(r.correlation), p=float(r.pvalue),
                           maxseg_a=float(np.median([enc[i][a]["max_seg"] for i in ins])),
                           maxseg_b=float(np.median([enc[i][b]["max_seg"] for i in ins])),
                           clear_a=float(np.median([enc[i][a]["min_clear"] for i in ins])),
                           clear_b=float(np.median([enc[i][b]["min_clear"] for i in ins])))
    return out


def safety(d, methods):
    out = {}
    for m in methods:
        recs = [d[i][m] for i in d if m in d[i]]
        if not recs:
            continue
        feas = np.concatenate([S.comp(r, "n_viol") == 0 for r in recs])
        clr = np.concatenate([S.comp(r, "clearance") for r in recs])
        mar = np.concatenate([S.comp(r, "thr_margin") for r in recs])
        out[m] = dict(n=int(len(feas)), feasible=int(feas.sum()),
                      min_clear=float(clr[feas].min()) if feas.any() else None,
                      tangent=int(np.sum(feas & (np.abs(mar) <= 1e-3))),
                      min_margin=float(mar[feas].min()) if feas.any() else None)
    return out


def plugin_heldout(dh, scen=None, ns=(10, 20)):
    """Plug-in test on the held-out scenarios (design log 18c): each host
    with DPF against the same host without it, scenario-level Wilcoxon
    signed-rank on the paired medians, Holm over the (host, n) tests.  The
    PSO -> DPF-PSO pair belongs to the ablation family and is not repeated."""
    scen = list(scen) if scen is not None else [f"H{k:02d}" for k in range(1, 41)]
    out = dict(tests={}, hosts=[])
    pvals, keys = [], []
    for host, hd in PLUGIN:
        if host == "PSO":
            continue
        for n in ns:
            insts = [f"{s}_n{n}" for s in scen if host in dh.get(f"{s}_n{n}", {}) and hd in dh.get(f"{s}_n{n}", {})]
            if len(insts) < 3:
                continue
            x = np.array([S.median(dh[i][hd]) for i in insts])        # host with DPF
            y = np.array([S.median(dh[i][host]) for i in insts])      # host alone
            g = (y - x) / y
            dd = y - x
            nz = dd[dd != 0]
            rk = rankdata(np.abs(nz))
            out["tests"][f"{host}|{n}"] = dict(
                n_scen=len(insts), win=float(np.mean(x < y)), gap_median=float(np.median(g)),
                gap_q1=float(np.percentile(g, 25)), gap_q3=float(np.percentile(g, 75)), p=S.signed_rank_p(x, y),
                w_plus=float(rk[nz > 0].sum()), w_minus=float(rk[nz < 0].sum()),
                feas_host=float(np.mean([S.feasible_rate(dh[i][host]) for i in insts])),
                feas_plug=float(np.mean([S.feasible_rate(dh[i][hd]) for i in insts])))
            pvals.append(out["tests"][f"{host}|{n}"]["p"])
            keys.append(f"{host}|{n}")
    for k, pa in zip(keys, S.holm(pvals)):
        t = out["tests"][k]
        t["p_holm"] = float(pa)
        t["sig"] = "=" if pa >= 0.05 else ("+" if t["w_plus"] > t["w_minus"] else "-")
    out["hosts"] = [h for h, _ in PLUGIN if any(k.startswith(h + "|") for k in out["tests"])]
    out["methods"] = out["hosts"]
    out["n"] = [n for n in ns if any(k.endswith(f"|{n}") for k in out["tests"])]
    out["family_size"] = len(keys)
    return out


def timecurve_results(tag="timing_curves", ref=REF, ns=(10, 20)):
    """CPU-time convergence (design log, section 16).  Per scenario: the CPU
    time at which the median best-so-far curve of the reference first reaches
    the final median cost of a method, divided by that method's median total
    CPU time (median over scenarios; inf if never), with the FE fraction of
    the same point for comparison.  Input: results/<tag>/ from
    src/timing_curves.py (pinned P-cores, process CPU time per checkpoint)."""
    d = S.load(tag)
    if not d:
        return {}
    methods = sorted({m for i in d for m in d[i]})
    out = {}
    for n in ns:
        insts = sorted(i for i in d if i.endswith(f"_n{n}") and all(m in d[i] for m in methods))
        if not insts or ref not in methods:
            continue
        for o in methods:
            if o == ref:
                continue
            ratio, fe, hit = [], [], []
            for i in insts:
                cr = np.median(d[i][ref]["curves"], axis=0)
                ct = np.median(d[i][ref]["ccurves"], axis=0)
                target = float(np.median(d[i][o]["best_f"]))
                t_o = float(np.median(d[i][o]["cpu"]))
                k = np.flatnonzero(cr <= target)
                ratio.append(ct[k[0]] / t_o if len(k) else np.inf)
                fe.append(k[0] / (len(cr) - 1) if len(k) else np.inf)
                hit.append(1.0 if len(k) else 0.0)
            out[f"{o}|{n}"] = dict(
                n_scen=len(insts), reach_share=float(np.mean(hit)),
                ratio_median=float(np.median(ratio)),
                ratio_q1=float(np.percentile(ratio, 25, method="nearest")),
                ratio_q3=float(np.percentile(ratio, 75, method="nearest")),
                fe_median=float(np.median(fe)),
                cpu_ref=float(np.median([np.median(d[i][ref]["cpu"]) for i in insts])),
                cpu_other=float(np.median([np.median(d[i][o]["cpu"]) for i in insts])),
                instances=insts)
    return out


def component_stats(d, methods, insts=None):
    """Composition of the final cost of the feasible final paths: share of
    the weighted components b1 F1 .. b4 F4 in J (median over runs), and the
    maximum turning angle and maximum climb-angle change per path (degrees,
    median over runs).  Descriptive only (design log, section 15 addendum);
    the angles are recomputed from the stored paths with the cost model's
    angle routine, so they do not depend on the cost-model variant."""
    from env import CostParams, PathProblem
    w = dict(F1="b1", F2="b2", F3="b3", F4="b4")
    out = {}
    insts = insts or sorted(d)
    probs = {}
    for m in methods:
        share = {k: [] for k in w}
        mx_turn, mx_dclimb = [], []
        n_paths = 0
        for inst in insts:
            if m not in d.get(inst, {}):
                continue
            rec = d[inst][m]
            feas = S.comp(rec, "n_viol") == 0
            if not feas.any():
                continue
            J = S.comp(rec, "J")[feas]
            if inst not in probs:
                name, n = parse_instance(inst)
                probs[inst] = PathProblem(get_scenario(name), n, CostParams())
            prob = probs[inst]
            for k, b in w.items():
                share[k].extend((getattr(prob.p, b) * S.comp(rec, k)[feas] / J).tolist())
            P = np.asarray(rec["paths"], float)[feas]          # stored paths include S and G
            _, turn, dclimb = prob.path_turns(P[:, :-1], P[:, 1:], detail=True)
            mx_turn.extend(np.max(turn, axis=-1).tolist())
            mx_dclimb.extend(np.max(dclimb, axis=-1).tolist())
            n_paths += int(feas.sum())
        if n_paths:
            mt, md = np.array(mx_turn), np.array(mx_dclimb)
            out[m] = dict(n_paths=n_paths,
                          share_median={k: float(np.median(v)) for k, v in share.items()},
                          max_turn_median=float(np.median(mt)), max_turn_q3=float(np.percentile(mt, 75)),
                          max_dclimb_median=float(np.median(md)),
                          max_dclimb_q3=float(np.percentile(md, 75)),
                          # descriptive kinematic indicators (design log 18g)
                          turn_le45=float(np.mean(mt <= 45.0)), turn_le60=float(np.mean(mt <= 60.0)),
                          dclimb_le30=float(np.mean(md <= 30.0)))
    return out


# ----------------------------------------------------------------------------
def fig_convergence(d, insts, methods, name, ncols=3, height=3.0):
    rows = int(np.ceil(len(insts) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(7.16, height), squeeze=False)
    for ax, inst in zip(axes.ravel(), insts):
        best = min(float(d[inst][m]["best_f"].min()) for m in d[inst])
        n = parse_instance(inst)[1]
        fes = np.linspace(0, 3000 * n, 101)[1:] / 1e3
        for m in methods:
            c = d[inst][m]["curves"][:, 1:]
            g = np.median((c - best) / best, axis=0)
            ax.plot(fes, np.maximum(g, 1e-4), color=COLORS.get(m, "k"), ls=STYLES.get(m, "-"),
                    lw=1.3 if m == REF else 0.9, label=lab(m))
        ax.set_yscale("log")
        ax.set_ylim(1e-4 * 0.8, 2.0)
        ax.set_title(f"{inst.split('_')[0]}, n={n}", pad=2)
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(pad=1)
    for ax in axes[-1]:
        ax.set_xlabel("FEs (×10$^3$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median relative gap")
    for ax in axes.ravel()[len(insts):]:
        ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(methods), 11), frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, w_pad=0.4, h_pad=0.5)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def fig_heldout(H, name, methods, Ht=None):
    """Box plots of the 40 scenario gaps of DPF-PSO to each default method and,
    below the line, of tuned DPF-PSO to each tuned method."""
    others = [(H, m, lab(m)) for m in methods if m != REF]
    if Ht:
        others += [(Ht, m, lab(m)) for m in Ht["methods"] if m != "DPF-PSO-T"]
    ns = [n for n in (10, 20) if n in H["n"]]
    fig, axes = plt.subplots(1, len(ns), figsize=(3.5, 2.6), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], ns):
        data = [100 * np.array(HH["tests"][f"{o}|{n}"]["gaps"]) if n in HH["n"] else [np.nan]
                for HH, o, _ in others]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=2), medianprops=dict(color="k", lw=0.8),
                        boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
        for patch, (_, o, _) in zip(bp["boxes"], others):
            patch.set_facecolor(COLORS.get(o.replace("-T", ""), "#aaaaaa"))
            patch.set_alpha(0.45 if o.endswith("-T") else 0.75)
        ax.axvline(0, color="k", lw=0.6)
        if Ht:
            ax.axhline(len(methods) - 0.5, color="k", lw=0.4, ls="--")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(-25, 80)
        ax.set_xticks([-10, 0, 10, 50])
        ax.set_xticklabels(["−10", "0", "10", "50"])
        ax.set_title(f"n = {n}", pad=2)
        ax.grid(axis="x", alpha=0.3, lw=0.4)
        ax.set_xlabel("Δ (%), > 0: DPF-PSO better")
    axes[0, 0].set_yticks(range(1, len(others) + 1))
    axes[0, 0].set_yticklabels([lb for _, _, lb in others], fontsize=6)
    axes[0, 0].invert_yaxis()
    fig.tight_layout(pad=0.3, w_pad=0.4)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def fig_paths(d, name, n=20, show=("SPSO", "DE", "IDP", "DPF-PSO"), scen=("S1", "S2", "S3", "S4", "S5", "S6")):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.75))
    for ax, key in zip(axes.ravel(), scen):
        s = get_scenario(key)
        T = s.terrain
        gx = np.arange(T.nx) * T.res
        gy = np.arange(T.ny) * T.res
        ax.contourf(gx, gy, T.Z, levels=14, cmap="terrain", alpha=0.85)
        ax.contour(gx, gy, T.Z, levels=14, colors="k", linewidths=0.15, alpha=0.5)
        for cx, cy, R in s.threats:
            ax.add_patch(Circle((cx, cy), R, fc="#ff000055", ec="r", lw=0.6))
            ax.add_patch(Circle((cx, cy), R + 11, fc="none", ec="r", lw=0.4, ls=":"))
        inst = f"{key}_n{n}"
        for m in show:
            if m not in d.get(inst, {}):
                continue
            rec = d[inst][m]
            i = int(np.argmin(rec["best_f"]))
            P = rec["paths"][i]
            ax.plot(P[:, 0], P[:, 1], color=COLORS[m], ls=STYLES[m], lw=1.4 if m == REF else 1.0,
                    label=lab(m), marker="o" if m == REF else None, ms=1.5)
        ax.plot(*s.start[:2], "ks", ms=3.5)
        ax.plot(*s.goal[:2], "k*", ms=5.5)
        ax.set_xlim(0, T.size_x)
        ax.set_ylim(0, T.size_y)
        ax.set_aspect("equal")
        ax.set_title(f"{key}: {s.title}", pad=2, fontsize=7)
        ax.tick_params(pad=1, labelsize=6)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(show), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.2, w_pad=0.3, h_pad=0.6)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def fig_paths_single(d, name, key, show=("SPSO", "DE", "IDP", "DPF-PSO")):
    s = get_scenario(key)
    T = s.terrain
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, n in zip(axes, (10, 20)):
        gx = np.arange(T.nx) * T.res
        gy = np.arange(T.ny) * T.res
        ax.contourf(gx, gy, T.Z, levels=14, cmap="terrain", alpha=0.85)
        for cx, cy, R in s.threats:
            ax.add_patch(Circle((cx, cy), R, fc="#ff000055", ec="r", lw=0.6))
        inst = f"{key}_n{n}"
        for m in show:
            if m in d.get(inst, {}):
                rec = d[inst][m]
                P = rec["paths"][int(np.argmin(rec["best_f"]))]
                ax.plot(P[:, 0], P[:, 1], color=COLORS[m], ls=STYLES[m], lw=1.4 if m == REF else 1.0, label=lab(m))
        ax.plot(*s.start[:2], "ks", ms=3.5)
        ax.plot(*s.goal[:2], "k*", ms=5.5)
        ax.set_xlim(0, T.size_x)
        ax.set_ylim(0, T.size_y)
        ax.set_aspect("equal")
        ax.set_title(f"{key} (public DEM of [5]), n = {n}", pad=2)
    axes[0].legend(loc="lower right", fontsize=6, frameon=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _agl_profile(P, T, step=0.5):
    seg = np.diff(P, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    s_all, g_all = [], []
    acc = 0.0
    for k in range(len(seg)):
        t = np.linspace(0, 1, max(2, int(L[k] / step)))
        z = P[k, 2] + t * seg[k, 2]
        g = z - T.height(P[k, 0] + t * seg[k, 0], P[k, 1] + t * seg[k, 1])
        s_all.append(acc + t * L[k])
        g_all.append(g)
        acc += L[k]
    sv = np.concatenate([[0.0], np.cumsum(L)])
    return np.concatenate(s_all), np.concatenate(g_all), sv, L


def worst_scenario(dh, a="SPSO-T", b="DPF-PSO-T", n=10):
    """Held-out scenario with the largest advantage of a over b."""
    gaps = {f"H{k:02d}_n{n}": (S.median(dh[f"H{k:02d}_n{n}"][a]) - S.median(dh[f"H{k:02d}_n{n}"][b]))
            / S.median(dh[f"H{k:02d}_n{n}"][a]) for k in range(1, 41)}
    return min(gaps, key=gaps.get), gaps


def fig_paths_band(dp, dh, name, n_c1=20, show_c1=("SPSO", "DE", "IDP", "DPF-PSO"),
                   show_h=("SPSO-T", "DPF-PSO-T")):
    """Left: best paths on the real terrain C1 (top view).  Right: height above
    ground along the best paths of tuned SPSO and tuned DPF-PSO on the held-out
    scenario where tuned SPSO had its largest advantage, with the altitude band
    [100, 200] m of the cost model, which applies only at waypoints."""
    s = get_scenario("C1")
    T = s.terrain
    inst = f"C1_n{n_c1}"
    hs, _ = worst_scenario(dh)
    Th = get_scenario(hs.split("_")[0]).terrain
    fig = plt.figure(figsize=(7.16, 2.05))
    gs = fig.add_gridspec(len(show_h), 2, width_ratios=(1.0, 2.4), wspace=0.22, hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    gx, gy = np.arange(T.nx) * T.res, np.arange(T.ny) * T.res
    ax.contourf(gx, gy, T.Z, levels=14, cmap="terrain", alpha=0.85)
    for cx, cy, R in s.threats:
        ax.add_patch(Circle((cx, cy), R, fc="#ff000055", ec="r", lw=0.6))
    for m in show_c1:
        rec = dp[inst][m]
        P = rec["paths"][int(np.argmin(rec["best_f"]))]
        ax.plot(P[:, 0], P[:, 1], color=COLORS[m], ls=STYLES[m], lw=1.3 if m == REF else 1.0, label=lab(m))
    ax.plot(*s.start[:2], "ks", ms=3.5)
    ax.plot(*s.goal[:2], "k*", ms=5.5)
    ax.set_xlim(0, T.size_x)
    ax.set_ylim(0, T.size_y)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6, pad=1)
    ax.legend(loc="lower right", fontsize=5.5, frameon=True)
    ax.set_title(f"(a) C1, real terrain of [5], n = {n_c1}", pad=2, fontsize=7)
    info = dict(scenario=hs)
    for r, m in enumerate(show_h):
        a2 = fig.add_subplot(gs[r, 1])
        rec = dh[hs][m]
        i = int(np.argmin(np.abs(rec["best_f"] - np.median(rec["best_f"]))))      # median run
        sd, g, sv, L = _agl_profile(rec["paths"][i], Th)
        info[m] = dict(cost=float(rec["best_f"][i]), merged=int(np.sum(L < 1e-6)),
                       outside=float(np.mean((g < 100) | (g > 200))), max_agl=float(g.max()),
                       min_agl=float(g.min()))
        a2.axhspan(100, 200, color="#9ecae1", alpha=0.35, lw=0)
        a2.axhline(20, color="k", lw=0.5, ls=":")
        c = COLORS[m.replace("-T", "")]
        a2.plot(sd, g, color=c, lw=1.0)
        a2.plot(sv[1:-1], np.interp(sv[1:-1], sd, g), "o", color=c, ms=2.2)
        nm = int(np.sum(L < 1e-6))
        out = float(np.mean((g < 100) | (g > 200)))
        a2.set_ylim(0, max(320, float(g.max()) + 30))
        a2.tick_params(labelsize=6, pad=1)
        a2.text(0.01, 0.95, f"{lab(m)}: cost {rec['best_f'][i]:.0f}, {nm} of 10 waypoints merged, "
                f"{100 * out:.0f}% of the track outside the band", transform=a2.transAxes, fontsize=5.8, va="top")
        if r == 0:
            a2.set_title(f"(b) Height above ground of median runs, held-out {hs.split('_')[0]}, n = 10",
                         pad=2, fontsize=7)
        if r < len(show_h) - 1:
            a2.set_xticklabels([])
        else:
            a2.set_xlabel("Horizontal distance along the path (m)", fontsize=7)
        a2.set_ylabel("Height (m)", fontsize=7)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return info


def fig_profile(d, name, inst="S6_n10", show=("SPSO-T", "DPF-PSO-T")):
    """Altitude profiles of the best path of each method along its own
    horizontal track, with the terrain under that track, the 20 m clearance
    limit and the waypoints (merged waypoints of SPSO appear as one marker)."""
    key = inst.split("_")[0]
    T = get_scenario(key).terrain
    fig, axes = plt.subplots(len(show), 1, figsize=(3.5, 1.35 * len(show) + 0.35), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), show):
        rec = d[inst][m]
        i = int(np.argmin(rec["best_f"]))
        P = rec["paths"][i]
        seg = np.diff(P, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        s_all, z_all, g_all = [], [], []
        acc = 0.0
        for k in range(len(seg)):
            t = np.linspace(0, 1, max(2, int(L[k] / 0.5)))
            s_all.append(acc + t * L[k])
            z_all.append(P[k, 2] + t * seg[k, 2])
            g_all.append(T.height(P[k, 0] + t * seg[k, 0], P[k, 1] + t * seg[k, 1]))
            acc += L[k]
        s_all, z_all, g_all = map(np.concatenate, (s_all, z_all, g_all))
        sv = np.concatenate([[0.0], np.cumsum(L)])
        ax.fill_between(s_all, 0, g_all, color="#c8b28e", alpha=0.6, lw=0, label="terrain under the path")
        ax.plot(s_all, g_all + 20.0, color="k", lw=0.5, ls=":", label="terrain + 20 m")
        ax.plot(s_all, z_all, color=COLORS[m.replace("-T", "")], lw=1.1, label="path")
        ax.plot(sv[1:-1], P[1:-1, 2], "o", ms=2.2, color=COLORS[m.replace("-T", "")], label="waypoints")
        nm = int(np.sum(L < 1e-6))
        cl = float(S.comp(rec, "clearance")[i])
        ax.set_title(f"{lab(m)}: cost {rec['best_f'][i]:.0f}, {nm} merged waypoints, "
                     f"min. clearance {cl:.1f} m", pad=2, fontsize=6.5)
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_ylabel("Altitude (m)")
    np.atleast_1d(axes)[-1].set_xlabel("Horizontal distance along the path (m)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.3, h_pad=0.4)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    summary = {}
    dm = alias_tuned(S.load("main"))
    if dm:
        R = design_results(dm, MAIN, "main")
        summary["main"] = R
        table_design(R, "table_main.md")
        tuned = [t + "-T" for t in TUNED]
        if has_all(dm, DESIGN, tuned):
            Rt = design_results(dm, tuned, "main_tuned", ref="DPF-PSO-T")
            summary["main_tuned"] = Rt
        sens = {}
        for v in ("DE-NP5D", "DE-NP10D", "SPSO-N100", "SPSO-N500"):
            if has_all(dm, DESIGN, [v, REF]):
                c = [S.compare_instance(dm, i, REF, [v])[v] for i in DESIGN]
                sens[v] = dict(wtl=S.wtl([x["sign"] for x in c]), gap_mean=float(np.mean([x["gap"] for x in c])))
        summary["sensitivity"] = sens
        summary["plugin"] = plugin_results(dm)
        summary["plugin_spread"] = plugin_spread(dm)
        summary["ablation"] = ablation_results(dm)
        summary["fe_composition"] = fe_composition(dm)
        summary["safety_main"] = safety(dm, MAIN)
        summary["encoding_main"] = encoding_stats(dm)
        summary["medians_all"] = {i: {m: S.median(dm[i][m]) for m in dm[i]} for i in DESIGN if i in dm}
        fig_convergence(dm, [f"S{k}_n20" for k in range(1, 7)], ["SPSO", "DE", "LSHADE", "CMAES", "IDP", REF],
                        "fig_conv_main.png", height=3.0)
        fig_convergence(dm, DESIGN, MAIN, "fig_conv_all.png", ncols=4, height=5.2)
        fig_paths(dm, "fig_paths.png")
        fig_profile(dm, "fig_profile.png")
    dh = alias_tuned(S.load("heldout"))
    if dh:
        methods_h = [m for m in MAIN if any(m in dh[i] for i in dh)]
        if methods_h != MAIN or len(dh) < 80:
            print(f"WARNING: held-out results incomplete ({len(dh)} instances, methods {methods_h})")
        Hm = heldout_results(dh, methods_h)
        summary["heldout"] = Hm
        table_heldout(Hm, "table_heldout.md")
        tuned = [t + "-T" for t in TUNED]
        Ht = None
        if all(any(t in dh[i] for i in dh) for t in tuned):
            Ht = heldout_results(dh, tuned, ref="DPF-PSO-T")
            Ht["ref"] = "DPF-PSO-T"
            summary["heldout_tuned"] = Ht
            table_heldout(Ht, "table_heldout_tuned.md", ref="DPF-PSO-T")
        fig_heldout(Hm, "fig_heldout.png", Hm["methods"], Ht)
        summary["safety_heldout"] = safety(dh, MAIN)
        summary["encoding_heldout"] = encoding_stats(dh)
        summary["merge_heldout"] = merge_analysis(dh, summary["encoding_heldout"])
        # third revision: altitude along the paths, anytime behaviour, ablation
        summary["agl_heldout"] = {str(n): agl_stats(dh, ["SPSO", "SPSO-T", "DE", "IDP", "DPF-PSO", "DPF-PSO-T"],
                                                    [i for i in dh if i.endswith(f"_n{n}")]) for n in (10, 20)}
        summary["anytime_heldout"] = anytime(dh, MAIN)
        abl = ["PSO", "DPF-PSO_noG", "DPF-PSO_greedy", "DPF-only"]
        abl += [m for m in ("DPF-PSO_noPG", "DPF-PSO_noA") if all(m in dh[i] for i in dh)]   # design log 18b
        abl.append(REF)
        if all(all(m in dh[i] for m in abl) for i in dh):
            summary["heldout_ablation"] = heldout_results(dh, abl)
        else:
            print("held-out ablation incomplete, skipped")
        # journal version (design log 18c, 18e, 18f): plug-in, local optimizer and
        # spherical-encoding families on the held-out scenarios
        if any(hd in dh[i] for i in dh for _, hd in PLUGIN if hd != REF):
            summary["heldout_plugin"] = plugin_heldout(dh)
        if all("MS-Powell" in dh[i] for i in dh):
            summary["heldout_mspowell"] = heldout_results(dh, ["MS-Powell", REF])
        sph = [m for m in ("DE-sph", "CMAES-sph") if all(m in dh[i] for i in dh)]
        if sph:
            summary["heldout_sph"] = heldout_results(dh, sph + [REF])
            summary["sph_vs_slot"] = {m: heldout_results(dh, [m.replace("-sph", ""), m], ref=m) for m in sph}
    de = S.load("heldout_eqtime")
    if de and dh:
        # design log 18a: every baseline at equal CPU time; methods whose budget
        # multiplier is 1.0 (not faster than DPF-PSO) keep their held-out runs
        mp = os.path.join(ROOT, "results", "timing", "multipliers.json")
        mult = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        merged, reused = {}, set()
        for i in dh:
            n = str(parse_instance(i)[1])
            rec = {REF: dh[i][REF]}
            for m in BASE:
                if m in de.get(i, {}):
                    rec[m] = de[i][m]
                elif mult.get(n, {}).get(m, 1.0) == 1.0 and m in dh[i]:
                    rec[m] = dh[i][m]
                    reused.add(f"{m}|{n}")
            merged[i] = rec
        meth = [m for m in BASE if all(m in merged[i] for i in merged)] + [REF]
        if len(merged) == 80 and len(meth) >= 4:
            summary["heldout_eqtime"] = heldout_results(merged, meth)
            summary["heldout_eqtime"]["reused_from_heldout"] = sorted(reused)
        else:
            print("held-out equal-time comparison incomplete, skipped")
    dbd = S.load("heldout_band")
    if dbd:
        bd = ["SPSO", "DE", "IDP", REF]
        bt = ["SPSO-T", "DPF-PSO-T"]
        if len(dbd) == 80 and all(all(m in dbd[i] for m in bd + bt) for i in dbd):
            Hb = heldout_results(dbd, bd)
            Hbt = heldout_results(dbd, bt, ref="DPF-PSO-T")
            holm_family(Hb, Hbt)                      # one family for the variant
            summary["heldout_band"], summary["heldout_band_tuned"] = Hb, Hbt
            sphb = [m for m in ("DE-sph", "CMAES-sph") if all(m in dbd[i] for i in dbd)]
            if sphb:                                  # design log 18f, band model
                summary["heldout_band_sph"] = heldout_results(dbd, sphb + [REF])
            summary["agl_band"] = {str(n): agl_stats(dbd, bd + bt, [i for i in dbd if i.endswith(f"_n{n}")])
                                   for n in (10, 20)}
            summary["safety_band"] = safety(dbd, bd + bt)
        else:
            print("altitude-band variant incomplete, skipped")
    # journal version: pre-registered confirmatory test on the fresh scenarios
    # H41-H80 (design log, section 15; families and verdict as written there)
    scen2 = [f"H{k:02d}" for k in range(41, 81)]
    six = ["SPSO", "SPSO-T", "DE", "IDP", REF, "DPF-PSO-T"]
    d2, d2b = S.load("heldout2"), S.load("heldout2_band")
    if d2 or d2b:
        complete = (len(d2) == 80 and len(d2b) == 80
                    and all(all(m in d2[i] for m in six) for i in d2)
                    and all(all(m in d2b[i] for m in six) for i in d2b))
        if complete:
            bd = ["SPSO", "DE", "IDP", REF]
            bt = ["SPSO-T", "DPF-PSO-T"]
            # family T (4 tests): the tuned pair under the original and the band model
            Ht2 = heldout_results(d2, bt, ref="DPF-PSO-T", scen=scen2)
            Ht2b = heldout_results(d2b, bt, ref="DPF-PSO-T", scen=scen2)
            holm_family(Ht2, Ht2b)
            # family D (6 tests): defaults under the band model
            Hd2b = heldout_results(d2b, bd, scen=scen2)
            # family R (6 tests, addendum): defaults under the original model
            Hd2 = heldout_results(d2, bd, scen=scen2)
            for key, H, r in (("heldout2_tuned", Ht2, "DPF-PSO-T"), ("heldout2_band_tuned", Ht2b, "DPF-PSO-T"),
                              ("heldout2_band", Hd2b, REF), ("heldout2", Hd2, REF)):
                H["ref"] = r
                summary[key] = H
                table_heldout(H, f"table_{key}.md", ref=r)
            sig = lambda H, o, n: H["tests"][f"{o}|{n}"]["sig"]
            orig = {str(n): sig(Ht2, "SPSO-T", n) for n in (10, 20)}     # '+': DPF-PSO-T better, '-': SPSO-T better
            band = {str(n): sig(Ht2b, "SPSO-T", n) for n in (10, 20)}
            h40 = summary.get("heldout_tuned", {}).get("tests", {})
            same_dir = (all(h40[f"SPSO-T|{n}"]["sig"] == orig[str(n)] for n in (10, 20))
                        if all(f"SPSO-T|{n}" in h40 for n in (10, 20)) else None)
            band_both = all(band[str(n)] == "+" for n in (10, 20))
            summary["heldout2_verdict"] = dict(
                original=orig, band=band, band_better_both_n=band_both, same_direction_as_H01_H40=same_dir,
                band_confirmed=bool(band_both and same_dir),
                criterion="design log 15: DPF-PSO-T significantly better than SPSO-T under the band model for "
                          "both n, and the original-model results have the same direction as on H01-H40")
            summary["agl_heldout2"] = {str(n): agl_stats(d2, six, [i for i in d2 if i.endswith(f"_n{n}")])
                                       for n in (10, 20)}
            summary["safety_heldout2"] = safety(d2, six)
            summary["safety_heldout2_band"] = safety(d2b, six)
            summary["components_heldout2"] = {str(n): component_stats(d2, six, [i for i in d2 if i.endswith(f"_n{n}")])
                                              for n in (10, 20)}
            summary["components_heldout2_band"] = {str(n): component_stats(d2b, six, [i for i in d2b if i.endswith(f"_n{n}")])
                                                   for n in (10, 20)}
        else:
            print(f"H41-H80 runs incomplete ({len(d2)} / {len(d2b)} instances), skipped")
    if dh:
        summary["components_heldout"] = {str(n): component_stats(dh, MAIN, [i for i in dh if i.endswith(f"_n{n}")])
                                         for n in (10, 20)}
    # journal version: fixed-lattice one-shot DP baseline (design log, section 16), family G:
    # GridDP vs DPF-PSO on H01-H40 and, if available, H41-H80 (original model), Holm over the 4 tests
    dg = S.load("heldout_grid")
    if dg and dh:
        G1 = {i: {"GridDP": dg[i]["GridDP"], REF: dh[i][REF]} for i in dg if "GridDP" in dg[i] and REF in dh.get(i, {})}
        G2 = {i: {"GridDP": dg[i]["GridDP"], REF: d2[i][REF]} for i in dg if "GridDP" in dg[i] and REF in (d2 or {}).get(i, {})}
        fams = []
        if len(G1) == 80:
            Hg1 = heldout_results(G1, ["GridDP", REF])
            fams.append(("heldout_grid", Hg1))
        if len(G2) == 80:
            Hg2 = heldout_results(G2, ["GridDP", REF], scen=scen2)
            fams.append(("heldout2_grid", Hg2))
        if fams:
            holm_family(*[H for _, H in fams])
            for key, H in fams:
                H["ref"] = REF
                summary[key] = H
                table_heldout(H, f"table_{key}.md")
            summary["components_grid"] = {str(n): component_stats(dg, ["GridDP"], [i for i in dg if i.endswith(f"_n{n}")])
                                          for n in (10, 20)}
            summary["safety_grid"] = safety(dg, ["GridDP"])
        else:
            print(f"GridDP runs incomplete ({len(G1)} / {len(G2)} instances), skipped")
    dp = alias_tuned(S.load("public"))
    tuned = [t + "-T" for t in TUNED]
    if dp and has_all(dp, list(dp), tuned):
        summary["public_tuned"] = design_results(dp, tuned, "public_tuned", ref="DPF-PSO-T")
    for tag, methods in (("public", MAIN), ("eqtime", None),
                         ("n30", MAIN), ("budget5", MAIN), ("b2", MAIN), ("b4", MAIN)):
        dd = alias_tuned(S.load(tag))
        if not dd:
            continue
        if tag == "eqtime":
            # DPF-PSO with its normal budget (from main) against baselines with multiplied budgets
            merged = {i: dict(dd[i], **{REF: dm[i][REF]}) for i in dd if REF in dm.get(i, {})}
            meth = [m for m in MAIN if all(m in merged[i] for i in merged)]
            summary[tag] = design_results(merged, meth, tag)
            table_design(summary[tag], f"table_{tag}.md")
            continue
        meth = [m for m in methods if all(m in dd[i] for i in dd)]
        if REF not in meth or meth != methods:
            print(f"WARNING: {tag} incomplete (methods present in all instances: {meth}); skipped")
            continue
        summary[tag] = design_results(dd, meth, tag)
        table_design(summary[tag], f"table_{tag}.md")
        if tag in ("b2", "b4"):
            summary[f"safety_{tag}"] = safety(dd, meth)
        if tag == "public":
            fig_paths_single(dd, "fig_paths_public.png", "C1")
            if dh:
                summary["fig_band_scenario"] = fig_paths_band(dd, dh, "fig_paths_band.png")
    tc = timecurve_results()
    if tc:
        summary["timecurves"] = tc
    tj = os.path.join(ROOT, "results", "timing", "timing.json")
    if os.path.exists(tj):
        with open(tj, encoding="utf-8") as fh:
            summary["timing"] = json.load(fh)["summary"]
        with open(os.path.join(ROOT, "results", "timing", "multipliers.json"), encoding="utf-8") as fh:
            summary["multipliers"] = json.load(fh)
    tun = {}
    for t in TUNED:
        f = os.path.join(ROOT, "results", "tuning", f"{t}.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                j = json.load(fh)
            tun[t] = dict(selected=j["selected"], selected_rank=j["selected_rank"], default_rank=j["default_rank"],
                          selected_index=j["selected_index"])
    summary["tuning"] = tun
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print("written", OUT)


if __name__ == "__main__":
    main()
