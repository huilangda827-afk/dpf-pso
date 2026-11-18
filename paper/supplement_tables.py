"""Tables of the supplementary material (from results/analysis/summary.json)."""
from __future__ import annotations

import json
import os

import numpy as np

import tables2 as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = "DPF-PSO"
MAIN = ["PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE", "CMAES", "IDP", "GLS", "DPF-PSO"]


def _load(*p):
    f = os.path.join(ROOT, *p)
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8-sig") as fh:
        return json.load(fh)


def heldout_detail(H):
    rows = ["| Compared method | n | Win / loss | Δ̃ [Q1, Q3] | Median A12 | Within-scenario +/≈/− | p | p~H~ |",
            "|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for o in [m for m in H["methods"] if m not in (REF, "DPF-PSO-T")]:
        for n in H["n"]:
            t = H["tests"][f"{o}|{n}"]
            rows.append(f"| {T.lab(o)} | {n} | {100 * t['win']:.0f}% / {100 * t['loss']:.0f}% | "
                        f"{T.pct(t['gap_median'])} [{T.pct(t['gap_q1'])}, {T.pct(t['gap_q3'])}] | "
                        f"{t['a12_median']:.2f} | {'/'.join(map(str, t['within_wtl']))} | {T.ptxt(t['p'])} | "
                        f"{T.ptxt(t['p_holm'])} |")
    return "\n".join(rows)


def heldout_ranks(H):
    ms = H["methods"]
    rows = ["| | " + " | ".join(T.SHORT.get(m, T.lab(m)) for m in ms) + " |", "|:--" + "|:-:" * len(ms) + "|"]
    for n in H["n"]:
        hn = H[f"n{n}"]
        rows.append(f"| Mean rank, n = {n} | " + " | ".join(f"{x:.2f}" for x in hn["ranks"]) + " |")
        rows.append(f"| Feasible runs, n = {n} | " + " | ".join(f"{100 * hn['feas'][m]:.0f}%" for m in ms) + " |")
    return "\n".join(rows)


def sensitivity(S):
    rows = ["| Variant | +/≈/− (DPF-PSO vs. variant) | Δ (mean over 12 instances) |", "|:--|:-:|:-:|"]
    for k, x in S.get("sensitivity", {}).items():
        rows.append(f"| {T.lab(k)} | {'/'.join(map(str, x['wtl']))} | {T.pct(x['gap_mean'])} |")
    return "\n".join(rows)


def fe_table(fe):
    rows = ["| n | Host FEs | Global fusion | Local fusion | Global fusions | Local fusions | Mean K (global) | Mean K (local) | Improving global / local |",
            "|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for n in ("10", "20"):
        f = fe[n]
        rows.append(f"| {n} | {100 * f['host_share']:.0f}% | {100 * f['global_share']:.0f}% | {100 * f['local_share']:.0f}% | "
                    f"{f['n_global']:.1f} | {f['n_local']:.0f} | {f['k_global']:.0f} | {f['k_local']:.1f} | "
                    f"{f['imp_global']:.1f} / {f['imp_local']:.1f} |")
    return "\n".join(rows)


SAFETY_LABEL = {"main": "S1–S6", "heldout": "H01–H40", "public": "C1", "eqtime": "Equal CPU time",
                "n30": "n = 30", "budget5": "5× budget", "b2": "b~2~ = 10", "b4": "b~4~ = 10",
                "heldout_eqtime": "H01–H40, equal CPU time", "heldout_band": "H01–H40, band along segments",
                "heldout2": "H41–H80", "heldout2_band": "H41–H80, band along segments",
                "heldout_grid": "H01–H80, lattice DP"}


def safety_table(sc):
    rows = ["| Experiment | Paths | Independently feasible | Disagreements | Reported feasible but infeasible | "
            "Min. exact clearance (m) | Max. height above ground (m) |", "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    missing = [k for k in sc if k not in SAFETY_LABEL]
    assert not missing, f"no label for safety-check tags {missing}"
    for k in [t for t in SAFETY_LABEL if t in sc]:
        x = sc[k]
        mc = "—" if x["min_clear"] is None else f"{x['min_clear']:.3f}"
        mt = "—" if x.get("max_top") is None else f"{x['max_top']:.3f}"
        rows.append(f"| {SAFETY_LABEL[k]} | {x['n_paths']:,} | {x['n_feasible']:,} | {x['n_mismatch']} | "
                    f"{x.get('n_reported_feasible_but_infeasible', '—')} | {mc} | {mt} |")
    recs = [r for k in sc for r in sc[k].get("mismatch_records", [])]
    if recs:
        rows.append("")
        rows.append("Disagreements: " + "; ".join(
            f"{T.lab(r['algo'])} on {T.ilab(r['inst'])}, run {r['run']}: exact clearance "
            f"{r['clearance']:.12f} m"
            + ("" if r.get("top") is None else f", maximum height {r['top']:.12f} m")
            + f", threat margin {r['margin']:.3f} m, rejected by the cost function and "
            f"accepted by the check (tolerance 10^−6^ m)" for r in recs) + ".")
    return "\n".join(rows)


def safety_methods(sc, tag="main"):
    rows = ["| Method | Feasible paths | Min. clearance (m) | Paths within 1 mm of R+D |", "|:--|:-:|:-:|:-:|"]
    pm = sc[tag]["per_method"]
    for m in MAIN:
        if m in pm:
            x = pm[m]
            mc = "—" if x["min_clear"] is None else f"{x['min_clear']:.3f}"
            rows.append(f"| {T.lab(m)} | {x['feasible']}/{x['n']} | {mc} | {x['tangent']} ({100 * x['tangent'] / max(1, x['feasible']):.0f}%) |")
    return "\n".join(rows)


def timing_detail(tj, mult):
    """CPU and wall time per run from the pinned benchmark (mean ± SD over
    6 instances x 5 runs) and the equal-time multipliers."""
    pi = tj["per_instance"]
    rows = ["| Method | CPU s, n = 10 | CPU s, n = 20 | Wall s, n = 10 | Wall s, n = 20 | Multiplier (10 / 20) |",
            "|:--|:-:|:-:|:-:|:-:|:-:|"]
    for m in MAIN:
        cells = []
        for key in ("cpu", "wall"):
            for n in (10, 20):
                x = np.concatenate([pi[i][m][key] for i in pi if i.endswith(f"_n{n}")])
                cells.append(f"{x.mean():.2f} ± {x.std(ddof=1):.2f}")
        mm = "1.0 / 1.0 (reference)" if m == REF else f"{mult['10'][m]:.1f} / {mult['20'][m]:.1f}"
        rows.append(f"| {T.lab(m)} | " + " | ".join(cells) + f" | {mm} |")
    return "\n".join(rows)


def anytime_table(A):
    rows = ["| Compared method | n | Δ̃ at 50% of the budget | Δ̃ at 100% | Budget share at which DPF-PSO reaches the method's final median | Scenarios where it does |",
            "|:--|:-:|:-:|:-:|:-:|:-:|"]
    for k, x in A.items():
        o, n = k.split("|")
        r = "never (median)" if not np.isfinite(x["reach_median"]) else f"{100 * x['reach_median']:.0f}%"
        rows.append(f"| {T.lab(o)} | {n} | {T.pct(x['gap_0.5'])} | {T.pct(x['gap_1'])} | {r} | {100 * x['reach_share']:.0f}% |")
    return "\n".join(rows)


def agl_table(AG, title_variant=""):
    rows = ["| Method | n | Track below 100 m | Track above 200 m | Median min. height (m) | Median max. height (m) |",
            "|:--|:-:|:-:|:-:|:-:|:-:|"]
    for n, x in AG.items():
        for m, a in x.items():
            rows.append(f"| {T.lab(m) if not m.endswith('-T') else T.lab(m) + ' (tuned)'} | {n} | "
                        f"{T.pct(a['below_mean'], signed=False)} | {T.pct(a['above_mean'], signed=False)} | "
                        f"{a['min_median']:.0f} | {a['max_median']:.0f} |")
    return "\n".join(rows)


def band_feas_table(Hb, Hbt):
    """Share of runs with a feasible final path in the altitude-band variant."""
    rows = ["| Method | Feasible runs, n = 10 | Feasible runs, n = 20 |", "|:--|:-:|:-:|"]
    for H, ms in ((Hb, [m for m in Hb["methods"]]), (Hbt, [m for m in Hbt["methods"]])):
        for m in ms:
            name = T.lab(m) + (" (tuned)" if m.endswith("-T") else "")
            rows.append(f"| {name} | " + " | ".join(f"{100 * H[f'n{n}']['feas'][m]:.1f}%" for n in H["n"]) + " |")
    return "\n".join(rows)


def tables(S):
    t = {}
    for key, name in (("heldout_ablation", "S_HO_ABL"), ("heldout_eqtime", "S_HO_EQ"),
                      ("heldout_band", "S_HO_BAND"), ("heldout_band_tuned", "S_HO_BANDT")):
        if key in S:
            t[name] = heldout_detail(S[key])
    if "heldout_band" in S and "heldout_band_tuned" in S:
        t["S_HO_BAND_FEAS"] = band_feas_table(S["heldout_band"], S["heldout_band_tuned"])
    if "anytime_heldout" in S:
        t["S_ANYTIME"] = anytime_table(S["anytime_heldout"])
    if "agl_heldout" in S:
        t["S_AGL"] = agl_table(S["agl_heldout"])
    if "agl_band" in S:
        t["S_AGL_BAND"] = agl_table(S["agl_band"])
    if "main" in S:
        t["S_MAIN"] = T.design(S["main"], MAIN)
    if "ablation" in S and "plugin" in S:
        t["S_SOURCES"] = T.sources(S["ablation"], S["plugin"],
                                   ["DPF-PSO", "DPF-PSO_noG", "DPF-PSO_noR", "DPF-PSO_noL", "DPF-PSO_noA",
                                    "DPF-PSO_greedy", "DPF-only", "PSO"],
                                   ["GWO", "WOA", "HHO", "DE", "LSHADE"])
    tj = _load("results", "timing", "timing.json")
    if tj and "multipliers" in S:
        t["S_TIMING"] = timing_detail(tj, S["multipliers"])
    if "main" in S:
        t["S_MAIN10"] = T.design_full(S["main"], 10)
        t["S_MAIN20"] = T.design_full(S["main"], 20)
    if "main_tuned" in S:
        t["S_TUNED"] = T.design(S["main_tuned"], ref="DPF-PSO-T")
    if "heldout" in S:
        t["S_HELDOUT"] = heldout_detail(S["heldout"])
        t["S_HELDOUT_RANKS"] = heldout_ranks(S["heldout"])
    if "heldout_tuned" in S:
        t["S_HELDOUT_TUNED"] = heldout_detail(S["heldout_tuned"])
    for tag in ("public", "n30", "budget5", "b2", "eqtime"):
        if tag in S:
            t[f"S_{tag.upper()}"] = T.design(S[tag], star=())
    t["S_SENS"] = sensitivity(S)
    if "fe_composition" in S:
        t["S_FE"] = fe_table(S["fe_composition"])
    if "tuning" in S:
        t["S_TUNING"] = T.tuning_table(S["tuning"], None)
    sc = _load("results", "analysis", "safety_check.json")
    if sc:
        t["S_SAFETY"] = safety_table(sc)
        if "main" in sc:
            t["S_SAFETY_M"] = safety_methods(sc, "main")
        if "b2" in sc:
            t["S_SAFETY_B2"] = safety_methods(sc, "b2")
    return t
