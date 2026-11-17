"""Build the paper and supplement (third revision) from the analysis results.

    python src/analysis2.py           # statistics, tables, figures
    python src/verify_paths.py main heldout public eqtime n30 budget5 b2 heldout_band heldout_eqtime
    python paper/build2.py            # paper.docx/pdf and supplement.docx/pdf
    python paper/build2.py --draft    # missing values are marked instead of failing

All numbers in the manuscript are computed here from
results/analysis/summary.json, results/analysis/safety_check.json and the
verification notes; nothing is typed in by hand.  Data-dependent sentences are
guarded by <!--ASSERT ...--> comments in the templates (checked in
check_prose()).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np

import tables2 as T

TITLE = ("Dynamic-Programming Path Fusion: An Optimal Recombination Operator for Swarm Optimizers "
         "in 3-D UAV Path Planning")
REF = "DPF-PSO"
MAIN = ["PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE", "CMAES", "IDP", "GLS", "DPF-PSO"]
BASE = [m for m in MAIN if m != REF]
BLACK = ["PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE", "CMAES"]       # black-box baselines
TUNED = ["DE-T", "LSHADE-T", "SPSO-T", "IDP-T", "CMAES-T", "DPF-PSO-T"]
ABL = ["DPF-PSO", "DPF-PSO_noG", "DPF-PSO_noR", "DPF-PSO_noL", "DPF-PSO_noA", "DPF-PSO_greedy",
       "DPF-only", "PSO"]
HOSTS = ["PSO", "GWO", "WOA", "HHO", "DE", "LSHADE"]
ROBUST = ["main", "main_tuned", "public", "public_tuned", "n30", "budget5", "b2", "eqtime"]


def load_json(*parts):
    p = os.path.join(ROOT, *parts)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8-sig") as fh:
        return json.load(fh)


def fmt_p(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def and_list(items):
    items = list(items)
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def inst_list(R, method, sign):
    return [T.ilab(i) for i in R["instances"] if R["per_instance"][i]["cmp"][method]["sign"] == sign]


# ------------------------------------------------------------ generated prose
# Sentences about the third-revision experiments are generated from the test
# results: every branch states only what the signs and gaps of the
# corresponding Holm family show, and patterns for which no wording was
# prepared raise an error, so that the text is then written by hand.
NS = ("10", "20")
SIGP = {"+": "significantly better than", "=": "not significantly different from",
        "-": "significantly worse than"}
SIGZH = {"+": "DPF-PSO 显著更好", "=": "无显著差异", "-": "对手显著更好"}
PAT_ORDER = [("+", "+"), ("+", "="), ("=", "+"), ("=", "="), ("+", "-"), ("-", "+"), ("=", "-"), ("-", "="),
             ("-", "-")]


def _t(R, m, n):
    return R["tests"][f"{m}|{n}"]


def _sig(R, m):
    return tuple(_t(R, m, n)["sig"] for n in NS)


def _u(x):
    return T.pct(abs(x), signed=False)


def _name(m, tuned=False):
    return f"tuned {T.lab(m)}" if tuned or m.endswith("-T") else T.lab(m)


def join_clauses(cl):
    cl = list(cl)
    return cl[0] if len(cl) == 1 else ", ".join(cl[:-1]) + ", and " + cl[-1]


def compare_clauses(R, methods, tuned=False):
    """Significance of the reference against each method of a held-out family
    for n = 10 and n = 20, as clauses grouped by pattern, e.g.
    'significantly better than SPSO and DE for both $n$'."""
    groups = {}
    for m in methods:
        groups.setdefault(_sig(R, m), []).append(m)
    out = []
    for pat in PAT_ORDER:
        ms = groups.get(pat)
        if not ms:
            continue
        names = and_list(_name(m, tuned) for m in ms)
        g = ""
        if len(ms) == 1:
            g = f" (Δ̃ = {T.pct(_t(R, ms[0], '10')['gap_median'])} and {T.pct(_t(R, ms[0], '20')['gap_median'])})"
        if pat[0] == pat[1]:
            out.append(f"{SIGP[pat[0]]} {names} for {'either' if pat[0] == '=' else 'both'} $n${g}")
        else:
            out.append(f"{SIGP[pat[0]]} {names} for $n=10$ but {SIGP[pat[1]]} {'it' if len(ms) == 1 else 'them'} "
                       f"for $n=20${g}")
    return out


def vs_variant(R, m, first=False):
    """'worse by 5.4% and 9.0%' etc. for an ablation variant m relative to
    DPF-PSO (gap > 0: the variant is worse)."""
    t = [_t(R, m, n) for n in NS]
    s = tuple(x["sig"] for x in t)
    g = [x["gap_median"] for x in t]
    sfx = (" for $n=10$", " for $n=20$") if first else ("", "")
    if s in (("+", "+"), ("-", "-")):
        word = "worse" if s[0] == "+" else "better"
        return f"{word} by {_u(g[0])}{sfx[0]} and {_u(g[1])}{sfx[1]}"
    if s == ("=", "="):
        return f"not significantly different for either $n$ (Δ̃ = {T.pct(g[0])} and {T.pct(g[1])})"

    def one(x):
        if x["sig"] == "=":
            return f"not significantly different (Δ̃ = {T.pct(x['gap_median'])})"
        return f"{'worse' if x['sig'] == '+' else 'better'} by {_u(x['gap_median'])}"
    return f"{one(t[0])} for $n=10$ and {one(t[1])} for $n=20$"


def family_zh(R, methods):
    """Chinese summary of a held-out family for the handover notes."""
    parts = []
    for m in methods:
        cells = [f"n = {n}：Δ̃ = {T.pct(_t(R, m, n)['gap_median'])}，{SIGZH[_t(R, m, n)['sig']]}" for n in NS]
        parts.append(f"{_name(m)}（{'；'.join(cells)}）")
    return "；".join(parts)


def prose(v, S):
    """Data-dependent sentences of the third revision (see the comment above)."""
    if "heldout_ablation" in S:
        H = S["heldout_ablation"]
        sg, so = _sig(H, "DPF-PSO_greedy"), _sig(H, "DPF-only")
        tail = ""
        if sg == ("+", "+"):
            tail = ", so the optimality of the recombination matters"
        elif "+" in sg and "-" not in sg:
            tail = f", so the optimality of the recombination matters for $n={NS[sg.index('+')]}$"
        v["hab_text"] = (
            f"Without DPF, PSO was {vs_variant(H, 'PSO', first=True)}; with a greedy slot-by-slot choice among "
            f"the same candidates instead of the dynamic program, DPF-PSO was {vs_variant(H, 'DPF-PSO_greedy')}"
            f"{tail}; without global fusion, it was {vs_variant(H, 'DPF-PSO_noG')}; and fusion without PSO moves "
            f"(DPF-only, which spends the budget on repeated fusions) was {vs_variant(H, 'DPF-only')}.")
        gmax = max(abs(_t(H, "DPF-only", n)["gap_median"]) for n in NS)
        small = "small but significant" if gmax <= 0.03 else "significant"
        if so == ("+", "+"):
            v["swarm_gain"] = f"The PSO moves thus add a {small} gain to the dynamic program."
            v["disc_swarm"] = f"The swarm adds a {small} gain"
        elif "+" in so and "-" not in so:
            nn = NS[so.index("+")]
            v["swarm_gain"] = (f"The PSO moves thus add a small gain to the dynamic program, significant only "
                               f"for $n={nn}$.")
            v["disc_swarm"] = f"The swarm adds a small gain, significant only for $n={nn}$"
        elif so == ("=", "="):
            v["swarm_gain"] = "The PSO moves thus added no significant gain to the dynamic program."
            v["disc_swarm"] = "The swarm adds no significant gain to the dynamic program"
        else:
            raise ValueError(f"DPF-only better than DPF-PSO on the held-out scenarios ({so}): write the text by hand")
        v["hab_zh"] = family_zh(H, ["PSO", "DPF-PSO_greedy", "DPF-PSO_noG", "DPF-only"])
    if "heldout_eqtime" in S:
        He = S["heldout_eqtime"]
        v["heq_text"] = "it was " + join_clauses(compare_clauses(He, ["SPSO", "DE", "IDP"])) + "."
        v["heq_zh"] = family_zh(He, ["SPSO", "DE", "IDP"])
        weak, weak_zh = [], []
        for m in ("SPSO", "DE", "IDP"):
            s = _sig(He, m)
            if "-" in s:
                raise ValueError(f"{m} better than DPF-PSO at equal CPU time ({s}): write the text by hand")
            bad = [n for n, x in zip(NS, s) if x != "+"]
            if bad:
                weak.append(f"{T.lab(m)} ({'either $n$' if len(bad) == 2 else f'$n={bad[0]}$'})")
                weak_zh.append(f"{T.lab(m)}（n = {'、'.join(bad)}）")
        v["heq_scope"] = ("; at equal CPU time, it was not significantly better than " + and_list(weak)
                          + " on the held-out scenarios") if weak else ""
        v["heq_weak_zh"] = ("等 CPU 时间下，DPF-PSO 在留出场景上与 " + "、".join(weak_zh) + " 无显著差异"
                            if weak_zh else "等 CPU 时间下，DPF-PSO 在留出场景上仍显著优于三个对手")
    if "heldout_band" in S:
        Hb, Hbt = S["heldout_band"], S["heldout_band_tuned"]
        s10, s20 = _sig(Hbt, "SPSO-T")
        g10, g20 = (T.pct(_t(Hbt, "SPSO-T", n)["gap_median"]) for n in NS)
        intro = ("To test this post hoc explanation, we repeated the held-out runs of SPSO, DE, IDP, DPF-PSO, tuned "
                 "SPSO and tuned DPF-PSO with the band enforced exactly along every segment: a segment closer than "
                 "100 m to the ground or more than 200 m above it is penalized like a clearance violation, and its "
                 "maximum height is computed with the exact clearance method on the mirrored terrain $-Z$ "
                 "(supplement S4).")
        if s10 == "+":
            if s20 == "+":
                tu = (f"Tuned SPSO then lost its advantage: tuned DPF-PSO was significantly better for both $n$ "
                      f"(Δ̃ = {g10} and {g20}; Table III).")
            elif s20 == "=":
                tu = (f"Tuned SPSO then lost its advantage: tuned DPF-PSO was significantly better for $n=10$ "
                      f"(Δ̃ = {g10}) and not significantly different for $n=20$ (Δ̃ = {g20}; Table III).")
            else:
                tu = (f"Tuned DPF-PSO was then significantly better than tuned SPSO for $n=10$ (Δ̃ = {g10}) but "
                      f"significantly worse for $n=20$ (Δ̃ = {g20}; Table III).")
        elif s10 == "=":
            tu = (f"The advantage of tuned SPSO for $n=10$ was then no longer significant (Δ̃ = {g10}); for $n=20$, "
                  f"tuned DPF-PSO was {SIGP[s20]} tuned SPSO (Δ̃ = {g20}; Table III).")
        else:
            tu = (f"Tuned SPSO kept a significant advantage for $n=10$ (Δ̃ = {g10}); for $n=20$, tuned DPF-PSO was "
                  f"{SIGP[s20]} it (Δ̃ = {g20}; Table III). Leaving the band therefore explains only part of "
                  f"its advantage.")
        de = ("With default configurations, DPF-PSO was "
              + join_clauses(compare_clauses(Hb, ["SPSO", "DE", "IDP"])) + ".")
        fe, fails = {}, []
        for R, ms in ((Hb, ["SPSO", "DE", "IDP", "DPF-PSO"]), (Hbt, ["SPSO-T", "DPF-PSO-T"])):
            for m in ms:
                fe[m] = [R[f"n{n}"]["feas"][m] for n in NS]
                fails += [(m, n, R[f"n{n}"]["n_infeasible"][m], R[f"n{n}"]["n_runs"][m]) for n in NS
                          if R[f"n{n}"]["n_infeasible"][m] > 0]
        low = sorted({m for m, *_ in fails}, key=list(fe).index)
        if not fails:
            fs = "All runs ended with a feasible path."
        else:
            fs = ("All runs ended with a feasible path except "
                  + and_list(f"{k} of the {N} runs of {_name(m)} for $n={n}$" for m, n, k, N in fails) + ".")
        af = sum(Hb[f"n{n}"]["all_fail"] for n in NS)
        if af:
            fs += f" On {af} of the 80 instances, no run of these four methods found a feasible path."
        v["band_text"] = " ".join([intro, tu, de, fs])
        v["_band_low"], v["_band_feas"], v["_band_fails"] = low, fe, fails
        if s10 == "+":
            res = {"+": "for both numbers of waypoints", "=": "for 10 waypoints and not significantly different for 20",
                   "-": "for 10 waypoints but worse for 20"}[s20]
            v["abs_spso"] = ("Tuned spherical-vector PSO was better for 10 waypoints because it can merge waypoints "
                             "and thereby leave the altitude band, which the cost model checks only at waypoints; "
                             f"with the band enforced along the path, tuned DPF-PSO was significantly better {res}.")
            v["disc_band"] = ("The altitude band of [5] applies at the waypoints only, and encodings that can merge "
                              "waypoints exploit this: the advantage of tuned SPSO for $n=10$ went with long segments "
                              "outside the band and turned into a significant disadvantage when the band was "
                              "enforced along the path.")
            v["conc_band"] = ("The advantage of tuned SPSO for 10 waypoints relied on leaving the altitude band "
                              "between waypoints and was reversed when the band was enforced along the path.")
        elif s10 == "=":
            r20 = {"+": ", and tuned DPF-PSO was significantly better for 20", "=": "",
                   "-": ", and tuned SPSO was significantly better for 20"}[s20]
            v["abs_spso"] = ("Tuned spherical-vector PSO was better for 10 waypoints because it can merge waypoints "
                             "and thereby leave the altitude band, which the cost model checks only at waypoints; "
                             "with the band enforced along the path, the difference for 10 waypoints was no longer "
                             f"significant{r20}.")
            v["disc_band"] = ("The altitude band of [5] applies at the waypoints only, and encodings that can merge "
                              "waypoints exploit this: the advantage of tuned SPSO for $n=10$ went with long segments "
                              "outside the band and was no longer significant when the band was enforced along the "
                              "path.")
            v["conc_band"] = ("The advantage of tuned SPSO for 10 waypoints relied on leaving the altitude band "
                              "between waypoints and was no longer significant when the band was enforced along the "
                              "path.")
        else:
            v["abs_spso"] = ("Tuned spherical-vector PSO, whose encoding can merge waypoints, was better for 10 "
                             "waypoints, also when the altitude band, which the cost model checks only at waypoints, "
                             "was enforced along the path.")
            v["disc_band"] = ("The altitude band of [5] applies at the waypoints only, and tuned SPSO left it on long "
                              "segments between merged waypoints; it kept its advantage for $n=10$ when the band was "
                              "enforced along the path, so merging waypoints helps also within the band.")
            v["conc_band"] = ("Tuned SPSO, which can merge waypoints, remained better for 10 waypoints also with the "
                              "altitude band enforced along the path.")
        v["hbd_zh"] = family_zh(Hb, ["SPSO", "DE", "IDP"])
        v["hbt_zh"] = family_zh(Hbt, ["SPSO-T"])
        v["band_feas_zh"] = ("所有运行都得到可行航迹" if not fails else
                             "除 " + "、".join(f"{_name(m)}（n = {n}）{N} 次运行中的 {k} 次" for m, n, k, N in fails)
                             + " 外，所有运行都得到可行航迹")


def safety_disagreements(sc):
    """Describe the disagreements between the cost function and the
    independent check (English for the paper, Chinese for the notes)."""
    recs = [dict(r, band=bool(sc[k].get("band_along"))) for k in sc for r in sc[k].get("mismatch_records", [])]
    if any(r["cost_feasible"] and not r["check_feasible"] for r in recs):
        raise ValueError("a path reported as feasible failed the independent check: inspect before writing")
    if not recs:
        return "the cost function and the check agreed on every path", "代价函数与复核在所有航迹上一致", 0.0
    kinds, devs = set(), []
    for r in recs:
        c = {"threat radius": abs(r["margin"])}
        if r["band"]:
            c["lower band limit"] = abs(r["clearance"] - 100.0)
            if r.get("top") is not None:
                c["upper band limit"] = abs(r["top"] - 200.0)
        else:
            c["clearance limit"] = abs(r["clearance"] - 20.0)
        ks = sorted(c, key=c.get)
        d = c[ks[0]]
        # name the limit only if the path is clearly closer to it than to any other
        kinds.add(ks[0] if len(ks) == 1 or c[ks[1]] > 10 * TOL_CHECK else "constraint limit")
        devs.append(d)
    dmax = max(devs)
    e = max(-12, int(np.ceil(np.log10(dmax)))) if dmax > 0 else -12     # stated bound: at least 1e-12 m
    kind = kinds.pop() if len(kinds) == 1 else "constraint limit"
    art = "a" if kind == "constraint limit" else "the"
    zh = {"threat radius": "威胁半径", "lower band limit": "高度带下限", "upper band limit": "高度带上限",
          "clearance limit": " 20 m 离地间隙限值", "constraint limit": "某个约束的限值"}[kind]
    n = len(recs)
    en = (f"the only disagreements were {n} paths that the cost function rejected and the check accepted, all "
          f"within $10^{{{e}}}$ m of {art} {kind} (rounding)")
    cn = (f"另有 {n} 条航迹被代价函数判为不可行、复核在 10⁻⁶ m 容差内判为可行，它们与{zh}的距离都小于 10^{e} m"
          f"（舍入误差；代价函数更严格，不影响安全结论）")
    return en, cn, dmax


TOL_CHECK = 1e-6      # tolerance of src/verify_paths.py


def compute():
    S = load_json("results", "analysis", "summary.json")
    v, t = {"TITLE": TITLE}, {}
    t["TABLE_SCEN"] = T.scenarios()
    # ------------------------------------------------------------ S1-S6
    R = S["main"]
    t["TABLE_MAIN"] = T.design(R, MAIN)
    rk = dict(zip(R["methods"], R["ranks"]))
    v.update(main_best=R["best_count"], main_n=len(R["instances"]), rank_dpf=f"{rk[REF]:.2f}",
             rank_idp=f"{rk['IDP']:.2f}", rank_de=f"{rk['DE']:.2f}")
    v["_main_order"] = sorted(MAIN, key=lambda m: rk[m])
    for m in BASE:
        v[f"wtl_{m}"] = "/".join(map(str, R["wtl"][m]))
    v["idp_plus"] = R["wtl"]["IDP"][0]
    v["de_minus_list"] = and_list(inst_list(R, "DE", "-"))
    v["_minus_methods"] = [m for m in BASE if R["wtl"][m][2] > 0]
    gi = [R["per_instance"][i]["cmp"]["IDP"]["gap"] for i in R["instances"]]
    v.update(idp_gap_lo=T.pct(min(gi)), idp_gap_hi=T.pct(max(gi)))
    gb = [min(R["per_instance"][i]["cmp"][m]["gap"] for m in BLACK) for i in R["instances"]]
    v.update(bb_gap_lo=T.pct(min(gb)), bb_gap_hi=T.pct(max(gb)))
    v["_bb_gaps"] = gb
    if "main_tuned" in S:
        Rt = S["main_tuned"]
        rt = dict(zip(Rt["methods"], Rt["ranks"]))
        v.update(tuned_best=Rt["best_count"], tuned_rank=f"{rt['DPF-PSO-T']:.2f}",
                 spso_t_minus=and_list(inst_list(Rt, "SPSO-T", "-")))
        v["_tuned_minus"] = {m: Rt["wtl"][m][2] for m in Rt["wtl"]}
        parts = [f"tuned {T.lab(m)} ({and_list(inst_list(Rt, m, '-'))})"
                 for m in Rt["methods"] if m != "DPF-PSO-T" and Rt["wtl"][m][2] > 0]
        v["tuned_minus_text"] = and_list(parts) if parts else "no tuned method"
        v["_tuned_minus_n"] = len(parts)
    E = S.get("encoding_main", {})
    if "S6_n10" in E and "SPSO-T" in E["S6_n10"]:
        e = E["S6_n10"]
        v["spso_merged"] = f"{e['SPSO-T']['merged']:.1f}"
        v["spso_maxseg"] = f"{e['SPSO-T']['max_seg']:.0f}"
        v["spso_clear"] = f"{e['SPSO-T']['min_clear']:.1f}"
        v["dpf_maxseg"] = f"{e['DPF-PSO-T']['max_seg']:.0f}"
        v["dpf_clear"] = f"{e['DPF-PSO-T']['min_clear']:.0f}"
        v["_enc"] = e
    MA = S.get("medians_all", {})
    if MA:
        sph = {"SPSO", "SPSO-T", "SPSO-N100", "SPSO-N500"}
        v["_spso_t_lowest"] = [i for i in MA if MA[i]["SPSO-T"] < min(x for m, x in MA[i].items() if m not in sph)]
        v["spso_t_lowest"] = and_list(T.ilab(i) for i in v["_spso_t_lowest"])
    # ------------------------------------------------------------ held-out
    H = S["heldout"]
    Ht = S.get("heldout_tuned")
    t["TABLE_HELDOUT"] = T.heldout(H, Ht)
    v["n_primary"] = len(H["tests"])
    v["n_secondary"] = len(Ht["tests"]) if Ht else 0
    for n in H["n"]:
        hn = H[f"n{n}"]
        v[f"ho_best{n}"] = hn["best_count"]
        v[f"ho_ns{n}"] = hn["n_scen"]
        v[f"ho_rank{n}"] = f"{hn['ranks'][H['methods'].index(REF)]:.2f}"
        v[f"ho_allfail{n}"] = hn["all_fail"]
        v[f"ho_feas{n}"] = f"{100 * hn['feas'][REF]:.0f}"
        for o in BASE:
            tt = H["tests"][f"{o}|{n}"]
            key = o.replace("-", "")
            v[f"ho_win_{key}{n}"] = f"{100 * tt['win']:.0f}"
            v[f"ho_loss_{key}{n}"] = f"{100 * tt['loss']:.0f}"
            v[f"ho_gap_{key}{n}"] = T.pct(tt["gap_median"])
            v[f"ho_gapu_{key}{n}"] = T.pct(abs(tt["gap_median"]), signed=False)
            v[f"ho_p_{key}{n}"] = fmt_p(tt["p_holm"])
            v[f"ho_a12_{key}{n}"] = f"{tt['a12_median']:.2f}"
            v[f"ho_sig_{key}{n}"] = tt["sig"]
        for s, nm in (("+", "plus"), ("=", "eq"), ("-", "minus")):
            ms = [o for o in BASE if H["tests"][f"{o}|{n}"]["sig"] == s]
            v[f"_ho_{nm}{n}"] = ms
            v[f"ho_{nm}_list{n}"] = and_list(T.lab(o) for o in ms)
        for o in ("IDP", "GLS", "SPSO", "DE"):
            w = H["tests"][f"{o}|{n}"]["within_wtl"]
            v[f"ho_within_{o}{n}"] = "/".join(map(str, w))
            v[f"ho_within_plus_{o}{n}"], v[f"ho_within_minus_{o}{n}"] = w[0], w[2]
        bb = [H["tests"][f"{o}|{n}"] for o in BLACK]
        v[f"ho_bb_minwin{n}"] = f"{100 * min(x['win'] for x in bb):.0f}"
        v[f"ho_bb_gaplo{n}"] = T.pct(min(x["gap_median"] for x in bb))
        v[f"ho_bb_gaphi{n}"] = T.pct(max(x["gap_median"] for x in bb))
        v[f"ho_bb_closest{n}"] = T.lab(min(BLACK, key=lambda o: H["tests"][f"{o}|{n}"]["gap_median"]))
        v[f"ho_bb_maxp{n}"] = fmt_p(max(x["p_holm"] for x in bb))
    v["_H"] = H
    v["ho_bb_nsig"] = sum(H["tests"][f"{o}|{n}"]["sig"] == "+" for o in BLACK for n in H["n"])
    v["ho_bb_ntests"] = len(BLACK) * len(H["n"])
    SLOT_BB = [m for m in BLACK if m != "SPSO"]
    for n in H["n"]:
        gs = [H["tests"][f"{o}|{n}"]["gap_median"] for o in SLOT_BB]
        v[f"ho_slot_gaplo{n}"], v[f"ho_slot_gaphi{n}"] = T.pct(min(gs)), T.pct(max(gs))
        v[f"ho_slot_gaplo{n}_u"] = T.pct(min(gs), signed=False)
        v[f"ho_slot_gaphi{n}_u"] = T.pct(max(gs), signed=False)
        v[f"_ho_slot_allsig{n}"] = all(H["tests"][f"{o}|{n}"]["sig"] == "+" for o in SLOT_BB)
    idp_gaps = [H["tests"][f"IDP|{n}"]["gap_median"] for n in H["n"]]
    if Ht:
        idp_gaps += [Ht["tests"][f"IDP-T|{n}"]["gap_median"] for n in Ht["n"]]
    v["_idp_maxabs"] = max(abs(x) for x in idp_gaps)
    v["idp_maxabs"] = T.pct(v["_idp_maxabs"], signed=False)
    v["_slot_gaps"] = {n: [H["tests"][f"{o}|{n}"]["gap_median"] for o in SLOT_BB] for n in H["n"]}
    MG = S.get("merge_heldout", {})
    for n, x in MG.items():
        v[f"mg_a_better{n}"] = x["a_better"]
        v[f"mg_merged{n}"] = f"{x['merged_median']:.1f}"
        v[f"mg_maxseg_a{n}"] = f"{x['maxseg_a']:.0f}"
        v[f"mg_maxseg_b{n}"] = f"{x['maxseg_b']:.0f}"
        v[f"mg_rho{n}"] = f"{x['rho']:.2f}".replace("-", "−")
        v[f"mg_p{n}"] = fmt_p(x["p"])
        v[f"_mg{n}"] = x
    if Ht:
        tun_others = [m for m in Ht["methods"] if m != "DPF-PSO-T"]
        for n in Ht["n"]:
            for o in tun_others:
                tt = Ht["tests"][f"{o}|{n}"]
                key = o.replace("-", "")
                v[f"hot_win_{key}{n}"] = f"{100 * tt['win']:.0f}"
                v[f"hot_loss_{key}{n}"] = f"{100 * tt['loss']:.0f}"
                v[f"hot_gap_{key}{n}"] = T.pct(tt["gap_median"])
                v[f"hot_q1_{key}{n}"] = T.pct(tt["gap_q1"])
                v[f"hot_q3_{key}{n}"] = T.pct(tt["gap_q3"])
                v[f"hot_p_{key}{n}"] = fmt_p(tt["p_holm"])
                v[f"hot_sig_{key}{n}"] = tt["sig"]
            for s, nm in (("+", "plus"), ("=", "eq"), ("-", "minus")):
                ms = [o for o in tun_others if Ht["tests"][f"{o}|{n}"]["sig"] == s]
                v[f"_hot_{nm}{n}"] = ms
                v[f"hot_{nm}_list{n}"] = and_list(f"tuned {T.lab(o)}" for o in ms)
                v[f"hot_{nm}_names{n}"] = and_list(T.lab(o) for o in ms)
    # ------------------------------------- further held-out families (3rd rev.)
    def fam(prefix, R):
        for k, tt in R["tests"].items():
            o, n = k.split("|")
            key = o.replace("-", "").replace("_", "")
            v[f"{prefix}_gap_{key}{n}"] = T.pct(tt["gap_median"])
            v[f"{prefix}_gapu_{key}{n}"] = T.pct(abs(tt["gap_median"]), signed=False)
            v[f"{prefix}_win_{key}{n}"] = f"{100 * tt['win']:.0f}"
            v[f"{prefix}_p_{key}{n}"] = fmt_p(tt["p_holm"])
            v[f"_{prefix}_sig_{key}{n}"] = tt["sig"]
            v[f"_{prefix}_gap_{key}{n}"] = tt["gap_median"]
        v[f"{prefix}_ntests"] = R.get("family_size", len(R["tests"]))

    blocks = []
    if "heldout_ablation" in S:
        fam("hab", S["heldout_ablation"])
        blocks.append(("Ablation:", S["heldout_ablation"],
                       {"PSO": "PSO (no DPF)", "DPF-PSO_noG": "no global fusion",
                        "DPF-PSO_greedy": "greedy, not DP", "DPF-only": "fusion only"}))
    eqm = [S["multipliers"][n][m] for n in ("10", "20") for m in ("IDP", "SPSO", "DE")]
    if "heldout_eqtime" in S:
        fam("heq", S["heldout_eqtime"])
        blocks.append(("Equal CPU time:", S["heldout_eqtime"], {"SPSO": "SPSO", "DE": "DE", "IDP": "IDP"}))
    if "heldout_band" in S:
        fam("hbd", S["heldout_band"])
        fam("hbt", S["heldout_band_tuned"])
        blocks.append(("Band:", S["heldout_band"], {"SPSO": "SPSO", "DE": "DE", "IDP": "IDP"}))
        blocks.append(("Band, tuned:", S["heldout_band_tuned"], {"SPSO-T": "SPSO"}))
    if blocks:
        t["TABLE_HELDOUT_MORE"] = T.heldout_more(blocks)
    v["heq_mult_lo"], v["heq_mult_hi"] = f"{min(eqm):.1f}", f"{max(eqm):.1f}"
    AG = S.get("agl_heldout", {})
    for n, x in AG.items():
        for m, a in x.items():
            key = m.replace("-", "")
            v[f"agl_out_{key}{n}"] = T.pct(a["outside_mean"], signed=False)
            v[f"agl_below_{key}{n}"] = T.pct(a["below_mean"], signed=False)
            v[f"agl_above_{key}{n}"] = T.pct(a["above_mean"], signed=False)
            v[f"_agl_out_{key}{n}"] = a["outside_mean"]
    AT = S.get("anytime_heldout", {})
    if AT:
        SLOT_BB2 = [m for m in BLACK if m != "SPSO"]
        for n in ("10", "20"):
            r = [AT[f"{o}|{n}"]["reach_median"] for o in SLOT_BB2]
            v[f"reach_lo{n}"] = f"{100 * min(r):.0f}"
            v[f"reach_hi{n}"] = f"{100 * max(r):.0f}"
            v[f"_reach{n}"] = r
            v[f"_reach_share{n}"] = [AT[f"{o}|{n}"]["reach_share"] for o in SLOT_BB2]
    # ------------------------------------------------------ ablation, plug-in
    t["TABLE_SOURCES"] = T.sources(S["ablation"], S["plugin"], ABL, [h for h in HOSTS if h != "PSO"])
    for a in ABL:
        if a != REF:
            v[f"abl_{a}"] = "/".join(map(str, S["ablation"][a]["wtl"]))
            v[f"ablgap_{a}"] = T.pct(S["ablation"][a]["gap_mean"])
            v[f"ablplus_{a}"] = S["ablation"][a]["wtl"][0]
            v[f"ablminus_{a}"] = S["ablation"][a]["wtl"][2]
    pg = {h: S["plugin"][h]["gap_mean"] for h in HOSTS}
    v.update(plug_lo=T.pct(min(pg.values())), plug_hi=T.pct(max(pg.values())),
             plug_lo_u=T.pct(min(pg.values()), signed=False), plug_hi_u=T.pct(max(pg.values()), signed=False),
             plug_lo_host=T.lab(min(pg, key=pg.get)), plug_hi_host=T.lab(max(pg, key=pg.get)))
    v["plug_spread"] = T.pct(S["plugin_spread"]["median"], signed=False)
    v["plug_spread0"] = T.pct(S["plugin_spread"]["median_without"], signed=False)
    worse = [f"{T.lab(h)} on {T.ilab(i)}" for h in HOSTS
             for i, s in zip(S["plugin"][h]["insts"], S["plugin"][h]["signs"]) if s == "-"]
    v["plug_worse_n"] = len(worse)
    v["plug_worse_text"] = and_list(worse)
    v["plug_total"] = sum(len(S["plugin"][h]["insts"]) for h in HOSTS)
    v["_plug"] = S["plugin"]
    # ------------------------------------------------------------ robustness
    t["TABLE_ROBUST"] = T.robustness(S, ROBUST)
    for tag in ROBUST:
        if tag in S:
            Rr = S[tag]
            ref = "DPF-PSO-T" if tag.endswith("tuned") else REF
            rr = dict(zip(Rr["methods"], Rr["ranks"]))
            others = [m for m in Rr["methods"] if m != ref]
            v[f"{tag}_best"] = Rr["best_count"]
            v[f"{tag}_n"] = len(Rr["instances"])
            v[f"{tag}_rank"] = f"{rr[ref]:.2f}"
            v[f"{tag}_first"] = T.lab(min(rr, key=rr.get))
            v[f"_{tag}_order"] = sorted(rr, key=rr.get)
            tot = np.sum([Rr["wtl"][m] for m in others], axis=0)
            v[f"{tag}_better"], v[f"{tag}_equal"], v[f"{tag}_worse"] = (int(x) for x in tot)
            v[f"{tag}_total"] = int(tot.sum())
            v[f"{tag}_worse_list"] = and_list(sorted({T.lab(m) for m in others if Rr["wtl"][m][2] > 0}))
            v[f"_{tag}_worse_methods"] = sorted({m for m in others if Rr["wtl"][m][2] > 0})
            v[f"{tag}_worse_pairs"] = "; ".join(f"{T.lab(m)} on {and_list(inst_list(Rr, m, '-'))}"
                                                for m in others if Rr["wtl"][m][2] > 0) or "none"
            v[f"_{tag}_eq_pairs"] = [(m, i) for m in others for i in Rr["instances"]
                                     if Rr["per_instance"][i]["cmp"][m]["sign"] == "="]
            v[f"_{tag}_minus_pairs"] = [(m, i) for m in others for i in Rr["instances"]
                                        if Rr["per_instance"][i]["cmp"][m]["sign"] == "-"]
            for m in others:
                v[f"{tag}_{m.replace('-', '')}_minus"] = and_list(inst_list(Rr, m, "-"))
            cl = "IDP-T" if tag.endswith("tuned") else "IDP"
            if cl in Rr["wtl"]:
                v[f"{tag}_vs_idp"] = "/".join(map(str, Rr["wtl"][cl]))
                v[f"_{tag}_vs_idp"] = Rr["wtl"][cl]
                gi = [Rr["per_instance"][i]["cmp"][cl]["gap"] for i in Rr["instances"]]
                v[f"{tag}_idp_gap"] = T.pct(float(np.mean(gi)))
            bbm = [m for m in others if m.replace("-T", "") in BLACK]
            if bbm:
                gb = [min(Rr["per_instance"][i]["cmp"][m]["gap"] for m in bbm) for i in Rr["instances"]]
                v[f"{tag}_bb_gap_lo"], v[f"{tag}_bb_gap_hi"] = T.pct(min(gb)), T.pct(max(gb))
                v[f"_{tag}_bb_gaps"] = gb
    if "budget5" in S:
        B5, M1 = S["budget5"], S["main"]
        ins = B5["instances"]

        def imp(m):
            return [(M1["per_instance"][i]["median"][m] - B5["per_instance"][i]["median"][m])
                    / M1["per_instance"][i]["median"][m] for i in ins]
        for m in ("SPSO", "GLS", "LSHADE", "DE", "IDP", "DPF-PSO"):
            v[f"b5_imp_{m.replace('-', '')}"] = T.pct(float(np.median(imp(m))), signed=False)
            v[f"_b5_imp_{m.replace('-', '')}"] = float(np.median(imp(m)))
        v["b5_imp_dpf_max"] = T.pct(max(abs(x) for x in imp("DPF-PSO")), signed=False)
        v["_b5_imp_dpf_max"] = max(abs(x) for x in imp("DPF-PSO"))
        v["main_bb_gap_med"] = T.pct(float(np.median(v["_bb_gaps"])), signed=False)
        v["budget5_bb_gap_med"] = T.pct(float(np.median(v["_budget5_bb_gaps"])), signed=False)
        v["_bb_gap_meds"] = (float(np.median(v["_bb_gaps"])), float(np.median(v["_budget5_bb_gaps"])))
    # ------------------------------------------------------------ cost, time
    fe = S["fe_composition"]
    v["fe_glob_one10"] = f"{100 * fe['10']['global_share'] / fe['10']['n_global']:.0f}"
    FB = S.get("fig_band_scenario")
    if FB:
        v["band_scen"] = FB["scenario"].split("_")[0]
        v["band_spso_max"] = f"{FB['SPSO-T']['max_agl']:.0f}"
        v["band_spso_out"] = f"{100 * FB['SPSO-T']['outside']:.0f}"
        v["band_dpf_out"] = f"{100 * FB['DPF-PSO-T']['outside']:.0f}"
        v["_band_spso_merged"] = FB["SPSO-T"]["merged"]
    for n in ("10", "20"):
        f = fe[n]
        v[f"fe_fus{n}"] = f"{100 * (1 - f['host_share']):.0f}"
        v[f"fe_glob{n}"] = f"{100 * f['global_share']:.0f}"
        v[f"fe_loc{n}"] = f"{100 * f['local_share']:.0f}"
        v[f"n_glob{n}"] = f"{f['n_global']:.1f}".replace(".0", "")
        v[f"n_loc{n}"] = f"{f['n_local']:.0f}"
        v[f"k_glob{n}"] = f"{f['k_global']:.0f}"
        v[f"k_loc{n}"] = f"{f['k_local']:.1f}"
    tim, mult = S["timing"], S["multipliers"]
    for n in ("10", "20"):
        med = np.median([tim[n][m]["cpu"] for m in BLACK if m != "CMAES"])
        v[f"t_dpf{n}"] = f"{tim[n][REF]['cpu']:.2f}"
        v[f"t_ratio{n}"] = f"{tim[n][REF]['cpu'] / med:.1f}"
    v["_slower"] = [m for m in BASE if all(tim[n][m]["cpu"] > tim[n][REF]["cpu"] for n in ("10", "20"))]
    allm = [x for n in ("10", "20") for m, x in mult[n].items() if x > 1.0]
    v.update(eq_lo=f"{min(allm):.1f}", eq_hi=f"{max(allm):.1f}")
    # ------------------------------------------------------------ tuning
    for a, x in S["tuning"].items():
        v[f"tun_{a.replace('-', '')}"] = f"{x['selected_rank']:.2f} / {x['default_rank']:.2f}"
    # ------------------------------------------------------------ safety
    sc = load_json("results", "analysis", "safety_check.json")
    v["safe_paths"] = f"{sum(sc[k]['n_paths'] for k in sc):,}"
    v["safe_feas"] = f"{sum(sc[k]['n_feasible'] for k in sc):,}"
    v["safe_mism"] = sum(sc[k]["n_mismatch"] for k in sc)
    v["safe_unsafe"] = sum(sc[k].get("n_reported_feasible_but_infeasible", sc[k]["n_mismatch"]) for k in sc)
    v["safe_conservative"] = sum(sc[k].get("n_rejected_but_feasible", 0) for k in sc)
    v["safe_disagree"], v["safe_disagree_zh"], v["_safe_mism_dev"] = safety_disagreements(sc)
    v["_safe_tags"] = sorted(sc)
    v["safe_band_paths"] = f"{sum(sc[k]['n_paths'] for k in sc if sc[k].get('band_along')):,}"
    bands = [sc[k] for k in sc if sc[k].get("band_along")]
    if bands:
        v["safe_band_attol"] = sum(x.get("n_at_band_tolerance", 0) for x in bands)
        mv = max((x.get("max_band_violation_reported_feasible") or 0.0) for x in bands)
        ex = int(np.floor(np.log10(mv))) if mv > 0 else 0
        v["safe_band_maxviol"] = f"${mv / 10.0 ** ex:.9f}\\times10^{{{ex}}}$" if mv > 0 else "0"
        v["_safe_band_maxviol"] = mv
    v["safe_minclr"] = f"{min(sc[k]['min_clear'] for k in sc if sc[k]['min_clear'] is not None):.3f}"
    v["_sc"] = sc
    keep = [k for k in sc if k != "b2"]
    tg = sum(x["tangent"] for k in keep for x in sc[k]["per_method"].values())
    nf = sum(x["feasible"] for k in keep for x in sc[k]["per_method"].values())
    tgd = sum(sc[k]["per_method"][m]["tangent"] for k in keep for m in sc[k]["per_method"] if m.startswith(REF))
    nfd = sum(sc[k]["per_method"][m]["feasible"] for k in keep for m in sc[k]["per_method"] if m.startswith(REF))
    v["tang_all"] = f"{100 * tg / max(1, nf):.0f}"
    v["tang_dpf"] = f"{100 * tgd / max(1, nfd):.0f}"
    if "b2" in sc:
        pm = sc["b2"]["per_method"]
        v["b2_tang"] = f"{100 * sum(x['tangent'] for x in pm.values()) / max(1, sum(x['feasible'] for x in pm.values())):.0f}"
    # ------------------------------------------------------------ notes
    clr = load_json("notes", "clearance_check.json")["summary"]
    v["clr_nseg"] = f"{clr['n_segments']:,}"
    v["clr_maxdiff"] = fmt_p(clr["max_abs_diff"])
    v["clr_vdiff"] = fmt_p(clr["max_violation_diff"])
    v["clr_dense"] = f"{clr['max_dense_minus_exact']:.3f}"
    ce = load_json("notes", "ceiling_check.json")
    if ce:
        v["ceil_nseg"] = f"{ce['summary']['n_segments']:,}"
        v["ceil_maxdiff"] = fmt_p(ce["summary"]["max_abs_diff"])
    pb = load_json("notes", "pruning_benchmark.json")
    v["prune_speedup"] = f"{pb['K100_dense_s'] / pb['K100_pruned_s']:.1f}"
    v["prune_cases"] = pb["n_cases"]
    cm = load_json("notes", "cmaes_restart_check.json")
    if cm:
        v["cma_old_s3"] = f"{cm['S3_n10']['pycma default tolerances (first runs, discarded)']['median']:.0f}"
        v["cma_new_s3"] = f"{cm['S3_n10'][cm['selected']]['median']:.0f}"
    prose(v, S)
    return v, t, S


def check_prose(v, S, name):
    """Assertions that guard data-dependent sentences of a template."""
    tpl = open(os.path.join(HERE, name), encoding="utf-8").read()
    for m in re.findall(r"<!--ASSERT (.*?)-->", tpl, flags=re.S):
        assert eval(m, {"v": v, "S": S, "np": np}), f"prose assertion failed in {name}: {m}"


def fill(template, v, t, strict=True):
    def rep(m):
        key = m.group(1)
        if key in t:
            return t[key]
        if key in v:
            return str(v[key])
        if strict:
            raise KeyError(f"missing value for @@{key}@@")
        return f"[MISSING:{key}]"
    out = re.sub(r"@@([A-Za-z0-9_\-+]+)@@", rep, template)
    return re.sub(r"<!--ASSERT .*?-->", "", out, flags=re.S)


def build(template, name, v, t, columns=2, strict=True):
    md = fill(open(os.path.join(HERE, template), encoding="utf-8").read(), v, t, strict)
    out_md = os.path.join(HERE, f"{name}.md")
    open(out_md, "w", encoding="utf-8").write(md)
    raw = os.path.join(HERE, f"{name}_raw.docx")
    subprocess.run(["pandoc", out_md, "-o", raw, "--reference-doc", os.path.join(HERE, "reference.docx"),
                    "--resource-path", HERE], check=True, cwd=HERE)
    import docx_tools as D
    D.postprocess(raw, os.path.join(HERE, f"{name}.docx"), columns=columns)
    os.remove(raw)
    pages = D.export_pdf(os.path.join(HERE, f"{name}.docx"), os.path.join(HERE, f"{name}.pdf"))
    import fitz
    doc = fitz.open(os.path.join(HERE, f"{name}.pdf"))
    for k, page in enumerate(doc):
        page.get_pixmap(dpi=100).save(os.path.join(HERE, f"preview_{name}_p{k + 1}.png"))
    print(f"{name}.docx / {name}.pdf: {pages} pages; PDF metadata: "
          f"{ {k: x for k, x in doc.metadata.items() if x} }")
    return pages


def main():
    strict = "--draft" not in sys.argv
    v, t, S = compute()
    import supplement_tables as ST
    t.update(ST.tables(S))
    if strict:
        check_prose(v, S, "paper_template2.md")
        check_prose(v, S, "supplement_template.md")
        check_prose(v, S, "handover_zh_template.md")
    json.dump({k: x for k, x in v.items() if not k.startswith("_")},
              open(os.path.join(HERE, "values2.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not only or "paper" in only:
        build("paper_template2.md", "paper", v, t, columns=2, strict=strict)
    if not only or "supplement" in only:
        build("supplement_template.md", "supplement", v, t, columns=1, strict=strict)
    if not only or "handover" in only:
        zh = fill(open(os.path.join(HERE, "handover_zh_template.md"), encoding="utf-8").read(), v, t, strict)
        zh = zh.replace("$", "")
        with open(os.path.join(ROOT, "投稿准备说明.md"), "w", encoding="utf-8") as fh:
            fh.write(zh)
        print("投稿准备说明.md written")


if __name__ == "__main__":
    main()
