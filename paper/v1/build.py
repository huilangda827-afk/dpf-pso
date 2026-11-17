"""Build the paper: compute all numbers/tables from the result files, fill the
template, convert with pandoc, post-process the .docx and export a PDF.

    python paper/build.py [main_tag] [eqtime_tag]

Every number in the manuscript comes from results/<tag>/*.npz through this
script; nothing is typed in by hand.
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
from scipy import stats

import analyze as A
from env import CostParams, build_scenarios

MAIN = ["PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE", "DPF-PSO"]
BASE = MAIN[:-1]
PLUG = [("PSO", "DPF-PSO"), ("GWO", "GWO+DPF"), ("WOA", "WOA+DPF"), ("DE", "DE+DPF"),
        ("LSHADE", "LSHADE+DPF")]
ABL = ["DPF-PSO", "DPF-PSO_noG", "DPF-PSO_noR", "DPF-PSO_noL", "DPF-PSO_noA",
       "DPF-PSO_greedy", "DPF-only", "PSO"]
ABL_LABEL = {"DPF-PSO": "DPF-PSO (full)", "DPF-PSO_noG": "w/o global fusion",
             "DPF-PSO_noR": "w/o random candidates", "DPF-PSO_noL": "w/o local fusion",
             "DPF-PSO_noA": "fixed σ schedule", "DPF-PSO_greedy": "greedy instead of DP",
             "DPF-only": "DPF without PSO moves", "PSO": "PSO (no DPF)"}
INSTS = A.INSTS
SIGN = {"+": "+", "-": "−", "=": "≈"}


def lab(a):
    return {"LSHADE": "L-SHADE", "LSHADE+DPF": "L-SHADE+DPF"}.get(a, a)


def sig(ref, other):
    return A.wilcoxon_sign(ref, other)[0]


def inst_label(i):
    """Compact label for tables: 'S1, 10'."""
    k, n = i.split("_n")
    return f"{k}, {n}"


def prose_label(i):
    """Label for running text: 'S1 (n = 10)'."""
    k, n = i.split("_n")
    return f"{k} ($n={n}$)"


def compute(d, deq):
    v, t = {}, {}
    # ---------------------------------------------------------- main table
    means = np.array([[d[i][a]["best_f"].mean() for a in MAIN] for i in INSTS])
    ranks = np.array([stats.rankdata(r) for r in means])
    avg_rank = ranks.mean(0)
    fr = stats.friedmanchisquare(*[means[:, k] for k in range(len(MAIN))])
    wtl = {a: [0, 0, 0] for a in BASE}
    rows = ["| Inst. | " + " | ".join(lab(a) for a in MAIN) + " |", "|:--" + "|:-:" * len(MAIN) + "|"]
    for ii, i in enumerate(INSTS):
        best = int(np.argmin(means[ii]))
        cells = []
        for k, a in enumerate(MAIN):
            b = d[i][a]["best_f"]
            s = ""
            if a != "DPF-PSO":
                s = sig(d[i]["DPF-PSO"]["best_f"], b)
                wtl[a][{"+": 0, "=": 1, "-": 2}[s]] += 1
                s = " " + SIGN[s]
            txt = f"{b.mean():.0f} ± {b.std(ddof=1):.0f}{s}"
            cells.append(f"**{txt}**" if k == best else txt)
        tag = "*" if i.split("_")[0] in A.HELD_OUT else ""
        rows.append(f"| {inst_label(i)}{tag} | " + " | ".join(cells) + " |")
    rows.append("| +/≈/− | " + " | ".join("/".join(map(str, wtl[a])) for a in BASE) + " | — |")
    rows.append("| Rank | " + " | ".join(f"{r:.2f}" for r in avg_rank) + " |")
    t["TABLE_MAIN"] = "\n".join(rows)
    order = np.argsort(avg_rank)
    v.update(
        n_inst=len(INSTS), n_runs=len(d[INSTS[0]]["DPF-PSO"]["best_f"]),
        n_best=int(sum(np.argmin(means[ii]) == MAIN.index("DPF-PSO") for ii in range(len(INSTS)))),
        rank_dpf=f"{avg_rank[-1]:.2f}", second=lab(MAIN[order[1]]), rank_second=f"{avg_rank[order[1]]:.2f}",
        rank_spso=f"{avg_rank[MAIN.index('SPSO')]:.2f}",
        fr_chi=f"{fr.statistic:.1f}", fr_p=_sci(fr.pvalue),
        wtl_de="/".join(map(str, wtl["DE"])), wtl_spso="/".join(map(str, wtl["SPSO"])),
        wtl_lshade="/".join(map(str, wtl["LSHADE"])),
        n_sig_all=int(sum(all(sig(d[i]["DPF-PSO"]["best_f"], d[i][a]["best_f"]) == "+" for a in BASE)
                          for i in INSTS)),
    )
    # improvement over the best competitor (by mean) per instance
    imp = []
    for ii, i in enumerate(INSTS):
        bm = min(means[ii][:-1])
        imp.append(100 * (bm - means[ii][-1]) / bm)
    imp = np.array(imp)
    n20 = [k for k, i in enumerate(INSTS) if i.endswith("n20")]
    n10 = [k for k, i in enumerate(INSTS) if i.endswith("n10")]
    v.update(imp_max=f"{imp.max():.1f}", imp_max_inst=inst_label(INSTS[int(np.argmax(imp))]),
             imp_mean20=f"{imp[n20].mean():.1f}", imp_mean10=f"{imp[n10].mean():.1f}",
             imp_min=f"{imp.min():.1f}", imp_min_inst=inst_label(INSTS[int(np.argmin(imp))]))
    v["_imp_max_inst_id"] = INSTS[int(np.argmax(imp))]
    v["_imp"] = dict(zip(INSTS, imp.tolist()))
    # instances on which some baseline is significantly better than DPF-PSO
    losses = [(a, i) for i in INSTS for a in BASE if sig(d[i]["DPF-PSO"]["best_f"], d[i][a]["best_f"]) == "-"]
    v["_losses"] = losses
    v["n_losses"] = len(losses)
    for i in ("S1_n10", "S2_n10"):
        k = i.split("_")[0].lower()
        for a, tag in (("DE", "de"), ("DPF-PSO", "dpf")):
            b = d[i][a]["best_f"]
            v[f"{k}_{tag}_mean"] = f"{b.mean():.0f}"
            v[f"{k}_{tag}_med"] = f"{np.median(b):.0f}"
    ho = [i for i in INSTS if i.split("_")[0] in A.HELD_OUT]
    v["ho_best"] = int(sum(np.argmin(means[INSTS.index(i)]) == len(MAIN) - 1 for i in ho))
    v["n_ho"] = len(ho)
    std_ratio = [d[i]["DPF-PSO"]["best_f"].std(ddof=1) / np.median([d[i][a]["best_f"].std(ddof=1) for a in BASE])
                 for i in INSTS]
    v["std_ratio"] = f"{np.median(std_ratio):.2f}"
    s1 = d["S1_n10"]
    v["s1_de"] = f"{s1['DE']['best_f'].mean():.0f}"
    v["s1_dpf"] = f"{s1['DPF-PSO']['best_f'].mean():.0f}"
    v["s1_gap"] = f"{100 * (s1['DPF-PSO']['best_f'].mean() / s1['DE']['best_f'].mean() - 1):.1f}"
    # ------------------------------------------------------------- plug-in
    rows = ["| Inst. | " + " | ".join(lab(h) for h, _ in PLUG) + " |", "|:--" + "|:-:" * len(PLUG) + "|"]
    acc = {h: [] for h, _ in PLUG}
    cnt = {h: [0, 0, 0] for h, _ in PLUG}
    for i in INSTS:
        cells = []
        for h, hd in PLUG:
            a, b = d[i][h]["best_f"], d[i][hd]["best_f"]
            ch = 100 * (b.mean() - a.mean()) / a.mean()
            s = sig(b, a)
            cnt[h][{"+": 0, "=": 1, "-": 2}[s]] += 1
            acc[h].append(ch)
            cells.append(f"{ch:+.1f}% {SIGN[s]}".replace("+", "+", 1).replace("-", "−"))
        rows.append(f"| {inst_label(i)} | " + " | ".join(cells) + " |")
    rows.append("| Mean | " + " | ".join(f"{np.mean(acc[h]):+.1f}%".replace("-", "−") for h, _ in PLUG) + " |")
    rows.append("| +/≈/− | " + " | ".join("/".join(map(str, cnt[h])) for h, _ in PLUG) + " |")
    t["TABLE_PLUGIN"] = "\n".join(rows)
    worse, chs = [], []
    for i in INSTS:
        for h, hd in PLUG:
            if sig(d[i][hd]["best_f"], d[i][h]["best_f"]) == "-":
                ch = 100 * (d[i][hd]["best_f"].mean() - d[i][h]["best_f"].mean()) / d[i][h]["best_f"].mean()
                worse.append((h, i))
                chs.append(ch)
    v["_plug_worse"] = worse
    v["plug_worse_list"] = ", ".join(f"{lab(h)} on {prose_label(i)}" for h, i in worse) or "none"
    v["plug_worse_range"] = (f"{min(chs):+.1f}% to {max(chs):+.1f}%".replace("-", "−") if chs else "n/a")
    red = {h: -np.mean(acc[h]) for h, _ in PLUG}
    v.update(plug_min=f"{min(red.values()):.1f}", plug_max=f"{max(red.values()):.1f}",
             plug_list=", ".join(f"{red[h]:.1f}" for h, _ in PLUG),
             plug_n_better=sum(c[0] for c in cnt.values()), plug_n_worse=sum(c[2] for c in cnt.values()),
             plug_n_total=len(PLUG) * len(INSTS))
    # ------------------------------------------------------------ ablation
    am = np.array([[d[i][a]["best_f"].mean() for a in ABL] for i in INSTS])
    ar = np.array([stats.rankdata(r) for r in am]).mean(0)
    rows = ["| Variant | Rank | Δ mean | +/≈/− |", "|:--|:-:|:-:|:-:|"]
    for k, a in enumerate(ABL):
        rel = 100 * np.mean([(d[i][a]["best_f"].mean() - d[i]["DPF-PSO"]["best_f"].mean())
                             / d[i]["DPF-PSO"]["best_f"].mean() for i in INSTS])
        if a == "DPF-PSO":
            w = "—"
            rel_s = "0"
        else:
            c = [0, 0, 0]
            worse_on = []
            for i in INSTS:
                s = sig(d[i]["DPF-PSO"]["best_f"], d[i][a]["best_f"])
                c[{"+": 0, "=": 1, "-": 2}[s]] += 1
                if s == "+":
                    worse_on.append(i)
            w = "/".join(map(str, c))
            rel_s = f"{rel:+.1f}%".replace("-", "−")
            v[f"abl_{a}"] = w
            v[f"ablrel_{a}"] = f"{rel:.1f}"
            v[f"ablworse_{a}"] = len(worse_on)
            v[f"ablworse20_{a}"] = sum(i.endswith("n20") for i in worse_on)
        rows.append(f"| {ABL_LABEL[a]} | {ar[k]:.2f} | {rel_s} | {w} |")
    t["TABLE_ABL"] = "\n".join(rows)
    # ------------------------------------------------------------- runtime
    # CPU seconds per run from the controlled benchmark (results/timing)
    with open(os.path.join(ROOT, "results", "timing", "timing.json"), encoding="utf-8-sig") as fh:
        tj = json.load(fh)
    tm = {n: {a: tj["summary"][str(n)][a]["cpu"] for a in MAIN} for n in (10, 20)}
    v["timing_runs"] = tj["meta"]["runs"]
    v["timing_workers"] = tj["meta"]["workers"]
    for n in (10, 20):
        ins = [i for i in INSTS if i.endswith(f"n{n}")]
        budget = 30000 if n == 10 else 60000
        v[f"dpshare{n}"] = f"{100 * np.mean([d[i]['DPF-PSO']['dp_fes'].mean() for i in ins]) / budget:.0f}"
        v[f"t_dpf{n}"] = f"{tm[n]['DPF-PSO']:.1f}"
        bt = [tm[n][a] for a in BASE]
        v[f"t_ratio{n}"] = f"{tm[n]['DPF-PSO'] / np.median(bt):.1f}"
        v[f"t_base_lo{n}"] = f"{min(bt):.1f}"
        v[f"t_base_hi{n}"] = f"{max(bt):.1f}"
    if deq:
        mult = deq["_mult"]
        wt = {a: [0, 0, 0] for a in BASE}
        best_cnt = 0
        for i in INSTS:
            ref = d[i]["DPF-PSO"]["best_f"]
            mm = [deq[i][a]["best_f"].mean() for a in BASE if a in deq[i]]
            if ref.mean() < min(mm):
                best_cnt += 1
            for a in BASE:
                if a in deq[i]:
                    wt[a][{"+": 0, "=": 1, "-": 2}[sig(ref, deq[i][a]["best_f"])]] += 1
        rows = ["| Method | CPU s, n = 10 | CPU s, n = 20 | Budget × (10 / 20) | +/≈/− equal time |",
                "|:--|:-:|:-:|:-:|:-:|"]
        for a in MAIN:
            if a == "DPF-PSO":
                rows.append(f"| {lab(a)} | {tm[10][a]:.2f} | {tm[20][a]:.2f} | 1.0 / 1.0 | — |")
            else:
                rows.append(f"| {lab(a)} | {tm[10][a]:.2f} | {tm[20][a]:.2f} | "
                            f"{mult['10'][a]:.1f} / {mult['20'][a]:.1f} | {'/'.join(map(str, wt[a]))} |")
        v["eq_best"] = best_cnt
        v["eq_wtl_de"] = "/".join(map(str, wt["DE"]))
        v["eq_wtl_spso"] = "/".join(map(str, wt["SPSO"]))
        v["eq_n_worse"] = sum(w[2] for w in wt.values())
        v["eq_n_better"] = sum(w[0] for w in wt.values())
        v["eq_total"] = len(BASE) * len(INSTS)
        v["eq_worse_list"] = _worse_list(d, deq)
        v["eq_mult_lo"] = f"{min(min(mult['10'].values()), min(mult['20'].values())):.1f}"
        v["eq_mult_hi"] = f"{max(max(mult['10'].values()), max(mult['20'].values())):.1f}"
    else:
        rows = ["| Method | CPU s, n = 10 | CPU s, n = 20 |", "|:--|:-:|:-:|"]
        for a in MAIN:
            rows.append(f"| {lab(a)} | {tm[10][a]:.2f} | {tm[20][a]:.2f} |")
    t["TABLE_TIME"] = "\n".join(rows)
    # ------------------------------------------- pruning benchmark, safety
    with open(os.path.join(ROOT, "notes", "pruning_benchmark.json"), encoding="utf-8-sig") as fh:
        pb = json.load(fh)
    assert not pb.get("provisional"), "run src/check_pruning.py for final timings"
    v.update(t_dense=f"{pb['K100_dense_s']:.2f}", t_pruned=f"{pb['K100_pruned_s']:.2f}",
             prune_speedup=f"{pb['K100_dense_s'] / pb['K100_pruned_s']:.1f}", prune_cases=pb["n_cases"])
    with open(os.path.join(ROOT, "results", A_TAG[0], "safety_check.json"), encoding="utf-8-sig") as fh:
        sc_ = json.load(fh)
    v.update({k: sc_[k] for k in ("n_paths_main", "n_feasible_main", "n_penetrate", "n_tangent_main",
                                  "min_gnd_all", "min_gnd_dpf", "n_infeasible_main")})
    from collections import Counter
    cnt_inf = Counter(" ".join(s.split()[:2]) for s in sc_["infeasible_main"])
    words = {0: "No", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    ninf = sc_["n_infeasible_main"]
    if ninf == 0:
        v["infeasible_main"] = "No final path was infeasible"
    elif len(cnt_inf) == 1:
        k = next(iter(cnt_inf))
        v["infeasible_main"] = (f"{words.get(ninf, ninf)} final paths, all from {lab(k.split()[0])} on "
                                f"{prose_label(k.split()[1])}, were infeasible")
    else:
        v["infeasible_main"] = (f"{words.get(ninf, ninf)} final paths were infeasible ("
                                + "; ".join(f"{c} from {lab(k.split()[0])} on {prose_label(k.split()[1])}"
                                            for k, c in cnt_inf.items()) + ")")
    # ------------------------------------------------- SPSO population size
    sp, wtls, losses_sp = {}, [], []
    names = {"SPSO": "50", "SPSO-N100": "100", "SPSO-N500": "500"}
    if all(a in d[i] for i in INSTS for a in names):
        for a in names:
            sp[a] = np.mean([d[i][a]["best_f"].mean() / d[i]["DPF-PSO"]["best_f"].mean() - 1 for i in INSTS])
            c = [0, 0, 0]
            for i in INSTS:
                s = sig(d[i]["DPF-PSO"]["best_f"], d[i][a]["best_f"])
                c[{"+": 0, "=": 1, "-": 2}[s]] += 1
                if s == "-":
                    losses_sp.append(f"{names[a]} particles on {prose_label(i)}")
            wtls.append("/".join(map(str, c)))
        v["spso_gaps"] = ", ".join(f"{100 * sp[a]:.1f}" for a in names)
        v["spso_wtls"] = ", ".join(wtls)
        v["spso_losses"] = ("; DPF-PSO was significantly worse only against " + ", ".join(losses_sp)) \
            if losses_sp else ""
        v["spso_min_wins"] = min(int(w.split("/")[0]) for w in wtls)
    # ------------------------------------------------------- scenario table
    sc = build_scenarios()
    rows = ["| Scen. | Elevation (m) | Threats | Start → goal | Use |", "|:--|:-:|:-:|:-:|:-:|"]
    for k in ("S1", "S2", "S3", "S4", "S5", "S6"):
        s = sc[k]
        T = s.terrain
        rows.append(f"| {k} | {T.Z.min():.0f}–{T.Z.max():.0f} | {len(s.threats)} | "
                    f"({s.start[0]:.0f}, {s.start[1]:.0f}) → ({s.goal[0]:.0f}, {s.goal[1]:.0f}) | "
                    f"{'test only' if k in A.HELD_OUT else 'design + test'} |")
    t["TABLE_SCEN"] = "\n".join(rows)
    p = CostParams()
    v.update(P0=f"{p.P0:.0e}".replace("e+0", "^").replace("^", "×10^") + "^", lam=f"{p.lam:.0f}",
             hc=f"{p.h_clear:.0f}", ns=p.n_samples)
    return v, t


def check_prose(v, d):
    """Data-dependent statements in paper_template.md; fail loudly if the data
    no longer supports them."""
    f = lambda k: float(v[k])  # noqa: E731
    assert v["_imp_max_inst_id"] == "S3_n20", v["_imp_max_inst_id"]
    assert f("imp_mean20") > f("imp_mean10")
    assert v["_losses"] == [("DE", "S1_n10"), ("DE", "S2_n10")], v["_losses"]
    # ablation: all significant losses of the noR / noL variants have n = 20
    for a in ("DPF-PSO_noR", "DPF-PSO_noL"):
        assert v[f"ablworse_{a}"] == v[f"ablworse20_{a}"] > 0, a
    assert v["dpshare10"] == v["dpshare20"], (v["dpshare10"], v["dpshare20"])
    # margins largest on S3 and S5 with n = 20 (Fig. 3 discussion)
    top2 = sorted(INSTS, key=lambda i: -v["_imp"][i])[:2]
    assert set(top2) == {"S3_n20", "S5_n20"}, top2
    assert f("s1_de_mean") < f("s1_dpf_mean") and f("s1_de_med") < f("s1_dpf_med")
    assert f("s2_de_med") < f("s2_dpf_med") and f("s2_de_mean") > f("s2_dpf_mean")
    gaps = [float(x) for x in v["spso_gaps"].split(", ")]
    assert gaps[0] > gaps[1] > gaps[2] > 0 and v["spso_min_wins"] >= 10, (gaps, v["spso_min_wins"])
    assert v["abl_DPF-PSO_noA"].startswith("0/"), v["abl_DPF-PSO_noA"]
    assert f("ablrel_DPF-PSO_noG") == max(f(k) for k in v if k.startswith("ablrel_") and k != "ablrel_PSO")
    assert v["ho_best"] == v["n_ho"]
    assert all(i.endswith("n10") and h in ("DE", "LSHADE") for h, i in v["_plug_worse"]), v["_plug_worse"]
    if "eq_best" in v:
        assert v["eq_best"] >= 1
    if "eqp_lo" in v:   # the pool-based multipliers must be the larger ones
        assert float(v["eqp_hi"]) >= float(v["eq_mult_hi"]), (v["eqp_hi"], v["eq_mult_hi"])


def _worse_list(d, deq):
    out = []
    for i in INSTS:
        for a in BASE:
            if a in deq[i] and sig(d[i]["DPF-PSO"]["best_f"], deq[i][a]["best_f"]) == "-":
                out.append(f"{lab(a)} on {prose_label(i)}")
    return ", ".join(out) if out else "none"


def _sci(p):
    """TeX form for use inside $...$."""
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def load_eq(tag):
    if not tag:
        return None
    path = os.path.join(ROOT, "results", tag)
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "multipliers.json")):
        return None
    d = A.load(tag)
    if any(len(d.get(i, {})) < len(BASE) for i in INSTS):
        return None                       # incomplete run
    with open(os.path.join(path, "multipliers.json"), encoding="utf-8-sig") as fh:
        d["_mult"] = json.load(fh)
    return d


def eq_stats(d, deq):
    wt = {a: [0, 0, 0] for a in BASE}
    best_cnt = 0
    for i in INSTS:
        ref = d[i]["DPF-PSO"]["best_f"]
        if ref.mean() < min(deq[i][a]["best_f"].mean() for a in BASE):
            best_cnt += 1
        for a in BASE:
            wt[a][{"+": 0, "=": 1, "-": 2}[sig(ref, deq[i][a]["best_f"])]] += 1
    mult = deq["_mult"]
    lo = min(min(mult["10"].values()), min(mult["20"].values()))
    hi = max(max(mult["10"].values()), max(mult["20"].values()))
    return dict(best=best_cnt, better=sum(w[0] for w in wt.values()), worse=sum(w[2] for w in wt.values()),
                total=len(BASE) * len(INSTS), lo=f"{lo:.1f}", hi=f"{hi:.1f}", wt=wt)


def fill(template, v, t, strict=True):
    def rep(m):
        key = m.group(1)
        if key in t:
            return t[key]
        if key not in v:
            if strict:
                raise KeyError(f"missing value for @@{key}@@")
            return f"[MISSING:{key}]"
        return str(v[key])
    return re.sub(r"@@([A-Za-z0-9_\-+]+)@@", rep, template)


A_TAG = ["main_v2"]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main_v2"
    eqtag = sys.argv[2] if len(sys.argv) > 2 else "eqtime"
    A_TAG[0] = tag
    d = A.load(tag)
    deq = load_eq(eqtag)
    v, t = compute(d, deq)
    dpool = load_eq("eqtime_pool")
    if deq is not None and dpool is not None:
        s = eq_stats(d, dpool)
        v.update(eqp_best=s["best"], eqp_better=s["better"], eqp_worse=s["worse"], eqp_total=s["total"],
                 eqp_lo=s["lo"], eqp_hi=s["hi"])
    elif deq is not None and "--draft" not in sys.argv:
        raise SystemExit("eqtime_pool results missing; rerun after src/run_eqtime.py --ref main_v2 --tag eqtime_pool")
    if "--draft" not in sys.argv:
        check_prose(v, d)
    with open(os.path.join(HERE, "values.json"), "w", encoding="utf-8") as fh:
        json.dump({k: val for k, val in v.items() if not k.startswith("_")}, fh, indent=1, ensure_ascii=False)
    src = open(os.path.join(HERE, "paper_template.md"), encoding="utf-8").read()
    if deq is None:
        src = re.sub(r"<!--EQ-->.*?<!--/EQ-->", "", src, flags=re.S)
    else:
        src = src.replace("<!--EQ-->", "").replace("<!--/EQ-->", "")
    md = fill(src, v, t, strict="--draft" not in sys.argv)
    out_md = os.path.join(HERE, "paper.md")
    open(out_md, "w", encoding="utf-8").write(md)
    raw = os.path.join(HERE, "paper_raw.docx")
    subprocess.run(["pandoc", out_md, "-o", raw, "--reference-doc", os.path.join(HERE, "reference.docx"),
                    "--resource-path", HERE], check=True, cwd=HERE)
    import docx_tools as T
    n_eq = T.postprocess(raw, os.path.join(HERE, "paper.docx"))
    pages = T.export_pdf(os.path.join(HERE, "paper.docx"), os.path.join(HERE, "paper.pdf"))
    print(f"paper.docx / paper.pdf written: {pages} pages, {n_eq} numbered equations")
    import fitz
    doc = fitz.open(os.path.join(HERE, "paper.pdf"))
    for k, page in enumerate(doc):
        page.get_pixmap(dpi=100).save(os.path.join(HERE, f"preview_p{k + 1}.png"))


if __name__ == "__main__":
    main()
