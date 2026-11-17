"""Markdown tables for the paper and the supplement, generated from
results/analysis/summary.json (see src/analysis2.py)."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from env import get_scenario  # noqa: E402

REF = "DPF-PSO"
LAB = {"LSHADE": "L-SHADE", "CMAES": "CMA-ES", "LSHADE+DPF": "L-SHADE+DPF", "DE-T": "DE",
       "LSHADE-T": "L-SHADE", "SPSO-T": "SPSO", "IDP-T": "IDP", "DPF-PSO-T": "DPF-PSO", "CMAES-T": "CMA-ES",
       "DPF-PSO_noG": "w/o global fusion", "DPF-PSO_noR": "w/o random candidates",
       "DPF-PSO_noL": "w/o local fusion", "DPF-PSO_noA": "fixed σ schedule",
       "DPF-PSO_greedy": "greedy instead of DP", "DPF-only": "fusion only (no PSO moves)",
       "DE-NP5D": "DE, NP = 5D", "DE-NP10D": "DE, NP = 10D", "SPSO-N100": "SPSO, 100 particles",
       "SPSO-N500": "SPSO, 500 particles"}
SHORT = {"LSHADE": "L-SH.", "CMAES": "CMA", "DPF-PSO": "DPF", "LSHADE-T": "L-SH.", "DPF-PSO-T": "DPF",
         "DE-T": "DE", "SPSO-T": "SPSO", "IDP-T": "IDP", "CMAES-T": "CMA"}
SIGN = {"+": "+", "=": "≈", "-": "−"}


def lab(a):
    return LAB.get(a, a)


def ilab(inst):
    """'S3_n20' -> 'S3/20' (scenario S3 with n = 20 waypoints)."""
    k, n = inst.split("_n")
    return f"{k}/{n}"


def pct(x, d=1, signed=True):
    if round(100 * x, d) == 0:
        x = 0.0                      # no "−0.0%"
    s = f"{100 * x:+.{d}f}%" if signed else f"{100 * x:.{d}f}%"
    return s.replace("-", "−")


def ptxt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------
def scenarios():
    rows = ["| Scen. | Elevation (m) | Threats | Start → goal | Use |", "|:--|:-:|:-:|:-:|:-:|"]
    for k in ("S1", "S2", "S3", "S4", "S5", "S6"):
        s = get_scenario(k)
        T = s.terrain
        rows.append(f"| {k} | {T.Z.min():.0f}–{T.Z.max():.0f} | {len(s.threats)} | "
                    f"({s.start[0]:.0f}, {s.start[1]:.0f}) → ({s.goal[0]:.0f}, {s.goal[1]:.0f}) | "
                    f"{'tuning + test' if k in ('S1', 'S2', 'S3', 'S4') else 'test'} |")
    hs = [get_scenario(f"H{k:02d}") for k in range(1, 41)]
    zmin = min(h.terrain.Z.min() for h in hs)
    zmax = max(h.terrain.Z.max() for h in hs)
    nt = [len(h.threats) for h in hs]
    rows.append(f"| H01–H40 | {zmin:.0f}–{zmax:.0f} | {min(nt)}–{max(nt)} | random, ≈800 m apart | held-out test |")
    c = get_scenario("C1")
    rows.append(f"| C1 [5] | {c.terrain.Z.min():.0f}–{c.terrain.Z.max():.0f} | {len(c.threats)} | "
                f"({c.start[0]:.0f}, {c.start[1]:.0f}) → ({c.goal[0]:.0f}, {c.goal[1]:.0f}) | public test |")
    return "\n".join(rows)


def design(R, methods=None, short=True, ref=REF, star=("S5", "S6"), show_feas=True):
    methods = methods or R["methods"]
    name = (lambda m: SHORT.get(m, lab(m))) if short else lab
    rows = ["| Inst. | " + " | ".join(name(m) for m in methods) + " |",
            "|:--" + "|:-:" * len(methods) + "|"]
    for inst in R["instances"]:
        r = R["per_instance"][inst]
        meds = [r["median"][m] for m in methods]
        b = int(np.argmin(meds))
        cells = []
        for k, m in enumerate(methods):
            s = "" if m == ref else " " + SIGN[r["cmp"][m]["sign"]]
            txt = f"{meds[k]:.0f}{s}"
            if show_feas and r["feas"][m] < 1.0:
                txt += f"^†^"
            cells.append(f"**{txt}**" if k == b else txt)
        tag = "*" if inst.split("_")[0] in star else ""
        rows.append(f"| {ilab(inst)}{tag} | " + " | ".join(cells) + " |")
    rows.append("| +/≈/− | " + " | ".join("—" if m == ref else "/".join(map(str, R["wtl"][m])) for m in methods) + " |")
    rk = dict(zip(R["methods"], R["ranks"]))
    rows.append("| Rank | " + " | ".join(f"{rk[m]:.2f}" for m in methods) + " |")
    return "\n".join(rows)


def design_full(R, n, methods=None, ref=REF):
    """Supplement: mean ± sd, median, feasible %, A12 of the reference."""
    methods = methods or R["methods"]
    rows = ["| Inst. | Method | Median | Mean ± SD | Feasible | A12 | Δ | Sign (Holm) |",
            "|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for inst in R["instances"]:
        if not inst.endswith(f"n{n}"):
            continue
        r = R["per_instance"][inst]
        for m in methods:
            c = r["cmp"].get(m)
            a12 = "—" if c is None else f"{c['a12']:.2f}"
            g = "—" if c is None else pct(c["gap"])
            s = "—" if c is None else SIGN[c["sign"]]
            rows.append(f"| {ilab(inst)} | {lab(m)} | {r['median'][m]:.1f} | {r['mean'][m]:.1f} ± {r['sd'][m]:.1f} | "
                        f"{100 * r['feas'][m]:.0f}% | {a12} | {g} | {s} |")
    return "\n".join(rows)


def heldout(H, Ht=None, ref=REF):
    rows = ["| Method | Win~10~ | Δ̃~10~ | p~H,10~ | Win~20~ | Δ̃~20~ | p~H,20~ |",
            "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]

    def block(HH, title=""):
        out = []
        for k, o in enumerate([m for m in HH["methods"] if m != HH.get("ref", ref) and m != ref and m != "DPF-PSO-T"]):
            cells = []
            for n in HH["n"]:
                t = HH["tests"][f"{o}|{n}"]
                s = "" if t["sig"] == "=" else ("^+^" if t["sig"] == "+" else "^−^")
                cells += [f"{100 * t['win']:.0f}%", pct(t["gap_median"]), ptxt(t["p_holm"]) + s]
            name = f"*{title}* {lab(o)}" if title and k == 0 else lab(o)
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        return out

    rows += block(H)
    if Ht:
        rows += block(Ht, "Tuned:")
    return "\n".join(rows)


def heldout_more(blocks):
    """Further paired comparisons on the held-out scenarios.  blocks: list of
    (title, H, {method: row label}); columns as in heldout()."""
    rows = ["| Compared with | Win~10~ | Δ̃~10~ | p~H,10~ | Win~20~ | Δ̃~20~ | p~H,20~ |",
            "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for title, H, labels in blocks:
        for k, (o, lb) in enumerate(labels.items()):
            cells = []
            for n in H["n"]:
                t = H["tests"][f"{o}|{n}"]
                s = "" if t["sig"] == "=" else ("^+^" if t["sig"] == "+" else "^−^")
                cells += [f"{100 * t['win']:.0f}%", pct(t["gap_median"]), ptxt(t["p_holm"]) + s]
            rows.append(f"| {f'*{title}* ' if k == 0 else ''}{lb} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def sources(ablation, plugin, order_abl, hosts):
    rows = ["| Configuration | Rank | Δ (mean) | +/≈/− |", "|:--|:-:|:-:|:-:|"]
    rows.append("| *Ablation of DPF-PSO (reference: full DPF-PSO)* | | | |")
    for v in order_abl:
        a = ablation[v]
        if v == REF:
            rows.append(f"| DPF-PSO (full) | {a['rank']:.2f} | — | — |")
        else:
            rows.append(f"| {lab(v)} | {a['rank']:.2f} | {pct(a['gap_mean'])} | {'/'.join(map(str, a['wtl']))} |")
    rows.append("| *DPF as a plug-in (reference: host alone)* | | | |")
    for h in hosts:
        p = plugin[h]
        rows.append(f"| {lab(h)} + DPF | — | {pct(p['gap_mean'])} | {'/'.join(map(str, p['wtl']))} |")
    return "\n".join(rows)


def timing_table(T, mult, eq, methods):
    rows = ["| Method | CPU s (n = 10) | CPU s (n = 20) | Budget × (10 / 20) | +/≈/− at equal time |",
            "|:--|:-:|:-:|:-:|:-:|"]
    for m in methods:
        c10, c20 = T["10"][m], T["20"][m]
        cells = [f"{c10['cpu']:.2f} ± {100 * c10['cpu_rel_sd']:.0f}%", f"{c20['cpu']:.2f} ± {100 * c20['cpu_rel_sd']:.0f}%"]
        if m == REF:
            cells += ["1.0 / 1.0", "—"]
        else:
            w = eq["wtl"].get(m) if eq else None
            cells += [f"{mult['10'][m]:.1f} / {mult['20'][m]:.1f}", "/".join(map(str, w)) if w else "n/a"]
        rows.append(f"| {lab(m)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


ROB_LABEL = {"main": "S1–S6", "main_tuned": "Tuned",        # tuned rows refer to the row above
             "public": "C1 [5]", "public_tuned": "Tuned",
             "n30": "n = 30", "budget5": "5× budget", "b2": "b~2~ = 10", "b4": "b~4~ = 10",
             "eqtime": "Equal time"}
BLACK = ["PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE", "CMAES"]


def robustness(S, tags, close="IDP"):
    """One row per experiment: instances, lowest median, mean rank of the
    reference and of the best other method, and +/≈/− against IDP and
    against the best-ranked black-box method of that experiment."""
    # dash counts set the relative column widths in pandoc
    rows = ["| Experiment | Inst. | Lowest median | Rank: DPF-PSO / best other | vs. IDP | vs. best black-box |",
            "|:--------------|:---:|:------:|:----------------:|:------:|:-----------:|"]
    for tag in tags:
        if tag not in S:
            continue
        R = S[tag]
        tuned = tag.endswith("tuned")
        ref = "DPF-PSO-T" if tuned else REF
        cl = close + "-T" if tuned else close
        rk = dict(zip(R["methods"], R["ranks"]))
        others = [m for m in R["methods"] if m != ref]
        bo = min(others, key=rk.get)
        bb = [m for m in others if m.replace("-T", "") in BLACK]
        bbest = min(bb, key=rk.get)
        vs = "/".join(map(str, R["wtl"][cl])) if cl in R["wtl"] else "—"
        rows.append(f"| {ROB_LABEL.get(tag, tag)} | {len(R['instances'])} | {R['best_count']} | "
                    f"{rk[ref]:.2f} / {rk[bo]:.2f} ({lab(bo)}) | {vs} | "
                    f"{'/'.join(map(str, R['wtl'][bbest]))} ({lab(bbest)}) |")
    return "\n".join(rows)


def tuning_table(tun, spaces):
    rows = ["| Algorithm | Selected configuration | Mean rank (selected / default) |", "|:--|:--|:-:|"]
    for a, t in tun.items():
        cfg = ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}" for k, v in t["selected"].items())
        rows.append(f"| {lab(a)} | {cfg} | {t['selected_rank']:.2f} / {t['default_rank']:.2f} |")
    return "\n".join(rows)
