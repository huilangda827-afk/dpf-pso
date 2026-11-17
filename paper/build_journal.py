"""Build the journal version (fourth revision) from the analysis results.

    python src/analysis2.py                 # statistics, tables, figures (incl. H41-H80, GridDP, time curves)
    python paper/build_journal.py           # journal.docx / journal.pdf (single column)
    python paper/build_journal.py --draft   # missing values are marked instead of failing
    python paper/build_journal.py --reference paper/mdpi_template/reference_mdpi.docx

Everything of build2.py is reused: the values dictionary v, the generated
prose of the third revision and the ASSERT guards.  This file adds the
values, tables and sentences of the journal-only experiments (design log,
sections 15 and 16): the pre-registered confirmatory test on H41-H80, the
one-shot lattice-DP baseline, the composition of the final cost and the
CPU-time convergence.  Nothing is typed in by hand.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np

import build2 as B
import journal_tables as JT
import tables2 as T

T.LAB["GridDP"] = "lattice DP"
T.LAB["MS-Powell"] = "multi-start Powell"
T.LAB["DE-sph"] = "DE (spherical)"
T.LAB["CMAES-sph"] = "CMA-ES (spherical)"
T.LAB["DPF-PSO_noPG"] = "w/o periodic global fusion"
if "b4" not in B.ROBUST:
    B.ROBUST.insert(B.ROBUST.index("b2") + 1, "b4")        # design log 18d
TITLE = ("Dynamic-Programming Path Fusion: An Optimal Recombination Operator for Swarm Optimizers "
         "in 3-D UAV Path Planning")
TEMPLATE = "paper_journal_template.md"
NS = ("10", "20")
REF = B.REF
SLOT_BB = [m for m in B.BLACK if m != "SPSO"]


def fam(v, prefix, R):
    """Per-test values of a held-out family (same keys as build2.compute)."""
    for k, tt in R["tests"].items():
        o, n = k.split("|")
        key = o.replace("-", "").replace("_", "")
        v[f"{prefix}_gap_{key}{n}"] = T.pct(tt["gap_median"])
        v[f"{prefix}_gapu_{key}{n}"] = T.pct(abs(tt["gap_median"]), signed=False)
        v[f"{prefix}_win_{key}{n}"] = f"{100 * tt['win']:.0f}"
        v[f"{prefix}_p_{key}{n}"] = B.fmt_p(tt["p_holm"])
        v[f"_{prefix}_sig_{key}{n}"] = tt["sig"]
        v[f"_{prefix}_gap_{key}{n}"] = tt["gap_median"]
    v[f"{prefix}_ntests"] = R.get("family_size", len(R["tests"]))


def _g(H, o, n):
    return T.pct(H["tests"][f"{o}|{n}"]["gap_median"])


def tuned_pair_sentence(H, model):
    """Result of tuned SPSO vs tuned DPF-PSO for both n under one cost model."""
    words = {"+": "tuned DPF-PSO was significantly better", "=": "the two were not significantly different",
             "-": "tuned SPSO was significantly better"}
    sig = [H["tests"][f"SPSO-T|{n}"]["sig"] for n in NS]
    if sig[0] == sig[1]:
        both = "either" if sig[0] == "=" else "both"
        return f"Under the {model}, {words[sig[0]]} for {both} $n$ (Δ̃ = {_g(H, 'SPSO-T', '10')} and {_g(H, 'SPSO-T', '20')})."
    parts = [f"{words[s]} for $n={n}$ (Δ̃ = {_g(H, 'SPSO-T', n)})" for n, s in zip(NS, sig)]
    return f"Under the {model}, {parts[0]} and {parts[1]}."


def lattice_sizes(n, n_h=3, n_tau=3, budget_per_wp=3000):
    """Lattice of GridDP for n waypoints (same rule as algorithms.grid_dp)."""
    nseg_max = budget_per_wp * n * (n + 1)
    a, b = n - 1, 2
    K = int((-b + math.sqrt(b * b + 4 * a * nseg_max)) / (2 * a))
    n_u = max(2, K // (n_h * n_tau))
    K_eff = n_u * n_h * n_tau
    fes = math.ceil((2 * K_eff + (n - 1) * K_eff ** 2) / (n + 1))
    return n_u, K_eff, fes / (budget_per_wp * n)


def prose_journal(v, S):
    """Data-dependent sentences of the journal-only experiments.  Every
    branch states only what the tests show; the wording for the confirmatory
    test follows the criterion written in design log section 15."""
    # ------------------------------------------------- H41-H80 (design log 15)
    if "heldout2_verdict" in S:
        V = S["heldout2_verdict"]
        Ht2, Ht2b, Hd2, Hd2b = S["heldout2_tuned"], S["heldout2_band_tuned"], S["heldout2"], S["heldout2_band"]
        fam(v, "h2t", Ht2)
        fam(v, "h2bt", Ht2b)
        fam(v, "h2d", Hd2)
        fam(v, "h2bd", Hd2b)
        v["h2_orig_text"] = tuned_pair_sentence(Ht2, "original cost model")
        v["h2_band_text"] = tuned_pair_sentence(Ht2b, "band model")
        v["h2_defaults_text"] = ("With default configurations, DPF-PSO was "
                                 + B.join_clauses(B.compare_clauses(Hd2b, ["SPSO", "DE", "IDP"]))
                                 + " under the band model, and "
                                 + B.join_clauses(B.compare_clauses(Hd2, ["SPSO", "DE", "IDP"]))
                                 + " under the original model.")
        v["_h2_confirmed"] = bool(V["band_confirmed"])
        if V["band_confirmed"]:
            v["h2_verdict"] = ("Both pre-registered conditions were met: the reversal under the band model and the "
                               "direction of the original-model comparison were reproduced on scenarios that played "
                               "no part in finding the effect, so we regard the band finding as confirmed.")
            v["band_status"] = "confirmed on 40 further held-out scenarios"
            v["band_status_short"] = "confirmed"
            v["abs_band"] = ("Tuned spherical-vector PSO was better for 10 waypoints; its paths left the altitude "
                             "band between waypoints, where the model does not check it, and with the band enforced "
                             "along the path tuned DPF-PSO was significantly better for both numbers of waypoints, "
                             "a result confirmed by a pre-registered test on 40 further scenarios.")
            v["contrib_band_word"] = "shows"
            v["conc_band_j"] = ("The advantage of tuned SPSO for 10 waypoints relied on leaving the altitude band "
                                "between waypoints and was reversed when the band was enforced along the path, both "
                                "on the scenarios where the effect was found and, by a pre-registered test, on 40 "
                                "further scenarios.")
            v["disc_band_j"] = ("The altitude band of [5] applies at the waypoints only, and encodings that can merge "
                                "waypoints exploit this: the advantage of tuned SPSO for $n=10$ went with long "
                                "segments outside the band, turned into a significant disadvantage when the band "
                                "was enforced along the path, and did so again on 40 further scenarios.")
        else:
            why = []
            if not V["band_better_both_n"]:
                why.append("tuned DPF-PSO was not significantly better than tuned SPSO under the band model for "
                           "both $n$")
            if not V["same_direction_as_H01_H40"]:
                why.append("the original-model comparison did not have the same direction as on H01–H40")
            v["h2_verdict"] = ("The pre-registered criterion was not met: " + " and ".join(why)
                               + ". We therefore report the band finding as exploratory.")
            v["band_status"] = "not confirmed on 40 further held-out scenarios"
            v["band_status_short"] = "not confirmed"
            v["abs_band"] = ("Tuned spherical-vector PSO was better for 10 waypoints; its paths left the altitude "
                             "band between waypoints, where the model does not check it. Enforcing the band along "
                             "the path reversed this on the scenarios where it was found, but a pre-registered test "
                             "on 40 further scenarios did not confirm the reversal, so this explanation remains "
                             "exploratory.")
            v["contrib_band_word"] = "suggests"
            v["conc_band_j"] = ("The advantage of tuned SPSO for 10 waypoints went with leaving the altitude band "
                                "between waypoints; its reversal under a band enforced along the path did not meet "
                                "the pre-registered criterion on 40 further scenarios and remains exploratory.")
            v["disc_band_j"] = ("The altitude band of [5] applies at the waypoints only, and encodings that can merge "
                                "waypoints exploit this: the advantage of tuned SPSO for $n=10$ went with long "
                                "segments outside the band. Its reversal under a band enforced along the path was "
                                "found post hoc and was not confirmed by the pre-registered test, so we do not "
                                "claim it.")
        AG2 = S.get("agl_heldout2", {}).get("10", {})
        if "SPSO-T" in AG2 and "DPF-PSO-T" in AG2:
            v["agl2_out_SPSOT10"] = T.pct(AG2["SPSO-T"]["outside_mean"], signed=False)
            v["agl2_out_DPFPSOT10"] = T.pct(AG2["DPF-PSO-T"]["outside_mean"], signed=False)
            v["_agl2_ratio10"] = AG2["SPSO-T"]["outside_mean"] / max(1e-9, AG2["DPF-PSO-T"]["outside_mean"])
        fails = []
        for R, ms in ((Hd2b, ["SPSO", "DE", "IDP", REF]), (Ht2b, ["SPSO-T", "DPF-PSO-T"])):
            for m in ms:
                fails += [(m, n, R[f"n{n}"]["n_infeasible"][m], R[f"n{n}"]["n_runs"][m]) for n in NS
                          if R[f"n{n}"]["n_infeasible"][m] > 0]
        v["h2_feas_text"] = ("All runs under the band model ended with a feasible path." if not fails else
                             "All runs under the band model ended with a feasible path except "
                             + B.and_list(f"{k} of the {N} runs of {B._name(m)} for $n={n}$" for m, n, k, N in fails) + ".")
    # ------------------------------------------------- lattice DP (design log 16)
    for key, prefix, setname in (("heldout_grid", "hg", "H01–H40"), ("heldout2_grid", "hg2", "H41–H80")):
        if key in S:
            Hg = S[key]
            fam(v, prefix, Hg)
            v[f"{prefix}_text"] = ("DPF-PSO was " + B.join_clauses(B.compare_clauses(Hg, ["GridDP"]))
                                   + f" on {setname}")
            v[f"_{prefix}_sig"] = tuple(Hg["tests"][f"GridDP|{n}"]["sig"] for n in NS)
    if "heldout_grid" in S:
        parts = [v["hg_text"]] + ([v["hg2_text"].replace("DPF-PSO was ", "")] if "hg2_text" in v else [])
        v["grid_text"] = "; and ".join(parts) + "."
        for n in (10, 20):
            n_u, K, share = lattice_sizes(n)
            v[f"grid_nu{n}"], v[f"grid_K{n}"], v[f"grid_fe{n}"] = n_u, K, f"{100 * share:.0f}"
        CG = S.get("components_grid", {})
        if CG:
            v["grid_turn10"] = f"{CG['10']['GridDP']['max_turn_median']:.0f}"
            v["grid_turn20"] = f"{CG['20']['GridDP']['max_turn_median']:.0f}"
        sg = S.get("safety_grid", {}).get("GridDP")
        if sg:
            v["grid_feas"] = f"{sg['feasible']}/{sg['n']}"
            v["_grid_all_feasible"] = sg["feasible"] == sg["n"]
    # ------------------------------------------------- second round (design log 18)
    H = S.get("heldout_ablation")
    if H and "DPF-PSO_noPG" in H["methods"] and "DPF-PSO_noA" in H["methods"]:
        s1, s2 = B._sig(H, "DPF-PSO_noPG"), B._sig(H, "DPF-PSO_noA")
        txt = (f"Two further components were tested on the held-out scenarios: without periodic global fusion "
               f"(keeping the initial one), DPF-PSO's variant was {B.vs_variant(H, 'DPF-PSO_noPG')}, and with a "
               f"fixed step size it was {B.vs_variant(H, 'DPF-PSO_noA')}")
        if s1 == ("=", "=") and s2 == ("=", "="):
            txt += (". Neither component contributes significantly, which agrees with the tuned configuration "
                    "(Section 5.3) that dropped both; a simpler DPF-PSO with a single global fusion and a fixed "
                    "step is therefore an adequate recommendation.")
        elif "-" in s1 + s2:
            txt += ("; removing a component helped in at least one case, so the default schedule is not the best "
                    "choice for every $n$ (Table 3).")
        else:
            keep = [nm for s, nm in ((s1, "periodic global fusion"), (s2, "step adaptation")) if "+" in s]
            txt += f"; {B.and_list(keep)} therefore contribute(s) significantly for at least one $n$."
        v["hab2_text"] = txt
        v["_hab2_sigs"] = (s1, s2)
    else:
        v["hab2_text"] = ""
    He = S.get("heldout_eqtime")
    if He and len(He["methods"]) > 4:
        others = [m for m in He["methods"] if m != REF]
        v["heq_all_text"] = ("Against the full set of baselines at equal CPU time (Table 3), DPF-PSO was "
                             + B.join_clauses(B.compare_clauses(He, others)) + ".")
        v["heq_mult_lo"], v["heq_mult_hi"] = v["eq_lo"], v["eq_hi"]
        v["heq_reused"] = B.and_list(f"{T.lab(k.split('|')[0])} ($n={k.split('|')[1]}$)" for k in He.get("reused_from_heldout", []))
        v["_heq_all_plus"] = [m for m in others if B._sig(He, m) == ("+", "+")]
        v["heq_all_nplus"] = len(v["_heq_all_plus"])
        v["heq_all_nbase"] = len(others)
    else:
        v["heq_all_text"] = ""
        v["heq_reused"] = "none"
    Hp = S.get("heldout_plugin")
    if Hp and not all(f"{h}|{n}" in Hp["tests"] for h in Hp["hosts"] for n in NS):
        Hp = None                                     # runs still in progress
    if Hp:
        hosts = Hp["hosts"]
        sigs = {h: tuple(Hp["tests"][f"{h}|{n}"]["sig"] for n in NS) for h in hosts}
        gaps = {n: [Hp["tests"][f"{h}|{n}"]["gap_median"] for h in hosts] for n in NS}
        v["plug_ho_lo10"], v["plug_ho_hi10"] = T.pct(min(gaps["10"]), signed=False), T.pct(max(gaps["10"]), signed=False)
        v["plug_ho_lo20"], v["plug_ho_hi20"] = T.pct(min(gaps["20"]), signed=False), T.pct(max(gaps["20"]), signed=False)
        allplus = all(s == ("+", "+") for s in sigs.values())
        if allplus:
            v["plug_ho_text"] = (f"On the held-out scenarios (Table 3), DPF lowered the median cost of all "
                                 f"{len(hosts)} hosts significantly for both $n$, by {v['plug_ho_lo10']}–{v['plug_ho_hi10']} "
                                 f"($n=10$) and {v['plug_ho_lo20']}–{v['plug_ho_hi20']} ($n=20$).")
        else:
            exc = [f"{T.lab(h)} ($n={n}$)" for h in hosts for n, s in zip(NS, sigs[h]) if s != "+"]
            v["plug_ho_text"] = (f"On the held-out scenarios (Table 3), DPF lowered the median cost of the hosts "
                                 f"significantly in {sum(s == '+' for x in sigs.values() for s in x)} of "
                                 f"{2 * len(hosts)} (host, $n$) pairs; the exceptions were {B.and_list(exc)}.")
        v["_plug_ho_allplus"] = allplus
        v["_plug_ho_min_gap"] = min(min(gaps["10"]), min(gaps["20"]))
    else:
        v["plug_ho_text"] = ""
    Hm = S.get("heldout_mspowell")
    if Hm:
        v["msp_text"] = ("Against the multi-start local optimizer, which represents gradient-free trajectory "
                         "optimization of the same cost from random starts, DPF-PSO was "
                         + B.join_clauses(B.compare_clauses(Hm, ["MS-Powell"])) + " (Table 4).")
        v["_msp_sig"] = B._sig(Hm, "MS-Powell")
        v["msp_gap10"], v["msp_gap20"] = (T.pct(Hm["tests"][f"MS-Powell|{n}"]["gap_median"]) for n in NS)
    else:
        v["msp_text"] = ""
    Hs, Hsb, Hvs = S.get("heldout_sph"), S.get("heldout_band_sph"), S.get("sph_vs_slot")
    if Hs and Hvs:
        parts = []
        for m, Hv in Hvs.items():
            o = m.replace("-sph", "")
            g = [Hv["tests"][f"{o}|{n}"]["gap_median"] for n in NS]
            s = [Hv["tests"][f"{o}|{n}"]["sig"] for n in NS]
            word = {"+": "better than", "=": "not significantly different from", "-": "worse than"}
            parts.append(f"{T.lab(o)} on the spherical encoding was {word[s[0]]} its slot-encoded version for $n=10$ "
                         f"(Δ̃ = {T.pct(g[0])}) and {word[s[1]]} it for $n=20$ (Δ̃ = {T.pct(g[1])})")
        sph_others = [m for m in Hs["methods"] if m != REF]
        txt = ("Encoding sensitivity (design log 18f). " + "; ".join(parts)
               + ". Against DPF-PSO under the original model, DPF-PSO was "
               + B.join_clauses(B.compare_clauses(Hs, sph_others)) + ".")
        if Hsb:
            sphb_others = [m for m in Hsb["methods"] if m != REF]
            txt += (" Under the band model, DPF-PSO was " + B.join_clauses(B.compare_clauses(Hsb, sphb_others))
                    + ", which is the pattern expected if the spherical encoding profits from leaving the band "
                      "between merged waypoints.")
            v["_sph_band_sigs"] = {m: B._sig(Hsb, m) for m in sphb_others}
        v["sph_text"] = txt
        v["_sph_sigs"] = {m: B._sig(Hs, m) for m in sph_others}
    else:
        v["sph_text"] = ""
    if "b4" in S and "_b4_bb_gaps" in v:
        g4, g1 = float(np.median(v["_b4_bb_gaps"])), float(np.median(v["_bb_gaps"]))
        v["b4_bb_gap_med"], v["main_bb_gap_med2"] = T.pct(g4, signed=False), T.pct(g1, signed=False)
        v["_b4_vs_main"] = (g4, g1)
        v["b4_more_or_less"] = "larger" if g4 > g1 else "smaller"
    C2 = S.get("components_heldout")
    if C2 and all("turn_le45" in C2[n][m] for n in NS for m in C2[n]):
        for n in NS:
            ms = [m for m in B.MAIN if m in C2[n]]
            t45 = {m: C2[n][m]["turn_le45"] for m in ms}
            c30 = {m: C2[n][m]["dclimb_le30"] for m in ms}
            v[f"kin_t45_dpf{n}"] = T.pct(t45[REF], 0, signed=False)
            v[f"kin_t45_lo{n}"] = T.pct(min(x for m, x in t45.items() if m != REF), 0, signed=False)
            v[f"kin_t45_hi{n}"] = T.pct(max(x for m, x in t45.items() if m != REF), 0, signed=False)
            v[f"kin_c30_dpf{n}"] = T.pct(c30[REF], 0, signed=False)
            v[f"kin_c30_lo{n}"] = T.pct(min(x for m, x in c30.items() if m != REF), 0, signed=False)
            v[f"kin_c30_hi{n}"] = T.pct(max(x for m, x in c30.items() if m != REF), 0, signed=False)
            v[f"_kin_t45_dpf{n}"], v[f"_kin_t45_best_other{n}"] = t45[REF], max(x for m, x in t45.items() if m != REF)
    # ------------------------------------------------- cost composition
    C = S.get("components_heldout")
    if C:
        for n in NS:
            ms = [m for m in B.MAIN if m in C[n]]
            for comp, nm in (("F1", "len"), ("F2", "thr"), ("F3", "alt"), ("F4", "smo")):
                xs = [C[n][m]["share_median"][comp] for m in ms]
                v[f"comp_{nm}_lo{n}"] = T.pct(min(xs), 1 if comp in ("F1", "F4") else 2, signed=False)
                v[f"comp_{nm}_hi{n}"] = T.pct(max(xs), 1 if comp in ("F1", "F4") else 2, signed=False)
                v[f"_comp_{nm}_max{n}"] = max(xs)
            alt = {m: C[n][m]["share_median"]["F3"] for m in ms}
            v[f"comp_alt_SPSO{n}"] = T.pct(alt.get("SPSO", float("nan")), 1, signed=False)
            v[f"_comp_alt_max_nospso{n}"] = max(x for m, x in alt.items() if m != "SPSO")
            v[f"comp_alt_hi_nospso{n}"] = T.pct(v[f"_comp_alt_max_nospso{n}"], 1, signed=False)
            v[f"comp_turn_dpf{n}"] = f"{C[n][REF]['max_turn_median']:.0f}"
            v[f"comp_climb_dpf{n}"] = f"{C[n][REF]['max_dclimb_median']:.0f}"
            tm = {m: C[n][m]["max_turn_median"] for m in ms}
            v[f"comp_turn_max_method{n}"] = T.lab(max(tm, key=tm.get))
            v[f"comp_turn_max{n}"] = f"{max(tm.values()):.0f}"
            v[f"comp_turn_min{n}"] = f"{min(tm.values()):.0f}"
    # ------------------------------------------------- CPU-time convergence
    TC = S.get("timecurves")
    if TC:
        sents = []
        for n in NS:
            fin, never = {}, []
            for m in SLOT_BB + ["SPSO", "IDP"]:
                r = TC.get(f"{m}|{n}")
                if not r:
                    continue
                ok = np.isfinite(r["ratio_median"])
                v[f"tc_ratio_{m}{n}"] = f"{r['ratio_median']:.2f}" if ok else "—"
                v[f"tc_reach_{m}{n}"] = f"{100 * r['reach_share']:.0f}"
                if ok:
                    fin[m] = r["ratio_median"]
                else:
                    never.append(m)
            bb = {m: x for m, x in fin.items() if m in SLOT_BB}
            if bb:
                v[f"tc_bb_lo{n}"], v[f"tc_bb_hi{n}"] = f"{100 * min(bb.values()):.0f}", f"{100 * max(bb.values()):.0f}"
                v[f"_tc_bb_max{n}"] = max(bb.values())
                s = f"after {v[f'tc_bb_lo{n}']}–{v[f'tc_bb_hi{n}']}% of their CPU time for $n={n}$"
                nv = [m for m in never if m in SLOT_BB]
                if nv:
                    s += f" (not within its own run time, on the median scenario, for {B.and_list(T.lab(m) for m in nv)})"
                sents.append(s)
            v[f"tc_never{n}"] = B.and_list(T.lab(m) for m in never) if never else "none"
            v[f"_tc_never{n}"] = never
            v[f"tc_n_scen{n}"] = next(iter(TC.values()))["n_scen"]
        v["tc_text"] = ("Measured in CPU time, DPF-PSO reached the final median cost of the black-box methods with "
                        "the slot encoding " + " and ".join(sents) + ".") if sents else ""
        v["tc_n_methods"] = len({k.split("|")[0] for k in TC})


def compute():
    v, t, S = B.compute()
    v["TITLE"] = TITLE
    prose_journal(v, S)
    # sentences generated for the conference version refer to its numbering
    JOURNAL_NAMES = [("Table III", "Table 3"), ("Table II", "Table 2"), ("Table IV", "Table 6"),
                     ("supplement Table", "Supplementary Material Table"),
                     ("supplement S", "Supplementary Material S"), ("Fig. 3", "Figure 3"), ("Fig. 2", "Figure 2")]
    for k, x in list(v.items()):
        if isinstance(x, str) and not k.startswith("_"):
            for a, b in JOURNAL_NAMES:
                x = x.replace(a, b)
            v[k] = x
    t["TABLE_SCEN"] = JT.scenarios()
    t["TABLE_HELDOUT_MORE"] = JT.heldout_more(S)
    t["TABLE_HELDOUT2"] = JT.heldout2(S)
    t["TABLE_GRID"] = JT.grid(S)
    t["TABLE_COMPONENTS"] = JT.components(S, B.MAIN)
    t["TABLE_TIMECURVES"] = JT.timecurves(S, ["SPSO", "DE", "LSHADE", "CMAES", "IDP"])
    return v, t, S


GUTTER = 2608          # twips: left indent of the MDPI text column (line-number gutter)
TEXT_FULL = 10433      # twips: A4 width minus the 13 mm margins of the template
TEXT_COL = TEXT_FULL - GUTTER


def mdpi_tables(d, wide_cols=8):
    """Align tables and equation tables with the MDPI text column; tables
    with many columns use the full width between the margins."""
    import docx_tools as D
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for tb in D._all_tables(d):
        ncols = max(len(r.cells) for r in tb.rows)
        wide = ncols >= wide_cols
        tblPr = tb._tbl.tblPr
        for tag in ("w:tblInd", "w:tblW", "w:jc"):
            for old in tblPr.findall(qn(tag)):
                tblPr.remove(old)
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "left")
        ind = OxmlElement("w:tblInd")
        ind.set(qn("w:w"), "0" if wide else str(GUTTER))
        ind.set(qn("w:type"), "dxa")
        w = OxmlElement("w:tblW")
        w.set(qn("w:w"), str(TEXT_FULL if wide else TEXT_COL))
        w.set(qn("w:type"), "dxa")
        for el in (jc, ind, w):
            tblPr.append(el)
        D._sort_tblpr(tblPr)


def postprocess_single(src, dst):
    """Single-column journal layout: tables, equation numbers, centred figures,
    everything aligned with the MDPI text column."""
    import docx
    import docx_tools as D
    from docx.shared import Twips
    d = docx.Document(src)
    D.style_tables(d, font_pt=8.5)
    n_eq = D.number_equations(d, width=Twips(TEXT_COL))
    D.center_figures(d)
    mdpi_tables(d)
    # figures: centred within the text column (left indent = gutter + centre alignment)
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    for pe in d.element.body.iter(qn("w:p")):
        if pe.findall(".//" + qn("w:drawing")):
            Paragraph(pe, d._body).paragraph_format.left_indent = Twips(GUTTER)
    d.save(dst)
    return n_eq


def supplement_source():
    """Journal supplement = conference supplement with a journal header, the
    figure widths of the MDPI text column and the sections S12-S14 appended."""
    src = open(os.path.join(HERE, "supplement_template.md"), encoding="utf-8").read()
    src = src.replace("[[COLS2]]\n", "")
    src = src.replace('::: {custom-style="Author"}\nAnonymous Authors\n:::', '::: {custom-style="Author"}\n[Author Name]\n:::')
    src = re.sub(r"\{width=1[5-9]\dmm\}", "{width=135mm}", src)
    src = src.replace("`paper/build2.py`", "`paper/build_journal.py`")
    return src + open(os.path.join(HERE, "supplement_journal_extra.md"), encoding="utf-8").read()


def build(template, name, v, t, strict, reference, cite_map=None):
    import renumber_refs as R
    src = open(os.path.join(HERE, template), encoding="utf-8").read() if template.endswith(".md") else template
    md = B.fill(src, v, t, strict)
    if cite_map is None:
        cite_map = R.first_appearance_map(md)
        md = R.renumber(md)                 # citations numbered by first appearance
    else:
        md = R.apply_map(md, cite_map)      # supplement: same numbers as the main text
    out_md = os.path.join(HERE, f"{name}.md")
    open(out_md, "w", encoding="utf-8").write(md)
    raw = os.path.join(HERE, f"{name}_raw.docx")
    subprocess.run(["pandoc", out_md, "-o", raw, "--reference-doc", reference, "--resource-path", HERE],
                   check=True, cwd=HERE)
    postprocess_single(raw, os.path.join(HERE, f"{name}.docx"))
    os.remove(raw)
    import docx_tools as D
    pages = D.export_pdf(os.path.join(HERE, f"{name}.docx"), os.path.join(HERE, f"{name}.pdf"))
    import fitz
    doc = fitz.open(os.path.join(HERE, f"{name}.pdf"))
    for k, page in enumerate(doc):
        page.get_pixmap(dpi=100).save(os.path.join(HERE, f"preview_{name}_p{k + 1}.png"))
    words = len(re.findall(r"\w+", re.sub(r"\$.*?\$", "", md)))
    print(f"{name}.docx / {name}.pdf: {pages} pages, about {words:,} words in the markdown source")
    return pages, cite_map


def main():
    strict = "--draft" not in sys.argv
    ref = os.path.join(HERE, "reference_mdpi.docx")          # made by make_reference_mdpi.py
    if not os.path.exists(ref):
        ref = os.path.join(HERE, "reference.docx")
    if "--reference" in sys.argv:
        ref = os.path.abspath(sys.argv[sys.argv.index("--reference") + 1])
    v, t, S = compute()
    import supplement_tables as ST
    t.update(ST.tables(S))
    t.update(JT.supplement_tables(S))
    if strict:
        B.check_prose(v, S, TEMPLATE)
        B.check_prose(v, S, "supplement_template.md")
    json.dump({k: x for k, x in v.items() if not k.startswith("_")},
              open(os.path.join(HERE, "values_journal.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    only = [a for a in sys.argv[1:] if not a.startswith("--") and not a.endswith(".docx")]
    _, cite_map = build(TEMPLATE, "journal", v, t, strict, ref)
    if not only or "supplement" in only:
        build(supplement_source(), "supplement_journal", v, t, strict, ref, cite_map=cite_map)


if __name__ == "__main__":
    main()
