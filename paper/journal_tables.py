"""Markdown tables that exist only in the journal version (fourth revision):
confirmatory test on H41-H80, lattice-DP baseline, cost composition and
CPU-time convergence.  Generated from results/analysis/summary.json."""
from __future__ import annotations

import numpy as np

import tables2 as T

REF = "DPF-PSO"


def scenarios():
    """Scenario table of the conference version plus the row for H41-H80."""
    from env import get_scenario
    rows = T.scenarios().split("\n")
    hs = [get_scenario(f"H{k:02d}") for k in range(41, 81)]
    zmin = min(h.terrain.Z.min() for h in hs)
    zmax = max(h.terrain.Z.max() for h in hs)
    nt = [len(h.threats) for h in hs]
    row = f"| H41–H80 | {zmin:.0f}–{zmax:.0f} | {min(nt)}–{max(nt)} | random, ≈800 m apart | pre-registered test |"
    i = next(k for k, r in enumerate(rows) if r.startswith("| H01–H40"))
    rows.insert(i + 1, row)
    return "\n".join(rows)


def heldout2(S):
    """Pre-registered families on H41-H80 (design log 15) as one table:
    tuned pair under both cost models (family T), defaults under the band
    model (family D) and under the original model (family R)."""
    blocks = []
    if "heldout2_tuned" in S:
        blocks.append(("Original model, tuned:", S["heldout2_tuned"], {"SPSO-T": "SPSO"}))
    if "heldout2_band_tuned" in S:
        blocks.append(("Band model, tuned:", S["heldout2_band_tuned"], {"SPSO-T": "SPSO"}))
    if "heldout2_band" in S:
        blocks.append(("Band model:", S["heldout2_band"], {"SPSO": "SPSO", "DE": "DE", "IDP": "IDP"}))
    if "heldout2" in S:
        blocks.append(("Original model:", S["heldout2"], {"SPSO": "SPSO", "DE": "DE", "IDP": "IDP"}))
    return T.heldout_more(blocks) if blocks else ""


def grid(S):
    """DPF-PSO against the one-shot lattice DP (design log 16, family G) and
    the multi-start local optimizer (design log 18e, own family)."""
    blocks = []
    if "heldout_grid" in S:
        blocks.append(("Lattice DP, H01–H40:", S["heldout_grid"], {"GridDP": "GridDP"}))
    if "heldout2_grid" in S:
        blocks.append(("Lattice DP, H41–H80:", S["heldout2_grid"], {"GridDP": "GridDP"}))
    if "heldout_mspowell" in S:
        blocks.append(("Local optimizer, H01–H40:", S["heldout_mspowell"], {"MS-Powell": "MS-Powell"}))
    return T.heldout_more(blocks) if blocks else ""


def heldout_more(S):
    """Table 3 of the journal: ablation (all variants), equal CPU time (all
    baselines), band variant, plug-in on the held-out scenarios and the
    spherical-encoding sensitivity, each block its own Holm family."""
    blocks = []
    if "heldout_ablation" in S:
        lab = {"PSO": "PSO (no DPF)", "DPF-PSO_noG": "no global fusion", "DPF-PSO_noPG": "no periodic global fusion",
               "DPF-PSO_noA": "fixed step size", "DPF-PSO_greedy": "greedy, not DP", "DPF-only": "fusion only"}
        blocks.append(("Ablation:", S["heldout_ablation"],
                       {m: lab[m] for m in S["heldout_ablation"]["methods"] if m in lab}))
    if "heldout_eqtime" in S:
        H = S["heldout_eqtime"]
        blocks.append(("Equal CPU time:", H, {m: T.lab(m) for m in H["methods"] if m != REF}))
    if "heldout_band" in S:
        blocks.append(("Band:", S["heldout_band"], {"SPSO": "SPSO", "DE": "DE", "IDP": "IDP"}))
        blocks.append(("Band, tuned:", S["heldout_band_tuned"], {"SPSO-T": "SPSO"}))
    if "heldout_plugin" in S:
        H = S["heldout_plugin"]
        blocks.append(("Plug-in, host+DPF vs. host:", H, {h: T.lab(h) for h in H["hosts"]}))
    if "heldout_sph" in S:
        blocks.append(("Spherical encoding:", S["heldout_sph"],
                       {m: T.lab(m.replace("-sph", "")) + " (spherical)" for m in S["heldout_sph"]["methods"] if m != REF}))
    if "heldout_band_sph" in S:
        blocks.append(("Spherical encoding, band:", S["heldout_band_sph"],
                       {m: T.lab(m.replace("-sph", "")) + " (spherical)" for m in S["heldout_band_sph"]["methods"] if m != REF}))
    return T.heldout_more(blocks) if blocks else ""


def components(S, methods, key="components_heldout"):
    """Composition of the final cost (median shares of the weighted terms) and
    the median of the maximum turning angle and climb-angle change per path."""
    C = S.get(key)
    if not C:
        return ""
    rows = ["| Method | Length~10~ | Threat~10~ | Altitude~10~ | Smooth.~10~ | Turn~10~ | Climb~10~ "
            "| Length~20~ | Threat~20~ | Altitude~20~ | Smooth.~20~ | Turn~20~ | Climb~20~ |",
            "|:--|" + ":-:|" * 12]
    for m in methods:
        cells = []
        for n in ("10", "20"):
            c = C.get(n, {}).get(m)
            if not c:
                cells += ["—"] * 6
                continue
            s = c["share_median"]
            cells += [T.pct(s["F1"], signed=False), T.pct(s["F2"], 2, signed=False), T.pct(s["F3"], 2, signed=False),
                      T.pct(s["F4"], signed=False), f"{c['max_turn_median']:.0f}°", f"{c['max_dclimb_median']:.0f}°"]
        rows.append(f"| {T.lab(m)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def timecurve_cpu(S, methods):
    """Median CPU time per run of the timing-curve experiment."""
    TC = S.get("timecurves")
    if not TC:
        return ""
    rows = ["| Method | CPU time~10~ (s) | CPU time~20~ (s) |", "|:--|:-:|:-:|"]
    for m in [REF] + methods:
        cells = []
        for n in ("10", "20"):
            r = next((TC[k] for k in TC if k.endswith(f"|{n}")), None)
            if r is None:
                cells.append("—")
            elif m == REF:
                cells.append(f"{r['cpu_ref']:.2f}")
            else:
                rr = TC.get(f"{m}|{n}")
                cells.append(f"{rr['cpu_other']:.2f}" if rr else "—")
        rows.append(f"| {T.lab(m)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def supplement_tables(S):
    """Tables of the journal-only supplement sections S12-S14."""
    import supplement_tables as ST
    t = {}
    for key, name in (("heldout2_tuned", "S_H2_T"), ("heldout2_band_tuned", "S_H2_BT"),
                      ("heldout2_band", "S_H2_D"), ("heldout2", "S_H2_R")):
        if key in S:
            t[name] = ST.heldout_detail(S[key])
    if "agl_heldout2" in S:
        t["S_AGL_H2"] = ST.agl_table(S["agl_heldout2"])
    if "heldout2_band" in S and "heldout2_band_tuned" in S:
        t["S_H2_BAND_FEAS"] = ST.band_feas_table(S["heldout2_band"], S["heldout2_band_tuned"])
    if "heldout_grid" in S:
        t["S_GRID"] = grid(S)
    if "timecurves" in S:
        t["S_TC_CPU"] = timecurve_cpu(S, ["SPSO", "DE", "LSHADE", "CMAES", "IDP"])
    # second round (design log 18)
    for key, name in (("heldout_sph", "S_SPH"), ("heldout_band_sph", "S_SPH_BAND"), ("heldout_mspowell", "S_MSP")):
        if key in S:
            t[name] = ST.heldout_detail(S[key])
    if "heldout_plugin" in S:
        t["S_PLUG_HO"] = T.heldout_more([("", S["heldout_plugin"], {h: T.lab(h) for h in S["heldout_plugin"]["hosts"]})])
    if "b4" in S:
        t["S_B4"] = T.design(S["b4"], star=())
    if "sph_vs_slot" in S:
        rows = ["| Method | Δ̃~10~ (spherical vs. slot) | p~H,10~ | Δ̃~20~ | p~H,20~ |", "|:--|:-:|:-:|:-:|:-:|"]
        for m, H in S["sph_vs_slot"].items():
            o = m.replace("-sph", "")
            cells = []
            for n in H["n"]:
                tt = H["tests"][f"{o}|{n}"]
                s = "" if tt["sig"] == "=" else ("^+^" if tt["sig"] == "+" else "^−^")
                cells += [T.pct(tt["gap_median"]), T.ptxt(tt["p_holm"]) + s]
            rows.append(f"| {T.lab(o)} | " + " | ".join(cells) + " |")
        t["S_SPH_VS_SLOT"] = "\n".join(rows)
    return t


def timecurves(S, methods):
    """CPU time at which DPF-PSO reaches the final median cost of a method,
    as a fraction of that method's CPU time (median over scenarios), with the
    FE fraction of the same point and the share of scenarios reached."""
    TC = S.get("timecurves")
    if not TC:
        return ""
    rows = ["| Method | CPU ratio~10~ | FE fraction~10~ | Reached~10~ | CPU ratio~20~ | FE fraction~20~ | Reached~20~ |",
            "|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]

    def f(x, pct=False):
        if not np.isfinite(x):
            return "—"
        return f"{100 * x:.0f}%" if pct else f"{x:.2f}"

    for m in methods:
        cells = []
        for n in ("10", "20"):
            r = TC.get(f"{m}|{n}")
            if not r:
                cells += ["—"] * 3
                continue
            cells += [f(r["ratio_median"]), f(r["fe_median"], pct=True), f"{100 * r['reach_share']:.0f}%"]
        rows.append(f"| {T.lab(m)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
