"""Quick look at stored results while the pipeline runs (not used by the paper).

    python notes/peek_results.py main
    python notes/peek_results.py heldout
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

import analysis2 as A
import stats2 as S
from algorithms import MAIN, TUNED


def show_design(d, methods, ref):
    R = A.design_results(d, methods, "x", ref=ref)
    print("mean ranks:", ", ".join(f"{m} {x:.2f}" for m, x in zip(R["methods"], R["ranks"])))
    print("lowest median:", R["best_count"], "of", len(R["instances"]))
    for m in methods:
        if m != ref:
            print(f"  {ref} vs {m}: +/=/- {R['wtl'][m]}")
    for i in R["instances"]:
        r = R["per_instance"][i]
        cells = []
        for m in methods:
            s = "" if m == ref else r["cmp"][m]["sign"]
            f = "" if r["feas"][m] == 1 else f"[{100 * r['feas'][m]:.0f}%]"
            cells.append(f"{m}:{r['median'][m]:.0f}{s}{f}")
        print(" ", i, " ".join(cells))
    return R


def main(tag):
    d = S.load(tag)
    if tag == "heldout":
        H = A.heldout_results(d, MAIN)
        for n in H["n"]:
            hn = H[f"n{n}"]
            print(f"n={n}: scenarios {hn['n_scen']}, DPF-PSO lowest median on {hn['best_count']}, all-fail {hn['all_fail']}")
            print("  mean ranks:", ", ".join(f"{m} {x:.2f}" for m, x in zip(H["methods"], hn["ranks"])))
            print("  feasible:", ", ".join(f"{m} {100 * hn['feas'][m]:.0f}%" for m in H["methods"]))
            for o in MAIN[:-1]:
                t = H["tests"][f"{o}|{n}"]
                print(f"  vs {o:7s} win {100 * t['win']:3.0f}% loss {100 * t['loss']:3.0f}% gap {100 * t['gap_median']:+6.2f}% "
                      f"[{100 * t['gap_q1']:+.2f}, {100 * t['gap_q3']:+.2f}] p {t['p']:.2e} pH {t['p_holm']:.2e} {t['sig']} "
                      f"A12 {t['a12_median']:.2f} within {t['within_wtl']}")
        tuned = [t + "-T" for t in TUNED]
        if all(any(t in d[i] for i in d) for t in tuned):
            Ht = A.heldout_results(d, tuned, ref="DPF-PSO-T")
            for n in Ht["n"]:
                print(f"tuned n={n}: scenarios {Ht[f'n{n}']['n_scen']}")
                for o in tuned[:-1]:
                    t = Ht["tests"][f"{o}|{n}"]
                    print(f"  vs {o:9s} win {100 * t['win']:3.0f}% gap {100 * t['gap_median']:+6.2f}% pH {t['p_holm']:.2e} {t['sig']}")
        return
    present = sorted({m for i in d for m in d[i]})
    main_m = [m for m in MAIN if m in present]
    show_design(d, main_m, "DPF-PSO")
    tuned = [t + "-T" for t in TUNED]
    if all(t in present for t in tuned):
        print("--- tuned")
        show_design(d, tuned, "DPF-PSO-T")
    if tag == "main":
        print("--- ablation", A.ablation_results(d))
        pl = A.plugin_results(d)
        for h, p in pl.items():
            print(f"--- plugin {h}: gap mean {100 * p['gap_mean']:+.2f}% median {100 * p['gap_median']:+.2f}% wtl {p['wtl']} "
                  f"globals {p['n_global']:.1f} locals {p['n_local']:.0f}")
        print("--- fe", A.fe_composition(d))
        for v in ("DE-NP5D", "DE-NP10D", "SPSO-N100", "SPSO-N500"):
            c = [S.compare_instance(d, i, "DPF-PSO", [v])[v] for i in A.DESIGN]
            print(f"--- sens {v}: wtl {S.wtl([x['sign'] for x in c])} gap {100 * np.mean([x['gap'] for x in c]):+.2f}%")
        c = [S.compare_instance(d, i, "DPF-PSO", ["DPF-PSO-T"])["DPF-PSO-T"] for i in A.DESIGN]
        print(f"--- default vs tuned DPF-PSO: wtl {S.wtl([x['sign'] for x in c])} gap {100 * np.mean([x['gap'] for x in c]):+.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
