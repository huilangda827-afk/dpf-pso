"""Scratch: print the third-revision held-out families from summary.json
(ASCII only, for logs).  Not needed to reproduce the paper."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "results", "analysis", "summary.json"), encoding="utf-8"))
for fam in ("heldout_ablation", "heldout_eqtime", "heldout_band", "heldout_band_tuned"):
    if fam not in S:
        print(fam, "missing")
        continue
    R = S[fam]
    print(f"== {fam} (family size {R.get('family_size', len(R['tests']))})")
    for k, t in R["tests"].items():
        print(f"  {k:22s} win {t['win']:.2f} loss {t['loss']:.2f} gap {100 * t['gap_median']:+.2f}% "
              f"[{100 * t['gap_q1']:+.2f}, {100 * t['gap_q3']:+.2f}] p {t['p']:.2e} pH {t['p_holm']:.2e} "
              f"sig {t['sig']} within {t['within_wtl']}")
    for n in R["n"]:
        x = R[f"n{n}"]
        print(f"  n={n}: feas " + ", ".join(f"{m} {100 * f:.1f}%" for m, f in x["feas"].items())
              + f"; all_fail {x['all_fail']}; ranks " + ", ".join(f"{m} {r:.2f}" for m, r in zip(R["methods"], x["ranks"])))
for key in ("agl_heldout", "agl_band"):
    if key in S:
        print("==", key)
        for n, x in S[key].items():
            for m, a in x.items():
                print(f"  n={n} {m:10s} below {100 * a['below_mean']:.1f}% above {100 * a['above_mean']:.1f}% "
                      f"min {a['min_median']:.0f} max {a['max_median']:.0f}")
if "safety_band" in S:
    print("== safety_band (cost-function view)")
    for m, x in S["safety_band"].items():
        print("  ", m, {k: x[k] for k in x if k in ("n", "feasible", "min_clear", "tangent")})
