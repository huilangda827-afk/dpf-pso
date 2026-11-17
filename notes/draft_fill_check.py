"""Scratch check: fill the paper and supplement templates in draft mode and
list the values that are still missing (third revision, while experiments
are running).  Not needed to reproduce the paper."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
import build2 as B  # noqa: E402
import supplement_tables as ST  # noqa: E402

v, t, S = B.compute()
t.update(ST.tables(S))
for name in ("paper_template2.md", "supplement_template.md"):
    tpl = open(os.path.join(ROOT, "paper", name), encoding="utf-8").read()
    md = B.fill(tpl, v, t, strict=False)
    print(name, "missing:", sorted(set(re.findall(r"MISSING:([A-Za-z0-9_\-+]+)", md))))
    failed = []
    for m in re.findall(r"<!--ASSERT (.*?)-->", tpl, flags=re.S):
        try:
            ok = eval(m, {"v": v, "S": S, "np": B.np})
        except Exception as e:          # missing keys while experiments run
            ok = f"error: {type(e).__name__}: {e}"
        if ok is not True:
            failed.append((m[:120], ok))
    for m, ok in failed:
        print("  ASSERT", ok, "|", m)
for k in ("safe_disagree", "hot_plus_list20", "_hot_plus10", "_hot_minus10", "_hot_eq20", "n_secondary",
          "hab_text", "swarm_gain", "heq_text", "band_text", "abs_spso", "disc_band", "conc_band"):
    print(k, "=", v.get(k))
