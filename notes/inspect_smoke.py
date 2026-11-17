import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

from env import COMP_KEYS, get_scenario

inst = sys.argv[1] if len(sys.argv) > 1 else "H01_n20"
tag = sys.argv[2] if len(sys.argv) > 2 else "smoke"
algos = sys.argv[3].split(",") if len(sys.argv) > 3 else ["SPSO", "HHO", "DPF-PSO", "IDP"]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = get_scenario(inst.split("_")[0])
print(s.title, "start", s.start, "goal", s.goal, "Z %.0f-%.0f" % (s.terrain.Z.min(), s.terrain.Z.max()))
for a in algos:
    d = np.load(os.path.join(root, "results", tag, f"{inst}__{a}.npz"))
    k = int(np.argmin(d["best_f"]))
    c = dict(zip(COMP_KEYS, d["comps"][k]))
    P = d["paths"][k]
    seg = np.diff(P, axis=0)
    hl = np.hypot(seg[:, 0], seg[:, 1])
    print(f"{a:8s} J={c['J']:.0f} F1={c['F1']:.0f} F2={c['F2']:.1f} F3={c['F3']:.0f} F4={c['F4']:.0f} "
          f"margin={c['thr_margin']:.1f} clear={c['clearance']:.1f} seg_len={hl.min():.0f}..{hl.max():.0f} "
          f"maxturn={c['max_turn']:.0f}")
