"""Quick textual summary of a results folder (mean/std/best/success rate)."""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(tag, order=None):
    files = sorted(glob.glob(os.path.join(ROOT, "results", tag, "*.npz")))
    data = {}
    for f in files:
        inst, algo = os.path.basename(f)[:-4].split("__")
        data.setdefault(inst, {})[algo] = np.load(f)
    for inst in sorted(data):
        print(f"== {inst}")
        algos = order or sorted(data[inst])
        for a in algos:
            if a not in data[inst]:
                continue
            d = data[inst][a]
            b = d["best_f"]
            feas = np.mean(d["comps"][:, 4] == 0) * 100
            print(f"  {a:14s} mean={b.mean():10.1f} std={b.std():9.1f} best={b.min():9.1f} "
                  f"worst={b.max():10.1f} feas={feas:5.1f}% t={d['times'].mean():.2f}s "
                  f"dpFE={d['dp_fes'].mean():7.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or None)
