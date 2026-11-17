"""Verification of the exact maximum height above ground (altitude-band
sensitivity variant).

Terrain.max_height uses the exact clearance machinery on a mirrored terrain
(-Z) with negated segment altitudes.  It is compared with
1. the independent reference implementation (clearance_ref) applied to the
   mirrored terrain, and
2. dense sampling with 10,001 points per segment (never above the exact value).
Output: notes/ceiling_check.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from check_clearance import random_segments
from clearance_ref import segment_min_clearance
from env import Terrain, get_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rng = np.random.default_rng(321)
    names = ["S1", "S4", "S6", "H01", "H17", "C1"]
    worst_ref, worst_v, worst_dense, n_seg = 0.0, 0.0, 0.0, 0
    report = {}
    for name in names:
        T = get_scenario(name).terrain
        M = Terrain(-T.Z, T.res)
        for near in (False, True):
            A, B = random_segments(T, rng, 300, near)
            if near:      # endpoints 160-240 m above ground: maximum close to 200 m
                A[:, 2] += 150.0
                B[:, 2] += 150.0
            fast = T.max_height(A, B)
            viol = T.ceiling_violation(A, B, 200.0)
            An, Bn = A.copy(), B.copy()
            An[:, 2] *= -1.0
            Bn[:, 2] *= -1.0
            ref = np.array([-segment_min_clearance(M, An[i], Bn[i]) for i in range(len(A))])
            d_ref = float(np.abs(fast - ref).max())
            d_v = float(np.abs(viol - np.maximum(ref - 200.0, 0.0)).max())
            sub = rng.choice(len(A), 60, replace=False)
            t = np.linspace(0.0, 1.0, 10001)
            dense = np.array([(A[i, 2] + t * (B[i, 2] - A[i, 2])
                               - T.height(A[i, 0] + t * (B[i, 0] - A[i, 0]),
                                          A[i, 1] + t * (B[i, 1] - A[i, 1]))).max() for i in sub])
            gap = ref[sub] - dense
            assert d_ref < 1e-6, (name, near, d_ref)
            assert d_v < 1e-6, (name, near, d_v)
            assert gap.min() > -1e-6, (name, near, gap.min())
            worst_ref, worst_v, worst_dense = max(worst_ref, d_ref), max(worst_v, d_v), max(worst_dense, gap.max())
            n_seg += len(A)
            report[f"{name}_{'near' if near else 'random'}"] = dict(n=len(A), violated=int((viol > 0).sum()),
                                                                  max_abs_diff=d_ref)
            print(f"{name:4s} {'near  ' if near else 'random'}: |fast-ref| {d_ref:.2e}, violation diff {d_v:.2e}, "
                  f"exact-dense in [{gap.min():.2e}, {gap.max():.2e}], above 200 m {int((viol > 0).sum())}/{len(A)}")
    report["summary"] = dict(n_segments=n_seg, max_abs_diff=worst_ref, max_violation_diff=worst_v,
                             max_exact_minus_dense=worst_dense)
    print(f"{n_seg} segments: max |vectorised - reference| = {worst_ref:.2e} m")
    with open(os.path.join(ROOT, "notes", "ceiling_check.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
