"""Verification of the exact ground-clearance computation.

1. Terrain.min_clearance (vectorised, screened) == independent reference
   (clearance_ref.segment_min_clearance) on random segments of every scenario.
2. Terrain.ground_violation == max(0, h_clear - reference minimum).
3. Reference minimum <= dense sampling (10,000 points) and close to it.
4. Batch independence: the violation of a segment does not depend on the
   other segments evaluated with it.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from clearance_ref import segment_min_clearance
from env import get_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def random_segments(T, rng, m, near):
    lo = np.array([5.0, 5.0])
    hi = np.array([T.size_x - 5.0, T.size_y - 5.0])
    a = lo + rng.random((m, 2)) * (hi - lo)
    L = np.where(rng.random(m) < 0.5, rng.uniform(5, 120, m), rng.uniform(120, 900, m))
    ang = rng.uniform(0, 2 * np.pi, m)
    b = np.clip(a + L[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1), lo, hi)
    if near:      # endpoints 10-60 m above ground: clearance minimum close to 20 m
        za = T.height(a[:, 0], a[:, 1]) + rng.uniform(10, 60, m)
        zb = T.height(b[:, 0], b[:, 1]) + rng.uniform(10, 60, m)
    else:
        za = T.height(a[:, 0], a[:, 1]) + rng.uniform(-50, 300, m)
        zb = T.height(b[:, 0], b[:, 1]) + rng.uniform(-50, 300, m)
    return np.c_[a, za], np.c_[b, zb]


def main():
    rng = np.random.default_rng(123)
    names = ["S1", "S2", "S3", "S4", "S5", "S6", "H01", "H17", "C1"]
    report = {}
    worst_ref, worst_v, worst_dense, n_seg = 0.0, 0.0, 0.0, 0
    for name in names:
        T = get_scenario(name).terrain
        for near in (False, True):
            A, B = random_segments(T, rng, 400, near)
            fast = T.min_clearance(A, B)
            viol = T.ground_violation(A, B, 20.0)
            ref = np.array([segment_min_clearance(T, A[i], B[i]) for i in range(len(A))])
            d_ref = np.abs(fast - ref).max()
            d_v = np.abs(viol - np.maximum(20.0 - ref, 0.0)).max()
            # dense sampling on a subset
            sub = rng.choice(len(A), 60, replace=False)
            t = np.linspace(0.0, 1.0, 10001)
            dense = np.array([(A[i, 2] + t * (B[i, 2] - A[i, 2])
                               - T.height(A[i, 0] + t * (B[i, 0] - A[i, 0]),
                                          A[i, 1] + t * (B[i, 1] - A[i, 1]))).min() for i in sub])
            gap = dense - ref[sub]
            assert d_ref < 1e-6, (name, near, d_ref)
            assert d_v < 1e-6, (name, near, d_v)
            assert gap.min() > -1e-6, (name, near, gap.min())
            worst_ref = max(worst_ref, d_ref)
            worst_v = max(worst_v, d_v)
            worst_dense = max(worst_dense, gap.max())
            n_seg += len(A)
            n_viol = int((viol > 0).sum())
            report[f"{name}_{'near' if near else 'random'}"] = dict(n=len(A), violated=n_viol,
                                                                  max_abs_diff=float(d_ref))
            print(f"{name:4s} {'near  ' if near else 'random'}: |fast-ref| max {d_ref:.2e}, "
                  f"violation diff {d_v:.2e}, dense-ref in [{gap.min():.2e}, {gap.max():.2e}], "
                  f"violated {n_viol}/{len(A)}")
        # batch independence: evaluate segments one by one versus in a batch
        A, B = random_segments(T, rng, 50, True)
        one = np.array([T.ground_violation(A[i:i + 1], B[i:i + 1], 20.0)[0] for i in range(len(A))])
        batch = T.ground_violation(A, B, 20.0)
        assert np.abs(one - batch).max() < 1e-9
    print(f"{n_seg} segments: max |vectorised - reference| = {worst_ref:.2e} m, "
          f"max violation difference = {worst_v:.2e} m; dense sampling never below the exact "
          f"minimum, at most {worst_dense:.3f} m above it")
    T = get_scenario("S4").terrain
    A, B = random_segments(T, rng, 20000, True)
    t0 = time.perf_counter()
    T.ground_violation(A, B, 20.0)
    dt = time.perf_counter() - t0
    print(f"speed: {len(A) / dt:,.0f} near-terrain segments per second (S4)")
    report["summary"] = dict(n_segments=n_seg, max_abs_diff=worst_ref, max_violation_diff=worst_v,
                             max_dense_minus_exact=worst_dense)
    with open(os.path.join(ROOT, "notes", "clearance_check.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
