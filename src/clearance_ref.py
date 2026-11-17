"""Independent reference implementation of the exact segment ground clearance.

Written separately from Terrain.ground_violation (different derivation): the
segment is split at every grid-line crossing; inside each sub-interval the
clearance is a quadratic in the segment parameter, which is reconstructed from
three evaluations of the bilinear terrain and minimised analytically.  Slow
(pure Python loop over cells), used only for verification.
"""
from __future__ import annotations

import math

import numpy as np


def segment_min_clearance(T, a, b):
    """Exact minimum of z_line - terrain along the segment a->b (3-vectors)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ga = a[:2] / T.res
    gb = b[:2] / T.res
    cuts = {0.0, 1.0}
    for k in range(2):
        lo, hi = min(ga[k], gb[k]), max(ga[k], gb[k])
        if hi - lo <= 0.0:
            continue
        for m in range(int(math.floor(lo)) + 1, int(math.ceil(hi))):
            s = (m - ga[k]) / (gb[k] - ga[k])
            if 0.0 < s < 1.0:
                cuts.add(s)
    cuts = sorted(cuts)

    def clear(s):
        p = a + s * (b - a)
        return float(p[2] - T.height(p[0], p[1]))

    best = math.inf
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        c0, c1 = clear(s0), clear(s1)
        best = min(best, c0, c1)
        if s1 - s0 < 1e-12:
            continue
        sm = 0.5 * (s0 + s1)
        cm = clear(sm)
        hw = 0.5 * (s1 - s0)
        gam = (c0 - 2.0 * cm + c1) / (2.0 * hw * hw)      # c(s) = cm + beta (s-sm) + gam (s-sm)^2
        beta = (c1 - c0) / (2.0 * hw)
        if gam > 0.0:
            ds = -beta / (2.0 * gam)
            if -hw < ds < hw:
                best = min(best, cm + beta * ds + gam * ds * ds)
    return best


def path_min_clearance(T, path):
    """Exact minimum clearance of a polyline path (k, 3) and per-segment values."""
    vals = [segment_min_clearance(T, path[i], path[i + 1]) for i in range(len(path) - 1)]
    return min(vals), np.array(vals)
