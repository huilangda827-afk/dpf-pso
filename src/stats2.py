"""Statistical helpers for the second revision.

Conventions (used everywhere in the paper and supplement):
* Location statistic: median of the final cost over runs.
* Relative gap of the proposed configuration P against another method B:
  Delta(B, P) = (median_B - median_P) / median_B.  Positive: P is better.
* Effect size A12(P, B): probability that a random run of P has a lower
  cost than a random run of B (ties count 1/2); 0.5 = no effect.
* Within-instance test: two-sided Wilcoxon rank-sum test; Holm correction
  over the comparisons of one family (stated per table).
* Cross-scenario test (held-out set): two-sided Wilcoxon signed-rank test on
  the paired scenario-level medians, scenario = unit of analysis; Holm
  correction over all (baseline, n) pairs.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag):
    data = {}
    for f in glob.glob(os.path.join(ROOT, "results", tag, "*.npz")):
        inst, algo = os.path.basename(f)[:-4].split("__")
        with np.load(f) as z:
            data.setdefault(inst, {})[algo] = {k: z[k] for k in z.files}
    return data


def comp(rec, key):
    keys = list(rec["comp_keys"])
    return rec["comps"][:, keys.index(key)]


def median(rec):
    return float(np.median(rec["best_f"]))


def feasible_rate(rec):
    return float(np.mean(comp(rec, "n_viol") == 0))


def a12(x, y):
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def ranksum_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.array_equal(np.sort(x), np.sort(y)):
        return 1.0
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def signed_rank_p(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)


def gap(med_other, med_prop):
    return (med_other - med_prop) / med_other


def compare_instance(d, inst, ref, others, alpha=0.05):
    """Rank-sum tests of ref against each method in `others` on one instance,
    Holm-corrected over `others`.  Returns {method: dict(sign, p, p_holm, a12, gap)}."""
    x = d[inst][ref]["best_f"]
    ps = [ranksum_p(x, d[inst][o]["best_f"]) for o in others]
    adj = holm(ps)
    out = {}
    for o, p, pa in zip(others, ps, adj):
        y = d[inst][o]["best_f"]
        e = a12(x, y)
        sign = "=" if pa >= alpha else ("+" if e > 0.5 else "-")
        out[o] = dict(sign=sign, p=p, p_holm=float(pa), a12=e, gap=gap(np.median(y), np.median(x)))
    return out


def wtl(signs):
    return [sum(s == "+" for s in signs), sum(s == "=" for s in signs), sum(s == "-" for s in signs)]


def friedman_ranks(meds):
    """meds: (instances, methods) -> mean ranks (1 = best)."""
    meds = np.asarray(meds, float)
    return np.array([stats.rankdata(r) for r in meds]).mean(axis=0)
