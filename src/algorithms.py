"""Optimisers used in the study.

Slot-encoded optimisers work on PathProblem.lb/ub (dimension 3n) and call an
Evaluator ``ev``; SPSO uses the spherical encoding (dimension 3n).  Each
optimiser runs until ev is exhausted; the best solution is kept by ev.

Baselines follow the original papers:
  PSO    - Shi & Eberhart (1998), linearly decreasing inertia weight (by FEs)
  SPSO   - Phung & Ha (2021), parameters of the authors' reference code
  GWO    - Mirjalili et al. (2014)
  WOA    - Mirjalili & Lewis (2016)
  HHO    - Heidari et al. (2019)
  DE     - Storn & Price (1997), DE/rand/1/bin (other strategies for tuning)
  LSHADE - Tanabe & Fukunaga (2014)
  CMAES  - Hansen & Ostermeier (2001), pycma, IPOP restarts (Auger & Hansen 2005)
  IDP    - iterative dynamic programming in the spirit of Luus (2000): exact DP
           over random candidates in a geometrically shrinking region
  GLS    - gray-box local search: one waypoint is moved at a time and only the
           affected terms are re-evaluated (2 segments = 2/(n+1) FE per move)
Optimisers that accept ``dpf`` call DPFusion.hook with their population memory
after initialisation (it = 0) and after every iteration.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from core import levy
from dpf import DPFusion, fe_cost, fuse_dp, layers_from

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _init(rng, lb, ub, N):
    return lb + rng.random((N, len(lb))) * (ub - lb)


def _rand_distinct(rng, N, k):
    """For each row i, k distinct indices from {0..N-1} \\ {i}."""
    keys = rng.random((N, N))
    keys[np.arange(N), np.arange(N)] = np.inf
    return np.argsort(keys, axis=1)[:, :k]


def _hook_pool(dpf, prob, ev, rng, it, X, f, LX, Lf):
    """Run the DPF hook on the pool [leaders; population].  Replacements that
    fall in the population part are written back to (X, f).  Returns the best
    fused (x, J) or None."""
    pool = np.vstack([LX, X])
    pf = np.concatenate([Lf, f])
    ks = dpf.hook(prob, ev, rng, pool, pf, it)
    m = len(LX)
    best = None
    for k in ks:
        if k >= m:
            X[k - m] = pool[k]
            f[k - m] = pf[k]
        if best is None or pf[k] < best[1]:
            best = (pool[k].copy(), pf[k])
    return best


# ----------------------------------------------------------------------------
def pso(prob, ev, rng, N=50, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vfrac=0.2, dpf=None):
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    X = _init(rng, lb, ub, N)
    V = np.zeros_like(X)
    f = ev(X)
    P, fP = X.copy(), f.copy()
    vmax = vfrac * (ub - lb)

    def sync(ks):
        for k in ks:
            X[k] = P[k]
            V[k] = 0.0

    if dpf is not None:
        sync(dpf.hook(prob, ev, rng, P, fP, 0))
    it = 0
    while not ev.exhausted:
        Pg = P[int(np.argmin(fP))]
        w = w_max - (w_max - w_min) * ev.frac
        V = w * V + c1 * rng.random((N, D)) * (P - X) + c2 * rng.random((N, D)) * (Pg - X)
        np.clip(V, -vmax, vmax, out=V)
        X = X + V
        out = (X < lb) | (X > ub)
        X = np.clip(X, lb, ub)
        V[out] = 0.0
        f = ev(X)
        imp = f < fP
        P[imp] = X[imp]
        fP[imp] = f[imp]
        it += 1
        if dpf is not None:
            sync(dpf.hook(prob, ev, rng, P, fP, it))
    return ev


# ----------------------------------------------------------------------------
def spso(prob, ev, rng, N=50, w0=1.0, w_end=0.98 ** 200, c1=1.5, c2=1.5, alpha=0.5):
    """SPSO (Phung & Ha 2021).  Inertia w = w0 (w_end/w0)^(FE/MaxFE); the
    default reproduces the reference schedule (w0 = 1, damping 0.98 over 200
    iterations) stretched over the FE budget; velocity clamping and mirroring
    as in the reference code."""
    lb, ub = prob.sph_lb, prob.sph_ub
    D = len(lb)
    X = _init(rng, lb, ub, N)
    V = np.zeros_like(X)
    f = ev(X)
    P, fP = X.copy(), f.copy()
    vmax = alpha * (ub - lb)
    while not ev.exhausted:
        g = int(np.argmin(fP))
        w = w0 * (w_end / w0) ** ev.frac
        V = w * V + c1 * rng.random((N, D)) * (P - X) + c2 * rng.random((N, D)) * (P[g] - X)
        np.clip(V, -vmax, vmax, out=V)
        X = X + V
        out = (X < lb) | (X > ub)
        V[out] = -V[out]
        X = np.clip(X, lb, ub)
        f = ev(X)
        imp = f < fP
        P[imp] = X[imp]
        fP[imp] = f[imp]
    return ev


# ----------------------------------------------------------------------------
def _top3(L, fL, X, f):
    poolX = np.vstack([L, X])
    poolf = np.concatenate([fL, f])
    idx, seen = [], set()
    for i in np.argsort(poolf, kind="stable"):
        key = poolX[i].tobytes()
        if key in seen:
            continue
        seen.add(key)
        idx.append(i)
        if len(idx) == 3:
            break
    while len(idx) < 3:
        idx.append(idx[-1])
    idx = np.array(idx)
    return poolX[idx].copy(), poolf[idx].copy()


def gwo(prob, ev, rng, N=50, dpf=None):
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    X = _init(rng, lb, ub, N)
    f = ev(X)
    L, fL = _top3(np.empty((0, D)), np.empty(0), X, f)
    if dpf is not None:
        b = _hook_pool(dpf, prob, ev, rng, 0, X, f, L, fL)
        if b is not None:
            L, fL = _top3(L, fL, b[0][None], np.array([b[1]]))
    it = 0
    while not ev.exhausted:
        a = 2.0 * (1.0 - ev.frac)
        Xn = np.zeros_like(X)
        for k in range(3):
            A = 2.0 * a * rng.random((N, D)) - a
            C = 2.0 * rng.random((N, D))
            Xn += L[k] - A * np.abs(C * L[k] - X)
        X = np.clip(Xn / 3.0, lb, ub)
        f = ev(X)
        L, fL = _top3(L, fL, X, f)
        it += 1
        if dpf is not None:
            b = _hook_pool(dpf, prob, ev, rng, it, X, f, L, fL)
            if b is not None:
                L, fL = _top3(L, fL, b[0][None], np.array([b[1]]))
    return ev


# ----------------------------------------------------------------------------
def woa(prob, ev, rng, N=50, b=1.0, dpf=None):
    lb, ub = prob.lb, prob.ub
    X = _init(rng, lb, ub, N)
    f = ev(X)
    i = int(np.argmin(f))
    best, fb = X[i].copy(), f[i]
    if dpf is not None:
        r = _hook_pool(dpf, prob, ev, rng, 0, X, f, best[None], np.array([fb]))
        if r is not None and r[1] < fb:
            best, fb = r
    it = 0
    while not ev.exhausted:
        a = 2.0 * (1.0 - ev.frac)
        a2 = -1.0 - ev.frac
        A = 2.0 * a * rng.random(N) - a
        C = 2.0 * rng.random(N)
        p = rng.random(N)
        l = (a2 - 1.0) * rng.random(N) + 1.0
        Xn = np.empty_like(X)
        m1 = (p < 0.5) & (np.abs(A) < 1.0)
        m2 = (p < 0.5) & (np.abs(A) >= 1.0)
        m3 = p >= 0.5
        Xn[m1] = best - A[m1, None] * np.abs(C[m1, None] * best - X[m1])
        Xr = X[rng.integers(0, N, int(m2.sum()))]
        Xn[m2] = Xr - A[m2, None] * np.abs(C[m2, None] * Xr - X[m2])
        Xn[m3] = np.abs(best - X[m3]) * np.exp(b * l[m3, None]) * np.cos(2 * np.pi * l[m3, None]) + best
        X = np.clip(Xn, lb, ub)
        f = ev(X)
        i = int(np.argmin(f))
        if f[i] < fb:
            best, fb = X[i].copy(), f[i]
        it += 1
        if dpf is not None:
            r = _hook_pool(dpf, prob, ev, rng, it, X, f, best[None], np.array([fb]))
            if r is not None and r[1] < fb:
                best, fb = r
    return ev


# ----------------------------------------------------------------------------
def hho(prob, ev, rng, N=50, beta=1.5, dpf=None):
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    X = _init(rng, lb, ub, N)
    f = ev(X)
    i = int(np.argmin(f))
    rab, fr = X[i].copy(), f[i]
    if dpf is not None:
        r = _hook_pool(dpf, prob, ev, rng, 0, X, f, rab[None], np.array([fr]))
        if r is not None and r[1] < fr:
            rab, fr = r
    it = 0
    while not ev.exhausted:
        E = 2.0 * (2.0 * rng.random(N) - 1.0) * (1.0 - ev.frac)
        absE = np.abs(E)
        Xm = X.mean(axis=0)
        q = rng.random(N)
        r = rng.random(N)
        Jmp = 2.0 * (1.0 - rng.random(N))
        Xn = X.copy()
        fn = f.copy()
        explore = absE >= 1.0
        e1 = explore & (q >= 0.5)
        e2 = explore & (q < 0.5)
        soft = (~explore) & (r >= 0.5) & (absE >= 0.5)
        hard = (~explore) & (r >= 0.5) & (absE < 0.5)
        sdive = (~explore) & (r < 0.5) & (absE >= 0.5)
        hdive = (~explore) & (r < 0.5) & (absE < 0.5)
        if e1.any():
            k = int(e1.sum())
            Xr = X[rng.integers(0, N, k)]
            r1, r2 = rng.random((k, 1)), rng.random((k, 1))
            Xn[e1] = Xr - r1 * np.abs(Xr - 2.0 * r2 * X[e1])
        if e2.any():
            k = int(e2.sum())
            r3, r4 = rng.random((k, 1)), rng.random((k, 1))
            Xn[e2] = (rab - Xm) - r3 * (lb + r4 * (ub - lb))
        if soft.any():
            Xn[soft] = (rab - X[soft]) - E[soft, None] * np.abs(Jmp[soft, None] * rab - X[soft])
        if hard.any():
            Xn[hard] = rab - E[hard, None] * np.abs(rab - X[hard])
        plain = e1 | e2 | soft | hard
        if plain.any():
            Xp = np.clip(Xn[plain], lb, ub)
            Xn[plain] = Xp
            fn[plain] = ev(Xp)
        dive = sdive | hdive
        if dive.any() and not ev.exhausted:
            idx = np.flatnonzero(dive)
            ref = np.where(sdive[idx, None], X[idx], Xm[None, :])
            Y = rab - E[idx, None] * np.abs(Jmp[idx, None] * rab - ref)
            Y = np.clip(Y, lb, ub)
            fY = ev(Y)
            okY = fY < f[idx]
            Xn[idx[okY]] = Y[okY]
            fn[idx[okY]] = fY[okY]
            rest = idx[~okY]
            if len(rest) and not ev.exhausted:
                Z = Y[~okY] + rng.random((len(rest), D)) * levy(rng, (len(rest), D), beta)
                Z = np.clip(Z, lb, ub)
                fZ = ev(Z)
                okZ = fZ < f[rest]
                Xn[rest[okZ]] = Z[okZ]
                fn[rest[okZ]] = fZ[okZ]
        X, f = Xn, fn
        i = int(np.argmin(f))
        if f[i] < fr:
            rab, fr = X[i].copy(), f[i]
        it += 1
        if dpf is not None:
            r = _hook_pool(dpf, prob, ev, rng, it, X, f, rab[None], np.array([fr]))
            if r is not None and r[1] < fr:
                rab, fr = r
    return ev


# ----------------------------------------------------------------------------
def de(prob, ev, rng, N=50, F=0.5, CR=0.9, strategy="rand1bin", dpf=None):
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    N = int(N)
    X = _init(rng, lb, ub, N)
    f = ev(X)
    if dpf is not None:
        dpf.hook(prob, ev, rng, X, f, 0)
    it = 0
    ar = np.arange(N)
    while not ev.exhausted:
        r = _rand_distinct(rng, N, 3)
        if strategy == "rand1bin":
            V = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        elif strategy == "best1bin":
            V = X[int(np.argmin(f))] + F * (X[r[:, 0]] - X[r[:, 1]])
        elif strategy == "ctb1bin":          # current-to-best/1/bin
            V = X + F * (X[int(np.argmin(f))] - X) + F * (X[r[:, 0]] - X[r[:, 1]])
        else:
            raise ValueError(strategy)
        mask = rng.random((N, D)) < CR
        mask[ar, rng.integers(0, D, N)] = True
        U = np.where(mask, V, X)
        out = (U < lb) | (U > ub)
        U = np.where(out, lb + rng.random((N, D)) * (ub - lb), U)
        fu = ev(U)
        better = fu <= f
        X[better] = U[better]
        f[better] = fu[better]
        it += 1
        if dpf is not None:
            dpf.hook(prob, ev, rng, X, f, it)
    return ev


# ----------------------------------------------------------------------------
def lshade(prob, ev, rng, r_ninit=18.0, N_min=4, H=6, r_arc=2.6, p_best=0.11, dpf=None):
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    N_init = int(round(r_ninit * D))
    N = N_init
    H = int(H)
    X = _init(rng, lb, ub, N)
    f = ev(X)
    if dpf is not None:
        dpf.hook(prob, ev, rng, X, f, 0)
    MF = np.full(H, 0.5)
    MCR = np.full(H, 0.5)      # NaN encodes the terminal value
    k = 0
    A = np.empty((0, D))
    it = 0
    while not ev.exhausted:
        ar = np.arange(N)
        ri = rng.integers(0, H, N)
        mcr = MCR[ri]
        CR = np.where(np.isnan(mcr), 0.0,
                      np.clip(np.nan_to_num(mcr) + 0.1 * rng.standard_normal(N), 0.0, 1.0))
        Fv = MF[ri] + 0.1 * rng.standard_cauchy(N)
        bad = Fv <= 0
        while bad.any():
            Fv[bad] = MF[ri[bad]] + 0.1 * rng.standard_cauchy(int(bad.sum()))
            bad = Fv <= 0
        Fv = np.minimum(Fv, 1.0)
        order = np.argsort(f, kind="stable")
        npb = max(2, int(round(p_best * N)))
        pb = order[rng.integers(0, npb, N)]
        r1 = rng.integers(0, N, N)
        bad = r1 == ar
        while bad.any():
            r1[bad] = rng.integers(0, N, int(bad.sum()))
            bad = r1 == ar
        PA = np.vstack([X, A]) if len(A) else X
        r2 = rng.integers(0, len(PA), N)
        bad = (r2 == ar) | (r2 == r1)
        while bad.any():
            r2[bad] = rng.integers(0, len(PA), int(bad.sum()))
            bad = (r2 == ar) | (r2 == r1)
        Fc = Fv[:, None]
        V = X + Fc * (X[pb] - X) + Fc * (X[r1] - PA[r2])
        V = np.where(V < lb, 0.5 * (lb + X), V)
        V = np.where(V > ub, 0.5 * (ub + X), V)
        mask = rng.random((N, D)) < CR[:, None]
        mask[ar, rng.integers(0, D, N)] = True
        U = np.where(mask, V, X)
        fu = ev(U)
        imp = fu < f
        if imp.any():
            A = np.vstack([A, X[imp]])
            SCR, SF, df = CR[imp], Fv[imp], f[imp] - fu[imp]
        keep = fu <= f
        X[keep] = U[keep]
        f[keep] = fu[keep]
        if imp.any():
            w = df / df.sum()
            MF[k] = np.sum(w * SF ** 2) / np.sum(w * SF)
            if np.isnan(MCR[k]) or SCR.max() == 0:
                MCR[k] = np.nan
            else:
                MCR[k] = np.sum(w * SCR ** 2) / np.sum(w * SCR)
            k = (k + 1) % H
        N_new = max(N_min, int(round((N_min - N_init) / ev.max * ev.used + N_init)))
        if N_new < N:
            sel = np.argsort(f, kind="stable")[:N_new]
            X, f, N = X[sel], f[sel], N_new
        amax = int(round(r_arc * N))
        if len(A) > amax:
            A = A[rng.choice(len(A), amax, replace=False)]
        it += 1
        if dpf is not None:
            dpf.hook(prob, ev, rng, X, f, it)
    return ev


# ----------------------------------------------------------------------------
def cmaes(prob, ev, rng, sigma0=0.3, popsize=None, ipop=True, tolx=1e-3, tolfun=1.0, tolstagnation=200,
          pop_mult=1.0):
    """CMA-ES (pycma) on the slot encoding scaled to [0, 1]^D, with IPOP
    restarts (population doubled after each restart) until the budget is used.
    Restart triggers are scaled to the problem: tolx in normalised
    coordinates, tolfun and tolfunhist in cost units, tolstagnation in
    iterations.  (With pycma's default tolerances the first run never stops
    within 3000n FEs, so IPOP would never restart.)"""
    import cma
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    lam = int(popsize or round(float(pop_mult) * (4 + int(3 * np.log(D)))))
    while not ev.exhausted:
        opts = {"bounds": [0.0, 1.0], "popsize": lam, "seed": int(rng.integers(1, 2 ** 31 - 1)),
                "verbose": -9, "verb_log": 0, "verb_disp": 0, "maxfevals": int(ev.remaining),
                "tolfun": float(tolfun), "tolfunhist": float(tolfun), "tolx": float(tolx),
                "tolstagnation": int(tolstagnation)}
        es = cma.CMAEvolutionStrategy(rng.random(D), sigma0, opts)
        while not es.stop() and not ev.exhausted:
            Y = np.asarray(es.ask())
            f = ev(lb + np.clip(Y, 0.0, 1.0) * (ub - lb))
            if not np.all(np.isfinite(f)):
                break
            es.tell(list(Y), f.tolist())
        if not ipop:
            break
        lam *= 2
    if not ev.exhausted:                      # without restarts: spend the rest randomly
        ev(_init(rng, lb, ub, ev.remaining))
    return ev


# ----------------------------------------------------------------------------
def idp(prob, ev, rng, K=20, r_min=1e-3, N0=50):
    """Iterative dynamic programming (structure-exploiting baseline).  Each round
    draws K-1 random candidates per slot uniformly in a box of half-width
    r * range around the incumbent's waypoint, adds the incumbent's waypoint
    and solves the fusion problem exactly.  r shrinks geometrically from 1 to
    r_min over the budget (Luus-style region contraction)."""
    lb, ub = prob.lb, prob.ub
    rngw = ub - lb
    X = _init(rng, lb, ub, int(N0))
    f = ev(X)
    best = X[int(np.argmin(f))].copy()
    fb = float(np.min(f))
    per_round = int(math.ceil(((prob.n - 1) * K * K + 2 * K) / (prob.n + 1)))
    rounds = max(1, ev.remaining // per_round)
    k = 0
    while not ev.exhausted:
        r = r_min ** (min(k, rounds - 1) / max(1, rounds - 1))
        lo = np.maximum(lb, best - r * rngw)       # half-width r * range: r = 1 covers
        hi = np.minimum(ub, best + r * rngw)       # the whole box in the first round
        C = np.vstack([best[None], lo + rng.random((K - 1, prob.dim)) * (hi - lo)])
        layers = layers_from(prob, C)
        fes, _ = fe_cost(prob, layers)
        if fes > ev.remaining:
            ev(_init(rng, lb, ub, ev.remaining))
            break
        x, J = fuse_dp(prob, layers)
        ev.charge(fes, x, J)
        if J < fb:
            best, fb = x, J
        k += 1
    return ev


# ----------------------------------------------------------------------------
def ms_powell(prob, ev, rng, xtol=1e-3, ftol=1e-4):
    """Multi-start local optimizer baseline (journal version, design log
    section 18e): bounded Powell direction-set search (scipy) on the slot
    variables scaled to [0, 1], restarted from a new uniform random start
    whenever it converges, until the FE budget is spent.  Every objective
    call goes through the evaluator and is charged as one FE.  Represents
    gradient-free trajectory optimization of the same cost."""
    from scipy.optimize import minimize
    lb, ub = prob.lb, prob.ub
    rngw = ub - lb

    class _Budget(Exception):
        pass

    def f(z):
        if ev.exhausted:
            raise _Budget
        return float(ev((lb + np.clip(z, 0.0, 1.0) * rngw)[None])[0])

    bounds = [(0.0, 1.0)] * prob.dim
    while not ev.exhausted:
        z0 = rng.random(prob.dim)
        try:
            minimize(f, z0, method="Powell", bounds=bounds,
                     options=dict(xtol=xtol, ftol=ftol, maxfev=max(1, ev.remaining), maxiter=10 ** 6))
        except _Budget:
            break
    return ev


# ----------------------------------------------------------------------------
def grid_dp(prob, ev, rng, n_h=3, n_tau=3):
    """Fixed-lattice baseline (journal version, design log section 16): one
    exact DP over a regular lattice of candidate positions per slot, using
    the largest lattice whose segment evaluations fit into the FE budget.
    Per slot: n_u equally spaced lateral offsets (slot bounds included), n_h
    heights (band limits and centre) and n_tau along-track offsets (bounds
    and 0).  Single shot: no search, no randomness, leftover FEs unused."""
    n = prob.n
    nseg_max = ev.remaining * (n + 1)
    # segments of one DP with K candidates per slot: 2K + (n - 1) K^2
    a, b = n - 1, 2
    K = int((-b + math.sqrt(b * b + 4 * a * nseg_max)) / (2 * a)) if a > 0 else nseg_max // 2
    n_u = max(2, K // (n_h * n_tau))
    layers = []
    for j in range(n):
        idx = prob.var_idx[j]
        lo, hi = prob.lb[idx], prob.ub[idx]
        axes = [np.linspace(lo[0], hi[0], n_u), np.linspace(lo[1], hi[1], n_h)]
        if len(idx) > 2:
            axes.append(np.linspace(lo[2], hi[2], n_tau))
        layers.append(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(idx)))
    fes, _ = fe_cost(prob, layers)
    while fes > ev.remaining and n_u > 2:        # guard against rounding
        n_u -= 1
        layers = [np.stack(np.meshgrid(np.linspace(prob.lb[i][0], prob.ub[i][0], n_u),
                                       *[np.linspace(prob.lb[i][k], prob.ub[i][k], (n_h, n_tau)[k - 1])
                                         for k in range(1, len(i))], indexing="ij"), axis=-1).reshape(-1, len(i))
                  for i in prob.var_idx]
        fes, _ = fe_cost(prob, layers)
    x, J = fuse_dp(prob, layers)
    ev.charge(fes, x, J)
    return ev


# ----------------------------------------------------------------------------
def gls(prob, ev, rng, B=20, sigma0=0.1, sigma_min=1e-4, N0=50):
    """Gray-box local search with incremental evaluation.  Each step picks one
    waypoint, proposes B Gaussian moves of it, evaluates only the affected terms
    (its altitude term, its two segments and up to three turns; charged as
    2/(n+1) FE per proposal) and keeps the best move if it improves the path.
    One-fifth success rule for the step size; restart from a random solution
    when the step falls below sigma_min."""
    lb, ub = prob.lb, prob.ub
    n = prob.n
    rngw = ub - lb
    X = _init(rng, lb, ub, int(N0))
    f = ev(X)
    x = X[int(np.argmin(f))].copy()
    frac_fe = 0.0
    per_prop = 2.0 / (n + 1)

    def state(x):
        P, h = prob.plane_to_points(x[None])
        F = prob.full_path(P)[0]                   # (n+2, 3)
        node = prob.node_cost(h[0])                # (n,)
        edge = prob.edge_cost(F[:-1], F[1:])       # (n+1,)
        turn = prob.tri_cost(F[:-2], F[1:-1], F[2:])   # (n,), turn at waypoint j+1
        return F, node, edge, turn

    F, node, edge, turn = state(x)
    fx = float(node.sum() + edge.sum() + turn.sum())
    sigma = sigma0
    up, down = 1.5, 1.5 ** -0.25
    while True:
        need = B * per_prop
        if frac_fe + need > ev.remaining:
            break
        j = int(rng.integers(n))                   # waypoint j+1 of the full path F
        idx = prob.var_idx[j]
        cand = np.clip(x[idx] + rng.standard_normal((B, len(idx))) * sigma * rngw[idx], lb[idx], ub[idx])
        Pc = prob.layer_points(j, cand)            # (B, 3)
        a, b = F[j], F[j + 2]                      # neighbours of the moved waypoint
        nc = prob.node_cost(cand[:, 1])
        e1 = prob.edge_cost(a[None], Pc)
        e2 = prob.edge_cost(Pc, b[None])
        tm = prob.tri_cost(a[None], Pc, b[None])   # turn at the moved waypoint
        dt = tm - turn[j]
        tp = tn = None
        if j >= 1:                                 # turn at F[j]
            tp = prob.tri_cost(F[j - 1][None], a[None], Pc)
            dt = dt + tp - turn[j - 1]
        if j + 1 <= n - 1:                         # turn at F[j+2]
            tn = prob.tri_cost(Pc, b[None], F[j + 3][None])
            dt = dt + tn - turn[j + 1]
        delta = nc - node[j] + e1 - edge[j] + e2 - edge[j + 1] + dt
        frac_fe += need
        if frac_fe >= 1.0:
            whole = int(frac_fe)
            ev.charge(whole)
            frac_fe -= whole
        k = int(np.argmin(delta))
        if delta[k] < -1e-9:
            # incremental update of the cached terms with the accepted move
            x[idx] = cand[k]
            F[j + 1] = Pc[k]
            node[j], edge[j], edge[j + 1] = nc[k], e1[k], e2[k]
            if j >= 1:
                turn[j - 1] = tp[k]
            turn[j] = tm[k]
            if j + 1 <= n - 1:
                turn[j + 1] = tn[k]
            fx = float(node.sum() + edge.sum() + turn.sum())
            ev.charge(0, x, fx)
            sigma = min(0.5, sigma * up)
        else:
            sigma *= down
        if sigma < sigma_min:                      # restart
            if ev.remaining < 1 + B * per_prop:
                break
            xr = _init(rng, lb, ub, 1)
            ev(xr)
            x = xr[0].copy()
            F, node, edge, turn = state(x)
            fx = float(node.sum() + edge.sum() + turn.sum())
            sigma = sigma0
    if ev.remaining > 0:
        ev.charge(ev.remaining)
    return ev


# ----------------------------------------------------------------------------
def dpf_only(prob, ev, rng, N=50, dpf=None, locals_per_global=20):
    """Ablation: DPF without any host search dynamics.  A random population is
    fused globally, then refined by local fusions only; a global fusion with
    fresh random candidates follows every `locals_per_global` local fusions."""
    X = _init(rng, prob.lb, prob.ub, N)
    f = ev(X)
    dpf._h0 = ev.used - ev.dp_fes
    dpf._accept(X, f, dpf.global_fuse(prob, ev, X, f, rng), [], "global")
    k = 0
    while not ev.exhausted:
        k += 1
        if k % locals_per_global == 0:
            res = dpf.global_fuse(prob, ev, X, f, rng)
            dpf._accept(X, f, res, [], "global")
        else:
            res = dpf.local_fuse(prob, ev, rng, X, f)
            dpf.adapt_sigma(dpf._accept(X, f, res, [], "local"), ev)
        if res is None:                     # budget too small for another fusion
            best = X[int(np.argmin(f))]
            sig = dpf.sigma * (prob.ub - prob.lb)
            ev(np.clip(best + rng.standard_normal((ev.remaining, prob.dim)) * sig, prob.lb, prob.ub))
    return ev


# ----------------------------------------------------------------------------
# Registry: name -> (function, encoding, kwargs factory(prob))
DPF_DEFAULT = dict(init_global=True, g_every=5000, l_every=250, g_max=50, g_random=50,
                   n_elite=5, n_perturb=6, use_mid=False, adapt=True, mode="dp")


class _SphView:
    """The problem seen through the spherical encoding of SPSO (journal
    version, design log 18f): bounds and dimension of the spherical
    variables, everything else delegated.  Lets the generic black-box
    optimizers run on SPSO's encoding; the runner evaluates with
    prob.evaluate_sph for encoding 'sph'."""

    def __init__(self, prob):
        self._p = prob
        self.lb, self.ub = prob.sph_lb, prob.sph_ub
        self.dim = len(self.lb)

    def __getattr__(self, name):
        return getattr(self._p, name)


def _sph(fn):
    return lambda prob, ev, rng, **kw: fn(_SphView(prob), ev, rng, **kw)


def _dpf(**kw):
    cfg = {**DPF_DEFAULT, **kw}
    return lambda prob: dict(dpf=DPFusion(**cfg))


def _const(**kw):
    return lambda prob: dict(kw)


ALGOS = {
    "PSO": (pso, "slot", _const()),
    "SPSO": (spso, "sph", _const()),
    "GWO": (gwo, "slot", _const()),
    "WOA": (woa, "slot", _const()),
    "HHO": (hho, "slot", _const()),
    "DE": (de, "slot", _const()),
    "LSHADE": (lshade, "slot", _const()),
    "CMAES": (cmaes, "slot", _const()),
    "IDP": (idp, "slot", _const()),
    "GLS": (gls, "slot", _const()),
    "GridDP": (grid_dp, "slot", _const()),      # journal version, design log 16
    "MS-Powell": (ms_powell, "slot", _const()),  # journal version, design log 18e
    "DE-sph": (_sph(de), "sph", _const()),       # journal version, design log 18f
    "CMAES-sph": (_sph(cmaes), "sph", _const()),
    # proposed
    "DPF-PSO": (pso, "slot", _dpf()),
    # plug-in study
    "GWO+DPF": (gwo, "slot", _dpf()),
    "WOA+DPF": (woa, "slot", _dpf()),
    "HHO+DPF": (hho, "slot", _dpf()),
    "DE+DPF": (de, "slot", _dpf()),
    "LSHADE+DPF": (lshade, "slot", _dpf()),
    # ablation (PSO host)
    "DPF-PSO_noG": (pso, "slot", _dpf(init_global=False, g_every=0)),
    "DPF-PSO_noPG": (pso, "slot", _dpf(g_every=0)),        # journal version, design log 18b: initial global fusion only
    "DPF-PSO_noR": (pso, "slot", _dpf(g_random=0)),
    "DPF-PSO_noL": (pso, "slot", _dpf(l_every=0)),
    "DPF-PSO_noA": (pso, "slot", _dpf(adapt=False)),
    "DPF-PSO_greedy": (pso, "slot", _dpf(mode="greedy")),
    "DPF-only": (dpf_only, "slot", _dpf()),
    # population-size sensitivity of the two strongest baselines
    "DE-NP5D": (de, "slot", lambda prob: dict(N=5 * prob.dim)),
    "DE-NP10D": (de, "slot", lambda prob: dict(N=10 * prob.dim)),
    "SPSO-N100": (spso, "sph", _const(N=100)),
    "SPSO-N500": (spso, "sph", _const(N=500)),
}

TUNED = ["DE", "LSHADE", "SPSO", "IDP", "DPF-PSO", "CMAES"]     # CMAES added in the third revision


def tuned_config(name):
    with open(os.path.join(ROOT, "results", "tuning", f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)["selected"]


def algo_spec(name):
    """(function, encoding, kwargs factory) for a registered name or a tuned
    variant '<name>-T' (configuration from results/tuning/<name>.json)."""
    if name in ALGOS:
        return ALGOS[name]
    if name.endswith("-T") and name[:-2] in TUNED:
        base = name[:-2]
        cfg = tuned_config(base)
        fn, enc, _ = ALGOS[base]
        return fn, enc, config_factory(base, cfg)
    raise KeyError(name)


def config_factory(base, cfg):
    """kwargs factory for a configuration dictionary of a tunable algorithm."""
    cfg = dict(cfg)
    if base == "DPF-PSO":
        return lambda prob: dict(dpf=DPFusion(**{**DPF_DEFAULT, **cfg}))
    if base == "DE":
        def f(prob):
            kw = dict(cfg)
            np_ = kw.pop("NP")
            kw["N"] = int(np_[:-1]) * prob.dim if isinstance(np_, str) else int(np_)
            return kw
        return f
    return lambda prob: dict(cfg)


MAIN = ["PSO", "SPSO", "GWO", "WOA", "HHO", "DE", "LSHADE", "CMAES", "IDP", "GLS", "DPF-PSO"]
PLUGIN = [("PSO", "DPF-PSO"), ("GWO", "GWO+DPF"), ("WOA", "WOA+DPF"), ("HHO", "HHO+DPF"),
          ("DE", "DE+DPF"), ("LSHADE", "LSHADE+DPF")]
ABLATION = ["PSO", "DPF-PSO_noG", "DPF-PSO_noR", "DPF-PSO_noL", "DPF-PSO_noA",
            "DPF-PSO_greedy", "DPF-only", "DPF-PSO"]
