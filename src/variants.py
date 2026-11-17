"""Design exploration for the proposed method (pilot only).

dpf_pso2 extends PSO with three optional mechanisms:
  * global fusion  : exact DP over the waypoints of ALL personal bests
                     (a layered random roadmap built from the swarm),
                     at initialisation and every `g_period` iterations;
  * local fusion   : DP over elites + Gaussian perturbations of gbest +
                     a midpoint candidate, every `l_period` iterations,
                     with success-based step-size adaptation;
  * stagnation restart of the non-elite particles.
"""
from __future__ import annotations

import numpy as np

from algorithms import _init
from dpf import DPFusion, select_elites


def _global_fuse(prob, ev, X, f, K=None):
    el = select_elites(X, f, K or len(X))
    dp = DPFusion(n_elite=len(el), n_perturb=0, use_mid=False)
    layers = dp.candidates(prob, el, el[0], 0.0, None)
    fes, _ = DPFusion.fe_cost(prob, layers)
    if fes > ev.remaining:
        return None
    x, J = DPFusion.fuse_dp(prob, layers)
    ev.charge(fes, x, J)
    return x, J


def _local_fuse(prob, ev, rng, elites, best, sigma, M, use_mid=True):
    n = prob.n
    parts = [elites]
    if M > 0:
        pert = best + rng.standard_normal((M, prob.dim)) * sigma
        parts.append(np.clip(pert, prob.lb, prob.ub))
    if use_mid:
        u = np.concatenate([[0.0], best[:n], [0.0]])
        h = np.concatenate([[prob.hS], best[n:], [prob.hG]])
        mid = np.concatenate([0.5 * (u[:-2] + u[2:]), 0.5 * (h[:-2] + h[2:])])
        parts.append(np.clip(mid, prob.lb, prob.ub)[None])
    C = np.vstack(parts)
    layers = [np.unique(np.stack([C[:, j], C[:, n + j]], -1), axis=0) for j in range(n)]
    fes, _ = DPFusion.fe_cost(prob, layers)
    if fes > ev.remaining:
        return None
    x, J = DPFusion.fuse_dp(prob, layers)
    ev.charge(fes, x, J)
    return x, J


def dpf_pso2(prob, ev, rng, N=50, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vfrac=0.2,
             init_global=True, g_period=0, g_K=None, l_period=5, E=5, M=6, use_mid=True,
             sig0=0.05, sig_min=1e-3, sig_max=0.2, adapt=True, restart=0, chi=None):
    lb, ub = prob.lb, prob.ub
    D = len(lb)
    rngw = ub - lb
    X = _init(rng, lb, ub, N)
    V = np.zeros_like(X)
    f = ev(X)
    P, fP = X.copy(), f.copy()
    vmax = vfrac * rngw

    def inject(x, fx):
        if fx < fP.min() - 1e-9:
            k = int(np.argmax(fP))
            P[k], fP[k], X[k], V[k] = x, fx, x, 0.0
            return True
        return False

    if init_global:
        res = _global_fuse(prob, ev, P, fP, g_K)
        if res is not None:
            inject(*res)
    sig = sig0
    it = 0
    stall = 0
    last_best = fP.min()
    while not ev.exhausted:
        g = int(np.argmin(fP))
        if chi is None:
            w = w_max - (w_max - w_min) * ev.frac
            V = w * V + c1 * rng.random((N, D)) * (P - X) + c2 * rng.random((N, D)) * (P[g] - X)
        else:
            V = chi * (V + c1 * rng.random((N, D)) * (P - X) + c2 * rng.random((N, D)) * (P[g] - X))
        np.clip(V, -vmax, vmax, out=V)
        X = X + V
        out = (X < lb) | (X > ub)
        X = np.clip(X, lb, ub)
        V[out] = 0.0
        f = ev(X)
        imp = f < fP
        P[imp], fP[imp] = X[imp], f[imp]
        it += 1
        if ev.exhausted:
            break
        if g_period and it % g_period == 0:
            res = _global_fuse(prob, ev, P, fP, g_K)
            if res is not None:
                inject(*res)
        if l_period and it % l_period == 0 and not ev.exhausted:
            g = int(np.argmin(fP))
            el = select_elites(P, fP, E)
            res = _local_fuse(prob, ev, rng, el, P[g].copy(), sig * rngw, M, use_mid)
            if res is not None:
                ok = inject(*res)
                if adapt:
                    sig = min(sig_max, sig * 1.5) if ok else max(sig_min, sig * 0.75)
                else:
                    sig = sig0 * (sig_min / sig0) ** ev.frac
        if restart:
            if fP.min() < last_best - 1e-9:
                last_best, stall = fP.min(), 0
            else:
                stall += 1
            if stall >= restart:
                keep = np.argsort(fP)[:max(1, N // 10)]
                mask = np.ones(N, bool)
                mask[keep] = False
                k = int(mask.sum())
                Xr = _init(rng, lb, ub, k)
                fr = ev(Xr)
                X[mask], V[mask], P[mask], fP[mask] = Xr, 0.0, Xr, fr
                stall = 0
                res = _global_fuse(prob, ev, P, fP, g_K)
                if res is not None:
                    inject(*res)
    return ev


def dp_only(prob, ev, rng, N=50, E=1, M=10, sig0=0.05, sig_min=1e-3, sig_max=0.2):
    """Reference: global DP over a random roadmap, then iterated local DP."""
    lb, ub = prob.lb, prob.ub
    X = _init(rng, lb, ub, N)
    f = ev(X)
    best, fb = X[np.argmin(f)].copy(), f.min()
    res = _global_fuse(prob, ev, X, f)
    if res is not None and res[1] < fb:
        best, fb = res
    sig = sig0
    rngw = ub - lb
    while not ev.exhausted:
        res = _local_fuse(prob, ev, rng, best[None], best, sig * rngw, M, True)
        if res is None:
            break
        if res[1] < fb - 1e-9:
            best, fb = res
            sig = min(sig_max, sig * 1.5)
        else:
            sig = max(sig_min, sig * 0.75)
    return ev


VARIANTS = {
    "V0-local": dict(init_global=False, adapt=False),
    "V1-init": dict(init_global=True, adapt=False),
    "V2-init+adapt": dict(init_global=True, adapt=True),
    "V3-init+adapt+glob50": dict(init_global=True, adapt=True, g_period=50),
    "V4-init+adapt+restart": dict(init_global=True, adapt=True, restart=60),
    "V5-init+adapt+chi": dict(init_global=True, adapt=True, chi=0.7298, c1=1.49618, c2=1.49618),
    "V6-init+adapt+M12": dict(init_global=True, adapt=True, M=12, l_period=10),
}
