"""Dynamic-programming path fusion (DPF): optimal recombination over candidate sets.

Because the cost J of a path S, P1..Pn, G is a sum of node terms A(P_j),
edge terms E(P_j, P_j+1) and turn terms T(P_j-1, P_j, P_j+1), the best path
whose j-th waypoint is chosen from a finite candidate set C_j can be found
exactly by dynamic programming over pairs of consecutive layers:

    W_1(S, r)  = E(S, r) + A(r)
    W_j(q, r)  = E(q, r) + A(r) + min_p [ W_j-1(p, q) + T(p, q, r) ]
    J*         = min_q W_n+1(q, G)

Complexity: sum_j |C_j||C_j+1| segment evaluations and
sum_j |C_j-1||C_j||C_j+1| turn evaluations.  If every parent path is contained
in the candidate sets, J* <= min(parent costs).

Fusion modes:
  * global fusion: candidates = waypoints of the g_max best distinct members
    of the population memory plus g_random fresh uniform random solutions;
  * local fusion : candidates = waypoints of the n_elite best members plus
    n_perturb Gaussian perturbations of the best member (success-based step).
The schedule is defined in host function evaluations (host FEs, i.e. FEs not
charged to fusion, counted after the initial population): a local fusion
every l_every host FEs and a global fusion every g_every host FEs.
"""
from __future__ import annotations

import math

import numpy as np


def select_elites(X, f, k):
    """The k best distinct rows of X (by f); k=None returns all distinct rows."""
    order = np.argsort(f, kind="stable")
    out, seen = [], set()
    for i in order:
        if not np.isfinite(f[i]):
            break
        key = X[i].tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(X[i])
        if k is not None and len(out) == k:
            break
    return np.array(out)


def layers_from(prob, C):
    """Per-waypoint candidate sets: rows of C split into the variables of
    waypoint j (u, h[, tau]), duplicate waypoints removed."""
    return [np.unique(C[:, idx], axis=0) for idx in prob.var_idx]


def assemble(prob, layers, choice):
    x = np.empty(prob.dim)
    for j, idx in enumerate(prob.var_idx):
        x[idx] = layers[j][choice[j]]
    return x


def fe_cost(prob, layers):
    """Segment evaluations of exact DP, expressed in full-path FEs (n+1 segments)."""
    K = [1] + [len(L) for L in layers] + [1]
    nseg = sum(K[j] * K[j + 1] for j in range(len(K) - 1))
    return int(math.ceil(nseg / (prob.n + 1))), nseg


def _dp_tables(prob, layers):
    n = prob.n
    pts = [prob.S[None]] + [prob.layer_points(j, L) for j, L in enumerate(layers)] + [prob.G[None]]
    node = [np.zeros(1)] + [prob.node_cost(L[:, 1]) for L in layers] + [np.zeros(1)]
    E, ANG = [], []
    for j in range(n + 1):
        A, B = pts[j][:, None, :], pts[j + 1][None, :, :]
        E.append(prob.edge_cost(A, B))                   # (K_j, K_j+1)
        ANG.append(prob.seg_angles(A, B))
    return node, E, ANG


def _step_dense(prob, W, A1, A2):
    """min_p [W(p,q) + T(p,q,r)] over the full (Kp, Kq, Kr) tensor."""
    a1 = tuple(a[:, :, None] for a in A1)                # edges p->q, (Kp, Kq, 1)
    a2 = tuple(a[None, :, :] for a in A2)                # edges q->r, (1, Kq, Kr)
    tot = W[:, :, None] + prob.turn_cost(a1, a2)         # (Kp, Kq, Kr)
    arg = np.argmin(tot, axis=0)                         # (Kq, Kr)
    return np.take_along_axis(tot, arg[None], axis=0)[0], arg


def _step_pruned(prob, W, A1, A2, t_max):
    """Same result as _step_dense.  Because 0 <= T <= t_max, a predecessor p
    with W(p,q) > min_p' W(p',q) + t_max can never be optimal for any r, so
    only the remaining (p, q) pairs are evaluated."""
    Wmin = W.min(axis=0)
    keep = W <= (Wmin + t_max + 1e-6 * (1.0 + np.abs(Wmin)))[None, :]
    qq, pp = np.nonzero(keep.T)                          # sorted by q, then p
    npair = len(qq)
    starts = np.flatnonzero(np.r_[True, qq[1:] != qq[:-1]])
    a1 = tuple(a[pp, qq][:, None] for a in A1)           # (npair, 1)
    a2 = tuple(a[qq, :] for a in A2)                     # (npair, Kr)
    tot = W[pp, qq][:, None] + prob.turn_cost(a1, a2)    # (npair, Kr)
    gmin = np.minimum.reduceat(tot, starts, axis=0)      # (Kq, Kr)
    counts = np.diff(np.r_[starts, npair])
    pos = np.where(tot == np.repeat(gmin, counts, axis=0), np.arange(npair)[:, None], npair)
    first = np.minimum.reduceat(pos, starts, axis=0)     # first (lowest p) minimiser
    return gmin, pp[first]


def fuse_dp(prob, layers, prune="auto"):
    """Exact DP fusion.  prune=True/False forces the pruned/dense step;
    'auto' prunes only when a candidate set is large (> 20), where the pruned
    step is faster.  Both steps return bit-identical results."""
    n = prob.n
    if prune == "auto":
        prune = max(len(L) for L in layers) > 20
    node, E, ANG = _dp_tables(prob, layers)
    t_max = prob.t_max
    W = E[0] + node[1][None, :]                          # (1, K1)
    back = []
    for j in range(2, n + 2):
        if prune:
            best, arg = _step_pruned(prob, W, ANG[j - 2], ANG[j - 1], t_max)
        else:
            best, arg = _step_dense(prob, W, ANG[j - 2], ANG[j - 1])
        W = best + E[j - 1] + node[j][None, :]
        back.append(arg)
    q = int(np.argmin(W[:, 0]))
    J = float(W[q, 0])
    choice = [0] * (n + 2)
    choice[n] = q
    for j in range(n + 1, 1, -1):
        choice[j - 2] = int(back[j - 2][choice[j - 1], choice[j]])
    return assemble(prob, layers, choice[1:n + 1]), J


def fuse_greedy(prob, layers):
    """Myopic layer-by-layer choice (ablation): same candidates, no DP."""
    prev2, prev = None, prob.S
    choice = []
    for j, L in enumerate(layers):
        P = prob.layer_points(j, L)
        c = prob.edge_cost(prev[None], P) + prob.node_cost(L[:, 1])
        if prev2 is not None:
            c = c + prob.tri_cost(prev2[None], prev[None], P)
        k = int(np.argmin(c))
        choice.append(k)
        prev2, prev = prev, P[k]
    return assemble(prob, layers, choice)


class DPFusion:
    def __init__(self, init_global=True, g_every=5000, l_every=250, g_max=50, g_random=50,
                 n_elite=5, n_perturb=6, use_mid=False, sigma0=0.05, sigma_min=1e-3,
                 sigma_max=0.2, adapt=True, mode="dp"):
        self.init_global = bool(init_global)
        self.g_every = int(g_every)        # host FEs between global fusions (0: never)
        self.l_every = int(l_every)        # host FEs between local fusions (0: never)
        self.g_max = int(g_max)
        self.g_random = int(g_random)
        self.n_elite = int(n_elite)
        self.n_perturb = int(n_perturb)
        self.use_mid = bool(use_mid)
        self.sigma0 = float(sigma0)
        self.sigma = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.adapt = bool(adapt)
        self.mode = mode
        self.calls = 0
        self.improved = 0
        self.stats = {k: 0 for k in ("n_global", "n_local", "fe_global", "fe_local",
                                     "imp_global", "imp_local")}
        self.stats["k_global"] = 0.0      # sum of mean candidate-set sizes
        self.stats["k_local"] = 0.0
        self._h0 = None
        self._kg = 0
        self._kl = 0

    # ------------------------------------------------------------- one fusion
    def _run(self, prob, ev, layers, kind):
        n = prob.n
        if self.mode == "dp":
            fes, _ = fe_cost(prob, layers)
            if fes > ev.remaining:
                return None
            x, J = fuse_dp(prob, layers)
        else:
            fes = int(math.ceil((sum(len(L) for L in layers) + 1) / (n + 1))) + 1
            if fes > ev.remaining:
                return None
            x = fuse_greedy(prob, layers)
            J = float(prob.evaluate(x[None])[0])
        ev.charge(fes, x, J)
        self.calls += 1
        self.stats[f"n_{kind}"] += 1
        self.stats[f"fe_{kind}"] += fes
        self.stats[f"k_{kind}"] += float(np.mean([len(L) for L in layers]))
        return x, J

    def global_fuse(self, prob, ev, X, f, rng=None, extra=None):
        el = select_elites(X, f, self.g_max)
        if len(el) == 0:
            return None
        parts = [el]
        if extra is not None and len(extra):
            parts.append(np.asarray(extra))
        if self.g_random > 0 and rng is not None:
            parts.append(prob.lb + rng.random((self.g_random, prob.dim)) * (prob.ub - prob.lb))
        return self._run(prob, ev, layers_from(prob, np.vstack(parts)), "global")

    def local_fuse(self, prob, ev, rng, X, f):
        el = select_elites(X, f, self.n_elite)
        if len(el) == 0:
            return None
        best = el[0]
        n = prob.n
        parts = [el]
        if self.n_perturb > 0:
            sig = self.sigma * (prob.ub - prob.lb)
            pert = best + rng.standard_normal((self.n_perturb, prob.dim)) * sig
            parts.append(np.clip(pert, prob.lb, prob.ub))
        if self.use_mid:
            ends = {0: (0.0, 0.0), 1: (prob.hS, prob.hG), 2: (0.0, 0.0)}
            mid = np.empty(prob.dim)
            for k in range(prob.nv):
                v = np.concatenate([[ends[k][0]], best[k * n:(k + 1) * n], [ends[k][1]]])
                mid[k * n:(k + 1) * n] = 0.5 * (v[:-2] + v[2:])
            parts.append(np.clip(mid, prob.lb, prob.ub)[None])
        return self._run(prob, ev, layers_from(prob, np.vstack(parts)), "local")

    def adapt_sigma(self, ok, ev):
        if self.adapt:
            self.sigma = min(self.sigma_max, self.sigma * 1.5) if ok \
                else max(self.sigma_min, self.sigma * 0.75)
        else:
            self.sigma = self.sigma0 * (self.sigma_min / self.sigma0) ** ev.frac

    # ------------------------------------------------------------- host hook
    def hook(self, prob, ev, rng, X, f, it, extra=None):
        """Called by a host optimiser with its population memory (X, f) after
        its initial population (it == 0) and after every iteration.  A fused
        path that improves on the best member replaces the worst member of
        (X, f) in place.  Returns the replaced row indices."""
        replaced = []
        if ev.exhausted:
            return replaced
        if it == 0 or self._h0 is None:
            self._h0 = ev.used - ev.dp_fes
            if self.init_global:
                self._accept(X, f, self.global_fuse(prob, ev, X, f, rng, extra), replaced, "global")
            return replaced
        h = ev.used - ev.dp_fes - self._h0
        if self.g_every and h // self.g_every > self._kg:
            self._kg = h // self.g_every
            self._accept(X, f, self.global_fuse(prob, ev, X, f, rng, extra), replaced, "global")
        if self.l_every and h // self.l_every > self._kl and not ev.exhausted:
            self._kl = h // self.l_every
            ok = self._accept(X, f, self.local_fuse(prob, ev, rng, X, f), replaced, "local")
            self.adapt_sigma(ok, ev)
        return replaced

    def _accept(self, X, f, res, replaced, kind):
        if res is None:
            return False
        x, J = res
        if J < np.min(f) - 1e-9:
            self.improved += 1
            self.stats[f"imp_{kind}"] += 1
            k = int(np.argmax(f))
            X[k] = x
            f[k] = J
            replaced.append(k)
            return True
        return False
