"""UAV 3-D path-planning environment with a decomposable cost model.

The cost model follows Phung & Ha, "Safety-enhanced UAV path planning with
spherical vector-based particle swarm optimization", Applied Soft Computing
107 (2021) 107376, Eqs. (1)-(9), with two documented modifications:

1. Infeasible elements (a segment entering an inflated threat cylinder, a
   segment whose ground clearance drops below ``h_clear``) receive a finite
   additive penalty ``M0 + lam * violation`` instead of +inf.  The total cost
   is therefore a sum of local terms defined on single waypoints, single
   segments and pairs of consecutive segments.
2. Ground clearance is enforced along the whole segment and computed EXACTLY
   for the bilinear terrain (the original model checks only the waypoints).

Exact clearance.  Inside one grid cell the bilinear terrain restricted to a
straight segment is a quadratic function of the segment parameter, and the
segment altitude is linear, so the minimum clearance in every cell crossed by
the segment has a closed form.  To keep this cheap, each segment is split into
short pieces; a piece is skipped when a conservative lower bound (lowest
altitude of the piece minus the maximum terrain height in a window that
contains the piece) already exceeds the relevant threshold, and evaluated
exactly cell by cell otherwise.  The returned violation is exact (up to
floating-point rounding) and does not depend on how segments are batched.

Coordinates: x, y in metres, z is absolute altitude.  A waypoint is stored as
(x, y, z_abs) together with its height above ground h.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, zoom

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BAND_TOL = 1e-6      # tolerance (m) of the along-segment altitude band (variant only)


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CostParams:
    b1: float = 5.0        # weight of path length F1           (Phung & Ha code)
    b2: float = 1.0        # weight of threat cost F2
    b3: float = 10.0       # weight of altitude cost F3
    b4: float = 1.0        # weight of smoothness cost F4
    a1: float = 1.0        # turning-angle coefficient (degrees)
    a2: float = 1.0        # climb-angle-change coefficient (degrees)
    drone: float = 1.0     # UAV size D (m)
    danger: float = 10.0   # danger distance S (m)
    h_min: float = 100.0   # altitude band above ground (m)
    h_max: float = 200.0
    h_clear: float = 20.0  # minimum ground clearance along segments (m), exact
    M0: float = 1e5        # fixed penalty per violated element
    lam: float = 1e3       # penalty per metre of violation
    band_along: bool = False   # sensitivity variant: enforce the altitude band
                               # [h_min, h_max] along every segment (exact), not
                               # only at the waypoints


# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------
class Terrain:
    """Elevation grid Z[iy, ix] with spacing ``res`` metres, bilinear lookup and
    exact segment ground-clearance computation."""

    DIL_LEVELS = (1, 2, 4, 8)      # window half-widths (cells) of the max maps
    PIECE_CELLS = 4.0              # maximum piece length in cells

    def __init__(self, Z: np.ndarray, res: float):
        self.Z = np.ascontiguousarray(Z, dtype=np.float64)
        self.res = float(res)
        self.ny, self.nx = self.Z.shape
        self.size_x = (self.nx - 1) * self.res
        self.size_y = (self.ny - 1) * self.res
        self._flat = self.Z.ravel()
        Zc = self.Z
        cm = np.maximum(np.maximum(Zc[:-1, :-1], Zc[:-1, 1:]), np.maximum(Zc[1:, :-1], Zc[1:, 1:]))
        # max over cells whose index differs by at most r in each direction;
        # float32 storage plus 1 mm guard keeps the bound conservative
        self.dil_r = np.array(self.DIL_LEVELS)
        self.dil = np.stack([maximum_filter(cm, size=2 * r + 1, mode="nearest")
                             for r in self.DIL_LEVELS]).astype(np.float32) + np.float32(1e-3)

    def height(self, x, y):
        gx = np.clip(np.asarray(x, dtype=np.float64) / self.res, 0.0, self.nx - 1.000001)
        gy = np.clip(np.asarray(y, dtype=np.float64) / self.res, 0.0, self.ny - 1.000001)
        i0 = gx.astype(np.intp)
        j0 = gy.astype(np.intp)
        fx = gx - i0
        fy = gy - j0
        k = j0 * self.nx + i0
        Zf = self._flat
        z00 = Zf[k]
        z01 = Zf[k + 1]
        z10 = Zf[k + self.nx]
        z11 = Zf[k + self.nx + 1]
        return (z00 * (1.0 - fx) + z01 * fx) * (1.0 - fy) + (z10 * (1.0 - fx) + z11 * fx) * fy

    # ------------------------------------------------------- exact clearance
    def _piece_min(self, gxa, gya, za, gxb, gyb, zb, chunk=40000):
        """Exact minimum of (altitude - terrain) along pieces a->b given in grid
        coordinates (gx = x/res, gy = y/res); 1-D arrays of equal length."""
        out = np.empty(len(gxa))
        for s in range(0, len(gxa), chunk):
            sl = slice(s, s + chunk)
            out[sl] = self._piece_min_chunk(gxa[sl], gya[sl], za[sl], gxb[sl], gyb[sl], zb[sl])
        return out

    def _piece_min_chunk(self, gxa, gya, za, gxb, gyb, zb):
        nx, ny = self.nx, self.ny
        dgx, dgy, dz = gxb - gxa, gyb - gya, zb - za
        parts = [np.zeros((len(gxa), 1))]
        for ga, gb, dg in ((gxa, gxb, dgx), (gya, gyb, dgy)):
            lo, hi = np.minimum(ga, gb), np.maximum(ga, gb)
            cnt = np.maximum(np.ceil(hi) - np.floor(lo) - 1.0, 0.0).astype(np.intp)
            cmax = int(cnt.max()) if len(cnt) else 0
            if cmax > 0:
                kk = (np.floor(lo) + 1.0)[:, None] + np.arange(cmax)[None, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (kk - ga[:, None]) / dg[:, None]
                parts.append(np.where(np.arange(cmax)[None, :] < cnt[:, None], s, 1.0))
        parts.append(np.ones((len(gxa), 1)))
        S = np.sort(np.clip(np.concatenate(parts, axis=1), 0.0, 1.0), axis=1)
        s0, s1 = S[:, :-1], S[:, 1:]
        sm = 0.5 * (s0 + s1)
        i = np.clip(np.floor(gxa[:, None] + sm * dgx[:, None]), 0, nx - 2).astype(np.intp)
        j = np.clip(np.floor(gya[:, None] + sm * dgy[:, None]), 0, ny - 2).astype(np.intp)
        k = j * nx + i
        Zf = self._flat
        z00, z01, z10, z11 = Zf[k], Zf[k + 1], Zf[k + nx], Zf[k + nx + 1]
        a, b, g = z01 - z00, z10 - z00, z00 - z01 - z10 + z11
        p0, p1 = gxa[:, None] - i, dgx[:, None]
        q0, q1 = gya[:, None] - j, dgy[:, None]
        C0 = z00 + a * p0 + b * q0 + g * p0 * q0          # terrain(s) = C0 + C1 s + C2 s^2
        C1 = a * p1 + b * q1 + g * (p0 * q1 + p1 * q0)
        C2 = g * p1 * q1
        e0, e1 = za[:, None] - C0, dz[:, None] - C1         # clearance(s) = e0 + e1 s - C2 s^2
        cmin = np.minimum(e0 + e1 * s0 - C2 * s0 * s0, e0 + e1 * s1 - C2 * s1 * s1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sv = e1 / (2.0 * C2)
            inside = (C2 < 0.0) & (sv > s0) & (sv < s1)
            cv = e0 + e1 * sv - C2 * sv * sv
        cmin = np.where(inside, np.minimum(cmin, cv), cmin)
        return cmin.min(axis=1)

    def ground_violation(self, A, B, h_clear, min_pieces=21, max_points=1_500_000, tau=None):
        """Exact v = max(0, h_clear - min clearance) for segments A->B ((M, 3)).

        With ``tau=np.inf`` the exact minimum clearance of every segment is
        returned instead (used for reporting)."""
        A = np.asarray(A, float).reshape(-1, 3)
        B = np.asarray(B, float).reshape(-1, 3)
        M = len(A)
        out = np.zeros(M) if tau is None else np.empty(M)
        if M == 0:
            return out
        D = B - A
        hl = np.hypot(D[:, 0], D[:, 1])
        unit = self.PIECE_CELLS * self.res

        def pieces(length):
            return int(min(400, max(min_pieces, np.ceil(length / unit))))

        if M <= 2048:
            return self._ground_chunk(A, D, pieces(hl.max()), h_clear, tau)
        # long batches (DP edge tables): process segments in groups of similar
        # length; the number of pieces only affects the cost, not the result
        order = np.argsort(hl, kind="stable")
        hl_s = hl[order]
        pos = 0
        while pos < M:
            m = pieces(hl_s[min(M - 1, pos + 1023)])
            end = min(M, pos + max(1024, max_points // (m + 1)))
            end = pos + int(np.searchsorted(hl_s[pos:end], m * unit, side="right"))
            idx = order[pos:end]
            out[idx] = self._ground_chunk(A[idx], D[idx], m, h_clear, tau)
            pos = end
        return out

    def _ground_chunk(self, a, d, m, h_clear, tau):
        """Segments a -> a+d, each split into m pieces of at most PIECE_CELLS cells."""
        t = np.linspace(0.0, 1.0, m + 1)
        r, nx, ny, L = self.res, self.nx, self.ny, len(self.dil_r)
        X = a[:, 0:1] + t * d[:, 0:1]
        Y = a[:, 1:2] + t * d[:, 1:2]
        Zl = a[:, 2:3] + t * d[:, 2:3]
        cs = (Zl - self.height(X, Y)).min(axis=1)
        thr = np.minimum(h_clear, cs) if tau is None else cs
        gx, gy = X / r, Y / r
        gxa, gxb, gya, gyb = gx[:, :-1], gx[:, 1:], gy[:, :-1], gy[:, 1:]
        lo_i = np.clip(np.floor(np.minimum(gxa, gxb)), 0, nx - 2)
        hi_i = np.clip(np.floor(np.maximum(gxa, gxb)), 0, nx - 2)
        lo_j = np.clip(np.floor(np.minimum(gya, gyb)), 0, ny - 2)
        hi_j = np.clip(np.floor(np.maximum(gya, gyb)), 0, ny - 2)
        mi = np.clip(np.floor(0.5 * (gxa + gxb)), 0, nx - 2)
        mj = np.clip(np.floor(0.5 * (gya + gyb)), 0, ny - 2)
        rneed = np.maximum(np.maximum(hi_i - mi, mi - lo_i), np.maximum(hi_j - mj, mj - lo_j))
        lev = np.searchsorted(self.dil_r, rneed, side="left")
        Dm = self.dil[np.minimum(lev, L - 1), mj.astype(np.intp), mi.astype(np.intp)]
        lb = np.minimum(Zl[:, :-1], Zl[:, 1:]) - np.where(lev >= L, np.inf, Dm)
        rows, cols = np.nonzero(lb < thr[:, None])
        cmin = cs.copy()
        if len(rows):
            ex = self._piece_min(gxa[rows, cols], gya[rows, cols], Zl[rows, cols],
                                 gxb[rows, cols], gyb[rows, cols], Zl[rows, cols + 1])
            starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])   # rows are sorted
            ur = rows[starts]
            cmin[ur] = np.minimum(cmin[ur], np.minimum.reduceat(ex, starts))
        if tau is None:
            return np.maximum(h_clear - cmin, 0.0)
        return cmin

    def min_clearance(self, A, B):
        """Exact minimum clearance of each segment A->B."""
        return self.ground_violation(A, B, 0.0, tau=np.inf)

    # ------------------------------------------- exact maximum height above ground
    @property
    def mirror(self):
        """Terrain with negated elevations.  For a segment with negated
        altitudes, its clearance is -(height above the true terrain), so the
        exact minimum clearance machinery gives the exact maximum height."""
        if getattr(self, "_mirror", None) is None:
            self._mirror = Terrain(-self.Z, self.res)
        return self._mirror

    @staticmethod
    def _negz(A):
        A = np.array(A, float).reshape(-1, 3)
        A[:, 2] = -A[:, 2]
        return A

    def ceiling_violation(self, A, B, h_max):
        """Exact v = max(0, max height above ground - h_max) for segments A->B."""
        return self.mirror.ground_violation(self._negz(A), self._negz(B), -float(h_max))

    def max_height(self, A, B):
        """Exact maximum height above ground of each segment A->B."""
        return -self.mirror.min_clearance(self._negz(A), self._negz(B))


# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    title: str
    terrain: Terrain
    threats: np.ndarray   # (K, 3): centre x, centre y, radius
    start: np.ndarray     # (3,): x, y, height above ground
    goal: np.ndarray      # (3,)


def gaussian_terrain(size, res, peaks, base=0.0, noise_amp=0.0, noise_scale=80.0, seed=0):
    """Sum of anisotropic Gaussian peaks plus optional smooth random relief."""
    g = np.arange(0.0, size + 0.5 * res, res)
    X, Y = np.meshgrid(g, g)
    Z = np.full_like(X, float(base))
    for px, py, h, sx, sy in peaks:
        Z += h * np.exp(-((X - px) / sx) ** 2 - ((Y - py) / sy) ** 2)
    if noise_amp > 0:
        rng = np.random.default_rng(seed)
        nc = int(np.ceil(size / noise_scale)) + 4
        coarse = rng.standard_normal((nc, nc))
        fine = zoom(coarse, len(g) / nc * 1.0001, order=3)[: len(g), : len(g)]
        fine = fine / np.max(np.abs(fine))
        Z += noise_amp * fine
    return np.maximum(Z, 0.0)


def _random_threats(rng, k, rmin, rmax, lo, hi, avoid, avoid_dist, min_gap, strict=True):
    out = []
    tries = 0
    while len(out) < k:
        tries += 1
        if tries > 200000:
            if strict:
                raise RuntimeError("threat sampling failed")
            break
        c = rng.uniform(lo, hi, 2)
        r = rng.uniform(rmin, rmax)
        if any(np.hypot(*(c - a)) < avoid_dist + r for a in avoid):
            continue
        if any(np.hypot(c[0] - t[0], c[1] - t[1]) < r + t[2] + min_gap for t in out):
            continue
        out.append((c[0], c[1], r))
    return np.array(out)


def build_scenarios(size=1000.0, res=5.0):
    """Design scenarios S1-S4 and the first two held-out scenarios S5, S6
    (unchanged from the first study)."""
    sc = {}

    peaks = [(250, 600, 120, 120, 140), (600, 250, 140, 130, 110), (500, 500, 100, 150, 150),
             (780, 520, 150, 110, 120), (380, 850, 110, 130, 100), (850, 820, 90, 100, 100)]
    T = Terrain(gaussian_terrain(size, res, peaks, base=20.0), res)
    thr = np.array([(310, 290, 70), (520, 500, 85), (700, 740, 70), (430, 690, 60)], float)
    sc["S1"] = Scenario("S1", "Rolling hills, 4 threats", T, thr,
                        np.array([100.0, 100.0, 150.0]), np.array([900.0, 900.0, 150.0]))

    peaks = [(200, 450, 260, 110, 130), (450, 200, 240, 120, 110), (480, 520, 300, 100, 110),
             (700, 350, 280, 120, 100), (350, 760, 250, 110, 120), (650, 680, 320, 100, 120),
             (850, 600, 220, 90, 120), (600, 880, 200, 120, 90), (880, 300, 180, 100, 100)]
    T = Terrain(gaussian_terrain(size, res, peaks, base=30.0), res)
    thr = np.array([(260, 300, 60), (400, 400, 75), (560, 560, 70), (470, 690, 65),
                    (720, 600, 75), (760, 820, 60)], float)
    sc["S2"] = Scenario("S2", "Mountains, 6 threats", T, thr,
                        np.array([100.0, 100.0, 150.0]), np.array([900.0, 900.0, 150.0]))

    peaks = [(200, 200, 200, 140, 140), (200, 800, 220, 140, 140), (500, 500, 160, 120, 200),
             (800, 200, 240, 140, 140), (800, 800, 200, 140, 140), (650, 450, 180, 90, 90),
             (350, 550, 150, 90, 90)]
    T = Terrain(gaussian_terrain(size, res, peaks, base=25.0), res)
    thr = np.array([(280, 320, 70), (280, 500, 80), (280, 690, 75),
                    (520, 200, 60), (520, 400, 75), (520, 610, 70), (520, 820, 65),
                    (760, 330, 70), (760, 520, 80), (760, 720, 75)], float)
    sc["S3"] = Scenario("S3", "Threat maze, 10 threats", T, thr,
                        np.array([50.0, 500.0, 150.0]), np.array([950.0, 500.0, 150.0]))

    rng = np.random.default_rng(2027)
    peaks = [(rng.uniform(50, 950), rng.uniform(50, 950), rng.uniform(150, 380),
              rng.uniform(70, 150), rng.uniform(70, 150)) for _ in range(14)]
    T = Terrain(gaussian_terrain(size, res, peaks, base=30.0, noise_amp=20.0, noise_scale=80.0, seed=7), res)
    start = np.array([100.0, 900.0, 150.0])
    goal = np.array([900.0, 100.0, 150.0])
    thr = _random_threats(rng, 12, 45.0, 75.0, 150.0, 850.0, [start[:2], goal[:2]], 150.0, 30.0)
    sc["S4"] = Scenario("S4", "Rugged terrain, 12 threats", T, thr, start, goal)

    def procedural(name, title, seed, n_peaks, h_rng, n_thr, r_rng, start, goal, noise):
        g = np.random.default_rng(seed)
        pk = [(g.uniform(50, 950), g.uniform(50, 950), g.uniform(*h_rng),
               g.uniform(70, 160), g.uniform(70, 160)) for _ in range(n_peaks)]
        Tg = Terrain(gaussian_terrain(size, res, pk, base=30.0, noise_amp=noise, noise_scale=90.0,
                                      seed=seed + 1), res)
        th = _random_threats(g, n_thr, r_rng[0], r_rng[1], 120.0, 880.0, [start[:2], goal[:2]], 120.0, 30.0)
        return Scenario(name, title, Tg, th, start, goal)

    sc["S5"] = procedural("S5", "Random mountains, 14 threats", 5150, 16, (120, 360), 14,
                          (40, 70), np.array([80.0, 300.0, 150.0]), np.array([920.0, 700.0, 150.0]), 15.0)
    sc["S6"] = procedural("S6", "Dense threats, 18 threats", 6160, 10, (100, 300), 18,
                          (35, 65), np.array([500.0, 60.0, 150.0]), np.array([500.0, 940.0, 150.0]), 25.0)
    return sc


HELDOUT_SEED = 90210          # fresh seed family, fixed after the design was frozen
N_HELDOUT = 40


def heldout_scenario(k, size=1000.0, res=5.0):
    """Held-out scenario H01..H40, generated from seed HELDOUT_SEED + k."""
    g = np.random.default_rng(HELDOUT_SEED + k)
    n_peaks = int(g.integers(6, 19))
    pk = [(g.uniform(50, 950), g.uniform(50, 950), g.uniform(100, 400),
           g.uniform(60, 160), g.uniform(60, 160)) for _ in range(n_peaks)]
    T = Terrain(gaussian_terrain(size, res, pk, base=g.uniform(20, 40), noise_amp=g.uniform(0, 25),
                                 noise_scale=90.0, seed=HELDOUT_SEED + 1000 + k), res)
    ang = g.uniform(0, 2 * np.pi)
    half = g.uniform(380.0, 430.0)
    d = np.array([np.cos(ang), np.sin(ang)])
    s = np.clip(500.0 - half * d, 60.0, 940.0)
    e = np.clip(500.0 + half * d, 60.0, 940.0)
    n_thr = int(g.integers(6, 19))
    thr = _random_threats(g, n_thr, 35.0, 80.0, 100.0, 900.0, [s, e], 120.0, 30.0, strict=False)
    return Scenario(f"H{k:02d}", f"Held-out {k:02d}: {n_peaks} peaks, {len(thr)} threats", T, thr,
                    np.array([s[0], s[1], 150.0]), np.array([e[0], e[1], 150.0]))


def public_scenario():
    """Scenario of the public SPSO code (Phung & Ha [5]): Christmas Island DEM,
    six threats, start (200, 100) and goal (800, 800) in 1-based pixel units of
    the reference code (1 pixel treated as 1 m, as in that code)."""
    from PIL import Image
    H = np.array(Image.open(os.path.join(DATA, "ChrismasTerrain.tif")), dtype=np.float64)
    H[H < 0] = 0.0
    T = Terrain(H, 1.0)
    thr = np.array([(400, 500, 80), (600, 200, 70), (500, 350, 80), (350, 200, 70),
                    (700, 550, 70), (650, 750, 80)], float)
    thr[:, :2] -= 1.0                       # 1-based (MATLAB) -> 0-based coordinates
    return Scenario("C1", "Christmas Island DEM [5], 6 threats", T, thr,
                    np.array([199.0, 99.0, 150.0]), np.array([799.0, 799.0, 150.0]))


_CACHE = {}


def get_scenario(name):
    """S1-S6, H01-H40 or C1 (cached per process)."""
    if name not in _CACHE:
        if name.startswith("S"):
            _CACHE.update(build_scenarios())
        elif name.startswith("H"):
            _CACHE[name] = heldout_scenario(int(name[1:]))
        elif name == "C1":
            _CACHE[name] = public_scenario()
        else:
            raise KeyError(name)
    return _CACHE[name]


def parse_instance(inst):
    name, n = inst.split("_n")
    return name, int(n)


# ----------------------------------------------------------------------------
# Path-planning problem
# ----------------------------------------------------------------------------
class PathProblem:
    """Minimise the cost of a path S -> P1 ... Pn -> G.

    Slot encoding (all generic optimisers and DPF): waypoint j lies in slot j
    of the horizontal S-G line, x = [u_1..u_n, h_1..h_n, tau_1..tau_n] with
    lateral offset u, height above ground h and along-track offset tau.

    Spherical encoding (SPSO baseline): x = [r_1..r_n, psi_1..psi_n,
    phi_1..phi_n] as in Phung & Ha's reference code.
    """

    def __init__(self, sc: Scenario, n: int, params: CostParams = CostParams(),
                 u_cap: float = 400.0, margin: float = 5.0, enc: str = "slot",
                 tau_max: float = 0.45):
        self.sc, self.n, self.p = sc, int(n), params
        self.enc = enc
        self.nv = 3 if enc == "slot" else 2      # variables per waypoint
        self.tau_max = float(tau_max)
        T = sc.terrain
        self.terrain = T
        self.xmin = self.ymin = margin
        self.xmax = T.size_x - margin
        self.ymax = T.size_y - margin
        sxy = np.asarray(sc.start[:2], float)
        gxy = np.asarray(sc.goal[:2], float)
        self.hS, self.hG = float(sc.start[2]), float(sc.goal[2])
        self.S = np.array([sxy[0], sxy[1], float(T.height(sxy[0], sxy[1])) + self.hS])
        self.G = np.array([gxy[0], gxy[1], float(T.height(gxy[0], gxy[1])) + self.hG])
        d = gxy - sxy
        self.L = float(np.hypot(d[0], d[1]))
        self.e1 = d / self.L
        self.e2 = np.array([-self.e1[1], self.e1[0]])
        s = np.arange(1, n + 1) * self.L / (n + 1)
        self.base = sxy + s[:, None] * self.e1           # (n, 2)
        lo = np.full(n, -np.inf)
        hi = np.full(n, np.inf)
        for c, (cmin, cmax) in enumerate(((self.xmin, self.xmax), (self.ymin, self.ymax))):
            e = self.e2[c]
            if abs(e) > 1e-12:
                a = (cmin - self.base[:, c]) / e
                b = (cmax - self.base[:, c]) / e
                lo = np.maximum(lo, np.minimum(a, b))
                hi = np.minimum(hi, np.maximum(a, b))
        self.u_lo = np.maximum(lo, -u_cap)
        self.u_hi = np.minimum(hi, u_cap)
        self.ds = self.L / (n + 1)               # nominal along-track spacing
        lbs = [self.u_lo, np.full(n, params.h_min)]
        ubs = [self.u_hi, np.full(n, params.h_max)]
        if self.nv == 3:                          # along-track offset tau in slot units
            lbs.append(np.full(n, -self.tau_max))
            ubs.append(np.full(n, self.tau_max))
        self.lb = np.concatenate(lbs)
        self.ub = np.concatenate(ubs)
        self.dim = self.nv * n
        self.var_idx = [np.array([j + k * n for k in range(self.nv)]) for j in range(n)]
        th = np.asarray(sc.threats, float).reshape(-1, 3)
        self.tcx, self.tcy, self.tR = th[:, 0].copy(), th[:, 1].copy(), th[:, 2].copy()
        # spherical bounds (SPSO reference code: r in [0, 2|SG|/n], angles +-pi/4)
        phi0 = float(np.arctan2(self.e1[1], self.e1[0]))
        ar = np.pi / 4
        self.sph_lb = np.concatenate([np.zeros(n), np.full(n, -ar), np.full(n, phi0 - ar)])
        self.sph_ub = np.concatenate([np.full(n, 2.0 * self.L / n), np.full(n, ar), np.full(n, phi0 + ar)])

    # ------------------------------------------------------------------ decode
    def layer_points(self, j, C):
        """Absolute 3-D points of layer j (0-based) for candidates C (K, nv):
        columns u (lateral offset), h (height above ground)[, tau]."""
        C = np.asarray(C, float)
        bx, by = self.base[j, 0], self.base[j, 1]
        if self.nv == 3:
            bx = bx + C[:, 2] * self.ds * self.e1[0]
            by = by + C[:, 2] * self.ds * self.e1[1]
        x = np.clip(bx + C[:, 0] * self.e2[0], self.xmin, self.xmax)
        y = np.clip(by + C[:, 0] * self.e2[1], self.ymin, self.ymax)
        z = self.terrain.height(x, y) + C[:, 1]
        return np.stack([x, y, z], axis=-1)

    def plane_to_points(self, X):
        """Decode slot solutions X (..., nv*n) into waypoints (..., n, 3)."""
        X = np.asarray(X, float)
        n = self.n
        u = X[..., :n]
        h = X[..., n:2 * n]
        bx, by = self.base[:, 0], self.base[:, 1]
        if self.nv == 3:
            off = X[..., 2 * n:3 * n] * self.ds
            bx = bx + off * self.e1[0]
            by = by + off * self.e1[1]
        x = np.clip(bx + u * self.e2[0], self.xmin, self.xmax)
        y = np.clip(by + u * self.e2[1], self.ymin, self.ymax)
        z = self.terrain.height(x, y) + h
        return np.stack([x, y, z], axis=-1), h

    def sph_to_points(self, Om):
        Om = np.asarray(Om, float)
        n = self.n
        r, psi, phi = Om[..., :n], Om[..., n:2 * n], Om[..., 2 * n:]
        dx = r * np.cos(psi) * np.cos(phi)
        dy = r * np.cos(psi) * np.sin(phi)
        dh = r * np.sin(psi)
        shp = r.shape[:-1]
        cx = np.full(shp, self.S[0])
        cy = np.full(shp, self.S[1])
        ch = np.full(shp, self.hS)
        xs = np.empty_like(r)
        ys = np.empty_like(r)
        hs = np.empty_like(r)
        for j in range(n):   # sequential clipping exactly as in the reference code
            cx = np.clip(cx + dx[..., j], self.xmin, self.xmax)
            cy = np.clip(cy + dy[..., j], self.ymin, self.ymax)
            ch = np.clip(ch + dh[..., j], self.p.h_min, self.p.h_max)
            xs[..., j], ys[..., j], hs[..., j] = cx, cy, ch
        z = self.terrain.height(xs, ys) + hs
        return np.stack([xs, ys, z], axis=-1), hs

    # --------------------------------------------------------- local cost terms
    def node_cost(self, h):
        """Altitude term A(P_j) (the bounds keep h inside the band; a band
        violation would be penalised like a violated segment)."""
        p = self.p
        v = np.maximum(p.h_min - h, 0.0) + np.maximum(h - p.h_max, 0.0)
        return p.b3 * np.abs(h - 0.5 * (p.h_min + p.h_max)) + (v > 0) * p.M0 + p.lam * v

    def threat_terms(self, A, B):
        """Danger cost, penetration depths and min margin to R+D per segment."""
        p = self.p
        ax, ay = A[..., 0, None], A[..., 1, None]
        dx, dy = B[..., 0, None] - ax, B[..., 1, None] - ay
        seg2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = np.clip(((self.tcx - ax) * dx + (self.tcy - ay) * dy) / seg2, 0.0, 1.0)
        dist = np.hypot(ax + t * dx - self.tcx, ay + t * dy - self.tcy)      # (..., K)
        inner = self.tR + p.drone
        thr = np.clip(inner + p.danger - dist, 0.0, p.danger).sum(axis=-1)
        depth = np.maximum(inner - dist, 0.0)
        margin = (dist - inner).min(axis=-1)
        return thr, depth, margin

    def edge_cost(self, A, B, detail=False):
        """Cost of segments A->B; A, B broadcastable arrays (..., 3)."""
        p = self.p
        A, B = np.broadcast_arrays(np.asarray(A, float), np.asarray(B, float))
        d = B - A
        length = np.sqrt(np.sum(d * d, axis=-1))
        thr, depth, margin = self.threat_terms(A, B)
        ncol = np.count_nonzero(depth > 0.0, axis=-1)
        if p.band_along:
            # altitude band along the whole segment: lower bound as a clearance
            # of h_min, upper bound as an exact ceiling of h_max.  Waypoints may
            # lie exactly on the band limits (h = h_min or h_max), where rounding
            # of order 1e-13 m would decide about a penalty of M0; violations
            # below BAND_TOL are therefore ignored (the safety check uses the
            # same tolerance).
            tv = self.terrain.ground_violation(A, B, max(p.h_clear, p.h_min)).reshape(length.shape)
            cv = self.terrain.ceiling_violation(A, B, p.h_max).reshape(length.shape)
            tv = np.where(tv > BAND_TOL, tv, 0.0)
            cv = np.where(cv > BAND_TOL, cv, 0.0)
            nviol = ncol + (tv > 0.0) + (cv > 0.0)
            tv = tv + cv
        else:
            tv = self.terrain.ground_violation(A, B, p.h_clear).reshape(length.shape)
            nviol = ncol + (tv > 0.0)
        cost = p.b1 * length + p.b2 * thr + nviol * p.M0 + p.lam * (depth.sum(axis=-1) + tv)
        if detail:
            return cost, length, thr, nviol, margin, tv
        return cost

    @staticmethod
    def seg_angles(A, B):
        """Heading theta (rad), climb angle gamma (deg) and a non-degenerate
        flag of the horizontal projection for segments A->B."""
        d = np.asarray(B, float) - np.asarray(A, float)
        hl = np.hypot(d[..., 0], d[..., 1])
        return (np.arctan2(d[..., 1], d[..., 0]), np.degrees(np.arctan2(d[..., 2], hl)), hl > 1e-9)

    def turn_cost(self, a1, a2, detail=False):
        """Smoothness cost between consecutive segments with angle tuples a1, a2.
        The turning angle |wrap(theta2 - theta1)| equals the angle between the
        horizontal projections (Phung & Ha, Eq. 6); degenerate projections give 0."""
        p = self.p
        th1, g1, ok1 = a1
        th2, g2, ok2 = a2
        turn = np.degrees(np.abs(np.mod(th2 - th1 + np.pi, 2.0 * np.pi) - np.pi)) * (ok1 & ok2)
        dclimb = np.abs(g2 - g1)
        smooth = p.b4 * (p.a1 * turn + p.a2 * dclimb)
        if detail:
            return smooth, turn, dclimb
        return smooth

    def tri_cost(self, A, B, C, detail=False):
        """Smoothness cost at the middle point B of consecutive segments A->B->C."""
        return self.turn_cost(self.seg_angles(A, B), self.seg_angles(B, C), detail)

    @property
    def t_max(self):
        """Upper bound of the turn term (each angle lies in [0, 180] degrees)."""
        p = self.p
        return p.b4 * (p.a1 + p.a2) * 180.0

    # --------------------------------------------------------------- full cost
    def full_path(self, P):
        lead = P.shape[:-2]
        S = np.broadcast_to(self.S, lead + (1, 3))
        G = np.broadcast_to(self.G, lead + (1, 3))
        return np.concatenate([S, P, G], axis=-2)

    def path_turns(self, A, B, detail=False):
        """Smoothness terms of whole paths with segments A->B (..., m, 3).

        Segments with zero horizontal length (possible only in the spherical
        encoding, e.g. r = 0 at its lower bound) are skipped and each turn is
        measured between consecutive non-degenerate segments, as in the
        reference code of [5]; otherwise a sharp turn could be hidden behind a
        collapsed waypoint.  Slot-encoded paths never contain such segments,
        so for them this equals the sum of the local turn terms used by DPF."""
        th, g, ok = self.seg_angles(A, B)
        m = th.shape[-1]
        idx = np.where(ok, np.arange(m), -1)
        last = np.maximum.accumulate(idx, axis=-1)
        prev = np.concatenate([np.full(last.shape[:-1] + (1,), -1), last[..., :-1]], axis=-1)
        has = ok & (prev >= 0)
        pc = np.maximum(prev, 0)
        th_p = np.take_along_axis(th, pc, axis=-1)
        g_p = np.take_along_axis(g, pc, axis=-1)
        p = self.p
        turn = np.degrees(np.abs(np.mod(th - th_p + np.pi, 2.0 * np.pi) - np.pi)) * has
        dclimb = np.abs(g - g_p) * has
        smooth = p.b4 * (p.a1 * turn + p.a2 * dclimb)
        if detail:
            return smooth, turn, dclimb
        return smooth

    def cost_points(self, P, h):
        F = self.full_path(P)
        A, B = F[..., :-1, :], F[..., 1:, :]
        J = self.node_cost(h).sum(axis=-1)
        J = J + self.edge_cost(A, B).sum(axis=-1)
        J = J + self.path_turns(A, B).sum(axis=-1)
        return J

    def evaluate(self, X):
        P, h = self.plane_to_points(np.atleast_2d(X))
        return self.cost_points(P, h)

    def evaluate_sph(self, Om):
        P, h = self.sph_to_points(np.atleast_2d(Om))
        return self.cost_points(P, h)

    def components(self, P, h):
        """Unweighted components of one path: F1 length, F2 threat, F3 altitude,
        F4 smoothness, number of violated elements, max turn, total cost J,
        minimum margin to the inflated threat radius R+D, exact minimum ground
        clearance."""
        p = self.p
        P = np.asarray(P, float)
        F = self.full_path(P[None])[0]
        _, length, thr, nv_e, margin, _ = self.edge_cost(F[:-1], F[1:], detail=True)
        _, turn, dclimb = self.path_turns(F[:-1], F[1:], detail=True)
        h = np.asarray(h, float)
        v = np.maximum(p.h_min - h, 0.0) + np.maximum(h - p.h_max, 0.0)
        clear = self.terrain.min_clearance(F[:-1], F[1:])
        top = self.terrain.max_height(F[:-1], F[1:])
        return dict(F1=float(length.sum()), F2=float(thr.sum()),
                    F3=float(np.abs(h - 0.5 * (p.h_min + p.h_max)).sum()),
                    F4=float((p.a1 * turn + p.a2 * dclimb).sum()),
                    n_viol=int(nv_e.sum() + np.count_nonzero(v > 0)),
                    max_turn=float(turn.max()),
                    J=float(self.cost_points(P[None], h[None])[0]),
                    thr_margin=float(margin.min()), clearance=float(clear.min()),
                    max_agl=float(top.max()))


COMP_KEYS = ("F1", "F2", "F3", "F4", "n_viol", "max_turn", "J", "thr_margin", "clearance", "max_agl")


def make_problem(inst, params: CostParams = CostParams()):
    name, n = parse_instance(inst)
    return PathProblem(get_scenario(name), n, params)
