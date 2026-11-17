"""Statistics, tables and figures for the paper (reads results/<tag>/*.npz).

    python src/analyze.py main

Outputs: results/<tag>/tables/*.csv|*.md and figures/*.png (600 dpi).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from algorithms import ABLATION, MAIN, PLUGIN
from env import PathProblem, build_scenarios

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
INSTS = [f"S{k}_n{n}" for n in (10, 20) for k in (1, 2, 3, 4, 5, 6)]
HELD_OUT = {"S5", "S6"}
LABEL = {"DPF-PSO": "DPF-PSO", "LSHADE": "L-SHADE", "LSHADE+DPF": "L-SHADE+DPF",
         "DPF-PSO_noG": "w/o global fusion", "DPF-PSO_noR": "w/o random candidates",
         "DPF-PSO_noL": "w/o local fusion", "DPF-PSO_noA": "w/o step adaptation",
         "DPF-PSO_greedy": "greedy instead of DP", "DPF-only": "DPF without PSO moves"}
COLORS = {"PSO": "#7f7f7f", "SPSO": "#1f77b4", "GWO": "#8c564b", "WOA": "#9467bd",
          "HHO": "#e377c2", "DE": "#2ca02c", "LSHADE": "#bcbd22", "DPF-PSO": "#d62728"}
STYLES = {"PSO": ":", "SPSO": "--", "GWO": "-.", "WOA": (0, (3, 1, 1, 1)), "HHO": (0, (5, 2)),
          "DE": "-.", "LSHADE": (0, (1, 1)), "DPF-PSO": "-"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "mathtext.fontset": "stix", "savefig.dpi": 600})


def lab(a):
    return LABEL.get(a, a)


def load(tag):
    data = {}
    for f in glob.glob(os.path.join(ROOT, "results", tag, "*.npz")):
        inst, algo = os.path.basename(f)[:-4].split("__")
        with np.load(f) as z:
            data.setdefault(inst, {})[algo] = {k: z[k] for k in z.files}
    return data


def wilcoxon_sign(ref, other, alpha=0.05):
    """'+' if ref is significantly better (lower), '-' if worse, '=' otherwise."""
    if np.allclose(ref, other):
        return "=", 1.0
    p = stats.mannwhitneyu(ref, other, alternative="two-sided").pvalue
    if p >= alpha:
        return "=", p
    return ("+" if np.median(ref) < np.median(other) else "-"), p


def fmt(x):
    return f"{x:,.1f}" if abs(x) < 1e5 else f"{x:.3g}"


# ----------------------------------------------------------------------------
def table_main(data, outdir):
    algos = [a for a in MAIN if all(a in data[i] for i in INSTS)]
    ref = "DPF-PSO"
    rows, md = [], []
    md.append("| Instance | Metric | " + " | ".join(lab(a) for a in algos) + " |")
    md.append("|" + "---|" * (len(algos) + 2))
    wtl = {a: [0, 0, 0] for a in algos if a != ref}
    means = np.zeros((len(INSTS), len(algos)))
    for ii, inst in enumerate(INSTS):
        m = [data[inst][a]["best_f"].mean() for a in algos]
        means[ii] = m
        best_mean = int(np.argmin(m))
        cells_mean, cells_std, cells_best, cells_sr = [], [], [], []
        for k, a in enumerate(algos):
            b = data[inst][a]["best_f"]
            sr = 100.0 * np.mean(data[inst][a]["comps"][:, 4] == 0)
            s = ""
            if a != ref:
                sign, p = wilcoxon_sign(data[inst][ref]["best_f"], b)
                s = {"+": "+", "-": "−", "=": "≈"}[sign]
                wtl[a][{"+": 0, "=": 1, "-": 2}[sign]] += 1
            mtxt = fmt(b.mean()) + (s and f" {s}")
            cells_mean.append(f"**{mtxt}**" if k == best_mean else mtxt)
            cells_std.append(fmt(b.std(ddof=1)))
            cells_best.append(fmt(b.min()))
            cells_sr.append(f"{sr:.0f}")
            rows.append(dict(instance=inst, algorithm=a, mean=b.mean(), std=b.std(ddof=1), best=b.min(),
                             worst=b.max(), median=np.median(b), success_rate=sr,
                             time=data[inst][a]["times"].mean(), sign=s))
        md.append(f"| {inst} | Mean | " + " | ".join(cells_mean) + " |")
        md.append(f"| | Std | " + " | ".join(cells_std) + " |")
        md.append(f"| | Best | " + " | ".join(cells_best) + " |")
    ranks = np.array([stats.rankdata(r) for r in means])
    avg_rank = ranks.mean(axis=0)
    fr = stats.friedmanchisquare(*[means[:, k] for k in range(len(algos))])
    md.append("| +/≈/− | | " + " | ".join("—" if a == ref else "/".join(map(str, wtl[a])) for a in algos) + " |")
    md.append("| Avg. rank | | " + " | ".join(f"{r:.2f}" for r in avg_rank) + " |")
    with open(os.path.join(outdir, "table_main.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + f"\n\nFriedman chi2={fr.statistic:.2f}, p={fr.pvalue:.2e}\n")
    _csv(os.path.join(outdir, "table_main.csv"), rows)
    return dict(algos=algos, avg_rank=avg_rank.tolist(), friedman=(fr.statistic, fr.pvalue),
                wtl=wtl, means=means.tolist())


def table_plugin(data, outdir):
    md = ["| Instance | " + " | ".join(f"{lab(h)} → +DPF" for h, _ in PLUGIN) + " |",
          "|" + "---|" * (len(PLUGIN) + 1)]
    rows = []
    imp_all = {h: [] for h, _ in PLUGIN}
    for inst in INSTS:
        cells = []
        for h, hd in PLUGIN:
            if h not in data[inst] or hd not in data[inst]:
                cells.append("n/a")
                continue
            a, b = data[inst][h]["best_f"], data[inst][hd]["best_f"]
            imp = 100.0 * (a.mean() - b.mean()) / a.mean()
            imp_all[h].append(imp)
            sign, p = wilcoxon_sign(b, a)
            cells.append(f"{fmt(a.mean())} → {fmt(b.mean())} ({imp:+.1f}%, {sign})")
            rows.append(dict(instance=inst, host=h, host_mean=a.mean(), dpf_mean=b.mean(),
                             improvement_pct=imp, p=p, sign=sign))
        md.append(f"| {inst} | " + " | ".join(cells) + " |")
    md.append("| Mean improvement | " + " | ".join(f"{np.mean(v):.1f}%" if v else "n/a"
                                               for v in imp_all.values()) + " |")
    with open(os.path.join(outdir, "table_plugin.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    _csv(os.path.join(outdir, "table_plugin.csv"), rows)
    return {h: float(np.mean(v)) for h, v in imp_all.items() if v}


def table_ablation(data, outdir):
    algos = [a for a in ABLATION if all(a in data[i] for i in INSTS)]
    md = ["| Variant | " + " | ".join(INSTS) + " | Avg. rank |", "|" + "---|" * (len(INSTS) + 2)]
    means = np.array([[data[i][a]["best_f"].mean() for a in algos] for i in INSTS])
    ranks = np.array([stats.rankdata(r) for r in means]).mean(axis=0)
    rows = []
    for k, a in enumerate(algos):
        cells = []
        for ii, inst in enumerate(INSTS):
            b = data[inst][a]["best_f"]
            s = ""
            if a != "DPF-PSO":
                s = {"+": " +", "-": " −", "=": " ≈"}[wilcoxon_sign(data[inst]["DPF-PSO"]["best_f"], b)[0]]
            txt = fmt(b.mean()) + s
            cells.append(f"**{txt}**" if k == int(np.argmin(means[ii])) else txt)
            rows.append(dict(variant=a, instance=inst, mean=b.mean(), std=b.std(ddof=1),
                             dp_fes=data[inst][a]["dp_fes"].mean(), time=data[inst][a]["times"].mean()))
        md.append(f"| {lab(a)} | " + " | ".join(cells) + f" | {ranks[k]:.2f} |")
    with open(os.path.join(outdir, "table_ablation.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    _csv(os.path.join(outdir, "table_ablation.csv"), rows)
    return dict(algos=algos, ranks=ranks.tolist())


def table_runtime(data, outdir):
    algos = [a for a in MAIN if all(a in data[i] for i in INSTS)]
    md = ["| n | " + " | ".join(lab(a) for a in algos) + " |", "|" + "---|" * (len(algos) + 1)]
    out = {}
    for n in (10, 20):
        ins = [i for i in INSTS if i.endswith(f"n{n}")]
        t = [np.mean([data[i][a]["times"].mean() for i in ins]) for a in algos]
        md.append(f"| {n} | " + " | ".join(f"{x:.2f}" for x in t) + " |")
        out[n] = dict(zip(algos, t))
        budget = 30000 if n == 10 else 60000
        share = np.mean([data[i]["DPF-PSO"]["dp_fes"].mean() for i in ins]) / budget
        out[f"dp_share_{n}"] = float(share)
    md.append("")
    md.append("Share of the FE budget charged to DP fusion in DPF-PSO: "
              f"n=10: {100 * out['dp_share_10']:.0f}%, n=20: {100 * out['dp_share_20']:.0f}%")
    with open(os.path.join(outdir, "table_runtime.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    return out


def _csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


# ----------------------------------------------------------------------------
def fig_convergence(data, name="fig_convergence.png"):
    algos = [a for a in MAIN if all(a in data[i] for i in INSTS)]
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 4.6))
    for ax, inst in zip(axes.ravel(), INSTS):
        n = int(inst.split("_n")[1])
        fes = np.linspace(0, 30000 if n == 10 else 60000, 101)
        hi = []
        for a in algos:
            c = data[inst][a]["curves"].mean(axis=0)
            ax.plot(fes[1:] / 1e3, c[1:], color=COLORS[a], ls=STYLES[a],
                    lw=1.3 if a == "DPF-PSO" else 0.9, label=lab(a))
            hi.append(c[10])
        lo = min(data[inst][a]["curves"].mean(axis=0)[-1] for a in algos)
        ax.set_ylim(lo * 0.97, min(np.percentile(hi, 60) * 1.05, lo * 1.9))
        ax.set_title(_title(inst), pad=2)
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(pad=1)
    for ax in axes[-1]:
        ax.set_xlabel("FEs (×10$^3$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean best cost")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.3, w_pad=0.4, h_pad=0.6)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def _title(inst):
    k, n = inst.split("_n")
    return f"{k}, n={n}" + (" (held-out)" if k in HELD_OUT else "")


def fig_boxplots(data, name="fig_boxplots.png"):
    algos = [a for a in MAIN if all(a in data[i] for i in INSTS)]
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 4.9))
    for ax, inst in zip(axes.ravel(), INSTS):
        vals = [data[inst][a]["best_f"] for a in algos]
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=2), medianprops=dict(color="k", lw=0.8),
                        boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for patch, a in zip(bp["boxes"], algos):
            patch.set_facecolor(COLORS[a])
            patch.set_alpha(0.75)
        ax.set_ylim(min(np.min(v) for v in vals) * 0.98, np.percentile(np.concatenate(vals), 97) * 1.03)
        ax.set_xticks(range(1, len(algos) + 1))
        if ax in axes[-1]:
            ax.set_xticklabels([lab(a) for a in algos], rotation=60, ha="right", fontsize=6)
        else:
            ax.set_xticklabels([])
        ax.set_title(_title(inst), pad=2)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.tick_params(pad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Final cost")
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.5)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def fig_paths(data, sc, n=20, show=("PSO", "SPSO", "DE", "DPF-PSO"), name="fig_paths.png"):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.75))
    axes = axes.ravel()
    for ax, key in zip(axes, ["S1", "S2", "S3", "S4", "S5", "S6"]):
        s = sc[key]
        T = s.terrain
        g = np.arange(T.nx) * T.res
        cs = ax.contourf(g, g, T.Z, levels=14, cmap="terrain", alpha=0.85)
        ax.contour(g, g, T.Z, levels=14, colors="k", linewidths=0.15, alpha=0.5)
        for cx, cy, R in s.threats:
            ax.add_patch(Circle((cx, cy), R, fc="#ff000055", ec="r", lw=0.6))
            ax.add_patch(Circle((cx, cy), R + 11, fc="none", ec="r", lw=0.4, ls=":"))
        inst = f"{key}_n{n}"
        for a in show:
            d = data[inst][a]
            i = int(np.argmin(d["best_f"]))
            P = d["paths"][i]
            ax.plot(P[:, 0], P[:, 1], color=COLORS[a], ls=STYLES[a], lw=1.4 if a == "DPF-PSO" else 1.0,
                    label=lab(a), marker="o" if a == "DPF-PSO" else None, ms=1.5)
        ax.plot(*s.start[:2], "ks", ms=3.5)
        ax.plot(*s.goal[:2], "k*", ms=5.5)
        ax.set_xlim(0, T.size_x)
        ax.set_ylim(0, T.size_y)
        ax.set_aspect("equal")
        ax.set_title(f"{key}: {s.title.replace('Held-out: ', '')}" + (" (held-out)" if key in HELD_OUT else ""),
                     pad=2, fontsize=7)
        ax.set_xticks([0, 500, 1000])
        ax.set_yticks([0, 500, 1000])
        ax.tick_params(pad=1, labelsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(show), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.2, w_pad=0.3, h_pad=0.6)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def fig_3d(data, sc, inst="S4_n20", show=("SPSO", "DE", "DPF-PSO"), name="fig_3d.png"):
    key = inst.split("_")[0]
    s = sc[key]
    T = s.terrain
    step = 4
    g = np.arange(0, T.nx, step) * T.res
    X, Y = np.meshgrid(g, g)
    Z = T.Z[::step, ::step]
    fig = plt.figure(figsize=(3.5, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.55, linewidth=0, antialiased=True, rstride=1, cstride=1)
    zt = Z.max() + 250
    th = np.linspace(0, 2 * np.pi, 40)
    for cx, cy, R in s.threats:
        for zz in (0, zt):
            ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), zz, color="r", lw=0.4, alpha=0.6)
        for t in th[::8]:
            ax.plot([cx + R * np.cos(t)] * 2, [cy + R * np.sin(t)] * 2, [0, zt], color="r", lw=0.3, alpha=0.4)
    for a in show:
        d = data[inst][a]
        i = int(np.argmin(d["best_f"]))
        P = d["paths"][i]
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=COLORS[a], ls=STYLES[a], lw=1.5 if a == "DPF-PSO" else 1.0,
                label=lab(a))
    ax.set_xlabel("x (m)", labelpad=-6)
    ax.set_ylabel("y (m)", labelpad=-6)
    ax.set_zlabel("z (m)", labelpad=-6)
    ax.tick_params(pad=-3, labelsize=5)
    ax.view_init(elev=38, azim=-125)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def _toy_dp(cand, obst, w_turn=0.8, pen=50.0):
    """Exact pair-state DP on a 2-D toy problem (used only for Fig. 1).
    cost = sum of segment lengths + w_turn * sum of turning angles (rad)
           + pen for every segment that intersects the circular obstacle."""
    cx, cy, r = obst

    def seg(x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy), 0, 1)
        hit = np.hypot(x1 + t * dx - cx, y1 + t * dy - cy) < r
        return np.hypot(dx, dy) + pen * hit

    def turn(y0, y1, y2):
        a0 = np.arctan2(y1 - y0, 1.0)
        a1 = np.arctan2(y2 - y1, 1.0)
        return w_turn * np.abs(a1 - a0)

    L = len(cand)
    E = [seg(j, cand[j][:, None], j + 1, cand[j + 1][None, :]) for j in range(L - 1)]
    W = E[0]
    back = []
    for j in range(2, L):
        T = turn(cand[j - 2][:, None, None], cand[j - 1][None, :, None], cand[j][None, None, :])
        tot = W[:, :, None] + T
        arg = tot.argmin(0)
        W = np.take_along_axis(tot, arg[None], 0)[0] + E[j - 1]
        back.append(arg)
    q = int(W[:, 0].argmin())
    ch = [0] * L
    ch[L - 2] = q
    for j in range(L - 1, 1, -1):
        ch[j - 2] = int(back[j - 2][ch[j - 1], ch[j]])

    def total(y):
        return sum(seg(j, y[j], j + 1, y[j + 1]) for j in range(L - 1)) + \
            sum(turn(y[j - 1], y[j], y[j + 1]) for j in range(1, L - 1))

    return np.array([cand[j][ch[j]] for j in range(L)]), float(W[q, 0]), total


def fig_dpf_schematic(name="fig_dpf_schematic.png"):
    rng = np.random.default_rng(3)
    n = 5
    xs = np.arange(n + 2)
    par = np.array([[0, 1.6, 2.4, 1.0, -0.4, 0.6, 0], [0, -1.2, -0.6, 0.9, 1.8, 0.9, 0],
                    [0, 0.4, -1.6, -2.0, -0.8, -1.4, 0]])
    cols = ["#1f77b4", "#2ca02c", "#9467bd"]
    obst = (3.0, 0.95, 0.5)
    pert = par[0] + rng.normal(0, 0.45, n + 2)
    pert[0] = pert[-1] = 0
    cand = [np.array([0.0])] + [np.unique(np.r_[par[:, j], pert[j]]) for j in range(1, n + 1)] + \
           [np.array([0.0])]
    best, J, total = _toy_dp(cand, obst)
    assert abs(total(best) - J) < 1e-9 and J <= min(total(p) for p in par) + 1e-9
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.add_patch(Circle(obst[:2], obst[2], fc="#ff000040", ec="r", lw=0.6, zorder=0))
    for j in range(n + 1):
        for a in cand[j]:
            for b in cand[j + 1]:
                ax.plot([j, j + 1], [a, b], color="0.85", lw=0.4, zorder=1)
    for p, c in zip(par, cols):
        ax.plot(xs, p, color=c, lw=0.9, ls="--", zorder=2)
        ax.plot(xs[1:-1], p[1:-1], "o", color=c, ms=3.2, zorder=3)
    ax.plot(xs[1:-1], pert[1:-1], "^", color="#ff7f0e", ms=3.2, zorder=3)
    ax.plot(xs, best, color="#d62728", lw=1.8, zorder=4)
    ax.plot(0, 0, "ks", ms=4, zorder=5)
    ax.plot(n + 1, 0, "k*", ms=7, zorder=5)
    for j in range(1, n + 1):
        ax.axvline(j, color="0.6", lw=0.4, ls=":", zorder=0)
        ax.text(j, 2.95, f"$C_{j}$", ha="center", fontsize=7)
    ax.text(0, -0.5, "S", ha="center", fontsize=7)
    ax.text(n + 1, -0.5, "G", ha="center", fontsize=7)
    ax.text(obst[0] + 0.55, obst[1] + 0.35, "threat", fontsize=5.5, color="r")
    ax.set_xlim(-0.4, n + 1.4)
    ax.set_ylim(-2.6, 3.3)
    ax.set_aspect("auto")
    ax.axis("off")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, ls="--", marker="o", ms=3, lw=0.9) for c in cols] + \
         [Line2D([], [], color="#ff7f0e", ls="", marker="^", ms=3),
          Line2D([], [], color="#d62728", lw=1.8)]
    ax.legend(hs, ["parent 1", "parent 2", "parent 3", "perturbation", "DP-optimal fusion"],
              loc="lower center", ncol=5, fontsize=5.5, frameon=False, bbox_to_anchor=(0.5, -0.17),
              handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    return dict(J=J, parents=[float(total(p)) for p in par])


def fig_time(data, insts=("S2_n20", "S3_n20"), name="fig_time.png"):
    algos = [a for a in MAIN if all(a in data[i] for i in INSTS)]
    fig, axes = plt.subplots(1, len(insts), figsize=(3.5, 1.7))
    for ax, inst in zip(axes, insts):
        lo = []
        for a in algos:
            t = data[inst][a]["tcurves"].mean(axis=0)
            c = data[inst][a]["curves"].mean(axis=0)
            ax.plot(t[1:], c[1:], color=COLORS[a], ls=STYLES[a], lw=1.3 if a == "DPF-PSO" else 0.9,
                    label=lab(a))
            lo.append(c[-1])
        ax.set_ylim(min(lo) * 0.97, min(lo) * 1.6)
        ax.set_xscale("log")
        ax.set_xlabel("CPU time (s)")
        ax.set_title(inst.replace("_n", ", n="), pad=2)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("Mean best cost")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def main(tag):
    data = load(tag)
    outdir = os.path.join(ROOT, "results", tag, "tables")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    summary = {}
    summary["main"] = table_main(data, outdir)
    summary["plugin"] = table_plugin(data, outdir)
    summary["ablation"] = table_ablation(data, outdir)
    summary["runtime"] = table_runtime(data, outdir)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    sc = build_scenarios()
    fig_dpf_schematic()
    fig_convergence(data)
    fig_boxplots(data)
    fig_paths(data, sc)
    fig_3d(data, sc)
    fig_time(data)
    for f in ("table_main.md", "table_plugin.md", "table_ablation.md", "table_runtime.md"):
        print(open(os.path.join(outdir, f), encoding="utf-8").read())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
