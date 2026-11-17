::: {custom-style="Title"}
Supplementary Material for "Dynamic-Programming Path Fusion: An Optimal Recombination Operator for Swarm Optimizers in 3-D UAV Path Planning"
:::

::: {custom-style="Author"}
Anonymous Authors
:::

[[COLS2]]

All numbers in this supplement are generated from the stored results by `src/analysis2.py` and `paper/build2.py` of the code package. Notation follows the paper: $\tilde J$ is the median final cost over runs, $\Delta=(\tilde J_B-\tilde J_P)/\tilde J_B$ is the gap of the proposed configuration $P$ to method $B$ ($\Delta>0$: $P$ better), $A_{12}$ is the probability that a run of $P$ ends with a lower cost than a run of $B$, and +/≈/− means $P$ significantly better / not significantly different / significantly worse after Holm correction.

# S1. Exact Ground Clearance

Let a segment run from $\mathbf{a}$ to $\mathbf{b}$, $\mathbf{p}(t)=\mathbf{a}+t(\mathbf{b}-\mathbf{a})$, $t\in[0,1]$. Inside the grid cell with corners $z_{00},z_{01},z_{10},z_{11}$ the terrain is $Z=z_{00}+(z_{01}-z_{00})\xi+(z_{10}-z_{00})\eta+(z_{00}-z_{01}-z_{10}+z_{11})\xi\eta$ with local coordinates $\xi,\eta\in[0,1]$ that are affine in $t$. Hence $Z(t)$ is a quadratic polynomial on the parameter interval $[t_0,t_1]$ in which the segment crosses the cell, and so is the clearance $c(t)=p_z(t)-Z(t)$. Its minimum over $[t_0,t_1]$ is attained at $t_0$, at $t_1$ or, if $c$ is convex there, at the stationary point. The cost function evaluates this minimum for every crossed cell. To save time, a segment is cut into pieces of at most four cells; for each piece, a lower bound of the clearance (lowest altitude of the piece minus the maximum terrain height in a window that contains it, taken from max-pooled terrain maps with window sizes 1, 2, 4 and 8 cells) is computed first, and the exact evaluation is skipped if the bound is at least $h_c$. The violation $(h_c-c_{\min})^+$ is therefore exact, and the reported minimum clearance of final paths is computed without screening.

Verification (`src/check_clearance.py`, results in `notes/clearance_check.json`): on 7,200 random segments in S1–S6, H01, H17 and C1 (half of them with endpoints 10–60 m above the ground, so that the minimum lies near the 20 m threshold), the vectorized implementation agreed with an independently written implementation (`src/clearance_ref.py`, which splits the segment at every grid-line crossing and reconstructs each quadratic from three terrain evaluations) to within $5.0\times10^{-13}$ m, and the violations agreed to within $4.7\times10^{-13}$ m. Dense sampling with 10,001 points per segment never gave a lower clearance than the exact minimum and exceeded it by at most 0.015 m. The violation of a segment did not depend on the other segments evaluated in the same batch.

# S2. Implementation Details and Protocol

1. *Inertia weight.* PSO (alone and inside DPF-PSO) decreases the inertia weight linearly from 0.9 to 0.4 with the fraction of the FE budget used, because the number of iterations is not known in advance when fusion consumes FEs. SPSO uses $w=w_0(w_{\mathrm{end}}/w_0)^{\phi}$ with $\phi$ the fraction of the budget used; the default reproduces the schedule of its public code ($w_0=1$, damping 0.98 per iteration over 200 iterations).
2. *Budget.* The evaluator stops exactly at the budget: rows of a batch beyond the budget are not evaluated. A fusion is skipped if its FE charge exceeds the remaining budget; the host then uses the rest. IDP fills a remainder smaller than one round with random samples; GLS charges whole FEs when the accumulated fractional charge exceeds one.
3. *Map limits.* The bounds of $u_j$ are the intersection of $[-400,400]$ m with the offsets for which $B_j+u_j\mathbf{e}_2$ lies in the map shrunk by 5 m; after the along-track shift, coordinates are clipped to the same rectangle. All slot-encoded methods share this decoder. SPSO clips every waypoint to the same rectangle and its height above ground to [100, 200] m, as in its public code.
4. *Violations.* Each segment adds $M_0$ for every threat whose inflated radius $R_k+D$ it enters and once more if its exact minimum clearance is below $h_c$, plus $\lambda$ times the sum of the penetration depths and of the clearance shortfall. A run is feasible if its final path has no violation.
5. *Random candidates.* The $N_r$ random solutions of a global fusion are not evaluated as paths and are not charged; only the segment evaluations of the fusion are charged.
6. *Distinctness.* Global fusion selects the $N_g$ best solutions with distinct decision vectors (exact equality); within every slot, duplicate waypoints are removed before the dynamic program.
7. *Seeds.* Pilot runs of the first study used seeds 20270201 + $r$, tuning used 30270000 + $r$, all final runs use 20271101 + $r$ (run $r$ of every method gets the same seed), the timing benchmark 40270000 + $r$, and held-out scenario H$k$ is generated from seed 90210 + $k$.
8. *Smoothness term.* $F_4$ follows Eq. (8) of [5] with unit coefficients; the public code of [5] adds an angle only if it exceeds 45°, which makes the cost piecewise constant in the angles. Segments with zero horizontal length, which only the spherical encoding of SPSO can produce, are skipped and the angles are measured between consecutive non-degenerate segments, as the public code does for the turning angle. Without this rule, an implementation that assigns a zero turn to such segments lets SPSO hide sharp turns behind collapsed waypoints.
9. *SPSO azimuth.* The public SPSO code swaps sine and cosine of the azimuth; we use the mapping that is consistent with the azimuth bounds. This matters only when $SG$ is not diagonal.
10. *CMA-ES.* pycma 4.5.0 on the slot encoding scaled to $[0,1]^{3n}$, bound handling of pycma, initial mean uniform at random, initial step 0.3, default population size $4+\lfloor3\ln(3n)\rfloor$ doubled at every restart (IPOP). A run restarts when the step size falls below $10^{-3}$ (pycma option tolx), when the recent best costs vary by less than 1 (tolfun, tolfunhist; costs are of the order of $10^3$–$10^4$), or after 200 iterations without progress (tolstagnation). With pycma's default thresholds, which we used first, the first run never stopped within $3000n$ FEs, so IPOP never restarted. We noticed this in the first CMA-ES results on S1–S6, chose the thresholds above in a check on S1 and S3 with four tuning seeds (`notes/check_cmaes2.py`; median on S3 with $n=10$: 10220 with the default thresholds, 7577 with the new ones), and re-ran CMA-ES in all experiments; the earlier CMA-ES results are not used.
11. *L-SHADE.* Initial population $18\cdot3n$ reduced linearly to 4, memory size 6, archive rate 2.6, $p=0.11$ (defaults of [28]). For the plug-in variant, DPF uses the current population as memory.
12. *Plug-in hosts.* DE and L-SHADE pass their population to DPF; GWO, WOA and HHO pass their population together with their leaders (the three best wolves, the best whale, the rabbit). An accepted fused path replaces the worst member. SPSO is not used as a host, because in its spherical encoding the $j$-th waypoint is not tied to slot $j$.

*Development history.* A first version of this study used a sampled clearance check (20 points per segment) and an iteration-based fusion schedule; its results are not used here. In the present study, the cost function (exact clearance, degenerate segments), the fusion schedule (counted in host FEs), the baselines and the tuning protocol were fixed before the final runs; the only later change concerns the restart thresholds of the CMA-ES baseline (item 10), which were changed after its first results on S1–S6 and before any of its held-out results existed. A two-run smoke test of the pipeline on S4, H01 and C1, run before the final experiments, revealed the zero-turn loophole described in item 8; the fix changed only the cost function, identically for all methods, and no algorithm or parameter was changed after the smoke test. The smoke-test results were discarded. After the held-out results were known, we added the tuning of CMA-ES (S3), the height-above-ground analysis, the held-out ablation and equal-time comparisons, and the altitude-band variant (S4). DPF-PSO, its default and tuned configurations and all earlier results were left unchanged. The altitude-band experiment tests an explanation that was suggested by the held-out results and is therefore exploratory.

# S3. Parameter Tuning

Each tuned method (DE, L-SHADE, SPSO, IDP, DPF-PSO and CMA-ES) received the default configuration plus 24 configurations drawn at random from the space below, each run with 5 seeds on S1–S4 with $n\in\{10,20\}$ (1,000 runs per method). The score of a configuration is the rank of its median cost among the 25 configurations on each instance, averaged over the 8 instances; ties are broken by the mean normalized median.

* DE: strategy ∈ {rand/1/bin, best/1/bin, current-to-best/1/bin}, $F\sim U(0.3,1)$, $CR\sim U(0,1)$, NP ∈ {20, 50, 100, 5D, 10D}.
* L-SHADE: $r^{N_{\mathrm{init}}}$ ∈ {5, 10, 18, 25}, $p\sim U(0.05,0.25)$, $H$ ∈ {5, 6, 10, 20}, $r^{\mathrm{arc}}\sim U(1,3)$.
* SPSO: $N$ ∈ {30, 50, 100, 200, 500}, $c_1,c_2\sim U(1,2.5)$, $w_0\sim U(0.5,1.2)$, $w_{\mathrm{end}}\sim\log U(0.005,0.5)$, velocity limit $\alpha\sim U(0.1,1)$.
* IDP: $K$ ∈ {10, 15, 20, 30, 40}, $r_{\min}\sim\log U(10^{-4},10^{-1})$, initial sample $N_0$ ∈ {20, 50, 100}.
* DPF-PSO: $N_r$ ∈ {0, 25, 50, 100}, $N_g$ ∈ {25, 50}, global period ∈ {0 (initial fusion only), 2500, 5000, 10000} host FEs, local period ∈ {100, 250, 500, 1000}, $N_e$ ∈ {2, 5, 10}, $N_p$ ∈ {3, 6, 12}, $\sigma_0\sim\log U(0.01,0.2)$, adaptive $\sigma$ ∈ {yes, no}.
* CMA-ES (tuned with the same protocol after the other methods): $\sigma_0\sim\log U(0.05,0.6)$, population size multiplier ∈ {1, 2, 4, 8}, tolx $\sim\log U(10^{-4},10^{-2})$, tolfun $\sim\log U(0.1,10)$, tolstagnation ∈ {100, 200, 400}. The default configuration had the best mean rank, so tuned CMA-ES is identical to CMA-ES.

The selected configuration of DPF-PSO uses only the initial global fusion (global period 0) and a fixed step size, so neither periodic global fusion nor the success-based step rule was part of the best configuration found; it was selected on S1–S4 before any held-out run.

: TABLE S1. Selected Configurations (Mean Rank among 25 Configurations on S1–S4)

| Algorithm | Selected configuration | Mean rank (selected / default) |
|:--|:--|:-:|
| DE | strategy = ctb1bin, F = 0.362, CR = 0.173, NP = 50 | 3.50 / 4.00 |
| L-SHADE | r_ninit = 5, p_best = 0.17, H = 10, r_arc = 2.42 | 3.75 / 14.88 |
| SPSO | N = 100, c1 = 1.6, c2 = 1.87, w0 = 1.19, w_end = 0.0987, alpha = 0.572 | 4.12 / 8.50 |
| IDP | K = 20, r_min = 0.000124, N0 = 100 | 6.62 / 13.12 |
| DPF-PSO | g_random = 100, g_max = 25, g_every = 0, l_every = 500, n_elite = 10, n_perturb = 12, sigma0 = 0.0221, adapt = False | 8.12 / 12.12 |
| CMA-ES | sigma0 = 0.3, pop_mult = 1, tolx = 0.001, tolfun = 1, tolstagnation = 200 | 5.88 / 5.88 |

# S4. Held-Out Scenarios

: TABLE S2. Default Configurations on H01–H40: Paired Comparison of DPF-PSO with Each Method

| Compared method | n | Win / loss | Δ̃ [Q1, Q3] | Median A12 | Within-scenario +/≈/− | p | p~H~ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| PSO | 10 | 88% / 12% | +5.4% [+1.5%, +11.3%] | 0.82 | 19/21/0 | <0.001 | <0.001 |
| PSO | 20 | 100% / 0% | +16.6% [+10.6%, +20.0%] | 0.97 | 37/3/0 | <0.001 | <0.001 |
| SPSO | 10 | 57% / 42% | +1.0% [−2.8%, +3.9%] | 0.57 | 15/14/11 | 0.444 | 0.887 |
| SPSO | 20 | 78% / 22% | +5.8% [+1.6%, +9.1%] | 0.85 | 23/13/4 | <0.001 | 0.003 |
| GWO | 10 | 100% / 0% | +14.5% [+9.8%, +20.8%] | 1.00 | 40/0/0 | <0.001 | <0.001 |
| GWO | 20 | 100% / 0% | +25.8% [+19.4%, +31.8%] | 1.00 | 40/0/0 | <0.001 | <0.001 |
| WOA | 10 | 100% / 0% | +18.2% [+11.0%, +29.5%] | 1.00 | 40/0/0 | <0.001 | <0.001 |
| WOA | 20 | 95% / 5% | +22.4% [+8.0%, +32.2%] | 1.00 | 36/4/0 | <0.001 | <0.001 |
| HHO | 10 | 100% / 0% | +12.2% [+7.5%, +20.4%] | 1.00 | 40/0/0 | <0.001 | <0.001 |
| HHO | 20 | 98% / 2% | +19.0% [+7.3%, +30.8%] | 1.00 | 35/5/0 | <0.001 | <0.001 |
| DE | 10 | 57% / 42% | +2.3% [−0.5%, +6.9%] | 0.64 | 12/21/7 | 0.001 | 0.003 |
| DE | 20 | 95% / 5% | +9.9% [+3.5%, +14.9%] | 0.96 | 28/12/0 | <0.001 | <0.001 |
| L-SHADE | 10 | 85% / 15% | +3.4% [+0.3%, +9.6%] | 0.90 | 22/16/2 | <0.001 | <0.001 |
| L-SHADE | 20 | 92% / 8% | +9.6% [+1.8%, +16.2%] | 1.00 | 31/9/0 | <0.001 | <0.001 |
| CMA-ES | 10 | 98% / 2% | +11.8% [+5.6%, +17.0%] | 1.00 | 35/5/0 | <0.001 | <0.001 |
| CMA-ES | 20 | 100% / 0% | +14.4% [+7.4%, +24.1%] | 0.98 | 31/9/0 | <0.001 | <0.001 |
| IDP | 10 | 85% / 15% | +0.8% [+0.4%, +1.2%] | 0.76 | 15/25/0 | <0.001 | <0.001 |
| IDP | 20 | 45% / 55% | −0.1% [−0.6%, +1.2%] | 0.43 | 0/37/3 | 0.666 | 0.887 |
| GLS | 10 | 92% / 8% | +2.1% [+0.9%, +6.0%] | 0.89 | 24/16/0 | <0.001 | <0.001 |
| GLS | 20 | 98% / 2% | +5.8% [+2.4%, +10.3%] | 0.86 | 23/17/0 | <0.001 | <0.001 |

::: {custom-style="TableNote"}
Win / loss: share of the 40 scenarios on which DPF-PSO has the lower / higher median. Δ̃ [Q1, Q3]: median and quartiles of the 40 scenario gaps. Within-scenario +/≈/−: rank-sum tests on each scenario (10 runs per method), Holm over the 10 methods. p: two-sided Wilcoxon signed-rank test on the 40 paired scenario medians; p~H~: Holm-adjusted over the 20 tests of this table.
:::

: TABLE S3. Mean Rank and Share of Feasible Runs on H01–H40

| | PSO | SPSO | GWO | WOA | HHO | DE | L-SH. | CMA | IDP | GLS | DPF |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Mean rank, n = 10 | 6.05 | 3.48 | 9.30 | 10.00 | 8.65 | 3.65 | 5.35 | 8.53 | 3.52 | 5.10 | 2.38 |
| Feasible runs, n = 10 | 100% | 100% | 100% | 99% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Mean rank, n = 20 | 7.78 | 4.15 | 9.97 | 8.72 | 7.90 | 5.47 | 5.67 | 7.50 | 2.15 | 4.67 | 2.00 |
| Feasible runs, n = 20 | 99% | 100% | 99% | 98% | 99% | 100% | 100% | 98% | 100% | 100% | 100% |

: TABLE S4. Tuned Configurations on H01–H40 (Reference: DPF-PSO-T)

| Compared method | n | Win / loss | Δ̃ [Q1, Q3] | Median A12 | Within-scenario +/≈/− | p | p~H~ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DE | 10 | 90% / 10% | +1.2% [+0.7%, +1.8%] | 0.82 | 22/18/0 | <0.001 | <0.001 |
| DE | 20 | 100% / 0% | +6.2% [+3.9%, +10.2%] | 0.94 | 33/7/0 | <0.001 | <0.001 |
| L-SHADE | 10 | 60% / 40% | +1.0% [−0.5%, +6.8%] | 0.59 | 11/19/10 | 0.005 | 0.009 |
| L-SHADE | 20 | 85% / 15% | +10.3% [+0.6%, +14.4%] | 0.96 | 28/12/0 | <0.001 | <0.001 |
| SPSO | 10 | 30% / 70% | −2.5% [−6.0%, +0.6%] | 0.08 | 3/16/21 | <0.001 | <0.001 |
| SPSO | 20 | 65% / 35% | +1.2% [−4.1%, +4.1%] | 0.64 | 9/23/8 | 0.656 | 0.656 |
| IDP | 10 | 95% / 5% | +1.1% [+0.6%, +1.6%] | 0.82 | 23/17/0 | <0.001 | <0.001 |
| IDP | 20 | 72% / 28% | +0.3% [−0.0%, +1.8%] | 0.69 | 9/31/0 | <0.001 | 0.001 |
| CMA-ES | 10 | 98% / 2% | +12.1% [+5.9%, +17.7%] | 1.00 | 36/4/0 | <0.001 | <0.001 |
| CMA-ES | 20 | 100% / 0% | +15.5% [+9.5%, +25.0%] | 1.00 | 38/2/0 | <0.001 | <0.001 |

::: {custom-style="TableNote"}
As Table S2, with the tuned configurations; Holm over the 10 tests of this table.
:::

*Merged waypoints of tuned SPSO.* Tuned SPSO had a lower median than tuned DPF-PSO on 28 ($n=10$) and 14 ($n=20$) of the 40 scenarios. The final paths of tuned SPSO had a median of 3.0 and 3.3 merged waypoints (segments of zero horizontal length) and median longest segments of 413 m and 395 m, against 144 m and 75 m for tuned DPF-PSO. Spearman's correlation between the mean number of merged waypoints of a scenario and the gap $(\tilde J_{\mathrm{SPSO}}-\tilde J_{\mathrm{DPF}})/\tilde J_{\mathrm{SPSO}}$ was −0.70 ($p=$ $6.0\times10^{-7}$) for $n=10$ and −0.44 ($p=$ 0.004) for $n=20$: the more waypoints tuned SPSO merged, the larger its advantage.

*Anytime behaviour.* The best-so-far costs of the held-out runs were recorded at 101 equally spaced points of the budget. Table S4a gives the gap of DPF-PSO at half and at the full budget, and the share of the budget after which the median best-so-far cost of DPF-PSO first reached the final median cost of the other method (median over the scenarios; "never" if this happened on fewer than half of them). For $n=10$, the initial global fusion of DPF-PSO is charged at once and takes about a quarter of the budget, so DPF-PSO cannot reach any target before that point.

: TABLE S4a. Anytime Comparison on H01–H40 (Default Configurations)

| Compared method | n | Δ̃ at 50% of the budget | Δ̃ at 100% | Budget share at which DPF-PSO reaches the method's final median | Scenarios where it does |
|:--|:-:|:-:|:-:|:-:|:-:|
| PSO | 10 | +11.8% | +5.4% | 34% | 88% |
| SPSO | 10 | −1.4% | +1.0% | 80% | 57% |
| GWO | 10 | +28.7% | +14.5% | 29% | 100% |
| WOA | 10 | +16.6% | +18.2% | 28% | 100% |
| HHO | 10 | +13.3% | +12.2% | 29% | 100% |
| DE | 10 | +2.7% | +2.3% | 60% | 57% |
| L-SHADE | 10 | +44.0% | +3.4% | 43% | 85% |
| CMA-ES | 10 | +10.6% | +11.8% | 30% | 98% |
| IDP | 10 | −0.6% | +0.8% | 80% | 85% |
| GLS | 10 | +3.8% | +2.1% | 73% | 92% |
| PSO | 20 | +36.8% | +16.6% | 18% | 100% |
| SPSO | 20 | +2.8% | +5.8% | 38% | 78% |
| GWO | 20 | +53.1% | +25.8% | 16% | 100% |
| WOA | 20 | +20.1% | +22.4% | 16% | 95% |
| HHO | 20 | +17.5% | +19.0% | 17% | 98% |
| DE | 20 | +10.5% | +9.9% | 38% | 95% |
| L-SHADE | 20 | +71.3% | +9.6% | 38% | 92% |
| CMA-ES | 20 | +13.8% | +14.4% | 18% | 100% |
| IDP | 20 | −0.6% | −0.1% | never (median) | 45% |
| GLS | 20 | +7.4% | +5.8% | 38% | 98% |

*Height above ground along the paths.* Table S4b reports, for the final paths on H01–H40, the share of the horizontal track below 100 m and above 200 m above the ground (1 m sampling, mean over paths) and the median minimum and maximum height. The altitude band of the cost model applies only at the waypoints. Table S4c gives the same for the variant in which the band is enforced exactly along every segment; the maximum height of a segment is computed with the exact clearance machinery applied to the mirrored terrain $-Z$ (`src/check_ceiling.py`: agreement with the independent implementation to within $4.0\times10^{-13}$ m on 3,600 random segments), and the independent safety check of S10 verifies the band along the segments for this variant. Band violations of up to $10^{-6}$ m are ignored in this variant, because waypoints lie exactly on the band limits ($h_j\in[100,200]$ m), where rounding of the order of $10^{-13}$ m would otherwise decide about a penalty of $M_0$.

: TABLE S4b. Height above Ground along the Final Held-Out Paths

| Method | n | Track below 100 m | Track above 200 m | Median min. height (m) | Median max. height (m) |
|:--|:-:|:-:|:-:|:-:|:-:|
| SPSO | 10 | 9.7% | 12.7% | 82 | 190 |
| SPSO (tuned) | 10 | 11.8% | 16.2% | 69 | 194 |
| DE | 10 | 6.2% | 3.1% | 103 | 179 |
| IDP | 10 | 2.4% | 1.5% | 120 | 166 |
| DPF-PSO | 10 | 3.2% | 1.8% | 113 | 171 |
| DPF-PSO (tuned) | 10 | 2.8% | 1.9% | 113 | 171 |
| SPSO | 20 | 7.2% | 8.4% | 103 | 169 |
| SPSO (tuned) | 20 | 8.4% | 10.7% | 90 | 174 |
| DE | 20 | 2.1% | 0.6% | 130 | 161 |
| IDP | 20 | 0.3% | 0.1% | 142 | 156 |
| DPF-PSO | 20 | 0.4% | 0.1% | 138 | 157 |
| DPF-PSO (tuned) | 20 | 0.4% | 0.1% | 139 | 156 |

: TABLE S4c. As Table S4b, Altitude Band Enforced along Every Segment

| Method | n | Track below 100 m | Track above 200 m | Median min. height (m) | Median max. height (m) |
|:--|:-:|:-:|:-:|:-:|:-:|
| SPSO | 10 | 0.0% | 0.0% | 108 | 181 |
| DE | 10 | 0.0% | 0.0% | 110 | 180 |
| IDP | 10 | 0.0% | 0.0% | 124 | 166 |
| DPF-PSO | 10 | 0.0% | 0.0% | 115 | 174 |
| SPSO (tuned) | 10 | 0.0% | 0.0% | 101 | 185 |
| DPF-PSO (tuned) | 10 | 0.0% | 0.0% | 115 | 173 |
| SPSO | 20 | 0.0% | 0.0% | 127 | 165 |
| DE | 20 | 0.0% | 0.0% | 133 | 159 |
| IDP | 20 | 0.0% | 0.0% | 141 | 156 |
| DPF-PSO | 20 | 0.0% | 0.0% | 139 | 157 |
| SPSO (tuned) | 20 | 0.0% | 0.0% | 123 | 166 |
| DPF-PSO (tuned) | 20 | 0.0% | 0.0% | 139 | 156 |

*Further held-out experiments.* Tables S4d–S4h give the details of the comparisons in Table III of the paper. The ablation variants and the baselines at equal CPU time were run with the same seeds as the held-out runs of Table S2; DPF-PSO is represented by its runs of Table S2 with its normal budget. The altitude-band experiment is a complete re-run of the six methods with the modified cost model; its costs are not comparable with those of the other tables.

: TABLE S4d. Held-Out Ablation: DPF-PSO against Each Variant

| Compared method | n | Win / loss | Δ̃ [Q1, Q3] | Median A12 | Within-scenario +/≈/− | p | p~H~ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| PSO | 10 | 88% / 12% | +5.4% [+1.5%, +11.3%] | 0.82 | 19/20/1 | <0.001 | <0.001 |
| PSO | 20 | 100% / 0% | +16.6% [+10.6%, +20.0%] | 0.97 | 38/2/0 | <0.001 | <0.001 |
| w/o global fusion | 10 | 85% / 15% | +1.8% [+0.4%, +4.9%] | 0.66 | 10/29/1 | <0.001 | <0.001 |
| w/o global fusion | 20 | 100% / 0% | +8.4% [+4.3%, +12.2%] | 0.94 | 32/8/0 | <0.001 | <0.001 |
| greedy instead of DP | 10 | 80% / 20% | +1.6% [+0.3%, +5.2%] | 0.78 | 17/23/0 | <0.001 | <0.001 |
| greedy instead of DP | 20 | 85% / 15% | +3.8% [+1.5%, +8.1%] | 0.85 | 23/17/0 | <0.001 | <0.001 |
| fusion only (no PSO moves) | 10 | 100% / 0% | +1.9% [+1.4%, +2.3%] | 0.91 | 25/15/0 | <0.001 | <0.001 |
| fusion only (no PSO moves) | 20 | 98% / 2% | +1.0% [+0.6%, +1.3%] | 0.70 | 11/29/0 | <0.001 | <0.001 |

: TABLE S4e. Held-Out Scenarios at Equal CPU Time: DPF-PSO (Normal Budget) against Baselines with Multiplied Budgets

| Compared method | n | Win / loss | Δ̃ [Q1, Q3] | Median A12 | Within-scenario +/≈/− | p | p~H~ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| IDP | 10 | 85% / 15% | +0.6% [+0.1%, +0.9%] | 0.72 | 14/26/0 | <0.001 | <0.001 |
| IDP | 20 | 38% / 62% | −0.2% [−0.6%, +0.9%] | 0.42 | 0/37/3 | 0.952 | 1.000 |
| SPSO | 10 | 50% / 50% | +0.1% [−3.8%, +2.9%] | 0.48 | 12/15/13 | 0.765 | 1.000 |
| SPSO | 20 | 68% / 32% | +4.4% [−1.9%, +6.2%] | 0.74 | 18/16/6 | 0.038 | 0.114 |
| DE | 10 | 55% / 45% | +2.1% [−0.8%, +6.5%] | 0.59 | 11/16/13 | 0.002 | 0.007 |
| DE | 20 | 88% / 12% | +9.5% [+2.2%, +14.0%] | 0.91 | 23/17/0 | <0.001 | <0.001 |

: TABLE S4f. Altitude Band Enforced along Every Segment: DPF-PSO against Default Baselines

| Compared method | n | Win / loss | Δ̃ [Q1, Q3] | Median A12 | Within-scenario +/≈/− | p | p~H~ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| SPSO | 10 | 98% / 2% | +6.0% [+3.1%, +9.3%] | 0.98 | 35/5/0 | <0.001 | <0.001 |
| SPSO | 20 | 100% / 0% | +8.2% [+6.2%, +10.9%] | 0.98 | 35/5/0 | <0.001 | <0.001 |
| DE | 10 | 45% / 55% | −0.3% [−0.6%, +7.6%] | 0.43 | 10/20/10 | 0.064 | 0.128 |
| DE | 20 | 98% / 2% | +6.9% [+2.5%, +11.6%] | 0.84 | 24/16/0 | <0.001 | <0.001 |
| IDP | 10 | 90% / 10% | +0.5% [+0.4%, +1.2%] | 0.73 | 12/28/0 | <0.001 | <0.001 |
| IDP | 20 | 45% / 55% | −0.2% [−0.6%, +1.7%] | 0.44 | 1/34/5 | 0.705 | 0.705 |

: TABLE S4g. Altitude Band Enforced along Every Segment: Tuned DPF-PSO against Tuned SPSO

| Compared method | n | Win / loss | Δ̃ [Q1, Q3] | Median A12 | Within-scenario +/≈/− | p | p~H~ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| SPSO | 10 | 88% / 12% | +2.8% [+0.6%, +4.4%] | 0.84 | 23/15/2 | <0.001 | <0.001 |
| SPSO | 20 | 95% / 5% | +4.6% [+2.7%, +7.4%] | 0.87 | 30/9/1 | <0.001 | <0.001 |

::: {custom-style="TableNote"}
Columns as in Table S2. Holm correction over the 8 tests of Table S4d, the 6 tests of Table S4e, and the 8 tests of Tables S4f and S4g together.
:::

: TABLE S4h. Share of Runs with a Feasible Final Path, Altitude Band Enforced along Every Segment

| Method | Feasible runs, n = 10 | Feasible runs, n = 20 |
|:--|:-:|:-:|
| SPSO | 100.0% | 100.0% |
| DE | 100.0% | 99.8% |
| IDP | 100.0% | 100.0% |
| DPF-PSO | 100.0% | 100.0% |
| SPSO (tuned) | 100.0% | 100.0% |
| DPF-PSO (tuned) | 100.0% | 100.0% |

# S5. Benchmark Instances S1–S6

: TABLE S5a. Median Final Cost on S1–S6 (30 Runs, Default Configurations)

| Inst. | PSO | SPSO | GWO | WOA | HHO | DE | L-SH. | CMA | IDP | GLS | DPF |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/10 | 6309 + | 6120 + | 6674 + | 6405 + | 6339 + | **6018 −** | 6078 ≈ | 6532 + | 6095 + | 6157 + | 6076 |
| S2/10 | 7256 + | 6743 + | 7790 + | 8243 + | 7965 + | **6387 ≈** | 7295 + | 7454 + | 6432 + | 6959 + | 6402 |
| S3/10 | 7391 + | 5927 + | 7761 + | 8621 + | 8312 + | 7102 + | 7738 + | 7679 + | 5231 + | 5993 + | **5193** |
| S4/10 | 7735 + | 7662 + | 8627 + | 8500 + | 8138 + | 7090 + | 7694 + | 7858 + | 6913 + | 7140 + | **6871** |
| S5/10* | 6252 + | 6015 + | 6830 + | 8722 + | 8435 + | 6175 + | 6294 + | 6665 + | 5612 + | 6006 + | **5534** |
| S6/10* | 4954 ≈ | 4974 ≈ | 5486 + | 5790 + | 5491 + | **4913 −** | 4951 + | 6179 + | 4985 + | 4965 ≈ | 4938 |
| S1/20 | 6799 + | 6543 + | 7564 + | 6458 + | 6378 + | 6670 + | 7261 + | 7000 + | 6259 ≈ | 6278 + | **6227** |
| S2/20 | 8851 + | 7568 + | 9853 + | 8484 + | 8576 + | 7807 + | 8178 + | 8756 + | 7105 + | 7935 + | **7079** |
| S3/20 | 9075 +^†^ | 6430 + | 10903 +^†^ | 9063 + | 8910 + | 8991 + | 8475 + | 8861 + | **5301 ≈** | 5730 + | 5308 |
| S4/20 | 9586 +^†^ | 8373 + | 10506 + | 9759 + | 8949 + | 7921 + | 8379 + | 8992 + | 7453 + | 8256 + | **7424** |
| S5/20* | 8158 + | 6832 + | 9155 + | 9755 + | 9283 + | 7043 + | 6942 + | 8976 + | **5722 ≈** | 6719 + | 5726 |
| S6/20* | 5541 + | 5501 + | 6224 + | 6337 + | 6272 + | 5182 ≈ | 5166 ≈ | 6972 + | 5164 ≈ | 5527 + | **5162** |
| +/≈/− | 11/1/0 | 11/1/0 | 12/0/0 | 12/0/0 | 12/0/0 | 8/2/2 | 10/2/0 | 12/0/0 | 8/4/0 | 11/1/0 | — |
| Rank | 7.25 | 4.50 | 9.83 | 9.50 | 8.42 | 4.00 | 5.75 | 8.58 | 2.50 | 4.25 | 1.42 |

::: {custom-style="TableNote"}
"S3/20": S3 with n = 20. +/≈/−: DPF-PSO significantly better / not different / significantly worse (two-sided rank-sum test, Holm over the 10 methods of each instance, α = 0.05); descriptive only. ^†^: some runs infeasible. Bold: lowest median. \*: S5 and S6 were not used for design or tuning. Rank: mean rank of the median over the 12 instances.
:::

: TABLE S5b. Ablation and Plug-in Study on S1–S6 (Δ Averaged over 12 Instances)

| Configuration | Rank | Δ (mean) | +/≈/− |
|:--|:-:|:-:|:-:|
| *Ablation of DPF-PSO (reference: full DPF-PSO)* | | | |
| DPF-PSO (full) | 2.33 | — | — |
| w/o global fusion | 6.08 | +4.9% | 9/2/1 |
| w/o random candidates | 2.83 | +0.3% | 3/8/1 |
| w/o local fusion | 3.75 | +0.2% | 2/10/0 |
| fixed σ schedule | 1.58 | −0.1% | 0/10/2 |
| greedy instead of DP | 6.08 | +3.1% | 11/1/0 |
| fusion only (no PSO moves) | 5.50 | +1.1% | 12/0/0 |
| PSO | 7.83 | +16.4% | 11/1/0 |
| *DPF as a plug-in (reference: host alone)* | | | |
| GWO + DPF | — | +23.1% | 12/0/0 |
| WOA + DPF | — | +22.0% | 12/0/0 |
| HHO + DPF | — | +19.7% | 12/0/0 |
| DE + DPF | — | +9.8% | 8/2/2 |
| L-SHADE + DPF | — | +12.4% | 9/2/1 |

::: {custom-style="TableNote"}
Ablation: Δ of DPF-PSO to the variant; +/≈/−: DPF-PSO significantly better / not different / worse (rank-sum, Holm over the 7 variants). Plug-in: Δ of the host with DPF to the host alone; +/≈/−: host with DPF significantly better / not different / worse (Holm over the 12 instances); PSO + DPF is DPF-PSO (row PSO). Rank: mean rank among the eight ablation configurations.
:::

: TABLE S5. Default Configurations, n = 10 (30 Runs)

| Inst. | Method | Median | Mean ± SD | Feasible | A12 | Δ | Sign (Holm) |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/10 | PSO | 6308.8 | 6368.4 ± 312.7 | 100% | 0.78 | +3.7% | + |
| S1/10 | SPSO | 6119.6 | 6220.9 ± 213.5 | 100% | 0.70 | +0.7% | + |
| S1/10 | GWO | 6674.0 | 6760.0 ± 372.6 | 100% | 1.00 | +9.0% | + |
| S1/10 | WOA | 6405.3 | 6537.0 ± 306.5 | 100% | 1.00 | +5.1% | + |
| S1/10 | HHO | 6339.3 | 6383.2 ± 135.1 | 100% | 0.99 | +4.2% | + |
| S1/10 | DE | 6017.8 | 6023.7 ± 14.6 | 100% | 0.06 | −1.0% | − |
| S1/10 | L-SHADE | 6077.5 | 6269.7 ± 347.3 | 100% | 0.55 | +0.0% | ≈ |
| S1/10 | CMA-ES | 6531.8 | 6554.4 ± 269.4 | 100% | 0.96 | +7.0% | + |
| S1/10 | IDP | 6095.4 | 6097.8 ± 29.8 | 100% | 0.74 | +0.3% | + |
| S1/10 | GLS | 6157.0 | 6203.5 ± 113.3 | 100% | 0.91 | +1.3% | + |
| S1/10 | DPF-PSO | 6076.0 | 6083.8 ± 50.6 | 100% | — | — | — |
| S2/10 | PSO | 7256.5 | 7397.5 ± 621.6 | 100% | 0.95 | +11.8% | + |
| S2/10 | SPSO | 6743.4 | 6899.2 ± 400.4 | 100% | 0.92 | +5.1% | + |
| S2/10 | GWO | 7790.0 | 7775.2 ± 749.0 | 100% | 1.00 | +17.8% | + |
| S2/10 | WOA | 8243.0 | 8226.3 ± 138.2 | 100% | 1.00 | +22.3% | + |
| S2/10 | HHO | 7965.2 | 7941.9 ± 241.3 | 100% | 1.00 | +19.6% | + |
| S2/10 | DE | 6386.7 | 6635.2 ± 391.7 | 100% | 0.45 | −0.2% | ≈ |
| S2/10 | L-SHADE | 7294.8 | 7301.8 ± 32.7 | 100% | 1.00 | +12.2% | + |
| S2/10 | CMA-ES | 7454.2 | 7500.2 ± 231.0 | 100% | 1.00 | +14.1% | + |
| S2/10 | IDP | 6432.3 | 6557.1 ± 244.8 | 100% | 0.72 | +0.5% | + |
| S2/10 | GLS | 6959.4 | 6947.5 ± 320.5 | 100% | 0.92 | +8.0% | + |
| S2/10 | DPF-PSO | 6401.5 | 6446.4 ± 119.1 | 100% | — | — | — |
| S3/10 | PSO | 7390.9 | 7787.2 ± 1313.5 | 100% | 1.00 | +29.7% | + |
| S3/10 | SPSO | 5926.9 | 6041.5 ± 501.5 | 100% | 0.99 | +12.4% | + |
| S3/10 | GWO | 7761.0 | 7601.5 ± 1316.0 | 100% | 1.00 | +33.1% | + |
| S3/10 | WOA | 8620.6 | 8538.6 ± 162.5 | 100% | 1.00 | +39.8% | + |
| S3/10 | HHO | 8312.0 | 8320.1 ± 138.4 | 100% | 1.00 | +37.5% | + |
| S3/10 | DE | 7102.4 | 7172.0 ± 598.4 | 100% | 1.00 | +26.9% | + |
| S3/10 | L-SHADE | 7737.6 | 7654.6 ± 446.3 | 100% | 1.00 | +32.9% | + |
| S3/10 | CMA-ES | 7678.9 | 8065.9 ± 891.5 | 100% | 1.00 | +32.4% | + |
| S3/10 | IDP | 5231.5 | 5311.3 ± 178.2 | 100% | 0.71 | +0.7% | + |
| S3/10 | GLS | 5992.6 | 6195.1 ± 794.9 | 100% | 0.92 | +13.3% | + |
| S3/10 | DPF-PSO | 5193.1 | 5232.8 ± 110.2 | 100% | — | — | — |
| S4/10 | PSO | 7735.5 | 7701.8 ± 650.1 | 100% | 0.93 | +11.2% | + |
| S4/10 | SPSO | 7661.8 | 7719.1 ± 341.3 | 100% | 1.00 | +10.3% | + |
| S4/10 | GWO | 8627.0 | 8691.0 ± 834.1 | 100% | 1.00 | +20.4% | + |
| S4/10 | WOA | 8499.6 | 9154.7 ± 1453.3 | 100% | 1.00 | +19.2% | + |
| S4/10 | HHO | 8137.5 | 8945.2 ± 1693.1 | 100% | 1.00 | +15.6% | + |
| S4/10 | DE | 7089.6 | 7159.7 ± 291.4 | 100% | 0.87 | +3.1% | + |
| S4/10 | L-SHADE | 7694.1 | 7733.2 ± 365.2 | 100% | 1.00 | +10.7% | + |
| S4/10 | CMA-ES | 7857.6 | 8073.9 ± 873.5 | 100% | 0.95 | +12.6% | + |
| S4/10 | IDP | 6912.5 | 6908.8 ± 57.4 | 100% | 0.67 | +0.6% | + |
| S4/10 | GLS | 7140.4 | 7207.1 ± 250.5 | 100% | 0.93 | +3.8% | + |
| S4/10 | DPF-PSO | 6870.7 | 6866.3 ± 89.5 | 100% | — | — | — |
| S5/10 | PSO | 6252.5 | 6307.1 ± 773.5 | 100% | 0.88 | +11.5% | + |
| S5/10 | SPSO | 6015.4 | 6233.2 ± 537.8 | 100% | 1.00 | +8.0% | + |
| S5/10 | GWO | 6830.1 | 6969.8 ± 803.1 | 100% | 1.00 | +19.0% | + |
| S5/10 | WOA | 8722.1 | 8361.4 ± 1267.4 | 100% | 1.00 | +36.6% | + |
| S5/10 | HHO | 8434.9 | 7866.3 ± 1110.2 | 100% | 1.00 | +34.4% | + |
| S5/10 | DE | 6175.0 | 6091.1 ± 326.5 | 100% | 0.87 | +10.4% | + |
| S5/10 | L-SHADE | 6293.6 | 6295.5 ± 173.2 | 100% | 0.99 | +12.1% | + |
| S5/10 | CMA-ES | 6665.0 | 7043.8 ± 871.8 | 100% | 1.00 | +17.0% | + |
| S5/10 | IDP | 5611.9 | 5583.5 ± 73.1 | 100% | 0.82 | +1.4% | + |
| S5/10 | GLS | 6005.7 | 6008.3 ± 553.3 | 100% | 0.69 | +7.9% | + |
| S5/10 | DPF-PSO | 5534.0 | 5524.1 ± 61.9 | 100% | — | — | — |
| S6/10 | PSO | 4954.3 | 5178.1 ± 463.3 | 100% | 0.68 | +0.3% | ≈ |
| S6/10 | SPSO | 4973.5 | 4979.9 ± 239.3 | 100% | 0.55 | +0.7% | ≈ |
| S6/10 | GWO | 5486.0 | 5500.6 ± 136.4 | 100% | 1.00 | +10.0% | + |
| S6/10 | WOA | 5790.2 | 5780.2 ± 237.6 | 100% | 1.00 | +14.7% | + |
| S6/10 | HHO | 5491.0 | 5532.9 ± 237.9 | 100% | 1.00 | +10.1% | + |
| S6/10 | DE | 4913.2 | 4910.4 ± 24.1 | 100% | 0.23 | −0.5% | − |
| S6/10 | L-SHADE | 4950.9 | 4958.6 ± 28.4 | 100% | 0.69 | +0.3% | + |
| S6/10 | CMA-ES | 6178.6 | 6024.3 ± 1010.9 | 100% | 0.98 | +20.1% | + |
| S6/10 | IDP | 4985.1 | 4988.8 ± 61.5 | 100% | 0.75 | +0.9% | + |
| S6/10 | GLS | 4965.3 | 4990.6 ± 121.0 | 100% | 0.68 | +0.5% | ≈ |
| S6/10 | DPF-PSO | 4938.5 | 4926.3 ± 53.5 | 100% | — | — | — |

: TABLE S6. Default Configurations, n = 20 (30 Runs)

| Inst. | Method | Median | Mean ± SD | Feasible | A12 | Δ | Sign (Holm) |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/20 | PSO | 6799.1 | 6961.0 ± 732.1 | 100% | 0.98 | +8.4% | + |
| S1/20 | SPSO | 6543.0 | 6556.7 ± 180.4 | 100% | 0.96 | +4.8% | + |
| S1/20 | GWO | 7564.1 | 7623.3 ± 465.7 | 100% | 1.00 | +17.7% | + |
| S1/20 | WOA | 6458.1 | 6572.1 ± 324.7 | 100% | 0.96 | +3.6% | + |
| S1/20 | HHO | 6377.6 | 6437.7 ± 159.1 | 100% | 0.95 | +2.4% | + |
| S1/20 | DE | 6669.8 | 6785.2 ± 317.4 | 100% | 0.98 | +6.6% | + |
| S1/20 | L-SHADE | 7260.8 | 7275.7 ± 44.4 | 100% | 1.00 | +14.2% | + |
| S1/20 | CMA-ES | 6999.7 | 7029.4 ± 311.3 | 100% | 1.00 | +11.0% | + |
| S1/20 | IDP | 6259.0 | 6379.0 ± 295.2 | 100% | 0.63 | +0.5% | ≈ |
| S1/20 | GLS | 6278.4 | 6342.9 ± 149.2 | 100% | 0.75 | +0.8% | + |
| S1/20 | DPF-PSO | 6227.1 | 6245.3 ± 97.7 | 100% | — | — | — |
| S2/20 | PSO | 8850.8 | 9303.8 ± 1133.0 | 100% | 1.00 | +20.0% | + |
| S2/20 | SPSO | 7567.8 | 7646.1 ± 438.0 | 100% | 0.91 | +6.5% | + |
| S2/20 | GWO | 9852.8 | 9726.8 ± 782.7 | 100% | 1.00 | +28.2% | + |
| S2/20 | WOA | 8483.8 | 8497.1 ± 67.9 | 100% | 1.00 | +16.6% | + |
| S2/20 | HHO | 8576.2 | 8590.8 ± 116.9 | 100% | 1.00 | +17.5% | + |
| S2/20 | DE | 7807.4 | 7886.4 ± 465.2 | 100% | 0.96 | +9.3% | + |
| S2/20 | L-SHADE | 8178.0 | 8199.0 ± 107.7 | 100% | 1.00 | +13.4% | + |
| S2/20 | CMA-ES | 8756.2 | 8725.4 ± 443.1 | 100% | 1.00 | +19.2% | + |
| S2/20 | IDP | 7104.9 | 7346.0 ± 346.2 | 100% | 0.76 | +0.4% | + |
| S2/20 | GLS | 7935.1 | 7872.3 ± 325.3 | 100% | 0.99 | +10.8% | + |
| S2/20 | DPF-PSO | 7078.9 | 7102.5 ± 147.9 | 100% | — | — | — |
| S3/20 | PSO | 9075.0 | 19044.1 ± 54131.8 | 97% | 1.00 | +41.5% | + |
| S3/20 | SPSO | 6430.1 | 6463.7 ± 533.6 | 100% | 0.99 | +17.4% | + |
| S3/20 | GWO | 10902.7 | 36737.5 ± 72087.6 | 87% | 1.00 | +51.3% | + |
| S3/20 | WOA | 9063.2 | 9038.1 ± 113.7 | 100% | 1.00 | +41.4% | + |
| S3/20 | HHO | 8909.9 | 8910.2 ± 86.3 | 100% | 1.00 | +40.4% | + |
| S3/20 | DE | 8991.4 | 9095.1 ± 901.9 | 100% | 1.00 | +41.0% | + |
| S3/20 | L-SHADE | 8475.4 | 9263.1 ± 1251.9 | 100% | 1.00 | +37.4% | + |
| S3/20 | CMA-ES | 8860.7 | 9585.0 ± 1303.9 | 100% | 1.00 | +40.1% | + |
| S3/20 | IDP | 5301.0 | 5397.7 ± 185.4 | 100% | 0.36 | −0.1% | ≈ |
| S3/20 | GLS | 5729.7 | 5854.8 ± 537.3 | 100% | 0.90 | +7.4% | + |
| S3/20 | DPF-PSO | 5308.1 | 5390.2 ± 170.0 | 100% | — | — | — |
| S4/20 | PSO | 9585.8 | 19693.9 ± 38034.3 | 93% | 1.00 | +22.6% | + |
| S4/20 | SPSO | 8373.3 | 8463.1 ± 427.1 | 100% | 1.00 | +11.3% | + |
| S4/20 | GWO | 10506.0 | 10643.5 ± 1014.6 | 100% | 1.00 | +29.3% | + |
| S4/20 | WOA | 9758.7 | 10091.1 ± 1285.4 | 100% | 1.00 | +23.9% | + |
| S4/20 | HHO | 8948.6 | 9339.0 ± 1135.0 | 100% | 1.00 | +17.0% | + |
| S4/20 | DE | 7921.1 | 8123.6 ± 471.1 | 100% | 1.00 | +6.3% | + |
| S4/20 | L-SHADE | 8379.2 | 8708.5 ± 564.5 | 100% | 1.00 | +11.4% | + |
| S4/20 | CMA-ES | 8991.8 | 9295.0 ± 938.0 | 100% | 1.00 | +17.4% | + |
| S4/20 | IDP | 7452.8 | 7508.4 ± 171.4 | 100% | 0.73 | +0.4% | + |
| S4/20 | GLS | 8256.1 | 8179.7 ± 366.4 | 100% | 1.00 | +10.1% | + |
| S4/20 | DPF-PSO | 7423.6 | 7417.3 ± 55.0 | 100% | — | — | — |
| S5/20 | PSO | 8157.6 | 8122.8 ± 1284.3 | 100% | 1.00 | +29.8% | + |
| S5/20 | SPSO | 6832.0 | 7091.1 ± 817.2 | 100% | 1.00 | +16.2% | + |
| S5/20 | GWO | 9155.5 | 8832.6 ± 1222.5 | 100% | 1.00 | +37.5% | + |
| S5/20 | WOA | 9754.7 | 9827.6 ± 387.6 | 100% | 1.00 | +41.3% | + |
| S5/20 | HHO | 9282.5 | 9222.1 ± 441.5 | 100% | 1.00 | +38.3% | + |
| S5/20 | DE | 7042.7 | 7073.1 ± 529.7 | 100% | 1.00 | +18.7% | + |
| S5/20 | L-SHADE | 6942.4 | 6961.1 ± 191.9 | 100% | 1.00 | +17.5% | + |
| S5/20 | CMA-ES | 8976.2 | 8676.2 ± 970.2 | 100% | 1.00 | +36.2% | + |
| S5/20 | IDP | 5721.8 | 5969.8 ± 552.3 | 100% | 0.58 | −0.1% | ≈ |
| S5/20 | GLS | 6718.6 | 6834.0 ± 659.0 | 100% | 0.96 | +14.8% | + |
| S5/20 | DPF-PSO | 5725.8 | 5726.5 ± 28.7 | 100% | — | — | — |
| S6/20 | PSO | 5540.6 | 5808.2 ± 1037.2 | 100% | 0.86 | +6.8% | + |
| S6/20 | SPSO | 5500.7 | 5451.0 ± 227.4 | 100% | 0.86 | +6.1% | + |
| S6/20 | GWO | 6224.4 | 6369.0 ± 535.0 | 100% | 1.00 | +17.1% | + |
| S6/20 | WOA | 6337.0 | 6968.8 ± 1579.2 | 100% | 1.00 | +18.5% | + |
| S6/20 | HHO | 6272.1 | 6265.9 ± 334.2 | 100% | 1.00 | +17.7% | + |
| S6/20 | DE | 5181.5 | 5347.4 ± 340.9 | 100% | 0.66 | +0.4% | ≈ |
| S6/20 | L-SHADE | 5166.4 | 5222.2 ± 141.2 | 100% | 0.65 | +0.1% | ≈ |
| S6/20 | CMA-ES | 6972.1 | 6691.8 ± 1048.1 | 100% | 0.88 | +26.0% | + |
| S6/20 | IDP | 5163.6 | 5140.5 ± 42.5 | 100% | 0.47 | +0.0% | ≈ |
| S6/20 | GLS | 5526.7 | 5655.4 ± 544.4 | 100% | 0.82 | +6.6% | + |
| S6/20 | DPF-PSO | 5162.5 | 5154.9 ± 78.0 | 100% | — | — | — |

::: {custom-style="TableNote"}
Δ, A~12~ and sign refer to DPF-PSO against the method in the row; Holm over the 10 methods of each instance.
:::

: TABLE S7. Tuned Configurations on S1–S6 (Median, 30 Runs; Reference: DPF-PSO-T)

| Inst. | DE | L-SH. | SPSO | IDP | DPF | CMA |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/10 | 6142 + | 6046 − | **5979 −** | 6097 + | 6068 | 6532 + |
| S2/10 | 6568 + | 7278 + | 6550 + | 6430 + | **6397** | 7454 + |
| S3/10 | 5429 + | 7654 + | 5332 + | 5218 + | **5190** | 7679 + |
| S4/10 | 7387 + | 7479 + | 7316 + | 6904 + | **6866** | 7858 + |
| S5/10* | **5531 ≈** | 6183 + | 5673 + | 5606 + | 5543 | 6665 + |
| S6/10* | 4967 + | 4912 ≈ | **4632 −** | 4962 + | 4931 | 6179 + |
| S1/20 | 6601 + | 7213 + | 6255 ≈ | 6259 + | **6183** | 7000 + |
| S2/20 | 8017 + | 8033 + | 7161 + | 7101 + | **7058** | 8756 + |
| S3/20 | 6105 + | 9461 + | 5986 + | 5297 ≈ | **5294** | 8861 + |
| S4/20 | 8695 + | 8771 + | 7941 + | 7455 + | **7402** | 8992 + |
| S5/20* | 6399 + | 6708 + | 6291 + | 5718 + | **5696** | 8976 + |
| S6/20* | 5232 + | **5103 ≈** | 5292 ≈ | 5139 ≈ | 5149 | 6972 + |
| +/≈/− | 11/1/0 | 9/2/1 | 8/2/2 | 10/2/0 | — | 12/0/0 |
| Rank | 3.92 | 4.33 | 2.83 | 2.50 | 1.58 | 5.83 |

# S6. Additional Experiments

*Public scenario C1.* The scenario of the public SPSO code: the Christmas Island elevation model (1 m per pixel as in that code, bilinear interpolation instead of rounding to the nearest pixel), its six threats and its start and goal. Start and goal heights are 150 m above ground.

: TABLE S8. C1, Default Configurations (Median, 30 Runs)

| Inst. | PSO | SPSO | GWO | WOA | HHO | DE | L-SH. | CMA | IDP | GLS | DPF |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| C1/10 | 5486 +^†^ | 5633 + | 5762 + | 6162 + | 5851 + | 5321 + | 5378 + | 5594 + | 4778 + | 5385 + | **4759** |
| C1/20 | 6708 + | 5646 + | 6578 +^†^ | 6499 + | 5998 + | 5846 + | 6209 + | 6616 +^†^ | 5567 + | 5391 ≈ | **5316** |
| +/≈/− | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 2/0/0 | 1/1/0 | — |
| Rank | 8.50 | 6.00 | 9.00 | 9.50 | 8.00 | 4.00 | 5.50 | 8.50 | 2.50 | 3.50 | 1.00 |

: TABLE S9. S1–S6 with n = 30 (Median, 20 Runs, Budget 90,000 FEs)

| Inst. | PSO | SPSO | GWO | WOA | HHO | DE | L-SH. | CMA | IDP | GLS | DPF |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/30 | 9080 + | 7335 + | 8290 +^†^ | 6570 + | 6423 ≈ | 7123 + | 7823 + | 7390 + | 6663 ≈ | 6413 ≈ | **6365** |
| S2/30 | 10588 +^†^ | 8091 + | 11242 +^†^ | 8566 + | 8819 + | 9429 + | 9971 + | 9259 + | 7915 + | 8431 + | **7234** |
| S3/30 | 14857 +^†^ | 7054 + | 132854 +^†^ | 9228 + | 9159 + | 9629 + | 11988 + | 10679 + | 5743 ≈ | 6488 + | **5505** |
| S4/30 | 12564 +^†^ | 9534 + | 12250 +^†^ | 9743 + | 9705 + | 9545 + | 9899 + | 10741 +^†^ | **7566 ≈** | 9084 + | 7621 |
| S5/30 | 10708 +^†^ | 8416 + | 10764 +^†^ | 10045 + | 9803 + | 7683 + | 8154 + | 9640 + | 5901 ≈ | 7283 + | **5815** |
| S6/30 | 7179 + | 6024 + | 6968 + | 6138 + | 6572 + | 5793 + | 5507 + | 8030 +^†^ | **5217 −** | 6232 + | 5329 |
| +/≈/− | 6/0/0 | 6/0/0 | 6/0/0 | 6/0/0 | 5/1/0 | 6/0/0 | 6/0/0 | 6/0/0 | 1/4/1 | 5/1/0 | — |
| Rank | 10.33 | 4.83 | 10.33 | 6.17 | 6.00 | 5.67 | 7.17 | 8.33 | 2.17 | 3.67 | 1.33 |

: TABLE S10. S1–S6 with a Five Times Larger Budget (Median, 10 Runs)

| Inst. | PSO | SPSO | GWO | WOA | HHO | DE | L-SH. | CMA | IDP | GLS | DPF |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/10 | 6309 + | **5988 ≈** | 6496 + | 6247 + | 6219 + | 6005 − | 6033 ≈ | 6326 ≈ | 6088 ≈ | 6088 + | 6059 |
| S2/10 | 7218 + | 6626 ≈ | 7257 + | 8240 + | 7757 + | **6367 ≈** | 7259 + | 7243 + | 6395 ≈ | 6692 + | 6386 |
| S3/10 | 7546 + | 5433 + | 5995 + | 8616 + | 8268 + | 7066 + | 7717 + | 7403 + | 5186 ≈ | 5196 ≈ | **5168** |
| S4/10 | 7354 + | 7434 + | 8081 + | 9097 + | 7747 + | 7233 + | 7386 ≈ | 7703 + | **6857 ≈** | 6879 ≈ | 6869 |
| S5/10 | 5563 ≈ | 5686 + | 6819 + | 7892 + | 8273 + | 6146 + | 6148 + | 6437 + | 5526 ≈ | **5465 ≈** | 5534 |
| S6/10 | 4949 ≈ | **4605 −^†^** | 5555 + | 5511 + | 5325 + | 4905 ≈ | 4902 − | 5028 ≈ | 4930 ≈ | 4901 ≈ | 4933 |
| S1/20 | 6713 + | 6364 ≈ | 7731 + | 6341 + | 6265 ≈ | 6712 + | 6490 + | 6698 + | 6585 ≈ | **6201 ≈** | 6255 |
| S2/20 | 9019 + | 7330 ≈ | 9199 + | 8478 + | 8464 + | 7745 + | 7930 + | 8511 + | 7074 ≈ | 7510 + | **7063** |
| S3/20 | 9872 +^†^ | 5850 + | 10465 +^†^ | 9059 + | 8849 + | 8967 + | 8347 + | 8299 + | **5295 −** | 5353 ≈ | 5320 |
| S4/20 | 9715 +^†^ | 7896 + | 10253 + | 9627 + | 8512 + | 7871 + | 8278 + | 8629 + | **7387 ≈** | 7870 + | 7417 |
| S5/20 | 8184 + | 6057 + | 7815 + | 9620 + | 9031 + | 6883 + | 6585 + | 8446 + | 5767 ≈ | 5907 + | **5708** |
| S6/20 | 5600 ≈ | **5060 ≈** | 6137 + | 5974 + | 5830 + | 5470 ≈ | 5071 − | 6718 ≈ | 5158 ≈ | 5125 ≈ | 5176 |
| +/≈/− | 9/3/0 | 6/5/1 | 12/0/0 | 12/0/0 | 11/1/0 | 8/3/1 | 8/2/2 | 9/3/0 | 0/11/1 | 5/7/0 | — |
| Rank | 7.83 | 3.67 | 9.58 | 9.25 | 8.17 | 5.08 | 5.67 | 8.17 | 2.92 | 3.08 | 2.58 |

: TABLE S11. S1–S6 with Danger Weight b~2~ = 10 (Median, 10 Runs)

| Inst. | PSO | SPSO | GWO | WOA | HHO | DE | L-SH. | CMA | IDP | GLS | DPF |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/10 | 6349 + | 6194 ≈ | 6743 + | 6424 + | 6342 + | **6022 −** | 6073 ≈ | 6506 + | 6111 ≈ | 6175 + | 6074 |
| S2/10 | 7321 + | 6797 ≈ | 7851 + | 8277 + | 8065 + | **6443 ≈** | 7297 + | 7356 + | 6479 ≈ | 6963 + | 6464 |
| S3/10 | 7663 + | 5799 + | 8177 + | 8625 + | 8320 + | 7249 + | 7741 + | 7661 + | 5284 ≈ | 6209 + | **5233** |
| S4/10 | 7564 + | 7704 + | 8410 + | 8956 + | 8524 + | 7308 ≈ | 7835 + | 7468 ≈ | **6956 ≈** | 7370 + | 6964 |
| S5/10 | 6053 ≈ | 5792 + | 7233 + | 8681 + | 8496 + | 6327 + | 6387 + | 7144 + | 5612 ≈ | 5579 ≈ | **5553** |
| S6/10 | 5041 ≈ | 4988 ≈ | 5559 + | 5824 + | 5735 + | **4936 −** | 4968 ≈ | 6612 + | 5006 ≈ | 5055 ≈ | 4976 |
| S1/20 | 7076 + | 6566 + | 7885 + | 6426 + | 6385 + | 6669 + | 7292 + | 7116 + | 6519 + | 6336 + | **6254** |
| S2/20 | 8824 + | 7602 + | 10031 + | 8493 + | 8633 + | 7528 + | 8147 + | 8782 + | 7114 ≈ | 8012 + | **7100** |
| S3/20 | 9006 + | 6654 + | 12233 +^†^ | 9079 + | 8911 + | 9262 + | 8481 + | 8826 + | 5369 ≈ | 5816 + | **5362** |
| S4/20 | 9415 +^†^ | 8657 + | 10846 + | 10140 + | 9168 + | 8294 + | 8712 + | 8901 + | 7463 ≈ | 8309 + | **7451** |
| S5/20 | 8278 + | 7146 + | 8128 + | 9778 + | 9398 + | 7284 + | 7041 + | 9013 + | 5906 ≈ | 6841 + | **5724** |
| S6/20 | 6153 + | 5348 ≈ | 6404 + | 6876 + | 6253 + | 5263 ≈ | **5233 ≈** | 6776 + | 5249 ≈ | 5487 ≈ | 5235 |
| +/≈/− | 10/2/0 | 8/4/0 | 12/0/0 | 12/0/0 | 12/0/0 | 7/3/2 | 9/3/0 | 11/1/0 | 1/11/0 | 9/3/0 | — |
| Rank | 7.42 | 4.75 | 9.58 | 9.58 | 8.33 | 4.17 | 5.42 | 8.17 | 2.83 | 4.17 | 1.58 |

: TABLE S12. Equal CPU Time: Baselines with Multiplied Budgets against DPF-PSO with Its Normal Budget (Median, 30 Runs)

| Inst. | PSO | SPSO | GWO | WOA | HHO | DE | L-SH. | CMA | IDP | GLS | DPF |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S1/10 | 6310 + | 6050 ≈ | 6593 + | 6330 + | 6308 + | **6006 −** | 6065 ≈ | 6532 + | 6093 + | 6157 + | 6076 |
| S2/10 | 7177 + | 6719 + | 7571 + | 8240 + | 7907 + | **6368 −** | 7285 + | 7454 + | 6415 + | 6959 + | 6402 |
| S3/10 | 7216 + | 5657 + | 7481 + | 8619 + | 8312 + | 7067 + | 7723 + | 7679 + | 5212 ≈ | 5993 + | **5193** |
| S4/10 | 7517 + | 7638 + | 8564 + | 9056 + | 8063 + | 6995 + | 7582 + | 7858 + | 6890 ≈ | 7140 + | **6871** |
| S5/10 | 6178 + | 5782 + | 6835 + | 8531 + | 8385 + | 6150 + | 6335 + | 6665 + | 5590 + | 6006 + | **5534** |
| S6/10 | 4950 ≈ | **4870 ≈^†^** | 5536 + | 5654 + | 5496 + | 4905 − | 4916 ≈ | 6179 + | 4984 + | 4965 ≈ | 4938 |
| S1/20 | 6844 + | 6469 + | 7377 + | 6378 + | 6339 + | 6558 + | 7261 + | 7000 + | 6261 ≈ | 6278 + | **6227** |
| S2/20 | 9382 + | 7278 + | 9540 + | 8479 + | 8551 + | 7784 + | 8178 + | 8756 + | 7088 + | 7935 + | **7079** |
| S3/20 | 9003 + | 6330 + | 10724 +^†^ | 9059 + | 8904 + | 8955 + | 8475 + | 8861 + | **5299 −** | 5730 + | 5308 |
| S4/20 | 9416 +^†^ | 8504 + | 10636 + | 9629 + | 8891 + | 7891 + | 8379 + | 8992 + | 7444 ≈ | 8256 + | **7424** |
| S5/20 | 8189 +^†^ | 6513 + | 8655 + | 9590 + | 9223 + | 7017 + | 6942 + | 8976 + | 5758 ≈ | 6719 + | **5726** |
| S6/20 | 5787 + | 5395 + | 6198 + | 6125 + | 6096 + | **5125 ≈** | 5166 ≈ | 6972 + | 5167 ≈ | 5527 + | 5162 |
| +/≈/− | 11/1/0 | 10/2/0 | 12/0/0 | 12/0/0 | 12/0/0 | 8/1/3 | 9/3/0 | 12/0/0 | 5/6/1 | 11/1/0 | — |
| Rank | 7.17 | 3.92 | 9.75 | 9.58 | 8.17 | 3.83 | 5.75 | 8.75 | 2.83 | 4.50 | 1.75 |

# S7. Population-Size Sensitivity

: TABLE S13. DPF-PSO against DE and SPSO with Other Population Sizes (S1–S6, 30 Runs)

| Variant | +/≈/− (DPF-PSO vs. variant) | Δ (mean over 12 instances) |
|:--|:-:|:-:|
| DE, NP = 5D | 12/0/0 | +40.9% |
| DE, NP = 10D | 12/0/0 | +53.1% |
| SPSO, 100 particles | 11/1/0 | +8.3% |
| SPSO, 500 particles | 10/2/0 | +6.7% |

# S8. Composition of the FE Budget of DPF-PSO

: TABLE S14. FE Composition, Number and Size of Fusions per Run (Mean over S1–S6 and 30 Runs)

| n | Host FEs | Global fusion | Local fusion | Global fusions | Local fusions | Mean K (global) | Mean K (local) | Improving global / local |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 10 | 36% | 52% | 12% | 2.0 | 42 | 97 | 10.2 | 2.0 / 40.1 |
| 20 | 36% | 51% | 13% | 4.0 | 86 | 92 | 9.7 | 4.0 / 78.2 |

::: {custom-style="TableNote"}
Host FEs: FEs of the PSO moves and the initial swarm. Mean K: mean candidate-set size per slot after removing duplicates. Improving: fusions per run whose result became the new best.
:::

# S9. Timing

CPU times come from a separate benchmark (`src/timing_benchmark.py`): 8 worker processes, each pinned to one logical processor of a different performance core of the i7-14650HX (the efficiency cores were not used), 5 runs per method and instance on S1–S6, methods interleaved. The equal-time multiplier of a baseline is the ratio of the CPU time of DPF-PSO to its own, rounded to one decimal and at least 1.0.

| Method | CPU s, n = 10 | CPU s, n = 20 | Wall s, n = 10 | Wall s, n = 20 | Multiplier (10 / 20) |
|:--|:-:|:-:|:-:|:-:|:-:|
| PSO | 1.07 ± 0.19 | 3.78 ± 0.87 | 1.17 ± 0.23 | 4.74 ± 2.08 | 1.9 / 1.7 |
| SPSO | 1.20 ± 0.21 | 3.79 ± 0.75 | 1.28 ± 0.20 | 4.87 ± 2.39 | 1.7 / 1.7 |
| GWO | 1.07 ± 0.20 | 3.56 ± 0.71 | 1.18 ± 0.30 | 4.72 ± 2.79 | 1.9 / 1.8 |
| WOA | 1.06 ± 0.19 | 3.34 ± 0.68 | 1.15 ± 0.19 | 4.36 ± 2.61 | 1.9 / 2.0 |
| HHO | 1.56 ± 0.28 | 4.07 ± 0.72 | 1.67 ± 0.28 | 5.08 ± 2.54 | 1.3 / 1.6 |
| DE | 1.12 ± 0.19 | 3.66 ± 0.70 | 1.20 ± 0.20 | 4.46 ± 2.39 | 1.8 / 1.8 |
| L-SHADE | 1.40 ± 0.24 | 7.00 ± 1.29 | 1.49 ± 0.24 | 8.68 ± 3.80 | 1.4 / 1.0 |
| CMA-ES | 5.02 ± 0.74 | 11.24 ± 1.92 | 5.42 ± 0.72 | 13.88 ± 5.65 | 1.0 / 1.0 |
| IDP | 1.27 ± 0.23 | 4.19 ± 0.73 | 1.35 ± 0.24 | 5.04 ± 1.88 | 1.6 / 1.6 |
| GLS | 6.03 ± 0.84 | 20.84 ± 3.50 | 6.44 ± 0.88 | 25.32 ± 11.07 | 1.0 / 1.0 |
| DPF-PSO | 1.99 ± 0.31 | 6.57 ± 1.22 | 2.11 ± 0.30 | 7.91 ± 2.58 | 1.0 / 1.0 (reference) |

# S10. Safety Check

Every final path of every experiment was re-checked by `src/verify_paths.py`, which does not use the cost function: threat distances are computed with an analytic point-to-segment distance in a plain loop, and ground clearance with the independent exact implementation of S1. A path is independently feasible if it keeps a horizontal distance of at least $R_k+D$ from every threat centre and an exact clearance of at least 20 m (tolerance $10^{-6}$ m). In the experiment with the altitude band enforced along the segments (4,800 of the paths in Table S15), a path must in addition stay at least 100 m and at most 200 m above the ground along every segment; the maximum height is computed by the same independent implementation applied to the mirrored terrain $-Z$. In this variant the cost function ignores band violations of up to $10^{-6}$ m (S4), and optimizers can end exactly on that tolerance: for 29 paths that the cost function accepted, the check computed a band violation slightly above $10^{-6}$ m, at most $1.000000282\times10^{-6}$ m. Because two independent floating-point implementations cannot agree on which side of the same threshold such a path lies, the check accepts band violations of up to $10^{-6}$ m plus $10^{-9}$ m for rounding. With the check's first setting, an identical tolerance of $10^{-6}$ m, these 29 paths were reported as disagreements.

: TABLE S15. Independent Check of All Final Paths

| Experiment | Paths | Independently feasible | Disagreements | Reported feasible but infeasible | Min. exact clearance (m) | Max. height above ground (m) |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| S1–S6 | 11,160 | 11,126 | 0 | 0 | 20.000 | — |
| H01–H40 | 15,200 | 15,164 | 0 | 0 | 20.000 | — |
| C1 | 960 | 956 | 0 | 0 | 85.344 | — |
| Equal CPU time | 3,600 | 3,593 | 1 | 0 | 20.000 | — |
| n = 30 | 1,320 | 1,280 | 0 | 0 | 20.000 | — |
| 5× budget | 1,320 | 1,317 | 1 | 0 | 20.000 | — |
| b~2~ = 10 | 1,320 | 1,315 | 0 | 0 | 20.000 | — |
| H01–H40, equal CPU time | 2,400 | 2,400 | 0 | 0 | 20.000 | — |
| H01–H40, band along segments | 4,800 | 4,799 | 0 | 0 | 100.000 | 200.000 |

Disagreements: SPSO on S6/10, run 18: exact clearance 20.000000000000 m, threat margin 0.024 m, rejected by the cost function and accepted by the check (tolerance 10^−6^ m); SPSO on S6/10, run 9: exact clearance 20.000000000000 m, threat margin 3.748 m, rejected by the cost function and accepted by the check (tolerance 10^−6^ m).

: TABLE S16. Feasible Paths, Clearance and Boundary Contact per Method (S1–S6, Default Weights)

| Method | Feasible paths | Min. clearance (m) | Paths within 1 mm of R+D |
|:--|:-:|:-:|:-:|
| PSO | 357/360 | 20.000 | 6 (2%) |
| SPSO | 360/360 | 20.000 | 164 (46%) |
| GWO | 356/360 | 20.023 | 1 (0%) |
| WOA | 360/360 | 20.000 | 119 (33%) |
| HHO | 360/360 | 20.002 | 8 (2%) |
| DE | 360/360 | 20.003 | 4 (1%) |
| L-SHADE | 360/360 | 20.000 | 17 (5%) |
| CMA-ES | 360/360 | 20.001 | 6 (2%) |
| IDP | 360/360 | 20.001 | 16 (4%) |
| GLS | 360/360 | 20.000 | 72 (20%) |
| DPF-PSO | 360/360 | 20.000 | 92 (26%) |

: TABLE S17. As Table S16 with Danger Weight b~2~ = 10

| Method | Feasible paths | Min. clearance (m) | Paths within 1 mm of R+D |
|:--|:-:|:-:|:-:|
| PSO | 119/120 | 20.001 | 0 (0%) |
| SPSO | 120/120 | 20.000 | 3 (2%) |
| GWO | 116/120 | 20.090 | 0 (0%) |
| WOA | 120/120 | 20.000 | 17 (14%) |
| HHO | 120/120 | 20.000 | 0 (0%) |
| DE | 120/120 | 20.017 | 0 (0%) |
| L-SHADE | 120/120 | 20.001 | 0 (0%) |
| CMA-ES | 120/120 | 20.006 | 0 (0%) |
| IDP | 120/120 | 20.004 | 0 (0%) |
| GLS | 120/120 | 20.000 | 2 (2%) |
| DPF-PSO | 120/120 | 20.000 | 0 (0%) |

# S11. Figures

![Fig. S1. Median relative gap to the best cost found on the instance by any run, versus FEs, for all 12 instances of S1–S6.](../figures/fig_conv_all.png){width=170mm}

![Fig. S2. Best paths (top view) on S1–S6 with n = 20.](../figures/fig_paths.png){width=170mm}

![Fig. S3. Best paths on the public scenario C1.](../figures/fig_paths_public.png){width=170mm}

![Fig. S4. Altitude profiles of the best paths of tuned SPSO and tuned DPF-PSO on S6 with n = 10, each along its own horizontal track, with the terrain under the track and the 20 m clearance limit. Tuned SPSO merges waypoints and flies long straight segments down to the clearance limit; the slot encoding places one waypoint 100–200 m above the ground in every slot.](../figures/fig_profile.png){width=110mm}
