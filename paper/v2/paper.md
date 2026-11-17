::: {custom-style="Title"}
Dynamic-Programming Path Fusion: Optimal Recombination for Swarm-Based 3-D UAV Path Planning
:::

::: {custom-style="Author"}
Anonymous Authors\
(author names and affiliations removed for double-blind review)
:::

[[COLS2]]

::: {custom-style="Abstract"}
*Abstract*—Swarm optimizers for three-dimensional (3-D) path planning of unmanned aerial vehicles (UAVs) usually treat the path cost as a black box, although the cost model of Phung and Ha is a sum of terms that each involve at most three consecutive waypoints. We exploit this structure with dynamic-programming path fusion (DPF), an optimal recombination operator that returns, in $O(nK^3)$ time, the least-cost path that can be assembled from $K$ candidate positions per waypoint taken from the swarm memory, perturbations and random samples; the result is never worse than any parent. Embedded in particle swarm optimization, it gives DPF-PSO. The evaluation uses an exactly computed ground clearance, an independent safety check of all final paths, equal tuning budgets, structure-exploiting baselines and 40 held-out scenarios. On these scenarios, DPF-PSO had significantly lower median costs than all black-box baselines with the same waypoint encoding, including DE, L-SHADE and CMA-ES (median gaps of 9.6% to 25.8% for 20 waypoints). Iterative dynamic programming, which applies the same dynamic program to random candidates, came within 1.1%, so most of the improvement comes from the dynamic program rather than from the swarm. Spherical-vector PSO, whose encoding can merge waypoints, was the strongest black-box competitor and, after tuning, better than DPF-PSO for 10 waypoints; with a five times larger budget, the black-box methods closed much of the gap. As a plug-in, DPF lowered the median cost of six host algorithms on the benchmark scenarios by 9.8% to 23.1% on average.
:::

::: {custom-style="Keywords"}
*Keywords*—UAV path planning, particle swarm optimization, dynamic programming, optimal recombination, gray-box optimization
:::

# I. Introduction

Path planning lets an unmanned aerial vehicle (UAV) reach its goal while avoiding terrain and threats and respecting its flight limits [1]. In 3-D environments with continuous waypoint coordinates, the problem is usually written as the minimization of a weighted cost over a fixed number of waypoints and solved with population-based metaheuristics, which cope with the non-convex landscape created by obstacles [2]. Particle swarm optimization (PSO) [3] and its variants are among the most used methods, from phase-angle-encoded quantum-behaved PSO [4] to spherical-vector PSO (SPSO) [5], which encodes each segment by its length, elevation and azimuth, and its recent extensions [6]. New metaheuristics are regularly evaluated on this problem [7]–[10], although the value of many metaphor-based designs has been questioned [11].

Most of these methods treat the path cost as a black box, although the cost model of [5], variants of which are used in later work [6], is a sum of terms that depend on one waypoint (altitude), two consecutive waypoints (length, threats, clearance) or three (turning and climbing angles). Gray-box optimization exploits such structure inside evolutionary search [12]; partition crossover, for example, returns the best of exponentially many offspring in linear time [13], [14]. Recombination that returns the best offspring that can be assembled from the parents' genes is called optimal or dynastically optimal recombination [15], [16], and recombination from the whole population gene-pool recombination [17]. Dynamic programming over candidate sets with a three-point term is classical for active contours [18]; driving planners select paths on lattices in station–lateral (Frenét) coordinates by dynamic programming [19], [20]; and iterative dynamic programming solves dynamic programs over candidates sampled in a shrinking region around the current best [21]. We combine these ideas in a different setting: the candidate sets come from the memory of a continuous population-based optimizer, the dynamic program is a plug-in recombination operator, and structure-exploiting competitors separate the effect of the swarm from that of the dynamic program.

This paper makes three contributions:

1) Dynamic-programming path fusion (DPF), an optimal recombination operator for waypoint-based path encodings. Given candidate positions for each waypoint, taken from the swarm memory, from perturbations and from random samples, DPF returns in $O(nK^3)$ time the least-cost path that can be assembled from them; the result is never worse than any parent, and an exact pruning rule reduces the computation.

2) DPF-PSO, which fuses the whole swarm memory with random samples (global fusion) and the best paths with Gaussian perturbations (local fusion), and a plug-in interface for population-based optimizers that use the same slot encoding.

3) An evaluation with exact ground clearance and an independent safety check, 40 held-out scenarios and the public scenario of [5], structure-exploiting baselines, CMA-ES, equal tuning budgets, and comparisons at equal evaluation budget and at equal CPU time.

# II. Problem Formulation

## A. Environment and Path Representation

The workspace is a rectangular area with terrain elevation $Z(x,y)$ given on a grid of spacing $\delta$ with bilinear interpolation and $N_t$ vertical cylindrical threats with centres $\mathbf{c}_k$ and radii $R_k$. A path consists of the start $S$, waypoints $P_1,\dots,P_n$ and the goal $G$. Let $L$ be the horizontal distance between $S$ and $G$, $\mathbf{e}_1$ the horizontal unit vector from $S$ to $G$ and $\mathbf{e}_2$ its left normal. The segment $SG$ is divided into $n$ slots centred at $B_j=S+j\Delta s\,\mathbf{e}_1$ with $\Delta s=L/(n+1)$, and waypoint $j$ is placed at

$$P_j^{xy}=B_j+\tau_j\Delta s\,\mathbf{e}_1+u_j\mathbf{e}_2,\qquad z_j=Z(P_j^{xy})+h_j$$

with along-track offset $\tau_j\in[-0.45,0.45]$, height above ground $h_j\in[100,200]$ m and lateral offset $u_j\in[-400,400]$ m, narrowed so that $B_j+u_j\mathbf{e}_2$ stays inside the map; a point moved outside the map by $\tau_j$ is clipped to it. The decision vector $\mathbf{x}=(u_j,h_j,\tau_j)_{j=1}^{n}$ has $3n$ variables, as in SPSO. Because the slots do not overlap, waypoints stay ordered along $SG$; this is the station–lateral parametrization of lattice planners [19], [20].

## B. Cost Function and Exact Ground Clearance

We use the cost model of Phung and Ha [5] with the weights of their public code, $b_1=5$, $b_2=1$, $b_3=10$, $b_4=1$:

$$J=b_1F_1+b_2F_2+b_3F_3+b_4F_4+\Pi$$

$F_1$ is the 3-D path length. $F_2=\sum_{j,k}\min\{S_d,(R_k+D+S_d-d_{jk})^+\}$ measures how deep segment $j$ enters the danger zone of width $S_d=10$ m around threat $k$, where $d_{jk}$ is the horizontal distance from $\mathbf{c}_k$ to the segment and $D=1$ m the UAV size. $F_3=\sum_j|h_j-150|$ penalizes deviations from the centre of the altitude band. $F_4$ sums, over the waypoints, the turning angle between the horizontal projections of the two segments that meet there and the change of their climb angles, in degrees, as in Eq. (8) of [5] with unit coefficients (the public code of [5] counts an angle only above 45°). Instead of the infinite cost of [5], a segment adds $M_0=10^5$ to $\Pi$ for every inflated threat radius $R_k+D$ it enters and once more if its minimum ground clearance $c_{\min}$ is below $h_c=20$ m, plus $\lambda=10^3$ times the sum of the penetration depths and of the shortfall $(h_c-c_{\min})^+$.

The model of [5] checks the terrain only at the waypoints; we enforce the clearance along every segment and compute $c_{\min}$ exactly. Inside a grid cell, the bilinear terrain along a segment is quadratic in the segment parameter and the segment altitude is linear, so the minimum clearance per cell has a closed form; pieces of a segment whose conservative bound already exceeds $h_c$ are skipped. The result agrees with an independent implementation to within $5\times10^{-13}$ m on 7,200 random segments (supplement S1).

With $P_0=S$ and $P_{n+1}=G$, every term of $J$ involves at most three consecutive points:

$$J=\sum_{j=1}^{n}A(P_j)+\sum_{j=0}^{n}E(P_j,P_{j+1})+\sum_{j=1}^{n}T(P_{j-1},P_j,P_{j+1})$$

The altitude term $A$ holds $b_3F_3$, the edge term $E$ the length, danger and violation terms, and the turn term $T$ the smoothness terms. $E$ needs terrain and threat computations and dominates the cost of an evaluation; $T$ is cheap arithmetic.

# III. Dynamic-Programming Path Fusion

## A. Fusion Problem and Recursion

Assume that for every slot $j$ a finite set $C_j$ of $K_j$ candidate waypoints is given, with $C_0=\{S\}$, $C_{n+1}=\{G\}$ and $K_0=K_{n+1}=1$. The fusion problem asks for the least-cost path whose $j$-th waypoint belongs to $C_j$ for every $j$; exhaustive search would examine $\prod_jK_j$ paths ($10^{40}$ for $n=20$, $K_j=100$). Because $T$ couples three consecutive points, the state of the dynamic program is the pair of the last two waypoints. Let $W_j(q,r)$, $q\in C_{j-1}$, $r\in C_j$, be the least cost of a partial path ending with $q,r$, including all terms up to $r$ except the turn at $r$. Then $W_1(S,r)=E(S,r)+A(r)$ and, for $j=2,\dots,n+1$,

$$W_j(q,r)=E(q,r)+A(r)+\min_{p\in C_{j-2}}\big[W_{j-1}(p,q)+T(p,q,r)\big]$$

with $A(G)=0$; the optimal cost is $J^*=\min_qW_{n+1}(q,G)$, and the minimizing $p$ stored for every $(q,r)$ gives the path by backtracking (Fig. 1).

*Remark 1:* $J^*$ is the least cost over all $\prod_jK_j$ assemblable paths and is not larger than the cost of any parent whose waypoints are in the candidate sets, because no term involves more than three consecutive points (principle of optimality [22]) and every parent is an assemblable path. DPF is therefore an optimal recombination operator in the sense of [15], [16], restricted to the candidate sets; it does not solve the continuous problem exactly.

![Fig. 1. DPF on a two-dimensional toy problem with five slots (cost: length + turning penalty + collision penalty). Grey lines: edges evaluated by the dynamic program. The fused path (red) combines waypoints of parents 2 and 3 with one perturbation, avoids the threat, and costs 10.6, against 13.9 for the best parent.](../figures/fig_dpf_schematic.png){width=76mm}

## B. Pruning, Cost Accounting and Pseudo-Code

Each angle in $F_4$ lies in $[0^\circ,180^\circ]$, so $0\le T\le T_{\max}=360\,b_4$. A predecessor $p$ with $W_{j-1}(p,q)>\min_{p'}W_{j-1}(p',q)+T_{\max}$ can never attain the minimum in (4) for any $r$ and is skipped (Algorithm 1, line 5). This pruning is exact; it returned bit-identical results in 324 random test cases and reduced the time of one fusion with $n=20$ and $K_j=100$ by a factor of 1.7. A fusion evaluates $\sum_jK_jK_{j+1}$ edge terms and $\sum_jK_{j-1}K_jK_{j+1}$ turn terms. Because edge terms dominate the cost of an evaluation, a fusion is charged $\lceil\sum_jK_jK_{j+1}/(n+1)\rceil$ function evaluations (FEs), the number of segments evaluated divided by the number of segments of one path; turn terms are not charged, and their cost appears in the CPU-time comparison (Section IV-F).

| **Algorithm 1:** DPF$(C_1,\dots,C_n)$ |
|:--|
| 1: compute $A(r)$ for all candidates and $E(q,r)$ for consecutive slots |
| 2: $W_1(S,r)\leftarrow E(S,r)+A(r)$ for all $r\in C_1$ |
| 3: **for** $j=2,\dots,n+1$ **do** |
| 4:   **for** each $q\in C_{j-1}$: $m_q\leftarrow\min_pW_{j-1}(p,q)$ |
| 5:     $\Pi_q\leftarrow\{p: W_{j-1}(p,q)\le m_q+T_{\max}\}$ (exact pruning) |
| 6:     **for** each $r\in C_j$: $W_j(q,r)\leftarrow E(q,r)+A(r)+\min_{p\in\Pi_q}[W_{j-1}(p,q)+T(p,q,r)]$; store $\arg\min$ |
| 7: $J^*\leftarrow\min_qW_{n+1}(q,G)$; backtrack the stored predecessors |
| 8: **return** the fused path and $J^*$ |

## C. Candidate Sets and DPF-PSO

*Global fusion* uses the $j$-th waypoints of the $N_g=50$ best distinct solutions in the population memory and of $N_r=50$ new uniform random solutions, which are not evaluated individually; it resembles the lattice of [20] with nodes supplied by the swarm. *Local fusion* uses the $N_e=5$ best solutions and $N_p=6$ Gaussian perturbations $\mathbf{x}^*+\sigma(\mathbf{x}_{\max}-\mathbf{x}_{\min})\odot\boldsymbol{\epsilon}$ of the best solution $\mathbf{x}^*$, clipped to the bounds; $\sigma$ starts at 0.05 and is multiplied by 1.5 after a successful local fusion and by 0.75 otherwise, within $[10^{-3},0.2]$. Duplicate solutions and, within a slot, duplicate waypoints are removed. DPF-PSO runs inertia-weight PSO [23] (50 particles, $c_1=c_2=2$, velocity limit 20% of the ranges, inertia weight decreasing linearly from 0.9 to 0.4 with the fraction of the budget used). It applies global fusion to the initial swarm and then, counted in host FEs (FEs not charged to fusion), global fusion every 5,000 and local fusion every 250 host FEs (every 100 and 5 iterations). A fused path that is cheaper than the best personal best replaces the worst one, and that particle is moved to it with zero velocity; a fusion that does not fit into the remaining budget is skipped. Any optimizer with a population memory and the slot encoding can use the operator with the same schedule.

# IV. Experiments

## A. Setup

*Scenarios (Table I):* S1–S6 are synthetic 1 km × 1 km scenarios with a 5 m terrain grid. S1–S4 were used to design DPF-PSO and are the only scenarios used for tuning; S5 and S6 were added after the design was fixed but were used in an earlier test. H01–H40 come from a procedural generator with a new seed family and were not used for any design or tuning decision (only H01 appeared in a two-run smoke test of the pipeline; supplement S2); they carry all confirmatory claims. C1 is the public scenario of [5] (a real elevation model with 1 m pixels and six threats). Each scenario is solved with $n=10$ and $n=20$ waypoints (30 and 60 variables); "S3/20" denotes S3 with $n=20$.

: TABLE I. Scenarios

| Scen. | Elevation (m) | Threats | Start → goal | Use |
|:--|:-:|:-:|:-:|:-:|
| S1 | 20–173 | 4 | (100, 100) → (900, 900) | tuning + test |
| S2 | 30–354 | 6 | (100, 100) → (900, 900) | tuning + test |
| S3 | 26–265 | 10 | (50, 500) → (950, 500) | tuning + test |
| S4 | 20–942 | 12 | (100, 900) → (900, 100) | tuning + test |
| S5 | 20–630 | 14 | (80, 300) → (920, 700) | test |
| S6 | 14–407 | 18 | (500, 60) → (500, 940) | test |
| H01–H40 | 1–859 | 6–18 | random, ≈800 m apart | held-out test |
| C1 [5] | 49–296 | 6 | (199, 99) → (799, 799) | public test |

*Methods:* the black-box baselines are PSO [23] (DPF-PSO without DPF), SPSO [5] with the parameters of its public code but 50 particles (500 in the code; 100 and 500 in supplement S7), the grey wolf optimizer (GWO) [24], the whale optimization algorithm (WOA) [25], Harris hawks optimization (HHO) [26], DE/rand/1/bin [27] ($F=0.5$, $CR=0.9$, 50 members), L-SHADE [28] and CMA-ES [29] with IPOP restarts [30] (supplement S2). Two baselines exploit the cost structure. Iterative dynamic programming (IDP) [21] solves in each round the fusion problem for the incumbent and 19 random candidates per slot drawn in a box around it, whose half-width shrinks geometrically from the full range to $10^{-3}$ of it over the budget. Gray-box local search (GLS) moves one waypoint at a time with 20 Gaussian proposals, re-evaluates only the affected terms (charged $2/(n+1)$ FE per proposal), adapts its step by the 1/5 rule and restarts when the step falls below $10^{-4}$. All methods except SPSO use the slot encoding.

*Tuning:* DE, L-SHADE, SPSO, IDP and DPF-PSO received the same tuning budget: the default and 24 random configurations from a parameter space, each run with 5 seeds on the 8 instances of S1–S4; the configuration with the lowest mean rank of the median cost was selected (suffix "-T"; supplement S3). Tuning used separate seeds and no test scenario.

*Budget and statistics:* the budget is $3000n$ FEs. Each method is run 30 times on S1–S6 and C1 and 10 times on each held-out scenario; run $r$ of every method uses the same seed. The gap of DPF-PSO ($P$) to a method $B$ is $\Delta=(\tilde J_B-\tilde J_P)/\tilde J_B$, where $\tilde J$ is the median final cost; $\Delta>0$ means that DPF-PSO is better. *Primary analysis:* on H01–H40, with the scenario as the unit, a two-sided Wilcoxon signed-rank test on the 40 paired scenario medians for each baseline and each $n$, with Holm correction [31] over the 20 tests ($\alpha=0.05$); the tuned configurations form a second family of 8 tests. *Descriptive analysis:* on single instances, two-sided rank-sum tests with Holm correction over the methods of the table, and the Vargha–Delaney effect size $A_{12}$ [32] (supplement). Mean ranks over S1–S6 are descriptive only, because their 12 instances come from six scenarios and are not independent. Python 3.13 with NumPy 2.2, one single-threaded process per run, Intel Core i7-14650HX.

## B. Held-Out Scenarios (Primary Analysis)

: TABLE II. Paired Comparison on the 40 Held-Out Scenarios

| Method | Win~10~ | Δ̃~10~ | p~H,10~ | Win~20~ | Δ̃~20~ | p~H,20~ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| PSO | 88% | +5.4% | <0.001^+^ | 100% | +16.6% | <0.001^+^ |
| SPSO | 57% | +1.0% | 0.887 | 78% | +5.8% | 0.003^+^ |
| GWO | 100% | +14.5% | <0.001^+^ | 100% | +25.8% | <0.001^+^ |
| WOA | 100% | +18.2% | <0.001^+^ | 95% | +22.4% | <0.001^+^ |
| HHO | 100% | +12.2% | <0.001^+^ | 98% | +19.0% | <0.001^+^ |
| DE | 57% | +2.3% | 0.003^+^ | 95% | +9.9% | <0.001^+^ |
| L-SHADE | 85% | +3.4% | <0.001^+^ | 92% | +9.6% | <0.001^+^ |
| CMA-ES | 98% | +11.8% | <0.001^+^ | 100% | +14.4% | <0.001^+^ |
| IDP | 85% | +0.8% | <0.001^+^ | 45% | −0.1% | 0.887 |
| GLS | 92% | +2.1% | <0.001^+^ | 98% | +5.8% | <0.001^+^ |
| *Tuned:* | | | | | | |
| DE | 90% | +1.2% | <0.001^+^ | 100% | +6.2% | <0.001^+^ |
| L-SHADE | 60% | +1.0% | 0.009^+^ | 85% | +10.3% | <0.001^+^ |
| SPSO | 30% | −2.5% | <0.001^−^ | 65% | +1.2% | 0.656 |
| IDP | 95% | +1.1% | <0.001^+^ | 72% | +0.3% | 0.001^+^ |

::: {custom-style="TableNote"}
Subscripts: number of waypoints $n$. Win: share of scenarios on which DPF-PSO (tuned DPF-PSO in the lower block) has the lower median. Δ̃: median of the 40 scenario gaps. p~H~: two-sided signed-rank test on the paired scenario medians, Holm-adjusted over the 20 tests of the upper and the 8 tests of the lower block; ^+^/^−^: significant in favour of / against DPF-PSO.
:::

![Fig. 2. Gaps Δ of DPF-PSO to the default methods (above the dashed line) and of tuned DPF-PSO to the tuned methods (below it) on the 40 held-out scenarios; logarithmic axis beyond ±10%.](../figures/fig_heldout.png){width=84mm}

With default configurations (Table II, Fig. 2), the median gaps of DPF-PSO to the black-box baselines ranged from +1.0% (SPSO) to +18.2% for $n=10$ and from +5.8% (SPSO) to +25.8% for $n=20$; 15 of these 16 comparisons were significant, the exception being SPSO for $n=10$ (DPF-PSO lower on 57% of the scenarios, $p_H=$ 0.887). GLS, which exploits the decomposition with local moves, was also clearly worse (+2.1% and +5.8%). IDP was the closest competitor: DPF-PSO had the lower median on 85% of the scenarios for $n=10$ (Δ̃ = +0.8%, significant), but only on 45% for $n=20$ (Δ̃ = −0.1%, $p_H=$ 0.887). After equal tuning, DPF-PSO remained significantly better than DE (+1.2% for $n=10$, +6.2% for $n=20$), L-SHADE (+1.0%, +10.3%) and IDP (+1.1%, +0.3%), whereas tuned SPSO was significantly better for $n=10$ (Δ̃ = −2.5%; tuned DPF-PSO lower on only 30% of the scenarios) and not significantly different for $n=20$ (Δ̃ = +1.2%). Section V traces the advantage of SPSO to its encoding.

[[WIDE]]

: TABLE III. Median Final Cost on S1–S6 (30 Runs, Default Configurations)

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
+/≈/−: DPF-PSO significantly better / not different / significantly worse (two-sided rank-sum test, Holm over the 10 methods of each instance, α = 0.05); descriptive only. ^†^: some runs infeasible (supplement Tables S5–S6). Bold: lowest median. \*: S5 and S6 were not used for design or tuning. Rank: mean rank of the median over the 12 instances.
:::

[[/WIDE]]

## C. Benchmark Scenarios S1–S6

On S1–S6 (Table III, descriptive), DPF-PSO had the lowest median on 7 of 12 instances and the best mean rank (1.42), followed by IDP (2.50) and DE (4.00). It was significantly better than IDP on 8 instances and never worse; only DE was significantly better, on S1/10 and S6/10. The gaps to IDP ranged from −0.1% to +1.4%, those to the best black-box baseline of each instance from −1.0% to +17.4%. Among the tuned methods (supplement Table S7), tuned DPF-PSO had the lowest median on 8 instances and the best mean rank (1.58); only tuned L-SHADE (S1/10) and tuned SPSO (S1/10 and S6/10) were significantly better than it on single instances.

## D. Sources of the Improvement

: TABLE IV. Ablation and Plug-in Study on S1–S6 (Δ Averaged over 12 Instances)

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

Without DPF (Table IV, row PSO) the mean gap is +16.4%, and with a greedy slot-by-slot choice among the same candidates instead of the dynamic program +3.1% (significant on 11 instances), so the optimality of the recombination matters. Global fusion is the most important component (+4.9%); random candidates and local fusion add little (+0.3%, +0.2%), and a fixed geometric decrease of $\sigma$ was never worse than the success-based rule and better on 2 instances. Fusion without PSO moves (DPF-only) was worse on all 12 instances (+1.1%). As a plug-in, DPF lowered the median cost of all six hosts on average, by 9.8% (DE) to 23.1% (GWO), but was significantly worse in 3 of 72 host–instance pairs (DE on S1/10, DE on S6/10 and L-SHADE on S6/20). With DPF, the medians of the six hosts were within 2.1% of each other on the median instance, against 24.4% without it.

## E. Robustness

: TABLE V. Further Experiments (Medians; +/≈/− as in Table III)

| Experiment | Inst. | Lowest median | Rank: DPF-PSO / best other | vs. IDP | vs. all others |
|:--|:-:|:-:|:-:|:-:|:-:|
| S1–S6, defaults | 12 | 7 | 1.42 / 2.50 (IDP) | 8/4/0 | 107/11/2 |
| S1–S6, tuned (5 methods) | 12 | 8 | 1.58 / 2.50 (IDP) | 10/2/0 | 38/7/3 |
| C1 [5], defaults | 2 | 2 | 1.00 / 2.50 (IDP) | 2/0/0 | 19/1/0 |
| C1 [5], tuned (5 methods) | 2 | 2 | 1.00 / 2.50 (IDP) | 2/0/0 | 8/0/0 |
| S1–S6, n = 30 | 6 | 4 | 1.33 / 2.17 (IDP) | 1/4/1 | 53/6/1 |
| S1–S6, 5× budget | 12 | 3 | 2.58 / 2.92 (IDP) | 0/11/1 | 80/35/5 |
| S1–S6, b~2~ = 10 | 12 | 7 | 1.58 / 2.83 (IDP) | 1/11/0 | 91/27/2 |
| S1–S6, equal CPU time | 12 | 7 | 1.75 / 2.83 (IDP) | 5/6/1 | 102/14/4 |

::: {custom-style="TableNote"}
Inst.: number of instances. Lowest median: instances on which DPF-PSO (tuned DPF-PSO in the tuned rows) has the lowest median. vs. all others: +/≈/− summed over all other methods and instances. Runs per instance: 30 (S1–S6, C1, equal CPU time), 20 ($n=30$), 10 (5× budget, $b_2=10$). Equal CPU time: baselines with budgets multiplied by 1.3–2.0 against DPF-PSO with its normal budget. Details in supplement S6.
:::

On the public scenario C1, DPF-PSO and tuned DPF-PSO had the lowest median for both $n$ and were significantly better than every other default or tuned method, except GLS for $n=20$. With $n=30$, DPF-PSO had the lowest median on 4 of 6 instances and the best mean rank (1.33); IDP was significantly better on S6/30, and the gap to the best black-box baseline of each instance ranged from +0.9% to +24.3%. With $b_2=10$, DPF-PSO and IDP kept the first two mean ranks and only DE was significantly better (on S1/10 and S6/10). With five times the budget, the black-box methods closed much of the gap: the median over instances of the gap to the best black-box baseline fell from 4.7% to 2.1%, because SPSO, GLS and L-SHADE lowered their medians by 5.6%, 5.0% and 1.7% (median over instances), whereas the median of DPF-PSO changed by at most 0.5%. DPF-PSO kept the best mean rank (2.58) but had the lowest median on only 3 of 12 instances, and significantly better were SPSO on S6/10; DE on S1/10; L-SHADE on S6/10 and S6/20; IDP on S3/20. Part of the advantage at the standard budget is therefore faster convergence. At equal CPU time, with budgets multiplied by 1.3–2.0, DPF-PSO still had the lowest median on 7 of 12 instances and the best mean rank (1.75); it was significantly worse in 4 of 120 comparisons (DE on S1/10, S2/10 and S6/10; IDP on S3/20).

## F. Computational Cost

Fusion took 64% ($n=10$) and 64% ($n=20$) of the FE budget of DPF-PSO, with 2 and 4 global fusions (97 and 92 distinct candidates per slot on average) and 42 and 86 local fusions per run (supplement S8). Because turn terms are not charged, a run took 1.99 s and 6.57 s of CPU time on one performance core, 1.8 and 1.7 times the median of the black-box baselines other than CMA-ES; CMA-ES and GLS were slower (supplement S9).

## G. Safety Check

An independent check that does not use the cost function (supplement S10) re-evaluated all 32,480 final paths of all experiments with analytic threat distances and an independently implemented exact clearance. No path that the cost function reported as feasible violated a constraint in the check; the only disagreements were 2 paths that the cost function rejected for a clearance shortfall below $10^{-12}$ m (rounding) and the check accepted within its tolerance of $10^{-6}$ m. The minimum clearance of the 32,351 feasible paths was 20.000 m. With the weights of [5], one metre of extra length costs five times as much as one metre of danger-zone intrusion, so 14% of the feasible paths (23% for DPF-PSO) pass within 1 mm of an inflated threat radius; with $b_2=10$ the share fell to 2%. Applications should choose the danger weight or an explicit margin according to their safety requirements.

# V. Discussion and Limitations

*Where the improvement comes from.* DPF-PSO was ahead of all black-box methods with the slot encoding, by margins that grew with $n$, whereas IDP, which applies the same dynamic program to random candidates around one incumbent, stayed within 1.1% of it in median gap on the held-out scenarios. The dynamic program over the decomposable cost thus accounts for most of the improvement; the swarm memory as a source of candidates adds a small, mostly significant gain (tuned DPF-PSO was significantly better than tuned IDP for both $n$ and default DPF-PSO than default IDP for $n=10$, and fusion without PSO moves was worse on all instances of S1–S6). Metaheuristics for this cost model should therefore be compared with a structure-exploiting baseline such as IDP.

*Representation.* SPSO's encoding can merge consecutive waypoints (segment length $r=0$). Because the altitude band of [5] applies only at waypoints, while between waypoints only the clearance constraint applies, a path with fewer distinct waypoints needs fewer climbs and descents and can fly long straight segments (supplement Fig. S4); the slot encoding, with one waypoint 100–200 m above the ground in every slot, cannot represent such paths. On the held-out scenarios with $n=10$, tuned SPSO had the lower median on 28 of 40 scenarios, with a median of 3.0 merged waypoints and longest segments of 413 m (tuned DPF-PSO: 144 m), and its advantage grew with the number of merged waypoints (Spearman's $\rho=$ −0.70, $p=$ $6.0\times10^{-7}$); on S1–S6 it had the lowest median of all configurations on S1/10 and S6/10. DPF could represent merged waypoints with transitions between non-adjacent slots, which keep the three-point structure but multiply the number of edge and turn evaluations.

*Scope.* The results hold for the cost model of [5] with exact clearance and unthresholded angle terms, for static, known environments and a single UAV, and for a budget of $3000n$ FEs, beyond which DPF-PSO improved little. The held-out scenarios come from one generator, and C1 is the only real terrain. At equal FEs, DPF-PSO needs about 1.7 times the CPU time of typical black-box baselines. The slot encoding cannot represent paths that move backwards along $SG$, and costs with global terms, such as a time budget, would need a larger state.

# VI. Conclusion

The UAV path cost of [5] decomposes into terms over at most three consecutive waypoints, and DPF uses this to return the best path that can be assembled from the swarm memory and sampled candidates. On 40 held-out scenarios, DPF-PSO was better than all black-box metaheuristics with the same encoding, by 2.3% to 18.2% for 10 waypoints and 9.6% to 25.8% for 20, and better than or not significantly different from iterative dynamic programming; DPF also improved six host algorithms. Most of the improvement comes from the dynamic program, part of it is faster convergence, since black-box methods closed much of the gap with five times the budget, and the encoding matters as well: tuned SPSO, which can merge waypoints, was better for 10 waypoints. Future work will extend DPF to transitions between non-adjacent slots, to multiple UAVs and to dynamic threats.

# Code and Data Availability

The code, the scenario generators, all seeds, the raw results and the scripts that produce every number, table and figure of this paper and its supplement are available at [anonymous repository link to be inserted by the authors].

# Acknowledgment

[Generative-AI disclosure required by the ICAISI 2027 guidelines; the authors must check and complete it before submission. Draft: The authors used Kiro, an AI agent in the Kiro IDE (underlying model Claude Opus 5.5, Anthropic), for literature search, implementation of the code, execution of the experiments, statistical analysis, preparation of the figures and tables, and drafting of the text. The authors [describe their own contributions, e.g., defined the research question, checked the derivations and the code, reproduced the main results, and revised the text] and take full responsibility for the content of this paper.]

# References

::: {custom-style="Reference"}
[1] S. Aggarwal and N. Kumar, "Path planning techniques for unmanned aerial vehicles: A review, solutions, and challenges," *Comput. Commun.*, vol. 149, pp. 270–299, 2020.

[2] M. Hooshyar and Y.-M. Huang, "Meta-heuristic algorithms in UAV path planning optimization: A systematic review (2018–2022)," *Drones*, vol. 7, no. 12, Art. no. 687, 2023.

[3] J. Kennedy and R. Eberhart, "Particle swarm optimization," in *Proc. ICNN'95 Int. Conf. Neural Netw.*, vol. 4, 1995, pp. 1942–1948.

[4] Y. Fu, M. Ding, and C. Zhou, "Phase angle-encoded and quantum-behaved particle swarm optimization applied to three-dimensional route planning for UAV," *IEEE Trans. Syst., Man, Cybern. A, Syst., Humans*, vol. 42, no. 2, pp. 511–526, 2012.

[5] M. D. Phung and Q. P. Ha, "Safety-enhanced UAV path planning with spherical vector-based particle swarm optimization," *Appl. Soft Comput.*, vol. 107, Art. no. 107376, 2021.

[6] Y. Liu, H. Zhang, H. Zheng, Q. Li, and Q. Tian, "A spherical vector-based adaptive evolutionary particle swarm optimization for UAV path planning under threat conditions," *Sci. Rep.*, vol. 15, Art. no. 2116, 2025.

[7] S. Fu, K. Li, H. Huang, C. Ma, Q. Fan, and Y. Zhu, "Red-billed blue magpie optimizer: A novel metaheuristic algorithm for 2D/3D UAV path planning and engineering design problems," *Artif. Intell. Rev.*, vol. 57, no. 6, Art. no. 134, 2024.

[8] Z. Zhan, D. Lai, C. Huang, Z. Zhang, Y. Deng, and J. Yang, "MSCSO: A modified sand cat swarm algorithm for 3D UAV path planning in complex environments with multiple threats," *Sensors*, vol. 25, no. 9, Art. no. 2730, 2025.

[9] Y. Wang, J. Liu, Y. Qian, and W. Yi, "Path planning for multi-UAV in a complex environment based on reinforcement-learning-driven continuous ant colony optimization," *Drones*, vol. 9, no. 9, Art. no. 638, 2025.

[10] F. Rasoulie and S. Pashazadeh, "3D UAV path optimization using a task-allocation and archive-guided mutation particle swarm optimization algorithm," *Sci. Rep.*, vol. 16, Art. no. 24356, 2026.

[11] K. Sörensen, "Metaheuristics—the metaphor exposed," *Int. Trans. Oper. Res.*, vol. 22, no. 1, pp. 3–18, 2015.

[12] L. D. Whitley, F. Chicano, and B. W. Goldman, "Gray box optimization for Mk landscapes (NK landscapes and MAX-kSAT)," *Evol. Comput.*, vol. 24, no. 3, pp. 491–519, 2016.

[13] D. Whitley, D. Hains, and A. Howe, "Tunneling between optima: Partition crossover for the traveling salesman problem," in *Proc. 11th Annu. Conf. Genet. Evol. Comput. (GECCO)*, 2009, pp. 915–922.

[14] R. Tinós, D. Whitley, and F. Chicano, "Partition crossover for pseudo-Boolean optimization," in *Proc. ACM Conf. Found. Genet. Algorithms (FOGA XIII)*, 2015, pp. 137–149.

[15] C. Cotta, E. Alba, and J. M. Troya, "Utilizing dynastically optimal forma recombination in hybrid genetic algorithms," in *Parallel Problem Solving from Nature—PPSN V* (LNCS 1498), 1998, pp. 305–314.

[16] A. V. Eremeev and J. V. Kovalenko, "Optimal recombination in genetic algorithms for combinatorial optimization problems: Part I," *Yugosl. J. Oper. Res.*, vol. 24, no. 1, pp. 1–20, 2014; "Part II," vol. 24, no. 2, pp. 165–186, 2014.

[17] H. Mühlenbein and H.-M. Voigt, "Gene pool recombination in genetic algorithms," in *Meta-Heuristics: Theory and Applications*, I. H. Osman and J. P. Kelly, Eds. Boston, MA, USA: Kluwer, 1996, pp. 53–62.

[18] A. A. Amini, T. E. Weymouth, and R. C. Jain, "Using dynamic programming for solving variational problems in vision," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 12, no. 9, pp. 855–867, 1990.

[19] M. Werling, J. Ziegler, S. Kammel, and S. Thrun, "Optimal trajectory generation for dynamic street scenarios in a Frenét frame," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2010, pp. 987–993.

[20] H. Fan *et al.*, "Baidu Apollo EM motion planner," arXiv:1807.08048, 2018.

[21] R. Luus, *Iterative Dynamic Programming*. Boca Raton, FL, USA: Chapman & Hall/CRC, 2000.

[22] R. Bellman, *Dynamic Programming*. Princeton, NJ, USA: Princeton Univ. Press, 1957.

[23] Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in *Proc. IEEE Int. Conf. Evol. Comput.*, 1998, pp. 69–73.

[24] S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey wolf optimizer," *Adv. Eng. Softw.*, vol. 69, pp. 46–61, 2014.

[25] S. Mirjalili and A. Lewis, "The whale optimization algorithm," *Adv. Eng. Softw.*, vol. 95, pp. 51–67, 2016.

[26] A. A. Heidari, S. Mirjalili, H. Faris, I. Aljarah, M. Mafarja, and H. Chen, "Harris hawks optimization: Algorithm and applications," *Future Gener. Comput. Syst.*, vol. 97, pp. 849–872, 2019.

[27] R. Storn and K. Price, "Differential evolution—A simple and efficient heuristic for global optimization over continuous spaces," *J. Global Optim.*, vol. 11, no. 4, pp. 341–359, 1997.

[28] R. Tanabe and A. S. Fukunaga, "Improving the search performance of SHADE using linear population size reduction," in *Proc. IEEE Congr. Evol. Comput. (CEC)*, 2014, pp. 1658–1665.

[29] N. Hansen and A. Ostermeier, "Completely derandomized self-adaptation in evolution strategies," *Evol. Comput.*, vol. 9, no. 2, pp. 159–195, 2001.

[30] A. Auger and N. Hansen, "A restart CMA evolution strategy with increasing population size," in *Proc. IEEE Congr. Evol. Comput. (CEC)*, vol. 2, 2005, pp. 1769–1776.

[31] S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.

[32] A. Vargha and H. D. Delaney, "A critique and improvement of the CL common language effect size statistics of McGraw and Wong," *J. Educ. Behav. Stat.*, vol. 25, no. 2, pp. 101–132, 2000.
:::
