::: {custom-style="Title"}
Exact Dynamic-Programming Path Fusion for Swarm-Based 3-D UAV Path Planning
:::

::: {custom-style="Author"}
[Author 1]^1^, [Author 2]^1^, [Corresponding Author]^1,\*^\
^1^[Department, University, City, Country]\
\*[e-mail address]
:::

[[COLS2]]

::: {custom-style="Abstract"}
*Abstract*—Swarm-intelligence optimizers are widely used for three-dimensional (3-D) path planning of unmanned aerial vehicles (UAVs), but they treat the path cost as a black box. The standard cost model, which combines path length, threat, altitude and smoothness terms, is in fact a sum of terms that each depend on at most three consecutive waypoints. We exploit this structure with dynamic-programming path fusion (DPF), a recombination operator that, given candidate positions for every waypoint collected from the swarm, returns the cheapest path that can be assembled from them. DPF is exact, never worse than the best parent path, and needs $O(nK^3)$ operations for $n$ waypoints and $K$ candidates per waypoint; an exact pruning rule based on a bound on the turn term reduces its run time by a factor of 2.4 for large candidate sets. Embedding DPF in particle swarm optimization gives DPF-PSO. On 12 instances (six terrain-and-threat scenarios, two of them generated only after the method was fixed, each with 10 and 20 waypoints) and under equal evaluation budgets, DPF-PSO obtained the lowest mean cost on 11 instances and the best average Friedman rank (1.08 versus 2.58 for DE) among eight algorithms, including spherical-vector PSO and L-SHADE. Added to five host algorithms, DPF lowered their mean cost by 10.2–25.3% on average. DPF-PSO needs more CPU time per evaluation; with the baselines given the same CPU time, it still had the lowest mean cost on 10 of 12 instances.
:::

::: {custom-style="Keywords"}
*Keywords*—UAV path planning, particle swarm optimization, dynamic programming, recombination, swarm intelligence
:::

# I. Introduction

Path planning lets an unmanned aerial vehicle (UAV) reach its goal while avoiding terrain and threats and respecting its flight limits [1]. In 3-D environments with continuous waypoint coordinates, the problem is usually written as the minimization of a weighted cost over a fixed number of waypoints and solved with population-based metaheuristics, which cope with the non-convex and discontinuous landscape created by obstacles [2]. Particle swarm optimization (PSO) [3] and its variants are among the most used methods, from phase-angle-encoded quantum-behaved PSO [4] to spherical-vector PSO (SPSO) [5], which encodes each path segment by its length, elevation and azimuth, and its recent extensions with chaotic initialization, adaptive coefficients and evolutionary operators [6]. New metaphor-based algorithms are also regularly evaluated on this problem [7], although the value of many such metaphors has been questioned [8].

These methods treat the path cost as a black box. Yet the most widely used cost model [5] is a sum of terms that depend on one waypoint (altitude), two consecutive waypoints (length, threat and terrain clearance) or three consecutive waypoints (turning and climbing angles). In combinatorial optimization, recombination operators that exploit this kind of separability are very effective: partition crossover for the traveling salesman problem returns the best of an exponential number of offspring at linear cost [9], [10], and memetic algorithms combine population search with problem-specific procedures in general [11]. We are not aware of work that uses the additive structure of the UAV path cost for exact recombination inside a continuous swarm optimizer.

This paper makes three contributions:

1) Dynamic-programming path fusion (DPF), an exact recombination operator. Given candidate positions for each waypoint, taken from the swarm memory, from perturbations and from random samples, DPF returns the least-cost path that can be assembled from them. The fused path is never worse than any parent, and an exact pruning rule reduces the computation without changing the result.

2) DPF-PSO, which fuses the whole swarm memory with random samples (global fusion) and the best paths with Gaussian perturbations (local fusion), and a plug-in interface that adds DPF to any population-based optimizer.

3) An evaluation on 12 instances with held-out scenarios, seven baselines, a plug-in study on five host algorithms, an ablation study, a CPU-time analysis and an independent safety check of all final paths.

# II. Problem Formulation

## A. Environment and Path Representation

The workspace is a 1 km × 1 km area with terrain elevation $Z(x,y)$ given on a 5 m grid with bilinear interpolation and $N_t$ vertical cylindrical threats with centres $\mathbf{c}_k$ and radii $R_k$. A path consists of the start $S$, waypoints $P_1,\dots,P_n$ and the goal $G$. Let $L$ be the horizontal distance between $S$ and $G$, $\mathbf{e}_1$ the horizontal unit vector from $S$ to $G$ and $\mathbf{e}_2$ its left normal. The segment $SG$ is divided into $n$ slots centred at $B_j=S+j\Delta s\,\mathbf{e}_1$ with $\Delta s=L/(n+1)$, and waypoint $j$ is placed at

$$P_j^{xy}=B_j+\tau_j\Delta s\,\mathbf{e}_1+u_j\mathbf{e}_2,\qquad z_j=Z(P_j^{xy})+h_j$$

where $u_j\in[-400,400]$ m is the lateral offset (restricted to the map), $\tau_j\in[-0.45,0.45]$ the along-track offset inside the slot and $h_j\in[100,200]$ m the height above ground. The decision vector $\mathbf{x}=(u_j,h_j,\tau_j)_{j=1}^{n}$ has $3n$ variables, the same number as in SPSO. Because the slots do not overlap, waypoints stay ordered along $SG$.

## B. Cost Function

We use the cost model of Phung and Ha [5] with the weights of their public code, $b_1=5$, $b_2=1$, $b_3=10$, $b_4=1$:

$$J=b_1F_1+b_2F_2+b_3F_3+b_4F_4+\Pi$$

$F_1$ is the 3-D path length. $F_2=\sum_{j,k}\min\{S_d,(R_k+D+S_d-d_{jk})^+\}$ measures how deep each segment $j$ enters a danger zone of width $S_d=10$ m around threat $k$, where $d_{jk}$ is the horizontal distance from $\mathbf{c}_k$ to the segment, $D=1$ m the UAV size and $(\cdot)^+=\max\{0,\cdot\}$. $F_3=\sum_j|h_j-\bar h|$ penalizes deviations from the centre $\bar h=150$ m of the altitude band. $F_4=\sum_{j=1}^{n}(\phi_j+\Delta\psi_j)$ sums, over the waypoints, the turning angle $\phi_j$ between the horizontal projections of the two segments that meet at $P_j$ and the change $\Delta\psi_j$ of their climb angles, in degrees [5, Eq. (8)]. Instead of the infinite cost of [5], every violated segment adds $\pi_0+\lambda v$ to $\Pi$, with $\pi_0=10^5$, $\lambda=10^3$ and $v$ the violation in metres. A segment is violated if it enters the inflated threat radius $R_k+D$ or if its ground clearance, checked at 20 interior points, falls below 20 m (the model of [5] checks the terrain only at the waypoints).

With $P_0=S$ and $P_{n+1}=G$, every term of $J$ involves at most three consecutive points:

$$J=\sum_{j=1}^{n}N(P_j)+\sum_{j=0}^{n}E(P_j,P_{j+1})+\sum_{j=1}^{n}T(P_{j-1},P_j,P_{j+1})$$

The node term $N$ holds the altitude cost, the edge term $E$ the length, threat cost and segment violations, and the turn term $T$ the smoothness cost. $E$ requires terrain sampling and distance computations and dominates the cost of an evaluation, whereas $T$ is cheap arithmetic.

# III. Dynamic-Programming Path Fusion

## A. Fusion Problem and Recursion

Assume that for every slot $j$ a finite set $C_j$ of $K_j$ candidate waypoints is available. The fusion problem asks for the path of minimum cost whose $j$-th waypoint belongs to $C_j$ for all $j$. Exhaustive search would examine $\prod_jK_j$ paths, i.e., $10^{40}$ for $n=20$ and $K_j=100$.

Because $T$ couples three consecutive points, the state of the dynamic program is the pair of the last two waypoints. Let $C_0=\{S\}$, $C_{n+1}=\{G\}$, $N(G)=0$, and let $W_j(q,r)$, $q\in C_{j-1}$, $r\in C_j$, be the minimum cost of a partial path ending with $q,r$, including all terms up to $r$ except the turn at $r$. Then $W_1(S,r)=E(S,r)+N(r)$ and, for $j=2,\dots,n+1$ and $p\in C_{j-2}$,

$$W_j(q,r)=E(q,r)+N(r)+\min_{p}\big[W_{j-1}(p,q)+T(p,q,r)\big]$$

The optimal cost is $J^*=\min_{q\in C_n}W_{n+1}(q,G)$, and storing the minimizing $p$ for every $(q,r)$ recovers the fused path by backtracking (Fig. 1).

*Proposition 1:* (i) $J^*$ is the minimum of $J$ over all $\prod_jK_j$ paths that can be assembled from the candidate sets. (ii) If every $C_j$ contains the $j$-th waypoint of each parent path, $J^*$ is not larger than the cost of the best parent.

*Proof:* (i) Once $P_1,\dots,P_j$ are fixed, the terms that remain to be added depend on the prefix only through $(P_{j-1},P_j)$, because no term involves more than three consecutive points. The principle of optimality [12] therefore holds for this state, and (4) is exact. (ii) Every parent is one of the assemblable paths. ∎

![Fig. 1. DPF on a two-dimensional toy problem with five slots (cost: length + turning penalty + collision penalty). Grey lines are the edges evaluated by the dynamic program. The fused path (red) combines waypoints of parents 2 and 3 with one perturbation, avoids the threat, and costs 10.6, compared with 13.9 for the best parent.](../figures/fig_dpf_schematic.png){width=84mm}

## B. Cost, Pruning and Accounting

One fusion evaluates $\sum_{j=0}^{n}K_jK_{j+1}$ edge terms and $\sum_{j=1}^{n}K_{j-1}K_jK_{j+1}$ turn terms. Each angle in $F_4$ lies in $[0^\circ,180^\circ]$, so $0\le T\le T_{\max}=360\,b_4$. A predecessor $p$ with $W_{j-1}(p,q)>\min_{p'}W_{j-1}(p',q)+T_{\max}$ can never attain the minimum in (4) for any $r$ and is skipped. The pruning is exact: it returned bit-identical results in 324 random test cases and in all runs of Section IV, and it reduced the time of one fusion with $n=20$ and $K_j=100$ from 1.13 s to 0.48 s. It is applied only when some $K_j>20$, because its overhead outweighs the savings for small candidate sets.

Because edge terms dominate the evaluation cost, each fusion is charged $\lceil\sum_jK_jK_{j+1}/(n+1)\rceil$ function evaluations (FEs), i.e., the number of segments evaluated divided by the number of segments of one path. Turn terms are not charged; their cost is reported as CPU time in Section IV-E.

## C. Candidate Sets

*Global fusion* builds $C_j$ from the $j$-th waypoints of the $N_g=50$ best distinct solutions in the swarm memory and of $N_r=50$ new uniformly random solutions. It recombines the whole swarm and resembles solving a layered roadmap [13], [14] whose nodes are supplied by the swarm. The swarm keeps the candidate sets small; a fixed grid would make the same dynamic program impractical, since with 1,000 points per slot and $n=20$ a single fusion would need $1.8\times10^{10}$ turn evaluations. *Local fusion* builds $C_j$ from the $N_e=5$ best solutions and $N_p=6$ Gaussian perturbations $\mathbf{x}^*+\sigma(\mathbf{x}_{\max}-\mathbf{x}_{\min})\odot\boldsymbol{\epsilon}$ of the best solution $\mathbf{x}^*$, clipped to the bounds. The step $\sigma$ starts at 0.05 and is multiplied by 1.5 after a successful local fusion and by 0.75 otherwise, within $[10^{-3},0.2]$.

## D. DPF-PSO and the Plug-in Interface

DPF-PSO (Algorithm 1) runs inertia-weight PSO [15] with 50 particles, inertia weight decreasing linearly from 0.9 to 0.4, $c_1=c_2=2$ and velocities limited to 20% of the variable ranges. A fused path is accepted if it is cheaper than the best personal best; it then replaces the worst personal best, and the corresponding particle is moved to it with zero velocity.

| **Algorithm 1:** DPF-PSO |
|:--|
| 1: initialize and evaluate the swarm; personal bests $\mathbf{p}_i$ |
| 2: global fusion of $\{\mathbf{p}_i\}$ and $N_r$ random solutions; accept |
| 3: **for** $t=1,2,\dots$ until the FE budget is used **do** |
| 4:   PSO velocity and position update; evaluate; update $\mathbf{p}_i$ |
| 5:   **if** $t \bmod 100=0$ **then** global fusion; accept |
| 6:   **if** $t \bmod 5=0$ **then** local fusion; accept; adapt $\sigma$ |
| 7: **end for**; return the best path |

The operator only needs a population memory with costs, so it can be added to other optimizers with the same schedule. For differential evolution (DE) and L-SHADE the memory is the population; for the grey wolf optimizer (GWO) and the whale optimization algorithm (WOA) it is the population together with the leaders.

# IV. Experiments

## A. Setup

*Scenarios:* Table I lists the six scenarios. S1–S4 were used while DPF-PSO was designed (pilot runs with 10–20 seeds); S5 and S6 were generated procedurally with new seeds after all design decisions had been frozen and were used only for the final test. Every scenario is solved with $n=10$ and $n=20$ waypoints (30 and 60 variables), giving 12 instances.

: TABLE I. Benchmark Scenarios

| Scen. | Elevation (m) | Threats | Start → goal | Use |
|:--|:-:|:-:|:-:|:-:|
| S1 | 20–173 | 4 | (100, 100) → (900, 900) | design + test |
| S2 | 30–354 | 6 | (100, 100) → (900, 900) | design + test |
| S3 | 26–265 | 10 | (50, 500) → (950, 500) | design + test |
| S4 | 20–942 | 12 | (100, 900) → (900, 100) | design + test |
| S5 | 20–630 | 14 | (80, 300) → (920, 700) | test only |
| S6 | 14–407 | 18 | (500, 60) → (500, 940) | test only |

*Baselines:* PSO [15], SPSO [5], GWO [16], WOA [17], Harris hawks optimization (HHO) [18], DE/rand/1/bin with $F=0.5$, $CR=0.9$ [19] and L-SHADE [20], each with the parameters recommended by its authors and without tuning. SPSO follows the authors' public code ($w=1$ damped by 0.98 per 1/200 of the budget, $c_1=c_2=1.5$, velocity clamping at half the range and velocity mirroring), except that we map the azimuth consistently with its bounds; the code swaps its sine and cosine, which matters only when $SG$ is not diagonal. All other methods use the slot encoding. The population size is 50 except for L-SHADE ($18\cdot3n$, linearly reduced to 4).

*Budget and statistics:* The budget is 30,000 FEs for $n=10$ and 60,000 FEs for $n=20$. Each method is run 30 times per instance, and run $r$ of every method uses the same random seed. Differences on an instance are tested with the two-sided Wilcoxon rank-sum test at $\alpha=0.05$, and rankings over instances with the Friedman test [21]. All methods were implemented in Python 3.13 with NumPy 2.2 (single-threaded) and run on an Intel Core i7-14650HX.

[[WIDE]]

: TABLE II. Final Cost (Mean ± Standard Deviation over 30 Runs)

| Inst. | PSO | SPSO | GWO | WOA | HHO | DE | L-SHADE | DPF-PSO |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S1, 10 | 6400 ± 314 + | 6190 ± 232 ≈ | 6650 ± 322 + | 6492 ± 258 + | 6353 ± 121 + | **6019 ± 10 −** | 6316 ± 384 ≈ | 6072 ± 22 |
| S2, 10 | 7319 ± 694 + | 6928 ± 413 + | 7847 ± 629 + | 8234 ± 209 + | 8024 ± 266 + | 6459 ± 227 − | 7304 ± 41 + | **6430 ± 100** |
| S3, 10 | 7305 ± 1566 + | 6087 ± 446 + | 7583 ± 1156 + | 8539 ± 159 + | 8276 ± 174 + | 7000 ± 626 + | 7613 ± 341 + | **5212 ± 65** |
| S4, 10 | 7941 ± 703 + | 7609 ± 428 + | 8799 ± 925 + | 9665 ± 1438 + | 8555 ± 1086 + | 7156 ± 283 + | 7915 ± 370 + | **6888 ± 65** |
| S5, 10* | 6424 ± 786 + | 6125 ± 450 + | 6471 ± 558 + | 8679 ± 1050 + | 7773 ± 1033 + | 6062 ± 372 + | 6310 ± 126 + | **5511 ± 90** |
| S6, 10* | 5085 ± 488 ≈ | 4969 ± 260 ≈ | 5548 ± 213 + | 6375 ± 2438 + | 5642 ± 313 + | 4919 ± 52 ≈ | 4957 ± 28 + | **4901 ± 58** |
| S1, 20 | 6987 ± 666 + | 6649 ± 223 + | 7625 ± 530 + | 6555 ± 239 + | 6467 ± 188 + | 6739 ± 302 + | 7273 ± 69 + | **6237 ± 104** |
| S2, 20 | 8886 ± 613 + | 7619 ± 517 + | 9407 ± 1155 + | 8518 ± 115 + | 8591 ± 153 + | 8041 ± 582 + | 8195 ± 140 + | **7111 ± 135** |
| S3, 20 | 9604 ± 2222 + | 6525 ± 405 + | 25504 ± 50378 + | 9056 ± 62 + | 8934 ± 71 + | 8858 ± 843 + | 9053 ± 1077 + | **5357 ± 139** |
| S4, 20 | 9573 ± 1472 + | 8694 ± 645 + | 11320 ± 1203 + | 10097 ± 1611 + | 9603 ± 1283 + | 8327 ± 537 + | 8848 ± 534 + | **7407 ± 53** |
| S5, 20* | 8205 ± 1250 + | 7261 ± 828 + | 9200 ± 1270 + | 9766 ± 789 + | 9333 ± 409 + | 7066 ± 321 + | 6973 ± 176 + | **5784 ± 292** |
| S6, 20* | 6008 ± 823 + | 5521 ± 219 + | 6250 ± 268 + | 6629 ± 1173 + | 6270 ± 418 + | 5235 ± 191 ≈ | 5195 ± 87 ≈ | **5153 ± 61** |
| +/≈/− | 11/1/0 | 10/2/0 | 12/0/0 | 12/0/0 | 12/0/0 | 8/2/2 | 10/2/0 | — |
| Rank | 5.42 | 3.08 | 6.83 | 7.00 | 5.92 | 2.58 | 4.08 | 1.08 |

::: {custom-style="TableNote"}
"S1, 10": scenario S1 with $n=10$ waypoints. +/≈/−: DPF-PSO significantly better / not significantly different / significantly worse (Wilcoxon rank-sum test, α = 0.05). Bold: lowest mean. \*: held-out scenario. Rank: average rank of the mean over the 12 instances.
:::

[[/WIDE]]

## B. Comparison with the Baselines

Table II shows that DPF-PSO obtained the lowest mean cost on 11 of the 12 instances and was significantly better than all seven baselines on 8 of them. Its average rank of 1.08 is followed by DE (2.58) and SPSO (3.08), and the Friedman test rejects equal performance ($\chi^2=64.1$, $p=2.2\times10^{-11}$). The margin over the best competitor on each instance is larger for $n=20$ (9.5% on average) than for $n=10$ (4.5%) and largest on S3 with $n=20$ (17.9%). On the held-out scenarios S5 and S6, DPF-PSO had the lowest mean on 4 of 4 instances, so the advantage does not depend on the scenarios used during design. Its results also vary less between runs: on the median instance, its standard deviation is 0.19 times the median standard deviation of the baselines.

DE is the only baseline that was significantly better than DPF-PSO on some instances (8/2/2 wins/ties/losses for DPF-PSO): on S1 with $n=10$, DE has both the lower mean (6019 versus 6072) and the lower median (6017 versus 6071); on S2 with $n=10$, DE has the lower median (6383 versus 6402) but a higher mean (6459 versus 6430) because some of its runs end in a poor route. Because SPSO was run with 50 particles instead of the 500 used by its authors, we repeated it with 100 and 500 particles and the same budget. The mean cost gaps to DPF-PSO were 11.3, 9.5, 9.2% for 50, 100 and 500 particles, with wins/ties/losses for DPF-PSO of 10/2/0, 10/1/1, 10/2/0; DPF-PSO was significantly worse only against 100 particles on S6 ($n=10$). Larger swarms narrow the gap slightly but do not change the ranking.

In Fig. 2, SPSO, WOA and HHO stagnate early and PSO, GWO, DE and L-SHADE improve slowly, whereas the initial global fusion of DPF-PSO immediately yields a cheaper path, which the local and periodic global fusions then improve. On S3 and S5 with $n=20$, where the margins are largest, the best DPF-PSO paths use a different corridor through the threats than those of SPSO and DE (Fig. 3).

[[WIDE]]

![Fig. 2. Mean best cost versus FEs over 30 runs. The steep drop of DPF-PSO marks the initial global fusion, whose FEs are charged at once; later steps mark the periodic global fusions.](../figures/fig_convergence.png){width=178mm}

[[/WIDE]]

## C. DPF as a Plug-in

: TABLE III. Change of the Mean Cost When DPF Is Added to a Host

| Inst. | PSO | GWO | WOA | DE | L-SHADE |
|:--|:-:|:-:|:-:|:-:|:-:|
| S1, 10 | −5.1% + | −7.7% + | −5.1% + | +0.9% − | −3.4% ≈ |
| S2, 10 | −12.1% + | −16.8% + | −20.0% + | −0.3% − | −11.9% + |
| S3, 10 | −28.7% + | −30.6% + | −37.9% + | −25.4% + | −31.0% + |
| S4, 10 | −13.3% + | −20.4% + | −26.8% + | −3.8% + | −11.3% + |
| S5, 10 | −14.2% + | −13.4% + | −35.0% + | −8.9% + | −11.4% + |
| S6, 10 | −3.6% ≈ | −9.5% + | −20.5% + | −0.2% ≈ | +0.5% − |
| S1, 20 | −10.7% + | −17.1% + | −3.0% + | −6.4% + | −13.5% + |
| S2, 20 | −20.0% + | −23.6% + | −15.2% + | −11.1% + | −12.7% + |
| S3, 20 | −44.2% + | −78.6% + | −39.0% + | −39.2% + | −35.1% + |
| S4, 20 | −22.6% + | −33.5% + | −24.9% + | −10.0% + | −12.6% + |
| S5, 20 | −29.5% + | −36.3% + | −39.8% + | −17.6% + | −15.1% + |
| S6, 20 | −14.2% + | −15.5% + | −19.7% + | −0.7% ≈ | +0.8% ≈ |
| Mean | −18.2% | −25.3% | −23.9% | −10.2% | −13.1% |
| +/≈/− | 11/1/0 | 12/0/0 | 12/0/0 | 8/2/2 | 9/2/1 |

::: {custom-style="TableNote"}
Negative values: lower cost with DPF. +/≈/−: host with DPF significantly better / not different / significantly worse than the host alone.
:::

Table III shows that DPF lowers the mean cost of every host on average, by 18.2, 25.3, 23.9, 10.2, 13.1% for PSO, GWO, WOA, DE and L-SHADE. Of the 60 host–instance pairs, 52 improved significantly and 3 became significantly worse (DE on S1 ($n=10$), DE on S2 ($n=10$), L-SHADE on S6 ($n=10$); mean changes −0.3% to +0.9%). All of them involve the two strongest hosts on instances with $n=10$, where the hosts already find good paths and the FEs charged for fusion are taken from their own search.

## D. Ablation Study

: TABLE IV. Ablation of DPF-PSO

| Variant | Rank | Δ mean | +/≈/− |
|:--|:-:|:-:|:-:|
| DPF-PSO (full) | 2.08 | 0 | — |
| w/o global fusion | 6.33 | +7.8% | 10/2/0 |
| w/o random candidates | 4.58 | +1.3% | 4/8/0 |
| w/o local fusion | 3.33 | +0.2% | 5/7/0 |
| fixed σ schedule | 1.58 | −0.1% | 0/11/1 |
| greedy instead of DP | 6.08 | +4.2% | 11/1/0 |
| DPF without PSO moves | 4.00 | +0.3% | 4/7/1 |
| PSO (no DPF) | 8.00 | +24.9% | 11/1/0 |

::: {custom-style="TableNote"}
Rank: average rank among the eight variants over the 12 instances. Δ mean: average relative change of the mean cost with respect to DPF-PSO. +/≈/−: DPF-PSO significantly better / not different / significantly worse.
:::

Table IV shows that global fusion is the most important component: without it the mean cost rises by 7.8% on average and significantly on 10 instances. Replacing the dynamic program by a greedy waypoint-by-waypoint choice from the same candidates raises the cost by 4.2% (significantly on 11 instances), so the exact fusion matters, not only the candidate sets. Removing the random candidates or the local fusion significantly raises the cost on 4 and 5 instances, respectively, all with $n=20$. Running the fusion schedule without PSO moves is significantly worse on 4 instances, so the swarm dynamics supply useful candidates. The success-based step rule brings no benefit: a fixed geometric decrease of $\sigma$ from 0.05 to $10^{-3}$ performs the same or better (0/11/1). We kept the rule because the configuration had been frozen before the held-out test; the fixed schedule is a simpler choice for future use.

[[WIDE]]

![Fig. 3. Top views of the best of 30 paths with $n=20$. Filled circles: threats; dotted circles: danger zones; square: start; star: goal.](../figures/fig_paths.png){width=178mm}

[[/WIDE]]

## E. Computational Cost

: TABLE V. Mean CPU Time per Run and Equal-Time Comparison

| Method | CPU s, n = 10 | CPU s, n = 20 | Budget × (10 / 20) | +/≈/− equal time |
|:--|:-:|:-:|:-:|:-:|
| PSO | 0.62 | 2.67 | 1.8 / 1.9 | 11/1/0 |
| SPSO | 0.72 | 3.02 | 1.5 / 1.7 | 10/2/0 |
| GWO | 0.65 | 2.71 | 1.7 / 1.9 | 12/0/0 |
| WOA | 0.63 | 2.54 | 1.7 / 2.0 | 12/0/0 |
| HHO | 0.94 | 3.17 | 1.2 / 1.6 | 12/0/0 |
| DE | 0.62 | 2.59 | 1.8 / 2.0 | 8/2/2 |
| L-SHADE | 0.70 | 4.60 | 1.6 / 1.1 | 9/2/1 |
| DPF-PSO | 1.09 | 5.11 | 1.0 / 1.0 | — |

::: {custom-style="TableNote"}
CPU time from a separate benchmark: 3 runs per method and instance in interleaved order, 6 processes in parallel. Budget ×: FE-budget multiplier that gives a baseline the CPU time of DPF-PSO. +/≈/−: DPF-PSO with its normal budget versus the baseline with the multiplied budget.
:::

Fusion takes 64% of the FE budget of DPF-PSO for both $n=10$ and $n=20$. Because turn terms are not charged, a run of DPF-PSO costs 1.1 s of CPU time for $n=10$ and 5.1 s for $n=20$, 1.7 and 1.9 times the median baseline time (Table V). To check whether the advantage remains at equal CPU time, every baseline was re-run with its FE budget multiplied by 1.1–2.0 so that its CPU time matched that of DPF-PSO. DPF-PSO still had the lowest mean cost on 10 of 12 instances and was significantly better in 74 and significantly worse in 3 of the 84 comparisons (worse: DE on S1 ($n=10$), L-SHADE on S1 ($n=10$), DE on S2 ($n=10$)). Under the heavier load of the main experiment (20 processes), the memory-bound dynamic program slowed down more than the baselines; with the correspondingly larger multipliers (1.0–2.9), DPF-PSO had the lowest mean on 10 of 12 instances and was significantly better in 74 and worse in 4 of the 84 comparisons.

## F. Safety Check

All 2880 final paths of the eight methods in Table II were re-checked independently, with threat distances computed analytically and ground clearance sampled every 1 m. Three final paths, all from GWO on S3 ($n=20$), were infeasible, in agreement with the cost function. No path reported as feasible enters an inflated threat radius. The minimum ground clearance is 16.3 m over all methods and 19.4 m for DPF-PSO, slightly below the required 20 m because clearance is enforced only at the check points. 383 paths pass within 1 mm of an inflated radius $R_k+D$: with the weights of [5], one metre of extra length costs five times as much as one metre of intrusion into the danger zone, so optimal paths run along the collision boundary. Applications should use a larger $b_2$ or an explicit safety margin.

# V. Limitations

DPF needs a cost that is a sum of local terms and an encoding in which waypoint $j$ always belongs to slot $j$. The slot encoding cannot represent paths that move backwards along $SG$, so maze-like environments would require another encoding. Costs with global terms, such as a total time budget, would require a larger DP state. Our experiments use synthetic static environments and a single UAV; real elevation data, dynamic threats and cooperative planning for multiple UAVs remain to be studied.

# VI. Conclusion

The UAV path cost used in most swarm-based planners decomposes into terms over at most three consecutive waypoints. DPF exploits this structure to recombine the swarm exactly and cheaply. DPF-PSO, which applies it to the whole swarm memory and around the best path, obtained the lowest mean cost on 11 of 12 instances against seven baselines, including all held-out instances, with larger margins for 20 waypoints and an advantage that remained at equal CPU time. DPF also improved five different host algorithms. Future work will address free-order encodings, multi-UAV planning and real terrain data.

# Acknowledgment

[To be completed by the authors. The ICAISI 2027 guidelines require disclosure of generative-AI use (tool name, version, function and the parts of the work concerned) in the Methodology or Acknowledgment section.]

# References

::: {custom-style="Reference"}
[1] S. Aggarwal and N. Kumar, "Path planning techniques for unmanned aerial vehicles: A review, solutions, and challenges," *Comput. Commun.*, vol. 149, pp. 270–299, 2020.

[2] M. Hooshyar and Y.-M. Huang, "Meta-heuristic algorithms in UAV path planning optimization: A systematic review (2018–2022)," *Drones*, vol. 7, no. 12, Art. no. 687, 2023.

[3] J. Kennedy and R. Eberhart, "Particle swarm optimization," in *Proc. ICNN'95 Int. Conf. Neural Netw.*, vol. 4, 1995, pp. 1942–1948.

[4] Y. Fu, M. Ding, and C. Zhou, "Phase angle-encoded and quantum-behaved particle swarm optimization applied to three-dimensional route planning for UAV," *IEEE Trans. Syst., Man, Cybern. A, Syst., Humans*, vol. 42, no. 2, pp. 511–526, 2012.

[5] M. D. Phung and Q. P. Ha, "Safety-enhanced UAV path planning with spherical vector-based particle swarm optimization," *Appl. Soft Comput.*, vol. 107, Art. no. 107376, 2021.

[6] Y. Liu, H. Zhang, H. Zheng, Q. Li, and Q. Tian, "A spherical vector-based adaptive evolutionary particle swarm optimization for UAV path planning under threat conditions," *Sci. Rep.*, vol. 15, Art. no. 2116, 2025.

[7] S. Fu, K. Li, H. Huang, C. Ma, Q. Fan, and Y. Zhu, "Red-billed blue magpie optimizer: A novel metaheuristic algorithm for 2D/3D UAV path planning and engineering design problems," *Artif. Intell. Rev.*, vol. 57, no. 6, Art. no. 134, 2024.

[8] K. Sörensen, "Metaheuristics—the metaphor exposed," *Int. Trans. Oper. Res.*, vol. 22, no. 1, pp. 3–18, 2015.

[9] D. Whitley, D. Hains, and A. Howe, "Tunneling between optima: Partition crossover for the traveling salesman problem," in *Proc. 11th Annu. Conf. Genet. Evol. Comput. (GECCO)*, 2009, pp. 915–922.

[10] R. Tinós, D. Whitley, and G. Ochoa, "A new generalized partition crossover for the traveling salesman problem: Tunneling between local optima," *Evol. Comput.*, vol. 28, no. 2, pp. 255–288, 2020.

[11] F. Neri and C. Cotta, "Memetic algorithms and memetic computing optimization: A literature review," *Swarm Evol. Comput.*, vol. 2, pp. 1–14, 2012.

[12] R. Bellman, *Dynamic Programming*. Princeton, NJ, USA: Princeton Univ. Press, 1957.

[13] L. E. Kavraki, P. Švestka, J.-C. Latombe, and M. H. Overmars, "Probabilistic roadmaps for path planning in high-dimensional configuration spaces," *IEEE Trans. Robot. Autom.*, vol. 12, no. 4, pp. 566–580, 1996.

[14] S. Karaman and E. Frazzoli, "Sampling-based algorithms for optimal motion planning," *Int. J. Robot. Res.*, vol. 30, no. 7, pp. 846–894, 2011.

[15] Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in *Proc. IEEE Int. Conf. Evol. Comput.*, 1998, pp. 69–73.

[16] S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey wolf optimizer," *Adv. Eng. Softw.*, vol. 69, pp. 46–61, 2014.

[17] S. Mirjalili and A. Lewis, "The whale optimization algorithm," *Adv. Eng. Softw.*, vol. 95, pp. 51–67, 2016.

[18] A. A. Heidari, S. Mirjalili, H. Faris, I. Aljarah, M. Mafarja, and H. Chen, "Harris hawks optimization: Algorithm and applications," *Future Gener. Comput. Syst.*, vol. 97, pp. 849–872, 2019.

[19] R. Storn and K. Price, "Differential evolution—A simple and efficient heuristic for global optimization over continuous spaces," *J. Global Optim.*, vol. 11, no. 4, pp. 341–359, 1997.

[20] R. Tanabe and A. S. Fukunaga, "Improving the search performance of SHADE using linear population size reduction," in *Proc. IEEE Congr. Evol. Comput. (CEC)*, 2014, pp. 1658–1665.

[21] J. Derrac, S. García, D. Molina, and F. Herrera, "A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms," *Swarm Evol. Comput.*, vol. 1, no. 1, pp. 3–18, 2011.
:::
