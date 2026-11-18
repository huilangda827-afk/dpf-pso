# Design log (research record)

This file records how the method and the experimental protocol were fixed, in
the order the decisions were made. All numbers come from runs stored under
`results/pilot*` (10–20 runs per configuration).

## 1. Problem model

* Cost model of Phung & Ha (Applied Soft Computing 107:107376, 2021): weights
  b = (5, 1, 10, 1), drone size D = 1 m, danger distance S = 10 m, altitude
  band [100, 200] m above ground, smoothness = sum of turning angles + sum of
  climb-angle changes (degrees).  Checked against the authors' public MATLAB
  code (github.com/duongpm/SPSO).
* Modification 1: +inf penalties replaced by `P0 + lam * violation` per
  violated element (P0 = 1e5, lam = 1e3), so that the total cost is a sum of
  node, edge and turn terms.  This additivity is what makes exact DP possible.
* Modification 2: ground clearance (20 m) checked at 20 interior points of
  every segment (original code: waypoints only).
* Six synthetic 1 km × 1 km scenarios (5 m grid, bilinear terrain lookup).
  S1–S4 were used during design; S5–S6 were generated afterwards with fresh
  seeds and used only for the final test (see §5).

## 2. Encoding

* First version: waypoint j fixed on the plane orthogonal to S–G at distance
  jL/(n+1) (2 variables per waypoint: lateral offset u, height h).
* Pilot 2/3 showed SPSO reaching lower costs than every plane-encoded method on
  S1/S2 with n = 10 (e.g. S1: 5938 vs 6136).  Component analysis: SPSO paths
  had shorter length and lower smoothness cost, i.e. the fixed along-track
  spacing was limiting.
* Fix: "slot" encoding with a third variable tau in [-0.45, 0.45] that shifts
  waypoint j along S–G inside its slot.  Slots do not overlap, so waypoints
  stay ordered and the cost stays decomposable.  All plane-encoded methods
  (baselines included) use the slot encoding (dimension 3n, same as SPSO).

## 3. Fusion operator

* Exact DP over pairs of consecutive layers; verified against brute force
  (n = 5) and against the full-path evaluation (gap < 1e-9) for all scenarios
  (`src/sanity_checks.py`).
* Cost accounting: one segment evaluation = 1/(n+1) FE.  Turn terms (no
  terrain lookup) are not counted in FEs; wall-clock time is reported instead.
* Pilot 2: global fusion of the initial population is the dominant component
  (removing it raised mean cost on every instance).  The midpoint candidate had
  no measurable effect (differences < 0.1 %) and was removed.
* Pilot 3/4: outcomes on n = 20 were bimodal: 2–3 of 10 seeds ended in a worse
  route class for every variant, because the initial global fusion fixes the
  route class.  Tested remedies: ring topology, more frequent global fusion,
  current positions as extra candidates, fresh uniform random candidates in
  global fusion.  Only the random candidates removed most of the bad runs.
* Pilot 5 (20 runs, 8 instances): g_random = 50 with g_period = 100 had the
  best or near-best mean on 7 of 8 instances and was fixed as the final
  configuration (`DPF_DEFAULT` in `src/algorithms.py`).
* Implementation change without effect on results: turn costs computed from
  per-edge heading/climb angles (identical values, DP 4–5x faster).

## 4. Baselines

* PSO, GWO, WOA, HHO, DE/rand/1/bin, L-SHADE with the parameter values of their
  original papers; SPSO with the parameters of the reference code (w = 1,
  damping 0.98 stretched over the FE budget, c1 = c2 = 1.5, velocity clamp 0.5,
  velocity mirroring).  No baseline was tuned.
* Same population size (50) for all except L-SHADE (18·D, linear reduction).
* Same FE budget: 30 000 (n = 10), 60 000 (n = 20).  Run r of every method uses
  seed 20270201 + r.

## 5. Held-out test

Because the DPF configuration was chosen on S1–S4, results on S1–S4 may favour
DPF-PSO.  S5 and S6 were generated after the configuration was frozen
(procedural terrain and threats, seeds 5150 and 6160) and were not used for any
design decision.  The paper should report S1–S4 and S5–S6 separately.

## 6. After the main run (results/main)

* Main run finished with the frozen configuration (30 runs, 12 instances,
  18 configurations).  Analysis: `python src/analyze.py main`.
* Observed: DPF-PSO ~4–5x slower per run than the baselines (turn terms of
  the DP are not charged as FEs).  Two things were added *without changing
  the method*:
  1. Exact pruning of DP predecessors (0 <= T <= 360·b4).  Verified
     bit-identical to the dense DP in 324 random cases
     (`src/check_pruning.py`).  The whole experiment was re-run as
     `results/main_v2` to measure CPU times with the faster code; solution
     quality must be identical to `results/main` (checked, see §7).
  2. SPSO with 100 and 500 particles (authors' setting is 500) as a
     sensitivity check of the strongest path-planning baseline.
* Planned equal-CPU-time comparison (`src/run_eqtime.py`): each baseline gets
  its FE budget multiplied by T(DPF-PSO)/T(baseline) measured in main_v2.
* Ablation finding to report honestly: the success-based sigma rule does not
  help (a fixed schedule is statistically equivalent on 11/12 instances).  The
  method was NOT changed after seeing this, because S5–S6 were already used.
* Independent safety re-check of all final paths (`src/verify_paths.py`):
  dense 1 m ground-clearance sampling and analytic threat distances.
* `results/main_v2` (pruned DP) is bit-identical to `results/main` in all 216
  (instance, algorithm) groups: best costs, paths and DP FE counts.
* CPU times measured in the 20-process pool are inflated unevenly (the DP is
  memory-bound and slows down more under contention: DPF-PSO/PSO ratio 2.5–2.7
  in the pool versus 1.8–1.9 in a quiet run).  Therefore Table V uses a
  separate controlled benchmark (`src/timing_benchmark.py`: 3 runs per method
  and instance, methods interleaved, 6 processes) and the equal-time budgets
  are derived from it (`results/eqtime`).  As a stress test, the equal-time
  comparison was repeated with the larger pool-based multipliers
  (`results/eqtime_pool`).  Both are reported in the paper.

## 7. Known limitations

* DP wall-clock cost: the K³ turn tensor makes DPF-PSO slower per FE than the
  baselines (reported in the runtime table).
* Slot encoding assumes progress along S–G (no backtracking paths).
* Static, known environment; single UAV.

# Second revision (review of 2026-11-05)

## 8. How the review was used

The review (DPF-PSO_ICAISI2027_修改建议.md) and the follow-up remarks were
treated as opinions.  Decisions, in order:

* **Safety is P0 (agree).**  A path reported as feasible must satisfy the
  20 m ground clearance exactly.  Sampling-based checks cannot guarantee
  this, so the clearance is now computed exactly inside the cost function
  (`env.Terrain.ground_violation`): bilinear terrain on a straight segment is
  a quadratic per grid cell, so the minimum per cell has a closed form;
  pieces whose conservative bound already clears the threshold are skipped.
  Verified against an independent implementation (`src/clearance_ref.py`) on
  7,200 random segments (max difference 5e-13 m), never above a 10,000-point
  dense sampling (`src/check_clearance.py`).  Because the cost model changed,
  every experiment was re-run; first-revision results are in
  `results/archive_v1` and are not used.
* **Degenerate segments (found while re-running, not in the review).**  The
  first implementation set the turning angle to zero when a segment had zero
  horizontal length.  SPSO produces such segments (r clipped to its lower
  bound 0), which hid turns behind collapsed waypoints.  Turns are now
  measured between consecutive non-degenerate segments, as in the reference
  code of [5].  Slot-encoded paths cannot contain such segments, so DPF is
  unaffected and the DP cost still equals the full evaluation (sanity check).
* **Statistics (agree, with one correction).**  The review proposed a
  Friedman post-hoc test and a signed-rank test over the 12 instances.  The
  12 instances come from 6 scenarios, so they are not independent; with 6
  scenarios as units, the smallest two-sided exact signed-rank p is 0.031,
  which cannot survive a Holm correction over 7 baselines.  Confirmatory
  cross-scenario claims are therefore made only on 40 new held-out scenarios
  (scenario = unit, two-sided signed-rank test on scenario medians, Holm over
  all baseline x n pairs).  Within-instance rank-sum tests are Holm-corrected
  within each table family; A12 effect sizes and medians are reported.
* **Relative gaps (agree).**  One definition everywhere:
  Delta(B, P) = (median_B - median_P) / median_B, positive if the proposed
  configuration P is better.
* **Fairness (agree).**  Equal tuning budget for DE, L-SHADE, SPSO, IDP and
  DPF-PSO on S1-S4 only (`src/tune.py`, 25 configurations x 8 instances x 5
  seeds each); DE population sensitivity (NP = 5D, 10D); SPSO with 100/500
  particles; CMA-ES (pycma, IPOP); HHO added as a DPF host.
* **Fusion schedule (agree).**  Measured in host FEs instead of iterations.
  For 50-member hosts this is identical to the old schedule; for L-SHADE the
  old iteration schedule meant that, apart from the initial fusion, global
  fusion practically never ran (population of 18D members).
* **Structure-exploiting baselines (agree).**  Two were added: iterative DP
  (IDP: exact DP over random candidates in a geometrically shrinking region,
  Luus-style) and a gray-box local search (GLS: one waypoint moved at a time,
  incremental evaluation charged 2/(n+1) FE per proposal).  Their results are
  reported whatever they show.
* **Generalisation (agree).**  40 held-out scenarios H01-H40 from a new seed
  family (90210 + k), generated after all design decisions; the public
  scenario of [5] (Christmas Island DEM, `data/get_dem.py`); n = 30; a 5x
  budget; b2 = 10.
* **Timing (agree).**  Benchmark pinned to one logical processor per P-core
  (8 processes, `src/cpuinfo_win.py`), 5 runs per method and instance, methods
  interleaved; equal-time multipliers derived from it.
* **Disagreements.**  (i) "Minor revision, Weak Accept to Accept" was not
  adopted as an assessment; the revision is substantial.  (ii) The review
  stated that the conference website gives no GenAI policy; it does
  (icaisi.net/UsageGuidelines), and the paper follows it.  (iii) irace was not
  used (R not installed); the tuning protocol is a documented random search
  with racing-free equal budgets.  (iv) The midpoint candidate and other
  first-revision options were not re-introduced.

## 9. Seeds

pilot runs: 20270201 + r; tuning: 30270000 + r; final experiments:
20271101 + r; timing benchmark: 40270000 + r; held-out scenarios: 90210 + k.

## 10. Changes during the final runs (2026-11-05 afternoon)

* **CMA-ES restart thresholds (baseline fix).**  The first CMA-ES results on
  S1–S6 (main step) were far behind all other methods.  Cause: with pycma's
  default tolerances (tolfun 1e-11, tolx 1e-12) the first run never stopped
  within 3000n FEs, so the IPOP restarts never happened.  A check on S1_n10,
  S3_n10 and S3_n20 with four tuning seeds (`notes/check_cmaes2.py`, output
  `notes/cmaes_restart_check.json`) compared scaled thresholds; the setting
  tolx = 1e-3 (normalised coordinates), tolfun = tolfunhist = 1 (cost units),
  tolstagnation = 200 iterations was adopted for all experiments.  The change
  makes the baseline stronger; it was decided before any held-out CMA-ES
  result existed.  Old CMA-ES files: `results/archive_cmaes_default_tol/`
  (not used).
* **Pipeline interruption.**  At about 15:57 the IDE terminal that ran
  `run_all.ps1` was closed and all worker processes were killed
  ("window-CLOSE event"; logs in `results/*_part1.txt`).  Completed
  (instance, algorithm) groups were kept (they are saved only when all runs
  of the group are finished), and the pipeline was restarted detached from
  the IDE with `--skip-existing`.  Because every run is seeded with
  20271101 + r, resumed groups are identical to uninterrupted ones.
* **Timing benchmark re-run** with the final CMA-ES on the idle machine; the
  equal-time multipliers used by the eqtime step come from this run.  The
  first timing run is in `results/archive_timing_cmaes_default_tol/`.
* **Pruning benchmark re-measured** on the idle machine with the exact
  clearance in the edge terms: the pruned DP is bit-identical to the dense DP
  in 324 cases and 1.7x faster for n = 20, K = 100 (first study: 2.4x; the
  edge terms, which pruning does not touch, are now more expensive).

## 11. After the held-out results (no method changes)

* Primary analysis (H01–H40, 10 runs per scenario, signed-rank test on 40
  scenario medians, Holm over 20 tests): DPF-PSO significantly better than
  every black-box baseline except SPSO for n = 10 (not significant); against
  IDP significantly better for n = 10 and not different for n = 20; GLS
  clearly worse.  Tuned family (Holm over 8 tests): tuned DPF-PSO better than
  tuned DE, L-SHADE and IDP for both n; tuned SPSO significantly better for
  n = 10 and not different for n = 20.
* Explanation checked on the stored paths (not a new experiment): SPSO's
  spherical encoding can merge waypoints (r = 0).  The altitude band of the
  cost model applies only at waypoints, so merged paths need fewer climbs and
  descents and use long straight segments.  On H01–H40 with n = 10 the
  number of merged waypoints of tuned SPSO correlates with its advantage
  (Spearman rho = -0.70).  Reported as a limitation of the slot encoding; the
  slot encoding and DPF-PSO were not changed.
* The paper's narrative was changed to match the data: the improvement over
  black-box metaheuristics comes mainly from the dynamic program (DPF-PSO and
  IDP are within about 1 %), the swarm adds a small gain, and the encoding
  matters (tuned SPSO).

## 12. Safety check bookkeeping

* `verify_paths.py` assigned rows to experiments by file name only, so a call
  with several tags mixed experiments that share instance and algorithm
  names (eqtime and b2 showed 4,800 and 4,920 paths).  Fixed: every row now
  carries its tag; all tags were re-checked.
* The check now separates the two kinds of disagreement.  "Reported feasible
  but infeasible" (the dangerous kind) was 0 in every experiment.  One path
  (SPSO, S6_n10, run 18 of the equal-time experiment) was rejected by the
  cost function because its last segment had an exact clearance of
  20 m - 2e-14 m (rounding), while the check accepts clearances within
  1e-6 m of the limit.  The cost function is therefore slightly stricter than
  the check, never looser.

## 13. Third revision (review of 2026-11-06), no change of DPF-PSO

The second-round review was again treated as a set of opinions.  DPF-PSO,
its default and tuned configurations, the scenarios and the primary analysis
were not changed, so H01–H40 remain held out for the method.  Additions:

* **CMA-ES tuning** with the same protocol as the other tuned methods
  (`src/tune.py --algos CMAES`).  The default configuration had the best
  mean rank, so tuned CMA-ES is identical to CMA-ES (verified bit-identical
  on 8 instances, `results/check_cmaest_identity/`); the analysis reuses the
  CMA-ES runs for it.  The tuned family of the primary analysis now has 10
  instead of 8 tests (stricter Holm correction).
* **Height above ground along the final paths** (existing data).  Tuned SPSO
  spends a much larger share of its track outside the altitude band than
  the slot-encoded methods; on H35 (n = 10) its median run merges all ten
  waypoints into one straight segment up to about 380 m above the ground.
  This supports the reviewer's hypothesis that the band, checked only at the
  waypoints, favours encodings that merge waypoints.
* **Altitude-band variant** (sensitivity analysis, `CostParams.band_along`):
  the band [100, 200] m is enforced exactly along every segment (lower bound
  as a clearance of 100 m, upper bound with the clearance machinery on the
  mirrored terrain -Z; `src/check_ceiling.py`).  Violations below 1e-6 m are
  ignored in this variant because waypoints lie exactly on the band limits,
  where rounding would otherwise decide about a penalty of M0 (found by the
  GLS incremental-evaluation check in `src/sanity_checks.py --band`).
* **Held-out ablation** (PSO, no global fusion, greedy, fusion only) and
  **held-out equal-CPU-time comparison** for IDP, SPSO and DE.
* **Anytime analysis** from the stored best-so-far curves.
* Another job of the user (FA-BPSO, 10 processes) shared the machine during
  these runs, so they use 12 workers; no CPU times from these runs are used.

## 14. Third-revision results and bookkeeping (2026-11-06)

* `run_rev3.ps1` (12 workers, FA-BPSO job of the user running in parallel):
  altitude-band variant 00:52–02:52 (4,800 runs), held-out ablation
  02:52–03:18 (2,400 runs), held-out equal CPU time from 03:18.
* **Held-out ablation**: all four variants (PSO, no global fusion, greedy,
  DPF-only) significantly worse than DPF-PSO for both n.  DPF-only was worse
  by a median of 1.9 % (n = 10) and 1.0 % (n = 20): the swarm adds a small
  but significant gain.  The paper sentence is generated from the signs
  (`paper/build2.py`, `prose()`).
* **Altitude-band variant**: tuned DPF-PSO significantly better than tuned
  SPSO for both n (+2.8 %, +4.6 %), i.e. the advantage of tuned SPSO in the
  default model is reversed.  With defaults, DPF-PSO is significantly better
  than SPSO (both n), better than IDP for n = 10 and not different for
  n = 20, and **not significantly different from DE for n = 10** (median
  gap -0.3 %, p_H = 0.13) but better for n = 20.  The paper reports all of
  this; the variant is labelled post hoc / exploratory.
* **Safety check of the band variant, tolerance fix.**  The first check used
  the same tolerance as the cost function (1e-6 m).  29 SPSO paths that the
  cost function accepted lie exactly on that tolerance (clearance
  99.999999000000 m or maximum height 200.000001000000 m) and were flagged as
  "reported feasible but infeasible" because the two implementations round
  differently (the check computed violations up to 1e-6 + 2.8e-13 m).  An
  identical threshold cannot be checked by an independent floating-point
  implementation, so the check now accepts BAND_TOL + 1e-9 m and records
  the largest band violation of the accepted paths and how many lie beyond
  1e-6 m (`max_band_violation_reported_feasible`, `n_at_band_tolerance`).
  Both numbers and the first result are reported in supplement S10.  With
  the fix: 4,800 paths, 0 disagreements, one infeasible DE run (n = 20)
  that the cost function also rejected.  The default-model checks are
  unchanged (tolerance 1e-6 m, cost function without tolerance).
* **Held-out equal CPU time** (03:18–03:48; IDP x1.6, SPSO x1.7, DE x1.8 of
  the FE budget, DPF-PSO with its normal budget): DPF-PSO significantly
  better than DE for both n and than IDP for n = 10 (+0.6 %); not
  significantly different from IDP for n = 20 (-0.2 %) and from SPSO for
  either n (+0.1 %, +4.4 %).  At equal CPU time the advantage over SPSO,
  which was significant for n = 20 at equal FEs, is no longer significant;
  the paper states this (generated sentence in IV-E).
* Independent check of the new held-out runs: heldout (now 15,200 paths
  with the ablation variants) and heldout_eqtime (2,400): 0 disagreements.
  All 42,080 final paths: 0 reported feasible but infeasible; the 2
  conservative rounding disagreements of the second revision remain.
* Layout: the third-revision paper fits 6 pages after moving Fig. 3 behind
  IV-E's text, compacting Table III/IV labels and shrinking Fig. 1.
* The paper text of the third revision no longer contains hand-written
  result sentences for the new experiments; `prose()` produces them and
  raises for patterns without prepared wording.

## 15. Journal version: confirmatory test of the band finding on H41–H80 (plan written 2026-11-17, before any run)

* Purpose.  The altitude-band finding of the third revision (section 14)
  rests on analyses designed after the H01–H40 results were known.  Before
  the journal version it is tested on 40 fresh scenarios H41–H80
  (`env.heldout_scenario(k)`, seeds 90210 + k, k = 41..80, never used
  before).  DPF-PSO, its default and tuned configurations, the budget and
  both cost models are unchanged.
* Runs (`run_rev4.ps1`, 16 workers, resumable): SPSO, SPSO-T, DE, IDP,
  DPF-PSO, DPF-PSO-T; n = 10 and 20; 10 runs each (seeds 20271101 + r);
  (a) original cost model -> `results/heldout2/`; (b) band enforced along
  every segment (`band_along: true`) -> `results/heldout2_band/`.
  6 x 2 x 80 x 10 = 9,600 runs.
* Pre-specified analysis: per-scenario medians of the 10 runs, two-sided
  Wilcoxon signed-rank test over the 40 scenarios, alpha = 0.05, Holm
  correction within family, families as in Table III:
  - Family T (tuned, 4 tests): SPSO-T vs DPF-PSO-T for n = 10, 20 under the
    original model and under the band model.  Prediction from H01–H40:
    original model, SPSO-T better for n = 10 and no difference for n = 20;
    band model, DPF-PSO-T better for both n.  The finding counts as
    confirmed only if DPF-PSO-T is significantly better than SPSO-T under
    the band model for both n and the original-model result has the same
    direction as on H01–H40 (the band reverses the sign).
  - Family D (defaults under the band model, 6 tests): DPF-PSO vs SPSO, DE,
    IDP for n = 10, 20.  Prediction: better than SPSO (both n); not
    different from DE for n = 10, better for n = 20; better than IDP for
    n = 10, not different for n = 20.  Reported against the predictions.
  - Share of the horizontal track outside the band (agl_stats) for SPSO-T
    and DPF-PSO-T under the original model, expected to replicate
    (SPSO-T much larger).
* Reporting: everything in this section is reported whatever the outcome.
  H01–H40 tables stay unchanged; H41–H80 get their own table.  Abstract and
  contribution wording are chosen after the result (confirmed -> "confirmed
  on 40 further scenarios"; otherwise -> "exploratory, not confirmed").
* Independent safety check after the runs: `verify_paths.py heldout2
  heldout2_band`.
* Machine shared with the author's daily use (16 of 24 threads); no CPU
  times from these runs are used.
* Addendum (2026-11-17, written before any result of these runs was looked
  at).  Also reported from the same runs, descriptively and with their own
  Holm correction: Family R (defaults under the original model, 6 tests):
  DPF-PSO vs SPSO, DE, IDP for n = 10, 20 on H41–H80 (replication of the
  corresponding rows of Table II).  From existing and new final paths, with
  no test: per-component cost shares (`comps` stored in every result file)
  and the maximum turning angle and maximum climb-angle change of each final
  path (median over runs per method).

## 16. Journal version: additional structural baseline and CPU-time convergence (plan written 2026-11-17, before any run)

* **GridDP** (`algorithms.grid_dp`): the "fixed dense lattice, one exact DP
  with the whole budget" baseline asked for in the simulated review.  Per
  slot, a regular lattice of candidate positions: n_h = 3 heights (band
  limits and centre), n_tau = 3 along-track offsets (bounds and 0) and n_u
  equally spaced lateral offsets including the slot bounds, with the largest
  lattice size K = n_u n_h n_tau whose segment evaluations
  2K + (n-1)K^2 fit into the FE budget 3000n (n+1) segments (same accounting
  as DPF; turn terms not billed, as everywhere).  Single shot, no search,
  no randomness.  Compared with DPF-PSO on H01–H40 and H41–H80 under the
  original cost model (10 runs of DPF-PSO vs the deterministic GridDP path,
  scenario-level Wilcoxon signed-rank, Holm over the 4 (set, n) tests, one
  family "G").  Added after the main results were known; reported as such.
* **CPU-time convergence** (`src/timing_curves.py`, tag `timing_curves`):
  same pinned P-core protocol as the timing benchmark, 8 held-out scenarios
  H01–H08 x n = 10, 20, methods SPSO, DE, L-SHADE, CMA-ES, IDP, DPF-PSO,
  5 runs (seeds 50270000 + r), process CPU time recorded at the 101 FE
  checkpoints.  Statistic: for each scenario, the CPU time at which the
  median best-so-far curve of DPF-PSO first reaches the final median cost of
  a method, divided by that method's median total CPU time (median over the
  8 scenarios; "> 1" if never).  This replaces the FE-based wording of the
  speed claim where the CPU-time version does not support it.

## 17. Results of sections 15 and 16 (2026-11-17, runs 06:54–09:18 and 09:19–09:26)

* **Pre-registered test on H41–H80 (section 15): confirmed.**  Family T:
  original model, SPSO-T better for n = 10 (Δ̃ = −1.8 %, p_H = 0.021) and
  no difference for n = 20 (+1.2 %, p_H = 0.17), the same directions as on
  H01–H40; band model, DPF-PSO-T better for both n (+2.5 %, p_H = 2e-7;
  +4.5 %, p_H = 1e-10).  Family D (band model, defaults): DPF-PSO better
  than SPSO (both n), not different from DE for n = 10 (−0.2 %, p_H = 0.066)
  and better for n = 20 (+6.0 %), better than IDP for n = 10 (+0.5 %) and
  not different for n = 20 (−0.0 %): every prediction of section 15 held.
  Family R (original model, defaults): SPSO not different for n = 10,
  better for n = 20; DE not different for n = 10, +11.9 % for n = 20; IDP
  +0.8 % / not different.  Track outside the band, n = 10, original model:
  SPSO-T 23.9 %, DPF-PSO-T 6.3 %.  Safety check: heldout2 4,800 paths, 0
  disagreements; heldout2_band 4,800 paths, 4,799 feasible (one DE run),
  0 disagreements.  `summary["heldout2_verdict"]["band_confirmed"] = true`
  by the criterion as written; the paper wording follows the "confirmed"
  branch of `paper/build_journal.py`.
* **GridDP (section 16).**  H01–H40: DPF-PSO better for both n (+1.8 %,
  p_H = 3e-9, wins 95 %; +1.6 %, p_H = 0.001, wins 75 %).  H41–H80: +1.7 %
  (p_H = 3e-5) and +2.0 % (p_H = 1e-5).  All GridDP paths feasible.  GridDP
  itself beat DE for n = 20 on H01–H40 (+8.7 %) and was on a par with DE and
  SPSO for n = 10 (checked descriptively before the family-G analysis).
* **CPU-time convergence (section 16), H01–H08, 5 runs, pinned P-cores.**
  Ratio of the CPU time at which DPF-PSO's median curve reaches a method's
  final median to that method's CPU time (median over 8 scenarios):
  n = 20: CMA-ES 0.14, L-SHADE 0.19, DE 0.49, SPSO 0.64, IDP not reached on
  the median scenario (reached on 4/8); n = 10: CMA-ES 0.13, L-SHADE 0.64,
  SPSO 1.02, IDP 1.41, DE not reached on the median scenario (4/8).  The
  FE-based speed claim therefore survives in CPU time for n = 20 against the
  black-box methods and does not hold for n = 10 against SPSO and DE.

## 18. Journal version, second round of additions (plan written 2026-11-17 10:05, before any run; after two external reviews of the drafts)

No change of DPF-PSO, of its configurations, of the cost models or of
the primary analysis.  All additions are post hoc with respect to H01–H40
and are labelled as such; predictions are written here before the runs.

* **18a. Equal CPU time for all baselines on H01–H40** (`heldout_eqtime`,
  so far SPSO, DE, IDP only).  Add PSO, GWO, WOA, HHO, L-SHADE with the
  budget multipliers of `results/timing/multipliers.json`; CMA-ES and GLS
  have multiplier 1.0 (they are slower than DPF-PSO and keep 3000n FEs), so
  their held-out runs are reused, as is L-SHADE for n = 20 (multiplier
  1.0).  Family: DPF-PSO against all 10 baselines, both n, Holm over the 20
  tests (replaces the 6-test family of Table 3).  Prediction: DPF-PSO
  remains significantly better than PSO, GWO, WOA, HHO, L-SHADE and CMA-ES
  for both n; SPSO and IDP as before.
* **18b. Held-out ablation of the remaining components.**  Two variants on
  H01–H40, 10 runs: `DPF-PSO_noPG` (initial global fusion kept, no periodic
  global fusion: g_every = 0, init_global = True) and `DPF-PSO_noA` (fixed
  σ, already defined).  Joined to the held-out ablation family (Holm over
  all its tests).  Prediction from the tuning result (the selected
  configuration has no periodic global fusion and a fixed step): neither
  variant is significantly worse than DPF-PSO.  If so, the paper says that
  these two components can be dropped.
* **18c. Plug-in on H01–H40.**  GWO+DPF, WOA+DPF, HHO+DPF, DE+DPF,
  LSHADE+DPF, 10 runs; each host with DPF against the same host without it
  (held-out runs), scenario-level signed-rank, Holm over the 10 (host, n)
  tests.  Prediction: every host improves significantly for both n.
* **18d. Weight sensitivity b4 = 10 on S1–S6** (tag `b4`, 10 runs, all
  methods of MAIN; mirrors the existing b2 = 10 experiment).  Descriptive
  (rank-sum per instance, Holm over methods).  Prediction: with a heavier
  smoothness term, the gap of DPF-PSO to the black-box methods grows and the
  gap to IDP stays within a few percent, because the three-point term is
  exactly what the dynamic program handles.  Reported whatever the outcome.
* **18e. Multi-start local optimizer baseline** (`MS-Powell`): scipy
  Powell with bounds from a uniform random start, restarted from a new
  random start when it converges (xtol 1e-3 of the ranges, ftol 1e-4) until
  the FE budget is spent; every evaluation charged through the evaluator.
  H01–H40, 10 runs, compared with DPF-PSO in a family of its own (2 tests).
  Represents gradient-free trajectory optimization; added on reviewer
  request.
* **18f. Encoding sensitivity**: DE and CMA-ES on the spherical encoding
  of SPSO (`DE-sph`, `CMAES-sph`), H01–H40, 10 runs, family of 4 tests
  against DPF-PSO; descriptive comparison with their slot-encoded versions.
  Tests whether the slot encoding handicaps the black-box baselines.  Done
  only if the generic optimizers can use the spherical bounds without
  change of their code.
* **18g. From existing data, no runs**: share of final paths with maximum
  turning angle ≤ 45° and ≤ 60° and maximum climb-angle change ≤ 30°
  (descriptive kinematic feasibility indicator), per method and n.
* **18h. Second real terrain (C2)** if a public DEM is obtained: all
  methods of MAIN, 30 runs, both n, as for C1.
* Runs 18a, 18c, 18d first (`run_rev5a.ps1`, 14 workers), 18b/18e/18f
  after their code exists (`run_rev5b.ps1`).  Independent safety check on
  every new tag before any number is used.
* **18f addendum (2026-11-17, after a single smoke run of DE-sph and
  CMAES-sph on H01/10 with a test seed, before any experiment run).**  The
  smoke run gave costs well below the DPF-PSO median on that instance.
  Because the spherical encoding can merge waypoints and the original model
  checks the altitude band at waypoints only (sections 13–15), the
  hypothesis is that any advantage of the spherically encoded black-box
  methods under the original model comes from that loophole.  Therefore
  18f runs DE-sph and CMAES-sph on H01–H40 under both cost models
  (`heldout` and `heldout_band`), 10 runs each.  Prediction: under the
  original model they are better than their slot-encoded versions and
  competitive with DPF-PSO for n = 10; under the band model the advantage
  disappears.  Families: original model, 4 tests vs DPF-PSO; band model,
  4 tests vs DPF-PSO; descriptive comparison with the slot versions.
  Reported whatever the outcome.

## 19. Results of section 18, part A (runs 15:01–17:21 on 2026-11-17)

* **18a, equal CPU time, all baselines (20 tests, Holm).**  DPF-PSO
  significantly better than PSO, GWO, WOA, HHO, DE, L-SHADE, CMA-ES and GLS
  for both n, than IDP for n = 10 (+0.6 %); not significantly different
  from IDP for n = 20 (−0.2 %) and from SPSO for either n (+0.1 %, +4.4 %).
  CMA-ES and GLS (multiplier 1.0) reuse their held-out runs.  Prediction
  held.  Safety check heldout_eqtime: 6,400 paths, 0 disagreements.
* **18c, plug-in on H01–H40 (10 tests, Holm).**  Every host improved
  significantly for both n: GWO +12.4 % / +23.6 %, WOA +16.5 % / +21.1 %,
  HHO +10.3 % / +17.7 %, DE +2.7 % / +8.6 %, L-SHADE +2.5 % / +9.7 %
  (n = 10 / 20).  Prediction held.
* **18d, b4 = 10 on S1–S6 (10 runs).**  DPF-PSO first in mean rank (1.83),
  lowest median on 5 of 12 instances, no method significantly better on any
  instance; median gap to the best black-box method 10.1 % against 4.7 %
  under the published weights.  Prediction held.  Safety check b4: 1,320
  paths, 0 disagreements.
* **18g (existing data).**  n = 20: 88 % of DPF-PSO paths without a turn
  sharper than 45° (other methods 9–87 %), 56 % without a climb-angle
  change above 30° (others 7–52 %).

## 20. Results of section 18, part B (runs 17:23–18:58 on 2026-11-17)

* **18b, ablation (joined to the held-out ablation family, Holm over 12
  tests).**  Without periodic global fusion: n = 10 not different
  (−0.1 %), n = 20 significantly worse (+2.6 %).  Fixed step size: n = 10
  not different (−0.1 %), n = 20 significantly *better* than DPF-PSO
  (−0.5 %, wins 92 % of scenarios).  The prediction (neither component
  matters) held for n = 10 and failed for n = 20: periodic global fusion
  contributes for n = 20, the step-size adaptation does not and is slightly
  harmful for n = 20.  DPF-PSO is not changed; the paper reports this.
* **18e, multi-start Powell (2 tests).**  DPF-PSO better for both n
  (+4.1 %, +19.2 %).
* **18f, spherical encoding.**  Against their slot-encoded versions, DE
  improved by 6.7 % / 16.1 % and CMA-ES by 16.6 % / 21.8 % (n = 10 / 20, all
  significant).  Original model, against DPF-PSO (4 tests): DE-sph and
  CMAES-sph significantly better for both n; DPF-PSO's median cost higher
  by 3.1 % / 4.8 % (DE) and 4.4 % / 7.9 % (CMA-ES).  Band model (4 tests):
  DPF-PSO better than DE-sph for n = 10 (+1.1 %), not different for n = 20
  (−1.9 %); not different from CMAES-sph for n = 10 (−0.8 %), CMAES-sph
  better for n = 20 (−3.0 %).  Against the addendum prediction: the gain
  over the slot versions held; "competitive with DPF-PSO for n = 10" was
  too weak (they were better for both n); "the advantage disappears under
  the band model" held except for CMA-ES with n = 20.  Interpretation: most
  of the advantage of the spherical encoding comes from the waypoint-only
  band check, the rest from merging waypoints within the band.  This is a
  limitation of the slot encoding, reported in the abstract, highlights,
  contributions, Section 6.3, discussion and conclusion.
* Safety check heldout (with all new variants) and heldout_band: 0
  disagreements; CMAES-sph under the band model 98.5 % / 99.7 % feasible.
